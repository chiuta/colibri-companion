//! LoRA adapters merged into the base weights while they are read.
//!
//! [`LoraSource`] wraps any weight file. Reading a tensor that an adapter targets returns the base
//! matrix in f32 with `W += scale·B·A` already added; every other tensor passes through untouched
//! (and keeps its quantised, memory-mapped form). A targeted matrix is therefore held as f32 instead
//! of quantised: that is the price of merging without a second rounding step.
//!
//! Row order. A llama GGUF keeps Q and K rows permuted and the loader undoes that *after* reading,
//! so a GGUF adapter made for the same file is merged in the same permuted space and needs nothing.
//! A llama GGUF adapter on a Hugging Face base is un-permuted first ([`prepare_adapter`]); a PEFT
//! adapter on a llama GGUF base would need the opposite shuffle, which is not implemented, so it is
//! refused with a clear message instead of silently producing a damaged model.
use companion_engine::arch::Spec;
use companion_engine::loader::{GgufFile, LoadError, Naming, RawTensor, TensorSource};
use companion_engine::lora::{Adapter, Key, Module};
use companion_engine::moe::{MoeError, RangeSource};
use std::path::Path;
use std::sync::Arc;

/// Which base file the adapters will be merged into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseKind {
    Hf,
    GgufLlama,
    GgufOther,
}

/// Load one adapter (a PEFT folder, or a `.gguf` adapter file) and make its row order match the base.
pub fn prepare_adapter(path: &Path, base: BaseKind, spec: &Spec) -> Result<Adapter, String> {
    let name = path.file_stem().or_else(|| path.file_name()).map_or_else(
        || "adapter".to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let is_gguf = path.is_file()
        && path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    if is_gguf {
        let f = GgufFile::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let llama_adapter = f.gguf.architecture() == Some("llama");
        let mut a =
            Adapter::from_gguf_file(&name, &f).map_err(|e| format!("{}: {e}", path.display()))?;
        if llama_adapter && base == BaseKind::Hf {
            a.unpermute_llama(spec.heads as usize, spec.kv_heads as usize)
                .map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Ok(a)
    } else {
        let dir = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or(Path::new("."))
        };
        if base == BaseKind::GgufLlama {
            return Err(format!(
                "{}: a Hugging Face (PEFT) adapter cannot be merged into a llama GGUF base yet (its Q/K rows are permuted); convert the adapter to GGUF or use the original safetensors model",
                path.display()
            ));
        }
        Adapter::from_peft_dir(&name, dir).map_err(|e| format!("{}: {e}", dir.display()))
    }
}

pub struct LoraSource {
    inner: Arc<dyn companion_engine::loader::WeightFile>,
    naming: Naming,
    adapters: Vec<(Arc<Adapter>, f32)>,
}

impl LoraSource {
    pub fn new(
        inner: Arc<dyn companion_engine::loader::WeightFile>,
        naming: Naming,
        adapters: Vec<(Arc<Adapter>, f32)>,
    ) -> LoraSource {
        LoraSource {
            inner,
            naming,
            adapters,
        }
    }

    fn key_of(&self, name: &str) -> Option<Key> {
        // routed experts have their own `down_proj`/`gate_proj` names; adapters never target those
        if name.contains("expert") || !name.ends_with(".weight") {
            return None;
        }
        let k = match self.naming {
            Naming::Hf => Key::from_hf(name),
            Naming::Gguf => Key::from_gguf(name),
        };
        (k.layer.is_some() && !matches!(k.module, Module::Other(_))).then_some(k)
    }

    /// Does any adapter change this tensor?
    pub fn targets(&self, name: &str) -> bool {
        self.key_of(name).is_some_and(|k| {
            self.adapters
                .iter()
                .any(|(a, s)| *s != 0.0 && a.pairs.contains_key(&k))
        })
    }
}

impl TensorSource for LoraSource {
    fn read(&self, name: &str) -> Result<Vec<f32>, LoadError> {
        let mut w = self.inner.read(name)?;
        let Some(key) = self.key_of(name) else {
            return Ok(w);
        };
        for (a, s) in &self.adapters {
            if *s == 0.0 {
                continue;
            }
            if let Some(p) = a.pairs.get(&key) {
                let (rows, cols) = if a.fan_in_fan_out {
                    (p.in_dim, p.out_dim)
                } else {
                    (p.out_dim, p.in_dim)
                };
                p.merge_dense(&mut w, rows, cols, *s, a.fan_in_fan_out)
                    .map_err(|e| {
                        LoadError::Format(format!(
                            "adapter `{}` does not fit `{name}`: {e}",
                            a.name
                        ))
                    })?;
            }
        }
        Ok(w)
    }
    fn has(&self, name: &str) -> bool {
        self.inner.has(name)
    }
    fn read_raw(&self, name: &str, mmap: bool) -> Result<Option<RawTensor>, LoadError> {
        if self.targets(name) {
            return Ok(None);
        }
        self.inner.read_raw(name, mmap)
    }
    fn infos(&self) -> Vec<companion_engine::gguf::TensorInfo> {
        self.inner.infos()
    }
}

impl RangeSource for LoraSource {
    fn read_range(&self, name: &str, start: usize, len: usize) -> Result<Vec<f32>, MoeError> {
        self.inner.read_range(name, start, len)
    }
    fn read_range_raw(
        &self,
        name: &str,
        start: usize,
        rows: usize,
        cols: usize,
        mmap: bool,
    ) -> Result<Option<RawTensor>, MoeError> {
        self.inner.read_range_raw(name, start, rows, cols, mmap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use companion_engine::lora::LoraPair;
    use std::collections::HashMap;

    struct Base;
    impl TensorSource for Base {
        fn read(&self, name: &str) -> Result<Vec<f32>, LoadError> {
            if name == "model.layers.0.self_attn.q_proj.weight" {
                Ok(vec![0.0; 6])
            } else {
                Ok(vec![1.0; 4])
            }
        }
        fn has(&self, _: &str) -> bool {
            true
        }
    }
    impl RangeSource for Base {
        fn read_range(&self, _: &str, _: usize, len: usize) -> Result<Vec<f32>, MoeError> {
            Ok(vec![0.0; len])
        }
    }

    fn adapter() -> Adapter {
        // 2×3 matrix, rank 1: B = [1,2]ᵀ, A = [1,0,1], alpha = r so scale 1 → delta rows [1,0,1] and [2,0,2]
        let p = LoraPair::new(vec![1.0, 0.0, 1.0], vec![1.0, 2.0], 1, 3, 2, 1.0, false).unwrap();
        let mut pairs = HashMap::new();
        pairs.insert(Key::new(0, Module::Q), p);
        Adapter {
            name: "t".into(),
            pairs,
            default_scale: 1.0,
            fan_in_fan_out: false,
        }
    }

    #[test]
    fn only_targeted_matrices_change() {
        let src = LoraSource::new(Arc::new(Base), Naming::Hf, vec![(Arc::new(adapter()), 1.0)]);
        assert_eq!(
            src.read("model.layers.0.self_attn.q_proj.weight").unwrap(),
            vec![1.0, 0.0, 1.0, 2.0, 0.0, 2.0]
        );
        assert_eq!(
            src.read("model.layers.0.self_attn.k_proj.weight").unwrap(),
            vec![1.0; 4]
        );
        assert!(src.targets("model.layers.0.self_attn.q_proj.weight"));
        assert!(!src.targets("model.layers.1.self_attn.q_proj.weight"));
        assert!(!src.targets("model.layers.0.mlp.experts.3.down_proj.weight"));
        let half = LoraSource::new(Arc::new(Base), Naming::Hf, vec![(Arc::new(adapter()), 0.5)]);
        assert_eq!(
            half.read("model.layers.0.self_attn.q_proj.weight").unwrap()[3],
            1.0
        );
        let off = LoraSource::new(Arc::new(Base), Naming::Hf, vec![(Arc::new(adapter()), 0.0)]);
        assert!(!off.targets("model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn a_wrong_shape_is_an_error_not_a_panic() {
        struct Wrong;
        impl TensorSource for Wrong {
            fn read(&self, _: &str) -> Result<Vec<f32>, LoadError> {
                Ok(vec![0.0; 8])
            }
            fn has(&self, _: &str) -> bool {
                true
            }
        }
        impl RangeSource for Wrong {
            fn read_range(&self, _: &str, _: usize, len: usize) -> Result<Vec<f32>, MoeError> {
                Ok(vec![0.0; len])
            }
        }
        let src = LoraSource::new(
            Arc::new(Wrong),
            Naming::Hf,
            vec![(Arc::new(adapter()), 1.0)],
        );
        assert!(
            src.read("model.layers.0.self_attn.q_proj.weight")
                .unwrap_err()
                .to_string()
                .contains("does not fit")
        );
    }
}
