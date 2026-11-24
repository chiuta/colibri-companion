//! Builds a [`Model`] from files: safetensors (single file or sharded directory with an index)
//! and GGUF. Every offset and size taken from a file is checked against the real file length
//! before anything is allocated, so a hostile or truncated file is an error, not a crash.
use crate::arch::Spec;
use crate::gguf::Gguf;
use crate::model::{Layer, Model, ModelError, Weights};
use crate::quant::{dequantize, DType};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum LoadError { Io(std::io::Error), Format(String), Missing(String), Model(ModelError) }
impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "{e}"),
            LoadError::Format(s) => write!(f, "{s}"),
            LoadError::Missing(n) => write!(f, "the model has no tensor `{n}`"),
            LoadError::Model(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for LoadError {}
impl From<std::io::Error> for LoadError { fn from(e: std::io::Error) -> Self { LoadError::Io(e) } }
impl From<ModelError> for LoadError { fn from(e: ModelError) -> Self { LoadError::Model(e) } }
fn fmt_err<T>(s: impl Into<String>) -> Result<T, LoadError> { Err(LoadError::Format(s.into())) }

/// Anything that can hand out a named tensor as f32.
pub trait TensorSource {
    fn read(&self, name: &str) -> Result<Vec<f32>, LoadError>;
    fn has(&self, name: &str) -> bool;
}

fn read_at(path: &Path, offset: u64, len: u64, file_len: u64) -> Result<Vec<u8>, LoadError> {
    if offset.checked_add(len).is_none_or(|e| e > file_len) { return fmt_err("a tensor points past the end of its file"); }
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut b = vec![0u8; len as usize];
    f.read_exact(&mut b)?;
    Ok(b)
}

// ---------------------------------------------------------------- safetensors

struct StEntry { file: usize, dtype: DType, elems: u64, start: u64, len: u64 }
pub struct SafeTensors { files: Vec<(PathBuf, u64)>, map: HashMap<String, StEntry> }

impl SafeTensors {
    /// A single `.safetensors` file, or a directory (uses `model.safetensors.index.json` when present).
    pub fn open(path: &Path) -> Result<SafeTensors, LoadError> {
        let mut files = vec![];
        if path.is_dir() {
            let idx = path.join("model.safetensors.index.json");
            if idx.exists() {
                let j: Value = serde_json::from_slice(&std::fs::read(&idx)?).map_err(|e| LoadError::Format(e.to_string()))?;
                let wm = j.get("weight_map").and_then(Value::as_object).ok_or_else(|| LoadError::Format("index has no weight_map".into()))?;
                let mut names: Vec<&str> = wm.values().filter_map(Value::as_str).collect();
                names.sort_unstable();
                names.dedup();
                for n in names {
                    if n.contains("..") || n.contains('/') || n.contains('\\') { return fmt_err("index names a file outside the model folder"); }
                    files.push(path.join(n));
                }
            } else {
                for e in std::fs::read_dir(path)? {
                    let p = e?.path();
                    if p.extension().is_some_and(|x| x == "safetensors") { files.push(p); }
                }
                files.sort();
            }
            if files.is_empty() { return fmt_err("no .safetensors files found"); }
        } else { files.push(path.to_path_buf()); }
        let mut map = HashMap::new();
        let mut infos = vec![];
        for (fi, p) in files.iter().enumerate() {
            let len = std::fs::metadata(p)?.len();
            let mut r = BufReader::new(File::open(p)?);
            let mut n = [0u8; 8];
            r.read_exact(&mut n).map_err(|_| LoadError::Format("file too short for a safetensors header".into()))?;
            let n = u64::from_le_bytes(n);
            if n > (1 << 27) || n.checked_add(8).is_none_or(|e| e > len) { return fmt_err("implausible safetensors header size"); }
            let mut h = vec![0u8; n as usize];
            r.read_exact(&mut h)?;
            let j: Value = serde_json::from_slice(&h).map_err(|e| LoadError::Format(format!("bad header: {e}")))?;
            let obj = j.as_object().ok_or_else(|| LoadError::Format("header is not an object".into()))?;
            let base = 8 + n;
            for (name, v) in obj {
                if name == "__metadata__" { continue; }
                let dt = match v.get("dtype").and_then(Value::as_str) {
                    Some("F32") => DType::F32, Some("F16") => DType::F16, Some("BF16") => DType::BF16,
                    other => return fmt_err(format!("tensor {name}: dtype {other:?} is not supported")),
                };
                let shape = v.get("shape").and_then(Value::as_array).ok_or_else(|| LoadError::Format(format!("{name}: no shape")))?;
                let mut elems = 1u64;
                for d in shape {
                    elems = d.as_u64().and_then(|d| elems.checked_mul(d)).ok_or_else(|| LoadError::Format(format!("{name}: bad shape")))?;
                }
                let off = v.get("data_offsets").and_then(Value::as_array).filter(|a| a.len() == 2)
                    .and_then(|a| Some((a[0].as_u64()?, a[1].as_u64()?))).ok_or_else(|| LoadError::Format(format!("{name}: bad data_offsets")))?;
                if off.1 < off.0 || Some(off.1 - off.0) != dt.bytes_for(elems) { return fmt_err(format!("{name}: size does not match shape")); }
                if base.checked_add(off.1).is_none_or(|e| e > len) { return fmt_err(format!("{name}: data past end of file")); }
                map.insert(name.clone(), StEntry { file: fi, dtype: dt, elems, start: base + off.0, len: off.1 - off.0 });
            }
            infos.push(len);
        }
        Ok(SafeTensors { files: files.into_iter().zip(infos).collect(), map })
    }
    pub fn names(&self) -> impl Iterator<Item = &str> { self.map.keys().map(String::as_str) }
}
impl TensorSource for SafeTensors {
    fn has(&self, name: &str) -> bool { self.map.contains_key(name) }
    fn read(&self, name: &str) -> Result<Vec<f32>, LoadError> {
        let e = self.map.get(name).ok_or_else(|| LoadError::Missing(name.into()))?;
        let (p, flen) = &self.files[e.file];
        let b = read_at(p, e.start, e.len, *flen)?;
        let mut out = vec![];
        dequantize(e.dtype, &b, e.elems as usize, &mut out).map_err(|e| LoadError::Format(e.to_string()))?;
        Ok(out)
    }
}

// ----------------------------------------------------------------------- GGUF

pub struct GgufFile { path: PathBuf, pub gguf: Gguf, len: u64 }
impl GgufFile {
    pub fn open(path: &Path) -> Result<GgufFile, LoadError> {
        let len = std::fs::metadata(path)?.len();
        let gguf = Gguf::parse(BufReader::new(File::open(path)?)).map_err(|e| LoadError::Format(e.to_string()))?;
        Ok(GgufFile { path: path.to_path_buf(), gguf, len })
    }
}
impl TensorSource for GgufFile {
    fn has(&self, name: &str) -> bool { self.gguf.tensors.iter().any(|t| t.name == name) }
    fn read(&self, name: &str) -> Result<Vec<f32>, LoadError> {
        let t = self.gguf.tensors.iter().find(|t| t.name == name).ok_or_else(|| LoadError::Missing(name.into()))?;
        let (n, bytes) = match (t.elements(), t.bytes()) { (Some(n), Some(b)) => (n, b), _ => return fmt_err(format!("{name}: bad size")) };
        let off = self.gguf.data_start.checked_add(t.offset).ok_or_else(|| LoadError::Format("offset overflow".into()))?;
        let b = read_at(&self.path, off, bytes, self.len)?;
        let mut out = vec![];
        dequantize(t.dtype, &b, n as usize, &mut out).map_err(|e| LoadError::Format(format!("{name}: {e}")))?;
        Ok(out)
    }
}

// --------------------------------------------------------------------- naming

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Naming { Hf, Gguf }

struct Names { embed: &'static str, final_norm: &'static str, head: &'static str }
fn names(n: Naming) -> Names {
    match n {
        Naming::Hf => Names { embed: "model.embed_tokens.weight", final_norm: "model.norm.weight", head: "lm_head.weight" },
        Naming::Gguf => Names { embed: "token_embd.weight", final_norm: "output_norm.weight", head: "output.weight" },
    }
}
fn layer_name(n: Naming, i: usize, part: &str) -> String {
    match n {
        Naming::Hf => match part {
            "attn_norm" => format!("model.layers.{i}.input_layernorm.weight"),
            "wq" => format!("model.layers.{i}.self_attn.q_proj.weight"),
            "wk" => format!("model.layers.{i}.self_attn.k_proj.weight"),
            "wv" => format!("model.layers.{i}.self_attn.v_proj.weight"),
            "wo" => format!("model.layers.{i}.self_attn.o_proj.weight"),
            "ffn_norm" => format!("model.layers.{i}.post_attention_layernorm.weight"),
            "w_gate" => format!("model.layers.{i}.mlp.gate_proj.weight"),
            "w_up" => format!("model.layers.{i}.mlp.up_proj.weight"),
            _ => format!("model.layers.{i}.mlp.down_proj.weight"),
        },
        Naming::Gguf => {
            let p = match part { "attn_norm" => "attn_norm", "wq" => "attn_q", "wk" => "attn_k", "wv" => "attn_v", "wo" => "attn_output",
                "ffn_norm" => "ffn_norm", "w_gate" => "ffn_gate", "w_up" => "ffn_up", _ => "ffn_down" };
            format!("blk.{i}.{p}.weight")
        }
    }
}

/// llama.cpp stores Llama's Q/K rows permuted for interleaved RoPE; undo it to our rotate-half layout.
pub fn unpermute_rows(w: &[f32], rows: usize, cols: usize, n_head: usize) -> Vec<f32> {
    let hd = rows / n_head;
    let half = hd / 2;
    let mut out = vec![0.0; w.len()];
    for h in 0..n_head {
        for t in 0..2 {
            for j in 0..half {
                let (dst, src) = (h * hd + t * half + j, h * hd + j * 2 + t);
                out[dst * cols..(dst + 1) * cols].copy_from_slice(&w[src * cols..(src + 1) * cols]);
            }
        }
    }
    out
}

/// Load a dense model. `llama_permuted` must be true for GGUF files whose architecture is `llama`.
pub fn load_dense(spec: Spec, src: &dyn TensorSource, naming: Naming, llama_permuted: bool) -> Result<Model, LoadError> {
    if spec.experts > 1 { return fmt_err("mixture-of-experts models are not loadable yet"); }
    let n = names(naming);
    let (h, hd) = (spec.hidden as usize, spec.head_dim as usize);
    let fix = |w: Vec<f32>, heads: u64| if llama_permuted && naming == Naming::Gguf {
        let rows = heads as usize * hd;
        if w.len() == rows * h && rows % (2 * heads as usize) == 0 { unpermute_rows(&w, rows, h, heads as usize) } else { w }
    } else { w };
    let mut layers = Vec::with_capacity(spec.layers as usize);
    for i in 0..spec.layers as usize {
        let g = |p: &str| src.read(&layer_name(naming, i, p));
        layers.push(Layer {
            attn_norm: g("attn_norm")?, wq: fix(g("wq")?, spec.heads), wk: fix(g("wk")?, spec.kv_heads), wv: g("wv")?, wo: g("wo")?,
            ffn_norm: g("ffn_norm")?, w_gate: g("w_gate")?, w_up: g("w_up")?, w_down: g("w_down")?,
        });
    }
    let embed = src.read(n.embed)?;
    let lm_head = if src.has(n.head) && !spec.tied_embeddings { Some(src.read(n.head)?) } else { None };
    let w = Weights { embed, layers, final_norm: src.read(n.final_norm)?, lm_head };
    Ok(Model::new(spec, w)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::tiny;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn tmpdir() -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!("engine-loader-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    /// (hf name, gguf name, shape, data)
    fn all(m: &Model) -> Vec<(String, String, Vec<usize>, Vec<f32>)> {
        let s = &m.spec;
        let (h, qd, kd, ff, v) = (s.hidden as usize, (s.heads * s.head_dim) as usize, (s.kv_heads * s.head_dim) as usize, s.ffn as usize, s.vocab as usize);
        let mut o = vec![("model.embed_tokens.weight".into(), "token_embd.weight".into(), vec![v, h], m.w.embed.clone()),
            ("model.norm.weight".into(), "output_norm.weight".into(), vec![h], m.w.final_norm.clone())];
        if let Some(l) = &m.w.lm_head { o.push(("lm_head.weight".into(), "output.weight".into(), vec![v, h], l.clone())); }
        for (i, l) in m.w.layers.iter().enumerate() {
            for (p, shape, d) in [("attn_norm", vec![h], &l.attn_norm), ("wq", vec![qd, h], &l.wq), ("wk", vec![kd, h], &l.wk), ("wv", vec![kd, h], &l.wv),
                ("wo", vec![h, qd], &l.wo), ("ffn_norm", vec![h], &l.ffn_norm), ("w_gate", vec![ff, h], &l.w_gate), ("w_up", vec![ff, h], &l.w_up), ("w_down", vec![h, ff], &l.w_down)] {
                o.push((layer_name(Naming::Hf, i, p), layer_name(Naming::Gguf, i, p), shape, d.clone()));
            }
        }
        o
    }
    fn write_st(path: &Path, ts: &[(String, Vec<usize>, Vec<u8>, &str)]) {
        let mut hdr = serde_json::Map::new();
        let mut off = 0;
        for (n, shape, bytes, dt) in ts {
            hdr.insert(n.clone(), serde_json::json!({"dtype": dt, "shape": shape, "data_offsets": [off, off + bytes.len()]}));
            off += bytes.len();
        }
        let h = serde_json::to_vec(&hdr).unwrap();
        let mut f = (h.len() as u64).to_le_bytes().to_vec();
        f.extend(h);
        for t in ts { f.extend(&t.2); }
        std::fs::write(path, f).unwrap();
    }
    fn f32b(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    fn bf16b(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect() }
    fn run(m: &Model) -> Vec<f32> { let mut st = m.new_state(); m.forward(&mut st, 3).unwrap(); m.forward(&mut st, 11).unwrap() }

    #[test]
    fn sharded_safetensors_with_index_reproduces_the_model() {
        let m = tiny(4, 2, false);
        let d = tmpdir();
        let t = all(&m);
        let (a, b) = t.split_at(t.len() / 2);
        let mk = |s: &[(String, String, Vec<usize>, Vec<f32>)]| s.iter().map(|(n, _, sh, d)| (n.clone(), sh.clone(), f32b(d), "F32")).collect::<Vec<_>>();
        write_st(&d.join("model-00001-of-00002.safetensors"), &mk(a));
        write_st(&d.join("model-00002-of-00002.safetensors"), &mk(b));
        let wm: serde_json::Map<String, Value> = a.iter().map(|x| (x.0.clone(), "model-00001-of-00002.safetensors".into()))
            .chain(b.iter().map(|x| (x.0.clone(), "model-00002-of-00002.safetensors".into()))).collect();
        std::fs::write(d.join("model.safetensors.index.json"), serde_json::json!({"weight_map": wm}).to_string()).unwrap();
        let src = SafeTensors::open(&d).unwrap();
        let loaded = load_dense(m.spec.clone(), &src, Naming::Hf, false).unwrap();
        assert_eq!(run(&m), run(&loaded));
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn bf16_files_load_with_bf16_precision() {
        let m = tiny(4, 4, true);
        let d = tmpdir();
        let ts: Vec<_> = all(&m).into_iter().map(|(n, _, sh, d)| (n, sh, bf16b(&d), "BF16")).collect();
        write_st(&d.join("m.safetensors"), &ts);
        let loaded = load_dense(m.spec.clone(), &SafeTensors::open(&d.join("m.safetensors")).unwrap(), Naming::Hf, false).unwrap();
        let (a, b) = (run(&m), run(&loaded));
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        assert!(worst < 0.05, "{worst}");
        assert!(loaded.w.lm_head.is_none());
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn broken_safetensors_are_errors() {
        let d = tmpdir();
        let p = d.join("x.safetensors");
        std::fs::write(&p, [1u8, 2, 3]).unwrap();
        assert!(SafeTensors::open(&p).is_err());
        std::fs::write(&p, u64::MAX.to_le_bytes()).unwrap();
        assert!(SafeTensors::open(&p).is_err());
        // a tensor that claims more bytes than the file holds
        let mut h = serde_json::to_vec(&serde_json::json!({"a": {"dtype": "F32", "shape": [1000000], "data_offsets": [0, 4000000]}})).unwrap();
        let mut f = (h.len() as u64).to_le_bytes().to_vec();
        f.append(&mut h);
        std::fs::write(&p, f).unwrap();
        assert!(SafeTensors::open(&p).is_err());
        // a size that disagrees with the shape
        write_st(&p, &[("a".into(), vec![3], vec![0; 8], "F32")]);
        assert!(SafeTensors::open(&p).is_err());
        // an index that points outside the folder
        std::fs::write(d.join("model.safetensors.index.json"), r#"{"weight_map":{"a":"../evil.safetensors"}}"#).unwrap();
        assert!(SafeTensors::open(&d).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn unpermute_inverts_the_llama_cpp_permutation() {
        let (heads, hd, cols) = (3usize, 8usize, 5usize);
        let rows = heads * hd;
        let orig: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let half = hd / 2;
        let mut perm = vec![0.0; orig.len()];
        for h in 0..heads { for t in 0..2 { for j in 0..half {
            let (hf, gg) = (h * hd + t * half + j, h * hd + j * 2 + t);
            perm[gg * cols..(gg + 1) * cols].copy_from_slice(&orig[hf * cols..(hf + 1) * cols]);
        } } }
        assert_ne!(perm, orig);
        assert_eq!(unpermute_rows(&perm, rows, cols, heads), orig);
    }

    #[test]
    fn gguf_f32_file_with_permuted_llama_weights_reproduces_the_model() {
        let m = tiny(4, 2, false);
        let d = tmpdir();
        let p = d.join("tiny.gguf");
        let permute = |w: &[f32], heads: usize, cols: usize| {
            let hd = 8; let half = 4; let mut o = vec![0.0; w.len()];
            for h in 0..heads { for t in 0..2 { for j in 0..half {
                let (hf, gg) = (h * hd + t * half + j, h * hd + j * 2 + t);
                o[gg * cols..(gg + 1) * cols].copy_from_slice(&w[hf * cols..(hf + 1) * cols]);
            } } }
            o
        };
        let mut v = b"GGUF".to_vec();
        let ts = all(&m);
        v.extend(3u32.to_le_bytes()); v.extend((ts.len() as u64).to_le_bytes()); v.extend(1u64.to_le_bytes());
        crate::gguf::tests::put_str(&mut v, "general.architecture"); v.extend(8u32.to_le_bytes()); crate::gguf::tests::put_str(&mut v, "llama");
        let mut data = vec![];
        let mut payload = vec![];
        for (_, g, sh, d) in &ts {
            let d = if g.contains("attn_q") { permute(d, 4, 16) } else if g.contains("attn_k") { permute(d, 2, 16) } else { d.clone() };
            while data.len() % 32 != 0 { data.push(0); }
            crate::gguf::tests::put_str(&mut payload, g);
            payload.extend((sh.len() as u32).to_le_bytes());
            for x in sh.iter().rev() { payload.extend((*x as u64).to_le_bytes()); }
            payload.extend(0u32.to_le_bytes());
            payload.extend((data.len() as u64).to_le_bytes());
            data.extend(f32b(&d));
        }
        v.extend(payload);
        while v.len() % 32 != 0 { v.push(0); }
        v.extend(data);
        std::fs::write(&p, v).unwrap();
        let f = GgufFile::open(&p).unwrap();
        let loaded = load_dense(m.spec.clone(), &f, Naming::Gguf, true).unwrap();
        assert_eq!(run(&m), run(&loaded));
        // truncated copy: a clean error
        let b = std::fs::read(&p).unwrap();
        std::fs::write(&p, &b[..b.len() - 100]).unwrap();
        assert!(load_dense(m.spec.clone(), &GgufFile::open(&p).unwrap(), Naming::Gguf, true).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn missing_tensor_is_named_in_the_error() {
        let d = tmpdir();
        write_st(&d.join("m.safetensors"), &[("model.embed_tokens.weight".into(), vec![1], f32b(&[1.0]), "F32")]);
        let m = tiny(4, 2, false);
        let e = load_dense(m.spec.clone(), &SafeTensors::open(&d).unwrap(), Naming::Hf, false).err().unwrap();
        assert!(e.to_string().contains("model.layers.0.input_layernorm.weight"), "{e}");
        std::fs::remove_dir_all(d).unwrap();
    }
}
