//! LoRA adapters, applied the three ways engines apply them.
//!
//! * **Load** PEFT adapters (`adapter_config.json` + `adapter_model.safetensors`) and GGUF adapters
//!   (`general.type = adapter`, tensors `<base>.lora_a` / `<base>.lora_b`, `adapter.lora.alpha`).
//! * **Runtime** (`y = W·x + s·B(A·x)`): two thin mat-vecs on top of the untouched base product, so
//!   a quantised base is never dequantised. Adapters are stacked, each with its own scale, chosen per
//!   request and hot-swapped through a [`Registry`].
//! * **Merge** (`W' = W + s·B·A`) into dense f32 weights; or, on request, into a quantised matrix by
//!   dequantising it, merging and re-quantising to Q8_0 ([`merge_into_q8_0`]). That last step rounds
//!   the whole matrix again: at most half a Q8_0 step (about 0.4 % of a block's largest weight) per
//!   weight, on top of whatever error the base format already had.
//!
//! Nothing here touches `model.rs`. The model asks `Selection::apply(&key, x, out)` after each
//! linear layer, with the [`Key`] of that layer (see [`Key::hf_base_name`] / [`Key::gguf_base_name`]
//! for the matching base tensor names).
//!
//! Supported: plain LoRA, rank/alpha patterns, rsLoRA scaling, `fan_in_fan_out`. Refused with a
//! clear [`LoraError::Unsupported`]: DoRA, trained biases, `modules_to_save`, embedding/conv adapters.
#![allow(clippy::collapsible_if, clippy::type_complexity)]
use crate::gguf::Value;
use crate::loader::{GgufFile, LoadError, SafeTensors, TensorSource};
use crate::ops::{axpy, matvec};
use crate::qmatmul::qmatvec;
use crate::quant::{DType, dequantize};
use serde_json::Value as Json;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Largest rank accepted from a file; real adapters are well below this.
pub const MAX_RANK: usize = 8192;

#[derive(Debug)]
pub enum LoraError {
    Format(String),
    Shape(String),
    Unsupported(String),
    Unknown(String),
    Quant(String),
    Load(LoadError),
}
impl std::fmt::Display for LoraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoraError::Format(s) => write!(f, "bad adapter file: {s}"),
            LoraError::Shape(s) => write!(f, "adapter shape mismatch: {s}"),
            LoraError::Unsupported(s) => write!(f, "adapter feature not supported: {s}"),
            LoraError::Unknown(s) => write!(f, "no adapter named `{s}`"),
            LoraError::Quant(s) => write!(f, "{s}"),
            LoraError::Load(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for LoraError {}
impl From<LoadError> for LoraError {
    fn from(e: LoadError) -> Self {
        LoraError::Load(e)
    }
}
fn fmt_err<T>(s: impl Into<String>) -> Result<T, LoraError> {
    Err(LoraError::Format(s.into()))
}
fn shape_err<T>(s: impl Into<String>) -> Result<T, LoraError> {
    Err(LoraError::Shape(s.into()))
}

// ----------------------------------------------------------------------------- keys

/// Which linear layer of a block an adapter targets.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Module {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
    Other(String),
}

/// Naming-independent identity of a weight matrix: block index plus role.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Key {
    pub layer: Option<usize>,
    pub module: Module,
}

impl Key {
    pub fn new(layer: usize, module: Module) -> Key {
        Key {
            layer: Some(layer),
            module,
        }
    }

    /// From a Hugging Face module path such as `model.layers.3.self_attn.q_proj` (a trailing
    /// `.weight` is accepted).
    pub fn from_hf(path: &str) -> Key {
        let path = path.strip_suffix(".weight").unwrap_or(path);
        let segs: Vec<&str> = path.split('.').collect();
        for i in 0..segs.len().saturating_sub(1) {
            if matches!(segs[i], "layers" | "h" | "blocks" | "block") {
                if let Ok(n) = segs[i + 1].parse::<usize>() {
                    let rest = &segs[i + 2..];
                    let module = match rest.last().copied() {
                        Some("q_proj") => Module::Q,
                        Some("k_proj") => Module::K,
                        Some("v_proj") => Module::V,
                        Some("o_proj") => Module::O,
                        Some("gate_proj") => Module::Gate,
                        Some("up_proj") => Module::Up,
                        Some("down_proj") => Module::Down,
                        _ => Module::Other(rest.join(".")),
                    };
                    return Key {
                        layer: Some(n),
                        module,
                    };
                }
            }
        }
        Key {
            layer: None,
            module: Module::Other(path.to_string()),
        }
    }

    /// From a GGUF base tensor name such as `blk.3.attn_q.weight`.
    pub fn from_gguf(name: &str) -> Key {
        let name = name.strip_suffix(".weight").unwrap_or(name);
        if let Some(rest) = name.strip_prefix("blk.") {
            if let Some((n, tail)) = rest.split_once('.') {
                if let Ok(n) = n.parse::<usize>() {
                    let module = match tail {
                        "attn_q" => Module::Q,
                        "attn_k" => Module::K,
                        "attn_v" => Module::V,
                        "attn_output" => Module::O,
                        "ffn_gate" => Module::Gate,
                        "ffn_up" => Module::Up,
                        "ffn_down" => Module::Down,
                        other => Module::Other(other.to_string()),
                    };
                    return Key {
                        layer: Some(n),
                        module,
                    };
                }
            }
        }
        Key {
            layer: None,
            module: Module::Other(name.to_string()),
        }
    }

    /// Name of the base tensor in a Hugging Face checkpoint.
    pub fn hf_base_name(&self) -> String {
        let (n, tail) = match (&self.layer, &self.module) {
            (Some(n), Module::Q) => (*n, "self_attn.q_proj".to_string()),
            (Some(n), Module::K) => (*n, "self_attn.k_proj".to_string()),
            (Some(n), Module::V) => (*n, "self_attn.v_proj".to_string()),
            (Some(n), Module::O) => (*n, "self_attn.o_proj".to_string()),
            (Some(n), Module::Gate) => (*n, "mlp.gate_proj".to_string()),
            (Some(n), Module::Up) => (*n, "mlp.up_proj".to_string()),
            (Some(n), Module::Down) => (*n, "mlp.down_proj".to_string()),
            (Some(n), Module::Other(o)) => (*n, o.clone()),
            (None, Module::Other(o)) => return format!("{o}.weight"),
            (None, _) => return "unknown.weight".to_string(),
        };
        format!("model.layers.{n}.{tail}.weight")
    }

    /// Name of the base tensor in a GGUF file.
    pub fn gguf_base_name(&self) -> String {
        let tail = match &self.module {
            Module::Q => "attn_q",
            Module::K => "attn_k",
            Module::V => "attn_v",
            Module::O => "attn_output",
            Module::Gate => "ffn_gate",
            Module::Up => "ffn_up",
            Module::Down => "ffn_down",
            Module::Other(o) => o.as_str(),
        };
        match self.layer {
            Some(n) => format!("blk.{n}.{tail}.weight"),
            None => format!("{tail}.weight"),
        }
    }
}

// ----------------------------------------------------------------------------- one pair

/// The low-rank update of one matrix: `A` is `r × in`, `B` is `out × r`, row-major.
#[derive(Clone, Debug)]
pub struct LoraPair {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub r: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub alpha: f32,
    /// `alpha / r`, or `alpha / sqrt(r)` for rsLoRA.
    pub scale: f32,
}

impl LoraPair {
    pub fn new(
        a: Vec<f32>,
        b: Vec<f32>,
        r: usize,
        in_dim: usize,
        out_dim: usize,
        alpha: f64,
        rslora: bool,
    ) -> Result<LoraPair, LoraError> {
        if r == 0 || r > MAX_RANK || in_dim == 0 || out_dim == 0 {
            return shape_err(format!("rank {r}, {in_dim} inputs, {out_dim} outputs"));
        }
        if r.checked_mul(in_dim) != Some(a.len()) {
            return shape_err(format!("A has {} values, expected {r}×{in_dim}", a.len()));
        }
        if out_dim.checked_mul(r) != Some(b.len()) {
            return shape_err(format!("B has {} values, expected {out_dim}×{r}", b.len()));
        }
        if !alpha.is_finite() || alpha < 0.0 {
            return fmt_err(format!("alpha {alpha} is not a finite non-negative number"));
        }
        if a.iter().chain(&b).any(|v| !v.is_finite()) {
            return fmt_err("adapter weights contain NaN or infinity");
        }
        let scale = if rslora {
            alpha / (r as f64).sqrt()
        } else {
            alpha / r as f64
        };
        Ok(LoraPair {
            a,
            b,
            r,
            in_dim,
            out_dim,
            alpha: alpha as f32,
            scale: scale as f32,
        })
    }

    /// `out += mult · scale · B(A·x)`.
    pub fn add_delta(&self, x: &[f32], mult: f32, out: &mut [f32]) -> Result<(), LoraError> {
        if x.len() != self.in_dim {
            return shape_err(format!(
                "input has {} values, adapter expects {}",
                x.len(),
                self.in_dim
            ));
        }
        if out.len() != self.out_dim {
            return shape_err(format!(
                "output has {} values, adapter produces {}",
                out.len(),
                self.out_dim
            ));
        }
        let mut t = vec![0.0f32; self.r];
        matvec(&self.a, self.in_dim, x, &mut t);
        let mut d = vec![0.0f32; self.out_dim];
        matvec(&self.b, self.r, &t, &mut d);
        axpy(mult * self.scale, &d, out);
        Ok(())
    }

    /// `W += mult · scale · B·A` for a dense `rows × cols` matrix. With `fan_in_fan_out` the stored
    /// matrix is the transpose (`in × out`, as GPT-2's Conv1D keeps it).
    pub fn merge_dense(
        &self,
        w: &mut [f32],
        rows: usize,
        cols: usize,
        mult: f32,
        fan_in_fan_out: bool,
    ) -> Result<(), LoraError> {
        let (want_rows, want_cols) = if fan_in_fan_out {
            (self.in_dim, self.out_dim)
        } else {
            (self.out_dim, self.in_dim)
        };
        if rows != want_rows || cols != want_cols {
            return shape_err(format!(
                "matrix is {rows}×{cols}, adapter targets {want_rows}×{want_cols}"
            ));
        }
        if rows.checked_mul(cols) != Some(w.len()) {
            return shape_err("matrix buffer has the wrong length");
        }
        let s = mult * self.scale;
        if !fan_in_fan_out {
            for o in 0..self.out_dim {
                let row = &mut w[o * cols..(o + 1) * cols];
                for k in 0..self.r {
                    axpy(
                        s * self.b[o * self.r + k],
                        &self.a[k * self.in_dim..(k + 1) * self.in_dim],
                        row,
                    );
                }
            }
        } else {
            let mut bt = vec![0.0f32; self.r * self.out_dim];
            for o in 0..self.out_dim {
                for k in 0..self.r {
                    bt[k * self.out_dim + o] = self.b[o * self.r + k];
                }
            }
            for i in 0..self.in_dim {
                let row = &mut w[i * cols..(i + 1) * cols];
                for k in 0..self.r {
                    axpy(
                        s * self.a[k * self.in_dim + i],
                        &bt[k * self.out_dim..(k + 1) * self.out_dim],
                        row,
                    );
                }
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------- adapter

/// One loaded adapter: a pair per targeted matrix.
#[derive(Clone, Debug)]
pub struct Adapter {
    pub name: String,
    pub pairs: HashMap<Key, LoraPair>,
    /// Scale used when a request does not name one.
    pub default_scale: f32,
    pub fan_in_fan_out: bool,
}

/// Settings of a PEFT `adapter_config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct PeftConfig {
    pub r: usize,
    pub alpha: f64,
    pub rank_pattern: Vec<(String, usize)>,
    pub alpha_pattern: Vec<(String, f64)>,
    pub rslora: bool,
    pub fan_in_fan_out: bool,
}

impl PeftConfig {
    pub fn parse(json: &str) -> Result<PeftConfig, LoraError> {
        let v: Json = serde_json::from_str(json)
            .map_err(|e| LoraError::Format(format!("adapter_config.json: {e}")))?;
        let o = v
            .as_object()
            .ok_or_else(|| LoraError::Format("adapter_config.json is not an object".into()))?;
        if let Some(t) = o.get("peft_type").and_then(Json::as_str) {
            if !t.eq_ignore_ascii_case("LORA") {
                return Err(LoraError::Unsupported(format!("peft_type {t}")));
            }
        }
        if o.get("use_dora").and_then(Json::as_bool) == Some(true) {
            return Err(LoraError::Unsupported("DoRA".into()));
        }
        if let Some(b) = o.get("bias").and_then(Json::as_str) {
            if b != "none" {
                return Err(LoraError::Unsupported(format!(
                    "trained biases (bias = {b})"
                )));
            }
        }
        match o.get("modules_to_save") {
            None | Some(Json::Null) => {}
            Some(Json::Array(a)) if a.is_empty() => {}
            Some(_) => return Err(LoraError::Unsupported("modules_to_save".into())),
        }
        let r = o
            .get("r")
            .map_or(Some(8), Json::as_u64)
            .filter(|r| (1..=MAX_RANK as u64).contains(r))
            .ok_or_else(|| LoraError::Format(format!("`r` must be between 1 and {MAX_RANK}")))?
            as usize;
        let alpha = o
            .get("lora_alpha")
            .map_or(Some(8.0), Json::as_f64)
            .filter(|a| a.is_finite() && *a >= 0.0)
            .ok_or_else(|| {
                LoraError::Format("`lora_alpha` must be a finite non-negative number".into())
            })?;
        let mut rank_pattern = vec![];
        if let Some(m) = o.get("rank_pattern").and_then(Json::as_object) {
            for (k, v) in m {
                let r = v
                    .as_u64()
                    .filter(|r| (1..=MAX_RANK as u64).contains(r))
                    .ok_or_else(|| {
                        LoraError::Format(format!("rank_pattern[{k}] is not a valid rank"))
                    })?;
                rank_pattern.push((k.clone(), r as usize));
            }
        }
        let mut alpha_pattern = vec![];
        if let Some(m) = o.get("alpha_pattern").and_then(Json::as_object) {
            for (k, v) in m {
                let a = v
                    .as_f64()
                    .filter(|a| a.is_finite() && *a >= 0.0)
                    .ok_or_else(|| {
                        LoraError::Format(format!("alpha_pattern[{k}] is not a valid alpha"))
                    })?;
                alpha_pattern.push((k.clone(), a));
            }
        }
        Ok(PeftConfig {
            r,
            alpha,
            rank_pattern,
            alpha_pattern,
            rslora: o.get("use_rslora").and_then(Json::as_bool).unwrap_or(false),
            fan_in_fan_out: o
                .get("fan_in_fan_out")
                .and_then(Json::as_bool)
                .unwrap_or(false),
        })
    }
    fn rank_for(&self, module: &str) -> usize {
        pattern_lookup(&self.rank_pattern, module).unwrap_or(self.r)
    }
    fn alpha_for(&self, module: &str) -> f64 {
        pattern_lookup(&self.alpha_pattern, module).unwrap_or(self.alpha)
    }
}

/// PEFT matches a pattern key against the module name exactly or as a dotted suffix; the longest wins.
fn pattern_lookup<T: Copy>(pat: &[(String, T)], module: &str) -> Option<T> {
    pat.iter()
        .filter(|(k, _)| {
            module == k
                || module
                    .strip_suffix(k.as_str())
                    .is_some_and(|p| p.ends_with('.'))
        })
        .max_by_key(|(k, _)| k.len())
        .map(|(_, v)| *v)
}

/// A (rows, cols) pair from tensor dimensions; only 2-D matrices are accepted.
fn dims2(name: &str, dims: &[u64]) -> Result<(usize, usize), LoraError> {
    match dims {
        [r, c] => match (usize::try_from(*r), usize::try_from(*c)) {
            (Ok(r), Ok(c)) if r > 0 && c > 0 => Ok((r, c)),
            _ => shape_err(format!("{name}: empty or huge dimension")),
        },
        d => Err(LoraError::Unsupported(format!(
            "{name} has {} dimensions (only matrices are supported)",
            d.len()
        ))),
    }
}

/// Read both halves of a pair and check them against each other and the expected rank.
#[allow(clippy::too_many_arguments)]
fn build_pair(
    src: &dyn TensorSource,
    a: (&str, usize, usize),
    b: (&str, usize, usize),
    expect_r: Option<usize>,
    alpha: f64,
    rslora: bool,
) -> Result<LoraPair, LoraError> {
    let (a_name, r, in_dim) = a;
    let (b_name, out_dim, rb) = b;
    if r != rb {
        return shape_err(format!("{a_name} has rank {r} but {b_name} has rank {rb}"));
    }
    if let Some(e) = expect_r {
        if e != r {
            return shape_err(format!(
                "{a_name}: file says rank {r}, configuration says {e}"
            ));
        }
    }
    if r > MAX_RANK {
        return shape_err(format!("rank {r} is above the limit {MAX_RANK}"));
    }
    let (an, bn) = (r.checked_mul(in_dim), out_dim.checked_mul(r));
    let (Some(an), Some(bn)) = (an, bn) else {
        return shape_err("sizes overflow");
    };
    let av = src.read(a_name)?;
    let bv = src.read(b_name)?;
    if av.len() != an {
        return shape_err(format!("{a_name}: {} values, expected {an}", av.len()));
    }
    if bv.len() != bn {
        return shape_err(format!("{b_name}: {} values, expected {bn}", bv.len()));
    }
    LoraPair::new(av, bv, r, in_dim, out_dim, alpha, rslora)
}

impl Adapter {
    /// Load a PEFT adapter from any tensor source that can list its tensors, plus the text of
    /// `adapter_config.json`.
    pub fn from_peft(
        name: &str,
        src: &dyn TensorSource,
        config_json: &str,
    ) -> Result<Adapter, LoraError> {
        let cfg = PeftConfig::parse(config_json)?;
        let infos = src.infos();
        if infos.is_empty() {
            return fmt_err("the adapter file lists no tensors");
        }
        type Half = Option<(String, Vec<u64>)>;
        let mut mods: HashMap<String, (Half, Half)> = HashMap::new();
        for t in &infos {
            let n = t.name.as_str();
            if n.contains("modules_to_save") {
                return Err(LoraError::Unsupported("modules_to_save tensors".into()));
            }
            if n.contains("lora_embedding") || n.contains("lora_magnitude") {
                return Err(LoraError::Unsupported(format!(
                    "{n} (embedding or DoRA adapters)"
                )));
            }
            let (idx, is_a) = match (n.find(".lora_A"), n.find(".lora_B")) {
                (Some(i), _) => (i, true),
                (None, Some(i)) => (i, false),
                _ => continue,
            };
            if n.ends_with(".bias") {
                return Err(LoraError::Unsupported(format!("{n} (trained bias)")));
            }
            if !n.ends_with(".weight") {
                continue;
            }
            let module = n[..idx]
                .strip_prefix("base_model.model.")
                .unwrap_or(&n[..idx])
                .to_string();
            let slot = mods.entry(module.clone()).or_default();
            let half = if is_a { &mut slot.0 } else { &mut slot.1 };
            if half.is_some() {
                return fmt_err(format!(
                    "two {} tensors for {module}",
                    if is_a { "lora_A" } else { "lora_B" }
                ));
            }
            *half = Some((t.name.clone(), t.dims.clone()));
        }
        if mods.is_empty() {
            return fmt_err("no LoRA tensors found");
        }
        let mut pairs = HashMap::new();
        let mut names: Vec<_> = mods.keys().cloned().collect();
        names.sort();
        for module in names {
            let (a, b) = &mods[&module];
            let (Some((an, ad)), Some((bn, bd))) = (a, b) else {
                return fmt_err(format!("{module} has only one of lora_A / lora_B"));
            };
            let (ar, ac) = dims2(an, ad)?;
            let (br, bc) = dims2(bn, bd)?;
            let p = build_pair(
                src,
                (an, ar, ac),
                (bn, br, bc),
                Some(cfg.rank_for(&module)),
                cfg.alpha_for(&module),
                cfg.rslora,
            )?;
            if pairs.insert(Key::from_hf(&module), p).is_some() {
                return fmt_err(format!("{module} appears twice"));
            }
        }
        let default_scale = 1.0;
        Ok(Adapter {
            name: name.to_string(),
            pairs,
            default_scale,
            fan_in_fan_out: cfg.fan_in_fan_out,
        })
    }

    /// Load `adapter_config.json` and `adapter_model.safetensors` from a PEFT folder.
    pub fn from_peft_dir(name: &str, dir: &Path) -> Result<Adapter, LoraError> {
        let cfg_path = dir.join("adapter_config.json");
        let len = std::fs::metadata(&cfg_path)
            .map_err(|e| LoraError::Load(e.into()))?
            .len();
        if len > (1 << 20) {
            return fmt_err("adapter_config.json is implausibly large");
        }
        let cfg = std::fs::read_to_string(&cfg_path).map_err(|e| LoraError::Load(e.into()))?;
        let st = SafeTensors::open(&dir.join("adapter_model.safetensors"))?;
        Adapter::from_peft(name, &st, &cfg)
    }

    /// Load a GGUF adapter from any tensor source; `alpha` comes from `adapter.lora.alpha`.
    pub fn from_gguf_source(
        name: &str,
        src: &dyn TensorSource,
        alpha: f64,
    ) -> Result<Adapter, LoraError> {
        let infos = src.infos();
        let mut mods: HashMap<String, [Option<(String, Vec<u64>)>; 2]> = HashMap::new();
        for t in &infos {
            let (base, which) = if let Some(b) = t.name.strip_suffix(".lora_a") {
                (b, 0)
            } else if let Some(b) = t.name.strip_suffix(".lora_b") {
                (b, 1)
            } else {
                continue;
            };
            let slot = &mut mods.entry(base.to_string()).or_default()[which];
            if slot.is_some() {
                return fmt_err(format!("two tensors for {}", t.name));
            }
            *slot = Some((t.name.clone(), t.dims.clone()));
        }
        if mods.is_empty() {
            return fmt_err("no LoRA tensors found");
        }
        let mut names: Vec<_> = mods.keys().cloned().collect();
        names.sort();
        let mut pairs = HashMap::new();
        for base in names {
            let [Some((an, ad)), Some((bn, bd))] = &mods[&base] else {
                return fmt_err(format!("{base} has only one of lora_a / lora_b"));
            };
            // GGUF lists [fastest, slowest] = [cols, rows].
            let (a_cols, a_rows) = dims2(an, ad)?;
            let (b_cols, b_rows) = dims2(bn, bd)?;
            let p = build_pair(
                src,
                (an, a_rows, a_cols),
                (bn, b_rows, b_cols),
                None,
                alpha,
                false,
            )?;
            if pairs.insert(Key::from_gguf(&base), p).is_some() {
                return fmt_err(format!("{base} appears twice"));
            }
        }
        Ok(Adapter {
            name: name.to_string(),
            pairs,
            default_scale: 1.0,
            fan_in_fan_out: false,
        })
    }

    /// Load a `.gguf` adapter file (checks `general.type`).
    pub fn from_gguf_file(name: &str, file: &GgufFile) -> Result<Adapter, LoraError> {
        let meta = &file.gguf.meta;
        match meta.get("general.type").and_then(Value::as_str) {
            Some("adapter") => {}
            Some(other) => return fmt_err(format!("general.type is `{other}`, not an adapter")),
            None => return fmt_err("the file has no general.type; it is not an adapter"),
        }
        if let Some(t) = meta.get("adapter.type").and_then(Value::as_str) {
            if t != "lora" {
                return Err(LoraError::Unsupported(format!("adapter.type {t}")));
            }
        }
        let alpha = meta
            .get("adapter.lora.alpha")
            .and_then(Value::as_f64)
            .ok_or_else(|| LoraError::Format("missing adapter.lora.alpha".into()))?;
        Adapter::from_gguf_source(name, file, alpha)
    }

    pub fn from_gguf_path(name: &str, path: &Path) -> Result<Adapter, LoraError> {
        Adapter::from_gguf_file(name, &GgufFile::open(path)?)
    }

    pub fn keys(&self) -> Vec<Key> {
        let mut k: Vec<_> = self.pairs.keys().cloned().collect();
        k.sort();
        k
    }

    /// Check every pair against the base model; `base_shape(key)` returns the `(out, in)` size of the
    /// base matrix, or `None` when the model has no such matrix.
    pub fn check_shapes(
        &self,
        base_shape: impl Fn(&Key) -> Option<(usize, usize)>,
    ) -> Result<(), LoraError> {
        for k in self.keys() {
            let p = &self.pairs[&k];
            match base_shape(&k) {
                None => {
                    return shape_err(format!(
                        "the model has no matrix {k:?} that adapter `{}` targets",
                        self.name
                    ));
                }
                Some((o, i)) if (o, i) != (p.out_dim, p.in_dim) => {
                    return shape_err(format!(
                        "{k:?}: adapter is {}×{}, model matrix is {o}×{i}",
                        p.out_dim, p.in_dim
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// GGUF files of the `llama` architecture keep Q and K rows permuted for interleaved RoPE (the
    /// loader undoes this for the base weights). The same row shuffle applies to the adapter's `B`.
    pub fn unpermute_llama(&mut self, heads: usize, kv_heads: usize) -> Result<(), LoraError> {
        for (k, p) in &mut self.pairs {
            let n = match k.module {
                Module::Q => heads,
                Module::K => kv_heads,
                _ => continue,
            };
            if n == 0 || p.out_dim % (2 * n) != 0 {
                return shape_err(format!(
                    "{} rows cannot be split into {n} heads of even width",
                    p.out_dim
                ));
            }
            p.b = unpermute_rows_local(&p.b, p.out_dim, p.r, n);
        }
        Ok(())
    }
}

/// Row shuffle that undoes llama.cpp's Q/K permutation: `out[h·hd + t·half + j] = in[h·hd + 2j + t]`.
fn unpermute_rows_local(w: &[f32], rows: usize, cols: usize, n_head: usize) -> Vec<f32> {
    let hd = rows / n_head;
    let half = hd / 2;
    let mut out = vec![0.0; w.len()];
    for h in 0..n_head {
        for t in 0..2 {
            for j in 0..half {
                let (dst, src) = (h * hd + t * half + j, h * hd + 2 * j + t);
                out[dst * cols..(dst + 1) * cols].copy_from_slice(&w[src * cols..(src + 1) * cols]);
            }
        }
    }
    out
}

// ----------------------------------------------------------------------------- registry

/// The adapters currently loaded, swappable while requests are running: a request holds a
/// [`Selection`] of `Arc`s, so removing or replacing an adapter never disturbs it.
#[derive(Default)]
pub struct Registry {
    inner: RwLock<HashMap<String, Arc<Adapter>>>,
}

impl Registry {
    pub fn new() -> Registry {
        Registry::default()
    }
    fn map(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, Arc<Adapter>>> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }
    /// Add or replace an adapter; returns the one it replaced.
    pub fn insert(&self, a: Adapter) -> Option<Arc<Adapter>> {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(a.name.clone(), Arc::new(a))
    }
    pub fn remove(&self, name: &str) -> Option<Arc<Adapter>> {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name)
    }
    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<_> = self.map().keys().cloned().collect();
        n.sort();
        n
    }
    pub fn len(&self) -> usize {
        self.map().len()
    }
    pub fn is_empty(&self) -> bool {
        self.map().is_empty()
    }
    pub fn get(&self, name: &str) -> Option<Arc<Adapter>> {
        self.map().get(name).cloned()
    }
    /// Pick adapters for one request, each with its own scale.
    pub fn select(&self, picks: &[(&str, f32)]) -> Result<Selection, LoraError> {
        let g = self.map();
        let mut entries = Vec::with_capacity(picks.len());
        for (n, s) in picks {
            if !s.is_finite() {
                return fmt_err(format!("scale of `{n}` is not finite"));
            }
            entries.push((
                g.get(*n)
                    .cloned()
                    .ok_or_else(|| LoraError::Unknown((*n).to_string()))?,
                *s,
            ));
        }
        Ok(Selection { entries })
    }
    /// Pick adapters by name at their default scale.
    pub fn select_defaults(&self, names: &[&str]) -> Result<Selection, LoraError> {
        let g = self.map();
        let mut entries = Vec::with_capacity(names.len());
        for n in names {
            let a = g
                .get(*n)
                .cloned()
                .ok_or_else(|| LoraError::Unknown((*n).to_string()))?;
            let s = a.default_scale;
            entries.push((a, s));
        }
        Ok(Selection { entries })
    }
}

/// The adapters one request uses, stacked: their updates simply add.
#[derive(Clone, Default)]
pub struct Selection {
    entries: Vec<(Arc<Adapter>, f32)>,
}

impl Selection {
    pub fn single(a: Adapter, scale: f32) -> Selection {
        Selection {
            entries: vec![(Arc::new(a), scale)],
        }
    }
    pub fn from_adapters(list: Vec<(Arc<Adapter>, f32)>) -> Selection {
        Selection { entries: list }
    }
    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(|(_, s)| *s == 0.0)
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn scale_of(&self, name: &str) -> Option<f32> {
        self.entries
            .iter()
            .find(|(a, _)| a.name == name)
            .map(|(_, s)| *s)
    }

    /// `out += Σ scale_i · B_i(A_i·x)` for the matrix `key`. Adapters that do not touch it, or have
    /// scale 0, cost nothing.
    pub fn apply(&self, key: &Key, x: &[f32], out: &mut [f32]) -> Result<(), LoraError> {
        for (a, s) in &self.entries {
            if *s == 0.0 {
                continue;
            }
            if let Some(p) = a.pairs.get(key) {
                p.add_delta(x, *s, out)?;
            }
        }
        Ok(())
    }

    /// The same for `n` tokens at once (`xs` is `n × in`, `out` is `n × out`).
    pub fn apply_batch(
        &self,
        key: &Key,
        xs: &[f32],
        n: usize,
        out: &mut [f32],
    ) -> Result<(), LoraError> {
        if n == 0 {
            return Ok(());
        }
        if !xs.len().is_multiple_of(n) || !out.len().is_multiple_of(n) {
            return shape_err("batch buffers do not divide into tokens");
        }
        let (i, o) = (xs.len() / n, out.len() / n);
        for t in 0..n {
            self.apply(key, &xs[t * i..(t + 1) * i], &mut out[t * o..(t + 1) * o])?;
        }
        Ok(())
    }

    /// Merge every selected adapter into a dense `rows × cols` base matrix.
    pub fn merge_dense(
        &self,
        key: &Key,
        w: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<(), LoraError> {
        for (a, s) in &self.entries {
            if *s == 0.0 {
                continue;
            }
            if let Some(p) = a.pairs.get(key) {
                p.merge_dense(w, rows, cols, *s, a.fan_in_fan_out)?;
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------- base products

fn quant_err(e: impl std::fmt::Display) -> LoraError {
    LoraError::Quant(e.to_string())
}

/// `out = W·x (+ adapters)` for a base matrix kept in any format `qmatvec` understands. The base
/// product runs on the stored bytes; the adapter adds two thin products on top.
pub fn linear_with_lora(
    dtype: DType,
    w: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
    lora: Option<(&Selection, &Key)>,
) -> Result<(), LoraError> {
    qmatvec(dtype, w, rows, cols, x, out).map_err(quant_err)?;
    if let Some((sel, key)) = lora {
        sel.apply(key, x, out)?;
    }
    Ok(())
}

/// Dequantise a base matrix and merge the adapters into it: dense f32 result, no further rounding.
pub fn merge_dequantised(
    dtype: DType,
    w: &[u8],
    rows: usize,
    cols: usize,
    sel: &Selection,
    key: &Key,
) -> Result<Vec<f32>, LoraError> {
    let n = rows
        .checked_mul(cols)
        .ok_or_else(|| LoraError::Shape("matrix size overflows".into()))?;
    let mut dense = Vec::new();
    dequantize(dtype, w, n, &mut dense).map_err(quant_err)?;
    sel.merge_dense(key, &mut dense, rows, cols)?;
    Ok(dense)
}

/// Merge into a quantised matrix and re-quantise the result to Q8_0 (`cols` must be a multiple of
/// 32). Costs one more rounding step of the whole matrix; see the module notes.
pub fn merge_into_q8_0(
    dtype: DType,
    w: &[u8],
    rows: usize,
    cols: usize,
    sel: &Selection,
    key: &Key,
) -> Result<Vec<u8>, LoraError> {
    if !cols.is_multiple_of(32) {
        return Err(LoraError::Unsupported(format!(
            "Q8_0 needs a multiple of 32 columns, got {cols}"
        )));
    }
    Ok(quantise_q8_0(&merge_dequantised(
        dtype, w, rows, cols, sel, key,
    )?))
}

/// IEEE binary16 with round-to-nearest-even.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return s | 0x7c00;
    }
    if e16 <= 0 {
        if e16 < -10 {
            return s;
        }
        let m24 = m | 0x80_0000;
        let shift = (14 - e16) as u32;
        let mut h = (m24 >> shift) as u16;
        let rem = m24 & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && h & 1 == 1) {
            h += 1;
        }
        return s | h;
    }
    let mut h = (((e16 as u32) << 10) | (m >> 13)) as u16;
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    s | h
}

/// Q8_0 blocks (f16 scale + 32 × i8) for a slice whose length is a multiple of 32.
pub fn quantise_q8_0(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(w.len() / 32 * 34);
    for blk in w.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let hd = f32_to_f16(amax / 127.0);
        let d = crate::quant::f16_to_f32(hd);
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&hd.to_le_bytes());
        out.extend(
            blk.iter()
                .map(|v| (v * inv).round().clamp(-127.0, 127.0) as i8 as u8),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::put_str;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn rng(s: &mut u64) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
    fn vecr(n: usize, s: &mut u64) -> Vec<f32> {
        (0..n).map(|_| rng(s)).collect()
    }
    fn f32b(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn tmpdir() -> std::path::PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "engine-lora-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn pair(out: usize, inn: usize, r: usize, alpha: f64, seed: u64) -> LoraPair {
        let mut s = seed | 1;
        LoraPair::new(
            vecr(r * inn, &mut s),
            vecr(out * r, &mut s),
            r,
            inn,
            out,
            alpha,
            false,
        )
        .unwrap()
    }
    fn adapter(name: &str, key: Key, p: LoraPair) -> Adapter {
        Adapter {
            name: name.into(),
            pairs: HashMap::from([(key, p)]),
            default_scale: 1.0,
            fan_in_fan_out: false,
        }
    }
    fn naive_matvec(w: &[f32], cols: usize, x: &[f32]) -> Vec<f32> {
        w.chunks(cols)
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }
    fn close(a: &[f32], b: &[f32], tol: f32) {
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{i}: {x} vs {y}");
        }
        assert_eq!(a.len(), b.len());
    }

    /// Write a safetensors file with named F32 tensors.
    fn write_st(path: &Path, ts: &[(&str, Vec<usize>, Vec<f32>)]) {
        let mut hdr = serde_json::Map::new();
        let mut off = 0;
        for (n, shape, data) in ts {
            hdr.insert((*n).to_string(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [off, off + data.len() * 4]}));
            off += data.len() * 4;
        }
        let h = serde_json::to_vec(&hdr).unwrap();
        let mut f = (h.len() as u64).to_le_bytes().to_vec();
        f.extend(h);
        for t in ts {
            f.extend(f32b(&t.2));
        }
        std::fs::write(path, f).unwrap();
    }
    const A: &str = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
    const B: &str = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight";
    fn peft_dir(cfg: &str, ts: &[(&str, Vec<usize>, Vec<f32>)]) -> std::path::PathBuf {
        let d = tmpdir();
        std::fs::write(d.join("adapter_config.json"), cfg).unwrap();
        write_st(&d.join("adapter_model.safetensors"), ts);
        d
    }

    #[test]
    fn keys_round_trip_between_the_two_naming_schemes() {
        for (hf, gg, key) in [
            (
                "model.layers.3.self_attn.q_proj",
                "blk.3.attn_q.weight",
                Key::new(3, Module::Q),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "blk.0.attn_k.weight",
                Key::new(0, Module::K),
            ),
            (
                "model.layers.12.self_attn.v_proj",
                "blk.12.attn_v.weight",
                Key::new(12, Module::V),
            ),
            (
                "model.layers.1.self_attn.o_proj",
                "blk.1.attn_output.weight",
                Key::new(1, Module::O),
            ),
            (
                "model.layers.2.mlp.gate_proj",
                "blk.2.ffn_gate.weight",
                Key::new(2, Module::Gate),
            ),
            (
                "model.layers.2.mlp.up_proj",
                "blk.2.ffn_up.weight",
                Key::new(2, Module::Up),
            ),
            (
                "model.layers.9.mlp.down_proj",
                "blk.9.ffn_down.weight",
                Key::new(9, Module::Down),
            ),
        ] {
            assert_eq!(Key::from_hf(hf), key, "{hf}");
            assert_eq!(Key::from_gguf(gg), key, "{gg}");
            assert_eq!(Key::from_hf(&key.hf_base_name()), key);
            assert_eq!(Key::from_gguf(&key.gguf_base_name()), key);
        }
        assert_eq!(
            Key::from_gguf("output.weight"),
            Key {
                layer: None,
                module: Module::Other("output".into())
            }
        );
        assert_eq!(
            Key::from_hf("lm_head"),
            Key {
                layer: None,
                module: Module::Other("lm_head".into())
            }
        );
        assert_eq!(
            Key::from_hf("model.layers.4.block_sparse_moe.gate").layer,
            Some(4)
        );
    }

    #[test]
    fn merged_equals_runtime_on_f32() {
        let (out, inn, r) = (12, 20, 4);
        let mut s = 7u64;
        let w = vecr(out * inn, &mut s);
        let x = vecr(inn, &mut s);
        let key = Key::new(0, Module::Q);
        let sel = Selection::single(adapter("a", key.clone(), pair(out, inn, r, 8.0, 3)), 0.75);
        let mut runtime = vec![0.0; out];
        linear_with_lora(
            DType::F32,
            &f32b(&w),
            out,
            inn,
            &x,
            &mut runtime,
            Some((&sel, &key)),
        )
        .unwrap();
        let mut merged = w.clone();
        sel.merge_dense(&key, &mut merged, out, inn).unwrap();
        close(&runtime, &naive_matvec(&merged, inn, &x), 1e-4);
        // and the update really changed something
        assert!(
            runtime
                .iter()
                .zip(naive_matvec(&w, inn, &x))
                .any(|(a, b)| (a - b).abs() > 1e-3)
        );
    }

    #[test]
    fn scale_zero_is_identity() {
        let (out, inn) = (6, 10);
        let mut s = 5u64;
        let w = vecr(out * inn, &mut s);
        let x = vecr(inn, &mut s);
        let key = Key::new(1, Module::V);
        let sel = Selection::single(adapter("a", key.clone(), pair(out, inn, 3, 16.0, 9)), 0.0);
        assert!(sel.is_empty());
        let (mut y0, mut y1) = (vec![0.0; out], vec![0.0; out]);
        qmatvec(DType::F32, &f32b(&w), out, inn, &x, &mut y0).unwrap();
        linear_with_lora(
            DType::F32,
            &f32b(&w),
            out,
            inn,
            &x,
            &mut y1,
            Some((&sel, &key)),
        )
        .unwrap();
        assert_eq!(y0, y1);
        let mut m = w.clone();
        sel.merge_dense(&key, &mut m, out, inn).unwrap();
        assert_eq!(m, w);
        // an adapter with alpha 0 is also the identity
        let sel = Selection::single(adapter("z", key.clone(), pair(out, inn, 3, 0.0, 9)), 1.0);
        let mut m = w.clone();
        sel.merge_dense(&key, &mut m, out, inn).unwrap();
        assert_eq!(m, w);
    }

    #[test]
    fn shape_mismatches_are_errors() {
        let p = pair(6, 10, 3, 6.0, 1);
        assert!(matches!(
            p.add_delta(&[0.0; 9], 1.0, &mut [0.0; 6]),
            Err(LoraError::Shape(_))
        ));
        assert!(matches!(
            p.add_delta(&[0.0; 10], 1.0, &mut [0.0; 5]),
            Err(LoraError::Shape(_))
        ));
        assert!(matches!(
            p.merge_dense(&mut [0.0; 60], 10, 6, 1.0, false),
            Err(LoraError::Shape(_))
        ));
        assert!(LoraPair::new(vec![0.0; 5], vec![0.0; 6], 3, 10, 2, 1.0, false).is_err());
        assert!(LoraPair::new(vec![], vec![], 0, 1, 1, 1.0, false).is_err());
        assert!(LoraPair::new(vec![0.0; 2], vec![0.0; 2], 1, 2, 2, f64::NAN, false).is_err());
        assert!(LoraPair::new(vec![f32::NAN, 0.0], vec![0.0; 2], 1, 2, 2, 1.0, false).is_err());
        let key = Key::new(0, Module::Q);
        let a = adapter("a", key.clone(), p);
        assert!(a.check_shapes(|_| Some((6, 10))).is_ok());
        assert!(a.check_shapes(|_| Some((6, 11))).is_err());
        assert!(a.check_shapes(|_| None).is_err());
    }

    #[test]
    fn stacking_is_linear() {
        let (out, inn) = (8, 12);
        let mut s = 21u64;
        let x = vecr(inn, &mut s);
        let key = Key::new(0, Module::Up);
        let (a1, a2) = (
            adapter("a1", key.clone(), pair(out, inn, 2, 4.0, 11)),
            adapter("a2", key.clone(), pair(out, inn, 5, 10.0, 12)),
        );
        let one = |a: &Adapter, sc: f32| {
            let mut y = vec![0.0; out];
            Selection::single(a.clone(), sc)
                .apply(&key, &x, &mut y)
                .unwrap();
            y
        };
        let both = {
            let mut y = vec![0.0; out];
            Selection::from_adapters(vec![
                (Arc::new(a1.clone()), 0.5),
                (Arc::new(a2.clone()), -1.5),
            ])
            .apply(&key, &x, &mut y)
            .unwrap();
            y
        };
        let sum: Vec<f32> = one(&a1, 0.5)
            .iter()
            .zip(one(&a2, -1.5))
            .map(|(a, b)| a + b)
            .collect();
        close(&both, &sum, 1e-5);
        // doubling the scale doubles the update
        let d1 = one(&a1, 1.0);
        let d2 = one(&a1, 2.0);
        close(&d2, &d1.iter().map(|v| v * 2.0).collect::<Vec<_>>(), 1e-5);
        // merged stack equals the runtime stack
        let w = vecr(out * inn, &mut s);
        let sel = Selection::from_adapters(vec![(Arc::new(a1), 0.5), (Arc::new(a2), -1.5)]);
        let mut m = w.clone();
        sel.merge_dense(&key, &mut m, out, inn).unwrap();
        let mut y = naive_matvec(&w, inn, &x);
        sel.apply(&key, &x, &mut y).unwrap();
        close(&naive_matvec(&m, inn, &x), &y, 1e-4);
        // a matrix no adapter touches is left alone
        let mut y = vec![1.0; out];
        sel.apply(&Key::new(7, Module::Up), &x, &mut y).unwrap();
        assert_eq!(y, vec![1.0; out]);
    }

    #[test]
    fn rank_and_alpha_patterns_are_honoured() {
        let cfg = r#"{"peft_type":"LORA","r":2,"lora_alpha":4,
            "rank_pattern":{"q_proj":3,"model.layers.1.self_attn.q_proj":4},
            "alpha_pattern":{"q_proj":12,"layers.1.self_attn.q_proj":20}}"#;
        let mut s = 3u64;
        let mut q = |layer: usize, r: usize| {
            (
                format!("base_model.model.model.layers.{layer}.self_attn.q_proj.lora_A.weight"),
                vec![r, 10],
                vecr(r * 10, &mut s),
            )
        };
        let (a0, a1) = (q(0, 3), q(1, 4));
        let (b0, b1) = (vecr(6 * 3, &mut s), vecr(6 * 4, &mut s));
        let d = peft_dir(
            cfg,
            &[
                (&a0.0, a0.1.clone(), a0.2.clone()),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    vec![6, 3],
                    b0,
                ),
                (&a1.0, a1.1.clone(), a1.2.clone()),
                (
                    "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight",
                    vec![6, 4],
                    b1,
                ),
            ],
        );
        let ad = Adapter::from_peft_dir("p", &d).unwrap();
        let (p0, p1) = (
            &ad.pairs[&Key::new(0, Module::Q)],
            &ad.pairs[&Key::new(1, Module::Q)],
        );
        assert_eq!((p0.r, p0.alpha), (3, 12.0));
        assert!((p0.scale - 4.0).abs() < 1e-6);
        assert_eq!((p1.r, p1.alpha), (4, 20.0));
        assert!((p1.scale - 5.0).abs() < 1e-6);
        // a rank that contradicts the configuration is refused
        let bad = cfg.replace(r#""q_proj":3,"#, r#""q_proj":2,"#);
        let d2 = peft_dir(
            &bad,
            &[
                (&a0.0, a0.1.clone(), a0.2.clone()),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    vec![6, 3],
                    vec![0.0; 18],
                ),
            ],
        );
        assert!(matches!(
            Adapter::from_peft_dir("p", &d2),
            Err(LoraError::Shape(_))
        ));
        for d in [d, d2] {
            std::fs::remove_dir_all(d).unwrap();
        }
    }

    #[test]
    fn rslora_scales_by_the_square_root_of_the_rank() {
        let p = LoraPair::new(vec![0.0; 16 * 3], vec![0.0; 2 * 16], 16, 3, 2, 8.0, true).unwrap();
        assert!((p.scale - 2.0).abs() < 1e-6); // 8 / sqrt(16)
        let p = LoraPair::new(vec![0.0; 16 * 3], vec![0.0; 2 * 16], 16, 3, 2, 8.0, false).unwrap();
        assert!((p.scale - 0.5).abs() < 1e-6);
        let cfg =
            PeftConfig::parse(r#"{"r":16,"lora_alpha":8,"use_rslora":true,"fan_in_fan_out":true}"#)
                .unwrap();
        assert!(cfg.rslora && cfg.fan_in_fan_out);
    }

    #[test]
    fn fan_in_fan_out_merges_into_the_transposed_matrix() {
        let (out, inn, r) = (7, 9, 3);
        let p = pair(out, inn, r, 6.0, 5);
        let mut s = 8u64;
        let w = vecr(out * inn, &mut s); // out × in
        let mut normal = w.clone();
        p.merge_dense(&mut normal, out, inn, 1.0, false).unwrap();
        let mut wt = vec![0.0; w.len()]; // in × out
        for o in 0..out {
            for i in 0..inn {
                wt[i * out + o] = w[o * inn + i];
            }
        }
        p.merge_dense(&mut wt, inn, out, 1.0, true).unwrap();
        for o in 0..out {
            for i in 0..inn {
                assert!((wt[i * out + o] - normal[o * inn + i]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn peft_files_load_and_apply() {
        let mut s = 31u64;
        let (a, b) = (vecr(4 * 10, &mut s), vecr(6 * 4, &mut s));
        let d = peft_dir(
            r#"{"peft_type":"LORA","r":4,"lora_alpha":8,"target_modules":["q_proj"],"bias":"none"}"#,
            &[(A, vec![4, 10], a.clone()), (B, vec![6, 4], b.clone())],
        );
        let ad = Adapter::from_peft_dir("demo", &d).unwrap();
        assert_eq!(ad.keys(), vec![Key::new(0, Module::Q)]);
        let p = &ad.pairs[&Key::new(0, Module::Q)];
        assert_eq!((p.r, p.in_dim, p.out_dim), (4, 10, 6));
        assert!((p.scale - 2.0).abs() < 1e-6);
        let x = vecr(10, &mut s);
        let mut y = vec![0.0; 6];
        p.add_delta(&x, 1.0, &mut y).unwrap();
        let t = naive_matvec(&a, 10, &x);
        let want: Vec<f32> = naive_matvec(&b, 4, &t).iter().map(|v| v * 2.0).collect();
        close(&y, &want, 1e-5);
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn hostile_peft_files_are_errors_not_panics() {
        let ok = (A, vec![4, 10], vec![0.5f32; 40]);
        let okb = (B, vec![6, 4], vec![0.5f32; 24]);
        let cfg = r#"{"r":4,"lora_alpha":8}"#;
        let cases: Vec<(&str, Vec<(&str, Vec<usize>, Vec<f32>)>)> = vec![
            (cfg, vec![ok.clone()]),                                      // B missing
            (cfg, vec![okb.clone()]),                                     // A missing
            (cfg, vec![ok.clone(), (B, vec![6, 3], vec![0.5; 18])]),      // ranks differ
            (cfg, vec![ok.clone(), (B, vec![6, 4], vec![f32::NAN; 24])]), // NaN
            (cfg, vec![ok.clone(), (B, vec![2, 3, 4], vec![0.5; 24])]),   // 3-D
            (cfg, vec![("x.weight", vec![1], vec![0.0])]),                // no LoRA tensors
            (
                cfg,
                vec![
                    ok.clone(),
                    okb.clone(),
                    (
                        "base_model.model.model.layers.0.self_attn.q_proj.lora_embedding_A",
                        vec![1],
                        vec![0.0],
                    ),
                ],
            ),
            (
                r#"{"r":4,"lora_alpha":8,"use_dora":true}"#,
                vec![ok.clone(), okb.clone()],
            ),
            (
                r#"{"r":4,"lora_alpha":8,"bias":"all"}"#,
                vec![ok.clone(), okb.clone()],
            ),
            (
                r#"{"r":4,"lora_alpha":8,"modules_to_save":["lm_head"]}"#,
                vec![ok.clone(), okb.clone()],
            ),
            (
                r#"{"r":4,"lora_alpha":8,"peft_type":"IA3"}"#,
                vec![ok.clone(), okb.clone()],
            ),
            (r#"{"r":0,"lora_alpha":8}"#, vec![ok.clone(), okb.clone()]),
            (r#"{"r":4,"lora_alpha":-1}"#, vec![ok.clone(), okb.clone()]),
            (
                r#"{"r":99999999,"lora_alpha":8}"#,
                vec![ok.clone(), okb.clone()],
            ),
            (
                r#"{"r":4,"lora_alpha":8,"rank_pattern":{"q_proj":"many"}}"#,
                vec![ok.clone(), okb.clone()],
            ),
            (r#"[1,2]"#, vec![ok.clone(), okb.clone()]),
            ("not json", vec![ok.clone(), okb.clone()]),
            (
                cfg,
                vec![
                    ok.clone(),
                    okb.clone(),
                    (
                        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
                        vec![6, 4],
                        vec![0.0; 24],
                    ),
                ],
            ), // duplicate B
        ];
        for (i, (c, ts)) in cases.iter().enumerate() {
            let d = peft_dir(c, ts);
            assert!(
                Adapter::from_peft_dir("x", &d).is_err(),
                "case {i} was accepted"
            );
            std::fs::remove_dir_all(d).unwrap();
        }
        // an oversized claim and a truncated file
        let d = peft_dir(cfg, &[ok.clone(), okb.clone()]);
        let f = d.join("adapter_model.safetensors");
        let b = std::fs::read(&f).unwrap();
        for cut in [0, 3, 8, 20, b.len() / 2, b.len() - 1] {
            std::fs::write(&f, &b[..cut]).unwrap();
            assert!(Adapter::from_peft_dir("x", &d).is_err(), "cut {cut}");
        }
        assert!(Adapter::from_peft_dir("x", &d.join("missing")).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }

    /// A GGUF adapter file: each tensor is `(name, [rows, cols] as the matrix, data)`; dims are
    /// written in GGUF order `[cols, rows]`.
    fn gguf_adapter(
        path: &Path,
        general_type: &str,
        alpha: Option<f32>,
        ts: &[(&str, usize, usize, Vec<f32>)],
    ) {
        let mut v = b"GGUF".to_vec();
        v.extend(3u32.to_le_bytes());
        v.extend((ts.len() as u64).to_le_bytes());
        v.extend((2 + alpha.is_some() as u64).to_le_bytes());
        put_str(&mut v, "general.type");
        v.extend(8u32.to_le_bytes());
        put_str(&mut v, general_type);
        put_str(&mut v, "general.architecture");
        v.extend(8u32.to_le_bytes());
        put_str(&mut v, "llama");
        if let Some(a) = alpha {
            put_str(&mut v, "adapter.lora.alpha");
            v.extend(6u32.to_le_bytes());
            v.extend(a.to_le_bytes());
        }
        let mut data = vec![];
        for (n, rows, cols, d) in ts {
            while !data.len().is_multiple_of(32) {
                data.push(0);
            }
            put_str(&mut v, n);
            v.extend(2u32.to_le_bytes());
            v.extend((*cols as u64).to_le_bytes());
            v.extend((*rows as u64).to_le_bytes());
            v.extend(0u32.to_le_bytes());
            v.extend((data.len() as u64).to_le_bytes());
            data.extend(f32b(d));
        }
        while !v.len().is_multiple_of(32) {
            v.push(0);
        }
        v.extend(data);
        std::fs::write(path, v).unwrap();
    }

    #[test]
    fn gguf_adapters_load_with_their_alpha() {
        let d = tmpdir();
        let p = d.join("lora.gguf");
        let mut s = 77u64;
        let (a, b) = (vecr(2 * 8, &mut s), vecr(6 * 2, &mut s));
        gguf_adapter(
            &p,
            "adapter",
            Some(4.0),
            &[
                ("blk.2.attn_q.weight.lora_a", 2, 8, a.clone()),
                ("blk.2.attn_q.weight.lora_b", 6, 2, b.clone()),
            ],
        );
        let ad = Adapter::from_gguf_path("g", &p).unwrap();
        let pr = &ad.pairs[&Key::new(2, Module::Q)];
        assert_eq!((pr.r, pr.in_dim, pr.out_dim, pr.alpha), (2, 8, 6, 4.0));
        assert!((pr.scale - 2.0).abs() < 1e-6);
        assert_eq!(pr.a, a);
        assert_eq!(pr.b, b);
        // not an adapter / no alpha / half a pair / mismatching rank
        gguf_adapter(
            &p,
            "model",
            Some(4.0),
            &[
                ("blk.2.attn_q.weight.lora_a", 2, 8, a.clone()),
                ("blk.2.attn_q.weight.lora_b", 6, 2, b.clone()),
            ],
        );
        assert!(Adapter::from_gguf_path("g", &p).is_err());
        gguf_adapter(
            &p,
            "adapter",
            None,
            &[
                ("blk.2.attn_q.weight.lora_a", 2, 8, a.clone()),
                ("blk.2.attn_q.weight.lora_b", 6, 2, b.clone()),
            ],
        );
        assert!(Adapter::from_gguf_path("g", &p).is_err());
        gguf_adapter(
            &p,
            "adapter",
            Some(4.0),
            &[("blk.2.attn_q.weight.lora_a", 2, 8, a.clone())],
        );
        assert!(Adapter::from_gguf_path("g", &p).is_err());
        gguf_adapter(
            &p,
            "adapter",
            Some(4.0),
            &[
                ("blk.2.attn_q.weight.lora_a", 2, 8, a),
                ("blk.2.attn_q.weight.lora_b", 6, 3, vec![0.0; 18]),
            ],
        );
        assert!(Adapter::from_gguf_path("g", &p).is_err());
        // truncated
        let bytes = std::fs::read(&p).unwrap();
        std::fs::write(&p, &bytes[..bytes.len() / 2]).unwrap();
        assert!(Adapter::from_gguf_path("g", &p).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn the_llama_row_permutation_is_undone_for_adapters() {
        let (heads, hd, inn, r) = (3usize, 8usize, 5usize, 2usize);
        let out = heads * hd;
        let mut s = 99u64;
        let w = vecr(out * inn, &mut s);
        let plain = pair(out, inn, r, 4.0, 13);
        // permute rows as llama.cpp does: hf row t·half+j sits at gguf row 2j+t
        let permute = |m: &[f32], cols: usize| {
            let half = hd / 2;
            let mut o = vec![0.0; m.len()];
            for h in 0..heads {
                for t in 0..2 {
                    for j in 0..half {
                        let (hf, gg) = (h * hd + t * half + j, h * hd + 2 * j + t);
                        o[gg * cols..(gg + 1) * cols]
                            .copy_from_slice(&m[hf * cols..(hf + 1) * cols]);
                    }
                }
            }
            o
        };
        let key = Key::new(0, Module::Q);
        let mut ad = adapter(
            "p",
            key.clone(),
            LoraPair {
                b: permute(&plain.b, r),
                ..plain.clone()
            },
        );
        assert_ne!(ad.pairs[&key].b, plain.b);
        ad.unpermute_llama(heads, heads).unwrap();
        assert_eq!(ad.pairs[&key].b, plain.b);
        // agrees with the loader's own helper
        assert_eq!(
            unpermute_rows_local(&permute(&w, inn), out, inn, heads),
            crate::loader::unpermute_rows(&permute(&w, inn), out, inn, heads)
        );
        assert!(ad.unpermute_llama(5, 5).is_err());
        assert!(ad.unpermute_llama(0, 0).is_err());
    }

    #[test]
    fn registry_swaps_adapters_under_running_requests() {
        let key = Key::new(0, Module::Q);
        let reg = Registry::new();
        assert!(reg.is_empty());
        assert!(
            reg.insert(adapter("a", key.clone(), pair(4, 6, 2, 4.0, 1)))
                .is_none()
        );
        reg.insert(adapter("b", key.clone(), pair(4, 6, 2, 4.0, 2)));
        assert_eq!(reg.names(), vec!["a", "b"]);
        let x = [0.3f32, -0.2, 0.9, 0.1, -0.7, 0.4];
        let run = |sel: &Selection| {
            let mut y = vec![0.0; 4];
            sel.apply(&key, &x, &mut y).unwrap();
            y
        };
        let held = reg.select(&[("a", 1.0)]).unwrap();
        let before = run(&held);
        // per-request scale
        let half = run(&reg.select(&[("a", 0.5)]).unwrap());
        close(
            &half,
            &before.iter().map(|v| v * 0.5).collect::<Vec<_>>(),
            1e-5,
        );
        assert_eq!(
            reg.select_defaults(&["a"]).unwrap().scale_of("a"),
            Some(1.0)
        );
        // hot swap: replace and remove while `held` is alive
        let old = reg.insert(adapter("a", key.clone(), pair(4, 6, 2, 40.0, 3)));
        assert!(old.is_some());
        assert_eq!(
            run(&held),
            before,
            "a running request keeps the adapter it started with"
        );
        assert_ne!(run(&reg.select(&[("a", 1.0)]).unwrap()), before);
        assert!(reg.remove("a").is_some());
        assert_eq!(run(&held), before);
        assert!(matches!(
            reg.select(&[("a", 1.0)]),
            Err(LoraError::Unknown(_))
        ));
        assert!(reg.select(&[("b", f32::NAN)]).is_err());
        // batch application equals token-by-token
        let sel = reg.select(&[("b", 1.0)]).unwrap();
        let xs: Vec<f32> = x.iter().chain(x.iter().rev()).copied().collect();
        let mut batch = vec![0.0; 8];
        sel.apply_batch(&key, &xs, 2, &mut batch).unwrap();
        let mut one = vec![0.0; 4];
        sel.apply(&key, &xs[6..], &mut one).unwrap();
        close(&batch[4..], &one, 1e-6);
    }

    #[test]
    fn f16_conversion_is_exact_for_every_half_and_rounds_the_rest() {
        for h in 0..=u16::MAX {
            let f = crate::quant::f16_to_f32(h);
            if f.is_nan() {
                assert!(crate::quant::f16_to_f32(f32_to_f16(f)).is_nan());
            } else {
                assert_eq!(f32_to_f16(f), h, "{h:#06x} {f}");
            }
        }
        let mut s = 5u64;
        for _ in 0..2000 {
            let x = rng(&mut s) * 100.0;
            let back = crate::quant::f16_to_f32(f32_to_f16(x));
            assert!((back - x).abs() <= x.abs() * 4.9e-4 + 6e-8, "{x} -> {back}");
        }
        assert_eq!(f32_to_f16(1e10), 0x7c00);
        assert_eq!(f32_to_f16(-1e10), 0xfc00);
        assert_eq!(f32_to_f16(1e-12), 0);
    }

    #[test]
    fn runtime_on_a_quantised_base_adds_exactly_the_adapter_delta() {
        let (out, inn, r) = (10, 64, 4);
        let mut s = 123u64;
        let w = vecr(out * inn, &mut s);
        let q8 = quantise_q8_0(&w);
        let x = vecr(inn, &mut s);
        let key = Key::new(0, Module::Gate);
        let ad = adapter("q", key.clone(), pair(out, inn, r, 8.0, 17));
        let sel = Selection::single(ad.clone(), 1.0);
        let (mut base, mut with) = (vec![0.0; out], vec![0.0; out]);
        linear_with_lora(DType::Q8_0, &q8, out, inn, &x, &mut base, None).unwrap();
        linear_with_lora(
            DType::Q8_0,
            &q8,
            out,
            inn,
            &x,
            &mut with,
            Some((&sel, &key)),
        )
        .unwrap();
        let mut delta = vec![0.0; out];
        ad.pairs[&key].add_delta(&x, 1.0, &mut delta).unwrap();
        for i in 0..out {
            assert!((with[i] - base[i] - delta[i]).abs() < 1e-5, "{i}");
        }
        // and the quantised product is close to the f32 one (activation quantisation noise)
        let exact = naive_matvec(&w, inn, &x);
        let err: f32 = base
            .iter()
            .zip(&exact)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 0.25, "{err}");
        // unsupported formats are an error, not a panic
        assert!(linear_with_lora(DType::Other(99), &q8, out, inn, &x, &mut with, None).is_err());
    }

    #[test]
    fn merging_into_q8_0_stays_within_one_quantisation_step() {
        let (out, inn, r) = (6, 64, 3);
        let mut s = 55u64;
        let w = vecr(out * inn, &mut s);
        let base = quantise_q8_0(&w);
        let key = Key::new(0, Module::Down);
        let sel = Selection::single(adapter("m", key.clone(), pair(out, inn, r, 6.0, 4)), 1.0);
        let exact = merge_dequantised(DType::Q8_0, &base, out, inn, &sel, &key).unwrap();
        let merged = merge_into_q8_0(DType::Q8_0, &base, out, inn, &sel, &key).unwrap();
        assert_eq!(merged.len(), out * inn / 32 * 34);
        let mut back = vec![];
        dequantize(DType::Q8_0, &merged, out * inn, &mut back).unwrap();
        for (blk_i, (e, b)) in exact.chunks(32).zip(back.chunks(32)).enumerate() {
            let step = e.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
            for (x, y) in e.iter().zip(b) {
                assert!(
                    (x - y).abs() <= step * 1.01 + 1e-7,
                    "block {blk_i}: {x} vs {y}"
                );
            }
        }
        // the update itself survived
        assert!(exact.iter().zip(&w).any(|(a, b)| (a - b).abs() > 1e-2));
        assert!(
            merge_into_q8_0(DType::Q8_0, &base, out, inn, &sel, &Key::new(9, Module::Q)).is_ok(),
            "untouched matrices re-quantise unchanged"
        );
        assert!(matches!(
            merge_into_q8_0(DType::Q8_0, &[0; 34], 1, 33, &sel, &key),
            Err(LoraError::Unsupported(_))
        ));
        assert!(merge_into_q8_0(DType::Q2K, &base, out, inn, &sel, &key).is_err());
    }
}
