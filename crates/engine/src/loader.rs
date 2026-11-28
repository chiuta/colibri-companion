//! Builds a [`Model`] from files: safetensors (single file or sharded directory with an index)
//! and GGUF. Every offset and size taken from a file is checked against the real file length
//! before anything is allocated, so a hostile or truncated file is an error, not a crash.
use crate::arch::Spec;
use crate::gguf::{Gguf, TensorInfo};
use crate::model::{Bytes, Experts, Layer, Model, ModelError, QWeight, Weight, Weights};
use crate::moe::{
    ExpertProvider, ExpertWeights, MoeBlock, MoeError, RangeSource, ReadMode, Router, RouterConfig,
    Scoring, StreamedExperts, names,
};
use crate::plan::{Verdict, plan};
use crate::quant::{DType, dequantize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Format(String),
    Missing(String),
    Model(ModelError),
}
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
impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}
impl From<ModelError> for LoadError {
    fn from(e: ModelError) -> Self {
        LoadError::Model(e)
    }
}
fn fmt_err<T>(s: impl Into<String>) -> Result<T, LoadError> {
    Err(LoadError::Format(s.into()))
}

/// A 2-D tensor in its stored form (quantised or half precision), not expanded to f32.
pub struct RawTensor {
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
    pub bytes: Bytes,
}

/// Anything that can hand out a named tensor as f32.
pub trait TensorSource {
    fn read(&self, name: &str) -> Result<Vec<f32>, LoadError>;
    fn has(&self, name: &str) -> bool;
    /// The stored bytes of a 2-D tensor whose type the engine can multiply directly, or `None`
    /// (the default) when the tensor must be read as f32. With `mmap` the bytes may be a window
    /// into a memory-mapped file instead of a copy.
    fn read_raw(&self, _name: &str, _mmap: bool) -> Result<Option<RawTensor>, LoadError> {
        Ok(None)
    }
    /// Every tensor's name, shape and stored type, for memory planning. Empty when unknown.
    fn infos(&self) -> Vec<TensorInfo> {
        vec![]
    }
}

/// A model file that can serve both whole tensors and byte ranges of stacked tensors; what
/// [`load_model`] needs to stream mixture-of-experts weights from disk.
pub trait WeightFile: TensorSource + RangeSource + Send + Sync {}
impl<T: TensorSource + RangeSource + Send + Sync> WeightFile for T {}

/// How a mixture-of-experts model is placed in memory (see [`load_model`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoeOptions {
    /// RAM the model may use. `None`: no limit, every expert is loaded up front. `Some`: the planner
    /// keeps attention, norms, embeddings and shared experts resident and streams routed experts from
    /// disk through a cache sized with what is left (an error if even that does not fit).
    pub ram_budget: Option<u64>,
    /// Force streaming with an expert cache of exactly this many bytes (overrides `ram_budget`).
    pub expert_cache: Option<u64>,
    /// While a layer computes, read ahead the experts the previous token used in the next MoE layer.
    pub prefetch: bool,
}
/// How [`load_dense_with`] should hold the big matrices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadOptions {
    /// Keep quantised / half-precision matrices as stored instead of expanding them to f32.
    pub prefer_quantized: bool,
    /// Memory-map the model files (falls back to reading when a map cannot be made).
    pub mmap: bool,
    /// Worker threads the model may use for one forward pass (0 = every pool thread).
    pub threads: usize,
}

fn read_at(path: &Path, offset: u64, len: u64, file_len: u64) -> Result<Vec<u8>, LoadError> {
    if offset.checked_add(len).is_none_or(|e| e > file_len) {
        return fmt_err("a tensor points past the end of its file");
    }
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut b = vec![0u8; len as usize];
    f.read_exact(&mut b)?;
    Ok(b)
}

/// Map a file read-only. A file truncated by another process while it is mapped can make later
/// reads fault (SIGBUS on Unix, an access violation on Windows), exactly as with any memory-mapped
/// model loader: bounds checks are made against the length at open time and cannot see a later
/// truncation. Do not modify a model file while it is loaded (`mmap: false` copies instead).
fn map_file(path: &Path) -> Option<Arc<memmap2::Mmap>> {
    let f = File::open(path).ok()?;
    // SAFETY: read-only mapping of a file this program never writes to.
    unsafe { memmap2::Mmap::map(&f) }.ok().map(Arc::new)
}

type MapCell = OnceLock<Option<Arc<memmap2::Mmap>>>;

/// Bytes `offset..offset+len` of a file: a mapped window when `mmap` works, otherwise a copy.
fn raw_window(
    path: &Path,
    cell: &MapCell,
    mmap: bool,
    offset: u64,
    len: u64,
    file_len: u64,
) -> Result<Bytes, LoadError> {
    if offset.checked_add(len).is_none_or(|e| e > file_len) {
        return fmt_err("a tensor points past the end of its file");
    }
    if mmap
        && let (Some(m), Ok(o), Ok(l)) = (
            cell.get_or_init(|| map_file(path)),
            usize::try_from(offset),
            usize::try_from(len),
        )
        && let Some(b) = Bytes::mapped(m.clone(), o, l)
    {
        return Ok(b);
    }
    Ok(Bytes::owned(read_at(path, offset, len, file_len)?))
}

// ---------------------------------------------------------------- safetensors

struct StEntry {
    file: usize,
    dtype: DType,
    shape: Vec<u64>,
    elems: u64,
    start: u64,
    len: u64,
}
struct StFile {
    path: PathBuf,
    len: u64,
    map: MapCell,
}
pub struct SafeTensors {
    files: Vec<StFile>,
    map: HashMap<String, StEntry>,
}

impl SafeTensors {
    /// A single `.safetensors` file, or a directory (uses `model.safetensors.index.json` when present).
    pub fn open(path: &Path) -> Result<SafeTensors, LoadError> {
        let mut files = vec![];
        if path.is_dir() {
            let idx = path.join("model.safetensors.index.json");
            if idx.exists() {
                let j: Value = serde_json::from_slice(&std::fs::read(&idx)?)
                    .map_err(|e| LoadError::Format(e.to_string()))?;
                let wm = j
                    .get("weight_map")
                    .and_then(Value::as_object)
                    .ok_or_else(|| LoadError::Format("index has no weight_map".into()))?;
                let mut names: Vec<&str> = wm.values().filter_map(Value::as_str).collect();
                names.sort_unstable();
                names.dedup();
                for n in names {
                    if n.contains("..") || n.contains('/') || n.contains('\\') {
                        return fmt_err("index names a file outside the model folder");
                    }
                    files.push(path.join(n));
                }
            } else {
                for e in std::fs::read_dir(path)? {
                    let p = e?.path();
                    if p.extension().is_some_and(|x| x == "safetensors") {
                        files.push(p);
                    }
                }
                files.sort();
            }
            if files.is_empty() {
                return fmt_err("no .safetensors files found");
            }
        } else {
            files.push(path.to_path_buf());
        }
        let mut map = HashMap::new();
        let mut infos = vec![];
        for (fi, p) in files.iter().enumerate() {
            let len = std::fs::metadata(p)?.len();
            let mut r = BufReader::new(File::open(p)?);
            let mut n = [0u8; 8];
            r.read_exact(&mut n)
                .map_err(|_| LoadError::Format("file too short for a safetensors header".into()))?;
            let n = u64::from_le_bytes(n);
            if n > (1 << 27) || n.checked_add(8).is_none_or(|e| e > len) {
                return fmt_err("implausible safetensors header size");
            }
            let mut h = vec![0u8; n as usize];
            r.read_exact(&mut h)?;
            let j: Value = serde_json::from_slice(&h)
                .map_err(|e| LoadError::Format(format!("bad header: {e}")))?;
            let obj = j
                .as_object()
                .ok_or_else(|| LoadError::Format("header is not an object".into()))?;
            let base = 8 + n;
            for (name, v) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dt = match v.get("dtype").and_then(Value::as_str) {
                    Some("F32") => DType::F32,
                    Some("F16") => DType::F16,
                    Some("BF16") => DType::BF16,
                    other => {
                        return fmt_err(format!("tensor {name}: dtype {other:?} is not supported"));
                    }
                };
                let shape = v
                    .get("shape")
                    .and_then(Value::as_array)
                    .ok_or_else(|| LoadError::Format(format!("{name}: no shape")))?;
                let mut elems = 1u64;
                for d in shape {
                    elems = d
                        .as_u64()
                        .and_then(|d| elems.checked_mul(d))
                        .ok_or_else(|| LoadError::Format(format!("{name}: bad shape")))?;
                }
                let off = v
                    .get("data_offsets")
                    .and_then(Value::as_array)
                    .filter(|a| a.len() == 2)
                    .and_then(|a| Some((a[0].as_u64()?, a[1].as_u64()?)))
                    .ok_or_else(|| LoadError::Format(format!("{name}: bad data_offsets")))?;
                if off.1 < off.0 || Some(off.1 - off.0) != dt.bytes_for(elems) {
                    return fmt_err(format!("{name}: size does not match shape"));
                }
                if base.checked_add(off.1).is_none_or(|e| e > len) {
                    return fmt_err(format!("{name}: data past end of file"));
                }
                let shape = shape.iter().filter_map(Value::as_u64).collect();
                map.insert(
                    name.clone(),
                    StEntry {
                        file: fi,
                        dtype: dt,
                        shape,
                        elems,
                        start: base + off.0,
                        len: off.1 - off.0,
                    },
                );
            }
            infos.push(len);
        }
        Ok(SafeTensors {
            files: files
                .into_iter()
                .zip(infos)
                .map(|(path, len)| StFile {
                    path,
                    len,
                    map: OnceLock::new(),
                })
                .collect(),
            map,
        })
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }
}
impl TensorSource for SafeTensors {
    fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
    fn read(&self, name: &str) -> Result<Vec<f32>, LoadError> {
        let e = self
            .map
            .get(name)
            .ok_or_else(|| LoadError::Missing(name.into()))?;
        let f = &self.files[e.file];
        let b = read_at(&f.path, e.start, e.len, f.len)?;
        let mut out = vec![];
        dequantize(e.dtype, &b, e.elems as usize, &mut out)
            .map_err(|e| LoadError::Format(e.to_string()))?;
        Ok(out)
    }
    fn infos(&self) -> Vec<TensorInfo> {
        self.map
            .iter()
            .map(|(n, e)| TensorInfo {
                name: n.clone(),
                dims: e.shape.clone(),
                dtype: e.dtype,
                offset: e.start,
            })
            .collect()
    }
    /// F16 / BF16 matrices are kept as stored (half the memory); F32 gains nothing and stays dense.
    fn read_raw(&self, name: &str, mmap: bool) -> Result<Option<RawTensor>, LoadError> {
        let Some(e) = self.map.get(name) else {
            return Ok(None);
        };
        if e.shape.len() != 2 || e.dtype == DType::F32 {
            return Ok(None);
        }
        let (Ok(rows), Ok(cols)) = (usize::try_from(e.shape[0]), usize::try_from(e.shape[1]))
        else {
            return Ok(None);
        };
        let f = &self.files[e.file];
        let bytes = raw_window(&f.path, &f.map, mmap, e.start, e.len, f.len)?;
        Ok(Some(RawTensor {
            dtype: e.dtype,
            rows,
            cols,
            bytes,
        }))
    }
}

fn load_err(e: LoadError) -> MoeError {
    MoeError::Load(e.to_string())
}

impl SafeTensors {
    /// Entry, byte offset and byte length of elements `start..start+len`.
    fn range(
        &self,
        name: &str,
        start: usize,
        len: usize,
    ) -> Result<(&StEntry, u64, u64), MoeError> {
        let e = self
            .map
            .get(name)
            .ok_or_else(|| MoeError::Missing(name.into()))?;
        let end = start
            .checked_add(len)
            .filter(|&x| x as u64 <= e.elems)
            .ok_or_else(|| MoeError::Shape(format!("{name}: range past the end")))?;
        let size = e.dtype.bytes_for(1).unwrap_or(0);
        let _ = end;
        Ok((e, e.start + start as u64 * size, len as u64 * size))
    }
}
impl RangeSource for SafeTensors {
    fn read_range(&self, name: &str, start: usize, len: usize) -> Result<Vec<f32>, MoeError> {
        let (e, off, n) = self.range(name, start, len)?;
        let f = &self.files[e.file];
        let b = read_at(&f.path, off, n, f.len).map_err(load_err)?;
        let mut out = vec![];
        dequantize(e.dtype, &b, len, &mut out).map_err(|e| MoeError::Load(e.to_string()))?;
        Ok(out)
    }
    fn read_range_raw(
        &self,
        name: &str,
        start: usize,
        rows: usize,
        cols: usize,
        mmap: bool,
    ) -> Result<Option<RawTensor>, MoeError> {
        let (e, off, n) = self.range(
            name,
            start,
            rows.checked_mul(cols)
                .ok_or_else(|| MoeError::Shape("overflow".into()))?,
        )?;
        if e.dtype == DType::F32 {
            return Ok(None);
        }
        let f = &self.files[e.file];
        Ok(Some(RawTensor {
            dtype: e.dtype,
            rows,
            cols,
            bytes: raw_window(&f.path, &f.map, mmap, off, n, f.len).map_err(load_err)?,
        }))
    }
}

// ----------------------------------------------------------------------- GGUF

pub struct GgufFile {
    path: PathBuf,
    pub gguf: Gguf,
    len: u64,
    map: MapCell,
}
impl GgufFile {
    pub fn open(path: &Path) -> Result<GgufFile, LoadError> {
        let len = std::fs::metadata(path)?.len();
        // the file length is known, so every size the header declares is checked against it
        let gguf = Gguf::parse_with_len(BufReader::new(File::open(path)?), len)
            .map_err(|e| LoadError::Format(e.to_string()))?;
        Ok(GgufFile {
            path: path.to_path_buf(),
            gguf,
            len,
            map: OnceLock::new(),
        })
    }
}
impl TensorSource for GgufFile {
    fn has(&self, name: &str) -> bool {
        self.gguf.tensors.iter().any(|t| t.name == name)
    }
    fn read(&self, name: &str) -> Result<Vec<f32>, LoadError> {
        let t = self
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| LoadError::Missing(name.into()))?;
        let (n, bytes) = match (t.elements(), t.bytes()) {
            (Some(n), Some(b)) => (n, b),
            _ => return fmt_err(format!("{name}: bad size")),
        };
        let off = self
            .gguf
            .data_start
            .checked_add(t.offset)
            .ok_or_else(|| LoadError::Format("offset overflow".into()))?;
        let b = read_at(&self.path, off, bytes, self.len)?;
        let mut out = vec![];
        dequantize(t.dtype, &b, n as usize, &mut out)
            .map_err(|e| LoadError::Format(format!("{name}: {e}")))?;
        Ok(out)
    }
    fn infos(&self) -> Vec<TensorInfo> {
        self.gguf.tensors.clone()
    }
    /// Quantised and half-precision matrices stay in their stored form (GGUF lists `[cols, rows]`).
    fn read_raw(&self, name: &str, mmap: bool) -> Result<Option<RawTensor>, LoadError> {
        let Some(t) = self.gguf.tensors.iter().find(|t| t.name == name) else {
            return Ok(None);
        };
        if t.dims.len() != 2 || t.dtype == DType::F32 || !t.dtype.decodable() {
            return Ok(None);
        }
        let (Ok(cols), Ok(rows)) = (usize::try_from(t.dims[0]), usize::try_from(t.dims[1])) else {
            return Ok(None);
        };
        let (Some(rb), Some(total)) = (t.dtype.bytes_for(cols as u64), t.bytes()) else {
            return Ok(None);
        };
        if (rows as u64).checked_mul(rb) != Some(total) {
            return Ok(None);
        }
        let off = self
            .gguf
            .data_start
            .checked_add(t.offset)
            .ok_or_else(|| LoadError::Format("offset overflow".into()))?;
        let bytes = raw_window(&self.path, &self.map, mmap, off, total, self.len)?;
        Ok(Some(RawTensor {
            dtype: t.dtype,
            rows,
            cols,
            bytes,
        }))
    }
}

impl GgufFile {
    /// The tensor, and the absolute file offset and byte length of elements `start..start+len`.
    /// Quantised data is addressed in whole blocks, so both must be block-aligned.
    fn range(
        &self,
        name: &str,
        start: usize,
        len: usize,
    ) -> Result<(&TensorInfo, u64, u64), MoeError> {
        let t = self
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| MoeError::Missing(name.into()))?;
        let total = t
            .elements()
            .ok_or_else(|| MoeError::Shape(format!("{name}: bad size")))?;
        if start.checked_add(len).is_none_or(|e| e as u64 > total) {
            return Err(MoeError::Shape(format!("{name}: range past the end")));
        }
        let (blk, blk_bytes) = t
            .dtype
            .geometry()
            .ok_or_else(|| MoeError::Load(format!("{name}: unknown tensor type")))?;
        if !start.is_multiple_of(blk) || !len.is_multiple_of(blk) {
            return Err(MoeError::Shape(format!(
                "{name}: range is not aligned to {blk}-element blocks"
            )));
        }
        let off = (start / blk) as u64 * blk_bytes as u64;
        let abs = self
            .gguf
            .data_start
            .checked_add(t.offset)
            .and_then(|o| o.checked_add(off))
            .ok_or_else(|| MoeError::Shape("offset overflow".into()))?;
        Ok((t, abs, (len / blk) as u64 * blk_bytes as u64))
    }
}
/// Reads just the blocks of one expert out of a stacked `[experts, rows, cols]` tensor.
impl RangeSource for GgufFile {
    fn read_range(&self, name: &str, start: usize, len: usize) -> Result<Vec<f32>, MoeError> {
        let (t, abs, n) = self.range(name, start, len)?;
        let b = read_at(&self.path, abs, n, self.len).map_err(load_err)?;
        let mut out = vec![];
        dequantize(t.dtype, &b, len, &mut out)
            .map_err(|e| MoeError::Load(format!("{name}: {e}")))?;
        Ok(out)
    }
    fn read_range_raw(
        &self,
        name: &str,
        start: usize,
        rows: usize,
        cols: usize,
        mmap: bool,
    ) -> Result<Option<RawTensor>, MoeError> {
        let (t, abs, n) = self.range(
            name,
            start,
            rows.checked_mul(cols)
                .ok_or_else(|| MoeError::Shape("overflow".into()))?,
        )?;
        if t.dtype == DType::F32 || !t.dtype.decodable() || t.dtype.bytes_for(cols as u64).is_none()
        {
            return Ok(None);
        }
        Ok(Some(RawTensor {
            dtype: t.dtype,
            rows,
            cols,
            bytes: raw_window(&self.path, &self.map, mmap, abs, n, self.len).map_err(load_err)?,
        }))
    }
}

// --------------------------------------------------------------------- naming

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Naming {
    Hf,
    Gguf,
}

/// Candidate names for a tensor, most common first. Several ecosystems name the same thing differently.
fn candidates(n: Naming, which: &str) -> Vec<String> {
    let v: &[&str] = match (n, which) {
        (Naming::Hf, "embed") => &["model.embed_tokens.weight"],
        (Naming::Hf, "final_norm") => &["model.norm.weight", "model.final_layernorm.weight"],
        (Naming::Hf, "head") => &["lm_head.weight"],
        (Naming::Gguf, "embed") => &["token_embd.weight"],
        (Naming::Gguf, "final_norm") => &["output_norm.weight"],
        (Naming::Gguf, "head") => &["output.weight"],
        _ => &[],
    };
    v.iter().map(|s| s.to_string()).collect()
}

fn layer_names(n: Naming, i: usize, part: &str, post_norms: bool) -> Vec<String> {
    let hf = |suffix: &[&str]| {
        suffix
            .iter()
            .map(|s| format!("model.layers.{i}.{s}.weight"))
            .collect::<Vec<_>>()
    };
    let gg = |s: &str| vec![format!("blk.{i}.{s}.weight")];
    match n {
        Naming::Hf => match part {
            "attn_norm" => hf(&["input_layernorm"]),
            "ffn_norm" => hf(&[if post_norms {
                "pre_feedforward_layernorm"
            } else {
                "post_attention_layernorm"
            }]),
            "post_attn_norm" => hf(&["post_attention_layernorm"]),
            "post_ffn_norm" => hf(&["post_feedforward_layernorm"]),
            "wq" => hf(&["self_attn.q_proj"]),
            "wk" => hf(&["self_attn.k_proj"]),
            "wv" => hf(&["self_attn.v_proj"]),
            "wo" => hf(&["self_attn.o_proj", "self_attn.dense"]),
            "q_norm" => hf(&["self_attn.q_norm"]),
            "k_norm" => hf(&["self_attn.k_norm"]),
            "wqkv" => hf(&["self_attn.qkv_proj"]),
            "w_gate" => hf(&["mlp.gate_proj"]),
            "w_gate_up" => hf(&["mlp.gate_up_proj"]),
            "w_up" => hf(&["mlp.up_proj", "mlp.c_fc", "mlp.fc1", "mlp.dense_h_to_4h"]),
            _ => hf(&[
                "mlp.down_proj",
                "mlp.c_proj",
                "mlp.fc2",
                "mlp.dense_4h_to_h",
            ]),
        },
        Naming::Gguf => match part {
            "attn_norm" => gg("attn_norm"),
            "ffn_norm" => gg("ffn_norm"),
            "post_attn_norm" => gg("post_attention_norm"),
            "post_ffn_norm" => gg("post_ffw_norm"),
            "wq" => gg("attn_q"),
            "wk" => gg("attn_k"),
            "wv" => gg("attn_v"),
            "wo" => gg("attn_output"),
            "q_norm" => gg("attn_q_norm"),
            "k_norm" => gg("attn_k_norm"),
            "wqkv" => gg("attn_qkv"),
            "w_gate" => gg("ffn_gate"),
            "w_gate_up" => vec![],
            "w_up" => gg("ffn_up"),
            _ => gg("ffn_down"),
        },
    }
}
/// The primary name (kept for callers and tests that only need the common case).
pub fn layer_name(n: Naming, i: usize, part: &str) -> String {
    layer_names(n, i, part, false).swap_remove(0)
}

/// Router (gate) tensor names: Mixtral, then Qwen-MoE / OLMoE style, then GGUF.
fn router_names(n: Naming, i: usize) -> Vec<String> {
    match n {
        Naming::Hf => vec![
            format!("model.layers.{i}.block_sparse_moe.gate.weight"),
            format!("model.layers.{i}.mlp.gate.weight"),
        ],
        Naming::Gguf => vec![format!("blk.{i}.ffn_gate_inp.weight")],
    }
}
/// How a layer's router follows from the model's [`Spec`].
fn router_config(
    spec: &Spec,
    hidden: usize,
    experts: usize,
    selection_bias: Option<Vec<f32>>,
) -> RouterConfig {
    RouterConfig {
        hidden,
        experts,
        top_k: spec.experts_used as usize,
        scoring: if spec.moe_sigmoid {
            Scoring::Sigmoid
        } else {
            Scoring::Softmax
        },
        renormalise: spec.norm_topk_prob,
        routed_scale: spec.routed_scale as f32,
        groups: (spec.moe_groups.0 > 1)
            .then_some((spec.moe_groups.0 as usize, spec.moe_groups.1 as usize)),
        selection_bias,
    }
}

/// The per-expert bias that only steers which experts are picked (DeepSeek-V3, GLM-4.5).
fn router_bias_names(n: Naming, i: usize) -> Vec<String> {
    match n {
        Naming::Hf => vec![format!("model.layers.{i}.mlp.gate.e_score_correction_bias")],
        Naming::Gguf => vec![format!("blk.{i}.exp_probs_b.bias")],
    }
}
/// Shared-expert tensors; `part` is `gate`, `up`, `down` or `gate_inp` (the sigmoid gate on their output).
fn shared_names(n: Naming, i: usize, part: &str) -> Vec<String> {
    match n {
        Naming::Hf => ["shared_expert", "shared_experts"]
            .iter()
            .map(|e| match part {
                "gate_inp" => format!("model.layers.{i}.mlp.{e}_gate.weight"),
                p => format!("model.layers.{i}.mlp.{e}.{p}_proj.weight"),
            })
            .collect(),
        Naming::Gguf => vec![match part {
            "gate_inp" => format!("blk.{i}.ffn_gate_inp_shexp.weight"),
            p => format!("blk.{i}.ffn_{p}_shexp.weight"),
        }],
    }
}

fn bias_name(w: &str) -> String {
    format!("{}.bias", w.strip_suffix(".weight").unwrap_or(w))
}
fn found<'a>(src: &dyn TensorSource, names: &'a [String]) -> Option<&'a String> {
    names.iter().find(|n| src.has(n))
}
fn req(src: &dyn TensorSource, names: &[String]) -> Result<Vec<f32>, LoadError> {
    match found(src, names) {
        Some(n) => src.read(n),
        None => Err(LoadError::Missing(
            names.first().cloned().unwrap_or_default(),
        )),
    }
}
/// [hybrid] Load one named matrix the way the dense layers load theirs (stored type or f32).
pub(crate) fn load_weight(
    src: &dyn TensorSource,
    name: &str,
    o: LoadOptions,
) -> Result<Weight, LoadError> {
    req_w(src, &[name.to_string()], o)
}
/// A big matrix: kept as stored when the options ask for it and the source can, otherwise f32.
fn req_w(src: &dyn TensorSource, names: &[String], o: LoadOptions) -> Result<Weight, LoadError> {
    let n = found(src, names)
        .ok_or_else(|| LoadError::Missing(names.first().cloned().unwrap_or_default()))?;
    if o.prefer_quantized
        && let Some(r) = src.read_raw(n, o.mmap)?
    {
        return Ok(Weight::Quant(QWeight::new(
            r.dtype, r.rows, r.cols, r.bytes,
        )?));
    }
    Ok(Weight::Dense(src.read(n)?))
}
fn opt(src: &dyn TensorSource, names: &[String]) -> Result<Option<Vec<f32>>, LoadError> {
    found(src, names).map(|n| src.read(n)).transpose()
}
/// Bias of whichever candidate weight exists.
fn opt_bias(src: &dyn TensorSource, names: &[String]) -> Result<Option<Vec<f32>>, LoadError> {
    match found(src, names) {
        Some(n) => {
            let b = bias_name(n);
            if src.has(&b) {
                src.read(&b).map(Some)
            } else {
                Ok(None)
            }
        }
        None => Ok(None),
    }
}

/// llama.cpp stores Llama's Q/K rows permuted for interleaved RoPE; undo it to our rotate-half layout.
pub fn unpermute_rows(w: &[f32], rows: usize, cols: usize, n_head: usize) -> Vec<f32> {
    permute_gen(w, rows, cols, n_head)
}

/// The row shuffle behind [`unpermute_rows`], for any element type (`row_len` elements per row).
fn permute_gen<T: Copy + Default>(w: &[T], rows: usize, row_len: usize, n_head: usize) -> Vec<T> {
    let hd = rows / n_head;
    let half = hd / 2;
    let mut out = vec![T::default(); w.len()];
    for h in 0..n_head {
        for t in 0..2 {
            for j in 0..half {
                let (dst, src) = (h * hd + t * half + j, h * hd + j * 2 + t);
                out[dst * row_len..(dst + 1) * row_len]
                    .copy_from_slice(&w[src * row_len..(src + 1) * row_len]);
            }
        }
    }
    out
}

/// Split `data` (rows concatenated) into pieces of the given row counts; `None` if sizes disagree.
fn split_rows(data: Vec<f32>, cols: usize, rows: &[usize]) -> Option<Vec<Vec<f32>>> {
    if data.len() != rows.iter().sum::<usize>() * cols {
        return None;
    }
    let mut at = 0;
    Some(
        rows.iter()
            .map(|r| {
                let p = data[at * cols..(at + r) * cols].to_vec();
                at += r;
                p
            })
            .collect(),
    )
}

/// [`split_rows`] for a [`Weight`]; quantised rows are independent, so they split on byte boundaries
/// (and stay memory-mapped when they were).
pub fn split_weight(w: Weight, cols: usize, rows: &[usize]) -> Option<Vec<Weight>> {
    match w {
        Weight::Dense(v) => Some(
            split_rows(v, cols, rows)?
                .into_iter()
                .map(Weight::Dense)
                .collect(),
        ),
        Weight::Quant(q) => {
            if q.cols != cols || q.rows != rows.iter().sum::<usize>() {
                return None;
            }
            let rb = q.row_bytes();
            let mut at = 0;
            rows.iter()
                .map(|&r| {
                    let b = q.bytes.slice(at * rb, r * rb)?;
                    at += r;
                    Some(Weight::Quant(QWeight::new(q.dtype, r, cols, b).ok()?))
                })
                .collect()
        }
    }
}

/// [`unpermute_rows`] for a [`Weight`]: whole quantised rows are moved, nothing is decoded.
pub fn unpermute_weight(w: Weight, rows: usize, cols: usize, n_head: usize) -> Weight {
    match w {
        Weight::Dense(v) => Weight::Dense(unpermute_rows(&v, rows, cols, n_head)),
        Weight::Quant(q) => {
            let rb = q.row_bytes();
            let out = permute_gen(q.bytes.as_slice(), rows, rb, n_head);
            match QWeight::new(q.dtype, q.rows, q.cols, Bytes::owned(out)) {
                Ok(n) => Weight::Quant(n),
                Err(_) => Weight::Quant(q),
            }
        }
    }
}

/// Load a dense model. `llama_permuted` must be true for GGUF files whose architecture is `llama`.
/// Understands separate or fused (Phi-3 style `qkv_proj` / `gate_up_proj`) projections, optional
/// biases, QK-norm and Gemma's extra norms.
pub fn load_dense(
    spec: Spec,
    src: &dyn TensorSource,
    naming: Naming,
    llama_permuted: bool,
) -> Result<Model, LoadError> {
    load_dense_with(spec, src, naming, llama_permuted, LoadOptions::default())
}

/// [`load_dense`] with control over how the big matrices are held (see [`LoadOptions`]). Models that
/// have mixture-of-experts layers need [`load_model`], which also takes ownership of the file so the
/// routed experts can be streamed from it.
pub fn load_dense_with(
    spec: Spec,
    src: &dyn TensorSource,
    naming: Naming,
    llama_permuted: bool,
    opts: LoadOptions,
) -> Result<Model, LoadError> {
    let b = build_weights(&spec, src, naming, llama_permuted, opts)?;
    if !b.moe_layers.is_empty() {
        return fmt_err("this model has mixture-of-experts layers: open it with load_model");
    }
    Ok(Model::new(spec, b.w)?)
}

struct Built {
    w: Weights,
    moe_layers: Vec<usize>,
    mixtral: bool,
}

/// One transformer layer's tensors (everything but the routed experts), read from `src`. Returns the layer
/// and whether it is a mixture-of-experts layer / a Mixtral-style one.
pub(crate) fn build_layer(
    spec: &Spec,
    src: &dyn TensorSource,
    naming: Naming,
    llama_permuted: bool,
    opts: LoadOptions,
    i: usize,
) -> Result<(Layer, bool, bool), LoadError> {
    let (h, hd, ff) = (
        spec.hidden as usize,
        spec.head_dim as usize,
        spec.ffn as usize,
    );
    let (qd, kd) = (
        (spec.heads * spec.head_dim) as usize,
        (spec.kv_heads * spec.head_dim) as usize,
    );
    let fix = |w: Weight, heads: u64| {
        if llama_permuted && naming == Naming::Gguf {
            let rows = heads as usize * hd;
            if w.len() == rows * h && rows.is_multiple_of(2 * heads as usize) {
                unpermute_weight(w, rows, h, heads as usize)
            } else {
                w
            }
        } else {
            w
        }
    };
    let bad = |what: &str, name: &str| {
        LoadError::Format(format!("{name}: {what} does not match the architecture"))
    };
    let (mut is_moe, mut is_mixtral) = (false, false);
    let nm = |p: &str| layer_names(naming, i, p, spec.post_norms);
    let attn_norm_names = nm("attn_norm");
    let mut l = Layer {
        attn_norm: req(src, &attn_norm_names)?,
        attn_norm_b: opt(
            src,
            &attn_norm_names
                .iter()
                .map(|n| bias_name(n))
                .collect::<Vec<_>>(),
        )?,
        ..Default::default()
    };
    // attention projections
    if let Some(fused) = found(src, &nm("wqkv")) {
        let parts = split_weight(
            req_w(src, std::slice::from_ref(fused), opts)?,
            h,
            &[qd, kd, kd],
        )
        .ok_or_else(|| bad("fused qkv shape", fused))?;
        let [q, k, v]: [Weight; 3] = parts.try_into().map_err(|_| bad("fused qkv", fused))?;
        (l.wq, l.wk, l.wv) = (q, k, v);
        if let Some(b) = opt_bias(src, &nm("wqkv"))? {
            let [bq, bk, bv]: [Vec<f32>; 3] = split_rows(b, 1, &[qd, kd, kd])
                .ok_or_else(|| bad("fused qkv bias", fused))?
                .try_into()
                .map_err(|_| bad("fused qkv bias", fused))?;
            (l.bq, l.bk, l.bv) = (Some(bq), Some(bk), Some(bv));
        }
    } else {
        l.wq = fix(req_w(src, &nm("wq"), opts)?, spec.heads);
        l.wk = fix(req_w(src, &nm("wk"), opts)?, spec.kv_heads);
        l.wv = req_w(src, &nm("wv"), opts)?;
        (l.bq, l.bk, l.bv) = (
            opt_bias(src, &nm("wq"))?,
            opt_bias(src, &nm("wk"))?,
            opt_bias(src, &nm("wv"))?,
        );
    }
    l.wo = req_w(src, &nm("wo"), opts)?;
    l.bo = opt_bias(src, &nm("wo"))?;
    if spec.qk_norm {
        l.q_norm = Some(req(src, &nm("q_norm"))?);
        l.k_norm = Some(req(src, &nm("k_norm"))?);
    }
    if spec.post_norms {
        l.post_attn_norm = Some(req(src, &nm("post_attn_norm"))?);
        l.post_ffn_norm = Some(req(src, &nm("post_ffn_norm"))?);
    }
    // feed-forward norm: parallel-residual models with a single norm reuse the attention norm
    let fnames = nm("ffn_norm");
    if found(src, &fnames).is_none() && spec.parallel_residual {
        l.ffn_norm = l.attn_norm.clone();
        l.ffn_norm_b = l.attn_norm_b.clone();
    } else {
        l.ffn_norm = req(src, &fnames)?;
        l.ffn_norm_b = opt(
            src,
            &fnames.iter().map(|n| bias_name(n)).collect::<Vec<_>>(),
        )?;
    }
    // feed-forward: routed experts when the layer has a router, otherwise the dense MLP
    if let Some(rn) = found(src, &router_names(naming, i)) {
        let experts = if spec.experts > 0 {
            spec.experts as usize
        } else {
            0
        };
        let rw = src.read(rn)?;
        let experts = if experts > 0 {
            experts
        } else {
            rw.len() / h.max(1)
        };
        let cfg = router_config(spec, h, experts, opt(src, &router_bias_names(naming, i))?);
        let router = Router::new(cfg, rw).map_err(|e| LoadError::Format(format!("{rn}: {e}")))?;
        let mut shared = vec![];
        let gate_names = shared_names(naming, i, "gate");
        if found(src, &gate_names).is_some() {
            shared.push(ExpertWeights {
                gate: req_w(src, &gate_names, opts)?,
                up: req_w(src, &shared_names(naming, i, "up"), opts)?,
                down: req_w(src, &shared_names(naming, i, "down"), opts)?,
            });
        }
        let mut block = MoeBlock::new(i, router, shared)
            .map_err(|e| LoadError::Format(format!("layer {i}: {e}")))?;
        if let Some(g) = opt(src, &shared_names(naming, i, "gate_inp"))? {
            block = block
                .with_shared_gate(g)
                .map_err(|e| LoadError::Format(format!("layer {i}: {e}")))?;
        }
        if rn.contains("block_sparse_moe") {
            is_mixtral = true;
        }
        l.moe = Some(Arc::new(block));
        is_moe = true;
    } else if spec.gated_mlp {
        if found(src, &nm("w_gate")).is_some() {
            l.w_gate = req_w(src, &nm("w_gate"), opts)?;
            l.w_up = req_w(src, &nm("w_up"), opts)?;
            (l.b_gate, l.b_up) = (opt_bias(src, &nm("w_gate"))?, opt_bias(src, &nm("w_up"))?);
        } else {
            // fused gate+up: HF `gate_up_proj`, or GGUF `ffn_up` holding both (gate rows first)
            let names = if found(src, &nm("w_gate_up")).is_some() {
                nm("w_gate_up")
            } else {
                nm("w_up")
            };
            let fused = req_w(src, &names, opts)?;
            let [g, u]: [Weight; 2] = split_weight(fused, h, &[ff, ff])
                .ok_or_else(|| LoadError::Missing(nm("w_gate")[0].clone()))?
                .try_into()
                .map_err(|_| bad("fused gate/up", &names[0]))?;
            (l.w_gate, l.w_up) = (g, u);
        }
    } else {
        l.w_up = req_w(src, &nm("w_up"), opts)?;
        l.b_up = opt_bias(src, &nm("w_up"))?;
    }
    if l.moe.is_none() {
        l.w_down = req_w(src, &nm("w_down"), opts)?;
        l.b_down = opt_bias(src, &nm("w_down"))?;
    }
    Ok((l, is_moe, is_mixtral))
}

/// Everything outside the layers (embedding, final norm, output head) around the given `layers`.
pub(crate) fn build_outer(
    spec: &Spec,
    src: &dyn TensorSource,
    naming: Naming,
    opts: LoadOptions,
    layers: Vec<Layer>,
) -> Result<Weights, LoadError> {
    let embed = req_w(src, &candidates(naming, "embed"), opts)?;
    let head_names = candidates(naming, "head");
    let fnorm = candidates(naming, "final_norm");
    let has_head = found(src, &head_names).is_some() && !spec.tied_embeddings;
    Ok(Weights {
        embed,
        layers,
        final_norm: req(src, &fnorm)?,
        final_norm_b: opt(src, &fnorm.iter().map(|n| bias_name(n)).collect::<Vec<_>>())?,
        lm_head: if has_head {
            Some(req_w(src, &head_names, opts)?)
        } else {
            None
        },
        lm_head_b: if has_head {
            opt(
                src,
                &head_names.iter().map(|n| bias_name(n)).collect::<Vec<_>>(),
            )?
        } else {
            None
        },
        experts: None,
    })
}

fn build_weights(
    spec: &Spec,
    src: &dyn TensorSource,
    naming: Naming,
    llama_permuted: bool,
    opts: LoadOptions,
) -> Result<Built, LoadError> {
    let (mut moe_layers, mut mixtral) = (vec![], false);
    let mut layers = Vec::with_capacity(spec.layers as usize);
    for i in 0..spec.layers as usize {
        let (l, is_moe, is_mixtral) = build_layer(spec, src, naming, llama_permuted, opts, i)?;
        if is_moe {
            moe_layers.push(i);
        }
        mixtral |= is_mixtral;
        layers.push(l);
    }
    let w = build_outer(spec, src, naming, opts, layers)?;
    Ok(Built {
        w,
        moe_layers,
        mixtral,
    })
}

/// Load a model that may have mixture-of-experts layers (Mixtral, Qwen-MoE, GGUF stacked experts).
/// Dense models load exactly as with [`load_dense_with`]. For MoE models the file stays open inside the
/// model: routed experts are either all read now (`moe.ram_budget` unset, or the model fits) or streamed
/// from disk on demand through a byte-budgeted cache sized by the memory planner.
pub fn load_model(
    spec: Spec,
    src: Arc<dyn WeightFile>,
    naming: Naming,
    llama_permuted: bool,
    opts: LoadOptions,
    moe: MoeOptions,
) -> Result<Model, LoadError> {
    // [arch registry] LongRoPE keeps its factor tables in GGUF tensors: read them now
    let mut spec = spec;
    crate::arch::load_rope_factors(&mut spec, &*src).map_err(LoadError::Format)?;
    let mut b = build_weights(&spec, &*src, naming, llama_permuted, opts)?;
    if !b.moe_layers.is_empty() {
        let provider = make_provider(&spec, &src, naming, opts, moe, &b)?;
        b.w.experts = Some(Experts {
            provider,
            prefetch: moe.prefetch,
        });
    }
    let mut m = Model::new(spec, b.w)?;
    m.set_threads(opts.threads);
    Ok(m)
}

fn make_provider(
    spec: &Spec,
    src: &Arc<dyn WeightFile>,
    naming: Naming,
    opts: LoadOptions,
    moe: MoeOptions,
    b: &Built,
) -> Result<Arc<dyn ExpertProvider>, LoadError> {
    let (h, f) = (spec.hidden as usize, spec.expert_ffn() as usize);
    let n_exp =
        b.w.layers
            .iter()
            .find_map(|l| l.moe.as_ref())
            .map_or(0, |m| m.router.config().experts);
    let mode = ReadMode {
        quantized: opts.prefer_quantized,
        mmap: opts.mmap,
    };
    // Budget for the expert cache. Without a limit everything is preloaded (`u64::MAX`, never evicts).
    let (stream, budget) = match (moe.expert_cache, moe.ram_budget) {
        (Some(c), _) => (true, c),
        (None, Some(ram)) => {
            let mut infos = src.infos();
            if infos.is_empty() {
                (false, u64::MAX)
            } else {
                // matrices not kept as stored are expanded to f32, which is what the planner must count
                if !opts.prefer_quantized {
                    infos.iter_mut().for_each(|t| t.dtype = DType::F32);
                }
                match plan(&infos, ram, spec.kv_bytes(spec.context.min(8192))) {
                    p if p.fits == Verdict::AllInMemory => (false, u64::MAX),
                    p if p.fits == Verdict::Streaming => (true, p.expert_cache),
                    p => {
                        return fmt_err(format!(
                            "the model does not fit in {ram} bytes of RAM even with the experts streamed from disk (the always-resident part and the KV cache need {} bytes more)",
                            match p.fits {
                                Verdict::TooSmall { missing } => missing,
                                _ => 0,
                            }
                        ));
                    }
                }
            }
        }
        (None, None) => (false, u64::MAX),
    };
    let s = match naming {
        Naming::Gguf => {
            let r: Arc<dyn RangeSource> = src.clone();
            StreamedExperts::stacked_with(r, names::gguf_stacked, h, f, budget, mode)
        }
        Naming::Hf => {
            let t: Arc<dyn TensorSource + Send + Sync> = src.clone();
            if b.mixtral {
                StreamedExperts::per_expert_with(t, names::mixtral, h, f, budget, mode)
            } else {
                StreamedExperts::per_expert_with(t, names::hf_experts, h, f, budget, mode)
            }
        }
    };
    if !stream {
        s.preload(b.moe_layers.iter().copied(), n_exp)
            .map_err(|e| LoadError::Format(e.to_string()))?;
    }
    Ok(Arc::new(s))
}

#[cfg(test)]
mod tests {
    #[test]
    fn router_follows_the_spec_sigmoid_bias_and_groups() {
        let mut sp = Spec::dense("glm4moe", 2, 8, 2, 2, 4, 16, 10, 64);
        sp.experts = 8;
        sp.experts_used = 2;
        sp.norm_topk_prob = true;
        sp.routed_scale = 1.8;
        let plain = super::router_config(&sp, 8, 8, None);
        assert_eq!(plain.scoring, crate::moe::Scoring::Softmax);
        assert!(plain.groups.is_none() && plain.selection_bias.is_none());
        sp.moe_sigmoid = true;
        sp.moe_groups = (4, 2);
        let bias: Vec<f32> = (0..8).map(|i| if i == 5 { 10.0 } else { 0.0 }).collect();
        let c = super::router_config(&sp, 8, 8, Some(bias));
        assert_eq!(c.scoring, crate::moe::Scoring::Sigmoid);
        assert_eq!((c.groups, c.routed_scale, c.top_k), (Some((4, 2)), 1.8, 2));
        // the bias steers selection (expert 5 wins) but is not part of the weights
        let router = crate::moe::Router::new(c, vec![0.1; 64]).unwrap();
        let r = router.route(&[1.0; 8]).unwrap();
        assert_eq!(r[0].0, 5);
        let sum: f32 = r.iter().map(|p| p.1).sum();
        assert!((sum - 1.8).abs() < 1e-4, "{sum}");
    }

    use super::*;
    use crate::arch::{Act, NormKind};
    use crate::model::tests::{base_spec, model_of, requantise, tiny};
    use std::sync::atomic::{AtomicU32, Ordering};

    fn tmpdir() -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "engine-loader-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    /// (hf name, gguf name, shape, data)
    fn all(m: &Model) -> Vec<(String, String, Vec<usize>, Vec<f32>)> {
        let s = &m.spec;
        let (h, qd, kd, ff, v) = (
            s.hidden as usize,
            (s.heads * s.head_dim) as usize,
            (s.kv_heads * s.head_dim) as usize,
            s.ffn as usize,
            s.vocab as usize,
        );
        let mut o = vec![
            (
                "model.embed_tokens.weight".into(),
                "token_embd.weight".into(),
                vec![v, h],
                m.w.embed.to_f32(),
            ),
            (
                "model.norm.weight".into(),
                "output_norm.weight".into(),
                vec![h],
                m.w.final_norm.clone(),
            ),
        ];
        if let Some(l) = &m.w.lm_head {
            o.push((
                "lm_head.weight".into(),
                "output.weight".into(),
                vec![v, h],
                l.to_f32(),
            ));
        }
        for (i, l) in m.w.layers.iter().enumerate() {
            for (p, shape, d) in [
                ("attn_norm", vec![h], l.attn_norm.clone()),
                ("wq", vec![qd, h], l.wq.to_f32()),
                ("wk", vec![kd, h], l.wk.to_f32()),
                ("wv", vec![kd, h], l.wv.to_f32()),
                ("wo", vec![h, qd], l.wo.to_f32()),
                ("ffn_norm", vec![h], l.ffn_norm.clone()),
                ("w_gate", vec![ff, h], l.w_gate.to_f32()),
                ("w_up", vec![ff, h], l.w_up.to_f32()),
                ("w_down", vec![h, ff], l.w_down.to_f32()),
            ] {
                o.push((
                    layer_name(Naming::Hf, i, p),
                    layer_name(Naming::Gguf, i, p),
                    shape,
                    d,
                ));
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
        for t in ts {
            f.extend(&t.2);
        }
        std::fs::write(path, f).unwrap();
    }
    fn f32b(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn bf16b(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
    fn run(m: &Model) -> Vec<f32> {
        let mut st = m.new_state();
        m.forward(&mut st, 3).unwrap();
        m.forward(&mut st, 11).unwrap()
    }

    #[test]
    fn sharded_safetensors_with_index_reproduces_the_model() {
        let m = tiny(4, 2, false);
        let d = tmpdir();
        let t = all(&m);
        let (a, b) = t.split_at(t.len() / 2);
        let mk = |s: &[(String, String, Vec<usize>, Vec<f32>)]| {
            s.iter()
                .map(|(n, _, sh, d)| (n.clone(), sh.clone(), f32b(d), "F32"))
                .collect::<Vec<_>>()
        };
        write_st(&d.join("model-00001-of-00002.safetensors"), &mk(a));
        write_st(&d.join("model-00002-of-00002.safetensors"), &mk(b));
        let wm: serde_json::Map<String, Value> = a
            .iter()
            .map(|x| (x.0.clone(), "model-00001-of-00002.safetensors".into()))
            .chain(
                b.iter()
                    .map(|x| (x.0.clone(), "model-00002-of-00002.safetensors".into())),
            )
            .collect();
        std::fs::write(
            d.join("model.safetensors.index.json"),
            serde_json::json!({"weight_map": wm}).to_string(),
        )
        .unwrap();
        let src = SafeTensors::open(&d).unwrap();
        let loaded = load_dense(m.spec.clone(), &src, Naming::Hf, false).unwrap();
        assert_eq!(run(&m), run(&loaded));
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn bf16_files_load_with_bf16_precision() {
        let m = tiny(4, 4, true);
        let d = tmpdir();
        let ts: Vec<_> = all(&m)
            .into_iter()
            .map(|(n, _, sh, d)| (n, sh, bf16b(&d), "BF16"))
            .collect();
        write_st(&d.join("m.safetensors"), &ts);
        let loaded = load_dense(
            m.spec.clone(),
            &SafeTensors::open(&d.join("m.safetensors")).unwrap(),
            Naming::Hf,
            false,
        )
        .unwrap();
        let (a, b) = (run(&m), run(&loaded));
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
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
        std::fs::write(
            d.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"../evil.safetensors"}}"#,
        )
        .unwrap();
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
        for h in 0..heads {
            for t in 0..2 {
                for j in 0..half {
                    let (hf, gg) = (h * hd + t * half + j, h * hd + j * 2 + t);
                    perm[gg * cols..(gg + 1) * cols]
                        .copy_from_slice(&orig[hf * cols..(hf + 1) * cols]);
                }
            }
        }
        assert_ne!(perm, orig);
        assert_eq!(unpermute_rows(&perm, rows, cols, heads), orig);
    }

    #[test]
    fn gguf_f32_file_with_permuted_llama_weights_reproduces_the_model() {
        let m = tiny(4, 2, false);
        let d = tmpdir();
        let p = d.join("tiny.gguf");
        let permute = |w: &[f32], heads: usize, cols: usize| {
            let hd = 8;
            let half = 4;
            let mut o = vec![0.0; w.len()];
            for h in 0..heads {
                for t in 0..2 {
                    for j in 0..half {
                        let (hf, gg) = (h * hd + t * half + j, h * hd + j * 2 + t);
                        o[gg * cols..(gg + 1) * cols]
                            .copy_from_slice(&w[hf * cols..(hf + 1) * cols]);
                    }
                }
            }
            o
        };
        let mut v = b"GGUF".to_vec();
        let ts = all(&m);
        v.extend(3u32.to_le_bytes());
        v.extend((ts.len() as u64).to_le_bytes());
        v.extend(1u64.to_le_bytes());
        crate::gguf::tests::put_str(&mut v, "general.architecture");
        v.extend(8u32.to_le_bytes());
        crate::gguf::tests::put_str(&mut v, "llama");
        let mut data = vec![];
        let mut payload = vec![];
        for (_, g, sh, d) in &ts {
            let d = if g.contains("attn_q") {
                permute(d, 4, 16)
            } else if g.contains("attn_k") {
                permute(d, 2, 16)
            } else {
                d.clone()
            };
            while data.len() % 32 != 0 {
                data.push(0);
            }
            crate::gguf::tests::put_str(&mut payload, g);
            payload.extend((sh.len() as u32).to_le_bytes());
            for x in sh.iter().rev() {
                payload.extend((*x as u64).to_le_bytes());
            }
            payload.extend(0u32.to_le_bytes());
            payload.extend((data.len() as u64).to_le_bytes());
            data.extend(f32b(&d));
        }
        v.extend(payload);
        while !v.len().is_multiple_of(32) {
            v.push(0);
        }
        v.extend(data);
        std::fs::write(&p, v).unwrap();
        let f = GgufFile::open(&p).unwrap();
        let loaded = load_dense(m.spec.clone(), &f, Naming::Gguf, true).unwrap();
        assert_eq!(run(&m), run(&loaded));
        // truncated copy: a clean error
        let b = std::fs::read(&p).unwrap();
        std::fs::write(&p, &b[..b.len() - 100]).unwrap();
        assert!(
            load_dense(
                m.spec.clone(),
                &GgufFile::open(&p).unwrap(),
                Naming::Gguf,
                true
            )
            .is_err()
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn missing_tensor_is_named_in_the_error() {
        let d = tmpdir();
        write_st(
            &d.join("m.safetensors"),
            &[(
                "model.embed_tokens.weight".into(),
                vec![1],
                f32b(&[1.0]),
                "F32",
            )],
        );
        let m = tiny(4, 2, false);
        let e = load_dense(
            m.spec.clone(),
            &SafeTensors::open(&d).unwrap(),
            Naming::Hf,
            false,
        )
        .err()
        .unwrap();
        assert!(
            e.to_string()
                .contains("model.layers.0.input_layernorm.weight"),
            "{e}"
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    // ---------------------------------------------------------- variant tensors, in-memory source

    struct MapSource(HashMap<String, Vec<f32>>);
    impl TensorSource for MapSource {
        fn has(&self, n: &str) -> bool {
            self.0.contains_key(n)
        }
        fn read(&self, n: &str) -> Result<Vec<f32>, LoadError> {
            self.0
                .get(n)
                .cloned()
                .ok_or_else(|| LoadError::Missing(n.into()))
        }
    }

    /// Every tensor of `m` under the names of `naming`. `fused` packs qkv and gate/up the Phi-3 way.
    fn ins(t: &mut HashMap<String, Vec<f32>>, n: String, v: &[f32]) {
        t.insert(n, v.to_vec());
    }
    fn ins_o(t: &mut HashMap<String, Vec<f32>>, n: String, v: &Option<Vec<f32>>) {
        if let Some(v) = v {
            t.insert(n, v.clone());
        }
    }
    fn export(m: &Model, naming: Naming, fused: bool) -> MapSource {
        let s = &m.spec;
        let mut t: HashMap<String, Vec<f32>> = HashMap::new();
        let nm = |i: usize, p: &str| layer_names(naming, i, p, s.post_norms).swap_remove(0);
        let bias = |n: &str| bias_name(n);
        ins(
            &mut t,
            candidates(naming, "embed").swap_remove(0),
            &m.w.embed.to_f32(),
        );
        let fnm = candidates(naming, "final_norm").swap_remove(0);
        ins(&mut t, fnm.clone(), &m.w.final_norm);
        if let Some(b) = &m.w.final_norm_b {
            ins(&mut t, bias(&fnm), b);
        }
        if let Some(l) = &m.w.lm_head {
            let n = candidates(naming, "head").swap_remove(0);
            ins(&mut t, n.clone(), &l.to_f32());
            if let Some(b) = &m.w.lm_head_b {
                ins(&mut t, bias(&n), b);
            }
        }
        for (i, l) in m.w.layers.iter().enumerate() {
            ins(&mut t, nm(i, "attn_norm"), &l.attn_norm);
            ins_o(&mut t, bias(&nm(i, "attn_norm")), &l.attn_norm_b);
            if fused {
                let cat = [l.wq.to_f32(), l.wk.to_f32(), l.wv.to_f32()].concat();
                ins(&mut t, nm(i, "wqkv"), &cat);
            } else {
                ins(&mut t, nm(i, "wq"), &l.wq.to_f32());
                ins(&mut t, nm(i, "wk"), &l.wk.to_f32());
                ins(&mut t, nm(i, "wv"), &l.wv.to_f32());
                ins_o(&mut t, bias(&nm(i, "wq")), &l.bq);
                ins_o(&mut t, bias(&nm(i, "wk")), &l.bk);
                ins_o(&mut t, bias(&nm(i, "wv")), &l.bv);
            }
            ins(&mut t, nm(i, "wo"), &l.wo.to_f32());
            ins_o(&mut t, bias(&nm(i, "wo")), &l.bo);
            ins_o(&mut t, nm(i, "q_norm"), &l.q_norm);
            ins_o(&mut t, nm(i, "k_norm"), &l.k_norm);
            ins_o(&mut t, nm(i, "post_attn_norm"), &l.post_attn_norm);
            ins_o(&mut t, nm(i, "post_ffn_norm"), &l.post_ffn_norm);
            ins(&mut t, nm(i, "ffn_norm"), &l.ffn_norm);
            ins_o(&mut t, bias(&nm(i, "ffn_norm")), &l.ffn_norm_b);
            if s.gated_mlp && fused {
                let cat = [l.w_gate.to_f32(), l.w_up.to_f32()].concat();
                ins(
                    &mut t,
                    if naming == Naming::Hf {
                        nm(i, "w_gate_up")
                    } else {
                        nm(i, "w_up")
                    },
                    &cat,
                );
            } else {
                if s.gated_mlp {
                    ins(&mut t, nm(i, "w_gate"), &l.w_gate.to_f32());
                    ins_o(&mut t, bias(&nm(i, "w_gate")), &l.b_gate);
                }
                ins(&mut t, nm(i, "w_up"), &l.w_up.to_f32());
                ins_o(&mut t, bias(&nm(i, "w_up")), &l.b_up);
            }
            ins(&mut t, nm(i, "w_down"), &l.w_down.to_f32());
            ins_o(&mut t, bias(&nm(i, "w_down")), &l.b_down);
        }
        MapSource(t)
    }
    fn same(a: &Model, b: &Model) {
        assert_eq!(run(a), run(b));
    }

    fn gemma_like() -> Model {
        let mut sp = base_spec(4, 2, false);
        sp.norm_offset = true;
        sp.act = Act::GeluTanh;
        sp.embed_scale = 4.0;
        sp.post_norms = true;
        sp.qk_norm = true;
        sp.attn_softcap = 5.0;
        sp.final_softcap = 3.0;
        sp.sliding_window = 3;
        sp.sliding_layers = vec![true, false, true];
        model_of(sp, true)
    }

    #[test]
    fn gemma_biases_qk_norm_and_post_norms_roundtrip_with_hf_and_gguf_names() {
        let m = gemma_like();
        for naming in [Naming::Hf, Naming::Gguf] {
            let src = export(&m, naming, false);
            let loaded = load_dense(m.spec.clone(), &src, naming, false).unwrap();
            same(&m, &loaded);
            assert!(
                loaded.w.layers[0].post_attn_norm.is_some()
                    && loaded.w.layers[0].q_norm.is_some()
                    && loaded.w.layers[0].bq.is_some()
            );
        }
        // Gemma's pre-feedforward norm has its own HF name; the Llama name is the post-attention one
        assert_eq!(
            layer_names(Naming::Hf, 2, "ffn_norm", true)[0],
            "model.layers.2.pre_feedforward_layernorm.weight"
        );
        assert_eq!(
            layer_names(Naming::Hf, 2, "ffn_norm", false)[0],
            "model.layers.2.post_attention_layernorm.weight"
        );
        assert_eq!(
            layer_names(Naming::Hf, 2, "post_attn_norm", true)[0],
            "model.layers.2.post_attention_layernorm.weight"
        );
        assert_eq!(
            layer_names(Naming::Gguf, 2, "post_ffn_norm", true)[0],
            "blk.2.post_ffw_norm.weight"
        );
        assert_eq!(
            layer_names(Naming::Gguf, 2, "q_norm", true)[0],
            "blk.2.attn_q_norm.weight"
        );
    }

    #[test]
    fn a_missing_required_extra_tensor_is_named() {
        let m = gemma_like();
        let mut src = export(&m, Naming::Hf, false);
        src.0.remove("model.layers.1.self_attn.k_norm.weight");
        let e = load_dense(m.spec.clone(), &src, Naming::Hf, false)
            .err()
            .unwrap();
        assert!(
            e.to_string()
                .contains("model.layers.1.self_attn.k_norm.weight"),
            "{e}"
        );
    }

    #[test]
    fn phi3_fused_qkv_and_gate_up_are_split_at_load_time() {
        let m = tiny(4, 2, false);
        for naming in [Naming::Hf, Naming::Gguf] {
            let src = export(&m, naming, true);
            if naming == Naming::Hf {
                assert!(
                    src.has("model.layers.0.self_attn.qkv_proj.weight")
                        && src.has("model.layers.0.mlp.gate_up_proj.weight")
                );
            }
            let loaded = load_dense(m.spec.clone(), &src, naming, false).unwrap();
            same(&m, &loaded);
        }
        // fused with a wrong row count is an error, not a mis-split
        let mut src = export(&m, Naming::Hf, true);
        src.0
            .get_mut("model.layers.0.self_attn.qkv_proj.weight")
            .unwrap()
            .pop();
        assert!(load_dense(m.spec.clone(), &src, Naming::Hf, false).is_err());
    }

    #[test]
    fn layer_norm_plain_mlp_models_use_their_own_tensor_names() {
        let mut sp = base_spec(4, 2, true);
        sp.norm = NormKind::Layer;
        sp.gated_mlp = false;
        sp.act = Act::GeluTanh;
        sp.qkv_bias = true;
        let m = model_of(sp, true);
        let mut src = export(&m, Naming::Hf, false);
        // StarCoder2 calls the MLP matrices c_fc / c_proj
        for (from, to) in [("mlp.up_proj", "mlp.c_fc"), ("mlp.down_proj", "mlp.c_proj")] {
            let keys: Vec<String> = src.0.keys().filter(|k| k.contains(from)).cloned().collect();
            for k in keys {
                let v = src.0.remove(&k).unwrap();
                src.0.insert(k.replace(from, to), v);
            }
        }
        let loaded = load_dense(m.spec.clone(), &src, Naming::Hf, false).unwrap();
        same(&m, &loaded);
        assert!(loaded.w.final_norm_b.is_some() && loaded.w.layers[0].b_up.is_some());
    }

    #[test]
    fn parallel_residual_with_a_single_norm_reuses_it() {
        let mut sp = base_spec(4, 4, false);
        sp.norm = NormKind::Layer;
        sp.gated_mlp = false;
        sp.act = Act::Gelu;
        sp.parallel_residual = true;
        let mut m = model_of(sp, true);
        for l in &mut m.w.layers {
            l.ffn_norm = l.attn_norm.clone();
            l.ffn_norm_b = l.attn_norm_b.clone();
        }
        let mut src = export(&m, Naming::Hf, false);
        src.0.retain(|k, _| !k.contains("post_attention_layernorm"));
        let loaded = load_dense(m.spec.clone(), &src, Naming::Hf, false).unwrap();
        same(&m, &loaded);
    }

    // ---------------------------------------------------- quantised loading, memory mapping

    /// A GGUF file with the given `(name, [rows, cols] or [len], gguf type id, bytes)` tensors.
    fn write_gguf(path: &Path, arch: &str, ts: &[(String, Vec<usize>, u32, Vec<u8>)]) {
        let mut v = b"GGUF".to_vec();
        v.extend(3u32.to_le_bytes());
        v.extend((ts.len() as u64).to_le_bytes());
        v.extend(1u64.to_le_bytes());
        crate::gguf::tests::put_str(&mut v, "general.architecture");
        v.extend(8u32.to_le_bytes());
        crate::gguf::tests::put_str(&mut v, arch);
        let (mut data, mut info) = (vec![], vec![]);
        for (name, shape, ty, bytes) in ts {
            while data.len() % 32 != 0 {
                data.push(0);
            }
            crate::gguf::tests::put_str(&mut info, name);
            info.extend((shape.len() as u32).to_le_bytes());
            for d in shape.iter().rev() {
                info.extend((*d as u64).to_le_bytes());
            }
            info.extend(ty.to_le_bytes());
            info.extend((data.len() as u64).to_le_bytes());
            data.extend(bytes);
        }
        v.extend(info);
        while !v.len().is_multiple_of(32) {
            v.push(0);
        }
        v.extend(data);
        std::fs::write(path, v).unwrap();
    }
    fn permute_rows_f32(w: &[f32], rows: usize, cols: usize, heads: usize) -> Vec<f32> {
        let (hd, half) = (rows / heads, rows / heads / 2);
        let mut o = vec![0.0; w.len()];
        for h in 0..heads {
            for t in 0..2 {
                for j in 0..half {
                    let (hf, gg) = (h * hd + t * half + j, h * hd + j * 2 + t);
                    o[gg * cols..(gg + 1) * cols].copy_from_slice(&w[hf * cols..(hf + 1) * cols]);
                }
            }
        }
        o
    }
    /// The wide test model as a Q8_0 GGUF (norms stay f32). `permute` stores Q/K the llama.cpp way.
    fn q8_gguf(m: &Model, path: &Path, permute: bool) {
        let s = &m.spec;
        let (h, qd, kd) = (
            s.hidden as usize,
            (s.heads * s.head_dim) as usize,
            (s.kv_heads * s.head_dim) as usize,
        );
        let q8 = |name: String, rows: usize, cols: usize, w: Vec<f32>| {
            (
                name,
                vec![rows, cols],
                8u32,
                crate::model::tests::q8_0_encode(&w),
            )
        };
        let f32t = |name: String, w: &[f32]| (name, vec![w.len()], 0u32, f32b(w));
        let mut ts = vec![
            q8(
                "token_embd.weight".into(),
                s.vocab as usize,
                h,
                m.w.embed.to_f32(),
            ),
            f32t("output_norm.weight".into(), &m.w.final_norm),
        ];
        if let Some(l) = &m.w.lm_head {
            ts.push(q8("output.weight".into(), s.vocab as usize, h, l.to_f32()));
        }
        for (i, l) in m.w.layers.iter().enumerate() {
            let n = |p: &str| layer_name(Naming::Gguf, i, p);
            let (wq, wk) = (l.wq.to_f32(), l.wk.to_f32());
            let (wq, wk) = if permute {
                (
                    permute_rows_f32(&wq, qd, h, s.heads as usize),
                    permute_rows_f32(&wk, kd, h, s.kv_heads as usize),
                )
            } else {
                (wq, wk)
            };
            ts.extend([
                f32t(n("attn_norm"), &l.attn_norm),
                q8(n("wq"), qd, h, wq),
                q8(n("wk"), kd, h, wk),
                q8(n("wv"), kd, h, l.wv.to_f32()),
                q8(n("wo"), h, qd, l.wo.to_f32()),
                f32t(n("ffn_norm"), &l.ffn_norm),
                q8(n("w_gate"), s.ffn as usize, h, l.w_gate.to_f32()),
                q8(n("w_up"), s.ffn as usize, h, l.w_up.to_f32()),
                q8(n("w_down"), h, s.ffn as usize, l.w_down.to_f32()),
            ]);
        }
        write_gguf(path, "llama", &ts);
    }
    fn wide() -> Model {
        let mut sp = crate::arch::Spec::dense("llama", 2, 32, 4, 2, 8, 64, 40, 300);
        sp.tied_embeddings = false;
        model_of(sp, false)
    }
    fn run_long(m: &Model) -> Vec<f32> {
        let mut st = m.new_state();
        m.forward_batch(&mut st, &[3, 7, 1, 19, 0, 7, 12, 5])
            .unwrap()
    }
    fn rel(a: &[f32], b: &[f32]) -> f32 {
        let sc = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
            / sc
    }

    #[test]
    fn q8_gguf_stays_quantised_with_and_without_mmap_and_matches_the_dense_model() {
        let dense = wide();
        let reference = requantise(&dense, DType::Q8_0);
        let want = run_long(&reference);
        let d = tmpdir();
        for permute in [false, true] {
            let p = d.join(format!("q8-{permute}.gguf"));
            q8_gguf(&dense, &p, permute);
            let f = GgufFile::open(&p).unwrap();
            for mmap in [false, true] {
                let m = load_dense_with(
                    dense.spec.clone(),
                    &f,
                    Naming::Gguf,
                    permute,
                    LoadOptions {
                        prefer_quantized: true,
                        mmap,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(m.w.layers[0].wq.as_dense().is_none() && m.w.embed.as_dense().is_none());
                // the same bytes through the same kernels: identical numbers
                assert_eq!(run_long(&m), want, "permute {permute} mmap {mmap}");
                assert!(m.weight_bytes() * 10 < dense.weight_bytes() * 3);
                // mapped windows are used for everything that was not reshuffled
                assert_eq!(m.w.layers[0].wv.is_mapped(), mmap);
                assert_eq!(m.w.layers[0].wq.is_mapped(), mmap && !permute);
            }
            // expanding to f32 instead gives (nearly) the same answers, and the original model is close too
            let expanded = load_dense_with(
                dense.spec.clone(),
                &f,
                Naming::Gguf,
                permute,
                LoadOptions::default(),
            )
            .unwrap();
            assert!(expanded.w.layers[0].wq.as_dense().is_some());
            assert!(rel(&run_long(&expanded), &want) < 0.05);
            assert!(rel(&run_long(&expanded), &run_long(&dense)) < 0.05);
        }
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn damaged_quantised_files_are_errors_in_both_modes() {
        let dense = wide();
        let d = tmpdir();
        let p = d.join("q8.gguf");
        q8_gguf(&dense, &p, false);
        let full = std::fs::read(&p).unwrap();
        for cut in [100usize, 5000, full.len() / 2] {
            std::fs::write(&p, &full[..full.len() - cut]).unwrap();
            for mmap in [false, true] {
                let r = GgufFile::open(&p).and_then(|f| {
                    load_dense_with(
                        dense.spec.clone(),
                        &f,
                        Naming::Gguf,
                        false,
                        LoadOptions {
                            prefer_quantized: true,
                            mmap,
                            ..Default::default()
                        },
                    )
                });
                assert!(r.is_err(), "cut {cut} mmap {mmap}");
            }
        }
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn bf16_safetensors_stay_half_size_and_mapped() {
        let m = wide();
        let d = tmpdir();
        let ts: Vec<_> = all(&m)
            .into_iter()
            .map(|(n, _, sh, d)| (n, sh, crate::model::tests::bf16_encode(&d), "BF16"))
            .collect();
        write_st(&d.join("m.safetensors"), &ts);
        let src = SafeTensors::open(&d.join("m.safetensors")).unwrap();
        let dense_bf = load_dense(m.spec.clone(), &src, Naming::Hf, false).unwrap();
        for mmap in [false, true] {
            let q = load_dense_with(
                m.spec.clone(),
                &src,
                Naming::Hf,
                false,
                LoadOptions {
                    prefer_quantized: true,
                    mmap,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(q.w.layers[1].w_down.as_dense().is_none());
            assert_eq!(q.w.layers[1].w_down.is_mapped(), mmap);
            assert!(rel(&run_long(&q), &run_long(&dense_bf)) < 1e-4);
            assert!(q.weight_bytes() * 2 <= dense_bf.weight_bytes() + 8);
        }
        // F32 files gain nothing from being kept raw and stay dense even when asked
        let d2 = tmpdir();
        write_st(
            &d2.join("m.safetensors"),
            &all(&m)
                .into_iter()
                .map(|(n, _, sh, d)| (n, sh, f32b(&d), "F32"))
                .collect::<Vec<_>>(),
        );
        let f = load_dense_with(
            m.spec.clone(),
            &SafeTensors::open(&d2).unwrap(),
            Naming::Hf,
            false,
            LoadOptions {
                prefer_quantized: true,
                mmap: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(f.w.layers[0].wq.as_dense().is_some());
        std::fs::remove_dir_all(d).unwrap();
        std::fs::remove_dir_all(d2).unwrap();
    }

    #[test]
    fn splitting_and_unpermuting_quantised_rows_agrees_with_f32() {
        let m = requantise(&wide(), DType::Q8_0);
        let w = m.w.layers[0].wq.clone(); // 32 x 32 quantised
        let dense = w.to_f32();
        let parts = split_weight(w.clone(), 32, &[8, 16, 8]).unwrap();
        let want = split_rows(dense.clone(), 32, &[8, 16, 8]).unwrap();
        for (p, d) in parts.iter().zip(&want) {
            assert_eq!(&p.to_f32(), d);
        }
        assert!(split_weight(w.clone(), 32, &[8, 16]).is_none());
        assert!(split_weight(w.clone(), 31, &[32]).is_none());
        let un = unpermute_weight(w, 32, 32, 4);
        assert!(matches!(un, Weight::Quant(_)));
        assert_eq!(un.to_f32(), unpermute_rows(&dense, 32, 32, 4));
    }

    /// Mixture-of-experts loading: synthetic Mixtral / Qwen-MoE checkpoints in safetensors and stacked GGUF form.
    mod moe_load {
        use super::*;
        use crate::model::tests::{mat, naive_moe, random_weights, reference_logits, rng};
        use crate::moe::{InMemoryExperts, Part};

        #[derive(Clone, Copy, PartialEq)]
        enum Kind {
            Mixtral,
            Qwen,
        }
        struct Fx {
            truth: Model,
            tensors: Vec<(String, String, Vec<usize>, Vec<f32>)>,
            experts: Vec<(usize, usize, [Vec<f32>; 3])>,
            n_exp: usize,
            moe_ffn: usize,
        }

        /// 3 layers; `dense_first` keeps layer 0 dense (Qwen style mixed stack).
        fn fx(kind: Kind, dense_first: bool) -> Fx {
            let (h, n_exp, shared) = (32usize, 6usize, kind == Kind::Qwen);
            let mut sp = crate::arch::Spec::dense("llama", 3, h as u64, 4, 2, 8, 64, 40, 300);
            sp.tied_embeddings = false;
            sp.experts = n_exp as u64;
            sp.experts_used = 2;
            sp.support = crate::arch::Support::Moe;
            let moe_ffn = if kind == Kind::Qwen { 32 } else { 64 };
            if kind == Kind::Qwen {
                sp.moe_ffn = moe_ffn as u64;
                sp.shared_experts = 1;
            }
            sp.norm_topk_prob = kind == Kind::Mixtral;
            let mut w = random_weights(&sp, 0x51ed_270b_c0ff_ee11, false);
            let mut seed = 0xabcd_ef12_3456_789a_u64;
            let mut tensors = all(&Model::new(sp.clone(), w.clone()).unwrap());
            let mut experts = vec![];
            let mut per_layer = vec![];
            for li in 0..3usize {
                if dense_first && li == 0 {
                    per_layer.push(vec![]);
                    continue;
                }
                let l = &mut w.layers[li];
                l.w_gate = Weight::Dense(vec![]);
                l.w_up = Weight::Dense(vec![]);
                l.w_down = Weight::Dense(vec![]);
                tensors.retain(|t| {
                    !(t.0.starts_with(&format!("model.layers.{li}.mlp."))
                        && (t.0.contains("gate_proj")
                            || t.0.contains("up_proj")
                            || t.0.contains("down_proj")))
                });
                let rw = mat(n_exp * h, &mut seed);
                let cfg = RouterConfig {
                    hidden: h,
                    experts: n_exp,
                    top_k: 2,
                    scoring: Scoring::Softmax,
                    renormalise: kind == Kind::Mixtral,
                    routed_scale: 1.0,
                    groups: None,
                    selection_bias: None,
                };
                let router = Router::new(cfg, rw.clone()).unwrap();
                let (rn_hf, rn_gg) = (
                    if kind == Kind::Mixtral {
                        format!("model.layers.{li}.block_sparse_moe.gate.weight")
                    } else {
                        format!("model.layers.{li}.mlp.gate.weight")
                    },
                    format!("blk.{li}.ffn_gate_inp.weight"),
                );
                tensors.push((rn_hf, rn_gg, vec![n_exp, h], rw));
                let mut sh = vec![];
                let mut blk_shared_gate = None;
                if shared {
                    let g = mat(moe_ffn * h, &mut seed);
                    let u = mat(moe_ffn * h, &mut seed);
                    let d = mat(h * moe_ffn, &mut seed);
                    for (p, sh_, v) in [
                        ("gate", vec![moe_ffn, h], &g),
                        ("up", vec![moe_ffn, h], &u),
                        ("down", vec![h, moe_ffn], &d),
                    ] {
                        tensors.push((
                            format!("model.layers.{li}.mlp.shared_expert.{p}_proj.weight"),
                            format!("blk.{li}.ffn_{p}_shexp.weight"),
                            sh_,
                            v.clone(),
                        ));
                    }
                    sh.push(ExpertWeights::dense(g, u, d));
                    let sg = mat(h, &mut seed);
                    tensors.push((
                        format!("model.layers.{li}.mlp.shared_expert_gate.weight"),
                        format!("blk.{li}.ffn_gate_inp_shexp.weight"),
                        vec![1, h],
                        sg.clone(),
                    ));
                    blk_shared_gate = Some(sg);
                }
                let mut blk = MoeBlock::new(li, router, sh).unwrap();
                if let Some(g) = blk_shared_gate {
                    blk = blk.with_shared_gate(g).unwrap();
                }
                l.moe = Some(Arc::new(blk));
                let mut lay = vec![];
                for e in 0..n_exp {
                    let g = mat(moe_ffn * h, &mut seed);
                    let u = mat(moe_ffn * h, &mut seed);
                    let d = mat(h * moe_ffn, &mut seed);
                    experts.push((li, e, [g.clone(), u.clone(), d.clone()]));
                    lay.push(ExpertWeights::dense(g, u, d));
                }
                per_layer.push(lay);
            }
            w.experts = Some(crate::model::Experts {
                provider: Arc::new(InMemoryExperts::new(per_layer)),
                prefetch: false,
            });
            let truth = Model::new(sp, w).unwrap();
            Fx {
                truth,
                tensors,
                experts,
                n_exp,
                moe_ffn,
            }
        }
        fn nm(kind: Kind, l: usize, e: usize, p: Part) -> String {
            if kind == Kind::Mixtral {
                names::mixtral(l, e, p)
            } else {
                names::hf_experts(l, e, p)
            }
        }
        fn part_shape(f: usize, h: usize, i: usize) -> Vec<usize> {
            if i == 2 { vec![h, f] } else { vec![f, h] }
        }
        const PARTS: [Part; 3] = [Part::Gate, Part::Up, Part::Down];

        fn write_hf(f: &Fx, kind: Kind, d: &Path) {
            let mut ts: Vec<(String, Vec<usize>, Vec<u8>, &str)> = f
                .tensors
                .iter()
                .map(|t| (t.0.clone(), t.2.clone(), f32b(&t.3), "F32"))
                .collect();
            for (l, e, p) in &f.experts {
                for (i, part) in PARTS.iter().enumerate() {
                    ts.push((
                        nm(kind, *l, *e, *part),
                        part_shape(f.moe_ffn, 32, i),
                        f32b(&p[i]),
                        "F32",
                    ));
                }
            }
            write_st(&d.join("model.safetensors"), &ts);
        }
        /// `q8`: store the experts (and nothing else) as Q8_0.
        fn write_gg(f: &Fx, path: &Path, q8: bool) {
            let mut ts: Vec<(String, Vec<usize>, u32, Vec<u8>)> = f
                .tensors
                .iter()
                .map(|t| (t.1.clone(), t.2.clone(), 0u32, f32b(&t.3)))
                .collect();
            let layers: std::collections::BTreeSet<usize> = f.experts.iter().map(|x| x.0).collect();
            for l in layers {
                for (i, part) in PARTS.iter().enumerate() {
                    let mut data = vec![];
                    let mut stacked = vec![];
                    for e in 0..f.n_exp {
                        let p = &f.experts.iter().find(|x| x.0 == l && x.1 == e).unwrap().2[i];
                        data.extend(if q8 {
                            crate::model::tests::q8_0_encode(p)
                        } else {
                            f32b(p)
                        });
                        stacked.extend(p.iter().copied());
                    }
                    let (r, c) = {
                        let s = part_shape(f.moe_ffn, 32, i);
                        (s[0], s[1])
                    };
                    ts.push((
                        names::gguf_stacked(l, *part),
                        vec![f.n_exp, r, c],
                        if q8 { 8 } else { 0 },
                        data,
                    ));
                }
            }
            write_gguf(path, "llama", &ts);
        }
        fn close(a: &[f32], b: &[f32]) {
            assert_eq!(a.len(), b.len());
            assert!(rel(a, b) < 1e-4, "rel {}", rel(a, b));
        }
        const TOK: [usize; 8] = [3, 7, 1, 19, 0, 7, 12, 5];
        fn lg(m: &Model) -> Vec<f32> {
            let mut st = m.new_state();
            m.forward_batch(&mut st, &TOK).unwrap()
        }

        #[test]
        fn the_synthetic_truth_matches_the_naive_reference_and_batch_equals_sequential() {
            for (k, df) in [(Kind::Mixtral, false), (Kind::Qwen, true)] {
                let f = fx(k, df);
                let m = &f.truth;
                let want = reference_logits(m, &TOK);
                close(&lg(m), &want[want.len() - m.spec.vocab as usize..]);
                let mut st = m.new_state();
                let mut last = vec![];
                for &t in &TOK {
                    last = m.forward(&mut st, t).unwrap();
                }
                close(&last, &lg(m));
                // the naive reference itself is exercised on the MoE layer
                let x = vec![0.3f32; 32];
                assert!(naive_moe(m, 2, &x).iter().any(|v| *v != 0.0));
            }
        }

        #[test]
        fn mixtral_safetensors_load_in_memory_and_streamed() {
            let f = fx(Kind::Mixtral, false);
            let want = lg(&f.truth);
            let d = tmpdir();
            write_hf(&f, Kind::Mixtral, &d);
            let src: Arc<dyn WeightFile> = Arc::new(SafeTensors::open(&d).unwrap());
            let spec = f.truth.spec.clone();
            let pre = load_model(
                spec.clone(),
                src.clone(),
                Naming::Hf,
                false,
                LoadOptions::default(),
                MoeOptions::default(),
            )
            .unwrap();
            close(&lg(&pre), &want);
            let per = (3 * 32 * 64 * 4) as u64;
            let ex = pre.w.experts.as_ref().unwrap();
            assert_eq!(ex.provider.resident_bytes(), per * 6 * 3);
            // a cache holding two experts: constant eviction, same answers
            for prefetch in [false, true] {
                let m = load_model(
                    spec.clone(),
                    src.clone(),
                    Naming::Hf,
                    false,
                    LoadOptions::default(),
                    MoeOptions {
                        expert_cache: Some(per * 2),
                        prefetch,
                        ..Default::default()
                    },
                )
                .unwrap();
                close(&lg(&m), &want);
                close(&lg(&m), &want);
                let p = &m.w.experts.as_ref().unwrap().provider;
                let st = p.stats().unwrap();
                assert!(st.misses > 0 && st.evictions > 0, "{st:?}");
                assert!(
                    p.resident_bytes() <= per * 2 + per,
                    "resident {} budget {}",
                    p.resident_bytes(),
                    per * 2
                );
            }
            std::fs::remove_dir_all(d).unwrap();
        }

        #[test]
        fn ram_budget_picks_in_memory_streaming_or_an_error() {
            let f = fx(Kind::Mixtral, false);
            let want = lg(&f.truth);
            let d = tmpdir();
            write_hf(&f, Kind::Mixtral, &d);
            let src: Arc<dyn WeightFile> = Arc::new(SafeTensors::open(&d).unwrap());
            let spec = f.truth.spec.clone();
            let total: u64 = src
                .infos()
                .iter()
                .map(|t| t.dtype.bytes_for(t.elements().unwrap_or(0)).unwrap_or(0))
                .sum();
            let go = |ram| {
                load_model(
                    spec.clone(),
                    src.clone(),
                    Naming::Hf,
                    false,
                    LoadOptions::default(),
                    MoeOptions {
                        ram_budget: Some(ram),
                        ..Default::default()
                    },
                )
            };
            let big = go(total * 100).unwrap();
            // everything fits: preloaded, so serving never reads from disk again
            let before = big
                .w
                .experts
                .as_ref()
                .unwrap()
                .provider
                .stats()
                .map_or(0, |s| s.bytes_read);
            close(&lg(&big), &want);
            assert_eq!(
                big.w
                    .experts
                    .as_ref()
                    .unwrap()
                    .provider
                    .stats()
                    .map_or(0, |s| s.bytes_read),
                before
            );
            let tiny_ram = go(1000);
            match tiny_ram {
                Err(e) => assert!(e.to_string().contains("does not fit"), "{e}"),
                Ok(_) => panic!("1000 bytes cannot hold anything"),
            }
            std::fs::remove_dir_all(d).unwrap();
        }

        #[test]
        fn qwen_style_mixed_stack_with_gated_shared_expert_and_other_expert_width() {
            let f = fx(Kind::Qwen, true);
            let want = lg(&f.truth);
            assert!(f.truth.w.layers[0].moe.is_none() && f.truth.w.layers[1].moe.is_some());
            let d = tmpdir();
            write_hf(&f, Kind::Qwen, &d);
            let src: Arc<dyn WeightFile> = Arc::new(SafeTensors::open(&d).unwrap());
            for cache in [None, Some(32 * 32 * 4 * 3 * 2)] {
                let m = load_model(
                    f.truth.spec.clone(),
                    src.clone(),
                    Naming::Hf,
                    false,
                    LoadOptions::default(),
                    MoeOptions {
                        expert_cache: cache,
                        prefetch: true,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(m.w.layers[1].moe.as_ref().unwrap().stored_bytes() > 0);
                close(&lg(&m), &want);
            }
            // load_dense refuses a checkpoint with experts instead of silently dropping them
            let e = load_dense(f.truth.spec.clone(), &*src, Naming::Hf, false)
                .err()
                .unwrap();
            assert!(e.to_string().contains("load_model"), "{e}");
            std::fs::remove_dir_all(d).unwrap();
        }

        #[test]
        fn gguf_stacked_experts_f32_and_q8_with_and_without_mmap() {
            let f = fx(Kind::Qwen, true);
            let d = tmpdir();
            let want = lg(&f.truth);
            let p = d.join("f32.gguf");
            write_gg(&f, &p, false);
            let src: Arc<dyn WeightFile> = Arc::new(GgufFile::open(&p).unwrap());
            for cache in [None, Some(32 * 32 * 4 * 3)] {
                let m = load_model(
                    f.truth.spec.clone(),
                    src.clone(),
                    Naming::Gguf,
                    false,
                    LoadOptions::default(),
                    MoeOptions {
                        expert_cache: cache,
                        ..Default::default()
                    },
                )
                .unwrap();
                close(&lg(&m), &want);
            }
            // quantised experts: kept as stored, identical to a dense model carrying the same Q8 numbers
            let pq = d.join("q8.gguf");
            write_gg(&f, &pq, true);
            let srcq: Arc<dyn WeightFile> = Arc::new(GgufFile::open(&pq).unwrap());
            let mut outs = vec![];
            for (quant, mmap, cache) in [
                (false, false, None),
                (true, false, None),
                (true, true, None),
                (true, true, Some(40_000u64)),
                (true, false, Some(40_000)),
            ] {
                let m = load_model(
                    f.truth.spec.clone(),
                    srcq.clone(),
                    Naming::Gguf,
                    false,
                    LoadOptions {
                        prefer_quantized: quant,
                        mmap,
                        ..Default::default()
                    },
                    MoeOptions {
                        expert_cache: cache,
                        ..Default::default()
                    },
                )
                .unwrap();
                let ex = m.w.experts.as_ref().unwrap().provider.expert(1, 3).unwrap();
                assert_eq!(ex.gate.as_dense().is_none(), quant);
                if quant && mmap {
                    assert!(ex.gate.is_mapped());
                }
                outs.push(lg(&m));
            }
            // f32-expanded and quantised-kept agree to rounding; every quantised variant is bit-identical
            assert!(
                rel(&outs[0], &outs[1]) < 0.02,
                "{}",
                rel(&outs[0], &outs[1])
            );
            for o in &outs[2..] {
                assert_eq!(o, &outs[1]);
            }
            // and Q8 stays near the f32 truth
            assert!(rel(&outs[1], &want) < 0.1);
            std::fs::remove_dir_all(d).unwrap();
        }

        #[test]
        fn ranged_reads_equal_slices_and_misaligned_ranges_fail() {
            let f = fx(Kind::Mixtral, false);
            let d = tmpdir();
            let p = d.join("r.gguf");
            write_gg(&f, &p, true);
            let g = GgufFile::open(&p).unwrap();
            let n = names::gguf_stacked(0, Part::Up);
            let all = g.read(&n).unwrap();
            let per = 64 * 32;
            assert_eq!(
                g.read_range(&n, 2 * per, per).unwrap(),
                all[2 * per..3 * per]
            );
            let raw = g
                .read_range_raw(&n, 2 * per, 64, 32, false)
                .unwrap()
                .unwrap();
            assert_eq!(raw.rows, 64);
            assert!(g.read_range(&n, 3, 32).is_err(), "start inside a block");
            assert!(g.read_range(&n, 0, 33).is_err(), "length not whole blocks");
            assert!(g.read_range(&n, 6 * per, per).is_err(), "past the tensor");
            assert!(g.read_range("nope", 0, 32).is_err());
            std::fs::remove_dir_all(d).unwrap();
        }

        #[test]
        fn hostile_checkpoints_are_errors_not_panics() {
            let f = fx(Kind::Mixtral, false);
            let d = tmpdir();
            let spec = f.truth.spec.clone();
            let open =
                |p: &Path| -> Arc<dyn WeightFile> { Arc::new(SafeTensors::open(p).unwrap()) };
            // a missing expert tensor: preloading fails at load, streaming at the first forward
            let mut ts: Vec<(String, Vec<usize>, Vec<u8>, &str)> = f
                .tensors
                .iter()
                .map(|t| (t.0.clone(), t.2.clone(), f32b(&t.3), "F32"))
                .collect();
            for (l, e, p) in &f.experts {
                if *l == 1 && *e == 4 {
                    continue;
                }
                for (i, part) in PARTS.iter().enumerate() {
                    ts.push((
                        nm(Kind::Mixtral, *l, *e, *part),
                        part_shape(64, 32, i),
                        f32b(&p[i]),
                        "F32",
                    ));
                }
            }
            write_st(&d.join("model.safetensors"), &ts);
            let src = open(&d);
            assert!(
                load_model(
                    spec.clone(),
                    src.clone(),
                    Naming::Hf,
                    false,
                    LoadOptions::default(),
                    MoeOptions::default()
                )
                .is_err()
            );
            let m = load_model(
                spec.clone(),
                src.clone(),
                Naming::Hf,
                false,
                LoadOptions::default(),
                MoeOptions {
                    expert_cache: Some(1 << 20),
                    ..Default::default()
                },
            )
            .unwrap();
            let mut hit = false;
            for t in 0..40usize {
                let mut st = m.new_state();
                if m.forward(&mut st, t % 300).is_err() {
                    hit = true;
                    break;
                }
            }
            // routing is data dependent; a model that never routes to the hole is also fine, but it must not panic
            let _ = hit;
            // wrong router length
            let mut bad = f.tensors.clone();
            for t in bad.iter_mut() {
                if t.0.contains("block_sparse_moe.gate") {
                    t.3.pop();
                    t.2 = vec![t.3.len()];
                }
            }
            let ts: Vec<_> = bad
                .iter()
                .map(|t| (t.0.clone(), t.2.clone(), f32b(&t.3), "F32"))
                .collect();
            write_st(&d.join("model.safetensors"), &ts);
            assert!(
                load_model(
                    spec.clone(),
                    open(&d),
                    Naming::Hf,
                    false,
                    LoadOptions::default(),
                    MoeOptions::default()
                )
                .is_err()
            );
            // spec says more experts than the router has rows
            let f2 = fx(Kind::Mixtral, false);
            write_hf(&f2, Kind::Mixtral, &d);
            let mut sp = spec.clone();
            sp.experts = 99;
            assert!(
                load_model(
                    sp,
                    open(&d),
                    Naming::Hf,
                    false,
                    LoadOptions::default(),
                    MoeOptions::default()
                )
                .is_err()
            );
            // top_k larger than the expert count
            let mut sp = spec.clone();
            sp.experts_used = 7;
            assert!(
                load_model(
                    sp,
                    open(&d),
                    Naming::Hf,
                    false,
                    LoadOptions::default(),
                    MoeOptions::default()
                )
                .is_err()
            );
            // truncated GGUF
            let p = d.join("t.gguf");
            write_gg(&f2, &p, true);
            let full = std::fs::read(&p).unwrap();
            for cut in [full.len() - 1, full.len() / 2, 100] {
                std::fs::write(&p, &full[..cut]).unwrap();
                if let Ok(g) = GgufFile::open(&p) {
                    let src: Arc<dyn WeightFile> = Arc::new(g);
                    let _ = load_model(
                        spec.clone(),
                        src,
                        Naming::Gguf,
                        false,
                        LoadOptions::default(),
                        MoeOptions {
                            expert_cache: Some(1 << 16),
                            ..Default::default()
                        },
                    );
                }
            }
            std::fs::remove_dir_all(d).unwrap();
        }

        #[test]
        fn wrong_shared_gate_length_is_rejected() {
            let f = fx(Kind::Qwen, true);
            let d = tmpdir();
            let mut bad = f.tensors.clone();
            for t in bad.iter_mut() {
                if t.0.ends_with("shared_expert_gate.weight") {
                    t.3.pop();
                    t.2 = vec![1, t.3.len()];
                }
            }
            let mut ts: Vec<(String, Vec<usize>, Vec<u8>, &str)> = bad
                .iter()
                .map(|t| (t.0.clone(), t.2.clone(), f32b(&t.3), "F32"))
                .collect();
            for (l, e, p) in &f.experts {
                for (i, part) in PARTS.iter().enumerate() {
                    ts.push((
                        nm(Kind::Qwen, *l, *e, *part),
                        part_shape(32, 32, i),
                        f32b(&p[i]),
                        "F32",
                    ));
                }
            }
            write_st(&d.join("model.safetensors"), &ts);
            let src: Arc<dyn WeightFile> = Arc::new(SafeTensors::open(&d).unwrap());
            assert!(
                load_model(
                    f.truth.spec.clone(),
                    src,
                    Naming::Hf,
                    false,
                    LoadOptions::default(),
                    MoeOptions::default()
                )
                .is_err()
            );
            std::fs::remove_dir_all(d).unwrap();
        }
        #[allow(dead_code)]
        fn _unused(s: &mut u64) -> f32 {
            rng(s)
        }
    }
}
