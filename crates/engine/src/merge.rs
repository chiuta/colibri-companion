//! Weight-space model merging, one tensor at a time.
//!
//! Inputs are safetensors (a file, or a folder with or without an index) or GGUF files; the
//! output is one safetensors file in F32, F16 or BF16. A tensor is read, combined and written
//! before the next one is touched, so the memory needed is a few copies of the *largest tensor*,
//! not of the model ([`Plan::peak_f32`] says how many f32 values, and [`MergeConfig::max_peak_bytes`]
//! turns an over-budget plan into an error instead of an out-of-memory kill). Quantised GGUF
//! tensors are dequantised to f32 on the fly; the output is never quantised, so merging
//! and then re-quantising is a separate step.
//!
//! Methods (all per tensor, in f32 with f64 where precision matters):
//! * `linear`: weighted average (normalised by default).
//! * `slerp`: spherical interpolation of two models by `t` (falls back to a straight lerp for
//!   parallel or zero vectors).
//! * `task`: task arithmetic, `base + scale * Σ wᵢ (xᵢ − base)`.
//! * `ties`: trim each delta to its top `density` by magnitude, elect a sign per element from the
//!   weighted sum, and average only the deltas that agree with it.
//! * `dare`: drop each delta element with probability `1 − density` (seeded, so the result is the
//!   same on every run), rescale the survivors, then add like `task` (or with TIES sign election).
//! * `passthrough`: copy one input's tensor; with `slices` this builds a "frankenmerge" whose layers
//!   come from layer ranges of different models.
//!
//! What is checked before anything is written: every input must list the same tensor names with
//! the same shapes (GGUF's reversed dimension order is normalised), GGUF files must agree on
//! `general.architecture`, safetensors folders with a `config.json` must agree on `model_type`,
//! GGUF and safetensors cannot be mixed (their tensor names differ), every tensor type must be
//! decodable, weights and parameters must be in range, and the output may not overwrite an
//! input. While merging, non-finite input values and F16 overflow are errors, never silent.
//!
//! Limits, honestly: merged weights are a *recipe*, not a guarantee. Models must share an
//! architecture, tokenizer and (for task/TIES/DARE) a common ancestor, or the result is
//! noise. Peak memory is accounted from tensor sizes, not measured. Only the weights are
//! written; copy `config.json`, tokenizer files and so on yourself, and for a frankenmerge set
//! `num_hidden_layers` (and any per-layer config lists) to [`MergeStats::out_layers`].

use crate::loader::{GgufFile, SafeTensors, TensorSource};
use crate::quantize::{f32_to_bf16, f32_to_f16};
use crate::sampling::Rng;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ------------------------------------------------------------------------------ errors

#[derive(Debug)]
pub enum MergeError {
    Io(std::io::Error),
    /// A tensor could not be read or decoded.
    Load(String),
    /// The configuration is malformed or out of range.
    Config(String),
    /// The inputs do not fit together (names, shapes, architecture, layers).
    Mismatch(String),
    /// An input or a result contains NaN or infinity.
    NonFinite {
        tensor: String,
        whence: String,
    },
    /// A value does not fit the chosen output type.
    Overflow {
        tensor: String,
        value: f32,
    },
    /// The plan needs more memory than the configured limit.
    OverBudget {
        need_bytes: u64,
        limit_bytes: u64,
    },
    Unsupported(String),
}
impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::Io(e) => write!(f, "i/o error: {e}"),
            MergeError::Load(s) => write!(f, "cannot read tensor: {s}"),
            MergeError::Config(s) => write!(f, "bad merge configuration: {s}"),
            MergeError::Mismatch(s) => write!(f, "the models do not fit together: {s}"),
            MergeError::NonFinite { tensor, whence } => {
                write!(f, "{tensor}: NaN or infinity in {whence}")
            }
            MergeError::Overflow { tensor, value } => write!(
                f,
                "{tensor}: value {value} does not fit the output type (use bf16 or f32)"
            ),
            MergeError::OverBudget {
                need_bytes,
                limit_bytes,
            } => write!(
                f,
                "merge needs about {need_bytes} bytes of memory, limit is {limit_bytes}"
            ),
            MergeError::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}
impl std::error::Error for MergeError {}
impl From<std::io::Error> for MergeError {
    fn from(e: std::io::Error) -> Self {
        MergeError::Io(e)
    }
}
impl From<crate::loader::LoadError> for MergeError {
    fn from(e: crate::loader::LoadError) -> Self {
        MergeError::Load(e.to_string())
    }
}

fn cfg_err<T>(s: impl Into<String>) -> Result<T, MergeError> {
    Err(MergeError::Config(s.into()))
}
fn mis<T>(s: impl Into<String>) -> Result<T, MergeError> {
    Err(MergeError::Mismatch(s.into()))
}

// ------------------------------------------------------------------------ configuration

#[derive(Clone, Debug, PartialEq)]
pub enum Method {
    Linear {
        normalize: bool,
    },
    Slerp {
        t: f32,
    },
    Task {
        scale: f32,
    },
    Ties {
        density: f32,
        scale: f32,
    },
    Dare {
        density: f32,
        scale: f32,
        sign_election: bool,
    },
    /// Copy this input's tensor unchanged.
    Passthrough {
        input: usize,
    },
}
impl Method {
    pub fn name(&self) -> &'static str {
        match self {
            Method::Linear { .. } => "linear",
            Method::Slerp { .. } => "slerp",
            Method::Task { .. } => "task",
            Method::Ties { .. } => "ties",
            Method::Dare { .. } => "dare",
            Method::Passthrough { .. } => "passthrough",
        }
    }
    fn needs_base(&self) -> bool {
        matches!(
            self,
            Method::Task { .. } | Method::Ties { .. } | Method::Dare { .. }
        )
    }
    fn elects(&self) -> bool {
        matches!(
            self,
            Method::Ties { .. }
                | Method::Dare {
                    sign_election: true,
                    ..
                }
        )
    }
    /// Number of tensor-sized f32 buffers alive at once.
    fn buffers(&self) -> u64 {
        match self {
            Method::Passthrough { .. } => 1,
            Method::Linear { .. } => 2,
            Method::Slerp { .. } | Method::Task { .. } => 3,
            Method::Dare {
                sign_election: false,
                ..
            } => 3,
            Method::Dare { .. } => 5,
            Method::Ties { .. } => 6,
        }
    }
    fn validate(&self) -> Result<(), MergeError> {
        match *self {
            Method::Slerp { t } if !(t.is_finite() && (0.0..=1.0).contains(&t)) => {
                cfg_err("slerp t must be in [0, 1]")
            }
            Method::Task { scale } if !scale.is_finite() => cfg_err("scale must be finite"),
            Method::Ties { density, scale } | Method::Dare { density, scale, .. } => {
                if !(density.is_finite() && density > 0.0 && density <= 1.0) {
                    return cfg_err("density must be in (0, 1]");
                }
                if !scale.is_finite() {
                    return cfg_err("scale must be finite");
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutDType {
    F32,
    F16,
    BF16,
}
impl OutDType {
    pub fn bytes(self) -> u64 {
        match self {
            OutDType::F32 => 4,
            _ => 2,
        }
    }
    fn st_name(self) -> &'static str {
        match self {
            OutDType::F32 => "F32",
            OutDType::F16 => "F16",
            OutDType::BF16 => "BF16",
        }
    }
    pub fn parse(s: &str) -> Option<OutDType> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "float32" => Some(OutDType::F32),
            "f16" | "float16" => Some(OutDType::F16),
            "bf16" | "bfloat16" => Some(OutDType::BF16),
            _ => None,
        }
    }
}

/// Overrides for tensors whose name matches `pattern` (`*` matches any run of characters; the
/// first matching rule wins).
#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    pub pattern: String,
    pub method: Option<Method>,
    pub weights: Option<Vec<f32>>,
}

/// Layers `start..end` (end exclusive) of input `input`, appended to the output stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slice {
    pub input: usize,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MergeConfig {
    pub method: Method,
    /// One weight per input; empty means "all 1" (then `linear` normalises to the mean).
    pub weights: Vec<f32>,
    /// The common ancestor, an index into the inputs; required by task, TIES and DARE.
    pub base: Option<usize>,
    pub rules: Vec<Rule>,
    /// When non-empty, build a layer-stacked model from these slices (the `method` is ignored).
    pub slices: Vec<Slice>,
    /// Which input supplies the non-layer tensors (embeddings, final norm, head) in slice mode.
    pub outer_from: Option<usize>,
    pub dtype: OutDType,
    pub seed: u64,
    /// Merge only the tensors every input has instead of failing on a difference.
    pub lenient: bool,
    /// Replace an existing output file.
    pub overwrite: bool,
    /// Refuse plans whose accounted peak memory exceeds this many bytes.
    pub max_peak_bytes: Option<u64>,
}
impl Default for MergeConfig {
    fn default() -> Self {
        MergeConfig {
            method: Method::Linear { normalize: true },
            weights: vec![],
            base: None,
            rules: vec![],
            slices: vec![],
            outer_from: None,
            dtype: OutDType::BF16,
            seed: 0,
            lenient: false,
            overwrite: false,
            max_peak_bytes: None,
        }
    }
}

fn jnum(o: &Map<String, Value>, k: &str) -> Result<Option<f64>, MergeError> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_f64()
            .filter(|x| x.is_finite())
            .map(Some)
            .ok_or_else(|| MergeError::Config(format!("\"{k}\" must be a number"))),
    }
}
fn jidx(o: &Map<String, Value>, k: &str) -> Result<Option<usize>, MergeError> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(|x| usize::try_from(x).ok())
            .map(Some)
            .ok_or_else(|| MergeError::Config(format!("\"{k}\" must be a non-negative integer"))),
    }
}
fn jbool(o: &Map<String, Value>, k: &str) -> Result<Option<bool>, MergeError> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        _ => cfg_err(format!("\"{k}\" must be true or false")),
    }
}
fn jweights(o: &Map<String, Value>, k: &str) -> Result<Option<Vec<f32>>, MergeError> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_f64()
                    .filter(|x| x.is_finite())
                    .map(|x| x as f32)
                    .ok_or_else(|| MergeError::Config(format!("\"{k}\" must hold finite numbers")))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        _ => cfg_err(format!("\"{k}\" must be an array")),
    }
}
fn only_keys(o: &Map<String, Value>, ok: &[&str], what: &str) -> Result<(), MergeError> {
    for k in o.keys() {
        if !ok.contains(&k.as_str()) {
            return cfg_err(format!("unknown key \"{k}\" in {what}"));
        }
    }
    Ok(())
}
/// Method parameters, looked up in `o` first and then in `fallback`.
fn parse_method(
    name: &str,
    o: &Map<String, Value>,
    fallback: &Map<String, Value>,
) -> Result<Method, MergeError> {
    let num =
        |k: &str| -> Result<Option<f64>, MergeError> { Ok(jnum(o, k)?.or(jnum(fallback, k)?)) };
    let m = match name {
        "linear" => Method::Linear {
            normalize: jbool(o, "normalize")?
                .or(jbool(fallback, "normalize")?)
                .unwrap_or(true),
        },
        "slerp" => Method::Slerp {
            t: num("t")?.unwrap_or(0.5) as f32,
        },
        "task" | "task_arithmetic" => Method::Task {
            scale: num("scale")?.unwrap_or(1.0) as f32,
        },
        "ties" => Method::Ties {
            density: num("density")?.unwrap_or(0.5) as f32,
            scale: num("scale")?.unwrap_or(1.0) as f32,
        },
        "dare" => Method::Dare {
            density: num("density")?.unwrap_or(0.5) as f32,
            scale: num("scale")?.unwrap_or(1.0) as f32,
            sign_election: jbool(o, "sign_election")?
                .or(jbool(fallback, "sign_election")?)
                .unwrap_or(false),
        },
        "passthrough" => Method::Passthrough {
            input: jidx(o, "input")?.or(jidx(fallback, "input")?).unwrap_or(0),
        },
        other => {
            return cfg_err(format!(
                "unknown method \"{other}\" (linear, slerp, task, ties, dare, passthrough)"
            ));
        }
    };
    m.validate()?;
    Ok(m)
}

impl MergeConfig {
    /// Parse a configuration such as
    /// `{"method":"ties","density":0.5,"base":0,"weights":[1,1],"dtype":"bf16","seed":7,
    ///   "rules":[{"match":"*norm*","method":"linear","weights":[0.5,0.5]}],
    ///   "slices":[{"input":0,"start":0,"end":16},{"input":1,"start":8,"end":24}]}`.
    /// Unknown keys are errors, so a typo cannot silently change a merge.
    pub fn from_json(s: &str) -> Result<MergeConfig, MergeError> {
        let v: Value = serde_json::from_str(s)
            .map_err(|e| MergeError::Config(format!("not valid JSON: {e}")))?;
        let o = v
            .as_object()
            .ok_or_else(|| MergeError::Config("the configuration must be a JSON object".into()))?;
        only_keys(
            o,
            &[
                "method",
                "density",
                "scale",
                "t",
                "normalize",
                "sign_election",
                "input",
                "weights",
                "base",
                "rules",
                "slices",
                "outer_from",
                "dtype",
                "seed",
                "lenient",
                "overwrite",
                "max_peak_bytes",
            ],
            "the configuration",
        )?;
        let mut c = MergeConfig::default();
        let mname = match o.get("method") {
            None => "linear",
            Some(Value::String(s)) => s.as_str(),
            _ => return cfg_err("\"method\" must be a string"),
        };
        c.method = parse_method(mname, o, &Map::new())?;
        c.weights = jweights(o, "weights")?.unwrap_or_default();
        c.base = jidx(o, "base")?;
        c.outer_from = jidx(o, "outer_from")?;
        if let Some(d) = o.get("dtype") {
            c.dtype = d
                .as_str()
                .and_then(OutDType::parse)
                .ok_or_else(|| MergeError::Config("\"dtype\" must be f32, f16 or bf16".into()))?;
        }
        if let Some(s) = o.get("seed") {
            c.seed = s.as_u64().ok_or_else(|| {
                MergeError::Config("\"seed\" must be a non-negative integer".into())
            })?;
        }
        c.lenient = jbool(o, "lenient")?.unwrap_or(false);
        c.overwrite = jbool(o, "overwrite")?.unwrap_or(false);
        if let Some(b) = jnum(o, "max_peak_bytes")? {
            if b < 0.0 {
                return cfg_err("\"max_peak_bytes\" must not be negative");
            }
            c.max_peak_bytes = Some(b as u64);
        }
        if let Some(r) = o.get("rules") {
            for item in r
                .as_array()
                .ok_or_else(|| MergeError::Config("\"rules\" must be an array".into()))?
            {
                let ro = item
                    .as_object()
                    .ok_or_else(|| MergeError::Config("each rule must be an object".into()))?;
                only_keys(
                    ro,
                    &[
                        "match",
                        "method",
                        "weights",
                        "density",
                        "scale",
                        "t",
                        "normalize",
                        "sign_election",
                        "input",
                    ],
                    "a rule",
                )?;
                let pattern = ro
                    .get("match")
                    .and_then(Value::as_str)
                    .ok_or_else(|| MergeError::Config("each rule needs a \"match\" string".into()))?
                    .to_string();
                let method = match ro.get("method") {
                    None => None,
                    Some(Value::String(n)) => Some(parse_method(n, ro, o)?),
                    _ => return cfg_err("a rule's \"method\" must be a string"),
                };
                c.rules.push(Rule {
                    pattern,
                    method,
                    weights: jweights(ro, "weights")?,
                });
            }
        }
        if let Some(s) = o.get("slices") {
            for item in s
                .as_array()
                .ok_or_else(|| MergeError::Config("\"slices\" must be an array".into()))?
            {
                let so = item
                    .as_object()
                    .ok_or_else(|| MergeError::Config("each slice must be an object".into()))?;
                only_keys(so, &["input", "start", "end"], "a slice")?;
                let g = |k: &str| {
                    jidx(so, k)?.ok_or_else(|| MergeError::Config(format!("a slice needs \"{k}\"")))
                };
                c.slices.push(Slice {
                    input: g("input")?,
                    start: g("start")?,
                    end: g("end")?,
                });
            }
        }
        Ok(c)
    }
}

// ----------------------------------------------------------------------------- inputs

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Shapes as in safetensors (row-major, outermost first).
    Numpy,
    /// GGUF lists the fastest-varying dimension first; shapes are reversed to compare.
    Gguf,
}

/// One model to merge from.
pub struct Input {
    pub label: String,
    src: Arc<dyn TensorSource>,
    layout: Layout,
    arch: Option<String>,
    path: Option<PathBuf>,
}
impl Input {
    /// A `.gguf` file, a `.safetensors` file, or a folder of safetensors shards.
    pub fn open(path: &Path) -> Result<Input, MergeError> {
        let label = path.display().to_string();
        let is_gguf = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
        if is_gguf {
            let g = GgufFile::open(path)?;
            let arch = g.gguf.architecture().map(str::to_string);
            return Ok(Input {
                label,
                src: Arc::new(g),
                layout: Layout::Gguf,
                arch,
                path: Some(path.to_path_buf()),
            });
        }
        let st = SafeTensors::open(path)?;
        let cfg = if path.is_dir() {
            path.join("config.json")
        } else {
            path.parent()
                .map(|p| p.join("config.json"))
                .unwrap_or_default()
        };
        let arch = std::fs::read(&cfg)
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .and_then(|v| {
                v.get("model_type")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            });
        Ok(Input {
            label,
            src: Arc::new(st),
            layout: Layout::Numpy,
            arch,
            path: Some(path.to_path_buf()),
        })
    }
    /// Any tensor source (used by tests and by callers that already hold one). `arch` is
    /// compared between inputs when both have one.
    pub fn from_source(
        label: impl Into<String>,
        src: Arc<dyn TensorSource>,
        layout: Layout,
        arch: Option<String>,
    ) -> Input {
        Input {
            label: label.into(),
            src,
            layout,
            arch,
            path: None,
        }
    }
    fn metas(&self) -> Result<BTreeMap<String, Meta>, MergeError> {
        let infos = self.src.infos();
        if infos.is_empty() {
            return Err(MergeError::Unsupported(format!(
                "{}: the source cannot list its tensors (or has none)",
                self.label
            )));
        }
        let mut m = BTreeMap::new();
        for t in infos {
            if !t.dtype.decodable() {
                return Err(MergeError::Unsupported(format!(
                    "{}: tensor {} has type {:?}, which cannot be decoded",
                    self.label, t.name, t.dtype
                )));
            }
            let mut shape = t.dims.clone();
            if self.layout == Layout::Gguf {
                shape.reverse();
            }
            let elems = shape
                .iter()
                .try_fold(1u64, |a, &d| a.checked_mul(d))
                .ok_or_else(|| {
                    MergeError::Load(format!("{}: {} has an absurd shape", self.label, t.name))
                })?;
            if usize::try_from(elems).is_err() {
                return Err(MergeError::Load(format!(
                    "{}: {} is too large for this machine",
                    self.label, t.name
                )));
            }
            if m.insert(
                t.name.clone(),
                Meta {
                    shape,
                    elems: elems as usize,
                },
            )
            .is_some()
            {
                return mis(format!("{}: tensor {} appears twice", self.label, t.name));
            }
        }
        Ok(m)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Meta {
    shape: Vec<u64>,
    elems: usize,
}

// ------------------------------------------------------------------------------- plan

#[derive(Clone, Debug)]
pub struct PlannedTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub elems: usize,
    /// The method's name, for display.
    pub method: &'static str,
    srcs: Vec<(usize, String)>,
    m: Method,
    weights: Vec<f32>,
    base_pos: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Plan {
    /// In output order (sorted by name).
    pub tensors: Vec<PlannedTensor>,
    /// Layers in the output when slices were used.
    pub out_layers: Option<usize>,
    pub out_bytes: u64,
    /// Accounted peak of tensor-sized f32 values alive at once (4 bytes each).
    pub peak_f32: u64,
    pub max_tensor_elems: u64,
    pub total_elems: u64,
    pub dtype: OutDType,
}

/// `*` matches any run of characters (including none); everything else matches itself.
pub fn glob_match(pat: &str, s: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pat.chars().collect(), s.chars().collect());
    let (mut pi, mut ti, mut star, mut mark) = (0usize, 0usize, None::<usize>, 0usize);
    while ti < t.len() {
        if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if pi < p.len() && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// Split a tensor name around its layer index: `model.layers.12.mlp.up.weight` →
/// (`model.layers.`, 12, `.mlp.up.weight`); GGUF's `blk.12.…` works the same way. Only the
/// first match counts.
pub fn layer_split(name: &str) -> Option<(String, usize, String)> {
    for key in ["layers.", "blk."] {
        let mut from = 0;
        while let Some(rel) = name[from..].find(key) {
            let at = from + rel;
            let ok_start = at == 0 || name.as_bytes()[at - 1] == b'.';
            let ds = at + key.len();
            let de = ds + name[ds..].bytes().take_while(u8::is_ascii_digit).count();
            if ok_start
                && de > ds
                && name[de..].starts_with('.')
                && let Ok(idx) = name[ds..de].parse::<usize>()
            {
                return Some((name[..ds].to_string(), idx, name[de..].to_string()));
            }
            from = at + key.len();
        }
    }
    None
}

fn shape_str(s: &[u64]) -> String {
    format!("{s:?}")
}

fn check_inputs(inputs: &[Input]) -> Result<(), MergeError> {
    if inputs.is_empty() {
        return mis("no input models");
    }
    let l0 = inputs[0].layout;
    for i in &inputs[1..] {
        if i.layout != l0 {
            return mis(format!(
                "{} and {} use different file kinds (GGUF and safetensors name their tensors differently); convert one first",
                inputs[0].label, i.label
            ));
        }
    }
    let mut arch: Option<(&str, &str)> = None;
    for i in inputs {
        if let Some(a) = &i.arch {
            match arch {
                None => arch = Some((a, &i.label)),
                Some((b, bl)) if a != b => {
                    return mis(format!(
                        "architecture \"{a}\" ({}) differs from \"{b}\" ({bl})",
                        i.label
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn fit_weights(w: &[f32], k: usize, what: &str) -> Result<Vec<f32>, MergeError> {
    if w.is_empty() {
        return Ok(vec![1.0; k]);
    }
    if w.len() != k {
        return cfg_err(format!("{what}: {} weights given for {k} inputs", w.len()));
    }
    if w.iter().any(|x| !x.is_finite()) {
        return cfg_err(format!("{what}: weights must be finite"));
    }
    Ok(w.to_vec())
}

/// Validate everything and compute what [`merge`] will do, without reading any weights.
pub fn plan(inputs: &[Input], cfg: &MergeConfig) -> Result<Plan, MergeError> {
    check_inputs(inputs)?;
    cfg.method.validate()?;
    let metas: Vec<BTreeMap<String, Meta>> =
        inputs.iter().map(Input::metas).collect::<Result<_, _>>()?;
    let mut tensors = vec![];
    let mut out_layers = None;
    if cfg.slices.is_empty() {
        plan_merge(inputs, cfg, &metas, &mut tensors)?;
    } else {
        out_layers = Some(plan_slices(inputs, cfg, &metas, &mut tensors)?);
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let (mut peak, mut max_e, mut total, mut bytes) = (0u64, 0u64, 0u64, 0u64);
    for t in &tensors {
        let e = t.elems as u64;
        max_e = max_e.max(e);
        total = total.saturating_add(e);
        peak = peak.max(e.saturating_mul(t.m.buffers()));
        bytes = bytes.saturating_add(e.saturating_mul(cfg.dtype.bytes()));
    }
    if let Some(limit) = cfg.max_peak_bytes
        && peak.saturating_mul(4) > limit
    {
        return Err(MergeError::OverBudget {
            need_bytes: peak.saturating_mul(4),
            limit_bytes: limit,
        });
    }
    if tensors.is_empty() {
        return mis("no tensors to write");
    }
    Ok(Plan {
        tensors,
        out_layers,
        out_bytes: bytes,
        peak_f32: peak,
        max_tensor_elems: max_e,
        total_elems: total,
        dtype: cfg.dtype,
    })
}

fn plan_merge(
    inputs: &[Input],
    cfg: &MergeConfig,
    metas: &[BTreeMap<String, Meta>],
    out: &mut Vec<PlannedTensor>,
) -> Result<(), MergeError> {
    let k = inputs.len();
    let global_w = fit_weights(&cfg.weights, k, "weights")?;
    if let Some(b) = cfg.base
        && b >= k
    {
        return cfg_err(format!("base {b} is not one of the {k} inputs"));
    }
    // Names every input has; a difference is an error unless `lenient`.
    let first: BTreeSet<&String> = metas[0].keys().collect();
    let mut common = first.clone();
    for (i, m) in metas.iter().enumerate().skip(1) {
        let here: BTreeSet<&String> = m.keys().collect();
        if !cfg.lenient && here != first {
            let miss: Vec<&&String> = first.difference(&here).take(3).collect();
            let extra: Vec<&&String> = here.difference(&first).take(3).collect();
            return mis(format!(
                "{} and {} list different tensors (missing in the second: {miss:?}; extra in the second: {extra:?})",
                inputs[0].label, inputs[i].label
            ));
        }
        common = common.intersection(&here).copied().collect();
    }
    for name in common {
        let m0 = &metas[0][name];
        for (i, m) in metas.iter().enumerate().skip(1) {
            if m[name].shape != m0.shape {
                return mis(format!(
                    "{name}: shape {} in {} but {} in {}",
                    shape_str(&m0.shape),
                    inputs[0].label,
                    shape_str(&m[name].shape),
                    inputs[i].label
                ));
            }
        }
        let rule = cfg.rules.iter().find(|r| glob_match(&r.pattern, name));
        let method = rule
            .and_then(|r| r.method.clone())
            .unwrap_or_else(|| cfg.method.clone());
        method.validate()?;
        let weights = match rule.and_then(|r| r.weights.as_ref()) {
            Some(w) => fit_weights(
                w,
                k,
                &format!("rule \"{}\"", rule.map_or("", |r| r.pattern.as_str())),
            )?,
            None => global_w.clone(),
        };
        let (srcs, weights, base_pos) = match &method {
            Method::Passthrough { input } => {
                if *input >= k {
                    return cfg_err(format!(
                        "passthrough input {input} is not one of the {k} inputs"
                    ));
                }
                (vec![(*input, name.clone())], vec![1.0], None)
            }
            _ => (
                (0..k).map(|i| (i, name.clone())).collect(),
                weights,
                cfg.base,
            ),
        };
        check_method(&method, srcs.len(), &weights, base_pos, name)?;
        out.push(PlannedTensor {
            name: name.clone(),
            shape: m0.shape.clone(),
            elems: m0.elems,
            method: method.name(),
            srcs,
            m: method,
            weights,
            base_pos,
        });
    }
    Ok(())
}

fn check_method(
    m: &Method,
    k: usize,
    w: &[f32],
    base: Option<usize>,
    name: &str,
) -> Result<(), MergeError> {
    match m {
        Method::Slerp { .. } if k != 2 => {
            return cfg_err(format!("{name}: slerp needs exactly two inputs, got {k}"));
        }
        Method::Linear { normalize: true } => {
            let s: f64 = w.iter().map(|&x| x as f64).sum();
            if s.abs() < 1e-9 {
                return cfg_err(format!(
                    "{name}: the weights sum to zero, so they cannot be normalised"
                ));
            }
        }
        _ => {}
    }
    if m.needs_base() {
        let Some(b) = base else {
            return cfg_err(format!(
                "{name}: {} needs \"base\" (the common ancestor)",
                m.name()
            ));
        };
        if k < 2 || b >= k {
            return cfg_err(format!(
                "{name}: {} needs a base and at least one other input",
                m.name()
            ));
        }
        if m.elects() && (0..k).any(|i| i != b && w[i] < 0.0) {
            return cfg_err(format!("{name}: sign election needs non-negative weights"));
        }
    }
    Ok(())
}

fn plan_slices(
    inputs: &[Input],
    cfg: &MergeConfig,
    metas: &[BTreeMap<String, Meta>],
    out: &mut Vec<PlannedTensor>,
) -> Result<usize, MergeError> {
    let k = inputs.len();
    // template ("model.layers.{}.mlp.up.weight") -> (real name, meta), per input and layer
    type Layers = BTreeMap<usize, BTreeMap<String, (String, Meta)>>;
    let mut layers: Vec<Layers> = vec![BTreeMap::new(); k];
    let mut outer: Vec<BTreeMap<&String, &Meta>> = vec![BTreeMap::new(); k];
    for (i, m) in metas.iter().enumerate() {
        for (name, meta) in m {
            match layer_split(name) {
                Some((pre, idx, suf)) => {
                    layers[i]
                        .entry(idx)
                        .or_default()
                        .insert(format!("{pre}{{}}{suf}"), (name.clone(), meta.clone()));
                }
                None => {
                    outer[i].insert(name, meta);
                }
            }
        }
    }
    for s in &cfg.slices {
        if s.input >= k {
            return cfg_err(format!(
                "slice input {} is not one of the {k} inputs",
                s.input
            ));
        }
        if s.start >= s.end {
            return cfg_err(format!(
                "slice {}..{} of input {} is empty or reversed",
                s.start, s.end, s.input
            ));
        }
        let have = layers[s.input].keys().next_back().map_or(0, |m| m + 1);
        if s.end > have {
            return mis(format!(
                "{} has {have} layers, slice asks for layers up to {}",
                inputs[s.input].label, s.end
            ));
        }
    }
    let outer_from = cfg.outer_from.unwrap_or(cfg.slices[0].input);
    if outer_from >= k {
        return cfg_err(format!(
            "outer_from {outer_from} is not one of the {k} inputs"
        ));
    }
    let mut reference: Option<BTreeMap<String, Vec<u64>>> = None; // template -> shape of the first output layer
    let mut out_idx = 0usize;
    for s in &cfg.slices {
        for l in s.start..s.end {
            let Some(group) = layers[s.input].get(&l) else {
                return mis(format!("{} has no layer {l}", inputs[s.input].label));
            };
            let shapes: BTreeMap<String, Vec<u64>> = group
                .iter()
                .map(|(t, (_, m))| (t.clone(), m.shape.clone()))
                .collect();
            match &reference {
                None => reference = Some(shapes),
                Some(r) if *r != shapes => {
                    let diff = r
                        .keys()
                        .find(|t| shapes.get(*t) != r.get(*t))
                        .or_else(|| shapes.keys().find(|t| !r.contains_key(*t)));
                    return mis(format!(
                        "layer {l} of {} has different tensors or shapes than the first slice's layers (first difference: {diff:?}); the models are not the same size",
                        inputs[s.input].label
                    ));
                }
                _ => {}
            }
            for (tmpl, (real, meta)) in group {
                let name = tmpl.replacen("{}", &out_idx.to_string(), 1);
                out.push(PlannedTensor {
                    name,
                    shape: meta.shape.clone(),
                    elems: meta.elems,
                    method: "passthrough",
                    srcs: vec![(s.input, real.clone())],
                    m: Method::Passthrough { input: s.input },
                    weights: vec![1.0],
                    base_pos: None,
                });
            }
            out_idx += 1;
        }
    }
    if outer[outer_from].is_empty() {
        return mis(format!(
            "{} has no non-layer tensors (embeddings, final norm); pick another \"outer_from\"",
            inputs[outer_from].label
        ));
    }
    for (name, meta) in &outer[outer_from] {
        for s in &cfg.slices {
            if let Some(other) = outer[s.input].get(*name)
                && other.shape != meta.shape
            {
                return mis(format!(
                    "{name}: shape {} in {} but {} in {}; hidden or vocabulary sizes differ",
                    shape_str(&meta.shape),
                    inputs[outer_from].label,
                    shape_str(&other.shape),
                    inputs[s.input].label
                ));
            }
        }
        out.push(PlannedTensor {
            name: (*name).clone(),
            shape: meta.shape.clone(),
            elems: meta.elems,
            method: "passthrough",
            srcs: vec![(outer_from, (*name).clone())],
            m: Method::Passthrough { input: outer_from },
            weights: vec![1.0],
            base_pos: None,
        });
    }
    Ok(out_idx)
}

// ------------------------------------------------------------------------------- maths

fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn read_checked(
    inputs: &[Input],
    idx: usize,
    name: &str,
    elems: usize,
) -> Result<Vec<f32>, MergeError> {
    let v = inputs[idx]
        .src
        .read(name)
        .map_err(|e| MergeError::Load(format!("{} / {name}: {e}", inputs[idx].label)))?;
    if v.len() != elems {
        return Err(MergeError::Load(format!(
            "{} / {name}: {} values, expected {elems}",
            inputs[idx].label,
            v.len()
        )));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(MergeError::NonFinite {
            tensor: name.to_string(),
            whence: format!("input {}", inputs[idx].label),
        });
    }
    Ok(v)
}

fn lerp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let t = t as f64;
    a.iter()
        .zip(b)
        .map(|(&x, &y)| ((1.0 - t) * x as f64 + t * y as f64) as f32)
        .collect()
}

/// Spherical interpolation of two flattened tensors. `t = 0` and `t = 1` return the inputs
/// exactly; nearly parallel or zero vectors use a plain lerp.
pub fn slerp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    if t <= 0.0 {
        return a.to_vec();
    }
    if t >= 1.0 {
        return b.to_vec();
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let (na, nb) = (na.sqrt(), nb.sqrt());
    if na < 1e-12 || nb < 1e-12 {
        return lerp(a, b, t);
    }
    let c = (dot / (na * nb)).clamp(-1.0, 1.0);
    if c.abs() > 0.9995 {
        return lerp(a, b, t);
    }
    let th = c.acos();
    let s = th.sin();
    let t = t as f64;
    let (w0, w1) = (((1.0 - t) * th).sin() / s, (t * th).sin() / s);
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (w0 * x as f64 + w1 * y as f64) as f32)
        .collect()
}

/// Keep the `k` largest magnitudes of `d` (exactly `k`; ties at the threshold go to the lowest
/// indices) and zero the rest.
pub fn trim_top(d: &mut [f32], k: usize, scratch: &mut Vec<f32>) {
    let n = d.len();
    if k >= n {
        return;
    }
    if k == 0 {
        d.fill(0.0);
        return;
    }
    scratch.clear();
    scratch.extend(d.iter().map(|x| x.abs()));
    let (_, th, _) = scratch.select_nth_unstable_by(n - k, f32::total_cmp);
    let th = *th;
    let greater = d.iter().filter(|x| x.abs() > th).count();
    let mut equal_left = k.saturating_sub(greater);
    for x in d.iter_mut() {
        let a = x.abs();
        if a > th {
            continue;
        }
        if a == th && equal_left > 0 {
            equal_left -= 1;
            continue;
        }
        *x = 0.0;
    }
}

/// Drop each element with probability `1 − density`, rescaling the survivors by `1 / density`.
pub fn dare_drop(d: &mut [f32], density: f32, rng: &mut Rng) {
    if density >= 1.0 {
        return;
    }
    let r = 1.0 / density;
    for x in d.iter_mut() {
        if rng.f32() < density {
            *x *= r;
        } else {
            *x = 0.0;
        }
    }
}

struct Delta<'a> {
    m: &'a Method,
    seed: u64,
    name: &'a str,
}
impl Delta<'_> {
    /// `x := prep(x − base)`, deterministic per (seed, tensor, input).
    fn prep(&self, x: &mut [f32], base: &[f32], pos: usize, scratch: &mut Vec<f32>) {
        for (v, &b) in x.iter_mut().zip(base) {
            *v -= b;
        }
        match *self.m {
            Method::Ties { density, .. } => trim_top(
                x,
                ((density as f64 * x.len() as f64).ceil() as usize).clamp(1, x.len().max(1)),
                scratch,
            ),
            Method::Dare { density, .. } => {
                let mut rng = Rng::new(
                    self.seed
                        ^ fnv(self.name).rotate_left(17)
                        ^ (pos as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                );
                dare_drop(x, density, &mut rng);
            }
            _ => {}
        }
    }
}

fn run_tensor(inputs: &[Input], t: &PlannedTensor, seed: u64) -> Result<Vec<f32>, MergeError> {
    let n = t.elems;
    let rd = |pos: usize| read_checked(inputs, t.srcs[pos].0, &t.srcs[pos].1, n);
    match &t.m {
        Method::Passthrough { .. } => rd(0),
        Method::Linear { normalize } => {
            let sum: f64 = t.weights.iter().map(|&w| w as f64).sum();
            let mut acc = vec![0f32; n];
            for pos in 0..t.srcs.len() {
                let w = if *normalize {
                    t.weights[pos] as f64 / sum
                } else {
                    t.weights[pos] as f64
                } as f32;
                let x = rd(pos)?;
                for (a, &v) in acc.iter_mut().zip(&x) {
                    *a += w * v;
                }
            }
            Ok(acc)
        }
        Method::Slerp { t: tt } => {
            let a = rd(0)?;
            let b = rd(1)?;
            Ok(slerp(&a, &b, *tt))
        }
        m @ (Method::Task { .. } | Method::Ties { .. } | Method::Dare { .. }) => {
            let (scale, elect) = match *m {
                Method::Task { scale }
                | Method::Ties { scale, .. }
                | Method::Dare { scale, .. } => (scale, m.elects()),
                _ => unreachable!(),
            };
            let bp = t
                .base_pos
                .ok_or_else(|| MergeError::Config(format!("{}: no base", t.name)))?;
            let base = rd(bp)?;
            let others: Vec<usize> = (0..t.srcs.len()).filter(|&p| p != bp).collect();
            let d = Delta {
                m,
                seed,
                name: &t.name,
            };
            let mut scratch = vec![];
            let mut acc = vec![0f32; n];
            if !elect {
                for &p in &others {
                    let mut x = rd(p)?;
                    d.prep(&mut x, &base, p, &mut scratch);
                    let w = t.weights[p];
                    for (a, &v) in acc.iter_mut().zip(&x) {
                        *a += w * v;
                    }
                }
                for (a, &b) in acc.iter_mut().zip(&base) {
                    *a = b + scale * *a;
                }
                return Ok(acc);
            }
            // sign election: pass 1 elects the sign from the weighted sum, pass 2 averages the agreeing deltas
            for &p in &others {
                let mut x = rd(p)?;
                d.prep(&mut x, &base, p, &mut scratch);
                let w = t.weights[p];
                for (a, &v) in acc.iter_mut().zip(&x) {
                    *a += w * v;
                }
            }
            let mut num = vec![0f32; n];
            let mut den = vec![0f32; n];
            for &p in &others {
                let mut x = rd(p)?;
                d.prep(&mut x, &base, p, &mut scratch);
                let w = t.weights[p];
                for i in 0..n {
                    let v = x[i];
                    if v != 0.0 && acc[i] != 0.0 && (v > 0.0) == (acc[i] > 0.0) {
                        num[i] += w * v;
                        den[i] += w;
                    }
                }
            }
            for i in 0..n {
                acc[i] = base[i]
                    + if den[i] > 0.0 {
                        scale * num[i] / den[i]
                    } else {
                        0.0
                    };
            }
            Ok(acc)
        }
    }
}

// ------------------------------------------------------------------------------ writing

/// Streams one safetensors file whose header is written first (offsets are known from shapes).
struct StWriter {
    w: BufWriter<File>,
    dtype: OutDType,
    expected: u64,
    written: u64,
}
impl StWriter {
    fn create(
        path: &Path,
        entries: &[(&str, &[u64], u64)],
        dtype: OutDType,
        meta: &BTreeMap<String, String>,
    ) -> Result<StWriter, MergeError> {
        let mut h = Map::new();
        let mut md = Map::new();
        md.insert("format".into(), json!("pt"));
        for (k, v) in meta {
            md.insert(k.clone(), json!(v));
        }
        h.insert("__metadata__".into(), Value::Object(md));
        let mut off = 0u64;
        for (name, shape, elems) in entries {
            let len = elems
                .checked_mul(dtype.bytes())
                .ok_or_else(|| MergeError::Config("output too large".into()))?;
            h.insert(
                (*name).into(),
                json!({"dtype": dtype.st_name(), "shape": shape, "data_offsets": [off, off + len]}),
            );
            off += len;
        }
        let mut hb =
            serde_json::to_vec(&Value::Object(h)).map_err(|e| MergeError::Config(e.to_string()))?;
        while hb.len() % 8 != 0 {
            hb.push(b' ');
        }
        // exclusive creation: never truncates or follows anything that is already there
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o644);
        }
        let mut w = BufWriter::with_capacity(1 << 20, o.open(path)?);
        w.write_all(&(hb.len() as u64).to_le_bytes())?;
        w.write_all(&hb)?;
        Ok(StWriter {
            w,
            dtype,
            expected: off,
            written: 0,
        })
    }
    fn put(&mut self, name: &str, data: &[f32]) -> Result<(), MergeError> {
        let mut buf = Vec::with_capacity(1 << 16);
        for chunk in data.chunks(1 << 14) {
            buf.clear();
            for &x in chunk {
                if !x.is_finite() {
                    return Err(MergeError::NonFinite {
                        tensor: name.to_string(),
                        whence: "the merged result".into(),
                    });
                }
                match self.dtype {
                    OutDType::F32 => buf.extend(x.to_le_bytes()),
                    OutDType::BF16 => buf.extend(f32_to_bf16(x).to_le_bytes()),
                    OutDType::F16 => {
                        if x.abs() > 65504.0 {
                            return Err(MergeError::Overflow {
                                tensor: name.to_string(),
                                value: x,
                            });
                        }
                        buf.extend(f32_to_f16(x).to_le_bytes());
                    }
                }
            }
            self.w.write_all(&buf)?;
            self.written += buf.len() as u64;
        }
        Ok(())
    }
    fn finish(mut self) -> Result<u64, MergeError> {
        if self.written != self.expected {
            return Err(MergeError::Load(format!(
                "wrote {} bytes, header promised {}",
                self.written, self.expected
            )));
        }
        self.w.flush()?;
        self.w.get_ref().sync_all()?;
        Ok(self.written)
    }
}

/// Write named f32 tensors as one safetensors file (used by the demo and the tests; also handy
/// for building small fixtures). Returns the number of data bytes written.
pub fn write_safetensors(
    path: &Path,
    tensors: &[(String, Vec<u64>, Vec<f32>)],
    dtype: OutDType,
    metadata: &BTreeMap<String, String>,
) -> Result<u64, MergeError> {
    let mut sorted: Vec<&(String, Vec<u64>, Vec<f32>)> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    for t in &sorted {
        if t.1.iter().try_fold(1u64, |a, &d| a.checked_mul(d)) != Some(t.2.len() as u64) {
            return mis(format!("{}: shape does not match the data length", t.0));
        }
    }
    let entries: Vec<(&str, &[u64], u64)> = sorted
        .iter()
        .map(|t| (t.0.as_str(), t.1.as_slice(), t.2.len() as u64))
        .collect();
    let mut w = StWriter::create(path, &entries, dtype, metadata)?;
    for t in &sorted {
        w.put(&t.0, &t.2)?;
    }
    w.finish()
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MergeStats {
    pub tensors: usize,
    pub elements: u64,
    pub bytes_written: u64,
    pub max_tensor_elems: u64,
    /// Accounted (not measured) peak of live tensor-sized f32 values.
    pub peak_f32: u64,
    pub out_layers: Option<usize>,
}

pub struct Progress<'a> {
    pub done: usize,
    pub total: usize,
    pub tensor: &'a str,
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

fn is_symlink(p: &Path) -> bool {
    std::fs::symlink_metadata(p).is_ok_and(|m| m.file_type().is_symlink())
}

/// Merge `inputs` into the safetensors file `out` following `cfg`. The output is written to
/// `<out>.part` and renamed on success; a failure removes the partial file.
pub fn merge(
    inputs: &[Input],
    cfg: &MergeConfig,
    out: &Path,
    progress: &mut dyn FnMut(&Progress),
) -> Result<MergeStats, MergeError> {
    let plan = plan(inputs, cfg)?;
    if out.is_dir() {
        return cfg_err("the output path is a folder; give a file name");
    }
    for i in inputs {
        if let Some(p) = &i.path {
            // a folder input holds shards: refuse an output inside it as well
            if same_file(p, out)
                || (p.is_dir()
                    && out.parent().is_some_and(|d| {
                        same_file(
                            p,
                            if d.as_os_str().is_empty() {
                                Path::new(".")
                            } else {
                                d
                            },
                        )
                    }))
            {
                return cfg_err(format!(
                    "the output {} would overwrite or sit inside the input {}",
                    out.display(),
                    i.label
                ));
            }
        }
    }
    if is_symlink(out) {
        return cfg_err(format!(
            "the output {} is a symbolic link; give a real file name",
            out.display()
        ));
    }
    if out.exists() && !cfg.overwrite {
        return cfg_err(format!(
            "{} already exists (set \"overwrite\": true to replace it)",
            out.display()
        ));
    }
    let part = {
        let mut n = out
            .file_name()
            .map(|s| s.to_os_string())
            .ok_or_else(|| MergeError::Config("the output needs a file name".into()))?;
        n.push(".part");
        out.with_file_name(n)
    };
    if is_symlink(&part) {
        return cfg_err(format!(
            "{} is a symbolic link; remove it first",
            part.display()
        ));
    }
    // a regular leftover from an interrupted run is ours to replace; the creation below is exclusive
    if part.is_file() {
        let _ = std::fs::remove_file(&part);
    }
    let mut meta = BTreeMap::new();
    meta.insert(
        "merged_by".to_string(),
        "companion-engine merge".to_string(),
    );
    meta.insert(
        "merge_method".to_string(),
        if cfg.slices.is_empty() {
            cfg.method.name().to_string()
        } else {
            "slices".to_string()
        },
    );
    if let Some(l) = plan.out_layers {
        meta.insert("merged_layers".to_string(), l.to_string());
    }
    let result = (|| {
        let entries: Vec<(&str, &[u64], u64)> = plan
            .tensors
            .iter()
            .map(|t| (t.name.as_str(), t.shape.as_slice(), t.elems as u64))
            .collect();
        let mut w = StWriter::create(&part, &entries, cfg.dtype, &meta)?;
        let total = plan.tensors.len();
        for (i, t) in plan.tensors.iter().enumerate() {
            let data = run_tensor(inputs, t, cfg.seed)?;
            w.put(&t.name, &data)?;
            drop(data);
            progress(&Progress {
                done: i + 1,
                total,
                tensor: &t.name,
            });
        }
        w.finish()
    })();
    match result {
        Ok(bytes) => {
            // decided again now, at write time: the plan-time check is long past (a merge takes minutes)
            let cleanup = |e: MergeError| {
                let _ = std::fs::remove_file(&part);
                e
            };
            if is_symlink(out) {
                return Err(cleanup(MergeError::Config(format!(
                    "the output {} became a symbolic link",
                    out.display()
                ))));
            }
            if out.is_dir() {
                return Err(cleanup(MergeError::Config(format!(
                    "the output {} became a folder",
                    out.display()
                ))));
            }
            if cfg.overwrite {
                std::fs::rename(&part, out).inspect_err(|_| {
                    let _ = std::fs::remove_file(&part);
                })?;
            } else {
                // hard_link fails atomically if `out` exists, so a file created meanwhile is never replaced
                match std::fs::hard_link(&part, out) {
                    Ok(()) => {
                        let _ = std::fs::remove_file(&part);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        return Err(cleanup(MergeError::Config(format!(
                            "{} already exists (set \"overwrite\": true to replace it)",
                            out.display()
                        ))));
                    }
                    // a filesystem without hard links: check, then rename
                    Err(_) => {
                        if out.exists() {
                            return Err(cleanup(MergeError::Config(format!(
                                "{} already exists (set \"overwrite\": true to replace it)",
                                out.display()
                            ))));
                        }
                        std::fs::rename(&part, out).inspect_err(|_| {
                            let _ = std::fs::remove_file(&part);
                        })?;
                    }
                }
            }
            Ok(MergeStats {
                tensors: plan.tensors.len(),
                elements: plan.total_elems,
                bytes_written: bytes,
                max_tensor_elems: plan.max_tensor_elems,
                peak_f32: plan.peak_f32,
                out_layers: plan.out_layers,
            })
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

// ------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_write::{GgufWriter, MetaValue, TensorDesc};
    use crate::quant::DType;
    use std::sync::Mutex;

    struct Mem {
        t: BTreeMap<String, (Vec<u64>, Vec<f32>)>,
        log: Mutex<Vec<String>>,
    }
    impl Mem {
        fn new(items: &[(&str, &[u64], Vec<f32>)]) -> Arc<Mem> {
            Arc::new(Mem {
                t: items
                    .iter()
                    .map(|(n, s, d)| (n.to_string(), (s.to_vec(), d.clone())))
                    .collect(),
                log: Mutex::new(vec![]),
            })
        }
    }
    impl TensorSource for Mem {
        fn read(&self, name: &str) -> Result<Vec<f32>, crate::loader::LoadError> {
            self.log.lock().unwrap().push(name.to_string());
            self.t
                .get(name)
                .map(|x| x.1.clone())
                .ok_or_else(|| crate::loader::LoadError::Missing(name.into()))
        }
        fn has(&self, name: &str) -> bool {
            self.t.contains_key(name)
        }
        fn infos(&self) -> Vec<crate::gguf::TensorInfo> {
            self.t
                .iter()
                .map(|(n, (s, _))| crate::gguf::TensorInfo {
                    name: n.clone(),
                    dims: s.clone(),
                    dtype: DType::F32,
                    offset: 0,
                })
                .collect()
        }
    }
    fn inp(m: &Arc<Mem>, label: &str) -> Input {
        Input::from_source(label, m.clone(), Layout::Numpy, None)
    }
    fn tmp(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "engine-merge-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn run(inputs: &[Input], cfg: &MergeConfig) -> Result<BTreeMap<String, Vec<f32>>, MergeError> {
        let p = plan(inputs, cfg)?;
        let mut out = BTreeMap::new();
        for t in &p.tensors {
            out.insert(t.name.clone(), run_tensor(inputs, t, cfg.seed)?);
        }
        Ok(out)
    }
    fn one(a: Vec<f32>, b: Vec<f32>) -> (Arc<Mem>, Arc<Mem>) {
        let n = a.len() as u64;
        (Mem::new(&[("w", &[n], a)]), Mem::new(&[("w", &[n], b)]))
    }
    fn cfg(m: Method) -> MergeConfig {
        MergeConfig {
            method: m,
            ..Default::default()
        }
    }
    fn rand_vec(seed: u64, n: usize) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| r.f32() * 2.0 - 1.0).collect()
    }

    #[test]
    fn glob_and_layer_split() {
        assert!(glob_match(
            "*norm*",
            "model.layers.3.input_layernorm.weight"
        ));
        assert!(glob_match("model.*.weight", "model.embed.weight"));
        assert!(!glob_match("model.*.bias", "model.embed.weight"));
        assert!(
            glob_match("*", "") && glob_match("a*b*c", "aXXbYYc") && !glob_match("a*b*c", "aXXbYY")
        );
        assert_eq!(
            layer_split("model.layers.12.mlp.up_proj.weight"),
            Some(("model.layers.".into(), 12, ".mlp.up_proj.weight".into()))
        );
        assert_eq!(
            layer_split("blk.0.attn_q.weight"),
            Some(("blk.".into(), 0, ".attn_q.weight".into()))
        );
        assert_eq!(layer_split("model.embed_tokens.weight"), None);
        assert_eq!(layer_split("model.layers.x.weight"), None);
        assert_eq!(layer_split("sublayers.3.w"), None);
        assert_eq!(layer_split("layers.7"), None);
    }

    #[test]
    fn linear_endpoints_are_exact_and_equal_weights_average() {
        let (a, b) = (rand_vec(1, 50), rand_vec(2, 50));
        let (ma, mb) = one(a.clone(), b.clone());
        let ins = [inp(&ma, "a"), inp(&mb, "b")];
        let mut c = cfg(Method::Linear { normalize: true });
        c.weights = vec![1.0, 0.0];
        assert_eq!(run(&ins, &c).unwrap()["w"], a);
        c.weights = vec![0.0, 1.0];
        assert_eq!(run(&ins, &c).unwrap()["w"], b);
        c.weights = vec![];
        let m = &run(&ins, &c).unwrap()["w"];
        for i in 0..50 {
            assert!((m[i] - (a[i] + b[i]) / 2.0).abs() < 1e-6);
        }
        c.weights = vec![3.0, 1.0]; // normalised to .75/.25
        let m = &run(&ins, &c).unwrap()["w"];
        for i in 0..50 {
            assert!((m[i] - (0.75 * a[i] + 0.25 * b[i])).abs() < 1e-6);
        }
        c.method = Method::Linear { normalize: false };
        c.weights = vec![2.0, -1.0];
        let m = &run(&ins, &c).unwrap()["w"];
        for i in 0..50 {
            assert!((m[i] - (2.0 * a[i] - b[i])).abs() < 1e-6);
        }
        c.method = Method::Linear { normalize: true };
        c.weights = vec![1.0, -1.0];
        assert!(
            matches!(plan(&ins, &c), Err(MergeError::Config(_))),
            "weights summing to zero cannot be normalised"
        );
    }

    #[test]
    fn slerp_endpoints_midpoint_and_degenerate_cases() {
        let (a, b) = (vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]);
        assert_eq!(slerp(&a, &b, 0.0), a);
        assert_eq!(slerp(&a, &b, 1.0), b);
        let m = slerp(&a, &b, 0.5);
        let norm = m.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "slerp keeps the norm of orthogonal unit vectors, got {norm}"
        );
        assert!(
            (m[0] - m[1]).abs() < 1e-6 && (m[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5
        );
        // lerp would shrink the norm to 0.707
        let l = lerp(&a, &b, 0.5);
        assert!(l.iter().map(|x| x * x).sum::<f32>().sqrt() < 0.72);
        // parallel, opposite-ish, zero: finite results
        for (x, y) in [
            (vec![1.0, 2.0], vec![2.0, 4.0]),
            (vec![0.0, 0.0], vec![1.0, 1.0]),
            (vec![1.0, 0.0], vec![-1.0, 1e-5]),
        ] {
            for t in [0.25, 0.5, 0.75] {
                assert!(slerp(&x, &y, t).iter().all(|v| v.is_finite()));
            }
        }
        // through the planner: exactly two inputs
        let (ma, mb) = one(a.clone(), b.clone());
        let ins = [inp(&ma, "a"), inp(&mb, "b")];
        let c = cfg(Method::Slerp { t: 0.5 });
        assert!((run(&ins, &c).unwrap()["w"][0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
        let three = [inp(&ma, "a"), inp(&mb, "b"), inp(&ma, "c")];
        assert!(matches!(plan(&three, &c), Err(MergeError::Config(_))));
    }

    #[test]
    fn task_arithmetic_adds_scaled_deltas() {
        let base = Mem::new(&[("w", &[3], vec![1.0, 1.0, 1.0])]);
        let x = Mem::new(&[("w", &[3], vec![2.0, 1.0, 0.0])]);
        let y = Mem::new(&[("w", &[3], vec![1.0, 3.0, 1.0])]);
        let ins = [inp(&base, "base"), inp(&x, "x"), inp(&y, "y")];
        let mut c = cfg(Method::Task { scale: 1.0 });
        c.base = Some(0);
        assert_eq!(run(&ins, &c).unwrap()["w"], vec![2.0, 3.0, 0.0]);
        c.method = Method::Task { scale: 0.5 };
        c.weights = vec![0.0, 1.0, 2.0];
        assert_eq!(run(&ins, &c).unwrap()["w"], vec![1.5, 3.0, 0.5]);
        c.base = None;
        assert!(
            matches!(plan(&ins, &c), Err(MergeError::Config(_))),
            "task arithmetic needs a base"
        );
        c.base = Some(9);
        assert!(matches!(plan(&ins, &c), Err(MergeError::Config(_))));
    }

    #[test]
    fn trim_keeps_exactly_the_top_k() {
        let mut s = vec![];
        let mut d = vec![0.1, -5.0, 3.0, -0.2, 4.0, 0.05];
        trim_top(&mut d, 3, &mut s);
        assert_eq!(d, vec![0.0, -5.0, 3.0, 0.0, 4.0, 0.0]);
        let mut d = vec![1.0, -1.0, 1.0, -1.0, 0.5];
        trim_top(&mut d, 2, &mut s); // ties at the threshold: lowest indices win
        assert_eq!(d, vec![1.0, -1.0, 0.0, 0.0, 0.0]);
        let mut d = vec![1.0, 2.0];
        trim_top(&mut d, 5, &mut s);
        assert_eq!(d, vec![1.0, 2.0]);
        trim_top(&mut d, 0, &mut s);
        assert_eq!(d, vec![0.0, 0.0]);
        let mut r = Rng::new(3);
        let mut big: Vec<f32> = (0..1000).map(|_| r.f32() - 0.5).collect();
        trim_top(&mut big, 100, &mut s);
        assert_eq!(big.iter().filter(|x| **x != 0.0).count(), 100);
    }

    #[test]
    fn ties_trims_elects_a_sign_and_averages_the_agreeing_deltas() {
        // base 0; deltas: x = [4, 1, -3, 0.1], y = [2, -2, -1, 0.1]
        let base = Mem::new(&[("w", &[4], vec![0.0; 4])]);
        let x = Mem::new(&[("w", &[4], vec![4.0, 1.0, -3.0, 0.1])]);
        let y = Mem::new(&[("w", &[4], vec![2.0, -2.0, -1.0, 0.1])]);
        let ins = [inp(&base, "b"), inp(&x, "x"), inp(&y, "y")];
        let mut c = cfg(Method::Ties {
            density: 1.0,
            scale: 1.0,
        });
        c.base = Some(0);
        let out = &run(&ins, &c).unwrap()["w"];
        assert!(
            (out[0] - 3.0).abs() < 1e-6,
            "both positive: mean of 4 and 2"
        );
        assert!(
            (out[1] - (-2.0)).abs() < 1e-6,
            "1 and -2 sum to negative: only -2 agrees"
        );
        assert!((out[2] - (-2.0)).abs() < 1e-6);
        assert!((out[3] - 0.1).abs() < 1e-6);
        // density 0.5 keeps the two largest of four per delta: x -> 4, -3 ; y -> 2, -2
        c.method = Method::Ties {
            density: 0.5,
            scale: 1.0,
        };
        let out = &run(&ins, &c).unwrap()["w"];
        assert_eq!(out, &vec![3.0, -2.0, -3.0, 0.0]);
        // identical inputs with density 1 give base + delta exactly
        let z = Mem::new(&[("w", &[4], vec![1.0, -2.0, 3.0, 4.0])]);
        let ins2 = [inp(&base, "b"), inp(&z, "z1"), inp(&z, "z2")];
        c.method = Method::Ties {
            density: 1.0,
            scale: 1.0,
        };
        assert_eq!(run(&ins2, &c).unwrap()["w"], vec![1.0, -2.0, 3.0, 4.0]);
        // scale
        c.method = Method::Ties {
            density: 1.0,
            scale: 0.5,
        };
        assert_eq!(run(&ins2, &c).unwrap()["w"], vec![0.5, -1.0, 1.5, 2.0]);
        // sign election needs non-negative weights
        c.weights = vec![1.0, -1.0, 1.0];
        assert!(matches!(plan(&ins2, &c), Err(MergeError::Config(_))));
    }

    #[test]
    fn dare_is_seeded_unbiased_and_density_one_equals_task_arithmetic() {
        let n = 20_000;
        let base = Mem::new(&[("w", &[n as u64], vec![0.0; n])]);
        let x = Mem::new(&[("w", &[n as u64], vec![1.0; n])]);
        let ins = [inp(&base, "b"), inp(&x, "x")];
        let mut c = cfg(Method::Dare {
            density: 0.3,
            scale: 1.0,
            sign_election: false,
        });
        c.base = Some(0);
        c.seed = 7;
        let a = run(&ins, &c).unwrap()["w"].clone();
        let again = run(&ins, &c).unwrap()["w"].clone();
        assert_eq!(a, again, "same seed, same result");
        c.seed = 8;
        let other = run(&ins, &c).unwrap()["w"].clone();
        assert_ne!(a, other);
        let kept = a.iter().filter(|v| **v != 0.0).count() as f64 / n as f64;
        assert!((kept - 0.3).abs() < 0.02, "kept fraction {kept}");
        assert!(
            a.iter().all(|v| *v == 0.0 || (v - 1.0 / 0.3).abs() < 1e-5),
            "survivors are rescaled by 1/density"
        );
        let mean = a.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
        assert!(
            (mean - 1.0).abs() < 0.05,
            "expected delta is preserved, mean {mean}"
        );
        c.method = Method::Dare {
            density: 1.0,
            scale: 1.0,
            sign_election: false,
        };
        let full = run(&ins, &c).unwrap()["w"].clone();
        c.method = Method::Task { scale: 1.0 };
        assert_eq!(full, run(&ins, &c).unwrap()["w"]);
        // with sign election and one delta input it keeps that delta's survivors
        c.method = Method::Dare {
            density: 0.5,
            scale: 1.0,
            sign_election: true,
        };
        let e = run(&ins, &c).unwrap()["w"].clone();
        assert!(e.iter().all(|v| *v == 0.0 || (*v - 2.0).abs() < 1e-6));
        // the mask depends on the tensor name, not on tensor order
        let two = Mem::new(&[("u", &[4], vec![0.0; 4]), ("w", &[n as u64], vec![0.0; n])]);
        let two_x = Mem::new(&[("u", &[4], vec![1.0; 4]), ("w", &[n as u64], vec![1.0; n])]);
        c.method = Method::Dare {
            density: 0.3,
            scale: 1.0,
            sign_election: false,
        };
        c.seed = 7;
        let got = run(&[inp(&two, "b"), inp(&two_x, "x")], &c).unwrap()["w"].clone();
        assert_eq!(got, a);
    }

    #[test]
    fn rules_override_method_and_weights_by_pattern() {
        let a = Mem::new(&[
            ("layers.0.norm.weight", &[2], vec![1.0, 1.0]),
            ("layers.0.mlp.weight", &[2], vec![1.0, 1.0]),
        ]);
        let b = Mem::new(&[
            ("layers.0.norm.weight", &[2], vec![3.0, 3.0]),
            ("layers.0.mlp.weight", &[2], vec![3.0, 3.0]),
        ]);
        let ins = [inp(&a, "a"), inp(&b, "b")];
        let mut c = cfg(Method::Linear { normalize: true });
        c.rules = vec![
            Rule {
                pattern: "*norm*".into(),
                method: Some(Method::Passthrough { input: 1 }),
                weights: None,
            },
            Rule {
                pattern: "*mlp*".into(),
                method: None,
                weights: Some(vec![1.0, 0.0]),
            },
        ];
        let o = run(&ins, &c).unwrap();
        assert_eq!(o["layers.0.norm.weight"], vec![3.0, 3.0]);
        assert_eq!(o["layers.0.mlp.weight"], vec![1.0, 1.0]);
        c.rules[1].weights = Some(vec![1.0]);
        assert!(
            matches!(plan(&ins, &c), Err(MergeError::Config(_))),
            "wrong weight count is an error"
        );
    }

    #[test]
    fn mismatches_are_reported_before_anything_is_read() {
        let a = Mem::new(&[("w", &[2, 3], vec![0.0; 6]), ("v", &[2], vec![0.0; 2])]);
        let shape = Mem::new(&[("w", &[3, 2], vec![0.0; 6]), ("v", &[2], vec![0.0; 2])]);
        let missing = Mem::new(&[("w", &[2, 3], vec![0.0; 6])]);
        let extra = Mem::new(&[
            ("w", &[2, 3], vec![0.0; 6]),
            ("v", &[2], vec![0.0; 2]),
            ("z", &[1], vec![0.0]),
        ]);
        let c = cfg(Method::Linear { normalize: true });
        let e = plan(&[inp(&a, "a"), inp(&shape, "s")], &c).unwrap_err();
        assert!(
            matches!(&e, MergeError::Mismatch(m) if m.contains("[2, 3]") && m.contains("[3, 2]")),
            "{e}"
        );
        assert!(matches!(
            plan(&[inp(&a, "a"), inp(&missing, "m")], &c),
            Err(MergeError::Mismatch(_))
        ));
        assert!(matches!(
            plan(&[inp(&a, "a"), inp(&extra, "e")], &c),
            Err(MergeError::Mismatch(_))
        ));
        assert!(matches!(plan(&[], &c), Err(MergeError::Mismatch(_))));
        let mut lenient = c.clone();
        lenient.lenient = true;
        let p = plan(&[inp(&a, "a"), inp(&missing, "m")], &lenient).unwrap();
        assert_eq!(p.tensors.len(), 1, "lenient merges only the common tensors");
        // architecture and layout
        let x = Input::from_source("x", a.clone(), Layout::Numpy, Some("llama".into()));
        let y = Input::from_source("y", a.clone(), Layout::Numpy, Some("qwen2".into()));
        assert!(
            matches!(plan(&[x, y], &c), Err(MergeError::Mismatch(m)) if m.contains("llama") && m.contains("qwen2"))
        );
        let g = Input::from_source("g", a.clone(), Layout::Gguf, None);
        assert!(
            matches!(plan(&[inp(&a, "a"), g], &c), Err(MergeError::Mismatch(m)) if m.contains("different file kinds"))
        );
        // a source that cannot list tensors
        struct Blind;
        impl TensorSource for Blind {
            fn read(&self, n: &str) -> Result<Vec<f32>, crate::loader::LoadError> {
                Err(crate::loader::LoadError::Missing(n.into()))
            }
            fn has(&self, _: &str) -> bool {
                false
            }
        }
        assert!(matches!(
            plan(
                &[Input::from_source(
                    "b",
                    Arc::new(Blind),
                    Layout::Numpy,
                    None
                )],
                &c
            ),
            Err(MergeError::Unsupported(_))
        ));
    }

    #[test]
    fn non_finite_inputs_and_f16_overflow_are_errors() {
        let ok = Mem::new(&[("w", &[2], vec![1.0, 2.0])]);
        let nan = Mem::new(&[("w", &[2], vec![1.0, f32::NAN])]);
        let inf = Mem::new(&[("w", &[2], vec![f32::INFINITY, 0.0])]);
        let c = cfg(Method::Linear { normalize: true });
        assert!(matches!(
            run(&[inp(&ok, "a"), inp(&nan, "b")], &c),
            Err(MergeError::NonFinite { .. })
        ));
        assert!(matches!(
            run(&[inp(&inf, "a"), inp(&ok, "b")], &c),
            Err(MergeError::NonFinite { .. })
        ));
        let d = tmp("ovf");
        let big = Mem::new(&[("w", &[2], vec![1e6, 1.0])]);
        let mut c16 = c.clone();
        c16.dtype = OutDType::F16;
        let r = merge(
            &[inp(&big, "a")],
            &c16,
            &d.join("o.safetensors"),
            &mut |_| {},
        );
        assert!(matches!(r, Err(MergeError::Overflow { .. })), "{r:?}");
        assert!(
            !d.join("o.safetensors").exists() && !d.join("o.safetensors.part").exists(),
            "no partial file is left behind"
        );
        c16.dtype = OutDType::BF16;
        merge(
            &[inp(&big, "a")],
            &c16,
            &d.join("o.safetensors"),
            &mut |_| {},
        )
        .unwrap();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn peak_memory_is_a_few_tensors_not_the_model() {
        // 12 tensors of 1000 values, three inputs; the log shows strictly sequential tensor access
        let make = |s: u64| {
            let items: Vec<(String, Vec<u64>, Vec<f32>)> = (0..12)
                .map(|i| {
                    (
                        format!("t{i:02}"),
                        vec![1000u64],
                        rand_vec(s * 100 + i, 1000),
                    )
                })
                .collect();
            Arc::new(Mem {
                t: items.into_iter().map(|(n, s, d)| (n, (s, d))).collect(),
                log: Mutex::new(vec![]),
            })
        };
        let (a, b, c3) = (make(1), make(2), make(3));
        let ins = [inp(&a, "a"), inp(&b, "b"), inp(&c3, "c")];
        let c = cfg(Method::Linear { normalize: true });
        let p = plan(&ins, &c).unwrap();
        assert_eq!(p.total_elems, 12_000);
        assert_eq!(p.max_tensor_elems, 1000);
        assert_eq!(
            p.peak_f32, 2000,
            "linear holds an accumulator and one input"
        );
        assert!(p.peak_f32 * 6 <= p.total_elems);
        let ties = MergeConfig {
            method: Method::Ties {
                density: 0.5,
                scale: 1.0,
            },
            base: Some(0),
            ..Default::default()
        };
        assert_eq!(plan(&ins, &ties).unwrap().peak_f32, 6000);
        // a memory limit turns into an error
        let mut tight = ties.clone();
        tight.max_peak_bytes = Some(10_000);
        assert!(matches!(
            plan(&ins, &tight),
            Err(MergeError::OverBudget {
                need_bytes: 24_000,
                limit_bytes: 10_000
            })
        ));
        run(&ins, &c).unwrap();
        let log = a.log.lock().unwrap();
        assert_eq!(log.len(), 12, "each tensor of an input is read once");
        let mut sorted = log.clone();
        sorted.sort();
        assert_eq!(*log, sorted, "tensors are visited one at a time, in order");
    }

    #[test]
    fn end_to_end_files_round_trip_in_all_output_types() {
        let d = tmp("e2e");
        let (a, b) = (rand_vec(5, 24), rand_vec(6, 24));
        let mut meta = BTreeMap::new();
        meta.insert("note".to_string(), "x".to_string());
        write_safetensors(
            &d.join("a.safetensors"),
            &[
                ("m.w".into(), vec![4, 6], a.clone()),
                ("m.b".into(), vec![4], a[..4].to_vec()),
            ],
            OutDType::F32,
            &meta,
        )
        .unwrap();
        write_safetensors(
            &d.join("b.safetensors"),
            &[
                ("m.w".into(), vec![4, 6], b.clone()),
                ("m.b".into(), vec![4], b[..4].to_vec()),
            ],
            OutDType::F32,
            &meta,
        )
        .unwrap();
        let ins = [
            Input::open(&d.join("a.safetensors")).unwrap(),
            Input::open(&d.join("b.safetensors")).unwrap(),
        ];
        for (dt, tol) in [
            (OutDType::F32, 1e-6f32),
            (OutDType::F16, 1e-3),
            (OutDType::BF16, 1e-2),
        ] {
            let mut c = cfg(Method::Linear { normalize: true });
            c.dtype = dt;
            let out = d.join(format!("out-{dt:?}.safetensors"));
            let mut seen = vec![];
            let st = merge(&ins, &c, &out, &mut |p| {
                seen.push((p.done, p.total, p.tensor.to_string()))
            })
            .unwrap();
            assert_eq!((st.tensors, st.elements), (2, 28));
            assert_eq!(st.bytes_written, 28 * dt.bytes());
            assert_eq!(seen.len(), 2);
            assert_eq!(seen[1].0, 2);
            let back = SafeTensors::open(&out).unwrap();
            let w = back.read("m.w").unwrap();
            for i in 0..24 {
                assert!(
                    (w[i] - (a[i] + b[i]) / 2.0).abs() <= tol,
                    "{dt:?} element {i}"
                );
            }
            assert_eq!(back.read("m.b").unwrap().len(), 4);
            // an existing output is protected
            assert!(
                matches!(merge(&ins, &c, &out, &mut |_| {}), Err(MergeError::Config(m)) if m.contains("already exists"))
            );
            c.overwrite = true;
            merge(&ins, &c, &out, &mut |_| {}).unwrap();
        }
        // the output may not be an input
        let mut c = cfg(Method::Linear { normalize: true });
        c.overwrite = true;
        assert!(
            matches!(merge(&ins, &c, &d.join("a.safetensors"), &mut |_| {}), Err(MergeError::Config(m)) if m.contains("overwrite"))
        );
        assert!(matches!(
            merge(&ins, &c, &d, &mut |_| {}),
            Err(MergeError::Config(_))
        ));
        // a folder input: the output may not go inside it
        let fd = d.join("shards");
        std::fs::create_dir_all(&fd).unwrap();
        std::fs::copy(d.join("a.safetensors"), fd.join("model.safetensors")).unwrap();
        let fi = [Input::open(&fd).unwrap()];
        assert!(
            matches!(merge(&fi, &c, &fd.join("merged.safetensors"), &mut |_| {}), Err(MergeError::Config(m)) if m.contains("inside"))
        );
        merge(&fi, &c, &d.join("outside.safetensors"), &mut |_| {}).unwrap();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn frankenmerge_stacks_layer_ranges_and_renumbers() {
        let model = |tag: f32, layers: usize, hidden: u64| {
            let mut items: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
                (
                    "model.embed.weight".into(),
                    vec![3, hidden],
                    vec![tag; 3 * hidden as usize],
                ),
                (
                    "model.norm.weight".into(),
                    vec![hidden],
                    vec![tag; hidden as usize],
                ),
            ];
            for l in 0..layers {
                items.push((
                    format!("model.layers.{l}.mlp.weight"),
                    vec![hidden],
                    vec![tag * 100.0 + l as f32; hidden as usize],
                ));
                items.push((
                    format!("model.layers.{l}.norm.weight"),
                    vec![hidden],
                    vec![tag * 100.0 + l as f32 + 0.5; hidden as usize],
                ));
            }
            Arc::new(Mem {
                t: items.into_iter().map(|(n, s, d)| (n, (s, d))).collect(),
                log: Mutex::new(vec![]),
            })
        };
        let (a, b) = (model(1.0, 4, 2), model(2.0, 4, 2));
        let ins = [inp(&a, "a"), inp(&b, "b")];
        let mut c = MergeConfig {
            slices: vec![
                Slice {
                    input: 0,
                    start: 0,
                    end: 2,
                },
                Slice {
                    input: 1,
                    start: 1,
                    end: 4,
                },
            ],
            ..Default::default()
        };
        let p = plan(&ins, &c).unwrap();
        assert_eq!(p.out_layers, Some(5));
        let out = run(&ins, &c).unwrap();
        assert_eq!(out.len(), 2 + 5 * 2);
        assert_eq!(out["model.layers.0.mlp.weight"], vec![100.0; 2]);
        assert_eq!(out["model.layers.1.mlp.weight"], vec![101.0; 2]);
        assert_eq!(out["model.layers.2.mlp.weight"], vec![201.0; 2]);
        assert_eq!(out["model.layers.4.norm.weight"], vec![203.5; 2]);
        assert_eq!(
            out["model.embed.weight"],
            vec![1.0; 6],
            "non-layer tensors come from the first slice's input"
        );
        c.outer_from = Some(1);
        assert_eq!(run(&ins, &c).unwrap()["model.norm.weight"], vec![2.0; 2]);
        // errors: range past the end, empty slice, different sizes, unknown input
        let bad = |s: Slice| {
            let mut c2 = c.clone();
            c2.slices = vec![s];
            plan(&ins, &c2)
        };
        assert!(matches!(
            bad(Slice {
                input: 0,
                start: 2,
                end: 9
            }),
            Err(MergeError::Mismatch(_))
        ));
        assert!(matches!(
            bad(Slice {
                input: 0,
                start: 2,
                end: 2
            }),
            Err(MergeError::Config(_))
        ));
        assert!(matches!(
            bad(Slice {
                input: 5,
                start: 0,
                end: 1
            }),
            Err(MergeError::Config(_))
        ));
        let wide = model(3.0, 4, 3);
        let mixed = [inp(&a, "a"), inp(&wide, "wide")];
        assert!(
            matches!(plan(&mixed, &c), Err(MergeError::Mismatch(m)) if m.contains("not the same size") || m.contains("hidden"))
        );
    }

    #[test]
    fn gguf_inputs_are_dequantised_and_shapes_are_normalised() {
        let d = tmp("gguf");
        let mk = |p: &Path, v: f32| {
            let f32s = |x: &[f32]| x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
            let q8: Vec<u8> = {
                let mut o = vec![];
                for _ in 0..2 {
                    o.extend(0x3c00u16.to_le_bytes());
                    o.extend((0..32).map(|i| (i as i8 - 16) as u8));
                }
                o
            };
            let descs = vec![
                TensorDesc::new("blk.0.w", &[4, 2], DType::F32),
                TensorDesc::new("q", &[32, 2], DType::Q8_0),
            ];
            let meta = vec![("general.architecture".to_string(), MetaValue::str("llama"))];
            let f = File::create(p).unwrap();
            let w = f32s(&[v; 8]);
            GgufWriter::write_all(BufWriter::new(f), &meta, descs, |i, _| {
                Ok(if i == 0 { w.clone() } else { q8.clone() })
            })
            .unwrap();
        };
        mk(&d.join("a.gguf"), 1.0);
        mk(&d.join("b.gguf"), 3.0);
        let ins = [
            Input::open(&d.join("a.gguf")).unwrap(),
            Input::open(&d.join("b.gguf")).unwrap(),
        ];
        let p = plan(&ins, &cfg(Method::Linear { normalize: true })).unwrap();
        let w = p.tensors.iter().find(|t| t.name == "blk.0.w").unwrap();
        assert_eq!(
            w.shape,
            vec![4, 2],
            "GGUF dimensions are reversed back to row-major order (the writer reverses its shape)"
        );
        let out = d.join("m.safetensors");
        let mut c = cfg(Method::Linear { normalize: true });
        c.dtype = OutDType::F32;
        merge(&ins, &c, &out, &mut |_| {}).unwrap();
        let back = SafeTensors::open(&out).unwrap();
        assert_eq!(back.read("blk.0.w").unwrap(), vec![2.0; 8]);
        let q = back.read("q").unwrap();
        assert_eq!(q.len(), 64);
        assert_eq!(
            q[0], -16.0,
            "Q8_0 dequantised, mean of two identical tensors is the same tensor"
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn json_configuration_is_strict() {
        let c = MergeConfig::from_json(r#"{"method":"ties","density":0.25,"scale":0.8,"base":0,"weights":[1,2,3],"dtype":"f16","seed":9,"lenient":true,
            "rules":[{"match":"*norm*","method":"linear","weights":[0.5,0.5]},{"match":"*embed*","method":"passthrough","input":1}],
            "slices":[{"input":0,"start":0,"end":16}]}"#).unwrap();
        assert_eq!(
            c.method,
            Method::Ties {
                density: 0.25,
                scale: 0.8
            }
        );
        assert_eq!(
            (c.base, c.dtype, c.seed, c.lenient),
            (Some(0), OutDType::F16, 9, true)
        );
        assert_eq!(c.weights, vec![1.0, 2.0, 3.0]);
        assert_eq!(c.rules[0].method, Some(Method::Linear { normalize: true }));
        assert_eq!(c.rules[1].method, Some(Method::Passthrough { input: 1 }));
        assert_eq!(
            c.slices,
            vec![Slice {
                input: 0,
                start: 0,
                end: 16
            }]
        );
        let m = MergeConfig::from_json(r#"{"method":"dare","density":0.1,"sign_election":true,"rules":[{"match":"*","method":"ties"}]}"#).unwrap();
        assert_eq!(
            m.method,
            Method::Dare {
                density: 0.1,
                scale: 1.0,
                sign_election: true
            }
        );
        assert_eq!(
            m.rules[0].method,
            Some(Method::Ties {
                density: 0.1,
                scale: 1.0
            }),
            "rules inherit the top-level parameters"
        );
        for bad in [
            r#"{"methd":"linear"}"#,
            r#"{"method":"magic"}"#,
            r#"{"method":"ties","density":0}"#,
            r#"{"method":"ties","density":1.5}"#,
            r#"{"method":"slerp","t":2}"#,
            r#"{"dtype":"f8"}"#,
            r#"{"weights":"x"}"#,
            r#"{"weights":[1,"a"]}"#,
            r#"[]"#,
            "not json",
            r#"{"rules":[{"method":"linear"}]}"#,
            r#"{"slices":[{"input":0,"start":1}]}"#,
            r#"{"slices":[{"input":0,"start":1,"end":2,"extra":1}]}"#,
            r#"{"seed":-1}"#,
            r#"{"base":-1}"#,
            r#"{"max_peak_bytes":-5}"#,
        ] {
            assert!(
                matches!(MergeConfig::from_json(bad), Err(MergeError::Config(_))),
                "{bad}"
            );
        }
        assert_eq!(
            MergeConfig::from_json("{}").unwrap(),
            MergeConfig::default()
        );
    }

    #[test]
    fn hostile_inputs_do_not_panic() {
        let c = cfg(Method::Linear { normalize: true });
        // a tensor whose declared shape disagrees with its data
        let lying = Arc::new(Mem {
            t: [("w".to_string(), (vec![10u64], vec![1.0f32; 3]))]
                .into_iter()
                .collect(),
            log: Mutex::new(vec![]),
        });
        assert!(matches!(
            run(&[inp(&lying, "l")], &c),
            Err(MergeError::Load(_))
        ));
        // absurd shapes
        let huge = Arc::new(Mem {
            t: [("w".to_string(), (vec![u64::MAX, 2], vec![]))]
                .into_iter()
                .collect(),
            log: Mutex::new(vec![]),
        });
        assert!(plan(&[inp(&huge, "h")], &c).is_err());
        // zero-sized tensors merge to zero-sized tensors
        let z = Mem::new(&[("w", &[0], vec![])]);
        assert!(run(&[inp(&z, "z"), inp(&z, "z2")], &c).unwrap()["w"].is_empty());
        // garbage files
        let d = tmp("hostile");
        std::fs::write(
            d.join("junk.safetensors"),
            b"\xff\xff\xff\xff\xff\xff\xff\x7fjunk",
        )
        .unwrap();
        assert!(Input::open(&d.join("junk.safetensors")).is_err());
        std::fs::write(d.join("junk.gguf"), b"GGUF\x03\0\0\0 not a gguf").unwrap();
        assert!(Input::open(&d.join("junk.gguf")).is_err());
        assert!(Input::open(&d.join("missing.safetensors")).is_err());
        // mismatched write shape
        assert!(
            write_safetensors(
                &d.join("x.safetensors"),
                &[("w".into(), vec![2, 2], vec![1.0; 3])],
                OutDType::F32,
                &BTreeMap::new()
            )
            .is_err()
        );
        // weights with NaN, wrong count; slerp t out of range
        assert!(matches!(
            MergeConfig::from_json(r#"{"weights":[1e999]}"#),
            Err(MergeError::Config(_))
        ));
        let mut w = c.clone();
        w.weights = vec![f32::NAN, 1.0];
        let a = Mem::new(&[("w", &[1], vec![1.0])]);
        assert!(matches!(
            plan(&[inp(&a, "a"), inp(&a, "b")], &w),
            Err(MergeError::Config(_))
        ));
        assert!(matches!(
            plan(&[inp(&a, "a")], &cfg(Method::Slerp { t: f32::NAN })),
            Err(MergeError::Config(_))
        ));
        std::fs::remove_dir_all(d).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_outputs_are_refused_and_a_file_that_appears_mid_merge_is_not_overwritten() {
        use std::os::unix::fs::symlink;
        let d = tmp("links");
        let a = Mem::new(&[("w", &[2], vec![1.0, 2.0])]);
        let ins = [inp(&a, "a")];
        let mut c = cfg(Method::Linear { normalize: true });
        // a symlinked output (even with overwrite) is refused, and its target is untouched
        let victim = d.join("victim");
        std::fs::write(&victim, b"precious").unwrap();
        let out = d.join("o.safetensors");
        symlink(&victim, &out).unwrap();
        for ow in [false, true] {
            c.overwrite = ow;
            assert!(
                matches!(merge(&ins, &c, &out, &mut |_| {}), Err(MergeError::Config(m)) if m.contains("symbolic link")),
                "overwrite={ow}"
            );
        }
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
        std::fs::remove_file(&out).unwrap();
        // a symlinked .part is refused, never written through
        let part = d.join("o.safetensors.part");
        symlink(&victim, &part).unwrap();
        assert!(
            matches!(merge(&ins, &c, &out, &mut |_| {}), Err(MergeError::Config(m)) if m.contains("symbolic link"))
        );
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
        std::fs::remove_file(&part).unwrap();
        // exclusive creation: the writer never reuses an existing path
        assert!(
            StWriter::create(
                &victim,
                &[("w", &[2u64][..], 2)],
                OutDType::F32,
                &BTreeMap::new()
            )
            .is_err()
        );
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
        // a stale regular .part from an interrupted run is replaced
        std::fs::write(&part, b"stale").unwrap();
        c.overwrite = false;
        merge(&ins, &c, &out, &mut |_| {}).unwrap();
        assert!(out.is_file() && !part.exists());
        std::fs::remove_file(&out).unwrap();
        // the overwrite decision is taken again at write time: a file created during the merge survives
        let o2 = d.join("late.safetensors");
        let late = o2.clone();
        let r = merge(&ins, &c, &o2, &mut |_| {
            std::fs::write(&late, b"someone else's").unwrap();
        });
        assert!(
            matches!(r, Err(MergeError::Config(m)) if m.contains("already exists")),
            "late file must not be replaced"
        );
        assert_eq!(std::fs::read(&o2).unwrap(), b"someone else's");
        assert!(
            !d.join("late.safetensors.part").exists(),
            "no partial file is left behind"
        );
        // ...and an output that turned into a symlink mid-merge is refused too
        let o3 = d.join("swap.safetensors");
        let swap = o3.clone();
        let v2 = victim.clone();
        c.overwrite = true;
        let r = merge(&ins, &c, &o3, &mut |_| {
            symlink(&v2, &swap).unwrap();
        });
        assert!(matches!(r, Err(MergeError::Config(m)) if m.contains("symbolic link")));
        assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
        assert!(!d.join("swap.safetensors.part").exists());
    }
}
