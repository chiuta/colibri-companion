//! Activation steering (control vectors) and a logit lens.
//!
//! * **Steering.** [`train`] reads the residual stream of a model at the last token of two sets
//!   of prompts (say, polite versus rude) and keeps, for every layer, the direction that separates
//!   them: the difference of the class means ([`Method::MeanDiff`]) or the first principal
//!   component of the paired differences ([`Method::Pca`]). A [`ControlVector`] is stored in a
//!   small versioned file stamped with the model fingerprint. [`Steering`] adds
//!   `strength[layer] * vector[layer]` to the residual stream after chosen layers while the
//!   model generates ([`forward_steered`], [`generate_steered`]); the weights are never touched.
//! * **Logit lens.** [`lens`] decodes the residual stream after every layer with the model's own
//!   final norm and output head and reports, per position, what each layer would predict, the
//!   layer from which the final answer is stable (the *commit layer*) and a summary of how
//!   concentrated the attention was.
//!
//! Everything is built on the public layer-by-layer API of [`crate::model`]
//! (`run_layers_batch`, `finish_batch`, `logits_from_hidden`): the normal decoder loop is not
//! changed and pays nothing for this module. Without steering the layer-by-layer path gives the
//! same logits, bit for bit, as [`Model::forward_batch`] on the same prompt.
//!
//! Limits, stated plainly:
//! * The model fingerprint covers architecture and tensor shapes, not weight values, so two
//!   fine-tunes of one base model share a fingerprint and a vector made on one is accepted for the other.
//! * With an accelerator backend each layer is run on its own, so the hidden state crosses to the
//!   device once per layer: correct, but slow.
//! * Attention summaries are computed from the accumulated attention mass per cache slot
//!   ([`State::track_scores`]), summed over heads and the queries of the call. They describe how
//!   concentrated attention was over the context, not per-head, per-query entropies. Sliding-window
//!   (ring) layers have no summary.
//! * After an `Err` from any function taking `&mut State`, discard that state.
//! * Nothing here was run on a real checkpoint; the tests use small synthetic models.
use crate::hub::{Sha256, hex};
use crate::model::{
    GenerateOptions, GenerateStats, Model, ModelError, PREFILL_CHUNK, Sampler, State, StopReason,
};
use serde_json::{Value, json};
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Largest hidden size a control-vector file may declare.
pub const MAX_HIDDEN: usize = 1 << 17;
/// Most layers a control-vector file may declare.
pub const MAX_LAYERS: usize = 1024;
/// Most prompts per class accepted by [`train`].
pub const MAX_PROMPTS: usize = 4096;
/// Most positions [`lens`] analyses in one call.
pub const MAX_LENS_POSITIONS: usize = 4096;
/// Most top-k entries per layer in a lens report.
pub const MAX_TOP_K: usize = 64;
const MAX_NAME: usize = 256;
const MAX_PCA_FLOATS: usize = 64 << 20;
const MAGIC: &[u8; 4] = b"CVEC";
const VERSION: u32 = 1;

#[derive(Debug, PartialEq)]
pub enum SteerError {
    Model(ModelError),
    /// A control-vector file is damaged, forged, truncated or from another version.
    Format(String),
    /// The vector was made for another model, or does not fit this one.
    Mismatch(String),
    /// A caller-supplied value is out of range.
    Param(String),
    Io(String),
}
impl std::fmt::Display for SteerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SteerError::Model(e) => write!(f, "model: {e}"),
            SteerError::Format(m) => write!(f, "control vector file: {m}"),
            SteerError::Mismatch(m) => write!(f, "control vector does not fit: {m}"),
            SteerError::Param(m) => write!(f, "bad parameter: {m}"),
            SteerError::Io(m) => write!(f, "i/o: {m}"),
        }
    }
}
impl std::error::Error for SteerError {}
impl From<ModelError> for SteerError {
    fn from(e: ModelError) -> Self {
        SteerError::Model(e)
    }
}

// ------------------------------------------------------------------ layer-by-layer runner

/// Called after each layer (and after steering): layer index, the batch hidden states
/// (`b * hidden` values), `b`, the position of the first token of the chunk, whether this is the last chunk.
type Hook<'a> = &'a mut dyn FnMut(usize, &[f32], usize, usize, bool);

/// Feed `tokens` through `model` one layer at a time, like `forward_batch` (same chunking, same
/// state bookkeeping), optionally steering and observing. Returns the hidden state of the last token.
fn run_layerwise(
    model: &Model,
    st: &mut State,
    tokens: &[usize],
    steer: Option<&Steering>,
    hook: Hook,
) -> Result<Vec<f32>, SteerError> {
    let s = &model.spec;
    let h = s.hidden as usize;
    if tokens.is_empty() {
        return Err(ModelError::NoTokens.into());
    }
    if let Some(&t) = tokens.iter().find(|&&t| t >= s.vocab as usize) {
        return Err(ModelError::Token(t).into());
    }
    if st.pos + tokens.len() > s.context as usize {
        return Err(ModelError::ContextFull(s.context as usize).into());
    }
    let layers = model.w.layers.len();
    let n_chunks = tokens.len().div_ceil(PREFILL_CHUNK);
    let mut last = vec![];
    for (ci, c) in tokens.chunks(PREFILL_CHUNK).enumerate() {
        let (b, pos0) = (c.len(), st.pos);
        let mut x = Vec::with_capacity(b * h);
        for &t in c {
            x.extend(model.embed_token(t)?);
        }
        for li in 0..layers {
            model.run_layers_batch(st, li..li + 1, &mut x, pos0)?;
            if let Some(sv) = steer {
                sv.apply(li, &mut x);
            }
            hook(li, &x, b, pos0, ci + 1 == n_chunks);
        }
        model.finish_batch(st, c);
        if ci + 1 == n_chunks {
            last = x[(b - 1) * h..b * h].to_vec();
        }
    }
    Ok(last)
}

fn no_hook(_: usize, _: &[f32], _: usize, _: usize, _: bool) {}

/// The residual stream after every layer at the last token of `tokens`: `layers` vectors of `hidden` values.
pub fn capture_last(model: &Model, tokens: &[usize]) -> Result<Vec<Vec<f32>>, SteerError> {
    let h = model.spec.hidden as usize;
    let mut out = vec![vec![0.0f32; h]; model.w.layers.len()];
    let mut st = model.new_state();
    let mut hook = |li: usize, x: &[f32], b: usize, _: usize, last: bool| {
        if last {
            out[li].copy_from_slice(&x[(b - 1) * h..b * h]);
        }
    };
    run_layerwise(model, &mut st, tokens, None, &mut hook)?;
    if out.iter().flatten().any(|v| !v.is_finite()) {
        return Err(SteerError::Param(
            "the model produced non-finite activations".into(),
        ));
    }
    Ok(out)
}

// ------------------------------------------------------------------ control vectors

/// How [`train`] turns activations into a direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Mean activation of the positive prompts minus that of the negative prompts.
    MeanDiff,
    /// First principal component (uncentred, so it follows the mean difference) of the per-pair
    /// differences, scaled by the mean projection. Needs as many negative as positive prompts, paired in order.
    Pca,
    /// A vector assembled by hand or with [`ControlVector::combine`].
    Custom,
}
impl Method {
    fn code(self) -> u8 {
        match self {
            Method::MeanDiff => 0,
            Method::Pca => 1,
            Method::Custom => 2,
        }
    }
    fn from_code(c: u8) -> Option<Method> {
        match c {
            0 => Some(Method::MeanDiff),
            1 => Some(Method::Pca),
            2 => Some(Method::Custom),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Method::MeanDiff => "mean-diff",
            Method::Pca => "pca",
            Method::Custom => "custom",
        }
    }
}

/// One direction per layer in the residual stream, to be added after that layer's output.
/// Natural scale: strength 1.0 adds the measured difference between the two prompt classes.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlVector {
    pub name: String,
    pub method: Method,
    /// Fingerprint of the model it was made on ([`Model::fingerprint`]).
    pub fingerprint: [u8; 32],
    pub hidden: usize,
    /// Prompts used per class (0 for hand-made vectors).
    pub positive_prompts: usize,
    pub negative_prompts: usize,
    /// Index = the layer after which the vector is added; `None` = nothing to add there.
    pub layers: Vec<Option<Vec<f32>>>,
}

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

impl ControlVector {
    /// Wrap hand-made per-layer vectors for `model` (for example an embedding row, to push towards a word).
    pub fn from_layers(
        model: &Model,
        name: &str,
        layers: Vec<Option<Vec<f32>>>,
    ) -> Result<ControlVector, SteerError> {
        let h = model.spec.hidden as usize;
        if name.len() > MAX_NAME {
            return Err(SteerError::Param("name longer than 256 bytes".into()));
        }
        if layers.len() != model.w.layers.len() {
            return Err(SteerError::Mismatch(format!(
                "{} layers given, the model has {}",
                layers.len(),
                model.w.layers.len()
            )));
        }
        for v in layers.iter().flatten() {
            if v.len() != h {
                return Err(SteerError::Mismatch(format!(
                    "a vector has {} values, the model's hidden size is {h}",
                    v.len()
                )));
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err(SteerError::Param("non-finite value".into()));
            }
        }
        Ok(ControlVector {
            name: name.into(),
            method: Method::Custom,
            fingerprint: model.fingerprint(),
            hidden: h,
            positive_prompts: 0,
            negative_prompts: 0,
            layers,
        })
    }

    /// Keep only the layers in `range` (a vector tuned for the middle of the network, say).
    pub fn restrict(&mut self, range: Range<usize>) {
        for (i, l) in self.layers.iter_mut().enumerate() {
            if !range.contains(&i) {
                *l = None;
            }
        }
    }

    /// Weighted sum of vectors made on the same model, for example `formal - casual + brief`.
    pub fn combine(
        parts: &[(&ControlVector, f32)],
        name: &str,
    ) -> Result<ControlVector, SteerError> {
        let first = parts
            .first()
            .ok_or_else(|| SteerError::Param("nothing to combine".into()))?
            .0;
        if name.len() > MAX_NAME {
            return Err(SteerError::Param("name longer than 256 bytes".into()));
        }
        let mut layers: Vec<Option<Vec<f32>>> = vec![None; first.layers.len()];
        for (cv, w) in parts {
            if cv.fingerprint != first.fingerprint
                || cv.hidden != first.hidden
                || cv.layers.len() != first.layers.len()
            {
                return Err(SteerError::Mismatch("vectors from different models".into()));
            }
            if !w.is_finite() {
                return Err(SteerError::Param("non-finite weight".into()));
            }
            for (slot, v) in layers.iter_mut().zip(&cv.layers) {
                if let Some(v) = v {
                    let acc = slot.get_or_insert_with(|| vec![0.0; first.hidden]);
                    acc.iter_mut().zip(v).for_each(|(a, b)| *a += w * b);
                }
            }
        }
        if layers.iter().flatten().flatten().any(|x| !x.is_finite()) {
            return Err(SteerError::Param("the sum overflowed".into()));
        }
        Ok(ControlVector {
            name: name.into(),
            method: Method::Custom,
            fingerprint: first.fingerprint,
            hidden: first.hidden,
            positive_prompts: 0,
            negative_prompts: 0,
            layers,
        })
    }

    /// `Ok` when this vector was made for `model` (same fingerprint, hidden size and layer count).
    pub fn check_model(&self, model: &Model) -> Result<(), SteerError> {
        if self.fingerprint != model.fingerprint() {
            return Err(SteerError::Mismatch(
                "made for another model (fingerprint differs)".into(),
            ));
        }
        if self.hidden != model.spec.hidden as usize || self.layers.len() != model.w.layers.len() {
            return Err(SteerError::Mismatch(
                "hidden size or layer count differs".into(),
            ));
        }
        Ok(())
    }

    /// L2 norm of each layer's vector (0 where there is none).
    pub fn norms(&self) -> Vec<f32> {
        self.layers
            .iter()
            .map(|l| l.as_ref().map_or(0.0, |v| norm(v) as f32))
            .collect()
    }

    /// Cosine similarity of each layer's vector with the previous layer's (`None` where undefined).
    pub fn adjacent_cosines(&self) -> Vec<Option<f32>> {
        (0..self.layers.len())
            .map(|i| {
                let (a, b) = (
                    self.layers.get(i.wrapping_sub(1))?.as_ref()?,
                    self.layers[i].as_ref()?,
                );
                let d = norm(a) * norm(b);
                (d > 1e-20).then(|| (dot(a, b) / d) as f32)
            })
            .collect()
    }

    /// Serialise: header, per-layer vectors, SHA-256 of everything before it. All little-endian.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(MAGIC);
        o.extend_from_slice(&VERSION.to_le_bytes());
        o.extend_from_slice(&self.fingerprint);
        for n in [
            self.hidden,
            self.layers.len(),
            self.positive_prompts,
            self.negative_prompts,
        ] {
            o.extend_from_slice(&(n as u32).to_le_bytes());
        }
        o.push(self.method.code());
        let name = self.name.as_bytes();
        let mut cut = name.len().min(MAX_NAME);
        while !self.name.is_char_boundary(cut) {
            cut -= 1;
        } // never split a UTF-8 character
        let name = &name[..cut];
        o.extend_from_slice(&(name.len() as u16).to_le_bytes());
        o.extend_from_slice(name);
        for l in &self.layers {
            match l {
                None => o.push(0),
                Some(v) => {
                    o.push(1);
                    for x in v {
                        o.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
        }
        let mut sh = Sha256::new();
        sh.update(&o);
        o.extend_from_slice(&sh.finish());
        o
    }

    /// Parse and validate a file. Every failure is an `Err`; nothing here panics on hostile input.
    pub fn from_bytes(data: &[u8]) -> Result<ControlVector, SteerError> {
        let bad = |m: &str| SteerError::Format(m.into());
        if data.len() < 4 + 4 + 32 + 16 + 1 + 2 + 32 {
            return Err(bad("too short"));
        }
        let (body, sum) = data.split_at(data.len() - 32);
        let mut sh = Sha256::new();
        sh.update(body);
        if sh.finish().as_slice() != sum {
            return Err(bad("checksum mismatch (damaged or edited)"));
        }
        let mut r = Reader { d: body, at: 0 };
        if r.take(4)? != MAGIC {
            return Err(bad("not a control-vector file"));
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(SteerError::Format(format!(
                "version {version} not supported"
            )));
        }
        let fingerprint: [u8; 32] = r
            .take(32)?
            .try_into()
            .map_err(|_| bad("short fingerprint"))?;
        let (hidden, layers, pn, nn) = (
            r.u32()? as usize,
            r.u32()? as usize,
            r.u32()? as usize,
            r.u32()? as usize,
        );
        if hidden == 0 || hidden > MAX_HIDDEN {
            return Err(bad("hidden size out of range"));
        }
        if layers == 0 || layers > MAX_LAYERS {
            return Err(bad("layer count out of range"));
        }
        let method = Method::from_code(r.take(1)?[0]).ok_or_else(|| bad("unknown method"))?;
        let nlen = r.u16()? as usize;
        if nlen > MAX_NAME {
            return Err(bad("name too long"));
        }
        let name = std::str::from_utf8(r.take(nlen)?)
            .map_err(|_| bad("name is not UTF-8"))?
            .to_string();
        let mut out = Vec::with_capacity(layers);
        for _ in 0..layers {
            match r.take(1)?[0] {
                0 => out.push(None),
                1 => {
                    let raw = r.take(hidden.checked_mul(4).ok_or_else(|| bad("size overflow"))?)?;
                    let v: Vec<f32> = raw
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    if v.iter().any(|x| !x.is_finite()) {
                        return Err(bad("non-finite value"));
                    }
                    out.push(Some(v));
                }
                _ => return Err(bad("bad layer flag")),
            }
        }
        if r.at != body.len() {
            return Err(bad("trailing bytes"));
        }
        Ok(ControlVector {
            name,
            method,
            fingerprint,
            hidden,
            positive_prompts: pn,
            negative_prompts: nn,
            layers: out,
        })
    }

    /// Write to `path` (temp file, then rename, so a crash never leaves half a file).
    pub fn save(&self, path: &Path) -> Result<(), SteerError> {
        let tmp = path.with_extension("cvec.tmp");
        std::fs::write(&tmp, self.to_bytes()).map_err(|e| SteerError::Io(e.to_string()))?;
        std::fs::rename(&tmp, path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            SteerError::Io(e.to_string())
        })
    }

    pub fn load(path: &Path) -> Result<ControlVector, SteerError> {
        let cap = (MAX_LAYERS * (MAX_HIDDEN * 4 + 1) + 4096) as u64;
        let len = std::fs::metadata(path)
            .map_err(|e| SteerError::Io(e.to_string()))?
            .len();
        if len > cap {
            return Err(SteerError::Format(
                "file larger than any valid control vector".into(),
            ));
        }
        ControlVector::from_bytes(&std::fs::read(path).map_err(|e| SteerError::Io(e.to_string()))?)
    }

    /// JSON summary: norms and layer-to-layer cosines, plus the vectors themselves when asked.
    pub fn to_json(&self, include_vectors: bool) -> Value {
        let norms = self.norms();
        let cos = self.adjacent_cosines();
        let per_layer: Vec<Value> = (0..self.layers.len()).map(|i| {
            let mut o = json!({ "layer": i, "present": self.layers[i].is_some(), "norm": norms[i], "cosine_with_previous": cos[i] });
            if include_vectors && let Some(v) = &self.layers[i] { o["vector"] = json!(v); }
            o
        }).collect();
        json!({
            "name": self.name, "method": self.method.name(), "model_fingerprint": hex(&self.fingerprint),
            "hidden": self.hidden, "layers": self.layers.len(),
            "positive_prompts": self.positive_prompts, "negative_prompts": self.negative_prompts,
            "per_layer": per_layer,
        })
    }
}

struct Reader<'a> {
    d: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], SteerError> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.d.len())
            .ok_or_else(|| SteerError::Format("truncated".into()))?;
        let s = &self.d[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, SteerError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u16(&mut self) -> Result<u16, SteerError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
}

/// First principal component of `diffs` (uncentred power iteration), scaled by the mean projection
/// of the rows onto it and signed so that the mean projection is positive. All zeros for all-zero input.
pub fn pca_direction(diffs: &[Vec<f32>]) -> Vec<f32> {
    let h = diffs.first().map_or(0, Vec::len);
    let mut mean = vec![0.0f64; h];
    for d in diffs {
        mean.iter_mut()
            .zip(d)
            .for_each(|(m, &x)| *m += f64::from(x) / diffs.len() as f64);
    }
    let mut v: Vec<f64> = mean.clone();
    if v.iter().map(|x| x * x).sum::<f64>().sqrt() < 1e-12 {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        v = (0..h)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f64 / (1u64 << 24) as f64 - 0.5
            })
            .collect();
    }
    let unit = |v: &mut Vec<f64>| -> bool {
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n < 1e-30 {
            false
        } else {
            v.iter_mut().for_each(|x| *x /= n);
            true
        }
    };
    if !unit(&mut v) {
        return vec![0.0; h];
    }
    for _ in 0..200 {
        let mut w = vec![0.0f64; h];
        for d in diffs {
            let p: f64 = d.iter().zip(&v).map(|(&x, y)| f64::from(x) * y).sum();
            w.iter_mut()
                .zip(d)
                .for_each(|(a, &x)| *a += p * f64::from(x));
        }
        if !unit(&mut w) {
            return vec![0.0; h];
        }
        let delta: f64 = w
            .iter()
            .zip(&v)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        v = w;
        if delta < 1e-9 {
            break;
        }
    }
    let mean_proj: f64 = diffs
        .iter()
        .map(|d| {
            d.iter()
                .zip(&v)
                .map(|(&x, y)| f64::from(x) * y)
                .sum::<f64>()
        })
        .sum::<f64>()
        / diffs.len() as f64;
    let scale = if mean_proj < 0.0 {
        -mean_proj
    } else {
        mean_proj
    };
    let sign = if mean_proj < 0.0 { -1.0 } else { 1.0 };
    v.iter().map(|&x| (x * sign * scale) as f32).collect()
}

/// Learn a control vector: run every prompt, take the residual stream at its last token after each
/// layer, and compare the positive class with the negative one. Prompts are token ids.
pub fn train(
    model: &Model,
    positive: &[Vec<usize>],
    negative: &[Vec<usize>],
    method: Method,
    name: &str,
) -> Result<ControlVector, SteerError> {
    if method == Method::Custom {
        return Err(SteerError::Param(
            "Method::Custom is not a training method".into(),
        ));
    }
    if positive.is_empty() || negative.is_empty() {
        return Err(SteerError::Param(
            "both prompt sets need at least one prompt".into(),
        ));
    }
    if positive.len() > MAX_PROMPTS || negative.len() > MAX_PROMPTS {
        return Err(SteerError::Param(format!(
            "at most {MAX_PROMPTS} prompts per class"
        )));
    }
    if name.len() > MAX_NAME {
        return Err(SteerError::Param("name longer than 256 bytes".into()));
    }
    let (h, nl) = (model.spec.hidden as usize, model.w.layers.len());
    let pca = method == Method::Pca;
    if pca {
        if positive.len() != negative.len() {
            return Err(SteerError::Param(
                "pca pairs prompts in order: both sets must have the same size".into(),
            ));
        }
        if positive.len().saturating_mul(nl).saturating_mul(h) > MAX_PCA_FLOATS {
            return Err(SteerError::Param(
                "too many prompts for pca at this model size".into(),
            ));
        }
    }
    let mut sum_p = vec![vec![0.0f64; h]; nl];
    let mut sum_n = vec![vec![0.0f64; h]; nl];
    let mut diffs: Vec<Vec<Vec<f32>>> = if pca {
        vec![Vec::with_capacity(positive.len()); nl]
    } else {
        vec![]
    };
    let add = |acc: &mut Vec<Vec<f64>>, c: &[Vec<f32>]| {
        for (a, v) in acc.iter_mut().zip(c) {
            a.iter_mut().zip(v).for_each(|(s, &x)| *s += f64::from(x));
        }
    };
    for i in 0..positive.len().max(negative.len()) {
        let cp = positive
            .get(i)
            .map(|p| capture_last(model, p))
            .transpose()?;
        let cn = negative
            .get(i)
            .map(|p| capture_last(model, p))
            .transpose()?;
        if let Some(c) = &cp {
            add(&mut sum_p, c);
        }
        if let Some(c) = &cn {
            add(&mut sum_n, c);
        }
        if let (true, Some(p), Some(n)) = (pca, &cp, &cn) {
            for (l, (a, b)) in p.iter().zip(n).enumerate() {
                diffs[l].push(a.iter().zip(b).map(|(x, y)| x - y).collect());
            }
        }
    }
    let (np, nn) = (positive.len() as f64, negative.len() as f64);
    let layers = (0..nl)
        .map(|l| {
            let v: Vec<f32> = if pca {
                pca_direction(&diffs[l])
            } else {
                sum_p[l]
                    .iter()
                    .zip(&sum_n[l])
                    .map(|(p, n)| (p / np - n / nn) as f32)
                    .collect()
            };
            Some(v)
        })
        .collect();
    Ok(ControlVector {
        name: name.into(),
        method,
        fingerprint: model.fingerprint(),
        hidden: h,
        positive_prompts: positive.len(),
        negative_prompts: negative.len(),
        layers,
    })
}

// ------------------------------------------------------------------ applying a vector

/// Strength per layer (multiplies the control vector; negative values push the other way).
#[derive(Debug, Clone, PartialEq)]
pub struct Profile(Vec<f32>);
impl Profile {
    pub fn new(strengths: Vec<f32>) -> Profile {
        Profile(strengths)
    }
    pub fn zero(layers: usize) -> Profile {
        Profile(vec![0.0; layers])
    }
    /// The same strength on every layer in `range`, zero elsewhere.
    pub fn uniform(layers: usize, range: Range<usize>, strength: f32) -> Profile {
        Profile(
            (0..layers)
                .map(|i| if range.contains(&i) { strength } else { 0.0 })
                .collect(),
        )
    }
    /// A Gaussian bump: `peak * exp(-((layer - center) / width)^2 / 2)`. A non-positive width gives a single layer.
    pub fn bell(layers: usize, center: f32, width: f32, peak: f32) -> Profile {
        Profile(
            (0..layers)
                .map(|i| {
                    let d = i as f32 - center;
                    if width > 0.0 {
                        peak * (-0.5 * (d / width) * (d / width)).exp()
                    } else if d.abs() < 0.5 {
                        peak
                    } else {
                        0.0
                    }
                })
                .collect(),
        )
    }
    pub fn strengths(&self) -> &[f32] {
        &self.0
    }
    /// All strengths scaled by `k`.
    pub fn scaled(&self, k: f32) -> Profile {
        Profile(self.0.iter().map(|s| s * k).collect())
    }
}

/// A control vector plus a strength profile, checked against the model it will run on.
pub struct Steering<'a> {
    cv: &'a ControlVector,
    profile: Profile,
}
impl<'a> Steering<'a> {
    pub fn new(
        model: &Model,
        cv: &'a ControlVector,
        profile: Profile,
    ) -> Result<Steering<'a>, SteerError> {
        cv.check_model(model)?;
        if profile.0.len() != cv.layers.len() {
            return Err(SteerError::Mismatch(format!(
                "profile has {} layers, the model has {}",
                profile.0.len(),
                cv.layers.len()
            )));
        }
        if profile.0.iter().any(|s| !s.is_finite()) {
            return Err(SteerError::Param("non-finite strength".into()));
        }
        Ok(Steering { cv, profile })
    }
    /// Same strength on every layer that has a vector.
    pub fn uniform(
        model: &Model,
        cv: &'a ControlVector,
        strength: f32,
    ) -> Result<Steering<'a>, SteerError> {
        Steering::new(
            model,
            cv,
            Profile::uniform(cv.layers.len(), 0..cv.layers.len(), strength),
        )
    }
    /// True when nothing would be added (every strength zero or no vector where one is nonzero).
    pub fn is_neutral(&self) -> bool {
        self.profile
            .0
            .iter()
            .zip(&self.cv.layers)
            .all(|(s, v)| *s == 0.0 || v.is_none())
    }
    pub fn profile(&self) -> &Profile {
        &self.profile
    }
    /// Add the layer's steering to every row of `x`.
    fn apply(&self, li: usize, x: &mut [f32]) {
        let s = self.profile.0[li];
        let Some(v) = self.cv.layers[li].as_deref().filter(|_| s != 0.0) else {
            return;
        };
        for row in x.chunks_mut(v.len()) {
            row.iter_mut().zip(v).for_each(|(a, b)| *a += s * b);
        }
    }
}

/// `forward_batch` with steering (`None` = none): feed `tokens`, return the logits after the last one.
/// With `None`, or a neutral steering, the result is bit-identical to [`Model::forward_batch`].
pub fn forward_steered(
    model: &Model,
    st: &mut State,
    tokens: &[usize],
    steering: Option<&Steering>,
) -> Result<Vec<f32>, SteerError> {
    let last = run_layerwise(
        model,
        st,
        tokens,
        steering.filter(|s| !s.is_neutral()),
        &mut no_hook,
    )?;
    let logits = model.logits_from_hidden(&last);
    if logits.is_empty() {
        return Err(SteerError::Model(ModelError::Shape(
            "could not compute logits".into(),
        )));
    }
    Ok(logits)
}

/// [`Model::generate`] with steering applied to the prompt and to every generated token.
pub fn generate_steered(
    model: &Model,
    st: &mut State,
    prompt: &[usize],
    sampler: &mut Sampler,
    opts: &GenerateOptions,
    steering: Option<&Steering>,
    on_token: &mut dyn FnMut(usize) -> bool,
) -> Result<GenerateStats, SteerError> {
    let mut stats = GenerateStats {
        prompt_tokens: prompt.len(),
        generated: 0,
        prefill_secs: 0.0,
        decode_secs: 0.0,
        stop: StopReason::MaxTokens,
    };
    let t0 = Instant::now();
    let mut logits = forward_steered(model, st, prompt, steering)?;
    stats.prefill_secs = t0.elapsed().as_secs_f64();
    sampler.observe_all(prompt);
    let t1 = Instant::now();
    loop {
        if stats.generated >= opts.max_tokens {
            stats.stop = StopReason::MaxTokens;
            break;
        }
        if opts.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            stats.stop = StopReason::Cancelled;
            break;
        }
        let tok = sampler.sample(&logits);
        sampler.observe(tok);
        if opts.stop_tokens.contains(&tok) {
            stats.stop = StopReason::StopToken;
            break;
        }
        stats.generated += 1;
        if !on_token(tok) {
            stats.stop = StopReason::Callback;
            break;
        }
        if stats.generated >= opts.max_tokens {
            stats.stop = StopReason::MaxTokens;
            break;
        }
        if st.pos >= model.spec.context as usize {
            stats.stop = StopReason::ContextFull;
            break;
        }
        logits = forward_steered(model, st, &[tok], steering)?;
    }
    stats.decode_secs = t1.elapsed().as_secs_f64();
    Ok(stats)
}

// ------------------------------------------------------------------ logit lens

/// Which positions of the fed tokens the lens analyses (decoding the residual stream through the
/// output head costs `vocab * hidden` per layer per position).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LensPositions {
    Last,
    LastN(usize),
    All,
}

pub struct LensOptions<'a> {
    pub top_k: usize,
    pub positions: LensPositions,
    /// Summarise attention concentration per layer (turns score tracking on for the call).
    pub attention: bool,
    /// Steer while analysing, to see where a control vector acts.
    pub steering: Option<&'a Steering<'a>>,
}
impl Default for LensOptions<'_> {
    fn default() -> Self {
        LensOptions {
            top_k: 5,
            positions: LensPositions::Last,
            attention: true,
            steering: None,
        }
    }
}

/// What one layer would predict at one position.
#[derive(Debug, Clone, PartialEq)]
pub struct LensLayer {
    pub layer: usize,
    /// Most likely tokens with their probabilities, best first.
    pub top: Vec<(usize, f32)>,
    /// Probability this layer gives the model's final top-1 token.
    pub final_prob: f32,
    /// Rank (1 = top) of the final top-1 token in this layer's prediction.
    pub final_rank: usize,
    /// Entropy of the layer's prediction, in nats.
    pub entropy: f32,
    /// KL(final prediction || this layer's prediction), in nats.
    pub kl_final_to_layer: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LensPosition {
    /// Index in the fed tokens.
    pub index: usize,
    pub token: usize,
    /// The model's own top-1 prediction after the last layer.
    pub final_top1: usize,
    pub layers: Vec<LensLayer>,
    /// First layer whose top-1 is the final answer (it may lose it again later).
    pub first_hit_layer: usize,
    /// Earliest layer from which the top-1 equals the final answer at every later layer.
    pub commit_layer: usize,
}

/// How attention mass was spread over the cached positions in one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct AttnSummary {
    pub slots: usize,
    /// Entropy of the accumulated attention-mass distribution divided by ln(slots): 0 = all on one slot, 1 = flat.
    pub entropy_norm: f32,
    /// Share of the mass on the single busiest slot.
    pub top_slot_share: f32,
    /// Share on the first slot (an "attention sink").
    pub first_slot_share: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LensReport {
    pub layers: usize,
    pub positions: Vec<LensPosition>,
    /// One entry per layer; `None` for ring layers or when not requested.
    pub attention: Vec<Option<AttnSummary>>,
    /// The model's logits after the last fed token.
    pub final_logits: Vec<f32>,
}

/// `(first_hit, commit)` layer from each layer's top-1 and the final answer (`tops` ends with it).
pub fn commit_layers(tops: &[usize], final_top1: usize) -> (usize, usize) {
    let first = tops
        .iter()
        .position(|&t| t == final_top1)
        .unwrap_or(tops.len().saturating_sub(1));
    let mut commit = tops.len().saturating_sub(1);
    while commit > 0 && tops[commit - 1] == final_top1 {
        commit -= 1;
    }
    (first, commit)
}

struct Dist {
    lse: f64,
}
fn lse(l: &[f32]) -> Dist {
    let m = l
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return Dist {
            lse: f64::NEG_INFINITY,
        };
    }
    Dist {
        lse: f64::from(m)
            + l.iter()
                .filter(|x| x.is_finite())
                .map(|&x| (f64::from(x) - f64::from(m)).exp())
                .sum::<f64>()
                .ln(),
    }
}

fn top_k(l: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (i, &x) in l.iter().enumerate() {
        if !x.is_finite() {
            continue;
        }
        if best.len() == k && x <= best[k - 1].0 {
            continue;
        }
        let at = best.partition_point(|&(v, _)| v >= x);
        best.insert(at, (x, i));
        best.truncate(k);
    }
    best.into_iter().map(|(v, i)| (i, v)).collect()
}

fn analyse_layer(
    layer: usize,
    logits: &[f32],
    fin: &[f32],
    fin_lse: &Dist,
    fin_top1: usize,
    k: usize,
) -> LensLayer {
    let d = lse(logits);
    let p = |x: f32| {
        if x.is_finite() && d.lse.is_finite() {
            (f64::from(x) - d.lse).exp()
        } else {
            0.0
        }
    };
    let top = top_k(logits, k)
        .into_iter()
        .map(|(i, x)| (i, p(x) as f32))
        .collect();
    let (mut ent, mut kl) = (0.0f64, 0.0f64);
    for (&lx, &fx) in logits.iter().zip(fin) {
        let (pl, pf) = (
            p(lx),
            if fx.is_finite() && fin_lse.lse.is_finite() {
                (f64::from(fx) - fin_lse.lse).exp()
            } else {
                0.0
            },
        );
        if pl > 0.0 {
            ent -= pl * pl.ln();
        }
        if pf > 0.0 {
            kl += pf * (pf.ln() - if pl > 0.0 { pl.ln() } else { -745.0 });
        }
    }
    let fl = logits[fin_top1];
    LensLayer {
        layer,
        top,
        final_prob: p(fl) as f32,
        final_rank: 1 + logits.iter().filter(|&&x| x > fl).count(),
        entropy: ent as f32,
        kl_final_to_layer: kl.clamp(0.0, 1e6) as f32,
    }
}

fn attn_summary(scores: &[f32]) -> Option<AttnSummary> {
    let total: f64 = scores.iter().map(|&x| f64::from(x.max(0.0))).sum();
    if scores.is_empty() || total <= 0.0 || !total.is_finite() {
        return None;
    }
    let ent: f64 = scores
        .iter()
        .map(|&x| f64::from(x.max(0.0)) / total)
        .filter(|&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    let norm = if scores.len() > 1 {
        ent / (scores.len() as f64).ln()
    } else {
        0.0
    };
    let top = scores.iter().copied().fold(0.0f32, f32::max);
    Some(AttnSummary {
        slots: scores.len(),
        entropy_norm: norm.clamp(0.0, 1.0) as f32,
        top_slot_share: (f64::from(top) / total) as f32,
        first_slot_share: (f64::from(scores[0].max(0.0)) / total) as f32,
    })
}

/// Feed `tokens` through `model` (advancing `st`, like `forward_batch`) and decode the residual
/// stream after every layer at the selected positions.
pub fn lens(
    model: &Model,
    st: &mut State,
    tokens: &[usize],
    opts: &LensOptions,
) -> Result<LensReport, SteerError> {
    let (h, nl) = (model.spec.hidden as usize, model.w.layers.len());
    if opts.top_k == 0 || opts.top_k > MAX_TOP_K {
        return Err(SteerError::Param(format!("top_k must be 1..={MAX_TOP_K}")));
    }
    let n = tokens.len();
    let selected: Vec<usize> = match opts.positions {
        _ if n == 0 => return Err(ModelError::NoTokens.into()),
        LensPositions::Last => vec![n - 1],
        LensPositions::LastN(k) => (n.saturating_sub(k.max(1))..n).collect(),
        LensPositions::All => (0..n).collect(),
    };
    if selected.len() > MAX_LENS_POSITIONS {
        return Err(SteerError::Param(format!(
            "at most {MAX_LENS_POSITIONS} positions per call"
        )));
    }
    let start = st.pos;
    let was_tracking = st.tracks_scores();
    if opts.attention {
        if was_tracking {
            st.clear_scores();
        } else {
            st.track_scores(true);
        }
    }
    let mut hidden: Vec<Vec<f32>> = vec![Vec::with_capacity(nl * h); selected.len()];
    let steer = opts.steering.filter(|s| !s.is_neutral());
    let mut hook = |_li: usize, x: &[f32], b: usize, pos0: usize, _last: bool| {
        let base = pos0 - start;
        for (slot, &idx) in selected.iter().enumerate() {
            if idx >= base && idx < base + b {
                let r = idx - base;
                hidden[slot].extend_from_slice(&x[r * h..(r + 1) * h]);
            }
        }
    };
    let run = run_layerwise(model, st, tokens, steer, &mut hook);
    let attention: Vec<Option<AttnSummary>> = (0..nl)
        .map(|li| {
            if opts.attention {
                st.layer_scores(li).and_then(attn_summary)
            } else {
                None
            }
        })
        .collect();
    if opts.attention && !was_tracking {
        st.track_scores(false);
    }
    let last = run?;
    let final_logits = model.logits_from_hidden(&last);
    if final_logits.is_empty() {
        return Err(SteerError::Model(ModelError::Shape(
            "could not compute logits".into(),
        )));
    }
    let mut positions = Vec::with_capacity(selected.len());
    for (slot, &idx) in selected.iter().enumerate() {
        let rows = &hidden[slot];
        if rows.len() != nl * h {
            return Err(SteerError::Param(
                "internal: a layer was not observed".into(),
            ));
        }
        let fin = model.logits_from_hidden(&rows[(nl - 1) * h..]);
        let fin_top1 = top_k(&fin, 1)
            .first()
            .map(|t| t.0)
            .ok_or_else(|| SteerError::Param("the final logits are not finite".into()))?;
        let fl = lse(&fin);
        let layers: Vec<LensLayer> = (0..nl)
            .map(|li| {
                let lg = model.logits_from_hidden(&rows[li * h..(li + 1) * h]);
                analyse_layer(li, &lg, &fin, &fl, fin_top1, opts.top_k)
            })
            .collect();
        let tops: Vec<usize> = layers
            .iter()
            .map(|l| l.top.first().map_or(usize::MAX, |t| t.0))
            .collect();
        let (first_hit_layer, commit_layer) = commit_layers(&tops, fin_top1);
        positions.push(LensPosition {
            index: idx,
            token: tokens[idx],
            final_top1: fin_top1,
            layers,
            first_hit_layer,
            commit_layer,
        });
    }
    Ok(LensReport {
        layers: nl,
        positions,
        attention,
        final_logits,
    })
}

impl LensReport {
    /// How many positions commit at each layer.
    pub fn commit_histogram(&self) -> Vec<usize> {
        let mut h = vec![0; self.layers];
        for p in &self.positions {
            if let Some(c) = h.get_mut(p.commit_layer) {
                *c += 1;
            }
        }
        h
    }
    /// Mean of `(commit_layer + 1) / layers` over the positions: 1.0 = answers only appear at the last layer.
    pub fn mean_commit_depth(&self) -> f32 {
        if self.positions.is_empty() || self.layers == 0 {
            return 0.0;
        }
        self.positions
            .iter()
            .map(|p| (p.commit_layer + 1) as f32 / self.layers as f32)
            .sum::<f32>()
            / self.positions.len() as f32
    }
    /// JSON for a UI. `decode` turns a token id into text (without it only ids are given).
    pub fn to_json_with(&self, decode: Option<&dyn Fn(usize) -> String>) -> Value {
        let tok = |t: usize| -> Value {
            match decode {
                Some(f) => json!({ "id": t, "text": f(t) }),
                None => json!({ "id": t }),
            }
        };
        let positions: Vec<Value> = self.positions.iter().map(|p| json!({
            "index": p.index, "token": tok(p.token), "final_top1": tok(p.final_top1),
            "first_hit_layer": p.first_hit_layer, "commit_layer": p.commit_layer,
            "layers": p.layers.iter().map(|l| json!({
                "layer": l.layer, "final_prob": l.final_prob, "final_rank": l.final_rank, "entropy": l.entropy, "kl_final_to_layer": l.kl_final_to_layer,
                "top": l.top.iter().map(|&(t, pr)| { let mut o = tok(t); o["prob"] = json!(pr); o }).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })).collect();
        let attn: Vec<Value> = self.attention.iter().enumerate().map(|(i, a)| match a {
            Some(a) => json!({ "layer": i, "slots": a.slots, "entropy_norm": a.entropy_norm, "top_slot_share": a.top_slot_share, "first_slot_share": a.first_slot_share }),
            None => json!({ "layer": i, "slots": null }),
        }).collect();
        json!({
            "layers": self.layers, "positions": positions, "attention": attn,
            "commit_histogram": self.commit_histogram(), "mean_commit_depth": self.mean_commit_depth(),
        })
    }
    pub fn to_json(&self) -> Value {
        self.to_json_with(None)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn to_bytes_cuts_a_long_multibyte_name_on_a_char_boundary() {
        for pad in ["", "a"] {
            let name = format!("{pad}{}", "é".repeat(MAX_NAME));
            let cv = ControlVector {
                name,
                method: Method::Custom,
                fingerprint: [0; 32],
                hidden: 2,
                positive_prompts: 0,
                negative_prompts: 0,
                layers: vec![Some(vec![1.0, 2.0])],
            };
            let back =
                ControlVector::from_bytes(&cv.to_bytes()).expect("a cut name still round-trips");
            assert!(back.name.len() <= MAX_NAME && cv.name.starts_with(&back.name));
            assert!(
                back.name.len() + 1 >= MAX_NAME,
                "cut no more than one character short"
            );
        }
    }

    use super::*;
    use crate::model::Model;
    use crate::model::tests::{base_spec, model_of, tiny};

    fn model() -> Model {
        tiny(4, 2, true)
    }
    fn prompt(n: usize, seed: usize) -> Vec<usize> {
        (0..n).map(|i| (i * 7 + seed * 3 + 1) % 20).collect()
    }
    fn argmax(l: &[f32]) -> usize {
        l.iter()
            .enumerate()
            .fold(
                (0, f32::NEG_INFINITY),
                |a, (i, &x)| if x > a.1 { (i, x) } else { a },
            )
            .0
    }
    fn softmax_of(l: &[f32], t: usize) -> f64 {
        let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f64 = l.iter().map(|&x| f64::from(x - m).exp()).sum();
        f64::from(l[t] - m).exp() / z
    }

    #[test]
    fn layerwise_path_is_bit_identical_to_forward_batch() {
        for m in [model(), model_of(base_spec(4, 2, false), true)] {
            let toks = prompt(11, 1);
            let mut a = m.new_state();
            let want = m.forward_batch(&mut a, &toks).unwrap();
            let mut b = m.new_state();
            let got = forward_steered(&m, &mut b, &toks, None).unwrap();
            assert_eq!(want, got);
            assert_eq!(a.pos, b.pos);
            assert_eq!(a.tokens(), b.tokens());
            // and the cache they leave behind behaves identically for the next token
            assert_eq!(
                m.forward(&mut a, 5).unwrap(),
                forward_steered(&m, &mut b, &[5], None).unwrap()
            );
        }
    }

    #[test]
    fn long_prompts_cross_chunks_identically() {
        let mut sp = base_spec(4, 2, true);
        sp.context = 400;
        let m = model_of(sp, false);
        let toks = prompt(PREFILL_CHUNK * 2 + 9, 2);
        let mut a = m.new_state();
        let want = m.forward_batch(&mut a, &toks).unwrap();
        let mut b = m.new_state();
        assert_eq!(want, forward_steered(&m, &mut b, &toks, None).unwrap());
    }

    #[test]
    fn strength_zero_is_bit_identical() {
        let m = model();
        let toks = prompt(9, 3);
        let cv = train(
            &m,
            &[prompt(5, 1), prompt(6, 2)],
            &[prompt(5, 3), prompt(4, 4)],
            Method::MeanDiff,
            "t",
        )
        .unwrap();
        let mut a = m.new_state();
        let want = m.forward_batch(&mut a, &toks).unwrap();
        for steer in [
            Steering::uniform(&m, &cv, 0.0).unwrap(),
            Steering::new(&m, &cv, Profile::zero(3)).unwrap(),
        ] {
            assert!(steer.is_neutral());
            let mut b = m.new_state();
            assert_eq!(
                want,
                forward_steered(&m, &mut b, &toks, Some(&steer)).unwrap()
            );
        }
        // a real strength changes the result
        let s1 = Steering::uniform(&m, &cv, 1.0).unwrap();
        let mut c = m.new_state();
        assert_ne!(want, forward_steered(&m, &mut c, &toks, Some(&s1)).unwrap());
    }

    /// A vector made of the output embedding of one token, added after the last layer, must raise
    /// that token's probability monotonically and end up making it the top-1 prediction.
    #[test]
    fn steering_moves_the_engineered_target_monotonically() {
        let m = model();
        let (h, target) = (m.spec.hidden as usize, 7usize);
        let row = m.w.embed.to_f32()[target * h..(target + 1) * h].to_vec();
        let toks = prompt(6, 5);
        for layer in [2usize, 1] {
            let mut layers = vec![None; 3];
            layers[layer] = Some(row.clone());
            let cv = ControlVector::from_layers(&m, "push", layers).unwrap();
            let probs: Vec<f64> = [0.0f32, 1.0, 2.0, 4.0, 8.0, 16.0, 40.0]
                .iter()
                .map(|&a| {
                    let steer =
                        Steering::new(&m, &cv, Profile::uniform(3, layer..layer + 1, a)).unwrap();
                    let mut st = m.new_state();
                    softmax_of(
                        &forward_steered(&m, &mut st, &toks, Some(&steer)).unwrap(),
                        target,
                    )
                })
                .collect();
            for w in probs.windows(2) {
                assert!(w[1] > w[0], "layer {layer}: {probs:?}");
            }
            let steer =
                Steering::new(&m, &cv, Profile::uniform(3, layer..layer + 1, 40.0)).unwrap();
            let mut st = m.new_state();
            assert_eq!(
                argmax(&forward_steered(&m, &mut st, &toks, Some(&steer)).unwrap()),
                target
            );
            // the opposite sign pushes the other way
            let neg = Steering::new(&m, &cv, Profile::uniform(3, layer..layer + 1, -16.0)).unwrap();
            let (mut a, mut b) = (m.new_state(), m.new_state());
            let base = softmax_of(&forward_steered(&m, &mut a, &toks, None).unwrap(), target);
            assert!(
                softmax_of(
                    &forward_steered(&m, &mut b, &toks, Some(&neg)).unwrap(),
                    target
                ) < base
            );
        }
    }

    #[test]
    fn steered_generation_uses_the_vector() {
        let m = model();
        let (h, target) = (m.spec.hidden as usize, 11usize);
        let mut layers = vec![None; 3];
        layers[2] = Some(m.w.embed.to_f32()[target * h..(target + 1) * h].to_vec());
        let cv = ControlVector::from_layers(&m, "push", layers).unwrap();
        let steer = Steering::new(&m, &cv, Profile::uniform(3, 2..3, 60.0)).unwrap();
        let run = |s: Option<&Steering>| {
            let mut st = m.new_state();
            let mut sm = Sampler::new(0.0, 1, 1);
            let mut out = vec![];
            generate_steered(
                &m,
                &mut st,
                &prompt(4, 1),
                &mut sm,
                &GenerateOptions {
                    max_tokens: 6,
                    ..Default::default()
                },
                s,
                &mut |t| {
                    out.push(t);
                    true
                },
            )
            .unwrap();
            out
        };
        assert!(run(Some(&steer)).iter().all(|&t| t == target));
        // without steering it matches the model's own generate
        let mut st = m.new_state();
        let mut sm = Sampler::new(0.0, 1, 1);
        let mut want = vec![];
        m.generate(
            &mut st,
            &prompt(4, 1),
            &mut sm,
            &GenerateOptions {
                max_tokens: 6,
                ..Default::default()
            },
            &mut |t| {
                want.push(t);
                true
            },
        )
        .unwrap();
        assert_eq!(run(None), want);
    }

    #[test]
    fn training_matches_an_independent_mean_difference() {
        let m = model();
        let (pos, neg) = (
            vec![prompt(5, 1), prompt(7, 2), prompt(3, 3)],
            vec![prompt(4, 4), prompt(6, 5)],
        );
        let cv = train(&m, &pos, &neg, Method::MeanDiff, "x").unwrap();
        assert_eq!(
            (
                cv.positive_prompts,
                cv.negative_prompts,
                cv.layers.len(),
                cv.hidden
            ),
            (3, 2, 3, 16)
        );
        let cap = |ps: &[Vec<usize>]| -> Vec<Vec<Vec<f32>>> {
            ps.iter().map(|p| capture_last(&m, p).unwrap()).collect()
        };
        let (cp, cn) = (cap(&pos), cap(&neg));
        for l in 0..3 {
            let v = cv.layers[l].as_ref().unwrap();
            for (j, &x) in v.iter().enumerate() {
                let mp: f32 = cp.iter().map(|c| c[l][j]).sum::<f32>() / 3.0;
                let mn: f32 = cn.iter().map(|c| c[l][j]).sum::<f32>() / 2.0;
                assert!((x - (mp - mn)).abs() < 1e-5);
            }
        }
        assert!(cv.norms().iter().all(|&n| n > 0.0));
    }

    #[test]
    fn pca_trains_on_pairs_and_follows_the_mean_difference() {
        let m = model();
        let base: Vec<Vec<usize>> = (0..6).map(|i| prompt(4 + i % 3, i)).collect();
        let pos: Vec<Vec<usize>> = base.iter().map(|b| [b.clone(), vec![3]].concat()).collect();
        let neg: Vec<Vec<usize>> = base.iter().map(|b| [b.clone(), vec![9]].concat()).collect();
        let (md, pc) = (
            train(&m, &pos, &neg, Method::MeanDiff, "m").unwrap(),
            train(&m, &pos, &neg, Method::Pca, "p").unwrap(),
        );
        for l in 0..3 {
            let (a, b) = (
                md.layers[l].as_ref().unwrap(),
                pc.layers[l].as_ref().unwrap(),
            );
            assert!(dot(a, b) / (norm(a) * norm(b)) > 0.9, "layer {l}");
        }
        assert!(train(&m, &pos, &neg[..3], Method::Pca, "p").is_err());
        assert!(train(&m, &[], &neg, Method::MeanDiff, "p").is_err());
        assert!(train(&m, &pos, &neg, Method::Custom, "p").is_err());
    }

    #[test]
    fn pca_direction_finds_the_dominant_axis_and_fixes_the_sign() {
        let u: Vec<f32> = (0..16)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.25 })
            .collect();
        let mut s = 7u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.05
        };
        let diffs: Vec<Vec<f32>> = (0..40)
            .map(|i| {
                u.iter()
                    .map(|&x| x * (1.0 + (i % 5) as f32 * 0.2) + r())
                    .collect()
            })
            .collect();
        let v = pca_direction(&diffs);
        assert!(dot(&v, &u) / (norm(&v) * norm(&u)) > 0.99);
        let flipped: Vec<Vec<f32>> = diffs
            .iter()
            .map(|d| d.iter().map(|x| -x).collect())
            .collect();
        let w = pca_direction(&flipped);
        assert!(dot(&w, &v) < 0.0, "sign follows the mean");
        assert!(
            pca_direction(&[vec![0.0; 4], vec![0.0; 4]])
                .iter()
                .all(|&x| x == 0.0)
        );
        assert!(pca_direction(&[]).is_empty());
    }

    #[test]
    fn file_round_trips_and_checks_the_model() {
        let m = model();
        let mut cv = train(
            &m,
            &[prompt(5, 1)],
            &[prompt(5, 2)],
            Method::MeanDiff,
            "polite · rude",
        )
        .unwrap();
        cv.restrict(1..3);
        assert!(cv.layers[0].is_none());
        let bytes = cv.to_bytes();
        assert_eq!(ControlVector::from_bytes(&bytes).unwrap(), cv);
        let path = std::env::temp_dir().join(format!("steer-test-{}.cvec", std::process::id()));
        cv.save(&path).unwrap();
        let back = ControlVector::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(back, cv);
        back.check_model(&m).unwrap();
        // another architecture does not accept it
        let other = model_of(base_spec(2, 1, true), false);
        assert!(back.check_model(&other).is_err());
        assert!(Steering::uniform(&other, &back, 1.0).is_err());
        assert!(Steering::new(&m, &back, Profile::zero(5)).is_err());
        assert!(Steering::new(&m, &back, Profile::new(vec![0.0, f32::NAN, 0.0])).is_err());
        assert!(ControlVector::load(Path::new("/nonexistent/x.cvec")).is_err());
    }

    #[test]
    fn hostile_files_are_errors_not_panics() {
        let m = model();
        let cv = train(&m, &[prompt(4, 1)], &[prompt(4, 2)], Method::MeanDiff, "n").unwrap();
        let good = cv.to_bytes();
        // every truncation
        for n in 0..good.len() {
            assert!(ControlVector::from_bytes(&good[..n]).is_err(), "prefix {n}");
        }
        // every single-byte change is caught by the checksum
        for i in 0..good.len() {
            let mut b = good.clone();
            b[i] ^= 0x5a;
            assert!(ControlVector::from_bytes(&b).is_err(), "byte {i}");
        }
        // forged files with a *valid* checksum: bad magic, version, sizes, flags, floats, trailing bytes
        let seal = |mut body: Vec<u8>| {
            let mut sh = Sha256::new();
            sh.update(&body);
            body.extend_from_slice(&sh.finish());
            body
        };
        let body = &good[..good.len() - 32];
        let edits: Vec<(usize, Vec<u8>)> = vec![
            (0, b"XXXX".to_vec()),
            (4, 9u32.to_le_bytes().to_vec()),
            (40, 0u32.to_le_bytes().to_vec()),
            (40, u32::MAX.to_le_bytes().to_vec()),
            (44, 0u32.to_le_bytes().to_vec()),
            (44, u32::MAX.to_le_bytes().to_vec()),
            (56, vec![77]),
            (57, vec![0xff, 0xff]),
        ];
        for (at, bytes) in edits {
            let mut b = body.to_vec();
            b[at..at + bytes.len()].copy_from_slice(&bytes);
            assert!(ControlVector::from_bytes(&seal(b)).is_err(), "edit at {at}");
        }
        let mut nan = body.to_vec();
        let first_float = nan.len() - 3 * (1 + 16 * 4) + 1;
        nan[first_float..first_float + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(ControlVector::from_bytes(&seal(nan)).is_err());
        let mut tail = body.to_vec();
        tail.push(0);
        assert!(ControlVector::from_bytes(&seal(tail)).is_err());
        let mut flag = body.to_vec();
        let at = flag.len() - 3 * (1 + 16 * 4);
        flag[at] = 9;
        assert!(ControlVector::from_bytes(&seal(flag)).is_err());
        // random garbage, sealed and unsealed
        let mut s = 0x1234_5678u64;
        for len in [0usize, 1, 31, 32, 100, 300, 2000] {
            let junk: Vec<u8> = (0..len)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s >> 33) as u8
                })
                .collect();
            assert!(ControlVector::from_bytes(&junk).is_err());
            let _ = ControlVector::from_bytes(&seal(junk));
        }
        // a file that claims enormous sizes but is tiny
        let mut huge = body[..57].to_vec();
        huge.extend_from_slice(&0u16.to_le_bytes());
        huge[40..44].copy_from_slice(&(MAX_HIDDEN as u32).to_le_bytes());
        huge[44..48].copy_from_slice(&(MAX_LAYERS as u32).to_le_bytes());
        assert!(ControlVector::from_bytes(&seal(huge)).is_err());
    }

    #[test]
    fn combine_adds_and_refuses_foreign_vectors() {
        let m = model();
        let a = train(&m, &[prompt(4, 1)], &[prompt(4, 2)], Method::MeanDiff, "a").unwrap();
        let b = train(&m, &[prompt(5, 3)], &[prompt(3, 4)], Method::MeanDiff, "b").unwrap();
        let c = ControlVector::combine(&[(&a, 1.0), (&b, -0.5)], "a-b/2").unwrap();
        assert_eq!(c.method, Method::Custom);
        for l in 0..3 {
            for j in 0..16 {
                assert!(
                    (c.layers[l].as_ref().unwrap()[j]
                        - (a.layers[l].as_ref().unwrap()[j]
                            - 0.5 * b.layers[l].as_ref().unwrap()[j]))
                        .abs()
                        < 1e-5
                );
            }
        }
        let other = model_of(base_spec(2, 1, true), false);
        let foreign = train(
            &other,
            &[prompt(4, 1)],
            &[prompt(4, 2)],
            Method::MeanDiff,
            "f",
        )
        .unwrap();
        assert!(ControlVector::combine(&[(&a, 1.0), (&foreign, 1.0)], "x").is_err());
        assert!(ControlVector::combine(&[], "x").is_err());
        assert!(ControlVector::combine(&[(&a, f32::INFINITY)], "x").is_err());
    }

    #[test]
    fn profiles_have_the_shapes_they_claim() {
        assert_eq!(
            Profile::uniform(5, 1..3, 2.0).strengths(),
            &[0.0, 2.0, 2.0, 0.0, 0.0]
        );
        let b = Profile::bell(9, 4.0, 1.5, 3.0);
        assert!((b.strengths()[4] - 3.0).abs() < 1e-6);
        assert!(
            b.strengths()[0] < b.strengths()[2]
                && b.strengths()[2] < b.strengths()[4]
                && (b.strengths()[3] - b.strengths()[5]).abs() < 1e-6
        );
        assert_eq!(
            Profile::bell(4, 2.0, 0.0, 1.0).strengths(),
            &[0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            Profile::uniform(3, 0..3, 1.0).scaled(2.0).strengths(),
            &[2.0, 2.0, 2.0]
        );
    }

    #[test]
    fn lens_last_layer_is_the_model_and_commit_is_consistent() {
        let m = model();
        let toks = prompt(10, 2);
        let mut a = m.new_state();
        let want = m.forward_batch(&mut a, &toks).unwrap();
        let mut b = m.new_state();
        let rep = lens(
            &m,
            &mut b,
            &toks,
            &LensOptions {
                top_k: 4,
                positions: LensPositions::All,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rep.final_logits, want);
        assert_eq!((rep.layers, rep.positions.len()), (3, 10));
        assert_eq!(a.pos, b.pos);
        assert!(!b.tracks_scores(), "tracking restored");
        for p in &rep.positions {
            assert_eq!(p.layers.len(), 3);
            let last = p.layers.last().unwrap();
            assert_eq!(last.top[0].0, p.final_top1);
            assert_eq!((last.final_rank, last.kl_final_to_layer < 1e-3), (1, true));
            assert!(p.first_hit_layer <= p.commit_layer && p.commit_layer < 3);
            for l in &p.layers {
                assert_eq!(l.top.len(), 4);
                assert!(l.top.windows(2).all(|w| w[0].1 >= w[1].1));
                assert!(
                    l.entropy >= 0.0
                        && l.entropy <= (20f32).ln() + 1e-4
                        && l.kl_final_to_layer >= 0.0
                        && l.final_rank >= 1
                );
                assert!((l.top.iter().map(|t| t.1).sum::<f32>() - 1.0) < 1e-4);
            }
        }
        assert_eq!(rep.commit_histogram().iter().sum::<usize>(), 10);
        assert!(rep.mean_commit_depth() > 0.0 && rep.mean_commit_depth() <= 1.0);
        // the final position equals the last-position-only report
        let mut c = m.new_state();
        let one = lens(
            &m,
            &mut c,
            &toks,
            &LensOptions {
                top_k: 4,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(one.positions.len(), 1);
        assert_eq!(one.positions[0], rep.positions[9]);
        let mut d = m.new_state();
        assert_eq!(
            lens(
                &m,
                &mut d,
                &toks,
                &LensOptions {
                    positions: LensPositions::LastN(3),
                    ..Default::default()
                }
            )
            .unwrap()
            .positions
            .len(),
            3
        );
    }

    #[test]
    fn lens_across_chunks_and_with_a_filled_state() {
        let mut sp = base_spec(4, 2, true);
        sp.context = 400;
        let m = model_of(sp, false);
        let toks = prompt(PREFILL_CHUNK + 20, 4);
        let mut a = m.new_state();
        let rep = lens(
            &m,
            &mut a,
            &toks,
            &LensOptions {
                positions: LensPositions::LastN(30),
                attention: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            rep.positions.iter().map(|p| p.index).collect::<Vec<_>>(),
            (toks.len() - 30..toks.len()).collect::<Vec<_>>()
        );
        let mut b = m.new_state();
        assert_eq!(m.forward_batch(&mut b, &toks).unwrap(), rep.final_logits);
        assert!(rep.attention.iter().all(Option::is_none));
        // a state that already holds a prefix: positions are relative to the fed tokens
        let mut c = m.new_state();
        m.forward_batch(&mut c, &toks[..40]).unwrap();
        let rest = lens(&m, &mut c, &toks[40..], &LensOptions::default()).unwrap();
        assert_eq!(rest.positions[0].index, toks.len() - 41);
        assert_eq!(rest.final_logits.len(), 20);
    }

    #[test]
    fn attention_summary_is_bounded_and_tracking_state_is_respected() {
        let m = model();
        let toks = prompt(12, 6);
        let mut st = m.new_state();
        let rep = lens(&m, &mut st, &toks, &LensOptions::default()).unwrap();
        assert_eq!(rep.attention.len(), 3);
        for a in rep.attention.iter().flatten() {
            assert_eq!(a.slots, 12);
            assert!(
                (0.0..=1.0).contains(&a.entropy_norm)
                    && a.top_slot_share >= 1.0 / 12.0 - 1e-6
                    && a.first_slot_share <= 1.0
            );
        }
        assert!(rep.attention.iter().any(Option::is_some));
        // a caller that already tracks keeps tracking
        let mut t = m.new_state();
        t.track_scores(true);
        lens(&m, &mut t, &toks, &LensOptions::default()).unwrap();
        assert!(t.tracks_scores());
        // helper edge cases
        assert!(attn_summary(&[]).is_none() && attn_summary(&[0.0, 0.0]).is_none());
        let flat = attn_summary(&[1.0; 8]).unwrap();
        assert!((flat.entropy_norm - 1.0).abs() < 1e-5);
        assert!(attn_summary(&[5.0, 0.0, 0.0]).unwrap().entropy_norm < 1e-6);
    }

    #[test]
    fn steering_shows_up_in_the_lens() {
        let m = model();
        let (h, target) = (m.spec.hidden as usize, 7usize);
        let mut layers = vec![None; 3];
        layers[1] = Some(m.w.embed.to_f32()[target * h..(target + 1) * h].to_vec());
        let cv = ControlVector::from_layers(&m, "push", layers).unwrap();
        let steer = Steering::new(&m, &cv, Profile::uniform(3, 1..2, 40.0)).unwrap();
        let toks = prompt(6, 5);
        let (mut a, mut b) = (m.new_state(), m.new_state());
        let plain = lens(&m, &mut a, &toks, &LensOptions::default()).unwrap();
        let pushed = lens(
            &m,
            &mut b,
            &toks,
            &LensOptions {
                steering: Some(&steer),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(pushed.positions[0].final_top1, target);
        assert_eq!(
            pushed.positions[0].layers[1].top[0].0, target,
            "the target already leads at the steered layer"
        );
        assert_ne!(
            plain.positions[0].layers[1].top,
            pushed.positions[0].layers[1].top
        );
        let (p0, q0) = (
            &plain.positions[0].layers[0],
            &pushed.positions[0].layers[0],
        );
        assert_eq!(
            (&p0.top, p0.entropy),
            (&q0.top, q0.entropy),
            "layers before the injection are untouched"
        );
    }

    #[test]
    fn commit_layers_logic() {
        assert_eq!(commit_layers(&[5, 5, 5], 5), (0, 0));
        assert_eq!(commit_layers(&[1, 2, 5], 5), (2, 2));
        assert_eq!(commit_layers(&[5, 2, 5, 5], 5), (0, 2));
        assert_eq!(commit_layers(&[1, 5, 5, 3, 5], 5), (1, 4));
        assert_eq!(commit_layers(&[7], 7), (0, 0));
        assert_eq!(commit_layers(&[], 7), (0, 0));
    }

    #[test]
    fn json_exports_parse_and_carry_the_numbers() {
        let m = model();
        let cv = train(&m, &[prompt(4, 1)], &[prompt(4, 2)], Method::MeanDiff, "j").unwrap();
        let j = cv.to_json(false);
        assert_eq!(j["name"], "j");
        assert_eq!(j["per_layer"].as_array().unwrap().len(), 3);
        assert!(j["per_layer"][0].get("vector").is_none());
        assert_eq!(
            cv.to_json(true)["per_layer"][1]["vector"]
                .as_array()
                .unwrap()
                .len(),
            16
        );
        let s = serde_json::to_string(&cv.to_json(true)).unwrap();
        assert!(serde_json::from_str::<Value>(&s).is_ok());
        let mut st = m.new_state();
        let rep = lens(
            &m,
            &mut st,
            &prompt(5, 1),
            &LensOptions {
                positions: LensPositions::All,
                ..Default::default()
            },
        )
        .unwrap();
        let dec = |t: usize| format!("<{t}>");
        let j = rep.to_json_with(Some(&dec));
        assert_eq!(j["positions"].as_array().unwrap().len(), 5);
        assert_eq!(
            j["positions"][0]["layers"][0]["top"][0]["text"]
                .as_str()
                .unwrap()
                .chars()
                .next(),
            Some('<')
        );
        assert!(
            j["positions"][0]["commit_layer"].is_u64()
                && j["attention"].as_array().unwrap().len() == 3
        );
        assert!(rep.to_json()["positions"][0]["token"].get("text").is_none());
        assert!(serde_json::from_str::<Value>(&serde_json::to_string(&j).unwrap()).is_ok());
    }

    #[test]
    fn bad_calls_are_errors() {
        let m = model();
        let mut st = m.new_state();
        assert!(forward_steered(&m, &mut st, &[], None).is_err());
        assert!(forward_steered(&m, &mut st, &[999], None).is_err());
        assert!(
            forward_steered(&m, &mut st, &vec![1; 40], None).is_err(),
            "context is 32"
        );
        assert!(
            lens(
                &m,
                &mut m.new_state(),
                &[1, 2],
                &LensOptions {
                    top_k: 0,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            lens(
                &m,
                &mut m.new_state(),
                &[1, 2],
                &LensOptions {
                    top_k: MAX_TOP_K + 1,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(lens(&m, &mut m.new_state(), &[], &LensOptions::default()).is_err());
        assert!(ControlVector::from_layers(&m, "x", vec![None; 2]).is_err());
        assert!(ControlVector::from_layers(&m, "x", vec![Some(vec![0.0; 3]), None, None]).is_err());
        assert!(
            ControlVector::from_layers(&m, "x", vec![Some(vec![f32::NAN; 16]), None, None])
                .is_err()
        );
        assert!(ControlVector::from_layers(&m, &"n".repeat(300), vec![None; 3]).is_err());
    }

    #[test]
    fn works_on_moe_and_sliding_window_models() {
        let m = crate::model::tests::accel_and_prefix::moe_model();
        let toks = prompt(8, 2);
        let mut a = m.new_state();
        let want = m.forward_batch(&mut a, &toks).unwrap();
        let mut b = m.new_state();
        assert_eq!(want, forward_steered(&m, &mut b, &toks, None).unwrap());
        let mut c = m.new_state();
        let rep = lens(&m, &mut c, &toks, &LensOptions::default()).unwrap();
        assert_eq!(rep.final_logits, want);
        assert_eq!(rep.layers, 3);
    }
}
