//! Decoder forward pass. One code path covers the Llama / Mistral / Qwen / Gemma / Phi-3 /
//! StarCoder2 family of variants: RMS- or layer-norm, grouped-query attention with (partial,
//! scaled) RoPE and a KV cache, optional biases, QK-norm, soft-capping and sliding windows,
//! gated or plain MLP, sequential or parallel residuals, optional tied output head.
//!
//! Weights are [`Weight`]s: plain f32, or quantised bytes (owned or memory-mapped) that are
//! multiplied in place without ever being expanded to f32. The KV cache can be f32, f16 or
//! int8, and sliding-window layers keep only their window in a ring buffer. One implementation
//! serves both single-token decoding and batched prefill ([`Model::forward_batch`]). A layer may
//! carry a mixture-of-experts feed-forward ([`crate::moe`]) instead of the dense MLP.
use crate::arch::{Act, NormKind, RopeScaling, Spec};
use crate::backend::{AttnArgs, Backend, Buf, Kv, WeightHandle, WeightRef};
use crate::moe::{ExpertProvider, MoeBlock, MoeError};
use crate::ops::softmax;
use crate::pool;
use crate::qmatmul::qmatvec_threads;
use crate::quant::{DType, QuantError, dequantize};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

/// Tokens processed per pass during prefill; bounds scratch memory for long prompts.
pub const PREFILL_CHUNK: usize = 128;

// ------------------------------------------------------------------ weights

/// Raw bytes of a tensor: owned, or a window into a memory-mapped file.
#[derive(Clone)]
pub enum Bytes {
    Owned(Arc<Vec<u8>>),
    Mapped {
        map: Arc<memmap2::Mmap>,
        start: usize,
        len: usize,
    },
}
impl Bytes {
    pub fn owned(v: Vec<u8>) -> Bytes {
        Bytes::Owned(Arc::new(v))
    }
    /// `None` when the window does not lie inside the map.
    pub fn mapped(map: Arc<memmap2::Mmap>, start: usize, len: usize) -> Option<Bytes> {
        (start.checked_add(len)? <= map.len()).then_some(Bytes::Mapped { map, start, len })
    }
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Bytes::Owned(v) => v,
            Bytes::Mapped { map, start, len } => &map[*start..*start + *len],
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Bytes::Owned(v) => v.len(),
            Bytes::Mapped { len, .. } => *len,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn is_mapped(&self) -> bool {
        matches!(self, Bytes::Mapped { .. })
    }
    /// A sub-window; mapped bytes stay mapped, owned bytes are copied.
    pub fn slice(&self, from: usize, len: usize) -> Option<Bytes> {
        if from.checked_add(len)? > self.len() {
            return None;
        }
        Some(match self {
            Bytes::Owned(v) => Bytes::owned(v[from..from + len].to_vec()),
            Bytes::Mapped { map, start, .. } => Bytes::Mapped {
                map: map.clone(),
                start: start + from,
                len,
            },
        })
    }
}

/// A weight matrix kept in its stored (quantised or half-precision) form, row-major `rows × cols`.
#[derive(Clone)]
pub struct QWeight {
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
    pub bytes: Bytes,
}
fn moe_err(e: MoeError) -> ModelError {
    ModelError::Experts(e.to_string())
}
fn qerr(e: QuantError) -> ModelError {
    ModelError::Shape(e.to_string())
}
impl QWeight {
    pub fn new(
        dtype: DType,
        rows: usize,
        cols: usize,
        bytes: Bytes,
    ) -> Result<QWeight, ModelError> {
        let rb = dtype
            .bytes_for(cols as u64)
            .filter(|_| dtype.decodable())
            .ok_or_else(|| {
                ModelError::Shape(format!(
                    "{dtype:?} rows of {cols} values cannot be kept quantised"
                ))
            })? as usize;
        if rows.checked_mul(rb) != Some(bytes.len()) {
            return Err(ModelError::Shape(format!(
                "{dtype:?} {rows}x{cols} needs {rows}*{rb} bytes, got {}",
                bytes.len()
            )));
        }
        Ok(QWeight {
            dtype,
            rows,
            cols,
            bytes,
        })
    }
    pub fn row_bytes(&self) -> usize {
        self.dtype.bytes_for(self.cols as u64).unwrap_or(0) as usize
    }
    fn row_into(&self, r: usize, buf: &mut Vec<f32>) -> Result<(), ModelError> {
        let rb = self.row_bytes();
        let b = self
            .bytes
            .as_slice()
            .get(r * rb..(r + 1) * rb)
            .ok_or_else(|| ModelError::Shape(format!("row {r} out of range")))?;
        dequantize(self.dtype, b, self.cols, buf).map_err(qerr)
    }
}

#[derive(Clone)]
pub enum Weight {
    Dense(Vec<f32>),
    Quant(QWeight),
}
impl Default for Weight {
    fn default() -> Self {
        Weight::Dense(Vec::new())
    }
}
impl From<Vec<f32>> for Weight {
    fn from(v: Vec<f32>) -> Self {
        Weight::Dense(v)
    }
}

/// Threads a call may use: the pool's size, or `nt` when the model was told to use fewer (`0` = all).
fn par_threads(nt: usize) -> usize {
    let p = pool::global().threads();
    if nt == 0 { p } else { nt.min(p).max(1) }
}
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
/// Weight bytes below which a quantised product is not worth splitting across threads.
const QTHREAD_MIN_BYTES: usize = 1 << 20;
/// Dot product with eight independent accumulators, so the compiler can vectorise it.
pub fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0f32; 8];
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}
fn dense_rows(w: &[f32], cols: usize, x: &[f32], out: &mut [f32]) {
    for (o, row) in out.iter_mut().zip(w.chunks_exact(cols)) {
        *o = dot8(row, x);
    }
}
fn dense_matvec(w: &[f32], cols: usize, x: &[f32], out: &mut [f32], nt: usize) {
    if cols == 0 {
        out.fill(0.0);
        return;
    }
    let nt = par_threads(nt);
    if w.len() < (1 << 18) || nt == 1 || out.len() < 2 * nt {
        return dense_rows(w, cols, x, out);
    }
    let min_rows = out.len().div_ceil(nt);
    pool::global().parallel_chunks_mut(out, min_rows, |start, o| {
        dense_rows(&w[start * cols..(start + o.len()) * cols], cols, x, o)
    });
}

impl Weight {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Weight::Dense(v) => v.len(),
            Weight::Quant(q) => q.rows * q.cols,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn as_dense(&self) -> Option<&[f32]> {
        if let Weight::Dense(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn is_mapped(&self) -> bool {
        matches!(self, Weight::Quant(q) if q.bytes.is_mapped())
    }
    /// Bytes this weight occupies as stored (resident for owned data, address space for mapped).
    pub fn stored_bytes(&self) -> usize {
        match self {
            Weight::Dense(v) => v.len() * 4,
            Weight::Quant(q) => q.bytes.len(),
        }
    }
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Weight::Dense(v) => v.clone(),
            Weight::Quant(q) => {
                let mut o = vec![];
                let _ = dequantize(q.dtype, q.bytes.as_slice(), q.rows * q.cols, &mut o);
                o
            }
        }
    }
    fn rows_of(&self, cols: usize) -> usize {
        match self {
            Weight::Dense(v) => v.len().checked_div(cols).unwrap_or(0),
            Weight::Quant(q) => q.rows,
        }
    }
    /// `out = W · x`, `W` being `out.len() × cols`.
    pub fn matvec(&self, cols: usize, x: &[f32], out: &mut [f32]) -> Result<(), ModelError> {
        self.matvec_nt(cols, x, out, 0)
    }
    /// Like [`Weight::matvec`] with an explicit thread limit (`0` = every pool thread).
    pub fn matvec_nt(
        &self,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
        nt: usize,
    ) -> Result<(), ModelError> {
        let bad = || {
            ModelError::Shape(format!(
                "matvec {}x{} with a vector of {}",
                out.len(),
                cols,
                x.len()
            ))
        };
        if x.len() != cols {
            return Err(bad());
        }
        match self {
            Weight::Dense(w) => {
                if w.len() != out.len() * cols {
                    return Err(bad());
                }
                dense_matvec(w, cols, x, out, nt);
                Ok(())
            }
            Weight::Quant(q) => {
                if q.cols != cols || q.rows != out.len() {
                    return Err(bad());
                }
                let n = if q.bytes.len() < QTHREAD_MIN_BYTES {
                    1
                } else {
                    par_threads(nt).min(q.rows / 4).max(1)
                };
                qmatvec_threads(q.dtype, q.bytes.as_slice(), q.rows, q.cols, x, out, n)
                    .map_err(qerr)
            }
        }
    }
    /// Rows `r0..r0+n` times all `b` token vectors, written as `yt[i*b + t]`.
    fn tile(
        &self,
        r0: usize,
        n: usize,
        cols: usize,
        x: &[f32],
        b: usize,
        yt: &mut [f32],
    ) -> Result<(), ModelError> {
        let mut buf = Vec::new();
        for i in 0..n {
            let r = r0 + i;
            let row: &[f32] = match self {
                Weight::Dense(w) => &w[r * cols..(r + 1) * cols],
                Weight::Quant(q) => {
                    q.row_into(r, &mut buf)?;
                    &buf
                }
            };
            for t in 0..b {
                yt[i * b + t] = dot8(row, &x[t * cols..(t + 1) * cols]);
            }
        }
        Ok(())
    }
    /// `out[t] = W · x[t]` for `b` token vectors at once. Each weight row is read once for the whole
    /// batch (quantised rows are decoded once), which is what makes prefill faster than `b` matvecs.
    pub fn matmat(
        &self,
        cols: usize,
        x: &[f32],
        b: usize,
        out: &mut [f32],
        yt: &mut Vec<f32>,
    ) -> Result<(), ModelError> {
        self.matmat_nt(cols, x, b, out, yt, 0)
    }
    /// Like [`Weight::matmat`] with an explicit thread limit (`0` = every pool thread).
    pub fn matmat_nt(
        &self,
        cols: usize,
        x: &[f32],
        b: usize,
        out: &mut [f32],
        yt: &mut Vec<f32>,
        nt: usize,
    ) -> Result<(), ModelError> {
        self.matmat_opt(cols, x, b, out, yt, nt, false)
    }
    /// [`Weight::matmat_nt`] with `exact`: a quantised weight then always uses the single-token path
    /// per row of the batch, so the result is bit-identical to `b` separate matvecs.
    #[allow(clippy::too_many_arguments)]
    pub fn matmat_opt(
        &self,
        cols: usize,
        x: &[f32],
        b: usize,
        out: &mut [f32],
        yt: &mut Vec<f32>,
        nt: usize,
        exact: bool,
    ) -> Result<(), ModelError> {
        let rows = self.rows_of(cols);
        if b == 0 || x.len() != b * cols || out.len() != b * rows {
            return Err(ModelError::Shape("matmat sizes disagree".into()));
        }
        if b == 1 || (matches!(self, Weight::Quant(_)) && (exact || b < 4)) {
            for t in 0..b {
                self.matvec_nt(
                    cols,
                    &x[t * cols..(t + 1) * cols],
                    &mut out[t * rows..(t + 1) * rows],
                    nt,
                )?;
            }
            return Ok(());
        }
        yt.clear();
        yt.resize(rows * b, 0.0);
        let nt = par_threads(nt);
        if rows * cols * b < (1 << 18) || nt == 1 || rows < 2 * nt {
            self.tile(0, rows, cols, x, b, yt)?;
        } else {
            // each chunk computes its rows privately, then copies them under a short lock
            let (dst, err): (Mutex<&mut [f32]>, Mutex<Option<ModelError>>) =
                (Mutex::new(yt.as_mut_slice()), Mutex::new(None));
            pool::global().parallel_for(rows, rows.div_ceil(nt), |r| {
                let mut loc = vec![0.0; r.len() * b];
                match self.tile(r.start, r.len(), cols, x, b, &mut loc) {
                    Ok(()) => lock(&dst)[r.start * b..r.end * b].copy_from_slice(&loc),
                    Err(e) => {
                        lock(&err).get_or_insert(e);
                    }
                }
            });
            if let Some(e) = lock(&err).take() {
                return Err(e);
            }
        }
        for r in 0..rows {
            for t in 0..b {
                out[t * rows + r] = yt[r * b + t];
            }
        }
        Ok(())
    }
    /// Copy row `row` (length `cols`) into `out`, decoding it if quantised.
    pub fn row_into(
        &self,
        row: usize,
        cols: usize,
        out: &mut [f32],
        tmp: &mut Vec<f32>,
    ) -> Result<(), ModelError> {
        match self {
            Weight::Dense(w) => out.copy_from_slice(
                w.get(row * cols..(row + 1) * cols)
                    .ok_or(ModelError::Token(row))?,
            ),
            Weight::Quant(q) => {
                if row >= q.rows || q.cols != cols {
                    return Err(ModelError::Token(row));
                }
                q.row_into(row, tmp)?;
                out.copy_from_slice(tmp);
            }
        }
        Ok(())
    }
}

#[derive(Default, Clone)]
pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub attn_norm_b: Option<Vec<f32>>,
    pub wq: Weight,
    pub wk: Weight,
    pub wv: Weight,
    pub wo: Weight,
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
    pub bo: Option<Vec<f32>>,
    /// Per-head RMSNorm weights (`head_dim` values) when `spec.qk_norm`.
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    /// Norm applied to the attention output before it is added back (`spec.post_norms`).
    pub post_attn_norm: Option<Vec<f32>>,
    /// Norm in front of the MLP (Llama's `post_attention_layernorm`, Gemma's `pre_feedforward_layernorm`).
    pub ffn_norm: Vec<f32>,
    pub ffn_norm_b: Option<Vec<f32>>,
    /// Empty when the MLP is not gated.
    pub w_gate: Weight,
    pub b_gate: Option<Vec<f32>>,
    pub w_up: Weight,
    pub b_up: Option<Vec<f32>>,
    pub w_down: Weight,
    pub b_down: Option<Vec<f32>>,
    pub post_ffn_norm: Option<Vec<f32>>,
    /// Mixture-of-experts feed-forward in place of the dense MLP above (which is then empty).
    pub moe: Option<Arc<MoeBlock>>,
}
/// Where routed experts come from, plus whether to read ahead.
#[derive(Clone)]
pub struct Experts {
    pub provider: Arc<dyn ExpertProvider>,
    /// Ask the provider for the experts the previous token used in the next MoE layer, while this one computes.
    pub prefetch: bool,
}
#[derive(Default, Clone)]
pub struct Weights {
    pub embed: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Vec<f32>,
    pub final_norm_b: Option<Vec<f32>>,
    /// `None` = tied to `embed`.
    pub lm_head: Option<Weight>,
    pub lm_head_b: Option<Vec<f32>>,
    /// Routed experts of every MoE layer; required when any layer has `moe`.
    pub experts: Option<Experts>,
}
impl Weights {
    /// Bytes taken by the big matrices as stored (for MoE models: routers, shared experts and the
    /// routed experts currently held in memory).
    pub fn stored_bytes(&self) -> usize {
        self.embed.stored_bytes()
            + self.lm_head.as_ref().map_or(0, Weight::stored_bytes)
            + self
                .layers
                .iter()
                .map(|l| {
                    [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]
                        .iter()
                        .map(|w| w.stored_bytes())
                        .sum::<usize>()
                        + l.moe.as_ref().map_or(0, |m| m.stored_bytes())
                })
                .sum::<usize>()
            + self
                .experts
                .as_ref()
                .map_or(0, |e| e.provider.resident_bytes() as usize)
    }
}
pub struct Model {
    pub spec: Spec,
    pub w: Weights,
    rope_global: (Vec<f64>, f64),
    rope_local: (Vec<f64>, f64),
    /// Thread limit for the heavy products (`0` = every thread of the pool).
    threads: usize,
    accel: Option<Accel>,
}

/// How [`Model::embed`] folds per-token hidden states into one vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    Mean,
    Last,
    First,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ModelError {
    Shape(String),
    Token(usize),
    ContextFull(usize),
    NoTokens,
    Experts(String),
    Backend(String),
}
impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Shape(s) => write!(f, "weights do not match the architecture: {s}"),
            ModelError::Token(t) => write!(f, "token {t} is outside the vocabulary"),
            ModelError::ContextFull(n) => write!(f, "the context window of {n} tokens is full"),
            ModelError::NoTokens => write!(f, "no tokens were given"),
            ModelError::Experts(s) => write!(f, "mixture-of-experts layer: {s}"),
            ModelError::Backend(s) => write!(f, "compute backend: {s}"),
        }
    }
}
impl std::error::Error for ModelError {}

// ------------------------------------------------------------------ primitives

// ------------------------------------------------------------------ primitives

/// RMS- or layer-norm according to the spec; `w` is applied as `1 + w` for Gemma-style RMSNorm.
pub fn norm(spec: &Spec, x: &[f32], w: &[f32], b: Option<&[f32]>, out: &mut [f32]) {
    let n = x.len().max(1) as f32;
    let eps = spec.norm_eps as f32;
    match spec.norm {
        NormKind::Rms => {
            let k = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / n + eps).sqrt();
            let off = if spec.norm_offset { 1.0 } else { 0.0 };
            for ((o, v), g) in out.iter_mut().zip(x).zip(w) {
                *o = v * k * (g + off);
            }
        }
        NormKind::Layer => {
            let mean = x.iter().sum::<f32>() / n;
            let k = 1.0 / (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n + eps).sqrt();
            for (i, ((o, v), g)) in out.iter_mut().zip(x).zip(w).enumerate() {
                *o = (v - mean) * k * g + b.map_or(0.0, |b| b[i]);
            }
        }
    }
}

fn erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * crate::simd::rmath::exp(-x * x);
    y.copysign(x)
}
pub fn activate(a: Act, x: f32) -> f32 {
    match a {
        Act::Silu => x / (1.0 + crate::simd::rmath::exp(-x)),
        Act::GeluTanh => {
            0.5 * x * (1.0 + crate::simd::rmath::tanh(0.797_884_6 * (x + 0.044_715 * x * x * x)))
        }
        Act::Gelu => 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)),
    }
}

/// Inverse frequencies (one per rotated pair) and the cos/sin magnitude factor for a RoPE setup.
pub fn rope_inv_freq(theta: f64, rotary_dim: usize, scaling: RopeScaling) -> (Vec<f64>, f64) {
    let half = rotary_dim / 2;
    let base: Vec<f64> = (0..half)
        .map(|i| crate::simd::rmath::powf64(theta, -(2.0 * i as f64) / rotary_dim as f64))
        .collect();
    match scaling {
        RopeScaling::None => (base, 1.0),
        // LongRoPE needs the per-pair factor tables of the Spec: callers use `Spec::rope_freqs`
        RopeScaling::LongRope { mscale, .. } => (base, mscale),
        RopeScaling::Nope => (vec![0.0; half], 1.0),
        RopeScaling::Linear { factor } => (base.into_iter().map(|f| f / factor).collect(), 1.0),
        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max,
        } => {
            let (low_w, high_w) = (
                original_max / low_freq_factor,
                original_max / high_freq_factor,
            );
            let v = base
                .into_iter()
                .map(|f| {
                    let wl = 2.0 * std::f64::consts::PI / f;
                    if wl < high_w {
                        f
                    } else if wl > low_w {
                        f / factor
                    } else {
                        let sm = (original_max / wl - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        (1.0 - sm) * f / factor + sm * f
                    }
                })
                .collect();
            (v, 1.0)
        }
        RopeScaling::Yarn {
            factor,
            original_max,
            beta_fast,
            beta_slow,
            mscale,
        } => {
            let d = rotary_dim as f64;
            let corr = |rot: f64| {
                d * crate::simd::rmath::ln64(original_max / (rot * 2.0 * std::f64::consts::PI))
                    / (2.0 * crate::simd::rmath::ln64(theta))
            };
            let lo = corr(beta_fast).floor().max(0.0);
            let hi = corr(beta_slow).ceil().min(d - 1.0);
            let hi = if hi == lo { hi + 0.001 } else { hi };
            let v = base
                .into_iter()
                .enumerate()
                .map(|(i, f)| {
                    let extra = 1.0 - ((i as f64 - lo) / (hi - lo)).clamp(0.0, 1.0);
                    f / factor * (1.0 - extra) + f * extra
                })
                .collect();
            (v, mscale)
        }
    }
}

/// Rotate the first `2 * inv.len()` dimensions of one head (rotate-half pairing).
pub fn rope_apply(head: &mut [f32], pos: usize, (inv, mscale): &(Vec<f64>, f64)) {
    let half = inv.len();
    for (i, f) in inv.iter().enumerate() {
        let (s, c) = crate::simd::rmath::sin_cos64(pos as f64 * f);
        let (s, c) = (s * mscale, c * mscale);
        let (a, b) = (head[i] as f64, head[i + half] as f64);
        head[i] = (a * c - b * s) as f32;
        head[i + half] = (a * s + b * c) as f32;
    }
}

fn add_bias(v: &mut [f32], b: &Option<Vec<f32>>) {
    if let Some(b) = b {
        v.iter_mut().zip(b).for_each(|(a, b)| *a += b);
    }
}
fn softcap(v: f32, cap: f64) -> f32 {
    if cap > 0.0 {
        (cap as f32) * crate::simd::rmath::tanh(v / cap as f32)
    } else {
        v
    }
}

fn rope_table(pos: usize, (inv, ms): &(Vec<f64>, f64), cos: &mut [f64], sin: &mut [f64]) {
    for (i, f) in inv.iter().enumerate() {
        let (s, c) = crate::simd::rmath::sin_cos64(pos as f64 * f);
        cos[i] = c * ms;
        sin[i] = s * ms;
    }
}
/// Rotate one head with a precomputed cos/sin row; identical numbers to [`rope_apply`].
fn rope_cs(head: &mut [f32], cos: &[f64], sin: &[f64]) {
    let half = cos.len();
    for i in 0..half {
        let (a, b) = (head[i] as f64, head[i + half] as f64);
        head[i] = (a * cos[i] - b * sin[i]) as f32;
        head[i + half] = (a * sin[i] + b * cos[i]) as f32;
    }
}

// ------------------------------------------------------------------ KV cache

/// Precision of the key/value cache. `F16` halves its memory, `Q8` (int8 with one scale per head
/// and position) quarters it, at a small cost in accuracy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvKind {
    #[default]
    F32,
    F16,
    Q8,
}

fn f16_lut() -> &'static [f32] {
    static L: OnceLock<Vec<f32>> = OnceLock::new();
    L.get_or_init(|| (0..=65535u16).map(crate::quant::f16_to_f32).collect())
}
/// Round-to-nearest-even f32 → IEEE half.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let ne = e - 127 + 15;
    if ne >= 31 {
        return sign | 0x7c00;
    }
    if ne <= 0 {
        if ne < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let shift = (14 - ne) as u32;
        let mut h = (m >> shift) as u16;
        let (rem, mid) = (m & ((1 << shift) - 1), 1u32 << (shift - 1));
        if rem > mid || (rem == mid && h & 1 == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = (((ne as u32) << 10) | (m >> 13)) as u16;
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h
}

#[derive(Clone)]
enum Plane {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Q8 { q: Vec<i8>, s: Vec<f32> },
}
impl Default for Plane {
    fn default() -> Self {
        Plane::F32(vec![])
    }
}
impl Plane {
    fn new(kind: KvKind) -> Plane {
        match kind {
            KvKind::F32 => Plane::F32(vec![]),
            KvKind::F16 => Plane::F16(vec![]),
            KvKind::Q8 => Plane::Q8 {
                q: vec![],
                s: vec![],
            },
        }
    }
    fn write(&mut self, slot: usize, kd: usize, hd: usize, src: &[f32]) {
        let (lo, hi) = (slot * kd, (slot + 1) * kd);
        match self {
            Plane::F32(v) => {
                if v.len() < hi {
                    v.resize(hi, 0.0);
                }
                v[lo..hi].copy_from_slice(src);
            }
            Plane::F16(v) => {
                if v.len() < hi {
                    v.resize(hi, 0);
                }
                for (d, s) in v[lo..hi].iter_mut().zip(src) {
                    *d = f32_to_f16(*s);
                }
            }
            Plane::Q8 { q, s } => {
                let kvh = kd / hd;
                if q.len() < hi {
                    q.resize(hi, 0);
                }
                if s.len() < (slot + 1) * kvh {
                    s.resize((slot + 1) * kvh, 0.0);
                }
                for h in 0..kvh {
                    let c = &src[h * hd..(h + 1) * hd];
                    let amax = c.iter().fold(0f32, |a, v| a.max(v.abs()));
                    let sc = if amax > 0.0 { amax / 127.0 } else { 0.0 };
                    s[slot * kvh + h] = sc;
                    let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 };
                    for (d, v) in q[lo + h * hd..lo + (h + 1) * hd].iter_mut().zip(c) {
                        *d = (v * inv).round().clamp(-127.0, 127.0) as i8;
                    }
                }
            }
        }
    }
    /// One position, decoded to f32 (`kd` values).
    fn read(&self, slot: usize, kd: usize, hd: usize, out: &mut [f32]) {
        let (lo, hi) = (slot * kd, (slot + 1) * kd);
        match self {
            Plane::F32(v) => out.copy_from_slice(&v[lo..hi]),
            Plane::F16(v) => {
                let l = f16_lut();
                for (d, s) in out.iter_mut().zip(&v[lo..hi]) {
                    *d = l[*s as usize];
                }
            }
            Plane::Q8 { q, s } => {
                let kvh = kd / hd;
                for h in 0..kvh {
                    let sc = s[slot * kvh + h];
                    for (d, x) in out[h * hd..(h + 1) * hd]
                        .iter_mut()
                        .zip(&q[lo + h * hd..lo + (h + 1) * hd])
                    {
                        *d = *x as f32 * sc;
                    }
                }
            }
        }
    }
    fn len_slots(&self, kd: usize, kvh: usize) -> usize {
        match self {
            Plane::F32(v) => v.len() / kd.max(1),
            Plane::F16(v) => v.len() / kd.max(1),
            Plane::Q8 { s, .. } => s.len() / kvh.max(1),
        }
    }
    fn dot(&self, slot: usize, kd: usize, hd: usize, head: usize, qh: &[f32]) -> f32 {
        let o = slot * kd + head * hd;
        match self {
            Plane::F32(v) => dot8(qh, &v[o..o + hd]),
            Plane::F16(v) => {
                let l = f16_lut();
                qh.iter()
                    .zip(&v[o..o + hd])
                    .map(|(a, b)| a * l[*b as usize])
                    .sum()
            }
            Plane::Q8 { q, s } => {
                qh.iter()
                    .zip(&q[o..o + hd])
                    .map(|(a, b)| a * *b as f32)
                    .sum::<f32>()
                    * s[slot * (kd / hd) + head]
            }
        }
    }
    fn axpy(&self, slot: usize, kd: usize, hd: usize, head: usize, p: f32, out: &mut [f32]) {
        let o = slot * kd + head * hd;
        match self {
            Plane::F32(v) => {
                for (d, x) in out.iter_mut().zip(&v[o..o + hd]) {
                    *d += p * x;
                }
            }
            Plane::F16(v) => {
                let l = f16_lut();
                for (d, x) in out.iter_mut().zip(&v[o..o + hd]) {
                    *d += p * l[*x as usize];
                }
            }
            Plane::Q8 { q, s } => {
                let k = p * s[slot * (kd / hd) + head];
                for (d, x) in out.iter_mut().zip(&q[o..o + hd]) {
                    *d += k * *x as f32;
                }
            }
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Plane::F32(v) => v.len() * 4,
            Plane::F16(v) => v.len() * 2,
            Plane::Q8 { q, s } => q.len() + s.len() * 4,
        }
    }
    /// Forget every position from `slots` on.
    fn truncate(&mut self, slots: usize, kd: usize, kvh: usize) {
        match self {
            Plane::F32(v) => v.truncate(slots * kd),
            Plane::F16(v) => v.truncate(slots * kd),
            Plane::Q8 { q, s } => {
                q.truncate(slots * kd);
                s.truncate(slots * kvh);
            }
        }
    }
}

#[derive(Clone, Default)]
struct LayerKv {
    k: Plane,
    v: Plane,
    /// ring capacity; 0 = keep everything
    cap: usize,
    /// values per position (`kv_heads × head_dim`) and kv heads
    kd: usize,
    kvh: usize,
}
impl LayerKv {
    fn slot(&self, pos: usize) -> usize {
        if self.cap > 0 { pos % self.cap } else { pos }
    }
}

/// Reusable work buffers, so decoding allocates nothing per token.
#[derive(Default)]
struct Scratch {
    x: Vec<f32>,
    xb: Vec<f32>,
    xm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    proj: Vec<f32>,
    tmp: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
    mlp: Vec<f32>,
    scores: Vec<f32>,
    rowtmp: Vec<f32>,
    headtmp: Vec<f32>,
    yt: Vec<f32>,
}
fn fit(v: &mut Vec<f32>, n: usize) {
    v.clear();
    v.resize(n, 0.0);
}

/// Per-conversation state: the KV cache (grows, or wraps for sliding-window layers), scratch
/// buffers, and the tokens fed so far (so a server can reuse a common prefix, see
/// [`Model::prefill_cached`]).
#[derive(Default)]
pub struct State {
    kv: Vec<LayerKv>,
    pub pos: usize,
    sc: Scratch,
    moe_last: Vec<Vec<usize>>,
    tokens: Vec<usize>,
    last: Option<Vec<f32>>,
    reused: usize,
    dev: Option<DevState>,
    /// Eviction bookkeeping (see the "KV-cache surgery" block below).
    kvx: KvExtra,
    /// Route quantised batched products through the per-token (activation-quantised) path, so a
    /// batch gives bit-identical logits to feeding the tokens one at a time. See [`State::set_exact_batch`].
    exact_batch: bool,
    /// Slot ranges (sorted, disjoint) whose cache rows were moved in from another state and not yet
    /// recomputed: they are close to, but not exactly, what a full prefill would have written.
    approx: Vec<(usize, usize)>,
}
impl State {
    /// Exact-batch mode. Off (default): a batch of 4 or more tokens through a quantised weight decodes
    /// each weight row to f32 once and multiplies f32 activations, which is faster for prefill but
    /// differs from single-token decoding (which rounds activations to 8 bits) by rounding noise.
    /// On: every batch size uses the single-token path, so results are bit-identical to token-by-token
    /// feeding. Dense (f32) weights are identical either way. Speculative, self-speculative and
    /// lookahead verification turn it on; the executor does for its equality guarantees.
    pub fn set_exact_batch(&mut self, on: bool) {
        self.exact_batch = on;
    }
    /// Whether exact-batch mode is on.
    pub fn exact_batch(&self) -> bool {
        self.exact_batch
    }
    /// Bytes currently held by the KV cache.
    pub fn kv_bytes(&self) -> usize {
        self.kv.iter().map(|l| l.k.bytes() + l.v.bytes()).sum()
    }
    /// The tokens this state has consumed, in order.
    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }
    /// How many leading tokens the last [`Model::prefill_cached`] call did not have to recompute.
    pub fn reused(&self) -> usize {
        self.reused
    }
    /// Can the state be rewound to `pos` tokens? Always, unless a sliding-window layer has already
    /// wrapped its ring buffer (the positions it would need again were overwritten).
    pub fn can_truncate(&self, pos: usize) -> bool {
        pos <= self.pos
            && (pos == self.pos
                || (self.kvx.identity_prefix(pos)
                    && self.kv.iter().all(|l| l.cap == 0 || self.pos <= l.cap)))
    }
    /// Rewind to `pos` tokens, dropping everything after them.
    pub fn truncate(&mut self, pos: usize) -> Result<(), ModelError> {
        if pos > self.pos {
            return Err(ModelError::Shape(format!(
                "cannot rewind to {pos}: the state holds {} tokens",
                self.pos
            )));
        }
        if pos == self.pos {
            return Ok(());
        }
        if !self.can_truncate(pos) {
            return Err(ModelError::Shape(
                "cannot rewind a sliding-window cache that has already wrapped".into(),
            ));
        }
        for l in &mut self.kv {
            let (kd, kvh) = (l.kd, l.kvh);
            l.k.truncate(pos, kd, kvh);
            l.v.truncate(pos, kd, kvh);
        }
        self.pos = pos;
        self.tokens.truncate(pos);
        self.last = None;
        self.kvx.after_truncate(pos);
        self.approx.retain_mut(|r| {
            r.1 = r.1.min(pos);
            r.0 < r.1
        });
        Ok(())
    }
    /// Back to an empty conversation (keeps the cache precision and the buffers).
    pub fn reset(&mut self) {
        for l in &mut self.kv {
            let (kd, kvh) = (l.kd, l.kvh);
            l.k.truncate(0, kd, kvh);
            l.v.truncate(0, kd, kvh);
        }
        self.pos = 0;
        self.tokens.clear();
        self.last = None;
        self.kvx.after_truncate(0);
        self.approx.clear();
        for m in &mut self.moe_last {
            m.clear();
        }
    }
    /// An independent copy to continue a different branch of the conversation. Sliding-window layers
    /// are small ring buffers, so the copy costs little there; full layers copy their cache.
    /// A state whose cache lives on an accelerator cannot be copied.
    pub fn fork(&self) -> Result<State, ModelError> {
        if self.dev.is_some() {
            return Err(ModelError::Backend(
                "a state with an accelerator-side cache cannot be forked".into(),
            ));
        }
        Ok(State {
            kv: self.kv.clone(),
            pos: self.pos,
            sc: Scratch::default(),
            moe_last: self.moe_last.clone(),
            tokens: self.tokens.clone(),
            last: self.last.clone(),
            reused: 0,
            dev: None,
            kvx: self.kvx.clone(),
            exact_batch: self.exact_batch,
            approx: self.approx.clone(),
        })
    }
}

// ------------------------------------------------------------------ cache splicing
// Moving cache rows between states, for reusing the work done on a text in another context.
// Keys are stored after rotary embedding, and rotations add up, so a key computed at position `p`
// is turned into the key for position `q` by rotating it by `q - p` more steps: exact, whatever
// the rotary table (without a second `mscale`). What cannot be moved is the *context*: a row
// holds what the token saw before it, so rows of the same text computed alone differ from the
// ones a full prefill would write (more in the later layers). `refresh_slots` repairs that and
// measures how big the difference was.

/// Rotate one head by `delta` positions (either way); the first `2 * inv.len()` dimensions only.
fn rope_shift(head: &mut [f32], delta: i64, inv: &[f64]) {
    let half = inv.len();
    for (i, f) in inv.iter().enumerate() {
        let (s, c) = crate::simd::rmath::sin_cos64(delta as f64 * f);
        let (a, b) = (head[i] as f64, head[i + half] as f64);
        head[i] = (a * c - b * s) as f32;
        head[i + half] = (a * s + b * c) as f32;
    }
}

impl State {
    /// Is every cache row exactly what a full prefill of [`State::tokens`] would have written?
    /// False after [`Model::splice_cache`] until [`Model::refresh_slots`] has recomputed the rows.
    pub fn is_exact(&self) -> bool {
        self.approx.is_empty()
    }
    /// How many cache rows are moved-in and not yet recomputed.
    pub fn approx_slots(&self) -> usize {
        self.approx.iter().map(|r| r.1 - r.0).sum()
    }
    fn mark_approx(&mut self, lo: usize, hi: usize) {
        self.approx.push((lo, hi));
        self.approx.sort_unstable();
        let mut out: Vec<(usize, usize)> = Vec::new();
        for r in self.approx.drain(..) {
            match out.last_mut() {
                Some(l) if r.0 <= l.1 => l.1 = l.1.max(r.1),
                _ => out.push(r),
            }
        }
        self.approx = out;
    }
    fn mark_exact(&mut self, lo: usize, hi: usize) {
        let mut out = Vec::new();
        for r in self.approx.drain(..) {
            if r.1 <= lo || r.0 >= hi {
                out.push(r);
                continue;
            }
            if r.0 < lo {
                out.push((r.0, lo));
            }
            if r.1 > hi {
                out.push((hi, r.1));
            }
        }
        self.approx = out;
    }
    /// One key row as f32 (for tests and diagnostics).
    pub fn key_row(&self, layer: usize, slot: usize) -> Option<Vec<f32>> {
        let l = self.kv.get(layer)?;
        if slot >= self.pos || slot >= l.k.len_slots(l.kd, l.kvh) {
            return None;
        }
        let mut out = vec![0f32; l.kd];
        l.k.read(slot, l.kd, l.kd / l.kvh.max(1), &mut out);
        Some(out)
    }
}

impl Model {
    /// The same checks for every operation that edits cache rows in place.
    fn editable(&self, st: &State) -> Result<(), ModelError> {
        let unsupported = |m: &str| Err(ModelError::Shape(m.into()));
        if self.accel.is_some() || st.dev.is_some() {
            return unsupported(
                "cache rows cannot be edited when the cache lives on an accelerator",
            );
        }
        if st.kv.len() != self.spec.layers as usize {
            return unsupported("the state was not made by new_state");
        }
        if st.is_compacted() || st.kvx.acc.is_some() {
            return unsupported(
                "cache rows cannot be edited after eviction or while scores are tracked",
            );
        }
        if st.kv.iter().any(|l| l.cap > 0) {
            return unsupported(
                "cache rows cannot be edited in a model with sliding-window layers",
            );
        }
        if st.tokens.len() != st.pos {
            return unsupported("the state does not know all its tokens");
        }
        Ok(())
    }

    /// Append `src`'s cache rows `range` to `dst`, rotating the keys to their new positions. `src`
    /// must be exact and made by this model with the same cache precision (F32 or F16). Cheap:
    /// no weights are read. `dst` is then approximate over those rows (see [`State::is_exact`]),
    /// except when it was empty and the rows came from the start of `src` (nothing moved).
    /// Nothing is changed on an `Err`.
    pub fn splice_cache(
        &self,
        dst: &mut State,
        src: &State,
        range: std::ops::Range<usize>,
    ) -> Result<(), ModelError> {
        self.editable(dst)?;
        self.editable(src)?;
        let bad = |m: &str| Err(ModelError::Shape(m.into()));
        if !src.is_exact() || !dst.is_exact() {
            return bad("only exact states can be spliced");
        }
        if range.start >= range.end || range.end > src.pos {
            return bad("the range of rows to splice is empty or outside the source state");
        }
        if dst.kv_kind() != src.kv_kind() || dst.kv_kind() == KvKind::Q8 {
            return bad("splicing needs the same cache precision, F32 or F16");
        }
        let n = range.end - range.start;
        if dst.pos + n > self.spec.context as usize {
            return Err(ModelError::ContextFull(self.spec.context as usize));
        }
        let delta = dst.pos as i64 - range.start as i64;
        let hd = self.spec.head_dim as usize;
        let inv = &self.rope_global.0;
        let (mut kr, mut vr) = (Vec::new(), Vec::new());
        let at = dst.pos;
        for (ld, ls) in dst.kv.iter_mut().zip(&src.kv) {
            let kd = ld.kd;
            if ls.kd != kd || kd % hd != 0 || ls.k.len_slots(kd, ls.kvh) < range.end {
                return bad("the two caches do not have the same shape");
            }
            kr.resize(kd, 0.0);
            vr.resize(kd, 0.0);
            for (j, slot) in range.clone().enumerate() {
                ls.k.read(slot, kd, hd, &mut kr);
                ls.v.read(slot, kd, hd, &mut vr);
                if delta != 0 {
                    for head in kr.chunks_mut(hd) {
                        rope_shift(head, delta, inv);
                    }
                }
                ld.k.write(at + j, kd, hd, &kr);
                ld.v.write(at + j, kd, hd, &vr);
            }
        }
        dst.tokens.extend_from_slice(&src.tokens[range]);
        dst.pos += n;
        dst.last = None;
        if !(at == 0 && delta == 0) {
            dst.mark_approx(at, at + n);
        }
        Ok(())
    }

    /// Recompute cache rows `from..from + len` of `st` from the tokens it already holds, in place,
    /// attending to the rows before `from` as they are now. Rows after the range are not touched
    /// (so repair runs from the start of a spliced text towards its end). Returns, per row, how
    /// much the recomputation changed it: the mean over layers of the relative change of the key
    /// and of the value. Zero means the moved-in row was already what a full prefill writes.
    pub fn refresh_slots(
        &self,
        st: &mut State,
        from: usize,
        len: usize,
    ) -> Result<Vec<f32>, ModelError> {
        self.editable(st)?;
        if len == 0 || from + len > st.pos {
            return Err(ModelError::Shape(
                "the rows to refresh are outside the state".into(),
            ));
        }
        let toks: Vec<usize> = st.tokens[from..from + len].to_vec();
        if toks.iter().any(|&t| t >= self.spec.vocab as usize) {
            let bad = toks
                .iter()
                .copied()
                .find(|&t| t >= self.spec.vocab as usize);
            return Err(ModelError::Token(bad.unwrap_or(0)));
        }
        let (h, nl) = (self.spec.hidden as usize, self.w.layers.len());
        let kd = st.kv.first().map(|l| l.kd).unwrap_or(0);
        let hd = self.spec.head_dim as usize;
        // rows as they are now
        let mut old = vec![0f32; nl * len * 2 * kd];
        for (li, l) in st.kv.iter().enumerate() {
            for t in 0..len {
                let o = (li * len + t) * 2 * kd;
                l.k.read(from + t, kd, hd, &mut old[o..o + kd]);
                l.v.read(from + t, kd, hd, &mut old[o + kd..o + 2 * kd]);
            }
        }
        let moe_len: Vec<usize> = st.moe_last.iter().map(Vec::len).collect();
        let mut done = 0usize;
        while done < len {
            let c = (len - done).min(PREFILL_CHUNK);
            let mut x = std::mem::take(&mut st.sc.x);
            fit(&mut x, c * h);
            let r = self
                .embed_rows(&toks[done..done + c], &mut x, &mut st.sc.rowtmp)
                .and_then(|()| self.layers_core(st, 0..nl, &mut x, c, from + done));
            st.sc.x = x;
            r?;
            done += c;
        }
        for (m, n) in st.moe_last.iter_mut().zip(moe_len) {
            m.truncate(n);
        }
        let (mut kn, mut vn) = (vec![0f32; kd], vec![0f32; kd]);
        let mut drift = vec![0f32; len];
        for (li, l) in st.kv.iter().enumerate() {
            for (t, d) in drift.iter_mut().enumerate() {
                let o = (li * len + t) * 2 * kd;
                l.k.read(from + t, kd, hd, &mut kn);
                l.v.read(from + t, kd, hd, &mut vn);
                let rel = |a: &[f32], b: &[f32]| {
                    let (mut diff, mut base) = (0f64, 0f64);
                    for (x, y) in a.iter().zip(b) {
                        diff += ((x - y) as f64).powi(2);
                        base += (*y as f64).powi(2);
                    }
                    (diff.sqrt() / (base.sqrt() + 1e-6)) as f32
                };
                *d += 0.5 * (rel(&kn, &old[o..o + kd]) + rel(&vn, &old[o + kd..o + 2 * kd]));
            }
        }
        drift.iter_mut().for_each(|d| *d /= nl.max(1) as f32);
        st.last = None;
        st.mark_exact(from, from + len);
        Ok(drift)
    }
}

#[allow(clippy::too_many_arguments)]
fn attend(
    kv: &LayerKv,
    kd: usize,
    hd: usize,
    nh: usize,
    group: usize,
    first: usize,
    n: usize,
    scale: f32,
    cap: f64,
    q: &[f32],
    att: &mut [f32],
    scores: &mut [f32],
    nt: usize,
) {
    let head = |h: usize, out: &mut [f32], sc: &mut [f32]| {
        let (kvh, qh) = (h / group, &q[h * hd..(h + 1) * hd]);
        for (i, s) in sc.iter_mut().enumerate() {
            *s = softcap(kv.k.dot(kv.slot(first + i), kd, hd, kvh, qh) * scale, cap);
        }
        softmax(sc);
        out.fill(0.0);
        for (i, p) in sc.iter().enumerate() {
            kv.v.axpy(kv.slot(first + i), kd, hd, kvh, *p, out);
        }
    };
    let nt = par_threads(nt).min(nh);
    if nt <= 1 || nh * n * hd < (1 << 19) {
        for (h, (o, sc)) in att.chunks_mut(hd).zip(scores.chunks_mut(n)).enumerate() {
            head(h, o, sc);
        }
    } else {
        let dst = Mutex::new(&mut *att);
        let head = &head;
        pool::global().parallel_for(nh, nh.div_ceil(nt), |r| {
            let (mut loc, mut sc) = (vec![0.0; r.len() * hd], vec![0.0; n]);
            for (j, h) in r.clone().enumerate() {
                head(h, &mut loc[j * hd..(j + 1) * hd], &mut sc);
            }
            lock(&dst)[r.start * hd..r.end * hd].copy_from_slice(&loc);
        });
    }
}

impl Model {
    pub fn new(spec: Spec, w: Weights) -> Result<Model, ModelError> {
        let (h, ff, v) = (spec.hidden as usize, spec.ffn as usize, spec.vocab as usize);
        let (hd, rd) = (spec.head_dim as usize, spec.rotary_dim as usize);
        let (qd, kd) = (
            spec.heads.saturating_mul(spec.head_dim) as usize,
            spec.kv_heads.saturating_mul(spec.head_dim) as usize,
        );
        if spec.kv_heads == 0 || !spec.heads.is_multiple_of(spec.kv_heads) {
            return Err(ModelError::Shape("heads not divisible by kv heads".into()));
        }
        if rd == 0 || rd % 2 != 0 || rd > hd {
            return Err(ModelError::Shape(format!(
                "rotary_dim {rd} must be even and within head_dim {hd}"
            )));
        }
        let chk = |name: &str, got: usize, want: usize| {
            if got == want {
                Ok(())
            } else {
                Err(ModelError::Shape(format!(
                    "{name}: {got} values, expected {want}"
                )))
            }
        };
        let chk_w = |name: &str, w: &Weight, rows: usize, cols: usize| match w {
            _ if rows.checked_mul(cols).is_none() => Err(ModelError::Shape(format!(
                "{name}: dimensions {rows}x{cols} are absurd"
            ))),
            Weight::Dense(x) => chk(name, x.len(), rows * cols),
            Weight::Quant(q) if rows * cols == 0 => chk(name, q.rows.saturating_mul(q.cols), 0),
            Weight::Quant(q) if q.rows == rows && q.cols == cols => Ok(()),
            Weight::Quant(q) => Err(ModelError::Shape(format!(
                "{name}: stored as {}x{}, expected {rows}x{cols}",
                q.rows, q.cols
            ))),
        };
        let chk_opt = |name: &str, got: &Option<Vec<f32>>, want: usize, required: bool| match got {
            Some(g) => chk(name, g.len(), want),
            None if required => Err(ModelError::Shape(format!("{name} is missing"))),
            None => Ok(()),
        };
        chk_w("embed", &w.embed, v, h)?;
        chk("final_norm", w.final_norm.len(), h)?;
        chk_opt("final_norm bias", &w.final_norm_b, h, false)?;
        if let Some(l) = &w.lm_head {
            chk_w("lm_head", l, v, h)?;
        }
        chk_opt("lm_head bias", &w.lm_head_b, v, false)?;
        chk("layers", w.layers.len(), spec.layers as usize)?;
        for (i, l) in w.layers.iter().enumerate() {
            let gate = if spec.gated_mlp { ff } else { 0 };
            for (n, wt, rows, cols) in [
                ("wq", &l.wq, qd, h),
                ("wk", &l.wk, kd, h),
                ("wv", &l.wv, kd, h),
                ("wo", &l.wo, h, qd),
            ] {
                chk_w(&format!("layer {i} {n}"), wt, rows, cols)?;
            }
            if let Some(m) = &l.moe {
                let bad = |what: String| ModelError::Experts(format!("layer {i}: {what}"));
                if w.experts.is_none() {
                    return Err(bad("the model has no expert provider".into()));
                }
                if m.hidden != h {
                    return Err(bad(format!(
                        "router expects hidden {}, the model has {h}",
                        m.hidden
                    )));
                }
                if m.layer != i {
                    return Err(bad(format!("the block was built for layer {}", m.layer)));
                }
                if !spec.gated_mlp {
                    return Err(bad(
                        "experts are gated MLPs, the architecture's MLP is not".into()
                    ));
                }
                let rc = m.router.config();
                if spec.experts > 0 && rc.experts as u64 != spec.experts {
                    return Err(bad(format!(
                        "router has {} experts, the architecture {}",
                        rc.experts, spec.experts
                    )));
                }
                for s in &m.shared {
                    s.validate(h, s.ffn(h)).map_err(|e| bad(e.to_string()))?;
                }
            } else {
                for (n, wt, rows, cols) in [
                    ("w_gate", &l.w_gate, gate, h),
                    ("w_up", &l.w_up, ff, h),
                    ("w_down", &l.w_down, h, ff),
                ] {
                    chk_w(&format!("layer {i} {n}"), wt, rows, cols)?;
                }
            }
            chk(&format!("layer {i} attn_norm"), l.attn_norm.len(), h)?;
            chk(&format!("layer {i} ffn_norm"), l.ffn_norm.len(), h)?;
            let c = |n: &str, g: &Option<Vec<f32>>, want: usize, req: bool| {
                chk_opt(&format!("layer {i} {n}"), g, want, req)
            };
            c(
                "attn_norm bias",
                &l.attn_norm_b,
                h,
                spec.norm == NormKind::Layer,
            )?;
            c(
                "ffn_norm bias",
                &l.ffn_norm_b,
                h,
                spec.norm == NormKind::Layer,
            )?;
            c("bq", &l.bq, qd, spec.qkv_bias)?;
            c("bk", &l.bk, kd, spec.qkv_bias)?;
            c("bv", &l.bv, kd, spec.qkv_bias)?;
            c("bo", &l.bo, h, false)?;
            c("b_gate", &l.b_gate, ff, false)?;
            c("b_up", &l.b_up, ff, false)?;
            c("b_down", &l.b_down, h, false)?;
            c("q_norm", &l.q_norm, hd, spec.qk_norm)?;
            c("k_norm", &l.k_norm, hd, spec.qk_norm)?;
            c("post_attn_norm", &l.post_attn_norm, h, spec.post_norms)?;
            c("post_ffn_norm", &l.post_ffn_norm, h, spec.post_norms)?;
        }
        if spec.norm == NormKind::Layer {
            chk_opt("final_norm bias", &w.final_norm_b, h, true)?;
        }
        let rope_global = spec.rope_freqs();
        let rope_local = if spec.rope_theta_local > 0.0 {
            rope_inv_freq(spec.rope_theta_local, rd, RopeScaling::None)
        } else {
            rope_global.clone()
        };
        Ok(Model {
            spec,
            w,
            rope_global,
            rope_local,
            threads: 0,
            accel: None,
        })
    }

    /// Bytes taken by the big weight matrices as stored (a memory-mapped model counts its file size).
    pub fn weight_bytes(&self) -> usize {
        self.w.stored_bytes()
    }

    pub fn new_state(&self) -> State {
        self.new_state_with(KvKind::F32)
    }
    /// State with the chosen KV precision. Sliding-window layers keep a ring of `sliding_window` positions.
    pub fn new_state_with(&self, kind: KvKind) -> State {
        let (kvh, hd) = (self.spec.kv_heads as usize, self.spec.head_dim as usize);
        let kv = (0..self.spec.layers as usize)
            .map(|i| LayerKv {
                k: Plane::new(kind),
                v: Plane::new(kind),
                cap: if self.spec.is_sliding(i) {
                    self.spec.sliding_window as usize
                } else {
                    0
                },
                kd: kvh * hd,
                kvh,
            })
            .collect();
        State {
            kv,
            pos: 0,
            sc: Scratch::default(),
            moe_last: vec![vec![]; self.spec.layers as usize],
            ..Default::default()
        }
    }

    /// Limit the threads used for the heavy products (`0` = every thread of the global pool).
    pub fn set_threads(&mut self, n: usize) {
        self.threads = n;
    }
    /// The thread limit in force (`0` = unlimited).
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Feed one token, return the logits for the next one.
    pub fn forward(&self, st: &mut State, token: usize) -> Result<Vec<f32>, ModelError> {
        self.forward_batch(st, &[token])
    }

    /// Feed many tokens (a prompt) and return the logits after the last one. Equal, within rounding,
    /// to calling [`Model::forward`] for each token, but each weight row is read once per chunk.
    pub fn forward_batch(&self, st: &mut State, tokens: &[usize]) -> Result<Vec<f32>, ModelError> {
        let s = &self.spec;
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        if let Some(&t) = tokens.iter().find(|&&t| t >= s.vocab as usize) {
            return Err(ModelError::Token(t));
        }
        if st.kv.len() != s.layers as usize {
            return Err(ModelError::Shape(
                "the state was not made by new_state".into(),
            ));
        }
        if st.pos + tokens.len() > s.context as usize {
            return Err(ModelError::ContextFull(s.context as usize));
        }
        if let Some(a) = &self.accel {
            if st.dev.as_ref().is_some_and(|d| d.owner != a.id) {
                return Err(ModelError::Backend(
                    "the state belongs to another backend".into(),
                ));
            }
            if st.pos + tokens.len() > a.kv_capacity {
                return Err(ModelError::ContextFull(a.kv_capacity));
            }
            if st.dev.is_none() {
                if st.pos > 0 {
                    return Err(ModelError::Backend("this state already holds a host-side cache; make it with the model that owns the backend".into()));
                }
                st.dev = Some(self.new_dev_state(a)?);
            }
        }
        let chunks = tokens.len().div_ceil(PREFILL_CHUNK);
        let mut last = vec![];
        st.last = None;
        for (i, c) in tokens.chunks(PREFILL_CHUNK).enumerate() {
            last = self.run_chunk(st, c, i + 1 == chunks)?;
            st.tokens.extend_from_slice(c);
        }
        Ok(last)
    }

    /// Like [`Model::forward_batch`] for a whole prompt, but keeps what `st` already knows: the
    /// longest common prefix of `tokens` and the tokens the state has consumed is not recomputed
    /// (a chat server calls this with the full conversation every turn). The state is rewound when
    /// it is ahead of the prefix; if it cannot be (a wrapped sliding-window ring) it starts over.
    /// Always feeds at least the last token unless its logits are still cached. See [`State::reused`].
    pub fn prefill_cached(&self, st: &mut State, tokens: &[usize]) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        if st.tokens.len() != st.pos {
            st.reset();
        }
        let common = st.reusable_prefix(tokens);
        if common == tokens.len()
            && st.pos == common
            && let Some(l) = &st.last
        {
            st.reused = common;
            return Ok(l.clone());
        }
        let mut keep = common.min(tokens.len() - 1);
        if st.pos > keep || st.is_compacted() {
            if st.can_truncate(keep) {
                st.truncate(keep)?;
                st.rebase_to_prefix();
            } else {
                st.reset();
                keep = 0;
            }
        }
        st.reused = keep;
        let logits = self.forward_batch(st, &tokens[keep..])?;
        st.last = Some(logits.clone());
        Ok(logits)
    }

    fn run_chunk(
        &self,
        st: &mut State,
        tokens: &[usize],
        want_logits: bool,
    ) -> Result<Vec<f32>, ModelError> {
        let (h, b, pos0) = (self.spec.hidden as usize, tokens.len(), st.pos);
        let mut x = std::mem::take(&mut st.sc.x);
        fit(&mut x, b * h);
        let r = self
            .embed_rows(tokens, &mut x, &mut st.sc.rowtmp)
            .and_then(|()| self.layers_core(st, 0..self.w.layers.len(), &mut x, b, pos0));
        if let Err(e) = r {
            st.sc.x = x;
            return Err(e);
        }
        st.pos += b;
        let out = if want_logits {
            Ok(self.logits_from_hidden_inner(&x[(b - 1) * h..b * h])?)
        } else {
            Ok(vec![])
        };
        st.sc.x = x;
        out
    }

    fn embed_rows(
        &self,
        tokens: &[usize],
        x: &mut [f32],
        rowtmp: &mut Vec<f32>,
    ) -> Result<(), ModelError> {
        let s = &self.spec;
        let h = s.hidden as usize;
        for (t, &tok) in tokens.iter().enumerate() {
            let row = &mut x[t * h..(t + 1) * h];
            self.w.embed.row_into(tok, h, row, rowtmp)?;
            if s.embed_scale != 1.0 {
                row.iter_mut().for_each(|a| *a *= s.embed_scale as f32);
            }
        }
        Ok(())
    }

    /// The hidden state of one token before layer 0 (embedding row, scaled as the architecture asks).
    pub fn embed_token(&self, token: usize) -> Result<Vec<f32>, ModelError> {
        if token >= self.spec.vocab as usize {
            return Err(ModelError::Token(token));
        }
        let mut x = vec![0.0; self.spec.hidden as usize];
        self.embed_rows(&[token], &mut x, &mut Vec::new())?;
        Ok(x)
    }

    /// Make `st` usable for `add` more positions (the checks `forward_batch` makes, accelerator included).
    fn ready(&self, st: &mut State, add: usize) -> Result<(), ModelError> {
        let s = &self.spec;
        if st.kv.len() != s.layers as usize {
            return Err(ModelError::Shape(
                "the state was not made by new_state".into(),
            ));
        }
        if st.pos + add > s.context as usize {
            return Err(ModelError::ContextFull(s.context as usize));
        }
        if let Some(a) = &self.accel {
            if st.dev.as_ref().is_some_and(|d| d.owner != a.id) {
                return Err(ModelError::Backend(
                    "the state belongs to another backend".into(),
                ));
            }
            if st.pos + add > a.kv_capacity {
                return Err(ModelError::ContextFull(a.kv_capacity));
            }
            if st.dev.is_none() {
                if st.pos > 0 {
                    return Err(ModelError::Backend("this state already holds a host-side cache; make it with the model that owns the backend".into()));
                }
                st.dev = Some(self.new_dev_state(a)?);
            }
        }
        Ok(())
    }

    /// Run the layers in `range` over the hidden state `x` (one token, `hidden` values) sitting at
    /// position `pos`, updating those layers' KV cache. Call it for consecutive ranges covering
    /// `0..layers` (on one machine or several), then [`Model::finish_token`]. `pos` must equal `st.pos`.
    pub fn run_layers(
        &self,
        st: &mut State,
        range: std::ops::Range<usize>,
        x: &mut [f32],
        pos: usize,
    ) -> Result<(), ModelError> {
        let h = self.spec.hidden as usize;
        if x.len() != h {
            return Err(ModelError::Shape(format!(
                "hidden state has {} values, expected {h}",
                x.len()
            )));
        }
        if range.start > range.end || range.end > self.w.layers.len() {
            return Err(ModelError::Shape(format!(
                "layer range {range:?} outside 0..{}",
                self.w.layers.len()
            )));
        }
        if pos != st.pos {
            return Err(ModelError::Shape(format!(
                "position {pos} but the state is at {}",
                st.pos
            )));
        }
        self.ready(st, 1)?;
        self.layers_core(st, range, x, 1, pos)
    }

    /// [`Model::run_layers`] for `b = x.len() / hidden` consecutive tokens starting at `pos0 == st.pos`
    /// (one weight read per layer for the whole batch). Close with [`Model::finish_batch`] once every
    /// layer has seen the batch.
    pub fn run_layers_batch(
        &self,
        st: &mut State,
        range: std::ops::Range<usize>,
        x: &mut [f32],
        pos0: usize,
    ) -> Result<(), ModelError> {
        let h = self.spec.hidden as usize;
        if x.is_empty() || !x.len().is_multiple_of(h) {
            return Err(ModelError::Shape(format!(
                "{} hidden values is not a whole number of {h}-wide tokens",
                x.len()
            )));
        }
        let b = x.len() / h;
        if range.start > range.end || range.end > self.w.layers.len() {
            return Err(ModelError::Shape(format!(
                "layer range {range:?} outside 0..{}",
                self.w.layers.len()
            )));
        }
        if pos0 != st.pos {
            return Err(ModelError::Shape(format!(
                "position {pos0} but the state is at {}",
                st.pos
            )));
        }
        self.ready(st, b)?;
        self.layers_core(st, range, x, b, pos0)
    }

    /// Close a batch that went through `run_layers_batch` for every layer: advance the position and
    /// record the tokens (as `forward_batch` does).
    pub fn finish_batch(&self, st: &mut State, tokens: &[usize]) {
        st.pos += tokens.len();
        st.tokens.extend_from_slice(tokens);
        st.last = None;
    }

    /// Close a token that went through `run_layers` for every layer: advance the position.
    pub fn finish_token(&self, st: &mut State) {
        st.pos += 1;
        st.last = None;
    }

    /// Final norm, output head, bias and soft-cap on a hidden state: the logits for the next token.
    /// Panics never; a malformed `x` (wrong length) yields an empty vector.
    pub fn logits_from_hidden(&self, x: &[f32]) -> Vec<f32> {
        self.logits_from_hidden_inner(x).unwrap_or_default()
    }

    fn logits_from_hidden_inner(&self, last: &[f32]) -> Result<Vec<f32>, ModelError> {
        let s = &self.spec;
        let h = s.hidden as usize;
        if last.len() != h {
            return Err(ModelError::Shape(format!(
                "hidden state has {} values, expected {h}",
                last.len()
            )));
        }
        let mut xl = vec![0.0; h];
        norm(
            s,
            last,
            &self.w.final_norm,
            self.w.final_norm_b.as_deref(),
            &mut xl,
        );
        let head = self.w.lm_head.as_ref().unwrap_or(&self.w.embed);
        let mut logits = vec![0.0; s.vocab as usize];
        head.matvec_nt(h, &xl, &mut logits, self.threads)?;
        add_bias(&mut logits, &self.w.lm_head_b);
        if s.final_softcap > 0.0 {
            logits
                .iter_mut()
                .for_each(|l| *l = softcap(*l, s.final_softcap));
        }
        Ok(logits)
    }

    /// A fixed-size vector for a token sequence: the final-normed hidden states of the last layer,
    /// pooled by `pooling`, optionally scaled to unit length. Uses a fresh state (nothing is cached or
    /// kept), runs the sequence in chunks of 256 tokens, and returns [`ModelError::ContextFull`] past the
    /// context window. Decoder models give useful vectors with `Pooling::Last` or `Pooling::Mean`.
    pub fn embed(
        &self,
        tokens: &[usize],
        pooling: Pooling,
        normalize: bool,
    ) -> Result<Vec<f32>, ModelError> {
        self.embed_chunked(tokens, pooling, normalize, 256)
    }

    fn embed_chunked(
        &self,
        tokens: &[usize],
        pooling: Pooling,
        normalize: bool,
        chunk_len: usize,
    ) -> Result<Vec<f32>, ModelError> {
        let s = &self.spec;
        let h = s.hidden as usize;
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        if let Some(&t) = tokens.iter().find(|&&t| t >= s.vocab as usize) {
            return Err(ModelError::Token(t));
        }
        if tokens.len() > s.context as usize {
            return Err(ModelError::ContextFull(s.context as usize));
        }
        let mut st = self.new_state();
        let (mut sum, mut first, mut last) = (vec![0.0f64; h], vec![0.0f32; h], vec![0.0f32; h]);
        let mut tmp = Vec::new();
        let mut xl = vec![0.0f32; h];
        let mut done = 0usize;
        for chunk in tokens.chunks(chunk_len.max(1)) {
            let mut x = vec![0.0f32; chunk.len() * h];
            self.embed_rows(chunk, &mut x, &mut tmp)?;
            let pos = st.pos;
            self.run_layers_batch(&mut st, 0..self.w.layers.len(), &mut x, pos)?;
            self.finish_batch(&mut st, chunk);
            for (i, row) in x.chunks(h).enumerate() {
                norm(
                    s,
                    row,
                    &self.w.final_norm,
                    self.w.final_norm_b.as_deref(),
                    &mut xl,
                );
                if done + i == 0 {
                    first.copy_from_slice(&xl);
                }
                if done + i + 1 == tokens.len() {
                    last.copy_from_slice(&xl);
                }
                for (a, b) in sum.iter_mut().zip(&xl) {
                    *a += f64::from(*b);
                }
            }
            done += chunk.len();
        }
        let mut v: Vec<f32> = match pooling {
            Pooling::First => first,
            Pooling::Last => last,
            Pooling::Mean => sum
                .iter()
                .map(|a| (*a / tokens.len() as f64) as f32)
                .collect(),
        };
        if normalize {
            let n = v
                .iter()
                .map(|a| f64::from(*a) * f64::from(*a))
                .sum::<f64>()
                .sqrt();
            if n > 0.0 && n.is_finite() {
                v.iter_mut().for_each(|a| *a = (f64::from(*a) / n) as f32);
            }
        }
        if v.iter().any(|a| !a.is_finite()) {
            return Err(ModelError::Shape("the embedding is not finite".into()));
        }
        Ok(v)
    }

    /// The layer loop: `b` consecutive tokens starting at `pos0`, hidden states in `x`, layers in `range`.
    fn layers_core(
        &self,
        st: &mut State,
        range: std::ops::Range<usize>,
        x: &mut [f32],
        b: usize,
        pos0: usize,
    ) -> Result<(), ModelError> {
        let s = &self.spec;
        let (h, hd, nh, nkv) = (
            s.hidden as usize,
            s.head_dim as usize,
            s.heads as usize,
            s.kv_heads as usize,
        );
        let (qd, kd, ff) = (nh * hd, nkv * hd, s.ffn as usize);
        let (group, nt) = (nh / nkv, self.threads);
        let scale = if s.attn_scale > 0.0 {
            s.attn_scale as f32
        } else {
            1.0 / (hd as f32).sqrt()
        };
        let half = self.rope_global.0.len();
        let gap = st.rope_gap();
        let State {
            kv,
            sc,
            moe_last,
            dev,
            kvx,
            exact_batch,
            ..
        } = st;
        let exact = *exact_batch;
        for (v, n) in [
            (&mut sc.xb, b * h),
            (&mut sc.xm, b * h),
            (&mut sc.q, b * qd),
            (&mut sc.k, b * kd),
            (&mut sc.v, b * kd),
            (&mut sc.att, b * qd),
            (&mut sc.proj, b * h),
            (&mut sc.tmp, h),
            (&mut sc.g, b * ff),
            (&mut sc.u, b * ff),
            (&mut sc.mlp, b * h),
            (&mut sc.headtmp, hd),
        ] {
            fit(v, n);
        }
        let Scratch {
            xb,
            xm,
            q,
            k,
            v,
            att,
            proj,
            tmp,
            g,
            u,
            mlp,
            scores,
            headtmp,
            yt,
            ..
        } = sc;
        let (mut cos_g, mut sin_g, mut cos_l, mut sin_l) = (
            vec![0.0; b * half],
            vec![0.0; b * half],
            vec![0.0; b * half],
            vec![0.0; b * half],
        );
        for t in 0..b {
            rope_table(
                pos0 + t + gap,
                &self.rope_global,
                &mut cos_g[t * half..(t + 1) * half],
                &mut sin_g[t * half..(t + 1) * half],
            );
            rope_table(
                pos0 + t + gap,
                &self.rope_local,
                &mut cos_l[t * half..(t + 1) * half],
                &mut sin_l[t * half..(t + 1) * half],
            );
        }
        let mut skip_to = 0;
        for li in range.clone() {
            let l = &self.w.layers[li];
            if li < skip_to {
                continue;
            }
            if let Some(a) = self.accel.as_ref().filter(|a| a.layers[li].is_some()) {
                // a run of consecutive accelerated layers: the hidden state crosses the boundary once each way
                let end = (li..range.end.min(a.layers.len()))
                    .find(|&j| a.layers[j].is_none())
                    .unwrap_or(range.end.min(a.layers.len()));
                let d = dev
                    .as_mut()
                    .ok_or_else(|| ModelError::Backend("no accelerator state".into()))?;
                self.run_device(a, d, li..end, b, pos0, &mut x[..])?;
                skip_to = end;
                continue;
            }
            let sliding = s.is_sliding(li);
            let (cos, sin) = if sliding && s.rope_theta_local > 0.0 {
                (&cos_l, &sin_l)
            } else {
                (&cos_g, &sin_g)
            };
            for t in 0..b {
                norm(
                    s,
                    &x[t * h..(t + 1) * h],
                    &l.attn_norm,
                    l.attn_norm_b.as_deref(),
                    &mut xb[t * h..(t + 1) * h],
                );
            }
            l.wq.matmat_opt(h, xb, b, q, yt, nt, exact)?;
            l.wk.matmat_opt(h, xb, b, k, yt, nt, exact)?;
            l.wv.matmat_opt(h, xb, b, v, yt, nt, exact)?;
            for t in 0..b {
                let (qt, kt) = (&mut q[t * qd..(t + 1) * qd], &mut k[t * kd..(t + 1) * kd]);
                add_bias(qt, &l.bq);
                add_bias(kt, &l.bk);
                add_bias(&mut v[t * kd..(t + 1) * kd], &l.bv);
                if let (Some(qn), Some(kn)) = (&l.q_norm, &l.k_norm) {
                    for hq in qt.chunks_mut(hd) {
                        norm(s, hq, qn, None, headtmp);
                        hq.copy_from_slice(headtmp);
                    }
                    for hk in kt.chunks_mut(hd) {
                        norm(s, hk, kn, None, headtmp);
                        hk.copy_from_slice(headtmp);
                    }
                }
                let (c, sn) = (
                    &cos[t * half..(t + 1) * half],
                    &sin[t * half..(t + 1) * half],
                );
                for hq in qt.chunks_mut(hd) {
                    rope_cs(hq, c, sn);
                }
                for hk in kt.chunks_mut(hd) {
                    rope_cs(hk, c, sn);
                }
            }
            let lkv = &mut kv[li];
            for t in 0..b {
                let p = pos0 + t;
                let slot = lkv.slot(p);
                lkv.k.write(slot, kd, hd, &k[t * kd..(t + 1) * kd]);
                lkv.v.write(slot, kd, hd, &v[t * kd..(t + 1) * kd]);
                let first = if sliding {
                    (p + 1).saturating_sub(s.sliding_window as usize)
                } else {
                    0
                };
                let n = p + 1 - first;
                if scores.len() < nh * n {
                    scores.resize(nh * n, 0.0);
                }
                attend(
                    lkv,
                    kd,
                    hd,
                    nh,
                    group,
                    first,
                    n,
                    scale,
                    s.attn_softcap,
                    &q[t * qd..(t + 1) * qd],
                    &mut att[t * qd..(t + 1) * qd],
                    &mut scores[..nh * n],
                    nt,
                );
                if let (Some(acc), 0) = (kvx.acc.as_mut(), lkv.cap) {
                    acc_add(
                        lkv,
                        &mut acc[li],
                        kd,
                        hd,
                        nh,
                        group,
                        n,
                        scale,
                        s.attn_softcap,
                        &q[t * qd..(t + 1) * qd],
                    );
                }
            }
            l.wo.matmat_opt(qd, att, b, proj, yt, nt, exact)?;
            for t in 0..b {
                let pr = &mut proj[t * h..(t + 1) * h];
                add_bias(pr, &l.bo);
                if let Some(pn) = &l.post_attn_norm {
                    norm(s, pr, pn, None, tmp);
                    pr.copy_from_slice(tmp);
                }
                let xt = &mut x[t * h..(t + 1) * h];
                // MLP input: the block input (parallel residual) or the residual stream after attention (sequential)
                if !s.parallel_residual {
                    xt.iter_mut().zip(&*pr).for_each(|(a, b)| *a += b);
                }
                norm(
                    s,
                    xt,
                    &l.ffn_norm,
                    l.ffn_norm_b.as_deref(),
                    &mut xm[t * h..(t + 1) * h],
                );
            }
            if let Some(moe) = &l.moe {
                let ex = self
                    .w
                    .experts
                    .as_ref()
                    .ok_or_else(|| ModelError::Experts("no expert provider".into()))?;
                let routing = moe.route_batch(&xm[..b * h], b).map_err(moe_err)?;
                if ex.prefetch {
                    // cheap predictor: the next MoE layer will probably want what the previous token gave it
                    if let Some(nl) =
                        (li + 1..self.w.layers.len()).find(|&j| self.w.layers[j].moe.is_some())
                        && !moe_last[nl].is_empty()
                    {
                        ex.provider.prefetch(nl, &moe_last[nl]);
                    }
                }
                moe_last[li] = routing.experts();
                let out = moe
                    .apply_with(
                        ex.provider.as_ref(),
                        s.act,
                        &xm[..b * h],
                        b,
                        &routing,
                        exact,
                    )
                    .map_err(moe_err)?;
                mlp.copy_from_slice(&out);
            } else {
                l.w_up.matmat_opt(h, xm, b, u, yt, nt, exact)?;
                for r in u.chunks_mut(ff) {
                    add_bias(r, &l.b_up);
                }
                if s.gated_mlp {
                    l.w_gate.matmat_opt(h, xm, b, g, yt, nt, exact)?;
                    for (gr, ur) in g.chunks_mut(ff).zip(u.chunks(ff)) {
                        add_bias(gr, &l.b_gate);
                        gr.iter_mut()
                            .zip(ur)
                            .for_each(|(a, b)| *a = activate(s.act, *a) * b);
                    }
                } else {
                    g.iter_mut()
                        .zip(u.iter())
                        .for_each(|(a, b)| *a = activate(s.act, *b));
                }
                l.w_down.matmat_opt(ff, g, b, mlp, yt, nt, exact)?;
            }
            for t in 0..b {
                let m = &mut mlp[t * h..(t + 1) * h];
                add_bias(m, &l.b_down);
                if let Some(pn) = &l.post_ffn_norm {
                    norm(s, m, pn, None, tmp);
                    m.copy_from_slice(tmp);
                }
                let xt = &mut x[t * h..(t + 1) * h];
                if s.parallel_residual {
                    xt.iter_mut()
                        .zip(&proj[t * h..(t + 1) * h])
                        .zip(&*m)
                        .for_each(|((a, b), c)| *a += b + c);
                } else {
                    xt.iter_mut().zip(&*m).for_each(|(a, b)| *a += b);
                }
            }
        }
        Ok(())
    }

    /// Prefill `prompt`, then sample until a stop token, `max_tokens`, cancellation, a `false` from
    /// `on_token`, or a full context. The stop token itself is not passed to `on_token`.
    pub fn generate(
        &self,
        st: &mut State,
        prompt: &[usize],
        sampler: &mut Sampler,
        opts: &GenerateOptions,
        on_token: &mut dyn FnMut(usize) -> bool,
    ) -> Result<GenerateStats, ModelError> {
        let mut stats = GenerateStats {
            prompt_tokens: prompt.len(),
            generated: 0,
            prefill_secs: 0.0,
            decode_secs: 0.0,
            stop: StopReason::MaxTokens,
        };
        let t0 = Instant::now();
        let mut logits = self.forward_batch(st, prompt)?;
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
            if st.pos >= self.spec.context as usize {
                stats.stop = StopReason::ContextFull;
                break;
            }
            logits = self.forward(st, tok)?;
        }
        stats.decode_secs = t1.elapsed().as_secs_f64();
        Ok(stats)
    }
}

pub struct GenerateOptions<'a> {
    pub max_tokens: usize,
    pub stop_tokens: Vec<usize>,
    pub cancel: Option<&'a AtomicBool>,
}
impl Default for GenerateOptions<'_> {
    fn default() -> Self {
        GenerateOptions {
            max_tokens: 256,
            stop_tokens: vec![],
            cancel: None,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    MaxTokens,
    StopToken,
    Cancelled,
    Callback,
    ContextFull,
}
#[derive(Debug, Clone)]
pub struct GenerateStats {
    pub prompt_tokens: usize,
    pub generated: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub stop: StopReason,
}
impl GenerateStats {
    pub fn prefill_tps(&self) -> f64 {
        if self.prefill_secs > 0.0 {
            self.prompt_tokens as f64 / self.prefill_secs
        } else {
            0.0
        }
    }
    pub fn decode_tps(&self) -> f64 {
        if self.decode_secs > 0.0 {
            self.generated as f64 / self.decode_secs
        } else {
            0.0
        }
    }
}

// ------------------------------------------------------------------ accelerator layers

static ACCEL_ID: AtomicU64 = AtomicU64::new(1);

/// One decoder layer resident in a [`Backend`]: its weights, uploaded once.
struct DevLayer {
    attn_norm: Buf,
    ffn_norm: Buf,
    wq: WeightHandle,
    wk: WeightHandle,
    wv: WeightHandle,
    wo: WeightHandle,
    w_up: WeightHandle,
    w_gate: WeightHandle,
    w_down: WeightHandle,
}
/// The backend, plus which layers run on it.
struct Accel {
    id: u64,
    be: Arc<dyn Backend>,
    layers: Vec<Option<DevLayer>>,
    kv_capacity: usize,
}
/// Per-state accelerator data: the layers' KV caches and the working buffers, all backend-side.
struct DevState {
    owner: u64,
    kv: Vec<Option<Kv>>,
    x: Buf,
    xb: Buf,
    xm: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    att: Buf,
    proj: Buf,
    u: Buf,
    g: Buf,
    mlp: Buf,
}

/// How [`Model::with_backend_opts`] splits the model between the host and a backend.
#[derive(Clone, Debug)]
pub struct AccelOptions {
    /// Offload at most this many leading layers (`None` = as many as fit and are supported).
    pub max_layers: Option<usize>,
    /// Positions the backend-side KV cache holds (`0` = `min(context, 4096)`). Longer conversations
    /// are refused with `ContextFull` rather than silently truncated.
    pub kv_capacity: usize,
    /// Bytes of device memory kept free for working buffers and the driver.
    pub reserve: u64,
}
impl Default for AccelOptions {
    fn default() -> Self {
        AccelOptions {
            max_layers: None,
            kv_capacity: 0,
            reserve: 256 << 20,
        }
    }
}

fn be_err(e: impl std::fmt::Display) -> ModelError {
    ModelError::Backend(e.to_string())
}

impl Model {
    /// Run the layers a backend can take through it (weights uploaded once, as many layers as fit
    /// its memory), keeping the rest — and the embedding and output head — on the host kernels.
    /// Without this call the model uses its built-in CPU path, which is the same arithmetic fused
    /// for speed; routing the CPU through [`crate::backend::CpuBackend`] works but is slower.
    pub fn with_backend(self, be: Arc<dyn Backend>) -> Result<Model, ModelError> {
        self.with_backend_opts(be, AccelOptions::default())
    }

    pub fn with_backend_opts(
        mut self,
        be: Arc<dyn Backend>,
        o: AccelOptions,
    ) -> Result<Model, ModelError> {
        let n = self.w.layers.len();
        let cap = if o.kv_capacity > 0 {
            o.kv_capacity
        } else {
            (self.spec.context as usize).min(4096)
        };
        let kd = (self.spec.kv_heads * self.spec.head_dim) as usize;
        let eligible: Vec<bool> = (0..n)
            .map(|i| self.layer_offloadable(i, be.as_ref()))
            .collect();
        let per_layer: Vec<u64> = self
            .w
            .layers
            .iter()
            .zip(&eligible)
            .map(|(l, &e)| {
                if e {
                    [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]
                        .iter()
                        .map(|w| w.stored_bytes() as u64)
                        .sum()
                } else {
                    0
                }
            })
            .collect();
        let by_memory = be.device_memory().map_or(n, |v| {
            crate::backend::offload_plan(&per_layer, (cap * kd * 2 * 4) as u64, 0, v, o.reserve)
                .gpu_layers
        });
        let limit = by_memory.min(o.max_layers.unwrap_or(n));
        let (h, qd, ff) = (
            self.spec.hidden as usize,
            (self.spec.heads * self.spec.head_dim) as usize,
            self.spec.ffn as usize,
        );
        let up = |w: &Weight, rows: usize, cols: usize| -> Result<WeightHandle, ModelError> {
            match w {
                Weight::Dense(v) => be.upload(WeightRef::dense(rows, cols, v)),
                Weight::Quant(q) => be.upload(WeightRef::quant(
                    q.dtype,
                    q.rows,
                    q.cols,
                    q.bytes.as_slice(),
                )),
            }
            .map_err(be_err)
        };
        let mut layers = Vec::with_capacity(n);
        for (i, l) in self.w.layers.iter().enumerate() {
            if i >= limit || !eligible[i] {
                layers.push(None);
                continue;
            }
            layers.push(Some(DevLayer {
                attn_norm: be.alloc_from(&l.attn_norm).map_err(be_err)?,
                ffn_norm: be.alloc_from(&l.ffn_norm).map_err(be_err)?,
                wq: up(&l.wq, qd, h)?,
                wk: up(&l.wk, kd, h)?,
                wv: up(&l.wv, kd, h)?,
                wo: up(&l.wo, h, qd)?,
                w_up: up(&l.w_up, ff, h)?,
                w_gate: up(&l.w_gate, ff, h)?,
                w_down: up(&l.w_down, h, ff)?,
            }));
        }
        self.accel = Some(Accel {
            id: ACCEL_ID.fetch_add(1, Ordering::Relaxed),
            be,
            layers,
            kv_capacity: cap,
        });
        Ok(self)
    }

    /// Which layers run on the backend (empty without one).
    pub fn offloaded(&self) -> Vec<bool> {
        self.accel
            .as_ref()
            .map_or(vec![], |a| a.layers.iter().map(Option::is_some).collect())
    }

    /// Can layer `i` run through the [`Backend`] trait unchanged? Variants the trait does not cover
    /// (biases, QK-norm, soft-capping, GELU, MoE, parallel residuals, scaled or partial RoPE, Gemma
    /// norms) keep their layer on the host path.
    fn layer_offloadable(&self, i: usize, be: &dyn Backend) -> bool {
        let (s, l) = (&self.spec, &self.w.layers[i]);
        let rope_ok =
            (s.is_sliding(i) && s.rope_theta_local > 0.0) || s.rope_scaling == RopeScaling::None;
        let supported = |w: &Weight| match w {
            Weight::Dense(_) => be.supports(DType::F32),
            Weight::Quant(q) => be.supports(q.dtype),
        };
        s.norm == NormKind::Rms
            && !s.norm_offset
            && s.act == Act::Silu
            && s.gated_mlp
            && !s.qk_norm
            && !s.qkv_bias
            && !s.post_norms
            && !s.parallel_residual
            && s.attn_softcap == 0.0
            && s.rotary_dim == s.head_dim
            && rope_ok
            && l.moe.is_none()
            && l.attn_norm_b.is_none()
            && l.ffn_norm_b.is_none()
            && [
                &l.bq,
                &l.bk,
                &l.bv,
                &l.bo,
                &l.q_norm,
                &l.k_norm,
                &l.post_attn_norm,
                &l.post_ffn_norm,
                &l.b_gate,
                &l.b_up,
                &l.b_down,
            ]
            .iter()
            .all(|o| o.is_none())
            && [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]
                .iter()
                .all(|w| supported(w))
    }

    fn new_dev_state(&self, a: &Accel) -> Result<DevState, ModelError> {
        let s = &self.spec;
        let (h, hd, kvh) = (s.hidden as usize, s.head_dim as usize, s.kv_heads as usize);
        let (qd, kd, ff) = ((s.heads * s.head_dim) as usize, kvh * hd, s.ffn as usize);
        let al = |n: usize| a.be.alloc(n).map_err(be_err);
        let kv = a
            .layers
            .iter()
            .map(|l| {
                l.as_ref()
                    .map(|_| a.be.kv_create(kvh, hd, a.kv_capacity).map_err(be_err))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DevState {
            owner: a.id,
            kv,
            x: al(h)?,
            xb: al(h)?,
            xm: al(h)?,
            q: al(qd)?,
            k: al(kd)?,
            v: al(kd)?,
            att: al(qd)?,
            proj: al(h)?,
            u: al(ff)?,
            g: al(ff)?,
            mlp: al(h)?,
        })
    }

    /// Run layers `range` (all accelerated) over `b` tokens, token by token: the layers' state is
    /// per position, so the result equals the layer-by-layer order used on the host.
    fn run_device(
        &self,
        a: &Accel,
        d: &mut DevState,
        range: std::ops::Range<usize>,
        b: usize,
        pos0: usize,
        x: &mut [f32],
    ) -> Result<(), ModelError> {
        let s = &self.spec;
        let (h, hd, nh, nkv) = (
            s.hidden as usize,
            s.head_dim as usize,
            s.heads as usize,
            s.kv_heads as usize,
        );
        let scale = if s.attn_scale > 0.0 {
            s.attn_scale as f32
        } else {
            1.0 / (hd as f32).sqrt()
        };
        let (be, eps) = (a.be.as_ref(), s.norm_eps as f32);
        for t in 0..b {
            let p = pos0 + t;
            be.write(&d.x, &x[t * h..(t + 1) * h]).map_err(be_err)?;
            for li in range.clone() {
                let (dl, kv) = match (&a.layers[li], &d.kv[li]) {
                    (Some(dl), Some(kv)) => (dl, kv),
                    _ => {
                        return Err(ModelError::Backend(format!(
                            "layer {li} is not on the backend"
                        )));
                    }
                };
                let sliding = s.is_sliding(li);
                let theta = if sliding && s.rope_theta_local > 0.0 {
                    s.rope_theta_local
                } else {
                    s.rope_theta
                };
                be.rmsnorm(&d.x, &dl.attn_norm, eps, &d.xb)
                    .map_err(be_err)?;
                be.matvec(&dl.wq, &d.xb, &d.q).map_err(be_err)?;
                be.matvec(&dl.wk, &d.xb, &d.k).map_err(be_err)?;
                be.matvec(&dl.wv, &d.xb, &d.v).map_err(be_err)?;
                be.rope(&d.q, nh, hd, p, theta).map_err(be_err)?;
                be.rope(&d.k, nkv, hd, p, theta).map_err(be_err)?;
                be.kv_append(kv, p, &d.k, &d.v).map_err(be_err)?;
                let window = sliding.then_some(s.sliding_window as usize);
                be.attention(
                    kv,
                    &d.q,
                    &d.att,
                    &AttnArgs {
                        n_heads: nh,
                        n_pos: p + 1,
                        scale,
                        window,
                    },
                )
                .map_err(be_err)?;
                be.matvec(&dl.wo, &d.att, &d.proj).map_err(be_err)?;
                be.add(&d.x, &d.proj).map_err(be_err)?;
                be.rmsnorm(&d.x, &dl.ffn_norm, eps, &d.xm).map_err(be_err)?;
                be.matvec(&dl.w_up, &d.xm, &d.u).map_err(be_err)?;
                be.matvec(&dl.w_gate, &d.xm, &d.g).map_err(be_err)?;
                be.silu_mul(&d.g, &d.u).map_err(be_err)?;
                be.matvec(&dl.w_down, &d.g, &d.mlp).map_err(be_err)?;
                be.add(&d.x, &d.mlp).map_err(be_err)?;
            }
            be.read(&d.x, &mut x[t * h..(t + 1) * h]).map_err(be_err)?;
        }
        Ok(())
    }
}

/// Token sampling. `temperature <= 0` is greedy. Optional top-k, top-p (nucleus) and a repetition
/// penalty over the last `repeat_last_n` tokens passed to [`Sampler::observe`].
pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    rng: u64,
    history: Vec<usize>,
}
impl Sampler {
    pub fn new(temperature: f32, top_k: usize, seed: u64) -> Sampler {
        Sampler {
            temperature,
            top_k,
            top_p: 1.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            rng: seed | 1,
            history: vec![],
        }
    }
    pub fn with_top_p(mut self, p: f32) -> Sampler {
        self.top_p = p;
        self
    }
    pub fn with_repetition_penalty(mut self, penalty: f32, last_n: usize) -> Sampler {
        self.repeat_penalty = penalty;
        self.repeat_last_n = last_n;
        self
    }
    pub fn observe(&mut self, token: usize) {
        self.history.push(token);
        if self.history.len() > 8192 {
            self.history.drain(..4096);
        }
    }
    pub fn observe_all(&mut self, tokens: &[usize]) {
        tokens.iter().for_each(|&t| self.observe(t));
    }
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
    fn next_f32(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn sample(&mut self, logits: &[f32]) -> usize {
        if logits.is_empty() {
            return 0;
        }
        let penalised;
        let logits =
            if self.repeat_penalty != 1.0 && self.repeat_last_n > 0 && !self.history.is_empty() {
                let mut seen: Vec<usize> =
                    self.history[self.history.len().saturating_sub(self.repeat_last_n)..].to_vec();
                seen.sort_unstable();
                seen.dedup();
                let mut l = logits.to_vec();
                let n = l.len();
                for t in seen.into_iter().filter(|&t| t < n) {
                    l[t] = if l[t] > 0.0 {
                        l[t] / self.repeat_penalty
                    } else {
                        l[t] * self.repeat_penalty
                    };
                }
                penalised = l;
                &penalised[..]
            } else {
                logits
            };
        let argmax = || {
            logits
                .iter()
                .enumerate()
                .fold(
                    (0, f32::NEG_INFINITY),
                    |b, (i, &v)| if v > b.1 { (i, v) } else { b },
                )
                .0
        };
        if self.temperature <= 0.0 {
            return argmax();
        }
        let inv_t = 1.0 / self.temperature;
        let mut cand: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, l * inv_t))
            .collect();
        let desc = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1);
        if self.top_k > 0 && self.top_k < cand.len() {
            cand.select_nth_unstable_by(self.top_k - 1, desc);
            cand.truncate(self.top_k);
        }
        if self.top_p < 1.0 {
            cand.sort_by(desc);
        }
        let m = cand.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = cand
            .iter()
            .map(|c| crate::simd::rmath::exp(c.1 - m))
            .collect();
        let mut total: f32 = probs.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return argmax();
        }
        if self.top_p < 1.0 {
            let (mut cum, mut keep) = (0.0, probs.len());
            for (i, p) in probs.iter().enumerate() {
                cum += p / total;
                if cum >= self.top_p {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
            cand.truncate(keep);
            total = probs.iter().sum();
        }
        let mut r = self.next_f32() * total;
        for (c, p) in cand.iter().zip(&probs) {
            if r < *p {
                return c.0;
            }
            r -= p;
        }
        cand[cand.len() - 1].0
    }
}

// ------------------------------------------------------------------ KV-cache surgery
// Everything in this block serves kvpolicy.rs (eviction) and snapshot.rs (save / restore).
// The rest of this file only reads `kvx.gap`, calls `acc_add` after attention, and keeps
// `kvx` in step when a state is rewound.

/// What eviction adds to a [`State`]. After the cache is compacted, slot `i` no longer holds the
/// token at RoPE position `i`: `abs[i]` does (empty = every slot sits at its own position), and the
/// next token is rotated at `pos + gap`, so its distance to every kept key is the original one.
#[derive(Clone, Default)]
pub(crate) struct KvExtra {
    abs: Vec<usize>,
    gap: usize,
    /// Per layer: attention probability each slot has received, summed over heads and queries
    /// (`None` = not tracked). Only layers that keep every position are tracked.
    acc: Option<Vec<Vec<f32>>>,
}
impl KvExtra {
    fn identity_prefix(&self, n: usize) -> bool {
        self.abs.is_empty() || self.abs.iter().take(n).enumerate().all(|(i, &a)| a == i)
    }
    fn after_truncate(&mut self, pos: usize) {
        // reachable only for an identity prefix (or an emptied state): positions are plain again
        self.abs.clear();
        self.gap = 0;
        if let Some(a) = &mut self.acc {
            for v in a.iter_mut() {
                v.truncate(pos);
            }
        }
    }
}

/// Add this query's attention probabilities (every head) to the per-slot accumulator.
#[allow(clippy::too_many_arguments)]
fn acc_add(
    kv: &LayerKv,
    acc: &mut Vec<f32>,
    kd: usize,
    hd: usize,
    nh: usize,
    group: usize,
    n: usize,
    scale: f32,
    cap: f64,
    q: &[f32],
) {
    if acc.len() < n {
        acc.resize(n, 0.0);
    }
    let mut sc = vec![0f32; n];
    for h in 0..nh {
        let (kvh, qh) = (h / group, &q[h * hd..(h + 1) * hd]);
        for (i, s) in sc.iter_mut().enumerate() {
            *s = softcap(kv.k.dot(i, kd, hd, kvh, qh) * scale, cap);
        }
        softmax(&mut sc);
        for (a, p) in acc.iter_mut().zip(&sc) {
            *a += p;
        }
    }
}

/// Raw cache contents, borrowed (for writing a snapshot).
pub(crate) enum PlaneRef<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
    Q8 { q: &'a [i8], s: &'a [f32] },
}
/// Raw cache contents, owned (read back from a snapshot).
pub(crate) enum PlaneData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Q8 { q: Vec<i8>, s: Vec<f32> },
}
pub(crate) struct LayerView<'a> {
    pub cap: usize,
    pub kd: usize,
    pub kvh: usize,
    pub k: PlaneRef<'a>,
    pub v: PlaneRef<'a>,
}

impl Plane {
    fn view(&self) -> PlaneRef<'_> {
        match self {
            Plane::F32(v) => PlaneRef::F32(v),
            Plane::F16(v) => PlaneRef::F16(v),
            Plane::Q8 { q, s } => PlaneRef::Q8 { q, s },
        }
    }
    fn same_kind(&self, d: &PlaneData) -> bool {
        matches!(
            (self, d),
            (Plane::F32(_), PlaneData::F32(_))
                | (Plane::F16(_), PlaneData::F16(_))
                | (Plane::Q8 { .. }, PlaneData::Q8 { .. })
        )
    }
    fn from_data(d: PlaneData) -> Plane {
        match d {
            PlaneData::F32(v) => Plane::F32(v),
            PlaneData::F16(v) => Plane::F16(v),
            PlaneData::Q8 { q, s } => Plane::Q8 { q, s },
        }
    }
    /// Whole slots held.
    fn slots(&self, kd: usize, kvh: usize) -> usize {
        match self {
            Plane::F32(v) => v.len() / kd.max(1),
            Plane::F16(v) => v.len() / kd.max(1),
            Plane::Q8 { q, s } => (q.len() / kd.max(1)).min(s.len() / kvh.max(1)),
        }
    }
    /// New plane whose slot `d` is a bit-exact copy of slot `src[d]` (quantised values are copied, never re-encoded).
    fn gather(&self, src: &[usize], kd: usize, kvh: usize) -> Plane {
        match self {
            Plane::F32(v) => Plane::F32(
                src.iter()
                    .flat_map(|&s| v[s * kd..(s + 1) * kd].iter().copied())
                    .collect(),
            ),
            Plane::F16(v) => Plane::F16(
                src.iter()
                    .flat_map(|&s| v[s * kd..(s + 1) * kd].iter().copied())
                    .collect(),
            ),
            Plane::Q8 { q, s } => Plane::Q8 {
                q: src
                    .iter()
                    .flat_map(|&i| q[i * kd..(i + 1) * kd].iter().copied())
                    .collect(),
                s: src
                    .iter()
                    .flat_map(|&i| s[i * kvh..(i + 1) * kvh].iter().copied())
                    .collect(),
            },
        }
    }
}

impl State {
    /// Precision of the cache.
    pub fn kv_kind(&self) -> KvKind {
        match self.kv.first().map(|l| &l.k) {
            Some(Plane::F16(_)) => KvKind::F16,
            Some(Plane::Q8 { .. }) => KvKind::Q8,
            _ => KvKind::F32,
        }
    }
    /// Number of layers the cache has.
    pub fn layer_count(&self) -> usize {
        self.kv.len()
    }
    /// Ring capacity of each layer (`0` = the layer keeps every position).
    pub fn ring_caps(&self) -> Vec<usize> {
        self.kv.iter().map(|l| l.cap).collect()
    }
    /// How far the RoPE position of the next token is ahead of `pos` (non-zero after eviction).
    pub fn rope_gap(&self) -> usize {
        self.kvx.gap
    }
    /// RoPE position the next token will get.
    pub fn rope_pos(&self) -> usize {
        self.pos + self.kvx.gap
    }
    /// RoPE position of the token held in slot `i`.
    pub fn abs_pos(&self, i: usize) -> usize {
        self.kvx.abs.get(i).copied().unwrap_or(i)
    }
    /// Has eviction run (slots no longer sit at their own RoPE position)?
    pub fn is_compacted(&self) -> bool {
        !self.kvx.abs.is_empty() || self.kvx.gap != 0
    }
    /// Start or stop accumulating the attention each slot receives (the signal H2O and SnapKV need).
    /// Off by default; when off the decoder does no extra work. Turning it on starts from zero.
    pub fn track_scores(&mut self, on: bool) {
        self.kvx.acc = on.then(|| vec![vec![]; self.kv.len()]);
    }
    pub fn tracks_scores(&self) -> bool {
        self.kvx.acc.is_some()
    }
    /// Forget the accumulated scores but keep tracking (SnapKV: count only the last window of queries).
    pub fn clear_scores(&mut self) {
        if let Some(a) = &mut self.kvx.acc {
            a.iter_mut().for_each(Vec::clear);
        }
    }
    /// Accumulated attention per slot for `layer`; `None` when not tracked or the layer is a ring.
    pub fn layer_scores(&self, layer: usize) -> Option<&[f32]> {
        if self.kv.get(layer)?.cap != 0 {
            return None;
        }
        self.kvx.acc.as_ref()?.get(layer).map(Vec::as_slice)
    }
    fn rebase_to_prefix(&mut self) {
        if self.kvx.identity_prefix(self.pos) {
            self.kvx.abs.clear();
            self.kvx.gap = 0;
        }
    }
    /// Leading tokens of `tokens` whose cached slots are valid for them: equal tokens at their own position.
    fn reusable_prefix(&self, tokens: &[usize]) -> usize {
        let id = self.kvx.abs.is_empty();
        self.tokens
            .iter()
            .zip(tokens)
            .enumerate()
            .take_while(|&(i, (a, b))| a == b && (id || self.kvx.abs.get(i) == Some(&i)))
            .count()
    }
    fn held_slots(&self, l: &LayerKv) -> usize {
        if l.cap > 0 {
            self.pos.min(l.cap)
        } else {
            self.pos
        }
    }

    /// Keep only the slots in `keep` (strictly increasing, all `< pos`), in every precision and in
    /// sliding-window rings. Keys stay as stored (already rotated for their original position); the
    /// next token is rotated at the original next position (`rope_gap`), so query–key distances are
    /// exactly what they were. Keeping everything changes nothing. A sliding-window layer can only
    /// keep what its ring still holds, and its last `ring` slots must be the last positions.
    pub fn compact(&mut self, keep: &[usize]) -> Result<(), ModelError> {
        let bad = |m: String| ModelError::Shape(format!("cannot compact the cache: {m}"));
        if self.dev.is_some() {
            return Err(ModelError::Backend(
                "a state with an accelerator-side cache cannot be compacted".into(),
            ));
        }
        let (n, m) = (self.pos, keep.len());
        if m == 0 || m > n {
            return Err(bad(format!("{m} slots to keep out of {n}")));
        }
        if keep.windows(2).any(|w| w[0] >= w[1]) || keep[m - 1] >= n {
            return Err(bad(
                "the slots must be increasing and inside the cache".into()
            ));
        }
        if self.tokens.len() != n {
            return Err(bad("the token history is out of step with the cache".into()));
        }
        if m == n {
            return Ok(());
        }
        for l in &self.kv {
            let (kd, kvh) = (l.kd, l.kvh);
            if l.k.slots(kd, kvh) < self.held_slots(l) || l.v.slots(kd, kvh) < self.held_slots(l) {
                return Err(bad(
                    "a layer holds fewer slots than the position counter".into()
                ));
            }
            if l.cap > 0
                && keep[m.saturating_sub(l.cap)..]
                    .iter()
                    .any(|&p| p < n.saturating_sub(l.cap))
            {
                return Err(bad(format!(
                    "a sliding-window layer of {} slots no longer holds the position asked for",
                    l.cap
                )));
            }
        }
        let next_abs = n + self.kvx.gap;
        let new_abs: Vec<usize> = keep.iter().map(|&i| self.abs_pos(i)).collect();
        for l in &mut self.kv {
            let (kd, kvh, cap) = (l.kd, l.kvh, l.cap);
            let src: Vec<usize> = if cap == 0 {
                keep.to_vec()
            } else {
                // rebuild the ring so that slot `j % cap` holds the j-th kept token
                let start = m.saturating_sub(cap);
                (0..m.min(cap))
                    .map(|d| {
                        keep[if m <= cap {
                            d
                        } else {
                            start + (d + cap - start % cap) % cap
                        }] % cap
                    })
                    .collect()
            };
            l.k = l.k.gather(&src, kd, kvh);
            l.v = l.v.gather(&src, kd, kvh);
        }
        if let Some(acc) = &mut self.kvx.acc {
            for (a, l) in acc.iter_mut().zip(&self.kv) {
                if l.cap == 0 {
                    *a = keep
                        .iter()
                        .map(|&i| a.get(i).copied().unwrap_or(0.0))
                        .collect();
                } else {
                    a.clear();
                }
            }
        }
        self.tokens = keep.iter().map(|&i| self.tokens[i]).collect();
        self.kvx.gap = next_abs - m;
        self.kvx.abs = if new_abs.iter().enumerate().all(|(i, &a)| a == i) {
            vec![]
        } else {
            new_abs
        };
        self.pos = m;
        self.last = None;
        self.reused = 0;
        Ok(())
    }

    pub(crate) fn kv_views(&self) -> Vec<LayerView<'_>> {
        self.kv
            .iter()
            .map(|l| LayerView {
                cap: l.cap,
                kd: l.kd,
                kvh: l.kvh,
                k: l.k.view(),
                v: l.v.view(),
            })
            .collect()
    }
    /// The eviction bookkeeping: RoPE position of each slot (empty = identity) and the gap.
    pub(crate) fn kvx_parts(&self) -> (&[usize], usize) {
        (&self.kvx.abs, self.kvx.gap)
    }
    /// Fill a freshly made state (`Model::new_state_with`) from snapshot contents, checking every size.
    pub(crate) fn restore_raw(
        &mut self,
        pos: usize,
        tokens: Vec<usize>,
        abs: Vec<usize>,
        gap: usize,
        planes: Vec<(PlaneData, PlaneData)>,
    ) -> Result<(), ModelError> {
        let bad = |m: &str| ModelError::Shape(format!("snapshot: {m}"));
        if self.dev.is_some() || self.pos != 0 || !self.tokens.is_empty() {
            return Err(bad("it can only be restored into a fresh state"));
        }
        if planes.len() != self.kv.len() {
            return Err(bad("the layer count differs"));
        }
        if tokens.len() != pos {
            return Err(bad("the token history does not match the position"));
        }
        if !abs.is_empty() && (abs.len() != pos || abs.windows(2).any(|w| w[0] >= w[1])) {
            return Err(bad("the position table is malformed"));
        }
        if abs.last().is_some_and(|&a| a >= pos.saturating_add(gap)) {
            return Err(bad("the position gap is inconsistent"));
        }
        for (l, (k, v)) in self.kv.iter().zip(&planes) {
            let slots = if l.cap > 0 { pos.min(l.cap) } else { pos };
            if !l.k.same_kind(k) || !l.v.same_kind(v) {
                return Err(bad("the cache precision differs"));
            }
            let len = |d: &PlaneData| match d {
                PlaneData::F32(x) => (x.len(), x.len()),
                PlaneData::F16(x) => (x.len(), x.len()),
                PlaneData::Q8 { q, s } => (q.len(), s.len() * l.kd / l.kvh.max(1)),
            };
            for d in [k, v] {
                let (a, b) = len(d);
                if a != slots * l.kd || b != slots * l.kd {
                    return Err(bad("a layer holds the wrong number of values"));
                }
            }
        }
        for (l, (k, v)) in self.kv.iter_mut().zip(planes) {
            l.k = Plane::from_data(k);
            l.v = Plane::from_data(v);
        }
        self.pos = pos;
        self.tokens = tokens;
        self.kvx.abs = abs;
        self.kvx.gap = gap;
        self.last = None;
        Ok(())
    }
}

impl Model {
    /// Identifies the model a cache or snapshot belongs to: SHA-256 over the architecture and the
    /// size, storage type and (for experts) shape of every tensor. Different weights of identical
    /// shape are not told apart; that would mean reading the whole model.
    pub fn fingerprint(&self) -> [u8; 32] {
        let mut h = crate::hub::Sha256::new();
        let mut put = |b: &[u8]| {
            h.update(&(b.len() as u64).to_le_bytes());
            h.update(b);
        };
        put(b"colibri-engine-model-v1");
        put(format!("{:?}", self.spec).as_bytes());
        let w = |wt: &Weight| -> Vec<u8> {
            let tag = match wt {
                Weight::Dense(_) => format!("dense:{}", wt.len()),
                Weight::Quant(q) => format!("{:?}:{}x{}", q.dtype, q.rows, q.cols),
            };
            format!("{tag}:{}", wt.stored_bytes()).into_bytes()
        };
        put(&w(&self.w.embed));
        if let Some(l) = &self.w.lm_head {
            put(&w(l));
        }
        put(&(self.w.final_norm.len() as u64).to_le_bytes());
        for (i, l) in self.w.layers.iter().enumerate() {
            put(&(i as u64).to_le_bytes());
            for t in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
                put(&w(t));
            }
            if let Some(m) = &l.moe {
                let c = m.router.config();
                put(format!(
                    "moe:{}:{}:{}:{}",
                    c.experts,
                    c.top_k,
                    m.shared.len(),
                    m.stored_bytes()
                )
                .as_bytes());
            }
        }
        h.finish()
    }
}

// ===================================================================== BEGIN vision hook
// Feeding embedding rows (image or audio features) into the prompt at chosen positions. Additive:
// no existing path calls into this block, so ordinary generation is unchanged in behaviour and speed.

/// What `State::tokens` records at a position that was fed from an embedding row instead of a token.
/// It never equals a real token id, so prefix reuse (`prefill_cached`) stops in front of such a
/// position, and replaying a recorded sequence through `forward_batch` fails loudly (`Token`).
pub const EMBEDDING_POSITION: usize = usize::MAX;

/// One piece of a mixed prompt for [`Model::forward_segments`].
#[derive(Clone, Copy, Debug)]
pub enum Segment<'a> {
    /// Ordinary token ids, embedded through the model's own table.
    Tokens(&'a [usize]),
    /// Rows of `hidden` floats placed directly in the residual stream: no table lookup and no
    /// `embed_scale`, because they are already in the model's hidden space (a vision projector's output).
    Rows(&'a [f32]),
}

impl Model {
    /// Like [`Model::forward_batch`] for a prompt that mixes tokens and embedding rows. Equal, bit for
    /// bit, to `forward_batch` when every segment is `Tokens`. Returns the logits after the last
    /// position. Rows must be a non-empty multiple of `hidden` floats, all finite. After an `Err` the
    /// state may be partly advanced; discard it. Positions fed from rows are recorded in
    /// [`State::tokens`] as [`EMBEDDING_POSITION`].
    pub fn forward_segments(
        &self,
        st: &mut State,
        segs: &[Segment<'_>],
    ) -> Result<Vec<f32>, ModelError> {
        enum Slot<'a> {
            Tok(usize),
            Row(&'a [f32]),
        }
        let s = &self.spec;
        let h = s.hidden as usize;
        let mut total = 0usize;
        for seg in segs {
            match seg {
                Segment::Tokens(t) => {
                    if let Some(&bad) = t.iter().find(|&&x| x >= s.vocab as usize) {
                        return Err(ModelError::Token(bad));
                    }
                    total += t.len();
                }
                Segment::Rows(r) => {
                    if r.is_empty() || !r.len().is_multiple_of(h) {
                        return Err(ModelError::Shape(format!(
                            "{} embedding values is not a whole number of {h}-wide rows",
                            r.len()
                        )));
                    }
                    if r.iter().any(|v| !v.is_finite()) {
                        return Err(ModelError::Shape(
                            "an embedding row holds a NaN or infinity".into(),
                        ));
                    }
                    total += r.len() / h;
                }
            }
        }
        if total == 0 {
            return Err(ModelError::NoTokens);
        }
        self.ready(st, total)?;
        let mut slots: Vec<Slot> = Vec::with_capacity(total);
        for seg in segs {
            match seg {
                Segment::Tokens(t) => slots.extend(t.iter().map(|&x| Slot::Tok(x))),
                Segment::Rows(r) => slots.extend(r.chunks_exact(h).map(Slot::Row)),
            }
        }
        st.last = None;
        let chunks = slots.len().div_ceil(PREFILL_CHUNK);
        let mut last = vec![];
        for (ci, chunk) in slots.chunks(PREFILL_CHUNK).enumerate() {
            let (b, pos0) = (chunk.len(), st.pos);
            let mut x = std::mem::take(&mut st.sc.x);
            fit(&mut x, b * h);
            let mut r = Ok(());
            for (i, slot) in chunk.iter().enumerate() {
                let row = &mut x[i * h..(i + 1) * h];
                match slot {
                    Slot::Row(v) => row.copy_from_slice(v),
                    Slot::Tok(t) => {
                        if let Err(e) = self.embed_rows(&[*t], row, &mut st.sc.rowtmp) {
                            r = Err(e);
                            break;
                        }
                    }
                }
            }
            if r.is_ok() {
                r = self.layers_core(st, 0..self.w.layers.len(), &mut x, b, pos0);
            }
            if let Err(e) = r {
                st.sc.x = x;
                return Err(e);
            }
            st.pos += b;
            st.tokens.extend(chunk.iter().map(|sl| match sl {
                Slot::Tok(t) => *t,
                Slot::Row(_) => EMBEDDING_POSITION,
            }));
            if ci + 1 == chunks {
                let out = self.logits_from_hidden_inner(&x[(b - 1) * h..b * h]);
                st.sc.x = x;
                last = out?;
            } else {
                st.sc.x = x;
            }
        }
        Ok(last)
    }
}
// ======================================================================= END vision hook

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn rng(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6
    }
    pub(crate) fn mat(n: usize, s: &mut u64) -> Vec<f32> {
        (0..n).map(|_| rng(s)).collect()
    }
    fn ones(n: usize, s: &mut u64) -> Vec<f32> {
        (0..n).map(|_| 1.0 + rng(s) * 0.2).collect()
    }
    fn some(on: bool, n: usize, s: &mut u64) -> Option<Vec<f32>> {
        on.then(|| mat(n, s))
    }

    pub fn base_spec(heads: u64, kv: u64, tied: bool) -> Spec {
        let mut sp = Spec::dense("llama", 3, 16, heads, kv, 8, 24, 20, 32);
        sp.tied_embeddings = tied;
        sp
    }

    /// Random weights for exactly the tensors `spec` asks for (plus optional biases when `extras`).
    pub fn random_weights(spec: &Spec, seed: u64, extras: bool) -> Weights {
        let mut s = seed;
        let (h, qd, kd, ff, v) = (
            spec.hidden as usize,
            (spec.heads * spec.head_dim) as usize,
            (spec.kv_heads * spec.head_dim) as usize,
            spec.ffn as usize,
            spec.vocab as usize,
        );
        let hd = spec.head_dim as usize;
        let ln = spec.norm == NormKind::Layer;
        let bias = spec.qkv_bias || extras;
        let layers = (0..spec.layers)
            .map(|_| Layer {
                attn_norm: ones(h, &mut s),
                attn_norm_b: some(ln, h, &mut s),
                wq: mat(qd * h, &mut s).into(),
                wk: mat(kd * h, &mut s).into(),
                wv: mat(kd * h, &mut s).into(),
                wo: mat(h * qd, &mut s).into(),
                bq: some(bias, qd, &mut s),
                bk: some(bias, kd, &mut s),
                bv: some(bias, kd, &mut s),
                bo: some(extras, h, &mut s),
                q_norm: some(spec.qk_norm, hd, &mut s).map(|v| v.iter().map(|x| x + 1.0).collect()),
                k_norm: some(spec.qk_norm, hd, &mut s).map(|v| v.iter().map(|x| x + 1.0).collect()),
                post_attn_norm: some(spec.post_norms, h, &mut s),
                ffn_norm: ones(h, &mut s),
                ffn_norm_b: some(ln, h, &mut s),
                w_gate: if spec.gated_mlp {
                    mat(ff * h, &mut s)
                } else {
                    vec![]
                }
                .into(),
                b_gate: some(extras && spec.gated_mlp, ff, &mut s),
                w_up: mat(ff * h, &mut s).into(),
                b_up: some(extras, ff, &mut s),
                w_down: mat(h * ff, &mut s).into(),
                b_down: some(extras, h, &mut s),
                post_ffn_norm: some(spec.post_norms, h, &mut s),
                ..Default::default()
            })
            .collect();
        Weights {
            embed: mat(v * h, &mut s).into(),
            layers,
            final_norm: ones(h, &mut s),
            final_norm_b: some(ln, h, &mut s),
            lm_head: if spec.tied_embeddings {
                None
            } else {
                Some(mat(v * h, &mut s).into())
            },
            lm_head_b: some(extras && !spec.tied_embeddings, v, &mut s),
            experts: None,
        }
    }
    pub fn model_of(spec: Spec, extras: bool) -> Model {
        let w = random_weights(&spec, 0x9e37_79b9_7f4a_7c15, extras);
        Model::new(spec, w).unwrap()
    }
    pub fn tiny(heads: u64, kv: u64, tied: bool) -> Model {
        model_of(base_spec(heads, kv, tied), false)
    }

    /// Independent reference: recompute every position from scratch, no cache, straightforward loops.
    pub(crate) fn reference_logits(m: &Model, tokens: &[usize]) -> Vec<f32> {
        let s = &m.spec;
        let (h, hd, nh, nkv, ff) = (
            s.hidden as usize,
            s.head_dim as usize,
            s.heads as usize,
            s.kv_heads as usize,
            s.ffn as usize,
        );
        let sc = if s.attn_scale > 0.0 {
            s.attn_scale as f32
        } else {
            1.0 / (hd as f32).sqrt()
        };
        let mut xs: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&t| {
                m.w.embed.as_dense().unwrap()[t * h..(t + 1) * h]
                    .iter()
                    .map(|v| v * s.embed_scale as f32)
                    .collect()
            })
            .collect();
        let mv = |w: &[f32], cols: usize, x: &[f32], b: &Option<Vec<f32>>| -> Vec<f32> {
            let mut o: Vec<f32> = w
                .chunks(cols)
                .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect();
            if let Some(b) = b {
                o.iter_mut().zip(b).for_each(|(a, b)| *a += b);
            }
            o
        };
        let nm = |x: &[f32], w: &[f32], b: Option<&[f32]>| {
            let mut o = vec![0.0; x.len()];
            norm(s, x, w, b, &mut o);
            o
        };
        for (li, l) in m.w.layers.iter().enumerate() {
            let theta_local = s.is_sliding(li) && s.rope_theta_local > 0.0;
            let tab = if theta_local {
                rope_inv_freq(s.rope_theta_local, s.rotary_dim as usize, RopeScaling::None)
            } else {
                s.rope_freqs()
            };
            let (mut ks, mut vs, mut qs) = (vec![], vec![], vec![]);
            for (p, x) in xs.iter().enumerate() {
                let xb = nm(x, &l.attn_norm, l.attn_norm_b.as_deref());
                let (mut q, mut k, v) = (
                    mv(l.wq.as_dense().unwrap(), h, &xb, &l.bq),
                    mv(l.wk.as_dense().unwrap(), h, &xb, &l.bk),
                    mv(l.wv.as_dense().unwrap(), h, &xb, &l.bv),
                );
                if let (Some(qn), Some(kn)) = (&l.q_norm, &l.k_norm) {
                    for c in q.chunks_mut(hd) {
                        let r = nm(c, qn, None);
                        c.copy_from_slice(&r);
                    }
                    for c in k.chunks_mut(hd) {
                        let r = nm(c, kn, None);
                        c.copy_from_slice(&r);
                    }
                }
                for c in q.iter_mut().collect::<Vec<_>>().chunks_mut(hd) {
                    rope_naive(c, p, &tab);
                }
                for c in k.iter_mut().collect::<Vec<_>>().chunks_mut(hd) {
                    rope_naive(c, p, &tab);
                }
                qs.push(q);
                ks.push(k);
                vs.push(v);
            }
            let mut new_xs = xs.clone();
            for p in 0..xs.len() {
                let mut att = vec![0.0; nh * hd];
                for head in 0..nh {
                    let kvh = head / (nh / nkv);
                    let lo = if s.is_sliding(li) {
                        (p + 1).saturating_sub(s.sliding_window as usize)
                    } else {
                        0
                    };
                    let mut sco: Vec<f32> = (lo..=p)
                        .map(|t| {
                            let d: f32 = (0..hd)
                                .map(|d| qs[p][head * hd + d] * ks[t][kvh * hd + d])
                                .sum::<f32>()
                                * sc;
                            if s.attn_softcap > 0.0 {
                                s.attn_softcap as f32
                                    * crate::simd::rmath::tanh(d / s.attn_softcap as f32)
                            } else {
                                d
                            }
                        })
                        .collect();
                    softmax(&mut sco);
                    for d in 0..hd {
                        att[head * hd + d] =
                            (lo..=p).map(|t| sco[t - lo] * vs[t][kvh * hd + d]).sum();
                    }
                }
                let mut o = mv(l.wo.as_dense().unwrap(), nh * hd, &att, &l.bo);
                if let Some(pn) = &l.post_attn_norm {
                    o = nm(&o, pn, None);
                }
                let resid_after_attn: Vec<f32> = xs[p].iter().zip(&o).map(|(a, b)| a + b).collect();
                let mlp_in = nm(
                    if s.parallel_residual {
                        &xs[p]
                    } else {
                        &resid_after_attn
                    },
                    &l.ffn_norm,
                    l.ffn_norm_b.as_deref(),
                );
                let mut d = if l.moe.is_some() {
                    naive_moe(m, li, &mlp_in)
                } else {
                    let up = mv(l.w_up.as_dense().unwrap(), h, &mlp_in, &l.b_up);
                    let act: Vec<f32> = if s.gated_mlp {
                        let gt = mv(l.w_gate.as_dense().unwrap(), h, &mlp_in, &l.b_gate);
                        (0..ff).map(|i| activate(s.act, gt[i]) * up[i]).collect()
                    } else {
                        up.iter().map(|&u| activate(s.act, u)).collect()
                    };
                    mv(l.w_down.as_dense().unwrap(), ff, &act, &l.b_down)
                };
                if let Some(pn) = &l.post_ffn_norm {
                    d = nm(&d, pn, None);
                }
                new_xs[p] = if s.parallel_residual {
                    (0..h).map(|i| xs[p][i] + o[i] + d[i]).collect()
                } else {
                    (0..h).map(|i| resid_after_attn[i] + d[i]).collect()
                };
            }
            xs = new_xs;
        }
        let xb = nm(
            xs.last().unwrap(),
            &m.w.final_norm,
            m.w.final_norm_b.as_deref(),
        );
        let head =
            m.w.lm_head
                .as_ref()
                .unwrap_or(&m.w.embed)
                .as_dense()
                .unwrap();
        let mut out = mv(head, h, &xb, &m.w.lm_head_b);
        if s.final_softcap > 0.0 {
            out.iter_mut().for_each(|l| {
                *l = s.final_softcap as f32 * crate::simd::rmath::tanh(*l / s.final_softcap as f32)
            });
        }
        out
    }
    /// Independent mixture-of-experts feed-forward: explicit softmax, sort-based top-k and plain loops,
    /// sharing nothing with `moe.rs` except the weights it is given.
    pub(crate) fn naive_moe(m: &Model, li: usize, x: &[f32]) -> Vec<f32> {
        let s = &m.spec;
        let (moe, h) = (m.w.layers[li].moe.as_ref().unwrap(), x.len());
        let (rc, rw) = (moe.router.config(), moe.router.weights());
        let logits: Vec<f32> = (0..rc.experts)
            .map(|e| (0..h).map(|i| rw[e * h + i] * x[i]).sum())
            .collect();
        let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let ex: Vec<f32> = logits
            .iter()
            .map(|l| crate::simd::rmath::exp(l - mx))
            .collect();
        let tot: f32 = ex.iter().sum();
        let probs: Vec<f32> = ex.iter().map(|v| v / tot).collect();
        let mut order: Vec<usize> = (0..rc.experts).collect();
        order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
        let chosen = &order[..rc.top_k];
        let norm: f32 = if rc.renormalise {
            chosen.iter().map(|&e| probs[e]).sum()
        } else {
            1.0
        };
        let ffn = |w: &ExpertWeightsView, x: &[f32]| -> Vec<f32> {
            let f = w.0.len() / h;
            let a: Vec<f32> = (0..f)
                .map(|j| {
                    activate(s.act, (0..h).map(|i| w.0[j * h + i] * x[i]).sum::<f32>())
                        * (0..h).map(|i| w.1[j * h + i] * x[i]).sum::<f32>()
                })
                .collect();
            (0..h)
                .map(|i| (0..f).map(|j| w.2[i * f + j] * a[j]).sum())
                .collect()
        };
        let mut out = vec![0.0f32; h];
        let provider = &m.w.experts.as_ref().unwrap().provider;
        for &e in chosen {
            let w = provider.expert(li, e).unwrap();
            let d = ffn(
                &ExpertWeightsView(w.gate.to_f32(), w.up.to_f32(), w.down.to_f32()),
                x,
            );
            let wt = probs[e] / norm * rc.routed_scale;
            out.iter_mut().zip(&d).for_each(|(o, v)| *o += wt * v);
        }
        for sh in &moe.shared {
            let d = ffn(
                &ExpertWeightsView(sh.gate.to_f32(), sh.up.to_f32(), sh.down.to_f32()),
                x,
            );
            let g = match &moe.shared_gate {
                Some(g) => {
                    1.0 / (1.0 + crate::simd::rmath::exp(-(0..h).map(|i| g[i] * x[i]).sum::<f32>()))
                }
                None => 1.0,
            };
            out.iter_mut().zip(&d).for_each(|(o, v)| *o += g * v);
        }
        out
    }
    pub(crate) struct ExpertWeightsView(pub Vec<f32>, pub Vec<f32>, pub Vec<f32>);
    fn rope_naive(c: &mut [&mut f32], pos: usize, (inv, ms): &(Vec<f64>, f64)) {
        let half = inv.len();
        for i in 0..half {
            let a = pos as f64 * inv[i];
            let (x, y) = (*c[i] as f64, *c[i + half] as f64);
            *c[i] = ((x * a.cos() - y * a.sin()) * ms) as f32;
            *c[i + half] = ((x * a.sin() + y * a.cos()) * ms) as f32;
        }
    }

    const TOKS: [usize; 9] = [3, 7, 1, 19, 0, 7, 12, 5, 9];
    fn decode(m: &Model) -> Vec<Vec<f32>> {
        let mut st = m.new_state();
        TOKS.iter()
            .map(|&t| m.forward(&mut st, t).unwrap())
            .collect()
    }
    /// The cached path must equal the reference at every step, and differ from `baseline`.
    fn check(name: &str, m: &Model, baseline: Option<&Model>) {
        let got = decode(m);
        for (i, g) in got.iter().enumerate() {
            let want = reference_logits(m, &TOKS[..=i]);
            for (a, b) in g.iter().zip(&want) {
                assert!((a - b).abs() < 2e-4, "{name} step {i}: {a} vs {b}");
            }
        }
        if let Some(b) = baseline {
            let other = decode(b);
            let d = got
                .iter()
                .zip(&other)
                .flat_map(|(x, y)| x.iter().zip(y).map(|(a, b)| (a - b).abs()))
                .fold(0.0, f32::max);
            assert!(d > 2e-4, "{name}: the feature changed nothing ({d})");
        }
    }
    /// Same weights, feature switched off in the spec.
    fn without(m: &Model, f: impl Fn(&mut Spec)) -> Model {
        let mut sp = m.spec.clone();
        f(&mut sp);
        Model::new(sp, m.w.clone()).unwrap()
    }

    #[test]
    fn cached_decoding_equals_full_recomputation_for_mha_gqa_and_mqa() {
        for (heads, kv, tied) in [(4, 4, false), (4, 2, true), (4, 1, false)] {
            check("plain", &tiny(heads, kv, tied), None);
        }
    }

    #[test]
    fn biases_everywhere() {
        let mut sp = base_spec(4, 2, false);
        sp.qkv_bias = true;
        let m = model_of(sp, true);
        let mut base = m.w.clone();
        for l in &mut base.layers {
            l.bq = None;
            l.bk = None;
            l.bv = None;
            l.bo = None;
            l.b_gate = None;
            l.b_up = None;
            l.b_down = None;
        }
        base.lm_head_b = None;
        let mut bs = m.spec.clone();
        bs.qkv_bias = false;
        check("biases", &m, Some(&Model::new(bs, base).unwrap()));
    }

    #[test]
    fn qwen2_qkv_bias_only() {
        let mut sp = base_spec(4, 2, true);
        sp.qkv_bias = true;
        let m = model_of(sp, false);
        let mut w = m.w.clone();
        for l in &mut w.layers {
            l.bq = None;
            l.bk = None;
            l.bv = None;
        }
        let mut bs = m.spec.clone();
        bs.qkv_bias = false;
        check("qwen2", &m, Some(&Model::new(bs, w).unwrap()));
    }

    #[test]
    fn qk_norm_per_head() {
        let mut sp = base_spec(4, 2, false);
        sp.qk_norm = true;
        let m = model_of(sp, false);
        let mut w = m.w.clone();
        for l in &mut w.layers {
            l.q_norm = None;
            l.k_norm = None;
        }
        let mut bs = m.spec.clone();
        bs.qk_norm = false;
        check("qk_norm", &m, Some(&Model::new(bs, w).unwrap()));
    }

    #[test]
    fn gemma_family_everything_at_once() {
        let mut sp = base_spec(4, 2, true);
        sp.family = "gemma2".into();
        sp.norm_offset = true;
        sp.act = Act::GeluTanh;
        sp.embed_scale = 4.0;
        sp.post_norms = true;
        sp.attn_softcap = 5.0;
        sp.final_softcap = 3.0;
        sp.attn_scale = 0.2;
        sp.sliding_window = 3;
        sp.sliding_layers = vec![true, false, true];
        sp.rope_theta_local = 500.0;
        let m = model_of(sp, false);
        check("gemma", &m, None);
        for (what, f) in [
            (
                "offset",
                Box::new(|s: &mut Spec| s.norm_offset = false) as Box<dyn Fn(&mut Spec)>,
            ),
            ("gelu", Box::new(|s| s.act = Act::Silu)),
            ("embed scale", Box::new(|s| s.embed_scale = 1.0)),
            ("softcap", Box::new(|s| s.attn_softcap = 0.0)),
            ("final softcap", Box::new(|s| s.final_softcap = 0.0)),
            ("window", Box::new(|s| s.sliding_layers = vec![false; 3])),
            ("local theta", Box::new(|s| s.rope_theta_local = 0.0)),
            ("scale", Box::new(|s| s.attn_scale = 0.0)),
        ] {
            let b = without(&m, &*f);
            let d = decode(&m)
                .iter()
                .zip(decode(&b))
                .flat_map(|(x, y)| {
                    x.iter()
                        .zip(y)
                        .map(|(a, b)| (a - b).abs())
                        .collect::<Vec<_>>()
                })
                .fold(0.0, f32::max);
            assert!(d > 1e-4, "{what} had no effect ({d})");
        }
    }

    #[test]
    fn sliding_window_really_forgets_old_tokens() {
        let mut sp = base_spec(4, 2, false);
        sp.sliding_window = 2;
        sp.sliding_layers = vec![true; 3];
        let m = model_of(sp, false);
        check("window", &m, None);
        // logits after [A, x, y] must not depend on A when every layer sees only 2 tokens... through one layer only:
        let mut one = base_spec(4, 2, false);
        one.layers = 1;
        one.sliding_window = 2;
        one.sliding_layers = vec![true];
        let m1 = model_of(one, false);
        let run = |first: usize| {
            let mut st = m1.new_state();
            [first, 4, 6]
                .iter()
                .map(|&t| m1.forward(&mut st, t).unwrap())
                .last()
                .unwrap()
        };
        assert_eq!(run(1), run(9));
        let full = model_of(
            {
                let mut f = base_spec(4, 2, false);
                f.layers = 1;
                f
            },
            false,
        );
        let runf = |first: usize| {
            let mut st = full.new_state();
            [first, 4, 6]
                .iter()
                .map(|&t| full.forward(&mut st, t).unwrap())
                .last()
                .unwrap()
        };
        assert_ne!(runf(1), runf(9));
    }

    #[test]
    fn layer_norm_with_plain_mlp_and_parallel_residual() {
        let mut sp = base_spec(4, 4, false);
        sp.norm = NormKind::Layer;
        sp.gated_mlp = false;
        sp.act = Act::Gelu;
        sp.parallel_residual = true;
        let m = model_of(sp, true);
        check(
            "neox",
            &m,
            Some(&without(&m, |s| s.parallel_residual = false)),
        );
        let mut sq = base_spec(4, 2, true);
        sq.norm = NormKind::Layer;
        sq.gated_mlp = false;
        sq.act = Act::GeluTanh;
        sq.qkv_bias = true;
        let m = model_of(sq, true);
        check(
            "starcoder2",
            &m,
            Some(&without(&m, |s| s.norm = NormKind::Rms)),
        );
    }

    #[test]
    fn partial_rotary_leaves_the_tail_untouched() {
        let mut sp = base_spec(4, 2, false);
        sp.rotary_dim = 4;
        let m = model_of(sp, false);
        check(
            "partial rotary",
            &m,
            Some(&without(&m, |s| s.rotary_dim = 8)),
        );
        let (inv, ms) = rope_inv_freq(10_000.0, 4, RopeScaling::None);
        let mut head = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        rope_apply(&mut head, 11, &(inv, ms));
        assert_eq!(&head[4..], &[5.0, 6.0, 7.0, 8.0]);
        assert_ne!(&head[..4], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn explicit_head_dim_and_attention_scale() {
        // hidden 16, 4 heads but head_dim 8 already differs from 16/4; also try 4 heads x 6
        let mut sp = Spec::dense("qwen3", 2, 16, 4, 2, 6, 24, 20, 32);
        sp.qk_norm = true;
        let m = model_of(sp, false);
        check("head_dim 6", &m, None);
        let mut sc = base_spec(4, 2, false);
        sc.attn_scale = 0.9;
        let m = model_of(sc, false);
        check("scale", &m, Some(&without(&m, |s| s.attn_scale = 0.0)));
    }

    #[test]
    fn rope_scaling_variants_alter_the_model_and_follow_their_formulas() {
        let (base, _) = rope_inv_freq(10_000.0, 8, RopeScaling::None);
        let (lin, _) = rope_inv_freq(10_000.0, 8, RopeScaling::Linear { factor: 4.0 });
        for (a, b) in base.iter().zip(&lin) {
            assert!((a / 4.0 - b).abs() < 1e-12);
        }
        // llama3: the highest-frequency pair is untouched, the lowest is divided by the factor
        let l3 = RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max: 8192.0,
        };
        let (b64, _) = rope_inv_freq(500_000.0, 64, RopeScaling::None);
        let (s64, _) = rope_inv_freq(500_000.0, 64, l3);
        assert_eq!(b64[0], s64[0]);
        assert!((b64[31] / 8.0 - s64[31]).abs() < 1e-18);
        assert!(
            s64.iter()
                .zip(&b64)
                .all(|(s, b)| s <= b && *s >= b / 8.0 - 1e-18)
        );
        // yarn: high-frequency end untouched, low-frequency end interpolated, magnitude factor kept
        let yr = RopeScaling::Yarn {
            factor: 4.0,
            original_max: 4096.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.2,
        };
        let (y, ms) = rope_inv_freq(10_000.0, 64, yr);
        let (b, _) = rope_inv_freq(10_000.0, 64, RopeScaling::None);
        assert_eq!(ms, 1.2);
        assert!((y[0] - b[0]).abs() < 1e-12);
        assert!((y[31] - b[31] / 4.0).abs() < 1e-12);
        // head_dim 8 is tiny, so use settings that bite there (long wavelengths, small original window)
        let l3_small = RopeScaling::Llama3 {
            factor: 16.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max: 64.0,
        };
        for scaling in [
            RopeScaling::Linear { factor: 16.0 },
            l3_small,
            RopeScaling::Yarn {
                factor: 16.0,
                original_max: 16.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                mscale: 1.1,
            },
        ] {
            let mut sp = base_spec(4, 2, false);
            sp.rope_scaling = scaling;
            let m = model_of(sp, false);
            check(
                "rope scaling",
                &m,
                Some(&without(&m, |s| s.rope_scaling = RopeScaling::None)),
            );
        }
    }

    /// Golden values computed independently (plain Python, public formulas of Llama-3.1 and YaRN),
    /// not by this crate. They pin the mid-band interpolation, the YaRN correction range bounds
    /// (floor for the low end, ceil for the high end) and the direction of the ramp.
    #[test]
    fn rope_scaling_matches_independent_golden_values() {
        fn close(got: &[f64], want: &[f64]) {
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-12 * w.abs().max(1e-300),
                    "pair {i}: got {g:e}, want {w:e}"
                );
            }
        }
        // theta 10000, rotary dim 16, factor 8, low 1, high 4, original window 64: pairs 1 and 2 are mid-band
        let l3 = RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max: 64.0,
        };
        let (v, ms) = rope_inv_freq(10_000.0, 16, l3);
        assert_eq!(ms, 1.0);
        close(
            &v,
            &[
                1.0,
                0.24438459943539834,
                0.013042256043820465,
                0.003952847075210474,
                0.00125,
                0.0003952847075210474,
                0.000125,
                3.952847075210474e-05,
            ],
        );
        // YaRN with whole-number correction bounds (0 and 5) and a ramp that starts at pair 0
        let y1 = RopeScaling::Yarn {
            factor: 4.0,
            original_max: 64.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
        };
        let (v, _) = rope_inv_freq(10_000.0, 32, y1);
        close(
            &v,
            &[
                1.0,
                0.4779901264117967,
                0.22135943621178655,
                0.09780536755214075,
                0.04,
                0.014058533129758728,
                0.007905694150420948,
                0.004445698525097307,
                0.0025,
                0.0014058533129758727,
                0.0007905694150420948,
                0.0004445698525097307,
                0.00025,
                0.00014058533129758727,
                7.905694150420948e-05,
                4.445698525097307e-05,
            ],
        );
        // YaRN with fractional bounds: the low correction index is 0.21 (floor 0, not 1) and the high one 2.02 (ceil 3, not 2)
        let y2 = RopeScaling::Yarn {
            factor: 2.0,
            original_max: 128.0,
            beta_fast: 16.0,
            beta_slow: 2.0,
            mscale: 1.0,
        };
        let (v, _) = rope_inv_freq(10_000.0, 16, y2);
        close(
            &v,
            &[
                1.0,
                0.26352313834736496,
                0.06666666666666668,
                0.015811388300841896,
                0.005,
                0.0015811388300841897,
                0.0005,
                0.00015811388300841897,
            ],
        );
    }

    /// With a large epsilon the effect is far above any numeric tolerance, so a dropped or
    /// misplaced epsilon cannot hide (the reference below is plain f64, written independently).
    #[test]
    fn norm_epsilon_is_applied_inside_the_square_root() {
        let x = [3.0f32, 4.0, -1.0, 2.0];
        let w = [1.0f32, 0.5, 2.0, 1.5];
        let b = [0.25f32, -0.5, 0.0, 1.0];
        let eps = 2.0f64;
        let n = x.len() as f64;
        let mut sp = base_spec(4, 2, false);
        sp.norm_eps = eps;
        let mut o = [0.0f32; 4];
        // RMSNorm: x / sqrt(mean(x^2) + eps) * w
        sp.norm = NormKind::Rms;
        norm(&sp, &x, &w, None, &mut o);
        let ms: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n;
        for i in 0..4 {
            let want = x[i] as f64 / (ms + eps).sqrt() * w[i] as f64;
            assert!(
                (o[i] as f64 - want).abs() < 1e-5,
                "rms {i}: {} vs {want}",
                o[i]
            );
        }
        // LayerNorm: (x - mean) / sqrt(var + eps) * w + b
        sp.norm = NormKind::Layer;
        norm(&sp, &x, &w, Some(&b), &mut o);
        let mean: f64 = x.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var: f64 = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        for i in 0..4 {
            let want = (x[i] as f64 - mean) / (var + eps).sqrt() * w[i] as f64 + b[i] as f64;
            assert!(
                (o[i] as f64 - want).abs() < 1e-5,
                "layer {i}: {} vs {want}",
                o[i]
            );
        }
    }

    #[test]
    fn activations_match_known_values() {
        assert!((activate(Act::Silu, 1.0) - 0.731_058_6).abs() < 1e-6);
        assert!((activate(Act::Gelu, 1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((activate(Act::GeluTanh, 1.0) - 0.841_192).abs() < 1e-5);
        assert!((activate(Act::Gelu, -3.0) + 0.004_049_7).abs() < 1e-6);
    }

    #[test]
    fn norms_match_known_values() {
        let mut sp = base_spec(4, 2, false);
        sp.norm_eps = 0.0;
        let mut o = [0.0; 2];
        norm(&sp, &[3.0, 4.0], &[0.0, 1.0], None, &mut o);
        let k = 1.0 / 12.5f32.sqrt();
        assert!((o[0] - 0.0).abs() < 1e-6 && (o[1] - 4.0 * k).abs() < 1e-6);
        sp.norm_offset = true; // weight applied as 1 + w
        norm(&sp, &[3.0, 4.0], &[0.0, 1.0], None, &mut o);
        assert!((o[0] - 3.0 * k).abs() < 1e-6 && (o[1] - 8.0 * k).abs() < 1e-6);
        sp.norm = NormKind::Layer;
        norm(&sp, &[1.0, 3.0], &[1.0, 1.0], Some(&[10.0, 20.0]), &mut o);
        assert!((o[0] - 9.0).abs() < 1e-5 && (o[1] - 21.0).abs() < 1e-5);
    }

    #[test]
    fn wrong_shapes_tokens_and_full_context_are_errors() {
        let m = tiny(4, 2, false);
        let mut st = m.new_state();
        assert_eq!(m.forward(&mut st, 20), Err(ModelError::Token(20)));
        for _ in 0..32 {
            m.forward(&mut st, 1).unwrap();
        }
        assert_eq!(m.forward(&mut st, 1), Err(ModelError::ContextFull(32)));
        let mut bad = tiny(4, 2, false);
        bad.w.layers[1].wq = Weight::Dense(vec![0.0]);
        assert!(matches!(
            Model::new(bad.spec.clone(), bad.w),
            Err(ModelError::Shape(_))
        ));
        // required-but-missing pieces
        let mut sp = base_spec(4, 2, false);
        sp.qk_norm = true;
        let w = random_weights(&base_spec(4, 2, false), 1, false);
        assert!(matches!(Model::new(sp, w), Err(ModelError::Shape(e)) if e.contains("q_norm")));
        let mut sp = base_spec(4, 2, false);
        sp.rotary_dim = 5;
        assert!(Model::new(sp, random_weights(&base_spec(4, 2, false), 1, false)).is_err());
    }

    #[test]
    fn greedy_is_deterministic_and_sampling_respects_top_k() {
        let m = tiny(4, 2, false);
        let run = || {
            let mut st = m.new_state();
            let mut t = 2;
            let mut g = Sampler::new(0.0, 0, 1);
            (0..6)
                .map(|_| {
                    t = g.sample(&m.forward(&mut st, t).unwrap());
                    t
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
        let logits = [0.0, 5.0, 4.0, -3.0, 1.0];
        let mut s = Sampler::new(1.0, 2, 42);
        for _ in 0..200 {
            assert!(matches!(s.sample(&logits), 1 | 2));
        }
    }

    // ------------------------------------------------------------ quantised weights, KV cache, batching

    pub fn q8_0_encode(w: &[f32]) -> Vec<u8> {
        let mut o = vec![];
        for blk in w.chunks(32) {
            let d = blk.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0;
            let dh = f32_to_f16(d);
            let dd = crate::quant::f16_to_f32(dh);
            o.extend(dh.to_le_bytes());
            for v in blk {
                o.push(if dd > 0.0 {
                    (v / dd).round().clamp(-127.0, 127.0)
                } else {
                    0.0
                } as i8 as u8);
            }
        }
        o
    }
    pub fn bf16_encode(w: &[f32]) -> Vec<u8> {
        w.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
    fn requant(w: &Weight, rows: usize, cols: usize, dt: DType) -> Weight {
        let v = w.as_dense().unwrap();
        if v.is_empty() {
            return w.clone();
        }
        let bytes = if dt == DType::Q8_0 {
            q8_0_encode(v)
        } else {
            bf16_encode(v)
        };
        Weight::Quant(QWeight::new(dt, rows, cols, Bytes::owned(bytes)).unwrap())
    }
    /// The same model with every big matrix stored as `dt` (Q8_0 or BF16). Needs hidden/ffn/qd multiples of 32 for Q8_0.
    pub fn requantise(m: &Model, dt: DType) -> Model {
        let s = &m.spec;
        let (h, qd, kd, ff, v) = (
            s.hidden as usize,
            (s.heads * s.head_dim) as usize,
            (s.kv_heads * s.head_dim) as usize,
            s.ffn as usize,
            s.vocab as usize,
        );
        let mut w = m.w.clone();
        w.embed = requant(&w.embed, v, h, dt);
        if let Some(l) = &w.lm_head {
            w.lm_head = Some(requant(l, v, h, dt));
        }
        for l in &mut w.layers {
            l.wq = requant(&l.wq, qd, h, dt);
            l.wk = requant(&l.wk, kd, h, dt);
            l.wv = requant(&l.wv, kd, h, dt);
            l.wo = requant(&l.wo, h, qd, dt);
            l.w_gate = requant(&l.w_gate, if s.gated_mlp { ff } else { 0 }, h, dt);
            l.w_up = requant(&l.w_up, ff, h, dt);
            l.w_down = requant(&l.w_down, h, ff, dt);
        }
        Model::new(m.spec.clone(), w).unwrap()
    }
    fn requant_any(w: &Weight, rows: usize, cols: usize, dt: DType) -> Weight {
        let v = w.as_dense().unwrap();
        if v.is_empty() {
            return w.clone();
        }
        let bytes = crate::quantize::quantize_matrix(dt, v, cols, None, 1).unwrap();
        Weight::Quant(QWeight::new(dt, rows, cols, Bytes::owned(bytes)).unwrap())
    }
    /// Like [`requantise`] for any block format the quantiser can write (Q8_0, Q4_0, Q4_K…). The
    /// column counts (hidden, ffn, query width) must be multiples of the format's block size.
    pub(crate) fn requantise_to(m: &Model, dt: DType) -> Model {
        let s = &m.spec;
        let (h, qd, kd, ff, v) = (
            s.hidden as usize,
            (s.heads * s.head_dim) as usize,
            (s.kv_heads * s.head_dim) as usize,
            s.ffn as usize,
            s.vocab as usize,
        );
        let mut w = m.w.clone();
        w.embed = requant_any(&w.embed, v, h, dt);
        if let Some(l) = &w.lm_head {
            w.lm_head = Some(requant_any(l, v, h, dt));
        }
        for l in &mut w.layers {
            l.wq = requant_any(&l.wq, qd, h, dt);
            l.wk = requant_any(&l.wk, kd, h, dt);
            l.wv = requant_any(&l.wv, kd, h, dt);
            l.wo = requant_any(&l.wo, h, qd, dt);
            l.w_gate = requant_any(&l.w_gate, if s.gated_mlp { ff } else { 0 }, h, dt);
            l.w_up = requant_any(&l.w_up, ff, h, dt);
            l.w_down = requant_any(&l.w_down, h, ff, dt);
        }
        Model::new(m.spec.clone(), w).unwrap()
    }
    /// A model whose width suits every block format (256-wide), for the exact-batch tests.
    pub(crate) fn block_model() -> Model {
        let mut sp = Spec::dense("llama", 4, 256, 4, 2, 64, 256, 48, 300);
        sp.tied_embeddings = false;
        model_of(sp, false)
    }
    fn wide_spec(tied: bool) -> Spec {
        let mut sp = Spec::dense("llama", 2, 32, 4, 2, 8, 64, 40, 300);
        sp.tied_embeddings = tied;
        sp
    }
    fn seq_logits(m: &Model, toks: &[usize]) -> Vec<f32> {
        let mut st = m.new_state();
        let mut last = vec![];
        for &t in toks {
            last = m.forward(&mut st, t).unwrap();
        }
        last
    }
    #[test]
    fn exact_batch_equals_token_by_token_for_quantised_weights_and_off_stays_close() {
        let base = block_model();
        let toks: Vec<usize> = (0..11).map(|i| (i * 7 + 3) % 48).collect();
        for dt in [DType::Q8_0, DType::Q4_0, DType::Q4K] {
            let m = requantise_to(&base, dt);
            let single = seq_logits(&m, &toks);
            // exact on: one batch of 11 tokens gives the token-by-token logits bit for bit
            let mut st = m.new_state();
            st.set_exact_batch(true);
            assert!(st.exact_batch());
            let batch = m.forward_batch(&mut st, &toks).unwrap();
            assert_eq!(
                single.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                batch.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{dt:?} exact"
            );
            // ... and so does the cache it leaves behind
            let mut s2 = m.new_state();
            for &t in &toks {
                m.forward(&mut s2, t).unwrap();
            }
            let (a, b) = (
                m.forward(&mut st, 5).unwrap(),
                m.forward(&mut s2, 5).unwrap(),
            );
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{dt:?} next step"
            );
            // forks keep the mode
            assert!(st.fork().unwrap().exact_batch());
            // exact off (the default): only rounding noise from the f32-activation path
            let mut st = m.new_state();
            let fast = m.forward_batch(&mut st, &toks).unwrap();
            let tol = 2.0e-1; // int8-activation rounding on a tiny random model; the point is "close", not "equal"
            assert!(
                rel_diff(&fast, &single) < tol,
                "{dt:?}: {}",
                rel_diff(&fast, &single)
            );
            // the reason the option exists: the fast path is not bit-identical to token-by-token
            assert!(
                fast.iter()
                    .zip(&single)
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "{dt:?}: expected rounding noise with exact off"
            );
        }
    }
    fn rel_diff(a: &[f32], b: &[f32]) -> f32 {
        let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
            / scale
    }

    #[test]
    fn half_precision_conversion() {
        for h in 0..=u16::MAX {
            let f = crate::quant::f16_to_f32(h);
            if f.is_nan() {
                assert!(crate::quant::f16_to_f32(f32_to_f16(f)).is_nan());
            } else {
                assert_eq!(f32_to_f16(f), h, "{h:#06x}");
            }
        }
        for i in -2000..2000 {
            let x = i as f32 * 0.0137;
            let r = crate::quant::f16_to_f32(f32_to_f16(x));
            assert!((r - x).abs() <= x.abs() * 6e-4 + 1e-7, "{x} -> {r}");
        }
        assert_eq!(f32_to_f16(1e9), 0x7c00);
        assert_eq!(f32_to_f16(-1e9), 0xfc00);
        assert_eq!(f32_to_f16(1e-9), 0);
        assert_eq!(f32_to_f16(65519.0), 0x7bff);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
    }

    #[test]
    fn dot8_and_threaded_matmat_match_plain_loops() {
        for n in [0usize, 1, 7, 8, 9, 33, 100] {
            let (a, b): (Vec<f32>, Vec<f32>) = (0..n)
                .map(|i| ((i as f32).sin(), (i as f32 * 0.3).cos()))
                .unzip();
            let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot8(&a, &b) - want).abs() < 1e-4, "{n}");
        }
        let (rows, cols, bt) = (1024usize, 512usize, 8usize);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.01)
            .collect();
        let x: Vec<f32> = (0..bt * cols).map(|i| (i as f32 * 0.01).sin()).collect();
        let wt = Weight::Dense(w.clone());
        let (mut out, mut yt) = (vec![0.0; bt * rows], vec![]);
        wt.matmat(cols, &x, bt, &mut out, &mut yt).unwrap();
        for t in 0..bt {
            let mut one = vec![0.0; rows];
            wt.matvec(cols, &x[t * cols..(t + 1) * cols], &mut one)
                .unwrap();
            assert_eq!(&out[t * rows..(t + 1) * rows], &one[..], "token {t}");
            let r = 77;
            let naive: f32 = (0..cols).map(|c| w[r * cols + c] * x[t * cols + c]).sum();
            assert!((one[r] - naive).abs() < 1e-3);
        }
        assert!(
            wt.matvec(cols, &x[..cols - 1], &mut vec![0.0; rows])
                .is_err()
        );
        assert!(wt.matmat(cols, &x, 0, &mut [], &mut yt).is_err());
    }

    #[test]
    fn quantised_weights_track_the_dense_model_and_use_less_memory() {
        for tied in [false, true] {
            let dense = model_of(wide_spec(tied), false);
            let toks = [3usize, 7, 1, 19, 0, 7, 12, 5, 9, 31];
            let want = seq_logits(&dense, &toks);
            let q8 = requantise(&dense, DType::Q8_0);
            assert!(q8.w.layers[0].wq.as_dense().is_none());
            let got = seq_logits(&q8, &toks);
            let d = rel_diff(&got, &want);
            assert!(d < 0.06, "q8 tied={tied}: relative error {d}");
            assert!(
                q8.weight_bytes() * 10 < dense.weight_bytes() * 3,
                "{} vs {}",
                q8.weight_bytes(),
                dense.weight_bytes()
            );
            // batched prefill decodes each row once; it must agree with token-by-token on the same quantised weights
            let mut st = q8.new_state();
            let b = q8.forward_batch(&mut st, &toks).unwrap();
            assert!(
                rel_diff(&b, &got) < 0.05,
                "quantised batch vs sequential: {}",
                rel_diff(&b, &got)
            );
            // BF16 kept as raw halves is exact against a dense model holding the same rounded numbers
            let mut rounded = dense.w.clone();
            let rd = |w: &mut Weight| {
                if let Weight::Dense(v) = w {
                    v.iter_mut()
                        .for_each(|x| *x = f32::from_bits((x.to_bits() >> 16) << 16));
                }
            };
            rd(&mut rounded.embed);
            if let Some(l) = &mut rounded.lm_head {
                rd(l);
            }
            for l in &mut rounded.layers {
                for w in [
                    &mut l.wq,
                    &mut l.wk,
                    &mut l.wv,
                    &mut l.wo,
                    &mut l.w_gate,
                    &mut l.w_up,
                    &mut l.w_down,
                ] {
                    rd(w);
                }
            }
            let bf_dense = Model::new(dense.spec.clone(), rounded).unwrap();
            let bf = requantise(&bf_dense, DType::BF16);
            assert!(rel_diff(&seq_logits(&bf, &toks), &seq_logits(&bf_dense, &toks)) < 1e-4);
            assert!(bf.weight_bytes() * 2 <= dense.weight_bytes() + 8);
        }
    }

    #[test]
    fn quantised_shapes_are_checked() {
        let dense = model_of(wide_spec(false), false);
        let q8 = requantise(&dense, DType::Q8_0);
        // wrong stored shape for a layer
        let mut w = q8.w.clone();
        if let Weight::Quant(q) = &mut w.layers[0].wq {
            q.rows -= 1;
        }
        assert!(matches!(
            Model::new(q8.spec.clone(), w),
            Err(ModelError::Shape(_))
        ));
        // rows that are not whole blocks, or bytes that do not add up, never construct
        assert!(QWeight::new(DType::Q8_0, 2, 33, Bytes::owned(vec![0; 200])).is_err());
        assert!(QWeight::new(DType::Q8_0, 2, 32, Bytes::owned(vec![0; 67])).is_err());
        assert!(QWeight::new(DType::Other(99), 1, 32, Bytes::owned(vec![0; 32])).is_err());
        assert!(QWeight::new(DType::Q8_0, usize::MAX, 32, Bytes::owned(vec![0; 34])).is_err());
        // an out-of-range token on a quantised embedding is a clean error
        let mut st = q8.new_state();
        assert_eq!(q8.forward(&mut st, 40), Err(ModelError::Token(40)));
        assert!(Bytes::owned(vec![1, 2, 3]).slice(2, 2).is_none());
        assert_eq!(
            Bytes::owned(vec![1, 2, 3]).slice(1, 2).unwrap().as_slice(),
            &[2, 3]
        );
    }

    #[test]
    fn kv_cache_precisions_trade_memory_for_accuracy() {
        let m = tiny(4, 2, false);
        let toks: Vec<usize> = (0..24).map(|i| (i * 7 + 3) % 20).collect();
        let run = |kind| {
            let mut st = m.new_state_with(kind);
            let mut last = vec![];
            for &t in &toks {
                last = m.forward(&mut st, t).unwrap();
            }
            (last, st.kv_bytes())
        };
        let (base, b32) = run(KvKind::F32);
        let (h, b16) = run(KvKind::F16);
        let (q, b8) = run(KvKind::Q8);
        assert_eq!(b32, 3 * 2 * 24 * 16 * 4);
        assert_eq!(b16 * 2, b32);
        assert!(b8 * 5 < b32 * 2, "{b8} vs {b32}");
        let (dh, dq) = (rel_diff(&h, &base), rel_diff(&q, &base));
        assert!(dh < 2e-3, "f16 kv: {dh}");
        assert!(dq < 3e-2, "q8 kv: {dq}");
        assert!(
            dh > 0.0 && dq > dh,
            "lower precision must actually change something ({dh}, {dq})"
        );
        // the default is exact
        assert_eq!(run(KvKind::default()).0, base);
    }

    #[test]
    fn sliding_layers_keep_only_their_window_and_still_match_the_reference() {
        let mut sp = base_spec(4, 2, true);
        sp.sliding_window = 3;
        sp.sliding_layers = vec![true, false, true];
        sp.rope_theta_local = 500.0;
        let m = model_of(sp, false);
        let toks: Vec<usize> = (0..30).map(|i| (i * 5 + 1) % 20).collect();
        let mut st = m.new_state();
        for (i, &t) in toks.iter().enumerate() {
            let got = m.forward(&mut st, t).unwrap();
            if [3usize, 9, 20, 29].contains(&i) {
                let want = reference_logits(&m, &toks[..=i]);
                for (a, b) in got.iter().zip(&want) {
                    assert!((a - b).abs() < 3e-4, "step {i}: {a} vs {b}");
                }
            }
        }
        // two ring layers of 3 positions and one full layer of 30, K and V, 2 kv heads x 8 dims, f32
        assert_eq!(st.kv_bytes(), (3 + 30 + 3) * 2 * 16 * 4);
    }

    #[test]
    fn embed_is_deterministic_unit_length_and_chunk_independent() {
        let m = tiny(4, 2, true);
        let toks = [3usize, 7, 1, 9, 4, 4, 2, 11];
        for p in [Pooling::Mean, Pooling::Last, Pooling::First] {
            let a = m.embed(&toks, p, true).unwrap();
            assert_eq!(a.len(), m.spec.hidden as usize);
            let n: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum();
            assert!((n - 1.0).abs() < 1e-4, "{p:?} norm {n}");
            assert_eq!(a, m.embed(&toks, p, true).unwrap());
            for c in [1, 3, 5, 100] {
                let b = m.embed_chunked(&toks, p, true, c).unwrap();
                for (x, y) in a.iter().zip(&b) {
                    assert!((x - y).abs() < 1e-5, "{p:?} chunk {c}: {x} vs {y}");
                }
            }
        }
        // a single token: every pooling agrees
        let one = [5usize];
        let (m1, l1, f1) = (
            m.embed(&one, Pooling::Mean, false).unwrap(),
            m.embed(&one, Pooling::Last, false).unwrap(),
            m.embed(&one, Pooling::First, false).unwrap(),
        );
        assert_eq!(m1, l1);
        assert_eq!(l1, f1);
        // different text, different vector
        assert_ne!(
            m.embed(&[1, 2, 3], Pooling::Mean, true).unwrap(),
            m.embed(&[3, 2, 1], Pooling::Last, true).unwrap()
        );
    }

    #[test]
    fn embed_rejects_bad_input() {
        let m = tiny(4, 2, true);
        assert_eq!(m.embed(&[], Pooling::Mean, true), Err(ModelError::NoTokens));
        assert_eq!(
            m.embed(&[999], Pooling::Mean, true),
            Err(ModelError::Token(999))
        );
        let long = vec![1usize; m.spec.context as usize + 1];
        assert!(matches!(
            m.embed(&long, Pooling::Mean, true),
            Err(ModelError::ContextFull(_))
        ));
        // exactly the context window is fine
        assert!(
            m.embed(&long[..m.spec.context as usize], Pooling::Mean, true)
                .is_ok()
        );
    }

    #[test]
    fn batched_prefill_equals_token_by_token() {
        let mut gem = base_spec(4, 2, true);
        gem.norm_offset = true;
        gem.act = Act::GeluTanh;
        gem.embed_scale = 4.0;
        gem.post_norms = true;
        gem.qk_norm = true;
        gem.attn_softcap = 5.0;
        gem.final_softcap = 3.0;
        gem.sliding_window = 3;
        gem.sliding_layers = vec![true, false, true];
        gem.rope_theta_local = 500.0;
        let mut neox = base_spec(4, 4, false);
        neox.norm = NormKind::Layer;
        neox.gated_mlp = false;
        neox.act = Act::Gelu;
        neox.parallel_residual = true;
        for (name, m) in [
            ("plain", tiny(4, 2, false)),
            ("gemma", model_of(gem, true)),
            ("neox", model_of(neox, true)),
            ("mqa", tiny(4, 1, true)),
        ] {
            let toks = [3usize, 7, 1, 19, 0, 7, 12, 5, 9, 2, 8, 14];
            let want = seq_logits(&m, &toks);
            let mut st = m.new_state();
            let all = m.forward_batch(&mut st, &toks).unwrap();
            assert_eq!(st.pos, toks.len());
            assert!(
                rel_diff(&all, &want) < 2e-4,
                "{name}: {}",
                rel_diff(&all, &want)
            );
            // split into uneven batches, then continue one token at a time
            let mut st2 = m.new_state();
            m.forward_batch(&mut st2, &toks[..5]).unwrap();
            m.forward_batch(&mut st2, &toks[5..6]).unwrap();
            let split = m.forward_batch(&mut st2, &toks[6..]).unwrap();
            assert!(
                rel_diff(&split, &want) < 2e-4,
                "{name} split: {}",
                rel_diff(&split, &want)
            );
            let next_a = m.forward(&mut st, 4).unwrap();
            let next_b = m.forward(&mut st2, 4).unwrap();
            assert!(rel_diff(&next_a, &next_b) < 2e-4, "{name} continuation");
        }
    }

    #[test]
    fn prefill_longer_than_one_chunk() {
        let m = model_of(wide_spec(false), false);
        let toks: Vec<usize> = (0..PREFILL_CHUNK + 22).map(|i| (i * 11 + 5) % 40).collect();
        let want = seq_logits(&m, &toks);
        let mut st = m.new_state();
        let got = m.forward_batch(&mut st, &toks).unwrap();
        assert!(rel_diff(&got, &want) < 1e-3, "{}", rel_diff(&got, &want));
        assert_eq!(st.pos, toks.len());
    }

    #[test]
    fn batch_errors_leave_the_state_untouched() {
        let m = tiny(4, 2, false);
        let mut st = m.new_state();
        assert_eq!(m.forward_batch(&mut st, &[]), Err(ModelError::NoTokens));
        assert_eq!(
            m.forward_batch(&mut st, &[1, 2, 99]),
            Err(ModelError::Token(99))
        );
        assert_eq!(
            m.forward_batch(&mut st, &[1; 33]),
            Err(ModelError::ContextFull(32))
        );
        assert_eq!(st.pos, 0);
        assert_eq!(st.kv_bytes(), 0);
        assert!(matches!(
            m.forward(&mut State::default(), 1),
            Err(ModelError::Shape(_))
        ));
        // after the failures the state still works and gives the usual answer
        assert_eq!(m.forward(&mut st, 3).unwrap(), seq_logits(&m, &[3]));
    }

    #[test]
    fn generate_stops_for_every_reason() {
        let m = tiny(4, 2, false);
        let prompt = [3usize, 7, 1];
        let run = |opts: &GenerateOptions, cb: &mut dyn FnMut(usize) -> bool| {
            let mut st = m.new_state();
            let mut s = Sampler::new(0.0, 0, 1);
            m.generate(&mut st, &prompt, &mut s, opts, cb).unwrap()
        };
        let mut got = vec![];
        let st = run(
            &GenerateOptions {
                max_tokens: 6,
                ..Default::default()
            },
            &mut |t| {
                got.push(t);
                true
            },
        );
        assert_eq!(
            (st.generated, st.stop, st.prompt_tokens),
            (6, StopReason::MaxTokens, 3)
        );
        // identical to driving the model by hand
        let mut state = m.new_state();
        let mut logits = m.forward_batch(&mut state, &prompt).unwrap();
        let mut by_hand = vec![];
        for _ in 0..6 {
            let t = Sampler::new(0.0, 0, 1).sample(&logits);
            by_hand.push(t);
            logits = m.forward(&mut state, t).unwrap();
        }
        assert_eq!(got, by_hand);
        // a stop token ends the run and is not reported
        let mut seen = vec![];
        let st = run(
            &GenerateOptions {
                max_tokens: 6,
                stop_tokens: vec![got[2]],
                ..Default::default()
            },
            &mut |t| {
                seen.push(t);
                true
            },
        );
        assert_eq!(st.stop, StopReason::StopToken);
        assert_eq!(
            seen,
            got[..got.iter().position(|&t| t == got[2]).unwrap()].to_vec()
        );
        // callback can stop it
        let mut n = 0;
        let st = run(&GenerateOptions::default(), &mut |_| {
            n += 1;
            n < 2
        });
        assert_eq!((st.generated, st.stop), (2, StopReason::Callback));
        // cancellation before the first token
        let flag = AtomicBool::new(true);
        let st = run(
            &GenerateOptions {
                cancel: Some(&flag),
                ..Default::default()
            },
            &mut |_| true,
        );
        assert_eq!((st.generated, st.stop), (0, StopReason::Cancelled));
        // a full context ends the run quietly
        let mut st = m.new_state();
        let long = vec![1usize; 30];
        let r = m
            .generate(
                &mut st,
                &long,
                &mut Sampler::new(0.0, 0, 1),
                &GenerateOptions {
                    max_tokens: 100,
                    ..Default::default()
                },
                &mut |_| true,
            )
            .unwrap();
        assert_eq!(
            (r.generated, r.stop, st.pos),
            (3, StopReason::ContextFull, 32)
        );
        // a prompt that does not fit is an error, an empty one too
        assert_eq!(
            m.generate(
                &mut m.new_state(),
                &[1; 40],
                &mut Sampler::new(0.0, 0, 1),
                &GenerateOptions::default(),
                &mut |_| true
            )
            .err(),
            Some(ModelError::ContextFull(32))
        );
        assert_eq!(
            m.generate(
                &mut m.new_state(),
                &[],
                &mut Sampler::new(0.0, 0, 1),
                &GenerateOptions::default(),
                &mut |_| true
            )
            .err(),
            Some(ModelError::NoTokens)
        );
    }

    #[test]
    fn sampler_top_p_and_repetition_penalty() {
        let logits = [10.0, 9.0, 0.0, 0.0, 0.0];
        let mut s = Sampler::new(1.0, 0, 7).with_top_p(0.5);
        for _ in 0..100 {
            assert_eq!(s.sample(&logits), 0);
        }
        let mut s = Sampler::new(1.0, 0, 7).with_top_p(0.9);
        let seen: std::collections::HashSet<usize> = (0..300).map(|_| s.sample(&logits)).collect();
        assert!(
            seen.is_subset(&[0usize, 1].into_iter().collect()) && seen.len() == 2,
            "{seen:?}"
        );
        // greedy with a penalty moves off a repeated token, for positive and negative logits
        let mut g = Sampler::new(0.0, 0, 1).with_repetition_penalty(2.0, 8);
        assert_eq!(g.sample(&[2.0, 1.9]), 0);
        g.observe(0);
        assert_eq!(g.sample(&[2.0, 1.9]), 1);
        assert_eq!(g.sample(&[-1.0, -1.1]), 1);
        // only the last n tokens count
        let mut g = Sampler::new(0.0, 0, 1).with_repetition_penalty(2.0, 1);
        g.observe_all(&[0, 1]);
        assert_eq!(g.sample(&[2.0, 1.9]), 0);
        g.clear_history();
        assert_eq!(g.sample(&[2.0, 1.9]), 0);
        // degenerate logits do not panic
        assert_eq!(Sampler::new(1.0, 0, 1).sample(&[f32::NAN, f32::NAN]), 0);
        assert_eq!(Sampler::new(1.0, 3, 1).sample(&[]), 0);
        // top-k on a big vocabulary only ever returns one of the k best
        let big: Vec<f32> = (0..50_000).map(|i| ((i * 7919) % 50_000) as f32).collect();
        let mut s = Sampler::new(1.0, 5, 3);
        for _ in 0..50 {
            assert!(big[s.sample(&big)] >= 49_995.0);
        }
    }

    // ------------------------------------------------------------ accelerator, prefix reuse, threads
    pub(crate) mod accel_and_prefix {
        use super::*;
        use crate::backend::{BackendError, CpuBackend};
        use crate::moe::{ExpertWeights, InMemoryExperts, MoeBlock, Router, RouterConfig, Scoring};
        use std::collections::HashMap;

        type R<T> = Result<T, BackendError>;
        /// An "accelerator" that wraps the CPU backend and counts every call.
        struct Counting {
            inner: CpuBackend,
            calls: Mutex<HashMap<&'static str, usize>>,
            mem: Option<u64>,
        }
        impl Counting {
            fn new(mem: Option<u64>) -> Arc<Counting> {
                Arc::new(Counting {
                    inner: CpuBackend::new(),
                    calls: Mutex::new(HashMap::new()),
                    mem,
                })
            }
            fn bump(&self, op: &'static str) {
                *lock(&self.calls).entry(op).or_insert(0) += 1;
            }
            fn n(&self, op: &str) -> usize {
                lock(&self.calls).get(op).copied().unwrap_or(0)
            }
        }
        impl Backend for Counting {
            fn name(&self) -> String {
                "counting".into()
            }
            fn is_accelerator(&self) -> bool {
                true
            }
            fn device_memory(&self) -> Option<u64> {
                self.mem
            }
            fn supports(&self, t: DType) -> bool {
                self.inner.supports(t)
            }
            fn alloc(&self, len: usize) -> R<Buf> {
                self.inner.alloc(len)
            }
            fn write(&self, dst: &Buf, src: &[f32]) -> R<()> {
                self.bump("write");
                self.inner.write(dst, src)
            }
            fn read(&self, src: &Buf, dst: &mut [f32]) -> R<()> {
                self.bump("read");
                self.inner.read(src, dst)
            }
            fn upload(&self, w: WeightRef<'_>) -> R<WeightHandle> {
                self.bump("upload");
                self.inner.upload(w)
            }
            fn matvec(&self, w: &WeightHandle, x: &Buf, out: &Buf) -> R<()> {
                self.bump("matvec");
                self.inner.matvec(w, x, out)
            }
            fn matmat(&self, w: &WeightHandle, x: &Buf, batch: usize, out: &Buf) -> R<()> {
                self.bump("matmat");
                self.inner.matmat(w, x, batch, out)
            }
            fn rmsnorm(&self, x: &Buf, weight: &Buf, eps: f32, out: &Buf) -> R<()> {
                self.bump("rmsnorm");
                self.inner.rmsnorm(x, weight, eps, out)
            }
            fn rope(&self, x: &Buf, n: usize, hd: usize, pos: usize, theta: f64) -> R<()> {
                self.bump("rope");
                self.inner.rope(x, n, hd, pos, theta)
            }
            fn silu_mul(&self, g: &Buf, u: &Buf) -> R<()> {
                self.bump("silu_mul");
                self.inner.silu_mul(g, u)
            }
            fn add(&self, a: &Buf, b: &Buf) -> R<()> {
                self.bump("add");
                self.inner.add(a, b)
            }
            fn kv_create(&self, kvh: usize, hd: usize, cap: usize) -> R<Kv> {
                self.bump("kv_create");
                self.inner.kv_create(kvh, hd, cap)
            }
            fn kv_append(&self, kv: &Kv, pos: usize, k: &Buf, v: &Buf) -> R<()> {
                self.bump("kv_append");
                self.inner.kv_append(kv, pos, k, v)
            }
            fn attention(&self, kv: &Kv, q: &Buf, out: &Buf, a: &AttnArgs) -> R<()> {
                self.bump("attention");
                self.inner.attention(kv, q, out, a)
            }
        }

        const T8: [usize; 8] = [3, 7, 1, 19, 0, 7, 12, 5];
        fn twin(m: &Model) -> Model {
            Model::new(m.spec.clone(), m.w.clone()).unwrap()
        }
        fn run_all(m: &Model) -> (Vec<Vec<f32>>, Vec<f32>) {
            let mut st = m.new_state();
            let steps = T8.iter().map(|&t| m.forward(&mut st, t).unwrap()).collect();
            let mut st = m.new_state();
            (steps, m.forward_batch(&mut st, &T8).unwrap())
        }

        #[test]
        fn offloaded_layers_go_through_the_backend_and_results_do_not_change() {
            let base = tiny(4, 2, false);
            let (want_steps, want_batch) = run_all(&base);
            // (offload limit, a layer with a bias that the backend cannot do, expected layers)
            for (limit, biased, expect) in [
                (None, None, vec![true, true, true]),
                (Some(2), None, vec![true, true, false]),
                (None, Some(1), vec![true, false, true]),
            ] {
                let mut m = twin(&base);
                if let Some(i) = biased {
                    m.w.layers[i].bq = Some(vec![0.0; 32]);
                }
                let be = Counting::new(None);
                let m = m
                    .with_backend_opts(
                        be.clone(),
                        AccelOptions {
                            max_layers: limit,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(m.offloaded(), expect);
                let l = expect.iter().filter(|&&e| e).count();
                assert_eq!(be.n("upload"), l * 7);
                let (steps, batch) = run_all(&m);
                for (i, (a, b)) in steps.iter().zip(&want_steps).enumerate() {
                    assert!(
                        rel_diff(a, b) < 2e-4,
                        "{expect:?} step {i}: {}",
                        rel_diff(a, b)
                    );
                }
                assert!(rel_diff(&batch, &want_batch) < 2e-4, "{expect:?} batch");
                // two passes of 8 tokens (token by token, then as one batch)
                let t = 2 * T8.len();
                assert_eq!(be.n("kv_create"), 2 * l, "{expect:?}");
                assert_eq!(be.n("matvec"), l * 7 * t);
                assert_eq!(be.n("attention"), l * t);
                assert_eq!(be.n("kv_append"), l * t);
                assert_eq!(be.n("rope"), l * 2 * t);
                assert_eq!(be.n("rmsnorm"), l * 2 * t);
                assert_eq!(be.n("add"), l * 2 * t);
                assert_eq!(be.n("silu_mul"), l * t);
                assert_eq!(be.n("matmat"), 0);
                // the hidden state crosses the boundary once per token per run of accelerated layers
                let runs =
                    expect.windows(2).filter(|w| w[1] && !w[0]).count() + usize::from(expect[0]);
                assert_eq!(be.n("read"), runs * t, "{expect:?}");
            }
        }

        #[test]
        fn device_memory_decides_how_many_layers_fit() {
            let base = tiny(4, 2, false);
            let per_layer: u64 = [
                &base.w.layers[0].wq,
                &base.w.layers[0].wk,
                &base.w.layers[0].wv,
                &base.w.layers[0].wo,
                &base.w.layers[0].w_gate,
                &base.w.layers[0].w_up,
                &base.w.layers[0].w_down,
            ]
            .iter()
            .map(|w| w.stored_bytes() as u64)
            .sum();
            let kv = 32 * 16 * 2 * 4; // context 32, 2 kv heads x 8
            let m = twin(&base)
                .with_backend_opts(
                    Counting::new(Some(per_layer + kv + per_layer / 2)),
                    AccelOptions {
                        reserve: 0,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(m.offloaded(), vec![true, false, false]);
            let m = twin(&base)
                .with_backend_opts(
                    Counting::new(Some(10)),
                    AccelOptions {
                        reserve: 0,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(m.offloaded(), vec![false, false, false]);
            // nothing offloaded: identical to the plain model
            let (a, _) = run_all(&m);
            let (b, _) = run_all(&base);
            assert_eq!(a, b);
        }

        #[test]
        fn quantised_weights_are_uploaded_as_they_are_and_unsupported_variants_stay_home() {
            let q = requantise(&model_of(wide_spec(false), false), DType::Q8_0);
            let be = Counting::new(None);
            let m = twin(&q).with_backend(be.clone()).unwrap();
            assert_eq!(m.offloaded(), vec![true, true]);
            let (a, _) = run_all(&m);
            let (b, _) = run_all(&q);
            for (x, y) in a.iter().zip(&b) {
                assert!(rel_diff(x, y) < 5e-2, "{}", rel_diff(x, y));
            }
            // Gemma-style norms and QK-norm are not in the trait: those layers are never offloaded
            let mut gem = base_spec(4, 2, true);
            gem.norm_offset = true;
            gem.act = Act::GeluTanh;
            gem.qk_norm = true;
            let m = model_of(gem, true)
                .with_backend(Counting::new(None))
                .unwrap();
            assert_eq!(m.offloaded(), vec![false, false, false]);
        }

        #[test]
        fn the_accelerator_cache_has_a_capacity_and_foreign_states_are_refused() {
            let be = Counting::new(None);
            let m = twin(&tiny(4, 2, false))
                .with_backend_opts(
                    be,
                    AccelOptions {
                        kv_capacity: 4,
                        ..Default::default()
                    },
                )
                .unwrap();
            let mut st = m.new_state();
            m.forward_batch(&mut st, &T8[..4]).unwrap();
            assert_eq!(m.forward(&mut st, 1), Err(ModelError::ContextFull(4)));
            assert!(st.fork().is_err());
            let host = tiny(4, 2, false);
            let mut hs = host.new_state();
            host.forward(&mut hs, 1).unwrap();
            assert!(matches!(m.forward(&mut hs, 2), Err(ModelError::Backend(_))));
        }

        // ------------------------------------------------------------------ prefix reuse
        const A: [usize; 8] = [3, 7, 1, 19, 0, 7, 12, 5];
        const B: [usize; 9] = [3, 7, 1, 19, 0, 9, 2, 4, 6]; // shares five tokens with A

        fn fresh(m: &Model, kind: KvKind, toks: &[usize]) -> Vec<f32> {
            let mut st = m.new_state_with(kind);
            m.forward_batch(&mut st, toks).unwrap()
        }

        /// `prefill_cached` must give what a fresh state gives, whatever the state held before.
        fn check_prefix(name: &str, m: &Model, kind: KvKind, tol: f32, expect_rewind: bool) {
            let mut st = m.new_state_with(kind);
            let la = m.prefill_cached(&mut st, &A).unwrap();
            assert_eq!(st.reused(), 0, "{name}");
            assert!(rel_diff(&la, &fresh(m, kind, &A)) <= tol, "{name} first");
            // the same prompt again: everything is reused
            assert_eq!(m.prefill_cached(&mut st, &A).unwrap(), la, "{name}");
            assert_eq!(st.reused(), A.len());
            // a longer prompt that starts with the old one
            let ext: Vec<usize> = A.iter().copied().chain([4, 8]).collect();
            let le = m.prefill_cached(&mut st, &ext).unwrap();
            assert_eq!(st.reused(), A.len(), "{name}");
            assert!(
                rel_diff(&le, &fresh(m, kind, &ext)) <= tol,
                "{name} extend: {}",
                rel_diff(&le, &fresh(m, kind, &ext))
            );
            assert_eq!(st.tokens(), &ext[..]);
            // a different continuation after five shared tokens
            let lb = m.prefill_cached(&mut st, &B).unwrap();
            assert_eq!(
                st.reused(),
                if expect_rewind { 5 } else { 0 },
                "{name} rewind"
            );
            assert!(
                rel_diff(&lb, &fresh(m, kind, &B)) <= tol,
                "{name} diverge: {}",
                rel_diff(&lb, &fresh(m, kind, &B))
            );
            assert_eq!((st.pos, st.tokens().len()), (B.len(), B.len()));
            // a shorter prompt that is a prefix of the cached one
            let ls = m.prefill_cached(&mut st, &B[..6]).unwrap();
            assert!(
                rel_diff(&ls, &fresh(m, kind, &B[..6])) <= tol,
                "{name} shorter"
            );
            assert_eq!(st.pos, 6);
            // and decoding goes on normally afterwards
            let next = m.forward(&mut st, 11).unwrap();
            let want = {
                let mut f = m.new_state_with(kind);
                m.forward_batch(&mut f, &B[..6]).unwrap();
                m.forward(&mut f, 11).unwrap()
            };
            assert!(rel_diff(&next, &want) <= tol, "{name} continue");
        }

        #[test]
        fn cached_prefill_equals_fresh_for_dense_models_and_every_kv_precision() {
            let m = tiny(4, 2, false);
            check_prefix("f32", &m, KvKind::F32, 0.0, true);
            check_prefix("f16", &m, KvKind::F16, 1e-5, true);
            check_prefix("q8", &m, KvKind::Q8, 1e-5, true);
            let q = requantise(&model_of(wide_spec(false), false), DType::Q8_0);
            check_prefix("quantised weights", &q, KvKind::F32, 3e-2, true); // activations are quantised per batch shape
        }

        #[test]
        fn cached_prefill_with_sliding_windows_rewinds_only_while_the_ring_has_not_wrapped() {
            let mk = |window: u64| {
                let mut sp = base_spec(4, 2, true);
                sp.sliding_window = window;
                sp.sliding_layers = vec![true, false, true];
                sp.rope_theta_local = 500.0;
                model_of(sp, false)
            };
            check_prefix("window 16", &mk(16), KvKind::F32, 1e-6, true);
            check_prefix(
                "window 3 (wrapped: starts over)",
                &mk(3),
                KvKind::F32,
                1e-6,
                false,
            );
            let mut st = mk(3).new_state();
            mk(3).forward_batch(&mut st, &A).unwrap();
            assert!(!st.can_truncate(5) && st.can_truncate(8));
            assert!(st.truncate(5).is_err());
        }

        pub(crate) fn moe_model() -> Model {
            let (h, n_exp, mf) = (16usize, 4usize, 24usize);
            let mut sp = crate::arch::Spec::dense("llama", 3, h as u64, 4, 2, 4, 32, 20, 32);
            sp.experts = n_exp as u64;
            sp.experts_used = 2;
            sp.support = crate::arch::Support::Moe;
            sp.moe_ffn = mf as u64;
            sp.shared_experts = 1;
            let mut w = random_weights(&sp, 0x1234_5678_9abc_def1, false);
            let mut seed = 0x0bad_cafe_f00d_1234_u64;
            let mut per_layer = vec![];
            for li in 0..3 {
                let l = &mut w.layers[li];
                l.w_gate = Weight::Dense(vec![]);
                l.w_up = Weight::Dense(vec![]);
                l.w_down = Weight::Dense(vec![]);
                let cfg = RouterConfig {
                    hidden: h,
                    experts: n_exp,
                    top_k: 2,
                    scoring: Scoring::Softmax,
                    renormalise: true,
                    routed_scale: 1.0,
                    groups: None,
                    selection_bias: None,
                };
                let router = Router::new(cfg, mat(n_exp * h, &mut seed)).unwrap();
                let sh = vec![ExpertWeights::dense(
                    mat(mf * h, &mut seed),
                    mat(mf * h, &mut seed),
                    mat(h * mf, &mut seed),
                )];
                l.moe = Some(Arc::new(MoeBlock::new(li, router, sh).unwrap()));
                per_layer.push(
                    (0..n_exp)
                        .map(|_| {
                            ExpertWeights::dense(
                                mat(mf * h, &mut seed),
                                mat(mf * h, &mut seed),
                                mat(h * mf, &mut seed),
                            )
                        })
                        .collect(),
                );
            }
            w.experts = Some(Experts {
                provider: Arc::new(InMemoryExperts::new(per_layer)),
                prefetch: true,
            });
            Model::new(sp, w).unwrap()
        }

        #[test]
        fn cached_prefill_equals_fresh_for_mixture_of_experts() {
            check_prefix("moe", &moe_model(), KvKind::F32, 1e-5, true);
        }

        #[test]
        fn forked_states_continue_independently_and_truncation_is_checked() {
            let m = tiny(4, 2, false);
            let mut st = m.new_state();
            m.forward_batch(&mut st, &A).unwrap();
            let mut f = st.fork().unwrap();
            let want = |extra: usize| {
                let mut s = m.new_state();
                m.forward_batch(&mut s, &A).unwrap();
                m.forward(&mut s, extra).unwrap()
            };
            assert_eq!(m.forward(&mut f, 4).unwrap(), want(4));
            assert_eq!(m.forward(&mut st, 9).unwrap(), want(9));
            assert_eq!((f.pos, st.pos), (9, 9));
            assert_ne!(f.tokens(), st.tokens());
            assert!(st.truncate(100).is_err());
            st.truncate(3).unwrap();
            assert_eq!((st.pos, st.tokens()), (3, &A[..3]));
            assert_eq!(
                m.forward_batch(&mut st, &A[3..]).unwrap(),
                fresh(&m, KvKind::F32, &A)
            );
            st.reset();
            assert_eq!((st.pos, st.kv_bytes(), st.tokens().len()), (0, 0, 0));
            assert!(m.prefill_cached(&mut st, &[]).is_err());
        }

        // ---------------------------------------------------------------------- threads
        #[test]
        fn thread_limits_and_the_pool_give_identical_numbers() {
            let (rows, cols) = (1024usize, 512usize);
            let w: Vec<f32> = (0..rows * cols)
                .map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.01)
                .collect();
            let wt = Weight::Dense(w);
            let x: Vec<f32> = (0..cols).map(|i| (i as f32).sin()).collect();
            let (mut a, mut b) = (vec![0.0; rows], vec![0.0; rows]);
            wt.matvec_nt(cols, &x, &mut a, 1).unwrap();
            wt.matvec_nt(cols, &x, &mut b, 0).unwrap();
            assert_eq!(a, b);
            let bt = 6;
            let xs: Vec<f32> = (0..bt * cols)
                .map(|i| ((i * 7) % 13) as f32 - 6.0)
                .collect();
            let (mut oa, mut ob, mut yt) = (vec![0.0; bt * rows], vec![0.0; bt * rows], vec![]);
            wt.matmat_nt(cols, &xs, bt, &mut oa, &mut yt, 1).unwrap();
            wt.matmat_nt(cols, &xs, bt, &mut ob, &mut yt, 0).unwrap();
            assert_eq!(oa, ob);
            wt.matmat_nt(cols, &xs, bt, &mut ob, &mut yt, 2).unwrap();
            assert_eq!(oa, ob);
            // attention over many heads and positions is large enough to split
            let (nh, nkv, hd, n) = (16usize, 4usize, 64usize, 600usize);
            let kd = nkv * hd;
            let plane = |seed: usize| {
                Plane::F32(
                    (0..n * kd)
                        .map(|i| (((i + seed) * 2654435761usize % 1000) as f32 - 500.0) / 500.0)
                        .collect(),
                )
            };
            let kv = LayerKv {
                k: plane(1),
                v: plane(2),
                cap: 0,
                kd,
                kvh: nkv,
            };
            let q: Vec<f32> = (0..nh * hd)
                .map(|i| ((i * 13 % 29) as f32 - 14.0) / 14.0)
                .collect();
            let (mut s1, mut s2) = (vec![0.0; nh * n], vec![0.0; nh * n]);
            let (mut o1, mut o2) = (vec![0.0; nh * hd], vec![0.0; nh * hd]);
            attend(
                &kv,
                kd,
                hd,
                nh,
                nh / nkv,
                0,
                n,
                0.125,
                0.0,
                &q,
                &mut o1,
                &mut s1,
                1,
            );
            attend(
                &kv,
                kd,
                hd,
                nh,
                nh / nkv,
                0,
                n,
                0.125,
                0.0,
                &q,
                &mut o2,
                &mut s2,
                0,
            );
            assert_eq!(o1, o2);
            // a whole model honours the limit and gets the same answer
            let mut m = tiny(4, 2, false);
            let a = run_all(&m).0;
            m.set_threads(1);
            assert_eq!(m.threads(), 1);
            assert_eq!(run_all(&m).0, a);
        }
    }

    #[test]
    fn split_layer_path_equals_forward() {
        let m = tiny(2, 1, false);
        let (mut a, mut b) = (m.new_state(), m.new_state());
        for (i, &t) in [3usize, 7, 1, 19, 5].iter().enumerate() {
            let want = m.forward(&mut a, t).unwrap();
            let mut x = m.embed_token(t).unwrap();
            m.run_layers(&mut b, 0..1, &mut x, i).unwrap();
            m.run_layers(&mut b, 1..3, &mut x, i).unwrap();
            m.finish_token(&mut b);
            assert_eq!(m.logits_from_hidden(&x), want, "token {i}");
        }
        assert_eq!(a.pos, b.pos);
        let mut x = m.embed_token(2).unwrap();
        let bp = b.pos;
        assert!(m.run_layers(&mut b, 0..9, &mut x, bp).is_err());
        assert!(m.run_layers(&mut b, 0..1, &mut x, 0).is_err());
        assert!(m.embed_token(10_000).is_err());
    }
}
