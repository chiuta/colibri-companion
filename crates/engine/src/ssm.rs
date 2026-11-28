//! Recurrent / state-space layers for the non-attention model families. Every block keeps a
//! fixed-size state per sequence instead of a KV cache, so memory and time per token do not grow
//! with the context.
//!
//! Formulations implemented (all f32, written from the public papers and reference descriptions):
//! * [`Mamba1`] – Mamba-1 selective SSM (Gu & Dao 2023), as in `MambaMixer`: `in_proj` → causal
//!   depthwise conv with state → SiLU → `x_proj` → (Δ, B, C) → `dt_proj` + softplus → scan
//!   `h = exp(ΔA)·h + Δ·B·x`, `y = C·h + D·x` → `· silu(z)` → `out_proj`. Falcon-Mamba's weightless
//!   RMS norm on Δ, B and C is optional (`bcdt_rms_eps`).
//! * [`Mamba2`] – Mamba-2 / SSD (Dao & Gu 2024): one `in_proj` giving `[z, xBC, dt]`, conv over `xBC`,
//!   multi-head scan with a scalar `A` per head and `n_groups` shared B/C, gated RMS norm (norm
//!   after the gate by default, `norm_before_gate` for the other order, normalised per group of
//!   `d_inner / n_groups`). `step` is the recurrence; `prefill` is the chunked SSD form
//!   (quadratic inside a chunk, state passed between chunks) and equals repeated `step`.
//! * [`Rwkv6Layer`] – RWKV-6 "Finch" (Peng et al. 2024): data-dependent token shift (ddlerp via LoRA),
//!   data-dependent decay `exp(-exp(w))`, per-head matrix state, bonus `u` ("time_faaaa"),
//!   per-head group norm, SiLU gate; channel mix with squared ReLU and sigmoid receptance.
//! * [`Rwkv7Layer`] – RWKV-7 "Goose" (Peng et al. 2025): generalised delta rule. Per head the state
//!   `S` (value × key) is updated as `S ← S·diag(w) − (S·κ) ⊗ (κ∘a) + v ⊗ k` with decay
//!   `w = exp(-exp(-1/2)·sigmoid(w0 + tanh(xw·w1)·w2))` (the reference's `exp(-exp(-softplus(-z) - 1/2))`),
//!   in-context learning rate `a`, removal key `κ = normalise(k∘k_k)`, `k ← k∘(1 + (a−1)∘k_a)`, value
//!   residual from layer 0 in later layers, output `y = S·r`, group norm, bonus `(Σ r∘k∘r_k)·v`, gate,
//!   output projection; channel mix is `Wv·relu(Wk·lerp)²` without a receptance gate. Formulation
//!   transcribed from the public reference (`rwkv_v7_demo.py` / `RWKV_Tmix_x070`) from memory, not
//!   checked against a checkpoint.
//!
//! GGUF note: llama.cpp stores `ssm_a` already as `A = -exp(A_log)`; pass `a_negated = true` when
//! loading from GGUF.
#![allow(clippy::manual_is_multiple_of)]
use crate::loader::TensorSource;
use crate::ops::{axpy, matvec, silu};
use serde_json::Value;

const MAX_DIM: usize = 1 << 24;
const MAX_STATE: usize = 1 << 30;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsmError {
    Config(String),
    Shape(String),
    Input(String),
    Load(String),
}
impl std::fmt::Display for SsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SsmError::Config(s) => write!(f, "bad recurrent-layer configuration: {s}"),
            SsmError::Shape(s) => write!(f, "weights do not match the configuration: {s}"),
            SsmError::Input(s) => write!(f, "bad input: {s}"),
            SsmError::Load(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for SsmError {}

fn cfg_err<T>(s: impl Into<String>) -> Result<T, SsmError> {
    Err(SsmError::Config(s.into()))
}
fn dim(name: &str, v: usize) -> Result<usize, SsmError> {
    if v == 0 || v > MAX_DIM {
        cfg_err(format!("{name} = {v} is out of range"))
    } else {
        Ok(v)
    }
}
fn mul(a: usize, b: usize) -> Result<usize, SsmError> {
    a.checked_mul(b)
        .filter(|&n| n <= MAX_STATE)
        .ok_or_else(|| SsmError::Config("dimensions are too large".into()))
}
fn expect(name: &str, got: usize, want: usize) -> Result<(), SsmError> {
    if got == want {
        Ok(())
    } else {
        Err(SsmError::Shape(format!(
            "{name}: {got} values, expected {want}"
        )))
    }
}
fn opt_len(name: &str, v: &Option<Vec<f32>>, want: usize) -> Result<(), SsmError> {
    v.as_ref().map_or(Ok(()), |v| expect(name, v.len(), want))
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        crate::simd::rmath::ln64(1.0 + f64::from(crate::simd::rmath::exp(x))) as f32
    }
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + crate::simd::rmath::exp(-x))
}

/// `W · x (+ bias)` with `W` stored row-major as `rows × cols` (the `nn.Linear` layout).
fn linear(w: &[f32], cols: usize, bias: Option<&[f32]>, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; w.len() / cols];
    matvec(w, cols, x, &mut out);
    if let Some(b) = bias {
        out.iter_mut().zip(b).for_each(|(o, b)| *o += b);
    }
    out
}
/// `xᵀ · W` with `W` stored as `x.len() × out_len` (the layout RWKV's LoRA matrices use).
fn vecmat(x: &[f32], w: &[f32], out_len: usize) -> Vec<f32> {
    let mut out = vec![0.0; out_len];
    for (i, xi) in x.iter().enumerate() {
        axpy(*xi, &w[i * out_len..(i + 1) * out_len], &mut out);
    }
    out
}
fn tokens(xs: &[f32], d: usize) -> Result<usize, SsmError> {
    if xs.len() % d != 0 {
        return Err(SsmError::Input(format!(
            "{} values is not a whole number of {d}-wide tokens",
            xs.len()
        )));
    }
    Ok(xs.len() / d)
}

/// Causal depthwise convolution of one token. `win` holds the previous `k-1` inputs of each
/// channel, oldest first; it is advanced by one position.
fn conv_step(
    win: &mut [f32],
    w: &[f32],
    bias: Option<&[f32]>,
    k: usize,
    input: &[f32],
    out: &mut [f32],
) {
    let m = k - 1;
    for (c, o) in out.iter_mut().enumerate().take(input.len()) {
        let wc = &w[c * k..(c + 1) * k];
        let mut acc = bias.map_or(0.0, |b| b[c]) + wc[m] * input[c];
        if m > 0 {
            let ws = &mut win[c * m..(c + 1) * m];
            for j in 0..m {
                acc += wc[j] * ws[j];
            }
            ws.copy_within(1.., 0);
            ws[m - 1] = input[c];
        }
        *o = acc;
    }
}

fn read_f32(src: &dyn TensorSource, name: &str) -> Result<Vec<f32>, SsmError> {
    src.read(name).map_err(|e| SsmError::Load(e.to_string()))
}
fn read_opt(src: &dyn TensorSource, name: &str) -> Result<Option<Vec<f32>>, SsmError> {
    if name.is_empty() || !src.has(name) {
        Ok(None)
    } else {
        read_f32(src, name).map(Some)
    }
}
fn a_to_log(mut a: Vec<f32>, negated: bool) -> Result<Vec<f32>, SsmError> {
    if negated {
        for v in &mut a {
            if v.is_nan() || *v >= 0.0 {
                return Err(SsmError::Load(
                    "ssm_a must hold negative values (A = -exp(A_log))".into(),
                ));
            }
            *v = crate::simd::rmath::ln64(f64::from(-*v)) as f32;
        }
    }
    Ok(a)
}

fn parse_obj(json: &str) -> Result<Value, SsmError> {
    let v: Value = serde_json::from_str(json).map_err(|e| SsmError::Config(e.to_string()))?;
    if v.is_object() {
        Ok(v)
    } else {
        cfg_err("the model description is not a JSON object")
    }
}
fn num(c: &Value, keys: &[&str]) -> Option<usize> {
    keys.iter()
        .find_map(|k| c.get(*k).and_then(Value::as_u64))
        .map(|v| v as usize)
}
fn flt(c: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| c.get(*k).and_then(Value::as_f64))
}
fn flag(c: &Value, key: &str, default: bool) -> bool {
    c.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// Tensor names of one Mamba / Mamba-2 layer. Empty string = not present in that naming.
#[derive(Clone, Debug)]
pub struct MambaNames {
    pub in_proj: String,
    pub in_bias: String,
    pub conv_w: String,
    pub conv_b: String,
    pub x_proj: String,
    pub dt_proj: String,
    pub dt_bias: String,
    pub a: String,
    pub d: String,
    pub out_proj: String,
    pub out_bias: String,
    pub norm: String,
}
impl MambaNames {
    /// Hugging Face `MambaForCausalLM` / `FalconMambaForCausalLM`.
    pub fn hf_mamba1(layer: usize) -> Self {
        let p = format!("backbone.layers.{layer}.mixer.");
        let n = |s: &str| format!("{p}{s}");
        MambaNames {
            in_proj: n("in_proj.weight"),
            in_bias: n("in_proj.bias"),
            conv_w: n("conv1d.weight"),
            conv_b: n("conv1d.bias"),
            x_proj: n("x_proj.weight"),
            dt_proj: n("dt_proj.weight"),
            dt_bias: n("dt_proj.bias"),
            a: n("A_log"),
            d: n("D"),
            out_proj: n("out_proj.weight"),
            out_bias: n("out_proj.bias"),
            norm: String::new(),
        }
    }
    /// Hugging Face `Mamba2ForCausalLM`.
    pub fn hf_mamba2(layer: usize) -> Self {
        let p = format!("backbone.layers.{layer}.mixer.");
        let n = |s: &str| format!("{p}{s}");
        MambaNames {
            in_proj: n("in_proj.weight"),
            in_bias: n("in_proj.bias"),
            conv_w: n("conv1d.weight"),
            conv_b: n("conv1d.bias"),
            x_proj: String::new(),
            dt_proj: String::new(),
            dt_bias: n("dt_bias"),
            a: n("A_log"),
            d: n("D"),
            out_proj: n("out_proj.weight"),
            out_bias: n("out_proj.bias"),
            norm: n("norm.weight"),
        }
    }
    /// llama.cpp GGUF names (`ssm_*`), for both Mamba and Mamba-2 (`ssm_a` is already `-exp(A_log)`).
    pub fn gguf(layer: usize) -> Self {
        let n = |s: &str| format!("blk.{layer}.{s}");
        MambaNames {
            in_proj: n("ssm_in.weight"),
            in_bias: n("ssm_in.bias"),
            conv_w: n("ssm_conv1d.weight"),
            conv_b: n("ssm_conv1d.bias"),
            x_proj: n("ssm_x.weight"),
            dt_proj: n("ssm_dt.weight"),
            dt_bias: n("ssm_dt.bias"),
            a: n("ssm_a"),
            d: n("ssm_d"),
            out_proj: n("ssm_out.weight"),
            out_bias: n("ssm_out.bias"),
            norm: n("ssm_norm.weight"),
        }
    }
}

// =================================================================================== Mamba-1

#[derive(Clone, Debug, PartialEq)]
pub struct Mamba1Config {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub dt_rank: usize,
    pub conv_bias: bool,
    pub bias: bool,
    /// Falcon-Mamba: weightless RMS norm (this epsilon) on Δ, B and C. `None` for plain Mamba.
    pub bcdt_rms_eps: Option<f32>,
}
impl Mamba1Config {
    pub fn from_hf_config(json: &str) -> Result<Self, SsmError> {
        let c = parse_obj(json)?;
        let d_model = num(&c, &["hidden_size", "d_model"])
            .ok_or_else(|| SsmError::Config("no hidden_size".into()))?;
        let expand = num(&c, &["expand"]).unwrap_or(2);
        let d_inner =
            num(&c, &["intermediate_size", "d_inner"]).unwrap_or(d_model.saturating_mul(expand));
        let dt_rank = match c.get("time_step_rank") {
            Some(Value::Number(n)) => n.as_u64().unwrap_or(0) as usize,
            _ => d_model.div_ceil(16),
        };
        let falcon = c.get("model_type").and_then(Value::as_str) == Some("falcon_mamba");
        let cfg = Mamba1Config {
            d_model,
            d_inner,
            d_state: num(&c, &["state_size", "d_state"]).unwrap_or(16),
            d_conv: num(&c, &["conv_kernel", "d_conv"]).unwrap_or(4),
            dt_rank,
            conv_bias: flag(&c, "use_conv_bias", true),
            bias: flag(&c, "use_bias", false),
            bcdt_rms_eps: falcon.then(|| flt(&c, &["mixer_rms_eps"]).unwrap_or(1e-6) as f32),
        };
        cfg.validate()?;
        Ok(cfg)
    }
    pub fn validate(&self) -> Result<(), SsmError> {
        for (n, v) in [
            ("hidden_size", self.d_model),
            ("d_inner", self.d_inner),
            ("d_state", self.d_state),
            ("d_conv", self.d_conv),
            ("dt_rank", self.dt_rank),
        ] {
            dim(n, v)?;
        }
        mul(self.d_inner, self.d_state)?;
        mul(self.d_inner, self.d_conv)?;
        mul(self.d_inner, self.d_model)?;
        mul(self.dt_rank + 2 * self.d_state, self.d_inner)?;
        Ok(())
    }
}

pub struct Mamba1Weights {
    pub in_proj: Vec<f32>,
    pub in_bias: Option<Vec<f32>>,
    pub conv_w: Vec<f32>,
    pub conv_b: Option<Vec<f32>>,
    pub x_proj: Vec<f32>,
    pub dt_proj: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub a_log: Vec<f32>,
    pub d: Vec<f32>,
    pub out_proj: Vec<f32>,
    pub out_bias: Option<Vec<f32>>,
}
pub struct Mamba1 {
    cfg: Mamba1Config,
    w: Mamba1Weights,
    bcdt_w: Option<[Vec<f32>; 3]>,
}

/// Per-sequence state: the conv window and the SSM hidden state.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba1State {
    conv: Vec<f32>,
    h: Vec<f32>,
}
impl Mamba1State {
    pub fn bytes(&self) -> usize {
        4 * (self.conv.len() + self.h.len())
    }
}

impl Mamba1 {
    /// Big matrices and their row lengths, for [`crate::slicegpu`].
    pub fn matrices(&self) -> Vec<(&[f32], usize)> {
        let (dm, di, r) = (self.cfg.d_model, self.cfg.d_inner, self.cfg.dt_rank);
        vec![
            (&self.w.in_proj[..], dm),
            (&self.w.x_proj[..], di),
            (&self.w.dt_proj[..], r),
            (&self.w.out_proj[..], di),
        ]
    }
    pub fn new(cfg: Mamba1Config, w: Mamba1Weights) -> Result<Self, SsmError> {
        cfg.validate()?;
        let (dm, di, n, k, r) = (
            cfg.d_model,
            cfg.d_inner,
            cfg.d_state,
            cfg.d_conv,
            cfg.dt_rank,
        );
        expect("in_proj", w.in_proj.len(), 2 * di * dm)?;
        opt_len("in_proj bias", &w.in_bias, 2 * di)?;
        expect("conv1d weight", w.conv_w.len(), di * k)?;
        opt_len("conv1d bias", &w.conv_b, di)?;
        expect("x_proj", w.x_proj.len(), (r + 2 * n) * di)?;
        expect("dt_proj", w.dt_proj.len(), di * r)?;
        expect("dt_proj bias", w.dt_bias.len(), di)?;
        expect("A_log", w.a_log.len(), di * n)?;
        expect("D", w.d.len(), di)?;
        expect("out_proj", w.out_proj.len(), dm * di)?;
        opt_len("out_proj bias", &w.out_bias, dm)?;
        Ok(Mamba1 {
            cfg,
            w,
            bcdt_w: None,
        })
    }
    /// Give the RMS norms on Δ, B and C (enabled by `bcdt_rms_eps`) learned scales, as Jamba does
    /// (`dt_layernorm`, `b_layernorm`, `c_layernorm`); Falcon-Mamba's norms have none.
    pub fn with_bcdt_weights(
        mut self,
        dt: Vec<f32>,
        b: Vec<f32>,
        c: Vec<f32>,
    ) -> Result<Self, SsmError> {
        if self.cfg.bcdt_rms_eps.is_none() {
            return Err(SsmError::Config(
                "the norms on dt, B and C are not enabled (bcdt_rms_eps)".into(),
            ));
        }
        expect("dt norm weight", dt.len(), self.cfg.dt_rank)?;
        expect("B norm weight", b.len(), self.cfg.d_state)?;
        expect("C norm weight", c.len(), self.cfg.d_state)?;
        self.bcdt_w = Some([dt, b, c]);
        Ok(self)
    }
    pub fn load(
        cfg: Mamba1Config,
        src: &dyn TensorSource,
        n: &MambaNames,
        a_negated: bool,
    ) -> Result<Self, SsmError> {
        let w = Mamba1Weights {
            in_proj: read_f32(src, &n.in_proj)?,
            in_bias: read_opt(src, &n.in_bias)?,
            conv_w: read_f32(src, &n.conv_w)?,
            conv_b: read_opt(src, &n.conv_b)?,
            x_proj: read_f32(src, &n.x_proj)?,
            dt_proj: read_f32(src, &n.dt_proj)?,
            dt_bias: read_f32(src, &n.dt_bias)?,
            a_log: a_to_log(read_f32(src, &n.a)?, a_negated)?,
            d: read_f32(src, &n.d)?,
            out_proj: read_f32(src, &n.out_proj)?,
            out_bias: read_opt(src, &n.out_bias)?,
        };
        Mamba1::new(cfg, w)
    }
    pub fn config(&self) -> &Mamba1Config {
        &self.cfg
    }
    pub fn new_state(&self) -> Mamba1State {
        Mamba1State {
            conv: vec![0.0; self.cfg.d_inner * (self.cfg.d_conv - 1)],
            h: vec![0.0; self.cfg.d_inner * self.cfg.d_state],
        }
    }

    pub fn step(&self, st: &mut Mamba1State, x: &[f32]) -> Result<Vec<f32>, SsmError> {
        let c = &self.cfg;
        if x.len() != c.d_model {
            return Err(SsmError::Input(format!(
                "{} values, expected {}",
                x.len(),
                c.d_model
            )));
        }
        let (di, n, r) = (c.d_inner, c.d_state, c.dt_rank);
        let xz = linear(&self.w.in_proj, c.d_model, self.w.in_bias.as_deref(), x);
        let (xi, z) = xz.split_at(di);
        let mut xc = vec![0.0; di];
        conv_step(
            &mut st.conv,
            &self.w.conv_w,
            self.w.conv_b.as_deref(),
            c.d_conv,
            xi,
            &mut xc,
        );
        xc.iter_mut().for_each(|v| *v = silu(*v));
        let mut dbc = linear(&self.w.x_proj, di, None, &xc);
        if let Some(eps) = c.bcdt_rms_eps {
            let (dt_part, bc) = dbc.split_at_mut(r);
            rms_inplace(dt_part, eps);
            let (b, cc) = bc.split_at_mut(n);
            rms_inplace(b, eps);
            rms_inplace(cc, eps);
            if let Some([wd, wb, wc]) = &self.bcdt_w {
                dt_part.iter_mut().zip(wd).for_each(|(v, w)| *v *= w);
                b.iter_mut().zip(wb).for_each(|(v, w)| *v *= w);
                cc.iter_mut().zip(wc).for_each(|(v, w)| *v *= w);
            }
        }
        let (dt_r, bc) = dbc.split_at(r);
        let (b, cc) = bc.split_at(n);
        let dt = linear(&self.w.dt_proj, r, Some(&self.w.dt_bias), dt_r);
        let mut y = vec![0.0; di];
        for ch in 0..di {
            let dtc = softplus(dt[ch]);
            let h = &mut st.h[ch * n..(ch + 1) * n];
            let a = &self.w.a_log[ch * n..(ch + 1) * n];
            let mut acc = 0.0;
            for s in 0..n {
                h[s] = crate::simd::rmath::exp(dtc * -crate::simd::rmath::exp(a[s])) * h[s]
                    + dtc * b[s] * xc[ch];
                acc += h[s] * cc[s];
            }
            y[ch] = (acc + self.w.d[ch] * xc[ch]) * silu(z[ch]);
        }
        Ok(linear(&self.w.out_proj, di, self.w.out_bias.as_deref(), &y))
    }

    /// Process `xs` (tokens × d_model, flattened). Identical to repeated [`Mamba1::step`]: the
    /// selective scan is input-dependent, so it is run token by token.
    pub fn prefill(&self, st: &mut Mamba1State, xs: &[f32]) -> Result<Vec<f32>, SsmError> {
        let t = tokens(xs, self.cfg.d_model)?;
        let mut out = Vec::with_capacity(t * self.cfg.d_model);
        for x in xs.chunks_exact(self.cfg.d_model) {
            out.extend(self.step(st, x)?);
        }
        Ok(out)
    }
}

fn rms_inplace(v: &mut [f32], eps: f32) {
    if v.is_empty() {
        return;
    }
    let k = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32 + eps).sqrt();
    v.iter_mut().for_each(|x| *x *= k);
}

// =================================================================================== Mamba-2

#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2Config {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub d_state: usize,
    pub n_groups: usize,
    pub d_conv: usize,
    pub conv_bias: bool,
    pub bias: bool,
    pub norm_eps: f32,
    pub norm_before_gate: bool,
    /// Length of the chunks used by [`Mamba2::prefill`].
    pub chunk_size: usize,
}
impl Mamba2Config {
    pub fn d_inner(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.n_groups * self.d_state
    }
    pub fn proj_dim(&self) -> usize {
        self.d_inner() + self.conv_dim() + self.n_heads
    }
    pub fn from_hf_config(json: &str) -> Result<Self, SsmError> {
        let c = parse_obj(json)?;
        let cfg = Mamba2Config {
            d_model: num(&c, &["hidden_size", "d_model"])
                .ok_or_else(|| SsmError::Config("no hidden_size".into()))?,
            n_heads: num(&c, &["num_heads", "nheads"])
                .ok_or_else(|| SsmError::Config("no num_heads".into()))?,
            head_dim: num(&c, &["head_dim", "headdim"]).unwrap_or(64),
            d_state: num(&c, &["state_size", "d_state"]).unwrap_or(128),
            n_groups: num(&c, &["n_groups", "ngroups"]).unwrap_or(1),
            d_conv: num(&c, &["conv_kernel", "d_conv"]).unwrap_or(4),
            conv_bias: flag(&c, "use_conv_bias", true),
            bias: flag(&c, "use_bias", false),
            norm_eps: flt(&c, &["layer_norm_epsilon", "norm_epsilon", "rms_norm_eps"])
                .unwrap_or(1e-5) as f32,
            norm_before_gate: flag(&c, "norm_before_gate", false),
            chunk_size: num(&c, &["chunk_size"]).unwrap_or(64),
        };
        cfg.validate()?;
        Ok(cfg)
    }
    pub fn validate(&self) -> Result<(), SsmError> {
        for (n, v) in [
            ("hidden_size", self.d_model),
            ("num_heads", self.n_heads),
            ("head_dim", self.head_dim),
            ("state_size", self.d_state),
            ("n_groups", self.n_groups),
            ("conv_kernel", self.d_conv),
            ("chunk_size", self.chunk_size),
        ] {
            dim(n, v)?;
        }
        if self.n_heads % self.n_groups != 0 {
            return cfg_err("num_heads is not a multiple of n_groups");
        }
        let di = mul(self.n_heads, self.head_dim)?;
        if di % self.n_groups != 0 {
            return cfg_err("d_inner is not a multiple of n_groups");
        }
        mul(mul(self.n_heads, self.head_dim)?, self.d_state)?;
        mul(self.proj_dim(), self.d_model)?;
        mul(self.conv_dim(), self.d_conv)?;
        Ok(())
    }
}

pub struct Mamba2Weights {
    pub in_proj: Vec<f32>,
    pub in_bias: Option<Vec<f32>>,
    pub conv_w: Vec<f32>,
    pub conv_b: Option<Vec<f32>>,
    pub dt_bias: Vec<f32>,
    pub a_log: Vec<f32>,
    pub d: Vec<f32>,
    pub norm_w: Vec<f32>,
    pub out_proj: Vec<f32>,
    pub out_bias: Option<Vec<f32>>,
}
pub struct Mamba2 {
    cfg: Mamba2Config,
    w: Mamba2Weights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2State {
    conv: Vec<f32>,
    h: Vec<f32>,
}
impl Mamba2State {
    pub fn bytes(&self) -> usize {
        4 * (self.conv.len() + self.h.len())
    }
}

impl Mamba2 {
    /// Big matrices and their row lengths, for [`crate::slicegpu`].
    pub fn matrices(&self) -> Vec<(&[f32], usize)> {
        vec![
            (&self.w.in_proj[..], self.cfg.d_model),
            (&self.w.out_proj[..], self.cfg.d_inner()),
        ]
    }
    pub fn new(cfg: Mamba2Config, w: Mamba2Weights) -> Result<Self, SsmError> {
        cfg.validate()?;
        let (dm, di, nh) = (cfg.d_model, cfg.d_inner(), cfg.n_heads);
        expect("in_proj", w.in_proj.len(), cfg.proj_dim() * dm)?;
        opt_len("in_proj bias", &w.in_bias, cfg.proj_dim())?;
        expect("conv1d weight", w.conv_w.len(), cfg.conv_dim() * cfg.d_conv)?;
        opt_len("conv1d bias", &w.conv_b, cfg.conv_dim())?;
        expect("dt_bias", w.dt_bias.len(), nh)?;
        expect("A_log", w.a_log.len(), nh)?;
        expect("D", w.d.len(), nh)?;
        expect("norm weight", w.norm_w.len(), di)?;
        expect("out_proj", w.out_proj.len(), dm * di)?;
        opt_len("out_proj bias", &w.out_bias, dm)?;
        Ok(Mamba2 { cfg, w })
    }
    pub fn load(
        cfg: Mamba2Config,
        src: &dyn TensorSource,
        n: &MambaNames,
        a_negated: bool,
    ) -> Result<Self, SsmError> {
        let w = Mamba2Weights {
            in_proj: read_f32(src, &n.in_proj)?,
            in_bias: read_opt(src, &n.in_bias)?,
            conv_w: read_f32(src, &n.conv_w)?,
            conv_b: read_opt(src, &n.conv_b)?,
            dt_bias: read_f32(src, &n.dt_bias)?,
            a_log: a_to_log(read_f32(src, &n.a)?, a_negated)?,
            d: read_f32(src, &n.d)?,
            norm_w: read_f32(src, &n.norm)?,
            out_proj: read_f32(src, &n.out_proj)?,
            out_bias: read_opt(src, &n.out_bias)?,
        };
        Mamba2::new(cfg, w)
    }
    pub fn config(&self) -> &Mamba2Config {
        &self.cfg
    }
    pub fn new_state(&self) -> Mamba2State {
        let c = &self.cfg;
        Mamba2State {
            conv: vec![0.0; c.conv_dim() * (c.d_conv - 1)],
            h: vec![0.0; c.n_heads * c.head_dim * c.d_state],
        }
    }

    /// Gated RMS norm (order per config) followed by `out_proj`.
    fn finish(&self, y: &[f32], z: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        let di = c.d_inner();
        let gs = di / c.n_groups;
        let mut t = y.to_vec();
        let norm = |t: &mut [f32]| {
            for g in 0..c.n_groups {
                let s = &mut t[g * gs..(g + 1) * gs];
                let k =
                    1.0 / (s.iter().map(|v| v * v).sum::<f32>() / gs as f32 + c.norm_eps).sqrt();
                for (i, v) in s.iter_mut().enumerate() {
                    *v *= k * self.w.norm_w[g * gs + i];
                }
            }
        };
        if c.norm_before_gate {
            norm(&mut t);
            t.iter_mut().zip(z).for_each(|(v, z)| *v *= silu(*z));
        } else {
            t.iter_mut().zip(z).for_each(|(v, z)| *v *= silu(*z));
            norm(&mut t);
        }
        linear(&self.w.out_proj, di, self.w.out_bias.as_deref(), &t)
    }

    pub fn step(&self, st: &mut Mamba2State, x: &[f32]) -> Result<Vec<f32>, SsmError> {
        let c = &self.cfg;
        if x.len() != c.d_model {
            return Err(SsmError::Input(format!(
                "{} values, expected {}",
                x.len(),
                c.d_model
            )));
        }
        let (di, cd, nh, hd, ds, g) = (
            c.d_inner(),
            c.conv_dim(),
            c.n_heads,
            c.head_dim,
            c.d_state,
            c.n_groups,
        );
        let p = linear(&self.w.in_proj, c.d_model, self.w.in_bias.as_deref(), x);
        let (z, rest) = p.split_at(di);
        let (xbc_in, dt) = rest.split_at(cd);
        let mut xbc = vec![0.0; cd];
        conv_step(
            &mut st.conv,
            &self.w.conv_w,
            self.w.conv_b.as_deref(),
            c.d_conv,
            xbc_in,
            &mut xbc,
        );
        xbc.iter_mut().for_each(|v| *v = silu(*v));
        let (xs, bc) = xbc.split_at(di);
        let (bs, cs) = bc.split_at(g * ds);
        let mut y = vec![0.0; di];
        for h in 0..nh {
            let dth = softplus(dt[h] + self.w.dt_bias[h]);
            let da = crate::simd::rmath::exp(dth * -crate::simd::rmath::exp(self.w.a_log[h]));
            let grp = h / (nh / g);
            let (bg, cg) = (&bs[grp * ds..(grp + 1) * ds], &cs[grp * ds..(grp + 1) * ds]);
            for q in 0..hd {
                let xv = xs[h * hd + q];
                let hs = &mut st.h[(h * hd + q) * ds..(h * hd + q + 1) * ds];
                let mut acc = 0.0;
                for s in 0..ds {
                    hs[s] = da * hs[s] + dth * xv * bg[s];
                    acc += hs[s] * cg[s];
                }
                y[h * hd + q] = acc + self.w.d[h] * xv;
            }
        }
        Ok(self.finish(&y, z))
    }

    /// Chunked SSD prefill: projections and the conv are batched, the scan runs chunk by chunk
    /// (a quadratic attention-like form inside the chunk plus the carried state). Equals repeated
    /// [`Mamba2::step`] and can be continued from any state.
    pub fn prefill(&self, st: &mut Mamba2State, xs_in: &[f32]) -> Result<Vec<f32>, SsmError> {
        let c = &self.cfg;
        let t = tokens(xs_in, c.d_model)?;
        if t == 0 {
            return Ok(vec![]);
        }
        let (di, cd, nh, hd, ds, g) = (
            c.d_inner(),
            c.conv_dim(),
            c.n_heads,
            c.head_dim,
            c.d_state,
            c.n_groups,
        );
        let (mut zs, mut xbc, mut dts) = (vec![0.0; t * di], vec![0.0; t * cd], vec![0.0; t * nh]);
        for (ti, x) in xs_in.chunks_exact(c.d_model).enumerate() {
            let p = linear(&self.w.in_proj, c.d_model, self.w.in_bias.as_deref(), x);
            zs[ti * di..(ti + 1) * di].copy_from_slice(&p[..di]);
            conv_step(
                &mut st.conv,
                &self.w.conv_w,
                self.w.conv_b.as_deref(),
                c.d_conv,
                &p[di..di + cd],
                &mut xbc[ti * cd..(ti + 1) * cd],
            );
            dts[ti * nh..(ti + 1) * nh].copy_from_slice(&p[di + cd..]);
        }
        xbc.iter_mut().for_each(|v| *v = silu(*v));
        let mut ys = vec![0.0; t * di];
        let cs = c.chunk_size.max(1);
        for h in 0..nh {
            let a = -crate::simd::rmath::exp(self.w.a_log[h]);
            let grp = h / (nh / g);
            let dtv: Vec<f32> = (0..t)
                .map(|i| softplus(dts[i * nh + h] + self.w.dt_bias[h]))
                .collect();
            let b_of = |tok: usize| &xbc[tok * cd + di + grp * ds..tok * cd + di + (grp + 1) * ds];
            let c_of = |tok: usize| {
                &xbc[tok * cd + di + g * ds + grp * ds..tok * cd + di + g * ds + (grp + 1) * ds]
            };
            let hs = &mut st.h[h * hd * ds..(h + 1) * hd * ds];
            let mut t0 = 0;
            while t0 < t {
                let l = cs.min(t - t0);
                let mut lc = vec![0f32; l];
                let mut acc = 0.0;
                for (i, v) in lc.iter_mut().enumerate() {
                    acc += dtv[t0 + i] * a;
                    *v = acc;
                }
                let mut cb = vec![0f32; l * l];
                for i in 0..l {
                    for j in 0..=i {
                        let dot: f32 = c_of(t0 + i)
                            .iter()
                            .zip(b_of(t0 + j))
                            .map(|(a, b)| a * b)
                            .sum();
                        cb[i * l + j] = dot * crate::simd::rmath::exp(lc[i] - lc[j]) * dtv[t0 + j];
                    }
                }
                for i in 0..l {
                    let ci = c_of(t0 + i);
                    for q in 0..hd {
                        let mut y = 0.0;
                        for j in 0..=i {
                            y += cb[i * l + j] * xbc[(t0 + j) * cd + h * hd + q];
                        }
                        let carried: f32 = ci
                            .iter()
                            .zip(&hs[q * ds..(q + 1) * ds])
                            .map(|(a, b)| a * b)
                            .sum();
                        y += crate::simd::rmath::exp(lc[i]) * carried
                            + self.w.d[h] * xbc[(t0 + i) * cd + h * hd + q];
                        ys[(t0 + i) * di + h * hd + q] = y;
                    }
                }
                let end = lc[l - 1];
                for q in 0..hd {
                    for s in 0..ds {
                        let mut v = crate::simd::rmath::exp(end) * hs[q * ds + s];
                        for j in 0..l {
                            v += crate::simd::rmath::exp(end - lc[j])
                                * dtv[t0 + j]
                                * xbc[(t0 + j) * cd + h * hd + q]
                                * b_of(t0 + j)[s];
                        }
                        hs[q * ds + s] = v;
                    }
                }
                t0 += l;
            }
        }
        let mut out = Vec::with_capacity(t * c.d_model);
        for ti in 0..t {
            out.extend(self.finish(&ys[ti * di..(ti + 1) * di], &zs[ti * di..(ti + 1) * di]));
        }
        Ok(out)
    }
}

// =================================================================================== RWKV-6

#[derive(Clone, Debug, PartialEq)]
pub struct Rwkv6Config {
    pub hidden: usize,
    pub head_size: usize,
    pub ffn: usize,
    pub ln_eps: f32,
}
impl Rwkv6Config {
    pub fn n_heads(&self) -> usize {
        self.hidden / self.head_size
    }
    pub fn from_hf_config(json: &str) -> Result<Self, SsmError> {
        let c = parse_obj(json)?;
        let hidden = num(&c, &["hidden_size", "n_embd"])
            .ok_or_else(|| SsmError::Config("no hidden_size".into()))?;
        let att = num(&c, &["attention_hidden_size"]).unwrap_or(hidden);
        if att != hidden {
            return cfg_err("attention_hidden_size differs from hidden_size (not supported)");
        }
        let head_size = num(&c, &["head_size", "head_dim"]).ok_or_else(|| {
            SsmError::Config("no head_size: this is not an RWKV-5/6 description".into())
        })?;
        let cfg = Rwkv6Config {
            hidden,
            head_size,
            ffn: num(&c, &["intermediate_size", "dim_ffn"]).unwrap_or(hidden.saturating_mul(7) / 2),
            ln_eps: flt(&c, &["layer_norm_epsilon", "norm_epsilon"]).unwrap_or(1e-5) as f32,
        };
        cfg.validate()?;
        Ok(cfg)
    }
    pub fn validate(&self) -> Result<(), SsmError> {
        for (n, v) in [
            ("hidden_size", self.hidden),
            ("head_size", self.head_size),
            ("intermediate_size", self.ffn),
        ] {
            dim(n, v)?;
        }
        if self.hidden % self.head_size != 0 {
            return cfg_err("hidden_size is not a multiple of head_size");
        }
        mul(self.hidden, self.hidden)?;
        mul(self.hidden, self.ffn)?;
        mul(self.n_heads() * self.head_size, self.head_size)?;
        Ok(())
    }
}

/// Native (BlinkDL) tensor names of one RWKV-6 block.
#[derive(Clone, Debug)]
pub struct Rwkv6Names {
    pub prefix: String,
}
impl Rwkv6Names {
    pub fn native(layer: usize) -> Self {
        Rwkv6Names {
            prefix: format!("blocks.{layer}."),
        }
    }
    fn n(&self, s: &str) -> String {
        format!("{}{s}", self.prefix)
    }
}

pub struct Rwkv6TimeMix {
    pub maa_x: Vec<f32>,
    pub maa: [Vec<f32>; 5], // order w, k, v, r, g
    pub maa_w1: Vec<f32>,
    pub maa_w2: Vec<f32>,
    pub decay: Vec<f32>,
    pub decay_w1: Vec<f32>,
    pub decay_w2: Vec<f32>,
    pub faaaa: Vec<f32>,
    pub wr: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wg: Vec<f32>,
    pub wo: Vec<f32>,
    pub ln_x_w: Vec<f32>,
    pub ln_x_b: Vec<f32>,
}
pub struct Rwkv6ChannelMix {
    pub maa_k: Vec<f32>,
    pub maa_r: Vec<f32>,
    pub wk: Vec<f32>,
    pub wr: Vec<f32>,
    pub wv: Vec<f32>,
}
pub struct Rwkv6Layer {
    cfg: Rwkv6Config,
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    att: Rwkv6TimeMix,
    ffn: Rwkv6ChannelMix,
    mix_extra: usize,
    decay_extra: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rwkv6State {
    shift_att: Vec<f32>,
    wkv: Vec<f32>,
    shift_ffn: Vec<f32>,
}
impl Rwkv6State {
    pub fn bytes(&self) -> usize {
        4 * (self.shift_att.len() + self.wkv.len() + self.shift_ffn.len())
    }
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let k = 1.0 / (var + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - mean) * k * w[i] + b[i])
        .collect()
}

impl Rwkv6Layer {
    /// Big matrices and their row lengths, for [`crate::slicegpu`].
    pub fn matrices(&self) -> Vec<(&[f32], usize)> {
        let (d, f) = (self.cfg.hidden, self.cfg.ffn);
        let (a, c) = (&self.att, &self.ffn);
        vec![
            (&a.wr[..], d),
            (&a.wk[..], d),
            (&a.wv[..], d),
            (&a.wg[..], d),
            (&a.wo[..], d),
            (&c.wk[..], d),
            (&c.wv[..], f),
            (&c.wr[..], d),
        ]
    }
    pub fn new(
        cfg: Rwkv6Config,
        ln: [Vec<f32>; 4],
        att: Rwkv6TimeMix,
        ffn: Rwkv6ChannelMix,
    ) -> Result<Self, SsmError> {
        cfg.validate()?;
        let (d, f, nh, hs) = (cfg.hidden, cfg.ffn, cfg.n_heads(), cfg.head_size);
        for (n, v) in [
            ("ln1 weight", &ln[0]),
            ("ln1 bias", &ln[1]),
            ("ln2 weight", &ln[2]),
            ("ln2 bias", &ln[3]),
        ] {
            expect(n, v.len(), d)?;
        }
        expect("time_maa_x", att.maa_x.len(), d)?;
        for m in &att.maa {
            expect("time_maa_*", m.len(), d)?;
        }
        if att.maa_w1.is_empty() || att.maa_w1.len() % (5 * d) != 0 {
            return Err(SsmError::Shape(
                "time_maa_w1 is not hidden × 5·extra".into(),
            ));
        }
        let mix_extra = att.maa_w1.len() / (5 * d);
        expect("time_maa_w2", att.maa_w2.len(), 5 * mix_extra * d)?;
        expect("time_decay", att.decay.len(), d)?;
        if att.decay_w1.is_empty() || att.decay_w1.len() % d != 0 {
            return Err(SsmError::Shape(
                "time_decay_w1 is not hidden × extra".into(),
            ));
        }
        let decay_extra = att.decay_w1.len() / d;
        expect("time_decay_w2", att.decay_w2.len(), decay_extra * d)?;
        expect("time_faaaa", att.faaaa.len(), nh * hs)?;
        for (n, v) in [
            ("receptance", &att.wr),
            ("key", &att.wk),
            ("value", &att.wv),
            ("gate", &att.wg),
            ("output", &att.wo),
        ] {
            expect(n, v.len(), d * d)?;
        }
        expect("ln_x weight", att.ln_x_w.len(), d)?;
        expect("ln_x bias", att.ln_x_b.len(), d)?;
        expect("ffn time_maa_k", ffn.maa_k.len(), d)?;
        expect("ffn time_maa_r", ffn.maa_r.len(), d)?;
        expect("ffn key", ffn.wk.len(), f * d)?;
        expect("ffn receptance", ffn.wr.len(), d * d)?;
        expect("ffn value", ffn.wv.len(), d * f)?;
        let [ln1_w, ln1_b, ln2_w, ln2_b] = ln;
        Ok(Rwkv6Layer {
            cfg,
            ln1_w,
            ln1_b,
            ln2_w,
            ln2_b,
            att,
            ffn,
            mix_extra,
            decay_extra,
        })
    }

    pub fn load(
        cfg: Rwkv6Config,
        src: &dyn TensorSource,
        names: &Rwkv6Names,
    ) -> Result<Self, SsmError> {
        let r = |s: &str| read_f32(src, &names.n(s));
        let ln = [
            r("ln1.weight")?,
            r("ln1.bias")?,
            r("ln2.weight")?,
            r("ln2.bias")?,
        ];
        let att = Rwkv6TimeMix {
            maa_x: r("att.time_maa_x")?,
            maa: [
                r("att.time_maa_w")?,
                r("att.time_maa_k")?,
                r("att.time_maa_v")?,
                r("att.time_maa_r")?,
                r("att.time_maa_g")?,
            ],
            maa_w1: r("att.time_maa_w1")?,
            maa_w2: r("att.time_maa_w2")?,
            decay: r("att.time_decay")?,
            decay_w1: r("att.time_decay_w1")?,
            decay_w2: r("att.time_decay_w2")?,
            faaaa: r("att.time_faaaa")?,
            wr: r("att.receptance.weight")?,
            wk: r("att.key.weight")?,
            wv: r("att.value.weight")?,
            wg: r("att.gate.weight")?,
            wo: r("att.output.weight")?,
            ln_x_w: r("att.ln_x.weight")?,
            ln_x_b: r("att.ln_x.bias")?,
        };
        let ffn = Rwkv6ChannelMix {
            maa_k: r("ffn.time_maa_k")?,
            maa_r: r("ffn.time_maa_r")?,
            wk: r("ffn.key.weight")?,
            wr: r("ffn.receptance.weight")?,
            wv: r("ffn.value.weight")?,
        };
        Rwkv6Layer::new(cfg, ln, att, ffn)
    }

    pub fn config(&self) -> &Rwkv6Config {
        &self.cfg
    }
    pub fn new_state(&self) -> Rwkv6State {
        let c = &self.cfg;
        Rwkv6State {
            shift_att: vec![0.0; c.hidden],
            wkv: vec![0.0; c.n_heads() * c.head_size * c.head_size],
            shift_ffn: vec![0.0; c.hidden],
        }
    }

    /// Time mix of one token (the input is already layer-normed).
    pub fn time_mix(&self, st: &mut Rwkv6State, x: &[f32]) -> Result<Vec<f32>, SsmError> {
        let (d, hs, nh) = (self.cfg.hidden, self.cfg.head_size, self.cfg.n_heads());
        if x.len() != d {
            return Err(SsmError::Input(format!("{} values, expected {d}", x.len())));
        }
        let a = &self.att;
        let xx: Vec<f32> = st.shift_att.iter().zip(x).map(|(p, x)| p - x).collect();
        let xxx: Vec<f32> = x
            .iter()
            .zip(&xx)
            .zip(&a.maa_x)
            .map(|((x, xx), m)| x + xx * m)
            .collect();
        let e = self.mix_extra;
        let t: Vec<f32> = vecmat(&xxx, &a.maa_w1, 5 * e)
            .into_iter()
            .map(f32::tanh)
            .collect();
        let mut mixed: Vec<Vec<f32>> = Vec::with_capacity(5);
        for j in 0..5 {
            let m = vecmat(
                &t[j * e..(j + 1) * e],
                &a.maa_w2[j * e * d..(j + 1) * e * d],
                d,
            );
            mixed.push(
                (0..d)
                    .map(|i| x[i] + xx[i] * (a.maa[j][i] + m[i]))
                    .collect(),
            );
        }
        let (xw, xk, xv, xr, xg) = (&mixed[0], &mixed[1], &mixed[2], &mixed[3], &mixed[4]);
        let r = linear(&a.wr, d, None, xr);
        let k = linear(&a.wk, d, None, xk);
        let v = linear(&a.wv, d, None, xv);
        let g: Vec<f32> = linear(&a.wg, d, None, xg).into_iter().map(silu).collect();
        let dw: Vec<f32> = vecmat(xw, &a.decay_w1, self.decay_extra)
            .into_iter()
            .map(f32::tanh)
            .collect();
        let w = vecmat(&dw, &a.decay_w2, d);
        let decay: Vec<f32> = (0..d)
            .map(|i| crate::simd::rmath::exp(-crate::simd::rmath::exp(a.decay[i] + w[i])))
            .collect();
        let mut y = vec![0.0; d];
        for h in 0..nh {
            let s = &mut st.wkv[h * hs * hs..(h + 1) * hs * hs]; // s[i * hs + j], key i, value j
            for j in 0..hs {
                let mut acc = 0.0;
                for i in 0..hs {
                    let (ki, vj) = (k[h * hs + i], v[h * hs + j]);
                    acc += r[h * hs + i] * (s[i * hs + j] + a.faaaa[h * hs + i] * ki * vj);
                }
                y[h * hs + j] = acc;
            }
            for i in 0..hs {
                for j in 0..hs {
                    s[i * hs + j] =
                        decay[h * hs + i] * s[i * hs + j] + k[h * hs + i] * v[h * hs + j];
                }
            }
        }
        // per-head group norm, eps scaled by head_size_divisor² = 64 as in the reference
        let eps = 64e-5f32;
        for h in 0..nh {
            let s = &mut y[h * hs..(h + 1) * hs];
            let mean = s.iter().sum::<f32>() / hs as f32;
            let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hs as f32;
            let kk = 1.0 / (var + eps).sqrt();
            for (i, v) in s.iter_mut().enumerate() {
                *v = (*v - mean) * kk * a.ln_x_w[h * hs + i] + a.ln_x_b[h * hs + i];
            }
        }
        y.iter_mut().zip(&g).for_each(|(y, g)| *y *= g);
        st.shift_att.copy_from_slice(x);
        Ok(linear(&a.wo, d, None, &y))
    }

    /// Channel mix of one token (input already layer-normed).
    pub fn channel_mix(&self, st: &mut Rwkv6State, x: &[f32]) -> Result<Vec<f32>, SsmError> {
        let d = self.cfg.hidden;
        if x.len() != d {
            return Err(SsmError::Input(format!("{} values, expected {d}", x.len())));
        }
        let f = &self.ffn;
        let xx: Vec<f32> = st.shift_ffn.iter().zip(x).map(|(p, x)| p - x).collect();
        let xk: Vec<f32> = (0..d).map(|i| x[i] + xx[i] * f.maa_k[i]).collect();
        let xr: Vec<f32> = (0..d).map(|i| x[i] + xx[i] * f.maa_r[i]).collect();
        let k: Vec<f32> = linear(&f.wk, d, None, &xk)
            .into_iter()
            .map(|v| v.max(0.0).powi(2))
            .collect();
        let kv = linear(&f.wv, self.cfg.ffn, None, &k);
        let r = linear(&f.wr, d, None, &xr);
        st.shift_ffn.copy_from_slice(x);
        Ok(kv.iter().zip(&r).map(|(kv, r)| kv * sigmoid(*r)).collect())
    }

    /// A whole block for one token: `x += time_mix(ln1(x)); x += channel_mix(ln2(x))`.
    pub fn step(&self, st: &mut Rwkv6State, x: &[f32]) -> Result<Vec<f32>, SsmError> {
        if x.len() != self.cfg.hidden {
            return Err(SsmError::Input(format!(
                "{} values, expected {}",
                x.len(),
                self.cfg.hidden
            )));
        }
        let mut h = x.to_vec();
        let a = self.time_mix(
            st,
            &layer_norm(&h, &self.ln1_w, &self.ln1_b, self.cfg.ln_eps),
        )?;
        h.iter_mut().zip(&a).for_each(|(h, a)| *h += a);
        let f = self.channel_mix(
            st,
            &layer_norm(&h, &self.ln2_w, &self.ln2_b, self.cfg.ln_eps),
        )?;
        h.iter_mut().zip(&f).for_each(|(h, f)| *h += f);
        Ok(h)
    }

    /// Repeated [`Rwkv6Layer::step`] (no chunked WKV form is implemented).
    pub fn prefill(&self, st: &mut Rwkv6State, xs: &[f32]) -> Result<Vec<f32>, SsmError> {
        let d = self.cfg.hidden;
        tokens(xs, d)?;
        let mut out = Vec::with_capacity(xs.len());
        for x in xs.chunks_exact(d) {
            out.extend(self.step(st, x)?);
        }
        Ok(out)
    }
}

// =================================================================================== RWKV-7

/// Decay constant `exp(-1/2)`: the RWKV-7 reference writes the per-channel decay as
/// `exp(-exp(-softplus(-z) - 1/2))`, which equals `exp(-exp(-1/2) · sigmoid(z))`.
const RWKV7_DECAY_SCALE: f32 = 0.606_530_66;

#[derive(Clone, Debug, PartialEq)]
pub struct Rwkv7Config {
    pub hidden: usize,
    pub head_size: usize,
    pub ffn: usize,
    pub ln_eps: f32,
    /// LoRA ranks of the decay, in-context learning rate, value-residual and gate projections.
    /// `0` means "take it from the weight shapes" when loading.
    pub decay_rank: usize,
    pub a_rank: usize,
    pub v_rank: usize,
    pub gate_rank: usize,
    /// Layers after the first mix in the first layer's value (`value_residual`).
    pub value_residual: bool,
}
impl Rwkv7Config {
    pub fn n_heads(&self) -> usize {
        self.hidden / self.head_size
    }
    /// From a Hugging Face (flash-linear-attention style) `config.json`: `hidden_size`, `head_dim`,
    /// `decay_low_rank_dim`, `a_low_rank_dim`, `v_low_rank_dim`, `gate_low_rank_dim`, `intermediate_size`,
    /// `norm_eps`, `value_residual`. A description without the RWKV-7 low-rank fields is refused.
    pub fn from_hf_config(json: &str) -> Result<Self, SsmError> {
        let c = parse_obj(json)?;
        let hidden = num(&c, &["hidden_size", "n_embd"])
            .ok_or_else(|| SsmError::Config("no hidden_size".into()))?;
        let is7 = c
            .get("model_type")
            .and_then(Value::as_str)
            .is_some_and(|t| t == "rwkv7")
            || num(&c, &["decay_low_rank_dim"]).is_some();
        if !is7 {
            return cfg_err("this is not an RWKV-7 description (no decay_low_rank_dim)");
        }
        let head_size = num(&c, &["head_dim", "head_size"]).unwrap_or(64);
        let cfg = Rwkv7Config {
            hidden,
            head_size,
            ffn: num(&c, &["intermediate_size", "dim_ffn"]).unwrap_or(hidden * 4),
            ln_eps: flt(&c, &["norm_eps", "layer_norm_epsilon", "norm_epsilon"]).unwrap_or(1e-5)
                as f32,
            decay_rank: num(&c, &["decay_low_rank_dim"]).unwrap_or(0),
            a_rank: num(&c, &["a_low_rank_dim"]).unwrap_or(0),
            v_rank: num(&c, &["v_low_rank_dim"]).unwrap_or(0),
            gate_rank: num(&c, &["gate_low_rank_dim"]).unwrap_or(0),
            value_residual: flag(&c, "value_residual", true),
        };
        cfg.validate()?;
        Ok(cfg)
    }
    pub fn validate(&self) -> Result<(), SsmError> {
        for (n, v) in [
            ("hidden_size", self.hidden),
            ("head_dim", self.head_size),
            ("intermediate_size", self.ffn),
        ] {
            dim(n, v)?;
        }
        for (n, v) in [
            ("decay_low_rank_dim", self.decay_rank),
            ("a_low_rank_dim", self.a_rank),
            ("v_low_rank_dim", self.v_rank),
            ("gate_low_rank_dim", self.gate_rank),
        ] {
            if v > MAX_DIM {
                return cfg_err(format!("{n} = {v} is out of range"));
            }
        }
        if self.hidden % self.head_size != 0 {
            return cfg_err("hidden_size is not a multiple of head_dim");
        }
        mul(self.hidden, self.ffn)?;
        mul(self.n_heads() * self.head_size, self.head_size)?;
        for r in [self.decay_rank, self.a_rank, self.v_rank, self.gate_rank] {
            mul(self.hidden, r)?;
        }
        Ok(())
    }
}

/// Tensor names of one RWKV-7 block: the native (BlinkDL) checkpoint names, or the Hugging Face
/// flash-linear-attention names (the latter written from memory, not checked against a checkpoint).
#[derive(Clone, Debug)]
pub struct Rwkv7Names {
    pub prefix: String,
    pub fla: bool,
}
impl Rwkv7Names {
    pub fn native(layer: usize) -> Self {
        Rwkv7Names {
            prefix: format!("blocks.{layer}."),
            fla: false,
        }
    }
    pub fn fla(layer: usize) -> Self {
        Rwkv7Names {
            prefix: format!("model.layers.{layer}."),
            fla: true,
        }
    }
    fn n(&self, s: &str) -> String {
        format!("{}{s}", self.prefix)
    }
}

/// Time-mix weights. The LoRA matrices use the native layout: `w1` is `hidden × rank`, `w2` is
/// `rank × hidden` (applied as `(x · w1) · w2`).
pub struct Rwkv7TimeMix {
    pub x_r: Vec<f32>,
    pub x_w: Vec<f32>,
    pub x_k: Vec<f32>,
    pub x_v: Vec<f32>,
    pub x_a: Vec<f32>,
    pub x_g: Vec<f32>,
    pub w0: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub a0: Vec<f32>,
    pub a1: Vec<f32>,
    pub a2: Vec<f32>,
    /// Value residual (`v0`, `v1`, `v2`); absent in the first layer.
    pub v: Option<[Vec<f32>; 3]>,
    pub g1: Vec<f32>,
    pub g2: Vec<f32>,
    pub k_k: Vec<f32>,
    pub k_a: Vec<f32>,
    pub r_k: Vec<f32>,
    pub wr: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ln_x_w: Vec<f32>,
    pub ln_x_b: Vec<f32>,
}
pub struct Rwkv7ChannelMix {
    pub x_k: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
}

pub struct Rwkv7Layer {
    cfg: Rwkv7Config,
    first: bool,
    ln0: Option<[Vec<f32>; 2]>,
    ln1: [Vec<f32>; 2],
    ln2: [Vec<f32>; 2],
    att: Rwkv7TimeMix,
    ffn: Rwkv7ChannelMix,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rwkv7State {
    shift_att: Vec<f32>,
    wkv: Vec<f32>,
    shift_ffn: Vec<f32>,
}
impl Rwkv7State {
    pub fn bytes(&self) -> usize {
        4 * (self.shift_att.len() + self.wkv.len() + self.shift_ffn.len())
    }
}

/// A LoRA pair in the native layout, with its optional bias (`w0`, `a0`, `v0`).
type Lora = (Vec<f32>, Vec<f32>, Option<Vec<f32>>);

fn transpose(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut o = vec![0.0; w.len()];
    for r in 0..rows {
        for c in 0..cols {
            o[c * rows + r] = w[r * cols + c];
        }
    }
    o
}

impl Rwkv7Layer {
    /// Big matrices and their row lengths, for [`crate::slicegpu`].
    pub fn matrices(&self) -> Vec<(&[f32], usize)> {
        let (d, f) = (self.cfg.hidden, self.cfg.ffn);
        let (a, c) = (&self.att, &self.ffn);
        vec![
            (&a.wr[..], d),
            (&a.wk[..], d),
            (&a.wv[..], d),
            (&a.wo[..], d),
            (&c.wk[..], d),
            (&c.wv[..], f),
        ]
    }
    /// `first` marks layer 0: it stores its value vector as `v_first` and has no value residual.
    /// `ln0` is the extra layer norm in front of layer 0 (embedding norm), `ln` is `[ln1 w, ln1 b, ln2 w, ln2 b]`.
    pub fn new(
        mut cfg: Rwkv7Config,
        first: bool,
        ln0: Option<[Vec<f32>; 2]>,
        ln: [Vec<f32>; 4],
        att: Rwkv7TimeMix,
        ffn: Rwkv7ChannelMix,
    ) -> Result<Self, SsmError> {
        cfg.validate()?;
        let (d, f, nh, hs) = (cfg.hidden, cfg.ffn, cfg.n_heads(), cfg.head_size);
        let rank = |name: &str, w1: usize, w2: usize, want: usize| -> Result<usize, SsmError> {
            if w1 == 0 || w1 % d != 0 {
                return Err(SsmError::Shape(format!("{name}1 is not hidden × rank")));
            }
            let r = w1 / d;
            expect(&format!("{name}2"), w2, r * d)?;
            if want != 0 && want != r {
                return Err(SsmError::Shape(format!(
                    "{name}: rank {r} but the configuration says {want}"
                )));
            }
            Ok(r)
        };
        cfg.decay_rank = rank("w", att.w1.len(), att.w2.len(), cfg.decay_rank)?;
        cfg.a_rank = rank("a", att.a1.len(), att.a2.len(), cfg.a_rank)?;
        cfg.gate_rank = rank("g", att.g1.len(), att.g2.len(), cfg.gate_rank)?;
        if let Some([v0, v1, v2]) = &att.v {
            expect("v0", v0.len(), d)?;
            cfg.v_rank = rank("v", v1.len(), v2.len(), cfg.v_rank)?;
        }
        // a later layer without the value-residual tensors is tolerated: the residual is simply off
        if first && att.v.is_some() {
            return Err(SsmError::Shape(
                "the first layer has no value residual".into(),
            ));
        }
        for (n, v) in [
            ("ln1 weight", &ln[0]),
            ("ln1 bias", &ln[1]),
            ("ln2 weight", &ln[2]),
            ("ln2 bias", &ln[3]),
        ] {
            expect(n, v.len(), d)?;
        }
        if let Some([w, b]) = &ln0 {
            expect("ln0 weight", w.len(), d)?;
            expect("ln0 bias", b.len(), d)?;
        }
        for (n, v) in [
            ("x_r", &att.x_r),
            ("x_w", &att.x_w),
            ("x_k", &att.x_k),
            ("x_v", &att.x_v),
            ("x_a", &att.x_a),
            ("x_g", &att.x_g),
            ("w0", &att.w0),
            ("a0", &att.a0),
            ("k_k", &att.k_k),
            ("k_a", &att.k_a),
            ("ln_x weight", &att.ln_x_w),
            ("ln_x bias", &att.ln_x_b),
        ] {
            expect(n, v.len(), d)?;
        }
        expect("r_k", att.r_k.len(), nh * hs)?;
        for (n, v) in [
            ("receptance", &att.wr),
            ("key", &att.wk),
            ("value", &att.wv),
            ("output", &att.wo),
        ] {
            expect(n, v.len(), d * d)?;
        }
        expect("ffn x_k", ffn.x_k.len(), d)?;
        expect("ffn key", ffn.wk.len(), f * d)?;
        expect("ffn value", ffn.wv.len(), d * f)?;
        let [l1w, l1b, l2w, l2b] = ln;
        Ok(Rwkv7Layer {
            cfg,
            first,
            ln0,
            ln1: [l1w, l1b],
            ln2: [l2w, l2b],
            att,
            ffn,
        })
    }

    /// Load one layer; the ranks are taken from the weight shapes. `layer == 0` is the first layer.
    pub fn load(
        cfg: Rwkv7Config,
        src: &dyn TensorSource,
        names: &Rwkv7Names,
        layer: usize,
    ) -> Result<Self, SsmError> {
        let first = layer == 0;
        let r = |s: &str| read_f32(src, &names.n(s));
        let d = cfg.hidden;
        let (ln0, ln, att, ffn);
        if !names.fla {
            ln0 = if first && src.has(&names.n("ln0.weight")) {
                Some([r("ln0.weight")?, r("ln0.bias")?])
            } else {
                None
            };
            ln = [
                r("ln1.weight")?,
                r("ln1.bias")?,
                r("ln2.weight")?,
                r("ln2.bias")?,
            ];
            let v = if !first && cfg.value_residual && src.has(&names.n("att.v0")) {
                Some([r("att.v0")?, r("att.v1")?, r("att.v2")?])
            } else {
                None
            };
            att = Rwkv7TimeMix {
                x_r: r("att.x_r")?,
                x_w: r("att.x_w")?,
                x_k: r("att.x_k")?,
                x_v: r("att.x_v")?,
                x_a: r("att.x_a")?,
                x_g: r("att.x_g")?,
                w0: r("att.w0")?,
                w1: r("att.w1")?,
                w2: r("att.w2")?,
                a0: r("att.a0")?,
                a1: r("att.a1")?,
                a2: r("att.a2")?,
                v,
                g1: r("att.g1")?,
                g2: r("att.g2")?,
                k_k: r("att.k_k")?,
                k_a: r("att.k_a")?,
                r_k: r("att.r_k")?,
                wr: r("att.receptance.weight")?,
                wk: r("att.key.weight")?,
                wv: r("att.value.weight")?,
                wo: r("att.output.weight")?,
                ln_x_w: r("att.ln_x.weight")?,
                ln_x_b: r("att.ln_x.bias")?,
            };
            ffn = Rwkv7ChannelMix {
                x_k: r("ffn.x_k")?,
                wk: r("ffn.key.weight")?,
                wv: r("ffn.value.weight")?,
            };
        } else {
            // flash-linear-attention layout: LoRA stored as nn.Linear pairs (down: rank × hidden, up: hidden × rank)
            let lora = |nm: &str, bias: bool| -> Result<Lora, SsmError> {
                let down = r(&format!("attn.{nm}.lora.0.weight"))?;
                let up = r(&format!("attn.{nm}.lora.2.weight"))?;
                if down.is_empty() || down.len() % d != 0 {
                    return Err(SsmError::Shape(format!(
                        "{nm} down projection is not rank × hidden"
                    )));
                }
                let rk = down.len() / d;
                let b = if bias {
                    Some(r(&format!("attn.{nm}.lora.2.bias"))?)
                } else {
                    None
                };
                if up.len() != d * rk {
                    return Err(SsmError::Shape(format!(
                        "{nm} up projection is not hidden × rank"
                    )));
                }
                Ok((transpose(&down, rk, d), transpose(&up, d, rk), b))
            };
            ln0 = if first && src.has(&names.n("pre_norm.weight")) {
                Some([r("pre_norm.weight")?, r("pre_norm.bias")?])
            } else {
                None
            };
            ln = [
                r("attn_norm.weight")?,
                r("attn_norm.bias")?,
                r("ffn_norm.weight")?,
                r("ffn_norm.bias")?,
            ];
            let (w1, w2, w0) = lora("w_lora", true)?;
            let (a1, a2, a0) = lora("a_lora", true)?;
            let (g1, g2, _) = lora("g_lora", false)?;
            let v =
                if !first && cfg.value_residual && src.has(&names.n("attn.v_lora.lora.0.weight")) {
                    let (v1, v2, v0) = lora("v_lora", true)?;
                    Some([v0.unwrap_or_default(), v1, v2])
                } else {
                    None
                };
            att = Rwkv7TimeMix {
                x_r: r("attn.x_r")?,
                x_w: r("attn.x_w")?,
                x_k: r("attn.x_k")?,
                x_v: r("attn.x_v")?,
                x_a: r("attn.x_a")?,
                x_g: r("attn.x_g")?,
                w0: w0.unwrap_or_default(),
                w1,
                w2,
                a0: a0.unwrap_or_default(),
                a1,
                a2,
                v,
                g1,
                g2,
                k_k: r("attn.k_k")?,
                k_a: r("attn.k_a")?,
                r_k: r("attn.r_k")?,
                wr: r("attn.r_proj.weight")?,
                wk: r("attn.k_proj.weight")?,
                wv: r("attn.v_proj.weight")?,
                wo: r("attn.o_proj.weight")?,
                ln_x_w: r("attn.g_norm.weight")?,
                ln_x_b: r("attn.g_norm.bias")?,
            };
            ffn = Rwkv7ChannelMix {
                x_k: r("ffn.x_k")?,
                wk: r("ffn.key.weight")?,
                wv: r("ffn.value.weight")?,
            };
        }
        Rwkv7Layer::new(cfg, first, ln0, ln, att, ffn)
    }

    pub fn config(&self) -> &Rwkv7Config {
        &self.cfg
    }
    pub fn is_first(&self) -> bool {
        self.first
    }
    pub fn new_state(&self) -> Rwkv7State {
        let c = &self.cfg;
        Rwkv7State {
            shift_att: vec![0.0; c.hidden],
            wkv: vec![0.0; c.n_heads() * c.head_size * c.head_size],
            shift_ffn: vec![0.0; c.hidden],
        }
    }

    /// Time mix of one token (input already layer-normed). `v_first` (hidden values) is written by
    /// the first layer and read by the others when the value residual is on.
    ///
    /// Per head, with state `S` (value index `i`, key index `j`):
    /// `S ← S·diag(w) − (S·κ) ⊗ (κ∘a) + v ⊗ k`, `y = S·r`, then group norm, plus `(Σ r∘k∘r_k)·v`, times the gate.
    pub fn time_mix(
        &self,
        st: &mut Rwkv7State,
        x: &[f32],
        v_first: &mut [f32],
    ) -> Result<Vec<f32>, SsmError> {
        let (d, hs, nh) = (self.cfg.hidden, self.cfg.head_size, self.cfg.n_heads());
        if x.len() != d {
            return Err(SsmError::Input(format!("{} values, expected {d}", x.len())));
        }
        if v_first.len() != d {
            return Err(SsmError::Input(format!(
                "v_first has {} values, expected {d}",
                v_first.len()
            )));
        }
        let a = &self.att;
        let xx: Vec<f32> = st.shift_att.iter().zip(x).map(|(p, x)| p - x).collect();
        let mix = |m: &[f32]| -> Vec<f32> { (0..d).map(|i| x[i] + xx[i] * m[i]).collect() };
        let (xr, xw, xk, xv, xa, xg) = (
            mix(&a.x_r),
            mix(&a.x_w),
            mix(&a.x_k),
            mix(&a.x_v),
            mix(&a.x_a),
            mix(&a.x_g),
        );
        let r = linear(&a.wr, d, None, &xr);
        let mut k = linear(&a.wk, d, None, &xk);
        let mut v = linear(&a.wv, d, None, &xv);
        // decay: w = exp(-exp(-1/2)·sigmoid(w0 + tanh(xw·w1)·w2))
        let t: Vec<f32> = vecmat(&xw, &a.w1, self.cfg.decay_rank)
            .into_iter()
            .map(f32::tanh)
            .collect();
        let wl = vecmat(&t, &a.w2, d);
        let decay: Vec<f32> = (0..d)
            .map(|i| crate::simd::rmath::exp(-RWKV7_DECAY_SCALE * sigmoid(a.w0[i] + wl[i])))
            .collect();
        // value residual
        if self.first {
            v_first.copy_from_slice(&v);
        } else if let Some([v0, v1, v2]) = &a.v {
            let l = vecmat(&vecmat(&xv, v1, self.cfg.v_rank), v2, d);
            for i in 0..d {
                v[i] += (v_first[i] - v[i]) * sigmoid(v0[i] + l[i]);
            }
        }
        // in-context learning rate and gate
        let al = vecmat(&vecmat(&xa, &a.a1, self.cfg.a_rank), &a.a2, d);
        let alpha: Vec<f32> = (0..d).map(|i| sigmoid(a.a0[i] + al[i])).collect();
        let gt: Vec<f32> = vecmat(&xg, &a.g1, self.cfg.gate_rank)
            .into_iter()
            .map(sigmoid)
            .collect();
        let g = vecmat(&gt, &a.g2, d);
        // removal key κ = normalise(k∘k_k) per head, then k ← k∘(1 + (a−1)∘k_a)
        let mut kk: Vec<f32> = (0..d).map(|i| k[i] * a.k_k[i]).collect();
        for h in kk.chunks_mut(hs) {
            let n = h.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            h.iter_mut().for_each(|v| *v /= n);
        }
        for i in 0..d {
            k[i] *= 1.0 + (alpha[i] - 1.0) * a.k_a[i];
        }
        let mut y = vec![0.0; d];
        for h in 0..nh {
            let o = h * hs;
            let s = &mut st.wkv[h * hs * hs..(h + 1) * hs * hs];
            for i in 0..hs {
                let row = &mut s[i * hs..(i + 1) * hs];
                let sk: f32 = row.iter().zip(&kk[o..o + hs]).map(|(s, kk)| s * kk).sum();
                let mut acc = 0.0;
                for j in 0..hs {
                    row[j] = row[j] * decay[o + j] - sk * (kk[o + j] * alpha[o + j])
                        + v[o + i] * k[o + j];
                    acc += row[j] * r[o + j];
                }
                y[o + i] = acc;
            }
        }
        // per-head group norm (eps 64e-5), then the bonus term
        let eps = 64e-5f32;
        for h in 0..nh {
            let o = h * hs;
            let s = &mut y[o..o + hs];
            let mean = s.iter().sum::<f32>() / hs as f32;
            let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hs as f32;
            let kn = 1.0 / (var + eps).sqrt();
            for (i, v) in s.iter_mut().enumerate() {
                *v = (*v - mean) * kn * a.ln_x_w[o + i] + a.ln_x_b[o + i];
            }
            let bonus: f32 = (0..hs).map(|j| r[o + j] * k[o + j] * a.r_k[o + j]).sum();
            for i in 0..hs {
                s[i] += bonus * v[o + i];
            }
        }
        y.iter_mut().zip(&g).for_each(|(y, g)| *y *= g);
        st.shift_att.copy_from_slice(x);
        Ok(linear(&a.wo, d, None, &y))
    }

    /// Channel mix of one token (input already layer-normed): `Wv · relu(Wk · lerp(x, shift))²`, no gate.
    pub fn channel_mix(&self, st: &mut Rwkv7State, x: &[f32]) -> Result<Vec<f32>, SsmError> {
        let d = self.cfg.hidden;
        if x.len() != d {
            return Err(SsmError::Input(format!("{} values, expected {d}", x.len())));
        }
        let f = &self.ffn;
        let xk: Vec<f32> = (0..d)
            .map(|i| x[i] + (st.shift_ffn[i] - x[i]) * f.x_k[i])
            .collect();
        let k: Vec<f32> = linear(&f.wk, d, None, &xk)
            .into_iter()
            .map(|v| v.max(0.0).powi(2))
            .collect();
        st.shift_ffn.copy_from_slice(x);
        Ok(linear(&f.wv, self.cfg.ffn, None, &k))
    }

    /// A whole block for one token: `x = ln0(x)` (first layer, if present); `x += time_mix(ln1(x))`; `x += channel_mix(ln2(x))`.
    pub fn step(
        &self,
        st: &mut Rwkv7State,
        x: &[f32],
        v_first: &mut [f32],
    ) -> Result<Vec<f32>, SsmError> {
        let d = self.cfg.hidden;
        if x.len() != d {
            return Err(SsmError::Input(format!("{} values, expected {d}", x.len())));
        }
        let mut h = match &self.ln0 {
            Some([w, b]) => layer_norm(x, w, b, self.cfg.ln_eps),
            None => x.to_vec(),
        };
        let a = self.time_mix(
            st,
            &layer_norm(&h, &self.ln1[0], &self.ln1[1], self.cfg.ln_eps),
            v_first,
        )?;
        h.iter_mut().zip(&a).for_each(|(h, a)| *h += a);
        let f = self.channel_mix(
            st,
            &layer_norm(&h, &self.ln2[0], &self.ln2[1], self.cfg.ln_eps),
        )?;
        h.iter_mut().zip(&f).for_each(|(h, f)| *h += f);
        Ok(h)
    }

    /// Repeated [`Rwkv7Layer::step`]; `v_first` holds one hidden-sized vector per token (`tokens × hidden`).
    pub fn prefill(
        &self,
        st: &mut Rwkv7State,
        xs: &[f32],
        v_first: &mut [f32],
    ) -> Result<Vec<f32>, SsmError> {
        let d = self.cfg.hidden;
        let n = tokens(xs, d)?;
        if v_first.len() != xs.len() {
            return Err(SsmError::Input(format!(
                "v_first has {} values, expected {} ({n} tokens)",
                v_first.len(),
                xs.len()
            )));
        }
        let mut out = Vec::with_capacity(xs.len());
        for (x, vf) in xs.chunks_exact(d).zip(v_first.chunks_exact_mut(d)) {
            out.extend(self.step(st, x, vf)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::loader::LoadError;
    use std::collections::HashMap;

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }
        fn v(&mut self, n: usize, s: f32) -> Vec<f32> {
            (0..n).map(|_| self.f() * s).collect()
        }
        fn pos(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.f() + 0.5).collect()
        }
    }
    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "at {i}: {x} vs {y}");
        }
    }
    fn inputs(r: &mut Rng, t: usize, d: usize) -> Vec<f32> {
        r.v(t * d, 2.0)
    }

    // ---------------------------------------------------------------- Mamba-1
    fn m1_cfg(k: usize, falcon: bool) -> Mamba1Config {
        Mamba1Config {
            d_model: 6,
            d_inner: 8,
            d_state: 3,
            d_conv: k,
            dt_rank: 2,
            conv_bias: true,
            bias: false,
            bcdt_rms_eps: falcon.then_some(1e-5),
        }
    }
    fn m1(k: usize, falcon: bool, seed: u64) -> Mamba1 {
        let mut r = Rng(seed);
        let c = m1_cfg(k, falcon);
        let (dm, di, n, rk) = (c.d_model, c.d_inner, c.d_state, c.dt_rank);
        let w = Mamba1Weights {
            in_proj: r.v(2 * di * dm, 0.8),
            in_bias: None,
            conv_w: r.v(di * k, 1.0),
            conv_b: Some(r.v(di, 0.5)),
            x_proj: r.v((rk + 2 * n) * di, 0.8),
            dt_proj: r.v(di * rk, 1.0),
            dt_bias: r.v(di, 1.0),
            a_log: (0..di * n)
                .map(|i| ((i % n) as f32 + 1.0).ln() + r.f() * 0.2)
                .collect(),
            d: r.pos(di),
            out_proj: r.v(dm * di, 0.8),
            out_bias: None,
        };
        Mamba1::new(c, w).unwrap()
    }

    /// Whole-sequence reference in f64: full-length padded convolution and an unrolled scan.
    fn naive_m1(m: &Mamba1, xs: &[f32]) -> Vec<f32> {
        let (c, w) = (&m.cfg, &m.w);
        let (dm, di, n, k, rk) = (c.d_model, c.d_inner, c.d_state, c.d_conv, c.dt_rank);
        let t = xs.len() / dm;
        let sp = |x: f64| (1.0 + x.exp()).ln();
        let si = |x: f64| x / (1.0 + (-x).exp());
        let lin = |w: &[f32], cols: usize, x: &[f64]| -> Vec<f64> {
            w.chunks(cols)
                .map(|r| r.iter().zip(x).map(|(a, b)| *a as f64 * b).sum())
                .collect()
        };
        let mut xi = vec![vec![0f64; di]; t];
        let mut zs = vec![vec![0f64; di]; t];
        for ti in 0..t {
            let x: Vec<f64> = xs[ti * dm..(ti + 1) * dm]
                .iter()
                .map(|v| *v as f64)
                .collect();
            let p = lin(&w.in_proj, dm, &x);
            xi[ti].copy_from_slice(&p[..di]);
            zs[ti].copy_from_slice(&p[di..]);
        }
        let mut h = vec![0f64; di * n];
        let mut out = vec![];
        for ti in 0..t {
            let mut xc = vec![0f64; di];
            for ch in 0..di {
                let mut s = w.conv_b.as_ref().unwrap()[ch] as f64;
                for j in 0..k {
                    let src = ti as isize - (k as isize - 1) + j as isize;
                    if src >= 0 {
                        s += w.conv_w[ch * k + j] as f64 * xi[src as usize][ch];
                    }
                }
                xc[ch] = si(s);
            }
            let mut dbc = lin(&w.x_proj, di, &xc);
            if let Some(eps) = c.bcdt_rms_eps {
                for rng in [0..rk, rk..rk + n, rk + n..rk + 2 * n] {
                    let s = &mut dbc[rng];
                    let k = 1.0
                        / (s.iter().map(|v| v * v).sum::<f64>() / s.len() as f64 + eps as f64)
                            .sqrt();
                    s.iter_mut().for_each(|v| *v *= k);
                }
            }
            let dt = lin(&w.dt_proj, rk, &dbc[..rk]);
            let mut y = vec![0f64; di];
            for ch in 0..di {
                let d = sp(dt[ch] + w.dt_bias[ch] as f64);
                let mut acc = 0.0;
                for s in 0..n {
                    let a = -(w.a_log[ch * n + s] as f64).exp();
                    h[ch * n + s] = (d * a).exp() * h[ch * n + s] + d * dbc[rk + s] * xc[ch];
                    acc += h[ch * n + s] * dbc[rk + n + s];
                }
                y[ch] = (acc + w.d[ch] as f64 * xc[ch]) * si(zs[ti][ch]);
            }
            out.extend(lin(&w.out_proj, di, &y).into_iter().map(|v| v as f32));
        }
        out
    }

    #[test]
    fn mamba1_matches_the_naive_reference_for_several_kernel_sizes() {
        for (k, falcon) in [(4, false), (1, false), (2, false), (3, false), (4, true)] {
            let m = m1(k, falcon, 11 + k as u64);
            let xs = inputs(&mut Rng(5), 30, 6); // 30 steps ≫ kernel size: the window wraps many times
            let mut st = m.new_state();
            close(&m.prefill(&mut st, &xs).unwrap(), &naive_m1(&m, &xs), 2e-4);
        }
    }

    #[test]
    fn mamba1_is_causal_and_prefill_can_be_split() {
        let m = m1(4, false, 3);
        let mut r = Rng(9);
        let xs = inputs(&mut r, 12, 6);
        let mut ys = xs.clone();
        for v in &mut ys[8 * 6..] {
            *v = r.f() * 9.0;
        }
        let (mut a, mut b) = (m.new_state(), m.new_state());
        let (oa, ob) = (
            m.prefill(&mut a, &xs).unwrap(),
            m.prefill(&mut b, &ys).unwrap(),
        );
        assert_eq!(oa[..8 * 6], ob[..8 * 6]);
        assert_ne!(oa[8 * 6..], ob[8 * 6..]);
        let (mut s1, mut s2) = (m.new_state(), m.new_state());
        let whole = m.prefill(&mut s1, &xs).unwrap();
        let mut parts = m.prefill(&mut s2, &xs[..5 * 6]).unwrap();
        parts.extend(m.prefill(&mut s2, &xs[5 * 6..]).unwrap());
        assert_eq!(whole, parts);
        assert_eq!(s1, s2);
    }

    #[test]
    fn mamba1_state_is_constant_size_and_stable_over_ten_thousand_steps() {
        let m = m1(4, false, 21);
        let mut st = m.new_state();
        let bytes = st.bytes();
        assert_eq!(bytes, 4 * (8 * 3 + 8 * 3));
        let mut r = Rng(77);
        let mut worst: f32 = 0.0;
        for _ in 0..10_000 {
            let y = m.step(&mut st, &r.v(6, 4.0)).unwrap();
            assert!(y.iter().all(|v| v.is_finite()));
            worst = y.iter().fold(worst, |a, v| a.max(v.abs()));
        }
        assert!(worst < 1e3, "{worst}");
        assert_eq!(st.bytes(), bytes);
    }

    // ---------------------------------------------------------------- Mamba-2
    fn m2_cfg(chunk: usize, before: bool) -> Mamba2Config {
        Mamba2Config {
            d_model: 6,
            n_heads: 4,
            head_dim: 3,
            d_state: 3,
            n_groups: 2,
            d_conv: 4,
            conv_bias: true,
            bias: false,
            norm_eps: 1e-5,
            norm_before_gate: before,
            chunk_size: chunk,
        }
    }
    fn m2(chunk: usize, before: bool, seed: u64) -> Mamba2 {
        let mut r = Rng(seed);
        let c = m2_cfg(chunk, before);
        let (dm, di, nh) = (c.d_model, c.d_inner(), c.n_heads);
        let w = Mamba2Weights {
            in_proj: r.v(c.proj_dim() * dm, 0.8),
            in_bias: None,
            conv_w: r.v(c.conv_dim() * c.d_conv, 1.0),
            conv_b: Some(r.v(c.conv_dim(), 0.5)),
            dt_bias: r.v(nh, 1.0),
            a_log: (0..nh)
                .map(|i| (i as f32 + 1.0).ln() + r.f() * 0.2)
                .collect(),
            d: r.pos(nh),
            norm_w: r.pos(di),
            out_proj: r.v(dm * di, 0.8),
            out_bias: None,
        };
        Mamba2::new(c, w).unwrap()
    }

    fn naive_m2(m: &Mamba2, xs: &[f32]) -> Vec<f32> {
        let (c, w) = (&m.cfg, &m.w);
        let (dm, di, cd, nh, hd, ds, g, k) = (
            c.d_model,
            c.d_inner(),
            c.conv_dim(),
            c.n_heads,
            c.head_dim,
            c.d_state,
            c.n_groups,
            c.d_conv,
        );
        let t = xs.len() / dm;
        let sp = |x: f64| (1.0 + x.exp()).ln();
        let si = |x: f64| x / (1.0 + (-x).exp());
        let lin = |w: &[f32], cols: usize, x: &[f64]| -> Vec<f64> {
            w.chunks(cols)
                .map(|r| r.iter().zip(x).map(|(a, b)| *a as f64 * b).sum())
                .collect()
        };
        let proj: Vec<Vec<f64>> = (0..t)
            .map(|ti| {
                lin(
                    &w.in_proj,
                    dm,
                    &xs[ti * dm..(ti + 1) * dm]
                        .iter()
                        .map(|v| *v as f64)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let mut h = vec![0f64; nh * hd * ds];
        let mut out = vec![];
        for ti in 0..t {
            let mut xbc = vec![0f64; cd];
            for ch in 0..cd {
                let mut s = w.conv_b.as_ref().unwrap()[ch] as f64;
                for j in 0..k {
                    let src = ti as isize - (k as isize - 1) + j as isize;
                    if src >= 0 {
                        s += w.conv_w[ch * k + j] as f64 * proj[src as usize][di + ch];
                    }
                }
                xbc[ch] = si(s);
            }
            let z = &proj[ti][..di];
            let mut y = vec![0f64; di];
            for hh in 0..nh {
                let dth = sp(proj[ti][di + cd + hh] + w.dt_bias[hh] as f64);
                let da = (dth * -(w.a_log[hh] as f64).exp()).exp();
                let grp = hh / (nh / g);
                for q in 0..hd {
                    let xv = xbc[hh * hd + q];
                    let mut acc = 0.0;
                    for s in 0..ds {
                        let (b, cc) = (xbc[di + grp * ds + s], xbc[di + g * ds + grp * ds + s]);
                        let hi = (hh * hd + q) * ds + s;
                        h[hi] = da * h[hi] + dth * xv * b;
                        acc += h[hi] * cc;
                    }
                    y[hh * hd + q] = acc + w.d[hh] as f64 * xv;
                }
            }
            let gs = di / g;
            let norm = |t: &mut Vec<f64>| {
                for gi in 0..g {
                    let ms =
                        t[gi * gs..(gi + 1) * gs].iter().map(|v| v * v).sum::<f64>() / gs as f64;
                    let kk = 1.0 / (ms + c.norm_eps as f64).sqrt();
                    for i in 0..gs {
                        t[gi * gs + i] *= kk * w.norm_w[gi * gs + i] as f64;
                    }
                }
            };
            if c.norm_before_gate {
                norm(&mut y);
                for i in 0..di {
                    y[i] *= si(z[i]);
                }
            } else {
                for i in 0..di {
                    y[i] *= si(z[i]);
                }
                norm(&mut y);
            }
            out.extend(lin(&w.out_proj, di, &y).into_iter().map(|v| v as f32));
        }
        out
    }

    #[test]
    fn mamba2_step_chunked_prefill_and_naive_reference_agree() {
        for before in [false, true] {
            let xs = inputs(&mut Rng(31), 70, 6);
            let reference = naive_m2(&m2(1, before, 4), &xs);
            for chunk in [1, 3, 8, 64, 1000] {
                let m = m2(chunk, before, 4);
                let mut st = m.new_state();
                close(&m.prefill(&mut st, &xs).unwrap(), &reference, 5e-4);
            }
            let m = m2(16, before, 4);
            let mut a = m.new_state();
            let mut stepped = vec![];
            for x in xs.chunks(6) {
                stepped.extend(m.step(&mut a, x).unwrap());
            }
            close(&stepped, &reference, 5e-4);
            let mut b = m.new_state();
            m.prefill(&mut b, &xs).unwrap();
            close(&a.h, &b.h, 5e-4);
            close(&a.conv, &b.conv, 1e-5);
        }
    }

    #[test]
    fn mamba2_prefill_continues_from_a_nonzero_state_and_is_causal() {
        let m = m2(7, false, 6);
        let mut r = Rng(2);
        let xs = inputs(&mut r, 40, 6);
        let (mut a, mut b) = (m.new_state(), m.new_state());
        let whole = m.prefill(&mut a, &xs).unwrap();
        let mut parts = m.prefill(&mut b, &xs[..13 * 6]).unwrap();
        parts.extend(m.prefill(&mut b, &xs[13 * 6..]).unwrap());
        close(&parts, &whole, 5e-4);
        let mut ys = xs.clone();
        for v in &mut ys[30 * 6..] {
            *v = r.f() * 7.0;
        }
        let mut c = m.new_state();
        let other = m.prefill(&mut c, &ys).unwrap();
        close(&other[..30 * 6], &whole[..30 * 6], 1e-5);
    }

    #[test]
    fn mamba2_is_stable_over_ten_thousand_steps() {
        let m = m2(64, false, 8);
        let mut st = m.new_state();
        let bytes = st.bytes();
        let mut r = Rng(5);
        let mut worst: f32 = 0.0;
        for _ in 0..10_000 {
            let y = m.step(&mut st, &r.v(6, 4.0)).unwrap();
            assert!(y.iter().all(|v| v.is_finite()));
            worst = y.iter().fold(worst, |a, v| a.max(v.abs()));
        }
        assert!(worst < 1e3, "{worst}");
        assert_eq!(st.bytes(), bytes);
    }

    // ---------------------------------------------------------------- RWKV-6
    fn rw(seed: u64) -> Rwkv6Layer {
        let mut r = Rng(seed);
        let (d, hs, f, e, ed) = (8usize, 4usize, 12usize, 3usize, 2usize);
        let cfg = Rwkv6Config {
            hidden: d,
            head_size: hs,
            ffn: f,
            ln_eps: 1e-5,
        };
        let ln = [r.pos(d), r.v(d, 0.2), r.pos(d), r.v(d, 0.2)];
        let att = Rwkv6TimeMix {
            maa_x: r.pos(d),
            maa: [r.pos(d), r.pos(d), r.pos(d), r.pos(d), r.pos(d)],
            maa_w1: r.v(d * 5 * e, 0.8),
            maa_w2: r.v(5 * e * d, 0.8),
            decay: r.v(d, 2.0),
            decay_w1: r.v(d * ed, 0.8),
            decay_w2: r.v(ed * d, 0.8),
            faaaa: r.v(d, 1.0),
            wr: r.v(d * d, 0.7),
            wk: r.v(d * d, 0.7),
            wv: r.v(d * d, 0.7),
            wg: r.v(d * d, 0.7),
            wo: r.v(d * d, 0.7),
            ln_x_w: r.pos(d),
            ln_x_b: r.v(d, 0.2),
        };
        let ffn = Rwkv6ChannelMix {
            maa_k: r.pos(d),
            maa_r: r.pos(d),
            wk: r.v(f * d, 0.7),
            wr: r.v(d * d, 0.7),
            wv: r.v(d * f, 0.7),
        };
        Rwkv6Layer::new(cfg, ln, att, ffn).unwrap()
    }

    /// Whole-history reference: the WKV state written as an explicit sum over past tokens.
    fn naive_rwkv(l: &Rwkv6Layer, xs: &[f32]) -> Vec<f32> {
        let (d, hs, nh) = (l.cfg.hidden, l.cfg.head_size, l.cfg.n_heads());
        let t = xs.len() / d;
        let ln = |x: &[f32], w: &[f32], b: &[f32]| layer_norm(x, w, b, l.cfg.ln_eps);
        let lin = |w: &[f32], cols: usize, x: &[f32]| -> Vec<f32> {
            w.chunks(cols)
                .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let mut hid: Vec<Vec<f32>> = xs.chunks(d).map(|c| c.to_vec()).collect();
        let a = &l.att;
        let normed: Vec<Vec<f32>> = hid.iter().map(|h| ln(h, &l.ln1_w, &l.ln1_b)).collect();
        let (mut rs, mut ks, mut vs, mut gs, mut ds) = (vec![], vec![], vec![], vec![], vec![]);
        for ti in 0..t {
            let x = &normed[ti];
            let prev = if ti == 0 {
                vec![0.0; d]
            } else {
                normed[ti - 1].clone()
            };
            let xx: Vec<f32> = (0..d).map(|i| prev[i] - x[i]).collect();
            let xxx: Vec<f32> = (0..d).map(|i| x[i] + xx[i] * a.maa_x[i]).collect();
            let e = l.mix_extra;
            let tt: Vec<f32> = (0..5 * e)
                .map(|j| {
                    (0..d)
                        .map(|i| xxx[i] * a.maa_w1[i * 5 * e + j])
                        .sum::<f32>()
                        .tanh()
                })
                .collect();
            let mix = |j: usize| -> Vec<f32> {
                (0..d)
                    .map(|i| {
                        let m: f32 = (0..e)
                            .map(|q| tt[j * e + q] * a.maa_w2[j * e * d + q * d + i])
                            .sum();
                        x[i] + xx[i] * (a.maa[j][i] + m)
                    })
                    .collect()
            };
            let (xw, xk, xv, xr, xg) = (mix(0), mix(1), mix(2), mix(3), mix(4));
            rs.push(lin(&a.wr, d, &xr));
            ks.push(lin(&a.wk, d, &xk));
            vs.push(lin(&a.wv, d, &xv));
            gs.push(lin(&a.wg, d, &xg).into_iter().map(silu).collect::<Vec<_>>());
            let ed = l.decay_extra;
            let dw: Vec<f32> = (0..ed)
                .map(|q| {
                    (0..d)
                        .map(|i| xw[i] * a.decay_w1[i * ed + q])
                        .sum::<f32>()
                        .tanh()
                })
                .collect();
            ds.push(
                (0..d)
                    .map(|i| {
                        let w: f32 = (0..ed).map(|q| dw[q] * a.decay_w2[q * d + i]).sum();
                        (-(a.decay[i] + w).exp()).exp()
                    })
                    .collect::<Vec<f32>>(),
            );
        }
        for ti in 0..t {
            let mut y = vec![0f32; d];
            for h in 0..nh {
                for j in 0..hs {
                    let mut acc = 0f64;
                    for i in 0..hs {
                        let idx = h * hs + i;
                        let mut s =
                            a.faaaa[idx] as f64 * ks[ti][idx] as f64 * vs[ti][h * hs + j] as f64;
                        for past in 0..ti {
                            let mut prod = 1f64;
                            for rr in past + 1..ti {
                                prod *= ds[rr][idx] as f64;
                            }
                            s += prod * ks[past][idx] as f64 * vs[past][h * hs + j] as f64;
                        }
                        acc += rs[ti][idx] as f64 * s;
                    }
                    y[h * hs + j] = acc as f32;
                }
                let s = &mut y[h * hs..(h + 1) * hs];
                let mean = s.iter().sum::<f32>() / hs as f32;
                let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hs as f32;
                let k = 1.0 / (var + 64e-5).sqrt();
                for (i, v) in s.iter_mut().enumerate() {
                    *v = (*v - mean) * k * a.ln_x_w[h * hs + i] + a.ln_x_b[h * hs + i];
                }
            }
            let yg: Vec<f32> = y.iter().zip(&gs[ti]).map(|(y, g)| y * g).collect();
            let o = lin(&a.wo, d, &yg);
            for i in 0..d {
                hid[ti][i] += o[i];
            }
        }
        let f = &l.ffn;
        let n2: Vec<Vec<f32>> = hid.iter().map(|h| ln(h, &l.ln2_w, &l.ln2_b)).collect();
        for ti in 0..t {
            let x = &n2[ti];
            let prev = if ti == 0 {
                vec![0.0; d]
            } else {
                n2[ti - 1].clone()
            };
            let xk: Vec<f32> = (0..d)
                .map(|i| x[i] + (prev[i] - x[i]) * f.maa_k[i])
                .collect();
            let xr: Vec<f32> = (0..d)
                .map(|i| x[i] + (prev[i] - x[i]) * f.maa_r[i])
                .collect();
            let k: Vec<f32> = lin(&f.wk, d, &xk)
                .into_iter()
                .map(|v| v.max(0.0) * v.max(0.0))
                .collect();
            let kv = lin(&f.wv, l.cfg.ffn, &k);
            let r = lin(&f.wr, d, &xr);
            for i in 0..d {
                hid[ti][i] += kv[i] / (1.0 + (-r[i]).exp());
            }
        }
        hid.concat()
    }

    #[test]
    fn rwkv6_matches_the_explicit_history_reference() {
        let l = rw(13);
        let xs = inputs(&mut Rng(8), 18, 8);
        let mut st = l.new_state();
        close(
            &l.prefill(&mut st, &xs).unwrap(),
            &naive_rwkv(&l, &xs),
            5e-4,
        );
    }

    #[test]
    fn rwkv6_is_causal_and_prefill_splits() {
        let l = rw(2);
        let mut r = Rng(6);
        let xs = inputs(&mut r, 14, 8);
        let (mut a, mut b, mut c) = (l.new_state(), l.new_state(), l.new_state());
        let whole = l.prefill(&mut a, &xs).unwrap();
        let mut ys = xs.clone();
        for v in &mut ys[9 * 8..] {
            *v = r.f() * 8.0;
        }
        let other = l.prefill(&mut b, &ys).unwrap();
        assert_eq!(whole[..9 * 8], other[..9 * 8]);
        let mut parts = l.prefill(&mut c, &xs[..4 * 8]).unwrap();
        parts.extend(l.prefill(&mut c, &xs[4 * 8..]).unwrap());
        assert_eq!(whole, parts);
    }

    #[test]
    fn rwkv6_is_stable_over_ten_thousand_steps_with_fixed_state() {
        let l = rw(17);
        let mut st = l.new_state();
        let bytes = st.bytes();
        assert_eq!(bytes, 4 * (8 + 2 * 4 * 4 + 8));
        let mut r = Rng(4);
        let mut worst: f32 = 0.0;
        for _ in 0..10_000 {
            // the layer is a residual block: feed a bounded token each time, not its own output
            let y = l.step(&mut st, &r.v(8, 2.0)).unwrap();
            assert!(y.iter().all(|v| v.is_finite()));
            worst = y.iter().fold(worst, |a, v| a.max(v.abs()));
        }
        assert!(worst < 1e3, "{worst}");
        assert_eq!(st.bytes(), bytes);
    }

    // ------------------------------------------------------- config, names, hostile input
    #[test]
    fn hf_configs_are_parsed() {
        let c = Mamba1Config::from_hf_config(r#"{"model_type":"mamba","hidden_size":768,"state_size":16,"expand":2,"conv_kernel":4,"time_step_rank":"auto"}"#).unwrap();
        assert_eq!(
            (c.d_inner, c.dt_rank, c.bcdt_rms_eps, c.conv_bias),
            (1536, 48, None, true)
        );
        let f = Mamba1Config::from_hf_config(r#"{"model_type":"falcon_mamba","hidden_size":64,"intermediate_size":128,"time_step_rank":4,"mixer_rms_eps":1e-5}"#).unwrap();
        assert_eq!((f.d_inner, f.dt_rank, f.bcdt_rms_eps), (128, 4, Some(1e-5)));
        let m = Mamba2Config::from_hf_config(r#"{"model_type":"mamba2","hidden_size":768,"num_heads":24,"head_dim":64,"state_size":128,"n_groups":1,"expand":2}"#).unwrap();
        assert_eq!(
            (m.d_inner(), m.conv_dim(), m.proj_dim()),
            (1536, 1536 + 256, 1536 + 1792 + 24)
        );
        let r = Rwkv6Config::from_hf_config(
            r#"{"hidden_size":2048,"head_size":64,"intermediate_size":7168}"#,
        )
        .unwrap();
        assert_eq!((r.n_heads(), r.ffn), (32, 7168));
        assert!(Rwkv6Config::from_hf_config(r#"{"hidden_size":2048}"#).is_err());
    }

    #[test]
    fn hostile_configs_weights_and_inputs_are_errors() {
        assert!(Mamba1Config::from_hf_config("[1]").is_err());
        assert!(Mamba1Config::from_hf_config("{}").is_err());
        assert!(Mamba1Config::from_hf_config(r#"{"hidden_size":0}"#).is_err());
        assert!(
            Mamba1Config::from_hf_config(
                r#"{"hidden_size":16777216,"expand":16777216,"state_size":16777216}"#
            )
            .is_err()
        );
        assert!(
            Mamba2Config::from_hf_config(r#"{"hidden_size":64,"num_heads":5,"n_groups":2}"#)
                .is_err()
        );
        assert!(
            Mamba2Config::from_hf_config(r#"{"hidden_size":64,"num_heads":18446744073709551615}"#)
                .is_err()
        );
        assert!(Rwkv6Config::from_hf_config(r#"{"hidden_size":100,"head_size":64}"#).is_err());
        assert!(
            Rwkv6Config::from_hf_config(
                r#"{"hidden_size":128,"head_size":64,"attention_hidden_size":64}"#
            )
            .is_err()
        );
        let bad = Mamba1Config {
            d_conv: 0,
            ..m1_cfg(4, false)
        };
        assert!(bad.validate().is_err());
        let m = m1(4, false, 1);
        let mut st = m.new_state();
        assert!(matches!(
            m.step(&mut st, &[0.0; 5]),
            Err(SsmError::Input(_))
        ));
        assert!(m.prefill(&mut st, &[0.0; 7]).is_err());
        assert!(m.prefill(&mut st, &[]).unwrap().is_empty());
        let mut r = Rng(1);
        let c = m1_cfg(4, false);
        let w = Mamba1Weights {
            in_proj: r.v(3, 1.0),
            in_bias: None,
            conv_w: vec![],
            conv_b: None,
            x_proj: vec![],
            dt_proj: vec![],
            dt_bias: vec![],
            a_log: vec![],
            d: vec![],
            out_proj: vec![],
            out_bias: None,
        };
        assert!(matches!(Mamba1::new(c, w), Err(SsmError::Shape(_))));
        let m2 = m2(8, false, 1);
        assert!(m2.step(&mut m2.new_state(), &[0.0; 2]).is_err());
        let l = rw(1);
        assert!(l.step(&mut l.new_state(), &[0.0; 3]).is_err());
        assert!(l.prefill(&mut l.new_state(), &[0.0; 9]).is_err());
    }

    struct Map(HashMap<String, Vec<f32>>);
    impl TensorSource for Map {
        fn read(&self, name: &str) -> Result<Vec<f32>, LoadError> {
            self.0
                .get(name)
                .cloned()
                .ok_or_else(|| LoadError::Missing(name.into()))
        }
        fn has(&self, name: &str) -> bool {
            self.0.contains_key(name)
        }
    }

    #[test]
    fn loading_from_hf_and_gguf_names_gives_the_same_layer() {
        let m = m1(4, false, 5);
        let w = &m.w;
        let (hf, gg) = (MambaNames::hf_mamba1(2), MambaNames::gguf(2));
        let neg_a: Vec<f32> = w.a_log.iter().map(|v| -v.exp()).collect();
        let put = |names: &MambaNames, a: Vec<f32>| {
            let mut t = HashMap::new();
            for (n, v) in [
                (&names.in_proj, &w.in_proj),
                (&names.conv_w, &w.conv_w),
                (&names.x_proj, &w.x_proj),
                (&names.dt_proj, &w.dt_proj),
                (&names.dt_bias, &w.dt_bias),
                (&names.d, &w.d),
                (&names.out_proj, &w.out_proj),
            ] {
                t.insert(n.clone(), v.clone());
            }
            t.insert(names.conv_b.clone(), w.conv_b.clone().unwrap());
            t.insert(names.a.clone(), a);
            Map(t)
        };
        let xs = inputs(&mut Rng(3), 6, 6);
        let want = m.prefill(&mut m.new_state(), &xs).unwrap();
        let a = Mamba1::load(m1_cfg(4, false), &put(&hf, w.a_log.clone()), &hf, false).unwrap();
        let b = Mamba1::load(m1_cfg(4, false), &put(&gg, neg_a.clone()), &gg, true).unwrap();
        close(&a.prefill(&mut a.new_state(), &xs).unwrap(), &want, 1e-5);
        close(&b.prefill(&mut b.new_state(), &xs).unwrap(), &want, 1e-4);
        assert!(hf.in_proj.starts_with("backbone.layers.2.mixer.") && gg.a == "blk.2.ssm_a");
        // GGUF A must be negative, a missing tensor is named, and optional ones may be absent
        let mut bad = put(&gg, w.a_log.clone());
        assert!(Mamba1::load(m1_cfg(4, false), &bad, &gg, true).is_err());
        bad.0.remove(&gg.out_proj);
        let e = Mamba1::load(m1_cfg(4, false), &bad, &gg, false)
            .err()
            .unwrap();
        assert!(e.to_string().contains("blk.2.ssm_out.weight"), "{e}");
    }

    #[test]
    fn rwkv_layer_loads_from_native_names() {
        let l = rw(23);
        let n = Rwkv6Names::native(5);
        let mut t = HashMap::new();
        let mut put = |s: &str, v: &Vec<f32>| {
            t.insert(n.n(s), v.clone());
        };
        put("ln1.weight", &l.ln1_w);
        put("ln1.bias", &l.ln1_b);
        put("ln2.weight", &l.ln2_w);
        put("ln2.bias", &l.ln2_b);
        let a = &l.att;
        put("att.time_maa_x", &a.maa_x);
        for (s, v) in ["w", "k", "v", "r", "g"].iter().zip(&a.maa) {
            put(&format!("att.time_maa_{s}"), v);
        }
        put("att.time_maa_w1", &a.maa_w1);
        put("att.time_maa_w2", &a.maa_w2);
        put("att.time_decay", &a.decay);
        put("att.time_decay_w1", &a.decay_w1);
        put("att.time_decay_w2", &a.decay_w2);
        put("att.time_faaaa", &a.faaaa);
        put("att.receptance.weight", &a.wr);
        put("att.key.weight", &a.wk);
        put("att.value.weight", &a.wv);
        put("att.gate.weight", &a.wg);
        put("att.output.weight", &a.wo);
        put("att.ln_x.weight", &a.ln_x_w);
        put("att.ln_x.bias", &a.ln_x_b);
        put("ffn.time_maa_k", &l.ffn.maa_k);
        put("ffn.time_maa_r", &l.ffn.maa_r);
        put("ffn.key.weight", &l.ffn.wk);
        put("ffn.receptance.weight", &l.ffn.wr);
        put("ffn.value.weight", &l.ffn.wv);
        let loaded = Rwkv6Layer::load(l.cfg.clone(), &Map(t), &n).unwrap();
        let xs = inputs(&mut Rng(1), 5, 8);
        assert_eq!(
            loaded.prefill(&mut loaded.new_state(), &xs).unwrap(),
            l.prefill(&mut l.new_state(), &xs).unwrap()
        );
    }

    // ---------------------------------------------------------------- RWKV-7
    fn r7_cfg() -> Rwkv7Config {
        Rwkv7Config {
            hidden: 8,
            head_size: 4,
            ffn: 12,
            ln_eps: 1e-5,
            decay_rank: 0,
            a_rank: 0,
            v_rank: 0,
            gate_rank: 0,
            value_residual: true,
        }
    }
    fn r7(first: bool, seed: u64) -> Rwkv7Layer {
        let mut r = Rng(seed);
        let (d, f) = (8usize, 12usize);
        let (rd, ra, rv, rg) = (3usize, 2usize, 2usize, 3usize);
        let att = Rwkv7TimeMix {
            x_r: r.pos(d),
            x_w: r.pos(d),
            x_k: r.pos(d),
            x_v: r.pos(d),
            x_a: r.pos(d),
            x_g: r.pos(d),
            w0: r.v(d, 2.0),
            w1: r.v(d * rd, 0.8),
            w2: r.v(rd * d, 0.8),
            a0: r.v(d, 1.0),
            a1: r.v(d * ra, 0.8),
            a2: r.v(ra * d, 0.8),
            v: if first {
                None
            } else {
                Some([r.v(d, 1.0), r.v(d * rv, 0.8), r.v(rv * d, 0.8)])
            },
            g1: r.v(d * rg, 0.8),
            g2: r.v(rg * d, 0.8),
            k_k: r.pos(d),
            k_a: r.pos(d),
            r_k: r.v(d, 1.0),
            wr: r.v(d * d, 0.7),
            wk: r.v(d * d, 0.7),
            wv: r.v(d * d, 0.7),
            wo: r.v(d * d, 0.7),
            ln_x_w: r.pos(d),
            ln_x_b: r.v(d, 0.2),
        };
        let ffn = Rwkv7ChannelMix {
            x_k: r.pos(d),
            wk: r.v(f * d, 0.7),
            wv: r.v(d * f, 0.7),
        };
        let ln0 = first.then(|| [r.pos(d), r.v(d, 0.2)]);
        Rwkv7Layer::new(
            r7_cfg(),
            first,
            ln0,
            [r.pos(d), r.v(d, 0.2), r.pos(d), r.v(d, 0.2)],
            att,
            ffn,
        )
        .unwrap()
    }

    /// Literal f64 transcription of the reference recurrence: the state is an explicit `N×N` matrix,
    /// `state @ ab` is a real matrix product, and the decay uses the softplus form `exp(-exp(-softplus(-z) - 0.5))`.
    fn naive_r7(l: &Rwkv7Layer, xs: &[f32], v_first_in: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
        let (d, hs, nh) = (l.cfg.hidden, l.cfg.head_size, l.cfg.n_heads());
        let (rd, ra, rv, rg) = (
            l.cfg.decay_rank,
            l.cfg.a_rank,
            l.cfg.v_rank,
            l.cfg.gate_rank,
        );
        let t = xs.len() / d;
        let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
        let sp = |x: f64| (1.0 + x.exp()).ln();
        let ln64 = |x: &[f64], w: &[f32], b: &[f32]| -> Vec<f64> {
            let n = x.len() as f64;
            let m = x.iter().sum::<f64>() / n;
            let var = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n;
            x.iter()
                .enumerate()
                .map(|(i, v)| {
                    (v - m) / (var + l.cfg.ln_eps as f64).sqrt() * w[i] as f64 + b[i] as f64
                })
                .collect()
        };
        let lin = |w: &[f32], cols: usize, x: &[f64]| -> Vec<f64> {
            w.chunks(cols)
                .map(|r| r.iter().zip(x).map(|(a, b)| *a as f64 * b).sum())
                .collect()
        };
        // (x · A) · B with A: d × rk, B: rk × d, optional tanh / sigmoid in between
        let lora =
            |x: &[f64], a: &[f32], b: &[f32], rk: usize, mid: &dyn Fn(f64) -> f64| -> Vec<f64> {
                let h: Vec<f64> = (0..rk)
                    .map(|q| mid((0..d).map(|i| x[i] * a[i * rk + q] as f64).sum()))
                    .collect();
                (0..d)
                    .map(|i| (0..rk).map(|q| h[q] * b[q * d + i] as f64).sum())
                    .collect()
            };
        let at = &l.att;
        let mut s = vec![0f64; nh * hs * hs];
        let (mut prev_att, mut prev_ffn) = (vec![0f64; d], vec![0f64; d]);
        let (mut out, mut vf_out) = (vec![], vec![]);
        for ti in 0..t {
            let mut h: Vec<f64> = xs[ti * d..(ti + 1) * d].iter().map(|v| *v as f64).collect();
            if let Some([w, b]) = &l.ln0 {
                h = ln64(&h, w, b);
            }
            let x = ln64(&h, &l.ln1[0], &l.ln1[1]);
            let xx: Vec<f64> = (0..d).map(|i| prev_att[i] - x[i]).collect();
            let mix =
                |m: &[f32]| -> Vec<f64> { (0..d).map(|i| x[i] + xx[i] * m[i] as f64).collect() };
            let (xr, xw, xk, xv, xa, xg) = (
                mix(&at.x_r),
                mix(&at.x_w),
                mix(&at.x_k),
                mix(&at.x_v),
                mix(&at.x_a),
                mix(&at.x_g),
            );
            let r = lin(&at.wr, d, &xr);
            let mut k = lin(&at.wk, d, &xk);
            let mut v = lin(&at.wv, d, &xv);
            let wl = lora(&xw, &at.w1, &at.w2, rd, &|x| x.tanh());
            let w: Vec<f64> = (0..d)
                .map(|i| -sp(-(at.w0[i] as f64 + wl[i])) - 0.5)
                .collect();
            let decay: Vec<f64> = w.iter().map(|w| (-w.exp()).exp()).collect();
            if l.first {
                vf_out.extend(v.iter().map(|v| *v as f32));
            } else if let Some([v0, v1, v2]) = &at.v {
                let vl = lora(&xv, v1, v2, rv, &|x| x);
                let vfi = v_first_in.unwrap();
                for i in 0..d {
                    v[i] += (vfi[ti * d + i] as f64 - v[i]) * sig(v0[i] as f64 + vl[i]);
                }
            }
            let al = lora(&xa, &at.a1, &at.a2, ra, &|x| x);
            let a: Vec<f64> = (0..d).map(|i| sig(at.a0[i] as f64 + al[i])).collect();
            let g = lora(&xg, &at.g1, &at.g2, rg, &sig);
            let mut kk: Vec<f64> = (0..d).map(|i| k[i] * at.k_k[i] as f64).collect();
            for hh in 0..nh {
                let n = kk[hh * hs..(hh + 1) * hs]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-12);
                kk[hh * hs..(hh + 1) * hs].iter_mut().for_each(|v| *v /= n);
            }
            for i in 0..d {
                k[i] *= 1.0 + (a[i] - 1.0) * at.k_a[i] as f64;
            }
            let mut y = vec![0f64; d];
            for hh in 0..nh {
                let o = hh * hs;
                // ab[j][l] = (-kk[j]) * (kk[l] * a[l]);  S' = S * diag(w) + S @ ab + v kᵀ
                let ab: Vec<f64> = (0..hs * hs)
                    .map(|q| -kk[o + q / hs] * (kk[o + q % hs] * a[o + q % hs]))
                    .collect();
                let old = s[hh * hs * hs..(hh + 1) * hs * hs].to_vec();
                for i in 0..hs {
                    for j in 0..hs {
                        let sab: f64 = (0..hs).map(|m| old[i * hs + m] * ab[m * hs + j]).sum();
                        s[hh * hs * hs + i * hs + j] =
                            old[i * hs + j] * decay[o + j] + sab + v[o + i] * k[o + j];
                    }
                    y[o + i] = (0..hs)
                        .map(|j| s[hh * hs * hs + i * hs + j] * r[o + j])
                        .sum();
                }
                let m = y[o..o + hs].iter().sum::<f64>() / hs as f64;
                let var = y[o..o + hs].iter().map(|v| (v - m) * (v - m)).sum::<f64>() / hs as f64;
                for i in 0..hs {
                    y[o + i] = (y[o + i] - m) / (var + 64e-5).sqrt() * at.ln_x_w[o + i] as f64
                        + at.ln_x_b[o + i] as f64;
                }
                let bonus: f64 = (0..hs)
                    .map(|j| r[o + j] * k[o + j] * at.r_k[o + j] as f64)
                    .sum();
                for i in 0..hs {
                    y[o + i] += bonus * v[o + i];
                }
            }
            let gated: Vec<f64> = y.iter().zip(&g).map(|(y, g)| y * g).collect();
            let att = lin(&at.wo, d, &gated);
            for i in 0..d {
                h[i] += att[i];
            }
            prev_att = x;
            let x2 = ln64(&h, &l.ln2[0], &l.ln2[1]);
            let fk: Vec<f64> = (0..d)
                .map(|i| x2[i] + (prev_ffn[i] - x2[i]) * l.ffn.x_k[i] as f64)
                .collect();
            let kv: Vec<f64> = lin(&l.ffn.wk, d, &fk)
                .into_iter()
                .map(|v| v.max(0.0).powi(2))
                .collect();
            let f = lin(&l.ffn.wv, l.cfg.ffn, &kv);
            prev_ffn = x2;
            out.extend((0..d).map(|i| (h[i] + f[i]) as f32));
        }
        (out, vf_out)
    }

    fn run_stack(layers: &[&Rwkv7Layer], xs: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let d = 8;
        let mut states: Vec<Rwkv7State> = layers.iter().map(|l| l.new_state()).collect();
        let (mut out, mut vfs) = (vec![], vec![]);
        for x in xs.chunks(d) {
            let mut vf = vec![0.0; d];
            let mut h = x.to_vec();
            for (l, st) in layers.iter().zip(&mut states) {
                h = l.step(st, &h, &mut vf).unwrap();
            }
            out.extend(h);
            vfs.extend(vf);
        }
        (out, vfs)
    }

    #[test]
    fn rwkv7_matches_the_literal_matrix_reference_across_two_layers() {
        let (l0, l1) = (r7(true, 31), r7(false, 32));
        assert!(l0.is_first() && !l1.is_first());
        assert_eq!(
            (
                l1.cfg.decay_rank,
                l1.cfg.a_rank,
                l1.cfg.v_rank,
                l1.cfg.gate_rank
            ),
            (3, 2, 2, 3)
        );
        let xs = inputs(&mut Rng(9), 14, 8);
        let (want0, vf) = naive_r7(&l0, &xs, None);
        let (want1, _) = naive_r7(&l1, &want0, Some(&vf));
        let (got, got_vf) = run_stack(&[&l0, &l1], &xs);
        close(&got_vf, &vf, 2e-3);
        close(&got, &want1, 3e-3);
        // a single later layer given an external v_first behaves the same as inside the stack
        let (single, _) = naive_r7(&l1, &xs, Some(&vf));
        let mut st = l1.new_state();
        let mut vfm = vf.clone();
        close(&l1.prefill(&mut st, &xs, &mut vfm).unwrap(), &single, 3e-3);
        assert_eq!(vfm, vf, "a later layer must not change v_first");
    }

    #[test]
    fn rwkv7_the_two_written_forms_of_the_decay_agree() {
        // exp(-exp(-softplus(-z) - 1/2)) == exp(-exp(-1/2) * sigmoid(z))
        for z in [-30.0f64, -3.0, -0.2, 0.0, 0.7, 4.0, 25.0] {
            let a = (-((-(1.0 + (-z).exp()).ln()) - 0.5).exp()).exp();
            let b = (-(-0.5f64).exp() * (1.0 / (1.0 + (-z).exp()))).exp();
            assert!((a - b).abs() < 1e-12, "{z}");
        }
        assert!((RWKV7_DECAY_SCALE as f64 - (-0.5f64).exp()).abs() < 1e-7);
    }

    #[test]
    fn rwkv7_prefill_equals_steps_is_causal_and_splits() {
        let l = r7(true, 5);
        let xs = inputs(&mut Rng(2), 9, 8);
        let (mut a, mut vfa) = (l.new_state(), vec![0.0; xs.len()]);
        let whole = l.prefill(&mut a, &xs, &mut vfa).unwrap();
        let (mut b, mut parts) = (l.new_state(), vec![]);
        for x in xs.chunks(8) {
            parts.extend(l.step(&mut b, x, &mut [0.0; 8]).unwrap());
        }
        assert_eq!(whole, parts);
        assert_eq!(a, b);
        // split prefill continues from the state
        let (mut c, mut vfc) = (l.new_state(), vec![0.0; 5 * 8]);
        let mut sp = l.prefill(&mut c, &xs[..5 * 8], &mut vfc).unwrap();
        sp.extend(l.prefill(&mut c, &xs[5 * 8..], &mut [0.0; 4 * 8]).unwrap());
        assert_eq!(sp, whole);
        // causality: changing later tokens does not change earlier outputs
        let mut ys = xs.clone();
        ys[6 * 8..].iter_mut().for_each(|v| *v = -*v + 0.37);
        let other = l
            .prefill(&mut l.new_state(), &ys, &mut vec![0.0; ys.len()])
            .unwrap();
        assert_eq!(&other[..6 * 8], &whole[..6 * 8]);
        assert_ne!(&other[6 * 8..], &whole[6 * 8..]);
    }

    #[test]
    fn rwkv7_is_stable_over_ten_thousand_steps_with_fixed_state() {
        for (first, seed) in [(true, 7u64), (false, 8)] {
            let l = r7(first, seed);
            let mut st = l.new_state();
            let bytes = st.bytes();
            assert_eq!(bytes, 4 * (8 + 2 * 4 * 4 + 8));
            let mut r = Rng(4);
            let mut worst: f32 = 0.0;
            for _ in 0..10_000 {
                let mut vf = r.v(8, 2.0);
                let y = l.step(&mut st, &r.v(8, 2.0), &mut vf).unwrap();
                assert!(y.iter().all(|v| v.is_finite()));
                worst = y.iter().fold(worst, |a, v| a.max(v.abs()));
            }
            assert!(worst < 1e3, "{worst}");
            assert!(st.wkv.iter().all(|v| v.is_finite()));
            assert_eq!(st.bytes(), bytes);
        }
    }

    #[test]
    fn rwkv7_config_hostile_inputs_and_shapes_are_errors() {
        let c = Rwkv7Config::from_hf_config(r#"{"model_type":"rwkv7","hidden_size":768,"head_dim":64,"intermediate_size":3072,
            "decay_low_rank_dim":64,"a_low_rank_dim":64,"v_low_rank_dim":32,"gate_low_rank_dim":128,"norm_eps":1e-5,"value_residual":false}"#).unwrap();
        assert_eq!(
            (
                c.n_heads(),
                c.ffn,
                c.decay_rank,
                c.a_rank,
                c.v_rank,
                c.gate_rank,
                c.value_residual
            ),
            (12, 3072, 64, 64, 32, 128, false)
        );
        assert!(
            Rwkv7Config::from_hf_config(r#"{"hidden_size":2048,"head_size":64}"#).is_err(),
            "an RWKV-6 description is not RWKV-7"
        );
        assert!(Rwkv7Config::from_hf_config("[1]").is_err());
        assert!(
            Rwkv7Config::from_hf_config(
                r#"{"hidden_size":100,"head_dim":64,"decay_low_rank_dim":8}"#
            )
            .is_err()
        );
        assert!(
            Rwkv7Config::from_hf_config(r#"{"hidden_size":0,"decay_low_rank_dim":8}"#).is_err()
        );
        assert!(Rwkv7Config::from_hf_config(r#"{"hidden_size":16777216,"head_dim":1,"decay_low_rank_dim":16777216,"intermediate_size":16777216}"#).is_err());
        let l = r7(false, 3);
        let mut st = l.new_state();
        assert!(matches!(
            l.step(&mut st, &[0.0; 3], &mut [0.0; 8]),
            Err(SsmError::Input(_))
        ));
        assert!(matches!(
            l.step(&mut st, &[0.0; 8], &mut [0.0; 5]),
            Err(SsmError::Input(_))
        ));
        assert!(l.prefill(&mut st, &[0.0; 9], &mut [0.0; 9]).is_err());
        assert!(l.prefill(&mut st, &[0.0; 16], &mut [0.0; 8]).is_err());
        assert!(l.prefill(&mut st, &[], &mut []).unwrap().is_empty());
        // wrong weight shapes: a truncated LoRA, a first layer carrying a value residual, a rank that contradicts the config
        let mut a = r7(false, 3);
        a.att.w2.pop();
        assert!(matches!(
            Rwkv7Layer::new(
                r7_cfg(),
                false,
                None,
                [vec![1.0; 8], vec![0.0; 8], vec![1.0; 8], vec![0.0; 8]],
                a.att,
                Rwkv7ChannelMix {
                    x_k: vec![0.0; 8],
                    wk: vec![0.0; 96],
                    wv: vec![0.0; 96]
                }
            ),
            Err(SsmError::Shape(_))
        ));
        let b = r7(false, 3);
        let ln = [vec![1.0; 8], vec![0.0; 8], vec![1.0; 8], vec![0.0; 8]];
        assert!(
            Rwkv7Layer::new(
                r7_cfg(),
                true,
                None,
                ln.clone(),
                b.att,
                Rwkv7ChannelMix {
                    x_k: vec![0.0; 8],
                    wk: vec![0.0; 96],
                    wv: vec![0.0; 96]
                }
            )
            .is_err()
        );
        let b = r7(false, 3);
        let bad = Rwkv7Config {
            decay_rank: 5,
            ..r7_cfg()
        };
        assert!(
            Rwkv7Layer::new(
                bad,
                false,
                None,
                ln,
                b.att,
                Rwkv7ChannelMix {
                    x_k: vec![0.0; 8],
                    wk: vec![0.0; 96],
                    wv: vec![0.0; 96]
                }
            )
            .is_err()
        );
    }

    fn put7(l: &Rwkv7Layer, n: &Rwkv7Names) -> Map {
        let mut t = HashMap::new();
        let a = &l.att;
        let d = l.cfg.hidden;
        let mut put = |s: &str, v: &[f32]| {
            t.insert(n.n(s), v.to_vec());
        };
        if !n.fla {
            if let Some([w, b]) = &l.ln0 {
                put("ln0.weight", w);
                put("ln0.bias", b);
            }
            put("ln1.weight", &l.ln1[0]);
            put("ln1.bias", &l.ln1[1]);
            put("ln2.weight", &l.ln2[0]);
            put("ln2.bias", &l.ln2[1]);
            for (s, v) in [
                ("x_r", &a.x_r),
                ("x_w", &a.x_w),
                ("x_k", &a.x_k),
                ("x_v", &a.x_v),
                ("x_a", &a.x_a),
                ("x_g", &a.x_g),
                ("w0", &a.w0),
                ("w1", &a.w1),
                ("w2", &a.w2),
                ("a0", &a.a0),
                ("a1", &a.a1),
                ("a2", &a.a2),
                ("g1", &a.g1),
                ("g2", &a.g2),
                ("k_k", &a.k_k),
                ("k_a", &a.k_a),
                ("r_k", &a.r_k),
                ("receptance.weight", &a.wr),
                ("key.weight", &a.wk),
                ("value.weight", &a.wv),
                ("output.weight", &a.wo),
                ("ln_x.weight", &a.ln_x_w),
                ("ln_x.bias", &a.ln_x_b),
            ] {
                put(&format!("att.{s}"), v);
            }
            if let Some([v0, v1, v2]) = &a.v {
                put("att.v0", v0);
                put("att.v1", v1);
                put("att.v2", v2);
            }
            put("ffn.x_k", &l.ffn.x_k);
            put("ffn.key.weight", &l.ffn.wk);
            put("ffn.value.weight", &l.ffn.wv);
        } else {
            if let Some([w, b]) = &l.ln0 {
                put("pre_norm.weight", w);
                put("pre_norm.bias", b);
            }
            put("attn_norm.weight", &l.ln1[0]);
            put("attn_norm.bias", &l.ln1[1]);
            put("ffn_norm.weight", &l.ln2[0]);
            put("ffn_norm.bias", &l.ln2[1]);
            for (s, v) in [
                ("x_r", &a.x_r),
                ("x_w", &a.x_w),
                ("x_k", &a.x_k),
                ("x_v", &a.x_v),
                ("x_a", &a.x_a),
                ("x_g", &a.x_g),
                ("k_k", &a.k_k),
                ("k_a", &a.k_a),
                ("r_k", &a.r_k),
                ("r_proj.weight", &a.wr),
                ("k_proj.weight", &a.wk),
                ("v_proj.weight", &a.wv),
                ("o_proj.weight", &a.wo),
                ("g_norm.weight", &a.ln_x_w),
                ("g_norm.bias", &a.ln_x_b),
            ] {
                put(&format!("attn.{s}"), v);
            }
            let mut lora = |nm: &str, w1: &[f32], w2: &[f32], bias: Option<&[f32]>| {
                let rk = w1.len() / d;
                put(&format!("attn.{nm}.lora.0.weight"), &transpose(w1, d, rk)); // rank × hidden
                put(&format!("attn.{nm}.lora.2.weight"), &transpose(w2, rk, d)); // hidden × rank
                if let Some(b) = bias {
                    put(&format!("attn.{nm}.lora.2.bias"), b);
                }
            };
            lora("w_lora", &a.w1, &a.w2, Some(&a.w0));
            lora("a_lora", &a.a1, &a.a2, Some(&a.a0));
            lora("g_lora", &a.g1, &a.g2, None);
            if let Some([v0, v1, v2]) = &a.v {
                lora("v_lora", v1, v2, Some(v0));
            }
            put("ffn.x_k", &l.ffn.x_k);
            put("ffn.key.weight", &l.ffn.wk);
            put("ffn.value.weight", &l.ffn.wv);
        }
        Map(t)
    }

    #[test]
    fn rwkv7_loads_from_native_and_fla_names_with_ranks_taken_from_the_shapes() {
        let xs = inputs(&mut Rng(1), 6, 8);
        let vf0: Vec<f32> = inputs(&mut Rng(2), 6, 8);
        for (first, idx) in [(true, 0usize), (false, 3)] {
            let l = r7(first, 11 + idx as u64);
            let want = l
                .prefill(&mut l.new_state(), &xs, &mut vf0.clone())
                .unwrap();
            for names in [Rwkv7Names::native(idx), Rwkv7Names::fla(idx)] {
                let src = put7(&l, &names);
                let loaded = Rwkv7Layer::load(r7_cfg(), &src, &names, idx).unwrap();
                assert_eq!(loaded.is_first(), first);
                assert_eq!(
                    (
                        loaded.cfg.decay_rank,
                        loaded.cfg.a_rank,
                        loaded.cfg.gate_rank
                    ),
                    (3, 2, 3)
                );
                assert_eq!(
                    loaded
                        .prefill(&mut loaded.new_state(), &xs, &mut vf0.clone())
                        .unwrap(),
                    want,
                    "{names:?}"
                );
            }
        }
        // a missing tensor is named; the layer index is part of the name
        let l = r7(false, 3);
        let n = Rwkv7Names::native(3);
        let mut src = put7(&l, &n);
        src.0.remove("blocks.3.att.k_a");
        let e = Rwkv7Layer::load(r7_cfg(), &src, &n, 3).err().unwrap();
        assert!(e.to_string().contains("blocks.3.att.k_a"), "{e}");
        // the value residual is skipped when the configuration turns it off
        let n = Rwkv7Names::native(3);
        let off = Rwkv7Config {
            value_residual: false,
            ..r7_cfg()
        };
        let loaded = Rwkv7Layer::load(off, &put7(&l, &n), &n, 3).unwrap();
        assert!(loaded.att.v.is_none());
    }
}
