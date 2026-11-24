//! Multi-head latent attention (DeepSeek-V2/V3, Kimi-K2), written from the public paper and
//! the published `config.json` fields. Standalone: it depends only on [`crate::ops`].
//!
//! Two implementations of the same function:
//! * [`Mla::forward_naive`] expands the latent into per-head keys and values and caches those
//!   (`heads · (nope + rope + v)` floats per token). It is the oracle for the tests.
//! * [`Mla::forward`] is the memory-efficient "absorbed" form. It caches only the compressed
//!   latent plus the shared rotary key (`kv_lora_rank + rope` floats per token) and folds
//!   `kv_b_proj` into the query and output side. For DeepSeek-V3 shapes that is about 1/71st of
//!   the naive cache, which is the whole point of MLA.
//!
//! Weights are row-major `out × in` f32 vectors. Convert a [`crate::model::Weight`] with
//! `Weight::to_f32()`; quantised projections can be wired in later without changing the maths.
//!
//! Rotary pairing follows DeepSeek's reference: adjacent pairs `(2i, 2i+1)` rotate together,
//! applied only to `q_pe` and the shared `k_pe`, with YaRN frequency blending and the `mscale`
//! corrections (softmax scale · mscale², cos/sin · mscale/mscale_all_dim ratio).
use crate::ops::{matvec, rmsnorm, softmax};
use serde_json::Value;

#[derive(Debug, PartialEq, Eq)]
pub enum MlaError {
    Config(String),
    Shape(String),
    Input(String),
}
impl std::fmt::Display for MlaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MlaError::Config(s) | MlaError::Shape(s) | MlaError::Input(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for MlaError {}

#[derive(Clone, Debug, PartialEq)]
pub struct Yarn {
    pub factor: f64,
    pub original_max_position: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub mscale: f64,
    pub mscale_all_dim: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MlaConfig {
    pub hidden: usize,
    pub heads: usize,
    /// `None` = full `q_proj`; `Some(r)` = low-rank `q_a_proj` → norm → `q_b_proj`.
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub v_head: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub yarn: Option<Yarn>,
}

const MAX_DIM: usize = 1 << 20;

fn ugte(c: &Value, k: &str) -> Option<usize> {
    c.get(k).and_then(Value::as_u64).map(|v| v as usize)
}
fn fget(c: &Value, k: &str) -> Option<f64> {
    c.get(k).and_then(Value::as_f64)
}

impl MlaConfig {
    pub fn validate(&self) -> Result<(), MlaError> {
        let bad = |m: &str| Err(MlaError::Config(m.to_string()));
        for (n, v) in [
            ("hidden", self.hidden),
            ("heads", self.heads),
            ("kv_lora_rank", self.kv_lora_rank),
            ("qk_nope_head_dim", self.qk_nope),
            ("qk_rope_head_dim", self.qk_rope),
            ("v_head_dim", self.v_head),
        ] {
            if v == 0 || v > MAX_DIM {
                return Err(MlaError::Config(format!("{n} = {v} is out of range")));
            }
        }
        if self.q_lora_rank.is_some_and(|r| r == 0 || r > MAX_DIM) {
            return bad("q_lora_rank is out of range");
        }
        if !self.qk_rope.is_multiple_of(2) {
            return bad("qk_rope_head_dim must be even");
        }
        if self
            .heads
            .checked_mul(self.qk_nope + self.qk_rope + self.v_head)
            .is_none_or(|n| n > (1 << 28))
        {
            return bad("head dimensions are implausibly large");
        }
        if !(self.rope_theta.is_finite() && self.rope_theta > 1.0) {
            return bad("rope_theta must be > 1");
        }
        if !(self.eps.is_finite() && self.eps >= 0.0) {
            return bad("rms_norm_eps is invalid");
        }
        if let Some(y) = &self.yarn
            && !(y.factor.is_finite()
                && y.factor >= 1.0
                && y.original_max_position.is_finite()
                && y.original_max_position > 0.0
                && y.beta_fast.is_finite()
                && y.beta_fast > 0.0
                && y.beta_slow.is_finite()
                && y.beta_slow > 0.0
                && y.mscale.is_finite()
                && y.mscale_all_dim.is_finite())
        {
            return bad("rope_scaling values are invalid");
        }
        Ok(())
    }

    /// Parse a Hugging Face DeepSeek-style `config.json` (a `text_config` wrapper is followed).
    pub fn from_hf_config(json: &str) -> Result<MlaConfig, MlaError> {
        let root: Value =
            serde_json::from_str(json).map_err(|e| MlaError::Config(format!("bad JSON: {e}")))?;
        if !root.is_object() {
            return Err(MlaError::Config("config is not an object".into()));
        }
        let c = root
            .get("text_config")
            .filter(|t| t.is_object())
            .unwrap_or(&root);
        let need =
            |k: &str| ugte(c, k).ok_or_else(|| MlaError::Config(format!("config has no `{k}`")));
        let rs = c
            .get("rope_scaling")
            .filter(|v| v.is_object())
            .or_else(|| c.get("rope_parameters").filter(|v| v.is_object()));
        let yarn = rs.and_then(|r| {
            let ty = r
                .get("type")
                .or_else(|| r.get("rope_type"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if ty != "yarn" {
                return None;
            }
            Some(Yarn {
                factor: fget(r, "factor").unwrap_or(1.0),
                original_max_position: fget(r, "original_max_position_embeddings")
                    .unwrap_or(4096.0),
                beta_fast: fget(r, "beta_fast").unwrap_or(32.0),
                beta_slow: fget(r, "beta_slow").unwrap_or(1.0),
                mscale: fget(r, "mscale").unwrap_or(1.0),
                mscale_all_dim: fget(r, "mscale_all_dim").unwrap_or(0.0),
            })
        });
        let cfg = MlaConfig {
            hidden: need("hidden_size")?,
            heads: need("num_attention_heads")?,
            q_lora_rank: ugte(c, "q_lora_rank"),
            kv_lora_rank: need("kv_lora_rank")?,
            qk_nope: need("qk_nope_head_dim")?,
            qk_rope: need("qk_rope_head_dim")?,
            v_head: need("v_head_dim")?,
            eps: fget(c, "rms_norm_eps").unwrap_or(1e-6) as f32,
            rope_theta: fget(c, "rope_theta").unwrap_or(10_000.0),
            yarn,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn q_head(&self) -> usize {
        self.qk_nope + self.qk_rope
    }
    /// Floats cached per token by the absorbed implementation.
    pub fn latent_floats_per_token(&self) -> usize {
        self.kv_lora_rank + self.qk_rope
    }
    /// Floats cached per token by the naive (expanded) implementation.
    pub fn full_floats_per_token(&self) -> usize {
        self.heads * (self.qk_nope + self.qk_rope + self.v_head)
    }
    pub fn latent_bytes_per_token(&self) -> usize {
        self.latent_floats_per_token() * 4
    }
    pub fn full_bytes_per_token(&self) -> usize {
        self.full_floats_per_token() * 4
    }
}

/// Dense projection weights, row-major `out × in`.
#[derive(Clone, Debug, Default)]
pub struct MlaWeights {
    pub q_a: Vec<f32>,
    pub q_a_norm: Vec<f32>,
    pub q_b: Vec<f32>,
    /// Used instead of the three above when `q_lora_rank` is `None`.
    pub q_proj: Vec<f32>,
    pub kv_a: Vec<f32>,
    pub kv_a_norm: Vec<f32>,
    /// `heads·(nope+v) × kv_lora_rank`; per head the `nope` key rows come first, then the `v` rows.
    pub kv_b: Vec<f32>,
    pub o: Vec<f32>,
}

/// Tensor names for one layer's attention block.
pub mod names {
    /// Hugging Face: (q_a, q_a_norm, q_b, q_proj, kv_a, kv_a_norm, kv_b, o).
    pub fn hf(layer: usize) -> [String; 8] {
        let p = format!("model.layers.{layer}.self_attn");
        [
            format!("{p}.q_a_proj.weight"),
            format!("{p}.q_a_layernorm.weight"),
            format!("{p}.q_b_proj.weight"),
            format!("{p}.q_proj.weight"),
            format!("{p}.kv_a_proj_with_mqa.weight"),
            format!("{p}.kv_a_layernorm.weight"),
            format!("{p}.kv_b_proj.weight"),
            format!("{p}.o_proj.weight"),
        ]
    }
    /// GGUF: (q_a, q_a_norm, q_b, q, kv_a_mqa, kv_a_norm, k_b, v_b, output). llama.cpp stores
    /// `kv_b_proj` split in two; see [`super::join_kv_b`].
    pub fn gguf(layer: usize) -> [String; 9] {
        let p = format!("blk.{layer}");
        [
            format!("{p}.attn_q_a.weight"),
            format!("{p}.attn_q_a_norm.weight"),
            format!("{p}.attn_q_b.weight"),
            format!("{p}.attn_q.weight"),
            format!("{p}.attn_kv_a_mqa.weight"),
            format!("{p}.attn_kv_a_norm.weight"),
            format!("{p}.attn_k_b.weight"),
            format!("{p}.attn_v_b.weight"),
            format!("{p}.attn_output.weight"),
        ]
    }
}

/// Rebuild the combined `kv_b_proj` from separate key/value up-projections laid out as
/// `[heads][nope][kv_lora]` and `[heads][v][kv_lora]`. NOTE: written from the format
/// description; the exact GGUF layout of `attn_k_b` (llama.cpp keeps it transposed) has not been
/// checked against a real file, so verify before relying on it.
pub fn join_kv_b(cfg: &MlaConfig, k_b: &[f32], v_b: &[f32]) -> Result<Vec<f32>, MlaError> {
    cfg.validate()?;
    let (kv, n, v) = (cfg.kv_lora_rank, cfg.qk_nope, cfg.v_head);
    if k_b.len() != cfg.heads * n * kv || v_b.len() != cfg.heads * v * kv {
        return Err(MlaError::Shape(
            "k_b / v_b sizes do not match the configuration".into(),
        ));
    }
    let mut out = Vec::with_capacity(cfg.heads * (n + v) * kv);
    for h in 0..cfg.heads {
        out.extend_from_slice(&k_b[h * n * kv..(h + 1) * n * kv]);
        out.extend_from_slice(&v_b[h * v * kv..(h + 1) * v * kv]);
    }
    Ok(out)
}

fn yarn_mscale(scale: f64, m: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * m * crate::simd::rmath::ln64(scale) + 1.0
    }
}

/// Inverse frequencies (one per rotating pair) and the cos/sin magnitude factor.
fn rope_table(cfg: &MlaConfig) -> (Vec<f64>, f64) {
    let d = cfg.qk_rope;
    let half = d / 2;
    let extra: Vec<f64> = (0..half)
        .map(|i| crate::simd::rmath::powf64(cfg.rope_theta, -(2.0 * i as f64) / d as f64))
        .collect();
    let Some(y) = &cfg.yarn else {
        return (extra, 1.0);
    };
    let corr = |rot: f64| {
        d as f64
            * crate::simd::rmath::ln64(y.original_max_position / (rot * 2.0 * std::f64::consts::PI))
            / (2.0 * crate::simd::rmath::ln64(cfg.rope_theta))
    };
    let low = corr(y.beta_fast).floor().max(0.0);
    let high = corr(y.beta_slow).ceil().min(d as f64 - 1.0);
    let high = if (high - low).abs() < f64::EPSILON {
        high + 0.001
    } else {
        high
    };
    let freqs = (0..half)
        .map(|i| {
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let mask = 1.0 - ramp;
            let inter = extra[i] / y.factor;
            inter * (1.0 - mask) + extra[i] * mask
        })
        .collect();
    (
        freqs,
        yarn_mscale(y.factor, y.mscale) / yarn_mscale(y.factor, y.mscale_all_dim),
    )
}

/// Latent cache: per token the normalised compressed kv and the rotated shared key.
#[derive(Clone, Debug, Default)]
pub struct MlaCache {
    latent: Vec<f32>,
    k_pe: Vec<f32>,
    len: usize,
}
impl MlaCache {
    pub fn new() -> MlaCache {
        MlaCache::default()
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn bytes(&self) -> usize {
        (self.latent.len() + self.k_pe.len()) * 4
    }
}

/// Expanded cache of the naive implementation: per token and head `[k_nope, k_pe]` and `v`.
#[derive(Clone, Debug, Default)]
pub struct NaiveCache {
    k: Vec<f32>,
    v: Vec<f32>,
    len: usize,
}
impl NaiveCache {
    pub fn new() -> NaiveCache {
        NaiveCache::default()
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * 4
    }
}

pub struct Mla {
    pub cfg: MlaConfig,
    w: MlaWeights,
    freqs: Vec<f64>,
    rope_ms: f64,
    scale: f32,
}

impl Mla {
    pub fn new(cfg: MlaConfig, w: MlaWeights) -> Result<Mla, MlaError> {
        cfg.validate()?;
        let (h, nh, qh, kv, rp) = (
            cfg.hidden,
            cfg.heads,
            cfg.q_head(),
            cfg.kv_lora_rank,
            cfg.qk_rope,
        );
        let chk = |n: &str, got: usize, want: usize| {
            if got == want {
                Ok(())
            } else {
                Err(MlaError::Shape(format!(
                    "{n}: {got} values, expected {want}"
                )))
            }
        };
        match cfg.q_lora_rank {
            Some(r) => {
                chk("q_a", w.q_a.len(), r * h)?;
                chk("q_a_norm", w.q_a_norm.len(), r)?;
                chk("q_b", w.q_b.len(), nh * qh * r)?;
            }
            None => chk("q_proj", w.q_proj.len(), nh * qh * h)?,
        }
        chk("kv_a", w.kv_a.len(), (kv + rp) * h)?;
        chk("kv_a_norm", w.kv_a_norm.len(), kv)?;
        chk("kv_b", w.kv_b.len(), nh * (cfg.qk_nope + cfg.v_head) * kv)?;
        chk("o", w.o.len(), h * nh * cfg.v_head)?;
        let (freqs, rope_ms) = rope_table(&cfg);
        let mut scale = 1.0 / (qh as f64).sqrt();
        if let Some(y) = &cfg.yarn {
            let m = yarn_mscale(y.factor, y.mscale_all_dim);
            scale *= m * m;
        }
        Ok(Mla {
            cfg,
            w,
            freqs,
            rope_ms,
            scale: scale as f32,
        })
    }

    fn rotate(&self, v: &mut [f32], pos: usize) {
        for (i, f) in self.freqs.iter().enumerate() {
            let (s, c) = crate::simd::rmath::sin_cos64(pos as f64 * f);
            let (c, s) = (c * self.rope_ms, s * self.rope_ms);
            let (a, b) = (v[2 * i] as f64, v[2 * i + 1] as f64);
            v[2 * i] = (a * c - b * s) as f32;
            v[2 * i + 1] = (a * s + b * c) as f32;
        }
    }

    fn check(&self, x: &[f32], pos: usize, cached: usize) -> Result<(), MlaError> {
        if x.len() != self.cfg.hidden {
            return Err(MlaError::Input(format!(
                "input has {} values, expected {}",
                x.len(),
                self.cfg.hidden
            )));
        }
        if pos != cached {
            return Err(MlaError::Input(format!(
                "position {pos} does not follow the {cached} cached tokens"
            )));
        }
        Ok(())
    }

    /// Query for every head, `[heads][nope | rope]`, with the rotary part already rotated.
    fn queries(&self, x: &[f32], pos: usize) -> Vec<f32> {
        let c = &self.cfg;
        let mut q = vec![0.0; c.heads * c.q_head()];
        match c.q_lora_rank {
            Some(r) => {
                let mut a = vec![0.0; r];
                matvec(&self.w.q_a, c.hidden, x, &mut a);
                let mut an = vec![0.0; r];
                rmsnorm(&a, &self.w.q_a_norm, c.eps, &mut an);
                matvec(&self.w.q_b, r, &an, &mut q);
            }
            None => matvec(&self.w.q_proj, c.hidden, x, &mut q),
        }
        for hq in q.chunks_mut(c.q_head()) {
            self.rotate(&mut hq[c.qk_nope..], pos);
        }
        q
    }

    /// Normalised latent and rotated shared key for one token.
    fn compress(&self, x: &[f32], pos: usize) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let mut a = vec![0.0; c.kv_lora_rank + c.qk_rope];
        matvec(&self.w.kv_a, c.hidden, x, &mut a);
        let mut lat = vec![0.0; c.kv_lora_rank];
        rmsnorm(&a[..c.kv_lora_rank], &self.w.kv_a_norm, c.eps, &mut lat);
        let mut pe = a[c.kv_lora_rank..].to_vec();
        self.rotate(&mut pe, pos);
        (lat, pe)
    }

    /// Reference implementation: expand keys and values per head and cache them.
    pub fn forward_naive(
        &self,
        cache: &mut NaiveCache,
        x: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, MlaError> {
        let c = &self.cfg;
        self.check(x, pos, cache.len)?;
        let q = self.queries(x, pos);
        let (lat, pe) = self.compress(x, pos);
        let mut kvb = vec![0.0; c.heads * (c.qk_nope + c.v_head)];
        matvec(&self.w.kv_b, c.kv_lora_rank, &lat, &mut kvb);
        for h in 0..c.heads {
            let hb = &kvb[h * (c.qk_nope + c.v_head)..(h + 1) * (c.qk_nope + c.v_head)];
            cache.k.extend_from_slice(&hb[..c.qk_nope]);
            cache.k.extend_from_slice(&pe);
            cache.v.extend_from_slice(&hb[c.qk_nope..]);
        }
        cache.len += 1;
        let (qh, kd) = (c.q_head(), c.q_head());
        let mut att = vec![0.0; c.heads * c.v_head];
        let mut sc = vec![0.0; cache.len];
        for h in 0..c.heads {
            let qv = &q[h * qh..(h + 1) * qh];
            for (t, s) in sc.iter_mut().enumerate() {
                let k = &cache.k[(t * c.heads + h) * kd..(t * c.heads + h + 1) * kd];
                *s = qv.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * self.scale;
            }
            softmax(&mut sc);
            let out = &mut att[h * c.v_head..(h + 1) * c.v_head];
            for (t, p) in sc.iter().enumerate() {
                let v = &cache.v[(t * c.heads + h) * c.v_head..(t * c.heads + h + 1) * c.v_head];
                for (o, vv) in out.iter_mut().zip(v) {
                    *o += p * vv;
                }
            }
        }
        let mut y = vec![0.0; c.hidden];
        matvec(&self.w.o, c.heads * c.v_head, &att, &mut y);
        Ok(y)
    }

    /// Absorbed implementation: caches only the latent and the rotary key.
    pub fn forward(
        &self,
        cache: &mut MlaCache,
        x: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, MlaError> {
        let c = &self.cfg;
        self.check(x, pos, cache.len)?;
        let q = self.queries(x, pos);
        let (lat, pe) = self.compress(x, pos);
        cache.latent.extend_from_slice(&lat);
        cache.k_pe.extend_from_slice(&pe);
        cache.len += 1;
        let (kv, rp, qh, per) = (c.kv_lora_rank, c.qk_rope, c.q_head(), c.qk_nope + c.v_head);
        let mut att = vec![0.0; c.heads * c.v_head];
        let (mut qa, mut ctx, mut sc) = (vec![0.0; kv], vec![0.0; kv], vec![0.0; cache.len]);
        for h in 0..c.heads {
            let qn = &q[h * qh..h * qh + c.qk_nope];
            let qp = &q[h * qh + c.qk_nope..(h + 1) * qh];
            // q_absorbed = W_UK^T · q_nope, with W_UK the `nope` key rows of this head's kv_b block
            qa.fill(0.0);
            for (i, qi) in qn.iter().enumerate() {
                let row = &self.w.kv_b[(h * per + i) * kv..(h * per + i + 1) * kv];
                for (a, w) in qa.iter_mut().zip(row) {
                    *a += qi * w;
                }
            }
            for (t, s) in sc.iter_mut().enumerate() {
                let l = &cache.latent[t * kv..(t + 1) * kv];
                let k = &cache.k_pe[t * rp..(t + 1) * rp];
                let s1: f32 = qa.iter().zip(l).map(|(a, b)| a * b).sum();
                let s2: f32 = qp.iter().zip(k).map(|(a, b)| a * b).sum();
                *s = (s1 + s2) * self.scale;
            }
            softmax(&mut sc);
            ctx.fill(0.0);
            for (t, p) in sc.iter().enumerate() {
                let l = &cache.latent[t * kv..(t + 1) * kv];
                for (a, b) in ctx.iter_mut().zip(l) {
                    *a += p * b;
                }
            }
            // output = W_UV · ctx, with W_UV the `v` rows of the same block
            let out = &mut att[h * c.v_head..(h + 1) * c.v_head];
            for (j, o) in out.iter_mut().enumerate() {
                let row = &self.w.kv_b
                    [(h * per + c.qk_nope + j) * kv..(h * per + c.qk_nope + j + 1) * kv];
                *o = row.iter().zip(&ctx).map(|(a, b)| a * b).sum();
            }
        }
        let mut y = vec![0.0; c.hidden];
        matvec(&self.w.o, c.heads * c.v_head, &att, &mut y);
        Ok(y)
    }

    /// Prefill: `xs` holds `n · hidden` values for consecutive positions starting at the cache
    /// length. Returns the `n · hidden` outputs; attention is causal by construction.
    pub fn forward_batch(&self, cache: &mut MlaCache, xs: &[f32]) -> Result<Vec<f32>, MlaError> {
        let h = self.cfg.hidden;
        if !xs.len().is_multiple_of(h) {
            return Err(MlaError::Input(
                "batch is not a whole number of tokens".into(),
            ));
        }
        let mut out = Vec::with_capacity(xs.len());
        for x in xs.chunks_exact(h) {
            let pos = cache.len;
            out.extend(self.forward(cache, x, pos)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(s: &mut u64, scale: f32) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
    }
    fn mat(n: usize, s: &mut u64, sc: f32) -> Vec<f32> {
        (0..n).map(|_| rnd(s, sc)).collect()
    }
    fn ones(n: usize, s: &mut u64) -> Vec<f32> {
        (0..n).map(|_| 1.0 + rnd(s, 0.2)).collect()
    }

    fn cfg(q_lora: Option<usize>, yarn: bool) -> MlaConfig {
        MlaConfig {
            hidden: 24,
            heads: 4,
            q_lora_rank: q_lora,
            kv_lora_rank: 10,
            qk_nope: 6,
            qk_rope: 4,
            v_head: 5,
            eps: 1e-6,
            rope_theta: 10_000.0,
            yarn: yarn.then_some(Yarn {
                factor: 8.0,
                original_max_position: 64.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                mscale: 0.7,
                mscale_all_dim: 0.7,
            }),
        }
    }
    fn build(c: &MlaConfig, seed: u64) -> Mla {
        let mut s = seed;
        let (h, nh, qh, kv) = (c.hidden, c.heads, c.qk_nope + c.qk_rope, c.kv_lora_rank);
        let mut w = MlaWeights::default();
        match c.q_lora_rank {
            Some(r) => {
                w.q_a = mat(r * h, &mut s, 0.6);
                w.q_a_norm = ones(r, &mut s);
                w.q_b = mat(nh * qh * r, &mut s, 0.6);
            }
            None => w.q_proj = mat(nh * qh * h, &mut s, 0.6),
        }
        w.kv_a = mat((kv + c.qk_rope) * h, &mut s, 0.6);
        w.kv_a_norm = ones(kv, &mut s);
        w.kv_b = mat(nh * (c.qk_nope + c.v_head) * kv, &mut s, 0.6);
        w.o = mat(h * nh * c.v_head, &mut s, 0.6);
        Mla::new(c.clone(), w).unwrap()
    }

    #[test]
    fn absorbed_equals_naive_over_many_positions_in_every_configuration() {
        for (q, yarn) in [
            (None, false),
            (Some(7), false),
            (None, true),
            (Some(7), true),
        ] {
            let c = cfg(q, yarn);
            let m = build(&c, 0x1234_5678_9abc_def1);
            let (mut a, mut n) = (MlaCache::new(), NaiveCache::new());
            let mut s = 99u64;
            for pos in 0..20 {
                let x = mat(c.hidden, &mut s, 2.0);
                let fast = m.forward(&mut a, &x, pos).unwrap();
                let slow = m.forward_naive(&mut n, &x, pos).unwrap();
                for (i, (p, r)) in fast.iter().zip(&slow).enumerate() {
                    assert!(
                        (p - r).abs() < 1e-4,
                        "q_lora {q:?} yarn {yarn} pos {pos} out {i}: {p} vs {r}"
                    );
                }
            }
        }
    }

    #[test]
    fn earlier_tokens_influence_later_outputs() {
        let c = cfg(Some(7), false);
        let m = build(&c, 7);
        let mut s = 5u64;
        let xs: Vec<Vec<f32>> = (0..3).map(|_| mat(c.hidden, &mut s, 2.0)).collect();
        let last = |first: &[f32]| {
            let mut ca = MlaCache::new();
            m.forward(&mut ca, first, 0).unwrap();
            m.forward(&mut ca, &xs[1], 1).unwrap();
            m.forward(&mut ca, &xs[2], 2).unwrap()
        };
        let (a, b) = (last(&xs[0]), last(&xs[2]));
        assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-3));
    }

    #[test]
    fn batch_prefill_matches_token_by_token() {
        let c = cfg(None, true);
        let m = build(&c, 11);
        let mut s = 3u64;
        let xs = mat(c.hidden * 6, &mut s, 2.0);
        let (mut a, mut b) = (MlaCache::new(), MlaCache::new());
        let batch = m.forward_batch(&mut a, &xs).unwrap();
        let mut one = vec![];
        for (i, x) in xs.chunks_exact(c.hidden).enumerate() {
            one.extend(m.forward(&mut b, x, i).unwrap());
        }
        assert_eq!(batch, one);
        assert!(m.forward_batch(&mut a, &xs[..c.hidden + 1]).is_err());
    }

    #[test]
    fn latent_cache_is_the_expected_fraction_of_the_full_cache() {
        let c = cfg(None, false);
        let m = build(&c, 1);
        let (mut a, mut n) = (MlaCache::new(), NaiveCache::new());
        let mut s = 8u64;
        for pos in 0..9 {
            let x = mat(c.hidden, &mut s, 1.0);
            m.forward(&mut a, &x, pos).unwrap();
            m.forward_naive(&mut n, &x, pos).unwrap();
        }
        assert_eq!(a.bytes(), 9 * (10 + 4) * 4);
        assert_eq!(n.bytes(), 9 * 4 * (6 + 4 + 5) * 4);
        assert_eq!(a.bytes(), 9 * c.latent_bytes_per_token());
        assert_eq!(n.bytes(), 9 * c.full_bytes_per_token());
        assert!(a.bytes() * 4 < n.bytes());
        // DeepSeek-V3 shape: 128 heads, nope 128, rope 64, v 128, latent 512 → 40960/576 ≈ 1/71
        let v3 = MlaConfig {
            hidden: 7168,
            heads: 128,
            q_lora_rank: Some(1536),
            kv_lora_rank: 512,
            qk_nope: 128,
            qk_rope: 64,
            v_head: 128,
            eps: 1e-6,
            rope_theta: 10_000.0,
            yarn: None,
        };
        let ratio = v3.full_bytes_per_token() as f64 / v3.latent_bytes_per_token() as f64;
        assert!((ratio - 40960.0 / 576.0).abs() < 1e-9, "{ratio}");
    }

    #[test]
    fn yarn_changes_the_function_and_scale_follows_mscale() {
        let (plain, yarn) = (cfg(None, false), cfg(None, true));
        let (a, b) = (build(&plain, 3), build(&yarn, 3));
        assert_eq!(a.scale, 1.0 / (10f32).sqrt());
        let ms = 0.1 * 0.7 * 8f64.ln() + 1.0;
        assert!((b.scale as f64 - ms * ms / 10f64.sqrt()).abs() < 1e-6);
        assert_eq!(b.rope_ms, 1.0); // mscale == mscale_all_dim
        let (mut ca, mut cb) = (MlaCache::new(), MlaCache::new());
        let mut s = 21u64;
        let mut differs = false;
        for pos in 0..6 {
            let x = mat(plain.hidden, &mut s, 2.0);
            let (p, q) = (
                a.forward(&mut ca, &x, pos).unwrap(),
                b.forward(&mut cb, &x, pos).unwrap(),
            );
            differs |= p.iter().zip(&q).any(|(x, y)| (x - y).abs() > 1e-4);
        }
        assert!(differs);
    }

    #[test]
    fn yarn_frequency_blend_keeps_high_frequencies_and_interpolates_low_ones() {
        let mut c = cfg(None, true);
        c.qk_rope = 64;
        let (f, _) = rope_table(&c);
        let plain = MlaConfig {
            yarn: None,
            ..c.clone()
        };
        let (p, _) = rope_table(&plain);
        assert!((f[0] - p[0]).abs() < 1e-12); // fastest pair is untouched
        assert!((f[31] - p[31] / 8.0).abs() < 1e-12); // slowest pair is divided by the factor
        assert!(f.windows(2).all(|w| w[0] >= w[1])); // still monotone
    }

    #[test]
    fn rotation_is_a_pure_rotation_and_position_zero_is_identity() {
        let m = build(&cfg(None, false), 5);
        let mut v = vec![0.3, -1.0, 2.0, 0.5];
        let orig = v.clone();
        m.rotate(&mut v, 0);
        assert_eq!(v, orig);
        m.rotate(&mut v, 41);
        let n = |a: &[f32]| a.iter().map(|x| x * x).sum::<f32>();
        assert!((n(&v) - n(&orig)).abs() < 1e-4);
    }

    #[test]
    fn hostile_dimensions_and_inputs_are_errors() {
        let mut c = cfg(None, false);
        c.qk_rope = 5;
        assert!(c.validate().is_err());
        for f in [
            |c: &mut MlaConfig| c.heads = 0,
            |c: &mut MlaConfig| c.kv_lora_rank = 0,
            |c: &mut MlaConfig| c.v_head = usize::MAX,
            |c: &mut MlaConfig| c.hidden = 1 << 30,
            |c: &mut MlaConfig| c.q_lora_rank = Some(0),
            |c: &mut MlaConfig| c.rope_theta = f64::NAN,
        ] {
            let mut c = cfg(None, false);
            f(&mut c);
            assert!(c.validate().is_err());
            assert!(Mla::new(c, MlaWeights::default()).is_err());
        }
        let good = cfg(Some(7), false);
        let mut w = build(&good, 1).w;
        w.kv_b.pop();
        assert!(matches!(Mla::new(good.clone(), w), Err(MlaError::Shape(_))));
        let m = build(&good, 1);
        let mut ca = MlaCache::new();
        assert!(m.forward(&mut ca, &[0.0; 3], 0).is_err());
        assert!(m.forward(&mut ca, &vec![0.0; good.hidden], 4).is_err()); // position must follow the cache
        assert!(m.forward(&mut ca, &vec![0.0; good.hidden], 0).is_ok());
    }

    #[test]
    fn hf_config_parses_deepseek_v3_style_json_and_rejects_nonsense() {
        let j = r#"{"model_type":"deepseek_v3","hidden_size":7168,"num_attention_heads":128,"q_lora_rank":1536,"kv_lora_rank":512,
            "qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,"rms_norm_eps":1e-6,"rope_theta":10000,
            "rope_scaling":{"type":"yarn","factor":40,"original_max_position_embeddings":4096,"beta_fast":32,"beta_slow":1,"mscale":1.0,"mscale_all_dim":1.0}}"#;
        let c = MlaConfig::from_hf_config(j).unwrap();
        assert_eq!(
            (c.heads, c.q_lora_rank, c.kv_lora_rank, c.qk_rope),
            (128, Some(1536), 512, 64)
        );
        assert_eq!(c.yarn.as_ref().unwrap().factor, 40.0);
        let v2lite = r#"{"hidden_size":2048,"num_attention_heads":16,"q_lora_rank":null,"kv_lora_rank":512,"qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128}"#;
        assert_eq!(MlaConfig::from_hf_config(v2lite).unwrap().q_lora_rank, None);
        let wrapped = format!(r#"{{"text_config":{v2lite}}}"#);
        assert!(MlaConfig::from_hf_config(&wrapped).is_ok());
        assert!(MlaConfig::from_hf_config("[]").is_err());
        assert!(MlaConfig::from_hf_config("{").is_err());
        assert!(MlaConfig::from_hf_config(r#"{"hidden_size":8}"#).is_err());
    }

    #[test]
    fn tensor_names_and_split_kv_b_join() {
        assert_eq!(
            names::hf(3)[4],
            "model.layers.3.self_attn.kv_a_proj_with_mqa.weight"
        );
        assert_eq!(names::gguf(3)[6], "blk.3.attn_k_b.weight");
        let c = cfg(None, false);
        let (nh, n, v, kv) = (c.heads, c.qk_nope, c.v_head, c.kv_lora_rank);
        let mut s = 4u64;
        let (k_b, v_b) = (mat(nh * n * kv, &mut s, 1.0), mat(nh * v * kv, &mut s, 1.0));
        let j = join_kv_b(&c, &k_b, &v_b).unwrap();
        assert_eq!(j.len(), nh * (n + v) * kv);
        assert_eq!(&j[..n * kv], &k_b[..n * kv]);
        assert_eq!(&j[n * kv..(n + v) * kv], &v_b[..v * kv]);
        assert_eq!(&j[(n + v) * kv..(n + v + n) * kv], &k_b[n * kv..2 * n * kv]);
        assert!(join_kv_b(&c, &k_b[1..], &v_b).is_err());
    }
}
