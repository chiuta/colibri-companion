//! The older decoder families, run from a GGUF file: GPT-2, GPT-J, GPT-NeoX (Pythia), Falcon,
//! BLOOM and MPT.
//!
//! They share one shape: LayerNorm, biases in most places, a plain (not gated) MLP and, depending on
//! the family, a fused query/key/value tensor, learned / rotary / ALiBi positions and a parallel
//! attention + MLP residual. This module is one configurable decoder with its own cache, reached
//! through `family::load_any`.
//!
//! Written from the published architectures. Checked on small synthetic models (including that the
//! three fused-QKV layouts give identical results for the same weights); **not** compared with real
//! checkpoints. OPT is not here: it has no GGUF conversion.
use crate::accel::{Accel, AttachReport, Item, key_of};
use crate::arch::Act;
use crate::backend::Backend;
use crate::gguf::Gguf;
use crate::loader::{LoadOptions, TensorSource, WeightFile, load_weight};
use crate::model::{ModelError, Weight, activate};
use std::sync::Arc;

#[derive(Debug)]
pub enum LegacyError {
    Config(String),
    Load(String),
}
impl std::fmt::Display for LegacyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LegacyError::Config(s) | LegacyError::Load(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for LegacyError {}
fn load_err(e: impl std::fmt::Display) -> LegacyError {
    LegacyError::Load(e.to_string())
}

/// GGUF architecture names this module runs.
pub fn handles(arch: &str) -> bool {
    matches!(
        arch,
        "gpt2" | "gptj" | "gptneox" | "gpt_neox" | "falcon" | "bloom" | "mpt"
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fam {
    Gpt2,
    GptJ,
    NeoX,
    Falcon,
    Bloom,
    Mpt,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pos {
    /// A table of position vectors added to the embedding.
    Learned,
    /// Rotary, rotating the pairs `(i, i + rot/2)`.
    RopeHalf,
    /// Rotary, rotating the pairs `(2i, 2i + 1)`.
    RopePairs,
    /// Linear attention biases with this maximum bias.
    Alibi(f32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Qkv {
    /// Three tensors.
    Separate,
    /// One tensor, `[all q; all k; all v]`.
    Contiguous,
    /// One tensor, per key/value head `[its q heads, k, v]` (NeoX, BLOOM, Falcon).
    Grouped,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LegacyConfig {
    pub family: Fam,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub context: usize,
    pub eps: f32,
    pub pos: Pos,
    pub act: Act,
    pub parallel: bool,
    pub qkv: Qkv,
    pub rot: usize,
    pub theta: f64,
    pub clamp: f32,
}

impl LegacyConfig {
    pub fn from_gguf(g: &Gguf) -> Result<LegacyConfig, LegacyError> {
        let arch = g.architecture().unwrap_or("").to_string();
        let family = match arch.as_str() {
            "gpt2" => Fam::Gpt2,
            "gptj" => Fam::GptJ,
            "gptneox" | "gpt_neox" => Fam::NeoX,
            "falcon" => Fam::Falcon,
            "bloom" => Fam::Bloom,
            "mpt" => Fam::Mpt,
            o => return Err(LegacyError::Config(format!("`{o}` is not a legacy family"))),
        };
        let need = |k: &str| {
            g.get_u64(k)
                .filter(|v| *v > 0)
                .ok_or_else(|| LegacyError::Config(format!("{arch}.{k} is missing")))
        };
        let layers = need("block_count")? as usize;
        let hidden = need("embedding_length")? as usize;
        let heads = need("attention.head_count")? as usize;
        if layers > 512 || hidden > 1 << 17 || heads > 1 << 12 {
            return Err(LegacyError::Config("implausible model size".into()));
        }
        let kv_heads = g
            .get_u64("attention.head_count_kv")
            .filter(|v| *v > 0)
            .map_or(heads, |v| v as usize);
        if !hidden.is_multiple_of(heads) || !heads.is_multiple_of(kv_heads) {
            return Err(LegacyError::Config(format!(
                "{hidden} hidden values or {heads} heads do not divide into {kv_heads} key/value heads"
            )));
        }
        let head_dim = hidden / heads;
        let ffn = g
            .get_u64("feed_forward_length")
            .map_or(4 * hidden, |v| v as usize);
        let rope_dim = g
            .get_u64("rope.dimension_count")
            .map_or(head_dim, |v| (v as usize).min(head_dim));
        let max_bias = g.get_f64("attention.max_alibi_bias").unwrap_or(8.0) as f32;
        let (pos, act, qkv, parallel) = match family {
            Fam::Gpt2 => (Pos::Learned, Act::GeluTanh, Qkv::Contiguous, false),
            Fam::GptJ => (Pos::RopePairs, Act::GeluTanh, Qkv::Separate, true),
            Fam::NeoX => (
                Pos::RopeHalf,
                Act::Gelu,
                Qkv::Grouped,
                g.get_u64("use_parallel_residual").is_none_or(|v| v != 0),
            ),
            Fam::Falcon => (Pos::RopeHalf, Act::Gelu, Qkv::Grouped, true),
            Fam::Bloom => (Pos::Alibi(8.0), Act::GeluTanh, Qkv::Grouped, false),
            Fam::Mpt => (Pos::Alibi(max_bias), Act::Gelu, Qkv::Contiguous, false),
        };
        let c = LegacyConfig {
            family,
            layers,
            hidden,
            heads,
            kv_heads,
            head_dim,
            ffn,
            context: g.get_u64("context_length").unwrap_or(2048) as usize,
            eps: g.get_f64("attention.layer_norm_epsilon").unwrap_or(1e-5) as f32,
            pos,
            act,
            parallel,
            qkv,
            rot: rope_dim & !1,
            theta: g.get_f64("rope.freq_base").unwrap_or(10_000.0),
            clamp: g.get_f64("attention.clamp_kqv").unwrap_or(0.0) as f32,
        };
        if matches!(c.pos, Pos::RopeHalf | Pos::RopePairs) && c.rot == 0 {
            return Err(LegacyError::Config("rope.dimension_count is zero".into()));
        }
        Ok(c)
    }
}

type Qkvs = (Vec<f32>, Vec<f32>, Vec<f32>);

struct Norm {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
}

enum Proj {
    Fused(Weight),
    Split(Weight, Weight, Weight),
}

struct Layer {
    n1: Norm,
    n2: Option<Norm>,
    qkv: Proj,
    qkv_b: Option<Vec<f32>>,
    wo: Weight,
    bo: Option<Vec<f32>>,
    up: Weight,
    bup: Option<Vec<f32>>,
    down: Weight,
    bdown: Option<Vec<f32>>,
}

pub struct LegacyModel {
    cfg: LegacyConfig,
    embed: Weight,
    pos_embed: Option<Vec<f32>>,
    embed_norm: Option<Norm>,
    layers: Vec<Layer>,
    out_norm: Norm,
    head: Option<Weight>,
    head_bias: Option<Vec<f32>>,
    vocab: usize,
    accel: Option<Accel>,
}

#[derive(Default, Clone)]
struct Cache {
    k: Vec<f32>,
    v: Vec<f32>,
}
pub struct LegacyState {
    caches: Vec<Cache>,
    pub pos: usize,
}
impl LegacyState {
    pub fn bytes(&self) -> usize {
        self.caches
            .iter()
            .map(|c| (c.k.len() + c.v.len()) * 4)
            .sum()
    }
}

fn mat(
    src: &dyn TensorSource,
    name: &str,
    cols: usize,
    o: LoadOptions,
) -> Result<(Weight, usize), LegacyError> {
    let w = load_weight(src, name, o).map_err(load_err)?;
    let len = w.len();
    if cols == 0 || !len.is_multiple_of(cols) {
        return Err(LegacyError::Load(format!(
            "{name}: {len} values do not fill rows of {cols}"
        )));
    }
    Ok((w, len / cols))
}
fn opt_vec(src: &dyn TensorSource, name: &str, n: usize) -> Result<Option<Vec<f32>>, LegacyError> {
    if !src.has(name) {
        return Ok(None);
    }
    let v = src.read(name).map_err(load_err)?;
    if v.len() != n {
        return Err(LegacyError::Load(format!(
            "{name}: {} values, expected {n}",
            v.len()
        )));
    }
    Ok(Some(v))
}
fn norm(src: &dyn TensorSource, base: &str, n: usize) -> Result<Option<Norm>, LegacyError> {
    let Some(w) = opt_vec(src, &format!("{base}.weight"), n)? else {
        return Ok(None);
    };
    let b = opt_vec(src, &format!("{base}.bias"), n)?;
    Ok(Some(Norm { w, b }))
}

fn layer_norm(x: &[f32], n: &Norm, eps: f32) -> Vec<f32> {
    let len = x.len() as f32;
    let mean = x.iter().sum::<f32>() / len;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / len;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(&n.w)
        .enumerate()
        .map(|(i, (v, w))| (v - mean) * inv * w + n.b.as_ref().map_or(0.0, |b| b[i]))
        .collect()
}

fn add_bias(v: &mut [f32], b: &Option<Vec<f32>>) {
    if let Some(b) = b {
        v.iter_mut().zip(b).for_each(|(a, b)| *a += b);
    }
}

/// ALiBi slopes for `n` heads.
pub fn alibi_slopes(n: usize, max_bias: f32) -> Vec<f32> {
    let p = if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two() / 2
    };
    let m0 = 2f32.powf(-max_bias / p as f32);
    let m1 = 2f32.powf(-(max_bias / 2.0) / p as f32);
    (0..n)
        .map(|i| {
            if i < p {
                m0.powi(i as i32 + 1)
            } else {
                m1.powi(2 * (i - p) as i32 + 1)
            }
        })
        .collect()
}

impl LegacyModel {
    pub fn load(
        cfg: LegacyConfig,
        src: &Arc<dyn WeightFile>,
        o: LoadOptions,
    ) -> Result<LegacyModel, LegacyError> {
        let s: &dyn TensorSource = src.as_ref();
        let h = cfg.hidden;
        let (qd, kd) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        let (embed, vocab) = mat(s, "token_embd.weight", h, o)?;
        let pos_embed = if cfg.pos == Pos::Learned {
            Some(s.read("position_embd.weight").map_err(load_err)?)
        } else {
            None
        };
        if let Some(p) = &pos_embed
            && (p.is_empty() || !p.len().is_multiple_of(h))
        {
            return Err(LegacyError::Load(
                "position_embd.weight has the wrong size".into(),
            ));
        }
        let embed_norm = norm(s, "token_embd_norm", h)?;
        let out_norm = norm(s, "output_norm", h)?
            .ok_or_else(|| LegacyError::Load("output_norm.weight is missing".into()))?;
        let head = if s.has("output.weight") {
            Some(mat(s, "output.weight", h, o)?.0)
        } else {
            None
        };
        let head_bias = opt_vec(s, "output.bias", vocab)?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let n = |p: &str| format!("blk.{i}.{p}");
            let n1 = norm(s, &n("attn_norm"), h)?
                .ok_or_else(|| LegacyError::Load(format!("blk.{i}.attn_norm.weight is missing")))?;
            // Falcon-40B style files carry a second norm for the MLP branch; a normal parallel
            // block (GPT-J, Falcon-7B) feeds both branches from `attn_norm`.
            let n2 = norm(s, &n("ffn_norm"), h)?.or(norm(s, &n("attn_norm_2"), h)?);
            let qkv_dim = qd + 2 * kd;
            let (qkv, qkv_b) = if s.has(&n("attn_qkv.weight")) {
                let (w, rows) = mat(s, &n("attn_qkv.weight"), h, o)?;
                if rows != qkv_dim {
                    return Err(LegacyError::Load(format!(
                        "blk.{i}.attn_qkv.weight has {rows} rows, expected {qkv_dim}"
                    )));
                }
                (Proj::Fused(w), opt_vec(s, &n("attn_qkv.bias"), qkv_dim)?)
            } else {
                let q = mat(s, &n("attn_q.weight"), h, o)?;
                let k = mat(s, &n("attn_k.weight"), h, o)?;
                let v = mat(s, &n("attn_v.weight"), h, o)?;
                if (q.1, k.1, v.1) != (qd, kd, kd) {
                    return Err(LegacyError::Load(format!(
                        "blk.{i}: attention matrices have {}/{}/{} rows, expected {qd}/{kd}/{kd}",
                        q.1, k.1, v.1
                    )));
                }
                let bq = opt_vec(s, &n("attn_q.bias"), qd)?;
                let bk = opt_vec(s, &n("attn_k.bias"), kd)?;
                let bv = opt_vec(s, &n("attn_v.bias"), kd)?;
                let b = match (bq, bk, bv) {
                    (Some(mut q), Some(k), Some(v)) => {
                        q.extend(k);
                        q.extend(v);
                        Some(q)
                    }
                    _ => None,
                };
                (Proj::Split(q.0, k.0, v.0), b)
            };
            let (wo, orows) = mat(s, &n("attn_output.weight"), qd, o)?;
            let (up, ff) = mat(s, &n("ffn_up.weight"), h, o)?;
            let (down, drows) = mat(s, &n("ffn_down.weight"), ff, o)?;
            if orows != h || drows != h {
                return Err(LegacyError::Load(format!(
                    "blk.{i}: output matrices have {orows}/{drows} rows, expected {h}"
                )));
            }
            if !cfg.parallel && n2.is_none() {
                return Err(LegacyError::Load(format!(
                    "blk.{i}.ffn_norm.weight is missing"
                )));
            }
            layers.push(Layer {
                n1,
                n2,
                qkv,
                qkv_b,
                wo,
                bo: opt_vec(s, &n("attn_output.bias"), h)?,
                up,
                bup: opt_vec(s, &n("ffn_up.bias"), ff)?,
                down,
                bdown: opt_vec(s, &n("ffn_down.bias"), h)?,
            });
        }
        Ok(LegacyModel {
            cfg,
            embed,
            pos_embed,
            embed_norm,
            layers,
            out_norm,
            head,
            head_bias,
            vocab,
            accel: None,
        })
    }

    fn mv(&self, w: &Weight, cols: usize, x: &[f32], out: &mut [f32]) -> Result<(), ModelError> {
        if let Some(a) = &self.accel {
            let head = self.head.as_ref().unwrap_or(&self.embed);
            if a.try_matvec(key_of(w), std::ptr::eq(w, head), cols, x, out) {
                return Ok(());
            }
        }
        w.matvec(cols, x, out)
    }

    /// Why the accelerator was switched off while running, if it was.
    pub fn accel_failure(&self) -> Option<String> {
        self.accel.as_ref().and_then(Accel::failure)
    }

    /// Put the big matrices on `be` (see [`Accel::build`]); never fails, the CPU stays the fall-back.
    pub fn attach_backend(&mut self, be: Arc<dyn Backend>) -> AttachReport {
        self.accel = None;
        let h = self.cfg.hidden;
        let qd = self.cfg.heads * self.cfg.head_dim;
        let mut items = vec![Item::new(
            self.head.as_ref().unwrap_or(&self.embed),
            h,
            true,
        )];
        for l in &self.layers {
            match &l.qkv {
                Proj::Fused(w) => items.push(Item::new(w, h, false)),
                Proj::Split(a, b, c) => {
                    items.extend([a, b, c].map(|w| Item::new(w, h, false)));
                }
            }
            items.push(Item::new(&l.wo, qd, false));
            items.push(Item::new(&l.up, h, false));
            items.push(Item::new(&l.down, l.up.len() / h, false));
        }
        let (acc, rep) = Accel::build(be, &items);
        self.accel = acc;
        rep
    }

    pub fn config(&self) -> &LegacyConfig {
        &self.cfg
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn new_state(&self) -> LegacyState {
        LegacyState {
            caches: vec![Cache::default(); self.cfg.layers],
            pos: 0,
        }
    }

    fn rope(&self, x: &mut [f32], heads: usize, pos: usize) {
        let c = &self.cfg;
        let (hd, rot) = (c.head_dim, c.rot.min(c.head_dim));
        for h in 0..heads {
            let v = &mut x[h * hd..(h + 1) * hd];
            for i in 0..rot / 2 {
                let f = c.theta.powf(-2.0 * i as f64 / rot as f64);
                let (s, co) = (pos as f64 * f).sin_cos();
                let (s, co) = (s as f32, co as f32);
                let (a, b) = if c.pos == Pos::RopeHalf {
                    (i, i + rot / 2)
                } else {
                    (2 * i, 2 * i + 1)
                };
                let (x0, x1) = (v[a], v[b]);
                v[a] = x0 * co - x1 * s;
                v[b] = x0 * s + x1 * co;
            }
        }
    }

    /// `(q, k, v)` of one token.
    fn qkv(&self, l: &Layer, a: &[f32]) -> Result<Qkvs, ModelError> {
        let c = &self.cfg;
        let h = c.hidden;
        let (nh, nkv, hd) = (c.heads, c.kv_heads, c.head_dim);
        let (qd, kd) = (nh * hd, nkv * hd);
        match &l.qkv {
            Proj::Split(wq, wk, wv) => {
                let (mut q, mut k, mut v) = (vec![0.0; qd], vec![0.0; kd], vec![0.0; kd]);
                self.mv(wq, h, a, &mut q)?;
                self.mv(wk, h, a, &mut k)?;
                self.mv(wv, h, a, &mut v)?;
                if let Some(b) = &l.qkv_b {
                    q.iter_mut().zip(&b[..qd]).for_each(|(a, b)| *a += b);
                    k.iter_mut().zip(&b[qd..qd + kd]).for_each(|(a, b)| *a += b);
                    v.iter_mut().zip(&b[qd + kd..]).for_each(|(a, b)| *a += b);
                }
                Ok((q, k, v))
            }
            Proj::Fused(w) => {
                let mut f = vec![0.0; qd + 2 * kd];
                self.mv(w, h, a, &mut f)?;
                add_bias(&mut f, &l.qkv_b);
                let (mut q, mut k, mut v) = (
                    Vec::with_capacity(qd),
                    Vec::with_capacity(kd),
                    Vec::with_capacity(kd),
                );
                match c.qkv {
                    Qkv::Grouped => {
                        let g = nh / nkv;
                        for kvh in 0..nkv {
                            let base = kvh * (g + 2) * hd;
                            q.extend_from_slice(&f[base..base + g * hd]);
                            k.extend_from_slice(&f[base + g * hd..base + (g + 1) * hd]);
                            v.extend_from_slice(&f[base + (g + 1) * hd..base + (g + 2) * hd]);
                        }
                    }
                    _ => {
                        q.extend_from_slice(&f[..qd]);
                        k.extend_from_slice(&f[qd..qd + kd]);
                        v.extend_from_slice(&f[qd + kd..]);
                    }
                }
                Ok((q, k, v))
            }
        }
    }

    fn mlp(&self, l: &Layer, x: &[f32]) -> Result<Vec<f32>, ModelError> {
        let ff = l.up.len() / self.cfg.hidden;
        let mut u = vec![0.0; ff];
        self.mv(&l.up, self.cfg.hidden, x, &mut u)?;
        add_bias(&mut u, &l.bup);
        u.iter_mut().for_each(|v| *v = activate(self.cfg.act, *v));
        let mut o = vec![0.0; self.cfg.hidden];
        self.mv(&l.down, ff, &u, &mut o)?;
        add_bias(&mut o, &l.bdown);
        Ok(o)
    }

    fn step(&self, st: &mut LegacyState, token: usize, want: bool) -> Result<Vec<f32>, ModelError> {
        let c = &self.cfg;
        if token >= self.vocab {
            return Err(ModelError::Token(token));
        }
        let (h, nh, nkv, hd) = (c.hidden, c.heads, c.kv_heads, c.head_dim);
        let pos = st.pos;
        if pos >= c.context.max(1) && c.pos == Pos::Learned {
            return Err(ModelError::Shape(format!(
                "position {pos} is past the {} the model has position vectors for",
                c.context
            )));
        }
        let mut x = vec![0.0; h];
        let mut tmp = vec![];
        self.embed.row_into(token, h, &mut x, &mut tmp)?;
        if let Some(p) = &self.pos_embed {
            let rows = p.len() / h;
            let r = pos.min(rows - 1);
            x.iter_mut()
                .zip(&p[r * h..(r + 1) * h])
                .for_each(|(a, b)| *a += b);
        }
        if let Some(n) = &self.embed_norm {
            x = layer_norm(&x, n, c.eps);
        }
        let slopes = match c.pos {
            Pos::Alibi(m) => alibi_slopes(nh, m),
            _ => vec![],
        };
        let group = nh / nkv;
        for (li, l) in self.layers.iter().enumerate() {
            let a = layer_norm(&x, &l.n1, c.eps);
            let (mut q, mut k, mut v) = self.qkv(l, &a)?;
            if c.clamp > 0.0 {
                for t in [&mut q, &mut k, &mut v] {
                    t.iter_mut().for_each(|e| *e = e.clamp(-c.clamp, c.clamp));
                }
            }
            if matches!(c.pos, Pos::RopeHalf | Pos::RopePairs) {
                self.rope(&mut q, nh, pos);
                self.rope(&mut k, nkv, pos);
            }
            let cache = &mut st.caches[li];
            cache.k.extend_from_slice(&k);
            cache.v.extend_from_slice(&v);
            let stored = cache.k.len() / (nkv * hd);
            let scale = 1.0 / (hd as f32).sqrt();
            let mut att = vec![0.0f32; nh * hd];
            let mut sc = vec![0.0f32; stored];
            for head in 0..nh {
                let kvh = head / group;
                let qh = &q[head * hd..(head + 1) * hd];
                for (t, s) in sc.iter_mut().enumerate() {
                    let kk = &cache.k[(t * nkv + kvh) * hd..(t * nkv + kvh + 1) * hd];
                    *s = qh.iter().zip(kk).map(|(a, b)| a * b).sum::<f32>() * scale;
                    if let Some(m) = slopes.get(head) {
                        *s += m * (t as f32 - pos as f32);
                    }
                }
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut z = 0.0;
                for s in sc.iter_mut() {
                    *s = (*s - m).exp();
                    z += *s;
                }
                let o = &mut att[head * hd..(head + 1) * hd];
                for (t, s) in sc.iter().enumerate() {
                    let vv = &cache.v[(t * nkv + kvh) * hd..(t * nkv + kvh + 1) * hd];
                    let w = s / z;
                    o.iter_mut().zip(vv).for_each(|(o, v)| *o += w * v);
                }
            }
            let mut proj = vec![0.0; h];
            self.mv(&l.wo, nh * hd, &att, &mut proj)?;
            add_bias(&mut proj, &l.bo);
            if c.parallel {
                let m_in = match &l.n2 {
                    Some(n) => layer_norm(&x, n, c.eps),
                    None => a,
                };
                let f = self.mlp(l, &m_in)?;
                x.iter_mut()
                    .zip(proj.iter().zip(&f))
                    .for_each(|(x, (p, f))| *x += p + f);
            } else {
                x.iter_mut().zip(&proj).for_each(|(x, p)| *x += p);
                let n2 =
                    l.n2.as_ref()
                        .ok_or_else(|| ModelError::Shape("no ffn_norm".into()))?;
                let f = self.mlp(l, &layer_norm(&x, n2, c.eps))?;
                x.iter_mut().zip(&f).for_each(|(x, f)| *x += f);
            }
        }
        st.pos += 1;
        if !want {
            return Ok(vec![]);
        }
        let xn = layer_norm(&x, &self.out_norm, c.eps);
        let mut logits = vec![0.0; self.vocab];
        self.mv(
            self.head.as_ref().unwrap_or(&self.embed),
            h,
            &xn,
            &mut logits,
        )?;
        add_bias(&mut logits, &self.head_bias);
        Ok(logits)
    }

    /// Feed `tokens`, return the logits of the last one. After an `Err` the state must be discarded.
    pub fn forward_batch(
        &self,
        st: &mut LegacyState,
        tokens: &[usize],
    ) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::Shape("no tokens".into()));
        }
        let mut out = vec![];
        for (i, t) in tokens.iter().enumerate() {
            out = self.step(st, *t, i + 1 == tokens.len())?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::{self, AnyModel};
    use crate::gguf_write::{GgufWriter, MetaValue as M, TensorDesc};
    use crate::quant::DType;
    use std::sync::atomic::{AtomicU32, Ordering};

    const H: usize = 16;
    const NH: usize = 4;
    const HD: usize = 4;
    const FF: usize = 24;
    const V: usize = 30;
    const L: usize = 2;

    struct Rng(u64);
    impl Rng {
        fn mat(&mut self, n: usize) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 ^= self.0 << 13;
                    self.0 ^= self.0 >> 7;
                    self.0 ^= self.0 << 17;
                    ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5
                })
                .collect()
        }
    }

    /// Weights shared by the layout tests: one q, k, v per layer, packed as asked.
    fn write(arch: &str, kv: usize, qkv: Qkv, parallel: bool, path: &std::path::Path) {
        let mut r = Rng(0x9e3779b97f4a7c15);
        let mut t: Vec<(String, Vec<u64>, Vec<f32>)> = vec![];
        let mut f = |n: &str, s: &[usize], d: Vec<f32>| {
            assert_eq!(s.iter().product::<usize>(), d.len(), "{n}");
            t.push((n.into(), s.iter().map(|x| *x as u64).collect(), d));
        };
        let ones = |n: usize| vec![1.0f32; n];
        let (qd, kd) = (NH * HD, kv * HD);
        f("token_embd.weight", &[V, H], r.mat(V * H));
        if arch == "gpt2" {
            f("position_embd.weight", &[32, H], r.mat(32 * H));
        }
        if arch == "bloom" {
            f("token_embd_norm.weight", &[H], ones(H));
            f("token_embd_norm.bias", &[H], r.mat(H));
        }
        f("output_norm.weight", &[H], ones(H));
        f("output_norm.bias", &[H], r.mat(H));
        for i in 0..L {
            let n = |p: &str| format!("blk.{i}.{p}");
            f(&n("attn_norm.weight"), &[H], ones(H));
            f(&n("attn_norm.bias"), &[H], r.mat(H));
            if !parallel || arch == "gptneox" {
                f(&n("ffn_norm.weight"), &[H], ones(H));
                f(&n("ffn_norm.bias"), &[H], r.mat(H));
            }
            let (q, k, v) = (r.mat(qd * H), r.mat(kd * H), r.mat(kd * H));
            match qkv {
                Qkv::Separate => {
                    f(&n("attn_q.weight"), &[qd, H], q);
                    f(&n("attn_k.weight"), &[kd, H], k);
                    f(&n("attn_v.weight"), &[kd, H], v);
                }
                Qkv::Contiguous => {
                    let mut all = q;
                    all.extend(k);
                    all.extend(v);
                    f(&n("attn_qkv.weight"), &[qd + 2 * kd, H], all);
                }
                Qkv::Grouped => {
                    let g = NH / kv;
                    let mut all = vec![];
                    for kvh in 0..kv {
                        all.extend(&q[kvh * g * HD * H..(kvh + 1) * g * HD * H]);
                        all.extend(&k[kvh * HD * H..(kvh + 1) * HD * H]);
                        all.extend(&v[kvh * HD * H..(kvh + 1) * HD * H]);
                    }
                    f(&n("attn_qkv.weight"), &[qd + 2 * kd, H], all);
                }
            }
            f(&n("attn_output.weight"), &[H, qd], r.mat(H * qd));
            f(&n("attn_output.bias"), &[H], r.mat(H));
            f(&n("ffn_up.weight"), &[FF, H], r.mat(FF * H));
            f(&n("ffn_up.bias"), &[FF], r.mat(FF));
            f(&n("ffn_down.weight"), &[H, FF], r.mat(H * FF));
            f(&n("ffn_down.bias"), &[H], r.mat(H));
        }
        let meta: Vec<(String, M)> = vec![
            ("general.architecture".into(), M::str(arch)),
            (format!("{arch}.block_count"), M::U32(L as u32)),
            (format!("{arch}.context_length"), M::U32(32)),
            (format!("{arch}.embedding_length"), M::U32(H as u32)),
            (format!("{arch}.feed_forward_length"), M::U32(FF as u32)),
            (format!("{arch}.attention.head_count"), M::U32(NH as u32)),
            (format!("{arch}.attention.head_count_kv"), M::U32(kv as u32)),
            (format!("{arch}.attention.layer_norm_epsilon"), M::F32(1e-5)),
            (format!("{arch}.rope.dimension_count"), M::U32(HD as u32)),
            (
                format!("{arch}.use_parallel_residual"),
                M::U32(parallel as u32),
            ),
        ];
        let descs: Vec<TensorDesc> = t
            .iter()
            .map(|(n, s, _)| TensorDesc::new(n.clone(), s, DType::F32))
            .collect();
        let file = std::fs::File::create(path).unwrap();
        GgufWriter::write_all(file, &meta, descs, |i, _| {
            Ok(t[i].2.iter().flat_map(|x| x.to_le_bytes()).collect())
        })
        .unwrap();
    }

    fn open(arch: &str, kv: usize, qkv: Qkv, parallel: bool) -> LegacyModel {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "legacy-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("m.gguf");
        write(arch, kv, qkv, parallel, &p);
        let m = match family::load_any(&p).unwrap() {
            AnyModel::Legacy(m) => *m,
            _ => panic!("not a legacy model"),
        };
        let _ = std::fs::remove_dir_all(&d);
        m
    }
    fn run(m: &LegacyModel, toks: &[usize]) -> Vec<f32> {
        m.forward_batch(&mut m.new_state(), toks).unwrap()
    }
    const TOKS: [usize; 7] = [3, 9, 1, 22, 5, 5, 17];

    #[test]
    fn every_family_runs_finite_and_deterministic() {
        for (a, kv, q, par) in [
            ("gpt2", NH, Qkv::Contiguous, false),
            ("gptj", NH, Qkv::Separate, true),
            ("gptneox", NH, Qkv::Grouped, true),
            ("falcon", 1, Qkv::Grouped, true),
            ("bloom", NH, Qkv::Grouped, false),
            ("mpt", NH, Qkv::Contiguous, false),
        ] {
            let m = open(a, kv, q, par);
            let l = run(&m, &TOKS);
            assert_eq!(l.len(), V, "{a}");
            assert!(l.iter().all(|v| v.is_finite()), "{a}");
            assert!(l.iter().any(|v| v.abs() > 1e-4), "{a}");
            assert_eq!(l, run(&m, &TOKS), "{a}");
        }
    }

    #[test]
    fn batch_equals_token_by_token() {
        for a in ["gpt2", "gptneox", "bloom"] {
            let m = open(
                a,
                NH,
                if a == "gpt2" {
                    Qkv::Contiguous
                } else {
                    Qkv::Grouped
                },
                a == "gptneox",
            );
            let whole = run(&m, &TOKS);
            let mut st = m.new_state();
            let mut last = vec![];
            for t in TOKS {
                last = m.forward_batch(&mut st, &[t]).unwrap();
            }
            assert_eq!(whole, last, "{a}");
            assert_eq!(st.pos, TOKS.len());
        }
    }

    #[test]
    fn fused_layouts_give_the_same_model() {
        let close = |a: &[f32], b: &[f32]| {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-4, "{x} {y}");
            }
        };
        // the same weights stored as three tensors or as the family's fused tensor
        for (arch, kv, fused, par) in [
            ("gptneox", NH, Qkv::Grouped, true),
            ("falcon", 1, Qkv::Grouped, true),
            ("falcon", 2, Qkv::Grouped, true),
            ("bloom", NH, Qkv::Grouped, false),
            ("gpt2", NH, Qkv::Contiguous, false),
            ("mpt", NH, Qkv::Contiguous, false),
        ] {
            let a = open(arch, kv, fused, par);
            let b = open(arch, kv, Qkv::Separate, par);
            close(&run(&a, &TOKS), &run(&b, &TOKS));
        }
    }

    #[test]
    fn positions_matter_and_alibi_slopes_are_right() {
        let m = open("gpt2", NH, Qkv::Contiguous, false);
        assert_ne!(run(&m, &[3, 4]), run(&m, &[4, 3]));
        let s = alibi_slopes(8, 8.0);
        assert!((s[0] - 0.5).abs() < 1e-6 && (s[7] - 1.0 / 256.0).abs() < 1e-8);
        let s12 = alibi_slopes(12, 8.0);
        assert_eq!(s12.len(), 12);
        assert!(s12.iter().all(|v| *v > 0.0 && *v < 1.0));
        let m = open("bloom", NH, Qkv::Grouped, false);
        assert_ne!(run(&m, &[3, 4, 5]), run(&m, &[3, 4, 6]));
    }

    #[test]
    fn bad_input_is_an_error_not_a_panic() {
        let m = open("gptj", NH, Qkv::Separate, true);
        assert!(m.forward_batch(&mut m.new_state(), &[]).is_err());
        assert!(m.forward_batch(&mut m.new_state(), &[V]).is_err());
        let g = open("gpt2", NH, Qkv::Contiguous, false);
        let mut st = g.new_state();
        let long: Vec<usize> = (0..40).map(|i| i % V).collect();
        assert!(g.forward_batch(&mut st, &long).is_err());
    }

    #[test]
    fn accelerator_matches_and_falls_back() {
        use crate::accel::testbe::Odd;
        let mut m = open("gptneox", NH, Qkv::Grouped, true);
        let cpu = run(&m, &TOKS);
        let rep = m.attach_backend(Odd::new(usize::MAX, 1.0, None));
        assert!(rep.active, "{rep:?}");
        for (x, y) in run(&m, &TOKS).iter().zip(&cpu) {
            assert!((x - y).abs() < 1e-4);
        }
        let mut m = open("gpt2", NH, Qkv::Contiguous, false);
        let cpu = run(&m, &TOKS);
        m.attach_backend(Odd::new(6, 1.0, None));
        for (x, y) in run(&m, &TOKS).iter().zip(&cpu) {
            assert!((x - y).abs() < 1e-4);
        }
        assert!(m.accel_failure().is_some());
        let rep = m.attach_backend(Odd::new(usize::MAX, 2.0, None));
        assert!(!rep.active);
    }
}
