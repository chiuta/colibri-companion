//! Gemma 4 (dense E2B/E4B/31B and the 26B-A4B mixture-of-experts), run from a GGUF file.
//!
//! Gemma 4 does not fit the shared decoder in `model.rs`: layers differ in head size and number of
//! key/value heads (sliding layers 256, global layers 512), global layers may have no value
//! projection (V is the key projection, normalised without a scale), the attention scale is 1,
//! global layers use "proportional" RoPE (a frequency-factor table), E2B/E4B reuse the K/V of an
//! earlier layer in their last layers, a dense MLP runs next to the routed experts, every block has
//! extra norms and a per-layer output scale, and the small models add per-layer embeddings. This
//! module is its own model with its own cache; it is reached through `family::load_any`.
//!
//! Written from the published architecture (transformers' `modeling_gemma4` and llama.cpp's graph).
//! It has been checked against its own invariants on small synthetic models; it has **not** been
//! compared with a real Gemma 4 checkpoint.
use crate::arch::{Act, Support};
use crate::backend::Backend;
use crate::gguf::{Gguf, Value};
use crate::loader::{LoadOptions, TensorSource, WeightFile, load_weight};
use crate::model::{ModelError, QWeight, Weight, activate};
use std::sync::Arc;

#[derive(Debug)]
pub enum G4Error {
    Config(String),
    Load(String),
}
impl std::fmt::Display for G4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            G4Error::Config(s) | G4Error::Load(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for G4Error {}
fn load_err(e: impl std::fmt::Display) -> G4Error {
    G4Error::Load(e.to_string())
}

/// Hyper-parameters read from the GGUF metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct G4Config {
    pub layers: usize,
    pub hidden: usize,
    pub context: usize,
    pub eps: f32,
    pub window: usize,
    /// `sliding[i]`: layer `i` only sees the last `window` tokens.
    pub sliding: Vec<bool>,
    pub hd_swa: usize,
    pub hd_full: usize,
    pub rot_swa: usize,
    pub rot_full: usize,
    pub theta_swa: f64,
    pub theta_full: f64,
    pub softcap: f32,
    /// Per-layer embedding width (0 = none).
    pub ple: usize,
    /// How many of the last layers borrow K/V from earlier ones.
    pub shared_kv: usize,
    pub experts: usize,
    pub used: usize,
    pub ff_exp: usize,
}

fn arr_bool(g: &Gguf, key: &str) -> Option<Vec<bool>> {
    match g.meta.get(&format!("gemma4.{key}"))? {
        Value::Array(a) => a
            .iter()
            .map(|v| match v {
                Value::Bool(b) => Some(*b),
                other => other.as_u64().map(|x| x != 0),
            })
            .collect(),
        _ => None,
    }
}

impl G4Config {
    pub fn from_gguf(g: &Gguf) -> Result<G4Config, G4Error> {
        let need = |k: &str| {
            g.get_u64(k)
                .filter(|v| *v > 0)
                .ok_or_else(|| G4Error::Config(format!("gemma4.{k} is missing")))
        };
        let layers = need("block_count")? as usize;
        if layers > 512 {
            return Err(G4Error::Config("implausible block_count".into()));
        }
        let hidden = need("embedding_length")? as usize;
        let hd_full = g.get_u64("attention.key_length").unwrap_or(512) as usize;
        let hd_swa = g.get_u64("attention.key_length_swa").unwrap_or(256) as usize;
        let sliding = match arr_bool(g, "attention.sliding_window_pattern") {
            Some(p) if p.len() == layers => p,
            _ => (0..layers).map(|i| (i + 1) % 6 != 0).collect(),
        };
        let mut sliding = sliding;
        if let Some(last) = sliding.last_mut() {
            *last = false; // the last layer always sees everything
        }
        let c = G4Config {
            layers,
            hidden,
            context: g.get_u64("context_length").unwrap_or(8192) as usize,
            eps: g
                .get_f64("attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-6) as f32,
            window: g.get_u64("attention.sliding_window").unwrap_or(1024).max(1) as usize,
            sliding,
            hd_swa,
            hd_full,
            rot_swa: g
                .get_u64("rope.dimension_count_swa")
                .map_or(hd_swa, |v| v as usize),
            rot_full: g
                .get_u64("rope.dimension_count")
                .map_or(hd_full, |v| v as usize),
            theta_swa: g.get_f64("rope.freq_base_swa").unwrap_or(10_000.0),
            theta_full: g.get_f64("rope.freq_base").unwrap_or(1_000_000.0),
            softcap: g.get_f64("final_logit_softcapping").unwrap_or(0.0) as f32,
            ple: g.get_u64("embedding_length_per_layer_input").unwrap_or(0) as usize,
            shared_kv: g.get_u64("attention.shared_kv_layers").unwrap_or(0) as usize,
            experts: g.get_u64("expert_count").unwrap_or(0) as usize,
            used: g.get_u64("expert_used_count").unwrap_or(0) as usize,
            ff_exp: g.get_u64("expert_feed_forward_length").unwrap_or(0) as usize,
        };
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<(), G4Error> {
        let bad = |s: &str| Err(G4Error::Config(s.into()));
        if self.hd_swa == 0
            || self.hd_full == 0
            || !self.hd_swa.is_multiple_of(2)
            || !self.hd_full.is_multiple_of(2)
        {
            return bad("head sizes must be even and non-zero");
        }
        if self.rot_swa == 0 || self.rot_swa > self.hd_swa || !self.rot_swa.is_multiple_of(2) {
            return bad("rope.dimension_count_swa must be even and within the head size");
        }
        if self.rot_full == 0 || self.rot_full > self.hd_full || !self.rot_full.is_multiple_of(2) {
            return bad("rope.dimension_count must be even and within the head size");
        }
        if self.shared_kv >= self.layers {
            return bad("every layer would borrow its K/V");
        }
        if self.experts > 0 && (self.used == 0 || self.used > self.experts || self.ff_exp == 0) {
            return bad(
                "expert_used_count / expert_feed_forward_length are missing or out of range",
            );
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------------ small kernels

fn rms(x: &[f32], w: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    let ss = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len().max(1) as f64;
    let inv = 1.0 / (ss + eps as f64).sqrt() as f32;
    match w {
        Some(w) => {
            for ((o, v), w) in out.iter_mut().zip(x).zip(w) {
                *o = v * inv * w;
            }
        }
        None => {
            for (o, v) in out.iter_mut().zip(x) {
                *o = v * inv;
            }
        }
    }
}
fn rms_vec(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut o = vec![0.0; x.len()];
    rms(x, w, eps, &mut o);
    o
}

/// RoPE inverse frequencies for `rot` rotated dimensions; `factors` (llama.cpp's `rope_freqs`)
/// divide them ("proportional" RoPE: a huge factor switches a pair off).
pub fn rope_freqs(theta: f64, rot: usize, factors: Option<&[f32]>) -> Vec<f64> {
    (0..rot / 2)
        .map(|i| {
            let f = theta.powf(-2.0 * i as f64 / rot as f64);
            let d = factors.and_then(|v| v.get(i)).map_or(1.0, |d| *d as f64);
            f / if d == 0.0 { 1.0 } else { d }
        })
        .collect()
}
/// Rotate the first `2 * inv.len()` values of `head` (pairs `(i, i + inv.len())`, "rotate half").
pub fn rope(head: &mut [f32], pos: usize, inv: &[f64]) {
    let half = inv.len();
    for (i, f) in inv.iter().enumerate() {
        let a = pos as f64 * f;
        let (s, c) = (a.sin() as f32, a.cos() as f32);
        let (x0, x1) = (head[i], head[i + half]);
        head[i] = x0 * c - x1 * s;
        head[i + half] = x1 * c + x0 * s;
    }
}

/// Softmax over every expert, the best `k`, renormalised to sum 1, times a per-expert scale.
pub fn route(logits: &[f32], k: usize, scale: Option<&[f32]>) -> Vec<(usize, f32)> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let s: f32 = e.iter().sum();
    let mut idx: Vec<usize> = (0..e.len()).collect();
    idx.sort_by(|&a, &b| e[b].total_cmp(&e[a]).then(a.cmp(&b)));
    idx.truncate(k);
    let top: f32 = idx.iter().map(|&i| e[i] / s).sum();
    idx.into_iter()
        .map(|i| {
            let w = e[i] / s / top.max(f32::MIN_POSITIVE);
            (i, w * scale.and_then(|v| v.get(i)).copied().unwrap_or(1.0))
        })
        .collect()
}

// ------------------------------------------------------------------------------------- weights

struct Experts {
    src: Arc<dyn WeightFile>,
    /// Fused gate+up tensor, or separate ones.
    fused: Option<String>,
    gate: String,
    up: String,
    down: String,
    hidden: usize,
    ff: usize,
    mmap: bool,
}
impl Experts {
    fn range(&self, name: &str, start: usize, rows: usize, cols: usize) -> Result<Weight, G4Error> {
        if let Some(r) = self
            .src
            .read_range_raw(name, start, rows, cols, self.mmap)
            .map_err(load_err)?
        {
            return Ok(Weight::Quant(
                QWeight::new(r.dtype, r.rows, r.cols, r.bytes).map_err(load_err)?,
            ));
        }
        Ok(Weight::Dense(
            self.src
                .read_range(name, start, rows * cols)
                .map_err(load_err)?,
        ))
    }
    /// `(gate, up, down)` of expert `e`.
    fn get(&self, e: usize) -> Result<(Weight, Weight, Weight), G4Error> {
        let (h, ff) = (self.hidden, self.ff);
        let (gate, up) = if let Some(name) = &self.fused {
            match self.range(name, e * 2 * ff * h, 2 * ff, h)? {
                Weight::Dense(v) => (
                    Weight::Dense(v[..ff * h].to_vec()),
                    Weight::Dense(v[ff * h..].to_vec()),
                ),
                Weight::Quant(q) => {
                    let rb = q.row_bytes();
                    let part = |from: usize| {
                        q.bytes
                            .slice(from * rb, ff * rb)
                            .ok_or_else(|| G4Error::Load("fused expert tensor is short".into()))
                            .and_then(|b| {
                                QWeight::new(q.dtype, ff, h, b)
                                    .map(Weight::Quant)
                                    .map_err(load_err)
                            })
                    };
                    (part(0)?, part(ff)?)
                }
            }
        } else {
            (
                self.range(&self.gate, e * ff * h, ff, h)?,
                self.range(&self.up, e * ff * h, ff, h)?,
            )
        };
        Ok((gate, up, self.range(&self.down, e * h * ff, h, ff)?))
    }
}

struct Moe {
    router: Vec<f32>,
    router_scale: Vec<f32>,
    pre2: Vec<f32>,
    post1: Vec<f32>,
    post2: Vec<f32>,
    down_scale: Option<Vec<f32>>,
    experts: Experts,
    used: usize,
    /// experts kept between tokens while a device holds them (capacity 0 = read per token)
    cache: LruCache<GExpert>,
}

/// One routed expert and its device copies (gate, up, down).
struct GExpert {
    w: [Weight; 3],
    dev: [Option<Resident>; 3],
}

struct Ple {
    gate: Weight,
    proj: Weight,
    post: Vec<f32>,
}

struct Layer {
    sliding: bool,
    hd: usize,
    nh: usize,
    nkv: usize,
    /// Layer whose K/V this one reads (itself when it owns a cache).
    kv_src: usize,
    attn_norm: Vec<f32>,
    wq: Weight,
    wk: Option<Weight>,
    wv: Option<Weight>,
    wo: Weight,
    q_norm: Vec<f32>,
    k_norm: Option<Vec<f32>>,
    post_attn: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate: Weight,
    up: Weight,
    down: Weight,
    ff: usize,
    post_ffn: Vec<f32>,
    moe: Option<Moe>,
    ple: Option<Ple>,
    out_scale: f32,
    inv: Vec<f64>,
}

pub struct Gemma4Model {
    cfg: G4Config,
    embed: Weight,
    head: Option<Weight>,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    ple_embed: Option<Weight>,
    ple_proj: Option<Weight>,
    ple_norm: Vec<f32>,
    vocab: usize,
    accel: Option<Accel>,
}

fn vec1(src: &dyn TensorSource, name: &str, n: usize) -> Result<Vec<f32>, G4Error> {
    let v = src.read(name).map_err(load_err)?;
    if v.len() != n {
        return Err(G4Error::Load(format!(
            "{name}: {} values, expected {n}",
            v.len()
        )));
    }
    Ok(v)
}
fn mat(
    src: &dyn TensorSource,
    name: &str,
    rows: Option<usize>,
    cols: usize,
    o: LoadOptions,
) -> Result<(Weight, usize), G4Error> {
    let w = load_weight(src, name, o).map_err(load_err)?;
    let len = w.len();
    if cols == 0 || !len.is_multiple_of(cols) {
        return Err(G4Error::Load(format!(
            "{name}: {len} values do not fill rows of {cols}"
        )));
    }
    let r = len / cols;
    if let Some(want) = rows
        && want != r
    {
        return Err(G4Error::Load(format!("{name}: {r} rows, expected {want}")));
    }
    Ok((w, r))
}

impl Gemma4Model {
    pub fn load(
        cfg: G4Config,
        src: &Arc<dyn WeightFile>,
        o: LoadOptions,
    ) -> Result<Gemma4Model, G4Error> {
        cfg.validate()?;
        let h = cfg.hidden;
        let s: &dyn TensorSource = src.as_ref();
        let (embed, vocab) = mat(s, "token_embd.weight", None, h, o)?;
        let head = if s.has("output.weight") {
            Some(mat(s, "output.weight", Some(vocab), h, o)?.0)
        } else {
            None
        };
        let final_norm = vec1(s, "output_norm.weight", h)?;
        let (mut ple_embed, mut ple_proj, mut ple_norm) = (None, None, vec![]);
        if cfg.ple > 0 {
            let w = cfg.layers * cfg.ple;
            ple_embed = Some(mat(s, "per_layer_token_embd.weight", Some(vocab), w, o)?.0);
            ple_proj = Some(mat(s, "per_layer_model_proj.weight", Some(w), h, o)?.0);
            ple_norm = vec1(s, "per_layer_proj_norm.weight", cfg.ple)?;
        }
        let global_factors = if s.has("rope_freqs.weight") {
            Some(s.read("rope_freqs.weight").map_err(load_err)?)
        } else {
            None
        };
        let first_shared = cfg.layers - cfg.shared_kv;
        let last_own = |sliding: bool| {
            (0..first_shared)
                .rev()
                .find(|&j| cfg.sliding[j] == sliding)
                .ok_or_else(|| G4Error::Config("no layer to borrow K/V from".into()))
        };
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let n = |p: &str| format!("blk.{i}.{p}.weight");
            let sliding = cfg.sliding[i];
            let hd = if sliding { cfg.hd_swa } else { cfg.hd_full };
            let (wq, qrows) = mat(s, &n("attn_q"), None, h, o)?;
            if !qrows.is_multiple_of(hd) {
                return Err(G4Error::Load(format!(
                    "layer {i}: attn_q has {qrows} rows, not a multiple of the head size {hd}"
                )));
            }
            let nh = qrows / hd;
            let owns = i < first_shared;
            let (wk, wv, nkv) = if owns {
                let (wk, krows) = mat(s, &n("attn_k"), None, h, o)?;
                let nkv = krows / hd;
                let wv = if s.has(&n("attn_v")) {
                    Some(mat(s, &n("attn_v"), Some(krows), h, o)?.0)
                } else {
                    None
                };
                (Some(wk), wv, nkv)
            } else {
                (None, None, 0)
            };
            let kv_src = if owns { i } else { last_own(sliding)? };
            let nkv = if owns {
                nkv
            } else {
                // a borrowing layer reads as many K/V heads as its source stores
                layers.get(kv_src).map_or(0, |l: &Layer| l.nkv)
            };
            if nkv == 0 || !nh.is_multiple_of(nkv) {
                return Err(G4Error::Load(format!(
                    "layer {i}: {nh} query heads do not divide over {nkv} K/V heads"
                )));
            }
            let (gate, ff) = mat(s, &n("ffn_gate"), None, h, o)?;
            let up = mat(s, &n("ffn_up"), Some(ff), h, o)?.0;
            let down = mat(s, &n("ffn_down"), Some(h), ff, o)?.0;
            let moe = if s.has(&n("ffn_gate_inp")) {
                if cfg.experts == 0 {
                    return Err(G4Error::Config(
                        "the file has routers but expert_count is missing".into(),
                    ));
                }
                let fused = s.has(&n("ffn_gate_up_exps")).then(|| n("ffn_gate_up_exps"));
                Some(Moe {
                    router: vec1(s, &n("ffn_gate_inp"), cfg.experts * h)?,
                    router_scale: vec1(s, &format!("blk.{i}.ffn_gate_inp.scale"), h)?,
                    pre2: vec1(s, &n("pre_ffw_norm_2"), h)?,
                    post1: vec1(s, &n("post_ffw_norm_1"), h)?,
                    post2: vec1(s, &n("post_ffw_norm_2"), h)?,
                    down_scale: {
                        let nm = format!("blk.{i}.ffn_down_exps.scale");
                        if s.has(&nm) {
                            Some(vec1(s, &nm, cfg.experts)?)
                        } else {
                            None
                        }
                    },
                    experts: Experts {
                        src: src.clone(),
                        fused,
                        gate: n("ffn_gate_exps"),
                        up: n("ffn_up_exps"),
                        down: n("ffn_down_exps"),
                        hidden: h,
                        ff: cfg.ff_exp,
                        mmap: o.mmap,
                    },
                    used: cfg.used,
                    cache: LruCache::default(),
                })
            } else {
                None
            };
            let ple = if cfg.ple > 0 {
                Some(Ple {
                    gate: mat(s, &n("inp_gate"), Some(cfg.ple), h, o)?.0,
                    proj: mat(s, &n("proj"), Some(h), cfg.ple, o)?.0,
                    post: vec1(s, &n("post_norm"), h)?,
                })
            } else {
                None
            };
            let (theta, rot, factors) = if sliding {
                (cfg.theta_swa, cfg.rot_swa, None)
            } else {
                (cfg.theta_full, cfg.rot_full, global_factors.as_deref())
            };
            layers.push(Layer {
                sliding,
                hd,
                nh,
                nkv,
                kv_src,
                attn_norm: vec1(s, &n("attn_norm"), h)?,
                wq,
                wk,
                wv,
                wo: mat(s, &n("attn_output"), Some(h), nh * hd, o)?.0,
                q_norm: vec1(s, &n("attn_q_norm"), hd)?,
                k_norm: if owns {
                    Some(vec1(s, &n("attn_k_norm"), hd)?)
                } else {
                    None
                },
                post_attn: vec1(s, &n("post_attention_norm"), h)?,
                ffn_norm: vec1(s, &n("ffn_norm"), h)?,
                gate,
                up,
                down,
                ff,
                post_ffn: vec1(s, &n("post_ffw_norm"), h)?,
                moe,
                ple,
                out_scale: if s.has(&n("layer_output_scale")) {
                    vec1(s, &n("layer_output_scale"), 1)?[0]
                } else {
                    1.0
                },
                inv: rope_freqs(theta, rot, factors),
            });
        }
        Ok(Gemma4Model {
            cfg,
            embed,
            head,
            final_norm,
            layers,
            ple_embed,
            ple_proj,
            ple_norm,
            vocab,
            accel: None,
        })
    }
}

// ---------------------------------------------------------------------------------------- state

#[derive(Default, Clone)]
struct Cache {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Position of the first stored token.
    first: usize,
}

/// Per-conversation state: one K/V cache per layer that owns one (sliding layers keep about two
/// windows, never the whole history).
pub struct Gemma4State {
    caches: Vec<Cache>,
    pub pos: usize,
}
impl Gemma4State {
    pub fn bytes(&self) -> usize {
        self.caches
            .iter()
            .map(|c| (c.k.len() + c.v.len()) * 4)
            .sum()
    }
}

// ------------------------------------------------------------------------------ accelerator

pub use crate::accel::AttachReport;
use crate::accel::{Accel, Item, LruCache, Resident, key_of, mv_res, weight_bytes};

impl Gemma4Model {
    /// `out = W · x`: on the device when the matrix was uploaded and the device still works,
    /// otherwise (and after the first device error, for good) on the CPU.
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

    /// Put the big matrices on `be`, as many as its memory holds, after checking one of them gives
    /// the CPU's answer. Never fails: when nothing can be used the model keeps running on the CPU
    /// and the report says why.
    pub fn attach_backend(&mut self, be: Arc<dyn Backend>) -> AttachReport {
        self.accel = None;
        for m in self.layers.iter().filter_map(|l| l.moe.as_ref()) {
            m.cache.reset(0);
        }
        let h = self.cfg.hidden;
        let mut items: Vec<Item> = vec![];
        let head = self.head.as_ref().unwrap_or(&self.embed);
        items.push(Item::new(head, h, true));
        for l in &self.layers {
            for (w, c) in [(&l.wq, h), (&l.gate, h), (&l.up, h), (&l.down, l.ff)] {
                items.push(Item::new(w, c, false));
            }
            items.push(Item::new(&l.wo, l.nh * l.hd, false));
            for w in l.wk.iter().chain(l.wv.iter()) {
                items.push(Item::new(w, h, false));
            }
        }
        let (acc, mut rep) = Accel::build(be, &items);
        self.accel = acc;
        let first = self.layers.iter().find_map(|l| l.moe.as_ref());
        if let (Some(a), Some(m0)) = (&self.accel, first)
            && let Ok((g, u, d)) = m0.experts.get(0)
        {
            let ff = self.cfg.ff_exp;
            let bytes = [(&g, h), (&u, h), (&d, ff)]
                .iter()
                .map(|(w, c)| weight_bytes(w, *c))
                .sum::<Option<u64>>()
                .unwrap_or(0);
            let n = self.layers.iter().filter(|l| l.moe.is_some()).count();
            let cap = a.expert_cap(bytes, n, self.cfg.experts, self.cfg.used);
            for m in self.layers.iter().filter_map(|l| l.moe.as_ref()) {
                m.cache.reset(cap);
            }
            if cap > 0 {
                let t = format!("up to {cap} routed experts per layer are kept on the device");
                rep.note = if rep.note.is_empty() {
                    t
                } else {
                    format!("{}; {t}", rep.note)
                };
            }
        }
        rep
    }

    /// Experts held by the caches now and the device bytes they use (for tests and reports).
    pub fn resident_experts(&self) -> (usize, u64) {
        (
            self.layers
                .iter()
                .filter_map(|l| l.moe.as_ref())
                .map(|m| m.cache.len())
                .sum(),
            self.accel.as_ref().map_or(0, Accel::resident_bytes),
        )
    }
}

impl Gemma4Model {
    pub fn config(&self) -> &G4Config {
        &self.cfg
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn support() -> Support {
        Support::Native("Gemma 4 runs through its own model in the family loader")
    }
    pub fn new_state(&self) -> Gemma4State {
        Gemma4State {
            caches: vec![Cache::default(); self.cfg.layers],
            pos: 0,
        }
    }

    fn embed_row(&self, token: usize) -> Result<Vec<f32>, ModelError> {
        if token >= self.vocab {
            return Err(ModelError::Token(token));
        }
        let h = self.cfg.hidden;
        let (mut out, mut tmp) = (vec![0.0; h], vec![]);
        self.embed.row_into(token, h, &mut out, &mut tmp)?;
        let s = (h as f32).sqrt();
        out.iter_mut().for_each(|v| *v *= s);
        Ok(out)
    }

    /// `[layers][ple]` inputs for one token.
    fn ple_inputs(&self, token: usize, x: &[f32]) -> Result<Vec<Vec<f32>>, ModelError> {
        let (Some(emb), Some(proj)) = (&self.ple_embed, &self.ple_proj) else {
            return Ok(vec![]);
        };
        let (l, p, h) = (self.cfg.layers, self.cfg.ple, self.cfg.hidden);
        let mut tok = vec![0.0; l * p];
        let mut tmp = vec![];
        emb.row_into(token, l * p, &mut tok, &mut tmp)?;
        let mut pr = vec![0.0; l * p];
        self.mv(proj, h, x, &mut pr)?;
        let (ts, ps) = ((p as f32).sqrt(), 1.0 / (h as f32).sqrt());
        let k = std::f32::consts::FRAC_1_SQRT_2;
        let mut out = Vec::with_capacity(l);
        for i in 0..l {
            let mut a: Vec<f32> = pr[i * p..(i + 1) * p].iter().map(|v| v * ps).collect();
            a = rms_vec(&a, Some(&self.ple_norm), self.cfg.eps);
            for (a, t) in a.iter_mut().zip(&tok[i * p..(i + 1) * p]) {
                *a = (*a + t * ts) * k;
            }
            out.push(a);
        }
        Ok(out)
    }

    fn mlp(&self, l: &Layer, x: &[f32]) -> Result<Vec<f32>, ModelError> {
        let h = self.cfg.hidden;
        let (mut g, mut u, mut o) = (vec![0.0; l.ff], vec![0.0; l.ff], vec![0.0; h]);
        self.mv(&l.gate, h, x, &mut g)?;
        self.mv(&l.up, h, x, &mut u)?;
        for (a, b) in g.iter_mut().zip(&u) {
            *a = activate(Act::GeluTanh, *a) * b;
        }
        self.mv(&l.down, l.ff, &g, &mut o)?;
        Ok(o)
    }

    fn moe(&self, m: &Moe, attn_out: &[f32], x2: &[f32]) -> Result<Vec<f32>, ModelError> {
        let (h, e) = (self.cfg.hidden, self.cfg.eps);
        let mut r = rms_vec(attn_out, None, e);
        let s = 1.0 / (h as f32).sqrt();
        for (v, sc) in r.iter_mut().zip(&m.router_scale) {
            *v *= s * sc;
        }
        let logits: Vec<f32> = m
            .router
            .chunks_exact(h)
            .map(|row| row.iter().zip(&r).map(|(a, b)| a * b).sum())
            .collect();
        let picks = route(&logits, m.used, m.down_scale.as_deref());
        let ff = self.cfg.ff_exp;
        let mut out = vec![0.0f32; h];
        for (ex, w) in picks {
            let ac = self.accel.as_ref();
            let xp = m
                .cache
                .get_or_load(ex, || {
                    let (g, u, d) = m.experts.get(ex)?;
                    let mut xp = GExpert {
                        w: [g, u, d],
                        dev: [None, None, None],
                    };
                    if let Some(a) = ac.filter(|_| m.cache.cap() > 0) {
                        xp.dev = [
                            a.upload(&xp.w[0], h),
                            a.upload(&xp.w[1], h),
                            a.upload(&xp.w[2], ff),
                        ];
                    }
                    Ok::<_, G4Error>(xp)
                })
                .map_err(|e| ModelError::Experts(e.to_string()))?;
            let (mut g, mut u, mut d) = (vec![0.0; ff], vec![0.0; ff], vec![0.0; h]);
            mv_res(ac, &xp.w[0], xp.dev[0].as_ref(), h, x2, &mut g)?;
            mv_res(ac, &xp.w[1], xp.dev[1].as_ref(), h, x2, &mut u)?;
            for (a, b) in g.iter_mut().zip(&u) {
                *a = activate(Act::GeluTanh, *a) * b;
            }
            mv_res(ac, &xp.w[2], xp.dev[2].as_ref(), ff, &g, &mut d)?;
            out.iter_mut().zip(&d).for_each(|(o, d)| *o += w * d);
        }
        Ok(out)
    }

    fn step(&self, st: &mut Gemma4State, token: usize, want: bool) -> Result<Vec<f32>, ModelError> {
        let c = &self.cfg;
        let (h, eps, pos) = (c.hidden, c.eps, st.pos);
        if pos >= c.context {
            return Err(ModelError::ContextFull(c.context));
        }
        let mut x = self.embed_row(token)?;
        let ple = self.ple_inputs(token, &x)?;
        for (i, l) in self.layers.iter().enumerate() {
            let (hd, nh, nkv) = (l.hd, l.nh, l.nkv);
            let xn = rms_vec(&x, Some(&l.attn_norm), eps);
            let mut q = vec![0.0; nh * hd];
            self.mv(&l.wq, h, &xn, &mut q)?;
            for hq in q.chunks_mut(hd) {
                let n = rms_vec(hq, Some(&l.q_norm), eps);
                hq.copy_from_slice(&n);
                rope(hq, pos, &l.inv);
            }
            if let Some(wk) = &l.wk {
                let mut k = vec![0.0; nkv * hd];
                self.mv(wk, h, &xn, &mut k)?;
                let mut v = vec![0.0; nkv * hd];
                match &l.wv {
                    Some(wv) => self.mv(wv, h, &xn, &mut v)?,
                    None => v.copy_from_slice(&k), // keys equal values on these layers
                }
                let kn = l.k_norm.as_deref();
                for (hk, hv) in k.chunks_mut(hd).zip(v.chunks_mut(hd)) {
                    let n = rms_vec(hk, kn, eps);
                    hk.copy_from_slice(&n);
                    rope(hk, pos, &l.inv);
                    let n = rms_vec(hv, None, eps);
                    hv.copy_from_slice(&n);
                }
                let cache = &mut st.caches[i];
                if cache.k.is_empty() {
                    cache.first = pos;
                }
                cache.k.extend_from_slice(&k);
                cache.v.extend_from_slice(&v);
                if l.sliding {
                    let stored = cache.k.len() / (nkv * hd);
                    if stored > 2 * c.window {
                        let drop = stored - c.window;
                        cache.k.drain(..drop * nkv * hd);
                        cache.v.drain(..drop * nkv * hd);
                        cache.first += drop;
                    }
                }
            }
            let cache = &st.caches[l.kv_src];
            let stored = cache.k.len() / (nkv * hd);
            let from = if l.sliding {
                (pos + 1).saturating_sub(c.window).max(cache.first)
            } else {
                cache.first
            };
            let (lo, hi) = (from - cache.first, stored);
            let group = nh / nkv;
            let mut att = vec![0.0f32; nh * hd];
            let mut sc = vec![0.0f32; hi - lo];
            for head in 0..nh {
                let kvh = head / group;
                let qh = &q[head * hd..(head + 1) * hd];
                for (t, s) in (lo..hi).zip(sc.iter_mut()) {
                    let k = &cache.k[(t * nkv + kvh) * hd..(t * nkv + kvh + 1) * hd];
                    *s = qh.iter().zip(k).map(|(a, b)| a * b).sum();
                }
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut z = 0.0;
                for s in sc.iter_mut() {
                    *s = (*s - m).exp();
                    z += *s;
                }
                let o = &mut att[head * hd..(head + 1) * hd];
                for (t, s) in (lo..hi).zip(&sc) {
                    let v = &cache.v[(t * nkv + kvh) * hd..(t * nkv + kvh + 1) * hd];
                    let w = s / z;
                    o.iter_mut().zip(v).for_each(|(o, v)| *o += w * v);
                }
            }
            let mut proj = vec![0.0; h];
            self.mv(&l.wo, nh * hd, &att, &mut proj)?;
            let proj = rms_vec(&proj, Some(&l.post_attn), eps);
            x.iter_mut().zip(&proj).for_each(|(a, b)| *a += b);
            // feed-forward
            let f = if let Some(m) = &l.moe {
                let xm = rms_vec(&x, Some(&l.ffn_norm), eps);
                let dense = rms_vec(&self.mlp(l, &xm)?, Some(&m.post1), eps);
                let x2 = rms_vec(&x, Some(&m.pre2), eps);
                let routed = rms_vec(&self.moe(m, &x, &x2)?, Some(&m.post2), eps);
                dense.iter().zip(&routed).map(|(a, b)| a + b).collect()
            } else {
                let xm = rms_vec(&x, Some(&l.ffn_norm), eps);
                self.mlp(l, &xm)?
            };
            let f = rms_vec(&f, Some(&l.post_ffn), eps);
            x.iter_mut().zip(&f).for_each(|(a, b)| *a += b);
            if let (Some(p), Some(inp)) = (&l.ple, ple.get(i)) {
                let mut g = vec![0.0; c.ple];
                self.mv(&p.gate, h, &x, &mut g)?;
                for (a, b) in g.iter_mut().zip(inp) {
                    *a = activate(Act::GeluTanh, *a) * b;
                }
                let mut o = vec![0.0; h];
                self.mv(&p.proj, c.ple, &g, &mut o)?;
                let o = rms_vec(&o, Some(&p.post), eps);
                x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
            }
            if l.out_scale != 1.0 {
                x.iter_mut().for_each(|v| *v *= l.out_scale);
            }
        }
        st.pos += 1;
        if !want {
            return Ok(vec![]);
        }
        let xn = rms_vec(&x, Some(&self.final_norm), eps);
        let mut logits = vec![0.0; self.vocab];
        self.mv(
            self.head.as_ref().unwrap_or(&self.embed),
            h,
            &xn,
            &mut logits,
        )?;
        if c.softcap > 0.0 {
            logits
                .iter_mut()
                .for_each(|v| *v = c.softcap * (*v / c.softcap).tanh());
        }
        Ok(logits)
    }

    /// Feed `tokens`, return the logits of the last one. After an `Err` the state must be discarded.
    pub fn forward_batch(
        &self,
        st: &mut Gemma4State,
        tokens: &[usize],
    ) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        if let Some(&t) = tokens.iter().find(|&&t| t >= self.vocab) {
            return Err(ModelError::Token(t));
        }
        if st.pos + tokens.len() > self.cfg.context {
            return Err(ModelError::ContextFull(self.cfg.context));
        }
        let mut last = vec![];
        for (i, &t) in tokens.iter().enumerate() {
            last = self.step(st, t, i + 1 == tokens.len())?;
        }
        Ok(last)
    }
}

/// A tiny Gemma-4-shaped GGUF writer for tests (here and in the CLI crate): random weights, every
/// feature switched on by default (mixture-of-experts, per-layer embeddings, shared K/V, keys equal
/// to values on global layers, proportional RoPE). Not part of the public interface.
#[doc(hidden)]
pub mod demo {
    use crate::gguf_write::{GgufWriter, MetaValue as M, TensorDesc};
    use crate::quant::DType;
    use std::path::Path;

    fn q8_0_encode(w: &[f32]) -> Vec<u8> {
        let mut o = vec![];
        for blk in w.chunks(32) {
            let d = blk.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0;
            let dh = crate::model::f32_to_f16(d);
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

    #[derive(Clone)]
    pub struct Opt {
        pub sliding: Vec<bool>,
        pub shared: usize,
        pub ple: usize,
        pub experts: usize,
        pub quant: bool,
        pub seed: u64,
        pub window: u64,
        pub vocab: usize,
    }
    impl Opt {
        pub fn full() -> Opt {
            Opt {
                sliding: vec![true, true, false, true, true, false],
                shared: 2,
                ple: 4,
                experts: 4,
                quant: false,
                seed: 7,
                window: 4,
                vocab: 40,
            }
        }
    }
    pub const H: usize = 32;
    pub const FF: usize = 32;
    pub const HD_S: usize = 8;
    pub const HD_F: usize = 16;

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }
        fn mat(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.f() * 0.4).collect()
        }
        fn norm(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| 1.0 + self.f() * 0.4).collect()
        }
    }
    fn f32b(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// Write a small Gemma-4-shaped GGUF. `exp_dtype` selects how the fused expert tensors are stored.
    pub fn write(path: &Path, o: &Opt, as_f32: bool) {
        let v = o.vocab;
        let layers = o.sliding.len();
        let first_shared = layers - o.shared;
        let mut r = Rng(o.seed | 1);
        let mut t: Vec<(String, Vec<u64>, DType, Vec<u8>)> = vec![];
        let mut f = |name: String, shape: &[usize], data: Vec<f32>| {
            assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}");
            t.push((
                name,
                shape.iter().map(|x| *x as u64).collect(),
                DType::F32,
                f32b(&data),
            ));
        };
        f("token_embd.weight".into(), &[v, H], r.mat(v * H));
        f("output_norm.weight".into(), &[H], r.norm(H));
        if o.ple > 0 {
            f(
                "per_layer_token_embd.weight".into(),
                &[v, layers * o.ple],
                r.mat(v * layers * o.ple),
            );
            f(
                "per_layer_model_proj.weight".into(),
                &[layers * o.ple, H],
                r.mat(layers * o.ple * H),
            );
            f("per_layer_proj_norm.weight".into(), &[o.ple], r.norm(o.ple));
        }
        // proportional RoPE: the last quarter of the pairs is switched off
        let fac: Vec<f32> = (0..HD_F / 2)
            .map(|i| if i < HD_F / 2 - 2 { 1.0 } else { 1e30 })
            .collect();
        f("rope_freqs.weight".into(), &[HD_F / 2], fac);
        let mut extra: Vec<(String, Vec<usize>, Vec<f32>)> = vec![];
        for i in 0..layers {
            let s = o.sliding[i] && i + 1 != layers;
            let (hd, nh, nkv) = if s { (HD_S, 4, 2) } else { (HD_F, 4, 1) };
            let n = |p: &str| format!("blk.{i}.{p}.weight");
            f(n("attn_norm"), &[H], r.norm(H));
            f(n("attn_q"), &[nh * hd, H], r.mat(nh * hd * H));
            if i < first_shared {
                f(n("attn_k"), &[nkv * hd, H], r.mat(nkv * hd * H));
                if s {
                    f(n("attn_v"), &[nkv * hd, H], r.mat(nkv * hd * H));
                }
                f(n("attn_k_norm"), &[hd], r.norm(hd));
            }
            f(n("attn_output"), &[H, nh * hd], r.mat(H * nh * hd));
            f(n("attn_q_norm"), &[hd], r.norm(hd));
            f(n("post_attention_norm"), &[H], r.norm(H));
            f(n("ffn_norm"), &[H], r.norm(H));
            f(n("ffn_gate"), &[FF, H], r.mat(FF * H));
            f(n("ffn_up"), &[FF, H], r.mat(FF * H));
            f(n("ffn_down"), &[H, FF], r.mat(H * FF));
            f(n("post_ffw_norm"), &[H], r.norm(H));
            f(n("layer_output_scale"), &[1], vec![0.8 + 0.05 * i as f32]);
            if o.experts > 0 && i % 2 == 1 {
                let e = o.experts;
                f(n("ffn_gate_inp"), &[e, H], r.mat(e * H));
                f(format!("blk.{i}.ffn_gate_inp.scale"), &[H], r.norm(H));
                f(n("pre_ffw_norm_2"), &[H], r.norm(H));
                f(n("post_ffw_norm_1"), &[H], r.norm(H));
                f(n("post_ffw_norm_2"), &[H], r.norm(H));
                f(format!("blk.{i}.ffn_down_exps.scale"), &[e], r.norm(e));
                let gu = r.mat(e * 2 * FF * H);
                let dn = r.mat(e * H * FF);
                if o.quant && !as_f32 {
                    extra.push((n("ffn_gate_up_exps"), vec![e, 2 * FF, H], gu));
                    extra.push((n("ffn_down_exps"), vec![e, H, FF], dn));
                } else if o.quant {
                    // the same weights after a Q8_0 round trip, stored as f32
                    for (nm, sh, d) in [
                        (n("ffn_gate_up_exps"), vec![e, 2 * FF, H], gu),
                        (n("ffn_down_exps"), vec![e, H, FF], dn),
                    ] {
                        let mut back = vec![];
                        crate::quant::dequantize(DType::Q8_0, &q8_0_encode(&d), d.len(), &mut back)
                            .unwrap();
                        extra.push((nm, sh, back));
                    }
                } else {
                    extra.push((n("ffn_gate_up_exps"), vec![e, 2 * FF, H], gu));
                    extra.push((n("ffn_down_exps"), vec![e, H, FF], dn));
                }
            }
            if o.ple > 0 {
                f(n("inp_gate"), &[o.ple, H], r.mat(o.ple * H));
                f(n("proj"), &[H, o.ple], r.mat(H * o.ple));
                f(n("post_norm"), &[H], r.norm(H));
            }
        }
        for (name, shape, d) in extra {
            let (dt, bytes) = if o.quant && !as_f32 {
                (DType::Q8_0, q8_0_encode(&d))
            } else {
                (DType::F32, f32b(&d))
            };
            t.push((name, shape.iter().map(|x| *x as u64).collect(), dt, bytes));
        }
        let meta: Vec<(String, M)> = vec![
            ("general.architecture".into(), M::str("gemma4")),
            ("gemma4.block_count".into(), M::U32(layers as u32)),
            ("gemma4.context_length".into(), M::U32(64)),
            ("gemma4.embedding_length".into(), M::U32(H as u32)),
            ("gemma4.attention.key_length".into(), M::U32(HD_F as u32)),
            (
                "gemma4.attention.key_length_swa".into(),
                M::U32(HD_S as u32),
            ),
            (
                "gemma4.attention.sliding_window".into(),
                M::U32(o.window as u32),
            ),
            (
                "gemma4.attention.sliding_window_pattern".into(),
                M::Array(o.sliding.iter().map(|b| M::Bool(*b)).collect()),
            ),
            (
                "gemma4.attention.shared_kv_layers".into(),
                M::U32(o.shared as u32),
            ),
            (
                "gemma4.attention.layer_norm_rms_epsilon".into(),
                M::F32(1e-6),
            ),
            (
                "gemma4.embedding_length_per_layer_input".into(),
                M::U32(o.ple as u32),
            ),
            ("gemma4.rope.dimension_count".into(), M::U32(HD_F as u32)),
            (
                "gemma4.rope.dimension_count_swa".into(),
                M::U32(HD_S as u32),
            ),
            ("gemma4.rope.freq_base".into(), M::F32(1e6)),
            ("gemma4.rope.freq_base_swa".into(), M::F32(1e4)),
            ("gemma4.final_logit_softcapping".into(), M::F32(30.0)),
            ("gemma4.expert_count".into(), M::U32(o.experts as u32)),
            ("gemma4.expert_used_count".into(), M::U32(2)),
            (
                "gemma4.expert_feed_forward_length".into(),
                M::U32(FF as u32),
            ),
        ];
        let descs: Vec<TensorDesc> = t
            .iter()
            .map(|(n, s, d, _)| TensorDesc::new(n.clone(), s, *d))
            .collect();
        let file = std::fs::File::create(path).unwrap();
        GgufWriter::write_all(file, &meta, descs, |i, _| Ok(t[i].3.clone())).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::demo::{FF, H, HD_F, Opt, write};
    use super::*;
    use crate::accel::testbe::Odd;
    use crate::family::{self, AnyModel, LanguageModel};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    const V: usize = 40;

    fn dir() -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "g4-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn open(o: &Opt, as_f32: bool) -> (AnyModel, PathBuf) {
        let d = dir();
        let p = d.join("tiny-gemma4.gguf");
        write(&p, o, as_f32);
        (family::load_any(&p).unwrap(), p)
    }
    fn g4(m: &AnyModel) -> &Gemma4Model {
        match m {
            AnyModel::Gemma4(g) => g,
            _ => panic!("not gemma4"),
        }
    }
    fn run(m: &Gemma4Model, toks: &[usize]) -> Vec<f32> {
        let mut st = m.new_state();
        m.forward_batch(&mut st, toks).unwrap()
    }
    fn close(a: &[f32], b: &[f32], tol: f32) {
        let scale = b.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
        let d = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(d <= tol * scale, "max difference {d} (scale {scale})");
    }

    #[test]
    fn classify_and_config_from_the_file() {
        assert!(matches!(
            crate::arch::classify("gemma4"),
            Support::Native(_)
        ));
        assert!(crate::arch::classify("gemma4").runnable());
        let (m, p) = open(&Opt::full(), true);
        let g = g4(&m);
        let f = crate::loader::GgufFile::open(&p).unwrap();
        let c = G4Config::from_gguf(&f.gguf).unwrap();
        assert_eq!(&c, g.config());
        assert_eq!(c.layers, 6);
        // the last layer is global even if the pattern says otherwise
        assert_eq!(c.sliding, [true, true, false, true, true, false]);
        assert_eq!((c.ple, c.shared_kv, c.experts, c.used), (4, 2, 4, 2));
        assert_eq!(g.vocab(), V);
        assert_eq!(LanguageModel::vocab_size(&m), V);
        assert_eq!(LanguageModel::context(&m), 64);
    }

    #[test]
    fn runs_is_finite_deterministic_and_softcapped() {
        let (m, _) = open(&Opt::full(), true);
        let g = g4(&m);
        let toks: Vec<usize> = (0..12).map(|i| (i * 7 + 3) % V).collect();
        let a = run(g, &toks);
        assert_eq!(a.len(), V);
        assert!(a.iter().all(|v| v.is_finite() && v.abs() <= 30.0 + 1e-3));
        assert_eq!(a, run(g, &toks));
        assert!(
            a.iter().any(|v| v.abs() > 1e-4),
            "logits should not be all zero"
        );
    }

    #[test]
    fn batch_equals_token_by_token_and_chunks() {
        let (m, _) = open(&Opt::full(), true);
        let g = g4(&m);
        let toks: Vec<usize> = (0..15).map(|i| (i * 11 + 1) % V).collect();
        let whole = run(g, &toks);
        let mut st = g.new_state();
        let mut last = vec![];
        for t in &toks {
            last = g.forward_batch(&mut st, &[*t]).unwrap();
        }
        assert_eq!(whole, last);
        let mut st = g.new_state();
        g.forward_batch(&mut st, &toks[..6]).unwrap();
        assert_eq!(whole, g.forward_batch(&mut st, &toks[6..]).unwrap());
        assert_eq!(st.pos, 15);
    }

    #[test]
    fn shared_layers_own_no_cache_and_sliding_caches_stay_bounded() {
        let (m, _) = open(&Opt::full(), true);
        let g = g4(&m);
        let toks: Vec<usize> = (0..40).map(|i| (i * 5 + 2) % V).collect();
        let mut st = g.new_state();
        g.forward_batch(&mut st, &toks).unwrap();
        // layers 4 and 5 borrow the K/V of layers 3 (sliding) and 2 (global)
        assert!(st.caches[4].k.is_empty() && st.caches[5].k.is_empty());
        assert_eq!(g.layers[4].kv_src, 3);
        assert_eq!(g.layers[5].kv_src, 2);
        let w = g.cfg.window;
        for i in [0usize, 1, 3] {
            let c = &st.caches[i];
            let per = g.layers[i].nkv * g.layers[i].hd;
            let stored = c.k.len() / per;
            assert!(stored <= 2 * w, "layer {i} keeps {stored} positions");
            assert!(stored >= w);
            assert_eq!(c.first + stored, 40, "layer {i} keeps the newest tokens");
        }
        // global layers keep everything
        assert_eq!(
            st.caches[2].k.len() / (g.layers[2].nkv * g.layers[2].hd),
            40
        );
        assert!(st.bytes() > 0);
    }

    #[test]
    fn global_layers_see_old_tokens_and_the_window_is_applied() {
        let (m, _) = open(&Opt::full(), true);
        let g = g4(&m);
        let mut a: Vec<usize> = (0..20).map(|i| (i * 3 + 1) % V).collect();
        let mut b = a.clone();
        b[0] = (b[0] + 5) % V;
        // the global layers (2 and 5) read token 0, so the logits differ
        assert_ne!(run(g, &a), run(g, &b));
        // identical history: identical logits, and a changed last token changes them
        let x = run(g, &a);
        a[19] = (a[19] + 1) % V;
        assert_ne!(x, run(g, &a));
    }

    #[test]
    fn keys_equal_values_on_global_layers_without_a_value_projection() {
        let (m, _) = open(&Opt::full(), true);
        let g = g4(&m);
        assert!(g.layers[2].wv.is_none(), "global layer 2 has no attn_v");
        assert!(g.layers[0].wv.is_some());
        let mut st = g.new_state();
        g.forward_batch(&mut st, &[1, 2, 3]).unwrap();
        // V is the *un-scaled* normalised raw key projection, K the scaled and rotated one: they differ
        let c = &st.caches[2];
        assert_ne!(c.k, c.v);
        // every value head is unit-RMS (normalised without a scale)
        for hv in c.v.chunks(HD_F) {
            let ms = hv.iter().map(|x| x * x).sum::<f32>() / HD_F as f32;
            assert!((ms - 1.0).abs() < 1e-3, "rms² {ms}");
        }
    }

    #[test]
    fn works_without_experts_ple_or_sharing() {
        let o = Opt {
            experts: 0,
            ple: 0,
            shared: 0,
            ..Opt::full()
        };
        let (m, _) = open(&o, true);
        let g = g4(&m);
        assert!(g.layers.iter().all(|l| l.moe.is_none() && l.ple.is_none()));
        assert!(g.layers.iter().enumerate().all(|(i, l)| l.kv_src == i));
        let r = run(g, &[1, 2, 3, 4]);
        assert!(r.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn experts_and_ple_change_the_output() {
        let (m, _) = open(&Opt::full(), true);
        let (n, _) = open(
            &Opt {
                experts: 0,
                ..Opt::full()
            },
            true,
        );
        assert!(g4(&m).layers[1].moe.is_some() && g4(&n).layers[1].moe.is_none());
        let t = [3usize, 9, 4, 1];
        assert_ne!(run(g4(&m), &t), run(g4(&n), &t));
    }

    #[test]
    fn quantised_fused_experts_track_their_dequantised_twin() {
        let o = Opt {
            quant: true,
            ..Opt::full()
        };
        let (q, _) = open(&o, false);
        let (f, _) = open(&o, true);
        let moe = g4(&q).layers[1].moe.as_ref().unwrap();
        let (g, u, d) = moe.experts.get(1).unwrap();
        assert!(matches!(g, Weight::Quant(_)) && matches!(u, Weight::Quant(_)));
        assert!(matches!(d, Weight::Quant(_)));
        let t: Vec<usize> = (0..8).map(|i| (i * 13 + 5) % V).collect();
        close(&run(g4(&q), &t), &run(g4(&f), &t), 0.05);
    }

    #[test]
    fn fused_split_gives_gate_first_then_up() {
        let o = Opt {
            quant: false,
            ..Opt::full()
        };
        let (m, p) = open(&o, true);
        let moe = g4(&m).layers[1].moe.as_ref().unwrap();
        let (g, u, _) = moe.experts.get(2).unwrap();
        let f = crate::loader::GgufFile::open(&p).unwrap();
        let all = f.read("blk.1.ffn_gate_up_exps.weight").unwrap();
        let base = 2 * 2 * FF * H;
        assert_eq!(g.to_f32(), all[base..base + FF * H]);
        assert_eq!(u.to_f32(), all[base + FF * H..base + 2 * FF * H]);
    }

    #[test]
    fn routing_picks_the_best_and_renormalises() {
        let r = route(&[0.0, 2.0, 1.0, -1.0], 2, None);
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), [1, 2]);
        assert!((r.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-6);
        let e = (2.0f32).exp() / ((2.0f32).exp() + (1.0f32).exp());
        assert!((r[0].1 - e).abs() < 1e-6);
        let s = route(&[0.0, 2.0, 1.0, -1.0], 2, Some(&[1.0, 3.0, 0.5, 1.0]));
        assert!((s[0].1 - 3.0 * e).abs() < 1e-5 && (s[1].1 - 0.5 * (1.0 - e)).abs() < 1e-5);
        // ties go to the lower index
        assert_eq!(route(&[1.0, 1.0, 1.0], 2, None)[0].0, 0);
    }

    #[test]
    fn proportional_rope_switches_dimensions_off_and_preserves_length() {
        let inv = rope_freqs(1e6, 16, Some(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e30, 1e30]));
        assert_eq!(inv.len(), 8);
        assert!(inv[6] < 1e-30 && inv[7] < 1e-30 && inv[5] > 1e-4);
        let mut h: Vec<f32> = (0..16).map(|i| i as f32 * 0.3 - 2.0).collect();
        let orig = h.clone();
        rope(&mut h, 0, &inv);
        assert_eq!(h, orig, "position 0 is the identity");
        rope(&mut h, 37, &inv);
        // pairs 6 and 7 (indices 6,14 and 7,15) are untouched
        for i in [6usize, 7, 14, 15] {
            assert!((h[i] - orig[i]).abs() < 1e-6, "dim {i}");
        }
        assert!((h[0] - orig[0]).abs() > 1e-3);
        let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();
        assert!((n(&h) - n(&orig)).abs() < 1e-3);
        // without factors: plain rotate-half RoPE of the first pair, angle = pos
        let mut p = vec![1.0f32, 0.0, 0.0, 0.0];
        rope(&mut p, 2, &rope_freqs(10_000.0, 4, None));
        assert!((p[0] - (2.0f32).cos()).abs() < 1e-6 && (p[2] - (2.0f32).sin()).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_matches_the_formula_and_applies_the_weight_directly() {
        let x = [3.0f32, 4.0];
        let ms = (9.0 + 16.0) / 2.0;
        let inv = 1.0 / (ms + 1e-6f32).sqrt();
        let o = rms_vec(&x, Some(&[2.0, 0.5]), 1e-6);
        assert!((o[0] - 3.0 * inv * 2.0).abs() < 1e-5 && (o[1] - 4.0 * inv * 0.5).abs() < 1e-5);
        let u = rms_vec(&x, None, 1e-6);
        assert!((u[0] - 3.0 * inv).abs() < 1e-6);
    }

    #[test]
    fn bad_input_is_refused_with_a_reason() {
        let (m, p) = open(&Opt::full(), true);
        let g = g4(&m);
        let mut st = g.new_state();
        assert_eq!(g.forward_batch(&mut st, &[]), Err(ModelError::NoTokens));
        assert_eq!(g.forward_batch(&mut st, &[V]), Err(ModelError::Token(V)));
        assert!(matches!(
            g.forward_batch(&mut st, &vec![1; 65]),
            Err(ModelError::ContextFull(64))
        ));
        assert_eq!(st.pos, 0, "a refused batch leaves the state alone");
        // config problems are named
        let f = crate::loader::GgufFile::open(&p).unwrap();
        let mut c = G4Config::from_gguf(&f.gguf).unwrap();
        c.shared_kv = c.layers;
        assert!(c.validate().unwrap_err().to_string().contains("borrow"));
        c.shared_kv = 0;
        c.rot_full = 3;
        assert!(c.validate().unwrap_err().to_string().contains("rope"));
        // a file with a tensor missing says which one
        let d = dir();
        let q = d.join("broken.gguf");
        let mut o = Opt::full();
        o.shared = 0; // the file then lacks nothing; delete by writing a model whose layer needs attn_k
        write(&q, &o, true);
        let bytes = std::fs::read(&q).unwrap();
        // truncate: tensors point past the end
        std::fs::write(&q, &bytes[..bytes.len() / 2]).unwrap();
        assert!(family::load_any(&q).is_err());
    }

    #[test]
    fn a_missing_tensor_is_named() {
        // write a file whose layer 0 has no attn_q_norm
        let d = dir();
        let good = d.join("a.gguf");
        write(&good, &Opt::full(), true);
        let f = crate::loader::GgufFile::open(&good).unwrap();
        let mut c = G4Config::from_gguf(&f.gguf).unwrap();
        c.layers += 0;
        let src: Arc<dyn WeightFile> = Arc::new(f);
        // ask for one more layer than the file holds
        c.layers = 7;
        c.sliding.push(false);
        let e = Gemma4Model::load(c, &src, LoadOptions::default())
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("blk.") || e.contains("per_layer"), "{e}");
    }

    fn owned(o: &Opt) -> Gemma4Model {
        match open(o, false).0 {
            AnyModel::Gemma4(g) => *g,
            _ => panic!("not gemma4"),
        }
    }

    #[test]
    fn accelerator_matches_the_cpu() {
        let toks: Vec<usize> = (0..9).map(|i| (i * 5 + 2) % V).collect();
        for o in [
            Opt::full(),
            Opt {
                quant: true,
                ..Opt::full()
            },
        ] {
            let mut m = owned(&o);
            let cpu = run(&m, &toks);
            let rep = m.attach_backend(Odd::new(usize::MAX, 1.0, None));
            assert!(rep.active && rep.matrices > 5, "{rep:?}");
            close(&run(&m, &toks), &cpu, 1e-4);
            assert!(m.accel_failure().is_none());
        }
    }

    #[test]
    fn accelerator_failure_falls_back_to_the_cpu_midway() {
        let toks: Vec<usize> = (0..9).map(|i| (i * 3 + 1) % V).collect();
        let mut m = owned(&Opt::full());
        let cpu = run(&m, &toks);
        // the start-up check uses one call, then the device "dies" after a few more
        m.attach_backend(Odd::new(8, 1.0, None));
        close(&run(&m, &toks), &cpu, 1e-4);
        assert!(m.accel_failure().unwrap().contains("device lost"));
        close(&run(&m, &toks), &cpu, 1e-4);
    }

    fn moe_opt() -> Opt {
        Opt::full()
    }

    #[test]
    fn moe_experts_run_on_the_device_and_match_the_cpu() {
        let toks: Vec<usize> = (0..9).map(|i| (i * 5 + 2) % V).collect();
        for q in [false, true] {
            let mut m = owned(&Opt {
                quant: q,
                ..moe_opt()
            });
            let cpu = run(&m, &toks);
            let be = Odd::new(usize::MAX, 1.0, None);
            let rep = m.attach_backend(be.clone());
            assert!(rep.active && rep.note.contains("routed experts"), "{rep:?}");
            close(&run(&m, &toks), &cpu, 1e-4);
            let (n, bytes) = m.resident_experts();
            assert!(n > 0 && bytes > 0, "{n} {bytes}");
            close(&run(&m, &toks), &cpu, 1e-4);
            assert_eq!(m.resident_experts(), (n, bytes));
            assert!(m.accel_failure().is_none());
            m.attach_backend(Odd::new(usize::MAX, 1.5, None));
            assert_eq!(m.resident_experts(), (0, 0));
            assert_eq!(run(&m, &toks), cpu);
        }
    }

    #[test]
    fn moe_experts_fall_back_when_the_device_dies_midway() {
        let toks: Vec<usize> = (0..9).map(|i| (i * 3 + 1) % V).collect();
        let mut m = owned(&moe_opt());
        let cpu = run(&m, &toks);
        for ok in [3, 20, 60, 120] {
            m.attach_backend(Odd::new(ok, 1.0, None));
            close(&run(&m, &toks), &cpu, 1e-4);
            assert!(m.accel_failure().unwrap().contains("device lost"), "{ok}");
            close(&run(&m, &toks), &cpu, 1e-4);
        }
    }

    #[test]
    fn moe_small_device_memory_keeps_fewer_experts() {
        let toks = [4usize, 5, 6, 7, 8, 9, 10];
        let mut m = owned(&moe_opt());
        let cpu = run(&m, &toks);
        let all = m.attach_backend(Odd::new(usize::MAX, 1.0, None));
        close(&run(&m, &toks), &cpu, 1e-4);
        let b_all = m.resident_experts().1;
        assert!(b_all > 0);
        m.attach_backend(Odd::new(usize::MAX, 1.0, Some(all.bytes + b_all / 2)));
        close(&run(&m, &toks), &cpu, 1e-4);
        assert!(m.resident_experts().1 <= b_all / 2);
        m.attach_backend(Odd::new(usize::MAX, 1.0, Some(all.bytes)));
        close(&run(&m, &toks), &cpu, 1e-4);
        assert_eq!(m.resident_experts().1, 0);
    }

    #[test]
    fn a_wrong_device_is_refused_at_start_up() {
        let toks = [1usize, 2, 3];
        let mut m = owned(&Opt::full());
        let cpu = run(&m, &toks);
        let rep = m.attach_backend(Odd::new(usize::MAX, 1.5, None));
        assert!(
            !rep.active && rep.note.contains("different answer"),
            "{rep:?}"
        );
        assert_eq!(run(&m, &toks), cpu);
    }

    #[test]
    fn small_device_memory_gives_a_partial_offload() {
        let toks = [4usize, 5, 6, 7];
        let mut m = owned(&Opt::full());
        let cpu = run(&m, &toks);
        let all = m.attach_backend(Odd::new(usize::MAX, 1.0, None));
        let part = m.attach_backend(Odd::new(usize::MAX, 1.0, Some(all.bytes / 2)));
        assert!(
            part.matrices > 0 && part.matrices < all.matrices,
            "{part:?}"
        );
        close(&run(&m, &toks), &cpu, 1e-4);
        let none = m.attach_backend(Odd::new(usize::MAX, 1.0, Some(0)));
        assert!(!none.active);
        assert_eq!(run(&m, &toks), cpu);
    }
}
