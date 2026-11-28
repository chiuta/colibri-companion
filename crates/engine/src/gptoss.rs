//! gpt-oss (20B and 120B), run from a GGUF file.
//!
//! Its decoder has features the shared one in `model.rs` does not: learned attention "sinks" (one
//! extra logit per head that takes part in the softmax but carries no value), biases on every
//! attention projection, sliding-window attention on every other layer, YaRN rotary embeddings, and
//! a mixture of experts whose members have biases and a clamped, sigmoid-gated activation
//! (`(up + 1) * gate * sigmoid(1.702 * gate)`, both halves limited to ±7), routed by a softmax over
//! the top-k logits only. The experts are usually stored as MXFP4. Own model, own cache, reached
//! through `family::load_any`.
//!
//! Written from the published architecture. Checked on small synthetic models; **not** compared
//! with a real gpt-oss file.
use crate::accel::{Accel, AttachReport, Item, key_of};
use crate::arch::RopeScaling;
use crate::backend::Backend;
use crate::gguf::Gguf;
use crate::loader::{LoadOptions, TensorSource, WeightFile, load_weight};
use crate::model::{ModelError, QWeight, Weight, rope_apply, rope_inv_freq};
use std::sync::Arc;

#[derive(Debug)]
pub enum GptOssError {
    Config(String),
    Load(String),
}
impl std::fmt::Display for GptOssError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GptOssError::Config(s) | GptOssError::Load(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for GptOssError {}
fn load_err(e: impl std::fmt::Display) -> GptOssError {
    GptOssError::Load(e.to_string())
}

/// GGUF architecture names this module runs.
pub fn handles(arch: &str) -> bool {
    matches!(arch, "gpt-oss" | "gpt_oss" | "gptoss")
}

const ALPHA: f32 = 1.702;
const LIMIT: f32 = 7.0;

#[derive(Clone, Debug, PartialEq)]
pub struct GptOssConfig {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub context: usize,
    pub eps: f32,
    pub window: usize,
    pub experts: usize,
    pub used: usize,
    pub ff: usize,
    pub theta: f64,
    pub yarn_factor: f64,
    pub yarn_original: f64,
}

impl GptOssConfig {
    pub fn from_gguf(g: &Gguf) -> Result<GptOssConfig, GptOssError> {
        let need = |k: &str| {
            g.get_u64(k)
                .filter(|v| *v > 0)
                .ok_or_else(|| GptOssError::Config(format!("gpt-oss.{k} is missing")))
        };
        let layers = need("block_count")? as usize;
        let hidden = need("embedding_length")? as usize;
        let heads = need("attention.head_count")? as usize;
        let experts = need("expert_count")? as usize;
        let used = need("expert_used_count")? as usize;
        if layers > 512 || hidden > 1 << 16 || heads > 1 << 10 || experts > 4096 {
            return Err(GptOssError::Config("implausible model size".into()));
        }
        let kv_heads = g
            .get_u64("attention.head_count_kv")
            .filter(|v| *v > 0)
            .map_or(heads, |v| v as usize);
        let head_dim = g
            .get_u64("attention.key_length")
            .filter(|v| *v > 0)
            .map_or(hidden / heads, |v| v as usize);
        if !heads.is_multiple_of(kv_heads) || head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(GptOssError::Config(format!(
                "{heads} heads, {kv_heads} key/value heads, head size {head_dim} do not fit"
            )));
        }
        if used > experts {
            return Err(GptOssError::Config(
                "more experts used than the model has".into(),
            ));
        }
        Ok(GptOssConfig {
            layers,
            hidden,
            heads,
            kv_heads,
            head_dim,
            context: g.get_u64("context_length").unwrap_or(131_072) as usize,
            eps: g
                .get_f64("attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-5) as f32,
            window: g.get_u64("attention.sliding_window").unwrap_or(128).max(1) as usize,
            experts,
            used,
            ff: g
                .get_u64("expert_feed_forward_length")
                .or_else(|| g.get_u64("feed_forward_length"))
                .map_or(hidden, |v| v as usize),
            theta: g.get_f64("rope.freq_base").unwrap_or(150_000.0),
            yarn_factor: g.get_f64("rope.scaling.factor").unwrap_or(32.0).max(1.0),
            yarn_original: g
                .get_u64("rope.scaling.original_context_length")
                .unwrap_or(4096) as f64,
        })
    }

    /// Layers 0, 2, 4… look only at the last `window` tokens; the others see everything.
    pub fn sliding(&self, layer: usize) -> bool {
        layer.is_multiple_of(2)
    }
}

struct Bank {
    src: Arc<dyn WeightFile>,
    gate: String,
    up: String,
    down: String,
    hidden: usize,
    ff: usize,
    mmap: bool,
}
impl Bank {
    fn weight(
        &self,
        name: &str,
        start: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Weight, GptOssError> {
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
    fn bias(&self, name: &str, e: usize, n: usize) -> Result<Vec<f32>, GptOssError> {
        self.src.read_range(name, e * n, n).map_err(load_err)
    }
    /// `(gate, up, down)` weights and biases of expert `e`.
    fn get(&self, e: usize) -> Result<Expert, GptOssError> {
        let (h, ff) = (self.hidden, self.ff);
        let b = |n: &str| format!("{}.bias", n.trim_end_matches(".weight"));
        Ok(Expert {
            gate: self.weight(&self.gate, e * ff * h, ff, h)?,
            up: self.weight(&self.up, e * ff * h, ff, h)?,
            down: self.weight(&self.down, e * h * ff, h, ff)?,
            gate_b: self.bias(&b(&self.gate), e, ff)?,
            up_b: self.bias(&b(&self.up), e, ff)?,
            down_b: self.bias(&b(&self.down), e, h)?,
        })
    }
}
struct Expert {
    gate: Weight,
    up: Weight,
    down: Weight,
    gate_b: Vec<f32>,
    up_b: Vec<f32>,
    down_b: Vec<f32>,
}

struct Layer {
    sliding: bool,
    attn_norm: Vec<f32>,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    bq: Vec<f32>,
    bk: Vec<f32>,
    bv: Vec<f32>,
    bo: Vec<f32>,
    sinks: Vec<f32>,
    ffn_norm: Vec<f32>,
    router: Vec<f32>,
    router_b: Vec<f32>,
    bank: Bank,
}

pub struct GptOssModel {
    cfg: GptOssConfig,
    embed: Weight,
    head: Option<Weight>,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    vocab: usize,
    rope: (Vec<f64>, f64),
    accel: Option<Accel>,
}

#[derive(Default, Clone)]
struct Cache {
    k: Vec<f32>,
    v: Vec<f32>,
    first: usize,
}
pub struct GptOssState {
    caches: Vec<Cache>,
    pub pos: usize,
}
impl GptOssState {
    pub fn bytes(&self) -> usize {
        self.caches
            .iter()
            .map(|c| (c.k.len() + c.v.len()) * 4)
            .sum()
    }
}

fn vec1(src: &dyn TensorSource, name: &str, n: usize) -> Result<Vec<f32>, GptOssError> {
    let v = src
        .read(name)
        .map_err(|_| GptOssError::Load(format!("{name} is missing or unreadable")))?;
    if v.len() != n {
        return Err(GptOssError::Load(format!(
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
) -> Result<(Weight, usize), GptOssError> {
    let w = load_weight(src, name, o).map_err(load_err)?;
    let len = w.len();
    if cols == 0 || !len.is_multiple_of(cols) {
        return Err(GptOssError::Load(format!(
            "{name}: {len} values do not fill rows of {cols}"
        )));
    }
    let r = len / cols;
    if let Some(want) = rows
        && want != r
    {
        return Err(GptOssError::Load(format!(
            "{name}: {r} rows, expected {want}"
        )));
    }
    Ok((w, r))
}

fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * inv * w).collect()
}

/// The activation of one expert: both halves limited to ±7, `(up + 1) * gate * sigmoid(1.702 gate)`.
pub fn expert_act(gate: f32, up: f32) -> f32 {
    let g = gate.min(LIMIT);
    let u = up.clamp(-LIMIT, LIMIT);
    (u + 1.0) * g / (1.0 + (-ALPHA * g).exp())
}

/// Best `k` logits, softmax over those only.
pub fn route(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    idx.truncate(k);
    let m = idx
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = idx.iter().map(|&i| (logits[i] - m).exp()).collect();
    let z: f32 = e.iter().sum();
    idx.into_iter().zip(e).map(|(i, e)| (i, e / z)).collect()
}

impl GptOssModel {
    pub fn load(
        cfg: GptOssConfig,
        src: &Arc<dyn WeightFile>,
        o: LoadOptions,
    ) -> Result<GptOssModel, GptOssError> {
        let s: &dyn TensorSource = src.as_ref();
        let h = cfg.hidden;
        let (qd, kd) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        let (embed, vocab) = mat(s, "token_embd.weight", None, h, o)?;
        let head = if s.has("output.weight") {
            Some(mat(s, "output.weight", Some(vocab), h, o)?.0)
        } else {
            None
        };
        let final_norm = vec1(s, "output_norm.weight", h)?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let n = |p: &str| format!("blk.{i}.{p}");
            let ffn_norm = if s.has(&n("post_attention_norm.weight")) {
                vec1(s, &n("post_attention_norm.weight"), h)?
            } else {
                vec1(s, &n("ffn_norm.weight"), h)?
            };
            layers.push(Layer {
                sliding: cfg.sliding(i),
                attn_norm: vec1(s, &n("attn_norm.weight"), h)?,
                wq: mat(s, &n("attn_q.weight"), Some(qd), h, o)?.0,
                wk: mat(s, &n("attn_k.weight"), Some(kd), h, o)?.0,
                wv: mat(s, &n("attn_v.weight"), Some(kd), h, o)?.0,
                wo: mat(s, &n("attn_output.weight"), Some(h), qd, o)?.0,
                bq: vec1(s, &n("attn_q.bias"), qd)?,
                bk: vec1(s, &n("attn_k.bias"), kd)?,
                bv: vec1(s, &n("attn_v.bias"), kd)?,
                bo: vec1(s, &n("attn_output.bias"), h)?,
                sinks: vec1(s, &n("attn_sinks.weight"), cfg.heads)?,
                ffn_norm,
                router: vec1(s, &n("ffn_gate_inp.weight"), cfg.experts * h)?,
                router_b: vec1(s, &n("ffn_gate_inp.bias"), cfg.experts)?,
                bank: Bank {
                    src: src.clone(),
                    gate: n("ffn_gate_exps.weight"),
                    up: n("ffn_up_exps.weight"),
                    down: n("ffn_down_exps.weight"),
                    hidden: h,
                    ff: cfg.ff,
                    mmap: o.mmap,
                },
            });
        }
        let ms = 1.0 + 0.1 * cfg.yarn_factor.ln();
        let rope = rope_inv_freq(
            cfg.theta,
            cfg.head_dim,
            RopeScaling::Yarn {
                factor: cfg.yarn_factor,
                original_max: cfg.yarn_original,
                beta_fast: 32.0,
                beta_slow: 1.0,
                mscale: ms,
            },
        );
        Ok(GptOssModel {
            cfg,
            embed,
            head,
            final_norm,
            layers,
            vocab,
            rope,
            accel: None,
        })
    }

    pub fn config(&self) -> &GptOssConfig {
        &self.cfg
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn new_state(&self) -> GptOssState {
        GptOssState {
            caches: vec![Cache::default(); self.cfg.layers],
            pos: 0,
        }
    }

    /// `out = W · x` on the device when the matrix lives there and it still works, else on the CPU.
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

    /// Put the attention matrices and the output head on `be` (the experts are read per token and
    /// stay on the CPU). Never fails: without a usable device the CPU keeps running.
    pub fn attach_backend(&mut self, be: Arc<dyn Backend>) -> AttachReport {
        self.accel = None;
        let (h, qd) = (self.cfg.hidden, self.cfg.heads * self.cfg.head_dim);
        let head = self.head.as_ref().unwrap_or(&self.embed);
        let mut items = vec![Item::new(head, h, true)];
        for l in &self.layers {
            items.extend([
                Item::new(&l.wq, h, false),
                Item::new(&l.wk, h, false),
                Item::new(&l.wv, h, false),
                Item::new(&l.wo, qd, false),
            ]);
        }
        let (acc, rep) = Accel::build(be, &items);
        self.accel = acc;
        rep
    }

    fn experts(&self, l: &Layer, x: &[f32]) -> Result<Vec<f32>, ModelError> {
        let (h, ff) = (self.cfg.hidden, self.cfg.ff);
        let logits: Vec<f32> = l
            .router
            .chunks_exact(h)
            .zip(&l.router_b)
            .map(|(row, b)| row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>() + b)
            .collect();
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::Shape("the router produced a non-number".into()));
        }
        let mut out = vec![0.0f32; h];
        for (e, w) in route(&logits, self.cfg.used) {
            let ex = l
                .bank
                .get(e)
                .map_err(|e| ModelError::Experts(e.to_string()))?;
            let (mut g, mut u, mut d) = (vec![0.0; ff], vec![0.0; ff], vec![0.0; h]);
            ex.gate.matvec(h, x, &mut g)?;
            ex.up.matvec(h, x, &mut u)?;
            for i in 0..ff {
                g[i] = expert_act(g[i] + ex.gate_b[i], u[i] + ex.up_b[i]);
            }
            ex.down.matvec(ff, &g, &mut d)?;
            out.iter_mut()
                .zip(d.iter().zip(&ex.down_b))
                .for_each(|(o, (d, b))| *o += w * (d + b));
        }
        Ok(out)
    }

    fn step(&self, st: &mut GptOssState, token: usize, want: bool) -> Result<Vec<f32>, ModelError> {
        let c = &self.cfg;
        let (h, nh, nkv, hd, eps, pos) = (c.hidden, c.heads, c.kv_heads, c.head_dim, c.eps, st.pos);
        if token >= self.vocab {
            return Err(ModelError::Token(token));
        }
        if pos >= c.context {
            return Err(ModelError::Shape(format!(
                "position {pos} is past the context of {}",
                c.context
            )));
        }
        let mut x = vec![0.0; h];
        let mut tmp = vec![];
        self.embed.row_into(token, h, &mut x, &mut tmp)?;
        let group = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        for (li, l) in self.layers.iter().enumerate() {
            let a = rms(&x, &l.attn_norm, eps);
            let (mut q, mut k, mut v) =
                (vec![0.0; nh * hd], vec![0.0; nkv * hd], vec![0.0; nkv * hd]);
            self.mv(&l.wq, h, &a, &mut q)?;
            self.mv(&l.wk, h, &a, &mut k)?;
            self.mv(&l.wv, h, &a, &mut v)?;
            for (t, b) in [(&mut q, &l.bq), (&mut k, &l.bk), (&mut v, &l.bv)] {
                t.iter_mut().zip(b).for_each(|(a, b)| *a += b);
            }
            for hh in 0..nh {
                rope_apply(&mut q[hh * hd..(hh + 1) * hd], pos, &self.rope);
            }
            for hh in 0..nkv {
                rope_apply(&mut k[hh * hd..(hh + 1) * hd], pos, &self.rope);
            }
            let cache = &mut st.caches[li];
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
            let stored = cache.k.len() / (nkv * hd);
            let from = if l.sliding {
                (pos + 1).saturating_sub(c.window).max(cache.first)
            } else {
                cache.first
            };
            let (lo, hi) = (from - cache.first, stored);
            let mut att = vec![0.0f32; nh * hd];
            let mut sc = vec![0.0f32; hi - lo];
            for head in 0..nh {
                let kvh = head / group;
                let qh = &q[head * hd..(head + 1) * hd];
                for (t, s) in (lo..hi).zip(sc.iter_mut()) {
                    let kk = &cache.k[(t * nkv + kvh) * hd..(t * nkv + kvh + 1) * hd];
                    *s = qh.iter().zip(kk).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                // the sink is one more logit in the softmax with no value behind it
                let sink = l.sinks[head];
                let m = sc.iter().cloned().fold(sink, f32::max);
                let mut z = (sink - m).exp();
                for s in sc.iter_mut() {
                    *s = (*s - m).exp();
                    z += *s;
                }
                let o = &mut att[head * hd..(head + 1) * hd];
                for (t, s) in (lo..hi).zip(&sc) {
                    let vv = &cache.v[(t * nkv + kvh) * hd..(t * nkv + kvh + 1) * hd];
                    let w = s / z;
                    o.iter_mut().zip(vv).for_each(|(o, v)| *o += w * v);
                }
            }
            let mut proj = vec![0.0; h];
            self.mv(&l.wo, nh * hd, &att, &mut proj)?;
            proj.iter_mut().zip(&l.bo).for_each(|(a, b)| *a += b);
            x.iter_mut().zip(&proj).for_each(|(a, b)| *a += b);
            let f = self.experts(l, &rms(&x, &l.ffn_norm, eps))?;
            x.iter_mut().zip(&f).for_each(|(a, b)| *a += b);
        }
        st.pos += 1;
        if !want {
            return Ok(vec![]);
        }
        let xn = rms(&x, &self.final_norm, eps);
        let mut logits = vec![0.0; self.vocab];
        self.mv(
            self.head.as_ref().unwrap_or(&self.embed),
            h,
            &xn,
            &mut logits,
        )?;
        Ok(logits)
    }

    /// Feed `tokens`, return the logits of the last one. After an `Err` the state must be discarded.
    pub fn forward_batch(
        &self,
        st: &mut GptOssState,
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

    const H: usize = 32;
    const FF: usize = 32;
    const NH: usize = 4;
    const NKV: usize = 2;
    const HD: usize = 8;
    const E: usize = 4;
    const V: usize = 30;
    const L: usize = 4;

    struct Rng(u64);
    impl Rng {
        fn mat(&mut self, n: usize, s: f32) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 ^= self.0 << 13;
                    self.0 ^= self.0 >> 7;
                    self.0 ^= self.0 << 17;
                    ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * s
                })
                .collect()
        }
    }

    #[derive(Clone, Copy)]
    struct Opt {
        mxfp4: bool,
        sink: f32,
        scale: f32,
        window: u32,
    }
    impl Opt {
        fn new() -> Opt {
            Opt {
                mxfp4: false,
                sink: 0.0,
                scale: 0.6,
                window: 3,
            }
        }
    }

    /// `quant_round_trip` = store the experts as f32 but with MXFP4's rounding applied.
    fn write(path: &std::path::Path, o: Opt, round_trip: bool) {
        let mut r = Rng(0x1234_5678_9abc_def1);
        let mut t: Vec<(String, Vec<u64>, DType, Vec<u8>)> = vec![];
        let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let mut f = |n: &str, s: &[usize], d: Vec<f32>| {
            assert_eq!(s.iter().product::<usize>(), d.len(), "{n}");
            t.push((
                n.into(),
                s.iter().map(|x| *x as u64).collect(),
                DType::F32,
                f32b(&d),
            ));
        };
        let norm = |r: &mut Rng, n| r.mat(n, 0.4).iter().map(|v| 1.0 + v).collect::<Vec<f32>>();
        f("token_embd.weight", &[V, H], r.mat(V * H, o.scale));
        f("output_norm.weight", &[H], norm(&mut r, H));
        f("output.weight", &[V, H], r.mat(V * H, o.scale));
        let mut big: Vec<(String, Vec<usize>, Vec<f32>)> = vec![];
        for i in 0..L {
            let n = |p: &str| format!("blk.{i}.{p}");
            f(&n("attn_norm.weight"), &[H], norm(&mut r, H));
            f(
                &n("attn_q.weight"),
                &[NH * HD, H],
                r.mat(NH * HD * H, o.scale),
            );
            f(
                &n("attn_k.weight"),
                &[NKV * HD, H],
                r.mat(NKV * HD * H, o.scale),
            );
            f(
                &n("attn_v.weight"),
                &[NKV * HD, H],
                r.mat(NKV * HD * H, o.scale),
            );
            f(
                &n("attn_output.weight"),
                &[H, NH * HD],
                r.mat(H * NH * HD, o.scale),
            );
            f(&n("attn_q.bias"), &[NH * HD], r.mat(NH * HD, 0.2));
            f(&n("attn_k.bias"), &[NKV * HD], r.mat(NKV * HD, 0.2));
            f(&n("attn_v.bias"), &[NKV * HD], r.mat(NKV * HD, 0.2));
            f(&n("attn_output.bias"), &[H], r.mat(H, 0.2));
            f(&n("attn_sinks.weight"), &[NH], vec![o.sink; NH]);
            f(&n("post_attention_norm.weight"), &[H], norm(&mut r, H));
            f(&n("ffn_gate_inp.weight"), &[E, H], r.mat(E * H, 1.0));
            f(&n("ffn_gate_inp.bias"), &[E], r.mat(E, 0.2));
            for (nm, shape) in [
                ("ffn_gate_exps", vec![E, FF, H]),
                ("ffn_up_exps", vec![E, FF, H]),
                ("ffn_down_exps", vec![E, H, FF]),
            ] {
                let d = r.mat(shape.iter().product(), o.scale);
                big.push((n(&format!("{nm}.weight")), shape.clone(), d));
                let (bn, bl) = (n(&format!("{nm}.bias")), shape[0] * shape[1]);
                f(&bn, &[shape[0], shape[1]], r.mat(bl, 0.2));
            }
        }
        for (name, shape, d) in big {
            if o.mxfp4 && !round_trip {
                t.push((
                    name,
                    shape.iter().map(|x| *x as u64).collect(),
                    DType::MXFP4,
                    crate::quant::enc_mxfp4(&d),
                ));
            } else {
                let d = if round_trip {
                    let mut back = vec![];
                    crate::quant::dequantize(
                        DType::MXFP4,
                        &crate::quant::enc_mxfp4(&d),
                        d.len(),
                        &mut back,
                    )
                    .unwrap();
                    back
                } else {
                    d
                };
                t.push((
                    name,
                    shape.iter().map(|x| *x as u64).collect(),
                    DType::F32,
                    f32b(&d),
                ));
            }
        }
        let a = "gpt-oss";
        let meta: Vec<(String, M)> = vec![
            ("general.architecture".into(), M::str(a)),
            (format!("{a}.block_count"), M::U32(L as u32)),
            (format!("{a}.context_length"), M::U32(64)),
            (format!("{a}.embedding_length"), M::U32(H as u32)),
            (format!("{a}.attention.head_count"), M::U32(NH as u32)),
            (format!("{a}.attention.head_count_kv"), M::U32(NKV as u32)),
            (format!("{a}.attention.key_length"), M::U32(HD as u32)),
            (format!("{a}.attention.sliding_window"), M::U32(o.window)),
            (format!("{a}.expert_count"), M::U32(E as u32)),
            (format!("{a}.expert_used_count"), M::U32(2)),
            (format!("{a}.expert_feed_forward_length"), M::U32(FF as u32)),
            (format!("{a}.rope.freq_base"), M::F32(150000.0)),
            (format!("{a}.rope.scaling.factor"), M::F32(32.0)),
            (
                format!("{a}.rope.scaling.original_context_length"),
                M::U32(16),
            ),
        ];
        let descs: Vec<TensorDesc> = t
            .iter()
            .map(|(n, s, d, _)| TensorDesc::new(n.clone(), s, *d))
            .collect();
        let file = std::fs::File::create(path).unwrap();
        GgufWriter::write_all(file, &meta, descs, |i, _| Ok(t[i].3.clone())).unwrap();
    }

    fn open(o: Opt, round_trip: bool) -> GptOssModel {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "gptoss-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("m.gguf");
        write(&p, o, round_trip);
        match family::load_any(&p).unwrap() {
            AnyModel::GptOss(m) => *m,
            _ => panic!("not gpt-oss"),
        }
    }
    fn run(m: &GptOssModel, toks: &[usize]) -> Vec<f32> {
        m.forward_batch(&mut m.new_state(), toks).unwrap()
    }
    const TOKS: [usize; 9] = [3, 9, 1, 22, 5, 5, 17, 2, 11];

    #[test]
    fn runs_finite_deterministic_and_batch_equals_steps() {
        let m = open(Opt::new(), false);
        let a = run(&m, &TOKS);
        assert_eq!(a.len(), V);
        assert!(a.iter().all(|v| v.is_finite()) && a.iter().any(|v| v.abs() > 1e-4));
        assert_eq!(a, run(&m, &TOKS));
        let mut st = m.new_state();
        let mut last = vec![];
        for t in TOKS {
            last = m.forward_batch(&mut st, &[t]).unwrap();
        }
        assert_eq!(a, last);
        assert_eq!(st.pos, TOKS.len());
    }

    #[test]
    fn the_sink_takes_attention_weight_away() {
        let a = run(&open(Opt::new(), false), &TOKS);
        let b = run(
            &open(
                Opt {
                    sink: 4.0,
                    ..Opt::new()
                },
                false,
            ),
            &TOKS,
        );
        let d = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(d > 1e-4, "{d}");
    }

    #[test]
    fn sliding_layers_forget_old_tokens_and_the_cache_stays_bounded() {
        let m = open(Opt::new(), false);
        let mut st = m.new_state();
        let long: Vec<usize> = (0..60).map(|i| (i * 7 + 1) % V).collect();
        m.forward_batch(&mut st, &long).unwrap();
        // sliding layers 0 and 2 keep at most two windows; full layers keep everything
        let per = NKV * HD * 2 * 4;
        assert!(st.caches[0].k.len() * 4 / (NKV * HD * 4) <= 2 * 3);
        assert_eq!(st.caches[1].k.len() / (NKV * HD), 60);
        assert!(st.bytes() > 0 && st.bytes() < 4 * 60 * per);
    }

    #[test]
    fn mxfp4_experts_equal_their_decoded_twin() {
        let q = open(
            Opt {
                mxfp4: true,
                ..Opt::new()
            },
            false,
        );
        let twin = open(
            Opt {
                mxfp4: true,
                ..Opt::new()
            },
            true,
        );
        // the quantised path also rounds the activations to 8 bits, so allow a few percent
        let (a, b) = (run(&q, &TOKS), run(&twin, &TOKS));
        let scale = b.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 0.05 * scale, "{x} {y}");
        }
    }

    #[test]
    fn activation_and_routing_follow_the_formulas() {
        // both halves are limited to 7, and `up + 1` scales the gated value
        assert_eq!(expert_act(100.0, 100.0), expert_act(7.0, 7.0));
        assert!((expert_act(0.0, 3.0) - 0.0).abs() < 1e-6);
        let g = 1.5f32;
        let want = (2.0 + 1.0) * g / (1.0 + (-1.702 * g).exp());
        assert!((expert_act(g, 2.0) - want).abs() < 1e-6);
        // softmax over the chosen logits only, so the weights sum to 1
        let r = route(&[0.1, 2.0, -1.0, 1.9], 2);
        assert_eq!(r.iter().map(|p| p.0).collect::<Vec<_>>(), [1, 3]);
        assert!((r.iter().map(|p| p.1).sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(r[0].1 > r[1].1);
    }

    #[test]
    fn accelerator_matches_the_cpu_and_falls_back_on_failure() {
        use crate::accel::testbe::Odd;
        let mut m = open(Opt::new(), false);
        let cpu = run(&m, &TOKS);
        let rep = m.attach_backend(Odd::new(usize::MAX, 1.0, None));
        assert!(rep.active && rep.matrices > 4, "{rep:?}");
        for (x, y) in run(&m, &TOKS).iter().zip(&cpu) {
            assert!((x - y).abs() < 1e-4);
        }
        assert!(m.accel_failure().is_none());
        m.attach_backend(Odd::new(8, 1.0, None));
        let r = run(&m, &TOKS);
        assert!(m.accel_failure().is_some());
        for (x, y) in r.iter().zip(&cpu) {
            assert!((x - y).abs() < 1e-4);
        }
        assert!(!m.attach_backend(Odd::new(usize::MAX, 1.5, None)).active);
    }

    #[test]
    fn bad_input_is_an_error_not_a_panic() {
        let m = open(Opt::new(), false);
        assert!(m.forward_batch(&mut m.new_state(), &[]).is_err());
        assert!(m.forward_batch(&mut m.new_state(), &[V]).is_err());
        let long: Vec<usize> = (0..70).map(|i| i % V).collect();
        assert!(m.forward_batch(&mut m.new_state(), &long).is_err());
    }
}
