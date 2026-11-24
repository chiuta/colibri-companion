//! Dense decoder forward pass (Llama / Mistral / Qwen / Gemma-style): RMS-norm, grouped-query
//! attention with RoPE and a KV cache, gated SiLU MLP, optional tied output head.
use crate::arch::Spec;
use crate::ops::{matvec, rmsnorm, rope, silu, softmax};

pub struct Layer {
    pub attn_norm: Vec<f32>, pub wq: Vec<f32>, pub wk: Vec<f32>, pub wv: Vec<f32>, pub wo: Vec<f32>,
    pub ffn_norm: Vec<f32>, pub w_gate: Vec<f32>, pub w_up: Vec<f32>, pub w_down: Vec<f32>,
}
pub struct Weights {
    pub embed: Vec<f32>, pub layers: Vec<Layer>, pub final_norm: Vec<f32>,
    /// `None` = tied to `embed`.
    pub lm_head: Option<Vec<f32>>,
}
pub struct Model { pub spec: Spec, pub w: Weights }

#[derive(Debug, PartialEq, Eq)]
pub enum ModelError { Shape(String), Token(usize), ContextFull(usize) }
impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Shape(s) => write!(f, "weights do not match the architecture: {s}"),
            ModelError::Token(t) => write!(f, "token {t} is outside the vocabulary"),
            ModelError::ContextFull(n) => write!(f, "the context window of {n} tokens is full"),
        }
    }
}
impl std::error::Error for ModelError {}

/// Per-conversation state: the KV cache grows one position at a time.
#[derive(Default)]
pub struct State { k: Vec<Vec<f32>>, v: Vec<Vec<f32>>, pub pos: usize }

impl Model {
    pub fn new(spec: Spec, w: Weights) -> Result<Model, ModelError> {
        let (h, ff, v) = (spec.hidden as usize, spec.ffn as usize, spec.vocab as usize);
        let (qd, kd) = ((spec.heads * spec.head_dim) as usize, (spec.kv_heads * spec.head_dim) as usize);
        if spec.heads % spec.kv_heads != 0 { return Err(ModelError::Shape("heads not divisible by kv heads".into())); }
        let chk = |name: &str, got: usize, want: usize| {
            if got == want { Ok(()) } else { Err(ModelError::Shape(format!("{name}: {got} values, expected {want}"))) }
        };
        chk("embed", w.embed.len(), v * h)?;
        chk("final_norm", w.final_norm.len(), h)?;
        if let Some(l) = &w.lm_head { chk("lm_head", l.len(), v * h)?; }
        chk("layers", w.layers.len(), spec.layers as usize)?;
        for (i, l) in w.layers.iter().enumerate() {
            for (n, got, want) in [("attn_norm", l.attn_norm.len(), h), ("wq", l.wq.len(), qd * h), ("wk", l.wk.len(), kd * h),
                ("wv", l.wv.len(), kd * h), ("wo", l.wo.len(), h * qd), ("ffn_norm", l.ffn_norm.len(), h),
                ("w_gate", l.w_gate.len(), ff * h), ("w_up", l.w_up.len(), ff * h), ("w_down", l.w_down.len(), h * ff)] {
                chk(&format!("layer {i} {n}"), got, want)?;
            }
        }
        Ok(Model { spec, w })
    }

    pub fn new_state(&self) -> State {
        State { k: vec![vec![]; self.spec.layers as usize], v: vec![vec![]; self.spec.layers as usize], pos: 0 }
    }

    /// Feed one token, return the logits for the next one.
    pub fn forward(&self, st: &mut State, token: usize) -> Result<Vec<f32>, ModelError> {
        let s = &self.spec;
        let (h, hd, nh, nkv) = (s.hidden as usize, s.head_dim as usize, s.heads as usize, s.kv_heads as usize);
        let (qd, kd, ff) = (nh * hd, nkv * hd, s.ffn as usize);
        if token >= s.vocab as usize { return Err(ModelError::Token(token)); }
        if st.pos >= s.context as usize { return Err(ModelError::ContextFull(s.context as usize)); }
        let eps = s.norm_eps as f32;
        let mut x = self.w.embed[token * h..(token + 1) * h].to_vec();
        let (mut xb, mut q, mut k, mut v) = (vec![0.0; h], vec![0.0; qd], vec![0.0; kd], vec![0.0; kd]);
        let (mut att, mut proj) = (vec![0.0; qd], vec![0.0; h]);
        let (mut g, mut u) = (vec![0.0; ff], vec![0.0; ff]);
        let group = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        for (li, l) in self.w.layers.iter().enumerate() {
            rmsnorm(&x, &l.attn_norm, eps, &mut xb);
            matvec(&l.wq, h, &xb, &mut q);
            matvec(&l.wk, h, &xb, &mut k);
            matvec(&l.wv, h, &xb, &mut v);
            for hq in q.chunks_mut(hd) { rope(hq, st.pos, s.rope_theta); }
            for hk in k.chunks_mut(hd) { rope(hk, st.pos, s.rope_theta); }
            st.k[li].extend_from_slice(&k);
            st.v[li].extend_from_slice(&v);
            let n = st.pos + 1;
            let mut scores = vec![0.0f32; n];
            for head in 0..nh {
                let kvh = head / group;
                let qh = &q[head * hd..(head + 1) * hd];
                for (t, sc) in scores.iter_mut().enumerate() {
                    let kt = &st.k[li][t * kd + kvh * hd..t * kd + (kvh + 1) * hd];
                    *sc = qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                softmax(&mut scores);
                let out = &mut att[head * hd..(head + 1) * hd];
                out.fill(0.0);
                for (t, p) in scores.iter().enumerate() {
                    let vt = &st.v[li][t * kd + kvh * hd..t * kd + (kvh + 1) * hd];
                    for (o, vv) in out.iter_mut().zip(vt) { *o += p * vv; }
                }
            }
            matvec(&l.wo, qd, &att, &mut proj);
            x.iter_mut().zip(&proj).for_each(|(a, b)| *a += b);
            rmsnorm(&x, &l.ffn_norm, eps, &mut xb);
            matvec(&l.w_gate, h, &xb, &mut g);
            matvec(&l.w_up, h, &xb, &mut u);
            g.iter_mut().zip(&u).for_each(|(a, b)| *a = silu(*a) * b);
            matvec(&l.w_down, ff, &g, &mut proj);
            x.iter_mut().zip(&proj).for_each(|(a, b)| *a += b);
        }
        st.pos += 1;
        rmsnorm(&x, &self.w.final_norm, eps, &mut xb);
        let head = self.w.lm_head.as_deref().unwrap_or(&self.w.embed);
        let mut logits = vec![0.0; s.vocab as usize];
        matvec(head, h, &xb, &mut logits);
        Ok(logits)
    }
}

/// Token sampling. `temperature <= 0` is greedy.
pub struct Sampler { pub temperature: f32, pub top_k: usize, rng: u64 }
impl Sampler {
    pub fn new(temperature: f32, top_k: usize, seed: u64) -> Sampler { Sampler { temperature, top_k, rng: seed | 1 } }
    fn next_f32(&mut self) -> f32 {
        self.rng ^= self.rng << 13; self.rng ^= self.rng >> 7; self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn sample(&mut self, logits: &[f32]) -> usize {
        let argmax = || logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
        if self.temperature <= 0.0 || logits.is_empty() { return argmax(); }
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        idx.truncate(if self.top_k == 0 { logits.len() } else { self.top_k.min(logits.len()) });
        let mut p: Vec<f32> = idx.iter().map(|&i| logits[i] / self.temperature).collect();
        softmax(&mut p);
        let mut r = self.next_f32();
        for (&i, pi) in idx.iter().zip(&p) { if r < *pi { return i; } r -= pi; }
        idx[idx.len() - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Support;

    fn rng(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17;
        ((*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6
    }
    fn mat(n: usize, s: &mut u64) -> Vec<f32> { (0..n).map(|_| rng(s)).collect() }
    fn ones(n: usize, s: &mut u64) -> Vec<f32> { (0..n).map(|_| 1.0 + rng(s) * 0.2).collect() }

    pub fn tiny(heads: u64, kv: u64, tied: bool) -> Model {
        let spec = Spec { family: "llama".into(), support: Support::Standard, layers: 3, hidden: 16, heads, kv_heads: kv,
            head_dim: 8, ffn: 24, vocab: 20, context: 32, rope_theta: 10_000.0, norm_eps: 1e-5,
            experts: 0, experts_used: 0, shared_experts: 0, tied_embeddings: tied };
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let (h, qd, kd, ff) = (16usize, (heads * 8) as usize, (kv * 8) as usize, 24usize);
        let layers = (0..3).map(|_| Layer {
            attn_norm: ones(h, &mut s), wq: mat(qd * h, &mut s), wk: mat(kd * h, &mut s), wv: mat(kd * h, &mut s),
            wo: mat(h * qd, &mut s), ffn_norm: ones(h, &mut s), w_gate: mat(ff * h, &mut s), w_up: mat(ff * h, &mut s),
            w_down: mat(h * ff, &mut s) }).collect();
        let w = Weights { embed: mat(20 * h, &mut s), layers, final_norm: ones(h, &mut s),
            lm_head: if tied { None } else { Some(mat(20 * h, &mut s)) } };
        Model::new(spec, w).unwrap()
    }

    /// Independent reference: recompute every position from scratch with full softmax attention.
    fn reference_logits(m: &Model, tokens: &[usize]) -> Vec<f32> {
        let s = &m.spec;
        let (h, hd, nh, nkv) = (s.hidden as usize, s.head_dim as usize, s.heads as usize, s.kv_heads as usize);
        let mut xs: Vec<Vec<f32>> = tokens.iter().map(|&t| m.w.embed[t * h..(t + 1) * h].to_vec()).collect();
        for l in &m.w.layers {
            let (mut ks, mut vs, mut qs) = (vec![], vec![], vec![]);
            for (p, x) in xs.iter().enumerate() {
                let mut xb = vec![0.0; h];
                rmsnorm(x, &l.attn_norm, s.norm_eps as f32, &mut xb);
                let (mut q, mut k, mut v) = (vec![0.0; nh * hd], vec![0.0; nkv * hd], vec![0.0; nkv * hd]);
                matvec(&l.wq, h, &xb, &mut q); matvec(&l.wk, h, &xb, &mut k); matvec(&l.wv, h, &xb, &mut v);
                q.chunks_mut(hd).for_each(|c| rope(c, p, s.rope_theta));
                k.chunks_mut(hd).for_each(|c| rope(c, p, s.rope_theta));
                qs.push(q); ks.push(k); vs.push(v);
            }
            for p in 0..xs.len() {
                let mut att = vec![0.0; nh * hd];
                for head in 0..nh {
                    let kvh = head / (nh / nkv);
                    let mut sc: Vec<f32> = (0..=p).map(|t| (0..hd).map(|d| qs[p][head * hd + d] * ks[t][kvh * hd + d]).sum::<f32>() / (hd as f32).sqrt()).collect();
                    softmax(&mut sc);
                    for d in 0..hd { att[head * hd + d] = (0..=p).map(|t| sc[t] * vs[t][kvh * hd + d]).sum(); }
                }
                let mut o = vec![0.0; h];
                matvec(&l.wo, nh * hd, &att, &mut o);
                for i in 0..h { xs[p][i] += o[i]; }
                let mut xb = vec![0.0; h];
                rmsnorm(&xs[p], &l.ffn_norm, s.norm_eps as f32, &mut xb);
                let ff = s.ffn as usize;
                let (mut g, mut u, mut d) = (vec![0.0; ff], vec![0.0; ff], vec![0.0; h]);
                matvec(&l.w_gate, h, &xb, &mut g); matvec(&l.w_up, h, &xb, &mut u);
                for i in 0..ff { g[i] = silu(g[i]) * u[i]; }
                matvec(&l.w_down, ff, &g, &mut d);
                for i in 0..h { xs[p][i] += d[i]; }
            }
        }
        let mut xb = vec![0.0; h];
        rmsnorm(xs.last().unwrap(), &m.w.final_norm, s.norm_eps as f32, &mut xb);
        let head = m.w.lm_head.as_deref().unwrap_or(&m.w.embed);
        let mut out = vec![0.0; s.vocab as usize];
        matvec(head, h, &xb, &mut out);
        out
    }

    #[test]
    fn cached_decoding_equals_full_recomputation_for_mha_gqa_and_mqa() {
        for (heads, kv, tied) in [(4, 4, false), (4, 2, true), (4, 1, false)] {
            let m = tiny(heads, kv, tied);
            let toks = [3usize, 7, 1, 19, 0, 7, 12];
            let mut st = m.new_state();
            for (i, &t) in toks.iter().enumerate() {
                let got = m.forward(&mut st, t).unwrap();
                let want = reference_logits(&m, &toks[..=i]);
                for (a, b) in got.iter().zip(&want) { assert!((a - b).abs() < 1e-4, "heads {heads} kv {kv} step {i}: {a} vs {b}"); }
            }
        }
    }

    #[test]
    fn wrong_shapes_tokens_and_full_context_are_errors() {
        let m = tiny(4, 2, false);
        let mut st = m.new_state();
        assert_eq!(m.forward(&mut st, 20), Err(ModelError::Token(20)));
        for _ in 0..32 { m.forward(&mut st, 1).unwrap(); }
        assert_eq!(m.forward(&mut st, 1), Err(ModelError::ContextFull(32)));
        let mut bad = tiny(4, 2, false);
        bad.w.layers[1].wq.pop();
        assert!(matches!(Model::new(bad.spec.clone(), bad.w), Err(ModelError::Shape(_))));
    }

    #[test]
    fn greedy_is_deterministic_and_sampling_respects_top_k() {
        let m = tiny(4, 2, false);
        let run = || { let mut st = m.new_state(); let mut t = 2; let mut g = Sampler::new(0.0, 0, 1);
            (0..6).map(|_| { t = g.sample(&m.forward(&mut st, t).unwrap()); t }).collect::<Vec<_>>() };
        assert_eq!(run(), run());
        let logits = [0.0, 5.0, 4.0, -3.0, 1.0];
        let mut s = Sampler::new(1.0, 2, 42);
        for _ in 0..200 { assert!(matches!(s.sample(&logits), 1 | 2)); }
    }
}
