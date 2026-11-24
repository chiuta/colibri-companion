//! Speculative decoding, independent of any concrete model.
//!
//! A cheap *drafter* proposes `k` tokens; the expensive *target* checks all of them in ONE
//! batched pass and keeps the longest prefix it agrees with, plus one token of its own. With
//! greedy decoding the output is token-for-token what plain greedy decoding would give. With
//! sampling, the acceptance rule of Leviathan et al. / Chen et al. ("accept with probability
//! `min(1, p/q)`, otherwise resample from `max(0, p - q)` normalised") keeps the output
//! distribution EXACTLY equal to sampling from the target.
//!
//! # Protocol with the target ([`Verifier`])
//! The target holds the KV state of `ctx[..len-1]`; the last context token is *pending* (not yet
//! fed). Each round feeds `[pending, d1..dk]` and gets `k+1` logit vectors: vector `i` is the
//! distribution of the token that follows batch position `i`. After deciding how many drafts
//! `a` were accepted, the round calls `rollback(fed + 1 + a)` (when `a < k`) so the rejected
//! drafts leave no trace, and the freshly emitted token becomes the new pending token.
//!
//! # Drafters
//! * [`DraftModel`]   – a smaller model run autoregressively (own rollback), optionally with a
//!   second candidate for the first position (`two_candidates`).
//! * [`NgramLookup`]  – prompt-lookup decoding: find the last n tokens earlier in the context
//!   and propose what followed them.
//! * [`NgramMap`]     – frequency statistics over n-grams: propose the most frequent follower.
//! * [`CacheDrafter`] – a bounded [`SuffixCache`] of earlier generations that survives across
//!   requests.
//! * [`Chain`]        – first drafter that returns something wins.
//!
//! Deterministic drafters (everything except [`DraftModel`]) propose with a one-hot `q`, for
//! which the rule above reduces to "accept with probability `p(d)`, else resample from `p`
//! without `d`".
use std::collections::{HashMap, VecDeque};

// ------------------------------------------------------------------------------------------
// Traits
// ------------------------------------------------------------------------------------------

/// The target model, seen as a stateful sequence evaluator.
pub trait Verifier {
    /// Feed `tokens` after the tokens already held and return one logit vector per fed token
    /// (vector `i` = distribution of the token that follows `tokens[i]`).
    fn logits_batch(&mut self, tokens: &[u32]) -> Vec<Vec<f32>>;
    /// Keep only the first `keep` positions of the state.
    fn rollback(&mut self, keep: usize);
}

/// What a drafter proposes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Draft {
    pub tokens: Vec<u32>,
    /// The (already processed: temperature, top-k, top-p) distribution each token was drawn
    /// from. `None` means the proposal was deterministic (one-hot).
    pub dists: Option<Vec<Vec<f32>>>,
    /// A second, independent draw from `dists[0]` for the first position (sampling only).
    pub alt: Option<u32>,
}
impl Draft {
    pub fn empty() -> Draft {
        Draft::default()
    }
    pub fn deterministic(tokens: Vec<u32>) -> Draft {
        Draft {
            tokens,
            dists: None,
            alt: None,
        }
    }
}

pub trait Drafter {
    /// Propose up to `k` tokens continuing `ctx` (which ends with the pending token).
    fn draft(&mut self, ctx: &[u32], k: usize, cfg: &SamplingCfg, rng: &mut Rng) -> Draft;
    /// Called after every round (and at the end) with the full, committed context.
    fn observe(&mut self, _ctx: &[u32]) {}
    fn name(&self) -> &'static str {
        "drafter"
    }
}

// ------------------------------------------------------------------------------------------
// Randomness and sampling helpers
// ------------------------------------------------------------------------------------------

/// SplitMix64: small, fast, good enough for acceptance tests and reproducible.
#[derive(Clone, Debug)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Index of the largest finite value (first one on ties); NaN counts as -inf. 0 for empty input.
pub fn argmax(x: &[f32]) -> u32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    let mut seen = false;
    for (i, &v) in x.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if !seen || v > best.1 {
            best = (i, v);
            seen = true;
        }
    }
    best.0 as u32
}

/// Sampling configuration applied identically to target and drafter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingCfg {
    /// `<= 0` (or NaN) means greedy.
    pub temperature: f32,
    /// 0 = off.
    pub top_k: usize,
    /// 1.0 (or more) = off.
    pub top_p: f32,
}
impl Default for SamplingCfg {
    fn default() -> Self {
        SamplingCfg {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}
impl SamplingCfg {
    pub fn greedy() -> SamplingCfg {
        SamplingCfg::default()
    }
    pub fn sampled(temperature: f32, top_k: usize, top_p: f32) -> SamplingCfg {
        SamplingCfg {
            temperature,
            top_k,
            top_p,
        }
    }
    pub fn is_greedy(&self) -> bool {
        self.temperature.is_nan() || self.temperature <= 0.0
    }

    /// Logits → probability distribution (always finite, sums to 1 unless `logits` is empty).
    /// Greedy gives a one-hot vector.
    pub fn probs(&self, logits: &[f32]) -> Vec<f32> {
        let n = logits.len();
        if n == 0 {
            return vec![];
        }
        if self.is_greedy() {
            let mut p = vec![0.0; n];
            p[argmax(logits) as usize] = 1.0;
            return p;
        }
        let uniform = || vec![1.0 / n as f32; n];
        if self.temperature.is_infinite() {
            return uniform();
        }
        let t = self.temperature as f64;
        let l: Vec<f64> = logits
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    -1e30
                } else {
                    (v as f64).clamp(-1e30, 1e30) / t
                }
            })
            .collect();
        let m = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut p: Vec<f64> = l.iter().map(|&v| (v - m).exp()).collect();
        let mut s: f64 = p.iter().sum();
        if !(s.is_finite() && s > 0.0) {
            return uniform();
        }
        p.iter_mut().for_each(|v| *v /= s);
        if self.top_k > 0 && self.top_k < n || (self.top_p > 0.0 && self.top_p < 1.0) {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
            let mut keep = n;
            if self.top_k > 0 {
                keep = keep.min(self.top_k);
            }
            if self.top_p > 0.0 && self.top_p < 1.0 {
                let mut cum = 0.0;
                let mut kp = n;
                for (rank, &i) in idx.iter().enumerate() {
                    cum += p[i];
                    if cum >= self.top_p as f64 {
                        kp = rank + 1;
                        break;
                    }
                }
                keep = keep.min(kp);
            }
            let keep = keep.max(1);
            for &i in &idx[keep..] {
                p[i] = 0.0;
            }
            s = p.iter().sum();
            p.iter_mut().for_each(|v| *v /= s);
        }
        p.into_iter().map(|v| v as f32).collect()
    }
}

/// Draw an index from a (nearly) normalised distribution; never returns a zero-probability index
/// unless the whole vector is zero.
pub fn sample_from(p: &[f32], rng: &mut Rng) -> u32 {
    let total: f64 = p.iter().map(|&v| v.max(0.0) as f64).sum();
    if total.is_nan() || total <= 0.0 {
        return 0;
    }
    let u = rng.f64() * total;
    let mut cum = 0.0;
    let mut last = 0usize;
    for (i, &v) in p.iter().enumerate() {
        if v > 0.0 {
            cum += v as f64;
            last = i;
            if u < cum {
                return i as u32;
            }
        }
    }
    last as u32
}

/// `max(0, p - q)` normalised; `q = None` means one-hot at `d`. Falls back to `p` if empty.
fn residual(p: &[f32], q: Option<&[f32]>, d: u32) -> Vec<f32> {
    let mut r: Vec<f32> = match q {
        Some(q) => p.iter().zip(q).map(|(&a, &b)| (a - b).max(0.0)).collect(),
        None => {
            let mut r = p.to_vec();
            if let Some(x) = r.get_mut(d as usize) {
                *x = 0.0;
            }
            r
        }
    };
    let s: f64 = r.iter().map(|&v| v as f64).sum();
    if s > 1e-12 {
        r.iter_mut().for_each(|v| *v = (*v as f64 / s) as f32);
        r
    } else {
        p.to_vec()
    }
}

fn accept_prob(p: f32, q: f32) -> f64 {
    if q > 0.0 {
        (p as f64 / q as f64).min(1.0)
    } else if p > 0.0 {
        1.0
    } else {
        0.0
    }
}

// ------------------------------------------------------------------------------------------
// Verification rules
// ------------------------------------------------------------------------------------------

/// Result of verifying one round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verdict {
    /// Number of drafted tokens accepted (a prefix).
    pub accepted: usize,
    /// The token the target contributes: a correction after a rejection, a bonus after full
    /// acceptance, or (when `alt_used`) the accepted second candidate.
    pub extra: u32,
    /// The second candidate for position 0 was accepted: the caller must feed it to the target
    /// to obtain the following bonus token.
    pub alt_used: bool,
}

/// Greedy verification: `logits[i]` is the target's raw distribution after batch position `i`
/// (`k+1` vectors for `k` drafts).
pub fn verify_greedy(logits: &[Vec<f32>], drafts: &[u32]) -> Verdict {
    let mut a = 0;
    while a < drafts.len() && a < logits.len() && argmax(&logits[a]) == drafts[a] {
        a += 1;
    }
    let extra = logits.get(a).map_or(0, |l| argmax(l));
    Verdict {
        accepted: a,
        extra,
        alt_used: false,
    }
}

/// Unbiased speculative sampling. `p[i]` are the target's PROCESSED distributions (`k+1`
/// vectors), `q[i]` the drafter's processed distributions (`None` = deterministic drafter),
/// `alt` an independent second draw from `q[0]` that is tried if the first draft is rejected.
pub fn verify_sampling(
    p: &[Vec<f32>],
    drafts: &[u32],
    q: Option<&[Vec<f32>]>,
    alt: Option<u32>,
    rng: &mut Rng,
) -> Verdict {
    let k = drafts.len().min(p.len().saturating_sub(1));
    for i in 0..k {
        let d = drafts[i];
        let pd = p[i].get(d as usize).copied().unwrap_or(0.0);
        let qi = q.and_then(|q| q.get(i)).map(Vec::as_slice);
        let qd = qi.map_or(1.0, |q| q.get(d as usize).copied().unwrap_or(0.0));
        if rng.f64() < accept_prob(pd, qd) {
            continue;
        }
        // rejected: residual, optionally with a second candidate at position 0
        let mut r = residual(&p[i], qi, d);
        if let (0, Some(a), Some(q0)) = (i, alt, qi) {
            let ra = r.get(a as usize).copied().unwrap_or(0.0);
            let qa = q0.get(a as usize).copied().unwrap_or(0.0);
            if rng.f64() < accept_prob(ra, qa) {
                return Verdict {
                    accepted: 0,
                    extra: a,
                    alt_used: true,
                };
            }
            r = residual(&r, Some(q0), a);
        }
        return Verdict {
            accepted: i,
            extra: sample_from(&r, rng),
            alt_used: false,
        };
    }
    let extra = p.get(k).map_or(0, |last| sample_from(last, rng));
    Verdict {
        accepted: k,
        extra,
        alt_used: false,
    }
}

// ------------------------------------------------------------------------------------------
// Adaptive draft length, configuration, statistics
// ------------------------------------------------------------------------------------------

/// Grows the draft length after fully accepted rounds and shrinks it when fewer than half the
/// drafts survive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveLen {
    pub min: usize,
    pub max: usize,
    pub cur: usize,
}
impl AdaptiveLen {
    pub fn new(min: usize, max: usize, start: usize) -> AdaptiveLen {
        let (min, max) = (min.min(max), max.max(min));
        AdaptiveLen {
            min,
            max,
            cur: start.clamp(min, max),
        }
    }
    pub fn update(&mut self, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        if accepted == drafted {
            self.cur = (self.cur + 1).min(self.max);
        } else if accepted * 2 < drafted {
            self.cur = self.cur.saturating_sub(1).max(self.min);
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpecConfig {
    pub max_new: usize,
    /// Generation ends right after one of these tokens is emitted (the token is included).
    pub stop: Vec<u32>,
    pub sampling: SamplingCfg,
    pub seed: u64,
    /// Draft length when `adaptive` is `None`.
    pub draft_len: usize,
    pub adaptive: Option<AdaptiveLen>,
    /// Most positions the target may hold.
    pub ctx_limit: Option<usize>,
}
impl Default for SpecConfig {
    fn default() -> Self {
        SpecConfig {
            max_new: 64,
            stop: vec![],
            sampling: SamplingCfg::default(),
            seed: 0,
            draft_len: 4,
            adaptive: None,
            ctx_limit: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpecStats {
    pub rounds: usize,
    /// Batched target passes during decoding (rounds plus any extra pass for a second candidate).
    pub passes: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub emitted: usize,
}
impl SpecStats {
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }
    pub fn tokens_per_pass(&self) -> f64 {
        if self.passes == 0 {
            0.0
        } else {
            self.emitted as f64 / self.passes as f64
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    MaxTokens,
    StopToken,
    ContextLimit,
}

#[derive(Clone, Debug)]
pub struct SpecOutput {
    pub tokens: Vec<u32>,
    pub stats: SpecStats,
    pub stop: StopReason,
}

#[derive(Debug, PartialEq, Eq)]
pub enum SpecError {
    EmptyPrompt,
    Verifier(String),
}
impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecError::EmptyPrompt => write!(f, "the prompt is empty"),
            SpecError::Verifier(s) => write!(f, "the target model misbehaved: {s}"),
        }
    }
}
impl std::error::Error for SpecError {}

fn checked_batch<V: Verifier + ?Sized>(
    t: &mut V,
    tokens: &[u32],
) -> Result<Vec<Vec<f32>>, SpecError> {
    let out = t.logits_batch(tokens);
    if out.len() != tokens.len() {
        return Err(SpecError::Verifier(format!(
            "{} logit vectors for {} tokens",
            out.len(),
            tokens.len()
        )));
    }
    if let Some(first) = out
        .first()
        .filter(|f| f.is_empty() || out.iter().any(|v| v.len() != f.len()))
    {
        let _ = first;
        return Err(SpecError::Verifier("inconsistent or empty logits".into()));
    }
    Ok(out)
}

/// Shared start: prefill everything but the last prompt token.
fn start<V: Verifier + ?Sized>(target: &mut V, prompt: &[u32]) -> Result<usize, SpecError> {
    if prompt.is_empty() {
        return Err(SpecError::EmptyPrompt);
    }
    let n = prompt.len() - 1;
    if n > 0 {
        checked_batch(target, &prompt[..n])?;
    }
    Ok(n)
}

/// Push emitted tokens, honouring stop tokens and `max_new`; returns the stop reason if hit.
fn push_tokens(
    emitted: &[u32],
    out: &mut Vec<u32>,
    ctx: &mut Vec<u32>,
    cfg: &SpecConfig,
) -> Option<StopReason> {
    for &t in emitted {
        out.push(t);
        ctx.push(t);
        if cfg.stop.contains(&t) {
            return Some(StopReason::StopToken);
        }
        if out.len() >= cfg.max_new {
            return Some(StopReason::MaxTokens);
        }
    }
    None
}

/// Baseline: ordinary one-token-at-a-time decoding with the same protocol and sampling.
pub fn generate_plain<V: Verifier + ?Sized>(
    target: &mut V,
    prompt: &[u32],
    cfg: &SpecConfig,
) -> Result<SpecOutput, SpecError> {
    let mut fed = start(target, prompt)?;
    let mut ctx = prompt.to_vec();
    let mut out = Vec::new();
    let mut rng = Rng::new(cfg.seed);
    let mut stats = SpecStats::default();
    if cfg.max_new == 0 {
        return Ok(SpecOutput {
            tokens: out,
            stats,
            stop: StopReason::MaxTokens,
        });
    }
    let reason = loop {
        if cfg.ctx_limit.is_some_and(|l| fed + 1 > l) {
            break StopReason::ContextLimit;
        }
        let pending = *ctx.last().unwrap_or(&0);
        let logits = checked_batch(target, &[pending])?;
        fed += 1;
        let t = if cfg.sampling.is_greedy() {
            argmax(&logits[0])
        } else {
            sample_from(&cfg.sampling.probs(&logits[0]), &mut rng)
        };
        stats.rounds += 1;
        stats.passes += 1;
        stats.emitted += 1;
        if let Some(r) = push_tokens(&[t], &mut out, &mut ctx, cfg) {
            break r;
        }
    };
    Ok(SpecOutput {
        tokens: out,
        stats,
        stop: reason,
    })
}

/// Speculative generation.
pub fn generate<V: Verifier + ?Sized, D: Drafter + ?Sized>(
    target: &mut V,
    drafter: &mut D,
    prompt: &[u32],
    cfg: &SpecConfig,
) -> Result<SpecOutput, SpecError> {
    let mut fed = start(target, prompt)?;
    let mut ctx = prompt.to_vec();
    let mut out: Vec<u32> = Vec::new();
    let mut rng = Rng::new(cfg.seed);
    let mut stats = SpecStats::default();
    let mut adaptive = cfg.adaptive;
    let greedy = cfg.sampling.is_greedy();
    if cfg.max_new == 0 {
        return Ok(SpecOutput {
            tokens: out,
            stats,
            stop: StopReason::MaxTokens,
        });
    }

    let reason = loop {
        let remaining = cfg.max_new - out.len();
        let mut k = adaptive
            .map_or(cfg.draft_len, |a| a.cur)
            .min(remaining.saturating_sub(1));
        if let Some(lim) = cfg.ctx_limit {
            if fed + 1 > lim {
                break StopReason::ContextLimit;
            }
            k = k.min(lim - fed - 1);
        }
        let mut draft = if k > 0 {
            drafter.draft(&ctx, k, &cfg.sampling, &mut rng)
        } else {
            Draft::empty()
        };
        draft.tokens.truncate(k);
        if let Some(d) = &mut draft.dists {
            d.truncate(draft.tokens.len());
        }
        if greedy {
            draft.alt = None;
            draft.dists = None;
        }
        if let Some(d) = draft
            .dists
            .as_ref()
            .filter(|d| d.len() < draft.tokens.len())
        {
            draft.tokens.truncate(d.len());
        }

        let pending = *ctx.last().unwrap_or(&0);
        let mut batch = Vec::with_capacity(draft.tokens.len() + 1);
        batch.push(pending);
        batch.extend_from_slice(&draft.tokens);
        let mut logits = checked_batch(target, &batch)?;
        let vocab = logits[0].len();

        // Drop drafts the target cannot even represent (hostile or buggy drafter).
        let mut valid = 0;
        for (i, &d) in draft.tokens.iter().enumerate() {
            let q_ok = draft.dists.as_ref().is_none_or(|q| q[i].len() == vocab);
            if (d as usize) < vocab && q_ok {
                valid += 1
            } else {
                break;
            }
        }
        if valid < draft.tokens.len() {
            draft.tokens.truncate(valid);
            target.rollback(fed + 1 + valid);
            logits.truncate(valid + 1);
        }
        let kk = draft.tokens.len();
        let alt = draft.alt.filter(|&a| {
            (a as usize) < vocab
                && kk > 0
                && draft.dists.as_ref().is_some_and(|q| q[0].len() == vocab)
        });

        stats.rounds += 1;
        stats.passes += 1;
        stats.drafted += kk;

        let (accepted, emitted): (usize, Vec<u32>) = if greedy {
            let v = verify_greedy(&logits, &draft.tokens);
            let mut e = draft.tokens[..v.accepted].to_vec();
            e.push(v.extra);
            if v.accepted < kk {
                target.rollback(fed + 1 + v.accepted);
            }
            fed += 1 + v.accepted;
            (v.accepted, e)
        } else {
            let p: Vec<Vec<f32>> = logits.iter().map(|l| cfg.sampling.probs(l)).collect();
            let v = verify_sampling(&p, &draft.tokens, draft.dists.as_deref(), alt, &mut rng);
            if v.alt_used {
                // The alternative first token replaced d1: re-sync the target on it and draw the bonus.
                target.rollback(fed + 1);
                let l2 = checked_batch(target, &[v.extra])?;
                stats.passes += 1;
                fed += 2;
                let bonus = sample_from(&cfg.sampling.probs(&l2[0]), &mut rng);
                (1, vec![v.extra, bonus])
            } else {
                let mut e = draft.tokens[..v.accepted].to_vec();
                e.push(v.extra);
                if v.accepted < kk {
                    target.rollback(fed + 1 + v.accepted);
                }
                fed += 1 + v.accepted;
                (v.accepted, e)
            }
        };
        stats.accepted += accepted.min(kk);
        if let Some(a) = &mut adaptive {
            a.update(
                kk,
                if emitted.len() >= 2 && accepted == 1 && kk >= 1 && emitted[0] != draft.tokens[0] {
                    0
                } else {
                    accepted.min(kk)
                },
            );
        }
        let before = out.len();
        let stop = push_tokens(&emitted, &mut out, &mut ctx, cfg);
        stats.emitted += out.len() - before;
        drafter.observe(&ctx);
        if let Some(r) = stop {
            break r;
        }
    };
    // Leave the target holding exactly ctx[..len-1], so generation can be resumed.
    let want = ctx.len() - 1;
    if fed > want {
        target.rollback(want);
    }
    drafter.observe(&ctx);
    Ok(SpecOutput {
        tokens: out,
        stats,
        stop: reason,
    })
}

// ------------------------------------------------------------------------------------------
// Drafters
// ------------------------------------------------------------------------------------------

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn hash_tokens(t: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ t.len() as u64;
    for &x in t {
        h = (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

/// A smaller model run autoregressively.
pub struct DraftModel<V: Verifier> {
    model: V,
    hist: Vec<u32>,
    two_candidates: bool,
}
impl<V: Verifier> DraftModel<V> {
    pub fn new(model: V) -> Self {
        DraftModel {
            model,
            hist: vec![],
            two_candidates: false,
        }
    }
    /// Also draw a second independent candidate for the first position (sampling only). The
    /// target pays one extra single-token pass whenever that candidate is the one accepted.
    pub fn with_two_candidates(mut self, on: bool) -> Self {
        self.two_candidates = on;
        self
    }
    pub fn into_inner(self) -> V {
        self.model
    }
    fn reset(&mut self) {
        self.model.rollback(0);
        self.hist.clear();
    }
}
impl<V: Verifier> Drafter for DraftModel<V> {
    fn name(&self) -> &'static str {
        "draft-model"
    }
    fn draft(&mut self, ctx: &[u32], k: usize, cfg: &SamplingCfg, rng: &mut Rng) -> Draft {
        if ctx.is_empty() || k == 0 {
            return Draft::empty();
        }
        let c = common_prefix(&self.hist, ctx).min(ctx.len() - 1);
        if c < self.hist.len() {
            self.model.rollback(c);
            self.hist.truncate(c);
        }
        let first = self.model.logits_batch(&ctx[c..]);
        if first.len() != ctx.len() - c {
            self.reset();
            return Draft::empty();
        }
        self.hist.extend_from_slice(&ctx[c..]);
        let mut last = match first.into_iter().next_back() {
            Some(l) if !l.is_empty() => l,
            _ => {
                self.reset();
                return Draft::empty();
            }
        };
        let greedy = cfg.is_greedy();
        let mut d = Draft::empty();
        let mut dists = Vec::new();
        for i in 0..k {
            let t = if greedy {
                argmax(&last)
            } else {
                let q = cfg.probs(&last);
                let t = sample_from(&q, rng);
                if i == 0 && self.two_candidates {
                    d.alt = Some(sample_from(&q, rng));
                }
                dists.push(q);
                t
            };
            d.tokens.push(t);
            if i + 1 < k {
                let o = self.model.logits_batch(&[t]);
                self.hist.push(t);
                match o.into_iter().next() {
                    Some(l) if !l.is_empty() => last = l,
                    _ => {
                        self.reset();
                        d.tokens.truncate(i + 1);
                        dists.truncate(i + 1);
                        break;
                    }
                }
            }
        }
        if !greedy {
            d.dists = Some(dists);
        }
        d
    }
}

/// Prompt-lookup decoding ("ngram-simple"): the most recent earlier occurrence of the last
/// `n` tokens (longest `n` first) supplies the proposal.
pub struct NgramLookup {
    n_min: usize,
    n_max: usize,
    idx: Vec<HashMap<u64, usize>>,
    hist: Vec<u32>,
    /// n-grams ending at positions `<= indexed` are in the index.
    indexed: usize,
}
impl NgramLookup {
    pub fn new(n_min: usize, n_max: usize) -> Self {
        let n_min = n_min.max(1);
        let n_max = n_max.max(n_min);
        NgramLookup {
            n_min,
            n_max,
            idx: (n_min..=n_max).map(|_| HashMap::new()).collect(),
            hist: vec![],
            indexed: 0,
        }
    }
    fn sync(&mut self, ctx: &[u32]) {
        if ctx.len() < self.hist.len() || ctx[..self.hist.len()] != self.hist[..] {
            self.idx.iter_mut().for_each(HashMap::clear);
            self.hist.clear();
            self.indexed = 0;
        }
        self.hist.extend_from_slice(&ctx[self.hist.len()..]);
        let upto = ctx.len().saturating_sub(1); // n-grams must end before the last token
        for e in (self.indexed + 1)..=upto {
            for (j, n) in (self.n_min..=self.n_max).enumerate() {
                if e >= n {
                    self.idx[j].insert(hash_tokens(&ctx[e - n..e]), e - n);
                }
            }
        }
        self.indexed = self.indexed.max(upto);
    }
    /// Longest-n match for the current suffix: (start, n).
    pub fn find(&mut self, ctx: &[u32]) -> Option<(usize, usize)> {
        self.sync(ctx);
        let l = ctx.len();
        for (j, n) in (self.n_min..=self.n_max)
            .enumerate()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            if l <= n {
                continue;
            }
            let suffix = &ctx[l - n..];
            if let Some(&s) = self.idx[j]
                .get(&hash_tokens(suffix))
                .filter(|&&s| s + n < l && ctx[s..s + n] == *suffix)
            {
                return Some((s, n));
            }
        }
        None
    }
}
impl Drafter for NgramLookup {
    fn name(&self) -> &'static str {
        "ngram-lookup"
    }
    fn draft(&mut self, ctx: &[u32], k: usize, _cfg: &SamplingCfg, _rng: &mut Rng) -> Draft {
        match self.find(ctx) {
            Some((s, n)) => Draft::deterministic(ctx[s + n..(s + n + k).min(ctx.len())].to_vec()),
            None => Draft::empty(),
        }
    }
}

/// N-gram statistics ("ngram-map"): follow the most frequent continuation, token by token, for
/// as long as it was seen at least `min_count` times. Forgets everything when it holds more than
/// `max_entries` contexts.
pub struct NgramMap {
    n: usize,
    min_count: u32,
    max_entries: usize,
    map: HashMap<Vec<u32>, Vec<(u32, u32, usize)>>,
    hist: Vec<u32>,
    learned: usize,
}
impl NgramMap {
    pub fn new(n: usize, min_count: u32, max_entries: usize) -> Self {
        NgramMap {
            n: n.max(1),
            min_count: min_count.max(1),
            max_entries: max_entries.max(1),
            map: HashMap::new(),
            hist: vec![],
            learned: 0,
        }
    }
    fn learn(&mut self, ctx: &[u32]) {
        if ctx.len() < self.hist.len() || ctx[..self.hist.len()] != self.hist[..] {
            self.map.clear();
            self.hist.clear();
            self.learned = 0;
        }
        self.hist.extend_from_slice(&ctx[self.hist.len()..]);
        // key ctx[i..i+n] followed by ctx[i+n]; the final token has no follower yet.
        let l = ctx.len();
        while self.learned + self.n < l {
            let i = self.learned;
            let key = ctx[i..i + self.n].to_vec();
            let next = ctx[i + self.n];
            if self.map.len() >= self.max_entries && !self.map.contains_key(&key) {
                self.map.clear();
            }
            let e = self.map.entry(key).or_default();
            match e.iter_mut().find(|x| x.0 == next) {
                Some(x) => {
                    x.1 += 1;
                    x.2 = i;
                }
                None => {
                    if let (true, Some(w)) = (
                        e.len() >= 8,
                        e.iter()
                            .enumerate()
                            .min_by_key(|(_, x)| (x.1, x.2))
                            .map(|(j, _)| j),
                    ) {
                        e.swap_remove(w);
                    }
                    e.push((next, 1, i));
                }
            }
            self.learned += 1;
        }
    }
    pub fn contexts(&self) -> usize {
        self.map.len()
    }
}
impl Drafter for NgramMap {
    fn name(&self) -> &'static str {
        "ngram-map"
    }
    fn draft(&mut self, ctx: &[u32], k: usize, _cfg: &SamplingCfg, _rng: &mut Rng) -> Draft {
        self.learn(ctx);
        let mut work: Vec<u32> = ctx[ctx.len().saturating_sub(self.n)..].to_vec();
        let mut out = Vec::new();
        while out.len() < k && work.len() >= self.n {
            let key = &work[work.len() - self.n..];
            let Some(best) = self
                .map
                .get(key)
                .and_then(|e| e.iter().max_by_key(|x| (x.1, x.2)))
            else {
                break;
            };
            if best.1 < self.min_count {
                break;
            }
            out.push(best.0);
            work.push(best.0);
        }
        Draft::deterministic(out)
    }
}

/// Bounded map from n-grams to what followed them, filled from finished or running
/// generations. FIFO eviction.
#[derive(Clone, Debug)]
pub struct SuffixCache {
    n_min: usize,
    n_max: usize,
    cont_len: usize,
    capacity: usize,
    map: HashMap<Vec<u32>, Vec<u32>>,
    order: VecDeque<Vec<u32>>,
}
impl SuffixCache {
    pub fn new(n_min: usize, n_max: usize, cont_len: usize, capacity: usize) -> Self {
        let n_min = n_min.max(1);
        SuffixCache {
            n_min,
            n_max: n_max.max(n_min),
            cont_len: cont_len.max(1),
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    /// Learn every n-gram of `tokens` that has at least one follower, starting at start index
    /// `from` (for incremental learning).
    pub fn learn_from(&mut self, tokens: &[u32], from: usize) {
        for n in self.n_min..=self.n_max {
            let mut i = from;
            while i + n < tokens.len() {
                let key = &tokens[i..i + n];
                let next = tokens[i + n..(i + n + self.cont_len).min(tokens.len())].to_vec();
                if let Some(v) = self.map.get_mut(key) {
                    *v = next;
                } else {
                    if self.map.len() >= self.capacity
                        && let Some(old) = self.order.pop_front()
                    {
                        self.map.remove(&old);
                    }
                    self.order.push_back(key.to_vec());
                    self.map.insert(key.to_vec(), next);
                }
                i += 1;
            }
        }
    }
    pub fn learn(&mut self, tokens: &[u32]) {
        self.learn_from(tokens, 0);
    }
    /// The continuation stored for the longest matching suffix of `ctx`.
    pub fn lookup(&self, ctx: &[u32]) -> Option<&[u32]> {
        (self.n_min..=self.n_max)
            .rev()
            .filter(|&n| ctx.len() >= n)
            .find_map(|n| self.map.get(&ctx[ctx.len() - n..]).map(Vec::as_slice))
    }
}

/// Drafts from a [`SuffixCache`], learning from the running context as it goes.
pub struct CacheDrafter {
    cache: SuffixCache,
    seen: Vec<u32>,
    learned: usize,
}
impl CacheDrafter {
    pub fn new(cache: SuffixCache) -> Self {
        CacheDrafter {
            cache,
            seen: vec![],
            learned: 0,
        }
    }
    pub fn into_cache(self) -> SuffixCache {
        self.cache
    }
    pub fn cache(&self) -> &SuffixCache {
        &self.cache
    }
}
impl Drafter for CacheDrafter {
    fn name(&self) -> &'static str {
        "suffix-cache"
    }
    fn observe(&mut self, ctx: &[u32]) {
        if ctx.len() < self.seen.len() || ctx[..self.seen.len()] != self.seen[..] {
            self.seen.clear();
            self.learned = 0;
        }
        self.seen.extend_from_slice(&ctx[self.seen.len()..]);
        let from = self.learned.saturating_sub(self.cache.n_max);
        self.cache.learn_from(ctx, from);
        self.learned = ctx.len();
    }
    fn draft(&mut self, ctx: &[u32], k: usize, _cfg: &SamplingCfg, _rng: &mut Rng) -> Draft {
        let mut work = ctx.to_vec();
        let mut out: Vec<u32> = Vec::new();
        while out.len() < k {
            let Some(c) = self.cache.lookup(&work) else {
                break;
            };
            if c.is_empty() {
                break;
            }
            let take = c.len().min(k - out.len());
            out.extend_from_slice(&c[..take]);
            work.extend_from_slice(&c[..take]);
        }
        Draft::deterministic(out)
    }
}

/// Tries each drafter in order; the first non-empty proposal wins. All of them observe.
pub struct Chain {
    drafters: Vec<Box<dyn Drafter>>,
}
impl Chain {
    pub fn new(drafters: Vec<Box<dyn Drafter>>) -> Self {
        Chain { drafters }
    }
}
impl Drafter for Chain {
    fn name(&self) -> &'static str {
        "chain"
    }
    fn draft(&mut self, ctx: &[u32], k: usize, cfg: &SamplingCfg, rng: &mut Rng) -> Draft {
        for d in &mut self.drafters {
            let r = d.draft(ctx, k, cfg, rng);
            if !r.tokens.is_empty() {
                return r;
            }
        }
        Draft::empty()
    }
    fn observe(&mut self, ctx: &[u32]) {
        for d in &mut self.drafters {
            d.observe(ctx);
        }
    }
}

// ------------------------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- a mock language model defined by explicit (hash-derived) probability tables ----------

    fn mix64(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn base(seed: u64, p2: u32, p1: u32, vocab: usize, scale: f32) -> Vec<f32> {
        (0..vocab)
            .map(|j| {
                let h = mix64(seed ^ mix64(p2 as u64 ^ mix64(p1 as u64 ^ mix64(j as u64 + 1))));
                ((h >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
            })
            .collect()
    }

    /// Order-2 model: logits depend on the last two tokens. `mix` blends the target tables
    /// (`seed_a`) with unrelated tables (`seed_b`): 1.0 = identical to the target.
    #[derive(Clone)]
    struct Mock {
        vocab: usize,
        seed_a: u64,
        seed_b: u64,
        mix: f32,
        scale: f32,
        fed: Vec<u32>,
        passes: usize,
        rollbacks: usize,
        fed_total: usize,
    }
    impl Mock {
        fn target(vocab: usize, seed: u64) -> Mock {
            Mock {
                vocab,
                seed_a: seed,
                seed_b: 0,
                mix: 1.0,
                scale: 2.0,
                fed: vec![],
                passes: 0,
                rollbacks: 0,
                fed_total: 0,
            }
        }
        fn draft(vocab: usize, seed: u64, mix: f32) -> Mock {
            Mock {
                mix,
                seed_b: seed ^ 0xABCDEF,
                ..Mock::target(vocab, seed)
            }
        }
        fn logits_for(&self, p2: u32, p1: u32) -> Vec<f32> {
            let a = base(self.seed_a, p2, p1, self.vocab, self.scale);
            if self.mix >= 1.0 {
                return a;
            }
            let b = base(self.seed_b, p2, p1, self.vocab, self.scale);
            a.iter()
                .zip(&b)
                .map(|(x, y)| self.mix * x + (1.0 - self.mix) * y)
                .collect()
        }
    }
    impl Verifier for Mock {
        fn logits_batch(&mut self, tokens: &[u32]) -> Vec<Vec<f32>> {
            self.passes += 1;
            self.fed_total += tokens.len();
            tokens
                .iter()
                .map(|&t| {
                    self.fed.push(t);
                    let n = self.fed.len();
                    let p2 = if n >= 2 { self.fed[n - 2] } else { u32::MAX };
                    self.logits_for(p2, t)
                })
                .collect()
        }
        fn rollback(&mut self, keep: usize) {
            assert!(
                keep <= self.fed.len(),
                "rollback({keep}) beyond the {} fed positions",
                self.fed.len()
            );
            self.rollbacks += 1;
            self.fed.truncate(keep);
        }
    }

    fn greedy_cfg(max_new: usize, k: usize) -> SpecConfig {
        SpecConfig {
            max_new,
            draft_len: k,
            ..Default::default()
        }
    }

    fn check_state(t: &Mock, prompt: &[u32], o: &SpecOutput) {
        let mut ctx = prompt.to_vec();
        ctx.extend(&o.tokens);
        assert_eq!(
            t.fed,
            ctx[..ctx.len() - 1],
            "target state must be exactly ctx without the pending token"
        );
    }

    // ---- greedy equality ----------------------------------------------------------------------

    /// A drafter that proposes tokens the target would NOT choose.
    struct AlwaysWrong {
        oracle: Mock,
        hist: Vec<u32>,
    }
    impl Drafter for AlwaysWrong {
        fn draft(&mut self, ctx: &[u32], k: usize, _c: &SamplingCfg, _r: &mut Rng) -> Draft {
            // walk the oracle along the context (fresh each call; tests are tiny)
            let mut m = self.oracle.clone();
            m.fed.clear();
            let mut last = m.logits_batch(ctx).pop().unwrap();
            let mut toks = vec![];
            for _ in 0..k {
                let right = argmax(&last);
                let wrong = (right + 1) % m.vocab as u32;
                toks.push(wrong);
                last = m.logits_batch(&[wrong]).pop().unwrap();
            }
            self.hist = toks.clone();
            Draft::deterministic(toks)
        }
    }
    struct RandomDrafter {
        vocab: u32,
    }
    impl Drafter for RandomDrafter {
        fn draft(&mut self, _c: &[u32], k: usize, _s: &SamplingCfg, rng: &mut Rng) -> Draft {
            let n = (rng.next_u64() % (k as u64 + 1)) as usize;
            Draft::deterministic(
                (0..n)
                    .map(|_| (rng.next_u64() % self.vocab as u64) as u32)
                    .collect(),
            )
        }
    }

    #[test]
    fn greedy_speculation_equals_plain_greedy_for_every_drafter_and_length() {
        let vocab = 8usize;
        for seed in 0..12u64 {
            let prompt = [1u32, 5, 2, 5, 2, 5];
            for stop in [vec![], vec![3u32, 7]] {
                for k in 0..=8usize {
                    let cfg = SpecConfig {
                        max_new: 40,
                        draft_len: k,
                        stop: stop.clone(),
                        seed,
                        ..Default::default()
                    };
                    let mut plain_t = Mock::target(vocab, seed);
                    let plain = generate_plain(&mut plain_t, &prompt, &cfg).unwrap();
                    check_state(&plain_t, &prompt, &plain);

                    let drafters: Vec<(&str, Box<dyn Drafter>)> = vec![
                        (
                            "model0",
                            Box::new(DraftModel::new(Mock::draft(vocab, seed, 0.0))),
                        ),
                        (
                            "model.6",
                            Box::new(DraftModel::new(Mock::draft(vocab, seed, 0.6))),
                        ),
                        (
                            "oracle",
                            Box::new(DraftModel::new(Mock::target(vocab, seed))),
                        ),
                        (
                            "wrong",
                            Box::new(AlwaysWrong {
                                oracle: Mock::target(vocab, seed),
                                hist: vec![],
                            }),
                        ),
                        (
                            "random",
                            Box::new(RandomDrafter {
                                vocab: vocab as u32 + 3,
                            }),
                        ), // some out-of-vocab tokens too
                        ("lookup", Box::new(NgramLookup::new(1, 3))),
                        ("map", Box::new(NgramMap::new(2, 1, 1000))),
                        (
                            "cache",
                            Box::new(CacheDrafter::new(SuffixCache::new(1, 3, 4, 256))),
                        ),
                        (
                            "chain",
                            Box::new(Chain::new(vec![
                                Box::new(NgramLookup::new(2, 4)),
                                Box::new(DraftModel::new(Mock::draft(vocab, seed, 0.5))),
                            ])),
                        ),
                    ];
                    for (name, mut d) in drafters {
                        let mut t = Mock::target(vocab, seed);
                        let o = generate(&mut t, d.as_mut(), &prompt, &cfg).unwrap();
                        assert_eq!(
                            o.tokens, plain.tokens,
                            "drafter {name} k {k} seed {seed} stop {stop:?}"
                        );
                        assert_eq!(o.stop, plain.stop, "{name}");
                        check_state(&t, &prompt, &o);
                    }
                }
            }
        }
    }

    #[test]
    fn oracle_is_always_accepted_and_wrong_drafts_never_are() {
        let vocab = 8;
        let prompt = [1u32, 2, 3];
        let cfg = greedy_cfg(30, 4);
        let mut t = Mock::target(vocab, 7);
        let mut d = DraftModel::new(Mock::target(vocab, 7));
        let o = generate(&mut t, &mut d, &prompt, &cfg).unwrap();
        assert_eq!(o.stats.accepted, o.stats.drafted);
        assert_eq!(o.stats.rounds, 30usize.div_ceil(5));
        assert!(o.stats.tokens_per_pass() > 4.0);

        let mut t = Mock::target(vocab, 7);
        let mut d = AlwaysWrong {
            oracle: Mock::target(vocab, 7),
            hist: vec![],
        };
        let o = generate(&mut t, &mut d, &prompt, &cfg).unwrap();
        assert_eq!(o.stats.accepted, 0);
        assert_eq!(o.stats.rounds, 30);
        assert_eq!(o.stats.tokens_per_pass(), 1.0);
    }

    #[test]
    fn rollback_accounting_matches_the_protocol() {
        let vocab = 8;
        let prompt = [1u32, 2, 3, 4];
        let cfg = greedy_cfg(25, 5);
        let mut t = Mock::target(vocab, 3);
        let mut d = RandomDrafter {
            vocab: vocab as u32,
        };
        let o = generate(&mut t, &mut d, &prompt, &cfg).unwrap();
        // every position fed was either kept or rolled back: total fed = prefill + batches
        assert_eq!(t.passes, 1 + o.stats.passes);
        assert!(t.fed_total >= prompt.len() - 1 + o.stats.rounds + o.stats.drafted);
        check_state(&t, &prompt, &o);
        // the drafter's own state never outgrows the context either
        let mut dm = DraftModel::new(Mock::draft(vocab, 3, 0.5));
        let mut t = Mock::target(vocab, 3);
        let o = generate(&mut t, &mut dm, &prompt, &cfg).unwrap();
        let inner = dm.into_inner();
        assert!(inner.fed.len() <= prompt.len() + o.tokens.len() + cfg.draft_len);
        assert!(
            inner
                .fed
                .iter()
                .zip(prompt.iter().chain(&o.tokens))
                .take(prompt.len() + 1)
                .all(|(a, b)| a == b)
        );
    }

    #[test]
    fn stop_tokens_max_new_and_context_limit() {
        let vocab = 6;
        let prompt = [0u32, 1];
        // stop right after the first stop token, even in the middle of an accepted run
        let cfg = SpecConfig {
            max_new: 50,
            draft_len: 6,
            stop: vec![2, 4],
            ..Default::default()
        };
        let mut t = Mock::target(vocab, 11);
        let plain = generate_plain(&mut Mock::target(vocab, 11), &prompt, &cfg).unwrap();
        let o = generate(
            &mut t,
            &mut DraftModel::new(Mock::target(vocab, 11)),
            &prompt,
            &cfg,
        )
        .unwrap();
        assert_eq!(o.tokens, plain.tokens);
        assert_eq!(o.stop, plain.stop);
        if o.stop == StopReason::StopToken {
            assert!(cfg.stop.contains(o.tokens.last().unwrap()));
            assert!(
                o.tokens[..o.tokens.len() - 1]
                    .iter()
                    .all(|x| !cfg.stop.contains(x))
            );
        }
        check_state(&t, &prompt, &o);
        // max_new = 0 and 1
        for n in [0usize, 1, 2, 3] {
            let cfg = greedy_cfg(n, 8);
            let o = generate(
                &mut Mock::target(vocab, 1),
                &mut DraftModel::new(Mock::target(vocab, 1)),
                &prompt,
                &cfg,
            )
            .unwrap();
            assert_eq!(o.tokens.len(), n);
        }
        // context limit: same output and stop reason as plain decoding
        for lim in 2..14usize {
            let cfg = SpecConfig {
                max_new: 100,
                draft_len: 5,
                ctx_limit: Some(lim),
                ..Default::default()
            };
            let plain = generate_plain(&mut Mock::target(vocab, 5), &prompt, &cfg).unwrap();
            let mut t = Mock::target(vocab, 5);
            let o = generate(
                &mut t,
                &mut DraftModel::new(Mock::draft(vocab, 5, 0.7)),
                &prompt,
                &cfg,
            )
            .unwrap();
            assert_eq!(o.tokens, plain.tokens, "limit {lim}");
            assert_eq!(o.stop, StopReason::ContextLimit);
            assert!(t.fed.len() <= lim);
        }
    }

    // ---- exactness of speculative sampling -------------------------------------------------------

    fn chi2(counts: &[f64], expected: &[f64]) -> f64 {
        let mut s = 0.0;
        for (&o, &e) in counts.iter().zip(expected) {
            if e <= 0.0 {
                assert_eq!(o, 0.0, "impossible outcome was produced");
            } else {
                s += (o - e) * (o - e) / e;
            }
        }
        s
    }

    /// Joint distribution of the first two generated tokens, speculative vs exact.
    fn joint_test(
        make: &dyn Fn(u64) -> Box<dyn Drafter>,
        cfg_s: SamplingCfg,
        runs: usize,
        k: usize,
        label: &str,
    ) {
        let vocab = 4usize;
        let prompt = [2u32, 0, 1, 2, 0];
        let tcfg = Mock::target(vocab, 99);
        let p1 = cfg_s.probs(&tcfg.logits_for(2, 0));
        let mut expected = vec![0.0f64; vocab * vocab];
        for t1 in 0..vocab {
            let p2 = cfg_s.probs(&tcfg.logits_for(0, t1 as u32));
            for t2 in 0..vocab {
                expected[t1 * vocab + t2] = runs as f64 * p1[t1] as f64 * p2[t2] as f64;
            }
        }
        let mut counts = vec![0.0f64; vocab * vocab];
        let mut acc = 0usize;
        let mut drafted = 0usize;
        for i in 0..runs {
            let cfg = SpecConfig {
                max_new: 2,
                draft_len: k,
                sampling: cfg_s,
                seed: 1000 + i as u64,
                ..Default::default()
            };
            let mut t = Mock::target(vocab, 99);
            let mut d = make(i as u64);
            let o = generate(&mut t, d.as_mut(), &prompt, &cfg).unwrap();
            assert_eq!(o.tokens.len(), 2);
            counts[o.tokens[0] as usize * vocab + o.tokens[1] as usize] += 1.0;
            acc += o.stats.accepted;
            drafted += o.stats.drafted;
        }
        let stat = chi2(&counts, &expected);
        // df <= 15: the 99.99% quantile is ~ 39; a biased rule at this sample size scores in the hundreds.
        assert!(
            stat < 45.0,
            "{label}: chi2 = {stat:.1} (accepted {acc}/{drafted})"
        );
        assert!(drafted > 0, "{label}: nothing was drafted");
    }

    #[test]
    fn speculative_sampling_preserves_the_target_distribution() {
        let runs = 30_000;
        for (name, cfg_s) in [
            ("t1", SamplingCfg::sampled(1.0, 0, 1.0)),
            ("t.7 top_p.9", SamplingCfg::sampled(0.7, 0, 0.9)),
            ("t1.3 top_k3", SamplingCfg::sampled(1.3, 3, 1.0)),
        ] {
            // a related-but-different draft model (partial acceptance)
            joint_test(
                &|i| {
                    Box::new(DraftModel::new(Mock::draft(
                        4,
                        99,
                        0.5 + (i % 2) as f32 * 0.1,
                    )))
                },
                cfg_s,
                runs,
                3,
                &format!("model {name}"),
            );
            // an unrelated draft model (mostly rejections)
            joint_test(
                &|_| Box::new(DraftModel::new(Mock::draft(4, 99, 0.0))),
                cfg_s,
                runs,
                2,
                &format!("unrelated {name}"),
            );
            // a perfect draft model
            joint_test(
                &|_| Box::new(DraftModel::new(Mock::target(4, 99))),
                cfg_s,
                runs,
                3,
                &format!("oracle {name}"),
            );
            // deterministic drafters (one-hot q)
            joint_test(
                &|_| Box::new(NgramLookup::new(1, 2)),
                cfg_s,
                runs,
                3,
                &format!("lookup {name}"),
            );
            joint_test(
                &|_| {
                    Box::new({
                        let mut c = SuffixCache::new(1, 2, 3, 64);
                        c.learn(&[2, 0, 1, 2, 0, 1, 2, 0]);
                        CacheDrafter::new(c)
                    })
                },
                cfg_s,
                runs,
                3,
                &format!("cache {name}"),
            );
        }
    }

    #[test]
    fn two_candidates_keep_the_distribution_exact() {
        for cfg_s in [
            SamplingCfg::sampled(1.0, 0, 1.0),
            SamplingCfg::sampled(0.8, 3, 1.0),
        ] {
            joint_test(
                &|_| Box::new(DraftModel::new(Mock::draft(4, 99, 0.3)).with_two_candidates(true)),
                cfg_s,
                30_000,
                3,
                "two candidates",
            );
        }
        // the extra pass is actually taken sometimes
        let prompt = [1u32, 2, 0];
        let mut extra = 0;
        for i in 0..2000 {
            let cfg = SpecConfig {
                max_new: 2,
                draft_len: 2,
                sampling: SamplingCfg::sampled(1.0, 0, 1.0),
                seed: i,
                ..Default::default()
            };
            let o = generate(
                &mut Mock::target(4, 99),
                &mut DraftModel::new(Mock::draft(4, 99, 0.2)).with_two_candidates(true),
                &prompt,
                &cfg,
            )
            .unwrap();
            if o.stats.passes > o.stats.rounds {
                extra += 1;
            }
        }
        assert!(extra > 50, "second candidate never used ({extra})");
    }

    #[test]
    fn single_step_rule_matches_the_target_for_explicit_tables() {
        let p = vec![0.5f32, 0.3, 0.2];
        let q = vec![0.1f32, 0.1, 0.8];
        let bonus = vec![0.25f32; 3];
        for with_alt in [false, true] {
            let mut rng = Rng::new(5);
            let n = 200_000;
            let mut counts = [0.0f64; 3];
            for _ in 0..n {
                let c1 = sample_from(&q, &mut rng);
                let c2 = sample_from(&q, &mut rng);
                let v = verify_sampling(
                    &[p.clone(), bonus.clone()],
                    &[c1],
                    Some(std::slice::from_ref(&q)),
                    with_alt.then_some(c2),
                    &mut rng,
                );
                let first = if v.accepted == 1 { c1 } else { v.extra };
                counts[first as usize] += 1.0;
            }
            let e: Vec<f64> = p.iter().map(|&x| x as f64 * n as f64).collect();
            let s = chi2(&counts, &e);
            assert!(s < 20.0, "alt {with_alt}: chi2 {s:.1}");
        }
    }

    #[test]
    fn sampling_without_a_draft_is_plain_sampling() {
        // k = 0 rounds just sample from the target
        let prompt = [1u32, 2, 0];
        let cfg_s = SamplingCfg::sampled(1.0, 0, 1.0);
        let t0 = Mock::target(4, 99);
        let p1 = cfg_s.probs(&t0.logits_for(2, 0));
        let mut counts = [0.0f64; 4];
        let runs = 20_000;
        for i in 0..runs {
            let cfg = SpecConfig {
                max_new: 1,
                draft_len: 0,
                sampling: cfg_s,
                seed: i as u64,
                ..Default::default()
            };
            let o = generate(
                &mut Mock::target(4, 99),
                &mut NgramLookup::new(1, 2),
                &prompt,
                &cfg,
            )
            .unwrap();
            counts[o.tokens[0] as usize] += 1.0;
        }
        let e: Vec<f64> = p1.iter().map(|&x| x as f64 * runs as f64).collect();
        assert!(chi2(&counts, &e) < 20.0);
    }

    // ---- processing of logits ---------------------------------------------------------------------

    #[test]
    fn probs_handles_temperature_top_k_top_p_and_garbage() {
        let l = [1.0f32, 3.0, 2.0, -1.0];
        let g = SamplingCfg::greedy().probs(&l);
        assert_eq!(g, vec![0.0, 1.0, 0.0, 0.0]);
        let p = SamplingCfg::sampled(1.0, 0, 1.0).probs(&l);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5 && p[1] > p[2] && p[2] > p[0]);
        let k2 = SamplingCfg::sampled(1.0, 2, 1.0).probs(&l);
        assert_eq!((k2[0], k2[3]), (0.0, 0.0));
        assert!((k2.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let tp = SamplingCfg::sampled(1.0, 0, 0.5).probs(&l);
        assert_eq!(tp.iter().filter(|&&x| x > 0.0).count(), 1);
        // garbage
        for bad in [
            vec![f32::NAN; 3],
            vec![f32::INFINITY, 1.0, f32::NEG_INFINITY],
            vec![f32::NEG_INFINITY; 3],
            vec![1e38, -1e38, 0.0],
        ] {
            for t in [0.0f32, 0.5, 1.0, f32::INFINITY, f32::NAN, -3.0] {
                let p = SamplingCfg::sampled(t, 2, 0.9).probs(&bad);
                assert_eq!(p.len(), 3);
                assert!(
                    p.iter().all(|x| x.is_finite() && *x >= 0.0),
                    "{bad:?} t={t}: {p:?}"
                );
                assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-3);
            }
        }
        assert!(SamplingCfg::sampled(1.0, 0, 1.0).probs(&[]).is_empty());
        assert_eq!(argmax(&[]), 0);
        assert_eq!(argmax(&[f32::NAN, 1.0, 1.0]), 1);
    }

    // ---- n-gram and cache drafters -------------------------------------------------------------------

    fn dr<D: Drafter>(d: &mut D, ctx: &[u32], k: usize) -> Vec<u32> {
        d.draft(ctx, k, &SamplingCfg::greedy(), &mut Rng::new(0))
            .tokens
    }

    #[test]
    fn lookup_matching_edge_cases() {
        let mut d = NgramLookup::new(1, 3);
        assert!(dr(&mut d, &[], 4).is_empty());
        assert!(dr(&mut d, &[7], 4).is_empty());
        assert!(dr(&mut d, &[1, 2, 3], 4).is_empty()); // no repetition at all
        // simple repeat: "5 6 7 ... 5 6" -> proposes "7 ..."
        let ctx = [5, 6, 7, 8, 9, 5, 6];
        assert_eq!(dr(&mut NgramLookup::new(1, 3), &ctx, 3), vec![7, 8, 9]);
        assert_eq!(dr(&mut NgramLookup::new(1, 3), &ctx, 1), vec![7]);
        // proposals never run past the end of the context (overlap with the suffix itself)
        assert_eq!(dr(&mut NgramLookup::new(1, 2), &[4, 4, 4, 4], 8), vec![4]);
        // the most recent earlier occurrence wins
        assert_eq!(
            dr(&mut NgramLookup::new(1, 1), &[3, 1, 3, 2, 3], 1),
            vec![2]
        );
        // the longest n wins over a more recent shorter match
        let ctx = [1, 2, 9, 7, 2, 8, 1, 2];
        assert_eq!(dr(&mut NgramLookup::new(1, 2), &ctx, 1), vec![9]);
        assert_eq!(dr(&mut NgramLookup::new(1, 1), &ctx, 1), vec![8]);
        // match at the very beginning of the context
        assert_eq!(
            dr(&mut NgramLookup::new(2, 2), &[8, 9, 1, 2, 8, 9], 2),
            vec![1, 2]
        );
        // context shorter than n
        assert!(dr(&mut NgramLookup::new(3, 3), &[1, 1], 3).is_empty());
    }

    #[test]
    fn lookup_is_incremental_and_survives_a_context_that_changes() {
        let mut d = NgramLookup::new(1, 2);
        let mut ctx = vec![1, 2, 3, 1, 2];
        assert_eq!(dr(&mut d, &ctx, 2), vec![3, 1]);
        ctx.extend([3, 4, 1, 2]);
        assert_eq!(dr(&mut d, &ctx, 3), vec![3, 4, 1]); // most recent "1 2" is followed by 3 4 1 2 -> 3 4 1
        // a different, shorter context: the index is rebuilt
        assert_eq!(dr(&mut d, &[9, 8, 9], 2), vec![8, 9]);
        assert!(dr(&mut d, &[5, 6, 7], 2).is_empty());
    }

    #[test]
    fn ngram_map_follows_frequencies_and_respects_min_count() {
        let mut d = NgramMap::new(1, 2, 100);
        // after 1 comes 2 three times and 9 once: propose 2; 2 is followed by 3 each time
        let ctx = [1, 2, 3, 1, 2, 3, 1, 9, 1, 2, 3, 1];
        assert_eq!(dr(&mut d, &ctx, 3), vec![2, 3, 1]);
        // min_count 4 is never reached
        assert!(dr(&mut NgramMap::new(1, 4, 100), &ctx, 3).is_empty());
        // unseen context
        assert!(dr(&mut NgramMap::new(1, 1, 100), &[1, 2, 3], 3).is_empty());
        // bounded: never more entries than allowed
        let mut small = NgramMap::new(1, 1, 3);
        let long: Vec<u32> = (0..200).map(|i| (i * 7 % 31) as u32).collect();
        dr(&mut small, &long, 2);
        assert!(small.contexts() <= 3);
        // a non-extending context resets the statistics
        let mut d = NgramMap::new(1, 1, 100);
        dr(&mut d, &[1, 2, 1, 2, 1], 1);
        assert!(dr(&mut d, &[5, 5, 5], 1) == vec![5]);
    }

    #[test]
    fn suffix_cache_learns_evicts_and_persists() {
        let mut c = SuffixCache::new(2, 3, 4, 1000);
        c.learn(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(c.lookup(&[9, 9, 2, 3]), Some(&[4, 5, 6][..]));
        assert_eq!(c.lookup(&[1, 2, 3]), Some(&[4, 5, 6][..])); // longest suffix first
        assert!(c.lookup(&[7, 7]).is_none());
        assert!(c.lookup(&[3]).is_none()); // shorter than n_min
        // the last token has no follower
        assert!(c.lookup(&[5, 6]).is_none());
        // bounded with FIFO eviction
        let mut small = SuffixCache::new(1, 1, 2, 3);
        small.learn(&[10, 11, 12, 13, 14, 15]);
        assert_eq!(small.len(), 3);
        assert!(small.lookup(&[10]).is_none() && small.lookup(&[14]).is_some());
        // continues across generations: learn in one request, draft in the next
        let mut cd = CacheDrafter::new(SuffixCache::new(1, 2, 4, 100));
        cd.observe(&[1, 2, 3, 4, 5, 6, 7]);
        let cache = cd.into_cache();
        let mut cd2 = CacheDrafter::new(cache);
        assert_eq!(dr(&mut cd2, &[0, 3, 4], 3), vec![5, 6, 7]);
        assert_eq!(dr(&mut cd2, &[0, 3, 4], 9), vec![5, 6, 7]);
    }

    // ---- adaptive length ------------------------------------------------------------------------------

    #[test]
    fn adaptive_length_grows_on_success_and_shrinks_on_failure() {
        let mut a = AdaptiveLen::new(1, 6, 3);
        for _ in 0..10 {
            a.update(a.cur, a.cur);
        }
        assert_eq!(a.cur, 6);
        for _ in 0..10 {
            a.update(a.cur, 0);
        }
        assert_eq!(a.cur, 1);
        a.update(0, 0);
        assert_eq!(a.cur, 1);
        let mut b = AdaptiveLen::new(2, 8, 4);
        b.update(4, 2); // exactly half: unchanged
        assert_eq!(b.cur, 4);
        assert_eq!(AdaptiveLen::new(5, 2, 100).cur, 5);

        // inside the engine: an oracle drafter drives the length to the maximum, a wrong one to the minimum
        let prompt = [1u32, 2, 3];
        let cfg = SpecConfig {
            max_new: 60,
            adaptive: Some(AdaptiveLen::new(1, 6, 2)),
            ..Default::default()
        };
        let o = generate(
            &mut Mock::target(8, 2),
            &mut DraftModel::new(Mock::target(8, 2)),
            &prompt,
            &cfg,
        )
        .unwrap();
        assert!(
            o.stats.drafted as f64 / o.stats.rounds as f64 > 3.0,
            "{:?}",
            o.stats
        );
        let o2 = generate(
            &mut Mock::target(8, 2),
            &mut AlwaysWrong {
                oracle: Mock::target(8, 2),
                hist: vec![],
            },
            &prompt,
            &cfg,
        )
        .unwrap();
        assert!(
            o2.stats.drafted as f64 / (o2.stats.rounds as f64) < 1.6,
            "{:?}",
            o2.stats
        );
        assert_eq!(
            o.tokens, o2.tokens,
            "adaptive length never changes greedy output"
        );
    }

    // ---- hostile inputs -------------------------------------------------------------------------------

    struct Broken(u8);
    impl Verifier for Broken {
        fn logits_batch(&mut self, tokens: &[u32]) -> Vec<Vec<f32>> {
            match self.0 {
                0 => vec![],                                  // wrong count
                1 => tokens.iter().map(|_| vec![]).collect(), // empty logits
                2 => tokens
                    .iter()
                    .enumerate()
                    .map(|(i, _)| vec![0.0; 3 + i])
                    .collect(), // ragged
                _ => tokens.iter().map(|_| vec![f32::NAN; 5]).collect(), // NaN everywhere
            }
        }
        fn rollback(&mut self, _keep: usize) {}
    }

    #[test]
    fn misbehaving_models_and_drafters_are_errors_or_harmless() {
        let cfg = greedy_cfg(10, 4);
        assert_eq!(
            generate(
                &mut Mock::target(4, 1),
                &mut NgramLookup::new(1, 2),
                &[],
                &cfg
            )
            .unwrap_err(),
            SpecError::EmptyPrompt
        );
        assert_eq!(
            generate_plain(&mut Mock::target(4, 1), &[], &cfg).unwrap_err(),
            SpecError::EmptyPrompt
        );
        for kind in 0..3u8 {
            let r = generate(
                &mut Broken(kind),
                &mut NgramLookup::new(1, 2),
                &[1, 2, 3],
                &cfg,
            );
            assert!(matches!(r, Err(SpecError::Verifier(_))), "kind {kind}");
        }
        // NaN logits: no panic, tokens stay inside the vocabulary
        for s in [SamplingCfg::greedy(), SamplingCfg::sampled(1.0, 0, 0.9)] {
            let cfg = SpecConfig {
                sampling: s,
                ..greedy_cfg(10, 3)
            };
            let o = generate(
                &mut Broken(3),
                &mut NgramLookup::new(1, 2),
                &[1, 2, 3, 1, 2],
                &cfg,
            )
            .unwrap();
            assert_eq!(o.tokens.len(), 10);
            assert!(o.tokens.iter().all(|&t| t < 5));
        }
        // a drafter returning far too many tokens, out-of-vocabulary tokens and mismatched dists
        struct Wild(u8);
        impl Drafter for Wild {
            fn draft(&mut self, _c: &[u32], k: usize, _s: &SamplingCfg, _r: &mut Rng) -> Draft {
                match self.0 {
                    0 => Draft::deterministic(vec![1; k * 10 + 50]),
                    1 => Draft::deterministic(vec![u32::MAX, 2, 3]),
                    2 => Draft {
                        tokens: vec![1, 2, 3],
                        dists: Some(vec![vec![0.5; 3]]),
                        alt: Some(u32::MAX),
                    },
                    _ => Draft {
                        tokens: vec![1, 2],
                        dists: Some(vec![vec![f32::NAN; 4], vec![]]),
                        alt: Some(2),
                    },
                }
            }
        }
        for kind in 0..4u8 {
            for s in [SamplingCfg::greedy(), SamplingCfg::sampled(0.9, 0, 1.0)] {
                let cfg = SpecConfig {
                    sampling: s,
                    ..greedy_cfg(20, 4)
                };
                let mut t = Mock::target(4, 4);
                let o = generate(&mut t, &mut Wild(kind), &[1, 2, 3], &cfg).unwrap();
                assert_eq!(o.tokens.len(), 20, "kind {kind}");
                assert!(o.tokens.iter().all(|&x| x < 4));
                check_state(&t, &[1, 2, 3], &o);
            }
        }
        // a huge draft length is capped by max_new
        let o = generate(
            &mut Mock::target(4, 4),
            &mut DraftModel::new(Mock::target(4, 4)),
            &[1],
            &greedy_cfg(5, usize::MAX / 2),
        )
        .unwrap();
        assert_eq!(o.tokens.len(), 5);
        // single-token prompt works (nothing to prefill)
        assert_eq!(
            generate(
                &mut Mock::target(4, 4),
                &mut NgramLookup::new(1, 1),
                &[2],
                &greedy_cfg(3, 2)
            )
            .unwrap()
            .tokens
            .len(),
            3
        );
    }

    #[test]
    fn rng_is_reproducible_and_uniform_enough() {
        let (mut a, mut b) = (Rng::new(1), Rng::new(1));
        assert!((0..100).all(|_| a.next_u64() == b.next_u64()));
        let mut r = Rng::new(2);
        let mean = (0..100_000).map(|_| r.f64()).sum::<f64>() / 100_000.0;
        assert!((mean - 0.5).abs() < 0.01);
        assert!((0..1000).all(|_| {
            let x = r.f64();
            (0.0..1.0).contains(&x)
        }));
    }

    // ---- benchmark-style (run with --ignored --nocapture) -------------------------------------------------

    #[test]
    #[ignore = "prints a table; run with --ignored --nocapture"]
    fn simulated_speedup_versus_acceptance() {
        let vocab = 16;
        let prompt = [1u32, 2, 3, 4];
        let (target_overhead_per_pos, draft_cost) = (0.04f64, 0.08f64);
        println!(
            "{:>6} {:>3} {:>9} {:>11} {:>9}",
            "mix", "k", "accept", "tok/pass", "speedup"
        );
        for mix in [0.0f32, 0.3, 0.5, 0.7, 0.9, 1.0] {
            for k in [1usize, 2, 4, 6, 8] {
                let (mut emitted, mut cost, mut acc, mut drafted, mut passes) =
                    (0usize, 0.0f64, 0usize, 0usize, 0usize);
                for seed in 0..40u64 {
                    let cfg = SpecConfig {
                        max_new: 64,
                        draft_len: k,
                        sampling: SamplingCfg::sampled(0.7, 0, 1.0),
                        seed,
                        ..Default::default()
                    };
                    let o = generate(
                        &mut Mock::target(vocab, seed),
                        &mut DraftModel::new(Mock::draft(vocab, seed, mix)),
                        &prompt,
                        &cfg,
                    )
                    .unwrap();
                    emitted += o.stats.emitted;
                    acc += o.stats.accepted;
                    drafted += o.stats.drafted;
                    passes += o.stats.passes;
                    // one target pass costs 1 plus a little per extra position; each drafted token costs `draft_cost`
                    cost += o.stats.passes as f64
                        + target_overhead_per_pos * o.stats.drafted as f64
                        + draft_cost * o.stats.drafted as f64;
                }
                println!(
                    "{mix:>6.1} {k:>3} {:>8.1}% {:>11.2} {:>8.2}x",
                    100.0 * acc as f64 / drafted.max(1) as f64,
                    emitted as f64 / passes.max(1) as f64,
                    emitted as f64 / cost
                );
            }
        }
    }
}
