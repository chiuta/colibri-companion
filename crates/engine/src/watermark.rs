//! Statistical "green list" watermarking of generated text, and its detector.
//!
//! Before each token is sampled, a keyed hash of the previous `context` tokens splits the
//! vocabulary into a *green* fraction `gamma` and the rest; [`Watermark::apply_to_logits`] (or
//! [`Watermark::apply_to_candidates`]) adds `delta` to the green logits, so green tokens become
//! somewhat more likely without any change to the model. Anyone who holds the key can count the
//! green tokens of a text; unwatermarked text has about `gamma` of them, so the count's z-score
//! and p-value say how surprising the surplus is ([`Watermark::score`], [`Watermark::scan`]).
//!
//! The hook works on a logits row or on [`Candidates`], so it can be placed right before the
//! sampler chain (after penalties and grammar masks, before temperature and truncation).
//!
//! What this can and cannot do, honestly:
//! * It is a statistical signal, not a proof. A p-value of 1e-6 means "this many green tokens
//!   would almost never happen by chance"; it says nothing about intent, and a false positive
//!   rate of `p` is expected on `p` of unwatermarked texts.
//! * Short texts carry little evidence: with `n` scored tokens the smallest possible p-value is
//!   `gamma^n` (1.5e-5 for 8 tokens at gamma 0.25). Plan on 100+ tokens.
//! * Low-entropy text (code, quotes, lists, facts with one right answer) offers the sampler no
//!   freedom to prefer green tokens, so it stays weakly marked however large `delta` is.
//! * Paraphrasing, translating, or editing a large share of the tokens removes the signal;
//!   a model that regenerates the text removes it completely. Large `delta` is more robust but
//!   changes the output distribution more (quality drops, and the text becomes easier to spot).
//! * Anyone with the key can also forge the mark, and anyone without it cannot verify it, so
//!   keep the key secret and do not treat a missing mark as proof of human authorship.
//! * Repeated phrases repeat their (context, token) pairs and would inflate the count; scoring
//!   counts each distinct pair once by default (`dedup`).

use crate::sampling::{Cand, Candidates};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq)]
pub struct WatermarkConfig {
    /// Secret key; the green lists are a keyed function of it.
    pub key: u64,
    /// Green fraction of the vocabulary, in (0, 1).
    pub gamma: f32,
    /// Logit bonus for green tokens, `0..=50`.
    pub delta: f32,
    /// How many previous tokens seed the split, `1..=8`. More makes the mark harder to learn and
    /// more fragile under edits.
    pub context: usize,
    /// Tokens that are never biased and never scored (end of text, newline, padding).
    pub ignore: Vec<u32>,
}
impl Default for WatermarkConfig {
    fn default() -> Self { WatermarkConfig { key: 0, gamma: 0.25, delta: 2.0, context: 1, ignore: vec![] } }
}

#[derive(Debug, PartialEq, Eq)]
pub enum WatermarkError { Gamma, Delta, Context }
impl std::fmt::Display for WatermarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            WatermarkError::Gamma => "gamma must be strictly between 0 and 1",
            WatermarkError::Delta => "delta must be a finite number from 0 to 50",
            WatermarkError::Context => "context must be 1 to 8 tokens",
        })
    }
}
impl std::error::Error for WatermarkError {}

/// Outcome of counting green tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// Tokens that were counted (with a full context, not ignored, distinct when deduplicating).
    pub scored: usize,
    pub green: usize,
    /// `gamma * scored`: what unwatermarked text would give on average.
    pub expected: f64,
    /// `(green - expected) / sqrt(scored * gamma * (1 - gamma))`; 0 when nothing was scored.
    pub z: f64,
    /// One-sided probability of at least this many green tokens by chance (exact binomial up to
    /// 3000 scored tokens, normal approximation beyond).
    pub p_value: f64,
    pub green_fraction: f64,
}

/// Best-scoring stretch of a text, for watermarked passages inside unwatermarked text.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowDetection {
    /// Index into the token list where the best window starts.
    pub start: usize,
    pub detection: Detection,
    pub windows: usize,
    /// `p_value` of the best window times the number of windows (Bonferroni; conservative
    /// because windows overlap), capped at 1.
    pub adjusted_p: f64,
}

#[derive(Clone, Debug)]
pub struct Watermark {
    key: u64,
    gamma: f64,
    delta: f32,
    context: usize,
    ignore: Vec<u32>,
}

const K1: u64 = 0x9e37_79b9_7f4a_7c15;
const K2: u64 = 0xc2b2_ae3d_27d4_eb4f;
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

impl Watermark {
    pub fn new(cfg: WatermarkConfig) -> Result<Watermark, WatermarkError> {
        if !(cfg.gamma.is_finite() && cfg.gamma > 0.0 && cfg.gamma < 1.0) { return Err(WatermarkError::Gamma); }
        if !(cfg.delta.is_finite() && (0.0..=50.0).contains(&cfg.delta)) { return Err(WatermarkError::Delta); }
        if !(1..=8).contains(&cfg.context) { return Err(WatermarkError::Context); }
        let mut ignore = cfg.ignore;
        ignore.sort_unstable();
        ignore.dedup();
        Ok(Watermark { key: cfg.key, gamma: cfg.gamma as f64, delta: cfg.delta, context: cfg.context, ignore })
    }

    pub fn gamma(&self) -> f64 { self.gamma }
    pub fn delta(&self) -> f32 { self.delta }
    pub fn context(&self) -> usize { self.context }

    /// Hash of the key and the last `context` tokens of `prev` (fewer if there are fewer).
    fn context_hash(&self, prev: &[u32]) -> u64 {
        let tail = &prev[prev.len().saturating_sub(self.context)..];
        let mut h = mix(self.key ^ K1);
        for &t in tail { h = mix(h ^ (t as u64 + 1).wrapping_mul(K1)); }
        h
    }
    fn token_hash(ctx: u64, token: u32) -> u64 { mix(ctx ^ (token as u64 + 1).wrapping_mul(K2)) }
    fn is_green_hash(&self, h: u64) -> bool { (((h >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)) < self.gamma }
    fn ignored(&self, t: u32) -> bool { !self.ignore.is_empty() && self.ignore.binary_search(&t).is_ok() }

    /// Is `token` on the green list after the context `prev`?
    pub fn is_green(&self, prev: &[u32], token: u32) -> bool {
        self.is_green_hash(Self::token_hash(self.context_hash(prev), token))
    }

    /// Add `delta` to the logit of every green token. Non-finite logits (removed or banned
    /// tokens) and ignored tokens are left alone. `prev` is everything before the token to be
    /// sampled, prompt included.
    pub fn apply_to_logits(&self, prev: &[u32], logits: &mut [f32]) {
        if self.delta == 0.0 { return; }
        let ctx = self.context_hash(prev);
        for (i, l) in logits.iter_mut().enumerate() {
            if !l.is_finite() { continue; }
            let id = i as u32;
            if self.ignored(id) { continue; }
            if self.is_green_hash(Self::token_hash(ctx, id)) { *l += self.delta; }
        }
    }

    /// The same bias on a candidate list (any order; probabilities are recomputed on demand).
    pub fn apply_to_candidates(&self, prev: &[u32], cands: &mut Candidates) {
        if self.delta == 0.0 { return; }
        let ctx = self.context_hash(prev);
        let items: Vec<Cand> = std::mem::take(&mut cands.items).into_iter().map(|mut c| {
            if c.logit.is_finite() && !self.ignored(c.id) && self.is_green_hash(Self::token_hash(ctx, c.id)) { c.logit += self.delta; }
            c
        }).collect();
        *cands = Candidates::from_items(items);
    }

    /// Probability mass the green list holds in `probs` after context `prev` (a diagnostic: how
    /// strongly this step can be marked).
    pub fn green_mass(&self, prev: &[u32], probs: &[f32]) -> f64 {
        let ctx = self.context_hash(prev);
        probs.iter().enumerate().filter(|(i, _)| self.is_green_hash(Self::token_hash(ctx, *i as u32))).map(|(_, &p)| p as f64).sum()
    }

    /// Count green tokens in `tokens`. `prefix` supplies context for the first tokens (the prompt's
    /// tail, when known); without it the first `context` tokens cannot be scored.
    pub fn score(&self, prefix: &[u32], tokens: &[u32], dedup: bool) -> Detection {
        let mut seen: HashSet<u64> = HashSet::new();
        let (mut n, mut g) = (0usize, 0usize);
        let take = prefix.len().min(self.context);
        let mut buf: Vec<u32> = Vec::with_capacity(take + tokens.len());
        buf.extend_from_slice(&prefix[prefix.len() - take..]);
        buf.extend_from_slice(tokens);
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = take + i;
            if pos < self.context || self.ignored(tok) { continue; }
            let h = Self::token_hash(self.context_hash(&buf[..pos]), tok);
            if dedup && !seen.insert(h) { continue; }
            n += 1;
            if self.is_green_hash(h) { g += 1; }
        }
        self.detection(n, g)
    }

    fn detection(&self, n: usize, g: usize) -> Detection {
        if n == 0 { return Detection { scored: 0, green: 0, expected: 0.0, z: 0.0, p_value: 1.0, green_fraction: 0.0 }; }
        let nf = n as f64;
        let expected = self.gamma * nf;
        let z = (g as f64 - expected) / (nf * self.gamma * (1.0 - self.gamma)).sqrt();
        Detection { scored: n, green: g, expected, z, p_value: binomial_tail(n, g, self.gamma), green_fraction: g as f64 / nf }
    }

    /// Slide a window of `window` tokens over the text (step a quarter window) and report the
    /// most surprising one, so a marked passage inside unmarked text is still found.
    pub fn scan(&self, prefix: &[u32], tokens: &[u32], window: usize, dedup: bool) -> Option<WindowDetection> {
        if tokens.is_empty() || window == 0 { return None; }
        let window = window.min(tokens.len());
        let step = (window / 4).max(1);
        let mut best: Option<(usize, Detection)> = None;
        let mut windows = 0usize;
        let mut start = 0usize;
        loop {
            let end = (start + window).min(tokens.len());
            let ctx_from = start.saturating_sub(self.context);
            let pre: Vec<u32> = if start == 0 { prefix.to_vec() } else { tokens[ctx_from..start].to_vec() };
            let d = self.score(&pre, &tokens[start..end], dedup);
            windows += 1;
            if best.as_ref().is_none_or(|(_, b)| d.p_value < b.p_value || (d.p_value == b.p_value && d.z > b.z)) { best = Some((start, d)); }
            if end == tokens.len() { break; }
            start += step;
        }
        best.map(|(start, detection)| {
            let adjusted_p = (detection.p_value * windows as f64).min(1.0);
            WindowDetection { start, detection, windows, adjusted_p }
        })
    }
}

// ---------------------------------------------------------------------------- statistics

/// Complementary error function (Numerical Recipes' Chebyshev fit, relative error < 1.2e-7).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23 + t * (1.000_023_68 + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06 + t * (0.278_868_07
        + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let r = t * poly.exp();
    if x >= 0.0 { r } else { 2.0 - r }
}
/// Upper tail of the standard normal distribution.
pub fn normal_sf(z: f64) -> f64 { 0.5 * erfc(z / std::f64::consts::SQRT_2) }

/// `P(X >= g)` for `X ~ Binomial(n, p)`: exact up to 3000 trials, normal approximation with
/// continuity correction beyond.
pub fn binomial_tail(n: usize, g: usize, p: f64) -> f64 {
    if g == 0 { return 1.0; }
    if g > n { return 0.0; }
    if n > 3000 {
        let nf = n as f64;
        let sd = (nf * p * (1.0 - p)).sqrt();
        return normal_sf((g as f64 - 0.5 - nf * p) / sd).clamp(0.0, 1.0);
    }
    let mut lnf = vec![0f64; n + 1];
    for i in 1..=n { lnf[i] = lnf[i - 1] + (i as f64).ln(); }
    let (lp, lq) = (p.ln(), (1.0 - p).ln());
    let mut s = 0f64;
    for i in g..=n { s += (lnf[n] - lnf[i] - lnf[n - i] + i as f64 * lp + (n - i) as f64 * lq).exp(); }
    s.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::Rng;

    fn wm(key: u64, gamma: f32, delta: f32, context: usize) -> Watermark { Watermark::new(WatermarkConfig { key, gamma, delta, context, ignore: vec![] }).unwrap() }

    /// Sample `n` tokens from a random model: per step, noisy logits over `vocab` tokens (std about 2),
    /// biased by the watermark when given, then drawn from the softmax.
    fn generate(w: Option<&Watermark>, vocab: usize, n: usize, seed: u64, prompt: &[u32]) -> Vec<u32> {
        let mut rng = Rng::new(seed);
        let mut seq: Vec<u32> = prompt.to_vec();
        for _ in 0..n {
            let mut l: Vec<f32> = (0..vocab).map(|_| ((rng.f32() + rng.f32() + rng.f32() + rng.f32()) - 2.0) * 3.4).collect();
            if let Some(w) = w { w.apply_to_logits(&seq, &mut l); }
            let m = l.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f64> = l.iter().map(|x| ((x - m) as f64).exp()).collect();
            let s: f64 = e.iter().sum();
            let mut u = rng.f32() as f64 * s;
            let mut pick = vocab - 1;
            for (i, v) in e.iter().enumerate() { if u < *v { pick = i; break; } u -= v; }
            seq.push(pick as u32);
        }
        seq[prompt.len()..].to_vec()
    }
    fn uniform_tokens(seed: u64, n: usize, vocab: u32) -> Vec<u32> { let mut r = Rng::new(seed); (0..n).map(|_| (r.next_u64() % vocab as u64) as u32).collect() }

    #[test]
    fn configuration_is_validated() {
        let ok = WatermarkConfig::default();
        assert!(Watermark::new(ok.clone()).is_ok());
        for g in [0.0, 1.0, -0.5, 1.5, f32::NAN, f32::INFINITY] { assert_eq!(Watermark::new(WatermarkConfig { gamma: g, ..ok.clone() }).unwrap_err(), WatermarkError::Gamma); }
        for d in [-1.0, f32::NAN, f32::INFINITY, 51.0] { assert_eq!(Watermark::new(WatermarkConfig { delta: d, ..ok.clone() }).unwrap_err(), WatermarkError::Delta); }
        for c in [0, 9, 1000] { assert_eq!(Watermark::new(WatermarkConfig { context: c, ..ok.clone() }).unwrap_err(), WatermarkError::Context); }
        assert!(Watermark::new(WatermarkConfig { delta: 0.0, ..ok }).is_ok());
    }

    #[test]
    fn green_lists_have_the_right_size_and_depend_on_key_and_context() {
        for gamma in [0.1f32, 0.25, 0.5, 0.9] {
            let w = wm(7, gamma, 2.0, 1);
            let g = (0..200_000u32).filter(|&t| w.is_green(&[42], t)).count() as f64 / 200_000.0;
            assert!((g - gamma as f64).abs() < 0.01, "gamma {gamma}: green share {g}");
        }
        let (a, b) = (wm(1, 0.5, 2.0, 1), wm(2, 0.5, 2.0, 1));
        let same = (0..5000u32).filter(|&t| a.is_green(&[9], t) == b.is_green(&[9], t)).count();
        assert!((2200..2800).contains(&same), "different keys give unrelated lists ({same}/5000 agree)");
        let diff_ctx = (0..5000u32).filter(|&t| a.is_green(&[9], t) == a.is_green(&[10], t)).count();
        assert!((2200..2800).contains(&diff_ctx), "different contexts give unrelated lists ({diff_ctx}/5000 agree)");
        assert_eq!(a.is_green(&[9], 77), a.is_green(&[9], 77));
        // only the last `context` tokens matter
        let w2 = wm(1, 0.5, 2.0, 2);
        for t in 0..200 { assert_eq!(w2.is_green(&[5, 6, 7], t), w2.is_green(&[1, 6, 7], t)); }
    }

    #[test]
    fn the_bias_goes_to_green_tokens_only() {
        let w = wm(3, 0.3, 2.5, 1);
        let mut l: Vec<f32> = (0..1000).map(|i| (i % 7) as f32).collect();
        l[10] = f32::NEG_INFINITY;
        l[11] = f32::NAN;
        let orig = l.clone();
        w.apply_to_logits(&[5], &mut l);
        for i in 0..1000 {
            if i == 11 { assert!(l[i].is_nan()); continue; }
            let expect = if orig[i].is_finite() && w.is_green(&[5], i as u32) { orig[i] + 2.5 } else { orig[i] };
            assert_eq!(l[i], expect, "token {i}");
        }
        assert_eq!(l[10], f32::NEG_INFINITY, "banned tokens stay banned");
        // candidates give the same numbers
        let mut c = Candidates::from_logits(&orig);
        w.apply_to_candidates(&[5], &mut c);
        for cand in &c.items { assert_eq!(cand.logit, if w.is_green(&[5], cand.id) { orig[cand.id as usize] + 2.5 } else { orig[cand.id as usize] }); }
        // ignored tokens are untouched, delta 0 is a no-op
        let wi = Watermark::new(WatermarkConfig { key: 3, gamma: 0.9, delta: 4.0, context: 1, ignore: vec![0, 1, 2, 2] }).unwrap();
        let mut z = vec![0f32; 50];
        wi.apply_to_logits(&[1], &mut z);
        assert!(z[..3].iter().all(|x| *x == 0.0) && z[3..].contains(&4.0));
        let w0 = wm(3, 0.5, 0.0, 1);
        let mut y = vec![1.0f32; 10];
        w0.apply_to_logits(&[1], &mut y);
        assert_eq!(y, vec![1.0; 10]);
        // mass diagnostic
        let probs = vec![0.001f32; 1000];
        let gm = w.green_mass(&[5], &probs);
        assert!((gm - 0.3).abs() < 0.06, "{gm}");
    }

    #[test]
    fn tail_probabilities_are_right() {
        assert!((normal_sf(0.0) - 0.5).abs() < 1e-7);
        assert!((normal_sf(1.96) - 0.0249979).abs() < 1e-6);
        assert!((normal_sf(5.0) / 2.866_515_7e-7 - 1.0).abs() < 1e-4);
        assert!((normal_sf(-1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((binomial_tail(10, 8, 0.5) - 56.0 / 1024.0).abs() < 1e-12);
        assert_eq!(binomial_tail(10, 0, 0.3), 1.0);
        assert_eq!(binomial_tail(10, 11, 0.3), 0.0);
        assert!((binomial_tail(8, 8, 0.25) - 0.25f64.powi(8)).abs() < 1e-15);
        // exact and normal agree around the switch-over
        let (e, a) = (binomial_tail(3000, 800, 0.25), binomial_tail(3001, 800, 0.25));
        assert!(e > 0.0 && (a / e).ln().abs() < 0.3, "{e} vs {a}");
        assert!(binomial_tail(100_000, 30_000, 0.25) < 1e-100 || binomial_tail(100_000, 30_000, 0.25) == 0.0);
    }

    #[test]
    fn unwatermarked_text_gives_the_expected_false_positive_rate() {
        let w = wm(99, 0.25, 2.0, 1);
        let trials = 3000;
        let (mut hits, mut sum, mut sq) = (0usize, 0f64, 0f64);
        for s in 0..trials {
            let t = uniform_tokens(1000 + s, 200, 5000);
            let d = w.score(&[], &t, true);
            assert!(d.scored >= 190, "{}", d.scored);
            if d.p_value < 0.01 { hits += 1; }
            sum += d.z;
            sq += d.z * d.z;
        }
        let rate = hits as f64 / trials as f64;
        assert!(rate < 0.02, "false positive rate at p<0.01 was {rate}");
        let (mean, var) = (sum / trials as f64, sq / trials as f64 - (sum / trials as f64).powi(2));
        assert!(mean.abs() < 0.1 && (var - 1.0).abs() < 0.15, "z has mean {mean}, variance {var}");
    }

    #[test]
    fn watermarked_generation_is_detected() {
        let w = wm(1234, 0.25, 2.0, 1);
        let prompt = [11u32, 22, 33];
        let mut detected = 0;
        let mut worst_z = f64::MAX;
        for s in 0..60 {
            let t = generate(Some(&w), 500, 200, s, &prompt);
            let d = w.score(&prompt, &t, true);
            if d.p_value < 1e-4 { detected += 1; }
            worst_z = worst_z.min(d.z);
        }
        assert!(detected >= 57, "detected {detected}/60 (worst z {worst_z})");
        // the same generator without the mark is not flagged, and a wrong key does not see it
        let mut flagged = 0;
        let other = wm(4321, 0.25, 2.0, 1);
        for s in 0..60 {
            let plain = generate(None, 500, 200, 500 + s, &prompt);
            if w.score(&prompt, &plain, true).p_value < 1e-4 { flagged += 1; }
            let marked = generate(Some(&w), 500, 200, 900 + s, &prompt);
            if other.score(&prompt, &marked, true).p_value < 1e-4 { flagged += 1; }
        }
        assert!(flagged <= 1, "{flagged} false alarms");
        // context of two tokens works the same way
        let w2 = wm(5, 0.25, 2.0, 2);
        let t = generate(Some(&w2), 500, 200, 77, &prompt);
        assert!(w2.score(&prompt, &t, true).p_value < 1e-4);
        // a larger delta marks harder, a zero delta not at all
        let strong = generate(Some(&wm(1234, 0.25, 4.0, 1)), 500, 200, 3, &prompt);
        let weak = generate(Some(&wm(1234, 0.25, 0.5, 1)), 500, 200, 3, &prompt);
        assert!(w.score(&prompt, &strong, true).z > w.score(&prompt, &weak, true).z);
        let none = generate(Some(&wm(1234, 0.25, 0.0, 1)), 500, 200, 3, &prompt);
        assert!(w.score(&prompt, &none, true).p_value > 1e-4);
    }

    #[test]
    fn partial_and_edited_text() {
        let w = wm(2024, 0.25, 2.5, 1);
        let marked = generate(Some(&w), 500, 200, 1, &[1]);
        let before = uniform_tokens(2, 400, 500);
        let after = uniform_tokens(3, 400, 500);
        let mut doc = before.clone();
        doc.extend(&marked);
        doc.extend(&after);
        let whole = w.score(&[], &doc, true);
        let found = w.scan(&[], &doc, 100, true).unwrap();
        assert!(found.adjusted_p < 1e-6, "adjusted p {}", found.adjusted_p);
        assert!((350..=520).contains(&found.start), "best window starts at {}", found.start);
        assert!(found.detection.z > whole.z, "the window is clearer than the whole document");
        assert!(found.windows > 10);
        // an unmarked document of the same length is not flagged by scanning
        let plain = uniform_tokens(4, 1000, 500);
        assert!(w.scan(&[], &plain, 100, true).unwrap().adjusted_p > 0.001);
        // cutting off the start or the end keeps the mark
        assert!(w.score(&[], &marked[57..], true).p_value < 1e-6);
        assert!(w.score(&[], &marked[..90], true).p_value < 1e-3);
        // a prefix supplies the context of the first token
        assert_eq!(w.score(&[1], &marked, true).scored, w.score(&[], &marked, true).scored + 1);
        // replacing tokens weakens the signal gradually; replacing all removes it
        let mut r = Rng::new(9);
        let edit = |frac: f32, r: &mut Rng| -> Vec<u32> { marked.iter().map(|&t| if r.f32() < frac { (r.next_u64() % 500) as u32 } else { t }).collect() };
        let (z30, z60) = (w.score(&[], &edit(0.3, &mut r), true).z, w.score(&[], &edit(0.6, &mut r), true).z);
        let z_full = w.score(&[], &marked, true).z;
        assert!(z_full > z30 && z30 > z60, "z falls with edits: {z_full} {z30} {z60}");
        assert!(z30 > 5.0, "30% replaced is still clearly marked, z={z30}");
        assert!(w.score(&[], &edit(1.0, &mut r), true).p_value > 1e-3, "fully rewritten text carries no mark");
    }

    #[test]
    fn short_texts_cannot_give_strong_evidence_and_repeats_do_not_inflate() {
        let w = wm(8, 0.25, 6.0, 1);
        for s in 0..40 {
            let t = generate(Some(&w), 300, 8, s, &[7]);
            let d = w.score(&[7], &t, true);
            assert!(d.scored <= 8);
            assert!(d.p_value >= 0.25f64.powi(d.scored as i32) * 0.999, "p {} below the floor for n={}", d.p_value, d.scored);
        }
        // a looping phrase would be scored hundreds of times without deduplication
        let phrase = uniform_tokens(5, 5, 1000);
        let looped: Vec<u32> = phrase.iter().cycle().take(400).copied().collect();
        let dedup = w.score(&[], &looped, true);
        let raw = w.score(&[], &looped, false);
        assert!(dedup.scored <= 6, "{}", dedup.scored);
        assert!(raw.scored >= 390);
        assert!(dedup.p_value > 1e-4);
        // empty and ignored-only inputs
        let e = w.score(&[], &[], true);
        assert_eq!((e.scored, e.green, e.z, e.p_value), (0, 0, 0.0, 1.0));
        assert!(w.scan(&[], &[], 10, true).is_none() && w.scan(&[], &[1, 2], 0, true).is_none());
        let wi = Watermark::new(WatermarkConfig { key: 8, gamma: 0.25, delta: 2.0, context: 1, ignore: vec![5] }).unwrap();
        assert_eq!(wi.score(&[], &[5, 5, 5, 5], false).scored, 0);
    }

    #[test]
    fn hostile_inputs_do_not_panic() {
        let w = wm(1, 0.5, 2.0, 8);
        let mut empty: Vec<f32> = vec![];
        w.apply_to_logits(&[], &mut empty);
        let mut c = Candidates::default();
        w.apply_to_candidates(&[], &mut c);
        assert!(c.is_empty());
        let t = [u32::MAX, 0, u32::MAX, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let d = w.score(&[u32::MAX; 20], &t, false);
        assert!(d.p_value.is_finite() && d.z.is_finite());
        assert!(w.scan(&[1, 2, 3], &t, 1000, true).is_some());
        let big = Watermark::new(WatermarkConfig { key: u64::MAX, gamma: 0.999, delta: 50.0, context: 8, ignore: vec![u32::MAX] }).unwrap();
        let mut l = vec![f32::MAX; 16];
        big.apply_to_logits(&[1; 3], &mut l);
        assert!(l.iter().all(|x| !x.is_nan()));
        assert!((big.green_mass(&[], &[]) - 0.0).abs() < 1e-12);
    }
}
