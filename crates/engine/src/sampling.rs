//! Sampler chain: penalties, DRY, top-n-σ, top-k, tail-free, typical, top-p, min-p, XTC,
//! (dynamic) temperature and Mirostat v1/v2, with grammar masks, logit bias and OpenAI-style
//! log-probabilities.
//!
//! Standalone: it knows nothing about models or tokenizers. A caller hands it one row of
//! logits plus the token history and gets a token back. Written from the published
//! descriptions of each method (the llama.cpp default order is kept), no code copied.
//!
//! Speed: the common configuration (top-k first) never builds a candidate list for the whole
//! vocabulary. One pass over the logits keeps a bounded buffer of survivors with a rising
//! threshold, so a 150k vocabulary costs one sanitising copy plus one scan.
//! Robustness: NaN logits count as removed, ±∞ is clamped, empty input or a mask that rejects
//! everything yields `Sampled::empty`, and every truncation keeps at least `min_keep` tokens.
use crate::simd;
use serde_json::Value;
use std::collections::VecDeque;

/// Logits at or below this are "removed" (masked, NaN, banned). Finite so arithmetic stays safe.
const DEAD: f32 = -1.0e30;
const ALIVE: f32 = -1.0e29;
const BIG: f32 = 1.0e30;
/// Below this temperature we simply take the arg-max.
const MIN_TEMP: f32 = 1.0e-4;
const MAX_WINDOW: usize = 1 << 16;
const DRY_MAX_WINDOW: usize = 1 << 14;
const MAX_PROBS: usize = 1000;

fn alive(x: f32) -> bool { x > ALIVE }
fn clean(x: f32) -> f32 { if x.is_nan() { DEAD } else { x.clamp(-BIG, BIG) } }

// ------------------------------------------------------------------------------- rng

/// xorshift64* with a splitmix seed scrambler. Deterministic per seed.
#[derive(Clone, Debug)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        Rng(if z == 0 { 0x2545_f491_4f6c_dd1d } else { z })
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in `[0, 1)`.
    pub fn f32(&mut self) -> f32 { (self.next_u64() >> 40) as f32 / 16_777_216.0 }
}

fn time_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    t ^ N.fetch_add(0x9e37_79b9, Ordering::Relaxed).wrapping_mul(0x2545_f491_4f6c_dd1d)
}

// ------------------------------------------------------------------------ candidates

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cand { pub id: u32, pub logit: f32, pub p: f32 }

/// Descending by logit, ties by lower id: the order every selection in this module uses.
fn by_logit(a: &Cand, b: &Cand) -> std::cmp::Ordering { b.logit.total_cmp(&a.logit).then(a.id.cmp(&b.id)) }

/// A list of candidate tokens with lazily computed probabilities and sort order.
#[derive(Clone, Debug, Default)]
pub struct Candidates {
    pub items: Vec<Cand>,
    sorted: bool,
    probs: bool,
    scratch: Vec<f32>,
}

impl Candidates {
    /// All finite logits of a row (NaN and removed entries are dropped).
    pub fn from_logits(l: &[f32]) -> Candidates {
        let items = l.iter().enumerate().map(|(i, &x)| (i, clean(x))).filter(|&(_, x)| alive(x))
            .map(|(i, x)| Cand { id: i as u32, logit: x, p: 0.0 }).collect();
        Candidates { items, sorted: false, probs: false, scratch: vec![] }
    }
    pub fn from_items(items: Vec<Cand>) -> Candidates { Candidates { items, sorted: false, probs: false, scratch: vec![] } }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn is_empty(&self) -> bool { self.items.is_empty() }
    pub fn ids(&self) -> Vec<u32> { self.items.iter().map(|c| c.id).collect() }
    fn invalidate(&mut self) { self.probs = false; }

    pub fn sort(&mut self) {
        if !self.sorted { self.items.sort_unstable_by(by_logit); self.sorted = true; }
    }

    /// Probabilities over the current candidates (renormalised).
    pub fn softmax(&mut self) {
        if self.probs || self.items.is_empty() { return; }
        let m = self.items.iter().fold(f32::NEG_INFINITY, |a, c| a.max(c.logit));
        if self.items.len() > 256 {
            let mut s = std::mem::take(&mut self.scratch);
            s.clear();
            s.extend(self.items.iter().map(|c| c.logit));
            let sum = simd::exp_sub_sum(&mut s, m);
            let k = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for (c, e) in self.items.iter_mut().zip(&s) { c.p = e * k; }
            self.scratch = s;
        } else {
            let mut sum = 0.0;
            for c in &mut self.items { c.p = (c.logit - m).exp(); sum += c.p; }
            let k = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for c in &mut self.items { c.p *= k; }
        }
        self.probs = true;
    }

    fn keep_floor(&mut self, min_keep: usize) -> usize { min_keep.max(1).min(self.items.len()) }

    /// Keep the `k` best (at least `min_keep`). `k == 0` disables.
    pub fn top_k(&mut self, k: usize, min_keep: usize) {
        if k == 0 || self.items.is_empty() { return; }
        let k = k.max(min_keep.max(1)).min(self.items.len());
        if k >= self.items.len() { return; }
        if self.sorted { self.items.truncate(k); } else {
            self.items.select_nth_unstable_by(k - 1, by_logit);
            self.items.truncate(k);
            self.sorted = false;
        }
        self.invalidate();
    }

    /// Nucleus: smallest prefix whose probability mass reaches `p`. `p >= 1` disables.
    pub fn top_p(&mut self, p: f32, min_keep: usize) {
        if p >= 1.0 || self.items.is_empty() { return; }
        self.sort();
        self.softmax();
        let floor = self.keep_floor(min_keep);
        let mut cum = 0.0;
        let mut keep = self.items.len();
        for (i, c) in self.items.iter().enumerate() {
            cum += c.p;
            if cum >= p { keep = i + 1; break; }
        }
        self.items.truncate(keep.max(floor));
        self.invalidate();
    }

    /// Keep tokens with `p >= m · p_max`. `m <= 0` disables.
    pub fn min_p(&mut self, m: f32, min_keep: usize) {
        if m <= 0.0 || self.items.is_empty() { return; }
        let m = m.min(1.0);
        let top = self.items.iter().fold(f32::NEG_INFINITY, |a, c| a.max(c.logit));
        let thr = top + m.ln();
        let floor = self.keep_floor(min_keep);
        if self.items.iter().filter(|c| c.logit >= thr).count() >= floor {
            self.items.retain(|c| c.logit >= thr);
        } else {
            self.sort();
            self.items.truncate(floor);
        }
        self.invalidate();
    }

    /// Locally typical sampling: keep the tokens whose surprise is closest to the entropy
    /// until their mass reaches `p`. `p` outside `(0,1)` disables.
    pub fn typical(&mut self, p: f32, min_keep: usize) {
        if !(p > 0.0 && p < 1.0) || self.items.len() < 2 { return; }
        self.softmax();
        let h: f32 = -self.items.iter().filter(|c| c.p > 0.0).map(|c| c.p * c.p.ln()).sum::<f32>();
        let mut v: Vec<(f32, Cand)> = self.items.iter().map(|c| (((-c.p.max(1e-30).ln()) - h).abs(), *c)).collect();
        v.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.id.cmp(&b.1.id)));
        let floor = min_keep.max(1).min(v.len());
        let mut cum = 0.0;
        let mut keep = v.len();
        for (i, (_, c)) in v.iter().enumerate() {
            cum += c.p;
            if cum >= p { keep = i + 1; break; }
        }
        v.truncate(keep.max(floor));
        self.items = v.into_iter().map(|(_, c)| c).collect();
        self.sorted = false;
        self.probs = false;
    }

    /// Tail-free sampling (llama.cpp formulation): cut where the cumulative normalised
    /// |second derivative| of the sorted probabilities exceeds `z`. `z >= 1` disables.
    pub fn tfs(&mut self, z: f32, min_keep: usize) {
        if !(z > 0.0 && z < 1.0) || self.items.len() <= 2 { return; }
        self.sort();
        self.softmax();
        let p: Vec<f32> = self.items.iter().map(|c| c.p).collect();
        let d1: Vec<f32> = p.windows(2).map(|w| w[0] - w[1]).collect();
        let d2: Vec<f32> = d1.windows(2).map(|w| (w[0] - w[1]).abs()).collect();
        let sum: f32 = d2.iter().sum();
        if sum.is_nan() || sum <= 0.0 { return; }
        let mut cum = 0.0;
        let mut last = self.items.len();
        for (i, d) in d2.iter().enumerate() {
            cum += d / sum;
            if cum > z && i >= min_keep { last = i; break; }
        }
        let floor = self.keep_floor(min_keep);
        self.items.truncate(last.max(floor));
        self.invalidate();
    }

    /// Top-n-σ: keep logits within `n` standard deviations of the maximum. `n <= 0` disables.
    pub fn top_n_sigma(&mut self, n: f32, min_keep: usize) {
        if n <= 0.0 || self.items.len() < 2 { return; }
        let (mut max, mut s, mut s2) = (f32::NEG_INFINITY, 0.0f64, 0.0f64);
        for c in &self.items { max = max.max(c.logit); s += c.logit as f64; s2 += (c.logit as f64) * (c.logit as f64); }
        let cnt = self.items.len() as f64;
        let mean = s / cnt;
        let sigma = (s2 / cnt - mean * mean).max(0.0).sqrt() as f32;
        let thr = max - n * sigma;
        let floor = self.keep_floor(min_keep);
        if self.items.iter().filter(|c| c.logit >= thr).count() >= floor { self.items.retain(|c| c.logit >= thr); }
        else { self.sort(); self.items.truncate(floor); }
        self.invalidate();
    }

    /// Exclude-top-choices: among tokens with `p >= threshold` drop all but the least likely.
    /// Needs at least two such tokens. (The caller decides whether to apply it at all.)
    pub fn xtc(&mut self, threshold: f32, min_keep: usize) {
        if !(threshold > 0.0 && threshold <= 0.5) || self.items.len() < 2 { return; }
        self.sort();
        self.softmax();
        let cnt = self.items.iter().take_while(|c| c.p >= threshold).count();
        if cnt >= 2 && self.items.len() - (cnt - 1) >= min_keep.max(1) {
            self.items.drain(..cnt - 1);
            self.invalidate();
        }
    }

    pub fn temperature(&mut self, t: f32) {
        if t <= 0.0 || t == 1.0 { return; }
        let k = 1.0 / t;
        for c in &mut self.items { c.logit *= k; }
        self.invalidate();
    }

    /// Entropy-scaled temperature in `[t - range, t + range]`.
    pub fn dynamic_temperature(&mut self, t: f32, range: f32, exponent: f32) {
        if self.items.len() <= 1 { return; }
        self.softmax();
        let h: f32 = -self.items.iter().filter(|c| c.p > 0.0).map(|c| c.p * c.p.ln()).sum::<f32>();
        let max_h = (self.items.len() as f32).ln();
        let norm = if max_h > 0.0 { (h / max_h).clamp(0.0, 1.0) } else { 0.0 };
        let (lo, hi) = ((t - range).max(1e-3), t + range);
        let temp = lo + (hi - lo) * norm.powf(exponent.max(1e-3));
        self.temperature(temp);
    }

    /// Highest logit, lowest id on ties.
    pub fn argmax(&self) -> Option<u32> {
        self.items.iter().min_by(|a, b| by_logit(a, b)).map(|c| c.id)
    }

    /// Index chosen by cumulative probability for a uniform `u` in `[0,1)`. Needs `softmax`.
    pub fn draw(&self, u: f32) -> usize {
        let mut cum = 0.0;
        for (i, c) in self.items.iter().enumerate() {
            cum += c.p;
            if u < cum { return i; }
        }
        // rounding left a sliver: the last candidate with positive mass
        self.items.iter().rposition(|c| c.p > 0.0).unwrap_or(self.items.len().saturating_sub(1))
    }
}

/// Keep the `k` largest alive values of `vals` (ties: lower id), best first. One pass with a
/// rising threshold; the buffer is compacted only when it fills, so the cost is O(n).
fn scan_top_k(vals: &[f32], k: usize, out: &mut Vec<Cand>) {
    out.clear();
    if k == 0 { return; }
    let cap = (4 * k).max(64);
    let mut thr = ALIVE;
    for (i, &v) in vals.iter().enumerate() {
        if v > thr {
            out.push(Cand { id: i as u32, logit: v, p: 0.0 });
            if out.len() >= cap {
                out.select_nth_unstable_by(k - 1, by_logit);
                out.truncate(k);
                // the k-th is not necessarily the minimum of the kept set
                thr = out.iter().fold(f32::INFINITY, |a, c| a.min(c.logit)).max(ALIVE);
            }
        }
    }
    if out.len() > k {
        out.select_nth_unstable_by(k - 1, by_logit);
        out.truncate(k);
    }
    out.sort_unstable_by(by_logit);
}

// ------------------------------------------------------------------------------ mask

/// A bitset over token ids: bit set = token allowed.
#[derive(Clone, Debug)]
pub struct TokenMask { words: Vec<u64>, n: usize }
impl TokenMask {
    pub fn new(n: usize, allowed: bool) -> TokenMask {
        let mut words = vec![if allowed { u64::MAX } else { 0 }; n.div_ceil(64)];
        if allowed && !n.is_multiple_of(64) && let Some(w) = words.last_mut() { *w = (1u64 << (n % 64)) - 1; }
        TokenMask { words, n }
    }
    pub fn from_fn(n: usize, f: impl Fn(u32) -> bool) -> TokenMask {
        let mut m = TokenMask::new(n, false);
        for i in 0..n { if f(i as u32) { m.allow(i as u32); } }
        m
    }
    pub fn len(&self) -> usize { self.n }
    pub fn is_empty(&self) -> bool { self.n == 0 }
    pub fn allow(&mut self, id: u32) { if (id as usize) < self.n { self.words[id as usize / 64] |= 1 << (id % 64); } }
    pub fn deny(&mut self, id: u32) { if (id as usize) < self.n { self.words[id as usize / 64] &= !(1 << (id % 64)); } }
    pub fn allows(&self, id: u32) -> bool { (id as usize) < self.n && self.words[id as usize / 64] >> (id % 64) & 1 == 1 }
    pub fn count(&self) -> usize { self.words.iter().map(|w| w.count_ones() as usize).sum() }
}

enum Mask<'a> { None, Bits(&'a TokenMask), Func(&'a dyn Fn(u32) -> bool) }

// ----------------------------------------------------------------------------- params

/// One step of the chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage { Penalties, Dry, TopNSigma, TopK, TfsZ, TypicalP, TopP, MinP, Xtc, Temperature }
impl Stage {
    pub fn parse(s: &str) -> Option<Stage> {
        Some(match s.to_ascii_lowercase().as_str() {
            "penalties" | "penalty" | "repetition" => Stage::Penalties,
            "dry" => Stage::Dry,
            "top_n_sigma" | "top-n-sigma" => Stage::TopNSigma,
            "top_k" | "top-k" => Stage::TopK,
            "tfs_z" | "tfs" => Stage::TfsZ,
            "typ_p" | "typical_p" | "typical" => Stage::TypicalP,
            "top_p" | "top-p" => Stage::TopP,
            "min_p" | "min-p" => Stage::MinP,
            "xtc" => Stage::Xtc,
            "temperature" | "temp" => Stage::Temperature,
            _ => return None,
        })
    }
}
/// llama.cpp's default order.
pub const DEFAULT_ORDER: [Stage; 9] = [Stage::Penalties, Stage::Dry, Stage::TopNSigma, Stage::TopK, Stage::TypicalP,
    Stage::TopP, Stage::MinP, Stage::Xtc, Stage::Temperature];

#[derive(Clone, Debug, PartialEq)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub typical_p: f32,
    pub top_n_sigma: f32,
    pub tfs_z: f32,
    pub xtc_probability: f32,
    pub xtc_threshold: f32,
    pub mirostat: u8,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    /// `-1` = whole history, `0` = off.
    pub dry_penalty_last_n: i64,
    pub dry_sequence_breakers: Vec<u32>,
    /// Breakers given as text in JSON; resolve them with the tokenizer and call
    /// [`SamplerChain::set_dry_breakers`].
    pub dry_breaker_strings: Vec<String>,
    pub repeat_penalty: f32,
    /// `-1` = whole history, `0` = off.
    pub repeat_last_n: i64,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub seed: Option<u64>,
    /// Return the log-probability of the chosen token.
    pub logprobs: bool,
    /// Also return this many best alternatives (implies `logprobs`).
    pub n_probs: usize,
    pub ignore_eos: bool,
    pub min_keep: usize,
    pub dynatemp_range: f32,
    pub dynatemp_exponent: f32,
    pub order: Vec<Stage>,
}

impl Default for SamplerParams {
    /// The llama.cpp defaults.
    fn default() -> Self {
        SamplerParams {
            temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, typical_p: 1.0, top_n_sigma: -1.0, tfs_z: 1.0,
            xtc_probability: 0.0, xtc_threshold: 0.1, mirostat: 0, mirostat_tau: 5.0, mirostat_eta: 0.1,
            dry_multiplier: 0.0, dry_base: 1.75, dry_allowed_length: 2, dry_penalty_last_n: -1,
            dry_sequence_breakers: vec![], dry_breaker_strings: vec![],
            repeat_penalty: 1.0, repeat_last_n: 64, frequency_penalty: 0.0, presence_penalty: 0.0,
            logit_bias: vec![], seed: None, logprobs: false, n_probs: 0, ignore_eos: false, min_keep: 1,
            dynatemp_range: 0.0, dynatemp_exponent: 1.0, order: DEFAULT_ORDER.to_vec(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParamError { NotAnObject, Json(String), Type(&'static str), Value(&'static str, String) }
impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NotAnObject => write!(f, "sampler parameters must be a JSON object"),
            ParamError::Json(e) => write!(f, "bad JSON: {e}"),
            ParamError::Type(k) => write!(f, "`{k}` has the wrong type"),
            ParamError::Value(k, why) => write!(f, "`{k}`: {why}"),
        }
    }
}
impl std::error::Error for ParamError {}

fn num(v: &Value, key: &'static str) -> Result<Option<f64>, ParamError> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(x) => x.as_f64().filter(|f| f.is_finite()).map(Some).ok_or(ParamError::Type(key)),
    }
}
fn boolean(v: &Value, key: &'static str) -> Result<Option<bool>, ParamError> {
    match v.get(key) { None | Some(Value::Null) => Ok(None), Some(Value::Bool(b)) => Ok(Some(*b)), Some(_) => Err(ParamError::Type(key)) }
}
fn token_id(x: &Value, key: &'static str) -> Result<u32, ParamError> {
    x.as_u64().filter(|n| *n <= u32::MAX as u64).map(|n| n as u32).ok_or(ParamError::Type(key))
}

impl SamplerParams {
    /// Identity settings: temperature 1, nothing truncated, no penalties.
    pub fn neutral() -> SamplerParams {
        SamplerParams { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, ..Default::default() }
    }

    pub fn from_json(s: &str) -> Result<SamplerParams, ParamError> {
        let v: Value = serde_json::from_str(s).map_err(|e| ParamError::Json(e.to_string()))?;
        SamplerParams::from_value(&v)
    }
    pub fn from_value(v: &Value) -> Result<SamplerParams, ParamError> { SamplerParams::default().overlay(v) }

    /// Apply the fields present in `v` on top of `self`. Numbers out of range are clamped, wrong
    /// types and unknown enum values are errors, unknown keys are ignored.
    pub fn overlay(mut self, v: &Value) -> Result<SamplerParams, ParamError> {
        if !v.is_object() { return Err(ParamError::NotAnObject); }
        if let Some(x) = num(v, "temperature")? { self.temperature = (x as f32).clamp(0.0, 1000.0); }
        if let Some(x) = num(v, "top_k")? { self.top_k = if x < 1.0 { 0 } else { x.min(1.0e9) as usize }; }
        if let Some(x) = num(v, "top_p")? { self.top_p = (x as f32).clamp(0.0, 1.0); }
        if let Some(x) = num(v, "min_p")? { self.min_p = (x as f32).clamp(0.0, 1.0); }
        if let Some(x) = match num(v, "typical_p")? { Some(x) => Some(x), None => num(v, "typ_p")? } { self.typical_p = (x as f32).clamp(0.0, 1.0); }
        if let Some(x) = num(v, "top_n_sigma")? { self.top_n_sigma = (x as f32).clamp(-1.0, 100.0); }
        if let Some(x) = num(v, "tfs_z")? { self.tfs_z = (x as f32).clamp(0.0, 1.0); }
        if let Some(x) = num(v, "xtc_probability")? { self.xtc_probability = (x as f32).clamp(0.0, 1.0); }
        if let Some(x) = num(v, "xtc_threshold")? { self.xtc_threshold = (x as f32).clamp(0.0, 1.0); }
        if let Some(x) = num(v, "mirostat")? {
            if x != 0.0 && x != 1.0 && x != 2.0 { return Err(ParamError::Value("mirostat", format!("{x} is not 0, 1 or 2"))); }
            self.mirostat = x as u8;
        }
        if let Some(x) = num(v, "mirostat_tau")? { self.mirostat_tau = (x as f32).clamp(0.0, 100.0); }
        if let Some(x) = num(v, "mirostat_eta")? { self.mirostat_eta = (x as f32).clamp(0.0, 1.0); }
        if let Some(x) = num(v, "dry_multiplier")? { self.dry_multiplier = (x as f32).clamp(0.0, 1000.0); }
        if let Some(x) = num(v, "dry_base")? { self.dry_base = (x as f32).clamp(1.0, 100.0); }
        if let Some(x) = num(v, "dry_allowed_length")? { self.dry_allowed_length = x.clamp(0.0, 1.0e6) as usize; }
        if let Some(x) = num(v, "dry_penalty_last_n")? { self.dry_penalty_last_n = if x < 0.0 { -1 } else { x.min(1.0e9) as i64 }; }
        match v.get("dry_sequence_breakers") {
            None | Some(Value::Null) => {}
            Some(Value::Array(a)) => {
                self.dry_sequence_breakers.clear();
                self.dry_breaker_strings.clear();
                for x in a {
                    match x {
                        Value::String(s) => self.dry_breaker_strings.push(s.clone()),
                        _ => self.dry_sequence_breakers.push(token_id(x, "dry_sequence_breakers")?),
                    }
                }
            }
            Some(_) => return Err(ParamError::Type("dry_sequence_breakers")),
        }
        if let Some(x) = num(v, "repeat_penalty").or_else(|_| num(v, "repetition_penalty"))? {
            self.repeat_penalty = if x <= 0.0 { 1.0 } else { (x as f32).min(100.0) };
        } else if let Some(x) = num(v, "repetition_penalty")? { self.repeat_penalty = if x <= 0.0 { 1.0 } else { (x as f32).min(100.0) }; }
        if let Some(x) = num(v, "repeat_last_n")? { self.repeat_last_n = if x < 0.0 { -1 } else { x.min(1.0e9) as i64 }; }
        if let Some(x) = num(v, "frequency_penalty")? { self.frequency_penalty = (x as f32).clamp(-10.0, 10.0); }
        if let Some(x) = num(v, "presence_penalty")? { self.presence_penalty = (x as f32).clamp(-10.0, 10.0); }
        if let Some(b) = v.get("logit_bias") { self.logit_bias = parse_bias(b)?; }
        match v.get("seed") {
            None | Some(Value::Null) => {}
            Some(x) => {
                self.seed = if let Some(u) = x.as_u64() { Some(u) } else if x.as_i64().is_some() { None }
                    else if x.as_f64().is_some_and(|f| f >= 0.0 && f.fract() == 0.0 && f < 1.8e19) { Some(x.as_f64().unwrap_or(0.0) as u64) }
                    else { return Err(ParamError::Type("seed")); };
            }
        }
        if let Some(b) = boolean(v, "logprobs")? { self.logprobs = b; }
        let n = num(v, "n_probs")?.or(num(v, "top_logprobs")?);
        if let Some(x) = n { self.n_probs = x.clamp(0.0, MAX_PROBS as f64) as usize; }
        if self.n_probs > 0 { self.logprobs = true; }
        if let Some(b) = boolean(v, "ignore_eos")? { self.ignore_eos = b; }
        if let Some(x) = num(v, "min_keep")? { self.min_keep = x.clamp(1.0, 1.0e6) as usize; }
        if let Some(x) = num(v, "dynatemp_range")? { self.dynatemp_range = (x as f32).clamp(0.0, 1000.0); }
        if let Some(x) = num(v, "dynatemp_exponent")? { self.dynatemp_exponent = if x > 0.0 { (x as f32).min(100.0) } else { 1.0 }; }
        match v.get("samplers") {
            None | Some(Value::Null) => {}
            Some(Value::Array(a)) => {
                let mut order = vec![];
                for x in a {
                    let s = x.as_str().ok_or(ParamError::Type("samplers"))?;
                    order.push(Stage::parse(s).ok_or_else(|| ParamError::Value("samplers", format!("unknown sampler `{s}`")))?);
                }
                if order.len() > 64 { return Err(ParamError::Value("samplers", "more than 64 entries".into())); }
                self.order = order;
            }
            Some(_) => return Err(ParamError::Type("samplers")),
        }
        Ok(self)
    }
}

fn bias_value(v: &Value) -> Result<Option<f32>, ParamError> {
    match v {
        Value::Null => Ok(None),
        Value::Bool(false) => Ok(Some(DEAD)),
        Value::Number(n) => n.as_f64().filter(|f| f.is_finite()).map(|f| Some((f as f32).clamp(-BIG, BIG))).ok_or(ParamError::Type("logit_bias")),
        _ => Err(ParamError::Type("logit_bias")),
    }
}
fn parse_bias(b: &Value) -> Result<Vec<(u32, f32)>, ParamError> {
    let mut out = vec![];
    match b {
        Value::Null => {}
        Value::Object(m) => {
            if m.len() > 1 << 16 { return Err(ParamError::Value("logit_bias", "too many entries".into())); }
            for (k, v) in m {
                let id: u32 = k.trim().parse().map_err(|_| ParamError::Value("logit_bias", format!("`{k}` is not a token id")))?;
                if let Some(x) = bias_value(v)? { out.push((id, x)); }
            }
        }
        Value::Array(a) => {
            if a.len() > 1 << 16 { return Err(ParamError::Value("logit_bias", "too many entries".into())); }
            for e in a {
                let pair = e.as_array().filter(|p| p.len() == 2).ok_or(ParamError::Type("logit_bias"))?;
                let id = token_id(&pair[0], "logit_bias")?;
                if let Some(x) = bias_value(&pair[1])? { out.push((id, x)); }
            }
        }
        _ => return Err(ParamError::Type("logit_bias")),
    }
    out.sort_by_key(|p| p.0);
    Ok(out)
}

// ------------------------------------------------------------------------------ chain

#[derive(Clone, Debug, PartialEq)]
pub struct Sampled {
    pub token: u32,
    /// Log-probability under the raw model distribution (before penalties, truncation and
    /// temperature), when requested.
    pub logprob: Option<f32>,
    /// Best alternatives by raw logit, `(token, logprob)`, best first (`n_probs` of them).
    pub top_logprobs: Vec<(u32, f32)>,
    /// Nothing could be sampled (no logits, or the mask rejected every token).
    pub empty: bool,
}
impl Sampled {
    fn none() -> Sampled { Sampled { token: 0, logprob: None, top_logprobs: vec![], empty: true } }
}

pub struct SamplerChain {
    p: SamplerParams,
    rng: Rng,
    mu: f32,
    last_surprise: f32,
    eos: Option<u32>,
    hist: VecDeque<u32>,
    // reusable buffers
    work: Vec<f32>,
    tmp: Vec<f32>,
    cands: Candidates,
    pen: Vec<(u32, u32)>,
    dry: Vec<(u32, f32)>,
    win: Vec<u32>,
    z: Vec<u32>,
}

impl SamplerChain {
    pub fn new(params: SamplerParams) -> SamplerChain {
        let seed = params.seed.unwrap_or_else(time_seed);
        let mu = 2.0 * params.mirostat_tau;
        let mut p = params;
        p.dry_sequence_breakers.sort_unstable();
        p.dry_sequence_breakers.dedup();
        SamplerChain { p, rng: Rng::new(seed), mu, last_surprise: 0.0, eos: None, hist: VecDeque::new(),
            work: vec![], tmp: vec![], cands: Candidates::default(), pen: vec![], dry: vec![], win: vec![], z: vec![] }
    }
    pub fn params(&self) -> &SamplerParams { &self.p }
    pub fn set_eos(&mut self, id: Option<u32>) { self.eos = id; }
    pub fn set_seed(&mut self, seed: u64) { self.rng = Rng::new(seed); }
    pub fn set_dry_breakers(&mut self, mut ids: Vec<u32>) { ids.sort_unstable(); ids.dedup(); self.p.dry_sequence_breakers = ids; }
    /// Mirostat's running target; starts at `2 · tau`.
    pub fn mu(&self) -> f32 { self.mu }
    /// Surprise (bits) of the token chosen by the last Mirostat step.
    pub fn last_surprise(&self) -> f32 { self.last_surprise }
    /// Forget the accepted history and Mirostat state; the RNG stream continues.
    pub fn reset(&mut self) { self.hist.clear(); self.mu = 2.0 * self.p.mirostat_tau; self.last_surprise = 0.0; }

    /// Record a token as part of the sequence so [`sample_accepted`](Self::sample_accepted)
    /// can apply penalties without the caller keeping the history. Mirostat's `mu` is
    /// updated at sample time, not here.
    pub fn accept(&mut self, token: u32) {
        self.hist.push_back(token);
        let cap = MAX_WINDOW.max(DRY_MAX_WINDOW);
        while self.hist.len() > cap { self.hist.pop_front(); }
    }

    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> Sampled { self.run(logits, history, Mask::None) }
    /// Like [`sample`](Self::sample) but only tokens for which `allowed` is true may be chosen.
    pub fn sample_masked(&mut self, logits: &[f32], history: &[u32], allowed: Option<&dyn Fn(u32) -> bool>) -> Sampled {
        match allowed { Some(f) => self.run(logits, history, Mask::Func(f)), None => self.run(logits, history, Mask::None) }
    }
    pub fn sample_with_bits(&mut self, logits: &[f32], history: &[u32], mask: &TokenMask) -> Sampled { self.run(logits, history, Mask::Bits(mask)) }
    /// Sample using the history recorded with [`accept`](Self::accept).
    pub fn sample_accepted(&mut self, logits: &[f32]) -> Sampled {
        let h: Vec<u32> = self.hist.iter().copied().collect();
        self.run(logits, &h, Mask::None)
    }

    /// The logits after mask, bias, EOS ban, penalties and DRY — nothing is truncated.
    /// For diagnostics and tests.
    pub fn adjusted_logits(&mut self, logits: &[f32], history: &[u32], allowed: Option<&dyn Fn(u32) -> bool>) -> Vec<f32> {
        let mask = match allowed { Some(f) => Mask::Func(f), None => Mask::None };
        let mut w = std::mem::take(&mut self.work);
        self.prepare(&mut w, logits, &mask);
        self.build_penalties(history);
        self.apply_sparse_full(&mut w, true, true);
        let out = w.clone();
        self.work = w;
        out
    }

    fn mods_needed(&self, mask: &Mask<'_>) -> bool {
        self.penalties_on() || self.dry_on() || !self.p.logit_bias.is_empty()
            || (self.p.ignore_eos && self.eos.is_some()) || !matches!(mask, Mask::None)
    }
    fn penalties_on(&self) -> bool {
        self.p.repeat_last_n != 0 && (self.p.repeat_penalty != 1.0 || self.p.frequency_penalty != 0.0 || self.p.presence_penalty != 0.0)
    }
    fn dry_on(&self) -> bool { self.p.dry_multiplier > 0.0 && self.p.dry_penalty_last_n != 0 }

    /// Sanitising copy plus mask, bias and EOS ban.
    fn prepare(&self, w: &mut Vec<f32>, logits: &[f32], mask: &Mask<'_>) {
        w.clear();
        w.extend(logits.iter().map(|&x| clean(x)));
        let n = w.len();
        match mask {
            Mask::None => {}
            Mask::Bits(m) => {
                for (wi, &bits) in m.words.iter().enumerate() {
                    if bits == u64::MAX { continue; }
                    for b in 0..64 {
                        let id = wi * 64 + b;
                        if id >= n { break; }
                        if bits >> b & 1 == 0 { w[id] = DEAD; }
                    }
                }
                for x in w.iter_mut().skip(m.n) { *x = DEAD; }
            }
            Mask::Func(f) => { for (i, x) in w.iter_mut().enumerate() { if alive(*x) && !f(i as u32) { *x = DEAD; } } }
        }
        for &(id, b) in &self.p.logit_bias {
            if let Some(x) = w.get_mut(id as usize)
                && alive(*x) { *x = if b <= DEAD { DEAD } else { clean(*x + b) }; }
        }
        if self.p.ignore_eos
            && let Some(x) = self.eos.and_then(|e| w.get_mut(e as usize)) { *x = DEAD; }
    }

    fn build_penalties(&mut self, history: &[u32]) {
        self.pen.clear();
        self.dry.clear();
        if self.penalties_on() {
            let n = if self.p.repeat_last_n < 0 { MAX_WINDOW } else { (self.p.repeat_last_n as usize).min(MAX_WINDOW) };
            let w = &history[history.len().saturating_sub(n)..];
            self.win.clear();
            self.win.extend_from_slice(w);
            self.win.sort_unstable();
            let mut i = 0;
            while i < self.win.len() {
                let mut j = i;
                while j < self.win.len() && self.win[j] == self.win[i] { j += 1; }
                self.pen.push((self.win[i], (j - i) as u32));
                i = j;
            }
        }
        if self.dry_on() { self.compute_dry(history); }
    }

    fn compute_dry(&mut self, history: &[u32]) {
        let n = if self.p.dry_penalty_last_n < 0 { DRY_MAX_WINDOW } else { (self.p.dry_penalty_last_n as usize).min(DRY_MAX_WINDOW) };
        let w = &history[history.len().saturating_sub(n)..];
        let allowed = self.p.dry_allowed_length.max(1);
        if w.len() < allowed + 1 { return; }
        let br = &self.p.dry_sequence_breakers;
        let is_br = |t: u32| br.binary_search(&t).is_ok();
        let suffix = match w.iter().rposition(|&t| is_br(t)) { Some(p) => w.len() - 1 - p, None => w.len() };
        if suffix < allowed { return; }
        // Z-function of the reversed window: z[j] = length of the common suffix between the
        // window ending at the last token and the window ending j tokens earlier.
        self.win.clear();
        self.win.extend(w.iter().rev());
        let r = &self.win;
        let m = r.len();
        self.z.clear();
        self.z.resize(m, 0);
        let (mut l, mut rr) = (0usize, 0usize);
        for i in 1..m {
            let mut zi = if i < rr { (rr - i).min(self.z[i - l] as usize) } else { 0 };
            while i + zi < m && r[zi] == r[i + zi] { zi += 1; }
            self.z[i] = zi as u32;
            if i + zi > rr { l = i; rr = i + zi; }
        }
        let base = self.p.dry_base as f64;
        let max_exp = if base > 1.0 + 1e-6 { (f32::MAX as f64).ln() / base.ln() } else { f64::MAX };
        for j in 1..m {
            let len = (self.z[j] as usize).min(suffix);
            if len < allowed { continue; }
            let tok = r[j - 1];
            if is_br(tok) { continue; }
            let e = ((len - allowed) as f64).min(max_exp);
            let pen = (self.p.dry_multiplier as f64 * base.powf(e)).min(1.0e20) as f32;
            match self.dry.iter_mut().find(|(t, _)| *t == tok) {
                Some(slot) => slot.1 = slot.1.max(pen),
                None => self.dry.push((tok, pen)),
            }
        }
        self.dry.sort_unstable_by_key(|p| p.0);
    }

    fn adjust(&self, id: u32, mut l: f32, pen: bool, dry: bool) -> f32 {
        if pen
            && let Ok(i) = self.pen.binary_search_by_key(&id, |p| p.0) {
                let c = self.pen[i].1 as f32;
                if self.p.repeat_penalty != 1.0 { l = if l > 0.0 { l / self.p.repeat_penalty } else { l * self.p.repeat_penalty }; }
                l -= c * self.p.frequency_penalty + self.p.presence_penalty;
            }
        if dry
            && let Ok(i) = self.dry.binary_search_by_key(&id, |p| p.0) { l -= self.dry[i].1; }
        l
    }
    fn apply_sparse_full(&self, w: &mut [f32], pen: bool, dry: bool) {
        if pen { for &(id, _) in &self.pen { if let Some(x) = w.get_mut(id as usize) && alive(*x) { *x = clean(self.adjust(id, *x, true, false)); } } }
        if dry { for &(id, _) in &self.dry { if let Some(x) = w.get_mut(id as usize) && alive(*x) { *x = clean(self.adjust(id, *x, false, true)); } } }
    }
    fn apply_sparse_list(&self, c: &mut Candidates, pen: bool, dry: bool) {
        for it in &mut c.items { it.logit = clean(self.adjust(it.id, it.logit, pen, dry)); }
        c.sorted = false;
        c.invalidate();
    }

    fn run(&mut self, logits: &[f32], history: &[u32], mask: Mask<'_>) -> Sampled {
        let n = logits.len();
        if n == 0 { return Sampled::none(); }
        let greedy = self.p.temperature < MIN_TEMP;
        let tok = if greedy && !self.mods_needed(&mask) {
            let mut best: Option<(u32, f32)> = None;
            for (i, &x) in logits.iter().enumerate() {
                let x = clean(x);
                if alive(x) && best.is_none_or(|b| x > b.1) { best = Some((i as u32, x)); }
            }
            best.map(|b| b.0)
        } else {
            self.run_chain(logits, history, &mask, greedy)
        };
        match tok {
            None => Sampled::none(),
            Some(t) => self.finish(t, logits),
        }
    }

    fn run_chain(&mut self, logits: &[f32], history: &[u32], mask: &Mask<'_>, greedy: bool) -> Option<u32> {
        let mut w = std::mem::take(&mut self.work);
        self.prepare(&mut w, logits, mask);
        self.build_penalties(history);
        let mut full = true; // candidates live in `w` (true) or in `self.cands` (false)
        let mut cands = std::mem::take(&mut self.cands);
        let miro = !greedy && self.p.mirostat != 0;
        let min_keep = self.p.min_keep.max(1);
        for si in 0..self.p.order.len() {
            let st = self.p.order[si];
            match st {
                Stage::Penalties => { if full { self.apply_sparse_full(&mut w, true, false); } else { self.apply_sparse_list(&mut cands, true, false); } }
                Stage::Dry => { if full { self.apply_sparse_full(&mut w, false, true); } else { self.apply_sparse_list(&mut cands, false, true); } }
                _ if greedy => {}
                Stage::Temperature => {
                    let t = self.p.temperature;
                    if self.p.dynatemp_range > 0.0 {
                        if full { cands = candidates_from(&w, std::mem::take(&mut cands)); full = false; }
                        cands.dynamic_temperature(t, self.p.dynatemp_range, self.p.dynatemp_exponent);
                    } else if t != 1.0 {
                        if full { let k = 1.0 / t; for x in w.iter_mut() { if alive(*x) { *x *= k; } } } else { cands.temperature(t); }
                    }
                }
                _ if miro => {}
                Stage::TopK => {
                    if self.p.top_k > 0 {
                        if full {
                            scan_top_k(&w, self.p.top_k.max(min_keep), &mut cands.items);
                            cands.sorted = true; cands.probs = false; full = false;
                        } else { cands.top_k(self.p.top_k, min_keep); }
                    }
                }
                Stage::MinP => {
                    if self.p.min_p > 0.0 {
                        if full {
                            let mx = simd::max(&w);
                            let thr = mx + self.p.min_p.min(1.0).ln();
                            cands.items.clear();
                            cands.items.extend(w.iter().enumerate().filter(|&(_, &x)| alive(x) && x >= thr).map(|(i, &x)| Cand { id: i as u32, logit: x, p: 0.0 }));
                            cands.sorted = false; cands.probs = false; full = false;
                            if cands.items.len() < min_keep {
                                scan_top_k(&w, min_keep, &mut cands.items);
                                cands.sorted = true;
                            }
                        } else { cands.min_p(self.p.min_p, min_keep); }
                    }
                }
                Stage::TopNSigma => {
                    if self.p.top_n_sigma > 0.0 {
                        if full { cands = candidates_from(&w, std::mem::take(&mut cands)); full = false; }
                        cands.top_n_sigma(self.p.top_n_sigma, min_keep);
                    }
                }
                Stage::TfsZ | Stage::TypicalP | Stage::TopP | Stage::Xtc => {
                    let active = match st {
                        Stage::TfsZ => self.p.tfs_z > 0.0 && self.p.tfs_z < 1.0,
                        Stage::TypicalP => self.p.typical_p > 0.0 && self.p.typical_p < 1.0,
                        Stage::TopP => self.p.top_p < 1.0,
                        _ => self.p.xtc_probability > 0.0,
                    };
                    if !active { continue; }
                    if st == Stage::Xtc {
                        let u = self.rng.f32(); // always drawn, so the stream does not depend on the outcome
                        if u >= self.p.xtc_probability { continue; }
                    }
                    if full { cands = candidates_from(&w, std::mem::take(&mut cands)); full = false; }
                    match st {
                        Stage::TfsZ => cands.tfs(self.p.tfs_z, min_keep),
                        Stage::TypicalP => cands.typical(self.p.typical_p, min_keep),
                        Stage::TopP => cands.top_p(self.p.top_p, min_keep),
                        _ => cands.xtc(self.p.xtc_threshold, min_keep),
                    }
                }
            }
        }
        let token = self.select(&mut w, &mut cands, full, greedy, miro);
        self.work = w;
        self.cands = cands;
        token
    }

    #[allow(clippy::ptr_arg)]
    fn select(&mut self, w: &mut Vec<f32>, cands: &mut Candidates, full: bool, greedy: bool, miro: bool) -> Option<u32> {
        if full && !miro {
            if greedy {
                let mut best: Option<(u32, f32)> = None;
                for (i, &x) in w.iter().enumerate() { if alive(x) && best.is_none_or(|b| x > b.1) { best = Some((i as u32, x)); } }
                return best.map(|b| b.0);
            }
            // Whole-vocabulary draw: in-place softmax on the work buffer.
            let m = simd::max(w);
            if !alive(m) { return None; }
            let sum = simd::exp_sub_sum(w, m);
            if !sum.is_finite() || sum <= 0.0 { return None; }
            let target = self.rng.f32() * sum;
            let (mut cum, mut last) = (0.0f32, None);
            for (i, &e) in w.iter().enumerate() {
                if e > 0.0 {
                    cum += e;
                    last = Some(i as u32);
                    if target < cum { return last; }
                }
            }
            return last;
        }
        if full { *cands = candidates_from(w, std::mem::take(cands)); }
        if cands.is_empty() { return None; }
        if greedy { return cands.argmax(); }
        if miro { return self.mirostat(cands); }
        cands.softmax();
        let idx = cands.draw(self.rng.f32());
        cands.items.get(idx).map(|c| c.id)
    }

    fn mirostat(&mut self, c: &mut Candidates) -> Option<u32> {
        c.sort();
        c.softmax();
        let (tau, eta) = (self.p.mirostat_tau, self.p.mirostat_eta);
        if self.p.mirostat == 2 {
            let mut keep = c.items.iter().take_while(|x| -(x.p.max(1e-30)).log2() <= self.mu).count();
            keep = keep.max(1).min(c.items.len());
            c.items.truncate(keep);
        } else {
            let m = 100.min(c.items.len());
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in 0..m.saturating_sub(1) {
                let (a, b) = (c.items[i].p as f64, c.items[i + 1].p as f64);
                if a <= 0.0 || b <= 0.0 { break; }
                let t = (((i + 2) as f64) / ((i + 1) as f64)).ln();
                num += t * (a / b).ln();
                den += t * t;
            }
            let s_hat = if den > 0.0 { (num / den).max(1e-3) } else { 1.0 };
            let eps = s_hat - 1.0;
            let nv = c.items.len() as f64;
            let k = ((eps * 2f64.powf(self.mu as f64)) / (1.0 - nv.powf(-eps))).powf(1.0 / s_hat);
            let k = if k.is_finite() { (k as usize).clamp(1, c.items.len()) } else { c.items.len() };
            c.items.truncate(k);
        }
        c.probs = false;
        c.softmax();
        let idx = c.draw(self.rng.f32());
        let chosen = *c.items.get(idx)?;
        self.last_surprise = -(chosen.p.max(1e-30)).log2();
        self.mu -= eta * (self.last_surprise - tau);
        if !self.mu.is_finite() { self.mu = 2.0 * tau; }
        Some(chosen.id)
    }

    /// Log-probabilities from the raw row (after only NaN/∞ cleaning).
    fn finish(&mut self, token: u32, logits: &[f32]) -> Sampled {
        if !(self.p.logprobs || self.p.n_probs > 0) {
            return Sampled { token, logprob: None, top_logprobs: vec![], empty: false };
        }
        let mut t = std::mem::take(&mut self.tmp);
        t.clear();
        t.extend(logits.iter().map(|&x| clean(x)));
        let m = simd::max(&t);
        let mut raw_of_token = t.get(token as usize).copied().unwrap_or(DEAD);
        if !alive(raw_of_token) { raw_of_token = DEAD; }
        let top = {
            let mut v = vec![];
            if self.p.n_probs > 0 { scan_top_k(&t, self.p.n_probs, &mut v); }
            v
        };
        let lse = if alive(m) {
            let s = simd::exp_sub_sum(&mut t, m);
            m + s.max(f32::MIN_POSITIVE).ln()
        } else { 0.0 };
        self.tmp = t;
        let lp = |x: f32| if alive(x) { x - lse } else { f32::NEG_INFINITY };
        Sampled { token, logprob: Some(lp(raw_of_token)), top_logprobs: top.iter().map(|c| (c.id, lp(c.logit))).collect(), empty: false }
    }
}

/// Build a candidate list from the work buffer, reusing the old list's allocation.
fn candidates_from(w: &[f32], mut old: Candidates) -> Candidates {
    old.items.clear();
    old.items.extend(w.iter().enumerate().filter(|&(_, &x)| alive(x)).map(|(i, &x)| Cand { id: i as u32, logit: x, p: 0.0 }));
    old.sorted = false;
    old.probs = false;
    old
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]
mod tests {
    use super::*;

    fn chain(f: impl FnOnce(&mut SamplerParams)) -> SamplerChain {
        let mut p = SamplerParams::neutral();
        p.seed = Some(7);
        f(&mut p);
        SamplerChain::new(p)
    }
    fn freq(c: &mut SamplerChain, l: &[f32], n: usize) -> Vec<usize> {
        let mut f = vec![0; l.len()];
        for _ in 0..n { f[c.sample(l, &[]).token as usize] += 1; }
        f
    }

    #[test]
    fn greedy_and_ties() {
        let mut c = chain(|p| p.temperature = 0.0);
        assert_eq!(c.sample(&[1.0, 5.0, 5.0, 2.0], &[]).token, 1);
        assert_eq!(c.sample(&[f32::NAN, f32::INFINITY, 3.0], &[]).token, 1);
    }
    #[test]
    fn robust_inputs() {
        let mut c = chain(|p| { p.top_k = 0; p.top_p = 0.0; p.min_p = 2.0; });
        assert!(c.sample(&[], &[]).empty);
        assert!(c.sample(&[f32::NAN; 4], &[]).empty);
        let s = c.sample(&[f32::NEG_INFINITY, 1.0, f32::NAN], &[]);
        assert_eq!(s.token, 1);
        let mut c = chain(|p| p.temperature = 0.5);
        assert!(c.sample(&[1.0, 2.0], &[]).token < 2);
    }
    #[test]
    fn mask_all_rejected_and_partial() {
        let mut c = chain(|_| {});
        let none = |_: u32| false;
        assert!(c.sample_masked(&[1.0, 2.0], &[], Some(&none)).empty);
        let odd = |i: u32| i % 2 == 1;
        for _ in 0..50 { assert_eq!(c.sample_masked(&[9.0, 1.0, 9.0, 1.0], &[], Some(&odd)).token % 2, 1); }
        let m = TokenMask::from_fn(4, |i| i == 3);
        assert_eq!(c.sample_with_bits(&[9.0, 9.0, 9.0, 0.0], &[], &m).token, 3);
        assert_eq!(m.count(), 1);
    }
    #[test]
    fn plain_softmax_statistics() {
        let l = [0.0f32, 1.0, 2.0];
        let mut c = chain(|_| {});
        let n = 40_000;
        let f = freq(&mut c, &l, n);
        let z: f32 = l.iter().map(|x| x.exp()).sum();
        for i in 0..3 {
            let e = l[i].exp() / z * n as f32;
            assert!((f[i] as f32 - e).abs() < 5.0 * e.sqrt(), "{f:?}");
        }
    }
    #[test]
    fn temperature_sharpens() {
        let l = [0.0f32, 1.0, 2.0];
        let mut c = chain(|p| p.temperature = 0.1);
        let f = freq(&mut c, &l, 2000);
        assert!(f[2] > 1990);
    }
    #[test]
    fn top_k_p_minp() {
        let l = [3.0f32, 2.0, 1.0, 0.0, -1.0];
        let mut c = chain(|p| p.top_k = 2);
        let f = freq(&mut c, &l, 3000);
        assert_eq!(f[2] + f[3] + f[4], 0);
        // probabilities of l: .636 .234 .086 .032 .012 → top_p 0.8 keeps 2
        let mut c = chain(|p| p.top_p = 0.8);
        let f = freq(&mut c, &l, 3000);
        assert_eq!(f[2] + f[3] + f[4], 0);
        assert!(f[1] > 0);
        // min_p 0.3: keeps p >= 0.3*0.636=0.19 → 2 tokens
        let mut c = chain(|p| p.min_p = 0.3);
        let f = freq(&mut c, &l, 3000);
        assert_eq!(f[2] + f[3] + f[4], 0);
        assert!(f[0] > f[1]);
    }
    #[test]
    fn truncation_renormalises_probabilities() {
        // Regression: top_p / top_k used to leave the pre-truncation probabilities in place, so
        // the draw gave the leftover mass to the last kept token (0.364 instead of 0.269 here).
        let l = [3.0f32, 2.0, 1.0, 0.0, -1.0];
        let mut a = Candidates::from_logits(&l);
        a.top_p(0.8, 1);
        a.softmax();
        let sum: f32 = a.items.iter().map(|c| c.p).sum();
        assert!((sum - 1.0).abs() < 1e-5, "mass after top_p = {sum}");
        let mut b = Candidates::from_logits(&l);
        b.top_k(2, 1);
        b.softmax();
        let sum: f32 = b.items.iter().map(|c| c.p).sum();
        assert!((sum - 1.0).abs() < 1e-5, "mass after top_k = {sum}");
        let want = (1.0f32).exp() / (1.0f32 + (1.0f32).exp()); // p(token 1 | top-2) = e^-1/(1+e^-1) ... checked below
        let _ = want;
        let r = b.items.iter().find(|c| c.id == 1).unwrap().p;
        let exact = (2.0f32).exp() / ((3.0f32).exp() + (2.0f32).exp());
        assert!((r - exact).abs() < 1e-5, "p(token 1) = {r}, expected {exact}");
    }
    #[test]
    fn min_keep_floor() {
        let l = [5.0f32, 0.0, 0.0, 0.0];
        let mut c = chain(|p| { p.top_p = 0.01; p.min_keep = 3; });
        let f = freq(&mut c, &l, 4000);
        assert!(f[1] + f[2] + f[0] > 0 && f[3] == 0 || f[0] > 0);
        assert_eq!(f[3], 0);
        let mut cs = Candidates::from_logits(&l);
        cs.top_p(0.01, 3);
        assert_eq!(cs.len(), 3);
    }
    #[test]
    fn top_k_large_vocab_matches_sort() {
        let mut r = Rng::new(3);
        let l: Vec<f32> = (0..20_000).map(|_| r.f32() * 10.0).collect();
        let mut out = vec![];
        scan_top_k(&l, 40, &mut out);
        let mut all: Vec<Cand> = l.iter().enumerate().map(|(i, &x)| Cand { id: i as u32, logit: x, p: 0.0 }).collect();
        all.sort_by(by_logit);
        assert_eq!(out, all[..40].to_vec());
    }
    #[test]
    fn typical_tfs_sigma_xtc() {
        let l = [4.0f32, 3.0, 2.5, 0.0, -2.0, -6.0];
        let mut c = Candidates::from_logits(&l);
        c.typical(0.5, 1);
        assert!(c.len() < 6 && !c.is_empty());
        let mut c = Candidates::from_logits(&l);
        c.tfs(0.5, 1);
        assert!(c.len() < 6);
        let mut c = Candidates::from_logits(&l);
        c.top_n_sigma(1.0, 1);
        assert!(c.items.iter().all(|x| x.logit >= 4.0 - 1.0 * 3.4));
        assert!(c.len() < 6);
        // XTC with threshold .1 removes the top tokens above it except the last of them
        let mut c = Candidates::from_logits(&[2.0, 2.0, 0.0, -10.0]);
        c.xtc(0.1, 1);
        assert_eq!(c.items.iter().map(|x| x.id).collect::<Vec<_>>(), vec![1, 2, 3]);
        let mut c = chain(|p| { p.xtc_probability = 1.0; p.xtc_threshold = 0.1; p.top_k = 0; });
        let f = freq(&mut c, &[3.0, 3.0, 0.0], 500);
        assert_eq!(f[0], 0);
        let mut c = chain(|p| { p.xtc_probability = 0.0; });
        assert!(freq(&mut c, &[3.0, 3.0, 0.0], 500)[0] > 100);
    }
    #[test]
    fn penalties() {
        let mut c = chain(|p| { p.repeat_penalty = 2.0; p.frequency_penalty = 1.0; p.presence_penalty = 0.5; p.repeat_last_n = 8; });
        let a = c.adjusted_logits(&[4.0, -2.0, 1.0], &[0, 0, 1], None);
        // token 0 seen twice: 4/2 - (2*1+.5) = -0.5 ; token 1 once: -2*2 - 1.5 = -5.5
        assert!((a[0] + 0.5).abs() < 1e-5 && (a[1] + 5.5).abs() < 1e-5 && a[2] == 1.0);
        let b = c.adjusted_logits(&[4.0, -2.0, 1.0], &[], None);
        assert_eq!(b, vec![4.0, -2.0, 1.0]);
        let mut c = chain(|p| { p.repeat_penalty = 2.0; p.repeat_last_n = 2; });
        let a = c.adjusted_logits(&[4.0, 4.0, 4.0], &[0, 1, 2], None);
        assert_eq!(a, vec![4.0, 2.0, 2.0]);
    }
    #[test]
    fn logit_bias_and_eos() {
        let mut c = chain(|p| { p.temperature = 0.0; p.logit_bias = vec![(1, 100.0), (0, -1e35)]; });
        assert_eq!(c.sample(&[9.0, 1.0, 2.0], &[]).token, 1);
        let mut c = chain(|p| { p.temperature = 0.0; p.ignore_eos = true; });
        c.set_eos(Some(0));
        assert_eq!(c.sample(&[9.0, 1.0], &[]).token, 1);
        let v = SamplerParams::from_json(r#"{"logit_bias":{"5":-100,"6":false}}"#).unwrap();
        assert_eq!(v.logit_bias, vec![(5, -100.0), (6, DEAD)]);
    }
    #[test]
    fn dry_penalises_repeat_continuation() {
        let mut c = chain(|p| { p.temperature = 0.0; p.dry_multiplier = 5.0; p.dry_allowed_length = 2; p.dry_base = 2.0; });
        // history: 1 2 3 9 1 2 3 → suffix "1 2 3" matched earlier, next token would be 9
        let h = [1, 2, 3, 9, 1, 2, 3];
        let a = c.adjusted_logits(&[0.0; 12], &h, None);
        // match length 3, allowed 2 → penalty 5*2^(3-2)=10
        assert!((a[9] + 10.0).abs() < 1e-4, "{}", a[9]);
        assert_eq!(a[4], 0.0);
        let s = c.sample(&{ let mut l = vec![0.0; 12]; l[9] = 1.0; l[5] = 0.5; l }, &h);
        assert_eq!(s.token, 5);
        // a breaker stops the match
        let mut c2 = chain(|p| { p.temperature = 0.0; p.dry_multiplier = 5.0; p.dry_sequence_breakers = vec![2]; });
        let a = c2.adjusted_logits(&[0.0; 12], &h, None);
        assert_eq!(a[9], 0.0);
    }
    #[test]
    fn mirostat_v2_tracks_tau() {
        let mut r = Rng::new(11);
        let mut c = chain(|p| { p.mirostat = 2; p.mirostat_tau = 3.0; p.mirostat_eta = 0.1; p.seed = Some(5); });
        let mut tot = 0.0;
        let n = 1500;
        for i in 0..n {
            let l: Vec<f32> = (0..500).map(|j| -(j as f32).ln_1p() * 1.5 + r.f32() * 0.2).collect();
            c.sample(&l, &[]);
            if i >= n / 2 { tot += c.last_surprise(); }
        }
        let mean = tot / (n / 2) as f32;
        assert!((mean - 3.0).abs() < 0.6, "mean surprise {mean}");
        let mut c1 = chain(|p| { p.mirostat = 1; p.mirostat_tau = 3.0; });
        for _ in 0..50 { let l: Vec<f32> = (0..200).map(|j| -(j as f32).ln_1p() * 1.5).collect(); assert!(c1.sample(&l, &[]).token < 200); }
        assert!(c1.mu().is_finite());
    }
    #[test]
    fn determinism() {
        let l: Vec<f32> = (0..300).map(|i| ((i * 37 % 101) as f32) / 10.0).collect();
        let run = |seed| { let mut c = chain(|p| { p.seed = Some(seed); p.top_k = 20; p.top_p = 0.9; p.temperature = 0.9; p.xtc_probability = 0.3; });
            (0..40).map(|_| c.sample(&l, &[]).token).collect::<Vec<_>>() };
        assert_eq!(run(1), run(1));
        assert_ne!(run(1), run(2));
    }
    #[test]
    fn logprobs_match_log_softmax() {
        let l = [1.0f32, 2.0, 3.0];
        let mut c = chain(|p| { p.temperature = 0.0; p.logprobs = true; p.n_probs = 2; });
        let s = c.sample(&l, &[]);
        let z: f32 = l.iter().map(|x| x.exp()).sum();
        assert_eq!(s.token, 2);
        assert!((s.logprob.unwrap() - (3.0 - z.ln())).abs() < 1e-4);
        assert_eq!(s.top_logprobs.len(), 2);
        assert_eq!(s.top_logprobs[1].0, 1);
        let mut c = chain(|_| {});
        assert!(c.sample(&l, &[]).logprob.is_none());
    }
    #[test]
    fn dynatemp_and_accept() {
        let mut c = chain(|p| { p.temperature = 1.0; p.dynatemp_range = 0.5; });
        assert!(c.sample(&[1.0, 2.0, 3.0], &[]).token < 3);
        let mut c = chain(|p| { p.temperature = 0.0; p.repeat_penalty = 3.0; });
        c.accept(1);
        assert_eq!(c.sample_accepted(&[1.0, 2.0]).token, 0);
    }
    #[test]
    fn json_params() {
        let p = SamplerParams::from_json(r#"{"temperature":0.3,"top_k":-5,"top_p":7,"seed":42,"n_probs":3,"samplers":["top_k","temp"],
            "dry_sequence_breakers":["\n",5],"mirostat":2,"repetition_penalty":1.2}"#).unwrap();
        assert_eq!(p.top_k, 0);
        assert_eq!(p.top_p, 1.0);
        assert_eq!(p.seed, Some(42));
        assert!(p.logprobs && p.n_probs == 3);
        assert_eq!(p.order, vec![Stage::TopK, Stage::Temperature]);
        assert_eq!(p.dry_sequence_breakers, vec![5]);
        assert_eq!(p.dry_breaker_strings, vec!["\n".to_string()]);
        assert_eq!(p.mirostat, 2);
        assert_eq!(p.repeat_penalty, 1.2);
    }
    #[test]
    fn hostile_json() {
        for s in ["[]", "null", "3", r#"{"temperature":"hot"}"#, r#"{"mirostat":3}"#, r#"{"samplers":["nope"]}"#,
                  r#"{"logit_bias":{"x":1}}"#, r#"{"logit_bias":5}"#, r#"{"seed":"a"}"#, r#"{"dry_sequence_breakers":{"a":1}}"#, "{", ""] {
            assert!(SamplerParams::from_json(s).is_err(), "{s}");
        }
        let p = SamplerParams::from_json(r#"{"temperature":1e300,"top_k":1e300,"min_keep":0,"n_probs":1e9,"seed":-1,"top_p":-3}"#).unwrap();
        assert!(p.temperature <= 1000.0 && p.min_keep == 1 && p.n_probs == MAX_PROBS && p.top_p == 0.0);
        let mut c = SamplerChain::new(p);
        assert!(c.sample(&[1.0, 2.0, 3.0], &[]).token < 3);
        assert!(SamplerParams::from_json("{}").is_ok());
    }
    #[test]
    fn masked_sentinel_exactly_zero() {
        let mut w = vec![1.0, DEAD, 2.0, DEAD];
        let m = simd::max(&w);
        let mut z = w.clone();
        let s = simd::exp_sub_sum(&mut z, m);
        assert_eq!(z[1], 0.0);
        assert_eq!(z[3], 0.0);
        assert!(s > 1.0);
        w.clear();
    }
    #[test]
    fn all_stages_random_fuzz_never_panics() {
        let mut r = Rng::new(99);
        for round in 0..200 {
            let n = 1 + (r.next_u64() % 400) as usize;
            let l: Vec<f32> = (0..n).map(|_| match r.next_u64() % 20 { 0 => f32::NAN, 1 => f32::INFINITY, 2 => f32::NEG_INFINITY, _ => r.f32() * 40.0 - 20.0 }).collect();
            let h: Vec<u32> = (0..(r.next_u64() % 60)).map(|_| (r.next_u64() % (n as u64 + 3)) as u32).collect();
            let mut p = SamplerParams::default();
            p.seed = Some(round);
            p.top_k = (r.next_u64() % 5) as usize * 7;
            p.top_p = r.f32();
            p.min_p = r.f32() * 0.5;
            p.typical_p = r.f32();
            p.tfs_z = r.f32();
            p.top_n_sigma = r.f32() * 3.0 - 0.5;
            p.xtc_probability = r.f32();
            p.mirostat = (r.next_u64() % 3) as u8;
            p.dry_multiplier = r.f32();
            p.dry_sequence_breakers = vec![1];
            p.repeat_penalty = 1.0 + r.f32();
            p.dynatemp_range = if round % 3 == 0 { 0.4 } else { 0.0 };
            p.temperature = if round % 7 == 0 { 0.0 } else { r.f32() * 2.0 };
            p.n_probs = (round % 4) as usize;
            p.order = vec![Stage::TfsZ, Stage::Penalties, Stage::Dry, Stage::TopNSigma, Stage::TopK, Stage::TypicalP, Stage::TopP, Stage::MinP, Stage::Xtc, Stage::Temperature];
            let mut c = SamplerChain::new(p);
            for _ in 0..3 {
                let s = c.sample(&l, &h);
                if !s.empty { assert!((s.token as usize) < n); let v = l[s.token as usize]; assert!(!v.is_nan() && v != f32::NEG_INFINITY); }
            }
        }
    }
}
