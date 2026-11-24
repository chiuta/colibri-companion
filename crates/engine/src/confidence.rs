//! Uncertainty analytics for trustworthy answers.
//!
//! Everything here is *data for a UI*: per-token statistics, word/sentence/answer roll-ups
//! (serialisable to JSON for a heat-map), a transparent hallucination-risk **heuristic**,
//! self-consistency scores over several sampled answers, and calibration tools.
//!
//! The risk score is a heuristic built from visible features. It is **not** ground truth and
//! says nothing about whether a statement is true; calibrate it on labelled data with
//! [`Calibrator`] before reading it as a probability.
//!
//! The module is standalone: it needs only logits, the answer text and token byte offsets.
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};

/// Largest surprisal we report (nats); keeps every number finite and JSON-safe.
pub const MAX_SURPRISAL: f32 = 1000.0;
/// Most answers [`consistency`] accepts (pairwise work is quadratic).
pub const MAX_SAMPLES: usize = 64;
const MAX_LCS_WORDS: usize = 2000;

#[derive(Debug, Clone, PartialEq)]
pub enum ConfError {
    Empty,
    LengthMismatch { a: usize, b: usize },
    BadSpan { index: usize },
    NonFinite,
    NeedBothClasses,
    BadParam(&'static str),
    Json(String),
}
impl std::fmt::Display for ConfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfError::Empty => write!(f, "nothing to analyse"),
            ConfError::LengthMismatch { a, b } => write!(f, "length mismatch: {a} vs {b}"),
            ConfError::BadSpan { index } => write!(f, "token {index} has an invalid byte span"),
            ConfError::NonFinite => write!(f, "a score is not a finite number"),
            ConfError::NeedBothClasses => write!(f, "the labels contain only one class"),
            ConfError::BadParam(p) => write!(f, "invalid parameter: {p}"),
            ConfError::Json(s) => write!(f, "invalid JSON: {s}"),
        }
    }
}
impl std::error::Error for ConfError {}

fn fin(x: f32) -> Value {
    if x.is_finite() { json!((x as f64 * 1e6).round() / 1e6) } else { Value::Null }
}
fn opt(x: Option<f32>) -> Value { x.map_or(Value::Null, fin) }

// ------------------------------------------------------------------ per-step statistics

/// Statistics of one next-token distribution (natural logs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepStats {
    pub entropy: f32,
    pub top1_prob: f32,
    pub top2_prob: f32,
    /// `top1_prob - top2_prob`.
    pub margin: f32,
    /// Variance of the surprisal under the distribution.
    pub varentropy: f32,
    /// `exp(entropy)`: how many equally likely tokens the model is torn between.
    pub eff_vocab: f32,
    /// 1-based rank of the chosen token (1 = the most likely), when a token was given.
    pub rank: Option<u32>,
    pub surprisal: Option<f32>,
    pub chosen_prob: Option<f32>,
}
impl StepStats {
    /// Probability the model gave the token it emitted; falls back to the top-1 probability.
    pub fn confidence(&self) -> f32 { self.chosen_prob.unwrap_or(self.top1_prob) }
    pub fn to_json(&self) -> Value {
        json!({"entropy": fin(self.entropy), "top1": fin(self.top1_prob), "top2": fin(self.top2_prob),
            "margin": fin(self.margin), "varentropy": fin(self.varentropy), "eff_vocab": fin(self.eff_vocab),
            "rank": self.rank, "surprisal": opt(self.surprisal), "chosen_prob": opt(self.chosen_prob)})
    }
}

/// Statistics for one step, single pass over the logits plus one for the rank, no allocation.
/// NaN logits count as "impossible"; `+inf` logits share all the probability. Returns `None`
/// when no token has any probability (empty row, all NaN / `-inf`).
pub fn step_stats(logits: &[f32], chosen: Option<usize>) -> Option<StepStats> {
    if logits.is_empty() { return None; }
    let (mut m1, mut m2, mut n_inf) = (f32::NEG_INFINITY, f32::NEG_INFINITY, 0usize);
    for &l in logits {
        if l.is_nan() { continue; }
        if l == f32::INFINITY { n_inf += 1; continue; }
        if l > m1 { m2 = m1; m1 = l; } else if l > m2 { m2 = l; }
    }
    let chosen_logit = chosen.and_then(|c| logits.get(c).copied());
    let rank_of = |l: f32| -> u32 {
        if l.is_nan() { return 1 + logits.iter().filter(|x| !x.is_nan()).count() as u32; }
        1 + logits.iter().filter(|&&x| x > l).count() as u32
    };
    if n_inf > 0 {
        let k = n_inf as f32;
        let (surprisal, prob) = match chosen_logit {
            Some(l) if l == f32::INFINITY => (Some(k.ln()), Some(1.0 / k)),
            Some(_) => (Some(MAX_SURPRISAL), Some(0.0)),
            None => (None, None),
        };
        let top2 = if n_inf >= 2 { 1.0 / k } else { 0.0 };
        return Some(StepStats { entropy: k.ln(), top1_prob: 1.0 / k, top2_prob: top2, margin: 1.0 / k - top2,
            varentropy: 0.0, eff_vocab: k, rank: chosen_logit.map(rank_of), surprisal, chosen_prob: prob });
    }
    if m1 == f32::NEG_INFINITY { return None; }
    let (mut a, mut b, mut c) = (0f64, 0f64, 0f64);
    for &l in logits {
        if l.is_nan() || l == f32::NEG_INFINITY { continue; }
        let d = (l - m1) as f64;
        let e = d.exp();
        a += e; b += e * d; c += e * d * d;
    }
    let ln_z = a.ln();
    let mean_d = b / a;
    let ent = (ln_z - mean_d).max(0.0);
    let e_s2 = ln_z * ln_z - 2.0 * ln_z * mean_d + c / a;
    let var = (e_s2 - ent * ent).max(0.0);
    let top1 = 1.0 / a;
    let top2 = if m2.is_finite() { ((m2 - m1) as f64).exp() / a } else { 0.0 };
    let (rank, surprisal, prob) = match chosen_logit {
        Some(l) if l.is_nan() || l == f32::NEG_INFINITY => (Some(rank_of(l)), Some(MAX_SURPRISAL), Some(0.0)),
        Some(l) => {
            let s = (ln_z - (l - m1) as f64).clamp(0.0, MAX_SURPRISAL as f64);
            (Some(rank_of(l)), Some(s as f32), Some((-s).exp() as f32))
        }
        None => (None, None, None),
    };
    Some(StepStats { entropy: ent as f32, top1_prob: top1 as f32, top2_prob: top2 as f32,
        margin: (top1 - top2) as f32, varentropy: var as f32, eff_vocab: ent.exp() as f32,
        rank, surprisal, chosen_prob: prob })
}

/// Running aggregate with O(1) memory, for callers that do not need the whole trace.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RunningAgg {
    pub count: usize,
    pub mean_entropy: f32,
    pub max_entropy: f32,
    pub min_conf: f32,
    pub mean_surprisal: f32,
    surprisal_n: usize,
}
impl RunningAgg {
    pub fn add(&mut self, s: &StepStats) {
        self.count += 1;
        let n = self.count as f32;
        self.mean_entropy += (s.entropy - self.mean_entropy) / n;
        self.max_entropy = self.max_entropy.max(s.entropy);
        self.min_conf = if self.count == 1 { s.confidence() } else { self.min_conf.min(s.confidence()) };
        if let Some(x) = s.surprisal {
            self.surprisal_n += 1;
            self.mean_surprisal += (x - self.mean_surprisal) / self.surprisal_n as f32;
        }
    }
}

/// Consumes logits one step at a time. Statistics are computed without allocating; the only
/// growth is the amortised `Vec` of results (reserve with [`Streaming::with_capacity`]).
#[derive(Debug, Default)]
pub struct Streaming { steps: Vec<StepStats>, agg: RunningAgg }
impl Streaming {
    pub fn new() -> Self { Self::default() }
    pub fn with_capacity(n: usize) -> Self { Streaming { steps: Vec::with_capacity(n), agg: RunningAgg::default() } }
    /// Record one step. `Err(Empty)` for a row with no usable probability (nothing is recorded).
    pub fn push(&mut self, logits: &[f32], chosen: Option<usize>) -> Result<StepStats, ConfError> {
        let s = step_stats(logits, chosen).ok_or(ConfError::Empty)?;
        self.agg.add(&s);
        self.steps.push(s);
        Ok(s)
    }
    pub fn steps(&self) -> &[StepStats] { &self.steps }
    pub fn aggregate(&self) -> RunningAgg { self.agg }
    pub fn clear(&mut self) { self.steps.clear(); self.agg = RunningAgg::default(); }
    pub fn into_steps(self) -> Vec<StepStats> { self.steps }
}

// --------------------------------------------------------------------- text structure

/// Byte spans of the whitespace-separated words of `text`.
pub fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let (mut out, mut start) = (vec![], None);
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() { out.push((s, i)); }
        } else if start.is_none() { start = Some(i); }
    }
    if let Some(s) = start { out.push((s, text.len())); }
    out
}

fn abbreviation_before(text: &str, pos: usize) -> bool {
    const ABBR: [&str; 14] = ["mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "fig", "no", "nr", "ing"];
    let before = &text[..pos];
    let word: String = before.chars().rev().take_while(|c| c.is_alphabetic()).collect::<Vec<_>>().into_iter().rev().collect();
    if word.chars().count() == 1 && word.chars().all(char::is_uppercase) { return true; }
    if ABBR.contains(&word.to_lowercase().as_str()) { return true; }
    let digits: String = before.chars().rev().take_while(char::is_ascii_digit).collect();
    if !digits.is_empty() && digits.len() <= 3 {
        let rest = before[..before.len() - digits.len()].trim_end_matches([' ', '\t']);
        return rest.is_empty() || rest.ends_with('\n');
    }
    false
}

/// Contiguous byte spans of sentences (each includes its trailing whitespace); together they
/// cover the whole text. Splits on `. ! ? … 。！？` and newlines, but not inside `3.14`, after
/// one-letter initials, common abbreviations or list numbers.
pub fn sentence_spans(text: &str) -> Vec<(usize, usize)> {
    let n = text.len();
    let mut out = vec![];
    if n == 0 { return out; }
    let ch: Vec<(usize, char)> = text.char_indices().collect();
    let (mut start, mut i) = (0usize, 0usize);
    while i < ch.len() {
        let (pos, c) = ch[i];
        if !matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？' | '\n') { i += 1; continue; }
        let mut j = i + 1;
        while j < ch.len() && matches!(ch[j].1, '.' | '!' | '?' | '…') { j += 1; }
        let mut k = j;
        while k < ch.len() && matches!(ch[k].1, '"' | '\'' | ')' | ']' | '»' | '”' | '’') { k += 1; }
        let cjk = matches!(c, '。' | '！' | '？');
        let next_break = k >= ch.len() || ch[k].1.is_whitespace() || cjk;
        let boundary = c == '\n' || (next_break && !(c == '.' && j == i + 1 && abbreviation_before(text, pos)));
        if boundary {
            let mut e = k;
            while e < ch.len() && ch[e].1.is_whitespace() { e += 1; }
            let end = if e < ch.len() { ch[e].0 } else { n };
            out.push((start, end));
            start = end;
            i = e;
        } else { i = k.max(i + 1); }
    }
    if start < n { out.push((start, n)); }
    out
}

fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() { return 0.0; }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f32;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f32)
}

// ------------------------------------------------------------------------------ trace

/// Byte range of one generated token inside the answer text (end exclusive).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenSpan { pub start: usize, pub end: usize }

#[derive(Clone, Debug, PartialEq)]
pub struct TokenView {
    pub start: usize, pub end: usize, pub stats: StepStats,
    pub conf: f32,
    /// `1 - conf`: 0 = certain, 1 = hopeless. Use it for the heat-map colour.
    pub heat: f32,
    pub word: Option<usize>, pub sentence: Option<usize>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct WordView {
    pub start: usize, pub end: usize, pub first_token: usize, pub n_tokens: usize,
    pub min_conf: f32, pub mean_entropy: f32, pub max_entropy: f32, pub surprisal_sum: f32, pub heat: f32,
}
#[derive(Clone, Debug, PartialEq)]
pub struct SentenceView {
    pub start: usize, pub end: usize, pub n_tokens: usize,
    pub geo_mean_conf: f32, pub min_conf: f32, pub mean_entropy: f32, pub heat: f32,
}
#[derive(Clone, Debug, PartialEq)]
pub struct AnswerSummary {
    pub n_tokens: usize, pub min_conf: f32, pub mean_conf: f32, pub p10_conf: f32,
    pub geo_mean_conf: f32, pub mean_entropy: f32, pub max_entropy: f32,
    pub perplexity: Option<f32>,
    /// Index of the sentence with the lowest geometric-mean confidence (first on ties).
    pub weakest_sentence: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UncertaintyTrace {
    pub text: String,
    pub tokens: Vec<TokenView>,
    pub words: Vec<WordView>,
    pub sentences: Vec<SentenceView>,
    pub summary: AnswerSummary,
}

fn geo_mean(confs: impl Iterator<Item = f32>) -> f32 {
    let (mut s, mut n) = (0f64, 0usize);
    for c in confs { s += (c.max(1e-12) as f64).ln(); n += 1; }
    if n == 0 { 0.0 } else { (s / n as f64).exp() as f32 }
}

impl UncertaintyTrace {
    /// Join generated-token statistics with their byte spans in `text`.
    pub fn build(text: &str, spans: &[TokenSpan], stats: &[StepStats]) -> Result<UncertaintyTrace, ConfError> {
        if spans.len() != stats.len() { return Err(ConfError::LengthMismatch { a: spans.len(), b: stats.len() }); }
        if spans.is_empty() { return Err(ConfError::Empty); }
        for (i, s) in spans.iter().enumerate() {
            if s.start > s.end || s.end > text.len() || !text.is_char_boundary(s.start) || !text.is_char_boundary(s.end) {
                return Err(ConfError::BadSpan { index: i });
            }
        }
        let wspans = word_spans(text);
        let sspans = sentence_spans(text);
        let mut tokens: Vec<TokenView> = Vec::with_capacity(spans.len());
        for (sp, st) in spans.iter().zip(stats) {
            let anchor = text[sp.start..sp.end].char_indices().find(|(_, c)| !c.is_whitespace()).map(|(o, _)| sp.start + o);
            let word = anchor.and_then(|a| {
                let i = wspans.partition_point(|w| w.0 <= a).checked_sub(1)?;
                (a < wspans[i].1).then_some(i)
            });
            let pos = anchor.unwrap_or(sp.start);
            let sentence = sspans.partition_point(|s| s.0 <= pos).checked_sub(1);
            let conf = st.confidence().clamp(0.0, 1.0);
            tokens.push(TokenView { start: sp.start, end: sp.end, stats: *st, conf, heat: 1.0 - conf, word, sentence });
        }
        let mut words: Vec<WordView> = vec![];
        for (wi, w) in wspans.iter().enumerate() {
            let idx: Vec<usize> = (0..tokens.len()).filter(|&t| tokens[t].word == Some(wi)).collect();
            if idx.is_empty() { words.push(WordView { start: w.0, end: w.1, first_token: usize::MAX, n_tokens: 0, min_conf: 1.0, mean_entropy: 0.0, max_entropy: 0.0, surprisal_sum: 0.0, heat: 0.0 }); continue; }
            let min_conf = idx.iter().map(|&t| tokens[t].conf).fold(1.0, f32::min);
            let ents: Vec<f32> = idx.iter().map(|&t| tokens[t].stats.entropy).collect();
            let ssum: f32 = idx.iter().map(|&t| tokens[t].stats.surprisal.unwrap_or(-tokens[t].conf.max(1e-12).ln())).sum();
            words.push(WordView { start: w.0, end: w.1, first_token: idx[0], n_tokens: idx.len(), min_conf,
                mean_entropy: ents.iter().sum::<f32>() / ents.len() as f32, max_entropy: ents.iter().copied().fold(0.0, f32::max),
                surprisal_sum: ssum, heat: 1.0 - min_conf });
        }
        let mut sentences: Vec<SentenceView> = vec![];
        for (si, s) in sspans.iter().enumerate() {
            let idx: Vec<usize> = (0..tokens.len()).filter(|&t| tokens[t].sentence == Some(si)).collect();
            let confs: Vec<f32> = idx.iter().map(|&t| tokens[t].conf).collect();
            let gm = geo_mean(confs.iter().copied());
            let ents: f32 = idx.iter().map(|&t| tokens[t].stats.entropy).sum();
            sentences.push(SentenceView { start: s.0, end: s.1, n_tokens: idx.len(), geo_mean_conf: if idx.is_empty() { 1.0 } else { gm },
                min_conf: confs.iter().copied().fold(1.0, f32::min),
                mean_entropy: if idx.is_empty() { 0.0 } else { ents / idx.len() as f32 },
                heat: if idx.is_empty() { 0.0 } else { 1.0 - gm } });
        }
        let mut sorted: Vec<f32> = tokens.iter().map(|t| t.conf).collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n = tokens.len() as f32;
        let surps: Vec<f32> = tokens.iter().filter_map(|t| t.stats.surprisal).collect();
        let mut weakest: Option<usize> = None;
        for (i, s) in sentences.iter().enumerate() {
            if s.n_tokens > 0 && weakest.is_none_or(|w| s.geo_mean_conf < sentences[w].geo_mean_conf) { weakest = Some(i); }
        }
        let summary = AnswerSummary {
            n_tokens: tokens.len(), min_conf: sorted[0], mean_conf: sorted.iter().sum::<f32>() / n,
            p10_conf: percentile(&sorted, 0.10), geo_mean_conf: geo_mean(sorted.iter().copied()),
            mean_entropy: tokens.iter().map(|t| t.stats.entropy).sum::<f32>() / n,
            max_entropy: tokens.iter().map(|t| t.stats.entropy).fold(0.0, f32::max),
            perplexity: (!surps.is_empty()).then(|| (surps.iter().map(|&x| x as f64).sum::<f64>() / surps.len() as f64).exp().min(1e30) as f32),
            weakest_sentence: weakest,
        };
        Ok(UncertaintyTrace { text: text.to_string(), tokens, words, sentences, summary })
    }

    pub fn to_json(&self) -> Value {
        let tokens: Vec<Value> = self.tokens.iter().map(|t| json!({"start": t.start, "end": t.end, "conf": fin(t.conf), "heat": fin(t.heat),
            "word": t.word, "sentence": t.sentence, "stats": t.stats.to_json()})).collect();
        let words: Vec<Value> = self.words.iter().map(|w| json!({"start": w.start, "end": w.end, "tokens": w.n_tokens,
            "min_conf": fin(w.min_conf), "mean_entropy": fin(w.mean_entropy), "max_entropy": fin(w.max_entropy),
            "surprisal": fin(w.surprisal_sum), "heat": fin(w.heat)})).collect();
        let sents: Vec<Value> = self.sentences.iter().map(|s| json!({"start": s.start, "end": s.end, "tokens": s.n_tokens,
            "geo_mean_conf": fin(s.geo_mean_conf), "min_conf": fin(s.min_conf), "mean_entropy": fin(s.mean_entropy), "heat": fin(s.heat)})).collect();
        let m = &self.summary;
        json!({"text": self.text, "tokens": tokens, "words": words, "sentences": sents,
            "summary": {"n_tokens": m.n_tokens, "min_conf": fin(m.min_conf), "mean_conf": fin(m.mean_conf), "p10_conf": fin(m.p10_conf),
                "geo_mean_conf": fin(m.geo_mean_conf), "mean_entropy": fin(m.mean_entropy), "max_entropy": fin(m.max_entropy),
                "perplexity": opt(m.perplexity), "weakest_sentence": m.weakest_sentence}})
    }
    pub fn to_json_string(&self) -> String { self.to_json().to_string() }
}

// ----------------------------------------------------------------- hallucination risk

/// Thresholds and weights of the risk heuristic. All fields are plain numbers a caller may tune.
#[derive(Clone, Debug, PartialEq)]
pub struct RiskOptions {
    /// A word / token below this confidence counts as "unsure".
    pub low_conf: f32,
    /// Entropy (nats) above which a digit continuing a number counts as high.
    pub digit_entropy: f32,
    /// Entropy jump (nats) over a copied run's mean that counts as a spike.
    pub spike_delta: f32,
    pub spike_min_entropy: f32,
    /// Consecutive context-supported tokens that form a "copy run".
    pub copy_run: usize,
    /// Word n-gram size for the context-containment test.
    pub ngram: usize,
    pub w_entity: f32, pub w_digit: f32, pub w_spike: f32, pub w_unsupported: f32, pub w_overall: f32,
}
impl Default for RiskOptions {
    fn default() -> Self {
        RiskOptions { low_conf: 0.5, digit_entropy: 1.0, spike_delta: 1.5, spike_min_entropy: 1.0, copy_run: 4, ngram: 3,
            w_entity: 0.6, w_digit: 0.4, w_spike: 0.3, w_unsupported: 0.5, w_overall: 0.4 }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Reason { pub code: &'static str, pub message: String, pub weight: f32, pub start: usize, pub end: usize }
/// Each feature is a fraction in 0..=1.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Features {
    pub entity_low_conf: f32, pub digit_entropy: f32, pub spike_after_copy: f32,
    pub unsupported_entity: f32, pub overall_low_conf: f32,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Risk { pub score: f32, pub features: Features, pub reasons: Vec<Reason> }
impl Risk {
    pub fn to_json(&self) -> Value {
        let r: Vec<Value> = self.reasons.iter().map(|r| json!({"code": r.code, "message": r.message, "weight": fin(r.weight), "start": r.start, "end": r.end})).collect();
        let f = &self.features;
        json!({"score": fin(self.score), "reasons": r, "features": {"entity_low_conf": fin(f.entity_low_conf),
            "digit_entropy": fin(f.digit_entropy), "spike_after_copy": fin(f.spike_after_copy),
            "unsupported_entity": fin(f.unsupported_entity), "overall_low_conf": fin(f.overall_low_conf)}})
    }
}

/// Lower-case, keep letters and digits only (so `1,234` and `1234` compare equal).
fn norm_word(w: &str) -> String { w.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect() }
fn norm_words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.chars().flat_map(char::to_lowercase).collect()).collect()
}
fn fnv(words: &[String]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words { for b in w.bytes().chain([0x1f]) { h ^= b as u64; h = h.wrapping_mul(0x100_0000_01b3); } }
    h
}
fn excerpt(s: &str) -> String { let t: String = s.chars().take(40).collect(); if t.len() < s.len() { format!("{t}…") } else { t } }

#[derive(PartialEq, Clone, Copy)]
enum EntityKind { Number, Acronym, Capitalised }
fn entity_kind(word: &str, first_in_sentence: bool) -> Option<EntityKind> {
    let w = word.trim_matches(|c: char| !c.is_alphanumeric());
    let n = w.chars().count();
    if n == 0 { return None; }
    if w.chars().any(char::is_numeric) { return Some(EntityKind::Number); }
    if n >= 2 && w.chars().all(|c| c.is_alphabetic() && c.is_uppercase()) { return Some(EntityKind::Acronym); }
    if n >= 2 && !first_in_sentence && w.chars().next().is_some_and(|c| c.is_alphabetic() && c.is_uppercase()) { return Some(EntityKind::Capitalised); }
    None
}

/// Heuristic hallucination-risk assessment (see the module docs: not ground truth).
/// `context` is the retrieved/source text the answer should rest on, if any.
pub fn assess(trace: &UncertaintyTrace, context: Option<&str>, o: &RiskOptions) -> Risk {
    let text = &trace.text;
    let mut reasons: Vec<Reason> = vec![];
    // first word of each sentence
    let first_words: HashSet<usize> = trace.sentences.iter().filter_map(|s| trace.words.iter().position(|w| w.start >= s.start && w.start < s.end)).collect();
    // 1 + 4: entity-like words
    let ctx_words: Option<HashSet<String>> = context.filter(|c| !c.trim().is_empty()).map(|c| norm_words(c).into_iter().map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect()).collect());
    let (mut ent_total, mut ent_low, mut ent_unsup) = (0usize, vec![], vec![]);
    for (wi, w) in trace.words.iter().enumerate() {
        if w.n_tokens == 0 { continue; }
        let raw = &text[w.start..w.end];
        if entity_kind(raw, first_words.contains(&wi)).is_none() { continue; }
        ent_total += 1;
        if w.min_conf < o.low_conf { ent_low.push(wi); }
        if let Some(cw) = &ctx_words {
            let n = norm_word(raw);
            if !n.is_empty() && !cw.contains(&n) { ent_unsup.push(wi); }
        }
    }
    let f_entity = if ent_total == 0 { 0.0 } else { ent_low.len() as f32 / ent_total as f32 };
    let f_unsup = if ctx_words.is_none() || ent_total == 0 { 0.0 } else { ent_unsup.len() as f32 / ent_total as f32 };
    let list = |idx: &[usize]| idx.iter().take(3).map(|&i| format!("«{}»", excerpt(text[trace.words[i].start..trace.words[i].end].trim_matches(|c: char| !c.is_alphanumeric())))).collect::<Vec<_>>().join(", ");
    if !ent_low.is_empty() {
        let w = &trace.words[ent_low[0]];
        reasons.push(Reason { code: "low_conf_entity", message: format!("low confidence on names/numbers: {}", list(&ent_low)), weight: o.w_entity * f_entity, start: w.start, end: w.end });
    }
    if !ent_unsup.is_empty() {
        let w = &trace.words[ent_unsup[0]];
        reasons.push(Reason { code: "unsupported_entity", message: format!("names/numbers not found in the source text: {}", list(&ent_unsup)), weight: o.w_unsupported * f_unsup, start: w.start, end: w.end });
    }
    // 2: high entropy on digits continuing a number
    let (mut dig_cand, mut dig_high, mut dig_first) = (0usize, 0usize, None);
    for t in 1..trace.tokens.len() {
        let (p, c) = (&trace.tokens[t - 1], &trace.tokens[t]);
        let prev_digit = text[p.start..p.end].chars().next_back().is_some_and(|ch| ch.is_ascii_digit());
        let cur_digit = text[c.start..c.end].chars().next().is_some_and(|ch| ch.is_ascii_digit());
        if prev_digit && cur_digit {
            dig_cand += 1;
            if c.stats.entropy >= o.digit_entropy { dig_high += 1; dig_first.get_or_insert((c.start, c.end)); }
        }
    }
    let f_digit = if dig_cand == 0 { 0.0 } else { dig_high as f32 / dig_cand as f32 };
    if let Some((s, e)) = dig_first {
        reasons.push(Reason { code: "digit_entropy", message: format!("{dig_high} of {dig_cand} digits inside numbers were uncertain"), weight: o.w_digit * f_digit, start: s, end: e });
    }
    // 3: entropy spike right after a run copied from the context
    let mut f_spike = 0.0;
    if let Some(ctx) = context.filter(|c| !c.trim().is_empty()) {
        let n_eff = o.ngram.max(1).min(norm_words(ctx).len()).min(trace.words.len().max(1));
        let cwords = norm_words(ctx);
        let set: HashSet<u64> = cwords.windows(n_eff.max(1)).map(fnv).collect();
        let nw: Vec<String> = trace.words.iter().map(|w| norm_word(&text[w.start..w.end])).collect();
        let mut sup = vec![false; nw.len()];
        if nw.len() >= n_eff && n_eff > 0 {
            for s in 0..=nw.len() - n_eff {
                if set.contains(&fnv(&nw[s..s + n_eff])) { for x in sup.iter_mut().skip(s).take(n_eff) { *x = true; } }
            }
        }
        let (mut run, mut run_ent, mut runs, mut spikes, mut spike_at) = (0usize, 0f32, 0usize, 0usize, None);
        let mut prev_sup = false;
        for tv in &trace.tokens {
            let is_sup = tv.word.map_or(prev_sup, |w| sup[w]);
            if is_sup { run += 1; run_ent += tv.stats.entropy; }
            else {
                if run >= o.copy_run.max(1) {
                    runs += 1;
                    if tv.stats.entropy >= o.spike_min_entropy && tv.stats.entropy >= run_ent / run as f32 + o.spike_delta { spikes += 1; spike_at.get_or_insert((tv.start, tv.end)); }
                }
                run = 0; run_ent = 0.0;
            }
            prev_sup = is_sup;
        }
        f_spike = if runs == 0 { 0.0 } else { spikes as f32 / runs as f32 };
        if let Some((s, e)) = spike_at {
            reasons.push(Reason { code: "spike_after_copy", message: format!("{spikes} of {runs} passages copied from the source ended in a sudden jump in uncertainty"), weight: o.w_spike * f_spike, start: s, end: e });
        }
    }
    // 5: overall
    let low = trace.tokens.iter().filter(|t| t.conf < o.low_conf).count();
    let f_overall = low as f32 / trace.tokens.len().max(1) as f32;
    if f_overall > 0.2 {
        let w = trace.tokens.iter().find(|t| t.conf < o.low_conf).map_or((0, 0), |t| (t.start, t.end));
        reasons.push(Reason { code: "low_conf_overall", message: format!("{low} of {} tokens were generated with low confidence", trace.tokens.len()), weight: o.w_overall * f_overall, start: w.0, end: w.1 });
    }
    let feats = Features { entity_low_conf: f_entity, digit_entropy: f_digit, spike_after_copy: f_spike, unsupported_entity: f_unsup, overall_low_conf: f_overall };
    let mut keep = 1.0f32;
    for (w, f) in [(o.w_entity, f_entity), (o.w_digit, f_digit), (o.w_spike, f_spike), (o.w_unsupported, f_unsup), (o.w_overall, f_overall)] {
        keep *= 1.0 - (w * f).clamp(0.0, 1.0);
    }
    reasons.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    Risk { score: (1.0 - keep).clamp(0.0, 1.0), features: feats, reasons }
}

// ------------------------------------------------------------------ self-consistency

/// Bag-of-words F1 between two texts (case- and punctuation-insensitive).
pub fn token_f1(a: &str, b: &str) -> f32 {
    let (wa, wb) = (norm_words(a), norm_words(b));
    if wa.is_empty() && wb.is_empty() { return 1.0; }
    if wa.is_empty() || wb.is_empty() { return 0.0; }
    let mut counts: HashMap<&str, i32> = HashMap::new();
    for w in &wa { *counts.entry(w).or_default() += 1; }
    let mut common = 0usize;
    for w in &wb { if let Some(c) = counts.get_mut(w.as_str()).filter(|c| **c > 0) { *c -= 1; common += 1; } }
    if common == 0 { return 0.0; }
    let (p, r) = (common as f32 / wb.len() as f32, common as f32 / wa.len() as f32);
    2.0 * p * r / (p + r)
}

/// ROUGE-L F1 (longest common subsequence of words; each side capped at 2000 words).
pub fn rouge_l(a: &str, b: &str) -> f32 {
    let (mut wa, mut wb) = (norm_words(a), norm_words(b));
    wa.truncate(MAX_LCS_WORDS); wb.truncate(MAX_LCS_WORDS);
    if wa.is_empty() && wb.is_empty() { return 1.0; }
    if wa.is_empty() || wb.is_empty() { return 0.0; }
    let mut prev = vec![0usize; wb.len() + 1];
    for x in &wa {
        let mut cur = vec![0usize; wb.len() + 1];
        for (j, y) in wb.iter().enumerate() { cur[j + 1] = if x == y { prev[j] + 1 } else { cur[j].max(prev[j + 1]) }; }
        prev = cur;
    }
    let l = prev[wb.len()] as f32;
    if l == 0.0 { return 0.0; }
    let (p, r) = (l / wb.len() as f32, l / wa.len() as f32);
    2.0 * p * r / (p + r)
}

/// Canonical form of a number written with `,` / `.` / spaces (`1,234.50`, `1.234,5`, `12 345`).
/// Returns `None` when the text is not a plain number.
pub fn normalize_number(s: &str) -> Option<String> {
    let t: String = s.trim().trim_matches(|c: char| matches!(c, '$' | '€' | '£' | '%' | '*' | '"' | '\'' | '.' | ',' | ':' | ';' | '!' | '?')).chars()
        .filter(|c| !matches!(c, ' ' | '\u{a0}' | '\u{202f}' | '\'' | '_')).collect();
    if t.is_empty() { return None; }
    let (neg, body) = match t.strip_prefix('-') { Some(b) => (true, b), None => (false, t.strip_prefix('+').unwrap_or(&t)) };
    if body.is_empty() || !body.chars().all(|c| c.is_ascii_digit() || c == '.' || c == ',') || !body.chars().any(|c| c.is_ascii_digit()) { return None; }
    let (dots, commas) = (body.matches('.').count(), body.matches(',').count());
    let dec: Option<char> = if dots > 0 && commas > 0 { Some(if body.rfind('.') > body.rfind(',') { '.' } else { ',' }) }
        else {
            let sep = if dots > 0 { '.' } else if commas > 0 { ',' } else { ' ' };
            let n = dots + commas;
            if n != 1 { None } else {
                let i = body.find(sep).unwrap();
                let (before, after) = (&body[..i], &body[i + 1..]);
                if after.len() == 3 && (1..=3).contains(&before.len()) && !before.starts_with('0') { None } else { Some(sep) }
            }
        };
    let mut int = String::new();
    let mut frac = String::new();
    let mut seen_dec = false;
    for c in body.chars() {
        if c.is_ascii_digit() { if seen_dec { frac.push(c) } else { int.push(c) } }
        else if Some(c) == dec { if seen_dec { return None; } seen_dec = true; }
    }
    let int = int.trim_start_matches('0');
    let int = if int.is_empty() { "0" } else { int };
    let frac = frac.trim_end_matches('0');
    let mut out = String::new();
    let zero = int == "0" && frac.is_empty();
    if neg && !zero { out.push('-'); }
    out.push_str(int);
    if !frac.is_empty() { out.push('.'); out.push_str(frac); }
    Some(out)
}

const MARKERS: [&str; 18] = ["the answer is", "final answer:", "answer:", "răspunsul este", "răspuns:", "la réponse est", "réponse :", "réponse:",
    "la risposta è", "risposta:", "la respuesta es", "respuesta:", "a resposta é", "resposta:", "die antwort ist", "antwort:", "a válasz:", "válasz:"];

/// Reduce a free-text answer to a comparable key: text after an "answer is/:" marker (8 languages),
/// else the whole text when short, else the last number; numbers are canonicalised.
pub fn extract_answer(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut cand: Option<String> = None;
    let best = MARKERS.iter().filter_map(|m| lower.rfind(m).map(|p| (p, m.len()))).max();
    if let Some((p, l)) = best {
        let rest = &lower[p + l..];
        let end = rest.char_indices().find(|&(i, c)| c == '\n' || (c == '.' && rest[i + 1..].chars().next().is_none_or(char::is_whitespace))).map_or(rest.len(), |(i, _)| i);
        let r = rest[..end].trim();
        if !r.is_empty() { cand = Some(r.to_string()); }
    }
    let cand = cand.unwrap_or_else(|| {
        let words = norm_words(&lower);
        let last_num = lower.split_whitespace().rev().find(|w| normalize_number(w).is_some()).map(str::to_string);
        if words.len() <= 6 && last_num.is_none() { return lower.clone(); }
        last_num.unwrap_or_else(|| lower.clone())
    });
    let t = cand.trim();
    if let Some(n) = normalize_number(t) { return n; }
    norm_words(t).join(" ")
}

#[derive(Clone, Debug, PartialEq)]
pub struct Majority {
    pub answer: String, pub votes: usize, pub k: usize,
    /// `votes / k`.
    pub confidence: f32,
    /// Votes ahead of the runner-up.
    pub lead: usize,
    /// 95 % Wilson lower bound of the winner's share.
    pub wilson_lower: f32,
    /// Index of the first sample that gave the winning answer.
    pub representative: usize,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Consistency {
    /// Mean pairwise token-F1 (`None` when fewer than two samples).
    pub agreement: Option<f32>,
    pub rouge_l: Option<f32>,
    pub majority: Option<Majority>,
    /// Mean F1 of each sample against the others: low values flag outliers.
    pub per_sample_support: Vec<f32>,
    pub keys: Vec<String>,
}

pub fn wilson_lower(successes: usize, n: usize) -> f32 {
    if n == 0 { return 0.0; }
    let (p, nf, z) = (successes as f64 / n as f64, n as f64, 1.645f64);
    let z2 = z * z;
    ((p + z2 / (2.0 * nf) - z * (p * (1.0 - p) / nf + z2 / (4.0 * nf * nf)).sqrt()) / (1.0 + z2 / nf)).clamp(0.0, 1.0) as f32
}

fn tally(keys: &[String]) -> Option<Majority> {
    let mut order: Vec<(&str, usize, usize)> = vec![]; // key, votes, first index
    for (i, k) in keys.iter().enumerate() {
        match order.iter_mut().find(|(x, _, _)| *x == k) { Some(e) => e.1 += 1, None => order.push((k, 1, i)) }
    }
    let mut best = *order.first()?;
    for e in &order { if e.1 > best.1 { best = *e; } }
    let second = order.iter().filter(|e| e.0 != best.0).map(|e| e.1).max().unwrap_or(0);
    Some(Majority { answer: best.0.to_string(), votes: best.1, k: keys.len(), confidence: best.1 as f32 / keys.len() as f32,
        lead: best.1 - second, wilson_lower: wilson_lower(best.1, keys.len()), representative: best.2 })
}

/// Agreement statistics over `k` sampled answers (at most [`MAX_SAMPLES`]).
pub fn consistency(answers: &[&str]) -> Result<Consistency, ConfError> {
    if answers.is_empty() { return Err(ConfError::Empty); }
    if answers.len() > MAX_SAMPLES { return Err(ConfError::BadParam("too many samples")); }
    let k = answers.len();
    let keys: Vec<String> = answers.iter().map(|a| extract_answer(a)).collect();
    let (mut f1s, mut rls, mut pairs) = (0f32, 0f32, 0usize);
    let mut support = vec![0f32; k];
    for i in 0..k { for j in i + 1..k {
        let f = token_f1(answers[i], answers[j]);
        f1s += f; rls += rouge_l(answers[i], answers[j]); pairs += 1;
        support[i] += f; support[j] += f;
    } }
    if k > 1 { for s in &mut support { *s /= (k - 1) as f32; } } else { support[0] = 1.0; }
    Ok(Consistency { agreement: (pairs > 0).then(|| f1s / pairs as f32), rouge_l: (pairs > 0).then(|| rls / pairs as f32),
        majority: tally(&keys), per_sample_support: support, keys })
}

/// Adaptive-k rule: stop sampling once more samples cannot change the winner, or the winner's
/// Wilson lower bound clears `stop_conf` after `min_k` samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EarlyStop { pub min_k: usize, pub max_k: usize, pub stop_conf: f32 }
impl Default for EarlyStop { fn default() -> Self { EarlyStop { min_k: 3, max_k: 10, stop_conf: 0.5 } } }
impl EarlyStop {
    /// `keys` are the [`extract_answer`] keys gathered so far.
    pub fn should_stop(&self, keys: &[String]) -> bool {
        let k = keys.len();
        if k >= self.max_k { return true; }
        if k == 0 { return false; }
        let Some(m) = tally(keys) else { return false };
        if m.lead > self.max_k - k { return true; }
        k >= self.min_k && m.wilson_lower >= self.stop_conf
    }
}

// ----------------------------------------------------------------------- calibration

#[derive(Clone, Debug, PartialEq)]
pub struct Bin { pub lo: f32, pub hi: f32, pub count: usize, pub mean_score: f32, pub frac_positive: f32 }

fn check_pairs(scores: &[f32], labels: &[bool]) -> Result<(), ConfError> {
    if scores.len() != labels.len() { return Err(ConfError::LengthMismatch { a: scores.len(), b: labels.len() }); }
    if scores.is_empty() { return Err(ConfError::Empty); }
    if scores.iter().any(|s| !s.is_finite()) { return Err(ConfError::NonFinite); }
    Ok(())
}
fn check_unit(scores: &[f32]) -> Result<(), ConfError> {
    if scores.iter().any(|s| !(0.0..=1.0).contains(s)) { Err(ConfError::BadParam("scores must lie in 0..=1")) } else { Ok(()) }
}
/// Equal-width reliability-diagram bins over `[0, 1]`; empty bins are included with count 0.
pub fn reliability_bins(scores: &[f32], labels: &[bool], n_bins: usize) -> Result<Vec<Bin>, ConfError> {
    check_pairs(scores, labels)?; check_unit(scores)?;
    if !(1..=1000).contains(&n_bins) { return Err(ConfError::BadParam("n_bins")); }
    let mut acc = vec![(0usize, 0f64, 0usize); n_bins];
    for (&s, &l) in scores.iter().zip(labels) {
        let b = ((s * n_bins as f32) as usize).min(n_bins - 1);
        acc[b].0 += 1; acc[b].1 += s as f64; acc[b].2 += l as usize;
    }
    Ok(acc.iter().enumerate().map(|(i, &(c, ss, pos))| Bin { lo: i as f32 / n_bins as f32, hi: (i + 1) as f32 / n_bins as f32, count: c,
        mean_score: if c == 0 { 0.0 } else { (ss / c as f64) as f32 }, frac_positive: if c == 0 { 0.0 } else { pos as f32 / c as f32 } }).collect())
}
/// Expected calibration error: the count-weighted mean gap between confidence and accuracy.
pub fn ece(scores: &[f32], labels: &[bool], n_bins: usize) -> Result<f32, ConfError> {
    let bins = reliability_bins(scores, labels, n_bins)?;
    let n = scores.len() as f32;
    Ok(bins.iter().map(|b| b.count as f32 / n * (b.frac_positive - b.mean_score).abs()).sum())
}
/// Maximum calibration error over non-empty bins.
pub fn mce(scores: &[f32], labels: &[bool], n_bins: usize) -> Result<f32, ConfError> {
    Ok(reliability_bins(scores, labels, n_bins)?.iter().filter(|b| b.count > 0).map(|b| (b.frac_positive - b.mean_score).abs()).fold(0.0, f32::max))
}
pub fn brier(scores: &[f32], labels: &[bool]) -> Result<f32, ConfError> {
    check_pairs(scores, labels)?; check_unit(scores)?;
    Ok(scores.iter().zip(labels).map(|(&s, &l)| { let d = s as f64 - l as u8 as f64; d * d }).sum::<f64>() as f32 / scores.len() as f32)
}
/// Area under the ROC curve (probability a random positive scores above a random negative; ties count half).
pub fn auroc(scores: &[f32], labels: &[bool]) -> Result<f32, ConfError> {
    check_pairs(scores, labels)?;
    let pos = labels.iter().filter(|&&l| l).count();
    let neg = labels.len() - pos;
    if pos == 0 || neg == 0 { return Err(ConfError::NeedBothClasses); }
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let (mut rank_sum, mut i) = (0f64, 0usize);
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && scores[idx[j + 1]] == scores[idx[i]] { j += 1; }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        rank_sum += idx[i..=j].iter().filter(|&&k| labels[k]).count() as f64 * avg;
        i = j + 1;
    }
    Ok(((rank_sum - pos as f64 * (pos as f64 + 1.0) / 2.0) / (pos as f64 * neg as f64)) as f32)
}

/// Pool-adjacent-violators: weighted isotonic (non-decreasing) regression of `y`.
pub(crate) fn pav(y: &[f64], w: &[f64]) -> Vec<f64> {
    let mut blocks: Vec<(f64, f64, usize)> = vec![]; // mean, weight, length
    for (&yi, &wi) in y.iter().zip(w) {
        blocks.push((yi, wi, 1));
        while blocks.len() > 1 && blocks[blocks.len() - 2].0 > blocks[blocks.len() - 1].0 {
            let b = blocks.pop().unwrap();
            let a = blocks.last_mut().unwrap();
            let tw = a.1 + b.1;
            a.0 = (a.0 * a.1 + b.0 * b.1) / tw; a.1 = tw; a.2 += b.2;
        }
    }
    blocks.iter().flat_map(|b| std::iter::repeat_n(b.0, b.2)).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method { Isotonic, Platt }

/// Maps a raw risk/confidence score to a calibrated probability of the positive label.
#[derive(Clone, Debug, PartialEq)]
pub enum Calibrator {
    Isotonic { xs: Vec<f32>, ys: Vec<f32> },
    Platt { a: f32, b: f32 },
}
impl Calibrator {
    /// Fit on `(score, label)` pairs; `label = true` is the event being predicted.
    pub fn fit(scores: &[f32], labels: &[bool], method: Method) -> Result<Calibrator, ConfError> {
        check_pairs(scores, labels)?;
        let pos = labels.iter().filter(|&&l| l).count();
        if pos == 0 || pos == labels.len() { return Err(ConfError::NeedBothClasses); }
        match method {
            Method::Isotonic => {
                let mut idx: Vec<usize> = (0..scores.len()).collect();
                idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
                let (mut xs, mut ys, mut ws): (Vec<f64>, Vec<f64>, Vec<f64>) = (vec![], vec![], vec![]);
                for &i in &idx {
                    if xs.last().is_some_and(|&x| x == scores[i] as f64) {
                        let n = ws.len() - 1;
                        ys[n] = (ys[n] * ws[n] + labels[i] as u8 as f64) / (ws[n] + 1.0); ws[n] += 1.0;
                    } else { xs.push(scores[i] as f64); ys.push(labels[i] as u8 as f64); ws.push(1.0); }
                }
                let fitted = pav(&ys, &ws);
                // collapse runs of equal fitted value into one knot at the run's weighted mean x
                let (mut kx, mut ky): (Vec<f32>, Vec<f32>) = (vec![], vec![]);
                let mut i = 0;
                while i < fitted.len() {
                    let mut j = i;
                    while j + 1 < fitted.len() && fitted[j + 1] == fitted[i] { j += 1; }
                    let wsum: f64 = ws[i..=j].iter().sum();
                    kx.push((xs[i..=j].iter().zip(&ws[i..=j]).map(|(x, w)| x * w).sum::<f64>() / wsum) as f32);
                    ky.push(fitted[i] as f32);
                    i = j + 1;
                }
                Ok(Calibrator::Isotonic { xs: kx, ys: ky })
            }
            Method::Platt => {
                let (np, nn) = (pos as f64, (labels.len() - pos) as f64);
                let (tp, tn) = ((np + 1.0) / (np + 2.0), 1.0 / (nn + 2.0));
                let t: Vec<f64> = labels.iter().map(|&l| if l { tp } else { tn }).collect();
                let s: Vec<f64> = scores.iter().map(|&x| x as f64).collect();
                let loss = |a: f64, b: f64| -> f64 {
                    s.iter().zip(&t).map(|(&x, &ti)| { let z = a * x + b; z.max(0.0) + (-z.abs()).exp().ln_1p() - ti * z }).sum()
                };
                let (mut a, mut b) = (0f64, ((np + 1.0) / (nn + 1.0)).ln());
                for _ in 0..200 {
                    let (mut ga, mut gb, mut haa, mut hab, mut hbb) = (0f64, 0f64, 1e-9, 0f64, 1e-9);
                    for (&x, &ti) in s.iter().zip(&t) {
                        let p = 1.0 / (1.0 + (-(a * x + b)).exp());
                        let (d, v) = (p - ti, p * (1.0 - p));
                        ga += d * x; gb += d; haa += v * x * x; hab += v * x; hbb += v;
                    }
                    if ga.abs().max(gb.abs()) < 1e-10 { break; }
                    let det = haa * hbb - hab * hab;
                    let (da, db) = if det.abs() < 1e-12 { (ga / haa, gb / hbb) } else { ((hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det) };
                    let (l0, mut step) = (loss(a, b), 1.0);
                    while step > 1e-10 && loss(a - step * da, b - step * db) > l0 + 1e-12 { step *= 0.5; }
                    if step <= 1e-10 { break; }
                    a -= step * da; b -= step * db;
                }
                if !a.is_finite() || !b.is_finite() { return Err(ConfError::NonFinite); }
                Ok(Calibrator::Platt { a: a as f32, b: b as f32 })
            }
        }
    }
    /// Calibrated probability, or `None` for a non-finite input.
    pub fn predict(&self, x: f32) -> Option<f32> {
        if !x.is_finite() { return None; }
        Some(match self {
            Calibrator::Platt { a, b } => (1.0 / (1.0 + (-(*a as f64 * x as f64 + *b as f64)).exp())) as f32,
            Calibrator::Isotonic { xs, ys } => {
                let i = xs.partition_point(|&k| k <= x);
                if i == 0 { ys[0] } else if i >= xs.len() { ys[ys.len() - 1] } else {
                    let (x0, x1, y0, y1) = (xs[i - 1], xs[i], ys[i - 1], ys[i]);
                    if x1 > x0 { y0 + (y1 - y0) * (x - x0) / (x1 - x0) } else { y0 }
                }
            }
        })
    }
    pub fn to_json(&self) -> Value {
        match self {
            Calibrator::Isotonic { xs, ys } => json!({"method": "isotonic", "xs": xs.iter().map(|&v| fin(v)).collect::<Vec<_>>(), "ys": ys.iter().map(|&v| fin(v)).collect::<Vec<_>>()}),
            Calibrator::Platt { a, b } => json!({"method": "platt", "a": fin(*a), "b": fin(*b)}),
        }
    }
    pub fn from_json(v: &Value) -> Result<Calibrator, ConfError> {
        let o: &Map<String, Value> = v.as_object().ok_or_else(|| ConfError::Json("not an object".into()))?;
        let num = |k: &str| o.get(k).and_then(Value::as_f64).filter(|x| x.is_finite()).map(|x| x as f32).ok_or_else(|| ConfError::Json(format!("bad `{k}`")));
        let arr = |k: &str| -> Result<Vec<f32>, ConfError> {
            o.get(k).and_then(Value::as_array).ok_or_else(|| ConfError::Json(format!("bad `{k}`")))?.iter()
                .map(|x| x.as_f64().filter(|f| f.is_finite()).map(|f| f as f32).ok_or_else(|| ConfError::Json(format!("bad `{k}`")))).collect()
        };
        match o.get("method").and_then(Value::as_str) {
            Some("platt") => Ok(Calibrator::Platt { a: num("a")?, b: num("b")? }),
            Some("isotonic") => {
                let (xs, ys) = (arr("xs")?, arr("ys")?);
                if xs.is_empty() || xs.len() != ys.len() { return Err(ConfError::Json("xs/ys length".into())); }
                if xs.windows(2).any(|w| w[1] < w[0]) || ys.windows(2).any(|w| w[1] < w[0]) || ys.iter().any(|y| !(0.0..=1.0).contains(y)) {
                    return Err(ConfError::Json("knots must be increasing with probabilities in 0..=1".into()));
                }
                Ok(Calibrator::Isotonic { xs, ys })
            }
            _ => Err(ConfError::Json("unknown method".into())),
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn f(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 }
    }
    fn reference(logits: &[f32]) -> (f64, f64, f64) {
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let e: Vec<f64> = logits.iter().map(|&l| (l as f64 - m).exp()).collect();
        let z: f64 = e.iter().sum();
        let p: Vec<f64> = e.iter().map(|x| x / z).collect();
        let h: f64 = -p.iter().filter(|&&x| x > 0.0).map(|&x| x * x.ln()).sum::<f64>();
        let v: f64 = p.iter().filter(|&&x| x > 0.0).map(|&x| x * (x.ln() + h).powi(2)).sum();
        (h, v, p.iter().cloned().fold(0.0, f64::max))
    }

    #[test]
    fn uniform_peaked_and_tie_distributions() {
        let u = step_stats(&[0.0; 8], Some(3)).unwrap();
        assert!((u.entropy - 8f32.ln()).abs() < 1e-5 && (u.top1_prob - 0.125).abs() < 1e-6);
        assert!(u.margin.abs() < 1e-6 && u.varentropy.abs() < 1e-5 && (u.eff_vocab - 8.0).abs() < 1e-3);
        assert_eq!(u.rank, Some(1));
        assert!((u.surprisal.unwrap() - 8f32.ln()).abs() < 1e-5);
        let p = step_stats(&[30.0, 0.0, 0.0, 0.0], Some(0)).unwrap();
        assert!(p.entropy < 1e-6 && p.top1_prob > 0.999_999 && p.margin > 0.999_999 && p.eff_vocab < 1.001);
        let t = step_stats(&[2.0, 2.0, -50.0], Some(1)).unwrap();
        assert!((t.entropy - 2f32.ln()).abs() < 1e-5 && t.margin.abs() < 1e-6);
        assert!((t.top1_prob - 0.5).abs() < 1e-6 && (t.top2_prob - 0.5).abs() < 1e-6);
        // varentropy of a two-point {1/4, 3/4} distribution, by hand: Var of -ln p
        let q = step_stats(&[(0.25f32).ln(), (0.75f32).ln()], None).unwrap();
        let (a, b) = (-(0.25f64).ln(), -(0.75f64).ln());
        let mean = 0.25 * a + 0.75 * b;
        let want = 0.25 * (a - mean).powi(2) + 0.75 * (b - mean).powi(2);
        assert!((q.varentropy as f64 - want).abs() < 1e-5, "{} vs {want}", q.varentropy);
        assert_eq!((q.rank, q.surprisal, q.chosen_prob), (None, None, None));
    }

    #[test]
    fn rank_and_surprisal_of_the_chosen_token() {
        let l = [1.0, 3.0, 2.0, 0.0];
        let s = step_stats(&l, Some(2)).unwrap();
        assert_eq!(s.rank, Some(2));
        let z: f64 = l.iter().map(|&x| (x as f64).exp()).sum();
        assert!((s.chosen_prob.unwrap() as f64 - 2f64.exp() / z).abs() < 1e-6);
        assert_eq!(step_stats(&l, Some(99)).unwrap().rank, None);
        assert_eq!(step_stats(&l, Some(1)).unwrap().rank, Some(1));
    }

    #[test]
    fn entropy_is_bounded_and_matches_the_two_pass_reference() {
        let mut r = Rng(7);
        for n in [1usize, 2, 5, 100, 3000] {
            for scale in [0.1f32, 1.0, 20.0] {
                let l: Vec<f32> = (0..n).map(|_| (r.f() - 0.5) * scale).collect();
                let s = step_stats(&l, Some(n / 2)).unwrap();
                let (h, v, t1) = reference(&l);
                assert!(s.entropy >= 0.0 && s.entropy <= (n as f32).ln() + 1e-4);
                assert!((s.entropy as f64 - h).abs() < 1e-4, "n={n} {} vs {h}", s.entropy);
                assert!((s.varentropy as f64 - v).abs() < 2e-3 * (1.0 + v), "n={n} {} vs {v}", s.varentropy);
                assert!((s.top1_prob as f64 - t1).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn streaming_equals_batch() {
        let mut r = Rng(11);
        let mut st = Streaming::with_capacity(40);
        let mut want = vec![];
        for i in 0..40 {
            let l: Vec<f32> = (0..257).map(|_| (r.f() - 0.5) * 12.0).collect();
            st.push(&l, Some(i % 257)).unwrap();
            want.push(step_stats(&l, Some(i % 257)).unwrap());
        }
        assert_eq!(st.steps(), &want[..]);
        let mean: f32 = want.iter().map(|s| s.entropy).sum::<f32>() / 40.0;
        assert!((st.aggregate().mean_entropy - mean).abs() < 1e-4 && st.aggregate().count == 40);
        assert!(st.aggregate().min_conf <= want.iter().map(|s| s.confidence()).fold(1.0, f32::min) + 1e-6);
        assert_eq!(st.push(&[], None), Err(ConfError::Empty));
        assert_eq!(st.steps().len(), 40);
    }

    #[test]
    fn hostile_logits_never_panic() {
        assert!(step_stats(&[], None).is_none());
        assert!(step_stats(&[f32::NAN; 4], Some(0)).is_none());
        assert!(step_stats(&[f32::NEG_INFINITY; 3], None).is_none());
        let s = step_stats(&[f32::INFINITY, 1.0, f32::INFINITY], Some(0)).unwrap();
        assert!((s.entropy - 2f32.ln()).abs() < 1e-6 && s.margin.abs() < 1e-6 && s.rank == Some(1));
        let s = step_stats(&[f32::INFINITY, 1.0], Some(1)).unwrap();
        assert_eq!((s.top1_prob, s.chosen_prob, s.surprisal), (1.0, Some(0.0), Some(MAX_SURPRISAL)));
        let s = step_stats(&[f32::NAN, 1.0, f32::NEG_INFINITY], Some(0)).unwrap();
        assert!(s.top1_prob == 1.0 && s.surprisal == Some(MAX_SURPRISAL) && s.rank == Some(3));
        let s = step_stats(&[1e30, -1e30, 0.0], Some(0)).unwrap();
        assert!(s.entropy.is_finite() && s.top1_prob == 1.0);
        let s = step_stats(&[5.0], Some(0)).unwrap();
        assert!(s.entropy == 0.0 && s.top1_prob == 1.0 && s.top2_prob == 0.0);
    }

    #[test]
    fn splitters_handle_decimals_initials_lists_and_odd_input() {
        let t = "Pi is 3.14 today. J. Smith met Dr. Jones! Really?\n1. First item\nDone";
        let s: Vec<&str> = sentence_spans(t).iter().map(|&(a, b)| t[a..b].trim()).collect();
        assert_eq!(s, ["Pi is 3.14 today.", "J. Smith met Dr. Jones!", "Really?", "1. First item", "Done"]);
        for text in ["", "...", "!!!", "\n\n", "a", "日本語。次の文！", "x\u{a0}y. z", "\"Hi.\" He left."] {
            let sp = sentence_spans(text);
            if !text.is_empty() {
                assert_eq!(sp.first().unwrap().0, 0);
                assert_eq!(sp.last().unwrap().1, text.len());
                assert!(sp.windows(2).all(|w| w[0].1 == w[1].0));
            }
        }
        assert_eq!(sentence_spans("日本語。次の文！").len(), 2);
        assert_eq!(word_spans(" a  bé\tc ").len(), 3);
    }

    fn demo_trace(text: &str, conf_of: impl Fn(&str) -> f32, ent_of: impl Fn(&str) -> f32) -> UncertaintyTrace {
        let (mut spans, mut stats) = (vec![], vec![]);
        let mut at = 0;
        for w in text.split_inclusive(' ') {
            let core = w.trim();
            spans.push(TokenSpan { start: at, end: at + w.len() });
            let c = conf_of(core);
            let mut s = step_stats(&[0.0, 0.0], Some(0)).unwrap();
            s.chosen_prob = Some(c); s.surprisal = Some(-c.ln()); s.entropy = ent_of(core);
            stats.push(s);
            at += w.len();
        }
        UncertaintyTrace::build(text, &spans, &stats).unwrap()
    }

    #[test]
    fn trace_rolls_tokens_up_to_words_sentences_and_answer() {
        let text = "Paris is big. Zorblax founded it in 1432.";
        let tr = demo_trace(text, |w| if w.starts_with("Zorb") || w == "1432." { 0.2 } else { 0.9 }, |_| 0.1);
        assert_eq!(tr.tokens.len(), 8);
        assert_eq!(tr.words.len(), 8);
        assert_eq!(tr.sentences.len(), 2);
        assert_eq!(tr.summary.weakest_sentence, Some(1));
        assert!(tr.sentences[1].geo_mean_conf < tr.sentences[0].geo_mean_conf);
        assert!((tr.summary.min_conf - 0.2).abs() < 1e-6);
        assert!(tr.summary.p10_conf >= 0.2 && tr.summary.p10_conf < 0.9);
        assert!((tr.summary.mean_conf - (6.0 * 0.9 + 2.0 * 0.2) / 8.0).abs() < 1e-5);
        assert!(tr.summary.perplexity.unwrap() > 1.0);
        let j = tr.to_json();
        assert_eq!(j["tokens"].as_array().unwrap().len(), 8);
        assert_eq!(j["summary"]["weakest_sentence"], 1);
        assert!(serde_json::from_str::<Value>(&tr.to_json_string()).is_ok());
    }

    #[test]
    fn bad_spans_are_errors() {
        let s = step_stats(&[0.0, 1.0], None).unwrap();
        let t = "é!";
        assert_eq!(UncertaintyTrace::build(t, &[TokenSpan { start: 1, end: 2 }], &[s]), Err(ConfError::BadSpan { index: 0 }));
        assert!(UncertaintyTrace::build(t, &[TokenSpan { start: 0, end: 9 }], &[s]).is_err());
        assert!(UncertaintyTrace::build(t, &[TokenSpan { start: 2, end: 1 }], &[s]).is_err());
        assert!(matches!(UncertaintyTrace::build(t, &[], &[]), Err(ConfError::Empty)));
        assert!(matches!(UncertaintyTrace::build(t, &[TokenSpan { start: 0, end: 2 }], &[]), Err(ConfError::LengthMismatch { .. })));
        // zero-width and pure-whitespace tokens and an empty text are accepted
        assert!(UncertaintyTrace::build("", &[TokenSpan { start: 0, end: 0 }], &[s]).is_ok());
        assert!(UncertaintyTrace::build("  ", &[TokenSpan { start: 0, end: 2 }], &[s]).is_ok());
    }

    #[test]
    fn risk_flags_unsure_and_unsupported_entities() {
        let ctx = "The Eiffel Tower in Paris was completed in 1889 by Gustave Eiffel.";
        let good = demo_trace("The Eiffel Tower was completed in 1889 by Gustave Eiffel.", |_| 0.95, |_| 0.1);
        let bad = demo_trace("The Eiffel Tower was completed in 1887 by Zorblax Quimby.", |w| if w == "1887" || w.starts_with("Zorb") || w.starts_with("Quim") { 0.2 } else { 0.95 }, |_| 0.1);
        let (rg, rb) = (assess(&good, Some(ctx), &RiskOptions::default()), assess(&bad, Some(ctx), &RiskOptions::default()));
        assert!(rg.score < 0.05, "{rg:?}");
        assert!(rb.score > 0.5, "{rb:?}");
        assert!(rb.reasons.iter().any(|r| r.code == "unsupported_entity" && r.message.contains("1887")));
        assert!(rb.reasons.iter().any(|r| r.code == "low_conf_entity"));
        assert!(rb.reasons.windows(2).all(|w| w[0].weight >= w[1].weight));
        assert!(rb.features.unsupported_entity > 0.4 && rg.features.unsupported_entity == 0.0);
        // without context there is no support feature, but low confidence still shows
        let rn = assess(&bad, None, &RiskOptions::default());
        assert_eq!(rn.features.unsupported_entity, 0.0);
        assert!(rn.score > 0.2 && rn.score < rb.score);
        assert!(rb.to_json()["reasons"].as_array().unwrap().len() >= 2);
    }

    #[test]
    fn risk_spots_digit_uncertainty_and_spike_after_copying() {
        // digits as separate tokens
        let text = "Total 1 2 3 4";
        let mut spans = vec![]; let mut stats = vec![];
        let mut at = 0;
        for (i, w) in text.split_inclusive(' ').enumerate() {
            spans.push(TokenSpan { start: at, end: at + w.len() });
            let mut s = step_stats(&[0.0, 0.0], Some(0)).unwrap();
            s.entropy = if i >= 2 { 2.0 } else { 0.1 };
            stats.push(s); at += w.len();
        }
        // make digits adjacent: token spans "1","2","3","4" without spaces
        let adj = "n 1234";
        let sp = [TokenSpan { start: 0, end: 2 }, TokenSpan { start: 2, end: 3 }, TokenSpan { start: 3, end: 4 }, TokenSpan { start: 4, end: 5 }, TokenSpan { start: 5, end: 6 }];
        let st: Vec<StepStats> = [0.1f32, 0.1, 2.0, 2.0, 0.1].iter().map(|&e| { let mut s = step_stats(&[0.0, 0.0], Some(0)).unwrap(); s.entropy = e; s }).collect();
        let tr = UncertaintyTrace::build(adj, &sp, &st).unwrap();
        let r = assess(&tr, None, &RiskOptions::default());
        assert!(r.reasons.iter().any(|x| x.code == "digit_entropy"), "{r:?}");
        assert!((r.features.digit_entropy - 2.0 / 3.0).abs() < 1e-6);
        let _ = (spans, stats);
        // copy run then spike
        let ctx = "alpha beta gamma delta epsilon zeta eta";
        let ans = "alpha beta gamma delta epsilon unknown";
        let tr = demo_trace(ans, |_| 0.9, |w| if w == "unknown" { 3.0 } else { 0.05 });
        let r = assess(&tr, Some(ctx), &RiskOptions::default());
        assert!(r.reasons.iter().any(|x| x.code == "spike_after_copy"), "{r:?}");
        assert!((r.features.spike_after_copy - 1.0).abs() < 1e-6);
        let flat = demo_trace(ans, |_| 0.9, |_| 0.05);
        assert_eq!(assess(&flat, Some(ctx), &RiskOptions::default()).features.spike_after_copy, 0.0);
    }

    #[test]
    fn f1_rouge_and_number_normalisation() {
        assert_eq!(token_f1("The cat sat", "the CAT sat!"), 1.0);
        assert_eq!(token_f1("", ""), 1.0);
        assert_eq!(token_f1("a", ""), 0.0);
        assert!((token_f1("a b c d", "a b x y") - 0.5).abs() < 1e-6);
        assert!((rouge_l("a b c d", "a x c d") - 0.75).abs() < 1e-6);
        assert!(rouge_l("a b c", "c b a") < 0.5);
        for (s, w) in [("1,234.50", "1234.5"), ("1.234,5", "1234.5"), ("12 345", "12345"), ("1,234", "1234"), ("3,14", "3.14"), ("0.50", "0.5"), ("-0", "0"),
            ("007", "7"), ("$99", "99"), ("45%", "45"), ("+8", "8"), ("1.234.567", "1234567")] {
            assert_eq!(normalize_number(s).as_deref(), Some(w), "{s}");
        }
        for s in ["", "abc", "1-2", "--3", ".", "1e5", "12a"] { assert_eq!(normalize_number(s), None, "{s}"); }
    }

    #[test]
    fn answer_extraction_and_majority_vote() {
        assert_eq!(extract_answer("Let me think... so the answer is 1,234."), "1234");
        assert_eq!(extract_answer("Răspunsul este: Paris."), "paris");
        assert_eq!(extract_answer("  42 "), "42");
        assert_eq!(extract_answer("First we add them, then we multiply, and the total after all that comes to 17 apples"), "17");
        assert_eq!(extract_answer("Final answer: The Moon\nthanks"), "the moon");
        let c = consistency(&["The answer is 12.", "I think 12", "answer: 12.0", "The answer is 13."]).unwrap();
        let m = c.majority.unwrap();
        assert_eq!((m.answer.as_str(), m.votes, m.k, m.lead, m.representative), ("12", 3, 4, 2, 0));
        assert!((m.confidence - 0.75).abs() < 1e-6);
        assert!(m.wilson_lower < m.confidence && m.wilson_lower > 0.3);
        assert!(c.per_sample_support[3] < c.per_sample_support[0]);
        let one = consistency(&["x"]).unwrap();
        assert_eq!((one.agreement, one.per_sample_support[0]), (None, 1.0));
        assert!(consistency(&[]).is_err());
        assert!(consistency(&vec!["a"; 65]).is_err());
        let same = consistency(&["yes", "yes"]).unwrap();
        assert_eq!(same.agreement, Some(1.0));
        assert_eq!(tally(&["a".into(), "b".into()]).unwrap().answer, "a"); // tie: first seen
    }

    #[test]
    fn early_stop_rule() {
        let r = EarlyStop { min_k: 3, max_k: 8, stop_conf: 0.5 };
        let k = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(!r.should_stop(&k(&[])));
        assert!(!r.should_stop(&k(&["a", "a"]))); // below min_k
        assert!(r.should_stop(&k(&["a", "a", "a"]))); // unanimous, Wilson ≥ 0.5
        assert!(!r.should_stop(&k(&["a", "b", "a"])));
        assert!(r.should_stop(&k(&["a"; 8])));
        // the winner can no longer be overtaken: lead 4 > 8 - 5... (k=5 left=3)
        assert!(!r.should_stop(&k(&["a", "a", "a", "a", "b"])));
        assert!(r.should_stop(&k(&["a", "a", "a", "a", "a", "b"])));
        let strict = EarlyStop { min_k: 3, max_k: 8, stop_conf: 0.95 };
        assert!(!strict.should_stop(&k(&["a", "a", "a"])));
    }

    #[test]
    fn ece_brier_and_bins_on_calibrated_vs_overconfident_data() {
        let mut r = Rng(3);
        let n = 20000;
        let scores: Vec<f32> = (0..n).map(|_| r.f()).collect();
        let calibrated: Vec<bool> = scores.iter().map(|&s| r.f() < s).collect();
        let overconf: Vec<bool> = scores.iter().map(|&s| r.f() < 0.5 + (s - 0.5) * 0.3).collect();
        let (e1, e2) = (ece(&scores, &calibrated, 10).unwrap(), ece(&scores, &overconf, 10).unwrap());
        assert!(e1 < 0.03, "{e1}");
        assert!(e2 > 0.1 && e2 > 3.0 * e1, "{e2}");
        assert!(mce(&scores, &overconf, 10).unwrap() >= e2);
        assert!(brier(&scores, &calibrated).unwrap() < brier(&scores, &overconf).unwrap());
        let bins = reliability_bins(&scores, &calibrated, 10).unwrap();
        assert_eq!(bins.len(), 10);
        assert_eq!(bins.iter().map(|b| b.count).sum::<usize>(), n);
        assert!((bins[9].hi - 1.0).abs() < 1e-6 && bins[9].mean_score > 0.9);
        // by hand: 4 items
        let e = ece(&[1.0, 1.0, 0.0, 0.0], &[true, false, false, false], 2).unwrap();
        assert!((e - 0.25).abs() < 1e-6);
        assert!((brier(&[1.0, 0.0], &[true, true]).unwrap() - 0.5).abs() < 1e-6);
        assert!(ece(&[0.5], &[true, false], 5).is_err());
        assert!(ece(&[1.5], &[true], 5).is_err());
        assert!(ece(&[f32::NAN], &[true], 5).is_err());
        assert!(ece(&[0.5], &[true], 0).is_err());
        assert!(ece(&[], &[], 5).is_err());
    }

    #[test]
    fn auroc_matches_the_naive_pairwise_count() {
        let mut r = Rng(5);
        for trial in 0..30 {
            let n = 5 + trial * 3;
            let scores: Vec<f32> = (0..n).map(|_| (r.f() * 6.0).floor()).collect(); // many ties
            let labels: Vec<bool> = (0..n).map(|i| i % 3 == 0 || r.f() < 0.4).collect();
            let (mut num, mut den) = (0f64, 0f64);
            for i in 0..n { for j in 0..n { if labels[i] && !labels[j] { den += 1.0; num += if scores[i] > scores[j] { 1.0 } else if scores[i] == scores[j] { 0.5 } else { 0.0 }; } } }
            if den == 0.0 { continue; }
            assert!((auroc(&scores, &labels).unwrap() as f64 - num / den).abs() < 1e-6, "trial {trial}");
        }
        assert_eq!(auroc(&[0.1, 0.9], &[false, true]).unwrap(), 1.0);
        assert_eq!(auroc(&[0.9, 0.1], &[false, true]).unwrap(), 0.0);
        assert_eq!(auroc(&[0.5, 0.5], &[false, true]).unwrap(), 0.5);
        assert_eq!(auroc(&[0.1, 0.2], &[true, true]), Err(ConfError::NeedBothClasses));
        assert!(auroc(&[f32::NAN, 0.2], &[true, false]).is_err());
    }

    #[test]
    fn pav_equals_the_brute_force_min_max_formula() {
        let mut r = Rng(9);
        for trial in 0..60 {
            let n = 1 + trial % 12;
            let y: Vec<f64> = (0..n).map(|_| (r.f() * 4.0).floor() as f64 / 4.0).collect();
            let got = pav(&y, &vec![1.0; n]);
            for i in 0..n {
                // isotonic fit at i = max over s<=i of min over t>=i of mean(y[s..=t])
                let mut best = f64::NEG_INFINITY;
                for s in 0..=i {
                    let mut low = f64::INFINITY;
                    for t in i..n { low = low.min(y[s..=t].iter().sum::<f64>() / (t - s + 1) as f64); }
                    best = best.max(low);
                }
                assert!((got[i] - best).abs() < 1e-9, "trial {trial} i {i}: {} vs {best}", got[i]);
            }
            assert!(got.windows(2).all(|w| w[0] <= w[1] + 1e-12));
        }
        // weights: a heavy first point pulls the pooled block
        let g = pav(&[1.0, 0.0], &[3.0, 1.0]);
        assert!((g[0] - 0.75).abs() < 1e-12 && (g[1] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn calibrators_improve_calibration_stay_monotone_and_round_trip() {
        let mut r = Rng(13);
        let n = 6000;
        let scores: Vec<f32> = (0..n).map(|_| r.f()).collect();
        // true probability is s^2 (the raw score is miscalibrated)
        let labels: Vec<bool> = scores.iter().map(|&s| r.f() < s * s).collect();
        let before = ece(&scores, &labels, 10).unwrap();
        for m in [Method::Isotonic, Method::Platt] {
            let c = Calibrator::fit(&scores, &labels, m).unwrap();
            let cal: Vec<f32> = scores.iter().map(|&s| c.predict(s).unwrap()).collect();
            let after = ece(&cal, &labels, 10).unwrap();
            assert!(after < before * 0.7 || after < 0.03, "{m:?} {after} vs {before}");
            let grid: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
            let p: Vec<f32> = grid.iter().map(|&x| c.predict(x).unwrap()).collect();
            assert!(p.windows(2).all(|w| w[0] <= w[1] + 1e-6), "{m:?} not monotone");
            assert!(p.iter().all(|x| (0.0..=1.0).contains(x)));
            let back = Calibrator::from_json(&serde_json::from_str(&c.to_json().to_string()).unwrap()).unwrap();
            for &x in &grid { assert!((back.predict(x).unwrap() - c.predict(x).unwrap()).abs() < 1e-4); }
            assert_eq!(c.predict(f32::NAN), None);
        }
        assert_eq!(Calibrator::fit(&[0.1, 0.2], &[true, true], Method::Platt), Err(ConfError::NeedBothClasses));
        assert!(Calibrator::fit(&[0.1], &[true, false], Method::Isotonic).is_err());
        // a score with ties and a perfectly separable set still fit
        assert!(Calibrator::fit(&[0.1, 0.1, 0.9, 0.9], &[false, false, true, true], Method::Platt).is_ok());
        let c = Calibrator::fit(&[0.1, 0.1, 0.9, 0.9], &[false, false, true, true], Method::Isotonic).unwrap();
        assert_eq!((c.predict(0.0), c.predict(1.0)), (Some(0.0), Some(1.0)));
    }

    #[test]
    fn hostile_calibrator_json_is_rejected() {
        for bad in [r#"[]"#, r#"{"method":"x"}"#, r#"{"method":"platt","a":"1","b":0}"#, r#"{"method":"isotonic","xs":[],"ys":[]}"#,
            r#"{"method":"isotonic","xs":[0.5,0.1],"ys":[0.1,0.2]}"#, r#"{"method":"isotonic","xs":[0,1],"ys":[0.9,0.1]}"#, r#"{"method":"isotonic","xs":[0,1],"ys":[0.1,2]}"#,
            r#"{"method":"isotonic","xs":[0,1],"ys":[0.1]}"#] {
            assert!(Calibrator::from_json(&serde_json::from_str(bad).unwrap()).is_err(), "{bad}");
        }
    }

    #[test]
    fn fuzz_never_panics() {
        let mut r = Rng(99);
        let alphabet: Vec<char> = "ab 1.,\n!?é日 \u{a0}\"'()…".chars().collect();
        for _ in 0..300 {
            let len = (r.next() % 60) as usize;
            let text: String = (0..len).map(|_| alphabet[(r.next() % alphabet.len() as u64) as usize]).collect();
            let _ = (sentence_spans(&text), word_spans(&text), extract_answer(&text), normalize_number(&text), token_f1(&text, "a b"), rouge_l(&text, "a b"));
            let bounds: Vec<usize> = text.char_indices().map(|(i, _)| i).chain([text.len()]).collect();
            let spans: Vec<TokenSpan> = (0..(1 + r.next() % 8) as usize).map(|_| {
                let (a, b) = (bounds[(r.next() % bounds.len() as u64) as usize], bounds[(r.next() % bounds.len() as u64) as usize]);
                TokenSpan { start: a.min(b), end: a.max(b) }
            }).collect();
            let stats: Vec<StepStats> = spans.iter().map(|_| {
                let l: Vec<f32> = (0..1 + r.next() % 9).map(|_| match r.next() % 12 { 0 => f32::NAN, 1 => f32::INFINITY, 2 => f32::NEG_INFINITY, _ => (r.f() - 0.5) * 100.0 }).collect();
                step_stats(&l, Some((r.next() % 12) as usize)).unwrap_or_else(|| step_stats(&[0.0], None).unwrap())
            }).collect();
            if let Ok(tr) = UncertaintyTrace::build(&text, &spans, &stats) {
                let risk = assess(&tr, Some(&text), &RiskOptions::default());
                assert!((0.0..=1.0).contains(&risk.score));
                let _ = tr.to_json_string();
            }
        }
    }
}
