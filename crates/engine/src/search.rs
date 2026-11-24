//! Test-time-compute tools: spend more computation at generation time to get better, faster or
//! more reliable answers from the *same* weights.
//!
//! * [`beam_search`] – beam search with a length penalty; every surviving beam owns a forked KV
//!   cache and the last child of a beam takes its parent's cache over, so width 1 never copies.
//! * [`sample_n`], [`best_of_n`], [`self_consistency`] – draw N continuations from one prefilled
//!   prompt (one fork each), then pick the best by a pluggable [`Scorer`] or by majority vote over
//!   normalised answers.
//! * [`lookup_generate`] – prompt-lookup / n-gram lookahead speculative decoding. No draft model:
//!   proposals come from earlier occurrences of the current suffix in the prompt, the output so far
//!   and (optionally) reference documents, and the target verifies them in one batched pass. In
//!   greedy mode the tokens are those plain greedy decoding gives (up to floating-point rounding
//!   between a batched and a one-token pass); with sampling the output follows the target's
//!   distribution (unbiased rejection sampling, see [`crate::spec`]).
//! * [`Tree`] – a branching conversation: fork a state at any token position, try alternatives, keep
//!   the best.
//!
//! Everything works on the [`SeqModel`] trait (implemented for [`Model`]) and uses `u32` tokens like
//! [`crate::spec`]. Pure std, deterministic for a given seed.
use crate::model::{Model, ModelError};
use crate::spec::{
    self, AdaptiveLen, Draft, Drafter, NgramLookup, Rng, SamplingCfg, SpecConfig, SpecError,
    SpecStats, SuffixCache, Verifier, argmax,
};
use std::collections::HashMap;

// ------------------------------------------------------------------------------------------
// Errors and the model abstraction
// ------------------------------------------------------------------------------------------

#[derive(Debug)]
pub enum SearchError {
    Model(ModelError),
    Spec(SpecError),
    Invalid(String),
}
impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::Model(e) => write!(f, "{e}"),
            SearchError::Spec(e) => write!(f, "{e}"),
            SearchError::Invalid(s) => write!(f, "invalid request: {s}"),
        }
    }
}
impl std::error::Error for SearchError {}
impl From<ModelError> for SearchError {
    fn from(e: ModelError) -> Self {
        SearchError::Model(e)
    }
}
impl From<SpecError> for SearchError {
    fn from(e: SpecError) -> Self {
        SearchError::Spec(e)
    }
}

/// What the search tools need from a model: logits for *every* fed position, and cheap copies and
/// rewinds of the conversation state.
pub trait SeqModel {
    type State;
    fn new_state(&self) -> Self::State;
    /// Feed `tokens`; return one logit vector per token (vector `i` predicts the token after `tokens[i]`).
    fn logits_all(&self, st: &mut Self::State, tokens: &[u32])
    -> Result<Vec<Vec<f32>>, ModelError>;
    /// Feed `tokens` and return only the logits after the last one. Models override this to project
    /// just the final hidden state (the default computes every row and keeps the last).
    fn logits_last(&self, st: &mut Self::State, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        self.logits_all(st, tokens)?
            .pop()
            .ok_or(ModelError::NoTokens)
    }
    /// Exact-batch mode (see `State::set_exact_batch`): batched passes give the same logits as feeding
    /// the tokens one at a time, even for quantised weights. A no-op for models that are always exact.
    fn set_exact(&self, _st: &mut Self::State, _on: bool) {}
    /// Keep only the first `pos` positions.
    fn rewind(&self, st: &mut Self::State, pos: usize) -> Result<(), ModelError>;
    fn fork_state(&self, st: &Self::State) -> Result<Self::State, ModelError>;
    fn position(&self, st: &Self::State) -> usize;
    fn vocab(&self) -> usize;
    fn context(&self) -> usize;
}

/// Tokens per batched pass in [`SeqModel::logits_all`] (bounds the scratch memory).
const CHUNK: usize = 128;

impl SeqModel for Model {
    type State = crate::model::State;
    fn new_state(&self) -> Self::State {
        Model::new_state(self)
    }
    fn logits_all(
        &self,
        st: &mut Self::State,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        self.logits_rows(st, tokens, false)
    }
    fn logits_last(&self, st: &mut Self::State, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        self.logits_rows(st, tokens, true)?
            .pop()
            .ok_or(ModelError::NoTokens)
    }
    fn set_exact(&self, st: &mut Self::State, on: bool) {
        st.set_exact_batch(on);
    }
    fn rewind(&self, st: &mut Self::State, pos: usize) -> Result<(), ModelError> {
        st.truncate(pos)
    }
    fn fork_state(&self, st: &Self::State) -> Result<Self::State, ModelError> {
        st.fork()
    }
    fn position(&self, st: &Self::State) -> usize {
        st.pos
    }
    fn vocab(&self) -> usize {
        self.spec.vocab as usize
    }
    fn context(&self) -> usize {
        self.spec.context as usize
    }
}

impl Model {
    /// Shared body of `logits_all` / `logits_last`: with `last_only`, only the final token's hidden
    /// state goes through the output head (one row instead of one per fed token).
    fn logits_rows(
        &self,
        st: &mut crate::model::State,
        tokens: &[u32],
        last_only: bool,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        let (h, layers, vocab) = (
            self.spec.hidden as usize,
            self.spec.layers as usize,
            self.spec.vocab as usize,
        );
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= vocab) {
            return Err(ModelError::Token(t as usize));
        }
        let mut out = Vec::with_capacity(if last_only { 1 } else { tokens.len() });
        let n_chunks = tokens.len().div_ceil(CHUNK);
        for (ci, c) in tokens.chunks(CHUNK).enumerate() {
            let mut x = Vec::with_capacity(c.len() * h);
            for &t in c {
                x.extend_from_slice(&self.embed_token(t as usize)?);
            }
            self.run_layers_batch(st, 0..layers, &mut x, st.pos)?;
            let ids: Vec<usize> = c.iter().map(|&t| t as usize).collect();
            self.finish_batch(st, &ids);
            // last_only: only the final chunk's final row goes through the (vocabulary-sized) output head
            let from = if !last_only {
                0
            } else if ci + 1 < n_chunks {
                c.len()
            } else {
                c.len() - 1
            };
            for row in x.chunks(h).skip(from) {
                let l = self.logits_from_hidden(row);
                if l.is_empty() {
                    return Err(ModelError::Shape(
                        "the output head produced no logits".into(),
                    ));
                }
                out.push(l);
            }
        }
        Ok(out)
    }
}

/// Feed `tokens` and return the logits after the last one.
fn feed_last<M: SeqModel + ?Sized>(
    m: &M,
    st: &mut M::State,
    tokens: &[u32],
) -> Result<Vec<f32>, ModelError> {
    m.logits_last(st, tokens)
}

// ------------------------------------------------------------------------------------------
// Small numeric helpers
// ------------------------------------------------------------------------------------------

/// `log softmax` in f64 (non-finite logits count as impossible).
pub fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![f64::NEG_INFINITY; logits.len()];
    }
    let sum: f64 = logits
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| ((v - max) as f64).exp())
        .sum();
    let lse = max as f64 + sum.ln();
    logits
        .iter()
        .map(|&v| {
            if v.is_finite() {
                v as f64 - lse
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect()
}

/// The `n` highest entries of `lp`, best first, ties broken by the lower token id.
fn top_n(lp: &[f64], n: usize) -> Vec<(u32, f64)> {
    let mut idx: Vec<u32> = (0..lp.len() as u32)
        .filter(|&i| lp[i as usize] > f64::NEG_INFINITY)
        .collect();
    let cmp = |a: &u32, b: &u32| lp[*b as usize].total_cmp(&lp[*a as usize]).then(a.cmp(b));
    let n = n.min(idx.len());
    if n == 0 {
        return vec![];
    }
    if n < idx.len() {
        idx.select_nth_unstable_by(n - 1, cmp);
        idx.truncate(n);
    }
    idx.sort_by(cmp);
    idx.into_iter().map(|i| (i, lp[i as usize])).collect()
}

fn len_norm(len: usize, alpha: f64) -> f64 {
    if alpha <= 0.0 {
        1.0
    } else {
        ((5.0 + len as f64) / 6.0).powf(alpha)
    }
}

/// How many tokens may be generated after a prompt of `prompt_len` (the last one is never fed back).
fn room<M: SeqModel + ?Sized>(
    m: &M,
    prompt_len: usize,
    max_new: usize,
) -> Result<usize, SearchError> {
    if prompt_len == 0 {
        return Err(SearchError::Invalid("the prompt is empty".into()));
    }
    if prompt_len > m.context() {
        return Err(SearchError::Model(ModelError::ContextFull(m.context())));
    }
    Ok(max_new.min(m.context() - prompt_len + 1))
}

// ------------------------------------------------------------------------------------------
// Greedy baseline and beam search
// ------------------------------------------------------------------------------------------

/// One finished (or truncated) continuation.
#[derive(Clone, Debug, PartialEq)]
pub struct Hypothesis {
    /// Generated tokens; the stop token itself is not included.
    pub tokens: Vec<u32>,
    /// Sum of log-probabilities under the model (the stop token's included when `stopped`).
    pub logprob: f64,
    /// `logprob` divided by the length penalty; what hypotheses are ranked by.
    pub score: f64,
    pub stopped: bool,
}

/// Plain greedy decoding through the same code path as the other tools (the reference for tests).
pub fn greedy_decode<M: SeqModel + ?Sized>(
    m: &M,
    prompt: &[u32],
    max_new: usize,
    stop: &[u32],
) -> Result<Hypothesis, SearchError> {
    let limit = room(m, prompt.len(), max_new)?;
    let mut st = m.new_state();
    let mut logits = feed_last(m, &mut st, prompt)?;
    let (mut tokens, mut lp) = (Vec::new(), 0.0);
    while tokens.len() < limit {
        let t = argmax(&logits);
        lp += log_softmax(&logits)[t as usize];
        if stop.contains(&t) {
            return Ok(Hypothesis {
                tokens,
                logprob: lp,
                score: lp,
                stopped: true,
            });
        }
        tokens.push(t);
        if tokens.len() < limit {
            logits = feed_last(m, &mut st, &[t])?;
        }
    }
    Ok(Hypothesis {
        tokens,
        logprob: lp,
        score: lp,
        stopped: false,
    })
}

#[derive(Clone, Debug)]
pub struct BeamConfig {
    /// Number of beams kept (and of hypotheses returned at most).
    pub width: usize,
    pub max_new: usize,
    pub stop: Vec<u32>,
    /// GNMT length penalty exponent: a hypothesis of `L` tokens is scored `logprob / ((5+L)/6)^alpha`.
    /// `0` ranks by raw log-probability (then beam width 1 is exactly greedy decoding).
    pub length_penalty: f32,
    /// Stop tokens are not allowed before this many tokens were generated.
    pub min_new: usize,
}
impl Default for BeamConfig {
    fn default() -> Self {
        BeamConfig {
            width: 4,
            max_new: 64,
            stop: vec![],
            length_penalty: 0.0,
            min_new: 0,
        }
    }
}

struct Beam<S> {
    tokens: Vec<u32>,
    lp: f64,
    st: Option<S>,
    lsm: Vec<f64>,
}

/// Beam search from a prompt. Returns up to `width` hypotheses, best first.
pub fn beam_search<M: SeqModel + ?Sized>(
    m: &M,
    prompt: &[u32],
    cfg: &BeamConfig,
) -> Result<Vec<Hypothesis>, SearchError> {
    room(m, prompt.len(), cfg.max_new)?;
    let mut st = m.new_state();
    let logits = feed_last(m, &mut st, prompt)?;
    beam_search_from(m, st, &logits, prompt.len(), cfg)
}

/// Beam search from a state that already holds `pos` tokens, whose next-token logits are `logits`.
pub fn beam_search_from<M: SeqModel + ?Sized>(
    m: &M,
    st: M::State,
    logits: &[f32],
    pos: usize,
    cfg: &BeamConfig,
) -> Result<Vec<Hypothesis>, SearchError> {
    let width = cfg.width.max(1);
    let alpha = f64::from(cfg.length_penalty.max(0.0));
    let limit = room(m, pos, cfg.max_new)?;
    let mut beams = vec![Beam {
        tokens: vec![],
        lp: 0.0,
        st: Some(st),
        lsm: log_softmax(logits),
    }];
    let mut finished: Vec<Hypothesis> = Vec::new();
    let rank = |a: &Hypothesis, b: &Hypothesis| {
        b.score
            .total_cmp(&a.score)
            .then(b.logprob.total_cmp(&a.logprob))
            .then(a.tokens.cmp(&b.tokens))
    };

    for step in 0..limit {
        // Candidates: every beam's best 2*width extensions, ranked by total log-probability.
        let mut cands: Vec<(f64, usize, u32)> = Vec::new();
        for (bi, b) in beams.iter().enumerate() {
            for (t, l) in top_n(&b.lsm, 2 * width + cfg.stop.len()) {
                if cfg.stop.contains(&t) && step < cfg.min_new {
                    continue;
                }
                cands.push((b.lp + l, bi, t));
            }
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut next: Vec<(f64, usize, u32)> = Vec::new();
        for (r, &(total, bi, t)) in cands.iter().take(2 * width).enumerate() {
            if cfg.stop.contains(&t) {
                if r < width {
                    finished.push(Hypothesis {
                        tokens: beams[bi].tokens.clone(),
                        logprob: total,
                        score: total / len_norm(step + 1, alpha),
                        stopped: true,
                    });
                }
            } else if next.len() < width {
                next.push((total, bi, t));
            }
        }
        finished.sort_by(rank);
        finished.truncate(width);
        if next.is_empty() {
            break;
        }
        if step + 1 == limit {
            for &(total, bi, t) in &next {
                let mut tokens = beams[bi].tokens.clone();
                tokens.push(t);
                finished.push(Hypothesis {
                    tokens,
                    logprob: total,
                    score: total / len_norm(step + 1, alpha),
                    stopped: false,
                });
            }
            break;
        }
        // Nothing alive can still beat the worst finished hypothesis (future log-probs are <= 0).
        if finished.len() >= width {
            let bound = next[0].0
                / if alpha > 0.0 {
                    len_norm(limit, alpha)
                } else {
                    1.0
                };
            if finished.last().is_some_and(|w| w.score >= bound) {
                break;
            }
        }
        // Materialise the survivors: the last child of a beam takes its cache, the others fork it.
        let mut remaining: HashMap<usize, usize> = HashMap::new();
        for &(_, bi, _) in &next {
            *remaining.entry(bi).or_insert(0) += 1;
        }
        let mut survivors = Vec::with_capacity(next.len());
        for &(total, bi, t) in &next {
            let left = remaining
                .get_mut(&bi)
                .map(|c| {
                    *c -= 1;
                    *c
                })
                .unwrap_or(0);
            let mut cs = if left == 0 {
                beams[bi]
                    .st
                    .take()
                    .ok_or_else(|| SearchError::Invalid("a beam lost its cache".into()))?
            } else {
                match beams[bi].st.as_ref() {
                    Some(s) => m.fork_state(s)?,
                    None => return Err(SearchError::Invalid("a beam lost its cache".into())),
                }
            };
            let lsm = log_softmax(&feed_last(m, &mut cs, &[t])?);
            let mut tokens = beams[bi].tokens.clone();
            tokens.push(t);
            survivors.push(Beam {
                tokens,
                lp: total,
                st: Some(cs),
                lsm,
            });
        }
        beams = survivors;
    }
    finished.sort_by(rank);
    finished.truncate(width);
    Ok(finished)
}

// ------------------------------------------------------------------------------------------
// Sampling N continuations, best-of-N, self-consistency
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    /// Generated tokens (without the stop token).
    pub tokens: Vec<u32>,
    /// Sum of log-probabilities of the drawn tokens under the model at temperature 1.
    pub logprob: f64,
    pub stopped: bool,
}

#[derive(Clone, Debug)]
pub struct SampleConfig {
    pub max_new: usize,
    pub stop: Vec<u32>,
    pub sampling: SamplingCfg,
    /// Sample `i` uses the stream `seed + i * golden-ratio`, so results do not depend on `n`.
    pub seed: u64,
}
impl Default for SampleConfig {
    fn default() -> Self {
        SampleConfig {
            max_new: 64,
            stop: vec![],
            sampling: SamplingCfg::sampled(0.8, 40, 0.95),
            seed: 0,
        }
    }
}

/// Draw `n` independent continuations of `prompt`. The prompt is processed once; every sample but
/// the last continues from a fork of its cache.
pub fn sample_n<M: SeqModel + ?Sized>(
    m: &M,
    prompt: &[u32],
    n: usize,
    cfg: &SampleConfig,
) -> Result<Vec<Sample>, SearchError> {
    let limit = room(m, prompt.len(), cfg.max_new)?;
    let mut base = Some(m.new_state());
    let logits = feed_last(
        m,
        base.as_mut()
            .ok_or_else(|| SearchError::Invalid("no state".into()))?,
        prompt,
    )?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut st = if i + 1 == n {
            base.take()
                .ok_or_else(|| SearchError::Invalid("no state".into()))?
        } else {
            m.fork_state(
                base.as_ref()
                    .ok_or_else(|| SearchError::Invalid("no state".into()))?,
            )?
        };
        let mut rng = Rng::new(
            cfg.seed
                .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        let (mut tokens, mut lp, mut stopped) = (Vec::new(), 0.0, false);
        let mut cur = logits.clone();
        while tokens.len() < limit {
            let t = if cfg.sampling.is_greedy() {
                argmax(&cur)
            } else {
                spec::sample_from(&cfg.sampling.probs(&cur), &mut rng)
            };
            lp += log_softmax(&cur)
                .get(t as usize)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            if cfg.stop.contains(&t) {
                stopped = true;
                break;
            }
            tokens.push(t);
            if tokens.len() < limit {
                cur = feed_last(m, &mut st, &[t])?;
            }
        }
        out.push(Sample {
            tokens,
            logprob: lp,
            stopped,
        });
    }
    Ok(out)
}

/// Rates a sample; higher is better. Closures `Fn(&Sample) -> f64` are scorers too (a verifier, a
/// reward model, a regex on the decoded text, ...).
pub trait Scorer {
    fn score(&self, s: &Sample) -> f64;
}
impl<F: Fn(&Sample) -> f64> Scorer for F {
    fn score(&self, s: &Sample) -> f64 {
        self(s)
    }
}
/// Total log-probability (favours short answers).
pub struct SumLogprob;
impl Scorer for SumLogprob {
    fn score(&self, s: &Sample) -> f64 {
        s.logprob
    }
}
/// Average log-probability per token.
pub struct MeanLogprob;
impl Scorer for MeanLogprob {
    fn score(&self, s: &Sample) -> f64 {
        s.logprob / s.tokens.len().max(1) as f64
    }
}

fn finite(v: f64) -> f64 {
    if v.is_nan() { f64::NEG_INFINITY } else { v }
}

/// The best sample by `scorer` (the earliest on ties); `None` when there are none.
pub fn best_of_n<'a>(samples: &'a [Sample], scorer: &dyn Scorer) -> Option<&'a Sample> {
    let mut best: Option<(&Sample, f64)> = None;
    for s in samples {
        let v = finite(scorer.score(s));
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((s, v));
        }
    }
    best.map(|(s, _)| s)
}

/// One answer and the samples that gave it.
#[derive(Clone, Debug, PartialEq)]
pub struct Vote {
    pub answer: String,
    pub count: usize,
    /// Indices into the sample slice, in order.
    pub members: Vec<usize>,
    /// The member with the highest score (the earliest without a scorer or on ties).
    pub best: usize,
    pub best_score: f64,
}

/// Majority vote. `key` maps a sample to its normalised answer (`None` discards the sample, e.g.
/// no answer could be extracted). Votes come back ordered by count, then by best score, then by
/// first appearance.
pub fn self_consistency(
    samples: &[Sample],
    key: &dyn Fn(&Sample) -> Option<String>,
    scorer: Option<&dyn Scorer>,
) -> Vec<Vote> {
    let mut votes: Vec<Vote> = Vec::new();
    let mut at: HashMap<String, usize> = HashMap::new();
    for (i, s) in samples.iter().enumerate() {
        let Some(k) = key(s) else { continue };
        let sc = scorer.map_or(0.0, |sc| finite(sc.score(s)));
        match at.get(&k) {
            Some(&vi) => {
                let v = &mut votes[vi];
                v.count += 1;
                v.members.push(i);
                if sc > v.best_score {
                    v.best = i;
                    v.best_score = sc;
                }
            }
            None => {
                at.insert(k.clone(), votes.len());
                votes.push(Vote {
                    answer: k,
                    count: 1,
                    members: vec![i],
                    best: i,
                    best_score: sc,
                });
            }
        }
    }
    let first = |v: &Vote| v.members[0];
    votes.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then(b.best_score.total_cmp(&a.best_score))
            .then(first(a).cmp(&first(b)))
    });
    votes
}

/// Share of the kept samples that agree with the winning answer (`0` with no votes).
pub fn agreement(votes: &[Vote]) -> f64 {
    let total: usize = votes.iter().map(|v| v.count).sum();
    if total == 0 {
        0.0
    } else {
        votes[0].count as f64 / total as f64
    }
}

/// Canonical form of a free-text answer: lower case, cedilla Romanian letters folded to the comma
/// forms, whitespace collapsed, surrounding quotes and punctuation removed.
pub fn normalise_answer(s: &str) -> String {
    let folded: String = s
        .chars()
        .map(|c| match c {
            'ş' => 'ș',
            'Ş' => 'Ș',
            'ţ' => 'ț',
            'Ţ' => 'Ț',
            c => c,
        })
        .collect();
    let lower = folded.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let joined = words.join(" ");
    joined
        .trim_matches(|c: char| {
            c.is_ascii_punctuation()
                || matches!(c, '«' | '»' | '„' | '”' | '“' | '’' | '‘' | '…' | '—' | '–')
        })
        .trim()
        .to_string()
}

/// The last number in `s`, canonical: no thousands commas, no trailing zeros (`"1,234.50"` -> `"1234.5"`).
pub fn last_number(s: &str) -> Option<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut i = chars.len();
    while i > 0 && !chars[i - 1].is_ascii_digit() {
        i -= 1;
    }
    if i == 0 {
        return None;
    }
    let mut j = i;
    while j > 0
        && (chars[j - 1].is_ascii_digit()
            || (matches!(chars[j - 1], '.' | ',')
                && j >= 2
                && chars[j - 2].is_ascii_digit()
                && j < chars.len()
                && chars[j].is_ascii_digit()))
    {
        j -= 1;
    }
    let neg = j > 0 && chars[j - 1] == '-' && (j == 1 || !chars[j - 2].is_alphanumeric());
    let raw: String = chars[j..i].iter().collect();
    // "1,234" and "1,234.5" use the comma as a thousands separator; "3,5" and "12,75" (Romanian,
    // French, German…) use it as the decimal mark.
    let decimal_comma = !raw.contains('.')
        && raw.matches(',').count() == 1
        && raw.rsplit(',').next().is_some_and(|f| f.len() != 3);
    let mut num: String = if decimal_comma {
        raw.replace(',', ".")
    } else {
        raw.replace(',', "")
    };
    if num.contains('.') {
        num = num.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    let num = num.trim_start_matches('0');
    let num = if num.is_empty() || num.starts_with('.') {
        format!("0{num}")
    } else {
        num.to_string()
    };
    Some(if neg && num != "0" {
        format!("-{num}")
    } else {
        num
    })
}

// ------------------------------------------------------------------------------------------
// Prompt-lookup / n-gram lookahead decoding
// ------------------------------------------------------------------------------------------

/// Adapter: a [`SeqModel`] and its state seen as a [`spec::Verifier`] over `u32` tokens.
pub struct ModelVerifier<'a, M: SeqModel + ?Sized> {
    m: &'a M,
    st: M::State,
    /// Every token fed so far, so a state that cannot be rewound (a wrapped sliding-window ring)
    /// can be rebuilt by replay.
    hist: Vec<u32>,
    /// Length of the first batch (the prompt). It is fed in the model's ordinary prefill mode, exactly
    /// like plain decoding; every later batch is a verification pass and runs in exact-batch mode so
    /// the verified tokens equal plain greedy decoding bit for bit (also for quantised weights).
    prompt_len: Option<usize>,
    error: Option<ModelError>,
}
impl<'a, M: SeqModel + ?Sized> ModelVerifier<'a, M> {
    pub fn new(m: &'a M) -> Self {
        ModelVerifier {
            m,
            st: m.new_state(),
            hist: vec![],
            prompt_len: None,
            error: None,
        }
    }
    /// The first model error seen (the `Verifier` interface can only signal it as an empty answer).
    pub fn take_error(&mut self) -> Option<ModelError> {
        self.error.take()
    }
    pub fn state(&self) -> &M::State {
        &self.st
    }
}
impl<M: SeqModel + ?Sized> Verifier for ModelVerifier<'_, M> {
    fn logits_batch(&mut self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let first = self.prompt_len.is_none();
        self.m.set_exact(&mut self.st, !first);
        match self.m.logits_all(&mut self.st, tokens) {
            Ok(v) => {
                if first {
                    self.prompt_len = Some(tokens.len());
                }
                self.hist.extend_from_slice(tokens);
                v
            }
            Err(e) => {
                self.error.get_or_insert(e);
                vec![]
            }
        }
    }
    fn rollback(&mut self, keep: usize) {
        if keep >= self.hist.len() {
            return;
        }
        self.hist.truncate(keep);
        if self.m.rewind(&mut self.st, keep).is_ok() {
            return;
        }
        // Not rewindable: start over from the kept tokens.
        self.st = self.m.new_state();
        let p = self.prompt_len.unwrap_or(0).min(self.hist.len());
        let (prompt, rest) = self.hist.split_at(p);
        self.m.set_exact(&mut self.st, false);
        if !prompt.is_empty()
            && let Err(e) = self.m.logits_all(&mut self.st, prompt)
        {
            self.error.get_or_insert(e);
        }
        self.m.set_exact(&mut self.st, true);
        if !rest.is_empty()
            && let Err(e) = self.m.logits_all(&mut self.st, rest)
        {
            self.error.get_or_insert(e);
        }
    }
}

/// Draft source: the longest earlier occurrence of the current suffix in the running context, and,
/// failing that, in optional reference documents (for instance retrieved passages the answer is
/// expected to quote).
pub struct Lookahead {
    ctx: NgramLookup,
    corpus: Option<SuffixCache>,
    n_max: usize,
}
impl Lookahead {
    pub fn new(n_min: usize, n_max: usize, corpus: &[Vec<u32>], draft_len: usize) -> Self {
        let corpus = if corpus.iter().all(|d| d.len() < 2) {
            None
        } else {
            let total: usize = corpus.iter().map(Vec::len).sum();
            let mut c = SuffixCache::new(
                n_min,
                n_max,
                draft_len.max(1),
                (total * (n_max.max(n_min) - n_min + 1)).max(16),
            );
            for d in corpus {
                c.learn(d);
            }
            Some(c)
        };
        Lookahead {
            ctx: NgramLookup::new(n_min, n_max),
            corpus,
            n_max: n_max.max(n_min),
        }
    }
}
impl Drafter for Lookahead {
    fn name(&self) -> &'static str {
        "lookahead"
    }
    fn draft(&mut self, ctx: &[u32], k: usize, cfg: &SamplingCfg, rng: &mut Rng) -> Draft {
        let d = self.ctx.draft(ctx, k, cfg, rng);
        if !d.tokens.is_empty() {
            return d;
        }
        if let Some(c) = &self.corpus {
            let tail = &ctx[ctx.len().saturating_sub(self.n_max)..];
            if let Some(cont) = c.lookup(tail) {
                return Draft::deterministic(cont.iter().copied().take(k).collect());
            }
        }
        Draft::empty()
    }
}

#[derive(Clone, Debug)]
pub struct LookupConfig {
    pub max_new: usize,
    /// Generation ends right after one of these (the stop token is included in the output).
    pub stop: Vec<u32>,
    pub sampling: SamplingCfg,
    pub seed: u64,
    /// Longest proposal per pass (the starting length when `adaptive`).
    pub draft_len: usize,
    pub adaptive: bool,
    /// Shortest and longest suffix looked up.
    pub n_min: usize,
    pub n_max: usize,
    /// Reference token sequences proposals may also come from.
    pub corpus: Vec<Vec<u32>>,
}
impl Default for LookupConfig {
    fn default() -> Self {
        LookupConfig {
            max_new: 64,
            stop: vec![],
            sampling: SamplingCfg::greedy(),
            seed: 0,
            draft_len: 6,
            adaptive: true,
            n_min: 1,
            n_max: 4,
            corpus: vec![],
        }
    }
}

#[derive(Clone, Debug)]
pub struct LookupOutput {
    pub tokens: Vec<u32>,
    pub stats: SpecStats,
    pub stop: spec::StopReason,
}

fn spec_cfg<M: SeqModel + ?Sized>(m: &M, cfg: &LookupConfig) -> SpecConfig {
    SpecConfig {
        max_new: cfg.max_new,
        stop: cfg.stop.clone(),
        sampling: cfg.sampling,
        seed: cfg.seed,
        draft_len: cfg.draft_len.max(1),
        adaptive: cfg
            .adaptive
            .then(|| AdaptiveLen::new(1, cfg.draft_len.max(1).max(8), cfg.draft_len.max(1))),
        ctx_limit: Some(m.context()),
    }
}

fn spec_failed<M: SeqModel + ?Sized>(v: &mut ModelVerifier<'_, M>, e: SpecError) -> SearchError {
    match v.take_error() {
        Some(me) => SearchError::Model(me),
        None => SearchError::Spec(e),
    }
}

/// Generate with prompt-lookup speculation (no draft model).
pub fn lookup_generate<M: SeqModel + ?Sized>(
    m: &M,
    prompt: &[u32],
    cfg: &LookupConfig,
) -> Result<LookupOutput, SearchError> {
    if prompt.len() > m.context() {
        return Err(SearchError::Model(ModelError::ContextFull(m.context())));
    }
    let mut v = ModelVerifier::new(m);
    let mut d = Lookahead::new(cfg.n_min, cfg.n_max, &cfg.corpus, cfg.draft_len);
    match spec::generate(&mut v, &mut d, prompt, &spec_cfg(m, cfg)) {
        Ok(o) => Ok(LookupOutput {
            tokens: o.tokens,
            stats: o.stats,
            stop: o.stop,
        }),
        Err(e) => Err(spec_failed(&mut v, e)),
    }
}

/// The one-token-at-a-time baseline with the same protocol, sampling and seed as [`lookup_generate`].
pub fn plain_generate<M: SeqModel + ?Sized>(
    m: &M,
    prompt: &[u32],
    cfg: &LookupConfig,
) -> Result<LookupOutput, SearchError> {
    if prompt.len() > m.context() {
        return Err(SearchError::Model(ModelError::ContextFull(m.context())));
    }
    let mut v = ModelVerifier::new(m);
    match spec::generate_plain(&mut v, prompt, &spec_cfg(m, cfg)) {
        Ok(o) => Ok(LookupOutput {
            tokens: o.tokens,
            stats: o.stats,
            stop: o.stop,
        }),
        Err(e) => Err(spec_failed(&mut v, e)),
    }
}

// ------------------------------------------------------------------------------------------
// Branching conversation
// ------------------------------------------------------------------------------------------

pub type NodeId = usize;

struct Node<S> {
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    /// The whole token sequence (prompt included) the node stands for.
    path: Vec<u32>,
    /// Log-probability of each token of `path` (prompt tokens count 0).
    lps: Vec<f64>,
    state: Option<S>,
    /// Logits for the token after `path`.
    logits: Vec<f32>,
}

/// A tree of continuations that share their ancestors' work: every node holds a cache of its own
/// (forked from its parent's), so branches can be extended, compared and dropped independently.
pub struct Tree<'m, M: SeqModel + ?Sized> {
    m: &'m M,
    nodes: Vec<Node<M::State>>,
}
impl<'m, M: SeqModel + ?Sized> Tree<'m, M> {
    /// Process `prompt`; node 0 is the root.
    pub fn new(m: &'m M, prompt: &[u32]) -> Result<Self, SearchError> {
        room(m, prompt.len(), 1)?;
        let mut st = m.new_state();
        let logits = feed_last(m, &mut st, prompt)?;
        Ok(Tree {
            m,
            nodes: vec![Node {
                parent: None,
                children: vec![],
                path: prompt.to_vec(),
                lps: vec![0.0; prompt.len()],
                state: Some(st),
                logits,
            }],
        })
    }
    pub fn root(&self) -> NodeId {
        0
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    fn node(&self, id: NodeId) -> Result<&Node<M::State>, SearchError> {
        self.nodes
            .get(id)
            .ok_or_else(|| SearchError::Invalid(format!("no node {id}")))
    }
    pub fn path(&self, id: NodeId) -> Result<&[u32], SearchError> {
        Ok(&self.node(id)?.path)
    }
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.nodes.get(id).and_then(|n| n.parent)
    }
    pub fn children(&self, id: NodeId) -> &[NodeId] {
        self.nodes.get(id).map_or(&[], |n| &n.children)
    }
    /// Next-token logits after the node's path.
    pub fn logits(&self, id: NodeId) -> Result<&[f32], SearchError> {
        Ok(&self.node(id)?.logits)
    }
    /// Sum of log-probabilities of the tokens added after the prompt.
    pub fn logprob(&self, id: NodeId) -> Result<f64, SearchError> {
        Ok(self.node(id)?.lps.iter().sum())
    }
    pub fn has_state(&self, id: NodeId) -> bool {
        self.nodes.get(id).is_some_and(|n| n.state.is_some())
    }
    /// Drop a node's cache to save memory; it is rebuilt by replaying its path when needed again.
    pub fn release_state(&mut self, id: NodeId) {
        if let Some(n) = self.nodes.get_mut(id) {
            n.state = None;
        }
    }
    pub fn leaves(&self) -> Vec<NodeId> {
        (0..self.nodes.len())
            .filter(|&i| self.nodes[i].children.is_empty())
            .collect()
    }

    /// A state holding exactly `path[..pos]`, forked from `id`'s or rebuilt by replay.
    fn state_upto(&self, id: NodeId, pos: usize) -> Result<M::State, SearchError> {
        let n = self.node(id)?;
        if let Some(s) = &n.state {
            let mut f = self.m.fork_state(s)?;
            if pos == n.path.len() {
                return Ok(f);
            }
            if self.m.rewind(&mut f, pos).is_ok() {
                return Ok(f);
            }
        }
        let mut st = self.m.new_state();
        if pos > 0 {
            self.m.logits_all(&mut st, &n.path[..pos])?;
        }
        Ok(st)
    }

    /// Extend `id` by `tokens` (a child node).
    pub fn branch(&mut self, id: NodeId, tokens: &[u32]) -> Result<NodeId, SearchError> {
        let len = self.node(id)?.path.len();
        self.branch_at(id, len, tokens)
    }

    /// Keep the first `pos` tokens of `id`'s path (`1..=len`), append `tokens` and make that a new
    /// node, a child of `id`. The cache of `id` is forked and rewound, or replayed if it cannot be.
    pub fn branch_at(
        &mut self,
        id: NodeId,
        pos: usize,
        tokens: &[u32],
    ) -> Result<NodeId, SearchError> {
        let (len, vocab) = (self.node(id)?.path.len(), self.m.vocab());
        if pos == 0 || pos > len {
            return Err(SearchError::Invalid(format!(
                "position {pos} is outside 1..={len}"
            )));
        }
        if tokens.is_empty() && pos == len {
            return Err(SearchError::Invalid("nothing to add".into()));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= vocab) {
            return Err(SearchError::Model(ModelError::Token(t as usize)));
        }
        if pos + tokens.len() > self.m.context() {
            return Err(SearchError::Model(ModelError::ContextFull(
                self.m.context(),
            )));
        }
        let mut path: Vec<u32> = self.nodes[id].path[..pos].to_vec();
        let mut lps: Vec<f64> = self.nodes[id].lps[..pos].to_vec();
        let (st, logits, first_lp);
        if pos == len {
            let mut s = self.state_upto(id, len)?;
            let out = self.m.logits_all(&mut s, tokens)?;
            first_lp = log_softmax(&self.nodes[id].logits);
            let mut all = out;
            let last = all.pop().ok_or(ModelError::NoTokens)?;
            // log-probs of tokens[1..] come from the logits after the preceding token
            let mut tl = vec![first_lp[tokens[0] as usize]];
            for (i, l) in all.iter().enumerate() {
                tl.push(log_softmax(l)[tokens[i + 1] as usize]);
            }
            lps.extend(tl);
            st = s;
            logits = last;
        } else {
            // Re-feed path[pos-1] so the logits that follow it are available.
            let mut s = self.state_upto(id, pos - 1)?;
            let mut feed = vec![self.nodes[id].path[pos - 1]];
            feed.extend_from_slice(tokens);
            let mut all = self.m.logits_all(&mut s, &feed)?;
            let last = all.pop().ok_or(ModelError::NoTokens)?;
            for (i, &t) in tokens.iter().enumerate() {
                lps.push(log_softmax(&all[i])[t as usize]);
            }
            st = s;
            logits = last;
        }
        path.extend_from_slice(tokens);
        let new = self.nodes.len();
        self.nodes.push(Node {
            parent: Some(id),
            children: vec![],
            path,
            lps,
            state: Some(st),
            logits,
        });
        self.nodes[id].children.push(new);
        Ok(new)
    }

    /// Children for the `k` most likely next tokens of `id`.
    pub fn expand_top(&mut self, id: NodeId, k: usize) -> Result<Vec<NodeId>, SearchError> {
        let lsm = log_softmax(&self.node(id)?.logits);
        top_n(&lsm, k)
            .into_iter()
            .map(|(t, _)| self.branch(id, &[t]))
            .collect()
    }

    /// Greedy continuation of `id` as one new node (stops at a stop token, which is not added).
    pub fn extend_greedy(
        &mut self,
        id: NodeId,
        max_new: usize,
        stop: &[u32],
    ) -> Result<NodeId, SearchError> {
        let len = self.node(id)?.path.len();
        // a node's logits require its last token to be fed, so the path may not exceed the context
        let limit = max_new.min(self.m.context() - len.min(self.m.context()));
        let mut st = self.state_upto(id, len)?;
        let (mut logits, mut path, mut lps) = (
            self.nodes[id].logits.clone(),
            self.nodes[id].path.clone(),
            self.nodes[id].lps.clone(),
        );
        let mut added = 0;
        while added < limit {
            let t = argmax(&logits);
            if stop.contains(&t) {
                break;
            }
            lps.push(log_softmax(&logits)[t as usize]);
            path.push(t);
            added += 1;
            if added < limit {
                logits = feed_last(self.m, &mut st, &[t])?;
            } else {
                logits = vec![];
            }
        }
        if added == 0 {
            return Err(SearchError::Invalid(
                "the greedy continuation is empty".into(),
            ));
        }
        if logits.is_empty() {
            logits = feed_last(self.m, &mut st, &path[path.len() - 1..])?;
        }
        let new = self.nodes.len();
        self.nodes.push(Node {
            parent: Some(id),
            children: vec![],
            path,
            lps,
            state: Some(st),
            logits,
        });
        self.nodes[id].children.push(new);
        Ok(new)
    }

    /// The leaf with the highest `score(path, logprob)`; the earliest on ties.
    pub fn best_leaf(&self, score: &dyn Fn(&[u32], f64) -> f64) -> Option<NodeId> {
        let mut best: Option<(NodeId, f64)> = None;
        for id in self.leaves() {
            let n = &self.nodes[id];
            let v = finite(score(&n.path, n.lps.iter().sum()));
            if best.is_none_or(|(_, b)| v > b) {
                best = Some((id, v));
            }
        }
        best.map(|(i, _)| i)
    }
}

// ------------------------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::tiny;

    fn model() -> Model {
        tiny(4, 2, false)
    }
    fn lcg(seed: u64, n: usize, vocab: u32) -> Vec<u32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as u32) % vocab
            })
            .collect()
    }

    #[test]
    fn logits_all_matches_one_token_forward() {
        let m = model();
        let toks = lcg(1, 9, 20);
        let mut a = Model::new_state(&m);
        let all = SeqModel::logits_all(&m, &mut a, &toks).unwrap();
        let mut b = Model::new_state(&m);
        for (i, &t) in toks.iter().enumerate() {
            let l = m.forward(&mut b, t as usize).unwrap();
            for (x, y) in all[i].iter().zip(&l) {
                assert!((x - y).abs() < 1e-4, "pos {i}: {x} vs {y}");
            }
        }
        assert_eq!(a.pos, 9);
    }

    #[test]
    fn log_softmax_normalises_and_handles_bad_values() {
        let l = log_softmax(&[1.0, 2.0, 3.0]);
        assert!((l.iter().map(|v| v.exp()).sum::<f64>() - 1.0).abs() < 1e-12);
        let l = log_softmax(&[f32::NAN, 0.0, f32::NEG_INFINITY]);
        assert_eq!(l[0], f64::NEG_INFINITY);
        assert!((l[1]).abs() < 1e-12);
        assert!(
            log_softmax(&[f32::NAN])
                .iter()
                .all(|v| *v == f64::NEG_INFINITY)
        );
    }

    #[test]
    fn beam_width_one_is_greedy() {
        let m = model();
        for seed in 0..6 {
            let prompt = lcg(seed, 3 + seed as usize, 20);
            let g = greedy_decode(&m, &prompt, 14, &[]).unwrap();
            let b = beam_search(
                &m,
                &prompt,
                &BeamConfig {
                    width: 1,
                    max_new: 14,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(b.len(), 1);
            assert_eq!(b[0].tokens, g.tokens, "seed {seed}");
            assert!((b[0].logprob - g.logprob).abs() < 1e-6);
        }
    }

    #[test]
    fn beam_width_one_is_greedy_with_stop_tokens() {
        let m = model();
        let prompt = lcg(3, 5, 20);
        let free = greedy_decode(&m, &prompt, 12, &[]).unwrap();
        // make the 4th greedy token a stop token: both must end right before it
        let stop = vec![free.tokens[3]];
        let g = greedy_decode(&m, &prompt, 12, &stop).unwrap();
        let b = beam_search(
            &m,
            &prompt,
            &BeamConfig {
                width: 1,
                max_new: 12,
                stop: stop.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(g.stopped && g.tokens.len() <= 3);
        assert_eq!(b[0].tokens, g.tokens);
        assert!(b[0].stopped);
    }

    #[test]
    fn wider_beam_never_scores_worse_and_is_sorted() {
        let m = model();
        let prompt = lcg(7, 4, 20);
        let one = beam_search(
            &m,
            &prompt,
            &BeamConfig {
                width: 1,
                max_new: 10,
                ..Default::default()
            },
        )
        .unwrap();
        let five = beam_search(
            &m,
            &prompt,
            &BeamConfig {
                width: 5,
                max_new: 10,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(five.len(), 5);
        assert!(five[0].logprob >= one[0].logprob - 1e-9);
        for w in five.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        // log-probabilities are what the model says
        let mut st = Model::new_state(&m);
        let mut all = prompt.clone();
        all.extend(&five[0].tokens);
        let lg = SeqModel::logits_all(&m, &mut st, &all).unwrap();
        let sum: f64 = five[0]
            .tokens
            .iter()
            .enumerate()
            .map(|(i, &t)| log_softmax(&lg[prompt.len() - 1 + i])[t as usize])
            .sum();
        assert!(
            (sum - five[0].logprob).abs() < 1e-4,
            "{sum} vs {}",
            five[0].logprob
        );
    }

    #[test]
    fn beam_is_deterministic_and_finds_the_exhaustive_optimum_when_wide_enough() {
        let m = model();
        let prompt = lcg(11, 3, 20);
        let a = beam_search(
            &m,
            &prompt,
            &BeamConfig {
                width: 3,
                max_new: 6,
                ..Default::default()
            },
        )
        .unwrap();
        let b = beam_search(
            &m,
            &prompt,
            &BeamConfig {
                width: 3,
                max_new: 6,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(a, b);
        // a beam as wide as the vocabulary is exhaustive for 2 tokens: compare with brute force
        let two = beam_search(
            &m,
            &prompt,
            &BeamConfig {
                width: 20,
                max_new: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let mut st = Model::new_state(&m);
        let l1 = log_softmax(&feed_last(&m, &mut st, &prompt).unwrap());
        let mut best = (f64::NEG_INFINITY, vec![]);
        for t1 in 0..20u32 {
            let mut s2 = st.fork().unwrap();
            let l2 = log_softmax(&feed_last(&m, &mut s2, &[t1]).unwrap());
            for t2 in 0..20u32 {
                let v = l1[t1 as usize] + l2[t2 as usize];
                if v > best.0 {
                    best = (v, vec![t1, t2]);
                }
            }
        }
        assert_eq!(two[0].tokens, best.1);
        assert!((two[0].logprob - best.0).abs() < 1e-6);
    }

    #[test]
    fn length_penalty_changes_the_ranking_not_the_logprobs() {
        let m = model();
        let prompt = lcg(5, 4, 20);
        let free = greedy_decode(&m, &prompt, 10, &[]).unwrap();
        let stop = vec![free.tokens[2]];
        let flat = beam_search(
            &m,
            &prompt,
            &BeamConfig {
                width: 4,
                max_new: 10,
                stop: stop.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        let pen = beam_search(
            &m,
            &prompt,
            &BeamConfig {
                width: 4,
                max_new: 10,
                stop,
                length_penalty: 1.5,
                ..Default::default()
            },
        )
        .unwrap();
        for h in flat.iter().chain(&pen) {
            assert!(h.logprob <= 0.0);
        }
        for w in pen.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        // a longer hypothesis is divided by a bigger penalty, so its score moves toward zero
        let h = &pen[0];
        assert!(h.score >= h.logprob - 1e-12);
    }

    #[test]
    fn min_new_blocks_early_stop_and_context_limit_is_respected() {
        let m = model();
        let prompt = lcg(9, 4, 20);
        let free = greedy_decode(&m, &prompt, 10, &[]).unwrap();
        let stop = vec![free.tokens[0]];
        let b = beam_search(
            &m,
            &prompt,
            &BeamConfig {
                width: 2,
                max_new: 8,
                stop,
                min_new: 3,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(b.iter().all(|h| h.tokens.len() >= 3 || !h.stopped));
        // the context is 32 tokens: a long request is cut, not an error
        let long = beam_search(
            &m,
            &lcg(2, 28, 20),
            &BeamConfig {
                width: 2,
                max_new: 100,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(long.iter().all(|h| 28 + h.tokens.len() <= 33));
        assert!(beam_search(&m, &[], &BeamConfig::default()).is_err());
        assert!(beam_search(&m, &lcg(1, 40, 20), &BeamConfig::default()).is_err());
    }

    #[test]
    fn best_of_n_and_vote() {
        let m = model();
        let prompt = lcg(13, 4, 20);
        let cfg = SampleConfig {
            max_new: 6,
            sampling: SamplingCfg::sampled(1.0, 0, 1.0),
            seed: 42,
            ..Default::default()
        };
        let a = sample_n(&m, &prompt, 8, &cfg).unwrap();
        let b = sample_n(&m, &prompt, 8, &cfg).unwrap();
        assert_eq!(a, b, "same seed, same samples");
        let fewer = sample_n(&m, &prompt, 3, &cfg).unwrap();
        assert_eq!(fewer, a[..3].to_vec(), "samples do not depend on n");
        assert!(
            a.windows(2).any(|w| w[0].tokens != w[1].tokens),
            "samples differ"
        );
        let best = best_of_n(&a, &SumLogprob).unwrap();
        assert!(a.iter().all(|s| s.logprob <= best.logprob));
        let by_len = best_of_n(&a, &|s: &Sample| s.tokens.len() as f64).unwrap();
        assert!(a.iter().all(|s| s.tokens.len() <= by_len.tokens.len()));
        assert!(best_of_n(&[], &SumLogprob).is_none());
        // NaN scores never win
        let nan = best_of_n(&a, &|s: &Sample| {
            if s.tokens[0].is_multiple_of(2) {
                f64::NAN
            } else {
                1.0
            }
        });
        assert!(nan.is_none_or(|s| s.tokens[0] % 2 == 1) || a.iter().all(|s| s.tokens[0] % 2 == 0));
        // greedy sampling is the greedy decode
        let g = sample_n(
            &m,
            &prompt,
            2,
            &SampleConfig {
                sampling: SamplingCfg::greedy(),
                max_new: 6,
                ..Default::default()
            },
        )
        .unwrap();
        let gd = greedy_decode(&m, &prompt, 6, &[]).unwrap();
        assert_eq!(g[0].tokens, gd.tokens);
        assert_eq!(g[0], g[1]);
    }

    #[test]
    fn sample_logprobs_are_the_models() {
        let m = model();
        let prompt = lcg(21, 3, 20);
        let cfg = SampleConfig {
            max_new: 5,
            sampling: SamplingCfg::sampled(0.7, 5, 0.9),
            seed: 5,
            ..Default::default()
        };
        for s in sample_n(&m, &prompt, 4, &cfg).unwrap() {
            let mut st = Model::new_state(&m);
            let mut all = prompt.clone();
            all.extend(&s.tokens);
            let lg = SeqModel::logits_all(&m, &mut st, &all).unwrap();
            let sum: f64 = s
                .tokens
                .iter()
                .enumerate()
                .map(|(i, &t)| log_softmax(&lg[prompt.len() - 1 + i])[t as usize])
                .sum();
            assert!((sum - s.logprob).abs() < 1e-4);
        }
    }

    fn mk(tokens: &[u32], lp: f64) -> Sample {
        Sample {
            tokens: tokens.to_vec(),
            logprob: lp,
            stopped: true,
        }
    }

    #[test]
    fn majority_vote_orders_by_count_then_score_then_appearance() {
        let s = vec![
            mk(&[1], -3.0),
            mk(&[2], -1.0),
            mk(&[1], -2.0),
            mk(&[9], -0.1),
            mk(&[2], -5.0),
            mk(&[1], -4.0),
        ];
        let key = |x: &Sample| {
            if x.tokens[0] == 9 {
                None
            } else {
                Some(format!("a{}", x.tokens[0]))
            }
        };
        let v = self_consistency(&s, &key, Some(&SumLogprob));
        assert_eq!(v.len(), 2);
        assert_eq!((v[0].answer.as_str(), v[0].count, v[0].best), ("a1", 3, 2));
        assert_eq!(v[1].members, vec![1, 4]);
        assert!((agreement(&v) - 0.6).abs() < 1e-12);
        // tie on count: the better-scored answer wins, then the earlier one
        let t = vec![mk(&[1], -3.0), mk(&[2], -1.0)];
        let v = self_consistency(
            &t,
            &|x: &Sample| Some(x.tokens[0].to_string()),
            Some(&SumLogprob),
        );
        assert_eq!(v[0].answer, "2");
        let v = self_consistency(&t, &|x: &Sample| Some(x.tokens[0].to_string()), None);
        assert_eq!(v[0].answer, "1");
        assert!(self_consistency(&[], &key, None).is_empty());
        assert_eq!(agreement(&[]), 0.0);
        assert!(self_consistency(&s, &|_| None, None).is_empty());
    }

    #[test]
    fn answer_normalisation() {
        assert_eq!(normalise_answer("  „Bucureşti”.  "), "bucurești");
        assert_eq!(
            normalise_answer("The   Answer\nis (42)!"),
            "the answer is (42"
        );
        assert_eq!(normalise_answer("Ţara"), "țara");
        assert_eq!(normalise_answer("..."), "");
        assert_eq!(
            last_number("so the total is 1,234.50 lei."),
            Some("1234.5".into())
        );
        assert_eq!(last_number("x = -7"), Some("-7".into()));
        assert_eq!(last_number("3.0 then 4.0"), Some("4".into()));
        assert_eq!(last_number("no digits"), None);
        assert_eq!(last_number("007"), Some("7".into()));
        assert_eq!(last_number("0.50"), Some("0.5".into()));
        assert_eq!(last_number("a-b 5"), Some("5".into()));
        assert_eq!(last_number("1,000 and 12"), Some("12".into()));
        assert_eq!(last_number("raspuns: 3,5"), Some("3.5".into()));
        assert_eq!(last_number("total 12,75 lei"), Some("12.75".into()));
        assert_eq!(last_number("1,234,567"), Some("1234567".into()));
    }

    #[test]
    fn lookup_greedy_equals_plain_greedy() {
        let m = model();
        let mut drafted = 0;
        for seed in 0..10u64 {
            let mut prompt = lcg(seed, 5, 20);
            let rep = prompt.clone();
            prompt.extend(&rep); // a repeated passage: lookup has something to find
            let cfg = LookupConfig {
                max_new: 18,
                ..Default::default()
            };
            let a = lookup_generate(&m, &prompt, &cfg).unwrap();
            let b = plain_generate(&m, &prompt, &cfg).unwrap();
            assert_eq!(a.tokens, b.tokens, "seed {seed}");
            assert_eq!(a.stop, b.stop);
            let g = greedy_decode(&m, &prompt, 18, &[]).unwrap();
            assert_eq!(a.tokens, g.tokens, "seed {seed} vs greedy");
            assert!(a.stats.passes <= b.stats.passes);
            drafted += a.stats.drafted;
        }
        assert!(drafted > 0, "the lookup never proposed anything");
    }

    #[test]
    fn lookup_greedy_equals_plain_greedy_on_quantised_weights() {
        use crate::model::tests::{block_model, requantise_to};
        use crate::quant::DType;
        for dt in [DType::Q8_0, DType::Q4_0, DType::Q4K] {
            let m = requantise_to(&block_model(), dt);
            for seed in 0..4u64 {
                let mut prompt = lcg(seed, 5, 40);
                let rep = prompt.clone();
                prompt.extend(&rep);
                let cfg = LookupConfig {
                    max_new: 14,
                    ..Default::default()
                };
                let a = lookup_generate(&m, &prompt, &cfg).unwrap();
                let b = plain_generate(&m, &prompt, &cfg).unwrap();
                assert_eq!(a.tokens, b.tokens, "{dt:?} seed {seed}");
            }
        }
    }

    #[test]
    fn logits_last_equals_the_last_row_of_logits_all_for_any_prompt_length() {
        let m = model();
        for n in [1usize, 2, 7, 31] {
            let toks = lcg(n as u64, n, 20);
            let mut s1 = m.new_state();
            let mut s2 = m.new_state();
            let all = SeqModel::logits_all(&m, &mut s1, &toks).unwrap();
            let last = SeqModel::logits_last(&m, &mut s2, &toks).unwrap();
            assert_eq!(all.len(), n, "one row per token from logits_all");
            assert_eq!(&last, all.last().unwrap(), "n={n}");
        }
    }

    #[test]
    fn lookup_saves_passes_when_the_text_repeats() {
        let m = model();
        let prompt = lcg(4, 6, 20);
        // let the model write its own (looping) continuation, then ask it again with that as context
        let first = plain_generate(
            &m,
            &prompt,
            &LookupConfig {
                max_new: 14,
                ..Default::default()
            },
        )
        .unwrap();
        let mut again = prompt.clone();
        again.extend(&first.tokens);
        let cfg = LookupConfig {
            max_new: 8,
            ..Default::default()
        };
        let a = lookup_generate(&m, &again, &cfg).unwrap();
        let b = plain_generate(&m, &again, &cfg).unwrap();
        assert_eq!(a.tokens, b.tokens);
        assert!(a.stats.emitted == 8 && a.stats.passes <= 8, "{:?}", a.stats);
    }

    #[test]
    fn lookup_respects_stop_tokens_and_max_new() {
        let m = model();
        let prompt = lcg(8, 6, 20);
        let free = plain_generate(
            &m,
            &prompt,
            &LookupConfig {
                max_new: 10,
                ..Default::default()
            },
        )
        .unwrap();
        let stop = vec![free.tokens[4]];
        let cfg = LookupConfig {
            max_new: 10,
            stop: stop.clone(),
            ..Default::default()
        };
        let a = lookup_generate(&m, &prompt, &cfg).unwrap();
        let b = plain_generate(&m, &prompt, &cfg).unwrap();
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.stop, spec::StopReason::StopToken);
        assert!(a.tokens.len() <= 5 && stop.contains(a.tokens.last().unwrap()));
        let zero = lookup_generate(
            &m,
            &prompt,
            &LookupConfig {
                max_new: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(zero.tokens.is_empty());
        assert!(lookup_generate(&m, &[], &LookupConfig::default()).is_err());
        assert!(lookup_generate(&m, &[999], &LookupConfig::default()).is_err());
    }

    #[test]
    fn lookup_context_limit_is_reported_not_overrun() {
        let m = model();
        let prompt = lcg(6, 30, 20);
        let o = lookup_generate(
            &m,
            &prompt,
            &LookupConfig {
                max_new: 50,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(30 + o.tokens.len() <= 33);
        assert_eq!(o.stop, spec::StopReason::ContextLimit);
        assert!(lookup_generate(&m, &lcg(1, 50, 20), &LookupConfig::default()).is_err());
    }

    #[test]
    fn corpus_proposals_come_from_reference_documents() {
        let doc: Vec<u32> = vec![10, 11, 12, 13, 14, 15, 16, 17];
        let mut d = Lookahead::new(2, 3, &[doc], 4);
        let mut rng = Rng::new(0);
        let got = d.draft(&[1, 2, 10, 11], 3, &SamplingCfg::greedy(), &mut rng);
        assert_eq!(got.tokens, vec![12, 13, 14]);
        assert!(
            d.draft(&[1, 2, 3], 3, &SamplingCfg::greedy(), &mut rng)
                .tokens
                .is_empty()
        );
        // in-context matches win over the corpus
        let got = d.draft(
            &[10, 11, 99, 98, 10, 11],
            2,
            &SamplingCfg::greedy(),
            &mut rng,
        );
        assert_eq!(got.tokens, vec![99, 98]);
    }

    #[test]
    fn lookup_with_a_corpus_still_matches_greedy() {
        let m = model();
        let prompt = lcg(2, 6, 20);
        let reference = plain_generate(
            &m,
            &prompt,
            &LookupConfig {
                max_new: 16,
                ..Default::default()
            },
        )
        .unwrap()
        .tokens;
        let cfg = LookupConfig {
            max_new: 16,
            corpus: vec![reference.clone()],
            ..Default::default()
        };
        let a = lookup_generate(&m, &prompt, &cfg).unwrap();
        assert_eq!(a.tokens, reference);
        assert!(
            a.stats.drafted > 0,
            "the reference text should have been used"
        );
        assert!(a.stats.accepted > 0);
    }

    #[test]
    fn lookup_sampling_is_seeded_and_follows_the_distribution() {
        let m = model();
        let mut prompt = lcg(3, 4, 20);
        let rep = prompt.clone();
        prompt.extend(&rep);
        let s = SamplingCfg::sampled(1.0, 2, 1.0);
        let cfg = |seed| LookupConfig {
            max_new: 2,
            sampling: s,
            seed,
            n_min: 1,
            ..Default::default()
        };
        assert_eq!(
            lookup_generate(&m, &prompt, &cfg(5)).unwrap().tokens,
            lookup_generate(&m, &prompt, &cfg(5)).unwrap().tokens
        );
        // empirical law of the first two tokens, speculative against plain (top-2 sampling: 6 outcomes)
        let n = 700u64;
        let (mut a, mut b): (HashMap<Vec<u32>, f64>, HashMap<Vec<u32>, f64>) =
            (HashMap::new(), HashMap::new());
        for seed in 0..n {
            *a.entry(lookup_generate(&m, &prompt, &cfg(seed)).unwrap().tokens)
                .or_default() += 1.0 / n as f64;
            *b.entry(
                plain_generate(&m, &prompt, &cfg(seed + 100_000))
                    .unwrap()
                    .tokens,
            )
            .or_default() += 1.0 / n as f64;
        }
        let keys: std::collections::HashSet<_> = a.keys().chain(b.keys()).cloned().collect();
        let tv: f64 = keys
            .iter()
            .map(|k| (a.get(k).unwrap_or(&0.0) - b.get(k).unwrap_or(&0.0)).abs())
            .sum::<f64>()
            / 2.0;
        assert!(tv < 0.12, "total variation {tv}");
    }

    #[test]
    fn tree_branches_share_work_and_match_a_fresh_run() {
        let m = model();
        let prompt = lcg(17, 5, 20);
        let mut t = Tree::new(&m, &prompt).unwrap();
        let kids = t.expand_top(t.root(), 3).unwrap();
        assert_eq!(kids.len(), 3);
        let g = greedy_decode(&m, &prompt, 1, &[]).unwrap();
        assert_eq!(t.path(kids[0]).unwrap().last(), g.tokens.last());
        // the node's logits equal a from-scratch run over its whole path
        let mut st = Model::new_state(&m);
        let fresh = feed_last(&m, &mut st, t.path(kids[1]).unwrap()).unwrap();
        for (x, y) in t.logits(kids[1]).unwrap().iter().zip(&fresh) {
            assert!((x - y).abs() < 1e-4);
        }
        // log-prob of a child = the model's log-prob of its token
        let lsm = log_softmax(t.logits(t.root()).unwrap());
        let tok = *t.path(kids[2]).unwrap().last().unwrap();
        assert!((t.logprob(kids[2]).unwrap() - lsm[tok as usize]).abs() < 1e-9);
        assert_eq!(t.children(t.root()), &kids[..]);
        assert_eq!(t.parent(kids[0]), Some(0));
    }

    #[test]
    fn tree_fork_at_a_position_and_rebuild_after_release() {
        let m = model();
        let prompt = lcg(19, 6, 20);
        let mut t = Tree::new(&m, &prompt).unwrap();
        let tail = t.extend_greedy(t.root(), 5, &[]).unwrap();
        assert_eq!(t.path(tail).unwrap().len(), 11);
        let g = greedy_decode(&m, &prompt, 5, &[]).unwrap();
        assert_eq!(&t.path(tail).unwrap()[6..], &g.tokens[..]);
        assert!((t.logprob(tail).unwrap() - g.logprob).abs() < 1e-5);
        // fork inside the generated part and add a different token
        let alt = (t.path(tail).unwrap()[8] + 1) % 20;
        let b = t.branch_at(tail, 8, &[alt]).unwrap();
        let mut want = t.path(tail).unwrap()[..8].to_vec();
        want.push(alt);
        assert_eq!(t.path(b).unwrap(), &want[..]);
        let mut st = Model::new_state(&m);
        let fresh = feed_last(&m, &mut st, &want).unwrap();
        for (x, y) in t.logits(b).unwrap().iter().zip(&fresh) {
            assert!((x - y).abs() < 1e-4);
        }
        // the original branch is untouched by the fork
        let mut st2 = Model::new_state(&m);
        let fresh2 = feed_last(&m, &mut st2, t.path(tail).unwrap()).unwrap();
        for (x, y) in t.logits(tail).unwrap().iter().zip(&fresh2) {
            assert!((x - y).abs() < 1e-4);
        }
        // drop a cache, branch again: the cache is rebuilt by replay
        t.release_state(tail);
        assert!(!t.has_state(tail));
        let c = t.branch(tail, &[1, 2]).unwrap();
        let mut want = t.path(tail).unwrap().to_vec();
        want.extend([1, 2]);
        let mut st3 = Model::new_state(&m);
        let fresh3 = feed_last(&m, &mut st3, &want).unwrap();
        for (x, y) in t.logits(c).unwrap().iter().zip(&fresh3) {
            assert!((x - y).abs() < 1e-4);
        }
        // best leaf by a custom score
        let best = t
            .best_leaf(&|path, lp| lp - 0.01 * path.len() as f64)
            .unwrap();
        assert!(t.leaves().contains(&best));
    }

    #[test]
    fn tree_rejects_bad_requests() {
        let m = model();
        let mut t = Tree::new(&m, &lcg(1, 4, 20)).unwrap();
        assert!(t.branch(7, &[1]).is_err());
        assert!(t.branch(0, &[]).is_err());
        assert!(t.branch(0, &[999]).is_err());
        assert!(t.branch_at(0, 0, &[1]).is_err());
        assert!(t.branch_at(0, 9, &[1]).is_err());
        assert!(t.path(5).is_err());
        assert!(Tree::new(&m, &[]).is_err());
        let long = t.branch(0, &lcg(2, 28, 20)).unwrap();
        assert!(t.branch(long, &[1]).is_err(), "past the context");
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn works_on_a_sliding_window_model() {
        use crate::model::tests::{base_spec, model_of};
        let mut sp = base_spec(4, 2, false);
        sp.sliding_window = 6;
        sp.sliding_layers = vec![true, false, true];
        let m = model_of(sp, false);
        let prompt = lcg(2, 12, 20);
        let a = lookup_generate(
            &m,
            &prompt,
            &LookupConfig {
                max_new: 8,
                ..Default::default()
            },
        )
        .unwrap();
        let g = greedy_decode(&m, &prompt, 8, &[]).unwrap();
        assert_eq!(a.tokens, g.tokens);
        let b = beam_search(
            &m,
            &prompt,
            &BeamConfig {
                width: 3,
                max_new: 6,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(b.len(), 3);
    }
}
