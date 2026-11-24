//! Measuring a model: perplexity, KL divergence against a reference, multiple-choice accuracy and
//! throughput benchmarks.
//!
//! Everything is generic over a closure `FnMut(&[u32]) -> Vec<Vec<f32>>` that returns one logit row per
//! input position (row `i` predicts token `i + 1`), so it works with any model or backend and does not
//! depend on `model.rs`. Probabilities are computed in f64 with a max-shifted log-sum-exp, so huge
//! logits neither overflow nor lose the tail.
#![allow(clippy::collapsible_if)]
use serde_json::{json, Value as Json};

#[derive(Debug, PartialEq)]
pub enum EvalError {
    /// Nothing to score, or a parameter that makes no sense (zero context, zero reps).
    Empty(String),
    /// The model returned something that does not fit the input.
    Shape(String),
    /// Bad dataset or non-finite numbers.
    Format(String),
}
impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { EvalError::Empty(s) | EvalError::Shape(s) | EvalError::Format(s) => write!(f, "{s}") }
    }
}
impl std::error::Error for EvalError {}
fn shape<T>(s: impl Into<String>) -> Result<T, EvalError> { Err(EvalError::Shape(s.into())) }
fn empty<T>(s: impl Into<String>) -> Result<T, EvalError> { Err(EvalError::Empty(s.into())) }
fn format<T>(s: impl Into<String>) -> Result<T, EvalError> { Err(EvalError::Format(s.into())) }

// ----------------------------------------------------------------------------- numerics

/// `ln Σ exp(l)` computed stably.
fn log_sum_exp(l: &[f32]) -> Result<f64, EvalError> {
    if l.is_empty() { return shape("empty logit row"); }
    if l.iter().any(|v| v.is_nan()) { return format("logits contain NaN"); }
    let m = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    if m == f64::NEG_INFINITY { return format("every logit is -infinity"); }
    if m == f64::INFINITY { return format("logits contain +infinity"); }
    Ok(m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln())
}

/// `ln softmax(l)[t]`.
pub fn log_prob(l: &[f32], t: usize) -> Result<f64, EvalError> {
    if t >= l.len() { return shape(format!("token {t} is outside a vocabulary of {}", l.len())); }
    Ok(l[t] as f64 - log_sum_exp(l)?)
}

/// `softmax(l)` in f64.
pub fn softmax(l: &[f32]) -> Result<Vec<f64>, EvalError> {
    let z = log_sum_exp(l)?;
    Ok(l.iter().map(|&v| (v as f64 - z).exp()).collect())
}

fn checked_logits(window: &[u32], rows: &[Vec<f32>]) -> Result<(), EvalError> {
    if rows.len() != window.len() { return shape(format!("model returned {} logit rows for {} tokens", rows.len(), window.len())); }
    Ok(())
}

// ----------------------------------------------------------------------------- perplexity

#[derive(Clone, Debug, PartialEq)]
pub struct PerplexityResult {
    pub ppl: f64,
    pub nll_mean: f64,
    pub tokens_scored: usize,
    pub chunks: usize,
    /// Standard error of `ppl` (delta method from the per-token spread).
    pub stderr: f64,
}
impl PerplexityResult {
    pub fn to_json(&self) -> Json {
        json!({"perplexity": self.ppl, "nll_mean": self.nll_mean, "tokens_scored": self.tokens_scored, "chunks": self.chunks, "stderr": self.stderr})
    }
}

struct Acc { n: usize, mean: f64, m2: f64, chunks: usize }
impl Acc {
    fn new() -> Acc { Acc { n: 0, mean: 0.0, m2: 0.0, chunks: 0 } }
    fn add(&mut self, nll: f64) { self.n += 1; let d = nll - self.mean; self.mean += d / self.n as f64; self.m2 += d * (nll - self.mean); }
    fn finish(self) -> Result<PerplexityResult, EvalError> {
        if self.n == 0 { return empty("no tokens were scored (need at least two tokens)"); }
        let n = self.n as f64;
        let mean = self.mean;
        let var = if self.n > 1 { self.m2 / (n - 1.0) } else { 0.0 };
        let ppl = mean.exp();
        Ok(PerplexityResult { ppl, nll_mean: mean, tokens_scored: self.n, chunks: self.chunks, stderr: ppl * (var / n).sqrt() })
    }
}

fn score_window(forward: &mut impl FnMut(&[u32]) -> Vec<Vec<f32>>, window: &[u32], first: usize, acc: &mut Acc) -> Result<(), EvalError> {
    let rows = forward(window);
    checked_logits(window, &rows)?;
    for i in first.max(1)..window.len() {
        acc.add(-log_prob(&rows[i - 1], window[i] as usize)?);
    }
    acc.chunks += 1;
    Ok(())
}

/// Sliding-window perplexity as Hugging Face computes it: windows of `ctx_len` tokens advance by
/// `stride`, and only tokens not already scored by the previous window are counted. The very first
/// token has no prediction and is never scored.
pub fn perplexity(mut forward: impl FnMut(&[u32]) -> Vec<Vec<f32>>, tokens: &[u32], ctx_len: usize, stride: usize) -> Result<PerplexityResult, EvalError> {
    if ctx_len < 2 || stride == 0 || stride > ctx_len { return empty("ctx_len must be at least 2 and stride in 1..=ctx_len"); }
    let (n, mut prev_end, mut acc) = (tokens.len(), 0usize, Acc::new());
    let mut begin = 0;
    while begin < n {
        let end = (begin + ctx_len).min(n);
        let trg = end - prev_end;
        let window = &tokens[begin..end];
        score_window(&mut forward, window, window.len() - trg, &mut acc)?;
        prev_end = end;
        if end == n { break; }
        begin += stride;
    }
    acc.finish()
}

/// llama.cpp-style perplexity: independent, non-overlapping chunks of `ctx_len` tokens; with
/// `second_half_only` only the second half of each chunk is scored (every scored token then has at
/// least `ctx_len / 2` tokens of context).
pub fn perplexity_chunked(mut forward: impl FnMut(&[u32]) -> Vec<Vec<f32>>, tokens: &[u32], ctx_len: usize, second_half_only: bool) -> Result<PerplexityResult, EvalError> {
    if ctx_len < 2 { return empty("ctx_len must be at least 2"); }
    let mut acc = Acc::new();
    for chunk in tokens.chunks(ctx_len).filter(|c| c.len() == ctx_len || c.len() > 1) {
        let first = if second_half_only { chunk.len() / 2 } else { 1 };
        score_window(&mut forward, chunk, first, &mut acc)?;
    }
    acc.finish()
}

// ----------------------------------------------------------------------------- KL divergence

#[derive(Clone, Debug, PartialEq)]
pub struct KlStats {
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub p90: f64,
    pub p99: f64,
    /// Share of positions where both models pick the same most likely token.
    pub top1_agreement: f64,
    /// Mean Jaccard overlap of the two nucleus sets (smallest sets reaching probability `p`).
    pub nucleus_jaccard: Option<f64>,
    /// Root mean square of `p_ref[target] - p_test[target]`.
    pub rms_dp: Option<f64>,
}
impl KlStats {
    pub fn to_json(&self) -> Json {
        json!({"n": self.n, "kl_mean": self.mean, "kl_median": self.median, "kl_min": self.min, "kl_max": self.max,
               "kl_p90": self.p90, "kl_p99": self.p99, "top1_agreement": self.top1_agreement,
               "nucleus_jaccard": self.nucleus_jaccard, "rms_dp": self.rms_dp})
    }
}

fn argmax(p: &[f64]) -> usize { p.iter().enumerate().fold(0, |b, (i, &v)| if v > p[b] { i } else { b }) }

fn nucleus(p: &[f64], top_p: f64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    let (mut cum, mut out) = (0.0, vec![]);
    for i in idx { out.push(i); cum += p[i]; if cum >= top_p { break; } }
    out.sort_unstable();
    out
}

/// Nearest-rank percentile of an ascending slice.
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    let rank = ((pct / 100.0 * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

/// `KL(ref ‖ test)` per position, summarised. `targets` (the actual next tokens) enables `rms_dp`;
/// `nucleus_p` enables the nucleus overlap.
pub fn kl_divergence(reference: &[Vec<f32>], test: &[Vec<f32>], targets: Option<&[u32]>, nucleus_p: Option<f64>) -> Result<KlStats, EvalError> {
    if reference.len() != test.len() { return shape(format!("{} reference rows but {} test rows", reference.len(), test.len())); }
    if reference.is_empty() { return empty("nothing to compare"); }
    if let Some(t) = targets { if t.len() != reference.len() { return shape("targets and logits differ in length"); } }
    if let Some(p) = nucleus_p { if !(p > 0.0 && p <= 1.0) { return format("nucleus_p must be in (0, 1]"); } }
    let (mut kls, mut agree, mut jac, mut dp2) = (Vec::with_capacity(reference.len()), 0usize, 0.0f64, 0.0f64);
    for (i, (r, t)) in reference.iter().zip(test).enumerate() {
        if r.len() != t.len() { return shape(format!("row {i}: vocabularies differ ({} vs {})", r.len(), t.len())); }
        let (zr, zt) = (log_sum_exp(r)?, log_sum_exp(t)?);
        let mut kl = 0.0;
        for (&a, &b) in r.iter().zip(t) {
            let lp = a as f64 - zr;
            let p = lp.exp();
            if p > 0.0 { kl += p * (lp - (b as f64 - zt)); }
        }
        kls.push(kl.max(0.0));
        let (pr, pt) = (softmax(r)?, softmax(t)?);
        if argmax(&pr) == argmax(&pt) { agree += 1; }
        if let Some(tp) = nucleus_p {
            let (a, b) = (nucleus(&pr, tp), nucleus(&pt, tp));
            let inter = a.iter().filter(|x| b.binary_search(x).is_ok()).count();
            jac += inter as f64 / (a.len() + b.len() - inter) as f64;
        }
        if let Some(tg) = targets {
            let k = tg[i] as usize;
            if k >= pr.len() { return shape(format!("target {k} is outside the vocabulary")); }
            dp2 += (pr[k] - pt[k]).powi(2);
        }
    }
    let n = kls.len();
    let mean = kls.iter().sum::<f64>() / n as f64;
    kls.sort_by(f64::total_cmp);
    Ok(KlStats {
        n, mean, median: percentile(&kls, 50.0), min: kls[0], max: kls[n - 1],
        p90: percentile(&kls, 90.0), p99: percentile(&kls, 99.0),
        top1_agreement: agree as f64 / n as f64,
        nucleus_jaccard: nucleus_p.map(|_| jac / n as f64),
        rms_dp: targets.map(|_| (dp2 / n as f64).sqrt()),
    })
}

// ----------------------------------------------------------------------------- multiple choice

/// One candidate answer: the context it follows and its continuation, already tokenised.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice { pub ctx: Vec<u32>, pub cont: Vec<u32>, pub chars: usize }

#[derive(Clone, Debug, PartialEq)]
pub struct McItem { pub choices: Vec<Choice>, pub label: usize }

/// How a continuation's log-likelihood is normalised before choosing the best answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm { Sum, PerToken, PerChar }

/// Log-likelihood of the continuation given its context, normalised.
pub fn score_choice(forward: &mut impl FnMut(&[u32]) -> Vec<Vec<f32>>, c: &Choice, norm: Norm) -> Result<f64, EvalError> {
    if c.ctx.is_empty() { return empty("a choice needs at least one context token"); }
    if c.cont.is_empty() { return empty("a choice needs at least one continuation token"); }
    let mut all = c.ctx.clone();
    all.extend(&c.cont);
    let rows = forward(&all);
    checked_logits(&all, &rows)?;
    let mut ll = 0.0;
    for (k, &t) in c.cont.iter().enumerate() { ll += log_prob(&rows[c.ctx.len() + k - 1], t as usize)?; }
    Ok(match norm {
        Norm::Sum => ll,
        Norm::PerToken => ll / c.cont.len() as f64,
        Norm::PerChar => ll / c.chars.max(1) as f64,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct McResult { pub accuracy: f64, pub stderr: f64, pub n: usize, pub correct: usize }
impl McResult {
    pub fn to_json(&self) -> Json { json!({"accuracy": self.accuracy, "stderr": self.stderr, "n": self.n, "correct": self.correct}) }
}

/// Pick the highest-scoring choice of each item (ties go to the first) and report accuracy with its
/// binomial standard error.
pub fn evaluate_multiple_choice(mut forward: impl FnMut(&[u32]) -> Vec<Vec<f32>>, items: &[McItem], norm: Norm) -> Result<McResult, EvalError> {
    if items.is_empty() { return empty("no items"); }
    let mut correct = 0;
    for (i, it) in items.iter().enumerate() {
        if it.choices.is_empty() || it.label >= it.choices.len() { return format(format!("item {i}: label {} with {} choices", it.label, it.choices.len())); }
        let mut best = (f64::NEG_INFINITY, 0);
        for (k, c) in it.choices.iter().enumerate() {
            let s = score_choice(&mut forward, c, norm)?;
            if k == 0 || s > best.0 { best = (s, k); }
        }
        if best.1 == it.label { correct += 1; }
    }
    let n = items.len() as f64;
    let acc = correct as f64 / n;
    Ok(McResult { accuracy: acc, stderr: (acc * (1.0 - acc) / n).sqrt(), n: items.len(), correct })
}

fn label_of(v: &Json) -> Option<usize> {
    match v {
        Json::Number(n) => n.as_u64().map(|x| x as usize),
        Json::String(s) => {
            let t = s.trim();
            if let Ok(n) = t.parse::<usize>() { return Some(n); }
            let mut ch = t.chars();
            match (ch.next(), ch.next()) { (Some(c), None) if c.is_ascii_alphabetic() => Some(c.to_ascii_uppercase() as usize - 'A' as usize), _ => None }
        }
        _ => None,
    }
}

/// Read a multiple-choice dataset: a JSON array (or `{"data": [...]}`) of items in any of these shapes
/// — generic `{ctx|question, choices|endings, label|answer}`, HellaSwag (`ctx`, `endings`, `label`),
/// MMLU (`question`, `choices`, `answer` as index or letter) and Winogrande (`sentence` with a `_`,
/// `option1`, `option2`, `answer` "1" or "2"). `encode` turns text into token ids.
pub fn load_mc_json(json: &str, encode: impl Fn(&str) -> Vec<u32>) -> Result<Vec<McItem>, EvalError> {
    let v: Json = serde_json::from_str(json).map_err(|e| EvalError::Format(format!("dataset: {e}")))?;
    let arr = match &v {
        Json::Array(a) => a,
        Json::Object(o) => match o.get("data").or_else(|| o.get("items")) { Some(Json::Array(a)) => a, _ => return format("dataset is an object without a `data` array") },
        _ => return format("dataset must be an array"),
    };
    if arr.len() > 10_000_000 { return format("dataset is too large"); }
    let mut out = Vec::with_capacity(arr.len());
    for (i, it) in arr.iter().enumerate() {
        let o = it.as_object().ok_or_else(|| EvalError::Format(format!("item {i} is not an object")))?;
        let text = |k: &str| o.get(k).and_then(Json::as_str);
        let item = if let (Some(sent), Some(o1), Some(o2)) = (text("sentence"), text("option1"), text("option2")) {
            let (pre, post) = sent.split_once('_').ok_or_else(|| EvalError::Format(format!("item {i}: sentence has no `_`")))?;
            let post = post.trim_start();
            if post.is_empty() { return format(format!("item {i}: nothing follows the blank")); }
            let label = o.get("answer").and_then(label_of).and_then(|a| a.checked_sub(1)).filter(|a| *a < 2)
                .ok_or_else(|| EvalError::Format(format!("item {i}: answer must be 1 or 2")))?;
            let cont = encode(&format!(" {post}"));
            let choices = [o1, o2].iter().map(|opt| Choice { ctx: encode(&format!("{pre}{opt}")), cont: cont.clone(), chars: post.chars().count() + 1 }).collect();
            McItem { choices, label }
        } else {
            let ctx = text("ctx").or_else(|| text("question")).or_else(|| text("query")).ok_or_else(|| EvalError::Format(format!("item {i}: no ctx/question")))?;
            let opts = o.get("endings").or_else(|| o.get("choices")).and_then(Json::as_array).ok_or_else(|| EvalError::Format(format!("item {i}: no endings/choices")))?;
            if opts.len() < 2 || opts.len() > 64 { return format(format!("item {i}: {} choices", opts.len())); }
            let label = o.get("label").or_else(|| o.get("answer")).and_then(label_of).filter(|l| *l < opts.len())
                .ok_or_else(|| EvalError::Format(format!("item {i}: missing or out-of-range label")))?;
            let c = encode(ctx);
            let mut choices = Vec::with_capacity(opts.len());
            for (k, op) in opts.iter().enumerate() {
                let s = op.as_str().ok_or_else(|| EvalError::Format(format!("item {i}: choice {k} is not text")))?;
                choices.push(Choice { ctx: c.clone(), cont: encode(&format!(" {s}")), chars: s.chars().count() + 1 });
            }
            McItem { choices, label }
        };
        if item.choices.iter().any(|c| c.ctx.is_empty() || c.cont.is_empty()) { return format(format!("item {i}: text encodes to no tokens")); }
        out.push(item);
    }
    Ok(out)
}

// ----------------------------------------------------------------------------- benchmark

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchCfg { pub prompt_tokens: usize, pub gen_tokens: usize, pub warmup: usize, pub reps: usize }
impl Default for BenchCfg { fn default() -> Self { BenchCfg { prompt_tokens: 512, gen_tokens: 128, warmup: 1, reps: 5 } } }

#[derive(Clone, Debug, PartialEq)]
pub struct Stats { pub n: usize, pub mean: f64, pub median: f64, pub stddev: f64, pub min: f64, pub max: f64 }
impl Stats {
    pub fn from(v: &[f64]) -> Option<Stats> {
        if v.is_empty() || v.iter().any(|x| !x.is_finite()) { return None; }
        let n = v.len();
        let mean = v.iter().sum::<f64>() / n as f64;
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        let median = if !n.is_multiple_of(2) { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 };
        let stddev = if n > 1 { (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt() } else { 0.0 };
        Some(Stats { n, mean, median, stddev, min: s[0], max: s[n - 1] })
    }
    pub fn to_json(&self) -> Json { json!({"n": self.n, "mean": self.mean, "median": self.median, "stddev": self.stddev, "min": self.min, "max": self.max}) }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchResult { pub cfg: BenchCfg, /// Prompt processing, tokens per second.
    pub pp: Stats, /// Token generation, tokens per second.
    pub tg: Stats }
impl BenchResult {
    pub fn to_json(&self) -> Json {
        json!({"prompt_tokens": self.cfg.prompt_tokens, "gen_tokens": self.cfg.gen_tokens, "reps": self.cfg.reps,
               "pp_tokens_per_s": self.pp.to_json(), "tg_tokens_per_s": self.tg.to_json()})
    }
    /// Two rows in the style of `llama-bench`: `pp512 | 1234.5 ± 12.3`.
    pub fn to_markdown(&self) -> String {
        format!("| test | t/s |\n| --- | ---: |\n| pp{} | {:.2} ± {:.2} |\n| tg{} | {:.2} ± {:.2} |\n",
            self.cfg.prompt_tokens, self.pp.mean, self.pp.stddev, self.cfg.gen_tokens, self.tg.mean, self.tg.stddev)
    }
}

/// Time prompt processing and generation. `reset` clears the model's state, `prefill(n)` processes an
/// `n`-token prompt, `decode()` produces one token, `clock()` returns seconds (injectable for tests).
pub fn bench(cfg: BenchCfg, mut reset: impl FnMut(), mut prefill: impl FnMut(usize), mut decode: impl FnMut(), mut clock: impl FnMut() -> f64) -> Result<BenchResult, EvalError> {
    if cfg.prompt_tokens == 0 || cfg.gen_tokens == 0 || cfg.reps == 0 { return empty("prompt_tokens, gen_tokens and reps must be positive"); }
    for _ in 0..cfg.warmup { reset(); prefill(cfg.prompt_tokens); for _ in 0..cfg.gen_tokens { decode(); } }
    let (mut pp, mut tg) = (Vec::with_capacity(cfg.reps), Vec::with_capacity(cfg.reps));
    for _ in 0..cfg.reps {
        reset();
        let t0 = clock();
        prefill(cfg.prompt_tokens);
        let t1 = clock();
        for _ in 0..cfg.gen_tokens { decode(); }
        let t2 = clock();
        let (a, b) = (t1 - t0, t2 - t1);
        if !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()) { return format("the clock did not advance"); }
        pp.push(cfg.prompt_tokens as f64 / a);
        tg.push(cfg.gen_tokens as f64 / b);
    }
    match (Stats::from(&pp), Stats::from(&tg)) { (Some(pp), Some(tg)) => Ok(BenchResult { cfg, pp, tg }), _ => format("no valid timings") }
}

// ----------------------------------------------------------------------------- report

/// Named results gathered into one JSON document or Markdown summary.
#[derive(Clone, Debug, Default)]
pub struct Report { pub title: String, pub sections: Vec<(String, Json)> }
impl Report {
    pub fn new(title: &str) -> Report { Report { title: title.to_string(), sections: vec![] } }
    pub fn add(&mut self, name: &str, value: Json) -> &mut Report { self.sections.push((name.to_string(), value)); self }
    pub fn to_json(&self) -> Json {
        let mut m = serde_json::Map::new();
        for (k, v) in &self.sections { m.insert(k.clone(), v.clone()); }
        json!({"title": self.title, "results": Json::Object(m)})
    }
    pub fn to_markdown(&self) -> String {
        let mut s = format!("# {}\n", self.title);
        for (name, v) in &self.sections {
            s.push_str(&format!("\n## {name}\n\n| metric | value |\n| --- | ---: |\n"));
            if let Some(o) = v.as_object() {
                for (k, x) in o {
                    let cell = match x { Json::Number(n) => n.as_f64().map_or(n.to_string(), |f| if f.fract() == 0.0 && f.abs() < 1e15 { format!("{f}") } else { format!("{f:.4}") }), Json::Null => "–".into(), other => other.to_string() };
                    s.push_str(&format!("| {k} | {cell} |\n"));
                }
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn uniform(v: usize) -> impl FnMut(&[u32]) -> Vec<Vec<f32>> { move |w: &[u32]| vec![vec![0.0; v]; w.len()] }
    /// Always puts overwhelming mass on the true next token (knows the text through `tokens`).
    fn oracle(v: usize) -> impl FnMut(&[u32]) -> Vec<Vec<f32>> {
        move |w: &[u32]| (0..w.len()).map(|i| { let mut r = vec![0.0; v]; if i + 1 < w.len() { r[w[i + 1] as usize] = 100.0; } r }).collect()
    }
    fn text(n: usize, v: u32) -> Vec<u32> { (0..n as u32).map(|i| (i * 7 + 3) % v).collect() }

    #[test]
    fn uniform_model_has_perplexity_equal_to_vocab_size() {
        let toks = text(100, 13);
        let r = perplexity(uniform(13), &toks, 32, 16).unwrap();
        assert!((r.ppl - 13.0).abs() < 1e-9, "{}", r.ppl);
        assert_eq!(r.tokens_scored, 99);
        assert!(r.stderr.abs() < 1e-9);
        let r = perplexity_chunked(uniform(13), &toks, 32, false).unwrap();
        assert!((r.ppl - 13.0).abs() < 1e-9);
    }

    #[test]
    fn oracle_has_perplexity_one() {
        let toks = text(80, 11);
        for stride in [20, 17, 1] {
            let r = perplexity(oracle(11), &toks, 40, stride).unwrap();
            assert!((r.ppl - 1.0).abs() < 1e-6, "stride {stride}: {}", r.ppl);
            assert_eq!(r.tokens_scored, 79, "every token but the first is scored exactly once");
        }
        let r = perplexity_chunked(oracle(11), &toks, 20, true).unwrap();
        assert!((r.ppl - 1.0).abs() < 1e-6);
        assert_eq!(r.tokens_scored, 4 * 10);
        assert_eq!(r.chunks, 4);
    }

    #[test]
    fn sliding_window_counts_each_token_once_even_with_a_ragged_tail() {
        let toks = text(53, 9);
        let seen = std::cell::RefCell::new(vec![]);
        let r = perplexity(|w: &[u32]| { seen.borrow_mut().push(w.len()); vec![vec![0.0; 9]; w.len()] }, &toks, 20, 7).unwrap();
        assert_eq!(r.tokens_scored, 52);
        assert!(seen.borrow().iter().all(|&l| l <= 20));
        assert!(perplexity(uniform(9), &toks, 1, 1).is_err());
        assert!(perplexity(uniform(9), &toks, 8, 0).is_err());
        assert!(perplexity(uniform(9), &toks, 8, 9).is_err());
        assert!(perplexity(uniform(9), &[1], 8, 4).is_err());
        assert!(perplexity(uniform(9), &[], 8, 4).is_err());
        // wrong number of rows from the model
        assert!(matches!(perplexity(|_: &[u32]| vec![vec![0.0; 9]], &toks, 8, 8), Err(EvalError::Shape(_))));
        // token outside the vocabulary
        assert!(perplexity(uniform(5), &[1, 9, 2], 8, 8).is_err());
    }

    #[test]
    fn huge_logits_do_not_overflow() {
        let l = vec![1.0e4f32; 8];
        assert!((log_prob(&l, 3).unwrap() + 8f64.ln()).abs() < 1e-9);
        let mut l = vec![-3.0e4f32; 5];
        l[2] = 3.0e4;
        assert!(log_prob(&l, 2).unwrap().abs() < 1e-9);
        assert!(log_prob(&l, 0).unwrap() < -1e4);
        let p = softmax(&[1000.0, 1000.0]).unwrap();
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert!(log_prob(&[f32::NAN, 0.0], 1).is_err());
        assert!(log_prob(&[f32::NEG_INFINITY; 3], 0).is_err());
        assert!(log_prob(&[0.0, 1.0], 2).is_err());
        assert!(log_prob(&[], 0).is_err());
    }

    #[test]
    fn kl_of_identical_models_is_zero_and_matches_a_hand_computed_case() {
        let rows: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, 1.0, -2.0, 0.5]).collect();
        let s = kl_divergence(&rows, &rows, None, Some(0.9)).unwrap();
        assert!(s.mean.abs() < 1e-12 && s.max.abs() < 1e-12);
        assert_eq!(s.top1_agreement, 1.0);
        assert!((s.nucleus_jaccard.unwrap() - 1.0).abs() < 1e-12);
        // p = [.5, .5], q = [.75, .25]
        let r = vec![vec![0.0f32, 0.0]];
        let t = vec![vec![3f32.ln(), 0.0]];
        let s = kl_divergence(&r, &t, Some(&[1]), None).unwrap();
        let want = 0.5 * (0.5f64 / 0.75).ln() + 0.5 * (0.5f64 / 0.25).ln();
        assert!((s.mean - want).abs() < 1e-6, "{} vs {want}", s.mean);
        assert!((s.rms_dp.unwrap() - 0.25).abs() < 1e-6);
        assert_eq!(s.top1_agreement, 1.0); // reference tie goes to index 0, which the test also prefers
    }

    #[test]
    fn kl_summary_orders_and_validates() {
        let r: Vec<Vec<f32>> = vec![vec![0.0, 0.0]; 10];
        let t: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.3, 0.0]).collect();
        let s = kl_divergence(&r, &t, None, None).unwrap();
        assert!(s.min <= s.median && s.median <= s.p90 && s.p90 <= s.p99 && s.p99 <= s.max);
        assert!(s.min.abs() < 1e-12 && s.max > 0.1);
        assert!(kl_divergence(&r, &t[..9], None, None).is_err());
        assert!(kl_divergence(&[], &[], None, None).is_err());
        assert!(kl_divergence(&r, &t, Some(&[0; 3]), None).is_err());
        assert!(kl_divergence(&r, &t, None, Some(0.0)).is_err());
        assert!(kl_divergence(&r, &t, None, Some(1.5)).is_err());
        assert!(kl_divergence(&[vec![0.0; 2]], &[vec![0.0; 3]], None, None).is_err());
        assert!(kl_divergence(&r, &t, Some(&[5; 10]), None).is_err());
    }

    /// A model that likes continuations made of token `fav`.
    fn fond_of(fav: u32, v: usize) -> impl FnMut(&[u32]) -> Vec<Vec<f32>> {
        move |w: &[u32]| (0..w.len()).map(|_| { let mut r = vec![0.0; v]; r[fav as usize] = 5.0; r }).collect()
    }

    #[test]
    fn multiple_choice_picks_the_best_continuation() {
        let ch = |cont: Vec<u32>| Choice { ctx: vec![1, 2], chars: cont.len(), cont };
        let items = vec![
            McItem { choices: vec![ch(vec![3]), ch(vec![7]), ch(vec![4])], label: 1 },
            McItem { choices: vec![ch(vec![7]), ch(vec![3])], label: 0 },
            McItem { choices: vec![ch(vec![3]), ch(vec![7])], label: 0 }, // wrong on purpose
        ];
        let r = evaluate_multiple_choice(fond_of(7, 10), &items, Norm::Sum).unwrap();
        assert_eq!((r.correct, r.n), (2, 3));
        assert!((r.accuracy - 2.0 / 3.0).abs() < 1e-12);
        assert!((r.stderr - (2.0f64 / 3.0 * (1.0 / 3.0) / 3.0).sqrt()).abs() < 1e-12);
        // sum favours the short answer, per-token does not penalise the long one
        let short = Choice { ctx: vec![1], cont: vec![7], chars: 1 };
        let long = Choice { ctx: vec![1], cont: vec![7, 7, 7], chars: 3 };
        let mut f = fond_of(7, 10);
        assert!(score_choice(&mut f, &short, Norm::Sum).unwrap() > score_choice(&mut f, &long, Norm::Sum).unwrap());
        let (a, b) = (score_choice(&mut f, &short, Norm::PerToken).unwrap(), score_choice(&mut f, &long, Norm::PerToken).unwrap());
        assert!((a - b).abs() < 1e-9);
        let (a, b) = (score_choice(&mut f, &short, Norm::PerChar).unwrap(), score_choice(&mut f, &long, Norm::PerChar).unwrap());
        assert!((a - b).abs() < 1e-9);
        // bad inputs
        assert!(evaluate_multiple_choice(fond_of(7, 10), &[], Norm::Sum).is_err());
        let bad = McItem { choices: vec![short.clone()], label: 3 };
        assert!(evaluate_multiple_choice(fond_of(7, 10), &[bad], Norm::Sum).is_err());
        let noctx = Choice { ctx: vec![], cont: vec![1], chars: 1 };
        assert!(score_choice(&mut f, &noctx, Norm::Sum).is_err());
        let nocont = Choice { ctx: vec![1], cont: vec![], chars: 0 };
        assert!(score_choice(&mut f, &nocont, Norm::Sum).is_err());
    }

    fn enc(s: &str) -> Vec<u32> { s.split_whitespace().map(|w| w.bytes().map(|b| b as u32).sum::<u32>() % 50 + 1).collect() }

    #[test]
    fn datasets_in_the_common_shapes_load() {
        let hs = r#"[{"ctx":"a man sits","endings":["on a chair","in the sea"],"label":"1"}]"#;
        let it = load_mc_json(hs, enc).unwrap();
        assert_eq!((it[0].label, it[0].choices.len()), (1, 2));
        let mmlu = r#"{"data":[{"question":"two plus two","choices":["three","four","five","six"],"answer":"B"},
                               {"question":"colour of grass","choices":["green","red"],"answer":0}]}"#;
        let it = load_mc_json(mmlu, enc).unwrap();
        assert_eq!((it[0].label, it[1].label, it[0].choices.len()), (1, 0, 4));
        let wg = r#"[{"sentence":"The trophy did not fit in the suitcase because the _ was too big.","option1":"trophy","option2":"suitcase","answer":"1"}]"#;
        let it = load_mc_json(wg, enc).unwrap();
        assert_eq!(it[0].label, 0);
        assert_eq!(it[0].choices[0].cont, it[0].choices[1].cont, "same suffix, different prefix");
        assert_ne!(it[0].choices[0].ctx, it[0].choices[1].ctx);
        // runs end to end
        let r = evaluate_multiple_choice(uniform(64), &it, Norm::Sum).unwrap();
        assert_eq!(r.n, 1);
    }

    #[test]
    fn hostile_datasets_are_errors() {
        for bad in ["", "{", "3", "{\"a\":1}", "[1]", "[{}]",
            r#"[{"ctx":"x","endings":["a"],"label":0}]"#,
            r#"[{"ctx":"x","endings":["a","b"],"label":5}]"#,
            r#"[{"ctx":"x","endings":["a","b"],"label":-1}]"#,
            r#"[{"ctx":"x","endings":["a",2],"label":0}]"#,
            r#"[{"ctx":"x","endings":["a","b"]}]"#,
            r#"[{"ctx":"","endings":["a","b"],"label":0}]"#,
            r#"[{"sentence":"no blank","option1":"a","option2":"b","answer":"1"}]"#,
            r#"[{"sentence":"blank at end _","option1":"a","option2":"b","answer":"1"}]"#,
            r#"[{"sentence":"x _ y","option1":"a","option2":"b","answer":"3"}]"#] {
            assert!(load_mc_json(bad, enc).is_err(), "{bad}");
        }
    }

    #[test]
    fn bench_statistics_come_from_the_injected_clock() {
        let t = Cell::new(0.0f64);
        // prefill takes 0.5 s for 100 tokens (200 t/s); decoding takes 0.01 s per token (100 t/s)
        let cfg = BenchCfg { prompt_tokens: 100, gen_tokens: 20, warmup: 2, reps: 4 };
        let resets = Cell::new(0);
        let r = bench(cfg, || resets.set(resets.get() + 1), |_| t.set(t.get() + 0.5), || t.set(t.get() + 0.01), || t.get()).unwrap();
        assert_eq!(resets.get(), 6);
        assert!((r.pp.mean - 200.0).abs() < 1e-6 && r.pp.stddev.abs() < 1e-6);
        assert!((r.tg.mean - 100.0).abs() < 1e-6);
        assert_eq!(r.pp.n, 4);
        let md = r.to_markdown();
        assert!(md.contains("pp100 | 200.00 ± 0.00") && md.contains("tg20 | 100.00 ± 0.00"), "{md}");
        assert_eq!(r.to_json()["pp_tokens_per_s"]["n"], 4);
        // a stopped clock and nonsense configs are errors
        assert!(bench(cfg, || {}, |_| {}, || {}, || 1.0).is_err());
        for c in [BenchCfg { reps: 0, ..cfg }, BenchCfg { prompt_tokens: 0, ..cfg }, BenchCfg { gen_tokens: 0, ..cfg }] {
            assert!(bench(c, || {}, |_| {}, || {}, || 0.0).is_err());
        }
    }

    #[test]
    fn stats_use_the_sample_standard_deviation() {
        let s = Stats::from(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
        assert!((s.mean - 5.0).abs() < 1e-12);
        assert!((s.median - 4.5).abs() < 1e-12);
        assert!((s.stddev - (32.0f64 / 7.0).sqrt()).abs() < 1e-12);
        assert_eq!((s.min, s.max), (2.0, 9.0));
        assert_eq!(Stats::from(&[3.0]).unwrap().stddev, 0.0);
        assert!(Stats::from(&[]).is_none());
        assert!(Stats::from(&[1.0, f64::NAN]).is_none());
    }

    #[test]
    fn reports_render_as_json_and_markdown() {
        let ppl = perplexity(uniform(4), &text(30, 4), 16, 16).unwrap();
        let mut rep = Report::new("demo model");
        rep.add("perplexity", ppl.to_json()).add("kl", json!({"kl_mean": 0.0123456, "nucleus_jaccard": null}));
        let j = rep.to_json();
        assert_eq!(j["title"], "demo model");
        assert!((j["results"]["perplexity"]["perplexity"].as_f64().unwrap() - 4.0).abs() < 1e-9);
        let md = rep.to_markdown();
        assert!(md.starts_with("# demo model") && md.contains("## perplexity") && md.contains("| kl_mean | 0.0123 |") && md.contains("| nucleus_jaccard | – |"), "{md}");
    }
}
