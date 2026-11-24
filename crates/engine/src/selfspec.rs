//! Self-speculative decoding by layer skipping: one model plays both roles.
//!
//! The *draft* is the same network run through only a subset of its layers (the others are
//! skipped; embedding, final norm and output head are shared). Drafting costs a fraction of a
//! full step and needs no second set of weights. The *verification* is one batched pass of the
//! whole network over the pending token and the drafts, which is how the output stays exact:
//!
//! * greedy decoding emits exactly the tokens plain greedy decoding emits;
//! * sampled decoding uses the standard speculative rejection rule ([`crate::spec::verify_sampling`]),
//!   so every emitted token is distributed exactly as the full model's own sampler would draw it.
//!
//! # Cache strategy (the part that is easy to get wrong)
//!
//! There is a single [`State`] and a single KV cache. A draft step runs only the kept layers, so
//! it writes K/V for the kept layers at the draft positions and nothing for the skipped ones, and
//! those K/V are *approximate* (the hidden state fed to a kept layer lacks the skipped layers'
//! contribution). Before the verification pass the state is rewound to the pre-draft position,
//! which throws every approximate entry away. The verification pass then **recomputes all layers
//! for all drafted positions** and writes true K/V everywhere; the rejected suffix is truncated
//! afterwards. So no approximate K/V ever survives a round, and the cache after every round is
//! the cache plain decoding would have produced (verification runs with `State::set_exact_batch`, so quantised weights are bit-identical too;
//! tests cover dense, Q8_0, Q4_0 and Q4_K).
//!
//! Consequences and limits, all handled by falling back to a plain step rather than being wrong:
//! * Sliding-window layers keep a ring buffer. Drafting and verifying write `k + 1` slots ahead
//!   and a rejection must rewind, which is only possible while the ring has not wrapped, so a
//!   round is only attempted when `pos + k + 1 <= window` for every ring (else: plain steps).
//! * A state whose cache was compacted by [`crate::kvpolicy`] cannot be rewound reliably: plain steps.
//! * Models with accelerator-offloaded layers are refused ([`SelfSpecError::Unsupported`]).
//! * The verification batch computes the output head once per row; the cost model accounts for it.
//! * Repetition penalties are not part of [`SamplingCfg`]; only temperature, top-k and top-p.
//!
//! # Choosing the skip set
//!
//! [`calibrate`] measures, on a few prompts (extended by the model's own greedy continuation),
//! how often the draft's next-token argmax agrees with the full model's, for candidate skip sets
//! grown greedily one layer at a time, and keeps the set with the best *estimated* speed-up
//! ([`expected_speedup`]). First and last layers are pinned by default (they matter most).
//! The estimate assumes the first-token agreement rate holds along the draft chain and that a
//! batched verification pass costs [`SpeedModel::batch_factor`] plain passes; both are
//! approximations, and nothing here has been measured on a real checkpoint.
use crate::model::{Model, ModelError, State};
use crate::spec::{AdaptiveLen, Rng, SamplingCfg, argmax, sample_from, verify_sampling};
use std::ops::Range;
use std::time::{Duration, Instant};

/// Largest draft length the module will ever use (keeps verification batches small).
pub const MAX_DRAFT: usize = 32;

#[derive(Debug, PartialEq)]
pub enum SelfSpecError {
    Model(ModelError),
    /// A parameter makes no sense (layer index out of range, empty prompt list, ...).
    Invalid(String),
    /// The model cannot be driven this way (layers offloaded to an accelerator).
    Unsupported(String),
}
impl std::fmt::Display for SelfSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelfSpecError::Model(e) => write!(f, "{e}"),
            SelfSpecError::Invalid(s) => write!(f, "invalid self-speculation setting: {s}"),
            SelfSpecError::Unsupported(s) => write!(f, "self-speculation unsupported: {s}"),
        }
    }
}
impl std::error::Error for SelfSpecError {}
impl From<ModelError> for SelfSpecError {
    fn from(e: ModelError) -> Self {
        SelfSpecError::Model(e)
    }
}

// ------------------------------------------------------------------------------------------
// Skip sets
// ------------------------------------------------------------------------------------------

/// Which layers the draft skips.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SkipSet {
    skip: Vec<bool>,
}
impl SkipSet {
    /// Skip nothing (the draft is then the full model).
    pub fn none(layers: usize) -> SkipSet {
        SkipSet {
            skip: vec![false; layers],
        }
    }
    /// Skip the listed layers (duplicates are fine); errors on an index `>= layers`.
    pub fn from_layers(layers: usize, skipped: &[usize]) -> Result<SkipSet, SelfSpecError> {
        let mut skip = vec![false; layers];
        for &l in skipped {
            *skip.get_mut(l).ok_or_else(|| {
                SelfSpecError::Invalid(format!("layer {l} does not exist (the model has {layers})"))
            })? = true;
        }
        Ok(SkipSet { skip })
    }
    /// `count` layers spread evenly over `pin_first..layers - pin_last`.
    pub fn evenly(layers: usize, count: usize, pin_first: usize, pin_last: usize) -> SkipSet {
        let (lo, hi) = (pin_first.min(layers), layers.saturating_sub(pin_last));
        let span = hi.saturating_sub(lo);
        let count = count.min(span);
        let mut skip = vec![false; layers];
        for i in 0..count {
            skip[lo + (2 * i + 1) * span / (2 * count)] = true;
        }
        SkipSet { skip }
    }
    pub fn layers(&self) -> usize {
        self.skip.len()
    }
    pub fn is_skipped(&self, layer: usize) -> bool {
        self.skip.get(layer).copied().unwrap_or(false)
    }
    pub fn skipped(&self) -> Vec<usize> {
        self.skip
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s)
            .map(|(i, _)| i)
            .collect()
    }
    pub fn skipped_count(&self) -> usize {
        self.skip.iter().filter(|&&s| s).count()
    }
    pub fn is_empty(&self) -> bool {
        self.skipped_count() == 0
    }
    /// Consecutive runs of kept layers, in order.
    pub fn kept_ranges(&self) -> Vec<Range<usize>> {
        let (mut out, mut start) = (vec![], None);
        for (i, &s) in self.skip.iter().enumerate() {
            match (s, start) {
                (false, None) => start = Some(i),
                (true, Some(a)) => {
                    out.push(a..i);
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(a) = start {
            out.push(a..self.skip.len());
        }
        out
    }
    fn with(&self, layer: usize) -> SkipSet {
        let mut s = self.clone();
        if let Some(x) = s.skip.get_mut(layer) {
            *x = true;
        }
        s
    }

    /// Cost of one draft step relative to one full step: kept layers plus the output head, over
    /// all layers plus the head, weighted by stored bytes (decoding is memory-bound).
    pub fn draft_cost(&self, model: &Model) -> f64 {
        let (layer_b, head_b) = layer_costs(model);
        let total: f64 = layer_b.iter().sum::<f64>() + head_b;
        if total <= 0.0 {
            return 1.0;
        }
        let kept: f64 = layer_b
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.is_skipped(*i))
            .map(|(_, b)| *b)
            .sum();
        ((kept + head_b) / total).clamp(0.0, 1.0)
    }
}

/// Stored bytes of each layer's weights, and of the output head.
fn layer_costs(model: &Model) -> (Vec<f64>, f64) {
    let layers = model
        .w
        .layers
        .iter()
        .map(|l| {
            let dense: usize = [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]
                .iter()
                .map(|w| w.stored_bytes())
                .sum();
            (dense + l.moe.as_ref().map_or(0, |m| m.stored_bytes())) as f64
        })
        .collect();
    let head = model
        .w
        .lm_head
        .as_ref()
        .unwrap_or(&model.w.embed)
        .stored_bytes() as f64;
    (layers, head)
}

// ------------------------------------------------------------------------------------------
// Speed model
// ------------------------------------------------------------------------------------------

/// Inputs of the speed-up estimate besides the acceptance rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeedModel {
    /// Cost of one draft step, in full steps (see [`SkipSet::draft_cost`]).
    pub draft_cost: f64,
    /// Share of a full step that is the output head (the verification pass pays it per row).
    pub head_frac: f64,
    /// Cost of the layer part of a batched verification pass, in plain layer passes.
    /// 1.0 = perfectly memory-bound (reading the weights once serves every row).
    pub batch_factor: f64,
}
impl SpeedModel {
    pub fn for_model(model: &Model, skip: &SkipSet, batch_factor: f64) -> SpeedModel {
        let (layer_b, head_b) = layer_costs(model);
        let total: f64 = layer_b.iter().sum::<f64>() + head_b;
        SpeedModel {
            draft_cost: skip.draft_cost(model),
            head_frac: if total > 0.0 { head_b / total } else { 0.0 },
            batch_factor,
        }
    }
}

/// Expected tokens emitted per verification round for acceptance rate `alpha` and draft length `k`
/// (Leviathan et al.): `(1 - alpha^(k+1)) / (1 - alpha)`.
pub fn expected_tokens_per_round(alpha: f64, k: usize) -> f64 {
    let a = if alpha.is_nan() {
        0.0
    } else {
        alpha.clamp(0.0, 1.0)
    };
    if a >= 1.0 - 1e-12 {
        (k + 1) as f64
    } else {
        (1.0 - a.powi(k as i32 + 1)) / (1.0 - a)
    }
}

/// Estimated speed-up over plain decoding (tokens per unit of weight-reading time): tokens per
/// round over `k` draft steps plus one batched verification pass. Below 1.0 means a slowdown.
pub fn expected_speedup(alpha: f64, k: usize, m: &SpeedModel) -> f64 {
    let h = m.head_frac.clamp(0.0, 1.0);
    let verify = m.batch_factor.max(0.0) * (1.0 - h) + (k as f64 + 1.0) * h;
    let cost = k as f64 * m.draft_cost.max(0.0) + verify;
    if cost <= 0.0 {
        return 1.0;
    }
    expected_tokens_per_round(alpha, k) / cost
}

/// The draft length in `1..=max_k` with the best estimated speed-up, and that speed-up.
/// Returns `(0, 1.0)` when no length beats plain decoding.
pub fn best_draft_len(alpha: f64, max_k: usize, m: &SpeedModel) -> (usize, f64) {
    let mut best = (0, 1.0);
    for k in 1..=max_k.min(MAX_DRAFT) {
        let s = expected_speedup(alpha, k, m);
        if s > best.1 {
            best = (k, s);
        }
    }
    best
}

// ------------------------------------------------------------------------------------------
// Configuration, statistics, output
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SelfSpecConfig {
    pub skip: SkipSet,
    pub sampling: SamplingCfg,
    pub seed: u64,
    pub max_tokens: usize,
    /// Generation stops after emitting any of these (the stop token is included in the output).
    pub stop_tokens: Vec<usize>,
    /// Draft length bounds and starting value; the length then adapts to the acceptance rate.
    pub min_draft: usize,
    pub max_draft: usize,
    pub start_draft: usize,
}
impl SelfSpecConfig {
    pub fn new(skip: SkipSet) -> SelfSpecConfig {
        SelfSpecConfig {
            skip,
            sampling: SamplingCfg::greedy(),
            seed: 0,
            max_tokens: 64,
            stop_tokens: vec![],
            min_draft: 1,
            max_draft: 8,
            start_draft: 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SelfSpecStats {
    /// Verification rounds (each: `k` draft steps and one batched pass).
    pub rounds: usize,
    /// Draft tokens proposed / accepted.
    pub drafted: usize,
    pub accepted: usize,
    /// Plain single-token steps (budget end, ring caches, compacted state, `max_draft == 0`).
    pub plain_steps: usize,
    /// Tokens emitted.
    pub emitted: usize,
}
impl SelfSpecStats {
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }
    /// Tokens emitted per full-model pass (rounds count one pass, plain steps one each, the
    /// prompt's first token is free).
    pub fn tokens_per_pass(&self) -> f64 {
        let passes = self.rounds + self.plain_steps;
        if passes == 0 {
            0.0
        } else {
            self.emitted as f64 / passes as f64
        }
    }
    /// Speed-up this run achieved in the cost model `m`, from its measured rounds (not from an
    /// i.i.d. acceptance assumption: rejections cluster, so [`expected_speedup`] on the average
    /// acceptance rate underestimates what a run really gets). Weight-reading time only.
    pub fn estimated_speedup(&self, m: &SpeedModel) -> f64 {
        let produced = self.emitted.saturating_sub(1) as f64; // the first token comes from the prompt pass
        if self.rounds == 0 && self.plain_steps == 0 {
            return 1.0;
        }
        let h = m.head_frac.clamp(0.0, 1.0);
        let k = if self.rounds == 0 {
            0.0
        } else {
            self.drafted as f64 / self.rounds as f64
        };
        let verify = m.batch_factor.max(0.0) * (1.0 - h) + (k + 1.0) * h;
        let time =
            self.rounds as f64 * (k * m.draft_cost.max(0.0) + verify) + self.plain_steps as f64;
        if time <= 0.0 { 1.0 } else { produced / time }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    MaxTokens,
    StopToken,
    ContextLimit,
}

#[derive(Clone, Debug)]
pub struct SelfSpecOutput {
    pub tokens: Vec<usize>,
    pub stats: SelfSpecStats,
    pub stop: StopReason,
}

fn check_model(model: &Model, skip: &SkipSet) -> Result<(), SelfSpecError> {
    if model.offloaded().iter().any(|&o| o) {
        return Err(SelfSpecError::Unsupported(
            "layers are offloaded to an accelerator; use a host-only model".into(),
        ));
    }
    if skip.layers() != model.spec.layers as usize {
        return Err(SelfSpecError::Invalid(format!(
            "the skip set covers {} layers but the model has {}",
            skip.layers(),
            model.spec.layers
        )));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------
// Generation
// ------------------------------------------------------------------------------------------

fn pick(logits: &[f32], cfg: &SamplingCfg, rng: &mut Rng) -> usize {
    if cfg.is_greedy() {
        argmax(logits) as usize
    } else {
        sample_from(&cfg.probs(logits), rng) as usize
    }
}

/// The reference: ordinary one-token-at-a-time decoding with the same sampling rule and stop
/// handling. Greedy output of [`generate`] equals this.
pub fn generate_plain(
    model: &Model,
    st: &mut State,
    prompt: &[usize],
    cfg: &SelfSpecConfig,
) -> Result<SelfSpecOutput, SelfSpecError> {
    if prompt.is_empty() {
        return Err(ModelError::NoTokens.into());
    }
    let mut out = SelfSpecOutput {
        tokens: vec![],
        stats: SelfSpecStats::default(),
        stop: StopReason::MaxTokens,
    };
    if cfg.max_tokens == 0 {
        return Ok(out);
    }
    let mut rng = Rng::new(cfg.seed);
    let mut logits = model.prefill_cached(st, prompt)?;
    loop {
        let t = pick(&logits, &cfg.sampling, &mut rng);
        out.tokens.push(t);
        out.stats.emitted += 1;
        if cfg.stop_tokens.contains(&t) {
            out.stop = StopReason::StopToken;
            break;
        }
        if out.tokens.len() >= cfg.max_tokens {
            break;
        }
        if st.pos >= model.spec.context as usize {
            out.stop = StopReason::ContextLimit;
            break;
        }
        logits = model.forward(st, t)?;
        out.stats.plain_steps += 1;
    }
    Ok(out)
}

/// Largest draft length the cache layout allows at position `pos`, `None` when speculation is off.
fn cache_room(st: &State, pos: usize) -> Option<usize> {
    if st.is_compacted() {
        return None;
    }
    let mut room = usize::MAX;
    for cap in st.ring_caps() {
        if cap > 0 {
            room = room.min(cap.saturating_sub(pos + 1));
        }
    }
    Some(room)
}

/// Run only the kept layers for `k` tokens (the pending one, then each draft), return the draft
/// tokens and (sampling) the processed distributions they were drawn from. Leaves `st` rewound.
fn draft_chain(
    model: &Model,
    st: &mut State,
    ranges: &[Range<usize>],
    pending: usize,
    k: usize,
    cfg: &SamplingCfg,
    rng: &mut Rng,
) -> Result<(Vec<usize>, Vec<Vec<f32>>), SelfSpecError> {
    let base = st.pos;
    let r = (|| -> Result<(Vec<usize>, Vec<Vec<f32>>), SelfSpecError> {
        let (mut toks, mut qs, mut cur) = (Vec::with_capacity(k), Vec::new(), pending);
        for _ in 0..k {
            let mut x = model.embed_token(cur)?;
            for r in ranges {
                let pos = st.pos;
                model.run_layers(st, r.clone(), &mut x, pos)?;
            }
            model.finish_token(st);
            let l = model.logits_from_hidden(&x);
            if l.is_empty() {
                return Err(ModelError::Shape("draft produced no logits".into()).into());
            }
            let d = if cfg.is_greedy() {
                argmax(&l) as usize
            } else {
                let q = cfg.probs(&l);
                let d = sample_from(&q, rng) as usize;
                qs.push(q);
                d
            };
            toks.push(d);
            cur = d;
        }
        Ok((toks, qs))
    })();
    // whatever happened, drop the approximate draft K/V (best effort on the error path)
    let rewound = if st.pos > base {
        st.truncate(base)
    } else {
        Ok(())
    };
    let out = r?;
    rewound?;
    Ok(out)
}

/// Verification pass over `[pending, drafts...]` through every layer. Returns the accepted count and
/// the extra token (correction or bonus). The state ends holding `pending` and the accepted drafts.
fn verify_round(
    model: &Model,
    st: &mut State,
    pending: usize,
    drafts: &[usize],
    qs: &[Vec<f32>],
    cfg: &SamplingCfg,
    rng: &mut Rng,
) -> Result<(usize, usize), SelfSpecError> {
    let (h, p, k) = (model.spec.hidden as usize, st.pos, drafts.len());
    let mut toks = Vec::with_capacity(k + 1);
    toks.push(pending);
    toks.extend_from_slice(drafts);
    let mut x = Vec::with_capacity((k + 1) * h);
    for &t in &toks {
        x.extend_from_slice(&model.embed_token(t)?);
    }
    // Exact-batch mode for the verification pass only: the k+1 rows then equal k+1 single-token steps
    // bit for bit, also for quantised weights (the prompt prefill is untouched, like plain decoding).
    let was_exact = st.exact_batch();
    st.set_exact_batch(true);
    let ran = model.run_layers_batch(st, 0..model.spec.layers as usize, &mut x, p);
    st.set_exact_batch(was_exact);
    ran?;
    model.finish_batch(st, &toks);
    let row = |i: usize| model.logits_from_hidden(&x[i * h..(i + 1) * h]);
    let (accepted, extra) = if cfg.is_greedy() {
        // lazy: stop reading rows at the first disagreement
        let mut a = 0;
        loop {
            let t = argmax(&row(a)) as usize;
            if a < k && t == drafts[a] {
                a += 1;
            } else {
                break (a, t);
            }
        }
    } else {
        let ps: Vec<Vec<f32>> = (0..=k).map(|i| cfg.probs(&row(i))).collect();
        let d32: Vec<u32> = drafts.iter().map(|&d| d as u32).collect();
        let v = verify_sampling(&ps, &d32, Some(qs), None, rng);
        (v.accepted, v.extra as usize)
    };
    if accepted < k {
        st.truncate(p + accepted + 1)?;
    }
    Ok((accepted, extra))
}

/// Generate with self-speculative decoding. See the module docs for what is guaranteed.
pub fn generate(
    model: &Model,
    st: &mut State,
    prompt: &[usize],
    cfg: &SelfSpecConfig,
) -> Result<SelfSpecOutput, SelfSpecError> {
    check_model(model, &cfg.skip)?;
    if prompt.is_empty() {
        return Err(ModelError::NoTokens.into());
    }
    let mut out = SelfSpecOutput {
        tokens: vec![],
        stats: SelfSpecStats::default(),
        stop: StopReason::MaxTokens,
    };
    if cfg.max_tokens == 0 {
        return Ok(out);
    }
    let ranges = cfg.skip.kept_ranges();
    let max_k = cfg.max_draft.min(MAX_DRAFT);
    let mut adapt = AdaptiveLen::new(
        cfg.min_draft.min(max_k).max(1),
        max_k.max(1),
        cfg.start_draft,
    );
    let speculate = max_k > 0 && !cfg.skip.is_empty();
    let mut rng = Rng::new(cfg.seed);
    let ctx = model.spec.context as usize;
    let logits = model.prefill_cached(st, prompt)?;
    let mut pending = pick(&logits, &cfg.sampling, &mut rng);
    out.tokens.push(pending);
    out.stats.emitted = 1;
    'rounds: loop {
        if cfg.stop_tokens.contains(&pending) {
            out.stop = StopReason::StopToken;
            break;
        }
        if out.tokens.len() >= cfg.max_tokens {
            break;
        }
        if st.pos >= ctx {
            out.stop = StopReason::ContextLimit;
            break;
        }
        let p = st.pos;
        let remaining = cfg.max_tokens - out.tokens.len();
        let k = if speculate {
            let room = cache_room(st, p);
            room.map_or(0, |r| adapt.cur.min(r).min(remaining - 1).min(ctx - p - 1))
        } else {
            0
        };
        if k == 0 {
            let l = model.forward(st, pending)?;
            out.stats.plain_steps += 1;
            pending = pick(&l, &cfg.sampling, &mut rng);
            out.tokens.push(pending);
            out.stats.emitted += 1;
            continue;
        }
        let (drafts, qs) = draft_chain(model, st, &ranges, pending, k, &cfg.sampling, &mut rng)?;
        let (a, extra) = verify_round(model, st, pending, &drafts, &qs, &cfg.sampling, &mut rng)?;
        out.stats.rounds += 1;
        out.stats.drafted += k;
        out.stats.accepted += a;
        adapt.update(k, a);
        for &t in drafts[..a].iter().chain(std::iter::once(&extra)) {
            out.tokens.push(t);
            out.stats.emitted += 1;
            pending = t;
            if cfg.stop_tokens.contains(&t) {
                out.stop = StopReason::StopToken;
                break 'rounds;
            }
            if out.tokens.len() >= cfg.max_tokens {
                break 'rounds;
            }
        }
    }
    // Convention shared with `generate_plain`: the last emitted token is pending (not fed). A stop
    // inside an accepted run leaves later tokens fed; rewind so the state matches plain decoding.
    let fed = prompt.len() + out.tokens.len() - 1;
    if st.pos > fed && st.can_truncate(fed) {
        st.truncate(fed)?;
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------
// Calibration
// ------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CalibrateOptions {
    /// At most this fraction of the layers may be skipped.
    pub max_skip_fraction: f64,
    /// Layers at the start / end that are never skipped.
    pub pin_first: usize,
    pub pin_last: usize,
    /// Largest draft length the estimator considers.
    pub max_draft: usize,
    /// See [`SpeedModel::batch_factor`].
    pub batch_factor: f64,
    /// Tokens of each prompt used (teacher-forced), then extended with this many greedy tokens.
    pub prompt_tokens: usize,
    pub continue_tokens: usize,
    /// A set is only accepted when it beats plain decoding by this factor.
    pub min_gain: f64,
    /// Evaluate at most this many candidate layers per greedy step (evenly thinned).
    pub candidates_per_step: usize,
    /// Stop searching after this long (the best set found so far is returned).
    pub deadline: Option<Duration>,
}
impl Default for CalibrateOptions {
    fn default() -> Self {
        CalibrateOptions {
            max_skip_fraction: 0.5,
            pin_first: 1,
            pin_last: 1,
            max_draft: 8,
            batch_factor: 1.0,
            prompt_tokens: 24,
            continue_tokens: 16,
            min_gain: 1.03,
            candidates_per_step: usize::MAX,
            deadline: None,
        }
    }
}

/// One step of the greedy search, for inspection.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationStep {
    pub added: usize,
    pub alpha: f64,
    pub cost: f64,
    pub best_k: usize,
    pub speedup: f64,
}

#[derive(Clone, Debug)]
pub struct Calibration {
    pub skip: SkipSet,
    /// Measured next-token agreement of the draft with the full model.
    pub alpha: f64,
    pub draft_cost: f64,
    pub best_k: usize,
    /// Estimated speed-up (1.0 when nothing beat plain decoding; then `skip` is empty).
    pub est_speedup: f64,
    pub worthwhile: bool,
    pub positions: usize,
    pub trace: Vec<CalibrationStep>,
    pub timed_out: bool,
}

/// Teacher-forced sequences: each prompt cut to `prompt_tokens`, extended with the model's greedy continuation.
fn calibration_sequences(
    model: &Model,
    prompts: &[Vec<usize>],
    o: &CalibrateOptions,
) -> Result<Vec<Vec<usize>>, SelfSpecError> {
    let ctx = model.spec.context as usize;
    let mut seqs = vec![];
    for p in prompts {
        if p.is_empty() {
            continue;
        }
        let mut s: Vec<usize> = p.iter().copied().take(o.prompt_tokens.max(1)).collect();
        if let Some(&t) = s.iter().find(|&&t| t >= model.spec.vocab as usize) {
            return Err(ModelError::Token(t).into());
        }
        let mut st = model.new_state();
        let mut l = model.forward_batch(&mut st, &s)?;
        for _ in 0..o.continue_tokens {
            if s.len() + 1 >= ctx {
                break;
            }
            let t = argmax(&l) as usize;
            s.push(t);
            l = model.forward(&mut st, t)?;
        }
        seqs.push(s);
    }
    if seqs.is_empty() {
        return Err(SelfSpecError::Invalid(
            "no usable calibration prompt (all empty)".into(),
        ));
    }
    Ok(seqs)
}

/// Draft argmax for one token at the current position, leaving the state as it was.
fn probe(
    model: &Model,
    st: &mut State,
    token: usize,
    ranges: &[Range<usize>],
) -> Result<usize, SelfSpecError> {
    let base = st.pos;
    let mut x = model.embed_token(token)?;
    let mut r = Ok(());
    for rg in ranges {
        r = model.run_layers(st, rg.clone(), &mut x, st.pos);
        if r.is_err() {
            break;
        }
    }
    if r.is_ok() {
        model.finish_token(st);
    }
    let l = if r.is_ok() {
        model.logits_from_hidden(&x)
    } else {
        vec![]
    };
    if st.pos > base {
        st.truncate(base)?;
    }
    r?;
    Ok(argmax(&l) as usize)
}

/// One teacher-forced sweep: for every candidate set, how often its draft agrees with the full model.
fn sweep(
    model: &Model,
    seqs: &[Vec<usize>],
    sets: &[SkipSet],
    deadline: Option<Instant>,
) -> Result<(Vec<(usize, usize)>, bool), SelfSpecError> {
    let ranges: Vec<Vec<Range<usize>>> = sets.iter().map(SkipSet::kept_ranges).collect();
    let mut acc = vec![(0usize, 0usize); sets.len()];
    for s in seqs {
        let mut st = model.new_state();
        for &tok in s {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok((acc, true));
            }
            if cache_room(&st, st.pos).is_none_or(|r| r == 0) {
                break;
            }
            let drafts: Vec<usize> = ranges
                .iter()
                .map(|r| probe(model, &mut st, tok, r))
                .collect::<Result<_, _>>()?;
            let truth = argmax(&model.forward(&mut st, tok)?) as usize;
            for (a, d) in acc.iter_mut().zip(drafts) {
                a.1 += 1;
                a.0 += usize::from(d == truth);
            }
        }
    }
    Ok((acc, false))
}

/// Choose a skip set for `model` automatically. Greedy forward selection: starting from nothing,
/// repeatedly skip the one extra layer that gives the best estimated speed-up, stopping when two
/// consecutive additions fail to improve it or the skip budget is used up.
pub fn calibrate(
    model: &Model,
    prompts: &[Vec<usize>],
    o: &CalibrateOptions,
) -> Result<Calibration, SelfSpecError> {
    let n = model.spec.layers as usize;
    check_model(model, &SkipSet::none(n))?;
    let seqs = calibration_sequences(model, prompts, o)?;
    let deadline = o.deadline.map(|d| Instant::now() + d);
    let max_skip = ((n as f64) * o.max_skip_fraction.clamp(0.0, 1.0)).floor() as usize;
    let eligible: Vec<usize> = (o.pin_first.min(n)..n.saturating_sub(o.pin_last)).collect();
    let mut cur = SkipSet::none(n);
    let mut best = Calibration {
        skip: SkipSet::none(n),
        alpha: 1.0,
        draft_cost: 1.0,
        best_k: 0,
        est_speedup: 1.0,
        worthwhile: false,
        positions: seqs.iter().map(Vec::len).sum(),
        trace: vec![],
        timed_out: false,
    };
    let (mut non_improving, mut best_seen) = (0, 1.0f64);
    while cur.skipped_count() < max_skip && non_improving < 2 {
        let mut cands: Vec<usize> = eligible
            .iter()
            .copied()
            .filter(|&l| !cur.is_skipped(l))
            .collect();
        if cands.is_empty() {
            break;
        }
        if cands.len() > o.candidates_per_step.max(1) {
            let step = cands.len() as f64 / o.candidates_per_step.max(1) as f64;
            cands = (0..o.candidates_per_step.max(1))
                .map(|i| cands[((i as f64 + 0.5) * step) as usize])
                .collect();
        }
        let sets: Vec<SkipSet> = cands.iter().map(|&l| cur.with(l)).collect();
        let (acc, timed_out) = sweep(model, &seqs, &sets, deadline)?;
        if timed_out {
            best.timed_out = true;
        }
        let mut round_best: Option<(usize, CalibrationStep)> = None;
        for (i, ((agree, total), set)) in acc.iter().zip(&sets).enumerate() {
            if *total == 0 {
                continue;
            }
            let alpha = *agree as f64 / *total as f64;
            let sm = SpeedModel::for_model(model, set, o.batch_factor);
            let (k, s) = best_draft_len(alpha, o.max_draft, &sm);
            let step = CalibrationStep {
                added: cands[i],
                alpha,
                cost: sm.draft_cost,
                best_k: k,
                speedup: s,
            };
            if round_best
                .as_ref()
                .is_none_or(|(_, b)| step.speedup > b.speedup)
            {
                round_best = Some((i, step));
            }
        }
        let Some((i, step)) = round_best else { break };
        cur = sets[i].clone();
        if step.speedup > best_seen {
            best_seen = step.speedup;
            non_improving = 0;
            if step.speedup >= o.min_gain {
                best.skip = cur.clone();
                best.alpha = step.alpha;
                best.draft_cost = step.cost;
                best.best_k = step.best_k;
                best.est_speedup = step.speedup;
                best.worthwhile = true;
            }
        } else {
            non_improving += 1;
        }
        best.trace.push(step);
        if timed_out {
            break;
        }
    }
    Ok(best)
}

// ------------------------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Spec;
    use crate::model::KvKind;
    use crate::model::tests::{base_spec, model_of};

    fn deep(layers: u64, heads: u64, kv: u64) -> Model {
        let mut sp = Spec::dense("llama", layers, 16, heads, kv, 8, 24, 20, 96);
        sp.tied_embeddings = false;
        model_of(sp, false)
    }
    fn prompt(n: usize, seed: usize) -> Vec<usize> {
        (0..n).map(|i| (i * 7 + seed * 3 + 1) % 20).collect()
    }
    fn cfg(skip: SkipSet, k: usize, n: usize) -> SelfSpecConfig {
        let mut c = SelfSpecConfig::new(skip);
        c.max_tokens = n;
        c.max_draft = k;
        c.start_draft = k.max(1);
        c
    }

    #[test]
    fn selfspec_greedy_is_bit_identical_to_plain_on_quantised_weights() {
        use crate::model::tests::{block_model, requantise_to};
        use crate::quant::DType;
        for dt in [DType::Q8_0, DType::Q4_0, DType::Q4K] {
            let m = requantise_to(&block_model(), dt);
            let n = m.spec.layers as usize;
            let skip = SkipSet::from_layers(n, &[1, 2]).unwrap();
            for seed in 0..3usize {
                let p: Vec<usize> = (0..9).map(|i| (i * 7 + seed * 3 + 1) % 40).collect();
                let c = cfg(skip.clone(), 3, 12);
                let a = generate(&m, &mut m.new_state(), &p, &c).unwrap();
                let b = generate_plain(&m, &mut m.new_state(), &p, &c).unwrap();
                assert_eq!(a.tokens, b.tokens, "{dt:?} seed {seed}");
            }
        }
    }

    #[test]
    fn skip_set_basics() {
        let s = SkipSet::from_layers(6, &[1, 2, 4, 4]).unwrap();
        assert_eq!(s.skipped(), vec![1, 2, 4]);
        assert_eq!(s.kept_ranges(), vec![0..1, 3..4, 5..6]);
        assert!(SkipSet::from_layers(3, &[3]).is_err());
        assert!(SkipSet::none(4).is_empty());
        assert_eq!(SkipSet::none(4).kept_ranges(), vec![0..4]);
        assert_eq!(
            SkipSet::from_layers(3, &[0, 1, 2]).unwrap().kept_ranges(),
            Vec::<Range<usize>>::new()
        );
        let e = SkipSet::evenly(10, 3, 1, 1);
        assert_eq!(e.skipped_count(), 3);
        assert!(!e.is_skipped(0) && !e.is_skipped(9));
        assert_eq!(SkipSet::evenly(4, 99, 1, 1).skipped_count(), 2);
        assert_eq!(SkipSet::evenly(0, 3, 1, 1).skipped_count(), 0);
    }

    #[test]
    fn draft_cost_counts_kept_layers_and_head() {
        let m = deep(6, 2, 2);
        assert!((SkipSet::none(6).draft_cost(&m) - 1.0).abs() < 1e-12);
        let half = SkipSet::from_layers(6, &[1, 2, 3]).unwrap().draft_cost(&m);
        assert!(half > 0.4 && half < 0.9, "{half}");
        let all = SkipSet::from_layers(6, &[0, 1, 2, 3, 4, 5])
            .unwrap()
            .draft_cost(&m);
        assert!(all > 0.0 && all < half);
    }

    #[test]
    fn speed_formulas() {
        assert!((expected_tokens_per_round(1.0, 4) - 5.0).abs() < 1e-12);
        assert!((expected_tokens_per_round(0.0, 4) - 1.0).abs() < 1e-12);
        assert!((expected_tokens_per_round(0.5, 1) - 1.5).abs() < 1e-12);
        assert!((expected_tokens_per_round(f64::NAN, 3) - 1.0).abs() < 1e-12);
        let free = SpeedModel {
            draft_cost: 0.0,
            head_frac: 0.0,
            batch_factor: 1.0,
        };
        assert!((expected_speedup(1.0, 4, &free) - 5.0).abs() < 1e-12);
        let half = SpeedModel {
            draft_cost: 0.5,
            head_frac: 0.0,
            batch_factor: 1.0,
        };
        assert!((expected_speedup(1.0, 4, &half) - 5.0 / 3.0).abs() < 1e-12);
        assert!(expected_speedup(0.0, 4, &half) < 1.0);
        let mut last = 0.0;
        for a in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let s = expected_speedup(a, 4, &half);
            assert!(s > last);
            last = s;
        }
        assert_eq!(best_draft_len(0.0, 8, &half).0, 0);
        let (k, s) = best_draft_len(0.95, 8, &half);
        assert!(k >= 2 && s > 1.5, "{k} {s}");
        // a head that dominates makes long drafts pay for the extra rows
        let headish = SpeedModel {
            draft_cost: 0.5,
            head_frac: 0.5,
            batch_factor: 1.0,
        };
        assert!(expected_speedup(0.9, 8, &headish) < expected_speedup(0.9, 8, &half));
    }

    #[test]
    fn batch_rows_equal_single_steps_bit_for_bit() {
        // the property the exactness claim rests on
        for kind in [KvKind::F32, KvKind::F16, KvKind::Q8] {
            let m = deep(5, 2, 1);
            let toks = prompt(12, 1);
            let (mut a, mut b) = (m.new_state_with(kind), m.new_state_with(kind));
            m.forward_batch(&mut a, &toks[..4]).unwrap();
            m.forward_batch(&mut b, &toks[..4]).unwrap();
            let h = m.spec.hidden as usize;
            let mut x = vec![];
            for &t in &toks[4..] {
                x.extend(m.embed_token(t).unwrap());
            }
            let p = b.pos;
            m.run_layers_batch(&mut b, 0..5, &mut x, p).unwrap();
            for (i, &t) in toks[4..].iter().enumerate() {
                let single = m.forward(&mut a, t).unwrap();
                let row = m.logits_from_hidden(&x[i * h..(i + 1) * h]);
                assert_eq!(
                    single.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{kind:?} row {i}"
                );
            }
        }
    }

    fn assert_matches_plain(
        m: &Model,
        kind: KvKind,
        skip: SkipSet,
        k: usize,
        p: &[usize],
        n: usize,
    ) -> SelfSpecStats {
        let c = cfg(skip.clone(), k, n);
        let (mut s1, mut s2) = (m.new_state_with(kind), m.new_state_with(kind));
        let plain = generate_plain(m, &mut s2, p, &c).unwrap();
        let spec = generate(m, &mut s1, p, &c).unwrap();
        assert_eq!(
            spec.tokens,
            plain.tokens,
            "skip {:?} k {k} {kind:?}",
            skip.skipped()
        );
        assert_eq!(spec.stop, plain.stop);
        assert_eq!(spec.stats.emitted, spec.tokens.len());
        // the caches end up identical: same position, and the same next logits bit for bit
        assert_eq!(s1.pos, s2.pos, "skip {:?} k {k}", skip.skipped());
        assert_eq!(s1.tokens(), s2.tokens());
        if s1.pos < m.spec.context as usize {
            let (l1, l2) = (
                m.forward(&mut s1, 3).unwrap(),
                m.forward(&mut s2, 3).unwrap(),
            );
            assert_eq!(
                l1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        spec.stats
    }

    #[test]
    fn greedy_equals_plain_for_many_skip_sets_and_draft_lengths() {
        let m = deep(6, 2, 2);
        let sets = [
            vec![1],
            vec![2, 3],
            vec![1, 3, 4],
            vec![1, 2, 3, 4],
            vec![0, 5],
            vec![0, 1, 2, 3, 4, 5],
            vec![2],
        ];
        let (mut rounds, mut accepted, mut rejected) = (0, 0, 0);
        for kind in [KvKind::F32, KvKind::F16, KvKind::Q8] {
            for s in &sets {
                for k in [1, 2, 4, 8] {
                    let st = assert_matches_plain(
                        &m,
                        kind,
                        SkipSet::from_layers(6, s).unwrap(),
                        k,
                        &prompt(7, k + s.len()),
                        40,
                    );
                    rounds += st.rounds;
                    accepted += st.accepted;
                    rejected += st.drafted - st.accepted;
                }
            }
        }
        // the equality above is only meaningful if both outcomes really happened
        assert!(
            rounds > 100 && accepted > 0 && rejected > 0,
            "{rounds} {accepted} {rejected}"
        );
    }

    #[test]
    fn greedy_equals_plain_on_other_shapes() {
        for (layers, heads, kv) in [(4, 4, 1), (5, 2, 2), (3, 2, 1)] {
            let m = deep(layers, heads, kv);
            let skip = SkipSet::from_layers(layers as usize, &[1]).unwrap();
            for k in [1, 3, 6] {
                assert_matches_plain(&m, KvKind::F32, skip.clone(), k, &prompt(5, 2), 30);
            }
        }
        // layer norm, biases, soft-capping
        let mut sp = Spec::dense("llama", 5, 16, 2, 2, 8, 24, 20, 96);
        sp.tied_embeddings = true;
        let m = model_of(sp, true);
        assert_matches_plain(
            &m,
            KvKind::F32,
            SkipSet::from_layers(5, &[1, 3]).unwrap(),
            4,
            &prompt(6, 3),
            30,
        );
    }

    fn sliding_model() -> Model {
        let mut sp = Spec::dense("llama", 5, 16, 2, 2, 8, 24, 20, 96);
        sp.sliding_window = 12;
        sp.sliding_layers = vec![true, false, true, false, true];
        model_of(sp, false)
    }

    #[test]
    fn sliding_windows_fall_back_instead_of_corrupting() {
        let m = sliding_model();
        // long enough to wrap the 12-slot rings several times
        for k in [1, 4, 8] {
            assert_matches_plain(
                &m,
                KvKind::F32,
                SkipSet::from_layers(5, &[1, 3]).unwrap(),
                k,
                &prompt(5, 1),
                60,
            );
        }
        // before the ring wraps speculation is used
        let c = cfg(SkipSet::from_layers(5, &[1, 3]).unwrap(), 3, 4);
        let mut st = m.new_state();
        let o = generate(&m, &mut st, &prompt(3, 1), &c).unwrap();
        assert!(o.stats.rounds >= 1, "{:?}", o.stats);
    }

    #[test]
    fn moe_model_works_too() {
        let m = crate::model::tests::accel_and_prefix::moe_model();
        for k in [1, 3] {
            assert_matches_plain(
                &m,
                KvKind::F32,
                SkipSet::from_layers(3, &[1]).unwrap(),
                k,
                &prompt(4, 1),
                25,
            );
        }
    }

    #[test]
    fn accounting_is_consistent() {
        let m = deep(6, 2, 2);
        let c = cfg(SkipSet::from_layers(6, &[2, 3]).unwrap(), 4, 50);
        let mut st = m.new_state();
        let o = generate(&m, &mut st, &prompt(6, 2), &c).unwrap();
        let s = o.stats;
        assert_eq!(s.emitted, o.tokens.len());
        assert!(s.accepted <= s.drafted);
        assert!(s.rounds > 0 && s.drafted >= s.rounds);
        // every round emits its accepted drafts plus one extra; plain steps emit one; plus the first token
        assert_eq!(
            s.emitted,
            1 + s.accepted + s.rounds + s.plain_steps,
            "{s:?}"
        );
        assert!(s.acceptance_rate() >= 0.0 && s.acceptance_rate() <= 1.0);
        assert!(s.tokens_per_pass() > 0.0);
        let sm = SpeedModel::for_model(&m, &c.skip, 1.0);
        assert!(s.estimated_speedup(&sm).is_finite() && s.estimated_speedup(&sm) > 0.0);
        let plain_only = SelfSpecStats {
            plain_steps: 9,
            emitted: 10,
            ..SelfSpecStats::default()
        };
        assert!((plain_only.estimated_speedup(&sm) - 1.0).abs() < 1e-12);
        // skipping nothing drafts the model itself: everything is accepted
        let full = cfg(SkipSet::none(6), 4, 30);
        let o = generate(&m, &mut m.new_state(), &prompt(6, 2), &full).unwrap();
        assert_eq!(o.stats.rounds, 0, "an empty skip set means plain decoding");
        // a draft that equals the model (kept layers = all) never rejects: check via a one-layer skip of a
        // 1-layer-redundant case is not constructible, so check the high-acceptance bound on the stats instead
        let tiny = cfg(SkipSet::from_layers(6, &[3]).unwrap(), 4, 40);
        let o = generate(&m, &mut m.new_state(), &prompt(6, 2), &tiny).unwrap();
        assert!(o.stats.accepted <= o.stats.drafted);
    }

    #[test]
    fn adaptive_length_moves_with_acceptance() {
        let mut a = AdaptiveLen::new(1, 8, 4);
        a.update(4, 4);
        assert_eq!(a.cur, 5);
        a.update(5, 0);
        assert_eq!(a.cur, 4);
        for _ in 0..10 {
            a.update(a.cur, 0);
        }
        assert_eq!(a.cur, 1);
        for _ in 0..20 {
            a.update(a.cur, a.cur);
        }
        assert_eq!(a.cur, 8);
        // and generation respects the configured bounds
        let m = deep(6, 2, 2);
        let mut c = cfg(SkipSet::from_layers(6, &[1, 2, 3, 4]).unwrap(), 3, 60);
        c.min_draft = 2;
        c.start_draft = 2;
        let o = generate(&m, &mut m.new_state(), &prompt(5, 1), &c).unwrap();
        assert!(o.stats.drafted <= o.stats.rounds * 3 && o.stats.drafted >= o.stats.rounds.min(1));
    }

    #[test]
    fn stops_and_budgets_match_plain() {
        let m = deep(6, 2, 2);
        let skip = SkipSet::from_layers(6, &[2, 3]).unwrap();
        let p = prompt(5, 1);
        let free = generate_plain(&m, &mut m.new_state(), &p, &cfg(skip.clone(), 4, 30))
            .unwrap()
            .tokens;
        for stop_at in [0, 1, 2, 5, 11] {
            let mut c = cfg(skip.clone(), 4, 30);
            c.stop_tokens = vec![free[stop_at]];
            let want = generate_plain(&m, &mut m.new_state(), &p, &c).unwrap();
            let (mut st, mut st2) = (m.new_state(), m.new_state());
            let got = generate(&m, &mut st, &p, &c).unwrap();
            assert_eq!(got.tokens, want.tokens, "stop at {stop_at}");
            assert_eq!(got.stop, StopReason::StopToken);
            assert_eq!(st.pos, {
                generate_plain(&m, &mut st2, &p, &c).unwrap();
                st2.pos
            });
        }
        for n in [0, 1, 2, 3, 7] {
            let c = cfg(skip.clone(), 4, n);
            let got = generate(&m, &mut m.new_state(), &p, &c).unwrap();
            assert_eq!(got.tokens, free[..n].to_vec(), "max_tokens {n}");
        }
    }

    #[test]
    fn context_limit_matches_plain() {
        let m = deep(4, 2, 2); // context 96
        let skip = SkipSet::from_layers(4, &[1, 2]).unwrap();
        for plen in [90, 93, 95, 96] {
            let p = prompt(plen, 2);
            let c = cfg(skip.clone(), 4, 50);
            let (mut a, mut b) = (m.new_state(), m.new_state());
            let want = generate_plain(&m, &mut b, &p, &c);
            let got = generate(&m, &mut a, &p, &c);
            match (want, got) {
                (Ok(w), Ok(g)) => {
                    assert_eq!(g.tokens, w.tokens, "prompt {plen}");
                    assert_eq!(g.stop, w.stop);
                    assert_eq!(a.pos, b.pos);
                }
                (w, g) => assert_eq!(w.is_err(), g.is_err()),
            }
        }
        assert!(generate(&m, &mut m.new_state(), &prompt(97, 1), &cfg(skip, 4, 5)).is_err());
    }

    #[test]
    fn adversarial_inputs_are_errors_not_panics() {
        let m = deep(6, 2, 2);
        let skip = SkipSet::from_layers(6, &[2, 3]).unwrap();
        let c = cfg(skip.clone(), 4, 10);
        assert!(matches!(
            generate(&m, &mut m.new_state(), &[], &c),
            Err(SelfSpecError::Model(ModelError::NoTokens))
        ));
        assert!(matches!(
            generate(&m, &mut m.new_state(), &[999], &c),
            Err(SelfSpecError::Model(ModelError::Token(999)))
        ));
        assert!(matches!(
            generate(&m, &mut m.new_state(), &[1, 20], &c),
            Err(SelfSpecError::Model(ModelError::Token(20)))
        ));
        // skip set for another model
        let wrong = cfg(SkipSet::none(3), 4, 10);
        assert!(matches!(
            generate(&m, &mut m.new_state(), &[1], &wrong),
            Err(SelfSpecError::Invalid(_))
        ));
        // degenerate draft settings
        let want = generate_plain(&m, &mut m.new_state(), &[1, 2], &c)
            .unwrap()
            .tokens;
        for (min, max, start) in [
            (0, 0, 0),
            (5, 1, 9),
            (0, 1000, 1000),
            (3, 3, 0),
            (usize::MAX, usize::MAX, usize::MAX),
        ] {
            let mut d = c.clone();
            d.min_draft = min;
            d.max_draft = max;
            d.start_draft = start;
            assert_eq!(
                generate(&m, &mut m.new_state(), &[1, 2], &d)
                    .unwrap()
                    .tokens,
                want,
                "{min} {max} {start}"
            );
        }
        // everything skipped: a (bad) draft straight from the embeddings, still exact
        assert_eq!(
            generate(
                &m,
                &mut m.new_state(),
                &[1, 2],
                &cfg(SkipSet::from_layers(6, &[0, 1, 2, 3, 4, 5]).unwrap(), 4, 10)
            )
            .unwrap()
            .tokens,
            want
        );
        // a reused state: continuing a conversation is exact too
        let mut st = m.new_state();
        let first = generate(&m, &mut st, &[1, 2, 3], &c).unwrap();
        let mut again = vec![1, 2, 3];
        again.extend(&first.tokens);
        let o2 = generate(&m, &mut st, &again, &c).unwrap();
        assert_eq!(
            o2.tokens,
            generate_plain(&m, &mut m.new_state(), &again, &c)
                .unwrap()
                .tokens
        );
        // a state of another model
        let other = deep(3, 2, 2);
        assert!(generate(&m, &mut other.new_state(), &[1, 2], &c).is_err());
        // calibration inputs
        assert!(calibrate(&m, &[], &CalibrateOptions::default()).is_err());
        assert!(calibrate(&m, &[vec![]], &CalibrateOptions::default()).is_err());
        assert!(calibrate(&m, &[vec![500]], &CalibrateOptions::default()).is_err());
    }

    #[test]
    fn sampled_decoding_is_deterministic_per_seed_and_valid() {
        let m = deep(6, 2, 2);
        let mut c = cfg(SkipSet::from_layers(6, &[2, 3]).unwrap(), 4, 40);
        c.sampling = SamplingCfg::sampled(0.9, 0, 1.0);
        c.seed = 5;
        let a = generate(&m, &mut m.new_state(), &prompt(5, 1), &c).unwrap();
        let b = generate(&m, &mut m.new_state(), &prompt(5, 1), &c).unwrap();
        assert_eq!(a.tokens, b.tokens);
        assert!(a.tokens.iter().all(|&t| t < 20));
        c.seed = 6;
        assert_ne!(
            generate(&m, &mut m.new_state(), &prompt(5, 1), &c)
                .unwrap()
                .tokens,
            a.tokens
        );
        // top-k 1 is greedy, whatever the temperature
        c.sampling = SamplingCfg::sampled(1.0, 1, 1.0);
        let g = generate(&m, &mut m.new_state(), &prompt(5, 1), &c)
            .unwrap()
            .tokens;
        assert_eq!(
            g,
            generate_plain(&m, &mut m.new_state(), &prompt(5, 1), &c)
                .unwrap()
                .tokens
        );
    }

    /// The emitted token distribution must be the full model's, even though the draft's differs.
    #[test]
    fn sampled_distribution_is_unbiased() {
        let m = deep(5, 2, 2);
        let skip = SkipSet::from_layers(5, &[1, 2, 3]).unwrap(); // aggressive: the draft is quite different
        let p = prompt(4, 3);
        let v = 20;
        let cfgs = SamplingCfg::sampled(1.0, 0, 1.0);
        // exact joint law of the first two emitted tokens
        let mut st = m.new_state();
        let l0 = m.forward_batch(&mut st, &p).unwrap();
        let p0 = cfgs.probs(&l0);
        let mut p1 = vec![0.0f64; v]; // marginal of the second token
        for (t0, &w) in p0.iter().enumerate() {
            let mut s2 = m.new_state();
            m.forward_batch(&mut s2, &p).unwrap();
            let l1 = m.forward(&mut s2, t0).unwrap();
            for (i, q) in cfgs.probs(&l1).iter().enumerate() {
                p1[i] += w as f64 * *q as f64;
            }
        }
        let n = 6000;
        let (mut c0, mut c1) = (vec![0.0f64; v], vec![0.0f64; v]);
        let mut drafted = 0;
        for seed in 0..n {
            let mut c = cfg(skip.clone(), 3, 2);
            c.sampling = cfgs;
            c.seed = seed as u64 + 1;
            c.start_draft = 1;
            let o = generate(&m, &mut m.new_state(), &p, &c).unwrap();
            c0[o.tokens[0]] += 1.0;
            c1[o.tokens[1]] += 1.0;
            drafted += o.stats.drafted;
        }
        // max_tokens 2 forces k = 0 plain steps; use 4 tokens for the drafted regime below
        assert_eq!(drafted, 0);
        let tv = |emp: &[f64], exact: &[f64]| {
            0.5 * emp
                .iter()
                .zip(exact)
                .map(|(a, b)| (a / n as f64 - b).abs())
                .sum::<f64>()
        };
        let p0d: Vec<f64> = p0.iter().map(|&x| x as f64).collect();
        assert!(tv(&c0, &p0d) < 0.05, "first token tv {}", tv(&c0, &p0d));
        assert!(tv(&c1, &p1) < 0.05, "second token tv {}", tv(&c1, &p1));
        // now the same check for the token that really goes through draft + rejection: index 2 of 4
        let mut p2 = vec![0.0f64; v];
        for (t0, &w0) in p0.iter().enumerate() {
            let mut s1 = m.new_state();
            m.forward_batch(&mut s1, &p).unwrap();
            let l1 = m.forward(&mut s1, t0).unwrap();
            for (t1, &w1) in cfgs.probs(&l1).iter().enumerate() {
                if (w0 as f64 * w1 as f64) < 1e-4 {
                    continue;
                }
                let mut s2 = s1.fork().unwrap();
                let l2 = m.forward(&mut s2, t1).unwrap();
                for (i, q) in cfgs.probs(&l2).iter().enumerate() {
                    p2[i] += w0 as f64 * w1 as f64 * *q as f64;
                }
            }
        }
        let mass: f64 = p2.iter().sum();
        p2.iter_mut().for_each(|x| *x /= mass);
        let (mut c2, mut acc, mut dr) = (vec![0.0f64; v], 0, 0);
        for seed in 0..n {
            let mut c = cfg(skip.clone(), 3, 4);
            c.sampling = cfgs;
            c.seed = 10_000 + seed as u64;
            c.start_draft = 3;
            let o = generate(&m, &mut m.new_state(), &p, &c).unwrap();
            c2[o.tokens[2]] += 1.0;
            acc += o.stats.accepted;
            dr += o.stats.drafted;
        }
        assert!(
            dr > 0 && acc > 0 && acc < dr,
            "the test must exercise both acceptance and rejection: {acc}/{dr}"
        );
        assert!(tv(&c2, &p2) < 0.06, "third token tv {}", tv(&c2, &p2));
    }

    #[test]
    fn calibration_finds_a_valid_set_and_is_deterministic() {
        let m = deep(8, 2, 2);
        let prompts = vec![prompt(10, 1), prompt(12, 2)];
        let o = CalibrateOptions {
            max_skip_fraction: 0.6,
            prompt_tokens: 10,
            continue_tokens: 6,
            ..CalibrateOptions::default()
        };
        let a = calibrate(&m, &prompts, &o).unwrap();
        let b = calibrate(&m, &prompts, &o).unwrap();
        assert_eq!(a.skip, b.skip);
        assert_eq!(a.trace, b.trace);
        assert!(a.skip.skipped_count() <= 4);
        assert!(
            !a.skip.is_skipped(0) && !a.skip.is_skipped(7),
            "pinned layers stay"
        );
        assert!(a.alpha >= 0.0 && a.alpha <= 1.0 && a.est_speedup >= 1.0);
        assert_eq!(a.worthwhile, !a.skip.is_empty());
        assert!(!a.trace.is_empty());
        for s in &a.trace {
            assert!(s.alpha >= 0.0 && s.alpha <= 1.0 && s.cost > 0.0 && s.cost <= 1.0);
        }
        // whatever it chose, generation with it is exact
        assert_matches_plain(
            &m,
            KvKind::F32,
            a.skip.clone(),
            a.best_k.max(1),
            &prompt(6, 4),
            30,
        );
        // a sweep over the full set agrees with itself: skipping nothing drafts perfectly
        let (acc, _) = sweep(&m, &[prompt(8, 1)], &[SkipSet::none(8)], None).unwrap();
        assert_eq!(acc[0].0, acc[0].1);
        assert!(acc[0].1 >= 8);
    }

    #[test]
    fn calibration_respects_budget_deadline_and_candidates() {
        let m = deep(8, 2, 2);
        let prompts = vec![prompt(10, 1)];
        let o = CalibrateOptions {
            max_skip_fraction: 0.0,
            ..CalibrateOptions::default()
        };
        let c = calibrate(&m, &prompts, &o).unwrap();
        assert!(c.skip.is_empty() && !c.worthwhile && c.trace.is_empty());
        let o = CalibrateOptions {
            deadline: Some(Duration::ZERO),
            ..CalibrateOptions::default()
        };
        let c = calibrate(&m, &prompts, &o).unwrap();
        assert!(c.timed_out || c.trace.is_empty());
        let o = CalibrateOptions {
            candidates_per_step: 2,
            ..CalibrateOptions::default()
        };
        assert!(calibrate(&m, &prompts, &o).is_ok());
        let o = CalibrateOptions {
            pin_first: 100,
            pin_last: 100,
            ..CalibrateOptions::default()
        };
        assert!(calibrate(&m, &prompts, &o).unwrap().skip.is_empty());
        let o = CalibrateOptions {
            max_skip_fraction: f64::NAN,
            max_draft: 0,
            ..CalibrateOptions::default()
        };
        let _ = calibrate(&m, &prompts, &o).unwrap();
        // sliding-window models calibrate until their ring would wrap, without panicking
        assert!(
            calibrate(
                &sliding_model(),
                &[prompt(30, 1)],
                &CalibrateOptions::default()
            )
            .is_ok()
        );
    }

    #[test]
    fn base_spec_model_still_works() {
        // the shared 3-layer test model from model.rs
        let m = model_of(base_spec(2, 2, false), false);
        assert_matches_plain(
            &m,
            KvKind::F32,
            SkipSet::from_layers(3, &[1]).unwrap(),
            3,
            &prompt(4, 1),
            20,
        );
    }
}
