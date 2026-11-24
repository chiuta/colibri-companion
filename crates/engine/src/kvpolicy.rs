//! KV-cache eviction policies: StreamingLLM attention sinks, H2O heavy hitters, SnapKV prompt
//! compression, and per-layer budgets, applied to a live [`State`] with [`State::apply_policy`].
//!
//! # How eviction works here
//!
//! A [`Policy`] looks at the per-slot importance scores and the cache length and returns the slots to
//! keep. [`State::apply_policy`] then compacts every layer in every precision (F32, F16, Q8, copied
//! bit for bit, never re-encoded), rebuilds sliding-window rings, and fixes the position counter and
//! the token history. Keeping everything (`budget >= len`) changes nothing at all: the logits that
//! follow are bit-identical to those of an untouched cache.
//!
//! # RoPE: what happens to kept keys
//!
//! Keys are stored *after* rotary embedding: the key written at position `p` already carries the
//! rotation for `p`. That has three consequences for eviction.
//!
//! * **No correction** (the new token is rotated at its slot index): the angle between a query and
//!   a kept key then differs from the true one by the number of slots evicted in between. Attention
//!   to a kept token is computed as if it were closer than it was. Short gaps barely matter, big ones
//!   (a long evicted middle) move the sinks to a distance the model never saw during training.
//! * **Position gap** (what this module does): kept keys are left exactly as stored and the next
//!   token is rotated at its *original* position, `pos + gap`, where `gap` grows by the number of
//!   evicted slots ([`State::rope_gap`], [`State::abs_pos`]). Rotary attention depends only on the
//!   difference of positions, so every query–key distance for kept tokens is exactly what it would
//!   have been without eviction. The cost: positions keep growing past the evicted tokens, so after
//!   a very long chat they pass the context length the model was trained for (the engine does not
//!   refuse that; models with scaled RoPE cope, others degrade the way they do on any long input).
//! * **Re-rotation** (StreamingLLM's cache-relative positions, *not* implemented): rotate each kept
//!   key by `(new_slot - old_position)` so positions stay within the trained range. It changes the
//!   distances between kept tokens (the sinks move next to the window), needs the cache rewritten
//!   (an extra rounding step for F16 and Q8), and was left out on purpose.
//!
//! # Limits
//!
//! * Slots are shared by all layers (there is one position counter), so a *per-layer budget* cannot
//!   leave different layers with different slot counts. [`LayerBudgets`] instead weights how much
//!   each layer's attention scores count when the shared keep-set is chosen.
//! * A sliding-window layer is a ring of `window` slots. Eviction must keep the last `window`
//!   positions untouched for those layers to stay exact, so the budget never goes below the largest
//!   ring (see [`Compaction::kept_tail`]).
//! * Evicted tokens are gone: [`Model::prefill_cached`] reuses only the leading tokens that still sit
//!   at their own position and recomputes the rest.
//! * Accumulated scores cost one extra pass over the cache per layer and query, so they are off
//!   unless [`State::track_scores`] is switched on. The decoder does no extra work when it is off.

use crate::model::{Model, ModelError, State};

/// Chooses which cache slots survive.
pub trait Policy {
    /// `scores[i]` is the importance of slot `i` (all zero when the policy does not need scores),
    /// `len` the number of slots, `budget` how many to keep (`budget < len`). Return strictly
    /// increasing slot numbers, at most `budget` of them.
    fn select_keep(&self, scores: &[f32], len: usize, budget: usize) -> Vec<usize>;
    /// Does the policy read `scores`? Then the state must have been tracking them.
    fn needs_scores(&self) -> bool { false }
    fn name(&self) -> &'static str;
}

/// StreamingLLM: the first `sinks` tokens plus the most recent ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamingLlm { pub sinks: usize }
impl Policy for StreamingLlm {
    fn select_keep(&self, _scores: &[f32], len: usize, budget: usize) -> Vec<usize> {
        if budget >= len { return (0..len).collect(); }
        let sinks = self.sinks.min(budget);
        let recent = budget - sinks;
        (0..sinks).chain(len - recent..len).collect()
    }
    fn name(&self) -> &'static str { "streaming-llm" }
}

/// Top `k` slots by score (ties go to the earlier slot), in slot order.
fn top_by_score(scores: &[f32], range: std::ops::Range<usize>, k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = range.collect();
    idx.sort_by(|&a, &b| scores.get(b).copied().unwrap_or(0.0).total_cmp(&scores.get(a).copied().unwrap_or(0.0)).then(a.cmp(&b)));
    idx.truncate(k);
    idx.sort_unstable();
    idx
}

/// H2O: the most recent tokens plus the "heavy hitters" with the most accumulated attention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct H2o {
    /// Share of the budget reserved for the newest tokens (the rest goes to heavy hitters).
    pub recent_share: f32,
}
impl Default for H2o { fn default() -> Self { H2o { recent_share: 0.5 } } }
impl Policy for H2o {
    fn select_keep(&self, scores: &[f32], len: usize, budget: usize) -> Vec<usize> {
        if budget >= len { return (0..len).collect(); }
        let recent = ((budget as f32 * self.recent_share.clamp(0.0, 1.0)).round() as usize).min(budget);
        let mut keep = top_by_score(scores, 0..len - recent, budget - recent);
        keep.extend(len - recent..len);
        keep
    }
    fn needs_scores(&self) -> bool { true }
    fn name(&self) -> &'static str { "h2o" }
}

/// SnapKV: after prefill, the last `window` prompt tokens vote (their attention, smoothed with an
/// average pool of `pool` slots so neighbours of important tokens survive too) for what to keep.
/// Feed it scores gathered only over that window: see [`prefill_snapkv`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapKv { pub window: usize, pub pool: usize }
impl Policy for SnapKv {
    fn select_keep(&self, scores: &[f32], len: usize, budget: usize) -> Vec<usize> {
        if budget >= len { return (0..len).collect(); }
        let window = self.window.min(budget);
        let head = len - window;
        let r = self.pool / 2;
        let pooled: Vec<f32> = (0..head).map(|i| {
            let (lo, hi) = (i.saturating_sub(r), (i + r + 1).min(head));
            scores[lo.min(scores.len())..hi.min(scores.len())].iter().sum::<f32>() / (hi - lo).max(1) as f32
        }).collect();
        let mut keep = top_by_score(&pooled, 0..head, budget - window);
        keep.extend(head..len);
        keep
    }
    fn needs_scores(&self) -> bool { true }
    fn name(&self) -> &'static str { "snapkv" }
}

/// How much each layer's attention scores count when the shared keep-set is chosen.
#[derive(Clone, Debug, PartialEq)]
pub enum LayerBudgets {
    /// Every layer counts the same.
    Uniform,
    /// Weights fall linearly with depth from `1.0` (first layer) to `last` (final layer), as in
    /// PyramidKV, where early layers spread their attention wide and deserve more of the cache.
    Pyramid { last: f32 },
    /// One non-negative weight per layer.
    Weights(Vec<f32>),
}
impl LayerBudgets {
    /// Weights summing to one (all zero when nothing counts).
    pub fn weights(&self, layers: usize) -> Vec<f32> {
        let raw: Vec<f32> = match self {
            LayerBudgets::Uniform => vec![1.0; layers],
            LayerBudgets::Pyramid { last } => (0..layers).map(|i| {
                let t = if layers > 1 { i as f32 / (layers - 1) as f32 } else { 0.0 };
                (1.0 + (last - 1.0) * t).max(0.0)
            }).collect(),
            LayerBudgets::Weights(w) => (0..layers).map(|i| w.get(i).copied().unwrap_or(0.0).max(0.0)).collect(),
        };
        let total: f32 = raw.iter().sum();
        if total > 0.0 { raw.iter().map(|w| w / total).collect() } else { vec![0.0; layers] }
    }
}

/// What [`State::apply_policy`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Compaction {
    pub before: usize,
    pub after: usize,
    pub evicted: usize,
    /// Newest slots kept regardless of the policy so sliding-window layers stay exact.
    pub kept_tail: usize,
    /// RoPE position of the next token minus the slot it will use.
    pub rope_gap: usize,
    pub bytes_before: usize,
    pub bytes_after: usize,
}

/// Per-slot importance: every tracked layer's scores, normalised to sum to one, weighted by `layers`.
pub fn pooled_scores(st: &State, layers: &LayerBudgets) -> Vec<f32> {
    let n = st.pos;
    let w = layers.weights(st.layer_count());
    let mut out = vec![0f32; n];
    for (li, wl) in w.iter().enumerate() {
        let Some(s) = st.layer_scores(li) else { continue };
        let total: f32 = s.iter().take(n).sum();
        if total <= 0.0 || *wl == 0.0 { continue; }
        for (o, v) in out.iter_mut().zip(s) { *o += wl * v / total; }
    }
    out
}

impl State {
    /// Evict slots with `policy` until at most `budget` remain (more when sliding-window layers force
    /// a longer tail, see [`Compaction::kept_tail`]). `layers` weights the attention scores of the
    /// layers for policies that read them. `budget >= pos` does nothing.
    pub fn apply_policy(&mut self, policy: &dyn Policy, budget: usize, layers: &LayerBudgets) -> Result<Compaction, ModelError> {
        let (len, bytes_before) = (self.pos, self.kv_bytes());
        let report = |st: &State, tail: usize| Compaction { before: len, after: st.pos, evicted: len - st.pos, kept_tail: tail, rope_gap: st.rope_gap(), bytes_before, bytes_after: st.kv_bytes() };
        if budget == 0 { return Err(ModelError::Shape("a cache budget of zero keeps nothing".into())); }
        if budget >= len { return Ok(report(self, 0)); }
        if policy.needs_scores() && !self.tracks_scores() {
            return Err(ModelError::Shape(format!("the {} policy reads attention scores: call track_scores(true) before feeding tokens", policy.name())));
        }
        let tail = self.ring_caps().into_iter().max().unwrap_or(0).min(len);
        let head = len - tail;
        let room = budget.saturating_sub(tail);
        let mut keep = if head == 0 || room == 0 { vec![] } else {
            let scores = if policy.needs_scores() { pooled_scores(self, layers) } else { vec![0.0; len] };
            let mut k = policy.select_keep(&scores[..head], head, room.min(head));
            k.sort_unstable();
            k.dedup();
            if k.last().is_some_and(|&l| l >= head) || k.len() > room {
                return Err(ModelError::Shape(format!("the {} policy returned slots outside its range or more than its budget", policy.name())));
            }
            k
        };
        keep.extend(head..len);
        self.compact(&keep)?;
        Ok(report(self, tail))
    }
}

/// SnapKV prefill: run the prompt on a fresh `st`, let its last `policy.window` tokens vote, then
/// compress the cache to `budget` slots. Returns the logits after the full prompt (computed with
/// the whole cache, as usual) and what was evicted.
pub fn prefill_snapkv(model: &Model, st: &mut State, tokens: &[usize], policy: &SnapKv, budget: usize, layers: &LayerBudgets) -> Result<(Vec<f32>, Compaction), ModelError> {
    if tokens.is_empty() { return Err(ModelError::NoTokens); }
    let w = policy.window.min(tokens.len());
    if st.pos != 0 { return Err(ModelError::Shape("prefill_snapkv needs a fresh state".into())); }
    if tokens.len() > w { model.forward_batch(st, &tokens[..tokens.len() - w])?; }
    st.track_scores(true);
    let logits = model.forward_batch(st, &tokens[tokens.len() - w..])?;
    let c = st.apply_policy(policy, budget, layers)?;
    st.track_scores(false);
    Ok((logits, c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::{base_spec, model_of, random_weights, reference_logits, tiny};
    use crate::model::{KvKind, Weight};

    const KINDS: [KvKind; 3] = [KvKind::F32, KvKind::F16, KvKind::Q8];
    /// A prompt without token 0 after its first position (token 0 is the engineered sink below).
    fn prompt(n: usize) -> Vec<usize> { std::iter::once(0).chain((1..n).map(|i| 1 + (i * 7 + 3) % 19)).collect() }
    fn l2(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt() }
    fn norm2(a: &[f32]) -> f32 { a.iter().map(|x| x * x).sum::<f32>().sqrt() }

    /// A model whose first token is a true attention sink in layer 0: every query puts nearly all its
    /// attention on it (a rank-one bump in the key projection plus a query bias on the low-frequency
    /// rotary pairs, which barely turn over 32 positions).
    fn sink_model() -> Model {
        let sp = base_spec(4, 2, true);
        let mut w = random_weights(&sp, 0x5151_aaaa_1234_9876, true);
        let (h, hd, qd, kd) = (16usize, 8usize, 32usize, 16usize);
        let mut e0 = vec![0.0; h];
        let x: Vec<f32> = w.embed.as_dense().unwrap()[0..h].iter().map(|v| v * sp.embed_scale as f32).collect();
        crate::model::norm(&sp, &x, &w.layers[0].attn_norm, w.layers[0].attn_norm_b.as_deref(), &mut e0);
        let n2: f32 = e0.iter().map(|v| v * v).sum();
        let lo = [2usize, 3, 6, 7];
        let l0 = &mut w.layers[0];
        if let Weight::Dense(wk) = &mut l0.wk {
            for r in 0..kd { if lo.contains(&(r % hd)) { for c in 0..h { wk[r * h + c] += 20.0 * e0[c] / n2; } } }
        }
        let bq = l0.bq.as_mut().unwrap();
        for (r, b) in bq.iter_mut().enumerate().take(qd) { if lo.contains(&(r % hd)) { *b += 2.0; } }
        Model::new(sp, w).unwrap()
    }

    fn fed(m: &Model, kind: KvKind, toks: &[usize]) -> State {
        let mut st = m.new_state_with(kind);
        m.forward_batch(&mut st, toks).unwrap();
        st
    }
    fn next(m: &Model, st: &State, t: usize) -> Vec<f32> { m.forward(&mut st.fork().unwrap(), t).unwrap() }

    #[test]
    fn a_budget_that_covers_the_cache_is_exactly_lossless_in_every_precision_and_model_shape() {
        let mk_sliding = || { let mut sp = base_spec(4, 2, true); sp.sliding_window = 6; sp.sliding_layers = vec![true, false, true]; sp.rope_theta_local = 500.0; model_of(sp, false) };
        let models = [("dense", tiny(4, 2, false)), ("sliding", mk_sliding()), ("moe", crate::model::tests::accel_and_prefix::moe_model())];
        let toks = prompt(14);
        let policies: [(&str, Box<dyn Policy>); 3] = [("sinks", Box::new(StreamingLlm { sinks: 3 })), ("h2o", Box::new(H2o::default())), ("snapkv", Box::new(SnapKv { window: 4, pool: 3 }))];
        for (name, m) in &models {
            for kind in KINDS {
                let want = next(m, &fed(m, kind, &toks), 5);
                for (pname, p) in &policies {
                    let mut st = m.new_state_with(kind);
                    st.track_scores(true);
                    m.forward_batch(&mut st, &toks).unwrap();
                    let c = st.apply_policy(p.as_ref(), toks.len() + 3, &LayerBudgets::Uniform).unwrap();
                    assert_eq!((c.evicted, c.rope_gap, st.pos, st.is_compacted()), (0, 0, 14, false));
                    assert_eq!(next(m, &st, 5), want, "{name} {kind:?} {pname}: lossless budget changed the logits");
                    // the full-length selection is the identity too
                    st.compact(&(0..14).collect::<Vec<_>>()).unwrap();
                    assert_eq!(next(m, &st, 5), want);
                }
            }
        }
    }

    #[test]
    fn keeping_the_newest_slots_keeps_every_query_key_distance_for_a_one_layer_model() {
        // With one layer a token's key and value depend only on the token and its position, so an
        // evicted cache must equal a fresh model that only ever saw the kept tokens, whatever their
        // absolute positions: rotary attention sees differences. Without the position gap the
        // new token would be rotated at the slot index and the result would differ.
        let mut sp = base_spec(4, 2, true);
        sp.layers = 1;
        let m = model_of(sp, false);
        let toks: Vec<usize> = (0..20).map(|i| (i * 5 + 2) % 20).collect();
        let kind = KvKind::F32;
        let mut st = fed(&m, kind, &toks);
        let keep: Vec<usize> = (12..20).collect();
        st.compact(&keep).unwrap();
        assert_eq!((st.pos, st.rope_gap(), st.rope_pos(), st.abs_pos(0), st.abs_pos(7)), (8, 12, 20, 12, 19));
        let got = m.forward(&mut st, 4).unwrap();
        let mut seq: Vec<usize> = toks[12..].to_vec();
        seq.push(4);
        let want = reference_logits(&m, &seq);
        let err = l2(&got, &want) / norm2(&want);
        assert!(err < 2e-4, "relative error {err}");
    }

    #[test]
    fn a_hole_in_the_middle_keeps_original_positions_and_history() {
        let m = tiny(4, 2, false);
        let toks = prompt(16);
        let mut st = fed(&m, KvKind::F32, &toks);
        st.compact(&[0, 1, 2, 11, 12, 13, 14, 15]).unwrap();
        assert_eq!(st.tokens(), &[toks[0], toks[1], toks[2], toks[11], toks[12], toks[13], toks[14], toks[15]]);
        assert_eq!((st.pos, st.rope_gap(), st.abs_pos(2), st.abs_pos(3), st.abs_pos(7)), (8, 8, 2, 11, 15));
        m.forward(&mut st, 3).unwrap();
        assert_eq!((st.pos, st.rope_pos(), st.abs_pos(8)), (9, 17, 8));
        // a second eviction composes with the first
        st.compact(&[0, 1, 4, 5, 6, 7, 8]).unwrap();
        assert_eq!((st.pos, st.abs_pos(2), st.abs_pos(3), st.rope_pos()), (7, 12, 13, 17));
        assert!(st.compact(&[]).is_err() && st.compact(&[3, 3]).is_err() && st.compact(&[0, 99]).is_err() && st.compact(&[4, 2]).is_err());
    }

    #[test]
    fn sinks_plus_window_beats_plain_truncation_when_the_first_token_is_a_sink() {
        let m = sink_model();
        let toks = prompt(26);
        let st = fed(&m, KvKind::F32, &toks);
        let full = next(&m, &st, 9);
        let run = |p: &dyn Policy| { let mut s = st.fork().unwrap(); s.apply_policy(p, 10, &LayerBudgets::Uniform).unwrap(); assert_eq!(s.pos, 10); next(&m, &s, 9) };
        let sinks = run(&StreamingLlm { sinks: 2 });
        let plain = run(&StreamingLlm { sinks: 0 });
        let (e_sinks, e_plain) = (l2(&sinks, &full) / norm2(&full), l2(&plain, &full) / norm2(&full));
        assert!(e_sinks < 0.5 * e_plain, "sinks {e_sinks}, plain truncation {e_plain}");
        // and the saving is real
        let mut s = st.fork().unwrap();
        let c = s.apply_policy(&StreamingLlm { sinks: 2 }, 10, &LayerBudgets::Uniform).unwrap();
        assert_eq!((c.before, c.after, c.evicted, c.rope_gap), (26, 10, 16, 16));
        assert!(c.bytes_after * 2 < c.bytes_before);
    }

    #[test]
    fn heavy_hitters_find_the_sink_and_beat_plain_truncation() {
        let m = sink_model();
        let toks = prompt(26);
        let mut st = m.new_state();
        st.track_scores(true);
        m.forward_batch(&mut st, &toks).unwrap();
        let sc = pooled_scores(&st, &LayerBudgets::Uniform);
        let top = sc.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(top, 0, "the engineered sink must hold the most attention: {sc:?}");
        let full = next(&m, &st, 9);
        let mut s = st.fork().unwrap();
        let c = s.apply_policy(&H2o::default(), 10, &LayerBudgets::Pyramid { last: 0.2 }).unwrap();
        assert_eq!(c.after, 10);
        assert_eq!(s.abs_pos(0), 0, "the heavy hitter survives");
        let mut plain = st.fork().unwrap();
        plain.apply_policy(&StreamingLlm { sinks: 0 }, 10, &LayerBudgets::Uniform).unwrap();
        let (e_h2o, e_plain) = (l2(&next(&m, &s, 9), &full) / norm2(&full), l2(&next(&m, &plain, 9), &full) / norm2(&full));
        assert!(e_h2o < 0.5 * e_plain, "h2o {e_h2o}, plain {e_plain}");
        // asking for scores that were never tracked is an error, not silent zeros
        let mut cold = fed(&m, KvKind::F32, &toks);
        assert!(cold.apply_policy(&H2o::default(), 10, &LayerBudgets::Uniform).is_err());
    }

    #[test]
    fn tracking_scores_does_not_change_the_numbers_and_costs_nothing_when_off() {
        let m = tiny(4, 2, false);
        let toks = prompt(12);
        let (mut a, mut b) = (m.new_state(), m.new_state());
        b.track_scores(true);
        assert!(!a.tracks_scores() && b.tracks_scores());
        assert_eq!(m.forward_batch(&mut a, &toks).unwrap(), m.forward_batch(&mut b, &toks).unwrap());
        assert!(a.layer_scores(0).is_none());
        // every query adds one unit of probability per head: 12 positions of 4 heads, causal
        let total: f32 = b.layer_scores(0).unwrap().iter().sum();
        assert!((total - 12.0 * 4.0).abs() < 1e-3, "{total}");
        b.truncate(5).unwrap();
        assert_eq!(b.layer_scores(1).unwrap().len(), 5);
        b.reset();
        assert!(b.tracks_scores() && b.layer_scores(0).unwrap().is_empty());
    }

    #[test]
    fn snapkv_prefill_compresses_after_the_full_prompt_and_keeps_its_window() {
        let m = sink_model();
        let toks = prompt(24);
        let mut st = m.new_state();
        let (logits, c) = prefill_snapkv(&m, &mut st, &toks, &SnapKv { window: 4, pool: 3 }, 9, &LayerBudgets::Uniform).unwrap();
        assert_eq!(logits, m.forward_batch(&mut m.new_state(), &toks).unwrap());
        assert_eq!((c.before, c.after, st.pos, st.tracks_scores()), (24, 9, 9, false));
        assert_eq!(&st.tokens()[5..], &toks[20..], "the observation window is kept");
        assert_eq!(st.abs_pos(0), 0, "the sink wins the vote");
        let more = m.forward(&mut st, 7).unwrap();
        assert!(more.iter().all(|v| v.is_finite()));
        assert!(prefill_snapkv(&m, &mut st, &toks, &SnapKv { window: 4, pool: 3 }, 9, &LayerBudgets::Uniform).is_err(), "needs a fresh state");
    }

    #[test]
    fn sliding_window_layers_stay_exact_and_the_tail_is_protected() {
        // every layer a ring of 5: whatever is evicted behind the last 5 slots cannot matter
        let mut sp = base_spec(4, 2, true);
        sp.sliding_window = 5; sp.sliding_layers = vec![true; 3]; sp.rope_theta_local = 500.0;
        let m = model_of(sp, false);
        let toks = prompt(22);
        for kind in KINDS {
            let st = fed(&m, kind, &toks);
            let want = next(&m, &st, 6);
            for budget in [1, 5, 8] {
                let mut s = st.fork().unwrap();
                let c = s.apply_policy(&StreamingLlm { sinks: 2 }, budget, &LayerBudgets::Uniform).unwrap();
                assert_eq!(c.kept_tail, 5);
                assert_eq!(s.pos, budget.max(5).max(if budget == 8 { 8 } else { 0 }), "budget {budget}");
                assert_eq!(next(&m, &s, 6), want, "{kind:?} budget {budget}: ring layers must not notice");
                // keep decoding past a wrap of the rebuilt ring
                let mut s = s;
                for t in 0..12 { let l = m.forward(&mut s, 1 + t % 18).unwrap(); assert!(l.iter().all(|v| v.is_finite())); }
            }
        }
        // a hand-made subset that drops something a ring still needs is refused
        let mut s = fed(&m, KvKind::F32, &toks);
        assert!(s.compact(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21]).is_err());
        assert!(s.compact(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21]).is_ok(), "dropping behind the ring's window is fine");
    }

    #[test]
    fn mixed_models_survive_eviction_and_keep_decoding() {
        let mut sp = base_spec(4, 2, true);
        sp.sliding_window = 4; sp.sliding_layers = vec![true, false, true]; sp.rope_theta_local = 500.0;
        let m = model_of(sp, false);
        for kind in KINDS {
            let mut st = m.new_state_with(kind);
            st.track_scores(true);
            m.forward_batch(&mut st, &prompt(20)).unwrap();
            let c = st.apply_policy(&H2o::default(), 9, &LayerBudgets::Uniform).unwrap();
            assert_eq!((c.after, c.kept_tail), (9, 4));
            for t in 0..8 { assert!(m.forward(&mut st, 1 + t).unwrap().iter().all(|v| v.is_finite())); }
            assert_eq!(st.pos, 17);
        }
    }

    #[test]
    fn cached_prefill_after_eviction_reuses_only_what_still_sits_at_its_position() {
        let m = tiny(4, 2, false);
        let toks = prompt(18);
        let mut st = fed(&m, KvKind::F32, &toks);
        st.compact(&[0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17]).unwrap();
        assert!(st.is_compacted() && st.can_truncate(3) && !st.can_truncate(5) && st.can_truncate(st.pos));
        let got = m.prefill_cached(&mut st, &toks).unwrap();
        assert_eq!(st.reused(), 3, "only the sinks sit at their own position");
        assert!(!st.is_compacted());
        let want = m.forward_batch(&mut m.new_state(), &toks).unwrap();
        assert!(l2(&got, &want) / norm2(&want) < 1e-5);
        // a prefix-only eviction (gap, but identity positions) is reusable up to its length
        let mut st = fed(&m, KvKind::F32, &toks);
        st.compact(&(0..6).collect::<Vec<_>>()).unwrap();
        assert_eq!((st.rope_gap(), st.is_compacted()), (12, true));
        let got = m.prefill_cached(&mut st, &toks).unwrap();
        assert_eq!((st.reused(), st.rope_gap()), (6, 0));
        assert!(l2(&got, &want) / norm2(&want) < 1e-5);
    }

    #[test]
    fn policies_and_layer_weights_behave_on_plain_inputs() {
        assert_eq!(StreamingLlm { sinks: 2 }.select_keep(&[], 10, 5), vec![0, 1, 7, 8, 9]);
        assert_eq!(StreamingLlm { sinks: 9 }.select_keep(&[], 10, 5), vec![0, 1, 2, 3, 4]);
        assert_eq!(StreamingLlm { sinks: 2 }.select_keep(&[], 4, 9), vec![0, 1, 2, 3]);
        let s = [0.1, 5.0, 0.2, 4.0, 0.3, 0.0, 0.0, 0.0];
        assert_eq!(H2o { recent_share: 0.5 }.select_keep(&s, 8, 4), vec![1, 3, 6, 7]);
        assert_eq!(H2o { recent_share: 0.0 }.select_keep(&s, 8, 3), vec![1, 3, 4]);
        // pooling lets the neighbours of the top slot in
        let sn = SnapKv { window: 2, pool: 3 }.select_keep(&[0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0], 8, 5);
        assert_eq!(sn, vec![1, 2, 3, 6, 7]);
        let w = LayerBudgets::Pyramid { last: 0.5 }.weights(3);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6 && w[0] > w[1] && w[1] > w[2]);
        assert_eq!(LayerBudgets::Weights(vec![0.0, 2.0]).weights(3), vec![0.0, 1.0, 0.0]);
        assert_eq!(LayerBudgets::Weights(vec![]).weights(2), vec![0.0, 0.0]);
        let m = tiny(4, 2, false);
        let mut st = fed(&m, KvKind::F32, &prompt(6));
        assert!(st.apply_policy(&StreamingLlm { sinks: 1 }, 0, &LayerBudgets::Uniform).is_err());
        struct Greedy;
        impl Policy for Greedy { fn select_keep(&self, _: &[f32], len: usize, _: usize) -> Vec<usize> { (0..len).collect() } fn name(&self) -> &'static str { "greedy" } }
        assert!(st.apply_policy(&Greedy, 3, &LayerBudgets::Uniform).is_err(), "a policy may not exceed its budget");
        assert_eq!(st.pos, 6, "a refused policy leaves the cache alone");
    }
}
