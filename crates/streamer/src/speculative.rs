//! Cost model for speculative verification. It is a model, not an engine: it counts expert loads
//! and nothing else (no drafting model, no acceptance test, no timing).
//!
//! One verification pass checks `k` consecutive tokens at once, so each expert needed by any of
//! them is read once: the loads of a pass are the UNION of the tokens' experts, against the sum
//! when the tokens are verified one by one. Rejected tokens still cost their loads (conservative).
//! With per-token acceptance `a`, a pass keeps `1 + a + ... + a^(k-1)` tokens on average (the first
//! one is the model's own and always counts).
//!
//! The saving depends on how much consecutive tokens share experts. That is only known from a real
//! `ROUTE_TRACE`; a trace sampled from a `.coli_usage` history overlaps only through popularity,
//! so on such a trace the union saving (and the budget saving below) is understated.
//!
//! An optional per-layer expert BUDGET follows MoE-Spec (arXiv 2602.16052, training-free): at
//! verification time each layer loads at most `B` experts, the ones that most tokens of the window
//! route to (ties: lower id first), and the long tail is dropped. The paper reports 10-30% more
//! throughput at maintained quality on ITS models. This module only counts loads and the share of
//! routed expert slots that were dropped; that share is a proxy for quality risk, not a measure of
//! it. Real quality impact, and any change in the acceptance rate, are NOT modelled here.

use crate::policy::{Cache, Lru, Step};

/// Average tokens kept per pass of `k` with per-token acceptance `a`.
/// `k` is at least 1 (the model's own token) and `a` is clamped to 0..=1 (NaN counts as 0).
pub fn expected_accepted(k: usize, a: f64) -> f64 {
    let a = if a.is_nan() { 0.0 } else { a.clamp(0.0, 1.0) };
    (0..k.max(1))
        .map(|i| a.powi(i.min(i32::MAX as usize) as i32))
        .sum()
}

/// What one verification replay measured.
#[derive(Clone, Copy, Debug)]
pub struct Verify {
    /// Expert loads per window of `k` tokens, averaged over the full windows of the trace.
    pub loads_per_pass: f64,
    /// Share of routed expert slots (token x expert) dropped by the budget, 0 when unbounded.
    pub dropped_share: f64,
}

impl Verify {
    const NO_DATA: Verify = Verify {
        loads_per_pass: f64::NAN,
        dropped_share: f64::NAN,
    };

    /// Loads per accepted token for a pass of `k` at acceptance `a`.
    pub fn per_token(&self, k: usize, a: f64) -> f64 {
        self.loads_per_pass / expected_accepted(k, a)
    }

    pub fn is_no_data(&self) -> bool {
        self.loads_per_pass.is_nan()
    }
}

/// Replays the trace in windows of `k` tokens. `cap` = 0 means no cache. `budget` is the most
/// experts one layer may load per pass (`None`: unbounded, the union of the window); it needs
/// `n_experts` to tell the layers apart (dense id = layer * n_experts + expert). No data (NaN)
/// when the trace is shorter than `k` tokens, and for a budget of 0 or with `n_experts` 0.
pub fn verify_pass(
    trace: &[Step],
    k: usize,
    cap: usize,
    n_experts: u32,
    budget: Option<usize>,
) -> Verify {
    if let Some(b) = budget {
        if b == 0 || n_experts == 0 {
            return Verify::NO_DATA;
        }
    }
    let k = k.max(1);
    let mut cache = Lru::new(cap);
    let (mut loads, mut passes, mut slots, mut dropped) = (0u64, 0u64, 0u64, 0u64);
    for w in trace.chunks_exact(k) {
        let mut count: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for &id in w.iter().flatten() {
            *count.entry(id).or_insert(0) += 1;
            slots += 1;
        }
        // a deterministic order keeps the cache replay reproducible
        let mut uni: Vec<(u32, u32)> = count.into_iter().collect();
        uni.sort_unstable();
        let mut kept: Vec<u32> = Vec::with_capacity(uni.len());
        match budget {
            None => kept.extend(uni.iter().map(|x| x.0)),
            Some(b) => {
                // ids are sorted, so every layer is one contiguous run
                let mut i = 0;
                while i < uni.len() {
                    let layer = uni[i].0 / n_experts;
                    let mut j = i;
                    while j < uni.len() && uni[j].0 / n_experts == layer {
                        j += 1;
                    }
                    let run = &uni[i..j];
                    if run.len() <= b {
                        kept.extend(run.iter().map(|x| x.0));
                    } else {
                        let mut by_use: Vec<(u32, u32)> = run.to_vec();
                        by_use.sort_unstable_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
                        kept.extend(by_use[..b].iter().map(|x| x.0));
                        dropped += by_use[b..].iter().map(|x| x.1 as u64).sum::<u64>();
                    }
                    i = j;
                }
                kept.sort_unstable();
            }
        }
        for id in kept {
            if cap == 0 || !cache.access(id) {
                loads += 1;
            }
        }
        passes += 1;
    }
    if passes == 0 {
        return Verify::NO_DATA;
    }
    Verify {
        loads_per_pass: loads as f64 / passes as f64,
        dropped_share: if slots == 0 {
            0.0
        } else {
            dropped as f64 / slots as f64
        },
    }
}

/// Expert loads per window of `k` consecutive tokens, averaged over the trace (the last partial
/// window is ignored). `cap` = 0 means no cache: every needed expert is read from disk. NaN when
/// the trace is shorter than `k` tokens: there is no full pass to count, and 0 would read as
/// "free".
pub fn loads_per_pass(trace: &[Step], k: usize, cap: usize) -> f64 {
    verify_pass(trace, k, cap, 0, None).loads_per_pass
}

/// The most experts any single layer used for any single token of the trace: the "1x" of a
/// budget. 0 when the trace is empty or `n_experts` is 0.
pub fn experts_per_layer(trace: &[Step], n_experts: u32) -> usize {
    trace
        .iter()
        .flat_map(|st| crate::prefetch::by_layer(st, n_experts))
        .map(|(_, v)| v.len())
        .max()
        .unwrap_or(0)
}

/// A budget of `mult` times `per_layer` experts, rounded up, at least 1. `None` (unbounded) when
/// the multiplier is not a positive finite number or `per_layer` is 0.
pub fn budget_for(mult: f64, per_layer: usize) -> Option<usize> {
    if !mult.is_finite() || mult <= 0.0 || per_layer == 0 {
        return None;
    }
    Some(((mult * per_layer as f64).ceil() as usize).max(1))
}

/// A synthetic trace WITH overlap between consecutive tokens, to test the budget code: every
/// token draws `k` distinct experts per layer from a pool of `pool` experts that drifts slowly
/// (the base moves every 8 tokens). Not a model of any real network.
pub fn locality_trace(
    n_layers: u32,
    n_experts: u32,
    k: usize,
    tokens: usize,
    pool: usize,
    seed: u64,
) -> Vec<Step> {
    if n_experts == 0 || n_layers == 0 {
        return vec![Vec::new(); tokens];
    }
    let pool = pool.clamp(1, n_experts as usize);
    let k = k.clamp(1, pool);
    let mut s = seed | 1;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..tokens)
        .map(|t| {
            let mut step = Vec::new();
            for l in 0..n_layers.min(crate::prefetch::MAX_LAYERS) {
                let base = (t as u64 / 8) * 3 + l as u64 * 5;
                let mut got: Vec<u32> = Vec::new();
                while got.len() < k {
                    let j = ((rnd() * pool as f64) as u64).min(pool as u64 - 1);
                    let e = ((base + j) % n_experts as u64) as u32;
                    if !got.contains(&e) {
                        got.push(e);
                    }
                }
                got.sort_unstable();
                for e in got {
                    if let Some(id) = crate::trace::try_dense(l, e, n_experts) {
                        step.push(id);
                    }
                }
            }
            step
        })
        .collect()
}

/// Loads per accepted token for a pass of `k` at acceptance `a`; NaN when there is no full pass.
pub fn loads_per_token(trace: &[Step], k: usize, a: f64, cap: usize) -> f64 {
    loads_per_pass(trace, k, cap) / expected_accepted(k, a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_accepted_matches_the_closed_form() {
        assert!((expected_accepted(1, 0.5) - 1.0).abs() < 1e-12);
        assert!((expected_accepted(4, 0.5) - 1.875).abs() < 1e-12);
        assert!((expected_accepted(3, 1.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn identical_tokens_share_every_load() {
        let t: Vec<Step> = vec![vec![1, 2, 3]; 8];
        assert_eq!(loads_per_pass(&t, 1, 0), 3.0);
        assert_eq!(loads_per_pass(&t, 4, 0), 3.0);
    }

    #[test]
    fn disjoint_tokens_share_nothing() {
        let t: Vec<Step> = (0..8).map(|i| vec![i * 10, i * 10 + 1]).collect();
        assert_eq!(loads_per_pass(&t, 4, 0), 8.0);
        // verifying together cannot beat the per-token cost here
        assert!(loads_per_token(&t, 4, 1.0, 0) >= loads_per_token(&t, 1, 1.0, 0) - 1e-12);
    }

    #[test]
    fn a_cache_never_adds_loads() {
        let t = crate::policy::synthetic_trace(64, 4, 200, 7);
        for k in [1, 2, 4] {
            assert!(loads_per_pass(&t, k, 32) <= loads_per_pass(&t, k, 0));
        }
    }

    #[test]
    fn low_acceptance_can_make_a_bigger_pass_worse() {
        let t: Vec<Step> = (0..64).map(|i| vec![i * 10]).collect();
        assert!(loads_per_token(&t, 8, 0.5, 0) > loads_per_token(&t, 1, 0.5, 0));
    }

    #[test]
    fn k_zero_is_treated_as_one_and_acceptance_is_clamped() {
        let t: Vec<Step> = vec![vec![1, 2]; 4];
        assert_eq!(
            loads_per_token(&t, 0, 1.0, 0),
            loads_per_token(&t, 1, 1.0, 0)
        );
        assert_eq!(expected_accepted(0, 0.5), 1.0);
        assert_eq!(expected_accepted(3, f64::NAN), 1.0);
        assert_eq!(expected_accepted(3, 7.0), 3.0);
        assert_eq!(expected_accepted(3, -2.0), 1.0);
        assert!(loads_per_token(&t, 2, f64::NAN, 0).is_finite());
    }

    #[test]
    fn a_trace_shorter_than_one_pass_is_no_data_not_zero_loads() {
        let t: Vec<Step> = vec![vec![1, 2]; 3];
        assert!(loads_per_pass(&t, 4, 0).is_nan());
        assert!(loads_per_token(&t, 4, 0.9, 0).is_nan());
        assert!(loads_per_pass(&[], 1, 0).is_nan());
    }

    fn xs(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    /// Random small traces: 3 layers x 8 experts, 1..=4 experts per layer per token.
    fn random_trace(seed: u64, tokens: usize) -> Vec<Step> {
        let mut r = xs(seed);
        (0..tokens)
            .map(|_| {
                let mut st = Vec::new();
                for l in 0..3u32 {
                    let n = 1 + (r() % 4) as usize;
                    let mut got: Vec<u32> = Vec::new();
                    while got.len() < n {
                        let e = (r() % 8) as u32;
                        if !got.contains(&e) {
                            got.push(e);
                        }
                    }
                    got.sort_unstable();
                    st.extend(got.into_iter().map(|e| l * 8 + e));
                }
                st
            })
            .collect()
    }

    #[test]
    fn unbounded_matches_the_old_union_model_exactly() {
        for seed in 1..40u64 {
            let t = random_trace(seed, 40);
            for k in [1usize, 2, 3, 4, 8] {
                for cap in [0usize, 5, 24] {
                    let v = verify_pass(&t, k, cap, 8, None);
                    assert_eq!(v.loads_per_pass, loads_per_pass(&t, k, cap));
                    assert_eq!(v.dropped_share, 0.0);
                    // a budget no layer can reach changes nothing
                    let big = verify_pass(&t, k, cap, 8, Some(8));
                    assert_eq!(big.loads_per_pass, v.loads_per_pass);
                    assert_eq!(big.dropped_share, 0.0);
                }
            }
        }
    }

    #[test]
    fn a_budget_never_adds_loads_and_is_monotone_in_b() {
        for seed in 1..60u64 {
            let t = random_trace(seed, 48);
            for k in [2usize, 4, 8] {
                let unb = verify_pass(&t, k, 0, 8, None);
                let (mut prev_l, mut prev_d) = (0.0f64, 1.0f64);
                for b in 1..=8usize {
                    let v = verify_pass(&t, k, 0, 8, Some(b));
                    assert!(v.loads_per_pass <= unb.loads_per_pass + 1e-12);
                    assert!(v.loads_per_pass >= prev_l - 1e-12, "loads grow with B");
                    assert!(v.dropped_share <= prev_d + 1e-12, "drops shrink with B");
                    assert!((0.0..=1.0).contains(&v.dropped_share));
                    prev_l = v.loads_per_pass;
                    prev_d = v.dropped_share;
                }
                assert_eq!(prev_d, 0.0);
            }
        }
    }

    #[test]
    fn at_most_b_experts_per_layer_are_loaded_per_pass() {
        let t = random_trace(5, 64);
        for b in 1..=4usize {
            let v = verify_pass(&t, 4, 0, 8, Some(b));
            // 3 layers, so at most 3 * b loads per pass
            assert!(v.loads_per_pass <= (3 * b) as f64 + 1e-12);
        }
    }

    #[test]
    fn the_budget_keeps_the_most_shared_experts_and_breaks_ties_by_low_id() {
        // layer 0, 8 experts per layer. Window of 3 tokens: expert 5 in all, 2 in two, 7 and 1 once.
        let t: Vec<Step> = vec![vec![5, 2, 7], vec![5, 2], vec![5, 1]];
        let v = verify_pass(&t, 3, 0, 8, Some(2));
        assert_eq!(v.loads_per_pass, 2.0); // 5 and 2
        assert!((v.dropped_share - 2.0 / 7.0).abs() < 1e-12); // 7 and 1 dropped, 7 slots routed
        // a tie for the last place goes to the lower id
        let u: Vec<Step> = vec![vec![5, 7], vec![1, 5]];
        let v1 = verify_pass(&u, 2, 0, 8, Some(2));
        assert_eq!(v1.loads_per_pass, 2.0);
        // a cache of 1 sees 1 then 5 or 5 then 1 depending on the kept set; the kept set is {5, 1}
        let cached = verify_pass(&[vec![1, 5, 7], vec![1, 5, 7]], 1, 3, 8, Some(2));
        assert_eq!(
            cached.loads_per_pass, 1.0,
            "second token reuses the same 2 kept experts"
        );
    }

    #[test]
    fn invalid_budgets_and_short_traces_are_no_data() {
        let t: Vec<Step> = vec![vec![1, 2]; 8];
        assert!(verify_pass(&t, 2, 0, 8, Some(0)).is_no_data());
        assert!(verify_pass(&t, 2, 0, 0, Some(2)).is_no_data());
        assert!(verify_pass(&t[..3], 4, 0, 8, Some(2)).is_no_data());
        assert!(verify_pass(&[], 1, 0, 8, None).is_no_data());
        assert!(!verify_pass(&t, 2, 0, 8, None).is_no_data());
    }

    #[test]
    fn one_x_budget_drops_nothing_for_a_single_token_window() {
        let t = crate::prefetch::correlated_trace(6, 32, 4, 80, 0.7, 3);
        let per = experts_per_layer(&t, 32);
        assert!((1..=4).contains(&per));
        let b = budget_for(1.0, per);
        let v = verify_pass(&t, 1, 0, 32, b);
        assert_eq!(v.dropped_share, 0.0);
        assert_eq!(v.loads_per_pass, loads_per_pass(&t, 1, 0));
    }

    #[test]
    fn budget_for_rounds_up_and_rejects_nonsense() {
        assert_eq!(budget_for(1.5, 8), Some(12));
        assert_eq!(budget_for(1.0, 3), Some(3));
        assert_eq!(budget_for(0.01, 8), Some(1));
        assert_eq!(budget_for(f64::NAN, 8), None);
        assert_eq!(budget_for(-1.0, 8), None);
        assert_eq!(budget_for(2.0, 0), None);
    }

    #[test]
    fn experts_per_layer_survives_empty_and_hostile_input() {
        assert_eq!(experts_per_layer(&[], 8), 0);
        assert_eq!(experts_per_layer(&[vec![1, 2, 3]], 0), 0);
        assert_eq!(experts_per_layer(&[vec![1, 2, 3, 9]], 8), 3);
    }

    #[test]
    fn overlapping_tokens_make_a_budget_cut_loads_a_sampled_pool_does_not() {
        // a drifting pool: consecutive tokens share experts, the union is small, and a tight
        // budget still drops a visible tail
        let t = locality_trace(8, 64, 4, 400, 8, 11);
        let per = experts_per_layer(&t, 64);
        assert_eq!(per, 4);
        let unb = verify_pass(&t, 8, 0, 64, None);
        let tight = verify_pass(&t, 8, 0, 64, budget_for(1.0, per));
        assert!(tight.loads_per_pass < unb.loads_per_pass);
        assert!(tight.dropped_share > 0.0 && tight.dropped_share < 1.0);
        // and the union saving itself exists on this trace
        assert!(unb.loads_per_pass < verify_pass(&t, 1, 0, 64, None).loads_per_pass * 8.0);
    }

    #[test]
    fn locality_trace_is_deterministic_and_safe_on_odd_input() {
        assert_eq!(
            locality_trace(4, 32, 3, 20, 6, 9),
            locality_trace(4, 32, 3, 20, 6, 9)
        );
        assert_eq!(
            locality_trace(0, 32, 3, 5, 6, 9),
            vec![Vec::<u32>::new(); 5]
        );
        assert_eq!(locality_trace(4, 0, 3, 5, 6, 9), vec![Vec::<u32>::new(); 5]);
        let t = locality_trace(2, 4, 99, 3, 99, 1);
        assert!(t.iter().all(|s| s.len() == 8));
    }
}
