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
//! `ROUTE_TRACE`; a trace sampled from a `.coli_usage` history overlaps only through popularity.

use crate::policy::{Cache, Lru, Step};
use std::collections::HashSet;

/// Average tokens kept per pass of `k` with per-token acceptance `a`.
/// `k` is at least 1 (the model's own token) and `a` is clamped to 0..=1 (NaN counts as 0).
pub fn expected_accepted(k: usize, a: f64) -> f64 {
    let a = if a.is_nan() { 0.0 } else { a.clamp(0.0, 1.0) };
    (0..k.max(1))
        .map(|i| a.powi(i.min(i32::MAX as usize) as i32))
        .sum()
}

/// Expert loads per window of `k` consecutive tokens, averaged over the trace (the last partial
/// window is ignored). `cap` = 0 means no cache: every needed expert is read from disk. NaN when
/// the trace is shorter than `k` tokens: there is no full pass to count, and 0 would read as
/// "free".
pub fn loads_per_pass(trace: &[Step], k: usize, cap: usize) -> f64 {
    let k = k.max(1);
    let mut cache = Lru::new(cap);
    let (mut loads, mut passes) = (0usize, 0usize);
    for w in trace.chunks_exact(k) {
        let uni: HashSet<u32> = w.iter().flatten().copied().collect();
        // a deterministic order keeps the cache replay reproducible
        let mut ids: Vec<u32> = uni.into_iter().collect();
        ids.sort_unstable();
        for id in ids {
            if cap == 0 || !cache.access(id) {
                loads += 1;
            }
        }
        passes += 1;
    }
    if passes == 0 {
        f64::NAN
    } else {
        loads as f64 / passes as f64
    }
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
}
