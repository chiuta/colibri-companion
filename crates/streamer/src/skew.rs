//! How concentrated expert routing really is, measured on real histories (`.coli_usage`).
//!
//! A history holds counts only, never the order of tokens, so everything here is about
//! popularity: what a fixed set of the most used experts would have served. Three things the
//! numbers must not be read as:
//! * coverage on the history a set was built from overstates what it serves on new prompts
//!   (see [`holdout_coverage`]);
//! * a short single-prompt session is far more concentrated than an aggregate of many prompts;
//! * it is a lower bound on a cache hit rate, since a recency cache serves part of the rest.

use crate::trace::Usage;

/// Fractions of all experts at which curves are reported.
pub const FRACTIONS: [f64; 5] = [0.01, 0.05, 0.10, 0.25, 0.50];

/// Experts that fit `frac` of the whole, at least one (0 when the history is empty).
fn budget(total: usize, frac: f64) -> usize {
    if total == 0 || !frac.is_finite() || frac <= 0.0 {
        return 0;
    }
    (((total as f64) * frac.min(1.0)).floor() as usize).clamp(1, total)
}

fn sorted_desc(counts: &[u64]) -> Vec<u64> {
    let mut s = counts.to_vec();
    s.sort_unstable_by(|a, b| b.cmp(a));
    s
}

fn sum_f(counts: &[u64]) -> f64 {
    counts.iter().map(|c| *c as f64).sum()
}

/// Share of all uses served by the `frac` most used experts of the whole history.
pub fn coverage_at(u: &Usage, frac: f64) -> f64 {
    let t = sum_f(&u.counts);
    if t <= 0.0 {
        return 0.0;
    }
    let s = sorted_desc(&u.counts);
    let n = budget(s.len(), frac);
    sum_f(&s[..n]) / t
}

/// The same, but every layer keeps the same number of its own experts (an equal per-layer
/// budget, rounded down, so it never uses more experts in total than the global ranking). A
/// global ranking is the optimum for a plain coverage target, so this is never higher than
/// [`coverage_at`]; the gap says what a per-layer split would cost.
pub fn equal_per_layer_coverage(u: &Usage, frac: f64) -> f64 {
    let t = sum_f(&u.counts);
    let ne = u.n_experts as usize;
    if t <= 0.0 || ne == 0 {
        return 0.0;
    }
    // floor per layer, never rounded up to one: the split must not use more experts in total
    // than the global ranking does, or the two would not be comparable
    let per_layer = if frac.is_finite() && frac > 0.0 {
        (((ne as f64) * frac.min(1.0)).floor() as usize).min(ne)
    } else {
        0
    };
    let mut served = 0.0;
    for layer in u.counts.chunks(ne) {
        let s = sorted_desc(layer);
        served += sum_f(&s[..per_layer]);
    }
    served / t
}

/// Rank experts on `train`, keep the top `frac`, and score that set on `test`. `None` when the
/// two histories do not describe the same model shape or `test` is empty.
pub fn holdout_coverage(train: &Usage, test: &Usage, frac: f64) -> Option<f64> {
    if train.n_layers != test.n_layers
        || train.n_experts != test.n_experts
        || train.counts.len() != test.counts.len()
    {
        return None;
    }
    let t = sum_f(&test.counts);
    if t <= 0.0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..train.counts.len()).collect();
    idx.sort_by(|a, b| train.counts[*b].cmp(&train.counts[*a]).then(a.cmp(b)));
    let n = budget(idx.len(), frac);
    let served: f64 = idx[..n].iter().map(|i| test.counts[*i] as f64).sum();
    Some(served / t)
}

/// Gini coefficient of the counts (0 = every expert equally used, towards 1 = one expert).
pub fn gini(counts: &[u64]) -> f64 {
    let n = counts.len();
    let t = sum_f(counts);
    if n == 0 || t <= 0.0 {
        return 0.0;
    }
    let mut asc = counts.to_vec();
    asc.sort_unstable();
    let weighted: f64 = asc
        .iter()
        .enumerate()
        .map(|(i, c)| (i as f64 + 1.0) * *c as f64)
        .sum();
    let nf = n as f64;
    (2.0 * weighted / (nf * t) - (nf + 1.0) / nf).clamp(0.0, 1.0)
}

/// `2^H` for the Shannon entropy H (bits) of the usage distribution: how many equally used
/// experts would give the same spread. Between 1 and the number of experts used.
pub fn effective_experts(counts: &[u64]) -> f64 {
    let t = sum_f(counts);
    if t <= 0.0 {
        return 0.0;
    }
    let h: f64 = counts
        .iter()
        .filter(|c| **c > 0)
        .map(|c| {
            let p = *c as f64 / t;
            -p * p.log2()
        })
        .sum();
    h.exp2()
}

/// Smallest share of all experts whose combined uses reach `target` (0..=1). `None` when the
/// history is empty or the target is not a share.
pub fn experts_for_coverage(u: &Usage, target: f64) -> Option<f64> {
    let t = sum_f(&u.counts);
    if t <= 0.0 || !(0.0..=1.0).contains(&target) {
        return None;
    }
    let s = sorted_desc(&u.counts);
    let mut acc = 0.0;
    for (i, c) in s.iter().enumerate() {
        acc += *c as f64;
        if acc / t >= target - 1e-12 {
            return Some((i + 1) as f64 / s.len() as f64);
        }
    }
    Some(1.0)
}

/// Spread over layers of the share of a layer's own uses served by its top 10% experts.
/// Returns (min, median, max); `None` when no layer has uses.
pub fn per_layer_top10(u: &Usage) -> Option<(f64, f64, f64)> {
    let ne = u.n_experts as usize;
    if ne == 0 {
        return None;
    }
    let mut v: Vec<f64> = u
        .counts
        .chunks(ne)
        .filter(|l| sum_f(l) > 0.0)
        .map(|l| {
            let s = sorted_desc(l);
            sum_f(&s[..budget(ne, 0.10)]) / sum_f(l)
        })
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some((v[0], v[v.len() / 2], v[v.len() - 1]))
}

/// One history, summarised.
#[derive(Debug, Clone, PartialEq)]
pub struct Skew {
    pub selections: u64,
    pub experts_total: usize,
    pub experts_never_used: usize,
    pub gini: f64,
    pub effective_experts: f64,
    /// `coverage[i]` is the share served by the top `FRACTIONS[i]` of experts.
    pub coverage: [f64; 5],
    /// Share of experts needed for 50/80/90/95% of uses.
    pub need: [f64; 4],
}

pub fn analyse(u: &Usage) -> Skew {
    let mut coverage = [0.0; 5];
    for (c, f) in coverage.iter_mut().zip(FRACTIONS) {
        *c = coverage_at(u, f);
    }
    let mut need = [0.0; 4];
    for (n, t) in need.iter_mut().zip([0.5, 0.8, 0.9, 0.95]) {
        *n = experts_for_coverage(u, t).unwrap_or(0.0);
    }
    Skew {
        selections: u.counts.iter().sum(),
        experts_total: u.counts.len(),
        experts_never_used: u.counts.iter().filter(|c| **c == 0).count(),
        gini: gini(&u.counts),
        effective_experts: effective_experts(&u.counts),
        coverage,
        need,
    }
}

/// One measured history, kept as constants so other crates can use the spread without the files.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkewSample {
    pub name: &'static str,
    /// An aggregate of many prompts (a shipped profile) or one short session.
    pub aggregate: bool,
    /// Coverage of the top `FRACTIONS` of experts, measured on the history itself.
    pub own: [f64; 5],
    /// The same ranking scored on the OTHER histories of the same model, when there are any:
    /// (min, max) over them. `None` when this model has no second history.
    pub held_out: Option<([f64; 5], [f64; 5])>,
}

/// Measured with `examples/profiles.rs` on the histories shipped with colibri
/// (`c/profiles/*.coli_usage`, `docs/experiments/*/history.usage.gz`, taken 2026-11-16).
/// Four of the five session files are byte-identical, so they count once.
pub const SKEW_SAMPLES: [SkewSample; 4] = [
    // GENERATED-SAMPLES-BEGIN
    SkewSample {
        name: "qwen36-35b (profile, 40x256)",
        aggregate: true,
        own: [0.0602, 0.2117, 0.3419, 0.5933, 0.8349],
        held_out: None,
    },
    SkewSample {
        name: "qwen38-flash-next (profile, 48x512)",
        aggregate: true,
        own: [0.0828, 0.2643, 0.4113, 0.6867, 0.9073],
        held_out: Some((
            [0.0166, 0.0576, 0.1123, 0.2822, 0.5481],
            [0.0471, 0.1756, 0.2963, 0.5628, 0.824],
        )),
    },
    SkewSample {
        name: "qwen38 session A (48x512)",
        aggregate: false,
        own: [0.2735, 0.5873, 0.747, 0.9368, 1.0],
        held_out: Some((
            [0.0053, 0.0556, 0.1222, 0.306, 0.552],
            [0.0088, 0.0732, 0.1723, 0.4014, 0.641],
        )),
    },
    SkewSample {
        name: "qwen38 session B, vulkan+MTP (48x512)",
        aggregate: false,
        own: [0.1988, 0.4618, 0.6303, 0.8877, 0.9973],
        held_out: Some((
            [0.0166, 0.062, 0.125, 0.3141, 0.6278],
            [0.0244, 0.1277, 0.2421, 0.4942, 0.7339],
        )),
    },
    // GENERATED-SAMPLES-END
];

fn lerp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    if x1 <= x0 {
        return y1;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Piecewise-linear coverage of one curve at `frac`, through (0,0), the five measured points
/// and (1,1).
fn curve_at(own: &[f64; 5], frac: f64) -> f64 {
    let f = if frac.is_finite() {
        frac.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut px = 0.0;
    let mut py = 0.0;
    for (x, y) in FRACTIONS.iter().zip(own.iter()).chain([(&1.0, &1.0)]) {
        if f <= *x {
            return lerp(px, py, *x, *y, f);
        }
        px = *x;
        py = *y;
    }
    1.0
}

/// A range, never one number: (min, median, max) of the share of expert uses a fixed pinned set
/// of the hottest experts serves when `frac` of all experts fit, over the measured histories.
/// `held_out = false`: each history scored on itself (an upper bound for new prompts).
/// `held_out = true`: ranking on one history and scoring on another (min of the worst cases,
/// max of the best cases), the honest choice for prompts the history did not see. Only the
/// three Qwen3.8 histories can be held out against each other, so this half says nothing about
/// other models.
pub fn pinned_hit_range(frac: f64, held_out: bool) -> (f64, f64, f64) {
    if held_out {
        // only histories that were actually scored on another history: mixing in a figure
        // measured on its own data would be optimistic by a factor of two to four.
        // min = lowest worst case, max = highest best case, middle = median of the midpoints.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut mids: Vec<f64> = Vec::new();
        for (l, h) in SKEW_SAMPLES.iter().filter_map(|s| s.held_out.as_ref()) {
            let (l, h) = (curve_at(l, frac), curve_at(h, frac));
            lo = lo.min(l);
            hi = hi.max(h);
            mids.push((l + h) / 2.0);
        }
        mids.sort_by(f64::total_cmp);
        return (lo, mids[mids.len() / 2], hi);
    }
    let mut v: Vec<f64> = SKEW_SAMPLES
        .iter()
        .map(|s| curve_at(&s.own, frac))
        .collect();
    v.sort_by(f64::total_cmp);
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::parse_usage;

    fn flat(n_layers: u32, n_experts: u32) -> Usage {
        Usage {
            n_layers,
            n_experts,
            counts: vec![10; (n_layers * n_experts) as usize],
        }
    }

    #[test]
    fn a_flat_history_has_no_skew() {
        let u = flat(2, 50);
        assert!((coverage_at(&u, 0.10) - 0.10).abs() < 1e-9);
        assert!(gini(&u.counts) < 1e-9);
        assert!((effective_experts(&u.counts) - 100.0).abs() < 1e-6);
        assert!((equal_per_layer_coverage(&u, 0.10) - 0.10).abs() < 1e-9);
    }

    #[test]
    fn one_hot_expert_is_total_skew() {
        let mut u = flat(1, 100);
        u.counts = vec![0; 100];
        u.counts[7] = 500;
        assert!((coverage_at(&u, 0.01) - 1.0).abs() < 1e-9);
        assert!(gini(&u.counts) > 0.98);
        assert!((effective_experts(&u.counts) - 1.0).abs() < 1e-9);
        assert_eq!(experts_for_coverage(&u, 0.95), Some(0.01));
    }

    #[test]
    fn a_global_ranking_is_never_worse_than_an_equal_split() {
        // layer 0 is hot, layer 1 is cold: the global set puts its budget where the uses are
        let u = parse_usage("-1 2 4\n0 0 90\n0 1 5\n1 0 3\n1 1 2\n").unwrap();
        let g = coverage_at(&u, 0.25);
        let p = equal_per_layer_coverage(&u, 0.25);
        assert!(g >= p - 1e-12);
        assert!(g > p);
    }

    #[test]
    fn holdout_needs_the_same_shape_and_scores_the_train_ranking() {
        let a = parse_usage("-1 1 4\n0 0 100\n0 1 1\n0 2 1\n0 3 1\n").unwrap();
        let b = parse_usage("-1 1 4\n0 0 1\n0 1 1\n0 2 1\n0 3 97\n").unwrap();
        // A's hottest expert is cold in B
        let h = holdout_coverage(&a, &b, 0.25).unwrap();
        assert!((h - 0.01).abs() < 1e-9);
        assert!((holdout_coverage(&a, &a, 0.25).unwrap() - 100.0 / 103.0).abs() < 1e-9);
        let other = parse_usage("-1 1 8\n0 0 1\n").unwrap();
        assert_eq!(holdout_coverage(&a, &other, 0.25), None);
        let empty = parse_usage("-1 1 4\n").unwrap();
        assert_eq!(holdout_coverage(&a, &empty, 0.25), None);
    }

    #[test]
    fn hostile_inputs_do_not_panic() {
        let empty = Usage::default();
        assert_eq!(coverage_at(&empty, 0.5), 0.0);
        assert_eq!(equal_per_layer_coverage(&empty, 0.5), 0.0);
        assert_eq!(experts_for_coverage(&empty, 0.5), None);
        assert_eq!(per_layer_top10(&empty), None);
        assert_eq!(analyse(&empty).experts_total, 0);
        let u = flat(1, 10);
        for f in [f64::NAN, f64::INFINITY, -1.0, 0.0, 2.0] {
            let c = coverage_at(&u, f);
            assert!((0.0..=1.0).contains(&c), "{f} -> {c}");
        }
        assert_eq!(experts_for_coverage(&u, 1.5), None);
        assert_eq!(experts_for_coverage(&u, f64::NAN), None);
        // counts near u64::MAX stay finite (summed in f64)
        let big = Usage {
            n_layers: 1,
            n_experts: 2,
            counts: vec![u64::MAX, u64::MAX],
        };
        assert!(coverage_at(&big, 0.5) > 0.49);
        assert!(gini(&big.counts).is_finite());
    }

    #[test]
    fn coverage_never_decreases_with_the_budget_and_stays_in_range() {
        let mut s = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..50 {
            let ne = 1 + (next() % 40) as u32;
            let nl = 1 + (next() % 5) as u32;
            let counts: Vec<u64> = (0..nl * ne).map(|_| next() % 1000).collect();
            let u = Usage {
                n_layers: nl,
                n_experts: ne,
                counts,
            };
            let mut prev = 0.0;
            for k in 0..=20 {
                let f = k as f64 / 20.0;
                let c = coverage_at(&u, f);
                assert!(c >= prev - 1e-12 && (0.0..=1.0 + 1e-12).contains(&c));
                assert!(equal_per_layer_coverage(&u, f) <= c + 1e-9);
                prev = c;
            }
        }
    }

    #[test]
    fn the_measured_samples_are_sane_curves() {
        for s in &SKEW_SAMPLES {
            assert!(
                s.own.windows(2).all(|w| w[0] <= w[1]),
                "{}: not monotone",
                s.name
            );
            assert!(s.own.iter().all(|c| (0.0..=1.0).contains(c)), "{}", s.name);
            // a pinned set of x% of the experts cannot beat covering everything,
            // and must beat random choice (coverage >= fraction)
            for (c, f) in s.own.iter().zip(FRACTIONS) {
                assert!(*c >= f - 1e-9, "{}: below the flat baseline", s.name);
            }
            if let Some((lo, hi)) = &s.held_out {
                for ((l, h), o) in lo.iter().zip(hi).zip(&s.own) {
                    assert!(*l <= *h + 1e-9, "{}", s.name);
                    assert!(*l <= *o + 1e-9, "{}: held-out beats own", s.name);
                }
            }
        }
    }

    #[test]
    fn the_range_is_ordered_bounded_and_monotone_in_the_pinned_share() {
        let mut prev = (0.0, 0.0, 0.0);
        for k in 0..=40 {
            let f = k as f64 / 40.0;
            for ho in [false, true] {
                let (lo, mid, hi) = pinned_hit_range(f, ho);
                assert!(lo <= mid + 1e-12 && mid <= hi + 1e-12, "{f}");
                assert!((0.0..=1.0 + 1e-12).contains(&hi));
            }
            let r = pinned_hit_range(f, false);
            assert!(r.0 >= prev.0 - 1e-12 && r.2 >= prev.2 - 1e-12);
            prev = r;
        }
        assert_eq!(pinned_hit_range(0.0, false), (0.0, 0.0, 0.0));
        assert_eq!(pinned_hit_range(1.0, false), (1.0, 1.0, 1.0));
        let (a, b, c) = pinned_hit_range(f64::NAN, false);
        assert_eq!((a, b, c), (0.0, 0.0, 0.0));
    }

    #[test]
    fn curve_interpolation_hits_the_measured_points() {
        let own = [0.1, 0.3, 0.5, 0.8, 0.95];
        for (x, y) in FRACTIONS.iter().zip(own) {
            assert!((curve_at(&own, *x) - y).abs() < 1e-12);
        }
        assert!((curve_at(&own, 0.005) - 0.05).abs() < 1e-12);
        assert!((curve_at(&own, 0.75) - 0.975).abs() < 1e-12);
    }
}
