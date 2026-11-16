//! A next-layer expert predictor. While layer L computes, the experts of layer L+1 could already
//! be read from disk if they were known. This predictor guesses them from the experts layer L
//! used, with plain co-occurrence counts (expert a at layer L, expert b at layer L+1, seen together
//! in the same token), falling back to per-layer popularity.
//!
//! This is a measurement, not an engine: it says what share of the loads a prefetcher with a given
//! budget could have issued in time. It only means something on a real trace; a trace sampled
//! from a popularity history has no correlation between layers by construction.

use crate::policy::Step;
use crate::trace::{split_dense, try_dense};
use std::collections::HashMap;

/// Experts of one token grouped by layer, in the order given: `(layer, experts)`. A new group
/// starts whenever the layer changes. Empty when `n_experts` is 0 (there is no layer structure).
pub fn by_layer(step: &Step, n_experts: u32) -> Vec<(u32, Vec<u32>)> {
    let mut out: Vec<(u32, Vec<u32>)> = Vec::new();
    for &id in step {
        let Some((l, e)) = split_dense(id, n_experts) else {
            return Vec::new();
        };
        match out.last_mut() {
            Some((pl, v)) if *pl == l => v.push(e),
            _ => out.push((l, vec![e])),
        }
    }
    out
}

/// Above this many experts per layer the predictor scores with a hash map instead of a flat array.
const FLAT_SCORE_LIMIT: u32 = 1 << 16;

#[derive(Default)]
pub struct Predictor {
    n_experts: u32,
    /// `next[a]`: how often each expert of layer L+1 followed dense expert `a` of layer L, as
    /// `(expert, count)` sorted by expert.
    next: HashMap<u32, Vec<(u32, u32)>>,
    /// Per layer, its experts by use (most used first, lowest id first among equals): the fallback
    /// when co-occurrence has nothing to say, and the popularity-only baseline.
    pop: HashMap<u32, Vec<u32>>,
}

impl Predictor {
    pub fn train(steps: &[Step], n_experts: u32) -> Self {
        let mut pair: HashMap<u64, u32> = HashMap::new(); // (dense a << 32 | y) -> count
        let mut used: HashMap<u64, u32> = HashMap::new(); // (layer << 32 | e) -> count
        for s in steps {
            let layers = by_layer(s, n_experts);
            for (l, es) in &layers {
                for &e in es {
                    let c = used.entry((*l as u64) << 32 | e as u64).or_default();
                    *c = c.saturating_add(1);
                }
            }
            for w in layers.windows(2) {
                let ((l0, a), (l1, b)) = (&w[0], &w[1]);
                if l0.checked_add(1) != Some(*l1) {
                    continue;
                }
                for &x in a {
                    let Some(key) = try_dense(*l0, x, n_experts) else {
                        continue;
                    };
                    for &y in b {
                        let c = pair.entry((key as u64) << 32 | y as u64).or_default();
                        *c = c.saturating_add(1);
                    }
                }
            }
        }
        let mut next: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
        for (k, c) in pair {
            next.entry((k >> 32) as u32)
                .or_default()
                .push((k as u32, c));
        }
        for row in next.values_mut() {
            row.sort_unstable();
        }
        let mut by_pop: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
        for (k, c) in used {
            by_pop
                .entry((k >> 32) as u32)
                .or_default()
                .push((k as u32, c));
        }
        let pop = by_pop
            .into_iter()
            .map(|(l, mut v)| {
                v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                (l, v.into_iter().map(|x| x.0).collect())
            })
            .collect();
        Self {
            n_experts,
            next,
            pop,
        }
    }

    /// Up to `budget` experts (plain indices, not dense ids) guessed for layer `layer + 1`, given
    /// the experts the token used at `layer`. Co-occurrence first, then popularity. The result for
    /// a smaller budget is always a prefix of the result for a larger one.
    pub fn predict(&self, layer: u32, used: &[u32], budget: usize) -> Vec<u32> {
        let Some(next_layer) = layer.checked_add(1) else {
            return Vec::new();
        };
        if budget == 0 {
            return Vec::new();
        }
        let rows: Vec<&Vec<(u32, u32)>> = used
            .iter()
            .filter_map(|&x| try_dense(layer, x, self.n_experts).and_then(|k| self.next.get(&k)))
            .collect();
        let mut v: Vec<(u32, u64)> = if self.n_experts <= FLAT_SCORE_LIMIT {
            let mut score = vec![0u64; self.n_experts as usize];
            let mut touched: Vec<u32> = Vec::new();
            for row in &rows {
                for &(y, c) in row.iter() {
                    let Some(slot) = score.get_mut(y as usize) else {
                        continue;
                    };
                    if *slot == 0 {
                        touched.push(y);
                    }
                    *slot += c as u64;
                }
            }
            touched
                .into_iter()
                .map(|y| (y, score[y as usize]))
                .collect()
        } else {
            let mut score: HashMap<u32, u64> = HashMap::new();
            for row in &rows {
                for &(y, c) in row.iter() {
                    *score.entry(y).or_default() += c as u64;
                }
            }
            score.into_iter().collect()
        };
        let rank = |a: &(u32, u64), b: &(u32, u64)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
        if v.len() > budget {
            v.select_nth_unstable_by(budget - 1, rank); // the best `budget`, unordered
            v.truncate(budget);
        }
        v.sort_unstable_by(rank);
        let mut out: Vec<u32> = v.into_iter().map(|x| x.0).collect();
        if out.len() < budget {
            let more = self.popular(next_layer, budget, &out);
            out.extend(more);
        }
        out
    }

    /// Popularity only: the baseline that needs no knowledge of the previous layer. At most
    /// `budget - skip.len()` experts, none of them in `skip`.
    pub fn popular(&self, layer: u32, budget: usize, skip: &[u32]) -> Vec<u32> {
        let Some(order) = self.pop.get(&layer) else {
            return Vec::new();
        };
        order
            .iter()
            .copied()
            .filter(|e| !skip.contains(e))
            .take(budget.saturating_sub(skip.len()))
            .collect()
    }
}

/// Share of the layer-by-layer loads the predictor would have had ready in time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub loads: u64,
    pub predicted: u64,
    pub popularity_only: u64,
}

impl Score {
    pub fn share(&self) -> f64 {
        ratio(self.predicted, self.loads)
    }
    pub fn baseline(&self) -> f64 {
        ratio(self.popularity_only, self.loads)
    }
}

fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 { 0.0 } else { a as f64 / b as f64 }
}

/// Trains on the first `train_share` of the tokens and scores on the rest. For every layer after
/// the first, the prefetch budget is `mult` times the number of experts that layer really used.
pub fn evaluate(trace: &[Step], n_experts: u32, train_share: f64, mult: usize) -> Score {
    evaluate_budgets(trace, n_experts, train_share, &[mult])[0]
}

/// `evaluate` for several budgets at once: the predictor is trained once and each prediction is
/// made at the largest budget, since a smaller budget's answer is a prefix of it. One `Score` per
/// entry of `mults`, in order.
pub fn evaluate_budgets(
    trace: &[Step],
    n_experts: u32,
    train_share: f64,
    mults: &[usize],
) -> Vec<Score> {
    let blank = Score {
        loads: 0,
        predicted: 0,
        popularity_only: 0,
    };
    let mut out = vec![blank; mults.len()];
    let cut = ((trace.len() as f64 * train_share) as usize).min(trace.len());
    let (train, test) = trace.split_at(cut);
    let p = Predictor::train(train, n_experts);
    let max_mult = mults.iter().copied().max().unwrap_or(0);
    for s in test {
        let layers = by_layer(s, n_experts);
        for w in layers.windows(2) {
            let ((l0, a), (l1, b)) = (&w[0], &w[1]);
            if l0.checked_add(1) != Some(*l1) {
                continue;
            }
            // loads are distinct experts: a repeated id in one layer is one read
            let mut b = b.clone();
            b.sort_unstable();
            b.dedup();
            let guess = p.predict(*l0, a, b.len().saturating_mul(max_mult));
            let base = p.popular(*l1, b.len().saturating_mul(max_mult), &[]);
            for (sc, &m) in out.iter_mut().zip(mults) {
                let budget = b.len().saturating_mul(m);
                let g = &guess[..guess.len().min(budget)];
                let pb = &base[..base.len().min(budget)];
                sc.loads += b.len() as u64;
                sc.predicted += b.iter().filter(|e| g.contains(e)).count() as u64;
                sc.popularity_only += b.iter().filter(|e| pb.contains(e)).count() as u64;
            }
        }
    }
    out
}

/// Largest number of layers `correlated_trace` will generate (deeper requests are cut here).
pub const MAX_LAYERS: u32 = 4096;
/// Largest number of experts per layer and token `correlated_trace` will draw.
pub const MAX_K: usize = 1 << 12;

/// A trace whose layers ARE correlated, to check the code: the experts of layer L+1 are a fixed
/// function of the experts of layer L, with probability `fidelity`; otherwise they are random.
/// `n_layers` is limited to `MAX_LAYERS` and `k` to `n_experts` and `MAX_K`; a layer whose dense ids no
/// longer fit a u32 ends the token.
pub fn correlated_trace(
    n_layers: u32,
    n_experts: u32,
    k: usize,
    tokens: usize,
    fidelity: f64,
    seed: u64,
) -> Vec<Step> {
    if n_experts == 0 {
        return vec![Vec::new(); tokens];
    }
    let k = k.min(n_experts as usize).min(MAX_K);
    let mut s = seed | 1;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let ne = n_experts as u64;
    (0..tokens)
        .map(|_| {
            let mut cur: Vec<u32> = (0..k).map(|_| (rnd() * n_experts as f64) as u32).collect();
            let mut step = Vec::new();
            for l in 0..n_layers.min(MAX_LAYERS) {
                cur.sort_unstable();
                cur.dedup();
                let ids: Vec<u32> = cur
                    .iter()
                    .filter_map(|e| try_dense(l, *e, n_experts))
                    .collect();
                if ids.len() != cur.len() {
                    break;
                }
                step.extend(ids);
                cur = cur
                    .iter()
                    .map(|&e| {
                        if rnd() < fidelity {
                            ((e as u64 * 7 + 3) % ne) as u32
                        } else {
                            ((rnd() * n_experts as f64) as u32).min(n_experts - 1)
                        }
                    })
                    .collect();
            }
            step
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{split_dense, try_dense};

    #[test]
    fn groups_a_token_by_layer() {
        let g = by_layer(&vec![1, 2, 11, 25], 10);
        assert_eq!(g, vec![(0, vec![1, 2]), (1, vec![1]), (2, vec![5])]);
    }

    #[test]
    fn a_fixed_mapping_is_learned() {
        let t = correlated_trace(6, 64, 4, 600, 1.0, 5);
        let s = evaluate(&t, 64, 0.5, 1);
        assert!(s.share() > 0.8, "{}", s.share());
        assert!(s.share() > s.baseline() + 0.3);
    }

    #[test]
    fn uncorrelated_layers_give_no_edge_over_popularity() {
        let t = correlated_trace(6, 64, 4, 600, 0.0, 5);
        let s = evaluate(&t, 64, 0.5, 2);
        assert!((s.share() - s.baseline()).abs() < 0.1);
    }

    #[test]
    fn a_bigger_budget_never_predicts_less() {
        let t = correlated_trace(6, 64, 4, 400, 0.6, 9);
        let a = evaluate(&t, 64, 0.5, 1).share();
        let b = evaluate(&t, 64, 0.5, 4).share();
        assert!(b >= a);
    }

    #[test]
    fn nothing_to_score_is_zero_not_a_panic() {
        assert_eq!(evaluate(&[], 64, 0.5, 1).share(), 0.0);
    }

    #[test]
    fn zero_experts_per_layer_is_empty_not_a_division_by_zero() {
        assert!(by_layer(&vec![1, 2, 3], 0).is_empty());
        assert_eq!(evaluate(&[vec![1, 2, 3]], 0, 0.5, 1).share(), 0.0);
        assert!(
            correlated_trace(3, 0, 2, 4, 0.5, 1)
                .iter()
                .all(|s| s.is_empty())
        );
    }

    #[test]
    fn the_top_layer_and_huge_expert_counts_do_not_overflow() {
        let p = Predictor::train(&[vec![u32::MAX]], 1); // layer u32::MAX with one expert per layer
        assert!(p.predict(u32::MAX, &[0], 4).is_empty());
        let big = Predictor::train(&[vec![0, 4_000_000_000]], 3_000_000_000);
        assert!(big.predict(0, &[0], 2).len() <= 2);
        let t = correlated_trace(2, 1_000_000_000, 3, 2, 1.0, 1); // e * 7 would overflow a u32
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn a_zero_multiplier_predicts_nothing() {
        let t = correlated_trace(6, 64, 4, 100, 1.0, 5);
        assert_eq!(evaluate(&t, 64, 0.5, 0).share(), 0.0);
    }

    // --- the optimised predictor against the straightforward original ---

    /// The first version: nested hash maps, a full sort of every candidate, a new training run
    /// per budget. Kept as the reference.
    struct RefPredictor {
        n_experts: u32,
        next: HashMap<u32, HashMap<u32, u32>>,
        pop: HashMap<u32, HashMap<u32, u32>>,
    }

    impl RefPredictor {
        fn train(steps: &[Step], n_experts: u32) -> Self {
            let mut p = Self {
                n_experts,
                next: HashMap::new(),
                pop: HashMap::new(),
            };
            for s in steps {
                let layers = by_layer(s, n_experts);
                for (l, es) in &layers {
                    for &e in es {
                        *p.pop.entry(*l).or_default().entry(e).or_default() += 1;
                    }
                }
                for w in layers.windows(2) {
                    let ((l0, a), (l1, b)) = (&w[0], &w[1]);
                    if l0.checked_add(1) != Some(*l1) {
                        continue;
                    }
                    for &x in a {
                        let row = p.next.entry(l0 * n_experts + x).or_default();
                        for &y in b {
                            *row.entry(y).or_default() += 1;
                        }
                    }
                }
            }
            p
        }
        fn predict(&self, layer: u32, used: &[u32], budget: usize) -> Vec<u32> {
            let mut score: HashMap<u32, u64> = HashMap::new();
            for &x in used {
                if let Some(row) = self.next.get(&(layer * self.n_experts + x)) {
                    for (&y, &c) in row {
                        *score.entry(y).or_default() += c as u64;
                    }
                }
            }
            let mut v: Vec<(u32, u64)> = score.into_iter().collect();
            v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut out: Vec<u32> = v.into_iter().take(budget).map(|x| x.0).collect();
            if out.len() < budget {
                out.extend(self.popular(layer + 1, budget, &out));
            }
            out
        }
        fn popular(&self, layer: u32, budget: usize, skip: &[u32]) -> Vec<u32> {
            let Some(m) = self.pop.get(&layer) else {
                return Vec::new();
            };
            let mut v: Vec<(u32, u32)> = m.iter().map(|(&e, &c)| (e, c)).collect();
            v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            v.into_iter()
                .map(|x| x.0)
                .filter(|e| !skip.contains(e))
                .take(budget.saturating_sub(skip.len()))
                .collect()
        }
    }

    fn ref_evaluate(trace: &[Step], n: u32, share: f64, mult: usize) -> Score {
        let cut = ((trace.len() as f64 * share) as usize).min(trace.len());
        let (train, test) = trace.split_at(cut);
        let p = RefPredictor::train(train, n);
        let mut sc = Score {
            loads: 0,
            predicted: 0,
            popularity_only: 0,
        };
        for s in test {
            let layers = by_layer(s, n);
            for w in layers.windows(2) {
                let ((l0, a), (l1, b)) = (&w[0], &w[1]);
                if l0 + 1 != *l1 {
                    continue;
                }
                let budget = b.len() * mult;
                let guess = p.predict(*l0, a, budget);
                let base = p.popular(*l1, budget, &[]);
                sc.loads += b.len() as u64;
                sc.predicted += b.iter().filter(|e| guess.contains(e)).count() as u64;
                sc.popularity_only += b.iter().filter(|e| base.contains(e)).count() as u64;
            }
        }
        sc
    }

    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    /// Random tokens with distinct experts inside each layer, partly correlated across layers.
    fn random_tokens(r: &mut impl FnMut() -> u64, n: u32, layers: u32, tokens: usize) -> Vec<Step> {
        (0..tokens)
            .map(|_| {
                let mut step = Vec::new();
                let mut prev: Vec<u32> = Vec::new();
                for l in 0..layers {
                    if r() % 9 == 0 && l > 0 {
                        continue; // a layer without routing rows
                    }
                    let want = 1 + (r() % 4) as usize;
                    let mut cur: Vec<u32> = Vec::new();
                    while cur.len() < want.min(n as usize) {
                        let e = if !prev.is_empty() && r() % 3 != 0 {
                            (prev[(r() % prev.len() as u64) as usize] * 5 + 1) % n
                        } else {
                            (r() % n as u64) as u32
                        };
                        if !cur.contains(&e) {
                            cur.push(e);
                        }
                    }
                    cur.sort_unstable();
                    step.extend(cur.iter().map(|e| l * n + e));
                    prev = cur;
                }
                step
            })
            .collect()
    }

    #[test]
    fn the_fast_predictor_scores_exactly_like_the_original_on_random_traces() {
        let mut r = xorshift(4242);
        let mut compared = 0u64;
        for _ in 0..150 {
            let n = 2 + (r() % 14) as u32;
            let layers = 2 + (r() % 5) as u32;
            let len = 20 + (r() % 80) as usize;
            let trace = random_tokens(&mut r, n, layers, len);
            let share = [0.3, 0.5, 0.8][(r() % 3) as usize];
            for mult in [0usize, 1, 2, 4] {
                let want = ref_evaluate(&trace, n, share, mult);
                let got = evaluate(&trace, n, share, mult);
                assert_eq!(got, want, "n {n} layers {layers} mult {mult} share {share}");
                compared += want.loads;
            }
        }
        assert!(compared > 5000, "{compared}");
    }

    #[test]
    fn several_budgets_in_one_pass_equal_one_pass_each() {
        let mut r = xorshift(99);
        for _ in 0..60 {
            let n = 3 + (r() % 10) as u32;
            let trace = random_tokens(&mut r, n, 5, 80);
            let all = evaluate_budgets(&trace, n, 0.5, &[4, 1, 2, 0, 3]);
            for (got, m) in all.iter().zip([4usize, 1, 2, 0, 3]) {
                assert_eq!(*got, evaluate(&trace, n, 0.5, m), "mult {m}");
            }
        }
        assert!(evaluate_budgets(&[], 4, 0.5, &[]).is_empty());
    }

    #[test]
    fn a_smaller_budget_is_always_a_prefix_of_a_larger_one() {
        let mut r = xorshift(7);
        let n = 12u32;
        let trace = random_tokens(&mut r, n, 5, 200);
        let p = Predictor::train(&trace, n);
        for _ in 0..300 {
            let used: Vec<u32> = (0..(1 + r() % 4))
                .map(|_| (r() % n as u64) as u32)
                .collect();
            let layer = (r() % 4) as u32;
            let big = p.predict(layer, &used, 9);
            for b in 0..=9 {
                assert_eq!(
                    p.predict(layer, &used, b),
                    big[..b.min(big.len())],
                    "budget {b}"
                );
            }
            let mut d = big.clone();
            d.sort_unstable();
            d.dedup();
            assert_eq!(d.len(), big.len(), "no expert predicted twice");
            assert!(big.iter().all(|e| *e < n));
        }
    }

    #[test]
    fn the_hash_map_path_for_huge_layers_agrees_with_the_flat_array_path() {
        // the same trace with the same ids, once under the flat limit and once above it
        let mut r = xorshift(5);
        let small = 40u32;
        let trace = random_tokens(&mut r, small, 4, 120);
        let a = Predictor::train(&trace, small);
        // same structure, ids spread over a layer width above FLAT_SCORE_LIMIT
        let wide = FLAT_SCORE_LIMIT + 1;
        let wide_trace: Vec<Step> = trace
            .iter()
            .map(|s| {
                s.iter()
                    .map(|id| (id / small) * wide + id % small)
                    .collect()
            })
            .collect();
        let b = Predictor::train(&wide_trace, wide);
        for l in 0..3 {
            for x in 0..small {
                assert_eq!(
                    a.predict(l, &[x], 6),
                    b.predict(l, &[x], 6),
                    "layer {l} expert {x}"
                );
            }
        }
    }

    #[test]
    fn by_layer_and_try_dense_round_trip_and_refuse_aliases() {
        let mut r = xorshift(1);
        for _ in 0..2000 {
            let n = 1 + (r() % 300) as u32;
            let (l, e) = ((r() % 1000) as u32, (r() % n as u64) as u32);
            let id = try_dense(l, e, n).unwrap();
            assert_eq!(split_dense(id, n), Some((l, e)));
            assert_eq!(by_layer(&vec![id], n), vec![(l, vec![e])]);
            // an expert number outside the layer would alias the next layer: refused
            assert_eq!(try_dense(l, n, n), None);
            assert_eq!(try_dense(l, n + (r() % 5) as u32, n), None);
        }
        for (l, e, n) in [
            (0, 0, 0),
            (u32::MAX, 0, 1),
            (u32::MAX, 0, u32::MAX),
            (1, u32::MAX, u32::MAX),
            (2, 0, 2_147_483_648),
            (1, 0, 4_294_967_295),
        ] {
            // never panics, never wraps: a result must split back to the same pair
            if let Some(id) = try_dense(l, e, n) {
                assert_eq!(split_dense(id, n), Some((l, e)));
            }
        }
        assert_eq!(split_dense(5, 0), None);
        assert!(by_layer(&vec![1, 2], 0).is_empty());
        assert_eq!(try_dense(u32::MAX, u32::MAX - 1, u32::MAX), None); // overflows u32
    }

    #[test]
    fn a_hostile_correlated_trace_request_is_bounded() {
        let t = correlated_trace(u32::MAX, 1, 1_000_000, 1, 1.0, 1);
        assert_eq!(t.len(), 1);
        assert!(t[0].len() <= MAX_LAYERS as usize);
        let t = correlated_trace(3, 4_000_000_000, usize::MAX, 1, 0.5, 1);
        assert_eq!(t.len(), 1);
        assert!(t[0].len() < 100_000);
    }
}
