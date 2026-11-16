//! A next-layer expert predictor. While layer L computes, the experts of layer L+1 could already
//! be read from disk if they were known. This predictor guesses them from the experts layer L
//! used, with plain co-occurrence counts (expert a at layer L, expert b at layer L+1, seen together
//! in the same token), falling back to per-layer popularity.
//!
//! This is a measurement, not an engine: it says what share of the loads a prefetcher with a given
//! budget could have issued in time. It only means something on a real trace; a trace sampled
//! from a popularity history has no correlation between layers by construction.

use crate::policy::Step;
use crate::trace::{dense, try_dense};
use std::collections::HashMap;

/// Experts of one token grouped by layer, in layer order: `(layer, experts)`.
pub fn by_layer(step: &Step, n_experts: u32) -> Vec<(u32, Vec<u32>)> {
    if n_experts == 0 {
        return Vec::new();
    }
    let mut out: Vec<(u32, Vec<u32>)> = Vec::new();
    for &id in step {
        let (l, e) = (id / n_experts, id % n_experts);
        match out.last_mut() {
            Some((pl, v)) if *pl == l => v.push(e),
            _ => out.push((l, vec![e])),
        }
    }
    out
}

#[derive(Default)]
pub struct Predictor {
    n_experts: u32,
    /// `next[a]`: how often each dense expert of layer L+1 followed dense expert `a` of layer L.
    next: HashMap<u32, HashMap<u32, u32>>,
    /// Per-layer popularity, the fallback when co-occurrence has nothing to say.
    pop: HashMap<u32, HashMap<u32, u32>>,
}

impl Predictor {
    pub fn train(steps: &[Step], n_experts: u32) -> Self {
        let mut p = Self {
            n_experts,
            ..Default::default()
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
                    let Some(key) = try_dense(*l0, x, n_experts) else {
                        continue;
                    };
                    let row = p.next.entry(key).or_default();
                    for &y in b {
                        *row.entry(y).or_default() += 1;
                    }
                }
            }
        }
        p
    }

    /// Up to `budget` experts (plain indices, not dense ids) guessed for layer `layer + 1`, given
    /// the experts the token used at `layer`. Co-occurrence first, then popularity.
    pub fn predict(&self, layer: u32, used: &[u32], budget: usize) -> Vec<u32> {
        let mut score: HashMap<u32, u64> = HashMap::new();
        let Some(next_layer) = layer.checked_add(1) else {
            return Vec::new();
        };
        for &x in used {
            let row = try_dense(layer, x, self.n_experts).and_then(|k| self.next.get(&k));
            if let Some(row) = row {
                for (&y, &c) in row {
                    *score.entry(y).or_default() += c as u64;
                }
            }
        }
        let mut v: Vec<(u32, u64)> = score.into_iter().collect();
        v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut out: Vec<u32> = v.into_iter().take(budget).map(|x| x.0).collect();
        if out.len() < budget {
            out.extend(self.popular(next_layer, budget, &out));
        }
        out
    }

    /// Popularity only: the baseline that needs no knowledge of the previous layer.
    pub fn popular(&self, layer: u32, budget: usize, skip: &[u32]) -> Vec<u32> {
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
    let cut = ((trace.len() as f64 * train_share) as usize).min(trace.len());
    let (train, test) = trace.split_at(cut);
    let p = Predictor::train(train, n_experts);
    let mut sc = Score {
        loads: 0,
        predicted: 0,
        popularity_only: 0,
    };
    for s in test {
        let layers = by_layer(s, n_experts);
        for w in layers.windows(2) {
            let ((l0, a), (l1, b)) = (&w[0], &w[1]);
            if l0.checked_add(1) != Some(*l1) {
                continue;
            }
            let budget = b.len().saturating_mul(mult);
            let guess = p.predict(*l0, a, budget);
            let base = p.popular(*l1, budget, &[]);
            sc.loads += b.len() as u64;
            sc.predicted += b.iter().filter(|e| guess.contains(e)).count() as u64;
            sc.popularity_only += b.iter().filter(|e| base.contains(e)).count() as u64;
        }
    }
    sc
}

/// A trace whose layers ARE correlated, to check the code: the experts of layer L+1 are a fixed
/// function of the experts of layer L, with probability `fidelity`; otherwise they are random.
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
            for l in 0..n_layers {
                cur.sort_unstable();
                cur.dedup();
                step.extend(cur.iter().map(|e| dense(l, *e, n_experts)));
                cur = cur
                    .iter()
                    .map(|&e| {
                        if rnd() < fidelity {
                            ((e as u64 * 7 + 3) % ne) as u32
                        } else {
                            (rnd() * n_experts as f64) as u32
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
}
