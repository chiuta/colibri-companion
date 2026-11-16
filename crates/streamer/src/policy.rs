//! Which experts to keep in memory. Replays a routing trace (the set of experts each token used)
//! against cache policies and against the optimal offline policy (Belady), which no real engine
//! can reach but which bounds how much any cleverer policy or prefetcher could ever gain.

use std::collections::HashMap;

/// One token's experts, as dense ids.
pub type Step = Vec<u32>;

pub trait Cache {
    /// Returns true on a hit, and makes room for `key` when it is a miss.
    fn access(&mut self, key: u32) -> bool;
}

pub struct Lru {
    cap: usize,
    clock: u64,
    last: HashMap<u32, u64>,
    order: std::collections::BTreeMap<u64, u32>,
}

impl Lru {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            clock: 0,
            last: HashMap::new(),
            order: std::collections::BTreeMap::new(),
        }
    }
}

impl Cache for Lru {
    fn access(&mut self, key: u32) -> bool {
        self.clock += 1;
        if let Some(old) = self.last.insert(key, self.clock) {
            self.order.remove(&old);
            self.order.insert(self.clock, key);
            return true;
        }
        if self.cap == 0 {
            self.last.remove(&key);
            return false;
        }
        self.order.insert(self.clock, key);
        if self.last.len() > self.cap {
            let (&t, &victim) = self.order.iter().next().unwrap();
            self.order.remove(&t);
            self.last.remove(&victim);
        }
        false
    }
}

/// Frequency with exponential decay: an expert's score is bumped on use and fades with time, so
/// experts a conversation keeps returning to stay while one-off visits leave first.
pub struct DecayFreq {
    cap: usize,
    decay: f64,
    clock: u64,
    score: HashMap<u32, (f64, u64)>,
}

impl DecayFreq {
    pub fn new(cap: usize, half_life: f64) -> Self {
        Self {
            cap,
            decay: (0.5f64).ln() / half_life.max(1.0),
            clock: 0,
            score: HashMap::new(),
        }
    }
    fn now(&self, s: (f64, u64)) -> f64 {
        s.0 * (self.decay * (self.clock - s.1) as f64).exp()
    }
}

impl Cache for DecayFreq {
    fn access(&mut self, key: u32) -> bool {
        self.clock += 1;
        let hit = self.score.contains_key(&key);
        if !hit && self.cap == 0 {
            return false;
        }
        if !hit && self.score.len() >= self.cap {
            let victim = *self
                .score
                .iter()
                .min_by(|a, b| {
                    // lowest score first; equal scores go by key so a replay is reproducible
                    // (HashMap iteration order is random per process)
                    self.now(*a.1).total_cmp(&self.now(*b.1)).then(a.0.cmp(b.0))
                })
                .map(|(k, _)| k)
                .unwrap();
            self.score.remove(&victim);
        }
        let cur = self.score.get(&key).map_or(0.0, |s| self.now(*s));
        self.score.insert(key, (cur + 1.0, self.clock));
        hit
    }
}

/// Hit rate of a policy over a trace (every access of every step counts once).
pub fn replay(trace: &[Step], cache: &mut dyn Cache) -> f64 {
    let (mut hits, mut total) = (0u64, 0u64);
    for step in trace {
        for k in step {
            total += 1;
            hits += cache.access(*k) as u64;
        }
    }
    if total == 0 {
        1.0
    } else {
        hits as f64 / total as f64
    }
}

/// Belady's optimal policy: on a miss evict the resident key whose next use is farthest away.
/// Needs the future, so it is a ceiling for measurement, never something an engine can run.
pub fn belady(trace: &[Step], cap: usize) -> f64 {
    let flat: Vec<u32> = trace.iter().flatten().copied().collect();
    let n = flat.len();
    // next[i]: the position of the next use of flat[i], or n when there is none
    let mut next = vec![n; n];
    let mut seen: HashMap<u32, usize> = HashMap::new();
    for i in (0..n).rev() {
        if let Some(p) = seen.insert(flat[i], i) {
            next[i] = p;
        }
    }
    let mut when: HashMap<u32, usize> = HashMap::new(); // resident key -> its next use
    let mut by_when: std::collections::BTreeSet<(usize, u32)> = Default::default();
    let mut hits = 0u64;
    for (i, &k) in flat.iter().enumerate() {
        if let Some(w) = when.get(&k).copied() {
            hits += 1;
            by_when.remove(&(w, k));
            when.insert(k, next[i]);
            by_when.insert((next[i], k));
            continue;
        }
        if cap == 0 {
            continue;
        }
        if when.len() >= cap {
            let &(w, victim) = by_when.iter().next_back().unwrap();
            by_when.remove(&(w, victim));
            when.remove(&victim);
        }
        when.insert(k, next[i]);
        by_when.insert((next[i], k));
    }
    if n == 0 { 1.0 } else { hits as f64 / n as f64 }
}

/// A deterministic synthetic trace: `tokens` steps of `k` distinct experts out of `n`, drawn from a
/// Zipf-like popularity with a topic that drifts, so recency and frequency both carry signal.
pub fn synthetic_trace(n: u32, k: usize, tokens: usize, seed: u64) -> Vec<Step> {
    if n == 0 {
        return vec![Vec::new(); tokens];
    }
    let k = k.min(n as usize); // cannot draw more distinct experts than exist
    let mut s = seed | 1;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut shift = 0u32;
    (0..tokens)
        .map(|t| {
            if t % 200 == 0 {
                shift = (rnd() * n as f64) as u32;
            }
            let mut step = Vec::with_capacity(k);
            // the skewed draw may be unable to reach k distinct ids (small n): after a bounded
            // number of tries, fill the rest in order so the loop always ends
            let mut tries = 0usize;
            while step.len() < k && tries < 64 * k {
                tries += 1;
                let u = rnd();
                let r = ((n as f64).powf(u) - 1.0) as u32 % n; // skewed toward small ranks
                let id = (r as u64 + shift as u64) as u32 % n;
                if !step.contains(&id) {
                    step.push(id);
                }
            }
            let mut j = 0u32;
            while step.len() < k {
                let id = (shift.wrapping_add(j)) % n;
                if !step.contains(&id) {
                    step.push(id);
                }
                j += 1;
            }
            step
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_working_set_that_fits_is_all_hits_after_the_first_pass() {
        let trace: Vec<Step> = (0..50).map(|_| vec![1, 2, 3]).collect();
        let r = replay(&trace, &mut Lru::new(3));
        assert!(r > 0.97, "{r}");
        assert!(belady(&trace, 3) > 0.97);
    }

    #[test]
    fn belady_is_never_beaten() {
        let trace = synthetic_trace(64, 4, 600, 7);
        for cap in [4, 8, 16, 32] {
            let b = belady(&trace, cap);
            assert!(b + 1e-9 >= replay(&trace, &mut Lru::new(cap)), "cap {cap}");
            assert!(
                b + 1e-9 >= replay(&trace, &mut DecayFreq::new(cap, 200.0)),
                "cap {cap}"
            );
        }
    }

    #[test]
    fn a_bigger_cache_never_hits_less_under_the_optimal_policy() {
        let trace = synthetic_trace(64, 4, 600, 11);
        let (a, b) = (belady(&trace, 8), belady(&trace, 16));
        assert!(b >= a);
    }

    #[test]
    fn zero_capacity_misses_everything() {
        let trace: Vec<Step> = vec![vec![1, 2], vec![1, 2]];
        assert_eq!(replay(&trace, &mut Lru::new(0)), 0.0);
        assert_eq!(belady(&trace, 0), 0.0);
    }

    // --- cross-checks against slow, obviously correct references ---

    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    fn random_trace(r: &mut impl FnMut() -> u64) -> Vec<Step> {
        let universe = 1 + r() % 12;
        let tokens = (r() % 14) as usize;
        (0..tokens)
            .map(|_| (0..(r() % 5)).map(|_| (r() % universe) as u32).collect())
            .collect()
    }

    fn naive_lru(flat: &[u32], cap: usize) -> u64 {
        let mut cache: Vec<u32> = Vec::new(); // front = least recently used
        let mut hits = 0;
        for &k in flat {
            if let Some(i) = cache.iter().position(|x| *x == k) {
                cache.remove(i);
                cache.push(k);
                hits += 1;
            } else if cap > 0 {
                if cache.len() == cap {
                    cache.remove(0);
                }
                cache.push(k);
            }
        }
        hits
    }

    fn naive_belady(flat: &[u32], cap: usize) -> u64 {
        let mut cache: Vec<u32> = Vec::new();
        let mut hits = 0;
        for (i, &k) in flat.iter().enumerate() {
            if cache.contains(&k) {
                hits += 1;
            } else if cap > 0 {
                if cache.len() == cap {
                    let next = |x: u32| flat[i + 1..].iter().position(|y| *y == x);
                    let victim = (0..cache.len())
                        .max_by_key(|&j| next(cache[j]).map_or(usize::MAX, |p| p))
                        .unwrap();
                    cache.remove(victim);
                }
                cache.push(k);
            }
        }
        hits
    }

    #[test]
    fn lru_and_belady_agree_with_brute_force_on_random_traces() {
        let mut r = xorshift(12345);
        for _ in 0..3000 {
            let trace = random_trace(&mut r);
            let flat: Vec<u32> = trace.iter().flatten().copied().collect();
            let cap = (r() % 9) as usize;
            let n = flat.len();
            let want_lru = naive_lru(&flat, cap);
            let got_lru = replay(&trace, &mut Lru::new(cap));
            let want_opt = naive_belady(&flat, cap);
            let got_opt = belady(&trace, cap);
            if n > 0 {
                assert_eq!(
                    got_lru,
                    want_lru as f64 / n as f64,
                    "lru {flat:?} cap {cap}"
                );
                assert_eq!(
                    got_opt,
                    want_opt as f64 / n as f64,
                    "opt {flat:?} cap {cap}"
                );
                assert!(got_opt >= got_lru);
                let d = replay(&trace, &mut DecayFreq::new(cap, 5.0));
                assert!(
                    got_opt + 1e-12 >= d,
                    "decay beat optimal {flat:?} cap {cap}"
                );
                assert!((0.0..=1.0).contains(&d));
            }
        }
    }

    #[test]
    fn optimal_never_hits_less_with_more_room_on_random_traces() {
        let mut r = xorshift(777);
        for _ in 0..500 {
            let trace = random_trace(&mut r);
            let mut prev = 0.0;
            for cap in 0..10 {
                let b = belady(&trace, cap);
                assert!(b + 1e-12 >= prev, "{trace:?} cap {cap}");
                prev = b;
            }
        }
    }

    #[test]
    fn decay_freq_is_reproducible_even_when_scores_tie() {
        // every key seen once: all scores tie, so the victim must follow the key order
        let trace: Vec<Step> = (0..40).map(|i| vec![i % 13, (i * 5 + 1) % 13]).collect();
        let first = replay(&trace, &mut DecayFreq::new(4, 1e9));
        for _ in 0..30 {
            assert_eq!(replay(&trace, &mut DecayFreq::new(4, 1e9)), first);
        }
    }

    #[test]
    fn decay_freq_survives_degenerate_half_lives_and_long_traces() {
        let trace = synthetic_trace(32, 4, 3000, 3);
        for hl in [0.0, f64::NAN, f64::INFINITY, -5.0, 1e-300] {
            let h = replay(&trace, &mut DecayFreq::new(8, hl));
            assert!(
                h.is_finite() && (0.0..=1.0).contains(&h),
                "half-life {hl}: {h}"
            );
        }
    }

    #[test]
    fn empty_traces_and_zero_capacity_do_not_panic() {
        assert_eq!(replay(&[], &mut Lru::new(0)), 1.0);
        assert_eq!(belady(&[], 0), 1.0);
        assert_eq!(replay(&[vec![], vec![]], &mut DecayFreq::new(0, 10.0)), 1.0);
    }

    #[test]
    fn a_synthetic_trace_asking_for_more_experts_than_exist_terminates() {
        let t = synthetic_trace(3, 8, 5, 1);
        assert!(t.iter().all(|s| s.len() == 3));
        assert_eq!(synthetic_trace(0, 4, 2, 1), vec![Vec::<u32>::new(); 2]);
    }

    #[test]
    fn a_skewed_draw_that_cannot_reach_k_distinct_ids_still_ends() {
        // n = 4: the power-law draw only ever reaches ranks 0..=2, so 4 distinct needs the fill
        let t = synthetic_trace(4, 4, 6, 9);
        assert!(t.iter().all(|s| {
            let mut v = s.clone();
            v.sort_unstable();
            v == vec![0, 1, 2, 3]
        }));
    }
}
