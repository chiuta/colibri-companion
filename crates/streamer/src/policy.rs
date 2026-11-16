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
                .min_by(|a, b| self.now(*a.1).total_cmp(&self.now(*b.1)))
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
            while step.len() < k {
                let u = rnd();
                let r = ((n as f64).powf(u) - 1.0) as u32 % n; // skewed toward small ranks
                let id = (r + shift) % n;
                if !step.contains(&id) {
                    step.push(id);
                }
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
}
