//! Which experts to keep in memory. Replays a routing trace (the set of experts each token used)
//! against cache policies and against the optimal offline policy (Belady), which no real engine
//! can reach but which bounds how much any cleverer policy or prefetcher could ever gain.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap as StdHashMap};
use std::hash::{BuildHasherDefault, Hasher};

/// A fast multiplicative hasher for small integer keys. The simulations hash millions of `u32`
/// expert ids, where the default SipHash dominates the run time; there is no untrusted party to
/// collide with, since the keys never decide anything but a cache's bookkeeping. As a side effect
/// iteration order is the same in every process.
#[derive(Default, Clone, Copy)]
pub struct FastHash(u64);

impl Hasher for FastHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write_u64(*b as u64);
        }
    }
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

/// A `HashMap` that uses `FastHash`.
pub type HashMap<K, V> = StdHashMap<K, V, BuildHasherDefault<FastHash>>;

/// One token's experts, as dense ids.
pub type Step = Vec<u32>;

pub trait Cache {
    /// Returns true on a hit, and makes room for `key` when it is a miss.
    fn access(&mut self, key: u32) -> bool;
}

/// Least recently used, in O(1) per access: a hash map from key to a slot, and the slots are
/// threaded into a doubly linked list (most recent at the head) stored in a vector.
pub struct Lru {
    cap: usize,
    slot: HashMap<u32, usize>,
    nodes: Vec<Node>,
    head: usize,
    tail: usize,
}

const NIL: usize = usize::MAX;

struct Node {
    key: u32,
    prev: usize,
    next: usize,
}

impl Lru {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            slot: HashMap::default(),
            nodes: Vec::new(),
            head: NIL,
            tail: NIL,
        }
    }

    fn unlink(&mut self, i: usize) {
        let (p, n) = (self.nodes[i].prev, self.nodes[i].next);
        match p {
            NIL => self.head = n,
            _ => self.nodes[p].next = n,
        }
        match n {
            NIL => self.tail = p,
            _ => self.nodes[n].prev = p,
        }
    }

    fn push_front(&mut self, i: usize) {
        self.nodes[i].prev = NIL;
        self.nodes[i].next = self.head;
        if self.head != NIL {
            self.nodes[self.head].prev = i;
        }
        self.head = i;
        if self.tail == NIL {
            self.tail = i;
        }
    }
}

impl Cache for Lru {
    fn access(&mut self, key: u32) -> bool {
        if let Some(&i) = self.slot.get(&key) {
            if self.head != i {
                self.unlink(i);
                self.push_front(i);
            }
            return true;
        }
        if self.cap == 0 {
            return false;
        }
        let i = if self.slot.len() >= self.cap {
            // reuse the slot of the least recently used key
            let t = self.tail;
            self.unlink(t);
            self.slot.remove(&self.nodes[t].key);
            self.nodes[t].key = key;
            t
        } else {
            self.nodes.push(Node {
                key,
                prev: NIL,
                next: NIL,
            });
            self.nodes.len() - 1
        };
        self.slot.insert(key, i);
        self.push_front(i);
        false
    }
}

/// Frequency with exponential decay: an expert's score is bumped on use and fades with time, so
/// experts a conversation keeps returning to stay while one-off visits leave first.
///
/// Every access is O(log n). Scores are kept as logarithms on a fixed time scale: with decay rate
/// `d` (negative), an entry whose score was `s` at tick `t0` has `z = ln(s) - d*t0`, and its score
/// at tick `t` is `exp(z + d*t)`. All entries fade at the same rate, so their order by `z` never
/// changes with time and a heap keyed on `z` always holds the next victim at the top. A hit
/// updates `z` with a numerically stable `ln(exp(a) + 1)`; stale heap entries are skipped when
/// popped (lazy deletion) and the heap is rebuilt when stale ones outnumber live ones.
///
/// The victim is the lowest score; equal scores go by the lowest key, so a replay is reproducible
/// (it also never depends on `HashMap` iteration order).
pub struct DecayFreq {
    cap: usize,
    decay: f64,
    clock: u64,
    stamp: u64,
    /// resident key -> (z, stamp of its live heap entry)
    live: HashMap<u32, (f64, u64)>,
    heap: BinaryHeap<Reverse<(Z, u32, u64)>>,
}

/// An f64 with a total order, so it can sit in a heap.
#[derive(Clone, Copy, PartialEq)]
struct Z(f64);
impl Eq for Z {}
impl PartialOrd for Z {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Z {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

impl DecayFreq {
    pub fn new(cap: usize, half_life: f64) -> Self {
        Self {
            cap,
            // `max` ignores a NaN half-life; anything below one tick counts as one tick
            decay: (0.5f64).ln() / half_life.max(1.0),
            clock: 0,
            stamp: 0,
            live: HashMap::default(),
            heap: BinaryHeap::new(),
        }
    }

    fn evict_one(&mut self) {
        while let Some(Reverse((_, k, st))) = self.heap.pop() {
            if self.live.get(&k).map(|v| v.1) == Some(st) {
                self.live.remove(&k);
                return;
            }
        }
    }

    fn compact(&mut self) {
        let items: Vec<_> = self
            .live
            .iter()
            .map(|(k, (z, st))| Reverse((Z(*z), *k, *st)))
            .collect();
        self.heap = BinaryHeap::from(items);
    }
}

impl Cache for DecayFreq {
    fn access(&mut self, key: u32) -> bool {
        self.clock += 1;
        let t = self.decay * self.clock as f64;
        let old = self.live.get(&key).map(|v| v.0);
        if old.is_none() {
            if self.cap == 0 {
                return false;
            }
            if self.live.len() >= self.cap {
                self.evict_one();
            }
        }
        // log of the new score: a miss starts at 1 (ln = 0); a hit adds 1 to the faded score
        let ln_s = match old {
            None => 0.0,
            Some(z) => {
                let a = z + t;
                if a > 0.0 {
                    a + (-a).exp().ln_1p()
                } else {
                    a.exp().ln_1p()
                }
            }
        };
        self.stamp += 1;
        let z = ln_s - t;
        self.live.insert(key, (z, self.stamp));
        self.heap.push(Reverse((Z(z), key, self.stamp)));
        if self.heap.len() > 2 * self.live.len() + 1024 {
            self.compact();
        }
        old.is_some()
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
///
/// Keys are first renumbered 0..n so the bookkeeping is plain arrays; the victim comes from a
/// max-heap of next-use positions with lazy deletion. Which of several keys that are never used
/// again goes first cannot change the hit count.
pub fn belady(trace: &[Step], cap: usize) -> f64 {
    let mut ids: HashMap<u32, usize> = HashMap::default();
    let flat: Vec<usize> = trace
        .iter()
        .flatten()
        .map(|k| {
            let next = ids.len();
            *ids.entry(*k).or_insert(next)
        })
        .collect();
    let n = flat.len();
    // next[i]: the position of the next use of flat[i], or n when there is none
    let mut next = vec![n; n];
    let mut seen = vec![n; ids.len()];
    for i in (0..n).rev() {
        next[i] = seen[flat[i]];
        seen[flat[i]] = i;
    }
    const OUT: usize = usize::MAX;
    let mut when = vec![OUT; ids.len()]; // resident key -> its next use, OUT when not resident
    let mut heap: std::collections::BinaryHeap<(usize, usize)> = Default::default();
    let (mut resident, mut hits) = (0usize, 0u64);
    for (i, &k) in flat.iter().enumerate() {
        if when[k] != OUT {
            hits += 1;
        } else if cap == 0 {
            continue;
        } else {
            if resident >= cap {
                while let Some((w, victim)) = heap.pop() {
                    if when[victim] == w {
                        when[victim] = OUT;
                        resident -= 1;
                        break;
                    }
                }
            }
            resident += 1;
        }
        when[k] = next[i];
        heap.push((next[i], k));
        if heap.len() > 2 * resident + (1 << 20) {
            heap = (0..when.len())
                .filter(|&x| when[x] != OUT)
                .map(|x| (when[x], x))
                .collect();
        }
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
                        .max_by_key(|&j| next(cache[j]).unwrap_or(usize::MAX))
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

    // --- DecayFreq: the heap version against the original O(cap)-per-miss scan ---

    /// The first implementation, kept as the reference: it scans the whole cache on every miss
    /// and recomputes every score from scratch.
    struct ScanDecayFreq {
        cap: usize,
        decay: f64,
        clock: u64,
        score: HashMap<u32, (f64, u64)>,
    }

    impl ScanDecayFreq {
        fn new(cap: usize, half_life: f64) -> Self {
            Self {
                cap,
                decay: (0.5f64).ln() / half_life.max(1.0),
                clock: 0,
                score: HashMap::default(),
            }
        }
        fn now(&self, s: (f64, u64)) -> f64 {
            s.0 * (self.decay * (self.clock - s.1) as f64).exp()
        }
    }

    impl Cache for ScanDecayFreq {
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
                    .min_by(|a, b| self.now(*a.1).total_cmp(&self.now(*b.1)).then(a.0.cmp(b.0)))
                    .map(|(k, _)| k)
                    .unwrap();
                self.score.remove(&victim);
            }
            let cur = self.score.get(&key).map_or(0.0, |s| self.now(*s));
            self.score.insert(key, (cur + 1.0, self.clock));
            hit
        }
    }

    fn hits(flat: &[u32], c: &mut dyn Cache) -> Vec<bool> {
        flat.iter().map(|k| c.access(*k)).collect()
    }

    #[test]
    fn the_heap_makes_the_same_decisions_as_the_scan_on_random_traces() {
        // Half-lives from 1.5 to 12345.6 on traces of up to 600 ticks: far enough from the
        // limit where scores that match to first order stop differing above f64 noise (about
        // half-life > 1e8), so both versions must pick the same victim every time.
        let mut r = xorshift(2026);
        let mut compared = 0u64;
        for case in 0..400 {
            let universe = 2 + r() % 40;
            let len = (r() % 600) as usize;
            // skew: half of the draws come from a hot quarter, so frequency matters
            let flat: Vec<u32> = (0..len)
                .map(|_| {
                    if r() % 2 == 0 {
                        (r() % universe.div_ceil(4)) as u32
                    } else {
                        (r() % universe) as u32
                    }
                })
                .collect();
            let cap = (r() % 14) as usize;
            for hl in [1.5, 3.3, 7.3, 41.7, 300.1, 12345.6] {
                let want = hits(&flat, &mut ScanDecayFreq::new(cap, hl));
                let got = hits(&flat, &mut DecayFreq::new(cap, hl));
                assert_eq!(got, want, "case {case} cap {cap} hl {hl} trace {flat:?}");
                compared += want.len() as u64;
            }
        }
        assert!(compared > 100_000, "{compared}");
    }

    #[test]
    fn the_heap_stays_close_to_the_live_set_however_long_the_trace() {
        let mut c = DecayFreq::new(16, 50.0);
        let mut r = xorshift(9);
        for _ in 0..200_000 {
            c.access((r() % 64) as u32);
            assert!(c.heap.len() <= 2 * c.live.len() + 1025);
            assert!(c.live.len() <= 16);
        }
    }

    #[test]
    fn scores_far_beyond_the_float_range_still_order_correctly() {
        // half-life of one tick: after 2000 ticks the linear score of an idle key is below 1e-600
        // and underflows to 0.0 in the scan version, which then evicts by key alone. In the log
        // domain it is still the lowest score, so the entry that was really used longest ago goes.
        let mut c = DecayFreq::new(2, 1.0);
        c.access(100); // used once, long ago
        for _ in 0..2000 {
            c.access(7); // keeps 7 alive
        }
        c.access(5); // cache is {100, 7}: a miss, one of them must go
        assert!(
            c.live.contains_key(&7) && c.live.contains_key(&5),
            "7 is hot, 100 is stale"
        );
        assert!(!c.live.contains_key(&100));
        for (k, (z, _)) in &c.live {
            assert!(z.is_finite(), "key {k}");
        }
    }

    #[test]
    fn a_very_long_replay_keeps_finite_scores_and_valid_hit_rates() {
        let mut c = DecayFreq::new(8, 300.0);
        let mut r = xorshift(31);
        let mut h = 0u64;
        for i in 0..2_000_000u64 {
            let key = if i % 3 == 0 {
                (r() % 6) as u32
            } else {
                (r() % 5000) as u32
            };
            h += c.access(key) as u64;
        }
        assert!(c.live.values().all(|(z, _)| z.is_finite()));
        assert!(h > 0 && h < 2_000_000);
    }

    #[test]
    fn without_decay_it_is_plain_frequency_with_ties_broken_by_key_like_the_scan() {
        // no fading: equal counts tie EXACTLY, so this checks the tie-break against the reference
        let mut r = xorshift(808);
        let mut ties_decided = 0usize;
        for _ in 0..300 {
            let universe = 3 + r() % 20;
            let flat: Vec<u32> = (0..(r() % 300) as usize)
                .map(|_| (r() % universe) as u32)
                .collect();
            let cap = 1 + (r() % 6) as usize;
            // only the exact no-decay case: for a finite half-life, two entries whose hit times have
            // the same sum agree to first order and differ by ~(ln2/hl)^2, which for a half-life
            // of 1e9 is below f64 noise: both versions then decide by rounding, differently
            let hl = f64::INFINITY;
            let want = hits(&flat, &mut ScanDecayFreq::new(cap, hl));
            let got = hits(&flat, &mut DecayFreq::new(cap, hl));
            assert_eq!(got, want, "cap {cap} trace {flat:?}");
            ties_decided += want.iter().filter(|h| !**h).count();
        }
        assert!(ties_decided > 1000);
    }

    #[test]
    fn lru_and_belady_match_the_references_on_larger_traces_and_extreme_keys() {
        let mut r = xorshift(555);
        let keys_pool: Vec<u32> = {
            let mut v: Vec<u32> = (0..150).collect();
            v.extend([u32::MAX, u32::MAX - 1, 1 << 31, (1 << 31) - 1]);
            v
        };
        for case in 0..40 {
            let len = 200 + (r() % 1500) as usize;
            let skew = 1 + r() % 5;
            let flat: Vec<u32> = (0..len)
                .map(|_| {
                    let a = (r() % keys_pool.len() as u64) as usize;
                    let b = (r() % keys_pool.len() as u64) as usize;
                    keys_pool[(a.min(b) * skew as usize / 5).min(keys_pool.len() - 1)]
                })
                .collect();
            let trace: Vec<Step> = flat.chunks(7).map(|c| c.to_vec()).collect();
            for cap in [0usize, 1, 2, 3, 17, 64, 200] {
                let n = flat.len() as f64;
                assert_eq!(
                    replay(&trace, &mut Lru::new(cap)),
                    naive_lru(&flat, cap) as f64 / n,
                    "lru case {case} cap {cap}"
                );
                assert_eq!(
                    belady(&trace, cap),
                    naive_belady(&flat, cap) as f64 / n,
                    "belady case {case} cap {cap}"
                );
            }
        }
    }

    #[test]
    fn belady_stays_correct_when_its_heap_is_rebuilt_midway() {
        // a long trace over few keys pushes the heap past its compaction threshold
        let mut r = xorshift(13);
        let flat: Vec<u32> = (0..(3 << 20)).map(|_| (r() % 6) as u32).collect();
        let trace = vec![flat.clone()];
        let b = belady(&trace, 4);
        // with 6 keys and room for 4, the optimal policy misses only when it must
        assert!(b > 0.6 && b < 1.0, "{b}");
        assert_eq!(belady(&trace, 6), 1.0 - 6.0 / flat.len() as f64);
    }

    #[test]
    fn lru_with_room_for_one_key_only_hits_on_repeats() {
        let flat = [1u32, 1, 2, 1, 1, 2, 2];
        let t = vec![flat.to_vec()];
        assert_eq!(replay(&t, &mut Lru::new(1)), 3.0 / 7.0);
    }
}
