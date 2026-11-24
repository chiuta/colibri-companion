//! Cheap timing for any [`Backend`]: how many calls of each kind, how long the host spent in them.
//!
//! Submission is deliberately simple and asynchronous-free: a device backend records work and
//! submits it once per token, when the host reads a result (`read`) or asks to wait
//! (`synchronize`). So on a device the time of a whole token shows up under `read`/`synchronize`
//! and the other calls only measure recording; on the CPU backend each call is real work. The
//! report says which it is rather than pretending the numbers are comparable.
use companion_engine::backend::*;
use companion_engine::quant::DType;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Op { Upload, Alloc, Write, Read, Matvec, Matmat, Rmsnorm, Rope, SiluMul, Add, KvCreate, KvAppend, Attention, Synchronize }

pub const OPS: [Op; 14] = [Op::Upload, Op::Alloc, Op::Write, Op::Read, Op::Matvec, Op::Matmat, Op::Rmsnorm, Op::Rope, Op::SiluMul, Op::Add, Op::KvCreate, Op::KvAppend, Op::Attention, Op::Synchronize];

impl Op {
    pub fn name(self) -> &'static str {
        match self {
            Op::Upload => "upload", Op::Alloc => "alloc", Op::Write => "write", Op::Read => "read", Op::Matvec => "matvec", Op::Matmat => "matmat",
            Op::Rmsnorm => "rmsnorm", Op::Rope => "rope", Op::SiluMul => "silu_mul", Op::Add => "add", Op::KvCreate => "kv_create", Op::KvAppend => "kv_append",
            Op::Attention => "attention", Op::Synchronize => "synchronize",
        }
    }
    /// Operations where a device backend actually waits for the work to finish.
    pub fn is_sync_point(self) -> bool { matches!(self, Op::Read | Op::Synchronize) }
    fn index(self) -> usize { OPS.iter().position(|o| *o == self).unwrap_or(0) }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpStat { pub calls: u64, pub total: Duration, pub max: Duration }

/// Counters a device backend keeps about its submissions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubmitStats {
    /// Command buffers handed to the queue.
    pub submits: u64,
    /// Compute passes recorded.
    pub dispatches: u64,
    pub bytes_up: u64,
    pub bytes_down: u64,
    /// Host time spent building and submitting commands.
    pub submit_time: Duration,
    /// Host time spent waiting for the device to finish and for read-back.
    pub wait_time: Duration,
}

impl SubmitStats {
    pub fn dispatches_per_submit(&self) -> f64 { if self.submits == 0 { 0.0 } else { self.dispatches as f64 / self.submits as f64 } }
    pub fn text(&self) -> String {
        format!("{} submits, {} dispatches ({:.1} per submit), {} KiB up, {} KiB down, {:.2} ms submitting, {:.2} ms waiting",
            self.submits, self.dispatches, self.dispatches_per_submit(), self.bytes_up >> 10, self.bytes_down >> 10, self.submit_time.as_secs_f64() * 1e3, self.wait_time.as_secs_f64() * 1e3)
    }
}

/// Per-operation figures taken from a [`TimedBackend`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Timings { pub backend: String, pub accelerator: bool, stats: [OpStat; 14] }

impl Timings {
    pub fn stat(&self, op: Op) -> OpStat { self.stats[op.index()] }
    pub fn total(&self) -> Duration { self.stats.iter().map(|s| s.total).sum() }
    pub fn calls(&self) -> u64 { self.stats.iter().map(|s| s.calls).sum() }
    /// Tokens per second for `tokens` tokens that took the host-time recorded here.
    pub fn tokens_per_second(&self, tokens: u64) -> f64 { let t = self.total().as_secs_f64(); if t > 0.0 { tokens as f64 / t } else { 0.0 } }
    pub fn report(&self) -> String {
        let mut out = format!("{}: {} calls, {:.2} ms in the backend\n", self.backend, self.calls(), self.total().as_secs_f64() * 1e3);
        let mut rows: Vec<(Op, OpStat)> = OPS.iter().map(|o| (*o, self.stat(*o))).filter(|(_, s)| s.calls > 0).collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1.total));
        for (o, s) in rows {
            out.push_str(&format!("  {:<12} {:>7} calls {:>10.3} ms total {:>9.4} ms max\n", o.name(), s.calls, s.total.as_secs_f64() * 1e3, s.max.as_secs_f64() * 1e3));
        }
        if self.accelerator { out.push_str("  note: work is submitted when the host reads a result, so device time appears under read/synchronize\n"); }
        out
    }
}

/// Wraps a backend and times every call. Forwards everything, changes nothing about the results.
pub struct TimedBackend<B: Backend> { inner: B, stats: Mutex<[OpStat; 14]> }

impl<B: Backend> TimedBackend<B> {
    pub fn new(inner: B) -> Self { TimedBackend { inner, stats: Mutex::new([OpStat::default(); 14]) } }
    pub fn inner(&self) -> &B { &self.inner }
    pub fn into_inner(self) -> B { self.inner }
    pub fn timings(&self) -> Timings {
        let stats = self.stats.lock().map(|g| *g).unwrap_or([OpStat::default(); 14]);
        Timings { backend: self.inner.name(), accelerator: self.inner.is_accelerator(), stats }
    }
    pub fn reset(&self) { if let Ok(mut g) = self.stats.lock() { *g = [OpStat::default(); 14]; } }
    fn time<T>(&self, op: Op, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let r = f();
        let d = t.elapsed();
        if let Ok(mut g) = self.stats.lock() {
            let s = &mut g[op.index()];
            s.calls += 1;
            s.total += d;
            s.max = s.max.max(d);
        }
        r
    }
}

impl<B: Backend> Backend for TimedBackend<B> {
    fn name(&self) -> String { format!("timed {}", self.inner.name()) }
    fn is_accelerator(&self) -> bool { self.inner.is_accelerator() }
    fn device_memory(&self) -> Option<u64> { self.inner.device_memory() }
    fn supports(&self, t: DType) -> bool { self.inner.supports(t) }
    fn alloc(&self, len: usize) -> Result<Buf> { self.time(Op::Alloc, || self.inner.alloc(len)) }
    fn write(&self, dst: &Buf, src: &[f32]) -> Result<()> { self.time(Op::Write, || self.inner.write(dst, src)) }
    fn read(&self, src: &Buf, dst: &mut [f32]) -> Result<()> { self.time(Op::Read, || self.inner.read(src, dst)) }
    fn upload(&self, w: WeightRef<'_>) -> Result<WeightHandle> { self.time(Op::Upload, || self.inner.upload(w)) }
    fn matvec(&self, w: &WeightHandle, x: &Buf, out: &Buf) -> Result<()> { self.time(Op::Matvec, || self.inner.matvec(w, x, out)) }
    fn matmat(&self, w: &WeightHandle, x: &Buf, batch: usize, out: &Buf) -> Result<()> { self.time(Op::Matmat, || self.inner.matmat(w, x, batch, out)) }
    fn rmsnorm(&self, x: &Buf, weight: &Buf, eps: f32, out: &Buf) -> Result<()> { self.time(Op::Rmsnorm, || self.inner.rmsnorm(x, weight, eps, out)) }
    fn rope(&self, x: &Buf, n_heads: usize, head_dim: usize, pos: usize, theta: f64) -> Result<()> { self.time(Op::Rope, || self.inner.rope(x, n_heads, head_dim, pos, theta)) }
    fn silu_mul(&self, g: &Buf, u: &Buf) -> Result<()> { self.time(Op::SiluMul, || self.inner.silu_mul(g, u)) }
    fn add(&self, a: &Buf, b: &Buf) -> Result<()> { self.time(Op::Add, || self.inner.add(a, b)) }
    fn kv_create(&self, kv_heads: usize, head_dim: usize, capacity: usize) -> Result<Kv> { self.time(Op::KvCreate, || self.inner.kv_create(kv_heads, head_dim, capacity)) }
    fn kv_append(&self, kv: &Kv, pos: usize, k: &Buf, v: &Buf) -> Result<()> { self.time(Op::KvAppend, || self.inner.kv_append(kv, pos, k, v)) }
    fn attention(&self, kv: &Kv, q: &Buf, out: &Buf, a: &AttnArgs) -> Result<()> { self.time(Op::Attention, || self.inner.attention(kv, q, out, a)) }
    fn synchronize(&self) -> Result<()> { self.time(Op::Synchronize, || self.inner.synchronize()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(n: usize, k: f32) -> Vec<f32> { (0..n).map(|i| (i as f32 * k).sin()).collect() }

    #[test]
    fn results_are_identical_to_the_untimed_backend_and_calls_are_counted() {
        let (t, plain) = (TimedBackend::new(CpuBackend::new()), CpuBackend::new());
        let (rows, cols) = (8usize, 16usize);
        let w = wave(rows * cols, 0.21);
        let run = |be: &dyn Backend| {
            let h = be.upload(WeightRef::dense(rows, cols, &w)).unwrap();
            let (x, o) = (be.alloc_from(&wave(cols, 0.7)).unwrap(), be.alloc(rows).unwrap());
            be.matvec(&h, &x, &o).unwrap();
            be.matvec(&h, &x, &o).unwrap();
            be.synchronize().unwrap();
            let mut out = vec![0.0; rows];
            be.read(&o, &mut out).unwrap();
            out
        };
        assert_eq!(run(&t), run(&plain));
        let tm = t.timings();
        assert_eq!(tm.stat(Op::Matvec).calls, 2);
        assert_eq!(tm.stat(Op::Upload).calls, 1);
        assert_eq!(tm.stat(Op::Read).calls, 1);
        assert_eq!(tm.stat(Op::Synchronize).calls, 1);
        assert_eq!(tm.stat(Op::Alloc).calls, 2, "alloc_from allocates through the wrapper's own alloc");
        assert_eq!(tm.calls(), OPS.iter().map(|o| tm.stat(*o).calls).sum::<u64>());
        assert!(tm.total() >= tm.stat(Op::Matvec).total);
        assert!(!tm.accelerator);
        let r = tm.report();
        assert!(r.contains("matvec") && r.contains("2 calls"), "{r}");
        assert!(!r.contains("submitted when the host reads"), "the CPU backend does no deferred submission");
    }

    #[test]
    fn reset_clears_and_the_wrapper_is_transparent() {
        let t = TimedBackend::new(CpuBackend::new());
        let b = t.alloc(4).unwrap();
        t.write(&b, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(t.timings().stat(Op::Write).calls, 1);
        t.reset();
        assert_eq!(t.timings().calls(), 0);
        assert!(t.name().starts_with("timed "));
        assert_eq!(t.device_memory(), CpuBackend::new().device_memory());
        assert!(t.supports(DType::Q4K) == CpuBackend::new().supports(DType::Q4K));
        // Errors pass through unchanged and are still counted.
        let bad = t.write(&b, &[1.0]);
        assert!(matches!(bad, Err(BackendError::Shape(_))));
        assert_eq!(t.timings().stat(Op::Write).calls, 1, "the failed call is counted too");
    }

    #[test]
    fn sync_points_and_the_submit_summary() {
        assert!(Op::Read.is_sync_point() && Op::Synchronize.is_sync_point() && !Op::Matvec.is_sync_point());
        let s = SubmitStats { submits: 2, dispatches: 90, bytes_up: 4096, bytes_down: 2048, ..Default::default() };
        assert_eq!(s.dispatches_per_submit(), 45.0);
        assert!(s.text().contains("45.0 per submit"));
        assert_eq!(SubmitStats::default().dispatches_per_submit(), 0.0);
        let mut seen = std::collections::HashSet::new();
        for o in OPS { assert!(seen.insert(o.index()), "{o:?} index must be unique"); }
    }

    #[test]
    fn a_whole_attention_step_through_the_wrapper_matches_the_plain_backend() {
        let (t, plain) = (TimedBackend::new(CpuBackend::new()), CpuBackend::new());
        let (kvh, hd, heads, cap) = (2usize, 8usize, 4usize, 8usize);
        let run = |be: &dyn Backend| {
            let kv = be.kv_create(kvh, hd, cap).unwrap();
            for p in 0..3 {
                let (k, v) = (be.alloc_from(&wave(kvh * hd, 0.3 + p as f32)).unwrap(), be.alloc_from(&wave(kvh * hd, 0.9 - p as f32 * 0.1)).unwrap());
                be.kv_append(&kv, p, &k, &v).unwrap();
            }
            let (q, o) = (be.alloc_from(&wave(heads * hd, 0.5)).unwrap(), be.alloc(heads * hd).unwrap());
            be.attention(&kv, &q, &o, &AttnArgs { n_heads: heads, n_pos: 3, scale: 0.35, window: None }).unwrap();
            let mut out = vec![0.0; heads * hd];
            be.read(&o, &mut out).unwrap();
            out
        };
        assert_eq!(run(&t), run(&plain));
        assert_eq!(t.timings().stat(Op::Attention).calls, 1);
        assert_eq!(t.timings().stat(Op::KvAppend).calls, 3);
    }
}
