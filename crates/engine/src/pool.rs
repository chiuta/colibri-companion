//! Persistent std-only thread pool for data-parallel loops (matrix–vector products, attention heads).
//!
//! * Workers are created once; a call never spawns a thread.
//! * `parallel_for` splits `0..n` into chunks claimed through an atomic counter, so fast cores
//!   simply claim more chunks (good on P+E / big.LITTLE machines). The caller works too.
//! * The call blocks until every chunk is finished, which is what makes borrowing closures sound.
//! * A panic in any chunk is caught, the remaining chunks are skipped and the first panic is
//!   re-raised in the caller. The pool stays usable afterwards.
//! * Re-entrant calls (a chunk calling `parallel_for`) and calls while another thread is using the
//!   pool run inline on the calling thread, so there is no deadlock and no oversubscription.
//!
//! Thread affinity is intentionally not set: pinning is platform specific and, with the dynamic
//! chunking above, rarely worth the risk. [`AffinityHint`] only documents where it would go.
use std::any::Any;
use std::cell::Cell;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;

thread_local! {
    /// True while this thread is executing chunks of a pool job (worker or participating caller).
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Best-effort placement hint. Currently a documented no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AffinityHint {
    #[default]
    None,
    PerformanceCores,
}

struct FnPtr(*const (dyn Fn(Range<usize>) + Sync));
// SAFETY: the pointee is `Sync`, and the pointer is only dereferenced while the owning
// `parallel_for` call is still blocked waiting for the job (see `run_chunks`).
unsafe impl Send for FnPtr {}
unsafe impl Sync for FnPtr {}

struct Job {
    f: FnPtr,
    n: usize,
    chunk: usize,
    total: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    failed: AtomicBool,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
    fin: Mutex<bool>,
    cv: Condvar,
}

fn run_chunks(job: &Job) {
    loop {
        let idx = job.next.fetch_add(1, Ordering::Relaxed);
        if idx >= job.total {
            break;
        }
        let start = idx * job.chunk; // idx < ceil(n/chunk) so this is < n
        let end = start.saturating_add(job.chunk).min(job.n);
        if !job.failed.load(Ordering::Acquire) {
            // SAFETY: this chunk is claimed but not yet counted in `done`, so `done < total` and the
            // caller of `parallel_for` is still blocked; the closure it borrowed is alive.
            let f = unsafe { &*job.f.0 };
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| f(start..end))) {
                job.failed.store(true, Ordering::Release);
                let mut g = lock(&job.panic);
                if g.is_none() {
                    *g = Some(p);
                }
            }
        }
        if job.done.fetch_add(1, Ordering::AcqRel) + 1 == job.total {
            *lock(&job.fin) = true;
            job.cv.notify_all();
        }
    }
}

struct Slot {
    job: Option<Arc<Job>>,
    epoch: u64,
    shutdown: bool,
}
struct Shared {
    slot: Mutex<Slot>,
    cv: Condvar,
    epoch_hint: AtomicU64,
    quit: AtomicBool,
}

fn worker(sh: Arc<Shared>) {
    let mut seen = 0u64;
    BUSY.with(|b| b.set(true));
    loop {
        // Short spin first: during decoding the next matrix product arrives within microseconds.
        let mut spins = 0u32;
        while sh.epoch_hint.load(Ordering::Acquire) == seen
            && !sh.quit.load(Ordering::Acquire)
            && spins < 3000
        {
            std::hint::spin_loop();
            spins += 1;
        }
        let job = {
            let mut g = lock(&sh.slot);
            loop {
                if g.shutdown {
                    return;
                }
                if g.epoch != seen {
                    seen = g.epoch;
                    if let Some(j) = g.job.clone() {
                        break j;
                    }
                    continue;
                }
                g = sh.cv.wait(g).unwrap_or_else(|e| e.into_inner());
            }
        };
        run_chunks(&job);
    }
}

struct SendPtr<T>(*mut T);
impl<T> SendPtr<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}
// SAFETY: only used to hand out disjoint sub-slices of one `&mut [T]` to different threads.
unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}

pub struct Pool {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
    call_lock: Mutex<()>,
    threads: usize,
}

impl Pool {
    /// A pool that runs jobs on `n` threads in total (the caller plus `n - 1` workers).
    pub fn new(n: usize) -> Pool {
        let want = n.max(1);
        let shared = Arc::new(Shared {
            slot: Mutex::new(Slot {
                job: None,
                epoch: 0,
                shutdown: false,
            }),
            cv: Condvar::new(),
            epoch_hint: AtomicU64::new(0),
            quit: AtomicBool::new(false),
        });
        let mut handles = Vec::new();
        for i in 1..want {
            let sh = shared.clone();
            match std::thread::Builder::new()
                .name(format!("engine-pool-{i}"))
                .spawn(move || worker(sh))
            {
                Ok(h) => handles.push(h),
                Err(_) => break, // fewer threads is fine
            }
        }
        let threads = handles.len() + 1;
        Pool {
            shared,
            handles,
            call_lock: Mutex::new(()),
            threads,
        }
    }

    /// Threads that take part in a job, including the caller.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Documented no-op, see the module docs.
    pub fn set_affinity_hint(&self, _hint: AffinityHint) {}

    /// Run `f` over disjoint sub-ranges covering `0..n`; every chunk has at least `min_chunk`
    /// items (except possibly the last). Returns when all chunks are done.
    pub fn parallel_for<F: Fn(Range<usize>) + Sync>(&self, n: usize, min_chunk: usize, f: F) {
        if n == 0 {
            return;
        }
        let min_chunk = min_chunk.max(1);
        if self.threads <= 1 || n <= min_chunk || BUSY.with(Cell::get) {
            return f(0..n);
        }
        let chunk = n.div_ceil(self.threads * 4).max(min_chunk);
        if chunk >= n {
            return f(0..n);
        }
        let Ok(guard) = self.call_lock.try_lock() else {
            return f(0..n);
        };

        let fref: &(dyn Fn(Range<usize>) + Sync) = &f;
        // SAFETY: lifetime erasure only. We do not return before `done == total`, after which no
        // thread can reach the dereference in `run_chunks` (all chunk indices are claimed).
        let fref: &'static (dyn Fn(Range<usize>) + Sync) = unsafe { std::mem::transmute(fref) };
        let total = n.div_ceil(chunk);
        let job = Arc::new(Job {
            f: FnPtr(fref as *const _),
            n,
            chunk,
            total,
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            failed: AtomicBool::new(false),
            panic: Mutex::new(None),
            fin: Mutex::new(false),
            cv: Condvar::new(),
        });
        {
            let mut g = lock(&self.shared.slot);
            g.job = Some(job.clone());
            g.epoch += 1;
            self.shared.epoch_hint.store(g.epoch, Ordering::Release);
        }
        self.shared.cv.notify_all();

        BUSY.with(|b| b.set(true));
        run_chunks(&job);
        BUSY.with(|b| b.set(false));
        {
            let mut fin = lock(&job.fin);
            while !*fin {
                fin = job.cv.wait(fin).unwrap_or_else(|e| e.into_inner());
            }
        }
        lock(&self.shared.slot).job = None;
        let payload = lock(&job.panic).take();
        drop(guard);
        if let Some(p) = payload {
            resume_unwind(p);
        }
    }

    /// Like [`Pool::parallel_for`] but hands each chunk its own mutable sub-slice and its start index.
    pub fn parallel_chunks_mut<T: Send, F: Fn(usize, &mut [T]) + Sync>(
        &self,
        data: &mut [T],
        min_chunk: usize,
        f: F,
    ) {
        let n = data.len();
        let base = SendPtr(data.as_mut_ptr());
        self.parallel_for(n, min_chunk, |r| {
            // SAFETY: `parallel_for` passes pairwise disjoint ranges inside 0..n, and `data` is
            // exclusively borrowed for the whole call.
            let s = unsafe { std::slice::from_raw_parts_mut(base.get().add(r.start), r.len()) };
            f(r.start, s);
        });
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.quit.store(true, Ordering::Release);
        lock(&self.shared.slot).shutdown = true;
        self.shared.cv.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// Process-wide pool, sized by `COLIBRI_THREADS` or [`crate::hw::quick_threads`].
pub fn global() -> &'static Pool {
    static P: OnceLock<Pool> = OnceLock::new();
    P.get_or_init(|| Pool::new(crate::hw::quick_threads()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn parallel_sum_equals_serial_and_chunks_tile_the_range() {
        let pool = Pool::new(4);
        for n in [1usize, 2, 7, 100, 1000, 12_345] {
            let total = AtomicUsize::new(0);
            let covered = Mutex::new(vec![0u8; n]);
            pool.parallel_for(n, 3, |r| {
                total.fetch_add(r.len(), Ordering::Relaxed);
                let mut c = covered.lock().unwrap();
                for i in r {
                    c[i] += 1;
                }
            });
            assert_eq!(total.load(Ordering::Relaxed), n);
            assert!(covered.lock().unwrap().iter().all(|&c| c == 1), "n={n}");
        }
        pool.parallel_for(0, 1, |_| panic!("must not be called"));
    }

    #[test]
    fn chunks_mut_writes_every_element_once() {
        let pool = Pool::new(3);
        let mut v = vec![0u64; 100_003];
        pool.parallel_chunks_mut(&mut v, 64, |start, s| {
            for (i, x) in s.iter_mut().enumerate() {
                *x = (start + i) as u64 * 2;
            }
        });
        assert!(v.iter().enumerate().all(|(i, &x)| x == i as u64 * 2));
        let serial: u64 = (0..100_003u64).map(|i| i * 2).sum();
        assert_eq!(v.iter().sum::<u64>(), serial);
    }

    #[test]
    fn nested_calls_run_inline_without_deadlock() {
        let pool = Pool::new(4);
        let total = AtomicUsize::new(0);
        pool.parallel_for(64, 1, |outer| {
            for _ in outer {
                pool.parallel_for(100, 1, |inner| {
                    total.fetch_add(inner.len(), Ordering::Relaxed);
                });
            }
        });
        assert_eq!(total.load(Ordering::Relaxed), 64 * 100);
    }

    #[test]
    fn a_panic_reaches_the_caller_and_the_pool_survives() {
        let pool = Pool::new(4);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.parallel_for(1000, 1, |r| {
                if r.contains(&500) {
                    panic!("boom");
                }
            })
        }));
        let msg = r.expect_err("panic must propagate");
        assert_eq!(msg.downcast_ref::<&str>(), Some(&"boom"));
        let n = AtomicUsize::new(0);
        pool.parallel_for(1000, 1, |r| {
            n.fetch_add(r.len(), Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn single_thread_pool_runs_inline_in_one_piece() {
        let pool = Pool::new(1);
        assert_eq!(pool.threads(), 1);
        let calls = Mutex::new(vec![]);
        pool.parallel_for(10, 1, |r| calls.lock().unwrap().push((r.start, r.end)));
        assert_eq!(*calls.lock().unwrap(), vec![(0, 10)]);
        assert_eq!(Pool::new(0).threads(), 1);
    }

    #[test]
    fn huge_ranges_do_not_overflow() {
        let pool = Pool::new(8);
        let n = usize::MAX - 5;
        let total = AtomicUsize::new(0);
        let calls = AtomicUsize::new(0);
        pool.parallel_for(n, 1, |r| {
            assert!(r.start < r.end && r.end <= n);
            total.fetch_add(r.len(), Ordering::Relaxed);
            calls.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), n);
        assert!(calls.load(Ordering::Relaxed) <= 8 * 4 + 1);
    }

    #[test]
    fn threads_are_reused_not_spawned_per_call() {
        let pool = Pool::new(4);
        let ids = Mutex::new(HashSet::new());
        for _ in 0..300 {
            pool.parallel_for(256, 1, |_| {
                ids.lock().unwrap().insert(std::thread::current().id());
            });
        }
        assert!(
            ids.lock().unwrap().len() <= pool.threads(),
            "{} ids",
            ids.lock().unwrap().len()
        );
    }

    #[test]
    fn concurrent_callers_are_all_served_correctly() {
        let pool = Arc::new(Pool::new(4));
        let hs: Vec<_> = (0..4)
            .map(|t| {
                let p = pool.clone();
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        let s = AtomicU64::new(0);
                        p.parallel_for(1000, 8, |r| {
                            s.fetch_add(r.map(|i| i as u64 + t).sum::<u64>(), Ordering::Relaxed);
                        });
                        assert_eq!(s.load(Ordering::Relaxed), 499_500 + 1000 * t);
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
    }

    #[test]
    fn dropping_a_pool_joins_its_workers() {
        let t = std::time::Instant::now();
        for _ in 0..20 {
            let p = Pool::new(6);
            p.parallel_for(1000, 1, |_| {});
        }
        assert!(t.elapsed().as_secs() < 10);
    }

    #[test]
    fn the_global_pool_exists_and_works() {
        let n = AtomicUsize::new(0);
        global().parallel_for(5000, 16, |r| {
            n.fetch_add(r.len(), Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 5000);
    }

    #[test]
    #[ignore = "microbenchmark: cargo test -p companion-engine --release pool::tests::bench -- --ignored --nocapture"]
    fn bench_dispatch_overhead() {
        let pool = Pool::new(std::thread::available_parallelism().map_or(4, |n| n.get()));
        let mut v = vec![1.0f32; 1 << 20];
        let t = std::time::Instant::now();
        let iters = 2000;
        for _ in 0..iters {
            pool.parallel_chunks_mut(&mut v, 4096, |_, s| {
                for x in s {
                    *x = *x * 1.0000001 + 1e-9;
                }
            });
        }
        let per = t.elapsed().as_secs_f64() / iters as f64;
        println!(
            "{} threads: {:.1} us per 1M-element pass",
            pool.threads(),
            per * 1e6
        );
        let t = std::time::Instant::now();
        for _ in 0..20_000 {
            pool.parallel_for(pool.threads() * 4, 1, |_| {});
        }
        println!(
            "empty dispatch: {:.2} us",
            t.elapsed().as_secs_f64() / 20_000.0 * 1e6
        );
    }
}
