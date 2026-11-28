//! GPU for the models that keep their weights as plain `Vec<f32>` (Mamba, RWKV, the state-space
//! half of the hybrids), whose products go through [`crate::ops::matvec`].
//!
//! A model registers its big matrices here; `ops::matvec` asks [`try_matvec`] first and only does
//! the product itself when the answer is "no" (matrix not registered, device failed, result not a
//! number). A [`Registration`] removes its matrices when it is dropped — declare it **before** the
//! weights in the model struct so it is dropped first and a freed address is never mistaken for a
//! registered one. After the first device error the device is left alone for good.
use crate::accel::AttachReport;
use crate::backend::{Backend, BackendError, Buf, WeightHandle, WeightRef};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Matrices smaller than this are not worth a trip to the device.
#[cfg(not(test))]
const MIN_LEN: usize = 1 << 14;
#[cfg(test)]
const MIN_LEN: usize = 16;

struct Dev {
    be: Arc<dyn Backend>,
    mats: HashMap<(usize, usize), WeightHandle>,
    bufs: Mutex<HashMap<(bool, usize), Buf>>,
    why: Mutex<Option<String>>,
}

static ANY: AtomicBool = AtomicBool::new(false);
static OFF: AtomicBool = AtomicBool::new(false);
static DEV: RwLock<Option<Dev>> = RwLock::new(None);

/// Keeps a model's matrices registered; dropping it unregisters them.
#[derive(Default)]
pub struct Registration(Vec<(usize, usize)>);

// A copy of a model has its weights at new addresses, so it starts with nothing registered.
impl Clone for Registration {
    fn clone(&self) -> Self {
        Registration::default()
    }
}
impl std::fmt::Debug for Registration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Registration({} matrices)", self.0.len())
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        if self.0.is_empty() {
            return;
        }
        if let Ok(mut g) = DEV.write()
            && let Some(d) = g.as_mut()
        {
            for k in &self.0 {
                d.mats.remove(k);
            }
            if d.mats.is_empty() {
                *g = None;
                ANY.store(false, Ordering::Release);
            }
        }
    }
}

impl Dev {
    /// `out[t] = W · x[t]` for `batch` vectors (one product when `batch` is 1).
    fn run(
        &self,
        h: &WeightHandle,
        cols: usize,
        batch: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let mut bufs = self
            .bufs
            .lock()
            .map_err(|_| BackendError::Device("lock".into()))?;
        for (is_out, len) in [(false, x.len()), (true, out.len())] {
            if let std::collections::hash_map::Entry::Vacant(v) = bufs.entry((is_out, len)) {
                v.insert(self.be.alloc(len)?);
            }
        }
        let (bx, bo) = (&bufs[&(false, x.len())], &bufs[&(true, out.len())]);
        if h.cols != cols || h.rows * batch != out.len() || x.len() != cols * batch {
            return Err(BackendError::Shape(format!(
                "{}x{} vs {cols} (batch {batch})",
                h.rows, h.cols
            )));
        }
        self.be.write(bx, x)?;
        if batch == 1 {
            self.be.matvec(h, bx, bo)?;
        } else {
            self.be.matmat(h, bx, batch, bo)?;
        }
        self.be.read(bo, out)
    }
}

/// `out = W · x` on the device when `w` is registered and the device works. `false` = do it on the CPU.
pub fn try_matvec(w: &[f32], cols: usize, x: &[f32], out: &mut [f32]) -> bool {
    if !ANY.load(Ordering::Acquire) || OFF.load(Ordering::Relaxed) || w.len() < MIN_LEN {
        return false;
    }
    let Ok(g) = DEV.read() else { return false };
    let Some(d) = g.as_ref() else { return false };
    let Some(h) = d.mats.get(&(w.as_ptr() as usize, w.len())) else {
        return false;
    };
    guard(d, d.run(h, cols, 1, x, out), out)
}

fn guard(d: &Dev, r: Result<(), BackendError>, out: &[f32]) -> bool {
    match r {
        Ok(()) if out.iter().all(|v| v.is_finite()) => true,
        r => {
            let why = match r {
                Err(e) => e.to_string(),
                Ok(()) => "non-finite result".into(),
            };
            OFF.store(true, Ordering::Relaxed);
            if let Ok(mut w) = d.why.lock() {
                w.get_or_insert(why);
            }
            false
        }
    }
}

/// Most vectors sent to the device in one batched product.
const BATCH: usize = 128;

/// `out[t] = W · x[t] (+ bias)` for the `n` rows of `x`, on the device when `w` is registered and
/// the device works (batched products, so a long sequence is not one round trip per token).
/// `false` = nothing usable was written: do the whole product on the CPU.
pub fn try_rows(
    w: &[f32],
    cols: usize,
    x: &[f32],
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> bool {
    if !ANY.load(Ordering::Acquire)
        || OFF.load(Ordering::Relaxed)
        || w.len() < MIN_LEN
        || cols == 0
        || n == 0
    {
        return false;
    }
    let rows = w.len() / cols;
    if x.len() != n * cols || out.len() != n * rows || bias.is_some_and(|b| b.len() != rows) {
        return false;
    }
    let Ok(g) = DEV.read() else { return false };
    let Some(d) = g.as_ref() else { return false };
    let Some(h) = d.mats.get(&(w.as_ptr() as usize, w.len())) else {
        return false;
    };
    for (xs, os) in x.chunks(BATCH * cols).zip(out.chunks_mut(BATCH * rows)) {
        let b = xs.len() / cols;
        if !guard(d, d.run(h, cols, b, xs, os), os) {
            return false;
        }
    }
    if let Some(b) = bias {
        for row in out.chunks_exact_mut(rows) {
            row.iter_mut().zip(b).for_each(|(o, b)| *o += b);
        }
    }
    true
}

/// `true` while some matrix is registered and the device has not failed: a cheap pre-check so
/// callers do not allocate for a device path that cannot be taken.
pub fn active() -> bool {
    ANY.load(Ordering::Acquire) && !OFF.load(Ordering::Relaxed)
}

/// Why the device was switched off while running, if it was.
pub fn failure() -> Option<String> {
    let g = DEV.read().ok()?;
    g.as_ref()?.why.lock().ok()?.clone()
}

/// Upload `mats` (`(values, row length)`), as many as the device holds, after checking one product
/// against the CPU. Never fails: the report says what happened.
pub fn register(be: Arc<dyn Backend>, mats: &[(&[f32], usize)]) -> (Registration, AttachReport) {
    let mut rep = AttachReport {
        total: mats.len(),
        ..Default::default()
    };
    let mut free = be.device_memory();
    let mut mine: Vec<((usize, usize), WeightHandle)> = vec![];
    let mut checked = false;
    for (w, cols) in mats {
        if w.len() < MIN_LEN || *cols == 0 || !w.len().is_multiple_of(*cols) {
            continue;
        }
        let r = WeightRef::dense(w.len() / cols, *cols, w);
        let need = r.validate().unwrap_or(u64::MAX);
        if let Some(f) = free
            && need > f
        {
            rep.note = format!("device memory is full after {} matrices", rep.matrices);
            break;
        }
        let hd = match be.upload(r) {
            Ok(h) => h,
            Err(e) => {
                rep.note = format!("upload stopped: {e}");
                break;
            }
        };
        if !hd.on_device && be.is_accelerator() {
            continue;
        }
        if !checked {
            checked = true;
            let x: Vec<f32> = (0..hd.cols)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) / 6.0)
                .collect();
            let (mut a, mut b) = (vec![0.0; hd.rows], vec![0.0; hd.rows]);
            let probe = Dev {
                be: be.clone(),
                mats: HashMap::new(),
                bufs: Mutex::default(),
                why: Mutex::default(),
            };
            let ok = probe.run(&hd, hd.cols, 1, &x, &mut a).is_ok();
            for (o, row) in b.iter_mut().zip(w.chunks_exact(*cols)) {
                *o = row.iter().zip(&x).map(|(p, q)| p * q).sum();
            }
            let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            let err = a
                .iter()
                .zip(&b)
                .fold(0f32, |m, (p, q)| m.max((p - q).abs()));
            if !ok || err > 0.02 * scale {
                rep.note = format!(
                    "the device gave a different answer from the CPU in the start-up check (error {err:.3e}); staying on the CPU"
                );
                return (Registration::default(), rep);
            }
        }
        if let Some(f) = free.as_mut() {
            *f = f.saturating_sub(need);
        }
        rep.matrices += 1;
        rep.bytes += need;
        mine.push(((w.as_ptr() as usize, w.len()), hd));
    }
    rep.active = !mine.is_empty();
    if !rep.active {
        if rep.note.is_empty() {
            rep.note = "no matrix could be placed on the device; staying on the CPU".into();
        }
        return (Registration::default(), rep);
    }
    if rep.matrices < rep.total && rep.note.is_empty() {
        rep.note = "small matrices stay on the CPU".into();
    }
    let keys: Vec<(usize, usize)> = mine.iter().map(|m| m.0).collect();
    if let Ok(mut g) = DEV.write() {
        let d = g.get_or_insert_with(|| Dev {
            be: be.clone(),
            mats: HashMap::new(),
            bufs: Mutex::default(),
            why: Mutex::default(),
        });
        // one device per process: a different backend instance would need its own buffers
        if !Arc::ptr_eq(&d.be, &be) && !d.mats.is_empty() {
            rep.active = false;
            rep.note = "another device is already in use for state-space models".into();
            return (Registration::default(), rep);
        }
        d.mats.extend(mine);
        OFF.store(false, Ordering::Relaxed);
        ANY.store(true, Ordering::Release);
    }
    (Registration(keys), rep)
}

/// Tests that register matrices share one process-wide registry, so they take turns.
#[cfg(test)]
pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A test device that is [`Odd`](crate::accel::testbe::Odd), but whose batched products can also
/// start to fail after a number of calls.
#[cfg(test)]
pub(crate) mod testbe {
    use crate::accel::testbe::Odd;
    use crate::backend::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct Flaky {
        inner: Arc<Odd>,
        matmat_ok: AtomicUsize,
    }
    /// `matvec_ok` / `matmat_ok`: calls that work before the device starts to fail.
    pub fn flaky(matvec_ok: usize, matmat_ok: usize) -> Arc<Flaky> {
        Arc::new(Flaky {
            inner: Odd::new(matvec_ok, 1.0, None),
            matmat_ok: AtomicUsize::new(matmat_ok),
        })
    }
    impl Backend for Flaky {
        fn name(&self) -> String {
            "flaky".into()
        }
        fn is_accelerator(&self) -> bool {
            false
        }
        fn device_memory(&self) -> Option<u64> {
            None
        }
        fn supports(&self, t: crate::quant::DType) -> bool {
            self.inner.supports(t)
        }
        fn alloc(&self, len: usize) -> Result<Buf> {
            self.inner.alloc(len)
        }
        fn write(&self, d: &Buf, s: &[f32]) -> Result<()> {
            self.inner.write(d, s)
        }
        fn read(&self, s: &Buf, d: &mut [f32]) -> Result<()> {
            self.inner.read(s, d)
        }
        fn upload(&self, w: WeightRef<'_>) -> Result<WeightHandle> {
            self.inner.upload(w)
        }
        fn matvec(&self, w: &WeightHandle, x: &Buf, o: &Buf) -> Result<()> {
            self.inner.matvec(w, x, o)
        }
        fn matmat(&self, w: &WeightHandle, x: &Buf, b: usize, o: &Buf) -> Result<()> {
            let left = self.matmat_ok.load(Ordering::Relaxed);
            if left == 0 {
                return Err(BackendError::Device("device lost".into()));
            }
            if left != usize::MAX {
                self.matmat_ok.store(left - 1, Ordering::Relaxed);
            }
            self.inner.matmat(w, x, b, o)
        }
        fn rmsnorm(&self, x: &Buf, w: &Buf, e: f32, o: &Buf) -> Result<()> {
            self.inner.rmsnorm(x, w, e, o)
        }
        fn rope(&self, x: &Buf, n: usize, hd: usize, p: usize, t: f64) -> Result<()> {
            self.inner.rope(x, n, hd, p, t)
        }
        fn silu_mul(&self, g: &Buf, u: &Buf) -> Result<()> {
            self.inner.silu_mul(g, u)
        }
        fn add(&self, a: &Buf, b: &Buf) -> Result<()> {
            self.inner.add(a, b)
        }
        fn kv_create(&self, k: usize, h: usize, c: usize) -> Result<Kv> {
            self.inner.kv_create(k, h, c)
        }
        fn kv_append(&self, kv: &Kv, p: usize, k: &Buf, v: &Buf) -> Result<()> {
            self.inner.kv_append(kv, p, k, v)
        }
        fn attention(&self, kv: &Kv, q: &Buf, o: &Buf, a: &AttnArgs) -> Result<()> {
            self.inner.attention(kv, q, o, a)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accel::testbe::Odd;

    fn mat(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| ((i * 31 % 17) as f32 - 8.0) / 9.0)
            .collect()
    }

    #[test]
    fn registered_matrices_match_the_cpu_and_unregister_on_drop() {
        let _g = test_lock();
        let w = mat(200, 128);
        let x: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let (reg, rep) = register(Odd::new(usize::MAX, 1.0, None), &[(&w, 128)]);
        assert!(rep.active, "{rep:?}");
        let mut a = vec![0.0; 200];
        assert!(try_matvec(&w, 128, &x, &mut a));
        let mut b = vec![0.0; 200];
        crate::ops::matvec(&w, 128, &x, &mut b);
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-4);
        }
        // another slice, even of the same size, is not registered
        let other = mat(200, 128);
        assert!(!try_matvec(&other, 128, &x, &mut a));
        drop(reg);
        assert!(!try_matvec(&w, 128, &x, &mut a));
    }

    #[test]
    fn a_device_error_falls_back_for_good_and_a_wrong_device_is_refused() {
        let _g = test_lock();
        let w = mat(200, 128);
        let x: Vec<f32> = (0..128).map(|i| (i as f32).cos()).collect();
        let (reg, rep) = register(Odd::new(2, 1.0, None), &[(&w, 128)]);
        assert!(rep.active);
        let mut out = vec![0.0; 200];
        let mut cpu = vec![0.0; 200];
        crate::ops::matvec(&w, 128, &x, &mut cpu);
        let mut seen_fail = false;
        for _ in 0..4 {
            crate::ops::matvec(&w, 128, &x, &mut out);
            for (p, q) in out.iter().zip(&cpu) {
                assert!((p - q).abs() < 1e-4);
            }
            seen_fail |= failure().is_some();
        }
        assert!(seen_fail);
        drop(reg);
        let (_r, rep) = register(Odd::new(usize::MAX, 1.5, None), &[(&w, 128)]);
        assert!(!rep.active, "{rep:?}");
    }

    #[test]
    fn batched_rows_match_the_cpu_with_bias_and_fall_back_when_the_device_fails() {
        let _g = test_lock();
        let (rows, cols, n) = (40, 32, 300);
        let w = mat(rows, cols);
        let bias: Vec<f32> = (0..rows).map(|i| i as f32 * 0.01).collect();
        let x: Vec<f32> = (0..n * cols).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut cpu = vec![0.0; n * rows];
        for (xt, ot) in x.chunks(cols).zip(cpu.chunks_mut(rows)) {
            crate::ops::matvec(&w, cols, xt, ot);
            ot.iter_mut().zip(&bias).for_each(|(o, b)| *o += b);
        }
        let mut out = vec![0.0; n * rows];
        assert!(
            !try_rows(&w, cols, &x, n, Some(&bias), &mut out),
            "not registered"
        );
        let (reg, rep) = register(testbe::flaky(usize::MAX, usize::MAX), &[(&w, cols)]);
        assert!(rep.active, "{rep:?}");
        assert!(try_rows(&w, cols, &x, n, Some(&bias), &mut out));
        for (p, q) in out.iter().zip(&cpu) {
            assert!((p - q).abs() < 1e-3);
        }
        // one row goes through the single-vector product
        assert!(try_rows(&w, cols, &x[..cols], 1, None, &mut out[..rows]));
        // wrong shapes are refused, not guessed at
        assert!(!try_rows(
            &w,
            cols,
            &x[..cols - 1],
            1,
            None,
            &mut out[..rows]
        ));
        assert!(!try_rows(
            &w,
            cols,
            &x[..cols],
            1,
            Some(&bias[..3]),
            &mut out[..rows]
        ));
        drop(reg);
        // the device dies in the third chunk of the batch: the caller is told to use the CPU
        let (reg, rep) = register(testbe::flaky(usize::MAX, 2), &[(&w, cols)]);
        assert!(rep.active);
        assert!(!try_rows(&w, cols, &x, n, Some(&bias), &mut out));
        assert!(failure().is_some());
        // and the device is left alone for good
        assert!(!try_rows(&w, cols, &x[..cols], 1, None, &mut out[..rows]));
        drop(reg);
    }
}
