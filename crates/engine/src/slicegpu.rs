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
    fn run(
        &self,
        h: &WeightHandle,
        cols: usize,
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
        if h.cols != cols || h.rows != out.len() {
            return Err(BackendError::Shape(format!(
                "{}x{} vs {cols}",
                h.rows, h.cols
            )));
        }
        self.be.write(bx, x)?;
        self.be.matvec(h, bx, bo)?;
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
    match d.run(h, cols, x, out) {
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
            let ok = probe.run(&hd, hd.cols, &x, &mut a).is_ok();
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
}
