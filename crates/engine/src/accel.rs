//! Big matrices of a model on a [`Backend`] (a GPU), with the fall-backs that make that safe.
//!
//! A model keeps its host weights and asks [`Accel::try_matvec`] first; when that says no — the
//! matrix was not uploaded, the device failed, or an answer was not a number — it multiplies on
//! the CPU. After the first device error the device is left alone for good. Before anything is
//! used, one product is compared with the CPU's, so a device that computes wrongly is refused.
use crate::backend::{Backend, BackendError, Buf, WeightHandle, WeightRef};
use crate::model::Weight;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// A matrix to offer to the device: `cols` is its row length.
pub struct Item<'a> {
    pub key: usize,
    pub w: &'a Weight,
    pub cols: usize,
    pub head: bool,
}
impl<'a> Item<'a> {
    pub fn new(w: &'a Weight, cols: usize, head: bool) -> Item<'a> {
        Item {
            key: key_of(w),
            w,
            cols,
            head,
        }
    }
}

/// Matrices are found by the address of the host weight they came from (stable while the model's
/// layer vector is not changed). The output head is found by its flag instead, because it lives
/// inline in the model, which may move.
pub fn key_of(w: &Weight) -> usize {
    w as *const Weight as usize
}

/// What attaching did, in words a person can read.
#[derive(Clone, Debug, Default)]
pub struct AttachReport {
    pub matrices: usize,
    pub total: usize,
    pub bytes: u64,
    pub active: bool,
    pub note: String,
}

pub struct Accel {
    be: Arc<dyn Backend>,
    handles: HashMap<usize, WeightHandle>,
    head: Option<WeightHandle>,
    bufs: Mutex<HashMap<(bool, usize), Buf>>,
    off: AtomicBool,
    why: Mutex<Option<String>>,
}

impl Accel {
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

    fn fail(&self, e: &BackendError) {
        self.off.store(true, Ordering::Relaxed);
        if let Ok(mut w) = self.why.lock() {
            w.get_or_insert_with(|| e.to_string());
        }
    }

    /// Multiply on the device if this matrix lives there and the device still works. `true` =
    /// `out` is filled; `false` = the caller must do it on the CPU.
    pub fn try_matvec(
        &self,
        key: usize,
        head: bool,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> bool {
        if self.off.load(Ordering::Relaxed) {
            return false;
        }
        let h = if head {
            self.head.as_ref()
        } else {
            self.handles.get(&key)
        };
        let Some(h) = h else { return false };
        match self.run(h, cols, x, out) {
            Ok(()) if out.iter().all(|v| v.is_finite()) => true,
            Ok(()) => {
                self.fail(&BackendError::Device("non-finite result".into()));
                false
            }
            Err(e) => {
                self.fail(&e);
                false
            }
        }
    }

    /// Why the device was switched off while running, if it was.
    pub fn failure(&self) -> Option<String> {
        self.why.lock().ok()?.clone()
    }

    /// Upload as many of `items` as the device holds, after checking the first against the CPU.
    /// Never fails: the report says what happened, and `None` means "use the CPU for everything".
    pub fn build(be: Arc<dyn Backend>, items: &[Item]) -> (Option<Accel>, AttachReport) {
        let mut rep = AttachReport {
            total: items.len(),
            ..Default::default()
        };
        let mut free = be.device_memory();
        let mut acc = Accel {
            be: be.clone(),
            handles: HashMap::new(),
            head: None,
            bufs: Mutex::default(),
            off: AtomicBool::new(false),
            why: Mutex::default(),
        };
        let mut checked = false;
        for it in items {
            let (rows, cols) = match it.w {
                Weight::Dense(v) => (v.len().checked_div(it.cols).unwrap_or(0), it.cols),
                Weight::Quant(q) => (q.rows, q.cols),
            };
            let r = match it.w {
                Weight::Dense(v) if cols > 0 && rows * cols == v.len() => {
                    WeightRef::dense(rows, cols, v)
                }
                Weight::Quant(q) if be.supports(q.dtype) => {
                    WeightRef::quant(q.dtype, q.rows, q.cols, q.bytes.as_slice())
                }
                _ => continue,
            };
            let need = r.validate().unwrap_or(u64::MAX);
            if let Some(f) = free
                && need > f
            {
                rep.note = format!("device memory is full after {} matrices", rep.matrices);
                break;
            }
            let hd = match be.upload(r) {
                Ok(x) => x,
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
                let ok = acc.run(&hd, hd.cols, &x, &mut a).is_ok()
                    && it.w.matvec(hd.cols, &x, &mut b).is_ok();
                let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
                let err = a
                    .iter()
                    .zip(&b)
                    .fold(0f32, |m, (p, q)| m.max((p - q).abs()));
                if !ok || err > 0.02 * scale {
                    rep.note = format!(
                        "the device gave a different answer from the CPU in the start-up check (error {err:.3e}); staying on the CPU"
                    );
                    return (None, rep);
                }
            }
            if let Some(f) = free.as_mut() {
                *f = f.saturating_sub(need);
            }
            rep.matrices += 1;
            rep.bytes += need;
            if it.head {
                acc.head = Some(hd);
            } else {
                acc.handles.insert(it.key, hd);
            }
        }
        rep.active = rep.matrices > 0;
        if rep.active {
            if rep.note.is_empty() && rep.matrices < rep.total {
                rep.note = "some matrices have no kernel on this device and stay on the CPU".into();
            }
            (Some(acc), rep)
        } else {
            if rep.note.is_empty() {
                rep.note = "no matrix could be placed on the device; staying on the CPU".into();
            }
            (None, rep)
        }
    }
}

#[cfg(test)]
pub mod testbe {
    use crate::backend::*;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    /// A backend that is the CPU one, except for what the test breaks.
    pub struct Odd {
        pub inner: crate::backend::CpuBackend,
        /// matvec calls that work before it starts to fail (`usize::MAX` = never)
        ok_calls: std::sync::atomic::AtomicUsize,
        /// multiply every result by this
        scale: f32,
        memory: Option<u64>,
    }
    impl Odd {
        pub fn new(ok_calls: usize, scale: f32, memory: Option<u64>) -> Arc<Odd> {
            Arc::new(Odd {
                inner: crate::backend::CpuBackend::new(),
                ok_calls: std::sync::atomic::AtomicUsize::new(ok_calls),
                scale,
                memory,
            })
        }
    }
    type R<T> = crate::backend::Result<T>;
    impl Backend for Odd {
        fn name(&self) -> String {
            "odd".into()
        }
        fn is_accelerator(&self) -> bool {
            false
        }
        fn device_memory(&self) -> Option<u64> {
            self.memory
        }
        fn supports(&self, t: crate::quant::DType) -> bool {
            self.inner.supports(t)
        }
        fn alloc(&self, len: usize) -> R<Buf> {
            self.inner.alloc(len)
        }
        fn write(&self, d: &Buf, s: &[f32]) -> R<()> {
            self.inner.write(d, s)
        }
        fn read(&self, s: &Buf, d: &mut [f32]) -> R<()> {
            self.inner.read(s, d)?;
            d.iter_mut().for_each(|v| *v *= self.scale);
            Ok(())
        }
        fn upload(&self, w: WeightRef<'_>) -> R<WeightHandle> {
            self.inner.upload(w)
        }
        fn matvec(&self, w: &WeightHandle, x: &Buf, o: &Buf) -> R<()> {
            let left = self.ok_calls.load(Ordering::Relaxed);
            if left == 0 {
                return Err(BackendError::Device("device lost".into()));
            }
            if left != usize::MAX {
                self.ok_calls.store(left - 1, Ordering::Relaxed);
            }
            self.inner.matvec(w, x, o)
        }
        fn matmat(&self, w: &WeightHandle, x: &Buf, b: usize, o: &Buf) -> R<()> {
            self.inner.matmat(w, x, b, o)
        }
        fn rmsnorm(&self, x: &Buf, w: &Buf, e: f32, o: &Buf) -> R<()> {
            self.inner.rmsnorm(x, w, e, o)
        }
        fn rope(&self, x: &Buf, n: usize, hd: usize, p: usize, t: f64) -> R<()> {
            self.inner.rope(x, n, hd, p, t)
        }
        fn silu_mul(&self, g: &Buf, u: &Buf) -> R<()> {
            self.inner.silu_mul(g, u)
        }
        fn add(&self, a: &Buf, b: &Buf) -> R<()> {
            self.inner.add(a, b)
        }
        fn kv_create(&self, k: usize, h: usize, c: usize) -> R<crate::backend::Kv> {
            self.inner.kv_create(k, h, c)
        }
        fn kv_append(&self, kv: &crate::backend::Kv, p: usize, k: &Buf, v: &Buf) -> R<()> {
            self.inner.kv_append(kv, p, k, v)
        }
        fn attention(
            &self,
            kv: &crate::backend::Kv,
            q: &Buf,
            o: &Buf,
            a: &crate::backend::AttnArgs,
        ) -> R<()> {
            self.inner.attention(kv, q, o, a)
        }
    }
}
