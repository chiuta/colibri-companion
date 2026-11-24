//! [`EmuBackend`]: the device backend with the GPU kernels *emulated in Rust*.
//!
//! It follows [`crate::gpu::GpuBackend`] step for step — the same repacked word layouts, the same
//! row chunking against a binding limit, the same VRAM budget with CPU fallback for what does not
//! fit, the same kernels' blocking and reduction order (from [`crate::emu`]) — but executes on the
//! host. That makes the whole offload path testable with no GPU: `Model::with_backend` can run a
//! real model through it and the logits can be compared with the CPU path.
//!
//! It is slow (it exists for checking, not for speed) and proves the *logic* of the device runner,
//! not the shaders' behaviour on a driver.
use crate::layout::{self, Format};
use crate::{emu, plan};
use companion_engine::backend::*;
use companion_engine::quant::DType;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

struct EBuf(Mutex<Vec<f32>>);
struct EWeight {
    fmt: Format,
    cols: usize,
    row_words: usize,
    chunks: Vec<(Vec<u32>, usize, usize)>,
}
struct EKv {
    k: Mutex<Vec<f32>>,
    v: Mutex<Vec<f32>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
fn shape<T>(msg: impl Into<String>) -> Result<T> {
    Err(BackendError::Shape(msg.into()))
}

pub struct EmuBackend {
    budget: u64,
    used: AtomicU64,
    max_binding: u64,
    cpu: CpuBackend,
    /// Weights that stayed on the CPU backend because the budget or a binding was too small.
    host_weights: AtomicU64,
}

impl EmuBackend {
    /// A stand-in for a device with `budget` bytes of memory and storage bindings of at most
    /// `max_binding` bytes.
    pub fn new(budget: u64, max_binding: u64) -> EmuBackend {
        EmuBackend {
            budget,
            used: AtomicU64::new(0),
            max_binding,
            cpu: CpuBackend::new(),
            host_weights: AtomicU64::new(0),
        }
    }
    /// How many uploads stayed on the host.
    pub fn host_weights(&self) -> u64 {
        self.host_weights.load(Ordering::Relaxed)
    }
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }
    fn reserve(&self, bytes: u64) -> Result<()> {
        let used = self.used.load(Ordering::Relaxed);
        if used.saturating_add(bytes) > self.budget {
            return Err(BackendError::OutOfMemory {
                needed: bytes,
                available: self.budget.saturating_sub(used),
            });
        }
        self.used.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }
    fn buf(b: &Buf) -> Result<&EBuf> {
        b.inner::<EBuf>()
            .ok_or_else(|| BackendError::Device("buffer belongs to another backend".into()))
    }
    fn host_matmat(&self, w: &WeightHandle, x: &Buf, batch: usize, out: &Buf) -> Result<()> {
        let xs = lock(&Self::buf(x)?.0).clone();
        let (cx, co) = (self.cpu.alloc_from(&xs)?, self.cpu.alloc(out.len)?);
        self.cpu.matmat(w, &cx, batch, &co)?;
        let mut ys = vec![0.0f32; out.len];
        self.cpu.read(&co, &mut ys)?;
        self.write(out, &ys)
    }
}

impl Backend for EmuBackend {
    fn name(&self) -> String {
        "gpu-emulation".into()
    }
    fn is_accelerator(&self) -> bool {
        true
    }
    fn device_memory(&self) -> Option<u64> {
        Some(self.budget.saturating_sub(self.used()))
    }
    fn supports(&self, t: DType) -> bool {
        Format::from_dtype(t).is_some()
    }

    fn alloc(&self, len: usize) -> Result<Buf> {
        Ok(Buf::new(len, Arc::new(EBuf(Mutex::new(vec![0.0; len])))))
    }
    fn write(&self, dst: &Buf, src: &[f32]) -> Result<()> {
        if dst.len != src.len() {
            return shape(format!(
                "write of {} values into a buffer of {}",
                src.len(),
                dst.len
            ));
        }
        lock(&Self::buf(dst)?.0).copy_from_slice(src);
        Ok(())
    }
    fn read(&self, src: &Buf, dst: &mut [f32]) -> Result<()> {
        if src.len != dst.len() {
            return shape(format!(
                "read of {} values from a buffer of {}",
                dst.len(),
                src.len
            ));
        }
        dst.copy_from_slice(&lock(&Self::buf(src)?.0));
        Ok(())
    }

    fn upload(&self, w: WeightRef<'_>) -> Result<WeightHandle> {
        let stored = w.validate()?;
        let Some(fmt) = Format::from_dtype(w.dtype).filter(|f| f.accepts_cols(w.cols)) else {
            self.host_weights.fetch_add(1, Ordering::Relaxed);
            return self.cpu.upload(w);
        };
        let row_words = fmt.row_words(w.cols);
        let (raw, dt): (Vec<u8>, DType) = match &w.data {
            WeightData::Dense(d) => (d.iter().flat_map(|f| f.to_le_bytes()).collect(), DType::F32),
            WeightData::Quant(b) => (b.to_vec(), w.dtype),
        };
        let device_bytes = (w.rows * row_words * 4) as u64;
        if self.reserve(device_bytes).is_err() {
            self.host_weights.fetch_add(1, Ordering::Relaxed);
            return self.cpu.upload(w);
        }
        let Some(ranges) = plan::chunk_rows(w.rows, row_words as u64 * 4, self.max_binding) else {
            self.used.fetch_sub(device_bytes, Ordering::Relaxed);
            self.host_weights.fetch_add(1, Ordering::Relaxed);
            return self.cpu.upload(w);
        };
        let words = layout::repack(dt, w.rows, w.cols, &raw)
            .map_err(|e| BackendError::Unsupported(e.to_string()))?;
        let chunks = ranges
            .into_iter()
            .map(|(s, n)| (words[s * row_words..(s + n) * row_words].to_vec(), s, n))
            .collect();
        Ok(WeightHandle::new(
            w.dtype,
            w.rows,
            w.cols,
            stored,
            true,
            Arc::new(EWeight {
                fmt,
                cols: w.cols,
                row_words,
                chunks,
            }),
        ))
    }

    fn matvec(&self, w: &WeightHandle, x: &Buf, out: &Buf) -> Result<()> {
        self.matmat(w, x, 1, out)
    }
    fn matmat(&self, w: &WeightHandle, x: &Buf, batch: usize, out: &Buf) -> Result<()> {
        if x.id == out.id {
            return shape("matmat input and output are the same buffer");
        }
        if batch == 0
            || batch > plan::MAX_DIM as usize
            || x.len != batch * w.cols
            || out.len != batch * w.rows
        {
            return shape(format!(
                "matmat {}x{} with x {} / out {} for batch {batch}",
                w.rows, w.cols, x.len, out.len
            ));
        }
        let Some(ew) = w.inner::<EWeight>() else {
            return self.host_matmat(w, x, batch, out);
        };
        let xs = lock(&Self::buf(x)?.0).clone();
        let mut o = lock(&Self::buf(out)?.0);
        for b in 0..batch {
            let xv = &xs[b * ew.cols..(b + 1) * ew.cols];
            for (words, start, n) in &ew.chunks {
                debug_assert_eq!(words.len(), n * ew.row_words);
                let y = emu::matvec(ew.fmt, words, *n, ew.cols, xv);
                o[b * w.rows + start..b * w.rows + start + n].copy_from_slice(&y);
            }
        }
        Ok(())
    }
    fn rmsnorm(&self, x: &Buf, weight: &Buf, eps: f32, out: &Buf) -> Result<()> {
        if x.len != out.len || weight.len != x.len || x.id == out.id {
            return shape("rmsnorm buffers");
        }
        let (xs, ws) = (
            lock(&Self::buf(x)?.0).clone(),
            lock(&Self::buf(weight)?.0).clone(),
        );
        lock(&Self::buf(out)?.0).copy_from_slice(&emu::rmsnorm(&xs, &ws, eps));
        Ok(())
    }
    fn rope(&self, x: &Buf, n_heads: usize, head_dim: usize, pos: usize, theta: f64) -> Result<()> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) || x.len != n_heads * head_dim {
            return shape("rope dimensions");
        }
        let (c, s) = plan::rope_tables(pos, head_dim, theta);
        emu::rope(&mut lock(&Self::buf(x)?.0), head_dim, &c, &s);
        Ok(())
    }
    fn silu_mul(&self, g: &Buf, u: &Buf) -> Result<()> {
        if g.len != u.len || g.id == u.id {
            return shape("silu_mul lengths");
        }
        let us = lock(&Self::buf(u)?.0).clone();
        emu::silu_mul(&mut lock(&Self::buf(g)?.0), &us);
        Ok(())
    }
    fn add(&self, a: &Buf, b: &Buf) -> Result<()> {
        if a.len != b.len || a.id == b.id {
            return shape("add lengths");
        }
        let bs = lock(&Self::buf(b)?.0).clone();
        emu::add(&mut lock(&Self::buf(a)?.0), &bs);
        Ok(())
    }

    fn kv_create(&self, kv_heads: usize, head_dim: usize, capacity: usize) -> Result<Kv> {
        let bytes = kv_heads
            .checked_mul(head_dim)
            .and_then(|n| n.checked_mul(capacity))
            .and_then(|n| n.checked_mul(4))
            .filter(|b| *b > 0 && (*b as u64) <= self.max_binding)
            .ok_or_else(|| BackendError::Shape("KV cache size".into()))? as u64;
        self.reserve(2 * bytes)?;
        let n = (bytes / 4) as usize;
        Ok(Kv::new(
            kv_heads,
            head_dim,
            capacity,
            Arc::new(EKv {
                k: Mutex::new(vec![0.0; n]),
                v: Mutex::new(vec![0.0; n]),
            }),
        ))
    }
    fn kv_append(&self, kv: &Kv, pos: usize, k: &Buf, v: &Buf) -> Result<()> {
        let row = kv.kv_heads * kv.head_dim;
        if pos >= kv.capacity || k.len != row || v.len != row {
            return shape("kv_append");
        }
        let c = kv
            .inner::<EKv>()
            .ok_or_else(|| BackendError::Device("cache belongs to another backend".into()))?;
        lock(&c.k)[pos * row..(pos + 1) * row].copy_from_slice(&lock(&Self::buf(k)?.0));
        lock(&c.v)[pos * row..(pos + 1) * row].copy_from_slice(&lock(&Self::buf(v)?.0));
        Ok(())
    }
    fn attention(&self, kv: &Kv, q: &Buf, out: &Buf, a: &AttnArgs) -> Result<()> {
        let AttnArgs {
            n_heads,
            n_pos,
            scale,
            window,
        } = *a;
        let hd = kv.head_dim;
        if n_heads == 0
            || n_heads > plan::MAX_DIM as usize
            || !n_heads.is_multiple_of(kv.kv_heads)
            || q.len != n_heads * hd
            || out.len != q.len
            || n_pos > kv.capacity
            || q.id == out.id
        {
            return shape("attention dimensions");
        }
        let start = window.map_or(0, |w| n_pos.saturating_sub(w.max(1)));
        if n_pos == 0 || start >= n_pos {
            return self.write(out, &vec![0.0; out.len]);
        }
        let c = kv
            .inner::<EKv>()
            .ok_or_else(|| BackendError::Device("cache belongs to another backend".into()))?;
        let qs = lock(&Self::buf(q)?.0).clone();
        let res = emu::attention(
            &qs,
            &lock(&c.k),
            &lock(&c.v),
            n_heads,
            kv.kv_heads,
            hd,
            n_pos,
            start,
            scale,
        );
        lock(&Self::buf(out)?.0).copy_from_slice(&res);
        Ok(())
    }
}
