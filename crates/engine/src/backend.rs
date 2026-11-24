//! Compute backends: one trait for the heavy per-token operations, so the same model can run on
//! the CPU kernels or on a GPU, or split between the two (see [`offload_plan`]).
//!
//! # Design
//! Everything the model needs per token is expressed on **device buffers** ([`Buf`]), not on host
//! slices. A GPU backend can then record a whole token (all layers) as one command stream and read
//! back exactly once, at [`Backend::read`]. The CPU backend implements the same calls with the
//! existing `qmatmul`/`ops`/`simd` kernels, so behaviour is identical and testable everywhere.
//!
//! # How `model.rs` adopts it (integration points, nothing here edits `model.rs`)
//! 1. **Load time** – for every `Weight` of a layer call [`Backend::upload`] once with a
//!    [`WeightRef`] (`Weight::Dense` → [`WeightRef::dense`], `Weight::Quant` → [`WeightRef::quant`]
//!    over `QWeight::bytes.as_slice()`), keep the [`WeightHandle`] next to the weight. Norm weights
//!    and biases are uploaded with [`Backend::alloc_from`]. Use [`offload_plan`] to decide which
//!    layers go to the accelerator; the rest keep using the CPU path (a second backend instance).
//! 2. **State creation** – `new_state_with` calls [`Backend::kv_create`] per layer (device-side KV).
//! 3. **`forward_batch`** – replace the per-layer calls: `rmsnorm` → [`Backend::rmsnorm`],
//!    `wq/wk/wv` → [`Backend::matvec`] / [`Backend::matmat`], per-head RoPE → [`Backend::rope`],
//!    cache write → [`Backend::kv_append`], the attention loop → [`Backend::attention`],
//!    `wo`/`w_down` → matvec, the residual adds → [`Backend::add`], the gated MLP →
//!    `matvec`×2 + [`Backend::silu_mul`] + `matvec`. Write the embedding with [`Backend::write`] and
//!    fetch the logits with one [`Backend::read`]; that read is the only synchronisation point.
//! 4. **Variants** that the trait does not cover yet (QK-norm, soft-capping, biases, GELU, sliding
//!    ring caches) stay on the CPU path for a layer that needs them; `supports()` plus a per-layer
//!    flag keeps the choice explicit.
use crate::hw::{GpuKind, HardwareProfile};
use crate::ops;
use crate::qmatmul;
use crate::quant::DType;
use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq)]
pub enum BackendError {
    Unsupported(String),
    Shape(String),
    OutOfMemory { needed: u64, available: u64 },
    Device(String),
}
impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::Unsupported(s) => write!(f, "not supported by this backend: {s}"),
            BackendError::Shape(s) => write!(f, "wrong shape: {s}"),
            BackendError::OutOfMemory { needed, available } => write!(
                f,
                "device memory: {needed} bytes needed, {available} available"
            ),
            BackendError::Device(s) => write!(f, "device error: {s}"),
        }
    }
}
impl std::error::Error for BackendError {}
pub type Result<T> = std::result::Result<T, BackendError>;

fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::Relaxed)
}
fn shape<T>(msg: impl Into<String>) -> Result<T> {
    Err(BackendError::Shape(msg.into()))
}

/// Host-side description of a weight matrix to upload.
pub enum WeightData<'a> {
    Dense(&'a [f32]),
    Quant(&'a [u8]),
}
pub struct WeightRef<'a> {
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
    pub data: WeightData<'a>,
}
impl<'a> WeightRef<'a> {
    pub fn dense(rows: usize, cols: usize, w: &'a [f32]) -> Self {
        WeightRef {
            dtype: DType::F32,
            rows,
            cols,
            data: WeightData::Dense(w),
        }
    }
    pub fn quant(dtype: DType, rows: usize, cols: usize, bytes: &'a [u8]) -> Self {
        WeightRef {
            dtype,
            rows,
            cols,
            data: WeightData::Quant(bytes),
        }
    }
    /// Check sizes and return the stored byte count.
    pub fn validate(&self) -> Result<u64> {
        if self.rows == 0 || self.cols == 0 {
            return shape("empty matrix");
        }
        match &self.data {
            WeightData::Dense(w) => {
                if self.dtype != DType::F32 {
                    return shape("dense data must have dtype F32");
                }
                if (w.len() as u64) != self.rows as u64 * self.cols as u64 {
                    return shape(format!(
                        "{} values for {}x{}",
                        w.len(),
                        self.rows,
                        self.cols
                    ));
                }
                Ok(w.len() as u64 * 4)
            }
            WeightData::Quant(b) => {
                let row = self.dtype.bytes_for(self.cols as u64).ok_or_else(|| {
                    BackendError::Unsupported(format!(
                        "{:?} with {} columns",
                        self.dtype, self.cols
                    ))
                })?;
                let total = row
                    .checked_mul(self.rows as u64)
                    .ok_or_else(|| BackendError::Shape("size overflow".into()))?;
                if b.len() as u64 != total {
                    return shape(format!(
                        "{} bytes for {}x{} {:?}, expected {total}",
                        b.len(),
                        self.rows,
                        self.cols,
                        self.dtype
                    ));
                }
                if !self.dtype.decodable() {
                    return Err(BackendError::Unsupported(format!("{:?}", self.dtype)));
                }
                Ok(total)
            }
        }
    }
}

/// A weight resident in a backend (host memory for the CPU, device memory for a GPU).
#[derive(Clone)]
pub struct WeightHandle {
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
    pub bytes: u64,
    pub on_device: bool,
    inner: Arc<dyn Any + Send + Sync>,
}
impl WeightHandle {
    pub fn new(
        dtype: DType,
        rows: usize,
        cols: usize,
        bytes: u64,
        on_device: bool,
        inner: Arc<dyn Any + Send + Sync>,
    ) -> Self {
        WeightHandle {
            dtype,
            rows,
            cols,
            bytes,
            on_device,
            inner,
        }
    }
    pub fn inner<T: 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }
}
/// A float vector living in a backend.
#[derive(Clone)]
pub struct Buf {
    pub id: u64,
    pub len: usize,
    inner: Arc<dyn Any + Send + Sync>,
}
impl Buf {
    pub fn new(len: usize, inner: Arc<dyn Any + Send + Sync>) -> Self {
        Buf {
            id: next_id(),
            len,
            inner,
        }
    }
    pub fn inner<T: 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }
}
/// A key/value cache for one layer, `capacity × kv_heads × head_dim` for keys and for values.
#[derive(Clone)]
pub struct Kv {
    pub id: u64,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub capacity: usize,
    inner: Arc<dyn Any + Send + Sync>,
}
impl Kv {
    pub fn new(
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        inner: Arc<dyn Any + Send + Sync>,
    ) -> Self {
        Kv {
            id: next_id(),
            kv_heads,
            head_dim,
            capacity,
            inner,
        }
    }
    pub fn inner<T: 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }
}

/// Per-call attention parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnArgs {
    pub n_heads: usize,
    pub n_pos: usize,
    pub scale: f32,
    pub window: Option<usize>,
}

pub trait Backend: Send + Sync {
    fn name(&self) -> String;
    /// `true` for devices with their own memory/queue (GPU); the CPU backend is not.
    fn is_accelerator(&self) -> bool;
    /// Bytes of dedicated device memory available for weights + caches; `None` = host memory.
    fn device_memory(&self) -> Option<u64>;
    fn supports(&self, t: DType) -> bool;

    fn alloc(&self, len: usize) -> Result<Buf>;
    fn write(&self, dst: &Buf, src: &[f32]) -> Result<()>;
    /// Copy a buffer back to the host. This is the synchronisation point of a token.
    fn read(&self, src: &Buf, dst: &mut [f32]) -> Result<()>;
    fn alloc_from(&self, data: &[f32]) -> Result<Buf> {
        let b = self.alloc(data.len())?;
        self.write(&b, data)?;
        Ok(b)
    }

    fn upload(&self, w: WeightRef<'_>) -> Result<WeightHandle>;
    /// `out[rows] = W · x[cols]`.
    fn matvec(&self, w: &WeightHandle, x: &Buf, out: &Buf) -> Result<()>;
    /// `out[b][rows] = W · x[b][cols]` for `batch` row-major vectors.
    fn matmat(&self, w: &WeightHandle, x: &Buf, batch: usize, out: &Buf) -> Result<()>;
    fn rmsnorm(&self, x: &Buf, weight: &Buf, eps: f32, out: &Buf) -> Result<()>;
    /// Rotate-half RoPE on `n_heads` consecutive heads of `head_dim`, in place.
    fn rope(&self, x: &Buf, n_heads: usize, head_dim: usize, pos: usize, theta: f64) -> Result<()>;
    /// `g[i] = silu(g[i]) * u[i]`.
    fn silu_mul(&self, g: &Buf, u: &Buf) -> Result<()>;
    /// `a += b`.
    fn add(&self, a: &Buf, b: &Buf) -> Result<()>;

    fn kv_create(&self, kv_heads: usize, head_dim: usize, capacity: usize) -> Result<Kv>;
    fn kv_append(&self, kv: &Kv, pos: usize, k: &Buf, v: &Buf) -> Result<()>;
    /// Softmax attention of `q` (`n_heads × head_dim`) over positions `0..n_pos` (or only the last
    /// `window` of them); grouped-query when `n_heads` is a multiple of the cache's `kv_heads`.
    fn attention(&self, kv: &Kv, q: &Buf, out: &Buf, a: &AttnArgs) -> Result<()>;
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

// ------------------------------------------------------------------------------- CPU backend

enum CpuWeight {
    Dense(Vec<f32>),
    Quant(Vec<u8>),
}
struct CpuBuf(Mutex<Vec<f32>>);
struct CpuKv {
    k: Mutex<Vec<f32>>,
    v: Mutex<Vec<f32>>,
}

/// The reference backend: the engine's own CPU kernels behind the common trait.
#[derive(Default)]
pub struct CpuBackend;
impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

fn cpu_buf(b: &Buf) -> Result<&CpuBuf> {
    b.inner::<CpuBuf>()
        .ok_or_else(|| BackendError::Device("buffer belongs to another backend".into()))
}
fn lock(b: &CpuBuf) -> Result<std::sync::MutexGuard<'_, Vec<f32>>> {
    b.0.lock()
        .map_err(|_| BackendError::Device("poisoned buffer".into()))
}
fn distinct(a: &Buf, b: &Buf) -> Result<()> {
    if a.id == b.id {
        shape("an operation was given the same buffer twice")
    } else {
        Ok(())
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> String {
        format!("cpu ({})", crate::simd::active())
    }
    fn is_accelerator(&self) -> bool {
        false
    }
    fn device_memory(&self) -> Option<u64> {
        None
    }
    fn supports(&self, t: DType) -> bool {
        t == DType::F32 || t.decodable()
    }

    fn alloc(&self, len: usize) -> Result<Buf> {
        Ok(Buf::new(len, Arc::new(CpuBuf(Mutex::new(vec![0.0; len])))))
    }
    fn write(&self, dst: &Buf, src: &[f32]) -> Result<()> {
        if dst.len != src.len() {
            return shape(format!(
                "write of {} values into a buffer of {}",
                src.len(),
                dst.len
            ));
        }
        lock(cpu_buf(dst)?)?.copy_from_slice(src);
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
        dst.copy_from_slice(&lock(cpu_buf(src)?)?);
        Ok(())
    }

    fn upload(&self, w: WeightRef<'_>) -> Result<WeightHandle> {
        let bytes = w.validate()?;
        let inner = match w.data {
            WeightData::Dense(d) => CpuWeight::Dense(d.to_vec()),
            WeightData::Quant(b) => CpuWeight::Quant(b.to_vec()),
        };
        Ok(WeightHandle::new(
            w.dtype,
            w.rows,
            w.cols,
            bytes,
            false,
            Arc::new(inner),
        ))
    }
    fn matvec(&self, w: &WeightHandle, x: &Buf, out: &Buf) -> Result<()> {
        self.matmat(w, x, 1, out)
    }
    fn matmat(&self, w: &WeightHandle, x: &Buf, batch: usize, out: &Buf) -> Result<()> {
        distinct(x, out)?;
        if x.len != batch * w.cols || out.len != batch * w.rows {
            return shape(format!(
                "matmat {}x{} with x {} / out {} for batch {batch}",
                w.rows, w.cols, x.len, out.len
            ));
        }
        let cw = w
            .inner::<CpuWeight>()
            .ok_or_else(|| BackendError::Device("weight belongs to another backend".into()))?;
        let (xg, mut og) = (lock(cpu_buf(x)?)?, lock(cpu_buf(out)?)?);
        for (xs, os) in xg.chunks_exact(w.cols).zip(og.chunks_exact_mut(w.rows)) {
            match cw {
                CpuWeight::Dense(d) => ops::matvec(d, w.cols, xs, os),
                CpuWeight::Quant(b) => qmatmul::qmatvec(w.dtype, b, w.rows, w.cols, xs, os)
                    .map_err(|e| BackendError::Unsupported(e.to_string()))?,
            }
        }
        Ok(())
    }
    fn rmsnorm(&self, x: &Buf, weight: &Buf, eps: f32, out: &Buf) -> Result<()> {
        distinct(x, out)?;
        if x.len != out.len || weight.len != x.len {
            return shape("rmsnorm lengths differ");
        }
        let (xg, wg, mut og) = (
            lock(cpu_buf(x)?)?,
            if weight.id == x.id {
                None
            } else {
                Some(lock(cpu_buf(weight)?)?)
            },
            lock(cpu_buf(out)?)?,
        );
        match &wg {
            Some(w) => ops::rmsnorm(&xg, w, eps, &mut og),
            None => {
                let w = xg.clone();
                ops::rmsnorm(&xg, &w, eps, &mut og)
            }
        }
        Ok(())
    }
    fn rope(&self, x: &Buf, n_heads: usize, head_dim: usize, pos: usize, theta: f64) -> Result<()> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) || x.len != n_heads * head_dim {
            return shape(format!(
                "rope on {} values as {n_heads} heads of {head_dim}",
                x.len
            ));
        }
        for h in lock(cpu_buf(x)?)?.chunks_exact_mut(head_dim) {
            ops::rope(h, pos, theta);
        }
        Ok(())
    }
    fn silu_mul(&self, g: &Buf, u: &Buf) -> Result<()> {
        distinct(g, u)?;
        if g.len != u.len {
            return shape("silu_mul lengths differ");
        }
        let (mut gg, ug) = (lock(cpu_buf(g)?)?, lock(cpu_buf(u)?)?);
        ops::silu_mul(&mut gg, &ug);
        Ok(())
    }
    fn add(&self, a: &Buf, b: &Buf) -> Result<()> {
        distinct(a, b)?;
        if a.len != b.len {
            return shape("add lengths differ");
        }
        let (mut ag, bg) = (lock(cpu_buf(a)?)?, lock(cpu_buf(b)?)?);
        ops::axpy(1.0, &bg, &mut ag);
        Ok(())
    }

    fn kv_create(&self, kv_heads: usize, head_dim: usize, capacity: usize) -> Result<Kv> {
        if kv_heads == 0 || head_dim == 0 || capacity == 0 {
            return shape("empty KV cache");
        }
        let n = kv_heads
            .checked_mul(head_dim)
            .and_then(|n| n.checked_mul(capacity))
            .ok_or_else(|| BackendError::Shape("KV size overflow".into()))?;
        Ok(Kv::new(
            kv_heads,
            head_dim,
            capacity,
            Arc::new(CpuKv {
                k: Mutex::new(vec![0.0; n]),
                v: Mutex::new(vec![0.0; n]),
            }),
        ))
    }
    fn kv_append(&self, kv: &Kv, pos: usize, k: &Buf, v: &Buf) -> Result<()> {
        let row = kv.kv_heads * kv.head_dim;
        if pos >= kv.capacity || k.len != row || v.len != row {
            return shape(format!(
                "kv_append at {pos} of {} with rows {}/{} (need {row})",
                kv.capacity, k.len, v.len
            ));
        }
        let c = kv
            .inner::<CpuKv>()
            .ok_or_else(|| BackendError::Device("cache belongs to another backend".into()))?;
        let (kg, vg) = (lock(cpu_buf(k)?)?, lock(cpu_buf(v)?)?);
        c.k.lock()
            .map_err(|_| BackendError::Device("poisoned".into()))?[pos * row..(pos + 1) * row]
            .copy_from_slice(&kg);
        c.v.lock()
            .map_err(|_| BackendError::Device("poisoned".into()))?[pos * row..(pos + 1) * row]
            .copy_from_slice(&vg);
        Ok(())
    }
    fn attention(&self, kv: &Kv, q: &Buf, out: &Buf, a: &AttnArgs) -> Result<()> {
        let AttnArgs {
            n_heads,
            n_pos,
            scale,
            window,
        } = *a;
        distinct(q, out)?;
        let (hd, nkv) = (kv.head_dim, kv.kv_heads);
        if n_heads == 0
            || !n_heads.is_multiple_of(nkv)
            || q.len != n_heads * hd
            || out.len != q.len
            || n_pos > kv.capacity
        {
            return shape("attention dimensions");
        }
        let c = kv
            .inner::<CpuKv>()
            .ok_or_else(|| BackendError::Device("cache belongs to another backend".into()))?;
        let (qg, mut og) = (lock(cpu_buf(q)?)?, lock(cpu_buf(out)?)?);
        og.fill(0.0);
        if n_pos == 0 {
            return Ok(());
        }
        let start = window.map_or(0, |w| n_pos.saturating_sub(w.max(1)));
        let (kc, vc) = (
            c.k.lock()
                .map_err(|_| BackendError::Device("poisoned".into()))?,
            c.v.lock()
                .map_err(|_| BackendError::Device("poisoned".into()))?,
        );
        let (row, group) = (nkv * hd, n_heads / nkv);
        let mut sc = vec![0.0f32; n_pos - start];
        for h in 0..n_heads {
            let kvh = h / group;
            let qh = &qg[h * hd..(h + 1) * hd];
            for (i, s) in sc.iter_mut().enumerate() {
                let t = start + i;
                *s = crate::simd::dot_f32(qh, &kc[t * row + kvh * hd..t * row + (kvh + 1) * hd])
                    * scale;
            }
            ops::softmax(&mut sc);
            let oh = &mut og[h * hd..(h + 1) * hd];
            for (i, p) in sc.iter().enumerate() {
                let t = start + i;
                ops::axpy(*p, &vc[t * row + kvh * hd..t * row + (kvh + 1) * hd], oh);
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------------- selector

/// What the user (or `ENGINE_BACKEND`) asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preference {
    Auto,
    Cpu,
    Gpu,
}
impl Preference {
    pub fn parse(s: &str) -> Option<Preference> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Some(Preference::Auto),
            "cpu" => Some(Preference::Cpu),
            "gpu" => Some(Preference::Gpu),
            _ => None,
        }
    }
    /// `ENGINE_BACKEND=cpu|gpu|auto`; anything else (or unset) means `Auto`.
    pub fn from_env() -> Preference {
        std::env::var("ENGINE_BACKEND")
            .ok()
            .and_then(|v| Preference::parse(&v))
            .unwrap_or(Preference::Auto)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub gpu: bool,
    pub reason: String,
}

/// Smallest dedicated VRAM for which automatic selection prefers the GPU.
pub const MIN_AUTO_VRAM: u64 = 2 << 30;

/// Pure decision from a hardware profile; no device is touched.
pub fn choose(profile: &HardwareProfile, pref: Preference) -> Choice {
    match pref {
        Preference::Cpu => Choice {
            gpu: false,
            reason: "ENGINE_BACKEND=cpu".into(),
        },
        Preference::Gpu => Choice {
            gpu: true,
            reason: "ENGINE_BACKEND=gpu".into(),
        },
        Preference::Auto => {
            let best = profile.gpus.iter().max_by_key(|g| g.vram.unwrap_or(0));
            match best {
                None => Choice {
                    gpu: false,
                    reason: "no GPU found".into(),
                },
                Some(g) if profile.unified_memory => Choice {
                    gpu: true,
                    reason: format!("{} shares memory with the CPU", g.name),
                },
                Some(g) => match g.vram {
                    Some(v) if v >= MIN_AUTO_VRAM => Choice {
                        gpu: true,
                        reason: format!("{} with {} MiB of memory", g.name, v >> 20),
                    },
                    Some(v) => Choice {
                        gpu: false,
                        reason: format!("{} has only {} MiB of memory", g.name, v >> 20),
                    },
                    None if matches!(g.kind, GpuKind::Nvidia | GpuKind::Amd) => Choice {
                        gpu: false,
                        reason: format!("{}: memory size unknown", g.name),
                    },
                    None => Choice {
                        gpu: false,
                        reason: format!("{} is not a compute-class GPU", g.name),
                    },
                },
            }
        }
    }
}

pub type GpuFactory<'a> = &'a dyn Fn() -> Result<Box<dyn Backend>>;

/// Build the backend for a profile. A GPU that was chosen but cannot be created falls back to the
/// CPU, and the returned text says why — never a silent switch.
pub fn build(
    profile: &HardwareProfile,
    pref: Preference,
    gpu: Option<GpuFactory<'_>>,
) -> (Box<dyn Backend>, String) {
    let c = choose(profile, pref);
    if c.gpu {
        match gpu {
            Some(f) => match f() {
                Ok(b) => {
                    let r = format!("{} ({})", b.name(), c.reason);
                    return (b, r);
                }
                Err(e) => {
                    return (
                        Box::new(CpuBackend),
                        format!("cpu: the GPU backend failed to start ({e}); {}", c.reason),
                    );
                }
            },
            None => {
                return (
                    Box::new(CpuBackend),
                    format!("cpu: this build has no GPU backend; {}", c.reason),
                );
            }
        }
    }
    (Box::new(CpuBackend), format!("cpu ({})", c.reason))
}

// ------------------------------------------------------------------------------ partial offload

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Offload {
    /// Layers `0..gpu_layers` run on the device.
    pub gpu_layers: usize,
    /// The output head also lives on the device (only when every layer does).
    pub head_on_gpu: bool,
    pub gpu_bytes: u64,
}

/// Planner-style split for a device that cannot hold the whole model: a contiguous prefix of layers
/// (so the hidden state crosses the CPU/GPU boundary once per token), then the head if everything
/// else fits. `per_layer` is each layer's weight bytes, `kv_per_layer` its cache bytes at the chosen
/// context length, `reserve` the working buffers and driver overhead.
pub fn offload_plan(
    per_layer: &[u64],
    kv_per_layer: u64,
    head: u64,
    vram: u64,
    reserve: u64,
) -> Offload {
    let mut left = vram.saturating_sub(reserve);
    let (mut n, mut used) = (0usize, 0u64);
    for &b in per_layer {
        let need = b.saturating_add(kv_per_layer);
        if need > left {
            break;
        }
        left -= need;
        used += need;
        n += 1;
    }
    let head_on_gpu = n == per_layer.len() && head <= left;
    if head_on_gpu {
        used += head;
    }
    Offload {
        gpu_layers: n,
        head_on_gpu,
        gpu_bytes: used,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw::Gpu;

    fn fill(be: &CpuBackend, v: &[f32]) -> Buf {
        be.alloc_from(v).unwrap()
    }
    fn get(be: &CpuBackend, b: &Buf) -> Vec<f32> {
        let mut o = vec![0.0; b.len];
        be.read(b, &mut o).unwrap();
        o
    }
    fn wave(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * k).sin()).collect()
    }

    #[test]
    fn dense_matvec_matches_ops() {
        let be = CpuBackend::new();
        let w = wave(6 * 10, 0.37);
        let h = be.upload(WeightRef::dense(6, 10, &w)).unwrap();
        let (x, o) = (fill(&be, &wave(10, 1.1)), be.alloc(6).unwrap());
        be.matvec(&h, &x, &o).unwrap();
        // independent f64 reference: y[r] = sum_c w[r][c] * x[c]
        let x64 = wave(10, 1.1);
        let want: Vec<f64> = (0..6)
            .map(|r| (0..10).map(|c| w[r * 10 + c] as f64 * x64[c] as f64).sum())
            .collect();
        for (g, wnt) in get(&be, &o).iter().zip(&want) {
            assert!(
                (*g as f64 - wnt).abs() < 1e-5 * (1.0 + wnt.abs()),
                "{g} vs {wnt}"
            );
        }
        assert!(!h.on_device && h.bytes == 240);
    }

    #[test]
    fn quantised_matvec_and_matmat_match_qmatmul() {
        let be = CpuBackend::new();
        let (rows, cols) = (4usize, 64usize);
        let mut bytes = vec![0u8; rows * 2 * 34];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i * 7 % 251) as u8;
        }
        for blk in bytes.chunks_mut(34) {
            blk[..2].copy_from_slice(&0x2e66u16.to_le_bytes());
        }
        let h = be
            .upload(WeightRef::quant(DType::Q8_0, rows, cols, &bytes))
            .unwrap();
        let xs: Vec<f32> = wave(2 * cols, 0.21);
        let (x, o) = (fill(&be, &xs), be.alloc(2 * rows).unwrap());
        be.matmat(&h, &x, 2, &o).unwrap();
        let mut want = vec![0.0; 2 * rows];
        for b in 0..2 {
            qmatmul::qmatvec(
                DType::Q8_0,
                &bytes,
                rows,
                cols,
                &xs[b * cols..(b + 1) * cols],
                &mut want[b * rows..(b + 1) * rows],
            )
            .unwrap();
        }
        assert_eq!(get(&be, &o), want);
    }

    #[test]
    fn bad_uploads_and_shapes_are_errors() {
        let be = CpuBackend::new();
        assert!(be.upload(WeightRef::dense(2, 2, &[1.0; 3])).is_err());
        assert!(
            be.upload(WeightRef::quant(DType::Q4_0, 1, 33, &[0; 18]))
                .is_err()
        );
        assert!(
            be.upload(WeightRef::quant(DType::Q8_0, 1, 32, &[0; 10]))
                .is_err()
        );
        assert!(
            be.upload(WeightRef::quant(DType::Other(77), 1, 32, &[0; 34]))
                .is_err()
        );
        let h = be.upload(WeightRef::dense(2, 2, &[1.0; 4])).unwrap();
        let (x, o) = (be.alloc(3).unwrap(), be.alloc(2).unwrap());
        assert!(be.matvec(&h, &x, &o).is_err());
        let x = be.alloc(2).unwrap();
        assert!(be.matvec(&h, &x, &x).is_err(), "aliased input/output");
        assert!(be.write(&x, &[1.0]).is_err());
    }

    #[test]
    fn elementwise_ops_match_the_reference_functions() {
        // Every expectation below is plain f64 maths written out here, not another call into `ops`.
        let be = CpuBackend::new();
        let close = |got: &[f32], want: &[f64], what: &str| {
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert!(
                    (*g as f64 - w).abs() < 2e-5 * (1.0 + w.abs()),
                    "{what}[{i}]: {g} vs {w}"
                );
            }
        };
        let (a, b) = (wave(24, 0.9), wave(24, 0.33));
        let (ba, bb) = (fill(&be, &a), fill(&be, &b));
        be.silu_mul(&ba, &bb).unwrap();
        let sm: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(&g, &u)| {
                let g = g as f64;
                g / (1.0 + (-g).exp()) * u as f64
            })
            .collect();
        close(&get(&be, &ba), &sm, "silu_mul");
        let after_silu = get(&be, &ba);
        be.add(&ba, &bb).unwrap();
        let sum: Vec<f64> = after_silu
            .iter()
            .zip(&b)
            .map(|(&x, &y)| x as f64 + y as f64)
            .collect();
        close(&get(&be, &ba), &sum, "add");

        // a large epsilon makes a dropped or misplaced epsilon unmistakable
        let (wv, out) = (wave(24, 0.5), be.alloc(24).unwrap());
        let (w, eps) = (fill(&be, &wv), 0.75f32);
        be.rmsnorm(&bb, &w, eps, &out).unwrap();
        let ms: f64 = b.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / 24.0;
        let rn: Vec<f64> = b
            .iter()
            .zip(&wv)
            .map(|(&x, &g)| x as f64 / (ms + eps as f64).sqrt() * g as f64)
            .collect();
        close(&get(&be, &out), &rn, "rmsnorm");

        // rotate-half RoPE: pair (i, i + d/2) rotated by pos * theta^(-2i/d)
        let r = fill(&be, &wave(32, 0.77));
        be.rope(&r, 4, 8, 13, 10_000.0).unwrap();
        let src = wave(32, 0.77);
        let mut want = vec![0.0f64; 32];
        for h in 0..4 {
            for i in 0..4 {
                let ang = 13.0 * 10_000f64.powf(-(2.0 * i as f64) / 8.0);
                let (x0, x1) = (src[h * 8 + i] as f64, src[h * 8 + i + 4] as f64);
                want[h * 8 + i] = x0 * ang.cos() - x1 * ang.sin();
                want[h * 8 + i + 4] = x0 * ang.sin() + x1 * ang.cos();
            }
        }
        close(&get(&be, &r), &want, "rope");
        assert!(be.rope(&r, 4, 7, 0, 1.0).is_err());
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_attention(
        q: &[f32],
        ks: &[Vec<f32>],
        vs: &[Vec<f32>],
        heads: usize,
        kv: usize,
        hd: usize,
        scale: f32,
        window: Option<usize>,
    ) -> Vec<f32> {
        let n = ks.len();
        let start = window.map_or(0, |w| n.saturating_sub(w));
        let mut out = vec![0.0f32; heads * hd];
        for h in 0..heads {
            let kvh = h / (heads / kv);
            let mut s: Vec<f64> = (start..n)
                .map(|t| {
                    (0..hd)
                        .map(|d| q[h * hd + d] as f64 * ks[t][kvh * hd + d] as f64)
                        .sum::<f64>()
                        * scale as f64
                })
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            s.iter_mut().for_each(|v| *v = (*v - m).exp());
            let z: f64 = s.iter().sum();
            for d in 0..hd {
                out[h * hd + d] = (0..s.len())
                    .map(|i| s[i] / z * vs[start + i][kvh * hd + d] as f64)
                    .sum::<f64>() as f32;
            }
        }
        out
    }

    #[test]
    fn attention_over_a_cache_matches_a_naive_reference_for_mha_gqa_mqa_and_windows() {
        let be = CpuBackend::new();
        for (heads, kvh, win) in [
            (4usize, 4usize, None),
            (4, 2, None),
            (4, 1, None),
            (4, 2, Some(3usize)),
        ] {
            let hd = 8;
            let kv = be.kv_create(kvh, hd, 16).unwrap();
            let (mut ks, mut vs) = (vec![], vec![]);
            for pos in 0..9 {
                let (k, v) = (
                    wave(kvh * hd, 0.3 + pos as f32 * 0.11),
                    wave(kvh * hd, 0.7 - pos as f32 * 0.05),
                );
                be.kv_append(&kv, pos, &fill(&be, &k), &fill(&be, &v))
                    .unwrap();
                ks.push(k);
                vs.push(v);
            }
            let q = wave(heads * hd, 1.3);
            let (qb, ob) = (fill(&be, &q), be.alloc(heads * hd).unwrap());
            be.attention(
                &kv,
                &qb,
                &ob,
                &AttnArgs {
                    n_heads: heads,
                    n_pos: 9,
                    scale: 0.35,
                    window: win,
                },
            )
            .unwrap();
            let want = naive_attention(&q, &ks, &vs, heads, kvh, hd, 0.35, win);
            for (a, b) in get(&be, &ob).iter().zip(&want) {
                assert!((a - b).abs() < 1e-5, "{heads}/{kvh}/{win:?}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn cache_and_attention_reject_bad_arguments() {
        let be = CpuBackend::new();
        assert!(be.kv_create(0, 8, 4).is_err());
        assert!(be.kv_create(usize::MAX, 8, 4).is_err());
        let kv = be.kv_create(2, 4, 4).unwrap();
        let row = be.alloc(8).unwrap();
        assert!(be.kv_append(&kv, 4, &row, &row).is_err());
        assert!(be.kv_append(&kv, 0, &be.alloc(7).unwrap(), &row).is_err());
        let (q, o) = (be.alloc(12).unwrap(), be.alloc(12).unwrap());
        assert!(
            be.attention(
                &kv,
                &q,
                &o,
                &AttnArgs {
                    n_heads: 3,
                    n_pos: 1,
                    scale: 1.0,
                    window: None
                }
            )
            .is_err(),
            "3 heads over 2 kv heads"
        );
        let (q, o) = (be.alloc(8).unwrap(), be.alloc(8).unwrap());
        be.attention(
            &kv,
            &q,
            &o,
            &AttnArgs {
                n_heads: 2,
                n_pos: 0,
                scale: 1.0,
                window: None,
            },
        )
        .unwrap();
        assert!(get(&be, &o).iter().all(|v| *v == 0.0));
        assert!(
            be.attention(
                &kv,
                &q,
                &o,
                &AttnArgs {
                    n_heads: 2,
                    n_pos: 5,
                    scale: 1.0,
                    window: None
                }
            )
            .is_err()
        );
    }

    fn profile(gpus: Vec<Gpu>, unified: bool) -> HardwareProfile {
        HardwareProfile {
            gpus,
            unified_memory: unified,
            ..Default::default()
        }
    }
    fn gpu(name: &str, vram: Option<u64>, kind: GpuKind) -> Gpu {
        Gpu {
            name: name.into(),
            vram,
            kind,
        }
    }

    #[test]
    fn selector_rules() {
        let big = profile(vec![gpu("RTX", Some(8 << 30), GpuKind::Nvidia)], false);
        assert!(choose(&big, Preference::Auto).gpu);
        assert!(!choose(&big, Preference::Cpu).gpu);
        let small = profile(vec![gpu("old", Some(512 << 20), GpuKind::Other)], false);
        assert!(!choose(&small, Preference::Auto).gpu);
        assert!(choose(&small, Preference::Gpu).gpu, "forcing wins");
        assert!(!choose(&profile(vec![], false), Preference::Auto).gpu);
        assert!(
            choose(
                &profile(vec![gpu("M2", None, GpuKind::Apple)], true),
                Preference::Auto
            )
            .gpu
        );
        assert!(
            !choose(
                &profile(vec![gpu("mystery", None, GpuKind::Nvidia)], false),
                Preference::Auto
            )
            .gpu
        );
        let two = profile(
            vec![
                gpu("a", Some(1 << 30), GpuKind::Amd),
                gpu("b", Some(16 << 30), GpuKind::Amd),
            ],
            false,
        );
        assert!(choose(&two, Preference::Auto).reason.contains('b'));
    }

    #[test]
    fn preference_parsing() {
        assert_eq!(Preference::parse("GPU"), Some(Preference::Gpu));
        assert_eq!(Preference::parse(" cpu "), Some(Preference::Cpu));
        assert_eq!(Preference::parse(""), Some(Preference::Auto));
        assert_eq!(Preference::parse("tpu"), None);
    }

    #[test]
    fn build_falls_back_and_says_why() {
        let big = profile(vec![gpu("RTX", Some(8 << 30), GpuKind::Nvidia)], false);
        let (b, why) = build(&big, Preference::Auto, None);
        assert!(
            !b.is_accelerator() && why.contains("no GPU backend"),
            "{why}"
        );
        let failing =
            || -> Result<Box<dyn Backend>> { Err(BackendError::Device("no adapter".into())) };
        let (b, why) = build(&big, Preference::Auto, Some(&failing));
        assert!(!b.is_accelerator() && why.contains("no adapter"), "{why}");
        let ok = || -> Result<Box<dyn Backend>> { Ok(Box::new(CpuBackend)) };
        let (_, why) = build(&big, Preference::Gpu, Some(&ok));
        assert!(why.contains("ENGINE_BACKEND=gpu"));
        let (b, why) = build(&big, Preference::Cpu, Some(&ok));
        assert!(b.name().starts_with("cpu") && why.contains("ENGINE_BACKEND=cpu"));
    }

    #[test]
    fn offload_is_a_prefix_and_the_head_goes_last() {
        let layers = [100u64; 8];
        let all = offload_plan(&layers, 10, 50, 2000, 100);
        assert_eq!(
            all,
            Offload {
                gpu_layers: 8,
                head_on_gpu: true,
                gpu_bytes: 8 * 110 + 50
            }
        );
        let part = offload_plan(&layers, 10, 50, 600, 100);
        assert_eq!(
            (part.gpu_layers, part.head_on_gpu, part.gpu_bytes),
            (4, false, 440)
        );
        let no_head = offload_plan(&layers, 10, 500, 1100, 100);
        assert_eq!((no_head.gpu_layers, no_head.head_on_gpu), (8, false));
        assert_eq!(offload_plan(&layers, 10, 50, 50, 100).gpu_layers, 0);
        assert_eq!(
            offload_plan(&[], 0, 0, 0, 0),
            Offload {
                gpu_layers: 0,
                head_on_gpu: true,
                gpu_bytes: 0
            }
        );
        assert_eq!(
            offload_plan(&[u64::MAX], u64::MAX, 1, u64::MAX - 1, 0).gpu_layers,
            0
        );
    }
}
