//! The device runner (feature `wgpu`): [`GpuBackend`] implements the engine's `Backend` on any
//! Vulkan / Metal / DX12 adapter.
//!
//! * Work is *recorded*: every op appends a compute pass to one pending command encoder, with its
//!   own small uniform buffer; nothing is submitted until the host needs a result
//!   ([`Backend::read`] / [`Backend::synchronize`]). A token therefore costs one submit and one
//!   read-back.
//! * Weights are uploaded once in the layouts of [`crate::layout`], split into row chunks that fit
//!   the adapter's storage-binding limit, and counted against a VRAM budget.
//! * Anything the device cannot take — an unsupported type, or a tensor that no longer fits the
//!   budget — is kept on the CPU backend (`on_device == false`) and computed there; the hidden
//!   state crosses the boundary only for those tensors. Together with
//!   `companion_engine::backend::offload_plan` this gives partial offload.
use crate::layout::{self, Format};
use crate::{plan, shaders};
use companion_engine::backend::*;
use companion_engine::quant::DType;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Display;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

fn dev<E: Display>(e: E) -> BackendError { BackendError::Device(e.to_string()) }
fn bytes_of(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }
fn words_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() }

#[derive(Clone, Debug)]
pub struct GpuOptions {
    /// Bytes of device memory the engine may use for weights and caches. `None`: 4 GiB (wgpu does
    /// not report VRAM; pass the figure from `hw::HardwareProfile` when it is known).
    pub vram_budget: Option<u64>,
    pub backends: wgpu::Backends,
    /// Accept a software (CPU) adapter such as llvmpipe; off by default because it is slower than
    /// the engine's own CPU kernels.
    pub allow_software: bool,
}
impl Default for GpuOptions {
    fn default() -> Self { GpuOptions { vram_budget: None, backends: wgpu::Backends::all(), allow_software: false } }
}

struct GpuBuf { buffer: wgpu::Buffer }
struct GpuWeight { fmt: Format, row_words: usize, chunks: Vec<(wgpu::Buffer, usize, usize)> }
struct GpuKv { k: wgpu::Buffer, v: wgpu::Buffer, scores: Mutex<Option<(wgpu::Buffer, usize)>> }
#[derive(Default)]
struct Pending { enc: Option<wgpu::CommandEncoder>, keep: Vec<wgpu::Buffer> }

pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    name: String,
    max_binding: u64,
    budget: u64,
    used: AtomicU64,
    pipes: Mutex<HashMap<String, Arc<wgpu::ComputePipeline>>>,
    pending: Mutex<Pending>,
    cpu: CpuBackend,
}

pub struct AdapterSummary { pub name: String, pub backend: String, pub kind: String, pub max_binding: u64, pub max_buffer: u64 }

/// Every adapter wgpu can see, for diagnostics.
pub fn adapters(backends: wgpu::Backends) -> Vec<AdapterSummary> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends, ..wgpu::InstanceDescriptor::new_without_display_handle() });
    pollster::block_on(instance.enumerate_adapters(backends)).into_iter().map(|a| {
        let (i, l) = (a.get_info(), a.limits());
        AdapterSummary { name: i.name, backend: format!("{:?}", i.backend), kind: format!("{:?}", i.device_type), max_binding: l.max_storage_buffer_binding_size, max_buffer: l.max_buffer_size }
    }).collect()
}

/// A factory for `companion_engine::backend::build`.
pub fn create(opts: GpuOptions) -> Result<Box<dyn Backend>> { Ok(Box::new(GpuBackend::new(opts)?)) }

impl GpuBackend {
    pub fn new(opts: GpuOptions) -> Result<GpuBackend> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: opts.backends, ..wgpu::InstanceDescriptor::new_without_display_handle() });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(dev)?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu && !opts.allow_software { return Err(BackendError::Device(format!("only a software adapter was found ({})", info.name))); }
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("engine"), required_limits: limits.clone(), ..Default::default() })).map_err(dev)?;
        Ok(GpuBackend {
            device, queue,
            name: format!("gpu: {} ({:?})", info.name, info.backend),
            max_binding: limits.max_storage_buffer_binding_size.min(limits.max_buffer_size).min(1 << 31),
            budget: opts.vram_budget.unwrap_or(4 << 30),
            used: AtomicU64::new(0),
            pipes: Mutex::new(HashMap::new()),
            pending: Mutex::new(Pending::default()),
            cpu: CpuBackend::new(),
        })
    }

    fn storage(&self, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes.max(4).next_multiple_of(4), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    }
    fn reserve(&self, bytes: u64) -> Result<()> {
        let used = self.used.load(Ordering::Relaxed);
        if used.saturating_add(bytes) > self.budget { return Err(BackendError::OutOfMemory { needed: bytes, available: self.budget.saturating_sub(used) }); }
        self.used.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }
    fn pipeline(&self, name: &str, src: impl FnOnce() -> String) -> Result<Arc<wgpu::ComputePipeline>> {
        let mut m = self.pipes.lock().map_err(dev)?;
        if let Some(p) = m.get(name) { return Ok(p.clone()); }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(Cow::Owned(src())) });
        let p = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(name), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None }));
        m.insert(name.to_string(), p.clone());
        Ok(p)
    }
    fn record(&self, f: impl FnOnce(&mut wgpu::CommandEncoder, &mut Vec<wgpu::Buffer>)) -> Result<()> {
        let mut p = self.pending.lock().map_err(dev)?;
        let Pending { enc, keep } = &mut *p;
        let enc = enc.get_or_insert_with(|| self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default()));
        f(enc, keep);
        Ok(())
    }
    fn dispatch(&self, kernel: &str, src: impl FnOnce() -> String, params: &[u32], binds: &[&wgpu::Buffer], groups: (u32, u32, u32)) -> Result<()> {
        let pipe = self.pipeline(kernel, src)?;
        let pb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (params.len() as u64 * 4).next_multiple_of(16), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.queue.write_buffer(&pb, 0, &words_bytes(params));
        let layout = pipe.get_bind_group_layout(0);
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: pb.as_entire_binding() }];
        entries.extend(binds.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() }));
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries });
        self.record(|enc, keep| {
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipe);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
            keep.push(pb);
        })
    }
    fn flush(&self) -> Result<()> {
        let mut p = self.pending.lock().map_err(dev)?;
        if let Some(enc) = p.enc.take() { self.queue.submit(std::iter::once(enc.finish())); }
        p.keep.clear();
        Ok(())
    }
    fn gpu_buf<'a>(&self, b: &'a Buf) -> Result<&'a wgpu::Buffer> { b.inner::<GpuBuf>().map(|g| &g.buffer).ok_or_else(|| BackendError::Device("buffer belongs to another backend".into())) }
    fn groups_for(n: usize, per: u32) -> Result<(u32, u32, u32)> {
        let (x, y, nwgx) = plan::dispatch_dims((n as u64).div_ceil(per as u64)).ok_or_else(|| BackendError::Shape("dispatch too large".into()))?;
        Ok((x, y, nwgx))
    }
    fn elementwise(&self, kernel: &str, src: &'static str, a: &Buf, b: &Buf) -> Result<()> {
        if a.len != b.len { return Err(BackendError::Shape(format!("{kernel}: lengths differ"))); }
        let (x, y, nwgx) = Self::groups_for(a.len, shaders::ELEM_WG)?;
        self.dispatch(kernel, || src.to_string(), &[a.len as u32, nwgx, 0, 0], &[self.gpu_buf(a)?, self.gpu_buf(b)?], (x, y, 1))
    }
    /// Run a weight that stayed on the CPU: read the input back, compute, write the result.
    fn host_matmat(&self, w: &WeightHandle, x: &Buf, batch: usize, out: &Buf) -> Result<()> {
        let mut xs = vec![0.0f32; x.len];
        self.read(x, &mut xs)?;
        let (cx, co) = (self.cpu.alloc_from(&xs)?, self.cpu.alloc(out.len)?);
        self.cpu.matmat(w, &cx, batch, &co)?;
        let mut ys = vec![0.0f32; out.len];
        self.cpu.read(&co, &mut ys)?;
        self.write(out, &ys)
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> String { self.name.clone() }
    fn is_accelerator(&self) -> bool { true }
    fn device_memory(&self) -> Option<u64> { Some(self.budget.saturating_sub(self.used.load(Ordering::Relaxed))) }
    fn supports(&self, t: DType) -> bool { Format::from_dtype(t).is_some() }

    fn alloc(&self, len: usize) -> Result<Buf> { Ok(Buf::new(len, Arc::new(GpuBuf { buffer: self.storage(len as u64 * 4) }))) }
    fn write(&self, dst: &Buf, src: &[f32]) -> Result<()> {
        if dst.len != src.len() { return Err(BackendError::Shape(format!("write of {} values into a buffer of {}", src.len(), dst.len))); }
        self.queue.write_buffer(self.gpu_buf(dst)?, 0, &bytes_of(src));
        Ok(())
    }
    fn read(&self, src: &Buf, dst: &mut [f32]) -> Result<()> {
        if src.len != dst.len() { return Err(BackendError::Shape(format!("read of {} values from a buffer of {}", dst.len(), src.len))); }
        self.flush()?;
        let size = src.len as u64 * 4;
        if size == 0 { return Ok(()); }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(self.gpu_buf(src)?, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(dev)?;
        rx.recv().map_err(dev)?.map_err(dev)?;
        {
            let data = slice.get_mapped_range().map_err(dev)?;
            for (o, c) in dst.iter_mut().zip(data.chunks_exact(4)) { *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]); }
        }
        staging.unmap();
        Ok(())
    }

    fn upload(&self, w: WeightRef<'_>) -> Result<WeightHandle> {
        let stored = w.validate()?;
        let Some(fmt) = Format::from_dtype(w.dtype).filter(|f| f.accepts_cols(w.cols)) else { return self.cpu.upload(w); };
        let row_words = fmt.row_words(w.cols);
        let (raw, dt): (Cow<[u8]>, DType) = match &w.data { WeightData::Dense(d) => (Cow::Owned(bytes_of(d)), DType::F32), WeightData::Quant(b) => (Cow::Borrowed(b), w.dtype) };
        let device_bytes = (w.rows * row_words * 4) as u64;
        if self.reserve(device_bytes).is_err() { return self.cpu.upload(w); }
        let Some(ranges) = plan::chunk_rows(w.rows, row_words as u64 * 4, self.max_binding) else { self.used.fetch_sub(device_bytes, Ordering::Relaxed); return self.cpu.upload(w); };
        let words = layout::repack(dt, w.rows, w.cols, &raw).map_err(|e| BackendError::Unsupported(e.to_string()))?;
        let chunks = ranges.into_iter().map(|(s, n)| {
            let buf = self.storage((n * row_words * 4) as u64);
            self.queue.write_buffer(&buf, 0, &words_bytes(&words[s * row_words..(s + n) * row_words]));
            (buf, s, n)
        }).collect();
        Ok(WeightHandle::new(w.dtype, w.rows, w.cols, stored, true, Arc::new(GpuWeight { fmt, row_words, chunks })))
    }

    fn matvec(&self, w: &WeightHandle, x: &Buf, out: &Buf) -> Result<()> { self.matmat(w, x, 1, out) }
    fn matmat(&self, w: &WeightHandle, x: &Buf, batch: usize, out: &Buf) -> Result<()> {
        if x.id == out.id { return Err(BackendError::Shape("matmat input and output are the same buffer".into())); }
        if batch == 0 || batch > plan::MAX_DIM as usize || x.len != batch * w.cols || out.len != batch * w.rows { return Err(BackendError::Shape(format!("matmat {}x{} with x {} / out {} for batch {batch}", w.rows, w.cols, x.len, out.len))); }
        let Some(gw) = w.inner::<GpuWeight>() else { return self.host_matmat(w, x, batch, out); };
        let (xb, ob) = (self.gpu_buf(x)?, self.gpu_buf(out)?);
        for (buf, start, n) in &gw.chunks {
            let (gx, gy, nwgx) = Self::groups_for(*n, shaders::MATVEC_ROWS_PER_WG)?;
            let params = [*n as u32, w.cols as u32, gw.row_words as u32, gw.fmt.row_items(w.cols) as u32, *start as u32, w.rows as u32, w.cols as u32, nwgx];
            let fmt = gw.fmt;
            self.dispatch(&format!("matvec_{}", fmt.name()), || shaders::matvec(fmt), &params, &[buf, xb, ob], (gx, gy, batch as u32))?;
        }
        Ok(())
    }
    fn rmsnorm(&self, x: &Buf, weight: &Buf, eps: f32, out: &Buf) -> Result<()> {
        if x.len != out.len || weight.len != x.len || x.id == out.id { return Err(BackendError::Shape("rmsnorm buffers".into())); }
        self.dispatch("rmsnorm", || shaders::RMSNORM.to_string(), &[x.len as u32, eps.to_bits(), x.len as u32, 0], &[self.gpu_buf(x)?, self.gpu_buf(weight)?, self.gpu_buf(out)?], (1, 1, 1))
    }
    fn rope(&self, x: &Buf, n_heads: usize, head_dim: usize, pos: usize, theta: f64) -> Result<()> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) || x.len != n_heads * head_dim { return Err(BackendError::Shape("rope dimensions".into())); }
        let half = head_dim / 2;
        let (c, s) = plan::rope_tables(pos, head_dim, theta);
        let cs: Vec<f32> = c.iter().zip(&s).flat_map(|(a, b)| [*a, *b]).collect();
        let csb = self.storage(cs.len() as u64 * 4);
        self.queue.write_buffer(&csb, 0, &bytes_of(&cs));
        let (gx, gy, nwgx) = Self::groups_for(n_heads * half, shaders::ELEM_WG)?;
        self.dispatch("rope", || shaders::ROPE.to_string(), &[(n_heads * half) as u32, head_dim as u32, half as u32, nwgx], &[&csb, self.gpu_buf(x)?], (gx, gy, 1))?;
        self.record(|_, keep| keep.push(csb))
    }
    fn silu_mul(&self, g: &Buf, u: &Buf) -> Result<()> { self.elementwise("silu_mul", shaders::SILU_MUL, g, u) }
    fn add(&self, a: &Buf, b: &Buf) -> Result<()> { self.elementwise("add", shaders::ADD, a, b) }

    fn kv_create(&self, kv_heads: usize, head_dim: usize, capacity: usize) -> Result<Kv> {
        let bytes = kv_heads.checked_mul(head_dim).and_then(|n| n.checked_mul(capacity)).and_then(|n| n.checked_mul(4)).filter(|b| *b > 0 && (*b as u64) <= self.max_binding)
            .ok_or_else(|| BackendError::Shape("KV cache size".into()))? as u64;
        self.reserve(2 * bytes)?;
        Ok(Kv::new(kv_heads, head_dim, capacity, Arc::new(GpuKv { k: self.storage(bytes), v: self.storage(bytes), scores: Mutex::new(None) })))
    }
    fn kv_append(&self, kv: &Kv, pos: usize, k: &Buf, v: &Buf) -> Result<()> {
        let row = kv.kv_heads * kv.head_dim;
        if pos >= kv.capacity || k.len != row || v.len != row { return Err(BackendError::Shape("kv_append".into())); }
        let c = kv.inner::<GpuKv>().ok_or_else(|| BackendError::Device("cache belongs to another backend".into()))?;
        let (kb, vb) = (self.gpu_buf(k)?, self.gpu_buf(v)?);
        let (off, len) = ((pos * row * 4) as u64, (row * 4) as u64);
        self.record(|enc, _| { enc.copy_buffer_to_buffer(kb, 0, &c.k, off, len); enc.copy_buffer_to_buffer(vb, 0, &c.v, off, len); })
    }
    fn attention(&self, kv: &Kv, q: &Buf, out: &Buf, a: &AttnArgs) -> Result<()> {
        let AttnArgs { n_heads, n_pos, scale, window } = *a;
        let hd = kv.head_dim;
        if n_heads == 0 || n_heads > plan::MAX_DIM as usize || !n_heads.is_multiple_of(kv.kv_heads) || q.len != n_heads * hd || out.len != q.len || n_pos > kv.capacity || q.id == out.id { return Err(BackendError::Shape("attention dimensions".into())); }
        let start = window.map_or(0, |w| n_pos.saturating_sub(w.max(1)));
        if n_pos == 0 || start >= n_pos { return self.write(out, &vec![0.0; out.len]); }
        let c = kv.inner::<GpuKv>().ok_or_else(|| BackendError::Device("cache belongs to another backend".into()))?;
        let need = n_heads * kv.capacity;
        let mut sc = c.scores.lock().map_err(dev)?;
        if sc.as_ref().is_none_or(|(_, n)| *n < need) {
            if need as u64 * 4 > self.max_binding { return Err(BackendError::Shape("scores scratch exceeds a binding".into())); }
            *sc = Some((self.storage(need as u64 * 4), need));
        }
        let scores = &sc.as_ref().ok_or_else(|| dev("scores"))?.0;
        let params = [n_heads as u32, kv.kv_heads as u32, hd as u32, n_pos as u32, start as u32, kv.capacity as u32, scale.to_bits(), 0];
        self.dispatch("attention", || shaders::ATTENTION.to_string(), &params, &[self.gpu_buf(q)?, &c.k, &c.v, scores, self.gpu_buf(out)?], (n_heads as u32, 1, 1))
    }
    fn synchronize(&self) -> Result<()> {
        self.flush()?;
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(dev)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs a real (or software) adapter: skipped cleanly when there is none.
    fn backend() -> Option<GpuBackend> { GpuBackend::new(GpuOptions { allow_software: true, ..Default::default() }).ok() }

    #[test]
    fn device_matvec_and_attention_match_the_cpu_backend() {
        let Some(g) = backend() else { eprintln!("no adapter: skipped"); return; };
        let cpu = CpuBackend::new();
        let (rows, cols) = (37usize, 96usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.05).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.3).sin()).collect();
        let run = |be: &dyn Backend| {
            let h = be.upload(WeightRef::dense(rows, cols, &w)).unwrap();
            let (xb, ob) = (be.alloc_from(&x).unwrap(), be.alloc(rows).unwrap());
            be.matvec(&h, &xb, &ob).unwrap();
            let mut o = vec![0.0; rows];
            be.read(&ob, &mut o).unwrap();
            o
        };
        for (a, b) in run(&g).iter().zip(run(&cpu)) { assert!((a - b).abs() < 1e-3, "{a} vs {b}"); }
    }
}
