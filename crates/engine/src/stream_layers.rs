//! Layer-streamed inference: run a dense model that is larger than RAM by keeping only a small
//! window of its transformer layers in memory at a time.
//!
//! The model's layers are read from disk in order, `0, 1, … n-1, 0, 1, …`, by one background
//! thread (the *prefetcher*) into a byte-budgeted [`BufferPool`]. The compute thread takes layer
//! `i`, runs the current chunk of tokens through it, and drops it at once, which frees budget for
//! the prefetcher to start on the layer after the last one it read. With a window of `k` layers
//! the prefetcher is up to `k - 1` layers ahead of the layer being computed, so disk and CPU
//! overlap; with a window of 1 the two alternate.
//!
//! * **Prefill** is processed in chunks of [`crate::model::PREFILL_CHUNK`] tokens: every layer is
//!   read once per chunk, so the cost of reading a layer is shared by the whole chunk.
//! * **Decode** has to read every layer for every token, so it is disk-bound:
//!   `tokens/s ≈ disk MB/s ÷ model MB` (see [`LayerPlan::decode_tps`]).
//! * The **KV cache** is an ordinary [`State`] in RAM. Quantised layers stay quantised: with
//!   [`LoadOptions::mmap`] a layer is a set of windows into the model file and the prefetcher
//!   touches its pages so they are read ahead, never dequantised.
//! * The embedding, output head and final norm stay resident (memory-mapped when the file allows).
//!
//! Limits: dense layers only. A layer with routed experts is an error here; mixture-of-experts
//! models stream their experts through [`crate::loader::load_model`] instead. After a layer is
//! dropped its pages stay in the operating system's page cache until the kernel needs them; the
//! pool budget counts the bytes this module holds, not the cache.
use crate::arch::{NormKind, Spec};
use crate::loader::{
    LoadError, LoadOptions, Naming, TensorSource, WeightFile, build_layer, build_outer,
};
use crate::model::{
    Bytes, Layer, Model, ModelError, PREFILL_CHUNK, QWeight, State, Weight, Weights,
};
use crate::quant::DType;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum StreamError {
    Load(LoadError),
    Model(ModelError),
    /// The memory budget cannot hold the model's resident part plus even one layer, or the model
    /// has a layer this module cannot stream.
    Plan(String),
    /// The prefetcher stopped (after an error, or the model is being dropped).
    Stopped,
}
impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::Load(e) => write!(f, "{e}"),
            StreamError::Model(e) => write!(f, "{e}"),
            StreamError::Plan(s) => write!(f, "{s}"),
            StreamError::Stopped => write!(f, "the layer prefetcher has stopped"),
        }
    }
}
impl std::error::Error for StreamError {}
impl From<LoadError> for StreamError {
    fn from(e: LoadError) -> Self {
        StreamError::Load(e)
    }
}
impl From<ModelError> for StreamError {
    fn from(e: ModelError) -> Self {
        StreamError::Model(e)
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ------------------------------------------------------------------ planning

/// How many layers fit, and what it costs, for a RAM budget (see [`plan_layers`]).
#[derive(Clone, Debug, PartialEq)]
pub struct LayerPlan {
    /// Layers that may be resident at once (the one computing plus the prefetched ones), at least 1.
    pub window_layers: usize,
    /// `window_layers × layer_bytes`: the pool budget.
    pub budget_bytes: u64,
    /// Embedding, head, norms and KV cache: always resident.
    pub resident_bytes: u64,
    /// Every layer fits at once, so streaming is pointless (load the model normally).
    pub all_resident: bool,
    /// Estimated decode speed when the disk delivers `disk_mbps`; `None` when `all_resident`.
    pub decode_tps: Option<f64>,
}

/// Largest window of `layer_bytes`-sized layers that fits `ram_budget` next to `outer_bytes` (embedding,
/// head, norms) and `kv_bytes`. At most `max_window` (a few layers are enough to hide the disk;
/// more only holds memory). `Err(missing)` when not even one layer fits: the number of bytes short.
pub fn plan_layers(
    layers: usize,
    layer_bytes: u64,
    outer_bytes: u64,
    kv_bytes: u64,
    ram_budget: u64,
    disk_mbps: f64,
    max_window: usize,
) -> Result<LayerPlan, u64> {
    let resident = outer_bytes.saturating_add(kv_bytes);
    let need_one = resident.saturating_add(layer_bytes);
    if need_one > ram_budget {
        return Err(need_one - ram_budget);
    }
    let free = ram_budget - resident;
    let fit = free.checked_div(layer_bytes).unwrap_or(layers as u64);
    let window = (fit.min(layers as u64) as usize).clamp(1, max_window.max(1).min(layers.max(1)));
    let all = layers > 0 && (fit as usize) >= layers && window >= layers;
    let model = layer_bytes.saturating_mul(layers as u64) as f64;
    let tps = (!all && disk_mbps > 0.0 && model > 0.0).then(|| disk_mbps * 1e6 / model);
    Ok(LayerPlan {
        window_layers: window,
        budget_bytes: window as u64 * layer_bytes,
        resident_bytes: resident,
        all_resident: all,
        decode_tps: tps,
    })
}

// ------------------------------------------------------------------ layer sources

/// Where a streamed model gets its layers from.
pub trait LayerSource: Send + Sync {
    fn layers(&self) -> usize;
    /// Bytes layer `i` will occupy once loaded (the pool reserves this much before loading).
    fn layer_bytes(&self, i: usize) -> usize;
    fn load(&self, i: usize) -> Result<Layer, LoadError>;
}

/// Bytes a loaded layer holds: matrices as stored, plus its f32 vectors.
pub fn layer_stored_bytes(l: &Layer) -> usize {
    let mats: usize = [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]
        .iter()
        .map(|w| w.stored_bytes())
        .sum();
    let vecs: usize = [&l.attn_norm, &l.ffn_norm]
        .iter()
        .map(|v| v.len())
        .sum::<usize>()
        + [
            &l.attn_norm_b,
            &l.bq,
            &l.bk,
            &l.bv,
            &l.bo,
            &l.q_norm,
            &l.k_norm,
            &l.post_attn_norm,
            &l.ffn_norm_b,
            &l.b_gate,
            &l.b_up,
            &l.b_down,
            &l.post_ffn_norm,
        ]
        .iter()
        .map(|v| v.as_ref().map_or(0, Vec::len))
        .sum::<usize>();
    mats + vecs * 4
}

/// Layers read from a model file (safetensors or GGUF) with the loader's own tensor logic.
pub struct FileLayers {
    spec: Spec,
    src: Arc<dyn WeightFile>,
    naming: Naming,
    permuted: bool,
    opts: LoadOptions,
    bytes: usize,
}
impl FileLayers {
    /// Reads layer 0 once to learn how big a layer is (all layers of a dense model are alike).
    pub fn new(
        spec: Spec,
        src: Arc<dyn WeightFile>,
        naming: Naming,
        llama_permuted: bool,
        opts: LoadOptions,
    ) -> Result<FileLayers, StreamError> {
        if spec.layers == 0 {
            return Err(StreamError::Plan("the model has no layers".into()));
        }
        let (l0, moe, _) = build_layer(&spec, &*src, naming, llama_permuted, opts, 0)?;
        if moe {
            return Err(StreamError::Plan(
                "layer 0 has routed experts: stream mixture-of-experts models with load_model"
                    .into(),
            ));
        }
        let bytes = layer_stored_bytes(&l0);
        Ok(FileLayers {
            spec,
            src,
            naming,
            permuted: llama_permuted,
            opts,
            bytes,
        })
    }
}
impl LayerSource for FileLayers {
    fn layers(&self) -> usize {
        self.spec.layers as usize
    }
    fn layer_bytes(&self, _i: usize) -> usize {
        self.bytes
    }
    fn load(&self, i: usize) -> Result<Layer, LoadError> {
        let (l, moe, _) = build_layer(
            &self.spec,
            &*self.src,
            self.naming,
            self.permuted,
            self.opts,
            i,
        )?;
        if moe {
            return Err(LoadError::Format(format!(
                "layer {i} has routed experts, which layer streaming does not handle"
            )));
        }
        Ok(l)
    }
}

/// A [`LayerSource`] that sleeps before every load: a stand-in for a slow disk in tests and demos.
pub struct ThrottledLayers<S: LayerSource> {
    pub inner: S,
    pub latency: Duration,
}
impl<S: LayerSource> LayerSource for ThrottledLayers<S> {
    fn layers(&self) -> usize {
        self.inner.layers()
    }
    fn layer_bytes(&self, i: usize) -> usize {
        self.inner.layer_bytes(i)
    }
    fn load(&self, i: usize) -> Result<Layer, LoadError> {
        std::thread::sleep(self.latency);
        self.inner.load(i)
    }
}

/// A [`TensorSource`] wrapper that sleeps before every read: the tensor-level version of a slow disk.
pub struct SlowTensors<T: TensorSource> {
    pub inner: T,
    pub latency: Duration,
}
impl<T: TensorSource> TensorSource for SlowTensors<T> {
    fn read(&self, name: &str) -> Result<Vec<f32>, LoadError> {
        std::thread::sleep(self.latency);
        self.inner.read(name)
    }
    fn has(&self, name: &str) -> bool {
        self.inner.has(name)
    }
    fn read_raw(
        &self,
        name: &str,
        mmap: bool,
    ) -> Result<Option<crate::loader::RawTensor>, LoadError> {
        std::thread::sleep(self.latency);
        self.inner.read_raw(name, mmap)
    }
    fn infos(&self) -> Vec<crate::gguf::TensorInfo> {
        self.inner.infos()
    }
}

/// Fault in every page of a memory-mapped layer so the read happens on the prefetch thread.
fn warm(l: &Layer) {
    let mut acc = 0u8;
    for w in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
        if let Weight::Quant(q) = w
            && q.bytes.is_mapped()
        {
            let b = q.bytes.as_slice();
            let mut i = 0;
            while i < b.len() {
                acc = acc.wrapping_add(b[i]);
                i += 4096;
            }
        }
    }
    std::hint::black_box(acc);
}

// ------------------------------------------------------------------ buffer pool

struct PoolState {
    used: usize,
    peak: usize,
    closed: bool,
}
/// A fixed byte budget that layers are charged against before they are read.
pub struct BufferPool {
    budget: usize,
    st: Mutex<PoolState>,
    cv: Condvar,
}
impl BufferPool {
    pub fn new(budget: usize) -> BufferPool {
        BufferPool {
            budget,
            st: Mutex::new(PoolState {
                used: 0,
                peak: 0,
                closed: false,
            }),
            cv: Condvar::new(),
        }
    }
    pub fn budget(&self) -> usize {
        self.budget
    }
    pub fn used(&self) -> usize {
        lock(&self.st).used
    }
    /// The most bytes that were ever charged at once.
    pub fn peak(&self) -> usize {
        lock(&self.st).peak
    }
    /// Wait until `n` bytes fit (or the pool is empty, so one oversized request cannot deadlock);
    /// `false` once the pool is closed.
    fn acquire(&self, n: usize) -> bool {
        let mut g = lock(&self.st);
        loop {
            if g.closed {
                return false;
            }
            if g.used == 0 || g.used.saturating_add(n) <= self.budget {
                break;
            }
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        g.used += n;
        g.peak = g.peak.max(g.used);
        true
    }
    fn adjust(&self, extra: usize) {
        let mut g = lock(&self.st);
        g.used += extra;
        g.peak = g.peak.max(g.used);
    }
    fn release(&self, n: usize) {
        let mut g = lock(&self.st);
        g.used = g.used.saturating_sub(n);
        self.cv.notify_all();
    }
    fn close(&self) {
        lock(&self.st).closed = true;
        self.cv.notify_all();
    }
}

// ------------------------------------------------------------------ prefetcher

/// What the streaming run measured.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamStats {
    pub layer_loads: u64,
    pub bytes_loaded: u64,
    /// Seconds the prefetch thread spent reading layers.
    pub load_secs: f64,
    /// Seconds the compute thread waited for a layer that was not there yet.
    pub wait_secs: f64,
    /// Seconds spent computing.
    pub compute_secs: f64,
    pub tokens: u64,
}
impl StreamStats {
    /// Measured read speed of the prefetch thread in MB/s.
    pub fn disk_mbps(&self) -> f64 {
        if self.load_secs > 0.0 {
            self.bytes_loaded as f64 / 1e6 / self.load_secs
        } else {
            0.0
        }
    }
}

struct Item {
    idx: usize,
    reserved: usize,
    layer: Result<Layer, LoadError>,
}
struct Queue {
    items: VecDeque<Item>,
    done: bool,
}
struct Shared {
    pool: BufferPool,
    q: Mutex<Queue>,
    ready: Condvar,
    stop: AtomicBool,
    alive: Arc<AtomicBool>,
    stats: Mutex<StreamStats>,
}

fn prefetch_loop(sh: Arc<Shared>, src: Arc<dyn LayerSource>) {
    let n = src.layers();
    let mut idx = 0usize;
    while !sh.stop.load(Ordering::Acquire) {
        let est = src.layer_bytes(idx);
        if !sh.pool.acquire(est) {
            break;
        }
        let t0 = Instant::now();
        let r = src.load(idx);
        let secs = t0.elapsed().as_secs_f64();
        let mut reserved = est;
        let failed = r.is_err();
        if let Ok(l) = &r {
            warm(l);
            let actual = layer_stored_bytes(l);
            if actual > est {
                sh.pool.adjust(actual - est);
                reserved = actual;
            }
            let mut s = lock(&sh.stats);
            s.layer_loads += 1;
            s.bytes_loaded += actual as u64;
            s.load_secs += t0.elapsed().as_secs_f64();
        } else {
            lock(&sh.stats).load_secs += secs;
        }
        {
            let mut q = lock(&sh.q);
            if sh.stop.load(Ordering::Acquire) {
                drop(q);
                sh.pool.release(reserved);
                break;
            }
            q.items.push_back(Item {
                idx,
                reserved,
                layer: r,
            });
            sh.ready.notify_all();
        }
        if failed {
            break;
        }
        idx = (idx + 1) % n.max(1);
    }
    lock(&sh.q).done = true;
    sh.ready.notify_all();
    sh.alive.store(false, Ordering::Release);
}

impl Shared {
    /// The prefetched layer `idx` (skipping stale ones left over from an aborted pass).
    fn take(&self, idx: usize) -> Result<(Layer, usize), StreamError> {
        let t0 = Instant::now();
        let mut q = lock(&self.q);
        let out = loop {
            while let Some(front) = q.items.front() {
                if front.idx == idx {
                    break;
                }
                let stale = q.items.pop_front().map_or(0, |i| i.reserved);
                self.pool.release(stale);
            }
            if let Some(it) = q.items.pop_front() {
                match it.layer {
                    Ok(l) => break Ok((l, it.reserved)),
                    Err(e) => {
                        self.pool.release(it.reserved);
                        break Err(StreamError::Load(e));
                    }
                }
            }
            if q.done {
                break Err(StreamError::Stopped);
            }
            q = self.ready.wait(q).unwrap_or_else(|e| e.into_inner());
        };
        drop(q);
        lock(&self.stats).wait_secs += t0.elapsed().as_secs_f64();
        out
    }
}

// ------------------------------------------------------------------ the streamed model

/// Layers with the right shapes and no data, so a [`Model`] can be built around them (its
/// constructor checks every layer's shape); each is swapped for the real layer while it computes.
fn placeholder_layers(spec: &Spec) -> Vec<Layer> {
    let (h, hd, ff) = (
        spec.hidden as usize,
        spec.head_dim as usize,
        spec.ffn as usize,
    );
    let (qd, kd) = (
        (spec.heads * spec.head_dim) as usize,
        (spec.kv_heads * spec.head_dim) as usize,
    );
    let ph = |rows: usize, cols: usize| {
        Weight::Quant(QWeight {
            dtype: DType::F16,
            rows,
            cols,
            bytes: Bytes::owned(vec![]),
        })
    };
    let v = |n: usize, on: bool| on.then(|| vec![0.0f32; n]);
    let ln = spec.norm == NormKind::Layer;
    (0..spec.layers)
        .map(|_| Layer {
            attn_norm: vec![0.0; h],
            attn_norm_b: v(h, ln),
            ffn_norm: vec![0.0; h],
            ffn_norm_b: v(h, ln),
            wq: ph(qd, h),
            wk: ph(kd, h),
            wv: ph(kd, h),
            wo: ph(h, qd),
            bq: v(qd, spec.qkv_bias),
            bk: v(kd, spec.qkv_bias),
            bv: v(kd, spec.qkv_bias),
            q_norm: v(hd, spec.qk_norm),
            k_norm: v(hd, spec.qk_norm),
            post_attn_norm: v(h, spec.post_norms),
            post_ffn_norm: v(h, spec.post_norms),
            w_gate: ph(if spec.gated_mlp { ff } else { 0 }, h),
            w_up: ph(ff, h),
            w_down: ph(h, ff),
            ..Default::default()
        })
        .collect()
}

/// A model whose layers live on disk. See the module documentation.
pub struct StreamedModel {
    model: Model,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    layers: usize,
    layer_bytes: usize,
    outer_bytes: usize,
}

impl StreamedModel {
    /// Open a model file (safetensors or GGUF) for streaming with `window_layers` layers resident at
    /// once. Use [`LoadOptions { prefer_quantized: true, mmap: true, .. }`](LoadOptions) so quantised layers
    /// are read as stored.
    pub fn open(
        spec: Spec,
        src: Arc<dyn WeightFile>,
        naming: Naming,
        llama_permuted: bool,
        opts: LoadOptions,
        window_layers: usize,
    ) -> Result<StreamedModel, StreamError> {
        let outer = StreamedModel::load_outer(&spec, &*src, naming, opts)?;
        let layers = FileLayers::new(spec.clone(), src, naming, llama_permuted, opts)?;
        let budget = layers.layer_bytes(0).saturating_mul(window_layers.max(1));
        let mut m = StreamedModel::from_parts(spec, outer, Arc::new(layers), budget)?;
        m.model.set_threads(opts.threads);
        Ok(m)
    }

    /// Embedding, head and final norm of a model file (what [`StreamedModel::from_parts`] takes as `outer`).
    pub fn load_outer(
        spec: &Spec,
        src: &dyn TensorSource,
        naming: Naming,
        opts: LoadOptions,
    ) -> Result<Weights, StreamError> {
        Ok(build_outer(
            spec,
            src,
            naming,
            opts,
            placeholder_layers(spec),
        )?)
    }

    /// Streaming around any [`LayerSource`] with an explicit pool `budget_bytes` (must hold at least
    /// the largest layer). `outer` supplies embedding, head and norms; its `layers` are ignored.
    pub fn from_parts(
        spec: Spec,
        mut outer: Weights,
        source: Arc<dyn LayerSource>,
        budget_bytes: usize,
    ) -> Result<StreamedModel, StreamError> {
        let n = spec.layers as usize;
        if source.layers() != n {
            return Err(StreamError::Plan(format!(
                "the source has {} layers, the architecture {n}",
                source.layers()
            )));
        }
        let biggest = (0..n).map(|i| source.layer_bytes(i)).max().unwrap_or(0);
        if biggest > budget_bytes {
            return Err(StreamError::Plan(format!(
                "a layer needs {biggest} bytes, the budget is {budget_bytes}"
            )));
        }
        outer.layers = placeholder_layers(&spec);
        let outer_bytes = outer.embed.stored_bytes()
            + outer.lm_head.as_ref().map_or(0, Weight::stored_bytes)
            + outer.final_norm.len() * 4;
        let model = Model::new(spec, outer)?;
        let alive = Arc::new(AtomicBool::new(true));
        let shared = Arc::new(Shared {
            pool: BufferPool::new(budget_bytes),
            q: Mutex::new(Queue {
                items: VecDeque::new(),
                done: false,
            }),
            ready: Condvar::new(),
            stop: AtomicBool::new(false),
            alive,
            stats: Mutex::new(StreamStats::default()),
        });
        let (sh, src) = (shared.clone(), source.clone());
        let thread = std::thread::Builder::new()
            .name("layer-prefetch".into())
            .spawn(move || prefetch_loop(sh, src))
            .map_err(|e| StreamError::Load(LoadError::Io(e)))?;
        Ok(StreamedModel {
            model,
            shared,
            thread: Some(thread),
            layers: n,
            layer_bytes: biggest,
            outer_bytes,
        })
    }

    pub fn spec(&self) -> &Spec {
        &self.model.spec
    }
    pub fn new_state(&self) -> State {
        self.model.new_state()
    }
    pub fn new_state_with(&self, kind: crate::model::KvKind) -> State {
        self.model.new_state_with(kind)
    }
    /// Bytes of embedding, head and final norm kept resident.
    pub fn outer_bytes(&self) -> usize {
        self.outer_bytes
    }
    /// Bytes of the largest layer.
    pub fn layer_bytes(&self) -> usize {
        self.layer_bytes
    }
    pub fn pool_budget(&self) -> usize {
        self.shared.pool.budget()
    }
    /// The most layer bytes ever held at once (never above [`StreamedModel::pool_budget`] for layers
    /// the source sized correctly).
    pub fn peak_resident_bytes(&self) -> usize {
        self.shared.pool.peak()
    }
    pub fn stats(&self) -> StreamStats {
        *lock(&self.shared.stats)
    }
    /// `false` once the prefetch thread has exited; it is cleared before `drop` returns.
    pub fn alive_flag(&self) -> Arc<AtomicBool> {
        self.shared.alive.clone()
    }
    /// Decode speed the disk allows: every token reads the whole model once.
    pub fn estimated_decode_tps(&self, disk_mbps: f64) -> f64 {
        let total = (self.layer_bytes * self.layers) as f64;
        if total > 0.0 {
            disk_mbps * 1e6 / total
        } else {
            0.0
        }
    }

    /// Feed one token, return the logits for the next one.
    pub fn forward(&mut self, st: &mut State, token: usize) -> Result<Vec<f32>, StreamError> {
        self.forward_batch(st, &[token])
    }

    /// Feed many tokens and return the logits after the last one. Tokens go through the layers in
    /// chunks of [`PREFILL_CHUNK`]; each chunk reads every layer once.
    pub fn forward_batch(
        &mut self,
        st: &mut State,
        tokens: &[usize],
    ) -> Result<Vec<f32>, StreamError> {
        let s = &self.model.spec;
        if tokens.is_empty() {
            return Err(ModelError::NoTokens.into());
        }
        if let Some(&t) = tokens.iter().find(|&&t| t >= s.vocab as usize) {
            return Err(ModelError::Token(t).into());
        }
        if st.pos + tokens.len() > s.context as usize {
            return Err(ModelError::ContextFull(s.context as usize).into());
        }
        let chunks = tokens.len().div_ceil(PREFILL_CHUNK);
        let mut last = vec![];
        for (i, c) in tokens.chunks(PREFILL_CHUNK).enumerate() {
            last = self.run_chunk(st, c, i + 1 == chunks)?;
        }
        Ok(last)
    }

    fn run_chunk(
        &mut self,
        st: &mut State,
        tokens: &[usize],
        want_logits: bool,
    ) -> Result<Vec<f32>, StreamError> {
        let h = self.model.spec.hidden as usize;
        let b = tokens.len();
        let pos0 = st.pos;
        let mut x = Vec::with_capacity(b * h);
        for &t in tokens {
            x.extend_from_slice(&self.model.embed_token(t)?);
        }
        for li in 0..self.layers {
            let (real, reserved) = self.shared.take(li)?;
            let ph = std::mem::replace(&mut self.model.w.layers[li], real);
            let t0 = Instant::now();
            let r = self.model.run_layers_batch(st, li..li + 1, &mut x, pos0);
            let real = std::mem::replace(&mut self.model.w.layers[li], ph);
            drop(real);
            self.shared.pool.release(reserved);
            lock(&self.shared.stats).compute_secs += t0.elapsed().as_secs_f64();
            r?;
        }
        self.model.finish_batch(st, tokens);
        lock(&self.shared.stats).tokens += b as u64;
        if want_logits {
            let out = self.model.logits_from_hidden(&x[(b - 1) * h..b * h]);
            if out.is_empty() {
                return Err(ModelError::Shape("the output head failed".into()).into());
            }
            Ok(out)
        } else {
            Ok(vec![])
        }
    }
}

impl Drop for StreamedModel {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.pool.close();
        self.shared.ready.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let mut q = lock(&self.shared.q);
        q.items.clear();
    }
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::SafeTensors;
    use crate::model::tests::{base_spec, model_of};
    use std::path::{Path, PathBuf};

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(
            format!(
                "stream-layers-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            )
            .replace(['(', ')', ' '], ""),
        );
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn f32b(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn write_st(path: &Path, ts: &[(String, Vec<usize>, Vec<u8>)]) {
        let mut hdr = serde_json::Map::new();
        let mut off = 0;
        for (n, shape, bytes) in ts {
            hdr.insert(n.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [off, off + bytes.len()]}));
            off += bytes.len();
        }
        let h = serde_json::to_vec(&hdr).unwrap();
        let mut f = (h.len() as u64).to_le_bytes().to_vec();
        f.extend(h);
        for t in ts {
            f.extend(&t.2);
        }
        std::fs::write(path, f).unwrap();
    }
    /// Every tensor of `m` under Hugging Face names, optionally leaving one out.
    fn tensors(m: &Model, skip: Option<&str>) -> Vec<(String, Vec<usize>, Vec<u8>)> {
        let s = &m.spec;
        let (h, qd, kd, ff, v) = (
            s.hidden as usize,
            (s.heads * s.head_dim) as usize,
            (s.kv_heads * s.head_dim) as usize,
            s.ffn as usize,
            s.vocab as usize,
        );
        let mut o = vec![
            (
                "model.embed_tokens.weight".to_string(),
                vec![v, h],
                f32b(&m.w.embed.to_f32()),
            ),
            (
                "model.norm.weight".to_string(),
                vec![h],
                f32b(&m.w.final_norm),
            ),
        ];
        if let Some(l) = &m.w.lm_head {
            o.push(("lm_head.weight".into(), vec![v, h], f32b(&l.to_f32())));
        }
        for (i, l) in m.w.layers.iter().enumerate() {
            for (p, shape, d) in [
                ("attn_norm", vec![h], l.attn_norm.clone()),
                ("wq", vec![qd, h], l.wq.to_f32()),
                ("wk", vec![kd, h], l.wk.to_f32()),
                ("wv", vec![kd, h], l.wv.to_f32()),
                ("wo", vec![h, qd], l.wo.to_f32()),
                ("ffn_norm", vec![h], l.ffn_norm.clone()),
                ("w_gate", vec![ff, h], l.w_gate.to_f32()),
                ("w_up", vec![ff, h], l.w_up.to_f32()),
                ("w_down", vec![h, ff], l.w_down.to_f32()),
            ] {
                o.push((crate::loader::layer_name(Naming::Hf, i, p), shape, f32b(&d)));
            }
        }
        o.retain(|t| Some(t.0.as_str()) != skip);
        o
    }
    fn setup(tag: &str, layers: u64) -> (Model, PathBuf, Arc<dyn WeightFile>) {
        let mut sp = base_spec(4, 2, false);
        sp.layers = layers;
        sp.context = 512;
        let m = model_of(sp, false);
        let d = tmpdir(tag);
        let p = d.join("m.safetensors");
        write_st(&p, &tensors(&m, None));
        let src: Arc<dyn WeightFile> = Arc::new(SafeTensors::open(&p).unwrap());
        (m, d, src)
    }
    fn open(m: &Model, src: &Arc<dyn WeightFile>, window: usize) -> StreamedModel {
        StreamedModel::open(
            m.spec.clone(),
            src.clone(),
            Naming::Hf,
            false,
            LoadOptions::default(),
            window,
        )
        .unwrap()
    }
    const PROMPT: [usize; 7] = [3, 11, 5, 19, 2, 7, 13];

    #[test]
    fn streamed_logits_equal_the_in_ram_model_for_every_window() {
        let (m, d, src) = setup("equal", 5);
        let mut st = m.new_state();
        let want = m.forward_batch(&mut st, &PROMPT).unwrap();
        let want2 = m.forward(&mut st, 4).unwrap();
        for window in [1, 2, 5] {
            let mut sm = open(&m, &src, window);
            let mut s = sm.new_state();
            assert_eq!(
                sm.forward_batch(&mut s, &PROMPT).unwrap(),
                want,
                "window {window}"
            );
            assert_eq!(
                sm.forward(&mut s, 4).unwrap(),
                want2,
                "window {window} decode"
            );
            assert_eq!(s.pos, PROMPT.len() + 1);
            assert_eq!(s.tokens(), st.tokens());
        }
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn token_by_token_equals_batched() {
        let (m, d, src) = setup("batch", 4);
        let mut sm = open(&m, &src, 2);
        let (mut a, mut b) = (sm.new_state(), sm.new_state());
        let batched = sm.forward_batch(&mut a, &PROMPT).unwrap();
        let mut one = vec![];
        for &t in &PROMPT {
            one = sm.forward(&mut b, t).unwrap();
        }
        let worst = batched
            .iter()
            .zip(&one)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(worst < 1e-4, "{worst}");
        // a prompt longer than one prefill chunk goes through several chunks
        let long: Vec<usize> = (0..PREFILL_CHUNK + 9).map(|i| (i * 7) % 20).collect();
        let mut ram = m.new_state();
        let mut s = sm.new_state();
        assert_eq!(
            sm.forward_batch(&mut s, &long).unwrap(),
            m.forward_batch(&mut ram, &long).unwrap()
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn peak_resident_bytes_stay_within_the_budget() {
        let (m, d, src) = setup("budget", 6);
        for window in [1usize, 2, 3] {
            let mut sm = open(&m, &src, window);
            let mut s = sm.new_state();
            sm.forward_batch(&mut s, &PROMPT).unwrap();
            for t in [1, 2, 3] {
                sm.forward(&mut s, t).unwrap();
            }
            let (peak, budget) = (sm.peak_resident_bytes(), sm.pool_budget());
            assert_eq!(budget, window * sm.layer_bytes());
            assert!(
                peak <= budget,
                "window {window}: peak {peak} over budget {budget}"
            );
            assert!(peak >= sm.layer_bytes());
        }
        assert!(
            StreamedModel::from_parts(
                m.spec.clone(),
                crate::model::tests::random_weights(&m.spec, 1, false),
                Arc::new(
                    FileLayers::new(
                        m.spec.clone(),
                        src.clone(),
                        Naming::Hf,
                        false,
                        LoadOptions::default()
                    )
                    .unwrap()
                ),
                10
            )
            .is_err(),
            "a budget below one layer is refused"
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn dropping_the_model_stops_the_prefetcher() {
        let (m, d, src) = setup("drop", 4);
        let sm = open(&m, &src, 3);
        let alive = sm.alive_flag();
        assert!(alive.load(Ordering::Acquire));
        drop(sm);
        assert!(!alive.load(Ordering::Acquire));
        // also while the prefetcher is blocked inside a slow load
        let slow = Arc::new(ThrottledLayers {
            inner: FileLayers::new(
                m.spec.clone(),
                src.clone(),
                Naming::Hf,
                false,
                LoadOptions::default(),
            )
            .unwrap(),
            latency: Duration::from_millis(50),
        });
        let sm = StreamedModel::from_parts(
            m.spec.clone(),
            crate::model::tests::random_weights(&m.spec, 1, false),
            slow,
            1 << 20,
        )
        .unwrap();
        let alive = sm.alive_flag();
        drop(sm);
        assert!(!alive.load(Ordering::Acquire));
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn hostile_and_truncated_files_are_errors() {
        let (m, d, _) = setup("hostile", 4);
        let full = d.join("m.safetensors");
        let bytes = std::fs::read(&full).unwrap();
        // truncated file: the loader refuses it before streaming starts
        let cut = d.join("cut.safetensors");
        std::fs::write(&cut, &bytes[..bytes.len() / 2]).unwrap();
        assert!(SafeTensors::open(&cut).is_err());
        // garbage
        let junk = d.join("junk.safetensors");
        std::fs::write(&junk, [7u8; 40]).unwrap();
        assert!(SafeTensors::open(&junk).is_err());
        // a tensor of layer 2 missing: opening works (layer 0 is fine), the forward pass reports it and does not hang
        let gap = d.join("gap.safetensors");
        write_st(
            &gap,
            &tensors(&m, Some("model.layers.2.mlp.up_proj.weight")),
        );
        let src: Arc<dyn WeightFile> = Arc::new(SafeTensors::open(&gap).unwrap());
        let mut sm = open(&m, &src, 2);
        let mut s = sm.new_state();
        assert!(sm.forward_batch(&mut s, &PROMPT).is_err());
        assert!(
            sm.forward(&mut s, 1).is_err(),
            "later calls fail instead of waiting forever"
        );
        // layer 0 missing: open itself fails
        let none = d.join("none.safetensors");
        write_st(
            &none,
            &tensors(&m, Some("model.layers.0.self_attn.q_proj.weight")),
        );
        let src: Arc<dyn WeightFile> = Arc::new(SafeTensors::open(&none).unwrap());
        assert!(
            StreamedModel::open(
                m.spec.clone(),
                src,
                Naming::Hf,
                false,
                LoadOptions::default(),
                2
            )
            .is_err()
        );
        // bad input to a healthy model
        let (m2, d2, src2) = setup("hostile2", 3);
        let mut sm = open(&m2, &src2, 1);
        let mut s = sm.new_state();
        assert!(sm.forward_batch(&mut s, &[]).is_err());
        assert!(sm.forward(&mut s, 10_000).is_err());
        assert!(
            sm.forward(&mut s, 1).is_ok(),
            "a rejected call leaves the pipeline usable"
        );
        std::fs::remove_dir_all(d).unwrap();
        std::fs::remove_dir_all(d2).unwrap();
    }

    #[test]
    fn prefetching_hides_a_slow_disk() {
        // a model big enough that a chunk's compute is comparable to a layer read
        let mut sp = base_spec(4, 2, false);
        sp.layers = 8;
        sp.hidden = 128;
        sp.head_dim = 32;
        sp.ffn = 512;
        sp.vocab = 64;
        sp.rotary_dim = 32;
        sp.context = 512;
        let m = model_of(sp, false);
        let prompt: Vec<usize> = (0..PREFILL_CHUNK).map(|i| i % 60).collect();
        // measure compute per layer on the in-RAM model
        let mut st = m.new_state();
        let t0 = Instant::now();
        m.forward_batch(&mut st, &prompt).unwrap();
        let per_layer = t0.elapsed() / m.spec.layers as u32;
        let latency = per_layer.max(Duration::from_millis(2));
        let run = |window: usize| {
            let src = Arc::new(ThrottledLayers {
                inner: RamLayers(m.w.layers.clone()),
                latency,
            });
            let mut sm = StreamedModel::from_parts(
                m.spec.clone(),
                crate::model::tests::random_weights(&m.spec, 1, false),
                src,
                window * RamLayers(m.w.layers.clone()).layer_bytes(0),
            )
            .unwrap();
            let mut s = sm.new_state();
            let t = Instant::now();
            let out = sm.forward_batch(&mut s, &prompt).unwrap();
            for _ in 0..3 {
                sm.forward_batch(&mut s, &prompt[..16]).unwrap();
            }
            (t.elapsed(), out, sm.stats())
        };
        let (t1, o1, _) = run(1);
        let (t3, o3, st3) = run(3);
        assert_eq!(o1, o3);
        assert!(
            t3.as_secs_f64() < t1.as_secs_f64() * 0.85,
            "window 1 took {t1:?}, window 3 took {t3:?}"
        );
        assert!(st3.disk_mbps() > 0.0 && st3.layer_loads >= 8);
    }

    /// Layers already in memory; lets a test throttle loads without a file.
    struct RamLayers(Vec<Layer>);
    impl LayerSource for RamLayers {
        fn layers(&self) -> usize {
            self.0.len()
        }
        fn layer_bytes(&self, i: usize) -> usize {
            layer_stored_bytes(&self.0[i])
        }
        fn load(&self, i: usize) -> Result<Layer, LoadError> {
            Ok(self.0[i].clone())
        }
    }

    #[test]
    fn plan_picks_the_largest_window_that_fits() {
        // 10 layers of 100 bytes, 50 resident + 20 kv
        let p = plan_layers(10, 100, 50, 20, 70 + 350, 500.0, 4).unwrap();
        assert_eq!(
            (
                p.window_layers,
                p.budget_bytes,
                p.resident_bytes,
                p.all_resident
            ),
            (3, 300, 70, false)
        );
        assert!((p.decode_tps.unwrap() - 500.0e6 / 1000.0).abs() < 1e-6);
        assert_eq!(
            plan_layers(10, 100, 50, 20, 70 + 100, 1.0, 4)
                .unwrap()
                .window_layers,
            1
        );
        assert_eq!(
            plan_layers(10, 100, 50, 20, 1_000_000, 1.0, 4)
                .unwrap()
                .window_layers,
            4,
            "capped at max_window"
        );
        let all = plan_layers(3, 100, 0, 0, 1_000_000, 1.0, 8).unwrap();
        assert!(all.all_resident && all.decode_tps.is_none() && all.window_layers == 3);
        assert_eq!(plan_layers(10, 100, 50, 20, 150, 1.0, 4), Err(20));
    }

    #[test]
    fn estimated_decode_speed_is_disk_over_model_size() {
        let (m, d, src) = setup("tps", 4);
        let sm = open(&m, &src, 1);
        let total = (sm.layer_bytes() * 4) as f64;
        assert!((sm.estimated_decode_tps(200.0) - 200.0e6 / total).abs() < 1e-9);
        std::fs::remove_dir_all(d).unwrap();
    }
}
