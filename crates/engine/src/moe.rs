//! Mixture-of-experts feed-forward block: router, expert FFN, and expert providers.
//!
//! * [`Router`] – softmax or sigmoid scoring, deterministic top-k (ties go to the lower index),
//!   optional renormalisation, optional DeepSeek-V3-style grouped selection with a selection-only bias.
//! * [`MoeBlock`] – routed experts plus always-on shared experts; a batch is grouped per expert so
//!   each expert is fetched once per batch.
//! * [`ExpertProvider`] – where expert weights come from: [`InMemoryExperts`], or
//!   [`StreamedExperts`] which reads them lazily from a [`TensorSource`] (one tensor per expert, as in
//!   Hugging Face Mixtral/Qwen-MoE) or from stacked tensors (GGUF `ffn_*_exps`) through a
//!   [`RangeSource`], behind a byte-budgeted LRU cache with a bounded background prefetcher.
//!
//! # Wiring the stacked GGUF reader later
//! `loader::GgufFile` knows `data_start + tensor.offset` and the dtype. To serve stacked experts without
//! reading a whole `[E, ffn, hidden]` tensor it should implement [`RangeSource::read_range`] by seeking to
//! `offset + start_elem * bytes_per_elem` (for block-quantised types: whole blocks) and decoding only
//! `len` elements. [`StackedFromTensors`] is the slow, always-correct adapter used until then.
use crate::arch::Act;
use crate::loader::{RawTensor, TensorSource};
use crate::model::{activate, QWeight, Weight};
use crate::ops::{matvec, softmax};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoeError {
    /// Router or block configuration is impossible (k > experts, bad groups, wrong sizes).
    Config(String),
    /// A tensor or vector has the wrong number of values.
    Shape(String),
    /// NaN or infinity in router logits or activations.
    NonFinite,
    /// A tensor, expert or layer does not exist.
    Missing(String),
    /// The underlying file or source failed.
    Load(String),
}
impl std::fmt::Display for MoeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeError::Config(s) => write!(f, "bad mixture-of-experts configuration: {s}"),
            MoeError::Shape(s) => write!(f, "bad expert shape: {s}"),
            MoeError::NonFinite => write!(f, "router received a non-finite value"),
            MoeError::Missing(s) => write!(f, "missing expert data: {s}"),
            MoeError::Load(s) => write!(f, "could not read expert: {s}"),
        }
    }
}
impl std::error::Error for MoeError {}

// ------------------------------------------------------------------ expert weights

/// One expert: gate and up are `ffn × hidden`, down is `hidden × ffn` (row-major). Each matrix is
/// either dense f32 or kept quantised as stored, and multiplied without being expanded.
#[derive(Clone)]
pub struct ExpertWeights { pub gate: Weight, pub up: Weight, pub down: Weight }

impl std::fmt::Debug for ExpertWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExpertWeights {{ gate: {}, up: {}, down: {} }}", self.gate.len(), self.up.len(), self.down.len())
    }
}
impl PartialEq for ExpertWeights {
    fn eq(&self, o: &Self) -> bool { self.gate.to_f32() == o.gate.to_f32() && self.up.to_f32() == o.up.to_f32() && self.down.to_f32() == o.down.to_f32() }
}

fn shape_err(e: crate::model::ModelError) -> MoeError { MoeError::Shape(e.to_string()) }

#[derive(Default)]
struct ExpertScratch { g: Vec<f32>, u: Vec<f32>, d: Vec<f32>, yt: Vec<f32> }

impl ExpertWeights {
    pub fn dense(gate: Vec<f32>, up: Vec<f32>, down: Vec<f32>) -> ExpertWeights {
        ExpertWeights { gate: gate.into(), up: up.into(), down: down.into() }
    }
    /// Bytes held while this expert is resident (quantised experts count their stored size).
    pub fn bytes(&self) -> u64 { (self.gate.stored_bytes() + self.up.stored_bytes() + self.down.stored_bytes()) as u64 }
    pub fn ffn(&self, hidden: usize) -> usize { self.gate.len().checked_div(hidden).unwrap_or(0) }
    pub fn validate(&self, hidden: usize, ffn: usize) -> Result<(), MoeError> {
        let n = hidden.checked_mul(ffn).ok_or_else(|| MoeError::Shape("size overflow".into()))?;
        let quant_ok = |w: &Weight, rows: usize, cols: usize| match w { Weight::Quant(q) => q.rows == rows && q.cols == cols, Weight::Dense(_) => true };
        if self.gate.len() == n && self.up.len() == n && self.down.len() == n
            && quant_ok(&self.gate, ffn, hidden) && quant_ok(&self.up, ffn, hidden) && quant_ok(&self.down, hidden, ffn) { return Ok(()); }
        Err(MoeError::Shape(format!("expected {n} values per matrix ({ffn}x{hidden}), got gate {} up {} down {}",
            self.gate.len(), self.up.len(), self.down.len())))
    }
    /// `down(act(gate x) * up x)` for `b` tokens at once; the result is left in `sc.d` (`b × hidden`).
    fn run(&self, hidden: usize, act: Act, xs: &[f32], b: usize, sc: &mut ExpertScratch) -> Result<(), MoeError> {
        let ffn = self.ffn(hidden);
        for (v, n) in [(&mut sc.g, b * ffn), (&mut sc.u, b * ffn), (&mut sc.d, b * hidden)] { v.clear(); v.resize(n, 0.0); }
        self.gate.matmat(hidden, xs, b, &mut sc.g, &mut sc.yt).map_err(shape_err)?;
        self.up.matmat(hidden, xs, b, &mut sc.u, &mut sc.yt).map_err(shape_err)?;
        sc.g.iter_mut().zip(&sc.u).for_each(|(a, u)| *a = activate(act, *a) * u);
        self.down.matmat(ffn, &sc.g, b, &mut sc.d, &mut sc.yt).map_err(shape_err)
    }
}

// ------------------------------------------------------------------------- router

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scoring { Softmax, Sigmoid }

#[derive(Clone, Debug)]
pub struct RouterConfig {
    pub hidden: usize,
    pub experts: usize,
    pub top_k: usize,
    pub scoring: Scoring,
    /// Divide the selected weights by their sum.
    pub renormalise: bool,
    /// Multiplier applied to the final routed weights.
    pub routed_scale: f32,
    /// `(n_group, topk_group)`: only experts in the best `topk_group` of `n_group` groups may be picked.
    pub groups: Option<(usize, usize)>,
    /// Added to scores for *selection only* (never to the weights), as in DeepSeek-V3.
    pub selection_bias: Option<Vec<f32>>,
}
impl RouterConfig {
    pub fn softmax(hidden: usize, experts: usize, top_k: usize) -> Self {
        RouterConfig { hidden, experts, top_k, scoring: Scoring::Softmax, renormalise: true, routed_scale: 1.0, groups: None, selection_bias: None }
    }
}

#[derive(Clone)]
pub struct Router { cfg: RouterConfig, weight: Vec<f32> }

/// Indices of the `k` largest scores, best first; equal scores keep the lower index first.
pub fn top_k(scores: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    // `+ 0.0` folds -0.0 into 0.0 so signed zeros tie like any other equal scores
    idx.sort_by(|&a, &b| (scores[b] + 0.0).total_cmp(&(scores[a] + 0.0)).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

impl Router {
    pub fn new(cfg: RouterConfig, weight: Vec<f32>) -> Result<Router, MoeError> {
        let err = |s: &str| Err(MoeError::Config(s.into()));
        if cfg.experts == 0 || cfg.hidden == 0 { return err("no experts or zero hidden size"); }
        if cfg.top_k == 0 || cfg.top_k > cfg.experts { return err("top_k must be between 1 and the number of experts"); }
        if weight.len() != cfg.experts.saturating_mul(cfg.hidden) {
            return Err(MoeError::Shape(format!("router has {} values, expected {}", weight.len(), cfg.experts * cfg.hidden)));
        }
        if let Some(b) = &cfg.selection_bias
            && b.len() != cfg.experts
        {
            return err("selection bias length differs from the number of experts");
        }
        if let Some((ng, tg)) = cfg.groups {
            if ng == 0 || tg == 0 || tg > ng || !cfg.experts.is_multiple_of(ng) { return err("invalid expert groups"); }
            if cfg.top_k > tg * (cfg.experts / ng) { return err("top_k is larger than the experts left after group selection"); }
        }
        Ok(Router { cfg, weight })
    }
    pub fn config(&self) -> &RouterConfig { &self.cfg }
    /// The `experts × hidden` router matrix.
    pub fn weights(&self) -> &[f32] { &self.weight }

    /// Chosen experts and their weights, in selection order.
    pub fn route(&self, x: &[f32]) -> Result<Vec<(usize, f32)>, MoeError> {
        let c = &self.cfg;
        if x.len() != c.hidden { return Err(MoeError::Shape(format!("input has {} values, expected {}", x.len(), c.hidden))); }
        let mut logits = vec![0.0; c.experts];
        matvec(&self.weight, c.hidden, x, &mut logits);
        if logits.iter().any(|v| !v.is_finite()) { return Err(MoeError::NonFinite); }
        match c.scoring {
            Scoring::Softmax => softmax(&mut logits),
            Scoring::Sigmoid => logits.iter_mut().for_each(|v| *v = 1.0 / (1.0 + (-*v).exp())),
        }
        let scores = logits;
        let mut sel: Vec<f32> = match &c.selection_bias {
            Some(b) => scores.iter().zip(b).map(|(s, b)| s + b).collect(),
            None => scores.clone(),
        };
        if sel.iter().any(|v| !v.is_finite()) { return Err(MoeError::NonFinite); }
        if let Some((ng, tg)) = c.groups {
            let gs = c.experts / ng;
            let gscore: Vec<f32> = sel.chunks(gs).map(|g| top_k(g, 2).iter().map(|&i| g[i]).sum()).collect();
            let keep = top_k(&gscore, tg);
            for (g, chunk) in sel.chunks_mut(gs).enumerate() {
                if !keep.contains(&g) { chunk.fill(f32::NEG_INFINITY); }
            }
        }
        let picked = top_k(&sel, c.top_k);
        let mut out: Vec<(usize, f32)> = picked.into_iter().map(|i| (i, scores[i])).collect();
        if c.renormalise {
            let sum: f32 = out.iter().map(|p| p.1).sum();
            if sum > 0.0 { out.iter_mut().for_each(|p| p.1 /= sum); }
        }
        out.iter_mut().for_each(|p| p.1 *= c.routed_scale);
        Ok(out)
    }
}

// ----------------------------------------------------------------------- providers

pub trait ExpertProvider: Send + Sync {
    fn expert(&self, layer: usize, id: usize) -> Result<Arc<ExpertWeights>, MoeError>;
    /// Hint that these experts will be needed soon. Default: ignore.
    fn prefetch(&self, _layer: usize, _ids: &[usize]) {}
    /// Cache statistics, for providers that have a cache.
    fn stats(&self) -> Option<Stats> { None }
    /// Bytes of expert weights currently held in memory.
    fn resident_bytes(&self) -> u64 { 0 }
}

pub struct InMemoryExperts { layers: Vec<Vec<Arc<ExpertWeights>>> }
impl InMemoryExperts {
    pub fn new(layers: Vec<Vec<ExpertWeights>>) -> Self {
        InMemoryExperts { layers: layers.into_iter().map(|l| l.into_iter().map(Arc::new).collect()).collect() }
    }
}
impl ExpertProvider for InMemoryExperts {
    fn expert(&self, layer: usize, id: usize) -> Result<Arc<ExpertWeights>, MoeError> {
        self.layers.get(layer).and_then(|l| l.get(id)).cloned().ok_or_else(|| MoeError::Missing(format!("layer {layer} expert {id}")))
    }
    fn resident_bytes(&self) -> u64 { self.layers.iter().flatten().map(|e| e.bytes()).sum() }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part { Gate, Up, Down }

/// Tensor-name conventions.
pub mod names {
    use super::Part;
    /// Mixtral: `block_sparse_moe.experts.E.w1` (gate) / `w3` (up) / `w2` (down).
    pub fn mixtral(layer: usize, id: usize, p: Part) -> String {
        let w = match p { Part::Gate => "w1", Part::Up => "w3", Part::Down => "w2" };
        format!("model.layers.{layer}.block_sparse_moe.experts.{id}.{w}.weight")
    }
    /// Qwen-MoE / OLMoE / DeepSeek style: `mlp.experts.E.{gate,up,down}_proj`.
    pub fn hf_experts(layer: usize, id: usize, p: Part) -> String {
        let w = match p { Part::Gate => "gate_proj", Part::Up => "up_proj", Part::Down => "down_proj" };
        format!("model.layers.{layer}.mlp.experts.{id}.{w}.weight")
    }
    /// GGUF stacked tensors `blk.N.ffn_{gate,up,down}_exps.weight`.
    pub fn gguf_stacked(layer: usize, p: Part) -> String {
        let w = match p { Part::Gate => "gate", Part::Up => "up", Part::Down => "down" };
        format!("blk.{layer}.ffn_{w}_exps.weight")
    }
}

/// Reads a run of elements out of a (stacked) tensor, in f32 elements, without the rest of it.
pub trait RangeSource: Send + Sync {
    fn read_range(&self, name: &str, start: usize, len: usize) -> Result<Vec<f32>, MoeError>;
    /// The same range as a `rows × cols` matrix in its stored (quantised or half-precision) form, or
    /// `None` (the default) when it has to be decoded to f32. `start` is in elements.
    fn read_range_raw(&self, _name: &str, _start: usize, _rows: usize, _cols: usize, _mmap: bool) -> Result<Option<RawTensor>, MoeError> { Ok(None) }
}

/// How streamed experts are held once read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadMode {
    /// Keep quantised / half-precision experts as stored instead of expanding them to f32.
    pub quantized: bool,
    /// Serve them as windows into the memory-mapped file when the source can.
    pub mmap: bool,
}

/// Slow adapter: reads the whole stacked tensor and slices it. Correct for any [`TensorSource`].
pub struct StackedFromTensors<S: TensorSource + Send + Sync>(pub S);
impl<S: TensorSource + Send + Sync> RangeSource for StackedFromTensors<S> {
    fn read_range(&self, name: &str, start: usize, len: usize) -> Result<Vec<f32>, MoeError> {
        let all = self.0.read(name).map_err(|e| MoeError::Load(e.to_string()))?;
        let end = start.checked_add(len).filter(|e| *e <= all.len()).ok_or_else(|| MoeError::Shape(format!("{name}: range past end")))?;
        Ok(all[start..end].to_vec())
    }
}

type PerExpertNamer = Box<dyn Fn(usize, usize, Part) -> String + Send + Sync>;
type StackedNamer = Box<dyn Fn(usize, Part) -> String + Send + Sync>;

enum Source {
    PerExpert { src: Arc<dyn TensorSource + Send + Sync>, name: PerExpertNamer },
    Stacked { src: Arc<dyn RangeSource>, name: StackedNamer },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats { pub hits: u64, pub misses: u64, pub evictions: u64, pub bytes_read: u64, pub prefetched: u64, pub resident_bytes: u64 }

#[derive(Default)]
struct Cache { map: HashMap<(usize, usize), (Arc<ExpertWeights>, u64)>, bytes: u64, tick: u64 }

const QUEUE_CAP: usize = 64;
const MAX_WORKERS: usize = 2;

struct Inner {
    source: Source, hidden: usize, ffn: usize, budget: u64, mode: ReadMode,
    cache: Mutex<Cache>,
    hits: AtomicU64, misses: AtomicU64, evictions: AtomicU64, bytes_read: AtomicU64, prefetched: AtomicU64,
    queue: Mutex<VecDeque<(usize, usize)>>, workers: AtomicUsize,
}

impl Inner {
    fn read_expert(&self, layer: usize, id: usize) -> Result<ExpertWeights, MoeError> {
        let (h, f) = (self.hidden, self.ffn);
        let n = h.checked_mul(f).ok_or_else(|| MoeError::Shape("size overflow".into()))?;
        let quant = |r: RawTensor, rows: usize, cols: usize, nm: &str| -> Result<Weight, MoeError> {
            if r.rows != rows || r.cols != cols { return Err(MoeError::Shape(format!("{nm}: stored as {}x{}, expected {rows}x{cols}", r.rows, r.cols))); }
            QWeight::new(r.dtype, r.rows, r.cols, r.bytes).map(Weight::Quant).map_err(shape_err)
        };
        let get = |p: Part| -> Result<Weight, MoeError> {
            let (rows, cols) = if p == Part::Down { (h, f) } else { (f, h) };
            match &self.source {
                Source::PerExpert { src, name } => {
                    let nm = name(layer, id, p);
                    if !src.has(&nm) { return Err(MoeError::Missing(nm)); }
                    if self.mode.quantized
                        && let Some(r) = src.read_raw(&nm, self.mode.mmap).map_err(|e| MoeError::Load(e.to_string()))?
                    {
                        return quant(r, rows, cols, &nm);
                    }
                    src.read(&nm).map(Weight::Dense).map_err(|e| MoeError::Load(e.to_string()))
                }
                Source::Stacked { src, name } => {
                    let nm = name(layer, p);
                    let start = id.checked_mul(n).ok_or_else(|| MoeError::Shape("overflow".into()))?;
                    if self.mode.quantized
                        && let Some(r) = src.read_range_raw(&nm, start, rows, cols, self.mode.mmap)?
                    {
                        return quant(r, rows, cols, &nm);
                    }
                    src.read_range(&nm, start, n).map(Weight::Dense)
                }
            }
        };
        let w = ExpertWeights { gate: get(Part::Gate)?, up: get(Part::Up)?, down: get(Part::Down)? };
        w.validate(self.hidden, self.ffn)?;
        self.bytes_read.fetch_add(w.bytes(), Ordering::Relaxed);
        Ok(w)
    }

    fn lock_cache(&self) -> std::sync::MutexGuard<'_, Cache> { self.cache.lock().unwrap_or_else(|e| e.into_inner()) }

    fn cached(&self, key: (usize, usize)) -> Option<Arc<ExpertWeights>> {
        let mut c = self.lock_cache();
        c.tick += 1;
        let t = c.tick;
        c.map.get_mut(&key).map(|e| { e.1 = t; e.0.clone() })
    }

    fn insert(&self, key: (usize, usize), w: Arc<ExpertWeights>) {
        let size = w.bytes();
        if size > self.budget { return; }
        let mut c = self.lock_cache();
        if c.map.contains_key(&key) { return; }
        while c.bytes + size > self.budget {
            let Some(victim) = c.map.iter().min_by_key(|(k, v)| (v.1, **k)).map(|(k, _)| *k) else { break };
            if let Some((old, _)) = c.map.remove(&victim) { c.bytes -= old.bytes(); self.evictions.fetch_add(1, Ordering::Relaxed); }
        }
        c.tick += 1;
        let t = c.tick;
        c.bytes += size;
        c.map.insert(key, (w, t));
    }

    fn load(&self, layer: usize, id: usize, demand: bool) -> Result<Arc<ExpertWeights>, MoeError> {
        if let Some(w) = self.cached((layer, id)) {
            if demand { self.hits.fetch_add(1, Ordering::Relaxed); }
            return Ok(w);
        }
        let w = Arc::new(self.read_expert(layer, id)?);
        if demand { self.misses.fetch_add(1, Ordering::Relaxed); } else { self.prefetched.fetch_add(1, Ordering::Relaxed); }
        self.insert((layer, id), w.clone());
        Ok(w)
    }

    fn try_add_worker(&self) -> bool {
        let mut cur = self.workers.load(Ordering::SeqCst);
        loop {
            if cur >= MAX_WORKERS { return false; }
            match self.workers.compare_exchange(cur, cur + 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return true,
                Err(n) => cur = n,
            }
        }
    }
    fn queue_len(&self) -> usize { self.queue.lock().unwrap_or_else(|e| e.into_inner()).len() }
}

fn worker(inner: Arc<Inner>) {
    loop {
        let job = inner.queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match job {
            Some((l, i)) => { let _ = inner.load(l, i, false); }
            None => {
                inner.workers.fetch_sub(1, Ordering::SeqCst);
                // a job may have been queued between the empty pop and the decrement
                if inner.queue_len() > 0 && inner.try_add_worker() { continue; }
                return;
            }
        }
    }
}

/// Experts read on demand, kept in a byte-budgeted LRU cache.
pub struct StreamedExperts { inner: Arc<Inner> }

impl StreamedExperts {
    fn make(source: Source, hidden: usize, ffn: usize, budget: u64, mode: ReadMode) -> Self {
        StreamedExperts { inner: Arc::new(Inner {
            source, hidden, ffn, budget, mode, cache: Mutex::new(Cache::default()),
            hits: 0.into(), misses: 0.into(), evictions: 0.into(), bytes_read: 0.into(), prefetched: 0.into(),
            queue: Mutex::new(VecDeque::new()), workers: 0.into(),
        }) }
    }
    /// One tensor per expert part (`names::mixtral`, `names::hf_experts`, …). `budget` in bytes
    /// (use `Plan::expert_cache`).
    pub fn per_expert(src: Arc<dyn TensorSource + Send + Sync>, name: impl Fn(usize, usize, Part) -> String + Send + Sync + 'static,
        hidden: usize, ffn: usize, budget: u64) -> Self {
        Self::per_expert_with(src, name, hidden, ffn, budget, ReadMode::default())
    }
    /// [`StreamedExperts::per_expert`] with control over how experts are held ([`ReadMode`]).
    pub fn per_expert_with(src: Arc<dyn TensorSource + Send + Sync>, name: impl Fn(usize, usize, Part) -> String + Send + Sync + 'static,
        hidden: usize, ffn: usize, budget: u64, mode: ReadMode) -> Self {
        Self::make(Source::PerExpert { src, name: Box::new(name) }, hidden, ffn, budget, mode)
    }
    /// One stacked tensor per part, experts contiguous (`names::gguf_stacked`).
    pub fn stacked(src: Arc<dyn RangeSource>, name: impl Fn(usize, Part) -> String + Send + Sync + 'static,
        hidden: usize, ffn: usize, budget: u64) -> Self {
        Self::stacked_with(src, name, hidden, ffn, budget, ReadMode::default())
    }
    /// [`StreamedExperts::stacked`] with control over how experts are held ([`ReadMode`]).
    pub fn stacked_with(src: Arc<dyn RangeSource>, name: impl Fn(usize, Part) -> String + Send + Sync + 'static,
        hidden: usize, ffn: usize, budget: u64, mode: ReadMode) -> Self {
        Self::make(Source::Stacked { src, name: Box::new(name) }, hidden, ffn, budget, mode)
    }
    /// Read every expert of the given layers now (used when the whole model fits in memory).
    pub fn preload(&self, layers: impl IntoIterator<Item = usize>, experts: usize) -> Result<(), MoeError> {
        for l in layers { for e in 0..experts { self.inner.load(l, e, false)?; } }
        Ok(())
    }
    pub fn stats(&self) -> Stats {
        let i = &self.inner;
        Stats { hits: i.hits.load(Ordering::Relaxed), misses: i.misses.load(Ordering::Relaxed), evictions: i.evictions.load(Ordering::Relaxed),
            bytes_read: i.bytes_read.load(Ordering::Relaxed), prefetched: i.prefetched.load(Ordering::Relaxed), resident_bytes: i.lock_cache().bytes }
    }
    pub fn budget(&self) -> u64 { self.inner.budget }
    /// Block until the prefetch queue is drained and idle (bounded wait, for tests and shutdown).
    pub fn wait_idle(&self) {
        for _ in 0..10_000 {
            if self.inner.workers.load(Ordering::SeqCst) == 0 && self.inner.queue_len() == 0 { return; }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

impl ExpertProvider for StreamedExperts {
    fn expert(&self, layer: usize, id: usize) -> Result<Arc<ExpertWeights>, MoeError> { self.inner.load(layer, id, true) }
    fn stats(&self) -> Option<Stats> { Some(StreamedExperts::stats(self)) }
    fn resident_bytes(&self) -> u64 { self.inner.lock_cache().bytes }

    fn prefetch(&self, layer: usize, ids: &[usize]) {
        let inner = &self.inner;
        {
            let mut q = inner.queue.lock().unwrap_or_else(|e| e.into_inner());
            for &id in ids {
                if q.len() >= QUEUE_CAP { break; }
                if !q.contains(&(layer, id)) { q.push_back((layer, id)); }
            }
        }
        while inner.queue_len() > 0 && inner.try_add_worker() {
            let w = inner.clone();
            if std::thread::Builder::new().name("moe-prefetch".into()).spawn(move || worker(w)).is_err() {
                inner.workers.fetch_sub(1, Ordering::SeqCst);
                break;
            }
        }
    }
}

// ------------------------------------------------------------------------- block

#[derive(Clone)]
pub struct MoeBlock {
    pub layer: usize,
    pub hidden: usize,
    pub router: Router,
    /// Always-on experts (any ffn width); their outputs are added to the routed ones.
    pub shared: Vec<ExpertWeights>,
    /// Qwen2-MoE: the shared experts' output is scaled by `sigmoid(gate · x)` per token.
    pub shared_gate: Option<Vec<f32>>,
}

/// Which tokens go to which expert, with their routing weights. Experts are in ascending id order, so
/// results never depend on batch composition.
#[derive(Clone, Debug, Default)]
pub struct Routing { groups: BTreeMap<usize, Vec<(usize, f32)>> }
impl Routing {
    /// The experts any token of the batch picked, ascending.
    pub fn experts(&self) -> Vec<usize> { self.groups.keys().copied().collect() }
}

impl MoeBlock {
    pub fn new(layer: usize, router: Router, shared: Vec<ExpertWeights>) -> Result<MoeBlock, MoeError> {
        let hidden = router.config().hidden;
        for s in &shared {
            let f = s.ffn(hidden);
            if f == 0 { return Err(MoeError::Shape("empty shared expert".into())); }
            s.validate(hidden, f)?;
        }
        Ok(MoeBlock { layer, hidden, router, shared, shared_gate: None })
    }
    /// Add the sigmoid gate on the shared experts (`hidden` values).
    pub fn with_shared_gate(mut self, gate: Vec<f32>) -> Result<MoeBlock, MoeError> {
        if gate.len() != self.hidden { return Err(MoeError::Shape(format!("shared gate has {} values, expected {}", gate.len(), self.hidden))); }
        self.shared_gate = Some(gate);
        Ok(self)
    }
    /// Bytes of the always-resident parts (router, shared experts).
    pub fn stored_bytes(&self) -> usize {
        4 * (self.router.weight.len() + self.shared_gate.as_ref().map_or(0, Vec::len)) + self.shared.iter().map(|e| e.bytes() as usize).sum::<usize>()
    }

    pub fn forward(&self, p: &dyn ExpertProvider, x: &[f32]) -> Result<Vec<f32>, MoeError> { self.forward_batch(p, x, 1) }

    /// Routing for `n` tokens of `hidden` values.
    pub fn route_batch(&self, xs: &[f32], n: usize) -> Result<Routing, MoeError> {
        let h = self.hidden;
        if n.checked_mul(h) != Some(xs.len()) { return Err(MoeError::Shape(format!("{} values for {n} tokens of {h}", xs.len()))); }
        let mut groups: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
        for (t, x) in xs.chunks_exact(h.max(1)).enumerate().take(n) {
            for (e, w) in self.router.route(x)? { groups.entry(e).or_default().push((t, w)); }
        }
        Ok(Routing { groups })
    }

    /// Run the experts for an already computed [`Routing`]; every expert is fetched once.
    pub fn apply(&self, p: &dyn ExpertProvider, act: Act, xs: &[f32], n: usize, routing: &Routing) -> Result<Vec<f32>, MoeError> {
        let h = self.hidden;
        if n.checked_mul(h) != Some(xs.len()) { return Err(MoeError::Shape(format!("{} values for {n} tokens of {h}", xs.len()))); }
        p.prefetch(self.layer, &routing.experts());
        let mut out = vec![0.0; xs.len()];
        let mut sc = ExpertScratch::default();
        for s in &self.shared {
            s.run(h, act, xs, n, &mut sc)?;
            for (t, (o, d)) in out.chunks_exact_mut(h).zip(sc.d.chunks_exact(h)).enumerate() {
                let g = match &self.shared_gate {
                    Some(g) => 1.0 / (1.0 + (-g.iter().zip(&xs[t * h..(t + 1) * h]).map(|(a, b)| a * b).sum::<f32>()).exp()),
                    None => 1.0,
                };
                o.iter_mut().zip(d).for_each(|(o, v)| *o += g * v);
            }
        }
        let mut gathered = Vec::new();
        for (&e, toks) in &routing.groups {
            let w = p.expert(self.layer, e)?;
            let ffn = w.ffn(h);
            w.validate(h, ffn).and_then(|_| if ffn == 0 { Err(MoeError::Shape("empty expert".into())) } else { Ok(()) })?;
            gathered.clear();
            for &(t, _) in toks { gathered.extend_from_slice(xs.get(t * h..(t + 1) * h).ok_or_else(|| MoeError::Shape("routing names a token outside the batch".into()))?); }
            w.run(h, act, &gathered, toks.len(), &mut sc)?;
            for ((t, wt), d) in toks.iter().zip(sc.d.chunks_exact(h)) {
                out[t * h..(t + 1) * h].iter_mut().zip(d).for_each(|(o, v)| *o += wt * v);
            }
        }
        if out.iter().any(|v| !v.is_finite()) { return Err(MoeError::NonFinite); }
        Ok(out)
    }

    /// `xs` holds `n` tokens of `hidden` values (SiLU experts). Tokens are grouped by expert (ascending
    /// expert id, so results never depend on batch composition) and every expert is fetched once.
    pub fn forward_batch(&self, p: &dyn ExpertProvider, xs: &[f32], n: usize) -> Result<Vec<f32>, MoeError> {
        let r = self.route_batch(xs, n)?;
        self.apply(p, Act::Silu, xs, n, &r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::LoadError;

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.8 }
        fn v(&mut self, n: usize) -> Vec<f32> { (0..n).map(|_| self.f()).collect() }
    }
    const H: usize = 8;
    const F: usize = 6;
    const E: usize = 8;

    fn experts(r: &mut Rng, layers: usize) -> Vec<Vec<ExpertWeights>> {
        (0..layers).map(|_| (0..E).map(|_| ExpertWeights::dense(r.v(H * F), r.v(H * F), r.v(H * F))).collect()).collect()
    }
    fn block(r: &mut Rng, k: usize, shared: bool) -> MoeBlock {
        let router = Router::new(RouterConfig::softmax(H, E, k), r.v(E * H)).unwrap();
        let sh = if shared { vec![ExpertWeights::dense(r.v(H * 10), r.v(H * 10), r.v(H * 10))] } else { vec![] };
        MoeBlock::new(0, router, sh).unwrap()
    }

    /// name → tensor, counting reads.
    struct Fake { t: HashMap<String, Vec<f32>>, reads: AtomicUsize }
    impl TensorSource for Fake {
        fn has(&self, n: &str) -> bool { self.t.contains_key(n) }
        fn read(&self, n: &str) -> Result<Vec<f32>, LoadError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.t.get(n).cloned().ok_or_else(|| LoadError::Missing(n.into()))
        }
    }
    fn per_expert_fake(ex: &[Vec<ExpertWeights>]) -> Fake {
        let mut t = HashMap::new();
        for (l, es) in ex.iter().enumerate() {
            for (i, e) in es.iter().enumerate() {
                t.insert(names::mixtral(l, i, Part::Gate), e.gate.to_f32());
                t.insert(names::mixtral(l, i, Part::Up), e.up.to_f32());
                t.insert(names::mixtral(l, i, Part::Down), e.down.to_f32());
            }
        }
        Fake { t, reads: 0.into() }
    }
    fn stacked_fake(ex: &[Vec<ExpertWeights>]) -> Fake {
        let mut t = HashMap::new();
        for (l, es) in ex.iter().enumerate() {
            t.insert(names::gguf_stacked(l, Part::Gate), es.iter().flat_map(|e| e.gate.to_f32()).collect());
            t.insert(names::gguf_stacked(l, Part::Up), es.iter().flat_map(|e| e.up.to_f32()).collect());
            t.insert(names::gguf_stacked(l, Part::Down), es.iter().flat_map(|e| e.down.to_f32()).collect());
        }
        Fake { t, reads: 0.into() }
    }
    fn tokens(r: &mut Rng, n: usize) -> Vec<f32> { r.v(n * H) }
    const ONE: u64 = (3 * H * F * 4) as u64;

    #[test]
    fn top_k_matches_brute_force_including_ties() {
        let mut r = Rng(7);
        for round in 0..300 {
            let n = 1 + round % 17;
            let s: Vec<f32> = (0..n).map(|_| (r.f() * 4.0).round()).collect(); // many ties
            for k in 0..=n {
                let mut left: Vec<usize> = (0..n).collect();
                let mut want = vec![];
                for _ in 0..k {
                    let best = *left.iter().fold(&left[0], |b, i| if s[*i] > s[*b] { i } else { b });
                    want.push(best);
                    left.retain(|i| *i != best);
                }
                assert_eq!(top_k(&s, k), want, "scores {s:?} k {k}");
            }
        }
    }

    #[test]
    fn renormalised_weights_sum_to_one_and_plain_softmax_does_not() {
        let mut r = Rng(3);
        let w = r.v(E * H);
        let a = Router::new(RouterConfig::softmax(H, E, 3), w.clone()).unwrap();
        let mut cfg = RouterConfig::softmax(H, E, 3);
        cfg.renormalise = false;
        let b = Router::new(cfg, w).unwrap();
        for _ in 0..20 {
            let x = r.v(H);
            let (ra, rb) = (a.route(&x).unwrap(), b.route(&x).unwrap());
            assert!((ra.iter().map(|p| p.1).sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(rb.iter().map(|p| p.1).sum::<f32>() < 1.0);
            assert_eq!(ra.iter().map(|p| p.0).collect::<Vec<_>>(), rb.iter().map(|p| p.0).collect::<Vec<_>>());
            assert!(ra.windows(2).all(|w| w[0].1 >= w[1].1));
        }
    }

    #[test]
    fn grouped_sigmoid_routing_respects_groups_and_bias_only_changes_selection() {
        let mut r = Rng(11);
        let w = r.v(E * H);
        let mut cfg = RouterConfig::softmax(H, E, 2);
        cfg.scoring = Scoring::Sigmoid;
        cfg.renormalise = false;
        cfg.groups = Some((4, 1)); // 4 groups of 2, keep 1 group → both picks inside one group
        let plain = Router::new(cfg.clone(), w.clone()).unwrap();
        cfg.selection_bias = Some(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0]);
        let biased = Router::new(cfg, w).unwrap();
        for _ in 0..20 {
            let x = r.v(H);
            let p = plain.route(&x).unwrap();
            assert_eq!(p[0].0 / 2, p[1].0 / 2, "both experts in the same group");
            let b = biased.route(&x).unwrap();
            assert_eq!(b.iter().map(|e| e.0).collect::<std::collections::BTreeSet<_>>(), [6, 7].into());
            for (e, wt) in &b { // weights are the raw sigmoid scores, not score+bias
                assert!(*wt > 0.0 && *wt < 1.0, "expert {e} weight {wt}");
            }
        }
        let mut cfg = RouterConfig::softmax(H, E, 2);
        cfg.scoring = Scoring::Sigmoid;
        cfg.routed_scale = 2.5;
        let scaled = Router::new(cfg, plain.weight.clone()).unwrap();
        let x = r.v(H);
        assert!((scaled.route(&x).unwrap().iter().map(|p| p.1).sum::<f32>() - 2.5).abs() < 1e-4);
    }

    #[test]
    fn hostile_configs_and_inputs_are_errors_not_panics() {
        let ok = |k, e| Router::new(RouterConfig::softmax(H, e, k), vec![0.1; e * H]);
        assert!(matches!(ok(9, 8), Err(MoeError::Config(_))));
        assert!(matches!(ok(0, 8), Err(MoeError::Config(_))));
        assert!(matches!(ok(1, 0), Err(MoeError::Config(_))));
        assert!(matches!(Router::new(RouterConfig::softmax(H, E, 2), vec![0.0; 3]), Err(MoeError::Shape(_))));
        let mut c = RouterConfig::softmax(H, E, 2);
        c.groups = Some((3, 1));
        assert!(Router::new(c.clone(), vec![0.0; E * H]).is_err());
        c.groups = Some((4, 5));
        assert!(Router::new(c.clone(), vec![0.0; E * H]).is_err());
        c.groups = Some((4, 1));
        c.top_k = 3; // only 2 experts survive one group
        assert!(Router::new(c, vec![0.0; E * H]).is_err());
        let r = ok(2, 8).unwrap();
        assert_eq!(r.route(&[f32::NAN; H]), Err(MoeError::NonFinite));
        assert_eq!(r.route(&[f32::INFINITY; H]), Err(MoeError::NonFinite));
        assert!(matches!(r.route(&[0.0; 3]), Err(MoeError::Shape(_))));
        // deterministic with all-equal logits: lowest indices win
        let z = Router::new(RouterConfig::softmax(H, E, 3), vec![0.0; E * H]).unwrap();
        assert_eq!(z.route(&[1.0; H]).unwrap().iter().map(|p| p.0).collect::<Vec<_>>(), vec![0, 1, 2]);
        let mut rng = Rng(1);
        let b = block(&mut rng, 2, false);
        let p = InMemoryExperts::new(experts(&mut rng, 1));
        assert!(b.forward_batch(&p, &[0.0; 5], 1).is_err());
        assert!(b.forward(&p, &[f32::NAN; H]).is_err());
        assert!(matches!(MoeBlock { layer: 3, ..block(&mut rng, 2, false) }.forward(&p, &[0.1; H]), Err(MoeError::Missing(_))));
    }

    #[test]
    fn block_matches_an_independent_dense_reference_and_batching_is_exact() {
        let mut r = Rng(21);
        let ex = experts(&mut r, 1);
        let b = block(&mut r, 3, true);
        let p = InMemoryExperts::new(ex.clone());
        let xs = tokens(&mut r, 5);
        let batch = b.forward_batch(&p, &xs, 5).unwrap();
        for t in 0..5 {
            let x = &xs[t * H..(t + 1) * H];
            let single = b.forward(&p, x).unwrap();
            assert_eq!(single, &batch[t * H..(t + 1) * H], "batch must not change a token's result");
            // reference: softmax over all experts, brute-force top-3, renormalise, plain loops
            let mut logits = vec![0.0f32; E];
            for (e, l) in logits.iter_mut().enumerate() { *l = (0..H).map(|i| b.router.weight[e * H + i] * x[i]).sum(); }
            softmax(&mut logits);
            let mut order: Vec<usize> = (0..E).collect();
            order.sort_by(|a, c| logits[*c].total_cmp(&logits[*a]).then(a.cmp(c)));
            let sel = &order[..3];
            let total: f32 = sel.iter().map(|e| logits[*e]).sum();
            let ffn_out = |w: &ExpertWeights, f: usize| -> Vec<f32> {
                let (wg, wu, wd) = (w.gate.to_f32(), w.up.to_f32(), w.down.to_f32());
                let a: Vec<f32> = (0..f).map(|j| {
                    let g: f32 = (0..H).map(|i| wg[j * H + i] * x[i]).sum();
                    let u: f32 = (0..H).map(|i| wu[j * H + i] * x[i]).sum();
                    crate::ops::silu(g) * u
                }).collect();
                (0..H).map(|i| (0..f).map(|j| wd[i * f + j] * a[j]).sum()).collect()
            };
            let mut want = ffn_out(&b.shared[0], 10);
            for &e in sel { for (o, v) in want.iter_mut().zip(ffn_out(&ex[0][e], F)) { *o += logits[e] / total * v; } }
            for (g, w) in single.iter().zip(&want) { assert!((g - w).abs() < 1e-4, "{g} vs {w}"); }
        }
    }

    #[test]
    fn streamed_provider_equals_in_memory_even_with_a_cache_that_forces_eviction() {
        let mut r = Rng(5);
        let ex = experts(&mut r, 2);
        let b = MoeBlock { layer: 1, ..block(&mut r, 3, false) };
        let mem = InMemoryExperts::new(ex.clone());
        let fake = Arc::new(per_expert_fake(&ex));
        let stacked = Arc::new(StackedFromTensors(stacked_fake(&ex)));
        let tiny = 2 * ONE; // room for two experts only
        let s1 = StreamedExperts::per_expert(fake.clone(), names::mixtral, H, F, tiny);
        let s2 = StreamedExperts::stacked(stacked, names::gguf_stacked, H, F, tiny);
        for round in 0..6 {
            let xs = tokens(&mut r, 1 + round);
            let n = 1 + round;
            let want = b.forward_batch(&mem, &xs, n).unwrap();
            assert_eq!(b.forward_batch(&s1, &xs, n).unwrap(), want);
            assert_eq!(b.forward_batch(&s2, &xs, n).unwrap(), want);
            for s in [&s1, &s2] { assert!(s.stats().resident_bytes <= tiny, "cache over budget"); }
        }
        let st = s1.stats();
        assert!(st.evictions > 0 && st.misses > 0 && st.bytes_read >= st.misses * ONE, "{st:?}");
    }

    #[test]
    fn cache_hits_avoid_rereads_and_oversized_experts_are_not_cached() {
        let mut r = Rng(9);
        let ex = experts(&mut r, 1);
        let fake = Arc::new(per_expert_fake(&ex));
        let s = StreamedExperts::per_expert(fake.clone(), names::mixtral, H, F, 10 * ONE);
        s.expert(0, 3).unwrap();
        let after_first = fake.reads.load(Ordering::Relaxed);
        assert_eq!(after_first, 3);
        for _ in 0..5 { s.expert(0, 3).unwrap(); }
        assert_eq!(fake.reads.load(Ordering::Relaxed), after_first);
        let st = s.stats();
        assert_eq!((st.hits, st.misses, st.resident_bytes), (5, 1, ONE));
        let none = StreamedExperts::per_expert(fake, names::mixtral, H, F, ONE - 1);
        none.expert(0, 1).unwrap();
        none.expert(0, 1).unwrap();
        assert_eq!((none.stats().resident_bytes, none.stats().misses), (0, 2));
    }

    #[test]
    fn missing_and_misshapen_tensors_are_errors() {
        let mut r = Rng(2);
        let ex = experts(&mut r, 1);
        let mut f = per_expert_fake(&ex);
        f.t.remove(&names::mixtral(0, 2, Part::Up));
        f.t.get_mut(&names::mixtral(0, 4, Part::Down)).unwrap().pop();
        let s = StreamedExperts::per_expert(Arc::new(f), names::mixtral, H, F, 100 * ONE);
        assert!(matches!(s.expert(0, 2), Err(MoeError::Missing(m)) if m.contains("experts.2.w3")));
        assert!(matches!(s.expert(0, 4), Err(MoeError::Shape(_))));
        assert!(s.expert(0, 1).is_ok());
        let st = StreamedExperts::stacked(Arc::new(StackedFromTensors(stacked_fake(&ex))), names::gguf_stacked, H, F, ONE);
        assert!(st.expert(0, E).is_err(), "expert id past the stacked tensor");
        assert!(st.expert(1, 0).is_err(), "layer that does not exist");
    }

    #[test]
    fn prefetcher_is_bounded_concurrent_and_changes_no_result() {
        let mut r = Rng(13);
        let ex = experts(&mut r, 1);
        let b = block(&mut r, 2, false);
        let mem = InMemoryExperts::new(ex.clone());
        let s = Arc::new(StreamedExperts::per_expert(Arc::new(per_expert_fake(&ex)), names::mixtral, H, F, 100 * ONE));
        // a flood of requests from several threads, far past the queue cap
        let hs: Vec<_> = (0..4).map(|t| { let s = s.clone(); std::thread::spawn(move || {
            for i in 0..200 { s.prefetch(0, &[(i + t) % E, (i * 3 + t) % E, 1000 + i]); }
        }) }).collect();
        // meanwhile use the provider on this thread
        let xs = tokens(&mut r, 4);
        let want = b.forward_batch(&mem, &xs, 4).unwrap();
        assert_eq!(b.forward_batch(&*s, &xs, 4).unwrap(), want);
        for h in hs { h.join().unwrap(); }
        s.wait_idle();
        assert_eq!(s.inner.workers.load(Ordering::SeqCst), 0);
        assert_eq!(s.inner.queue_len(), 0);
        assert!(s.stats().prefetched + s.stats().misses >= E as u64 - 1);
        let before = s.stats().misses;
        for e in 0..E { s.expert(0, e).unwrap(); }
        assert_eq!(s.stats().misses, before, "everything was already resident");
        assert_eq!(b.forward_batch(&*s, &xs, 4).unwrap(), want);
        // queue never grows past its cap
        s.prefetch(0, &(0..1000).collect::<Vec<_>>());
        assert!(s.inner.queue_len() <= QUEUE_CAP);
        s.wait_idle();
    }

    #[test]
    fn prefetch_makes_the_next_demand_a_hit() {
        let mut r = Rng(17);
        let ex = experts(&mut r, 1);
        let s = StreamedExperts::per_expert(Arc::new(per_expert_fake(&ex)), names::mixtral, H, F, 100 * ONE);
        s.prefetch(0, &[2, 5]);
        s.wait_idle();
        s.expert(0, 2).unwrap();
        s.expert(0, 5).unwrap();
        let st = s.stats();
        assert_eq!((st.hits, st.misses, st.prefetched), (2, 0, 2));
    }

    #[test]
    fn name_conventions() {
        assert_eq!(names::mixtral(3, 7, Part::Gate), "model.layers.3.block_sparse_moe.experts.7.w1.weight");
        assert_eq!(names::mixtral(3, 7, Part::Down), "model.layers.3.block_sparse_moe.experts.7.w2.weight");
        assert_eq!(names::hf_experts(0, 1, Part::Up), "model.layers.0.mlp.experts.1.up_proj.weight");
        assert_eq!(names::gguf_stacked(2, Part::Down), "blk.2.ffn_down_exps.weight");
    }
}
