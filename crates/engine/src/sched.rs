//! Continuous batching, paged KV memory and prefix caching — the engine-independent core.
//!
//! Everything here is pure bookkeeping: no tensors, no model. A model executor receives a
//! [`StepPlan`] (which tokens of which sequences to run, at which positions, through which
//! block table), runs it, and returns the sampled tokens as a [`StepResult`]. That keeps the
//! scheduler testable with a mock executor that reads and writes a fake paged KV memory and
//! so catches any wrong sharing, copy-on-write or recompute bug end to end.
//!
//! Layers, bottom up:
//! * [`BlockAllocator`] – fixed-size KV blocks with reference counts and copy-on-write.
//! * [`PrefixCache`]    – hash-chained full blocks (a radix tree in disguise) with LRU eviction;
//!   blocks used by a live sequence are pinned by their reference count.
//! * [`Scheduler`]      – priority queue with aging, `n_parallel` slots, chunked prefill inside a
//!   per-step token budget, preemption by recompute, cancellation, deadlines, statistics.
//!
//! Time is logical: callers pass `now` (any monotone unit, e.g. milliseconds) and deadlines use
//! the same unit. Nothing here reads a clock.
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub type SeqId = u64;
pub type BlockId = u32;

/// Upper bound on blocks in one pool (keeps hostile budgets from allocating gigabytes of bookkeeping).
pub const MAX_BLOCKS: usize = 1 << 22;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedError {
    BadConfig(&'static str),
    /// A single sequence of the requested size can never fit the pool.
    BudgetTooSmall { needed_blocks: u64, total_blocks: u64 },
    EmptyPrompt,
    ZeroNewTokens,
    PromptTooLong { len: usize, max: usize },
    QueueFull,
    OutOfBlocks,
    DoubleFree(BlockId),
    BadBlock(BlockId),
    UnknownSeq(SeqId),
    NotForkable(&'static str),
    NoSlot,
    BadResult(String),
    StalePlan,
}
impl std::fmt::Display for SchedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use SchedError::*;
        match self {
            BadConfig(w) => write!(f, "bad scheduler configuration: {w}"),
            BudgetTooSmall { needed_blocks, total_blocks } => {
                write!(f, "the request needs {needed_blocks} KV blocks but the pool has only {total_blocks}")
            }
            EmptyPrompt => write!(f, "the prompt is empty"),
            ZeroNewTokens => write!(f, "max_new_tokens is zero"),
            PromptTooLong { len, max } => write!(f, "the prompt has {len} tokens but the context holds {max}"),
            QueueFull => write!(f, "the request queue is full"),
            OutOfBlocks => write!(f, "no free KV block"),
            DoubleFree(b) => write!(f, "block {b} was freed twice"),
            BadBlock(b) => write!(f, "block {b} does not exist or is not in use"),
            UnknownSeq(s) => write!(f, "unknown sequence {s}"),
            NotForkable(w) => write!(f, "cannot fork: {w}"),
            NoSlot => write!(f, "no free slot"),
            BadResult(w) => write!(f, "inconsistent executor result: {w}"),
            StalePlan => write!(f, "the plan does not match the scheduler's current state"),
        }
    }
}
impl std::error::Error for SchedError {}

// ------------------------------------------------------------------------ allocator

/// A pool of fixed-size KV blocks with reference counts.
#[derive(Debug, Clone)]
pub struct BlockAllocator {
    block_tokens: usize,
    refs: Vec<u32>,
    /// Stack of free ids; the lowest id is popped first so runs are deterministic.
    free: Vec<BlockId>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fragmentation {
    pub used: usize,
    pub free: usize,
    /// Longest run of consecutive free block ids.
    pub largest_free_run: usize,
    /// `1 - largest_free_run / free` (0 when nothing is free).
    pub ratio: f64,
}

impl BlockAllocator {
    /// Number of blocks a byte budget buys (capped at [`MAX_BLOCKS`]).
    pub fn blocks_for_budget(block_tokens: usize, bytes_per_token: u64, budget_bytes: u64) -> Result<usize, SchedError> {
        if block_tokens == 0 { return Err(SchedError::BadConfig("block_tokens is zero")); }
        if bytes_per_token == 0 { return Err(SchedError::BadConfig("bytes_per_token is zero")); }
        let per_block = block_tokens as u128 * bytes_per_token as u128;
        let total = budget_bytes as u128 / per_block;
        if total == 0 { return Err(SchedError::BudgetTooSmall { needed_blocks: 1, total_blocks: 0 }); }
        Ok(total.min(MAX_BLOCKS as u128) as usize)
    }
    pub fn with_blocks(block_tokens: usize, total: usize) -> Result<BlockAllocator, SchedError> {
        if block_tokens == 0 { return Err(SchedError::BadConfig("block_tokens is zero")); }
        if total == 0 { return Err(SchedError::BudgetTooSmall { needed_blocks: 1, total_blocks: 0 }); }
        if total > MAX_BLOCKS { return Err(SchedError::BadConfig("too many blocks")); }
        Ok(BlockAllocator { block_tokens, refs: vec![0; total], free: (0..total as BlockId).rev().collect() })
    }
    pub fn block_tokens(&self) -> usize { self.block_tokens }
    pub fn total(&self) -> usize { self.refs.len() }
    pub fn free_count(&self) -> usize { self.free.len() }
    pub fn used(&self) -> usize { self.refs.len() - self.free.len() }
    pub fn ref_count(&self, b: BlockId) -> u32 { self.refs.get(b as usize).copied().unwrap_or(0) }
    pub fn is_shared(&self, b: BlockId) -> bool { self.ref_count(b) > 1 }

    /// A fresh block with one reference, or `None` when the pool is empty.
    pub fn alloc(&mut self) -> Option<BlockId> {
        let b = self.free.pop()?;
        self.refs[b as usize] = 1;
        Some(b)
    }
    /// Allocate, asking `evict` to free something while the pool is empty. `evict` returns
    /// `false` when it has nothing left to give.
    pub fn alloc_with_evict(&mut self, evict: &mut dyn FnMut(&mut BlockAllocator) -> bool) -> Option<BlockId> {
        loop {
            if let Some(b) = self.alloc() { return Some(b); }
            if !evict(self) { return None; }
        }
    }
    pub fn retain(&mut self, b: BlockId) -> Result<(), SchedError> {
        match self.refs.get_mut(b as usize) {
            Some(r) if *r > 0 => { *r += 1; Ok(()) }
            _ => Err(SchedError::BadBlock(b)),
        }
    }
    /// Drop one reference; returns `true` when the block went back to the pool.
    pub fn release(&mut self, b: BlockId) -> Result<bool, SchedError> {
        match self.refs.get_mut(b as usize) {
            None => Err(SchedError::BadBlock(b)),
            Some(0) => Err(SchedError::DoubleFree(b)),
            Some(r) => {
                *r -= 1;
                if *r == 0 { self.free.push(b); Ok(true) } else { Ok(false) }
            }
        }
    }
    /// Share a whole block table (a forked sequence): one more reference on every block.
    pub fn fork(&mut self, blocks: &[BlockId]) -> Result<Vec<BlockId>, SchedError> {
        for (i, &b) in blocks.iter().enumerate() {
            if let Err(e) = self.retain(b) {
                for &done in &blocks[..i] { let _ = self.release(done); }
                return Err(e);
            }
        }
        Ok(blocks.to_vec())
    }
    /// Copy-on-write: a block with one owner is returned as is; a shared one is replaced by a
    /// fresh block and the caller must copy the data. Returns `(block, copied)`.
    pub fn make_unique(&mut self, b: BlockId) -> Result<(BlockId, bool), SchedError> {
        match self.ref_count(b) {
            0 => Err(SchedError::BadBlock(b)),
            1 => Ok((b, false)),
            _ => {
                let nb = self.alloc().ok_or(SchedError::OutOfBlocks)?;
                self.refs[b as usize] -= 1;
                Ok((nb, true))
            }
        }
    }
    pub fn largest_free_run(&self) -> usize {
        let mut ids: Vec<BlockId> = self.free.clone();
        ids.sort_unstable();
        let (mut best, mut run, mut prev) = (0usize, 0usize, None::<BlockId>);
        for id in ids {
            run = if prev.is_some_and(|p| p + 1 == id) { run + 1 } else { 1 };
            best = best.max(run);
            prev = Some(id);
        }
        best
    }
    pub fn fragmentation(&self) -> Fragmentation {
        let free = self.free.len();
        let run = self.largest_free_run();
        Fragmentation { used: self.used(), free, largest_free_run: run, ratio: if free == 0 { 0.0 } else { 1.0 - run as f64 / free as f64 } }
    }
}

// ------------------------------------------------------------------------ prefix cache

fn chain_hash(parent: Option<u64>, tokens: &[u32]) -> u64 {
    let mut h = parent.map_or(0xcbf2_9ce4_8422_2325, |p| p.rotate_left(17) ^ 0x9e37_79b9_7f4a_7c15);
    for &t in tokens {
        h ^= t as u64 + 1;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

#[derive(Debug, Clone)]
struct Entry {
    block: BlockId,
    parent: Option<u64>,
    tokens: Vec<u32>,
    children: u32,
    tick: u64,
}

/// Maps prefixes of token sequences, in whole blocks, to the KV blocks that hold them.
/// The cache owns one reference on every block it lists; a block that a live sequence also
/// holds therefore has a reference count above one and is never evicted.
#[derive(Debug, Clone)]
pub struct PrefixCache {
    bt: usize,
    map: HashMap<u64, Entry>,
    lru: BTreeSet<(u64, u64)>,
    tick: u64,
    pub hit_tokens: u64,
    pub lookup_tokens: u64,
}

impl PrefixCache {
    pub fn new(block_tokens: usize) -> PrefixCache {
        PrefixCache { bt: block_tokens.max(1), map: HashMap::new(), lru: BTreeSet::new(), tick: 0, hit_tokens: 0, lookup_tokens: 0 }
    }
    pub fn len(&self) -> usize { self.map.len() }
    pub fn is_empty(&self) -> bool { self.map.is_empty() }

    fn chain(&self, tokens: &[u32], max_blocks: usize) -> Vec<(u64, BlockId)> {
        let mut out = vec![];
        let mut parent = None;
        for chunk in tokens.chunks_exact(self.bt).take(max_blocks) {
            let h = chain_hash(parent, chunk);
            match self.map.get(&h) {
                Some(e) if e.parent == parent && e.tokens == chunk => { out.push((h, e.block)); parent = Some(h); }
                _ => break,
            }
        }
        out
    }
    /// Longest cached prefix of `tokens` (whole blocks, at most `max_blocks`), without side effects.
    pub fn peek(&self, tokens: &[u32], max_blocks: usize) -> Vec<BlockId> {
        self.chain(tokens, max_blocks).into_iter().map(|(_, b)| b).collect()
    }
    fn touch(&mut self, h: u64) {
        self.tick += 1;
        if let Some(e) = self.map.get_mut(&h) {
            self.lru.remove(&(e.tick, h));
            e.tick = self.tick;
            self.lru.insert((e.tick, h));
        }
    }
    /// Like [`peek`](Self::peek) but hands the blocks to the caller: each gets one more
    /// reference, the entries become most-recently-used, and the hit statistics move.
    pub fn acquire(&mut self, alloc: &mut BlockAllocator, tokens: &[u32], max_blocks: usize) -> Vec<BlockId> {
        let considered = (tokens.len() / self.bt).min(max_blocks);
        self.lookup_tokens += (considered * self.bt) as u64;
        let found = self.chain(tokens, max_blocks);
        let mut blocks = Vec::with_capacity(found.len());
        for &(h, b) in &found {
            if alloc.retain(b).is_err() { break; }
            blocks.push(b);
            self.touch(h);
        }
        self.hit_tokens += (blocks.len() * self.bt) as u64;
        blocks
    }
    /// Register the whole blocks of `tokens` (held in `blocks`). Already-known blocks are only
    /// refreshed; new ones get a cache reference. Returns how many entries were added.
    pub fn insert(&mut self, alloc: &mut BlockAllocator, tokens: &[u32], blocks: &[BlockId]) -> usize {
        let n = (tokens.len() / self.bt).min(blocks.len());
        let (mut parent, mut added) = (None, 0);
        for i in 0..n {
            let chunk = &tokens[i * self.bt..(i + 1) * self.bt];
            let h = chain_hash(parent, chunk);
            match self.map.get(&h) {
                Some(e) if e.parent == parent && e.tokens == chunk => { self.touch(h); }
                Some(_) => break, // hash collision: leave the rest uncached
                None => {
                    if alloc.retain(blocks[i]).is_err() { break; }
                    self.tick += 1;
                    self.map.insert(h, Entry { block: blocks[i], parent, tokens: chunk.to_vec(), children: 0, tick: self.tick });
                    self.lru.insert((self.tick, h));
                    if let Some(p) = parent.and_then(|p| self.map.get_mut(&p)) { p.children += 1; }
                    added += 1;
                }
            }
            parent = Some(h);
        }
        added
    }
    /// Drop the least-recently-used leaf that no sequence is using. Returns `false` if there is none.
    pub fn evict_one(&mut self, alloc: &mut BlockAllocator) -> bool {
        let victim = self.lru.iter().find(|&&(_, h)| {
            self.map.get(&h).is_some_and(|e| e.children == 0 && alloc.ref_count(e.block) == 1)
        }).copied();
        let Some((tick, h)) = victim else { return false };
        self.lru.remove(&(tick, h));
        if let Some(e) = self.map.remove(&h) {
            if let Some(p) = e.parent.and_then(|p| self.map.get_mut(&p)) { p.children -= 1; }
            let _ = alloc.release(e.block);
        }
        true
    }
    /// Blocks only the cache holds (what eviction can still give back).
    pub fn evictable(&self, alloc: &BlockAllocator) -> usize {
        self.map.values().filter(|e| alloc.ref_count(e.block) == 1).count()
    }
    pub fn hit_rate(&self) -> f64 {
        if self.lookup_tokens == 0 { 0.0 } else { self.hit_tokens as f64 / self.lookup_tokens as f64 }
    }
    fn blocks(&self) -> impl Iterator<Item = BlockId> + '_ { self.map.values().map(|e| e.block) }
}

// ------------------------------------------------------------------------ requests, plans, events

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub prompt: Vec<u32>,
    pub max_new_tokens: usize,
    /// Higher runs first. Waiting time raises the effective priority, so nothing starves.
    pub priority: i32,
    /// Absolute logical time after which the request is abandoned.
    pub deadline: Option<u64>,
    pub stop_tokens: Vec<u32>,
}
impl Request {
    pub fn new(prompt: Vec<u32>, max_new_tokens: usize) -> Request {
        Request { prompt, max_new_tokens, priority: 0, deadline: None, stop_tokens: vec![] }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind { Prefill, Decode }

/// One sequence's share of a step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepItem {
    pub seq: SeqId,
    pub tokens: Vec<u32>,
    /// Absolute positions of `tokens` in the sequence.
    pub positions: Vec<usize>,
    /// Every block of the sequence, in order; position `p` lives in `block_table[p / block_tokens]`.
    pub block_table: Vec<BlockId>,
    /// Whether the executor must return a sampled token (the last token of this item).
    pub sample: bool,
    pub kind: StepKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StepPlan {
    pub step: u64,
    /// Copy-on-write copies `(from, to)` the executor must perform before running the items.
    pub copies: Vec<(BlockId, BlockId)>,
    pub items: Vec<StepItem>,
}
impl StepPlan {
    pub fn total_tokens(&self) -> usize { self.items.iter().map(|i| i.tokens.len()).sum() }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StepResult {
    /// One sampled token per item that asked for one.
    pub sampled: Vec<(SeqId, u32)>,
}

pub trait Executor {
    fn run(&mut self, plan: &StepPlan) -> StepResult;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason { Length, Stop, Cancelled, Deadline, ContextFull }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Token { seq: SeqId, token: u32 },
    Finished { seq: SeqId, reason: FinishReason, generated: usize },
    /// The sequence lost its blocks and goes back to the queue; it will be recomputed.
    Preempted { seq: SeqId },
}

// ------------------------------------------------------------------------ statistics

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SchedStats {
    pub submitted: u64,
    pub completed: u64,
    pub cancelled: u64,
    pub expired: u64,
    pub preemptions: u64,
    pub cow_copies: u64,
    pub steps: u64,
    pub tokens_generated: u64,
    /// Sum of tokens run per step.
    pub batch_tokens: u64,
    /// Sum of the token budget of every non-empty step.
    pub batch_capacity: u64,
    pub prefix_hit_tokens: u64,
    pub prefix_lookup_tokens: u64,
    // gauges, filled by `Scheduler::stats`
    pub queue_depth: usize,
    pub running: usize,
    pub blocks_used: usize,
    pub blocks_total: usize,
    pub cached_blocks: usize,
}
impl SchedStats {
    pub fn batch_utilisation(&self) -> f64 {
        if self.batch_capacity == 0 { 0.0 } else { self.batch_tokens as f64 / self.batch_capacity as f64 }
    }
    pub fn prefix_hit_rate(&self) -> f64 {
        if self.prefix_lookup_tokens == 0 { 0.0 } else { self.prefix_hit_tokens as f64 / self.prefix_lookup_tokens as f64 }
    }
    pub fn tokens_per_second(&self, elapsed_secs: f64) -> f64 {
        if elapsed_secs > 0.0 { self.tokens_generated as f64 / elapsed_secs } else { 0.0 }
    }
    /// Prometheus text exposition, metric names prefixed with `prefix` (e.g. `corbul_`).
    pub fn prometheus(&self, prefix: &str) -> String {
        let mut s = String::new();
        let mut m = |name: &str, kind: &str, help: &str, v: f64| {
            s.push_str(&format!("# HELP {prefix}{name} {help}\n# TYPE {prefix}{name} {kind}\n{prefix}{name} {v}\n"));
        };
        m("requests_submitted_total", "counter", "Requests accepted", self.submitted as f64);
        m("requests_completed_total", "counter", "Requests finished normally", self.completed as f64);
        m("requests_cancelled_total", "counter", "Requests cancelled", self.cancelled as f64);
        m("requests_expired_total", "counter", "Requests that missed their deadline", self.expired as f64);
        m("preemptions_total", "counter", "Sequences preempted for lack of KV blocks", self.preemptions as f64);
        m("tokens_generated_total", "counter", "Tokens generated", self.tokens_generated as f64);
        m("queue_depth", "gauge", "Requests waiting", self.queue_depth as f64);
        m("running_sequences", "gauge", "Sequences in a slot", self.running as f64);
        m("batch_utilisation", "gauge", "Share of the per-step token budget used", self.batch_utilisation());
        m("prefix_cache_hit_rate", "gauge", "Share of prompt tokens served from the prefix cache", self.prefix_hit_rate());
        m("kv_blocks_used", "gauge", "KV blocks in use", self.blocks_used as f64);
        m("kv_blocks_total", "gauge", "KV blocks in the pool", self.blocks_total as f64);
        m("kv_cached_blocks", "gauge", "Blocks listed in the prefix cache", self.cached_blocks as f64);
        s
    }
}

// ------------------------------------------------------------------------ scheduler

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedConfig {
    pub block_tokens: usize,
    /// Slots: sequences that may be in flight at once.
    pub n_parallel: usize,
    /// Token budget of one step (decode tokens first, prefill chunks fill the rest).
    /// Must be at least `n_parallel` so a prefilling sequence always finds room.
    pub max_batch_tokens: usize,
    /// Most prompt tokens one sequence may prefill in one step.
    pub prefill_chunk: usize,
    pub max_context: usize,
    /// Most sequences (waiting + running) held at once.
    pub queue_limit: usize,
    /// Steps of waiting that are worth one priority point.
    pub aging_period: u64,
    pub prefix_cache: bool,
}
impl Default for SchedConfig {
    fn default() -> Self {
        SchedConfig { block_tokens: 16, n_parallel: 4, max_batch_tokens: 512, prefill_chunk: 256, max_context: 8192,
            queue_limit: 1024, aging_period: 16, prefix_cache: true }
    }
}
impl SchedConfig {
    fn validate(&self) -> Result<(), SchedError> {
        let bad = |w| Err(SchedError::BadConfig(w));
        if self.block_tokens == 0 { return bad("block_tokens is zero"); }
        if self.n_parallel == 0 { return bad("n_parallel is zero"); }
        if self.max_batch_tokens < self.n_parallel { return bad("max_batch_tokens is below n_parallel"); }
        if self.prefill_chunk == 0 { return bad("prefill_chunk is zero"); }
        if self.max_context < 2 { return bad("max_context is below 2"); }
        if self.queue_limit == 0 { return bad("queue_limit is zero"); }
        if self.aging_period == 0 { return bad("aging_period is zero"); }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Seq {
    req: Request,
    tokens: Vec<u32>,
    /// Tokens whose KV is in `blocks`.
    computed: usize,
    blocks: Vec<BlockId>,
    /// Leading blocks already registered in the prefix cache.
    cached_blocks: usize,
    generated: usize,
    max_new: usize,
    born_step: u64,
}
impl Seq {
    fn remaining(&self) -> usize { self.tokens.len() - self.computed }
    fn is_decode(&self) -> bool { self.generated > 0 && self.remaining() == 1 }
}

pub struct Scheduler {
    cfg: SchedConfig,
    alloc: BlockAllocator,
    cache: PrefixCache,
    seqs: BTreeMap<SeqId, Seq>,
    /// Slot order.
    running: Vec<SeqId>,
    waiting: Vec<SeqId>,
    next_id: SeqId,
    step_no: u64,
    rr: usize,
    events: Vec<Event>,
    stats: SchedStats,
}

impl Scheduler {
    /// Size the pool from a byte budget: `bytes_per_token` is the KV cost of one token.
    pub fn new(cfg: SchedConfig, kv_budget_bytes: u64, bytes_per_token: u64) -> Result<Scheduler, SchedError> {
        cfg.validate()?;
        let total = BlockAllocator::blocks_for_budget(cfg.block_tokens, bytes_per_token, kv_budget_bytes)?;
        Scheduler::with_blocks(cfg, total)
    }
    pub fn with_blocks(cfg: SchedConfig, total_blocks: usize) -> Result<Scheduler, SchedError> {
        cfg.validate()?;
        let alloc = BlockAllocator::with_blocks(cfg.block_tokens, total_blocks)?;
        let cache = PrefixCache::new(cfg.block_tokens);
        Ok(Scheduler { cfg, alloc, cache, seqs: BTreeMap::new(), running: vec![], waiting: vec![], next_id: 1, step_no: 0,
            rr: 0, events: vec![], stats: SchedStats::default() })
    }

    pub fn config(&self) -> &SchedConfig { &self.cfg }
    pub fn allocator(&self) -> &BlockAllocator { &self.alloc }
    pub fn cache(&self) -> &PrefixCache { &self.cache }
    pub fn is_idle(&self) -> bool { self.running.is_empty() && self.waiting.is_empty() }
    pub fn queue_depth(&self) -> usize { self.waiting.len() }
    pub fn running(&self) -> &[SeqId] { &self.running }
    pub fn generated(&self, seq: SeqId) -> Option<usize> { self.seqs.get(&seq).map(|s| s.generated) }

    pub fn stats(&self) -> SchedStats {
        let mut s = self.stats.clone();
        s.prefix_hit_tokens = self.cache.hit_tokens;
        s.prefix_lookup_tokens = self.cache.lookup_tokens;
        s.queue_depth = self.waiting.len();
        s.running = self.running.len();
        s.blocks_used = self.alloc.used();
        s.blocks_total = self.alloc.total();
        s.cached_blocks = self.cache.len();
        s
    }
    pub fn drain_events(&mut self) -> Vec<Event> { std::mem::take(&mut self.events) }

    /// Queue a request. Hostile sizes are refused here, so nothing later can run out of memory
    /// for lack of a sequence that could never fit.
    pub fn submit(&mut self, req: Request) -> Result<SeqId, SchedError> {
        if req.prompt.is_empty() { return Err(SchedError::EmptyPrompt); }
        if req.prompt.len() >= self.cfg.max_context {
            return Err(SchedError::PromptTooLong { len: req.prompt.len(), max: self.cfg.max_context });
        }
        if req.max_new_tokens == 0 { return Err(SchedError::ZeroNewTokens); }
        if self.seqs.len() >= self.cfg.queue_limit { return Err(SchedError::QueueFull); }
        let max_new = req.max_new_tokens.min(self.cfg.max_context - req.prompt.len());
        let needed = (req.prompt.len() as u64 + max_new as u64).div_ceil(self.cfg.block_tokens as u64);
        if needed > self.alloc.total() as u64 {
            return Err(SchedError::BudgetTooSmall { needed_blocks: needed, total_blocks: self.alloc.total() as u64 });
        }
        let id = self.next_id;
        self.next_id += 1;
        self.seqs.insert(id, Seq { tokens: req.prompt.clone(), computed: 0, blocks: vec![], cached_blocks: 0,
            generated: 0, max_new, born_step: self.step_no, req });
        self.waiting.push(id);
        self.stats.submitted += 1;
        Ok(id)
    }

    /// Clone a decoding sequence (parallel sampling, beam search): the child shares every
    /// block and copies the partial one on its first write. Needs a free slot.
    pub fn fork(&mut self, seq: SeqId) -> Result<SeqId, SchedError> {
        let s = self.seqs.get(&seq).ok_or(SchedError::UnknownSeq(seq))?;
        if !self.running.contains(&seq) { return Err(SchedError::NotForkable("the sequence is not running")); }
        if self.running.len() >= self.cfg.n_parallel { return Err(SchedError::NoSlot); }
        if self.seqs.len() >= self.cfg.queue_limit { return Err(SchedError::QueueFull); }
        let blocks = self.alloc.fork(&s.blocks)?;
        let mut child = s.clone();
        child.blocks = blocks;
        let id = self.next_id;
        self.next_id += 1;
        child.born_step = self.step_no;
        self.seqs.insert(id, child);
        self.running.push(id);
        self.stats.submitted += 1;
        Ok(id)
    }

    /// Stop a request now and free its blocks. Returns `false` if it is unknown or already done.
    pub fn cancel(&mut self, seq: SeqId) -> bool {
        if !self.seqs.contains_key(&seq) { return false; }
        self.finish(seq, FinishReason::Cancelled);
        true
    }

    fn eff_priority(&self, s: &Seq) -> i64 {
        s.req.priority as i64 + ((self.step_no - s.born_step) / self.cfg.aging_period) as i64
    }

    fn finish(&mut self, id: SeqId, reason: FinishReason) {
        let bt = self.cfg.block_tokens;
        let Some(mut s) = self.seqs.remove(&id) else { return };
        if self.cfg.prefix_cache && reason != FinishReason::Cancelled {
            let full = s.computed / bt;
            if full > s.cached_blocks { self.cache.insert(&mut self.alloc, &s.tokens[..full * bt], &s.blocks[..full]); }
        }
        for b in s.blocks.drain(..) { let _ = self.alloc.release(b); }
        self.running.retain(|&x| x != id);
        self.waiting.retain(|&x| x != id);
        match reason {
            FinishReason::Cancelled => self.stats.cancelled += 1,
            FinishReason::Deadline => self.stats.expired += 1,
            _ => self.stats.completed += 1,
        }
        self.events.push(Event::Finished { seq: id, reason, generated: s.generated });
    }

    /// Send a running sequence back to the queue, keeping its tokens but not its blocks.
    fn park(&mut self, id: SeqId, items: &mut Vec<StepItem>, copies: &mut Vec<(SeqId, BlockId, BlockId)>) {
        items.retain(|i| i.seq != id);
        copies.retain(|c| c.0 != id);
        let Some(s) = self.seqs.get_mut(&id) else { return };
        for b in s.blocks.drain(..) { let _ = self.alloc.release(b); }
        s.computed = 0;
        s.cached_blocks = 0;
        self.running.retain(|&x| x != id);
        self.waiting.push(id);
        self.stats.preemptions += 1;
        self.events.push(Event::Preempted { seq: id });
    }

    /// A block for `requester`: from the pool, else by evicting cache, else by preempting the
    /// least important sequence. `None` means the requester itself was the least important and
    /// has been parked.
    fn alloc_block(&mut self, requester: SeqId, items: &mut Vec<StepItem>, copies: &mut Vec<(SeqId, BlockId, BlockId)>) -> Option<BlockId> {
        loop {
            if let Some(b) = self.alloc.alloc() { return Some(b); }
            if self.cfg.prefix_cache && self.cache.evict_one(&mut self.alloc) { continue; }
            // least important = lowest effective priority, then the youngest
            let victim = self.running.iter().copied()
                .min_by_key(|id| { let s = &self.seqs[id]; (self.eff_priority(s), std::cmp::Reverse(*id)) });
            match victim {
                Some(v) if v != requester => self.park(v, items, copies),
                _ => { self.park(requester, items, copies); return None; }
            }
        }
    }

    fn schedule(&mut self, id: SeqId, want: usize, items: &mut Vec<StepItem>, copies: &mut Vec<(SeqId, BlockId, BlockId)>) {
        let bt = self.cfg.block_tokens;
        let (computed, total_len) = { let s = &self.seqs[&id]; (s.computed, s.tokens.len()) };
        if computed % bt != 0 {
            // writing into a partly filled block: it must not be shared
            let bi = computed / bt;
            let b = self.seqs[&id].blocks[bi];
            if self.alloc.ref_count(b) > 1 {
                let Some(nb) = self.alloc_block(id, items, copies) else { return };
                if let Some(s) = self.seqs.get_mut(&id) { s.blocks[bi] = nb; }
                let _ = self.alloc.release(b);
                copies.push((id, b, nb));
            }
        }
        let need = (computed + want).div_ceil(bt);
        while self.seqs[&id].blocks.len() < need {
            let Some(nb) = self.alloc_block(id, items, copies) else { return };
            if let Some(s) = self.seqs.get_mut(&id) { s.blocks.push(nb); }
        }
        let s = &self.seqs[&id];
        let kind = if s.is_decode() { StepKind::Decode } else { StepKind::Prefill };
        items.push(StepItem {
            seq: id,
            tokens: s.tokens[computed..computed + want].to_vec(),
            positions: (computed..computed + want).collect(),
            block_table: s.blocks.clone(),
            sample: computed + want == total_len,
            kind,
        });
    }

    fn expire(&mut self, now: u64) {
        let late: Vec<SeqId> = self.seqs.iter().filter(|(_, s)| s.req.deadline.is_some_and(|d| now > d)).map(|(&id, _)| id).collect();
        for id in late { self.finish(id, FinishReason::Deadline); }
    }

    fn admit(&mut self) {
        let bt = self.cfg.block_tokens;
        while self.running.len() < self.cfg.n_parallel {
            let best = self.waiting.iter().copied()
                .max_by_key(|id| { let s = &self.seqs[id]; (self.eff_priority(s), std::cmp::Reverse(*id)) });
            let Some(id) = best else { break };
            let (len, first) = { let s = &self.seqs[&id]; (s.tokens.len(), s.tokens.len().min(self.cfg.prefill_chunk)) };
            let available = self.alloc.free_count() + if self.cfg.prefix_cache { self.cache.evictable(&self.alloc) } else { 0 };
            if !self.running.is_empty() && available < first.div_ceil(bt) { break; }
            self.waiting.retain(|&x| x != id);
            if self.cfg.prefix_cache {
                // keep at least one token to compute, so there are logits to sample from
                let max_blocks = (len - 1) / bt;
                let s = self.seqs.get_mut(&id).unwrap();
                let hit = self.cache.acquire(&mut self.alloc, &s.tokens, max_blocks);
                s.computed = hit.len() * bt;
                s.cached_blocks = hit.len();
                s.blocks = hit;
            }
            self.running.push(id);
        }
    }

    /// Compose the next step. Decode tokens come first (one per sequence), then prefill chunks
    /// share what is left of the token budget; the starting point rotates every step.
    pub fn plan(&mut self, now: u64) -> StepPlan {
        self.step_no += 1;
        self.expire(now);
        self.admit();
        let budget = self.cfg.max_batch_tokens;
        let n = self.running.len();
        let order: Vec<SeqId> = if n == 0 { vec![] } else { (0..n).map(|i| self.running[(self.rr + i) % n]).collect() };
        self.rr = self.rr.wrapping_add(1);
        let (dec, pre): (Vec<SeqId>, Vec<SeqId>) = order.into_iter().partition(|id| self.seqs[id].is_decode());
        let (mut items, mut copies) = (Vec::new(), Vec::new());
        for id in dec.into_iter().chain(pre) {
            let used: usize = items.iter().map(|i: &StepItem| i.tokens.len()).sum();
            if used >= budget { break; }
            if !self.running.contains(&id) { continue; } // parked by someone else's allocation
            let s = &self.seqs[&id];
            let want = if s.is_decode() { 1 } else { s.remaining().min(self.cfg.prefill_chunk).min(budget - used) };
            if want > 0 { self.schedule(id, want, &mut items, &mut copies); }
        }
        if !items.is_empty() {
            self.stats.steps += 1;
            self.stats.batch_tokens += items.iter().map(|i| i.tokens.len() as u64).sum::<u64>();
            self.stats.batch_capacity += budget as u64;
        }
        self.stats.cow_copies += copies.len() as u64;
        StepPlan { step: self.step_no, copies: copies.into_iter().map(|(_, a, b)| (a, b)).collect(), items }
    }

    /// Apply an executor's result. Nothing changes if the result is inconsistent with the plan.
    /// Items of sequences cancelled since the plan was made are ignored.
    pub fn commit(&mut self, plan: &StepPlan, result: &StepResult) -> Result<Vec<Event>, SchedError> {
        if plan.step != self.step_no { return Err(SchedError::StalePlan); }
        let mut sampled: HashMap<SeqId, u32> = HashMap::new();
        for &(seq, tok) in &result.sampled {
            if sampled.insert(seq, tok).is_some() { return Err(SchedError::BadResult(format!("two tokens for sequence {seq}"))); }
        }
        for it in &plan.items {
            let Some(s) = self.seqs.get(&it.seq) else { continue };
            if !self.running.contains(&it.seq) || it.positions.first() != Some(&s.computed) { return Err(SchedError::StalePlan); }
            match (it.sample, sampled.remove(&it.seq).is_some() || false) {
                (true, false) => return Err(SchedError::BadResult(format!("no token for sequence {}", it.seq))),
                (false, true) => return Err(SchedError::BadResult(format!("unrequested token for sequence {}", it.seq))),
                _ => {}
            }
        }
        // `sampled` is consumed above only for validation; rebuild it for the application pass
        let sampled: HashMap<SeqId, u32> = result.sampled.iter().copied().collect();
        if let Some(extra) = result.sampled.iter().find(|(seq, _)| !plan.items.iter().any(|i| i.seq == *seq)) {
            return Err(SchedError::BadResult(format!("token for sequence {} which is not in the plan", extra.0)));
        }
        let bt = self.cfg.block_tokens;
        for it in &plan.items {
            let Some(s) = self.seqs.get_mut(&it.seq) else { continue };
            s.computed += it.tokens.len();
            let full = s.computed / bt;
            if self.cfg.prefix_cache && full > s.cached_blocks {
                self.cache.insert(&mut self.alloc, &s.tokens[..full * bt], &s.blocks[..full]);
                s.cached_blocks = full;
            }
            if !it.sample { continue; }
            let tok = sampled[&it.seq];
            s.tokens.push(tok);
            s.generated += 1;
            self.stats.tokens_generated += 1;
            self.events.push(Event::Token { seq: it.seq, token: tok });
            let reason = if s.req.stop_tokens.contains(&tok) { Some(FinishReason::Stop) }
                else if s.generated >= s.max_new { Some(FinishReason::Length) }
                else if s.tokens.len() >= self.cfg.max_context { Some(FinishReason::ContextFull) }
                else { None };
            if let Some(r) = reason { self.finish(it.seq, r); }
        }
        Ok(self.drain_events())
    }

    /// One full iteration: plan, run, commit. Events of an empty plan (deadlines) still come back.
    pub fn step(&mut self, now: u64, exec: &mut dyn Executor) -> Result<Vec<Event>, SchedError> {
        let plan = self.plan(now);
        if plan.items.is_empty() { return Ok(self.drain_events()); }
        let res = exec.run(&plan);
        self.commit(&plan, &res)
    }

    /// Check the bookkeeping: every reference count equals what sequences and the cache hold,
    /// nothing leaks, nothing exceeds the pool. Cheap enough to call after every step in tests.
    pub fn check_invariants(&self) -> Result<(), String> {
        let bt = self.cfg.block_tokens;
        let mut expect = vec![0u32; self.alloc.total()];
        for (id, s) in &self.seqs {
            let running = self.running.contains(id);
            let waiting = self.waiting.contains(id);
            if running == waiting { return Err(format!("sequence {id} is in {} lists", if running { "both" } else { "no" })); }
            if waiting && (!s.blocks.is_empty() || s.computed != 0) { return Err(format!("waiting sequence {id} holds blocks")); }
            if s.blocks.len() < s.computed.div_ceil(bt) { return Err(format!("sequence {id} has fewer blocks than computed tokens")); }
            if s.computed > s.tokens.len() { return Err(format!("sequence {id} computed beyond its tokens")); }
            for &b in &s.blocks { *expect.get_mut(b as usize).ok_or("block id out of range")? += 1; }
        }
        for b in self.cache.blocks() { *expect.get_mut(b as usize).ok_or("cache block out of range")? += 1; }
        for (b, &e) in expect.iter().enumerate() {
            let got = self.alloc.ref_count(b as BlockId);
            if got != e { return Err(format!("block {b}: reference count {got}, expected {e}")); }
        }
        let used = expect.iter().filter(|&&e| e > 0).count();
        if used != self.alloc.used() { return Err(format!("{used} blocks referenced but {} used", self.alloc.used())); }
        if self.alloc.used() + self.alloc.free_count() != self.alloc.total() { return Err("free + used != total".into()); }
        if self.running.len() > self.cfg.n_parallel { return Err("more running sequences than slots".into()); }
        Ok(())
    }
}

// ============================================================================ model executor
//
// Everything above is bookkeeping. This section runs a [`Model`] under the scheduler:
//
// * [`ModelExecutor`] implements [`Executor`]: for every item of a [`StepPlan`] it brings that
//   sequence's [`State`] to the item's start position, feeds the item's tokens, and — when the
//   scheduler asks for a sample — turns the logits into a token with the request's own sampler.
// * [`BatchEngine`] owns a scheduler and an executor and keeps them in step (registering requests,
//   handing finished or preempted caches back, cancelling a request whose model call failed).
//
// Memory model, stated plainly: the scheduler's paged blocks are *accounting*. The model keeps one
// contiguous KV cache per sequence (`State`), so the block table in a plan is not read, and
// copy-on-write copies need no work. The pool size is what bounds concurrency; size it with
// [`kv_bytes_per_token`] so it matches what the states really hold. A prefix hit reported by the
// scheduler is honoured from a small pool of retired states (longest common prefix wins) and
// recomputed when no retired state has it, so a hit never changes a result, only how long it takes.
//
// Exactness: for dense f32 weights the result of a request is bit-identical to running it alone
// (one prefill, then one token at a time), whatever the batch shape, chunking, preemption or prefix
// reuse. With quantised weights activations are quantised per batch shape, so results agree only
// within that quantisation noise.

use crate::model::{KvKind, Model, ModelError, Sampler, State, PREFILL_CHUNK};
use std::sync::Arc;

/// Sampling settings of one request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerParams {
    /// `<= 0` is greedy.
    pub temperature: f32,
    /// 0 = off.
    pub top_k: usize,
    /// 1.0 = off.
    pub top_p: f32,
    /// 1.0 = off.
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}
impl Default for SamplerParams {
    fn default() -> Self { SamplerParams { temperature: 0.0, top_k: 0, top_p: 1.0, repeat_penalty: 1.0, repeat_last_n: 64, seed: 1 } }
}
impl SamplerParams {
    pub fn greedy() -> SamplerParams { SamplerParams::default() }
    pub fn sampled(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> SamplerParams {
        SamplerParams { temperature, top_k, top_p, seed, ..SamplerParams::default() }
    }
    /// The sampler these settings describe (the same one `Model::generate` is given).
    pub fn build(&self) -> Sampler {
        Sampler::new(self.temperature, self.top_k, self.seed).with_top_p(self.top_p).with_repetition_penalty(self.repeat_penalty, self.repeat_last_n)
    }
}

/// How the executor holds caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecConfig {
    /// Precision of every sequence's KV cache.
    pub kv: KvKind,
    /// Finished or preempted caches kept for prefix reuse (at most this many).
    pub retired_max_states: usize,
    /// ... and at most this many bytes of KV in total.
    pub retired_max_bytes: usize,
    /// Run the items of one step on their own threads (they are independent). Results do not change.
    pub parallel_items: bool,
}
impl Default for ExecConfig {
    fn default() -> Self { ExecConfig { kv: KvKind::F32, retired_max_states: 16, retired_max_bytes: 256 << 20, parallel_items: false } }
}

/// KV bytes one token costs in `model` at precision `kind` — the number to give
/// [`Scheduler::new`] as `bytes_per_token`.
pub fn kv_bytes_per_token(model: &Model, kind: KvKind) -> u64 {
    let s = &model.spec;
    let per = s.layers.saturating_mul(2).saturating_mul(s.kv_heads).saturating_mul(s.head_dim);
    let b = match kind { KvKind::F32 => per.saturating_mul(4), KvKind::F16 => per.saturating_mul(2), KvKind::Q8 => per.saturating_add(per.div_ceil(s.head_dim.max(1)).saturating_mul(4)) }; // int8 plus one f32 scale per head and position
    b.max(1)
}

/// What the executor has done so far.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecStats {
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
    pub sampled: u64,
    /// Tokens whose KV was taken over from an existing cache instead of being computed.
    pub reused_tokens: u64,
    /// Tokens recomputed because no cache held them (after a preemption or a scheduler prefix hit
    /// whose cache was no longer here).
    pub recomputed_tokens: u64,
    pub failures: u64,
    // gauges
    pub retired_states: usize,
    pub retired_bytes: usize,
    pub live_states: usize,
}

struct Rec {
    tokens: Vec<usize>,
    state: Option<State>,
    sampler: Sampler,
    params: SamplerParams,
}
struct Retired { state: State, stamp: u64 }
struct Work { seq: SeqId, state: State, tokens: Vec<usize>, sample: bool }
type Out = (SeqId, State, Result<Option<Vec<f32>>, ModelError>);

/// Called with `(sequence, position of the token about to be sampled, logits)` for every sample.
pub type LogitObserver = Box<dyn FnMut(SeqId, usize, &[f32]) + Send>;

fn common_prefix(a: &[usize], b: &[usize]) -> usize { a.iter().zip(b).take_while(|(x, y)| x == y).count() }

/// Feed `toks` without computing logits (prefill chunks nobody samples from).
fn feed(model: &Model, st: &mut State, toks: &[usize]) -> Result<(), ModelError> {
    let h = model.spec.hidden as usize;
    let layers = model.spec.layers as usize;
    for c in toks.chunks(PREFILL_CHUNK) {
        let mut x = vec![0.0f32; c.len() * h];
        for (i, &t) in c.iter().enumerate() { x[i * h..(i + 1) * h].copy_from_slice(&model.embed_token(t)?); }
        let pos = st.pos;
        model.run_layers_batch(st, 0..layers, &mut x, pos)?;
        model.finish_batch(st, c);
    }
    Ok(())
}

fn exec_work(model: &Model, w: Work) -> Out {
    let Work { seq, mut state, tokens, sample } = w;
    let r = if sample { model.forward_batch(&mut state, &tokens).map(Some) } else { feed(model, &mut state, &tokens).map(|()| None) };
    (seq, state, r)
}

/// Runs plans on a [`Model`], one `State` per sequence.
pub struct ModelExecutor {
    model: Arc<Model>,
    cfg: ExecConfig,
    recs: HashMap<SeqId, Rec>,
    retired: Vec<Retired>,
    clock: u64,
    stats: ExecStats,
    failures: Vec<(SeqId, String)>,
    observer: Option<LogitObserver>,
}

impl ModelExecutor {
    pub fn new(model: Arc<Model>, cfg: ExecConfig) -> ModelExecutor {
        ModelExecutor { model, cfg, recs: HashMap::new(), retired: vec![], clock: 0, stats: ExecStats::default(), failures: vec![], observer: None }
    }
    pub fn model(&self) -> &Arc<Model> { &self.model }
    pub fn config(&self) -> &ExecConfig { &self.cfg }
    pub fn stats(&self) -> ExecStats {
        let mut s = self.stats.clone();
        s.retired_states = self.retired.len();
        s.retired_bytes = self.retired_bytes();
        s.live_states = self.recs.values().filter(|r| r.state.is_some()).count();
        s
    }
    /// Look at the logits before each token is sampled (log-probabilities, tests, tracing).
    pub fn set_logit_observer(&mut self, f: Option<LogitObserver>) { self.observer = f; }
    /// Sequences whose model call failed since the last call, with the reason. A failed sequence
    /// gets no sampled token; [`BatchEngine`] cancels it.
    pub fn take_failures(&mut self) -> Vec<(SeqId, String)> { std::mem::take(&mut self.failures) }

    /// Start tracking a sequence the scheduler knows as `seq`.
    pub fn register(&mut self, seq: SeqId, prompt: &[u32], params: SamplerParams) -> Result<(), ModelError> {
        let vocab = self.model.spec.vocab as usize;
        if let Some(&t) = prompt.iter().find(|&&t| t as usize >= vocab) { return Err(ModelError::Token(t as usize)); }
        let tokens: Vec<usize> = prompt.iter().map(|&t| t as usize).collect();
        let mut sampler = params.build();
        sampler.observe_all(&tokens);
        self.recs.insert(seq, Rec { tokens, state: None, sampler, params });
        Ok(())
    }

    /// `child` continues `parent` from where it is now (both must be between steps). The child's
    /// sampler gets a seed derived from the parent's, so sampled children diverge; greedy ones do not.
    pub fn fork_seq(&mut self, parent: SeqId, child: SeqId) -> Result<(), ModelError> {
        let p = self.recs.get(&parent).ok_or_else(|| ModelError::Shape(format!("unknown sequence {parent}")))?;
        let state = match &p.state { Some(s) => Some(s.fork()?), None => None };
        let mut params = p.params;
        params.seed = params.seed.wrapping_add(child.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let tokens = p.tokens.clone();
        let mut sampler = params.build();
        sampler.observe_all(&tokens);
        self.recs.insert(child, Rec { tokens, state, sampler, params });
        Ok(())
    }

    /// The scheduler is done with `seq`. Its cache is kept for prefix reuse unless `keep` is false.
    pub fn release(&mut self, seq: SeqId, keep: bool) {
        if let Some(rec) = self.recs.remove(&seq)
            && keep
            && let Some(st) = rec.state
        {
            self.retire(st);
        }
    }

    /// The scheduler took `seq`'s blocks away (preemption): keep the cache only in the retired pool,
    /// where it is bounded and can serve the recompute.
    pub fn park(&mut self, seq: SeqId) {
        let st = self.recs.get_mut(&seq).and_then(|r| r.state.take());
        if let Some(st) = st { self.retire(st); }
    }

    fn retired_bytes(&self) -> usize { self.retired.iter().map(|r| r.state.kv_bytes()).sum() }

    fn retire(&mut self, st: State) {
        if st.pos == 0 || st.is_compacted() || st.tokens().len() != st.pos { return; }
        self.clock += 1;
        self.retired.push(Retired { state: st, stamp: self.clock });
        while self.retired.len() > self.cfg.retired_max_states || self.retired_bytes() > self.cfg.retired_max_bytes {
            let Some(old) = self.retired.iter().enumerate().min_by_key(|(_, r)| r.stamp).map(|(i, _)| i) else { break };
            self.retired.swap_remove(old);
        }
    }

    /// The retired state sharing the longest prefix with `want`, forked (or taken, when `want`
    /// extends it completely, because then nobody else is likelier to need it than we are).
    fn take_best(&mut self, want: &[usize]) -> Option<State> {
        let best = self.retired.iter().enumerate()
            .map(|(i, r)| (common_prefix(r.state.tokens(), want), r.stamp, i))
            .filter(|&(c, _, _)| c > 0)
            .max_by_key(|&(c, stamp, _)| (c, stamp))?;
        let (c, _, i) = best;
        self.clock += 1;
        let (len, forked) = { let r = &mut self.retired[i]; r.stamp = self.clock; (r.state.tokens().len(), r.state.fork()) };
        match forked {
            Ok(st) if c < len => Some(st),
            _ => Some(self.retired.swap_remove(i).state),
        }
    }

    fn fail(&mut self, seq: SeqId, msg: String) {
        self.stats.failures += 1;
        self.failures.push((seq, msg));
    }

    /// A state positioned exactly at `p` tokens of `seq`'s history.
    fn prepare(&mut self, seq: SeqId, p: usize) -> Result<State, ModelError> {
        let model = self.model.clone();
        let (own, want) = {
            let rec = self.recs.get_mut(&seq).ok_or_else(|| ModelError::Shape(format!("unknown sequence {seq}")))?;
            (rec.state.take(), rec.tokens[..p].to_vec())
        };
        let mut st = match own {
            Some(s) => s,
            None => self.take_best(&want).unwrap_or_else(|| model.new_state_with(self.cfg.kv)),
        };
        let mut keep = if st.is_compacted() || st.tokens().len() != st.pos { 0 } else { common_prefix(st.tokens(), &want).min(p) };
        if keep == 0 { st.reset(); } else if st.pos > keep {
            if st.can_truncate(keep) { st.truncate(keep)?; } else { st.reset(); keep = 0; }
        }
        self.stats.reused_tokens += keep as u64;
        if p > keep {
            self.stats.recomputed_tokens += (p - keep) as u64;
            feed(&model, &mut st, &want[keep..])?;
        }
        Ok(st)
    }

    fn prepare_item(&mut self, it: &StepItem) -> Result<Work, String> {
        let rec = self.recs.get(&it.seq).ok_or_else(|| format!("unknown sequence {}", it.seq))?;
        let (Some(&p), n) = (it.positions.first(), it.tokens.len()) else { return Err("empty plan item".into()) };
        if n == 0 || it.positions.len() != n { return Err("malformed plan item".into()); }
        let hist = rec.tokens.len();
        if p.checked_add(n).is_none_or(|e| e > hist) { return Err(format!("plan item runs past the sequence ({p}+{n} > {hist})")); }
        let tokens: Vec<usize> = it.tokens.iter().map(|&t| t as usize).collect();
        if rec.tokens[p..p + n] != tokens[..] { return Err("plan tokens differ from the sequence's history".into()); }
        if it.sample != (p + n == hist) { return Err("plan item asks for a sample at the wrong place".into()); }
        let state = self.prepare(it.seq, p).map_err(|e| e.to_string())?;
        match it.kind { StepKind::Prefill => self.stats.prefill_tokens += n as u64, StepKind::Decode => self.stats.decode_tokens += n as u64 }
        Ok(Work { seq: it.seq, state, tokens, sample: it.sample })
    }
}

impl Executor for ModelExecutor {
    /// Run every item. An item that cannot be run (model error, plan that does not match the
    /// sequence) is reported through [`ModelExecutor::take_failures`] and gets no token; use
    /// [`BatchEngine`], which cancels such sequences, rather than feeding this to
    /// [`Scheduler::step`] directly if failures are possible.
    fn run(&mut self, plan: &StepPlan) -> StepResult {
        let model = self.model.clone();
        let mut works = Vec::with_capacity(plan.items.len());
        for it in &plan.items {
            match self.prepare_item(it) {
                Ok(w) => works.push(w),
                Err(msg) => self.fail(it.seq, msg),
            }
        }
        let outs: Vec<Out> = if self.cfg.parallel_items && works.len() > 1 {
            let m: &Model = &model;
            std::thread::scope(|s| {
                let hs: Vec<_> = works.into_iter().map(|w| s.spawn(move || exec_work(m, w))).collect();
                hs.into_iter().filter_map(|h| h.join().ok()).collect()
            })
        } else {
            works.into_iter().map(|w| exec_work(&model, w)).collect()
        };
        let mut res = StepResult::default();
        for (seq, st, r) in outs {
            match r {
                Err(e) => self.fail(seq, e.to_string()),
                Ok(logits) => {
                    let Some(rec) = self.recs.get_mut(&seq) else { continue };
                    rec.state = Some(st);
                    let Some(l) = logits else { continue };
                    if let Some(o) = self.observer.as_mut() { o(seq, rec.tokens.len(), &l); }
                    let tok = rec.sampler.sample(&l);
                    rec.sampler.observe(tok);
                    rec.tokens.push(tok);
                    self.stats.sampled += 1;
                    res.sampled.push((seq, tok as u32));
                }
            }
        }
        res
    }
}

// ------------------------------------------------------------------------ engine

#[derive(Debug)]
pub enum EngineError {
    Sched(SchedError),
    Model(ModelError),
    /// The scheduler had work but planned nothing for many steps in a row.
    Stalled,
}
impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Sched(e) => write!(f, "{e}"),
            EngineError::Model(e) => write!(f, "{e}"),
            EngineError::Stalled => write!(f, "the scheduler stopped making progress"),
        }
    }
}
impl std::error::Error for EngineError {}
impl From<SchedError> for EngineError { fn from(e: SchedError) -> Self { EngineError::Sched(e) } }
impl From<ModelError> for EngineError { fn from(e: ModelError) -> Self { EngineError::Model(e) } }

/// What a streaming callback tells the engine to do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    /// Cancel the sequence the event belongs to.
    CancelSeq,
    /// Stop running now; unfinished requests stay queued and `run_until_idle` can be called again.
    Halt,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StepReport {
    /// Tokens the model processed this step (prefill and decode).
    pub tokens_run: usize,
    pub events: Vec<Event>,
    /// Sequences cancelled because their model call failed, with the reason.
    pub failures: Vec<(SeqId, String)>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunSummary {
    pub steps: u64,
    pub failures: Vec<(SeqId, String)>,
    pub halted: bool,
}

/// A scheduler and a model executor kept in step: submit requests, call [`BatchEngine::step`] or
/// [`BatchEngine::run_until_idle`], read tokens from the events.
pub struct BatchEngine {
    sched: Scheduler,
    exec: ModelExecutor,
    clock: u64,
}

impl BatchEngine {
    /// Size the KV pool from a byte budget at the executor's cache precision. `cfg.max_context` is
    /// lowered to the model's context if it is larger.
    pub fn new(model: Arc<Model>, mut cfg: SchedConfig, kv_budget_bytes: u64, ecfg: ExecConfig) -> Result<BatchEngine, SchedError> {
        cfg.max_context = cfg.max_context.min(model.spec.context as usize);
        let per = kv_bytes_per_token(&model, ecfg.kv);
        let sched = Scheduler::new(cfg, kv_budget_bytes, per)?;
        Ok(BatchEngine { sched, exec: ModelExecutor::new(model, ecfg), clock: 0 })
    }
    /// Like [`BatchEngine::new`] with the pool given in blocks.
    pub fn with_blocks(model: Arc<Model>, mut cfg: SchedConfig, total_blocks: usize, ecfg: ExecConfig) -> Result<BatchEngine, SchedError> {
        cfg.max_context = cfg.max_context.min(model.spec.context as usize);
        let sched = Scheduler::with_blocks(cfg, total_blocks)?;
        Ok(BatchEngine { sched, exec: ModelExecutor::new(model, ecfg), clock: 0 })
    }

    pub fn scheduler(&self) -> &Scheduler { &self.sched }
    pub fn executor(&self) -> &ModelExecutor { &self.exec }
    pub fn executor_mut(&mut self) -> &mut ModelExecutor { &mut self.exec }
    pub fn is_idle(&self) -> bool { self.sched.is_idle() }
    /// Logical time: the number of steps taken. Request deadlines are in these units.
    pub fn now(&self) -> u64 { self.clock }

    pub fn submit(&mut self, req: Request, params: SamplerParams) -> Result<SeqId, EngineError> {
        let vocab = self.exec.model.spec.vocab as usize;
        if let Some(&t) = req.prompt.iter().find(|&&t| t as usize >= vocab) { return Err(ModelError::Token(t as usize).into()); }
        let prompt = req.prompt.clone();
        let id = self.sched.submit(req)?;
        self.exec.register(id, &prompt, params)?;
        Ok(id)
    }

    /// Parallel sampling / branching: see [`Scheduler::fork`]. The child continues from the same cache.
    pub fn fork(&mut self, seq: SeqId) -> Result<SeqId, EngineError> {
        let child = self.sched.fork(seq)?;
        if let Err(e) = self.exec.fork_seq(seq, child) {
            self.sched.cancel(child);
            self.sched.drain_events();
            return Err(e.into());
        }
        Ok(child)
    }

    /// Stop a request. Its `Finished { reason: Cancelled }` event comes with the next step.
    pub fn cancel(&mut self, seq: SeqId) -> bool {
        let ok = self.sched.cancel(seq);
        if ok { self.exec.release(seq, false); }
        ok
    }

    fn after(&mut self, events: &[Event]) {
        for ev in events {
            match ev {
                Event::Finished { seq, reason, .. } => self.exec.release(*seq, *reason != FinishReason::Cancelled),
                Event::Preempted { seq } => self.exec.park(*seq),
                Event::Token { .. } => {}
            }
        }
    }

    /// One iteration: plan, run on the model, commit.
    pub fn step(&mut self) -> Result<StepReport, EngineError> {
        self.clock += 1;
        let plan = self.sched.plan(self.clock);
        if plan.items.is_empty() {
            let events = self.sched.drain_events();
            self.after(&events);
            return Ok(StepReport { tokens_run: 0, events, failures: vec![] });
        }
        let res = self.exec.run(&plan);
        let failures = self.exec.take_failures();
        for (seq, _) in &failures { self.sched.cancel(*seq); }
        let events = self.sched.commit(&plan, &res)?;
        self.after(&events);
        Ok(StepReport { tokens_run: plan.total_tokens(), events, failures })
    }

    /// Step until nothing is left, handing every event to `on_event`.
    pub fn run_until_idle(&mut self, on_event: &mut dyn FnMut(&Event) -> Flow) -> Result<RunSummary, EngineError> {
        let mut sum = RunSummary::default();
        let mut stalled = 0u32;
        while !self.sched.is_idle() {
            let rep = self.step()?;
            sum.steps += 1;
            sum.failures.extend(rep.failures);
            if rep.tokens_run == 0 && rep.events.is_empty() {
                stalled += 1;
                if stalled > 10_000 { return Err(EngineError::Stalled); }
            } else {
                stalled = 0;
            }
            for ev in &rep.events {
                match on_event(ev) {
                    Flow::Continue => {}
                    Flow::CancelSeq => match ev {
                        Event::Token { seq, .. } | Event::Finished { seq, .. } | Event::Preempted { seq } => { self.cancel(*seq); }
                    },
                    Flow::Halt => sum.halted = true,
                }
            }
            if sum.halted { break; }
        }
        // events produced by cancellations made in the last callbacks
        let rest = self.sched.drain_events();
        self.after(&rest);
        for ev in &rest { on_event(ev); }
        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn below(&mut self, n: u64) -> u64 { if n == 0 { 0 } else { self.next() % n } }
    }

    // ---------------------------------------------------------------- allocator

    #[test]
    fn allocator_matches_a_reference_model_under_random_operations() {
        for seed in 1..=20u64 {
            let mut rng = Rng(seed * 0x9e37_79b9 + 1);
            let total = 1 + rng.below(40) as usize;
            let mut a = BlockAllocator::with_blocks(4, total).unwrap();
            let mut model: HashMap<BlockId, u32> = HashMap::new();
            for _ in 0..3000 {
                match rng.below(5) {
                    0 | 1 => {
                        let got = a.alloc();
                        if model.len() == total { assert!(got.is_none()); } else {
                            let b = got.expect("pool not empty");
                            assert!(model.insert(b, 1).is_none(), "handed out a block twice");
                        }
                    }
                    2 => if let Some(&b) = model.keys().nth(rng.below(model.len().max(1) as u64) as usize) {
                        a.retain(b).unwrap();
                        *model.get_mut(&b).unwrap() += 1;
                    },
                    3 => if let Some(&b) = model.keys().nth(rng.below(model.len().max(1) as u64) as usize) {
                        let freed = a.release(b).unwrap();
                        let r = model.get_mut(&b).unwrap();
                        *r -= 1;
                        assert_eq!(freed, *r == 0);
                        if *r == 0 { model.remove(&b); }
                    },
                    _ => if let Some(&b) = model.keys().nth(rng.below(model.len().max(1) as u64) as usize) {
                        match a.make_unique(b) {
                            Ok((nb, copied)) => {
                                if model[&b] == 1 { assert!(!copied && nb == b); } else {
                                    assert!(copied && nb != b);
                                    *model.get_mut(&b).unwrap() -= 1;
                                    assert!(model.insert(nb, 1).is_none());
                                }
                            }
                            Err(e) => { assert_eq!(e, SchedError::OutOfBlocks); assert!(model[&b] > 1 && model.len() == total); }
                        }
                    },
                }
                assert_eq!(a.used(), model.len());
                assert_eq!(a.free_count(), total - model.len());
                for (&b, &r) in &model { assert_eq!(a.ref_count(b), r); }
            }
            // drain: everything returns, no leak
            let live: Vec<_> = model.iter().map(|(&b, &r)| (b, r)).collect();
            for (b, r) in live { for _ in 0..r { a.release(b).unwrap(); } }
            assert_eq!((a.used(), a.free_count()), (0, total));
        }
    }

    #[test]
    fn allocator_refuses_double_frees_and_bad_ids() {
        let mut a = BlockAllocator::with_blocks(8, 3).unwrap();
        let b = a.alloc().unwrap();
        assert_eq!(a.release(b), Ok(true));
        assert_eq!(a.release(b), Err(SchedError::DoubleFree(b)));
        assert_eq!(a.release(99), Err(SchedError::BadBlock(99)));
        assert_eq!(a.retain(b), Err(SchedError::BadBlock(b)));
        assert_eq!(a.make_unique(b), Err(SchedError::BadBlock(b)));
        // a failed fork rolls back
        let c = a.alloc().unwrap();
        assert!(a.fork(&[c, 42]).is_err());
        assert_eq!(a.ref_count(c), 1);
    }

    #[test]
    fn copy_on_write_and_fork() {
        let mut a = BlockAllocator::with_blocks(4, 4).unwrap();
        let b = a.alloc().unwrap();
        let table = a.fork(&[b]).unwrap();
        assert_eq!(a.ref_count(b), 2);
        let (nb, copied) = a.make_unique(table[0]).unwrap();
        assert!(copied && nb != b);
        assert_eq!((a.ref_count(b), a.ref_count(nb)), (1, 1));
        assert_eq!(a.make_unique(b), Ok((b, false)));
    }

    #[test]
    fn budget_sizing_and_fragmentation() {
        assert_eq!(BlockAllocator::blocks_for_budget(16, 1000, 16_000 * 10).unwrap(), 10);
        assert!(matches!(BlockAllocator::blocks_for_budget(16, 1000, 100), Err(SchedError::BudgetTooSmall { .. })));
        assert!(BlockAllocator::blocks_for_budget(0, 1, 1).is_err());
        assert!(BlockAllocator::blocks_for_budget(1, 0, 1).is_err());
        assert_eq!(BlockAllocator::blocks_for_budget(1, 1, u64::MAX).unwrap(), MAX_BLOCKS);
        assert!(BlockAllocator::blocks_for_budget(usize::MAX, u64::MAX, u64::MAX).is_err());
        let mut a = BlockAllocator::with_blocks(2, 6).unwrap();
        let ids: Vec<_> = (0..6).map(|_| a.alloc().unwrap()).collect();
        assert_eq!(a.fragmentation().free, 0);
        a.release(ids[1]).unwrap();
        a.release(ids[3]).unwrap();
        a.release(ids[4]).unwrap();
        let f = a.fragmentation();
        assert_eq!((f.used, f.free, f.largest_free_run), (3, 3, 2));
        assert!((f.ratio - (1.0 - 2.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn alloc_with_evict_calls_the_hook_until_it_gives_up() {
        let mut a = BlockAllocator::with_blocks(2, 2).unwrap();
        let held = [a.alloc().unwrap(), a.alloc().unwrap()];
        let mut left = held.to_vec();
        let got = a.alloc_with_evict(&mut |al| match left.pop() { Some(b) => { al.release(b).unwrap(); true } None => false });
        assert!(got.is_some());
        let got = a.alloc_with_evict(&mut |al| match left.pop() { Some(b) => { al.release(b).unwrap(); true } None => false });
        assert!(got.is_some());
        assert!(a.alloc_with_evict(&mut |_| false).is_none());
    }

    // ---------------------------------------------------------------- prefix cache

    #[test]
    fn prefix_cache_agrees_with_brute_force() {
        let bt = 3;
        for seed in 1..=15u64 {
            let mut rng = Rng(seed * 7919);
            let mut a = BlockAllocator::with_blocks(bt, 2000).unwrap();
            let mut c = PrefixCache::new(bt);
            // reference: block-prefix → block id of the first insertion
            let mut reference: HashMap<Vec<u32>, BlockId> = HashMap::new();
            let mut sources: Vec<Vec<u32>> = vec![];
            for _ in 0..200 {
                let seq: Vec<u32> = if !sources.is_empty() && rng.below(2) == 0 {
                    // share a prefix with an earlier sequence, then diverge
                    let base = &sources[rng.below(sources.len() as u64) as usize];
                    let keep = rng.below(base.len() as u64 + 1) as usize;
                    let mut s = base[..keep].to_vec();
                    for _ in 0..rng.below(10) { s.push(rng.below(3) as u32); }
                    s
                } else {
                    (0..rng.below(14)).map(|_| rng.below(3) as u32).collect()
                };
                let blocks: Vec<BlockId> = (0..seq.len() / bt).map(|_| a.alloc().unwrap()).collect();
                let found = c.peek(&seq, usize::MAX);
                let mut want = vec![];
                for i in 1..=seq.len() / bt {
                    match reference.get(&seq[..i * bt]) { Some(&b) => want.push(b), None => break }
                }
                assert_eq!(found, want, "seed {seed} seq {seq:?}");
                c.insert(&mut a, &seq, &blocks);
                for (i, &b) in blocks.iter().enumerate() { reference.entry(seq[..(i + 1) * bt].to_vec()).or_insert(b); }
                // the caller's own references go away; the cache keeps its own
                for &b in &blocks { a.release(b).unwrap(); }
                sources.push(seq);
            }
            assert_eq!(c.len(), reference.len());
            // everything unused can be evicted, nothing leaks
            while c.evict_one(&mut a) {}
            assert_eq!((c.len(), a.used()), (0, 0));
        }
    }

    #[test]
    fn prefix_cache_evicts_least_recently_used_leaves_and_spares_pinned_blocks() {
        let mut a = BlockAllocator::with_blocks(2, 10).unwrap();
        let mut c = PrefixCache::new(2);
        let mk = |a: &mut BlockAllocator, n: usize| (0..n).map(|_| a.alloc().unwrap()).collect::<Vec<_>>();
        let (s1, s2) = ([1u32, 1, 2, 2], [9u32, 9, 8, 8]);
        let (b1, b2) = (mk(&mut a, 2), mk(&mut a, 2));
        c.insert(&mut a, &s1, &b1);
        c.insert(&mut a, &s2, &b2);
        for &b in b1.iter().chain(&b2) { a.release(b).unwrap(); }
        // touch s1 so s2's leaf is the oldest
        let hit = c.acquire(&mut a, &s1, 2);
        assert_eq!(hit, b1);
        assert_eq!(c.evictable(&a), 2); // only s2's two blocks are unused
        assert!(c.evict_one(&mut a)); // s2's leaf
        assert_eq!(c.peek(&s2, 2), vec![b2[0]]);
        assert!(c.evict_one(&mut a)); // s2's root (now a leaf)
        assert!(c.peek(&s2, 2).is_empty());
        assert!(!c.evict_one(&mut a), "s1 is pinned by `hit`");
        assert_eq!(c.peek(&s1, 2), b1);
        for &b in &hit { a.release(b).unwrap(); }
        // internal nodes go only after their children
        assert!(c.evict_one(&mut a));
        assert_eq!(c.peek(&s1, 2), vec![b1[0]]);
        assert!(c.evict_one(&mut a));
        assert_eq!((c.len(), a.used()), (0, 0));
    }

    #[test]
    fn prefix_cache_statistics_and_collision_safety() {
        let mut a = BlockAllocator::with_blocks(2, 8).unwrap();
        let mut c = PrefixCache::new(2);
        let blocks = [a.alloc().unwrap(), a.alloc().unwrap()];
        c.insert(&mut a, &[1, 2, 3, 4], &blocks);
        let hit = c.acquire(&mut a, &[1, 2, 3, 9, 5], usize::MAX);
        assert_eq!(hit.len(), 1);
        assert_eq!((c.hit_tokens, c.lookup_tokens), (2, 4));
        assert!((c.hit_rate() - 0.5).abs() < 1e-9);
        assert!(c.acquire(&mut a, &[7, 7, 7, 7], 9).is_empty());
        assert!(c.peek(&[1, 2, 3, 4], 0).is_empty());
        // an unknown block id is not cached
        assert_eq!(c.insert(&mut a, &[5, 5], &[77]), 0);
    }

    // ---------------------------------------------------------------- scheduler simulations

    const VOCAB: u32 = 997;
    fn next_token(ctx: &[u32]) -> u32 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &t in ctx { h = (h ^ t as u64).wrapping_mul(0x100_0000_01b3); h ^= h >> 31; }
        (h % VOCAB as u64) as u32
    }
    /// What a request must produce when decoded in isolation.
    fn reference(req: &Request, max_context: usize) -> Vec<u32> {
        let max_new = req.max_new_tokens.min(max_context - req.prompt.len());
        let mut ctx = req.prompt.clone();
        let mut out = vec![];
        while out.len() < max_new {
            let t = next_token(&ctx);
            ctx.push(t);
            out.push(t);
            if req.stop_tokens.contains(&t) { break; }
        }
        out
    }

    /// A fake model with a paged KV memory: it writes the tokens of each item into the blocks the
    /// plan names and reads the whole context back through the block table to "sample".
    /// Wrong sharing, a missed copy-on-write or a bad recompute shows up as a wrong token.
    struct Mock { mem: HashMap<(BlockId, usize), u32>, bt: usize, max_item: usize, max_step: usize, max_prefill: usize }
    impl Mock {
        fn new(bt: usize) -> Mock { Mock { mem: HashMap::new(), bt, max_item: 0, max_step: 0, max_prefill: 0 } }
        fn run(&mut self, s: &Scheduler, plan: &StepPlan) -> StepResult {
            for &(from, to) in &plan.copies {
                for off in 0..self.bt {
                    if let Some(&v) = self.mem.get(&(from, off)) { self.mem.insert((to, off), v); }
                }
            }
            self.max_step = self.max_step.max(plan.total_tokens());
            let mut out = StepResult::default();
            for it in &plan.items {
                self.max_item = self.max_item.max(it.tokens.len());
                if it.kind == StepKind::Prefill { self.max_prefill = self.max_prefill.max(it.tokens.len()); }
                assert_eq!(it.tokens.len(), it.positions.len());
                for (i, &t) in it.tokens.iter().enumerate() {
                    let p = it.positions[i];
                    let blk = it.block_table[p / self.bt];
                    assert_eq!(s.allocator().ref_count(blk), 1, "wrote into a shared or cached block");
                    self.mem.insert((blk, p % self.bt), t);
                }
                if it.sample {
                    let last = *it.positions.last().unwrap();
                    let ctx: Vec<u32> = (0..=last).map(|p| *self.mem.get(&(it.block_table[p / self.bt], p % self.bt)).expect("unwritten KV")).collect();
                    out.sampled.push((it.seq, next_token(&ctx)));
                }
            }
            out
        }
    }

    fn drive(s: &mut Scheduler, m: &mut Mock, now: u64) -> Result<Vec<Event>, SchedError> {
        let plan = s.plan(now);
        if plan.items.is_empty() { return Ok(s.drain_events()); }
        assert!(plan.total_tokens() <= s.config().max_batch_tokens, "step over budget");
        let res = m.run(s, &plan);
        let ev = s.commit(&plan, &res)?;
        s.check_invariants().expect("invariants");
        assert!(s.allocator().used() <= s.allocator().total());
        Ok(ev)
    }

    #[derive(Default)]
    struct Collected { tokens: BTreeMap<SeqId, Vec<u32>>, finished: BTreeMap<SeqId, Vec<FinishReason>>, preempted: usize }
    impl Collected {
        fn add(&mut self, ev: Vec<Event>) {
            for e in ev {
                match e {
                    Event::Token { seq, token } => self.tokens.entry(seq).or_default().push(token),
                    Event::Finished { seq, reason, .. } => self.finished.entry(seq).or_default().push(reason),
                    Event::Preempted { .. } => self.preempted += 1,
                }
            }
        }
    }

    fn cfg(bt: usize, par: usize, batch: usize, chunk: usize) -> SchedConfig {
        SchedConfig { block_tokens: bt, n_parallel: par, max_batch_tokens: batch, prefill_chunk: chunk, max_context: 512,
            queue_limit: 10_000, aging_period: 8, prefix_cache: true }
    }

    /// Hundreds of random requests against a deliberately small pool.
    fn simulate(seed: u64, n_req: usize, blocks: usize, cancel: bool) -> (Vec<(SeqId, Request)>, Collected, Scheduler, Mock, u64) {
        let mut rng = Rng(seed);
        let c = cfg(4, 4, 24, 8);
        let mut s = Scheduler::with_blocks(c.clone(), blocks).unwrap();
        let mut m = Mock::new(4);
        let mut col = Collected::default();
        let shared: Vec<u32> = (0..24).map(|_| rng.below(50) as u32).collect();
        let mut reqs: Vec<(SeqId, Request)> = vec![];
        let mut submitted = 0;
        let mut step = 0u64;
        while submitted < n_req || !s.is_idle() {
            step += 1;
            assert!(step < 400_000, "scheduler did not finish: starvation or livelock");
            for _ in 0..rng.below(3) {
                if submitted >= n_req { break; }
                let mut prompt = if rng.below(3) == 0 { shared.clone() } else { vec![] };
                for _ in 0..1 + rng.below(40) { prompt.push(rng.below(50) as u32); }
                let mut r = Request::new(prompt, 1 + rng.below(30) as usize);
                r.priority = rng.below(4) as i32 - 1;
                if rng.below(4) == 0 { r.stop_tokens = (0..40).map(|_| rng.below(VOCAB as u64) as u32).collect(); }
                if let Ok(id) = s.submit(r.clone()) { reqs.push((id, r)); }
                submitted += 1;
            }
            if cancel && rng.below(15) == 0 && !reqs.is_empty() {
                let (id, _) = &reqs[rng.below(reqs.len() as u64) as usize];
                s.cancel(*id);
            }
            col.add(drive(&mut s, &mut m, step).unwrap());
        }
        col.add(s.drain_events());
        (reqs, col, s, m, step)
    }

    #[test]
    fn simulation_every_request_finishes_once_with_the_right_tokens() {
        for seed in [1u64, 2, 3, 4, 5, 6] {
            let (reqs, col, mut s, m, _) = simulate(seed * 0x1234_5679, 250, 40, false);
            assert!(reqs.len() > 200);
            for (id, r) in &reqs {
                assert_eq!(col.finished.get(id).map(Vec::len), Some(1), "request {id} must finish exactly once");
                let want = reference(r, 512);
                assert_eq!(col.tokens.get(id).cloned().unwrap_or_default(), want, "seed {seed} request {id}: wrong or reordered tokens");
                let why = col.finished[id][0];
                assert!(matches!(why, FinishReason::Length | FinishReason::Stop));
            }
            assert!(m.max_step <= 24 && m.max_prefill <= 8, "budget {} chunk {}", m.max_step, m.max_prefill);
            // no leaks: once the cache is emptied nothing is left
            s.check_invariants().unwrap();
            while { let c = &mut s.cache; c.evict_one(&mut s.alloc) } {}
            assert_eq!(s.allocator().used(), 0, "blocks leaked");
            let st = s.stats();
            assert_eq!(st.completed as usize, reqs.len());
            assert!(st.batch_utilisation() > 0.0 && st.batch_utilisation() <= 1.0);
        }
    }

    #[test]
    fn simulation_survives_cancellation_and_memory_pressure() {
        let mut total_preempted = 0;
        for seed in [11u64, 12, 13] {
            let (reqs, col, s, _m, _) = simulate(seed * 0xabcd_ef01, 300, 30, true);
            total_preempted += col.preempted;
            for (id, r) in &reqs {
                assert_eq!(col.finished.get(id).map(Vec::len), Some(1), "request {id}");
                let got = col.tokens.get(id).cloned().unwrap_or_default();
                let want = reference(r, 512);
                if col.finished[id][0] == FinishReason::Cancelled { assert!(want.starts_with(&got), "cancelled output is a prefix"); }
                else { assert_eq!(got, want); }
            }
            assert!(s.stats().cancelled > 0);
            assert!(s.is_idle());
        }
        assert!(total_preempted > 0, "the small pool should have forced preemptions");
    }

    #[test]
    fn simulation_is_deterministic() {
        let run = || { let (_, col, s, _, steps) = simulate(77, 120, 32, true); (col.tokens, col.finished, col.preempted, steps, s.stats()) };
        assert_eq!(run(), run());
    }

    #[test]
    fn prefix_cache_serves_repeated_prompts_and_saves_prefill() {
        let c = cfg(4, 2, 16, 16);
        let mut s = Scheduler::with_blocks(c, 64).unwrap();
        let mut m = Mock::new(4);
        let prompt: Vec<u32> = (0..33).map(|i| i as u32 % 7).collect();
        let mut col = Collected::default();
        let mut prefill_tokens = vec![];
        for round in 0..3u64 {
            let id = s.submit(Request::new(prompt.clone(), 5)).unwrap();
            let mut ptoks = 0;
            while !s.is_idle() {
                let plan = s.plan(round);
                ptoks += plan.items.iter().filter(|i| i.kind == StepKind::Prefill).map(|i| i.tokens.len()).sum::<usize>();
                let res = m.run(&s, &plan);
                col.add(s.commit(&plan, &res).unwrap());
            }
            prefill_tokens.push(ptoks);
            assert_eq!(col.tokens[&id], reference(&Request::new(prompt.clone(), 5), 512));
        }
        assert_eq!(prefill_tokens[0], 33);
        assert!(prefill_tokens[1] < 33 && prefill_tokens[2] < 33, "{prefill_tokens:?}");
        assert!(s.stats().prefix_hit_rate() > 0.3);
    }

    #[test]
    fn long_prefill_is_chunked_and_never_stalls_decoding() {
        let c = cfg(4, 2, 12, 10);
        let mut s = Scheduler::with_blocks(c, 400).unwrap();
        let mut m = Mock::new(4);
        let mut col = Collected::default();
        let short = s.submit(Request::new(vec![1, 2, 3], 60)).unwrap();
        // let the short request reach decoding
        for step in 0..3 { col.add(drive(&mut s, &mut m, step).unwrap()); }
        let long_prompt: Vec<u32> = (0..300).map(|i| (i * 7 % 31) as u32).collect();
        let long = s.submit(Request::new(long_prompt.clone(), 4)).unwrap();
        let mut steps_with_both = 0;
        let mut step = 10;
        while !s.is_idle() {
            step += 1;
            let plan = s.plan(step);
            let res = m.run(&s, &plan);
            let long_prefilling = plan.items.iter().any(|i| i.seq == long && i.kind == StepKind::Prefill);
            let short_decoding = plan.items.iter().any(|i| i.seq == short && i.kind == StepKind::Decode);
            if long_prefilling {
                assert!(short_decoding || !s.running().contains(&short), "decode stalled by prefill");
                steps_with_both += 1;
            }
            for i in &plan.items { if i.kind == StepKind::Prefill { assert!(i.tokens.len() <= 10); } }
            assert!(plan.total_tokens() <= 12);
            col.add(s.commit(&plan, &res).unwrap());
            s.check_invariants().unwrap();
        }
        assert!(steps_with_both >= 25, "300 tokens at <=10 per step need many steps, got {steps_with_both}");
        assert_eq!(col.tokens[&long], reference(&Request::new(long_prompt, 4), 512));
        assert_eq!(col.tokens[&short].len(), 60);
    }

    #[test]
    fn preemption_recovers_with_the_right_output() {
        // two long sequences cannot share a pool that fits just over one
        let c = cfg(4, 2, 8, 8);
        let mut s = Scheduler::with_blocks(c, 14).unwrap();
        let mut m = Mock::new(4);
        let mut col = Collected::default();
        let a = Request { priority: 1, ..Request::new((0..20).collect(), 30) };
        let b = Request { priority: 0, ..Request::new((100..120).collect(), 30) };
        let ia = s.submit(a.clone()).unwrap();
        let ib = s.submit(b.clone()).unwrap();
        let mut step = 0;
        while !s.is_idle() { step += 1; assert!(step < 5000); col.add(drive(&mut s, &mut m, step).unwrap()); }
        assert!(col.preempted > 0, "expected the lower-priority sequence to be preempted");
        assert!(s.stats().preemptions as usize >= col.preempted);
        assert_eq!(col.tokens[&ia], reference(&a, 512));
        assert_eq!(col.tokens[&ib], reference(&b, 512));
    }

    #[test]
    fn low_priority_requests_are_not_starved_by_a_stream_of_urgent_ones() {
        let c = SchedConfig { aging_period: 4, ..cfg(4, 1, 8, 8) };
        let mut s = Scheduler::with_blocks(c, 40).unwrap();
        let mut m = Mock::new(4);
        let low = s.submit(Request { priority: -5, ..Request::new(vec![1, 2, 3], 3) }).unwrap();
        let mut done_at = None;
        let mut step = 0u64;
        while done_at.is_none() {
            step += 1;
            assert!(step < 2000, "starved");
            // a new urgent request arrives every step
            let _ = s.submit(Request { priority: 3, ..Request::new(vec![9, 9, step as u32 % 50], 2) });
            for e in drive(&mut s, &mut m, step).unwrap() {
                if matches!(e, Event::Finished { seq, .. } if seq == low) { done_at = Some(step); }
            }
        }
        assert!(done_at.unwrap() <= 4 * 9 + 40, "took {done_at:?} steps");
    }

    #[test]
    fn cancellation_frees_blocks_at_once() {
        let mut s = Scheduler::with_blocks(cfg(4, 2, 16, 16), 20).unwrap();
        let mut m = Mock::new(4);
        let id = s.submit(Request::new((0..30).collect(), 20)).unwrap();
        let waiting = s.submit(Request::new(vec![5; 9], 2)).unwrap();
        let _ = drive(&mut s, &mut m, 1).unwrap();
        assert!(s.allocator().used() > 0);
        let before = s.allocator().used();
        assert!(s.cancel(id));
        assert!(!s.cancel(id), "second cancel is a no-op");
        assert!(s.allocator().used() <= before);
        assert!(s.cancel(waiting));
        s.check_invariants().unwrap();
        assert_eq!(s.allocator().used(), s.cache().len(), "only the cache may still hold blocks");
        let ev = s.drain_events();
        assert_eq!(ev.iter().filter(|e| matches!(e, Event::Finished { reason: FinishReason::Cancelled, .. })).count(), 2);
        assert!(s.is_idle());
    }

    #[test]
    fn cancelling_between_plan_and_commit_is_harmless() {
        let mut s = Scheduler::with_blocks(cfg(4, 2, 16, 16), 20).unwrap();
        let mut m = Mock::new(4);
        let a = s.submit(Request::new(vec![1, 2, 3, 4, 5], 4)).unwrap();
        let b = s.submit(Request::new(vec![6, 7, 8, 9, 10], 4)).unwrap();
        let plan = s.plan(1);
        let res = m.run(&s, &plan);
        s.cancel(a);
        let ev = s.commit(&plan, &res).unwrap();
        assert!(ev.iter().all(|e| !matches!(e, Event::Token { seq, .. } if *seq == a)));
        assert!(ev.iter().any(|e| matches!(e, Event::Token { seq, .. } if *seq == b)));
        s.check_invariants().unwrap();
    }

    #[test]
    fn forked_sequences_copy_on_write() {
        let mut s = Scheduler::with_blocks(cfg(4, 3, 16, 16), 40).unwrap();
        let mut m = Mock::new(4);
        let mut col = Collected::default();
        let req = Request::new(vec![3, 1, 4, 1, 5, 9], 12);
        let parent = s.submit(req.clone()).unwrap();
        for step in 1..=2 { col.add(drive(&mut s, &mut m, step).unwrap()); }
        assert!(s.generated(parent).unwrap() >= 1);
        let child = s.fork(parent).unwrap();
        let mut step = 2;
        let mut copies = 0;
        while !s.is_idle() {
            step += 1;
            let plan = s.plan(step);
            copies += plan.copies.len();
            let res = m.run(&s, &plan);
            col.add(s.commit(&plan, &res).unwrap());
            s.check_invariants().unwrap();
        }
        assert!(copies >= 1, "the shared partial block must be copied before a write");
        // the child continues exactly like the parent would have
        let want = reference(&req, 512);
        assert_eq!(col.tokens[&parent], want);
        assert_eq!(col.tokens[&child], want[want.len() - col.tokens[&child].len()..]);
        assert!(s.stats().cow_copies >= 1);
    }

    #[test]
    fn deadlines_stop_waiting_and_running_requests() {
        let mut s = Scheduler::with_blocks(cfg(4, 1, 8, 8), 30).unwrap();
        let mut m = Mock::new(4);
        let run = s.submit(Request { deadline: Some(5), ..Request::new(vec![1, 2], 100) }).unwrap();
        let wait = s.submit(Request { deadline: Some(3), ..Request::new(vec![3, 4], 100) }).unwrap();
        let mut col = Collected::default();
        for now in 1..=8 { col.add(drive(&mut s, &mut m, now).unwrap()); }
        assert_eq!(col.finished[&run], vec![FinishReason::Deadline]);
        assert_eq!(col.finished[&wait], vec![FinishReason::Deadline]);
        assert!(s.is_idle());
        assert_eq!(s.stats().expired, 2);
        s.check_invariants().unwrap();
    }

    #[test]
    fn stop_tokens_and_context_limit_end_a_sequence() {
        let mut c = cfg(4, 1, 8, 8);
        c.max_context = 12;
        let mut s = Scheduler::with_blocks(c, 10).unwrap();
        let mut m = Mock::new(4);
        let mut col = Collected::default();
        let r = Request::new(vec![1, 2, 3, 4, 5, 6, 7, 8], usize::MAX); // clamped to 4 new tokens
        let id = s.submit(r.clone()).unwrap();
        let mut now = 0;
        while !s.is_idle() { now += 1; col.add(drive(&mut s, &mut m, now).unwrap()); }
        assert_eq!(col.tokens[&id].len(), 4);
        assert_eq!(col.finished[&id], vec![FinishReason::Length]);
        // stop token: the first token the model would emit
        let first = reference(&Request::new(vec![2, 2], 5), 12)[0];
        let id = s.submit(Request { stop_tokens: vec![first], ..Request::new(vec![2, 2], 5) }).unwrap();
        while !s.is_idle() { now += 1; col.add(drive(&mut s, &mut m, now).unwrap()); }
        assert_eq!(col.tokens[&id], vec![first]);
        assert_eq!(col.finished[&id], vec![FinishReason::Stop]);
    }

    #[test]
    fn hostile_requests_and_configs_are_clean_errors() {
        let mut s = Scheduler::with_blocks(cfg(4, 2, 8, 8), 5).unwrap();
        assert_eq!(s.submit(Request::new(vec![], 3)), Err(SchedError::EmptyPrompt));
        assert_eq!(s.submit(Request::new(vec![1], 0)), Err(SchedError::ZeroNewTokens));
        assert!(matches!(s.submit(Request::new(vec![1; 600], 1)), Err(SchedError::PromptTooLong { .. })));
        // 5 blocks of 4 tokens = 20 tokens; a 10-token prompt asking for 100 more can never fit
        assert!(matches!(s.submit(Request::new(vec![1; 10], 100)), Err(SchedError::BudgetTooSmall { needed_blocks, total_blocks: 5 }) if needed_blocks > 5));
        assert!(matches!(s.submit(Request::new(vec![1; 10], usize::MAX)), Err(SchedError::BudgetTooSmall { .. })));
        assert!(s.submit(Request::new(vec![u32::MAX; 4], 4)).is_ok());
        assert!(!s.is_idle());
        // an empty scheduler plans nothing and does not hang
        let mut e = Scheduler::with_blocks(cfg(4, 2, 8, 8), 5).unwrap();
        assert!(e.plan(0).items.is_empty());
        assert!(e.drain_events().is_empty());
        // bad configs
        for bad in [SchedConfig { block_tokens: 0, ..cfg(4, 1, 4, 4) }, SchedConfig { n_parallel: 0, ..cfg(4, 1, 4, 4) },
            SchedConfig { max_batch_tokens: 1, n_parallel: 2, ..cfg(4, 1, 4, 4) }, SchedConfig { prefill_chunk: 0, ..cfg(4, 1, 4, 4) },
            SchedConfig { max_context: 1, ..cfg(4, 1, 4, 4) }, SchedConfig { queue_limit: 0, ..cfg(4, 1, 4, 4) },
            SchedConfig { aging_period: 0, ..cfg(4, 1, 4, 4) }] {
            assert!(matches!(Scheduler::with_blocks(bad, 4), Err(SchedError::BadConfig(_))));
        }
        assert!(Scheduler::new(cfg(4, 1, 4, 4), 0, 1).is_err());
        assert!(Scheduler::new(cfg(4, 1, 4, 4), 100, 0).is_err());
        assert!(Scheduler::new(cfg(4, 1, 4, 4), 1000, 10).is_ok());
        assert!(Scheduler::with_blocks(cfg(4, 1, 4, 4), 0).is_err());
        assert!(Scheduler::with_blocks(cfg(4, 1, 4, 4), usize::MAX).is_err());
        // queue limit
        let mut q = Scheduler::with_blocks(SchedConfig { queue_limit: 2, ..cfg(4, 1, 4, 4) }, 8).unwrap();
        q.submit(Request::new(vec![1], 1)).unwrap();
        q.submit(Request::new(vec![1], 1)).unwrap();
        assert_eq!(q.submit(Request::new(vec![1], 1)), Err(SchedError::QueueFull));
    }

    #[test]
    fn inconsistent_results_change_nothing() {
        let mut s = Scheduler::with_blocks(cfg(4, 2, 16, 16), 20).unwrap();
        let mut m = Mock::new(4);
        let id = s.submit(Request::new(vec![1, 2, 3], 4)).unwrap();
        let plan = s.plan(1);
        let good = m.run(&s, &plan);
        let snapshot = (s.allocator().used(), s.generated(id));
        assert!(matches!(s.commit(&plan, &StepResult::default()), Err(SchedError::BadResult(_))));
        let dup = StepResult { sampled: vec![good.sampled[0], good.sampled[0]] };
        assert!(matches!(s.commit(&plan, &dup), Err(SchedError::BadResult(_))));
        let stranger = StepResult { sampled: vec![good.sampled[0], (999, 1)] };
        assert!(matches!(s.commit(&plan, &stranger), Err(SchedError::BadResult(_))));
        assert_eq!((s.allocator().used(), s.generated(id)), snapshot);
        s.check_invariants().unwrap();
        assert!(s.commit(&plan, &good).is_ok());
        assert_eq!(s.commit(&plan, &good), Err(SchedError::StalePlan));
        assert_eq!(s.fork(12345), Err(SchedError::UnknownSeq(12345)));
    }

    #[test]
    fn metrics_render_as_prometheus_text() {
        let (_, _, s, _, _) = simulate(5, 40, 40, false);
        let st = s.stats();
        assert!(st.tokens_generated > 0 && st.steps > 0);
        assert!(st.tokens_per_second(2.0) > 0.0 && st.tokens_per_second(0.0) == 0.0);
        let text = st.prometheus("corbul_");
        for name in ["requests_completed_total", "queue_depth", "batch_utilisation", "prefix_cache_hit_rate", "kv_blocks_used"] {
            assert!(text.contains(&format!("# TYPE corbul_{name} ")), "{name}");
        }
        assert!(text.lines().filter(|l| !l.starts_with('#')).all(|l| l.split(' ').count() == 2));
    }

    #[test]
    fn executor_trait_drives_a_whole_workload() {
        struct E<'a> { mock: Mock, s: *const Scheduler, _p: std::marker::PhantomData<&'a ()> }
        impl Executor for E<'_> {
            // SAFETY of the raw pointer: only read, while `step` holds no other reference to the scheduler's allocator
            fn run(&mut self, plan: &StepPlan) -> StepResult { self.mock.run(unsafe { &*self.s }, plan) }
        }
        let mut s = Scheduler::with_blocks(cfg(4, 3, 16, 8), 60).unwrap();
        let ids: Vec<SeqId> = (0..8).map(|i| s.submit(Request::new(vec![i as u32 + 1; 5 + i], 6)).unwrap()).collect();
        let mut e = E { mock: Mock::new(4), s: &s as *const Scheduler, _p: std::marker::PhantomData };
        let mut col = Collected::default();
        let mut now = 0;
        while !s.is_idle() {
            now += 1;
            e.s = &s as *const Scheduler;
            col.add(s.step(now, &mut e).unwrap());
            assert!(now < 1000);
        }
        for (i, id) in ids.iter().enumerate() { assert_eq!(col.tokens[id].len(), 6, "request {i}"); }
    }
}

#[cfg(test)]
mod exec_tests {
    use super::*;
    use crate::model::tests::{base_spec, model_of, tiny};
    use std::sync::Mutex;

    type Logs = Arc<Mutex<HashMap<SeqId, Vec<Vec<f32>>>>>;

    struct Ref { out: Vec<u32>, logits: Vec<Vec<f32>>, reason: FinishReason }

    /// One request run alone, the way a sequential server would: one prefill, then token by token.
    fn reference(model: &Model, req: &Request, params: &SamplerParams, kv: KvKind, max_context: usize) -> Ref {
        let toks: Vec<usize> = req.prompt.iter().map(|&t| t as usize).collect();
        let max_new = req.max_new_tokens.min(max_context - req.prompt.len());
        let mut st = model.new_state_with(kv);
        let mut sampler = params.build();
        let mut logits = model.forward_batch(&mut st, &toks).unwrap();
        sampler.observe_all(&toks);
        let (mut out, mut logs) = (vec![], vec![]);
        let reason = loop {
            logs.push(logits.clone());
            let t = sampler.sample(&logits);
            sampler.observe(t);
            out.push(t as u32);
            if req.stop_tokens.contains(&(t as u32)) { break FinishReason::Stop; }
            if out.len() >= max_new { break FinishReason::Length; }
            if toks.len() + out.len() >= max_context { break FinishReason::ContextFull; }
            logits = model.forward(&mut st, t).unwrap();
        };
        Ref { out, logits: logs, reason }
    }

    fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }

    struct R(u64);
    impl R {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn below(&mut self, n: u64) -> u64 { self.next() % n.max(1) }
    }

    fn requests(n: usize, seed: u64, vocab: u32) -> Vec<Request> {
        let mut r = R(seed.wrapping_mul(0x9E37_79B9) | 1);
        (0..n).map(|_| {
            let len = 3 + r.below(10) as usize;
            let prompt: Vec<u32> = (0..len).map(|_| r.below(vocab as u64) as u32).collect();
            let mut q = Request::new(prompt, 2 + r.below(11) as usize);
            if r.below(3) == 0 { q.stop_tokens = vec![3, 7, 11]; }
            q
        }).collect()
    }

    fn cfg(parallel: usize, batch: usize, chunk: usize) -> SchedConfig {
        SchedConfig { block_tokens: 4, n_parallel: parallel, max_batch_tokens: batch, prefill_chunk: chunk, max_context: 32,
            queue_limit: 64, aging_period: 4, prefix_cache: true }
    }

    struct Outcome { tokens: HashMap<SeqId, Vec<u32>>, reasons: HashMap<SeqId, FinishReason>, steps: u64, logs: Logs }

    fn attach_logs(e: &mut BatchEngine) -> Logs {
        let logs: Logs = Arc::new(Mutex::new(HashMap::new()));
        let l2 = logs.clone();
        e.executor_mut().set_logit_observer(Some(Box::new(move |seq, _pos, l| { l2.lock().unwrap().entry(seq).or_default().push(l.to_vec()); })));
        logs
    }

    /// Step to the end, checking the scheduler's invariants after every step.
    fn drive(e: &mut BatchEngine, logs: Logs) -> Outcome {
        let (mut tokens, mut reasons, mut steps) = (HashMap::<SeqId, Vec<u32>>::new(), HashMap::new(), 0u64);
        while !e.is_idle() {
            let rep = e.step().unwrap();
            steps += 1;
            e.scheduler().check_invariants().unwrap();
            assert!(rep.failures.is_empty(), "{:?}", rep.failures);
            for ev in rep.events {
                match ev {
                    Event::Token { seq, token } => tokens.entry(seq).or_default().push(token),
                    Event::Finished { seq, reason, .. } => { reasons.insert(seq, reason); }
                    Event::Preempted { .. } => {}
                }
            }
            assert!(steps < 5000, "no progress");
        }
        Outcome { tokens, reasons, steps, logs }
    }

    fn check_all(model: &Model, ids: &[(SeqId, Request, SamplerParams)], o: &Outcome, kv: KvKind, max_context: usize, what: &str) {
        for (i, (id, req, params)) in ids.iter().enumerate() {
            let want = reference(model, req, params, kv, max_context);
            assert_eq!(o.tokens.get(id).cloned().unwrap_or_default(), want.out, "{what}: request {i} tokens");
            assert_eq!(o.reasons.get(id), Some(&want.reason), "{what}: request {i} finish reason");
            let got = o.logs.lock().unwrap().get(id).cloned().unwrap_or_default();
            assert_eq!(got.len(), want.logits.len(), "{what}: request {i} sample count");
            for (k, (a, b)) in got.iter().zip(&want.logits).enumerate() {
                assert_eq!(bits(a), bits(b), "{what}: request {i} logits of sample {k}");
            }
        }
    }

    fn submit_all(e: &mut BatchEngine, reqs: &[Request], params: impl Fn(usize) -> SamplerParams) -> Vec<(SeqId, Request, SamplerParams)> {
        reqs.iter().enumerate().map(|(i, r)| {
            let p = params(i);
            (e.submit(r.clone(), p).unwrap(), r.clone(), p)
        }).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case(model: Model, sc: SchedConfig, blocks: usize, ec: ExecConfig, n: usize, seed: u64, params: impl Fn(usize) -> SamplerParams, what: &str) -> BatchEngine {
        let model = Arc::new(model);
        let mut e = BatchEngine::with_blocks(model.clone(), sc, blocks, ec).unwrap();
        let logs = attach_logs(&mut e);
        let reqs = requests(n, seed, model.spec.vocab as u32);
        let ids = submit_all(&mut e, &reqs, params);
        let o = drive(&mut e, logs);
        check_all(&model, &ids, &o, ec.kv, e.scheduler().config().max_context, what);
        e
    }

    #[test]
    fn greedy_batches_equal_sequential_generation_bit_for_bit() {
        for (chunk, batch, par) in [(5, 7, 3), (3, 4, 4), (256, 512, 2), (1, 4, 4)] {
            let e = run_case(tiny(4, 2, true), cfg(par, batch, chunk), 64, ExecConfig::default(), 9, 7 + chunk as u64, |_| SamplerParams::greedy(), "greedy");
            assert!(e.scheduler().stats().completed >= 9);
        }
    }

    #[test]
    fn a_small_pool_forces_preemption_and_the_results_do_not_change() {
        let e = run_case(tiny(4, 2, true), cfg(4, 6, 4), 9, ExecConfig::default(), 12, 3, |_| SamplerParams::greedy(), "preempted");
        let st = e.scheduler().stats();
        assert!(st.preemptions > 0, "the pool was meant to be too small: {st:?}");
        assert!(e.executor().stats().recomputed_tokens > 0 || e.executor().stats().reused_tokens > 0);
    }

    #[test]
    fn sampled_requests_match_the_same_sampler_run_alone() {
        let params = |i: usize| {
            let mut p = SamplerParams::sampled(0.7 + 0.1 * (i % 4) as f32, [0, 5, 3][i % 3], [1.0, 0.9][i % 2], 100 + i as u64);
            if i.is_multiple_of(3) { p.repeat_penalty = 1.2; p.repeat_last_n = 8; }
            p
        };
        run_case(tiny(4, 2, true), cfg(3, 8, 4), 64, ExecConfig::default(), 10, 11, params, "sampled");
        run_case(tiny(4, 2, true), cfg(4, 6, 4), 9, ExecConfig::default(), 10, 12, params, "sampled and preempted");
    }

    #[test]
    fn every_kv_precision_matches_its_own_sequential_run() {
        for kv in [KvKind::F16, KvKind::Q8] {
            let ec = ExecConfig { kv, ..ExecConfig::default() };
            run_case(tiny(4, 2, true), cfg(3, 7, 5), 64, ec, 8, 21, |_| SamplerParams::greedy(), "kv");
            run_case(tiny(4, 2, true), cfg(4, 6, 4), 9, ec, 8, 22, |_| SamplerParams::greedy(), "kv preempted");
        }
    }

    #[test]
    fn a_mixture_of_experts_model_matches_too() {
        run_case(crate::model::tests::accel_and_prefix::moe_model(), cfg(3, 6, 4), 64, ExecConfig::default(), 6, 31, |_| SamplerParams::greedy(), "moe");
    }

    #[test]
    fn a_wrapped_sliding_window_cache_is_recomputed_not_trusted() {
        let mk = || { let mut sp = base_spec(4, 2, true); sp.sliding_window = 3; sp.sliding_layers = vec![true, false, true]; sp.rope_theta_local = 500.0; model_of(sp, false) };
        let e = run_case(mk(), cfg(4, 6, 4), 9, ExecConfig::default(), 8, 41, |_| SamplerParams::greedy(), "sliding window");
        assert!(e.scheduler().stats().completed >= 8);
    }

    #[test]
    fn running_the_items_of_a_step_on_their_own_threads_changes_nothing() {
        let ec = ExecConfig { parallel_items: true, ..ExecConfig::default() };
        run_case(tiny(4, 2, true), cfg(4, 8, 4), 64, ec, 10, 51, |_| SamplerParams::greedy(), "parallel items");
        run_case(tiny(4, 2, true), cfg(4, 6, 4), 9, ec, 10, 52, |i| SamplerParams::sampled(0.9, 4, 1.0, i as u64), "parallel items, preempted");
    }

    #[test]
    fn a_shared_prefix_is_served_from_a_retired_cache() {
        let model = Arc::new(tiny(4, 2, true));
        let mut e = BatchEngine::with_blocks(model.clone(), cfg(2, 16, 8), 64, ExecConfig::default()).unwrap();
        let logs = attach_logs(&mut e);
        let shared: Vec<u32> = vec![5, 9, 2, 14, 7, 1, 3, 11, 6];
        let mut ids = vec![];
        for tail in [[4u32, 8], [12, 0], [4, 8]] {
            let mut p = shared.clone();
            p.extend(tail);
            let req = Request::new(p, 5);
            ids.push((e.submit(req.clone(), SamplerParams::greedy()).unwrap(), req, SamplerParams::greedy()));
            // one at a time, so each finished cache is there for the next
            while !e.is_idle() { e.step().unwrap(); }
        }
        // re-run the same engine's outputs through the reference by replaying the events
        let o = {
            // events were consumed by the loop above, so rebuild them from a second identical run
            let mut e2 = BatchEngine::with_blocks(model.clone(), cfg(2, 16, 8), 64, ExecConfig::default()).unwrap();
            let logs2 = attach_logs(&mut e2);
            let mut ids2 = vec![];
            let mut out = Outcome { tokens: HashMap::new(), reasons: HashMap::new(), steps: 0, logs: logs2.clone() };
            for (_, req, p) in &ids {
                let id = e2.submit(req.clone(), *p).unwrap();
                ids2.push((id, req.clone(), *p));
                while !e2.is_idle() {
                    let rep = e2.step().unwrap();
                    out.steps += 1;
                    for ev in rep.events {
                        match ev {
                            Event::Token { seq, token } => out.tokens.entry(seq).or_default().push(token),
                            Event::Finished { seq, reason, .. } => { out.reasons.insert(seq, reason); }
                            Event::Preempted { .. } => {}
                        }
                    }
                }
            }
            check_all(&model, &ids2, &out, KvKind::F32, 32, "prefix");
            assert!(e2.scheduler().stats().prefix_hit_tokens > 0, "the scheduler should have found the shared blocks");
            assert!(e2.executor().stats().reused_tokens >= 8, "{:?}", e2.executor().stats());
            out
        };
        assert!(o.steps > 0);
        drop(logs);
        // the very same prompt twice in a row reuses nearly everything
        assert!(e.executor().stats().reused_tokens > 0);
    }

    #[test]
    fn a_prefix_hit_without_a_cache_is_recomputed_exactly() {
        // retired pool of zero states: the scheduler still reports hits, the executor must recompute
        let ec = ExecConfig { retired_max_states: 0, ..ExecConfig::default() };
        let model = Arc::new(tiny(4, 2, true));
        let mut e = BatchEngine::with_blocks(model.clone(), cfg(2, 16, 8), 64, ec).unwrap();
        let logs = attach_logs(&mut e);
        let shared: Vec<u32> = vec![5, 9, 2, 14, 7, 1, 3, 11, 6, 2];
        let mut ids = vec![];
        let mut out = Outcome { tokens: HashMap::new(), reasons: HashMap::new(), steps: 0, logs };
        for tail in [3u32, 9] {
            let mut p = shared.clone();
            p.push(tail);
            let req = Request::new(p, 4);
            let id = e.submit(req.clone(), SamplerParams::greedy()).unwrap();
            ids.push((id, req, SamplerParams::greedy()));
            while !e.is_idle() {
                for ev in e.step().unwrap().events {
                    match ev {
                        Event::Token { seq, token } => out.tokens.entry(seq).or_default().push(token),
                        Event::Finished { seq, reason, .. } => { out.reasons.insert(seq, reason); }
                        Event::Preempted { .. } => {}
                    }
                }
            }
        }
        check_all(&model, &ids, &out, KvKind::F32, 32, "recomputed hit");
        assert!(e.scheduler().stats().prefix_hit_tokens > 0);
        assert!(e.executor().stats().recomputed_tokens > 0);
        assert_eq!(e.executor().stats().retired_states, 0);
    }

    #[test]
    fn cancelling_one_request_leaves_the_others_exact_and_frees_its_blocks() {
        let model = Arc::new(tiny(4, 2, true));
        let mut e = BatchEngine::with_blocks(model.clone(), cfg(3, 7, 4), 64, ExecConfig::default()).unwrap();
        let logs = attach_logs(&mut e);
        let reqs = requests(6, 61, 20);
        let ids = submit_all(&mut e, &reqs, |_| SamplerParams::greedy());
        let victim = ids[2].0;
        let (mut tokens, mut reasons) = (HashMap::<SeqId, Vec<u32>>::new(), HashMap::new());
        let mut steps = 0;
        while !e.is_idle() {
            if steps == 3 { assert!(e.cancel(victim)); assert!(!e.cancel(victim)); }
            let rep = e.step().unwrap();
            steps += 1;
            e.scheduler().check_invariants().unwrap();
            for ev in rep.events {
                match ev {
                    Event::Token { seq, token } => tokens.entry(seq).or_default().push(token),
                    Event::Finished { seq, reason, .. } => { reasons.insert(seq, reason); }
                    Event::Preempted { .. } => {}
                }
            }
        }
        assert_eq!(reasons[&victim], FinishReason::Cancelled);
        let rest: Vec<_> = ids.iter().filter(|(id, ..)| *id != victim).cloned().collect();
        let o = Outcome { tokens, reasons, steps, logs };
        check_all(&model, &rest, &o, KvKind::F32, 32, "after a cancellation");
        let st = e.scheduler().stats();
        assert_eq!(st.cancelled, 1);
        assert_eq!(st.blocks_used, st.cached_blocks, "only the prefix cache may still hold blocks");
    }

    #[test]
    fn a_deadline_ends_a_request_and_the_rest_carry_on() {
        let model = Arc::new(tiny(4, 2, true));
        let mut e = BatchEngine::with_blocks(model.clone(), cfg(2, 8, 4), 64, ExecConfig::default()).unwrap();
        let mut slow = Request::new(vec![1, 2, 3, 4], 20);
        slow.deadline = Some(3);
        let slow_id = e.submit(slow, SamplerParams::greedy()).unwrap();
        let ok = Request::new(vec![5, 6, 7], 6);
        let ok_id = e.submit(ok.clone(), SamplerParams::greedy()).unwrap();
        let mut got = vec![];
        let mut why = HashMap::new();
        let sum = e.run_until_idle(&mut |ev| {
            match ev {
                Event::Token { seq, token } if *seq == ok_id => got.push(*token),
                Event::Finished { seq, reason, .. } => { why.insert(*seq, *reason); }
                _ => {}
            }
            Flow::Continue
        }).unwrap();
        assert!(sum.failures.is_empty());
        assert_eq!(why[&slow_id], FinishReason::Deadline);
        assert_eq!(got, reference(&model, &ok, &SamplerParams::greedy(), KvKind::F32, 32).out);
        e.scheduler().check_invariants().unwrap();
    }

    #[test]
    fn the_streaming_callback_can_cancel_a_request_and_halt_the_run() {
        let model = Arc::new(tiny(4, 2, true));
        let mut e = BatchEngine::with_blocks(model.clone(), cfg(3, 8, 4), 64, ExecConfig::default()).unwrap();
        let reqs = [Request::new(vec![1, 2, 3], 12), Request::new(vec![4, 5, 6, 7], 8)];
        let a = e.submit(reqs[0].clone(), SamplerParams::greedy()).unwrap();
        let b = e.submit(reqs[1].clone(), SamplerParams::greedy()).unwrap();
        let (mut na, mut done_a) = (0, None);
        let mut tb = vec![];
        e.run_until_idle(&mut |ev| match ev {
            Event::Token { seq, .. } if *seq == a => { na += 1; if na == 3 { Flow::CancelSeq } else { Flow::Continue } }
            Event::Token { seq, token } if *seq == b => { tb.push(*token); Flow::Continue }
            Event::Finished { seq, reason, generated } if *seq == a => { done_a = Some((*reason, *generated)); Flow::Continue }
            _ => Flow::Continue,
        }).unwrap();
        let (reason, generated) = done_a.expect("the cancelled request reports its end");
        assert_eq!(reason, FinishReason::Cancelled);
        assert!(generated >= 3);
        assert_eq!(tb, reference(&model, &reqs[1], &SamplerParams::greedy(), KvKind::F32, 32).out);
        assert!(e.is_idle());

        // halt, then resume: the final text is the same as one uninterrupted run
        let mut e = BatchEngine::with_blocks(model.clone(), cfg(2, 8, 4), 64, ExecConfig::default()).unwrap();
        let r = Request::new(vec![9, 8, 7, 6], 10);
        e.submit(r.clone(), SamplerParams::greedy()).unwrap();
        let mut toks = vec![];
        let first = e.run_until_idle(&mut |ev| { if let Event::Token { token, .. } = ev { toks.push(*token); }; if toks.len() == 4 { Flow::Halt } else { Flow::Continue } }).unwrap();
        assert!(first.halted && !e.is_idle());
        e.run_until_idle(&mut |ev| { if let Event::Token { token, .. } = ev { toks.push(*token); } Flow::Continue }).unwrap();
        assert_eq!(toks, reference(&model, &r, &SamplerParams::greedy(), KvKind::F32, 32).out);
    }

    #[test]
    fn a_forked_sequence_continues_exactly_like_its_parent() {
        let model = Arc::new(tiny(4, 2, true));
        let mut e = BatchEngine::with_blocks(model.clone(), cfg(3, 8, 4), 64, ExecConfig::default()).unwrap();
        let r = Request::new(vec![2, 4, 6, 8, 10], 10);
        let parent = e.submit(r.clone(), SamplerParams::greedy()).unwrap();
        let mut child = None;
        let (mut p, mut c) = (vec![], vec![]);
        let mut at_fork = 0;
        while !e.is_idle() {
            if child.is_none() && p.len() == 3 { at_fork = p.len(); child = Some(e.fork(parent).unwrap()); }
            let rep = e.step().unwrap();
            e.scheduler().check_invariants().unwrap();
            for ev in rep.events {
                if let Event::Token { seq, token } = ev { if seq == parent { p.push(token) } else { c.push(token) } }
            }
        }
        let want = reference(&model, &r, &SamplerParams::greedy(), KvKind::F32, 32).out;
        assert_eq!(p, want);
        assert_eq!(c, want[at_fork..], "the child shares the parent's history and continues identically");
    }

    #[test]
    fn hostile_requests_are_errors_and_a_bad_plan_is_a_failure_not_a_panic() {
        let model = Arc::new(tiny(4, 2, true));
        let mut e = BatchEngine::with_blocks(model.clone(), cfg(2, 8, 4), 16, ExecConfig::default()).unwrap();
        assert!(matches!(e.submit(Request::new(vec![1, 99], 3), SamplerParams::greedy()), Err(EngineError::Model(ModelError::Token(99)))));
        assert!(matches!(e.submit(Request::new(vec![], 3), SamplerParams::greedy()), Err(EngineError::Sched(SchedError::EmptyPrompt))));
        assert!(matches!(e.submit(Request::new(vec![1; 40], 3), SamplerParams::greedy()), Err(EngineError::Sched(SchedError::PromptTooLong { .. }))));
        assert!(matches!(e.submit(Request::new(vec![1], 0), SamplerParams::greedy()), Err(EngineError::Sched(SchedError::ZeroNewTokens))));
        assert!(matches!(e.fork(77), Err(EngineError::Sched(_))));
        assert!(!e.cancel(77));
        assert!(e.is_idle(), "nothing hostile may stay queued");
        e.scheduler().check_invariants().unwrap();

        // a plan that disagrees with the executor's idea of the sequence
        let mut x = ModelExecutor::new(model.clone(), ExecConfig::default());
        x.register(1, &[1, 2, 3], SamplerParams::greedy()).unwrap();
        let item = |seq, tokens: Vec<u32>, positions: Vec<usize>, sample| StepItem { seq, tokens, positions, block_table: vec![0], sample, kind: StepKind::Prefill };
        let plan = StepPlan { step: 1, copies: vec![], items: vec![
            item(1, vec![1, 2, 9], vec![0, 1, 2], true),       // wrong tokens
            item(2, vec![1], vec![0], true),                   // unknown sequence
            item(1, vec![1, 2, 3], vec![0, 1, 2], false),      // sample flag wrong
            item(1, vec![], vec![], false),                    // empty
            item(1, vec![1, 2, 3, 4], vec![0, 1, 2, 3], true), // past the history
        ] };
        let res = x.run(&plan);
        assert!(res.sampled.is_empty());
        assert_eq!(x.take_failures().len(), 5);
        // and the executor still works afterwards
        let ok = StepPlan { step: 2, copies: vec![], items: vec![item(1, vec![1, 2, 3], vec![0, 1, 2], true)] };
        assert_eq!(x.run(&ok).sampled.len(), 1);
        assert!(x.register(3, &[200], SamplerParams::greedy()).is_err());
    }

    #[test]
    fn the_retired_pool_stays_within_its_limits() {
        let ec = ExecConfig { retired_max_states: 2, ..ExecConfig::default() };
        let e = run_case(tiny(4, 2, true), cfg(2, 8, 4), 64, ec, 12, 71, |_| SamplerParams::greedy(), "small retired pool");
        assert!(e.executor().stats().retired_states <= 2);
        let ec = ExecConfig { retired_max_bytes: 1, ..ExecConfig::default() };
        let e = run_case(tiny(4, 2, true), cfg(2, 8, 4), 64, ec, 6, 72, |_| SamplerParams::greedy(), "tiny retired budget");
        assert_eq!(e.executor().stats().retired_states, 0);
        assert_eq!(e.executor().stats().live_states, 0, "finished sequences hold no state");
    }

    #[test]
    fn kv_bytes_per_token_matches_what_a_state_really_holds() {
        let m = tiny(4, 2, true);
        for kind in [KvKind::F32, KvKind::F16, KvKind::Q8] {
            let mut st = m.new_state_with(kind);
            m.forward(&mut st, 1).unwrap();
            let one = st.kv_bytes();
            m.forward(&mut st, 2).unwrap();
            let step = (st.kv_bytes() - one) as u64;
            let est = kv_bytes_per_token(&m, kind);
            assert!(est >= step && est <= step * 2, "{kind:?}: estimate {est} vs {step}");
        }
        let pool = BatchEngine::new(Arc::new(tiny(4, 2, true)), cfg(2, 8, 4), 64 * 4 * kv_bytes_per_token(&tiny(4, 2, true), KvKind::F32), ExecConfig::default()).unwrap();
        assert_eq!(pool.scheduler().allocator().total(), 64);
    }

    #[test]
    fn batching_needs_far_fewer_steps_than_one_request_at_a_time() {
        let model = Arc::new(tiny(4, 2, true));
        let reqs: Vec<Request> = (0..8).map(|i| Request::new(vec![1 + i as u32, 2, 3, 4, 5, 6], 10)).collect();
        // sequential: one prefill plus nine single-token calls per request
        let t0 = std::time::Instant::now();
        let want: Vec<Vec<u32>> = reqs.iter().map(|r| reference(&model, r, &SamplerParams::greedy(), KvKind::F32, 32).out).collect();
        let seq_secs = t0.elapsed().as_secs_f64();
        let seq_calls: usize = want.iter().map(Vec::len).sum();

        let mut e = BatchEngine::with_blocks(model.clone(), cfg(8, 64, 16), 128, ExecConfig::default()).unwrap();
        let ids: Vec<SeqId> = reqs.iter().map(|r| e.submit(r.clone(), SamplerParams::greedy()).unwrap()).collect();
        let mut got: HashMap<SeqId, Vec<u32>> = HashMap::new();
        let t1 = std::time::Instant::now();
        let sum = e.run_until_idle(&mut |ev| { if let Event::Token { seq, token } = ev { got.entry(*seq).or_default().push(*token); } Flow::Continue }).unwrap();
        let eng_secs = t1.elapsed().as_secs_f64();
        for (i, id) in ids.iter().enumerate() { assert_eq!(got[id], want[i], "request {i}"); }
        let st = e.scheduler().stats();
        assert_eq!(st.tokens_generated as usize, seq_calls);
        assert!(sum.steps as usize * 4 < seq_calls, "{} steps for {seq_calls} sequential model calls", sum.steps);
        assert!(st.batch_utilisation() > 0.0);
        eprintln!("throughput (synthetic, debug build, shared machine): sequential {seq_secs:.3}s, engine {eng_secs:.3}s, {} steps for {seq_calls} tokens, utilisation {:.2}",
            sum.steps, st.batch_utilisation());
    }
}
