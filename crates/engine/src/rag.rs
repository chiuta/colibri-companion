//! Local retrieval stack: embeddings, chunking, an HNSW index, BM25, rank fusion and citations.
//!
//! Everything here is pure Rust with no network and no extra dependencies:
//!
//! * [`Embedder`] – anything that turns text into a vector. [`ModelEmbedder`] pools the hidden
//!   states of any [`Model`] (mean / last token / first token, L2 normalisation, optional
//!   Matryoshka truncation); [`HashEmbedder`] is a deterministic hashing-trick bag of word and
//!   character n-grams that needs no model at all.
//! * [`chunk_text`] – sentence- and paragraph-aware chunking with overlap that keeps byte offsets
//!   into the original document, and does not split Romanian abbreviations such as "dr." or "nr.".
//! * [`Hnsw`] – an approximate nearest-neighbour graph (cosine or dot product, `f32` or int8
//!   vectors, tombstone deletion, deterministic level draws, checksummed file format that is safe
//!   to load from an untrusted file), with an exact [`Hnsw::brute_force`] to cross-check against.
//! * [`Bm25Index`] – keyword ranking with diacritic folding (ș/ş, ț/ţ, ă, â, î all match their
//!   plain forms), fused with the vectors through reciprocal rank fusion in [`KnowledgeBase`].
//! * [`CitedContext`] – the retrieved passages as numbered sources `[S1]`, `[S2]`, … that plug
//!   straight into [`crate::quotelock`]: its citations map back to byte ranges in the documents.
//!
//! The embeddings that [`ModelEmbedder`] produces are only as good as the model: a plain
//! generative decoder gives usable but not state-of-the-art vectors, a real embedding model
//! (loaded the same way) gives much better ones. Nothing here was run against a real checkpoint.

use crate::model::{Model, ModelError};
use crate::quotelock::{Citation, Options as QuoteOptions, QuoteError, QuoteIndex};
use crate::receipt::sha256;
use crate::tokenizer::Tokenizer;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::path::Path;

// ───────────────────────────── errors ─────────────────────────────

/// Everything that can go wrong in this module.
#[derive(Debug, Clone, PartialEq)]
pub enum RagError {
    /// A vector had the wrong length.
    Dim { expected: usize, got: usize },
    /// Nothing to embed or index (empty text, no tokens).
    Empty,
    /// A vector held NaN or an infinity.
    NotFinite,
    /// The model failed while embedding.
    Model(String),
    /// A file was truncated, corrupt, forged or of another version.
    Format(String),
    /// A parameter outside its allowed range.
    Param(String),
    /// Reading or writing a file failed.
    Io(String),
    /// An id that is not in the index.
    Unknown(u64),
}

impl fmt::Display for RagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RagError::Dim { expected, got } => write!(f, "vector has {got} values, expected {expected}"),
            RagError::Empty => write!(f, "nothing to embed"),
            RagError::NotFinite => write!(f, "the vector contains NaN or infinity"),
            RagError::Model(m) => write!(f, "model error: {m}"),
            RagError::Format(m) => write!(f, "bad index file: {m}"),
            RagError::Param(m) => write!(f, "bad parameter: {m}"),
            RagError::Io(m) => write!(f, "i/o error: {m}"),
            RagError::Unknown(i) => write!(f, "no entry with id {i}"),
        }
    }
}
impl std::error::Error for RagError {}
impl From<ModelError> for RagError {
    fn from(e: ModelError) -> Self { RagError::Model(e.to_string()) }
}

type R<T> = Result<T, RagError>;

// ───────────────────────────── vector helpers ─────────────────────────────

/// Scale `v` to unit length in place. A zero (or non-finite) vector is left as it is.
pub fn l2_normalize(v: &mut [f32]) {
    // scale by the largest magnitude first so squares neither overflow (1e30) nor vanish (1e-30)
    let m = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
    if !m.is_finite() || m == 0.0 { return; }
    for x in v.iter_mut() { *x /= m; }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 { for x in v.iter_mut() { *x /= n; } }
}

/// Matryoshka truncation: keep the first `k` values and renormalise. `k` is clamped to `1..=len`.
pub fn matryoshka(v: &[f32], k: usize) -> Vec<f32> {
    let k = k.clamp(1, v.len().max(1)).min(v.len());
    let mut out = v[..k].to_vec();
    l2_normalize(&mut out);
    out
}

/// Plain dot product (the shorter length wins).
pub fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

/// How the per-token hidden states become one vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Average over all tokens (the usual choice for embedding models).
    Mean,
    /// The last token (decoder-style embedders: it has seen the whole text).
    Last,
    /// The first token (BERT-style `[CLS]`).
    Cls,
}

/// Pool `rows` (consecutive token vectors of width `width`) into one vector.
pub fn pool(rows: &[f32], width: usize, how: Pooling) -> Vec<f32> {
    if width == 0 || rows.len() < width { return Vec::new(); }
    let n = rows.len() / width;
    match how {
        Pooling::Cls => rows[..width].to_vec(),
        Pooling::Last => rows[(n - 1) * width..n * width].to_vec(),
        Pooling::Mean => {
            let mut out = vec![0.0f32; width];
            for r in rows.chunks_exact(width) { for (o, x) in out.iter_mut().zip(r) { *o += x; } }
            let inv = 1.0 / n as f32;
            for o in out.iter_mut() { *o *= inv; }
            out
        }
    }
}

// ───────────────────────────── embedders ─────────────────────────────

/// Anything that maps text to a fixed-size vector.
pub trait Embedder {
    /// Length of the vectors this embedder returns.
    fn dim(&self) -> usize;
    /// Embed one text. Errors on empty input.
    fn embed(&self, text: &str) -> R<Vec<f32>>;
    /// Embed several texts (override to batch).
    fn embed_batch(&self, texts: &[&str]) -> R<Vec<Vec<f32>>> { texts.iter().map(|t| self.embed(t)).collect() }
    /// A name for this embedder's behaviour, saved with a [`KnowledgeBase`] and compared when it
    /// is loaded again: vectors from one embedder are meaningless to another. The default only
    /// knows the dimension; implementations should add whatever changes their output.
    fn id(&self) -> String { format!("dim:{}", self.dim()) }
}

type Tokenize<'a> = Box<dyn Fn(&str) -> Vec<usize> + 'a>;

/// Embeddings from the hidden states of a [`Model`].
///
/// The rows are the output of the last layer, each normalised by a parameter-free RMS norm (the
/// model's own final norm weights are not used), then pooled, optionally Matryoshka-truncated and
/// L2-normalised.
pub struct ModelEmbedder<'a> {
    model: &'a Model,
    tokenize: Tokenize<'a>,
    pub pooling: Pooling,
    /// Return unit vectors (default true).
    pub normalize: bool,
    /// Keep only the first `k` dimensions (then renormalise).
    pub matryoshka: Option<usize>,
    /// Longer texts are cut to this many tokens (and to the model's context).
    pub max_tokens: usize,
}

impl<'a> ModelEmbedder<'a> {
    /// Embed with `model`, tokenising with `tok` (a beginning-of-sequence token is added if the
    /// tokenizer has one).
    pub fn new(model: &'a Model, tok: &'a Tokenizer, pooling: Pooling) -> Self {
        Self::with_tokenizer(model, move |t| tok.encode(t, true).into_iter().map(|x| x as usize).collect(), pooling)
    }
    /// Embed with any tokenisation function.
    pub fn with_tokenizer(model: &'a Model, f: impl Fn(&str) -> Vec<usize> + 'a, pooling: Pooling) -> Self {
        ModelEmbedder { model, tokenize: Box::new(f), pooling, normalize: true, matryoshka: None, max_tokens: 512 }
    }
    /// The pooled-from rows: one RMS-normalised hidden vector per token.
    pub fn hidden_rows(&self, tokens: &[usize]) -> R<Vec<f32>> {
        let s = &self.model.spec;
        let limit = self.max_tokens.max(1).min(s.context as usize);
        let tokens = &tokens[..tokens.len().min(limit)];
        if tokens.is_empty() { return Err(RagError::Empty); }
        let h = s.hidden as usize;
        let mut x = Vec::with_capacity(tokens.len() * h);
        for &t in tokens { x.extend(self.model.embed_token(t)?); }
        let mut st = self.model.new_state();
        self.model.run_layers_batch(&mut st, 0..s.layers as usize, &mut x, 0)?;
        self.model.finish_batch(&mut st, tokens);
        for row in x.chunks_exact_mut(h) {
            let ms = row.iter().map(|v| v * v).sum::<f32>() / h as f32;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for v in row.iter_mut() { *v *= inv; }
        }
        if x.iter().any(|v| !v.is_finite()) { return Err(RagError::NotFinite); }
        Ok(x)
    }
}

impl Embedder for ModelEmbedder<'_> {
    fn dim(&self) -> usize {
        let h = self.model.spec.hidden as usize;
        self.matryoshka.map_or(h, |k| k.clamp(1, h))
    }
    fn embed(&self, text: &str) -> R<Vec<f32>> {
        let toks = (self.tokenize)(text);
        let rows = self.hidden_rows(&toks)?;
        let mut v = pool(&rows, self.model.spec.hidden as usize, self.pooling);
        if let Some(k) = self.matryoshka { v.truncate(k.clamp(1, v.len().max(1))); }
        if self.normalize { l2_normalize(&mut v); }
        Ok(v)
    }
    /// Weights, pooling, normalisation, truncation and token limit. The weights are fingerprinted,
    /// which reads them all: call it when saving or loading, not per text. The tokenizer is not
    /// part of the name (a closure has none), so keep using the one that built the index.
    fn id(&self) -> String {
        let fp: String = self.model.fingerprint().iter().take(12).map(|b| format!("{b:02x}")).collect();
        format!("model:{fp}:{:?}:{}:{:?}:{}:{}", self.pooling, self.normalize, self.matryoshka, self.max_tokens, self.dim())
    }
}

/// Fold one character to its search form: lower case, no diacritics (Romanian ă â î ș ş ț ţ and the
/// common Latin accents), so "Țară" and "tara" are the same word.
fn fold_char(c: char, out: &mut String) {
    for l in c.to_lowercase() {
        match l {
            'ă' | 'â' | 'à' | 'á' | 'ä' | 'ã' | 'å' | 'ā' => out.push('a'),
            'î' | 'ì' | 'í' | 'ï' | 'ī' => out.push('i'),
            'ș' | 'ş' | 'ś' | 'š' => out.push('s'),
            'ț' | 'ţ' => out.push('t'),
            'ç' | 'ć' | 'č' => out.push('c'),
            'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ę' => out.push('e'),
            'ñ' | 'ń' => out.push('n'),
            'ò' | 'ó' | 'ô' | 'ö' | 'õ' | 'ő' | 'ø' => out.push('o'),
            'ù' | 'ú' | 'û' | 'ü' | 'ű' | 'ū' => out.push('u'),
            'ý' | 'ÿ' => out.push('y'),
            'ž' | 'ź' | 'ż' => out.push('z'),
            'ß' => out.push_str("ss"),
            other => out.push(other),
        }
    }
}

/// Split into lower-case, diacritic-free alphanumeric words. Words are cut at 64 characters.
pub fn fold_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut n = 0usize;
    for c in text.chars() {
        // combining marks (U+0300..U+036F) glue to the previous letter: drop them
        if ('\u{300}'..='\u{36F}').contains(&c) { continue; }
        if c.is_alphanumeric() {
            if n < 64 { fold_char(c, &mut cur); n += 1; }
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
            n = 0;
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

fn fnv(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // final avalanche so the low and high bits are both usable
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h
}

/// Deterministic embedder with no model: words, word pairs and character n-grams hashed into
/// `dim` buckets with a random sign (the hashing trick), log-damped, then unit length. It finds
/// texts that share words and word stems, tolerates typos and diacritics, and knows no semantics.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dim: usize,
    seed: u64,
    /// Longest word-token sequence considered (cap on work for hostile input).
    pub max_words: usize,
}

impl HashEmbedder {
    /// `dim` is clamped to `8..=65536`.
    pub fn new(dim: usize, seed: u64) -> Self { HashEmbedder { dim: dim.clamp(8, 65536), seed, max_words: 200_000 } }
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> usize { self.dim }
    fn id(&self) -> String { format!("hash:{}:{:x}:{}", self.dim, self.seed, self.max_words) }
    fn embed(&self, text: &str) -> R<Vec<f32>> {
        let mut words = fold_tokens(text);
        words.truncate(self.max_words);
        if words.is_empty() { return Err(RagError::Empty); }
        let mut counts: HashMap<u64, f32> = HashMap::new();
        let mut add = |kind: u8, s: &str, w: f32| {
            let mut key = Vec::with_capacity(s.len() + 1);
            key.push(kind);
            key.extend_from_slice(s.as_bytes());
            *counts.entry(fnv(self.seed, &key)).or_insert(0.0) += w;
        };
        for (i, w) in words.iter().enumerate() {
            add(1, w, 1.0);
            if let Some(nx) = words.get(i + 1) { add(2, &format!("{w} {nx}"), 0.7); }
            let padded: Vec<char> = std::iter::once('<').chain(w.chars()).chain(std::iter::once('>')).collect();
            for n in 3..=5usize {
                if padded.len() < n { break; }
                for win in padded.windows(n) { add(3, &win.iter().collect::<String>(), 0.35); }
            }
        }
        let mut v = vec![0.0f32; self.dim];
        for (h, c) in counts {
            let idx = (h % self.dim as u64) as usize;
            let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
            v[idx] += sign * (1.0 + c).ln();
        }
        l2_normalize(&mut v);
        Ok(v)
    }
}

// ───────────────────────────── chunking ─────────────────────────────

/// How to cut a document.
#[derive(Debug, Clone)]
pub struct ChunkOptions {
    /// Target upper bound for a chunk, in characters (at least 16).
    pub max_chars: usize,
    /// Characters of the previous chunk repeated at the start of the next (whole sentences only).
    pub overlap_chars: usize,
}
impl Default for ChunkOptions {
    fn default() -> Self { ChunkOptions { max_chars: 800, overlap_chars: 120 } }
}

/// One piece of a document: `text == &document[byte_start..byte_end]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub byte_start: usize,
    pub byte_end: usize,
    pub text: String,
}

const ABBREVIATIONS: &[&str] = &[
    "dr", "dna", "dl", "d-na", "d-l", "nr", "str", "bd", "bdul", "ing", "prof", "conf", "lect", "sc", "etc", "ex", "pag", "pg", "fig", "tel",
    "cca", "ibid", "vol", "ed", "art", "alin", "lit", "pct", "mr", "mrs", "ms", "st", "jr", "sr", "vs", "inc", "ltd", "no", "cap", "mat", "jud",
];

fn is_abbreviation(word: &str) -> bool {
    let w: String = word.chars().filter(|c| !c.is_ascii_punctuation() || *c == '-').collect::<String>().to_lowercase();
    ABBREVIATIONS.contains(&w.as_str()) || (w.chars().count() == 1 && w.chars().all(|c| c.is_alphabetic()))
}

/// Byte ranges of the sentence-like units of `text` (trimmed, never empty).
fn units(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let b = text.as_bytes();
    let mut start: Option<usize> = None;
    let mut last_end = 0usize; // end of the last non-space char
    let mut it = text.char_indices().peekable();
    let flush = |start: &mut Option<usize>, end: usize, out: &mut Vec<(usize, usize)>| {
        if let Some(s) = start.take().filter(|&s| end > s) { out.push((s, end)); }
    };
    while let Some((i, c)) = it.next() {
        if c.is_whitespace() {
            // a blank line (two newlines) always ends a unit
            if c == '\n' {
                let mut j = i + 1;
                let mut nl = 1;
                while j < b.len() && (b[j] == b' ' || b[j] == b'\t' || b[j] == b'\r' || b[j] == b'\n') {
                    if b[j] == b'\n' { nl += 1; }
                    j += 1;
                }
                if nl >= 2 { flush(&mut start, last_end, &mut out); }
            }
            continue;
        }
        if start.is_none() { start = Some(i); }
        last_end = i + c.len_utf8();
        if matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？') {
            // swallow closing quotes and brackets right after the terminator
            let mut end = last_end;
            while let Some(&(k, n)) = it.peek() {
                if matches!(n, '.' | '!' | '?' | '…' | '"' | '\'' | ')' | ']' | '”' | '»' | '’') { end = k + n.len_utf8(); it.next(); } else { break; }
            }
            last_end = end;
            let rest = &text[end..];
            let mut chars = rest.chars();
            let ws = chars.next().is_some_and(|x| x.is_whitespace());
            if ws || rest.is_empty() {
                let next = rest.trim_start().chars().next();
                // the word that ends here
                let s0 = start.unwrap_or(i);
                let word = text[s0..end].rsplit(char::is_whitespace).next().unwrap_or("");
                let word = word.trim_end_matches(['.', '!', '?', '…', '"', '\'', ')', ']', '”', '»', '’']);
                let abbr = c == '.' && is_abbreviation(word);
                let decimal = false;
                let boundary = match next {
                    None => true,
                    Some(n) => !abbr && !decimal && !n.is_lowercase(),
                };
                if boundary { flush(&mut start, last_end, &mut out); }
            }
        }
    }
    flush(&mut start, last_end, &mut out);
    out
}

/// Cut `text` into chunks of at most about `opts.max_chars` characters on sentence boundaries,
/// repeating up to `opts.overlap_chars` characters of whole sentences at the start of the next
/// chunk. A sentence longer than the limit is cut at word boundaries. Chunk offsets are byte
/// offsets into `text` on character boundaries, so `text[c.byte_start..c.byte_end] == c.text`.
/// Empty or blank text gives no chunks.
pub fn chunk_text(text: &str, opts: &ChunkOptions) -> Vec<Chunk> {
    let max = opts.max_chars.max(16);
    let overlap = opts.overlap_chars.min(max / 2);
    // split oversized units at word boundaries
    let mut us: Vec<(usize, usize)> = Vec::new();
    for (s, e) in units(text) {
        let mut cur = s;
        while cur < e {
            let mut cut = e;
            let mut last_space: Option<usize> = None;
            let mut over = false;
            for (count, (off, ch)) in text[cur..e].char_indices().enumerate() {
                if count == max { cut = cur + off; over = true; break; }
                if ch.is_whitespace() { last_space = Some(cur + off); }
            }
            if let Some(sp) = last_space.filter(|&sp| over && sp > cur) { cut = sp; }
            let piece_end = text[cur..cut].trim_end().len() + cur;
            if piece_end > cur { us.push((cur, piece_end)); }
            let rest = text[cut..e].trim_start();
            cur = e - rest.len();
            if cut >= e { break; }
        }
    }
    if us.is_empty() { return Vec::new(); }
    // char counts: chars before each unit start, and up to each unit end
    let mut before = Vec::with_capacity(us.len());
    let mut upto = Vec::with_capacity(us.len());
    let (mut pos, mut chars) = (0usize, 0usize);
    for &(s, e) in &us {
        chars += text[pos..s].chars().count();
        before.push(chars);
        chars += text[s..e].chars().count();
        upto.push(chars);
        pos = e;
    }
    let n = us.len();
    let mut out = Vec::new();
    let mut i = 0usize;
    loop {
        let mut j = i + 1;
        while j < n && upto[j] - before[i] <= max { j += 1; }
        let (s, e) = (us[i].0, us[j - 1].1);
        out.push(Chunk { byte_start: s, byte_end: e, text: text[s..e].to_string() });
        if j >= n { break; }
        let mut k = j;
        while k > i + 1 && upto[j - 1] - before[k - 1] <= overlap { k -= 1; }
        i = k.max(i + 1);
    }
    out
}

// ───────────────────────────── HNSW ─────────────────────────────

/// How two vectors are compared. Higher scores are always better.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Cosine similarity (vectors are normalised on insert).
    Cosine,
    /// Raw dot product.
    Dot,
}

/// How the index stores vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    /// 4 bytes per value.
    F32,
    /// 1 byte per value plus one scale per vector (4× smaller, small accuracy loss).
    Int8,
}

/// Index parameters. The defaults give ≥ 0.95 recall@10 on random data at the sizes tested.
#[derive(Debug, Clone, PartialEq)]
pub struct HnswParams {
    /// Neighbours kept per node on the upper layers (layer 0 keeps `2·m`). 2..=256.
    pub m: usize,
    /// Candidate list size while building. 1..=100000.
    pub ef_construction: usize,
    /// Candidate list size while searching (raised to `k` if smaller). 1..=100000.
    pub ef_search: usize,
    pub metric: Metric,
    pub storage: Storage,
    /// Seed of the level generator: the same seed and insert order give the same graph.
    pub seed: u64,
}
impl Default for HnswParams {
    fn default() -> Self { HnswParams { m: 16, ef_construction: 200, ef_search: 64, metric: Metric::Cosine, storage: Storage::F32, seed: 0x5eed_c0de } }
}

const MAX_LEVEL: usize = 15;
const MAX_DIM: usize = 65536;
const MAGIC: &[u8; 8] = b"CORHNSW1";

#[derive(Clone, Copy, PartialEq)]
struct Ord32(f32);
impl Eq for Ord32 {}
impl PartialOrd for Ord32 {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
impl Ord for Ord32 {
    fn cmp(&self, o: &Self) -> Ordering { self.0.total_cmp(&o.0) }
}

#[derive(Clone)]
struct Node {
    id: u64,
    deleted: bool,
    links: Vec<Vec<u32>>,
    /// Next node holding exactly the same stored vector (a chain hanging off a graph member), or
    /// `u32::MAX`. Exact duplicates are kept out of the graph: thousands of identical vectors would
    /// otherwise fill every neighbour list and cut the nodes behind them off from the entry point.
    twin: u32,
}

#[derive(Clone)]
enum Vecs {
    F32(Vec<f32>),
    I8 { q: Vec<i8>, scale: Vec<f32> },
}

/// Hierarchical navigable small-world graph (Malkov & Yashunin). Ids are caller-chosen `u64`s.
#[derive(Clone)]
pub struct Hnsw {
    dim: usize,
    p: HnswParams,
    nodes: Vec<Node>,
    vecs: Vecs,
    by_id: HashMap<u64, u32>,
    entry: Option<u32>,
    max_level: usize,
    rng: u64,
    dead: usize,
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn check_params(dim: usize, p: &HnswParams) -> R<()> {
    if dim == 0 || dim > MAX_DIM { return Err(RagError::Param(format!("dimension {dim} outside 1..={MAX_DIM}"))); }
    if !(2..=256).contains(&p.m) { return Err(RagError::Param(format!("m = {} outside 2..=256", p.m))); }
    if !(1..=100_000).contains(&p.ef_construction) { return Err(RagError::Param("ef_construction outside 1..=100000".into())); }
    if !(1..=100_000).contains(&p.ef_search) { return Err(RagError::Param("ef_search outside 1..=100000".into())); }
    Ok(())
}

impl Hnsw {
    /// An empty index for `dim`-dimensional vectors.
    pub fn new(dim: usize, params: HnswParams) -> R<Hnsw> {
        check_params(dim, &params)?;
        let vecs = match params.storage { Storage::F32 => Vecs::F32(Vec::new()), Storage::Int8 => Vecs::I8 { q: Vec::new(), scale: Vec::new() } };
        let rng = params.seed;
        Ok(Hnsw { dim, p: params, nodes: Vec::new(), vecs, by_id: HashMap::new(), entry: None, max_level: 0, rng, dead: 0 })
    }
    pub fn dim(&self) -> usize { self.dim }
    pub fn params(&self) -> &HnswParams { &self.p }
    /// Live (not deleted) entries.
    pub fn len(&self) -> usize { self.nodes.len() - self.dead }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    /// Deleted entries still occupying space until [`Hnsw::compact`].
    pub fn tombstones(&self) -> usize { self.dead }
    pub fn contains(&self, id: u64) -> bool { self.by_id.contains_key(&id) }
    /// Ids of live entries in insertion order.
    pub fn ids(&self) -> Vec<u64> { self.nodes.iter().filter(|n| !n.deleted).map(|n| n.id).collect() }

    fn max_conn(&self, level: usize) -> usize { if level == 0 { self.p.m * 2 } else { self.p.m } }

    fn random_level(&mut self) -> usize {
        let u = ((splitmix(&mut self.rng) >> 11) as f64 + 1.0) / (1u64 << 53) as f64; // (0, 1]
        let ml = 1.0 / (self.p.m as f64).ln();
        ((-u.ln() * ml).floor() as usize).min(MAX_LEVEL)
    }

    /// The stored vector of node `i` as `f32` (dequantised for int8).
    pub fn vector_of(&self, id: u64) -> Option<Vec<f32>> {
        let i = *self.by_id.get(&id)? as usize;
        Some(self.decode(i))
    }
    fn decode(&self, i: usize) -> Vec<f32> {
        let d = self.dim;
        match &self.vecs {
            Vecs::F32(v) => v[i * d..(i + 1) * d].to_vec(),
            Vecs::I8 { q, scale } => q[i * d..(i + 1) * d].iter().map(|&x| x as f32 * scale[i] / 127.0).collect(),
        }
    }

    fn score_qn(&self, q: &[f32], i: usize) -> f32 {
        let d = self.dim;
        match &self.vecs {
            Vecs::F32(v) => dot(q, &v[i * d..(i + 1) * d]),
            Vecs::I8 { q: qs, scale } => {
                let s: f32 = q.iter().zip(&qs[i * d..(i + 1) * d]).map(|(a, &b)| a * b as f32).sum();
                s * scale[i] / 127.0
            }
        }
    }
    fn score_nn(&self, a: usize, b: usize) -> f32 {
        let d = self.dim;
        match &self.vecs {
            Vecs::F32(v) => dot(&v[a * d..(a + 1) * d], &v[b * d..(b + 1) * d]),
            Vecs::I8 { q, scale } => {
                let s: i32 = q[a * d..(a + 1) * d].iter().zip(&q[b * d..(b + 1) * d]).map(|(&x, &y)| x as i32 * y as i32).sum();
                s as f32 * scale[a] * scale[b] / (127.0 * 127.0)
            }
        }
    }

    fn prepare(&self, v: &[f32]) -> R<Vec<f32>> {
        if v.len() != self.dim { return Err(RagError::Dim { expected: self.dim, got: v.len() }); }
        if v.iter().any(|x| !x.is_finite()) { return Err(RagError::NotFinite); }
        let mut p = v.to_vec();
        if self.p.metric == Metric::Cosine { l2_normalize(&mut p); }
        Ok(p)
    }

    fn push_vector(&mut self, v: &[f32]) {
        match &mut self.vecs {
            Vecs::F32(all) => all.extend_from_slice(v),
            Vecs::I8 { q, scale } => {
                let m = v.iter().fold(0.0f32, |a, x| a.max(x.abs()));
                let s = if m > 0.0 { m } else { 1.0 };
                scale.push(s);
                q.extend(v.iter().map(|x| (x / s * 127.0).round().clamp(-127.0, 127.0) as i8));
            }
        }
    }

    /// Do nodes `a` and `b` hold exactly the same stored vector?
    fn same_stored(&self, a: usize, b: usize) -> bool {
        let d = self.dim;
        match &self.vecs {
            Vecs::F32(v) => v[a * d..(a + 1) * d] == v[b * d..(b + 1) * d],
            Vecs::I8 { q, scale } => scale[a] == scale[b] && q[a * d..(a + 1) * d] == q[b * d..(b + 1) * d],
        }
    }

    /// Best-first search on one layer; returns up to `ef` (score, node) pairs, best first.
    fn search_layer(&self, q: &[f32], eps: &[u32], ef: usize, level: usize) -> Vec<(f32, u32)> {
        let mut seen = vec![0u64; self.nodes.len() / 64 + 1];
        let mut cand: BinaryHeap<(Ord32, u32)> = BinaryHeap::new();
        let mut res: BinaryHeap<Reverse<(Ord32, u32)>> = BinaryHeap::new();
        for &e in eps {
            if seen[e as usize / 64] >> (e % 64) & 1 == 1 { continue; }
            seen[e as usize / 64] |= 1 << (e % 64);
            let s = self.score_qn(q, e as usize);
            cand.push((Ord32(s), e));
            res.push(Reverse((Ord32(s), e)));
            if res.len() > ef { res.pop(); }
        }
        while let Some((Ord32(s), c)) = cand.pop() {
            if res.len() >= ef && s < res.peek().map_or(f32::MIN, |r| (r.0).0.0) { break; }
            if level == 0 {
                let mut t = self.nodes[c as usize].twin;
                while t != u32::MAX && seen[t as usize / 64] >> (t % 64) & 1 == 0 {
                    seen[t as usize / 64] |= 1 << (t % 64);
                    let st = self.score_qn(q, t as usize);
                    if res.len() < ef || st > res.peek().map_or(f32::MIN, |r| (r.0).0.0) {
                        res.push(Reverse((Ord32(st), t)));
                        if res.len() > ef { res.pop(); }
                    }
                    t = self.nodes[t as usize].twin;
                }
            }
            let links = match self.nodes[c as usize].links.get(level) { Some(l) => l, None => continue };
            for &nb in links {
                if seen[nb as usize / 64] >> (nb % 64) & 1 == 1 { continue; }
                seen[nb as usize / 64] |= 1 << (nb % 64);
                let sn = self.score_qn(q, nb as usize);
                if res.len() < ef || sn > res.peek().map_or(f32::MIN, |r| (r.0).0.0) {
                    cand.push((Ord32(sn), nb));
                    res.push(Reverse((Ord32(sn), nb)));
                    if res.len() > ef { res.pop(); }
                }
            }
        }
        let mut v: Vec<(f32, u32)> = res.into_iter().map(|Reverse((s, n))| (s.0, n)).collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        v
    }

    fn greedy(&self, q: &[f32], mut ep: u32, level: usize) -> u32 {
        let mut best = self.score_qn(q, ep as usize);
        loop {
            let mut moved = false;
            if let Some(links) = self.nodes[ep as usize].links.get(level) {
                for &nb in links {
                    let s = self.score_qn(q, nb as usize);
                    if s > best { best = s; ep = nb; moved = true; }
                }
            }
            if !moved { return ep; }
        }
    }

    /// Neighbour-selection heuristic: keep a candidate only if it is closer to the query than to
    /// any already kept neighbour (spreads links in all directions), then top up with the rest.
    fn select(&self, cands: &[(f32, u32)], m: usize) -> Vec<u32> {
        let mut kept: Vec<(f32, u32)> = Vec::with_capacity(m);
        let mut pruned: Vec<u32> = Vec::new();
        for &(s, c) in cands {
            if kept.len() >= m { break; }
            if kept.iter().all(|&(_, r)| s > self.score_nn(c as usize, r as usize)) { kept.push((s, c)); } else { pruned.push(c); }
        }
        let mut out: Vec<u32> = kept.into_iter().map(|x| x.1).collect();
        for c in pruned { if out.len() >= m { break; } out.push(c); }
        out
    }

    /// Add `v` under `id`. If `id` exists the old entry is deleted and replaced.
    pub fn insert(&mut self, id: u64, v: &[f32]) -> R<()> {
        let pv = self.prepare(v)?;
        if self.by_id.contains_key(&id) { self.remove(id)?; }
        let idx = self.nodes.len() as u32;
        if idx == u32::MAX { return Err(RagError::Param("index is full".into())); }
        let level = self.random_level();
        self.push_vector(&pv);
        self.nodes.push(Node { id, deleted: false, links: vec![Vec::new(); level + 1], twin: u32::MAX });
        self.by_id.insert(id, idx);
        let Some(entry) = self.entry else {
            self.entry = Some(idx);
            self.max_level = level;
            return Ok(());
        };
        // an exact duplicate of a graph member hangs off that member instead of entering the graph
        let mut top = entry;
        for l in (1..=self.max_level).rev() { top = self.greedy(&pv, top, l); }
        let dup = self.search_layer(&pv, &[top], 8, 0).first().map(|x| x.1).filter(|&near| self.same_stored(idx as usize, near as usize));
        if let Some(near) = dup {
            self.nodes[idx as usize].links = vec![Vec::new()];
            self.nodes[idx as usize].twin = self.nodes[near as usize].twin;
            self.nodes[near as usize].twin = idx;
            return Ok(());
        }
        let mut ep = entry;
        for l in (level + 1..=self.max_level).rev() { ep = self.greedy(&pv, ep, l); }
        let mut eps = vec![ep];
        for l in (0..=level.min(self.max_level)).rev() {
            let cands = self.search_layer(&pv, &eps, self.p.ef_construction, l);
            let sel = self.select(&cands, self.p.m);
            self.nodes[idx as usize].links[l] = sel.clone();
            let cap = self.max_conn(l);
            for &nb in &sel {
                let nbi = nb as usize;
                if self.nodes[nbi].links.len() <= l { continue; }
                self.nodes[nbi].links[l].push(idx);
                if self.nodes[nbi].links[l].len() > cap {
                    let mut scored: Vec<(f32, u32)> = self.nodes[nbi].links[l].iter().map(|&o| (self.score_nn(nbi, o as usize), o)).collect();
                    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                    let keep = self.select(&scored, cap);
                    self.nodes[nbi].links[l] = keep;
                }
            }
            eps = cands.iter().map(|c| c.1).collect();
            if eps.is_empty() { eps = vec![ep]; }
        }
        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(idx);
        }
        Ok(())
    }

    /// Mark `id` deleted. It stops appearing in results; the graph keeps routing through it until
    /// [`Hnsw::compact`].
    pub fn remove(&mut self, id: u64) -> R<()> {
        let idx = self.by_id.remove(&id).ok_or(RagError::Unknown(id))?;
        self.nodes[idx as usize].deleted = true;
        self.dead += 1;
        Ok(())
    }

    /// Rebuild the graph without the tombstones (same insertion order, same seed, so the result
    /// is deterministic). Returns how many entries were dropped.
    pub fn compact(&mut self) -> R<usize> {
        let dropped = self.dead;
        if dropped == 0 { return Ok(0); }
        let mut fresh = Hnsw::new(self.dim, self.p.clone())?;
        for (i, n) in self.nodes.iter().enumerate() {
            if !n.deleted { fresh.insert(n.id, &self.decode(i))?; }
        }
        *self = fresh;
        Ok(dropped)
    }

    /// The `k` best live entries for `query`, best first, as `(id, score)`. `ef` overrides
    /// `ef_search`. Ties break by id so results are deterministic.
    pub fn search(&self, query: &[f32], k: usize, ef: Option<usize>) -> R<Vec<(u64, f32)>> {
        let q = self.prepare(query)?;
        let Some(entry) = self.entry else { return Ok(Vec::new()) };
        if k == 0 { return Ok(Vec::new()); }
        let mut ep = entry;
        for l in (1..=self.max_level).rev() { ep = self.greedy(&q, ep, l); }
        // tombstones take up result slots: widen the search to compensate
        let k = k.min(self.nodes.len());
        let base = ef.unwrap_or(self.p.ef_search).max(k);
        let efe = base.saturating_add(self.dead.min(base.saturating_mul(4))).min(self.nodes.len().max(1));
        let found = self.search_layer(&q, &[ep], efe, 0);
        let mut out: Vec<(u64, f32)> = found.into_iter().filter(|&(_, n)| !self.nodes[n as usize].deleted).map(|(s, n)| (self.nodes[n as usize].id, s)).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        Ok(out)
    }

    /// Exact search by scanning every live entry (the reference for recall measurements).
    pub fn brute_force(&self, query: &[f32], k: usize) -> R<Vec<(u64, f32)>> {
        let q = self.prepare(query)?;
        let mut all: Vec<(u64, f32)> = self.nodes.iter().enumerate().filter(|(_, n)| !n.deleted).map(|(i, n)| (n.id, self.score_qn(&q, i))).collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.truncate(k);
        Ok(all)
    }

    // ── persistence ──

    /// Serialise (little-endian, SHA-256 of everything before it as the last 32 bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(MAGIC);
        let u32le = |o: &mut Vec<u8>, v: u32| o.extend_from_slice(&v.to_le_bytes());
        u32le(&mut o, 1);
        u32le(&mut o, self.dim as u32);
        u32le(&mut o, self.p.m as u32);
        u32le(&mut o, self.p.ef_construction as u32);
        u32le(&mut o, self.p.ef_search as u32);
        o.push(match self.p.metric { Metric::Cosine => 0, Metric::Dot => 1 });
        o.push(match self.p.storage { Storage::F32 => 0, Storage::Int8 => 1 });
        o.extend_from_slice(&self.p.seed.to_le_bytes());
        o.extend_from_slice(&self.rng.to_le_bytes());
        u32le(&mut o, self.nodes.len() as u32);
        u32le(&mut o, self.entry.unwrap_or(u32::MAX));
        u32le(&mut o, self.max_level as u32);
        for (i, n) in self.nodes.iter().enumerate() {
            o.extend_from_slice(&n.id.to_le_bytes());
            o.push(n.deleted as u8);
            o.push((n.links.len() - 1) as u8);
            u32le(&mut o, n.twin);
            match &self.vecs {
                Vecs::F32(v) => for x in &v[i * self.dim..(i + 1) * self.dim] { o.extend_from_slice(&x.to_le_bytes()); },
                Vecs::I8 { q, scale } => {
                    o.extend_from_slice(&scale[i].to_le_bytes());
                    o.extend(q[i * self.dim..(i + 1) * self.dim].iter().map(|&x| x as u8));
                }
            }
            for l in &n.links {
                u32le(&mut o, l.len() as u32);
                for &x in l { u32le(&mut o, x); }
            }
        }
        let sum = sha256(&o);
        o.extend_from_slice(&sum);
        o
    }

    /// Parse bytes from [`Hnsw::to_bytes`]. Every field is validated; any inconsistency is an
    /// error, never a panic or an oversized allocation (the checksum only detects accidents, so a
    /// forged file with a correct checksum is checked just as hard).
    pub fn from_bytes(b: &[u8]) -> R<Hnsw> {
        let bad = |m: &str| RagError::Format(m.to_string());
        if b.len() < MAGIC.len() + 32 { return Err(bad("file too short")); }
        let (body, sum) = b.split_at(b.len() - 32);
        if sha256(body) != sum { return Err(bad("checksum mismatch")); }
        let mut r = Reader { b: body, p: 0 };
        if r.take(8)? != MAGIC { return Err(bad("not an index file")); }
        if r.u32()? != 1 { return Err(bad("unsupported version")); }
        let dim = r.u32()? as usize;
        let m = r.u32()? as usize;
        let efc = r.u32()? as usize;
        let efs = r.u32()? as usize;
        let metric = match r.u8()? { 0 => Metric::Cosine, 1 => Metric::Dot, _ => return Err(bad("unknown metric")) };
        let storage = match r.u8()? { 0 => Storage::F32, 1 => Storage::Int8, _ => return Err(bad("unknown storage")) };
        let seed = r.u64()?;
        let rng = r.u64()?;
        let n = r.u32()? as usize;
        let entry = r.u32()?;
        let max_level = r.u32()? as usize;
        let params = HnswParams { m, ef_construction: efc, ef_search: efs, metric, storage, seed };
        check_params(dim, &params).map_err(|e| RagError::Format(e.to_string()))?;
        if max_level > MAX_LEVEL { return Err(bad("level out of range")); }
        let per_vec = match storage { Storage::F32 => dim * 4, Storage::Int8 => dim + 4 };
        let min_node = 8 + 1 + 1 + 4 + per_vec + 4;
        if n.checked_mul(min_node).is_none_or(|t| t > r.left()) { return Err(bad("node count larger than the file")); }
        let mut h = Hnsw::new(dim, params)?;
        h.rng = rng;
        h.nodes.reserve(n);
        let mut levels = Vec::with_capacity(n);
        for i in 0..n {
            let id = r.u64()?;
            let deleted = match r.u8()? { 0 => false, 1 => true, _ => return Err(bad("bad deleted flag")) };
            let level = r.u8()? as usize;
            if level > MAX_LEVEL { return Err(bad("node level out of range")); }
            let twin = r.u32()?;
            if twin != u32::MAX && (twin as usize >= n || twin as usize == i) { return Err(bad("twin out of range")); }
            match &mut h.vecs {
                Vecs::F32(v) => {
                    for _ in 0..dim {
                        let x = f32::from_le_bytes(r.take(4)?.try_into().map_err(|_| bad("short read"))?);
                        if !x.is_finite() { return Err(bad("non-finite value")); }
                        v.push(x);
                    }
                }
                Vecs::I8 { q, scale } => {
                    let s = f32::from_le_bytes(r.take(4)?.try_into().map_err(|_| bad("short read"))?);
                    if !s.is_finite() || s < 0.0 { return Err(bad("bad scale")); }
                    scale.push(s);
                    q.extend(r.take(dim)?.iter().map(|&x| x as i8));
                }
            }
            let mut links = Vec::with_capacity(level + 1);
            for l in 0..=level {
                let c = r.u32()? as usize;
                if c > h.max_conn(l) || c.checked_mul(4).is_none_or(|t| t > r.left()) { return Err(bad("too many links")); }
                let mut v = Vec::with_capacity(c);
                for _ in 0..c {
                    let x = r.u32()?;
                    if x as usize >= n || x as usize == i { return Err(bad("link out of range")); }
                    v.push(x);
                }
                links.push(v);
            }
            if !deleted && h.by_id.insert(id, i as u32).is_some() { return Err(bad("duplicate id")); }
            if deleted { h.dead += 1; }
            levels.push(level);
            h.nodes.push(Node { id, deleted, links, twin });
        }
        if r.left() != 0 { return Err(bad("trailing bytes")); }
        for node in &h.nodes {
            for (l, lk) in node.links.iter().enumerate() {
                if lk.iter().any(|&x| levels[x as usize] < l) { return Err(bad("link to a node that is not on that layer")); }
            }
        }
        // twin chains: each node has at most one predecessor and there are no cycles
        let mut pred = vec![false; n];
        for node in &h.nodes {
            if node.twin != u32::MAX {
                if pred[node.twin as usize] { return Err(bad("node is the twin of two nodes")); }
                pred[node.twin as usize] = true;
            }
        }
        let mut reached = 0usize;
        for (i, _) in pred.iter().enumerate().filter(|(_, p)| !**p) {
            let mut t = i;
            loop {
                reached += 1;
                let nx = h.nodes[t].twin;
                if nx == u32::MAX { break; }
                t = nx as usize;
            }
        }
        if reached != n { return Err(bad("twin chain forms a cycle")); }
        if n == 0 {
            if entry != u32::MAX { return Err(bad("entry point in an empty index")); }
        } else {
            if entry as usize >= n || levels[entry as usize] != max_level { return Err(bad("bad entry point")); }
            if levels.iter().any(|&l| l > max_level) { return Err(bad("a node is above the top layer")); }
            h.entry = Some(entry);
            h.max_level = max_level;
        }
        Ok(h)
    }

    /// Write to `path` (to a temporary file next to it first, then renamed).
    pub fn save(&self, path: &Path) -> R<()> {
        let tmp = path.with_extension("tmp-hnsw");
        std::fs::write(&tmp, self.to_bytes()).map_err(|e| RagError::Io(e.to_string()))?;
        std::fs::rename(&tmp, path).map_err(|e| { let _ = std::fs::remove_file(&tmp); RagError::Io(e.to_string()) })
    }
    /// Read a file written by [`Hnsw::save`].
    pub fn load(path: &Path) -> R<Hnsw> {
        let b = std::fs::read(path).map_err(|e| RagError::Io(e.to_string()))?;
        Hnsw::from_bytes(&b)
    }
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn left(&self) -> usize { self.b.len() - self.p }
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        if n > self.left() { return Err(RagError::Format("unexpected end of file".into())); }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> R<u8> { Ok(self.take(1)?[0]) }
    fn u32(&mut self) -> R<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| RagError::Format("short read".into()))?)) }
    fn u64(&mut self) -> R<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| RagError::Format("short read".into()))?)) }
}

// ───────────────────────────── BM25 ─────────────────────────────

/// Function words (already folded) that carry no topic: Romanian and English.
const STOP_WORDS: &[&str] = &[
    "a", "al", "ale", "am", "an", "ar", "are", "as", "au", "ca", "care", "ce", "cand", "cat", "cate", "ci", "cu", "cum", "da", "dar", "de", "deci",
    "din", "dupa", "ea", "ei", "el", "era", "este", "fi", "fie", "fost", "iar", "in", "intre", "la", "le", "lor", "lui", "ma", "mai", "ne", "nu", "o",
    "or", "pe", "pentru", "prin", "sa", "sau", "se", "si", "spre", "sunt", "tot", "un", "una", "unde", "unei", "unui", "va", "voi", "the", "and", "of",
    "to", "on", "for", "with", "is", "are", "was", "were", "be", "by", "at", "it", "this", "that", "from", "how", "what", "an",
];

struct DocEntry {
    id: u64,
    len: u32,
    terms: Vec<String>,
}

/// Okapi BM25 keyword index over diacritic-folded words (`k1 = 1.2`, `b = 0.75`).
#[derive(Default)]
pub struct Bm25Index {
    /// Index only the first `n` characters of each word (a crude stemmer: "sarmale" and
    /// "sarmalele" meet at `n = 6`). Words not longer than `n` are unchanged. `None` keeps whole words.
    prefix: Option<usize>,
    /// Ignore very common Romanian and English function words.
    stop: bool,
    docs: Vec<Option<DocEntry>>,
    by_id: HashMap<u64, u32>,
    postings: HashMap<String, Vec<(u32, u32)>>,
    total_len: u64,
    live: usize,
}

impl Bm25Index {
    pub fn new() -> Self { Self::default() }
    /// Index with word-prefix stemming and/or stop-word removal (set before adding documents).
    pub fn with_options(prefix: Option<usize>, drop_stop_words: bool) -> Self {
        Bm25Index { prefix: prefix.map(|n| n.max(2)), stop: drop_stop_words, ..Self::default() }
    }
    fn terms(&self, text: &str) -> Vec<String> {
        let all = fold_tokens(text);
        let kept: Vec<String> = if self.stop { all.iter().filter(|w| !STOP_WORDS.contains(&w.as_str())).cloned().collect() } else { all.clone() };
        match self.prefix {
            Some(n) => kept.into_iter().map(|w| w.chars().take(n).collect()).collect(),
            None => kept,
        }
    }
    pub fn len(&self) -> usize { self.live }
    pub fn is_empty(&self) -> bool { self.live == 0 }

    /// Index `text` under `id` (replacing an existing entry with that id).
    pub fn add(&mut self, id: u64, text: &str) {
        if self.by_id.contains_key(&id) { self.remove(id); }
        let toks = self.terms(text);
        let mut tf: HashMap<&str, u32> = HashMap::new();
        for t in &toks { *tf.entry(t.as_str()).or_insert(0) += 1; }
        let idx = self.docs.len() as u32;
        let mut terms = Vec::with_capacity(tf.len());
        for (t, c) in tf {
            self.postings.entry(t.to_string()).or_default().push((idx, c));
            terms.push(t.to_string());
        }
        self.total_len += toks.len() as u64;
        self.docs.push(Some(DocEntry { id, len: toks.len() as u32, terms }));
        self.by_id.insert(id, idx);
        self.live += 1;
    }

    /// Drop `id`. Returns whether it was there.
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(idx) = self.by_id.remove(&id) else { return false };
        if let Some(d) = self.docs[idx as usize].take() {
            self.total_len -= d.len as u64;
            for t in &d.terms {
                if let Some(p) = self.postings.get_mut(t) {
                    p.retain(|&(i, _)| i != idx);
                    if p.is_empty() { self.postings.remove(t); }
                }
            }
            self.live -= 1;
        }
        true
    }

    /// The `k` best matches as `(id, score)`, best first (ties by id). Words that appear in no
    /// document are ignored; a query with no known word gives no results.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
        if self.live == 0 || k == 0 { return Vec::new(); }
        let mut q = self.terms(query);
        q.sort();
        q.dedup();
        let (k1, b) = (1.2f32, 0.75f32);
        let n = self.live as f32;
        let avg = (self.total_len as f32 / n).max(1.0);
        let mut acc: HashMap<u32, f32> = HashMap::new();
        for t in &q {
            let Some(p) = self.postings.get(t) else { continue };
            let df = p.len() as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for &(d, tf) in p {
                let Some(doc) = &self.docs[d as usize] else { continue };
                let tf = tf as f32;
                let s = idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * doc.len as f32 / avg));
                *acc.entry(d).or_insert(0.0) += s;
            }
        }
        let mut out: Vec<(u64, f32)> = acc.into_iter().filter_map(|(d, s)| self.docs[d as usize].as_ref().map(|e| (e.id, s))).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }
}

const BM_MAGIC: &[u8; 8] = b"CORBM25\x01";
const MAX_TERM_BYTES: usize = 1024;

impl Bm25Index {
    /// Ids of the indexed entries, in insertion order.
    pub fn ids(&self) -> Vec<u64> { self.docs.iter().flatten().map(|d| d.id).collect() }

    /// Serialise (little-endian, SHA-256 of everything before it as the last 32 bytes). Only live
    /// entries are written; terms are sorted, so equal indexes give equal bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut per: Vec<Vec<(&str, u32)>> = vec![Vec::new(); self.docs.len()];
        for (t, list) in &self.postings { for &(d, tf) in list { per[d as usize].push((t.as_str(), tf)); } }
        let mut o = Vec::new();
        o.extend_from_slice(BM_MAGIC);
        o.extend_from_slice(&self.prefix.map_or(u32::MAX, |n| n as u32).to_le_bytes());
        o.push(self.stop as u8);
        o.extend_from_slice(&(self.live as u32).to_le_bytes());
        for (i, d) in self.docs.iter().enumerate() {
            let Some(d) = d else { continue };
            o.extend_from_slice(&d.id.to_le_bytes());
            o.extend_from_slice(&d.len.to_le_bytes());
            per[i].sort();
            o.extend_from_slice(&(per[i].len() as u32).to_le_bytes());
            for (t, tf) in &per[i] {
                o.extend_from_slice(&(t.len() as u32).to_le_bytes());
                o.extend_from_slice(t.as_bytes());
                o.extend_from_slice(&tf.to_le_bytes());
            }
        }
        let sum = sha256(&o);
        o.extend_from_slice(&sum);
        o
    }

    /// Parse bytes from [`Bm25Index::to_bytes`]. Every count is checked against the bytes that are
    /// left and each entry must be self-consistent (its term frequencies add up to its length), so
    /// a damaged or forged file is an error, never a panic or a huge allocation.
    pub fn from_bytes(b: &[u8]) -> R<Bm25Index> {
        let bad = |m: &str| RagError::Format(format!("keyword index: {m}"));
        if b.len() < BM_MAGIC.len() + 32 { return Err(bad("file too short")); }
        let (body, sum) = b.split_at(b.len() - 32);
        if sha256(body) != sum { return Err(bad("checksum mismatch")); }
        let mut r = Reader { b: body, p: 0 };
        if r.take(8)? != BM_MAGIC { return Err(bad("not a keyword index")); }
        let prefix = match r.u32()? { u32::MAX => None, n @ 2..=64 => Some(n as usize), _ => return Err(bad("prefix length out of range")) };
        let stop = match r.u8()? { 0 => false, 1 => true, _ => return Err(bad("bad stop-word flag")) };
        let n = r.u32()? as usize;
        if n.checked_mul(16).is_none_or(|t| t > r.left()) { return Err(bad("entry count larger than the file")); }
        let mut ix = Bm25Index { prefix, stop, ..Bm25Index::default() };
        for _ in 0..n {
            let id = r.u64()?;
            let len = r.u32()?;
            let nt = r.u32()? as usize;
            if nt.checked_mul(9).is_none_or(|t| t > r.left()) { return Err(bad("term count larger than the file")); }
            if ix.by_id.contains_key(&id) { return Err(bad("duplicate id")); }
            let idx = ix.docs.len() as u32;
            let (mut terms, mut total) = (Vec::with_capacity(nt), 0u64);
            for _ in 0..nt {
                let tl = r.u32()? as usize;
                if tl == 0 || tl > MAX_TERM_BYTES { return Err(bad("term length out of range")); }
                let t = std::str::from_utf8(r.take(tl)?).map_err(|_| bad("term is not UTF-8"))?.to_string();
                let tf = r.u32()?;
                if tf == 0 { return Err(bad("zero term frequency")); }
                total += tf as u64;
                terms.push((t, tf));
            }
            if total != len as u64 { return Err(bad("term frequencies do not add up to the entry length")); }
            terms.sort();
            if terms.windows(2).any(|w| w[0].0 == w[1].0) { return Err(bad("a term appears twice in one entry")); }
            for (t, tf) in &terms { ix.postings.entry(t.clone()).or_default().push((idx, *tf)); }
            ix.docs.push(Some(DocEntry { id, len, terms: terms.into_iter().map(|x| x.0).collect() }));
            ix.by_id.insert(id, idx);
            ix.total_len += len as u64;
            ix.live += 1;
        }
        if r.left() != 0 { return Err(bad("trailing bytes")); }
        Ok(ix)
    }
}

// ───────────────────────────── fusion ─────────────────────────────

/// Reciprocal rank fusion: each list contributes `weight / (k + rank)` per id (rank from 1).
/// Returns `(id, fused score)`, best first, ties by id. `k` is usually 60.
pub fn rrf(lists: &[(&[u64], f32)], k: f32) -> Vec<(u64, f32)> {
    let mut acc: HashMap<u64, f32> = HashMap::new();
    for (list, w) in lists {
        for (r, id) in list.iter().enumerate() { *acc.entry(*id).or_insert(0.0) += w / (k + r as f32 + 1.0); }
    }
    let mut out: Vec<(u64, f32)> = acc.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

// ───────────────────────────── knowledge base ─────────────────────────────

/// How to search a [`KnowledgeBase`].
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Passages to return.
    pub k: usize,
    /// How many candidates each ranker contributes before fusion (at least `k`).
    pub candidates: usize,
    pub vector_weight: f32,
    pub keyword_weight: f32,
    pub rrf_k: f32,
}
impl Default for SearchOptions {
    fn default() -> Self { SearchOptions { k: 5, candidates: 50, vector_weight: 1.0, keyword_weight: 1.0, rrf_k: 60.0 } }
}

#[derive(Debug, Clone)]
struct ChunkRec {
    doc: u64,
    byte_start: usize,
    byte_end: usize,
    text: String,
}

/// One retrieved passage.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub chunk_id: u64,
    pub doc_id: u64,
    pub doc_name: String,
    /// Fused (RRF) score.
    pub score: f32,
    /// Similarity from the vector index, if this chunk was among its candidates.
    pub vector_score: Option<f32>,
    /// BM25 score, if this chunk was among its candidates.
    pub keyword_score: Option<f32>,
    pub byte_start: usize,
    pub byte_end: usize,
    pub text: String,
}

/// Documents → chunks → vector + keyword indexes, searched together.
pub struct KnowledgeBase<'a> {
    embedder: Box<dyn Embedder + 'a>,
    vectors: Hnsw,
    keywords: Bm25Index,
    chunks: HashMap<u64, ChunkRec>,
    docs: HashMap<u64, DocRec>,
    next_chunk: u64,
    next_doc: u64,
}

#[derive(Debug, Clone)]
struct DocRec {
    name: String,
    chunks: Vec<u64>,
    /// SHA-256 of the text the document was added with (to skip unchanged files on re-ingest).
    hash: [u8; 32],
}

/// What [`KnowledgeBase::upsert_document`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Upsert {
    /// A new name.
    Added(u64),
    /// The name existed with different text: the new version is `new`, the old one is gone.
    Replaced { old: u64, new: u64 },
    /// The name existed with identical text; nothing changed.
    Unchanged(u64),
}

/// A document as listed by [`KnowledgeBase::documents`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocInfo {
    pub id: u64,
    pub name: String,
    pub chunks: usize,
    pub hash: [u8; 32],
}

impl<'a> KnowledgeBase<'a> {
    /// Empty knowledge base using `embedder` and the given index parameters.
    pub fn new(embedder: Box<dyn Embedder + 'a>, params: HnswParams) -> R<KnowledgeBase<'a>> {
        let vectors = Hnsw::new(embedder.dim(), params)?;
        Ok(KnowledgeBase { embedder, vectors, keywords: Bm25Index::with_options(Some(5), true), chunks: HashMap::new(), docs: HashMap::new(), next_chunk: 1, next_doc: 1 })
    }
    pub fn chunk_count(&self) -> usize { self.chunks.len() }
    pub fn doc_count(&self) -> usize { self.docs.len() }
    pub fn vector_index(&self) -> &Hnsw { &self.vectors }

    /// Chunk, embed and index `text`. Returns the document id. Chunks that cannot be embedded
    /// (no words) are skipped.
    pub fn add_document(&mut self, name: &str, text: &str, opts: &ChunkOptions) -> R<u64> {
        let doc = self.next_doc;
        self.next_doc += 1;
        let mut ids = Vec::new();
        for c in chunk_text(text, opts) {
            let v = match self.embedder.embed(&c.text) {
                Ok(v) => v,
                Err(RagError::Empty) => continue,
                Err(e) => {
                    for id in &ids { let _ = self.vectors.remove(*id); self.keywords.remove(*id); self.chunks.remove(id); }
                    return Err(e);
                }
            };
            let id = self.next_chunk;
            self.next_chunk += 1;
            self.vectors.insert(id, &v)?;
            self.keywords.add(id, &c.text);
            self.chunks.insert(id, ChunkRec { doc, byte_start: c.byte_start, byte_end: c.byte_end, text: c.text });
            ids.push(id);
        }
        self.docs.insert(doc, DocRec { name: name.to_string(), chunks: ids, hash: sha256(text.as_bytes()) });
        Ok(doc)
    }

    /// Add `text` under `name`, or bring an existing document of that name up to date: unchanged
    /// text costs nothing, changed text is indexed first and the old version dropped after, so a
    /// failure leaves the old version in place.
    pub fn upsert_document(&mut self, name: &str, text: &str, opts: &ChunkOptions) -> R<Upsert> {
        let old = self.find_document(name);
        if let Some(o) = old
            && self.docs.get(&o).is_some_and(|d| d.hash == sha256(text.as_bytes()))
        {
            return Ok(Upsert::Unchanged(o));
        }
        let new = self.add_document(name, text, opts)?;
        match old {
            Some(o) => { self.remove_document(o)?; Ok(Upsert::Replaced { old: o, new }) }
            None => Ok(Upsert::Added(new)),
        }
    }

    /// The newest document with this name.
    pub fn find_document(&self, name: &str) -> Option<u64> {
        self.docs.iter().filter(|(_, d)| d.name == name).map(|(&id, _)| id).max()
    }

    /// Every document, by id.
    pub fn documents(&self) -> Vec<DocInfo> {
        let mut v: Vec<DocInfo> = self.docs.iter().map(|(&id, d)| DocInfo { id, name: d.name.clone(), chunks: d.chunks.len(), hash: d.hash }).collect();
        v.sort_by_key(|d| d.id);
        v
    }

    /// Rebuild the vector graph without the entries of removed documents. Returns how many were dropped.
    pub fn compact(&mut self) -> R<usize> { self.vectors.compact() }

    /// Forget a document and all its chunks.
    pub fn remove_document(&mut self, doc: u64) -> R<()> {
        let rec = self.docs.remove(&doc).ok_or(RagError::Unknown(doc))?;
        for id in rec.chunks {
            let _ = self.vectors.remove(id);
            self.keywords.remove(id);
            self.chunks.remove(&id);
        }
        Ok(())
    }

    /// Hybrid search: vector neighbours and BM25 matches, fused by reciprocal rank.
    pub fn search(&self, query: &str, o: &SearchOptions) -> R<Vec<Hit>> {
        let cand = o.candidates.max(o.k).max(1);
        let vec_hits = match self.embedder.embed(query) {
            Ok(q) => self.vectors.search(&q, cand, None)?,
            Err(RagError::Empty) => Vec::new(),
            Err(e) => return Err(e),
        };
        let kw_hits = self.keywords.search(query, cand);
        let vi: Vec<u64> = vec_hits.iter().map(|x| x.0).collect();
        let ki: Vec<u64> = kw_hits.iter().map(|x| x.0).collect();
        let fused = rrf(&[(&vi, o.vector_weight), (&ki, o.keyword_weight)], o.rrf_k);
        let vs: HashMap<u64, f32> = vec_hits.into_iter().collect();
        let ks: HashMap<u64, f32> = kw_hits.into_iter().collect();
        let mut out = Vec::new();
        for (id, score) in fused.into_iter().take(o.k) {
            let Some(c) = self.chunks.get(&id) else { continue };
            let name = self.docs.get(&c.doc).map(|d| d.name.clone()).unwrap_or_default();
            out.push(Hit {
                chunk_id: id, doc_id: c.doc, doc_name: name, score,
                vector_score: vs.get(&id).copied(), keyword_score: ks.get(&id).copied(),
                byte_start: c.byte_start, byte_end: c.byte_end, text: c.text.clone(),
            });
        }
        Ok(out)
    }

    /// Search and package the passages as numbered sources for the quote lock and the prompt.
    pub fn retrieve_cited(&self, query: &str, o: &SearchOptions) -> R<CitedContext> {
        Ok(CitedContext::from_hits(self.search(query, o)?))
    }
}

// ───────────────────────────── knowledge-base files ─────────────────────────────

const KB_MAGIC: &[u8; 8] = b"CORKB001";
const KB_VERSION: u32 = 1;
const MAX_TEXT: usize = 64 << 20;

fn put_str(o: &mut Vec<u8>, s: &str) {
    o.extend_from_slice(&(s.len() as u32).to_le_bytes());
    o.extend_from_slice(s.as_bytes());
}
fn put_blob(o: &mut Vec<u8>, b: &[u8]) {
    o.extend_from_slice(&(b.len() as u64).to_le_bytes());
    o.extend_from_slice(b);
}
impl<'a> Reader<'a> {
    fn text(&mut self, max: usize) -> R<String> {
        let n = self.u32()? as usize;
        if n > max { return Err(RagError::Format("string longer than allowed".into())); }
        std::str::from_utf8(self.take(n)?).map(str::to_string).map_err(|_| RagError::Format("string is not UTF-8".into()))
    }
    fn blob(&mut self) -> R<&'a [u8]> {
        let n = self.u64()?;
        if n > self.left() as u64 { return Err(RagError::Format("section longer than the file".into())); }
        self.take(n as usize)
    }
}

impl<'a> KnowledgeBase<'a> {
    /// The whole knowledge base as bytes: the chunk and document tables, the keyword index and the
    /// vector graph (each with its own checksum) and a SHA-256 over everything. The embedder is not
    /// stored, only its [`Embedder::id`]; give the same one to [`KnowledgeBase::from_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(KB_MAGIC);
        o.extend_from_slice(&KB_VERSION.to_le_bytes());
        put_str(&mut o, &self.embedder.id());
        o.extend_from_slice(&(self.embedder.dim() as u32).to_le_bytes());
        o.extend_from_slice(&self.next_chunk.to_le_bytes());
        o.extend_from_slice(&self.next_doc.to_le_bytes());
        let mut docs: Vec<(&u64, &DocRec)> = self.docs.iter().collect();
        docs.sort_by_key(|d| *d.0);
        o.extend_from_slice(&(docs.len() as u32).to_le_bytes());
        for (id, d) in docs {
            o.extend_from_slice(&id.to_le_bytes());
            put_str(&mut o, &d.name);
            o.extend_from_slice(&d.hash);
            o.extend_from_slice(&(d.chunks.len() as u32).to_le_bytes());
            for c in &d.chunks { o.extend_from_slice(&c.to_le_bytes()); }
        }
        let mut chunks: Vec<(&u64, &ChunkRec)> = self.chunks.iter().collect();
        chunks.sort_by_key(|c| *c.0);
        o.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
        for (id, c) in chunks {
            o.extend_from_slice(&id.to_le_bytes());
            o.extend_from_slice(&c.doc.to_le_bytes());
            o.extend_from_slice(&(c.byte_start as u64).to_le_bytes());
            o.extend_from_slice(&(c.byte_end as u64).to_le_bytes());
            put_str(&mut o, &c.text);
        }
        put_blob(&mut o, &self.keywords.to_bytes());
        put_blob(&mut o, &self.vectors.to_bytes());
        let sum = sha256(&o);
        o.extend_from_slice(&sum);
        o
    }

    /// Parse bytes from [`KnowledgeBase::to_bytes`]. Nothing in the file is trusted: counts are
    /// checked against what is left, the tables must agree with each other and with both indexes,
    /// and `embedder` must be the one that made the vectors (same [`Embedder::id`]).
    pub fn from_bytes(b: &[u8], embedder: Box<dyn Embedder + 'a>) -> R<KnowledgeBase<'a>> {
        let bad = |m: &str| RagError::Format(format!("knowledge base: {m}"));
        if b.len() < KB_MAGIC.len() + 32 { return Err(bad("file too short")); }
        let (body, sum) = b.split_at(b.len() - 32);
        if sha256(body) != sum { return Err(bad("checksum mismatch")); }
        let mut r = Reader { b: body, p: 0 };
        if r.take(8)? != KB_MAGIC { return Err(bad("not a knowledge base")); }
        if r.u32()? != KB_VERSION { return Err(bad("unsupported version")); }
        let saved_id = r.text(4096)?;
        let dim = r.u32()? as usize;
        if saved_id != embedder.id() || dim != embedder.dim() {
            return Err(bad(&format!("it was built with embedder '{saved_id}' (dimension {dim}), not '{}' (dimension {})", embedder.id(), embedder.dim())));
        }
        let (next_chunk, next_doc) = (r.u64()?, r.u64()?);
        let nd = r.u32()? as usize;
        if nd.checked_mul(8 + 4 + 32 + 4).is_none_or(|t| t > r.left()) { return Err(bad("document count larger than the file")); }
        let mut docs: HashMap<u64, DocRec> = HashMap::new();
        for _ in 0..nd {
            let id = r.u64()?;
            let name = r.text(1 << 16)?;
            let hash: [u8; 32] = r.take(32)?.try_into().map_err(|_| bad("short read"))?;
            let nc = r.u32()? as usize;
            if nc.checked_mul(8).is_none_or(|t| t > r.left()) { return Err(bad("chunk list larger than the file")); }
            let mut chunks = Vec::with_capacity(nc);
            for _ in 0..nc { chunks.push(r.u64()?); }
            if id >= next_doc || docs.insert(id, DocRec { name, chunks, hash }).is_some() { return Err(bad("bad or duplicate document id")); }
        }
        let nc = r.u32()? as usize;
        if nc.checked_mul(8 * 4 + 4).is_none_or(|t| t > r.left()) { return Err(bad("chunk count larger than the file")); }
        let mut chunks: HashMap<u64, ChunkRec> = HashMap::new();
        for _ in 0..nc {
            let id = r.u64()?;
            let doc = r.u64()?;
            let (bs, be) = (r.u64()?, r.u64()?);
            let text = r.text(MAX_TEXT)?;
            let (Ok(byte_start), Ok(byte_end)) = (usize::try_from(bs), usize::try_from(be)) else { return Err(bad("offset out of range")) };
            if byte_end < byte_start || byte_end - byte_start != text.len() { return Err(bad("a chunk's byte range does not match its text")); }
            if id >= next_chunk || chunks.insert(id, ChunkRec { doc, byte_start, byte_end, text }).is_some() { return Err(bad("bad or duplicate chunk id")); }
        }
        // the two tables must describe each other exactly
        let mut listed = 0usize;
        for (did, d) in &docs {
            for c in &d.chunks {
                match chunks.get(c) { Some(ch) if ch.doc == *did => listed += 1, _ => return Err(bad("a document lists a chunk that is not its own")) }
            }
        }
        if listed != chunks.len() { return Err(bad("a chunk belongs to no document")); }
        let keywords = Bm25Index::from_bytes(r.blob()?)?;
        let vectors = Hnsw::from_bytes(r.blob()?)?;
        if r.left() != 0 { return Err(bad("trailing bytes")); }
        if vectors.dim() != embedder.dim() { return Err(bad("the vector index has another dimension than the embedder")); }
        let same = |mut a: Vec<u64>| { a.sort_unstable(); let mut c: Vec<u64> = chunks.keys().copied().collect(); c.sort_unstable(); a == c };
        if !same(keywords.ids()) { return Err(bad("the keyword index and the chunk table disagree")); }
        if !same(vectors.ids()) { return Err(bad("the vector index and the chunk table disagree")); }
        Ok(KnowledgeBase { embedder, vectors, keywords, chunks, docs, next_chunk, next_doc })
    }

    /// Write to `path` (to a temporary file next to it first, then renamed, so a crash leaves the
    /// old file intact).
    pub fn save(&self, path: &Path) -> R<()> {
        let tmp = path.with_extension("tmp-kb");
        std::fs::write(&tmp, self.to_bytes()).map_err(|e| RagError::Io(e.to_string()))?;
        std::fs::rename(&tmp, path).map_err(|e| { let _ = std::fs::remove_file(&tmp); RagError::Io(e.to_string()) })
    }
    /// Read a file written by [`KnowledgeBase::save`].
    pub fn load(path: &Path, embedder: Box<dyn Embedder + 'a>) -> R<KnowledgeBase<'a>> {
        let b = std::fs::read(path).map_err(|e| RagError::Io(e.to_string()))?;
        KnowledgeBase::from_bytes(&b, embedder)
    }
}

// ───────────────────────────── citations ─────────────────────────────

/// One numbered source: `[S{tag}]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceRef {
    /// 1-based, matches the `[S3]` tag and [`Citation::source_id`].
    pub tag: usize,
    pub doc_id: u64,
    pub doc_name: String,
    pub chunk_id: u64,
    /// Where this passage starts and ends in the document, in bytes.
    pub byte_start: usize,
    pub byte_end: usize,
    pub text: String,
    pub score: f32,
}

/// A span of the original document that a quote came from.
#[derive(Debug, Clone, PartialEq)]
pub struct DocSpan {
    pub doc_id: u64,
    pub doc_name: String,
    pub byte_start: usize,
    pub byte_end: usize,
}

/// Retrieved passages ready to prompt with and to cite from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CitedContext {
    pub sources: Vec<SourceRef>,
}

impl CitedContext {
    pub fn from_hits(hits: Vec<Hit>) -> CitedContext {
        CitedContext {
            sources: hits.into_iter().enumerate().map(|(i, h)| SourceRef {
                tag: i + 1, doc_id: h.doc_id, doc_name: h.doc_name, chunk_id: h.chunk_id,
                byte_start: h.byte_start, byte_end: h.byte_end, text: h.text, score: h.score,
            }).collect(),
        }
    }
    /// The source texts in tag order (what [`QuoteIndex::new`] takes).
    pub fn texts(&self) -> Vec<&str> { self.sources.iter().map(|s| s.text.as_str()).collect() }
    /// Build the quote-lock index over these sources (`[S1]` is `sources[0]`).
    pub fn quote_index(&self, opts: QuoteOptions) -> Result<QuoteIndex, QuoteError> { QuoteIndex::new(&self.texts(), opts) }
    /// The passages as a prompt block: `[S1] (name)` then the text, blank line between.
    pub fn prompt_block(&self) -> String {
        let mut s = String::new();
        for r in &self.sources {
            s.push_str(&format!("[S{}] ({})\n{}\n\n", r.tag, r.doc_name, r.text));
        }
        s
    }
    /// Map a quote-lock citation back to the byte range it covers in the original document.
    /// `Citation` offsets are character offsets into the source text; `None` if the tag or the
    /// range does not fit this context.
    pub fn locate(&self, c: &Citation) -> Option<DocSpan> {
        let src = self.sources.get(c.source_id.checked_sub(1)?)?;
        if c.start > c.end { return None; }
        let mut bytes = src.text.char_indices().map(|(i, _)| i).chain(std::iter::once(src.text.len()));
        let a = bytes.clone().nth(c.start)?;
        let b = bytes.nth(c.end)?;
        Some(DocSpan { doc_id: src.doc_id, doc_name: src.doc_name.clone(), byte_start: src.byte_start + a, byte_end: src.byte_start + b })
    }
}

// ───────────────────────────── ingesting files ─────────────────────────────
//
// Plain text, Markdown, HTML and simple PDFs. Every extractor is lossy by nature, so what comes
// back is the *extracted* text and all byte offsets (chunks, hits, citations) refer to that text.
// For UTF-8 text and Markdown the extracted text is the file itself ([`Extracted::exact_offsets`]).

use std::path::PathBuf;

/// What kind of file a path is, from its extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocKind { Text, Markdown, Html, Pdf }

impl DocKind {
    pub fn of_path(p: &Path) -> Option<DocKind> {
        match p.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "txt" | "text" | "log" | "rst" | "csv" | "tsv" | "json" | "xml" | "yaml" | "yml" | "toml" => Some(DocKind::Text),
            "md" | "markdown" => Some(DocKind::Markdown),
            "html" | "htm" | "xhtml" => Some(DocKind::Html),
            "pdf" => Some(DocKind::Pdf),
            _ => None,
        }
    }
}

/// Text taken out of a file.
#[derive(Debug, Clone, PartialEq)]
pub struct Extracted {
    pub kind: DocKind,
    pub text: String,
    /// `text` is byte for byte the file's content, so offsets in `text` are offsets in the file.
    pub exact_offsets: bool,
}

// ── inflate (RFC 1951) ──

struct Bits<'a> { d: &'a [u8], pos: usize, buf: u32, n: u32 }
impl<'a> Bits<'a> {
    fn new(d: &'a [u8]) -> Self { Bits { d, pos: 0, buf: 0, n: 0 } }
    fn need(&mut self, k: u32) -> Option<()> {
        while self.n < k {
            let b = *self.d.get(self.pos)?;
            self.pos += 1;
            self.buf |= (b as u32) << self.n;
            self.n += 8;
        }
        Some(())
    }
    fn bits(&mut self, k: u32) -> Option<u32> {
        if k == 0 { return Some(0); }
        self.need(k)?;
        let v = self.buf & ((1u32 << k) - 1);
        self.buf >>= k;
        self.n -= k;
        Some(v)
    }
    fn align(&mut self) { self.buf = 0; self.n = 0; }
}

struct Huff { count: [u16; 16], symbol: Vec<u16> }
impl Huff {
    /// Canonical Huffman table from code lengths; `None` if the lengths over-subscribe the code.
    fn new(lengths: &[u8]) -> Option<Huff> {
        let mut count = [0u16; 16];
        for &l in lengths { count[l as usize] += 1; }
        let mut left: i32 = 1;
        for &c in &count[1..] {
            left <<= 1;
            left -= c as i32;
            if left < 0 { return None; }
        }
        let mut offs = [0u16; 16];
        for len in 1..15 { offs[len + 1] = offs[len] + count[len]; }
        let mut symbol = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 { symbol[offs[l as usize] as usize] = sym as u16; offs[l as usize] += 1; }
        }
        Some(Huff { count, symbol })
    }
    fn decode(&self, br: &mut Bits) -> Option<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= br.bits(1)? as i32;
            let c = self.count[len] as i32;
            if code - c < first { return self.symbol.get((index + (code - first)) as usize).copied(); }
            index += c;
            first += c;
            first <<= 1;
            code <<= 1;
        }
        None
    }
}

const LBASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEXT: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DBASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DEXT: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

/// Decompress a raw DEFLATE stream. Output beyond `limit` bytes is an error (a few bytes can
/// expand a thousandfold), as is anything malformed or cut short.
pub fn inflate(data: &[u8], limit: usize) -> R<Vec<u8>> {
    let bad = |m: &str| RagError::Format(format!("deflate: {m}"));
    let mut br = Bits::new(data);
    let mut out: Vec<u8> = Vec::new();
    loop {
        let last = br.bits(1).ok_or_else(|| bad("truncated"))?;
        let kind = br.bits(2).ok_or_else(|| bad("truncated"))?;
        match kind {
            0 => {
                br.align();
                let p = br.pos;
                let hdr = data.get(p..p + 4).ok_or_else(|| bad("truncated"))?;
                let (len, nlen) = (u16::from_le_bytes([hdr[0], hdr[1]]), u16::from_le_bytes([hdr[2], hdr[3]]));
                if len != !nlen { return Err(bad("stored block length check failed")); }
                let body = data.get(p + 4..p + 4 + len as usize).ok_or_else(|| bad("truncated"))?;
                if out.len() + body.len() > limit { return Err(bad("output larger than the limit")); }
                out.extend_from_slice(body);
                br.pos = p + 4 + len as usize;
            }
            1 | 2 => {
                let (lit, dist) = if kind == 1 {
                    let mut l = [0u8; 288];
                    for (i, x) in l.iter_mut().enumerate() { *x = match i { 0..=143 => 8, 144..=255 => 9, 256..=279 => 7, _ => 8 }; }
                    (Huff::new(&l).ok_or_else(|| bad("fixed table"))?, Huff::new(&[5u8; 30]).ok_or_else(|| bad("fixed table"))?)
                } else {
                    let hlit = br.bits(5).ok_or_else(|| bad("truncated"))? as usize + 257;
                    let hdist = br.bits(5).ok_or_else(|| bad("truncated"))? as usize + 1;
                    let hclen = br.bits(4).ok_or_else(|| bad("truncated"))? as usize + 4;
                    if hlit > 286 || hdist > 30 { return Err(bad("too many codes")); }
                    const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                    let mut cl = [0u8; 19];
                    for &o in ORDER.iter().take(hclen) { cl[o] = br.bits(3).ok_or_else(|| bad("truncated"))? as u8; }
                    let clh = Huff::new(&cl).ok_or_else(|| bad("code-length table"))?;
                    let mut lens = vec![0u8; hlit + hdist];
                    let mut i = 0;
                    while i < hlit + hdist {
                        let sym = clh.decode(&mut br).ok_or_else(|| bad("bad code-length symbol"))?;
                        if sym < 16 { lens[i] = sym as u8; i += 1; continue; }
                        let (val, rep) = match sym {
                            16 => { if i == 0 { return Err(bad("repeat with nothing before")); } (lens[i - 1], 3 + br.bits(2).ok_or_else(|| bad("truncated"))? as usize) }
                            17 => (0, 3 + br.bits(3).ok_or_else(|| bad("truncated"))? as usize),
                            _ => (0, 11 + br.bits(7).ok_or_else(|| bad("truncated"))? as usize),
                        };
                        if i + rep > hlit + hdist { return Err(bad("code lengths overrun")); }
                        lens[i..i + rep].fill(val);
                        i += rep;
                    }
                    if lens[256] == 0 { return Err(bad("no end-of-block code")); }
                    (Huff::new(&lens[..hlit]).ok_or_else(|| bad("literal table"))?, Huff::new(&lens[hlit..]).ok_or_else(|| bad("distance table"))?)
                };
                loop {
                    let sym = lit.decode(&mut br).ok_or_else(|| bad("bad literal symbol"))? as usize;
                    match sym {
                        0..=255 => {
                            if out.len() >= limit { return Err(bad("output larger than the limit")); }
                            out.push(sym as u8);
                        }
                        256 => break,
                        257..=285 => {
                            let i = sym - 257;
                            let len = LBASE[i] as usize + br.bits(LEXT[i] as u32).ok_or_else(|| bad("truncated"))? as usize;
                            let ds = dist.decode(&mut br).ok_or_else(|| bad("bad distance symbol"))? as usize;
                            if ds >= 30 { return Err(bad("distance symbol out of range")); }
                            let d = DBASE[ds] as usize + br.bits(DEXT[ds] as u32).ok_or_else(|| bad("truncated"))? as usize;
                            if d > out.len() { return Err(bad("distance reaches before the start")); }
                            if out.len() + len > limit { return Err(bad("output larger than the limit")); }
                            for _ in 0..len { out.push(out[out.len() - d]); }
                        }
                        _ => return Err(bad("bad literal symbol")),
                    }
                }
            }
            _ => return Err(bad("reserved block type")),
        }
        if last == 1 { break; }
    }
    Ok(out)
}

/// Decompress a zlib stream (RFC 1950): two header bytes, DEFLATE data; the checksum is ignored.
pub fn zlib_decompress(data: &[u8], limit: usize) -> R<Vec<u8>> {
    if data.len() < 2 || data[0] & 0x0f != 8 || !(((data[0] as u16) << 8) | data[1] as u16).is_multiple_of(31) || data[1] & 0x20 != 0 {
        return Err(RagError::Format("zlib: bad header".into()));
    }
    inflate(&data[2..], limit)
}

// ── HTML ──

fn entity(name: &str) -> Option<char> {
    if let Some(num) = name.strip_prefix('#') {
        let v = match num.strip_prefix(['x', 'X']) { Some(h) => u32::from_str_radix(h, 16).ok()?, None => num.parse().ok()? };
        return char::from_u32(v).filter(|c| *c != '\0');
    }
    Some(match name {
        "amp" => '&', "lt" => '<', "gt" => '>', "quot" => '"', "apos" => '\'', "nbsp" => ' ', "copy" => '©', "reg" => '®',
        "ndash" => '–', "mdash" => '—', "hellip" => '…', "laquo" => '«', "raquo" => '»', "lsquo" => '‘', "rsquo" => '’',
        "ldquo" => '“', "rdquo" => '”', "bdquo" => '„', "euro" => '€', "deg" => '°',
        "abreve" => 'ă', "Abreve" => 'Ă', "acirc" => 'â', "Acirc" => 'Â', "icirc" => 'î', "Icirc" => 'Î',
        "scedil" => 'ş', "Scedil" => 'Ş', "tcedil" => 'ţ', "Tcedil" => 'Ţ',
        "aacute" => 'á', "eacute" => 'é', "iacute" => 'í', "oacute" => 'ó', "uacute" => 'ú', "agrave" => 'à', "egrave" => 'è',
        "auml" => 'ä', "ouml" => 'ö', "uuml" => 'ü', "szlig" => 'ß', "ccedil" => 'ç', "ntilde" => 'ñ',
        _ => return None,
    })
}

/// Visible text of an HTML page: tags, comments, `<script>` and `<style>` are dropped, entities
/// decoded, block elements become line breaks, runs of blanks collapse to one space and runs of
/// blank lines to one.
pub fn html_to_text(html: &str) -> String {
    const BLOCK: &[&str] = &["p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "header", "footer", "nav", "main",
        "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "hr", "dd", "dt", "dl", "figure", "figcaption", "form", "title"];
    let b = html.as_bytes();
    let mut out = String::with_capacity(html.len() / 2);
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'<' => {
                if html[i..].starts_with("<!--") {
                    i = html[i + 4..].find("-->").map_or(b.len(), |e| i + 4 + e + 3);
                    continue;
                }
                // tag name
                let mut j = i + 1;
                let closing = b.get(j) == Some(&b'/');
                if closing { j += 1; }
                let ns = j;
                while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b':') { j += 1; }
                if j == ns { // a lone '<' that is text
                    out.push('<');
                    i += 1;
                    continue;
                }
                let name = html[ns..j].to_ascii_lowercase();
                // to the closing '>', skipping quoted attribute values
                let mut q = 0u8;
                while j < b.len() {
                    match (q, b[j]) {
                        (0, b'"') | (0, b'\'') => q = b[j],
                        (x, c) if x != 0 && c == x => q = 0,
                        (0, b'>') => break,
                        _ => {}
                    }
                    j += 1;
                }
                i = (j + 1).min(b.len());
                if !closing && (name == "script" || name == "style") {
                    let lower_rest = html[i..].to_ascii_lowercase();
                    let end = format!("</{name}");
                    i = lower_rest.find(&end).map_or(b.len(), |e| {
                        let from = i + e;
                        html[from..].find('>').map_or(b.len(), |g| from + g + 1)
                    });
                } else if BLOCK.contains(&name.as_str()) {
                    out.push('\n');
                }
            }
            b'&' => {
                let rest = &html[i + 1..];
                let end = rest.char_indices().take(12).find(|&(_, c)| c == ';').map(|(k, _)| k);
                match end.and_then(|e| entity(&rest[..e]).map(|c| (c, e))) {
                    Some((c, e)) => { out.push(c); i += e + 2; }
                    None => { out.push('&'); i += 1; }
                }
            }
            _ => {
                let ch = html[i..].chars().next().unwrap_or('\u{fffd}');
                out.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    collapse_blanks(&out)
}

/// Spaces and tabs → one space, lines trimmed, at most one blank line in a row.
fn collapse_blanks(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blank_run = 0;
    for line in s.lines() {
        let mut l = String::with_capacity(line.len());
        let mut sp = false;
        for c in line.chars() {
            if c.is_whitespace() || c == '\u{a0}' { sp = true; continue; }
            if sp && !l.is_empty() { l.push(' '); }
            sp = false;
            l.push(c);
        }
        if l.is_empty() {
            blank_run += 1;
            if blank_run == 1 && !out.is_empty() { out.push('\n'); }
        } else {
            blank_run = 0;
            out.push_str(&l);
            out.push('\n');
        }
    }
    out.truncate(out.trim_end().len());
    out
}

// ── PDF ──

fn find(h: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from >= h.len() { return None; }
    h[from..].windows(needle.len()).position(|w| w == needle).map(|p| p + from)
}
fn rfind(h: &[u8], needle: &[u8], before: usize) -> Option<usize> {
    let end = before.min(h.len());
    if needle.len() > end { return None; }
    h[..end].windows(needle.len()).rposition(|w| w == needle)
}

fn winansi(b: u8) -> char {
    const HI: [char; 32] = ['€', '\u{81}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{8d}', 'Ž', '\u{8f}',
        '\u{90}', '‘', '’', '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u{9d}', 'ž', 'Ÿ'];
    if (0x80..0xa0).contains(&b) { HI[(b - 0x80) as usize] } else { b as char }
}

/// Text from one decompressed content stream: the strings shown by `Tj`, `TJ`, `'` and `"` inside
/// `BT … ET`, with a line break at line-moving operators and a space at big kerning gaps.
fn pdf_content_text(c: &[u8]) -> String {
    let mut out = String::new();
    let mut i = 0;
    let mut in_text = false;
    let mut pending: Vec<u8> = Vec::new(); // strings seen since the last operator
    let mut array_depth = 0;
    let mut last_num: Option<f64> = None;
    let mut nums: Vec<f64> = Vec::new();
    let flush = |pending: &mut Vec<u8>, out: &mut String| {
        out.extend(pending.iter().map(|&b| winansi(b)));
        pending.clear();
    };
    while i < c.len() {
        let ch = c[i];
        match ch {
            b'(' => {
                let mut depth = 1;
                i += 1;
                let mut s = Vec::new();
                while i < c.len() && depth > 0 {
                    match c[i] {
                        b'\\' => {
                            i += 1;
                            match c.get(i) {
                                Some(b'n') => s.push(b'\n'), Some(b'r') => s.push(b'\r'), Some(b't') => s.push(b'\t'),
                                Some(b'b') => s.push(8), Some(b'f') => s.push(12),
                                Some(d @ b'0'..=b'7') => {
                                    let mut v = (d - b'0') as u32;
                                    for _ in 0..2 {
                                        match c.get(i + 1) { Some(e @ b'0'..=b'7') => { v = v * 8 + (e - b'0') as u32; i += 1; } _ => break }
                                    }
                                    s.push((v & 0xff) as u8);
                                }
                                Some(b'\r') => { if c.get(i + 1) == Some(&b'\n') { i += 1; } }
                                Some(b'\n') => {}
                                Some(&o) => s.push(o),
                                None => {}
                            }
                        }
                        b'(' => { depth += 1; s.push(b'('); }
                        b')' => { depth -= 1; if depth > 0 { s.push(b')'); } }
                        o => s.push(o),
                    }
                    i += 1;
                }
                if in_text { pending.extend(s); }
                continue;
            }
            b'<' if c.get(i + 1) != Some(&b'<') => {
                let end = c[i..].iter().position(|&x| x == b'>').map_or(c.len(), |e| i + e);
                let hex: Vec<u8> = c[i + 1..end].iter().filter(|x| x.is_ascii_hexdigit()).copied().collect();
                let bytes: Vec<u8> = hex.chunks(2).map(|p| {
                    let v = |x: u8| (x as char).to_digit(16).unwrap_or(0) as u8;
                    v(p[0]) << 4 | p.get(1).map_or(0, |&q| v(q))
                }).collect();
                if in_text { pending.extend(bytes); }
                i = (end + 1).min(c.len());
                continue;
            }
            b'[' => { array_depth += 1; i += 1; continue; }
            b']' => { array_depth = (array_depth - 1).max(0); i += 1; continue; }
            b'%' => { while i < c.len() && c[i] != b'\n' && c[i] != b'\r' { i += 1; } continue; }
            b'-' | b'+' | b'.' | b'0'..=b'9' => {
                let st = i;
                i += 1;
                while i < c.len() && (c[i].is_ascii_digit() || c[i] == b'.') { i += 1; }
                let v = std::str::from_utf8(&c[st..i]).ok().and_then(|t| t.parse::<f64>().ok());
                if let Some(v) = v {
                    if array_depth > 0 && in_text && v < -200.0 && !pending.is_empty() { pending.push(b' '); }
                    last_num = Some(v);
                    nums.push(v);
                    if nums.len() > 8 { nums.remove(0); }
                }
                continue;
            }
            b if b.is_ascii_alphabetic() || b == b'\'' || b == b'"' || b == b'*' => {
                let st = i;
                i += 1;
                while i < c.len() && (c[i].is_ascii_alphanumeric() || c[i] == b'*' || c[i] == b'\'' || c[i] == b'"') { i += 1; }
                match &c[st..i] {
                    b"BT" => { in_text = true; pending.clear(); }
                    b"ET" => { flush(&mut pending, &mut out); in_text = false; out.push('\n'); }
                    b"Tj" | b"TJ" if in_text => flush(&mut pending, &mut out),
                    b"'" | b"\"" if in_text => { out.push('\n'); flush(&mut pending, &mut out); }
                    b"T*" if in_text => { flush(&mut pending, &mut out); out.push('\n'); }
                    b"Td" | b"TD" if in_text => {
                        flush(&mut pending, &mut out);
                        // a vertical move is a new line, a horizontal one a gap
                        if last_num.is_some_and(|y| y.abs() > 0.5) { out.push('\n'); } else if !out.ends_with([' ', '\n']) { out.push(' '); }
                    }
                    b"Tm" if in_text => { flush(&mut pending, &mut out); if !out.ends_with(['\n', ' ']) { out.push('\n'); } }
                    _ => {}
                }
                last_num = None;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    out
}

/// Text from a PDF whose pages use ordinary single-byte fonts. It reads the content streams
/// (stored or Flate-compressed) and the text operators in them; it does **not** read font
/// encodings, so a PDF that maps its own glyph numbers to letters (almost every Romanian one with
/// ș/ț, and anything made from scans) gives garbage or nothing, and that is reported as an error
/// instead of returning noise. Encrypted files are refused.
pub fn pdf_to_text(bytes: &[u8]) -> R<String> {
    const LIMIT: usize = 64 << 20;
    let bad = |m: &str| RagError::Format(format!("pdf: {m}"));
    if !bytes.starts_with(b"%PDF-") { return Err(bad("missing %PDF header")); }
    if find(bytes, b"/Encrypt", 0).is_some() { return Err(bad("the file is encrypted")); }
    let (mut pages, mut total_out) = (Vec::<String>::new(), 0usize);
    let mut pos = 0;
    let mut seen = 0usize;
    while let Some(kw) = find(bytes, b"stream", pos) {
        pos = kw + 6;
        if kw >= 3 && &bytes[kw - 3..kw] == b"end" { continue; } // "endstream"
        seen += 1;
        if seen > 200_000 { return Err(bad("too many streams")); }
        let mut d = kw + 6;
        if bytes.get(d) == Some(&b'\r') { d += 1; }
        if bytes.get(d) == Some(&b'\n') { d += 1; }
        let Some(end) = find(bytes, b"endstream", d) else { break };
        pos = end + 9;
        let dict_start = rfind(bytes, b"obj", kw).unwrap_or(0);
        let dict = &bytes[dict_start..kw];
        let has = |k: &[u8]| find(dict, k, 0).is_some();
        if has(b"/Image") || has(b"/FontFile") || has(b"/Length1") || has(b"/XRef") || has(b"/ObjStm") || has(b"/ICCBased") || has(b"/Metadata") { continue; }
        let raw = &bytes[d..end];
        let data = if has(b"/FlateDecode") || has(b"/Fl ") {
            if has(b"/DecodeParms") && has(b"/Predictor") { continue; }
            match zlib_decompress(raw, LIMIT.saturating_sub(total_out)) { Ok(v) => v, Err(_) => continue }
        } else if has(b"/Filter") { continue } else { raw.to_vec() };
        total_out += data.len();
        if !data.windows(2).any(|w| w == b"BT") { continue; }
        let t = pdf_content_text(&data);
        if t.trim().is_empty() { continue; }
        pages.push(t);
    }
    let text = collapse_blanks(&pages.join("\n\n"));
    let letters = text.chars().filter(|c| c.is_alphabetic()).count();
    let junk = text.chars().filter(|c| c.is_control() && !c.is_whitespace() || *c == '\u{fffd}').count();
    if letters < 10 || junk * 10 > text.chars().count().max(1) {
        return Err(bad("no readable text (scanned pages, or fonts with their own encoding, which this reader does not decode)"));
    }
    Ok(text)
}

// ── files and directories ──

/// Read `bytes` as the kind of file `kind` names.
pub fn extract_text(kind: DocKind, bytes: &[u8]) -> R<Extracted> {
    let decode = |bytes: &[u8]| -> (String, bool) {
        if bytes.starts_with(&[0xff, 0xfe]) || bytes.starts_with(&[0xfe, 0xff]) {
            let be = bytes[0] == 0xfe;
            let units: Vec<u16> = bytes[2..].chunks_exact(2).map(|p| if be { u16::from_be_bytes([p[0], p[1]]) } else { u16::from_le_bytes([p[0], p[1]]) }).collect();
            return (String::from_utf16_lossy(&units), false);
        }
        let (body, bom) = match bytes.strip_prefix(&[0xef, 0xbb, 0xbf]) { Some(r) => (r, true), None => (bytes, false) };
        match std::str::from_utf8(body) {
            Ok(s) => (s.to_string(), !bom),
            // not UTF-8: read it as Windows-1252, the usual legacy encoding of Western text
            Err(_) => (body.iter().map(|&b| winansi(b)).collect(), false),
        }
    };
    match kind {
        DocKind::Text | DocKind::Markdown => { let (text, exact) = decode(bytes); Ok(Extracted { kind, text, exact_offsets: exact }) }
        DocKind::Html => Ok(Extracted { kind, text: html_to_text(&decode(bytes).0), exact_offsets: false }),
        DocKind::Pdf => Ok(Extracted { kind, text: pdf_to_text(bytes)?, exact_offsets: false }),
    }
}

/// Limits and choices for walking a directory.
#[derive(Debug, Clone)]
pub struct IngestOptions {
    pub chunk: ChunkOptions,
    /// Larger files are skipped.
    pub max_file_bytes: u64,
    pub max_files: usize,
    pub max_depth: usize,
    /// Only these kinds are read.
    pub kinds: Vec<DocKind>,
    /// Remove documents that came from this directory earlier but whose file is gone.
    pub prune: bool,
}
impl Default for IngestOptions {
    fn default() -> Self {
        IngestOptions { chunk: ChunkOptions::default(), max_file_bytes: 32 << 20, max_files: 50_000, max_depth: 16,
            kinds: vec![DocKind::Text, DocKind::Markdown, DocKind::Html, DocKind::Pdf], prune: false }
    }
}

/// One file's text and its chunks (offsets into `text`).
#[derive(Debug, Clone, PartialEq)]
pub struct FileChunks {
    pub path: PathBuf,
    /// Path relative to the directory, with `/` separators: the document name.
    pub name: String,
    pub kind: DocKind,
    pub text: String,
    pub exact_offsets: bool,
    pub chunks: Vec<Chunk>,
}

/// Why a file was left out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skipped { pub name: String, pub reason: String }

fn walk(dir: &Path, rel: &str, depth: usize, o: &IngestOptions, files: &mut Vec<(PathBuf, String, DocKind)>, skipped: &mut Vec<Skipped>) -> R<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir).map_err(|e| RagError::Io(format!("{}: {e}", dir.display())))?
        .filter_map(Result::ok).collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let name = e.file_name().to_string_lossy().into_owned();
        let rel_name = if rel.is_empty() { name.clone() } else { format!("{rel}/{name}") };
        let Ok(meta) = std::fs::symlink_metadata(e.path()) else { skipped.push(Skipped { name: rel_name, reason: "cannot read its metadata".into() }); continue };
        if meta.file_type().is_symlink() { skipped.push(Skipped { name: rel_name, reason: "symbolic link (not followed)".into() }); continue; }
        if meta.is_dir() {
            if name.starts_with('.') { continue; }
            if depth >= o.max_depth { skipped.push(Skipped { name: rel_name, reason: "directory nested deeper than the limit".into() }); continue; }
            walk(&e.path(), &rel_name, depth + 1, o, files, skipped)?;
        } else if meta.is_file() {
            let Some(kind) = DocKind::of_path(&e.path()).filter(|k| o.kinds.contains(k)) else { continue };
            if meta.len() > o.max_file_bytes { skipped.push(Skipped { name: rel_name, reason: format!("{} bytes is over the {}-byte limit", meta.len(), o.max_file_bytes) }); continue; }
            if files.len() >= o.max_files { skipped.push(Skipped { name: rel_name, reason: "file count limit reached".into() }); continue; }
            files.push((e.path(), rel_name, kind));
        }
    }
    Ok(())
}

/// Read every supported file under `dir` (no symbolic links, no hidden directories, size, count and
/// depth limits) and cut it into chunks, without embedding anything. Files come in name order.
pub fn chunk_dir(dir: &Path, o: &IngestOptions) -> R<(Vec<FileChunks>, Vec<Skipped>)> {
    let (mut files, mut skipped) = (Vec::new(), Vec::new());
    walk(dir, "", 0, o, &mut files, &mut skipped)?;
    let mut out = Vec::new();
    for (path, name, kind) in files {
        let bytes = match std::fs::read(&path) { Ok(b) => b, Err(e) => { skipped.push(Skipped { name, reason: e.to_string() }); continue } };
        match extract_text(kind, &bytes) {
            Ok(x) => {
                let chunks = chunk_text(&x.text, &o.chunk);
                if chunks.is_empty() { skipped.push(Skipped { name, reason: "no text".into() }); continue; }
                out.push(FileChunks { path, name, kind, text: x.text, exact_offsets: x.exact_offsets, chunks });
            }
            Err(e) => skipped.push(Skipped { name, reason: e.to_string() }),
        }
    }
    Ok((out, skipped))
}

/// What [`KnowledgeBase::ingest_dir`] did, by document name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IngestReport {
    pub added: Vec<String>,
    pub replaced: Vec<String>,
    pub unchanged: Vec<String>,
    pub removed: Vec<String>,
    pub skipped: Vec<Skipped>,
    /// Files that could be read but failed while being embedded (the old version, if any, stays).
    pub failed: Vec<Skipped>,
}

impl KnowledgeBase<'_> {
    /// Bring the knowledge base in line with a directory: new files are added, changed files
    /// replaced (identical ones cost nothing), and with `prune` documents whose file has vanished
    /// are removed. Document names are the files' relative paths, so another directory with the
    /// same relative names would overwrite them: use one knowledge base per tree.
    pub fn ingest_dir(&mut self, dir: &Path, o: &IngestOptions) -> R<IngestReport> {
        let (files, skipped) = chunk_dir(dir, o)?;
        let mut rep = IngestReport { skipped, ..IngestReport::default() };
        let mut present = std::collections::HashSet::new();
        for f in &files {
            present.insert(f.name.clone());
            match self.upsert_document(&f.name, &f.text, &o.chunk) {
                Ok(Upsert::Added(_)) => rep.added.push(f.name.clone()),
                Ok(Upsert::Replaced { .. }) => rep.replaced.push(f.name.clone()),
                Ok(Upsert::Unchanged(_)) => rep.unchanged.push(f.name.clone()),
                Err(e) => rep.failed.push(Skipped { name: f.name.clone(), reason: e.to_string() }),
            }
        }
        if o.prune {
            // only names that were skipped for a reason other than "gone" are kept
            let kept: std::collections::HashSet<&str> = rep.skipped.iter().map(|s| s.name.as_str()).collect();
            for d in self.documents() {
                if !present.contains(&d.name) && !kept.contains(d.name.as_str()) {
                    self.remove_document(d.id)?;
                    rep.removed.push(d.name);
                }
            }
        }
        Ok(rep)
    }
}

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(s: &mut u64) -> f32 { (splitmix(s) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0 }
    fn rvec(d: usize, s: &mut u64) -> Vec<f32> { (0..d).map(|_| rnd(s)).collect() }

    fn recall(idx: &Hnsw, queries: usize, k: usize, seed: u64) -> f64 {
        let mut s = seed;
        let (mut hit, mut tot) = (0usize, 0usize);
        for _ in 0..queries {
            let q = rvec(idx.dim(), &mut s);
            let exact: Vec<u64> = idx.brute_force(&q, k).unwrap().iter().map(|x| x.0).collect();
            let got: Vec<u64> = idx.search(&q, k, None).unwrap().iter().map(|x| x.0).collect();
            hit += got.iter().filter(|g| exact.contains(g)).count();
            tot += exact.len();
        }
        hit as f64 / tot as f64
    }

    fn build(n: usize, d: usize, p: HnswParams, seed: u64) -> Hnsw {
        let mut s = seed;
        let mut h = Hnsw::new(d, p).unwrap();
        for i in 0..n { h.insert(i as u64 * 3 + 1, &rvec(d, &mut s)).unwrap(); }
        h
    }

    #[test]
    fn recall_at_10_is_high_cosine_f32() {
        let h = build(3000, 32, HnswParams::default(), 11);
        let r = recall(&h, 100, 10, 99);
        assert!(r >= 0.95, "recall {r}");
    }

    #[test]
    fn recall_at_10_is_high_dot_and_int8() {
        let p = HnswParams { metric: Metric::Dot, ..Default::default() };
        let h = build(2000, 24, p, 5);
        let r = recall(&h, 80, 10, 7);
        assert!(r >= 0.95, "dot recall {r}");
        let p = HnswParams { storage: Storage::Int8, ..Default::default() };
        let h = build(2000, 24, p, 5);
        let r = recall(&h, 80, 10, 7);
        assert!(r >= 0.95, "int8 recall {r}");
    }

    #[test]
    fn int8_is_smaller_and_close_to_f32() {
        let a = build(500, 64, HnswParams::default(), 3);
        let b = build(500, 64, HnswParams { storage: Storage::Int8, ..Default::default() }, 3);
        assert!(b.to_bytes().len() < a.to_bytes().len() * 6 / 10);
        let v = a.vector_of(1).unwrap();
        let w = b.vector_of(1).unwrap();
        assert!(dot(&v, &w) > 0.999);
    }

    #[test]
    fn exact_match_is_found_first() {
        let mut s = 42;
        let mut h = Hnsw::new(16, HnswParams::default()).unwrap();
        let mut keep = Vec::new();
        for i in 0..400u64 { let v = rvec(16, &mut s); if i == 123 { keep = v.clone(); } h.insert(i, &v).unwrap(); }
        let r = h.search(&keep, 1, None).unwrap();
        assert_eq!(r[0].0, 123);
        assert!(r[0].1 > 0.9999);
    }

    #[test]
    fn deterministic_graph_and_results() {
        let a = build(600, 16, HnswParams::default(), 8);
        let b = build(600, 16, HnswParams::default(), 8);
        assert_eq!(a.to_bytes(), b.to_bytes());
        let c = build(600, 16, HnswParams { seed: 1, ..Default::default() }, 8);
        assert_ne!(a.to_bytes(), c.to_bytes());
        let q = vec![0.3; 16];
        assert_eq!(a.search(&q, 10, None).unwrap(), b.search(&q, 10, None).unwrap());
    }

    #[test]
    fn persistence_round_trip_both_storages() {
        for st in [Storage::F32, Storage::Int8] {
            let mut h = build(300, 12, HnswParams { storage: st, ..Default::default() }, 2);
            h.remove(4).unwrap();
            let bytes = h.to_bytes();
            let g = Hnsw::from_bytes(&bytes).unwrap();
            assert_eq!(g.to_bytes(), bytes);
            assert_eq!(g.len(), h.len());
            assert_eq!(g.tombstones(), 1);
            let q = vec![0.1; 12];
            assert_eq!(g.search(&q, 10, None).unwrap(), h.search(&q, 10, None).unwrap());
            // an index loaded from disk keeps accepting inserts
            let mut g = g;
            g.insert(10_000, &q).unwrap();
            assert_eq!(g.search(&q, 1, None).unwrap()[0].0, 10_000);
        }
    }

    #[test]
    fn save_and_load_file() {
        let dir = std::env::temp_dir().join(format!("rag-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("idx.bin");
        let h = build(100, 8, HnswParams::default(), 1);
        h.save(&p).unwrap();
        assert!(!dir.join("idx.tmp-hnsw").exists());
        let g = Hnsw::load(&p).unwrap();
        assert_eq!(g.len(), 100);
        assert!(matches!(Hnsw::load(&dir.join("missing.bin")), Err(RagError::Io(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tombstones_hide_entries_and_compact_drops_them() {
        let mut h = build(500, 16, HnswParams::default(), 6);
        let q = vec![0.2; 16];
        let top: Vec<u64> = h.search(&q, 10, None).unwrap().iter().map(|x| x.0).collect();
        for id in &top[..5] { h.remove(*id).unwrap(); }
        assert_eq!(h.len(), 495);
        let after = h.search(&q, 10, None).unwrap();
        assert_eq!(after.len(), 10);
        assert!(after.iter().all(|x| !top[..5].contains(&x.0)));
        assert!(!h.contains(top[0]));
        assert_eq!(h.remove(top[0]), Err(RagError::Unknown(top[0])));
        // remove most of the index: results stay correct and only ever live
        for id in h.ids().into_iter().take(400) { h.remove(id).unwrap(); }
        let r = h.search(&q, 10, None).unwrap();
        assert!(r.iter().all(|x| h.contains(x.0)));
        assert!(recall(&h, 30, 5, 3) >= 0.9);
        assert_eq!(h.compact().unwrap(), 405);
        assert_eq!(h.tombstones(), 0);
        assert_eq!(h.len(), 95);
        assert!(recall(&h, 30, 5, 3) >= 0.95);
        // everything gone
        for id in h.ids() { h.remove(id).unwrap(); }
        assert!(h.search(&q, 5, None).unwrap().is_empty());
        h.compact().unwrap();
        assert!(h.is_empty());
        h.insert(1, &q).unwrap();
        assert_eq!(h.search(&q, 1, None).unwrap()[0].0, 1);
    }

    #[test]
    fn replacing_an_id_keeps_one_live_entry() {
        let mut h = Hnsw::new(4, HnswParams::default()).unwrap();
        h.insert(7, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        h.insert(7, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(h.len(), 1);
        let r = h.search(&[0.0, 1.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(r.len(), 1);
        assert!(r[0].1 > 0.999);
    }

    #[test]
    fn adversarial_inputs_are_rejected_not_fatal() {
        let mut h = Hnsw::new(4, HnswParams::default()).unwrap();
        assert!(matches!(h.insert(1, &[1.0, 2.0]), Err(RagError::Dim { .. })));
        assert_eq!(h.insert(1, &[f32::NAN, 0.0, 0.0, 0.0]), Err(RagError::NotFinite));
        assert_eq!(h.insert(1, &[f32::INFINITY, 0.0, 0.0, 0.0]), Err(RagError::NotFinite));
        assert!(h.search(&[0.0; 4], 3, None).unwrap().is_empty());
        h.insert(1, &[0.0; 4]).unwrap(); // zero vector
        h.insert(2, &[1e30, 1e30, 0.0, 0.0]).unwrap();
        h.insert(3, &[1e-30, 0.0, 0.0, 0.0]).unwrap();
        for _ in 0..50 { h.insert(9, &[1.0, 1.0, 1.0, 1.0]).unwrap(); }
        assert_eq!(h.search(&[0.0; 4], 10, None).unwrap().len(), 4);
        assert!(h.search(&[1.0; 4], 0, None).unwrap().is_empty());
        assert!(h.search(&[1.0; 4], usize::MAX, None).unwrap().len() <= 4);
        assert!(matches!(h.search(&[1.0; 5], 1, None), Err(RagError::Dim { .. })));
        assert!(Hnsw::new(0, HnswParams::default()).is_err());
        assert!(Hnsw::new(MAX_DIM + 1, HnswParams::default()).is_err());
        assert!(Hnsw::new(4, HnswParams { m: 1, ..Default::default() }).is_err());
        assert!(Hnsw::new(4, HnswParams { ef_search: 0, ..Default::default() }).is_err());
        // identical vectors in bulk
        let mut h = Hnsw::new(8, HnswParams::default()).unwrap();
        for i in 0..300u64 { h.insert(i, &[0.5; 8]).unwrap(); }
        assert_eq!(h.search(&[0.5; 8], 10, None).unwrap().len(), 10);
    }

    #[test]
    fn mass_duplicates_do_not_hide_other_entries() {
        for st in [Storage::F32, Storage::Int8] {
            let mut s = 21u64;
            let mut h = Hnsw::new(8, HnswParams { storage: st, ..Default::default() }).unwrap();
            let mut others = Vec::new();
            for i in 0..60u64 { let v = rvec(8, &mut s); others.push(v.clone()); h.insert(i, &v).unwrap(); }
            for i in 100..600u64 { h.insert(i, &[0.3, -0.2, 0.9, 0.1, 0.0, 0.5, -0.7, 0.2]).unwrap(); }
            for (i, v) in others.iter().enumerate() {
                assert_eq!(h.search(v, 1, None).unwrap()[0].0, i as u64, "{st:?}");
            }
            // every duplicate is reachable too, and replaced ones stay replaced
            let dup = [0.3, -0.2, 0.9, 0.1, 0.0, 0.5, -0.7, 0.2];
            assert_eq!(h.search(&dup, 500, Some(600)).unwrap().len(), 500);
            h.insert(100, &others[0]).unwrap();
            assert_eq!(h.len(), 560);
            let g = Hnsw::from_bytes(&h.to_bytes()).unwrap();
            assert_eq!(g.search(&dup, 499, Some(600)).unwrap().len(), 499);
        }
    }

    fn reseal(mut body: Vec<u8>) -> Vec<u8> {
        let s = sha256(&body);
        body.extend_from_slice(&s);
        body
    }

    #[test]
    fn hostile_files_never_panic() {
        let h = build(60, 8, HnswParams::default(), 4);
        let good = h.to_bytes();
        // truncation at every length
        for n in 0..good.len() { assert!(Hnsw::from_bytes(&good[..n]).is_err(), "len {n}"); }
        // every single-bit flip is caught by the checksum
        for i in (0..good.len()).step_by(7) {
            let mut b = good.clone();
            b[i] ^= 0x10;
            assert!(Hnsw::from_bytes(&b).is_err(), "flip {i}");
        }
        // forged files with a *valid* checksum: mutate the body at many places and reseal
        let body = &good[..good.len() - 32];
        let mut s = 77u64;
        let mut accepted = 0;
        for _ in 0..3000 {
            let mut b = body.to_vec();
            for _ in 0..1 + splitmix(&mut s) % 3 {
                let i = (splitmix(&mut s) % b.len() as u64) as usize;
                b[i] = splitmix(&mut s) as u8;
            }
            if let Ok(g) = Hnsw::from_bytes(&reseal(b)) {
                accepted += 1;
                // whatever got through must still be searchable without panicking
                let _ = g.search(&[0.1; 8], 5, None);
                let _ = g.brute_force(&[0.1; 8], 5);
            }
        }
        assert!(accepted < 3000);
        // absurd headers
        let mut b = body.to_vec();
        b[8 + 4..8 + 8].copy_from_slice(&u32::MAX.to_le_bytes()); // dim
        assert!(Hnsw::from_bytes(&reseal(b)).is_err());
        let mut b = body.to_vec();
        let off = 8 + 4 + 4 * 4 + 2 + 8 + 8; // node count
        b[off..off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Hnsw::from_bytes(&reseal(b)).is_err());
        assert!(Hnsw::from_bytes(&[]).is_err());
        assert!(Hnsw::from_bytes(&[0u8; 100]).is_err());
        let mut b = body.to_vec();
        b.extend_from_slice(&[1, 2, 3]);
        assert!(Hnsw::from_bytes(&reseal(b)).is_err());
        // wrong magic with valid checksum
        let mut b = body.to_vec();
        b[0] = b'X';
        assert!(Hnsw::from_bytes(&reseal(b)).is_err());
    }

    #[test]
    fn chunking_keeps_offsets_and_sentences() {
        let text = "Prima propoziție e scurtă. A doua este puțin mai lungă, dar tot ușoară! Dr. Popescu a venit la ora 5. Apoi a plecat.\n\nUn paragraf nou începe aici. Și continuă cu încă o idee.";
        let opts = ChunkOptions { max_chars: 60, overlap_chars: 30 };
        let cs = chunk_text(text, &opts);
        assert!(cs.len() >= 3);
        for c in &cs {
            assert_eq!(&text[c.byte_start..c.byte_end], c.text);
            assert!(c.text.chars().count() <= 60, "{:?}", c.text);
            assert_eq!(c.text, c.text.trim());
        }
        // starts strictly advance, and together the chunks cover the whole text
        for w in cs.windows(2) { assert!(w[1].byte_start > w[0].byte_start); assert!(w[1].byte_end > w[0].byte_end); }
        assert_eq!(cs[0].byte_start, 0);
        assert_eq!(cs.last().unwrap().byte_end, text.len());
        // "Dr. Popescu" is not cut after "Dr."
        assert!(!cs.iter().any(|c| c.text.ends_with("Dr.")));
        // no overlap: chunks tile the text without sharing sentences
        let flat = chunk_text(text, &ChunkOptions { max_chars: 60, overlap_chars: 0 });
        for w in flat.windows(2) { assert!(w[1].byte_start >= w[0].byte_end); }
        // overlap really repeats text
        assert!(cs.windows(2).any(|w| w[1].byte_start < w[0].byte_end));
    }

    #[test]
    fn chunking_paragraphs_and_edge_cases() {
        assert!(chunk_text("", &ChunkOptions::default()).is_empty());
        assert!(chunk_text("  \n\t\n ", &ChunkOptions::default()).is_empty());
        let one = chunk_text("Doar o propoziție.", &ChunkOptions::default());
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].text, "Doar o propoziție.");
        // paragraphs split even without punctuation
        let two = chunk_text("primul rând fără punct\n\nal doilea rând", &ChunkOptions { max_chars: 24, overlap_chars: 0 });
        assert_eq!(two.len(), 2);
        // a very long word is hard-cut on char boundaries (multi-byte letters)
        let long = "ș".repeat(500);
        let cs = chunk_text(&long, &ChunkOptions { max_chars: 100, overlap_chars: 10 });
        assert!(cs.len() >= 5);
        for c in &cs { assert_eq!(&long[c.byte_start..c.byte_end], c.text); assert!(c.text.chars().count() <= 100); }
        // tiny limits are clamped and still terminate
        let cs = chunk_text("a b c d e f g h i j k l m n o p q r s t u v w x y z", &ChunkOptions { max_chars: 0, overlap_chars: 1000 });
        assert!(!cs.is_empty() && cs.len() < 200);
        // quotes after the terminator stay with the sentence
        let q = chunk_text("El a spus: „Vin acum!” Apoi a plecat.", &ChunkOptions { max_chars: 24, overlap_chars: 0 });
        assert!(q[0].text.ends_with('”'), "{:?}", q);
    }

    #[test]
    fn chunking_hostile_input_terminates() {
        let mut s = 5u64;
        for _ in 0..200 {
            let n = (splitmix(&mut s) % 300) as usize;
            let text: String = (0..n).map(|_| ['a', ' ', '.', '\n', 'Ș', '!', '“', 'é', '…', ' '][(splitmix(&mut s) % 10) as usize]).collect();
            let opts = ChunkOptions { max_chars: (splitmix(&mut s) % 80) as usize, overlap_chars: (splitmix(&mut s) % 90) as usize };
            for c in chunk_text(&text, &opts) { assert_eq!(&text[c.byte_start..c.byte_end], c.text); assert!(!c.text.is_empty()); }
        }
    }

    #[test]
    fn fold_tokens_matches_diacritics() {
        assert_eq!(fold_tokens("Țara Șoarecilor, ŞI ţărâna!"), vec!["tara", "soarecilor", "si", "tarana"]);
        assert_eq!(fold_tokens("Straße  Café—ok"), vec!["strasse", "cafe", "ok"]);
        assert!(fold_tokens("...  ").is_empty());
        assert_eq!(fold_tokens("a\u{0301}b"), vec!["ab"]);
        assert_eq!(fold_tokens(&"x".repeat(1000))[0].len(), 64);
    }

    #[test]
    fn hash_embedder_is_deterministic_and_lexical() {
        let e = HashEmbedder::new(512, 1);
        let a = e.embed("Pisica doarme pe canapea în sufragerie").unwrap();
        assert_eq!(a, e.embed("Pisica doarme pe canapea în sufragerie").unwrap());
        assert!((dot(&a, &a) - 1.0).abs() < 1e-4);
        let same_no_diacritics = e.embed("pisica doarme pe canapea in sufragerie").unwrap();
        assert!(dot(&a, &same_no_diacritics) > 0.999);
        let typo = e.embed("Pisica doarme pe canapeaa în sufragerie").unwrap();
        let other = e.embed("Parlamentul a votat bugetul pentru anul viitor").unwrap();
        assert!(dot(&a, &typo) > dot(&a, &other) + 0.4);
        assert_ne!(a, HashEmbedder::new(512, 2).embed("Pisica doarme pe canapea în sufragerie").unwrap());
        assert_eq!(e.embed("   "), Err(RagError::Empty));
        assert_eq!(e.embed("!!! ???"), Err(RagError::Empty));
        let big = "cuvânt ".repeat(50_000);
        assert!(e.embed(&big).is_ok());
    }

    #[test]
    fn pooling_and_matryoshka() {
        let rows = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 tokens, width 2
        assert_eq!(pool(&rows, 2, Pooling::Cls), vec![1.0, 2.0]);
        assert_eq!(pool(&rows, 2, Pooling::Last), vec![5.0, 6.0]);
        assert_eq!(pool(&rows, 2, Pooling::Mean), vec![3.0, 4.0]);
        assert!(pool(&rows, 0, Pooling::Mean).is_empty());
        assert!(pool(&[1.0], 2, Pooling::Mean).is_empty());
        let m = matryoshka(&[3.0, 4.0, 12.0], 2);
        assert_eq!(m.len(), 2);
        assert!((m[0] - 0.6).abs() < 1e-6 && (m[1] - 0.8).abs() < 1e-6);
        assert_eq!(matryoshka(&[1.0, 0.0], 10).len(), 2);
        assert_eq!(matryoshka(&[1.0, 0.0], 0).len(), 1);
        let mut z = vec![0.0; 3];
        l2_normalize(&mut z);
        assert_eq!(z, vec![0.0; 3]);
    }

    #[test]
    fn model_embedder_on_a_synthetic_model() {
        let m = crate::model::tests::tiny(4, 2, true);
        let vocab = m.spec.vocab as usize;
        let tok = move |t: &str| t.bytes().map(|b| b as usize % vocab).collect::<Vec<_>>();
        let mut e = ModelEmbedder::with_tokenizer(&m, tok, Pooling::Mean);
        let h = m.spec.hidden as usize;
        assert_eq!(e.dim(), h);
        let a = e.embed("hello world").unwrap();
        assert_eq!(a.len(), h);
        assert!((dot(&a, &a) - 1.0).abs() < 1e-4);
        assert_eq!(a, e.embed("hello world").unwrap());
        assert_ne!(a, e.embed("something else entirely").unwrap());
        for p in [Pooling::Last, Pooling::Cls] {
            e.pooling = p;
            let v = e.embed("hello world").unwrap();
            assert_eq!(v.len(), h);
            assert!(v.iter().all(|x| x.is_finite()));
        }
        e.pooling = Pooling::Mean;
        e.matryoshka = Some(4);
        let t = e.embed("hello world").unwrap();
        assert_eq!(t.len(), 4);
        assert_eq!(e.dim(), 4);
        assert!((dot(&t, &t) - 1.0).abs() < 1e-4);
        e.normalize = false;
        assert!((dot(&e.embed("hello world").unwrap(), &t) - 1.0).abs() > -1.0);
        assert_eq!(e.embed(""), Err(RagError::Empty));
        // overly long text is cut, not an error
        e.max_tokens = 8;
        assert!(e.embed(&"abc ".repeat(1000)).is_ok());
        // out-of-vocabulary token ids surface as errors, not panics
        let bad = ModelEmbedder::with_tokenizer(&m, |_| vec![usize::MAX], Pooling::Mean);
        assert!(matches!(bad.embed("x"), Err(RagError::Model(_))));
        // a model embedder plugs into the knowledge base
        let mut kb = KnowledgeBase::new(Box::new(ModelEmbedder::with_tokenizer(&m, move |t: &str| t.bytes().map(|b| b as usize % vocab).collect(), Pooling::Mean)), HnswParams::default()).unwrap();
        kb.add_document("d", "Primul text. Al doilea text.", &ChunkOptions::default()).unwrap();
        assert!(!kb.search("text", &SearchOptions::default()).unwrap().is_empty());
    }

    #[test]
    fn bm25_ranks_and_folds() {
        let mut b = Bm25Index::new();
        b.add(1, "Pisica neagră doarme pe canapea");
        b.add(2, "Câinele aleargă prin parc după minge");
        b.add(3, "Pisica și câinele se joacă împreună, pisica sare");
        b.add(4, "Parlamentul a adoptat legea bugetului");
        let r = b.search("pisica", 10);
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), vec![3, 1]);
        // diacritics and case do not matter
        assert_eq!(b.search("PISICĂ", 10).iter().map(|x| x.0).collect::<Vec<_>>(), vec![3, 1]);
        assert_eq!(b.search("canapea ȘI", 10)[0].0, 1);
        // stop words and prefix stemming (opt-in)
        let mut st = Bm25Index::with_options(Some(5), true);
        st.add(1, "Sarmalele se fac din carne tocată");
        st.add(2, "Pisicile se spală singure cu limba");
        assert_eq!(st.search("cum se face sarmaua", 5).iter().map(|x| x.0).collect::<Vec<_>>(), vec![1]);
        assert!(st.search("se cu", 5).is_empty(), "a query of only stop words matches nothing");
        assert_eq!(b.search("cainele", 10).len(), 2);
        assert!(b.search("zzz", 10).is_empty());
        assert!(b.search("", 10).is_empty());
        assert!(b.search("pisica", 0).is_empty());
        // rare words outweigh common ones
        let r = b.search("pisica parlamentul", 10);
        assert_eq!(r[0].0, 4);
        // removal and replacement
        assert!(b.remove(1));
        assert!(!b.remove(1));
        assert_eq!(b.search("pisica", 10).iter().map(|x| x.0).collect::<Vec<_>>(), vec![3]);
        b.add(3, "altceva complet");
        assert!(b.search("pisica", 10).is_empty());
        assert_eq!(b.len(), 3);
        // duplicate query words count once
        assert_eq!(b.search("parc parc parc", 10), b.search("parc", 10));
    }

    #[test]
    fn rrf_fuses_and_is_stable() {
        let a = [1u64, 2, 3];
        let b = [3u64, 2, 9];
        let f = rrf(&[(&a, 1.0), (&b, 1.0)], 60.0);
        assert_eq!(f[0].0, 3); // 1/61 + 1/63 just beats 1/62 + 1/62
        assert_eq!(f[1].0, 2);
        assert_eq!(f.len(), 4);
        let w = rrf(&[(&a, 1.0), (&b, 0.0)], 60.0);
        assert_eq!(w[0].0, 1);
        assert!(rrf(&[], 60.0).is_empty());
        assert_eq!(rrf(&[(&a, 1.0), (&b, 1.0)], 60.0), f);
    }

    fn corpus() -> Vec<(&'static str, &'static str)> {
        vec![
            ("pisici", "Pisica domestică este un animal carnivor. Ea doarme mult și vânează șoareci. Pisicile se spală singure cu limba."),
            ("cosmos", "Soarele este o stea din galaxia noastră. Planetele se rotesc în jurul Soarelui. Jupiter este cea mai mare planetă."),
            ("bucătărie", "Sarmalele se fac din carne tocată și varză murată. Se fierb la foc mic câteva ore. Se servesc cu smântână și mămăligă."),
            ("legi", "Directiva 2005/36/CE reglementează recunoașterea calificărilor profesionale. Autoritățile competente analizează dosarele candidaților."),
        ]
    }

    #[test]
    fn knowledge_base_hybrid_search_and_citations() {
        let mut kb = KnowledgeBase::new(Box::new(HashEmbedder::new(1024, 3)), HnswParams::default()).unwrap();
        let opts = ChunkOptions { max_chars: 90, overlap_chars: 20 };
        let mut docs = HashMap::new();
        let full: HashMap<&str, &str> = corpus().into_iter().collect();
        for (n, t) in corpus() { docs.insert(kb.add_document(n, t, &opts).unwrap(), n); }
        assert_eq!(kb.doc_count(), 4);
        assert!(kb.chunk_count() >= 8);
        let hits = kb.search("cum se prepară sarmale cu varză", &SearchOptions { k: 3, ..Default::default() }).unwrap();
        assert_eq!(hits[0].doc_name, "bucătărie");
        assert!(hits[0].keyword_score.is_some() && hits[0].vector_score.is_some());
        let hits = kb.search("calificari profesionale directiva", &SearchOptions::default()).unwrap();
        assert_eq!(hits[0].doc_name, "legi");
        for h in &hits { assert_eq!(&full[h.doc_name.as_str()][h.byte_start..h.byte_end], h.text); }
        // citations map back to the document
        let ctx = kb.retrieve_cited("planetele se rotesc în jurul Soarelui", &SearchOptions { k: 2, ..Default::default() }).unwrap();
        assert_eq!(ctx.sources[0].tag, 1);
        assert!(ctx.prompt_block().starts_with("[S1] (cosmos)"));
        let idx = ctx.quote_index(QuoteOptions::default()).unwrap();
        assert_eq!(idx.source_count(), ctx.sources.len());
        let quote = "Planetele se rotesc în jurul Soarelui";
        let cit = idx.find(quote).expect("quote is verbatim in a source (case-insensitive)");
        let span = ctx.locate(&cit).unwrap();
        let doc = full[span.doc_name.as_str()];
        assert_eq!(doc[span.byte_start..span.byte_end].to_lowercase(), quote.to_lowercase());
        // a citation that does not fit is refused
        assert!(ctx.locate(&Citation { source_id: 0, start: 0, end: 1, text: String::new() }).is_none());
        assert!(ctx.locate(&Citation { source_id: 99, start: 0, end: 1, text: String::new() }).is_none());
        assert!(ctx.locate(&Citation { source_id: 1, start: 5, end: 3, text: String::new() }).is_none());
        assert!(ctx.locate(&Citation { source_id: 1, start: 0, end: 100_000, text: String::new() }).is_none());
        // removing a document removes it from every ranker
        let cosmos = *docs.iter().find(|(_, n)| **n == "cosmos").unwrap().0;
        kb.remove_document(cosmos).unwrap();
        assert_eq!(kb.remove_document(cosmos), Err(RagError::Unknown(cosmos)));
        let hits = kb.search("Jupiter planeta Soarele", &SearchOptions::default()).unwrap();
        assert!(hits.iter().all(|h| h.doc_name != "cosmos"));
        // empty and hostile queries are not errors
        assert!(kb.search("", &SearchOptions::default()).unwrap().is_empty());
        assert!(kb.search("?!", &SearchOptions::default()).unwrap().is_empty());
        let empty = KnowledgeBase::new(Box::new(HashEmbedder::new(64, 0)), HnswParams::default()).unwrap();
        assert!(empty.search("orice", &SearchOptions::default()).unwrap().is_empty());
        // an empty document is fine
        kb.add_document("gol", "   ", &opts).unwrap();
    }

    #[test]
    fn knowledge_base_lock_integration_is_verbatim() {
        let mut kb = KnowledgeBase::new(Box::new(HashEmbedder::new(256, 9)), HnswParams::default()).unwrap();
        for (n, t) in corpus() { kb.add_document(n, t, &ChunkOptions::default()).unwrap(); }
        let ctx = kb.retrieve_cited("pisica doarme", &SearchOptions { k: 1, ..Default::default() }).unwrap();
        let answer = format!("Răspuns: [S1]«{}»", ctx.sources[0].text);
        let rep = crate::quotelock::verify_text(&answer, &ctx.texts()).unwrap();
        assert_eq!(rep.quotes.len(), 1);
        assert!(rep.all_verbatim());
        let forged = answer.replace("doarme", "zboară");
        assert!(!crate::quotelock::verify_text(&forged, &ctx.texts()).unwrap().all_verbatim());
    }
}

#[cfg(test)]
mod persist_tests {
    use super::*;

    fn kb<'a>() -> KnowledgeBase<'a> { KnowledgeBase::new(Box::new(HashEmbedder::new(64, 7)), HnswParams::default()).unwrap() }
    fn small() -> ChunkOptions { ChunkOptions { max_chars: 120, overlap_chars: 20 } }
    fn unhex(s: &str) -> Vec<u8> { (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect() }
    fn so() -> SearchOptions { SearchOptions::default() }

    fn filled<'a>() -> KnowledgeBase<'a> {
        let mut k = kb();
        k.add_document("cats.txt", "Cats are small carnivorous mammals. They purr when content and hunt mice at night. A cat sleeps most of the day.", &small()).unwrap();
        k.add_document("rust.md", "Rust is a systems language with ownership and borrowing. The borrow checker prevents data races at compile time.", &small()).unwrap();
        k.add_document("sea.txt", "The ocean covers most of the planet. Tides follow the moon and whales sing across the deep water.", &small()).unwrap();
        k
    }

    fn names(k: &KnowledgeBase, q: &str) -> Vec<(u64, String)> {
        k.search(q, &so()).unwrap().iter().map(|h| (h.chunk_id, h.text.clone())).collect()
    }

    #[test]
    fn bm25_round_trip_gives_identical_rankings() {
        let mut b = Bm25Index::new();
        for (i, t) in ["red apple pie", "green apple tart", "blue ocean wave", "apple apple apple"].iter().enumerate() { b.add(i as u64 + 1, t); }
        b.remove(2);
        let r = Bm25Index::from_bytes(&b.to_bytes()).unwrap();
        assert_eq!(r.len(), b.len());
        for q in ["apple", "ocean", "pie tart", "nothing"] { assert_eq!(r.search(q, 5), b.search(q, 5), "{q}"); }
        assert_eq!(r.ids(), b.ids());
    }

    #[test]
    fn bm25_bytes_reject_truncation_and_flips() {
        let mut b = Bm25Index::new();
        for i in 0..20u64 { b.add(i + 1, &format!("word{} common text number {}", i, i % 3)); }
        let bytes = b.to_bytes();
        for n in 0..bytes.len() { let _ = Bm25Index::from_bytes(&bytes[..n]); }
        assert!(Bm25Index::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut s = 5u64;
        for _ in 0..300 {
            let mut m = bytes.clone();
            let p = (splitmix(&mut s) as usize) % m.len();
            m[p] ^= 1 << (splitmix(&mut s) % 8);
            let _ = Bm25Index::from_bytes(&m); // never panics
        }
    }

    #[test]
    fn knowledge_base_round_trip_searches_the_same() {
        let k = filled();
        let bytes = k.to_bytes();
        let r = KnowledgeBase::from_bytes(&bytes, Box::new(HashEmbedder::new(64, 7))).unwrap();
        assert_eq!((r.doc_count(), r.chunk_count()), (k.doc_count(), k.chunk_count()));
        for q in ["cat purr", "borrow checker", "whales ocean", "moon tides", "zzz unknown"] { assert_eq!(names(&r, q), names(&k, q), "{q}"); }
        assert_eq!(r.documents(), k.documents());
        assert_eq!(r.to_bytes(), bytes, "saving what was loaded gives the same bytes");
    }

    #[test]
    fn save_and_load_through_a_file_and_continue_adding() {
        let dir = std::env::temp_dir().join(format!("corbul-kb-{}-{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("kb.bin");
        let k = filled();
        k.save(&path).unwrap();
        let mut r = KnowledgeBase::load(&path, Box::new(HashEmbedder::new(64, 7))).unwrap();
        let id = r.add_document("new.txt", "Volcanoes erupt molten rock and ash.", &small()).unwrap();
        let hit = r.search("volcano ash", &so()).unwrap();
        assert!(hit.iter().any(|h| h.doc_id == id));
        assert!(r.save(&path).is_ok(), "overwriting works");
        assert!(KnowledgeBase::load(&path, Box::new(HashEmbedder::new(64, 7))).unwrap().doc_count() == 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn removed_documents_stay_removed_after_a_round_trip_and_compaction() {
        let mut k = filled();
        let rust = k.find_document("rust.md").unwrap();
        k.remove_document(rust).unwrap();
        assert!(k.vector_index().tombstones() > 0);
        let r = KnowledgeBase::from_bytes(&k.to_bytes(), Box::new(HashEmbedder::new(64, 7))).unwrap();
        assert_eq!(names(&r, "borrow checker ownership"), names(&k, "borrow checker ownership"));
        assert!(r.search("borrow checker ownership", &so()).unwrap().iter().all(|h| h.doc_id != rust));
        let mut c = r;
        assert!(c.compact().unwrap() > 0);
        assert_eq!(c.vector_index().tombstones(), 0);
        let c2 = KnowledgeBase::from_bytes(&c.to_bytes(), Box::new(HashEmbedder::new(64, 7))).unwrap();
        assert_eq!(c2.chunk_count(), k.chunk_count());
    }

    #[test]
    fn a_different_embedder_is_refused() {
        let bytes = filled().to_bytes();
        assert!(KnowledgeBase::from_bytes(&bytes, Box::new(HashEmbedder::new(64, 8))).is_err(), "other seed");
        assert!(KnowledgeBase::from_bytes(&bytes, Box::new(HashEmbedder::new(32, 7))).is_err(), "other dimension");
    }

    #[test]
    fn every_truncation_is_an_error_and_nothing_panics() {
        let bytes = filled().to_bytes();
        for n in 0..bytes.len() {
            assert!(KnowledgeBase::from_bytes(&bytes[..n], Box::new(HashEmbedder::new(64, 7))).is_err(), "prefix {n}");
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(KnowledgeBase::from_bytes(&longer, Box::new(HashEmbedder::new(64, 7))).is_err(), "trailing byte");
    }

    #[test]
    fn every_bit_flip_is_caught_by_the_checksum() {
        let bytes = filled().to_bytes();
        let mut s = 99u64;
        for i in 0..bytes.len().min(400) {
            let mut m = bytes.clone();
            let p = if i < 200 { i } else { (splitmix(&mut s) as usize) % m.len() };
            m[p] ^= 1 << (splitmix(&mut s) % 8);
            assert!(KnowledgeBase::from_bytes(&m, Box::new(HashEmbedder::new(64, 7))).is_err(), "flip at {p}");
        }
    }

    #[test]
    fn forged_files_with_a_valid_checksum_never_panic() {
        let bytes = filled().to_bytes();
        let body = bytes.len() - 32;
        let mut s = 1234u64;
        let (mut accepted, mut rejected) = (0, 0);
        for round in 0..1500 {
            let mut m = bytes[..body].to_vec();
            for _ in 0..1 + round % 3 {
                let p = (splitmix(&mut s) as usize) % m.len();
                m[p] = match splitmix(&mut s) % 3 { 0 => 0xff, 1 => 0, _ => splitmix(&mut s) as u8 };
            }
            let sum = sha256(&m);
            m.extend_from_slice(&sum);
            match KnowledgeBase::from_bytes(&m, Box::new(HashEmbedder::new(64, 7))) {
                Ok(k) => { accepted += 1; let _ = k.search("cats", &so()); let _ = k.to_bytes(); }
                Err(_) => rejected += 1,
            }
        }
        assert!(rejected > 0, "some forgeries must be refused (accepted {accepted})");
    }

    #[test]
    fn huge_declared_counts_do_not_allocate() {
        // magic + version + garbage lengths, with a correct checksum
        let mut m = b"CORKB001".to_vec();
        m.extend_from_slice(&[0xff; 64]);
        let sum = sha256(&m);
        m.extend_from_slice(&sum);
        assert!(KnowledgeBase::from_bytes(&m, Box::new(HashEmbedder::new(64, 7))).is_err());
        assert!(KnowledgeBase::from_bytes(b"", Box::new(HashEmbedder::new(64, 7))).is_err());
        assert!(KnowledgeBase::from_bytes(b"not a knowledge base at all, just text........", Box::new(HashEmbedder::new(64, 7))).is_err());
    }

    #[test]
    fn upsert_adds_replaces_and_skips_unchanged() {
        let mut k = kb();
        let a = match k.upsert_document("a.txt", "alpha beta gamma delta", &small()).unwrap() { Upsert::Added(i) => i, o => panic!("{o:?}") };
        assert_eq!(k.upsert_document("a.txt", "alpha beta gamma delta", &small()).unwrap(), Upsert::Unchanged(a));
        let before = k.chunk_count();
        let new = match k.upsert_document("a.txt", "completely different words here", &small()).unwrap() {
            Upsert::Replaced { old, new } => { assert_eq!(old, a); new }
            o => panic!("{o:?}"),
        };
        assert_eq!(k.doc_count(), 1);
        assert_eq!(k.find_document("a.txt"), Some(new));
        assert!(before >= 1 && k.chunk_count() >= 1);
        assert!(k.search("alpha gamma", &so()).unwrap().iter().all(|h| h.doc_id == new));
    }

    // ── inflate ──

    #[test]
    fn inflate_stored_fixed_and_dynamic_blocks() {
        let fixed = unhex("78dacb48cdc9c957c84022cbf38b725200687d08c5");
        assert_eq!(zlib_decompress(&fixed, 1 << 20).unwrap(), b"hello hello hello world");
        let stored = unhex("7801011100eeff73746f72656420626c6f636b20746578743c3406a2");
        assert_eq!(zlib_decompress(&stored, 1 << 20).unwrap(), b"stored block text");
        let dynamic = unhex("78dabdcb471682300045d1ad7c37e0b197a1bd2b0ad65980000142203461f5661338beef191e4592332b80294519c1115ff8398f5388824a648a435257b085db86d158ac11d5f10aa68a4a967970584115d53442c8925c48f5ba690bb3f962b95a6fb6bbfde1783a5fb4eb4d37ee8fe7ebfdf997a0d3edf507c3d1783205312d9b3aaec7fc20e49188d110fe004bd38a4f");
        let out = zlib_decompress(&dynamic, 1 << 20).unwrap();
        assert_eq!(out.len(), 422);
        assert!(out.starts_with(b"The quick brown fox jumps over the lazy dog. The quick"));
        assert!(out.ends_with(b"abcdefghijklmnop "));
    }

    #[test]
    fn inflate_limit_truncation_and_garbage() {
        let fixed = unhex("78dacb48cdc9c957c84022cbf38b725200687d08c5");
        assert!(zlib_decompress(&fixed, 5).is_err(), "limit");
        for n in 0..fixed.len() - 1 { assert!(zlib_decompress(&fixed[..n], 1 << 20).is_err() || n >= fixed.len() - 5, "prefix {n}"); }
        let dynamic = unhex("78dabdcb471682300045d1ad7c37e0b197a1bd2b0ad65980000142203461f5661338beef191e4592332b80294519c1115ff8398f5388824a648a435257b085db86d158ac11d5f10aa68a4a967970584115d53442c8925c48f5ba690bb3f962b95a6fb6bbfde1783a5fb4eb4d37ee8fe7ebfdf997a0d3edf507c3d1783205312d9b3aaec7fc20e49188d110fe004bd38a4f");
        let mut s = 3u64;
        for _ in 0..2000 {
            let mut m = dynamic.clone();
            for _ in 0..1 + (splitmix(&mut s) % 4) { let p = 2 + (splitmix(&mut s) as usize) % (m.len() - 2); m[p] = splitmix(&mut s) as u8; }
            let _ = zlib_decompress(&m, 4096); // no panic, bounded output
        }
        // reserved block type, and a stored block whose length check fails
        assert!(inflate(&[0b0000_0111], 100).is_err());
        assert!(inflate(&[0x01, 0x05, 0x00, 0x00, 0x00, 1, 2, 3, 4, 5], 100).is_err());
        // a bomb: 1 MiB of zeros from a few hundred bytes must stop at the limit
        let mut bomb = vec![0x78u8, 0x9c, 0x63, 0x60, 0x18, 0x05, 0xa3, 0x60, 0x14, 0x0c, 0x77, 0x00, 0x00];
        bomb.extend_from_slice(&[0, 0, 0, 0]);
        assert!(zlib_decompress(&bomb, 1000).map(|v| v.len() <= 1000).unwrap_or(true));
    }

    // ── html ──

    #[test]
    fn html_is_stripped_and_entities_decoded() {
        let h = "<html><head><title>T</title><style>p{color:red}</style><script>var a='<p>no</p>';</script></head>\
                 <body><!-- hidden --><h1>Titlu &amp; Co</h1><p>Prima   linie<br>a doua &lt;b&gt; &#x21; &#33; &abreve;&scedil;&tcedil;</p>\
                 <a href=\"x>y\">link</a> 5 < 6 &unknown; &</body></html>";
        let t = html_to_text(h);
        assert!(!t.contains("color") && !t.contains("var a") && !t.contains("hidden") && !t.contains("<p>"), "{t}");
        assert!(t.contains("Titlu & Co"), "{t}");
        assert!(t.contains("Prima linie\na doua <b> ! ! ăşţ"), "{t}");
        assert!(t.contains("link 5 < 6 &unknown; &"), "{t}");
        assert!(!t.contains("\n\n\n") && !t.starts_with('\n'));
        // hostile: unterminated everything
        for s in ["<", "<a", "<a href=\"", "<!--", "<script>", "&#99999999999;", "&#xD800;", "<p>\u{1F600}</p", "<style>x"] { let _ = html_to_text(s); }
    }

    // ── pdf ──

    fn pdf_with(objs: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut p = b"%PDF-1.4\n".to_vec();
        for (i, (dict, body)) in objs.iter().enumerate() {
            p.extend_from_slice(format!("{} 0 obj\n<< {} /Length {} >>\nstream\n", i + 1, dict, body.len()).as_bytes());
            p.extend_from_slice(body);
            p.extend_from_slice(b"\nendstream\nendobj\n");
        }
        p.extend_from_slice(b"trailer\n<< /Size 9 >>\n%%EOF\n");
        p
    }

    #[test]
    fn plain_and_flate_pdf_text() {
        let plain = pdf_with(&[("", b"BT /F1 12 Tf 72 700 Td (Hello \\(PDF\\) world \\351t\\351) Tj 0 -14 Td [(Second ) -300 (line)] TJ ET".to_vec())]);
        let t = pdf_to_text(&plain).unwrap();
        assert!(t.contains("Hello (PDF) world été"), "{t}");
        assert!(t.contains("Second line"), "{t}");
        assert!(t.find("Hello").unwrap() < t.find("Second").unwrap());
        let flate = pdf_with(&[("/Filter /FlateDecode", unhex("789c15cab10a83301440d15fb96332485fb4e02e1aa453c1b7950ea58960792620057fbff4cc67502e31105a74a56fe945d0849bb359e53e46ce7a58623dea4eb4d7377bf483d084eb3f3edc92dfb5243c4d2782b3ad64ff446f4cfa03871c1926"))]);
        let t = pdf_to_text(&flate).unwrap();
        assert!(t.contains("Hello PDF world from Flate"), "{t}");
    }

    #[test]
    fn pdf_refusals_are_honest() {
        assert!(pdf_to_text(b"not a pdf").is_err());
        let enc = pdf_with(&[("/Encrypt 1", b"BT (Secret text that is long enough) Tj ET".to_vec())]);
        assert!(pdf_to_text(&enc).unwrap_err().to_string().contains("encrypted"));
        let none = pdf_with(&[("/Subtype /Image", vec![1, 2, 3])]);
        assert!(pdf_to_text(&none).is_err());
        for n in 0..60 { let _ = pdf_to_text(&enc[..n.min(enc.len())]); }
    }

    // ── files ──

    #[test]
    fn text_decoding_and_offsets() {
        let x = extract_text(DocKind::Text, "plain ăș text".as_bytes()).unwrap();
        assert!(x.exact_offsets && x.text == "plain ăș text");
        let bom = extract_text(DocKind::Markdown, b"\xef\xbb\xbfhi").unwrap();
        assert!(!bom.exact_offsets && bom.text == "hi");
        let l1 = extract_text(DocKind::Text, b"caf\xe9").unwrap();
        assert!(!l1.exact_offsets && l1.text == "café");
        let u16 = extract_text(DocKind::Text, &[0xff, 0xfe, b'h', 0, b'i', 0]).unwrap();
        assert_eq!(u16.text, "hi");
        let h = extract_text(DocKind::Html, b"<p>x</p>").unwrap();
        assert!(!h.exact_offsets);
        assert_eq!(DocKind::of_path(Path::new("a/B.MD")), Some(DocKind::Markdown));
        assert_eq!(DocKind::of_path(Path::new("a.exe")), None);
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("corbul-ing-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn directory_ingestion_offsets_limits_and_reingest() {
        let d = tmp("dir");
        std::fs::create_dir_all(d.join("sub/deeper")).unwrap();
        std::fs::create_dir_all(d.join(".hidden")).unwrap();
        let a = "Alpha document about lighthouses and the sea. ".repeat(12);
        std::fs::write(d.join("a.txt"), &a).unwrap();
        std::fs::write(d.join("sub/b.md"), "# Titlu\n\nRust ownership and borrowing explained in detail for beginners.").unwrap();
        std::fs::write(d.join("sub/deeper/c.html"), "<html><body><p>Volcanoes erupt ash and molten rock.</p><script>x()</script></body></html>").unwrap();
        std::fs::write(d.join("sub/p.pdf"), pdf_with(&[("", b"BT (A simple pdf page about astronomy and telescopes) Tj ET".to_vec())])).unwrap();
        std::fs::write(d.join("sub/scan.pdf"), pdf_with(&[("/Subtype /Image", vec![0; 10])])).unwrap();
        std::fs::write(d.join(".hidden/h.txt"), "hidden file text that is ignored").unwrap();
        std::fs::write(d.join("note.bin"), "ignored extension").unwrap();
        std::fs::write(d.join("empty.txt"), "").unwrap();
        std::fs::write(d.join("big.txt"), vec![b'x'; 5000]).unwrap();
        #[cfg(unix)] std::os::unix::fs::symlink(d.join("a.txt"), d.join("link.txt")).unwrap();

        let mut o = IngestOptions { chunk: small(), max_file_bytes: 4000, ..IngestOptions::default() };
        let (files, skipped) = chunk_dir(&d, &o).unwrap();
        let got: Vec<&str> = files.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(got, ["a.txt", "sub/b.md", "sub/deeper/c.html", "sub/p.pdf"], "{skipped:?}");
        for n in ["big.txt", "empty.txt", "sub/scan.pdf"] { assert!(skipped.iter().any(|s| s.name == n), "{n} in {skipped:?}"); }
        #[cfg(unix)] assert!(skipped.iter().any(|s| s.name == "link.txt" && s.reason.contains("symbolic")));
        // offsets are real offsets into the text the chunks came from (and into the file for exact ones)
        for f in &files {
            for c in &f.chunks { assert_eq!(&f.text[c.byte_start..c.byte_end], c.text, "{}", f.name); }
        }
        let fa = &files[0];
        assert!(fa.exact_offsets);
        let raw = std::fs::read_to_string(d.join("a.txt")).unwrap();
        for c in &fa.chunks { assert_eq!(&raw[c.byte_start..c.byte_end], c.text); }
        assert!(!files[2].text.contains("x()"));

        // ingest, search, re-ingest unchanged, change one file, delete one, prune
        let mut k = kb();
        let r1 = k.ingest_dir(&d, &o).unwrap();
        assert_eq!(r1.added.len(), 4);
        assert!(k.search("volcanoes ash", &so()).unwrap().iter().any(|h| h.text.contains("Volcanoes")));
        let r2 = k.ingest_dir(&d, &o).unwrap();
        assert_eq!((r2.added.len(), r2.replaced.len(), r2.unchanged.len()), (0, 0, 4));
        std::fs::write(d.join("sub/b.md"), "# Titlu\n\nCompletely rewritten text on gardening and compost.").unwrap();
        std::fs::remove_file(d.join("sub/p.pdf")).unwrap();
        let r3 = k.ingest_dir(&d, &o).unwrap();
        assert_eq!(r3.replaced, ["sub/b.md"]);
        assert!(r3.removed.is_empty(), "no prune requested");
        o.prune = true;
        let r4 = k.ingest_dir(&d, &o).unwrap();
        assert_eq!(r4.removed, ["sub/p.pdf"]);
        assert_eq!(k.doc_count(), 3);
        assert!(k.search("gardening compost", &so()).unwrap().iter().any(|h| h.text.contains("gardening")));
        assert!(k.search("telescopes astronomy", &so()).unwrap().iter().all(|h| !h.text.contains("astronomy")));
        // and it survives a save / load
        let back = KnowledgeBase::from_bytes(&k.to_bytes(), Box::new(HashEmbedder::new(64, 7))).unwrap();
        assert_eq!(back.documents(), k.documents());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn depth_and_count_limits_and_missing_directory() {
        let d = tmp("lim");
        std::fs::create_dir_all(d.join("l1/l2")).unwrap();
        for p in ["x.txt", "l1/y.txt", "l1/l2/z.txt"] { std::fs::write(d.join(p), format!("some words in {p} to be indexed")).unwrap(); }
        let mut o = IngestOptions { max_depth: 1, ..IngestOptions::default() };
        let (f, s) = chunk_dir(&d, &o).unwrap();
        assert_eq!(f.len(), 2);
        assert!(s.iter().any(|x| x.name == "l1/l2"));
        o.max_depth = 16;
        o.max_files = 1;
        let (f, s) = chunk_dir(&d, &o).unwrap();
        assert_eq!(f.len(), 1);
        assert!(s.iter().any(|x| x.reason.contains("limit")));
        assert!(chunk_dir(&d.join("nope"), &o).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
