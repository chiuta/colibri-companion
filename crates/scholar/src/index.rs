// Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//! Storage and retrieval: documents, passages, BM25, optional caller-supplied dense vectors, rank fusion.
//!
//! The crate contains no embedding model: a caller that has one passes `Vec<f32>` per passage
//! (`Chunk::emb`) and a query vector to [`Searcher::search`]; the two rankings are merged with
//! Reciprocal Rank Fusion. The store is JSON on disk; the BM25 index is cached next to it in a
//! small bounds-checked binary file that is rebuilt whenever the library fingerprint changes.

use crate::phrase;
use crate::unitext;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Crate error: deliberately small, no external error crates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A file could not be read or written.
    Io(String),
    /// The data is malformed or from another version.
    Format(String),
    /// An input exceeded a safety limit (size, depth, count).
    Limit(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(s) => write!(f, "i/o error: {s}"),
            Error::Format(s) => write!(f, "format error: {s}"),
            Error::Limit(s) => write!(f, "limit exceeded: {s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

/// Result alias used across the retrieval modules.
pub type Result<T> = std::result::Result<T, Error>;

/// Largest store file `Store::load` will read (guards against a hostile or runaway file).
pub const MAX_STORE_BYTES: u64 = 2 << 30;
/// Most distinct query terms scored (a hostile multi-megabyte query costs no more than this).
const MAX_QUERY_TERMS: usize = 256;

/// One ingested document.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Doc {
    /// Path or identifier the document was ingested from.
    pub path: String,
    /// Display title.
    pub title: String,
    /// 64-bit content hash (FNV-1a), used for change detection.
    pub hash: u64,
    /// Number of pages (0 when the format has none).
    pub pages: usize,
}

/// One searchable passage.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Chunk {
    /// Index into `Store::docs`.
    pub doc: usize,
    /// 1-based page number, 0 when the format has no pages (txt, md, html).
    pub page: usize,
    /// Passage text.
    pub text: String,
    /// Optional caller-supplied embedding (L2-normalised by the caller, or via [`normalize`]).
    pub emb: Option<Vec<f32>>,
}

/// The library: documents and their passages.
#[derive(Serialize, Deserialize, Default, Debug)]
pub struct Store {
    /// Documents.
    pub docs: Vec<Doc>,
    /// Passages, each pointing at a document.
    pub chunks: Vec<Chunk>,
    /// Name of the model that produced `Chunk::emb`, if any (informational).
    pub embed_model: Option<String>,
    /// Where this store was loaded from; the BM25 cache lives next to it (`<store>.bm25`).
    #[serde(skip)]
    pub source: Option<PathBuf>,
    /// SHA-256 of each document's bytes at ingestion, keyed by path with the 64-bit hash it belongs
    /// to. Persisted in a sidecar file (`<store>.sha256`).
    #[serde(skip)]
    pub sha: HashMap<String, (u64, String)>,
}

fn sidecar(p: &Path, ext: &str) -> PathBuf {
    let mut o = p.as_os_str().to_owned();
    o.push(ext);
    o.into()
}

fn read_sha(p: &Path) -> HashMap<String, (u64, String)> {
    let mut out = HashMap::new();
    if let Ok(b) = std::fs::read(sidecar(p, ".sha256"))
        && let Ok(m) = serde_json::from_slice::<HashMap<String, (String, String)>>(&b)
    {
        for (k, (h, s)) in m {
            if let Ok(h) = u64::from_str_radix(&h, 16) {
                out.insert(k, (h, s));
            }
        }
    }
    out
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    // Unique temp name per process, flushed before the rename: a crash or a second writer can
    // never leave a half-written file.
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp)?;
        std::io::Write::write_all(&mut f, bytes)?;
        f.sync_all()?;
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}

impl Store {
    /// Load a store; a missing file gives an empty store that remembers the path.
    pub fn load(path: &Path) -> Result<Store> {
        if !path.exists() {
            return Ok(Store {
                source: Some(path.to_path_buf()),
                ..Store::default()
            });
        }
        let len = std::fs::metadata(path)?.len();
        if len > MAX_STORE_BYTES {
            return Err(Error::Limit(format!("store file is {len} bytes")));
        }
        let bytes = std::fs::read(path)?;
        let mut st: Store = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Format(format!("store is corrupt or from another version: {e}")))?;
        st.validate()?;
        st.source = Some(path.to_path_buf());
        st.sha = read_sha(path);
        Ok(st)
    }

    /// Structural check: every passage points at an existing document.
    pub fn validate(&self) -> Result<()> {
        if let Some(c) = self.chunks.iter().find(|c| c.doc >= self.docs.len()) {
            return Err(Error::Format(format!(
                "passage refers to missing document {}",
                c.doc
            )));
        }
        Ok(())
    }

    /// Write the store (and the SHA-256 sidecar) atomically.
    pub fn save(&self, path: &Path) -> Result<()> {
        let bytes = serde_json::to_vec(self).map_err(|e| Error::Format(e.to_string()))?;
        write_atomic(path, &bytes)?;
        let side = sidecar(path, ".sha256");
        if self.sha.is_empty() {
            let _ = std::fs::remove_file(&side);
        } else {
            let m: std::collections::BTreeMap<&String, (String, &String)> = self
                .sha
                .iter()
                .map(|(k, (h, s))| (k, (format!("{h:016x}"), s)))
                .collect();
            let b = serde_json::to_vec(&m).map_err(|e| Error::Format(e.to_string()))?;
            std::fs::write(&side, b)?;
        }
        Ok(())
    }

    /// The SHA-256 recorded when this document was ingested, if it matches the document as stored.
    pub fn doc_sha256(&self, id: usize) -> Option<&str> {
        let d = self.docs.get(id)?;
        match self.sha.get(&d.path) {
            Some((h, s)) if *h == d.hash => Some(s),
            _ => None,
        }
    }

    /// Index of the document with this path.
    pub fn doc_by_path(&self, p: &str) -> Option<usize> {
        self.docs.iter().position(|d| d.path == p)
    }

    /// Remove a document and its passages, keeping ids consistent. Unknown ids are ignored.
    pub fn remove_doc(&mut self, id: usize) {
        if id >= self.docs.len() {
            return;
        }
        let gone = self.docs.remove(id);
        self.sha.remove(&gone.path);
        self.chunks.retain(|c| c.doc != id);
        for c in &mut self.chunks {
            if c.doc > id {
                c.doc -= 1;
            }
        }
    }

    /// Fingerprint of the library: paths, hashes, token rules version, passage count and text size.
    /// Any edit changes it, which invalidates the BM25 caches.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut eat = |b: &[u8]| {
            for &x in b {
                h ^= x as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        for d in &self.docs {
            eat(d.path.as_bytes());
            eat(&[0]);
            eat(&d.hash.to_le_bytes());
        }
        eat(&unitext::RULES_VERSION.to_le_bytes());
        eat(&(self.chunks.len() as u64).to_le_bytes());
        eat(&(self.chunks.iter().map(|c| c.text.len()).sum::<usize>() as u64).to_le_bytes());
        h
    }
}

// ---------------------------------------------------------------- tokenizing

/// Index tokens of a text (Unicode folding and word segmentation, see [`unitext`]).
pub fn tokenize(s: &str) -> Vec<String> {
    unitext::token_strings(s)
}

// --------------------------------------------------------------------- BM25

const K1: f32 = 1.5;
const B: f32 = 0.75;
const MAGIC: &[u8; 4] = b"CSB1";

/// BM25 inverted index over the passages of a store.
#[derive(Debug, PartialEq)]
pub struct Bm25 {
    postings: HashMap<String, Vec<(u32, u32)>>,
    lens: Vec<u32>,
    avg: f32,
}

impl Bm25 {
    /// Tokenise every passage and build the index.
    pub fn new(chunks: &[Chunk]) -> Bm25 {
        let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        let mut lens = Vec::with_capacity(chunks.len());
        for (i, c) in chunks.iter().enumerate() {
            let toks = tokenize(&c.text);
            lens.push(toks.len().min(u32::MAX as usize) as u32);
            let mut tf: HashMap<&str, u32> = HashMap::new();
            for t in &toks {
                *tf.entry(t.as_str()).or_default() += 1;
            }
            for (t, n) in tf {
                postings
                    .entry(t.to_string())
                    .or_default()
                    .push((i as u32, n));
            }
        }
        let avg = if lens.is_empty() {
            1.0
        } else {
            (lens.iter().map(|&l| l as f64).sum::<f64>() / lens.len() as f64) as f32
        };
        Bm25 {
            postings,
            lens,
            avg: avg.max(1e-3),
        }
    }

    /// Top-`n` passage indices with BM25 scores, best first. `keep` filters candidates before ranking.
    pub fn search(&self, query: &str, n: usize, keep: &dyn Fn(usize) -> bool) -> Vec<(usize, f32)> {
        let big_n = self.lens.len() as f32;
        let mut acc: HashMap<u32, f32> = HashMap::new();
        let mut seen = HashSet::new();
        for t in tokenize(query) {
            if seen.len() >= MAX_QUERY_TERMS {
                break;
            }
            if !seen.insert(t.clone()) {
                continue;
            }
            let Some(plist) = self.postings.get(&t) else {
                continue;
            };
            let df = plist.len() as f32;
            let idf = ((big_n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(ci, tf) in plist {
                let tf = tf as f32;
                let dl = self.lens[ci as usize] as f32;
                let s = idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / self.avg));
                *acc.entry(ci).or_default() += s;
            }
        }
        let mut v: Vec<(usize, f32)> = acc
            .into_iter()
            .map(|(i, s)| (i as usize, s))
            .filter(|(i, _)| keep(*i))
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(n);
        v
    }

    /// Serialise with the library fingerprint `key`.
    pub fn to_bytes(&self, key: u64) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(MAGIC);
        o.extend_from_slice(&key.to_le_bytes());
        o.extend_from_slice(&self.avg.to_le_bytes());
        o.extend_from_slice(&(self.lens.len() as u32).to_le_bytes());
        for l in &self.lens {
            o.extend_from_slice(&l.to_le_bytes());
        }
        let mut terms: Vec<&String> = self.postings.keys().collect();
        terms.sort();
        o.extend_from_slice(&(terms.len() as u32).to_le_bytes());
        for t in terms {
            o.extend_from_slice(&(t.len() as u32).to_le_bytes());
            o.extend_from_slice(t.as_bytes());
            let p = &self.postings[t];
            o.extend_from_slice(&(p.len() as u32).to_le_bytes());
            for &(c, n) in p {
                o.extend_from_slice(&c.to_le_bytes());
                o.extend_from_slice(&n.to_le_bytes());
            }
        }
        o
    }

    /// Parse a cache file; `None` if it is malformed, truncated, for another fingerprint or
    /// internally inconsistent. Never panics and never allocates more than the input justifies.
    pub fn from_bytes(b: &[u8], key: u64) -> Option<Bm25> {
        struct R<'a>(&'a [u8]);
        impl<'a> R<'a> {
            fn take(&mut self, n: usize) -> Option<&'a [u8]> {
                if self.0.len() < n {
                    return None;
                }
                let (a, r) = self.0.split_at(n);
                self.0 = r;
                Some(a)
            }
            fn u32(&mut self) -> Option<u32> {
                Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
            }
        }
        let mut r = R(b);
        if r.take(4)? != MAGIC {
            return None;
        }
        if u64::from_le_bytes(r.take(8)?.try_into().ok()?) != key {
            return None;
        }
        let avg = f32::from_le_bytes(r.take(4)?.try_into().ok()?);
        if !avg.is_finite() || avg <= 0.0 {
            return None;
        }
        let nl = r.u32()? as usize;
        if nl.checked_mul(4)? > r.0.len() {
            return None;
        }
        let mut lens = Vec::with_capacity(nl);
        for _ in 0..nl {
            lens.push(r.u32()?);
        }
        let nt = r.u32()? as usize;
        if nt.checked_mul(8)? > r.0.len() {
            return None;
        }
        let mut postings = HashMap::with_capacity(nt);
        for _ in 0..nt {
            let tl = r.u32()? as usize;
            let t = String::from_utf8(r.take(tl)?.to_vec()).ok()?;
            let np = r.u32()? as usize;
            if np.checked_mul(8)? > r.0.len() {
                return None;
            }
            let mut p = Vec::with_capacity(np);
            for _ in 0..np {
                let c = r.u32()?;
                let n = r.u32()?;
                if c as usize >= nl {
                    return None;
                }
                p.push((c, n));
            }
            postings.insert(t, p);
        }
        if !r.0.is_empty() {
            return None;
        }
        Some(Bm25 {
            postings,
            lens,
            avg,
        })
    }
}

/// Fraction (0..1) of the query's distinct terms that occur in `text`. A cheap, honest relevance
/// signal: low values mean the passage barely matches what was asked.
pub fn coverage(query: &str, text: &str) -> f32 {
    let q: HashSet<String> = tokenize(query).into_iter().collect();
    if q.is_empty() {
        return 0.0;
    }
    let t: HashSet<String> = tokenize(text).into_iter().collect();
    q.iter().filter(|w| t.contains(*w)).count() as f32 / q.len() as f32
}

// ------------------------------------------------------------------- dense

/// L2-normalise a vector in place (zero and non-finite vectors are left alone).
pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 && n.is_finite() {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

fn dense_search(
    chunks: &[Chunk],
    q: &[f32],
    n: usize,
    keep: &dyn Fn(usize) -> bool,
) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = chunks
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            if !keep(i) {
                return None;
            }
            let e = c.emb.as_ref()?;
            if e.len() != q.len() {
                return None;
            }
            let s: f32 = e.iter().zip(q).map(|(a, b)| a * b).sum();
            s.is_finite().then_some((i, s))
        })
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(n);
    v
}

// ------------------------------------------------------------------ hybrid

/// One ranked passage.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    /// Index into `Store::chunks`.
    pub chunk: usize,
    /// Fused (RRF) score; only the order is meaningful.
    pub score: f32,
}

/// A search with its diagnostics (phrase-syntax warnings or an error that stopped the query).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// Ranked passages.
    pub hits: Vec<Hit>,
    /// Non-fatal notes about the query syntax.
    pub warnings: Vec<String>,
    /// Set when the query was not run (for example an unreadable phrase).
    pub error: Option<String>,
}

static BM25_CACHE: Mutex<Option<(u64, Arc<Bm25>)>> = Mutex::new(None);

/// Searches one [`Store`].
pub struct Searcher<'a> {
    /// The library being searched.
    pub store: &'a Store,
    bm25: Arc<Bm25>,
    /// Quoted phrases and `~k` proximity in the query are hard filters. Off by default (a quote
    /// mark is ordinary punctuation); switch on with [`Searcher::with_phrases`].
    phrases: bool,
}

impl<'a> Searcher<'a> {
    /// Build a searcher. Uses the in-memory cache, then the on-disk cache next to the store, and
    /// only then tokenises the library; a changed library fingerprint rebuilds both.
    pub fn new(store: &'a Store) -> Self {
        let key = store.fingerprint();
        let mut g = BM25_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((k, b)) = g.as_ref()
            && *k == key
        {
            return Self::with(store, b.clone());
        }
        let disk = store.source.as_ref().map(|p| sidecar(p, ".bm25"));
        if let Some(dp) = &disk
            && let Ok(bytes) = std::fs::read(dp)
            && let Some(b) = Bm25::from_bytes(&bytes, key)
            && b.lens.len() == store.chunks.len()
        {
            let b = Arc::new(b);
            *g = Some((key, b.clone()));
            return Self::with(store, b);
        }
        let b = Arc::new(Bm25::new(&store.chunks));
        if let Some(dp) = &disk {
            let _ = write_atomic(dp, &b.to_bytes(key));
        }
        *g = Some((key, b.clone()));
        Self::with(store, b)
    }

    fn with(store: &'a Store, bm25: Arc<Bm25>) -> Self {
        Searcher {
            store,
            bm25,
            phrases: false,
        }
    }

    /// Treat `"exact phrase"` and `"a b"~k` in queries as required phrases. Off by default.
    pub fn with_phrases(mut self, on: bool) -> Self {
        self.phrases = on;
        self
    }

    /// Reciprocal-rank fusion of BM25 and (optionally) dense retrieval. `max_per_doc` keeps one big
    /// book from crowding out everything else. `allow` restricts retrieval to those document ids.
    pub fn search(
        &self,
        query: &str,
        q_emb: Option<&[f32]>,
        k: usize,
        max_per_doc: usize,
        allow: Option<&HashSet<usize>>,
    ) -> Vec<Hit> {
        self.report(query, q_emb, k, max_per_doc, allow, self.phrases)
            .hits
    }

    /// As [`Searcher::search`] but the text is always plain terms: quotes and `~k` mean nothing.
    pub fn search_terms(
        &self,
        query: &str,
        q_emb: Option<&[f32]>,
        k: usize,
        max_per_doc: usize,
        allow: Option<&HashSet<usize>>,
    ) -> Vec<Hit> {
        self.report(query, q_emb, k, max_per_doc, allow, false).hits
    }

    /// As [`Searcher::search`] (phrase mode per `with_phrases`) but also returns warnings/errors
    /// instead of printing them.
    pub fn search_report(
        &self,
        query: &str,
        q_emb: Option<&[f32]>,
        k: usize,
        max_per_doc: usize,
        allow: Option<&HashSet<usize>>,
    ) -> Report {
        self.report(query, q_emb, k, max_per_doc, allow, self.phrases)
    }

    fn report(
        &self,
        query: &str,
        q_emb: Option<&[f32]>,
        k: usize,
        max_per_doc: usize,
        allow: Option<&HashSet<usize>>,
        phrases: bool,
    ) -> Report {
        const POOL: usize = 200;
        const RRF: f32 = 60.0;
        let chunks = &self.store.chunks;
        let parsed = if phrases {
            phrase::parse(query)
        } else {
            Default::default()
        };
        let mut rep = Report {
            warnings: parsed.warnings.clone(),
            ..Report::default()
        };
        if let Some(e) = &parsed.error {
            rep.error = Some(format!("query not run: {e}"));
            return rep;
        }
        // phrase syntax removed; phrase words still count as scoring terms
        let query = if phrases {
            parsed.scoring.as_str()
        } else {
            query
        };
        // passages must contain every phrase; decided once per passage, before ranking
        let verdicts: std::cell::RefCell<HashMap<usize, bool>> = Default::default();
        let phrase_ok = |ci: usize| -> bool {
            if parsed.phrases.is_empty() {
                return true;
            }
            *verdicts
                .borrow_mut()
                .entry(ci)
                .or_insert_with(|| phrase::passage_ok(&parsed.phrases, &chunks[ci].text))
        };
        let keep =
            |ci: usize| allow.map(|a| a.contains(&chunks[ci].doc)).unwrap_or(true) && phrase_ok(ci);
        let mut fused: HashMap<usize, f32> = HashMap::new();
        for (rank, (ci, _)) in self.bm25.search(query, POOL, &keep).into_iter().enumerate() {
            *fused.entry(ci).or_default() += 1.0 / (RRF + rank as f32 + 1.0);
        }
        if let Some(q) = q_emb {
            for (rank, (ci, _)) in dense_search(chunks, q, POOL, &keep).into_iter().enumerate() {
                *fused.entry(ci).or_default() += 1.0 / (RRF + rank as f32 + 1.0);
            }
        }
        let mut v: Vec<Hit> = fused
            .into_iter()
            .map(|(chunk, score)| Hit { chunk, score })
            .collect();
        v.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.chunk.cmp(&b.chunk)));
        let mut per_doc: HashMap<usize, usize> = HashMap::new();
        for h in v {
            if rep.hits.len() >= k {
                break;
            }
            let n = per_doc.entry(chunks[h.chunk].doc).or_default();
            if *n >= max_per_doc {
                continue;
            }
            *n += 1;
            rep.hits.push(h);
        }
        rep
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(texts: &[&str]) -> Store {
        let mut s = Store::default();
        s.docs.push(Doc {
            path: "a".into(),
            title: "A".into(),
            hash: 0,
            pages: 1,
        });
        for t in texts {
            s.chunks.push(Chunk {
                doc: 0,
                page: 1,
                text: t.to_string(),
                emb: None,
            });
        }
        s
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cs-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn sha256_sidecar_roundtrips() {
        let dir = tmpdir("sha");
        let p = dir.join("i.json");
        let mut s = mk(&["alpha beta"]);
        s.docs[0].hash = 77;
        s.sha.insert(s.docs[0].path.clone(), (77, "ab".repeat(32)));
        s.save(&p).unwrap();
        let l = Store::load(&p).unwrap();
        assert_eq!(l.doc_sha256(0), Some("ab".repeat(32).as_str()));
        std::fs::remove_file(sidecar(&p, ".sha256")).unwrap();
        assert_eq!(Store::load(&p).unwrap().doc_sha256(0), None);
        let mut s2 = Store::load(&p).unwrap();
        s2.sha
            .insert(s2.docs[0].path.clone(), (78, "cd".repeat(32)));
        assert_eq!(s2.doc_sha256(0), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_store_is_an_error_not_a_panic() {
        let dir = tmpdir("bad");
        let p = dir.join("i.json");
        std::fs::write(&p, b"\xff\x00not json").unwrap();
        assert!(matches!(Store::load(&p), Err(Error::Format(_))));
        std::fs::write(&p, br#"{"docs":[],"chunks":[{"doc":3,"page":1,"text":"x","emb":null}],"embed_model":null}"#).unwrap();
        assert!(matches!(Store::load(&p), Err(Error::Format(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn folds_romanian_diacritics() {
        assert_eq!(
            tokenize("Științific ȘI țară"),
            tokenize("stiintific si tara")
        );
    }

    #[test]
    fn ligature_query_finds_plain_text_and_back_with_same_ranking() {
        let s = mk(&[
            "The final report of the committee",
            "Cooking pasta requires boiling water",
            "A final decision and a final appeal",
        ]);
        let plain = Searcher::new(&s).search("final", None, 3, 10, None);
        let lig = Searcher::new(&s).search("\u{fb01}nal", None, 3, 10, None);
        let ids = |r: &[Hit]| r.iter().map(|h| h.chunk).collect::<Vec<_>>();
        assert_eq!(ids(&plain), ids(&lig));
        assert!(!plain.is_empty() && plain.iter().all(|h| h.chunk != 1));
        let s2 = mk(&["The \u{fb01}nal report", "Cooking pasta"]);
        assert_eq!(
            Searcher::new(&s2).search("final", None, 3, 10, None)[0].chunk,
            0
        );
    }

    #[test]
    fn phrase_queries_filter_passages() {
        let s = mk(&[
            "The burden of proof lies with the authority",
            "The proof of the burden is heavy",
            "The burden of proof is on the claimant, not the authority or the tribunal",
        ]);
        let ids = |q: &str| {
            let mut v: Vec<usize> = Searcher::new(&s)
                .with_phrases(true)
                .search(q, None, 5, 10, None)
                .iter()
                .map(|h| h.chunk)
                .collect();
            v.sort();
            v
        };
        assert_eq!(ids("\"burden of proof\""), vec![0, 2]);
        assert_eq!(ids("burden of proof"), vec![0, 1, 2]);
        let r = Searcher::new(&s).with_phrases(true).search(
            "\"burden of proof\" authority",
            None,
            5,
            10,
            None,
        );
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|h| h.chunk != 1));
        assert_eq!(ids("\"proof burden\"~2"), vec![0, 2]);
        assert_eq!(ids("\"proof burden\"~3"), vec![0, 1, 2]);
        assert!(ids("\"burden proof\"~0").is_empty());
        assert_eq!(ids("\"burden of proof"), vec![0, 1, 2]);
        assert_eq!(ids("\"proof\""), ids("proof"));
        assert!(ids("\"zzzz yyyy\"").is_empty());
    }

    #[test]
    fn phrase_errors_are_reported_not_printed() {
        let s = mk(&["burden of proof"]);
        let r = Searcher::new(&s).with_phrases(true).search_report(
            "\"burden proof\"~0",
            None,
            5,
            10,
            None,
        );
        assert!(r.hits.is_empty());
        assert!(r.error.is_some());
    }

    #[test]
    fn quotes_are_ordinary_punctuation_unless_phrases_are_switched_on() {
        let s = mk(&[
            "The burden of proof lies with the authority",
            "The proof of the burden is heavy",
            "Pasta needs boiling water",
        ]);
        let ids = |q: &str, on: bool| {
            Searcher::new(&s)
                .with_phrases(on)
                .search(q, None, 5, 10, None)
                .iter()
                .map(|h| (h.chunk, (h.score * 1e6) as i64))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            ids("\"burden of proof\"", false),
            ids("burden of proof", false)
        );
        assert_eq!(ids("what is the \"clean room\" burden?", false).len(), 2);
        assert_eq!(
            ids("\"proof burden\"~3", false),
            ids("proof burden 3", false)
        );
        assert_eq!(ids("\"zzzz yyyy\" burden", false), ids("burden", false));
        assert!(!ids("\"proof burden\"~x", false).is_empty());
        assert_eq!(ids("\"burden of proof\"", true).len(), 1);
        assert!(ids("\"zzzz yyyy\" burden", true).is_empty());
    }

    #[test]
    fn phrase_does_not_cross_a_page_boundary() {
        let mut s = mk(&[
            "The burden of",
            "proof lies here",
            "Then the burden of proof returns",
        ]);
        s.chunks[0].page = 3;
        s.chunks[1].page = 4;
        s.chunks[2].page = 4;
        let r =
            Searcher::new(&s)
                .with_phrases(true)
                .search("\"burden of proof\"", None, 5, 10, None);
        assert_eq!(r.iter().map(|h| h.chunk).collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn bm25_ranks_relevant_first() {
        let s = mk(&[
            "The recognition of professional qualifications under the directive",
            "Cooking pasta requires boiling water and salt",
            "Credential evaluation and recognition of diplomas",
        ]);
        let r = Searcher::new(&s).search("recognition of qualifications", None, 3, 10, None);
        assert_eq!(r[0].chunk, 0);
        assert!(r.iter().all(|h| h.chunk != 1));
    }

    #[test]
    fn prefix_stemming_matches_inflections() {
        let s = mk(&[
            "recunoașterea calificărilor profesionale",
            "ceva fără legătură",
        ]);
        let r = Searcher::new(&s).search("recunoastere calificare profesionala", None, 2, 10, None);
        assert_eq!(r[0].chunk, 0);
    }

    #[test]
    fn dense_fusion_and_doc_cap() {
        let mut s = mk(&["alpha", "alpha beta", "alpha gamma"]);
        s.chunks[2].emb = Some(vec![1.0, 0.0]);
        s.chunks[0].emb = Some(vec![0.0, 1.0]);
        let sr = Searcher::new(&s);
        let r = sr.search("alpha", Some(&[1.0, 0.0]), 10, 2, None);
        assert_eq!(r.len(), 2, "per-doc cap applies");
        let r = sr.search("zzzz", Some(&[1.0, 0.0]), 10, 10, None);
        assert_eq!(r[0].chunk, 2, "dense-only retrieval ranks by cosine");
        let r = sr.search("alpha", Some(&[1.0, 0.0]), 10, 10, None);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn nan_embeddings_never_panic() {
        let mut s = mk(&["alpha", "alpha beta"]);
        s.chunks[0].emb = Some(vec![f32::NAN, 1.0]);
        s.chunks[1].emb = Some(vec![1.0, 0.0]);
        let r = Searcher::new(&s).search("alpha", Some(&[f32::NAN, 0.0]), 10, 10, None);
        assert_eq!(r.len(), 2);
        let r = Searcher::new(&s).search("alpha", Some(&[1.0, 0.0]), 10, 10, None);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn only_filter_restricts_documents() {
        let mut s = mk(&["alpha one", "alpha two"]);
        s.docs.push(Doc {
            path: "b".into(),
            title: "B".into(),
            hash: 0,
            pages: 1,
        });
        s.chunks.push(Chunk {
            doc: 1,
            page: 1,
            text: "alpha three".into(),
            emb: None,
        });
        let allow: HashSet<usize> = [1].into_iter().collect();
        let r = Searcher::new(&s).search("alpha", None, 10, 10, Some(&allow));
        assert_eq!(r.len(), 1);
        assert_eq!(s.chunks[r[0].chunk].doc, 1);
    }

    #[test]
    fn coverage_measures_term_overlap() {
        assert!((coverage("burden of proof", "the burden of proof lies here") - 1.0).abs() < 1e-6);
        assert!(coverage("burden of proof", "pasta and olive oil") < 0.01);
        assert!((coverage("burden pasta", "the burden is heavy") - 0.5).abs() < 1e-6);
    }

    #[test]
    fn remove_doc_remaps_ids() {
        let mut s = mk(&["x"]);
        s.docs.push(Doc {
            path: "b".into(),
            title: "B".into(),
            hash: 0,
            pages: 1,
        });
        s.chunks.push(Chunk {
            doc: 1,
            page: 1,
            text: "y".into(),
            emb: None,
        });
        s.remove_doc(0);
        assert_eq!(s.chunks.len(), 1);
        assert_eq!(s.chunks[0].doc, 0);
        s.remove_doc(99);
        assert_eq!(s.docs.len(), 1);
    }

    #[test]
    fn bm25_cache_roundtrip_and_fingerprint_invalidation() {
        let dir = tmpdir("cache");
        let p = dir.join("i.json");
        let s = mk(&["alpha beta gamma", "beta delta", "epsilon"]);
        s.save(&p).unwrap();
        let mut s = Store::load(&p).unwrap();
        let first = Searcher::new(&s).search("beta", None, 5, 5, None);
        assert!(sidecar(&p, ".bm25").exists());
        let key = s.fingerprint();
        let b = Bm25::from_bytes(&std::fs::read(sidecar(&p, ".bm25")).unwrap(), key).unwrap();
        assert_eq!(b, Bm25::new(&s.chunks));
        // library changes -> fingerprint changes -> cache is not reused
        s.chunks.push(Chunk {
            doc: 0,
            page: 1,
            text: "beta beta beta".into(),
            emb: None,
        });
        assert_ne!(s.fingerprint(), key);
        let second = Searcher::new(&s).search("beta", None, 5, 5, None);
        assert_eq!(second.len(), first.len() + 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hostile_cache_bytes_are_rejected() {
        let s = mk(&["alpha beta", "gamma"]);
        let good = Bm25::new(&s.chunks).to_bytes(5);
        assert!(Bm25::from_bytes(&good, 5).is_some());
        assert!(Bm25::from_bytes(&good, 6).is_none());
        for cut in 0..good.len() {
            assert!(Bm25::from_bytes(&good[..cut], 5).is_none(), "cut {cut}");
        }
        let mut x = 0x9E3779B9u64;
        for _ in 0..2000 {
            let mut b = good.clone();
            for _ in 0..3 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let i = (x as usize) % b.len();
                b[i] = (x >> 20) as u8;
            }
            let _ = Bm25::from_bytes(&b, 5); // must not panic or over-allocate
        }
        let mut huge = good.clone();
        huge[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Bm25::from_bytes(&huge, 5).is_none());
    }

    #[test]
    fn huge_query_is_bounded() {
        let s = mk(&["alpha beta"]);
        let q = "alpha ".repeat(50_000) + &(0..5000).map(|i| format!("w{i} ")).collect::<String>();
        assert_eq!(Searcher::new(&s).search(&q, None, 3, 3, None).len(), 1);
    }
}
