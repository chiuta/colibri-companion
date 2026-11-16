// Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//! The retrieval facade: build a library from documents, search it, get passages with their page
//! numbers and how much of the question each covers.
//!
//! ```
//! use companion_scholar::retrieve::{Index, SearchOptions};
//! let mut ix = Index::new();
//! ix.add_bytes("law.txt", "Law", "txt", b"The burden of proof lies with the authority.", false).unwrap();
//! let r = ix.search("burden of proof", None, &SearchOptions::default());
//! assert_eq!(r.passages[0].doc_title, "Law");
//! assert_eq!(r.passages[0].coverage, 1.0);
//! ```
//!
//! No model is bundled: to add dense retrieval, compute embeddings elsewhere, attach them with
//! [`Index::set_embedding`] and pass a query embedding to [`Index::search`]; the two rankings are
//! merged with Reciprocal Rank Fusion.

use crate::index::{Chunk, Error, Result, Searcher, Store, coverage};
use crate::ingest::{self, Outcome};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub use crate::index::{Hit, Report};

/// A retrieved passage with its provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct Passage {
    /// Index of the document in the library.
    pub doc: usize,
    /// Document identity (relative path or key).
    pub doc_path: String,
    /// Document title.
    pub doc_title: String,
    /// 1-based page; 0 when the format has no pages.
    pub page: usize,
    /// Passage text.
    pub text: String,
    /// Fused ranking score; only the order is meaningful.
    pub score: f32,
    /// Share (0..1) of the query's distinct terms found in the passage; multiply by 100 for a percentage.
    pub coverage: f32,
}

/// Search settings.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Passages returned.
    pub k: usize,
    /// At most this many passages from one document.
    pub max_per_doc: usize,
    /// Treat `"exact phrase"` and `"a b"~k` as required phrases.
    pub phrases: bool,
    /// Restrict to these document ids (`None` = whole library).
    pub allow: Option<HashSet<usize>>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions {
            k: 8,
            max_per_doc: 3,
            phrases: false,
            allow: None,
        }
    }
}

/// Answer to a search.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SearchResult {
    /// Ranked passages, best first.
    pub passages: Vec<Passage>,
    /// Non-fatal notes about the query.
    pub warnings: Vec<String>,
    /// Set when the query was not run.
    pub error: Option<String>,
}

/// A searchable library.
#[derive(Debug, Default)]
pub struct Index {
    store: Store,
}

impl Index {
    /// Empty library.
    pub fn new() -> Self {
        Index::default()
    }

    /// Wrap an existing store (validated).
    pub fn from_store(store: Store) -> Result<Self> {
        store.validate()?;
        Ok(Index { store })
    }

    /// Load from disk; a missing file gives an empty library bound to that path.
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Index {
            store: Store::load(path)?,
        })
    }

    /// Save to the path it was loaded from, or to `path`.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.store.save(path)
    }

    /// The underlying store (documents, passages).
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Mutable access to the store; the BM25 cache follows the library fingerprint automatically.
    pub fn store_mut(&mut self) -> &mut Store {
        &mut self.store
    }

    /// Add or refresh a document from bytes. `ext` selects the reader (pdf, docx, epub, html, tex,
    /// md, txt…).
    pub fn add_bytes(
        &mut self,
        key: &str,
        title: &str,
        ext: &str,
        bytes: &[u8],
        force: bool,
    ) -> Result<Outcome> {
        ingest::ingest_bytes(&mut self.store, key, title, ext, bytes, force)
    }

    /// Add or refresh one file (`base` is the folder the library lives in, see [`ingest::doc_key`]).
    pub fn add_file(&mut self, path: &Path, base: &Path, force: bool) -> Result<Outcome> {
        ingest::ingest_file(&mut self.store, path, base, force)
    }

    /// Add every supported file under `paths`; one result per file, so one bad file never stops the rest.
    pub fn add_paths(
        &mut self,
        paths: &[PathBuf],
        base: &Path,
        force: bool,
    ) -> Vec<(PathBuf, Result<Outcome>)> {
        ingest::collect_files(paths)
            .into_iter()
            .map(|f| {
                let r = self.add_file(&f, base, force);
                (f, r)
            })
            .collect()
    }

    /// Remove a document by id.
    pub fn remove(&mut self, doc: usize) {
        self.store.remove_doc(doc);
    }

    /// Attach a caller-computed embedding to a passage (normalised here). Errors on an unknown passage
    /// or a non-finite vector.
    pub fn set_embedding(&mut self, chunk: usize, mut emb: Vec<f32>) -> Result<()> {
        if emb.iter().any(|x| !x.is_finite()) {
            return Err(Error::Format("embedding holds NaN or infinity".into()));
        }
        crate::index::normalize(&mut emb);
        let c: &mut Chunk = self
            .store
            .chunks
            .get_mut(chunk)
            .ok_or_else(|| Error::Format(format!("no passage {chunk}")))?;
        c.emb = Some(emb);
        Ok(())
    }

    /// Search. `q_emb` is the query embedding when passages carry embeddings (same model and
    /// dimension), else `None`.
    pub fn search(&self, query: &str, q_emb: Option<&[f32]>, opts: &SearchOptions) -> SearchResult {
        let mut qe: Option<Vec<f32>> = None;
        if let Some(q) = q_emb
            && q.iter().all(|x| x.is_finite())
        {
            let mut v = q.to_vec();
            crate::index::normalize(&mut v);
            qe = Some(v);
        }
        let sr = Searcher::new(&self.store).with_phrases(opts.phrases);
        let rep = sr.search_report(
            query,
            qe.as_deref(),
            opts.k,
            opts.max_per_doc,
            opts.allow.as_ref(),
        );
        let passages = rep
            .hits
            .iter()
            .map(|h| {
                let c = &self.store.chunks[h.chunk];
                let d = &self.store.docs[c.doc];
                Passage {
                    doc: c.doc,
                    doc_path: d.path.clone(),
                    doc_title: d.title.clone(),
                    page: c.page,
                    text: c.text.clone(),
                    score: h.score,
                    coverage: coverage(query, &c.text),
                }
            })
            .collect();
        SearchResult {
            passages,
            warnings: rep.warnings,
            error: rep.error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lib() -> Index {
        let mut ix = Index::new();
        let pages = "The burden of proof lies with the authority.\x0cPasta needs boiling water and salt.\x0cThe proof of the burden is heavy for the claimant.";
        ix.add_bytes("law.txt", "Law", "txt", pages.as_bytes(), false)
            .unwrap();
        ix.add_bytes("cook.md", "Cooking", "md", b"Olive oil and garlic.", false)
            .unwrap();
        ix
    }

    #[test]
    fn passages_carry_page_and_coverage() {
        let ix = lib();
        let r = ix.search("burden of proof", None, &SearchOptions::default());
        assert!(r.error.is_none());
        let top = &r.passages[0];
        assert_eq!((top.doc_title.as_str(), top.page), ("Law", 1));
        assert_eq!(top.coverage, 1.0);
        assert!(r.passages.iter().all(|p| p.doc_title != "Cooking"));
        let r = ix.search("burden pasta", None, &SearchOptions::default());
        assert!(r.passages.iter().any(|p| (p.coverage - 0.5).abs() < 1e-6));
    }

    #[test]
    fn phrases_allow_and_errors() {
        let ix = lib();
        let o = SearchOptions {
            phrases: true,
            ..SearchOptions::default()
        };
        let r = ix.search("\"burden of proof\"", None, &o);
        assert_eq!(r.passages.len(), 1);
        assert_eq!(r.passages[0].page, 1);
        let r = ix.search("\"burden proof\"~0", None, &o);
        assert!(r.error.is_some() && r.passages.is_empty());
        let o = SearchOptions {
            allow: Some([1].into_iter().collect()),
            ..SearchOptions::default()
        };
        assert!(ix.search("burden", None, &o).passages.is_empty());
        assert_eq!(ix.search("garlic", None, &o).passages.len(), 1);
    }

    #[test]
    fn embeddings_fuse_and_bad_ones_are_refused() {
        let mut ix = lib();
        let n = ix.store().chunks.len();
        for i in 0..n {
            ix.set_embedding(i, vec![0.0, 1.0]).unwrap();
        }
        let target = ix
            .store()
            .chunks
            .iter()
            .position(|c| c.text.contains("Pasta"))
            .unwrap();
        ix.set_embedding(target, vec![5.0, 0.0]).unwrap();
        let r = ix.search("zzzz", Some(&[2.0, 0.0]), &SearchOptions::default());
        assert!(r.passages[0].text.contains("Pasta"));
        assert!(ix.set_embedding(0, vec![f32::NAN]).is_err());
        assert!(ix.set_embedding(999, vec![1.0]).is_err());
        // NaN query embedding is ignored, not a panic
        assert!(
            !ix.search("garlic", Some(&[f32::NAN, 1.0]), &SearchOptions::default())
                .passages
                .is_empty()
        );
    }

    #[test]
    fn save_load_and_cache_follow_the_library() {
        let dir = std::env::temp_dir().join(format!("cs-ret-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("lib.json");
        let mut ix = lib();
        ix.save(&p).unwrap();
        let mut ix2 = Index::load(&p).unwrap();
        let before = ix2.search("garlic", None, &SearchOptions::default());
        assert_eq!(before.passages.len(), 1);
        ix2.add_bytes("new.txt", "New", "txt", b"garlic soup recipe", false)
            .unwrap();
        let after = ix2.search("garlic", None, &SearchOptions::default());
        assert_eq!(
            after.passages.len(),
            2,
            "cache rebuilt after the library changed"
        );
        ix.remove(0);
        assert_eq!(ix.store().docs.len(), 1);
        ix.store().validate().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hostile_queries_never_panic() {
        let ix = lib();
        let o = SearchOptions {
            phrases: true,
            ..SearchOptions::default()
        };
        let mut x = 0xA5A5_1234_9876_5432u64;
        let alphabet: Vec<char> = "\"“”«»~ ab12-é\u{0}\u{FFFF}İ\n\\".chars().collect();
        for _ in 0..2000 {
            let mut q = String::new();
            for _ in 0..24 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                q.push(alphabet[(x % alphabet.len() as u64) as usize]);
            }
            let _ = ix.search(&q, None, &o);
            let _ = ix.search(&q, None, &SearchOptions::default());
        }
        let big = "\"a b\"~3 ".repeat(50_000);
        let _ = ix.search(&big, None, &o);
        let none = SearchOptions {
            k: 0,
            ..SearchOptions::default()
        };
        assert!(ix.search("burden", None, &none).passages.is_empty());
    }

    /// Differential check: every passage the phrase filter returns really contains the phrase words in
    /// order, and plain mode is a superset of phrase mode.
    #[test]
    fn phrase_results_are_a_subset_of_plain_results() {
        let mut ix = Index::new();
        let words = [
            "burden",
            "proof",
            "of",
            "the",
            "claimant",
            "authority",
            "law",
        ];
        let mut x = 0x1357_9BDF_2468_ACE0u64;
        for d in 0..6 {
            let mut t = String::new();
            for _ in 0..400 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                t.push_str(words[(x % words.len() as u64) as usize]);
                t.push(' ');
            }
            ix.add_bytes(&format!("d{d}"), "D", "txt", t.as_bytes(), false)
                .unwrap();
        }
        let plain = SearchOptions {
            k: 100,
            max_per_doc: 100,
            ..SearchOptions::default()
        };
        let phr = SearchOptions {
            phrases: true,
            ..plain.clone()
        };
        let a = ix.search("burden of proof", None, &plain);
        let b = ix.search("\"burden of proof\"", None, &phr);
        assert!(!b.passages.is_empty());
        let aset: HashSet<&str> = a.passages.iter().map(|p| p.text.as_str()).collect();
        for p in &b.passages {
            assert!(aset.contains(p.text.as_str()));
            assert!(p.text.contains("burden of proof"), "{}", p.text);
        }
    }
}
