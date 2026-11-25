//! The page-numbered shelf of a collection: the same documents as the collection, read again by
//! the scholar reader so that every passage knows its page. It feeds what the plain collection
//! cannot do: search with page numbers, the citation check with real pages, provenance receipts
//! (which need document identities) and the retrieval evaluation. Nothing leaves the computer.
//!
//! The shelf is a file next to the collections. Adding files rewrites it under a lock, so two
//! background reads cannot interleave.

use crate::audit::{self, AuditOptions, AuditReport, MemPassages, SourceInfo};
use crate::evalset::{self, EvalReport};
use crate::index::Store;
use crate::ingest::Outcome;
use crate::receipt::{self, ExcerptRecord, Issue, PassageSet};
use crate::retrieve::{Index, Passage, SearchOptions, SearchResult};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Largest file the shelf reads (bytes). Bigger files are skipped, not truncated.
pub const MAX_FILE_BYTES: u64 = 200 << 20;

static WRITE: Mutex<()> = Mutex::new(());

/// `report.pdf` is cited as `report`. A dot followed by a number is part of the name, not an
/// extension: "Chapter 2.5" and "Chapter 2" are two different documents.
pub fn stem(name: &str) -> String {
    match name.rsplit_once('.') {
        Some((s, ext))
            if !s.is_empty()
                && ext.len() <= 5
                && !ext.contains(' ')
                && ext.chars().next().is_some_and(char::is_alphabetic) =>
        {
            s.to_string()
        }
        _ => name.to_string(),
    }
}

/// The shelf file of a collection inside `dir`; the id is reduced to safe characters.
pub fn shelf_file(dir: &Path, coll_id: &str) -> PathBuf {
    let safe: String = coll_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        .take(80)
        .collect();
    dir.join(format!("{safe}.shelf.json"))
}

/// What happened to one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Added {
    /// Read: this many passages.
    Passages(usize),
    /// Already there, unchanged.
    Same,
    /// No text could be read (a scan without a text layer, say).
    NoText,
    /// Could not be read; the reason.
    Failed(String),
}

/// Reads `files` into the shelf at `path` (created when missing) and saves it. Returns what
/// happened to each file, by file name.
pub fn add_files(path: &Path, files: &[PathBuf]) -> Vec<(String, Added)> {
    let _lock = WRITE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut ix = match Index::load(path) {
        Ok(ix) => ix,
        Err(_) => Index::new(),
    };
    let mut out = Vec::with_capacity(files.len());
    let mut changed = false;
    for f in files {
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let res = read_one(&mut ix, f, &name);
        changed |= matches!(res, Added::Passages(_));
        out.push((name, res));
    }
    if changed && let Err(e) = ix.save(path) {
        for (_, r) in out.iter_mut() {
            if matches!(r, Added::Passages(_)) {
                *r = Added::Failed(e.to_string());
            }
        }
    }
    out
}

fn read_one(ix: &mut Index, f: &Path, name: &str) -> Added {
    if name.is_empty() {
        return Added::Failed("no file name".into());
    }
    match std::fs::metadata(f) {
        Ok(m) if m.len() > MAX_FILE_BYTES => return Added::Failed("file too large".into()),
        Ok(_) => {}
        Err(e) => return Added::Failed(e.to_string()),
    }
    let bytes = match std::fs::read(f) {
        Ok(b) => b,
        Err(e) => return Added::Failed(e.to_string()),
    };
    let ext = f
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ix.add_bytes(name, &stem(name), &ext, &bytes, false) {
        Ok(Outcome::Added { chunks }) => Added::Passages(chunks),
        Ok(Outcome::Unchanged) => Added::Same,
        Ok(Outcome::NoText) => Added::NoText,
        Err(e) => Added::Failed(e.to_string()),
    }
}

/// Takes one document out of the shelf at `path`, by file name. Missing shelf: nothing happens.
pub fn remove_file(path: &Path, name: &str) {
    let _lock = WRITE.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(mut ix) = Index::load(path) else {
        return;
    };
    if let Some(id) = ix.store().doc_by_path(name) {
        ix.remove(id);
        let _ = ix.save(path);
    }
}

/// Deletes the shelf file and its side files.
pub fn delete(path: &Path) {
    let _lock = WRITE.lock().unwrap_or_else(|e| e.into_inner());
    for suffix in ["", ".sha256", ".bm25"] {
        let mut p = path.as_os_str().to_owned();
        p.push(suffix);
        let _ = std::fs::remove_file(PathBuf::from(p));
    }
}

/// A loaded shelf.
#[derive(Debug, Default)]
pub struct Shelf {
    index: Index,
}

impl Shelf {
    /// Loads the shelf at `path`; a missing or unreadable file gives an empty shelf.
    pub fn load(path: &Path) -> Shelf {
        let _lock = WRITE.lock().unwrap_or_else(|e| e.into_inner());
        Shelf {
            index: Index::load(path).unwrap_or_default(),
        }
    }

    /// In-memory shelf from files' bytes (for tests and callers that have no files).
    pub fn from_bytes(items: &[(&str, &str, &[u8])]) -> Shelf {
        let mut ix = Index::new();
        for (name, ext, bytes) in items {
            let _ = ix.add_bytes(name, &stem(name), ext, bytes, false);
        }
        Shelf { index: ix }
    }

    pub fn store(&self) -> &Store {
        self.index.store()
    }

    pub fn is_empty(&self) -> bool {
        self.store().chunks.is_empty()
    }

    /// (documents, passages, documents with real page numbers).
    pub fn counts(&self) -> (usize, usize, usize) {
        let s = self.store();
        (
            s.docs.len(),
            s.chunks.len(),
            s.docs.iter().filter(|d| d.pages > 0).count(),
        )
    }

    /// Best passages for a question, with pages and how much of the question each covers.
    pub fn search(&self, query: &str, k: usize) -> SearchResult {
        self.index.search(
            query,
            None,
            &SearchOptions {
                k: k.clamp(1, 50),
                phrases: true,
                ..SearchOptions::default()
            },
        )
    }

    /// The pandoc citation that points at a passage: `[@report, p. 3]`, or `[@report]` when the
    /// document has no pages.
    pub fn citation(p: &Passage) -> String {
        let key = stem(&p.doc_path);
        if p.page > 0 {
            format!("[@{key}, p. {}]", p.page)
        } else {
            format!("[@{key}]")
        }
    }

    /// The sources the citation check reads, with real page numbers.
    pub fn sources(&self) -> MemPassages {
        let s = self.store();
        let mut per: Vec<Vec<(u32, String)>> = vec![Vec::new(); s.docs.len()];
        for c in &s.chunks {
            if let Some(v) = per.get_mut(c.doc) {
                v.push((u32::try_from(c.page).unwrap_or(0), c.text.clone()));
            }
        }
        let mut m = MemPassages::default();
        for (d, p) in s.docs.iter().zip(per) {
            m.add(
                SourceInfo {
                    key: stem(&d.path),
                    aliases: vec![d.path.clone()],
                    title: d.title.clone(),
                    pages: u32::try_from(d.pages).unwrap_or(0),
                    ..SourceInfo::default()
                },
                p,
            );
        }
        m
    }

    /// The citation check against this shelf.
    pub fn audit(&self, text: &str, opts: &AuditOptions) -> AuditReport {
        audit::audit(&self.sources(), text, opts)
    }

    /// The passages of the shelf as a receipt verifier wants them.
    pub fn passage_set(&self) -> PassageSet {
        let s = self.store();
        let mut set = PassageSet::new();
        for c in &s.chunks {
            if let Some(d) = s.docs.get(c.doc) {
                set.insert(
                    &d.path,
                    u32::try_from(c.page).unwrap_or(0),
                    &c.text,
                    s.doc_sha256(c.doc),
                );
            }
        }
        set
    }

    /// A provenance receipt for an answer written from `passages`. With a `key` the receipt is
    /// sealed (HMAC-SHA-256); without one it is plainly hashed.
    pub fn issue_receipt(
        &self,
        question: &str,
        model: &str,
        answer: &str,
        passages: &[Passage],
        key: Option<&[u8]>,
        tool: &str,
    ) -> Result<String, String> {
        let s = self.store();
        let recs: Vec<ExcerptRecord<'_>> = passages
            .iter()
            .filter_map(|p| {
                let d = s.docs.get(p.doc)?;
                Some(ExcerptRecord {
                    path: &d.path,
                    title: &d.title,
                    page: u32::try_from(p.page).unwrap_or(0),
                    text: &p.text,
                    doc_sha256: s.doc_sha256(p.doc),
                    weak_hash: Some(d.hash),
                })
            })
            .collect();
        let shas: Vec<Option<&str>> = (0..s.docs.len()).map(|i| s.doc_sha256(i)).collect();
        receipt::issue(&Issue {
            question,
            model,
            mode: "ask",
            answer,
            excerpts: &recs,
            created: receipt::now_secs(),
            tool,
            index: receipt::index_fingerprint(&shas),
            key: key.filter(|k| !k.is_empty()),
        })
        .map_err(|e| e.to_string())
    }

    /// Verifies a receipt against this shelf: the digest, the seal (when a key is given), the
    /// answer (when given) and that every excerpt still exists unchanged.
    pub fn verify_receipt(
        &self,
        receipt_text: &str,
        key: Option<&[u8]>,
        answer: Option<&str>,
    ) -> receipt::Verification {
        receipt::verify(
            receipt_text,
            key.filter(|k| !k.is_empty()),
            answer.filter(|a| !a.is_empty()),
            Some(&self.passage_set()),
        )
    }

    /// Runs a question set (one JSON object per line: `{"q": "...", "doc": "title", "page": 3}`).
    pub fn evaluate(&self, cases: &str) -> Result<EvalReport, String> {
        let cases = evalset::parse_cases(cases).map_err(|e| e.to_string())?;
        if cases.is_empty() {
            return Err("no questions".into());
        }
        Ok(evalset::evaluate(self.store(), &cases, &|_| None, &|_| {
            None
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipt::Status;

    fn mini_pdf(text: &str) -> Vec<u8> {
        let content = format!("BT /F1 12 Tf 72 700 Td ({text}) Tj ET");
        let objs = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_string(),
            format!("<< /Length {} >>\nstream\n{}\nendstream", content.len(), content),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offs = Vec::new();
        for (i, o) in objs.iter().enumerate() {
            offs.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n{}\nendobj\n", i + 1, o).as_bytes());
        }
        let xref = out.len();
        out.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes(),
        );
        for o in offs {
            out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
                objs.len() + 1,
                xref
            )
            .as_bytes(),
        );
        out
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("shelf-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn shelf() -> Shelf {
        Shelf::from_bytes(&[
            (
                "law.txt",
                "txt",
                b"The burden of proof lies with the authority in every administrative case.",
            ),
            (
                "scan.pdf",
                "pdf",
                &mini_pdf("Reactor output was 42 megawatts"),
            ),
        ])
    }

    #[test]
    fn stem_keeps_numbered_names_whole() {
        assert_eq!(stem("report.pdf"), "report");
        assert_eq!(stem("Chapter 2.5"), "Chapter 2.5");
        assert_eq!(stem("noext"), "noext");
    }

    #[test]
    fn search_returns_pages_and_a_pandoc_citation() {
        let s = shelf();
        let (docs, passages, paged) = s.counts();
        assert_eq!((docs, paged), (2, 1));
        assert!(passages >= 2);
        let r = s.search("megawatts reactor", 5);
        let p = &r.passages[0];
        assert_eq!(p.page, 1);
        assert_eq!(Shelf::citation(p), "[@scan, p. 1]");
        let r = s.search("burden of proof", 5);
        assert_eq!(Shelf::citation(&r.passages[0]), "[@law]");
    }

    #[test]
    fn the_check_reads_real_page_numbers() {
        let s = shelf();
        let ok = s.audit(
            "The reactor produced 42 megawatts [@scan, p. 1].",
            &AuditOptions::default(),
        );
        assert!(
            ok.findings.iter().all(|f| !f.is_problem()),
            "{:?}",
            ok.findings
        );
        let wrong = s.audit(
            "The reactor produced 42 megawatts [@scan, p. 9].",
            &AuditOptions::default(),
        );
        assert!(
            wrong.findings.iter().any(|f| f.is_problem()),
            "{:?}",
            wrong.findings
        );
    }

    #[test]
    fn a_receipt_round_trips_and_notices_tampering() {
        let s = shelf();
        let r = s.search("megawatts", 3);
        let answer = "The reactor produced 42 megawatts.";
        let text = s
            .issue_receipt(
                "How much?",
                "test-model",
                answer,
                &r.passages,
                Some(b"k"),
                "test 1",
            )
            .unwrap();
        let v = s.verify_receipt(&text, Some(b"k"), Some(answer));
        assert!(v.ok, "{:?}", v.lines());
        assert!(v.has("hmac", Status::Pass), "{:?}", v.lines());
        // a different answer, a wrong key and an edited receipt are all noticed
        assert!(
            !s.verify_receipt(&text, Some(b"k"), Some("Another answer."))
                .ok
        );
        assert!(!s.verify_receipt(&text, Some(b"other"), Some(answer)).ok);
        assert!(
            !s.verify_receipt(&text.replace("test-model", "other-model"), Some(b"k"), None)
                .ok
        );
        // a shelf where the passage is gone cannot vouch for the excerpt
        let empty = Shelf::default();
        assert!(!empty.verify_receipt(&text, Some(b"k"), Some(answer)).ok);
        // an empty key is the same as no key; an empty answer is not compared
        assert!(s.verify_receipt(&text, None, Some("")).ok);
    }

    #[test]
    fn files_go_in_by_name_and_out_again() {
        let d = tmp("files");
        let a = d.join("note.txt");
        std::fs::write(
            &a,
            "Alpha beta gamma delta. The tribunal decided on Tuesday.",
        )
        .unwrap();
        let big = d.join("empty.txt");
        std::fs::write(&big, "").unwrap();
        let path = shelf_file(&d, "c1/../x");
        assert!(path.ends_with("c1x.shelf.json"));
        let r = add_files(&path, &[a.clone(), big, d.join("missing.txt")]);
        assert!(matches!(r[0].1, Added::Passages(n) if n >= 1), "{r:?}");
        assert!(matches!(r[1].1, Added::NoText | Added::Failed(_)), "{r:?}");
        assert!(matches!(r[2].1, Added::Failed(_)), "{r:?}");
        assert_eq!(add_files(&path, &[a])[0].1, Added::Same);
        let s = Shelf::load(&path);
        assert_eq!(s.counts().0, 1);
        assert_eq!(s.search("tribunal", 3).passages[0].doc_path, "note.txt");
        remove_file(&path, "note.txt");
        assert!(Shelf::load(&path).is_empty());
        delete(&path);
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_evaluation_finds_the_document_that_answers() {
        let s = shelf();
        let rep = s
            .evaluate(
                "# comment\n{\"q\": \"burden of proof authority\", \"doc\": \"law\"}\n\
                 {\"q\": \"reactor megawatts\", \"doc\": \"scan\", \"page\": 1}\n\
                 {\"q\": \"anything\", \"doc\": \"nothing here\"}",
            )
            .unwrap();
        assert_eq!(rep.cases, 2);
        assert_eq!(rep.recall_at_1, 1.0);
        assert_eq!(rep.unresolved.len(), 1);
        assert!(s.evaluate("not json").is_err());
        assert!(s.evaluate("# only a comment").is_err());
    }
}
