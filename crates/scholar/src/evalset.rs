// Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//! Retrieval evaluation on your own library: questions with the document (and optionally page) that
//! answers each; reports recall@1/3/10 and MRR. Without this, "the search is good" is a feeling.
//! No model needed.

use crate::index::{Error, Result, Searcher, Store};
use serde_json::{Value, json};

/// Most cases read from one file (a hostile file cannot make an evaluation run forever).
pub const MAX_CASES: usize = 100_000;

/// One question with the document (and optionally the page) that answers it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Case {
    /// The question.
    pub q: String,
    /// Document: a title or path fragment, or whatever the caller's resolver understands.
    pub doc: String,
    /// Expected 1-based page; passages without pages (page 0) also count.
    pub page: Option<usize>,
}

/// Parse one JSON object per line: `{"q": "...", "doc": "title part", "page": 12}`. Blank lines and
/// lines starting with `#` are skipped.
pub fn parse_cases(text: &str) -> Result<Vec<Case>> {
    let mut out = Vec::new();
    for (i, l) in text.lines().enumerate() {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if out.len() >= MAX_CASES {
            return Err(Error::Limit(format!("more than {MAX_CASES} cases")));
        }
        let v: Value =
            serde_json::from_str(l).map_err(|e| Error::Format(format!("line {}: {e}", i + 1)))?;
        let q = v["q"]
            .as_str()
            .ok_or_else(|| Error::Format(format!("line {}: missing \"q\"", i + 1)))?;
        let doc = v["doc"]
            .as_str()
            .ok_or_else(|| Error::Format(format!("line {}: missing \"doc\"", i + 1)))?;
        out.push(Case {
            q: q.to_string(),
            doc: doc.to_string(),
            page: v["page"].as_u64().and_then(|p| usize::try_from(p).ok()),
        });
    }
    if out.is_empty() {
        return Err(Error::Format(
            "no cases found (one JSON object per line: {\"q\": \"...\", \"doc\": \"title part\", \"page\": 12})"
                .into(),
        ));
    }
    Ok(out)
}

/// Result of an evaluation run.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalReport {
    /// Cases whose document was found in the library.
    pub cases: usize,
    /// Share of cases answered at rank 1 / within 3 / within 10 (0..1, 3 decimals).
    pub recall_at_1: f64,
    /// See `recall_at_1`.
    pub recall_at_3: f64,
    /// See `recall_at_1`.
    pub recall_at_10: f64,
    /// Mean reciprocal rank over the top 10.
    pub mrr: f64,
    /// Cases whose expected passage was not in the top 10: `(question, document title, page)`.
    pub misses: Vec<(String, String, Option<usize>)>,
    /// Cases naming a document the library does not have: `(question, document spec)`.
    pub unresolved: Vec<(String, String)>,
}

impl EvalReport {
    /// JSON form (`recall@1`, … keys as in TEZAUR).
    pub fn to_json(&self) -> Value {
        json!({
            "cases": self.cases,
            "recall@1": self.recall_at_1,
            "recall@3": self.recall_at_3,
            "recall@10": self.recall_at_10,
            "mrr": self.mrr,
            "misses": self.misses.iter().map(|(q, d, p)| json!({"q": q, "doc": d, "page": p})).collect::<Vec<_>>(),
            "unresolved_docs": self.unresolved.iter().map(|(q, d)| json!({"q": q, "doc": d})).collect::<Vec<_>>(),
        })
    }

    /// Short Markdown report.
    pub fn markdown(&self) -> String {
        let mut s = format!(
            "# Retrieval evaluation\n\n{} questions — recall@1 {}, recall@3 {}, recall@10 {}, MRR {}\n\nThis measures your library and your questions only; it says nothing about answer quality.\n\n",
            self.cases, self.recall_at_1, self.recall_at_3, self.recall_at_10, self.mrr
        );
        for (q, d, p) in &self.misses {
            let page = p.map(|p| p.to_string()).unwrap_or_else(|| "null".into());
            s.push_str(&format!("- miss: “{q}” → expected {d} (p. {page})\n"));
        }
        for (q, d) in &self.unresolved {
            s.push_str(&format!("- unknown document “{d}” in: {q}\n"));
        }
        s
    }
}

/// Which document does `spec` name: the caller's resolver first (citation keys, file stems), else a
/// case-insensitive part of the title or path.
fn resolve(store: &Store, resolver: &dyn Fn(&str) -> Option<usize>, spec: &str) -> Option<usize> {
    resolver(spec)
        .filter(|&d| d < store.docs.len())
        .or_else(|| {
            let s = spec.to_lowercase();
            if s.is_empty() {
                return None;
            }
            store.docs.iter().position(|d| {
                d.title.to_lowercase().contains(&s) || d.path.to_lowercase().contains(&s)
            })
        })
}

/// Run `cases` against `store`. `resolver` maps a case's `doc` to a document id (return `None` to
/// fall back to title/path matching); `emb` supplies a query embedding when the library has them.
pub fn evaluate(
    store: &Store,
    cases: &[Case],
    resolver: &dyn Fn(&str) -> Option<usize>,
    emb: &dyn Fn(&str) -> Option<Vec<f32>>,
) -> EvalReport {
    let sr = Searcher::new(store);
    let (mut r1, mut r3, mut r10, mut rr, mut n) = (0usize, 0usize, 0usize, 0f64, 0usize);
    let mut misses = Vec::new();
    let mut unresolved = Vec::new();
    for c in cases {
        let Some(d) = resolve(store, resolver, &c.doc) else {
            unresolved.push((c.q.clone(), c.doc.clone()));
            continue;
        };
        n += 1;
        let qe = emb(&c.q);
        let hits = sr.search(&c.q, qe.as_deref(), 10, 10, None);
        let rank = hits.iter().position(|h| {
            let ch = &store.chunks[h.chunk];
            ch.doc == d && c.page.map(|p| ch.page == p || ch.page == 0).unwrap_or(true)
        });
        match rank {
            Some(r) => {
                rr += 1.0 / (r as f64 + 1.0);
                if r < 1 {
                    r1 += 1
                }
                if r < 3 {
                    r3 += 1
                }
                r10 += 1;
            }
            None => misses.push((c.q.clone(), store.docs[d].title.clone(), c.page)),
        }
    }
    let f = |x: usize| {
        if n == 0 {
            0.0
        } else {
            ((x as f64 / n as f64) * 1000.0).round() / 1000.0
        }
    };
    EvalReport {
        cases: n,
        recall_at_1: f(r1),
        recall_at_3: f(r3),
        recall_at_10: f(r10),
        mrr: if n == 0 {
            0.0
        } else {
            ((rr / n as f64) * 1000.0).round() / 1000.0
        },
        misses,
        unresolved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Chunk, Doc};

    fn store() -> Store {
        let mut s = Store::default();
        for (i, (t, txt)) in [
            ("Alpha Book", "mitochondria produce energy in the cell"),
            ("Beta Book", "parliament passed the budget law yesterday"),
        ]
        .iter()
        .enumerate()
        {
            s.docs.push(Doc {
                path: format!("/x/{}.txt", t.to_lowercase().replace(' ', "")),
                title: t.to_string(),
                hash: i as u64,
                pages: 0,
            });
            s.chunks.push(Chunk {
                doc: i,
                page: 0,
                text: txt.to_string(),
                emb: None,
            });
        }
        s
    }

    #[test]
    fn recall_and_mrr() {
        let cases = parse_cases("{\"q\":\"energy from mitochondria\",\"doc\":\"alpha\"}\n{\"q\":\"budget law\",\"doc\":\"Beta Book\"}\n{\"q\":\"budget law\",\"doc\":\"alpha\"}\n{\"q\":\"x\",\"doc\":\"nope\"}\n").unwrap();
        let r = evaluate(&store(), &cases, &|_| None, &|_| None);
        assert_eq!(r.cases, 3);
        assert_eq!(r.recall_at_1, 0.667);
        assert_eq!(r.unresolved.len(), 1);
        assert_eq!(r.misses.len(), 1);
        let j = r.to_json();
        assert_eq!(j["recall@1"], 0.667);
        assert!(
            r.markdown()
                .contains("miss: “budget law” → expected Alpha Book")
        );
        assert!(parse_cases("not json").is_err());
        assert!(parse_cases("").is_err());
        assert!(parse_cases("# only a comment\n").is_err());
        assert!(parse_cases("{\"q\":1,\"doc\":\"a\"}").is_err());
    }

    #[test]
    fn resolver_and_page_filter() {
        let s = store();
        let cases = parse_cases("{\"q\":\"budget law\",\"doc\":\"KEY1\",\"page\":7}").unwrap();
        // page 0 passages count for any expected page
        let r = evaluate(&s, &cases, &|k| (k == "KEY1").then_some(1), &|_| None);
        assert_eq!(r.recall_at_1, 1.0);
        // an out-of-range id from the resolver is ignored, not a panic
        let r = evaluate(&s, &cases, &|_| Some(99), &|_| None);
        assert_eq!(r.cases, 0);
        assert_eq!(r.unresolved.len(), 1);
        // an empty library and empty spec never panic
        let r = evaluate(&Store::default(), &cases, &|_| None, &|_| None);
        assert_eq!((r.cases, r.mrr), (0, 0.0));
        let c0 = parse_cases("{\"q\":\"x\",\"doc\":\"\"}").unwrap();
        assert_eq!(evaluate(&s, &c0, &|_| None, &|_| None).unresolved.len(), 1);
    }

    #[test]
    fn hostile_case_files_are_errors() {
        assert!(parse_cases("{\"q\":\"a\",\"doc\":\"b\",\"page\":-3}").is_ok()); // negative page ignored
        assert!(parse_cases(&"{\"q\":\"a\"".repeat(5)).is_err());
        let deep = "[".repeat(100_000);
        assert!(parse_cases(&deep).is_err());
    }
}
