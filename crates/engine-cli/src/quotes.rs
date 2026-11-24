//! Quote-locked answers: while the model writes between `«` and `»` it can only copy text that is really
//! in one of the supplied sources, and afterwards every quote is checked again and reported.
//!
//! API: a request may carry `"sources": ["text", {"id": 2, "text": "…"}]` (chat and completion endpoints). The
//! sources are numbered from 1 in the order given (`[S1]`, `[S2]`, …) and listed in the system prompt; the
//! reply is steered so that anything between `«»` is verbatim. The response gains `citations` (per choice)
//! with the verification report. `POST /v1/quotes/verify` checks any text against any sources, without a model.
use crate::genloop::Constraint;
use companion_engine::quotelock::{Options, QuoteIndex, QuoteLock, QuoteStatus, QuoteVocab, Report};
use companion_engine::tokenizer::Tokenizer;
use serde_json::{json, Value};
use std::sync::Arc;

/// Largest number of sources accepted in one request.
pub const MAX_SOURCES: usize = 64;

/// Read `sources` from a request body: strings, or objects with a `text` (or `content`) field.
pub fn parse_sources(v: Option<&Value>) -> Result<Vec<String>, String> {
    let Some(v) = v.filter(|v| !v.is_null()) else { return Ok(vec![]) };
    let a = v.as_array().ok_or("`sources` must be an array")?;
    if a.len() > MAX_SOURCES { return Err(format!("at most {MAX_SOURCES} sources are accepted")); }
    a.iter().enumerate().map(|(i, s)| match s {
        Value::String(t) if !t.trim().is_empty() => Ok(t.clone()),
        Value::Object(o) => o.get("text").or_else(|| o.get("content")).and_then(Value::as_str).filter(|t| !t.trim().is_empty()).map(str::to_string)
            .ok_or_else(|| format!("`sources[{i}]` needs a non-empty `text`")),
        _ => Err(format!("`sources[{i}]` must be a non-empty string or an object with `text`")),
    }).collect()
}

pub fn index(sources: &[String]) -> Result<Arc<QuoteIndex>, String> {
    let refs: Vec<&str> = sources.iter().map(String::as_str).collect();
    QuoteIndex::new(&refs, Options::default()).map(Arc::new).map_err(|e| format!("sources: {e}"))
}

pub fn vocab(tok: &Tokenizer) -> Arc<QuoteVocab> { Arc::new(QuoteVocab::from_tokenizer(tok)) }

/// The text that tells the model what the sources are and how to quote them.
pub fn instructions(sources: &[String]) -> String {
    let mut s = String::from("Answer using only the numbered sources below. When you quote, write the source tag and then the exact words between « and », for example [S2]«exact words». Everything between « and » must be copied unchanged from that source.\n");
    for (i, t) in sources.iter().enumerate() { s.push_str(&format!("\n[S{}]\n{}\n", i + 1, t.trim())); }
    s
}

struct Lock { lock: QuoteLock }
impl Constraint for Lock {
    fn apply(&mut self, logits: &mut [f32]) { if !self.lock.is_dead() { self.lock.apply(logits); } }
    fn accept(&mut self, token: usize) { let _ = self.lock.accept(token as u32); }
}

pub fn constraint(idx: Arc<QuoteIndex>, vocab: Arc<QuoteVocab>) -> Box<dyn Constraint> { Box::new(Lock { lock: QuoteLock::new(idx, vocab) }) }

fn status_name(s: &QuoteStatus) -> (&'static str, Option<usize>) {
    match s {
        QuoteStatus::Verbatim => ("verbatim", None),
        QuoteStatus::WrongSource { found_in } => ("wrong_source", Some(*found_in)),
        QuoteStatus::Truncated => ("truncated", None),
        QuoteStatus::Altered => ("altered", None),
        QuoteStatus::Merged => ("merged", None),
        QuoteStatus::Unsupported => ("unsupported", None),
    }
}

pub fn report_json(r: &Report) -> Value {
    let quotes: Vec<Value> = r.quotes.iter().map(|q| {
        let (st, found) = status_name(&q.status);
        let mut v = json!({"text": q.text, "tag": q.tag, "closed": q.closed, "status": st, "similarity": q.similarity, "support": q.support});
        if let Some(f) = found { v["found_in"] = json!(f); }
        if let Some(c) = &q.citation { v["source"] = json!(c.source_id); v["start"] = json!(c.start); v["end"] = json!(c.end); }
        v
    }).collect();
    let sentences: Vec<Value> = r.sentences.iter().map(|s| json!({"text": s.text, "support": s.support})).collect();
    json!({"quotes": quotes, "sentences": sentences, "all_verbatim": r.all_verbatim(),
        "counts": {"verbatim": r.verbatim, "wrong_source": r.wrong_source, "truncated": r.truncated, "altered": r.altered, "merged": r.merged, "unsupported": r.unsupported}})
}

/// Check `output` against `sources` (no model needed).
pub fn verify(output: &str, sources: &[String]) -> Result<Value, String> {
    let idx = index(sources)?;
    Ok(report_json(&idx.verify(output)))
}

/// Add `citations` to a finished response: one report per choice for the OpenAI shapes, otherwise
/// one at the top level for the first choice.
pub fn attach(resp: &mut Value, idx: &QuoteIndex, texts: &[&str]) {
    if let Some(choices) = resp.get_mut("choices").and_then(Value::as_array_mut) {
        for (c, t) in choices.iter_mut().zip(texts) { c["citations"] = report_json(&idx.verify(t)); }
    } else if let Some(t) = texts.first() {
        resp["citations"] = report_json(&idx.verify(t));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_parse() {
        assert_eq!(parse_sources(None).unwrap(), Vec::<String>::new());
        assert_eq!(parse_sources(Some(&json!(["a b", {"text": "c d"}, {"content": "e f"}]))).unwrap(), vec!["a b", "c d", "e f"]);
        assert!(parse_sources(Some(&json!("x"))).is_err());
        assert!(parse_sources(Some(&json!([""]))).is_err());
        assert!(parse_sources(Some(&json!([3]))).is_err());
        assert!(parse_sources(Some(&json!(vec!["x"; 65]))).is_err());
    }

    #[test]
    fn verification_classifies_quotes() {
        let sources = vec!["The quick brown fox jumps over the lazy dog near the river bank.".to_string(), "Completely different second source about astronomy and stars.".to_string()];
        let v = verify("As said [S1]«quick brown fox jumps over the lazy dog» and [S1]«a made up sentence that is nowhere»", &sources).unwrap();
        assert_eq!(v["quotes"][0]["status"], "verbatim");
        assert_eq!(v["quotes"][0]["source"], 1);
        assert_ne!(v["quotes"][1]["status"], "verbatim");
        assert_eq!(v["all_verbatim"], false);
        let w = verify("[S2]«quick brown fox jumps over the lazy dog»", &sources).unwrap();
        assert_eq!(w["quotes"][0]["status"], "wrong_source");
        assert_eq!(w["quotes"][0]["found_in"], 1);
    }

    #[test]
    fn instructions_number_the_sources() {
        let t = instructions(&["one".into(), "two".into()]);
        assert!(t.contains("[S1]\none") && t.contains("[S2]\ntwo"));
    }

    #[test]
    fn attach_shapes() {
        let idx = index(&["alpha beta gamma delta epsilon".to_string()]).unwrap();
        let mut oai = json!({"choices": [{"index": 0}]});
        attach(&mut oai, &idx, &["[S1]«alpha beta gamma delta»"]);
        assert_eq!(oai["choices"][0]["citations"]["quotes"][0]["status"], "verbatim");
        let mut other = json!({"message": {}});
        attach(&mut other, &idx, &["no quotes"]);
        assert_eq!(other["citations"]["quotes"].as_array().unwrap().len(), 0);
    }
}
