//! `corbul embed` and `corbul rerank`: sentence vectors and relevance scores from a BERT-family
//! encoder folder (`config.json`, `tokenizer.json`, safetensors; see `companion_engine::bert`).
//!
//! ```text
//! corbul embed  MODEL [TEXT…] [--file F] [--pool cls|mean] [--no-normalize] [--json]
//! corbul rerank MODEL --query Q [DOC…] [--file F] [--top N] [--json]
//! ```
//! `--file` holds one text per line. `rerank` needs a model with a classification head (a
//! cross-encoder such as the ms-marco MiniLM models); `embed` works with any BERT encoder.
use crate::args::Args;
use companion_engine::bert::{Encoder, Pool};
use serde_json::json;
use std::path::Path;

/// Most texts one call takes, and the most bytes read from `--file`.
pub const MAX_TEXTS: usize = 4096;
const MAX_FILE: u64 = 64 << 20;

pub fn pool_of(a: &Args) -> Result<Option<Pool>, String> {
    match a.get("pool") {
        None => Ok(None),
        Some("cls") | Some("first") => Ok(Some(Pool::Cls)),
        Some("mean") => Ok(Some(Pool::Mean)),
        Some(o) => Err(format!("--pool: `{o}` is not cls or mean")),
    }
}

/// The texts after the model on the command line plus the lines of `--file`.
fn texts(a: &Args, skip: usize) -> Result<Vec<String>, String> {
    let mut v: Vec<String> = a.pos.iter().skip(skip).cloned().collect();
    if let Some(f) = a.get("file") {
        let len = std::fs::metadata(f).map_err(|e| format!("{f}: {e}"))?.len();
        if len > MAX_FILE {
            return Err(format!("{f}: larger than {} MiB", MAX_FILE >> 20));
        }
        let s = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
        v.extend(s.lines().filter(|l| !l.trim().is_empty()).map(String::from));
    }
    if v.len() > MAX_TEXTS {
        return Err(format!("at most {MAX_TEXTS} texts per call"));
    }
    Ok(v)
}

fn open(a: &Args, what: &str) -> Result<Encoder, String> {
    let p = a
        .pos
        .first()
        .ok_or_else(|| format!("{what}: give a BERT-family model folder"))?;
    let path = Path::new(p);
    if !path.is_dir() {
        return Err(format!(
            "{what}: `{p}` is not a folder (only Hugging Face BERT folders are supported)"
        ));
    }
    Encoder::open(path).map_err(|e| e.to_string())
}

fn f(x: f32) -> String {
    format!("{x:.7}")
}

/// `corbul embed`.
pub fn embed(a: &Args) -> Result<(), String> {
    let e = open(a, "embed")?;
    let t = texts(a, 1)?;
    if t.is_empty() {
        return Err("embed: give at least one TEXT (or --file F)".into());
    }
    let pool = pool_of(a)?;
    let norm = !a.flag("no-normalize");
    let mut vecs = vec![];
    for x in &t {
        vecs.push(e.embed_text(x, pool, norm).map_err(|e| e.to_string())?);
    }
    if a.flag("json") {
        outln!(
            "{}",
            json!({"model": e.name, "dimensions": vecs[0].len(),
                "pooling": match pool.unwrap_or(e.pool) { Pool::Cls => "cls", Pool::Mean => "mean" },
                "normalized": norm, "embeddings": vecs})
        );
    } else {
        for v in &vecs {
            let line: Vec<String> = v.iter().map(|x| f(*x)).collect();
            out!("{}\n", line.join(" "));
        }
    }
    Ok(())
}

/// `corbul rerank`.
pub fn rerank(a: &Args) -> Result<(), String> {
    let e = open(a, "rerank")?;
    let q = a
        .get("query")
        .ok_or("rerank: give --query TEXT")?
        .to_string();
    let docs = texts(a, 1)?;
    if docs.is_empty() {
        return Err("rerank: give at least one DOC (or --file F)".into());
    }
    if !e.bert.has_classifier() {
        return Err("rerank: this model has no classification head (use a cross-encoder)".into());
    }
    let mut scored: Vec<(usize, f32)> = vec![];
    for (i, d) in docs.iter().enumerate() {
        scored.push((i, e.score(&q, d).map_err(|e| e.to_string())?));
    }
    scored.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
    if let Some(n) = a.num::<usize>("top")? {
        scored.truncate(n);
    }
    if a.flag("json") {
        let r: Vec<_> = scored
            .iter()
            .map(|(i, s)| json!({"index": i, "score": s, "document": docs[*i]}))
            .collect();
        outln!("{}", json!({"model": e.name, "results": r}));
    } else {
        for (i, s) in scored {
            outln!("{:>10.5}  #{i}  {}", s, docs[i]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use companion_engine::bert::demo::write_folder;

    fn args(v: &[&str]) -> Args {
        Args::parse(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }
    fn dir(tag: &str, cls: bool) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("enc-cmd-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        write_folder(&d, cls, 4).unwrap();
        d
    }

    #[test]
    fn embed_and_rerank_run_on_a_folder() {
        let d = dir("ok", true);
        let m = d.to_string_lossy().to_string();
        embed(&args(&[&m, "hello world", "world hello", "--json"])).unwrap();
        embed(&args(&[&m, "hello", "--pool", "cls", "--no-normalize"])).unwrap();
        rerank(&args(&[
            &m,
            "--query",
            "hello",
            "world",
            "hello hello",
            "--top",
            "1",
        ]))
        .unwrap();
        rerank(&args(&[&m, "--query", "hello", "world", "--json"])).unwrap();
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn mistakes_are_messages_not_panics() {
        let plain = dir("plain", false);
        let m = plain.to_string_lossy().to_string();
        assert!(embed(&args(&[])).is_err());
        assert!(
            embed(&args(&["/no/such/folder", "x"]))
                .unwrap_err()
                .contains("folder")
        );
        assert!(embed(&args(&[&m])).unwrap_err().contains("TEXT"));
        assert!(
            embed(&args(&[&m, "x", "--pool", "max"]))
                .unwrap_err()
                .contains("--pool")
        );
        assert!(rerank(&args(&[&m, "doc"])).unwrap_err().contains("--query"));
        assert!(
            rerank(&args(&[&m, "--query", "q"]))
                .unwrap_err()
                .contains("DOC")
        );
        assert!(
            rerank(&args(&[&m, "--query", "q", "doc"]))
                .unwrap_err()
                .contains("classification head")
        );
        assert!(embed(&args(&[&m, "x", "--file", "/no/such/file"])).is_err());
        std::fs::remove_dir_all(&plain).unwrap();
    }
}
