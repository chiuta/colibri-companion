//! `corbul rag add|search|ask`: a small local document library you can search and ask questions of.
//!
//! A library is a folder: `docs/` holds the text of every added file (named by file name; adding a
//! file with the same name replaces it) and `library.json` records the settings. The engine's
//! `KnowledgeBase` has no file format of its own yet, so every `search` and `ask` re-reads `docs/`,
//! chunks and indexes it again (fast for hundreds of documents; not meant for millions of chunks).
//!
//! * `rag add LIB FILE|DIR…`: copies `.txt`, `.md`, `.markdown`, `.csv`, `.json`, `.rst`, `.html`, `.log`,
//!   `.rs`, `.py`, … (any file that is valid UTF-8, up to 8 MiB) into the library. PDFs and Office files are
//!   not read; convert them to text first.
//! * `rag search LIB --query Q [--k 5] [--json]`: hybrid search (vectors + keywords, fused by rank).
//!   The default embedder is a hashing embedder that knows words and word parts, not meaning; with
//!   `--embed-model MODEL` the hidden states of a model are used instead (slower; a real embedding model
//!   works much better than a chat model, and every search re-embeds the library).
//! * `rag ask LIB MODEL --question Q [--k 4]`: retrieves passages and answers from them with the quote
//!   lock on: text between « » is copied verbatim from a numbered source and checked afterwards.
use crate::args::Args;
use crate::cmds;
use companion_engine::rag::{chunk_text, ChunkOptions, CitedContext, Embedder, HashEmbedder, HnswParams, KnowledgeBase, ModelEmbedder, Pooling, SearchOptions};
use std::path::{Path, PathBuf};

const MAX_FILE: u64 = 8 << 20;
const DIM: usize = 512;

fn docs_dir(lib: &Path) -> PathBuf { lib.join("docs") }

fn safe_name(p: &Path) -> Result<String, String> {
    let n = p.file_name().map(|n| n.to_string_lossy().into_owned()).ok_or_else(|| format!("{}: no file name", p.display()))?;
    if n.starts_with('.') || n.contains(['/', '\\']) { return Err(format!("{}: unsuitable file name", p.display())); }
    Ok(n)
}

fn collect(p: &Path, out: &mut Vec<PathBuf>, depth: usize) -> Result<(), String> {
    if p.is_dir() {
        if depth > 8 { return Ok(()); }
        let mut v: Vec<PathBuf> = std::fs::read_dir(p).map_err(|e| format!("{}: {e}", p.display()))?.filter_map(|e| e.ok().map(|e| e.path())).collect();
        v.sort();
        for c in v { if !c.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')) { collect(&c, out, depth + 1)?; } }
    } else { out.push(p.to_path_buf()); }
    Ok(())
}

pub fn add(lib: &Path, files: &[String]) -> Result<(usize, usize), String> {
    let dd = docs_dir(lib);
    std::fs::create_dir_all(&dd).map_err(|e| format!("{}: {e}", dd.display()))?;
    let (mut added, mut skipped) = (0, 0);
    let mut all = vec![];
    for f in files { collect(Path::new(f), &mut all, 0)?; }
    for p in all {
        let ok = std::fs::metadata(&p).map(|m| m.is_file() && m.len() <= MAX_FILE).unwrap_or(false);
        let text = if ok { std::fs::read(&p).ok().and_then(|b| String::from_utf8(b).ok()) } else { None };
        match (text, safe_name(&p)) {
            (Some(t), Ok(n)) if !t.trim().is_empty() => { std::fs::write(dd.join(&n), t).map_err(|e| format!("{n}: {e}"))?; added += 1; }
            _ => { skipped += 1; eprintln!("skipped {} (not UTF-8 text, empty, or over 8 MiB)", p.display()); }
        }
    }
    std::fs::write(lib.join("library.json"), serde_json::json!({"format": "corbul-library-1", "embedder": format!("hash:{DIM}")}).to_string()).map_err(|e| e.to_string())?;
    Ok((added, skipped))
}

fn load_docs(lib: &Path) -> Result<Vec<(String, String)>, String> {
    let dd = docs_dir(lib);
    if !dd.is_dir() { return Err(format!("{}: not a library (create one with `corbul rag add {} FILE…`)", lib.display(), lib.display())); }
    let mut names: Vec<PathBuf> = std::fs::read_dir(&dd).map_err(|e| e.to_string())?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file()).collect();
    names.sort();
    let docs: Vec<(String, String)> = names.iter().filter_map(|p| Some((p.file_name()?.to_string_lossy().into_owned(), std::fs::read_to_string(p).ok()?))).collect();
    if docs.is_empty() { return Err(format!("{}: the library has no documents", lib.display())); }
    Ok(docs)
}

fn fill<'a>(embedder: Box<dyn Embedder + 'a>, docs: &[(String, String)], a: &Args) -> Result<KnowledgeBase<'a>, String> {
    let mut kb = KnowledgeBase::new(embedder, HnswParams::default()).map_err(|e| e.to_string())?;
    let co = ChunkOptions { max_chars: a.num("chunk-chars")?.unwrap_or(800), overlap_chars: a.num("overlap")?.unwrap_or(120) };
    if co.max_chars < 16 || co.overlap_chars >= co.max_chars { return Err("--chunk-chars must be at least 16 and larger than --overlap".into()); }
    for (n, t) in docs { if !chunk_text(t, &co).is_empty() { kb.add_document(n, t, &co).map_err(|e| format!("{n}: {e}"))?; } }
    if kb.chunk_count() == 0 { return Err("the documents contain no indexable text".into()); }
    Ok(kb)
}

fn search_opts(a: &Args, default_k: usize) -> Result<SearchOptions, String> {
    let k: usize = a.num("k")?.unwrap_or(default_k);
    if !(1..=100).contains(&k) { return Err("--k must be between 1 and 100".into()); }
    Ok(SearchOptions { k, candidates: (k * 8).max(50), ..SearchOptions::default() })
}

pub fn cmd(a: &Args) -> Result<(), String> {
    let sub = a.pos.first().map(String::as_str).unwrap_or("");
    let rest = Args { pos: a.pos.iter().skip(1).cloned().collect(), opts: a.opts.clone() };
    cmds::apply_global(&rest);
    match sub {
        "add" => {
            let lib = rest.pos.first().ok_or("rag add: give the library folder and the files")?;
            if rest.pos.len() < 2 { return Err("rag add: give at least one file or folder to add".into()); }
            let (n, s) = add(Path::new(lib), &rest.pos[1..])?;
            crate::outln!("added {n} document(s) to {lib}{}", if s > 0 { format!(" ({s} skipped)") } else { String::new() });
            Ok(())
        }
        "search" => search(&rest),
        "ask" => ask(&rest),
        _ => Err("rag: use `add LIB FILE…`, `search LIB --query Q` or `ask LIB MODEL --question Q`".into()),
    }
}

fn search(a: &Args) -> Result<(), String> {
    let lib = a.pos.first().ok_or("rag search: give the library folder")?;
    let q = a.get("query").map(String::from).or_else(|| a.pos.get(1).cloned()).ok_or("rag search: give --query TEXT")?;
    let docs = load_docs(Path::new(lib))?;
    let o = search_opts(a, 5)?;
    let hits = match a.get("embed-model") {
        None => fill(Box::new(HashEmbedder::new(DIM, 1)), &docs, a)?.search(&q, &o).map_err(|e| e.to_string())?,
        Some(m) => {
            let (prep, cfg) = cmds::open_prepared(&Args { pos: vec![m.to_string()], opts: a.opts.clone() })?;
            let model = cmds::build_model_with_threads(&prep, &cfg, a)?;
            eprintln!("embedding the library with {} (this repeats on every search)", prep.name);
            fill(Box::new(ModelEmbedder::new(&model, &prep.tok, Pooling::Mean)), &docs, a)?.search(&q, &o).map_err(|e| e.to_string())?
        }
    };
    if a.flag("json") {
        let j: Vec<_> = hits.iter().enumerate().map(|(i, h)| serde_json::json!({"rank": i + 1, "document": h.doc_name, "score": h.score, "byte_start": h.byte_start, "byte_end": h.byte_end, "text": h.text})).collect();
        crate::outln!("{}", serde_json::to_string_pretty(&j).unwrap_or_default());
    } else {
        if hits.is_empty() { crate::outln!("no passages found"); }
        for (i, h) in hits.iter().enumerate() { crate::outln!("[{}] {} (bytes {}–{}, score {:.3})\n{}\n", i + 1, h.doc_name, h.byte_start, h.byte_end, h.score, h.text.trim()); }
    }
    Ok(())
}

fn ask(a: &Args) -> Result<(), String> {
    let (lib, model) = (a.pos.first().ok_or("rag ask: give the library folder and the model")?, a.pos.get(1).ok_or("rag ask: give the model after the library folder")?);
    let q = a.get("question").map(String::from).or_else(|| a.pos.get(2).cloned()).ok_or("rag ask: give --question TEXT")?;
    let docs = load_docs(Path::new(lib))?;
    let o = search_opts(a, 4)?;
    let hits = fill(Box::new(HashEmbedder::new(DIM, 1)), &docs, a)?.search(&q, &o).map_err(|e| e.to_string())?;
    if hits.is_empty() { return Err("nothing in the library matches the question; not answering from thin air".into()); }
    let ctx = CitedContext::from_hits(hits);
    for s in &ctx.sources { eprintln!("[S{}] {} (bytes {}–{})", s.tag, s.doc_name, s.byte_start, s.byte_end); }
    let sources: Vec<String> = ctx.sources.iter().map(|s| s.text.clone()).collect();
    let margs = Args { pos: vec![model.clone()], opts: a.opts.iter().filter(|(k, _)| !matches!(k.as_str(), "question" | "k" | "chunk-chars" | "overlap" | "embed-model")).cloned().collect() };
    if crate::famrun::wants(Path::new(model)) { return Err("rag ask needs a dense model (state-space and hybrid models have no quote lock)".into()); }
    let (prep, cfg) = cmds::open_prepared(&margs)?;
    cmds::generate_run(&margs, &prep, &cfg, q, sources)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;
    fn a(s: &str) -> Args { Args::parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>()) }

    #[test]
    fn add_search_and_ask() {
        let d = tmpdir("rag");
        let lib = d.join("lib");
        let docs = d.join("in");
        std::fs::create_dir_all(&docs).unwrap();
        std::fs::write(docs.join("a.txt"), "The harbour opens at six in the morning. Ferries leave every half hour. Tickets cost four euros.").unwrap();
        std::fs::write(docs.join("b.md"), "Mamaliga se face din malai si apa. Se amesteca pana se ingroasa. Se serveste cu branza.").unwrap();
        std::fs::write(docs.join("bin.dat"), [0xff, 0xfe, 0x00, 0x80]).unwrap();
        cmd(&a(&format!("add {} {}", lib.display(), docs.display()))).unwrap();
        assert!(lib.join("docs/a.txt").is_file() && !lib.join("docs/bin.dat").exists());
        cmd(&a(&format!("search {} --query ferries --k 2", lib.display()))).unwrap();
        cmd(&a(&format!("search {} --query mamaliga --json", lib.display()))).unwrap();
        assert!(cmd(&a(&format!("search {} --query x --k 0", lib.display()))).is_err());
        assert!(cmd(&a(&format!("search {}/nothing --query x", d.display()))).is_err());
        assert!(cmd(&a("add")).is_err());
        let m = d.join("m");
        std::fs::create_dir_all(&m).unwrap();
        write_model_ctx(&m, true, 1024);
        let r = cmd(&a(&format!("ask {} {} --question ferries --k 1 --max-tokens 4 --temperature 0 --quiet", lib.display(), m.display())));
        if let Err(e) = &r { panic!("ask failed: {e}"); }
        std::fs::remove_dir_all(d).unwrap();
    }
}
