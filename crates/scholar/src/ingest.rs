// Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//! Reading books and articles (pdf, docx, epub, txt, md, html, tex) into page-tagged passages.
//!
//! Everything works on bytes, without network access or external programs. Each reader is bounded:
//! archive members and whole archives are size-capped (zip bombs), member names are only ever used
//! as lookup keys (never as file-system paths, so zip-slip cannot write anything), and PDFs are
//! text-layer only (see [`crate::pdfx`]).

use crate::index::{Chunk, Doc, Error, Result, Store};
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

/// Words per passage.
pub const CHUNK_WORDS: usize = 200;
/// Words shared between consecutive passages.
pub const OVERLAP_WORDS: usize = 40;
/// Largest input file read, in bytes.
pub const MAX_FILE: u64 = 512 << 20;
/// Largest text a single archive member may expand to (zip bombs).
pub const MAX_MEMBER: u64 = 64 << 20;
/// Largest total text read out of one archive.
pub const MAX_ARCHIVE_TEXT: u64 = 256 << 20;
/// Most entries an archive may declare.
pub const MAX_ENTRIES: usize = 20_000;
/// Most text kept from one document, in bytes.
pub const MAX_DOC_TEXT: usize = 256 << 20;
/// Deepest directory level `collect_files` descends.
const MAX_DEPTH: usize = 64;

/// True for file types this crate can read.
pub fn is_supported(p: &Path) -> bool {
    matches!(
        ext_of(p).as_str(),
        "pdf" | "docx" | "epub" | "txt" | "md" | "markdown" | "html" | "htm" | "tex"
    )
}

fn ext_of(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
}

/// Supported files under `paths` (files as given, directories recursively), sorted and de-duplicated.
/// Symbolic links to directories are not followed, so a link cycle cannot loop.
pub fn collect_files(paths: &[PathBuf]) -> Vec<PathBuf> {
    fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        if depth > MAX_DEPTH {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.flatten() {
            let p = e.path();
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                walk(&p, depth + 1, out);
            } else if (ft.is_file() || (ft.is_symlink() && p.is_file())) && is_supported(&p) {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            walk(p, 0, &mut out);
        } else if p.is_file() {
            out.push(p.clone());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// 64-bit FNV-1a hash (change detection, not security).
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// --------------------------------------------------------------------- html

/// Markup to text. A `<` opens a tag only when a letter, `/`, `!` or `?` follows, so "a < b" survives;
/// `<script>` and `<style>` bodies are dropped; common entities are decoded once.
pub(crate) fn strip_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let lower = s.to_ascii_lowercase(); // ASCII-only: byte offsets identical to `s`
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut in_tag = false;
    while i < bytes.len() {
        if !in_tag {
            for blk in ["<script", "<style"] {
                if lower[i..].starts_with(blk) {
                    let close = format!("</{}", &blk[1..]);
                    i = lower[i..]
                        .find(&close)
                        .map(|j| i + j)
                        .unwrap_or(bytes.len());
                    break;
                }
            }
            if i >= bytes.len() {
                break;
            }
        }
        let c = match s[i..].chars().next() {
            Some(c) => c,
            None => break,
        };
        match c {
            '<' if !in_tag => {
                let next = bytes.get(i + 1).copied().unwrap_or(b' ');
                if next.is_ascii_alphabetic() || matches!(next, b'/' | b'!' | b'?') {
                    in_tag = true;
                } else {
                    out.push('<');
                }
            }
            '>' if in_tag => {
                in_tag = false;
                out.push(' ');
            }
            _ if !in_tag => out.push(c),
            _ => {}
        }
        i += c.len_utf8();
    }
    decode_entities(&out)
}

pub(crate) fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        // only the next 11 bytes can hold the ';' of an entity: a longer search is quadratic on "&&&&…"
        let window = &rest.as_bytes()[..rest.len().min(11)];
        match window.iter().position(|&b| b == b';') {
            Some(j) => {
                let ent = &rest[1..j];
                let rep: Option<String> = match ent {
                    "amp" => Some("&".into()),
                    "lt" => Some("<".into()),
                    "gt" => Some(">".into()),
                    "quot" => Some("\"".into()),
                    "apos" => Some("'".into()),
                    "nbsp" => Some(" ".into()),
                    "shy" => Some(String::new()),
                    "hellip" => Some("…".into()),
                    "mdash" => Some("—".into()),
                    "ndash" => Some("–".into()),
                    "lsquo" => Some("‘".into()),
                    "rsquo" => Some("’".into()),
                    "ldquo" => Some("“".into()),
                    "rdquo" => Some("”".into()),
                    e if e.starts_with("#x") || e.starts_with("#X") => {
                        u32::from_str_radix(&e[2..], 16)
                            .ok()
                            .and_then(printable)
                            .map(String::from)
                    }
                    e if e.starts_with('#') => e[1..]
                        .parse::<u32>()
                        .ok()
                        .and_then(printable)
                        .map(String::from),
                    _ => None,
                };
                match rep {
                    Some(r) => {
                        out.push_str(&r);
                        rest = &rest[j + 1..];
                    }
                    None => {
                        out.push('&');
                        rest = &rest[1..];
                    }
                }
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// A character reference's value, refusing NUL and other control characters.
fn printable(n: u32) -> Option<char> {
    char::from_u32(n).filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
}

// ---------------------------------------------------------------------- tex

/// LaTeX source to readable text: comments and the preamble dropped, section commands kept as
/// plain lines, `\textbf{x}`-style wrappers reduced to their argument, references, labels and
/// package lines removed, math left as written. Not a TeX engine; it only has to leave words
/// intact for search.
pub fn strip_tex(src: &str) -> String {
    let body = match src.find("\\begin{document}") {
        Some(i) => &src[i + "\\begin{document}".len()..],
        None => src,
    };
    let body = body.split("\\end{document}").next().unwrap_or(body);
    let mut text = String::with_capacity(body.len());
    for line in body.lines() {
        // an unescaped % starts a comment
        let mut cut = line.len();
        let mut prev = '\0';
        for (i, c) in line.char_indices() {
            if c == '%' && prev != '\\' {
                cut = i;
                break;
            }
            prev = c;
        }
        text.push_str(&line[..cut]);
        text.push('\n');
    }
    const DROP: [&str; 14] = [
        "label",
        "ref",
        "eqref",
        "cite",
        "citep",
        "citet",
        "usepackage",
        "documentclass",
        "bibliographystyle",
        "bibliography",
        "includegraphics",
        "pagestyle",
        "vspace",
        "hspace",
    ];
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' && i + 1 < chars.len() {
            let n = chars[i + 1];
            if n == '\\' {
                out.push('\n');
                i += 2;
                continue;
            }
            if !n.is_ascii_alphabetic() {
                // escaped symbol such as \% \& \_ \{ : keep the symbol
                if matches!(n, '%' | '&' | '_' | '#' | '$' | '{' | '}') {
                    out.push(n);
                }
                i += 2;
                continue;
            }
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_alphabetic() {
                j += 1;
            }
            let name: String = chars[i + 1..j].iter().collect();
            if j < chars.len() && chars[j] == '*' {
                j += 1;
            }
            // skip optional [..] arguments
            while j < chars.len() && chars[j] == '[' {
                match chars[j..].iter().position(|&x| x == ']') {
                    Some(k) => j += k + 1,
                    None => break,
                }
            }
            if matches!(
                name.as_str(),
                "section" | "subsection" | "subsubsection" | "chapter" | "part" | "paragraph"
            ) {
                out.push_str("\n\n");
            }
            if matches!(name.as_str(), "par" | "item" | "newline" | "noindent") {
                out.push('\n');
            }
            if DROP.contains(&name.as_str()) && j < chars.len() && chars[j] == '{' {
                let mut depth = 0usize;
                let mut k = j;
                while k < chars.len() {
                    match chars[k] {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                k += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }
                i = k;
                continue;
            }
            if matches!(name.as_str(), "begin" | "end") && j < chars.len() && chars[j] == '{' {
                // environment name is not text
                match chars[j..].iter().position(|&x| x == '}') {
                    Some(k) => {
                        out.push('\n');
                        i = j + k + 1;
                    }
                    None => i = j,
                }
                continue;
            }
            i = j;
            continue;
        }
        if c == '{' || c == '}' || c == '~' {
            out.push(' ');
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out.lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------- zip

type Zip<'a> = zip::ZipArchive<Cursor<&'a [u8]>>;

/// A zip archive with a running budget of text read out of it.
pub(crate) struct Archive<'a> {
    z: Zip<'a>,
    spent: u64,
}

impl<'a> Archive<'a> {
    pub(crate) fn open(bytes: &'a [u8]) -> Result<Archive<'a>> {
        let z = zip::ZipArchive::new(Cursor::new(bytes))
            .map_err(|e| Error::Format(format!("not a valid zip container: {e}")))?;
        if z.len() > MAX_ENTRIES {
            return Err(Error::Limit(format!(
                "archive declares {} entries",
                z.len()
            )));
        }
        Ok(Archive { z, spent: 0 })
    }

    /// Member names (never used as paths).
    pub(crate) fn names(&mut self) -> Vec<String> {
        let mut v = Vec::new();
        for i in 0..self.z.len() {
            if let Ok(f) = self.z.by_index_raw(i) {
                if let Ok(n) = f.name() {
                    v.push(n.to_string());
                }
            }
        }
        v
    }

    /// Member text. Errors on invalid UTF-8, on a member over `MAX_MEMBER` once expanded, and on
    /// an archive whose total exceeds `MAX_ARCHIVE_TEXT`.
    pub(crate) fn read(&mut self, name: &str) -> Result<String> {
        let mut f = self
            .z
            .by_name(name)
            .map_err(|e| Error::Format(format!("{name}: {e}")))?;
        if f.size() > MAX_MEMBER {
            return Err(Error::Limit(format!("{name} declares {} bytes", f.size())));
        }
        let mut buf = Vec::new();
        Read::read_to_end(&mut Read::take(&mut f, MAX_MEMBER + 1), &mut buf)
            .map_err(|e| Error::Format(format!("{name}: {e}")))?;
        if buf.len() as u64 > MAX_MEMBER {
            return Err(Error::Limit(format!(
                "{name} expands to more than {} MB (possible zip bomb)",
                MAX_MEMBER >> 20
            )));
        }
        self.spent += buf.len() as u64;
        if self.spent > MAX_ARCHIVE_TEXT {
            return Err(Error::Limit("archive expands past the total cap".into()));
        }
        String::from_utf8(buf).map_err(|_| Error::Format(format!("{name} is not valid UTF-8")))
    }

    pub(crate) fn read_opt(&mut self, name: &str) -> Option<String> {
        self.read(name).ok()
    }
}

/// Word document to plain text, one line per paragraph.
fn docx_text(bytes: &[u8]) -> Result<String> {
    let mut z = Archive::open(bytes)?;
    let xml = z.read("word/document.xml").map_err(|e| match e {
        Error::Limit(_) => e,
        _ => Error::Format("word/document.xml missing or unreadable".into()),
    })?;
    let xml = xml
        .replace("</w:p>", "\n")
        .replace("<w:tab/>", " ")
        .replace("<w:br/>", "\n");
    Ok(strip_html(&xml)
        .lines()
        .map(|l| l.trim())
        .collect::<Vec<_>>()
        .join("\n"))
}

/// Value of `name="…"` (or single-quoted) inside one opening tag.
pub(crate) fn attr(tag: &str, name: &str) -> Option<String> {
    for q in ['"', '\''] {
        let pat = format!("{name}={q}");
        if let Some(i) = tag.find(&pat) {
            let rest = &tag[i + pat.len()..];
            if let Some(j) = rest.find(q) {
                return Some(rest[..j].to_string());
            }
        }
    }
    None
}

/// Text between `<name` and `>` for every opening tag called `name` (also `opf:name`).
pub(crate) fn open_tags<'a>(xml: &'a str, name: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    for open in [format!("<{name}"), format!("<opf:{name}")] {
        let mut rest = xml;
        while let Some(i) = rest.find(&open) {
            let after = &rest[i + open.len()..];
            if after.starts_with(|c: char| c.is_whitespace() || c == '/' || c == '>')
                && let Some(j) = after.find('>')
            {
                out.push(&after[..j]);
                rest = &after[j..];
                continue;
            }
            rest = after;
        }
    }
    out
}

pub(crate) fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            // bytes, not &str slices: slicing could split a multi-byte character
            let hex = std::str::from_utf8(&b[i + 1..i + 3])
                .ok()
                .and_then(|h| u8::from_str_radix(h, 16).ok());
            if let Some(v) = hex {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

/// Joins `dir` and `href`, resolving `.` / `..` and dropping any `#fragment`. `..` above the
/// archive root is ignored, so the result can only name something inside the archive.
pub(crate) fn resolve_href(dir: &str, href: &str) -> String {
    let href = percent_decode(href.split('#').next().unwrap_or(""));
    let mut parts: Vec<&str> = dir.split('/').filter(|p| !p.is_empty()).collect();
    for seg in href.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            x => parts.push(x),
        }
    }
    parts.join("/")
}

/// Content documents in the book's own reading order (OPF spine), navigation documents skipped.
fn epub_order(z: &mut Archive<'_>) -> Option<Vec<String>> {
    let container = z.read_opt("META-INF/container.xml")?;
    let opf_path = open_tags(&container, "rootfile")
        .iter()
        .find_map(|t| attr(t, "full-path"))?;
    let opf = z.read_opt(&opf_path)?;
    let dir = opf_path
        .rsplit_once('/')
        .map(|(d, _)| d.to_string())
        .unwrap_or_default();
    let mut manifest: HashMap<String, (String, bool)> = HashMap::new();
    for t in open_tags(&opf, "item") {
        if let (Some(id), Some(href)) = (attr(t, "id"), attr(t, "href")) {
            let nav = attr(t, "properties")
                .map(|p| p.split_whitespace().any(|w| w == "nav"))
                .unwrap_or(false);
            manifest.insert(id, (href, nav));
        }
    }
    let order: Vec<String> = open_tags(&opf, "itemref")
        .iter()
        .filter_map(|t| attr(t, "idref"))
        .filter_map(|id| manifest.get(&id))
        .filter(|(_, nav)| !nav)
        .map(|(href, _)| resolve_href(&dir, href))
        .collect();
    if order.is_empty() { None } else { Some(order) }
}

/// EPUB to the text of its content documents in reading order (spine); if the package file cannot
/// be understood, falls back to file-name order.
fn epub_text(bytes: &[u8]) -> Result<String> {
    let mut z = Archive::open(bytes)?;
    if let Some(order) = epub_order(&mut z) {
        let text = order
            .iter()
            .filter_map(|n| z.read_opt(n))
            .map(|x| strip_html(&x))
            .collect::<Vec<_>>()
            .join("\n\n");
        if !text.trim().is_empty() {
            return Ok(text);
        }
    }
    let mut names: Vec<String> = z
        .names()
        .into_iter()
        .filter(|n| {
            let l = n.to_lowercase();
            (l.ends_with(".xhtml") || l.ends_with(".html") || l.ends_with(".htm"))
                && !l.contains("toc")
                && !l.contains("nav")
        })
        .collect();
    names.sort();
    let mut parts = Vec::new();
    for n in names {
        parts.push(strip_html(&z.read(&n)?));
    }
    Ok(parts.join("\n\n"))
}

// -------------------------------------------------------------------- pages

/// `(page number, text)` for a document given its bytes and extension (without the dot). Page 0
/// means the format has no pages. Plain text with form feeds is split into numbered pages.
pub fn read_pages_bytes(ext: &str, bytes: &[u8]) -> Result<Vec<(usize, String)>> {
    if bytes.len() as u64 > MAX_FILE {
        return Err(Error::Limit(format!("file is {} bytes", bytes.len())));
    }
    let ext = ext.trim_start_matches('.').to_lowercase();
    let pages = match ext.as_str() {
        "pdf" => crate::pdfx::text_pages(bytes)?
            .into_iter()
            .enumerate()
            .map(|(i, t)| (i + 1, t))
            .collect(),
        "docx" => vec![(0, docx_text(bytes)?)],
        "epub" => match crate::epubpages::read(bytes) {
            Some(p) => p,
            None => vec![(0, epub_text(bytes)?)],
        },
        _ => {
            let text = String::from_utf8_lossy(bytes).to_string();
            match ext.as_str() {
                "html" | "htm" => vec![(0, strip_html(&text))],
                "tex" => vec![(0, strip_tex(&text))],
                _ if text.contains('\x0c') => text
                    .split('\x0c')
                    .enumerate()
                    .map(|(i, t)| (i + 1, t.to_string()))
                    .collect(),
                _ => vec![(0, text)],
            }
        }
    };
    let total: usize = pages.iter().map(|p| p.1.len()).sum();
    if total > MAX_DOC_TEXT {
        return Err(Error::Limit(format!("document text is {total} bytes")));
    }
    Ok(pages)
}

/// As [`read_pages_bytes`], reading the file at `path` (extension decides the format).
pub fn read_pages(path: &Path) -> Result<Vec<(usize, String)>> {
    let bytes = read_file(path)?;
    read_pages_bytes(&ext_of(path), &bytes)
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    let len = std::fs::metadata(path)?.len();
    if len > MAX_FILE {
        return Err(Error::Limit(format!("{} is {len} bytes", path.display())));
    }
    Ok(std::fs::read(path)?)
}

// ----------------------------------------------------------------- chunking

/// A word that plausibly ends a sentence ("…proof." / "…?" / "…!"), ignoring closing quotes and
/// brackets. Very short tokens ("p.", "1.") do not count, which avoids most abbreviation cuts.
fn ends_sentence(w: &str) -> bool {
    let t = w.trim_end_matches(['"', '\'', ')', ']', '”', '’', '»']);
    t.chars().count() > 2 && t.ends_with(['.', '!', '?', '…'])
}

/// Joins `word-<newline>rest` when a letter precedes the hyphen and a lower-case letter follows the
/// line break; a hyphen before a digit or a capital is kept (list item, dash, compound with a name).
fn rejoin_hyphenated(text: &str) -> String {
    let ch: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < ch.len() {
        if ch[i] == '-'
            && ch.get(i + 1) == Some(&'\n')
            && i > 0
            && ch[i - 1].is_alphabetic()
            && ch.get(i + 2).is_some_and(|c| c.is_lowercase())
        {
            i += 2;
            continue;
        }
        out.push(ch[i]);
        i += 1;
    }
    out
}

/// Splits text into overlapping word windows of about `size` words. Windows end at a sentence
/// boundary when one lies in the last 30% of the window, and the overlap starts at a sentence
/// start when possible, so passages read as whole sentences. Text with no punctuation falls back
/// to plain fixed windows. Always terminates: each window starts strictly after the previous one.
pub fn chunk_words(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let text = rejoin_hyphenated(text);
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = words.len();
    if n == 0 {
        return vec![];
    }
    let size = size.max(1);
    let mut out = Vec::new();
    let mut i = 0;
    loop {
        let hard = (i + size).min(n);
        let mut end = hard;
        if hard < n {
            let lo = i + size * 7 / 10;
            if let Some(j) = (lo..hard).rev().find(|&j| ends_sentence(words[j])) {
                end = j + 1;
            }
        }
        out.push(words[i..end].join(" "));
        if end == n {
            break;
        }
        let mut start = end.saturating_sub(overlap).max(i + 1);
        if let Some(k) = (start..end).find(|&k| k > 0 && ends_sentence(words[k - 1])) {
            start = k;
        }
        i = start;
    }
    // Drop a tiny trailing fragment that is entirely overlap.
    if out.len() > 1
        && out
            .last()
            .map(|c| c.split_whitespace().count() <= overlap)
            .unwrap_or(false)
    {
        out.pop();
    }
    out
}

// ---------------------------------------------------------------- documents

/// Title from a file name: stem with `_` and `-` as spaces.
pub fn pretty_title(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled")
        .replace(['_', '-'], " ")
        .trim()
        .to_string()
}

/// What `ingest_file` / `ingest_bytes` did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// New or refreshed document with this many passages.
    Added {
        /// Passages stored.
        chunks: usize,
    },
    /// Same content as before; nothing changed.
    Unchanged,
    /// The file held no extractable text (for instance a scan without a text layer).
    NoText,
}

/// Canonical path without the Windows `\\?\` verbatim prefix.
fn canonical(p: &Path) -> PathBuf {
    let c = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    match c.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
        _ => c,
    }
}

/// Identity of a document inside the store. Files under `base` are stored relative to it with `/`
/// separators, so the whole folder can move to another drive or machine. Others keep absolute paths.
pub fn doc_key(path: &Path, base: &Path) -> String {
    let canon = canonical(path);
    let b = canonical(base);
    match canon.strip_prefix(&b) {
        Ok(rel) if !rel.as_os_str().is_empty() => rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("/"),
        _ => canon.to_string_lossy().to_string(),
    }
}

/// Adds or refreshes one file. `base` is the folder the store lives in (see [`doc_key`]).
/// `force` re-reads files even when unchanged.
pub fn ingest_file(store: &mut Store, path: &Path, base: &Path, force: bool) -> Result<Outcome> {
    let bytes = read_file(path)?;
    ingest_bytes(
        store,
        &doc_key(path, base),
        &pretty_title(path),
        &ext_of(path),
        &bytes,
        force,
    )
}

/// Adds or refreshes one document given its bytes, a unique `key`, a title and the extension that
/// selects the reader. The SHA-256 of the bytes is recorded for provenance receipts.
pub fn ingest_bytes(
    store: &mut Store,
    key: &str,
    title: &str,
    ext: &str,
    bytes: &[u8],
    force: bool,
) -> Result<Outcome> {
    let hash = fnv1a(bytes);
    let sha = crate::sha256::sha256_hex(bytes);
    let existing = store.doc_by_path(key);
    if let Some(id) = existing
        && store.docs[id].hash == hash
        && !force
    {
        return Ok(Outcome::Unchanged);
    }
    // read first: a failing refresh must not lose the previous version
    let pages = read_pages_bytes(ext, bytes)?;
    // PDF-like: pages are 1..n. Formats without pages: 0. An EPUB with declared pages starts with an
    // unpaged front part, and its page count is the largest printed number found.
    let npages = if pages.first().map(|p| p.0).unwrap_or(0) == 0 {
        pages.iter().map(|p| p.0).max().unwrap_or(0)
    } else {
        pages.len()
    };
    let mut new_chunks = Vec::new();
    for (page, text) in pages {
        for c in crate::pdfx::chunk_page(&text, CHUNK_WORDS, OVERLAP_WORDS, &chunk_words) {
            new_chunks.push(Chunk {
                doc: 0, // set below, once the document's id is known
                page,
                text: c,
                emb: None,
            });
        }
    }
    if new_chunks.is_empty() {
        return Ok(Outcome::NoText);
    }
    if let Some(id) = existing {
        store.remove_doc(id);
    }
    // removal shifts ids: the new document is always appended last
    let doc_id = store.docs.len();
    for c in &mut new_chunks {
        c.doc = doc_id;
    }
    let n = new_chunks.len();
    store.sha.insert(key.to_string(), (hash, sha));
    store.docs.push(Doc {
        path: key.to_string(),
        title: title.to_string(),
        hash,
        pages: npages,
    });
    store.chunks.extend(new_chunks);
    Ok(Outcome::Added { chunks: n })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cs-ing-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    pub(crate) fn make_zip(files: &[(&str, &str)]) -> Vec<u8> {
        use std::io::Write;
        let mut z = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (n, c) in files {
            z.start_file(*n, zip::write::SimpleFileOptions::default())
                .unwrap();
            z.write_all(c.as_bytes()).unwrap();
        }
        z.finish().unwrap().into_inner()
    }

    #[test]
    fn ingest_records_sha256_identity() {
        let dir = tmp("sha");
        let f = dir.join("a.txt");
        std::fs::write(&f, "abc and some more words to index here").unwrap();
        let mut st = Store::default();
        ingest_file(&mut st, &f, &dir, false).unwrap();
        assert_eq!(
            st.doc_sha256(0).unwrap(),
            crate::sha256::sha256_hex(b"abc and some more words to index here")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn chunking_overlaps_and_covers() {
        let text = (0..450)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let c = chunk_words(&text, 200, 40);
        assert_eq!(c.len(), 3);
        assert!(c[0].starts_with("w0 "));
        assert!(c[1].starts_with("w160 "));
        assert!(c[2].ends_with("w449"));
    }

    #[test]
    fn keys_are_relative_under_base_and_absolute_elsewhere() {
        let root = tmp("keys");
        std::fs::create_dir_all(root.join("Data/Library/sub")).unwrap();
        std::fs::create_dir_all(root.join("elsewhere")).unwrap();
        let inside = root.join("Data/Library/sub/book.txt");
        let outside = root.join("elsewhere/paper.txt");
        std::fs::write(&inside, "alpha beta gamma").unwrap();
        std::fs::write(&outside, "delta").unwrap();
        let base = root.join("Data");
        assert_eq!(doc_key(&inside, &base), "Library/sub/book.txt");
        assert!(
            doc_key(&outside, &base)
                .replace('\\', "/")
                .ends_with("elsewhere/paper.txt")
        );
        assert!(Path::new(&doc_key(&outside, &base)).is_absolute());
        let mut st = Store::default();
        assert!(matches!(
            ingest_file(&mut st, &inside, &base, false).unwrap(),
            Outcome::Added { .. }
        ));
        assert_eq!(
            ingest_file(&mut st, &inside, &base, false).unwrap(),
            Outcome::Unchanged
        );
        assert_eq!(st.docs[0].path, "Library/sub/book.txt");
        assert!(matches!(
            ingest_file(&mut st, &inside, &base, true).unwrap(),
            Outcome::Added { .. }
        ));
        assert_eq!(st.docs.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn refresh_keeps_ids_consistent_with_other_documents() {
        let mut st = Store::default();
        ingest_bytes(&mut st, "a", "A", "txt", b"alpha one two three", false).unwrap();
        ingest_bytes(&mut st, "b", "B", "txt", b"beta four five six", false).unwrap();
        ingest_bytes(&mut st, "a", "A", "txt", b"alpha changed text now", false).unwrap();
        assert_eq!(st.docs.len(), 2);
        st.validate().unwrap();
        for c in &st.chunks {
            let d = &st.docs[c.doc];
            assert!(
                (d.path == "a" && c.text.contains("changed"))
                    || (d.path == "b" && c.text.contains("beta"))
            );
        }
        // a failing refresh keeps the previous version
        assert!(ingest_bytes(&mut st, "b", "B", "pdf", b"garbage", false).is_err());
        assert_eq!(st.docs.len(), 2);
        st.validate().unwrap();
    }

    #[test]
    fn chunks_end_on_sentence_boundaries() {
        let text = (0..60)
            .map(|i| format!("Sentence number {i} says something about the topic at hand today."))
            .collect::<Vec<_>>()
            .join(" ");
        let c = chunk_words(&text, 100, 20);
        assert!(c.len() > 3);
        for ch in &c {
            assert!(
                ch.starts_with("Sentence"),
                "chunk must start at a sentence start: {ch:.40}"
            );
            assert!(ch.ends_with("today."), "chunk must end at a sentence end");
        }
        let joined = c.join(" ");
        for i in 0..60 {
            assert!(
                joined.contains(&format!("number {i} ")),
                "sentence {i} lost"
            );
        }
    }

    #[test]
    fn epub_follows_spine_not_file_names() {
        let e = make_zip(&[
            (
                "META-INF/container.xml",
                "<container><rootfiles><rootfile full-path=\"OPS/book.opf\" media-type=\"application/oebps-package+xml\"/></rootfiles></container>",
            ),
            (
                "OPS/book.opf",
                "<package><manifest><item id=\"n\" href=\"nav.xhtml\" properties=\"nav\"/><item id=\"a\" href=\"text/a%20first.xhtml\"/><item id=\"b\" href=\"text/b.xhtml#top\"/><item id=\"c\" href=\"../OPS/text/c.xhtml\"/></manifest><spine><itemref idref=\"n\"/><itemref idref=\"c\"/><itemref idref=\"a\"/><itemref idref=\"b\"/></spine></package>",
            ),
            ("OPS/nav.xhtml", "<p>NAVIGATION</p>"),
            ("OPS/text/a first.xhtml", "<p>ALPHA</p>"),
            ("OPS/text/b.xhtml", "<p>BRAVO</p>"),
            ("OPS/text/c.xhtml", "<p>CHARLIE</p>"),
        ]);
        let t = &read_pages_bytes("epub", &e).unwrap()[0].1;
        let (c, a, b) = (
            t.find("CHARLIE").unwrap(),
            t.find("ALPHA").unwrap(),
            t.find("BRAVO").unwrap(),
        );
        assert!(c < a && a < b, "spine order is C, A, B: {t:?}");
        assert!(!t.contains("NAVIGATION"));
    }

    #[test]
    fn percent_decoding_is_safe_on_odd_input() {
        assert_eq!(percent_decode("a%20b%C8%99"), "a bș");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%é9"), "%é9");
        assert_eq!(percent_decode("%zz"), "%zz");
    }

    #[test]
    fn dehyphenation_only_between_letters() {
        assert_eq!(rejoin_hyphenated("dip-\nlomas"), "diplomas");
        assert_eq!(rejoin_hyphenated("see-\nSmith"), "see-\nSmith");
        assert_eq!(rejoin_hyphenated("page 3-\n4"), "page 3-\n4");
        assert_eq!(rejoin_hyphenated("țară-\nîn"), "țarăîn");
        assert_eq!(
            chunk_words("recogni-\ntion of dip-\nlomas", 50, 10)[0],
            "recognition of diplomas"
        );
    }

    #[test]
    fn decodes_entities() {
        assert_eq!(
            decode_entities("a &amp; b &lt;c&gt; &#8211; &#x21; &bogus; &"),
            "a & b <c> – ! &bogus; &"
        );
        assert_eq!(decode_entities("x&#0;y&#1;z"), "x&#0;y&#1;z"); // control characters refused
    }

    #[test]
    fn reads_docx_and_epub_without_double_decoding() {
        let d = make_zip(&[(
            "word/document.xml",
            "<w:document><w:p><w:r><w:t>Prima &amp; a doua</w:t></w:r></w:p><w:p><w:r><w:t>Al doilea paragraf &amp;lt;b&amp;gt;</w:t></w:r></w:p></w:document>",
        )]);
        let t = &read_pages_bytes("docx", &d).unwrap()[0].1;
        assert!(t.contains("Prima & a doua") && t.contains("Al doilea paragraf"));
        assert!(t.contains("&lt;b&gt;"), "decoded exactly once: {t:?}");
        assert_eq!(t.lines().filter(|l| !l.is_empty()).count(), 2, "{t:?}");
        let e = make_zip(&[
            (
                "OEBPS/ch01.xhtml",
                "<html><body><p>Capitolul unu</p></body></html>",
            ),
            (
                "OEBPS/ch02.xhtml",
                "<html><body><p>Capitolul doi</p></body></html>",
            ),
            ("OEBPS/toc.xhtml", "<p>SKIPPED</p>"),
            ("OEBPS/cover.jpg", "x"),
        ]);
        let t = &read_pages_bytes("epub", &e).unwrap()[0].1;
        assert!(t.find("unu").unwrap() < t.find("doi").unwrap());
        assert!(!t.contains("SKIPPED"));
    }

    #[test]
    fn html_strip_drops_scripts_and_keeps_less_than() {
        let t = strip_html("<p>Hello</p><script>var x=1;</script><b>world</b>");
        assert!(t.contains("Hello") && t.contains("world") && !t.contains("var x"));
        let t = strip_html("<p>if a < b and c > d then</p>");
        assert!(t.contains("a < b and c > d"), "{t:?}");
        let t = strip_html("<style>p{x}</style>after <script>never closed");
        assert!(t.contains("after") && !t.contains("never"));
    }

    #[test]
    fn tex_reduces_to_words() {
        let t = strip_tex(
            "\\documentclass{article}\\usepackage{x}\n\\begin{document}\n\\section{Intro}\nThe \\textbf{burden} of proof~lies \\cite{a} here. % hidden\nCost is 50\\% off.\\label{eq}\n\\end{document}\ntrailing",
        );
        assert!(
            t.contains("Intro") && t.contains("burden of proof lies") && t.contains("50% off"),
            "{t:?}"
        );
        assert!(
            !t.contains("hidden")
                && !t.contains("documentclass")
                && !t.contains("trailing")
                && !t.contains("cite")
        );
    }

    #[test]
    fn form_feeds_make_text_pages() {
        let p = read_pages_bytes("txt", b"one\x0ctwo\x0cthree").unwrap();
        assert_eq!(p.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(read_pages_bytes("md", b"plain").unwrap()[0].0, 0);
    }

    #[test]
    fn invalid_utf8_and_empty_inputs_are_fine() {
        let p = read_pages_bytes("txt", b"caf\xff\xfe good").unwrap();
        assert!(p[0].1.contains("good"));
        let mut st = Store::default();
        assert_eq!(
            ingest_bytes(&mut st, "e", "E", "txt", b"", false).unwrap(),
            Outcome::NoText
        );
        assert_eq!(
            ingest_bytes(&mut st, "e", "E", "txt", b"  \n ", false).unwrap(),
            Outcome::NoText
        );
    }

    #[test]
    fn hostile_archives_are_refused_not_crashed_on() {
        // not a zip, truncated zip, zip bomb, entry naming a parent directory
        assert!(read_pages_bytes("docx", b"PK\x03\x04 nope").is_err());
        let good = make_zip(&[("word/document.xml", "<w:p><w:t>hi</w:t></w:p>")]);
        for cut in [4, 30, good.len() / 2, good.len() - 3] {
            assert!(read_pages_bytes("docx", &good[..cut]).is_err());
        }
        let big = "a".repeat((MAX_MEMBER as usize) + 1024);
        let bomb = make_zip(&[("word/document.xml", &big)]);
        assert!(
            bomb.len() < (MAX_MEMBER as usize) / 50,
            "deflate shrinks it a lot"
        );
        assert!(matches!(
            read_pages_bytes("docx", &bomb),
            Err(Error::Limit(_))
        ));
        let slip = make_zip(&[
            ("../../etc/evil.xhtml", "<p>zipslip</p>"),
            ("a.xhtml", "<p>ok</p>"),
        ]);
        let t = &read_pages_bytes("epub", &slip).unwrap()[0].1;
        assert!(t.contains("ok")); // names are lookup keys only; nothing is written anywhere
        assert_eq!(resolve_href("a", "../../../x.xhtml"), "x.xhtml");
        assert!(read_pages_bytes("epub", &make_zip(&[("a.xhtml", "\u{0}")])).is_ok());
    }

    #[test]
    fn random_bytes_never_panic_any_reader() {
        let mut x = 0xDEADBEEFCAFEF00Du64;
        for ext in ["txt", "md", "html", "tex", "docx", "epub", "pdf"] {
            for len in [0usize, 1, 7, 64, 1000] {
                let mut b = Vec::with_capacity(len);
                for _ in 0..len {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    b.push(x as u8);
                }
                let _ = read_pages_bytes(ext, &b);
            }
        }
    }

    #[test]
    fn deep_html_and_giant_lines_stay_linear() {
        let t = std::time::Instant::now();
        assert_eq!(decode_entities(&"&".repeat(1 << 20)).len(), 1 << 20);
        assert!(t.elapsed().as_secs() < 5, "entity decoding is quadratic");
        let deep = "<div>".repeat(200_000) + "text" + &"</div>".repeat(200_000);
        assert!(strip_html(&deep).contains("text"));
        let line = "word ".repeat(400_000);
        assert!(!chunk_words(&line, 200, 40).is_empty());
        let nospace = "x".repeat(5_000_000);
        assert_eq!(chunk_words(&nospace, 200, 40).len(), 1);
    }

    #[test]
    fn chunk_words_properties() {
        let mut x = 0x2545F4914F6CDD1Du64;
        for _ in 0..300 {
            let mut s = String::new();
            let n = {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % 700) as usize
            };
            for _ in 0..n {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                s.push_str(["alpha", "beta.", "gamma", "delta?", "x-\nyz"][(x % 5) as usize]);
                s.push(' ');
            }
            let size = 1 + (x as usize >> 8) % 60;
            let overlap = (x as usize >> 16) % 30;
            let c = chunk_words(&s, size, overlap);
            if s.split_whitespace().next().is_none() {
                assert!(c.is_empty());
                continue;
            }
            assert!(!c.is_empty());
            assert!(c.iter().all(|w| !w.trim().is_empty()));
            assert!(c.iter().all(|w| w.split_whitespace().count() <= size));
        }
    }
}
