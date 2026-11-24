//! Clean-room tokenizer: byte-level BPE (GPT-2 / Llama-3 / Qwen) and SentencePiece-style BPE
//! (Llama-2 / Mistral / Gemma), read from a Hugging Face `tokenizer.json` or from GGUF metadata.
//!
//! Written from the public descriptions of the formats only. Dependencies: `serde_json`.
//!
//! * Added / special tokens are matched in the raw text first (leftmost, longest), then the
//!   remaining text is pre-tokenised and merged. Merging uses a rank-ordered priority queue over a
//!   linked list of symbols, so it is `O(n log n)` even for a single enormous "word".
//! * The pre-tokenizer is hand-written (no regex engine) and reproduces the GPT-2 pattern and the
//!   Llama-3 / Qwen-2 patterns, chosen by looking at the regex the model file declares.
//! * Nothing here panics on hostile input: malformed JSON, absurd ids, truncated GGUF arrays and
//!   invalid UTF-8 produced by decoding are all handled.
//!
//! Known limits: no Unicode normalisation (Qwen's NFC normaliser is skipped), regexes other than
//! the three families above are approximated by the closest one, WordPiece/Unigram JSON models are
//! rejected, GGUF token arrays longer than 2^20 entries are truncated by the GGUF reader, and in
//! SentencePiece mode the dummy-prefix `▁` is added only at the very start of the text.
use crate::gguf::{Gguf, Value as GValue};
use serde_json::Value;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

const SPACE: char = '\u{2581}';
const MAX_ID_HARD: u64 = 1 << 22;

#[derive(Debug, PartialEq, Eq)]
pub enum TokError { Json(String), Unsupported(String), Invalid(String) }
impl std::fmt::Display for TokError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokError::Json(s) => write!(f, "tokenizer file is not valid JSON: {s}"),
            TokError::Unsupported(s) => write!(f, "tokenizer type not supported: {s}"),
            TokError::Invalid(s) => write!(f, "tokenizer file is inconsistent: {s}"),
        }
    }
}
impl std::error::Error for TokError {}
fn invalid<T>(s: impl Into<String>) -> Result<T, TokError> { Err(TokError::Invalid(s.into())) }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Bytes mapped to printable characters (GPT-2 alphabet); lossless for any input.
    ByteLevel,
    /// `▁` marks spaces; characters missing from the vocabulary fall back to `<0xNN>` bytes.
    SentencePiece,
}

/// Which pre-tokenisation pattern to emulate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreCfg {
    /// Llama-3 style (`[^\r\n\p{L}\p{N}]?\p{L}+`, newline handling) instead of GPT-2 style.
    pub new_style: bool,
    /// Contractions (`'s`, `'T`, …) match case-insensitively.
    pub ci_contractions: bool,
    /// Longest run of digits kept together (GPT-2: unlimited, Llama-3: 3, Qwen-2: 1).
    pub digit_max: usize,
}
impl PreCfg {
    pub const GPT2: PreCfg = PreCfg { new_style: false, ci_contractions: false, digit_max: usize::MAX };
    pub const LLAMA3: PreCfg = PreCfg { new_style: true, ci_contractions: true, digit_max: 3 };
    pub const QWEN2: PreCfg = PreCfg { new_style: true, ci_contractions: true, digit_max: 1 };
}

#[derive(Debug)]
pub struct Tokenizer {
    pub kind: Kind,
    pub pre: PreCfg,
    tokens: Vec<String>,
    vocab: HashMap<String, u32>,
    /// (left, right) → (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// (content, id), longest first
    added: Vec<(String, u32)>,
    added_first: HashMap<u8, Vec<usize>>,
    /// 0 = ordinary, 1 = added, 2 = added and special; indexed by id
    added_flag: Vec<u8>,
    byte_ids: [Option<u32>; 256],
    b2c: [char; 256],
    rev: HashMap<char, u8>,
    ignore_merges: bool,
    add_prefix: bool,
    pub unk: Option<u32>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
}

// ------------------------------------------------------------------ helpers

/// GPT-2's reversible byte → printable character table, and its inverse.
fn byte_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut t = ['\0'; 256];
    let mut rev = HashMap::with_capacity(256);
    let mut extra = 0u32;
    for (b, slot) in t.iter_mut().enumerate() {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable { b as u32 } else { extra += 1; 255 + extra };
        *slot = char::from_u32(c).unwrap_or('\u{fffd}');
        rev.insert(*slot, b as u8);
    }
    (t, rev)
}

fn parse_byte_token(t: &str) -> Option<u8> {
    let b = t.as_bytes();
    if b.len() != 6 || &b[..3] != b"<0x" || b[5] != b'>' { return None; }
    let hex = |c: u8| (c as char).to_digit(16);
    Some((hex(b[3])? * 16 + hex(b[4])?) as u8)
}

fn make_vocab(tokens: &[String]) -> HashMap<String, u32> {
    let mut v = HashMap::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        if !t.is_empty() { v.entry(t.clone()).or_insert(i as u32); }
    }
    v
}

fn resolve_merges(vocab: &HashMap<String, u32>, pairs: impl Iterator<Item = (String, String)>) -> Vec<(u32, u32, u32)> {
    pairs.filter_map(|(a, b)| {
        let (l, r) = (*vocab.get(&a)?, *vocab.get(&b)?);
        let m = *vocab.get(&format!("{a}{b}"))?;
        Some((l, r, m))
    }).collect()
}

fn merge_pair(v: &Value) -> Option<(String, String)> {
    match v {
        Value::String(s) => s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())),
        Value::Array(a) if a.len() == 2 => Some((a[0].as_str()?.to_string(), a[1].as_str()?.to_string())),
        _ => None,
    }
}

struct Parts {
    kind: Kind, pre: PreCfg, tokens: Vec<String>, vocab: HashMap<String, u32>,
    merge_ids: Vec<(u32, u32, u32)>, added: Vec<(u32, bool)>,
    unk: Option<u32>, bos: Option<u32>, eos: Option<u32>, add_prefix: bool, ignore_merges: bool,
}

fn assemble(p: Parts) -> Tokenizer {
    let (b2c, rev) = byte_tables();
    let mut merges = HashMap::with_capacity(p.merge_ids.len());
    for (rank, (l, r, m)) in p.merge_ids.into_iter().enumerate() {
        merges.entry((l, r)).or_insert((rank as u32, m));
    }
    let mut byte_ids = [None; 256];
    match p.kind {
        Kind::ByteLevel => for (b, c) in b2c.iter().enumerate() { byte_ids[b] = p.vocab.get(&c.to_string()).copied(); },
        Kind::SentencePiece => for (id, t) in p.tokens.iter().enumerate() {
            if let Some(b) = parse_byte_token(t) { byte_ids[b as usize].get_or_insert(id as u32); }
        },
    }
    let mut added_flag = vec![0u8; p.tokens.len()];
    let mut added: Vec<(String, u32)> = vec![];
    for (id, special) in p.added {
        let Some(content) = p.tokens.get(id as usize).filter(|c| !c.is_empty()) else { continue };
        added_flag[id as usize] = 1 + special as u8;
        added.push((content.clone(), id));
    }
    added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
    let mut added_first: HashMap<u8, Vec<usize>> = HashMap::new();
    for (k, (s, _)) in added.iter().enumerate() { added_first.entry(s.as_bytes()[0]).or_default().push(k); }
    Tokenizer {
        kind: p.kind, pre: p.pre, tokens: p.tokens, vocab: p.vocab, merges, added, added_first, added_flag,
        byte_ids, b2c, rev, ignore_merges: p.ignore_merges, add_prefix: p.add_prefix,
        unk: p.unk, bos: p.bos, eos: p.eos,
    }
}

// ------------------------------------------------------------ pre-tokenizer

fn is_l(c: char) -> bool { c.is_alphabetic() }
fn is_n(c: char) -> bool { c.is_numeric() }
fn is_ws(c: char) -> bool { c.is_whitespace() }
fn is_nl(c: char) -> bool { c == '\r' || c == '\n' }
fn is_p(c: char) -> bool { !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric() }
fn at(cs: &[(usize, char)], k: usize) -> Option<char> { cs.get(k).map(|x| x.1) }
fn run(cs: &[(usize, char)], mut k: usize, f: fn(char) -> bool) -> usize {
    while k < cs.len() && f(cs[k].1) { k += 1; }
    k
}

/// Characters consumed by a contraction starting at the apostrophe at `i` (0 = none).
fn contraction(cs: &[(usize, char)], i: usize, ci: bool) -> usize {
    let g = |k: usize| at(cs, k).map(|c| if ci { c.to_ascii_lowercase() } else { c });
    match g(i + 1) {
        Some('s' | 't' | 'm' | 'd') => 2,
        Some('r') if g(i + 2) == Some('e') => 3,
        Some('v') if g(i + 2) == Some('e') => 3,
        Some('l') if g(i + 2) == Some('l') => 3,
        _ => 0,
    }
}

/// End (exclusive, in chars) of the piece that starts at char `i`; always `> i`.
fn match_at(cs: &[(usize, char)], i: usize, cfg: &PreCfg) -> usize {
    let n = cs.len();
    let c = cs[i].1;
    if c == '\'' {
        let l = contraction(cs, i, cfg.ci_contractions);
        if l > 0 { return i + l; }
    }
    if cfg.new_style {
        if is_l(c) { return run(cs, i, is_l); }
        if !is_nl(c) && !is_n(c) && at(cs, i + 1).is_some_and(is_l) { return run(cs, i + 1, is_l); }
        if is_n(c) {
            let mut k = i;
            while k < n && k - i < cfg.digit_max && is_n(cs[k].1) { k += 1; }
            return k;
        }
        let s = if c == ' ' { i + 1 } else { i };
        if at(cs, s).is_some_and(is_p) {
            let mut k = run(cs, s, is_p);
            while at(cs, k).is_some_and(is_nl) { k += 1; }
            return k;
        }
        if is_ws(c) {
            let e = run(cs, i, is_ws);
            if let Some(k) = (i..e).rev().find(|&k| is_nl(cs[k].1)) { return k + 1; }
        }
    } else {
        let s = if c == ' ' { i + 1 } else { i };
        if at(cs, s).is_some_and(is_l) { return run(cs, s, is_l); }
        if at(cs, s).is_some_and(is_n) { return run(cs, s, is_n); }
        if at(cs, s).is_some_and(is_p) { return run(cs, s, is_p); }
    }
    if is_ws(c) {
        let e = run(cs, i, is_ws);
        // `\s+(?!\S)` leaves the last blank for the next word; a lone blank is taken by `\s+`.
        return if e >= n || e - i == 1 { e } else { e - 1 };
    }
    i + 1
}

/// Split text into the pieces the model's regex would produce.
pub fn split_pieces<'a>(text: &'a str, cfg: &PreCfg) -> Vec<&'a str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let j = match_at(&cs, i, cfg).max(i + 1);
        let end = if j < cs.len() { cs[j].0 } else { text.len() };
        out.push(&text[cs[i].0..end]);
        i = j;
    }
    out
}

// ------------------------------------------------------------------- loaders

fn walk<'a>(v: &'a Value, f: &mut dyn FnMut(&'a Value)) {
    f(v);
    match v {
        Value::Object(o) => o.values().for_each(|x| walk(x, f)),
        Value::Array(a) => a.iter().for_each(|x| walk(x, f)),
        _ => {}
    }
}

fn has_type(v: Option<&Value>, name: &str) -> bool {
    let mut found = false;
    if let Some(v) = v { walk(v, &mut |x| if x.get("type").and_then(Value::as_str) == Some(name) { found = true; }); }
    found
}

fn pre_from_regexes(root: &Value) -> PreCfg {
    let mut pats: Vec<&str> = vec![];
    if let Some(p) = root.get("pre_tokenizer") {
        walk(p, &mut |x| if let Some(r) = x.get("pattern").and_then(|p| p.get("Regex")).and_then(Value::as_str) { pats.push(r); });
    }
    for p in pats {
        if p.contains(r"[^\r\n\p{L}\p{N}]?\p{L}+") {
            return PreCfg {
                new_style: true,
                ci_contractions: p.contains("(?i:"),
                digit_max: if p.contains(r"\p{N}{1,") { 3 } else { 1 },
            };
        }
    }
    PreCfg::GPT2
}

impl Tokenizer {
    pub fn from_file(path: &std::path::Path) -> Result<Tokenizer, TokError> {
        let len = std::fs::metadata(path).map_err(|e| TokError::Invalid(e.to_string()))?.len();
        if len > 1 << 29 { return invalid("tokenizer file is implausibly large"); }
        let s = std::fs::read_to_string(path).map_err(|e| TokError::Invalid(e.to_string()))?;
        Tokenizer::from_json(&s)
    }

    /// Hugging Face `tokenizer.json` with `model.type` BPE.
    pub fn from_json(s: &str) -> Result<Tokenizer, TokError> {
        let root: Value = serde_json::from_str(s).map_err(|e| TokError::Json(e.to_string()))?;
        let model = root.get("model").and_then(Value::as_object).ok_or_else(|| TokError::Invalid("no `model` object".into()))?;
        match model.get("type").and_then(Value::as_str) {
            Some("BPE") | None => {}
            Some(o) => return Err(TokError::Unsupported(format!("model type {o}"))),
        }
        let vocab_obj = model.get("vocab").and_then(Value::as_object).ok_or_else(|| TokError::Invalid("no vocab".into()))?;
        let added_arr: &[Value] = root.get("added_tokens").and_then(Value::as_array).map_or(&[], Vec::as_slice);
        let limit = MAX_ID_HARD.min(4 * (vocab_obj.len() + added_arr.len()) as u64 + 65_536);
        let mut tokens: Vec<String> = vec![];
        let mut place = |id: u64, s: &str| -> Result<(), TokError> {
            if id >= limit { return invalid(format!("token id {id} is out of range")); }
            if tokens.len() <= id as usize { tokens.resize(id as usize + 1, String::new()); }
            tokens[id as usize] = s.to_string();
            Ok(())
        };
        for (t, id) in vocab_obj {
            place(id.as_u64().ok_or_else(|| TokError::Invalid(format!("bad id for {t:?}")))?, t)?;
        }
        let mut added = vec![];
        for a in added_arr {
            let (Some(id), Some(c)) = (a.get("id").and_then(Value::as_u64), a.get("content").and_then(Value::as_str)) else { continue };
            place(id, c)?;
            added.push((id as u32, a.get("special").and_then(Value::as_bool).unwrap_or(false)));
        }
        let vocab = make_vocab(&tokens);
        let merge_pairs = model.get("merges").and_then(Value::as_array).map_or(&[][..], Vec::as_slice).iter().filter_map(merge_pair);
        let merge_ids = resolve_merges(&vocab, merge_pairs);

        let byte_level = has_type(root.get("pre_tokenizer"), "ByteLevel") || has_type(root.get("decoder"), "ByteLevel");
        let kind = if byte_level { Kind::ByteLevel } else { Kind::SentencePiece };
        let pre = if byte_level { pre_from_regexes(&root) } else { PreCfg::GPT2 };
        let norm = root.get("normalizer").map(Value::to_string).unwrap_or_default();
        let pre_s = root.get("pre_tokenizer").map(Value::to_string).unwrap_or_default();
        let add_prefix = !byte_level && (norm.contains("\"Prepend\"")
            || (pre_s.contains("Metaspace") && !pre_s.contains("\"prepend_scheme\":\"never\"") && !pre_s.contains("\"add_prefix_space\":false")));

        // bos / eos from a TemplateProcessing "single" template: specials before / after the sequence.
        let (mut bos, mut eos) = (None, None);
        if let Some(pp) = root.get("post_processor") {
            let mut singles: Vec<&Vec<Value>> = vec![];
            walk(pp, &mut |x| if x.get("type").and_then(Value::as_str) == Some("TemplateProcessing") {
                if let Some(a) = x.get("single").and_then(Value::as_array) { singles.push(a); }
            });
            if let Some(items) = singles.first() {
                let mut seen_seq = false;
                for it in items.iter() {
                    if it.get("Sequence").is_some() { seen_seq = true; continue; }
                    let id = it.get("SpecialToken").and_then(|s| s.get("id")).and_then(Value::as_str).and_then(|n| vocab.get(n)).copied();
                    if seen_seq { eos = eos.or(id) } else { bos = bos.or(id) }
                }
            }
        }
        let unk = model.get("unk_token").and_then(Value::as_str).and_then(|u| vocab.get(u)).copied();
        let ignore_merges = model.get("ignore_merges").and_then(Value::as_bool).unwrap_or(false);
        Ok(assemble(Parts { kind, pre, tokens, vocab, merge_ids, added, unk, bos, eos, add_prefix, ignore_merges }))
    }

    /// GGUF `tokenizer.ggml.*` metadata. `gpt2` = byte-level BPE; `llama` = SentencePiece
    /// (merge order derived from token scores when the file carries no merge list).
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, TokError> {
        let meta = |k: &str| g.meta.get(k);
        let model = meta("tokenizer.ggml.model").and_then(GValue::as_str).ok_or_else(|| TokError::Invalid("no tokenizer.ggml.model".into()))?;
        let kind = match model {
            "gpt2" => Kind::ByteLevel,
            "llama" => Kind::SentencePiece,
            o => return Err(TokError::Unsupported(format!("GGUF tokenizer model {o}"))),
        };
        let tokens: Vec<String> = match meta("tokenizer.ggml.tokens") {
            Some(GValue::Array(a)) => a.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect(),
            _ => return invalid("no tokenizer.ggml.tokens"),
        };
        if tokens.is_empty() { return invalid("empty vocabulary"); }
        let types: Vec<i64> = match meta("tokenizer.ggml.token_type") {
            Some(GValue::Array(a)) => a.iter().map(|x| match x { GValue::U64(v) => *v as i64, GValue::I64(v) => *v, _ => 1 }).collect(),
            _ => vec![],
        };
        let scores: Vec<f64> = match meta("tokenizer.ggml.scores") {
            Some(GValue::Array(a)) => a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect(),
            _ => vec![],
        };
        let vocab = make_vocab(&tokens);
        let merge_strs: Option<Vec<(String, String)>> = match meta("tokenizer.ggml.merges") {
            Some(GValue::Array(a)) if !a.is_empty() => Some(a.iter().filter_map(|x| x.as_str()?.split_once(' ').map(|(l, r)| (l.to_string(), r.to_string()))).collect()),
            _ => None,
        };
        let merge_ids = match (merge_strs, kind) {
            (Some(m), _) => resolve_merges(&vocab, m.into_iter()),
            (None, Kind::SentencePiece) => derive_merges(&tokens, &vocab, &types, &scores),
            (None, Kind::ByteLevel) => return invalid("byte-level BPE vocabulary without merges"),
        };
        let added = types.iter().enumerate().filter(|(i, t)| (**t == 3 || **t == 4) && *i < tokens.len()).map(|(i, t)| (i as u32, *t == 3)).collect();
        let id_of = |k: &str| meta(k).and_then(GValue::as_u64).filter(|v| (*v as usize) < tokens.len()).map(|v| v as u32);
        let pre = match meta("tokenizer.ggml.pre").and_then(GValue::as_str) {
            None | Some("default" | "gpt-2" | "gpt2") => PreCfg::GPT2,
            Some(p) if p.contains("qwen") => PreCfg::QWEN2,
            Some(_) => PreCfg::LLAMA3,
        };
        let add_prefix = match meta("tokenizer.ggml.add_space_prefix") { Some(GValue::Bool(b)) => *b, _ => true };
        let (unk, bos, eos) = (id_of("tokenizer.ggml.unknown_token_id"), id_of("tokenizer.ggml.bos_token_id"), id_of("tokenizer.ggml.eos_token_id"));
        Ok(assemble(Parts { kind, pre, tokens, vocab, merge_ids, added, unk, bos, eos, add_prefix: add_prefix && kind == Kind::SentencePiece, ignore_merges: false }))
    }
}

/// Merge order for SentencePiece vocabularies: a pair merges if the joined text is a token, and
/// higher-scored results merge first (ties: lower id first).
fn derive_merges(tokens: &[String], vocab: &HashMap<String, u32>, types: &[i64], scores: &[f64]) -> Vec<(u32, u32, u32)> {
    let normal = |id: u32| types.get(id as usize).is_none_or(|t| *t == 1);
    let mut c: Vec<(f64, u32, u32, u32)> = vec![];
    for (id, t) in tokens.iter().enumerate() {
        let id = id as u32;
        let n = t.chars().count();
        if !(2..=64).contains(&n) || vocab.get(t) != Some(&id) || !normal(id) { continue; }
        let score = scores.get(id as usize).copied().unwrap_or(0.0);
        for (bi, _) in t.char_indices().skip(1) {
            let (l, r) = t.split_at(bi);
            if let (Some(&li), Some(&ri)) = (vocab.get(l), vocab.get(r)) {
                if normal(li) && normal(ri) { c.push((score, li, ri, id)); }
            }
        }
    }
    c.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.3.cmp(&b.3)));
    c.into_iter().map(|(_, l, r, m)| (l, r, m)).collect()
}

// ------------------------------------------------------------------ encoding

enum Seg<'a> { Text(&'a str), Tok(u32) }

impl Tokenizer {
    pub fn vocab_size(&self) -> usize { self.tokens.len() }
    pub fn token_to_id(&self, t: &str) -> Option<u32> { self.vocab.get(t).copied() }
    pub fn id_to_token(&self, id: u32) -> Option<&str> { self.tokens.get(id as usize).map(String::as_str) }
    pub fn is_special(&self, id: u32) -> bool { self.added_flag.get(id as usize) == Some(&2) }

    fn segments<'a>(&self, text: &'a str) -> Vec<Seg<'a>> {
        if self.added.is_empty() { return vec![Seg::Text(text)]; }
        let b = text.as_bytes();
        let (mut out, mut i, mut last) = (vec![], 0, 0);
        'scan: while i < b.len() {
            if let Some(c) = self.added_first.get(&b[i]) {
                for &k in c {
                    let (s, id) = &self.added[k];
                    if b[i..].starts_with(s.as_bytes()) {
                        if last < i { out.push(Seg::Text(&text[last..i])); }
                        out.push(Seg::Tok(*id));
                        i += s.len();
                        last = i;
                        continue 'scan;
                    }
                }
            }
            i += 1;
        }
        if last < b.len() { out.push(Seg::Text(&text[last..])); }
        out
    }

    /// Merge a symbol sequence by rank using a heap over a doubly linked list.
    fn bpe(&self, ids: Vec<u32>) -> Vec<u32> {
        let n = ids.len();
        if n < 2 || self.merges.is_empty() { return ids; }
        const NONE: usize = usize::MAX;
        let mut id = ids;
        let mut next: Vec<usize> = (1..=n).map(|k| if k < n { k } else { NONE }).collect();
        let mut prev: Vec<usize> = (0..n).map(|k| if k == 0 { NONE } else { k - 1 }).collect();
        let mut alive = vec![true; n];
        let mut heap = BinaryHeap::new();
        for k in 0..n - 1 {
            if let Some(&(r, _)) = self.merges.get(&(id[k], id[k + 1])) { heap.push(Reverse((r, k, id[k], id[k + 1]))); }
        }
        while let Some(Reverse((_, l, li, ri))) = heap.pop() {
            if !alive[l] { continue; }
            let r = next[l];
            if r == NONE || id[l] != li || id[r] != ri { continue; }
            let Some(&(_, m)) = self.merges.get(&(li, ri)) else { continue };
            id[l] = m;
            alive[r] = false;
            let nr = next[r];
            next[l] = nr;
            if nr != NONE {
                prev[nr] = l;
                if let Some(&(rk, _)) = self.merges.get(&(m, id[nr])) { heap.push(Reverse((rk, l, m, id[nr]))); }
            }
            let p = prev[l];
            if p != NONE {
                if let Some(&(rk, _)) = self.merges.get(&(id[p], m)) { heap.push(Reverse((rk, p, id[p], m))); }
            }
        }
        let mut out = Vec::with_capacity(n);
        let mut k = 0;
        while k != NONE { out.push(id[k]); k = next[k]; }
        out
    }

    fn encode_bytelevel(&self, text: &str, out: &mut Vec<u32>) {
        for piece in split_pieces(text, &self.pre) {
            if self.ignore_merges {
                let mapped: String = piece.bytes().map(|b| self.b2c[b as usize]).collect();
                if let Some(&id) = self.vocab.get(&mapped) { out.push(id); continue; }
            }
            let ids: Vec<u32> = piece.bytes().filter_map(|b| self.byte_ids[b as usize].or(self.unk)).collect();
            out.extend(self.bpe(ids));
        }
    }

    fn encode_sp(&self, text: &str, prefix: bool, out: &mut Vec<u32>) {
        let mut ids = vec![];
        let mut buf = [0u8; 4];
        let mut symbols = text.chars().map(|c| if c == ' ' { SPACE } else { c });
        let first = if prefix { Some(SPACE) } else { None };
        for ch in first.into_iter().chain(&mut symbols) {
            let s = ch.encode_utf8(&mut buf);
            if let Some(&id) = self.vocab.get(&*s) { ids.push(id); continue; }
            let bytes: Option<Vec<u32>> = s.bytes().map(|b| self.byte_ids[b as usize]).collect();
            match bytes { Some(v) => ids.extend(v), None => ids.extend(self.unk) }
        }
        out.extend(self.bpe(ids));
    }

    /// Text → token ids. `add_bos` prepends the model's BOS token if it has one.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 2);
        if add_bos { out.extend(self.bos); }
        for (idx, seg) in self.segments(text).into_iter().enumerate() {
            match seg {
                Seg::Tok(id) => out.push(id),
                Seg::Text(t) => match self.kind {
                    Kind::ByteLevel => self.encode_bytelevel(t, &mut out),
                    Kind::SentencePiece => self.encode_sp(t, idx == 0 && self.add_prefix, &mut out),
                },
            }
        }
        out
    }

    /// Token ids → text (special tokens are written out as their text).
    pub fn decode(&self, ids: &[u32]) -> String { self.decode_with(ids, false) }

    pub fn decode_with(&self, ids: &[u32], skip_special: bool) -> String {
        let mut d = self.decoder();
        d.skip_special = skip_special;
        let mut s: String = ids.iter().map(|&i| d.push(i)).collect();
        s.push_str(&d.finish());
        s
    }

    /// Streaming decoder: feed ids one by one, get text as soon as it is complete UTF-8.
    pub fn decoder(&self) -> Decoder<'_> { Decoder { tok: self, pending: vec![], first: true, skip_special: false } }

    /// Append the raw bytes of one token; true if the token contributed any text.
    fn append_token_bytes(&self, id: u32, first: bool, skip_special: bool, out: &mut Vec<u8>) -> bool {
        let Some(t) = self.tokens.get(id as usize) else { return false };
        let flag = self.added_flag.get(id as usize).copied().unwrap_or(0);
        if flag != 0 {
            if skip_special && flag == 2 { return false; }
            out.extend_from_slice(t.as_bytes());
            return !t.is_empty();
        }
        match self.kind {
            Kind::ByteLevel => {
                for ch in t.chars() {
                    match self.rev.get(&ch) {
                        Some(&b) => out.push(b),
                        None => out.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes()),
                    }
                }
                !t.is_empty()
            }
            Kind::SentencePiece => {
                if let Some(b) = parse_byte_token(t) { out.push(b); return true; }
                let start = out.len();
                for ch in t.chars() {
                    if ch == SPACE { out.push(b' ') } else { out.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes()) }
                }
                let produced = out.len() > start;
                if first && self.add_prefix && produced && out[start] == b' ' { out.remove(start); }
                produced
            }
        }
    }
}

/// Incremental decoder that never emits half a character.
pub struct Decoder<'a> {
    tok: &'a Tokenizer,
    pending: Vec<u8>,
    first: bool,
    pub skip_special: bool,
}
impl Decoder<'_> {
    pub fn push(&mut self, id: u32) -> String {
        if self.tok.append_token_bytes(id, self.first, self.skip_special, &mut self.pending) { self.first = false; }
        self.drain(false)
    }
    /// Flush anything left over (an incomplete trailing sequence becomes U+FFFD).
    pub fn finish(&mut self) -> String { self.drain(true) }

    fn drain(&mut self, flush: bool) -> String {
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => { out.push_str(s); self.pending.clear(); return out; }
                Err(e) => {
                    let v = e.valid_up_to();
                    out.push_str(&String::from_utf8_lossy(&self.pending[..v]));
                    match e.error_len() {
                        Some(n) => { out.push('\u{fffd}'); self.pending.drain(..v + n); }
                        None => {
                            self.pending.drain(..v);
                            if flush { out.push('\u{fffd}'); self.pending.clear(); }
                            return out;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const LLAMA3_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const QWEN2_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn gpt2_pre() -> Value { json!({"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}) }
    fn regex_pre(re: &str) -> Value {
        json!({"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": re}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}]})
    }

    /// 256 byte tokens, then one token per merge result, then the added tokens.
    fn bl(merges: &[(&str, &str)], added: &[(&str, bool)], pre: Value, ignore: bool) -> Tokenizer {
        let (t, _) = byte_tables();
        let mut vocab = serde_json::Map::new();
        let mut n = 0u64;
        for c in t.iter() { vocab.insert(c.to_string(), n.into()); n += 1; }
        let mut ms = vec![];
        for (a, b) in merges {
            let m = format!("{a}{b}");
            if !vocab.contains_key(&m) { vocab.insert(m, n.into()); n += 1; }
            ms.push(Value::String(format!("{a} {b}")));
        }
        let mut ad = vec![];
        for (c, sp) in added { ad.push(json!({"id": n, "content": c, "special": sp})); n += 1; }
        let j = json!({"model": {"type": "BPE", "vocab": vocab, "merges": ms, "ignore_merges": ignore},
            "added_tokens": ad, "pre_tokenizer": pre});
        Tokenizer::from_json(&j.to_string()).unwrap()
    }
    fn hello() -> Tokenizer {
        bl(&[("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), ("Ġ", "hello")], &[("<|bos|>", true), ("<|eos|>", true)], gpt2_pre(), false)
    }

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }
    fn nasty(seed: u64, max: usize) -> String {
        let mut s = seed | 1;
        let alpha: Vec<char> = "aAzZ09 \n\r\t'\".,!?-_é😀漢字日本語العربية\u{0301}\u{200d}\u{0}\u{ffff}<|>ß Σ٣".chars().collect();
        let len = (rng(&mut s) % max as u64) as usize;
        (0..len).filter_map(|_| {
            let r = rng(&mut s);
            if r % 5 == 0 { char::from_u32((r >> 8) as u32 % 0x11_0000) } else { Some(alpha[(r >> 8) as usize % alpha.len()]) }
        }).collect()
    }

    // ----------------------------------------------------------- splitting

    #[test]
    fn gpt2_pieces() {
        let p = |t| split_pieces(t, &PreCfg::GPT2);
        assert_eq!(p("Hello, world!  x"), ["Hello", ",", " world", "!", " ", " x"]);
        assert_eq!(p("don't DON'T"), ["don", "'t", " DON", "'", "T"]);
        assert_eq!(p("a  \n b"), ["a", "  \n", " b"]);
        assert_eq!(p("x 12 345.6"), ["x", " 12", " 345", ".", "6"]);
        assert_eq!(p("trailing   "), ["trailing", "   "]);
        assert_eq!(p(""), Vec::<&str>::new());
    }

    #[test]
    fn llama3_and_qwen_pieces() {
        let p = |t, c| split_pieces(t, &c);
        assert_eq!(p("Hello, world!\n\nfoo 12345", PreCfg::LLAMA3), ["Hello", ",", " world", "!\n\n", "foo", " ", "123", "45"]);
        assert_eq!(p("DON'T go", PreCfg::LLAMA3), ["DON", "'T", " go"]);
        assert_eq!(p("12345", PreCfg::QWEN2), ["1", "2", "3", "4", "5"]);
        assert_eq!(p("a\r\n  \nb", PreCfg::LLAMA3), ["a", "\r\n  \n", "b"]);
        assert_eq!(p("$x\t.y", PreCfg::LLAMA3), ["$x", "\t", ".y"]);
        assert_eq!(p("  ", PreCfg::LLAMA3), ["  "]);
    }

    #[test]
    fn pieces_always_tile_the_text() {
        for cfg in [PreCfg::GPT2, PreCfg::LLAMA3, PreCfg::QWEN2] {
            for seed in 0..300 {
                let t = nasty(seed, 200);
                let ps = split_pieces(&t, &cfg);
                assert!(ps.iter().all(|p| !p.is_empty()));
                assert_eq!(ps.concat(), t);
            }
        }
    }

    #[test]
    fn regex_in_the_file_selects_the_pattern() {
        assert_eq!(bl(&[], &[], regex_pre(LLAMA3_RE), false).pre, PreCfg::LLAMA3);
        assert_eq!(bl(&[], &[], regex_pre(QWEN2_RE), false).pre, PreCfg::QWEN2);
        assert_eq!(bl(&[], &[], gpt2_pre(), false).pre, PreCfg::GPT2);
    }

    // ------------------------------------------------------------ byte-level

    #[test]
    fn merges_build_words_and_respect_rank() {
        let t = hello();
        let id = |s: &str| t.token_to_id(s).unwrap();
        assert_eq!(t.encode("hello hello", false), [id("hello"), id("Ġhello")]);
        assert_eq!(t.encode("hell", false), [id("hell")]);
        // rank decides: "b c" (0) beats "a b" (1)
        let t = bl(&[("b", "c"), ("a", "b")], &[], gpt2_pre(), false);
        assert_eq!(t.encode("abc", false), [t.token_to_id("a").unwrap(), t.token_to_id("bc").unwrap()]);
    }

    #[test]
    fn ignore_merges_takes_whole_pieces_found_in_the_vocabulary() {
        // "hello" is a token, but the merges alone only reach "he"
        let merges = [("h", "e"), ("he", "llo")];
        let on = bl(&merges, &[], gpt2_pre(), true);
        assert_eq!(on.encode("hello", false), [on.token_to_id("hello").unwrap()]);
        let off = bl(&merges, &[], gpt2_pre(), false);
        assert_eq!(off.encode("hello", false).len(), 4);
    }

    #[test]
    fn specials_are_matched_first_longest_and_decode_back() {
        let t = bl(&[], &[("<|a|>", true), ("<|a|>b", false), ("<", false)], gpt2_pre(), false);
        let ids = t.encode("x<|a|>by<|a|>", false);
        let (a, ab) = (t.token_to_id("<|a|>").unwrap(), t.token_to_id("<|a|>b").unwrap());
        assert_eq!(ids.iter().filter(|&&i| i == ab).count(), 1);
        assert_eq!(ids.iter().filter(|&&i| i == a).count(), 1);
        assert_eq!(t.decode(&ids), "x<|a|>by<|a|>");
        assert_eq!(t.decode_with(&ids, true), "x<|a|>by"); // `<|a|>b` is not special, `<|a|>` is
        assert!(t.is_special(a) && !t.is_special(ab));
    }

    #[test]
    fn bos_is_added_on_request() {
        let mut t = hello();
        t.bos = t.token_to_id("<|bos|>");
        assert_eq!(t.encode("hi", true)[0], t.bos.unwrap());
        assert_ne!(t.encode("hi", false).first().copied(), t.bos);
    }

    #[test]
    fn every_string_round_trips() {
        let t = hello();
        for s in ["", "plain ascii", "emoji 😀🎉 and ZWJ 👩‍👩‍👧", "漢字 日本語 한국어", "  leading and\ttabs\r\nand\n\n\nnewlines   ",
            "العربية עברית", "e\u{0301} combining", "nul\0char", "<|bos|>mixed<|eos|>text<|bos|>", "'s 't 're don't"] {
            assert_eq!(t.decode(&t.encode(s, false)), s, "{s:?}");
        }
        for seed in 0..400 {
            let s = nasty(seed, 300);
            assert_eq!(t.decode(&t.encode(&s, false)), s, "seed {seed}");
        }
    }

    #[test]
    fn streaming_decoder_never_splits_a_character() {
        let t = hello();
        let s = "é😀 漢 hello";
        let ids = t.encode(s, false);
        let mut d = t.decoder();
        let mut got = String::new();
        let mut empties = 0;
        for &i in &ids {
            let piece = d.push(i);
            if piece.is_empty() { empties += 1; }
            got.push_str(&piece);
        }
        got.push_str(&d.finish());
        assert_eq!(got, s);
        assert!(empties >= 4, "multi-byte characters must be held back ({empties})");
        // a dangling lead byte is flushed as U+FFFD
        let mut d = t.decoder();
        assert_eq!(d.push(t.byte_ids[0xC3].unwrap()), "");
        assert_eq!(d.finish(), "\u{fffd}");
        // an invalid continuation does not swallow the next character
        let mut d = t.decoder();
        let mut o = d.push(t.byte_ids[0xC3].unwrap());
        o += &d.push(t.byte_ids[b'a' as usize].unwrap());
        assert_eq!(o, "\u{fffd}a");
    }

    #[test]
    fn long_inputs_are_fast_and_lossless() {
        let t = bl(&[("a", "a"), ("aa", "aa")], &[], gpt2_pre(), false);
        let s = "a".repeat(200_000);
        let ids = t.encode(&s, false);
        assert!(ids.len() <= 50_000 + 3);
        assert_eq!(t.decode(&ids), s);
        let s = "😀!".repeat(60_000);
        assert_eq!(t.decode(&t.encode(&s, false)), s);
    }

    // --------------------------------------------------------- SentencePiece

    fn sp_json() -> String {
        let mut vocab = serde_json::Map::new();
        let mut n = 0u64;
        for t in ["<unk>", "<s>", "</s>"] { vocab.insert(t.into(), n.into()); n += 1; }
        for b in 0..256 { vocab.insert(format!("<0x{b:02X}>"), n.into()); n += 1; }
        for t in ["▁", "h", "e", "l", "o", "w", "r", "d", "▁h", "▁he", "ll", "▁hell", "▁hello", "▁w", "or", "▁wor", "ld", "▁world"] {
            vocab.insert(t.into(), n.into()); n += 1;
        }
        let merges = ["▁ h", "▁h e", "l l", "▁he ll", "▁hell o", "▁ w", "o r", "▁w or", "l d", "▁wor ld"];
        json!({"model": {"type": "BPE", "vocab": vocab, "merges": merges, "byte_fallback": true, "unk_token": "<unk>"},
            "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
            "decoder": {"type": "Sequence", "decoders": [{"type": "Replace"}, {"type": "ByteFallback"}, {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
            "added_tokens": [{"id": 0, "content": "<unk>", "special": true}, {"id": 1, "content": "<s>", "special": true}, {"id": 2, "content": "</s>", "special": true}],
            "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}]}
        }).to_string()
    }

    #[test]
    fn sentencepiece_words_prefix_and_decode() {
        let t = Tokenizer::from_json(&sp_json()).unwrap();
        assert_eq!(t.kind, Kind::SentencePiece);
        let id = |s: &str| t.token_to_id(s).unwrap();
        assert_eq!(t.encode("hello world", false), [id("▁hello"), id("▁world")]);
        assert_eq!(t.bos, Some(1));
        assert_eq!(t.encode("hello", true)[0], 1);
        for s in ["hello world", " hi", "hello  world ", "naïve 😀\nline", "", "<s>hello</s>"] {
            assert_eq!(t.decode(&t.encode(s, false)), s, "{s:?}");
        }
    }

    #[test]
    fn sentencepiece_byte_fallback_and_unknown() {
        let t = Tokenizer::from_json(&sp_json()).unwrap();
        let ids = t.encode("é", false);
        assert!(ids.contains(&t.token_to_id("<0xC3>").unwrap()) && ids.contains(&t.token_to_id("<0xA9>").unwrap()));
        // no byte tokens at all: unknown characters become <unk>
        let j = json!({"model": {"type": "BPE", "vocab": {"<unk>": 0, "▁": 1, "a": 2}, "merges": [], "unk_token": "<unk>"},
            "normalizer": {"type": "Prepend", "prepend": "▁"}});
        let t = Tokenizer::from_json(&j.to_string()).unwrap();
        assert_eq!(t.encode("aZ", false), [1, 2, 0]);
    }

    // ------------------------------------------------------------------ GGUF

    fn gstr(s: &str) -> GValue { GValue::Str(s.into()) }

    #[test]
    fn gguf_byte_level() {
        let (t, _) = byte_tables();
        let mut toks: Vec<GValue> = t.iter().map(|c| gstr(&c.to_string())).collect();
        for x in ["he", "ll", "hell"] { toks.push(gstr(x)); }
        toks.push(gstr("<|endoftext|>"));
        let mut types = vec![GValue::U64(1); toks.len()];
        *types.last_mut().unwrap() = GValue::U64(3);
        let mut g = Gguf::default();
        g.meta.insert("tokenizer.ggml.model".into(), gstr("gpt2"));
        g.meta.insert("tokenizer.ggml.pre".into(), gstr("llama-bpe"));
        let eos = toks.len() as u64 - 1;
        g.meta.insert("tokenizer.ggml.tokens".into(), GValue::Array(toks));
        g.meta.insert("tokenizer.ggml.token_type".into(), GValue::Array(types));
        g.meta.insert("tokenizer.ggml.merges".into(), GValue::Array(vec![gstr("h e"), gstr("l l"), gstr("he ll")]));
        g.meta.insert("tokenizer.ggml.eos_token_id".into(), GValue::U64(eos));
        let tk = Tokenizer::from_gguf(&g).unwrap();
        assert_eq!(tk.pre, PreCfg::LLAMA3);
        assert_eq!(tk.eos, Some(eos as u32));
        let ids = tk.encode("hell<|endoftext|>x", false);
        assert_eq!(ids, [tk.token_to_id("hell").unwrap(), eos as u32, tk.token_to_id("x").unwrap()]);
        assert_eq!(tk.decode(&ids), "hell<|endoftext|>x");
        assert_eq!(tk.decode_with(&ids, true), "hellx");
    }

    #[test]
    fn gguf_sentencepiece_derives_merges_from_scores() {
        let toks = ["<unk>", "<s>", "</s>", "<0x0A>", "▁", "a", "b", "▁a", "ab", "▁ab"];
        let types = [2, 3, 3, 6, 1, 1, 1, 1, 1, 1];
        let scores = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -2.0, -1.5, -3.0];
        let mut g = Gguf::default();
        g.meta.insert("tokenizer.ggml.model".into(), gstr("llama"));
        g.meta.insert("tokenizer.ggml.tokens".into(), GValue::Array(toks.iter().map(|t| gstr(t)).collect()));
        g.meta.insert("tokenizer.ggml.token_type".into(), GValue::Array(types.iter().map(|t| GValue::U64(*t)).collect()));
        g.meta.insert("tokenizer.ggml.scores".into(), GValue::Array(scores.iter().map(|s| GValue::F64(*s)).collect()));
        g.meta.insert("tokenizer.ggml.unknown_token_id".into(), GValue::U64(0));
        g.meta.insert("tokenizer.ggml.bos_token_id".into(), GValue::U64(1));
        let t = Tokenizer::from_gguf(&g).unwrap();
        assert_eq!(t.encode("ab", false), [9]);
        assert_eq!(t.encode("b", false), [4, 6]);
        assert_eq!(t.encode("\n", false), [4, 3]);
        assert_eq!(t.encode("ab", true), [1, 9]);
        for s in ["ab", "b a", "\n", "a\nb"] { assert_eq!(t.decode(&t.encode(s, false)), s, "{s:?}"); }
        assert!(t.is_special(1) && !t.is_special(3));
    }

    #[test]
    fn gguf_problems_are_errors() {
        let mut g = Gguf::default();
        assert!(Tokenizer::from_gguf(&g).is_err());
        g.meta.insert("tokenizer.ggml.model".into(), gstr("bert"));
        assert!(matches!(Tokenizer::from_gguf(&g), Err(TokError::Unsupported(_))));
        g.meta.insert("tokenizer.ggml.model".into(), gstr("gpt2"));
        g.meta.insert("tokenizer.ggml.tokens".into(), GValue::Array(vec![gstr("a")]));
        assert!(Tokenizer::from_gguf(&g).is_err()); // byte-level without merges
        g.meta.insert("tokenizer.ggml.tokens".into(), GValue::Array(vec![]));
        assert!(Tokenizer::from_gguf(&g).is_err());
    }

    // --------------------------------------------------------- hostile files

    #[test]
    fn broken_json_is_an_error_never_a_panic() {
        for s in ["", "{}", "[]", "null", "{\"model\":3}", "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{}}}",
            "{\"model\":{\"type\":\"Unigram\",\"vocab\":[]}}", "{\"model\":{\"vocab\":{\"a\":99999999999}}}",
            "{\"model\":{\"vocab\":{\"a\":-1}}}", "{\"model\":{\"vocab\":{\"a\":\"x\"}}}"] {
            assert!(Tokenizer::from_json(s).is_err(), "{s}");
        }
        let deep = "[".repeat(5000);
        assert!(Tokenizer::from_json(&deep).is_err());
        let valid = sp_json();
        for k in (0..valid.len()).filter(|k| valid.is_char_boundary(*k)) {
            assert!(Tokenizer::from_json(&valid[..k]).is_err(), "prefix {k}");
        }
        let mut s = 7u64;
        let bytes = valid.as_bytes();
        for _ in 0..300 {
            let mut b = bytes.to_vec();
            for _ in 0..1 + rng(&mut s) % 6 { let i = rng(&mut s) as usize % b.len(); b[i] = rng(&mut s) as u8; }
            if let Ok(t) = Tokenizer::from_json(&String::from_utf8_lossy(&b)) {
                let _ = t.decode(&t.encode("héllo wörld 😀", true)); // a mutated but valid file must still behave
            }
        }
    }

    #[test]
    fn hostile_merges_and_ids_do_not_break_encoding() {
        // merges that refer to missing tokens, empty strings and cycles
        let j = json!({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "": 3},
            "merges": ["a b", "a ", " b", "x y", ["a", "b"], "ab ab", 5, "", "a"]},
            "pre_tokenizer": {"type": "ByteLevel"}, "added_tokens": [{"id": 3, "content": "", "special": true}]});
        let t = Tokenizer::from_json(&j.to_string()).unwrap();
        let _ = t.decode(&t.encode("ababab abab", true));
        let _ = t.decode(&[0, 1, 2, 3, 99, u32::MAX]);
        assert_eq!(t.id_to_token(u32::MAX), None);
    }

    #[test]
    fn fuzz_encode_decode_on_every_kind() {
        let toks = [hello(), Tokenizer::from_json(&sp_json()).unwrap(),
            bl(&[("a", "a"), ("aa", "a")], &[("<s>", true)], regex_pre(QWEN2_RE), true)];
        for (k, t) in toks.iter().enumerate() {
            for seed in 0..300 {
                let s = nasty(seed * 3 + k as u64, 250);
                let ids = t.encode(&s, true);
                assert!(ids.iter().all(|&i| (i as usize) < t.vocab_size()), "tokenizer {k} seed {seed}");
                let _ = t.decode(&ids);
                let mut d = t.decoder();
                let streamed: String = ids.iter().map(|&i| d.push(i)).collect::<String>() + &d.finish();
                assert_eq!(streamed, t.decode(&ids));
            }
        }
    }
}
