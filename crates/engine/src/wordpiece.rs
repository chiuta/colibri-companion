//! WordPiece tokenizer for BERT-family encoders (BERT, MiniLM, bge-small/base v1.5, e5 (English),
//! ms-marco cross-encoders and relatives).
//!
//! Written from the published algorithm: clean the text, optionally pad CJK characters with spaces,
//! lower-case and strip accents, split on whitespace and punctuation, then match each word greedily
//! against the vocabulary, longest piece first, with `##` marking pieces that continue a word; a
//! word with no full match, or longer than 100 characters, becomes `[UNK]`.
//!
//! Loads a Hugging Face `tokenizer.json` (model type `WordPiece`) or a GGUF file whose
//! `tokenizer.ggml.model` is `bert`. The GGUF convention (word-start pieces carry a leading `▁`,
//! continuation pieces carry none) is implemented from memory of how such files are written and
//! has not been checked against a real file; the JSON route is the one to prefer.
//!
//! Accent stripping decomposes the Latin letters of Latin-1 Supplement, Latin Extended-A and the
//! Romanian comma-below letters (`ș`, `ț`) and drops combining marks U+0300–U+036F; other scripts
//! keep their accents (the reference does a full Unicode decomposition).

use crate::gguf::{Gguf, Value as GValue};
use crate::tokenizer::uni as unicode;
use serde_json::Value;
use std::collections::HashMap;

const MAX_ID: u64 = 1 << 22;
/// Most characters accepted in one text (guards the quadratic-looking greedy match on hostile input).
const MAX_TEXT_CHARS: usize = 8 << 20;

#[derive(Debug, PartialEq, Eq)]
pub enum WpError {
    Json(String),
    Unsupported(String),
    Invalid(String),
}
impl std::fmt::Display for WpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WpError::Json(s) => write!(f, "tokenizer file is not valid JSON: {s}"),
            WpError::Unsupported(s) => write!(f, "tokenizer type not supported: {s}"),
            WpError::Invalid(s) => write!(f, "tokenizer file is inconsistent: {s}"),
        }
    }
}
impl std::error::Error for WpError {}
fn invalid<T>(s: impl Into<String>) -> Result<T, WpError> {
    Err(WpError::Invalid(s.into()))
}

#[derive(Debug, Clone)]
pub struct WordPiece {
    tokens: Vec<String>,
    vocab: HashMap<String, u32>,
    pub unk: u32,
    pub cls: Option<u32>,
    pub sep: Option<u32>,
    pub pad: Option<u32>,
    pub mask: Option<u32>,
    pub lowercase: bool,
    pub strip_accents: bool,
    pub chinese_spaces: bool,
    prefix: String,
    max_chars: usize,
    /// Ids that never appear in decoded text.
    special: Vec<u32>,
}

/// Replace an accented Latin letter by its base letter (`None` if it has no entry).
fn base_letter(c: char) -> Option<char> {
    const T: &[(&str, char)] = &[
        ("ÀÁÂÃÄÅĀĂĄǍ", 'A'),
        ("àáâãäåāăąǎ", 'a'),
        ("ÇĆĈĊČ", 'C'),
        ("çćĉċč", 'c'),
        ("ĎĐ", 'D'),
        ("ďđ", 'd'),
        ("ÈÉÊËĒĔĖĘĚ", 'E'),
        ("èéêëēĕėęě", 'e'),
        ("ĜĞĠĢ", 'G'),
        ("ĝğġģ", 'g'),
        ("ĤĦ", 'H'),
        ("ĥħ", 'h'),
        ("ÌÍÎÏĨĪĬĮİǏ", 'I'),
        ("ìíîïĩīĭįǐ", 'i'),
        ("Ĵ", 'J'),
        ("ĵ", 'j'),
        ("Ķ", 'K'),
        ("ķ", 'k'),
        ("ĹĻĽĿŁ", 'L'),
        ("ĺļľŀł", 'l'),
        ("ÑŃŅŇ", 'N'),
        ("ñńņň", 'n'),
        ("ÒÓÔÕÖØŌŎŐǑƠ", 'O'),
        ("òóôõöøōŏőǒơ", 'o'),
        ("ŔŖŘ", 'R'),
        ("ŕŗř", 'r'),
        ("ŚŜŞŠȘ", 'S'),
        ("śŝşšș", 's'),
        ("ŢŤŦȚ", 'T'),
        ("ţťŧț", 't'),
        ("ÙÚÛÜŨŪŬŮŰŲǓƯ", 'U'),
        ("ùúûüũūŭůűųǔư", 'u'),
        ("Ŵ", 'W'),
        ("ŵ", 'w'),
        ("ÝŶŸ", 'Y'),
        ("ýÿŷ", 'y'),
        ("ŹŻŽ", 'Z'),
        ("źżž", 'z'),
    ];
    // Latin Extended Additional, the Vietnamese block: pairs (upper, lower) in runs of one base letter
    let u = c as u32;
    if (0x1EA0..=0x1EF9).contains(&u) {
        let base = match u {
            0x1EA0..=0x1EB7 => 'a',
            0x1EB8..=0x1EC7 => 'e',
            0x1EC8..=0x1ECB => 'i',
            0x1ECC..=0x1EE3 => 'o',
            0x1EE4..=0x1EF1 => 'u',
            _ => 'y',
        };
        return Some(if u.is_multiple_of(2) {
            base.to_ascii_uppercase()
        } else {
            base
        });
    }
    T.iter().find(|(s, _)| s.contains(c)).map(|(_, b)| *b)
}

fn is_cjk(c: char) -> bool {
    let u = c as u32;
    (0x4E00..=0x9FFF).contains(&u)
        || (0x3400..=0x4DBF).contains(&u)
        || (0x20000..=0x2A6DF).contains(&u)
        || (0x2A700..=0x2B73F).contains(&u)
        || (0x2B740..=0x2B81F).contains(&u)
        || (0x2B820..=0x2CEAF).contains(&u)
        || (0xF900..=0xFAFF).contains(&u)
        || (0x2F800..=0x2FA1F).contains(&u)
}

/// BERT's punctuation test: every ASCII non-alphanumeric printable, plus Unicode `\p{P}`.
fn is_punct(c: char) -> bool {
    let u = c as u32;
    (33..=47).contains(&u)
        || (58..=64).contains(&u)
        || (91..=96).contains(&u)
        || (123..=126).contains(&u)
        || unicode::is_punct(c)
}

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r') || (c as u32 > 127 && c.is_whitespace())
}

fn is_control(c: char) -> bool {
    if matches!(c, '\t' | '\n' | '\r') {
        return false;
    }
    c.is_control()
}

impl WordPiece {
    fn finish(
        tokens: Vec<String>,
        unk: &str,
        prefix: String,
        max_chars: usize,
        lowercase: bool,
        strip: bool,
        chinese: bool,
    ) -> Result<WordPiece, WpError> {
        if tokens.is_empty() {
            return invalid("empty vocabulary");
        }
        let mut vocab = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            if !t.is_empty() {
                vocab.entry(t.clone()).or_insert(i as u32);
            }
        }
        let find = |n: &str| vocab.get(n).copied();
        let Some(unk) = find(unk) else {
            return invalid(format!("the unknown token {unk} is not in the vocabulary"));
        };
        let (cls, sep, pad, mask) = (find("[CLS]"), find("[SEP]"), find("[PAD]"), find("[MASK]"));
        let mut special: Vec<u32> = [Some(unk), cls, sep, pad, mask]
            .into_iter()
            .flatten()
            .collect();
        special.sort_unstable();
        special.dedup();
        Ok(WordPiece {
            tokens,
            vocab,
            unk,
            cls,
            sep,
            pad,
            mask,
            lowercase,
            strip_accents: strip,
            chinese_spaces: chinese,
            prefix,
            max_chars: max_chars.max(1),
            special,
        })
    }

    /// From a Hugging Face `tokenizer.json` whose model is `WordPiece`.
    pub fn from_json(s: &str) -> Result<WordPiece, WpError> {
        let root: Value = serde_json::from_str(s).map_err(|e| WpError::Json(e.to_string()))?;
        let model = root
            .get("model")
            .and_then(Value::as_object)
            .ok_or_else(|| WpError::Invalid("no `model` object".into()))?;
        match model.get("type").and_then(Value::as_str) {
            Some("WordPiece") => {}
            o => {
                return Err(WpError::Unsupported(format!(
                    "model type {}",
                    o.unwrap_or("(none)")
                )));
            }
        }
        let vocab_obj = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| WpError::Invalid("no vocab".into()))?;
        let limit = MAX_ID.min(4 * vocab_obj.len() as u64 + 65_536);
        let mut tokens: Vec<String> = vec![];
        for (t, id) in vocab_obj {
            let id = id
                .as_u64()
                .ok_or_else(|| WpError::Invalid(format!("bad id for {t:?}")))?;
            if id >= limit {
                return invalid(format!("token id {id} is out of range"));
            }
            if tokens.len() <= id as usize {
                tokens.resize(id as usize + 1, String::new());
            }
            tokens[id as usize] = t.clone();
        }
        // added tokens (specials) may extend or override the table
        if let Some(a) = root.get("added_tokens").and_then(Value::as_array) {
            for x in a {
                let (Some(id), Some(c)) = (
                    x.get("id").and_then(Value::as_u64),
                    x.get("content").and_then(Value::as_str),
                ) else {
                    continue;
                };
                if id >= limit {
                    return invalid(format!("added token id {id} is out of range"));
                }
                if tokens.len() <= id as usize {
                    tokens.resize(id as usize + 1, String::new());
                }
                if tokens[id as usize].is_empty() {
                    tokens[id as usize] = c.to_string();
                }
            }
        }
        let unk = model
            .get("unk_token")
            .and_then(Value::as_str)
            .unwrap_or("[UNK]")
            .to_string();
        let prefix = model
            .get("continuing_subword_prefix")
            .and_then(Value::as_str)
            .unwrap_or("##")
            .to_string();
        let max_chars = model
            .get("max_input_chars_per_word")
            .and_then(Value::as_u64)
            .unwrap_or(100)
            .min(1 << 16) as usize;
        // the normaliser: BertNormalizer, or defaults when it is absent
        let (mut lower, mut strip, mut chinese) = (true, None::<bool>, true);
        if let Some(n) = root.get("normalizer").filter(|n| !n.is_null()) {
            let mut one = |n: &Value| {
                if n.get("type").and_then(Value::as_str) == Some("BertNormalizer") {
                    lower = n.get("lowercase").and_then(Value::as_bool).unwrap_or(true);
                    strip = n.get("strip_accents").and_then(Value::as_bool);
                    chinese = n
                        .get("handle_chinese_chars")
                        .and_then(Value::as_bool)
                        .unwrap_or(true);
                }
            };
            if let Some(seq) = n.get("normalizers").and_then(Value::as_array) {
                seq.iter().for_each(&mut one);
            } else {
                one(n);
            }
        }
        // strip_accents unset follows lowercase, as the reference does
        let strip = strip.unwrap_or(lower);
        WordPiece::finish(tokens, &unk, prefix, max_chars, lower, strip, chinese)
    }

    pub fn from_file(path: &std::path::Path) -> Result<WordPiece, WpError> {
        let len = std::fs::metadata(path)
            .map_err(|e| WpError::Invalid(e.to_string()))?
            .len();
        if len > 256 << 20 {
            return invalid("tokenizer file is larger than 256 MiB");
        }
        let s = std::fs::read_to_string(path).map_err(|e| WpError::Invalid(e.to_string()))?;
        WordPiece::from_json(&s)
    }

    /// From a GGUF file whose `tokenizer.ggml.model` is `bert`.
    pub fn from_gguf(g: &Gguf) -> Result<WordPiece, WpError> {
        let m = g
            .meta
            .get("tokenizer.ggml.model")
            .and_then(GValue::as_str)
            .unwrap_or("");
        if m != "bert" {
            return Err(WpError::Unsupported(format!("GGUF tokenizer model {m}")));
        }
        let raw: Vec<String> = match g.meta.get("tokenizer.ggml.tokens") {
            Some(GValue::Array(a)) => a
                .iter()
                .map(|x| x.as_str().unwrap_or("").to_string())
                .collect(),
            _ => return invalid("no tokenizer.ggml.tokens"),
        };
        if raw.is_empty() || raw.len() as u64 > MAX_ID {
            return invalid("the vocabulary is empty or too large");
        }
        let id = |k: &str| {
            g.meta
                .get(k)
                .and_then(GValue::as_u64)
                .map(|v| v as usize)
                .filter(|&v| v < raw.len())
        };
        let keep: Vec<usize> = [
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.seperator_token_id",
            "tokenizer.ggml.separator_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.cls_token_id",
            "tokenizer.ggml.mask_token_id",
        ]
        .iter()
        .filter_map(|k| id(k))
        .collect();
        // word-start pieces carry "▁"; continuation pieces do not (and get "##" back); specials are kept
        let tokens: Vec<String> = raw
            .iter()
            .enumerate()
            .map(|(i, t)| {
                if keep.contains(&i) || (t.starts_with('[') && t.ends_with(']') && t.len() > 2) {
                    t.clone()
                } else if let Some(r) = t.strip_prefix('▁') {
                    r.to_string()
                } else {
                    format!("##{t}")
                }
            })
            .collect();
        let unk = id("tokenizer.ggml.unknown_token_id")
            .map(|i| tokens[i].clone())
            .unwrap_or_else(|| "[UNK]".into());
        // llama.cpp lower-cases BERT input, so the metadata does not say; assume the common uncased case
        WordPiece::finish(tokens, &unk, "##".into(), 100, true, true, true)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
    pub fn token_to_id(&self, t: &str) -> Option<u32> {
        self.vocab.get(t).copied()
    }
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens
            .get(id as usize)
            .map(String::as_str)
            .filter(|s| !s.is_empty())
    }
    pub fn is_special(&self, id: u32) -> bool {
        self.special.binary_search(&id).is_ok()
    }

    /// Clean, pad CJK, lower-case and strip accents.
    pub fn normalize(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        for c in text.chars().take(MAX_TEXT_CHARS) {
            if c == '\0' || c == '\u{fffd}' || is_control(c) {
                continue;
            }
            let c = if is_ws(c) { ' ' } else { c };
            if self.chinese_spaces && is_cjk(c) {
                out.push(' ');
                out.push(c);
                out.push(' ');
                continue;
            }
            if self.strip_accents {
                if (0x300..=0x36f).contains(&(c as u32)) {
                    continue;
                }
                if let Some(b) = base_letter(c) {
                    out.push_str_lower(b, self.lowercase);
                    continue;
                }
            }
            out.push_str_lower(c, self.lowercase);
        }
        out
    }

    /// Words and single punctuation marks of an already normalised text.
    fn split(&self, norm: &str) -> Vec<String> {
        let mut words = vec![];
        for w in norm.split(' ').filter(|w| !w.is_empty()) {
            let mut cur = String::new();
            for c in w.chars() {
                if is_punct(c) {
                    if !cur.is_empty() {
                        words.push(std::mem::take(&mut cur));
                    }
                    words.push(c.to_string());
                } else {
                    cur.push(c);
                }
            }
            if !cur.is_empty() {
                words.push(cur);
            }
        }
        words
    }

    /// Greedy longest-match-first pieces of one word; `[UNK]` when it cannot be covered.
    fn pieces(&self, word: &str, out: &mut Vec<u32>) {
        let chars: Vec<(usize, char)> = word.char_indices().collect();
        if chars.len() > self.max_chars {
            out.push(self.unk);
            return;
        }
        let mut found = vec![];
        let mut start = 0usize;
        while start < chars.len() {
            let mut end = chars.len();
            let mut hit = None;
            while start < end {
                let a = chars[start].0;
                let b = if end == chars.len() {
                    word.len()
                } else {
                    chars[end].0
                };
                let piece = &word[a..b];
                let id = if start > 0 {
                    self.vocab.get(&format!("{}{piece}", self.prefix))
                } else {
                    self.vocab.get(piece)
                };
                if let Some(&id) = id {
                    hit = Some(id);
                    break;
                }
                end -= 1;
            }
            match hit {
                Some(id) => {
                    found.push(id);
                    start = end;
                }
                None => {
                    out.push(self.unk);
                    return;
                }
            }
        }
        out.extend(found);
    }

    /// Token ids of `text` without `[CLS]`/`[SEP]`. Special-token spellings in the text (`[SEP]`…)
    /// become the special tokens.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![];
        let mut rest = text;
        loop {
            // the earliest special spelling
            let mut at: Option<(usize, usize, u32)> = None;
            for &id in &self.special {
                if let Some(t) = self.id_to_token(id).filter(|t| t.starts_with('['))
                    && let Some(p) = rest.find(t)
                    && at.is_none_or(|(q, _, _)| p < q)
                {
                    at = Some((p, t.len(), id));
                }
            }
            let (chunk, next) = match at {
                Some((p, l, id)) => (&rest[..p], Some((&rest[p + l..], id))),
                None => (rest, None),
            };
            for w in self.split(&self.normalize(chunk)) {
                self.pieces(&w, &mut ids);
            }
            match next {
                Some((r, id)) => {
                    ids.push(id);
                    rest = r;
                }
                None => break,
            }
        }
        ids
    }

    /// `[CLS] text [SEP]`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut v: Vec<u32> = self.cls.into_iter().collect();
        v.extend(self.tokenize(text));
        v.extend(self.sep);
        v
    }

    /// `[CLS] a [SEP] b [SEP]` with the segment id of every token (0 for the first text, 1 for the
    /// second), cut to at most `max_len` tokens by trimming the longer text first.
    pub fn encode_pair(&self, a: &str, b: &str, max_len: usize) -> (Vec<u32>, Vec<u32>) {
        let (mut x, mut y) = (self.tokenize(a), self.tokenize(b));
        let special = self.cls.is_some() as usize + 2 * self.sep.is_some() as usize;
        let budget = max_len.saturating_sub(special);
        while x.len() + y.len() > budget {
            if x.len() >= y.len() {
                x.pop();
            } else {
                y.pop();
            }
        }
        let mut ids: Vec<u32> = self.cls.into_iter().collect();
        ids.extend(&x);
        ids.extend(self.sep);
        let mut types = vec![0u32; ids.len()];
        ids.extend(&y);
        ids.extend(self.sep);
        types.resize(ids.len(), 1);
        (ids, types)
    }

    /// Text of `ids`: pieces joined, `##` pieces glued to the previous one, special tokens dropped.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if self.is_special(id) {
                continue;
            }
            let Some(t) = self.id_to_token(id) else {
                continue;
            };
            if let Some(r) = t.strip_prefix(self.prefix.as_str()) {
                out.push_str(r);
            } else {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(t);
            }
        }
        out
    }
}

trait PushLower {
    fn push_str_lower(&mut self, c: char, lower: bool);
}
impl PushLower for String {
    fn push_str_lower(&mut self, c: char, lower: bool) {
        if lower {
            self.extend(c.to_lowercase());
        } else {
            self.push(c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> WordPiece {
        let toks = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "un", "##aff",
            "##able", "!", ",", "a", "##s", "cafe", "sa", "##rut", "##a",
        ];
        let vocab: serde_json::Map<String, Value> = toks
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i.into()))
            .collect();
        let j = serde_json::json!({"model": {"type": "WordPiece", "vocab": vocab, "unk_token": "[UNK]"},
            "normalizer": {"type": "BertNormalizer", "lowercase": true}});
        WordPiece::from_json(&j.to_string()).unwrap()
    }

    #[test]
    fn greedy_longest_match_and_continuations() {
        let t = tiny();
        let id = |s: &str| t.token_to_id(s).unwrap();
        assert_eq!(
            t.tokenize("Hello, world!"),
            vec![id("hello"), id(","), id("world"), id("!")]
        );
        assert_eq!(
            t.tokenize("unaffable"),
            vec![id("un"), id("##aff"), id("##able")]
        );
        // no full cover: the whole word is [UNK]
        assert_eq!(t.tokenize("unaffordable"), vec![t.unk]);
        assert_eq!(t.tokenize(""), Vec::<u32>::new());
    }

    #[test]
    fn accents_case_and_whitespace_are_normalised() {
        let t = tiny();
        let id = |s: &str| t.token_to_id(s).unwrap();
        assert_eq!(t.tokenize("  CAFÉ\t"), vec![id("cafe")]);
        assert_eq!(t.tokenize("Șa"), vec![id("sa")]);
        // Vietnamese: stacked diacritics and horn letters reduce to the base letter
        assert_eq!(t.tokenize("Cửa"), t.tokenize("Cua"));
        assert_eq!(t.tokenize("ỆẠ"), t.tokenize("ea"));
        assert_eq!(t.tokenize("hello\u{0}"), vec![id("hello")]);
    }

    #[test]
    fn specials_in_text_and_pair_encoding() {
        let t = tiny();
        let id = |s: &str| t.token_to_id(s).unwrap();
        assert_eq!(
            t.encode("hello [SEP] world"),
            vec![
                id("[CLS]"),
                id("hello"),
                id("[SEP]"),
                id("world"),
                id("[SEP]")
            ]
        );
        let (ids, ty) = t.encode_pair("hello", "world !", 100);
        assert_eq!(
            ids,
            vec![
                id("[CLS]"),
                id("hello"),
                id("[SEP]"),
                id("world"),
                id("!"),
                id("[SEP]")
            ]
        );
        assert_eq!(ty, vec![0, 0, 0, 1, 1, 1]);
        // truncation keeps the specials and trims the longer side
        let (ids, ty) = t.encode_pair("hello hello hello hello", "world", 6);
        assert_eq!(ids.len(), 6);
        assert_eq!(ty.len(), 6);
        assert_eq!(*ids.last().unwrap(), id("[SEP]"));
        assert_eq!(ids[0], id("[CLS]"));
        assert!(ids.contains(&id("world")));
    }

    #[test]
    fn cjk_is_split_per_character_and_long_words_are_unknown() {
        let mut j: Value = serde_json::json!({"model": {"type": "WordPiece", "vocab": {"[UNK]": 0, "a": 1, "中": 2, "文": 3}, "unk_token": "[UNK]", "max_input_chars_per_word": 5}});
        let t = WordPiece::from_json(&j.to_string()).unwrap();
        assert_eq!(t.tokenize("中文"), vec![2, 3]);
        assert_eq!(t.tokenize("aaaaaa"), vec![0]);
        assert_eq!(t.tokenize("aaaaa"), vec![0]); // "aa…" has no ##a piece in this vocab
        j["model"]["vocab"]["##a"] = 4.into();
        let t = WordPiece::from_json(&j.to_string()).unwrap();
        assert_eq!(t.tokenize("aaaaa"), vec![1, 4, 4, 4, 4]);
    }

    #[test]
    fn decode_glues_continuations_and_drops_specials() {
        let t = tiny();
        let ids = t.encode("unaffable world");
        assert_eq!(t.decode(&ids), "unaffable world");
    }

    #[test]
    fn hostile_input_is_an_error_not_a_panic() {
        assert!(WordPiece::from_json("{").is_err());
        assert!(WordPiece::from_json(r#"{"model":{"type":"BPE","vocab":{}}}"#).is_err());
        assert!(WordPiece::from_json(r#"{"model":{"type":"WordPiece","vocab":{"a":0}}}"#).is_err());
        assert!(
            WordPiece::from_json(
                r#"{"model":{"type":"WordPiece","vocab":{"[UNK]":999999999999}}}"#
            )
            .is_err()
        );
        let t = tiny();
        let big = "x".repeat(200_000);
        assert_eq!(t.tokenize(&big), vec![t.unk]);
        assert!(!t.encode(&"hello ".repeat(10_000)).is_empty());
    }
}
