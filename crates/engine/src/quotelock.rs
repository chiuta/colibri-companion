//! Quote-locked decoding and citation verification.
//!
//! Research and citation workflows need a guarantee that anything the model
//! puts between quote markers really is in the provided sources.  This module
//! enforces that *while decoding*: every passage inside `«` … `»` must be a
//! verbatim substring (under a tolerant normalisation) of one of the source
//! passages, checked token by token.  An optional `[S3]` tag directly before
//! the opening marker restricts the quote to source number 3.
//!
//! Pieces:
//! * [`QuoteIndex`] – normalised suffix-array index over the sources, plus
//!   [`QuoteIndex::verify`] to audit finished text ([`Report`]).
//! * [`QuoteVocab`] – the token byte trie the masks are computed from.
//! * [`QuoteLock`] – the constraint (`mask()` / `accept(token)`), compatible
//!   in shape with `grammar::Constraint`.
//!
//! Normalisation (sources and model output alike): whitespace runs collapse,
//! NFC, optional case folding, curly/straight quotes and the hyphen/dash
//! family are equivalent, and cedilla/comma-below `ş ș ţ ț` are equivalent.
//! Citations always point at offsets (in chars) of the *original* source.
//!
//! Known limits: the model's output is assumed to be NFC already; a token that
//! ends in the middle of a character is only allowed if some source character
//! could complete it; the closing marker must be the last bytes of its token.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, OnceLock};

use crate::grammar::{BitSet, TokenTrie, apply_mask};
use crate::tokenizer::Tokenizer;

// ───────────────────────────── errors & options ─────────────────────────────

/// Errors from building an index or driving the constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum QuoteError {
    NoSources,
    EmptySource(usize),
    SourceTooLarge {
        total: usize,
        limit: usize,
    },
    /// The token was not allowed by the current mask.
    Violation(u32),
    /// The constraint reached a state with no way forward.
    Dead,
    /// The text holds more `«…»` passages than `Options::max_quotes` allows to be audited.
    TooManyQuotes {
        limit: usize,
    },
    /// Auditing the text would cost more than `Options::max_verify_work` units of work.
    WorkBudget {
        limit: u64,
    },
}

impl fmt::Display for QuoteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuoteError::NoSources => write!(f, "no source passages given"),
            QuoteError::EmptySource(i) => {
                write!(f, "source {} is empty after normalisation", i + 1)
            }
            QuoteError::SourceTooLarge { total, limit } => {
                write!(f, "sources total {total} bytes, limit is {limit}")
            }
            QuoteError::Violation(t) => write!(f, "token {t} violates the quote lock"),
            QuoteError::Dead => write!(f, "quote lock is in a dead state"),
            QuoteError::TooManyQuotes { limit } => {
                write!(f, "more than {limit} quoted passages to check")
            }
            QuoteError::WorkBudget { limit } => write!(
                f,
                "checking the quotes would exceed the work budget of {limit} units"
            ),
        }
    }
}
impl std::error::Error for QuoteError {}

type R<T> = Result<T, QuoteError>;

/// Tuning knobs.
#[derive(Debug, Clone)]
pub struct Options {
    /// A quote may only be closed after this many normalised chars (unless it
    /// reaches the end of its source).
    pub min_chars: usize,
    pub case_fold: bool,
    /// Total source size guard.
    pub max_source_bytes: usize,
    /// Edit-similarity at or above which a non-verbatim quote counts as altered.
    pub fuzzy_threshold: f32,
    /// Fold Latin letters with diacritics onto their base letter (ă â î ș ț → a a i s t, é → e, …)
    /// when matching. Off by default: `fată`/`fata` are different Romanian words. The
    /// cedilla/comma-below pairs ş/ș and ţ/ț are always treated as equal, with or without this.
    pub ignore_diacritics: bool,
    /// Most `«…»` passages [`QuoteIndex::verify_checked`] will audit in one call.
    pub max_quotes: usize,
    /// Budget for one `verify` call, in work units: edit-distance cells for fuzzy matching plus a
    /// few units per byte of every quote. The default allows about thirty worst-case fuzzy checks.
    pub max_verify_work: u64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            min_chars: 12,
            case_fold: true,
            max_source_bytes: 50 << 20,
            fuzzy_threshold: 0.8,
            ignore_diacritics: false,
            max_quotes: 256,
            max_verify_work: 1_000_000_000,
        }
    }
}

/// Where a quote came from.  `start`/`end` are char offsets into the original
/// source text; `source_id` is the 1-based tag number (`[S3]` → 3).
#[derive(Debug, Clone, PartialEq)]
pub struct Citation {
    pub source_id: usize,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

// ───────────────────────────── normalisation ─────────────────────────────

enum Fold {
    Skip,
    Space,
    Ch(char),
}

/// The base letter of a Latin letter with a diacritic (Romanian, French, German, Spanish, Italian,
/// Portuguese, Hungarian, Czech/Polish common cases); anything else is returned as is.
fn base_letter(c: char) -> char {
    match c {
        'ă' | 'â' | 'à' | 'á' | 'ä' | 'ã' | 'å' | 'ā' | 'ą' => 'a',
        'Ă' | 'Â' | 'À' | 'Á' | 'Ä' | 'Ã' | 'Å' | 'Ā' | 'Ą' => 'A',
        'î' | 'ì' | 'í' | 'ï' | 'ī' | 'į' | 'ı' => 'i',
        'Î' | 'Ì' | 'Í' | 'Ï' | 'Ī' | 'Į' | 'İ' => 'I',
        'ș' | 'ş' | 'š' | 'ś' => 's',
        'Ș' | 'Ş' | 'Š' | 'Ś' => 'S',
        'ț' | 'ţ' | 'ť' => 't',
        'Ț' | 'Ţ' | 'Ť' => 'T',
        'é' | 'è' | 'ê' | 'ë' | 'ē' | 'ę' | 'ě' => 'e',
        'É' | 'È' | 'Ê' | 'Ë' | 'Ē' | 'Ę' | 'Ě' => 'E',
        'ó' | 'ò' | 'ô' | 'ö' | 'õ' | 'ő' | 'ō' | 'ø' => 'o',
        'Ó' | 'Ò' | 'Ô' | 'Ö' | 'Õ' | 'Ő' | 'Ō' | 'Ø' => 'O',
        'ú' | 'ù' | 'û' | 'ü' | 'ű' | 'ū' | 'ů' => 'u',
        'Ú' | 'Ù' | 'Û' | 'Ü' | 'Ű' | 'Ū' | 'Ů' => 'U',
        'ç' | 'ć' | 'č' => 'c',
        'Ç' | 'Ć' | 'Č' => 'C',
        'ñ' | 'ń' | 'ň' => 'n',
        'Ñ' | 'Ń' | 'Ň' => 'N',
        'ý' | 'ÿ' => 'y',
        'Ý' | 'Ÿ' => 'Y',
        'ž' | 'ź' | 'ż' => 'z',
        'Ž' | 'Ź' | 'Ż' => 'Z',
        'ł' => 'l',
        'Ł' => 'L',
        'ď' | 'đ' => 'd',
        'Ď' | 'Đ' => 'D',
        'ř' => 'r',
        'Ř' => 'R',
        c => c,
    }
}

fn fold_ex(c: char, cf: bool, base: bool) -> Fold {
    match c {
        '\u{AD}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}' => {
            return Fold::Skip;
        }
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' | '\u{2039}'
        | '\u{203A}' => return Fold::Ch('\''),
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{AB}' | '\u{BB}' | '\u{2033}' => {
            return Fold::Ch('"');
        }
        '\u{2010}'..='\u{2015}' | '\u{2212}' | '\u{FE58}' | '\u{FE63}' | '\u{FF0D}' => {
            return Fold::Ch('-');
        }
        _ => {}
    }
    if c.is_whitespace() || c.is_control() {
        return Fold::Space;
    }
    let c = if base { base_letter(c) } else { c };
    let c = if cf {
        c.to_lowercase().next().unwrap_or(c)
    } else {
        c
    };
    Fold::Ch(match c {
        '\u{15F}' | '\u{219}' => '\u{219}',
        '\u{15E}' | '\u{218}' => '\u{218}',
        '\u{163}' | '\u{21B}' => '\u{21B}',
        '\u{162}' | '\u{21A}' => '\u{21A}',
        c => c,
    })
}

fn is_combining(c: char) -> bool {
    matches!(c as u32, 0x300..=0x36F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}

const SENTINEL: u8 = 0x1F;

struct Norm {
    bytes: Vec<u8>,
    cstart: Vec<u32>,
    ostart: Vec<u32>,
    oend: Vec<u32>,
}

/// Normalise `text`; with `map` also records, for every normalised char, its
/// byte offset and the original char range it came from.  The result always
/// ends with the sentinel byte.
#[cfg(test)]
fn normalize(text: &str, cf: bool, map: bool) -> Norm {
    normalize_ex(text, cf, false, map)
}

fn normalize_ex(text: &str, cf: bool, base: bool, map: bool) -> Norm {
    let mut n = Norm {
        bytes: Vec::with_capacity(text.len() + 1),
        cstart: vec![],
        ostart: vec![],
        oend: vec![],
    };
    let mut it = text.chars().peekable();
    let mut pos = 0u32;
    let mut last_space = true;
    let mut buf = String::new();
    while let Some(c) = it.next() {
        let start = pos;
        pos += 1;
        buf.clear();
        buf.push(c);
        while let Some(&m) = it.peek() {
            if !is_combining(m) {
                break;
            }
            buf.push(m);
            it.next();
            pos += 1;
        }
        let end = pos;
        let s: Cow<str> = if buf.len() > c.len_utf8() {
            Tokenizer::nfc(&buf)
        } else {
            Cow::Borrowed(buf.as_str())
        };
        for ch in s.chars() {
            match fold_ex(ch, cf, base) {
                Fold::Skip => {}
                Fold::Space => {
                    if !last_space {
                        push_char(&mut n, ' ', start, end, map);
                        last_space = true;
                    } else if map && !n.oend.is_empty() {
                        *n.oend.last_mut().unwrap() = end;
                    }
                }
                Fold::Ch(x) => {
                    push_char(&mut n, x, start, end, map);
                    last_space = false;
                }
            }
        }
    }
    if n.bytes.last() == Some(&b' ') {
        n.bytes.pop();
        if map {
            n.cstart.pop();
            n.ostart.pop();
            n.oend.pop();
        }
    }
    if map {
        n.cstart.push(n.bytes.len() as u32);
        n.ostart.push(pos);
        n.oend.push(pos);
    }
    n.bytes.push(SENTINEL);
    n
}

fn push_char(n: &mut Norm, x: char, s: u32, e: u32, map: bool) {
    if map {
        n.cstart.push(n.bytes.len() as u32);
        n.ostart.push(s);
        n.oend.push(e);
    }
    let mut b = [0u8; 4];
    n.bytes.extend_from_slice(x.encode_utf8(&mut b).as_bytes());
}

// ───────────────────────────── suffix array (SA-IS) ─────────────────────────────

const NONE: u32 = u32::MAX;

fn sa_is(s: &[u32], upper: usize) -> Vec<u32> {
    let n = s.len();
    match n {
        0 => return vec![],
        1 => return vec![0],
        2 => return if s[0] < s[1] { vec![0, 1] } else { vec![1, 0] },
        _ => {}
    }
    let mut sa = vec![0u32; n];
    let mut ls = vec![false; n];
    for i in (0..n - 1).rev() {
        ls[i] = if s[i] == s[i + 1] {
            ls[i + 1]
        } else {
            s[i] < s[i + 1]
        };
    }
    let mut sum_l = vec![0u32; upper + 1];
    let mut sum_s = vec![0u32; upper + 1];
    for i in 0..n {
        if !ls[i] {
            sum_s[s[i] as usize] += 1;
        } else {
            sum_l[s[i] as usize + 1] += 1;
        }
    }
    for i in 0..=upper {
        sum_s[i] += sum_l[i];
        if i < upper {
            sum_l[i + 1] += sum_s[i];
        }
    }
    let induce = |lms: &[u32], sa: &mut [u32]| {
        sa.iter_mut().for_each(|v| *v = NONE);
        let mut buf = sum_s.clone();
        for &d in lms {
            if d as usize == n {
                continue;
            }
            let c = s[d as usize] as usize;
            sa[buf[c] as usize] = d;
            buf[c] += 1;
        }
        buf.copy_from_slice(&sum_l);
        let c = s[n - 1] as usize;
        sa[buf[c] as usize] = (n - 1) as u32;
        buf[c] += 1;
        for i in 0..n {
            let v = sa[i];
            if v != NONE && v >= 1 && !ls[v as usize - 1] {
                let c = s[v as usize - 1] as usize;
                sa[buf[c] as usize] = v - 1;
                buf[c] += 1;
            }
        }
        buf.copy_from_slice(&sum_l);
        for i in (0..n).rev() {
            let v = sa[i];
            if v != NONE && v >= 1 && ls[v as usize - 1] {
                let c = s[v as usize - 1] as usize + 1;
                buf[c] -= 1;
                sa[buf[c] as usize] = v - 1;
            }
        }
    };
    let mut lms_map = vec![NONE; n + 1];
    let mut m = 0u32;
    for i in 1..n {
        if !ls[i - 1] && ls[i] {
            lms_map[i] = m;
            m += 1;
        }
    }
    let mut lms: Vec<u32> = Vec::with_capacity(m as usize);
    for i in 1..n {
        if !ls[i - 1] && ls[i] {
            lms.push(i as u32);
        }
    }
    induce(&lms, &mut sa);
    if m > 0 {
        let mut sorted: Vec<u32> = Vec::with_capacity(m as usize);
        for &v in &sa {
            if v != NONE && lms_map[v as usize] != NONE {
                sorted.push(v);
            }
        }
        let mut rec_s = vec![0u32; m as usize];
        let mut rec_upper = 0usize;
        rec_s[lms_map[sorted[0] as usize] as usize] = 0;
        for i in 1..m as usize {
            let mut l = sorted[i - 1] as usize;
            let mut r = sorted[i] as usize;
            let end_l = if lms_map[l] + 1 < m {
                lms[lms_map[l] as usize + 1] as usize
            } else {
                n
            };
            let end_r = if lms_map[r] + 1 < m {
                lms[lms_map[r] as usize + 1] as usize
            } else {
                n
            };
            let mut same = true;
            if end_l - l != end_r - r {
                same = false;
            } else {
                while l < end_l {
                    if s[l] != s[r] {
                        break;
                    }
                    l += 1;
                    r += 1;
                }
                if l == n || s[l] != s[r] {
                    same = false;
                }
            }
            if !same {
                rec_upper += 1;
            }
            rec_s[lms_map[sorted[i] as usize] as usize] = rec_upper as u32;
        }
        let rec_sa = sa_is(&rec_s, rec_upper);
        for i in 0..m as usize {
            sorted[i] = lms[rec_sa[i] as usize];
        }
        induce(&sorted, &mut sa);
    }
    sa
}

fn suffix_array(text: &[u8]) -> Vec<u32> {
    let s: Vec<u32> = text.iter().map(|&b| b as u32).collect();
    sa_is(&s, 255)
}

// ───────────────────────────── the index ─────────────────────────────

struct SrcIdx {
    text: Vec<u8>,
    sa: Vec<u32>,
    cstart: Vec<u32>,
    ostart: Vec<u32>,
    oend: Vec<u32>,
    orig: String,
}

impl SrcIdx {
    #[inline]
    fn byte_at(&self, p: u32, d: u32) -> u8 {
        self.text.get((p + d) as usize).copied().unwrap_or(0)
    }
    /// Index of the normalised char containing byte offset `b`.
    fn nchar_of(&self, b: u32) -> usize {
        self.cstart.partition_point(|&c| c <= b).saturating_sub(1)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Iv {
    src: u32,
    lo: u32,
    hi: u32,
}

#[derive(Default)]
struct WordStats {
    tf: HashMap<u64, u32>,
    g2: HashSet<u64>,
    g3: HashSet<u64>,
    len: u32,
}

/// Normalised, searchable view of the source passages.
pub struct QuoteIndex {
    srcs: Vec<SrcIdx>,
    opts: Options,
    prefixes: HashMap<Vec<u8>, Vec<char>>,
    words: OnceLock<Vec<WordStats>>,
}

enum Step {
    Ok(usize, usize),
    Dead,
    Closed,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct QSt {
    pend: [u8; 4],
    plen: u8,
    need: u8,
    space: bool,
    idle: u8,
    nch: u32,
    d: u32,
}

impl QSt {
    fn fresh() -> QSt {
        QSt {
            pend: [0; 4],
            plen: 0,
            need: 0,
            space: false,
            idle: 0,
            nch: 0,
            d: 0,
        }
    }
}

fn add_prefixes(m: &mut HashMap<Vec<u8>, Vec<char>>, v: char) {
    let mut b = [0u8; 4];
    let e = v.encode_utf8(&mut b).as_bytes();
    for k in 1..e.len() {
        let l = m.entry(e[..k].to_vec()).or_default();
        if !l.contains(&v) {
            l.push(v);
        }
    }
}

fn variants(n: char, out: &mut Vec<char>) {
    out.push(n);
    out.extend(n.to_uppercase());
    let extra: &[char] = match n {
        '"' => &['\u{201C}', '\u{201D}', '\u{201E}', '\u{201F}', '\u{2033}'],
        '\'' => &[
            '\u{2018}', '\u{2019}', '\u{201A}', '\u{201B}', '\u{2032}', '\u{2039}', '\u{203A}',
        ],
        '-' => &[
            '\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2015}', '\u{2212}',
        ],
        ' ' => &[
            '\u{A0}', '\u{2002}', '\u{2003}', '\u{2009}', '\u{202F}', '\u{3000}', '\u{2028}',
        ],
        '\u{219}' => &['\u{15F}', '\u{15E}', '\u{218}'],
        '\u{21B}' => &['\u{163}', '\u{162}', '\u{21A}'],
        _ => &[],
    };
    out.extend_from_slice(extra);
}

impl QuoteIndex {
    /// Index `sources` (source `i` is cited as `[S{i+1}]`).
    pub fn new(sources: &[&str], opts: Options) -> R<QuoteIndex> {
        if sources.is_empty() {
            return Err(QuoteError::NoSources);
        }
        let total: usize = sources.iter().map(|s| s.len()).sum();
        if total > opts.max_source_bytes {
            return Err(QuoteError::SourceTooLarge {
                total,
                limit: opts.max_source_bytes,
            });
        }
        let mut srcs = Vec::with_capacity(sources.len());
        for (i, s) in sources.iter().enumerate() {
            let n = normalize_ex(s, opts.case_fold, opts.ignore_diacritics, true);
            if n.bytes.len() <= 1 {
                return Err(QuoteError::EmptySource(i));
            }
            let sa = suffix_array(&n.bytes);
            srcs.push(SrcIdx {
                text: n.bytes,
                sa,
                cstart: n.cstart,
                ostart: n.ostart,
                oend: n.oend,
                orig: (*s).to_string(),
            });
        }
        let mut chars: HashSet<char> = HashSet::new();
        for s in &srcs {
            if let Ok(t) = std::str::from_utf8(&s.text) {
                chars.extend(t.chars());
            }
        }
        chars.insert(' ');
        let mut prefixes: HashMap<Vec<u8>, Vec<char>> = HashMap::new();
        let mut vs = Vec::new();
        for &c in &chars {
            vs.clear();
            variants(c, &mut vs);
            vs.dedup();
            for &v in &vs {
                add_prefixes(&mut prefixes, v);
            }
        }
        for c in [
            '\u{BB}', '\u{AD}', '\u{200B}', '\u{200C}', '\u{200D}', '\u{2060}', '\u{FEFF}',
        ] {
            add_prefixes(&mut prefixes, c);
        }
        Ok(QuoteIndex {
            srcs,
            opts,
            prefixes,
            words: OnceLock::new(),
        })
    }

    pub fn source_count(&self) -> usize {
        self.srcs.len()
    }

    fn all_ivs(&self, restrict: Option<usize>) -> Vec<Iv> {
        self.srcs
            .iter()
            .enumerate()
            .filter(|(i, _)| restrict.is_none_or(|r| r == *i))
            .map(|(i, s)| Iv {
                src: i as u32,
                lo: 0,
                hi: s.sa.len() as u32,
            })
            .collect()
    }

    fn narrow(&self, iv: Iv, d: u32, b: u8) -> Option<Iv> {
        let s = &self.srcs[iv.src as usize];
        let sl = &s.sa[iv.lo as usize..iv.hi as usize];
        let a = sl.partition_point(|&p| s.byte_at(p, d) < b);
        let z = sl.partition_point(|&p| s.byte_at(p, d) <= b);
        if a == z {
            None
        } else {
            Some(Iv {
                src: iv.src,
                lo: iv.lo + a as u32,
                hi: iv.lo + z as u32,
            })
        }
    }

    /// Is there an occurrence followed by a non-alphanumeric byte (or the end)?
    fn boundary(&self, ivs: &[Iv], d: u32) -> bool {
        const RANGES: [(u8, u8); 4] = [(0, 0x30), (0x3A, 0x41), (0x5B, 0x61), (0x7B, 0x80)];
        for iv in ivs {
            let s = &self.srcs[iv.src as usize];
            let sl = &s.sa[iv.lo as usize..iv.hi as usize];
            for (a, z) in RANGES {
                let x = sl.partition_point(|&p| s.byte_at(p, d) < a);
                let y = sl.partition_point(|&p| s.byte_at(p, d) < z);
                if x < y {
                    return true;
                }
            }
        }
        false
    }

    fn reaches_end(&self, ivs: &[Iv], d: u32) -> bool {
        ivs.iter().any(|iv| {
            let s = &self.srcs[iv.src as usize];
            s.byte_at(s.sa[iv.lo as usize], d) <= SENTINEL
        })
    }

    fn feed(&self, arena: &mut Vec<Iv>, fr: (usize, usize), st: &mut QSt, b: u8) -> Step {
        if st.plen == 0 {
            st.need = match b {
                0..=0x7F => 1,
                0xC2..=0xDF => 2,
                0xE0..=0xEF => 3,
                0xF0..=0xF4 => 4,
                _ => return Step::Dead,
            };
            st.pend[0] = b;
            st.plen = 1;
        } else {
            if b & 0xC0 != 0x80 {
                return Step::Dead;
            }
            st.pend[st.plen as usize] = b;
            st.plen += 1;
        }
        if st.plen < st.need {
            return Step::Ok(fr.0, fr.1);
        }
        let n = st.plen as usize;
        st.plen = 0;
        let c = match std::str::from_utf8(&st.pend[..n]) {
            Ok(s) => s.chars().next().unwrap(),
            Err(_) => return Step::Dead,
        };
        self.feed_char(arena, fr, st, c)
    }

    fn feed_char(&self, arena: &mut Vec<Iv>, fr: (usize, usize), st: &mut QSt, c: char) -> Step {
        if c == '\u{BB}' {
            let cur = &arena[fr.0..fr.1];
            let ok = st.nch > 0
                && (st.nch as usize >= self.opts.min_chars || self.reaches_end(cur, st.d))
                && self.boundary(cur, st.d);
            return if ok { Step::Closed } else { Step::Dead };
        }
        if c == '\u{AB}' {
            return Step::Dead;
        }
        match fold_ex(c, self.opts.case_fold, self.opts.ignore_diacritics) {
            Fold::Skip => self.idle_step(fr, st),
            Fold::Space => {
                if st.nch == 0 || st.space {
                    return self.idle_step(fr, st);
                }
                // a real separator: consume it now so dead ends are detected early
                match self.push_bytes(arena, fr, st, b" ") {
                    Some(cur) => {
                        st.space = true;
                        st.idle = 0;
                        Step::Ok(cur.0, cur.1)
                    }
                    None => Step::Dead,
                }
            }
            Fold::Ch(x) => {
                let mut eb = [0u8; 4];
                let e = x.encode_utf8(&mut eb).as_bytes();
                match self.push_bytes(arena, fr, st, e) {
                    Some(cur) => {
                        st.space = false;
                        st.idle = 0;
                        Step::Ok(cur.0, cur.1)
                    }
                    None => Step::Dead,
                }
            }
        }
    }

    /// Chars that add nothing (extra whitespace, zero-width) are capped so a
    /// quote cannot stall on them.
    fn idle_step(&self, fr: (usize, usize), st: &mut QSt) -> Step {
        st.idle += 1;
        if st.idle > 2 {
            Step::Dead
        } else {
            Step::Ok(fr.0, fr.1)
        }
    }

    fn push_bytes(
        &self,
        arena: &mut Vec<Iv>,
        fr: (usize, usize),
        st: &mut QSt,
        bytes: &[u8],
    ) -> Option<(usize, usize)> {
        let mut cur = fr;
        let mut d = st.d;
        for &b in bytes {
            let start = arena.len();
            for i in cur.0..cur.1 {
                let iv = arena[i];
                if let Some(n) = self.narrow(iv, d, b) {
                    arena.push(n);
                }
            }
            if arena.len() == start {
                return None;
            }
            cur = (start, arena.len());
            d += 1;
        }
        st.d = d;
        st.nch += 1;
        Some(cur)
    }

    /// Can the partially received character be completed into something that
    /// still matches a source?
    fn completable(&self, arena: &mut Vec<Iv>, fr: (usize, usize), st: &QSt) -> bool {
        let plen = st.plen as usize;
        let Some(cs) = self.prefixes.get(&st.pend[..plen]) else {
            return false;
        };
        for &c in cs {
            let save = arena.len();
            let mut s2 = *st;
            let mut e = [0u8; 4];
            let enc = c.encode_utf8(&mut e).as_bytes();
            let mut cur = fr;
            let mut ok = true;
            for &b in &enc[plen..] {
                match self.feed(arena, cur, &mut s2, b) {
                    Step::Ok(a, z) => cur = (a, z),
                    Step::Closed => {}
                    Step::Dead => {
                        ok = false;
                        break;
                    }
                }
            }
            arena.truncate(save);
            if ok {
                return true;
            }
        }
        false
    }

    fn dfs(
        &self,
        v: &QuoteVocab,
        node: u32,
        st: &QSt,
        fr: (usize, usize),
        arena: &mut Vec<Iv>,
        out: &mut BitSet,
    ) {
        let (ks, ke) = (
            v.kid_start[node as usize] as usize,
            v.kid_start[node as usize + 1] as usize,
        );
        for &(b, child) in &v.kids[ks..ke] {
            let save = arena.len();
            let mut s2 = *st;
            match self.feed(arena, fr, &mut s2, b) {
                Step::Ok(a, z) => {
                    if s2.plen == 0 || self.completable(arena, (a, z), &s2) {
                        v.set_tokens(child, out);
                    }
                    self.dfs(v, child, &s2, (a, z), arena, out);
                }
                Step::Closed => v.set_tokens(child, out),
                Step::Dead => {}
            }
            arena.truncate(save);
        }
    }

    fn build_mask(&self, v: &QuoteVocab, ivs: &[Iv], st: &QSt) -> BitSet {
        let mut arena = ivs.to_vec();
        let mut out = BitSet::new(v.n);
        let fr = (0, arena.len());
        self.dfs(v, 0, st, fr, &mut arena, &mut out);
        out
    }

    fn cache_key(&self, ivs: &[Iv], st: &QSt) -> Vec<u32> {
        let mut k = Vec::with_capacity(4 + ivs.len() * 3);
        k.push(st.nch.min(self.opts.min_chars as u32));
        k.push(st.d);
        k.push(u32::from_le_bytes(st.pend));
        k.push(
            st.plen as u32
                | (st.need as u32) << 8
                | (st.space as u32) << 16
                | (st.idle as u32) << 24,
        );
        for iv in ivs {
            k.extend_from_slice(&[iv.src, iv.lo, iv.hi]);
        }
        k
    }

    fn citation_for(&self, src: usize, pos: u32, len: u32) -> Citation {
        let s = &self.srcs[src];
        let a = s.nchar_of(pos);
        let z = s.nchar_of(pos + len.max(1) - 1);
        let (os, oe) = (s.ostart[a] as usize, s.oend[z] as usize);
        let text: String = s
            .orig
            .chars()
            .skip(os)
            .take(oe.saturating_sub(os))
            .collect();
        Citation {
            source_id: src + 1,
            start: os,
            end: oe,
            text,
        }
    }

    fn citation_from_ivs(&self, ivs: &[Iv], d: u32) -> Option<Citation> {
        let iv = ivs.first()?;
        let s = &self.srcs[iv.src as usize];
        let hi = iv.hi.min(iv.lo + 4096);
        let pos = s.sa[iv.lo as usize..hi as usize].iter().copied().min()?;
        Some(self.citation_for(iv.src as usize, pos, d))
    }

    /// Longest prefix of `q` found in any allowed source: (len, ivs).
    fn longest_prefix(&self, q: &[u8], restrict: Option<usize>) -> (usize, Vec<Iv>) {
        let mut best = (0usize, Vec::new());
        for iv0 in self.all_ivs(restrict) {
            let mut iv = iv0;
            let mut len = 0;
            for (d, &b) in q.iter().enumerate() {
                match self.narrow(iv, d as u32, b) {
                    Some(n) => {
                        iv = n;
                        len = d + 1;
                    }
                    None => break,
                }
            }
            if len > best.0 {
                best = (len, vec![iv]);
            }
        }
        best
    }

    /// First verbatim occurrence of the (already normalised) bytes.
    fn find_exact(&self, nq: &[u8], restrict: Option<usize>) -> Option<(usize, u32)> {
        for iv0 in self.all_ivs(restrict) {
            let mut iv = iv0;
            let mut ok = true;
            for (d, &b) in nq.iter().enumerate() {
                match self.narrow(iv, d as u32, b) {
                    Some(n) => iv = n,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                let s = &self.srcs[iv.src as usize];
                let hi = iv.hi.min(iv.lo + 4096);
                let pos = s.sa[iv.lo as usize..hi as usize].iter().copied().min()?;
                return Some((iv.src as usize, pos));
            }
        }
        None
    }

    /// Find `quote` verbatim (under normalisation) in the sources.
    pub fn find(&self, quote: &str) -> Option<Citation> {
        let n = normalize_ex(
            quote,
            self.opts.case_fold,
            self.opts.ignore_diacritics,
            false,
        );
        let nq = &n.bytes[..n.bytes.len() - 1];
        if nq.is_empty() {
            return None;
        }
        let (src, pos) = self.find_exact(nq, None)?;
        Some(self.citation_for(src, pos, nq.len() as u32))
    }

    // ───────────── verification ─────────────

    fn word_stats(&self) -> &Vec<WordStats> {
        self.words.get_or_init(|| {
            self.srcs
                .iter()
                .map(|s| {
                    let mut w = WordStats::default();
                    let hs = word_hashes(&s.text);
                    w.len = hs.len() as u32;
                    for (i, &h) in hs.iter().enumerate() {
                        *w.tf.entry(h).or_insert(0) += 1;
                        if i >= 1 {
                            w.g2.insert(gram(&hs[i - 1..=i]));
                        }
                        if i >= 2 {
                            w.g3.insert(gram(&hs[i - 2..=i]));
                        }
                    }
                    w
                })
                .collect()
        })
    }

    /// Lexical support of `text` by the sources, in `[0, 1]`: word n-gram
    /// containment blended with a normalised BM25 score; best source wins.
    pub fn support(&self, text: &str) -> f32 {
        let n = normalize_ex(
            text,
            self.opts.case_fold,
            self.opts.ignore_diacritics,
            false,
        );
        let hs = word_hashes(&n.bytes);
        if hs.is_empty() {
            return 0.0;
        }
        let stats = self.word_stats();
        let (k1, b) = (1.2f32, 0.75f32);
        let nsrc = stats.len() as f32;
        let avg = stats.iter().map(|w| w.len as f32).sum::<f32>() / nsrc;
        let mut uniq: Vec<u64> = hs.clone();
        uniq.sort_unstable();
        uniq.dedup();
        let idf: Vec<f32> = uniq
            .iter()
            .map(|h| {
                let df = stats.iter().filter(|w| w.tf.contains_key(h)).count() as f32;
                (1.0 + (nsrc - df + 0.5) / (df + 0.5)).ln()
            })
            .collect();
        let ideal: f32 = idf.iter().map(|i| i * (k1 + 1.0)).sum::<f32>().max(1e-9);
        let mut best = 0.0f32;
        for w in stats {
            let mut sc = 0.0;
            for (h, i) in uniq.iter().zip(&idf) {
                if let Some(&tf) = w.tf.get(h) {
                    let tf = tf as f32;
                    sc += i * tf * (k1 + 1.0)
                        / (tf + k1 * (1.0 - b + b * w.len as f32 / avg.max(1.0)));
                }
            }
            let bm = (sc / ideal).clamp(0.0, 1.0);
            let (grams, set): (Vec<u64>, &HashSet<u64>) = if hs.len() >= 3 {
                (hs.windows(3).map(gram).collect(), &w.g3)
            } else if hs.len() == 2 {
                (vec![gram(&hs)], &w.g2)
            } else {
                (hs.iter().map(|h| gram(&[*h])).collect(), &w.g2)
            };
            let cont = if hs.len() == 1 {
                if w.tf.contains_key(&hs[0]) { 1.0 } else { 0.0 }
            } else {
                grams.iter().filter(|g| set.contains(g)).count() as f32 / grams.len().max(1) as f32
            };
            best = best.max(0.7 * cont + 0.3 * bm);
        }
        best.clamp(0.0, 1.0)
    }

    fn fuzzy(
        &self,
        ns: &str,
        restrict: Option<usize>,
        bud: &mut Budget,
    ) -> Option<(f32, Citation)> {
        let q: Vec<char> = ns.chars().collect();
        let m = q.len();
        if !(6..=1500).contains(&m) {
            return None;
        }
        let nq = ns.as_bytes();
        let stride = (nq.len() / 16).max(4);
        let mut windows: Vec<(usize, usize, usize)> = Vec::new(); // src, nstart, nend
        let mut p = 0;
        while p < nq.len() && windows.len() < 8 {
            if ns.is_char_boundary(p) {
                let (len, ivs) = self.longest_prefix(&nq[p..], restrict);
                if len >= 5 {
                    let iv = ivs[0];
                    let s = &self.srcs[iv.src as usize];
                    for &pos in s.sa[iv.lo as usize..iv.hi.min(iv.lo + 3) as usize].iter() {
                        let c0 = s.nchar_of(pos);
                        let off = ns[..p].chars().count();
                        let slack = m / 4 + 2;
                        let a = c0.saturating_sub(off + slack);
                        let z = (c0 + (m - off) + slack).min(s.cstart.len() - 1);
                        if !windows
                            .iter()
                            .any(|w| w.0 == iv.src as usize && w.1.abs_diff(a) < 4)
                            && windows.len() < 8
                        {
                            windows.push((iv.src as usize, a, z));
                        }
                    }
                }
            }
            p += stride;
        }
        let mut best: Option<(u32, usize, usize, usize, usize)> = None; // cost, src, nstart, ws, we
        for (src, a, z) in windows {
            let s = &self.srcs[src];
            let (ba, bz) = (s.cstart[a] as usize, s.cstart[z] as usize);
            let w: Vec<char> = match std::str::from_utf8(&s.text[ba..bz]) {
                Ok(t) => t.chars().collect(),
                Err(_) => continue,
            };
            if w.is_empty() || w.len() > 2600 {
                continue;
            }
            if !bud.charge((q.len() as u64 + 1) * (w.len() as u64 + 1)) {
                break;
            }
            let (cost, st, en) = sellers(&q, &w);
            if best.is_none_or(|b| cost < b.0) {
                best = Some((cost, src, a, st, en));
            }
        }
        let (cost, src, a, st, en) = best?;
        if en <= st {
            return None;
        }
        let sim = 1.0 - cost as f32 / m as f32;
        let s = &self.srcs[src];
        let (os, oe) = (s.ostart[a + st] as usize, s.oend[a + en - 1] as usize);
        let text: String = s
            .orig
            .chars()
            .skip(os)
            .take(oe.saturating_sub(os))
            .collect();
        Some((
            sim.max(0.0),
            Citation {
                source_id: src + 1,
                start: os,
                end: oe,
                text,
            },
        ))
    }

    fn check_quote(&self, rq: &RawQuote, bud: &mut Budget) -> QuoteCheck {
        let restrict = rq
            .tag
            .filter(|t| *t >= 1 && *t <= self.srcs.len())
            .map(|t| t - 1);
        let n = normalize_ex(
            &rq.text,
            self.opts.case_fold,
            self.opts.ignore_diacritics,
            false,
        );
        let nq = &n.bytes[..n.bytes.len() - 1];
        let mk = |status, citation, similarity, support| QuoteCheck {
            text: rq.text.clone(),
            tag: rq.tag,
            closed: rq.closed,
            status,
            citation,
            similarity,
            support,
            char_start: rq.start,
            char_end: rq.end,
        };
        if nq.is_empty() {
            return mk(QuoteStatus::Unsupported, None, 0.0, 0.0);
        }
        let ns = String::from_utf8_lossy(nq).into_owned();
        if !bud.charge(8 * nq.len() as u64 + 64) {
            return mk(QuoteStatus::Unsupported, None, 0.0, 0.0);
        }
        let support = self.support(&ns);
        let found = self
            .find_exact(nq, restrict)
            .map(|x| (x, true))
            .or_else(|| {
                if restrict.is_some() {
                    self.find_exact(nq, None).map(|x| (x, false))
                } else {
                    None
                }
            });
        if let Some(((src, pos), in_tag)) = found {
            let cit = self.citation_for(src, pos, nq.len() as u32);
            if !in_tag {
                return mk(
                    QuoteStatus::WrongSource { found_in: src + 1 },
                    Some(cit),
                    1.0,
                    support,
                );
            }
            let s = &self.srcs[src];
            let alnum = |b: u8| b.is_ascii_alphanumeric() || b >= 0x80;
            let before = if pos > 0 {
                s.text[pos as usize - 1]
            } else {
                b' '
            };
            let after = s.byte_at(pos, nq.len() as u32);
            let mid =
                (alnum(before) && alnum(nq[0])) || (alnum(after) && alnum(*nq.last().unwrap()));
            return mk(
                if mid {
                    QuoteStatus::Truncated
                } else {
                    QuoteStatus::Verbatim
                },
                Some(cit),
                1.0,
                support,
            );
        }
        // not verbatim: fuzzy, then merged fragments
        let fz = self.fuzzy(&ns, restrict, bud);
        let sim = fz.as_ref().map_or(0.0, |x| x.0);
        if let Some((sim, cit)) = fz.filter(|x| x.0 >= self.opts.fuzzy_threshold) {
            return mk(QuoteStatus::Altered, Some(cit), sim, support);
        }
        let mut pieces = 0usize;
        let mut covered = 0usize;
        let mut first: Option<Citation> = None;
        let mut p = 0;
        while p < nq.len() {
            let (mut len, ivs) = self.longest_prefix(&nq[p..], restrict);
            while len > 0 && !ns.is_char_boundary(p + len) {
                len -= 1;
            }
            if len >= 6 {
                pieces += 1;
                covered += len;
                if first.is_none() {
                    first = self.citation_from_ivs(&ivs, len as u32);
                }
                p += len;
            } else {
                p += 1;
            }
        }
        if pieces >= 2 && covered as f32 >= 0.8 * nq.len() as f32 {
            return mk(QuoteStatus::Merged, first, sim, support);
        }
        mk(QuoteStatus::Unsupported, None, sim, support)
    }

    /// Audit finished text: every `«…»` passage is classified; sentences
    /// outside quotes get a lexical support score.  Best effort under the
    /// [`Options`] limits: passages beyond `max_quotes`, or work beyond
    /// `max_verify_work`, are left unchecked (reported `Unsupported`) and
    /// `Report::limit_hit` is set.  Use [`QuoteIndex::verify_checked`] to get an error instead.
    pub fn verify(&self, output: &str) -> Report {
        self.verify_inner(output, false)
            .expect("best-effort verify never errors")
    }

    /// Like [`QuoteIndex::verify`], but returns `TooManyQuotes` / `WorkBudget`
    /// instead of a partial report when a limit is reached.
    pub fn verify_checked(&self, output: &str) -> R<Report> {
        self.verify_inner(output, true)
    }

    fn verify_inner(&self, output: &str, strict: bool) -> R<Report> {
        let (quotes, rest, too_many) = extract_quotes(output, self.opts.max_quotes);
        if too_many && strict {
            return Err(QuoteError::TooManyQuotes {
                limit: self.opts.max_quotes,
            });
        }
        let mut bud = Budget::new(self.opts.max_verify_work);
        let checks: Vec<QuoteCheck> = quotes
            .iter()
            .map(|q| self.check_quote(q, &mut bud))
            .collect();
        let mut sentences = Vec::new();
        for s in rest.split(['.', '!', '?', '\n']) {
            let t = s.trim();
            if t.split_whitespace().count() >= 3 {
                if !bud.charge(4 * t.len() as u64 + 16) {
                    break;
                }
                sentences.push(ClaimSupport {
                    text: t.to_string(),
                    support: self.support(t),
                });
            }
        }
        if bud.hit && strict {
            return Err(QuoteError::WorkBudget {
                limit: self.opts.max_verify_work,
            });
        }
        let count = |f: fn(&QuoteStatus) -> bool| checks.iter().filter(|c| f(&c.status)).count();
        Ok(Report {
            verbatim: count(|s| matches!(s, QuoteStatus::Verbatim)),
            wrong_source: count(|s| matches!(s, QuoteStatus::WrongSource { .. })),
            truncated: count(|s| matches!(s, QuoteStatus::Truncated)),
            altered: count(|s| matches!(s, QuoteStatus::Altered)),
            merged: count(|s| matches!(s, QuoteStatus::Merged)),
            unsupported: count(|s| matches!(s, QuoteStatus::Unsupported)),
            quotes: checks,
            sentences,
            limit_hit: too_many || bud.hit,
        })
    }
}

/// Work counter for one `verify` call.
struct Budget {
    left: u64,
    hit: bool,
}

impl Budget {
    fn new(limit: u64) -> Budget {
        Budget {
            left: limit,
            hit: false,
        }
    }
    /// Spend `n` units; false (and the sticky `hit` flag) when they are not there.
    fn charge(&mut self, n: u64) -> bool {
        if self.hit || n > self.left {
            self.hit = true;
            return false;
        }
        self.left -= n;
        true
    }
}

/// Convenience: index `sources` and verify `output` in one call.
pub fn verify_text(output: &str, sources: &[&str]) -> R<Report> {
    Ok(QuoteIndex::new(sources, Options::default())?.verify(output))
}

fn word_hashes(norm: &[u8]) -> Vec<u64> {
    let mut out = Vec::new();
    let mut h: u64 = 0xcbf29ce484222325;
    let mut inw = false;
    for &b in norm {
        let alnum = b.is_ascii_alphanumeric() || b >= 0x80;
        if alnum {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            inw = true;
        } else if inw {
            out.push(h);
            h = 0xcbf29ce484222325;
            inw = false;
        }
    }
    if inw {
        out.push(h);
    }
    out
}

fn gram(hs: &[u64]) -> u64 {
    hs.iter().fold(0x9E3779B97F4A7C15u64, |a, &h| {
        (a.rotate_left(23) ^ h).wrapping_mul(0x100000001b3)
    })
}

/// Semi-global edit distance of `q` against any substring of `w`:
/// (cost, start, end) with `w[start..end]` the best window.
fn sellers(q: &[char], w: &[char]) -> (u32, usize, usize) {
    let mut prev: Vec<(u32, u32)> = (0..=w.len() as u32).map(|j| (0, j)).collect();
    let mut cur = vec![(0u32, 0u32); w.len() + 1];
    for i in 1..=q.len() {
        cur[0] = (i as u32, 0);
        for j in 1..=w.len() {
            let diag = (prev[j - 1].0 + (q[i - 1] != w[j - 1]) as u32, prev[j - 1].1);
            let up = (prev[j].0 + 1, prev[j].1);
            let left = (cur[j - 1].0 + 1, cur[j - 1].1);
            let mut b = diag;
            if up.0 < b.0 {
                b = up;
            }
            if left.0 < b.0 {
                b = left;
            }
            cur[j] = b;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let (mut bj, mut bc) = (0usize, u32::MAX);
    for (j, c) in prev.iter().enumerate() {
        if c.0 < bc {
            bc = c.0;
            bj = j;
        }
    }
    (bc, prev[bj].1 as usize, bj)
}

// ───────────────────────────── report types ─────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum QuoteStatus {
    /// Exact (normalised) substring of the cited source.
    Verbatim,
    /// Verbatim, but not in the source its `[Sn]` tag names.
    WrongSource {
        found_in: usize,
    },
    /// Verbatim but cut in the middle of a word at either end.
    Truncated,
    /// Close to a passage but with edits.
    Altered,
    /// Stitched together from separate fragments.
    Merged,
    Unsupported,
}

#[derive(Debug, Clone)]
pub struct QuoteCheck {
    pub text: String,
    pub tag: Option<usize>,
    pub closed: bool,
    pub status: QuoteStatus,
    pub citation: Option<Citation>,
    /// Best edit similarity found (1.0 for exact).
    pub similarity: f32,
    pub support: f32,
    pub char_start: usize,
    pub char_end: usize,
}

#[derive(Debug, Clone)]
pub struct ClaimSupport {
    pub text: String,
    pub support: f32,
}

#[derive(Debug, Clone)]
pub struct Report {
    pub quotes: Vec<QuoteCheck>,
    pub sentences: Vec<ClaimSupport>,
    pub verbatim: usize,
    pub wrong_source: usize,
    pub truncated: usize,
    pub altered: usize,
    pub merged: usize,
    pub unsupported: usize,
    /// A work or quote-count limit stopped the audit early; unchecked quotes count as unsupported.
    pub limit_hit: bool,
}

impl Report {
    /// True when there is at least one quote and all are verbatim and closed.
    pub fn all_verbatim(&self) -> bool {
        !self.quotes.is_empty()
            && self
                .quotes
                .iter()
                .all(|q| q.status == QuoteStatus::Verbatim && q.closed)
    }
}

struct RawQuote {
    tag: Option<usize>,
    text: String,
    start: usize,
    end: usize,
    closed: bool,
}

/// Split output into quotes and the text outside them.  A second `«` inside a
/// quote is kept as ordinary text, so nesting cannot recurse.
fn extract_quotes(out: &str, max_quotes: usize) -> (Vec<RawQuote>, String, bool) {
    let chars: Vec<char> = out.chars().collect();
    let mut quotes = Vec::new();
    let mut rest = String::with_capacity(out.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\u{AB}' {
            if quotes.len() >= max_quotes {
                return (quotes, rest, true);
            }
            // tag directly before?  `[S<digits>]` is ASCII and was just pushed to `rest`,
            // so it is removed by truncating bytes: no re-scan, linear overall.
            let mut tag = None;
            if i >= 1 && chars[i - 1] == ']' {
                let mut j = i - 1;
                while j > 0 && chars[j - 1].is_ascii_digit() {
                    j -= 1;
                }
                let nd = i - 1 - j;
                if j >= 2 && chars[j - 1] == 'S' && chars[j - 2] == '[' && nd > 0 {
                    let digits: String = chars[j..i - 1].iter().collect();
                    tag = digits.parse().ok();
                    let cut = rest.len().saturating_sub(nd + 3);
                    rest.truncate(cut);
                }
            }
            let start = i;
            let mut k = i + 1;
            let mut text = String::new();
            let mut closed = false;
            while k < chars.len() {
                if chars[k] == '\u{BB}' {
                    closed = true;
                    break;
                }
                text.push(chars[k]);
                k += 1;
            }
            quotes.push(RawQuote {
                tag,
                text,
                start,
                end: k.min(chars.len()),
                closed,
            });
            rest.push(' ');
            i = k + 1;
        } else {
            rest.push(chars[i]);
            i += 1;
        }
    }
    (quotes, rest, false)
}

// ───────────────────────────── vocabulary trie ─────────────────────────────

/// Token byte trie in flat arrays, plus the two precomputed "outside" masks.
pub struct QuoteVocab {
    n: usize,
    data: Vec<u8>,
    offs: Vec<u32>,
    kid_start: Vec<u32>,
    kids: Vec<(u8, u32)>,
    tok_start: Vec<u32>,
    toks: Vec<u32>,
    outside: [Arc<BitSet>; 2],
}

const OPEN: [u8; 2] = [0xC2, 0xAB];
const CLOSE_B: u8 = 0xBB;
const MAX_TOK: usize = 512;

/// 0 = dead, 1 = fine, 2 = ends exactly on the opening marker.
fn outside_class(bytes: &[u8], mut p: bool) -> u8 {
    for (i, &b) in bytes.iter().enumerate() {
        if p {
            if b == OPEN[1] {
                return if i + 1 == bytes.len() { 2 } else { 0 };
            }
            if b == CLOSE_B {
                return 0;
            }
        }
        p = b == OPEN[0];
    }
    1
}

impl QuoteVocab {
    /// Build from raw token bytes; an empty entry marks a special token.
    pub fn from_token_bytes(tokens: &[Vec<u8>]) -> QuoteVocab {
        let n = tokens.len();
        let mut data = Vec::new();
        let mut offs = vec![0u32];
        for t in tokens {
            data.extend_from_slice(t);
            offs.push(data.len() as u32);
        }
        type TNode = (Vec<(u8, u32)>, Vec<u32>);
        let mut nodes: Vec<TNode> = vec![(vec![], vec![])];
        for (id, t) in tokens.iter().enumerate() {
            if t.is_empty() || t.len() > MAX_TOK {
                continue;
            }
            let mut cur = 0usize;
            for &b in t {
                let next = match nodes[cur].0.iter().find(|k| k.0 == b) {
                    Some(k) => k.1 as usize,
                    None => {
                        nodes.push((vec![], vec![]));
                        let ni = nodes.len() - 1;
                        nodes[cur].0.push((b, ni as u32));
                        ni
                    }
                };
                cur = next;
            }
            nodes[cur].1.push(id as u32);
        }
        let mut kid_start = Vec::with_capacity(nodes.len() + 1);
        let mut tok_start = Vec::with_capacity(nodes.len() + 1);
        let (mut kids, mut toks) = (Vec::new(), Vec::new());
        for (k, t) in &nodes {
            kid_start.push(kids.len() as u32);
            tok_start.push(toks.len() as u32);
            kids.extend_from_slice(k);
            toks.extend_from_slice(t);
        }
        kid_start.push(kids.len() as u32);
        tok_start.push(toks.len() as u32);
        let mk = |p: bool| {
            let mut m = BitSet::new(n);
            for (id, t) in tokens.iter().enumerate() {
                if t.is_empty() || outside_class(t, p) != 0 {
                    m.set(id);
                }
            }
            Arc::new(m)
        };
        QuoteVocab {
            n,
            data,
            offs,
            kid_start,
            kids,
            tok_start,
            toks,
            outside: [mk(false), mk(true)],
        }
    }

    pub fn from_trie(t: &TokenTrie) -> QuoteVocab {
        let toks: Vec<Vec<u8>> = (0..t.vocab_size() as u32)
            .map(|i| t.token_bytes(i).to_vec())
            .collect();
        Self::from_token_bytes(&toks)
    }

    pub fn from_tokenizer(t: &Tokenizer) -> QuoteVocab {
        Self::from_trie(&TokenTrie::from_tokenizer(t))
    }

    pub fn vocab_size(&self) -> usize {
        self.n
    }

    fn bytes(&self, id: u32) -> &[u8] {
        &self.data[self.offs[id as usize] as usize..self.offs[id as usize + 1] as usize]
    }

    fn set_tokens(&self, node: u32, out: &mut BitSet) {
        for &t in &self.toks
            [self.tok_start[node as usize] as usize..self.tok_start[node as usize + 1] as usize]
        {
            out.set(t as usize);
        }
    }
}

// ───────────────────────────── the constraint ─────────────────────────────

enum Mode {
    Outside { p: bool },
    Inside { st: QSt, ivs: Vec<Iv> },
    Dead,
}

/// Token-by-token quote lock.  Use like `grammar::Constraint`: read
/// [`mask`](Self::mask), sample, then [`accept`](Self::accept) the token.
pub struct QuoteLock {
    idx: Arc<QuoteIndex>,
    vocab: Arc<QuoteVocab>,
    mode: Mode,
    cur: Arc<BitSet>,
    cache: HashMap<Vec<u32>, Arc<BitSet>>,
    tail: Vec<u8>,
    restrict: Option<usize>,
    citations: Vec<Citation>,
    cache_hits: usize,
    cache_misses: usize,
}

const CACHE_CAP: usize = 256;

impl QuoteLock {
    pub fn new(idx: Arc<QuoteIndex>, vocab: Arc<QuoteVocab>) -> QuoteLock {
        let cur = vocab.outside[0].clone();
        QuoteLock {
            idx,
            vocab,
            mode: Mode::Outside { p: false },
            cur,
            cache: HashMap::new(),
            tail: vec![],
            restrict: None,
            citations: vec![],
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Tokens allowed next.
    pub fn mask(&self) -> &BitSet {
        &self.cur
    }

    pub fn apply(&self, logits: &mut [f32]) {
        apply_mask(logits, &self.cur);
    }

    /// Quotes completed so far, with their sources.
    pub fn citations(&self) -> &[Citation] {
        &self.citations
    }

    pub fn is_inside_quote(&self) -> bool {
        matches!(self.mode, Mode::Inside { .. })
    }

    /// No way forward (should not happen with byte-complete vocabularies).
    pub fn is_dead(&self) -> bool {
        matches!(self.mode, Mode::Dead) || self.cur.count() == 0
    }

    /// True when not inside an open quote.
    pub fn is_complete(&self) -> bool {
        matches!(self.mode, Mode::Outside { .. })
    }

    pub fn is_finished(&self) -> bool {
        self.is_dead()
    }

    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache_hits, self.cache_misses)
    }

    pub fn reset(&mut self) {
        self.mode = Mode::Outside { p: false };
        self.cur = self.vocab.outside[0].clone();
        self.tail.clear();
        self.restrict = None;
        self.citations.clear();
    }

    fn parse_tag(&self) -> Option<usize> {
        let t = &self.tail;
        let t = &t[..t.len().checked_sub(2)?];
        let t = t.strip_suffix(b"]")?;
        let mut k = t.len();
        while k > 0 && t[k - 1].is_ascii_digit() {
            k -= 1;
        }
        if k == t.len() || k < 2 || t[k - 1] != b'S' || t[k - 2] != b'[' {
            return None;
        }
        std::str::from_utf8(&t[k..]).ok()?.parse().ok()
    }

    /// Feed one token.  Returns `true` while outside an open quote.
    pub fn accept(&mut self, token: u32) -> R<bool> {
        if token as usize >= self.vocab.n || !self.cur.get(token as usize) {
            return Err(QuoteError::Violation(token));
        }
        let vocab = self.vocab.clone();
        let idx = self.idx.clone();
        for &b in vocab.bytes(token) {
            match &mut self.mode {
                Mode::Dead => return Err(QuoteError::Dead),
                Mode::Outside { p } => {
                    self.tail.push(b);
                    if self.tail.len() > 32 {
                        self.tail.drain(..16);
                    }
                    if *p && b == OPEN[1] {
                        let tag = self.parse_tag();
                        self.restrict = tag
                            .filter(|t| *t >= 1 && *t <= idx.srcs.len())
                            .map(|t| t - 1);
                        self.mode = Mode::Inside {
                            st: QSt::fresh(),
                            ivs: idx.all_ivs(self.restrict),
                        };
                    } else if *p && b == CLOSE_B {
                        self.mode = Mode::Dead;
                        return Err(QuoteError::Dead);
                    } else {
                        *p = b == OPEN[0];
                    }
                }
                Mode::Inside { st, ivs } => {
                    let mut arena = std::mem::take(ivs);
                    let fr = (0, arena.len());
                    match idx.feed(&mut arena, fr, st, b) {
                        Step::Ok(a, z) => *ivs = arena[a..z].to_vec(),
                        Step::Closed => {
                            let d = st.d;
                            if let Some(c) = idx.citation_from_ivs(&arena, d) {
                                self.citations.push(c);
                            }
                            self.mode = Mode::Outside { p: false };
                            self.tail.clear();
                        }
                        Step::Dead => {
                            self.mode = Mode::Dead;
                            return Err(QuoteError::Dead);
                        }
                    }
                }
            }
        }
        self.refresh();
        Ok(self.is_complete())
    }

    fn refresh(&mut self) {
        self.cur = match &self.mode {
            Mode::Dead => Arc::new(BitSet::new(self.vocab.n)),
            Mode::Outside { p } => self.vocab.outside[*p as usize].clone(),
            Mode::Inside { st, ivs } => {
                let key = self.idx.cache_key(ivs, st);
                if let Some(m) = self.cache.get(&key) {
                    self.cache_hits += 1;
                    m.clone()
                } else {
                    self.cache_misses += 1;
                    let m = Arc::new(self.idx.build_mask(&self.vocab, ivs, st));
                    if self.cache.len() >= CACHE_CAP {
                        self.cache.clear();
                    }
                    self.cache.insert(key, m.clone());
                    m
                }
            }
        };
    }
}

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const S1: &str =
        "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs!";
    const S2: &str =
        "Șoarecele și pisica trăiesc în „casa” veche — ţara noastră e frumoasă, spune bunica.";

    fn rng(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    fn vocab() -> (QuoteVocab, u32, u32) {
        let mut t: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        t[0] = vec![];
        for w in [
            " the", "the", " quick", " brown", " fox", " over", " lazy", " dog", "«", "»", " of",
            "[S2]", " pisica", "ing",
        ] {
            t.push(w.as_bytes().to_vec());
        }
        t.push(vec![]); // eos
        let open = t.iter().position(|x| x == "«".as_bytes()).unwrap() as u32;
        let close = t.iter().position(|x| x == "»".as_bytes()).unwrap() as u32;
        (QuoteVocab::from_token_bytes(&t), open, close)
    }

    fn lock(src: &[&str], min: usize) -> (QuoteLock, u32, u32, Arc<QuoteIndex>) {
        let idx = Arc::new(
            QuoteIndex::new(
                src,
                Options {
                    min_chars: min,
                    ..Options::default()
                },
            )
            .unwrap(),
        );
        let (v, o, c) = vocab();
        (QuoteLock::new(idx.clone(), Arc::new(v)), o, c, idx)
    }

    fn feed_str(l: &mut QuoteLock, s: &str) {
        for b in s.bytes() {
            l.accept(b as u32)
                .unwrap_or_else(|e| panic!("byte {:?} of {s:?}: {e}", b as char));
        }
    }

    #[test]
    fn suffix_array_matches_naive() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        for len in [1usize, 2, 3, 5, 17, 64, 300, 2000] {
            for alpha in [2u64, 4, 26] {
                let mut t: Vec<u8> = (0..len)
                    .map(|_| 0x20 + (rng(&mut seed) % alpha) as u8)
                    .collect();
                t.push(SENTINEL);
                let sa = suffix_array(&t);
                let mut naive: Vec<u32> = (0..t.len() as u32).collect();
                naive.sort_by(|&a, &b| t[a as usize..].cmp(&t[b as usize..]));
                assert_eq!(sa, naive, "len {len} alpha {alpha}");
            }
        }
    }

    #[test]
    fn extract_quotes_is_linear_and_capped() {
        // 50k tagged quotes would be quadratic with the old per-tag rebuild of `rest`.
        let mut big = String::new();
        for _ in 0..50_000 {
            big.push_str("word [S1]\u{AB}x\u{BB} ");
        }
        let t = std::time::Instant::now();
        let (q, rest, over) = extract_quotes(&big, usize::MAX);
        assert!(t.elapsed().as_secs() < 30);
        assert_eq!(q.len(), 50_000);
        assert!(!over && !rest.contains("[S1]"));
        let (q, _, over) = extract_quotes(&big, 10);
        assert_eq!(q.len(), 10);
        assert!(over);
    }

    #[test]
    fn verify_checked_reports_limits() {
        let src = "The quick brown fox jumps over the lazy dog near the river bank today.";
        let o = Options {
            max_quotes: 2,
            ..Options::default()
        };
        let idx = QuoteIndex::new(&[src], o).unwrap();
        let out = "[S1]\u{AB}quick brown fox jumps\u{BB} [S1]\u{AB}lazy dog near the river\u{BB} [S1]\u{AB}the quick brown fox\u{BB}";
        assert!(matches!(
            idx.verify_checked(out),
            Err(QuoteError::TooManyQuotes { limit: 2 })
        ));
        let r = idx.verify(out);
        assert!(r.limit_hit && r.quotes.len() == 2);
        let o = Options {
            max_verify_work: 10,
            ..Options::default()
        };
        let idx = QuoteIndex::new(&[src], o).unwrap();
        assert!(matches!(
            idx.verify_checked(out),
            Err(QuoteError::WorkBudget { .. })
        ));
        let r = idx.verify(out);
        assert!(r.limit_hit && r.verbatim == 0);
        let ok = QuoteIndex::new(&[src], Options::default())
            .unwrap()
            .verify_checked(out)
            .unwrap();
        assert!(!ok.limit_hit);
    }

    #[test]
    fn fuzzy_work_is_bounded() {
        let src = "alpha beta gamma delta epsilon ".repeat(400);
        let o = Options {
            max_verify_work: 200_000,
            ..Options::default()
        };
        let idx = QuoteIndex::new(&[&src], o).unwrap();
        let q = "alpha beta gamma delta epsilon alpha betX gamma delta epsilon ".repeat(20);
        let out = format!("\u{AB}{q}\u{BB} \u{AB}{q}\u{BB} \u{AB}{q}\u{BB}");
        let t = std::time::Instant::now();
        let r = idx.verify(&out);
        assert!(t.elapsed().as_secs() < 5);
        assert!(r.limit_hit);
    }

    #[test]
    fn ignore_diacritics_option() {
        let src = "Copilul merge la școală și mănâncă țuică în fața casei.";
        let strict = QuoteIndex::new(&[src], Options::default()).unwrap();
        let plain = "Copilul merge la scoala si mananca tuica in fata casei";
        let r = strict.verify(&format!("\u{AB}{plain}\u{BB}"));
        assert_ne!(r.verbatim, 1);
        let loose = QuoteIndex::new(
            &[src],
            Options {
                ignore_diacritics: true,
                ..Options::default()
            },
        )
        .unwrap();
        let r = loose.verify(&format!("\u{AB}{plain}\u{BB}"));
        assert_eq!(r.verbatim, 1);
        // cedilla and comma-below are equal in both modes
        let r = strict.verify("\u{AB}merge la şcoală şi mănâncă ţuică\u{BB}");
        assert_eq!(r.verbatim, 1);
    }

    #[test]
    fn normalisation_equivalences() {
        let a = normalize("  Hello,\t“World” — şi ţara  ", true, false).bytes;
        let b = normalize("hello, \"world\" - și țara", true, false).bytes;
        assert_eq!(a, b);
        let c = normalize("Ana", false, false).bytes;
        let d = normalize("ana", false, false).bytes;
        assert_ne!(c, d);
        // decomposed e + acute equals precomposed
        assert_eq!(
            normalize("e\u{301}", true, false).bytes,
            normalize("é", true, false).bytes
        );
    }

    #[test]
    fn find_gives_original_offsets() {
        let idx = QuoteIndex::new(&[S1, S2], Options::default()).unwrap();
        let c = idx.find("QUICK   brown fox").unwrap();
        assert_eq!(c.source_id, 1);
        assert_eq!(c.text, "quick brown fox");
        assert_eq!((c.start, c.end), (4, 19));
        let c2 = idx.find("casa” veche - tara").map(|c| c.source_id);
        assert_eq!(c2, None); // ţ vs t is not an equivalence
        let c3 = idx.find("„casa” veche — țara noastră").unwrap();
        assert_eq!(c3.source_id, 2);
        assert!(c3.text.contains("ţara"));
        assert!(idx.find("lazy cat").is_none());
    }

    #[test]
    fn constraint_only_allows_substrings() {
        let (mut l, _, _, idx) = lock(&[S1, S2], 6);
        feed_str(&mut l, "As the text says: «");
        assert!(l.is_inside_quote());
        // 'z' occurs only in "lazy"/"dozen", "zy" ok, but "zx" is not a substring
        feed_str(&mut l, "lazy");
        assert!(l.mask().get(b' ' as usize));
        assert!(!l.mask().get(b'q' as usize));
        feed_str(&mut l, " dog");
        // after "lazy dog" a close is allowed (boundary: '.' follows)
        assert!(l.mask().get(b'.' as usize));
        for b in "»".bytes() {
            l.accept(b as u32).unwrap();
        }
        assert!(l.is_complete());
        assert_eq!(l.citations().len(), 1);
        assert_eq!(l.citations()[0].text, "lazy dog");
        let rep = idx.verify("As the text says: «lazy dog»");
        assert!(rep.all_verbatim());
    }

    #[test]
    fn min_length_and_word_boundary() {
        // can the closing marker be completed after `text`?
        let can_close = |text: &str| {
            let (mut l, _, _, _) = lock(&[S1], 6);
            feed_str(&mut l, text);
            l.mask().get(0xC2) && {
                l.accept(0xC2).unwrap();
                l.mask().get(0xBB)
            }
        };
        assert!(!can_close("«qui"), "too short and mid-word");
        assert!(!can_close("«quick"), "shorter than min_chars");
        assert!(can_close("«quick brown"));
        assert!(!can_close("«quick brown fo"), "mid-word");
        assert!(can_close("«quick brown fox"));
    }

    #[test]
    fn tag_restricts_source() {
        let (mut l, _, _, _) = lock(&[S1, S2], 4);
        feed_str(&mut l, "[S2]«");
        assert!(!l.mask().get(b'q' as usize), "q only appears in source 1");
        assert!(l.mask().get(b'p' as usize));
        let (mut l2, _, _, _) = lock(&[S1, S2], 4);
        feed_str(&mut l2, "«");
        assert!(l2.mask().get(b'q' as usize));
    }

    #[test]
    fn markers_split_across_tokens_and_specials() {
        let (mut l, open, _, _) = lock(&[S1], 4);
        let eos = l.vocab.n as u32 - 1;
        assert!(l.mask().get(eos as usize));
        l.accept(0xC2).unwrap();
        // token "«x" style (marker + more) must be impossible: no such token, but 0xAB then more is fine
        assert!(l.accept(0xAB).is_ok());
        assert!(l.is_inside_quote());
        assert!(!l.mask().get(eos as usize));
        assert!(!l.mask().get(open as usize), "no nested open");
    }

    #[test]
    fn random_walk_stays_verbatim() {
        let (mut l, open, close, idx) = lock(&[S1, S2], 5);
        let mut seed = 99u64;
        let mut out: Vec<u8> = vec![];
        let mut quotes = 0;
        for _ in 0..4000 {
            let allowed: Vec<u32> = l.mask().iter().collect();
            assert!(
                !allowed.is_empty(),
                "dead end after {:?}",
                String::from_utf8_lossy(&out)
            );
            let r = rng(&mut seed);
            let tok = if !l.is_inside_quote() && r.is_multiple_of(3) && l.mask().get(open as usize)
            {
                open
            } else if l.is_inside_quote() && r.is_multiple_of(5) && l.mask().get(close as usize) {
                close
            } else if l.is_inside_quote() && r.is_multiple_of(7) && l.mask().get(0xC2) {
                0xC2
            } else {
                allowed[(r >> 8) as usize % allowed.len()]
            };
            if l.is_inside_quote() && tok == 0xC2 && !l.mask().get(0xC2) {
                continue;
            }
            let was = l.citations().len();
            l.accept(tok).unwrap();
            out.extend_from_slice(l.vocab.bytes(tok));
            if l.citations().len() > was {
                quotes += 1;
            }
            if quotes >= 30 {
                break;
            }
        }
        assert!(quotes >= 5, "only {quotes} quotes");
        let text = String::from_utf8_lossy(&out);
        let rep = idx.verify(&text);
        let bad: Vec<_> = rep
            .quotes
            .iter()
            .filter(|q| {
                q.closed && !matches!(q.status, QuoteStatus::Verbatim | QuoteStatus::Truncated)
            })
            .map(|q| (&q.text, &q.status))
            .collect();
        assert!(bad.is_empty(), "non-verbatim closed quotes: {bad:?}");
    }

    #[test]
    fn mask_cache_hits() {
        let (mut l, _, _, _) = lock(&[S1], 4);
        feed_str(&mut l, "«the");
        let (_, m0) = l.cache_stats();
        l.reset();
        feed_str(&mut l, "«the");
        let (h, m1) = l.cache_stats();
        assert_eq!(m0, m1);
        assert!(h > 0);
    }

    #[test]
    fn verify_classifies() {
        let idx = QuoteIndex::new(
            &[
                S1,
                "Totally different source about astronomy and the moons of Jupiter orbit.",
            ],
            Options::default(),
        )
        .unwrap();
        let r = idx.verify("«quick brown fox jumps» ok");
        assert_eq!(r.quotes[0].status, QuoteStatus::Verbatim);
        assert_eq!(r.quotes[0].citation.as_ref().unwrap().source_id, 1);

        let r = idx.verify("[S2]«quick brown fox jumps»");
        assert_eq!(r.quotes[0].status, QuoteStatus::WrongSource { found_in: 1 });

        let r = idx.verify("«quick brown fax jumps over»");
        assert_eq!(
            r.quotes[0].status,
            QuoteStatus::Altered,
            "{:?}",
            r.quotes[0]
        );
        assert!(r.quotes[0].similarity > 0.8);

        let r = idx.verify("«uick brown fo»");
        assert_eq!(r.quotes[0].status, QuoteStatus::Truncated);

        let r = idx.verify("«the lazy dog. the moons of Jupiter orbit»");
        assert!(
            matches!(
                r.quotes[0].status,
                QuoteStatus::Merged | QuoteStatus::Altered
            ),
            "{:?}",
            r.quotes[0].status
        );
        assert_ne!(r.quotes[0].status, QuoteStatus::Verbatim);

        let r = idx.verify("«colourless green ideas sleep furiously»");
        assert_eq!(r.quotes[0].status, QuoteStatus::Unsupported);
        assert_eq!(r.unsupported, 1);

        let r = idx.verify("«unclosed quote quick brown");
        assert!(!r.quotes[0].closed);
    }

    #[test]
    fn support_scores_order() {
        let idx = QuoteIndex::new(&[S1], Options::default()).unwrap();
        let hi = idx.support("the quick brown fox jumps over the lazy dog");
        let mid = idx.support("the quick brown cat jumps over the lazy dog");
        let lo = idx.support("quantum chromodynamics governs gluons");
        assert!(hi > mid && mid > lo, "{hi} {mid} {lo}");
        let r = idx.verify("The fox jumps over the lazy dog. Gluons bind quarks together tightly.");
        assert_eq!(r.sentences.len(), 2);
        assert!(r.sentences[0].support > r.sentences[1].support);
    }

    #[test]
    fn limits_and_hostile_input() {
        assert_eq!(
            QuoteIndex::new(&[], Options::default()).err(),
            Some(QuoteError::NoSources)
        );
        assert_eq!(
            QuoteIndex::new(&["ok text", "   \n\t "], Options::default()).err(),
            Some(QuoteError::EmptySource(1))
        );
        let o = Options {
            max_source_bytes: 10,
            ..Options::default()
        };
        assert!(matches!(
            QuoteIndex::new(&["0123456789ab"], o),
            Err(QuoteError::SourceTooLarge { .. })
        ));
        // markers inside sources are ordinary quotes
        let idx = QuoteIndex::new(&["he said «hello there» to me"], Options::default()).unwrap();
        assert!(idx.find("said \"hello there\"").is_some());
        // nested / stray markers in output are bounded
        let r = idx.verify("«a «b «c » d » »»» «");
        assert!(r.quotes.len() <= 3);
        let _ = idx.verify(&"«".repeat(10_000));
        let _ = idx.verify(&"»«".repeat(10_000));
    }

    #[test]
    fn index_ten_megabytes_is_fast_enough() {
        let mut s = String::new();
        let mut seed = 7u64;
        while s.len() < 2_000_000 {
            let w = 2 + rng(&mut seed) % 8;
            for _ in 0..w {
                s.push((b'a' + (rng(&mut seed) % 26) as u8) as char);
            }
            s.push(' ');
        }
        let t = std::time::Instant::now();
        let idx = QuoteIndex::new(&[&s], Options::default()).unwrap();
        assert!(t.elapsed().as_secs() < 20, "{:?}", t.elapsed());
        assert!(idx.find(&s[1000..1060]).is_some());
    }
}
