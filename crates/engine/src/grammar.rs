//! Structured-output decoding: grammar-constrained generation as llama.cpp, mistral.rs and
//! Ollama offer it, written clean-room from the public GBNF / JSON-Schema / regex descriptions.
//!
//! Pieces, bottom up:
//! * [`Node`] – one syntax tree shared by every front end ([`Grammar::from_gbnf`],
//!   [`Grammar::from_regex`], [`Grammar::from_json_schema`], [`Grammar::json_object`]).
//! * [`Grammar`] – the compiled form: rules of alternatives of elements (a character set or a
//!   rule reference). Repetitions are expanded into helper rules under a hard size cap, so a
//!   hostile `{1000000}` is an error, not an out-of-memory.
//! * [`GrammarState`] – an incremental pushdown matcher over Unicode code points. It keeps a set
//!   of stacks (so ambiguous grammars are matched correctly), stacks are persistent linked lists
//!   (so `clone` is cheap and rollback is free), tail calls are eliminated (right recursion runs
//!   in constant stack) and growth is bounded: too many live stacks is an error.
//!   Left recursion that would make expansion unbounded is rejected at compile time.
//! * [`TokenTrie`] + [`GrammarState::allowed_tokens`] – the vocabulary as a byte trie; one
//!   depth-first walk with pruning gives the set of tokens that keep the output inside the
//!   grammar, including tokens that end in the middle of a UTF-8 sequence.
//! * [`Constraint`] – what a sampler calls: `mask()` before sampling, `accept(token)` after.
//!   End-of-sequence is allowed only when the grammar is complete. Masks are cached by state.
//!
//! Limits (documented, not hidden): JSON-Schema numeric bounds (`minimum`, `maximum`,
//! `multipleOf`) are not enforced except that a non-negative `minimum` on an integer removes
//! the sign; `patternProperties`, `if/then/else`, `not` and remote `$ref`s are unsupported; the
//! regex front end has no look-around, back-references or `\p{..}`; inside JSON strings a
//! pattern's own `"`/`\`/control characters become escapes but sets simply exclude them.
#![allow(clippy::collapsible_if)] // nested ifs read better than let-chains here

use crate::tokenizer::{Kind, Tokenizer};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Total grammar size (elements plus rules) after expansion.
pub const MAX_ELEMS: usize = 400_000;
const MAX_SRC: usize = 1 << 20;
const MAX_NEST: usize = 128;
const MAX_STACKS: usize = 8192;
const MAX_WORK: usize = 2_000_000;

// ------------------------------------------------------------------------- errors

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarError {
    Syntax {
        pos: usize,
        msg: String,
    },
    Undefined(String),
    NoRoot(String),
    LeftRecursion(String),
    Unproductive(String),
    TooLarge(&'static str),
    /// Too many simultaneous parse stacks or too much work for one step.
    StateOverflow,
    /// Computing one token mask exceeded its operation or wall-clock budget.
    MaskBudget,
    Unsupported(String),
    Schema(String),
    Regex(String),
}
impl std::fmt::Display for GrammarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrammarError::Syntax { pos, msg } => write!(f, "grammar syntax error at {pos}: {msg}"),
            GrammarError::Undefined(n) => write!(f, "rule `{n}` is used but never defined"),
            GrammarError::NoRoot(n) => write!(f, "the grammar has no `{n}` rule"),
            GrammarError::LeftRecursion(n) => {
                write!(f, "rule `{n}` is left-recursive, which is not supported")
            }
            GrammarError::Unproductive(n) => {
                write!(f, "rule `{n}` can never produce a finite string")
            }
            GrammarError::TooLarge(w) => write!(f, "the grammar is too large ({w})"),
            GrammarError::StateOverflow => write!(
                f,
                "the grammar is too ambiguous: too many live parse states"
            ),
            GrammarError::MaskBudget => write!(
                f,
                "computing the token mask exceeded its work or time budget"
            ),
            GrammarError::Unsupported(s) => write!(f, "unsupported: {s}"),
            GrammarError::Schema(s) => write!(f, "JSON schema: {s}"),
            GrammarError::Regex(s) => write!(f, "regex: {s}"),
        }
    }
}
impl std::error::Error for GrammarError {}
type R<T> = Result<T, GrammarError>;

// ------------------------------------------------------------------------- bit set

/// A fixed-size set of token ids.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BitSet {
    w: Vec<u64>,
    n: usize,
}
impl BitSet {
    pub fn new(n: usize) -> BitSet {
        BitSet {
            w: vec![0; n.div_ceil(64)],
            n,
        }
    }
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.w.iter().all(|&x| x == 0)
    }
    pub fn get(&self, i: usize) -> bool {
        i < self.n && (self.w[i >> 6] >> (i & 63)) & 1 == 1
    }
    pub fn set(&mut self, i: usize) {
        if i < self.n {
            self.w[i >> 6] |= 1 << (i & 63);
        }
    }
    pub fn clear(&mut self, i: usize) {
        if i < self.n {
            self.w[i >> 6] &= !(1 << (i & 63));
        }
    }
    pub fn count(&self) -> usize {
        self.w.iter().map(|x| x.count_ones() as usize).sum()
    }
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.w.iter().enumerate().flat_map(|(wi, &word)| {
            let mut x = word;
            std::iter::from_fn(move || {
                if x == 0 {
                    return None;
                }
                let b = x.trailing_zeros();
                x &= x - 1;
                Some((wi as u32) * 64 + b)
            })
        })
    }
}

/// Forbid every token outside `mask` by sending its logit to −∞.
pub fn apply_mask(logits: &mut [f32], mask: &BitSet) {
    for (i, l) in logits.iter_mut().enumerate() {
        if !mask.get(i) {
            *l = f32::NEG_INFINITY;
        }
    }
}

// ------------------------------------------------------------------------- character sets

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CharSet {
    neg: bool,
    ranges: Vec<(u32, u32)>,
    ascii: u128,
}
impl CharSet {
    pub fn new(neg: bool, mut ranges: Vec<(u32, u32)>) -> CharSet {
        for r in &mut ranges {
            r.1 = r.1.min(0x10FFFF);
        }
        ranges.retain(|r| r.0 <= r.1);
        ranges.sort_unstable();
        let mut m: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
        for r in ranges {
            if let Some(l) = m.last_mut() {
                if r.0 <= l.1.saturating_add(1) {
                    l.1 = l.1.max(r.1);
                    continue;
                }
            }
            m.push(r);
        }
        let mut ascii = 0u128;
        for c in 0..128u32 {
            let inside = m.iter().any(|r| r.0 <= c && c <= r.1);
            if inside != neg {
                ascii |= 1u128 << c;
            }
        }
        CharSet {
            neg,
            ranges: m,
            ascii,
        }
    }
    pub fn single(c: char) -> CharSet {
        CharSet::new(false, vec![(c as u32, c as u32)])
    }
    pub fn any() -> CharSet {
        CharSet::new(true, vec![])
    }
    fn in_ranges(&self, c: u32) -> bool {
        self.ranges
            .binary_search_by(|r| {
                if r.1 < c {
                    std::cmp::Ordering::Less
                } else if r.0 > c {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
    }
    #[inline]
    pub fn matches(&self, c: u32) -> bool {
        if c < 128 {
            (self.ascii >> c) & 1 == 1
        } else {
            self.in_ranges(c) != self.neg
        }
    }
    /// Does any code point in `lo..=hi` match?
    pub fn intersects(&self, lo: u32, hi: u32) -> bool {
        if self.neg {
            !self.ranges.iter().any(|r| r.0 <= lo && hi <= r.1)
        } else {
            self.ranges.iter().any(|r| r.0 <= hi && lo <= r.1)
        }
    }
    /// A few characters that match (for tests, demos and hints).
    pub fn representatives(&self, out: &mut Vec<char>, limit: usize) {
        if self.neg {
            for c in (' '..='~').chain(['é', '漢', '😀']) {
                if out.len() >= limit {
                    return;
                }
                if self.matches(c as u32) {
                    out.push(c);
                }
            }
        } else {
            for r in &self.ranges {
                for c in [r.0, r.0 + (r.1 - r.0) / 2, r.1] {
                    if out.len() >= limit {
                        return;
                    }
                    if let Some(ch) = char::from_u32(c) {
                        out.push(ch);
                    }
                }
            }
        }
    }
}

fn subtract(ranges: &[(u32, u32)], cut: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut cur = ranges.to_vec();
    for &(cl, ch) in cut {
        let mut next = Vec::with_capacity(cur.len() + 1);
        for (a, b) in cur {
            if b < cl || a > ch {
                next.push((a, b));
                continue;
            }
            if a < cl {
                next.push((a, cl - 1));
            }
            if b > ch {
                next.push((ch + 1, b));
            }
        }
        cur = next;
    }
    cur
}
fn complement(ranges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    subtract(&[(0, 0x10FFFF)], ranges)
}

// ------------------------------------------------------------------------- syntax tree

/// The syntax tree every front end produces.
#[derive(Clone, Debug)]
pub enum Node {
    Empty,
    Lit(String),
    Set(CharSet),
    Ref(String),
    Seq(Vec<Node>),
    Alt(Vec<Node>),
    /// `min` copies, then up to `max − min` optional ones (`None` = unbounded).
    Repeat(Box<Node>, u32, Option<u32>),
}
fn lit(s: &str) -> Node {
    Node::Lit(s.to_string())
}
fn rref(s: &str) -> Node {
    Node::Ref(s.to_string())
}
fn seq(v: Vec<Node>) -> Node {
    Node::Seq(v)
}
fn alt(v: Vec<Node>) -> Node {
    Node::Alt(v)
}
fn opt(n: Node) -> Node {
    Node::Repeat(Box::new(n), 0, Some(1))
}
fn star(n: Node) -> Node {
    Node::Repeat(Box::new(n), 0, None)
}
fn rep(n: Node, a: u32, b: Option<u32>) -> Node {
    Node::Repeat(Box::new(n), a, b)
}
fn setr(r: &[(char, char)]) -> Node {
    Node::Set(CharSet::new(
        false,
        r.iter().map(|&(a, b)| (a as u32, b as u32)).collect(),
    ))
}

// ------------------------------------------------------------------------- regex front end

fn json_escape_char(c: char) -> String {
    match c {
        '"' => "\\\"".into(),
        '\\' => "\\\\".into(),
        '\n' => "\\n".into(),
        '\r' => "\\r".into(),
        '\t' => "\\t".into(),
        c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32),
        c => c.to_string(),
    }
}
const JSON_BAD: [(u32, u32); 3] = [(0, 0x1f), (0x22, 0x22), (0x5c, 0x5c)];

enum ClassItem {
    Ch(u32),
    Ranges(Vec<(u32, u32)>),
}
const DIGIT: [(u32, u32); 1] = [(0x30, 0x39)];
const WORD: [(u32, u32); 4] = [(0x30, 0x39), (0x41, 0x5a), (0x5f, 0x5f), (0x61, 0x7a)];
const SPACE: [(u32, u32); 2] = [(9, 13), (32, 32)];

struct Rx {
    c: Vec<char>,
    i: usize,
    json: bool,
    depth: usize,
}
impl Rx {
    fn err<T>(&self, m: &str) -> R<T> {
        Err(GrammarError::Regex(format!("{m} (at {})", self.i)))
    }
    fn peek(&self) -> Option<char> {
        self.c.get(self.i).copied()
    }
    fn eat(&mut self, ch: char) -> bool {
        if self.peek() == Some(ch) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn alt(&mut self) -> R<Node> {
        let mut alts = vec![self.seq()?];
        while self.eat('|') {
            alts.push(self.seq()?);
        }
        Ok(if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            Node::Alt(alts)
        })
    }
    fn seq(&mut self) -> R<Node> {
        let mut items = vec![];
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom()?;
            items.push(self.quant(atom)?);
        }
        Ok(match items.len() {
            0 => Node::Empty,
            1 => items.pop().unwrap(),
            _ => Node::Seq(items),
        })
    }
    fn number(&mut self) -> Option<u32> {
        let s = self.i;
        let mut v: u64 = 0;
        while let Some(d) = self.peek().and_then(|c| c.to_digit(10)) {
            v = v * 10 + d as u64;
            if v > 100_000_000 {
                return None;
            }
            self.i += 1;
        }
        if self.i == s { None } else { Some(v as u32) }
    }
    fn quant(&mut self, mut atom: Node) -> R<Node> {
        loop {
            let (min, max) = match self.peek() {
                Some('*') => {
                    self.i += 1;
                    (0, None)
                }
                Some('+') => {
                    self.i += 1;
                    (1, None)
                }
                Some('?') => {
                    self.i += 1;
                    (0, Some(1))
                }
                Some('{') => {
                    let save = self.i;
                    self.i += 1;
                    let a = self.number();
                    let r = if self.eat(',') {
                        let b = self.number();
                        if a.is_none() && b.is_none() {
                            None
                        } else {
                            Some((a.unwrap_or(0), b))
                        }
                    } else {
                        a.map(|a| (a, Some(a)))
                    };
                    if let (Some(rv), true) = (r, self.eat('}')) {
                        rv
                    } else {
                        self.i = save;
                        return Ok(atom);
                    }
                }
                _ => return Ok(atom),
            };
            if let Some(m) = max {
                if m < min {
                    return self.err("repetition upper bound below lower bound");
                }
            }
            if !self.eat('?') {
                self.eat('+');
            }
            atom = Node::Repeat(Box::new(atom), min, max);
        }
    }
    fn hex(&mut self, n: usize) -> R<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let Some(d) = self.peek().and_then(|c| c.to_digit(16)) else {
                return self.err("bad hex escape");
            };
            v = v * 16 + d;
            self.i += 1;
        }
        Ok(v)
    }
    fn escape(&mut self) -> R<ClassItem> {
        let Some(c) = self.peek() else {
            return self.err("dangling backslash");
        };
        self.i += 1;
        Ok(match c {
            'd' => ClassItem::Ranges(DIGIT.to_vec()),
            'D' => ClassItem::Ranges(complement(&DIGIT)),
            'w' => ClassItem::Ranges(WORD.to_vec()),
            'W' => ClassItem::Ranges(complement(&WORD)),
            's' => ClassItem::Ranges(SPACE.to_vec()),
            'S' => ClassItem::Ranges(complement(&SPACE)),
            'n' => ClassItem::Ch(10),
            'r' => ClassItem::Ch(13),
            't' => ClassItem::Ch(9),
            'f' => ClassItem::Ch(12),
            'v' => ClassItem::Ch(11),
            '0' => ClassItem::Ch(0),
            'x' => ClassItem::Ch(self.hex(2)?),
            'u' => {
                if self.eat('{') {
                    let mut v = 0u32;
                    let mut n = 0;
                    while let Some(d) = self.peek().and_then(|c| c.to_digit(16)) {
                        v = v.saturating_mul(16).saturating_add(d);
                        self.i += 1;
                        n += 1;
                        if n > 8 {
                            break;
                        }
                    }
                    if !self.eat('}') || n == 0 || v > 0x10FFFF {
                        return self.err("bad \\u{..} escape");
                    }
                    ClassItem::Ch(v)
                } else {
                    ClassItem::Ch(self.hex(4)?)
                }
            }
            'b' | 'B' | 'A' | 'Z' | 'z' | 'G' => {
                return self.err("anchors and word boundaries inside a pattern are not supported");
            }
            '1'..='9' => return self.err("back-references are not supported"),
            'p' | 'P' | 'k' => {
                return self.err("\\p{..} and named back-references are not supported");
            }
            c => ClassItem::Ch(c as u32),
        })
    }
    fn lit_char(&self, c: u32) -> R<Node> {
        let Some(ch) = char::from_u32(c) else {
            return self.err("not a valid character");
        };
        Ok(Node::Lit(if self.json {
            json_escape_char(ch)
        } else {
            ch.to_string()
        }))
    }
    fn set_node(&self, neg: bool, ranges: Vec<(u32, u32)>) -> Node {
        if self.json {
            if neg {
                let mut r = ranges;
                r.extend(JSON_BAD);
                Node::Set(CharSet::new(true, r))
            } else {
                Node::Set(CharSet::new(false, subtract(&ranges, &JSON_BAD)))
            }
        } else {
            Node::Set(CharSet::new(neg, ranges))
        }
    }
    fn atom(&mut self) -> R<Node> {
        let Some(c) = self.peek() else {
            return self.err("unexpected end");
        };
        self.i += 1;
        match c {
            '(' => {
                self.depth += 1;
                if self.depth > MAX_NEST {
                    return Err(GrammarError::TooLarge("regex nesting"));
                }
                if self.eat('?') {
                    if self.eat(':') {
                    } else if self.peek() == Some('<')
                        && !matches!(self.c.get(self.i + 1), Some('=') | Some('!'))
                    {
                        while let Some(ch) = self.peek() {
                            self.i += 1;
                            if ch == '>' {
                                break;
                            }
                        }
                    } else {
                        return self.err("look-around is not supported");
                    }
                }
                let n = self.alt()?;
                if !self.eat(')') {
                    return self.err("missing )");
                }
                self.depth -= 1;
                Ok(n)
            }
            '[' => self.class(),
            '.' => Ok(self.set_node(true, vec![(10, 10), (13, 13)])),
            '^' | '$' => Ok(Node::Empty),
            '\\' => match self.escape()? {
                ClassItem::Ch(c) => self.lit_char(c),
                ClassItem::Ranges(r) => Ok(self.set_node(false, r)),
            },
            '*' | '+' | '?' => self.err("nothing to repeat"),
            c => self.lit_char(c as u32),
        }
    }
    fn class(&mut self) -> R<Node> {
        let neg = self.eat('^');
        let mut ranges: Vec<(u32, u32)> = vec![];
        let mut first = true;
        loop {
            let Some(c) = self.peek() else {
                return self.err("missing ]");
            };
            if c == ']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            self.i += 1;
            let item = if c == '\\' {
                self.escape()?
            } else {
                ClassItem::Ch(c as u32)
            };
            match item {
                ClassItem::Ranges(r) => ranges.extend(r),
                ClassItem::Ch(lo) => {
                    if self.peek() == Some('-') && self.c.get(self.i + 1).is_some_and(|&n| n != ']')
                    {
                        self.i += 1;
                        let c2 = self.peek().unwrap();
                        self.i += 1;
                        let hi = if c2 == '\\' {
                            match self.escape()? {
                                ClassItem::Ch(h) => h,
                                ClassItem::Ranges(_) => return self.err("bad range"),
                            }
                        } else {
                            c2 as u32
                        };
                        if hi < lo {
                            return self.err("range out of order");
                        }
                        ranges.push((lo, hi));
                    } else {
                        ranges.push((lo, lo));
                    }
                }
            }
        }
        Ok(self.set_node(neg, ranges))
    }
}

/// Regex → syntax tree. Returns the tree and whether the pattern was anchored at each end
/// (`^`, `$`); unanchored ends are the caller's to widen when "search" semantics are wanted.
pub fn regex_node(pattern: &str, json_string: bool) -> R<(Node, bool, bool)> {
    if pattern.len() > MAX_SRC {
        return Err(GrammarError::TooLarge("pattern"));
    }
    let mut chars: Vec<char> = pattern.chars().collect();
    let a = chars.first() == Some(&'^');
    if a {
        chars.remove(0);
    }
    let mut e = false;
    if chars.last() == Some(&'$') {
        let bs = chars
            .iter()
            .rev()
            .skip(1)
            .take_while(|&&c| c == '\\')
            .count();
        if bs % 2 == 0 {
            chars.pop();
            e = true;
        }
    }
    let mut rx = Rx {
        c: chars,
        i: 0,
        json: json_string,
        depth: 0,
    };
    let n = rx.alt()?;
    if rx.i < rx.c.len() {
        return rx.err("unbalanced )");
    }
    Ok((n, a, e))
}

// ------------------------------------------------------------------------- compiled grammar

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Elem {
    Set(u32),
    Rule(u32),
}

/// A compiled grammar. Cheap to share: wrap it in an `Arc`.
#[derive(Debug)]
pub struct Grammar {
    rules: Vec<Vec<Vec<Elem>>>,
    sets: Vec<CharSet>,
    names: Vec<String>,
    root: u32,
}

struct Lower {
    ids: HashMap<String, u32>,
    rules: Vec<Vec<Vec<Elem>>>,
    names: Vec<String>,
    sets: Vec<CharSet>,
    set_ix: HashMap<CharSet, u32>,
    size: usize,
}
impl Lower {
    fn grow(&mut self, n: usize) -> R<()> {
        self.size = self.size.saturating_add(n);
        if self.size > MAX_ELEMS {
            Err(GrammarError::TooLarge(
                "more than 400000 elements after expanding repetitions",
            ))
        } else {
            Ok(())
        }
    }
    fn new_rule(&mut self, alts: Vec<Vec<Elem>>) -> R<u32> {
        self.grow(1)?;
        let id = self.rules.len() as u32;
        self.rules.push(alts);
        self.names.push(format!("${id}"));
        Ok(id)
    }
    fn set(&mut self, cs: CharSet) -> u32 {
        if let Some(&i) = self.set_ix.get(&cs) {
            return i;
        }
        let i = self.sets.len() as u32;
        self.sets.push(cs.clone());
        self.set_ix.insert(cs, i);
        i
    }
    fn alts(&mut self, n: &Node, depth: usize) -> R<Vec<Vec<Elem>>> {
        if depth > 256 {
            return Err(GrammarError::TooLarge("nesting"));
        }
        Ok(match n {
            Node::Empty => vec![vec![]],
            Node::Lit(s) => {
                self.grow(s.chars().count())?;
                vec![
                    s.chars()
                        .map(|c| Elem::Set(self.set(CharSet::single(c))))
                        .collect(),
                ]
            }
            Node::Set(cs) => {
                self.grow(1)?;
                vec![vec![Elem::Set(self.set(cs.clone()))]]
            }
            Node::Ref(name) => {
                let Some(&id) = self.ids.get(name) else {
                    return Err(GrammarError::Undefined(name.clone()));
                };
                self.grow(1)?;
                vec![vec![Elem::Rule(id)]]
            }
            Node::Seq(items) => {
                let mut out = vec![];
                for it in items {
                    let mut a = self.alts(it, depth + 1)?;
                    if a.len() == 1 {
                        out.append(&mut a[0]);
                    } else {
                        let r = self.new_rule(a)?;
                        self.grow(1)?;
                        out.push(Elem::Rule(r));
                    }
                }
                vec![out]
            }
            Node::Alt(items) => {
                let mut out = vec![];
                for it in items {
                    out.append(&mut self.alts(it, depth + 1)?);
                }
                out
            }
            Node::Repeat(inner, min, max) => {
                if let Some(m) = max {
                    if m < min {
                        return Err(GrammarError::Syntax {
                            pos: 0,
                            msg: "repetition upper bound below lower bound".into(),
                        });
                    }
                }
                let copies = *min as usize + max.map_or(2, |m| (m - min) as usize * 3);
                if copies > MAX_ELEMS || self.size.saturating_add(copies) > MAX_ELEMS {
                    return Err(GrammarError::TooLarge("repetition count"));
                }
                let mut a = self.alts(inner, depth + 1)?;
                let e = if a.len() == 1 && a[0].len() == 1 {
                    a[0][0]
                } else {
                    let r = self.new_rule(std::mem::take(&mut a))?;
                    Elem::Rule(r)
                };
                self.grow(copies)?;
                let mut out: Vec<Elem> = vec![e; *min as usize];
                match max {
                    None => {
                        let s = self.rules.len() as u32;
                        let r = self.new_rule(vec![vec![e, Elem::Rule(s)], vec![]])?;
                        out.push(Elem::Rule(r));
                    }
                    Some(mx) => {
                        let mut tail: Option<Elem> = None;
                        for _ in 0..(mx - min) {
                            let mut alt0 = vec![e];
                            if let Some(t) = tail {
                                alt0.push(t);
                            }
                            let r = self.new_rule(vec![alt0, vec![]])?;
                            tail = Some(Elem::Rule(r));
                        }
                        if let Some(t) = tail {
                            out.push(t);
                        }
                    }
                }
                vec![out]
            }
        })
    }
}

/// Least fixed point "some alternative has all elements good", in linear time.
fn least_fixpoint(rules: &[Vec<Vec<Elem>>], set_ok: bool) -> Vec<bool> {
    let n = rules.len();
    let mut holds = vec![false; n];
    let mut pending: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut watchers: Vec<Vec<(u32, u32)>> = vec![vec![]; n];
    let mut queue: Vec<u32> = vec![];
    for (ri, alts) in rules.iter().enumerate() {
        let mut p = Vec::with_capacity(alts.len());
        for (ai, alt) in alts.iter().enumerate() {
            let mut cnt = 0u32;
            let mut dead = false;
            for e in alt {
                match e {
                    Elem::Set(_) => {
                        if !set_ok {
                            dead = true
                        }
                    }
                    Elem::Rule(r) => {
                        cnt += 1;
                        watchers[*r as usize].push((ri as u32, ai as u32));
                    }
                }
            }
            if dead {
                cnt = u32::MAX / 2;
            }
            if cnt == 0 && !holds[ri] {
                holds[ri] = true;
                queue.push(ri as u32);
            }
            p.push(cnt);
        }
        pending.push(p);
    }
    while let Some(r) = queue.pop() {
        let ws = std::mem::take(&mut watchers[r as usize]);
        for (ri, ai) in ws {
            let c = &mut pending[ri as usize][ai as usize];
            if *c >= u32::MAX / 4 {
                continue;
            }
            *c -= 1;
            if *c == 0 && !holds[ri as usize] {
                holds[ri as usize] = true;
                queue.push(ri);
            }
        }
    }
    holds
}

impl Grammar {
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
    pub fn element_count(&self) -> usize {
        self.rules.iter().flat_map(|a| a.iter()).map(Vec::len).sum()
    }
    pub fn rule_name(&self, i: usize) -> &str {
        &self.names[i]
    }

    /// Compile named rules; `root` names the start rule.
    pub fn from_nodes(defs: Vec<(String, Node)>, root: &str) -> R<Grammar> {
        let mut l = Lower {
            ids: HashMap::new(),
            rules: vec![],
            names: vec![],
            sets: vec![],
            set_ix: HashMap::new(),
            size: 0,
        };
        for (name, _) in &defs {
            if l.ids.insert(name.clone(), l.names.len() as u32).is_some() {
                return Err(GrammarError::Syntax {
                    pos: 0,
                    msg: format!("rule `{name}` is defined twice"),
                });
            }
            l.names.push(name.clone());
            l.rules.push(vec![]);
        }
        let Some(&root_id) = l.ids.get(root) else {
            return Err(GrammarError::NoRoot(root.to_string()));
        };
        for (i, (_, node)) in defs.iter().enumerate() {
            let a = l.alts(node, 0)?;
            l.rules[i] = a;
        }
        let g = Grammar {
            rules: l.rules,
            sets: l.sets,
            names: l.names,
            root: root_id,
        };
        g.analyse()?;
        Ok(g)
    }

    fn analyse(&self) -> R<()> {
        let productive = least_fixpoint(&self.rules, true);
        if !productive[self.root as usize] {
            return Err(GrammarError::Unproductive(
                self.names[self.root as usize].clone(),
            ));
        }
        let nullable = least_fixpoint(&self.rules, false);
        let n = self.rules.len();
        let mut adj: Vec<Vec<(u32, bool)>> = vec![vec![]; n];
        for (i, alts) in self.rules.iter().enumerate() {
            for alt in alts {
                for (k, e) in alt.iter().enumerate() {
                    match e {
                        Elem::Set(_) => break,
                        Elem::Rule(r) => {
                            adj[i].push((*r, k + 1 == alt.len()));
                            if !nullable[*r as usize] {
                                break;
                            }
                        }
                    }
                }
            }
        }
        // iterative Tarjan
        let (mut index, mut low, mut comp) = (vec![u32::MAX; n], vec![0u32; n], vec![u32::MAX; n]);
        let (mut on, mut st, mut counter, mut ncomp) =
            (vec![false; n], Vec::<u32>::new(), 0u32, 0u32);
        for s in 0..n {
            if index[s] != u32::MAX {
                continue;
            }
            let mut call: Vec<(u32, usize)> = vec![(s as u32, 0)];
            index[s] = counter;
            low[s] = counter;
            counter += 1;
            st.push(s as u32);
            on[s] = true;
            while let Some(&mut (v, ref mut ei)) = call.last_mut() {
                let v = v as usize;
                if *ei < adj[v].len() {
                    let w = adj[v][*ei].0 as usize;
                    *ei += 1;
                    if index[w] == u32::MAX {
                        index[w] = counter;
                        low[w] = counter;
                        counter += 1;
                        st.push(w as u32);
                        on[w] = true;
                        call.push((w as u32, 0));
                    } else if on[w] {
                        low[v] = low[v].min(index[w]);
                    }
                } else {
                    call.pop();
                    if let Some(&(p, _)) = call.last() {
                        low[p as usize] = low[p as usize].min(low[v]);
                    }
                    if low[v] == index[v] {
                        while let Some(w) = st.pop() {
                            on[w as usize] = false;
                            comp[w as usize] = ncomp;
                            if w as usize == v {
                                break;
                            }
                        }
                        ncomp += 1;
                    }
                }
            }
        }
        for v in 0..n {
            for &(w, tail) in &adj[v] {
                if !tail && comp[v] == comp[w as usize] {
                    return Err(GrammarError::LeftRecursion(self.names[v].clone()));
                }
            }
        }
        Ok(())
    }

    /// Does the whole of `text` match the grammar?
    pub fn matches(self: &Arc<Self>, text: &str) -> R<bool> {
        let mut s = GrammarState::new(self.clone())?;
        Ok(s.advance_bytes(text.as_bytes())? && s.is_complete())
    }

    pub fn from_regex(pattern: &str) -> R<Grammar> {
        let (n, _, _) = regex_node(pattern, false)?;
        Grammar::from_nodes(vec![("root".into(), n)], "root")
    }

    pub fn from_gbnf(src: &str, root: &str) -> R<Grammar> {
        Grammar::from_nodes(parse_gbnf(src)?, root)
    }

    pub fn from_json_schema(schema: &Value, opts: &SchemaOptions) -> R<Grammar> {
        schema_to_grammar(schema, opts)
    }
    pub fn from_json_schema_str(schema: &str, opts: &SchemaOptions) -> R<Grammar> {
        let v: Value =
            serde_json::from_str(schema).map_err(|e| GrammarError::Schema(e.to_string()))?;
        schema_to_grammar(&v, opts)
    }
    /// Any JSON object (`response_format: {"type": "json_object"}`).
    pub fn json_object(opts: &SchemaOptions) -> R<Grammar> {
        let mut c = Conv::new(&Value::Null, opts);
        c.prims();
        c.add("root".into(), rref("object-any"));
        Grammar::from_nodes(c.defs, "root")
    }
}

// ------------------------------------------------------------------------- GBNF front end

struct Gb {
    c: Vec<char>,
    i: usize,
    depth: usize,
}
impl Gb {
    fn err<T>(&self, m: &str) -> R<T> {
        Err(GrammarError::Syntax {
            pos: self.i,
            msg: m.to_string(),
        })
    }
    fn peek(&self) -> Option<char> {
        self.c.get(self.i).copied()
    }
    /// Skip spaces, tabs and comments; newlines too when `nl`.
    fn ws(&mut self, nl: bool) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\r' || (nl && c == '\n') {
                self.i += 1;
            } else if c == '#' {
                while let Some(c) = self.peek() {
                    if c == '\n' {
                        break;
                    }
                    self.i += 1;
                }
            } else {
                break;
            }
        }
    }
    fn ident(&mut self) -> Option<String> {
        let s = self.i;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            self.i += 1;
        }
        if self.i == s {
            None
        } else {
            Some(self.c[s..self.i].iter().collect())
        }
    }
    fn alts(&mut self, nested: bool) -> R<Node> {
        let mut alts = vec![self.seq(nested)?];
        loop {
            // a `|` on the next line continues the rule
            let save = self.i;
            self.ws(true);
            if self.peek() == Some('|') {
                self.i += 1;
                self.ws(true);
                alts.push(self.seq(nested)?);
            } else {
                self.i = save;
                break;
            }
        }
        Ok(if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            Node::Alt(alts)
        })
    }
    fn seq(&mut self, nested: bool) -> R<Node> {
        let mut items = vec![];
        loop {
            self.ws(nested);
            match self.peek() {
                None | Some('|') => break,
                Some(')') => {
                    if nested {
                        break;
                    } else {
                        return self.err("unexpected )");
                    }
                }
                Some('\n') => break,
                _ => {}
            }
            let atom = self.atom()?;
            items.push(self.post(atom)?);
        }
        Ok(match items.len() {
            0 => Node::Empty,
            1 => items.pop().unwrap(),
            _ => Node::Seq(items),
        })
    }
    fn num(&mut self) -> R<Option<u32>> {
        let s = self.i;
        let mut v = 0u64;
        while let Some(d) = self.peek().and_then(|c| c.to_digit(10)) {
            v = v * 10 + d as u64;
            if v > 100_000_000 {
                return Err(GrammarError::TooLarge("repetition count"));
            }
            self.i += 1;
        }
        Ok(if self.i == s { None } else { Some(v as u32) })
    }
    fn post(&mut self, mut atom: Node) -> R<Node> {
        loop {
            let (min, max) = match self.peek() {
                Some('*') => {
                    self.i += 1;
                    (0, None)
                }
                Some('+') => {
                    self.i += 1;
                    (1, None)
                }
                Some('?') => {
                    self.i += 1;
                    (0, Some(1))
                }
                Some('{') => {
                    self.i += 1;
                    self.ws(false);
                    let a = self.num()?;
                    self.ws(false);
                    let r = if self.peek() == Some(',') {
                        self.i += 1;
                        self.ws(false);
                        let b = self.num()?;
                        (a.unwrap_or(0), b)
                    } else {
                        match a {
                            Some(a) => (a, Some(a)),
                            None => return self.err("empty {}"),
                        }
                    };
                    self.ws(false);
                    if self.peek() != Some('}') {
                        return self.err("missing }");
                    }
                    self.i += 1;
                    if let Some(m) = r.1 {
                        if m < r.0 {
                            return self.err("upper bound below lower bound");
                        }
                    }
                    r
                }
                _ => return Ok(atom),
            };
            atom = Node::Repeat(Box::new(atom), min, max);
        }
    }
    fn hex(&mut self, n: usize) -> R<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let Some(d) = self.peek().and_then(|c| c.to_digit(16)) else {
                return self.err("bad hex escape");
            };
            v = v * 16 + d;
            self.i += 1;
        }
        Ok(v)
    }
    fn esc(&mut self) -> R<u32> {
        let Some(c) = self.peek() else {
            return self.err("dangling backslash");
        };
        self.i += 1;
        Ok(match c {
            'n' => 10,
            'r' => 13,
            't' => 9,
            '\\' => 92,
            '"' => 34,
            '\'' => 39,
            '[' => 91,
            ']' => 93,
            '-' => 45,
            '^' => 94,
            '.' => 46,
            'x' => self.hex(2)?,
            'u' => self.hex(4)?,
            'U' => self.hex(8)?,
            _ => return self.err("unknown escape"),
        })
    }
    fn atom(&mut self) -> R<Node> {
        let c = self.peek().unwrap();
        match c {
            '"' => {
                self.i += 1;
                let mut s = String::new();
                loop {
                    let Some(c) = self.peek() else {
                        return self.err("unterminated string");
                    };
                    self.i += 1;
                    match c {
                        '"' => break,
                        '\\' => {
                            let v = self.esc()?;
                            match char::from_u32(v) {
                                Some(ch) => s.push(ch),
                                None => return self.err("not a character"),
                            }
                        }
                        '\n' => return self.err("newline inside a string"),
                        c => s.push(c),
                    }
                }
                Ok(Node::Lit(s))
            }
            '[' => {
                self.i += 1;
                let neg = if self.peek() == Some('^') {
                    self.i += 1;
                    true
                } else {
                    false
                };
                let mut ranges = vec![];
                loop {
                    let Some(c) = self.peek() else {
                        return self.err("unterminated [");
                    };
                    self.i += 1;
                    if c == ']' {
                        break;
                    }
                    let lo = if c == '\\' { self.esc()? } else { c as u32 };
                    if self.peek() == Some('-') && self.c.get(self.i + 1).is_some_and(|&n| n != ']')
                    {
                        self.i += 1;
                        let c2 = self.peek().unwrap();
                        self.i += 1;
                        let hi = if c2 == '\\' { self.esc()? } else { c2 as u32 };
                        if hi < lo {
                            return self.err("range out of order");
                        }
                        ranges.push((lo, hi));
                    } else {
                        ranges.push((lo, lo));
                    }
                }
                Ok(Node::Set(CharSet::new(neg, ranges)))
            }
            '(' => {
                self.i += 1;
                self.depth += 1;
                if self.depth > MAX_NEST {
                    return Err(GrammarError::TooLarge("group nesting"));
                }
                let n = self.alts(true)?;
                self.ws(true);
                if self.peek() != Some(')') {
                    return self.err("missing )");
                }
                self.i += 1;
                self.depth -= 1;
                Ok(n)
            }
            '.' => {
                self.i += 1;
                Ok(Node::Set(CharSet::any()))
            }
            '<' => Err(GrammarError::Unsupported(
                "token references (<[id]>) in grammars".into(),
            )),
            '*' | '+' | '?' | '{' => self.err("nothing to repeat"),
            _ => match self.ident() {
                Some(n) => Ok(Node::Ref(n)),
                None => self.err("unexpected character"),
            },
        }
    }
}

/// Parse GBNF text into named rules.
pub fn parse_gbnf(src: &str) -> R<Vec<(String, Node)>> {
    if src.len() > MAX_SRC {
        return Err(GrammarError::TooLarge("grammar text"));
    }
    let mut p = Gb {
        c: src.chars().collect(),
        i: 0,
        depth: 0,
    };
    let mut out = vec![];
    loop {
        p.ws(true);
        if p.peek().is_none() {
            break;
        }
        let Some(name) = p.ident() else {
            return p.err("expected a rule name");
        };
        p.ws(false);
        if !(p.peek() == Some(':')
            && p.c.get(p.i + 1) == Some(&':')
            && p.c.get(p.i + 2) == Some(&'='))
        {
            return p.err("expected ::=");
        }
        p.i += 3;
        p.ws(false);
        let body = p.alts(false)?;
        p.ws(false);
        match p.peek() {
            None | Some('\n') => {}
            _ => return p.err("unexpected text after the rule"),
        }
        out.push((name, body));
    }
    Ok(out)
}

// ------------------------------------------------------------------------- pushdown matcher

const MAX_DEPTH: u32 = 50_000;

fn mix(a: u64, b: u64) -> u64 {
    let x = (a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .rotate_left(27)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 31)
}

/// One cell of a persistent parse stack: "next element to match is
/// `rules[rule][alt][idx]`, and when this alternative ends continue at `parent`".
struct Frame {
    rule: u32,
    alt: u32,
    idx: u32,
    depth: u32,
    hash: u64,
    parent: Stack,
}
type Stack = Option<Arc<Frame>>;

impl Frame {
    fn new(rule: u32, alt: u32, idx: u32, parent: Stack) -> R<Frame> {
        let depth = parent.as_ref().map_or(0, |p| p.depth + 1);
        if depth > MAX_DEPTH {
            return Err(GrammarError::StateOverflow);
        }
        let hash = mix(
            mix(mix(rule as u64 + 1, alt as u64), idx as u64),
            stack_hash(&parent),
        );
        Ok(Frame {
            rule,
            alt,
            idx,
            depth,
            hash,
            parent,
        })
    }
}
impl Drop for Frame {
    // iterative drop: a deeply nested stack must not overflow the real stack
    fn drop(&mut self) {
        let mut next = self.parent.take();
        while let Some(a) = next {
            match Arc::try_unwrap(a) {
                Ok(mut f) => next = f.parent.take(),
                Err(_) => break,
            }
        }
    }
}
fn stack_hash(s: &Stack) -> u64 {
    s.as_ref().map_or(0x5151_5151, |f| f.hash)
}

/// All live parse stacks (each with a character set on top) plus "accepting" flag.
struct StackSet {
    v: Vec<Stack>,
    accepting: bool,
    fp: u64,
    tops: Vec<u32>,
}

fn expand(g: &Grammar, start: Vec<Stack>) -> R<StackSet> {
    let mut work = start;
    let mut seen: HashSet<u64> = HashSet::new();
    let mut out: Vec<Stack> = vec![];
    let mut accepting = false;
    let mut budget = MAX_WORK;
    while let Some(st) = work.pop() {
        if budget == 0 {
            return Err(GrammarError::StateOverflow);
        }
        budget -= 1;
        if !seen.insert(stack_hash(&st)) {
            continue;
        }
        let Some(n) = st else {
            accepting = true;
            continue;
        };
        let alt = &g.rules[n.rule as usize][n.alt as usize];
        if n.idx as usize >= alt.len() {
            work.push(n.parent.clone());
            continue;
        }
        match alt[n.idx as usize] {
            Elem::Set(_) => {
                out.push(Some(n));
                if out.len() > MAX_STACKS {
                    return Err(GrammarError::StateOverflow);
                }
            }
            Elem::Rule(r) => {
                // tail-call elimination: a call in last position needs no return frame
                let next = if n.idx as usize + 1 == alt.len() {
                    n.parent.clone()
                } else {
                    Some(Arc::new(Frame::new(
                        n.rule,
                        n.alt,
                        n.idx + 1,
                        n.parent.clone(),
                    )?))
                };
                for a in 0..g.rules[r as usize].len() {
                    work.push(Some(Arc::new(Frame::new(r, a as u32, 0, next.clone())?)));
                }
            }
        }
    }
    let mut hs: Vec<u64> = out.iter().map(stack_hash).collect();
    hs.sort_unstable();
    let fp = mix(hs.iter().fold(7u64, |a, &h| mix(a, h)), accepting as u64);
    let mut tops: Vec<u32> = out
        .iter()
        .filter_map(|s| {
            let n = s.as_ref()?;
            match g.rules[n.rule as usize][n.alt as usize][n.idx as usize] {
                Elem::Set(i) => Some(i),
                _ => None,
            }
        })
        .collect();
    tops.sort_unstable();
    tops.dedup();
    Ok(StackSet {
        v: out,
        accepting,
        fp,
        tops,
    })
}

fn step_set(g: &Grammar, set: &StackSet, c: u32) -> R<Option<StackSet>> {
    let mut start = vec![];
    for st in &set.v {
        let n = st.as_ref().unwrap();
        if let Elem::Set(s) = g.rules[n.rule as usize][n.alt as usize][n.idx as usize] {
            if g.sets[s as usize].matches(c) {
                start.push(Some(Arc::new(Frame::new(
                    n.rule,
                    n.alt,
                    n.idx + 1,
                    n.parent.clone(),
                )?)));
            }
        }
    }
    if start.is_empty() {
        return Ok(None);
    }
    let s = expand(g, start)?;
    Ok(if s.v.is_empty() && !s.accepting {
        None
    } else {
        Some(s)
    })
}

/// Partially decoded UTF-8 character: bits so far and the range of code points still possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Partial {
    acc: u32,
    need: u8,
    lo: u32,
    hi: u32,
}
const NO_PART: Partial = Partial {
    acc: 0,
    need: 0,
    lo: 0,
    hi: 0,
};

/// Feed one byte; `Err` = invalid UTF-8 (overlong, surrogate, > U+10FFFF included).
fn utf8_feed(p: Partial, b: u8) -> Result<(Partial, Option<u32>), ()> {
    if p.need == 0 {
        return match b {
            0..=0x7f => Ok((NO_PART, Some(b as u32))),
            0xc2..=0xdf => {
                let lo = ((b & 0x1f) as u32) << 6;
                Ok((
                    Partial {
                        acc: lo,
                        need: 1,
                        lo,
                        hi: lo | 0x3f,
                    },
                    None,
                ))
            }
            0xe0..=0xef => {
                let base = ((b & 0x0f) as u32) << 12;
                let lo = if b == 0xe0 { 0x800 } else { base };
                let hi = if b == 0xed { 0xd7ff } else { base | 0xfff };
                Ok((
                    Partial {
                        acc: base,
                        need: 2,
                        lo,
                        hi,
                    },
                    None,
                ))
            }
            0xf0..=0xf4 => {
                let base = ((b & 0x07) as u32) << 18;
                let lo = if b == 0xf0 { 0x10000 } else { base };
                let hi = if b == 0xf4 { 0x10ffff } else { base | 0x3ffff };
                Ok((
                    Partial {
                        acc: base,
                        need: 3,
                        lo,
                        hi,
                    },
                    None,
                ))
            }
            _ => Err(()),
        };
    }
    if !(0x80..=0xbf).contains(&b) {
        return Err(());
    }
    let need = p.need - 1;
    let shift = 6 * need as u32;
    let acc = p.acc | (((b & 0x3f) as u32) << shift);
    let lo = p.lo.max(acc);
    let hi = p.hi.min(acc | ((1u32 << shift) - 1));
    if lo > hi {
        return Err(());
    }
    if need == 0 {
        Ok((NO_PART, Some(acc)))
    } else {
        Ok((Partial { acc, need, lo, hi }, None))
    }
}

/// Memo of `(state fingerprint, char) -> next state`, shared between mask computations.
/// Bounded by an estimate of the bytes it holds (default 64 MiB) as well as by entry count;
/// when either bound is hit it is cleared (entries still in use stay alive through their `Arc`).
pub struct StepCache {
    m: HashMap<(u64, u32), Option<Arc<StackSet>>>,
    bytes: usize,
    max_bytes: usize,
}
impl Default for StepCache {
    fn default() -> Self {
        StepCache::with_limit(64 << 20)
    }
}
const STEP_ENTRY_BASE: usize = 96;
impl StepCache {
    pub fn new() -> StepCache {
        StepCache::default()
    }
    pub fn with_limit(max_bytes: usize) -> StepCache {
        StepCache {
            m: HashMap::new(),
            bytes: 0,
            max_bytes,
        }
    }
    /// Estimated bytes currently held.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn len(&self) -> usize {
        self.m.len()
    }
    pub fn is_empty(&self) -> bool {
        self.m.is_empty()
    }
    fn step(&mut self, g: &Grammar, set: &Arc<StackSet>, c: u32) -> Option<Arc<StackSet>> {
        if let Some(r) = self.m.get(&(set.fp, c)) {
            return r.clone();
        }
        let r = step_set(g, set, c).ok().flatten().map(Arc::new);
        let sz = STEP_ENTRY_BASE + r.as_ref().map_or(0, |x| x.v.len() * 48 + x.tops.len() * 4);
        if self.m.len() > 200_000 || self.bytes + sz > self.max_bytes {
            self.m.clear();
            self.bytes = 0;
        }
        self.bytes += sz;
        self.m.insert((set.fp, c), r.clone());
        r
    }
}

/// Limits for computing one token mask.
#[derive(Clone, Copy, Debug)]
pub struct MaskBudget {
    /// Trie nodes visited plus parse stacks stepped.
    pub max_ops: u64,
    /// Wall-clock limit (checked every 1024 operations); `None` = no clock limit.
    pub max_time: Option<std::time::Duration>,
}
impl Default for MaskBudget {
    fn default() -> Self {
        MaskBudget {
            max_ops: 200_000_000,
            max_time: Some(std::time::Duration::from_secs(5)),
        }
    }
}

/// Compiled grammars keyed by their source text, bounded by their total estimated size
/// (default 256 MiB): inserting past the bound evicts the oldest entries first.
pub struct GrammarCache {
    m: HashMap<String, (Arc<Grammar>, usize)>,
    order: std::collections::VecDeque<String>,
    bytes: usize,
    max_bytes: usize,
}
impl Default for GrammarCache {
    fn default() -> Self {
        GrammarCache::with_limit(256 << 20)
    }
}
impl GrammarCache {
    pub fn with_limit(max_bytes: usize) -> GrammarCache {
        GrammarCache {
            m: HashMap::new(),
            order: Default::default(),
            bytes: 0,
            max_bytes,
        }
    }
    pub fn get(&self, key: &str) -> Option<Arc<Grammar>> {
        self.m.get(key).map(|x| x.0.clone())
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn len(&self) -> usize {
        self.m.len()
    }
    pub fn is_empty(&self) -> bool {
        self.m.is_empty()
    }
    /// Insert (a grammar bigger than the whole bound is returned but not kept).
    pub fn insert(&mut self, key: &str, g: Arc<Grammar>) -> Arc<Grammar> {
        let sz = key.len() + g.element_count() * 64 + g.rule_count() * 128 + 256;
        if let Some(old) = self.m.remove(key) {
            self.bytes -= old.1;
            self.order.retain(|k| k != key);
        }
        if sz > self.max_bytes {
            return g;
        }
        while self.bytes + sz > self.max_bytes {
            match self.order.pop_front() {
                Some(k) => {
                    if let Some(o) = self.m.remove(&k) {
                        self.bytes -= o.1;
                    }
                }
                None => break,
            }
        }
        self.bytes += sz;
        self.order.push_back(key.to_string());
        self.m.insert(key.to_string(), (g.clone(), sz));
        g
    }
}

/// Incremental matcher: feed bytes (or chars), ask what may come next.
/// Cloning is cheap (shares the persistent stacks).
#[derive(Clone)]
pub struct GrammarState {
    g: Arc<Grammar>,
    set: Arc<StackSet>,
    part: Partial,
    dead: bool,
}

impl GrammarState {
    pub fn new(g: Arc<Grammar>) -> R<GrammarState> {
        let root = g.root;
        let mut start = vec![];
        for a in 0..g.rules[root as usize].len() {
            start.push(Some(Arc::new(Frame::new(root, a as u32, 0, None)?)));
        }
        let set = Arc::new(expand(&g, start)?);
        Ok(GrammarState {
            g,
            set,
            part: NO_PART,
            dead: false,
        })
    }
    pub fn grammar(&self) -> &Arc<Grammar> {
        &self.g
    }
    /// The input so far can no longer be completed.
    pub fn is_dead(&self) -> bool {
        self.dead
    }
    /// The input so far is a full sentence of the grammar (and ends on a character boundary).
    pub fn is_complete(&self) -> bool {
        !self.dead && self.part.need == 0 && self.set.accepting
    }
    /// Identity of the state, usable as a cache key.
    pub fn fingerprint(&self) -> u64 {
        mix(
            mix(
                self.set.fp,
                self.part.acc as u64 | (self.part.need as u64) << 32,
            ),
            self.dead as u64,
        )
    }
    pub fn stack_count(&self) -> usize {
        self.set.v.len()
    }

    pub fn advance_char(&mut self, c: char) -> R<bool> {
        self.advance_cp(c as u32)
    }
    fn advance_cp(&mut self, c: u32) -> R<bool> {
        if self.dead || self.part.need != 0 {
            self.dead = true;
            return Ok(false);
        }
        match step_set(&self.g, &self.set, c)? {
            Some(s) => {
                self.set = Arc::new(s);
                Ok(true)
            }
            None => {
                self.dead = true;
                Ok(false)
            }
        }
    }
    /// Feed one byte; `Ok(false)` means the grammar (or UTF-8) rejects it and the state is now dead.
    pub fn advance_byte(&mut self, b: u8) -> R<bool> {
        if self.dead {
            return Ok(false);
        }
        match utf8_feed(self.part, b) {
            Err(()) => {
                self.dead = true;
                Ok(false)
            }
            Ok((p, None)) => {
                if !self
                    .set
                    .tops
                    .iter()
                    .any(|&i| self.g.sets[i as usize].intersects(p.lo, p.hi))
                {
                    self.dead = true;
                    return Ok(false);
                }
                self.part = p;
                Ok(true)
            }
            Ok((_, Some(cp))) => {
                self.part = NO_PART;
                self.advance_cp(cp)
            }
        }
    }
    pub fn advance_bytes(&mut self, bytes: &[u8]) -> R<bool> {
        for &b in bytes {
            if !self.advance_byte(b)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub fn advance_str(&mut self, s: &str) -> R<bool> {
        self.advance_bytes(s.as_bytes())
    }

    /// Advance by a whole vocabulary token; on rejection the state is left untouched.
    pub fn advance_token(&mut self, trie: &TokenTrie, id: u32) -> R<bool> {
        let b = trie.token_bytes(id);
        if b.is_empty() {
            return Ok(false);
        }
        let mut t = self.clone();
        if t.advance_bytes(b)? {
            *self = t;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Some character that is allowed next (for tests / demos), if any.
    pub fn sample_next_chars(&self, limit: usize) -> Vec<char> {
        let mut v = vec![];
        if self.part.need == 0 && !self.dead {
            for &i in &self.set.tops {
                self.g.sets[i as usize].representatives(&mut v, limit);
            }
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    pub fn allowed_tokens(&self, trie: &TokenTrie) -> BitSet {
        self.allowed_tokens_cached(trie, &mut StepCache::new())
    }
    /// Bit set of vocabulary tokens whose bytes keep the grammar alive; EOS only when complete.
    ///
    /// Best effort under the default [`MaskBudget`]: if the budget runs out the returned set holds
    /// only tokens found so far (every one valid, but possibly incomplete); use
    /// [`GrammarState::try_allowed_tokens`] to get `GrammarError::MaskBudget` instead.
    pub fn allowed_tokens_cached(&self, trie: &TokenTrie, cache: &mut StepCache) -> BitSet {
        self.mask_inner(trie, cache, MaskBudget::default()).0
    }
    /// Like [`GrammarState::allowed_tokens_cached`] with an explicit budget and a typed error.
    pub fn try_allowed_tokens(
        &self,
        trie: &TokenTrie,
        cache: &mut StepCache,
        budget: MaskBudget,
    ) -> R<BitSet> {
        let (m, over) = self.mask_inner(trie, cache, budget);
        if over {
            Err(GrammarError::MaskBudget)
        } else {
            Ok(m)
        }
    }
    fn mask_inner(
        &self,
        trie: &TokenTrie,
        cache: &mut StepCache,
        budget: MaskBudget,
    ) -> (BitSet, bool) {
        let deadline = budget.max_time.map(|d| std::time::Instant::now() + d);
        let mut m = Mask {
            g: &self.g,
            trie,
            out: BitSet::new(trie.vocab_size()),
            cache,
            ops: 0,
            max_ops: budget.max_ops,
            deadline,
            over: false,
        };
        if !self.dead && !(self.set.v.is_empty() && self.part.need == 0) {
            m.walk(0, &self.set, self.part);
        }
        let over = m.over;
        let mut out = m.out;
        if let Some(e) = trie.eos {
            if self.is_complete() {
                out.set(e as usize);
            }
        }
        (out, over)
    }
}

struct Mask<'a> {
    g: &'a Grammar,
    trie: &'a TokenTrie,
    out: BitSet,
    cache: &'a mut StepCache,
    ops: u64,
    max_ops: u64,
    deadline: Option<std::time::Instant>,
    over: bool,
}
impl Mask<'_> {
    fn tick(&mut self, n: u64) -> bool {
        if self.over {
            return false;
        }
        let before = self.ops;
        self.ops = self.ops.saturating_add(n);
        if self.ops > self.max_ops {
            self.over = true;
            return false;
        }
        if before >> 10 != self.ops >> 10 {
            if let Some(d) = self.deadline {
                if std::time::Instant::now() >= d {
                    self.over = true;
                    return false;
                }
            }
        }
        true
    }
    fn walk(&mut self, node: u32, set: &Arc<StackSet>, part: Partial) {
        if !self.tick(1 + set.v.len() as u64) {
            return;
        }
        let n = self.trie.nodes[node as usize];
        for &t in &self.trie.toks[n.tk as usize..(n.tk + n.tn) as usize] {
            self.out.set(t as usize);
        }
        for k in n.ch..n.ch + n.cn {
            let (b, child) = self.trie.children[k as usize];
            match utf8_feed(part, b) {
                Err(()) => {}
                Ok((_, Some(cp))) => {
                    if let Some(ns) = self.cache.step(self.g, set, cp) {
                        self.walk(child, &ns, NO_PART);
                    }
                }
                Ok((p, None)) => {
                    if set
                        .tops
                        .iter()
                        .any(|&i| self.g.sets[i as usize].intersects(p.lo, p.hi))
                    {
                        self.walk(child, set, p);
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------------- token trie

#[derive(Clone, Copy, Debug)]
struct TNode {
    ch: u32,
    cn: u32,
    tk: u32,
    tn: u32,
}

/// Byte trie over the vocabulary (each token as the bytes it decodes to).
pub struct TokenTrie {
    nodes: Vec<TNode>,
    children: Vec<(u8, u32)>,
    toks: Vec<u32>,
    bytes: Vec<u8>,
    offs: Vec<u32>,
    pub eos: Option<u32>,
}

const MAX_TOKEN_BYTES: usize = 256;

fn gpt2_inverse() -> HashMap<char, u8> {
    let mut keep: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut m = HashMap::new();
    for &b in &keep {
        m.insert(char::from_u32(b).unwrap(), b as u8);
    }
    keep.sort_unstable();
    let mut n = 0u32;
    for b in 0..256u32 {
        if keep.binary_search(&b).is_err() {
            m.insert(char::from_u32(256 + n).unwrap(), b as u8);
            n += 1;
        }
    }
    m
}

impl TokenTrie {
    /// Build from explicit token bytes (empty = never allowed, e.g. special tokens).
    pub fn from_token_bytes(tokens: &[Vec<u8>], eos: Option<u32>) -> TokenTrie {
        struct B {
            ch: Vec<(u8, u32)>,
            tk: Vec<u32>,
        }
        let mut nodes = vec![B {
            ch: vec![],
            tk: vec![],
        }];
        let mut bytes = vec![];
        let mut offs = vec![0u32];
        for (id, t) in tokens.iter().enumerate() {
            if !t.is_empty() && t.len() <= MAX_TOKEN_BYTES {
                bytes.extend_from_slice(t);
                let mut cur = 0usize;
                for &b in t {
                    cur = match nodes[cur].ch.iter().find(|c| c.0 == b) {
                        Some(&(_, c)) => c as usize,
                        None => {
                            nodes.push(B {
                                ch: vec![],
                                tk: vec![],
                            });
                            let c = nodes.len() - 1;
                            nodes[cur].ch.push((b, c as u32));
                            c
                        }
                    };
                }
                nodes[cur].tk.push(id as u32);
            }
            offs.push(bytes.len() as u32);
        }
        let (mut fl, mut children, mut toks) = (Vec::with_capacity(nodes.len()), vec![], vec![]);
        for b in &mut nodes {
            b.ch.sort_unstable();
            fl.push(TNode {
                ch: children.len() as u32,
                cn: b.ch.len() as u32,
                tk: toks.len() as u32,
                tn: b.tk.len() as u32,
            });
            children.extend_from_slice(&b.ch);
            toks.extend_from_slice(&b.tk);
        }
        TokenTrie {
            nodes: fl,
            children,
            toks,
            bytes,
            offs,
            eos,
        }
    }

    /// Build from a loaded tokenizer; special tokens are excluded, EOS is handled separately.
    pub fn from_tokenizer(t: &Tokenizer) -> TokenTrie {
        let inv = gpt2_inverse();
        let byte_level = matches!(t.kind, Kind::ByteLevel);
        let mut toks = Vec::with_capacity(t.vocab_size());
        for id in 0..t.vocab_size() as u32 {
            let Some(s) = t.id_to_token(id) else {
                toks.push(vec![]);
                continue;
            };
            if t.is_special(id) {
                toks.push(vec![]);
                continue;
            }
            let mut out = vec![];
            if byte_level {
                for ch in s.chars() {
                    match inv.get(&ch) {
                        Some(&b) => out.push(b),
                        None => {
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                }
            } else if s.len() == 6
                && s.starts_with("<0x")
                && s.ends_with('>')
                && u8::from_str_radix(&s[3..5], 16).is_ok()
            {
                out.push(u8::from_str_radix(&s[3..5], 16).unwrap());
            } else {
                out.extend_from_slice(s.replace('\u{2581}', " ").as_bytes());
            }
            toks.push(out);
        }
        TokenTrie::from_token_bytes(&toks, t.eos)
    }

    pub fn vocab_size(&self) -> usize {
        self.offs.len() - 1
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        let i = id as usize;
        if i + 1 >= self.offs.len() {
            return &[];
        }
        &self.bytes[self.offs[i] as usize..self.offs[i + 1] as usize]
    }
}

// ------------------------------------------------------------------------- sampler-facing constraint

const MASK_CACHE: usize = 2048;
/// Total bytes the per-state mask cache may hold before it is cleared.
const MASK_CACHE_BYTES: usize = 128 << 20;

/// What a sampler holds: `mask()` before sampling, `accept(token)` after.
pub struct Constraint {
    state: GrammarState,
    trie: Arc<TokenTrie>,
    masks: HashMap<u64, BitSet>,
    steps: StepCache,
    finished: bool,
    budget: MaskBudget,
    mask_bytes: usize,
    budget_hit: bool,
}

impl Constraint {
    pub fn new(grammar: Arc<Grammar>, trie: Arc<TokenTrie>) -> R<Constraint> {
        Ok(Constraint {
            state: GrammarState::new(grammar)?,
            trie,
            masks: HashMap::new(),
            steps: StepCache::new(),
            finished: false,
            budget: MaskBudget::default(),
            mask_bytes: 0,
            budget_hit: false,
        })
    }
    pub fn state(&self) -> &GrammarState {
        &self.state
    }
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }
    pub fn is_finished(&self) -> bool {
        self.finished
    }
    pub fn reset(&mut self) -> R<()> {
        self.state = GrammarState::new(self.state.g.clone())?;
        self.finished = false;
        Ok(())
    }
    /// Replace the per-mask work/time budget.
    pub fn set_budget(&mut self, b: MaskBudget) {
        self.budget = b;
    }
    /// True once a mask computation ran out of budget (the mask served was then partial).
    pub fn budget_hit(&self) -> bool {
        self.budget_hit
    }
    fn compute(&mut self, key: u64) -> R<()> {
        if !self.masks.contains_key(&key) {
            let one = self.trie.vocab_size().div_ceil(64) * 8 + 48;
            if self.masks.len() >= MASK_CACHE || self.mask_bytes + one > MASK_CACHE_BYTES {
                self.masks.clear();
                self.mask_bytes = 0;
            }
            let m = if self.finished {
                let mut m = BitSet::new(self.trie.vocab_size());
                if let Some(e) = self.trie.eos {
                    m.set(e as usize);
                }
                m
            } else {
                match self
                    .state
                    .try_allowed_tokens(&self.trie, &mut self.steps, self.budget)
                {
                    Ok(m) => m,
                    Err(e) => {
                        self.budget_hit = true;
                        return Err(e);
                    }
                }
            };
            self.mask_bytes += one;
            self.masks.insert(key, m);
        }
        Ok(())
    }
    /// Tokens the model may emit now, or `MaskBudget` when computing them was too expensive.
    pub fn try_mask(&mut self) -> R<&BitSet> {
        let key = mix(self.state.fingerprint(), self.finished as u64);
        self.compute(key)?;
        Ok(&self.masks[&key])
    }
    /// Tokens the model may emit now (EOS only when the output is a complete sentence).
    /// On budget exhaustion the partial mask is served (not cached) and `budget_hit()` turns true.
    pub fn mask(&mut self) -> &BitSet {
        let key = mix(self.state.fingerprint(), self.finished as u64);
        if self.compute(key).is_err() {
            let (m, _) = self
                .state
                .mask_inner(&self.trie, &mut self.steps, self.budget);
            self.masks.insert(key, m);
            self.mask_bytes += self.trie.vocab_size().div_ceil(64) * 8 + 48;
        }
        &self.masks[&key]
    }
    /// Set disallowed logits to -inf.
    pub fn apply(&mut self, logits: &mut [f32]) {
        let m = self.mask().clone();
        apply_mask(logits, &m);
    }
    /// Tell the constraint which token was sampled. `Ok(false)` = token not allowed (state unchanged).
    pub fn accept(&mut self, token: u32) -> R<bool> {
        if self.finished {
            return Ok(self.trie.eos == Some(token));
        }
        if self.trie.eos == Some(token) {
            if self.state.is_complete() {
                self.finished = true;
                return Ok(true);
            }
            return Ok(false);
        }
        self.state.advance_token(&self.trie, token)
    }
}

// ------------------------------------------------------------------------- JSON schema -> grammar

/// How much white space the generated JSON may contain between tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Whitespace {
    /// None at all: `{"a":1,"b":[2,3]}`.
    Compact,
    /// At most one space: `{"a": 1, "b": [2, 3]}`.
    Single,
    /// Up to `max_ws` spaces, tabs or newlines at every gap.
    Flexible,
}

#[derive(Clone, Debug)]
pub struct SchemaOptions {
    pub whitespace: Whitespace,
    /// Upper bound of one white-space run in `Flexible` mode (keeps the model from looping on blanks).
    pub max_ws: u32,
    /// Allow keys not listed in `properties` when the schema does not say (JSON Schema's default is true,
    /// but for structured output `false` is what callers expect).
    pub additional_properties: bool,
}
impl Default for SchemaOptions {
    fn default() -> Self {
        SchemaOptions {
            whitespace: Whitespace::Flexible,
            max_ws: 4,
            additional_properties: false,
        }
    }
}

struct Conv {
    defs: Vec<(String, Node)>,
    opts: SchemaOptions,
    root: Value,
    refs: HashMap<String, String>,
    counter: usize,
    prims_done: bool,
}

fn digit() -> Node {
    setr(&[('0', '9')])
}

impl Conv {
    fn new(root: &Value, opts: &SchemaOptions) -> Conv {
        Conv {
            defs: vec![],
            opts: opts.clone(),
            root: root.clone(),
            refs: HashMap::new(),
            counter: 0,
            prims_done: false,
        }
    }
    fn add(&mut self, name: String, n: Node) {
        self.defs.push((name, n));
    }
    fn fresh(&mut self, p: &str) -> String {
        self.counter += 1;
        format!("{p}-{}", self.counter)
    }
    /// Define `n` as a rule of its own and return a reference to it.
    fn rule(&mut self, p: &str, n: Node) -> Node {
        let name = self.fresh(p);
        self.add(name.clone(), n);
        Node::Ref(name)
    }

    fn prims(&mut self) {
        if self.prims_done {
            return;
        }
        self.prims_done = true;
        let ws = match self.opts.whitespace {
            Whitespace::Compact => Node::Empty,
            Whitespace::Single => opt(lit(" ")),
            Whitespace::Flexible => rep(
                setr(&[(' ', ' '), ('\t', '\t'), ('\n', '\n')]),
                0,
                Some(self.opts.max_ws),
            ),
        };
        self.add("ws".into(), ws);
        self.add("boolean".into(), alt(vec![lit("true"), lit("false")]));
        self.add("null".into(), lit("null"));
        let int = alt(vec![
            lit("0"),
            seq(vec![setr(&[('1', '9')]), star(digit())]),
        ]);
        self.add("uinteger".into(), int.clone());
        self.add("integer".into(), seq(vec![opt(lit("-")), int.clone()]));
        let frac = seq(vec![
            opt(seq(vec![lit("."), rep(digit(), 1, None)])),
            opt(seq(vec![
                setr(&[('e', 'e'), ('E', 'E')]),
                opt(setr(&[('+', '+'), ('-', '-')])),
                rep(digit(), 1, None),
            ])),
        ]);
        self.add("unumber".into(), seq(vec![int.clone(), frac.clone()]));
        self.add("number".into(), seq(vec![opt(lit("-")), int, frac]));
        let esc = alt(vec![
            setr(&[
                ('"', '"'),
                ('\\', '\\'),
                ('/', '/'),
                ('b', 'b'),
                ('f', 'f'),
                ('n', 'n'),
                ('r', 'r'),
                ('t', 't'),
            ]),
            seq(vec![
                lit("u"),
                rep(setr(&[('0', '9'), ('a', 'f'), ('A', 'F')]), 4, Some(4)),
            ]),
        ]);
        self.add(
            "string-char".into(),
            alt(vec![
                Node::Set(CharSet::new(true, JSON_BAD.to_vec())),
                seq(vec![lit("\\"), esc]),
            ]),
        );
        self.add(
            "string".into(),
            seq(vec![lit("\""), star(rref("string-char")), lit("\"")]),
        );
        self.add(
            "value-any".into(),
            alt(vec![
                rref("object-any"),
                rref("array-any"),
                rref("string"),
                rref("number"),
                rref("boolean"),
                rref("null"),
            ]),
        );
        let pair = || {
            seq(vec![
                rref("string"),
                rref("ws"),
                lit(":"),
                rref("ws"),
                rref("value-any"),
            ])
        };
        let sep = || seq(vec![rref("ws"), lit(","), rref("ws")]);
        self.add(
            "object-any".into(),
            seq(vec![
                lit("{"),
                rref("ws"),
                opt(seq(vec![pair(), star(seq(vec![sep(), pair()]))])),
                rref("ws"),
                lit("}"),
            ]),
        );
        self.add(
            "array-any".into(),
            seq(vec![
                lit("["),
                rref("ws"),
                opt(seq(vec![
                    rref("value-any"),
                    star(seq(vec![sep(), rref("value-any")])),
                ])),
                rref("ws"),
                lit("]"),
            ]),
        );
    }

    fn sep(&self) -> Node {
        seq(vec![rref("ws"), lit(","), rref("ws")])
    }

    fn resolve(&self, r: &str) -> R<Value> {
        if r == "#" {
            return Ok(self.root.clone());
        }
        if let Some(p) = r.strip_prefix('#') {
            if p.starts_with('/') {
                if let Some(v) = self.root.pointer(p) {
                    return Ok(v.clone());
                }
            }
            return Err(GrammarError::Schema(format!("cannot resolve $ref {r}")));
        }
        Err(GrammarError::Unsupported(format!("non-local $ref {r}")))
    }

    fn visit(&mut self, s: &Value, depth: usize) -> R<Node> {
        if depth > 64 {
            return Err(GrammarError::TooLarge("schema nesting"));
        }
        let obj = match s {
            Value::Bool(true) => return Ok(rref("value-any")),
            Value::Bool(false) => {
                return Err(GrammarError::Schema(
                    "the schema `false` matches nothing".into(),
                ));
            }
            Value::Object(o) => o,
            _ => {
                return Err(GrammarError::Schema(
                    "a schema must be an object or a boolean".into(),
                ));
            }
        };
        if let Some(Value::String(r)) = obj.get("$ref") {
            if let Some(name) = self.refs.get(r) {
                return Ok(Node::Ref(name.clone()));
            }
            let name = self.fresh("ref");
            self.refs.insert(r.clone(), name.clone());
            let target = self.resolve(r)?;
            let n = self.visit(&target, depth + 1)?;
            self.add(name.clone(), n);
            return Ok(Node::Ref(name));
        }
        let inner = self.visit_core(obj, depth)?;
        if obj.get("nullable") == Some(&Value::Bool(true)) {
            return Ok(alt(vec![inner, rref("null")]));
        }
        Ok(inner)
    }

    fn visit_core(&mut self, obj: &serde_json::Map<String, Value>, depth: usize) -> R<Node> {
        if let Some(c) = obj.get("const") {
            return Ok(Node::Lit(c.to_string()));
        }
        if let Some(Value::Array(e)) = obj.get("enum") {
            if e.is_empty() {
                return Err(GrammarError::Schema("empty enum".into()));
            }
            return Ok(alt(e.iter().map(|v| Node::Lit(v.to_string())).collect()));
        }
        if let Some(Value::Array(all)) = obj.get("allOf") {
            let merged = self.merge_all(obj, all)?;
            return self.visit(&merged, depth + 1);
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(Value::Array(a)) = obj.get(key) {
                if a.is_empty() {
                    return Err(GrammarError::Schema(format!("empty {key}")));
                }
                let mut alts = vec![];
                for sub in a {
                    alts.push(self.visit(sub, depth + 1)?);
                }
                return Ok(self.rule("any", alt(alts)));
            }
        }
        match obj.get("type") {
            Some(Value::Array(ts)) => {
                let mut alts = vec![];
                for t in ts {
                    let mut o = obj.clone();
                    o.insert("type".into(), t.clone());
                    alts.push(self.visit_core(&o, depth + 1)?);
                }
                if alts.is_empty() {
                    return Err(GrammarError::Schema("empty type list".into()));
                }
                Ok(self.rule("any", alt(alts)))
            }
            Some(Value::String(t)) => self.typed(t, obj, depth),
            Some(_) => Err(GrammarError::Schema(
                "`type` must be a string or an array".into(),
            )),
            None => {
                if obj.contains_key("properties") || obj.contains_key("additionalProperties") {
                    self.typed("object", obj, depth)
                } else if obj.contains_key("items") || obj.contains_key("prefixItems") {
                    self.typed("array", obj, depth)
                } else {
                    Ok(rref("value-any"))
                }
            }
        }
    }

    fn merge_all(&self, base: &serde_json::Map<String, Value>, all: &[Value]) -> R<Value> {
        let mut out = base.clone();
        out.remove("allOf");
        for sub in all {
            let sub = match sub.get("$ref").and_then(Value::as_str) {
                Some(r) => self.resolve(r)?,
                None => sub.clone(),
            };
            let Value::Object(so) = sub else {
                return Err(GrammarError::Schema(
                    "allOf member must be an object".into(),
                ));
            };
            for (k, v) in so {
                match (k.as_str(), out.get_mut(&k)) {
                    ("properties", Some(Value::Object(m))) => {
                        if let Value::Object(n) = v {
                            for (a, b) in n {
                                m.entry(a).or_insert(b);
                            }
                        }
                    }
                    ("required", Some(Value::Array(m))) => {
                        if let Value::Array(n) = v {
                            for x in n {
                                if !m.contains(&x) {
                                    m.push(x);
                                }
                            }
                        }
                    }
                    (_, Some(_)) => {}
                    (_, None) => {
                        out.insert(k, v);
                    }
                }
            }
        }
        Ok(Value::Object(out))
    }

    fn typed(&mut self, t: &str, o: &serde_json::Map<String, Value>, depth: usize) -> R<Node> {
        let uint = |k: &str| o.get(k).and_then(Value::as_f64).is_some_and(|m| m >= 0.0);
        let non_neg = uint("minimum")
            || o.get("exclusiveMinimum")
                .and_then(Value::as_f64)
                .is_some_and(|m| m >= 0.0);
        match t {
            "boolean" => Ok(rref("boolean")),
            "null" => Ok(rref("null")),
            "integer" => Ok(rref(if non_neg { "uinteger" } else { "integer" })),
            "number" => Ok(rref(if non_neg { "unumber" } else { "number" })),
            "string" => self.string(o),
            "array" => self.array(o, depth),
            "object" => self.object(o, depth),
            other => Err(GrammarError::Schema(format!("unknown type `{other}`"))),
        }
    }

    fn string(&mut self, o: &serde_json::Map<String, Value>) -> R<Node> {
        let q = || lit("\"");
        let min = o
            .get("minLength")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(u32::MAX as u64) as u32;
        let max = o
            .get("maxLength")
            .and_then(Value::as_u64)
            .map(|m| m.min(u32::MAX as u64) as u32);
        if let Some(Value::String(p)) = o.get("pattern") {
            let (n, a, b) = regex_node(p, true)?;
            let any = || star(rref("string-char"));
            let mut parts = vec![q()];
            if !a {
                parts.push(any());
            }
            parts.push(n);
            if !b {
                parts.push(any());
            }
            parts.push(q());
            return Ok(self.rule("pattern", seq(parts)));
        }
        if let Some(Value::String(f)) = o.get("format") {
            const DATE: &str = r"[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])";
            const TIME: &str = r"([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.[0-9]+)?(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])";
            let pat = match f.as_str() {
                "date" => Some(DATE.to_string()),
                "time" => Some(TIME.to_string()),
                "date-time" => Some(format!("{DATE}T{TIME}")),
                "uuid" => Some(
                    "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
                        .to_string(),
                ),
                "email" => Some(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}".to_string()),
                _ => None,
            };
            if let Some(p) = pat {
                let (n, _, _) = regex_node(&p, true)?;
                return Ok(self.rule("format", seq(vec![q(), n, q()])));
            }
        }
        if min == 0 && max.is_none() {
            return Ok(rref("string"));
        }
        if let Some(m) = max {
            if m < min {
                return Err(GrammarError::Schema("maxLength below minLength".into()));
            }
        }
        Ok(self.rule(
            "str",
            seq(vec![q(), rep(rref("string-char"), min, max), q()]),
        ))
    }

    fn array(&mut self, o: &serde_json::Map<String, Value>, depth: usize) -> R<Node> {
        let min = o
            .get("minItems")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(u32::MAX as u64) as u32;
        let max = o
            .get("maxItems")
            .and_then(Value::as_u64)
            .map(|m| m.min(u32::MAX as u64) as u32);
        if let Some(m) = max {
            if m < min {
                return Err(GrammarError::Schema("maxItems below minItems".into()));
            }
        }
        let open = || seq(vec![lit("["), rref("ws")]);
        let close = || seq(vec![rref("ws"), lit("]")]);
        if let Some(Value::Array(prefix)) = o.get("prefixItems") {
            let mut parts = vec![open()];
            for (i, p) in prefix.iter().enumerate() {
                if i > 0 {
                    parts.push(self.sep());
                }
                parts.push(self.visit(p, depth + 1)?);
            }
            if let Some(items @ Value::Object(_)) = o.get("items") {
                let item = self.visit(items, depth + 1)?;
                let item = self.rule("item", item);
                let left = max.map(|m| m.saturating_sub(prefix.len() as u32));
                parts.push(rep(seq(vec![self.sep(), item]), 0, left));
            }
            parts.push(close());
            return Ok(self.rule("tuple", seq(parts)));
        }
        let item = match o.get("items") {
            Some(i) => self.visit(i, depth + 1)?,
            None => rref("value-any"),
        };
        let item = self.rule("item", item);
        if max == Some(0) {
            return Ok(self.rule("array", seq(vec![open(), close()])));
        }
        let rest = rep(
            seq(vec![self.sep(), item.clone()]),
            min.saturating_sub(1),
            max.map(|m| m - 1),
        );
        let body = seq(vec![item, rest]);
        let body = if min == 0 { opt(body) } else { body };
        Ok(self.rule("array", seq(vec![open(), body, close()])))
    }

    fn object(&mut self, o: &serde_json::Map<String, Value>, depth: usize) -> R<Node> {
        let props: Vec<(String, Value)> = match o.get("properties") {
            Some(Value::Object(m)) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            Some(_) => {
                return Err(GrammarError::Schema(
                    "`properties` must be an object".into(),
                ));
            }
            None => vec![],
        };
        let required: HashSet<String> = match o.get("required") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            _ => HashSet::new(),
        };
        for r in &required {
            if !props.iter().any(|(k, _)| k == r)
                && matches!(o.get("additionalProperties"), Some(Value::Bool(false)))
            {
                return Err(GrammarError::Schema(format!(
                    "required property `{r}` is not allowed by the schema"
                )));
            }
        }
        let extra: Option<Node> = match o.get("additionalProperties") {
            Some(Value::Bool(false)) => None,
            Some(Value::Bool(true)) => Some(rref("value-any")),
            Some(v @ Value::Object(_)) => Some(self.visit(v, depth + 1)?),
            _ => {
                if self.opts.additional_properties {
                    Some(rref("value-any"))
                } else {
                    None
                }
            }
        };
        if props.is_empty() && extra.is_none() && required.is_empty() {
            return Ok(self.rule("object", seq(vec![lit("{"), rref("ws"), lit("}")])));
        }
        if props.is_empty() && matches!(extra, Some(Node::Ref(ref n)) if n == "value-any") {
            return Ok(rref("object-any"));
        }
        let mut kvs = vec![];
        for (k, v) in &props {
            let val = self.visit(v, depth + 1)?;
            let kv = seq(vec![
                Node::Lit(Value::String(k.clone()).to_string()),
                rref("ws"),
                lit(":"),
                rref("ws"),
                val,
            ]);
            kvs.push((self.rule("kv", kv), required.contains(k)));
        }
        let extra_pair = extra.map(|e| {
            let p = seq(vec![rref("string"), rref("ws"), lit(":"), rref("ws"), e]);
            self.rule("kv", p)
        });
        let n = kvs.len();
        // chain[i][started]: the properties from i on, `started` = something was already written
        let base = self.fresh("obj");
        let name = |i: usize, st: bool| format!("{base}-{i}-{}", st as u8);
        for i in (0..=n).rev() {
            for st in [false, true] {
                let comma = if st { self.sep() } else { Node::Empty };
                let body = if i == n {
                    match &extra_pair {
                        None => Node::Empty,
                        Some(p) if st => star(seq(vec![self.sep(), p.clone()])),
                        Some(p) => {
                            opt(seq(vec![p.clone(), star(seq(vec![self.sep(), p.clone()]))]))
                        }
                    }
                } else {
                    let (kv, req) = &kvs[i];
                    let with = seq(vec![comma, kv.clone(), rref(&name(i + 1, true))]);
                    if *req {
                        with
                    } else {
                        alt(vec![with, rref(&name(i + 1, st))])
                    }
                };
                self.add(name(i, st), body);
            }
        }
        let obj = seq(vec![
            lit("{"),
            rref("ws"),
            rref(&name(0, false)),
            rref("ws"),
            lit("}"),
        ]);
        Ok(self.rule("object", obj))
    }
}

fn schema_to_grammar(schema: &Value, opts: &SchemaOptions) -> R<Grammar> {
    let mut c = Conv::new(schema, opts);
    c.prims();
    let n = c.visit(schema, 0)?;
    c.add("root".into(), n);
    Grammar::from_nodes(c.defs, "root")
}

/// Turn an OpenAI-style `response_format` object into a grammar.
/// `Ok(None)` for plain text (no constraint).
///
/// Understands `{"type":"json_object"}` (optionally with `"schema"`),
/// `{"type":"json_schema","json_schema":{"schema":{..}}}` and `{"type":"grammar","grammar":"<GBNF>"}`.
pub fn response_format_to_grammar(rf: &Value, opts: &SchemaOptions) -> R<Option<Grammar>> {
    let t = rf.get("type").and_then(Value::as_str).unwrap_or("text");
    match t {
        "text" => Ok(None),
        "json_object" => match rf.get("schema") {
            Some(s) => Ok(Some(Grammar::from_json_schema(s, opts)?)),
            None => Ok(Some(Grammar::json_object(opts)?)),
        },
        "json_schema" => {
            let s = rf
                .pointer("/json_schema/schema")
                .or_else(|| rf.get("schema"))
                .ok_or_else(|| {
                    GrammarError::Schema("json_schema response_format without a schema".into())
                })?;
            Ok(Some(Grammar::from_json_schema(s, opts)?))
        }
        "grammar" => {
            let g = rf
                .get("grammar")
                .and_then(Value::as_str)
                .ok_or_else(|| GrammarError::Schema("missing grammar text".into()))?;
            Ok(Some(Grammar::from_gbnf(g, "root")?))
        }
        other => Err(GrammarError::Unsupported(format!(
            "response_format type `{other}`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{Duration, Instant};

    fn g(src: &str) -> Arc<Grammar> {
        Arc::new(Grammar::from_gbnf(src, "root").unwrap())
    }
    fn schema(v: &Value, ws: Whitespace) -> Arc<Grammar> {
        Arc::new(
            Grammar::from_json_schema(
                v,
                &SchemaOptions {
                    whitespace: ws,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
    }
    fn timed<T>(f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let r = f();
        assert!(
            t.elapsed() < Duration::from_secs(30),
            "took {:?}",
            t.elapsed()
        );
        r
    }
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    const ARITH: &str = r#"
root ::= expr
expr ::= term (("+" | "-") term)*
term ::= factor (("*" | "/") factor)*
factor ::= [0-9]+ | "(" expr ")"
"#;

    #[test]
    fn arithmetic() {
        let a = g(ARITH);
        for ok in ["1", "12+3", "1+2*(3-4)/5", "((7))", "(1+2)*(3+4)"] {
            assert!(a.matches(ok).unwrap(), "{ok}");
        }
        for bad in ["", "1+", "(1", "1)", "+1", "1 + 2", "a"] {
            assert!(!a.matches(bad).unwrap(), "{bad}");
        }
    }

    #[test]
    fn ambiguity_and_prefixes() {
        let a = g(r#"root ::= "x"* "x""#);
        assert!(!a.matches("").unwrap());
        assert!(a.matches("x").unwrap() && a.matches("xxxxx").unwrap());
        let b = g(r#"root ::= a b ; a ::= "x" | "xx" ; b ::= "x" | "xx""#
            .replace(';', "\n")
            .as_str());
        for s in ["xx", "xxx", "xxxx"] {
            assert!(b.matches(s).unwrap(), "{s}");
        }
        assert!(!b.matches("x").unwrap() && !b.matches("xxxxx").unwrap());
        // state completeness while a longer match is still possible
        let mut s = GrammarState::new(a).unwrap();
        s.advance_str("xx").unwrap();
        assert!(s.is_complete() && !s.is_dead());
    }

    #[test]
    fn recursion_is_bounded() {
        let r = g(r#"root ::= "a" root | """#);
        assert!(r.matches(&"a".repeat(30_000)).unwrap());
        let c = g(r#"root ::= "(" root ")" | """#);
        let deep = format!("{}{}", "(".repeat(5000), ")".repeat(5000));
        timed(|| assert!(c.matches(&deep).unwrap()));
        assert!(!c.matches("(()").unwrap());
        // drop of a very deep stack must not overflow the real stack
        let mut s = GrammarState::new(c).unwrap();
        s.advance_str(&"(".repeat(40_000)).unwrap();
        drop(s);
    }

    #[test]
    fn left_recursion_rejected() {
        assert!(matches!(
            Grammar::from_gbnf(r#"root ::= root "a" | "b""#, "root"),
            Err(GrammarError::LeftRecursion(_))
        ));
        assert!(matches!(
            Grammar::from_gbnf("root ::= a \"x\"\na ::= b \"z\"\nb ::= a | \"y\"", "root"),
            Err(GrammarError::LeftRecursion(_))
        ));
        // a pure tail cycle consumes nothing but cannot grow the stack, so it is merely redundant
        let tail = g("root ::= a \"x\"\na ::= b\nb ::= a | \"y\"");
        assert!(tail.matches("yx").unwrap() && !tail.matches("x").unwrap());
        // hidden behind a nullable prefix
        assert!(matches!(
            Grammar::from_gbnf("root ::= n root \"a\" | \"b\"\nn ::= \"\"", "root"),
            Err(GrammarError::LeftRecursion(_))
        ));
        // right recursion is fine
        assert!(Grammar::from_gbnf("root ::= \"a\" root | \"b\"", "root").is_ok());
    }

    #[test]
    fn errors() {
        assert!(matches!(
            Grammar::from_gbnf("root ::= missing", "root"),
            Err(GrammarError::Undefined(_))
        ));
        assert!(matches!(
            Grammar::from_gbnf("x ::= \"a\"", "root"),
            Err(GrammarError::NoRoot(_))
        ));
        assert!(matches!(
            Grammar::from_gbnf("root ::= \"a", "root"),
            Err(GrammarError::Syntax { .. })
        ));
        assert!(matches!(
            Grammar::from_gbnf("root ::= [z-a]", "root"),
            Err(GrammarError::Syntax { .. })
        ));
        assert!(matches!(
            Grammar::from_gbnf("root ::= (\"a\"", "root"),
            Err(GrammarError::Syntax { .. })
        ));
        assert!(Grammar::from_gbnf("root ::= root2\nroot2 ::= root2 \"a\"", "root").is_err());
        assert!(Grammar::from_gbnf("root ::= \"a\"{3,2}", "root").is_err());
        assert!(Grammar::from_gbnf("# comment\nroot ::= \"a\" # tail\n", "root").is_ok());
    }

    #[test]
    fn hostile_grammars_fail_fast() {
        timed(|| assert!(Grammar::from_gbnf(r#"root ::= "a"{1000000}"#, "root").is_err()));
        timed(|| assert!(Grammar::from_gbnf(r#"root ::= "a"{0,100000000000}"#, "root").is_err()));
        // nested counts share one rule per level, so this stays small and legal
        timed(|| assert!(Grammar::from_gbnf(r#"root ::= ("a"{1000}){1000}"#, "root").is_ok()));
        timed(|| {
            assert!(Grammar::from_gbnf(r#"root ::= (("a"{1000}){1000}){1000}"#, "root").is_ok())
        });
        timed(|| {
            assert!(
                Grammar::from_gbnf(
                    r#"root ::= "a"{1000} "b"{1000} "c"{1000} ("d"{1000} ){1000} [a-z]{400000}"#,
                    "root"
                )
                .is_err()
            )
        });
        timed(|| {
            assert!(
                Grammar::from_gbnf(
                    &format!(
                        "root ::= {}\"a\"{}",
                        "(".repeat(100_000),
                        ")".repeat(100_000)
                    ),
                    "root"
                )
                .is_err()
            )
        });
        timed(|| {
            assert!(
                Grammar::from_regex(&format!("{}a{}", "(".repeat(100_000), ")".repeat(100_000)))
                    .is_err()
            )
        });
        timed(|| {
            let _ = Grammar::from_regex("((a{1000}){1000}){1000}");
        });
        timed(|| assert!(Grammar::from_regex("[a-z]{500000}").is_err()));
        timed(|| {
            let _ = Grammar::from_regex("a{99999999999}");
        }); // too large to be a count: read as literal text
        timed(|| assert!(Grammar::from_gbnf(&"x".repeat(MAX_SRC + 1), "root").is_err()));
        // exponential-looking but dedup keeps it small
        let amb = g("root ::= x*\nx ::= \"a\" | \"a\" \"a\" | \"a\" \"a\" \"a\"");
        timed(|| assert!(amb.matches(&"a".repeat(300)).unwrap()));
        let amb2 = g(
            "root ::= (a | b)*\na ::= \"x\" a2\nb ::= \"x\" b2\na2 ::= \"y\" | \"\"\nb2 ::= \"y\" | \"\"",
        );
        timed(|| assert!(amb2.matches(&"xy".repeat(500)).unwrap()));
        // a grammar whose live stacks multiply is cut off with an error, not a hang
        let blow = Grammar::from_gbnf(
            "root ::= s{12}\ns ::= \"a\"? \"a\"? \"a\"? \"a\"? \"a\"? \"a\"?",
            "root",
        );
        if let Ok(b) = blow {
            timed(|| {
                let _ = Arc::new(b).matches(&"a".repeat(80));
            });
        }
    }

    #[test]
    fn regex_grammars() {
        let r = Arc::new(Grammar::from_regex(r"[a-z]+@[a-z]+\.(com|org)").unwrap());
        assert!(r.matches("bob@site.com").unwrap() && r.matches("a@b.org").unwrap());
        assert!(!r.matches("bob@site.net").unwrap() && !r.matches("@x.com").unwrap());
        let d = Arc::new(Grammar::from_regex(r"\d{3}-\d{2,4}").unwrap());
        assert!(d.matches("123-45").unwrap() && d.matches("123-4567").unwrap());
        assert!(
            !d.matches("123-4").unwrap()
                && !d.matches("12-45").unwrap()
                && !d.matches("123-45678").unwrap()
        );
        let n = Arc::new(Grammar::from_regex(r"[^aeiou\s]+").unwrap());
        assert!(n.matches("xyz").unwrap() && !n.matches("xay").unwrap());
        let u = Arc::new(Grammar::from_regex("h.llo|caf\u{e9}").unwrap());
        assert!(u.matches("h\u{e9}llo").unwrap() && u.matches("caf\u{e9}").unwrap());
    }

    #[test]
    fn unicode_and_chunking() {
        let a = g(r#"root ::= [^x]* "€" ."#);
        let text = "héllo wörld€𝄞";
        let mut whole = GrammarState::new(a.clone()).unwrap();
        assert!(whole.advance_str(text).unwrap() && whole.is_complete());
        for chunk in 1..=5 {
            let mut s = GrammarState::new(a.clone()).unwrap();
            for c in text.as_bytes().chunks(chunk) {
                assert!(s.advance_bytes(c).unwrap());
            }
            assert!(s.is_complete(), "chunk {chunk}");
        }
        // incomplete character is not a complete sentence
        let mut s = GrammarState::new(a.clone()).unwrap();
        s.advance_str("€").unwrap();
        s.advance_byte(0xf0).unwrap();
        assert!(!s.is_complete());
        // invalid utf-8 kills the state
        for bad in [
            &[0xc0u8, 0x80][..],
            &[0xed, 0xa0, 0x80],
            &[0xf5],
            &[0x80],
            &[0xe0, 0x80, 0x80],
        ] {
            let mut s = GrammarState::new(a.clone()).unwrap();
            assert!(!s.advance_bytes(bad).unwrap(), "{bad:?}");
            assert!(s.is_dead());
        }
    }

    // ---- JSON schema

    fn valid(s: &Value, v: &Value) -> bool {
        if let Some(e) = s.get("enum") {
            return e.as_array().unwrap().contains(v);
        }
        if let Some(c) = s.get("const") {
            return c == v;
        }
        match s.get("type").and_then(Value::as_str) {
            Some("object") => {
                let Some(o) = v.as_object() else { return false };
                let props = s.get("properties").and_then(Value::as_object);
                if let Some(Value::Array(r)) = s.get("required") {
                    if !r.iter().all(|k| o.contains_key(k.as_str().unwrap())) {
                        return false;
                    }
                }
                o.iter()
                    .all(|(k, x)| props.and_then(|p| p.get(k)).is_some_and(|ps| valid(ps, x)))
            }
            Some("array") => {
                let Some(a) = v.as_array() else { return false };
                a.len()
                    <= s.get("maxItems")
                        .and_then(Value::as_u64)
                        .unwrap_or(u64::MAX) as usize
                    && a.len() >= s.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize
                    && a.iter().all(|x| s.get("items").is_none_or(|i| valid(i, x)))
            }
            Some("string") => {
                let Some(t) = v.as_str() else { return false };
                let n = t.chars().count() as u64;
                n >= s.get("minLength").and_then(Value::as_u64).unwrap_or(0)
                    && n <= s
                        .get("maxLength")
                        .and_then(Value::as_u64)
                        .unwrap_or(u64::MAX)
            }
            Some("integer") => v.as_f64().is_some_and(|f| {
                f.fract() == 0.0
                    && f >= s.get("minimum").and_then(Value::as_f64).unwrap_or(f64::MIN)
            }),
            Some("boolean") => v.is_boolean(),
            _ => true,
        }
    }

    /// Walk the grammar with random allowed characters until it completes.
    fn random_doc(gr: &Arc<Grammar>, rng: &mut Rng) -> Option<String> {
        let mut s = GrammarState::new(gr.clone()).unwrap();
        let mut out = String::new();
        for _ in 0..4000 {
            if s.is_complete() && rng.below(3) == 0 {
                return Some(out);
            }
            let cs = s.sample_next_chars(24);
            if cs.is_empty() {
                return if s.is_complete() { Some(out) } else { None };
            }
            let c = cs[rng.below(cs.len())];
            assert!(
                s.advance_char(c).unwrap(),
                "representative {c:?} rejected after {out:?}"
            );
            out.push(c);
        }
        if s.is_complete() { Some(out) } else { None }
    }

    fn test_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 8},
                "age": {"type": "integer", "minimum": 0},
                "tags": {"type": "array", "items": {"type": "string", "maxLength": 3}, "maxItems": 3},
                "kind": {"enum": ["a", "b", null]},
                "inner": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
            },
            "required": ["name", "age"]
        })
    }

    #[test]
    fn schema_random_walk_validates() {
        let sc = test_schema();
        for ws in [
            Whitespace::Compact,
            Whitespace::Single,
            Whitespace::Flexible,
        ] {
            let gr = schema(&sc, ws);
            let mut rng = Rng(42 + ws as u64);
            let mut done = 0;
            for _ in 0..60 {
                if let Some(doc) = random_doc(&gr, &mut rng) {
                    let v: Value =
                        serde_json::from_str(&doc).unwrap_or_else(|e| panic!("{e}: {doc}"));
                    assert!(valid(&sc, &v), "invalid for schema: {doc}");
                    assert!(gr.matches(&doc).unwrap());
                    done += 1;
                }
            }
            assert!(done >= 30, "only {done} walks finished for {ws:?}");
        }
    }

    #[test]
    fn schema_required_optional_order() {
        let sc = json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}},"required":["a","c"]});
        let gr = schema(&sc, Whitespace::Compact);
        for ok in [r#"{"a":1,"c":2}"#, r#"{"a":1,"b":2,"c":3}"#] {
            assert!(gr.matches(ok).unwrap(), "{ok}");
        }
        for bad in [
            r#"{"c":2,"a":1}"#,
            r#"{"a":1}"#,
            r#"{"a":1,"c":2,"b":3}"#,
            r#"{"a":1,"d":2,"c":3}"#,
            r#"{"a":1,"c":2,}"#,
            r#"{"a":-1,"c":2.5}"#,
        ] {
            assert!(!gr.matches(bad).unwrap(), "{bad}");
        }
        let all_opt =
            json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}});
        let go = schema(&all_opt, Whitespace::Compact);
        for ok in ["{}", r#"{"a":1}"#, r#"{"b":1}"#, r#"{"a":1,"b":2}"#] {
            assert!(go.matches(ok).unwrap(), "{ok}");
        }
        assert!(!go.matches(r#"{,"b":1}"#).unwrap() && !go.matches(r#"{"a":1,}"#).unwrap());
        // additionalProperties
        let extra = json!({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":{"type":"boolean"}});
        let ge = schema(&extra, Whitespace::Compact);
        assert!(
            ge.matches(r#"{"a":1,"x":true,"y":false}"#).unwrap()
                && !ge.matches(r#"{"a":1,"x":3}"#).unwrap()
        );
    }

    #[test]
    fn schema_features() {
        let c = Whitespace::Compact;
        let any = schema(
            &json!({"anyOf":[{"type":"integer"},{"type":"string","pattern":"^[a-c]{2}$"},{"const":[1,2]}]}),
            c,
        );
        for ok in ["5", "-3", r#""ab""#, "[1,2]"] {
            assert!(any.matches(ok).unwrap(), "{ok}");
        }
        for bad in [r#""abc""#, r#""ad""#, "[1]", "5.5", "01"] {
            assert!(!any.matches(bad).unwrap(), "{bad}");
        }
        let nl = schema(&json!({"type":["string","null"]}), c);
        assert!(
            nl.matches("null").unwrap()
                && nl.matches(r#""x\né""#).unwrap()
                && !nl.matches("1").unwrap()
        );
        let nb = schema(&json!({"type":"integer","nullable":true}), c);
        assert!(nb.matches("null").unwrap() && nb.matches("7").unwrap());
        let arr = schema(
            &json!({"type":"array","items":{"type":"boolean"},"minItems":2,"maxItems":3}),
            c,
        );
        assert!(arr.matches("[true,false]").unwrap() && arr.matches("[true,false,true]").unwrap());
        assert!(
            !arr.matches("[true]").unwrap()
                && !arr.matches("[true,true,true,true]").unwrap()
                && !arr.matches("[]").unwrap()
        );
        let tup = schema(
            &json!({"type":"array","prefixItems":[{"type":"integer"},{"type":"string"}]}),
            c,
        );
        assert!(tup.matches(r#"[1,"x"]"#).unwrap() && !tup.matches(r#"["x",1]"#).unwrap());
        let fmt = schema(&json!({"type":"string","format":"date"}), c);
        assert!(
            fmt.matches(r#""2026-11-24""#).unwrap() && !fmt.matches(r#""2026-13-24""#).unwrap()
        );
        let dt = schema(&json!({"type":"string","format":"date-time"}), c);
        assert!(dt.matches(r#""2026-11-24T10:20:30Z""#).unwrap());
        let uu = schema(&json!({"type":"string","format":"uuid"}), c);
        assert!(
            uu.matches(r#""123e4567-e89b-12d3-a456-426614174000""#)
                .unwrap()
        );
        let ml = schema(&json!({"type":"string","minLength":2,"maxLength":3}), c);
        assert!(
            ml.matches(r#""ab""#).unwrap()
                && ml.matches(r#""é€x""#).unwrap()
                && !ml.matches(r#""a""#).unwrap()
                && !ml.matches(r#""abcd""#).unwrap()
        );
        let all = schema(
            &json!({"allOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]},{"properties":{"b":{"type":"boolean"}},"required":["b"]}]}),
            c,
        );
        assert!(all.matches(r#"{"a":1,"b":true}"#).unwrap() && !all.matches(r#"{"a":1}"#).unwrap());
        let un = schema(&json!({"type":"integer","minimum":0}), c);
        assert!(un.matches("12").unwrap() && !un.matches("-1").unwrap());
        assert!(Grammar::from_json_schema(&json!(false), &SchemaOptions::default()).is_err());
        assert!(
            Grammar::from_json_schema(&json!({"type":"wat"}), &SchemaOptions::default()).is_err()
        );
        assert!(
            Grammar::from_json_schema(&json!({"$ref":"http://x/y"}), &SchemaOptions::default())
                .is_err()
        );
        assert!(
            Grammar::from_json_schema(&json!({"$ref":"#/nope"}), &SchemaOptions::default())
                .is_err()
        );
    }

    #[test]
    fn schema_refs_and_recursion() {
        let sc = json!({
            "$defs": {"node": {"type":"object","properties":{"id":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["id","kids"]}},
            "$ref": "#/$defs/node"
        });
        let gr = schema(&sc, Whitespace::Compact);
        assert!(
            gr.matches(
                r#"{"id":1,"kids":[{"id":2,"kids":[]},{"id":3,"kids":[{"id":4,"kids":[]}]}]}"#
            )
            .unwrap()
        );
        assert!(!gr.matches(r#"{"id":1,"kids":[{"id":2}]}"#).unwrap());
        let root_ref = json!({"type":"object","properties":{"next":{"$ref":"#"}}});
        let gr = schema(&root_ref, Whitespace::Compact);
        assert!(gr.matches(r#"{"next":{"next":{}}}"#).unwrap());
        // a reference cycle without any consumption is rejected
        assert!(Grammar::from_json_schema(&json!({"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"}), &SchemaOptions::default()).is_err());
    }

    #[test]
    fn json_object_and_response_format() {
        let o = SchemaOptions {
            whitespace: Whitespace::Flexible,
            ..Default::default()
        };
        let gr = Arc::new(Grammar::json_object(&o).unwrap());
        for ok in [
            r#"{"a":1,"b":[true,null,"x",{"c":-2.5e3}]}"#,
            "{ }",
            "{\n  \"a\": 1\n}",
        ] {
            assert!(gr.matches(ok).unwrap(), "{ok}");
        }
        for bad in ["[1]", r#"{"a":}"#, r#"{"a":1,}"#, "{'a':1}", "{\"a\":01}"] {
            assert!(!gr.matches(bad).unwrap(), "{bad}");
        }
        assert!(
            response_format_to_grammar(&json!({"type":"text"}), &o)
                .unwrap()
                .is_none()
        );
        assert!(
            response_format_to_grammar(&json!({"type":"json_object"}), &o)
                .unwrap()
                .is_some()
        );
        let js = response_format_to_grammar(&json!({"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}}}), &SchemaOptions { whitespace: Whitespace::Compact, ..o.clone() }).unwrap().unwrap();
        assert!(Arc::new(js).matches(r#"{"n":3}"#).unwrap());
        let gb =
            response_format_to_grammar(&json!({"type":"grammar","grammar":"root ::= \"hi\""}), &o)
                .unwrap()
                .unwrap();
        assert!(Arc::new(gb).matches("hi").unwrap());
        assert!(response_format_to_grammar(&json!({"type":"weird"}), &o).is_err());
    }

    // ---- token masks

    fn toy_vocab() -> (Vec<Vec<u8>>, u32) {
        let mut v: Vec<Vec<u8>> = (0u8..128).map(|b| vec![b]).collect();
        for s in [
            "{\"", "\":", "\",", "true", "false", "null", "\"name\"", "   ", " \"", "},", "\":\"",
            "123", "0.5", "\u{e9}", "\u{20ac}", "[{", "}]", "\"a\"", "ab", "abc",
        ] {
            v.push(s.as_bytes().to_vec());
        }
        for b in [
            &[0xc3u8][..],
            &[0xa9],
            &[0xe2],
            &[0x82],
            &[0xac],
            &[0xe2, 0x82],
            &[0x82, 0xac],
            &[0xc0, 0x80],
            &[0xed, 0xa0],
            &[0xf4, 0x90],
            &[0xf0, 0x9f],
            &[0x9f, 0x98, 0x80],
        ] {
            v.push(b.to_vec());
        }
        v.push(vec![]); // a special token
        let eos = v.len() as u32;
        v.push(vec![]);
        (v, eos)
    }

    #[test]
    fn partial_utf8_tokens() {
        let (toks, eos) = toy_vocab();
        let trie = Arc::new(TokenTrie::from_token_bytes(&toks, Some(eos)));
        let id = |b: &[u8]| toks.iter().position(|t| t == b).unwrap() as u32;
        let gr = g(r#"root ::= "é" | "€""#);
        let mut c = Constraint::new(gr, trie.clone()).unwrap();
        let m = c.mask().clone();
        for ok in [
            &[0xc3u8][..],
            &[0xe2],
            &[0xe2, 0x82],
            "é".as_bytes(),
            "€".as_bytes(),
        ] {
            assert!(m.get(id(ok) as usize), "{ok:?}");
        }
        for no in [
            &[0xa9u8][..],
            &[0x82],
            &[0xac],
            &[0xc0, 0x80],
            &[0xed, 0xa0],
            &[0xf4, 0x90],
            &[0xf0, 0x9f],
            b"a",
        ] {
            assert!(!m.get(id(no) as usize), "{no:?}");
        }
        assert!(!m.get(eos as usize));
        assert!(c.accept(id(&[0xe2])).unwrap());
        let m = c.mask().clone();
        assert!(m.get(id(&[0x82, 0xac]) as usize) && m.get(id(&[0x82]) as usize));
        assert!(!m.get(id(&[0xa9]) as usize) && !m.get(id(&[0xac]) as usize));
        assert!(!c.is_complete());
        assert!(c.accept(id(&[0x82, 0xac])).unwrap());
        assert!(c.is_complete());
        assert!(c.mask().get(eos as usize) && c.mask().count() == 1);
        assert!(c.accept(eos).unwrap() && c.is_finished());
    }

    #[test]
    fn mask_equals_brute_force() {
        let (toks, eos) = toy_vocab();
        let trie = Arc::new(TokenTrie::from_token_bytes(&toks, Some(eos)));
        let grammars = [
            schema(&test_schema(), Whitespace::Flexible),
            g(ARITH),
            Arc::new(Grammar::json_object(&SchemaOptions::default()).unwrap()),
        ];
        for (gi, gr) in grammars.iter().enumerate() {
            for seed in 0..6u64 {
                let mut rng = Rng(seed * 31 + gi as u64);
                let mut c = Constraint::new(gr.clone(), trie.clone()).unwrap();
                for step in 0..40 {
                    let mask = c.mask().clone();
                    let mut oracle = BitSet::new(toks.len());
                    for id in 0..toks.len() as u32 {
                        let mut t = c.state().clone();
                        if t.advance_token(&trie, id).unwrap() {
                            oracle.set(id as usize);
                        }
                    }
                    if c.is_complete() {
                        oracle.set(eos as usize);
                    }
                    assert_eq!(mask, oracle, "grammar {gi} seed {seed} step {step}");
                    let allowed: Vec<u32> = mask.iter().collect();
                    if allowed.is_empty() {
                        break;
                    }
                    let t = allowed[rng.below(allowed.len())];
                    assert!(c.accept(t).unwrap());
                    if c.is_finished() {
                        break;
                    }
                }
            }
        }
    }

    #[test]
    fn constraint_rejects_and_finishes() {
        let (toks, eos) = toy_vocab();
        let trie = Arc::new(TokenTrie::from_token_bytes(&toks, Some(eos)));
        let id = |s: &str| toks.iter().position(|t| t == s.as_bytes()).unwrap() as u32;
        let gr = schema(
            &json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}),
            Whitespace::Compact,
        );
        let mut c = Constraint::new(gr, trie).unwrap();
        assert!(!c.accept(id("a")).unwrap());
        assert!(!c.accept(eos).unwrap());
        for t in ["{\"", "o", "k", "\":", "true", "}"] {
            assert!(c.accept(id(t)).unwrap(), "{t}");
        }
        assert!(c.is_complete());
        assert!(c.accept(eos).unwrap());
        assert!(!c.accept(id("a")).unwrap());
        c.reset().unwrap();
        assert!(!c.is_complete() && !c.is_finished());
        let mut logits = vec![0.0f32; toks.len()];
        c.apply(&mut logits);
        assert!(logits[id("{\"") as usize] == 0.0 && logits[id("a") as usize] == f32::NEG_INFINITY);
    }

    #[test]
    fn trie_from_tokenizer_variants_and_bytes() {
        let (toks, eos) = toy_vocab();
        let trie = TokenTrie::from_token_bytes(&toks, Some(eos));
        assert_eq!(trie.vocab_size(), toks.len());
        assert_eq!(trie.token_bytes(eos), b"");
        assert_eq!(trie.token_bytes(b'a' as u32), b"a");
        assert_eq!(trie.token_bytes(u32::MAX), b"");
        let inv = gpt2_inverse();
        assert_eq!(inv.len(), 256);
        assert_eq!(inv[&'\u{120}'], b' ');
        assert_eq!(inv[&'a'], b'a');
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    fn trie() -> TokenTrie {
        let mut toks: Vec<Vec<u8>> = (0u8..=127).map(|b| vec![b]).collect();
        toks.push(b"ab".to_vec());
        TokenTrie::from_token_bytes(&toks, None)
    }

    #[test]
    fn mask_budget_returns_typed_error() {
        let g = Arc::new(Grammar::from_gbnf("root ::= [a-z]+", "root").unwrap());
        let st = GrammarState::new(g).unwrap();
        let t = trie();
        let mut c = StepCache::new();
        let tiny = MaskBudget {
            max_ops: 3,
            max_time: None,
        };
        assert_eq!(
            st.try_allowed_tokens(&t, &mut c, tiny).unwrap_err(),
            GrammarError::MaskBudget
        );
        assert!(
            st.try_allowed_tokens(&t, &mut c, MaskBudget::default())
                .is_ok()
        );
        // already-expired clock
        let zero = MaskBudget {
            max_ops: u64::MAX,
            max_time: Some(std::time::Duration::ZERO),
        };
        let _ = st.try_allowed_tokens(&t, &mut c, zero); // must terminate either way
    }

    #[test]
    fn constraint_reports_budget_hit() {
        let g = Arc::new(Grammar::from_gbnf("root ::= [a-z]+", "root").unwrap());
        let mut c = Constraint::new(g, Arc::new(trie())).unwrap();
        c.set_budget(MaskBudget {
            max_ops: 2,
            max_time: None,
        });
        assert!(matches!(c.try_mask(), Err(GrammarError::MaskBudget)));
        let _ = c.mask();
        assert!(c.budget_hit());
    }

    #[test]
    fn step_cache_is_bounded_by_bytes() {
        let g = Arc::new(Grammar::from_gbnf("root ::= [a-z]+", "root").unwrap());
        let st = GrammarState::new(g).unwrap();
        let t = trie();
        let mut c = StepCache::with_limit(4 * STEP_ENTRY_BASE);
        for _ in 0..3 {
            let _ = st.allowed_tokens_cached(&t, &mut c);
            assert!(
                c.bytes() <= 4 * STEP_ENTRY_BASE + 4096,
                "bytes {}",
                c.bytes()
            );
        }
        assert!(c.len() < 40);
    }

    #[test]
    fn grammar_cache_is_bounded_by_size() {
        let mut cache = GrammarCache::with_limit(4000);
        for i in 0..50 {
            let g = Arc::new(Grammar::from_gbnf(&format!("root ::= \"x{i}\""), "root").unwrap());
            cache.insert(&format!("k{i}"), g);
            assert!(cache.bytes() <= 4000);
        }
        assert!(cache.len() < 50 && cache.get("k49").is_some() && cache.get("k0").is_none());
        let big = Arc::new(Grammar::from_gbnf("root ::= \"y\"", "root").unwrap());
        let mut tiny = GrammarCache::with_limit(10);
        tiny.insert("a", big);
        assert!(tiny.is_empty());
    }
}
