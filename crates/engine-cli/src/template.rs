//! Chat templates: a small, safe subset of Jinja, plus built-in fallbacks.
//!
//! Supported: `{{ }}`, `{% %}`, `{# #}` with `-` whitespace control and Hugging Face's
//! `trim_blocks` / `lstrip_blocks` behaviour; `for` (with `else`, `if`, `loop.*`, `break`,
//! `continue`), `if`/`elif`/`else`, `set` (names and `ns.attr`), `generation` blocks (transparent);
//! expressions with `~ + - * // %`, comparisons, `in`, `not in`, `is [not] defined|none|string|…`,
//! `and`/`or`/`not`, conditional expressions, indexing, slicing, attributes, list and dict literals;
//! filters and string methods listed in [`apply_filter`] / [`call_method`]; globals `raise_exception`,
//! `namespace`, `range`, `strftime_now`.
//!
//! Safety: there is no way to reach files, the network or Rust internals; evaluation is bounded by a
//! step limit, an output limit and a nesting limit, and every failure is an `Err`, never a panic.
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::rc::Rc;

pub const MAX_STEPS: usize = 300_000;
pub const MAX_OUTPUT: usize = 4 << 20;
const MAX_DEPTH: usize = 48;
const MAX_TEMPLATE: usize = 1 << 20;

#[derive(Debug, Clone, PartialEq)]
pub enum TemplateError {
    Syntax(String),
    Runtime(String),
    /// `raise_exception(...)` in the template — typically "roles must alternate".
    Raised(String),
    Limit(&'static str),
}
impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemplateError::Syntax(s) => write!(f, "chat template syntax error: {s}"),
            TemplateError::Runtime(s) => write!(f, "chat template error: {s}"),
            TemplateError::Raised(s) => write!(f, "{s}"),
            TemplateError::Limit(s) => write!(f, "chat template exceeded its {s} limit"),
        }
    }
}
impl std::error::Error for TemplateError {}
type R<T> = Result<T, TemplateError>;
fn syn<T>(s: impl Into<String>) -> R<T> { Err(TemplateError::Syntax(s.into())) }
fn rt<T>(s: impl Into<String>) -> R<T> { Err(TemplateError::Runtime(s.into())) }

// ------------------------------------------------------------------------------ values

/// Insertion-ordered string map (Python dicts keep order, and `tojson` output must match it).
#[derive(Clone, Debug, Default)]
pub struct OMap(Vec<(String, V)>);
impl OMap {
    pub fn len(&self) -> usize { self.0.len() }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    pub fn get(&self, k: &str) -> Option<&V> { self.0.iter().find(|(n, _)| n == k).map(|(_, v)| v) }
    pub fn contains_key(&self, k: &str) -> bool { self.get(k).is_some() }
    pub fn insert(&mut self, k: String, v: V) {
        match self.0.iter_mut().find(|(n, _)| *n == k) { Some(e) => e.1 = v, None => self.0.push((k, v)) }
    }
    pub fn keys(&self) -> impl Iterator<Item = &String> { self.0.iter().map(|(k, _)| k) }
    pub fn values(&self) -> impl Iterator<Item = &V> { self.0.iter().map(|(_, v)| v) }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> { self.0.iter().map(|(k, v)| (k, v)) }
}
impl FromIterator<(String, V)> for OMap {
    fn from_iter<I: IntoIterator<Item = (String, V)>>(it: I) -> OMap { let mut m = OMap::default(); for (k, v) in it { m.insert(k, v); } m }
}

#[derive(Clone, Debug)]
pub enum V {
    None,
    Undefined,
    Bool(bool),
    Int(i64),
    Str(String),
    List(Rc<RefCell<Vec<V>>>),
    Map(Rc<RefCell<OMap>>),
}
impl V {
    pub fn str(s: impl Into<String>) -> V { V::Str(s.into()) }
    pub fn list(v: Vec<V>) -> V { V::List(Rc::new(RefCell::new(v))) }
    pub fn map(m: OMap) -> V { V::Map(Rc::new(RefCell::new(m))) }
    fn truthy(&self) -> bool {
        match self {
            V::None | V::Undefined => false,
            V::Bool(b) => *b,
            V::Int(i) => *i != 0,
            V::Str(s) => !s.is_empty(),
            V::List(l) => !l.borrow().is_empty(),
            V::Map(m) => !m.borrow().is_empty(),
        }
    }
    fn kind(&self) -> &'static str {
        match self { V::None => "none", V::Undefined => "undefined", V::Bool(_) => "bool", V::Int(_) => "int", V::Str(_) => "string", V::List(_) => "list", V::Map(_) => "mapping" }
    }
    fn render(&self) -> String {
        match self {
            V::None => "None".into(),
            V::Undefined => String::new(),
            V::Bool(true) => "True".into(),
            V::Bool(false) => "False".into(),
            V::Int(i) => i.to_string(),
            V::Str(s) => s.clone(),
            other => to_json(other, None, 0),
        }
    }
    fn eq(&self, o: &V) -> bool {
        match (self, o) {
            (V::None, V::None) | (V::Undefined, V::Undefined) | (V::None, V::Undefined) | (V::Undefined, V::None) => true,
            (V::Bool(a), V::Bool(b)) => a == b,
            (V::Int(a), V::Int(b)) => a == b,
            (V::Bool(a), V::Int(b)) | (V::Int(b), V::Bool(a)) => (*a as i64) == *b,
            (V::Str(a), V::Str(b)) => a == b,
            (V::List(a), V::List(b)) => { let (a, b) = (a.borrow(), b.borrow()); a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.eq(y)) }
            (V::Map(a), V::Map(b)) => { let (a, b) = (a.borrow(), b.borrow()); a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|w| v.eq(w))) }
            _ => false,
        }
    }
    /// Build a value from JSON (used for messages and tool definitions).
    pub fn from_json(j: &serde_json::Value) -> V {
        use serde_json::Value as J;
        match j {
            J::Null => V::None,
            J::Bool(b) => V::Bool(*b),
            J::Number(n) => n.as_i64().map_or_else(|| V::Str(n.to_string()), V::Int),
            J::String(s) => V::Str(s.clone()),
            J::Array(a) => V::list(a.iter().map(V::from_json).collect()),
            J::Object(o) => V::map(o.iter().map(|(k, v)| (k.clone(), V::from_json(v))).collect()),
        }
    }
}

/// Python-style `json.dumps(..., ensure_ascii=False)`, which is what chat templates expect from `tojson`.
fn to_json(v: &V, indent: Option<usize>, level: usize) -> String {
    fn esc(s: &str) -> String {
        let mut o = String::from('"');
        for c in s.chars() {
            match c {
                '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"),
                c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                c => o.push(c),
            }
        }
        o.push('"');
        o
    }
    let nl = |lvl: usize| indent.map_or(String::new(), |n| format!("\n{}", " ".repeat(n * lvl)));
    match v {
        V::None | V::Undefined => "null".into(),
        V::Bool(b) => b.to_string(),
        V::Int(i) => i.to_string(),
        V::Str(s) => esc(s),
        V::List(l) => {
            let l = l.borrow();
            if l.is_empty() { return "[]".into(); }
            let items: Vec<String> = l.iter().map(|x| format!("{}{}", nl(level + 1), to_json(x, indent, level + 1))).collect();
            format!("[{}{}]", items.join(if indent.is_some() { "," } else { ", " }), nl(level))
        }
        V::Map(m) => {
            let m = m.borrow();
            if m.is_empty() { return "{}".into(); }
            let items: Vec<String> = m.iter().map(|(k, x)| format!("{}{}: {}", nl(level + 1), esc(k), to_json(x, indent, level + 1))).collect();
            format!("{{{}{}}}", items.join(if indent.is_some() { "," } else { ", " }), nl(level))
        }
    }
}

// ------------------------------------------------------------------------------ lexer (template level)

#[derive(Debug, Clone)]
enum Piece { Text(String), Expr(String), Stmt(String) }

fn find_close(src: &str, from: usize, closer: &str) -> Option<usize> {
    let b = src.as_bytes();
    let c = closer.as_bytes();
    let (mut i, mut quote) = (from, 0u8);
    while i < b.len() {
        let ch = b[i];
        if quote != 0 {
            if ch == b'\\' { i += 2; continue; }
            if ch == quote { quote = 0; }
        } else if ch == b'\'' || ch == b'"' { quote = ch; }
        else if b[i..].starts_with(c) { return Some(i); }
        i += 1;
    }
    None
}

fn lex(src: &str) -> R<Vec<Piece>> {
    if src.len() > MAX_TEMPLATE { return Err(TemplateError::Limit("template size")); }
    let mut out = vec![];
    let (mut i, mut at_line_start) = (0usize, true);
    loop {
        let next = ["{{", "{%", "{#"].iter().filter_map(|o| src[i..].find(o).map(|p| (p + i, *o))).min_by_key(|x| x.0);
        let Some((pos, open)) = next else {
            if i < src.len() { out.push(Piece::Text(src[i..].to_string())); }
            return Ok(out);
        };
        let mut text = src[i..pos].to_string();
        let mut j = pos + 2;
        let dash_l = src[j..].starts_with('-');
        if dash_l { j += 1; text.truncate(text.trim_end().len()); }
        else if open != "{{" {
            // lstrip_blocks: spaces and tabs before a block tag that starts a line
            let stripped = text.trim_end_matches([' ', '\t']);
            let at_ls = if stripped.is_empty() { at_line_start } else { stripped.ends_with('\n') };
            if at_ls { text.truncate(stripped.len()); }
        }
        if !text.is_empty() { out.push(Piece::Text(text)); }
        let closer = match open { "{{" => "}}", "{%" => "%}", _ => "#}" };
        let Some(end) = find_close(src, j, closer) else { return syn(format!("unterminated `{open}`")); };
        let mut inner = &src[j..end];
        let dash_r = inner.ends_with('-');
        if dash_r { inner = &inner[..inner.len() - 1]; }
        match open { "{{" => out.push(Piece::Expr(inner.to_string())), "{%" => out.push(Piece::Stmt(inner.to_string())), _ => {} }
        i = end + 2;
        at_line_start = false;
        if dash_r {
            let rest = &src[i..];
            i += rest.len() - rest.trim_start().len();
        } else if open != "{{" {
            if src[i..].starts_with("\r\n") { i += 2; at_line_start = true; }
            else if src[i..].starts_with('\n') { i += 1; at_line_start = true; }
        }
        if src[..i].ends_with('\n') { at_line_start = true; }
    }
}

// ------------------------------------------------------------------------------ expression lexer

#[derive(Debug, Clone, PartialEq)]
enum Tok { Name(String), Int(i64), Str(String), Op(&'static str) }

fn unescape(s: &str) -> String {
    let mut o = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' { o.push(c); continue; }
        match it.next() {
            Some('n') => o.push('\n'), Some('t') => o.push('\t'), Some('r') => o.push('\r'),
            Some('\\') => o.push('\\'), Some('\'') => o.push('\''), Some('"') => o.push('"'),
            Some('u') => {
                let h: String = it.clone().take(4).collect();
                match u32::from_str_radix(&h, 16).ok().and_then(char::from_u32) {
                    Some(ch) if h.len() == 4 => { o.push(ch); for _ in 0..4 { it.next(); } }
                    _ => o.push_str("\\u"),
                }
            }
            Some(x) => { o.push('\\'); o.push(x); }
            None => o.push('\\'),
        }
    }
    o
}

fn toks(s: &str) -> R<Vec<Tok>> {
    const OPS: [&str; 22] = ["==", "!=", "<=", ">=", "//", "**", "~", "+", "-", "*", "/", "%", "<", ">", "=", "|", ".", ",", ":", "(", ")", "["];
    const OPS2: [&str; 3] = ["]", "{", "}"];
    let b = s.as_bytes();
    let (mut i, mut out) = (0, vec![]);
    'outer: while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() { i += 1; continue; }
        if c == b'\'' || c == b'"' {
            let mut j = i + 1;
            while j < b.len() && b[j] != c { if b[j] == b'\\' { j += 1; } j += 1; }
            if j >= b.len() { return syn("unterminated string"); }
            out.push(Tok::Str(unescape(&s[i + 1..j])));
            i = j + 1;
            continue;
        }
        if c.is_ascii_digit() {
            let mut j = i;
            while j < b.len() && b[j].is_ascii_digit() { j += 1; }
            if j < b.len() && b[j] == b'.' && b.get(j + 1).is_some_and(u8::is_ascii_digit) { return syn("floating point numbers are not supported"); }
            out.push(Tok::Int(s[i..j].parse().map_err(|_| TemplateError::Syntax("number too large".into()))?));
            i = j;
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let mut j = i;
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') { j += 1; }
            out.push(Tok::Name(s[i..j].to_string()));
            i = j;
            continue;
        }
        for op in OPS.iter().chain(OPS2.iter()) {
            if s[i..].starts_with(op) { out.push(Tok::Op(op)); i += op.len(); continue 'outer; }
        }
        if s[i..].starts_with('!') { return syn("unexpected `!`"); }
        return syn(format!("unexpected character {:?}", s[i..].chars().next().unwrap_or('?')));
    }
    if out.len() > 1500 { return syn("expression too long"); }
    Ok(out)
}

// ------------------------------------------------------------------------------ expression AST + parser

#[derive(Debug, Clone)]
enum Lit { None, Bool(bool), Int(i64), Str(String) }
impl Lit {
    fn val(&self) -> V { match self { Lit::None => V::None, Lit::Bool(b) => V::Bool(*b), Lit::Int(i) => V::Int(*i), Lit::Str(s) => V::Str(s.clone()) } }
}

#[derive(Debug, Clone)]
enum E {
    Lit(Lit),
    Var(String),
    List(Vec<E>),
    Dict(Vec<(E, E)>),
    Attr(Box<E>, String),
    Index(Box<E>, Box<E>),
    Slice(Box<E>, Option<Box<E>>, Option<Box<E>>),
    Call(Box<E>, Vec<E>, Vec<(String, E)>),
    Filter(Box<E>, String, Vec<E>, Vec<(String, E)>),
    Bin(&'static str, Box<E>, Box<E>),
    And(Box<E>, Box<E>),
    Or(Box<E>, Box<E>),
    Not(Box<E>),
    Neg(Box<E>),
    In(Box<E>, Box<E>, bool),
    Test(Box<E>, String, bool, Vec<E>),
    Cond(Box<E>, Box<E>, Option<Box<E>>),
}

struct P { t: Vec<Tok>, i: usize, depth: usize }
impl P {
    fn peek(&self) -> Option<&Tok> { self.t.get(self.i) }
    fn op(&self, o: &str) -> bool { matches!(self.peek(), Some(Tok::Op(x)) if *x == o) }
    fn name(&self, n: &str) -> bool { matches!(self.peek(), Some(Tok::Name(x)) if x == n) }
    fn eat_op(&mut self, o: &str) -> bool { if self.op(o) { self.i += 1; true } else { false } }
    fn eat_name(&mut self, n: &str) -> bool { if self.name(n) { self.i += 1; true } else { false } }
    fn expect_op(&mut self, o: &str) -> R<()> { if self.eat_op(o) { Ok(()) } else { syn(format!("expected `{o}`")) } }
    fn done(&self) -> bool { self.i >= self.t.len() }
    fn enter(&mut self) -> R<()> { self.depth += 1; if self.depth > MAX_DEPTH { syn("expression nested too deeply") } else { Ok(()) } }

    fn expr(&mut self) -> R<E> {
        self.enter()?;
        let r = self.cond();
        self.depth -= 1;
        r
    }
    fn cond(&mut self) -> R<E> {
        let a = self.or()?;
        if self.eat_name("if") {
            let c = self.or()?;
            let e = if self.eat_name("else") { Some(Box::new(self.cond()?)) } else { None };
            return Ok(E::Cond(Box::new(a), Box::new(c), e));
        }
        Ok(a)
    }
    fn or(&mut self) -> R<E> {
        let mut a = self.and()?;
        while self.eat_name("or") { a = E::Or(Box::new(a), Box::new(self.and()?)); }
        Ok(a)
    }
    fn and(&mut self) -> R<E> {
        let mut a = self.not()?;
        while self.eat_name("and") { a = E::And(Box::new(a), Box::new(self.not()?)); }
        Ok(a)
    }
    fn not(&mut self) -> R<E> {
        if self.eat_name("not") { self.enter()?; let r = self.not(); self.depth -= 1; return Ok(E::Not(Box::new(r?))); }
        self.cmp()
    }
    fn cmp(&mut self) -> R<E> {
        let mut a = self.concat()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(o)) if matches!(*o, "==" | "!=" | "<" | ">" | "<=" | ">=") => { let o = *o; self.i += 1; o }
                Some(Tok::Name(n)) if n == "in" => { self.i += 1; a = E::In(Box::new(a), Box::new(self.concat()?), false); continue; }
                Some(Tok::Name(n)) if n == "not" && matches!(self.t.get(self.i + 1), Some(Tok::Name(x)) if x == "in") => {
                    self.i += 2; a = E::In(Box::new(a), Box::new(self.concat()?), true); continue;
                }
                Some(Tok::Name(n)) if n == "is" => {
                    self.i += 1;
                    let neg = self.eat_name("not");
                    let Some(Tok::Name(test)) = self.peek().cloned() else { return syn("expected a test name after `is`") };
                    self.i += 1;
                    let mut args = vec![];
                    if self.eat_op("(") { args = self.args_plain()?; }
                    else if matches!(self.peek(), Some(Tok::Int(_)) | Some(Tok::Str(_))) && matches!(test.as_str(), "divisibleby" | "eq" | "ne" | "equalto") { args.push(self.concat()?); }
                    a = E::Test(Box::new(a), test, neg, args);
                    continue;
                }
                _ => break,
            };
            a = E::Bin(op, Box::new(a), Box::new(self.concat()?));
        }
        Ok(a)
    }
    fn args_plain(&mut self) -> R<Vec<E>> {
        let mut v = vec![];
        while !self.op(")") { v.push(self.expr()?); if !self.eat_op(",") { break; } }
        self.expect_op(")")?;
        Ok(v)
    }
    fn concat(&mut self) -> R<E> {
        let mut a = self.add()?;
        while self.eat_op("~") { a = E::Bin("~", Box::new(a), Box::new(self.add()?)); }
        Ok(a)
    }
    fn add(&mut self) -> R<E> {
        let mut a = self.mul()?;
        loop {
            let o = if self.op("+") { "+" } else if self.op("-") { "-" } else { break };
            self.i += 1;
            a = E::Bin(o, Box::new(a), Box::new(self.mul()?));
        }
        Ok(a)
    }
    fn mul(&mut self) -> R<E> {
        let mut a = self.unary()?;
        loop {
            let o = ["*", "//", "/", "%"].into_iter().find(|o| self.op(o));
            let Some(o) = o else { break };
            self.i += 1;
            a = E::Bin(o, Box::new(a), Box::new(self.unary()?));
        }
        Ok(a)
    }
    fn unary(&mut self) -> R<E> {
        if self.eat_op("-") { self.enter()?; let r = self.unary(); self.depth -= 1; return Ok(E::Neg(Box::new(r?))); }
        if self.eat_op("+") { return self.unary(); }
        self.postfix()
    }
    fn postfix(&mut self) -> R<E> {
        let mut e = self.primary()?;
        loop {
            if self.eat_op(".") {
                let Some(Tok::Name(n)) = self.peek().cloned() else { return syn("expected a name after `.`") };
                self.i += 1;
                e = E::Attr(Box::new(e), n);
            } else if self.eat_op("[") {
                self.enter()?;
                let r = self.subscript(e);
                self.depth -= 1;
                e = r?;
            } else if self.eat_op("(") {
                let (a, k) = self.call_args()?;
                e = E::Call(Box::new(e), a, k);
            } else if self.eat_op("|") {
                let Some(Tok::Name(n)) = self.peek().cloned() else { return syn("expected a filter name after `|`") };
                self.i += 1;
                let (a, k) = if self.eat_op("(") { self.call_args()? } else { (vec![], vec![]) };
                e = E::Filter(Box::new(e), n, a, k);
            } else { break; }
        }
        Ok(e)
    }
    fn subscript(&mut self, base: E) -> R<E> {
        let lo = if self.op(":") { None } else { Some(Box::new(self.expr()?)) };
        if self.eat_op(":") {
            let hi = if self.op("]") { None } else { Some(Box::new(self.expr()?)) };
            self.expect_op("]")?;
            return Ok(E::Slice(Box::new(base), lo, hi));
        }
        self.expect_op("]")?;
        match lo { Some(i) => Ok(E::Index(Box::new(base), i)), None => syn("empty subscript") }
    }
    fn call_args(&mut self) -> R<CallArgs> {
        let (mut a, mut k) = (vec![], vec![]);
        while !self.op(")") {
            if let (Some(Tok::Name(n)), Some(Tok::Op("="))) = (self.peek().cloned(), self.t.get(self.i + 1)) {
                self.i += 2;
                k.push((n, self.expr()?));
            } else { a.push(self.expr()?); }
            if !self.eat_op(",") { break; }
        }
        self.expect_op(")")?;
        Ok((a, k))
    }
    fn primary(&mut self) -> R<E> {
        let Some(t) = self.peek().cloned() else { return syn("unexpected end of expression") };
        self.i += 1;
        match t {
            Tok::Int(i) => Ok(E::Lit(Lit::Int(i))),
            Tok::Str(mut s) => {
                while let Some(Tok::Str(n)) = self.peek() { s.push_str(n); self.i += 1; } // adjacent literals join
                Ok(E::Lit(Lit::Str(s)))
            }
            Tok::Name(n) => Ok(match n.as_str() {
                "true" | "True" => E::Lit(Lit::Bool(true)),
                "false" | "False" => E::Lit(Lit::Bool(false)),
                "none" | "None" => E::Lit(Lit::None),
                _ => E::Var(n),
            }),
            Tok::Op("(") => {
                self.enter()?;
                let e = self.expr();
                self.depth -= 1;
                let e = e?;
                if self.eat_op(",") { // a tuple: treat as a list
                    let mut v = vec![e];
                    while !self.op(")") { v.push(self.expr()?); if !self.eat_op(",") { break; } }
                    self.expect_op(")")?;
                    return Ok(E::List(v));
                }
                self.expect_op(")")?;
                Ok(e)
            }
            Tok::Op("[") => {
                self.enter()?;
                let mut v = vec![];
                while !self.op("]") { v.push(self.expr()?); if !self.eat_op(",") { break; } }
                self.depth -= 1;
                self.expect_op("]")?;
                Ok(E::List(v))
            }
            Tok::Op("{") => {
                self.enter()?;
                let mut v = vec![];
                while !self.op("}") {
                    let k = self.expr()?;
                    self.expect_op(":")?;
                    v.push((k, self.expr()?));
                    if !self.eat_op(",") { break; }
                }
                self.depth -= 1;
                self.expect_op("}")?;
                Ok(E::Dict(v))
            }
            other => syn(format!("unexpected {other:?}")),
        }
    }
}

fn parse_expr(s: &str) -> R<E> {
    let mut p = P { t: toks(s)?, i: 0, depth: 0 };
    let e = p.expr()?;
    if !p.done() { return syn(format!("unexpected {:?} after expression", p.t[p.i])); }
    Ok(e)
}

// ------------------------------------------------------------------------------ statements

#[derive(Debug, Clone)]
enum Target { Name(String), Attr(String, String) }

#[derive(Debug, Clone)]
enum Node {
    Text(String),
    Print(E),
    If(Vec<(E, Vec<Node>)>, Vec<Node>),
    For { vars: Vec<String>, iter: E, cond: Option<E>, body: Vec<Node>, orelse: Vec<Node> },
    Set(Target, E),
    Break,
    Continue,
}

enum Open { If(Vec<(E, Vec<Node>)>, Option<E>), For(Vec<String>, E, Option<E>, Option<Vec<Node>>), Generation }

type CallArgs = (Vec<E>, Vec<(String, E)>);

/// A parsed template.
#[derive(Debug, Clone)]
pub struct Template { nodes: Vec<Node> }

impl Template {
    pub fn parse(src: &str) -> R<Template> {
        let pieces = lex(src)?;
        // stack of (open block, nodes collected so far in the *current* branch)
        let mut stack: Vec<(Open, Vec<Node>)> = vec![];
        let mut cur: Vec<Node> = vec![];
        for piece in pieces {
            match piece {
                Piece::Text(t) => cur.push(Node::Text(t)),
                Piece::Expr(s) => cur.push(Node::Print(parse_expr(&s)?)),
                Piece::Stmt(s) => {
                    let t = toks(&s)?;
                    let Some(Tok::Name(kw)) = t.first().cloned() else { return syn("empty statement") };
                    let rest = |from: usize| t[from..].to_vec();
                    match kw.as_str() {
                        "if" => {
                            if stack.len() >= MAX_DEPTH { return syn("blocks nested too deeply"); }
                            let c = whole_expr(rest(1))?;
                            stack.push((Open::If(vec![], Some(c)), std::mem::take(&mut cur)));
                        }
                        "elif" => {
                            let Some((Open::If(branches, c), _)) = stack.last_mut() else { return syn("`elif` without `if`") };
                            let Some(prev) = c.take() else { return syn("`elif` after `else`") };
                            branches.push((prev, std::mem::take(&mut cur)));
                            *c = Some(whole_expr(rest(1))?);
                        }
                        "else" => match stack.last_mut() {
                            Some((Open::If(branches, c), _)) => {
                                let Some(prev) = c.take() else { return syn("duplicate `else`") };
                                branches.push((prev, std::mem::take(&mut cur)));
                            }
                            Some((Open::For(_, _, _, orelse), _)) => {
                                if orelse.is_some() { return syn("duplicate `else`"); }
                                // from here on `cur` collects the else branch; the loop body is parked in the slot
                                *orelse = Some(std::mem::take(&mut cur));
                            }
                            _ => return syn("`else` outside `if` or `for`"),
                        },
                        "endif" => {
                            let Some((Open::If(mut branches, c), outer)) = stack.pop() else { return syn("`endif` without `if`") };
                            let orelse = match c {
                                Some(last) => { branches.push((last, std::mem::take(&mut cur))); vec![] }
                                None => std::mem::take(&mut cur),
                            };
                            cur = outer;
                            cur.push(Node::If(branches, orelse));
                        }
                        "for" => {
                            if stack.len() >= MAX_DEPTH { return syn("blocks nested too deeply"); }
                            let Some(in_pos) = t.iter().position(|x| matches!(x, Tok::Name(n) if n == "in")) else { return syn("`for` without `in`") };
                            let mut vars = vec![];
                            for x in &t[1..in_pos] {
                                match x { Tok::Name(n) => vars.push(n.clone()), Tok::Op(",") => {}, _ => return syn("bad loop variable") }
                            }
                            if vars.is_empty() { return syn("`for` without a variable"); }
                            let mut tail = rest(in_pos + 1);
                            let mut cond = None;
                            let mut depth = 0i32;
                            let ifp = tail.iter().position(|x| match x {
                                Tok::Op("(") | Tok::Op("[") | Tok::Op("{") => { depth += 1; false }
                                Tok::Op(")") | Tok::Op("]") | Tok::Op("}") => { depth -= 1; false }
                                Tok::Name(n) => depth == 0 && n == "if",
                                _ => false,
                            });
                            if let Some(ifp) = ifp {
                                cond = Some(whole_expr(tail[ifp + 1..].to_vec())?);
                                tail.truncate(ifp);
                            }
                            stack.push((Open::For(vars, whole_expr(tail)?, cond, None), std::mem::take(&mut cur)));
                        }
                        "endfor" => {
                            let Some((Open::For(vars, iter, cond, loop_body), outer)) = stack.pop() else { return syn("`endfor` without `for`") };
                            let (body, orelse) = match loop_body { Some(b) => (b, std::mem::take(&mut cur)), None => (std::mem::take(&mut cur), vec![]) };
                            cur = outer;
                            cur.push(Node::For { vars, iter, cond, body, orelse });
                        }
                        "generation" => { if stack.len() >= MAX_DEPTH { return syn("blocks nested too deeply"); } stack.push((Open::Generation, std::mem::take(&mut cur))); }
                        "endgeneration" => {
                            let Some((Open::Generation, mut outer)) = stack.pop() else { return syn("`endgeneration` without `generation`") };
                            outer.append(&mut cur);
                            cur = outer;
                        }
                        "set" => {
                            let Some(eq) = t.iter().position(|x| matches!(x, Tok::Op("="))) else { return syn("only `{% set x = ... %}` is supported") };
                            let target = match &t[1..eq] {
                                [Tok::Name(n)] => Target::Name(n.clone()),
                                [Tok::Name(n), Tok::Op("."), Tok::Name(a)] => Target::Attr(n.clone(), a.clone()),
                                _ => return syn("unsupported `set` target"),
                            };
                            cur.push(Node::Set(target, whole_expr(rest(eq + 1))?));
                        }
                        "break" => cur.push(Node::Break),
                        "continue" => cur.push(Node::Continue),
                        other => return syn(format!("unsupported statement `{other}`")),
                    }
                }
            }
        }
        if !stack.is_empty() { return syn("a block is not closed"); }
        Ok(Template { nodes: cur })
    }

    /// Render with the given top-level variables.
    pub fn render(&self, vars: &BTreeMap<String, V>) -> R<String> {
        let mut c = Ctx { scopes: vec![vars.iter().map(|(k, v)| (k.clone(), v.clone())).collect()], steps: 0, out: String::new() };
        match c.run(&self.nodes)? { Flow::Next => {}, _ => return rt("`break`/`continue` outside a loop") }
        Ok(c.out)
    }
}

fn whole_expr(t: Vec<Tok>) -> R<E> {
    let mut p = P { t, i: 0, depth: 0 };
    let e = p.expr()?;
    if !p.done() { return syn("unexpected trailing tokens"); }
    Ok(e)
}

// ------------------------------------------------------------------------------ evaluation

enum Flow { Next, Break, Continue }

struct Ctx { scopes: Vec<HashMap<String, V>>, steps: usize, out: String }

impl Ctx {
    fn tick(&mut self) -> R<()> {
        self.steps += 1;
        if self.steps > MAX_STEPS { Err(TemplateError::Limit("step")) } else { Ok(()) }
    }
    fn emit(&mut self, s: &str) -> R<()> {
        if self.out.len() + s.len() > MAX_OUTPUT { return Err(TemplateError::Limit("output size")); }
        self.out.push_str(s);
        Ok(())
    }
    fn lookup(&self, n: &str) -> V {
        for s in self.scopes.iter().rev() { if let Some(v) = s.get(n) { return v.clone(); } }
        V::Undefined
    }
    fn run(&mut self, nodes: &[Node]) -> R<Flow> {
        for n in nodes {
            self.tick()?;
            match n {
                Node::Text(t) => self.emit(t)?,
                Node::Print(e) => { let v = self.eval(e)?; self.emit(&v.render())?; }
                Node::If(branches, orelse) => {
                    let mut done = false;
                    for (c, body) in branches {
                        if self.eval(c)?.truthy() {
                            done = true;
                            match self.run(body)? { Flow::Next => {}, f => return Ok(f) }
                            break;
                        }
                    }
                    if !done { match self.run(orelse)? { Flow::Next => {}, f => return Ok(f) } }
                }
                Node::Set(t, e) => {
                    let v = self.eval(e)?;
                    match t {
                        Target::Name(n) => { if let Some(s) = self.scopes.last_mut() { s.insert(n.clone(), v); } }
                        Target::Attr(n, a) => match self.lookup(n) {
                            V::Map(m) => { m.borrow_mut().insert(a.clone(), v); }
                            _ => return rt(format!("`{n}` is not a namespace")),
                        },
                    }
                }
                Node::Break => return Ok(Flow::Break),
                Node::Continue => return Ok(Flow::Continue),
                Node::For { vars, iter, cond, body, orelse } => {
                    let items: Vec<V> = match self.eval(iter)? {
                        V::List(l) => l.borrow().clone(),
                        V::Map(m) => m.borrow().keys().map(|k| V::Str(k.clone())).collect(),
                        V::Str(s) => s.chars().map(|c| V::Str(c.to_string())).collect(),
                        V::None | V::Undefined => vec![],
                        other => return rt(format!("cannot loop over a {}", other.kind())),
                    };
                    self.scopes.push(HashMap::new());
                    // `if` filter first, so loop.index counts only the kept items
                    let mut kept = Vec::with_capacity(items.len());
                    for it in items {
                        self.tick()?;
                        if let Some(c) = cond {
                            self.bind(vars, &it)?;
                            if !self.eval(c)?.truthy() { continue; }
                        }
                        kept.push(it);
                    }
                    let n = kept.len();
                    let mut result = Flow::Next;
                    for (idx, it) in kept.into_iter().enumerate() {
                        self.tick()?;
                        self.bind(vars, &it)?;
                        let lp: OMap = [
                            ("index", V::Int(idx as i64 + 1)), ("index0", V::Int(idx as i64)),
                            ("revindex", V::Int((n - idx) as i64)), ("revindex0", V::Int((n - idx) as i64 - 1)),
                            ("first", V::Bool(idx == 0)), ("last", V::Bool(idx + 1 == n)), ("length", V::Int(n as i64)),
                        ].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
                        if let Some(s) = self.scopes.last_mut() { s.insert("loop".into(), V::map(lp)); }
                        match self.run(body) {
                            Ok(Flow::Next) | Ok(Flow::Continue) => {}
                            Ok(Flow::Break) => break,
                            Err(e) => { self.scopes.pop(); return Err(e); }
                        }
                    }
                    self.scopes.pop();
                    if n == 0 && let f @ (Flow::Break | Flow::Continue) = self.run(orelse)? { result = f; }
                    if !matches!(result, Flow::Next) { return Ok(result); }
                }
            }
        }
        Ok(Flow::Next)
    }
    fn bind(&mut self, vars: &[String], item: &V) -> R<()> {
        let Some(s) = self.scopes.last_mut() else { return Ok(()) };
        if vars.len() == 1 { s.insert(vars[0].clone(), item.clone()); return Ok(()); }
        match item {
            V::List(l) if l.borrow().len() == vars.len() => { for (n, v) in vars.iter().zip(l.borrow().iter()) { s.insert(n.clone(), v.clone()); } Ok(()) }
            _ => rt("cannot unpack the loop item"),
        }
    }

    fn eval(&mut self, e: &E) -> R<V> {
        self.tick()?;
        Ok(match e {
            E::Lit(l) => l.val(),
            E::Var(n) => self.lookup(n),
            E::List(items) => { let mut v = vec![]; for i in items { v.push(self.eval(i)?); } V::list(v) }
            E::Dict(items) => {
                let mut m = OMap::default();
                for (k, v) in items { let k = self.eval(k)?; let v = self.eval(v)?; m.insert(k.render(), v); }
                V::map(m)
            }
            E::Attr(b, n) => { let b = self.eval(b)?; attr(&b, n) }
            E::Index(b, i) => { let (b, i) = (self.eval(b)?, self.eval(i)?); index(&b, &i)? }
            E::Slice(b, lo, hi) => {
                let b = self.eval(b)?;
                let lo = match lo { Some(x) => Some(self.eval(x)?), None => None };
                let hi = match hi { Some(x) => Some(self.eval(x)?), None => None };
                slice(&b, lo, hi)?
            }
            E::Not(x) => V::Bool(!self.eval(x)?.truthy()),
            E::Neg(x) => match self.eval(x)? { V::Int(i) => V::Int(i.checked_neg().ok_or_else(|| TemplateError::Runtime("overflow".into()))?), o => return rt(format!("cannot negate a {}", o.kind())) },
            E::And(a, b) => { let a = self.eval(a)?; if !a.truthy() { a } else { self.eval(b)? } }
            E::Or(a, b) => { let a = self.eval(a)?; if a.truthy() { a } else { self.eval(b)? } }
            E::Cond(a, c, e) => {
                if self.eval(c)?.truthy() { self.eval(a)? } else { match e { Some(e) => self.eval(e)?, None => V::Undefined } }
            }
            E::In(a, b, neg) => { let (a, b) = (self.eval(a)?, self.eval(b)?); V::Bool(contains(&b, &a)? != *neg) }
            E::Bin(op, a, b) => { let (a, b) = (self.eval(a)?, self.eval(b)?); binop(op, &a, &b)? }
            E::Test(x, name, neg, args) => {
                let v = self.eval(x)?;
                let mut av = vec![];
                for a in args { av.push(self.eval(a)?); }
                V::Bool(run_test(&v, name, &av)? != *neg)
            }
            E::Filter(x, name, args, kw) => {
                let v = self.eval(x)?;
                let mut av = vec![];
                for a in args { av.push(self.eval(a)?); }
                let mut kv = vec![];
                for (k, a) in kw { kv.push((k.clone(), self.eval(a)?)); }
                apply_filter(&v, name, &av, &kv)?
            }
            E::Call(f, args, kw) => {
                let mut av = vec![];
                for a in args { av.push(self.eval(a)?); }
                let mut kv = vec![];
                for (k, a) in kw { kv.push((k.clone(), self.eval(a)?)); }
                match &**f {
                    E::Var(n) => call_global(n, &av, &kv)?,
                    E::Attr(obj, m) => { let o = self.eval(obj)?; call_method(&o, m, &av)? }
                    _ => return rt("this value cannot be called"),
                }
            }
        })
    }
}

fn attr(b: &V, n: &str) -> V {
    match b {
        V::Map(m) => m.borrow().get(n).cloned().unwrap_or(V::Undefined),
        _ => V::Undefined,
    }
}

fn norm_index(i: i64, len: usize) -> Option<usize> {
    let j = if i < 0 { len as i64 + i } else { i };
    if j < 0 || j as usize >= len { None } else { Some(j as usize) }
}

fn index(b: &V, i: &V) -> R<V> {
    Ok(match (b, i) {
        (V::List(l), V::Int(i)) => { let l = l.borrow(); norm_index(*i, l.len()).map_or(V::Undefined, |j| l[j].clone()) }
        (V::Str(s), V::Int(i)) => { let c: Vec<char> = s.chars().collect(); norm_index(*i, c.len()).map_or(V::Undefined, |j| V::Str(c[j].to_string())) }
        (V::Map(m), k) => m.borrow().get(&k.render()).cloned().unwrap_or(V::Undefined),
        (V::None | V::Undefined, _) => V::Undefined,
        (b, i) => return rt(format!("cannot index a {} with a {}", b.kind(), i.kind())),
    })
}

fn slice(b: &V, lo: Option<V>, hi: Option<V>) -> R<V> {
    let bound = |v: Option<V>, len: usize, default: usize| -> R<usize> {
        match v {
            None | Some(V::None) => Ok(default),
            Some(V::Int(i)) => Ok(if i < 0 { (len as i64 + i).max(0) as usize } else { (i as usize).min(len) }),
            Some(o) => rt(format!("slice bounds must be integers, not {}", o.kind())),
        }
    };
    match b {
        V::List(l) => { let l = l.borrow(); let (a, z) = (bound(lo, l.len(), 0)?, bound(hi, l.len(), l.len())?); Ok(V::list(if a < z { l[a..z].to_vec() } else { vec![] })) }
        V::Str(s) => { let c: Vec<char> = s.chars().collect(); let (a, z) = (bound(lo, c.len(), 0)?, bound(hi, c.len(), c.len())?); Ok(V::Str(if a < z { c[a..z].iter().collect() } else { String::new() })) }
        V::None | V::Undefined => Ok(V::Undefined),
        o => rt(format!("cannot slice a {}", o.kind())),
    }
}

fn contains(hay: &V, needle: &V) -> R<bool> {
    Ok(match (hay, needle) {
        (V::Str(h), V::Str(n)) => h.contains(n.as_str()),
        (V::List(l), n) => l.borrow().iter().any(|x| x.eq(n)),
        (V::Map(m), k) => m.borrow().contains_key(&k.render()),
        (V::None | V::Undefined, _) => false,
        (h, n) => return rt(format!("cannot test whether a {} is in a {}", n.kind(), h.kind())),
    })
}

fn binop(op: &str, a: &V, b: &V) -> R<V> {
    Ok(match (op, a, b) {
        ("~", a, b) => V::Str(a.render() + &b.render()),
        ("+", V::Str(x), V::Str(y)) => V::Str(x.clone() + y),
        ("+", V::List(x), V::List(y)) => { let mut v = x.borrow().clone(); v.extend(y.borrow().iter().cloned()); V::list(v) }
        ("==", a, b) => V::Bool(a.eq(b)),
        ("!=", a, b) => V::Bool(!a.eq(b)),
        (o @ ("<" | ">" | "<=" | ">="), V::Int(x), V::Int(y)) => V::Bool(match o { "<" => x < y, ">" => x > y, "<=" => x <= y, _ => x >= y }),
        (o @ ("<" | ">" | "<=" | ">="), V::Str(x), V::Str(y)) => V::Bool(match o { "<" => x < y, ">" => x > y, "<=" => x <= y, _ => x >= y }),
        (o, V::Int(x), V::Int(y)) => {
            let r = match o {
                "+" => x.checked_add(*y), "-" => x.checked_sub(*y), "*" => x.checked_mul(*y),
                "//" => if *y == 0 { return rt("division by zero") } else { Some(x.div_euclid(*y)) },
                "%" => if *y == 0 { return rt("division by zero") } else { Some(x.rem_euclid(*y)) },
                "/" => if *y == 0 { return rt("division by zero") } else if x % y == 0 { Some(x / y) } else { return rt("non-integer division is not supported") },
                _ => return rt(format!("unsupported operator {o}")),
            };
            V::Int(r.ok_or_else(|| TemplateError::Runtime("integer overflow".into()))?)
        }
        ("*", V::Str(s), V::Int(n)) | ("*", V::Int(n), V::Str(s)) => {
            if *n < 0 || (*n as usize).saturating_mul(s.len()) > MAX_OUTPUT { return Err(TemplateError::Limit("output size")); }
            V::Str(s.repeat(*n as usize))
        }
        (o, a, b) => return rt(format!("unsupported operands for {o}: {} and {}", a.kind(), b.kind())),
    })
}

fn run_test(v: &V, name: &str, args: &[V]) -> R<bool> {
    Ok(match name {
        "defined" => !matches!(v, V::Undefined),
        "undefined" => matches!(v, V::Undefined),
        "none" => matches!(v, V::None),
        "string" => matches!(v, V::Str(_)),
        "number" | "integer" => matches!(v, V::Int(_)),
        "boolean" => matches!(v, V::Bool(_)),
        "true" => matches!(v, V::Bool(true)),
        "false" => matches!(v, V::Bool(false)),
        "mapping" => matches!(v, V::Map(_)),
        "sequence" | "iterable" => matches!(v, V::List(_) | V::Str(_) | V::Map(_)),
        "odd" => matches!(v, V::Int(i) if i.rem_euclid(2) == 1),
        "even" => matches!(v, V::Int(i) if i.rem_euclid(2) == 0),
        "divisibleby" => match (v, args.first()) { (V::Int(a), Some(V::Int(b))) if *b != 0 => a % b == 0, _ => false },
        "eq" | "equalto" | "sameas" => args.first().is_some_and(|a| v.eq(a)),
        "gt" | "greaterthan" => matches!((v, args.first()), (V::Int(a), Some(V::Int(b))) if a > b),
        "ge" => matches!((v, args.first()), (V::Int(a), Some(V::Int(b))) if a >= b),
        "lt" | "lessthan" => matches!((v, args.first()), (V::Int(a), Some(V::Int(b))) if a < b),
        "le" => matches!((v, args.first()), (V::Int(a), Some(V::Int(b))) if a <= b),
        "in" => args.first().is_some_and(|c| match c { V::List(l) => l.borrow().iter().any(|x| x.eq(v)), V::Str(s) => s.contains(&string_of(v)), V::Map(m) => m.borrow().contains_key(&string_of(v)), _ => false }),
        "ne" => args.first().is_some_and(|a| !v.eq(a)),
        other => return rt(format!("unknown test `{other}`")),
    })
}

fn string_of(v: &V) -> String { v.render() }

/// `a.b.c` attribute lookup (what `attribute='a.b'` means in Jinja filters).
fn dotted(v: &V, path: &str) -> V { path.split('.').fold(v.clone(), |cur, part| match part.parse::<i64>() { Ok(i) => index(&cur, &V::Int(i)).unwrap_or(V::Undefined), Err(_) => attr(&cur, part) }) }

/// Filters: trim, length/count, upper, lower, title, capitalize, string, int, list, first, last, join,
/// default/d, tojson, replace, reverse, abs, safe, escape, items, unique, selectattr(…) is not supported.
fn apply_filter(v: &V, name: &str, args: &[V], kw: &[(String, V)]) -> R<V> {
    Ok(match name {
        "trim" => V::Str(string_of(v).trim().to_string()),
        "length" | "count" => V::Int(match v {
            V::Str(s) => s.chars().count(), V::List(l) => l.borrow().len(), V::Map(m) => m.borrow().len(), V::None | V::Undefined => 0,
            o => return rt(format!("a {} has no length", o.kind())),
        } as i64),
        "upper" => V::Str(string_of(v).to_uppercase()),
        "lower" => V::Str(string_of(v).to_lowercase()),
        "title" => V::Str(title(&string_of(v))),
        "capitalize" => { let s = string_of(v); let mut c = s.chars(); V::Str(match c.next() { Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(), None => String::new() }) }
        "string" | "safe" | "escape" | "e" => V::Str(string_of(v)),
        "int" => match v { V::Int(i) => V::Int(*i), V::Bool(b) => V::Int(*b as i64), V::Str(s) => V::Int(s.trim().parse().unwrap_or(0)), _ => V::Int(0) },
        "abs" => match v { V::Int(i) => V::Int(i.checked_abs().unwrap_or(i64::MAX)), o => return rt(format!("cannot take abs of a {}", o.kind())) },
        "list" => match v {
            V::List(_) => v.clone(),
            V::Str(s) => V::list(s.chars().map(|c| V::Str(c.to_string())).collect()),
            V::Map(m) => V::list(m.borrow().keys().map(|k| V::Str(k.clone())).collect()),
            _ => V::list(vec![]),
        },
        "first" => match v { V::List(l) => l.borrow().first().cloned().unwrap_or(V::Undefined), V::Str(s) => s.chars().next().map_or(V::Undefined, |c| V::Str(c.to_string())), _ => V::Undefined },
        "last" => match v { V::List(l) => l.borrow().last().cloned().unwrap_or(V::Undefined), V::Str(s) => s.chars().last().map_or(V::Undefined, |c| V::Str(c.to_string())), _ => V::Undefined },
        "reverse" => match v { V::List(l) => V::list(l.borrow().iter().rev().cloned().collect()), V::Str(s) => V::Str(s.chars().rev().collect()), o => return rt(format!("cannot reverse a {}", o.kind())) },
        "join" => match v {
            V::List(l) => { let sep = args.first().map(string_of).unwrap_or_default(); V::Str(l.borrow().iter().map(string_of).collect::<Vec<_>>().join(&sep)) }
            o => return rt(format!("cannot join a {}", o.kind())),
        },
        "items" => match v { V::Map(m) => V::list(m.borrow().iter().map(|(k, x)| V::list(vec![V::Str(k.clone()), x.clone()])).collect()), _ => V::list(vec![]) },
        "replace" => match (v, args) {
            (V::Str(s), [a, b, ..]) => V::Str(s.replace(&string_of(a), &string_of(b))),
            _ => return rt("`replace` needs a string and two arguments"),
        },
        "default" | "d" => {
            let by_bool = matches!(args.get(1), Some(V::Bool(true))) || kw.iter().any(|(k, x)| k == "boolean" && x.truthy());
            let dflt = args.first().cloned().or_else(|| kw.iter().find(|(k, _)| k == "default_value").map(|(_, x)| x.clone())).unwrap_or(V::Str(String::new()));
            if matches!(v, V::Undefined) || (by_bool && !v.truthy()) { dflt } else { v.clone() }
        }
        "tojson" => {
            let indent = kw.iter().find(|(k, _)| k == "indent").map(|(_, x)| x).or(args.first());
            V::Str(to_json(v, match indent { Some(V::Int(n)) if *n > 0 && *n <= 16 => Some(*n as usize), _ => None }, 0))
        }
        "selectattr" | "rejectattr" | "select" | "reject" => {
            let want = name.starts_with("select");
            let attr_mode = name.ends_with("attr");
            let items: Vec<V> = match v { V::List(l) => l.borrow().clone(), V::None | V::Undefined => vec![], o => return rt(format!("cannot filter a {}", o.kind())) };
            let (target, rest) = if attr_mode { (args.first().map(string_of), args.get(1..).unwrap_or(&[])) } else { (None, args) };
            let test = rest.first().map(string_of);
            let targs = rest.get(1..).unwrap_or(&[]);
            let mut out = vec![];
            for it in items {
                let probe = match &target { Some(t) => dotted(&it, t), None => it.clone() };
                let ok = match &test { Some(t) => run_test(&probe, t, targs)?, None => probe.truthy() };
                if ok == want { out.push(it); }
            }
            V::list(out)
        }
        "map" => {
            let items: Vec<V> = match v { V::List(l) => l.borrow().clone(), V::None | V::Undefined => vec![], o => return rt(format!("cannot map a {}", o.kind())) };
            let attribute = kw.iter().find(|(k, _)| k == "attribute").map(|(_, x)| string_of(x));
            let dflt = kw.iter().find(|(k, _)| k == "default").map(|(_, x)| x.clone());
            match (attribute, args.first()) {
                (Some(a), _) => V::list(items.iter().map(|i| match dotted(i, &a) { V::Undefined => dflt.clone().unwrap_or(V::Undefined), x => x }).collect()),
                (None, Some(f)) => { let f = string_of(f); let mut o = vec![]; for i in &items { o.push(apply_filter(i, &f, args.get(1..).unwrap_or(&[]), &[])?); } V::list(o) }
                _ => V::list(items),
            }
        }
        "sort" => {
            let mut items: Vec<V> = match v { V::List(l) => l.borrow().clone(), _ => return rt("cannot sort a non-list") };
            let attribute = kw.iter().find(|(k, _)| k == "attribute").map(|(_, x)| string_of(x));
            let key = |x: &V| -> V { match &attribute { Some(a) => dotted(x, a), None => x.clone() } };
            items.sort_by(|a, b| match (key(a), key(b)) { (V::Int(x), V::Int(y)) => x.cmp(&y), (x, y) => x.render().cmp(&y.render()) });
            if kw.iter().any(|(k, x)| k == "reverse" && x.truthy()) { items.reverse(); }
            V::list(items)
        }
        "unique" => {
            let mut out: Vec<V> = vec![];
            if let V::List(l) = v { for x in l.borrow().iter() { if !out.iter().any(|y| y.eq(x)) { out.push(x.clone()); } } }
            V::list(out)
        }
        "sum" => match v { V::List(l) => V::Int(l.borrow().iter().map(|x| if let V::Int(i) = x { *i } else { 0 }).sum()), _ => V::Int(0) },
        "min" | "max" => match v {
            V::List(l) => { let l = l.borrow(); let it = l.iter().filter_map(|x| if let V::Int(i) = x { Some(*i) } else { None }); (if name == "min" { it.min() } else { it.max() }).map_or(V::Undefined, V::Int) }
            _ => V::Undefined,
        },
        "indent" => {
            let n = match args.first() { Some(V::Int(n)) if *n > 0 && *n <= 64 => *n as usize, _ => 4 };
            let pad = " ".repeat(n);
            V::Str(string_of(v).split('\n').enumerate().map(|(i, l)| if i == 0 || l.is_empty() { l.to_string() } else { format!("{pad}{l}") }).collect::<Vec<_>>().join("\n"))
        }
        "dictsort" => match v { V::Map(m) => { let mut e: Vec<(String, V)> = m.borrow().iter().map(|(k, x)| (k.clone(), x.clone())).collect(); e.sort_by(|a, b| a.0.cmp(&b.0)); V::list(e.into_iter().map(|(k, x)| V::list(vec![V::Str(k), x])).collect()) } _ => V::list(vec![]) },
        "float" => V::Int(match v { V::Int(i) => *i, V::Str(s) => s.trim().parse::<f64>().map_or(0, |f| f as i64), _ => 0 }),
        "wordcount" => V::Int(string_of(v).split_whitespace().count() as i64),
        "bool" => V::Bool(v.truthy()),
        other => return rt(format!("unsupported filter `{other}`")),
    })
}

fn title(s: &str) -> String {
    let mut out = String::new();
    let mut start = true;
    for c in s.chars() {
        if c.is_alphanumeric() { if start { out.extend(c.to_uppercase()); } else { out.extend(c.to_lowercase()); } start = false; } else { out.push(c); start = true; }
    }
    out
}

fn call_method(o: &V, m: &str, a: &[V]) -> R<V> {
    let arg_s = |i: usize| a.get(i).map(string_of);
    Ok(match (o, m) {
        (V::Str(s), "strip") => V::Str(match arg_s(0) { Some(c) => s.trim_matches(|x| c.contains(x)).to_string(), None => s.trim().to_string() }),
        (V::Str(s), "lstrip") => V::Str(match arg_s(0) { Some(c) => s.trim_start_matches(|x| c.contains(x)).to_string(), None => s.trim_start().to_string() }),
        (V::Str(s), "rstrip") => V::Str(match arg_s(0) { Some(c) => s.trim_end_matches(|x| c.contains(x)).to_string(), None => s.trim_end().to_string() }),
        (V::Str(s), "startswith") => V::Bool(arg_s(0).is_some_and(|p| s.starts_with(&p))),
        (V::Str(s), "endswith") => V::Bool(arg_s(0).is_some_and(|p| s.ends_with(&p))),
        (V::Str(s), "upper") => V::Str(s.to_uppercase()),
        (V::Str(s), "lower") => V::Str(s.to_lowercase()),
        (V::Str(s), "title") => V::Str(title(s)),
        (V::Str(s), "replace") => V::Str(s.replace(&arg_s(0).unwrap_or_default(), &arg_s(1).unwrap_or_default())),
        (V::Str(s), "split") => {
            let max = match a.get(1) { Some(V::Int(n)) if *n >= 0 => Some(*n as usize + 1), _ => None };
            let parts: Vec<V> = match (arg_s(0), max) {
                (Some(sep), _) if sep.is_empty() => return rt("empty separator"),
                (Some(sep), Some(n)) => s.splitn(n, &sep).map(V::str).collect(),
                (Some(sep), None) => s.split(&sep).map(V::str).collect(),
                (None, _) => s.split_whitespace().map(V::str).collect(),
            };
            V::list(parts)
        }
        (V::Str(sep), "join") => match a.first() { Some(V::List(l)) => V::Str(l.borrow().iter().map(string_of).collect::<Vec<_>>().join(sep)), _ => return rt("`join` needs a list") },
        (V::Map(m), "items") => V::list(m.borrow().iter().map(|(k, x)| V::list(vec![V::Str(k.clone()), x.clone()])).collect()),
        (V::Map(m), "keys") => V::list(m.borrow().keys().map(|k| V::Str(k.clone())).collect()),
        (V::Map(m), "values") => V::list(m.borrow().values().cloned().collect()),
        (V::Map(m), "get") => m.borrow().get(&arg_s(0).unwrap_or_default()).cloned().unwrap_or_else(|| a.get(1).cloned().unwrap_or(V::None)),
        (V::List(l), "append") => { if l.borrow().len() >= 100_000 { return Err(TemplateError::Limit("list size")); } l.borrow_mut().push(a.first().cloned().unwrap_or(V::None)); V::None }
        (V::Str(s), "find") => V::Int(arg_s(0).and_then(|p| s.find(&p)).map_or(-1, |b| s[..b].chars().count() as i64)),
        (V::Str(s), "count") => V::Int(match arg_s(0) { Some(p) if !p.is_empty() => s.matches(&p).count() as i64, _ => 0 }),
        (V::Str(s), "capitalize") => { let mut c = s.chars(); V::Str(match c.next() { Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(), None => String::new() }) }
        (V::Str(s), "splitlines") => V::list(s.lines().map(V::str).collect()),
        (V::Str(s), "isdigit") => V::Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())),
        (V::Str(s), "isalpha") => V::Bool(!s.is_empty() && s.chars().all(char::is_alphabetic)),
        (V::Str(s), "rsplit") => { let sep = arg_s(0).unwrap_or_else(|| " ".into()); if sep.is_empty() { return rt("empty separator"); } match a.get(1) { Some(V::Int(n)) if *n >= 0 => V::list({ let mut v: Vec<V> = s.rsplitn(*n as usize + 1, &sep).map(V::str).collect(); v.reverse(); v }), _ => V::list(s.split(&sep).map(V::str).collect()) } }
        (V::Str(s), "format") => { let mut out = s.clone(); for x in a { out = out.replacen("{}", &string_of(x), 1); } V::Str(out) }
        (V::Map(m), "update") => { if let Some(V::Map(o)) = a.first() { let add: Vec<(String, V)> = o.borrow().iter().map(|(k, x)| (k.clone(), x.clone())).collect(); for (k, x) in add { m.borrow_mut().insert(k, x); } } V::None }
        (V::Map(m), "pop") => { let k = arg_s(0).unwrap_or_default(); let mut mm = m.borrow_mut(); let old = mm.get(&k).cloned(); let rest: Vec<(String, V)> = mm.iter().filter(|(n, _)| **n != k).map(|(n, x)| (n.clone(), x.clone())).collect(); *mm = rest.into_iter().collect(); old.unwrap_or_else(|| a.get(1).cloned().unwrap_or(V::None)) }
        (V::List(l), "pop") => { let mut l = l.borrow_mut(); if l.is_empty() { V::Undefined } else { l.pop().unwrap_or(V::Undefined) } }
        (V::List(l), "extend") => { if let Some(V::List(o)) = a.first() { let add = o.borrow().clone(); if l.borrow().len() + add.len() > 100_000 { return Err(TemplateError::Limit("list size")); } l.borrow_mut().extend(add); } V::None }
        (V::Undefined | V::None, _) => V::Undefined,
        (o, m) => return rt(format!("a {} has no method `{m}`", o.kind())),
    })
}

fn call_global(name: &str, a: &[V], kw: &[(String, V)]) -> R<V> {
    match name {
        "raise_exception" => Err(TemplateError::Raised(a.first().map(string_of).unwrap_or_else(|| "the chat template raised an error".into()))),
        "namespace" => Ok(V::map(kw.iter().cloned().collect())),
        "dict" => Ok(V::map(kw.iter().cloned().collect())),
        "range" => {
            let (lo, hi) = match a { [V::Int(n)] => (0, *n), [V::Int(x), V::Int(y)] => (*x, *y), _ => return rt("`range` needs integers") };
            if hi.saturating_sub(lo) > 100_000 { return Err(TemplateError::Limit("range size")); }
            Ok(V::list((lo..hi.max(lo)).map(V::Int).collect()))
        }
        "strftime_now" => Ok(V::Str(strftime(&a.first().map(string_of).unwrap_or_default(), now_secs()))),
        other => rt(format!("unknown function `{other}`")),
    }
}

fn now_secs() -> i64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64) }

/// Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
pub fn civil(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d, (rem / 3600) as u32, (rem % 3600 / 60) as u32, (rem % 60) as u32)
}

const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

pub fn strftime(fmt: &str, secs: i64) -> String {
    let (y, mo, d, h, mi, s) = civil(secs);
    let mut o = String::new();
    let mut it = fmt.chars().peekable();
    while let Some(c) = it.next() {
        if c != '%' { o.push(c); continue; }
        let mut nopad = false;
        if it.peek() == Some(&'-') { nopad = true; it.next(); }
        let pad = |n: u32| if nopad { n.to_string() } else { format!("{n:02}") };
        match it.next() {
            Some('Y') => o.push_str(&y.to_string()),
            Some('m') => o.push_str(&pad(mo)),
            Some('d') => o.push_str(&pad(d)),
            Some('H') => o.push_str(&pad(h)),
            Some('M') => o.push_str(&pad(mi)),
            Some('S') => o.push_str(&pad(s)),
            Some('B') => o.push_str(MONTHS[(mo as usize).clamp(1, 12) - 1]),
            Some('b') => o.push_str(&MONTHS[(mo as usize).clamp(1, 12) - 1][..3]),
            Some('%') => o.push('%'),
            Some(x) => { o.push('%'); o.push(x); }
            None => o.push('%'),
        }
    }
    o
}

// ------------------------------------------------------------------------------ chat layer

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// OpenAI-shaped `tool_calls` of an assistant turn (`arguments` is a JSON string or an object).
    pub tool_calls: Vec<serde_json::Value>,
    /// For `tool` messages: the call this is the result of.
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}
impl Message {
    pub fn new(role: &str, content: &str) -> Message { Message { role: role.into(), content: content.into(), ..Default::default() } }

    /// The message as template data; tool-call arguments become objects (what Hugging Face templates expect).
    fn to_v(&self) -> V {
        let mut m = OMap::default();
        m.insert("role".into(), V::str(&self.role));
        m.insert("content".into(), V::str(&self.content));
        if !self.tool_calls.is_empty() { m.insert("tool_calls".into(), V::list(self.tool_calls.iter().map(call_to_v).collect())); }
        if let Some(i) = &self.tool_call_id { m.insert("tool_call_id".into(), V::str(i)); }
        if let Some(n) = &self.name { m.insert("name".into(), V::str(n)); }
        V::map(m)
    }
}

/// One tool call as template data: `arguments` parsed from its JSON string when needed.
fn call_to_v(c: &serde_json::Value) -> V {
    let mut c = c.clone();
    if let Some(f) = c.get_mut("function")
        && let Some(a) = f.get_mut("arguments")
        && let Some(s) = a.as_str()
    {
        *a = serde_json::from_str(s).unwrap_or_else(|_| serde_json::json!({}));
    }
    V::from_json(&c)
}

/// (name, arguments object) of a tool call in OpenAI or flat shape.
fn call_parts(c: &serde_json::Value) -> (String, serde_json::Value) {
    let f = c.get("function").unwrap_or(c);
    let name = f.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
    let args = match f.get("arguments") {
        Some(serde_json::Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| serde_json::json!({})),
        Some(v) => v.clone(),
        None => serde_json::json!({}),
    };
    (name, args)
}

/// Each tool's `function` object (accepts both `{type:function,function:{..}}` and the bare form).
fn tool_fns(tools: &[serde_json::Value]) -> Vec<serde_json::Value> {
    tools.iter().map(|t| t.get("function").cloned().unwrap_or_else(|| t.clone())).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builtin { ChatMl, Llama3, Gemma, Mistral, Plain }

impl Builtin {
    pub fn name(self) -> &'static str { match self { Builtin::ChatMl => "chatml", Builtin::Llama3 => "llama3", Builtin::Gemma => "gemma", Builtin::Mistral => "mistral", Builtin::Plain => "plain" } }

    /// Pick by the model family and the special tokens its vocabulary has.
    pub fn detect(family: &str, has_token: &dyn Fn(&str) -> bool) -> Builtin {
        if has_token("<|start_header_id|>") && has_token("<|eot_id|>") { Builtin::Llama3 }
        else if has_token("<|im_start|>") && has_token("<|im_end|>") { Builtin::ChatMl }
        else if has_token("<start_of_turn>") { Builtin::Gemma }
        else if has_token("[INST]") || family.starts_with("mistral") || family.starts_with("mixtral") { Builtin::Mistral }
        else if family.starts_with("gemma") { Builtin::Gemma }
        else if family.starts_with("qwen") { Builtin::ChatMl }
        else { Builtin::Plain }
    }

    pub fn render(self, msgs: &[Message], add_generation_prompt: bool, bos: &str, eos: &str) -> String {
        self.render_with(msgs, &[], add_generation_prompt, bos, eos)
    }

    /// Render with tool definitions. The plain families get the same layout their instruct models were trained on
    /// (Hermes/Qwen `<tools>`, Llama 3.1 JSON tools in the first user turn, Mistral `[AVAILABLE_TOOLS]`).
    pub fn render_with(self, msgs: &[Message], tools: &[serde_json::Value], add_generation_prompt: bool, bos: &str, eos: &str) -> String {
        if !tools.is_empty() || msgs.iter().any(|m| !m.tool_calls.is_empty() || m.role == "tool") { return self.render_tools(msgs, tools, add_generation_prompt, bos, eos); }
        let mut o = String::new();
        match self {
            Builtin::ChatMl => {
                for m in msgs { o.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content)); }
                if add_generation_prompt { o.push_str("<|im_start|>assistant\n"); }
            }
            Builtin::Llama3 => {
                o.push_str(bos);
                for m in msgs { o.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", m.role, m.content.trim())); }
                if add_generation_prompt { o.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n"); }
            }
            Builtin::Gemma => {
                o.push_str(bos);
                let mut system = String::new();
                let mut first_user = true;
                for m in msgs {
                    match m.role.as_str() {
                        "system" => { system = m.content.trim().to_string(); }
                        r => {
                            let role = if r == "assistant" { "model" } else { r };
                            let mut c = m.content.trim().to_string();
                            if first_user && r == "user" && !system.is_empty() { c = format!("{system}\n\n{c}"); first_user = false; }
                            o.push_str(&format!("<start_of_turn>{role}\n{c}<end_of_turn>\n"));
                        }
                    }
                }
                if add_generation_prompt { o.push_str("<start_of_turn>model\n"); }
            }
            Builtin::Mistral => {
                o.push_str(bos);
                let mut system = String::new();
                for m in msgs {
                    match m.role.as_str() {
                        "system" => system = m.content.clone(),
                        "assistant" => { o.push_str(&m.content); o.push_str(eos); }
                        _ => {
                            if system.is_empty() { o.push_str(&format!("[INST] {} [/INST]", m.content)); }
                            else { o.push_str(&format!("[INST] {}\n\n{} [/INST]", system, m.content)); system.clear(); }
                        }
                    }
                }
            }
            Builtin::Plain => {
                for m in msgs {
                    let label = match m.role.as_str() { "user" => "User", "assistant" => "Assistant", "system" => "System", r => r };
                    o.push_str(&format!("{label}: {}\n", m.content));
                }
                if add_generation_prompt { o.push_str("Assistant:"); }
            }
        }
        o
    }
}

const HERMES_TOOLS_HEAD: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const HERMES_TOOLS_TAIL: &str = "</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

/// `json.dumps(v, ensure_ascii=False)`: the layout chat templates produce with `| tojson`.
pub fn py_json(v: &serde_json::Value) -> String { to_json(&V::from_json(v), None, 0) }
fn py_json_indent(v: &serde_json::Value, n: usize) -> String { to_json(&V::from_json(v), Some(n), 0) }

pub fn hermes_block(tools: &[serde_json::Value]) -> String {
    let mut o = String::from(HERMES_TOOLS_HEAD);
    for t in tools { o.push('\n'); o.push_str(&py_json(t)); }
    o.push('\n');
    o.push_str(HERMES_TOOLS_TAIL);
    o
}

fn hermes_call(c: &serde_json::Value) -> String {
    let (name, args) = call_parts(c);
    format!("<tool_call>\n{}\n</tool_call>", py_json(&serde_json::json!({ "name": name, "arguments": args })))
}

impl Builtin {
    fn render_tools(self, msgs: &[Message], tools: &[serde_json::Value], add_generation_prompt: bool, bos: &str, eos: &str) -> String {
        let mut o = String::new();
        match self {
            Builtin::ChatMl | Builtin::Gemma | Builtin::Plain => {
                // Hermes layout for all three; Gemma/Plain only differ in the turn markers.
                let hermes = hermes_block(tools);
                let (sys, rest): (Option<&Message>, &[Message]) = match msgs.first() { Some(m) if m.role == "system" => (Some(m), &msgs[1..]), _ => (None, msgs) };
                let mut sys_text = sys.map_or(String::new(), |m| m.content.clone());
                if !tools.is_empty() {
                    if sys_text.is_empty() && self == Builtin::ChatMl { sys_text = "You are a helpful assistant.".into(); }
                    if !sys_text.is_empty() { sys_text.push_str("\n\n"); }
                    sys_text.push_str(&hermes);
                }
                let turn = |role: &str, text: &str| -> String { match self {
                    Builtin::ChatMl => format!("<|im_start|>{role}\n{text}<|im_end|>\n"),
                    Builtin::Gemma => format!("<start_of_turn>{}\n{}<end_of_turn>\n", if role == "assistant" { "model" } else { "user" }, text.trim()),
                    _ => format!("{}: {text}\n", match role { "assistant" => "Assistant", "system" => "System", _ => "User" }),
                } };
                if self == Builtin::Gemma { o.push_str(bos); }
                if !sys_text.is_empty() { o.push_str(&if self == Builtin::Gemma { turn("user", &sys_text) } else { turn("system", &sys_text) }); }
                let mut i = 0;
                while i < rest.len() {
                    let m = &rest[i];
                    match m.role.as_str() {
                        "assistant" => {
                            let mut t = m.content.clone();
                            for c in &m.tool_calls { if !t.is_empty() { t.push('\n'); } t.push_str(&hermes_call(c)); }
                            o.push_str(&turn("assistant", &t));
                        }
                        "tool" => {
                            // consecutive tool results share one user turn
                            let mut t = String::new();
                            while i < rest.len() && rest[i].role == "tool" {
                                if !t.is_empty() { t.push('\n'); }
                                t.push_str(&format!("<tool_response>\n{}\n</tool_response>", rest[i].content));
                                i += 1;
                            }
                            o.push_str(&turn("user", &t));
                            continue;
                        }
                        r => o.push_str(&turn(r, &m.content)),
                    }
                    i += 1;
                }
                if add_generation_prompt { o.push_str(match self { Builtin::ChatMl => "<|im_start|>assistant\n", Builtin::Gemma => "<start_of_turn>model\n", _ => "Assistant:" }); }
            }
            Builtin::Llama3 => {
                o.push_str(bos);
                let sys = msgs.first().filter(|m| m.role == "system");
                o.push_str("<|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\n");
                if let Some(m) = sys { o.push('\n'); o.push_str(m.content.trim()); }
                o.push_str("<|eot_id|>");
                let rest = if sys.is_some() { &msgs[1..] } else { msgs };
                let mut first_user = true;
                for m in rest {
                    match m.role.as_str() {
                        "user" if first_user && !tools.is_empty() => {
                            first_user = false;
                            let mut t = String::from("Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables.\n\n");
                            for f in tool_fns(tools) { t.push_str(&py_json_indent(&f, 4)); t.push_str("\n\n"); }
                            t.push_str(m.content.trim());
                            o.push_str(&format!("<|start_header_id|>user<|end_header_id|>\n\n{t}<|eot_id|>"));
                        }
                        "assistant" if !m.tool_calls.is_empty() => {
                            o.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                            for c in &m.tool_calls { let (n, a) = call_parts(c); o.push_str(&py_json(&serde_json::json!({ "name": n, "parameters": a }))); }
                            o.push_str("<|eot_id|>");
                        }
                        "tool" => o.push_str(&format!("<|start_header_id|>ipython<|end_header_id|>\n\n{}<|eot_id|>", py_json(&serde_json::Value::String(m.content.trim().to_string())))),
                        r => { if r == "user" { first_user = false; } o.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", r, m.content.trim())); }
                    }
                }
                if add_generation_prompt { o.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n"); }
            }
            Builtin::Mistral => {
                o.push_str(bos);
                let mut system = String::new();
                let last_user = msgs.iter().rposition(|m| m.role == "user");
                for (i, m) in msgs.iter().enumerate() {
                    match m.role.as_str() {
                        "system" => system = m.content.clone(),
                        "user" => {
                            if Some(i) == last_user && !tools.is_empty() { o.push_str(&format!("[AVAILABLE_TOOLS] {}[/AVAILABLE_TOOLS]", py_json(&serde_json::Value::Array(tools.to_vec())))); }
                            if system.is_empty() { o.push_str(&format!("[INST] {}[/INST]", m.content)); }
                            else { o.push_str(&format!("[INST] {}\n\n{}[/INST]", system, m.content)); system.clear(); }
                        }
                        "assistant" => {
                            if m.tool_calls.is_empty() { o.push_str(&m.content); }
                            else {
                                let calls: Vec<serde_json::Value> = m.tool_calls.iter().enumerate().map(|(k, c)| { let (n, a) = call_parts(c); let id = c.get("id").and_then(|x| x.as_str()).map_or_else(|| format!("call{k:05}"), str::to_string); serde_json::json!({ "name": n, "arguments": a, "id": id }) }).collect();
                                o.push_str(&format!("[TOOL_CALLS] {}", py_json(&serde_json::Value::Array(calls))));
                            }
                            o.push_str(eos);
                        }
                        "tool" => o.push_str(&format!("[TOOL_RESULTS] {}[/TOOL_RESULTS]", py_json(&serde_json::json!({ "content": m.content, "call_id": m.tool_call_id.clone().unwrap_or_default() })))),
                        _ => {}
                    }
                }
            }
        }
        o
    }
}

/// How to turn a conversation into the model's prompt text.
#[derive(Clone, Debug)]
pub enum ChatTemplate { Jinja(Template), Builtin(Builtin) }

impl ChatTemplate {
    /// Parse the model's own template; fall back to a built-in one when it cannot be parsed.
    /// The second value is a warning to show the user in that case.
    pub fn from_source(src: Option<&str>, fallback: Builtin) -> (ChatTemplate, Option<String>) {
        match src.map(str::trim).filter(|s| !s.is_empty()) {
            Some(s) => match Template::parse(s) {
                Ok(t) => (ChatTemplate::Jinja(t), None),
                Err(e) => (ChatTemplate::Builtin(fallback), Some(format!("{e}; using the built-in `{}` template instead", fallback.name()))),
            },
            None => (ChatTemplate::Builtin(fallback), None),
        }
    }

    pub fn render(&self, msgs: &[Message], add_generation_prompt: bool, bos: &str, eos: &str) -> R<String> {
        self.render_with(msgs, &[], add_generation_prompt, bos, eos)
    }

    /// Render with OpenAI-shaped `tools` (each `{type:"function", function:{name, description, parameters}}`).
    pub fn render_with(&self, msgs: &[Message], tools: &[serde_json::Value], add_generation_prompt: bool, bos: &str, eos: &str) -> R<String> {
        match self {
            ChatTemplate::Builtin(b) => Ok(b.render_with(msgs, tools, add_generation_prompt, bos, eos)),
            ChatTemplate::Jinja(t) => {
                let ms: Vec<V> = msgs.iter().map(Message::to_v).collect();
                let mut vars: BTreeMap<String, V> = [
                    ("messages", V::list(ms)), ("add_generation_prompt", V::Bool(add_generation_prompt)),
                    ("bos_token", V::str(bos)), ("eos_token", V::str(eos)),
                ].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
                if !tools.is_empty() { vars.insert("tools".into(), V::list(tools.iter().map(V::from_json).collect())); }
                t.render(&vars)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(src: &str, vars: &[(&str, V)]) -> R<String> {
        Template::parse(src)?.render(&vars.iter().map(|(k, v)| (k.to_string(), v.clone())).collect())
    }
    fn msgs() -> Vec<Message> { vec![Message::new("system", "Be brief."), Message::new("user", "Hi  "), Message::new("assistant", "Hello!"), Message::new("user", "Bye")] }
    fn no_sys() -> Vec<Message> { vec![Message::new("user", "Hi  "), Message::new("assistant", "Hello!"), Message::new("user", "Bye")] }

    const CHATML: &str = "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";
    const LLAMA3: &str = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}";
    const GEMMA: &str = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}";
    const MISTRAL: &str = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";

    #[test]
    fn builtin_templates_give_the_known_strings() {
        let c = Builtin::ChatMl.render(&msgs(), true, "", "");
        assert_eq!(c, "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi  <|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n");
        let l = Builtin::Llama3.render(&msgs(), true, "<|begin_of_text|>", "<|eot_id|>");
        assert_eq!(l, "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBye<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
        let g = Builtin::Gemma.render(&msgs(), true, "<bos>", "<eos>");
        assert_eq!(g, "<bos><start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nBye<end_of_turn>\n<start_of_turn>model\n");
        let m = Builtin::Mistral.render(&msgs(), false, "<s>", "</s>");
        assert_eq!(m, "<s>[INST] Be brief.\n\nHi   [/INST]Hello!</s>[INST] Bye [/INST]");
        let p = Builtin::Plain.render(&no_sys(), true, "", "");
        assert_eq!(p, "User: Hi  \nAssistant: Hello!\nUser: Bye\nAssistant:");
    }

    #[test]
    fn real_world_jinja_templates_match_the_builtin_renderings() {
        let t = |s: &str| ChatTemplate::Jinja(Template::parse(s).unwrap());
        assert_eq!(t(CHATML).render(&msgs(), true, "", "").unwrap(), Builtin::ChatMl.render(&msgs(), true, "", ""));
        assert_eq!(t(CHATML).render(&msgs(), false, "", "").unwrap(), Builtin::ChatMl.render(&msgs(), false, "", ""));
        assert_eq!(t(LLAMA3).render(&msgs(), true, "<|begin_of_text|>", "").unwrap(), Builtin::Llama3.render(&msgs(), true, "<|begin_of_text|>", ""));
        assert_eq!(t(GEMMA).render(&no_sys(), true, "<bos>", "").unwrap(), "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nBye<end_of_turn>\n<start_of_turn>model\n");
        assert_eq!(t(MISTRAL).render(&no_sys(), false, "<s>", "</s>").unwrap(), "<s>[INST] Hi   [/INST]Hello!</s>[INST] Bye [/INST]");
    }

    #[test]
    fn raise_exception_surfaces_the_message() {
        let t = ChatTemplate::Jinja(Template::parse(GEMMA).unwrap());
        let e = t.render(&msgs(), true, "<bos>", "").unwrap_err();
        assert_eq!(e, TemplateError::Raised("System role not supported".into()));
        let two_users = vec![Message::new("user", "a"), Message::new("user", "b")];
        assert!(matches!(t.render(&two_users, true, "", ""), Err(TemplateError::Raised(m)) if m.contains("alternate")));
    }

    #[test]
    fn trim_blocks_and_lstrip_blocks_match_the_hugging_face_environment() {
        let src = "a\n  {% if true %}\nb\n  {% endif %}\nc\n";
        assert_eq!(render(src, &[]).unwrap(), "a\nb\nc\n");
        let src = "{% for i in [1,2] -%}\n  {{ i }}\n{% endfor %}";
        assert_eq!(render(src, &[]).unwrap(), "1\n2\n");
        assert_eq!(render("x {{- ' y ' -}} z", &[]).unwrap(), "x y z");
        assert_eq!(render("{# gone #}k", &[]).unwrap(), "k");
    }

    #[test]
    fn loops_scopes_and_namespaces() {
        let src = "{% set ns = namespace(n=0, s='') %}{% for x in items %}{% set ns.n = ns.n + x %}{% set ns.s = ns.s ~ x %}{% endfor %}{{ ns.n }}/{{ ns.s }}";
        assert_eq!(render(src, &[("items", V::list(vec![V::Int(1), V::Int(2), V::Int(3)]))]).unwrap(), "6/123");
        // a plain `set` inside a loop does not leak out
        assert_eq!(render("{% set a = 1 %}{% for x in [1] %}{% set a = 2 %}{% endfor %}{{ a }}", &[]).unwrap(), "1");
        assert_eq!(render("{% for x in [] %}a{% else %}empty{% endfor %}", &[]).unwrap(), "empty");
        assert_eq!(render("{% for x in [1,2,3,4] if x is even %}{{ loop.index }}:{{ x }}{{ ',' if not loop.last }}{% endfor %}", &[]).unwrap(), "1:2,2:4");
        assert_eq!(render("{% for x in [1,2,3,4] %}{% if x == 2 %}{% continue %}{% endif %}{% if x == 4 %}{% break %}{% endif %}{{ x }}{% endfor %}", &[]).unwrap(), "13");
        assert_eq!(render("{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}", &[("d", V::from_json(&serde_json::json!({"a": 1, "b": "x"})))]).unwrap(), "a=1;b=x;");
    }

    #[test]
    fn expressions_filters_and_methods() {
        let r = |s: &str| render(s, &[("m", V::from_json(&serde_json::json!([{"role": "user", "content": " hi "}, {"role": "assistant", "content": "yo"}])))]).unwrap();
        assert_eq!(r("{{ m[0].content | trim | upper }}"), "HI");
        assert_eq!(r("{{ m | length }}{{ m[1:] | length }}{{ m[-1]['role'] }}"), "21assistant");
        assert_eq!(r("{{ 'a' if m else 'b' }}{{ 'x' not in 'abc' }}{{ 7 // 2 }}{{ -3 % 5 }}"), "aTrue32");
        assert_eq!(r("{{ nope is defined }}{{ nope | default('d') }}{{ m[5] is defined }}"), "FalsedFalse");
        assert_eq!(r("{{ 'a,b,c'.split(',') | length }}{{ ' x '.strip() }}{{ 'abc'.startswith('ab') }}{{ ', '.join(['p', 'q']) }}"), "3xTruep, q");
        assert_eq!(r("{{ {'a': [1, 'x'], 'b': none} | tojson }}"), "{\"a\": [1, \"x\"], \"b\": null}");
        assert_eq!(r("{{ [1,2] | tojson(indent=2) }}"), "[\n  1,\n  2\n]");
        assert_eq!(r("{{ (1, 2) | length }}{{ range(3) | list | length }}{{ 'ab' * 2 }}"), "23abab");
        assert_eq!(r("{{ 1 < 2 and 'a' < 'b' }}{{ not true or false }}"), "TrueFalse");
        assert_eq!(strftime("%d %b %Y %-d %B", 1_721_952_000), "26 Jul 2024 26 July");
    }

    #[test]
    fn syntax_errors_are_errors_not_panics() {
        for bad in ["{{", "{% if %}", "{% endfor %}", "{% for x in %}", "{{ 1 + }}", "{% macro f() %}", "{{ 1.5 }}", "{{ a.[0] }}", "{% if x %}", "{{ 'x }}",
            "{% set x %}", "{{ ) }}", "{% else %}", "{% elif x %}", "{{ x | }}", "{{ [1, }}"] {
            assert!(Template::parse(bad).is_err(), "{bad}");
        }
        assert!(render("{{ 1 // 0 }}", &[]).is_err());
        assert!(render("{{ 'a' - 1 }}", &[]).is_err());
        assert!(render("{{ x | nosuch }}", &[]).is_err());
        assert!(render("{{ f() }}", &[]).is_err());
        assert!(render("{% break %}", &[]).is_err());
    }

    #[test]
    fn hostile_templates_hit_the_limits() {
        let e = render("{% for i in range(100000) %}{% for j in range(100000) %}x{% endfor %}{% endfor %}", &[]).unwrap_err();
        assert!(matches!(e, TemplateError::Limit(_)), "{e}");
        let e = render("{% set s = 'abcdefgh' %}{% for i in range(40) %}{% set s = s * 5 %}{{ s }}{% endfor %}", &[]).unwrap_err();
        assert!(matches!(e, TemplateError::Limit(_) | TemplateError::Runtime(_)), "{e}");
        assert!(matches!(render("{{ range(10000000) }}", &[]).unwrap_err(), TemplateError::Limit(_)));
        let deep = format!("{}1{}", "(".repeat(5000), ")".repeat(5000));
        assert!(Template::parse(&format!("{{{{ {deep} }}}}")).is_err());
        let deep = format!("{}x{}", "[".repeat(5000), "]".repeat(5000));
        assert!(Template::parse(&format!("{{{{ {deep} }}}}")).is_err());
        let blocks = "{% if true %}".repeat(500) + &"{% endif %}".repeat(500);
        assert!(Template::parse(&blocks).is_err());
        assert!(Template::parse(&"-".repeat(2 << 20)).is_err());
        let nots = "not ".repeat(5000);
        assert!(Template::parse(&format!("{{{{ {nots} true }}}}")).is_err());
        let negs = "-".repeat(5000);
        assert!(Template::parse(&format!("{{{{ {negs}1 }}}}")).is_err());
    }

    #[test]
    fn fuzzed_templates_never_panic() {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let seeds = [CHATML, LLAMA3, GEMMA, MISTRAL, "{% for x in y %}{{ x | tojson }}{% endfor %}"];
        let alphabet: Vec<char> = "{}%#-|.,:()[]'\" \n\\abz019+*/~=<>!".chars().collect();
        for round in 0..3000 {
            let mut t: Vec<char> = seeds[round % seeds.len()].chars().collect();
            for _ in 0..(next() % 6 + 1) {
                if t.is_empty() { break; }
                let at = (next() as usize) % t.len();
                match next() % 3 { 0 => { t.remove(at); } 1 => t.insert(at, alphabet[(next() as usize) % alphabet.len()]), _ => t[at] = alphabet[(next() as usize) % alphabet.len()] }
            }
            let src: String = t.into_iter().collect();
            if let Ok(tpl) = Template::parse(&src) {
                let _ = ChatTemplate::Jinja(tpl).render(&msgs(), true, "<s>", "</s>");
            }
        }
        // pure noise
        for _ in 0..2000 {
            let src: String = (0..(next() % 40)).map(|_| alphabet[(next() as usize) % alphabet.len()]).collect();
            let _ = Template::parse(&src).and_then(|t| t.render(&BTreeMap::new()));
        }
    }

    #[test]
    fn detection_prefers_the_vocabulary_over_the_family_name() {
        let has = |set: &'static [&'static str]| move |t: &str| set.contains(&t);
        assert_eq!(Builtin::detect("llama", &has(&["<|start_header_id|>", "<|eot_id|>"])), Builtin::Llama3);
        assert_eq!(Builtin::detect("qwen2", &has(&["<|im_start|>", "<|im_end|>"])), Builtin::ChatMl);
        assert_eq!(Builtin::detect("gemma2", &has(&["<start_of_turn>"])), Builtin::Gemma);
        assert_eq!(Builtin::detect("llama", &has(&["[INST]"])), Builtin::Mistral);
        assert_eq!(Builtin::detect("mistral", &has(&[])), Builtin::Mistral);
        assert_eq!(Builtin::detect("falcon", &has(&[])), Builtin::Plain);
    }

    #[test]
    fn an_unparsable_model_template_falls_back_with_a_warning() {
        let (t, w) = ChatTemplate::from_source(Some("{% macro x() %}"), Builtin::ChatMl);
        assert!(matches!(t, ChatTemplate::Builtin(Builtin::ChatMl)));
        assert!(w.unwrap().contains("chatml"));
        let (t, w) = ChatTemplate::from_source(None, Builtin::Gemma);
        assert!(matches!(t, ChatTemplate::Builtin(Builtin::Gemma)) && w.is_none());
    }

    // ---------------------------------------------------------------------- tools

    fn weather_tool() -> serde_json::Value {
        serde_json::json!({"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}})
    }

    fn assistant_call() -> Message {
        let mut m = Message::new("assistant", "");
        m.tool_calls = vec![serde_json::json!({"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"unit\": \"c\"}"}})];
        m
    }

    fn tool_result(text: &str) -> Message {
        let mut m = Message::new("tool", text);
        m.tool_call_id = Some("call_1".into());
        m
    }

    const QWEN25: &str = r#"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"#;

    #[test]
    fn qwen25_tool_template_matches_the_builtin_hermes_layout() {
        let (t, w) = ChatTemplate::from_source(Some(QWEN25), Builtin::ChatMl);
        assert!(w.is_none(), "{w:?}");
        let tools = [weather_tool()];
        let msgs = vec![Message::new("system", "Be brief."), Message::new("user", "Weather in Paris?"), assistant_call(), tool_result("18 C"), tool_result("sunny"), Message::new("user", "thanks")];
        let jinja = t.render_with(&msgs, &tools, true, "", "<|im_end|>").unwrap();
        let builtin = ChatTemplate::Builtin(Builtin::ChatMl).render_with(&msgs, &tools, true, "", "<|im_end|>").unwrap();
        assert_eq!(jinja, builtin);
        assert!(jinja.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"unit\": {\"type\": \"string\", \"enum\": [\"c\", \"f\"]}}, \"required\": [\"city\"]}}}\n</tools>"));
        assert!(jinja.contains("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"unit\": \"c\"}}\n</tool_call><|im_end|>"));
        assert!(jinja.contains("<|im_start|>user\n<tool_response>\n18 C\n</tool_response>\n<tool_response>\nsunny\n</tool_response><|im_end|>"));
        // without tools nothing about tools appears
        let plain = t.render(&[Message::new("user", "hi")], true, "", "").unwrap();
        assert_eq!(plain, "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    const LLAMA31: &str = r#"{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
        {%- else %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"#;

    #[test]
    fn llama31_tool_template_renders_like_the_real_one() {
        let (t, w) = ChatTemplate::from_source(Some(LLAMA31), Builtin::Llama3);
        assert!(w.is_none(), "{w:?}");
        let tools = [weather_tool()];
        let msgs = vec![Message::new("system", "You are helpful."), Message::new("user", "Weather in Paris?"), assistant_call(), tool_result("18 C")];
        let s = t.render_with(&msgs, &tools, true, "<|begin_of_text|>", "<|eot_id|>").unwrap();
        assert!(s.starts_with("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following functions"), "{s}");
        // the tool is pretty-printed with 4 spaces, in the user turn
        assert!(s.contains("{\n    \"type\": \"function\",\n    \"function\": {\n        \"name\": \"get_weather\","), "{s}");
        assert!(s.contains("Weather in Paris?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\", \"unit\": \"c\"}}<|eot_id|>"), "{s}");
        // (a string is "iterable" in Jinja, so the real template quotes it with tojson)
        assert!(s.ends_with("<|start_header_id|>ipython<|end_header_id|>\n\n\"18 C\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"), "{s}");
        // two parallel calls are refused by this template, and the error reaches the caller
        let mut two = assistant_call();
        two.tool_calls.push(two.tool_calls[0].clone());
        let e = t.render_with(&[Message::new("user", "x"), two], &tools, true, "", "").unwrap_err();
        assert!(e.to_string().contains("single tool-calls"), "{e}");
        // the builtin Llama 3.1 layout agrees with the real template on the shared parts
        let b = ChatTemplate::Builtin(Builtin::Llama3).render_with(&msgs, &tools, true, "<|begin_of_text|>", "<|eot_id|>").unwrap();
        assert!(b.contains("{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\", \"unit\": \"c\"}}<|eot_id|>") && b.contains("ipython<|end_header_id|>\n\n\"18 C\"<|eot_id|>"));
    }

    #[test]
    fn mistral_style_template_uses_selectattr_and_namespace() {
        let src = r#"{{- bos_token }}
{%- set user_messages = messages | selectattr("role", "equalto", "user") | list %}
{%- set ns = namespace(n=0, names=[]) %}
{%- for m in messages %}{% set ns.n = ns.n + loop.index %}{% if m.role == "assistant" %}{% set _ = ns.names.append(m.content | upper) %}{% endif %}{% endfor %}
{%- if tools is defined and tools %}[AVAILABLE_TOOLS] {{ tools | map(attribute="function.name") | join(",") }}[/AVAILABLE_TOOLS]{% endif %}
{{- user_messages | length }}|{{ ns.n }}|{{ ns.names | join("+") }}|{{ messages | map(attribute="role") | unique | sort | join(",") }}"#;
        let t = Template::parse(src).unwrap();
        let mut vars = BTreeMap::new();
        vars.insert("bos_token".to_string(), V::str("<s>"));
        vars.insert("messages".to_string(), V::from_json(&serde_json::json!([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}])));
        vars.insert("tools".to_string(), V::from_json(&serde_json::json!([weather_tool()])));
        assert_eq!(t.render(&vars).unwrap(), "<s>[AVAILABLE_TOOLS] get_weather[/AVAILABLE_TOOLS]2|6|B|assistant,user");
    }

    #[test]
    fn tojson_keeps_insertion_order_and_python_layout() {
        let t = Template::parse("{{ d | tojson }}|{{ d | tojson(indent=2) }}|{{ {'z': 1, 'a': [1, 2, {'k': none}]} | tojson }}").unwrap();
        let mut vars = BTreeMap::new();
        vars.insert("d".to_string(), V::from_json(&serde_json::json!({"zebra": 1, "apple": {"mid": "é\n", "beta": true}, "list": []})));
        assert_eq!(t.render(&vars).unwrap(), "{\"zebra\": 1, \"apple\": {\"mid\": \"é\\n\", \"beta\": true}, \"list\": []}|{\n  \"zebra\": 1,\n  \"apple\": {\n    \"mid\": \"é\\n\",\n    \"beta\": true\n  },\n  \"list\": []\n}|{\"z\": 1, \"a\": [1, 2, {\"k\": null}]}");
    }

    #[test]
    fn more_filters_and_string_methods_used_by_real_templates() {
        let t = Template::parse("{{ 'a,b,c'.split(',')[-1] }}|{{ ' x '.strip() }}|{{ 'abc'.startswith('ab') }}|{{ 'abcabc'.count('b') }}|{{ 'abc'.find('c') }}|{{ 'ab\\ncd'.splitlines() | length }}|{{ [3, 1, 2] | sort | join('') }}|{{ [3, 1, 2] | sum }}|{{ [1, 2, 3] | max }}|{{ 'a\\nb' | indent(2) }}|{{ items | rejectattr('on') | map(attribute='n') | join('') }}|{{ items | selectattr('n', 'gt', 1) | list | length }}|{{ 5 is lt(7) }}|{{ 'a' is in(['a', 'b']) }}").unwrap();
        let mut vars = BTreeMap::new();
        vars.insert("items".to_string(), V::from_json(&serde_json::json!([{"n": 1, "on": true}, {"n": 2, "on": false}, {"n": 3}])));
        assert_eq!(t.render(&vars).unwrap(), "c|x|True|2|2|2|123|6|3|a\n  b|23|2|True|True");
    }

    #[test]
    fn tools_with_a_mistral_style_builtin() {
        let tools = [weather_tool()];
        let msgs = vec![Message::new("user", "Weather?"), assistant_call(), tool_result("18 C"), Message::new("user", "ok")];
        let s = ChatTemplate::Builtin(Builtin::Mistral).render_with(&msgs, &tools, true, "<s>", "</s>").unwrap();
        assert!(s.starts_with("<s>[INST] Weather?[/INST][TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"unit\": \"c\"}, \"id\": \"call_1\"}]</s>[TOOL_RESULTS] {\"content\": \"18 C\", \"call_id\": \"call_1\"}[/TOOL_RESULTS]"), "{s}");
        assert!(s.contains("[AVAILABLE_TOOLS] [{\"type\": \"function\"") && s.ends_with("[/AVAILABLE_TOOLS][INST] ok[/INST]"), "{s}");
        // without tools and tool turns the old layout is untouched
        assert_eq!(ChatTemplate::Builtin(Builtin::Mistral).render(&[Message::new("user", "hi")], true, "<s>", "</s>").unwrap(), "<s>[INST] hi [/INST]");
    }

    #[test]
    fn gemma_and_plain_get_hermes_style_tools() {
        let tools = [weather_tool()];
        let g = ChatTemplate::Builtin(Builtin::Gemma).render_with(&[Message::new("user", "hi")], &tools, true, "<bos>", "").unwrap();
        assert!(g.starts_with("<bos><start_of_turn>user\n# Tools") && g.contains("<tools>\n{\"type\": \"function\"") && g.ends_with("<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n"), "{g}");
        let p = ChatTemplate::Builtin(Builtin::Plain).render_with(&[Message::new("user", "hi")], &tools, true, "", "").unwrap();
        assert!(p.starts_with("System: # Tools") && p.ends_with("User: hi\nAssistant:"), "{p}");
    }
}
