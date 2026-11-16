//! Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//!
//! JSON Canonicalization (RFC 8785) for receipts: a strict JSON reader that keeps key order and rejects duplicate
//! keys, lone surrogates, leading zeros and anything nested deeper than 64 levels, and a writer that emits the one
//! canonical byte form (keys sorted by UTF-16 code units, ECMAScript number formatting).
//!
//! Differences from the original: a typed error instead of `anyhow`; string decoding and duplicate-key detection
//! are linear (the original re-validated the rest of the input for every character of a string and compared every
//! key with every earlier one); the number grammar is the JSON one (no `007`, `1.`, `.5`, `-`); the writer is
//! depth-limited too, so a value built by hand cannot overflow the stack.

use std::collections::HashSet;
use std::fmt;

/// Nesting limit for both reader and writer.
pub const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JcsError(pub String);

impl fmt::Display for JcsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for JcsError {}

type Result<T> = std::result::Result<T, JcsError>;

fn err<T>(m: impl Into<String>) -> Result<T> {
    Err(JcsError(m.into()))
}

#[derive(Debug, Clone, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::Obj(v) => v.iter().find(|(n, _)| n == k).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            J::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[J]> {
        match self {
            J::Arr(a) => Some(a),
            _ => None,
        }
    }

    /// A non-negative integer that a double holds exactly (below 2^53).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            J::Num(n) if *n >= 0.0 && n.fract() == 0.0 && *n < 9007199254740992.0 => {
                Some(*n as u64)
            }
            _ => None,
        }
    }

    /// Remove a property from an object, returning it.
    pub fn take(&mut self, k: &str) -> Option<J> {
        if let J::Obj(v) = self
            && let Some(i) = v.iter().position(|(n, _)| n == k)
        {
            return Some(v.remove(i).1);
        }
        None
    }
}

/// From a `serde_json` value (numbers that are not finite doubles are refused).
pub fn from_value(v: &serde_json::Value) -> Result<J> {
    from_value_at(v, 0)
}

fn from_value_at(v: &serde_json::Value, depth: usize) -> Result<J> {
    use serde_json::Value as V;
    if depth > MAX_DEPTH {
        return err("nesting too deep");
    }
    Ok(match v {
        V::Null => J::Null,
        V::Bool(b) => J::Bool(*b),
        V::Number(n) => match n.as_f64() {
            Some(f) if f.is_finite() => J::Num(f),
            _ => return err("number out of range"),
        },
        V::String(s) => J::Str(s.clone()),
        V::Array(a) => J::Arr(
            a.iter()
                .map(|x| from_value_at(x, depth + 1))
                .collect::<Result<_>>()?,
        ),
        V::Object(o) => {
            let mut out = Vec::with_capacity(o.len());
            for (k, x) in o {
                out.push((k.clone(), from_value_at(x, depth + 1)?));
            }
            J::Obj(out)
        }
    })
}

// ------------------------------------------------------------------ reader

pub fn parse(text: &str) -> Result<J> {
    let mut p = P {
        s: text,
        b: text.as_bytes(),
        i: 0,
    };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.i != p.b.len() {
        return err("trailing characters after the JSON value");
    }
    Ok(v)
}

struct P<'a> {
    s: &'a str,
    b: &'a [u8],
    i: usize,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<()> {
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }

    fn lit(&mut self, s: &str, v: J) -> Result<J> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(v)
        } else {
            err(format!("invalid literal at byte {}", self.i))
        }
    }

    fn value(&mut self, depth: usize) -> Result<J> {
        if depth > MAX_DEPTH {
            return err("nesting too deep");
        }
        match self.b.get(self.i) {
            Some(b'n') => self.lit("null", J::Null),
            Some(b't') => self.lit("true", J::Bool(true)),
            Some(b'f') => self.lit("false", J::Bool(false)),
            Some(b'"') => Ok(J::Str(self.string()?)),
            Some(b'[') => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(J::Arr(v));
                }
                loop {
                    self.ws();
                    v.push(self.value(depth + 1)?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(J::Arr(v));
                        }
                        _ => return err(format!("expected ',' or ']' at byte {}", self.i)),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut v: Vec<(String, J)> = Vec::new();
                let mut seen: HashSet<String> = HashSet::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(J::Obj(v));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    if !seen.insert(k.clone()) {
                        return err(format!("duplicate property name {k:?}"));
                    }
                    self.ws();
                    self.eat(b':')?;
                    self.ws();
                    v.push((k, self.value(depth + 1)?));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(J::Obj(v));
                        }
                        _ => return err(format!("expected ',' or '}}' at byte {}", self.i)),
                    }
                }
            }
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            _ => err(format!("unexpected input at byte {}", self.i)),
        }
    }

    /// `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`
    fn number(&mut self) -> Result<J> {
        let start = self.i;
        let digits = |p: &mut Self| {
            let s = p.i;
            while p.i < p.b.len() && p.b[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i - s
        };
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        match self.b.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                digits(self);
            }
            _ => return err(format!("bad number at byte {start}")),
        }
        if self.b.get(self.i) == Some(&b'.') {
            self.i += 1;
            if digits(self) == 0 {
                return err(format!("bad number at byte {start}"));
            }
        }
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if digits(self) == 0 {
                return err(format!("bad number at byte {start}"));
            }
        }
        let t = &self.s[start..self.i];
        match t.parse::<f64>() {
            Ok(n) if n.is_finite() => Ok(J::Num(n)),
            _ => err(format!("number out of range: {t}")),
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        let s = match self.b.get(self.i..self.i + 4) {
            Some(s) => s,
            None => return err("short \\u escape"),
        };
        let mut v = 0u32;
        for &c in s {
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return err("bad \\u escape"),
            };
            v = v << 4 | d as u32;
        }
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            // copy the run of ordinary bytes up to the next quote, backslash or control character; those
            // three are ASCII, so both ends of the run are character boundaries of the &str
            let run_start = self.i;
            while self.i < self.b.len() && !matches!(self.b[self.i], b'"' | b'\\' | 0..=0x1f) {
                self.i += 1;
            }
            out.push_str(&self.s[run_start..self.i]);
            let c = match self.b.get(self.i) {
                Some(c) => *c,
                None => return err("unterminated string"),
            };
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    let e = match self.b.get(self.i) {
                        Some(e) => *e,
                        None => return err("bad escape"),
                    };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let u = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&u) {
                                if self.b.get(self.i..self.i + 2) != Some(b"\\u") {
                                    return err("lone surrogate in string");
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return err("lone surrogate in string");
                                }
                                0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&u) {
                                return err("lone surrogate in string");
                            } else {
                                u
                            };
                            match char::from_u32(cp) {
                                Some(ch) => out.push(ch),
                                None => return err("bad code point"),
                            }
                        }
                        _ => return err("bad escape"),
                    }
                }
                _ => return err("control character in string"),
            }
        }
    }
}

// ------------------------------------------------------------------ writer

/// Canonical bytes. `ints_only` is the receipt-v2 rule: numbers must be integers below 2^53 in magnitude.
pub fn canonical(v: &J, ints_only: bool) -> Result<Vec<u8>> {
    let mut s = String::new();
    write(v, ints_only, &mut s, 0)?;
    Ok(s.into_bytes())
}

fn write(v: &J, ints_only: bool, out: &mut String, depth: usize) -> Result<()> {
    if depth > MAX_DEPTH {
        return err("nesting too deep");
    }
    match v {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Num(n) => {
            if ints_only && (n.fract() != 0.0 || n.abs() >= 9007199254740992.0) {
                return err(format!(
                    "receipt numbers must be integers below 2^53; put {n} in a string"
                ));
            }
            out.push_str(&number(*n)?);
        }
        J::Str(s) => string(s, out),
        J::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write(x, ints_only, out, depth + 1)?;
            }
            out.push(']');
        }
        J::Obj(o) => {
            let mut keys: Vec<(&String, &J)> = o.iter().map(|(k, v)| (k, v)).collect();
            keys.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
            for w in keys.windows(2) {
                if w[0].0 == w[1].0 {
                    return err(format!("duplicate property name {:?}", w[0].0));
                }
            }
            out.push('{');
            for (i, (k, x)) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                string(k, out);
                out.push(':');
                write(x, ints_only, out, depth + 1)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// ECMAScript Number-to-string for a finite double (shortest round-trip digits).
pub fn number(n: f64) -> Result<String> {
    if !n.is_finite() {
        return err("NaN and Infinity cannot be represented");
    }
    if n == 0.0 {
        return Ok("0".into());
    }
    let sign = if n < 0.0 { "-" } else { "" };
    let e = format!("{:e}", n.abs()); // shortest digits, e.g. "1.2345e-7"
    let (mant, exp) = match e.split_once('e') {
        Some(p) => p,
        None => return err("unformattable number"),
    };
    let exp: i32 = match exp.parse() {
        Ok(x) => x,
        Err(_) => return err("unformattable number"),
    };
    let mut digits: String = mant.chars().filter(|c| *c != '.').collect();
    tie_to_even(n.abs(), &mut digits, exp);
    let k = digits.len() as i32;
    let nn = exp + 1;
    let body = if k <= nn && nn <= 21 {
        format!("{digits}{}", "0".repeat((nn - k) as usize))
    } else if 0 < nn && nn <= 21 {
        format!("{}.{}", &digits[..nn as usize], &digits[nn as usize..])
    } else if -6 < nn && nn <= 0 {
        format!("0.{}{digits}", "0".repeat((-nn) as usize))
    } else {
        let m = if k == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        format!(
            "{m}e{}{}",
            if nn - 1 < 0 { "-" } else { "+" },
            (nn - 1).abs()
        )
    };
    Ok(format!("{sign}{body}"))
}

/// ECMAScript picks, among the shortest digit strings that read back as `n`, the one closest to `n`, and the even one
/// when two are equally close. Rust's shortest formatting rounds such a tie up; this puts the even digit back.
fn tie_to_even(n: f64, digits: &mut String, exp: i32) {
    let last = match digits.bytes().last() {
        Some(b) if b > b'0' && (b - b'0') % 2 == 1 => b,
        _ => return,
    };
    let exact = format!("{n:.45e}");
    let Some((m, _)) = exact.split_once('e') else {
        return;
    };
    let ex: String = m.chars().filter(|c| *c != '.').collect();
    let k = digits.len();
    let tail = match ex.get(k..) {
        Some(t) => t,
        None => return,
    };
    // exactly half a unit in the last place: "5000..." after the digits kept
    if !(tail.starts_with('5') && tail[1..].bytes().all(|b| b == b'0')) {
        return;
    }
    let mut cand = digits.clone();
    cand.pop();
    cand.push((last - 1) as char);
    let text = format!("{}e{}", cand, exp - (k as i32 - 1));
    if text.parse::<f64>().ok() == Some(n) {
        *digits = cand;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canon(s: &str) -> String {
        String::from_utf8(canonical(&parse(s).unwrap(), false).unwrap()).unwrap()
    }

    #[test]
    fn utf16_key_order_differs_from_utf8() {
        // RFC 8785 section 3.2.3
        let src = r#"{"\u20ac":"Euro Sign","\r":"Carriage Return","\ufb33":"Hebrew Dalet With Dagesh","1":"One","\ud83d\ude00":"Emoji","\u0080":"Control","\u00f6":"Latin o diaeresis"}"#;
        let v = parse(src).unwrap();
        let c = String::from_utf8(canonical(&v, false).unwrap()).unwrap();
        let order = [
            "\"\\r\":",
            "\"1\":",
            "\"\u{80}\":",
            "\"\u{f6}\":",
            "\"\u{20ac}\":",
            "\"\u{1F600}\":",
            "\"\u{fb33}\":",
        ];
        let mut at = 0;
        for k in order {
            let p = c[at..]
                .find(k)
                .unwrap_or_else(|| panic!("{k} out of order in {c}"));
            at += p + k.len();
        }
        // UTF-8 byte order would put U+FB33 before the emoji, so naive output hashes differently
        let J::Obj(o) = &v else { unreachable!() };
        let mut by_bytes: Vec<&str> = o.iter().map(|(k, _)| k.as_str()).collect();
        by_bytes.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        assert_eq!(&by_bytes[5..], ["\u{fb33}", "\u{1F600}"]);
    }

    #[test]
    fn rfc_example() {
        // RFC 8785 section 3.2.4
        let src = "{\n \"numbers\": [333333333.33333329, 1E30, 4.50, 2e-3, 0.000000000000000000000000001],\n \"string\": \"\\u20ac$\\u000F\\u000aA'\\u0042\\u0022\\u005c\\\\\\\"\\/\",\n \"literals\": [null, true, false]\n}";
        assert_eq!(
            canon(src),
            "{\"literals\":[null,true,false],\"numbers\":[333333333.3333333,1e+30,4.5,0.002,1e-27],\"string\":\"€$\\u000f\\nA'B\\\"\\\\\\\\\\\"/\"}"
        );
    }

    #[test]
    fn rfc_number_table() {
        // RFC 8785 appendix B (IEEE bit patterns -> ECMAScript text)
        for (bits, want) in [
            (0x0000000000000000u64, "0"),
            (0x8000000000000000, "0"),
            (0x0000000000000001, "5e-324"),
            (0x8000000000000001, "-5e-324"),
            (0x7fefffffffffffff, "1.7976931348623157e+308"),
            (0xffefffffffffffff, "-1.7976931348623157e+308"),
            (0x4340000000000000, "9007199254740992"),
            (0xc340000000000000, "-9007199254740992"),
            (0x4430000000000000, "295147905179352830000"),
            (0x44b52d02c7e14af5, "9.999999999999997e+22"),
            (0x44b52d02c7e14af6, "1e+23"),
            (0x44b52d02c7e14af7, "1.0000000000000001e+23"),
            (0x444b1ae4d6e2ef4e, "999999999999999700000"),
            (0x444b1ae4d6e2ef4f, "999999999999999900000"),
            (0x444b1ae4d6e2ef50, "1e+21"),
            (0x3eb0c6f7a0b5ed8c, "9.999999999999997e-7"),
            (0x3eb0c6f7a0b5ed8d, "0.000001"),
            (0x41b3de4355555553, "333333333.3333332"),
            (0x41b3de4355555554, "333333333.33333325"),
            (0x41b3de4355555555, "333333333.3333333"),
            (0x41b3de4355555556, "333333333.3333334"),
            (0x41b3de4355555557, "333333333.33333343"),
            (0xbecbf647612f3696, "-0.0000033333333333333333"),
            (0x43143ff3c1cb0959, "1424953923781206.2"),
        ] {
            assert_eq!(number(f64::from_bits(bits)).unwrap(), want, "{bits:#x}");
        }
        assert!(number(f64::from_bits(0x7ff0000000000000)).is_err());
        assert!(number(f64::NAN).is_err());
        assert_eq!(number(-1.5).unwrap(), "-1.5");
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse(r#"{"a":1,"a":2}"#).is_err());
        assert!(parse(r#""\ud800""#).is_err());
        assert!(parse(r#""\udc00""#).is_err());
        assert!(parse(r#""\ud83d\ude00""#).is_ok());
        assert!(parse("1e999").is_err());
        assert!(canonical(&parse("1.5").unwrap(), true).is_err());
        assert!(canonical(&parse("9007199254740993").unwrap(), true).is_err());
        for bad in [
            "007",
            "-",
            "1.",
            ".5",
            "+1",
            "1e",
            "1e+",
            "--1",
            "0x10",
            "[1,]",
            "{\"a\":}",
            "{,}",
            "[",
            "\"a",
            "\"a\nb\"",
            "tru",
            "nul",
            "",
            "  ",
            "{} {}",
            "\"\\x\"",
            "\"\\u12\"",
            "\"\\u12g4\"",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn depth_is_limited_in_both_directions() {
        let deep = "[".repeat(100) + &"]".repeat(100);
        assert!(parse(&deep).is_err());
        let ok = "[".repeat(60) + &"]".repeat(60);
        assert!(parse(&ok).is_ok());
        let mut v = J::Null;
        for _ in 0..200 {
            v = J::Arr(vec![v]);
        }
        assert!(canonical(&v, false).is_err());
    }

    #[test]
    fn long_strings_and_many_keys_are_linear() {
        let long = format!("\"{}\"", "é".repeat(500_000));
        let t = std::time::Instant::now();
        assert_eq!(
            parse(&long).unwrap().as_str().unwrap().chars().count(),
            500_000
        );
        let mut obj = String::from("{");
        for i in 0..50_000 {
            if i > 0 {
                obj.push(',');
            }
            obj.push_str(&format!("\"k{i}\":{i}"));
        }
        obj.push('}');
        let v = parse(&obj).unwrap();
        assert_eq!(v.get("k49999").and_then(J::as_u64), Some(49999));
        assert!(t.elapsed().as_secs() < 5, "parsing must not be quadratic");
    }

    #[test]
    fn round_trip_is_stable() {
        let mut s = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..300 {
            let mut o = Vec::new();
            for i in 0..(next() % 6) {
                let k: String = (0..(next() % 5))
                    .map(|_| char::from_u32(0x20 + (next() % 0x600) as u32).unwrap_or('x'))
                    .collect::<String>()
                    + &i.to_string();
                let v = match next() % 4 {
                    0 => J::Null,
                    1 => J::Bool(next() % 2 == 0),
                    2 => J::Num((next() % 1_000_000) as f64),
                    _ => J::Str(
                        (0..(next() % 8))
                            .map(|_| char::from_u32((next() % 0x3000) as u32).unwrap_or('y'))
                            .collect(),
                    ),
                };
                o.push((k, v));
            }
            let j = J::Obj(o);
            let c1 = canonical(&j, true).unwrap();
            let again = parse(std::str::from_utf8(&c1).unwrap()).unwrap();
            assert_eq!(canonical(&again, true).unwrap(), c1);
        }
    }

    #[test]
    fn from_value_and_take() {
        let v: serde_json::Value = serde_json::from_str(r#"{"a":[1,2],"b":"x"}"#).unwrap();
        let mut j = from_value(&v).unwrap();
        assert_eq!(j.get("b").and_then(J::as_str), Some("x"));
        assert_eq!(j.get("a").and_then(J::as_array).map(<[J]>::len), Some(2));
        assert_eq!(j.take("b"), Some(J::Str("x".into())));
        assert!(j.get("b").is_none());
    }
}
