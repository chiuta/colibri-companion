//! Tool (function) calling: find the calls in what a model wrote, for every format the common instruct
//! families use, and do it incrementally for streaming.
//!
//! Formats understood (all of them can appear several times, and the text around them is kept as content):
//!  * Hermes / Qwen / many fine-tunes: `<tool_call>{"name": …, "arguments": {…}}</tool_call>`
//!  * Llama 3.1: `<|python_tag|>{"name": …, "parameters": {…}}` (several separated by `;`), and the older
//!    `<function=name>{…}</function>`
//!  * Mistral: `[TOOL_CALLS] [{"name": …, "arguments": {…}, "id": …}]`
//!  * a JSON object or array in a ```json fence, or as the whole reply (only when the name is a declared tool)
//!
//! [`scan`] is stateless: it looks at all the text so far. [`ToolStream`] turns successive scans into deltas.
use serde_json::{Deserializer, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    /// The arguments as a JSON string (what OpenAI's API carries).
    pub arguments: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Scanned {
    /// The visible text: everything except the call regions. For a prefix of the text this is a prefix of the
    /// content of the whole text, so it can be streamed.
    pub content: String,
    pub calls: Vec<ToolCall>,
    /// The text ends inside something that may still turn out to be a call.
    pub open: bool,
}

const MARKERS: [&str; 5] = ["<tool_call>", "<|python_tag|>", "[TOOL_CALLS]", "<function=", "```"];
/// Control tokens that can trail a call when the vocabulary's specials are decoded.
const TRAILERS: [&str; 5] = ["<|eom_id|>", "<|eot_id|>", "<|im_end|>", "</s>", "<|end_of_text|>"];

enum J { Done(Value, usize), Incomplete, Bad }

fn first_json(s: &str) -> J {
    let mut it = Deserializer::from_str(s).into_iter::<Value>();
    match it.next() {
        Some(Ok(v)) => J::Done(v, it.byte_offset()),
        Some(Err(e)) if e.is_eof() => J::Incomplete,
        Some(Err(_)) => J::Bad,
        None => J::Incomplete,
    }
}

fn to_call(v: &Value, names: &[String], strict: bool) -> Option<ToolCall> {
    let o = v.as_object()?;
    let f = match o.get("function") { Some(Value::Object(f)) => f, _ => o };
    let name = f.get("name").and_then(Value::as_str).or_else(|| o.get("function").and_then(Value::as_str))?;
    if name.is_empty() || name.len() > 128 { return None; }
    let args = f.get("arguments").or_else(|| f.get("parameters")).or_else(|| f.get("args"));
    if strict && (args.is_none() || !names.iter().any(|n| n == name)) { return None; }
    let arguments = match args {
        None | Some(Value::Null) => "{}".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
    };
    Some(ToolCall { name: name.to_string(), arguments })
}

fn calls_of(v: &Value, names: &[String], strict: bool) -> Option<Vec<ToolCall>> {
    match v {
        Value::Array(a) if !a.is_empty() && a.len() <= 64 => a.iter().map(|x| to_call(x, names, strict)).collect(),
        Value::Object(_) => to_call(v, names, strict).map(|c| vec![c]),
        _ => None,
    }
}

/// Longest suffix of `s` that is a proper prefix of one of the markers.
fn marker_prefix_len(s: &str) -> usize {
    let mut best = 0;
    for m in MARKERS {
        for k in (best + 1..=(m.len() - 1).min(s.len())).rev() {
            if s.is_char_boundary(s.len() - k) && s.ends_with(&m[..k]) { best = k; break; }
        }
    }
    best
}

fn next_marker(s: &str) -> Option<(usize, &'static str)> {
    MARKERS.iter().filter_map(|m| s.find(m).map(|i| (i, *m))).min_by_key(|(i, _)| *i)
}

/// Parse `text` (everything the model has written so far). `fin` = generation is over.
pub fn scan(text: &str, fin: bool, names: &[String]) -> Scanned {
    let mut out = Scanned::default();
    let mut pos = 0usize;
    // the whole reply being a bare JSON call
    let lead = text.len() - text.trim_start().len();
    let rest0 = &text[lead..];
    if rest0.starts_with('{') || rest0.starts_with('[') {
        match first_json(rest0) {
            J::Done(v, end) => {
                if let Some(c) = calls_of(&v, names, true) {
                    out.calls = c;
                    pos = lead + end;
                    // anything after is plain text (or a trailing control token)
                    let tail = text[pos..].trim_start();
                    let tail = TRAILERS.iter().fold(tail, |t, k| t.strip_suffix(k).unwrap_or(t).trim_end());
                    if !fin && partial_trailer(tail) { out.open = true; return out; }
                    if !tail.is_empty() { return finish_rest(out, &text[pos..], fin, names); }
                    return out;
                }
            }
            J::Incomplete if !fin => { out.open = true; return out; }
            _ => {}
        }
    }
    while pos < text.len() {
        let rest = &text[pos..];
        let Some((i, m)) = next_marker(rest) else {
            let hold = if fin { 0 } else { marker_prefix_len(rest) };
            out.content.push_str(&rest[..rest.len() - hold]);
            out.open = hold > 0;
            return out;
        };
        out.content.push_str(&rest[..i]);
        let after = pos + i + m.len();
        let body = &text[after..];
        match m {
            "<tool_call>" => {
                let (inner, next) = match body.find("</tool_call>") {
                    Some(e) => (&body[..e], after + e + "</tool_call>".len()),
                    None if fin => (body, text.len()),
                    None => { out.open = true; return out; }
                };
                match first_json(inner.trim()) {
                    J::Done(v, _) => match calls_of(&v, names, false) {
                        Some(c) => out.calls.extend(c),
                        None => out.content.push_str(&text[pos + i..next]),
                    },
                    _ => out.content.push_str(&text[pos + i..next]),
                }
                pos = next;
            }
            "<|python_tag|>" | "[TOOL_CALLS]" => {
                // a run of JSON values (separated by `;` or whitespace); array values hold several calls
                let mut at = 0usize;
                let mut got = false;
                loop {
                    let b = &body[at..];
                    let skip = b.len() - b.trim_start_matches(|c: char| c.is_whitespace() || c == ';').len();
                    let b = &b[skip..];
                    if b.is_empty() { if !fin { out.open = true; } break; }
                    match first_json(b) {
                        J::Done(v, end) => match calls_of(&v, names, false) {
                            Some(c) => { out.calls.extend(c); got = true; at += skip + end; }
                            None => break,
                        },
                        J::Incomplete if !fin => { out.open = true; break; }
                        _ => break,
                    }
                }
                if out.open { return out; }
                if !got {
                    // not a call after all: keep the marker as text
                    out.content.push_str(m);
                    pos = after;
                    continue;
                }
                pos = after + at;
                // swallow trailing control tokens
                loop {
                    let t = text[pos..].trim_start();
                    let skipped = text.len() - pos - t.len();
                    match TRAILERS.iter().find(|k| t.starts_with(**k)) { Some(k) => pos += skipped + k.len(), None => break }
                }
                if !fin && partial_trailer(text[pos..].trim_start()) { out.open = true; return out; }
            }
            "<function=" => {
                let Some(gt) = body.find('>') else { if fin { out.content.push_str(&text[pos + i..]); return out; } out.open = true; return out };
                let name = body[..gt].trim().to_string();
                let after_name = after + gt + 1;
                let (inner, next) = match text[after_name..].find("</function>") {
                    Some(e) => (&text[after_name..after_name + e], after_name + e + "</function>".len()),
                    None if fin => (&text[after_name..], text.len()),
                    None => { out.open = true; return out; }
                };
                let inner = inner.trim();
                if name.is_empty() || name.len() > 128 || !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')) {
                    out.content.push_str(&text[pos + i..next]);
                } else {
                    out.calls.push(ToolCall { name, arguments: if inner.is_empty() { "{}".into() } else { inner.to_string() } });
                }
                pos = next;
            }
            _ => {
                // ``` — a call only when it is a json-ish fence whose content is a declared call
                let Some(nl) = body.find('\n') else {
                    if fin { out.content.push_str(&text[pos + i..]); return out; }
                    out.open = true;
                    return out;
                };
                let info = body[..nl].trim().to_ascii_lowercase();
                if !matches!(info.as_str(), "" | "json" | "tool_call" | "tool_calls") {
                    out.content.push_str("```");
                    pos = after;
                    continue;
                }
                let inner_start = after + nl + 1;
                let Some(e) = text[inner_start..].find("```") else {
                    if fin { out.content.push_str(&text[pos + i..]); return out; }
                    out.open = true;
                    return out;
                };
                let next = inner_start + e + 3;
                match first_json(text[inner_start..inner_start + e].trim()).into_value().and_then(|v| calls_of(&v, names, true)) {
                    Some(c) => out.calls.extend(c),
                    None => out.content.push_str(&text[pos + i..next]),
                }
                pos = next;
            }
        }
    }
    out
}

fn partial_trailer(t: &str) -> bool { !t.is_empty() && TRAILERS.iter().any(|k| k.len() > t.len() && k.starts_with(t)) }

/// After a bare-JSON call the rest of the reply is ordinary text (it may contain more markers).
fn finish_rest(mut out: Scanned, rest: &str, fin: bool, names: &[String]) -> Scanned {
    let more = scan(rest, fin, names);
    out.content.push_str(&more.content);
    out.calls.extend(more.calls);
    out.open = more.open;
    out
}

impl J {
    fn into_value(self) -> Option<Value> { match self { J::Done(v, _) => Some(v), _ => None } }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Delta { Text(String), Call(usize, ToolCall) }

/// Turns the growing text into deltas: new content as soon as it is safe, each call once it is complete.
#[derive(Debug, Default)]
pub struct ToolStream { names: Vec<String>, sent: usize, calls: usize }
impl ToolStream {
    pub fn new(names: Vec<String>) -> ToolStream { ToolStream { names, sent: 0, calls: 0 } }
    pub fn push(&mut self, all: &str, fin: bool) -> (Vec<Delta>, Scanned) {
        let s = scan(all, fin, &self.names);
        let mut d = vec![];
        if s.content.len() > self.sent && s.content.is_char_boundary(self.sent) {
            d.push(Delta::Text(s.content[self.sent..].to_string()));
            self.sent = s.content.len();
        }
        while self.calls < s.calls.len() {
            d.push(Delta::Call(self.calls, s.calls[self.calls].clone()));
            self.calls += 1;
        }
        (d, s)
    }
}

// ------------------------------------------------------------------------------ request side

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice { Auto, None, Required, Named(String) }

/// OpenAI `tool_choice`: "auto" | "none" | "required" | {"type":"function","function":{"name":…}}.
/// Anthropic's `{"type":"auto"|"any"|"tool","name":…}` is accepted too.
pub fn parse_choice(v: Option<&Value>) -> Result<ToolChoice, String> {
    Ok(match v {
        None | Some(Value::Null) => ToolChoice::Auto,
        Some(Value::String(s)) => match s.as_str() {
            "auto" => ToolChoice::Auto, "none" => ToolChoice::None, "required" | "any" => ToolChoice::Required,
            o => return Err(format!("unsupported `tool_choice` \"{o}\"")),
        },
        Some(Value::Object(o)) => {
            let ty = o.get("type").and_then(Value::as_str).unwrap_or("function");
            match ty {
                "auto" => ToolChoice::Auto, "none" => ToolChoice::None, "any" => ToolChoice::Required,
                _ => {
                    let n = o.get("function").and_then(|f| f.get("name")).or_else(|| o.get("name")).and_then(Value::as_str);
                    ToolChoice::Named(n.ok_or("`tool_choice` names no function")?.to_string())
                }
            }
        }
        Some(_) => return Err("`tool_choice` must be a string or an object".into()),
    })
}

/// Validate the `tools` list; returns it normalised to `{type:"function", function:{name, description, parameters}}`.
pub fn normalize_tools(v: Option<&Value>) -> Result<Vec<Value>, String> {
    let Some(v) = v.filter(|v| !v.is_null()) else { return Ok(vec![]) };
    let Value::Array(a) = v else { return Err("`tools` must be a list".into()) };
    if a.len() > 128 { return Err("too many tools".into()); }
    let mut out = vec![];
    for t in a {
        // OpenAI: {type, function:{…}}; Anthropic: {name, description, input_schema}; bare function objects
        let f = match t.get("function") {
            Some(f @ Value::Object(_)) => f.clone(),
            _ => {
                let mut f = t.clone();
                if let Some(o) = f.as_object_mut() {
                    o.remove("type");
                    if let Some(s) = o.remove("input_schema") { o.insert("parameters".into(), s); }
                }
                f
            }
        };
        let name = f.get("name").and_then(Value::as_str).ok_or("every tool needs a `name`")?;
        if name.is_empty() || name.len() > 128 || !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b':')) {
            return Err(format!("invalid tool name \"{name}\""));
        }
        let mut g = serde_json::Map::new();
        g.insert("name".into(), json_str(name));
        if let Some(d) = f.get("description").filter(|d| d.is_string()) { g.insert("description".into(), d.clone()); }
        g.insert("parameters".into(), f.get("parameters").filter(|p| p.is_object()).cloned().unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})));
        out.push(serde_json::json!({"type": "function", "function": Value::Object(g)}));
    }
    Ok(out)
}

fn json_str(s: &str) -> Value { Value::String(s.to_string()) }

pub fn tool_names(tools: &[Value]) -> Vec<String> {
    tools.iter().filter_map(|t| t.get("function").and_then(|f| f.get("name")).and_then(Value::as_str).map(str::to_string)).collect()
}

/// The instruction added to the system turn when the request forces a call.
pub fn choice_hint(c: &ToolChoice) -> Option<String> {
    match c {
        ToolChoice::Required => Some("You must answer by calling at least one of the provided functions.".into()),
        ToolChoice::Named(n) => Some(format!("You must answer by calling the function `{n}`.")),
        _ => None,
    }
}

/// `call_` + 24 hex digits; `salt` makes ids differ between calls of one response.
pub fn call_id(seed: u64, salt: usize) -> String {
    let mut x = seed ^ (salt as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 33; x = x.wrapping_mul(0xff51_afd7_ed55_8ccd); x ^= x >> 33;
    let y = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53) ^ (x >> 29);
    format!("call_{x:016x}{:08x}", y as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> { vec!["get_weather".into(), "search".into()] }
    fn sc(t: &str) -> Scanned { scan(t, true, &names()) }
    fn args(c: &ToolCall) -> Value { serde_json::from_str(&c.arguments).unwrap() }

    #[test]
    fn hermes_qwen() {
        let t = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"unit\": \"celsius\"}}\n</tool_call>";
        let s = sc(t);
        assert_eq!(s.content, "Let me check.\n");
        assert_eq!(s.calls.len(), 1);
        assert_eq!(s.calls[0].name, "get_weather");
        assert_eq!(args(&s.calls[0])["city"], "Paris");
    }

    #[test]
    fn hermes_parallel_and_unclosed() {
        let t = "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"a\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"b\"}}\n</tool_call>";
        assert_eq!(sc(t).calls.len(), 2);
        // generation stopped before the closing tag
        let s = sc("<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"a\"}}");
        assert_eq!(s.calls.len(), 1);
        // while streaming it is still open
        let s = scan("<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"a\"}}", false, &names());
        assert!(s.open && s.calls.is_empty());
    }

    #[test]
    fn llama31_python_tag() {
        let s = sc("<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}<|eom_id|>");
        assert_eq!(s.calls.len(), 1);
        assert_eq!(args(&s.calls[0])["city"], "Rome");
        assert_eq!(s.content, "");
        let s = sc("<|python_tag|>{\"name\": \"search\", \"parameters\": {\"q\": \"x\"}}; {\"name\": \"search\", \"parameters\": {\"q\": \"y\"}}");
        assert_eq!(s.calls.len(), 2);
    }

    #[test]
    fn llama_function_tag() {
        let s = sc("<function=get_weather>{\"city\": \"Oslo\"}</function>");
        assert_eq!(s.calls, vec![ToolCall { name: "get_weather".into(), arguments: "{\"city\": \"Oslo\"}".into() }]);
    }

    #[test]
    fn mistral_tool_calls() {
        let s = sc("[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Lyon\"}, \"id\": \"abc123XYZ\"}, {\"name\": \"search\", \"arguments\": {\"q\": \"z\"}, \"id\": \"def456UVW\"}]</s>");
        assert_eq!(s.calls.len(), 2);
        assert_eq!(s.calls[1].name, "search");
        assert_eq!(s.content, "");
    }

    #[test]
    fn bare_json_and_fence() {
        let s = sc("{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Bern\"}}");
        assert_eq!(s.calls.len(), 1);
        let s = sc("Sure:\n```json\n{\"name\": \"search\", \"arguments\": {\"q\": \"rust\"}}\n```\nDone.");
        assert_eq!(s.calls.len(), 1);
        assert_eq!(s.content, "Sure:\n\nDone.");
        // unknown names and ordinary JSON answers stay text
        let s = sc("{\"name\": \"Alice\", \"age\": 3}");
        assert!(s.calls.is_empty());
        assert_eq!(s.content, "{\"name\": \"Alice\", \"age\": 3}");
        let s = sc("```json\n{\"name\": \"nope\", \"arguments\": {}}\n```");
        assert!(s.calls.is_empty());
        assert!(s.content.contains("nope"));
        let s = sc("```rust\nfn main() {}\n```");
        assert_eq!(s.content, "```rust\nfn main() {}\n```");
    }

    #[test]
    fn plain_text_untouched() {
        for t in ["", "hello", "a < b and [c] <d>", "use `code` and ``` fences", "[1, 2, 3]"] {
            let s = sc(t);
            assert_eq!(s.content, t, "{t:?}");
            assert!(s.calls.is_empty());
        }
    }

    #[test]
    fn openai_shaped_call_object() {
        let s = sc("<tool_call>{\"type\": \"function\", \"function\": {\"name\": \"search\", \"arguments\": \"{\\\"q\\\": 1}\"}}</tool_call>");
        assert_eq!(s.calls[0].arguments, "{\"q\": 1}");
    }

    const SAMPLES: [&str; 7] = [
        "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\nthen more text",
        "<|python_tag|>{\"name\": \"search\", \"parameters\": {\"q\": \"x <b>\"}}<|eom_id|>",
        "[TOOL_CALLS] [{\"name\": \"search\", \"arguments\": {\"q\": \"é€ü\"}, \"id\": \"a\"}]",
        "Here:\n```json\n{\"name\": \"search\", \"arguments\": {\"q\": \"1\"}}\n```\nok",
        "just words with a ` tick and [brackets] and <tags>",
        "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Oslo\"}}",
        "<function=search>{\"q\": \"a\"}</function> after",
    ];

    /// For every way of cutting the text in two, scanning the prefix first never contradicts scanning the whole,
    /// and streaming the pieces reproduces the final content and calls exactly.
    #[test]
    fn streaming_is_consistent_for_every_split() {
        for t in SAMPLES {
            let full = scan(t, true, &names());
            for cut in (0..=t.len()).filter(|c| t.is_char_boundary(*c)) {
                let pre = scan(&t[..cut], false, &names());
                assert!(full.content.starts_with(&pre.content), "{t:?} cut {cut}: {:?} vs {:?}", pre.content, full.content);
                assert!(pre.calls.len() <= full.calls.len() && pre.calls[..] == full.calls[..pre.calls.len()], "{t:?} cut {cut}");
            }
            // char by char
            let mut st = ToolStream::new(names());
            let (mut text, mut calls) = (String::new(), vec![]);
            let mut acc = String::new();
            for ch in t.chars() {
                acc.push(ch);
                for d in st.push(&acc, false).0 { match d { Delta::Text(x) => text.push_str(&x), Delta::Call(i, c) => { assert_eq!(i, calls.len()); calls.push(c); } } }
            }
            for d in st.push(&acc, true).0 { match d { Delta::Text(x) => text.push_str(&x), Delta::Call(i, c) => { assert_eq!(i, calls.len()); calls.push(c); } } }
            assert_eq!(text, full.content, "{t:?}");
            assert_eq!(calls, full.calls, "{t:?}");
        }
    }

    #[test]
    fn fuzz_never_panics_and_stays_monotonic() {
        let alphabet: Vec<&str> = vec!["<tool_call>", "</tool_call>", "<|python_tag|>", "[TOOL_CALLS]", "```", "json", "\n", "{", "}", "[", "]", "\"name\"", ":", "\"search\"", "\"arguments\"", ",", "é", " ", ";", "<function=", ">", "</function>", "x", "<|eom_id|>"];
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        for _ in 0..4000 {
            let n = (next() % 14) as usize;
            let t: String = (0..n).map(|_| alphabet[(next() % alphabet.len() as u64) as usize]).collect();
            let full = scan(&t, true, &names());
            assert!(!full.open);
            for cut in (0..=t.len()).filter(|c| t.is_char_boundary(*c)) {
                let pre = scan(&t[..cut], false, &names());
                assert!(full.content.starts_with(&pre.content) || !pre.open && false, "{t:?} cut {cut}: {:?} vs {:?}", pre.content, full.content);
                assert!(pre.calls.len() <= full.calls.len(), "{t:?} cut {cut}");
            }
        }
    }

    #[test]
    fn choice_and_tools_parsing() {
        use serde_json::json;
        assert_eq!(parse_choice(Some(&json!("required"))).unwrap(), ToolChoice::Required);
        assert_eq!(parse_choice(Some(&json!({"type": "function", "function": {"name": "f"}}))).unwrap(), ToolChoice::Named("f".into()));
        assert_eq!(parse_choice(Some(&json!({"type": "any"}))).unwrap(), ToolChoice::Required);
        assert_eq!(parse_choice(Some(&json!({"type": "tool", "name": "g"}))).unwrap(), ToolChoice::Named("g".into()));
        assert!(parse_choice(Some(&json!("sometimes"))).is_err());
        let t = normalize_tools(Some(&json!([{"name": "a", "description": "d", "input_schema": {"type": "object"}}, {"type": "function", "function": {"name": "b"}}]))).unwrap();
        assert_eq!(tool_names(&t), vec!["a", "b"]);
        assert_eq!(t[0]["function"]["parameters"], json!({"type": "object"}));
        assert!(normalize_tools(Some(&json!([{"name": "bad name!"}]))).is_err());
        assert!(normalize_tools(Some(&json!("x"))).is_err());
        assert_ne!(call_id(1, 0), call_id(1, 1));
        assert_eq!(call_id(7, 3).len(), 29);
    }
}
