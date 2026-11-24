//! Ollama-style `Modelfile` parser.
//!
//! ```text
//! FROM ./qwen2.5-7b-instruct        # a model folder, a GGUF file, or the name of a model in the registry
//! PARAMETER temperature 0.2
//! PARAMETER stop "<|im_end|>"
//! SYSTEM """You are terse."""
//! TEMPLATE """{{ ... }}"""
//! ADAPTER ./lora-folder
//! MESSAGE user Hello
//! ```
//! Instructions are case-insensitive, `#` starts a comment line, and values may be wrapped in `"""` to span lines.
use serde_json::{json, Map, Value};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Modelfile {
    pub from: String,
    pub params: Vec<(String, String)>,
    pub template: Option<String>,
    pub system: Option<String>,
    pub adapters: Vec<String>,
    pub messages: Vec<(String, String)>,
    pub license: Option<String>,
}

const MAX_BYTES: usize = 1 << 20;

fn unquote(s: &str) -> String {
    let s = s.trim();
    if let Some(i) = s.strip_prefix("\"\"\"").and_then(|r| r.strip_suffix("\"\"\"")) { return i.to_string(); }
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') { return s[1..s.len() - 1].replace("\\\"", "\""); }
    s.to_string()
}

impl Modelfile {
    pub fn parse(text: &str) -> Result<Modelfile, String> {
        if text.len() > MAX_BYTES { return Err("the Modelfile is too large".into()); }
        let mut m = Modelfile::default();
        let mut lines = text.lines().enumerate().peekable();
        while let Some((n, raw)) = lines.next() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let (word, rest) = line.split_once(char::is_whitespace).map_or((line, ""), |(a, b)| (a, b.trim_start()));
            let word = word.to_ascii_lowercase();
            // a value opening with """ continues until the closing """
            let mut value = rest.to_string();
            if rest.starts_with("\"\"\"") && !(rest.len() >= 6 && rest.ends_with("\"\"\"")) {
                loop {
                    let Some((_, next)) = lines.next() else { return Err(format!("line {}: unterminated \"\"\" block", n + 1)) };
                    value.push('\n');
                    value.push_str(next);
                    if next.trim_end().ends_with("\"\"\"") { break; }
                }
            }
            let v = unquote(&value);
            let at = |msg: &str| format!("line {}: {msg}", n + 1);
            match word.as_str() {
                "from" => { if v.is_empty() { return Err(at("FROM needs a model")); }
                    if !m.from.is_empty() { return Err(at("FROM given twice")); } m.from = v; }
                "parameter" => {
                    let (k, val) = rest.split_once(char::is_whitespace).ok_or_else(|| at("PARAMETER needs a name and a value"))?;
                    let val = unquote(val);
                    if val.is_empty() { return Err(at("PARAMETER needs a value")); }
                    m.params.push((k.to_ascii_lowercase(), val));
                }
                "template" => m.template = Some(v),
                "system" => m.system = Some(v),
                "adapter" => { if v.is_empty() { return Err(at("ADAPTER needs a path")); } m.adapters.push(v); }
                "license" => m.license = Some(v),
                "message" => {
                    let (role, content) = v.split_once(char::is_whitespace).ok_or_else(|| at("MESSAGE needs a role and text"))?;
                    if !matches!(role.to_ascii_lowercase().as_str(), "system" | "user" | "assistant") { return Err(at("MESSAGE role must be system, user or assistant")); }
                    m.messages.push((role.to_ascii_lowercase(), unquote(content)));
                }
                other => return Err(at(&format!("unknown instruction `{other}`"))),
            }
        }
        if m.from.is_empty() { return Err("a Modelfile needs a FROM line".into()); }
        Ok(m)
    }

    /// Parameters as request options (numbers as numbers, repeated `stop` as a list).
    pub fn options(&self) -> Map<String, Value> {
        let mut o = Map::new();
        for (k, v) in &self.params {
            if k == "stop" {
                if let Value::Array(a) = o.entry("stop").or_insert_with(|| json!([])) { a.push(json!(v)); }
                continue;
            }
            let val = if let Ok(i) = v.parse::<i64>() { json!(i) } else if let Some(f) = v.parse::<f64>().ok().filter(|f| f.is_finite()) { json!(f) } else { match v.as_str() { "true" => json!(true), "false" => json!(false), s => json!(s) } };
            o.insert(k.clone(), val);
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_example() {
        let m = Modelfile::parse("# a comment\nFROM ./qwen\nparameter temperature 0.2\nPARAMETER num_ctx 4096\nPARAMETER stop \"<|im_end|>\"\nPARAMETER stop END\nSYSTEM \"\"\"You are terse.\nVery.\"\"\"\nTEMPLATE \"\"\"\n{{ messages }}\n\"\"\"\nADAPTER ./lora\nMESSAGE user Hi there\nMESSAGE assistant \"Hello\"\n").unwrap();
        assert_eq!(m.from, "./qwen");
        assert_eq!(m.system.as_deref(), Some("You are terse.\nVery."));
        assert_eq!(m.template.as_deref(), Some("\n{{ messages }}\n"));
        assert_eq!(m.adapters, vec!["./lora"]);
        assert_eq!(m.messages, vec![("user".into(), "Hi there".into()), ("assistant".into(), "Hello".into())]);
        let o = m.options();
        assert_eq!(o["temperature"], json!(0.2));
        assert_eq!(o["num_ctx"], json!(4096));
        assert_eq!(o["stop"], json!(["<|im_end|>", "END"]));
    }

    #[test]
    fn one_line_triple_quotes() {
        let m = Modelfile::parse("FROM x\nSYSTEM \"\"\"short\"\"\"").unwrap();
        assert_eq!(m.system.as_deref(), Some("short"));
    }

    #[test]
    fn errors() {
        for bad in ["", "# only a comment", "PARAMETER a 1", "FROM", "FROM a\nFROM b", "FROM a\nBOGUS x", "FROM a\nSYSTEM \"\"\"open", "FROM a\nPARAMETER x", "FROM a\nMESSAGE robot hi", "FROM a\nADAPTER"] {
            assert!(Modelfile::parse(bad).is_err(), "{bad:?}");
        }
        assert!(Modelfile::parse(&format!("FROM a\nSYSTEM {}", "x".repeat(MAX_BYTES))).is_err());
    }

    #[test]
    fn fuzz_never_panics() {
        let parts = ["FROM ", "PARAMETER ", "SYSTEM ", "TEMPLATE ", "\"\"\"", "\"", "\n", " ", "a", "stop", "#", "MESSAGE user ", "ADAPTER "];
        let mut s = 99u64;
        for _ in 0..3000 {
            let mut t = String::new();
            for _ in 0..(s % 12) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; t.push_str(parts[(s % parts.len() as u64) as usize]); }
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let _ = Modelfile::parse(&t).map(|m| m.options());
        }
    }
}
