//! Structured output: `response_format` (`json_object`, `json_schema`), llama.cpp's `grammar` / `json_schema`
//! fields and Ollama's `format`, all compiled to the engine's token-level grammar constraint.
use crate::genloop::Constraint;
use crate::registry::Slot;
use companion_engine::grammar::{
    Constraint as Engine, Grammar, SchemaOptions, TokenTrie, Whitespace,
};
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub enum Spec {
    JsonObject,
    Schema(Value),
    Gbnf(String),
    Regex(String),
}

impl Spec {
    /// Read the structured-output fields of a request body (any of the dialects). `None` = free text.
    pub fn from_body(body: &Value) -> Result<Option<Spec>, String> {
        if let Some(g) = body
            .get("grammar")
            .and_then(Value::as_str)
            .filter(|g| !g.trim().is_empty())
        {
            return Ok(Some(Spec::Gbnf(g.to_string())));
        }
        if let Some(s) = body.get("json_schema").filter(|s| s.is_object()) {
            return Ok(Some(Spec::Schema(s.clone())));
        }
        if let Some(r) = body
            .get("regex")
            .and_then(Value::as_str)
            .filter(|g| !g.is_empty())
        {
            return Ok(Some(Spec::Regex(r.to_string())));
        }
        match body.get("response_format") {
            None | Some(Value::Null) => {}
            Some(f) => match f.get("type").and_then(Value::as_str) {
                Some("text") | None => {}
                Some("json_object") => return Ok(Some(Spec::JsonObject)),
                Some("json_schema") => {
                    let s = f
                        .get("json_schema")
                        .and_then(|j| j.get("schema"))
                        .or_else(|| f.get("schema"))
                        .filter(|s| s.is_object())
                        .ok_or("`response_format.json_schema.schema` must be an object")?;
                    return Ok(Some(Spec::Schema(s.clone())));
                }
                Some(o) => return Err(format!("unsupported `response_format.type` \"{o}\"")),
            },
        }
        // Ollama
        match body.get("format") {
            Some(Value::String(s)) if s == "json" => Ok(Some(Spec::JsonObject)),
            Some(s @ Value::Object(_)) => Ok(Some(Spec::Schema(s.clone()))),
            Some(Value::Null) | None => Ok(None),
            Some(_) => Err("`format` must be \"json\" or a JSON schema".into()),
        }
    }
}

fn opts() -> SchemaOptions {
    SchemaOptions {
        whitespace: Whitespace::Flexible,
        max_ws: 2,
        additional_properties: false,
    }
}

struct Adapter {
    c: Engine,
    stops: Vec<usize>,
}
impl Constraint for Adapter {
    fn apply(&mut self, logits: &mut [f32]) {
        // once the output is a complete sentence the model may also end with any of its own end tokens
        let keep: Vec<(usize, f32)> = if self.c.is_complete() {
            self.stops
                .iter()
                .filter_map(|&i| logits.get(i).map(|l| (i, *l)))
                .collect()
        } else {
            vec![]
        };
        self.c.apply(logits);
        for (i, l) in keep {
            logits[i] = l;
        }
    }
    fn accept(&mut self, token: usize) {
        let _ = self.c.accept(token as u32);
    }
}

/// Compile (cached per model) and start a constraint for one generated reply.
pub fn build(slot: &Slot, spec: &Spec) -> Result<Box<dyn Constraint>, String> {
    let key = match spec {
        Spec::JsonObject => "json_object".to_string(),
        Spec::Schema(v) => format!("schema:{v}"),
        Spec::Gbnf(g) => format!("gbnf:{g}"),
        Spec::Regex(r) => format!("regex:{r}"),
    };
    let cached = slot
        .grammars
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&key)
        .cloned();
    let g = match cached {
        Some(g) => g,
        None => {
            let g = Arc::new(
                match spec {
                    Spec::JsonObject => Grammar::json_object(&opts()),
                    Spec::Schema(v) => Grammar::from_json_schema(v, &opts()),
                    Spec::Gbnf(s) => Grammar::from_gbnf(s, "root"),
                    Spec::Regex(r) => Grammar::from_regex(r),
                }
                .map_err(|e| format!("cannot use that structured-output request: {e}"))?,
            );
            let mut m = slot.grammars.lock().unwrap_or_else(|e| e.into_inner());
            if m.len() >= 32 {
                m.clear();
            }
            m.insert(key, g.clone());
            g
        }
    };
    let trie: Arc<TokenTrie> = slot
        .trie
        .get_or_init(|| Arc::new(TokenTrie::from_tokenizer(&slot.loaded.tok)))
        .clone();
    let c = Engine::new(g, trie).map_err(|e| e.to_string())?;
    Ok(Box::new(Adapter {
        c,
        stops: slot.loaded.stop_ids.clone(),
    }))
}

/// Compile a constraint without a model slot (the command line: nothing is cached between calls).
pub fn build_standalone(
    tok: &companion_engine::tokenizer::Tokenizer,
    stops: &[usize],
    spec: &Spec,
) -> Result<Box<dyn Constraint>, String> {
    let g = Arc::new(
        match spec {
            Spec::JsonObject => Grammar::json_object(&opts()),
            Spec::Schema(v) => Grammar::from_json_schema(v, &opts()),
            Spec::Gbnf(s) => Grammar::from_gbnf(s, "root"),
            Spec::Regex(r) => Grammar::from_regex(r),
        }
        .map_err(|e| format!("cannot use that structured-output request: {e}"))?,
    );
    let trie = Arc::new(TokenTrie::from_tokenizer(tok));
    let c = Engine::new(g, trie).map_err(|e| e.to_string())?;
    Ok(Box::new(Adapter {
        c,
        stops: stops.to_vec(),
    }))
}

impl Spec {
    /// `--json`, `--json-schema FILE|JSON`, `--regex PATTERN`, `--grammar FILE` (GBNF). At most one.
    pub fn from_args(args: &crate::args::Args) -> Result<Option<Spec>, String> {
        let mut found: Vec<Spec> = vec![];
        if args.flag("json") {
            found.push(Spec::JsonObject);
        }
        if let Some(s) = args.get("json-schema") {
            let text = if s.trim_start().starts_with('{') {
                s.to_string()
            } else {
                std::fs::read_to_string(s).map_err(|e| format!("--json-schema {s}: {e}"))?
            };
            let v: Value =
                serde_json::from_str(&text).map_err(|e| format!("--json-schema: not JSON: {e}"))?;
            if !v.is_object() {
                return Err("--json-schema: the schema must be a JSON object".into());
            }
            found.push(Spec::Schema(v));
        }
        if let Some(r) = args.get("regex") {
            found.push(Spec::Regex(r.to_string()));
        }
        if let Some(f) = args.get("grammar") {
            found.push(Spec::Gbnf(
                std::fs::read_to_string(f).map_err(|e| format!("--grammar {f}: {e}"))?,
            ));
        }
        if found.len() > 1 {
            return Err("use only one of --json, --json-schema, --regex and --grammar".into());
        }
        Ok(found.pop())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dialects() {
        assert_eq!(Spec::from_body(&json!({})).unwrap(), None);
        assert_eq!(
            Spec::from_body(&json!({"response_format": {"type": "text"}})).unwrap(),
            None
        );
        assert_eq!(
            Spec::from_body(&json!({"response_format": {"type": "json_object"}})).unwrap(),
            Some(Spec::JsonObject)
        );
        let s = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
        assert_eq!(Spec::from_body(&json!({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": s}}})).unwrap(), Some(Spec::Schema(s.clone())));
        assert_eq!(
            Spec::from_body(&json!({"format": "json"})).unwrap(),
            Some(Spec::JsonObject)
        );
        assert_eq!(
            Spec::from_body(&json!({"format": s})).unwrap(),
            Some(Spec::Schema(s.clone()))
        );
        assert_eq!(
            Spec::from_body(&json!({"json_schema": s})).unwrap(),
            Some(Spec::Schema(s))
        );
        assert_eq!(
            Spec::from_body(&json!({"grammar": "root ::= \"a\""})).unwrap(),
            Some(Spec::Gbnf("root ::= \"a\"".into()))
        );
        assert!(Spec::from_body(&json!({"response_format": {"type": "xml"}})).is_err());
        assert!(Spec::from_body(&json!({"response_format": {"type": "json_schema"}})).is_err());
        assert!(Spec::from_body(&json!({"format": 3})).is_err());
    }
}
