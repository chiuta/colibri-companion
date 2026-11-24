//! Anthropic Messages API (`POST /v1/messages`): request → our messages and tools, replies and the streaming
//! event sequence (`message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop`).
use crate::template::Message;
use crate::tools::{self, ToolChoice};
use serde_json::{Value, json};

pub struct Parsed {
    pub messages: Vec<Message>,
    pub tools: Vec<Value>,
    pub choice: ToolChoice,
    pub stops: Vec<String>,
}

fn text_of(v: Option<&Value>) -> Result<String, String> {
    match v {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(a)) => Ok(a
            .iter()
            .filter_map(|b| match b {
                Value::String(s) => Some(s.as_str()),
                Value::Object(o)
                    if o.get("type")
                        .and_then(Value::as_str)
                        .is_none_or(|t| t == "text") =>
                {
                    o.get("text").and_then(Value::as_str)
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")),
        Some(_) => Err("`content` must be a string or a list of content blocks".into()),
    }
}

/// `system`, `messages`, `tools`, `tool_choice`, `stop_sequences` of a Messages request.
pub fn parse(body: &Value) -> Result<Parsed, String> {
    let mut messages = vec![];
    let system = text_of(body.get("system"))?;
    if !system.is_empty() {
        messages.push(Message::new("system", &system));
    }
    let Some(Value::Array(list)) = body.get("messages") else {
        return Err("`messages` must be a list".into());
    };
    if list.is_empty() {
        return Err("`messages` is empty".into());
    }
    if list.len() > 512 {
        return Err("too many messages".into());
    }
    for (i, m) in list.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("message {i} needs a `role`"))?;
        if role != "user" && role != "assistant" {
            return Err(format!(
                "message {i}: `role` must be \"user\" or \"assistant\""
            ));
        }
        let blocks = match m.get("content") {
            Some(Value::Array(a)) => a.clone(),
            Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
            None | Some(Value::Null) => vec![],
            Some(_) => {
                return Err(format!(
                    "message {i}: `content` must be a string or a list of blocks"
                ));
            }
        };
        let (mut text, mut calls, mut results) = (String::new(), vec![], vec![]);
        for b in &blocks {
            match b.get("type").and_then(Value::as_str).unwrap_or("text") {
                "text" => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
                "tool_use" => {
                    let name = b
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| format!("message {i}: a `tool_use` block needs a `name`"))?;
                    let id = b.get("id").and_then(Value::as_str).unwrap_or("");
                    let input = b.get("input").cloned().unwrap_or_else(|| json!({}));
                    calls.push(json!({"id": id, "type": "function", "function": {"name": name, "arguments": input.to_string()}}));
                }
                "tool_result" => {
                    let id = b
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let mut c = text_of(b.get("content"))?;
                    if b.get("is_error").and_then(Value::as_bool) == Some(true) {
                        c = format!("Error: {c}");
                    }
                    results.push((id, c));
                }
                // images, documents, thinking: not supported by text models
                _ => {}
            }
        }
        if role == "assistant" {
            let mut msg = Message::new("assistant", &text);
            msg.tool_calls = calls;
            messages.push(msg);
        } else {
            // tool results come first (they answer the previous assistant turn), then whatever the user typed
            for (id, c) in results {
                let mut t = Message::new("tool", &c);
                t.tool_call_id = Some(id);
                messages.push(t);
            }
            if !text.is_empty() || blocks.is_empty() {
                messages.push(Message::new("user", &text));
            }
        }
    }
    let tools = tools::normalize_tools(body.get("tools"))?;
    let choice = tools::parse_choice(body.get("tool_choice"))?;
    let stops = match body.get("stop_sequences") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect(),
        _ => vec![],
    };
    Ok(Parsed {
        messages,
        tools,
        choice,
        stops,
    })
}

pub fn stop_reason(finish: &str, stop_sequence: &Option<String>) -> &'static str {
    match finish {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ if stop_sequence.is_some() => "stop_sequence",
        _ => "end_turn",
    }
}

/// `input` of a tool_use block: the arguments string parsed (an object), or `{}` when it is not valid JSON.
fn input_of(arguments: &str) -> Value {
    serde_json::from_str(arguments)
        .ok()
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}))
}

#[allow(clippy::too_many_arguments)]
pub fn message_json(
    id: &str,
    model: &str,
    text: &str,
    calls: &[(String, String, String)],
    stop_reason: &str,
    stop_sequence: &Option<String>,
    input_tokens: usize,
    output_tokens: usize,
) -> Value {
    let mut content = vec![];
    if !text.is_empty() || calls.is_empty() {
        content.push(json!({"type": "text", "text": text}));
    }
    for (cid, name, args) in calls {
        content.push(json!({"type": "tool_use", "id": cid, "name": name, "input": input_of(args)}));
    }
    json!({"id": id, "type": "message", "role": "assistant", "model": model, "content": content, "stop_reason": stop_reason, "stop_sequence": stop_sequence,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}})
}

pub fn error_body(kind: &str, msg: &str) -> Value {
    let t = match kind {
        "server_error" => "api_error",
        "authentication_error"
        | "not_found_error"
        | "rate_limit_error"
        | "overloaded_error"
        | "invalid_request_error" => kind,
        _ => "api_error",
    };
    json!({"type": "error", "error": {"type": t, "message": msg}})
}

/// One server-sent event: (event name, data).
pub type Ev = (&'static str, Value);

pub fn sse_bytes(e: &Ev) -> Vec<u8> {
    format!("event: {}\ndata: {}\n\n", e.0, e.1).into_bytes()
}

/// The streaming state machine: text blocks open lazily, a tool call closes the text block and is a block of its own.
#[derive(Default)]
pub struct StreamState {
    next: usize,
    open_text: Option<usize>,
    blocks: usize,
}
impl StreamState {
    pub fn start(&mut self, id: &str, model: &str, input_tokens: usize) -> Vec<Ev> {
        vec![
            (
                "message_start",
                json!({"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}}}),
            ),
            ("ping", json!({"type": "ping"})),
        ]
    }
    fn close_text(&mut self, out: &mut Vec<Ev>) {
        if let Some(i) = self.open_text.take() {
            out.push((
                "content_block_stop",
                json!({"type": "content_block_stop", "index": i}),
            ));
        }
    }
    pub fn text(&mut self, piece: &str) -> Vec<Ev> {
        let mut out = vec![];
        if piece.is_empty() {
            return out;
        }
        let i = match self.open_text {
            Some(i) => i,
            None => {
                let i = self.next;
                self.next += 1;
                self.blocks += 1;
                self.open_text = Some(i);
                out.push(("content_block_start", json!({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})));
                i
            }
        };
        out.push(("content_block_delta", json!({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": piece}})));
        out
    }
    pub fn call(&mut self, id: &str, name: &str, arguments: &str) -> Vec<Ev> {
        let mut out = vec![];
        self.close_text(&mut out);
        let i = self.next;
        self.next += 1;
        self.blocks += 1;
        out.push(("content_block_start", json!({"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}})));
        let args = if serde_json::from_str::<Value>(arguments).is_ok_and(|v| v.is_object()) {
            arguments.to_string()
        } else {
            "{}".to_string()
        };
        out.push(("content_block_delta", json!({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": args}})));
        out.push((
            "content_block_stop",
            json!({"type": "content_block_stop", "index": i}),
        ));
        out
    }
    pub fn finish(
        &mut self,
        stop_reason: &str,
        stop_sequence: &Option<String>,
        output_tokens: usize,
    ) -> Vec<Ev> {
        let mut out = vec![];
        if self.blocks == 0 {
            out.push(("content_block_start", json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})));
            out.push((
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0}),
            ));
        }
        self.close_text(&mut out);
        out.push(("message_delta", json!({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence}, "usage": {"output_tokens": output_tokens}})));
        out.push(("message_stop", json!({"type": "message_stop"})));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(evs: &[Ev]) -> Vec<&'static str> {
        evs.iter().map(|e| e.0).collect()
    }

    #[test]
    fn plain_text_stream_order() {
        let mut s = StreamState::default();
        let mut all = s.start("msg_1", "m", 7);
        all.extend(s.text("Hel"));
        all.extend(s.text("lo"));
        all.extend(s.finish("end_turn", &None, 2));
        assert_eq!(
            names(&all),
            [
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert_eq!(all[0].1["message"]["usage"]["input_tokens"], 7);
        assert_eq!(all[3].1["delta"]["text"], "Hel");
        assert_eq!(all[6].1["delta"]["stop_reason"], "end_turn");
        assert_eq!(all[6].1["usage"]["output_tokens"], 2);
    }

    #[test]
    fn text_then_two_tool_calls_have_increasing_indices() {
        let mut s = StreamState::default();
        let mut all = s.start("msg_1", "m", 1);
        all.extend(s.text("Let me look."));
        all.extend(s.call("toolu_1", "get_weather", "{\"city\":\"Paris\"}"));
        all.extend(s.call("toolu_2", "search", "not json"));
        all.extend(s.finish("tool_use", &None, 9));
        assert_eq!(
            names(&all),
            [
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        let idx: Vec<i64> = all
            .iter()
            .filter(|e| e.0 == "content_block_start")
            .map(|e| e.1["index"].as_i64().unwrap())
            .collect();
        assert_eq!(idx, [0, 1, 2]);
        assert_eq!(all[5].1["content_block"]["type"], "tool_use");
        assert_eq!(all[6].1["delta"]["type"], "input_json_delta");
        assert_eq!(all[6].1["delta"]["partial_json"], "{\"city\":\"Paris\"}");
        assert_eq!(all[9].1["delta"]["partial_json"], "{}");
        // every start has a matching stop, in order, and no delta arrives outside its block
        let mut open: Option<i64> = None;
        for e in &all {
            match e.0 {
                "content_block_start" => {
                    assert!(open.is_none());
                    open = e.1["index"].as_i64();
                }
                "content_block_delta" => assert_eq!(open, e.1["index"].as_i64()),
                "content_block_stop" => {
                    assert_eq!(open, e.1["index"].as_i64());
                    open = None;
                }
                _ => {}
            }
        }
        assert!(open.is_none());
        assert_eq!(all[11].1["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn empty_reply_still_has_one_block() {
        let mut s = StreamState::default();
        let mut all = s.start("m", "m", 0);
        all.extend(s.text(""));
        all.extend(s.finish("max_tokens", &None, 0));
        assert_eq!(
            names(&all),
            [
                "message_start",
                "ping",
                "content_block_start",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
    }

    #[test]
    fn parse_a_tool_round_trip() {
        let body = json!({
            "system": [{"type": "text", "text": "Be brief."}],
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "content": [{"type": "text", "text": "Checking."}, {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "18C"}]}, {"type": "text", "text": "thanks"}]}
            ],
            "tools": [{"name": "get_weather", "description": "d", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "stop_sequences": ["\n\nHuman:"]
        });
        let p = parse(&body).unwrap();
        let roles: Vec<&str> = p.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool", "user"]);
        assert_eq!(
            p.messages[2].tool_calls[0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            p.messages[2].tool_calls[0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
        assert_eq!(p.messages[3].content, "18C");
        assert_eq!(p.messages[3].tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(p.messages[4].content, "thanks");
        assert_eq!(p.tools[0]["function"]["parameters"]["type"], "object");
        assert_eq!(p.choice, ToolChoice::Named("get_weather".into()));
        assert_eq!(p.stops, vec!["\n\nHuman:"]);
    }

    #[test]
    fn parse_rejects_nonsense() {
        for bad in [
            json!({}),
            json!({"messages": []}),
            json!({"messages": [{"role": "system", "content": "x"}]}),
            json!({"messages": [{"role": "user", "content": 5}]}),
            json!({"messages": [{"content": "x"}]}),
            json!({"messages": [{"role": "user", "content": "x"}], "tools": [{"description": "no name"}]}),
        ] {
            assert!(parse(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn message_json_shapes() {
        let m = message_json(
            "msg_1",
            "m",
            "",
            &[("toolu_1".into(), "f".into(), "{\"a\":1}".into())],
            "tool_use",
            &None,
            3,
            4,
        );
        assert_eq!(m["content"][0]["type"], "tool_use");
        assert_eq!(m["content"][0]["input"], json!({"a": 1}));
        assert_eq!(m["usage"], json!({"input_tokens": 3, "output_tokens": 4}));
        let m = message_json("msg_1", "m", "hi", &[], "end_turn", &Some("X".into()), 1, 1);
        assert_eq!(m["content"][0]["text"], "hi");
        assert_eq!(m["stop_sequence"], "X");
        assert_eq!(stop_reason("stop", &Some("X".into())), "stop_sequence");
        assert_eq!(stop_reason("length", &None), "max_tokens");
        assert_eq!(stop_reason("tool_calls", &None), "tool_use");
        assert_eq!(
            error_body("server_error", "x")["error"]["type"],
            "api_error"
        );
        assert_eq!(error_body("rate_limit_error", "x")["type"], "error");
    }
}
