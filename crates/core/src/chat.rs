//! A small client for any OpenAI-compatible server (colibri's `coli serve` is one):
//! health, model list and streamed chat completions over plain HTTP + SSE.

use crate::http::{self, CancelToken, RequestOptions};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Pictures as `data:` URLs; sent in the OpenAI "content parts" form.
    #[serde(default)]
    pub images: Vec<String>,
}

impl Serialize for ChatMessage {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut m = ser.serialize_map(Some(2))?;
        m.serialize_entry("role", &self.role)?;
        if self.images.is_empty() {
            m.serialize_entry("content", &self.content)?;
        } else {
            let mut parts = vec![json!({ "type": "text", "text": self.content })];
            parts.extend(
                self.images
                    .iter()
                    .map(|u| json!({ "type": "image_url", "image_url": { "url": u } })),
            );
            m.serialize_entry("content", &parts)?;
        }
        m.end()
    }
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.to_string(),
            content: content.into(),
            images: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Thinking {
    /// Let the model decide (send nothing).
    #[default]
    Auto,
    On,
    Off,
}

#[derive(Clone, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: Option<u32>,
    pub thinking: Thinking,
}

impl ChatRequest {
    pub fn to_body(&self) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": self.messages,
            "stream": true,
            "stream_options": { "include_usage": true },
            "temperature": self.temperature,
            "top_p": self.top_p,
        });
        if let Some(n) = self.max_tokens {
            body["max_tokens"] = json!(n);
        }
        match self.thinking {
            Thinking::Auto => {}
            Thinking::On => body["enable_thinking"] = json!(true),
            Thinking::Off => body["enable_thinking"] = json!(false),
        }
        body
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    Content(String),
    Reasoning(String),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamSummary {
    pub finish_reason: Option<String>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    /// `completion_tokens` was counted from the pieces received (the server sent no usage), so it
    /// is an estimate, not a figure the server reported.
    pub completion_estimated: bool,
    pub first_token_ms: Option<u64>,
    pub total_ms: u64,
    pub cancelled: bool,
}

impl StreamSummary {
    /// Generation speed, from the first token to the last.
    pub fn tokens_per_second(&self) -> Option<f64> {
        let n = self.completion_tokens? as f64;
        let ms = self.total_ms.checked_sub(self.first_token_ms?)? as f64;
        (n > 1.0 && ms > 50.0).then(|| (n - 1.0) / (ms / 1000.0))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// Nothing is listening: the server is not running.
    Unreachable,
    /// The server is up but still loading the model, or its queue is full.
    Busy,
    Unauthorized,
    ContextTooLong,
    BadRequest,
    Server,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatError {
    pub kind: ErrorKind,
    pub message: String,
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ChatError {}

fn classify(status: u16, code: &str, message: &str) -> ErrorKind {
    if code == "context_length_exceeded"
        || message.contains("context") && message.contains("exceed")
    {
        ErrorKind::ContextTooLong
    } else if status == 401 || status == 403 {
        ErrorKind::Unauthorized
    } else if status == 429 || status == 503 {
        ErrorKind::Busy
    } else if (400..500).contains(&status) {
        ErrorKind::BadRequest
    } else {
        ErrorKind::Server
    }
}

/// Pull `{"error":{"message":..,"code":..}}` out of an error body.
pub fn parse_error_body(status: u16, body: &str) -> ChatError {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let message = v
        .pointer("/error/message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| {
            let t = body.trim();
            if t.is_empty() {
                format!("HTTP {status}")
            } else {
                t.chars().take(300).collect()
            }
        });
    let code = v
        .pointer("/error/code")
        .and_then(Value::as_str)
        .unwrap_or("");
    ChatError {
        kind: classify(status, code, &message),
        message,
    }
}

// ------------------------------------------------------------------ <think> splitting

/// Some servers put the reasoning inside the text, between `<think>` and `</think>`.
/// This splits a stream of chunks (tags may be cut anywhere) into reasoning and answer.
#[derive(Debug, Default)]
pub struct ThinkSplitter {
    inside: bool,
    pending: String,
}

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

impl ThinkSplitter {
    pub fn feed(&mut self, chunk: &str) -> Vec<StreamEvent> {
        self.pending.push_str(chunk);
        let mut out = Vec::new();
        loop {
            let tag = if self.inside { CLOSE } else { OPEN };
            if let Some(pos) = self.pending.find(tag) {
                let before: String = self.pending[..pos].to_string();
                self.emit(&mut out, before);
                self.pending = self.pending[pos + tag.len()..].to_string();
                self.inside = !self.inside;
                continue;
            }
            // Keep a possible partial tag at the end (e.g. "<thi") for the next chunk.
            let keep = partial_suffix_len(&self.pending, tag);
            let cut = self.pending.len() - keep;
            let ready: String = self.pending[..cut].to_string();
            self.emit(&mut out, ready);
            self.pending = self.pending[cut..].to_string();
            break;
        }
        out
    }

    /// Flush whatever is held back when the stream ends.
    pub fn finish(&mut self) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.pending);
        self.emit(&mut out, rest);
        out
    }

    fn emit(&self, out: &mut Vec<StreamEvent>, text: String) {
        if text.is_empty() {
            return;
        }
        out.push(if self.inside {
            StreamEvent::Reasoning(text)
        } else {
            StreamEvent::Content(text)
        });
    }
}

/// Longest suffix of `text` that is a proper prefix of `tag`.
fn partial_suffix_len(text: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&n| {
            text.is_char_boundary(text.len().saturating_sub(n))
                && n <= text.len()
                && tag.starts_with(&text[text.len() - n..])
        })
        .unwrap_or(0)
}

// ------------------------------------------------------------------ SSE line parsing

#[derive(Debug, PartialEq)]
pub enum SseLine {
    Data(String),
    Done,
    Ignore,
}

pub fn parse_sse_line(line: &str) -> SseLine {
    let line = line.trim_end_matches(['\r', '\n']);
    let Some(rest) = line.strip_prefix("data:") else {
        return SseLine::Ignore;
    };
    let data = rest.strip_prefix(' ').unwrap_or(rest);
    if data.trim() == "[DONE]" {
        SseLine::Done
    } else {
        SseLine::Data(data.to_string())
    }
}

// ------------------------------------------------------------------ health

#[derive(Clone, Debug, PartialEq)]
pub enum Health {
    /// Nobody answers at this address.
    Down,
    /// Something answers but is not ready (loading the model, restarting).
    Loading(String),
    Ready(ServerFacts),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServerFacts {
    pub arch: Option<String>,
    pub context_tokens: Option<u64>,
    pub max_tokens: Option<u64>,
    pub kv_slots: Option<u64>,
    pub ram_used_gb: Option<f64>,
    pub supports_images: bool,
}

impl ServerFacts {
    pub fn from_health(v: &Value) -> Self {
        Self {
            arch: v.get("arch").and_then(Value::as_str).map(str::to_string),
            context_tokens: v.get("context_tokens").and_then(Value::as_u64),
            max_tokens: v.get("max_tokens").and_then(Value::as_u64),
            kv_slots: v.get("kv_slots").and_then(Value::as_u64),
            ram_used_gb: v.pointer("/system/rss_gb").and_then(Value::as_f64),
            supports_images: v
                .get("input_modalities")
                .and_then(Value::as_array)
                .is_some_and(|a| a.iter().any(|m| m.as_str() == Some("image"))),
        }
    }
}

// ------------------------------------------------------------------ client

/// What `/v1/packs/complete` answered.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackReply {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub rows_copied: u64,
    pub rows_recomputed: u64,
    /// Rows that did not have to go through the network because an earlier run had them.
    pub rows_saved: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PackFail {
    Unsupported,
    Error(ChatError),
}

#[derive(Clone, Debug)]
pub struct ChatClient {
    /// Without trailing slash and without `/v1`, e.g. `http://127.0.0.1:8000`.
    pub base: String,
    pub api_key: Option<String>,
}

/// Accepts `http://host:port`, `http://host:port/` or `http://host:port/v1`.
pub fn normalize_base(url: &str) -> String {
    let u = url.trim().trim_end_matches('/');
    let u = u.strip_suffix("/v1").unwrap_or(u);
    if u.starts_with("http://") || u.starts_with("https://") {
        u.to_string()
    } else {
        format!("http://{u}")
    }
}

impl ChatClient {
    pub fn new(base: &str, api_key: Option<String>) -> Self {
        Self {
            base: normalize_base(base),
            api_key: api_key.filter(|k| !k.trim().is_empty()),
        }
    }

    fn opts<'a>(
        &'a self,
        read_timeout: Option<Duration>,
        cancel: Option<&'a CancelToken>,
    ) -> RequestOptions<'a> {
        RequestOptions {
            connect_timeout: Duration::from_secs(3),
            read_timeout,
            bearer: self.api_key.as_deref(),
            cancel,
        }
    }

    pub fn health(&self) -> Health {
        let Ok(resp) = http::send(
            &self.base,
            "GET",
            "/health",
            None,
            &self.opts(Some(Duration::from_secs(5)), None),
        ) else {
            return Health::Down;
        };
        let status = resp.status;
        let v: Value = serde_json::from_str(&resp.text()).unwrap_or(Value::Null);
        match status {
            200 if v.get("status").and_then(Value::as_str).unwrap_or("ok") == "ok" => {
                Health::Ready(ServerFacts::from_health(&v))
            }
            _ => Health::Loading(
                v.get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("loading")
                    .to_string(),
            ),
        }
    }

    /// Ask corbul to answer from `packs` (texts it keeps the work for) and `question`: the
    /// non-streaming `/v1/packs/complete`. [`PackFail::Unsupported`] means this server has no such
    /// endpoint (any other engine, or corbul started without `--packs`): use the ordinary chat.
    pub fn pack_complete(
        &self,
        packs: &[String],
        question: &str,
        max_tokens: u32,
        cancel: &CancelToken,
    ) -> Result<PackReply, PackFail> {
        let body = serde_json::json!({
            "packs": packs, "question": question, "max_tokens": max_tokens,
        });
        let resp = http::send(
            &self.base,
            "POST",
            "/v1/packs/complete",
            Some(&body.to_string()),
            &self.opts(None, Some(cancel)),
        )
        .map_err(|e| PackFail::Error(unreachable(e)))?;
        let (status, text) = (resp.status, resp.text());
        if matches!(status, 404 | 405 | 501) {
            return Err(PackFail::Unsupported);
        }
        if status != 200 {
            return Err(PackFail::Error(parse_error_body(status, &text)));
        }
        let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        let n = |p: &str| v.pointer(p).and_then(Value::as_u64).unwrap_or(0);
        Ok(PackReply {
            text: v["text"].as_str().unwrap_or("").to_string(),
            prompt_tokens: n("/usage/prompt_tokens"),
            completion_tokens: n("/usage/completion_tokens"),
            rows_copied: n("/packs/rows_copied"),
            rows_recomputed: n("/packs/rows_recomputed"),
            rows_saved: n("/packs/rows_saved"),
        })
    }

    pub fn model_ids(&self) -> Result<Vec<String>, ChatError> {
        let resp = http::send(
            &self.base,
            "GET",
            "/v1/models",
            None,
            &self.opts(Some(Duration::from_secs(8)), None),
        )
        .map_err(unreachable)?;
        let status = resp.status;
        let body = resp.text();
        if status != 200 {
            return Err(parse_error_body(status, &body));
        }
        let v: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        Ok(v["data"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|m| m["id"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Stream one completion. `on_event` is called for every piece of text as it
    /// arrives. `cancel.cancel()` from another thread stops it at once: the
    /// connection is closed, which also tells the server to stop generating.
    pub fn stream_chat(
        &self,
        req: &ChatRequest,
        cancel: &CancelToken,
        mut on_event: impl FnMut(StreamEvent),
    ) -> Result<StreamSummary, ChatError> {
        let started = Instant::now();
        // No timeout: a big model may think for minutes before the first token.
        let resp = http::send(
            &self.base,
            "POST",
            "/v1/chat/completions",
            Some(&req.to_body().to_string()),
            &self.opts(None, Some(cancel)),
        )
        .map_err(unreachable)?;
        if resp.status != 200 {
            let status = resp.status;
            return Err(parse_error_body(status, &resp.text()));
        }

        let mut body = resp.body;
        let mut summary = StreamSummary::default();
        let mut splitter = ThinkSplitter::default();
        let mut chunks = 0u64;

        let mut emit = |ev: StreamEvent, summary: &mut StreamSummary| {
            if summary.first_token_ms.is_none() {
                summary.first_token_ms = Some(started.elapsed().as_millis() as u64);
            }
            on_event(ev);
        };

        loop {
            if cancel.is_cancelled() {
                break;
            }
            let line = match body.read_line_lossy() {
                Ok(Some(l)) => l,
                Ok(None) => break,
                Err(e) => {
                    // A reset after some text arrived (or a cancel) still leaves a usable partial answer.
                    if cancel.is_cancelled() || summary.first_token_ms.is_some() {
                        break;
                    }
                    return Err(ChatError {
                        kind: ErrorKind::Server,
                        message: e.to_string(),
                    });
                }
            };
            let data = match parse_sse_line(&line) {
                SseLine::Data(d) => d,
                SseLine::Done => break,
                SseLine::Ignore => continue,
            };
            let Ok(v) = serde_json::from_str::<Value>(&data) else {
                continue;
            };
            if let Some(err) = v.get("error") {
                let message = err["message"]
                    .as_str()
                    .unwrap_or("the server reported an error")
                    .to_string();
                return Err(ChatError {
                    kind: classify(500, err["code"].as_str().unwrap_or(""), &message),
                    message,
                });
            }
            if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
                summary.prompt_tokens = u["prompt_tokens"].as_u64().or(summary.prompt_tokens);
                summary.completion_tokens = u["completion_tokens"]
                    .as_u64()
                    .or(summary.completion_tokens);
            }
            let Some(choice) = v["choices"].get(0) else {
                continue;
            };
            if let Some(reason) = choice["finish_reason"].as_str() {
                summary.finish_reason = Some(reason.to_string());
            }
            let delta = &choice["delta"];
            for key in ["reasoning_content", "reasoning"] {
                if let Some(t) = delta[key].as_str().filter(|t| !t.is_empty()) {
                    chunks += 1;
                    emit(StreamEvent::Reasoning(t.to_string()), &mut summary);
                }
            }
            if let Some(t) = delta["content"].as_str().filter(|t| !t.is_empty()) {
                chunks += 1;
                for ev in splitter.feed(t) {
                    emit(ev, &mut summary);
                }
            }
        }
        for ev in splitter.finish() {
            emit(ev, &mut summary);
        }
        summary.cancelled = cancel.is_cancelled();
        summary.total_ms = started.elapsed().as_millis() as u64;
        if summary.completion_tokens.is_none() && chunks > 0 {
            summary.completion_tokens = Some(chunks); // one chunk is about one token
            summary.completion_estimated = true;
        }
        Ok(summary)
    }
}

fn unreachable(e: std::io::Error) -> ChatError {
    ChatError {
        kind: ErrorKind::Unreachable,
        message: e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_shot(reply: &'static str) -> String {
        use std::io::{Read, Write};
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut c, _)) = l.accept() {
                // Read the whole request first: closing with unread bytes makes Windows reset the
                // connection before the client has read the reply.
                let mut got = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let n = c.read(&mut buf).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    got.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&got).to_lowercase();
                    if let Some(h) = text.find("\r\n\r\n") {
                        let len = text[..h]
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        if got.len() >= h + 4 + len {
                            break;
                        }
                    }
                }
                let _ = c.write_all(reply.as_bytes());
                let _ = c.flush();
                let _ = c.shutdown(std::net::Shutdown::Write);
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn pack_complete_reads_the_answer_and_tells_unsupported_apart() {
        let ok = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 142\r\nConnection: close\r\n\r\n{\"text\":\"six\",\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":1},\"packs\":{\"rows_copied\":7,\"rows_recomputed\":2,\"rows_saved\":5}}";
        let body = ok.split("\r\n\r\n").nth(1).unwrap().len();
        let ok = Box::leak(ok.replace("142", &body.to_string()).into_boxed_str());
        let c = ChatClient::new(&one_shot(ok), None);
        let r = c
            .pack_complete(&["a".into()], "q", 8, &CancelToken::new())
            .unwrap();
        assert_eq!(
            (r.text.as_str(), r.rows_saved, r.prompt_tokens),
            ("six", 5, 9)
        );
        let c = ChatClient::new(
            &one_shot(
                "HTTP/1.1 501 Not Implemented\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
            ),
            None,
        );
        assert_eq!(
            c.pack_complete(&["a".into()], "q", 8, &CancelToken::new()),
            Err(PackFail::Unsupported)
        );
    }
    use std::sync::Arc;
    use std::thread;

    fn texts(events: &[StreamEvent]) -> (String, String) {
        let (mut c, mut r) = (String::new(), String::new());
        for e in events {
            match e {
                StreamEvent::Content(t) => c.push_str(t),
                StreamEvent::Reasoning(t) => r.push_str(t),
            }
        }
        (c, r)
    }

    #[test]
    fn splitter_handles_tags_cut_anywhere() {
        let full = "<think>I should greet.</think>Hello there!";
        // Try every possible split point of the stream into two chunks, and char-by-char.
        for cut in 0..=full.len() {
            let mut s = ThinkSplitter::default();
            let mut ev = s.feed(&full[..cut]);
            ev.extend(s.feed(&full[cut..]));
            ev.extend(s.finish());
            let (c, r) = texts(&ev);
            assert_eq!(
                (c.as_str(), r.as_str()),
                ("Hello there!", "I should greet."),
                "cut at {cut}"
            );
        }
        let mut s = ThinkSplitter::default();
        let mut ev = Vec::new();
        for ch in full.chars() {
            ev.extend(s.feed(&ch.to_string()));
        }
        ev.extend(s.finish());
        assert_eq!(
            texts(&ev),
            ("Hello there!".to_string(), "I should greet.".to_string())
        );
    }

    #[test]
    fn splitter_leaves_ordinary_angle_brackets_and_unicode_alone() {
        let mut s = ThinkSplitter::default();
        let mut ev = s.feed("a < b and <thin");
        ev.extend(s.feed("g> ș ț ă ✓"));
        ev.extend(s.finish());
        assert_eq!(texts(&ev).0, "a < b and <thing> ș ț ă ✓");
    }

    #[test]
    fn sse_lines() {
        assert_eq!(
            parse_sse_line("data: {\"a\":1}\n"),
            SseLine::Data("{\"a\":1}".into())
        );
        assert_eq!(parse_sse_line("data: [DONE]\r\n"), SseLine::Done);
        assert_eq!(parse_sse_line(": keep-alive"), SseLine::Ignore);
        assert_eq!(parse_sse_line(""), SseLine::Ignore);
    }

    #[test]
    fn base_urls_are_forgiving() {
        assert_eq!(normalize_base("127.0.0.1:8000"), "http://127.0.0.1:8000");
        assert_eq!(
            normalize_base(" http://localhost:8000/v1/ "),
            "http://localhost:8000"
        );
    }

    #[test]
    fn error_bodies() {
        let e = parse_error_body(
            400,
            r#"{"error":{"message":"too long","code":"context_length_exceeded"}}"#,
        );
        assert_eq!(e.kind, ErrorKind::ContextTooLong);
        assert_eq!(parse_error_body(503, "").kind, ErrorKind::Busy);
        assert_eq!(parse_error_body(401, "no").kind, ErrorKind::Unauthorized);
    }

    #[test]
    fn request_body_shape() {
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![ChatMessage::new("user", "hi")],
            temperature: 0.5,
            top_p: 0.9,
            max_tokens: None,
            thinking: Thinking::Off,
        };
        let b = req.to_body();
        assert_eq!(b["stream"], true);
        assert_eq!(b["enable_thinking"], false);
        assert!(b.get("max_tokens").is_none());
        let auto = ChatRequest {
            thinking: Thinking::Auto,
            ..req
        };
        assert!(auto.to_body().get("enable_thinking").is_none());
    }

    /// A tiny OpenAI-style server: streams a reasoning delta, the answer in pieces, usage.
    fn serve_once(chunks: Vec<String>, status: u16) -> (String, thread::JoinHandle<()>) {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let addr = format!("http://{}", server.server_addr().to_ip().unwrap());
        let handle = thread::spawn(move || {
            let mut request = server.recv().unwrap();
            let mut body = String::new();
            std::io::Read::read_to_string(request.as_reader(), &mut body).unwrap();
            assert!(body.contains("\"stream\":true"));
            let payload: String = chunks.concat();
            let response = tiny_http::Response::from_string(payload)
                .with_status_code(status)
                .with_header(
                    "Content-Type: text/event-stream"
                        .parse::<tiny_http::Header>()
                        .unwrap(),
                );
            request.respond(response).unwrap();
        });
        (addr, handle)
    }

    fn sse(v: Value) -> String {
        format!("data: {}\n\n", v)
    }

    #[test]
    fn streams_reasoning_content_and_usage() {
        let chunks = vec![
            sse(json!({"choices":[{"delta":{"role":"assistant","content":""}}]})),
            sse(json!({"choices":[{"delta":{"reasoning_content":"hmm "}}]})),
            sse(json!({"choices":[{"delta":{"content":"Bună "}}]})),
            sse(json!({"choices":[{"delta":{"content":"ziua!"}}]})),
            sse(json!({"choices":[{"delta":{},"finish_reason":"stop"}]})),
            sse(json!({"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}})),
            "data: [DONE]\n\n".to_string(),
        ];
        let (addr, h) = serve_once(chunks, 200);
        let client = ChatClient::new(&addr, None);
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![ChatMessage::new("user", "salut")],
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: None,
            thinking: Thinking::Auto,
        };
        let mut events = Vec::new();
        let summary = client
            .stream_chat(&req, &CancelToken::new(), |e| events.push(e))
            .unwrap();
        h.join().unwrap();
        assert_eq!(
            texts(&events),
            ("Bună ziua!".to_string(), "hmm ".to_string())
        );
        assert_eq!(summary.finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            (summary.prompt_tokens, summary.completion_tokens),
            (Some(5), Some(3))
        );
        // the server reported the count: it is not an estimate, even if it equals the piece count
        assert!(!summary.completion_estimated);
        assert!(!summary.cancelled);
    }

    #[test]
    fn a_stream_without_usage_is_marked_as_estimated() {
        let chunks = vec![
            sse(json!({"choices":[{"delta":{"content":"a "}}]})),
            sse(json!({"choices":[{"delta":{"content":"b"}}]})),
            "data: [DONE]\n\n".to_string(),
        ];
        let (addr, h) = serve_once(chunks, 200);
        let client = ChatClient::new(&addr, None);
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![ChatMessage::new("user", "x")],
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: None,
            thinking: Thinking::Auto,
        };
        let summary = client
            .stream_chat(&req, &CancelToken::new(), |_| {})
            .unwrap();
        h.join().unwrap();
        assert_eq!(summary.completion_tokens, Some(2));
        assert!(summary.completion_estimated);
    }

    #[test]
    fn splits_think_tags_inside_content() {
        let chunks = vec![
            sse(json!({"choices":[{"delta":{"content":"<thi"}}]})),
            sse(json!({"choices":[{"delta":{"content":"nk>plan</thi"}}]})),
            sse(json!({"choices":[{"delta":{"content":"nk>Answer"}}]})),
            "data: [DONE]\n\n".to_string(),
        ];
        let (addr, h) = serve_once(chunks, 200);
        let client = ChatClient::new(&addr, None);
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![],
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: None,
            thinking: Thinking::Auto,
        };
        let mut events = Vec::new();
        client
            .stream_chat(&req, &CancelToken::new(), |e| events.push(e))
            .unwrap();
        h.join().unwrap();
        assert_eq!(texts(&events), ("Answer".to_string(), "plan".to_string()));
    }

    #[test]
    fn reports_server_errors_in_plain_terms() {
        let (addr, h) = serve_once(
            vec![r#"{"error":{"message":"queue full","code":"queue_full"}}"#.to_string()],
            429,
        );
        let client = ChatClient::new(&addr, None);
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![],
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: None,
            thinking: Thinking::Auto,
        };
        let err = client
            .stream_chat(&req, &CancelToken::new(), |_| {})
            .unwrap_err();
        h.join().unwrap();
        assert_eq!(err.kind, ErrorKind::Busy);
        assert_eq!(err.message, "queue full");
    }

    #[test]
    fn nothing_listening_is_unreachable_and_health_down() {
        let client = ChatClient::new("http://127.0.0.1:9", None);
        assert_eq!(client.health(), Health::Down);
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![],
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: None,
            thinking: Thinking::Auto,
        };
        assert_eq!(
            client
                .stream_chat(&req, &CancelToken::new(), |_| {})
                .unwrap_err()
                .kind,
            ErrorKind::Unreachable
        );
    }

    #[test]
    fn health_reads_facts() {
        let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").unwrap());
        let addr = format!("http://{}", server.server_addr().to_ip().unwrap());
        let s2 = server.clone();
        let h = thread::spawn(move || {
            let r = s2.recv().unwrap();
            let body = json!({"status":"ok","arch":"qwen36","context_tokens":32768,"max_tokens":4096,"input_modalities":["text","image"],"system":{"rss_gb":9.9}});
            r.respond(tiny_http::Response::from_string(body.to_string()))
                .unwrap();
        });
        match ChatClient::new(&addr, None).health() {
            Health::Ready(f) => {
                assert_eq!(f.arch.as_deref(), Some("qwen36"));
                assert_eq!(f.context_tokens, Some(32768));
                assert!(f.supports_images);
                assert_eq!(f.ram_used_gb, Some(9.9));
            }
            other => panic!("{other:?}"),
        }
        h.join().unwrap();
    }

    #[test]
    fn pictures_go_out_as_content_parts() {
        let plain = serde_json::to_value(ChatMessage::new("user", "salut")).unwrap();
        assert_eq!(plain, json!({"role": "user", "content": "salut"}));
        let mut m = ChatMessage::new("user", "ce e asta?");
        m.images.push("data:image/png;base64,AAAA".into());
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(
            v["content"][0],
            json!({"type": "text", "text": "ce e asta?"})
        );
        assert_eq!(
            v["content"][1]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
    }

    /// Speed of the client side alone: 200 000 tiny pieces from a local server, measured in
    /// debug mode, so a real build is faster still. Fails if the client cannot keep up with a
    /// model far quicker than any that runs on a home computer.
    #[test]
    fn the_client_keeps_up_with_a_very_fast_model() {
        let n = 200_000usize;
        let chunks: Vec<String> = (0..n)
            .map(|i| sse(json!({"choices":[{"delta":{"content":format!("w{} ", i % 10)}}]})))
            .chain(std::iter::once("data: [DONE]\n\n".to_string()))
            .collect();
        let (addr, h) = serve_once(chunks, 200);
        let client = ChatClient::new(&addr, None);
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![ChatMessage::new("user", "x")],
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: None,
            thinking: Thinking::Off,
        };
        let start = std::time::Instant::now();
        let mut got = 0usize;
        client
            .stream_chat(&req, &CancelToken::new(), |e| {
                if matches!(e, StreamEvent::Content(_)) {
                    got += 1;
                }
            })
            .unwrap();
        h.join().unwrap();
        let rate = got as f64 / start.elapsed().as_secs_f64();
        eprintln!("client throughput: {rate:.0} pieces/s");
        assert_eq!(got, n);
        // a slow shared runner manages well under a laptop; a quadratic reader would be orders slower
        assert!(rate > 3_000.0, "too slow: {rate:.0}/s");
    }
}
