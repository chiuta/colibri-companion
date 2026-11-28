//! `corbul serve` for models that do not go through the dense loader (Gemma 4, state-space and
//! hybrid models): a small OpenAI-compatible server over `family::AnyModel`.
//!
//! What it answers: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` and
//! `POST /v1/completions` (streaming or not), with the model's own chat template, stop strings,
//! `max_tokens`, `temperature`, `top_p`, `top_k` and `seed`. A conversation that continues the
//! previous one reuses its state instead of reading the whole history again. What it does not do:
//! tool calls, structured output, several choices, embeddings, context packs (`/v1/packs/complete`
//! answers 501, which Companion reads as "use the ordinary chat"), the Anthropic and Ollama
//! styles. One request runs at a time; the others wait.
use crate::args::Args;
use crate::famrun::{Fam, open as open_fam, template_of};
use crate::http::{self, Limits};
use crate::template::{ChatTemplate, Message};
use companion_engine::family::{self, AnyState, LanguageModel};
use companion_engine::model::{GenerateOptions, Sampler, StopReason};
use serde_json::{Value, json};
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

struct Cached {
    tokens: Vec<usize>,
    state: AnyState,
}

pub struct Served {
    fam: Fam,
    template: ChatTemplate,
    bos: String,
    eos: String,
    name: String,
    arch: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    ctx: usize,
    cache: Mutex<Option<Cached>>,
    busy: Mutex<()>,
    stop: AtomicBool,
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

fn err_json(kind: &str, msg: &str) -> Value {
    json!({"error": {"message": msg, "type": kind, "code": kind}})
}

/// What a request asks for.
#[derive(Debug, Clone)]
pub struct Params {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub stream: bool,
    pub usage: bool,
    pub stops: Vec<String>,
}

fn text_of(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str).or_else(|| p.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

pub fn messages_of(body: &Value) -> Result<Vec<Message>, String> {
    let a = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("`messages` is missing")?;
    let mut out = vec![];
    for m in a {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or("a message has no `role`")?;
        out.push(Message::new(
            role,
            &m.get("content").map(text_of).unwrap_or_default(),
        ));
    }
    if out.is_empty() {
        return Err("`messages` is empty".into());
    }
    Ok(out)
}

impl Served {
    pub fn params(&self, b: &Value) -> Params {
        let f = |k: &str| b.get(k).and_then(Value::as_f64);
        let u = |k: &str| b.get(k).and_then(Value::as_u64);
        let stops = match b.get("stop") {
            Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|s| s.as_str().filter(|s| !s.is_empty()).map(str::to_string))
                .take(8)
                .collect(),
            _ => vec![],
        };
        Params {
            max_tokens: u("max_tokens")
                .or_else(|| u("max_completion_tokens"))
                .map_or(self.max_tokens, |n| n as usize)
                .max(1),
            temperature: f("temperature").map_or(self.temperature, |v| v.max(0.0) as f32),
            top_p: f("top_p").map_or(self.top_p, |v| v.clamp(0.0, 1.0) as f32),
            top_k: u("top_k").map_or(self.top_k, |v| v as usize),
            seed: u("seed").unwrap_or_else(|| now() as u64),
            stream: b.get("stream").and_then(Value::as_bool).unwrap_or(false),
            usage: b
                .pointer("/stream_options/include_usage")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            stops,
        }
    }

    fn prompt_ids(&self, chat: bool, b: &Value) -> Result<Vec<usize>, String> {
        let ids = if chat {
            let msgs = messages_of(b)?;
            let text = self
                .template
                .render(&msgs, true, &self.bos, &self.eos)
                .map_err(|e| e.to_string())?;
            self.fam.tok.encode(&text, false)
        } else {
            let p = match b.get("prompt") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(a)) => a
                    .first()
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                _ => return Err("`prompt` is missing".into()),
            };
            self.fam.tok.encode(&p, true)
        };
        if ids.is_empty() {
            return Err("the prompt is empty".into());
        }
        Ok(ids.into_iter().map(|i| i as usize).collect())
    }
}

/// Withholds the last few characters so a stop string split over two pieces is still caught.
pub struct StopFilter {
    stops: Vec<String>,
    held: String,
    hold: usize,
    pub hit: bool,
}
impl StopFilter {
    pub fn new(stops: &[String]) -> StopFilter {
        StopFilter {
            stops: stops.to_vec(),
            held: String::new(),
            hold: stops.iter().map(|s| s.chars().count()).max().unwrap_or(0),
            hit: false,
        }
    }
    /// The text that can be shown now.
    pub fn push(&mut self, piece: &str) -> String {
        if self.hit {
            return String::new();
        }
        self.held.push_str(piece);
        let cut = self.stops.iter().filter_map(|s| self.held.find(s)).min();
        if let Some(i) = cut {
            self.hit = true;
            let out = self.held[..i].to_string();
            self.held.clear();
            return out;
        }
        let n = self.held.chars().count();
        if n <= self.hold.saturating_sub(1) {
            return String::new();
        }
        let keep = self.hold.saturating_sub(1);
        let split = self
            .held
            .char_indices()
            .nth(n - keep)
            .map_or(self.held.len(), |(i, _)| i);
        let out = self.held[..split].to_string();
        self.held.drain(..split);
        out
    }
    /// What is still held when generation ends.
    pub fn finish(&mut self) -> String {
        if self.hit {
            return String::new();
        }
        std::mem::take(&mut self.held)
    }
}

fn sse(v: &Value) -> Vec<u8> {
    format!("data: {v}\n\n").into_bytes()
}

struct Done {
    text: String,
    finish: &'static str,
    prompt_tokens: usize,
    completion_tokens: usize,
}

impl Served {
    /// Run one generation. `piece` gets the text as it can be shown; it returns `false` to stop
    /// (the client left).
    fn generate(
        &self,
        ids: &[usize],
        p: &Params,
        piece: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Done, (u16, String)> {
        let ctx = self.ctx.min(self.fam.model.context());
        if ids.len() >= ctx {
            return Err((
                400,
                format!(
                    "the prompt is {} tokens but the context window is {ctx}",
                    ids.len()
                ),
            ));
        }
        let _turn = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        // continue the previous conversation when this prompt starts with what it already read
        let reused = {
            let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            match c.take() {
                Some(k) if ids.len() > k.tokens.len() && ids[..k.tokens.len()] == k.tokens[..] => {
                    Some(k)
                }
                _ => None,
            }
        };
        let (mut st, fed) = match reused {
            Some(k) => {
                let n = k.tokens.len();
                (k.state, n)
            }
            None => (self.fam.model.new_state(), 0),
        };
        let new = &ids[fed..];
        let mut sampler = Sampler::new(p.temperature, p.top_k, p.seed).with_top_p(p.top_p);
        let max = p.max_tokens.min(ctx - ids.len());
        let cancel = AtomicBool::new(false);
        let opts = GenerateOptions {
            max_tokens: max,
            stop_tokens: self.fam.stop.clone(),
            cancel: Some(&cancel),
        };
        let mut dec = self.fam.tok.decoder();
        dec.skip_special = true;
        let mut filter = StopFilter::new(&p.stops);
        let mut text = String::new();
        let mut produced: Vec<usize> = vec![];
        let mut alive = true;
        let mut emit = |s: &str, text: &mut String, alive: &mut bool, filter: &mut StopFilter| {
            let shown = filter.push(s);
            if !shown.is_empty() {
                text.push_str(&shown);
                if !piece(&shown) {
                    *alive = false;
                }
            }
        };
        let stats = family::generate(
            &self.fam.model,
            &mut st,
            new,
            &mut sampler,
            &opts,
            &mut |t| {
                produced.push(t);
                let s = dec.push(t as u32);
                if !s.is_empty() {
                    emit(&s, &mut text, &mut alive, &mut filter);
                }
                alive && !filter.hit && !self.stop.load(Ordering::Relaxed)
            },
        )
        .map_err(|e| (500, e.to_string()))?;
        let tail = dec.finish();
        if !tail.is_empty() {
            emit(&tail, &mut text, &mut alive, &mut filter);
        }
        let rest = filter.finish();
        if !rest.is_empty() {
            text.push_str(&rest);
            if !piece(&rest) {
                alive = false;
            }
        }
        let finish = if filter.hit {
            "stop"
        } else {
            match stats.stop {
                StopReason::StopToken => "stop",
                StopReason::ContextFull => "length",
                StopReason::Cancelled if !alive => "cancelled",
                _ => "length",
            }
        };
        // keep the state for the next turn: what it has read is the prompt plus what it generated
        if alive {
            let have = self.fam.model.position(&st);
            let mut all = ids.to_vec();
            all.extend(&produced);
            all.truncate(have);
            if all.len() == have {
                *self.cache.lock().unwrap_or_else(|e| e.into_inner()) = Some(Cached {
                    tokens: all,
                    state: st,
                });
            }
        }
        Ok(Done {
            text,
            finish,
            prompt_tokens: ids.len(),
            completion_tokens: produced.len(),
        })
    }
}

fn usage(d: &Done) -> Value {
    json!({"prompt_tokens": d.prompt_tokens, "completion_tokens": d.completion_tokens,
           "total_tokens": d.prompt_tokens + d.completion_tokens})
}

fn complete(s: &Served, req: &http::Request, stream: &TcpStream, chat: bool) {
    let mut w = stream;
    let body: Value = match serde_json::from_slice(&req.body) {
        Ok(b) => b,
        Err(e) => {
            let _ = http::write_json(
                &mut w,
                400,
                &err_json(
                    "invalid_request_error",
                    &format!("the body is not JSON: {e}"),
                ),
                &[],
            );
            return;
        }
    };
    let ids = match s.prompt_ids(chat, &body) {
        Ok(i) => i,
        Err(e) => {
            let _ = http::write_json(&mut w, 400, &err_json("invalid_request_error", &e), &[]);
            return;
        }
    };
    let p = s.params(&body);
    let id = format!("chatcmpl-{:x}", now() as u64 ^ p.seed);
    let (object, chunk_object) = if chat {
        ("chat.completion", "chat.completion.chunk")
    } else {
        ("text_completion", "text_completion")
    };
    let delta = |text: &str, role: bool, finish: Option<&str>| {
        let choice = if chat {
            let mut d = json!({});
            if role {
                d["role"] = json!("assistant");
            }
            if !text.is_empty() {
                d["content"] = json!(text);
            }
            json!({"index": 0, "delta": d, "finish_reason": finish})
        } else {
            json!({"index": 0, "text": text, "finish_reason": finish})
        };
        json!({"id": id, "object": chunk_object, "created": now(), "model": s.name, "choices": [choice]})
    };
    if p.stream {
        let mut started = false;
        let mut w2 = stream;
        let r = s.generate(&ids, &p, &mut |t| {
            if !started {
                started = true;
                if http::start_chunked(&mut w2, 200, "text/event-stream", &[]).is_err() {
                    return false;
                }
                if chat && http::write_chunk(&mut w2, &sse(&delta("", true, None))).is_err() {
                    return false;
                }
            }
            http::write_chunk(&mut w2, &sse(&delta(t, false, None))).is_ok()
        });
        match r {
            Ok(d) => {
                if !started {
                    let _ = http::start_chunked(&mut w2, 200, "text/event-stream", &[]);
                    if chat {
                        let _ = http::write_chunk(&mut w2, &sse(&delta("", true, None)));
                    }
                }
                let mut last = delta("", false, Some(d.finish));
                if p.usage {
                    last["usage"] = usage(&d);
                }
                let _ = http::write_chunk(&mut w2, &sse(&last));
                let _ = http::write_chunk(&mut w2, b"data: [DONE]\n\n");
                let _ = http::end_chunked(&mut w2);
            }
            Err((code, m)) => {
                if started {
                    let _ = http::write_chunk(&mut w2, &sse(&err_json("server_error", &m)));
                    let _ = http::end_chunked(&mut w2);
                } else {
                    let _ = http::write_json(&mut w2, code, &err_json("server_error", &m), &[]);
                }
            }
        }
    } else {
        match s.generate(&ids, &p, &mut |_| true) {
            Ok(d) => {
                let choice = if chat {
                    json!({"index": 0, "message": {"role": "assistant", "content": d.text}, "finish_reason": d.finish})
                } else {
                    json!({"index": 0, "text": d.text, "finish_reason": d.finish})
                };
                let _ = http::write_json(
                    &mut w,
                    200,
                    &json!({"id": id, "object": object, "created": now(), "model": s.name,
                            "choices": [choice], "usage": usage(&d)}),
                    &[],
                );
            }
            Err((code, m)) => {
                let kind = if code == 400 {
                    "invalid_request_error"
                } else {
                    "server_error"
                };
                let _ = http::write_json(&mut w, code, &err_json(kind, &m), &[]);
            }
        }
    }
}

fn route(s: &Served, req: &http::Request, stream: &TcpStream) {
    let mut w = stream;
    let path = req.path.trim_end_matches('/');
    match (req.method.as_str(), path) {
        ("GET", "/health" | "/v1/health") => {
            let _ = http::write_json(
                &mut w,
                200,
                &json!({"status": "ok", "engine": "corbul", "model": s.name, "arch": s.arch,
                        "context_tokens": s.ctx.min(s.fam.model.context()),
                        "max_tokens": s.max_tokens, "kv_slots": 1}),
                &[],
            );
        }
        ("GET", "/v1/models") => {
            let _ = http::write_json(
                &mut w,
                200,
                &json!({"object": "list", "data": [{"id": s.name, "object": "model",
                        "created": 0, "owned_by": "corbul"}]}),
                &[],
            );
        }
        ("POST", "/v1/chat/completions") => complete(s, req, stream, true),
        ("POST", "/v1/completions") => complete(s, req, stream, false),
        ("POST", "/v1/packs/complete") => {
            let _ = http::write_json(
                &mut w,
                501,
                &err_json(
                    "not_implemented",
                    "context packs are not available for this kind of model; use the ordinary chat",
                ),
                &[],
            );
        }
        ("POST", _) | ("GET", _) => {
            let _ = http::write_json(
                &mut w,
                404,
                &err_json(
                    "not_found",
                    &format!(
                        "{path} is not served for this kind of model (it has /health, /v1/models, /v1/chat/completions, /v1/completions)"
                    ),
                ),
                &[],
            );
        }
        _ => {
            let _ = http::write_json(
                &mut w,
                405,
                &err_json("method_not_allowed", "method not allowed"),
                &[],
            );
        }
    }
}

fn handle(s: &Served, stream: TcpStream) {
    let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_nodelay(true);
    let mut r = &stream;
    let req = http::read_request(&mut r, &Limits::default(), &mut || {
        let _ = (&stream).write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
    });
    match req {
        Ok(req) => route(s, &req, &stream),
        Err(e) if e.answerable() => {
            let _ = http::write_json(
                &mut &stream,
                e.status(),
                &err_json("invalid_request_error", &e.message()),
                &[],
            );
        }
        Err(_) => {}
    }
    let _ = (&stream).flush();
    let _ = stream.shutdown(std::net::Shutdown::Write);
    // read what the client may still send, so closing does not reset the connection
    let _ = stream.set_read_timeout(Some(Duration::from_millis(100)));
    let mut sink = [0u8; 2048];
    for _ in 0..4 {
        match std::io::Read::read(&mut &stream, &mut sink) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
    }
}

/// Load `path` and build the server state (no socket yet).
pub fn build(path: &Path, args: &Args) -> Result<Served, String> {
    let fam = open_fam(path)?;
    let (template, bos, eos, warn, arch) = template_of(path, &fam.tok)?;
    for w in warn {
        eprintln!("warning: {w}");
    }
    let p = crate::runner::sampler_params(args, 0.7)?;
    let ctx = args
        .num::<usize>("ctx")?
        .filter(|c| *c > 0)
        .unwrap_or(usize::MAX);
    Ok(Served {
        name: args
            .get("model-name")
            .map_or_else(|| fam.name.clone(), str::to_string),
        fam,
        template,
        bos,
        eos,
        arch,
        max_tokens: args.num("max-tokens")?.unwrap_or(1024),
        temperature: p.temperature,
        top_p: p.top_p,
        top_k: p.top_k,
        ctx,
        cache: Mutex::new(None),
        busy: Mutex::new(()),
        stop: AtomicBool::new(false),
    })
}

/// Start accepting connections; returns the address and a stop function.
pub fn spawn(
    s: Arc<Served>,
    host: &str,
    port: u16,
) -> std::io::Result<(std::net::SocketAddr, impl FnOnce())> {
    let l = TcpListener::bind((host, port))?;
    let addr = l.local_addr()?;
    let stop = Arc::new(AtomicBool::new(false));
    let (stop2, s2) = (stop.clone(), s.clone());
    let t = std::thread::Builder::new()
        .name("corbul-fam-accept".into())
        .spawn(move || {
            for c in l.incoming() {
                if stop2.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(c) = c else {
                    std::thread::sleep(Duration::from_millis(20));
                    continue;
                };
                let s3 = s2.clone();
                let _ = std::thread::Builder::new()
                    .name("corbul-fam-conn".into())
                    .stack_size(16 << 20)
                    .spawn(move || handle(&s3, c));
            }
        })?;
    let wake = addr;
    Ok((addr, move || {
        stop.store(true, Ordering::SeqCst);
        s.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(wake);
        let _ = t.join();
    }))
}

/// `corbul serve MODEL` for a model this server handles.
pub fn serve(args: &Args) -> Result<(), String> {
    let path = args
        .pos
        .first()
        .ok_or("serve: give a model folder or .gguf file")?;
    for k in ["models-dir", "encoder", "keys-file"] {
        if args.get(k).is_some() {
            return Err(format!(
                "--{k} is not available for this kind of model (Gemma 4, state-space and hybrid models are served one at a time)"
            ));
        }
    }
    let t = std::time::Instant::now();
    let s = Arc::new(build(Path::new(path), args)?);
    eprintln!(
        "loaded {} in {:.1}s ({}, context {})",
        s.name,
        t.elapsed().as_secs_f64(),
        s.arch,
        s.ctx.min(s.fam.model.context())
    );
    let host = args.get("host").unwrap_or("127.0.0.1").to_string();
    let port: u16 = args.num("port")?.unwrap_or(8080);
    let (addr, stop) = spawn(s.clone(), &host, port).map_err(|e| format!("cannot listen: {e}"))?;
    crate::shutdown::install();
    eprintln!("serving {} on http://{addr}", s.name);
    eprintln!("  OpenAI: /v1/chat/completions /v1/completions /v1/models · /health");
    while !crate::shutdown::requested() {
        std::thread::sleep(Duration::from_millis(100));
    }
    eprintln!("shutting down …");
    stop();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::{tmpdir, write_model};
    use companion_engine::gemma4::demo;
    use std::io::Read;

    fn model_dir(tag: &str) -> std::path::PathBuf {
        let d = tmpdir(tag);
        write_model(&d, true); // byte-level tokenizer.json (vocabulary 258) beside the weights
        let o = demo::Opt {
            vocab: crate::load::testkit::VOCAB,
            ..demo::Opt::full()
        };
        demo::write(&d.join("m.gguf"), &o, true);
        d
    }
    fn start(d: &std::path::Path, extra: &[&str]) -> (std::net::SocketAddr, Box<dyn FnOnce()>) {
        let mut v = vec![d.join("m.gguf").to_string_lossy().to_string()];
        v.extend(extra.iter().map(|s| s.to_string()));
        let s = Arc::new(build(&d.join("m.gguf"), &Args::parse(&v)).unwrap());
        let (a, stop) = spawn(s, "127.0.0.1", 0).unwrap();
        (a, Box::new(stop))
    }
    fn ask(a: std::net::SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
        let mut c = TcpStream::connect(a).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
        write!(
            c,
            "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut raw = vec![];
        let _ = c.read_to_end(&mut raw);
        let t = String::from_utf8_lossy(&raw).to_string();
        let code = t[9..12].parse().unwrap();
        let (head, rest) = t.split_once("\r\n\r\n").unwrap_or((&t, ""));
        let body = if head.to_lowercase().contains("transfer-encoding: chunked") {
            let mut out = String::new();
            let mut r = rest;
            while let Some((len, tail)) = r.split_once("\r\n") {
                let n = usize::from_str_radix(len.trim(), 16).unwrap_or(0);
                if n == 0 || tail.len() < n {
                    break;
                }
                out.push_str(&tail[..n]);
                r = tail[n..].trim_start_matches("\r\n");
            }
            out
        } else {
            rest.to_string()
        };
        (code, body)
    }

    #[test]
    fn health_models_chat_stream_and_errors() {
        let d = model_dir("famserve-http");
        let (a, stop) = start(&d, &[]);
        let (c, h) = ask(a, "GET", "/health", "");
        let h: Value = serde_json::from_str(&h).unwrap();
        assert_eq!(
            (c, h["status"].as_str(), h["arch"].as_str()),
            (200, Some("ok"), Some("gemma4"))
        );
        let (_, m) = ask(a, "GET", "/v1/models", "");
        let m: Value = serde_json::from_str(&m).unwrap();
        assert_eq!(m["data"][0]["id"], "m");
        let req = |stream: bool| {
            json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                   "max_tokens": 6, "temperature": 0, "stream": stream,
                   "stream_options": {"include_usage": true}})
            .to_string()
        };
        let (c, r) = ask(a, "POST", "/v1/chat/completions", &req(false));
        assert_eq!(c, 200, "{r}");
        let r: Value = serde_json::from_str(&r).unwrap();
        assert_eq!(r["object"], "chat.completion");
        let text = r["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(r["usage"]["completion_tokens"].as_u64().unwrap() >= 1);
        // the streamed answer is the same text, ends with [DONE] and carries usage
        let (c, s) = ask(a, "POST", "/v1/chat/completions", &req(true));
        assert_eq!(c, 200);
        assert!(s.trim_end().ends_with("data: [DONE]"), "{s}");
        let mut joined = String::new();
        let mut usage = None;
        for l in s.lines().filter_map(|l| l.strip_prefix("data: ")) {
            if let Ok(v) = serde_json::from_str::<Value>(l) {
                if let Some(t) = v["choices"][0]["delta"]["content"].as_str() {
                    joined.push_str(t);
                }
                if v.get("usage").is_some() {
                    usage = Some(v["usage"].clone());
                }
            }
        }
        assert_eq!(joined, text);
        assert_eq!(
            usage.unwrap()["completion_tokens"],
            r["usage"]["completion_tokens"]
        );
        // errors are JSON with a reason
        let (c, e) = ask(a, "POST", "/v1/chat/completions", "{nope");
        assert_eq!(c, 400, "{e}");
        let (c, e) = ask(a, "POST", "/v1/chat/completions", r#"{"messages":[]}"#);
        assert_eq!(c, 400);
        assert!(e.contains("empty"));
        let (c, _) = ask(a, "POST", "/v1/packs/complete", "{}");
        assert_eq!(c, 501);
        let (c, _) = ask(a, "GET", "/nothing", "");
        assert_eq!(c, 404);
        stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_continued_conversation_reuses_its_state_and_gives_the_same_answer() {
        let d = model_dir("famserve-reuse");
        let ask_c = |a, p: &str| {
            let b = json!({"prompt": p, "max_tokens": 5, "temperature": 0}).to_string();
            let (c, r) = ask(a, "POST", "/v1/completions", &b);
            assert_eq!(c, 200, "{r}");
            serde_json::from_str::<Value>(&r).unwrap()["choices"][0]["text"]
                .as_str()
                .unwrap()
                .to_string()
        };
        let (a, stop) = start(&d, &[]);
        let first = ask_c(a, "abc");
        let follow = format!("abc{first}xy");
        let warm = ask_c(a, &follow); // the server still holds the state of the first request
        let cold_prompt = ask_c(a, "zzz"); // unrelated: forgets the state
        let _ = cold_prompt;
        let cold = ask_c(a, &follow); // same prompt, no state to reuse
        assert_eq!(warm, cold);
        stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn stop_strings_cut_the_text_even_when_split_over_pieces() {
        let mut f = StopFilter::new(&["END".to_string()]);
        let mut out = String::new();
        for p in ["ab", "cE", "N", "Dxyz"] {
            out.push_str(&f.push(p));
        }
        out.push_str(&f.finish());
        assert_eq!(out, "abc");
        assert!(f.hit);
        let mut g = StopFilter::new(&[]);
        assert_eq!(g.push("héllo"), "héllo");
        let mut h = StopFilter::new(&["zz".into()]);
        let mut o = h.push("aé");
        o.push_str(&h.push("b"));
        o.push_str(&h.finish());
        assert_eq!(o, "aéb");
    }

    #[test]
    fn a_prompt_longer_than_the_window_and_a_bad_message_are_refused() {
        let d = model_dir("famserve-ctx");
        let (a, stop) = start(&d, &["--ctx", "8"]);
        let long = "x".repeat(40);
        let (c, e) = ask(
            a,
            "POST",
            "/v1/completions",
            &json!({"prompt": long}).to_string(),
        );
        assert_eq!(c, 400);
        assert!(e.contains("context window"), "{e}");
        let (c, e) = ask(
            a,
            "POST",
            "/v1/chat/completions",
            &json!({"messages": [{"content": "no role"}]}).to_string(),
        );
        assert_eq!(c, 400);
        assert!(e.contains("role"), "{e}");
        stop();
        std::fs::remove_dir_all(d).unwrap();
    }
}
