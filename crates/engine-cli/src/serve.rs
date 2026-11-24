//! Local HTTP server: OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/models`), Anthropic
//! Messages (`/v1/messages`), a subset of Ollama's API (`/api/chat`, `/api/generate`, `/api/tags`, `/api/show`,
//! `/api/create`, …) and llama-server's utility endpoints (`/tokenize`, `/detokenize`, `/apply-template`,
//! `/props`, `/slots`), plus Prometheus `/metrics`.
//!
//! Several models can sit behind one server ([`Registry`]); each model runs one generation at a time (requests
//! queue), different models run in parallel. Each connection has its own thread. Binds to loopback by default,
//! checks the `Host` header on loopback binds (DNS-rebinding defence), optionally requires a bearer token or
//! API keys (per-key model lists and rate limits), and stops generating as soon as the client goes away.
use crate::anthropic;
use crate::auth::{KeyEntry, KeyStore, RateLimiter, SystemClock, ct_eq};
use crate::constraint::{self, Spec};
pub use crate::genloop::StopFilter;
use crate::genloop::{ChoiceOut, Constraint, GenParams, Sink, legacy_logprobs, run_choice};
use crate::http::{self, Limits};
use crate::load::Loaded;
use crate::metrics::{Metrics, esc};
use crate::quotes;
use crate::registry::{RegCfg, RegError, Registry, Slot};
use crate::template::{Message, civil};
use crate::tools::{self, ToolChoice};
use companion_engine::model::ModelError;
use serde_json::{Value, json};
use std::cell::{Cell, RefCell};
use std::io::{self, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, TryLockError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Where request-log lines go (`None` = nowhere).
#[derive(Clone)]
pub struct LogSink(pub Arc<dyn Fn(&str) + Send + Sync>);
impl std::fmt::Debug for LogSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LogSink")
    }
}

#[derive(Clone, Debug)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    /// Required as `Authorization: Bearer …` when set (`/health` stays open). Bypasses key limits.
    pub token: Option<String>,
    /// API keys with per-key model lists and rate limits (`Authorization: Bearer` or `x-api-key`).
    pub keys: Option<KeyStore>,
    /// Origins allowed by CORS (`"*"` = any). `None` = no CORS headers.
    pub cors: Option<Vec<String>>,
    pub max_connections: usize,
    /// Defaults for requests that do not say.
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub model_id: Option<String>,
    pub limits: Limits,
    /// How long a request may wait in the queue for the model.
    pub queue_wait: Duration,
    /// One line per request (method, path, status, time, model, key name, token counts — never the prompt).
    pub log: Option<LogSink>,
    /// Also log the prompt text (off by default: prompts are private).
    pub log_prompts: bool,
    /// How long shutdown waits for running requests.
    pub drain: Duration,
    /// Let API-key holders (not only the master token / open mode) use the administrative endpoints
    /// (`/api/create`, `/api/delete`, `/models/load`, `/models/unload`).
    pub allow_admin: bool,
    /// Largest KV cache one request may need (prompt + reply, times `n`), in bytes. Larger requests get a 400.
    pub max_kv_bytes: u64,
    /// Requests of the CPU-heavy kinds (tokenize, template, quote verification, …) running at once.
    pub max_heavy: usize,
}
impl Default for ServeConfig {
    fn default() -> ServeConfig {
        ServeConfig {
            host: "127.0.0.1".into(),
            port: 8080,
            token: None,
            keys: None,
            cors: None,
            max_connections: 64,
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            model_id: None,
            limits: Limits::default(),
            queue_wait: Duration::from_secs(600),
            log: None,
            log_prompts: false,
            drain: Duration::from_secs(10),
            allow_admin: false,
            max_kv_bytes: 8 << 30,
            max_heavy: 4,
        }
    }
}

pub struct App {
    pub registry: Registry,
    pub cfg: ServeConfig,
    pub metrics: Metrics,
    limiter: RateLimiter<SystemClock>,
    counter: AtomicU64,
    active: AtomicUsize,
    loopback: AtomicBool,
    draining: AtomicBool,
    heavy: AtomicUsize,
}
impl App {
    /// A server for one model that is already loaded.
    pub fn new(loaded: Loaded, cfg: ServeConfig) -> App {
        let name = cfg.model_id.clone().unwrap_or_else(|| loaded.name.clone());
        let reg = Registry::single(loaded, &name, RegCfg::default());
        App::with_registry(reg, cfg)
    }
    pub fn with_registry(registry: Registry, cfg: ServeConfig) -> App {
        App {
            registry,
            cfg,
            metrics: Metrics::new(),
            limiter: RateLimiter::default(),
            counter: AtomicU64::new(1),
            active: AtomicUsize::new(0),
            loopback: AtomicBool::new(true),
            draining: AtomicBool::new(false),
            heavy: AtomicUsize::new(0),
        }
    }
    fn next_id(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::Relaxed)
    }
}

pub struct Handle {
    pub addr: SocketAddr,
    stop: Arc<AtomicBool>,
    app: Arc<App>,
    thread: Option<JoinHandle<()>>,
    sweeper: Option<JoinHandle<()>>,
}
impl Handle {
    /// Block until the server stops (it only stops through [`Handle::stop`]).
    pub fn join(mut self) {
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
    /// Stop accepting, let running requests finish (up to `cfg.drain`), then return.
    pub fn stop(mut self) {
        self.app.draining.store(true, Ordering::SeqCst);
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(500));
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let t0 = Instant::now();
        while self.app.active.load(Ordering::SeqCst) > 0 && t0.elapsed() < self.app.cfg.drain {
            std::thread::sleep(Duration::from_millis(10));
        }
        if let Some(t) = self.sweeper.take() {
            let _ = t.join();
        }
    }
}

pub fn spawn(app: Arc<App>) -> io::Result<Handle> {
    let listener = TcpListener::bind((app.cfg.host.as_str(), app.cfg.port))?;
    let addr = listener.local_addr()?;
    app.loopback
        .store(addr.ip().is_loopback(), Ordering::SeqCst);
    let stop = Arc::new(AtomicBool::new(false));
    let (stop2, app1) = (stop.clone(), app.clone());
    let thread = std::thread::Builder::new()
        .name("corbul-accept".into())
        .spawn(move || {
            for conn in listener.incoming() {
                if stop2.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(stream) = conn else {
                    std::thread::sleep(Duration::from_millis(20));
                    continue;
                };
                if app1.draining.load(Ordering::SeqCst) {
                    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
                    let _ = http::write_json(
                        &mut &stream,
                        503,
                        &err_body(
                            Style::Oai,
                            "overloaded_error",
                            "the server is shutting down",
                        ),
                        &[],
                    );
                    continue;
                }
                if app1.active.fetch_add(1, Ordering::SeqCst) >= app1.cfg.max_connections {
                    app1.active.fetch_sub(1, Ordering::SeqCst);
                    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
                    let _ = http::write_json(
                        &mut &stream,
                        503,
                        &err_body(Style::Oai, "overloaded_error", "too many connections"),
                        &[("Retry-After", "5")],
                    );
                    continue;
                }
                let app2 = app1.clone();
                let spawned = std::thread::Builder::new()
                    .name("corbul-conn".into())
                    .stack_size(16 << 20)
                    .spawn(move || {
                        struct Guard(Arc<App>);
                        impl Drop for Guard {
                            fn drop(&mut self) {
                                self.0.active.fetch_sub(1, Ordering::SeqCst);
                            }
                        }
                        let _g = Guard(app2.clone());
                        handle(stream, &app2);
                    });
                if spawned.is_err() {
                    app1.active.fetch_sub(1, Ordering::SeqCst);
                }
            }
        })?;
    // unload models that sat idle too long
    let (stop3, app3) = (stop.clone(), app.clone());
    let sweeper = std::thread::Builder::new()
        .name("corbul-sweep".into())
        .spawn(move || {
            let mut tick = 0u32;
            while !stop3.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(100));
                tick += 1;
                if tick.is_multiple_of(10) {
                    app3.registry.sweep_idle();
                }
            }
        })?;
    Ok(Handle {
        addr,
        stop,
        app,
        thread: Some(thread),
        sweeper: Some(sweeper),
    })
}

// ------------------------------------------------------------------------------ errors

#[derive(Debug)]
struct ApiError {
    status: u16,
    kind: &'static str,
    message: String,
    retry_after: Option<u64>,
}
fn api_err(status: u16, kind: &'static str, msg: impl Into<String>) -> ApiError {
    ApiError {
        status,
        kind,
        message: msg.into(),
        retry_after: None,
    }
}
fn bad(msg: impl Into<String>) -> ApiError {
    api_err(400, "invalid_request_error", msg)
}
fn server_err(msg: impl Into<String>) -> ApiError {
    api_err(500, "server_error", msg)
}

/// Which error shape a client expects.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Style {
    Oai,
    Ollama,
    Anthropic,
}

fn style_of(path: &str) -> Style {
    if path.starts_with("/api/") {
        Style::Ollama
    } else if path.starts_with("/v1/messages") {
        Style::Anthropic
    } else {
        Style::Oai
    }
}

fn err_body(style: Style, kind: &str, msg: &str) -> Value {
    match style {
        Style::Ollama => json!({"error": msg}),
        Style::Anthropic => anthropic::error_body(kind, msg),
        Style::Oai => json!({"error": {"message": msg, "type": kind, "param": null, "code": null}}),
    }
}

fn model_error(e: ModelError) -> ApiError {
    match e {
        ModelError::Token(t) => bad(format!(
            "the tokenizer produced token {t}, which is outside the model's vocabulary"
        )),
        ModelError::ContextFull(n) => bad(format!(
            "the prompt does not fit the model's context window of {n} tokens"
        )),
        ModelError::NoTokens => bad("the prompt is empty"),
        other => server_err(other.to_string()),
    }
}

fn reg_error(e: RegError) -> ApiError {
    match e {
        RegError::NotFound(n) => api_err(
            404,
            "not_found_error",
            format!("the model `{n}` does not exist"),
        ),
        RegError::Load(m) => {
            eprintln!("corbul: could not load a model: {m}");
            server_err(format!("could not load the model: {}", redact_paths(&m)))
        }
        RegError::Busy => api_err(
            503,
            "overloaded_error",
            "every model slot is busy; try again later",
        ),
    }
}

// ------------------------------------------------------------------------------ small helpers

/// Replace anything that looks like a file system path in `m` with `<path>`: error text goes to remote
/// clients and must not reveal the server's directory layout (the full text goes to the server's stderr).
pub fn redact_paths(m: &str) -> String {
    m.split(' ')
        .map(|w| {
            let core = w.trim_matches(|c: char| {
                matches!(
                    c,
                    '`' | '\'' | '"' | ',' | ';' | ':' | '(' | ')' | '[' | ']'
                )
            });
            let drive = core.len() > 2
                && core.as_bytes()[1] == b':'
                && core.as_bytes()[0].is_ascii_alphabetic();
            if core.contains('/') || core.contains('\\') || drive {
                w.replace(core, "<path>")
            } else {
                w.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Percent-decode a URL query component (`+` is a space). Invalid escapes are kept as they are.
fn pct_decode(s: &str) -> String {
    let b = s.as_bytes();
    let (mut out, mut i) = (Vec::with_capacity(b.len()), 0);
    while i < b.len() {
        match b[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < b.len()
                && b[i + 1].is_ascii_hexdigit()
                && b[i + 2].is_ascii_hexdigit() =>
            {
                out.push(u8::from_str_radix(&s[i + 1..i + 3], 16).unwrap_or(b'%'));
                i += 3;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

fn iso_now() -> String {
    let (y, mo, d, h, mi, s) = civil(now_secs());
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Has the peer closed its end? (Looks without consuming.)
fn client_gone(s: &TcpStream) -> bool {
    if s.set_nonblocking(true).is_err() {
        return true;
    }
    let mut b = [0u8; 1];
    let r = s.peek(&mut b);
    let _ = s.set_nonblocking(false);
    match r {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => e.kind() != io::ErrorKind::WouldBlock && e.kind() != io::ErrorKind::Interrupted,
    }
}

// ------------------------------------------------------------------------------ responses

#[derive(Default)]
struct ReqInfo {
    model: String,
    key: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    prompt: Option<String>,
}

/// Everything a handler writes goes through here: CORS and request-id headers, the status for the log.
struct Out<'a> {
    w: &'a TcpStream,
    common: Vec<(&'static str, String)>,
    status: Cell<u16>,
    info: RefCell<ReqInfo>,
}
impl Out<'_> {
    fn hdrs<'b>(&'b self, extra: &[(&'b str, &'b str)]) -> Vec<(&'b str, &'b str)> {
        self.common
            .iter()
            .map(|(k, v)| (*k, v.as_str()))
            .chain(extra.iter().copied())
            .collect()
    }
    fn json_h(&self, code: u16, v: &Value, extra: &[(&str, &str)]) {
        self.status.set(code);
        let mut w = self.w;
        let _ = http::write_json(&mut w, code, v, &self.hdrs(extra));
    }
    fn json(&self, code: u16, v: &Value) {
        self.json_h(code, v, &[])
    }
    fn bytes(&self, code: u16, ct: &str, body: &[u8], extra: &[(&str, &str)]) {
        self.status.set(code);
        let mut w = self.w;
        let _ = http::write_response(&mut w, code, ct, body, &self.hdrs(extra));
    }
    fn error(&self, style: Style, e: &ApiError) {
        let ra = e.retry_after.map(|s| s.to_string());
        let extra: Vec<(&str, &str)> = ra
            .as_deref()
            .map(|s| ("Retry-After", s))
            .into_iter()
            .collect();
        self.json_h(e.status, &err_body(style, e.kind, &e.message), &extra);
    }
    fn start_chunked(&self, ct: &str) -> io::Result<()> {
        self.status.set(200);
        let mut w = self.w;
        http::start_chunked(
            &mut w,
            200,
            ct,
            &self.hdrs(&[("Cache-Control", "no-cache")]),
        )
    }
    fn chunk(&self, data: &[u8]) -> io::Result<()> {
        let mut w = self.w;
        http::write_chunk(&mut w, data)
    }
    fn end(&self) {
        let mut w = self.w;
        let _ = http::end_chunked(&mut w);
    }
}

// ------------------------------------------------------------------------------ request parsing

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    OaiChat,
    OaiCompletion,
    OllamaChat,
    OllamaGenerate,
    Anthropic,
}
impl Kind {
    fn ollama(self) -> bool {
        matches!(self, Kind::OllamaChat | Kind::OllamaGenerate)
    }
    fn chat(self) -> bool {
        matches!(self, Kind::OaiChat | Kind::OllamaChat | Kind::Anthropic)
    }
    fn style(self) -> Style {
        if self.ollama() {
            Style::Ollama
        } else if self == Kind::Anthropic {
            Style::Anthropic
        } else {
            Style::Oai
        }
    }
}

#[derive(Debug, Clone)]
struct Params {
    sp: GenParams,
    stream: bool,
    include_usage: bool,
    n: usize,
}

fn num(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64).filter(|x| x.is_finite())
}

fn params(
    app: &App,
    slot: &Slot,
    body: &Value,
    kind: Kind,
    extra_stops: Vec<String>,
) -> Result<Params, ApiError> {
    let opts = if kind.ollama() {
        body.get("options")
            .filter(|o| o.is_object())
            .unwrap_or(&Value::Null)
    } else {
        body
    };
    let get = |k: &str| {
        num(opts.get(k))
            .or_else(|| num(body.get(k)))
            .or_else(|| num(slot.options.get(k)))
    };
    let vocab = slot.loaded.model.spec.vocab as usize;
    let ctx = slot.loaded.model.spec.context as usize;
    let mut max = app.cfg.max_tokens;
    let asked = match kind {
        Kind::OllamaChat | Kind::OllamaGenerate => get("num_predict"),
        Kind::Anthropic => num(body.get("max_tokens")),
        _ => num(body.get("max_tokens"))
            .or_else(|| num(body.get("max_completion_tokens")))
            .or_else(|| num(slot.options.get("num_predict"))),
    };
    if kind == Kind::Anthropic && asked.is_none() {
        return Err(bad("`max_tokens` is required"));
    }
    if let Some(n) = asked {
        if n >= 0.0 {
            max = n as usize;
        } else {
            max = ctx;
        }
    }
    let stop_of = |v: Option<&Value>| -> Vec<String> {
        match v {
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect(),
            _ => vec![],
        }
    };
    let mut stops = stop_of(body.get("stop"));
    if stops.is_empty() {
        stops = stop_of(opts.get("stop"));
    }
    if stops.is_empty() {
        stops = stop_of(slot.options.get("stop"));
    }
    stops.extend(extra_stops);
    stops.retain(|s| !s.is_empty() && s.len() <= 64);
    stops.dedup();
    stops.truncate(8);
    let mut bias = vec![];
    if let Some(Value::Object(m)) = body.get("logit_bias") {
        for (k, v) in m.iter().take(300) {
            let (Ok(id), Some(b)) = (k.parse::<usize>(), v.as_f64()) else {
                return Err(bad("`logit_bias` maps token ids to numbers"));
            };
            if id < vocab {
                bias.push((id, b.clamp(-100.0, 100.0) as f32));
            }
        }
    }
    let logprobs = match kind {
        Kind::OaiChat if body.get("logprobs").and_then(Value::as_bool) == Some(true) => {
            Some(num(body.get("top_logprobs")).map_or(0, |n| n.clamp(0.0, 20.0) as usize))
        }
        Kind::OaiCompletion => num(body.get("logprobs")).map(|n| n.clamp(0.0, 20.0) as usize),
        _ => None,
    };
    let n = match kind {
        Kind::OaiChat | Kind::OaiCompletion => num(body.get("n")).map_or(1, |n| n as usize),
        _ => 1,
    };
    if !(1..=8).contains(&n) {
        return Err(bad("`n` must be between 1 and 8"));
    }
    let stream_default = kind.ollama();
    let t = get("temperature").map_or(app.cfg.temperature, |t| t.clamp(0.0, 5.0) as f32);
    Ok(Params {
        sp: GenParams {
            max_tokens: max.min(ctx),
            temperature: t,
            top_p: get("top_p").map_or(app.cfg.top_p, |t| {
                if t > 0.0 {
                    t.min(1.0) as f32
                } else {
                    app.cfg.top_p
                }
            }),
            top_k: get("top_k").map_or(app.cfg.top_k, |k| (k.max(0.0) as usize).min(1000)),
            seed: get("seed").map_or_else(
                || (now_secs() as u64) ^ app.next_id().wrapping_mul(0x9e37_79b9_7f4a_7c15),
                |s| s as u64,
            ),
            rep: get("repetition_penalty")
                .or_else(|| get("repeat_penalty"))
                .map_or(1.0, |r| r.clamp(0.5, 2.0) as f32),
            presence: get("presence_penalty").map_or(0.0, |p| p.clamp(-2.0, 2.0) as f32),
            frequency: get("frequency_penalty").map_or(0.0, |p| p.clamp(-2.0, 2.0) as f32),
            stops,
            logit_bias: bias,
            logprobs,
        },
        stream: body
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(stream_default),
        include_usage: body
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        n,
    })
}

fn content_text(v: Option<&Value>) -> Result<String, ApiError> {
    match v {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => Ok(parts
            .iter()
            .filter_map(|p| match p {
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
        Some(_) => Err(bad("`content` must be a string or a list of text parts")),
    }
}

/// An assistant message's `tool_calls` in either OpenAI (`arguments` is a string) or Ollama (an object) shape.
fn parse_tool_calls(v: Option<&Value>) -> Result<Vec<Value>, ApiError> {
    let Some(Value::Array(a)) = v else {
        return Ok(vec![]);
    };
    a.iter()
        .enumerate()
        .map(|(i, c)| {
            let f = c
                .get("function")
                .ok_or_else(|| bad("every tool call needs a `function`"))?;
            let name = f
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| !n.is_empty())
                .ok_or_else(|| bad("a tool call's function needs a `name`"))?;
            let args = match f.get("arguments") {
                Some(Value::String(s)) => s.clone(),
                Some(v @ Value::Object(_)) => v.to_string(),
                None | Some(Value::Null) => "{}".into(),
                Some(_) => return Err(bad("`arguments` must be a JSON string or object")),
            };
            let id = c
                .get("id")
                .and_then(Value::as_str)
                .map_or_else(|| format!("call_{i}"), str::to_string);
            Ok(json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}}))
        })
        .collect()
}

fn parse_messages(v: Option<&Value>) -> Result<Vec<Message>, ApiError> {
    let Some(Value::Array(a)) = v else {
        return Err(bad("`messages` must be a list"));
    };
    if a.is_empty() {
        return Err(bad("`messages` is empty"));
    }
    if a.len() > 512 {
        return Err(bad("too many messages"));
    }
    a.iter()
        .map(|m| {
            let role = m
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("every message needs a string `role`"))?;
            if role.is_empty()
                || role.len() > 32
                || !role
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            {
                return Err(bad("unsupported `role`"));
            }
            let role = match role {
                "developer" => "system",
                "function" => "tool",
                r => r,
            };
            let mut msg = Message {
                role: role.to_string(),
                content: content_text(m.get("content"))?,
                ..Default::default()
            };
            if role == "assistant" {
                msg.tool_calls = parse_tool_calls(m.get("tool_calls"))?;
            }
            if role == "tool" {
                msg.tool_call_id = m
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                msg.name = m
                    .get("name")
                    .or_else(|| m.get("tool_name"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
            }
            Ok(msg)
        })
        .collect()
}

/// Everything one request asks for, after parsing.
struct Req {
    kind: Kind,
    msgs: Vec<Message>,
    prompt: String,
    raw: bool,
    tools: Vec<Value>,
    choice: ToolChoice,
    parallel: bool,
    p: Params,
    spec: Option<Spec>,
    /// Quote-locked mode: the sources, numbered from 1, and their index.
    quote_idx: Option<Arc<companion_engine::quotelock::QuoteIndex>>,
    /// Steer the reply with the lock (`quote_lock`, default on); off still checks the quotes afterwards.
    quote_lock: bool,
}
impl Req {
    fn tools_active(&self) -> bool {
        !self.tools.is_empty() && self.choice != ToolChoice::None
    }
}

fn parse_request(app: &App, slot: &Slot, body: &Value, kind: Kind) -> Result<Req, ApiError> {
    let mut extra_stops = vec![];
    let (mut msgs, mut tools, mut choice, mut prompt, mut raw) =
        (vec![], vec![], ToolChoice::Auto, String::new(), false);
    if kind == Kind::Anthropic {
        let a = anthropic::parse(body).map_err(bad)?;
        (msgs, tools, choice, extra_stops) = (a.messages, a.tools, a.choice, a.stops);
    } else if kind.chat() {
        msgs = parse_messages(body.get("messages"))?;
        tools = tools::normalize_tools(body.get("tools")).map_err(bad)?;
        choice = tools::parse_choice(body.get("tool_choice")).map_err(bad)?;
    } else {
        prompt = match body.get("prompt") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(a)) if a.len() == 1 && a[0].is_string() => {
                a[0].as_str().unwrap_or("").to_string()
            }
            _ => return Err(bad("`prompt` must be a string")),
        };
        raw = body.get("raw").and_then(Value::as_bool).unwrap_or(false);
    }
    if let ToolChoice::Named(n) = &choice
        && !tools::tool_names(&tools).contains(n)
    {
        return Err(bad(format!(
            "`tool_choice` names `{n}`, which is not in `tools`"
        )));
    }
    if matches!(choice, ToolChoice::Required | ToolChoice::Named(_)) && tools.is_empty() {
        return Err(bad("`tool_choice` needs `tools`"));
    }
    let spec = Spec::from_body(body).map_err(bad)?;
    let sources = quotes::parse_sources(body.get("sources")).map_err(bad)?;
    let mut quote_idx = None;
    if !sources.is_empty() {
        if spec.is_some() {
            return Err(bad(
                "`sources` (quote lock) cannot be combined with structured output (`response_format`, `grammar`, …) in one request",
            ));
        }
        let idx = quotes::index(&sources).map_err(bad)?;
        let note = quotes::instructions(&sources);
        if kind.chat() {
            match msgs.first_mut() {
                Some(m) if m.role == "system" => {
                    m.content.push_str("\n\n");
                    m.content.push_str(&note);
                }
                _ => msgs.insert(0, Message::new("system", &note)),
            }
        } else if !raw {
            prompt = format!("{note}\n{prompt}");
        }
        quote_idx = Some(idx);
    }
    let p = params(app, slot, body, kind, extra_stops)?;
    if p.sp.logprobs.is_some() && kind == Kind::OaiChat && p.n > 1 && p.stream { /* allowed: entries carry the choice index */
    }
    Ok(Req {
        kind,
        msgs,
        prompt,
        raw,
        tools,
        choice,
        parallel: body
            .get("parallel_tool_calls")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        p,
        spec,
        quote_idx,
        quote_lock: body
            .get("quote_lock")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    })
}

fn to_ids(v: Vec<u32>) -> Vec<usize> {
    v.into_iter().map(|i| i as usize).collect()
}

/// The messages the template sees: the Modelfile's system prompt and examples, the request's messages, and a
/// reminder when a call is forced.
fn chat_messages(slot: &Slot, rq: &Req) -> Vec<Message> {
    let mut msgs: Vec<Message> = vec![];
    let has_system = rq.msgs.first().is_some_and(|m| m.role == "system");
    if !has_system && let Some(s) = &slot.system {
        msgs.push(Message::new("system", s));
    }
    let mut it = rq.msgs.iter().cloned();
    if has_system && let Some(first) = it.next() {
        msgs.push(first);
    }
    msgs.extend(slot.preface.iter().cloned());
    msgs.extend(it);
    if rq.tools_active()
        && let Some(h) = tools::choice_hint(&rq.choice)
    {
        match msgs.first_mut() {
            Some(m) if m.role == "system" => {
                m.content.push_str("\n\n");
                m.content.push_str(&h);
            }
            _ => msgs.insert(0, Message::new("system", &h)),
        }
    }
    msgs
}

/// The prompt text for a chat request (also what `/apply-template` returns).
fn render_chat(slot: &Slot, msgs: &[Message], tools: &[Value]) -> Result<String, ApiError> {
    let l = &slot.loaded;
    let text = l
        .template
        .render_with(msgs, tools, true, &l.bos, &l.eos)
        .map_err(|e| bad(e.to_string()))?;
    // a template that knows nothing about tools: describe them in the system prompt instead (Hermes layout)
    let first = tools::tool_names(tools);
    if let Some(n) = first.first()
        && !text.contains(n.as_str())
    {
        let mut m2: Vec<Message> = msgs.to_vec();
        let block = crate::template::hermes_block(tools);
        match m2.first_mut() {
            Some(m) if m.role == "system" => {
                m.content.push_str("\n\n");
                m.content.push_str(&block);
            }
            _ => m2.insert(0, Message::new("system", &block)),
        }
        return l
            .template
            .render_with(&m2, &[], true, &l.bos, &l.eos)
            .map_err(|e| bad(e.to_string()));
    }
    Ok(text)
}

fn build_prompt(slot: &Slot, rq: &Req) -> Result<(Vec<usize>, String), ApiError> {
    let l = &slot.loaded;
    let (ids, text) = if rq.kind.chat() {
        let msgs = chat_messages(slot, rq);
        let text = render_chat(slot, &msgs, if rq.tools_active() { &rq.tools } else { &[] })?;
        (l.tok.encode(&text, false), text)
    } else {
        (l.tok.encode(&rq.prompt, !rq.raw), rq.prompt.clone())
    };
    if ids.is_empty() {
        return Err(bad("the prompt is empty"));
    }
    Ok((to_ids(ids), text))
}

// ------------------------------------------------------------------------------ generation

struct Outcome {
    choices: Vec<ChoiceOut>,
    prompt_tokens: usize,
    completion_tokens: usize,
    secs: f64,
}
impl Outcome {
    fn cancelled(prompt_tokens: usize) -> Outcome {
        Outcome {
            choices: vec![ChoiceOut {
                finish: "cancelled",
                ..Default::default()
            }],
            prompt_tokens,
            completion_tokens: 0,
            secs: 0.0,
        }
    }
    fn finish(&self) -> &'static str {
        self.choices.first().map_or("stop", |c| c.finish)
    }
}

/// The constraint a request asks for: structured output, or the quote lock over its sources.
fn make_constraint(slot: &Slot, rq: &Req) -> Result<Option<Box<dyn Constraint>>, String> {
    if let Some(s) = &rq.spec {
        return constraint::build(slot, s).map(Some);
    }
    if let Some(idx) = rq.quote_idx.as_ref().filter(|_| rq.quote_lock) {
        let v = slot
            .quote_vocab
            .get_or_init(|| quotes::vocab(&slot.loaded.tok))
            .clone();
        return Ok(Some(quotes::constraint(idx.clone(), v)));
    }
    Ok(None)
}

/// Upper bound of the KV bytes `tokens` positions take (every layer treated as full attention).
fn kv_bytes_for(l: &Loaded, tokens: u64) -> u64 {
    use companion_engine::model::KvKind;
    let sp = &l.model.spec;
    let per_elem = match l.kv {
        KvKind::F32 => 4.0,
        KvKind::F16 => 2.0,
        KvKind::Q8 => 1.125,
    };
    let per_token = 2.0 * sp.layers as f64 * sp.kv_heads as f64 * sp.head_dim as f64 * per_elem;
    (per_token * tokens as f64) as u64
}

#[allow(clippy::too_many_arguments)]
fn generate(
    app: &App,
    slot: &Arc<Slot>,
    prompt: &[usize],
    rq: &Req,
    alive: &dyn Fn() -> bool,
    sink: &mut dyn Sink,
    t_req: Instant,
    id_seed: u64,
) -> Result<Outcome, ApiError> {
    let m = &slot.loaded.model;
    let ctx = m.spec.context as usize;
    if prompt.len() >= ctx {
        return Err(bad(format!(
            "the prompt is {} tokens but the context window is {ctx}",
            prompt.len()
        )));
    }
    // the KV cache grows with prompt + reply, once per choice: refuse what would not fit the configured budget
    let want = prompt
        .len()
        .saturating_add(rq.p.sp.max_tokens.min(ctx - prompt.len())) as u64;
    let need = kv_bytes_for(&slot.loaded, want).saturating_mul(rq.p.n as u64);
    if need > app.cfg.max_kv_bytes {
        return Err(bad(format!(
            "this request needs about {} MiB of KV cache (limit {} MiB): shorten the prompt, lower max_tokens or n",
            need >> 20,
            app.cfg.max_kv_bytes >> 20
        )));
    }
    // compile the grammar before queueing: a bad schema is the client's error
    let mut first_constraint: Option<Box<dyn Constraint>> =
        make_constraint(slot, rq).map_err(bad)?;
    // wait for the model, noticing if the client leaves meanwhile
    let waited = Instant::now();
    app.metrics.queued.fetch_add(1, Ordering::Relaxed);
    let gate = loop {
        match slot.gate.try_lock() {
            Ok(g) => break Some(g),
            Err(TryLockError::Poisoned(e)) => break Some(e.into_inner()),
            Err(TryLockError::WouldBlock) => {
                if !alive() {
                    break None;
                }
                if waited.elapsed() > app.cfg.queue_wait {
                    app.metrics.queued.fetch_sub(1, Ordering::Relaxed);
                    return Err(api_err(
                        503,
                        "overloaded_error",
                        "the model is busy; try again later",
                    ));
                }
                std::thread::sleep(Duration::from_millis(15));
            }
        }
    };
    app.metrics.queued.fetch_sub(1, Ordering::Relaxed);
    app.metrics
        .queue_wait
        .observe(waited.elapsed().as_secs_f64());
    let Some(mut gate) = gate else {
        return Ok(Outcome::cancelled(prompt.len()));
    };
    slot.touch();
    let t0 = Instant::now();
    if !sink.start() {
        return Ok(Outcome::cancelled(prompt.len()));
    }
    if rq.p.sp.max_tokens == 0 {
        return Ok(Outcome {
            choices: vec![ChoiceOut {
                finish: "length",
                ..Default::default()
            }],
            prompt_tokens: prompt.len(),
            completion_tokens: 0,
            secs: 0.0,
        });
    }
    let mut st = gate
        .take()
        .unwrap_or_else(|| m.new_state_with(slot.loaded.kv));
    let logits = m.prefill_cached(&mut st, prompt).map_err(model_error)?;
    let base = if rq.p.n > 1 {
        Some(st.fork().map_err(model_error)?)
    } else {
        None
    };
    let mut sp = rq.p.sp.clone();
    sp.max_tokens = sp.max_tokens.min(ctx - prompt.len());
    let names = if rq.tools_active() {
        Some(tools::tool_names(&rq.tools))
    } else {
        None
    };
    let mut choices = vec![];
    for ci in 0..rq.p.n {
        let cons = if ci == 0 {
            first_constraint.take()
        } else {
            make_constraint(slot, rq).map_err(server_err)?
        };
        let res = if ci == 0 {
            run_choice(
                slot,
                &mut st,
                logits.clone(),
                prompt,
                &sp,
                names.clone(),
                cons,
                alive,
                sink,
                ci,
                id_seed,
                t_req,
            )
        } else {
            let mut other = base
                .as_ref()
                .ok_or_else(|| server_err("no forked state"))?
                .fork()
                .map_err(model_error)?;
            run_choice(
                slot,
                &mut other,
                logits.clone(),
                prompt,
                &sp,
                names.clone(),
                cons,
                alive,
                sink,
                ci,
                id_seed,
                t_req,
            )
        };
        let mut c = res.map_err(model_error)?;
        if !rq.parallel && c.calls.len() > 1 {
            c.calls.truncate(1);
        }
        let stop = c.finish == "cancelled";
        choices.push(c);
        if stop {
            break;
        }
    }
    *gate = Some(st);
    let completion_tokens = choices.iter().map(|c| c.tokens).sum();
    Ok(Outcome {
        choices,
        prompt_tokens: prompt.len(),
        completion_tokens,
        secs: t0.elapsed().as_secs_f64(),
    })
}

// ------------------------------------------------------------------------------ response shapes

struct Ids {
    id: String,
    created: i64,
    model: String,
}

fn usage(o: &Outcome) -> Value {
    json!({"prompt_tokens": o.prompt_tokens, "completion_tokens": o.completion_tokens, "total_tokens": o.prompt_tokens + o.completion_tokens})
}

fn oai_tool_calls(c: &ChoiceOut) -> Value {
    Value::Array(c.calls.iter().map(|(id, t)| json!({"id": id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}})).collect())
}

fn oai_finish(c: &ChoiceOut) -> &'static str {
    if c.finish == "cancelled" {
        "stop"
    } else {
        c.finish
    }
}

fn full_json(rq: &Req, ids: &Ids, o: &Outcome) -> Value {
    let first = o.choices.first().cloned().unwrap_or_default();
    match rq.kind {
        Kind::OaiChat => {
            let choices: Vec<Value> = o.choices.iter().enumerate().map(|(i, c)| {
                let mut msg = json!({"role": "assistant", "content": if c.text.is_empty() && !c.calls.is_empty() { Value::Null } else { json!(c.text) }});
                if !c.calls.is_empty() { msg["tool_calls"] = oai_tool_calls(c); }
                json!({"index": i, "message": msg, "logprobs": if rq.p.sp.logprobs.is_some() { json!({"content": c.logprobs}) } else { Value::Null }, "finish_reason": oai_finish(c)})
            }).collect();
            json!({"id": ids.id, "object": "chat.completion", "created": ids.created, "model": ids.model, "choices": choices, "usage": usage(o)})
        }
        Kind::OaiCompletion => {
            let choices: Vec<Value> = o.choices.iter().enumerate().map(|(i, c)| json!({"index": i, "text": c.text, "logprobs": if rq.p.sp.logprobs.is_some() { legacy_logprobs(&c.logprobs, c.text.chars().count()) } else { Value::Null }, "finish_reason": oai_finish(c)})).collect();
            json!({"id": ids.id, "object": "text_completion", "created": ids.created, "model": ids.model, "choices": choices, "usage": usage(o)})
        }
        Kind::OllamaChat => {
            let mut v = ollama_done(ids, o);
            v["message"] = ollama_message(&first, &first.text);
            v
        }
        Kind::OllamaGenerate => {
            let mut v = ollama_done(ids, o);
            v["response"] = json!(first.text);
            v
        }
        Kind::Anthropic => {
            let calls: Vec<(String, String, String)> = first
                .calls
                .iter()
                .map(|(id, t)| (id.clone(), t.name.clone(), t.arguments.clone()))
                .collect();
            anthropic::message_json(
                &ids.id,
                &ids.model,
                &first.text,
                &calls,
                anthropic::stop_reason(first.finish, &first.stop_sequence),
                &first.stop_sequence,
                o.prompt_tokens,
                o.completion_tokens,
            )
        }
    }
}

fn ollama_message(c: &ChoiceOut, text: &str) -> Value {
    let mut m = json!({"role": "assistant", "content": text});
    if !c.calls.is_empty() {
        m["tool_calls"] = Value::Array(c.calls.iter().map(|(_, t)| json!({"function": {"name": t.name, "arguments": serde_json::from_str::<Value>(&t.arguments).unwrap_or_else(|_| json!({}))}})).collect());
    }
    m
}

fn ollama_done(ids: &Ids, o: &Outcome) -> Value {
    let ns = (o.secs * 1e9) as u64;
    json!({"model": ids.model, "created_at": iso_now(), "done": true, "done_reason": if o.finish() == "length" { "length" } else { "stop" }, "total_duration": ns, "prompt_eval_count": o.prompt_tokens, "eval_count": o.completion_tokens})
}

fn sse(v: &Value) -> Vec<u8> {
    format!("data: {v}\n\n").into_bytes()
}

/// Streams each kind's wire format as the pieces arrive.
struct Streamer<'a> {
    out: &'a Out<'a>,
    rq: &'a Req,
    ids: &'a Ids,
    prompt_tokens: usize,
    started: bool,
    anth: anthropic::StreamState,
}
impl Streamer<'_> {
    fn write(&self, b: &[u8]) -> bool {
        self.out.chunk(b).is_ok()
    }
    fn events(&self, evs: Vec<anthropic::Ev>) -> bool {
        evs.iter().all(|e| self.write(&anthropic::sse_bytes(e)))
    }
    fn chunk_json(
        &self,
        choice: usize,
        delta: Value,
        finish: Value,
        logprobs: Option<Value>,
    ) -> Vec<u8> {
        let mut c = json!({"index": choice, "delta": delta, "finish_reason": finish});
        if let Some(l) = logprobs {
            c["logprobs"] = l;
        }
        sse(
            &json!({"id": self.ids.id, "object": "chat.completion.chunk", "created": self.ids.created, "model": self.ids.model, "choices": [c]}),
        )
    }
    fn finish(&mut self, o: &Outcome) {
        if !self.started {
            return;
        }
        match self.rq.kind {
            Kind::OaiChat | Kind::OaiCompletion => {
                for (i, c) in o.choices.iter().enumerate() {
                    let b = if self.rq.kind == Kind::OaiChat {
                        self.chunk_json(i, json!({}), json!(oai_finish(c)), None)
                    } else {
                        sse(
                            &json!({"id": self.ids.id, "object": "text_completion", "created": self.ids.created, "model": self.ids.model, "choices": [{"index": i, "text": "", "logprobs": null, "finish_reason": oai_finish(c)}]}),
                        )
                    };
                    self.write(&b);
                }
                if self.rq.p.include_usage {
                    self.write(&sse(&json!({"id": self.ids.id, "object": "chat.completion.chunk", "created": self.ids.created, "model": self.ids.model, "choices": [], "usage": usage(o)})));
                }
                self.write(b"data: [DONE]\n\n");
            }
            Kind::OllamaChat => {
                let mut v = ollama_done(self.ids, o);
                v["message"] = json!({"role": "assistant", "content": ""});
                self.write(format!("{v}\n").as_bytes());
            }
            Kind::OllamaGenerate => {
                let mut v = ollama_done(self.ids, o);
                v["response"] = json!("");
                self.write(format!("{v}\n").as_bytes());
            }
            Kind::Anthropic => {
                let c = o.choices.first().cloned().unwrap_or_default();
                let evs = self.anth.finish(
                    anthropic::stop_reason(c.finish, &c.stop_sequence),
                    &c.stop_sequence,
                    o.completion_tokens,
                );
                self.events(evs);
            }
        }
        self.out.end();
    }
}
impl Sink for Streamer<'_> {
    fn start(&mut self) -> bool {
        let ct = if self.rq.kind.ollama() {
            "application/x-ndjson"
        } else {
            "text/event-stream"
        };
        if self.out.start_chunked(ct).is_err() {
            return false;
        }
        self.started = true;
        match self.rq.kind {
            Kind::OaiChat => (0..self.rq.p.n).all(|i| {
                self.write(&self.chunk_json(
                    i,
                    json!({"role": "assistant", "content": ""}),
                    Value::Null,
                    None,
                ))
            }),
            Kind::Anthropic => {
                let evs = self
                    .anth
                    .start(&self.ids.id, &self.ids.model, self.prompt_tokens);
                self.events(evs)
            }
            _ => true,
        }
    }
    fn text(&mut self, choice: usize, piece: &str, lp: &[Value]) -> bool {
        match self.rq.kind {
            Kind::OaiChat => {
                let l = (!lp.is_empty()).then(|| json!({"content": lp}));
                self.write(&self.chunk_json(choice, json!({"content": piece}), Value::Null, l))
            }
            Kind::OaiCompletion => {
                let l = if lp.is_empty() { Value::Null } else { legacy_logprobs(lp, piece.chars().count()) };
                self.write(&sse(&json!({"id": self.ids.id, "object": "text_completion", "created": self.ids.created, "model": self.ids.model, "choices": [{"index": choice, "text": piece, "logprobs": l, "finish_reason": null}]})))
            }
            Kind::OllamaChat => self.write(format!("{}\n", json!({"model": self.ids.model, "created_at": iso_now(), "message": {"role": "assistant", "content": piece}, "done": false})).as_bytes()),
            Kind::OllamaGenerate => self.write(format!("{}\n", json!({"model": self.ids.model, "created_at": iso_now(), "response": piece, "done": false})).as_bytes()),
            Kind::Anthropic => { let e = self.anth.text(piece); self.events(e) }
        }
    }
    fn call(&mut self, choice: usize, index: usize, id: &str, name: &str, arguments: &str) -> bool {
        match self.rq.kind {
            Kind::OaiChat => self.write(&self.chunk_json(choice, json!({"tool_calls": [{"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": arguments}}]}), Value::Null, None)),
            Kind::OllamaChat => {
                let args = serde_json::from_str::<Value>(arguments).unwrap_or_else(|_| json!({}));
                self.write(format!("{}\n", json!({"model": self.ids.model, "created_at": iso_now(), "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}, "done": false})).as_bytes())
            }
            Kind::Anthropic => { let e = self.anth.call(id, name, arguments); self.events(e) }
            _ => true,
        }
    }
}

/// Collects everything for a non-streaming reply.
struct Collect;
impl Sink for Collect {
    fn start(&mut self) -> bool {
        true
    }
    fn text(&mut self, _: usize, _: &str, _: &[Value]) -> bool {
        true
    }
    fn call(&mut self, _: usize, _: usize, _: &str, _: &str, _: &str) -> bool {
        true
    }
}

// ------------------------------------------------------------------------------ connection handling

type Principal = Option<KeyEntry>;
type Denied = (u16, Value, Vec<(&'static str, &'static str)>);

fn handle(stream: TcpStream, app: &App) {
    let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_nodelay(true);
    let t0 = Instant::now();
    let mut reader = &stream;
    let req = http::read_request(&mut reader, &app.cfg.limits, &mut || {
        let _ = (&stream).write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
    });
    let req = match req {
        Ok(r) => r,
        Err(e) => {
            if e.answerable() {
                let _ = http::write_json(
                    &mut &stream,
                    e.status(),
                    &err_body(Style::Oai, "invalid_request_error", &e.message()),
                    &[],
                );
            }
            return;
        }
    };
    let rid = format!("req_{:x}{:x}", now_secs(), app.next_id());
    let mut common = vec![("X-Request-Id", rid)];
    if let Some(allowed) = &app.cfg.cors
        && let Some(origin) = req.header("origin")
    {
        if allowed.iter().any(|o| o == "*") {
            common.push(("Access-Control-Allow-Origin", "*".into()));
        } else if allowed.iter().any(|o| o == origin) {
            common.push(("Access-Control-Allow-Origin", origin.to_string()));
            common.push(("Vary", "Origin".into()));
        }
    }
    let out = Out {
        w: &stream,
        common,
        status: Cell::new(0),
        info: RefCell::new(ReqInfo::default()),
    };
    app.metrics.in_flight.fetch_add(1, Ordering::Relaxed);
    let path = req
        .path
        .split('?')
        .next()
        .unwrap_or("")
        .trim_end_matches('/')
        .to_string();
    let path = if path.is_empty() {
        "/".to_string()
    } else {
        path
    };
    if req.method == "OPTIONS" {
        preflight(app, &req, &out);
    } else {
        match guard(app, &req, &path) {
            Err((status, body, extra)) => out.json_h(status, &body, &extra),
            Ok(who) => {
                if let Some(k) = &who {
                    out.info.borrow_mut().key = k.name.clone();
                }
                route(app, &req, &path, &out, &who, &stream);
            }
        }
    }
    app.metrics.in_flight.fetch_sub(1, Ordering::Relaxed);
    let status = out.status.get();
    app.metrics.request(&label(&path), status);
    app.metrics.latency.observe(t0.elapsed().as_secs_f64());
    let info = out.info.borrow();
    app.metrics
        .tokens(&info.model, info.prompt_tokens, info.completion_tokens);
    if let Some(log) = &app.cfg.log {
        let mut line = format!(
            "{} {} {} -> {} {}ms",
            iso_now(),
            req.method,
            path,
            status,
            t0.elapsed().as_millis()
        );
        if !info.model.is_empty() {
            line.push_str(&format!(" model={}", info.model));
        }
        if !info.key.is_empty() {
            line.push_str(&format!(" key={}", info.key));
        }
        if info.prompt_tokens + info.completion_tokens > 0 {
            line.push_str(&format!(
                " tokens={}+{}",
                info.prompt_tokens, info.completion_tokens
            ));
        }
        if app.cfg.log_prompts
            && let Some(p) = &info.prompt
        {
            line.push_str(&format!(
                " prompt={:?}",
                p.chars().take(2000).collect::<String>()
            ));
        }
        (log.0)(&line);
    }
}

/// A bounded label for the request counter.
fn label(path: &str) -> String {
    const KNOWN: [&str; 34] = [
        "/",
        "/ui",
        "/capabilities",
        "/v1/capabilities",
        "/health",
        "/v1/health",
        "/metrics",
        "/props",
        "/slots",
        "/tokenize",
        "/detokenize",
        "/apply-template",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/embeddings",
        "/v1/rerank",
        "/rerank",
        "/api/chat",
        "/api/generate",
        "/api/tags",
        "/api/ps",
        "/api/show",
        "/api/create",
        "/api/delete",
        "/api/version",
        "/api/embed",
        "/api/embeddings",
        "/models/load",
        "/models/unload",
        "/models",
        "/v1/quotes/verify",
    ];
    if KNOWN.contains(&path) {
        path.to_string()
    } else if path.starts_with("/v1/models/") {
        "/v1/models/{id}".into()
    } else {
        "other".into()
    }
}

fn preflight(app: &App, req: &http::Request, out: &Out) {
    let mut extra: Vec<(&str, String)> = vec![("Allow", "GET, POST, DELETE, OPTIONS".into())];
    if app.cfg.cors.is_some() {
        extra.push((
            "Access-Control-Allow-Methods",
            "GET, POST, DELETE, OPTIONS".into(),
        ));
        extra.push((
            "Access-Control-Allow-Headers",
            req.header("access-control-request-headers")
                .unwrap_or("authorization, content-type, x-api-key, anthropic-version")
                .to_string(),
        ));
        extra.push(("Access-Control-Max-Age", "600".into()));
    }
    let e: Vec<(&str, &str)> = extra.iter().map(|(k, v)| (*k, v.as_str())).collect();
    out.bytes(204, "text/plain", b"", &e);
}

/// Host check (loopback binds) and credentials: the single `--token`, or an API key.
fn authenticate(app: &App, req: &http::Request, path: &str) -> Result<Principal, Denied> {
    let style = style_of(path);
    if app.loopback.load(Ordering::SeqCst)
        && let Some(h) = req.header("host")
    {
        let host = if let Some(rest) = h.strip_prefix('[') {
            rest.split(']').next().unwrap_or("")
        } else {
            h.rsplit_once(':')
                .filter(|(_, p)| p.bytes().all(|b| b.is_ascii_digit()))
                .map_or(h, |(a, _)| a)
        };
        if !matches!(
            host.to_ascii_lowercase().as_str(),
            "localhost" | "127.0.0.1" | "::1"
        ) {
            return Err((
                403,
                err_body(
                    style,
                    "invalid_request_error",
                    "unexpected Host header for a loopback server",
                ),
                vec![],
            ));
        }
    }
    // the page itself is public (it asks for the token); everything it calls is not
    if crate::webui::is_page(path) && matches!(req.method.as_str(), "GET" | "HEAD") {
        return Ok(None);
    }
    if path == "/health"
        || path == "/v1/health"
        || (app.cfg.token.is_none() && app.cfg.keys.is_none())
    {
        return Ok(None);
    }
    let given = req
        .header("authorization")
        .and_then(|a| {
            a.strip_prefix("Bearer ")
                .or_else(|| a.strip_prefix("bearer "))
        })
        .or_else(|| req.header("x-api-key"))
        .unwrap_or("")
        .trim();
    if let Some(tok) = &app.cfg.token
        && ct_eq(given, tok)
    {
        return Ok(None);
    }
    if let Some(keys) = &app.cfg.keys
        && let Some(k) = keys.authenticate(given)
    {
        return Ok(Some(k.clone()));
    }
    Err((
        401,
        err_body(
            style,
            "authentication_error",
            "missing or wrong bearer token",
        ),
        vec![("WWW-Authenticate", "Bearer")],
    ))
}

/// Administrative endpoints: they load/unload models and change what the server serves.
fn is_admin(m: &str, path: &str) -> bool {
    matches!(
        (m, path),
        ("POST", "/api/create" | "/models/load" | "/models/unload")
            | ("DELETE" | "POST", "/api/delete")
    )
}

/// Media type of a header value (`application/json; charset=utf-8` → `application/json`), lower-cased.
fn media_type(v: &str) -> String {
    v.split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

/// `Origin` is `scheme://host[:port]`; same-origin when its authority equals the request's `Host`.
fn same_origin(origin: &str, host: Option<&str>) -> bool {
    let Some(host) = host else { return false };
    origin
        .split_once("://")
        .is_some_and(|(_, auth)| auth.eq_ignore_ascii_case(host))
}

fn guard(app: &App, req: &http::Request, path: &str) -> Result<Principal, Denied> {
    let style = style_of(path);
    let deny =
        |status: u16, ty: &str, msg: &str| -> Denied { (status, err_body(style, ty, msg), vec![]) };
    // A browser page on another site can make the user's browser send requests to this server; the only thing
    // that distinguishes them is `Origin`. Anything from a foreign origin that was not allowed with --cors is refused.
    if let Some(origin) = req.header("origin") {
        let listed = app
            .cfg
            .cors
            .as_ref()
            .is_some_and(|l| l.iter().any(|o| o == "*" || o == origin));
        if !listed && !same_origin(origin, req.header("host")) {
            return Err(deny(
                403,
                "permission_error",
                "cross-origin request refused (allow the origin with --cors)",
            ));
        }
    }
    // A "simple" cross-site POST (text/plain, form) needs no preflight; JSON bodies do, so insist on JSON.
    if matches!(req.method.as_str(), "POST" | "PUT" | "PATCH" | "DELETE")
        && !req.body.is_empty()
        && media_type(req.header("content-type").unwrap_or("")) != "application/json"
    {
        return Err(deny(
            415,
            "invalid_request_error",
            "send the body with Content-Type: application/json",
        ));
    }
    let who = authenticate(app, req, path)?;
    if is_admin(&req.method, path) && who.is_some() && !app.cfg.allow_admin {
        return Err(deny(
            403,
            "permission_error",
            "this endpoint needs the server's master token (or --allow-admin)",
        ));
    }
    Ok(who)
}

fn allowed(who: &Principal, model: &str) -> bool {
    who.as_ref().is_none_or(|k| k.allows(model))
}

/// The model a request names (loading it when needed), after the key's model list was checked.
fn slot_for(app: &App, who: &Principal, requested: Option<&str>) -> Result<Arc<Slot>, ApiError> {
    let name = app.registry.canonical(requested).map_err(reg_error)?;
    if !allowed(who, &name) {
        return Err(api_err(
            403,
            "permission_error",
            format!("this API key may not use the model `{name}`"),
        ));
    }
    app.registry.get(Some(&name)).map_err(reg_error)
}

fn model_obj(i: &crate::registry::Info, ctx: Option<u64>) -> Value {
    let mut v = json!({"id": i.name, "object": "model", "created": 0, "owned_by": "corbul", "loaded": i.loaded});
    if let Some(c) = ctx {
        v["context_length"] = json!(c);
    }
    if !i.aliases.is_empty() {
        v["aliases"] = json!(i.aliases);
    }
    v
}

fn ctx_of(app: &App, name: &str) -> Option<u64> {
    app.registry
        .loaded_slots()
        .iter()
        .find(|s| s.name == name)
        .map(|s| s.loaded.model.spec.context)
}

fn body_json(req: &http::Request) -> Result<Value, ApiError> {
    match serde_json::from_slice(&req.body) {
        Ok(v @ Value::Object(_)) => Ok(v),
        Ok(_) => Err(bad("the request body must be a JSON object")),
        Err(e) => Err(bad(format!("invalid JSON: {e}"))),
    }
}

fn query_param(req: &http::Request, key: &str) -> Option<String> {
    req.query.split('&').find_map(|kv| {
        kv.split_once('=')
            .filter(|(k, _)| pct_decode(k) == key)
            .map(|(_, v)| pct_decode(v))
    })
}

/// Counts one running CPU-heavy request; released on drop.
struct HeavyPermit<'a>(&'a AtomicUsize);
impl<'a> HeavyPermit<'a> {
    fn acquire(c: &'a AtomicUsize, max: usize) -> Option<HeavyPermit<'a>> {
        if c.fetch_add(1, Ordering::SeqCst) >= max.max(1) {
            c.fetch_sub(1, Ordering::SeqCst);
            None
        } else {
            Some(HeavyPermit(c))
        }
    }
}
impl Drop for HeavyPermit<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn route(
    app: &App,
    req: &http::Request,
    path: &str,
    out: &Out,
    who: &Principal,
    stream: &TcpStream,
) {
    let m = req.method.as_str();
    let style = style_of(path);
    let fail = |e: ApiError| out.error(style, &e);
    // CPU-heavy endpoints: charged to the key's request limit and capped in how many run at once
    let _permit = if m == "POST"
        && matches!(
            path,
            "/tokenize"
                | "/detokenize"
                | "/apply-template"
                | "/v1/messages/count_tokens"
                | "/v1/quotes/verify"
        ) {
        if let Some(k) = who
            && let Err(wait) = app.limiter.admit(k, 0)
        {
            return fail(ApiError {
                retry_after: Some(if wait.is_finite() {
                    wait.ceil().max(1.0) as u64
                } else {
                    60
                }),
                ..api_err(429, "rate_limit_error", "rate limit reached for this key")
            });
        }
        match HeavyPermit::acquire(&app.heavy, app.cfg.max_heavy) {
            Some(p) => Some(p),
            None => {
                return fail(ApiError {
                    retry_after: Some(1),
                    ..api_err(
                        429,
                        "rate_limit_error",
                        "too many heavy requests at once; retry shortly",
                    )
                });
            }
        }
    } else {
        None
    };
    let not_impl = |what: &str| {
        out.json(501, &err_body(style, "not_implemented_error", &format!("{what} are not available: this engine build exposes no hidden-state / pooling output, so embeddings and reranking cannot be computed")))
    };
    match (m, path) {
        ("GET" | "HEAD", "/" | "/ui" | "/ui/") => out.bytes(
            200,
            "text/html; charset=utf-8",
            crate::webui::INDEX.as_bytes(),
            crate::webui::HEADERS,
        ),
        ("GET", "/capabilities" | "/v1/capabilities") => {
            let v: Vec<Value> = crate::cmds::CAPABILITIES
                .iter()
                .map(|(n, s, d)| json!({"feature": n, "supported": *s == "yes", "notes": d}))
                .collect();
            out.json(
                200,
                &json!({"object": "list", "data": v, "models_loaded": app.registry.len()}),
            );
        }
        ("GET", "/health" | "/v1/health") => {
            if app.draining.load(Ordering::SeqCst) {
                out.json(503, &json!({"status": "shutting down"}))
            } else {
                out.json(200, &json!({"status": "ok"}))
            }
        }
        ("GET", "/metrics") => out.bytes(
            200,
            "text/plain; version=0.0.4; charset=utf-8",
            metrics_text(app).as_bytes(),
            &[],
        ),
        ("GET", "/v1/models") => {
            let data: Vec<Value> = app
                .registry
                .infos()
                .iter()
                .filter(|i| allowed(who, &i.name))
                .map(|i| model_obj(i, ctx_of(app, &i.name)))
                .collect();
            out.json(200, &json!({"object": "list", "data": data}));
        }
        ("GET", p) if p.starts_with("/v1/models/") => {
            let id = &p["/v1/models/".len()..];
            match app.registry.info(id).filter(|i| allowed(who, &i.name)) {
                Some(i) => out.json(200, &model_obj(&i, ctx_of(app, &i.name))),
                None => out.json(404, &err_body(style, "not_found_error", "no such model")),
            }
        }
        ("GET", "/api/tags") => {
            let models: Vec<Value> = app.registry.infos().iter().filter(|i| allowed(who, &i.name)).map(|i| {
                let family = app.registry.loaded_slots().iter().find(|s| s.name == i.name).map_or_else(|| "unknown".to_string(), |s| s.loaded.model.spec.family.to_string());
                json!({"name": i.name, "model": i.name, "modified_at": iso_now(), "size": i.bytes, "details": {"family": family, "format": "corbul"}})
            }).collect();
            out.json(200, &json!({"models": models}));
        }
        ("GET", "/api/ps") => {
            let models: Vec<Value> = app.registry.loaded_slots().iter().filter(|s| allowed(who, &s.name)).map(|s| json!({"name": s.name, "model": s.name, "size": s.loaded.model.weight_bytes(), "size_vram": 0, "expires_at": null})).collect();
            out.json(200, &json!({"models": models}));
        }
        ("GET", "/api/version") => out.json(200, &json!({"version": env!("CARGO_PKG_VERSION")})),
        ("POST", "/api/show") => match body_json(req).and_then(|b| {
            slot_for(
                app,
                who,
                b.get("model")
                    .or_else(|| b.get("name"))
                    .and_then(Value::as_str),
            )
        }) {
            Ok(s) => {
                let spec = &s.loaded.model.spec;
                let mf = app.registry.info(&s.name).and_then(|i| i.modelfile);
                out.json(200, &json!({"modelfile": mf.map(|f| format!("FROM {}\n", std::path::Path::new(&f).file_name().map_or_else(|| s.name.clone(), |n| n.to_string_lossy().into_owned()))).unwrap_or_default(), "parameters": s.options.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join("\n"),
                    "template": "", "system": s.system, "details": {"family": spec.family, "format": "corbul", "parameter_size": ""},
                    "model_info": {"general.architecture": spec.family, "context_length": spec.context, "block_count": spec.layers, "embedding_length": spec.hidden, "vocab_size": spec.vocab},
                    "capabilities": ["completion", "tools"]}));
            }
            Err(e) => fail(e),
        },
        ("POST", "/api/create") => match body_json(req) {
            Ok(b) => {
                let name = b
                    .get("model")
                    .or_else(|| b.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let text = b
                    .get("modelfile")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| {
                        b.get("from").and_then(Value::as_str).map(|f| {
                            format!(
                                "FROM {f}\n{}{}",
                                b.get("system")
                                    .and_then(Value::as_str)
                                    .map_or(String::new(), |s| format!("SYSTEM \"\"\"{s}\"\"\"\n")),
                                b.get("template")
                                    .and_then(Value::as_str)
                                    .map_or(String::new(), |t| format!(
                                        "TEMPLATE \"\"\"{t}\"\"\"\n"
                                    ))
                            )
                        })
                    });
                match text
                    .ok_or_else(|| "`modelfile` (or `from`) is required".to_string())
                    .and_then(|t| app.registry.create_restricted(name, &t))
                {
                    Ok(()) => out.json(200, &json!({"status": "success"})),
                    Err(e) => out.json(400, &err_body(style, "invalid_request_error", &e)),
                }
            }
            Err(e) => fail(e),
        },
        ("DELETE" | "POST", "/api/delete") => match body_json(req) {
            Ok(b) => match app.registry.delete(
                b.get("model")
                    .or_else(|| b.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            ) {
                Ok(()) => out.bytes(200, "application/json", b"{}", &[]),
                Err(e) => out.json(404, &err_body(style, "not_found_error", &e)),
            },
            Err(e) => fail(e),
        },
        ("POST", "/models/load") => match body_json(req)
            .and_then(|b| slot_for(app, who, b.get("model").and_then(Value::as_str)))
        {
            Ok(s) => out.json(200, &json!({"model": s.name, "loaded": true})),
            Err(e) => fail(e),
        },
        ("POST", "/models/unload") => match body_json(req) {
            Ok(b) => {
                let name = app
                    .registry
                    .canonical(b.get("model").and_then(Value::as_str))
                    .unwrap_or_default();
                if !allowed(who, &name) {
                    return fail(api_err(
                        403,
                        "permission_error",
                        "this API key may not use that model",
                    ));
                }
                out.json(
                    200,
                    &json!({"model": name, "unloaded": app.registry.unload(&name)}),
                );
            }
            Err(e) => fail(e),
        },
        ("GET", "/props") => match slot_for(app, who, query_param(req, "model").as_deref()) {
            Ok(s) => {
                let l = &s.loaded;
                out.json(200, &json!({"model": s.name, "total_slots": 1, "bos_token": l.bos, "eos_token": l.eos, "chat_template": l.builtin.name(),
                    "n_ctx": l.model.spec.context, "n_vocab": l.model.spec.vocab, "default_generation_settings": {"n_ctx": l.model.spec.context, "params": {"temperature": app.cfg.temperature, "top_p": app.cfg.top_p, "top_k": app.cfg.top_k, "n_predict": app.cfg.max_tokens}},
                    "modalities": {"vision": false, "audio": false}, "build_info": format!("corbul {}", env!("CARGO_PKG_VERSION")), "tools": true}));
            }
            Err(e) => fail(e),
        },
        ("GET", "/slots") => {
            let slots: Vec<Value> = app.registry.loaded_slots().iter().filter(|s| allowed(who, &s.name)).enumerate().map(|(i, s)| {
                let (busy, cached) = match s.gate.try_lock() { Ok(g) => (false, g.as_ref().map_or(0, |st| st.tokens().len())), Err(_) => (true, 0) };
                json!({"id": i, "model": s.name, "n_ctx": s.loaded.model.spec.context, "is_processing": busy, "n_cached": cached, "speculative": false, "idle_seconds": s.idle_secs()})
            }).collect();
            out.json(200, &Value::Array(slots));
        }
        ("POST", "/tokenize") => match body_json(req).and_then(|b| {
            slot_for(app, who, b.get("model").and_then(Value::as_str)).map(|s| (b, s))
        }) {
            Ok((b, s)) => {
                let Some(text) = b
                    .get("content")
                    .or_else(|| b.get("prompt"))
                    .and_then(Value::as_str)
                else {
                    return fail(bad("`content` must be a string"));
                };
                let ids = s.loaded.tok.encode(
                    text,
                    b.get("add_special")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                );
                if b.get("with_pieces").and_then(Value::as_bool) == Some(true) {
                    let t: Vec<Value> = ids
                        .iter()
                        .map(|i| json!({"id": i, "piece": s.loaded.tok.decode(&[*i])}))
                        .collect();
                    out.json(200, &json!({"tokens": t}));
                } else {
                    out.json(200, &json!({"tokens": ids}));
                }
            }
            Err(e) => fail(e),
        },
        ("POST", "/detokenize") => match body_json(req).and_then(|b| {
            slot_for(app, who, b.get("model").and_then(Value::as_str)).map(|s| (b, s))
        }) {
            Ok((b, s)) => {
                let Some(Value::Array(a)) = b.get("tokens") else {
                    return fail(bad("`tokens` must be a list of ids"));
                };
                let vocab = s.loaded.model.spec.vocab;
                let ids: Option<Vec<u32>> = a
                    .iter()
                    .map(|x| x.as_u64().filter(|i| *i < vocab).map(|i| i as u32))
                    .collect();
                match ids {
                    Some(ids) => out.json(200, &json!({"content": s.loaded.tok.decode(&ids)})),
                    None => fail(bad("every token must be an id inside the vocabulary")),
                }
            }
            Err(e) => fail(e),
        },
        ("POST", "/apply-template") => match body_json(req).and_then(|b| {
            slot_for(app, who, b.get("model").and_then(Value::as_str)).map(|s| (b, s))
        }) {
            Ok((b, s)) => {
                let go = || -> Result<String, ApiError> {
                    let msgs = parse_messages(b.get("messages"))?;
                    let tools = tools::normalize_tools(b.get("tools")).map_err(bad)?;
                    let rq = Req {
                        kind: Kind::OaiChat,
                        msgs,
                        prompt: String::new(),
                        raw: false,
                        tools,
                        choice: ToolChoice::Auto,
                        parallel: true,
                        p: params(app, &s, &json!({}), Kind::OaiChat, vec![])?,
                        spec: None,
                        quote_idx: None,
                        quote_lock: false,
                    };
                    render_chat(&s, &chat_messages(&s, &rq), &rq.tools)
                };
                match go() {
                    Ok(p) => out.json(200, &json!({"prompt": p})),
                    Err(e) => fail(e),
                }
            }
            Err(e) => fail(e),
        },
        ("POST", "/v1/embeddings" | "/api/embed" | "/api/embeddings") => not_impl("embeddings"),
        ("POST", "/v1/rerank" | "/rerank") => not_impl("rerank requests"),
        ("POST", "/v1/messages/count_tokens") => match count_tokens(app, req, who) {
            Ok(n) => out.json(200, &json!({"input_tokens": n})),
            Err(e) => fail(e),
        },
        ("POST", "/v1/quotes/verify") => match body_json(req).and_then(|b| {
            let sources = quotes::parse_sources(b.get("sources")).map_err(bad)?;
            if sources.is_empty() {
                return Err(bad("`sources` must list at least one source"));
            }
            let text = b
                .get("text")
                .or_else(|| b.get("output"))
                .and_then(Value::as_str)
                .ok_or_else(|| bad("`text` (the answer to check) is required"))?;
            quotes::verify(text, &sources).map_err(bad)
        }) {
            Ok(v) => out.json(200, &v),
            Err(e) => fail(e),
        },
        ("POST", "/v1/chat/completions") => completion(app, req, out, who, stream, Kind::OaiChat),
        ("POST", "/v1/completions") => completion(app, req, out, who, stream, Kind::OaiCompletion),
        ("POST", "/v1/messages") => completion(app, req, out, who, stream, Kind::Anthropic),
        ("POST", "/api/chat") => completion(app, req, out, who, stream, Kind::OllamaChat),
        ("POST", "/api/generate") => completion(app, req, out, who, stream, Kind::OllamaGenerate),
        (
            _,
            "/"
            | "/health"
            | "/v1/health"
            | "/metrics"
            | "/props"
            | "/slots"
            | "/tokenize"
            | "/detokenize"
            | "/apply-template"
            | "/v1/models"
            | "/api/tags"
            | "/api/ps"
            | "/api/show"
            | "/api/create"
            | "/api/delete"
            | "/api/version"
            | "/models/load"
            | "/models/unload"
            | "/v1/embeddings"
            | "/v1/rerank"
            | "/rerank"
            | "/api/embed"
            | "/api/embeddings"
            | "/v1/chat/completions"
            | "/v1/completions"
            | "/v1/messages"
            | "/v1/messages/count_tokens"
            | "/api/chat"
            | "/api/generate"
            | "/v1/quotes/verify",
        ) => {
            out.json_h(
                405,
                &err_body(style, "invalid_request_error", "method not allowed"),
                &[("Allow", "GET, POST, OPTIONS")],
            );
        }
        _ => out.json(404, &err_body(style, "not_found_error", "unknown endpoint")),
    }
}

fn count_tokens(app: &App, req: &http::Request, who: &Principal) -> Result<usize, ApiError> {
    let body = body_json(req)?;
    let slot = slot_for(app, who, body.get("model").and_then(Value::as_str))?;
    let rq = parse_request(app, &slot, &body, Kind::Anthropic).or_else(|e| {
        if e.message.contains("max_tokens") {
            parse_request(
                app,
                &slot,
                &{
                    let mut b = body.clone();
                    b["max_tokens"] = json!(1);
                    b
                },
                Kind::Anthropic,
            )
        } else {
            Err(e)
        }
    })?;
    Ok(build_prompt(&slot, &rq)?.0.len())
}

fn metrics_text(app: &App) -> String {
    let slots = app.registry.loaded_slots();
    let mut extra: Vec<(&str, &str, f64, String)> = vec![
        (
            "corbul_models_configured",
            "Models the server can load.",
            app.registry.len() as f64,
            String::new(),
        ),
        (
            "corbul_models_loaded",
            "Models in memory.",
            slots.len() as f64,
            String::new(),
        ),
        (
            "corbul_connections_active",
            "Open client connections.",
            app.active.load(Ordering::Relaxed) as f64,
            String::new(),
        ),
        (
            "corbul_model_loads_total",
            "Times a model was loaded from disk.",
            app.registry.loads.load(Ordering::Relaxed) as f64,
            String::new(),
        ),
    ];
    for s in &slots {
        let lbl = format!("model=\"{}\"", esc(&s.name));
        extra.push((
            "corbul_model_weight_bytes",
            "Bytes of weights of a loaded model.",
            s.loaded.model.weight_bytes() as f64,
            lbl.clone(),
        ));
        extra.push((
            "corbul_model_idle_seconds",
            "Seconds since a loaded model was last used.",
            s.idle_secs() as f64,
            lbl.clone(),
        ));
        let (busy, cached) = match s.gate.try_lock() {
            Ok(g) => (0.0, g.as_ref().map_or(0, |st| st.tokens().len())),
            Err(_) => (1.0, 0),
        };
        extra.push((
            "corbul_model_busy",
            "1 while a model is generating.",
            busy,
            lbl.clone(),
        ));
        extra.push((
            "corbul_kv_cache_tokens",
            "Tokens held in a model's cached conversation.",
            cached as f64,
            lbl,
        ));
    }
    app.metrics.render(&extra)
}

fn completion(
    app: &App,
    req: &http::Request,
    out: &Out,
    who: &Principal,
    stream: &TcpStream,
    kind: Kind,
) {
    let style = kind.style();
    let t_req = Instant::now();
    let fail = |e: ApiError| out.error(style, &e);
    let body = match body_json(req) {
        Ok(b) => b,
        Err(e) => return fail(e),
    };
    let slot = match slot_for(app, who, body.get("model").and_then(Value::as_str)) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    out.info.borrow_mut().model = slot.name.clone();
    let rq = match parse_request(app, &slot, &body, kind) {
        Ok(r) => r,
        Err(e) => return fail(e),
    };
    let (prompt, text) = match build_prompt(&slot, &rq) {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    if app.cfg.log_prompts {
        out.info.borrow_mut().prompt = Some(text);
    }
    out.info.borrow_mut().prompt_tokens = prompt.len();
    if let Some(k) = who
        && let Err(wait) = app.limiter.admit(k, prompt.len())
    {
        return if wait.is_finite() {
            fail(ApiError {
                retry_after: Some(wait.ceil().max(1.0) as u64),
                ..api_err(
                    429,
                    "rate_limit_error",
                    format!(
                        "rate limit reached for this key; retry in {:.0}s",
                        wait.ceil()
                    ),
                )
            })
        } else {
            fail(bad(
                "this request is larger than the key's tokens-per-minute limit",
            ))
        };
    }
    let n = app.next_id();
    let model = slot.name.clone();
    let prefix = match kind {
        Kind::OaiCompletion => "cmpl",
        Kind::Anthropic => "msg",
        _ => "chatcmpl",
    };
    let ids = Ids {
        id: format!("{prefix}-{n:x}{:x}", now_secs()),
        created: now_secs(),
        model,
    };
    let id_seed = rq.p.sp.seed ^ n;
    let alive = || !client_gone(stream);
    let result = if rq.p.stream {
        let mut sink = Streamer {
            out,
            rq: &rq,
            ids: &ids,
            prompt_tokens: prompt.len(),
            started: false,
            anth: Default::default(),
        };
        let r = generate(app, &slot, &prompt, &rq, &alive, &mut sink, t_req, id_seed);
        match &r {
            Ok(o) => sink.finish(o),
            Err(e) => {
                if sink.started {
                    let msg = err_body(style, e.kind, &e.message);
                    let b = if kind == Kind::Anthropic {
                        anthropic::sse_bytes(&("error", msg))
                    } else if kind.ollama() {
                        format!("{msg}\n").into_bytes()
                    } else {
                        sse(&msg)
                    };
                    let _ = out.chunk(&b);
                    out.end();
                } else {
                    fail(ApiError {
                        status: e.status,
                        kind: e.kind,
                        message: e.message.clone(),
                        retry_after: e.retry_after,
                    });
                }
            }
        }
        r
    } else {
        let r = generate(
            app,
            &slot,
            &prompt,
            &rq,
            &alive,
            &mut Collect,
            t_req,
            id_seed,
        );
        match &r {
            Ok(o) => {
                if o.finish() != "cancelled" {
                    let mut j = full_json(&rq, &ids, o);
                    if let Some(idx) = &rq.quote_idx {
                        let texts: Vec<&str> = o.choices.iter().map(|c| c.text.as_str()).collect();
                        quotes::attach(&mut j, idx, &texts);
                    }
                    out.json(200, &j);
                }
            }
            Err(e) => fail(ApiError {
                status: e.status,
                kind: e.kind,
                message: e.message.clone(),
                retry_after: e.retry_after,
            }),
        }
        r
    };
    if let Ok(o) = result {
        out.info.borrow_mut().completion_tokens = o.completion_tokens;
        app.metrics.generated(o.completion_tokens, o.secs);
        if let Some(c) = o.choices.first().filter(|c| c.tokens > 0) {
            app.metrics.ttft.observe(c.ttft);
        }
    }
}

// ------------------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;
    use crate::load::{LoadCfg, load};
    use std::io::{Read, Write};

    fn start(
        token: Option<&str>,
        tweak: impl FnOnce(&mut ServeConfig),
    ) -> (Handle, std::path::PathBuf) {
        let d = tmpdir("serve");
        write_model(&d, true);
        let loaded = load(&d, &LoadCfg::default()).unwrap();
        let mut cfg = ServeConfig {
            port: 0,
            token: token.map(str::to_string),
            max_tokens: 8,
            temperature: 0.0,
            ..ServeConfig::default()
        };
        tweak(&mut cfg);
        (spawn(Arc::new(App::new(loaded, cfg))).unwrap(), d)
    }

    struct Reply {
        status: u16,
        head: String,
        body: Vec<u8>,
    }
    impl Reply {
        fn json(&self) -> Value {
            serde_json::from_slice(&self.body).unwrap_or_else(|e| {
                panic!("not JSON ({e}): {}", String::from_utf8_lossy(&self.body))
            })
        }
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.body).into_owned()
        }
    }

    fn raw(addr: SocketAddr, bytes: &[u8]) -> Reply {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
        let _ = s.write_all(bytes);
        let mut all = vec![];
        let _ = s.read_to_end(&mut all);
        let split = all
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .unwrap_or_else(|| panic!("no head in {:?}", String::from_utf8_lossy(&all)));
        let head = String::from_utf8_lossy(&all[..split]).into_owned();
        let status: u16 = head.split(' ').nth(1).unwrap().parse().unwrap();
        let mut body = all[split + 4..].to_vec();
        if head
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            body = http::decode_chunked(&body, 1 << 24)
                .unwrap()
                .expect("complete chunked body")
                .0;
        }
        Reply { status, head, body }
    }

    fn post(addr: SocketAddr, path: &str, body: &str, extra: &str) -> Reply {
        raw(addr, format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{extra}\r\n{body}", body.len()).as_bytes())
    }
    fn get(addr: SocketAddr, path: &str, extra: &str) -> Reply {
        raw(
            addr,
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n{extra}\r\n").as_bytes(),
        )
    }

    const CHAT: &str = r#"{"model":"x","messages":[{"role":"system","content":"s"},{"role":"user","content":"Hello"}],"max_tokens":6,"temperature":0}"#;

    #[test]
    fn the_web_page_is_public_but_the_api_is_not() {
        let (h, d) = start(Some("sesame"), |_| {});
        for path in ["/", "/ui"] {
            let r = get(h.addr, path, "");
            assert_eq!(r.status, 200, "{path}");
            assert!(r.head.to_ascii_lowercase().contains("text/html"));
            for hdr in [
                "content-security-policy",
                "x-frame-options: deny",
                "cache-control: no-store",
                "x-content-type-options: nosniff",
            ] {
                assert!(r.head.to_ascii_lowercase().contains(hdr), "{path}: {hdr}");
            }
            assert!(r.text().contains("<html"));
        }
        assert_eq!(get(h.addr, "/v1/models", "").status, 401);
        assert_eq!(get(h.addr, "/capabilities", "").status, 401);
        let c = get(
            h.addr,
            "/v1/capabilities",
            "Authorization: Bearer sesame\r\n",
        );
        assert_eq!(c.status, 200);
        assert!(c.json()["data"].as_array().unwrap().len() > 10);
        // a wrong Host is refused even for the page (DNS-rebinding guard)
        assert_eq!(
            raw(h.addr, b"GET /ui HTTP/1.1\r\nHost: evil.example\r\n\r\n").status,
            403
        );
        assert_eq!(post(h.addr, "/ui", "{}", "").status, 401);
        assert_eq!(
            post(h.addr, "/ui", "{}", "Authorization: Bearer sesame\r\n").status,
            404
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn health_models_and_unknown_routes() {
        let (h, d) = start(None, |_| {});
        assert_eq!(get(h.addr, "/health", "").json()["status"], "ok");
        let m = get(h.addr, "/v1/models", "").json();
        assert_eq!(m["object"], "list");
        assert!(
            m["data"][0]["id"]
                .as_str()
                .unwrap()
                .starts_with("corbul-serve")
        );
        assert_eq!(m["data"][0]["context_length"], 96);
        assert_eq!(get(h.addr, "/v1/models/nope", "").status, 404);
        assert_eq!(get(h.addr, "/nope", "").status, 404);
        assert_eq!(get(h.addr, "/v1/chat/completions", "").status, 405);
        assert_eq!(
            raw(
                h.addr,
                b"OPTIONS /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"
            )
            .status,
            204
        );
        assert_eq!(
            get(h.addr, "/api/tags", "").json()["models"][0]["details"]["family"],
            "llama"
        );
        assert!(get(h.addr, "/api/version", "").json()["version"].is_string());
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn chat_completion_non_streaming_is_deterministic() {
        let (h, d) = start(None, |_| {});
        let a = post(h.addr, "/v1/chat/completions", CHAT, "");
        assert_eq!(a.status, 200, "{}", a.text());
        let j = a.json();
        assert_eq!(j["object"], "chat.completion");
        assert_eq!(j["choices"][0]["message"]["role"], "assistant");
        assert!(j["choices"][0]["message"]["content"].is_string());
        assert!(matches!(
            j["choices"][0]["finish_reason"].as_str(),
            Some("length" | "stop")
        ));
        let u = &j["usage"];
        assert!(u["prompt_tokens"].as_u64().unwrap() > 5);
        assert!(u["completion_tokens"].as_u64().unwrap() <= 6);
        assert_eq!(
            u["total_tokens"].as_u64().unwrap(),
            u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap()
        );
        let b = post(h.addr, "/v1/chat/completions", CHAT, "").json();
        assert_eq!(j["choices"], b["choices"]);
        // the same through the legacy completions endpoint
        let c = post(
            h.addr,
            "/v1/completions",
            r#"{"prompt":"abc","max_tokens":4,"temperature":0}"#,
            "",
        );
        assert_eq!(c.status, 200, "{}", c.text());
        assert_eq!(c.json()["object"], "text_completion");
        assert!(c.json()["choices"][0]["text"].is_string());
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn streaming_matches_non_streaming_and_ends_with_done() {
        let (h, d) = start(None, |_| {});
        let plain = post(h.addr, "/v1/chat/completions", CHAT, "").json();
        let want = plain["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();
        let body = CHAT.replace(
            r#""temperature":0"#,
            r#""temperature":0,"stream":true,"stream_options":{"include_usage":true}"#,
        );
        let r = post(h.addr, "/v1/chat/completions", &body, "");
        assert_eq!(r.status, 200);
        assert!(
            r.head.contains("text/event-stream")
                && r.head
                    .to_ascii_lowercase()
                    .contains("transfer-encoding: chunked"),
            "{}",
            r.head
        );
        let text = r.text();
        assert!(text.ends_with("data: [DONE]\n\n"), "{text:?}");
        let mut got = String::new();
        let (mut saw_role, mut finish, mut usage) = (false, None, None);
        for line in text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|l| *l != "[DONE]")
        {
            let v: Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["object"], "chat.completion.chunk");
            if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                got.push_str(c);
            }
            if v["choices"][0]["delta"]["role"] == "assistant" {
                saw_role = true;
            }
            if let Some(f) = v["choices"][0]["finish_reason"].as_str() {
                finish = Some(f.to_string());
            }
            if v.get("usage").is_some() {
                usage = Some(v["usage"].clone());
            }
        }
        assert!(saw_role);
        assert_eq!(got, want);
        assert_eq!(
            finish.as_deref(),
            plain["choices"][0]["finish_reason"].as_str()
        );
        assert_eq!(
            usage.unwrap()["prompt_tokens"],
            plain["usage"]["prompt_tokens"]
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn ollama_style_endpoints() {
        let (h, d) = start(None, |_| {});
        let r = post(
            h.addr,
            "/api/chat",
            r#"{"model":"x","messages":[{"role":"user","content":"Hi"}],"options":{"temperature":0,"num_predict":5}}"#,
            "",
        );
        assert_eq!(r.status, 200, "{}", r.text());
        assert!(r.head.contains("application/x-ndjson"));
        let lines: Vec<Value> = r
            .text()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert!(!lines.is_empty());
        let last = lines.last().unwrap();
        assert_eq!(last["done"], true);
        assert!(last["eval_count"].as_u64().unwrap() <= 5);
        assert!(
            lines[..lines.len() - 1]
                .iter()
                .all(|l| l["done"] == false && l["message"]["role"] == "assistant")
        );
        let one = post(
            h.addr,
            "/api/chat",
            r#"{"messages":[{"role":"user","content":"Hi"}],"stream":false,"options":{"temperature":0,"num_predict":5}}"#,
            "",
        );
        assert_eq!(one.json()["done"], true);
        assert!(one.json()["message"]["content"].is_string());
        let g = post(
            h.addr,
            "/api/generate",
            r#"{"prompt":"Hi","stream":false,"options":{"temperature":0,"num_predict":3}}"#,
            "",
        );
        assert!(g.json()["response"].is_string(), "{}", g.text());
        let e = post(h.addr, "/api/chat", r#"{"messages":"no"}"#, "");
        assert_eq!(e.status, 400);
        assert!(e.json()["error"].is_string());
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn the_bearer_token_is_enforced_except_on_health() {
        let (h, d) = start(Some("s3cret"), |_| {});
        assert_eq!(get(h.addr, "/health", "").status, 200);
        assert_eq!(get(h.addr, "/v1/models", "").status, 401);
        let r = get(h.addr, "/v1/models", "Authorization: Bearer wrong\r\n");
        assert_eq!(r.status, 401);
        assert!(r.head.contains("WWW-Authenticate: Bearer"));
        assert_eq!(
            get(h.addr, "/v1/models", "Authorization: Bearer s3cr3t\r\n").status,
            401
        );
        assert_eq!(
            get(h.addr, "/v1/models", "Authorization: Bearer s3cret\r\n").status,
            200
        );
        assert_eq!(post(h.addr, "/v1/chat/completions", CHAT, "").status, 401);
        assert_eq!(
            post(
                h.addr,
                "/v1/chat/completions",
                CHAT,
                "Authorization: Bearer s3cret\r\n"
            )
            .status,
            200
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn bad_requests_get_clean_errors() {
        let (h, d) = start(None, |c| c.limits.max_body = 2000);
        let p = |b: &str| post(h.addr, "/v1/chat/completions", b, "");
        assert_eq!(p("not json").status, 400);
        assert_eq!(p("[1,2]").status, 400);
        assert_eq!(p("{}").status, 400);
        assert_eq!(p(r#"{"messages":[]}"#).status, 400);
        assert_eq!(p(r#"{"messages":[{"content":"x"}]}"#).status, 400);
        assert_eq!(
            p(r#"{"messages":[{"role":"us er","content":"x"}]}"#).status,
            400
        );
        assert_eq!(
            p(r#"{"messages":[{"role":"user","content":5}]}"#).status,
            400
        );
        let long = format!(
            r#"{{"messages":[{{"role":"user","content":"{}"}}]}}"#,
            "a".repeat(500)
        );
        let r = p(&long);
        assert_eq!(r.status, 400, "{}", r.text());
        assert!(
            r.json()["error"]["message"]
                .as_str()
                .unwrap()
                .contains("context window")
        );
        assert_eq!(p(&"x".repeat(5000)).status, 413);
        assert_eq!(
            post(h.addr, "/v1/completions", r#"{"prompt":5}"#, "").status,
            400
        );
        // content given as a list of parts works
        let parts = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{}},{"type":"text","text":"b"}]}],"max_tokens":2,"temperature":0}"#;
        assert_eq!(p(parts).status, 200);
        assert_eq!(
            raw(
                h.addr,
                b"GET / HTTP/1.1\r\nHost: localhost\r\nBad Header: x\r\n\r\n"
            )
            .status,
            400
        );
        assert_eq!(
            raw(
                h.addr,
                format!(
                    "GET / HTTP/1.1\r\nHost: localhost\r\nX: {}\r\n\r\n",
                    "a".repeat(40_000)
                )
                .as_bytes()
            )
            .status,
            431
        );
        assert_eq!(
            raw(h.addr, b"GET / HTTP/1.1\r\nHost: evil.example\r\n\r\n").status,
            403
        );
        assert_eq!(
            raw(h.addr, b"GET / HTTP/9.9\r\nHost: localhost\r\n\r\n").status,
            505
        );
        // the server is still fine afterwards
        assert_eq!(get(h.addr, "/health", "").status, 200);
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn chunked_request_bodies_work() {
        let (h, d) = start(None, |_| {});
        let (a, b) = CHAT.split_at(20);
        let req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{a}\r\n{:x}\r\n{b}\r\n0\r\n\r\n",
            a.len(),
            b.len()
        );
        let r = raw(h.addr, req.as_bytes());
        assert_eq!(r.status, 200, "{}", r.text());
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_slow_client_is_cut_off_and_does_not_block_others() {
        let (h, d) = start(None, |c| c.limits.head_time = Duration::from_millis(400));
        let mut slow = TcpStream::connect(h.addr).unwrap();
        slow.set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        slow.write_all(b"GET /health HTTP/1.1\r\nX-Slow:").unwrap();
        // meanwhile another client is served immediately
        assert_eq!(get(h.addr, "/health", "").status, 200);
        let t = Instant::now();
        let mut out = vec![];
        let _ = slow.read_to_end(&mut out);
        assert!(
            String::from_utf8_lossy(&out).starts_with("HTTP/1.1 408"),
            "{:?}",
            String::from_utf8_lossy(&out)
        );
        assert!(t.elapsed() < Duration::from_secs(5));
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_client_that_disconnects_mid_stream_frees_the_model() {
        let (h, d) = start(None, |c| c.max_tokens = 90);
        for _ in 0..5 {
            let body = r#"{"messages":[{"role":"user","content":"x"}],"stream":true,"max_tokens":80,"temperature":1.0}"#;
            let mut s = TcpStream::connect(h.addr).unwrap();
            s.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
            let mut first = [0u8; 16];
            let _ = s.read(&mut first);
            drop(s);
        }
        let t = Instant::now();
        assert_eq!(post(h.addr, "/v1/chat/completions", CHAT, "").status, 200);
        assert!(t.elapsed() < Duration::from_secs(15));
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn concurrent_requests_queue_and_agree() {
        let (h, d) = start(None, |_| {});
        let addr = h.addr;
        let outs: Vec<Value> = (0..6)
            .map(|_| {
                std::thread::spawn(move || post(addr, "/v1/chat/completions", CHAT, "").json())
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect();
        assert!(outs.iter().all(|o| o["choices"] == outs[0]["choices"]));
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn too_many_connections_get_503() {
        let (h, d) = start(None, |c| c.max_connections = 1);
        // hold one connection open without sending anything
        let hold = TcpStream::connect(h.addr).unwrap();
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(get(h.addr, "/health", "").status, 503);
        drop(hold);
        std::thread::sleep(Duration::from_millis(700));
        assert_eq!(get(h.addr, "/health", "").status, 200);
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn stop_sequences_cut_and_hold_back_correctly() {
        let mut f = StopFilter::new(&["END".to_string(), "é!".to_string()]);
        let mut out = String::new();
        for piece in ["ab", "c E", "N", "Dxyz"] {
            let (o, hit) = f.push(piece);
            out.push_str(&o);
            if hit {
                break;
            }
        }
        assert_eq!(out, "abc ");
        let mut f = StopFilter::new(&["END".to_string()]);
        let (a, h1) = f.push("xE");
        assert_eq!((a.as_str(), h1), ("x", false));
        let (b, h2) = f.push("Nx");
        assert_eq!((b.as_str(), h2), ("ENx", false));
        let mut f = StopFilter::new(&["END".to_string()]);
        f.push("EN");
        assert_eq!(f.flush(), "EN");
        let mut f = StopFilter::new(&["é!".to_string()]);
        let (o, _) = f.push("aé");
        assert_eq!(o, "a");
        let (o, hit) = f.push("!z");
        assert!(hit && o.is_empty());
        let mut f = StopFilter::new(&[]);
        assert_eq!(f.push("anything"), ("anything".to_string(), false));
        // a stop sequence in the request ends generation with finish_reason "stop"
        let (h, d) = start(None, |_| {});
        let r = post(h.addr, "/v1/chat/completions", r#"{"messages":[{"role":"user","content":"Hello"}],"max_tokens":40,"temperature":0,"stop":["\u0000\u0001\u0002","a"]}"#, "").json();
        let text = r["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(!text.contains('a'));
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn helpers_behave() {
        assert!(ct_eq("abc", "abc") && !ct_eq("abc", "abd") && !ct_eq("abc", "ab"));
        assert_eq!(iso_now().len(), 20);
        assert!(
            model_error(ModelError::Token(9))
                .message
                .contains("vocabulary")
        );
    }

    // ---------------------------------------------------------------------- features: tools, Anthropic, metrics, keys, models

    fn start_ctx(ctx: usize, tweak: impl FnOnce(&mut ServeConfig)) -> (Handle, std::path::PathBuf) {
        let d = tmpdir("servectx");
        write_model_ctx(&d, true, ctx);
        let loaded = load(&d, &LoadCfg::default()).unwrap();
        let mut cfg = ServeConfig {
            port: 0,
            max_tokens: 8,
            temperature: 0.0,
            ..ServeConfig::default()
        };
        tweak(&mut cfg);
        (spawn(Arc::new(App::new(loaded, cfg))).unwrap(), d)
    }

    fn weather_tool() -> Value {
        json!({"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})
    }

    /// Forces the byte-level test model to write exactly one Hermes tool call.
    const ONE_CALL: &str = r#"root ::= "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>""#;
    const TWO_CALLS: &str = r#"root ::= "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call><tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}</tool_call>""#;

    fn chat_with_tools(extra: Value) -> String {
        let mut b = json!({"model": "x", "messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [weather_tool()], "max_tokens": 200, "temperature": 0, "grammar": ONE_CALL});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b.to_string()
    }

    #[test]
    fn a_tool_call_comes_back_as_tool_calls() {
        let (h, d) = start_ctx(2048, |_| {});
        let r = post(
            h.addr,
            "/v1/chat/completions",
            &chat_with_tools(json!({})),
            "",
        );
        assert_eq!(r.status, 200, "{}", r.text());
        let j = r.json();
        let c = &j["choices"][0];
        assert_eq!(c["finish_reason"], "tool_calls");
        assert!(c["message"]["content"].is_null(), "{c}");
        let call = &c["message"]["tool_calls"][0];
        assert!(call["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        let args: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"city": "Paris"}));
        // tool_choice "none": the same text stays text
        let r = post(
            h.addr,
            "/v1/chat/completions",
            &chat_with_tools(json!({"tool_choice": "none"})),
            "",
        )
        .json();
        assert!(r["choices"][0]["message"].get("tool_calls").is_none());
        assert!(
            r["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .starts_with("<tool_call>"),
            "{r}"
        );
        // naming a function that is not declared is the client's mistake
        assert_eq!(
            post(
                h.addr,
                "/v1/chat/completions",
                &chat_with_tools(
                    json!({"tool_choice": {"type": "function", "function": {"name": "nope"}}})
                ),
                ""
            )
            .status,
            400
        );
        assert_eq!(post(h.addr, "/v1/chat/completions", &chat_with_tools(json!({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}})), "").json()["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            post(
                h.addr,
                "/v1/chat/completions",
                &chat_with_tools(json!({"tools": "x"})),
                ""
            )
            .status,
            400
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn parallel_tool_calls_have_distinct_ids_and_can_be_limited() {
        let (h, d) = start_ctx(2048, |_| {});
        let b = |extra: Value| {
            chat_with_tools({
                let mut e = extra;
                e["grammar"] = json!(TWO_CALLS);
                e
            })
        };
        let j = post(h.addr, "/v1/chat/completions", &b(json!({})), "").json();
        let calls = j["choices"][0]["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2, "{j}");
        assert_ne!(calls[0]["id"], calls[1]["id"]);
        assert!(
            calls[1]["function"]["arguments"]
                .as_str()
                .unwrap()
                .contains("Rome")
        );
        let j = post(
            h.addr,
            "/v1/chat/completions",
            &b(json!({"parallel_tool_calls": false})),
            "",
        )
        .json();
        assert_eq!(
            j["choices"][0]["message"]["tool_calls"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn streamed_tool_calls_arrive_as_deltas() {
        let (h, d) = start_ctx(2048, |_| {});
        let r = post(
            h.addr,
            "/v1/chat/completions",
            &chat_with_tools(json!({"stream": true})),
            "",
        );
        assert_eq!(r.status, 200);
        let text = r.text();
        assert!(text.ends_with("data: [DONE]\n\n"));
        let (mut name, mut args, mut finish, mut content) =
            (String::new(), String::new(), None, String::new());
        for line in text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|l| *l != "[DONE]")
        {
            let v: Value = serde_json::from_str(line).unwrap();
            let d = &v["choices"][0]["delta"];
            if let Some(c) = d["content"].as_str() {
                content.push_str(c);
            }
            if let Some(t) = d["tool_calls"].get(0) {
                assert_eq!(t["index"], 0);
                name.push_str(t["function"]["name"].as_str().unwrap_or(""));
                args.push_str(t["function"]["arguments"].as_str().unwrap_or(""));
            }
            if let Some(f) = v["choices"][0]["finish_reason"].as_str() {
                finish = Some(f.to_string());
            }
        }
        assert_eq!(
            (name.as_str(), finish.as_deref()),
            ("get_weather", Some("tool_calls"))
        );
        assert_eq!(
            serde_json::from_str::<Value>(&args).unwrap(),
            json!({"city": "Paris"})
        );
        assert_eq!(
            content, "",
            "the call markup must not leak into the content"
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn tool_results_round_trip_through_the_template() {
        let (h, d) = start_ctx(2048, |_| {});
        let msgs = json!([{"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "18 C and sunny"}]);
        let r = post(
            h.addr,
            "/apply-template",
            &json!({"messages": msgs, "tools": [weather_tool()]}).to_string(),
            "",
        );
        assert_eq!(r.status, 200, "{}", r.text());
        let p = r.json()["prompt"].as_str().unwrap().to_string();
        assert!(
            p.contains("get_weather")
                && p.contains("18 C and sunny")
                && p.ends_with("<|im_start|>assistant\n"),
            "{p}"
        );
        let r = post(h.addr, "/v1/chat/completions", &json!({"messages": msgs, "tools": [weather_tool()], "max_tokens": 4, "temperature": 0}).to_string(), "");
        assert_eq!(r.status, 200, "{}", r.text());
        // a malformed assistant tool call is a 400
        let bad = json!({"messages": [{"role": "assistant", "tool_calls": [{"function": {}}]}]})
            .to_string();
        assert_eq!(post(h.addr, "/v1/chat/completions", &bad, "").status, 400);
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn anthropic_messages_non_streaming_and_streaming() {
        let (h, d) = start_ctx(2048, |_| {});
        let tool = json!({"name": "get_weather", "description": "d", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}});
        let body = json!({"model": "x", "max_tokens": 200, "temperature": 0, "system": "Be brief.", "messages": [{"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]}], "tools": [tool], "grammar": ONE_CALL});
        let r = post(
            h.addr,
            "/v1/messages",
            &body.to_string(),
            "anthropic-version: 2023-06-01\r\n",
        );
        assert_eq!(r.status, 200, "{}", r.text());
        let j = r.json();
        assert_eq!(
            (
                j["type"].as_str(),
                j["role"].as_str(),
                j["stop_reason"].as_str()
            ),
            (Some("message"), Some("assistant"), Some("tool_use"))
        );
        assert_eq!(j["content"][0]["type"], "tool_use");
        assert_eq!(j["content"][0]["name"], "get_weather");
        assert_eq!(j["content"][0]["input"], json!({"city": "Paris"}));
        assert!(j["content"][0]["id"].as_str().unwrap().starts_with("call_"));
        assert!(
            j["usage"]["input_tokens"].as_u64().unwrap() > 10
                && j["usage"]["output_tokens"].as_u64().unwrap() > 10
        );
        // streaming: the documented event order
        let mut sb = body.clone();
        sb["stream"] = json!(true);
        let r = post(h.addr, "/v1/messages", &sb.to_string(), "");
        assert_eq!(r.status, 200);
        assert!(r.head.contains("text/event-stream"));
        let text = r.text();
        let events: Vec<(String, Value)> = text
            .split("\n\n")
            .filter(|b| !b.is_empty())
            .map(|b| {
                let ev = b
                    .lines()
                    .find_map(|l| l.strip_prefix("event: "))
                    .unwrap()
                    .to_string();
                let data: Value =
                    serde_json::from_str(b.lines().find_map(|l| l.strip_prefix("data: ")).unwrap())
                        .unwrap();
                assert_eq!(data["type"], ev.as_str());
                (ev, data)
            })
            .collect();
        let names: Vec<&str> = events.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(names.first(), Some(&"message_start"));
        assert_eq!(names[names.len() - 2..], ["message_delta", "message_stop"]);
        let start = names
            .iter()
            .position(|n| *n == "content_block_start")
            .unwrap();
        assert_eq!(
            &names[start..start + 3],
            [
                "content_block_start",
                "content_block_delta",
                "content_block_stop"
            ]
        );
        assert_eq!(events[start].1["content_block"]["type"], "tool_use");
        assert_eq!(events[start + 1].1["delta"]["type"], "input_json_delta");
        assert_eq!(
            serde_json::from_str::<Value>(
                events[start + 1].1["delta"]["partial_json"]
                    .as_str()
                    .unwrap()
            )
            .unwrap(),
            json!({"city": "Paris"})
        );
        assert_eq!(
            events[events.len() - 2].1["delta"]["stop_reason"],
            "tool_use"
        );
        // plain text reply, max_tokens stop reason
        let r = post(h.addr, "/v1/messages", &json!({"max_tokens": 3, "temperature": 0, "messages": [{"role": "user", "content": "Hi"}]}).to_string(), "").json();
        assert_eq!(r["content"][0]["type"], "text");
        assert!(matches!(
            r["stop_reason"].as_str(),
            Some("max_tokens" | "end_turn")
        ));
        // token counting agrees with what a request is charged
        let c = post(
            h.addr,
            "/v1/messages/count_tokens",
            &json!({"messages": [{"role": "user", "content": "Hi"}]}).to_string(),
            "",
        )
        .json();
        assert_eq!(c["input_tokens"], r["usage"]["input_tokens"]);
        // errors use Anthropic's shape
        let e = post(
            h.addr,
            "/v1/messages",
            &json!({"messages": [{"role": "user", "content": "Hi"}]}).to_string(),
            "",
        );
        assert_eq!(e.status, 400);
        assert_eq!(
            (
                e.json()["type"].as_str(),
                e.json()["error"]["type"].as_str()
            ),
            (Some("error"), Some("invalid_request_error"))
        );
        assert!(
            e.json()["error"]["message"]
                .as_str()
                .unwrap()
                .contains("max_tokens")
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn openai_extras_n_logprobs_bias_seed_and_limits() {
        let (h, d) = start(None, |_| {});
        let base = |extra: Value| {
            let mut b = json!({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 6, "temperature": 0});
            for (k, v) in extra.as_object().unwrap() {
                b[k] = v.clone();
            }
            b.to_string()
        };
        let p = |extra: Value| post(h.addr, "/v1/chat/completions", &base(extra), "");
        // n choices
        let j = p(json!({"n": 3, "temperature": 1.0})).json();
        assert_eq!(j["choices"].as_array().unwrap().len(), 3);
        assert_eq!(j["choices"][2]["index"], 2);
        let sum: u64 = 0;
        assert!(j["usage"]["completion_tokens"].as_u64().unwrap() >= sum);
        assert_eq!(p(json!({"n": 0})).status, 400);
        assert_eq!(p(json!({"n": 99})).status, 400);
        // logit_bias forces a token
        let j = p(json!({"logit_bias": {"65": 100}})).json();
        assert_eq!(j["choices"][0]["message"]["content"], "AAAAAA");
        assert_eq!(p(json!({"logit_bias": {"x": 1}})).status, 400);
        // logprobs
        let j = p(json!({"logprobs": true, "top_logprobs": 3, "max_tokens": 4})).json();
        let lp = j["choices"][0]["logprobs"]["content"].as_array().unwrap();
        assert_eq!(
            lp.len() as u64,
            j["usage"]["completion_tokens"].as_u64().unwrap()
        );
        for e in lp {
            assert!(e["logprob"].as_f64().unwrap() <= 0.0);
            assert_eq!(e["top_logprobs"].as_array().unwrap().len(), 3);
            assert!(
                (e["top_logprobs"][0]["logprob"].as_f64().unwrap()
                    - e["logprob"].as_f64().unwrap())
                .abs()
                    < 1e-4,
                "greedy picks the most likely token"
            );
        }
        assert!(p(json!({})).json()["choices"][0]["logprobs"].is_null());
        // legacy completions logprobs
        let c = post(
            h.addr,
            "/v1/completions",
            r#"{"prompt":"ab","max_tokens":3,"temperature":0,"logprobs":2}"#,
            "",
        )
        .json();
        assert_eq!(
            c["choices"][0]["logprobs"]["tokens"]
                .as_array()
                .unwrap()
                .len(),
            c["usage"]["completion_tokens"].as_u64().unwrap() as usize
        );
        // max_completion_tokens, seed
        assert!(p(json!({"max_tokens": null, "max_completion_tokens": 2})).json()["usage"]["completion_tokens"].as_u64().unwrap() <= 2);
        let s1 = p(json!({"temperature": 1.0, "seed": 7})).json();
        let s2 = p(json!({"temperature": 1.0, "seed": 7})).json();
        assert_eq!(s1["choices"], s2["choices"]);
        // penalties are accepted
        assert_eq!(
            p(json!({"presence_penalty": 1.5, "frequency_penalty": 0.5})).status,
            200
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn structured_output_constrains_the_reply() {
        let (h, d) = start(None, |_| {});
        let r = post(h.addr, "/v1/chat/completions", &json!({"messages": [{"role": "user", "content": "yes or no?"}], "max_tokens": 12, "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": {"enum": ["yes", "no"]}}}}).to_string(), "");
        assert_eq!(r.status, 200, "{}", r.text());
        let t = r.json()["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(t == "\"yes\"" || t == "\"no\"", "{t:?}");
        let r = post(h.addr, "/v1/chat/completions", &json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 12, "temperature": 0, "response_format": {"type": "json_object"}}).to_string(), "");
        assert!(
            r.json()["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .starts_with('{'),
            "{}",
            r.text()
        );
        // Ollama's `format`
        let r = post(h.addr, "/api/generate", &json!({"prompt": "x", "stream": false, "format": {"enum": ["a", "b"]}, "options": {"temperature": 0, "num_predict": 8}}).to_string(), "");
        let t = r.json()["response"].as_str().unwrap_or("").to_string();
        assert!(t == "\"a\"" || t == "\"b\"", "{t:?} {}", r.text());
        // a broken grammar or schema is a 400, not a hang
        assert_eq!(
            post(
                h.addr,
                "/v1/chat/completions",
                &json!({"messages": [{"role": "user", "content": "x"}], "grammar": "root ::= ("})
                    .to_string(),
                ""
            )
            .status,
            400
        );
        assert_eq!(post(h.addr, "/v1/chat/completions", &json!({"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "xml"}}).to_string(), "").status, 400);
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn metrics_and_utility_endpoints() {
        let (h, d) = start(None, |_| {});
        assert_eq!(post(h.addr, "/v1/chat/completions", CHAT, "").status, 200);
        let m = get(h.addr, "/metrics", "");
        assert_eq!(m.status, 200);
        assert!(m.head.contains("text/plain; version=0.0.4"));
        let t = m.text();
        for want in [
            "corbul_requests_total{endpoint=\"/v1/chat/completions\",status=\"200\"} 1",
            "corbul_prompt_tokens_total{model=\"corbul-serve",
            "corbul_completion_tokens_total{",
            "corbul_request_duration_seconds_count",
            "corbul_models_loaded 1",
            "corbul_queue_depth 0",
            "corbul_tokens_per_second ",
            "corbul_time_to_first_token_seconds_count 1",
            "corbul_model_weight_bytes{model=",
        ] {
            assert!(t.contains(want), "missing {want} in\n{t}");
        }
        // /tokenize and /detokenize
        let tk = post(h.addr, "/tokenize", r#"{"content":"Hi"}"#, "").json();
        assert_eq!(tk["tokens"], json!([72, 105]));
        assert_eq!(
            post(
                h.addr,
                "/tokenize",
                r#"{"content":"Hi","with_pieces":true}"#,
                ""
            )
            .json()["tokens"][1]["piece"],
            "i"
        );
        assert_eq!(
            post(h.addr, "/detokenize", r#"{"tokens":[72,105]}"#, "").json()["content"],
            "Hi"
        );
        assert_eq!(
            post(h.addr, "/detokenize", r#"{"tokens":[99999]}"#, "").status,
            400
        );
        assert_eq!(
            post(h.addr, "/tokenize", r#"{"content":5}"#, "").status,
            400
        );
        // /apply-template
        let ap = post(
            h.addr,
            "/apply-template",
            r#"{"messages":[{"role":"user","content":"Hi"}]}"#,
            "",
        )
        .json();
        assert_eq!(
            ap["prompt"],
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        // /props and /slots
        let p = get(h.addr, "/props", "").json();
        assert_eq!(p["n_ctx"], 96);
        assert!(p["model"].as_str().unwrap().starts_with("corbul-serve"));
        let s = get(h.addr, "/slots", "").json();
        assert_eq!(s[0]["n_ctx"], 96);
        assert_eq!(s[0]["is_processing"], false);
        assert!(
            s[0]["n_cached"].as_u64().unwrap() > 0,
            "the conversation stays cached for the next turn"
        );
        // embeddings and rerank say plainly that they are not available
        for path in ["/v1/embeddings", "/api/embed", "/v1/rerank", "/rerank"] {
            let r = post(h.addr, path, "{}", "");
            assert_eq!(r.status, 501, "{path}");
            assert!(r.json().to_string().contains("not available"), "{path}");
        }
        assert_eq!(
            get(h.addr, "/metrics", "")
                .text()
                .matches("# TYPE corbul_requests_total")
                .count(),
            1
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn api_keys_limit_models_and_rates() {
        let keys = KeyStore::parse(
            r#"{"keys":[
            {"key":"sk-alice-0001","name":"alice","models":["corbul-serve*"]},
            {"key":"sk-bob-000002","name":"bob","models":["something-else"]},
            {"key":"sk-carol-0003","name":"carol","rpm":2},
            {"key":"sk-dave-00004","name":"dave","tpm":5}]}"#,
        )
        .unwrap();
        let (h, d) = start(None, |c| c.keys = Some(keys));
        let auth = |k: &str| format!("Authorization: Bearer {k}\r\n");
        assert_eq!(get(h.addr, "/v1/models", "").status, 401);
        assert_eq!(get(h.addr, "/health", "").status, 200);
        assert_eq!(
            get(h.addr, "/v1/models", &auth("sk-nobody-0000")).status,
            401
        );
        assert_eq!(
            post(h.addr, "/v1/chat/completions", CHAT, &auth("sk-alice-0001")).status,
            200
        );
        assert_eq!(
            post(
                h.addr,
                "/v1/chat/completions",
                CHAT,
                "x-api-key: sk-alice-0001\r\n"
            )
            .status,
            200
        );
        // bob may not use this model, and does not even see it listed
        let r = post(h.addr, "/v1/chat/completions", CHAT, &auth("sk-bob-000002"));
        assert_eq!(r.status, 403, "{}", r.text());
        assert_eq!(
            get(h.addr, "/v1/models", &auth("sk-bob-000002")).json()["data"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            get(h.addr, "/v1/models", &auth("sk-alice-0001")).json()["data"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        // carol: two requests a minute
        assert_eq!(
            post(h.addr, "/v1/chat/completions", CHAT, &auth("sk-carol-0003")).status,
            200
        );
        assert_eq!(
            post(h.addr, "/v1/chat/completions", CHAT, &auth("sk-carol-0003")).status,
            200
        );
        let r = post(h.addr, "/v1/chat/completions", CHAT, &auth("sk-carol-0003"));
        assert_eq!(r.status, 429, "{}", r.text());
        assert!(
            r.head.to_ascii_lowercase().contains("retry-after: "),
            "{}",
            r.head
        );
        assert_eq!(r.json()["error"]["type"], "rate_limit_error");
        // alice is unaffected by carol's limit
        assert_eq!(
            post(h.addr, "/v1/chat/completions", CHAT, &auth("sk-alice-0001")).status,
            200
        );
        // dave: a prompt bigger than his token budget can never succeed
        let r = post(h.addr, "/v1/chat/completions", CHAT, &auth("sk-dave-00004"));
        assert_eq!(r.status, 400);
        assert!(
            r.json()["error"]["message"]
                .as_str()
                .unwrap()
                .contains("tokens-per-minute")
        );
        // the Anthropic route reports auth failures in its own shape
        let r = post(h.addr, "/v1/messages", "{}", "");
        assert_eq!((r.status, r.json()["type"].as_str()), (401, Some("error")));
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn the_single_token_and_keys_work_together() {
        let keys = KeyStore::parse(r#"["sk-alice-0001"]"#).unwrap();
        let (h, d) = start(Some("admin-token"), |c| c.keys = Some(keys));
        assert_eq!(
            get(
                h.addr,
                "/v1/models",
                "Authorization: Bearer admin-token\r\n"
            )
            .status,
            200
        );
        assert_eq!(
            get(
                h.addr,
                "/v1/models",
                "Authorization: Bearer sk-alice-0001\r\n"
            )
            .status,
            200
        );
        assert_eq!(
            get(h.addr, "/v1/models", "Authorization: Bearer nope\r\n").status,
            401
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn cors_headers_and_preflight() {
        let (h, d) = start(Some("tok"), |c| {
            c.cors = Some(vec!["https://app.example".into()])
        });
        let ok = get(h.addr, "/health", "Origin: https://app.example\r\n");
        assert!(
            ok.head
                .contains("Access-Control-Allow-Origin: https://app.example")
                && ok.head.contains("Vary: Origin"),
            "{}",
            ok.head
        );
        assert!(
            !get(h.addr, "/health", "Origin: https://evil.example\r\n")
                .head
                .contains("Access-Control-Allow-Origin")
        );
        // browsers send the preflight without credentials
        let pf = raw(h.addr, b"OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nOrigin: https://app.example\r\nAccess-Control-Request-Method: POST\r\nAccess-Control-Request-Headers: authorization,content-type\r\n\r\n");
        assert_eq!(pf.status, 204);
        assert!(
            pf.head
                .contains("Access-Control-Allow-Origin: https://app.example")
                && pf.head.contains("Access-Control-Allow-Methods")
                && pf
                    .head
                    .contains("Access-Control-Allow-Headers: authorization,content-type"),
            "{}",
            pf.head
        );
        assert!(
            !get(h.addr, "/health", "")
                .head
                .contains("Access-Control-Allow-Origin")
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
        let (h, d) = start(None, |c| c.cors = Some(vec!["*".into()]));
        assert!(
            get(h.addr, "/health", "Origin: https://anything.example\r\n")
                .head
                .contains("Access-Control-Allow-Origin: *")
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
        let (h, d) = start(None, |_| {});
        assert!(
            !get(h.addr, "/health", "Origin: https://anything.example\r\n")
                .head
                .contains("Access-Control-Allow-Origin")
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn the_request_log_never_holds_prompts_unless_asked() {
        use std::sync::Mutex;
        let lines: Arc<Mutex<Vec<String>>> = Arc::default();
        let l2 = lines.clone();
        let (h, d) = start(None, move |c| {
            c.log = Some(LogSink(Arc::new(move |s: &str| {
                l2.lock().unwrap().push(s.to_string())
            })))
        });
        let secret = r#"{"messages":[{"role":"user","content":"my secret prompt"}],"max_tokens":3,"temperature":0}"#;
        assert_eq!(post(h.addr, "/v1/chat/completions", secret, "").status, 200);
        assert_eq!(get(h.addr, "/nope", "").status, 404);
        let got = lines.lock().unwrap().clone();
        assert!(
            got.iter()
                .any(|l| l.contains("POST /v1/chat/completions -> 200")
                    && l.contains("model=corbul-serve")
                    && l.contains("tokens=")),
            "{got:?}"
        );
        assert!(got.iter().any(|l| l.contains("GET /nope -> 404")));
        assert!(!got.iter().any(|l| l.contains("secret")), "{got:?}");
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
        let lines: Arc<Mutex<Vec<String>>> = Arc::default();
        let l2 = lines.clone();
        let (h, d) = start(None, move |c| {
            c.log_prompts = true;
            c.log = Some(LogSink(Arc::new(move |s: &str| {
                l2.lock().unwrap().push(s.to_string())
            })));
        });
        post(h.addr, "/v1/chat/completions", secret, "");
        assert!(
            lines
                .lock()
                .unwrap()
                .iter()
                .any(|l| l.contains("my secret prompt"))
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn several_models_one_server() {
        let root = tmpdir("multi");
        for n in ["alpha", "beta"] {
            let m = root.join(n);
            std::fs::create_dir_all(&m).unwrap();
            write_model(&m, true);
        }
        std::fs::write(root.join("wordy.Modelfile"), "FROM alpha\nSYSTEM \"\"\"You always answer in a very very very long-winded way.\"\"\"\nPARAMETER num_predict 2\n").unwrap();
        let reg = Registry::new(RegCfg {
            max_loaded: 1,
            ..RegCfg::default()
        });
        reg.scan_dir(&root).unwrap();
        reg.alias("gpt-4", "beta").unwrap();
        let app = Arc::new(App::with_registry(
            reg,
            ServeConfig {
                port: 0,
                max_tokens: 4,
                temperature: 0.0,
                ..ServeConfig::default()
            },
        ));
        let h = spawn(app.clone()).unwrap();
        let listed = get(h.addr, "/v1/models", "").json();
        let ids: Vec<&str> = listed["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["alpha", "beta", "wordy"]);
        assert!(
            listed["data"]
                .as_array()
                .unwrap()
                .iter()
                .all(|m| m["loaded"] == false),
            "nothing loads until it is used"
        );
        let ask = |model: &str| {
            post(h.addr, "/v1/chat/completions", &json!({"model": model, "messages": [{"role": "user", "content": "Hi"}], "temperature": 0}).to_string(), "")
        };
        let a = ask("alpha");
        assert_eq!(a.status, 200, "{}", a.text());
        assert_eq!(a.json()["model"], "alpha");
        assert_eq!(app.registry.loaded_count(), 1);
        let b = ask("gpt-4");
        assert_eq!(
            b.json()["model"],
            "beta",
            "the alias resolves to the real name"
        );
        assert_eq!(app.registry.loaded_count(), 1, "max_loaded=1 evicted alpha");
        assert!(
            !get(h.addr, "/api/ps", "").json()["models"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["name"] == "alpha")
        );
        let w = ask("wordy").json();
        assert!(
            w["usage"]["prompt_tokens"].as_u64().unwrap()
                > a.json()["usage"]["prompt_tokens"].as_u64().unwrap() + 30,
            "the Modelfile's system prompt is part of the prompt"
        );
        assert!(
            w["usage"]["completion_tokens"].as_u64().unwrap() <= 2,
            "and its PARAMETER is the default"
        );
        let missing = ask("nope");
        assert_eq!(missing.status, 404);
        assert_eq!(missing.json()["error"]["type"], "not_found_error");
        // no model field → the default (first) model
        assert_eq!(
            post(
                h.addr,
                "/v1/chat/completions",
                r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":2}"#,
                ""
            )
            .json()["model"],
            "alpha"
        );
        // admin: unload
        let u = post(h.addr, "/models/unload", r#"{"model":"alpha"}"#, "").json();
        assert_eq!(u["model"], "alpha");
        h.stop();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ollama_create_show_ps_delete() {
        let (h, d) = start(None, |_| {});
        let base = get(h.addr, "/v1/models", "").json()["data"][0]["id"]
            .as_str()
            .unwrap()
            .to_string();
        let mf = format!("FROM {base}\nSYSTEM \"\"\"s\"\"\"\nPARAMETER num_predict 3\n");
        let r = post(
            h.addr,
            "/api/create",
            &json!({"model": "mini", "modelfile": mf}).to_string(),
            "",
        );
        assert_eq!(
            (r.status, r.json()["status"].as_str()),
            (200, Some("success")),
            "{}",
            r.text()
        );
        assert!(
            get(h.addr, "/api/tags", "").json()["models"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["name"] == "mini")
        );
        let c = post(h.addr, "/api/chat", &json!({"model": "mini", "messages": [{"role": "user", "content": "Hi"}], "stream": false, "options": {"temperature": 0}}).to_string(), "").json();
        assert_eq!(c["done"], true);
        assert!(c["eval_count"].as_u64().unwrap() <= 3, "{c}");
        let s = post(h.addr, "/api/show", r#"{"model":"mini"}"#, "").json();
        assert_eq!(s["system"], "s");
        assert!(s["parameters"].as_str().unwrap().contains("num_predict"));
        assert!(
            get(h.addr, "/api/ps", "").json()["models"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["name"] == "mini")
        );
        let bad = post(
            h.addr,
            "/api/create",
            r#"{"model":"x","modelfile":"FROM /does/not/exist"}"#,
            "",
        );
        assert_eq!(bad.status, 400);
        assert!(bad.json()["error"].is_string());
        assert_eq!(raw(h.addr, b"DELETE /api/delete HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 16\r\nConnection: close\r\n\r\n{\"model\":\"mini\"}").status, 200);
        assert!(
            !get(h.addr, "/api/tags", "").json()["models"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["name"] == "mini")
        );
        assert_eq!(
            post(h.addr, "/api/delete", r#"{"model":"mini"}"#, "").status,
            404
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn shutdown_lets_a_running_request_finish() {
        let (h, d) = start(None, |c| c.max_tokens = 60);
        let addr = h.addr;
        let body =
            r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":60,"temperature":1.0}"#;
        let t = std::thread::spawn(move || post(addr, "/v1/chat/completions", body, ""));
        // wait until the connection is being served
        let t0 = Instant::now();
        while h.app.active.load(Ordering::SeqCst) == 0 && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(2));
        }
        let app = h.app.clone();
        h.stop();
        assert_eq!(
            app.active.load(Ordering::SeqCst),
            0,
            "stop() waits for running requests"
        );
        let r = t.join().unwrap();
        assert_eq!(r.status, 200, "{}", r.text());
        assert!(r.json()["choices"][0]["message"]["content"].is_string());
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn repeating_a_conversation_reuses_the_cached_prefix() {
        let (h, d) = start(None, |_| {});
        let one = |extra: &str| {
            post(h.addr, "/v1/chat/completions", &format!(r#"{{"messages":[{{"role":"system","content":"s"}},{{"role":"user","content":"Hello{extra}"}}],"max_tokens":3,"temperature":0}}"#), "").json()
        };
        let a = one("");
        let b = one("");
        assert_eq!(
            a["choices"], b["choices"],
            "reusing the cache must not change the answer"
        );
        let c = one(" there");
        assert_eq!(
            c["usage"]["prompt_tokens"].as_u64().unwrap(),
            a["usage"]["prompt_tokens"].as_u64().unwrap() + 6
        );
        // and an unrelated prompt after it is unaffected by what was cached
        let d2 = post(h.addr, "/v1/chat/completions", CHAT, "").json();
        let fresh = {
            let (h2, dd) = start(None, |_| {});
            let r = post(h2.addr, "/v1/chat/completions", CHAT, "").json();
            h2.stop();
            std::fs::remove_dir_all(dd).unwrap();
            r
        };
        assert_eq!(d2["choices"], fresh["choices"]);
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn quote_lock_requests_carry_citations_and_bad_ones_are_refused() {
        let (h, d) = start_ctx(1024, |c| c.max_tokens = 24);
        let src = r#""sources":["The quick brown fox jumps over the lazy dog near the river bank.","Second source about stars and planets."]"#;
        let plain = post(h.addr, "/v1/chat/completions", CHAT, "").json();
        let locked = post(
            h.addr,
            "/v1/chat/completions",
            &format!(
                r#"{{"messages":[{{"role":"user","content":"Hello"}}],"max_tokens":8,"temperature":0,{src}}}"#
            ),
            "",
        );
        assert_eq!(locked.status, 200, "{}", locked.text());
        let j = locked.json();
        assert!(j["choices"][0]["citations"]["quotes"].is_array(), "{j}");
        assert!(j["choices"][0]["citations"]["counts"].is_object());
        assert!(
            j["usage"]["prompt_tokens"].as_u64().unwrap()
                > plain["usage"]["prompt_tokens"].as_u64().unwrap(),
            "the sources are part of the prompt"
        );
        // the lock can be switched off and the answer still checked
        let off = post(h.addr, "/v1/chat/completions", &format!(r#"{{"messages":[{{"role":"user","content":"Hello"}}],"max_tokens":4,"temperature":0,"quote_lock":false,{src}}}"#), "").json();
        assert!(off["choices"][0]["citations"].is_object());
        // other dialects
        let ol = post(h.addr, "/api/chat", &format!(r#"{{"messages":[{{"role":"user","content":"Hi"}}],"stream":false,"options":{{"temperature":0,"num_predict":4}},{src}}}"#), "").json();
        assert!(ol["citations"]["quotes"].is_array(), "{ol}");
        // refusals
        let both = post(
            h.addr,
            "/v1/chat/completions",
            &format!(
                r#"{{"messages":[{{"role":"user","content":"Hi"}}],"response_format":{{"type":"json_object"}},{src}}}"#
            ),
            "",
        );
        assert_eq!(both.status, 400);
        assert_eq!(
            post(
                h.addr,
                "/v1/chat/completions",
                r#"{"messages":[{"role":"user","content":"Hi"}],"sources":"nope"}"#,
                ""
            )
            .status,
            400
        );
        assert_eq!(
            post(
                h.addr,
                "/v1/chat/completions",
                r#"{"messages":[{"role":"user","content":"Hi"}],"sources":[""]}"#,
                ""
            )
            .status,
            400
        );
        // no sources: no citations key
        assert!(plain["choices"][0].get("citations").is_none());
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn the_verify_endpoint_checks_text_against_sources() {
        let (h, d) = start(None, |_| {});
        let body = r#"{"text":"He wrote [S1]«quick brown fox jumps over the lazy dog» and [S1]«nothing like this at all anywhere»","sources":["The quick brown fox jumps over the lazy dog near the river bank."]}"#;
        let r = post(h.addr, "/v1/quotes/verify", body, "");
        assert_eq!(r.status, 200, "{}", r.text());
        let j = r.json();
        assert_eq!(j["quotes"][0]["status"], "verbatim");
        assert_ne!(j["quotes"][1]["status"], "verbatim");
        assert_eq!(j["all_verbatim"], false);
        assert_eq!(
            post(h.addr, "/v1/quotes/verify", r#"{"text":"x"}"#, "").status,
            400
        );
        assert_eq!(
            post(h.addr, "/v1/quotes/verify", r#"{"sources":["a b c"]}"#, "").status,
            400
        );
        assert_eq!(get(h.addr, "/v1/quotes/verify", "").status, 405);
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    // ---------------------------------------------------------------------- hostile-review regressions

    #[test]
    fn cross_site_simple_requests_cannot_drive_the_server() {
        // open mode: no token, so the only defences are Content-Type and Origin
        let (h, d) = start(None, |_| {});
        let plain = raw(h.addr, format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{CHAT}", CHAT.len()).as_bytes());
        assert_eq!(plain.status, 415, "{}", plain.text());
        let form = raw(h.addr, format!("POST /api/create HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/x-www-form-urlencoded\r\nContent-Length: {}\r\n\r\n{CHAT}", CHAT.len()).as_bytes());
        assert_eq!(form.status, 415);
        let none = raw(h.addr, format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{CHAT}", CHAT.len()).as_bytes());
        assert_eq!(none.status, 415);
        // JSON with a charset parameter is fine
        assert_eq!(post(h.addr, "/v1/chat/completions", CHAT, "").status, 200);
        let cs = raw(h.addr, format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: Application/JSON; charset=utf-8\r\nContent-Length: {}\r\n\r\n{CHAT}", CHAT.len()).as_bytes());
        assert_eq!(cs.status, 200, "{}", cs.text());
        // a foreign Origin is refused, whatever the content type
        let evil = post(
            h.addr,
            "/v1/chat/completions",
            CHAT,
            "Origin: https://evil.example\r\n",
        );
        assert_eq!(evil.status, 403, "{}", evil.text());
        assert_eq!(
            get(h.addr, "/v1/models", "Origin: https://evil.example\r\n").status,
            403
        );
        assert_eq!(
            post(h.addr, "/v1/chat/completions", CHAT, "Origin: null\r\n").status,
            403
        );
        // the page's own origin (same host) is fine
        assert_eq!(
            post(
                h.addr,
                "/v1/chat/completions",
                CHAT,
                "Origin: http://localhost\r\n"
            )
            .status,
            200
        );
        // no Origin (curl, SDKs) is fine
        assert_eq!(get(h.addr, "/v1/models", "").status, 200);
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
        // an origin allowed with --cors passes; others still do not
        let (h, d) = start(None, |c| c.cors = Some(vec!["https://app.example".into()]));
        assert_eq!(
            post(
                h.addr,
                "/v1/chat/completions",
                CHAT,
                "Origin: https://app.example\r\n"
            )
            .status,
            200
        );
        assert_eq!(
            post(
                h.addr,
                "/v1/chat/completions",
                CHAT,
                "Origin: https://evil.example\r\n"
            )
            .status,
            403
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn admin_endpoints_need_the_master_token() {
        let keys = KeyStore::parse(r#"["sk-alice-0001"]"#).unwrap();
        let (h, d) = start(Some("admin-token-0123456789"), |c| c.keys = Some(keys));
        let alice = "Authorization: Bearer sk-alice-0001\r\n";
        let master = "Authorization: Bearer admin-token-0123456789\r\n";
        for (path, body) in [
            ("/api/create", r#"{"model":"m","from":"x"}"#),
            ("/api/delete", r#"{"model":"x"}"#),
            ("/models/load", "{}"),
            ("/models/unload", "{}"),
        ] {
            let r = post(h.addr, path, body, alice);
            assert_eq!(r.status, 403, "{path}: {}", r.text());
            assert!(r.text().contains("master token"), "{path}: {}", r.text());
            assert_ne!(
                post(h.addr, path, body, master).status,
                403,
                "{path} must stay open to the master token"
            );
        }
        // an ordinary call still works for the key
        assert_eq!(
            post(h.addr, "/v1/chat/completions", CHAT, alice).status,
            200
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
        // with --allow-admin the key holder may
        let keys = KeyStore::parse(r#"["sk-alice-0001"]"#).unwrap();
        let (h, d) = start(Some("admin-token-0123456789"), |c| {
            c.keys = Some(keys);
            c.allow_admin = true;
        });
        assert_ne!(post(h.addr, "/models/load", "{}", alice).status, 403);
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn props_honours_the_query_and_reveals_no_paths() {
        let root = tmpdir("props");
        for n in ["alpha", "beta"] {
            let m = root.join(n);
            std::fs::create_dir_all(&m).unwrap();
            write_model(&m, true);
        }
        let reg = Registry::new(RegCfg::default());
        reg.scan_dir(&root).unwrap();
        let h = spawn(Arc::new(App::with_registry(
            reg,
            ServeConfig {
                port: 0,
                max_tokens: 4,
                temperature: 0.0,
                ..ServeConfig::default()
            },
        )))
        .unwrap();
        assert_eq!(
            get(h.addr, "/props", "").json()["model"],
            "alpha",
            "no query: the default model"
        );
        let b = get(h.addr, "/props?model=beta", "");
        assert_eq!(b.status, 200, "{}", b.text());
        assert_eq!(
            b.json()["model"],
            "beta",
            "the query must be read (it used to be ignored)"
        );
        assert!(b.json().get("model_path").is_none(), "{}", b.text());
        assert!(
            !b.text().contains(root.to_str().unwrap()),
            "the server's directory leaked: {}",
            b.text()
        );
        assert_eq!(get(h.addr, "/props?model=nope", "").status, 404);
        assert_eq!(
            get(h.addr, "/props?x=1&model=%62eta", "").json()["model"],
            "beta",
            "percent-escapes decode"
        );
        h.stop();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn create_over_http_cannot_point_at_arbitrary_paths() {
        let (h, d) = start(None, |_| {});
        let id = get(h.addr, "/v1/models", "").json()["data"][0]["id"]
            .as_str()
            .unwrap()
            .to_string();
        // a real directory that is neither a served model nor inside a served directory
        let outside = tmpdir("outside");
        let a = post(
            h.addr,
            "/api/create",
            &json!({"model": "evil", "modelfile": format!("FROM {}", outside.display())})
                .to_string(),
            "",
        );
        let b = post(
            h.addr,
            "/api/create",
            &json!({"model": "evil", "modelfile": "FROM /definitely/not/here"}).to_string(),
            "",
        );
        assert_eq!(a.status, 400, "{}", a.text());
        assert_eq!(
            a.json()["error"],
            b.json()["error"],
            "existing and missing paths must look the same"
        );
        assert!(!a.text().contains(outside.to_str().unwrap()));
        let _ = std::fs::remove_dir_all(&outside);
        for p in ["/etc", "/", "../../../etc", "/etc/passwd"] {
            assert_eq!(
                post(
                    h.addr,
                    "/api/create",
                    &json!({"model": "evil", "modelfile": format!("FROM {p}")}).to_string(),
                    ""
                )
                .status,
                400,
                "{p}"
            );
        }
        // a served model by name works, and /api/show prints no directory
        assert_eq!(
            post(
                h.addr,
                "/api/create",
                &json!({"model": "okay", "modelfile": format!("FROM {id}")}).to_string(),
                ""
            )
            .status,
            200
        );
        let s = post(h.addr, "/api/show", r#"{"model":"okay"}"#, "").json();
        let mf = s["modelfile"].as_str().unwrap();
        assert!(
            !mf.contains('/') && !mf.contains(d.to_str().unwrap()),
            "{mf}"
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn cpu_heavy_endpoints_are_rate_limited_and_capped() {
        let keys = KeyStore::parse(r#"{"keys":[{"key":"sk-carol-0003","name":"carol","rpm":2}]}"#)
            .unwrap();
        let (h, d) = start(None, |c| c.keys = Some(keys));
        let auth = "Authorization: Bearer sk-carol-0003\r\n";
        assert_eq!(
            post(h.addr, "/tokenize", r#"{"content":"hi"}"#, auth).status,
            200
        );
        assert_eq!(
            post(
                h.addr,
                "/v1/quotes/verify",
                r#"{"text":"a b","sources":["a b c"]}"#,
                auth
            )
            .status,
            200
        );
        let r = post(h.addr, "/tokenize", r#"{"content":"hi"}"#, auth);
        assert_eq!(r.status, 429, "{}", r.text());
        assert!(r.head.to_ascii_lowercase().contains("retry-after: "));
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
        // the permit counter: at most `max` at once, released on drop
        let c = AtomicUsize::new(0);
        let (a, b) = (HeavyPermit::acquire(&c, 2), HeavyPermit::acquire(&c, 2));
        assert!(a.is_some() && b.is_some() && HeavyPermit::acquire(&c, 2).is_none());
        drop(a);
        assert!(HeavyPermit::acquire(&c, 2).is_some());
        assert_eq!(
            c.load(Ordering::SeqCst),
            1,
            "a refused acquire must not leak a count"
        );
        // max_heavy = 0 still lets one through (a misconfiguration must not brick the endpoints)
        assert!(HeavyPermit::acquire(&AtomicUsize::new(0), 0).is_some());
    }

    #[test]
    fn quote_verification_inputs_are_capped_over_http() {
        let (h, d) = start(None, |_| {});
        let big = "word ".repeat(crate::quotes::MAX_ANSWER_BYTES / 5 + 10);
        let r = post(
            h.addr,
            "/v1/quotes/verify",
            &json!({"text": big, "sources": ["some source"]}).to_string(),
            "",
        );
        assert_eq!(r.status, 400, "{}", &r.text()[..200.min(r.text().len())]);
        let many: Vec<&str> = vec!["a"; 200];
        assert_eq!(
            post(
                h.addr,
                "/v1/quotes/verify",
                &json!({"text": "a", "sources": many}).to_string(),
                ""
            )
            .status,
            400
        );
        let src = "x ".repeat(crate::quotes::MAX_SOURCE_BYTES / 2 + 10);
        assert_eq!(
            post(
                h.addr,
                "/v1/quotes/verify",
                &json!({"text": "x", "sources": [src]}).to_string(),
                ""
            )
            .status,
            400
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_request_needing_too_much_kv_cache_is_refused() {
        let (h, d) = start(None, |c| c.max_kv_bytes = 64);
        let r = post(h.addr, "/v1/chat/completions", CHAT, "");
        assert_eq!(r.status, 400, "{}", r.text());
        assert!(
            r.json()["error"]["message"]
                .as_str()
                .unwrap()
                .contains("KV cache")
        );
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
        // n forks multiply the need: a budget that fits one choice must refuse eight
        let (h, d) = start(None, |c| c.max_kv_bytes = 40_000);
        let one = post(h.addr, "/v1/chat/completions", CHAT, "");
        assert_eq!(one.status, 200, "{}", one.text());
        let eight = post(
            h.addr,
            "/v1/chat/completions",
            &CHAT.replacen("\"max_tokens\":6", "\"max_tokens\":6,\"n\":8", 1),
            "",
        );
        assert_eq!(eight.status, 400, "{}", eight.text());
        h.stop();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn error_text_and_query_helpers_hide_paths_and_decode() {
        assert_eq!(
            redact_paths("cannot open /home/me/models/x.gguf: not found"),
            "cannot open <path>: not found"
        );
        assert_eq!(
            redact_paths("no `C:\\Users\\me\\m` here"),
            "no `<path>` here"
        );
        assert_eq!(
            redact_paths("plain message, no paths"),
            "plain message, no paths"
        );
        assert_eq!(pct_decode("a%2Fb+c%zz%4"), "a/b c%zz%4");
        let r = http::Request {
            query: "x=1&model=a%20b&y".into(),
            ..Default::default()
        };
        assert_eq!(query_param(&r, "model").as_deref(), Some("a b"));
        assert_eq!(query_param(&r, "nope"), None);
    }
}
