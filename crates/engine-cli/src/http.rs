//! A strict, bounded HTTP/1.1 request reader and the matching response writers.
//!
//! Everything a client controls is capped: head size, header count, body size, and the total time
//! allowed for the head and for the body (slowloris). Ambiguous framing — both `Content-Length` and
//! `Transfer-Encoding`, repeated or non-numeric lengths, stray control characters — is rejected, never
//! guessed at. The parsing functions are pure so they can be fuzzed without sockets.
use std::io::{self, Read, Write};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Limits {
    pub max_head: usize,
    pub max_headers: usize,
    pub max_body: usize,
    /// Total time allowed to receive the request line and headers.
    pub head_time: Duration,
    /// Total time allowed to receive the body.
    pub body_time: Duration,
}
impl Default for Limits {
    fn default() -> Limits {
        Limits { max_head: 16 * 1024, max_headers: 64, max_body: 4 << 20, head_time: Duration::from_secs(10), body_time: Duration::from_secs(30) }
    }
}

#[derive(Debug)]
pub enum HttpError {
    /// The client sent something invalid (400).
    Bad(&'static str),
    /// 431
    HeadTooLarge,
    /// 413
    BodyTooLarge,
    /// 501
    NotImplemented(&'static str),
    /// 505
    Version,
    /// 408
    Timeout,
    /// The connection ended before a request began; nothing to answer.
    Closed,
    Io(io::Error),
}
impl HttpError {
    pub fn status(&self) -> u16 {
        match self { HttpError::Bad(_) => 400, HttpError::HeadTooLarge => 431, HttpError::BodyTooLarge => 413, HttpError::NotImplemented(_) => 501,
            HttpError::Version => 505, HttpError::Timeout => 408, HttpError::Closed | HttpError::Io(_) => 400 }
    }
    pub fn message(&self) -> String {
        match self {
            HttpError::Bad(m) | HttpError::NotImplemented(m) => (*m).to_string(),
            HttpError::HeadTooLarge => "request headers are too large".into(),
            HttpError::BodyTooLarge => "request body is too large".into(),
            HttpError::Version => "only HTTP/1.0 and HTTP/1.1 are supported".into(),
            HttpError::Timeout => "the request took too long to arrive".into(),
            HttpError::Closed => "connection closed".into(),
            HttpError::Io(e) => e.to_string(),
        }
    }
    /// Is there anyone left to send an error response to?
    pub fn answerable(&self) -> bool { !matches!(self, HttpError::Closed | HttpError::Io(_)) }
}

#[derive(Debug, Default, Clone)]
pub struct Request {
    pub method: String,
    /// Path without the query string.
    pub path: String,
    pub query: String,
    /// Lower-case names.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}
impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

fn is_token(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}

/// Parse the request line and headers (no body). `head` excludes the final blank line.
pub fn parse_head(head: &[u8], lim: &Limits) -> Result<Request, HttpError> {
    let text = std::str::from_utf8(head).map_err(|_| HttpError::Bad("the request head is not valid UTF-8"))?;
    if text.bytes().any(|b| (b < 0x20 && b != b'\r' && b != b'\n' && b != b'\t') || b == 0x7f) { return Err(HttpError::Bad("control character in the request head")); }
    let mut lines = text.split("\r\n");
    let first = lines.next().unwrap_or("");
    if first.contains('\n') || first.contains('\r') { return Err(HttpError::Bad("bare line break in the request line")); }
    let mut parts = first.split(' ');
    let (Some(method), Some(target), Some(version), None) = (parts.next(), parts.next(), parts.next(), parts.next()) else { return Err(HttpError::Bad("malformed request line")) };
    if !is_token(method) || method.len() > 16 { return Err(HttpError::Bad("bad method")); }
    if !target.starts_with('/') || target.len() > 2048 { return Err(HttpError::Bad("bad request target")); }
    match version { "HTTP/1.1" | "HTTP/1.0" => {}, v if v.starts_with("HTTP/") => return Err(HttpError::Version), _ => return Err(HttpError::Bad("malformed request line")) }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let mut headers = vec![];
    for line in lines {
        if line.is_empty() { return Err(HttpError::Bad("blank line inside the headers")); }
        if line.contains('\n') || line.contains('\r') { return Err(HttpError::Bad("bare line break in a header")); }
        if line.starts_with(' ') || line.starts_with('\t') { return Err(HttpError::Bad("folded headers are not accepted")); }
        let Some((name, value)) = line.split_once(':') else { return Err(HttpError::Bad("header without a colon")) };
        if !is_token(name) { return Err(HttpError::Bad("bad header name")); }
        if headers.len() >= lim.max_headers { return Err(HttpError::HeadTooLarge); }
        headers.push((name.to_ascii_lowercase(), value.trim_matches([' ', '\t']).to_string()));
    }
    Ok(Request { method: method.to_string(), path: path.to_string(), query: query.to_string(), headers, body: vec![] })
}

/// How the body is framed.
#[derive(Debug, PartialEq, Eq)]
pub enum Framing { None, Length(usize), Chunked }

pub fn framing(req: &Request, lim: &Limits) -> Result<Framing, HttpError> {
    let te: Vec<&str> = req.headers.iter().filter(|(k, _)| k == "transfer-encoding").map(|(_, v)| v.as_str()).collect();
    let cl: Vec<&str> = req.headers.iter().filter(|(k, _)| k == "content-length").map(|(_, v)| v.as_str()).collect();
    if !te.is_empty() {
        if !cl.is_empty() { return Err(HttpError::Bad("both Content-Length and Transfer-Encoding")); }
        if te.len() != 1 || !te[0].eq_ignore_ascii_case("chunked") { return Err(HttpError::NotImplemented("only chunked transfer encoding is supported")); }
        return Ok(Framing::Chunked);
    }
    match cl.as_slice() {
        [] => Ok(Framing::None),
        [one] => {
            if one.is_empty() || one.len() > 12 || !one.bytes().all(|b| b.is_ascii_digit()) { return Err(HttpError::Bad("bad Content-Length")); }
            let n: usize = one.parse().map_err(|_| HttpError::Bad("bad Content-Length"))?;
            if n > lim.max_body { return Err(HttpError::BodyTooLarge); }
            Ok(if n == 0 { Framing::None } else { Framing::Length(n) })
        }
        _ => Err(HttpError::Bad("repeated Content-Length")),
    }
}

/// Try to decode a chunked body from the bytes received so far.
/// `Ok(None)` means more bytes are needed; `Ok(Some((body, used)))` is a complete body.
pub fn decode_chunked(buf: &[u8], max_body: usize) -> Result<Option<(Vec<u8>, usize)>, HttpError> {
    let (mut pos, mut body) = (0usize, Vec::new());
    loop {
        let Some(eol) = find_crlf(&buf[pos..]) else {
            if buf.len() - pos > 300 { return Err(HttpError::Bad("chunk size line too long")); }
            return Ok(None);
        };
        let line = &buf[pos..pos + eol];
        let size_part = line.split(|&b| b == b';').next().unwrap_or(&[]);
        if size_part.is_empty() || size_part.len() > 8 || !size_part.iter().all(u8::is_ascii_hexdigit) { return Err(HttpError::Bad("bad chunk size")); }
        let size = usize::from_str_radix(std::str::from_utf8(size_part).unwrap_or("0"), 16).map_err(|_| HttpError::Bad("bad chunk size"))?;
        pos += eol + 2;
        if size == 0 {
            // trailers until a blank line
            let start = pos;
            loop {
                let Some(e) = find_crlf(&buf[pos..]) else {
                    if buf.len() - start > 8192 { return Err(HttpError::Bad("trailers too long")); }
                    return Ok(None);
                };
                if e == 0 { return Ok(Some((body, pos + 2))); }
                pos += e + 2;
                if pos - start > 8192 { return Err(HttpError::Bad("trailers too long")); }
            }
        }
        if body.len().saturating_add(size) > max_body { return Err(HttpError::BodyTooLarge); }
        if buf.len() < pos + size + 2 { return Ok(None); }
        body.extend_from_slice(&buf[pos..pos + size]);
        if &buf[pos + size..pos + size + 2] != b"\r\n" { return Err(HttpError::Bad("chunk not terminated by CRLF")); }
        pos += size + 2;
    }
}

fn find_crlf(b: &[u8]) -> Option<usize> { b.windows(2).position(|w| w == b"\r\n") }

fn is_timeout(e: &io::Error) -> bool { matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) }

/// Read one complete request. `on_continue` is called when the client asked for `100-continue`.
/// The reader should have a short read timeout (a second or two) so the overall deadlines can be checked.
pub fn read_request<R: Read>(r: &mut R, lim: &Limits, on_continue: &mut dyn FnMut()) -> Result<Request, HttpError> {
    let t0 = Instant::now();
    let mut buf: Vec<u8> = Vec::with_capacity(2048);
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") { break p; }
        if buf.len() > lim.max_head { return Err(HttpError::HeadTooLarge); }
        if t0.elapsed() > lim.head_time { return Err(HttpError::Timeout); }
        match r.read(&mut chunk) {
            Ok(0) => return Err(if buf.is_empty() { HttpError::Closed } else { HttpError::Bad("the request ended inside its head") }),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) if is_timeout(&e) => { if buf.is_empty() && t0.elapsed() > lim.head_time { return Err(HttpError::Closed); } }
            Err(e) => return Err(HttpError::Io(e)),
        }
    };
    if head_end > lim.max_head { return Err(HttpError::HeadTooLarge); }
    let mut req = parse_head(&buf[..head_end], lim)?;
    let mut rest: Vec<u8> = buf[head_end + 4..].to_vec();
    let fr = framing(&req, lim)?;
    if fr != Framing::None && req.header("expect").is_some_and(|e| e.eq_ignore_ascii_case("100-continue")) && rest.is_empty() { on_continue(); }
    let t1 = Instant::now();
    match fr {
        Framing::None => {}
        Framing::Length(n) => {
            while rest.len() < n {
                if t1.elapsed() > lim.body_time { return Err(HttpError::Timeout); }
                let want = (n - rest.len()).min(chunk.len());
                match r.read(&mut chunk[..want]) {
                    Ok(0) => return Err(HttpError::Bad("the request body ended early")),
                    Ok(k) => rest.extend_from_slice(&chunk[..k]),
                    Err(e) if e.kind() == io::ErrorKind::Interrupted || is_timeout(&e) => {}
                    Err(e) => return Err(HttpError::Io(e)),
                }
            }
            rest.truncate(n);
            req.body = rest;
        }
        Framing::Chunked => loop {
            if let Some((body, _)) = decode_chunked(&rest, lim.max_body)? { req.body = body; break; }
            if rest.len() > lim.max_body.saturating_mul(2).saturating_add(8192) { return Err(HttpError::BodyTooLarge); }
            if t1.elapsed() > lim.body_time { return Err(HttpError::Timeout); }
            match r.read(&mut chunk) {
                Ok(0) => return Err(HttpError::Bad("the chunked body ended early")),
                Ok(k) => rest.extend_from_slice(&chunk[..k]),
                Err(e) if e.kind() == io::ErrorKind::Interrupted || is_timeout(&e) => {}
                Err(e) => return Err(HttpError::Io(e)),
            }
        },
    }
    Ok(req)
}

pub fn reason(code: u16) -> &'static str {
    match code {
        100 => "Continue", 200 => "OK", 204 => "No Content", 400 => "Bad Request", 401 => "Unauthorized", 403 => "Forbidden", 404 => "Not Found",
        405 => "Method Not Allowed", 408 => "Request Timeout", 413 => "Payload Too Large", 422 => "Unprocessable Entity", 431 => "Request Header Fields Too Large",
        500 => "Internal Server Error", 501 => "Not Implemented", 503 => "Service Unavailable", 505 => "HTTP Version Not Supported", _ => "Status",
    }
}

fn clean(v: &str) -> String { v.chars().filter(|c| !c.is_control()).collect() }

/// A complete, fixed-length response; the connection closes afterwards.
pub fn write_response(w: &mut dyn Write, code: u16, content_type: &str, body: &[u8], extra: &[(&str, &str)]) -> io::Result<()> {
    let mut head = format!("HTTP/1.1 {code} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n",
        reason(code), clean(content_type), body.len());
    for (k, v) in extra { head.push_str(&format!("{}: {}\r\n", clean(k), clean(v))); }
    head.push_str("\r\n");
    w.write_all(head.as_bytes())?;
    w.write_all(body)?;
    w.flush()
}

pub fn write_json(w: &mut dyn Write, code: u16, v: &serde_json::Value, extra: &[(&str, &str)]) -> io::Result<()> {
    write_response(w, code, "application/json", v.to_string().as_bytes(), extra)
}

/// Start a streaming response using chunked transfer encoding (server-sent events, NDJSON).
pub fn start_chunked(w: &mut dyn Write, code: u16, content_type: &str, extra: &[(&str, &str)]) -> io::Result<()> {
    let mut head = format!("HTTP/1.1 {code} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nCache-Control: no-store\r\nX-Accel-Buffering: no\r\n",
        reason(code), clean(content_type));
    for (k, v) in extra { head.push_str(&format!("{}: {}\r\n", clean(k), clean(v))); }
    head.push_str("\r\n");
    w.write_all(head.as_bytes())?;
    w.flush()
}
pub fn write_chunk(w: &mut dyn Write, data: &[u8]) -> io::Result<()> {
    if data.is_empty() { return Ok(()); }
    w.write_all(format!("{:x}\r\n", data.len()).as_bytes())?;
    w.write_all(data)?;
    w.write_all(b"\r\n")?;
    w.flush()
}
pub fn end_chunked(w: &mut dyn Write) -> io::Result<()> {
    w.write_all(b"0\r\n\r\n")?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn read(bytes: &[u8]) -> Result<Request, HttpError> { read_request(&mut Cursor::new(bytes.to_vec()), &Limits::default(), &mut || {}) }

    #[test]
    fn a_normal_post_is_parsed() {
        let r = read(b"POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-Length: 7\r\nAuthorization:   Bearer abc \r\n\r\n{\"a\":1}").unwrap();
        assert_eq!((r.method.as_str(), r.path.as_str(), r.query.as_str()), ("POST", "/v1/chat/completions", "x=1"));
        assert_eq!(r.body, b"{\"a\":1}");
        assert_eq!(r.header("authorization"), Some("Bearer abc"));
        assert_eq!(r.header("host"), Some("localhost"));
    }

    #[test]
    fn chunked_bodies_are_decoded_whole_or_in_pieces() {
        let raw = b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nWiki\r\n5;ext=1\r\npedia\r\n0\r\nTrailer: v\r\n\r\n";
        assert_eq!(read(raw).unwrap().body, b"Wikipedia");
        // one byte at a time
        struct Drip(Vec<u8>, usize);
        impl Read for Drip { fn read(&mut self, b: &mut [u8]) -> io::Result<usize> { if self.1 >= self.0.len() { return Ok(0); } b[0] = self.0[self.1]; self.1 += 1; Ok(1) } }
        let r = read_request(&mut Drip(raw.to_vec(), 0), &Limits::default(), &mut || {}).unwrap();
        assert_eq!(r.body, b"Wikipedia");
        assert_eq!(decode_chunked(b"4\r\nWi", 100).unwrap(), None);
    }

    #[test]
    fn ambiguous_or_hostile_framing_is_rejected() {
        let bad: Vec<&[u8]> = vec![
            b"POST / HTTP/1.1\r\nContent-Length: 3\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
            b"POST / HTTP/1.1\r\nContent-Length: 3\r\nContent-Length: 4\r\n\r\nabcd",
            b"POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n",
            b"POST / HTTP/1.1\r\nContent-Length: 0x10\r\n\r\n",
            b"POST / HTTP/1.1\r\nContent-Length: 99999999999999999999\r\n\r\n",
            b"POST / HTTP/1.1\r\nTransfer-Encoding: gzip, chunked\r\n\r\n",
            b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nZZ\r\n",
            b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabcXX0\r\n\r\n",
            b"GET / HTTP/1.1\r\n Folded: x\r\n\r\n",
            b"GET / HTTP/1.1\r\nBad Name: x\r\n\r\n",
            b"GET / HTTP/1.1\r\nNoColon\r\n\r\n",
            b"GET / HTTP/1.1\r\nX: a\x00b\r\n\r\n",
            b"GET / HTTP/1.1\r\nX: a\rb\r\n\r\n",
            b"GET  / HTTP/1.1\r\n\r\n",
            b"GET / HTTP/1.1 extra\r\n\r\n",
            b"GET http://evil/ HTTP/1.1\r\n\r\n",
            b"G\x01T / HTTP/1.1\r\n\r\n",
            b"GET / FTP/1.1\r\n\r\n",
            b"\xff\xfe / HTTP/1.1\r\n\r\n",
        ];
        for b in bad { assert!(read(b).is_err(), "{:?}", String::from_utf8_lossy(b)); }
        assert!(matches!(read(b"GET / HTTP/2.0\r\n\r\n"), Err(HttpError::Version)));
    }

    #[test]
    fn size_limits_hold() {
        let big_head = format!("GET / HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(40_000));
        assert!(matches!(read(big_head.as_bytes()), Err(HttpError::HeadTooLarge)));
        let mut many = String::from("GET / HTTP/1.1\r\n");
        for i in 0..200 { many.push_str(&format!("H{i}: v\r\n")); }
        many.push_str("\r\n");
        assert!(matches!(read(many.as_bytes()), Err(HttpError::HeadTooLarge)));
        assert!(matches!(read(b"POST / HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n"), Err(HttpError::BodyTooLarge)));
        let lim = Limits { max_body: 8, ..Limits::default() };
        let r = read_request(&mut Cursor::new(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nabcde\r\n5\r\nabcde\r\n0\r\n\r\n".to_vec()), &lim, &mut || {});
        assert!(matches!(r, Err(HttpError::BodyTooLarge)));
        assert!(matches!(read(b""), Err(HttpError::Closed)));
        assert!(read(b"GET / HTTP/1.1\r\nHost: x").is_err());
        assert!(read(b"POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc").is_err());
    }

    #[test]
    fn a_stalled_client_times_out() {
        struct Stall;
        impl Read for Stall { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { std::thread::sleep(Duration::from_millis(5)); Err(io::ErrorKind::WouldBlock.into()) } }
        let lim = Limits { head_time: Duration::from_millis(40), ..Limits::default() };
        let r = read_request(&mut Stall, &lim, &mut || {});
        assert!(matches!(r, Err(HttpError::Closed) | Err(HttpError::Timeout)));
        // a slow trickle that never finishes the head
        struct Trickle(usize);
        impl Read for Trickle { fn read(&mut self, b: &mut [u8]) -> io::Result<usize> { std::thread::sleep(Duration::from_millis(5)); b[0] = b"GET / HTTP/1.1\r\nX: "[self.0 % 19]; self.0 += 1; Ok(1) } }
        assert!(matches!(read_request(&mut Trickle(0), &lim, &mut || {}), Err(HttpError::Timeout) | Err(HttpError::HeadTooLarge)));
    }

    #[test]
    fn expect_continue_is_acknowledged() {
        let mut called = false;
        let r = read_request(&mut Cursor::new(b"POST / HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\nhi".to_vec()), &Limits::default(), &mut || called = true);
        assert!(r.is_ok());
        // the body had already arrived, so no interim response is needed
        assert!(!called);
    }

    #[test]
    fn fuzzed_requests_never_panic_and_stay_bounded() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let seeds: Vec<&[u8]> = vec![
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\n\r\nhello",
            b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nWiki\r\n0\r\n\r\n",
            b"GET /health HTTP/1.0\r\nAuthorization: Bearer t\r\n\r\n",
        ];
        let lim = Limits { max_body: 1 << 16, ..Limits::default() };
        for round in 0..6000 {
            let mut b = seeds[round % seeds.len()].to_vec();
            for _ in 0..(next() % 5 + 1) {
                if b.is_empty() { break; }
                let at = (next() as usize) % b.len();
                match next() % 4 { 0 => { b.remove(at); } 1 => b.insert(at, next() as u8), 2 => b[at] = next() as u8, _ => { let n = (next() % 50) as usize; let seg: Vec<u8> = b.iter().cycle().skip(at).take(n).cloned().collect(); b.splice(at..at, seg); } }
            }
            let _ = read_request(&mut Cursor::new(b), &lim, &mut || {});
        }
        for _ in 0..3000 {
            let n = (next() % 300) as usize;
            let b: Vec<u8> = (0..n).map(|_| b"0123456789abcdefABCDEF;\r\n xyz"[(next() % 29) as usize]).collect();
            let _ = decode_chunked(&b, 1 << 12);
            let _ = parse_head(&b, &lim);
        }
    }

    #[test]
    fn responses_are_well_formed_and_scrubbed() {
        let mut out = vec![];
        write_response(&mut out, 200, "text/plain", b"hi", &[("X-A", "b\r\nInjected: 1")]).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.starts_with("HTTP/1.1 200 OK\r\n") && s.contains("Content-Length: 2\r\n") && s.ends_with("\r\n\r\nhi"));
        assert!(!s.contains("\r\nInjected: 1"));
        let mut out = vec![];
        start_chunked(&mut out, 200, "text/event-stream", &[]).unwrap();
        write_chunk(&mut out, b"data: 1\n\n").unwrap();
        end_chunked(&mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.ends_with("9\r\ndata: 1\n\n\r\n0\r\n\r\n"));
    }
}
