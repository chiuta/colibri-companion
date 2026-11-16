//! A deliberately tiny HTTP/1.1 client for `http://` servers on the local machine.
//!
//! Why not a library: colibri's gateway streams answers as close-framed
//! server-sent events, and the interface must be able to stop one *instantly*
//! (closing the socket is what tells the server to stop generating). Owning the
//! socket makes that a one-liner. It also keeps the dependency list, and with it
//! the list of licences to carry, short. HTTPS is not supported on purpose.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Lets another thread cut a request short: sets a flag and closes the socket,
/// which makes a blocked read return at once.
#[derive(Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
    socket: Arc<Mutex<Option<TcpStream>>>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
        if let Ok(guard) = self.socket.lock() {
            if let Some(s) = guard.as_ref() {
                let _ = s.shutdown(Shutdown::Both);
            }
        }
    }
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
    fn attach(&self, stream: &TcpStream) {
        if let Ok(mut g) = self.socket.lock() {
            *g = stream.try_clone().ok();
        }
        if self.is_cancelled() {
            let _ = stream.shutdown(Shutdown::Both);
        }
    }
}

/// A socket whose reads wake up regularly to look at the cancel flag.
///
/// Shutting a socket down from another thread is not enough on Windows: a read that is
/// already waiting may keep waiting. Polling every 150 ms works the same everywhere, and
/// `cancel()` still closes the socket so the server notices and stops.
pub struct PollingStream {
    inner: TcpStream,
    cancel: Option<CancelToken>,
    /// The caller's own timeout (`None`: wait for as long as it takes).
    limit: Option<Duration>,
}

const POLL: Duration = Duration::from_millis(150);

impl Read for PollingStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let started = std::time::Instant::now();
        loop {
            if self.cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
                return Ok(0);
            }
            match self.inner.read(buf) {
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) =>
                {
                    if self.cancel.is_none() || self.limit.is_some_and(|l| started.elapsed() >= l) {
                        return Err(e);
                    }
                }
                other => return other,
            }
        }
    }
}

/// Longest status, header or chunk-size line accepted, and most headers in one response. A server
/// that never ends a line cannot make the program grow without limit.
const MAX_LINE: u64 = 16 * 1024;
const MAX_HEADERS: usize = 200;
/// Most bytes `Response::text` keeps (health checks and model lists are tiny).
/// Longest line of a streamed body (one server-sent event); real ones are a few kilobytes.
const MAX_BODY_LINE: u64 = 32 * 1024 * 1024;
const MAX_TEXT: u64 = 64 * 1024 * 1024;

/// `read_line`, but a line longer than [`MAX_LINE`] is an error, and bytes that are not UTF-8
/// (a header in Latin-1 from some server) are replaced instead of failing the whole response.
fn read_line_capped<R: BufRead>(r: &mut R, line: &mut String) -> io::Result<usize> {
    let mut raw = Vec::new();
    let n = r.by_ref().take(MAX_LINE + 1).read_until(b'\n', &mut raw)?;
    if n as u64 > MAX_LINE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "a line is too long",
        ));
    }
    line.push_str(&String::from_utf8_lossy(&raw));
    Ok(n)
}

/// `http://host:port` -> (host, port). Only plain HTTP.
pub fn parse_authority(base: &str) -> Result<(String, u16), String> {
    if base.starts_with("https://") {
        return Err(
            "https:// is not supported here; colibri serves plain http:// on this computer".into(),
        );
    }
    let rest = base.strip_prefix("http://").unwrap_or(base);
    let authority = rest.split('/').next().unwrap_or("");
    if authority.is_empty() {
        return Err("the server address is empty".into());
    }
    // [::1]:8000, host:8000, host
    if let Some(stripped) = authority.strip_prefix('[') {
        let (host, tail) = stripped.split_once(']').ok_or("bad IPv6 address")?;
        let port = tail
            .strip_prefix(':')
            .map(|p| p.parse().map_err(|_| "bad port"))
            .transpose()?
            .unwrap_or(80);
        return Ok((host.to_string(), port));
    }
    match authority.rsplit_once(':') {
        Some((h, p)) => Ok((
            h.to_string(),
            p.parse().map_err(|_| format!("bad port: {p}"))?,
        )),
        None => Ok((authority.to_string(), 80)),
    }
}

enum Framing {
    Chunked { left: usize, done: bool },
    Length(u64),
    UntilClose,
}

/// The decoded body of a response (chunked and length framing removed).
pub struct Body<R: Read> {
    reader: BufReader<R>,
    framing: Framing,
}

impl<R: Read> Read for Body<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        match &mut self.framing {
            Framing::UntilClose => self.reader.read(out),
            Framing::Length(n) => {
                if *n == 0 {
                    return Ok(0);
                }
                let max = out.len().min(usize::try_from(*n).unwrap_or(usize::MAX));
                let got = self.reader.read(&mut out[..max])?;
                *n -= got as u64;
                Ok(got)
            }
            Framing::Chunked { left, done } => loop {
                if *done {
                    return Ok(0);
                }
                if *left == 0 {
                    let mut line = String::new();
                    if read_line_capped(&mut self.reader, &mut line)? == 0 {
                        *done = true;
                        return Ok(0);
                    }
                    let t = line.trim();
                    if t.is_empty() {
                        continue; // the CRLF that ends the previous chunk
                    }
                    let size_text = t.split(';').next().unwrap_or("").trim();
                    let size = usize::from_str_radix(size_text, 16).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "bad chunk size")
                    })?;
                    if size == 0 {
                        loop {
                            line.clear();
                            if read_line_capped(&mut self.reader, &mut line)? == 0
                                || line.trim().is_empty()
                            {
                                break;
                            }
                        }
                        *done = true;
                        return Ok(0);
                    }
                    *left = size;
                }
                let max = out.len().min(*left);
                let got = self.reader.read(&mut out[..max])?;
                if got == 0 {
                    *done = true;
                    return Ok(0);
                }
                *left -= got;
                return Ok(got);
            },
        }
    }
}

impl<R: Read> Body<R> {
    /// One line (without its newline); `None` at the end of the body.
    pub fn read_line_lossy(&mut self) -> io::Result<Option<String>> {
        // Byte by byte (the socket is already buffered underneath), so a line that is
        // cut by an error or a closed connection is never half-lost.
        let mut buf = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            match self.read(&mut byte) {
                Ok(0) => {
                    return Ok(if buf.is_empty() {
                        None
                    } else {
                        Some(String::from_utf8_lossy(&buf).into_owned())
                    });
                }
                Ok(_) => {
                    if buf.len() as u64 > MAX_BODY_LINE {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "a line is too long",
                        ));
                    }
                    if byte[0] == b'\n' {
                        if buf.last() == Some(&b'\r') {
                            buf.pop();
                        }
                        return Ok(Some(String::from_utf8_lossy(&buf).into_owned()));
                    }
                    buf.push(byte[0]);
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
    }
}

pub struct Response<R: Read> {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Body<R>,
}

impl<R: Read> Response<R> {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    pub fn text(mut self) -> String {
        let mut bytes = Vec::new();
        let _ = (&mut self.body).take(MAX_TEXT).read_to_end(&mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Read the status line and headers, and set up body decoding.
pub fn read_response<R: Read>(reader: R) -> io::Result<Response<R>> {
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    read_line_capped(&mut reader, &mut line)?;
    let mut parts = line.split_whitespace();
    let version = parts.next().unwrap_or("");
    let status: u16 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .filter(|_| version.starts_with("HTTP/"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "that is not an HTTP server"))?;
    let mut headers = Vec::new();
    loop {
        line.clear();
        if read_line_capped(&mut reader, &mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        if headers.len() >= MAX_HEADERS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "too many headers",
            ));
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    let get = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.to_lowercase())
    };
    let framing = if get("transfer-encoding").is_some_and(|v| v.contains("chunked")) {
        Framing::Chunked {
            left: 0,
            done: false,
        }
    } else if let Some(n) = get("content-length").and_then(|v| v.parse::<u64>().ok()) {
        Framing::Length(n)
    } else if status == 204 || status == 304 || (100..200).contains(&status) {
        Framing::Length(0)
    } else {
        Framing::UntilClose
    };
    Ok(Response {
        status,
        headers,
        body: Body { reader, framing },
    })
}

pub struct RequestOptions<'a> {
    pub connect_timeout: Duration,
    /// `None`: wait as long as it takes (a big model may think for minutes).
    pub read_timeout: Option<Duration>,
    pub bearer: Option<&'a str>,
    pub cancel: Option<&'a CancelToken>,
}

impl Default for RequestOptions<'_> {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(3),
            read_timeout: Some(Duration::from_secs(8)),
            bearer: None,
            cancel: None,
        }
    }
}

/// Send one request (`Connection: close`) and read the response head.
pub fn send(
    base: &str,
    method: &str,
    path: &str,
    json_body: Option<&str>,
    opts: &RequestOptions<'_>,
) -> io::Result<Response<PollingStream>> {
    let (host, port) =
        parse_authority(base).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    // Nothing that ends a header line or the request may reach the wire: a key pasted with a
    // trailing newline would otherwise break the request, or add headers of its own.
    let bearer = opts.bearer.map(str::trim);
    let clean = |t: &str| !t.chars().any(|c| c.is_control());
    if !clean(method)
        || !clean(path)
        || path.contains(' ')
        || !path.starts_with('/')
        || !clean(&host)
        || bearer.is_some_and(|k| !clean(k))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the request contains a control character",
        ));
    }
    let mut last_err = io::Error::new(io::ErrorKind::NotFound, "no address");
    let mut stream = None;
    for addr in (host.as_str(), port).to_socket_addrs()? {
        match TcpStream::connect_timeout(&addr, opts.connect_timeout) {
            Ok(s) => {
                crate::net::record(addr, true);
                stream = Some(s);
                break;
            }
            Err(e) => {
                crate::net::record(addr, false);
                last_err = e;
            }
        }
    }
    let mut stream = stream.ok_or(last_err)?;
    let poll = opts.cancel.is_some();
    stream.set_read_timeout(if poll {
        Some(opts.read_timeout.map_or(POLL, |l| l.min(POLL)))
    } else {
        opts.read_timeout
    })?;
    stream.set_nodelay(true).ok();
    if let Some(token) = opts.cancel {
        token.attach(&stream);
    }
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUser-Agent: colibri-companion\r\nAccept: */*\r\nConnection: close\r\n"
    );
    if let Some(k) = bearer.filter(|k| !k.is_empty()) {
        head.push_str(&format!("Authorization: Bearer {k}\r\n"));
    }
    if let Some(body) = json_body {
        head.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            body.len()
        ));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    if let Some(body) = json_body {
        stream.write_all(body.as_bytes())?;
    }
    stream.flush()?;
    read_response(PollingStream {
        inner: stream,
        cancel: opts.cancel.cloned(),
        limit: opts.read_timeout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn body_of(raw: &str) -> (u16, String) {
        let r = read_response(Cursor::new(raw.as_bytes().to_vec())).unwrap();
        let status = r.status;
        (status, r.text())
    }

    #[test]
    fn content_length_chunked_and_close_framing() {
        assert_eq!(
            body_of("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhelloEXTRA"),
            (200, "hello".into())
        );
        assert_eq!(
            body_of(
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n7;ext=1\r\n, world\r\n0\r\nX-T: 1\r\n\r\n"
            ),
            (200, "hello, world".into())
        );
        assert_eq!(
            body_of("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\ndata: 1\n\n"),
            (200, "data: 1\n\n".into())
        );
        assert_eq!(
            body_of("HTTP/1.1 503 Busy\r\nContent-Length: 0\r\n\r\n"),
            (503, "".into())
        );
    }

    #[test]
    fn rejects_non_http() {
        assert!(read_response(Cursor::new(b"SSH-2.0-OpenSSH\r\n".to_vec())).is_err());
    }

    #[test]
    fn reads_lines_across_chunk_boundaries() {
        let raw = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n6\r\ndata: \r\n9\r\nabc\r\n\r\nxy\r\n0\r\n\r\n";
        let mut r = read_response(Cursor::new(raw.as_bytes().to_vec())).unwrap();
        assert_eq!(
            r.body.read_line_lossy().unwrap().as_deref(),
            Some("data: abc")
        );
        assert_eq!(r.body.read_line_lossy().unwrap().as_deref(), Some(""));
        assert_eq!(r.body.read_line_lossy().unwrap().as_deref(), Some("xy"));
        assert_eq!(r.body.read_line_lossy().unwrap(), None);
    }

    #[test]
    fn authorities() {
        assert_eq!(
            parse_authority("http://127.0.0.1:8000").unwrap(),
            ("127.0.0.1".into(), 8000)
        );
        assert_eq!(
            parse_authority("localhost").unwrap(),
            ("localhost".into(), 80)
        );
        assert_eq!(
            parse_authority("http://[::1]:9/x").unwrap(),
            ("::1".into(), 9)
        );
        assert!(parse_authority("https://example.com").is_err());
        assert!(parse_authority("http://h:notaport").is_err());
    }

    #[test]
    fn cancel_unblocks_a_waiting_read() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 512];
            let _ = s.read(&mut buf);
            s.write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\ndata: first\n")
                .unwrap();
            std::thread::sleep(Duration::from_secs(5)); // never sends more
        });
        let token = CancelToken::new();
        let opts = RequestOptions {
            read_timeout: None,
            cancel: Some(&token),
            ..Default::default()
        };
        let mut resp = send(&format!("http://{addr}"), "GET", "/", None, &opts).unwrap();
        assert_eq!(
            resp.body.read_line_lossy().unwrap().as_deref(),
            Some("data: first")
        );
        let t2 = token.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            t2.cancel();
        });
        let started = std::time::Instant::now();
        let end = resp.body.read_line_lossy();
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "cancel must not wait for the server"
        );
        assert!(matches!(end, Ok(None) | Err(_)));
        drop(server); // the thread ends on its own
    }

    #[test]
    fn endless_lines_and_header_floods_are_refused() {
        let long = format!("HTTP/1.1 200 OK\r\nX: {}\r\n\r\n", "a".repeat(20_000));
        assert!(read_response(Cursor::new(long.into_bytes())).is_err());
        let flood = format!("HTTP/1.1 200 OK\r\n{}\r\n", "X: 1\r\n".repeat(500));
        assert!(read_response(Cursor::new(flood.into_bytes())).is_err());
        let status = format!("HTTP/1.1 200 {}", "a".repeat(20_000));
        assert!(read_response(Cursor::new(status.into_bytes())).is_err());
    }

    #[test]
    fn a_key_with_a_newline_never_reaches_the_wire() {
        let opts = |k| RequestOptions {
            bearer: Some(k),
            ..Default::default()
        };
        for bad in ["abc\r\nX-Evil: 1", "ab\ncd"] {
            let e = send("http://127.0.0.1:1", "GET", "/v1/models", None, &opts(bad))
                .err()
                .unwrap();
            assert_eq!(e.kind(), io::ErrorKind::InvalidInput, "{bad:?}");
        }
        let e = send(
            "http://127.0.0.1:1",
            "GET",
            "/a\r\nHost: x",
            None,
            &opts("k"),
        )
        .err()
        .unwrap();
        assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
        // a trailing newline from a paste is trimmed, not an error: the failure is the closed port
        let e = send(
            "http://127.0.0.1:1",
            "GET",
            "/v1/models",
            None,
            &opts("key\n"),
        )
        .err()
        .unwrap();
        assert_ne!(e.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn a_header_that_is_not_utf8_does_not_break_the_response() {
        let mut raw = b"HTTP/1.1 200 OK\r\nX-Name: caf\xe9\r\nContent-Length: 2\r\n\r\nhi".to_vec();
        raw.extend_from_slice(b"");
        let r = read_response(Cursor::new(raw)).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.text(), "hi");
    }

    /// Reading a long server-sent stream line by line (`cargo test --release -p companion-core
    /// perf_ -- --ignored --nocapture`). Prints only.
    #[test]
    #[ignore]
    fn perf_reading_a_long_event_stream() {
        let line =
            "data: {\"choices\":[{\"delta\":{\"content\":\"some words of the answer \"}}]}\n\n";
        let text = line.repeat(150_000); // about 12 MB, around 150k tokens
        let mut body = Body {
            reader: BufReader::new(io::Cursor::new(text.as_bytes())),
            framing: Framing::UntilClose,
        };
        let t = std::time::Instant::now();
        let mut lines = 0u64;
        while let Some(l) = body.read_line_lossy().unwrap() {
            lines += l.len() as u64 & 1;
            lines += 1;
        }
        let secs = t.elapsed().as_secs_f64();
        println!(
            "perf_http read_line_lossy: {:.1} MB/s ({} lines in {:.1} ms; a fast model streams about 0.15 MB/s)",
            text.len() as f64 / 1e6 / secs,
            lines,
            secs * 1000.0
        );
    }
}
