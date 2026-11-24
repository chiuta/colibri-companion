//! Model downloader for Hugging Face and look-alike hubs, without a TLS stack of its own:
//! the transfer is delegated to `curl` (`curl.exe` ships with Windows 10+), with `wget` as a
//! fallback. What this module adds on top is everything around the transfer:
//!
//! * repo-spec parsing (`org/name@rev`, `hf://…`, full URLs),
//! * the file listing (tree API, or the model-info `siblings` as a fallback),
//! * choosing the right artefacts (safetensors shards, or one GGUF quant incl. split files),
//! * resumable downloads (`.part` then atomic rename), size and SHA-256 verification,
//! * path-traversal-safe destinations, cancellation, progress, and friendly error classes.
//!
//! The access token is never put on a command line: it goes into a short-lived 0600 header file.
//! It is sent by curl only: `wget` would carry it along redirects to other hosts, so with wget the
//! transfer is anonymous (public repositories still work).
//!
//! Every transfer has a ceiling: a per-file one (`--max-filesize`, and the running size is checked
//! while downloading, because a server can omit `Content-Length`) and a total one. Both have large
//! defaults and can be lowered with [`Downloader::with_max_filesize`] / [`Downloader::with_max_total`].
//! A file whose size the hub did not announce is *unknown*, not empty: it is refused unless the
//! caller set an explicit per-file cap (small metadata files get a 256 MiB cap of their own).
//! The revision is resolved to a commit SHA once, so listing and downloading see the same commit.
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

// ------------------------------------------------------------------------------- errors

/// Failure classes, so a front end can say something useful instead of "curl exited with 22".
#[derive(Debug, PartialEq, Eq)]
pub enum HubError {
    /// No network, DNS failure or timeout.
    Offline,
    /// 401/403: the model is gated (licence not accepted) or needs a token.
    Gated {
        status: u16,
    },
    NotFound,
    RateLimited,
    DiskFull,
    /// Certificate or handshake problem (clock, proxy, antivirus interception).
    Tls,
    Http(u16),
    Cancelled,
    /// Size or SHA-256 does not match what the hub announced.
    Integrity(String),
    BadSpec(String),
    UnsafePath(String),
    /// Neither curl nor wget could be started.
    NoTool,
    /// Plain http to a non-loopback host, or an endpoint that is not a URL.
    InsecureUrl(String),
    Io(String),
    Parse(String),
    /// The repo has neither safetensors nor GGUF weights.
    NoWeights,
    /// The transfer tool failed in a way we do not classify.
    Tool(String),
    /// A file or the whole download is larger than the allowed ceiling.
    TooLarge(String),
    /// The hub did not announce this file's size and the caller set no explicit cap.
    UnknownSize(String),
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HubError::Offline => {
                write!(f, "cannot reach the model hub (no network, DNS or timeout)")
            }
            HubError::Gated { status } => write!(
                f,
                "access refused ({status}): the model is gated — accept its licence on the hub and provide an access token (HF_TOKEN)"
            ),
            HubError::NotFound => write!(
                f,
                "the model or file was not found on the hub (check the name and revision)"
            ),
            HubError::RateLimited => write!(
                f,
                "the hub is rate-limiting this address (429); wait a few minutes or use a token"
            ),
            HubError::DiskFull => write!(f, "the disk is full"),
            HubError::Tls => write!(
                f,
                "the secure connection failed (certificate or handshake); check the clock, proxy and antivirus"
            ),
            HubError::Http(s) => write!(f, "the hub answered with HTTP {s}"),
            HubError::Cancelled => write!(f, "download cancelled"),
            HubError::Integrity(s) => write!(f, "integrity check failed: {s}"),
            HubError::BadSpec(s) => write!(f, "not a model reference: {s}"),
            HubError::UnsafePath(s) => write!(f, "refusing an unsafe file name from the hub: {s}"),
            HubError::NoTool => write!(f, "neither curl nor wget is installed"),
            HubError::InsecureUrl(s) => write!(f, "refusing a non-https address: {s}"),
            HubError::Io(s) => write!(f, "{s}"),
            HubError::Parse(s) => write!(f, "unexpected answer from the hub: {s}"),
            HubError::NoWeights => write!(f, "the repository has no safetensors or GGUF weights"),
            HubError::Tool(s) => write!(f, "the download tool failed: {s}"),
            HubError::TooLarge(s) => write!(
                f,
                "refusing a download larger than the allowed ceiling: {s}"
            ),
            HubError::UnknownSize(s) => write!(
                f,
                "the hub did not say how big {s} is; set an explicit size cap to download it anyway"
            ),
        }
    }
}
impl std::error::Error for HubError {}
impl From<std::io::Error> for HubError {
    fn from(e: std::io::Error) -> Self {
        if e.raw_os_error() == Some(28) || e.to_string().to_ascii_lowercase().contains("no space") {
            HubError::DiskFull
        } else {
            HubError::Io(e.to_string())
        }
    }
}

fn http_status_in(stderr: &str) -> Option<u16> {
    for marker in ["returned error: ", "ERROR "] {
        if let Some(i) = stderr.find(marker) {
            let digits: String = stderr[i + marker.len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            if let Ok(n) = digits.parse::<u16>()
                && (100..600).contains(&n)
            {
                return Some(n);
            }
        }
    }
    None
}

/// Map a tool exit code and its stderr to a [`HubError`].
pub fn classify(tool: &Tool, code: Option<i32>, stderr: &str) -> HubError {
    let low = stderr.to_ascii_lowercase();
    if low.contains("no space left") || low.contains("disk full") || low.contains("disk quota") {
        return HubError::DiskFull;
    }
    if let Some(s) = http_status_in(stderr) {
        return match s {
            401 | 403 => HubError::Gated { status: s },
            404 => HubError::NotFound,
            429 => HubError::RateLimited,
            o => HubError::Http(o),
        };
    }
    match (tool, code) {
        (Tool::Curl(_), Some(6 | 7 | 28 | 52 | 56)) => HubError::Offline,
        (
            Tool::Curl(_),
            Some(35 | 51 | 53 | 54 | 58 | 59 | 60 | 64 | 66 | 77 | 80 | 82 | 83 | 90 | 91),
        ) => HubError::Tls,
        (Tool::Curl(_), Some(23)) => HubError::DiskFull,
        (Tool::Wget(_), Some(4)) => HubError::Offline,
        (Tool::Wget(_), Some(5)) => HubError::Tls,
        (Tool::Wget(_), Some(3)) => HubError::Io("could not write the file".into()),
        _ => HubError::Tool(
            stderr
                .trim()
                .lines()
                .last()
                .unwrap_or("no message")
                .to_string(),
        ),
    }
}

// ----------------------------------------------------------------------------- SHA-256

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Streaming SHA-256.
pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    used: usize,
    len: u64,
}
impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}
impl Sha256 {
    pub fn new() -> Sha256 {
        Sha256 {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0; 64],
            used: 0,
            len: 0,
        }
    }
    fn block(h: &mut [u32; 8], b: &[u8]) {
        let mut w = [0u32; 64];
        for (i, c) in b.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut bb, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & bb) ^ (a & c) ^ (bb & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = bb;
            bb = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, bb, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    pub fn update(&mut self, mut data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u64);
        if self.used > 0 {
            let take = (64 - self.used).min(data.len());
            self.buf[self.used..self.used + take].copy_from_slice(&data[..take]);
            self.used += take;
            data = &data[take..];
            if self.used == 64 {
                let b = self.buf;
                Self::block(&mut self.h, &b);
                self.used = 0;
            }
        }
        let mut chunks = data.chunks_exact(64);
        for c in &mut chunks {
            Self::block(&mut self.h, c);
        }
        let rest = chunks.remainder();
        self.buf[..rest.len()].copy_from_slice(rest);
        self.used += rest.len();
    }
    pub fn finish(mut self) -> [u8; 32] {
        let bits = self.len.wrapping_mul(8);
        let mut pad = vec![0x80u8];
        while (self.used + pad.len()) % 64 != 56 {
            pad.push(0);
        }
        pad.extend(bits.to_be_bytes());
        let len = self.len;
        self.update(&pad);
        self.len = len;
        let mut out = [0u8; 32];
        for (o, v) in out.chunks_exact_mut(4).zip(self.h) {
            o.copy_from_slice(&v.to_be_bytes());
        }
        out
    }
}
pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
pub fn sha256_hex(data: &[u8]) -> String {
    let mut s = Sha256::new();
    s.update(data);
    hex(&s.finish())
}
pub fn sha256_file(p: &Path) -> Result<String, HubError> {
    let mut f = File::open(p)?;
    let mut s = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        s.update(&buf[..n]);
    }
    Ok(hex(&s.finish()))
}

// ------------------------------------------------------------------------ repo + hubs

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepoSpec {
    pub repo: String,
    pub revision: String,
}

fn valid_segment(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 96
        && s != "."
        && s != ".."
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}
fn valid_revision(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 200
        && !s.contains("..")
        && !s.starts_with('/')
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/'))
}

impl RepoSpec {
    /// Accepts `org/name`, `name`, `org/name@rev`, `hf://org/name@rev`, `hf://models/org/name`,
    /// and `https://<host>/org/name[/tree|blob|resolve/<rev>/…]`.
    pub fn parse(input: &str) -> Result<RepoSpec, HubError> {
        let bad = |why: &str| HubError::BadSpec(format!("{input:?}: {why}"));
        let s = input.trim();
        let s = s.split(['?', '#']).next().unwrap_or("");
        let (path, mut rev) = if let Some(rest) = s.strip_prefix("https://") {
            let (_, p) = rest
                .split_once('/')
                .ok_or_else(|| bad("no repository in the address"))?;
            (p.trim_matches('/').to_string(), None)
        } else if let Some(rest) = s.strip_prefix("hf://").or_else(|| s.strip_prefix("hf:")) {
            let r = rest.trim_matches('/');
            let r = r.strip_prefix("models/").unwrap_or(r);
            match r.rsplit_once('@') {
                Some((p, v)) => (p.to_string(), Some(v.to_string())),
                None => (r.to_string(), None),
            }
        } else if s.starts_with("http://") {
            return Err(bad("only https addresses are accepted"));
        } else {
            match s.rsplit_once('@') {
                Some((p, v)) => (p.trim_matches('/').to_string(), Some(v.to_string())),
                None => (s.trim_matches('/').to_string(), None),
            }
        };
        let mut parts: Vec<&str> = path.split('/').collect();
        if parts.first() == Some(&"models") && parts.len() > 1 {
            parts.remove(0);
        }
        let n = if parts.len() >= 2 { 2 } else { 1 };
        if parts.is_empty() || parts.iter().take(n).any(|p| !valid_segment(p)) {
            return Err(bad("expected org/name"));
        }
        let repo = parts[..n].join("/");
        if parts.len() > n {
            if matches!(parts[n], "tree" | "blob" | "resolve") && parts.len() > n + 1 {
                rev = Some(parts[n + 1].replace("%2F", "/").replace("%2f", "/"));
            } else if rev.is_none() && parts.len() > 2 {
                return Err(bad("unexpected extra path"));
            }
        }
        let revision = rev.unwrap_or_else(|| "main".into());
        if !valid_revision(&revision) {
            return Err(bad("bad revision"));
        }
        Ok(RepoSpec { repo, revision })
    }
}

fn enc(s: &str, keep_slash: bool) -> String {
    let mut o = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric()
            || matches!(b, b'-' | b'.' | b'_' | b'~')
            || (keep_slash && b == b'/')
        {
            o.push(b as char)
        } else {
            o.push_str(&format!("%{b:02X}"))
        }
    }
    o
}

/// Where to talk to. `HfCompatible` covers hf-mirror style mirrors and self-hosted hubs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Hub {
    HuggingFace,
    HfCompatible(String),
    ModelScope,
}

impl Hub {
    /// Honours `HF_ENDPOINT`.
    pub fn from_env() -> Hub {
        match std::env::var("HF_ENDPOINT") {
            Ok(e)
                if !e.trim().is_empty() && e.trim_end_matches('/') != "https://huggingface.co" =>
            {
                Hub::HfCompatible(e.trim().trim_end_matches('/').to_string())
            }
            _ => Hub::HuggingFace,
        }
    }
    fn base(&self) -> &str {
        match self {
            Hub::HuggingFace => "https://huggingface.co",
            Hub::HfCompatible(b) => b,
            Hub::ModelScope => "https://www.modelscope.cn",
        }
    }
    pub fn tree_url(&self, s: &RepoSpec) -> String {
        match self {
            Hub::ModelScope => format!(
                "{}/api/v1/models/{}/repo/files?Revision={}&Recursive=true",
                self.base(),
                enc(&s.repo, true),
                enc(&s.revision, false)
            ),
            _ => format!(
                "{}/api/models/{}/tree/{}?recursive=1",
                self.base(),
                enc(&s.repo, true),
                enc(&s.revision, false)
            ),
        }
    }
    pub fn info_url(&self, s: &RepoSpec) -> String {
        format!(
            "{}/api/models/{}/revision/{}",
            self.base(),
            enc(&s.repo, true),
            enc(&s.revision, false)
        )
    }
    pub fn file_url(&self, s: &RepoSpec, path: &str) -> String {
        match self {
            Hub::ModelScope => format!(
                "{}/api/v1/models/{}/repo?Revision={}&FilePath={}",
                self.base(),
                enc(&s.repo, true),
                enc(&s.revision, false),
                enc(path, false)
            ),
            _ => format!(
                "{}/{}/resolve/{}/{}",
                self.base(),
                enc(&s.repo, true),
                enc(&s.revision, false),
                enc(path, true)
            ),
        }
    }
}

// --------------------------------------------------------------------- file listings

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    pub path: String,
    /// 0 when the hub did not say.
    pub size: u64,
    /// Lower-case hex SHA-256 when the hub announces one (LFS files).
    pub sha256: Option<String>,
}

fn good_sha(s: &str) -> Option<String> {
    (s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())).then(|| s.to_ascii_lowercase())
}

/// Parse the answer of the tree API (Hugging Face style) or ModelScope's file list.
pub fn parse_tree(json: &str) -> Result<Vec<RemoteFile>, HubError> {
    let v: Value = serde_json::from_str(json).map_err(|e| HubError::Parse(e.to_string()))?;
    let mut out = vec![];
    if let Some(arr) = v.as_array() {
        for it in arr {
            if it.get("type").and_then(Value::as_str) != Some("file") {
                continue;
            }
            let path = it
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| HubError::Parse("entry without path".into()))?;
            let lfs = it.get("lfs");
            let size = lfs
                .and_then(|l| l.get("size"))
                .or_else(|| it.get("size"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let sha256 = lfs
                .and_then(|l| l.get("oid"))
                .and_then(Value::as_str)
                .and_then(|o| good_sha(o.strip_prefix("sha256:").unwrap_or(o)));
            out.push(RemoteFile {
                path: path.into(),
                size,
                sha256,
            });
        }
    } else if let Some(files) = v
        .get("Data")
        .and_then(|d| d.get("Files"))
        .and_then(Value::as_array)
    {
        for it in files {
            if it.get("Type").and_then(Value::as_str) != Some("blob") {
                continue;
            }
            let path = it
                .get("Path")
                .and_then(Value::as_str)
                .ok_or_else(|| HubError::Parse("entry without Path".into()))?;
            out.push(RemoteFile {
                path: path.into(),
                size: it.get("Size").and_then(Value::as_u64).unwrap_or(0),
                sha256: it.get("Sha256").and_then(Value::as_str).and_then(good_sha),
            });
        }
    } else {
        let why = v
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("not a file list");
        return Err(HubError::Parse(why.into()));
    }
    Ok(out)
}

/// Fallback: `siblings` from the model-info endpoint (names, usually without sizes).
pub fn parse_model_info(json: &str) -> Result<Vec<RemoteFile>, HubError> {
    let v: Value = serde_json::from_str(json).map_err(|e| HubError::Parse(e.to_string()))?;
    let sib = v
        .get("siblings")
        .and_then(Value::as_array)
        .ok_or_else(|| HubError::Parse("no siblings".into()))?;
    Ok(sib
        .iter()
        .filter_map(|s| {
            let path = s.get("rfilename")?.as_str()?.to_string();
            let lfs = s.get("lfs");
            Some(RemoteFile {
                path,
                size: lfs
                    .and_then(|l| l.get("size"))
                    .or_else(|| s.get("size"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                sha256: lfs
                    .and_then(|l| l.get("sha256"))
                    .and_then(Value::as_str)
                    .and_then(good_sha),
            })
        })
        .collect())
}

// ------------------------------------------------------------------------- selection

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Auto,
    Safetensors,
    Gguf,
}

#[derive(Clone, Debug)]
pub struct Select {
    pub format: Format,
    /// Quantisation tags in order of preference (matched case-insensitively against GGUF names).
    pub quants: Vec<String>,
}
impl Default for Select {
    fn default() -> Self {
        Select {
            format: Format::Auto,
            quants: [
                "Q4_K_M", "Q5_K_M", "Q8_0", "Q6_K", "Q4_K_S", "Q5_K_S", "Q4_0", "Q3_K_M", "IQ4_XS",
                "F16", "BF16",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        }
    }
}

const SMALL: [&str; 9] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
];

fn quant_in(upper_path: &str, tag: &str) -> bool {
    let tag = tag.to_ascii_uppercase();
    upper_path.match_indices(&tag).any(|(i, _)| {
        let before = upper_path[..i].chars().next_back();
        let after = upper_path[i + tag.len()..].chars().next();
        before.is_none_or(|c| !c.is_ascii_alphanumeric())
            && after.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_')
    })
}

/// `name-00002-of-00005.gguf` → (`name`, 2, 5); a single file is (`name`, 1, 1).
fn gguf_split(path: &str) -> (String, u32, u32) {
    let stem = &path[..path.len() - 5];
    let b = stem.as_bytes();
    if b.len() >= 15 && stem.is_char_boundary(stem.len() - 15) {
        let t = &stem[stem.len() - 15..];
        let tb = t.as_bytes();
        if tb[0] == b'-'
            && t.get(6..10) == Some("-of-")
            && tb[1..6].iter().all(|c| c.is_ascii_digit())
            && t[10..].bytes().all(|c| c.is_ascii_digit())
            && let (Ok(i), Ok(n)) = (t[1..6].parse(), t[10..].parse())
        {
            return (stem[..stem.len() - 15].to_string(), i, n);
        }
    }
    (stem.to_string(), 1, 1)
}

/// Pick what to download from a repo listing.
pub fn select_files(files: &[RemoteFile], sel: &Select) -> Result<Vec<RemoteFile>, HubError> {
    let lower = |f: &RemoteFile| f.path.to_ascii_lowercase();
    let root = |f: &&RemoteFile| !f.path.contains('/');
    let mut st: Vec<&RemoteFile> = files
        .iter()
        .filter(root)
        .filter(|f| lower(f).ends_with(".safetensors"))
        .collect();
    if st.iter().any(|f| f.path.starts_with("model")) {
        st.retain(|f| f.path.starts_with("model"));
    }
    let has_gguf = files
        .iter()
        .any(|f| lower(f).ends_with(".gguf") && !lower(f).contains("mmproj"));
    let fmt = match sel.format {
        Format::Auto if !st.is_empty() => Format::Safetensors,
        Format::Auto if has_gguf => Format::Gguf,
        Format::Auto => return Err(HubError::NoWeights),
        f => f,
    };
    let mut out: Vec<RemoteFile> = files
        .iter()
        .filter(root)
        .filter(|f| SMALL.contains(&f.path.as_str()))
        .cloned()
        .collect();
    match fmt {
        Format::Safetensors => {
            if st.is_empty() {
                return Err(HubError::NoWeights);
            }
            out.extend(
                files
                    .iter()
                    .filter(|f| f.path == "model.safetensors.index.json")
                    .cloned(),
            );
            out.extend(st.into_iter().cloned());
        }
        _ => {
            let mut groups: BTreeMap<String, Vec<(u32, u32, &RemoteFile)>> = BTreeMap::new();
            for f in files {
                let l = lower(f);
                if !l.ends_with(".gguf") || l.contains("mmproj") || l.contains("imatrix") {
                    continue;
                }
                let (k, i, n) = gguf_split(&f.path);
                groups.entry(k).or_default().push((i, n, f));
            }
            let mut best: Option<((usize, u64), Vec<&RemoteFile>)> = None;
            for (key, mut parts) in groups {
                parts.sort_by_key(|p| p.0);
                let n = parts[0].1;
                let complete = parts.iter().all(|p| p.1 == n)
                    && parts.len() as u32 == n
                    && parts.iter().enumerate().all(|(i, p)| p.0 == i as u32 + 1);
                if !complete {
                    continue;
                }
                let up = key.to_ascii_uppercase();
                let rank = sel
                    .quants
                    .iter()
                    .position(|q| quant_in(&up, q))
                    .unwrap_or(usize::MAX);
                let size: u64 = parts.iter().map(|p| p.2.size).sum();
                if best.as_ref().is_none_or(|(b, _)| (rank, size) < *b) {
                    best = Some(((rank, size), parts.into_iter().map(|p| p.2).collect()));
                }
            }
            let (_, g) = best.ok_or(HubError::NoWeights)?;
            out.extend(g.into_iter().cloned());
        }
    }
    out.sort_by(|a, b| {
        (
            a.path.ends_with(".safetensors") || a.path.ends_with(".gguf"),
            &a.path,
        )
            .cmp(&(
                b.path.ends_with(".safetensors") || b.path.ends_with(".gguf"),
                &b.path,
            ))
    });
    out.dedup_by(|a, b| a.path == b.path);
    Ok(out)
}

pub fn estimate_bytes(files: &[RemoteFile]) -> u64 {
    files.iter().fold(0u64, |a, f| a.saturating_add(f.size))
}

/// `df`'s arguments; `--` keeps a directory called `-x…` from being read as an option.
fn df_args(dir: &Path) -> Vec<std::ffi::OsString> {
    vec!["-Pk".into(), "--".into(), dir.as_os_str().to_os_string()]
}

/// Free bytes on the volume holding `dir` (`df` on Unix); `None` when it cannot be known.
pub fn free_space(dir: &Path) -> Option<u64> {
    if cfg!(windows) {
        return None;
    }
    let out = Command::new("df")
        .args(df_args(dir))
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let kb: u64 = text
        .lines()
        .nth(1)?
        .split_whitespace()
        .nth(3)?
        .parse()
        .ok()?;
    kb.checked_mul(1024)
}

// -------------------------------------------------------------------- safe paths

/// Join a remote file name onto `dest`, refusing anything that could leave it.
pub fn safe_join(dest: &Path, name: &str) -> Result<PathBuf, HubError> {
    let bad = || HubError::UnsafePath(name.chars().take(80).collect());
    if name.is_empty()
        || name.len() > 1024
        || name.starts_with('/')
        || name.contains(['\\', ':', '\0'])
        || name.chars().any(char::is_control)
    {
        return Err(bad());
    }
    let mut p = dest.to_path_buf();
    for seg in name.split('/') {
        let up = seg.split('.').next().unwrap_or("").to_ascii_uppercase();
        let reserved = matches!(up.as_str(), "CON" | "PRN" | "AUX" | "NUL")
            || (up.len() == 4
                && (up.starts_with("COM") || up.starts_with("LPT"))
                && up.as_bytes()[3].is_ascii_digit());
        if seg.is_empty()
            || seg == "."
            || seg == ".."
            || seg.ends_with('.')
            || seg.ends_with(' ')
            || reserved
        {
            return Err(bad());
        }
        p.push(seg);
    }
    Ok(p)
}

// ---------------------------------------------------------------------- the engine

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tool {
    Curl(String),
    Wget(String),
}
impl Tool {
    pub fn detect() -> Result<Tool, HubError> {
        let curl: &[&str] = if cfg!(windows) {
            &["curl.exe", "curl"]
        } else {
            &["curl"]
        };
        for c in curl {
            if works(c) {
                return Ok(Tool::Curl((*c).into()));
            }
        }
        let wget: &[&str] = if cfg!(windows) {
            &["wget.exe"]
        } else {
            &["wget"]
        };
        for w in wget {
            if works(w) {
                return Ok(Tool::Wget((*w).into()));
            }
        }
        Err(HubError::NoTool)
    }
}
fn works(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

pub struct Progress<'a> {
    pub file: &'a str,
    pub file_done: u64,
    /// 0 when unknown.
    pub file_total: u64,
    pub done: u64,
    pub total: u64,
}

struct TempFile(PathBuf);
impl TempFile {
    fn new(tag: &str, content: Option<&str>) -> Result<TempFile, HubError> {
        static N: AtomicU32 = AtomicU32::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let p = std::env::temp_dir().join(format!(
            "hub-{}-{}-{nanos}-{tag}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let mut o = OpenOptions::new();
        o.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
        let mut f = o.open(&p)?;
        if let Some(c) = content {
            f.write_all(c.as_bytes())?;
        }
        Ok(TempFile(p))
    }
}
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub struct Downloader {
    hub: Hub,
    tool: Tool,
    token: Option<String>,
    loopback_http: bool,
    /// Refuse single files larger than this (curl `--max-filesize`, and the size is watched while
    /// downloading). `None` means [`DEFAULT_MAX_FILE`] — never "no limit". Setting it also allows
    /// files whose size the hub did not announce.
    pub max_filesize: Option<u64>,
    /// Refuse a download whose files add up to more than this (`None`: [`DEFAULT_MAX_TOTAL`]).
    pub max_total: Option<u64>,
    /// `(repo, revision)` → commit SHA, resolved once per downloader.
    pins: std::sync::Mutex<std::collections::HashMap<(String, String), String>>,
}

/// Is `url` plain http to 127.0.0.1 / localhost? The authority is parsed the way curl does: anything
/// with user information (`http://127.0.0.1@evil.example/`), a backslash or a control character is
/// not loopback, whatever precedes the `@`.
fn loopback(url: &str) -> bool {
    let Some(r) = url.strip_prefix("http://") else {
        return false;
    };
    let auth = r.split(['/', '?', '#']).next().unwrap_or("");
    if auth.contains(['@', '\\']) || auth.chars().any(|c| c.is_control() || c == ' ') {
        return false;
    }
    let host = match auth.rsplit_once(':') {
        Some((h, p)) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) => h,
        Some(_) => return false,
        None => auth,
    };
    host == "127.0.0.1" || host == "localhost"
}

/// Ceiling for one file when nothing else is set (1 TiB).
pub const DEFAULT_MAX_FILE: u64 = 1 << 40;
/// Ceiling for one whole download when nothing else is set (4 TiB).
pub const DEFAULT_MAX_TOTAL: u64 = 1 << 42;
/// What a file of unannounced size may be when it is not a weights file and the caller set no cap.
const UNKNOWN_SMALL_FILE: u64 = 256 << 20;
/// Ceiling on a listing / metadata answer.
const TEXT_LIMIT: u64 = 64 << 20;

fn is_weights(path: &str) -> bool {
    let l = path.to_ascii_lowercase();
    [
        ".safetensors",
        ".gguf",
        ".bin",
        ".pt",
        ".pth",
        ".onnx",
        ".ckpt",
        ".h5",
        ".msgpack",
    ]
    .iter()
    .any(|e| l.ends_with(e))
}

impl Downloader {
    /// Detects curl/wget and reads `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`.
    pub fn new(hub: Hub) -> Result<Downloader, HubError> {
        let token = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
            .iter()
            .find_map(|k| std::env::var(k).ok())
            .filter(|t| !t.trim().is_empty());
        Ok(Downloader {
            hub,
            tool: Tool::detect()?,
            token: token.map(|t| t.trim().to_string()),
            loopback_http: false,
            max_filesize: None,
            max_total: None,
            pins: Default::default(),
        })
    }
    /// Per-file ceiling in bytes; also allows files of unannounced size, up to that cap.
    pub fn with_max_filesize(mut self, n: u64) -> Self {
        self.max_filesize = Some(n);
        self
    }
    /// Ceiling on the sum of all files of one download.
    pub fn with_max_total(mut self, n: u64) -> Self {
        self.max_total = Some(n);
        self
    }
    fn ceiling(&self) -> u64 {
        self.max_filesize.unwrap_or(DEFAULT_MAX_FILE)
    }
    /// The most a file may be: its announced size, or for an unannounced one the cap.
    fn file_limit(&self, f: &RemoteFile) -> Result<u64, HubError> {
        if f.size > 0 {
            if f.size > self.ceiling() {
                return Err(HubError::TooLarge(format!(
                    "{} is {} bytes, the per-file ceiling is {}",
                    f.path,
                    f.size,
                    self.ceiling()
                )));
            }
            return Ok(f.size);
        }
        match self.max_filesize {
            Some(m) => Ok(m),
            None if !is_weights(&f.path) => Ok(UNKNOWN_SMALL_FILE),
            None => Err(HubError::UnknownSize(f.path.chars().take(80).collect())),
        }
    }
    pub fn with_token(mut self, t: Option<String>) -> Self {
        self.token = t;
        self
    }
    pub fn with_tool(mut self, t: Tool) -> Self {
        self.tool = t;
        self
    }
    /// Allow plain http to 127.0.0.1/localhost only (for tests and local mirrors).
    pub fn with_loopback_http(mut self, yes: bool) -> Self {
        self.loopback_http = yes;
        self
    }
    pub fn tool(&self) -> &Tool {
        &self.tool
    }

    fn check_url(&self, url: &str) -> Result<(), HubError> {
        if url.starts_with("https://") || (self.loopback_http && loopback(url)) {
            Ok(())
        } else {
            Err(HubError::InsecureUrl(url.chars().take(80).collect()))
        }
    }

    fn header_text(&self) -> Result<Option<String>, HubError> {
        // wget would repeat the header on a redirect to another host: it gets no credentials at all
        if matches!(self.tool, Tool::Wget(_)) {
            return Ok(None);
        }
        let Some(t) = &self.token else {
            return Ok(None);
        };
        if t.chars().any(|c| c.is_control() || c == ' ') {
            return Err(HubError::BadSpec(
                "the access token contains invalid characters".into(),
            ));
        }
        Ok(Some(match self.tool {
            Tool::Curl(_) => format!("Authorization: Bearer {t}\n"),
            Tool::Wget(_) => return Ok(None),
        }))
    }

    /// One transfer. `tick` gets the current size of `out` about 20 times a second; the transfer is
    /// killed as soon as `out` grows beyond `limit` bytes (a server may omit `Content-Length`).
    fn run(
        &self,
        url: &str,
        out: &Path,
        resume: bool,
        limit: u64,
        cancel: &AtomicBool,
        tick: &mut dyn FnMut(u64),
    ) -> Result<(), HubError> {
        self.check_url(url)?;
        let hdr = match self.header_text()? {
            Some(h) => Some(TempFile::new("hdr", Some(&h))?),
            None => None,
        };
        let errf = TempFile::new("err", None)?;
        let local = loopback(url);
        let mut cmd = match &self.tool {
            Tool::Curl(c) => {
                let mut cmd = Command::new(c);
                cmd.args([
                    "-sS",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--retry-delay",
                    "2",
                    "--connect-timeout",
                    "20",
                ]);
                cmd.args([
                    "--proto",
                    if local { "=http,https" } else { "=https" },
                    "--proto-redir",
                    "=https",
                ]);
                if local {
                    cmd.args(["--noproxy", "*"]);
                }
                if resume {
                    cmd.args(["-C", "-"]);
                }
                cmd.arg("--max-filesize")
                    .arg(limit.max(TEXT_LIMIT).to_string());
                if let Some(h) = &hdr {
                    cmd.arg("-H").arg(format!("@{}", h.0.display()));
                }
                cmd.arg("-o").arg(out).arg("--").arg(url);
                cmd
            }
            Tool::Wget(w) => {
                let mut cmd = Command::new(w);
                cmd.args(["-q", "--tries=3", "--timeout=30", "--max-redirect=5"]);
                if !local {
                    cmd.arg("--https-only");
                } else {
                    cmd.arg("--no-proxy");
                }
                if resume {
                    cmd.arg("-c");
                }
                if let Some(h) = &hdr {
                    cmd.arg(format!("--config={}", h.0.display()));
                }
                cmd.arg("-O").arg(out).arg("--").arg(url);
                cmd
            }
        };
        let mut child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(File::create(&errf.0)?))
            .spawn()
            .map_err(|_| HubError::NoTool)?;
        let status = loop {
            if cancel.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(HubError::Cancelled);
            }
            if let Some(s) = child.try_wait()? {
                break s;
            }
            let size = fs::metadata(out).map_or(0, |m| m.len());
            if size > limit {
                let _ = child.kill();
                let _ = child.wait();
                return Err(HubError::Integrity(format!(
                    "the server sent more than the {limit} bytes allowed for this file"
                )));
            }
            tick(size);
            std::thread::sleep(Duration::from_millis(50));
        };
        if fs::metadata(out).map_or(0, |m| m.len()) > limit {
            return Err(HubError::Integrity(format!(
                "the server sent more than the {limit} bytes allowed for this file"
            )));
        }
        if status.success() {
            return Ok(());
        }
        let mut err = String::new();
        let _ = File::open(&errf.0).and_then(|mut f| f.read_to_string(&mut err));
        Err(classify(&self.tool, status.code(), &err))
    }

    fn fetch_text(&self, url: &str, cancel: &AtomicBool) -> Result<String, HubError> {
        let t = TempFile::new("txt", None)?;
        self.run(url, &t.0, false, TEXT_LIMIT, cancel, &mut |_| {})?;
        let mut b = vec![];
        File::open(&t.0)?.take(64 << 20).read_to_end(&mut b)?;
        String::from_utf8(b).map_err(|_| HubError::Parse("answer is not UTF-8".into()))
    }

    /// List the files of a repo (tree API first, model-info `siblings` as a fallback).
    pub fn list_files(
        &self,
        spec: &RepoSpec,
        cancel: &AtomicBool,
    ) -> Result<Vec<RemoteFile>, HubError> {
        let spec = &self.pinned(spec, cancel);
        match self
            .fetch_text(&self.hub.tree_url(spec), cancel)
            .and_then(|t| parse_tree(&t))
        {
            Ok(f) => Ok(f),
            Err(
                e @ (HubError::Offline
                | HubError::Gated { .. }
                | HubError::NotFound
                | HubError::RateLimited
                | HubError::Cancelled
                | HubError::Tls
                | HubError::NoTool
                | HubError::InsecureUrl(_)),
            ) => Err(e),
            Err(_) if self.hub != Hub::ModelScope => {
                parse_model_info(&self.fetch_text(&self.hub.info_url(spec), cancel)?)
            }
            Err(e) => Err(e),
        }
    }

    /// `spec` with its revision replaced by the commit SHA it points to right now, so a branch that
    /// moves between listing and downloading cannot mix two commits. Needs the hub's model-info
    /// endpoint (Hugging Face and compatible mirrors); where that is unavailable (ModelScope, an
    /// offline mirror, an answer without a `sha`) the spec comes back unchanged. The answer is
    /// remembered, so [`list_files`](Downloader::list_files) and
    /// [`download_all`](Downloader::download_all) agree.
    pub fn pinned(&self, spec: &RepoSpec, cancel: &AtomicBool) -> RepoSpec {
        let is_sha = |r: &str| r.len() == 40 && r.bytes().all(|b| b.is_ascii_hexdigit());
        if is_sha(&spec.revision) || self.hub == Hub::ModelScope {
            return spec.clone();
        }
        let key = (spec.repo.clone(), spec.revision.clone());
        if let Some(sha) = self.pins.lock().ok().and_then(|p| p.get(&key).cloned()) {
            return RepoSpec {
                repo: spec.repo.clone(),
                revision: sha,
            };
        }
        let sha = self
            .fetch_text(&self.hub.info_url(spec), cancel)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| {
                v.get("sha")
                    .and_then(Value::as_str)
                    .map(str::to_ascii_lowercase)
            })
            .filter(|s| is_sha(s));
        match sha {
            Some(sha) => {
                if let Ok(mut p) = self.pins.lock() {
                    p.insert(key, sha.clone());
                }
                RepoSpec {
                    repo: spec.repo.clone(),
                    revision: sha,
                }
            }
            None => {
                // remember the miss too: one lookup per downloader, not one per call
                if let Ok(mut p) = self.pins.lock() {
                    p.insert(key, spec.revision.clone());
                }
                spec.clone()
            }
        }
    }

    /// Download one file into `dest`, resuming a `.part`, verifying size and SHA-256.
    /// Returns the final path. `base`/`total` only feed the progress callback.
    #[allow(clippy::too_many_arguments)]
    fn download_file(
        &self,
        spec: &RepoSpec,
        f: &RemoteFile,
        dest: &Path,
        part: &Path,
        base: u64,
        total: u64,
        cancel: &AtomicBool,
        on: &mut dyn FnMut(&Progress),
    ) -> Result<PathBuf, HubError> {
        let limit = self.file_limit(f)?;
        let final_path = safe_join(dest, &f.path)?;
        if let Some(parent) = final_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut report = |done: u64| {
            on(&Progress {
                file: &f.path,
                file_done: done,
                file_total: f.size,
                done: base.saturating_add(done),
                total,
            })
        };
        let complete = |p: &Path| -> Result<bool, HubError> {
            match fs::metadata(p) {
                Ok(m) if f.size == 0 || m.len() == f.size => Ok(match &f.sha256 {
                    Some(h) => sha256_file(p)? == *h,
                    None => f.size > 0,
                }),
                _ => Ok(false),
            }
        };
        if final_path.is_file() {
            if complete(&final_path)? {
                report(f.size);
                return Ok(final_path);
            }
            fs::remove_file(&final_path)?;
        }
        // never write through a link somebody left where the partial file goes
        if fs::symlink_metadata(part).is_ok_and(|m| !m.file_type().is_file()) {
            fs::remove_file(part)?;
        }
        if f.size > 0 && fs::metadata(part).is_ok_and(|m| m.len() > f.size) {
            fs::remove_file(part)?;
        }
        let url = self.hub.file_url(spec, &f.path);
        if !(f.size > 0 && fs::metadata(part).is_ok_and(|m| m.len() == f.size)) {
            match self.run(&url, part, true, limit, cancel, &mut |n| report(n)) {
                Err(HubError::Tool(m)) if m.contains("range") || m.contains("(33)") => {
                    // the server ignored Range: start over once
                    let _ = fs::remove_file(part);
                    self.run(&url, part, false, limit, cancel, &mut |n| report(n))?;
                }
                Err(e @ HubError::Integrity(_)) => {
                    let _ = fs::remove_file(part);
                    return Err(e);
                }
                r => r?,
            }
        }
        let got = fs::metadata(part)?.len();
        if f.size > 0 && got != f.size {
            let _ = fs::remove_file(part);
            return Err(HubError::Integrity(format!(
                "{}: got {got} bytes, expected {}",
                f.path, f.size
            )));
        }
        if let Some(h) = &f.sha256
            && sha256_file(part)? != *h
        {
            let _ = fs::remove_file(part);
            return Err(HubError::Integrity(format!(
                "{}: SHA-256 differs from the hub's",
                f.path
            )));
        }
        if cfg!(windows) {
            let _ = fs::remove_file(&final_path);
        }
        fs::rename(part, &final_path)?;
        report(got);
        Ok(final_path)
    }

    /// Download all `files` into `dest`. The revision is pinned to a commit first (see
    /// [`pinned`](Downloader::pinned)); sizes are checked against the ceilings and, when it can be
    /// known, against the free space, before anything is fetched.
    pub fn download_all(
        &self,
        spec: &RepoSpec,
        files: &[RemoteFile],
        dest: &Path,
        cancel: &AtomicBool,
        on: &mut dyn FnMut(&Progress),
    ) -> Result<Vec<PathBuf>, HubError> {
        let spec = &self.pinned(spec, cancel);
        // what each file may weigh: its announced size, or the explicit cap for an unannounced one
        let mut limits = Vec::with_capacity(files.len());
        for f in files {
            limits.push(self.file_limit(f)?);
        }
        let total = limits.iter().fold(0u64, |a, l| a.saturating_add(*l));
        let max_total = self.max_total.unwrap_or(DEFAULT_MAX_TOTAL);
        if total > max_total {
            return Err(HubError::TooLarge(format!(
                "{total} bytes in {} files, the total ceiling is {max_total}",
                files.len()
            )));
        }
        // partial-file names: `a.part` unless the hub also lists a file of that very name
        let finals: std::collections::HashSet<PathBuf> = files
            .iter()
            .filter_map(|f| safe_join(dest, &f.path).ok())
            .collect();
        let part_of = |p: &Path| -> PathBuf {
            let mut n = 0u32;
            loop {
                let mut q = p.as_os_str().to_os_string();
                q.push(if n == 0 {
                    ".part".to_string()
                } else {
                    format!(".part{n}")
                });
                let q = PathBuf::from(q);
                if !finals.contains(&q) {
                    return q;
                }
                n += 1;
            }
        };
        fs::create_dir_all(dest)?;
        let mut have = 0u64;
        for (f, lim) in files.iter().zip(&limits) {
            if let Ok(p) = safe_join(dest, &f.path) {
                have = have.saturating_add(
                    fs::metadata(&p)
                        .or_else(|_| fs::metadata(part_of(&p)))
                        .map_or(0, |m| m.len().min(*lim)),
                );
            }
        }
        if let Some(free) = free_space(dest)
            && total.saturating_sub(have) > free
        {
            return Err(HubError::DiskFull);
        }
        let (mut base, mut out) = (0u64, vec![]);
        for (f, lim) in files.iter().zip(&limits) {
            if cancel.load(Ordering::Relaxed) {
                return Err(HubError::Cancelled);
            }
            let part = part_of(&safe_join(dest, &f.path)?);
            out.push(self.download_file(spec, f, dest, &part, base, total, cancel, on)?);
            base = base.saturating_add(*lim);
        }
        Ok(out)
    }
}

// --------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    fn rf(p: &str, size: u64) -> RemoteFile {
        RemoteFile {
            path: p.into(),
            size,
            sha256: None,
        }
    }

    #[test]
    fn sha256_nist_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let mut s = Sha256::new();
        for _ in 0..1000 {
            s.update(&[b'a'; 1000]);
        }
        assert_eq!(
            hex(&s.finish()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn sha256_streaming_equals_one_shot_across_every_split() {
        let data: Vec<u8> = (0..300u32).map(|i| (i * 7) as u8).collect();
        let want = sha256_hex(&data);
        for cut in [0, 1, 55, 56, 63, 64, 65, 127, 128, 200, 300] {
            let mut s = Sha256::new();
            s.update(&data[..cut]);
            s.update(&data[cut..]);
            assert_eq!(hex(&s.finish()), want, "cut {cut}");
        }
    }

    #[test]
    fn repo_specs() {
        let ok = |s: &str| RepoSpec::parse(s).unwrap();
        let r = |a: &str, b: &str| RepoSpec {
            repo: a.into(),
            revision: b.into(),
        };
        assert_eq!(ok("Qwen/Qwen3-0.6B"), r("Qwen/Qwen3-0.6B", "main"));
        assert_eq!(ok("org/name@v1.0"), r("org/name", "v1.0"));
        assert_eq!(ok("hf://org/name@abc123"), r("org/name", "abc123"));
        assert_eq!(ok("hf://models/org/name"), r("org/name", "main"));
        assert_eq!(ok("gpt2"), r("gpt2", "main"));
        assert_eq!(ok("https://huggingface.co/org/name"), r("org/name", "main"));
        assert_eq!(
            ok("https://huggingface.co/org/name/tree/dev?x=1"),
            r("org/name", "dev")
        );
        assert_eq!(
            ok("https://hf-mirror.com/org/name/resolve/refs%2Fpr%2F1/model.gguf"),
            r("org/name", "refs/pr/1")
        );
        for bad in [
            "",
            "http://huggingface.co/a/b",
            "a/../b",
            "../x",
            "a b/c",
            "org/name@../x",
            "a/b/c",
            "https://huggingface.co",
        ] {
            assert!(RepoSpec::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn urls() {
        let s = RepoSpec::parse("org/name@refs/pr/1").unwrap();
        assert_eq!(
            Hub::HuggingFace.tree_url(&s),
            "https://huggingface.co/api/models/org/name/tree/refs%2Fpr%2F1?recursive=1"
        );
        assert_eq!(
            Hub::HuggingFace.file_url(&s, "a b/c.gguf"),
            "https://huggingface.co/org/name/resolve/refs%2Fpr%2F1/a%20b/c.gguf"
        );
        let m = Hub::HfCompatible("https://hf-mirror.com".into());
        assert!(
            m.file_url(&RepoSpec::parse("o/n").unwrap(), "x.json")
                .starts_with("https://hf-mirror.com/o/n/resolve/main/")
        );
        let ms = Hub::ModelScope.file_url(&RepoSpec::parse("o/n").unwrap(), "dir/x.bin");
        assert!(ms.contains("FilePath=dir%2Fx.bin") && ms.contains("Revision=main"));
    }

    #[test]
    fn tree_parsing_prefers_lfs_data() {
        let sha = "a".repeat(64);
        let j = format!(
            r#"[{{"type":"directory","path":"sub"}},
            {{"type":"file","oid":"deadbeef","size":134,"path":"config.json"}},
            {{"type":"file","oid":"x","size":135,"path":"m.safetensors","lfs":{{"oid":"{sha}","size":999,"pointerSize":134}}}},
            {{"type":"file","size":1,"path":"bad.bin","lfs":{{"oid":"nothex","size":5}}}}]"#
        );
        let f = parse_tree(&j).unwrap();
        assert_eq!(f.len(), 3);
        assert_eq!((f[0].size, f[0].sha256.clone()), (134, None));
        assert_eq!((f[1].size, f[1].sha256.clone()), (999, Some(sha)));
        assert_eq!(f[2].sha256, None);
        assert!(parse_tree(r#"{"error":"Repository not found"}"#).is_err());
        assert!(parse_tree("nope").is_err());
        let ms = parse_tree(&format!(r#"{{"Data":{{"Files":[{{"Path":"a.bin","Size":7,"Type":"blob","Sha256":"{}"}},{{"Path":"d","Type":"tree"}}]}}}}"#, "B".repeat(64))).unwrap();
        assert_eq!((ms.len(), ms[0].sha256.clone().unwrap().len()), (1, 64));
        let info = parse_model_info(
            r#"{"siblings":[{"rfilename":"config.json"},{"rfilename":"w.safetensors","size":5}]}"#,
        )
        .unwrap();
        assert_eq!((info[0].size, info[1].size), (0, 5));
    }

    #[test]
    fn safetensors_selection_skips_duplicates_and_unrelated_files() {
        let files = [
            rf("config.json", 1),
            rf("model-00001-of-00002.safetensors", 10),
            rf("model-00002-of-00002.safetensors", 10),
            rf("model.safetensors.index.json", 1),
            rf("consolidated.safetensors", 20),
            rf("pytorch_model.bin", 99),
            rf("tokenizer.json", 2),
            rf("onnx/model.onnx", 50),
            rf("sub/config.json", 1),
            rf("README.md", 1),
        ];
        let got = select_files(&files, &Select::default()).unwrap();
        let names: Vec<&str> = got.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            names,
            [
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );
        assert_eq!(estimate_bytes(&got), 1 + 1 + 2 + 20);
    }

    #[test]
    fn gguf_selection_follows_the_preference_list_and_split_rules() {
        let files = [
            rf("config.json", 1),
            rf("m-Q8_0.gguf", 80),
            rf("m-Q4_K_S.gguf", 40),
            rf("m-Q4_K_M.gguf", 45),
            rf("m-BF16.gguf", 160),
            rf("mmproj-F16.gguf", 5),
            rf("m.imatrix.gguf", 5),
        ];
        let pick = |s: &Select| {
            select_files(&files, s)
                .unwrap()
                .iter()
                .map(|f| f.path.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(pick(&Select::default()), ["config.json", "m-Q4_K_M.gguf"]);
        let mut s = Select {
            quants: vec!["q8_0".into()],
            ..Select::default()
        };
        assert_eq!(pick(&s)[1], "m-Q8_0.gguf");
        s.quants = vec!["F16".into()]; // must not match BF16
        assert_eq!(pick(&s)[1], "m-Q4_K_S.gguf"); // nothing ranked: the smallest
        // split files: only complete groups count; all shards are returned
        let sp = [
            rf("big-Q4_K_M-00001-of-00003.gguf", 10),
            rf("big-Q4_K_M-00002-of-00003.gguf", 10),
            rf("big-Q4_K_M-00003-of-00003.gguf", 10),
            rf("big-Q8_0-00001-of-00002.gguf", 20),
            rf("Q5_K_M/big-00001-of-00002.gguf", 7),
            rf("Q5_K_M/big-00002-of-00002.gguf", 7),
        ];
        let got = select_files(&sp, &Select::default()).unwrap();
        assert_eq!(got.len(), 3);
        assert!(got.iter().all(|f| f.path.contains("Q4_K_M")));
        let missing = [rf("x-Q4_K_M-00001-of-00002.gguf", 1), rf("y-Q8_0.gguf", 9)];
        assert_eq!(
            select_files(&missing, &Select::default()).unwrap()[0].path,
            "y-Q8_0.gguf"
        );
        assert_eq!(
            select_files(&[rf("README.md", 1), rf("m.bin", 1)], &Select::default()),
            Err(HubError::NoWeights)
        );
        assert_eq!(
            select_files(
                &[rf("m-Q4_0.gguf", 1)],
                &Select {
                    format: Format::Safetensors,
                    ..Select::default()
                }
            ),
            Err(HubError::NoWeights)
        );
    }

    #[test]
    fn hostile_remote_names_cannot_leave_the_destination() {
        let d = Path::new("/tmp/dest");
        for bad in [
            "",
            "../x",
            "a/../../x",
            "/etc/passwd",
            "a\\b",
            "C:/x",
            "a//b",
            "./a",
            "a/.",
            "x\0y",
            "a\nb",
            "con.txt",
            "dir/NUL",
            "name.",
            "COM1",
        ] {
            assert!(safe_join(d, bad).is_err(), "{bad:?}");
        }
        assert_eq!(
            safe_join(d, "sub/model-00001.safetensors").unwrap(),
            d.join("sub").join("model-00001.safetensors")
        );
    }

    #[test]
    fn errors_are_classified() {
        let c = Tool::Curl("curl".into());
        let w = Tool::Wget("wget".into());
        assert_eq!(
            classify(
                &c,
                Some(22),
                "curl: (22) The requested URL returned error: 404"
            ),
            HubError::NotFound
        );
        assert_eq!(
            classify(
                &c,
                Some(22),
                "curl: (22) The requested URL returned error: 403"
            ),
            HubError::Gated { status: 403 }
        );
        assert_eq!(
            classify(
                &c,
                Some(22),
                "curl: (22) The requested URL returned error: 401"
            ),
            HubError::Gated { status: 401 }
        );
        assert_eq!(
            classify(
                &c,
                Some(22),
                "curl: (22) The requested URL returned error: 429"
            ),
            HubError::RateLimited
        );
        assert_eq!(
            classify(
                &c,
                Some(22),
                "curl: (22) The requested URL returned error: 503"
            ),
            HubError::Http(503)
        );
        assert_eq!(
            classify(&c, Some(6), "curl: (6) Could not resolve host"),
            HubError::Offline
        );
        assert_eq!(classify(&c, Some(28), ""), HubError::Offline);
        assert_eq!(
            classify(&c, Some(60), "SSL certificate problem"),
            HubError::Tls
        );
        assert_eq!(
            classify(&c, Some(23), "curl: (23) Failure writing output"),
            HubError::DiskFull
        );
        assert_eq!(
            classify(&c, Some(1), "write failed: No space left on device"),
            HubError::DiskFull
        );
        assert_eq!(
            classify(
                &w,
                Some(8),
                "HTTP request sent, awaiting response... 404 \nERROR 404: Not Found."
            ),
            HubError::NotFound
        );
        assert_eq!(classify(&w, Some(4), ""), HubError::Offline);
        assert!(
            matches!(classify(&c, Some(99), "boom\nlast line"), HubError::Tool(m) if m == "last line")
        );
        assert!(
            HubError::Gated { status: 403 }
                .to_string()
                .contains("HF_TOKEN")
        );
    }

    #[test]
    fn plain_http_is_refused_unless_loopback_is_allowed() {
        let Ok(d) = Downloader::new(Hub::HuggingFace) else {
            return skipped("no download tool (curl/wget) available");
        };
        assert!(matches!(
            d.check_url("http://example.com/x"),
            Err(HubError::InsecureUrl(_))
        ));
        assert!(matches!(
            d.check_url("http://127.0.0.1:9/x"),
            Err(HubError::InsecureUrl(_))
        ));
        assert!(d.check_url("https://huggingface.co/x").is_ok());
        let d = d.with_loopback_http(true);
        assert!(d.check_url("http://127.0.0.1:9/x").is_ok());
        assert!(matches!(
            d.check_url("http://evil.example/x"),
            Err(HubError::InsecureUrl(_))
        ));
        assert!(matches!(
            d.check_url("http://127.0.0.1.evil.example/x"),
            Err(HubError::InsecureUrl(_))
        ));
    }

    #[test]
    fn a_token_never_reaches_a_command_line() {
        let d = Downloader::new(Hub::HuggingFace)
            .unwrap()
            .with_token(Some("hf_secret".into()))
            .with_tool(Tool::Curl("curl".into()));
        assert!(d.header_text().unwrap().unwrap().contains("hf_secret")); // goes to the 0600 file only
        let bad = Downloader::new(Hub::HuggingFace)
            .unwrap()
            .with_token(Some("a\nX-Evil: 1".into()))
            .with_tool(Tool::Curl("curl".into()));
        assert!(bad.header_text().is_err());
    }

    // ---- a tiny range-capable HTTP server on loopback

    type Log = Arc<Mutex<Vec<String>>>;

    fn serve(files: HashMap<String, Vec<u8>>, tree: String) -> (u16, Log) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let log: Log = Arc::default();
        let (files, log2) = (Arc::new(files), log.clone());
        std::thread::spawn(move || {
            for c in l.incoming().flatten() {
                let (files, log, tree) = (files.clone(), log2.clone(), tree.clone());
                std::thread::spawn(move || {
                    let mut r = BufReader::new(c.try_clone().unwrap());
                    let mut first = String::new();
                    if r.read_line(&mut first).is_err() {
                        return;
                    }
                    let mut range = None;
                    loop {
                        let mut h = String::new();
                        if r.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" {
                            break;
                        }
                        if let Some(v) = h.to_ascii_lowercase().strip_prefix("range: bytes=") {
                            range = v.trim().trim_end_matches('-').parse::<usize>().ok();
                        }
                    }
                    let path = first.split_whitespace().nth(1).unwrap_or("/").to_string();
                    log.lock().unwrap().push(format!("{path} range={range:?}"));
                    let mut w = c;
                    let reply = |w: &mut std::net::TcpStream,
                                 code: &str,
                                 body: &[u8],
                                 extra: &str| {
                        let _ = write!(
                            w,
                            "HTTP/1.1 {code}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = w.write_all(body);
                    };
                    if path.starts_with("/gated/") {
                        return reply(&mut w, "401 Unauthorized", b"{}", "");
                    }
                    if path.starts_with("/api/models/") {
                        return reply(&mut w, "200 OK", tree.as_bytes(), "");
                    }
                    let key = path.splitn(6, '/').nth(5).unwrap_or("").to_string();
                    let Some(body) = files.get(&key) else {
                        return reply(&mut w, "404 Not Found", b"no", "");
                    };
                    let start = range.unwrap_or(0).min(body.len());
                    let (code, extra) = if range.is_some() {
                        (
                            "206 Partial Content",
                            format!(
                                "Content-Range: bytes {start}-{}/{}\r\n",
                                body.len() - 1,
                                body.len()
                            ),
                        )
                    } else {
                        ("200 OK", String::new())
                    };
                    if key == "slow.bin" {
                        let _ = write!(
                            w,
                            "HTTP/1.1 {code}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
                            body.len() - start
                        );
                        for ch in body[start..].chunks(8192) {
                            if w.write_all(ch).is_err() {
                                return;
                            }
                            std::thread::sleep(Duration::from_millis(40));
                        }
                    } else {
                        reply(&mut w, code, &body[start..], &extra);
                    }
                });
            }
        });
        (port, log)
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hub-test-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }
    fn bytes(n: usize, seed: u8) -> Vec<u8> {
        (0..n)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }
    fn dl(port: u16) -> Option<Downloader> {
        Downloader::new(Hub::HfCompatible(format!("http://127.0.0.1:{port}")))
            .ok()
            .map(|d| d.with_loopback_http(true).with_token(None))
    }

    #[test]
    fn end_to_end_against_a_local_server_with_listing_selection_resume_and_verification() {
        let (cfg, w1, w2) = (
            b"{\"hidden_size\":8}".to_vec(),
            bytes(70_000, 1),
            bytes(33_333, 2),
        );
        let tree = serde_json::json!([
            {"type":"file","path":"config.json","size":cfg.len()},
            {"type":"file","path":"model-00001-of-00002.safetensors","size":w1.len(),"lfs":{"oid":sha256_hex(&w1),"size":w1.len()}},
            {"type":"file","path":"model-00002-of-00002.safetensors","size":w2.len(),"lfs":{"oid":sha256_hex(&w2),"size":w2.len()}},
            {"type":"file","path":"pytorch_model.bin","size":5}
        ]).to_string();
        let files = HashMap::from([
            ("config.json".into(), cfg.clone()),
            ("model-00001-of-00002.safetensors".into(), w1.clone()),
            ("model-00002-of-00002.safetensors".into(), w2.clone()),
        ]);
        let (port, log) = serve(files, tree);
        let Some(d) = dl(port) else {
            eprintln!("no curl/wget: skipped");
            return;
        };
        let cancel = AtomicBool::new(false);
        let spec = RepoSpec::parse("org/name").unwrap();
        let listed = d.list_files(&spec, &cancel).unwrap();
        let chosen = select_files(&listed, &Select::default()).unwrap();
        assert_eq!(chosen.len(), 3);
        let dest = tmp("e2e");
        // a previous interrupted run left the first 1000 bytes of shard 1
        fs::write(
            dest.join("model-00001-of-00002.safetensors.part"),
            &w1[..1000],
        )
        .unwrap();
        let mut last = (0, 0);
        let paths = d
            .download_all(&spec, &chosen, &dest, &cancel, &mut |p| {
                last = (p.done, p.total)
            })
            .unwrap();
        assert_eq!(paths.len(), 3);
        assert_eq!(
            fs::read(dest.join("model-00001-of-00002.safetensors")).unwrap(),
            w1
        );
        assert_eq!(fs::read(dest.join("config.json")).unwrap(), cfg);
        assert_eq!(last, (estimate_bytes(&chosen), estimate_bytes(&chosen)));
        assert!(!dest.join("model-00001-of-00002.safetensors.part").exists());
        assert!(
            log.lock()
                .unwrap()
                .iter()
                .any(|l| l.contains("model-00001") && l.contains("range=Some(1000)")),
            "{:?}",
            log.lock().unwrap()
        );
        // second run: everything is complete and verified, nothing is fetched
        let before = log.lock().unwrap().len();
        d.download_all(&spec, &chosen, &dest, &cancel, &mut |_| {})
            .unwrap();
        assert_eq!(log.lock().unwrap().len(), before);
        fs::remove_dir_all(dest).unwrap();
    }

    #[test]
    fn a_corrupted_download_is_rejected_and_not_kept() {
        let good = bytes(5000, 3);
        let mut served = good.clone();
        served[100] ^= 1;
        let (port, _) = serve(
            HashMap::from([("w.safetensors".into(), served)]),
            "[]".into(),
        );
        let Some(d) = dl(port) else {
            return skipped("curl is not installed or not working");
        };
        let dest = tmp("corrupt");
        let f = RemoteFile {
            path: "w.safetensors".into(),
            size: 5000,
            sha256: Some(sha256_hex(&good)),
        };
        let e = d
            .download_all(
                &RepoSpec::parse("o/n").unwrap(),
                &[f],
                &dest,
                &AtomicBool::new(false),
                &mut |_| {},
            )
            .unwrap_err();
        assert!(matches!(e, HubError::Integrity(_)), "{e}");
        assert!(!dest.join("w.safetensors").exists() && !dest.join("w.safetensors.part").exists());
        // wrong size, no hash
        let f = RemoteFile {
            path: "w.safetensors".into(),
            size: 4999,
            sha256: None,
        };
        let r = d.download_all(
            &RepoSpec::parse("o/n").unwrap(),
            &[f],
            &dest,
            &AtomicBool::new(false),
            &mut |_| {},
        );
        assert!(matches!(r, Err(HubError::Integrity(_))), "{r:?}");
        fs::remove_dir_all(dest).unwrap();
    }

    #[test]
    fn http_failures_map_to_friendly_errors() {
        let (port, _) = serve(HashMap::new(), "[]".into());
        let Some(d) = dl(port) else {
            return skipped("curl is not installed or not working");
        };
        let dest = tmp("errs");
        let spec = RepoSpec::parse("o/n").unwrap();
        let cancel = AtomicBool::new(false);
        let f = rf("nothing.bin", 10);
        assert_eq!(
            d.download_all(&spec, &[f], &dest, &cancel, &mut |_| {}),
            Err(HubError::NotFound)
        );
        let gated = RepoSpec::parse("gated/x").unwrap();
        // the fake server answers 401 for /gated/… file paths
        let d2 = Downloader::new(Hub::HfCompatible(format!("http://127.0.0.1:{port}/gated")))
            .unwrap()
            .with_loopback_http(true)
            .with_token(None);
        let r = d2.download_all(&gated, &[rf("x.bin", 1)], &dest, &cancel, &mut |_| {});
        assert!(matches!(r, Err(HubError::Gated { status: 401 })), "{r:?}");
        // nothing is listening here
        let closed = TcpListener::bind("127.0.0.1:0").unwrap();
        let p = closed.local_addr().unwrap().port();
        drop(closed);
        let d3 = dl(p).unwrap();
        assert_eq!(
            d3.download_all(&spec, &[rf("x.bin", 1)], &dest, &cancel, &mut |_| {}),
            Err(HubError::Offline)
        );
        fs::remove_dir_all(dest).unwrap();
    }

    #[test]
    fn cancelling_stops_the_transfer_and_a_later_run_resumes() {
        let data = bytes(400_000, 9);
        let (port, log) = serve(
            HashMap::from([("slow.bin".into(), data.clone())]),
            "[]".into(),
        );
        let Some(d) = dl(port) else {
            return skipped("curl is not installed or not working");
        };
        let dest = tmp("cancel");
        let spec = RepoSpec::parse("o/n").unwrap();
        let f = RemoteFile {
            path: "slow.bin".into(),
            size: data.len() as u64,
            sha256: Some(sha256_hex(&data)),
        };
        let cancel = AtomicBool::new(false);
        let mut polls = 0;
        let r = d.download_all(&spec, std::slice::from_ref(&f), &dest, &cancel, &mut |_| {
            polls += 1;
            if polls >= 4 {
                cancel.store(true, Ordering::Relaxed);
            }
        });
        assert_eq!(r, Err(HubError::Cancelled));
        assert!(!dest.join("slow.bin").exists());
        let kept = fs::metadata(dest.join("slow.bin.part")).map_or(0, |m| m.len());
        let cancel = AtomicBool::new(false);
        d.download_all(&spec, &[f], &dest, &cancel, &mut |_| {})
            .unwrap();
        assert_eq!(fs::read(dest.join("slow.bin")).unwrap(), data);
        if kept > 0 {
            assert!(
                log.lock()
                    .unwrap()
                    .iter()
                    .any(|l| l.contains(&format!("range=Some({kept})"))),
                "{:?}",
                log.lock().unwrap()
            );
        }
        // a flag that is already set stops before any request
        let n = log.lock().unwrap().len();
        assert_eq!(
            d.download_all(
                &spec,
                &[rf("slow.bin", 1)],
                &tmp("cancel2"),
                &AtomicBool::new(true),
                &mut |_| {}
            ),
            Err(HubError::Cancelled)
        );
        assert_eq!(log.lock().unwrap().len(), n);
        fs::remove_dir_all(dest).unwrap();
    }

    #[test]
    fn a_hostile_listing_cannot_write_outside_the_destination() {
        let (port, _) = serve(HashMap::new(), "[]".into());
        let Some(d) = dl(port) else {
            return skipped("curl is not installed or not working");
        };
        let dest = tmp("evil");
        let e = d.download_all(
            &RepoSpec::parse("o/n").unwrap(),
            &[rf("../escape.bin", 1)],
            &dest,
            &AtomicBool::new(false),
            &mut |_| {},
        );
        assert!(matches!(e, Err(HubError::UnsafePath(_))));
        assert!(!dest.parent().unwrap().join("escape.bin").exists());
        fs::remove_dir_all(dest).unwrap();
    }

    #[test]
    fn free_space_is_reported_on_unix() {
        if cfg!(unix) {
            assert!(free_space(&std::env::temp_dir()).is_some_and(|n| n > 0));
        }
        assert_eq!(free_space(Path::new("/definitely/not/there")), None);
    }

    #[test]
    fn the_wget_fallback_downloads_too() {
        let data = bytes(20_000, 5);
        let (port, _) = serve(HashMap::from([("w.bin".into(), data.clone())]), "[]".into());
        if !works("wget") {
            return;
        }
        let Some(d) = dl(port) else {
            return skipped("curl is not installed or not working");
        };
        let d = d.with_tool(Tool::Wget("wget".into()));
        let dest = tmp("wget");
        let f = RemoteFile {
            path: "w.bin".into(),
            size: data.len() as u64,
            sha256: Some(sha256_hex(&data)),
        };
        d.download_all(
            &RepoSpec::parse("o/n").unwrap(),
            &[f],
            &dest,
            &AtomicBool::new(false),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(fs::read(dest.join("w.bin")).unwrap(), data);
        fs::remove_dir_all(dest).unwrap();
    }

    // ---- hardening ----

    /// A server whose every answer is raw bytes chosen by `h(path)`; also logs the paths.
    fn serve_raw(h: impl Fn(&str) -> Vec<u8> + Send + Sync + 'static) -> (u16, Log) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let log: Log = Arc::default();
        let (h, log2) = (Arc::new(h), log.clone());
        std::thread::spawn(move || {
            for c in l.incoming().flatten() {
                let (h, log) = (h.clone(), log2.clone());
                std::thread::spawn(move || {
                    let mut r = BufReader::new(c.try_clone().unwrap());
                    let mut first = String::new();
                    if r.read_line(&mut first).is_err() {
                        return;
                    }
                    loop {
                        let mut x = String::new();
                        if r.read_line(&mut x).unwrap_or(0) == 0 || x == "\r\n" {
                            break;
                        }
                    }
                    let path = first.split_whitespace().nth(1).unwrap_or("/").to_string();
                    log.lock().unwrap().push(path.clone());
                    let mut w = c;
                    let _ = w.write_all(&h(&path));
                });
            }
        });
        (port, log)
    }
    fn http(code: &str, extra: &str, body: &[u8]) -> Vec<u8> {
        let mut v = format!(
            "HTTP/1.1 {code}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        v.extend_from_slice(body);
        v
    }
    /// A loud skip: the test did not run. Set `ENGINE_REQUIRE_TOOLS=1` (CI does) to make that a failure.
    fn skipped(why: &str) {
        eprintln!("SKIPPED: hub test not run: {why}");
        assert!(
            std::env::var("ENGINE_REQUIRE_TOOLS").map_or(true, |v| v.is_empty()),
            "ENGINE_REQUIRE_TOOLS is set but: {why}"
        );
    }
    fn curl_dl(port: u16) -> Option<Downloader> {
        if !works("curl") {
            return None;
        }
        Some(
            Downloader::new(Hub::HfCompatible(format!("http://127.0.0.1:{port}")))
                .ok()?
                .with_tool(Tool::Curl("curl".into()))
                .with_loopback_http(true)
                .with_token(None),
        )
    }

    #[test]
    fn loopback_is_decided_on_the_real_host_not_on_what_precedes_an_at_sign() {
        for yes in [
            "http://127.0.0.1/x",
            "http://127.0.0.1:8080/x",
            "http://localhost/",
            "http://localhost:1",
            "http://127.0.0.1?q=1",
            "http://localhost#f",
        ] {
            assert!(loopback(yes), "{yes}");
        }
        for no in [
            "http://127.0.0.1@evil.example/x",
            "http://localhost:80@evil.example/",
            "http://evil.example@127.0.0.1/",
            "http://user:pw@127.0.0.1/",
            "http://127.0.0.1\\@evil.example/",
            "http://127.0.0.1.evil.example/",
            "http://localhost.evil.example/",
            "http://127.0.0.1:80x/",
            "http://127.0.0.1:/\n",
            "http://[::1]/",
            "https://127.0.0.1/",
            "http://evil.example/?@127.0.0.1",
            "http://evil.example/127.0.0.1",
        ] {
            assert!(!loopback(no), "{no}");
        }
        if let Ok(d) = Downloader::new(Hub::HuggingFace) {
            let d = d.with_loopback_http(true);
            assert!(matches!(
                d.check_url("http://127.0.0.1@evil.example/x"),
                Err(HubError::InsecureUrl(_))
            ));
        }
    }

    #[test]
    fn df_gets_a_double_dash_before_the_directory() {
        let a = df_args(Path::new("-weird"));
        let a: Vec<_> = a.iter().map(|x| x.to_string_lossy().into_owned()).collect();
        assert_eq!(a, ["-Pk", "--", "-weird"]);
    }

    #[test]
    fn a_missing_size_is_unknown_and_is_refused_for_weights_unless_a_cap_is_set() {
        let (port, _) = serve_raw(|_| http("404 Not Found", "", b"no"));
        let Some(d) = curl_dl(port) else {
            return skipped("curl is not installed or not working");
        };
        let dest = tmp("unknown");
        let spec = RepoSpec::parse("o/n").unwrap();
        let cancel = AtomicBool::new(false);
        let r = d.download_all(
            &spec,
            &[rf("model.safetensors", 0)],
            &dest,
            &cancel,
            &mut |_| {},
        );
        assert!(matches!(r, Err(HubError::UnknownSize(_))), "{r:?}");
        let r = d.download_all(&spec, &[rf("m.gguf", 0)], &dest, &cancel, &mut |_| {});
        assert!(matches!(r, Err(HubError::UnknownSize(_))), "{r:?}");
        // small metadata files of unknown size are bounded by their own implicit cap instead
        let r = d.download_all(&spec, &[rf("config.json", 0)], &dest, &cancel, &mut |_| {});
        assert_eq!(
            r,
            Err(HubError::NotFound),
            "it is attempted (and the server says 404)"
        );
        // with an explicit cap the weights file is attempted too
        let d = d.with_max_filesize(1 << 20);
        let r = d.download_all(
            &spec,
            &[rf("model.safetensors", 0)],
            &dest,
            &cancel,
            &mut |_| {},
        );
        assert_eq!(r, Err(HubError::NotFound));
        fs::remove_dir_all(dest).unwrap();
    }

    #[test]
    fn per_file_and_total_ceilings_and_saturating_sizes() {
        let (port, log) = serve_raw(|_| http("404 Not Found", "", b"no"));
        let Some(d) = curl_dl(port) else {
            return skipped("curl is not installed or not working");
        };
        let dest = tmp("ceil");
        let spec = RepoSpec::parse("o/n").unwrap();
        let cancel = AtomicBool::new(false);
        let small = |n: u64| d_with(port, |d| d.with_max_filesize(n));
        fn d_with(port: u16, f: impl FnOnce(Downloader) -> Downloader) -> Downloader {
            f(curl_dl(port).unwrap())
        }
        let r = small(10).download_all(&spec, &[rf("a.bin", 11)], &dest, &cancel, &mut |_| {});
        assert!(matches!(r, Err(HubError::TooLarge(_))), "{r:?}");
        let r = d_with(port, |d| d.with_max_total(10)).download_all(
            &spec,
            &[rf("a.bin", 6), rf("b.bin", 6)],
            &dest,
            &cancel,
            &mut |_| {},
        );
        assert!(matches!(r, Err(HubError::TooLarge(_))), "{r:?}");
        // sizes that overflow u64 when added: no wrap-around into a small number, no panic
        let huge = [rf("a.bin", u64::MAX), rf("b.bin", u64::MAX)];
        assert_eq!(estimate_bytes(&huge), u64::MAX);
        let r = d.download_all(&spec, &huge, &dest, &cancel, &mut |_| {});
        assert!(matches!(r, Err(HubError::TooLarge(_))), "{r:?}");
        assert!(
            !log.lock().unwrap().iter().any(|p| p.contains("resolve")),
            "refused before any file request"
        );
        // the defaults are real ceilings, not "unlimited"
        assert!(
            d.max_filesize.is_none()
                && d.ceiling() == DEFAULT_MAX_FILE
                && DEFAULT_MAX_TOTAL > DEFAULT_MAX_FILE
        );
        fs::remove_dir_all(dest).unwrap();
    }

    #[test]
    fn a_server_that_sends_more_than_announced_or_never_stops_is_cut_off() {
        let flood = bytes(3_000_000, 4);
        let f2 = flood.clone();
        // no Content-Length: the body just runs until the connection closes
        let (port, _) = serve_raw(move |_| {
            let mut v = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".to_vec();
            v.extend_from_slice(&f2);
            v
        });
        let Some(d) = curl_dl(port) else {
            return skipped("curl is not installed or not working");
        };
        let dest = tmp("flood");
        let spec = RepoSpec::parse("o/n").unwrap();
        let cancel = AtomicBool::new(false);
        let r = d.download_all(&spec, &[rf("a.bin", 1000)], &dest, &cancel, &mut |_| {});
        assert!(
            matches!(r, Err(HubError::Integrity(_))),
            "announced 1000 bytes, got a flood: {r:?}"
        );
        assert!(!dest.join("a.bin").exists() && !dest.join("a.bin.part").exists());
        let d = d.with_max_filesize(100_000);
        let r = d.download_all(&spec, &[rf("u.bin", 0)], &dest, &cancel, &mut |_| {});
        assert!(
            matches!(r, Err(HubError::Integrity(_))),
            "unknown size, capped at 100000: {r:?}"
        );
        assert!(!dest.join("u.bin").exists() && !dest.join("u.bin.part").exists());
        fs::remove_dir_all(dest).unwrap();
    }

    #[test]
    fn a_file_and_its_dot_part_neighbour_do_not_collide() {
        let (a, ap) = (bytes(3000, 1), bytes(2000, 2));
        let files = HashMap::from([
            ("a".to_string(), a.clone()),
            ("a.part".to_string(), ap.clone()),
        ]);
        let (port, _) = serve(files, "[]".into());
        let Some(d) = dl(port) else {
            return skipped("curl is not installed or not working");
        };
        let dest = tmp("collide");
        let list = [
            RemoteFile {
                path: "a".into(),
                size: 3000,
                sha256: Some(sha256_hex(&a)),
            },
            RemoteFile {
                path: "a.part".into(),
                size: 2000,
                sha256: Some(sha256_hex(&ap)),
            },
        ];
        d.download_all(
            &RepoSpec::parse("o/n").unwrap(),
            &list,
            &dest,
            &AtomicBool::new(false),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(fs::read(dest.join("a")).unwrap(), a);
        assert_eq!(fs::read(dest.join("a.part")).unwrap(), ap);
        let mut left: Vec<_> = fs::read_dir(&dest)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(left, ["a", "a.part"], "no stray partial file");
        fs::remove_dir_all(dest).unwrap();
    }

    #[test]
    fn the_wget_fallback_never_gets_the_token() {
        let w = Downloader::new(Hub::HuggingFace)
            .unwrap()
            .with_tool(Tool::Wget("wget".into()))
            .with_token(Some("hf_secret".into()));
        assert_eq!(w.header_text().unwrap(), None);
        let c = Downloader::new(Hub::HuggingFace)
            .unwrap()
            .with_tool(Tool::Curl("curl".into()))
            .with_token(Some("hf_secret".into()));
        assert!(c.header_text().unwrap().unwrap().contains("hf_secret"));
    }

    #[test]
    fn the_revision_is_pinned_to_a_commit_before_listing_and_downloading() {
        let sha = "0123456789abcdef0123456789abcdef01234567";
        let body = bytes(500, 7);
        let (b2, s2) = (body.clone(), sha.to_string());
        let tree = serde_json::json!([{"type":"file","path":"f.json","size":500}]).to_string();
        let (port, log) = serve_raw(move |p| {
            if p.starts_with("/api/models/o/n/revision/main") {
                http(
                    "200 OK",
                    "",
                    format!(r#"{{"sha":"{s2}","siblings":[]}}"#).as_bytes(),
                )
            } else if p.starts_with(&format!("/api/models/o/n/tree/{s2}")) {
                http("200 OK", "", tree.as_bytes())
            } else if p.starts_with(&format!("/o/n/resolve/{s2}/f.json")) {
                http("200 OK", "", &b2)
            } else {
                http("404 Not Found", "", b"moved on")
            }
        });
        let Some(d) = curl_dl(port) else {
            return skipped("curl is not installed or not working");
        };
        let spec = RepoSpec::parse("o/n").unwrap();
        let cancel = AtomicBool::new(false);
        let dest = tmp("pin");
        let files = d.list_files(&spec, &cancel).unwrap();
        d.download_all(&spec, &files, &dest, &cancel, &mut |_| {})
            .unwrap();
        assert_eq!(fs::read(dest.join("f.json")).unwrap(), body);
        let l = log.lock().unwrap().clone();
        assert!(
            l.iter().any(|p| p.contains(&format!("/tree/{sha}"))),
            "{l:?}"
        );
        assert!(
            l.iter().any(|p| p.contains(&format!("/resolve/{sha}/"))),
            "{l:?}"
        );
        assert!(
            !l.iter()
                .any(|p| p.contains("/tree/main") || p.contains("/resolve/main")),
            "the moving name is never used after pinning: {l:?}"
        );
        assert_eq!(
            l.iter().filter(|p| p.contains("/revision/")).count(),
            1,
            "resolved once"
        );
        // a hub without the endpoint is used as before (the revision stays as given)
        assert_eq!(
            d.pinned(
                &RepoSpec {
                    repo: "o/n".into(),
                    revision: sha.into()
                },
                &cancel
            )
            .revision,
            sha
        );
        fs::remove_dir_all(dest).unwrap();
    }

    #[test]
    fn a_redirect_to_plain_http_is_refused() {
        let (target_port, target_log) =
            serve_raw(|_| http("200 OK", "", b"you should never see this"));
        let loc = format!("Location: http://127.0.0.1:{target_port}/target\r\n");
        let (port, _) = serve_raw(move |_| http("302 Found", &loc, b""));
        let Some(d) = curl_dl(port) else {
            return skipped("curl is not installed or not working");
        };
        let dest = tmp("redir");
        let r = d.download_all(
            &RepoSpec::parse("o/n").unwrap(),
            &[rf("f.bin", 10)],
            &dest,
            &AtomicBool::new(false),
            &mut |_| {},
        );
        assert!(r.is_err(), "{r:?}");
        assert!(
            target_log.lock().unwrap().is_empty(),
            "the http target must never be contacted: {:?}",
            target_log.lock().unwrap()
        );
        assert!(!dest.join("f.bin").exists());
        fs::remove_dir_all(dest).unwrap();
    }
}
