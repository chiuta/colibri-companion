//! Pipeline-parallel inference over a LAN ("mesh").
//!
//! A model too big for one machine is cut into contiguous layer ranges, one per machine. The
//! coordinator embeds the token, runs its own layers, hands the hidden state (`hidden` floats, a
//! few tens of KB) to the next machine, and so on; the last machine returns it and the
//! coordinator applies the output head. Each machine keeps the KV cache of its own layers only.
//! A prompt is *pipelined*: while machine 2 works on token 5, machine 1 already works on token 6.
//!
//! * **Every machine opens the same model file** (memory-mapped, so only the layers it is given are
//!   ever paged in). A fingerprint of the architecture and a few weights is checked on connect.
//! * **Authentication**: a pre-shared key. Mutual HMAC-SHA256 challenge/response, then every frame
//!   carries a truncated HMAC over its direction, sequence number, type and payload, so a frame
//!   cannot be forged, replayed, reordered or altered. Nothing is *encrypted*: use a trusted LAN
//!   or a VPN if the hidden states themselves are sensitive.
//! * **Partitioning** follows the machines' speeds (thread counts unless overridden) and optional
//!   per-layer costs ([`partition`]).
//! * **Failure recovery**: when a worker disappears or errs, it is dropped, the layers are
//!   re-divided among the survivors (the coordinator can always run everything alone), the caches
//!   are rebuilt by replaying the tokens, and the call is retried.
//! * **Discovery**: workers answer a UDP probe ([`discover`]).
//!
//! Verified over loopback TCP only; real networks, firewalls and heterogeneous machines are untested.

use crate::family::LanguageModel;
use crate::hub::Sha256;
use crate::model::{Model, ModelError, State, f32_to_f16};
use crate::quant::f16_to_f32;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs, UdpSocket};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Default UDP port workers answer discovery probes on.
pub const DISCOVERY_PORT: u16 = 47813;
const MAGIC: &[u8; 5] = b"CMSH1";
const MAX_FRAME: usize = 16 << 20;
const MAC_LEN: usize = 16;
const MAX_SESSIONS: usize = 32;
/// A peer that has not proved it knows the key within this time is cut off.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(3);
/// Connections that are still in the handshake, over the whole worker.
const MAX_PENDING_HANDSHAKES: usize = 16;
/// Open connections from one IP address (a coordinator needs one).
const MAX_CONNS_PER_IP: usize = 4;
/// Open connections over the whole worker.
const MAX_CONNS_TOTAL: usize = 64;
/// UDP discovery probes are padded to this length, and a reply is never longer than its probe.
const PROBE_LEN: usize = 128;
/// Discovery replies: at most this many per second overall, and one per source per interval.
const DISCOVERY_PER_SECOND: usize = 20;
const DISCOVERY_PER_SOURCE: Duration = Duration::from_millis(250);

// ------------------------------------------------------------------ errors and config

#[derive(Debug)]
pub enum MeshError {
    Config(String),
    Io(String),
    /// The peer does not know the shared key.
    Auth,
    /// The peer runs a different model.
    Model(String),
    Protocol(String),
    /// A worker reported an error.
    Worker(String),
    Model_(ModelError),
}
impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshError::Config(s) => write!(f, "mesh configuration: {s}"),
            MeshError::Io(s) => write!(f, "mesh network: {s}"),
            MeshError::Auth => write!(f, "mesh authentication failed (wrong shared key?)"),
            MeshError::Model(s) => write!(f, "mesh model mismatch: {s}"),
            MeshError::Protocol(s) => write!(f, "mesh protocol: {s}"),
            MeshError::Worker(s) => write!(f, "mesh worker: {s}"),
            MeshError::Model_(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for MeshError {}
impl From<io::Error> for MeshError {
    fn from(e: io::Error) -> Self {
        MeshError::Io(e.to_string())
    }
}
impl From<ModelError> for MeshError {
    fn from(e: ModelError) -> Self {
        MeshError::Model_(e)
    }
}
impl From<MeshError> for ModelError {
    fn from(e: MeshError) -> Self {
        match e {
            MeshError::Model_(m) => m,
            o => ModelError::Backend(o.to_string()),
        }
    }
}

/// How hidden states travel. `F16` halves the traffic and costs a little accuracy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wire {
    F32,
    F16,
}

#[derive(Clone)]
pub struct MeshConfig {
    /// Pre-shared secret (at least 8 bytes; 16+ random ones recommended).
    pub key: Vec<u8>,
    pub wire: Wire,
    /// Network timeout for one step.
    pub timeout: Duration,
    /// Relative speed of each worker, in `connect` order (default: its reported thread count).
    pub speeds: Option<Vec<f64>>,
    /// Relative speed of the coordinator (default: its thread count; `0` = it only embeds and finishes).
    pub local_speed: Option<f64>,
    /// Relative cost of each layer (default: all equal) — e.g. heavier for mixture-of-experts layers.
    pub costs: Option<Vec<f64>>,
}
impl std::fmt::Debug for MeshConfig {
    // The pre-shared key must never reach a log, a panic message or a bug report.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshConfig")
            .field("key", &format_args!("<redacted, {} bytes>", self.key.len()))
            .field("wire", &self.wire)
            .field("timeout", &self.timeout)
            .field("speeds", &self.speeds)
            .field("local_speed", &self.local_speed)
            .field("costs", &self.costs)
            .finish()
    }
}

/// Advice about a shared key, or `None` when it looks fine. The handshake sends an HMAC of the
/// key in the clear, so anyone who can see one handshake can try guesses offline: a short or
/// predictable passphrase is brute-forced quickly. Use 16 or more random bytes.
pub fn key_strength_warning(key: &[u8]) -> Option<String> {
    if key.len() < 8 {
        return Some("the key is shorter than 8 bytes and will be refused".into());
    }
    if key.len() < 16 {
        return Some(format!(
            "the key is {} bytes; use 16 or more random bytes (a handshake can be brute-forced offline)",
            key.len()
        ));
    }
    let mut seen = [false; 256];
    for &b in key {
        seen[b as usize] = true;
    }
    let distinct = seen.iter().filter(|&&x| x).count();
    if distinct < 6 {
        return Some(format!(
            "the key uses only {distinct} distinct byte values; use random bytes"
        ));
    }
    None
}

impl Default for MeshConfig {
    fn default() -> Self {
        MeshConfig {
            key: vec![],
            wire: Wire::F32,
            timeout: Duration::from_secs(120),
            speeds: None,
            local_speed: None,
            costs: None,
        }
    }
}
impl MeshConfig {
    pub fn with_key(key: impl Into<Vec<u8>>) -> MeshConfig {
        MeshConfig {
            key: key.into(),
            ..Default::default()
        }
    }
    fn check(&self) -> Result<(), MeshError> {
        if self.key.len() < 8 {
            Err(MeshError::Config(
                "the shared key must be at least 8 bytes".into(),
            ))
        } else {
            Ok(())
        }
    }
}

// ------------------------------------------------------------------ crypto helpers

/// HMAC-SHA256 over the concatenation of `parts`.
pub fn hmac_sha256(key: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut k = [0u8; 64];
    if key.len() > 64 {
        let mut s = Sha256::new();
        s.update(key);
        k[..32].copy_from_slice(&s.finish());
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let (mut ip, mut op) = ([0x36u8; 64], [0x5cu8; 64]);
    for i in 0..64 {
        ip[i] ^= k[i];
        op[i] ^= k[i];
    }
    let mut inner = Sha256::new();
    inner.update(&ip);
    for p in parts {
        inner.update(p);
    }
    let ih = inner.finish();
    let mut outer = Sha256::new();
    outer.update(&op);
    outer.update(&ih);
    outer.finish()
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut out = [0u8; N];
    #[cfg(unix)]
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut out))
        .is_ok()
    {
        return out;
    }
    static CTR: AtomicU32 = AtomicU32::new(0);
    for (i, c) in out.chunks_mut(8).enumerate() {
        let mut h = RandomState::new().build_hasher();
        h.write_u32(CTR.fetch_add(1, Ordering::Relaxed));
        h.write_usize(i);
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        );
        let v = h.finish().to_le_bytes();
        c.copy_from_slice(&v[..c.len()]);
    }
    out
}

/// Identifies the architecture and a few weights, so machines with different models refuse each other.
pub fn model_fingerprint(m: &Model) -> [u8; 32] {
    let s = &m.spec;
    let mut h = Sha256::new();
    h.update(
        format!(
            "{}|{}|{}|{}|{}|{}|{}|{}",
            s.layers, s.hidden, s.heads, s.kv_heads, s.head_dim, s.ffn, s.vocab, s.context
        )
        .as_bytes(),
    );
    let hid = s.hidden as usize;
    let (mut row, mut tmp) = (vec![0.0f32; hid], Vec::new());
    for r in [0usize, (s.vocab as usize).saturating_sub(1)] {
        if m.w.embed.row_into(r, hid, &mut row, &mut tmp).is_ok() {
            for v in &row {
                h.update(&v.to_bits().to_le_bytes());
            }
        }
    }
    for v in m.w.final_norm.iter().chain(
        m.w.layers
            .first()
            .map(|l| l.attn_norm.as_slice())
            .unwrap_or(&[]),
    ) {
        h.update(&v.to_bits().to_le_bytes());
    }
    h.finish()
}

fn hex32(b: &[u8]) -> String {
    crate::hub::hex(b)
}
fn unhex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 || !s.is_ascii() {
        return None;
    }
    let mut o = [0u8; 32];
    for (i, b) in o.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(o)
}

// ------------------------------------------------------------------ framed, authenticated channel

const T_INFO: u8 = 1;
const T_ASSIGN: u8 = 2;
const T_ACK: u8 = 3;
const T_ERR: u8 = 4;
const T_STEP: u8 = 5;
const T_HIDDEN: u8 = 6;
const T_TRUNC: u8 = 7;
const T_CLOSE: u8 = 8;

struct Chan {
    s: TcpStream,
    key: [u8; 32],
    tx_dir: u8,
    tx_seq: u64,
    rx_seq: u64,
}
impl Chan {
    fn mac(&self, dir: u8, seq: u64, ty: u8, payload: &[u8]) -> [u8; 32] {
        hmac_sha256(&self.key, &[&[dir], &seq.to_le_bytes(), &[ty], payload])
    }
    fn send(&mut self, ty: u8, payload: &[u8]) -> io::Result<()> {
        if payload.len() > MAX_FRAME {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "frame too large",
            ));
        }
        let mac = self.mac(self.tx_dir, self.tx_seq, ty, payload);
        self.tx_seq += 1;
        let mut buf = Vec::with_capacity(5 + payload.len() + MAC_LEN);
        buf.extend_from_slice(&((1 + payload.len() + MAC_LEN) as u32).to_le_bytes());
        buf.push(ty);
        buf.extend_from_slice(payload);
        buf.extend_from_slice(&mac[..MAC_LEN]);
        self.s.write_all(&buf)?;
        self.s.flush()
    }
    fn recv(&mut self) -> io::Result<(u8, Vec<u8>)> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        let mut l = [0u8; 4];
        self.s.read_exact(&mut l)?;
        let n = u32::from_le_bytes(l) as usize;
        if !(1 + MAC_LEN..=1 + MAX_FRAME + MAC_LEN).contains(&n) {
            return Err(bad("bad frame length"));
        }
        let mut body = vec![0u8; n];
        self.s.read_exact(&mut body)?;
        let (head, mac) = body.split_at(n - MAC_LEN);
        let want = self.mac(3 - self.tx_dir, self.rx_seq, head[0], &head[1..]);
        if !ct_eq(&want[..MAC_LEN], mac) {
            return Err(bad("frame authentication failed"));
        }
        self.rx_seq += 1;
        Ok((head[0], head[1..].to_vec()))
    }
}

fn handshake(
    mut s: TcpStream,
    key: &[u8],
    coordinator: bool,
    timeout: Duration,
) -> Result<Chan, MeshError> {
    s.set_read_timeout(Some(timeout))?;
    s.set_write_timeout(Some(timeout))?;
    let _ = s.set_nodelay(true);
    let auth = |e: io::Error| {
        if matches!(
            e.kind(),
            io::ErrorKind::UnexpectedEof
                | io::ErrorKind::ConnectionReset
                | io::ErrorKind::BrokenPipe
        ) {
            MeshError::Auth
        } else {
            MeshError::from(e)
        }
    };
    let (nw, nc);
    if coordinator {
        let mut hello = [0u8; 5 + 16];
        s.read_exact(&mut hello).map_err(MeshError::from)?;
        if &hello[..5] != MAGIC {
            return Err(MeshError::Protocol("not a mesh worker".into()));
        }
        nw = <[u8; 16]>::try_from(&hello[5..]).unwrap();
        nc = random_bytes::<16>();
        let proof = hmac_sha256(key, &[b"C", &nw, &nc]);
        s.write_all(&[&nc[..], &proof[..]].concat())?;
        let mut back = [0u8; 32];
        s.read_exact(&mut back).map_err(auth)?;
        if !ct_eq(&back, &hmac_sha256(key, &[b"W", &nw, &nc])) {
            return Err(MeshError::Auth);
        }
    } else {
        nw = random_bytes::<16>();
        s.write_all(&[&MAGIC[..], &nw[..]].concat())?;
        let mut m = [0u8; 48];
        s.read_exact(&mut m).map_err(MeshError::from)?;
        nc = <[u8; 16]>::try_from(&m[..16]).unwrap();
        if !ct_eq(&m[16..], &hmac_sha256(key, &[b"C", &nw, &nc])) {
            return Err(MeshError::Auth);
        }
        s.write_all(&hmac_sha256(key, &[b"W", &nw, &nc]))?;
    }
    Ok(Chan {
        s,
        key: hmac_sha256(key, &[b"session", &nw, &nc]),
        tx_dir: if coordinator { 1 } else { 2 },
        tx_seq: 0,
        rx_seq: 0,
    })
}

fn encode(x: &[f32], wire: Wire) -> Vec<u8> {
    match wire {
        Wire::F32 => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Wire::F16 => x
            .iter()
            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
            .collect(),
    }
}
fn decode(b: &[u8], wire: Wire, n: usize) -> Result<Vec<f32>, MeshError> {
    let w = if wire == Wire::F32 { 4 } else { 2 };
    if b.len() != n * w {
        return Err(MeshError::Protocol(format!(
            "hidden state of {} bytes, expected {}",
            b.len(),
            n * w
        )));
    }
    Ok(match wire {
        Wire::F32 => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Wire::F16 => b
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    })
}
fn u32_at(b: &[u8], at: usize) -> Result<u32, MeshError> {
    b.get(at..at + 4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .ok_or_else(|| MeshError::Protocol("short message".into()))
}

// ------------------------------------------------------------------ partitioning

/// Divide `costs.len()` layers into one contiguous range per entry of `speeds`, proportional to
/// the speeds (by cumulative cost). Every machine with a positive speed gets at least one layer
/// when there are enough layers; machines with a zero, negative or NaN speed get an empty range.
/// The ranges are in order and cover `0..costs.len()` exactly (when any speed is positive).
pub fn partition(costs: &[f64], speeds: &[f64]) -> Vec<Range<usize>> {
    let n = costs.len();
    let ok: Vec<usize> = (0..speeds.len())
        .filter(|&i| speeds[i].is_finite() && speeds[i] > 0.0)
        .collect();
    let mut out = vec![0..0; speeds.len()];
    if ok.is_empty() || n == 0 {
        return out;
    }
    let costs: Vec<f64> = costs
        .iter()
        .map(|&c| if c.is_finite() && c > 0.0 { c } else { 1e-9 })
        .collect();
    let mut cum = vec![0.0; n + 1];
    for i in 0..n {
        cum[i + 1] = cum[i] + costs[i];
    }
    let used: Vec<usize> = if ok.len() > n {
        let mut by: Vec<usize> = ok.clone();
        by.sort_by(|&a, &b| speeds[b].partial_cmp(&speeds[a]).unwrap().then(a.cmp(&b)));
        by.truncate(n);
        by.sort_unstable();
        by
    } else {
        ok
    };
    let (total, tspeed) = (cum[n], used.iter().map(|&i| speeds[i]).sum::<f64>());
    let (mut start, mut acc) = (0usize, 0.0);
    for (k, &i) in used.iter().enumerate() {
        acc += speeds[i];
        let left = used.len() - 1 - k;
        let cut = if left == 0 {
            n
        } else {
            let target = total * acc / tspeed;
            let best = (0..=n)
                .min_by(|&a, &b| {
                    (cum[a] - target)
                        .abs()
                        .partial_cmp(&(cum[b] - target).abs())
                        .unwrap()
                        .then(a.cmp(&b))
                })
                .unwrap_or(n);
            best.clamp(start + 1, n - left)
        };
        out[i] = start..cut;
        start = cut;
    }
    out
}

// ------------------------------------------------------------------ worker

/// A running worker: serves layer ranges to a coordinator. Dropping it stops it.
pub struct Worker {
    addr: SocketAddr,
    discovery: Option<SocketAddr>,
    stop: Arc<AtomicBool>,
    conns: Arc<Mutex<Vec<TcpStream>>>,
    joins: Vec<JoinHandle<()>>,
}
/// Connection accounting for the accept loop: bounds how much an unauthenticated peer can hold.
#[derive(Default)]
struct Limits {
    pending: AtomicUsize,
    per_ip: Mutex<std::collections::HashMap<std::net::IpAddr, usize>>,
    total: AtomicUsize,
}
impl Limits {
    /// Reserve a slot for a new connection from `ip`, or `None` when a cap is reached.
    fn admit(self: &Arc<Self>, ip: std::net::IpAddr) -> Option<(ConnGuard, PendingGuard)> {
        if self.total.load(Ordering::SeqCst) >= MAX_CONNS_TOTAL
            || self.pending.load(Ordering::SeqCst) >= MAX_PENDING_HANDSHAKES
        {
            return None;
        }
        {
            let mut m = self.per_ip.lock().ok()?;
            let n = m.entry(ip).or_insert(0);
            if *n >= MAX_CONNS_PER_IP {
                return None;
            }
            *n += 1;
        }
        self.total.fetch_add(1, Ordering::SeqCst);
        self.pending.fetch_add(1, Ordering::SeqCst);
        Some((
            ConnGuard {
                l: self.clone(),
                ip,
            },
            PendingGuard {
                l: Some(self.clone()),
            },
        ))
    }
}
struct ConnGuard {
    l: Arc<Limits>,
    ip: std::net::IpAddr,
}
impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.l.total.fetch_sub(1, Ordering::SeqCst);
        if let Ok(mut m) = self.l.per_ip.lock()
            && let Some(n) = m.get_mut(&self.ip)
        {
            *n = n.saturating_sub(1);
            if *n == 0 {
                m.remove(&self.ip);
            }
        }
    }
}
/// Released as soon as the handshake ends (successfully or not), or when the connection is dropped.
struct PendingGuard {
    l: Option<Arc<Limits>>,
}
impl PendingGuard {
    fn release(&mut self) {
        if let Some(l) = self.l.take() {
            l.pending.fetch_sub(1, Ordering::SeqCst);
        }
    }
}
impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.release();
    }
}

impl Worker {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
    /// The UDP address answering [`discover`] probes, if discovery was enabled.
    pub fn discovery_addr(&self) -> Option<SocketAddr> {
        self.discovery
    }
    /// Stop serving and cut every connection (a coordinator sees a network failure).
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Ok(c) = self.conns.lock() {
            for s in c.iter() {
                let _ = s.shutdown(std::net::Shutdown::Both);
            }
        }
        for j in self.joins.drain(..) {
            let _ = j.join();
        }
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Start a worker on `bind` (for example `"0.0.0.0:7070"`; port 0 picks one). With
/// `discovery_port`, it also answers UDP discovery probes there (`Some(0)` picks a port).
pub fn spawn_worker(
    model: Arc<Model>,
    bind: impl ToSocketAddrs,
    cfg: MeshConfig,
    discovery_port: Option<u16>,
) -> Result<Worker, MeshError> {
    cfg.check()?;
    let listener = TcpListener::bind(bind)?;
    listener.set_nonblocking(true)?;
    let addr = listener.local_addr()?;
    let stop = Arc::new(AtomicBool::new(false));
    let conns: Arc<Mutex<Vec<TcpStream>>> = Arc::default();
    let fp = model_fingerprint(&model);
    let mut joins = vec![];
    let mut discovery = None;
    if let Some(p) = discovery_port {
        let u = UdpSocket::bind(("0.0.0.0", p))?;
        u.set_read_timeout(Some(Duration::from_millis(50)))?;
        discovery = Some(u.local_addr()?);
        let (stop, model) = (stop.clone(), model.clone());
        joins.push(thread::spawn(move || {
            let reply = format!(
                "CMSH! {} {} {} {}",
                addr.port(),
                hex32(&fp),
                crate::pool::global().threads(),
                model.spec.layers
            );
            let mut buf = [0u8; 512];
            let mut recent: std::collections::HashMap<std::net::IpAddr, Instant> =
                Default::default();
            let (mut window, mut in_window) = (Instant::now(), 0usize);
            while !stop.load(Ordering::SeqCst) {
                let Ok((n, from)) = u.recv_from(&mut buf) else {
                    continue;
                };
                // A probe must start with the magic and be at least as long as the reply: a spoofed
                // source address then gets back no more bytes than it sent (no amplification).
                if n < reply.len() || n < PROBE_LEN || &buf[..5] != b"CMSH?" {
                    continue;
                }
                let now = Instant::now();
                if now.duration_since(window) >= Duration::from_secs(1) {
                    window = now;
                    in_window = 0;
                }
                if in_window >= DISCOVERY_PER_SECOND {
                    continue;
                }
                if recent.len() > 1024 {
                    recent.retain(|_, t| now.duration_since(*t) < DISCOVERY_PER_SOURCE);
                }
                if recent
                    .get(&from.ip())
                    .is_some_and(|t| now.duration_since(*t) < DISCOVERY_PER_SOURCE)
                {
                    continue;
                }
                recent.insert(from.ip(), now);
                in_window += 1;
                let _ = u.send_to(reply.as_bytes(), from);
            }
        }));
    }
    let (stop2, conns2) = (stop.clone(), conns.clone());
    let limits = Arc::new(Limits::default());
    joins.push(thread::spawn(move || {
        let mut kids: Vec<JoinHandle<()>> = vec![];
        while !stop2.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((s, peer)) => {
                    let _ = s.set_nonblocking(false);
                    let Some((conn_guard, pending)) = limits.admit(peer.ip()) else {
                        let _ = s.shutdown(std::net::Shutdown::Both); // over a cap: refuse without a thread
                        continue;
                    };
                    if let (Ok(c), Ok(mut g)) = (s.try_clone(), conns2.lock()) {
                        g.retain(|x| x.peer_addr().is_ok());
                        g.push(c);
                    }
                    let (model, cfg, stop) = (model.clone(), cfg.clone(), stop2.clone());
                    kids.push(thread::spawn(move || {
                        let _conn = conn_guard;
                        let _ = serve_conn(&model, s, &cfg, &stop, fp, pending);
                    }));
                    kids.retain(|k| !k.is_finished());
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(20))
                }
                Err(_) => thread::sleep(Duration::from_millis(100)),
            }
        }
        if let Ok(c) = conns2.lock() {
            for s in c.iter() {
                let _ = s.shutdown(std::net::Shutdown::Both);
            }
        }
        for k in kids {
            let _ = k.join();
        }
    }));
    Ok(Worker {
        addr,
        discovery,
        stop,
        conns,
        joins,
    })
}

fn serve_conn(
    model: &Model,
    s: TcpStream,
    cfg: &MeshConfig,
    stop: &AtomicBool,
    fp: [u8; 32],
    pending: PendingGuard,
) -> Result<(), MeshError> {
    let guard = s.try_clone()?;
    let r = serve_authenticated(model, s, cfg, stop, fp, pending);
    let _ = guard.shutdown(std::net::Shutdown::Both); // the accept loop keeps a clone: close for real
    r
}

fn serve_authenticated(
    model: &Model,
    s: TcpStream,
    cfg: &MeshConfig,
    stop: &AtomicBool,
    fp: [u8; 32],
    mut pending: PendingGuard,
) -> Result<(), MeshError> {
    let hs = handshake(s, &cfg.key, false, HANDSHAKE_TIMEOUT);
    pending.release(); // authenticated or not, this connection no longer counts as a pending handshake
    let mut ch = hs?;
    // after the handshake a coordinator may idle between turns; wake up now and then to see `stop`
    ch.s.set_read_timeout(Some(Duration::from_millis(250)))?;
    let h = model.spec.hidden as usize;
    let mut info = fp.to_vec();
    info.extend_from_slice(&(crate::pool::global().threads() as u32).to_le_bytes());
    info.extend_from_slice(&(model.spec.layers as u32).to_le_bytes());
    ch.send(T_INFO, &info)?;
    let (mut range, mut wire): (Option<Range<usize>>, Wire) = (None, Wire::F32);
    let mut sessions: Vec<(u32, State)> = vec![];
    loop {
        if stop.load(Ordering::SeqCst) {
            return Ok(());
        }
        let (ty, p) = match ch.recv() {
            Ok(m) => m,
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                continue;
            }
            Err(_) => return Ok(()),
        };
        let reply: Result<(u8, Vec<u8>), MeshError> = (|| match ty {
            T_ASSIGN => {
                if p.len() != 41 {
                    return Err(MeshError::Protocol("bad assign".into()));
                }
                if !ct_eq(&p[..32], &fp) {
                    return Err(MeshError::Model(
                        "this worker holds a different model".into(),
                    ));
                }
                let (a, b) = (u32_at(&p, 32)? as usize, u32_at(&p, 36)? as usize);
                if a > b || b > model.spec.layers as usize {
                    return Err(MeshError::Protocol(format!(
                        "layer range {a}..{b} is outside the model"
                    )));
                }
                wire = if p[40] == 1 { Wire::F16 } else { Wire::F32 };
                range = Some(a..b);
                Ok((T_ACK, vec![]))
            }
            T_STEP => {
                let r = range
                    .clone()
                    .ok_or_else(|| MeshError::Protocol("step before assign".into()))?;
                let (id, pos) = (u32_at(&p, 0)?, u32_at(&p, 4)? as usize);
                let mut x = decode(&p[8..], wire, h)?;
                let i = match sessions.iter().position(|(k, _)| *k == id) {
                    Some(i) => i,
                    None => {
                        if sessions.len() >= MAX_SESSIONS {
                            sessions.remove(0);
                        }
                        sessions.push((id, model.new_state()));
                        sessions.len() - 1
                    }
                };
                let st = &mut sessions[i].1;
                if pos == 0 && st.pos > 0 {
                    st.reset();
                }
                if st.pos != pos {
                    return Err(MeshError::Protocol(format!(
                        "session {id} is at position {}, asked for {pos}",
                        st.pos
                    )));
                }
                model.run_layers(st, r, &mut x, pos)?;
                model.finish_token(st);
                Ok((T_HIDDEN, encode(&x, wire)))
            }
            T_TRUNC => {
                let (id, pos) = (u32_at(&p, 0)?, u32_at(&p, 4)? as usize);
                if let Some((_, st)) = sessions.iter_mut().find(|(k, _)| *k == id) {
                    st.truncate(pos)?;
                }
                Ok((T_ACK, vec![]))
            }
            T_CLOSE => {
                let id = u32_at(&p, 0)?;
                sessions.retain(|(k, _)| *k != id);
                Ok((T_ACK, vec![]))
            }
            _ => Err(MeshError::Protocol(format!("unknown message type {ty}"))),
        })();
        match reply {
            Ok((t, b)) => ch.send(t, &b)?,
            Err(e) => ch.send(T_ERR, e.to_string().as_bytes())?,
        }
    }
}

// ------------------------------------------------------------------ discovery

/// A worker found by [`discover`].
#[derive(Clone, Debug)]
pub struct Found {
    pub addr: SocketAddr,
    pub fingerprint: [u8; 32],
    pub threads: usize,
    pub layers: usize,
}

/// Probe `targets` (UDP; use `255.255.255.255:47813` for a LAN broadcast) and collect the workers
/// that answer within `wait`. Only workers with the model whose `fingerprint` is given are kept.
pub fn discover(
    targets: &[SocketAddr],
    wait: Duration,
    fingerprint: Option<[u8; 32]>,
) -> Vec<Found> {
    let Ok(u) = UdpSocket::bind(("0.0.0.0", 0)) else {
        return vec![];
    };
    let _ = u.set_broadcast(true);
    let _ = u.set_read_timeout(Some(Duration::from_millis(50)));
    let mut probe = [0u8; PROBE_LEN];
    probe[..5].copy_from_slice(b"CMSH?");
    for t in targets {
        let _ = u.send_to(&probe, t);
    }
    let (end, mut out, mut buf) = (Instant::now() + wait, Vec::<Found>::new(), [0u8; 256]);
    while Instant::now() < end {
        let Ok((n, from)) = u.recv_from(&mut buf) else {
            continue;
        };
        let text = String::from_utf8_lossy(&buf[..n]).to_string();
        let f: Vec<&str> = text.split(' ').collect();
        if f.len() != 5 || f[0] != "CMSH!" {
            continue;
        }
        let (Ok(port), Some(fp), Ok(threads), Ok(layers)) = (
            f[1].parse::<u16>(),
            unhex32(f[2]),
            f[3].parse::<usize>(),
            f[4].parse::<usize>(),
        ) else {
            continue;
        };
        if fingerprint.is_some_and(|w| w != fp) {
            continue;
        }
        let addr = SocketAddr::new(from.ip(), port);
        if !out.iter().any(|o| o.addr == addr) {
            out.push(Found {
                addr,
                fingerprint: fp,
                threads,
                layers,
            });
        }
    }
    out
}

// ------------------------------------------------------------------ coordinator

struct Remote {
    id: usize,
    addr: SocketAddr,
    ch: Chan,
    threads: usize,
    range: Range<usize>,
}

struct Inner {
    remotes: Vec<Remote>,
    local: Range<usize>,
    epoch: u64,
    next_id: usize,
}

/// A position in a conversation spread over the mesh. Make one with [`Mesh::new_state`].
pub struct MeshState {
    id: u32,
    local: State,
    tokens: Vec<usize>,
    epoch: u64,
}
impl MeshState {
    pub fn position(&self) -> usize {
        self.tokens.len()
    }
    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }
}

/// One machine in the layout, in pipeline order (the coordinator is first).
#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub name: String,
    pub layers: Range<usize>,
    pub threads: usize,
}

struct Fail {
    worker: Option<usize>,
    err: MeshError,
}
type Item = Result<(usize, Vec<f32>), Fail>;

/// The coordinator: holds the whole model file (for embeddings, head and its own layers) and the
/// connections to the workers.
pub struct Mesh {
    model: Arc<Model>,
    cfg: MeshConfig,
    fp: [u8; 32],
    inner: Mutex<Inner>,
    next_session: AtomicU32,
}

impl Mesh {
    /// Connect to `workers`, check they run the same model, and divide the layers among everyone.
    pub fn connect(
        model: Arc<Model>,
        workers: &[SocketAddr],
        cfg: MeshConfig,
    ) -> Result<Mesh, MeshError> {
        cfg.check()?;
        let m = Mesh {
            fp: model_fingerprint(&model),
            model,
            cfg,
            inner: Mutex::new(Inner {
                remotes: vec![],
                local: 0..0,
                epoch: 0,
                next_id: 0,
            }),
            next_session: AtomicU32::new(1),
        };
        {
            let mut g = m
                .inner
                .lock()
                .map_err(|_| MeshError::Protocol("poisoned".into()))?;
            for a in workers {
                m.dial(&mut g, *a)?;
            }
            m.relayout(&mut g).map_err(|f| f.err)?;
        }
        Ok(m)
    }

    /// Add a worker (or bring a restarted one back) and re-divide the layers.
    pub fn add_worker(&self, addr: SocketAddr) -> Result<(), MeshError> {
        let mut g = self
            .inner
            .lock()
            .map_err(|_| MeshError::Protocol("poisoned".into()))?;
        self.dial(&mut g, addr)?;
        self.relayout(&mut g).map_err(|f| f.err)
    }

    fn dial(&self, g: &mut Inner, addr: SocketAddr) -> Result<(), MeshError> {
        let s = TcpStream::connect_timeout(&addr, self.cfg.timeout.min(Duration::from_secs(10)))?;
        let mut ch = handshake(s, &self.cfg.key, true, self.cfg.timeout)?;
        let (ty, p) = ch.recv().map_err(|e| MeshError::Io(e.to_string()))?;
        if ty != T_INFO || p.len() != 40 {
            return Err(MeshError::Protocol("bad worker info".into()));
        }
        if !ct_eq(&p[..32], &self.fp) {
            return Err(MeshError::Model(format!("{addr} holds a different model")));
        }
        let threads = u32_at(&p, 32)? as usize;
        g.remotes.push(Remote {
            id: g.next_id,
            addr,
            ch,
            threads,
            range: 0..0,
        });
        g.next_id += 1;
        Ok(())
    }

    /// Recompute the layout and tell every worker. Bumps the epoch, so every state is rebuilt lazily.
    fn relayout(&self, g: &mut Inner) -> Result<(), Fail> {
        let layers = self.model.spec.layers as usize;
        let costs = match &self.cfg.costs {
            Some(c) if c.len() == layers => c.clone(),
            _ => vec![1.0; layers],
        };
        let mut speeds = vec![
            self.cfg
                .local_speed
                .unwrap_or(crate::pool::global().threads() as f64),
        ];
        for (i, r) in g.remotes.iter().enumerate() {
            speeds.push(
                self.cfg
                    .speeds
                    .as_ref()
                    .and_then(|s| s.get(r.id).or(s.get(i)))
                    .copied()
                    .unwrap_or(r.threads as f64),
            );
        }
        let mut parts = partition(&costs, &speeds);
        if parts.iter().all(|r| r.is_empty()) {
            parts = vec![0..0; g.remotes.len() + 1];
            parts[0] = 0..layers;
        }
        g.local = parts[0].clone();
        let wire = self.cfg.wire;
        for (r, range) in g.remotes.iter_mut().zip(&parts[1..]) {
            r.range = range.clone();
            let mut p = self.fp.to_vec();
            p.extend_from_slice(&(range.start as u32).to_le_bytes());
            p.extend_from_slice(&(range.end as u32).to_le_bytes());
            p.push(u8::from(wire == Wire::F16));
            let fail = |e: MeshError| Fail {
                worker: Some(r.id),
                err: e,
            };
            r.ch.send(T_ASSIGN, &p).map_err(|e| fail(e.into()))?;
            match r.ch.recv().map_err(|e| fail(e.into()))? {
                (T_ACK, _) => {}
                (T_ERR, m) => {
                    return Err(fail(MeshError::Worker(String::from_utf8_lossy(&m).into())));
                }
                _ => {
                    return Err(fail(MeshError::Protocol(
                        "unexpected reply to assign".into(),
                    )));
                }
            }
        }
        g.epoch += 1;
        Ok(())
    }

    /// The machines and their layers, in pipeline order.
    pub fn layout(&self) -> Vec<NodeInfo> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut v = vec![NodeInfo {
            name: "coordinator".into(),
            layers: g.local.clone(),
            threads: crate::pool::global().threads(),
        }];
        v.extend(g.remotes.iter().map(|r| NodeInfo {
            name: r.addr.to_string(),
            layers: r.range.clone(),
            threads: r.threads,
        }));
        v
    }

    /// Number of connected workers.
    pub fn workers(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remotes
            .len()
    }

    pub fn new_state(&self) -> MeshState {
        MeshState {
            id: self.next_session.fetch_add(1, Ordering::Relaxed),
            local: self.model.new_state(),
            tokens: vec![],
            epoch: 0,
        }
    }

    /// Forget the conversation (workers free its cache too).
    pub fn reset(&self, st: &mut MeshState) {
        if let Ok(mut g) = self.inner.lock() {
            self.close_remote(&mut g, st.id);
        }
        st.id = self.next_session.fetch_add(1, Ordering::Relaxed);
        st.local.reset();
        st.tokens.clear();
    }

    fn close_remote(&self, g: &mut Inner, id: u32) {
        for r in g.remotes.iter_mut() {
            if r.ch.send(T_CLOSE, &id.to_le_bytes()).is_ok() {
                let _ = r.ch.recv();
            }
        }
    }

    /// Rewind the conversation to `keep` tokens everywhere; if a cache cannot rewind, start over.
    pub fn truncate(&self, st: &mut MeshState, keep: usize) -> Result<(), MeshError> {
        if keep >= st.tokens.len() {
            return Ok(());
        }
        let mut g = self
            .inner
            .lock()
            .map_err(|_| MeshError::Protocol("poisoned".into()))?;
        if st.epoch != g.epoch || !st.local.can_truncate(keep) {
            drop(g);
            let kept: Vec<usize> = st.tokens[..keep].to_vec();
            self.reset(st);
            if !kept.is_empty() {
                self.try_forward_batch(st, &kept)?;
            }
            return Ok(());
        }
        st.local.truncate(keep)?;
        let mut p = st.id.to_le_bytes().to_vec();
        p.extend_from_slice(&(keep as u32).to_le_bytes());
        let mut dead = None;
        for r in g.remotes.iter_mut() {
            let ok = r.ch.send(T_TRUNC, &p).is_ok() && matches!(r.ch.recv(), Ok((T_ACK, _)));
            if !ok {
                dead = Some(r.id);
                break;
            }
        }
        st.tokens.truncate(keep);
        if let Some(id) = dead {
            g.remotes.retain(|r| r.id != id);
            let _ = self.relayout(&mut g);
        }
        Ok(())
    }

    /// Feed `tokens` and return the logits after the last one; survives workers dying (see module docs).
    pub fn try_forward_batch(
        &self,
        st: &mut MeshState,
        tokens: &[usize],
    ) -> Result<Vec<f32>, MeshError> {
        let s = &self.model.spec;
        if tokens.is_empty() {
            return Err(ModelError::NoTokens.into());
        }
        if let Some(&t) = tokens.iter().find(|&&t| t >= s.vocab as usize) {
            return Err(ModelError::Token(t).into());
        }
        if st.tokens.len() + tokens.len() > s.context as usize {
            return Err(ModelError::ContextFull(s.context as usize).into());
        }
        let mut g = self
            .inner
            .lock()
            .map_err(|_| MeshError::Protocol("poisoned".into()))?;
        let mut tries = g.remotes.len() + 1;
        loop {
            let r = self
                .ensure_current(&mut g, st)
                .and_then(|()| self.pipeline(&mut g, st, tokens));
            match r {
                Ok(x) => {
                    st.tokens.extend_from_slice(tokens);
                    return Ok(self.model.logits_from_hidden(&x));
                }
                Err(f) => {
                    st.epoch = u64::MAX; // whatever happened, the caches may be half-written
                    let Some(id) = f.worker else {
                        return Err(f.err);
                    };
                    if tries == 0 {
                        return Err(f.err);
                    }
                    tries -= 1;
                    g.remotes.retain(|r| r.id != id);
                    if let Err(Fail {
                        worker: Some(id2), ..
                    }) = self.relayout(&mut g)
                    {
                        g.remotes.retain(|r| r.id != id2);
                        let _ = self.relayout(&mut g);
                    }
                }
            }
        }
    }

    /// Like `try_forward_batch`, but keeps what the state already knows: the longest common prefix is not recomputed.
    pub fn prefill_cached(
        &self,
        st: &mut MeshState,
        tokens: &[usize],
    ) -> Result<Vec<f32>, MeshError> {
        if tokens.is_empty() {
            return Err(ModelError::NoTokens.into());
        }
        let common = st
            .tokens
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        let keep = common.min(tokens.len() - 1);
        if st.tokens.len() > keep {
            self.truncate(st, keep)?;
        }
        self.try_forward_batch(st, &tokens[st.tokens.len().min(tokens.len() - 1)..])
    }

    /// Rebuild the caches (new session id, replay the history) when the layout changed since `st` last ran.
    fn ensure_current(&self, g: &mut Inner, st: &mut MeshState) -> Result<(), Fail> {
        if st.epoch == g.epoch {
            return Ok(());
        }
        self.close_remote(g, st.id);
        st.id = self.next_session.fetch_add(1, Ordering::Relaxed);
        st.local = self.model.new_state();
        let history = st.tokens.clone();
        if !history.is_empty() {
            self.pipeline(g, st, &history)?;
        }
        st.epoch = g.epoch;
        Ok(())
    }

    /// Run `tokens` through every stage, overlapping the stages; returns the last hidden state.
    fn pipeline(
        &self,
        g: &mut Inner,
        st: &mut MeshState,
        tokens: &[usize],
    ) -> Result<Vec<f32>, Fail> {
        let (model, wire, h) = (&*self.model, self.cfg.wire, self.model.spec.hidden as usize);
        let (session, pos0, local) = (st.id, st.local.pos, g.local.clone());
        let stages: Vec<&mut Remote> = g
            .remotes
            .iter_mut()
            .filter(|r| !r.range.is_empty())
            .collect();
        let n = tokens.len();
        let local_state = &mut st.local;
        thread::scope(|sc| {
            let (first_tx, mut rx) = mpsc::channel::<Item>();
            for r in stages {
                let (tx, next_rx) = mpsc::channel::<Item>();
                let input = std::mem::replace(&mut rx, next_rx);
                sc.spawn(move || {
                    for item in input {
                        let (pos, x) = match item {
                            Ok(v) => v,
                            Err(f) => {
                                let _ = tx.send(Err(f));
                                return;
                            }
                        };
                        let res = (|| -> Result<Vec<f32>, MeshError> {
                            let mut p = session.to_le_bytes().to_vec();
                            p.extend_from_slice(&(pos as u32).to_le_bytes());
                            p.extend_from_slice(&encode(&x, wire));
                            r.ch.send(T_STEP, &p)?;
                            match r.ch.recv()? {
                                (T_HIDDEN, b) => decode(&b, wire, h),
                                (T_ERR, m) => {
                                    Err(MeshError::Worker(String::from_utf8_lossy(&m).into()))
                                }
                                (t, _) => {
                                    Err(MeshError::Protocol(format!("unexpected reply type {t}")))
                                }
                            }
                        })();
                        match res {
                            Ok(y) => {
                                if tx.send(Ok((pos, y))).is_err() {
                                    return;
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(Err(Fail {
                                    worker: Some(r.id),
                                    err: e,
                                }));
                                return;
                            }
                        }
                    }
                });
            }
            // local stage: embed, run own layers, feed the first remote stage
            let mut failed = None;
            for (j, &t) in tokens.iter().enumerate() {
                let r = (|| -> Result<Vec<f32>, MeshError> {
                    let mut x = model.embed_token(t)?;
                    model.run_layers(local_state, local.clone(), &mut x, pos0 + j)?;
                    model.finish_token(local_state);
                    Ok(x)
                })();
                match r {
                    Ok(x) => {
                        if first_tx.send(Ok((pos0 + j, x))).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        failed = Some(Fail {
                            worker: None,
                            err: e,
                        });
                        let _ = first_tx.send(Err(Fail {
                            worker: None,
                            err: MeshError::Protocol("local stage failed".into()),
                        }));
                        break;
                    }
                }
            }
            drop(first_tx);
            let (mut last, mut got, mut err) = (None, 0usize, None);
            for item in rx {
                match item {
                    Ok((_, x)) => {
                        got += 1;
                        last = Some(x);
                    }
                    Err(f) => {
                        err.get_or_insert(f);
                    }
                }
            }
            if let Some(f) = failed {
                return Err(f);
            }
            if let Some(f) = err {
                return Err(f);
            }
            match last {
                Some(x) if got == n && x.len() == h => Ok(x),
                _ => Err(Fail {
                    worker: None,
                    err: MeshError::Protocol("the pipeline ended early".into()),
                }),
            }
        })
    }
}

impl LanguageModel for Mesh {
    type State = MeshState;
    fn new_state(&self) -> MeshState {
        Mesh::new_state(self)
    }
    fn forward_batch(&self, st: &mut MeshState, tokens: &[usize]) -> Result<Vec<f32>, ModelError> {
        self.try_forward_batch(st, tokens).map_err(ModelError::from)
    }
    fn vocab_size(&self) -> usize {
        self.model.spec.vocab as usize
    }
    fn context(&self) -> usize {
        self.model.spec.context as usize
    }
    fn position(&self, st: &MeshState) -> usize {
        st.position()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::{base_spec, model_of};

    fn model() -> Arc<Model> {
        let mut sp = base_spec(2, 1, false);
        sp.layers = 6;
        Arc::new(model_of(sp, false))
    }
    fn cfg() -> MeshConfig {
        MeshConfig::with_key(b"correct horse battery".to_vec())
    }
    fn local(a: &str) -> String {
        a.to_string()
    }

    fn hexs(b: [u8; 32]) -> String {
        hex32(&b)
    }

    #[test]
    fn hmac_matches_rfc4231() {
        assert_eq!(
            hexs(hmac_sha256(&[0x0b; 20], &[b"Hi There"])),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        assert_eq!(
            hexs(hmac_sha256(
                b"Jefe",
                &[b"what do ya want ", b"for nothing?"]
            )),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // a key longer than the block is hashed first (test case 6)
        assert_eq!(
            hexs(hmac_sha256(
                &[0xaa; 131],
                &[b"Test Using Larger Than Block-Size Key - Hash Key First"]
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn partition_covers_everything_in_order() {
        let mut seed = 7u64;
        for _ in 0..500 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let layers = (seed >> 33) as usize % 40;
            let nodes = 1 + (seed >> 20) as usize % 7;
            let speeds: Vec<f64> = (0..nodes).map(|i| ((seed >> (i * 5)) % 9) as f64).collect();
            let costs: Vec<f64> = (0..layers)
                .map(|i| 1.0 + ((seed >> (i % 40)) & 3) as f64)
                .collect();
            let p = partition(&costs, &speeds);
            assert_eq!(p.len(), nodes);
            let pos = speeds.iter().filter(|&&s| s > 0.0).count();
            if pos == 0 || layers == 0 {
                assert!(p.iter().all(|r| r.is_empty()));
                continue;
            }
            let mut at = 0;
            for (r, &s) in p.iter().zip(&speeds) {
                if r.is_empty() {
                    continue;
                }
                assert!(s > 0.0 && r.start == at, "{p:?}");
                at = r.end;
            }
            assert_eq!(at, layers, "{p:?} speeds {speeds:?}");
            if layers >= pos {
                assert!(
                    speeds
                        .iter()
                        .zip(&p)
                        .all(|(&s, r)| s <= 0.0 || !r.is_empty()),
                    "{p:?} {speeds:?}"
                );
            }
        }
        assert_eq!(partition(&[1.0; 8], &[1.0, 1.0]), vec![0..4, 4..8]);
        assert_eq!(partition(&[1.0; 8], &[3.0, 1.0]), vec![0..6, 6..8]);
        assert_eq!(
            partition(&[1.0; 8], &[1.0, 0.0, f64::NAN, 1.0]),
            vec![0..4, 0..0, 0..0, 4..8]
        );
        let heavy = [1.0, 1.0, 1.0, 1.0, 10.0, 10.0];
        assert_eq!(partition(&heavy, &[1.0, 1.0]), vec![0..5, 5..6]);
    }

    #[test]
    fn frames_survive_loopback_and_reject_tampering() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t = thread::spawn(move || {
            let (s, _) = l.accept().unwrap();
            let mut c = handshake(s, b"0123456789", false, Duration::from_secs(5)).unwrap();
            let (ty, p) = c.recv().unwrap();
            c.send(ty, &p).unwrap();
            // a forged frame: valid length, wrong MAC
            let mut raw = Vec::new();
            raw.extend_from_slice(&((1 + 4 + MAC_LEN) as u32).to_le_bytes());
            raw.push(9);
            raw.extend_from_slice(&[1, 2, 3, 4]);
            raw.extend_from_slice(&[0u8; MAC_LEN]);
            c.s.write_all(&raw).unwrap();
            thread::sleep(Duration::from_millis(200));
        });
        let mut c = handshake(
            TcpStream::connect(addr).unwrap(),
            b"0123456789",
            true,
            Duration::from_secs(5),
        )
        .unwrap();
        c.send(T_STEP, b"hello").unwrap();
        assert_eq!(c.recv().unwrap(), (T_STEP, b"hello".to_vec()));
        assert!(c.recv().is_err(), "a frame with a bad MAC must be refused");
        t.join().unwrap();
    }

    #[test]
    fn oversized_frame_is_refused_before_allocating() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t = thread::spawn(move || {
            let (s, _) = l.accept().unwrap();
            let mut c = handshake(s, b"0123456789", false, Duration::from_secs(5)).unwrap();
            c.s.write_all(&u32::MAX.to_le_bytes()).unwrap();
            thread::sleep(Duration::from_millis(100));
        });
        let mut c = handshake(
            TcpStream::connect(addr).unwrap(),
            b"0123456789",
            true,
            Duration::from_secs(5),
        )
        .unwrap();
        assert!(c.recv().is_err());
        t.join().unwrap();
    }

    fn reference(m: &Model, tokens: &[usize]) -> Vec<f32> {
        let mut st = m.new_state();
        m.forward_batch(&mut st, tokens).unwrap()
    }

    #[test]
    fn mesh_equals_a_single_machine() {
        let m = model();
        let w1 = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), None).unwrap();
        let w2 = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), None).unwrap();
        let mesh = Mesh::connect(m.clone(), &[w1.addr(), w2.addr()], cfg()).unwrap();
        let lay = mesh.layout();
        assert_eq!(lay.len(), 3);
        assert_eq!(lay.iter().map(|n| n.layers.len()).sum::<usize>(), 6);
        assert_eq!(lay.last().unwrap().layers.end, 6);
        let mut st = mesh.new_state();
        let prompt = [3usize, 7, 1, 19, 5, 2];
        let got = mesh.try_forward_batch(&mut st, &prompt).unwrap();
        assert_eq!(got, reference(&m, &prompt));
        // decode one token at a time
        let mut all = prompt.to_vec();
        for t in [4usize, 11, 0] {
            all.push(t);
            assert_eq!(
                mesh.try_forward_batch(&mut st, &[t]).unwrap(),
                reference(&m, &all)
            );
        }
        assert_eq!(st.position(), all.len());
        assert!(mesh.try_forward_batch(&mut st, &[9999]).is_err());
        assert!(mesh.try_forward_batch(&mut st, &[]).is_err());
    }

    #[test]
    fn two_conversations_do_not_mix() {
        let m = model();
        let w = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), None).unwrap();
        let mesh = Mesh::connect(m.clone(), &[w.addr()], cfg()).unwrap();
        let (mut a, mut b) = (mesh.new_state(), mesh.new_state());
        mesh.try_forward_batch(&mut a, &[1, 2, 3]).unwrap();
        mesh.try_forward_batch(&mut b, &[9, 8]).unwrap();
        assert_eq!(
            mesh.try_forward_batch(&mut a, &[4]).unwrap(),
            reference(&m, &[1, 2, 3, 4])
        );
        assert_eq!(
            mesh.try_forward_batch(&mut b, &[7]).unwrap(),
            reference(&m, &[9, 8, 7])
        );
    }

    #[test]
    fn coordinator_with_zero_speed_only_embeds() {
        let m = model();
        let w = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), None).unwrap();
        let mut c = cfg();
        c.local_speed = Some(0.0);
        let mesh = Mesh::connect(m.clone(), &[w.addr()], c).unwrap();
        assert!(mesh.layout()[0].layers.is_empty());
        let mut st = mesh.new_state();
        assert_eq!(
            mesh.try_forward_batch(&mut st, &[5, 6, 7]).unwrap(),
            reference(&m, &[5, 6, 7])
        );
    }

    #[test]
    fn f16_wire_is_close() {
        let m = model();
        let w = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), None).unwrap();
        let mut c = cfg();
        c.wire = Wire::F16;
        let mesh = Mesh::connect(m.clone(), &[w.addr()], c).unwrap();
        let mut st = mesh.new_state();
        let got = mesh.try_forward_batch(&mut st, &[5, 6, 7, 8]).unwrap();
        let want = reference(&m, &[5, 6, 7, 8]);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 0.05, "max error {err}");
    }

    #[test]
    fn wrong_key_and_wrong_model_are_refused() {
        let m = model();
        let w = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), None).unwrap();
        let bad = MeshConfig::with_key(b"another key entirely".to_vec());
        let e = Mesh::connect(m.clone(), &[w.addr()], bad).err();
        assert!(matches!(e, Some(MeshError::Auth)), "{e:?}");
        assert!(matches!(
            Mesh::connect(
                m.clone(),
                &[w.addr()],
                MeshConfig::with_key(b"short".to_vec())
            ),
            Err(MeshError::Config(_))
        ));
        let mut other = base_spec(2, 1, false);
        other.layers = 6;
        let different = Arc::new(
            crate::model::Model::new(
                other.clone(),
                crate::model::tests::random_weights(&other, 12345, false),
            )
            .unwrap(),
        );
        assert!(matches!(
            Mesh::connect(different, &[w.addr()], cfg()),
            Err(MeshError::Model(_))
        ));
        // the worker is still alive for the right client
        assert!(Mesh::connect(m, &[w.addr()], cfg()).is_ok());
    }

    #[test]
    fn a_dying_worker_is_replaced_by_the_survivors() {
        let m = model();
        let mut w1 = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), None).unwrap();
        let w2 = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), None).unwrap();
        let mesh = Mesh::connect(m.clone(), &[w1.addr(), w2.addr()], cfg()).unwrap();
        let mut st = mesh.new_state();
        let mut all = vec![3usize, 1, 4, 1, 5];
        assert_eq!(
            mesh.try_forward_batch(&mut st, &all).unwrap(),
            reference(&m, &all)
        );
        w1.stop();
        all.push(9);
        assert_eq!(
            mesh.try_forward_batch(&mut st, &[9]).unwrap(),
            reference(&m, &all),
            "recovered after losing a worker"
        );
        assert_eq!(mesh.workers(), 1);
        drop(w2);
        all.push(2);
        assert_eq!(
            mesh.try_forward_batch(&mut st, &[2]).unwrap(),
            reference(&m, &all),
            "recovered down to the coordinator alone"
        );
        assert_eq!(mesh.workers(), 0);
        assert_eq!(mesh.layout()[0].layers, 0..6);
    }

    #[test]
    fn a_restarted_worker_can_rejoin() {
        let m = model();
        let w = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), None).unwrap();
        let mesh = Mesh::connect(m.clone(), &[], cfg()).unwrap();
        let mut st = mesh.new_state();
        mesh.try_forward_batch(&mut st, &[1, 2, 3]).unwrap();
        mesh.add_worker(w.addr()).unwrap();
        assert_eq!(mesh.workers(), 1);
        assert_eq!(
            mesh.try_forward_batch(&mut st, &[4]).unwrap(),
            reference(&m, &[1, 2, 3, 4])
        );
    }

    #[test]
    fn rewind_and_prefix_reuse() {
        let m = model();
        let w = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), None).unwrap();
        let mesh = Mesh::connect(m.clone(), &[w.addr()], cfg()).unwrap();
        let mut st = mesh.new_state();
        mesh.try_forward_batch(&mut st, &[1, 2, 3, 4, 5]).unwrap();
        mesh.truncate(&mut st, 3).unwrap();
        assert_eq!(st.position(), 3);
        assert_eq!(
            mesh.try_forward_batch(&mut st, &[8, 9]).unwrap(),
            reference(&m, &[1, 2, 3, 8, 9])
        );
        // a second turn sharing a prefix with the first
        let got = mesh.prefill_cached(&mut st, &[1, 2, 3, 8, 6]).unwrap();
        assert_eq!(got, reference(&m, &[1, 2, 3, 8, 6]));
        assert_eq!(st.tokens(), &[1, 2, 3, 8, 6]);
        mesh.reset(&mut st);
        assert_eq!(st.position(), 0);
        assert_eq!(
            mesh.try_forward_batch(&mut st, &[2]).unwrap(),
            reference(&m, &[2])
        );
    }

    #[test]
    fn workers_answer_discovery_probes() {
        let m = model();
        let w = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), Some(0)).unwrap();
        let probe = SocketAddr::new(
            "127.0.0.1".parse().unwrap(),
            w.discovery_addr().unwrap().port(),
        );
        let found = discover(
            &[probe],
            Duration::from_millis(500),
            Some(model_fingerprint(&m)),
        );
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].addr.port(), w.addr().port());
        assert_eq!(found[0].layers, 6);
        let mesh = Mesh::connect(m.clone(), &[found[0].addr], cfg()).unwrap();
        assert_eq!(mesh.workers(), 1);
        // a different model fingerprint filters it out
        assert!(discover(&[probe], Duration::from_millis(300), Some([7u8; 32])).is_empty());
        let _ = local("");
    }

    #[test]
    fn works_as_a_language_model() {
        let m = model();
        let w = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), None).unwrap();
        let mesh = Mesh::connect(m.clone(), &[w.addr()], cfg()).unwrap();
        let mut st = LanguageModel::new_state(&mesh);
        let l = LanguageModel::forward(&mesh, &mut st, 3).unwrap();
        assert_eq!(l, reference(&m, &[3]));
        assert_eq!(LanguageModel::position(&mesh, &st), 1);
        assert_eq!(LanguageModel::vocab_size(&mesh), m.spec.vocab as usize);
    }

    // ---------------------------------------------------------------- hardening regressions

    #[test]
    fn debug_output_never_contains_the_key() {
        let c = MeshConfig::with_key(b"super-secret-passphrase".to_vec());
        let text = format!("{c:?}");
        assert!(!text.contains("super-secret"), "{text}");
        assert!(
            !text.contains("115"),
            "byte values of the key must not appear: {text}"
        ); // 's' = 115
        assert!(text.contains("redacted"), "{text}");
    }

    #[test]
    fn key_strength_warnings() {
        assert!(key_strength_warning(b"short").unwrap().contains("refused"));
        assert!(key_strength_warning(b"12345678").unwrap().contains("16"));
        assert!(
            key_strength_warning(&[7u8; 32])
                .unwrap()
                .contains("distinct")
        );
        assert!(key_strength_warning(b"correct horse battery staple").is_none());
        assert!(MeshConfig::with_key(b"short".to_vec()).check().is_err());
        assert!(
            spawn_worker(
                model(),
                "127.0.0.1:0",
                MeshConfig::with_key(b"1234567".to_vec()),
                None
            )
            .is_err()
        );
    }

    /// A coordinator-side and a worker-side channel on the same key, over loopback.
    fn pair() -> (Chan, Chan) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t = thread::spawn(move || {
            let (s, _) = l.accept().unwrap();
            handshake(s, b"0123456789", false, Duration::from_secs(5)).unwrap()
        });
        let c = handshake(
            TcpStream::connect(addr).unwrap(),
            b"0123456789",
            true,
            Duration::from_secs(5),
        )
        .unwrap();
        let w = t.join().unwrap();
        w.s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        (c, w)
    }
    /// A frame whose MAC is computed over `mac_*` but which carries `wire_*` on the wire.
    fn frame(
        c: &Chan,
        mac_dir: u8,
        mac_seq: u64,
        mac_ty: u8,
        mac_payload: &[u8],
        wire_ty: u8,
        wire_payload: &[u8],
    ) -> Vec<u8> {
        let mac = c.mac(mac_dir, mac_seq, mac_ty, mac_payload);
        let mut raw = ((1 + wire_payload.len() + MAC_LEN) as u32)
            .to_le_bytes()
            .to_vec();
        raw.push(wire_ty);
        raw.extend_from_slice(wire_payload);
        raw.extend_from_slice(&mac[..MAC_LEN]);
        raw
    }

    #[test]
    fn the_mac_covers_direction_sequence_type_and_payload_each_on_its_own() {
        let p = b"payload bytes";
        // control: a correctly built frame is accepted, so every rejection below is down to one field
        let (mut c, mut w) = pair();
        let ok = frame(&c, c.tx_dir, 0, T_STEP, p, T_STEP, p);
        c.s.write_all(&ok).unwrap();
        assert_eq!(w.recv().unwrap(), (T_STEP, p.to_vec()));

        type Case = (&'static str, Box<dyn Fn(&Chan) -> Vec<u8>>);
        let cases: Vec<Case> = vec![
            (
                "direction",
                Box::new(|c| frame(c, 3 - c.tx_dir, 0, T_STEP, p, T_STEP, p)),
            ),
            (
                "sequence number",
                Box::new(|c| frame(c, c.tx_dir, 1, T_STEP, p, T_STEP, p)),
            ),
            (
                "type substitution",
                Box::new(|c| frame(c, c.tx_dir, 0, T_STEP, p, T_CLOSE, p)),
            ),
            (
                "payload",
                Box::new(|c| frame(c, c.tx_dir, 0, T_STEP, p, T_STEP, b"payload byteX")),
            ),
            (
                "payload length",
                Box::new(|c| frame(c, c.tx_dir, 0, T_STEP, p, T_STEP, &p[..p.len() - 1])),
            ),
        ];
        for (what, build) in &cases {
            let (mut c, mut w) = pair();
            c.s.write_all(&build(&c)).unwrap();
            let e = w.recv().expect_err(what);
            assert!(e.to_string().contains("authentication"), "{what}: {e}");
        }
    }

    #[test]
    fn replayed_reordered_and_reflected_frames_are_refused() {
        // replay: the same valid frame twice
        let (mut c, mut w) = pair();
        let f0 = frame(&c, c.tx_dir, 0, T_STEP, b"a", T_STEP, b"a");
        c.s.write_all(&f0).unwrap();
        assert!(w.recv().is_ok());
        c.s.write_all(&f0).unwrap();
        assert!(w.recv().is_err(), "a replayed frame must be refused");

        // reorder: sequence 1 arrives before sequence 0
        let (mut c, mut w) = pair();
        let f1 = frame(&c, c.tx_dir, 1, T_STEP, b"b", T_STEP, b"b");
        c.s.write_all(&f1).unwrap();
        assert!(w.recv().is_err(), "an out-of-order frame must be refused");

        // reflection: the receiver's own frame sent back to it
        let (mut c, mut w) = pair();
        w.send(T_HIDDEN, b"reply").unwrap();
        let mut echoed = vec![0u8; 4 + 1 + 5 + MAC_LEN];
        c.s.read_exact(&mut echoed).unwrap();
        w.s.write_all(&echoed).unwrap();
        w.s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        assert!(w.recv().is_err(), "a reflected frame must be refused");

        // a frame from another session (other key material) is refused too
        let (c, mut w) = pair();
        let (mut c2, _w2) = pair();
        c2.s = c.s.try_clone().unwrap();
        c2.send(T_STEP, b"x").unwrap();
        assert!(
            w.recv().is_err(),
            "a frame from another session must be refused"
        );
    }

    #[test]
    fn a_silent_peer_is_cut_off_quickly_and_floods_are_capped() {
        let m = model();
        let w = spawn_worker(m.clone(), "127.0.0.1:0", cfg(), None).unwrap();
        // open more raw connections than one IP may hold and never speak: the extras are refused
        let mut socks: Vec<TcpStream> = (0..MAX_CONNS_PER_IP + 3)
            .map(|_| TcpStream::connect(w.addr()).unwrap())
            .collect();
        thread::sleep(Duration::from_millis(300));
        let mut refused = 0;
        for s in socks.iter_mut() {
            s.set_read_timeout(Some(Duration::from_millis(200)))
                .unwrap();
            let mut buf = [0u8; 64];
            // an admitted connection receives the worker's hello (21 bytes); a refused one sees EOF
            match s.read(&mut buf) {
                Ok(0) => refused += 1,
                Ok(_) => {}
                Err(_) => {}
            }
        }
        assert!(
            refused >= 3,
            "only {refused} of {} connections were refused",
            MAX_CONNS_PER_IP + 3
        );
        // the admitted ones are dropped by the handshake timeout, not held for 10 s
        let t0 = Instant::now();
        for s in socks.iter_mut() {
            s.set_read_timeout(Some(HANDSHAKE_TIMEOUT + Duration::from_secs(10)))
                .unwrap();
            let mut buf = [0u8; 64];
            loop {
                match s.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => continue,
                }
            }
        }
        assert!(
            t0.elapsed() < HANDSHAKE_TIMEOUT + Duration::from_secs(10),
            "pre-auth connections held for {:?}",
            t0.elapsed()
        );
        // and once they are gone a legitimate coordinator is served again
        thread::sleep(Duration::from_millis(200));
        assert!(Mesh::connect(m, &[w.addr()], cfg()).is_ok());
    }

    #[test]
    fn discovery_never_amplifies_and_is_rate_limited() {
        let m = model();
        let w = spawn_worker(m, "127.0.0.1:0", cfg(), Some(0)).unwrap();
        let target = SocketAddr::new(
            "127.0.0.1".parse().unwrap(),
            w.discovery_addr().unwrap().port(),
        );
        let u = UdpSocket::bind("127.0.0.1:0").unwrap();
        u.set_read_timeout(Some(Duration::from_millis(400)))
            .unwrap();
        let mut buf = [0u8; 512];
        // the old five-byte probe, and a short probe, get no answer at all
        u.send_to(b"CMSH?", target).unwrap();
        assert!(
            u.recv_from(&mut buf).is_err(),
            "a bare probe must not be answered"
        );
        let mut short = [0u8; 40];
        short[..5].copy_from_slice(b"CMSH?");
        u.send_to(&short, target).unwrap();
        assert!(
            u.recv_from(&mut buf).is_err(),
            "a probe shorter than the reply must not be answered"
        );
        // a padded probe is answered, with a reply no longer than the probe
        let mut probe = [0u8; PROBE_LEN];
        probe[..5].copy_from_slice(b"CMSH?");
        u.send_to(&probe, target).unwrap();
        let (n, _) = u.recv_from(&mut buf).expect("a padded probe is answered");
        assert!(
            n <= probe.len(),
            "reply of {n} bytes to a probe of {} bytes",
            probe.len()
        );
        // a second probe at once from the same source is dropped
        u.send_to(&probe, target).unwrap();
        u.send_to(&probe, target).unwrap();
        assert!(
            u.recv_from(&mut buf).is_err(),
            "probes inside the per-source interval must be dropped"
        );
        // after the interval it answers again
        thread::sleep(DISCOVERY_PER_SOURCE + Duration::from_millis(100));
        u.send_to(&probe, target).unwrap();
        assert!(u.recv_from(&mut buf).is_ok());
    }
}
