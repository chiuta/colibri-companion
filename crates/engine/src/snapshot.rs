//! Save and restore a conversation's [`State`]: the KV cache in any precision, the token history,
//! the position and the eviction bookkeeping, in one file that can be trusted and resumed.
//!
//! # File layout (little-endian)
//!
//! ```text
//! 0   magic    "CMPSNAP1"
//! 8   u32      format version (1)
//! 12  u32      flags (bit 0: payload is LZ-compressed, planes are byte-shuffled)
//! 16  [u8; 32] model fingerprint, see Model::fingerprint
//! 48  u64      payload length once decompressed
//! 56  u64      payload length as stored
//! 64  ...      payload
//! ..  [u8; 32] SHA-256 of everything above
//! ```
//!
//! The hash is checked first, then the fingerprint, then every length in the payload is checked
//! against what is left of the file before anything is allocated: a damaged, truncated or foreign
//! file is an `Err`, never a panic and never a state that continues with wrong numbers.
//! Restoring is bit-exact: the logits that follow a round trip equal those of the original state.
//!
//! Files are written to a temporary name and renamed into place, so a reader sees either the old
//! file or the new one. A [`SnapshotStore`] keeps a directory of snapshots under a size cap, keyed by
//! a hash of the token prefix they hold, evicting the least recently used.
//!
//! Not saved: accumulated attention scores (switch tracking on again after a restore) and states
//! whose cache lives on an accelerator.

use crate::hub::Sha256;
use crate::model::{KvKind, Model, ModelError, PlaneData, PlaneRef, State};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

const MAGIC: &[u8; 8] = b"CMPSNAP1";
const VERSION: u32 = 1;
const HEADER: usize = 64;
const TRAILER: usize = 32;
const FLAG_COMPRESSED: u32 = 1;

#[derive(Debug)]
pub enum SnapshotError {
    Io(String),
    /// Not a snapshot, truncated, or inconsistent inside.
    Format(String),
    /// The hash does not match: the file was damaged.
    Corrupt,
    /// A snapshot of another model (or another architecture).
    WrongModel,
    Version(u32),
    /// Can't be done for this state or request.
    Unsupported(String),
    Model(ModelError),
}
impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::Io(s) => write!(f, "snapshot I/O: {s}"),
            SnapshotError::Format(s) => write!(f, "not a usable snapshot: {s}"),
            SnapshotError::Corrupt => write!(f, "the snapshot is damaged (checksum mismatch)"),
            SnapshotError::WrongModel => write!(f, "the snapshot belongs to another model"),
            SnapshotError::Version(v) => write!(f, "snapshot format version {v} is not supported"),
            SnapshotError::Unsupported(s) => write!(f, "{s}"),
            SnapshotError::Model(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for SnapshotError {}
impl From<std::io::Error> for SnapshotError { fn from(e: std::io::Error) -> Self { SnapshotError::Io(e.to_string()) } }
impl From<ModelError> for SnapshotError { fn from(e: ModelError) -> Self { SnapshotError::Model(e) } }
type Res<T> = Result<T, SnapshotError>;
fn fmt_err<T>(m: &str) -> Res<T> { Err(SnapshotError::Format(m.into())) }

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SnapOptions {
    /// Byte-shuffle the cache and LZ-compress the file. Small gain on dense F32 caches, larger on
    /// F16/Q8 and on caches with repeated structure; costs time on save and load.
    pub compress: bool,
}

// ---------------------------------------------------------------- compression

fn shuffle(d: &[u8], w: usize) -> Vec<u8> {
    let n = d.len() / w;
    let mut o = vec![0u8; d.len()];
    for i in 0..n { for j in 0..w { o[j * n + i] = d[i * w + j]; } }
    o
}
fn unshuffle(d: &[u8], w: usize) -> Vec<u8> {
    let n = d.len() / w;
    let mut o = vec![0u8; d.len()];
    for i in 0..n { for j in 0..w { o[i * w + j] = d[j * n + i]; } }
    o
}

fn lz_emit(out: &mut Vec<u8>, lits: &[u8], m: Option<(usize, usize)>) {
    let (ll, ml) = (lits.len(), m.map_or(0, |(_, l)| l - 4));
    out.push(((ll.min(15) as u8) << 4) | ml.min(15) as u8);
    let ext = |out: &mut Vec<u8>, v: usize| { let mut r = v - 15; while r >= 255 { out.push(255); r -= 255; } out.push(r as u8); };
    if ll >= 15 { ext(out, ll); }
    out.extend_from_slice(lits);
    if let Some((off, _)) = m {
        out.extend_from_slice(&(off as u16).to_le_bytes());
        if ml >= 15 { ext(out, ml); }
    }
}
/// LZ77 block compression (4-byte minimum match, 64 KiB window).
fn lz_compress(src: &[u8]) -> Vec<u8> {
    let n = src.len();
    let mut out = Vec::with_capacity(n / 2 + 16);
    let mut table = vec![u32::MAX; 1 << 16];
    let (mut i, mut anchor) = (0usize, 0usize);
    while i + 4 <= n && n < u32::MAX as usize {
        let w = u32::from_le_bytes([src[i], src[i + 1], src[i + 2], src[i + 3]]);
        let h = (w.wrapping_mul(2_654_435_761) >> 16) as usize;
        let cand = table[h];
        table[h] = i as u32;
        if cand != u32::MAX && i - cand as usize <= 65_535 && src[cand as usize..cand as usize + 4] == src[i..i + 4] {
            let c = cand as usize;
            let mut l = 4;
            while i + l < n && src[c + l] == src[i + l] { l += 1; }
            lz_emit(&mut out, &src[anchor..i], Some((i - c, l)));
            i += l;
            anchor = i;
        } else { i += 1; }
    }
    lz_emit(&mut out, &src[anchor..], None);
    out
}
fn lz_decompress(src: &[u8], raw_len: usize) -> Res<Vec<u8>> {
    let bad = || SnapshotError::Format("the compressed payload is damaged".into());
    let mut out: Vec<u8> = Vec::with_capacity(raw_len);
    let mut i = 0;
    let ext = |i: &mut usize, mut v: usize| -> Res<usize> {
        loop {
            let b = *src.get(*i).ok_or_else(bad)?;
            *i += 1;
            v += b as usize;
            if v > raw_len + 15 { return Err(bad()); }
            if b != 255 { return Ok(v); }
        }
    };
    while i < src.len() {
        let tok = src[i];
        i += 1;
        let mut ll = (tok >> 4) as usize;
        if ll == 15 { ll = ext(&mut i, 15)?; }
        if i + ll > src.len() || out.len() + ll > raw_len { return Err(bad()); }
        out.extend_from_slice(&src[i..i + ll]);
        i += ll;
        if i == src.len() { break; }
        if i + 2 > src.len() { return Err(bad()); }
        let off = u16::from_le_bytes([src[i], src[i + 1]]) as usize;
        i += 2;
        let mut ml = (tok & 15) as usize;
        if ml == 15 { ml = ext(&mut i, 15)?; }
        ml += 4;
        if off == 0 || off > out.len() || out.len() + ml > raw_len { return Err(bad()); }
        let from = out.len() - off;
        for k in 0..ml { let b = out[from + k]; out.push(b); }
    }
    if out.len() != raw_len { return Err(bad()); }
    Ok(out)
}

// ---------------------------------------------------------------- payload

struct W(Vec<u8>);
impl W {
    fn u64(&mut self, v: usize) { self.0.extend_from_slice(&(v as u64).to_le_bytes()); }
    fn bytes(&mut self, b: &[u8]) { self.0.extend_from_slice(b); }
}
struct R<'a> { b: &'a [u8], i: usize }
impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let end = self.i.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| SnapshotError::Format("the file ends inside a record".into()))?;
        let s = &self.b[self.i..end];
        self.i = end;
        Ok(s)
    }
    fn u64(&mut self) -> Res<usize> {
        let v = u64::from_le_bytes(self.take(8)?.try_into().unwrap_or([0; 8]));
        usize::try_from(v).map_err(|_| SnapshotError::Format("a length does not fit this machine".into()))
    }
    fn left(&self) -> usize { self.b.len() - self.i }
}

fn put_f32(w: &mut W, v: &[f32], shuf: bool) {
    w.u64(v.len());
    let raw: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    w.bytes(&if shuf { shuffle(&raw, 4) } else { raw });
}
fn put_plane(w: &mut W, p: &PlaneRef<'_>, shuf: bool) {
    match p {
        PlaneRef::F32(v) => put_f32(w, v, shuf),
        PlaneRef::F16(v) => {
            w.u64(v.len());
            let raw: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            w.bytes(&if shuf { shuffle(&raw, 2) } else { raw });
        }
        PlaneRef::Q8 { q, s } => {
            w.u64(q.len());
            w.bytes(&q.iter().map(|&b| b as u8).collect::<Vec<u8>>());
            put_f32(w, s, shuf);
        }
    }
}
fn get_f32(r: &mut R<'_>, shuf: bool) -> Res<Vec<f32>> {
    let n = r.u64()?;
    let raw = r.take(n.checked_mul(4).ok_or_else(|| SnapshotError::Format("a length overflows".into()))?)?;
    let raw = if shuf { unshuffle(raw, 4) } else { raw.to_vec() };
    Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
fn get_plane(r: &mut R<'_>, kind: KvKind, shuf: bool) -> Res<PlaneData> {
    Ok(match kind {
        KvKind::F32 => PlaneData::F32(get_f32(r, shuf)?),
        KvKind::F16 => {
            let n = r.u64()?;
            let raw = r.take(n.checked_mul(2).ok_or_else(|| SnapshotError::Format("a length overflows".into()))?)?;
            let raw = if shuf { unshuffle(raw, 2) } else { raw.to_vec() };
            PlaneData::F16(raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
        }
        KvKind::Q8 => {
            let n = r.u64()?;
            let q = r.take(n)?.iter().map(|&b| b as i8).collect();
            PlaneData::Q8 { q, s: get_f32(r, shuf)? }
        }
    })
}
fn kind_code(k: KvKind) -> u8 { match k { KvKind::F32 => 0, KvKind::F16 => 1, KvKind::Q8 => 2 } }

fn digest(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    for p in parts { h.update(p); }
    h.finish()
}

/// The whole snapshot as bytes.
pub fn encode(model: &Model, st: &State, opts: SnapOptions) -> Res<Vec<u8>> {
    let views = st.kv_views();
    if views.len() != model.spec.layers as usize { return Err(SnapshotError::Unsupported("the state was not made by this model".into())); }
    let held = |cap: usize| if cap > 0 { st.pos.min(cap) } else { st.pos };
    for v in &views {
        let want = held(v.cap) * v.kd;
        let len = |p: &PlaneRef<'_>| match p { PlaneRef::F32(x) => x.len(), PlaneRef::F16(x) => x.len(), PlaneRef::Q8 { q, .. } => q.len() };
        if len(&v.k) != want || len(&v.v) != want {
            return Err(SnapshotError::Unsupported("the cache is not held on the host (accelerator state) and cannot be saved".into()));
        }
    }
    if st.tokens().len() != st.pos { return Err(SnapshotError::Unsupported("the token history is out of step with the cache".into())); }
    if st.tokens().iter().any(|&t| t > u32::MAX as usize) { return Err(SnapshotError::Unsupported("token ids above 2^32 are not storable".into())); }
    let shuf = opts.compress;
    let (abs, gap) = st.kvx_parts();
    let mut w = W(Vec::new());
    w.u64(st.pos);
    w.u64(gap);
    w.bytes(&[kind_code(st.kv_kind())]);
    w.u64(views.len());
    w.bytes(&st.tokens().iter().flat_map(|&t| (t as u32).to_le_bytes()).collect::<Vec<u8>>());
    w.u64(abs.len());
    w.bytes(&abs.iter().flat_map(|&a| (a as u64).to_le_bytes()).collect::<Vec<u8>>());
    for v in &views {
        w.u64(v.cap);
        w.u64(v.kd);
        w.u64(v.kvh);
        put_plane(&mut w, &v.k, shuf);
        put_plane(&mut w, &v.v, shuf);
    }
    let raw_len = w.0.len();
    let stored = if opts.compress { lz_compress(&w.0) } else { w.0 };
    let mut out = Vec::with_capacity(HEADER + stored.len() + TRAILER);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(if opts.compress { FLAG_COMPRESSED } else { 0 }).to_le_bytes());
    out.extend_from_slice(&model.fingerprint());
    out.extend_from_slice(&(raw_len as u64).to_le_bytes());
    out.extend_from_slice(&(stored.len() as u64).to_le_bytes());
    out.extend_from_slice(&stored);
    let h = digest(&[&out]);
    out.extend_from_slice(&h);
    Ok(out)
}

/// A state back from `bytes`, bit for bit what [`encode`] saw. Errors for anything damaged,
/// truncated, foreign or of another model.
pub fn decode(model: &Model, bytes: &[u8]) -> Res<State> {
    if bytes.len() < HEADER + TRAILER || &bytes[..8] != MAGIC { return fmt_err("not a snapshot file (or cut short)"); }
    let u32at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let u64at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap_or([0; 8]));
    let version = u32at(8);
    if version != VERSION { return Err(SnapshotError::Version(version)); }
    let body = bytes.len() - TRAILER;
    if digest(&[&bytes[..body]]) != bytes[body..] { return Err(SnapshotError::Corrupt); }
    if bytes[16..48] != model.fingerprint() { return Err(SnapshotError::WrongModel); }
    let (raw_len, stored_len) = (u64at(48), u64at(56));
    if stored_len != (body - HEADER) as u64 { return fmt_err("the payload length does not match the file"); }
    let flags = u32at(12);
    if flags & !FLAG_COMPRESSED != 0 { return fmt_err("unknown flags"); }
    let compressed = flags & FLAG_COMPRESSED != 0;
    let stored = &bytes[HEADER..body];
    let raw_len = usize::try_from(raw_len).map_err(|_| SnapshotError::Format("payload too large".into()))?;
    let owned;
    let raw: &[u8] = if compressed {
        if raw_len > stored.len().saturating_mul(255).saturating_add(64) { return fmt_err("implausible compression ratio"); }
        owned = lz_decompress(stored, raw_len)?;
        &owned
    } else {
        if raw_len != stored.len() { return fmt_err("the payload length does not match"); }
        stored
    };
    let mut r = R { b: raw, i: 0 };
    let (pos, gap) = (r.u64()?, r.u64()?);
    let kind = match r.take(1)?[0] { 0 => KvKind::F32, 1 => KvKind::F16, 2 => KvKind::Q8, _ => return fmt_err("unknown cache precision") };
    let layers = r.u64()?;
    let s = &model.spec;
    if layers != s.layers as usize { return fmt_err("the layer count differs from the model"); }
    if pos > s.context as usize { return fmt_err("more tokens than the model's context"); }
    if pos.checked_mul(4).is_none_or(|b| b > r.left()) { return fmt_err("the token list is cut short"); }
    let tokens: Vec<usize> = r.take(pos * 4)?.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize).collect();
    if tokens.iter().any(|&t| t >= s.vocab as usize) { return fmt_err("a token lies outside the vocabulary"); }
    let na = r.u64()?;
    if na != 0 && na != pos { return fmt_err("the position table has the wrong length"); }
    if na.checked_mul(8).is_none_or(|b| b > r.left()) { return fmt_err("the position table is cut short"); }
    let abs: Vec<usize> = r.take(na * 8)?.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as usize).collect();
    let mut st = model.new_state_with(kind);
    let shapes: Vec<(usize, usize, usize)> = st.kv_views().iter().map(|v| (v.cap, v.kd, v.kvh)).collect();
    let mut planes = Vec::with_capacity(shapes.len());
    for (cap, kd, kvh) in shapes {
        if (r.u64()?, r.u64()?, r.u64()?) != (cap, kd, kvh) { return fmt_err("a layer's shape differs from the model"); }
        let k = get_plane(&mut r, kind, compressed)?;
        let v = get_plane(&mut r, kind, compressed)?;
        planes.push((k, v));
    }
    if r.left() != 0 { return fmt_err("unexpected bytes after the last layer"); }
    st.restore_raw(pos, tokens, abs, gap, planes)?;
    Ok(st)
}

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
/// Write `bytes` to `path` through a temporary file in the same directory (flushed to disk, then
/// renamed), so no reader ever sees a half-written file.
fn write_atomic(path: &Path, bytes: &[u8]) -> Res<()> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let nanos = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let tmp = dir.join(format!(".{name}.{}.{}.{nanos}.tmp", std::process::id(), TMP_SEQ.fetch_add(1, Ordering::Relaxed)));
    let r = (|| -> std::io::Result<()> {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        fs::rename(&tmp, path)
    })();
    if r.is_err() { let _ = fs::remove_file(&tmp); }
    Ok(r?)
}

/// Save `st` to `path`; returns the file size.
pub fn save(model: &Model, st: &State, path: &Path, opts: SnapOptions) -> Res<u64> {
    let b = encode(model, st, opts)?;
    write_atomic(path, &b)?;
    Ok(b.len() as u64)
}
/// Load a state saved by [`save`].
pub fn load(model: &Model, path: &Path) -> Res<State> { decode(model, &fs::read(path)?) }

/// Rewind `st` to the leading tokens it shares with `tokens` (and at most `tokens.len()`). A cache
/// that cannot be rewound (a sliding-window ring that has wrapped) comes back empty.
/// Returns how many tokens are kept; feed the rest with [`Model::prefill_cached`].
pub fn trim_to_prefix(st: &mut State, tokens: &[usize]) -> Result<usize, ModelError> {
    let common = st.tokens().iter().zip(tokens).enumerate().take_while(|&(i, (a, b))| a == b && st.abs_pos(i) == i).count();
    if common == st.pos { return Ok(common); }
    if st.can_truncate(common) { st.truncate(common)?; Ok(common) } else { st.reset(); Ok(0) }
}
/// [`decode`] then [`trim_to_prefix`]: restore only what is still valid for `tokens`.
pub fn decode_prefix(model: &Model, bytes: &[u8], tokens: &[usize]) -> Res<(State, usize)> {
    let mut st = decode(model, bytes)?;
    let kept = trim_to_prefix(&mut st, tokens)?;
    Ok((st, kept))
}
/// [`load`] then [`trim_to_prefix`].
pub fn load_prefix(model: &Model, path: &Path, tokens: &[usize]) -> Res<(State, usize)> { decode_prefix(model, &fs::read(path)?, tokens) }

// ---------------------------------------------------------------- store

/// A directory of snapshots capped in total size, evicting the least recently used. A snapshot is
/// named after the model, the number of tokens it holds and a hash of those tokens, so the one
/// holding the longest prefix of a prompt is found without opening any file. Safe to share between
/// threads and processes: files appear atomically, and a file deleted under a reader is a miss.
pub struct SnapshotStore {
    dir: PathBuf,
    cap: u64,
    clock: Mutex<SystemTime>,
}

fn token_key(tokens: &[usize]) -> String {
    let mut h = Sha256::new();
    for t in tokens { h.update(&(*t as u32).to_le_bytes()); }
    crate::hub::hex(&h.finish()[..16])
}
fn entry_name(model: &Model, tokens: &[usize]) -> String {
    format!("{}-{:012}-{}.snap", crate::hub::hex(&model.fingerprint()[..8]), tokens.len(), token_key(tokens))
}
struct Entry { name: String, size: u64, mtime: SystemTime }

impl SnapshotStore {
    /// Open (creating) `dir`, keeping at most `cap_bytes` of snapshots in it.
    pub fn open(dir: impl Into<PathBuf>, cap_bytes: u64) -> Res<SnapshotStore> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        let s = SnapshotStore { dir, cap: cap_bytes, clock: Mutex::new(SystemTime::UNIX_EPOCH) };
        // leftovers of a crashed writer
        if let Ok(rd) = fs::read_dir(&s.dir) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|a| a > Duration::from_secs(3600));
                if n.ends_with(".tmp") && old { let _ = fs::remove_file(e.path()); }
            }
        }
        let newest = s.entries().iter().map(|e| e.mtime).max().unwrap_or(SystemTime::UNIX_EPOCH);
        *s.clock.lock().unwrap_or_else(|e| e.into_inner()) = newest;
        Ok(s)
    }
    pub fn dir(&self) -> &Path { &self.dir }
    pub fn capacity(&self) -> u64 { self.cap }

    fn entries(&self) -> Vec<Entry> {
        let Ok(rd) = fs::read_dir(&self.dir) else { return vec![] };
        rd.flatten().filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !name.ends_with(".snap") { return None; }
            let m = e.metadata().ok()?;
            Some(Entry { name, size: m.len(), mtime: m.modified().ok()? })
        }).collect()
    }
    /// Total bytes of snapshots held.
    pub fn used_bytes(&self) -> u64 { self.entries().iter().map(|e| e.size).sum() }
    /// Number of snapshots held.
    pub fn len(&self) -> usize { self.entries().len() }
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// A modification time later than every one handed out before, so recency is a strict order
    /// even on file systems with coarse timestamps.
    fn tick(&self) -> SystemTime {
        let mut c = self.clock.lock().unwrap_or_else(|e| e.into_inner());
        let t = SystemTime::now().max(*c + Duration::from_millis(2));
        *c = t;
        t
    }
    fn touch(&self, name: &str) {
        let t = self.tick();
        if let Ok(f) = fs::OpenOptions::new().write(true).open(self.dir.join(name)) { let _ = f.set_modified(t); }
    }
    fn evict(&self) {
        let _g = self.clock.lock().unwrap_or_else(|e| e.into_inner());
        let mut es = self.entries();
        let mut total: u64 = es.iter().map(|e| e.size).sum();
        es.sort_by(|a, b| a.mtime.cmp(&b.mtime).then_with(|| a.name.cmp(&b.name)));
        for e in es {
            if total <= self.cap { break; }
            match fs::remove_file(self.dir.join(&e.name)) {
                Ok(()) => total = total.saturating_sub(e.size),
                Err(x) if x.kind() == std::io::ErrorKind::NotFound => total = total.saturating_sub(e.size),
                Err(_) => {}
            }
        }
    }

    /// Store `st` under the hash of its token history. `Ok(false)` when it alone is bigger than the
    /// cap. A state that has been through eviction is refused: its slots are not at their own
    /// positions, so it is no prefix of any prompt.
    pub fn put(&self, model: &Model, st: &State, opts: SnapOptions) -> Res<bool> {
        if st.is_compacted() { return Err(SnapshotError::Unsupported("a cache that went through eviction cannot be stored by prompt prefix".into())); }
        if st.pos == 0 { return Err(SnapshotError::Unsupported("an empty state is not worth storing".into())); }
        let bytes = encode(model, st, opts)?;
        if bytes.len() as u64 > self.cap { return Ok(false); }
        let name = entry_name(model, st.tokens());
        write_atomic(&self.dir.join(&name), &bytes)?;
        self.touch(&name);
        self.evict();
        Ok(true)
    }

    fn read_entry(&self, model: &Model, name: &str, tokens: &[usize]) -> Res<Option<State>> {
        let path = self.dir.join(name);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        match decode(model, &bytes) {
            Ok(st) if st.tokens() == tokens => { self.touch(name); Ok(Some(st)) }
            Ok(_) => { let _ = fs::remove_file(&path); Ok(None) }
            Err(SnapshotError::WrongModel) => Ok(None),
            Err(SnapshotError::Io(e)) => Err(SnapshotError::Io(e)),
            Err(_) => { let _ = fs::remove_file(&path); Ok(None) }
        }
    }
    /// The snapshot holding exactly `tokens`, if any (and refreshing its recency).
    pub fn get(&self, model: &Model, tokens: &[usize]) -> Res<Option<State>> {
        if tokens.is_empty() { return Ok(None); }
        self.read_entry(model, &entry_name(model, tokens), tokens)
    }
    /// The longest stored prefix of `tokens`: its state and length. Damaged files met on the way are
    /// deleted. Feed the rest with [`Model::prefill_cached`] (which always recomputes the last token,
    /// and starts over for a sliding-window cache that has wrapped).
    pub fn best_prefix(&self, model: &Model, tokens: &[usize]) -> Res<Option<(State, usize)>> {
        let fp = crate::hub::hex(&model.fingerprint()[..8]);
        let mut lens: Vec<usize> = self.entries().iter().filter_map(|e| {
            let mut it = e.name.trim_end_matches(".snap").splitn(3, '-');
            (it.next()? == fp).then_some(())?;
            it.next()?.parse::<usize>().ok()
        }).filter(|&l| l > 0 && l <= tokens.len()).collect();
        lens.sort_unstable_by(|a, b| b.cmp(a));
        lens.dedup();
        for l in lens {
            if let Some(st) = self.read_entry(model, &entry_name(model, &tokens[..l]), &tokens[..l])? { return Ok(Some((st, l))); }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kvpolicy::{LayerBudgets, StreamingLlm};
    use crate::model::tests::{base_spec, model_of, random_weights, tiny};

    const KINDS: [KvKind; 3] = [KvKind::F32, KvKind::F16, KvKind::Q8];
    fn prompt(n: usize) -> Vec<usize> { (0..n).map(|i| (i * 7 + 3) % 19 + 1).collect() }

    struct Tmp(PathBuf);
    impl Tmp {
        fn new(tag: &str) -> Tmp {
            let n = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
            let p = std::env::temp_dir().join(format!("cmp-snap-{tag}-{}-{n}", std::process::id()));
            fs::create_dir_all(&p).unwrap();
            Tmp(p)
        }
    }
    impl Drop for Tmp { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

    fn models() -> Vec<(&'static str, Model)> {
        let mut sw = base_spec(4, 2, true);
        sw.sliding_window = 5; sw.sliding_layers = vec![true, false, true]; sw.rope_theta_local = 500.0;
        vec![("dense", tiny(4, 2, false)), ("sliding", model_of(sw, false)), ("moe", crate::model::tests::accel_and_prefix::moe_model())]
    }
    fn continue_with(m: &Model, st: &mut State, n: usize) -> Vec<Vec<f32>> { (0..n).map(|i| m.forward(st, 1 + (i * 5) % 18).unwrap()).collect() }

    #[test]
    fn a_round_trip_continues_bit_exactly_for_every_precision_and_model_kind() {
        for (name, m) in models() {
            for kind in KINDS {
                for compress in [false, true] {
                    // 7 tokens: the sliding ring (5) has wrapped; 4 tokens: it has not
                    for n in [4usize, 7, 13] {
                        let mut st = m.new_state_with(kind);
                        m.forward_batch(&mut st, &prompt(n)).unwrap();
                        let bytes = encode(&m, &st, SnapOptions { compress }).unwrap();
                        let mut back = decode(&m, &bytes).unwrap();
                        assert_eq!((back.pos, back.tokens(), back.kv_bytes(), back.kv_kind()), (st.pos, st.tokens(), st.kv_bytes(), kind), "{name} {kind:?}");
                        let want = continue_with(&m, &mut st, 6);
                        assert_eq!(continue_with(&m, &mut back, 6), want, "{name} {kind:?} compress={compress} n={n}: logits after a round trip differ");
                    }
                }
            }
        }
    }

    #[test]
    fn a_state_that_went_through_eviction_round_trips_with_its_positions() {
        let m = tiny(4, 2, false);
        let mut st = m.new_state_with(KvKind::Q8);
        m.forward_batch(&mut st, &prompt(20)).unwrap();
        st.apply_policy(&StreamingLlm { sinks: 3 }, 9, &LayerBudgets::Uniform).unwrap();
        let mut back = decode(&m, &encode(&m, &st, SnapOptions { compress: true }).unwrap()).unwrap();
        assert_eq!((back.rope_gap(), back.abs_pos(3), back.is_compacted()), (st.rope_gap(), st.abs_pos(3), true));
        assert_eq!(continue_with(&m, &mut back, 5), continue_with(&m, &mut st, 5));
    }

    #[test]
    fn files_are_written_atomically_and_load_back() {
        let t = Tmp::new("file");
        let m = tiny(4, 2, false);
        let mut st = m.new_state();
        m.forward_batch(&mut st, &prompt(9)).unwrap();
        let p = t.0.join("chat.snap");
        let size = save(&m, &st, &p, SnapOptions::default()).unwrap();
        assert_eq!(fs::metadata(&p).unwrap().len(), size);
        assert!(fs::read_dir(&t.0).unwrap().flatten().all(|e| !e.file_name().to_string_lossy().ends_with(".tmp")), "no temporary file is left behind");
        let mut back = load(&m, &p).unwrap();
        assert_eq!(continue_with(&m, &mut back, 3), continue_with(&m, &mut st, 3));
        // overwriting replaces the file in one step
        let mut st2 = m.new_state();
        m.forward_batch(&mut st2, &prompt(5)).unwrap();
        save(&m, &st2, &p, SnapOptions { compress: true }).unwrap();
        assert_eq!(load(&m, &p).unwrap().pos, 5);
        assert!(matches!(load(&m, &t.0.join("missing.snap")), Err(SnapshotError::Io(_))));
    }

    /// Recompute the checksum so only the field under test is wrong.
    fn reseal(mut b: Vec<u8>) -> Vec<u8> {
        let n = b.len() - TRAILER;
        let h = digest(&[&b[..n]]);
        b[n..].copy_from_slice(&h);
        b
    }

    #[test]
    fn damaged_truncated_foreign_and_forged_files_are_errors() {
        let m = tiny(4, 2, false);
        let mut st = m.new_state_with(KvKind::F16);
        m.forward_batch(&mut st, &prompt(11)).unwrap();
        for compress in [false, true] {
            let good = encode(&m, &st, SnapOptions { compress }).unwrap();
            assert!(decode(&m, &good).is_ok());
            // every byte, flipped one at a time (sampled), is caught
            for i in (0..good.len()).step_by(7).chain([0, 8, 12, 16, 48, 56, 63, 64, good.len() - 1]) {
                let mut b = good.clone();
                b[i] ^= 0x40;
                assert!(decode(&m, &b).is_err(), "flip at {i} (compress={compress}) went unnoticed");
            }
            // cut anywhere
            for n in [0, 1, 7, 8, 63, 64, 95, good.len() / 2, good.len() - 33, good.len() - 1] {
                assert!(decode(&m, &good[..n]).is_err(), "cut to {n} accepted");
            }
            // appended garbage
            let mut b = good.clone();
            b.extend_from_slice(b"junk");
            assert!(decode(&m, &b).is_err());
            // a checksum that is right but a version that is not
            let mut b = good.clone();
            b[8] = 9;
            assert!(matches!(decode(&m, &reseal(b)), Err(SnapshotError::Version(9))));
            // unknown flag, wrong stored length, wrong raw length: all internally "valid" files, all rejected
            let mut b = good.clone();
            b[12] |= 0x80;
            assert!(decode(&m, &reseal(b)).is_err());
            let mut b = good.clone();
            b[56] ^= 1;
            assert!(decode(&m, &reseal(b)).is_err());
            let mut b = good.clone();
            b[48] ^= 1;
            assert!(decode(&m, &reseal(b)).is_err());
            // a payload that lies about its contents (token count huge)
            let mut b = good.clone();
            b[HEADER] = 0xff; b[HEADER + 1] = 0xff;
            if !compress { assert!(decode(&m, &reseal(b)).is_err()); }
        }
        assert!(decode(&m, b"").is_err() && decode(&m, &[0u8; 200]).is_err());
        // another model: different weights of the same shape are not told apart, a different shape is
        let good = encode(&m, &st, SnapOptions::default()).unwrap();
        let other_shape = tiny(2, 2, false);
        assert!(matches!(decode(&other_shape, &good), Err(SnapshotError::WrongModel)));
        let mut sp = base_spec(4, 2, false);
        sp.layers = 2;
        let fewer = Model::new(sp.clone(), random_weights(&sp, 7, false)).unwrap();
        assert!(matches!(decode(&fewer, &good), Err(SnapshotError::WrongModel)));
        let mut sp = base_spec(4, 2, false);
        sp.rope_theta = 12345.0;
        let theta = Model::new(sp.clone(), random_weights(&sp, 7, false)).unwrap();
        assert!(matches!(decode(&theta, &good), Err(SnapshotError::WrongModel)));
        assert_ne!(m.fingerprint(), other_shape.fingerprint());
        assert_eq!(m.fingerprint(), tiny(4, 2, false).fingerprint());
    }

    #[test]
    fn compression_round_trips_arbitrary_bytes_and_rejects_garbage_without_panicking() {
        let mut seed = 0x1234_5678_9abc_def1_u64;
        let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        let mut cases: Vec<Vec<u8>> = vec![vec![], vec![7], vec![0; 1000], (0..=255u8).cycle().take(5000).collect(), b"abcabcabcabcabcabcabcabc".to_vec()];
        cases.push((0..4000).map(|_| (next() % 3) as u8).collect());
        cases.push((0..3000).map(|_| next() as u8).collect());
        cases.push((0..70_000).map(|i| ((i / 300) % 251) as u8).collect());
        for c in &cases {
            assert_eq!(&lz_decompress(&lz_compress(c), c.len()).unwrap(), c);
            if c.len() % 4 == 0 { assert_eq!(&unshuffle(&shuffle(c, 4), 4), c); }
        }
        assert!(lz_compress(&vec![0u8; 10_000]).len() < 200);
        for _ in 0..300 {
            let n = (next() % 200) as usize;
            let junk: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let _ = lz_decompress(&junk, (next() % 500) as usize);
        }
        assert!(lz_decompress(&lz_compress(b"hello hello hello hello"), 5).is_err());
        assert!(lz_decompress(&lz_compress(b"hello hello hello hello"), 100).is_err());
    }

    #[test]
    fn partial_restore_keeps_only_the_common_prefix_and_resumes_with_cached_prefill() {
        let m = tiny(4, 2, false);
        let full = prompt(14);
        let mut st = m.new_state();
        m.forward_batch(&mut st, &full).unwrap();
        let bytes = encode(&m, &st, SnapOptions::default()).unwrap();
        let mut asked = full[..8].to_vec();
        asked.extend_from_slice(&[2, 4, 6, 8, 10]);
        let (mut part, kept) = decode_prefix(&m, &bytes, &asked).unwrap();
        assert_eq!((kept, part.pos, part.tokens()), (8, 8, &full[..8]));
        let got = m.prefill_cached(&mut part, &asked).unwrap();
        let want = m.forward_batch(&mut m.new_state(), &asked).unwrap();
        assert_eq!(part.reused(), 8);
        let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 1e-4, "{err}");
        // a request that extends the snapshot reuses all of it; an unrelated one reuses nothing
        let mut longer = full.clone();
        longer.push(5);
        assert_eq!(decode_prefix(&m, &bytes, &longer).unwrap().1, 14);
        assert_eq!(decode_prefix(&m, &bytes, &[19, 19]).unwrap().1, 0);
        // a wrapped sliding-window ring cannot be rewound: it restores empty rather than wrong
        let mut sw = base_spec(4, 2, true);
        sw.sliding_window = 3; sw.sliding_layers = vec![true, false, true]; sw.rope_theta_local = 500.0;
        let m = model_of(sw, false);
        let mut st = m.new_state();
        m.forward_batch(&mut st, &full).unwrap();
        let bytes = encode(&m, &st, SnapOptions::default()).unwrap();
        let (s, kept) = decode_prefix(&m, &bytes, &asked).unwrap();
        assert_eq!((kept, s.pos), (0, 0));
        assert_eq!(decode_prefix(&m, &bytes, &full).unwrap().1, 14, "the whole snapshot is still valid for its own prompt");
    }

    fn filled(m: &Model, toks: &[usize]) -> State { let mut s = m.new_state(); m.forward_batch(&mut s, toks).unwrap(); s }

    #[test]
    fn the_store_evicts_least_recently_used_and_honours_its_cap() {
        let t = Tmp::new("lru");
        let m = tiny(4, 2, false);
        let (a, b, c) = (prompt(10), prompt(11), prompt(12));
        let one = encode(&m, &filled(&m, &a), SnapOptions::default()).unwrap().len() as u64;
        let store = SnapshotStore::open(&t.0, one * 5 / 2 + 400).unwrap();
        assert!(store.put(&m, &filled(&m, &a), SnapOptions::default()).unwrap());
        assert!(store.put(&m, &filled(&m, &b), SnapOptions::default()).unwrap());
        assert_eq!(store.len(), 2);
        assert!(store.get(&m, &a).unwrap().is_some(), "reading a refreshes it");
        assert!(store.put(&m, &filled(&m, &c), SnapOptions::default()).unwrap());
        assert!(store.used_bytes() <= store.capacity());
        assert!(store.get(&m, &b).unwrap().is_none(), "b was the least recently used");
        assert!(store.get(&m, &a).unwrap().is_some() && store.get(&m, &c).unwrap().is_some());
        // a state bigger than the whole cap is not stored and evicts nothing
        let small = Tmp::new("small");
        let tiny_store = SnapshotStore::open(&small.0, 100).unwrap();
        assert!(!tiny_store.put(&m, &filled(&m, &a), SnapOptions::default()).unwrap());
        assert_eq!(tiny_store.len(), 0);
        // compacted and empty states are refused
        let mut ev = filled(&m, &prompt(16));
        ev.apply_policy(&StreamingLlm { sinks: 2 }, 6, &LayerBudgets::Uniform).unwrap();
        assert!(store.put(&m, &ev, SnapOptions::default()).is_err());
        assert!(store.put(&m, &m.new_state(), SnapOptions::default()).is_err());
        // reopening sees what is there
        let again = SnapshotStore::open(&t.0, store.capacity()).unwrap();
        assert_eq!(again.len(), store.len());
    }

    #[test]
    fn the_store_resumes_a_long_document_from_its_longest_stored_prefix() {
        let t = Tmp::new("prefix");
        let m = tiny(4, 2, false);
        let doc = prompt(20);
        let store = SnapshotStore::open(&t.0, 1 << 24).unwrap();
        store.put(&m, &filled(&m, &doc[..6]), SnapOptions { compress: true }).unwrap();
        store.put(&m, &filled(&m, &doc[..14]), SnapOptions::default()).unwrap();
        let mut question = doc[..14].to_vec();
        question.extend_from_slice(&[3, 1, 4, 1, 5]);
        let (mut st, n) = store.best_prefix(&m, &question).unwrap().unwrap();
        assert_eq!((n, st.pos), (14, 14));
        let got = m.prefill_cached(&mut st, &question).unwrap();
        assert_eq!(st.reused(), 14);
        let want = m.forward_batch(&mut m.new_state(), &question).unwrap();
        assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-4));
        // a prompt that diverges at token 10 matches the shorter snapshot only
        let mut other = doc[..10].to_vec();
        other.extend_from_slice(&[9, 9, 9, 9]);
        assert_eq!(store.best_prefix(&m, &other).unwrap().unwrap().1, 6);
        assert!(store.best_prefix(&m, &[19, 19, 19]).unwrap().is_none());
        assert!(store.best_prefix(&m, &[]).unwrap().is_none());
        // another model sees nothing
        assert!(store.best_prefix(&tiny(2, 2, false), &question).unwrap().is_none());
    }

    #[test]
    fn damaged_entries_are_dropped_and_a_foreign_model_is_left_alone() {
        let t = Tmp::new("bad");
        let m = tiny(4, 2, false);
        let a = prompt(9);
        let store = SnapshotStore::open(&t.0, 1 << 24).unwrap();
        store.put(&m, &filled(&m, &a), SnapOptions::default()).unwrap();
        let file = t.0.join(entry_name(&m, &a));
        let mut bytes = fs::read(&file).unwrap();
        let n = bytes.len();
        bytes[n / 2] ^= 1;
        fs::write(&file, &bytes).unwrap();
        assert!(store.get(&m, &a).unwrap().is_none());
        assert!(!file.exists(), "a damaged entry is removed so it is not tried again");
        // truncated entry
        store.put(&m, &filled(&m, &a), SnapOptions::default()).unwrap();
        let good = fs::read(&file).unwrap();
        fs::write(&file, &good[..good.len() - 40]).unwrap();
        assert!(store.best_prefix(&m, &a).unwrap().is_none());
        assert!(!file.exists());
        // an entry filed under another model's name is a miss for the others but is kept
        store.put(&m, &filled(&m, &a), SnapOptions::default()).unwrap();
        let other = tiny(2, 2, false);
        assert!(store.get(&other, &a).unwrap().is_none());
        assert!(store.get(&m, &a).unwrap().is_some());
        // a renamed file (name says one prompt, content holds another) is not trusted
        let b = prompt(7);
        fs::copy(&file, t.0.join(entry_name(&m, &b))).unwrap();
        assert!(store.get(&m, &b).unwrap().is_none());
    }

    #[test]
    fn concurrent_writers_and_readers_never_see_a_torn_file() {
        let t = Tmp::new("conc");
        let m = tiny(4, 2, false);
        let seqs: Vec<Vec<usize>> = (0..6).map(|i| (0..8 + i).map(|j| (j * (i + 3) + i) % 19 + 1).collect()).collect();
        let states: Vec<State> = seqs.iter().map(|s| filled(&m, s)).collect();
        let want: Vec<Vec<f32>> = states.iter().map(|s| m.forward(&mut s.fork().unwrap(), 2).unwrap()).collect();
        let one = encode(&m, &states[0], SnapOptions::default()).unwrap().len() as u64;
        let store = SnapshotStore::open(&t.0, one * 3).unwrap();
        std::thread::scope(|sc| {
            for w in 0..6 {
                let (store, m, seqs, states, want) = (&store, &m, &seqs, &states, &want);
                sc.spawn(move || {
                    for round in 0..12 {
                        let i = (w + round) % seqs.len();
                        store.put(m, &states[i], SnapOptions { compress: round % 2 == 0 }).unwrap();
                        let j = (w * 5 + round * 3) % seqs.len();
                        if let Some(mut s) = store.get(m, &seqs[j]).unwrap() {
                            assert_eq!(s.tokens(), seqs[j].as_slice());
                            assert_eq!(&m.forward(&mut s, 2).unwrap(), &want[j], "a snapshot read while others write must be whole");
                        }
                        let _ = store.best_prefix(m, &seqs[j]).unwrap();
                    }
                });
            }
        });
        assert!(store.used_bytes() <= store.capacity() + one, "cap respected up to a write in flight: {} > {}", store.used_bytes(), store.capacity());
        store.put(&m, &states[0], SnapOptions::default()).unwrap();
        assert!(store.used_bytes() <= store.capacity());
        let leftovers = fs::read_dir(&t.0).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().ends_with(".tmp")).count();
        assert_eq!(leftovers, 0);
    }

    #[test]
    fn fingerprints_follow_the_shape_of_the_tensors() {
        let a = tiny(4, 2, false);
        let q = crate::model::tests::requantise(&a, crate::quant::DType::BF16);
        assert_ne!(a.fingerprint(), q.fingerprint(), "the same architecture in another storage type is another model");
        assert_eq!(q.fingerprint(), crate::model::tests::requantise(&tiny(4, 2, false), crate::quant::DType::BF16).fingerprint());
        let moe = crate::model::tests::accel_and_prefix::moe_model();
        assert_ne!(moe.fingerprint(), a.fingerprint());
    }
}
