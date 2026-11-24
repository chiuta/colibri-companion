#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]
//! Shared machinery for the engine's adversarial suite (`adversarial.rs` includes this as a module).
//!
//! * a deterministic xorshift generator, text and byte mutators with nasty Unicode and extreme
//!   integers (the kind that make size arithmetic overflow);
//! * `hammer_*` runners: every call runs under `catch_unwind`, is timed (5 s cap by default), is
//!   watched by a watchdog thread (a hang kills the process with a message instead of stalling CI)
//!   and is measured by a counting allocator: one call may not allocate more than `MAX_PEAK`, and
//!   no single allocation request above `MAX_REQUEST` is ever granted, so a header claiming 2^60
//!   elements cannot take the machine down;
//! * builders for tiny but valid GGUF / safetensors / tokenizer files to mutate.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

// ------------------------------------------------------------------ counting allocator

/// One call may not hold more than this at its peak.
pub const MAX_PEAK: usize = 768 << 20;
/// A single allocation request above this is refused (the process then aborts with a message).
pub const MAX_REQUEST: usize = 1 << 30;

static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

pub struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if l.size() > MAX_REQUEST {
            let _ = std::io::stderr().write_all(
                b"ALLOCATION REFUSED: a single request above 1 GiB (hostile size not bounded?)\n",
            );
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            grow(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if l.size() > MAX_REQUEST {
            let _ = std::io::stderr().write_all(
                b"ALLOCATION REFUSED: a single request above 1 GiB (hostile size not bounded?)\n",
            );
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            grow(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        CUR.fetch_sub(l.size(), Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        if new > MAX_REQUEST {
            let _ = std::io::stderr().write_all(
                b"ALLOCATION REFUSED: a single request above 1 GiB (hostile size not bounded?)\n",
            );
            return std::ptr::null_mut();
        }
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            CUR.fetch_sub(l.size(), Ordering::Relaxed);
            grow(new);
        }
        q
    }
}
fn grow(n: usize) {
    let c = CUR.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(c, Ordering::Relaxed);
    LARGEST.fetch_max(n, Ordering::Relaxed);
}

#[global_allocator]
static ALLOC: Counting = Counting;

// ---------------------------------------------------------------------------- generator

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
    pub fn chance(&mut self, one_in: usize) -> bool {
        self.below(one_in) == 0
    }
    pub fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6
    }
    /// A float that is often not an ordinary number.
    pub fn nasty_f32(&mut self) -> f32 {
        match self.below(12) {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => f32::MAX,
            4 => f32::MIN,
            5 => f32::MIN_POSITIVE / 4.0,
            6 => -0.0,
            7 => 1e30,
            _ => self.f32(),
        }
    }
    pub fn extreme_u64(&mut self) -> u64 {
        const E: [u64; 14] = [
            0,
            1,
            2,
            3,
            7,
            255,
            65_535,
            1 << 31,
            1 << 32,
            (1 << 32) + 1,
            1 << 40,
            1 << 60,
            1 << 63,
            u64::MAX,
        ];
        E[self.below(E.len())]
    }
}

pub fn iterations() -> usize {
    std::env::var("ADVERSARIAL_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500)
}

pub const NASTY: &[&str] = &[
    "\0",
    "\u{feff}",
    "\u{202e}",
    "\u{200b}",
    "é",
    "e\u{301}",
    "😀",
    "𝕏",
    "\u{10ffff}",
    "\r\n",
    "\\",
    "\"",
    "'",
    "{",
    "}",
    "[",
    "]",
    "(",
    ")",
    "<",
    ">",
    "&",
    "%",
    "$",
    "..",
    "../",
    "..\\",
    "/",
    "\\\\",
    "C:\\",
    "-",
    ":",
    ";",
    ",",
    "\t",
    "\u{1b}[31m",
    "9999999999999999999999",
    "-1",
    "1e999",
    "NaN",
    "null",
    "true",
    "\u{d55c}",
    "\u{1100}\u{1161}\u{11a8}",
    "ａ",
    "ǆ",
    "\u{0301}\u{0301}\u{0301}",
    "ſ",
    "İ",
    "ß",
    "▁",
    "Ġ",
    "<0x41>",
    "<unk>",
    "<s>",
    "</s>",
    "<|endoftext|>",
    "<0xFF>",
    "�",
];

pub fn random_text(r: &mut Rng) -> String {
    let n = r.below(300);
    let mut s = String::new();
    for _ in 0..n {
        match r.below(7) {
            0 => s.push_str(NASTY[r.below(NASTY.len())]),
            1 => s.push(char::from_u32(r.next_u64() as u32 % 0x30000).unwrap_or('x')),
            2 => s.push((b' ' + (r.next_u64() % 95) as u8) as char),
            3 => s.push('\n'),
            4 => s.push_str(&r.below(100_000).to_string()),
            _ => s.push((b'a' + (r.next_u64() % 26) as u8) as char),
        }
    }
    s
}

pub fn mutate_str(seed: &str, r: &mut Rng) -> String {
    let mut v: Vec<char> = seed.chars().collect();
    for _ in 0..1 + r.below(8) {
        let at = r.below(v.len() + 1);
        match r.below(6) {
            0 if !v.is_empty() => {
                v.remove(at.min(v.len() - 1));
            }
            1 => {
                for c in NASTY[r.below(NASTY.len())].chars().rev() {
                    v.insert(at, c);
                }
            }
            2 if !v.is_empty() => {
                let c = v[r.below(v.len())];
                for _ in 0..1 + r.below(300) {
                    v.insert(at, c);
                }
            }
            3 => v.truncate(at),
            4 => {
                // replace a run of digits with an extreme number
                if let Some(i) = v
                    .iter()
                    .skip(at)
                    .position(|c| c.is_ascii_digit())
                    .map(|p| p + at)
                {
                    let mut j = i;
                    while j < v.len() && v[j].is_ascii_digit() {
                        j += 1;
                    }
                    let e: Vec<char> = r.extreme_u64().to_string().chars().collect();
                    v.splice(i..j, e);
                }
            }
            _ if v.len() > 2 => {
                let (a, b) = (r.below(v.len()), r.below(v.len()));
                v.swap(a, b);
            }
            _ => {}
        }
    }
    v.into_iter().collect()
}

pub fn mutate_bytes(seed: &[u8], r: &mut Rng) -> Vec<u8> {
    let mut v = seed.to_vec();
    for _ in 0..1 + r.below(6) {
        let at = r.below(v.len() + 1);
        match r.below(8) {
            0 if !v.is_empty() => {
                let i = at.min(v.len() - 1);
                v[i] ^= 1 << r.below(8);
            }
            1 if !v.is_empty() => {
                let i = at.min(v.len() - 1);
                v[i] = r.next_u64() as u8;
            }
            2 if v.len() >= 8 => {
                let i = at.min(v.len() - 8);
                v[i..i + 8].copy_from_slice(&r.extreme_u64().to_le_bytes());
            }
            3 if v.len() >= 4 => {
                let i = at.min(v.len() - 4);
                v[i..i + 4].copy_from_slice(&(r.extreme_u64() as u32).to_le_bytes());
            }
            4 => v.truncate(at),
            5 => {
                for _ in 0..1 + r.below(64) {
                    v.insert(at, r.next_u64() as u8);
                }
            }
            6 if !v.is_empty() => {
                let a = r.below(v.len());
                let b = (a + 1 + r.below(64)).min(v.len());
                let chunk = v[a..b].to_vec();
                for (k, c) in chunk.into_iter().enumerate() {
                    v.insert(at.min(v.len()) + k, c);
                }
            }
            _ if !v.is_empty() => {
                let a = r.below(v.len());
                let b = (a + 1 + r.below(64)).min(v.len());
                v.drain(a..b);
            }
            _ => {}
        }
    }
    v
}

pub fn deep(open: &str, close: &str, n: usize) -> String {
    format!("{}{}", open.repeat(n), close.repeat(n))
}

// --------------------------------------------------------------------------- the runners

static SERIAL: Mutex<()> = Mutex::new(());
static CALL_START_MS: AtomicU64 = AtomicU64::new(0);
static BASE: OnceLock<Instant> = OnceLock::new();

fn now_ms() -> u64 {
    BASE.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1
}

fn watchdog() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        std::thread::spawn(|| {
            loop {
                std::thread::sleep(Duration::from_secs(2));
                let s = CALL_START_MS.load(Ordering::Relaxed);
                if s != 0 && now_ms().saturating_sub(s) > 120_000 {
                    eprintln!("WATCHDOG: one call has been running for more than 120 s: it hangs");
                    std::process::exit(101);
                }
            }
        });
    });
}

/// Runs one call with all the guards; returns a description of what went wrong, if anything.
fn guarded<R>(cap: Duration, f: impl FnOnce() -> R) -> Result<R, String> {
    watchdog();
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    PEAK.store(CUR.load(Ordering::Relaxed), Ordering::Relaxed);
    LARGEST.store(0, Ordering::Relaxed);
    let base = CUR.load(Ordering::Relaxed);
    CALL_START_MS.store(now_ms(), Ordering::Relaxed);
    let t = Instant::now();
    let res = catch_unwind(AssertUnwindSafe(f));
    let d = t.elapsed();
    CALL_START_MS.store(0, Ordering::Relaxed);
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(base);
    match res {
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            Err(format!("panicked: {msg}"))
        }
        Ok(_) if d > cap => Err(format!("took {d:?} (cap {cap:?})")),
        Ok(_) if peak > MAX_PEAK => Err(format!(
            "allocated {} MiB at its peak (cap {} MiB)",
            peak >> 20,
            MAX_PEAK >> 20
        )),
        Ok(r) => Ok(r),
    }
}

fn shown(b: &[u8]) -> String {
    let s = String::from_utf8_lossy(&b[..b.len().min(300)]);
    format!("{} bytes: {s:?}", b.len())
}

/// Runs `f` on many hostile strings derived from `seeds`.
pub fn hammer_str(name: &str, seeds: &[&str], f: impl Fn(&str)) {
    hammer_str_cap(name, Duration::from_secs(5), seeds, f)
}

pub fn hammer_str_cap(name: &str, cap: Duration, seeds: &[&str], f: impl Fn(&str)) {
    let n = iterations();
    let mut r =
        Rng::new(0xE761 ^ name.len() as u64 ^ (name.bytes().map(|b| b as u64).sum::<u64>() << 8));
    let mut inputs: Vec<String> = vec![
        String::new(),
        "\0".repeat(1000),
        "a".repeat(200_000),
        deep("(", ")", 20_000),
        deep("[", "]", 20_000),
        deep("{", "}", 20_000),
        deep("{\"a\":", "}", 5_000),
        "\n".repeat(100_000),
        "😀".repeat(50_000),
        "9".repeat(100_000),
        "-".repeat(100_000),
        "\u{301}".repeat(50_000),
        " ".repeat(100_000),
        "../".repeat(10_000),
    ];
    inputs.extend(seeds.iter().map(|s| s.to_string()));
    for i in 0..n {
        inputs.push(if seeds.is_empty() || i % 4 == 0 {
            random_text(&mut r)
        } else {
            mutate_str(seeds[r.below(seeds.len())], &mut r)
        });
    }
    let mut worst = Duration::ZERO;
    for (i, inp) in inputs.iter().enumerate() {
        let t = Instant::now();
        if let Err(why) = guarded(cap, || f(inp)) {
            panic!("{name}: input #{i} ({}): {why}", shown(inp.as_bytes()));
        }
        worst = worst.max(t.elapsed());
    }
    eprintln!("{name}: {} inputs, slowest {worst:?}", inputs.len());
}

/// Runs `f` on many hostile byte strings derived from `seeds` (valid files to mutate).
pub fn hammer_bytes(name: &str, seeds: &[Vec<u8>], f: impl Fn(&[u8])) {
    hammer_bytes_cap(name, Duration::from_secs(5), seeds, f)
}

pub fn hammer_bytes_cap(name: &str, cap: Duration, seeds: &[Vec<u8>], f: impl Fn(&[u8])) {
    let n = iterations();
    let mut r =
        Rng::new(0xB17E5 ^ name.len() as u64 ^ (name.bytes().map(|b| b as u64).sum::<u64>() << 8));
    let mut inputs: Vec<Vec<u8>> = vec![vec![], vec![0; 1000], vec![0xff; 1000], vec![0; 100_000]];
    inputs.extend(seeds.iter().cloned());
    // every prefix of every seed, thinned
    for s in seeds {
        let step = (s.len() / 200).max(1);
        for k in (0..s.len()).step_by(step) {
            inputs.push(s[..k].to_vec());
        }
    }
    for i in 0..n {
        inputs.push(if seeds.is_empty() || i % 8 == 0 {
            (0..r.below(400)).map(|_| r.next_u64() as u8).collect()
        } else {
            mutate_bytes(&seeds[r.below(seeds.len())], &mut r)
        });
    }
    let mut worst = Duration::ZERO;
    for (i, inp) in inputs.iter().enumerate() {
        let t = Instant::now();
        if let Err(why) = guarded(cap, || f(inp)) {
            panic!("{name}: input #{i} ({}): {why}", shown(inp));
        }
        worst = worst.max(t.elapsed());
    }
    eprintln!("{name}: {} inputs, slowest {worst:?}", inputs.len());
}

/// Runs `f(rng)` `n` times with a fresh generator state each time (for numeric/structured fuzzing).
pub fn hammer_rng(name: &str, f: impl Fn(&mut Rng)) {
    hammer_rng_cap(name, Duration::from_secs(5), f)
}

pub fn hammer_rng_cap(name: &str, cap: Duration, f: impl Fn(&mut Rng)) {
    let n = iterations();
    let mut worst = Duration::ZERO;
    for i in 0..n {
        let seed = (i as u64) ^ ((name.len() as u64) << 32);
        let t = Instant::now();
        if let Err(why) = guarded(cap, || f(&mut Rng::new(seed))) {
            panic!("{name}: case #{i} (Rng::new({seed})): {why}");
        }
        worst = worst.max(t.elapsed());
    }
    eprintln!("{name}: {n} cases, slowest {worst:?}");
}

/// A single guarded call (for hand-written hostile cases).
pub fn once(name: &str, f: impl FnOnce()) {
    if let Err(why) = guarded(Duration::from_secs(10), f) {
        panic!("{name}: {why}");
    }
}

// -------------------------------------------------------------------------- temp folders

pub struct Tmp(pub PathBuf);
impl Tmp {
    pub fn new(tag: &str) -> Tmp {
        static N: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "engine-adv-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).unwrap();
        Tmp(p)
    }
    pub fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }
}
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ----------------------------------------------------------------- tiny model builders

/// The architecture every tiny model file here describes.
pub const HIDDEN: usize = 16;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 4;
pub const FFN: usize = 24;
pub const VOCAB: usize = 20;
pub const CONTEXT: usize = 32;
pub const LAYERS: usize = 2;

pub fn tiny_spec() -> companion_engine::arch::Spec {
    companion_engine::arch::Spec::dense(
        "llama",
        LAYERS as u64,
        HIDDEN as u64,
        HEADS as u64,
        KV_HEADS as u64,
        HEAD_DIM as u64,
        FFN as u64,
        VOCAB as u64,
        CONTEXT as u64,
    )
}

pub const CONFIG_JSON: &str = r#"{"model_type":"llama","hidden_size":16,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,"intermediate_size":24,"vocab_size":20,"max_position_embeddings":32,"rope_theta":10000.0,"rms_norm_eps":1e-5}"#;

pub struct T {
    pub hf: String,
    pub gguf: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn tiny_tensors(seed: u64) -> Vec<T> {
    use companion_engine::loader::{Naming, layer_name};
    let mut r = Rng::new(seed);
    let mut mat = |n: usize| -> Vec<f32> { (0..n).map(|_| r.f32()).collect() };
    let (qd, kd) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let mut out = vec![
        T {
            hf: "model.embed_tokens.weight".into(),
            gguf: "token_embd.weight".into(),
            shape: vec![VOCAB, HIDDEN],
            data: mat(VOCAB * HIDDEN),
        },
        T {
            hf: "model.norm.weight".into(),
            gguf: "output_norm.weight".into(),
            shape: vec![HIDDEN],
            data: mat(HIDDEN).iter().map(|x| 1.0 + x).collect(),
        },
        T {
            hf: "lm_head.weight".into(),
            gguf: "output.weight".into(),
            shape: vec![VOCAB, HIDDEN],
            data: mat(VOCAB * HIDDEN),
        },
    ];
    for i in 0..LAYERS {
        for (part, shape) in [
            ("attn_norm", vec![HIDDEN]),
            ("wq", vec![qd, HIDDEN]),
            ("wk", vec![kd, HIDDEN]),
            ("wv", vec![kd, HIDDEN]),
            ("wo", vec![HIDDEN, qd]),
            ("ffn_norm", vec![HIDDEN]),
            ("w_gate", vec![FFN, HIDDEN]),
            ("w_up", vec![FFN, HIDDEN]),
            ("w_down", vec![HIDDEN, FFN]),
        ] {
            let n: usize = shape.iter().product();
            let data = if shape.len() == 1 {
                mat(n).iter().map(|x| 1.0 + x).collect()
            } else {
                mat(n)
            };
            out.push(T {
                hf: layer_name(Naming::Hf, i, part),
                gguf: layer_name(Naming::Gguf, i, part),
                shape,
                data,
            });
        }
    }
    out
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A valid single-file safetensors with the tiny model (dtype F32, or BF16 when `bf16`).
pub fn safetensors_bytes(seed: u64, bf16: bool) -> Vec<u8> {
    let mut hdr = BTreeMap::new();
    let mut blob = vec![];
    for t in tiny_tensors(seed) {
        let bytes: Vec<u8> = if bf16 {
            t.data
                .iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        } else {
            f32_bytes(&t.data)
        };
        hdr.insert(t.hf.clone(), serde_json::json!({"dtype": if bf16 { "BF16" } else { "F32" }, "shape": t.shape, "data_offsets": [blob.len(), blob.len() + bytes.len()]}));
        blob.extend(bytes);
    }
    hdr.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
    let h = serde_json::to_vec(&hdr).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(blob);
    out
}

pub enum Kv {
    U32(u32),
    U64(u64),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(u32, Vec<Kv>),
}

fn put_str(v: &mut Vec<u8>, s: &str) {
    v.extend((s.len() as u64).to_le_bytes());
    v.extend(s.as_bytes());
}
fn kv_type(k: &Kv) -> u32 {
    match k {
        Kv::U32(_) => 4,
        Kv::I32(_) => 5,
        Kv::F32(_) => 6,
        Kv::Bool(_) => 7,
        Kv::Str(_) => 8,
        Kv::Arr(..) => 9,
        Kv::U64(_) => 10,
    }
}
fn put_kv(v: &mut Vec<u8>, k: &Kv) {
    match k {
        Kv::U32(x) => v.extend(x.to_le_bytes()),
        Kv::I32(x) => v.extend(x.to_le_bytes()),
        Kv::U64(x) => v.extend(x.to_le_bytes()),
        Kv::F32(x) => v.extend(x.to_le_bytes()),
        Kv::Bool(x) => v.push(*x as u8),
        Kv::Str(s) => put_str(v, s),
        Kv::Arr(t, items) => {
            v.extend(t.to_le_bytes());
            v.extend((items.len() as u64).to_le_bytes());
            for i in items {
                put_kv(v, i);
            }
        }
    }
}

/// A GGUF v3 file. `tensors` are (name, dims as GGUF lists them, ggml type id, raw bytes).
pub fn gguf_bytes(kvs: &[(&str, Kv)], tensors: &[(String, Vec<u64>, u32, Vec<u8>)]) -> Vec<u8> {
    let mut v = b"GGUF".to_vec();
    v.extend(3u32.to_le_bytes());
    v.extend((tensors.len() as u64).to_le_bytes());
    v.extend((kvs.len() as u64).to_le_bytes());
    for (k, x) in kvs {
        put_str(&mut v, k);
        v.extend(kv_type(x).to_le_bytes());
        put_kv(&mut v, x);
    }
    let mut data = vec![];
    for (name, dims, ty, bytes) in tensors {
        while data.len() % 32 != 0 {
            data.push(0);
        }
        put_str(&mut v, name);
        v.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            v.extend(d.to_le_bytes());
        }
        v.extend(ty.to_le_bytes());
        v.extend((data.len() as u64).to_le_bytes());
        data.extend(bytes);
    }
    while !v.len().is_multiple_of(32) {
        v.push(0);
    }
    v.extend(data);
    v
}

/// The tiny model as GGUF: all tensors F32 (type 0), or the 2-D ones as F16 (type 1) when `half`.
pub fn tiny_gguf(seed: u64, half: bool) -> Vec<u8> {
    let ts: Vec<_> = tiny_tensors(seed)
        .into_iter()
        .map(|t| {
            let dims: Vec<u64> = t.shape.iter().rev().map(|&d| d as u64).collect();
            if half && t.shape.len() == 2 {
                let b: Vec<u8> = t
                    .data
                    .iter()
                    .flat_map(|x| f16_bits(*x).to_le_bytes())
                    .collect();
                (t.gguf, dims, 1u32, b)
            } else {
                (t.gguf, dims, 0u32, f32_bytes(&t.data))
            }
        })
        .collect();
    gguf_bytes(
        &[
            ("general.architecture", Kv::Str("llama".into())),
            ("llama.block_count", Kv::U32(LAYERS as u32)),
            ("llama.embedding_length", Kv::U32(HIDDEN as u32)),
            ("llama.attention.head_count", Kv::U32(HEADS as u32)),
            ("llama.attention.head_count_kv", Kv::U32(KV_HEADS as u32)),
            ("llama.feed_forward_length", Kv::U32(FFN as u32)),
            ("llama.context_length", Kv::U32(CONTEXT as u32)),
            ("llama.rope.freq_base", Kv::F32(10_000.0)),
            ("general.alignment", Kv::U32(32)),
            (
                "tokenizer.ggml.tokens",
                Kv::Arr(8, (0..VOCAB).map(|i| Kv::Str(format!("t{i}"))).collect()),
            ),
        ],
        &ts,
    )
}

pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let (s, e, m) = (
        ((b >> 16) & 0x8000) as u16,
        ((b >> 23) & 0xff) as i32 - 127 + 15,
        b & 0x7f_ffff,
    );
    if e <= 0 {
        s
    } else if e >= 31 {
        s | 0x7c00
    } else {
        s | ((e as u16) << 10) | (m >> 13) as u16
    }
}

/// GPT-2's byte → printable-character table, so a byte-level tokenizer can be written down.
pub fn byte_to_char() -> [char; 256] {
    let mut bs: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut t = ['\0'; 256];
    for (b, c) in bs.into_iter().zip(cs) {
        t[b as usize] = char::from_u32(c).unwrap();
    }
    t
}

/// A byte-level BPE `tokenizer.json` with all 256 byte tokens, a few merges and special tokens:
/// any text encodes, and decode(encode(x)) must give x back.
pub fn byte_level_tokenizer_json() -> String {
    let t = byte_to_char();
    let mut vocab = serde_json::Map::new();
    for b in 0..256 {
        vocab.insert(t[b].to_string(), (b as u64).into());
    }
    let merges = [
        ("h", "e"),
        ("l", "l"),
        ("he", "ll"),
        ("hell", "o"),
        (&t[b' ' as usize].to_string(), "t"),
        ("t", "h"),
    ];
    let mut ms = vec![];
    let mut next = 256u64;
    for (a, b) in merges {
        let joined = format!("{a}{b}");
        if !vocab.contains_key(&joined) {
            vocab.insert(joined, next.into());
            next += 1;
        }
        ms.push(format!("{a} {b}"));
    }
    serde_json::json!({
        "version": "1.0",
        "added_tokens": [{"id": next, "content": "<|endoftext|>", "special": true}, {"id": next + 1, "content": "<|im_start|>", "special": true}],
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": vocab, "merges": ms}
    }).to_string()
}

/// A SentencePiece-style BPE `tokenizer.json` (▁ markers, <0xNN> byte fallback).
pub fn spm_tokenizer_json() -> String {
    let mut vocab = serde_json::Map::new();
    let mut id = 0u64;
    for t in ["<unk>", "<s>", "</s>"] {
        vocab.insert(t.into(), id.into());
        id += 1;
    }
    for b in 0..256 {
        vocab.insert(format!("<0x{b:02X}>"), id.into());
        id += 1;
    }
    for c in [
        "▁", "a", "b", "c", "d", "e", "h", "l", "o", "t", "▁t", "▁th", "he", "ll", "hell", "hello",
    ] {
        vocab.insert(c.into(), id.into());
        id += 1;
    }
    serde_json::json!({
        "added_tokens": [{"id": 0, "content": "<unk>", "special": true}, {"id": 1, "content": "<s>", "special": true}, {"id": 2, "content": "</s>", "special": true}],
        "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
        "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
        "model": {"type": "BPE", "vocab": vocab, "merges": ["▁ t", "▁t h", "h e", "l l", "he ll", "hell o"], "byte_fallback": true, "unk_token": "<unk>"}
    }).to_string()
}

/// Tokenizer metadata as GGUF key/values (byte-level `gpt2` model).
pub fn tokenizer_gguf(model: &str) -> Vec<u8> {
    let t = byte_to_char();
    let mut tokens: Vec<Kv> = (0..256).map(|b| Kv::Str(t[b].to_string())).collect();
    tokens.push(Kv::Str("he".into()));
    tokens.push(Kv::Str("<|endoftext|>".into()));
    let n = tokens.len();
    let mut types: Vec<Kv> = (0..n - 1).map(|_| Kv::I32(1)).collect();
    types.push(Kv::I32(3));
    gguf_bytes(
        &[
            ("general.architecture", Kv::Str("llama".into())),
            ("tokenizer.ggml.model", Kv::Str(model.into())),
            ("tokenizer.ggml.pre", Kv::Str("default".into())),
            ("tokenizer.ggml.tokens", Kv::Arr(8, tokens)),
            ("tokenizer.ggml.token_type", Kv::Arr(5, types)),
            (
                "tokenizer.ggml.merges",
                Kv::Arr(8, vec![Kv::Str("h e".into())]),
            ),
            (
                "tokenizer.ggml.scores",
                Kv::Arr(6, (0..n).map(|i| Kv::F32(-(i as f32))).collect()),
            ),
            ("tokenizer.ggml.bos_token_id", Kv::U32(256)),
            ("tokenizer.ggml.eos_token_id", Kv::U32(257)),
        ],
        &[],
    )
}
