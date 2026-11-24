//! Generation receipts: a small JSON proof of *which model produced which tokens, how*.
//!
//! A [`Receipt`] records the engine version, a fingerprint of the model (SHA-256 over the
//! configuration and a table of tensors, each with a Merkle root over 1 MiB chunks, so a corrupted
//! shard or chunk can be localised), a fingerprint of the tokenizer, the sampler parameters and
//! seed, hashes of the prompt and output tokens, whether reproducible mode was on, the ISA level
//! the run used (not identifying), a timestamp, and a hash chain over the generation steps
//! (optionally including the top-k logits of every step).
//!
//! With reproducible mode on (see [`crate::simd::set_reproducible`]) the same model, prompt and
//! sampler give the same tokens *and* the same logits bits on every machine, so [`replay`] can
//! re-run a receipt and confirm the output byte for byte, or name the first step that differs.
//!
//! What a receipt is **not**: there are no asymmetric signatures here. The `digest` is a plain
//! SHA-256 of the canonical body: it catches accidental edits and any change that was not
//! re-sealed. The optional `mac` is HMAC-SHA256 under a key you hold; only someone with that key
//! can produce or check it. A receipt proves nothing to a third party who does not share the key,
//! except by re-running it ([`replay`]) against the same weights.
//!
//! The text form is canonical JSON: keys sorted, no whitespace, integers as integers, floats as
//! the shortest round-trip decimal. Sampler floats are authoritative as their IEEE bits
//! (`*_bits`); the decimal next to them is for people.
use crate::hub::{hex, Sha256};
use crate::model::{Model, ModelError, Sampler, Weight};
use crate::simd;
use serde_json::{Map, Value};
use std::fmt;

/// Size of one Merkle leaf in bytes.
pub const CHUNK: usize = 1 << 20;
/// Receipt format number.
pub const FORMAT: u64 = 1;
/// Receipts larger than this are refused when parsing.
pub const MAX_JSON_BYTES: usize = 256 << 20;

const MAX_TENSORS: usize = 1 << 20;
const MAX_LEAVES: usize = 1 << 24;

// ------------------------------------------------------------------------------- errors

#[derive(Debug, PartialEq)]
pub enum ReceiptError {
    /// The text is not a valid receipt (bad JSON, wrong type, unknown field, bad hash, …).
    Malformed(String),
    /// The `digest` does not match the body.
    DigestMismatch,
    /// The receipt carries a MAC but the key is wrong, or the key was given but there is no MAC.
    BadMac,
    /// A tensor's Merkle root differs from the receipt (and, with leaves recorded, which chunk).
    TensorMismatch { tensor: String, chunk: Option<usize> },
    /// The model's shape or tensor list differs from the receipt.
    ModelMismatch(String),
    /// The prompt does not hash to the receipt's prompt hash.
    PromptMismatch,
    /// The receipt asks for reproducible mode but it is off in this process.
    ReproducibleModeOff,
    /// This kind of model cannot be fingerprinted yet.
    Unsupported(String),
    Model(String),
}
impl fmt::Display for ReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReceiptError::Malformed(m) => write!(f, "malformed receipt: {m}"),
            ReceiptError::DigestMismatch => write!(f, "receipt digest does not match its contents"),
            ReceiptError::BadMac => write!(f, "receipt MAC missing or wrong"),
            ReceiptError::TensorMismatch { tensor, chunk: Some(c) } => write!(f, "tensor `{tensor}` differs from the receipt in chunk {c}"),
            ReceiptError::TensorMismatch { tensor, chunk: None } => write!(f, "tensor `{tensor}` differs from the receipt"),
            ReceiptError::ModelMismatch(m) => write!(f, "model differs from the receipt: {m}"),
            ReceiptError::PromptMismatch => write!(f, "prompt does not match the receipt"),
            ReceiptError::ReproducibleModeOff => write!(f, "the receipt was made in reproducible mode; turn it on to replay"),
            ReceiptError::Unsupported(m) => write!(f, "unsupported: {m}"),
            ReceiptError::Model(m) => write!(f, "model error: {m}"),
        }
    }
}
impl std::error::Error for ReceiptError {}
impl From<ModelError> for ReceiptError { fn from(e: ModelError) -> Self { ReceiptError::Model(e.to_string()) } }
fn bad<T>(m: impl Into<String>) -> Result<T, ReceiptError> { Err(ReceiptError::Malformed(m.into())) }

// ----------------------------------------------------------------------- SHA-256 / HMAC

/// SHA-256 of `data`.
pub fn sha256(data: &[u8]) -> [u8; 32] { let mut s = Sha256::new(); s.update(data); s.finish() }

/// HMAC-SHA256 (RFC 2104 / 4231).
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 64];
    if key.len() > 64 { k[..32].copy_from_slice(&sha256(key)); } else { k[..key.len()].copy_from_slice(key); }
    let (mut ipad, mut opad) = ([0x36u8; 64], [0x5cu8; 64]);
    for i in 0..64 { ipad[i] ^= k[i]; opad[i] ^= k[i]; }
    let mut inner = Sha256::new();
    inner.update(&ipad);
    inner.update(msg);
    let ih = inner.finish();
    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(&ih);
    outer.finish()
}

fn ct_eq(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

// ------------------------------------------------------------------------- Merkle tree

fn leaf_hash(chunk: &[u8]) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(&[0x00]);
    s.update(chunk);
    s.finish()
}
fn node_hash(l: &[u8; 32], r: &[u8; 32]) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(&[0x01]);
    s.update(l);
    s.update(r);
    s.finish()
}
/// Root over leaf hashes (an odd node is carried up unchanged). No leaves = the hash of one empty chunk.
pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() { return leaf_hash(&[]); }
    let mut level: Vec<[u8; 32]> = leaves.to_vec();
    while level.len() > 1 {
        level = level.chunks(2).map(|p| if p.len() == 2 { node_hash(&p[0], &p[1]) } else { p[0] }).collect();
    }
    level[0]
}
/// Leaf hashes of `data` in [`CHUNK`]-sized pieces.
pub fn merkle_leaves(data: &[u8]) -> Vec<[u8; 32]> { data.chunks(CHUNK).map(leaf_hash).collect() }

// -------------------------------------------------------------------- canonical JSON

fn canon(v: &Value, out: &mut String) -> Result<(), ReceiptError> {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(u) = n.as_u64() { out.push_str(&u.to_string()); }
            else if let Some(i) = n.as_i64() { out.push_str(&i.to_string()); }
            else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                if !f.is_finite() { return bad("non-finite number"); }
                out.push_str(&format!("{f}"));
            }
        }
        Value::String(s) => out.push_str(&serde_json::to_string(s).map_err(|e| ReceiptError::Malformed(e.to_string()))?),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() { if i > 0 { out.push(','); } canon(x, out)?; }
            out.push(']');
        }
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 { out.push(','); }
                out.push_str(&serde_json::to_string(k).map_err(|e| ReceiptError::Malformed(e.to_string()))?);
                out.push(':');
                canon(&m[k], out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}
/// Canonical text of a JSON value: sorted keys, no whitespace, shortest round-trip floats.
pub fn canonical_json(v: &Value) -> Result<String, ReceiptError> { let mut s = String::new(); canon(v, &mut s)?; Ok(s) }

// ---------------------------------------------------------------------- data model

/// One entry of the tensor table.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorRecord {
    pub name: String,
    /// `"f32"` or the quantisation type's name.
    pub kind: String,
    pub rows: u64,
    pub cols: u64,
    pub bytes: u64,
    /// Merkle root over the tensor's 1 MiB chunks (hex).
    pub root: String,
    /// The leaf hashes (hex), when recorded: lets a mismatch name the chunk.
    pub leaves: Option<Vec<String>>,
}

/// Sampler configuration. Floats are compared and stored by their bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerParams { pub temperature: f32, pub top_k: u64, pub top_p: f32, pub repeat_penalty: f32, pub repeat_last_n: u64, pub seed: u64 }
impl SamplerParams {
    pub fn greedy() -> SamplerParams { SamplerParams { temperature: 0.0, top_k: 0, top_p: 1.0, repeat_penalty: 1.0, repeat_last_n: 64, seed: 1 } }
    pub fn sampler(&self) -> Sampler {
        Sampler::new(self.temperature, self.top_k as usize, self.seed).with_top_p(self.top_p).with_repetition_penalty(self.repeat_penalty, self.repeat_last_n as usize)
    }
}

/// The proof.
#[derive(Clone, Debug, PartialEq)]
pub struct Receipt {
    pub format: u64,
    pub engine_version: String,
    pub model_fingerprint: String,
    pub config_hash: String,
    pub tensors: Vec<TensorRecord>,
    pub tokenizer_fingerprint: String,
    pub sampler: SamplerParams,
    pub prompt_len: u64,
    pub prompt_hash: String,
    pub output_len: u64,
    pub output_hash: String,
    pub reproducible: bool,
    /// ISA level of the run, e.g. `x86_64/avx2+fma`. Not identifying; not part of replay equality.
    pub hardware: String,
    /// Seconds since the Unix epoch.
    pub timestamp: u64,
    /// How many top logits go into each step hash (0 = tokens only).
    pub logit_topk: u64,
    /// Head of the step hash chain.
    pub chain_head: String,
    /// The chain after every step, when kept: lets replay name the first step that differs.
    pub steps: Option<Vec<String>>,
    /// SHA-256 of the canonical body (everything but `digest` and `mac`).
    pub digest: String,
    /// HMAC-SHA256 under the user's key over everything but `mac`.
    pub mac: Option<String>,
}

/// What [`Receipt::verify`] established.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verified {
    /// The MAC was present and checked against the key that was given.
    pub authenticated: bool,
}

// ------------------------------------------------------------------ JSON encode / parse

const KEYS: [&str; 19] = ["chain_head", "config_hash", "digest", "engine_version", "format", "hardware", "logit_topk", "mac", "model_fingerprint",
    "output_hash", "output_len", "prompt_hash", "prompt_len", "reproducible", "sampler", "steps", "tensors", "timestamp", "tokenizer_fingerprint"];

fn f32_json(x: f32) -> Value { serde_json::Number::from_f64(format!("{x}").parse::<f64>().unwrap_or(0.0)).map_or(Value::Null, Value::Number) }

impl Receipt {
    fn value(&self, with_digest: bool, with_mac: bool) -> Value {
        let mut m = Map::new();
        m.insert("format".into(), self.format.into());
        m.insert("engine_version".into(), self.engine_version.clone().into());
        m.insert("model_fingerprint".into(), self.model_fingerprint.clone().into());
        m.insert("config_hash".into(), self.config_hash.clone().into());
        let ts: Vec<Value> = self.tensors.iter().map(|t| {
            let mut o = Map::new();
            o.insert("name".into(), t.name.clone().into());
            o.insert("kind".into(), t.kind.clone().into());
            o.insert("rows".into(), t.rows.into());
            o.insert("cols".into(), t.cols.into());
            o.insert("bytes".into(), t.bytes.into());
            o.insert("root".into(), t.root.clone().into());
            if let Some(l) = &t.leaves { o.insert("leaves".into(), Value::Array(l.iter().cloned().map(Value::from).collect())); }
            Value::Object(o)
        }).collect();
        m.insert("tensors".into(), Value::Array(ts));
        m.insert("tokenizer_fingerprint".into(), self.tokenizer_fingerprint.clone().into());
        let mut s = Map::new();
        s.insert("temperature".into(), f32_json(self.sampler.temperature));
        s.insert("temperature_bits".into(), self.sampler.temperature.to_bits().into());
        s.insert("top_k".into(), self.sampler.top_k.into());
        s.insert("top_p".into(), f32_json(self.sampler.top_p));
        s.insert("top_p_bits".into(), self.sampler.top_p.to_bits().into());
        s.insert("repeat_penalty".into(), f32_json(self.sampler.repeat_penalty));
        s.insert("repeat_penalty_bits".into(), self.sampler.repeat_penalty.to_bits().into());
        s.insert("repeat_last_n".into(), self.sampler.repeat_last_n.into());
        s.insert("seed".into(), self.sampler.seed.into());
        m.insert("sampler".into(), Value::Object(s));
        m.insert("prompt_len".into(), self.prompt_len.into());
        m.insert("prompt_hash".into(), self.prompt_hash.clone().into());
        m.insert("output_len".into(), self.output_len.into());
        m.insert("output_hash".into(), self.output_hash.clone().into());
        m.insert("reproducible".into(), self.reproducible.into());
        m.insert("hardware".into(), self.hardware.clone().into());
        m.insert("timestamp".into(), self.timestamp.into());
        m.insert("logit_topk".into(), self.logit_topk.into());
        m.insert("chain_head".into(), self.chain_head.clone().into());
        if let Some(st) = &self.steps { m.insert("steps".into(), Value::Array(st.iter().cloned().map(Value::from).collect())); }
        if with_digest { m.insert("digest".into(), self.digest.clone().into()); }
        if with_mac && let Some(mac) = &self.mac { m.insert("mac".into(), mac.clone().into()); }
        Value::Object(m)
    }
    fn body_text(&self) -> Result<String, ReceiptError> { canonical_json(&self.value(false, false)) }
    fn mac_text(&self) -> Result<String, ReceiptError> { canonical_json(&self.value(true, false)) }

    /// Compute the digest and, with a key, the MAC. Call after changing any field.
    pub fn seal(&mut self, key: Option<&[u8]>) -> Result<(), ReceiptError> {
        self.digest = hex(&sha256(self.body_text()?.as_bytes()));
        self.mac = match key { Some(k) => Some(hex(&hmac_sha256(k, self.mac_text()?.as_bytes()))), None => None };
        Ok(())
    }

    /// Canonical JSON text of the whole receipt.
    pub fn to_json(&self) -> String { canonical_json(&self.value(true, true)).unwrap_or_default() }

    /// Parse, strictly: unknown fields, wrong types, bad hashes, inconsistent lengths and
    /// oversized inputs are all errors. Does not check the digest; call [`Receipt::verify`].
    pub fn from_json(text: &str) -> Result<Receipt, ReceiptError> {
        if text.len() > MAX_JSON_BYTES { return bad("too large"); }
        let v: Value = serde_json::from_str(text).map_err(|e| ReceiptError::Malformed(e.to_string()))?;
        let m = v.as_object().ok_or_else(|| ReceiptError::Malformed("not an object".into()))?;
        for k in m.keys() { if !KEYS.contains(&k.as_str()) { return bad(format!("unknown field `{k}`")); } }
        let sampler = {
            let s = obj(m, "sampler")?;
            for k in s.keys() { if !["temperature", "temperature_bits", "top_k", "top_p", "top_p_bits", "repeat_penalty", "repeat_penalty_bits", "repeat_last_n", "seed"].contains(&k.as_str()) { return bad(format!("unknown sampler field `{k}`")); } }
            let fl = |name: &str| -> Result<f32, ReceiptError> {
                let bits = u32::try_from(u64_of(s, &format!("{name}_bits"))?).map_err(|_| ReceiptError::Malformed(format!("{name}_bits out of range")))?;
                let x = f32::from_bits(bits);
                let shown = s.get(name).and_then(Value::as_f64).ok_or_else(|| ReceiptError::Malformed(format!("`{name}` missing")))?;
                if !x.is_finite() || format!("{x}").parse::<f64>().ok() != Some(shown) { return bad(format!("`{name}` disagrees with its bits")); }
                Ok(x)
            };
            SamplerParams { temperature: fl("temperature")?, top_k: u64_of(s, "top_k")?, top_p: fl("top_p")?, repeat_penalty: fl("repeat_penalty")?, repeat_last_n: u64_of(s, "repeat_last_n")?, seed: u64_of(s, "seed")? }
        };
        let tarr = m.get("tensors").and_then(Value::as_array).ok_or_else(|| ReceiptError::Malformed("`tensors` missing".into()))?;
        if tarr.len() > MAX_TENSORS { return bad("too many tensors"); }
        let mut tensors = Vec::with_capacity(tarr.len());
        for t in tarr {
            let o = t.as_object().ok_or_else(|| ReceiptError::Malformed("tensor entry is not an object".into()))?;
            for k in o.keys() { if !["name", "kind", "rows", "cols", "bytes", "root", "leaves"].contains(&k.as_str()) { return bad(format!("unknown tensor field `{k}`")); } }
            let bytes = u64_of(o, "bytes")?;
            let leaves = match o.get("leaves") {
                None => None,
                Some(l) => {
                    let a = l.as_array().ok_or_else(|| ReceiptError::Malformed("`leaves` is not an array".into()))?;
                    if a.len() > MAX_LEAVES { return bad("too many leaves"); }
                    if a.len() as u64 != bytes.div_ceil(CHUNK as u64) { return bad("leaf count does not match the tensor size"); }
                    Some(a.iter().map(|x| x.as_str().filter(|s| is_hex(s, 64)).map(str::to_string).ok_or_else(|| ReceiptError::Malformed("bad leaf hash".into()))).collect::<Result<Vec<_>, _>>()?)
                }
            };
            tensors.push(TensorRecord { name: str_of(o, "name")?, kind: str_of(o, "kind")?, rows: u64_of(o, "rows")?, cols: u64_of(o, "cols")?, bytes, root: hex_of(o, "root")?, leaves });
        }
        let output_len = u64_of(m, "output_len")?;
        let steps = match m.get("steps") {
            None => None,
            Some(a) => {
                let a = a.as_array().ok_or_else(|| ReceiptError::Malformed("`steps` is not an array".into()))?;
                if a.len() as u64 != output_len { return bad("`steps` length does not match `output_len`"); }
                Some(a.iter().map(|x| x.as_str().filter(|s| is_hex(s, 64)).map(str::to_string).ok_or_else(|| ReceiptError::Malformed("bad step hash".into()))).collect::<Result<Vec<_>, _>>()?)
            }
        };
        let r = Receipt {
            format: u64_of(m, "format")?,
            engine_version: str_of(m, "engine_version")?,
            model_fingerprint: hex_of(m, "model_fingerprint")?,
            config_hash: hex_of(m, "config_hash")?,
            tensors,
            tokenizer_fingerprint: hex_of(m, "tokenizer_fingerprint")?,
            sampler,
            prompt_len: u64_of(m, "prompt_len")?,
            prompt_hash: hex_of(m, "prompt_hash")?,
            output_len,
            output_hash: hex_of(m, "output_hash")?,
            reproducible: bool_of(m, "reproducible")?,
            hardware: str_of(m, "hardware")?,
            timestamp: u64_of(m, "timestamp")?,
            logit_topk: u64_of(m, "logit_topk")?,
            chain_head: hex_of(m, "chain_head")?,
            steps,
            digest: hex_of(m, "digest")?,
            mac: match m.get("mac") { None => None, Some(x) => Some(x.as_str().filter(|s| is_hex(s, 64)).map(str::to_string).ok_or_else(|| ReceiptError::Malformed("bad mac".into()))?) },
        };
        if r.format != FORMAT { return bad(format!("unsupported format {}", r.format)); }
        if r.logit_topk > 4096 { return bad("`logit_topk` out of range"); }
        if let Some(st) = &r.steps
            && st.last().map(String::as_str).unwrap_or(&chain_start_hex(&r.prompt_hash)) != r.chain_head { return bad("`chain_head` is not the last step"); }
        Ok(r)
    }

    /// Check the digest, and the MAC when a key is given. With a key the receipt must carry a MAC;
    /// without one a present MAC is simply not checked (`authenticated: false`).
    pub fn verify(&self, key: Option<&[u8]>) -> Result<Verified, ReceiptError> {
        let want = hex(&sha256(self.body_text()?.as_bytes()));
        if !is_hex(&self.digest, 64) || !ct_eq(&want, &self.digest) { return Err(ReceiptError::DigestMismatch); }
        // The model fingerprint must follow from the recorded config and tensor table.
        if model_fingerprint_of(&self.config_hash, &self.tensors) != self.model_fingerprint { return Err(ReceiptError::DigestMismatch); }
        match (key, &self.mac) {
            (None, _) => Ok(Verified { authenticated: false }),
            (Some(_), None) => Err(ReceiptError::BadMac),
            (Some(k), Some(mac)) => {
                let want = hex(&hmac_sha256(k, self.mac_text()?.as_bytes()));
                if ct_eq(&want, mac) { Ok(Verified { authenticated: true }) } else { Err(ReceiptError::BadMac) }
            }
        }
    }
}

fn is_hex(s: &str, n: usize) -> bool { s.len() == n && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) }
fn obj<'a>(m: &'a Map<String, Value>, k: &str) -> Result<&'a Map<String, Value>, ReceiptError> {
    m.get(k).and_then(Value::as_object).ok_or_else(|| ReceiptError::Malformed(format!("`{k}` missing or not an object")))
}
fn u64_of(m: &Map<String, Value>, k: &str) -> Result<u64, ReceiptError> {
    m.get(k).and_then(Value::as_u64).ok_or_else(|| ReceiptError::Malformed(format!("`{k}` missing or not an unsigned integer")))
}
fn str_of(m: &Map<String, Value>, k: &str) -> Result<String, ReceiptError> {
    let s = m.get(k).and_then(Value::as_str).ok_or_else(|| ReceiptError::Malformed(format!("`{k}` missing or not a string")))?;
    if s.len() > 4096 { return bad(format!("`{k}` too long")); }
    Ok(s.to_string())
}
fn bool_of(m: &Map<String, Value>, k: &str) -> Result<bool, ReceiptError> {
    m.get(k).and_then(Value::as_bool).ok_or_else(|| ReceiptError::Malformed(format!("`{k}` missing or not a boolean")))
}
fn hex_of(m: &Map<String, Value>, k: &str) -> Result<String, ReceiptError> {
    let s = str_of(m, k)?;
    if is_hex(&s, 64) { Ok(s) } else { bad(format!("`{k}` is not a 64-digit lowercase hex hash")) }
}

// ------------------------------------------------------------------------ fingerprints

fn put_str(s: &mut Sha256, x: &str) { s.update(&(x.len() as u64).to_le_bytes()); s.update(x.as_bytes()); }

fn model_fingerprint_of(config_hash: &str, tensors: &[TensorRecord]) -> String {
    let mut s = Sha256::new();
    put_str(&mut s, "corbul-model/v1");
    put_str(&mut s, config_hash);
    s.update(&(tensors.len() as u64).to_le_bytes());
    for t in tensors {
        put_str(&mut s, &t.name);
        put_str(&mut s, &t.kind);
        s.update(&t.rows.to_le_bytes());
        s.update(&t.cols.to_le_bytes());
        s.update(&t.bytes.to_le_bytes());
        put_str(&mut s, &t.root);
    }
    hex(&s.finish())
}

/// Hash of the model configuration (`Spec`'s debug form: ties a receipt to the engine's own idea
/// of the configuration, so it can change between engine versions that add fields).
pub fn config_hash(model: &Model) -> String { hex(&sha256(format!("corbul-spec/v1\n{:?}", model.spec).as_bytes())) }

enum Data<'a> { F32(&'a [f32]), Raw(&'a [u8]) }
struct Src<'a> { name: String, kind: String, rows: usize, cols: usize, data: Data<'a> }
impl Src<'_> {
    fn nbytes(&self) -> usize { match &self.data { Data::F32(v) => v.len() * 4, Data::Raw(b) => b.len() } }
    fn leaves(&self) -> Vec<[u8; 32]> {
        match &self.data {
            Data::Raw(b) => merkle_leaves(b),
            Data::F32(v) => {
                let mut buf = Vec::with_capacity(CHUNK);
                v.chunks(CHUNK / 4).map(|c| { buf.clear(); for x in c { buf.extend_from_slice(&x.to_le_bytes()); } leaf_hash(&buf) }).collect()
            }
        }
    }
}

fn sources(model: &Model) -> Result<Vec<Src<'_>>, ReceiptError> {
    let w = &model.w;
    if w.experts.is_some() || w.layers.iter().any(|l| l.moe.is_some()) {
        return Err(ReceiptError::Unsupported("mixture-of-experts weights are not fingerprinted yet".into()));
    }
    let mut v = Vec::new();
    fn weight<'a>(v: &mut Vec<Src<'a>>, name: String, x: &'a Weight) {
        match x {
            Weight::Dense(d) => if !d.is_empty() { v.push(Src { name, kind: "f32".into(), rows: 1, cols: d.len(), data: Data::F32(d) }) },
            Weight::Quant(q) => v.push(Src { name, kind: format!("{:?}", q.dtype), rows: q.rows, cols: q.cols, data: Data::Raw(q.bytes.as_slice()) }),
        }
    }
    fn vecf<'a>(v: &mut Vec<Src<'a>>, name: String, x: &'a [f32]) {
        if !x.is_empty() { v.push(Src { name, kind: "f32".into(), rows: 1, cols: x.len(), data: Data::F32(x) }) }
    }
    weight(&mut v, "embed".into(), &w.embed);
    for (i, l) in w.layers.iter().enumerate() {
        let n = |s: &str| format!("l{i}.{s}");
        vecf(&mut v, n("attn_norm"), &l.attn_norm);
        if let Some(x) = &l.attn_norm_b { vecf(&mut v, n("attn_norm_b"), x); }
        for (s, x) in [("wq", &l.wq), ("wk", &l.wk), ("wv", &l.wv), ("wo", &l.wo)] { weight(&mut v, n(s), x); }
        for (s, x) in [("bq", &l.bq), ("bk", &l.bk), ("bv", &l.bv), ("bo", &l.bo), ("q_norm", &l.q_norm), ("k_norm", &l.k_norm), ("post_attn_norm", &l.post_attn_norm)] {
            if let Some(x) = x { vecf(&mut v, n(s), x); }
        }
        vecf(&mut v, n("ffn_norm"), &l.ffn_norm);
        if let Some(x) = &l.ffn_norm_b { vecf(&mut v, n("ffn_norm_b"), x); }
        weight(&mut v, n("w_gate"), &l.w_gate);
        if let Some(x) = &l.b_gate { vecf(&mut v, n("b_gate"), x); }
        weight(&mut v, n("w_up"), &l.w_up);
        if let Some(x) = &l.b_up { vecf(&mut v, n("b_up"), x); }
        weight(&mut v, n("w_down"), &l.w_down);
        if let Some(x) = &l.b_down { vecf(&mut v, n("b_down"), x); }
        if let Some(x) = &l.post_ffn_norm { vecf(&mut v, n("post_ffn_norm"), x); }
    }
    vecf(&mut v, "final_norm".into(), &w.final_norm);
    if let Some(x) = &w.final_norm_b { vecf(&mut v, "final_norm_b".into(), x); }
    if let Some(x) = &w.lm_head { weight(&mut v, "lm_head".into(), x); }
    if let Some(x) = &w.lm_head_b { vecf(&mut v, "lm_head_b".into(), x); }
    Ok(v)
}

/// Fingerprint of the model's weights: the tensor table with a Merkle root per tensor (and, with
/// `with_leaves`, every 1 MiB leaf hash so that a damaged chunk can be named).
pub fn fingerprint_weights(model: &Model, with_leaves: bool) -> Result<Vec<TensorRecord>, ReceiptError> {
    Ok(sources(model)?.iter().map(|s| {
        let leaves = s.leaves();
        TensorRecord {
            name: s.name.clone(), kind: s.kind.clone(), rows: s.rows as u64, cols: s.cols as u64, bytes: s.nbytes() as u64,
            root: hex(&merkle_root(&leaves)),
            leaves: with_leaves.then(|| leaves.iter().map(|l| hex(l)).collect()),
        }
    }).collect())
}

/// Fingerprint of the tokenizer: vocabulary size and every token string with its special flag.
pub fn fingerprint_tokenizer(tok: &crate::tokenizer::Tokenizer) -> String {
    let mut s = Sha256::new();
    put_str(&mut s, "corbul-tokenizer/v1");
    let n = tok.vocab_size();
    s.update(&(n as u64).to_le_bytes());
    for id in 0..n as u32 {
        put_str(&mut s, tok.id_to_token(id).unwrap_or(""));
        s.update(&[tok.is_special(id) as u8]);
    }
    hex(&s.finish())
}

/// Check the model's weights against the receipt. Names the tensor — and the 1 MiB chunk when the
/// receipt carries leaf hashes — that differs first.
pub fn verify_weights(model: &Model, r: &Receipt) -> Result<(), ReceiptError> {
    if config_hash(model) != r.config_hash { return Err(ReceiptError::ModelMismatch("configuration differs".into())); }
    let src = sources(model)?;
    if src.len() != r.tensors.len() { return Err(ReceiptError::ModelMismatch(format!("{} tensors, the receipt lists {}", src.len(), r.tensors.len()))); }
    for (s, t) in src.iter().zip(&r.tensors) {
        if s.name != t.name || s.kind != t.kind || s.rows as u64 != t.rows || s.cols as u64 != t.cols || s.nbytes() as u64 != t.bytes {
            return Err(ReceiptError::ModelMismatch(format!("tensor `{}` has a different name, type or shape", t.name)));
        }
        let leaves = s.leaves();
        if hex(&merkle_root(&leaves)) != t.root {
            let chunk = t.leaves.as_ref().and_then(|want| leaves.iter().zip(want).position(|(a, b)| hex(a) != *b));
            return Err(ReceiptError::TensorMismatch { tensor: t.name.clone(), chunk });
        }
    }
    Ok(())
}

// ------------------------------------------------------------------- tokens and chain

fn hash_tokens(tokens: &[usize]) -> String {
    let mut s = Sha256::new();
    put_str(&mut s, "corbul-tokens/v1");
    s.update(&(tokens.len() as u64).to_le_bytes());
    for t in tokens { s.update(&(*t as u64).to_le_bytes()); }
    hex(&s.finish())
}
fn chain_start(prompt_hash: &str) -> [u8; 32] {
    let mut s = Sha256::new();
    put_str(&mut s, "corbul-chain/v1");
    put_str(&mut s, prompt_hash);
    s.finish()
}
fn chain_start_hex(prompt_hash: &str) -> String { hex(&chain_start(prompt_hash)) }

/// The `k` largest logits (value, then lowest index first), hashed by index and IEEE bits.
fn topk_hash(logits: &[f32], k: usize) -> [u8; 32] {
    let mut best: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
    for (i, &l) in logits.iter().enumerate() {
        if best.len() == k && best.last().is_some_and(|b| l.total_cmp(&b.1).is_le()) { continue; }
        let at = best.iter().position(|b| l.total_cmp(&b.1).is_gt()).unwrap_or(best.len());
        best.insert(at, (i, l));
        best.truncate(k);
    }
    let mut s = Sha256::new();
    for (i, l) in best { s.update(&(i as u64).to_le_bytes()); s.update(&l.to_bits().to_le_bytes()); }
    s.finish()
}
fn step(prev: &[u8; 32], token: usize, logits: &[f32], topk: usize) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(prev);
    s.update(&(token as u64).to_le_bytes());
    if topk > 0 { s.update(&topk_hash(logits, topk)); }
    s.finish()
}

// ------------------------------------------------------------- recording and replay

/// How to generate.
#[derive(Clone, Debug)]
pub struct GenerateSpec { pub sampler: SamplerParams, pub max_tokens: usize, pub stop_token: Option<usize> }

/// What to record.
#[derive(Clone, Debug, Default)]
pub struct RecordOptions {
    /// Top logits hashed into each step (0 = tokens only).
    pub logit_topk: usize,
    /// Keep the whole chain so a replay can name the first differing step.
    pub keep_steps: bool,
    /// Keep every Merkle leaf so a damaged chunk can be named.
    pub merkle_leaves: bool,
    pub tokenizer_fingerprint: Option<String>,
    /// Override the clock (tests, reproducible archives).
    pub timestamp: Option<u64>,
    /// HMAC key.
    pub key: Option<Vec<u8>>,
}

struct Run { tokens: Vec<usize>, steps: Vec<[u8; 32]>, head: [u8; 32] }

fn run(model: &Model, prompt: &[usize], g: &GenerateSpec, topk: usize) -> Result<Run, ReceiptError> {
    let mut st = model.new_state();
    let mut sampler = g.sampler.sampler();
    let mut logits = model.forward_batch(&mut st, prompt)?;
    sampler.observe_all(prompt);
    let mut head = chain_start(&hash_tokens(prompt));
    let (mut tokens, mut steps) = (Vec::new(), Vec::new());
    while tokens.len() < g.max_tokens {
        let tok = sampler.sample(&logits);
        sampler.observe(tok);
        head = step(&head, tok, &logits, topk);
        tokens.push(tok);
        steps.push(head);
        if g.stop_token == Some(tok) || tokens.len() >= g.max_tokens || st.pos >= model.spec.context as usize { break; }
        logits = model.forward(&mut st, tok)?;
    }
    Ok(Run { tokens, steps, head })
}

/// Generate and prove it. Turn reproducible mode on first (`simd::set_reproducible(true)`) if the
/// receipt should be replayable on other machines.
pub fn generate_with_receipt(model: &Model, prompt: &[usize], g: &GenerateSpec, o: &RecordOptions) -> Result<(Vec<usize>, Receipt), ReceiptError> {
    if g.max_tokens == 0 { return Err(ReceiptError::Model("max_tokens must be at least 1".into())); }
    let tensors = fingerprint_weights(model, o.merkle_leaves)?;
    let cfg = config_hash(model);
    let r = run(model, prompt, g, o.logit_topk)?;
    let prompt_hash = hash_tokens(prompt);
    let mut rc = Receipt {
        format: FORMAT,
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        model_fingerprint: model_fingerprint_of(&cfg, &tensors),
        config_hash: cfg,
        tensors,
        tokenizer_fingerprint: o.tokenizer_fingerprint.clone().unwrap_or_else(|| hex(&sha256(b"none"))),
        sampler: g.sampler,
        prompt_len: prompt.len() as u64,
        prompt_hash,
        output_len: r.tokens.len() as u64,
        output_hash: hash_tokens(&r.tokens),
        reproducible: simd::is_reproducible(),
        hardware: format!("{}/{}", std::env::consts::ARCH, simd::active()),
        timestamp: o.timestamp.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())),
        logit_topk: o.logit_topk as u64,
        chain_head: hex(&r.head),
        steps: o.keep_steps.then(|| r.steps.iter().map(|h| hex(h)).collect()),
        digest: String::new(),
        mac: None,
    };
    rc.seal(o.key.as_deref())?;
    Ok((r.tokens, rc))
}

/// Result of [`replay`].
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayReport {
    /// Output tokens and the hash chain are identical to the receipt's.
    pub identical: bool,
    /// First output position whose chain hash differs (needs `steps` in the receipt).
    pub first_divergence: Option<usize>,
    pub tokens: Vec<usize>,
}

/// Re-run the generation of `r` and compare. Checks the receipt itself, the prompt, the weights
/// (naming a damaged tensor/chunk) and that reproducible mode is on when the receipt needs it.
/// A mismatch of the *output* is not an error: it is `identical == false` in the report.
pub fn replay(r: &Receipt, model: &Model, prompt: &[usize], key: Option<&[u8]>) -> Result<ReplayReport, ReceiptError> {
    r.verify(key)?;
    if r.prompt_len != prompt.len() as u64 || r.prompt_hash != hash_tokens(prompt) { return Err(ReceiptError::PromptMismatch); }
    if r.reproducible && !simd::is_reproducible() { return Err(ReceiptError::ReproducibleModeOff); }
    verify_weights(model, r)?;
    let g = GenerateSpec { sampler: r.sampler, max_tokens: usize::try_from(r.output_len).map_err(|_| ReceiptError::Malformed("`output_len` too large".into()))?, stop_token: None };
    if g.max_tokens == 0 { return Err(ReceiptError::Malformed("`output_len` is zero".into())); }
    let run = run(model, prompt, &g, r.logit_topk as usize)?;
    let first_divergence = r.steps.as_ref().and_then(|want| run.steps.iter().zip(want).position(|(a, b)| hex(a) != *b));
    let identical = first_divergence.is_none() && hex(&run.head) == r.chain_head && hash_tokens(&run.tokens) == r.output_hash;
    Ok(ReplayReport { identical, first_divergence, tokens: run.tokens })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::{base_spec, model_of, random_weights, tiny};

    fn unhex(s: &str) -> Vec<u8> { (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect() }

    #[test]
    fn sha256_known_answers() {
        assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn hmac_rfc4231_vectors() {
        // (key, data, expected) — test cases 1, 2, 3, 4, 6 and 7 of RFC 4231.
        let cases: [(Vec<u8>, Vec<u8>, &str); 6] = [
            (vec![0x0b; 20], b"Hi There".to_vec(), "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"),
            (b"Jefe".to_vec(), b"what do ya want for nothing?".to_vec(), "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"),
            (vec![0xaa; 20], vec![0xdd; 50], "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"),
            (unhex("0102030405060708090a0b0c0d0e0f10111213141516171819"), vec![0xcd; 50], "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b"),
            (vec![0xaa; 131], b"Test Using Larger Than Block-Size Key - Hash Key First".to_vec(), "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"),
            (vec![0xaa; 131], b"This is a test using a larger than block-size key and a larger than block-size data. The key needs to be hashed before being used by the HMAC algorithm.".to_vec(), "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"),
        ];
        for (i, (k, d, want)) in cases.iter().enumerate() { assert_eq!(hex(&hmac_sha256(k, d)), *want, "case {i}"); }
    }

    #[test]
    fn canonical_json_is_sorted_compact_and_stable() {
        let v: Value = serde_json::from_str(r#"{"b": [1, 2.5, {"z": true, "a": null}], "a": "x\ny", "c": 1.0, "d": -3}"#).unwrap();
        assert_eq!(canonical_json(&v).unwrap(), r#"{"a":"x\ny","b":[1,2.5,{"a":null,"z":true}],"c":1,"d":-3}"#);
        // Same value, different input order and spacing.
        let w: Value = serde_json::from_str(r#"{"d":-3,"c":1,"a":"x\ny","b":[1,2.5,{"a":null,"z":true}]}"#).unwrap();
        assert_eq!(canonical_json(&v).unwrap(), canonical_json(&w).unwrap());
    }

    fn sample_receipt(key: Option<&[u8]>, leaves: bool) -> (Model, Vec<usize>, GenerateSpec, Receipt, Vec<usize>) {
        let m = tiny(4, 2, false);
        let prompt = vec![3usize, 7, 1, 19];
        let g = GenerateSpec { sampler: SamplerParams { temperature: 0.8, top_k: 8, top_p: 0.9, repeat_penalty: 1.1, repeat_last_n: 16, seed: 42 }, max_tokens: 6, stop_token: None };
        let o = RecordOptions { logit_topk: 4, keep_steps: true, merkle_leaves: leaves, key: key.map(|k| k.to_vec()), timestamp: Some(1_700_000_000), ..Default::default() };
        let (toks, r) = generate_with_receipt(&m, &prompt, &g, &o).unwrap();
        (m, prompt, g, r, toks)
    }

    #[test]
    fn round_trip_and_verify() {
        let (_, _, _, r, toks) = sample_receipt(Some(b"k3y"), true);
        assert_eq!(toks.len(), 6);
        let text = r.to_json();
        let back = Receipt::from_json(&text).unwrap();
        assert_eq!(back, r);
        assert_eq!(back.to_json(), text, "canonical text is a fixed point");
        assert_eq!(back.verify(Some(b"k3y")).unwrap(), Verified { authenticated: true });
        assert_eq!(back.verify(None).unwrap(), Verified { authenticated: false });
        assert_eq!(back.verify(Some(b"wrong")), Err(ReceiptError::BadMac));
        let (_, _, _, nokey, _) = sample_receipt(None, false);
        assert_eq!(nokey.verify(Some(b"k3y")), Err(ReceiptError::BadMac), "a key is given but the receipt has no MAC");
        assert_eq!(nokey.verify(None).unwrap(), Verified { authenticated: false });
    }

    /// Change every leaf value of the JSON in turn; each altered text must be rejected.
    #[test]
    fn any_field_tamper_is_detected() {
        let (_, _, _, r, _) = sample_receipt(Some(b"k3y"), false);
        let v: Value = serde_json::from_str(&r.to_json()).unwrap();
        fn paths(v: &Value, cur: &mut Vec<String>, out: &mut Vec<Vec<String>>) {
            match v {
                Value::Object(m) => for (k, x) in m { cur.push(k.clone()); paths(x, cur, out); cur.pop(); },
                Value::Array(a) => for (i, x) in a.iter().enumerate() { cur.push(i.to_string()); paths(x, cur, out); cur.pop(); },
                _ => out.push(cur.clone()),
            }
        }
        let mut all = vec![];
        paths(&v, &mut vec![], &mut all);
        assert!(all.len() > 40, "{}", all.len());
        for p in &all {
            let mut w = v.clone();
            let mut at = &mut w;
            for k in p { at = match at { Value::Object(m) => m.get_mut(k).unwrap(), Value::Array(a) => &mut a[k.parse::<usize>().unwrap()], _ => unreachable!() }; }
            *at = match at.clone() {
                Value::String(s) => Value::String(if s.is_empty() { "x".into() } else { let mut c: Vec<char> = s.chars().collect(); c[0] = if c[0] == '0' { '1' } else { '0' }; c.into_iter().collect() }),
                Value::Number(n) => Value::from(n.as_u64().map_or(1, |x| x ^ 1)),
                Value::Bool(b) => Value::Bool(!b),
                other => other,
            };
            if w == v { continue; }
            let text = serde_json::to_string(&w).unwrap();
            let res = Receipt::from_json(&text).and_then(|r| r.verify(Some(b"k3y")));
            assert!(res.is_err(), "tampering with {p:?} went unnoticed");
        }
        // Deleting a field, adding one, truncating: all rejected.
        let mut w = v.clone();
        w.as_object_mut().unwrap().remove("output_hash");
        assert!(Receipt::from_json(&w.to_string()).is_err());
        w = v.clone();
        w.as_object_mut().unwrap().insert("extra".into(), 1.into());
        assert!(Receipt::from_json(&w.to_string()).is_err());
        let t = r.to_json();
        assert!(Receipt::from_json(&t[..t.len() - 1]).is_err());
        // Re-sealing without the key gives a valid digest but cannot forge the MAC.
        let mut forged = Receipt::from_json(&t).unwrap();
        forged.output_hash = hex(&sha256(b"other"));
        forged.seal(None).unwrap();
        assert_eq!(forged.verify(None).unwrap(), Verified { authenticated: false });
        assert_eq!(forged.verify(Some(b"k3y")), Err(ReceiptError::BadMac));
    }

    #[test]
    fn hostile_json_is_an_error_not_a_panic() {
        let (_, _, _, r, _) = sample_receipt(None, true);
        let good = r.to_json();
        let hostile = [
            "", "null", "[]", "{}", "\"x\"", "{\"format\":1}", "{{{{", "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[",
            &"[".repeat(100_000), &"{\"a\":".repeat(5_000),
            &good.replace("\"format\":1", "\"format\":2"),
            &good.replace("\"format\":1", "\"format\":-1"),
            &good.replace("\"prompt_len\":4", "\"prompt_len\":1e99"),
            &good.replace("\"prompt_len\":4", "\"prompt_len\":\"4\""),
            &good.replace("\"seed\":42", "\"seed\":18446744073709551616"),
            &good.replace("\"reproducible\":false", "\"reproducible\":0"),
            &good.replace(&r.digest, "ABCDEF"),
            &good.replace(&r.digest, &"G".repeat(64)),
            &good.replace(&r.digest, &r.digest.to_uppercase()),
            &good.replace("\"output_len\":6", "\"output_len\":7"),
            &good.replace("\"output_len\":6", "\"output_len\":99999999999999"),
            &good.replace("\"logit_topk\":4", "\"logit_topk\":99999999"),
            &good.replace("\"temperature_bits\":", "\"temperature_bits\":4294967296,\"x\":"),
            &good.replace("\"temperature_bits\":1061997773", "\"temperature_bits\":2143289344"),
            "{\"tensors\":[[]]}", "\u{0}", "\u{feff}{}",
        ];
        for h in hostile {
            let res = Receipt::from_json(h).and_then(|r| r.verify(None));
            assert!(res.is_err(), "accepted: {}", &h[..h.len().min(80)]);
        }
        assert!(Receipt::from_json(&good).is_ok());
        assert!(Receipt::from_json(&"x".repeat(MAX_JSON_BYTES + 1)).is_err());
    }

    #[test]
    fn replay_confirms_the_output_and_names_the_first_difference() {
        let (m, prompt, _g, r, toks) = sample_receipt(Some(b"k"), true);
        let rep = replay(&r, &m, &prompt, Some(b"k")).unwrap();
        assert!(rep.identical, "{rep:?}");
        assert_eq!(rep.tokens, toks);
        assert_eq!(rep.first_divergence, None);
        // Another prompt is refused outright.
        assert_eq!(replay(&r, &m, &[3, 7, 1, 20], Some(b"k")), Err(ReceiptError::PromptMismatch));
        // A forged-but-resealed step chain makes the replay differ, at the step that was changed.
        let mut f = r.clone();
        let mut steps = f.steps.clone().unwrap();
        steps[3] = hex(&sha256(b"forged"));
        f.steps = Some(steps);
        f.seal(Some(b"k")).unwrap();
        let rep = replay(&f, &m, &prompt, Some(b"k")).unwrap();
        assert!(!rep.identical);
        assert_eq!(rep.first_divergence, Some(3), "the altered step is named");
        let mut f = r.clone();
        f.output_hash = hex(&sha256(b"other output"));
        f.seal(Some(b"k")).unwrap();
        let rep = replay(&f, &m, &prompt, Some(b"k")).unwrap();
        assert!(!rep.identical);
        // A different sampler seed in the receipt gives different tokens and a first divergence.
        let mut f = r.clone();
        f.sampler.seed = 43;
        f.seal(Some(b"k")).unwrap();
        let rep = replay(&f, &m, &prompt, Some(b"k")).unwrap();
        if !rep.identical { assert!(rep.first_divergence.is_some()); }
    }

    #[test]
    fn a_different_model_is_refused() {
        let (_, prompt, _, r, _) = sample_receipt(None, false);
        let other = tiny(4, 4, false);
        assert!(matches!(replay(&r, &other, &prompt, None), Err(ReceiptError::ModelMismatch(_))));
        let mut spec = base_spec(4, 2, false);
        spec.ffn = 32;
        let other = model_of(spec, false);
        assert!(replay(&r, &other, &prompt, None).is_err());
    }

    #[test]
    fn merkle_localises_a_flipped_bit_to_its_chunk() {
        // 20_000 × 64 f32 = 5 MiB of embeddings, i.e. five chunks.
        let spec = crate::arch::Spec::dense("llama", 1, 64, 4, 4, 16, 64, 20_000, 32);
        let w = random_weights(&spec, 7, false);
        let m = crate::model::Model::new(spec.clone(), w.clone()).unwrap();
        let prompt = [1usize, 2, 3];
        let g = GenerateSpec { sampler: SamplerParams::greedy(), max_tokens: 2, stop_token: None };
        let (_, r) = generate_with_receipt(&m, &prompt, &g, &RecordOptions { merkle_leaves: true, ..Default::default() }).unwrap();
        let emb = r.tensors.iter().find(|t| t.name == "embed").unwrap();
        assert_eq!(emb.leaves.as_ref().unwrap().len(), 5);
        verify_weights(&m, &r).unwrap();
        // Flip one bit in element 3·262144 + 5 → chunk 3.
        let mut w2 = w.clone();
        let Weight::Dense(e) = &mut w2.embed else { panic!() };
        e[3 * (CHUNK / 4) + 5] = f32::from_bits(e[3 * (CHUNK / 4) + 5].to_bits() ^ 1);
        let m2 = crate::model::Model::new(spec.clone(), w2.clone()).unwrap();
        assert_eq!(verify_weights(&m2, &r), Err(ReceiptError::TensorMismatch { tensor: "embed".into(), chunk: Some(3) }));
        // The last bit of the last chunk.
        let mut w3 = w.clone();
        let Weight::Dense(e) = &mut w3.embed else { panic!() };
        let last = e.len() - 1;
        e[last] = f32::from_bits(e[last].to_bits() ^ 0x8000_0000);
        let m3 = crate::model::Model::new(spec.clone(), w3).unwrap();
        assert_eq!(verify_weights(&m3, &r), Err(ReceiptError::TensorMismatch { tensor: "embed".into(), chunk: Some(4) }));
        // A damaged norm weight in another tensor is named too.
        let mut w4 = w.clone();
        w4.layers[0].ffn_norm[0] += 1.0;
        let m4 = crate::model::Model::new(spec.clone(), w4).unwrap();
        assert_eq!(verify_weights(&m4, &r), Err(ReceiptError::TensorMismatch { tensor: "l0.ffn_norm".into(), chunk: Some(0) }));
        // Without leaves only the tensor is named.
        let (_, r2) = generate_with_receipt(&m, &prompt, &g, &RecordOptions::default()).unwrap();
        assert_eq!(verify_weights(&m2, &r2), Err(ReceiptError::TensorMismatch { tensor: "embed".into(), chunk: None }));
        // And the replay refuses to run on damaged weights.
        assert!(matches!(replay(&r, &m2, &prompt, None), Err(ReceiptError::TensorMismatch { .. })));
    }

    #[test]
    fn merkle_root_basics() {
        assert_eq!(merkle_leaves(&[]).len(), 0);
        let data: Vec<u8> = (0..(CHUNK * 2 + 17)).map(|i| (i * 31 % 251) as u8).collect();
        let leaves = merkle_leaves(&data);
        assert_eq!(leaves.len(), 3);
        let root = merkle_root(&leaves);
        let want = node_hash(&node_hash(&leaves[0], &leaves[1]), &leaves[2]);
        assert_eq!(root, want);
        let mut d2 = data.clone();
        d2[CHUNK + 1] ^= 4;
        assert_ne!(merkle_root(&merkle_leaves(&d2)), root);
        // Swapping two equal-length chunks changes the root (position matters).
        let mut d3 = data.clone();
        let (a, b) = d3.split_at_mut(CHUNK);
        a.swap_with_slice(&mut b[..CHUNK]);
        assert_ne!(merkle_root(&merkle_leaves(&d3)), root);
    }

    #[test]
    fn topk_hash_is_order_and_value_sensitive() {
        let l = [0.5f32, 2.0, -1.0, 2.0, 0.1];
        // Ties go to the lower index; values outside the top 2 do not matter.
        assert_eq!(topk_hash(&l, 2), topk_hash(&[0.0, 2.0, 0.0, 2.0, 0.0], 2));
        assert_ne!(topk_hash(&l, 2), topk_hash(&[0.0, 2.0, 0.0, 2.5, 0.0], 2));
        assert_ne!(topk_hash(&l, 3), topk_hash(&[0.6, 2.0, -1.0, 2.0, 0.2], 3));
        assert_eq!(topk_hash(&l, 8), topk_hash(&l, 8));
    }
}
