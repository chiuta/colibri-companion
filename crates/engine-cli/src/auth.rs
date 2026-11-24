//! API keys with per-key model allow-lists and token-bucket rate limits.
//!
//! Keys file (JSON):
//! ```json
//! {"keys": [
//!   {"key": "sk-alice", "name": "alice", "models": ["qwen*", "llama3"], "rpm": 60, "tpm": 100000},
//!   {"key": "sk-bob"}
//! ]}
//! ```
//! A bare list of such objects, or of key strings, works too. `models` entries may end in `*`.
//! `rpm` / `tpm` are requests / prompt tokens per minute (absent = unlimited); bursts up to one minute's worth.
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub struct KeyEntry {
    pub key: String,
    pub name: String,
    /// Empty = every model.
    pub models: Vec<String>,
    pub rpm: Option<f64>,
    pub tpm: Option<f64>,
}

impl KeyEntry {
    pub fn allows(&self, model: &str) -> bool {
        self.models.is_empty() || self.models.iter().any(|p| match p.strip_suffix('*') { Some(pre) => model.starts_with(pre), None => p == model })
    }
}

#[derive(Clone, Debug, Default)]
pub struct KeyStore { pub keys: Vec<KeyEntry> }

fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    // compare over the longer length so timing does not reveal where the first difference is
    let n = a.len().max(b.len());
    let mut acc = (a.len() ^ b.len()) as u8 | ((a.len() ^ b.len()) >> 8) as u8;
    for i in 0..n { acc |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0); }
    acc == 0
}

impl KeyStore {
    pub fn parse(text: &str) -> Result<KeyStore, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("keys file is not valid JSON: {e}"))?;
        let list = match &v { Value::Array(a) => a, Value::Object(o) => match o.get("keys") { Some(Value::Array(a)) => a, _ => return Err("keys file needs a `keys` list".into()) }, _ => return Err("keys file must be a list or an object with `keys`".into()) };
        let mut keys = vec![];
        for (i, e) in list.iter().enumerate() {
            let k = match e {
                Value::String(s) => KeyEntry { key: s.clone(), name: format!("key{}", i + 1), models: vec![], rpm: None, tpm: None },
                Value::Object(o) => {
                    let key = o.get("key").and_then(Value::as_str).ok_or_else(|| format!("key #{} has no `key`", i + 1))?.to_string();
                    let models = match o.get("models") {
                        None | Some(Value::Null) => vec![],
                        Some(Value::Array(m)) => m.iter().map(|x| x.as_str().map(str::to_string).ok_or_else(|| format!("key #{}: `models` must be strings", i + 1))).collect::<Result<_, _>>()?,
                        Some(_) => return Err(format!("key #{}: `models` must be a list", i + 1)),
                    };
                    let lim = |n: &str| -> Result<Option<f64>, String> {
                        match o.get(n) { None | Some(Value::Null) => Ok(None), Some(x) => x.as_f64().filter(|r| *r > 0.0 && r.is_finite()).map(Some).ok_or_else(|| format!("key #{}: `{n}` must be a positive number", i + 1)) }
                    };
                    KeyEntry { key, name: o.get("name").and_then(Value::as_str).map_or_else(|| format!("key{}", i + 1), str::to_string), models, rpm: lim("rpm")?, tpm: lim("tpm")? }
                }
                _ => return Err(format!("key #{} must be a string or an object", i + 1)),
            };
            if k.key.len() < 8 { return Err(format!("key #{} is shorter than 8 characters", i + 1)); }
            if keys.iter().any(|o: &KeyEntry| o.key == k.key) { return Err(format!("key #{} is a duplicate", i + 1)); }
            keys.push(k);
        }
        if keys.is_empty() { return Err("keys file holds no keys".into()); }
        Ok(KeyStore { keys })
    }

    pub fn load(path: &std::path::Path) -> Result<KeyStore, String> {
        KeyStore::parse(&std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?)
    }

    /// Looks at every entry (no early exit) so the time does not depend on which key matched.
    pub fn authenticate(&self, token: &str) -> Option<&KeyEntry> {
        let mut found = None;
        for k in &self.keys { if ct_eq(&k.key, token) && found.is_none() { found = Some(k); } }
        found
    }
}

// ------------------------------------------------------------------------------ rate limiting

pub trait Clock: Send + Sync { fn now(&self) -> f64; }

pub struct SystemClock(Instant);
impl Default for SystemClock { fn default() -> Self { SystemClock(Instant::now()) } }
impl Clock for SystemClock { fn now(&self) -> f64 { self.0.elapsed().as_secs_f64() } }

#[derive(Clone, Copy, Debug)]
struct Bucket { tokens: f64, at: f64 }

/// Token buckets keyed by (client, limit name). Capacity = one minute of the rate.
pub struct RateLimiter<C: Clock = SystemClock> { clock: C, buckets: Mutex<HashMap<(String, &'static str), Bucket>> }

impl<C: Clock + Default> Default for RateLimiter<C> { fn default() -> Self { RateLimiter::new(C::default()) } }

impl<C: Clock> RateLimiter<C> {
    pub fn new(clock: C) -> RateLimiter<C> { RateLimiter { clock, buckets: Mutex::new(HashMap::new()) } }

    /// Take `cost` from `key`'s bucket for `limit` (`per_minute` units refill per minute).
    /// `Err(seconds)` = how long until it would succeed (`f64::INFINITY` when `cost` exceeds the capacity).
    pub fn take(&self, key: &str, limit: &'static str, per_minute: f64, cost: f64) -> Result<(), f64> {
        let cap = per_minute;
        if cost > cap { return Err(f64::INFINITY); }
        let rate = per_minute / 60.0;
        let now = self.clock.now();
        let mut m = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        if m.len() > 100_000 { m.clear(); }
        let b = m.entry((key.to_string(), limit)).or_insert(Bucket { tokens: cap, at: now });
        b.tokens = (b.tokens + (now - b.at).max(0.0) * rate).min(cap);
        b.at = now;
        if b.tokens >= cost { b.tokens -= cost; Ok(()) } else { Err((cost - b.tokens) / rate) }
    }

    /// Both of a key's limits for one request; nothing is consumed from `rpm` when `tpm` refuses.
    pub fn admit(&self, k: &KeyEntry, prompt_tokens: usize) -> Result<(), f64> {
        if let Some(t) = k.tpm { self.take(&k.key, "tpm", t, prompt_tokens as f64)?; }
        if let Some(r) = k.rpm { self.take(&k.key, "rpm", r, 1.0)?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Default)]
    struct Fake(AtomicU64);
    impl Fake { fn set(&self, s: f64) { self.0.store(s.to_bits(), Ordering::SeqCst); } }
    impl Clock for &'static Fake { fn now(&self) -> f64 { f64::from_bits(self.0.load(Ordering::SeqCst)) } }
    fn fake() -> &'static Fake { Box::leak(Box::new(Fake::default())) }

    #[test]
    fn bucket_refills_with_the_clock() {
        let c = fake();
        let rl = RateLimiter::new(c);
        for _ in 0..60 { rl.take("a", "rpm", 60.0, 1.0).unwrap(); }
        let wait = rl.take("a", "rpm", 60.0, 1.0).unwrap_err();
        assert!((wait - 1.0).abs() < 1e-9, "{wait}");
        c.set(0.5);
        assert!(rl.take("a", "rpm", 60.0, 1.0).is_err());
        c.set(1.0);
        rl.take("a", "rpm", 60.0, 1.0).unwrap();
        assert!(rl.take("a", "rpm", 60.0, 1.0).is_err());
        // long idle: refills only to the capacity
        c.set(10_000.0);
        for _ in 0..60 { rl.take("a", "rpm", 60.0, 1.0).unwrap(); }
        assert!(rl.take("a", "rpm", 60.0, 1.0).is_err());
    }

    #[test]
    fn keys_are_independent_and_costs_can_exceed() {
        let c = fake();
        let rl = RateLimiter::new(c);
        rl.take("a", "tpm", 100.0, 100.0).unwrap();
        assert!(rl.take("a", "tpm", 100.0, 1.0).is_err());
        rl.take("b", "tpm", 100.0, 100.0).unwrap();
        assert_eq!(rl.take("b", "tpm", 100.0, 101.0), Err(f64::INFINITY));
        // a clock that goes backwards never mints tokens
        c.set(5.0);
        rl.take("c", "rpm", 60.0, 60.0).unwrap();
        c.set(1.0);
        assert!(rl.take("c", "rpm", 60.0, 1.0).is_err());
    }

    #[test]
    fn admit_charges_both_or_neither_rpm() {
        let c = fake();
        let rl = RateLimiter::new(c);
        let k = KeyEntry { key: "sk-12345678".into(), name: "n".into(), models: vec![], rpm: Some(2.0), tpm: Some(50.0) };
        rl.admit(&k, 30).unwrap();
        assert!(rl.admit(&k, 30).is_err()); // tpm refuses; rpm not charged
        rl.admit(&k, 10).unwrap(); // second request still allowed by rpm
        assert!(rl.admit(&k, 1).is_err()); // now rpm is out
    }

    #[test]
    fn parse_keys() {
        let s = KeyStore::parse(r#"{"keys":[{"key":"sk-alice-1","name":"alice","models":["qwen*","llama3"],"rpm":60},"sk-plain-02"]}"#).unwrap();
        assert_eq!(s.keys.len(), 2);
        let a = s.authenticate("sk-alice-1").unwrap();
        assert_eq!(a.name, "alice");
        assert!(a.allows("qwen2.5-7b") && a.allows("llama3") && !a.allows("llama3.1") && !a.allows("mistral"));
        assert!(s.authenticate("sk-plain-02").unwrap().allows("anything"));
        assert!(s.authenticate("sk-alice-").is_none() && s.authenticate("").is_none() && s.authenticate("sk-alice-12").is_none());
        for bad in ["", "{}", "[]", r#"[{"name":"x"}]"#, r#"["short"]"#, r#"["sk-aaaaaaaa","sk-aaaaaaaa"]"#, r#"[{"key":"sk-aaaaaaaa","rpm":-1}]"#, r#"[{"key":"sk-aaaaaaaa","models":"x"}]"#, "[1]"] {
            assert!(KeyStore::parse(bad).is_err(), "{bad}");
        }
        assert_eq!(KeyStore::parse(r#"["sk-aaaaaaaa"]"#).unwrap().keys[0].name, "key1");
    }

    #[test]
    fn constant_time_compare_semantics() {
        assert!(ct_eq("abc", "abc") && !ct_eq("abc", "abd") && !ct_eq("abc", "ab") && !ct_eq("", "a") && ct_eq("", ""));
    }
}
