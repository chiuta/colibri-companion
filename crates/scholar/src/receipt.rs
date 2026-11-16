// Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//! Provenance receipts: a small JSON record that ties one question, the model that answered, the answer and each
//! passage it was shown together by SHA-256, so that a later change to any of them is detectable.
//!
//! # What a receipt proves
//!
//! * **Intact**: the `digest` is the SHA-256 of the receipt's own contents in the canonical form of RFC 8785 (JCS,
//!   receipt format 2, `"tezaur-receipt/2"`), so reformatting the file or reordering its keys changes nothing and any
//!   edit of a field is caught.
//! * **Sealed** (optional): the `hmac` is HMAC-SHA-256 over the same bytes with a key only you know; nobody without
//!   the key can issue or alter a receipt that passes. The key is a parameter here (the original read an
//!   environment variable), and the comparison takes the same time wherever the first difference is.
//! * **Answer matches**: the receipt names the answer by SHA-256, so a copy of the answer can be checked against it.
//! * **Passages still exist**: each passage is named by SHA-256, page and document; a [`PassageSet`] built from the
//!   current library says whether they are still there unchanged.
//!
//! It does **not** prove that the answer is correct, that the passages say what the answer says (that is what
//! [`crate::audit`] is for), who wrote anything, or when: `created` is whatever clock the issuing computer had. A
//! receipt without an HMAC can be re-issued by anyone who edits the answer and recomputes the digest; only the
//! seal stops that.
//!
//! Only format 2 is issued. Format 1 (the original default) is still verified: its digest depended on the key order
//! of a particular JSON library, which is reproduced here explicitly (sorted keys, compact), so it no longer
//! depends on a build feature of `serde_json`.

use crate::jcs::{self, J};
use crate::sanitize;
use crate::sha256::{ct_eq_hex, hex, hmac_sha256, sha256_hex};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fmt;

pub const FORMAT_V1: &str = "tezaur-receipt/1";
pub const FORMAT_V2: &str = "tezaur-receipt/2";

/// Largest receipt text that is parsed.
pub const MAX_RECEIPT_BYTES: usize = 16 << 20;
/// Most excerpts in one receipt.
pub const MAX_EXCERPTS: usize = 10_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptError(pub String);

impl fmt::Display for ReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ReceiptError {}

fn rerr<T>(m: impl Into<String>) -> Result<T, ReceiptError> {
    Err(ReceiptError(m.into()))
}

/// Seconds since the Unix epoch on this computer's clock (0 if the clock is before 1970).
pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// One passage the model was shown.
#[derive(Clone, Debug)]
pub struct ExcerptRecord<'a> {
    /// Where the document lives (a path or URL; sanitised to one line).
    pub path: &'a str,
    pub title: &'a str,
    pub page: u32,
    /// The passage text exactly as it was shown to the model (only its SHA-256 goes into the receipt).
    pub text: &'a str,
    /// SHA-256 of the document file, hex; `None` for documents indexed before identities were recorded (the
    /// receipt then says so and the identity is weak).
    pub doc_sha256: Option<&'a str>,
    /// The 64-bit non-cryptographic document hash of the original, when the index has one.
    pub weak_hash: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct Issue<'a> {
    pub question: &'a str,
    pub model: &'a str,
    /// How the answer was produced ("ask", "draft"...); free text, one line.
    pub mode: &'a str,
    pub answer: &'a str,
    pub excerpts: &'a [ExcerptRecord<'a>],
    /// Seconds since the epoch; see [`now_secs`].
    pub created: u64,
    /// Name and version of the issuing program.
    pub tool: &'a str,
    /// From [`index_fingerprint`]; `None` when some document has no identity.
    pub index: Option<String>,
    /// The seal key; `None` for an unsealed receipt. Must not be empty.
    pub key: Option<&'a [u8]>,
}

/// SHA-256 over the documents' SHA-256 identities, sorted and joined by commas; `None` unless every document has
/// one (a fingerprint with holes would prove nothing).
pub fn index_fingerprint(doc_sha256: &[Option<&str>]) -> Option<String> {
    let mut h: Vec<&str> = Vec::with_capacity(doc_sha256.len());
    for d in doc_sha256 {
        h.push((*d)?);
    }
    h.sort_unstable();
    Some(sha256_hex(h.join(",").as_bytes()))
}

/// Issues a format-2 receipt as a JSON value (digest and optional HMAC included).
pub fn issue_value(i: &Issue<'_>) -> Result<Value, ReceiptError> {
    if i.excerpts.len() > MAX_EXCERPTS {
        return rerr(format!("a receipt holds at most {MAX_EXCERPTS} excerpts"));
    }
    if matches!(i.key, Some(k) if k.is_empty()) {
        return rerr("the seal key is empty");
    }
    let ex: Vec<Value> = i
        .excerpts
        .iter()
        .enumerate()
        .map(|(n, e)| {
            let mut v = json!({
                "n": n + 1,
                "path": sanitize::line(e.path),
                "title": sanitize::line(e.title),
                "page": e.page,
                "chunk_sha256": sha256_hex(e.text.as_bytes()),
            });
            if let Some(h) = e.weak_hash {
                v["doc_hash"] = json!(format!("{h:016x}"));
            }
            match e.doc_sha256 {
                Some(s) => v["doc_sha256"] = json!(s.trim().to_ascii_lowercase()),
                None => v["doc_identity"] = json!("weak (64-bit, non-cryptographic)"),
            }
            v
        })
        .collect();
    let mut body = json!({
        "format": FORMAT_V2,
        "created": i.created,
        "tool": sanitize::short(i.tool, 120),
        "model": sanitize::short(i.model, 200),
        "mode": sanitize::short(i.mode, 80),
        "question": i.question,
        "answer_sha256": sha256_hex(i.answer.as_bytes()),
        "excerpts": ex,
    });
    match &i.index {
        Some(f) => body["index"] = json!(f),
        None => {
            body["index_omitted"] = json!(
                "some documents were indexed before SHA-256 identities were recorded; re-ingest them to get an index fingerprint"
            )
        }
    }
    let j = jcs::from_value(&body).map_err(|e| ReceiptError(e.to_string()))?;
    let bytes = jcs::canonical(&j, true).map_err(|e| ReceiptError(e.to_string()))?;
    body["digest"] = json!(sha256_hex(&bytes));
    if let Some(k) = i.key {
        body["hmac"] = json!(hex(&hmac_sha256(k, &bytes)));
    }
    Ok(body)
}

/// Issues a format-2 receipt as pretty-printed JSON text.
pub fn issue(i: &Issue<'_>) -> Result<String, ReceiptError> {
    let v = issue_value(i)?;
    serde_json::to_string_pretty(&v).map_err(|e| ReceiptError(e.to_string()))
}

// ---------------------------------------------------------------- the library side

/// The passages of the current library, keyed so that a receipt's excerpts are looked up in constant time (the
/// original hashed every chunk of the index once per excerpt).
#[derive(Default, Debug, Clone)]
pub struct PassageSet {
    /// (path, page, SHA-256 of the passage text) -> document identities seen there.
    by_key: HashMap<(String, u32, String), Vec<Option<String>>>,
    n: usize,
}

impl PassageSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one passage of the library.
    pub fn insert(&mut self, path: &str, page: u32, text: &str, doc_sha256: Option<&str>) {
        self.n += 1;
        self.by_key
            .entry((sanitize::line(path), page, sha256_hex(text.as_bytes())))
            .or_default()
            .push(doc_sha256.map(|s| s.trim().to_ascii_lowercase()));
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Is there a passage with this path, page and text hash, and (when both sides have one) the same document
    /// identity?
    fn has(&self, path: &str, page: u32, chunk_sha256: &str, want_doc: Option<&str>) -> bool {
        let k = (path.to_string(), page, chunk_sha256.to_ascii_lowercase());
        self.by_key.get(&k).is_some_and(|docs| {
            docs.iter().any(|have| match (want_doc, have) {
                (Some(w), Some(h)) => w.eq_ignore_ascii_case(h),
                _ => true,
            })
        })
    }
}

// ---------------------------------------------------------------- verification

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Pass,
    Fail,
    /// Informational: something was not checked, or is weaker than it could be.
    Note,
}

#[derive(Clone, Debug)]
pub struct Check {
    /// Stable code: `format`, `digest`, `hmac`, `hmac-unchecked`, `answer`, `excerpt-missing`, `excerpts`,
    /// `weak-identity`, `unreadable`.
    pub code: &'static str,
    pub status: Status,
    pub detail: String,
}

#[derive(Clone, Debug)]
pub struct Verification {
    /// True when no check failed. A receipt without an HMAC can be `ok` and still have been re-issued by someone
    /// else; see the module docs.
    pub ok: bool,
    pub checks: Vec<Check>,
}

impl Verification {
    pub fn has(&self, code: &str, status: Status) -> bool {
        self.checks
            .iter()
            .any(|c| c.code == code && c.status == status)
    }

    /// One line per check with a leading mark.
    pub fn lines(&self) -> Vec<String> {
        self.checks
            .iter()
            .map(|c| {
                let m = match c.status {
                    Status::Pass => "✓",
                    Status::Fail => "✗",
                    Status::Note => "•",
                };
                format!("{m} {}", c.detail)
            })
            .collect()
    }

    fn push(&mut self, code: &'static str, status: Status, detail: impl Into<String>) {
        if status == Status::Fail {
            self.ok = false;
        }
        self.checks.push(Check {
            code,
            status,
            detail: detail.into(),
        });
    }
}

/// What was claimed about each excerpt, as read from the receipt.
struct Claim {
    n: String,
    path: String,
    page: u64,
    chunk_sha256: String,
    doc_sha256: Option<String>,
}

/// Verifies a receipt as written in a file: any spacing and key order are accepted.
///
/// * `key`: the seal key. With a key, the receipt must carry a matching `hmac` (a receipt stripped of its seal fails);
///   without one, an `hmac` is reported as not checked.
/// * `answer`: a copy of the answer, to check it is the one the receipt was issued for.
/// * `library`: the current passages, to check that every excerpt still exists unchanged.
pub fn verify(
    receipt_text: &str,
    key: Option<&[u8]>,
    answer: Option<&str>,
    library: Option<&PassageSet>,
) -> Verification {
    let mut v = Verification {
        ok: true,
        checks: Vec::new(),
    };
    if receipt_text.len() > MAX_RECEIPT_BYTES {
        v.push(
            "unreadable",
            Status::Fail,
            "the receipt is too large to be a receipt",
        );
        return v;
    }
    // format first, from a lenient read, so an old receipt is not rejected by the strict reader
    let Ok(lenient) = serde_json::from_str::<Value>(receipt_text) else {
        v.push("unreadable", Status::Fail, "not a JSON receipt");
        return v;
    };
    match lenient.get("format").and_then(Value::as_str) {
        Some(FORMAT_V2) => verify_v2(receipt_text, key, answer, library, v),
        Some(FORMAT_V1) => verify_v1(lenient, key, answer, library, v),
        Some(other) => {
            v.push(
                "format",
                Status::Fail,
                format!("unknown receipt format {:?}", sanitize::short(other, 60)),
            );
            v
        }
        None => {
            v.push(
                "format",
                Status::Fail,
                "the receipt does not name its format",
            );
            v
        }
    }
}

fn verify_v2(
    text: &str,
    key: Option<&[u8]>,
    answer: Option<&str>,
    library: Option<&PassageSet>,
    mut v: Verification,
) -> Verification {
    // one strict reader for both the digest and the fields: no duplicate keys, so no differing readings
    let mut j = match jcs::parse(text) {
        Ok(j) => j,
        Err(e) => {
            v.push("unreadable", Status::Fail, format!("invalid receipt: {e}"));
            return v;
        }
    };
    let digest = j
        .take("digest")
        .and_then(|d| d.as_str().map(str::to_string));
    let hm = j.take("hmac").and_then(|d| d.as_str().map(str::to_string));
    let bytes = match jcs::canonical(&j, true) {
        Ok(b) => b,
        Err(e) => {
            v.push("unreadable", Status::Fail, format!("invalid receipt: {e}"));
            return v;
        }
    };
    let calc = sha256_hex(&bytes);
    seal_checks(&mut v, digest.as_deref(), &calc, hm.as_deref(), key, &bytes);
    let answer_sha = j
        .get("answer_sha256")
        .and_then(J::as_str)
        .map(str::to_string);
    answer_check(&mut v, answer, answer_sha.as_deref());
    if let Some(lib) = library {
        let claims = match j.get("excerpts") {
            Some(J::Arr(a)) => a.iter().map(claim_v2).collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        excerpt_checks(&mut v, &claims, lib, true);
    }
    v
}

fn claim_v2(e: &J) -> Option<Claim> {
    Some(Claim {
        n: match e.get("n") {
            Some(J::Num(n)) => jcs::number(*n).unwrap_or_default(),
            _ => "?".into(),
        },
        path: e.get("path")?.as_str()?.to_string(),
        page: e.get("page").and_then(J::as_u64)?,
        chunk_sha256: e.get("chunk_sha256")?.as_str()?.to_string(),
        doc_sha256: e.get("doc_sha256").and_then(J::as_str).map(str::to_string),
    })
}

/// The bytes a format-1 digest covers: the body as compact JSON with keys in sorted (byte) order, which is what
/// `serde_json` produced for the original.
fn v1_bytes(v: &Value, out: &mut String) {
    match v {
        Value::Object(o) => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push(':');
                v1_bytes(&o[k], out);
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                v1_bytes(x, out);
            }
            out.push(']');
        }
        other => out.push_str(&serde_json::to_string(other).unwrap_or_default()),
    }
}

fn verify_v1(
    mut body: Value,
    key: Option<&[u8]>,
    answer: Option<&str>,
    library: Option<&PassageSet>,
    mut v: Verification,
) -> Verification {
    let digest = body
        .as_object_mut()
        .and_then(|o| o.remove("digest"))
        .and_then(|d| d.as_str().map(str::to_string));
    let hm = body
        .as_object_mut()
        .and_then(|o| o.remove("hmac"))
        .and_then(|d| d.as_str().map(str::to_string));
    let mut s = String::new();
    v1_bytes(&body, &mut s);
    let calc = sha256_hex(s.as_bytes());
    // in format 1 the HMAC is over the hex digest, not over the body
    let mac_msg = calc.clone().into_bytes();
    seal_checks(
        &mut v,
        digest.as_deref(),
        &calc,
        hm.as_deref(),
        key,
        &mac_msg,
    );
    answer_check(
        &mut v,
        answer,
        body.get("answer_sha256").and_then(Value::as_str),
    );
    if let Some(lib) = library {
        let claims: Vec<Option<Claim>> = body
            .get("excerpts")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|e| {
                        Some(Claim {
                            n: e.get("n")
                                .map(Value::to_string)
                                .unwrap_or_else(|| "?".into()),
                            path: e.get("path")?.as_str()?.to_string(),
                            page: e.get("page").and_then(Value::as_u64)?,
                            chunk_sha256: e.get("chunk_sha256")?.as_str()?.to_string(),
                            doc_sha256: e
                                .get("doc_sha256")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        excerpt_checks(&mut v, &claims, lib, false);
    }
    v
}

fn seal_checks(
    v: &mut Verification,
    digest: Option<&str>,
    calc: &str,
    hmac: Option<&str>,
    key: Option<&[u8]>,
    mac_msg: &[u8],
) {
    match digest {
        Some(d) if ct_eq_hex(d, calc) => v.push(
            "digest",
            Status::Pass,
            "receipt is intact (digest matches its contents)",
        ),
        Some(_) => v.push(
            "digest",
            Status::Fail,
            "receipt was altered: digest does not match its contents",
        ),
        None => v.push("digest", Status::Fail, "the receipt has no digest"),
    }
    if let Some(h) = hmac {
        match key {
            Some(k) if !k.is_empty() => {
                if ct_eq_hex(&hex(&hmac_sha256(k, mac_msg)), h) {
                    v.push("hmac", Status::Pass, "HMAC matches the key you gave");
                } else {
                    v.push("hmac", Status::Fail, "HMAC does not match the key you gave");
                }
            }
            _ => v.push(
                "hmac-unchecked",
                Status::Note,
                "receipt carries an HMAC, but no key was given: the seal was not checked",
            ),
        }
    } else if key.is_some() {
        // someone who gives a key expects a seal: a receipt stripped of its HMAC must not pass as sealed
        v.push(
            "hmac",
            Status::Fail,
            "a key was given, but this receipt has no HMAC: it is not sealed",
        );
    }
}

fn answer_check(v: &mut Verification, answer: Option<&str>, want: Option<&str>) {
    if let Some(a) = answer {
        if want.is_some_and(|w| ct_eq_hex(&sha256_hex(a.as_bytes()), w)) {
            v.push(
                "answer",
                Status::Pass,
                "the answer text is exactly the one this receipt was issued for",
            );
        } else {
            v.push(
                "answer",
                Status::Fail,
                "the answer text differs from the one this receipt was issued for",
            );
        }
    }
}

fn excerpt_checks(v: &mut Verification, claims: &[Option<Claim>], lib: &PassageSet, strict: bool) {
    let (mut found, mut total) = (0usize, 0usize);
    for c in claims.iter().take(MAX_EXCERPTS) {
        total += 1;
        let Some(c) = c else {
            v.push(
                "excerpt-missing",
                Status::Fail,
                format!("excerpt {total} is malformed"),
            );
            continue;
        };
        let page = u32::try_from(c.page).unwrap_or(u32::MAX);
        if lib.has(
            &sanitize::line(&c.path),
            page,
            &c.chunk_sha256,
            c.doc_sha256.as_deref(),
        ) {
            found += 1;
        } else {
            v.push(
                "excerpt-missing",
                Status::Fail,
                format!(
                    "excerpt {} ({}, p. {}) is not in this index as recorded",
                    sanitize::short(&c.n, 12),
                    sanitize::short(&c.path, 120),
                    c.page
                ),
            );
        }
        if strict && c.doc_sha256.is_none() {
            v.push(
                "weak-identity",
                Status::Note,
                format!(
                    "excerpt {} names its document by a weak identity (64-bit, non-cryptographic)",
                    sanitize::short(&c.n, 12)
                ),
            );
        }
    }
    if claims.len() > MAX_EXCERPTS {
        v.push(
            "excerpts",
            Status::Fail,
            format!(
                "more than {MAX_EXCERPTS} excerpts: only the first {MAX_EXCERPTS} were checked"
            ),
        );
    }
    if found == total && claims.len() <= MAX_EXCERPTS {
        v.push(
            "excerpts",
            Status::Pass,
            format!("all {total} excerpts exist unchanged in the current index"),
        );
    }
}

#[cfg(test)]
mod tests;
