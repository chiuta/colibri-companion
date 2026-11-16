use super::*;

const KEY: &[u8] = b"correct horse battery staple";

fn ex<'a>() -> Vec<ExcerptRecord<'a>> {
    vec![
        ExcerptRecord {
            path: "Library/smith2019.pdf",
            title: "Smith 2019",
            page: 3,
            text: "The burden of proof lies with the competent authority.",
            doc_sha256: Some("aa11"),
            weak_hash: Some(7),
        },
        ExcerptRecord {
            path: "Library/doe2020.txt",
            title: "Doe 2020",
            page: 0,
            text: "Mutual trust underpins recognition.",
            doc_sha256: None,
            weak_hash: None,
        },
    ]
}

fn issue_with<'a>(excerpts: &'a [ExcerptRecord<'a>], key: Option<&'a [u8]>) -> Issue<'a> {
    Issue {
        question: "Who bears the burden of proof?",
        model: "llama-3-8b",
        mode: "ask",
        answer: "The authority [1].",
        excerpts,
        created: 1_800_000_000,
        tool: "colibri-companion 0.1",
        index: None,
        key,
    }
}

fn lib() -> PassageSet {
    let mut l = PassageSet::new();
    l.insert(
        "Library/smith2019.pdf",
        3,
        "The burden of proof lies with the competent authority.",
        Some("AA11"),
    );
    l.insert(
        "Library/doe2020.txt",
        0,
        "Mutual trust underpins recognition.",
        None,
    );
    l
}

#[test]
fn issue_and_verify_round_trip() {
    let e = ex();
    for key in [None, Some(KEY)] {
        let r = issue(&issue_with(&e, key)).unwrap();
        let v = verify(&r, key, Some("The authority [1]."), Some(&lib()));
        assert!(v.ok, "{:?}", v.lines());
        assert!(
            v.has("digest", Status::Pass)
                && v.has("answer", Status::Pass)
                && v.has("excerpts", Status::Pass)
        );
        assert_eq!(v.has("hmac", Status::Pass), key.is_some());
        assert!(v.has("weak-identity", Status::Note)); // the second excerpt has no SHA-256 document identity
    }
}

#[test]
fn reformatting_and_key_order_do_not_matter() {
    let e = ex();
    let r = issue(&issue_with(&e, Some(KEY))).unwrap();
    let val: Value = serde_json::from_str(&r).unwrap();
    // compact, then re-serialised with the keys in reverse order
    let compact = serde_json::to_string(&val).unwrap();
    assert!(verify(&compact, Some(KEY), None, None).ok);
    let Value::Object(o) = &val else {
        unreachable!()
    };
    let mut rev = String::from("{");
    for (i, (k, x)) in o.iter().rev().enumerate() {
        if i > 0 {
            rev.push_str(",\n   ");
        }
        rev.push_str(&format!("{}  :  {}", serde_json::to_string(k).unwrap(), x));
    }
    rev.push('}');
    assert!(verify(&rev, Some(KEY), None, None).ok, "{rev}");
}

#[test]
fn every_edited_field_is_caught() {
    let e = ex();
    let r = issue(&issue_with(&e, None)).unwrap();
    let val: Value = serde_json::from_str(&r).unwrap();
    let edits: Vec<(&str, Value)> = vec![
        ("question", json!("Another question?")),
        ("model", json!("gpt-5")),
        ("created", json!(1)),
        ("answer_sha256", json!("00")),
        ("mode", json!("x")),
        ("tool", json!("x")),
        ("excerpts", json!([])),
        ("index_omitted", json!("fine")),
    ];
    for (k, nv) in edits {
        let mut m = val.clone();
        m[k] = nv;
        let v = verify(&m.to_string(), None, None, None);
        assert!(
            !v.ok && v.has("digest", Status::Fail),
            "editing {k} went unnoticed"
        );
    }
    // an added field is also a change
    let mut m = val.clone();
    m["extra"] = json!(1);
    assert!(!verify(&m.to_string(), None, None, None).ok);
}

#[test]
fn without_the_key_a_receipt_can_be_reissued_with_it_it_cannot() {
    let e = ex();
    let r = issue(&issue_with(&e, Some(KEY))).unwrap();
    let mut m: Value = serde_json::from_str(&r).unwrap();
    // an attacker swaps the answer hash and recomputes the digest, keeping the old HMAC
    m["answer_sha256"] = json!(sha256_hex(b"forged answer"));
    m.as_object_mut().unwrap().remove("digest");
    m.as_object_mut().unwrap().remove("hmac");
    let bytes = jcs::canonical(&jcs::from_value(&m).unwrap(), true).unwrap();
    m["digest"] = json!(sha256_hex(&bytes));
    // without the seal the forgery is indistinguishable from the real thing
    assert!(verify(&m.to_string(), None, Some("forged answer"), None).ok);
    // with an HMAC of the wrong key or no HMAC at all, the key holder sees it
    let v = verify(&m.to_string(), Some(KEY), Some("forged answer"), None);
    assert!(
        !v.ok && v.has("hmac", Status::Fail),
        "a stripped seal must not pass when a key is given"
    );
    m["hmac"] = json!(hex(&hmac_sha256(b"attacker key", &bytes)));
    let v = verify(&m.to_string(), Some(KEY), None, None);
    assert!(!v.ok && v.has("hmac", Status::Fail));
}

#[test]
fn seal_comparison_ignores_hex_case_and_checks_the_key() {
    let e = ex();
    let r = issue(&issue_with(&e, Some(KEY))).unwrap();
    let mut m: Value = serde_json::from_str(&r).unwrap();
    let up = m["hmac"].as_str().unwrap().to_uppercase();
    m["hmac"] = json!(up);
    assert!(verify(&m.to_string(), Some(KEY), None, None).ok);
    let v = verify(&m.to_string(), Some(b"wrong"), None, None);
    assert!(!v.ok);
    let v = verify(&m.to_string(), None, None, None);
    assert!(v.ok && v.has("hmac-unchecked", Status::Note));
}

#[test]
fn answer_and_library_checks() {
    let e = ex();
    let r = issue(&issue_with(&e, None)).unwrap();
    assert!(!verify(&r, None, Some("The authority [1]"), None).ok); // one character off
    // the library has changed: a passage was edited
    let mut changed = PassageSet::new();
    changed.insert(
        "Library/smith2019.pdf",
        3,
        "The burden of proof lies with the applicant.",
        Some("aa11"),
    );
    changed.insert(
        "Library/doe2020.txt",
        0,
        "Mutual trust underpins recognition.",
        None,
    );
    let v = verify(&r, None, None, Some(&changed));
    assert!(!v.ok && v.has("excerpt-missing", Status::Fail));
    // same passage text, other document
    let mut other_doc = PassageSet::new();
    other_doc.insert(
        "Library/smith2019.pdf",
        3,
        "The burden of proof lies with the competent authority.",
        Some("bb22"),
    );
    other_doc.insert(
        "Library/doe2020.txt",
        0,
        "Mutual trust underpins recognition.",
        None,
    );
    assert!(!verify(&r, None, None, Some(&other_doc)).ok);
    // wrong page
    let mut wrong_page = PassageSet::new();
    wrong_page.insert(
        "Library/smith2019.pdf",
        4,
        "The burden of proof lies with the competent authority.",
        Some("aa11"),
    );
    wrong_page.insert(
        "Library/doe2020.txt",
        0,
        "Mutual trust underpins recognition.",
        None,
    );
    assert!(!verify(&r, None, None, Some(&wrong_page)).ok);
}

#[test]
fn index_fingerprint_needs_every_identity() {
    assert!(index_fingerprint(&[Some("b"), None]).is_none());
    let a = index_fingerprint(&[Some("b"), Some("a")]).unwrap();
    assert_eq!(a, index_fingerprint(&[Some("a"), Some("b")]).unwrap()); // order does not matter
    assert_eq!(a, sha256_hex(b"a,b"));
    let e = ex();
    let mut i = issue_with(&e, None);
    i.index = Some(a.clone());
    let v = issue_value(&i).unwrap();
    assert_eq!(v["index"], json!(a));
    assert!(v.get("index_omitted").is_none());
}

#[test]
fn a_frozen_receipt_keeps_verifying() {
    // issued once, then frozen: the canonical form and the digest must never change, or old receipts break
    let e = [ExcerptRecord {
        path: "a.txt",
        title: "A",
        page: 1,
        text: "abc",
        doc_sha256: Some("00"),
        weak_hash: None,
    }];
    let mut i = issue_with(&e, Some(b"k"));
    i.index = Some("ff".into());
    let v = issue_value(&i).unwrap();
    let mut body = v.clone();
    body.as_object_mut().unwrap().remove("digest");
    body.as_object_mut().unwrap().remove("hmac");
    let canon =
        String::from_utf8(jcs::canonical(&jcs::from_value(&body).unwrap(), true).unwrap()).unwrap();
    assert_eq!(
        canon,
        "{\"answer_sha256\":\"c6a2b5a3bb06fb755673f5a3833f31cf24fe6802f0b7c074daa8d45c68736bc5\",\"created\":1800000000,\"excerpts\":[{\"chunk_sha256\":\"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\",\"doc_sha256\":\"00\",\"n\":1,\"page\":1,\"path\":\"a.txt\",\"title\":\"A\"}],\"format\":\"tezaur-receipt/2\",\"index\":\"ff\",\"mode\":\"ask\",\"model\":\"llama-3-8b\",\"question\":\"Who bears the burden of proof?\",\"tool\":\"colibri-companion 0.1\"}"
    );
}

#[test]
fn format_one_receipts_still_verify() {
    // the original default: digest = SHA-256 of the body serialised with sorted keys, HMAC over the hex digest
    let mut body = json!({
        "answer_sha256": sha256_hex(b"old answer"),
        "created": 1_700_000_000u64,
        "excerpts": [{"chunk_sha256": sha256_hex(b"chunk"), "doc_hash": "0000000000000007", "n": 1, "page": 2, "path": "old.pdf", "title": "Old"}],
        "format": FORMAT_V1,
        "index": "abc",
        "mode": "ask",
        "model": "m",
        "question": "q? é",
        "tool": "tezaur 0.9",
    });
    let mut s = String::new();
    v1_bytes(&body, &mut s);
    let digest = sha256_hex(s.as_bytes());
    body["digest"] = json!(digest.clone());
    body["hmac"] = json!(hex(&hmac_sha256(b"k1", digest.as_bytes())));
    let mut lib = PassageSet::new();
    lib.insert("old.pdf", 2, "chunk", None);
    let v = verify(
        &body.to_string(),
        Some(b"k1"),
        Some("old answer"),
        Some(&lib),
    );
    assert!(v.ok, "{:?}", v.lines());
    assert!(!verify(&body.to_string(), Some(b"k2"), None, None).ok);
    body["question"] = json!("changed");
    assert!(!verify(&body.to_string(), None, None, None).ok);
}

#[test]
fn issue_refuses_what_cannot_be_canonical_or_safe() {
    let e = ex();
    let mut i = issue_with(&e, None);
    i.created = u64::MAX; // not an integer below 2^53
    assert!(issue(&i).is_err());
    let i = issue_with(&e, Some(b""));
    assert!(issue(&i).is_err());
    let many: Vec<ExcerptRecord<'_>> = (0..MAX_EXCERPTS + 1)
        .map(|_| ExcerptRecord {
            path: "p",
            title: "t",
            page: 1,
            text: "x",
            doc_sha256: None,
            weak_hash: None,
        })
        .collect();
    assert!(issue(&issue_with(&many, None)).is_err());
    // control characters in titles and paths cannot reach the receipt
    let bad = [ExcerptRecord {
        path: "a\u{202e}\nb",
        title: "t\u{0}",
        page: 1,
        text: "x",
        doc_sha256: None,
        weak_hash: None,
    }];
    let r = issue(&issue_with(&bad, None)).unwrap();
    assert!(!r.contains("\\u202e") && !r.contains("\u{202e}") && !r.contains("\\u0000"));
}

#[test]
fn hostile_receipts_never_panic() {
    let e = ex();
    let good = issue(&issue_with(&e, Some(KEY))).unwrap();
    for bad in [
        String::new(),
        "null".to_string(),
        "[]".to_string(),
        "{}".to_string(),
        r#"{"format":"tezaur-receipt/2"}"#.to_string(),
        r#"{"format":"tezaur-receipt/2","digest":5,"hmac":[]}"#.to_string(),
        r#"{"format":"tezaur-receipt/2","format":"tezaur-receipt/2"}"#.to_string(),
        r#"{"format":"tezaur-receipt/2","excerpts":[1,"a",{"path":1}]}"#.to_string(),
        r#"{"format":"tezaur-receipt/1","excerpts":[1,"a",{"path":1}]}"#.to_string(),
        r#"{"format":"tezaur-receipt/9"}"#.to_string(),
        r#"{"format":"tezaur-receipt/2","created":1.5}"#.to_string(),
        "[".repeat(100_000),
        format!(
            "{{\"format\":\"tezaur-receipt/2\",\"x\":{}1{}}}",
            "[".repeat(5_000),
            "]".repeat(5_000)
        ),
        "x".repeat(MAX_RECEIPT_BYTES + 1),
    ] {
        let v = verify(&bad, Some(KEY), Some("a"), Some(&lib()));
        assert!(!v.ok, "{}", &bad.chars().take(60).collect::<String>());
    }
    // a duplicated digest property is refused rather than resolved in someone's favour
    let dup = good.replacen("\"digest\"", "\"digest\": \"00\", \"digest\"", 1);
    assert!(!verify(&dup, Some(KEY), None, None).ok);
    // random single-character mutations: never a panic, and an accepted receipt must still be a byte-wise
    // equal reading of the original after canonicalisation
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let chars: Vec<char> = good.chars().collect();
    let original = jcs::canonical(&jcs::parse(&good).unwrap(), true).unwrap();
    for _ in 0..500 {
        let mut c = chars.clone();
        let at = (next() as usize) % c.len();
        c[at] = ['0', 'a', '"', '{', ',', ' ', 'é'][(next() as usize) % 7];
        let t: String = c.into_iter().collect();
        let v = verify(&t, Some(KEY), None, None);
        if v.ok {
            assert_eq!(
                jcs::canonical(&jcs::parse(&t).unwrap(), true).unwrap(),
                original
            );
        }
    }
}

#[test]
fn a_huge_number_of_excerpts_is_reported_not_trusted() {
    let excerpts: Vec<Value> = (0..MAX_EXCERPTS + 50)
        .map(|n| json!({"n": n, "path": "p", "page": 1, "chunk_sha256": "00"}))
        .collect();
    let mut body = json!({"format": FORMAT_V2, "excerpts": excerpts});
    let bytes = jcs::canonical(&jcs::from_value(&body).unwrap(), true).unwrap();
    body["digest"] = json!(sha256_hex(&bytes));
    let t0 = std::time::Instant::now();
    let v = verify(&body.to_string(), None, None, Some(&lib()));
    assert!(!v.ok && v.has("excerpts", Status::Fail));
    assert!(t0.elapsed().as_secs() < 20);
}
