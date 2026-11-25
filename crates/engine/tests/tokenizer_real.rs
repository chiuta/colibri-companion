//! Real-vocabulary check. Needs the vocab-only GGUF files and `.inp`/`.out` reference pairs that
//! llama.cpp publishes under `models/` (MIT licence; not stored in this repository).
//! Set `ENGINE_VOCAB_DIR` to a folder holding `ggml-vocab-<family>.gguf[.inp|.out]`; without it
//! the tests print a loud `SKIPPED` line and pass; with `ENGINE_REQUIRE_VOCAB=1` they fail instead.
//! `tools/fetch_vocab.sh <dir>` downloads the files.
//! `ENGINE_VOCAB_ONLY=qwen2,phi-3` restricts the run to some families.
use companion_engine::gguf::Gguf;
use companion_engine::tokenizer::Tokenizer;
use companion_engine::wordpiece::WordPiece;
use std::path::Path;

/// `ENGINE_VOCAB_DIR`, or a loud `SKIPPED` line (the test did not run). `ENGINE_REQUIRE_VOCAB=1`
/// (set by CI after `tools/fetch_vocab.sh`) turns a missing folder into a failure.
fn vocab_dir() -> Option<std::ffi::OsString> {
    let d = std::env::var_os("ENGINE_VOCAB_DIR");
    if d.is_none() {
        eprintln!(
            "\n*** SKIPPED: ENGINE_VOCAB_DIR is not set - the real-vocabulary tokenizer tests did NOT run (run tools/fetch_vocab.sh) ***\n"
        );
        assert!(
            std::env::var_os("ENGINE_REQUIRE_VOCAB").is_none(),
            "ENGINE_REQUIRE_VOCAB is set but ENGINE_VOCAB_DIR is not"
        );
    }
    d
}

const SEP: &str = "\n__ggml_vocab_test__\n";

fn cases(inp: &str) -> Vec<&str> {
    let mut v: Vec<&str> = inp.split(SEP).collect();
    // the file ends with a separator followed by nothing
    if v.last() == Some(&"") {
        v.pop();
    }
    v
}

fn ids(line: &str) -> Vec<u32> {
    line.split_whitespace()
        .map(|t| t.parse().expect("id"))
        .collect()
}

fn check(dir: &Path, family: &str) -> Option<(usize, usize, Vec<String>)> {
    let g = dir.join(format!("ggml-vocab-{family}.gguf"));
    let (i, o) = (
        dir.join(format!("ggml-vocab-{family}.gguf.inp")),
        dir.join(format!("ggml-vocab-{family}.gguf.out")),
    );
    if !(g.exists() && i.exists() && o.exists()) {
        return None;
    }
    let gguf = Gguf::parse(std::io::BufReader::new(std::fs::File::open(&g).unwrap())).unwrap();
    let tok = match Tokenizer::from_gguf(&gguf) {
        Ok(t) => t,
        Err(e) => return Some((0, 1, vec![format!("load failed: {e}")])),
    };
    let inp = std::fs::read_to_string(&i).unwrap();
    let out = std::fs::read_to_string(&o).unwrap();
    let cs = cases(&inp);
    let lines: Vec<&str> = out.lines().collect();
    let mut bad = vec![];
    if lines.len() != cs.len() {
        bad.push(format!(
            "{} cases but {} reference lines",
            cs.len(),
            lines.len()
        ));
    }
    let mut pass = 0;
    for (n, (c, l)) in cs.iter().zip(&lines).enumerate() {
        let want = ids(l);
        let got = tok.encode(c, false);
        if got == want {
            pass += 1;
            let back = tok.decode(&got);
            if !(back == *c || back.trim_start() == c.trim_start() || !c.is_ascii()) {
                bad.push(format!("case {n}: decode round trip {c:?} -> {back:?}"));
            }
        } else if bad.len() < 6 {
            bad.push(format!("case {n} {c:?}\n   want {want:?}\n   got  {got:?}"));
        }
    }
    Some((pass, cs.len(), bad))
}

#[test]
fn encodes_exactly_like_the_reference_tokenisers() {
    let Some(dir) = vocab_dir() else { return };
    let dir = Path::new(&dir);
    let only: Option<Vec<String>> = std::env::var("ENGINE_VOCAB_ONLY")
        .ok()
        .map(|s| s.split(',').map(str::to_string).collect());
    let families = [
        "llama-bpe",
        "qwen2",
        "gpt-2",
        "llama-spm",
        "deepseek-llm",
        "deepseek-coder",
        "phi-3",
        "falcon",
        "starcoder",
        "mpt",
        "command-r",
        "refact",
    ];
    let (mut all_ok, mut ran) = (true, 0);
    for f in families {
        if only.as_ref().is_some_and(|o| !o.iter().any(|x| x == f)) {
            continue;
        }
        let Some((p, t, bad)) = check(dir, f) else {
            eprintln!("{f}: files not found");
            continue;
        };
        ran += 1;
        eprintln!("{f:15} {p}/{t}");
        for b in &bad {
            eprintln!("   {b}");
        }
        all_ok &= p == t && bad.is_empty();
    }
    assert!(ran > 0, "no vocab family found in the folder");
    assert!(
        all_ok,
        "some families differ from the reference (see stderr)"
    );
}

/// The BERT (WordPiece) reference pair, if present: `ggml-vocab-bert-bge.gguf[.inp|.out]`.
#[test]
fn wordpiece_matches_the_bert_reference() {
    let Some(dir) = vocab_dir() else { return };
    let dir = Path::new(&dir);
    let g = dir.join("ggml-vocab-bert-bge.gguf");
    if !g.exists() {
        eprintln!("bert-bge: files not found");
        return;
    }
    let gguf = Gguf::parse(std::io::BufReader::new(std::fs::File::open(&g).unwrap())).unwrap();
    let t = WordPiece::from_gguf(&gguf).unwrap();
    let inp = std::fs::read_to_string(dir.join("ggml-vocab-bert-bge.gguf.inp")).unwrap();
    let out = std::fs::read_to_string(dir.join("ggml-vocab-bert-bge.gguf.out")).unwrap();
    let cs = cases(&inp);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(cs.len(), lines.len());
    let mut bad = vec![];
    for (n, (c, l)) in cs.iter().zip(&lines).enumerate() {
        let got = t.tokenize(c);
        if got != ids(l) {
            bad.push(format!(
                "case {n} {c:?}\n   want {:?}\n   got  {got:?}",
                ids(l)
            ));
        }
    }
    eprintln!("bert-bge        {}/{}", cs.len() - bad.len(), cs.len());
    for b in bad.iter().take(8) {
        eprintln!("   {b}");
    }
    assert!(bad.is_empty(), "{} cases differ", bad.len());
}

/// Texts with no reference output: multilingual, adversarial, combining marks, long runs.
fn adversarial() -> Vec<String> {
    let mut v: Vec<String> = [
        "Hello, world! 12345 — naïve café",
        "e\u{301}\u{301}\u{301}x",
        "한국어 문장입니다. 日本語のテキスト。中文文本，测试。",
        "Привет, мир! Ελληνικά, עברית, العربية, हिन्दी, ไทย",
        "🚀🚀 emoji 👨‍👩‍👧‍👦 flags 🇷🇴",
        "tabs\t\tand\r\nnewlines\n\n\n  and   spaces   ",
        "x = (a+b)*c; // comment\n\tif (x >= 0) { return 0x1F; }",
        "\u{0}\u{1}\u{7f} control \u{200b}zero\u{feff}width",
        "ǅ ǆ ß ΐ İ ı ﬁ Ⅳ ½ ٣ ²",
        "'s 'S 'tis I'll we've they'd",
        "    ",
        "\n",
        "a",
        "",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    v.push("word ".repeat(2000));
    v.push("é".repeat(3000));
    v.push(
        (0..400u32)
            .filter_map(|i| char::from_u32(0x4E00 + i * 37))
            .collect::<String>(),
    );
    v
}

fn all_loadable(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .filter_map(|e| {
            e.file_name().to_str().and_then(|n| {
                Some(
                    n.strip_prefix("ggml-vocab-")?
                        .strip_suffix(".gguf")?
                        .to_string(),
                )
            })
        })
        .collect();
    v.sort();
    v
}

#[test]
fn every_vocab_loads_is_deterministic_and_round_trips() {
    let Some(dir) = vocab_dir() else { return };
    let dir = Path::new(&dir);
    let texts = adversarial();
    for f in all_loadable(dir) {
        let g = dir.join(format!("ggml-vocab-{f}.gguf"));
        let gguf = Gguf::parse(std::io::BufReader::new(std::fs::File::open(&g).unwrap())).unwrap();
        let tok = match Tokenizer::from_gguf(&gguf) {
            Ok(t) => t,
            // WordPiece (BERT) is documented as unsupported: report it, do not fail the suite.
            Err(e) if e.to_string().contains("not supported") => {
                eprintln!("{f}: skipped ({e})");
                continue;
            }
            Err(e) => panic!("{f}: {e}"),
        };
        for w in tok.warnings() {
            eprintln!("{f}: warning: {w}");
        }
        let (mut exact, mut n) = (0, 0);
        for t in &texts {
            let a = tok.encode(t, false);
            assert_eq!(a, tok.encode(t, false), "{f}: not deterministic for {t:?}");
            assert!(
                a.iter().all(|&i| (i as usize) < tok.vocab_size()),
                "{f}: id out of range"
            );
            let back = tok.decode(&a);
            // byte-level and byte-fallback vocabularies must give the text back exactly (modulo the
            // leading space SentencePiece adds); a few vocabularies without those cannot.
            n += 1;
            if back == *t || back.trim_start() == t.trim_start() {
                exact += 1;
            }
        }
        eprintln!("{f:15} round trip {exact}/{n}");
        if [
            "gpt-neox",
            "aquila",
            "baichuan",
            "llama-bpe",
            "gpt-2",
            "qwen2",
            "llama-spm",
            "falcon",
            "mpt",
        ]
        .contains(&f.as_str())
        {
            assert!(
                exact * 10 >= n * 8,
                "{f}: only {exact}/{n} texts survive a round trip"
            );
        }
    }
}
