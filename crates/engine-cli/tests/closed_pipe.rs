//! `corbul capabilities | head -1` must end quietly, not with a panic backtrace.
use std::process::{Command, Stdio};

#[test]
fn a_closed_standard_output_is_not_a_panic() {
    let mut c = Command::new(env!("CARGO_BIN_EXE_corbul"))
        .arg("capabilities")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(c.stdout.take()); // the reader goes away before anything is written
    let out = c.wait_with_output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!err.contains("panicked"), "{err}");
    assert!(
        out.status.success() || out.status.code() == Some(0),
        "{:?} {err}",
        out.status
    );
}

#[test]
fn the_new_subcommands_are_listed_in_the_help() {
    let o = Command::new(env!("CARGO_BIN_EXE_corbul"))
        .arg("help")
        .output()
        .unwrap();
    let t = String::from_utf8_lossy(&o.stdout);
    for w in [
        "autoquant",
        "quantize",
        "merge",
        "steer",
        "rag",
        "watermark",
        "search",
        "--draft",
        "--lookahead",
        "--self-spec",
        "--watermark",
        "/ui",
    ] {
        assert!(t.contains(w), "help does not mention {w}");
    }
}
