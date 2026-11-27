//! `corbul map-fit SRC DST` and `corbul map-check SRC DST`: carry cache rows from one model to
//! another (see `crates/engine/src/kvmap.rs`).
//!
//! * `map-fit SRC DST --text FILE --out MAP [--rows 1024] [--ridge 0.001]` reads the text with both
//!   models, fits a linear map from SRC's cache rows to DST's for the same tokens, prints how well it
//!   predicts rows it did not see (against the trivial guess) and writes the map. The two models
//!   must tokenize the text identically; if they do not, nothing is fitted.
//! * `map-check SRC DST --map MAP --text FILE --question Q [--tolerance 0.02] [--max-new 16]` splits
//!   the text into two packs written by SRC, brings them into DST through the map and answers the
//!   question three ways: from DST's own full prefill (the truth), from the mapped cache with the
//!   usual repair, and from the mapped cache with no repair at all. It prints how many rows the
//!   repair had to recompute and how far each answer is from the truth. Nothing here says a map
//!   will pay off for your pair of models; it tells you.
use crate::args::Args;
use crate::cmds;
use companion_engine::ctxpack::{ComposeOptions, compose_mapped};
use companion_engine::kvmap::{self, KvMap};
use companion_engine::model::{Model, State};
use companion_engine::tokenizer::Tokenizer;
use std::path::Path;

struct Side {
    model: Model,
    tok: Tokenizer,
}

fn open(path: &str, a: &Args) -> Result<Side, String> {
    if crate::famrun::wants(Path::new(path)) {
        return Err(format!("{path}: mapping needs a dense model"));
    }
    let (prep, cfg) = cmds::open_prepared(&Args {
        pos: vec![path.to_string()],
        opts: a.opts.clone(),
    })?;
    let model = prep.build_model(&cfg)?;
    Ok(Side {
        model,
        tok: prep.tok,
    })
}

fn read_text(a: &Args, what: &str) -> Result<String, String> {
    let f = a.get("text").ok_or(format!("{what}: give --text FILE"))?;
    let md = std::fs::metadata(f).map_err(|e| format!("{f}: {e}"))?;
    if !md.is_file() || md.len() > (8 << 20) {
        return Err(format!("{f}: not a file, or larger than 8 MiB"));
    }
    String::from_utf8(std::fs::read(f).map_err(|e| format!("{f}: {e}"))?)
        .map_err(|_| format!("{f}: not valid UTF-8 text"))
}

fn ids(t: &Tokenizer, text: &str, bos: bool) -> Vec<usize> {
    t.encode_plain(text, bos)
        .into_iter()
        .map(|x| x as usize)
        .collect()
}

/// Both tokenizers must give the very same ids for `text`, or rows cannot be paired.
fn same_tokens(s: &Side, d: &Side, text: &str, bos: bool) -> Result<Vec<usize>, String> {
    let (a, b) = (ids(&s.tok, text, bos), ids(&d.tok, text, bos));
    if a != b {
        return Err(
            "the two models do not tokenize this text the same way, so their cache rows cannot be paired"
                .into(),
        );
    }
    Ok(a)
}

fn two(a: &Args, what: &str) -> Result<(Side, Side), String> {
    if a.pos.len() != 2 {
        return Err(format!(
            "{what}: give the source model and the target model"
        ));
    }
    Ok((open(&a.pos[0], a)?, open(&a.pos[1], a)?))
}

pub fn fit_cmd(a: &Args) -> Result<(), String> {
    cmds::apply_global(a);
    let out = a.get("out").ok_or("map-fit: give --out FILE")?.to_string();
    let rows: usize = a.num("rows")?.unwrap_or(1024);
    let ridge: f32 = a.num("ridge")?.unwrap_or(1e-3);
    if rows < 10 || ridge.is_nan() || ridge < 0.0 {
        return Err("map-fit: --rows must be at least 10 and --ridge 0 or more".into());
    }
    let text = read_text(a, "map-fit")?;
    let (s, d) = two(a, "map-fit")?;
    let toks = same_tokens(&s, &d, &text, true)?;
    if toks.len() < 10 {
        return Err("map-fit: the text is too short (hundreds of tokens are better)".into());
    }
    let t0 = std::time::Instant::now();
    let map = kvmap::fit(&s.model, &d.model, &toks, rows, ridge).map_err(|e| e.to_string())?;
    let st = map.stats;
    crate::outln!(
        "fitted on {} rows ({} held out) in {:.1}s",
        st.rows,
        st.held_out,
        t0.elapsed().as_secs_f64()
    );
    crate::outln!(
        "error on held-out rows (smaller is better; the second figure is the trivial guess): keys {:.3} vs {:.3}, values {:.3} vs {:.3}",
        st.k_err,
        st.k_base,
        st.v_err,
        st.v_base
    );
    if st.k_err >= st.k_base || st.v_err >= st.v_base {
        crate::outln!(
            "warning: the map is not clearly better than guessing the mean; use more text or a larger --ridge"
        );
    }
    let tmp = format!("{out}.part");
    std::fs::write(&tmp, map.to_bytes()).map_err(|e| format!("{tmp}: {e}"))?;
    std::fs::rename(&tmp, &out).map_err(|e| format!("{out}: {e}"))?;
    crate::outln!("map written to {out}");
    Ok(())
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|x| x.0)
        .unwrap_or(0)
}

fn greedy(m: &Model, st: &mut State, mut logits: Vec<f32>, n: usize) -> Vec<usize> {
    let ctx = m.spec.context as usize;
    let mut out = Vec::new();
    for _ in 0..n {
        let t = argmax(&logits);
        if st.pos + st.rope_gap() + 1 >= ctx {
            break;
        }
        out.push(t);
        match m.forward(st, t) {
            Ok(l) => logits = l,
            Err(_) => break,
        }
    }
    out
}

pub fn check_cmd(a: &Args) -> Result<(), String> {
    cmds::apply_global(a);
    let mapf = a.get("map").ok_or("map-check: give --map FILE")?;
    let question = a.get("question").ok_or("map-check: give --question TEXT")?;
    let tol: f32 = a.num("tolerance")?.unwrap_or(0.02);
    let max_new: usize = a.num("max-new")?.unwrap_or(16);
    if tol.is_nan() || tol < 0.0 {
        return Err("map-check: --tolerance must be 0 or more".into());
    }
    let md = std::fs::metadata(mapf).map_err(|e| format!("{mapf}: {e}"))?;
    if !md.is_file() || md.len() > (4 << 30) {
        return Err(format!("{mapf}: not a map file"));
    }
    let map = KvMap::from_bytes(&std::fs::read(mapf).map_err(|e| format!("{mapf}: {e}"))?)
        .map_err(|e| format!("{mapf}: {e}"))?;
    let text = read_text(a, "map-check")?;
    let (s, d) = two(a, "map-check")?;
    if !map.matches(&s.model, &d.model) {
        return Err("map-check: this map was fitted for a different pair of models".into());
    }
    let doc = same_tokens(&s, &d, &text, true)?;
    let tail = same_tokens(&s, &d, question, false)?;
    let ctx = d.model.spec.context.min(s.model.spec.context) as usize;
    if doc.len() < 4 || tail.is_empty() || doc.len() + tail.len() + max_new >= ctx {
        return Err(format!(
            "map-check: the text needs at least 4 tokens and text + question + answer must fit {ctx} tokens"
        ));
    }
    let mid = doc.len() / 2;
    let mut packs: Vec<State> = Vec::new();
    for part in [&doc[..mid], &doc[mid..]] {
        let mut st = s.model.new_state();
        s.model
            .prefill_cached(&mut st, part)
            .map_err(|e| e.to_string())?;
        packs.push(st);
    }
    let refs: Vec<&State> = packs.iter().collect();
    // the truth: the target reads everything itself
    let all: Vec<usize> = doc.iter().chain(&tail).copied().collect();
    let t0 = std::time::Instant::now();
    let mut truth = d.model.new_state();
    let tl = d
        .model
        .prefill_cached(&mut truth, &all)
        .map_err(|e| e.to_string())?;
    let full_s = t0.elapsed().as_secs_f64();
    let want = greedy(&d.model, &mut truth, tl.clone(), max_new);
    crate::outln!(
        "full prefill by the target: {} tokens in {full_s:.2}s",
        all.len()
    );
    for (name, o) in [
        (
            "mapped + repair",
            ComposeOptions {
                tolerance: tol,
                ..Default::default()
            },
        ),
        (
            "mapped, no repair",
            ComposeOptions {
                probe: 1,
                tolerance: f32::INFINITY,
                max_refresh: 0,
            },
        ),
    ] {
        let t0 = std::time::Instant::now();
        let (mut st, lg, rep) = compose_mapped(&map, &s.model, &refs, &d.model, &tail, &o)
            .map_err(|e| e.to_string())?;
        let dt = t0.elapsed().as_secs_f64();
        let diff = lg
            .iter()
            .zip(&tl)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let got = greedy(&d.model, &mut st, lg.clone(), max_new);
        let same = got.iter().zip(&want).take_while(|(x, y)| x == y).count();
        crate::outln!(
            "{name:<18} {dt:.2}s, rows recomputed {}/{}, largest score difference {diff:.4}, same first token: {}, matching leading tokens: {same}/{}",
            rep.refreshed,
            rep.spliced,
            argmax(&lg) == argmax(&tl),
            want.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;

    fn args(pos: &[&Path], extra: &[(&str, String)]) -> Args {
        let mut v: Vec<String> = pos.iter().map(|p| p.display().to_string()).collect();
        for (k, val) in extra {
            v.push(format!("--{k}"));
            v.push(val.clone());
        }
        Args::parse(&v)
    }

    #[test]
    fn fit_then_check_end_to_end_and_bad_input_is_refused() {
        let d = tmpdir("kvmap");
        let (a, b) = (d.join("a"), d.join("b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        write_model(&a, false);
        write_model_seeded(&b, 0x1234_5678_9abc_def1);
        let t = d.join("t.txt");
        std::fs::write(
            &t,
            "The harbour opens at six and ferries leave every half hour until dusk falls over the quiet bay.",
        )
        .unwrap();
        let short = d.join("short.txt");
        std::fs::write(
            &short,
            "The harbour opens at six and ferries leave until dusk.",
        )
        .unwrap();
        let m = d.join("m.bin");
        let ok = |extra: &[(&str, String)]| {
            let mut e = vec![
                ("text", t.display().to_string()),
                ("out", m.display().to_string()),
            ];
            e.extend_from_slice(extra);
            fit_cmd(&args(&[&a, &b], &e))
        };
        ok(&[("rows", "90".into())]).unwrap();
        assert!(m.is_file() && !d.join("m.bin.part").exists());
        let chk = |extra: &[(&str, String)]| {
            let mut e = vec![
                ("text", short.display().to_string()),
                ("map", m.display().to_string()),
                ("question", "When".into()),
                ("max-new", "3".into()),
            ];
            e.extend_from_slice(extra);
            check_cmd(&args(&[&a, &b], &e))
        };
        chk(&[]).unwrap();
        chk(&[("tolerance", "0".into())]).unwrap();
        // refused: wrong pair, bad numbers, missing options, damaged map
        assert!(
            check_cmd(&args(
                &[&b, &a],
                &[
                    ("text", t.display().to_string()),
                    ("map", m.display().to_string()),
                    ("question", "x".into()),
                ]
            ))
            .is_err()
        );
        assert!(chk(&[("tolerance", "-1".into())]).is_err());
        assert!(ok(&[("rows", "3".into())]).is_err());
        assert!(ok(&[("ridge", "-1".into())]).is_err());
        assert!(fit_cmd(&args(&[&a, &b], &[("text", t.display().to_string())])).is_err());
        assert!(
            fit_cmd(&args(
                &[&a],
                &[
                    ("text", t.display().to_string()),
                    ("out", m.display().to_string()),
                ]
            ))
            .is_err()
        );
        std::fs::write(&m, b"garbage").unwrap();
        assert!(chk(&[]).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
