//! `corbul packs MODEL --session DIR --pack FILE… --question TEXT`: answer a question about several
//! texts, reusing the work done on each text the last time it appeared (see `docs/CONTEXT-PACKS.md`).
//!
//! Every `--pack` file is prefilled once, alone, and kept in the session folder. Later runs, with
//! the files in any order and with any question, copy the stored rows into place and repair only
//! the joins. A prompt that starts with the BOS token (when the model has one) then each text in the
//! order given then the question; no chat template is applied, this is a plain-text completion.
//!
//! * `--tolerance X` (default 0.02) how little the rows of a join may still move before repair stops;
//!   `0` repairs everything and gives exactly what a full prefill gives.
//! * `--probe N` (default 8) rows recomputed before the first measurement.
//! * `--max-new N` (default 64) tokens to write; `--check` also runs the plain full prefill and prints
//!   how far the two answers' first-token scores are apart and the time each took.
//!
//! Without `--question` the files are only shelved.
use crate::args::{Args, parse_size};
use crate::cmds;
use companion_engine::ctxpack::{ComposeOptions, Shelf, compose};
use companion_engine::model::{KvKind, State};
use companion_engine::snapshot::SnapshotStore;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_FILE: u64 = 8 << 20;

fn read_text(p: &str) -> Result<String, String> {
    let path = Path::new(p);
    let md = std::fs::metadata(path).map_err(|e| format!("{p}: {e}"))?;
    if !md.is_file() || md.len() > MAX_FILE {
        return Err(format!("{p}: not a file, or larger than 8 MiB"));
    }
    String::from_utf8(std::fs::read(path).map_err(|e| format!("{p}: {e}"))?)
        .map_err(|_| format!("{p}: not valid UTF-8 text"))
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|x| x.0)
        .unwrap_or(0)
}

pub fn cmd(a: &Args) -> Result<(), String> {
    cmds::apply_global(a);
    let model_path = a.pos.first().ok_or("packs: give the model")?.clone();
    if crate::famrun::wants(Path::new(&model_path)) {
        return Err(
            "packs needs a dense model (state-space and hybrid models have no splicable cache)"
                .into(),
        );
    }
    let dir = a
        .get("session")
        .ok_or("packs: give --session DIR (where the packs are kept)")?;
    let cap = match a.get("session-cap") {
        Some(c) => parse_size(c)?,
        None => 2 << 30,
    };
    let files = a.all("pack");
    if files.is_empty() {
        return Err("packs: give at least one --pack FILE".into());
    }
    let texts = files
        .iter()
        .map(|f| read_text(f))
        .collect::<Result<Vec<_>, _>>()?;
    let opts = ComposeOptions {
        probe: a.num("probe")?.unwrap_or(8),
        tolerance: a.num("tolerance")?.unwrap_or(0.02),
        max_refresh: usize::MAX,
    };
    if opts.tolerance.is_nan() || opts.tolerance < 0.0 || opts.probe == 0 {
        return Err("packs: --tolerance must be 0 or more and --probe at least 1".into());
    }
    let (prep, cfg) = cmds::open_prepared(&Args {
        pos: vec![model_path],
        opts: a.opts.clone(),
    })?;
    let model = prep.build_model(&cfg)?;
    let shelf = Shelf::new(
        SnapshotStore::open(PathBuf::from(dir), cap)
            .map_err(|e| format!("--session {dir}: {e}"))?,
    );
    let ctx = model.spec.context as usize;
    let mut segs: Vec<Vec<usize>> = Vec::new();
    let head: Vec<usize> = prep
        .tok
        .encode("", true)
        .into_iter()
        .map(|t| t as usize)
        .collect();
    if !head.is_empty() {
        segs.push(head);
    }
    for t in &texts {
        let ids: Vec<usize> = prep
            .tok
            .encode(t, false)
            .into_iter()
            .map(|t| t as usize)
            .collect();
        if ids.is_empty() {
            return Err("packs: a text became no tokens".into());
        }
        segs.push(ids);
    }
    if segs.iter().map(Vec::len).sum::<usize>() >= ctx {
        return Err(format!(
            "packs: the texts do not fit the {ctx}-token context window"
        ));
    }
    let mut states: Vec<State> = Vec::new();
    for (i, s) in segs.iter().enumerate() {
        let t0 = Instant::now();
        let (st, hit) = shelf.pack(&model, s).map_err(|e| e.to_string())?;
        eprintln!(
            "pack {}: {} tokens, {} ({:.2}s)",
            i,
            s.len(),
            if hit {
                "from the shelf"
            } else {
                "computed and shelved"
            },
            t0.elapsed().as_secs_f64()
        );
        states.push(st);
    }
    let Some(q) = a.get("question") else {
        return Ok(());
    };
    let tail: Vec<usize> = prep
        .tok
        .encode(q, false)
        .into_iter()
        .map(|t| t as usize)
        .collect();
    if tail.is_empty() {
        return Err("packs: the question became no tokens".into());
    }
    let refs: Vec<&State> = states.iter().collect();
    let t0 = Instant::now();
    let (mut st, logits, rep) = compose(&model, &refs, &tail, &opts).map_err(|e| e.to_string())?;
    let took = t0.elapsed().as_secs_f64();
    eprintln!(
        "composed in {took:.2}s: {} rows copied, {} recomputed to repair {} join(s), {} fresh; {} rows did not go through the network",
        rep.spliced,
        rep.refreshed,
        rep.joins.len(),
        rep.fresh,
        rep.saved()
    );
    for (i, (n, d)) in rep.joins.iter().enumerate() {
        eprintln!("  join {}: {n} rows recomputed, last change {d:.4}", i + 1);
    }
    if a.flag("check") {
        let all: Vec<usize> = segs.iter().flatten().chain(&tail).copied().collect();
        let mut full = model.new_state_with(KvKind::F32);
        let t1 = Instant::now();
        let fl = model
            .prefill_cached(&mut full, &all)
            .map_err(|e| e.to_string())?;
        let dt = fl
            .iter()
            .zip(&logits)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "check: full prefill {:.2}s; largest score difference {dt:.4}; same first token: {}",
            t1.elapsed().as_secs_f64(),
            argmax(&fl) == argmax(&logits)
        );
    }
    let max_new: usize = a.num("max-new")?.unwrap_or(64);
    let (mut out, mut cur) = (Vec::<u32>::new(), logits);
    for _ in 0..max_new {
        let next = argmax(&cur);
        if prep.stop_ids.contains(&next) || st.pos >= ctx {
            break;
        }
        out.push(next as u32);
        cur = model.forward(&mut st, next).map_err(|e| e.to_string())?;
    }
    crate::outln!("{}", prep.tok.decode(&out));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;
    fn a(s: &[String]) -> Args {
        Args::parse(s)
    }

    #[test]
    fn shelves_then_reuses_and_checks() {
        let d = tmpdir("packs");
        let m = d.join("m");
        std::fs::create_dir_all(&m).unwrap();
        write_model(&m, false);
        let (f1, f2) = (d.join("one.txt"), d.join("two.txt"));
        std::fs::write(&f1, "The harbour opens at six.").unwrap();
        std::fs::write(&f2, "Ferries leave every half hour.").unwrap();
        let args = |q: bool, tol: &str| {
            let mut v: Vec<String> = vec![m.display().to_string()];
            for (k, val) in [
                ("session", d.join("s").display().to_string()),
                ("pack", f1.display().to_string()),
                ("pack", f2.display().to_string()),
                ("tolerance", tol.into()),
                ("max-new", "4".into()),
            ] {
                v.push(format!("--{k}"));
                v.push(val);
            }
            if q {
                v.push("--question".into());
                v.push("When".into());
                v.push("--check".into());
            }
            a(&v)
        };
        cmd(&args(false, "0")).unwrap();
        cmd(&args(true, "0")).unwrap();
        cmd(&args(true, "0.5")).unwrap();
        assert!(std::fs::read_dir(d.join("s")).unwrap().count() >= 2);
        let mut bad = args(true, "0");
        bad.opts.retain(|(k, _)| k != "pack");
        assert!(cmd(&bad).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
