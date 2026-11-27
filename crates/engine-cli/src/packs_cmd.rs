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
//! * `--pack-kv f16` keeps the packs in F16 (half the size, F16 rounding on every row), `--pack-compress`
//!   compresses the pack files without loss.
//! * `--shrink auto|R` shortens the packs: `auto` only when the texts would not fit the context
//!   window, `R` (0.1 to 1) always, keeping about that fraction of each pack's rows; `--shrink-skip N`
//!   (default 1) leaves the first N texts whole, `--shrink-window N` (default 16) sets the rows that
//!   vote, `--no-refit` skips the fit of the kept values (see `docs/KV-FIT.md`).
//! * `--max-new N` (default 64) tokens to write; `--check` also runs the plain full prefill and prints
//!   how far the two answers' first-token scores are apart and the time each took.
//!
//! Without `--question` the files are only shelved.
use crate::args::{Args, parse_size};
use crate::cmds;
use companion_engine::ctxpack::{
    ComposeOptions, ComposeReport, Shelf, ShrinkOptions, compose, shrink_pack,
};
use companion_engine::model::{KvKind, Model, State};
use companion_engine::snapshot::SnapshotStore;
use companion_engine::tokenizer::Tokenizer;
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
    let mut shelf = Shelf::new(
        SnapshotStore::open(PathBuf::from(dir), cap)
            .map_err(|e| format!("--session {dir}: {e}"))?,
    )
    .with_compression(a.flag("pack-compress"));
    shelf = match a.get("pack-kv") {
        None | Some("f32") => shelf,
        Some("f16") => shelf.with_kind(KvKind::F16).map_err(|e| e.to_string())?,
        Some(o) => return Err(format!("packs: --pack-kv {o} is not f32 or f16")),
    };
    let max_new: usize = a.num("max-new")?.unwrap_or(64);
    let q = a.get("question");
    let shrink = match a.get("shrink") {
        None => None,
        Some("auto") => Some(Shrink::default()),
        Some(v) => Some(Shrink {
            ratio: Some(v.parse::<f32>().map_err(|_| {
                format!("packs: --shrink {v}: give auto or a fraction between 0.1 and 1")
            })?),
            ..Shrink::default()
        }),
    }
    .map(|mut sh| -> Result<Shrink, String> {
        sh.skip = a.num("shrink-skip")?.unwrap_or(sh.skip);
        sh.opts.window = a.num("shrink-window")?.unwrap_or(sh.opts.window);
        sh.opts.refit = !a.flag("no-refit");
        if sh.opts.window == 0 {
            return Err("packs: --shrink-window must be at least 1".into());
        }
        Ok(sh)
    })
    .transpose()?;
    let r = answer(
        &model,
        &prep.tok,
        &prep.stop_ids,
        &shelf,
        &texts,
        q,
        &opts,
        max_new,
        a.flag("check"),
        shrink.as_ref(),
        &mut |l| eprintln!("{l}"),
    )?;
    if q.is_some() {
        crate::outln!("{}", r.text);
    }
    Ok(())
}

/// Shorten packs when asked to, or when the texts would not fit the context window.
#[derive(Clone, Debug)]
pub struct Shrink {
    /// Keep this fraction of the rows of every shrinkable pack (`None`: only as much as needed).
    pub ratio: Option<f32>,
    /// With no `ratio`: shrink only when the texts and the question would not fit.
    pub auto: bool,
    /// The first this-many texts are never shrunk (they usually hold the instructions). The BOS
    /// pack is never shrunk either.
    pub skip: usize,
    pub opts: ShrinkOptions,
}
impl Default for Shrink {
    fn default() -> Self {
        Shrink {
            ratio: None,
            auto: true,
            skip: 1,
            opts: ShrinkOptions::default(),
        }
    }
}
/// Lowest fraction of a pack's rows that is ever asked for.
pub const MIN_SHRINK_RATIO: f32 = 0.1;

/// What [`answer`] found.
pub struct Answer {
    /// Per shrunk pack: its index, tokens before, tokens after.
    pub shrunk: Vec<(usize, usize, usize)>,
    pub text: String,
    pub report: ComposeReport,
    /// Per text: was its pack already on the shelf?
    pub shelved: Vec<bool>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// Shelve every text as a pack and, when there is a `question`, compose them in the order given,
/// then write up to `max_new` tokens greedily. `log` receives human-readable progress lines.
#[allow(clippy::too_many_arguments)]
pub fn answer(
    model: &Model,
    tok: &Tokenizer,
    stop_ids: &[usize],
    shelf: &Shelf,
    texts: &[String],
    question: Option<&str>,
    opts: &ComposeOptions,
    max_new: usize,
    check: bool,
    shrink: Option<&Shrink>,
    log: &mut dyn FnMut(String),
) -> Result<Answer, String> {
    let ctx = model.spec.context as usize;
    // the texts are documents and a question from outside: control tokens written in them stay text
    let ids = |t: &str, bos: bool| -> Vec<usize> {
        tok.encode_plain(t, bos)
            .into_iter()
            .map(|t| t as usize)
            .collect()
    };
    let mut segs: Vec<Vec<usize>> = Vec::new();
    let head = ids("", true);
    if !head.is_empty() {
        segs.push(head);
    }
    for t in texts {
        let v = ids(t, false);
        if v.is_empty() {
            return Err("a text became no tokens".into());
        }
        segs.push(v);
    }
    let tail_len = question.map_or(0, |q| ids(q, false).len());
    let orig_total: usize = segs.iter().map(Vec::len).sum();
    if shrink.is_none() && orig_total + tail_len >= ctx {
        return Err(format!(
            "the texts and the question do not fit the {ctx}-token context window"
        ));
    }
    if segs.iter().any(|s| s.len() + tail_len >= ctx) {
        return Err(format!(
            "a text and the question do not fit the {ctx}-token context window"
        ));
    }
    let (mut states, mut shelved): (Vec<State>, Vec<bool>) = (Vec::new(), Vec::new());
    for (i, s) in segs.iter().enumerate() {
        let t0 = Instant::now();
        let (st, hit) = shelf.pack(model, s).map_err(|e| e.to_string())?;
        log(format!(
            "pack {i}: {} tokens, {} ({:.2}s)",
            s.len(),
            if hit {
                "from the shelf"
            } else {
                "computed and shelved"
            },
            t0.elapsed().as_secs_f64()
        ));
        states.push(st);
        shelved.push(hit);
    }
    let mut shrunk = Vec::new();
    if let (Some(sh), true) = (shrink, question.is_some()) {
        let first = (segs.len() - texts.len()) + sh.skip;
        let lens: Vec<usize> = states.iter().map(|s| s.tokens().len()).collect();
        let eligible: usize = lens.iter().skip(first).sum();
        // leave room for the answer: at most 256 tokens of it count against the window
        let need = (orig_total + tail_len + max_new.min(256)).saturating_sub(ctx.saturating_sub(1));
        let ratio = match sh.ratio {
            Some(r) if r.is_nan() || !(0.0..=1.0).contains(&r) => {
                return Err("the shrink ratio must be between 0 and 1".into());
            }
            Some(r) => Some(r.max(MIN_SHRINK_RATIO)),
            None if sh.auto && need > 0 => {
                if eligible <= need {
                    return Err(format!(
                        "the texts do not fit the {ctx}-token context window even with every shrinkable text removed"
                    ));
                }
                Some(((eligible - need) as f32 / eligible as f32).max(MIN_SHRINK_RATIO))
            }
            None => None,
        };
        if let Some(r) = ratio.filter(|r| *r < 1.0) {
            for i in first..states.len() {
                let t0 = Instant::now();
                let st = shrink_pack(model, &states[i], r, &sh.opts).map_err(|e| e.to_string())?;
                if st.tokens().len() < lens[i] {
                    log(format!(
                        "pack {i}: shrunk {} -> {} tokens ({:.2}s)",
                        lens[i],
                        st.tokens().len(),
                        t0.elapsed().as_secs_f64()
                    ));
                    shrunk.push((i, lens[i], st.tokens().len()));
                    states[i] = st;
                }
            }
        }
        let now: usize = states.iter().map(|s| s.tokens().len()).sum();
        if now + tail_len >= ctx {
            return Err(format!(
                "the texts and the question do not fit the {ctx}-token context window even shrunk"
            ));
        }
    }
    let mut out = Answer {
        shrunk,
        text: String::new(),
        report: ComposeReport::default(),
        shelved,
        prompt_tokens: states.iter().map(|s| s.tokens().len()).sum(),
        completion_tokens: 0,
    };
    let Some(q) = question else {
        return Ok(out);
    };
    let tail = ids(q, false);
    if tail.is_empty() {
        return Err("the question became no tokens".into());
    }
    out.prompt_tokens += tail.len();
    let refs: Vec<&State> = states.iter().collect();
    let t0 = Instant::now();
    let (mut st, logits, rep) = compose(model, &refs, &tail, opts).map_err(|e| e.to_string())?;
    log(format!(
        "composed in {:.2}s: {} rows copied, {} recomputed to repair {} join(s), {} fresh; {} rows did not go through the network",
        t0.elapsed().as_secs_f64(),
        rep.spliced,
        rep.refreshed,
        rep.joins.len(),
        rep.fresh,
        rep.saved()
    ));
    for (i, (n, d)) in rep.joins.iter().enumerate() {
        log(format!(
            "  join {}: {n} rows recomputed, last change {d:.4}",
            i + 1
        ));
    }
    out.report = rep;
    if check && orig_total + tail_len >= ctx {
        log("check: skipped, the whole text does not fit the context window".into());
    } else if check {
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
        log(format!(
            "check: full prefill {:.2}s; largest score difference {dt:.4}; same first token: {}",
            t1.elapsed().as_secs_f64(),
            argmax(&fl) == argmax(&logits)
        ));
    }
    let (mut gen_ids, mut cur) = (Vec::<u32>::new(), logits);
    for _ in 0..max_new {
        let next = argmax(&cur);
        if stop_ids.contains(&next) || st.pos >= ctx {
            break;
        }
        gen_ids.push(next as u32);
        cur = model.forward(&mut st, next).map_err(|e| e.to_string())?;
    }
    out.completion_tokens = gen_ids.len();
    out.text = tok.decode(&gen_ids);
    Ok(out)
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
        // F16 and compressed packs over the same folder: remade, then used
        let mut f16 = args(true, "0");
        f16.opts.push(("pack-kv".into(), Some("f16".into())));
        f16.opts.push(("pack-compress".into(), None));
        cmd(&f16).unwrap();
        cmd(&f16).unwrap();
        let mut q8 = args(true, "0");
        q8.opts.push(("pack-kv".into(), Some("q8".into())));
        assert!(cmd(&q8).is_err());
        let mut bad = args(true, "0");
        bad.opts.retain(|(k, _)| k != "pack");
        assert!(cmd(&bad).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
    #[test]
    fn texts_that_do_not_fit_are_shrunk_when_asked_and_refused_otherwise() {
        let d = tmpdir("packs-shrink");
        let m = d.join("m");
        std::fs::create_dir_all(&m).unwrap();
        write_model(&m, false);
        let files: Vec<_> = (0..3).map(|i| d.join(format!("t{i}.txt"))).collect();
        std::fs::write(&files[0], "Answer only from the texts below.").unwrap();
        std::fs::write(
            &files[1],
            "The harbour opens at six and ferries leave until dusk.",
        )
        .unwrap();
        std::fs::write(
            &files[2],
            "Tickets cost four coins and children travel free.",
        )
        .unwrap();
        let args = |extra: &[(&str, Option<&str>)]| {
            let mut v: Vec<String> = vec![m.display().to_string()];
            for (k, val) in [
                ("session", d.join("s").display().to_string()),
                ("question", "When does it open".into()),
                ("max-new", "2".into()),
            ] {
                v.push(format!("--{k}"));
                v.push(val);
            }
            for f in &files {
                v.push("--pack".into());
                v.push(f.display().to_string());
            }
            for (k, val) in extra {
                v.push(format!("--{k}"));
                if let Some(x) = val {
                    v.push((*x).to_string());
                }
            }
            Args::parse(&v)
        };
        // together they are longer than the 96-token window
        assert!(cmd(&args(&[])).is_err());
        cmd(&args(&[("shrink", Some("auto"))])).unwrap();
        cmd(&args(&[("shrink", Some("auto")), ("no-refit", None)])).unwrap();
        cmd(&args(&[("shrink", Some("0.05"))])).unwrap();
        cmd(&args(&[
            ("shrink", Some("0.5")),
            ("shrink-skip", Some("0")),
        ]))
        .unwrap();
        assert!(cmd(&args(&[("shrink", Some("7"))])).is_err());
        assert!(cmd(&args(&[("shrink", Some("abc"))])).is_err());
        assert!(
            cmd(&args(&[
                ("shrink", Some("auto")),
                ("shrink-window", Some("0"))
            ]))
            .is_err()
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}
