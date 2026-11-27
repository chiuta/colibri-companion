//! `corbul kvfit MODEL --text FILE --question Q`: measure what shrinking the cache costs, with and
//! without refitting the values that stay (see `crates/engine/src/kvfit.rs`).
//!
//! The text is prefilled; its last `--window` tokens vote for which rows to keep (SnapKV) and are
//! recorded as the probes of the fit. Three copies of the state then answer the question: the full
//! cache, the cache shrunk to `--budget` of its rows by plain eviction, and the same shrunk cache
//! with refitted values. Printed: the fit error on the probes, how far each shrunk cache's first
//! score vector is from the full one, whether the first token agrees, and how many of the first
//! `--max-new` greedy tokens agree. Nothing here says it will help on your model; it tells you.
//!
//! * `--budget R` (default 0.25) fraction of rows kept, `--window N` (default 16) probes and voters,
//!   `--ridge X` (default 0.01), `--max-new N` (default 32).
use crate::args::Args;
use crate::cmds;
use companion_engine::kvpolicy::{LayerBudgets, SnapKv};
use companion_engine::model::{Model, State};
use std::path::Path;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|x| x.0)
        .unwrap_or(0)
}

fn greedy(model: &Model, st: &mut State, mut logits: Vec<f32>, n: usize, ctx: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for _ in 0..n {
        let t = argmax(&logits);
        if st.pos + st.rope_gap() + 1 >= ctx {
            break;
        }
        out.push(t);
        match model.forward(st, t) {
            Ok(l) => logits = l,
            Err(_) => break,
        }
    }
    out
}

pub fn cmd(a: &Args) -> Result<(), String> {
    cmds::apply_global(a);
    let model_path = a.pos.first().ok_or("kvfit: give the model")?.clone();
    if crate::famrun::wants(Path::new(&model_path)) {
        return Err("kvfit needs a dense model".into());
    }
    let text_file = a.get("text").ok_or("kvfit: give --text FILE")?;
    let question = a.get("question").ok_or("kvfit: give --question TEXT")?;
    let md = std::fs::metadata(text_file).map_err(|e| format!("{text_file}: {e}"))?;
    if !md.is_file() || md.len() > (8 << 20) {
        return Err(format!("{text_file}: not a file, or larger than 8 MiB"));
    }
    let text =
        String::from_utf8(std::fs::read(text_file).map_err(|e| format!("{text_file}: {e}"))?)
            .map_err(|_| format!("{text_file}: not valid UTF-8 text"))?;
    let budget: f64 = a.num("budget")?.unwrap_or(0.25);
    let window: usize = a.num("window")?.unwrap_or(16);
    let ridge: f32 = a.num("ridge")?.unwrap_or(0.01);
    let max_new: usize = a.num("max-new")?.unwrap_or(32);
    if !(budget > 0.0 && budget < 1.0) || window == 0 || ridge.is_nan() || ridge < 0.0 {
        return Err(
            "kvfit: --budget must be between 0 and 1, --window at least 1, --ridge 0 or more"
                .into(),
        );
    }
    let (prep, cfg) = cmds::open_prepared(&Args {
        pos: vec![model_path],
        opts: a.opts.clone(),
    })?;
    let model = prep.build_model(&cfg)?;
    let ctx = model.spec.context as usize;
    let ids = |t: &str, bos: bool| -> Vec<usize> {
        prep.tok
            .encode_plain(t, bos)
            .into_iter()
            .map(|x| x as usize)
            .collect()
    };
    let (doc, tail) = (ids(&text, true), ids(question, false));
    if tail.is_empty() || doc.len() <= window + 1 || doc.len() + tail.len() + max_new >= ctx {
        return Err(format!(
            "kvfit: the text must be longer than --window and text + question + answer must fit {ctx} tokens"
        ));
    }
    let keep = ((doc.len() as f64 * budget).ceil() as usize).clamp(window + 1, doc.len() - 1);
    let w = window;
    let mut base = model.new_state();
    model
        .forward_batch(&mut base, &doc[..doc.len() - w])
        .map_err(|e| e.to_string())?;
    base.track_scores(true);
    base.record_queries(w);
    model
        .forward_batch(&mut base, &doc[doc.len() - w..])
        .map_err(|e| e.to_string())?;
    let snap = SnapKv { window: w, pool: 5 };
    // full
    let mut full = base.fork().map_err(|e| e.to_string())?;
    let fl = model
        .forward_batch(&mut full, &tail)
        .map_err(|e| e.to_string())?;
    let full_text = greedy(&model, &mut full, fl.clone(), max_new, ctx);
    let report = |name: &str, fit: bool| -> Result<(), String> {
        let mut st = base.fork().map_err(|e| e.to_string())?;
        let t0 = std::time::Instant::now();
        let c;
        let mut fe = None;
        if fit {
            let (cc, f) = st
                .apply_policy_fitted(&snap, keep, &LayerBudgets::Uniform, ridge)
                .map_err(|e| e.to_string())?;
            c = cc;
            fe = f;
        } else {
            c = st
                .apply_policy(&snap, keep, &LayerBudgets::Uniform)
                .map_err(|e| e.to_string())?;
        }
        let dt = t0.elapsed().as_secs_f64();
        let l = model
            .forward_batch(&mut st, &tail)
            .map_err(|e| e.to_string())?;
        let diff = l
            .iter()
            .zip(&fl)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let out = greedy(&model, &mut st, l.clone(), max_new, ctx);
        let same = out
            .iter()
            .zip(&full_text)
            .take_while(|(x, y)| x == y)
            .count();
        crate::outln!(
            "{name:<14} rows {}→{} ({:.0}% of the cache bytes), shrink {dt:.2}s, largest score difference {diff:.4}, same first token: {}, matching leading tokens: {same}/{}",
            c.before,
            c.after,
            100.0 * c.bytes_after as f64 / c.bytes_before.max(1) as f64,
            argmax(&l) == argmax(&fl),
            full_text.len()
        );
        if let Some(f) = fe {
            crate::outln!(
                "{:<14} fitted {} layers / {} heads; error on the probes {:.4} → {:.4}",
                "",
                f.layers,
                f.heads,
                f.error_before,
                f.error_after
            );
        }
        Ok(())
    };
    report("plain eviction", false)?;
    report("refitted", true)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;

    #[test]
    fn runs_end_to_end_and_refuses_bad_settings() {
        let d = tmpdir("kvfit");
        let m = d.join("m");
        std::fs::create_dir_all(&m).unwrap();
        write_model(&m, false);
        let f = d.join("t.txt");
        std::fs::write(&f, "The harbour opens at six and ferries leave.").unwrap();
        let args = |extra: &[(&str, &str)]| {
            let mut v: Vec<String> = vec![m.display().to_string()];
            for (k, val) in [
                ("text", f.display().to_string()),
                ("question", "When".into()),
                ("window", "3".into()),
                ("max-new", "2".into()),
            ]
            .into_iter()
            .chain(extra.iter().map(|(k, v)| (*k, v.to_string())))
            {
                v.push(format!("--{k}"));
                v.push(val);
            }
            Args::parse(&v)
        };
        let r = cmd(&args(&[]));
        assert!(
            r.is_ok() || r.as_ref().unwrap_err().contains("must fit"),
            "{r:?}"
        );
        assert!(cmd(&args(&[("budget", "1.5")])).is_err());
        assert!(cmd(&args(&[("budget", "0")])).is_err());
        assert!(cmd(&args(&[("ridge", "-1")])).is_err());
        assert!(cmd(&args(&[("window", "0")])).is_err());
        let mut no_text = args(&[]);
        no_text.opts.retain(|(k, _)| k != "text");
        assert!(cmd(&no_text).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
