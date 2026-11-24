//! `corbul steer train|apply|lens`: activation steering and a logit lens.
//!
//! * `train MODEL --positive FILE --negative FILE --out FILE.cvec`: reads one prompt per line from each
//!   file, runs the model, and keeps for every layer the direction that separates the two sets (`--method
//!   mean-diff` or `pca`). The weights are not touched; the vector is a small file stamped with the model's
//!   shape fingerprint (not its weight values: two fine-tunes of one base accept each other's vectors).
//! * `apply MODEL --vector FILE.cvec --strength S --prompt TEXT`: generates while adding the vector to the
//!   residual stream (`--layers A-B` limits where). Strength 0 is exactly plain generation; start small
//!   (±0.5 to ±2 in units of the vector's norm) because large values make text incoherent.
//! * `lens MODEL --prompt TEXT`: for each layer, what the model would predict if it stopped there, and the
//!   layer from which its final answer stays fixed.
//!
//! Not available for state-space/hybrid models, with structured output, speculation, receipts or sessions.
//! Nothing here was checked on a real checkpoint; the tests use tiny synthetic models.
use crate::args::Args;
use crate::cmds;
use crate::template::Message;
use companion_engine::model::{GenerateOptions, Sampler, StopReason};
use companion_engine::steer::{self, ControlVector, LensOptions, LensPositions, Method, Profile, Steering};
use std::path::Path;
use std::sync::atomic::AtomicBool;

fn lines(path: &str) -> Result<Vec<String>, String> {
    let t = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let v: Vec<String> = t.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).map(String::from).collect();
    if v.is_empty() { return Err(format!("{path}: no prompts (one per line, `#` starts a comment)")); }
    Ok(v)
}

pub fn parse_range(s: &str, layers: usize) -> Result<std::ops::Range<usize>, String> {
    let bad = || format!("--layers `{s}`: use a range such as 8-20 (layers start at 0, the model has {layers})");
    let (a, b) = match s.split_once('-') { Some((a, b)) => (a.trim().parse::<usize>().map_err(|_| bad())?, b.trim().parse::<usize>().map_err(|_| bad())?), None => { let a = s.trim().parse::<usize>().map_err(|_| bad())?; (a, a) } };
    if a > b || b >= layers { return Err(bad()); }
    Ok(a..b + 1)
}

pub fn cmd(args: &Args) -> Result<(), String> {
    let sub = args.pos.first().map(String::as_str).unwrap_or("");
    let rest = Args { pos: args.pos.iter().skip(1).cloned().collect(), opts: args.opts.clone() };
    cmds::apply_global(&rest);
    match sub {
        "train" => train(&rest),
        "apply" => apply(&rest),
        "lens" => lens(&rest),
        _ => Err("steer: use `train MODEL --positive F --negative F --out V`, `apply MODEL --vector V --strength S --prompt T` or `lens MODEL --prompt T`".into()),
    }
}

fn train(a: &Args) -> Result<(), String> {
    let out = a.get("out").ok_or("steer train: --out FILE is required")?;
    let method = match a.get("method").unwrap_or("mean-diff") { "mean-diff" => Method::MeanDiff, "pca" => Method::Pca, m => return Err(format!("--method `{m}` is not mean-diff or pca")) };
    let (pos, neg) = (lines(a.get("positive").ok_or("steer train: --positive FILE is required")?)?, lines(a.get("negative").ok_or("steer train: --negative FILE is required")?)?);
    let (prep, cfg) = cmds::open_prepared(a)?;
    let model = cmds::build_model_with_threads(&prep, &cfg, a)?;
    let enc = |v: &[String]| -> Vec<Vec<usize>> { v.iter().map(|l| prep.tok.encode(l, true).into_iter().map(|i| i as usize).collect()).collect() };
    let name = a.get("name").map_or_else(|| Path::new(out).file_stem().map_or("vector".to_string(), |s| s.to_string_lossy().into_owned()), String::from);
    let cv = steer::train(&model, &enc(&pos), &enc(&neg), method, &name).map_err(|e| e.to_string())?;
    cv.save(Path::new(out)).map_err(|e| e.to_string())?;
    let norms = cv.norms();
    let top = norms.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |(i, _)| i);
    crate::outln!("wrote {out}: {} layers, {} positive and {} negative prompts, strongest at layer {top} (norm {:.2})", norms.len(), pos.len(), neg.len(), norms.get(top).copied().unwrap_or(0.0));
    if pos.len().min(neg.len()) < 8 { eprintln!("note: fewer than 8 prompts per side gives a noisy direction"); }
    Ok(())
}

fn apply(a: &Args) -> Result<(), String> {
    crate::fast::refuse(a, false, "steering")?;
    for f in ["lookahead", "self-spec"] { if a.flag(f) { return Err(format!("--{f} cannot be combined with steering")); } }
    if a.get("draft").is_some() || a.get("skip-layers").is_some() { return Err("speculative decoding cannot be combined with steering".into()); }
    let vf = a.get("vector").ok_or("steer apply: --vector FILE is required")?;
    let strength: f32 = a.num("strength")?.unwrap_or(1.0);
    if !strength.is_finite() || strength.abs() > 50.0 { return Err("--strength must be a number between -50 and 50".into()); }
    let prompt = match (a.get("prompt"), a.get("prompt-file")) {
        (Some(p), _) => p.to_string(),
        (None, Some(f)) => std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?,
        _ => a.pos.get(1).cloned().ok_or("steer apply: give --prompt TEXT")?,
    };
    let (prep, cfg) = cmds::open_prepared(a)?;
    let model = cmds::build_model_with_threads(&prep, &cfg, a)?;
    let cv = ControlVector::load(Path::new(vf)).map_err(|e| e.to_string())?;
    cv.check_model(&model).map_err(|e| format!("{vf}: {e}"))?;
    let nl = model.spec.layers as usize;
    let range = match a.get("layers") { Some(r) => parse_range(r, nl)?, None => 0..nl };
    let st = Steering::new(&model, &cv, Profile::uniform(nl, range, strength)).map_err(|e| e.to_string())?;
    let ids = cmds::prompt_ids(&prep, &[Message::new("user", &prompt)], a.flag("raw").then_some(prompt.as_str()))?;
    let p = crate::runner::sampler_params(a, 0.7)?;
    let mut sampler = Sampler::new(p.temperature, p.top_k, p.seed.unwrap_or(1)).with_top_p(p.top_p);
    let cancel = AtomicBool::new(false);
    let opts = GenerateOptions { max_tokens: a.num("max-tokens")?.unwrap_or(256), stop_tokens: prep.stop_ids.clone(), cancel: Some(&cancel) };
    let mut s = model.new_state_with(cfg.kv);
    let mut dec = prep.tok.decoder();
    dec.skip_special = true;
    let stats = steer::generate_steered(&model, &mut s, &ids, &mut sampler, &opts, Some(&st), &mut |t| { crate::out!("{}", dec.push(t as u32)); true }).map_err(|e| e.to_string())?;
    crate::outln!("{}", dec.finish());
    if !a.flag("quiet") {
        eprintln!("[steered by {vf} · strength {strength} · {} tokens · {}{}]", stats.generated, match stats.stop { StopReason::StopToken => "stop", StopReason::ContextFull => "context full", StopReason::Cancelled => "cancelled", _ => "length" },
            if st.is_neutral() { " · neutral: the vector has no effect here" } else { "" });
    }
    Ok(())
}

fn lens(a: &Args) -> Result<(), String> {
    let prompt = match (a.get("prompt"), a.get("prompt-file")) {
        (Some(p), _) => p.to_string(),
        (None, Some(f)) => std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?,
        _ => a.pos.get(1).cloned().ok_or("steer lens: give --prompt TEXT")?,
    };
    let (prep, cfg) = cmds::open_prepared(a)?;
    let model = cmds::build_model_with_threads(&prep, &cfg, a)?;
    let positions = match a.get("positions").unwrap_or("last") { "last" => LensPositions::Last, "all" => LensPositions::All, n => LensPositions::LastN(n.parse().map_err(|_| "--positions is last, all or a number".to_string())?) };
    let opts = LensOptions { top_k: a.num("top-k")?.unwrap_or(3), positions, attention: !a.flag("no-attention"), steering: None };
    let ids = cmds::prompt_ids(&prep, &[Message::new("user", &prompt)], a.flag("raw").then_some(prompt.as_str()))?;
    let mut st = model.new_state_with(cfg.kv);
    let rep = steer::lens(&model, &mut st, &ids, &opts).map_err(|e| e.to_string())?;
    let dec = |t: usize| prep.tok.decode(&[t as u32]).replace('\n', "\\n");
    if a.flag("json") {
        crate::outln!("{}", serde_json::to_string_pretty(&rep.to_json_with(Some(&dec))).unwrap_or_default());
        return Ok(());
    }
    for p in &rep.positions {
        crate::outln!("position {} ({:?}) → final answer {:?}; first right at layer {}, fixed from layer {} of {}", p.index, dec(p.token), dec(p.final_top1), p.first_hit_layer, p.commit_layer, rep.layers);
        for l in &p.layers {
            let top: Vec<String> = l.top.iter().map(|(t, pr)| format!("{:?} {:.0}%", dec(*t), pr * 100.0)).collect();
            crate::outln!("  layer {:>3}  {}   (final answer ranks {}, entropy {:.2})", l.layer, top.join(" · "), l.final_rank, l.entropy);
        }
    }
    crate::outln!("mean commit depth {:.1} of {} layers", rep.mean_commit_depth(), rep.layers);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;
    fn a(s: &str) -> Args { Args::parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>()) }

    #[test]
    fn layer_ranges() {
        assert_eq!(parse_range("2-4", 8).unwrap(), 2..5);
        assert_eq!(parse_range("3", 8).unwrap(), 3..4);
        assert!(parse_range("4-2", 8).is_err());
        assert!(parse_range("0-8", 8).is_err());
        assert!(parse_range("x", 8).is_err());
    }

    #[test]
    fn train_apply_and_lens_on_a_tiny_model() {
        let d = tmpdir("steer");
        write_model_ctx(&d, true, 96);
        let m = d.to_string_lossy().to_string();
        let (p, n, v) = (d.join("pos.txt"), d.join("neg.txt"), d.join("v.cvec"));
        std::fs::write(&p, "abc\nabcabc\nbca\n# comment\n").unwrap();
        std::fs::write(&n, "zzz\nzyx\nxyz\n").unwrap();
        for method in ["mean-diff", "pca"] {
            cmd(&a(&format!("train {m} --positive {} --negative {} --out {} --method {method}", p.display(), n.display(), v.display()))).unwrap();
            assert!(v.is_file());
        }
        cmd(&a(&format!("apply {m} --vector {} --strength 1 --prompt abc --max-tokens 4 --temperature 0", v.display()))).unwrap();
        cmd(&a(&format!("apply {m} --vector {} --strength 0 --prompt abc --max-tokens 4 --temperature 0 --layers 0-0", v.display()))).unwrap();
        cmd(&a(&format!("lens {m} --prompt abc --top-k 2"))).unwrap();
        cmd(&a(&format!("lens {m} --prompt abc --json --positions all"))).unwrap();
        assert!(cmd(&a(&format!("apply {m} --prompt abc"))).is_err());
        assert!(cmd(&a(&format!("apply {m} --vector {} --strength 99 --prompt abc", v.display()))).is_err());
        assert!(cmd(&a(&format!("apply {m} --vector {} --min-p 0.1 --prompt abc", v.display()))).is_err());
        assert!(cmd(&a("nope")).is_err());
        assert!(cmd(&a(&format!("train {m} --positive {} --out x", p.display()))).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }
}
