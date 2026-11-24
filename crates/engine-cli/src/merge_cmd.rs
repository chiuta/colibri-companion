//! `corbul merge CONFIG.json [--out FILE.safetensors] [--model PATH]…`: combine models weight by weight.
//!
//! The config is the engine's merge JSON (`method`: linear, slerp, task, ties, dare, passthrough;
//! `weights`, `base`, `rules`, `slices`, `dtype`, `seed` …) plus two keys read here: `"models"` (a list
//! of folders or files) and `"output"`. `--model` (repeatable) and `--out` replace them. `--dry-run`
//! prints the plan and the memory it needs and writes nothing. Only weights are written: copy
//! `config.json` and the tokenizer files yourself (a layer-stacking merge needs `num_hidden_layers`
//! changed). Merging unrelated models gives noise; check the result before trusting it.
use crate::args::Args;
use companion_engine::merge::{self, Input, MergeConfig};
use serde_json::Value;
use std::path::Path;

pub fn cmd(a: &Args) -> Result<(), String> {
    let cf = a.pos.first().ok_or("merge: give the CONFIG.json file")?;
    let text = std::fs::read_to_string(cf).map_err(|e| format!("{cf}: {e}"))?;
    let mut v: Value =
        serde_json::from_str(&text).map_err(|e| format!("{cf}: not valid JSON: {e}"))?;
    let o = v
        .as_object_mut()
        .ok_or_else(|| format!("{cf}: the configuration must be a JSON object"))?;
    let models: Vec<String> = match o.remove("models") {
        Some(Value::Array(l)) => l
            .into_iter()
            .map(|x| {
                x.as_str()
                    .map(String::from)
                    .ok_or("\"models\" must list paths")
            })
            .collect::<Result<_, _>>()?,
        None => vec![],
        _ => return Err("\"models\" must be a list of paths".into()),
    };
    let out_cfg = o
        .remove("output")
        .and_then(|x| x.as_str().map(String::from));
    let models = if a.all("model").is_empty() {
        models
    } else {
        a.all("model").into_iter().map(String::from).collect()
    };
    let out = a.get("out").map(String::from).or(out_cfg);
    if models.len() < 2 && !matches!(o.get("method").and_then(Value::as_str), Some("passthrough")) {
        return Err(
            "merge: list at least two models (\"models\" in the config, or --model A --model B)"
                .into(),
        );
    }
    if models.is_empty() {
        return Err("merge: no models".into());
    }
    let mut cfg = MergeConfig::from_json(&v.to_string()).map_err(|e| e.to_string())?;
    if a.flag("overwrite") {
        cfg.overwrite = true;
    }
    if let Some(b) = a.get("max-memory") {
        cfg.max_peak_bytes = Some(crate::args::parse_size(b)?);
    }
    let inputs: Vec<Input> = models
        .iter()
        .map(|m| Input::open(Path::new(m)).map_err(|e| format!("{m}: {e}")))
        .collect::<Result<_, _>>()?;
    let plan = merge::plan(&inputs, &cfg).map_err(|e| e.to_string())?;
    crate::outln!(
        "{} tensors, {:.2} GiB output ({:?}), peak about {:.2} GiB of working memory{}",
        plan.tensors.len(),
        plan.out_bytes as f64 / (1u64 << 30) as f64,
        plan.dtype,
        plan.peak_f32 as f64 * 4.0 / (1u64 << 30) as f64,
        plan.out_layers.map_or(String::new(), |n| format!(
            ", {n} layers (set num_hidden_layers to {n})"
        ))
    );
    if a.flag("dry-run") {
        for t in plan
            .tensors
            .iter()
            .take(if a.flag("verbose") { usize::MAX } else { 8 })
        {
            crate::outln!("  {}  {:?}  {}", t.name, t.shape, t.method);
        }
        return Ok(());
    }
    let out =
        out.ok_or("merge: --out FILE.safetensors (or \"output\" in the config) is required")?;
    let mut last = 0usize;
    let st = merge::merge(&inputs, &cfg, Path::new(&out), &mut |p| {
        if p.done * 10 / p.total.max(1) > last {
            last = p.done * 10 / p.total.max(1);
            eprintln!("merge: {}% ({} of {} tensors)", last * 10, p.done, p.total);
        }
    })
    .map_err(|e| e.to_string())?;
    crate::outln!(
        "wrote {out}: {} tensors, {:.2} MiB",
        st.tensors,
        st.bytes_written as f64 / 1048576.0
    );
    eprintln!(
        "note: only weights were written; copy config.json and the tokenizer files next to it before loading"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;
    fn a(s: &str) -> Args {
        Args::parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>())
    }

    #[test]
    fn merges_two_tiny_models() {
        let (d1, d2) = (tmpdir("mg1"), tmpdir("mg2"));
        write_model_ctx(&d1, true, 96);
        write_model_ctx(&d2, true, 96);
        let cfg = d1.join("merge.json");
        std::fs::write(
            &cfg,
            format!(
                "{{\"method\":\"linear\",\"dtype\":\"f32\",\"models\":[{:?},{:?}]}}",
                d1.display().to_string(),
                d2.display().to_string()
            ),
        )
        .unwrap();
        let dout = tmpdir("mg3");
        let out = dout.join("merged.safetensors");
        cmd(&a(&format!("{} --dry-run", cfg.display()))).unwrap();
        cmd(&a(&format!("{} --out {}", cfg.display(), out.display()))).unwrap();
        assert!(out.is_file());
        assert!(cmd(&a(&format!("{} --out {}", cfg.display(), out.display()))).is_err()); // exists, no --overwrite
        std::fs::write(&cfg, "{\"method\":\"linear\",\"models\":[\"x\"]}").unwrap();
        assert!(cmd(&a(&cfg.display().to_string())).is_err());
        std::fs::write(&cfg, "{\"bogus\":1,\"models\":[\"a\",\"b\"]}").unwrap();
        assert!(cmd(&a(&cfg.display().to_string())).is_err());
        assert!(cmd(&a("")).is_err());
        std::fs::remove_dir_all(d1).unwrap();
        std::fs::remove_dir_all(d2).unwrap();
        std::fs::remove_dir_all(dout).unwrap();
    }
}
