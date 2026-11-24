//! `corbul autoquant` and `corbul quantize`.
//!
//! `autoquant MODEL --budget B` decides, tensor by tensor, which GGUF format each weight matrix gets so
//! that the file fits the budget with as little damage as possible, and writes the decision as a recipe
//! (`--recipe FILE`) and a report (`--report FILE`). `quantize IN OUT.gguf --type Q4_K_M | --recipe FILE`
//! then writes the GGUF.
//!
//! Honest limits of `autoquant`:
//! * It uses the weight-error proxy only (how far the quantised weights are from the originals, relative
//!   to their size, optionally weighted by an importance matrix). It never runs the model, so it cannot
//!   see that one layer matters more to the output than its weights suggest. The report says so. A
//!   measured (KL divergence) run needs the model's activations and is a library feature, not wired here.
//! * Large matrices are judged on their first `--sample-rows` rows (default 256) to keep it fast.
//! * Input must be a Hugging Face folder of a dense or mixture-of-experts decoder. Expert tensors and
//!   everything it cannot name are kept at their stored size and are not tuned.
//! * The recipe file is this program's own small JSON: `{"base": "Q4_K_M", "overrides": {"blk.0.attn_q.weight": "Q6_K"}}`.
use crate::args::{parse_size, Args};
use companion_engine::autoquant::{self, AutoOpts, Budget, Control, Group, Method, Probe, ProxyData};
use companion_engine::quant::DType;
use companion_engine::loader::{SafeTensors, TensorSource};
use companion_engine::quantize::{self, dtype_name, parse_dtype, ConvertOptions, Imatrix, Recipe};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

/// Hugging Face tensor name → GGUF name for the weight matrices of a decoder layer, or `None`.
pub fn gguf_name(hf: &str) -> Option<String> {
    let mut n = hf;
    while let Some(r) = n.strip_prefix("language_model.") { n = r; }
    match n {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        _ => {}
    }
    let rest = n.strip_prefix("model.layers.")?;
    let (i, rest) = rest.split_once('.')?;
    let i: usize = i.parse().ok()?;
    let part = match rest {
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" | "self_attn.dense.weight" => "attn_output.weight",
        "self_attn.qkv_proj.weight" => "attn_qkv.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" | "mlp.gate_up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("blk.{i}.{part}"))
}

struct Entry { hf: String, rows: usize, cols: usize }

/// The weight-error probe over a Hugging Face folder.
pub struct FolderProbe { st: SafeTensors, groups: Vec<Group>, entries: Vec<Entry>, fixed: u64, sample_rows: usize, imatrix: Option<Imatrix>, pub kept: usize }

impl FolderProbe {
    pub fn open(dir: &Path, sample_rows: usize, imatrix: Option<Imatrix>, candidates: &[DType]) -> Result<FolderProbe, String> {
        if !dir.join("config.json").is_file() { return Err("autoquant reads a Hugging Face model folder (config.json + *.safetensors)".into()); }
        let st = SafeTensors::open(dir).map_err(|e| e.to_string())?;
        let mut infos = st.infos();
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        let (mut groups, mut entries, mut fixed, mut kept) = (vec![], vec![], 0u64, 0usize);
        for t in &infos {
            let size = t.bytes().unwrap_or(0);
            match gguf_name(&t.name) {
                Some(g) if t.dims.len() == 2 && t.dims[0] > 0 && t.dims[1] > 0 && candidates.iter().any(|c| c.bytes_for(t.dims[1]).is_some()) => {
                    let layer = g.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|x| x.parse().ok());
                    groups.push(Group { name: g.clone(), tensors: vec![g], layer, params: t.dims[0] * t.dims[1], cols: t.dims[1] as usize });
                    entries.push(Entry { hf: t.name.clone(), rows: t.dims[0] as usize, cols: t.dims[1] as usize });
                }
                _ => { fixed += size; if t.dims.len() >= 2 { kept += 1; } }
            }
        }
        if groups.is_empty() { return Err("no weight matrix with a known name has rows that any of the candidate formats can hold (block formats need rows of 32 or 256 values)".into()); }
        Ok(FolderProbe { st, groups, entries, fixed, sample_rows: sample_rows.max(1), imatrix, kept })
    }
}

impl Probe for FolderProbe {
    fn groups(&self) -> Vec<Group> { self.groups.clone() }
    fn fixed_bytes(&self) -> u64 { self.fixed }
    fn damage(&mut self, _g: usize, _t: DType) -> Result<f64, String> { Err("this probe only supports the weight-error proxy".into()) }
    fn joint_damage(&mut self, _a: &[(usize, DType)]) -> Result<f64, String> { Err("this probe only supports the weight-error proxy".into()) }
    fn proxy_data(&self, group: usize) -> Option<ProxyData> {
        let e = self.entries.get(group)?;
        let mut data = self.st.read(&e.hf).ok()?;
        if data.len() != e.rows * e.cols { return None; }
        // judge big matrices on evenly spaced rows
        if e.rows > self.sample_rows {
            let keep: Vec<usize> = (0..self.sample_rows).map(|i| i * e.rows / self.sample_rows).collect();
            let mut out = Vec::with_capacity(keep.len() * e.cols);
            for r in keep { out.extend_from_slice(&data[r * e.cols..(r + 1) * e.cols]); }
            data = out;
        }
        let imp = self.imatrix.as_ref().and_then(|m| m.for_cols(&self.groups[group].name, e.cols));
        Some(ProxyData { data, cols: e.cols, imp, scale: 1.0 })
    }
}

/// `4.5bpw`, `Q4_K_M`-style format name, or a size such as `6g`.
pub fn parse_budget(s: &str) -> Result<Budget, String> {
    let l = s.trim().to_ascii_lowercase();
    if let Some(n) = l.strip_suffix("bpw") { return n.trim().parse::<f64>().ok().filter(|v| *v > 0.5 && *v < 33.0).map(Budget::Bpw).ok_or_else(|| format!("--budget `{s}`: bits per weight must be between 0.5 and 33")); }
    if let Some(t) = parse_dtype(s).or_else(|| Recipe::named(s).map(|r| r.default)) { return Ok(Budget::Uniform(t)); }
    parse_size(s).map(Budget::Bytes).map_err(|_| format!("--budget `{s}`: use a size (6g), bits per weight (4.5bpw) or a format (Q4_K)"))
}

pub fn parse_candidates(s: Option<&str>) -> Result<Vec<DType>, String> {
    match s {
        None => Ok(vec![DType::Q3K, DType::Q4K, DType::Q5K, DType::Q6K, DType::Q8_0]),
        Some(l) => l.split(',').map(str::trim).filter(|x| !x.is_empty()).map(|x| parse_dtype(x).filter(|d| d.decodable() && d.geometry().is_some_and(|g| g.0 > 1)).ok_or_else(|| format!("--candidates: `{x}` is not a quantised format (try Q3_K,Q4_K,Q5_K,Q6_K,Q8_0)"))).collect(),
    }
}

pub fn recipe_to_json(base: &str, r: &Recipe) -> Value {
    let mut o = serde_json::Map::new();
    for (p, t) in &r.overrides { o.insert(p.clone(), json!(dtype_name(*t))); }
    json!({"format": "corbul-recipe-1", "base": base, "overrides": o})
}

pub fn recipe_from_json(text: &str) -> Result<Recipe, String> {
    let j: Value = serde_json::from_str(text).map_err(|e| format!("recipe: {e}"))?;
    let base = j.get("base").and_then(Value::as_str).ok_or("recipe: `base` (for example \"Q4_K_M\") is missing")?;
    let mut r = Recipe::named(base).ok_or_else(|| format!("recipe: unknown base `{base}` (one of {})", Recipe::NAMES.join(", ")))?;
    if let Some(o) = j.get("overrides") {
        for (p, t) in o.as_object().ok_or("recipe: `overrides` must be an object")? {
            let d = t.as_str().and_then(parse_dtype).ok_or_else(|| format!("recipe: `{p}` has an unknown format"))?;
            r = r.with_override(p, d);
        }
    }
    Ok(r)
}

pub fn autoquant(args: &Args) -> Result<(), String> {
    let model = args.pos.first().ok_or("autoquant: give the model folder")?;
    let budget = parse_budget(args.get("budget").ok_or("autoquant: --budget is required (for example 4.5bpw, 6g or Q4_K)")?)?;
    let candidates = parse_candidates(args.get("candidates"))?;
    let base = args.get("base").unwrap_or("Q4_K_M");
    if Recipe::named(base).is_none() { return Err(format!("--base `{base}` is not one of {}", Recipe::NAMES.join(", "))); }
    let imatrix = args.get("imatrix").map(|p| Imatrix::load(Path::new(p)).map_err(|e| format!("--imatrix {p}: {e}"))).transpose()?;
    let mut probe = FolderProbe::open(Path::new(model), args.num("sample-rows")?.unwrap_or(256), imatrix, &candidates)?;
    let threads: usize = args.num("threads")?.unwrap_or(4);
    let opts = AutoOpts { threads, verify: false, ..AutoOpts::default() };
    eprintln!("autoquant: {} weight matrices, estimating damage from the weights alone (the model is not run)", probe.groups.len());
    let out = autoquant::auto_quant(&mut probe, &candidates, Method::Proxy, budget, &opts, &mut Control::none()).map_err(|e| e.to_string())?;
    let recipe = out.plan.to_recipe(base, &probe.groups).map_err(|e| e.to_string())?;
    let mut report = out.to_markdown();
    report += &format!("\n## Limits\n\nDamage is estimated from the weights alone (relative rounding error of the first {} rows of each matrix{}), not measured on the model's output. {} tensors that are not plain weight matrices (norms, routers, experts) keep their stored size and are not tuned.\n",
        args.num::<usize>("sample-rows")?.unwrap_or(256), if args.get("imatrix").is_some() { ", weighted by the importance matrix" } else { "" }, probe.kept);
    match args.get("report") { Some(f) => std::fs::write(f, &report).map_err(|e| format!("--report {f}: {e}"))?, None if !args.flag("json") => crate::outln!("{report}"), None => {} }
    let rj = recipe_to_json(base, &recipe);
    match args.get("recipe") {
        Some(f) => { std::fs::write(f, serde_json::to_string_pretty(&rj).unwrap_or_default()).map_err(|e| format!("--recipe {f}: {e}"))?; eprintln!("recipe written to {f}; next: corbul quantize {model} OUT.gguf --recipe {f}"); }
        None if args.flag("json") => crate::outln!("{}", serde_json::to_string_pretty(&json!({"recipe": rj, "report": out.to_json()})).unwrap_or_default()),
        None => eprintln!("(add --recipe FILE to keep the plan)"),
    }
    eprintln!("plan: {:.2} MiB at {:.2} bits per weight on the grouped tensors", out.plan.bytes as f64 / 1048576.0, out.plan.bpw);
    Ok(())
}

pub fn quantize_cmd(args: &Args) -> Result<(), String> {
    let (inp, outp) = (args.pos.first().ok_or("quantize: give INPUT and OUTPUT.gguf")?, args.pos.get(1).ok_or("quantize: give the OUTPUT.gguf path")?);
    let recipe = match (args.get("recipe"), args.get("type")) {
        (Some(_), Some(_)) => return Err("use --type or --recipe, not both".into()),
        (Some(f), None) => recipe_from_json(&std::fs::read_to_string(f).map_err(|e| format!("--recipe {f}: {e}"))?)?,
        (None, t) => { let t = t.unwrap_or("Q4_K_M"); Recipe::named(t).ok_or_else(|| format!("--type `{t}` is not one of {}", Recipe::NAMES.join(", ")))? }
    };
    let mut o = ConvertOptions::new(recipe);
    if let Some(t) = args.num("threads")? { o.threads = t; }
    o.name = args.get("name").map(String::from);
    if let Some(p) = args.get("imatrix") { o.imatrix = Some(Imatrix::load(Path::new(p)).map_err(|e| format!("--imatrix {p}: {e}"))?); }
    let (inp, outp) = (Path::new(inp), Path::new(outp));
    if inp == outp { return Err("the output must not overwrite the input".into()); }
    let rep = if inp.is_dir() { quantize::convert_hf_dir(inp, outp, &o) } else { quantize::quantize(inp, outp, &o) }.map_err(|e| e.to_string())?;
    let mut by: HashMap<String, usize> = HashMap::new();
    for (_, t) in &rep.types { *by.entry(dtype_name(*t)).or_default() += 1; }
    let mut kinds: Vec<_> = by.into_iter().collect();
    kinds.sort();
    crate::outln!("wrote {} ({} tensors, {:.2} MiB, architecture {})", outp.display(), rep.tensors, rep.output_bytes as f64 / 1048576.0, rep.arch);
    crate::outln!("formats       {}", kinds.iter().map(|(k, n)| format!("{k}×{n}")).collect::<Vec<_>>().join("  "));
    for w in &rep.warnings { eprintln!("warning: {w}"); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;
    fn a(s: &str) -> Args { Args::parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>()) }

    #[test]
    fn names_map_to_gguf() {
        assert_eq!(gguf_name("model.layers.3.self_attn.q_proj.weight").unwrap(), "blk.3.attn_q.weight");
        assert_eq!(gguf_name("model.embed_tokens.weight").unwrap(), "token_embd.weight");
        assert!(gguf_name("model.layers.3.input_layernorm.weight").is_none());
        assert!(gguf_name("model.layers.3.mlp.experts.0.up_proj.weight").is_none());
    }

    #[test]
    fn budgets_parse() {
        assert_eq!(parse_budget("4.5bpw").unwrap(), Budget::Bpw(4.5));
        assert_eq!(parse_budget("Q4_K").unwrap(), Budget::Uniform(DType::Q4K));
        assert_eq!(parse_budget("1m").unwrap(), Budget::Bytes(1 << 20));
        assert!(parse_budget("lots").is_err());
        assert!(parse_candidates(Some("Q4_K,F16")).is_err());
        assert_eq!(parse_candidates(Some("Q4_K, Q8_0")).unwrap(), vec![DType::Q4K, DType::Q8_0]);
    }

    #[test]
    fn recipes_round_trip() {
        let r = Recipe::named("Q4_K_M").unwrap().with_override("blk.0.attn_q.weight", DType::Q6K);
        let back = recipe_from_json(&recipe_to_json("Q4_K_M", &r).to_string()).unwrap();
        assert_eq!(back.overrides, r.overrides);
        assert!(recipe_from_json("{}").is_err());
        assert!(recipe_from_json("{\"base\":\"nope\"}").is_err());
    }

    #[test]
    fn autoquant_then_quantize_a_tiny_model() {
        let d = tmpdir("aq");
        write_model_ctx(&d, true, 96);
        let m = d.to_string_lossy().to_string();
        let (rec, rep, out) = (d.join("r.json"), d.join("r.md"), d.join("out.gguf"));
        // tiny matrices have 32-wide rows: only formats with 32-wide blocks fit
        let e = autoquant(&a(&format!("{m} --budget 6bpw --candidates Q4_0,Q8_0 --recipe {} --report {}", rec.display(), rep.display()))).unwrap_err();
        assert!(e.contains("candidate formats"), "{e}");
        std::fs::write(&rec, recipe_to_json("Q8_0", &Recipe::named("Q8_0").unwrap().with_override("blk.0.*", DType::F16)).to_string()).unwrap();
        quantize_cmd(&a(&format!("{m} {} --recipe {}", out.display(), rec.display()))).unwrap();
        assert!(out.is_file());
        assert!(quantize_cmd(&a(&format!("{m} {} --type Q4_K_M --recipe {}", out.display(), rec.display()))).is_err());
        assert!(autoquant(&a(&format!("{m} --budget 0.1bpw"))).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }
}
