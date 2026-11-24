//! Convert a Hugging Face folder to GGUF, or requantise a GGUF.
//!
//! ```text
//! cargo run --release --example quantize -- convert <hf_dir> <out.gguf> [RECIPE] [--imatrix FILE] [--set PATTERN=TYPE]… [--threads N] [--name NAME]
//! cargo run --release --example quantize -- requant <in.gguf> <out.gguf> [RECIPE] [--imatrix FILE] [--set PATTERN=TYPE]…
//! cargo run --release --example quantize -- plan <RECIPE> <tensor-name> <rows> <cols> <layers>
//! ```
//! `RECIPE` is one of F32 F16 BF16 Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 Q2_K Q3_K_M Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K (default Q4_K_M).
//! `--set "*attn_v.weight=Q8_0"` forces tensors matching a `*` pattern to a type.
use companion_engine::quant::DType;
use companion_engine::quantize::{
    ConvertOptions, Imatrix, Recipe, convert_hf_dir, dtype_name, parse_dtype, quantize,
};
use std::path::PathBuf;

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos: Vec<&str> = vec![];
    let (mut imatrix, mut sets, mut threads, mut name) = (None, vec![], None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--imatrix" => imatrix = it.next().cloned(),
            "--set" => sets.push(
                it.next()
                    .cloned()
                    .unwrap_or_else(|| die("--set needs PATTERN=TYPE")),
            ),
            "--threads" => threads = it.next().and_then(|v| v.parse::<usize>().ok()),
            "--name" => name = it.next().cloned(),
            s => pos.push(s),
        }
    }
    let recipe_of = |s: Option<&&str>| {
        let n = s.copied().unwrap_or("Q4_K_M");
        let mut r = Recipe::named(n).unwrap_or_else(|| {
            die(&format!(
                "unknown recipe {n}; try one of {:?}",
                Recipe::NAMES
            ))
        });
        for s in &sets {
            let (p, t) = s
                .split_once('=')
                .unwrap_or_else(|| die("--set needs PATTERN=TYPE"));
            r = r.with_override(
                p,
                parse_dtype(t).unwrap_or_else(|| die(&format!("unknown type {t}"))),
            );
        }
        r
    };
    let options = |r: Recipe| {
        let mut o = ConvertOptions::new(r);
        if let Some(p) = &imatrix {
            o.imatrix = Some(
                Imatrix::load(&PathBuf::from(p)).unwrap_or_else(|e| die(&format!("imatrix: {e}"))),
            );
        }
        if let Some(t) = threads {
            o.threads = t.max(1);
        }
        o.name = name.clone();
        o
    };
    match pos.first().copied() {
        Some(cmd @ ("convert" | "requant")) if pos.len() >= 3 => {
            let o = options(recipe_of(pos.get(3)));
            let (src, dst) = (PathBuf::from(pos[1]), PathBuf::from(pos[2]));
            let t = std::time::Instant::now();
            let rep = if cmd == "convert" {
                convert_hf_dir(&src, &dst, &o)
            } else {
                quantize(&src, &dst, &o)
            }
            .unwrap_or_else(|e| die(&format!("error: {e}")));
            for w in &rep.warnings {
                eprintln!("warning: {w}");
            }
            let mut count: Vec<(String, usize)> = vec![];
            for (_, d) in &rep.types {
                let n = dtype_name(*d);
                match count.iter_mut().find(|c| c.0 == n) {
                    Some(c) => c.1 += 1,
                    None => count.push((n, 1)),
                }
            }
            println!(
                "{} ({}): {} tensors, {:.1} MiB in {:.1}s — {}",
                dst.display(),
                rep.arch,
                rep.tensors,
                rep.output_bytes as f64 / 1048576.0,
                t.elapsed().as_secs_f64(),
                count
                    .iter()
                    .map(|(n, c)| format!("{c}×{n}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        Some("plan") if pos.len() == 6 => {
            let r = recipe_of(pos.get(1));
            let p = |i: usize| {
                pos[i]
                    .parse::<u64>()
                    .unwrap_or_else(|_| die("rows, cols and layers must be numbers"))
            };
            let t: DType = r.plan(pos[2], &[p(3), p(4)], p(5) as usize);
            println!("{} → {}", pos[2], dtype_name(t));
        }
        _ => die(
            "usage: quantize convert <hf_dir> <out.gguf> [RECIPE] | requant <in> <out> [RECIPE] | plan <RECIPE> <name> <rows> <cols> <layers>   [--imatrix F] [--set PAT=TYPE] [--threads N] [--name N]",
        ),
    }
}
