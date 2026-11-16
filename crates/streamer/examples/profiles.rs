//! `profiles` measures how concentrated expert routing is on real `.coli_usage` histories.
//!   cargo run --release -p expert-streamer --example profiles -- LABEL=FILE [LABEL=FILE ...] [--emit]
//! A history is counts only (no order), so this is popularity, never recency. Histories with the
//! same shape are scored against each other (rank on one, score on the other); byte-identical
//! histories count once. `--emit` prints the `SKEW_SAMPLES` block for `src/skew.rs`.
use expert_streamer::skew::*;
use expert_streamer::trace::{Usage, read_usage_file};

struct Entry {
    label: String,
    aggregate: bool,
    usage: Usage,
}

fn pct(x: f64) -> String {
    format!("{:5.1}%", 100.0 * x)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let emit = args.iter().any(|a| a == "--emit");
    let mut entries: Vec<Entry> = Vec::new();
    for a in args.iter().filter(|a| !a.starts_with("--")) {
        let Some((label, path)) = a.split_once('=') else {
            eprintln!("error: expected LABEL=FILE, got `{a}`");
            std::process::exit(1);
        };
        let usage = match read_usage_file(path) {
            Ok(u) => u,
            Err(e) => {
                eprintln!("error: {path}: {e}");
                std::process::exit(1);
            }
        };
        if let Some(same) = entries.iter().find(|e| e.usage == usage) {
            eprintln!(
                "note: `{label}` is identical to `{}`; counted once",
                same.label
            );
            continue;
        }
        entries.push(Entry {
            label: label.to_string(),
            aggregate: path.ends_with(".coli_usage"),
            usage,
        });
    }
    if entries.is_empty() {
        eprintln!("usage: profiles LABEL=FILE [LABEL=FILE ...] [--emit]");
        std::process::exit(1);
    }

    println!(
        "## Concentration (best-case fixed set of the hottest experts, on the history itself)\n"
    );
    println!(
        "| history | layers x experts | uses | never used | Gini | effective experts | top 1% | 5% | 10% | 25% | 50% | experts for 50% / 80% / 90% / 95% of uses |"
    );
    println!("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|");
    for e in &entries {
        let s = analyse(&e.usage);
        println!(
            "| {} | {}x{} | {} | {} | {:.2} | {:.0} ({}) | {} | {} | {} | {} | {} | {} / {} / {} / {} |",
            e.label,
            e.usage.n_layers,
            e.usage.n_experts,
            s.selections,
            pct(s.experts_never_used as f64 / s.experts_total.max(1) as f64),
            s.gini,
            s.effective_experts,
            pct(s.effective_experts / s.experts_total.max(1) as f64).trim(),
            pct(s.coverage[0]),
            pct(s.coverage[1]),
            pct(s.coverage[2]),
            pct(s.coverage[3]),
            pct(s.coverage[4]),
            pct(s.need[0]).trim(),
            pct(s.need[1]).trim(),
            pct(s.need[2]).trim(),
            pct(s.need[3]).trim(),
        );
    }

    println!("\n## Per layer: does a per-layer budget beat one global ranking?\n");
    println!(
        "A global ranking is optimal for plain coverage, so the useful number is what an EQUAL split per layer would lose.\n"
    );
    println!(
        "| history | layer top-10% share (min / median / max) | global 10% | equal per layer 10% | global 25% | equal per layer 25% |"
    );
    println!("|---|---|---:|---:|---:|---:|");
    for e in &entries {
        let (lo, mid, hi) = per_layer_top10(&e.usage).unwrap_or((0.0, 0.0, 0.0));
        println!(
            "| {} | {} / {} / {} | {} | {} | {} | {} |",
            e.label,
            pct(lo).trim(),
            pct(mid).trim(),
            pct(hi).trim(),
            pct(coverage_at(&e.usage, 0.10)),
            pct(equal_per_layer_coverage(&e.usage, 0.10)),
            pct(coverage_at(&e.usage, 0.25)),
            pct(equal_per_layer_coverage(&e.usage, 0.25)),
        );
    }

    println!("\n## Held out: rank on one history, score on another of the same model shape\n");
    println!("| ranked on | scored on | top 1% | 5% | 10% | 25% | 50% |");
    println!("|---|---|---:|---:|---:|---:|---:|");
    let mut held: Vec<Option<([f64; 5], [f64; 5])>> = vec![None; entries.len()];
    for (i, train) in entries.iter().enumerate() {
        for (j, test) in entries.iter().enumerate() {
            if i == j {
                continue;
            }
            let mut row = [0.0; 5];
            let mut ok = true;
            for (r, f) in row.iter_mut().zip(FRACTIONS) {
                match holdout_coverage(&train.usage, &test.usage, f) {
                    Some(c) => *r = c,
                    None => ok = false,
                }
            }
            if !ok {
                continue;
            }
            println!(
                "| {} | {} | {} | {} | {} | {} | {} |",
                train.label,
                test.label,
                pct(row[0]),
                pct(row[1]),
                pct(row[2]),
                pct(row[3]),
                pct(row[4])
            );
            let slot = held[i].get_or_insert((row, row));
            for (k, v) in row.iter().enumerate() {
                slot.0[k] = slot.0[k].min(*v);
                slot.1[k] = slot.1[k].max(*v);
            }
        }
    }

    println!("\n## Spread over the histories (own coverage, no hold-out)\n");
    println!("| pinned share of all experts | min | median | max |");
    println!("|---:|---:|---:|---:|");
    for (k, f) in FRACTIONS.iter().enumerate() {
        let mut v: Vec<f64> = entries
            .iter()
            .map(|e| analyse(&e.usage).coverage[k])
            .collect();
        v.sort_by(f64::total_cmp);
        println!(
            "| {:.0}% | {} | {} | {} |",
            100.0 * f,
            pct(v[0]).trim(),
            pct(v[v.len() / 2]).trim(),
            pct(v[v.len() - 1]).trim()
        );
    }

    println!("\n## Range from SKEW_SAMPLES (what other crates can use)\n");
    println!(
        "| pinned share of all experts | own history: min / median / max | held out on another history: min / median / max |"
    );
    println!("|---:|---|---|");
    for f in [0.05, 0.10, 0.25, 0.50] {
        let (a, b, c) = pinned_hit_range(f, false);
        let (d, e, g) = pinned_hit_range(f, true);
        println!(
            "| {:.0}% | {} / {} / {} | {} / {} / {} |",
            100.0 * f,
            pct(a).trim(),
            pct(b).trim(),
            pct(c).trim(),
            pct(d).trim(),
            pct(e).trim(),
            pct(g).trim()
        );
    }

    if emit {
        println!("\n// ---- paste between the GENERATED-SAMPLES markers in src/skew.rs ----");
        for (e, h) in entries.iter().zip(&held) {
            let s = analyse(&e.usage);
            println!("    SkewSample {{");
            println!("        name: {:?},", e.label);
            println!("        aggregate: {},", e.aggregate);
            println!("        own: {:?},", round4(&s.coverage));
            match h {
                Some((lo, hi)) => println!(
                    "        held_out: Some(({:?}, {:?})),",
                    round4(lo),
                    round4(hi)
                ),
                None => println!("        held_out: None,"),
            }
            println!("    }},");
        }
    }
}

fn round4(v: &[f64; 5]) -> [f64; 5] {
    let mut o = [0.0; 5];
    for (x, y) in o.iter_mut().zip(v) {
        *x = (y * 10000.0).round() / 10000.0;
    }
    o
}
