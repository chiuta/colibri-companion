//! Cost model for verifying k drafted tokens per pass (loads per accepted token). Not an engine.
//!   cargo run --release -p expert-streamer --example speculative \
//!       [-- --usage FILE [--k 8] | --trace FILE --experts N] [--cache PCT] [--budget 1,1.5,2]
//! `--budget` lists multiples of the experts a single token uses per layer: each layer then loads
//! at most that many experts per pass (MoE-Spec style) and the share of routed slots dropped is
//! printed as a quality-risk proxy. Real quality impact is NOT measured here.
//! Real overlap between consecutive tokens is only known from a real ROUTE_TRACE; a trace sampled
//! from a `.coli_usage` history overlaps only through popularity, which understates it.
use expert_streamer::policy::{Step, synthetic_trace};
use expert_streamer::prefetch::correlated_trace;
use expert_streamer::speculative::{budget_for, experts_per_layer, locality_trace, verify_pass};
use expert_streamer::trace::*;

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
}

fn or_exit<T>(r: Result<T, String>) -> T {
    r.unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1)
    })
}

fn number<T: std::str::FromStr>(name: &str) -> Option<T> {
    arg(name).map(|v| {
        or_exit(
            v.parse()
                .map_err(|_| format!("{name} needs a number, got `{v}`")),
        )
    })
}

/// One input: label, experts in all, experts per layer, the trace.
type Input = (String, usize, u32, Vec<Step>);

fn report(input: &Input, pct: usize, mults: &[f64]) {
    let (label, universe, ne, trace) = input;
    let cap = (universe / 100 * pct + universe % 100 * pct / 100).max(1);
    let per = experts_per_layer(trace, *ne);
    println!(
        "\n{label}: {} tokens, {universe} experts, up to {per} per layer per token. A cost model, not an engine.",
        trace.len()
    );
    for (name, c) in [("no cache", 0usize), ("cache", cap)] {
        println!(
            "\n{name}{}: expert loads per accepted token (and dropped share of routed slots)",
            if c > 0 {
                format!(" {pct}%")
            } else {
                String::new()
            }
        );
        println!(
            "{:>3} {:>10} {:>8} {:>8} {:>8} {:>9}",
            "k", "budget", "a=50%", "a=70%", "a=90%", "dropped"
        );
        for k in [1usize, 2, 4, 8] {
            let mut rows: Vec<(String, Option<usize>)> = vec![("unbounded".into(), None)];
            if k > 1 {
                for m in mults {
                    if let Some(b) = budget_for(*m, per) {
                        rows.push((format!("{m}x = {b}"), Some(b)));
                    }
                }
            }
            for (name, b) in rows {
                // one replay per (k, cache, budget): the acceptance only divides the result
                let v = verify_pass(trace, k, c, *ne, b);
                if v.is_no_data() {
                    println!("{k:>3} {name:>10}  no data (trace shorter than {k} tokens)");
                    continue;
                }
                let cells: Vec<String> = [0.5, 0.7, 0.9]
                    .iter()
                    .map(|a| format!("{:>8.1}", v.per_token(k, *a)))
                    .collect();
                println!(
                    "{k:>3} {name:>10} {} {:>8.1}%",
                    cells.join(" "),
                    100.0 * v.dropped_share
                );
            }
        }
    }
}

fn main() {
    let pct: usize = number("--cache").unwrap_or(10);
    if pct > 100 {
        or_exit::<()>(Err(format!("--cache is a percentage, 0..=100, got {pct}")));
    }
    let mults: Vec<f64> = match arg("--budget") {
        Some(v) => v
            .split(',')
            .map(|x| {
                or_exit(
                    x.trim()
                        .parse::<f64>()
                        .ok()
                        .filter(|m| m.is_finite() && *m > 0.0)
                        .ok_or(format!("--budget needs positive numbers, got `{x}`")),
                )
            })
            .collect(),
        None => vec![1.0, 1.5, 2.0],
    };
    let mut inputs: Vec<Input> = Vec::new();
    if let Some(f) = arg("--usage") {
        let u = or_exit(read_usage_file(&f));
        let k: usize = number("--k").unwrap_or(8);
        let t = or_exit(u.try_sample_trace(k, 800, 42));
        inputs.push((
            format!("{f}: sampled, popularity only, no overlap by design"),
            u.n_layers as usize * u.n_experts as usize,
            u.n_experts,
            t,
        ));
    } else if let Some(f) = arg("--trace") {
        let ne: u32 = or_exit(number("--experts").ok_or("--trace needs --experts N".to_string()));
        let t = or_exit(read_trace_file(&f, ne));
        let uni = t.iter().flatten().max().map_or(0, |m| *m as usize + 1);
        inputs.push((format!("{f}: a real trace"), uni, ne, t));
    } else {
        inputs.push((
            "synthetic trace, 1 layer, no overlap".into(),
            256,
            256,
            synthetic_trace(256, 8, 4000, 42),
        ));
        inputs.push((
            "correlated across layers (70%), tokens independent".into(),
            40 * 256,
            256,
            correlated_trace(40, 256, 8, 800, 0.7, 42),
        ));
        inputs.push((
            "drifting pool, tokens overlap (a test of the code, not of any model)".into(),
            40 * 256,
            256,
            locality_trace(40, 256, 8, 800, 16, 42),
        ));
    }
    for i in &inputs {
        report(i, pct, &mults);
    }
}
