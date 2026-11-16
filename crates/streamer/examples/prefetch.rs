//! How much of the next layer's expert loads a co-occurrence prefetcher could have ready.
//!   cargo run --release -p expert-streamer --example prefetch -- --trace FILE --experts 256
//!       a REAL `ROUTE_TRACE=` capture: the only run that says anything about a real model
//!   cargo run --release -p expert-streamer --example prefetch -- --usage FILE [--k 8]
//!       sampled from a popularity history: layers are independent BY CONSTRUCTION, so the
//!       predictor can only match plain popularity here; shown to calibrate, not to conclude
//!   (no arguments) a synthetic trace with correlated layers, to check the code
use expert_streamer::prefetch::*;
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

fn main() {
    let tokens: usize = number("--tokens").unwrap_or(1500);
    let (label, ne, trace) = if let Some(f) = arg("--trace") {
        let ne: u32 = or_exit(number("--experts").ok_or("--trace needs --experts N".to_string()));
        let t = or_exit(read_trace_file(&f, ne));
        (format!("{f}: a REAL trace"), ne, t)
    } else if let Some(f) = arg("--usage") {
        let u = or_exit(read_usage_file(&f));
        let k: usize = number("--k").unwrap_or(8);
        let t = u.sample_trace(k, tokens, 42);
        (
            format!("{f}: SAMPLED from popularity, layers independent by construction"),
            u.n_experts,
            t,
        )
    } else {
        let t = correlated_trace(40, 256, 8, tokens, 0.7, 42);
        (
            "synthetic, layers correlated (70% fidelity)".to_string(),
            256,
            t,
        )
    };
    println!(
        "{label}\n{} tokens, trained on the first half, scored on the second",
        trace.len()
    );
    println!("{:>8} {:>10} {:>12}", "budget", "predicted", "popularity");
    for m in [1usize, 2, 4] {
        let s = evaluate(&trace, ne, 0.5, m);
        println!(
            "{:>7}x {:>9.1}% {:>11.1}%",
            m,
            100.0 * s.share(),
            100.0 * s.baseline()
        );
    }
    println!(
        "budget = multiple of the experts the layer really used; loads counted from layer 2 on"
    );
}
