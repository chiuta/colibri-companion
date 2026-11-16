//! `headroom` compares cache policies and what is left to gain.
//!   cargo run --release -p expert-streamer --example headroom
//!       a SYNTHETIC trace (policy comparison only)
//!   cargo run --release -p expert-streamer --example headroom -- --usage FILE [--k 8] [--tokens 600]
//!       a trace sampled from a real `.coli_usage` history: real popularity, no recency
//!   cargo run --release -p expert-streamer --example headroom -- --trace FILE --experts 256
//!       a real `ROUTE_TRACE=` stream: real popularity AND real order
use expert_streamer::policy::*;
use expert_streamer::trace::*;

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
}

fn main() {
    let tokens: usize = arg("--tokens").and_then(|v| v.parse().ok()).unwrap_or(600);
    let (label, universe, trace, usage) = if let Some(f) = arg("--usage") {
        let u = parse_usage(&std::fs::read_to_string(&f).expect("read usage")).expect("usage");
        let k: usize = arg("--k").and_then(|v| v.parse().ok()).unwrap_or(8);
        let uni = (u.n_layers * u.n_experts) as usize;
        let t = u.sample_trace(k, tokens, 42);
        (
            format!("{f}: popularity from the history, {k} experts per layer"),
            uni,
            t,
            Some(u),
        )
    } else if let Some(f) = arg("--trace") {
        let ne: u32 = arg("--experts")
            .and_then(|v| v.parse().ok())
            .expect("--experts N");
        let t = parse_trace(&std::fs::read_to_string(&f).expect("read trace"), ne);
        let uni = t.iter().flatten().max().map_or(0, |m| *m as usize + 1);
        (format!("{f}: a real trace"), uni, t, None)
    } else {
        let t = synthetic_trace(256, 8, 4000, 42);
        ("synthetic trace".into(), 256, t, None)
    };
    println!("{label}\n{} tokens, {universe} experts in all", trace.len());
    println!(
        "{:>7} {:>8} {:>10} {:>9} {:>9}",
        "cache", "LRU", "DecayFreq", "pinned*", "optimal"
    );
    for pct in [5usize, 10, 25, 50] {
        let cap = (universe * pct / 100).max(1);
        let decay = if cap <= 3000 {
            format!(
                "{:.1}%",
                100.0 * replay(&trace, &mut DecayFreq::new(cap, 300.0))
            )
        } else {
            "-".into()
        };
        let pinned = usage.as_ref().map_or("-".into(), |u| {
            format!("{:.1}%", 100.0 * u.static_coverage(cap))
        });
        println!(
            "{:>6}% {:>7.1}% {:>10} {:>9} {:>8.1}%",
            pct,
            100.0 * replay(&trace, &mut Lru::new(cap)),
            decay,
            pinned,
            100.0 * belady(&trace, cap)
        );
    }
    println!("* a fixed set of the most used experts, best case on this history");
}
