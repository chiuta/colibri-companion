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
    let tokens: usize = number("--tokens").unwrap_or(600);
    let (label, universe, trace, usage) = if let Some(f) = arg("--usage") {
        let u = or_exit(read_usage_file(&f));
        let k: usize = number("--k").unwrap_or(8);
        let uni = u.n_layers as usize * u.n_experts as usize;
        let t = u.sample_trace(k, tokens, 42);
        (
            format!("{f}: popularity from the history, {k} experts per layer"),
            uni,
            t,
            Some(u),
        )
    } else if let Some(f) = arg("--trace") {
        let ne: u32 = or_exit(number("--experts").ok_or("--trace needs --experts N".to_string()));
        let t = or_exit(read_trace_file(&f, ne));
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
        let pinned = match &usage {
            Some(u) if u.total() > 0 => format!("{:.1}%", 100.0 * u.static_coverage(cap)),
            _ => "n/a".into(), // no history, or a history with no uses: nothing to pin
        };
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
