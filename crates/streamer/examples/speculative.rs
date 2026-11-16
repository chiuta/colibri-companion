//! Cost model for verifying k drafted tokens per pass (loads per accepted token). Not an engine.
//!   cargo run --release -p expert-streamer --example speculative [-- --usage FILE | --trace FILE --experts N] [--cache PCT]
//! Real overlap between consecutive tokens is only known from a real ROUTE_TRACE; a trace sampled
//! from a `.coli_usage` history overlaps only through popularity.
use expert_streamer::policy::synthetic_trace;
use expert_streamer::speculative::loads_per_token;
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
    let (label, universe, trace) = if let Some(f) = arg("--usage") {
        let u = or_exit(read_usage_file(&f));
        let t = u.sample_trace(8, 800, 42);
        (
            format!("{f}: sampled, popularity only"),
            u.n_layers as usize * u.n_experts as usize,
            t,
        )
    } else if let Some(f) = arg("--trace") {
        let ne: u32 = or_exit(number("--experts").ok_or("--trace needs --experts N".to_string()));
        let t = or_exit(read_trace_file(&f, ne));
        let uni = t.iter().flatten().max().map_or(0, |m| *m as usize + 1);
        (format!("{f}: a real trace"), uni, t)
    } else {
        (
            "synthetic trace".into(),
            256,
            synthetic_trace(256, 8, 4000, 42),
        )
    };
    let pct: usize = number("--cache").unwrap_or(10);
    let cap = (universe * pct / 100).max(1);
    println!(
        "{label}: {} tokens, {universe} experts. A cost model, not an engine.",
        trace.len()
    );
    for (name, c) in [("no cache", 0usize), ("cache", cap)] {
        println!(
            "\n{name}{}: expert loads per accepted token",
            if c > 0 {
                format!(" {pct}%")
            } else {
                String::new()
            }
        );
        println!("{:>4} {:>9} {:>9} {:>9}", "k", "a=50%", "a=70%", "a=90%");
        for k in [1usize, 2, 4, 8] {
            let v: Vec<String> = [0.5, 0.7, 0.9]
                .iter()
                .map(|a| format!("{:>9.1}", loads_per_token(&trace, k, *a, c)))
                .collect();
            println!("{k:>4} {}", v.join(" "));
        }
    }
}
