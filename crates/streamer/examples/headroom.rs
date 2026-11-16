//! `cargo run -p expert-streamer --example headroom -- [experts] [per_token] [tokens]`
//! Replays a SYNTHETIC routing trace: it shows how the policies compare and how much room an
//! ideal policy leaves, not how any real model routes.
use expert_streamer::policy::*;

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let (n, k, t) = (
        a.first().copied().unwrap_or(256) as u32,
        a.get(1).copied().unwrap_or(8),
        a.get(2).copied().unwrap_or(4000),
    );
    let trace = synthetic_trace(n, k, t, 42);
    println!("{n} experts, {k} per token, {t} tokens (synthetic)");
    println!("{:>8} {:>8} {:>10} {:>9}", "cache", "LRU", "DecayFreq", "optimal");
    for pct in [5, 10, 25, 50] {
        let cap = (n as usize * pct / 100).max(1);
        println!(
            "{:>7}% {:>7.1}% {:>9.1}% {:>8.1}%",
            pct,
            100.0 * replay(&trace, &mut Lru::new(cap)),
            100.0 * replay(&trace, &mut DecayFreq::new(cap, 300.0)),
            100.0 * belady(&trace, cap)
        );
    }
}
