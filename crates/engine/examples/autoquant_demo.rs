//! Mixed-precision recipe search on a synthetic network with two engineered sensitive layers.
//!
//! `cargo run -p companion-engine --example autoquant_demo -- [bits-per-weight] [measured|hybrid|proxy] [exact|greedy]`
//!
//! Prints the sensitivity table, the chosen formats, predicted versus measured damage, and a comparison
//! with every uniform format. It needs no model file; it says nothing about how a real checkpoint behaves.
use companion_engine::autoquant::synthetic::SyntheticNet;
use companion_engine::autoquant::*;
use companion_engine::hw::HardwareProfile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bpw: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4.5);
    let method = match args.get(2).map(String::as_str) {
        Some("proxy") => Method::Proxy,
        Some("hybrid") => Method::Hybrid { anchor: None },
        _ => Method::Measured,
    };
    let solver = if args.get(3).map(String::as_str) == Some("greedy") { Solver::Greedy } else { Solver::Exact };

    let mut net = SyntheticNet::new(7, 8, 256, 64, 24, &[1, 5]);
    let speed = SpeedModel::from_profile(&HardwareProfile::detect(None));
    let opts = AutoOpts { solve: SolveOpts { solver, speed: Some(speed), speed_weight: 0.0, ..Default::default() }, ..Default::default() };

    let mut last = String::new();
    let mut on_progress = |p: &Progress| {
        let line = format!("[{} {}/{}] {}", p.phase, p.done, p.total, p.detail);
        if line != last { eprintln!("{line}"); last = line; }
    };
    let mut ctl = Control { progress: Some(&mut on_progress), cancel: None };
    let out = match auto_quant(&mut net, &DEFAULT_CANDIDATES, method, Budget::Bpw(bpw), &opts, &mut ctl) {
        Ok(o) => o,
        Err(e) => { eprintln!("failed: {e}"); std::process::exit(1); }
    };
    println!("{}", out.to_markdown());

    match out.plan.to_recipe("Q4_K_M", &out.sens.groups) {
        Ok(r) => {
            println!("## Recipe overrides (base {})\n", r.name);
            for (pattern, t) in &r.overrides { println!("  {pattern} -> {}", companion_engine::quantize::dtype_name(*t)); }
        }
        Err(e) => eprintln!("recipe: {e}"),
    }
}
