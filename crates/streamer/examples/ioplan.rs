//! `ioplan` prints tokens/s against queue depth for a drive, with and without overlapping the
//! reads of the next layer with the compute of the current one. Defaults are ASSUMPTIONS, and the
//! overlapped column is a ceiling that assumes the next layer's experts are known in advance.
//!   cargo run --release -p expert-streamer --example ioplan
//!   cargo run --release -p expert-streamer --example ioplan -- --measured-mbs 1800 --hit 0.3
//! Flags: --measured-mbs N  --seq-mbs N  --rand-mbs N  --latency-us N  --expert-mb N
//!        --layers N  --k N  --hit F  --compute-ms N
use expert_streamer::ioplan::Params;

fn raw(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1).cloned())
}

fn fail(msg: String) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

/// A flag's value as a finite number; a flag with a missing or unreadable value is an error,
/// never silently the default (`--hit 0,3` must not run with the default hit rate).
fn arg(name: &str) -> Option<f64> {
    let has = std::env::args().any(|x| x == name);
    match (raw(name), has) {
        (None, true) => fail(format!("{name} needs a value")),
        (None, false) => None,
        (Some(v), _) => match v.parse::<f64>() {
            Ok(n) if n.is_finite() => Some(n),
            _ => fail(format!("{name} needs a number, got `{v}`")),
        },
    }
}

/// A whole number of at least `min`.
fn count(name: &str, min: u32) -> Option<u32> {
    arg(name).map(|v| {
        if v.fract() != 0.0 || v < min as f64 || v > u32::MAX as f64 {
            fail(format!("{name} needs a whole number from {min}, got {v}"));
        }
        v as u32
    })
}

fn tune(mut p: Params) -> Params {
    if let Some(v) = arg("--seq-mbs") {
        p.seq_bw = v * 1e6;
    }
    if let Some(v) = arg("--rand-mbs") {
        p.rand_bw = v * 1e6;
    }
    if let Some(v) = arg("--latency-us") {
        p.latency = v * 1e-6;
    }
    if let Some(v) = arg("--expert-mb") {
        p.expert_bytes = v * 1e6;
    }
    if let Some(v) = count("--layers", 1) {
        p.layers = v;
    }
    if let Some(v) = count("--k", 1) {
        p.experts_per_layer = v;
    }
    if let Some(v) = arg("--hit") {
        p.hit_rate = v;
    }
    if let Some(v) = arg("--compute-ms") {
        p.compute = v * 1e-3;
    }
    p
}

fn show(name: &str, p: &Params) {
    if let Err(e) = p.validate() {
        eprintln!("error: {name}: {e}");
        std::process::exit(1);
    }
    println!(
        "\n{name}: {:.0} MB/s single, {:.0} MB/s ceiling, {:.0} us, expert {:.1} MB, {} layers x {} experts, hit {:.0}%, compute {:.1} ms/layer",
        p.seq_bw / 1e6,
        p.rand_bw / 1e6,
        p.latency * 1e6,
        p.expert_bytes / 1e6,
        p.layers,
        p.experts_per_layer,
        p.hit_rate * 100.0,
        p.compute * 1e3
    );
    match p.saturation_depth() {
        Some(q) => println!("saturates at queue depth {q}"),
        None => println!("does not saturate within queue depth 64"),
    }
    println!(
        "{:>6} {:>10} {:>12} {:>13}",
        "depth", "GB/s", "tok/s serial", "tok/s ceiling"
    );
    for q in [1u32, 2, 4, 8, 16, 32, 64] {
        println!(
            "{:>6} {:>10.2} {:>12.2} {:>13.2}",
            q,
            p.throughput(q) / 1e9,
            p.tokens_per_s_serial(q),
            p.tokens_per_s_overlapped(q)
        );
    }
}

fn main() {
    println!("model with ASSUMED defaults; replace them with a measurement");
    if let Some(m) = arg("--measured-mbs") {
        if m <= 0.0 {
            fail(format!("--measured-mbs must be above 0, got {m}"));
        }
        show("measured drive", &tune(Params::measured(m)));
    } else {
        show("NVMe (assumed)", &tune(Params::nvme()));
        show("SATA SSD (assumed)", &tune(Params::sata()));
    }
}
