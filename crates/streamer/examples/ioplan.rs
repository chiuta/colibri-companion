//! `ioplan` prints tokens/s against queue depth for a drive, with and without overlapping the
//! reads of the next layer with the compute of the current one. Defaults are ASSUMPTIONS.
//!   cargo run --release -p expert-streamer --example ioplan
//!   cargo run --release -p expert-streamer --example ioplan -- --measured-mbs 1800 --hit 0.3
//! Flags: --measured-mbs N  --seq-mbs N  --rand-mbs N  --latency-us N  --expert-mb N
//!        --layers N  --k N  --hit F  --compute-ms N
use expert_streamer::ioplan::Params;

fn arg(name: &str) -> Option<f64> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1)?.parse().ok())
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
    if let Some(v) = arg("--layers") {
        p.layers = v as u32;
    }
    if let Some(v) = arg("--k") {
        p.experts_per_layer = v as u32;
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
    println!("saturates at queue depth {}", p.saturation_depth());
    println!(
        "{:>6} {:>10} {:>12} {:>12}",
        "depth", "GB/s", "tok/s serial", "tok/s overlap"
    );
    for q in [1u32, 2, 4, 8, 16, 32, 64] {
        println!(
            "{:>6} {:>10.2} {:>12.2} {:>12.2}",
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
        show("measured drive", &tune(Params::measured(m)));
    } else {
        show("NVMe (assumed)", &tune(Params::nvme()));
        show("SATA SSD (assumed)", &tune(Params::sata()));
    }
}
