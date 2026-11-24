//! The capability report `doctor` shows, plus the kernels and weight formats.
//!   cargo run -p companion-engine-gpu --example gpu_info [-- --json]
//!   cargo run -p companion-engine-gpu --features wgpu --example gpu_info
use companion_engine_gpu::{layout, report, shaders};

fn main() {
    let profile = companion_engine::hw::HardwareProfile::detect(None);
    let r = report::capability_report(Some(&profile));
    if std::env::args().any(|a| a == "--json") {
        println!("{}", r.json());
        return;
    }
    print!("{}", r.text());
    println!("kernels ({}):", shaders::all().len());
    for (n, s) in shaders::all() {
        println!("  {n:<14} {:>5} bytes of WGSL", s.len());
    }
    println!("weight formats:");
    for f in layout::ALL {
        println!(
            "  {:<5} block = {} elems / {} words, {} work items per block",
            f.name(),
            f.block_elems(),
            f.block_words(),
            f.items_per_block()
        );
    }
}
