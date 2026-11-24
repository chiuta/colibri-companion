//! Lists the compute kernels and, when built with `--features wgpu`, the adapters wgpu can see.
use companion_engine_gpu::{layout, shaders};

fn main() {
    println!("kernels ({}):", shaders::all().len());
    for (n, s) in shaders::all() { println!("  {n:<14} {:>5} bytes of WGSL", s.len()); }
    println!("weight formats:");
    for f in layout::ALL { println!("  {:<5} block = {} elems / {} words, {} work items per block", f.name(), f.block_elems(), f.block_words(), f.items_per_block()); }
    #[cfg(feature = "wgpu")]
    {
        let list = companion_engine_gpu::gpu::adapters(wgpu::Backends::all());
        if list.is_empty() { println!("no GPU adapter found"); }
        for a in list { println!("adapter: {} [{} / {}], max storage binding {} MiB, max buffer {} MiB", a.name, a.backend, a.kind, a.max_binding >> 20, a.max_buffer >> 20); }
    }
    #[cfg(not(feature = "wgpu"))]
    println!("built without the `wgpu` feature: no device access (rebuild with --features wgpu)");
}
