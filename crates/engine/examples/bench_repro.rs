//! Throughput cost of reproducible mode: kernels (normal vs fixed-order twin, per variant) and a
//! small model's decode speed.
//!
//! `cargo run --release -p companion-engine --example bench_repro`
use companion_engine::arch::Spec;
use companion_engine::model::{Layer, Model, Weight, Weights};
use companion_engine::simd;
use std::hint::black_box;
use std::time::Instant;

fn best_secs(mut f: impl FnMut(), reps: usize) -> f64 {
    (0..5).map(|_| { let t = Instant::now(); for _ in 0..reps { f(); } t.elapsed().as_secs_f64() / reps as f64 }).fold(f64::MAX, f64::min)
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0 }
    fn vec(&mut self, n: usize, s: f32) -> Vec<f32> { (0..n).map(|_| self.unit() * s).collect() }
}

fn kernels(variant: &str) {
    simd::force_variant(variant);
    let mut r = Rng(1);
    let n = 4096usize;
    let (a, b) = (r.vec(n, 1.0), r.vec(n, 1.0));
    let (c, s) = (r.vec(n / 2, 1.0), r.vec(n / 2, 1.0));
    let mut y = r.vec(n, 1.0);
    let mut rows: Vec<(&str, f64, f64)> = vec![];
    for name in ["dot_f32", "sumsq", "axpy", "mul3", "exp_sub_sum", "silu_mul", "rope"] {
        let mut t = [0f64; 2];
        for (i, repro) in [false, true].into_iter().enumerate() {
            simd::set_reproducible(repro);
            let reps = 2000;
            t[i] = match name {
                "dot_f32" => best_secs(|| { black_box(simd::dot_f32(black_box(&a), black_box(&b))); }, reps),
                "sumsq" => best_secs(|| { black_box(simd::sumsq(black_box(&a))); }, reps),
                "axpy" => best_secs(|| simd::axpy(0.5, black_box(&a), &mut y), reps),
                "mul3" => { let mut o = vec![0f32; n]; best_secs(|| { simd::mul3(black_box(&a), black_box(&b), 0.5, &mut o); black_box(&o); }, reps) }
                "exp_sub_sum" => { let mut x = a.clone(); best_secs(|| { x.copy_from_slice(&a); black_box(simd::exp_sub_sum(&mut x, 1.0)); }, reps) }
                "silu_mul" => { let mut g = a.clone(); best_secs(|| { g.copy_from_slice(&a); simd::silu_mul(&mut g, black_box(&b)); }, reps) }
                _ => { let mut h = a.clone(); best_secs(|| simd::rope_rotate(&mut h, black_box(&c), black_box(&s)), reps) }
            };
        }
        rows.push((name, t[0], t[1]));
    }
    println!("variant {variant}: n = {n}");
    for (name, normal, repro) in rows {
        println!("  {name:<12} normal {:>7.2} GB/s   reproducible {:>7.2} GB/s   cost ×{:.2}", n as f64 * 8.0 / normal / 1e9, n as f64 * 8.0 / repro / 1e9, repro / normal);
    }
}

fn model() -> Model {
    let spec = Spec::dense("llama", 4, 512, 8, 8, 64, 1536, 2048, 128);
    let mut r = Rng(9);
    let (h, ff, v) = (512usize, 1536usize, 2048usize);
    let layers = (0..4).map(|_| Layer {
        attn_norm: vec![1.0; h], wq: Weight::Dense(r.vec(h * h, 0.05)), wk: Weight::Dense(r.vec(h * h, 0.05)), wv: Weight::Dense(r.vec(h * h, 0.05)), wo: Weight::Dense(r.vec(h * h, 0.05)),
        ffn_norm: vec![1.0; h], w_gate: Weight::Dense(r.vec(ff * h, 0.05)), w_up: Weight::Dense(r.vec(ff * h, 0.05)), w_down: Weight::Dense(r.vec(h * ff, 0.05)), ..Default::default()
    }).collect();
    let w = Weights { embed: Weight::Dense(r.vec(v * h, 0.05)), layers, final_norm: vec![1.0; h], final_norm_b: None, lm_head: Some(Weight::Dense(r.vec(v * h, 0.05))), lm_head_b: None, experts: None };
    Model::new(spec, w).expect("model")
}

fn decode(m: &Model) -> f64 {
    let mut st = m.new_state();
    m.forward_batch(&mut st, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let t = Instant::now();
    let n = 40;
    for i in 0..n { black_box(m.forward(&mut st, 10 + i).unwrap()); }
    n as f64 / t.elapsed().as_secs_f64()
}

fn main() {
    println!("default variant: {}   pool threads: {}", simd::active(), companion_engine::pool::global().threads());
    for v in simd::available().iter().map(|k| k.name) { kernels(v); }
    simd::force_variant("default");
    let mut m = model();
    println!("\nmodel: 4 layers, hidden 512, f32 weights (decode tokens/s, best of 3)");
    for nt in [1usize, 0] {
        m.set_threads(nt);
        let mut t = [0f64; 2];
        for (i, repro) in [false, true].into_iter().enumerate() {
            simd::set_reproducible(repro);
            t[i] = (0..3).map(|_| decode(&m)).fold(0.0, f64::max);
        }
        println!("  threads {:>2}: normal {:>7.1} tok/s   reproducible {:>7.1} tok/s   slowdown ×{:.2}", if nt == 0 { companion_engine::pool::global().threads() } else { nt }, t[0], t[1], t[0] / t[1]);
    }
}
