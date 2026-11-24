//! Synthetic-model speed test: decode, prefill (token by token vs batched), quantised weights, KV precision.
//!   cargo run --release --example bench_model [-- tiny|mid]
use companion_engine::arch::Spec;
use companion_engine::model::{Layer, KvKind, Model, Weights};
use std::time::Instant;

struct Rng(u64);
impl Rng { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1 } }
fn mat(n: usize, r: &mut Rng) -> Vec<f32> { (0..n).map(|_| r.f()).collect() }
fn ones(n: usize) -> Vec<f32> { vec![1.0; n] }

#[allow(clippy::too_many_arguments)]
fn build(layers: u64, h: u64, heads: u64, kv: u64, hd: u64, ff: u64, vocab: u64, ctx: u64) -> Model {
    let spec = Spec::dense("llama", layers, h, heads, kv, hd, ff, vocab, ctx);
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let (h, qd, kd, ff, v) = (h as usize, (heads * hd) as usize, (kv * hd) as usize, ff as usize, vocab as usize);
    let ls = (0..layers).map(|_| Layer { attn_norm: ones(h), wq: mat(qd * h, &mut r).into(), wk: mat(kd * h, &mut r).into(), wv: mat(kd * h, &mut r).into(),
        wo: mat(h * qd, &mut r).into(), ffn_norm: ones(h), w_gate: mat(ff * h, &mut r).into(), w_up: mat(ff * h, &mut r).into(), w_down: mat(h * ff, &mut r).into(), ..Default::default() }).collect();
    Model::new(spec, Weights { embed: mat(v * h, &mut r).into(), layers: ls, final_norm: ones(h), lm_head: Some(mat(v * h, &mut r).into()), ..Default::default() }).unwrap()
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "mid".into());
    let m = if which == "tiny" { build(3, 16, 4, 2, 8, 24, 20, 512) } else { build(8, 512, 8, 4, 64, 1408, 8000, 1024) };
    let toks: Vec<usize> = (0..256).map(|i| (i * 7 + 3) % m.spec.vocab as usize).collect();
    // decode: 128 single tokens after a 64-token prompt
    let mut st = m.new_state();
    m.forward_batch(&mut st, &toks[..64]).unwrap();
    let t = Instant::now();
    for &x in &toks[64..192] { m.forward(&mut st, x).unwrap(); }
    println!("{which}: decode            {:8.1} tok/s", 128.0 / t.elapsed().as_secs_f64());
    // prefill 256 tokens: one at a time vs batched
    let mut st = m.new_state();
    let t = Instant::now();
    for &x in &toks { m.forward(&mut st, x).unwrap(); }
    let seq = 256.0 / t.elapsed().as_secs_f64();
    let mut st = m.new_state();
    let t = Instant::now();
    m.forward_batch(&mut st, &toks).unwrap();
    let bat = 256.0 / t.elapsed().as_secs_f64();
    println!("{which}: prefill sequential {seq:8.1} tok/s | batched {bat:8.1} tok/s | x{:.2}", bat / seq);
    for kind in [KvKind::F32, KvKind::F16, KvKind::Q8] {
        let mut st = m.new_state_with(kind);
        let t = Instant::now();
        m.forward_batch(&mut st, &toks).unwrap();
        for &x in &toks[..64] { if st.pos < m.spec.context as usize { m.forward(&mut st, x).unwrap(); } }
        println!("{which}: kv {kind:?}: {:6.2} MB, {:.2}s for prefill+64 decode", st.kv_bytes() as f64 / 1e6, t.elapsed().as_secs_f64());
    }
    println!("{which}: weights {:.1} MB", m.weight_bytes() as f64 / 1e6);
}
