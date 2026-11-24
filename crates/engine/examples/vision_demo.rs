//! Synthetic picture -> random CLIP-like encoder -> projector -> tiny random language model.
//! Meaningless output; it exercises the whole image path and prints shapes and timings.
//! Run: `cargo run -p companion-engine --example vision_demo`
use companion_engine::imageio::{self, synthetic};
use companion_engine::vision::{forward_pieces, AnyRes, Piece, VisionModel, VisionSpec};
use std::time::Instant;
use companion_engine::arch::Spec;
use companion_engine::model::{Layer, Model, Weights};

struct Rng(u64);
impl Rng { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6 } }
fn mat(n: usize, r: &mut Rng) -> Vec<f32> { (0..n).map(|_| r.f()).collect() }

fn build() -> Model {
    let (layers, h, heads, kv, hd, ff, vocab) = (6u64, 32u64, 4u64, 2u64, 8u64, 48u64, 40u64);
    let spec = Spec::dense("llama", layers, h, heads, kv, hd, ff, vocab, 128);
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let (h, qd, kd, ff, v) = (h as usize, (heads * hd) as usize, (kv * hd) as usize, ff as usize, vocab as usize);
    let ls = (0..layers).map(|_| Layer { attn_norm: vec![1.0; h], wq: mat(qd * h, &mut r).into(), wk: mat(kd * h, &mut r).into(), wv: mat(kd * h, &mut r).into(),
        wo: mat(h * qd, &mut r).into(), ffn_norm: vec![1.0; h], w_gate: mat(ff * h, &mut r).into(), w_up: mat(ff * h, &mut r).into(), w_down: mat(h * ff, &mut r).into(), ..Default::default() }).collect();
    Model::new(spec, Weights { embed: mat(v * h, &mut r).into(), layers: ls, final_norm: vec![1.0; h], lm_head: Some(mat(v * h, &mut r).into()), ..Default::default() }).unwrap()
}


fn main() {
    let lm = build();
    let spec = VisionSpec::clip(32, 8, 32, 4, 4, 64);
    let vm = VisionModel::random(spec, lm.spec.hidden as usize, 1).expect("encoder");
    // round-trip through a real file format first
    let png = imageio::encode_png(&synthetic(120, 80, 3));
    let img = imageio::decode(&png).expect("decode");
    let t = Instant::now();
    let one = vm.embed_image(&img).expect("embed");
    println!("single view: {} rows x {} in {:?}", one.count, one.dim, t.elapsed());
    let t = Instant::now();
    let big = vm.embed_anyres(&img, &AnyRes { max_tiles: 4, thumbnail: true }).expect("anyres");
    println!("anyres: {} rows in {:?}", big.count, t.elapsed());
    let mut st = lm.new_state();
    let logits = forward_pieces(&lm, &mut st, &[Piece::Text(&[1, 2]), Piece::Image(&one), Piece::Text(&[3])]).expect("prefill");
    let best = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0).unwrap();
    println!("language model consumed {} positions; argmax token {best}", st.tokens().len());
}
