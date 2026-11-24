//! Steering and logit lens on a small synthetic model (random weights: it shows the mechanics,
//! not meaningful behaviour). With a real checkpoint, train on tokenised contrastive prompts.
//!   cargo run --example steer_demo
use companion_engine::arch::Spec;
use companion_engine::model::{Layer, Model, Weights};
use companion_engine::steer::{self, ControlVector, LensOptions, LensPositions, Method, Profile, Steering};

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
    let m = build();
    let prompt: Vec<usize> = (0..10).map(|i| (i * 7 + 2) % 40).collect();
    // two contrastive prompt sets: sequences ending in token 3 versus token 9
    let pos: Vec<Vec<usize>> = (0..8).map(|i| [vec![(i * 5 + 1) % 40, (i * 3 + 2) % 40], vec![3]].concat()).collect();
    let neg: Vec<Vec<usize>> = (0..8).map(|i| [vec![(i * 5 + 1) % 40, (i * 3 + 2) % 40], vec![9]].concat()).collect();
    let cv = steer::train(&m, &pos, &neg, Method::MeanDiff, "three-vs-nine").unwrap();
    println!("trained '{}' ({}): norms per layer {:?}", cv.name, cv.method.name(), cv.norms().iter().map(|n| (n * 100.0).round() / 100.0).collect::<Vec<_>>());

    let path = std::env::temp_dir().join("steer-demo.cvec");
    cv.save(&path).unwrap();
    let cv = ControlVector::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    println!("saved and reloaded: {} bytes", cv.to_bytes().len());

    println!("\nstrength sweep, bell profile around layer 3:");
    for a in [0.0f32, 0.5, 1.0, 2.0, 4.0] {
        let steering = Steering::new(&m, &cv, Profile::bell(6, 3.0, 1.0, a)).unwrap();
        let mut st = m.new_state();
        let logits = steer::forward_steered(&m, &mut st, &prompt, Some(&steering)).unwrap();
        let top = logits.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b });
        println!("  strength {a:>3}: top-1 token {:>2} (logit {:.3})", top.0, top.1);
    }

    let mut st = m.new_state();
    let rep = steer::lens(&m, &mut st, &prompt, &LensOptions { top_k: 3, positions: LensPositions::LastN(3), ..Default::default() }).unwrap();
    println!("\nlogit lens, last 3 positions:");
    for p in &rep.positions {
        let row: Vec<String> = p.layers.iter().map(|l| format!("{}", l.top[0].0)).collect();
        println!("  pos {:>2}: layer top-1s [{}] final {} | first hit L{}, commits at L{}", p.index, row.join(" "), p.final_top1, p.first_hit_layer, p.commit_layer);
    }
    println!("mean commit depth {:.2}; attention first-slot share per layer: {:?}", rep.mean_commit_depth(),
        rep.attention.iter().map(|a| a.as_ref().map(|a| (a.first_slot_share * 100.0).round() / 100.0)).collect::<Vec<_>>());
    println!("\nJSON (first 300 chars): {}", &rep.to_json().to_string()[..300]);
}
