//! Reproducible mode, end to end: a small model produces the same logits *bits* whatever SIMD
//! variant runs and however many threads the products are split over, and a receipt made under one
//! configuration replays under another. (A separate test binary: the switches are process-wide.)
use companion_engine::arch::{Act, Spec};
use companion_engine::model::{Bytes, Layer, Model, QWeight, Weight, Weights};
use companion_engine::quant::DType;
use companion_engine::quantize::quantize_matrix;
use companion_engine::receipt::{generate_with_receipt, replay, GenerateSpec, RecordOptions, SamplerParams};
use companion_engine::simd;
use std::sync::Mutex;

static LOCK: Mutex<()> = Mutex::new(());

fn init() {
    // Seven pool threads even on a small machine, set before the pool is first used.
    #[allow(unused_unsafe)]
    unsafe { std::env::set_var("COLIBRI_THREADS", "7"); }
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0 }
    fn mat(&mut self, n: usize) -> Vec<f32> { (0..n).map(|_| self.unit() * 0.08).collect() }
    fn ones(&mut self, n: usize) -> Vec<f32> { (0..n).map(|_| 1.0 + self.unit() * 0.1).collect() }
}

/// `quant`: (attention type, gate, up, down) when the big matrices should be kept quantised.
fn build(spec: Spec, seed: u64, quant: Option<[DType; 4]>) -> Model {
    let mut r = Rng(seed);
    let (h, qd, kd, ff, v) = (spec.hidden as usize, (spec.heads * spec.head_dim) as usize, (spec.kv_heads * spec.head_dim) as usize, spec.ffn as usize, spec.vocab as usize);
    let w = |rows: usize, cols: usize, t: Option<DType>, r: &mut Rng| -> Weight {
        let d = r.mat(rows * cols);
        match t {
            None => Weight::Dense(d),
            Some(t) => {
                let b = quantize_matrix(t, &d, cols, None, 1).expect("quantise");
                Weight::Quant(QWeight::new(t, rows, cols, Bytes::owned(b)).expect("qweight"))
            }
        }
    };
    let q = |i: usize| quant.map(|a| a[i]);
    let layers = (0..spec.layers).map(|_| Layer {
        attn_norm: r.ones(h), wq: w(qd, h, q(0), &mut r), wk: w(kd, h, q(0), &mut r), wv: w(kd, h, q(0), &mut r), wo: w(h, qd, q(0), &mut r),
        ffn_norm: r.ones(h), w_gate: w(ff, h, q(1), &mut r), w_up: w(ff, h, q(2), &mut r), w_down: w(h, ff, q(3), &mut r), ..Default::default()
    }).collect();
    let weights = Weights { embed: Weight::Dense(r.mat(v * h)), layers, final_norm: r.ones(h), final_norm_b: None, lm_head: Some(w(v, h, None, &mut r)), lm_head_b: None, experts: None };
    Model::new(spec, weights).expect("model")
}

fn dense_model() -> Model {
    // Gelu-tanh, both soft-caps and a large rope base: every libm user in the forward path.
    let mut s = Spec::dense("llama", 2, 128, 4, 2, 32, 2048, 512, 64);
    s.act = Act::GeluTanh;
    s.attn_softcap = 50.0;
    s.final_softcap = 30.0;
    s.rope_theta = 1_000_000.0;
    build(s, 0x1234_5678_9abc_def1, None)
}
fn quant_model() -> Model {
    let s = Spec::dense("llama", 1, 256, 8, 8, 32, 8192, 512, 64);
    build(s, 0x0fed_cba9_8765_4321, Some([DType::Q8_0, DType::Q4_0, DType::Q4K, DType::Q6K]))
}

/// Logit bits after feeding 6 prompt tokens in one batch and then 5 more one by one.
fn logits_bits(m: &Model) -> Vec<u32> {
    let mut st = m.new_state();
    let mut out: Vec<u32> = m.forward_batch(&mut st, &[5, 17, 301, 2, 99, 450]).unwrap().iter().map(|x| x.to_bits()).collect();
    for t in [7usize, 88, 3, 400, 12] { out.extend(m.forward(&mut st, t).unwrap().iter().map(|x| x.to_bits())); }
    out
}

fn variants() -> Vec<&'static str> { simd::available().iter().map(|k| k.name).collect() }

#[test]
fn logits_are_bit_identical_across_variants_and_thread_counts() {
    init();
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    simd::set_reproducible(true);
    for (label, mut m) in [("dense", dense_model()), ("quantised", quant_model())] {
        let mut reference: Option<Vec<u32>> = None;
        for v in variants() {
            assert!(simd::force_variant(v));
            for nt in [1usize, 2, 7] {
                m.set_threads(nt);
                let bits = logits_bits(&m);
                match &reference {
                    None => reference = Some(bits),
                    Some(r) => assert!(*r == bits, "{label}: variant {v} with {nt} threads differs from the first run"),
                }
            }
        }
        eprintln!("{label}: {} variants × 3 thread counts bit-identical ({} logits)", variants().len(), reference.unwrap().len());
    }
    // The threads really were available.
    assert!(companion_engine::pool::global().threads() >= 2);
    simd::force_variant("default");
}

#[test]
fn a_receipt_replays_under_another_variant_and_thread_count() {
    init();
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    simd::set_reproducible(true);
    let mut m = dense_model();
    let prompt = [5usize, 17, 301, 2];
    let g = GenerateSpec { sampler: SamplerParams { temperature: 0.7, top_k: 20, top_p: 0.95, repeat_penalty: 1.05, repeat_last_n: 32, seed: 2024 }, max_tokens: 12, stop_token: None };
    let o = RecordOptions { logit_topk: 8, keep_steps: true, merkle_leaves: true, key: Some(b"secret".to_vec()), timestamp: Some(1_800_000_000), ..Default::default() };
    let all = variants();
    assert!(simd::force_variant(all[0]));
    m.set_threads(1);
    let (tokens, receipt) = generate_with_receipt(&m, &prompt, &g, &o).unwrap();
    assert!(receipt.reproducible);
    let text = receipt.to_json();
    for v in &all {
        assert!(simd::force_variant(v));
        for nt in [1usize, 2, 7] {
            m.set_threads(nt);
            let rec = companion_engine::receipt::Receipt::from_json(&text).unwrap();
            let rep = replay(&rec, &m, &prompt, Some(b"secret")).unwrap();
            assert!(rep.identical, "variant {v}, {nt} threads: {rep:?}");
            assert_eq!(rep.tokens, tokens);
        }
    }
    // With the mode off, a reproducible receipt is not replayed (its guarantee would not hold).
    simd::set_reproducible(false);
    assert_eq!(replay(&receipt, &m, &prompt, Some(b"secret")), Err(companion_engine::receipt::ReceiptError::ReproducibleModeOff));
    simd::force_variant("default");
}

#[test]
fn normal_mode_is_free_to_differ_between_variants() {
    init();
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    simd::set_reproducible(false);
    let m = dense_model();
    let mut seen = std::collections::HashSet::new();
    for v in variants() {
        simd::force_variant(v);
        seen.insert(logits_bits(&m));
    }
    eprintln!("normal mode: {} distinct logit streams over {} variants", seen.len(), variants().len());
    simd::force_variant("default");
}
