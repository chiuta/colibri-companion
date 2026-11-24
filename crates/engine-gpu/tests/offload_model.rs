//! The whole offload path with no GPU: a real `Model` runs through `Model::with_backend` on the
//! emulated device, and its logits, its layer split and its memory use are checked against the CPU
//! path and against the planner's prediction.
use companion_engine::arch::Spec;
use companion_engine::backend::Backend;
use companion_engine::model::{AccelOptions, Bytes, Layer, Model, QWeight, Weight, Weights};
use companion_engine::quant::DType;
use companion_engine::quantize::quantize_matrix;
use companion_engine_gpu::caps::DeviceClass;
use companion_engine_gpu::emubackend::EmuBackend;
use companion_engine_gpu::offload::{Limiter, Request, layer_device_bytes, plan_offload};
use std::sync::Arc;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
    }
}

fn spec(layers: u64) -> Spec {
    Spec::dense("llama", layers, 256, 4, 2, 64, 512, 32, 64)
}

fn weight(r: &mut Rng, rows: usize, cols: usize, dtype: Option<DType>) -> Weight {
    let data: Vec<f32> = (0..rows * cols).map(|_| r.f()).collect();
    match dtype {
        None => Weight::Dense(data),
        Some(t) => Weight::Quant(
            QWeight::new(
                t,
                rows,
                cols,
                Bytes::owned(quantize_matrix(t, &data, cols, None, 1).unwrap()),
            )
            .unwrap(),
        ),
    }
}

fn model(layers: u64, dtype: Option<DType>) -> Model {
    let s = spec(layers);
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let (h, qd, kd, ff, v) = (256usize, 256usize, 128usize, 512usize, 32usize);
    let ls = (0..layers)
        .map(|_| Layer {
            attn_norm: vec![1.0; h],
            wq: weight(&mut r, qd, h, dtype),
            wk: weight(&mut r, kd, h, dtype),
            wv: weight(&mut r, kd, h, dtype),
            wo: weight(&mut r, h, qd, dtype),
            ffn_norm: vec![1.0; h],
            w_gate: weight(&mut r, ff, h, dtype),
            w_up: weight(&mut r, ff, h, dtype),
            w_down: weight(&mut r, h, ff, dtype),
            ..Default::default()
        })
        .collect();
    let embed: Vec<f32> = (0..v * h).map(|_| r.f()).collect();
    let head: Vec<f32> = (0..v * h).map(|_| r.f()).collect();
    Model::new(
        s,
        Weights {
            embed: embed.into(),
            layers: ls,
            final_norm: vec![1.0; h],
            lm_head: Some(head.into()),
            ..Default::default()
        },
    )
    .unwrap()
}

const TOKENS: [usize; 6] = [3, 17, 5, 30, 0, 11];

fn run(m: &Model) -> Vec<Vec<f32>> {
    let mut st = m.new_state();
    TOKENS
        .iter()
        .map(|&t| m.forward(&mut st, t).unwrap())
        .collect()
}
fn run_batch(m: &Model) -> Vec<f32> {
    let mut st = m.new_state();
    m.forward_batch(&mut st, &TOKENS).unwrap()
}
fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}
fn emu(budget: u64) -> Arc<EmuBackend> {
    Arc::new(EmuBackend::new(budget, 1 << 30))
}

#[test]
fn dense_layers_on_the_emulated_device_match_the_cpu_path() {
    let cpu = model(3, None);
    let want = run(&cpu);
    let want_batch = run_batch(&cpu);
    let be = emu(1 << 30);
    let dev = model(3, None).with_backend(be.clone()).unwrap();
    assert_eq!(dev.offloaded(), vec![true; 3]);
    assert_eq!(be.host_weights(), 0);
    for (i, (a, b)) in run(&dev).iter().zip(&want).enumerate() {
        assert!(rel_err(a, b) < 1e-4, "token {i}: {}", rel_err(a, b));
    }
    assert!(rel_err(&run_batch(&dev), &want_batch) < 1e-4);
}

#[test]
fn every_quantised_format_matches_the_cpu_within_activation_rounding() {
    for t in [DType::Q8_0, DType::Q4_0, DType::Q4K, DType::Q6K] {
        let cpu = model(2, Some(t));
        let want = run(&cpu);
        let be = emu(1 << 30);
        let dev = model(2, Some(t)).with_backend(be.clone()).unwrap();
        assert_eq!(dev.offloaded(), vec![true; 2], "{t:?}");
        assert_eq!(
            be.host_weights(),
            0,
            "{t:?}: every weight must have a device kernel"
        );
        // The CPU path rounds the activations to 8 bits before the integer dot; the device path
        // multiplies the dequantised weights by exact activations. Same weights, so the two stay close.
        for (i, (a, b)) in run(&dev).iter().zip(&want).enumerate() {
            assert!(
                rel_err(a, b) < 0.05,
                "{t:?} token {i}: rel err {}",
                rel_err(a, b)
            );
        }
    }
}

#[test]
fn the_planner_counts_exactly_the_bytes_the_device_takes() {
    for t in [
        None,
        Some(DType::Q8_0),
        Some(DType::Q4_0),
        Some(DType::Q4K),
        Some(DType::Q6K),
    ] {
        let m = model(3, t);
        let s = m.spec.clone();
        let be = emu(1 << 30);
        let dev = m.with_backend(be.clone()).unwrap();
        let per_layer = layer_device_bytes(&s, t.unwrap_or(DType::F32)).unwrap();
        assert_eq!(
            be.used(),
            per_layer * 3,
            "{t:?}: planner and uploads disagree"
        );
        drop(dev);
    }
}

#[test]
fn partial_offload_follows_the_planner_and_still_computes_the_same_thing() {
    let s = spec(3);
    let per_layer = layer_device_bytes(&s, DType::F32).unwrap();
    let cap = 64u64;
    let kv_layer = cap * 128 * 8;
    let reserve = 1u64 << 20;
    // Room for two layers and their caches, not three.
    let budget = reserve + 2 * (per_layer + kv_layer) + (per_layer / 2);
    let req = Request {
        spec: &s,
        dtype: DType::F32,
        context: cap,
        vram: budget,
        reserve,
        binding: 1 << 30,
        class: DeviceClass::Discrete,
        cpu_decode_tps: None,
    };
    let plan = plan_offload(&req);
    assert_eq!(
        (plan.gpu_layers, plan.limiter),
        (2, Limiter::Memory),
        "{plan:?}"
    );
    assert_eq!(plan.kv_capacity, cap);

    let be = emu(budget);
    let dev = model(3, None)
        .with_backend_opts(be.clone(), plan.accel_options(reserve))
        .unwrap();
    assert_eq!(
        dev.offloaded(),
        vec![true, true, false],
        "the model must take exactly the planned layers"
    );
    assert_eq!(be.host_weights(), 0);
    let mut st = dev.new_state();
    let _ = dev.forward(&mut st, 1).unwrap();
    assert!(
        be.used() <= budget,
        "the device never goes over its budget: {} > {budget}",
        be.used()
    );

    let cpu = model(3, None);
    for (i, (a, b)) in run(&dev).iter().zip(run(&cpu)).enumerate() {
        assert!(rel_err(a, &b) < 1e-4, "mixed host/device token {i}");
    }

    // The model's own rule (no plan options) lands on the same split for dense weights.
    let own = model(3, None)
        .with_backend_opts(
            emu(budget),
            AccelOptions {
                max_layers: None,
                kv_capacity: cap as usize,
                reserve,
            },
        )
        .unwrap();
    assert_eq!(own.offloaded(), dev.offloaded());
}

#[test]
fn a_small_binding_limit_splits_the_weights_without_changing_a_bit() {
    let whole = model(2, Some(DType::Q4K))
        .with_backend(Arc::new(EmuBackend::new(1 << 30, 1 << 30)))
        .unwrap();
    // 40 KB bindings: the 144 KB-and-up matrices must be uploaded in several row chunks.
    let chunked = model(2, Some(DType::Q4K))
        .with_backend(Arc::new(EmuBackend::new(1 << 30, 40_000)))
        .unwrap();
    assert_eq!(chunked.offloaded(), vec![true; 2]);
    for (a, b) in run(&whole).iter().zip(run(&chunked)) {
        assert_eq!(*a, b, "row chunking must not change any result");
    }
}

#[test]
fn what_does_not_fit_stays_on_the_host_and_is_still_correct() {
    // A device too small for any matrix: every upload falls back to the CPU backend.
    let be = Arc::new(EmuBackend::new(4096, 1 << 30));
    let w: Vec<f32> = (0..16 * 32)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.1)
        .collect();
    let h = be
        .upload(companion_engine::backend::WeightRef::dense(16, 32, &w))
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(h.on_device, "a small matrix fits");
    let big: Vec<f32> = (0..64 * 256)
        .map(|i| ((i * 3 % 11) as f32 - 5.0) * 0.05)
        .collect();
    let hb = be
        .upload(companion_engine::backend::WeightRef::dense(64, 256, &big))
        .unwrap();
    assert!(!hb.on_device, "an oversized matrix is kept on the host");
    assert_eq!(be.host_weights(), 1);
    let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.2).sin()).collect();
    let (xb, ob) = (be.alloc_from(&x).unwrap(), be.alloc(64).unwrap());
    be.matvec(&hb, &xb, &ob).unwrap();
    let mut got = vec![0.0; 64];
    be.read(&ob, &mut got).unwrap();
    let mut want = vec![0.0; 64];
    companion_engine::ops::matvec(&big, 256, &x, &mut want);
    assert_eq!(got, want);
}

#[test]
fn the_planner_refuses_what_the_kernels_cannot_do_and_the_model_agrees() {
    let mut s = spec(2);
    s.qk_norm = true;
    let plan = plan_offload(&Request {
        spec: &s,
        dtype: DType::F32,
        context: 64,
        vram: 1 << 30,
        reserve: 0,
        binding: 1 << 30,
        class: DeviceClass::Discrete,
        cpu_decode_tps: None,
    });
    assert_eq!(
        (plan.gpu_layers, plan.limiter),
        (0, Limiter::NoEligibleLayer)
    );
}
