//! Layer-streamed inference demo: writes a synthetic safetensors model, then runs it with only a few
//! layers in memory and a simulated slow disk, comparing window sizes.
//!
//!     cargo run --release -p companion-engine --example stream_demo -- [layers] [disk_ms_per_layer]
use companion_engine::arch::Spec;
use companion_engine::loader::{LoadOptions, Naming, SafeTensors, WeightFile, layer_name};
use companion_engine::stream_layers::{
    FileLayers, LayerSource, StreamedModel, ThrottledLayers, plan_layers,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let layers: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(12);
    let ms: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(20);
    let (h, ff, v) = (256usize, 1024usize, 512usize);
    let spec = Spec::dense(
        "llama", layers, h as u64, 8, 4, 32, ff as u64, v as u64, 1024,
    );
    let (qd, kd) = (256usize, 128usize);

    let dir = std::env::temp_dir().join(format!("stream-demo-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rnd = |n: usize, norm: bool| -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                (if norm { 1.0 + 0.1 * u } else { 0.1 * u }).to_le_bytes()
            })
            .collect()
    };
    let mut ts: Vec<(String, Vec<usize>, Vec<u8>)> = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![v, h],
            rnd(v * h, false),
        ),
        ("model.norm.weight".into(), vec![h], rnd(h, true)),
        ("lm_head.weight".into(), vec![v, h], rnd(v * h, false)),
    ];
    for i in 0..layers as usize {
        for (p, shape, norm) in [
            ("attn_norm", vec![h], true),
            ("wq", vec![qd, h], false),
            ("wk", vec![kd, h], false),
            ("wv", vec![kd, h], false),
            ("wo", vec![h, qd], false),
            ("ffn_norm", vec![h], true),
            ("w_gate", vec![ff, h], false),
            ("w_up", vec![ff, h], false),
            ("w_down", vec![h, ff], false),
        ] {
            let n = shape.iter().product();
            ts.push((layer_name(Naming::Hf, i, p), shape, rnd(n, norm)));
        }
    }
    let mut hdr = serde_json::Map::new();
    let mut off = 0;
    for (n, shape, b) in &ts {
        hdr.insert(n.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [off, off + b.len()]}));
        off += b.len();
    }
    let hb = serde_json::to_vec(&hdr).unwrap();
    let mut f = (hb.len() as u64).to_le_bytes().to_vec();
    f.extend(hb);
    for t in &ts {
        f.extend(&t.2);
    }
    std::fs::write(&path, &f).unwrap();
    println!(
        "model: {layers} layers, {:.1} MB on disk",
        f.len() as f64 / 1e6
    );

    let src: Arc<dyn WeightFile> = Arc::new(SafeTensors::open(&path).unwrap());
    let fl = FileLayers::new(
        spec.clone(),
        src.clone(),
        Naming::Hf,
        false,
        LoadOptions::default(),
    )
    .unwrap();
    let lb = fl.layer_bytes(0) as u64;
    let outer = (2 * v * h * 4) as u64;
    let plan = plan_layers(
        layers as usize,
        lb,
        outer,
        1 << 20,
        outer + (1 << 20) + 3 * lb,
        1000.0,
        4,
    )
    .unwrap();
    println!(
        "plan for a budget of 3 layers: window {} layers, {} MB pool, decode ≈ {:.1} tok/s at 1000 MB/s",
        plan.window_layers,
        plan.budget_bytes / 1_000_000,
        plan.decode_tps.unwrap_or(0.0)
    );

    let prompt: Vec<usize> = (0..64).map(|i| (i * 13) % v).collect();
    for window in [1usize, 2, 4] {
        let slow = Arc::new(ThrottledLayers {
            inner: FileLayers::new(
                spec.clone(),
                src.clone(),
                Naming::Hf,
                false,
                LoadOptions::default(),
            )
            .unwrap(),
            latency: Duration::from_millis(ms),
        });
        let outer_w =
            StreamedModel::load_outer(&spec, &*src, Naming::Hf, LoadOptions::default()).unwrap();
        let mut sm =
            StreamedModel::from_parts(spec.clone(), outer_w, slow, window * lb as usize).unwrap();
        let mut st = sm.new_state();
        let t = Instant::now();
        let logits = sm.forward_batch(&mut st, &prompt).unwrap();
        let prefill = t.elapsed();
        let t = Instant::now();
        let mut tok = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |x| x.0);
        for _ in 0..8 {
            let l = sm.forward(&mut st, tok).unwrap();
            tok = l
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |x| x.0);
        }
        let dec = t.elapsed();
        let s = sm.stats();
        println!(
            "window {window}: prefill {:>6.0} ms, decode {:>5.2} tok/s, peak {:>5.1} MB of {:>5.1} MB budget, waited {:>5.0} ms for disk",
            prefill.as_secs_f64() * 1e3,
            8.0 / dec.as_secs_f64(),
            sm.peak_resident_bytes() as f64 / 1e6,
            sm.pool_budget() as f64 / 1e6,
            s.wait_secs * 1e3
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}
