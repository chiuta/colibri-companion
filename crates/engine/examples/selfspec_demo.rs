//! Self-speculative decoding demo on a synthetic model (no checkpoint needed):
//! calibrate a layer-skip set, decode with and without it, compare outputs and speed.
//!
//! `cargo run --release -p companion-engine --example selfspec_demo -- [layers] [tokens]`
use companion_engine::arch::Spec;
use companion_engine::model::{Layer, Model, Weights};
use companion_engine::selfspec::*;
use std::time::Instant;

fn rnd(n: usize, s: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            ((*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.3
        })
        .collect()
}

/// A random model whose upper layers are nudged towards the identity, the way trained networks
/// have layers that matter little: that is what makes layer skipping pay off.
fn model(layers: u64) -> Model {
    let (h, hd, ff, v) = (256usize, 32usize, 512usize, 512usize);
    let mut sp = Spec::dense(
        "llama", layers, h as u64, 8, 8, hd as u64, ff as u64, v as u64, 1024,
    );
    sp.tied_embeddings = false;
    let mut s = 0x1234_5678_9abc_def1u64;
    let ls = (0..layers as usize)
        .map(|i| {
            let damp = if i == 0 || i + 1 == layers as usize {
                1.0
            } else {
                0.15
            };
            let m =
                |n: usize, s: &mut u64| rnd(n, s).into_iter().map(|x| x * damp).collect::<Vec<_>>();
            Layer {
                attn_norm: vec![1.0; h],
                ffn_norm: vec![1.0; h],
                wq: m(h * h, &mut s).into(),
                wk: m(h * h, &mut s).into(),
                wv: m(h * h, &mut s).into(),
                wo: m(h * h, &mut s).into(),
                w_gate: m(ff * h, &mut s).into(),
                w_up: m(ff * h, &mut s).into(),
                w_down: m(h * ff, &mut s).into(),
                ..Default::default()
            }
        })
        .collect();
    let w = Weights {
        embed: rnd(v * h, &mut s).into(),
        layers: ls,
        final_norm: vec![1.0; h],
        final_norm_b: None,
        lm_head: Some(rnd(v * h, &mut s).into()),
        lm_head_b: None,
        experts: None,
    };
    Model::new(sp, w).expect("model")
}

fn main() {
    let mut a = std::env::args().skip(1);
    let layers: u64 = a.next().and_then(|x| x.parse().ok()).unwrap_or(16);
    let tokens: usize = a.next().and_then(|x| x.parse().ok()).unwrap_or(96);
    let m = model(layers);
    let prompts: Vec<Vec<usize>> = (0..3)
        .map(|j| (0..16).map(|i| (i * 31 + j * 17 + 5) % 512).collect())
        .collect();
    let t = Instant::now();
    let cal = calibrate(
        &m,
        &prompts,
        &CalibrateOptions {
            max_draft: 8,
            ..CalibrateOptions::default()
        },
    )
    .expect("calibrate");
    println!(
        "calibration: {:.2}s, skip {:?}, agreement {:.2}, draft cost {:.2}, best k {}, estimated speed-up x{:.2}{}",
        t.elapsed().as_secs_f64(),
        cal.skip.skipped(),
        cal.alpha,
        cal.draft_cost,
        cal.best_k,
        cal.est_speedup,
        if cal.worthwhile {
            ""
        } else {
            " (not worthwhile)"
        }
    );
    for s in &cal.trace {
        println!(
            "  + layer {:>2}: agreement {:.2} cost {:.2} k {} est x{:.2}",
            s.added, s.alpha, s.cost, s.best_k, s.speedup
        );
    }

    let prompt = prompts[0].clone();
    let mut cfg = SelfSpecConfig::new(cal.skip.clone());
    cfg.max_tokens = tokens;
    cfg.max_draft = 8;
    cfg.start_draft = cal.best_k.max(2);
    let (mut s1, mut s2) = (m.new_state(), m.new_state());
    let t = Instant::now();
    let plain = generate_plain(&m, &mut s1, &prompt, &cfg).expect("plain");
    let tp = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let spec = generate(&m, &mut s2, &prompt, &cfg).expect("spec");
    let ts = t.elapsed().as_secs_f64();
    println!("identical output: {}", plain.tokens == spec.tokens);
    println!(
        "plain: {:.1} tok/s | self-speculative: {:.1} tok/s | measured x{:.2}",
        plain.tokens.len() as f64 / tp,
        spec.tokens.len() as f64 / ts,
        tp / ts
    );
    let st = spec.stats;
    println!(
        "rounds {} drafted {} accepted {} ({:.0}%) plain steps {} tokens per full pass {:.2}",
        st.rounds,
        st.drafted,
        st.accepted,
        100.0 * st.acceptance_rate(),
        st.plain_steps,
        st.tokens_per_pass()
    );
    println!(
        "estimator on this run: x{:.2}",
        st.estimated_speedup(&SpeedModel::for_model(&m, &cal.skip, 1.0))
    );
}
