use companion_engine::sampling::{Rng, SamplerChain, SamplerParams};
use std::time::Instant;

fn main() {
    let n = 150_000;
    let mut r = Rng::new(1);
    let l: Vec<f32> = (0..n).map(|_| r.f32() * 20.0 - 10.0).collect();
    let hist: Vec<u32> = (0..512).map(|i| (i * 7919 % n) as u32).collect();
    let cases: [(&str, &str); 6] = [
        ("greedy", r#"{"temperature":0}"#),
        ("default (k40,p.95,minp.05)", "{}"),
        ("temp only", r#"{"top_k":0,"top_p":1,"min_p":0}"#),
        ("top_p only", r#"{"top_k":0,"top_p":0.9,"min_p":0}"#),
        ("full (penalties+dry+typ+xtc)", r#"{"repeat_penalty":1.1,"dry_multiplier":0.8,"typical_p":0.9,"xtc_probability":0.5,"seed":1}"#),
        ("mirostat2", r#"{"mirostat":2,"top_k":0}"#),
    ];
    for (name, js) in cases {
        let mut c = SamplerChain::new(SamplerParams::from_json(js).unwrap());
        for _ in 0..20 { c.sample(&l, &hist); }
        let it = 300;
        let t = Instant::now();
        for _ in 0..it { std::hint::black_box(c.sample(&l, &hist)); }
        let s = t.elapsed().as_secs_f64();
        println!("{name:<32} {:>9.0} samples/s  ({:.0} us)", it as f64 / s, s / it as f64 * 1e6);
    }
}
