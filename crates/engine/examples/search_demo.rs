//! Test-time compute on a local model: lookahead decoding, beam search, best-of-N, self-consistency.
//!
//!   cargo run --release --example search_demo -- <model.gguf | model-dir> --prompt "2+2=" \
//!       --mode lookup|beam|vote [--tokenizer tokenizer.json] [--max-tokens 64] [--width 4] [--n 8] \
//!       [--temp 0.8] [--seed 1] [--no-bos]
//!
//! `lookup` prints the plain and the lookahead run side by side (same tokens, fewer passes when the
//! text repeats); `beam` prints the beams; `vote` draws N samples and takes the majority answer
//! (the last number in each sample).
use companion_engine::arch::{from_gguf, from_hf_config, Support};
use companion_engine::loader::{load_dense_with, GgufFile, LoadOptions, Naming, SafeTensors};
use companion_engine::search::{agreement, beam_search, last_number, lookup_generate, plain_generate, sample_n, self_consistency, BeamConfig, LookupConfig, MeanLogprob, SampleConfig};
use companion_engine::spec::SamplingCfg;
use companion_engine::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::time::Instant;

fn die(msg: impl std::fmt::Display) -> ! { eprintln!("error: {msg}"); std::process::exit(1) }

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut model, mut tokenizer) = (None::<PathBuf>, None::<PathBuf>);
    let (mut prompt, mut mode) = (String::from("Once upon a time"), String::from("lookup"));
    let (mut max_tokens, mut width, mut n, mut temp, mut seed, mut bos) = (64usize, 4usize, 8usize, 0.8f32, 1u64, true);
    while let Some(a) = args.next() {
        let mut val = |k: &str| args.next().unwrap_or_else(|| die(format!("{k} needs a value")));
        let num = |k: &str, v: String| v.parse::<f64>().unwrap_or_else(|_| die(format!("{k}: `{v}` is not a number")));
        match a.as_str() {
            "--prompt" => prompt = val("--prompt"),
            "--mode" => mode = val("--mode"),
            "--tokenizer" => tokenizer = Some(val("--tokenizer").into()),
            "--max-tokens" => max_tokens = num("--max-tokens", val("--max-tokens")) as usize,
            "--width" => width = num("--width", val("--width")) as usize,
            "--n" => n = num("--n", val("--n")) as usize,
            "--temp" => temp = num("--temp", val("--temp")) as f32,
            "--seed" => seed = num("--seed", val("--seed")) as u64,
            "--no-bos" => bos = false,
            "-h" | "--help" => { println!("usage: search_demo <model> --prompt TEXT --mode lookup|beam|vote; see the file header"); return; }
            p if !p.starts_with("--") && model.is_none() => model = Some(p.into()),
            o => die(format!("unknown argument `{o}`")),
        }
    }
    let model = model.unwrap_or_else(|| die("give a model file or folder"));
    let opts = LoadOptions { prefer_quantized: true, mmap: true, ..Default::default() };
    let (m, tok) = if model.is_dir() {
        let cfg = std::fs::read_to_string(model.join("config.json")).unwrap_or_else(|e| die(format!("config.json: {e}")));
        let spec = from_hf_config(&cfg).unwrap_or_else(|e| die(e));
        check(&spec.support, &spec.family);
        let src = SafeTensors::open(&model).unwrap_or_else(|e| die(e));
        let m = load_dense_with(spec, &src, Naming::Hf, false, opts).unwrap_or_else(|e| die(e));
        let tp = tokenizer.unwrap_or_else(|| model.join("tokenizer.json"));
        (m, Tokenizer::from_file(&tp).unwrap_or_else(|e| die(format!("{}: {e}", tp.display()))))
    } else {
        let f = GgufFile::open(&model).unwrap_or_else(|e| die(e));
        let spec = from_gguf(&f.gguf).unwrap_or_else(|e| die(e));
        check(&spec.support, &spec.family);
        let permuted = f.gguf.architecture() == Some("llama");
        let m = load_dense_with(spec, &f, Naming::Gguf, permuted, opts).unwrap_or_else(|e| die(e));
        let t = match tokenizer {
            Some(p) => Tokenizer::from_file(&p).unwrap_or_else(|e| die(format!("{}: {e}", p.display()))),
            None => Tokenizer::from_gguf(&f.gguf).unwrap_or_else(|e| die(format!("tokenizer in the GGUF: {e}"))),
        };
        (m, t)
    };
    let ids = tok.encode(&prompt, bos);
    if ids.is_empty() { die("the prompt is empty"); }
    let stop: Vec<u32> = tok.eos.map(|e| vec![e]).unwrap_or_default();
    let text = |t: &[u32]| { let mut d = tok.decoder(); let mut s: String = t.iter().map(|&x| d.push(x)).collect(); s.push_str(&d.finish()); s };
    match mode.as_str() {
        "lookup" => {
            let cfg = LookupConfig { max_new: max_tokens, stop, sampling: SamplingCfg::greedy(), seed, ..Default::default() };
            let t = Instant::now();
            let plain = plain_generate(&m, &ids, &cfg).unwrap_or_else(|e| die(e));
            let tp = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let fast = lookup_generate(&m, &ids, &cfg).unwrap_or_else(|e| die(e));
            let tf = t.elapsed().as_secs_f64();
            println!("{prompt}{}", text(&fast.tokens));
            eprintln!("same tokens: {}", plain.tokens == fast.tokens);
            eprintln!("plain:  {} tokens in {} passes, {tp:.2}s", plain.stats.emitted, plain.stats.passes);
            eprintln!("lookup: {} tokens in {} passes, {tf:.2}s, acceptance {:.0}%, {:.2} tokens/pass",
                fast.stats.emitted, fast.stats.passes, fast.stats.acceptance_rate() * 100.0, fast.stats.tokens_per_pass());
        }
        "beam" => {
            let cfg = BeamConfig { width, max_new: max_tokens, stop, length_penalty: 0.6, ..Default::default() };
            for (i, h) in beam_search(&m, &ids, &cfg).unwrap_or_else(|e| die(e)).iter().enumerate() {
                println!("#{i} logprob {:.2} score {:.3}{}: {}{}", h.logprob, h.score, if h.stopped { " (stopped)" } else { "" }, prompt, text(&h.tokens));
            }
        }
        "vote" => {
            let cfg = SampleConfig { max_new: max_tokens, stop, sampling: SamplingCfg::sampled(temp, 40, 0.95), seed };
            let samples = sample_n(&m, &ids, n, &cfg).unwrap_or_else(|e| die(e));
            let votes = self_consistency(&samples, &|s| last_number(&text(&s.tokens)), Some(&MeanLogprob));
            for v in &votes { println!("{:>3} x {}", v.count, v.answer); }
            if votes.is_empty() { println!("no sample contained a number"); } else { println!("agreement {:.0}%", agreement(&votes) * 100.0); }
        }
        o => die(format!("unknown mode `{o}` (lookup, beam, vote)")),
    }
}

fn check(s: &Support, family: &str) {
    if let Support::Planned(why) = s { die(format!("`{family}` is not supported yet: {why}")); }
}
