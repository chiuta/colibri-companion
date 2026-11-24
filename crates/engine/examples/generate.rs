//! Generate text from a local model with the engine.
//!
//!   cargo run --release --example generate -- <model.gguf | model-dir> --prompt "Hello" \
//!       [--tokenizer tokenizer.json] [--max-tokens 128] [--temp 0.8] [--top-k 40] [--top-p 0.95] \
//!       [--repeat-penalty 1.1] [--seed 1] [--kv f32|f16|q8] [--no-quant] [--no-mmap] [--no-bos]
//!
//! A GGUF file brings its own tokenizer; a Hugging Face folder needs `config.json`, safetensors
//! (single or sharded) and `tokenizer.json`.
use companion_engine::arch::{Support, from_gguf, from_hf_config};
use companion_engine::loader::{GgufFile, LoadOptions, Naming, SafeTensors, load_dense_with};
use companion_engine::model::{GenerateOptions, KvKind, Sampler};
use companion_engine::tokenizer::Tokenizer;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut model: Option<PathBuf> = None;
    let (mut prompt, mut tokenizer) = (String::from("Once upon a time"), None::<PathBuf>);
    let (mut max_tokens, mut temp, mut top_k, mut top_p, mut rep, mut seed) =
        (128usize, 0.8f32, 40usize, 0.95f32, 1.0f32, 1u64);
    let (mut kv, mut quant, mut mmap, mut bos) = (KvKind::F32, true, true, true);
    while let Some(a) = args.next() {
        let mut val = |n: &str| {
            args.next()
                .unwrap_or_else(|| die(format!("{n} needs a value")))
        };
        let num = |n: &str, v: String| {
            v.parse::<f64>()
                .unwrap_or_else(|_| die(format!("{n}: `{v}` is not a number")))
        };
        match a.as_str() {
            "--prompt" => prompt = val("--prompt"),
            "--tokenizer" => tokenizer = Some(val("--tokenizer").into()),
            "--max-tokens" => max_tokens = num("--max-tokens", val("--max-tokens")) as usize,
            "--temp" => temp = num("--temp", val("--temp")) as f32,
            "--top-k" => top_k = num("--top-k", val("--top-k")) as usize,
            "--top-p" => top_p = num("--top-p", val("--top-p")) as f32,
            "--repeat-penalty" => rep = num("--repeat-penalty", val("--repeat-penalty")) as f32,
            "--seed" => seed = num("--seed", val("--seed")) as u64,
            "--kv" => {
                kv = match val("--kv").as_str() {
                    "f32" => KvKind::F32,
                    "f16" => KvKind::F16,
                    "q8" => KvKind::Q8,
                    o => die(format!("--kv: `{o}` is not f32, f16 or q8")),
                }
            }
            "--no-quant" => quant = false,
            "--no-mmap" => mmap = false,
            "--no-bos" => bos = false,
            "-h" | "--help" => {
                println!(
                    "usage: generate <model.gguf | model-dir> --prompt TEXT [options]; see the file header"
                );
                return;
            }
            p if !p.starts_with("--") && model.is_none() => model = Some(p.into()),
            o => die(format!("unknown argument `{o}`")),
        }
    }
    let model = model.unwrap_or_else(|| die("give a model file or folder"));
    let t0 = Instant::now();
    let opts = LoadOptions {
        prefer_quantized: quant,
        mmap,
        ..Default::default()
    };
    let (m, tok) = if model.is_dir() {
        let cfg = std::fs::read_to_string(model.join("config.json"))
            .unwrap_or_else(|e| die(format!("config.json: {e}")));
        let spec = from_hf_config(&cfg).unwrap_or_else(|e| die(e));
        check(&spec.support, &spec.family);
        let src = SafeTensors::open(&model).unwrap_or_else(|e| die(e));
        let m = load_dense_with(spec, &src, Naming::Hf, false, opts).unwrap_or_else(|e| die(e));
        let tp = tokenizer.unwrap_or_else(|| model.join("tokenizer.json"));
        (
            m,
            Tokenizer::from_file(&tp).unwrap_or_else(|e| die(format!("{}: {e}", tp.display()))),
        )
    } else {
        let f = GgufFile::open(&model).unwrap_or_else(|e| die(e));
        let spec = from_gguf(&f.gguf).unwrap_or_else(|e| die(e));
        check(&spec.support, &spec.family);
        let permuted = f.gguf.architecture() == Some("llama");
        let m = load_dense_with(spec, &f, Naming::Gguf, permuted, opts).unwrap_or_else(|e| die(e));
        let t = match tokenizer {
            Some(p) => {
                Tokenizer::from_file(&p).unwrap_or_else(|e| die(format!("{}: {e}", p.display())))
            }
            None => Tokenizer::from_gguf(&f.gguf)
                .unwrap_or_else(|e| die(format!("tokenizer in the GGUF: {e}"))),
        };
        (m, t)
    };
    eprintln!(
        "loaded {} layers, {:.1} MB of weights in {:.2}s",
        m.spec.layers,
        m.weight_bytes() as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    let ids: Vec<usize> = tok
        .encode(&prompt, bos)
        .into_iter()
        .map(|i| i as usize)
        .collect();
    if ids.is_empty() {
        die("the prompt is empty");
    }
    let mut st = m.new_state_with(kv);
    let mut sampler = Sampler::new(temp, top_k, seed)
        .with_top_p(top_p)
        .with_repetition_penalty(rep, 64);
    let gen_opts = GenerateOptions {
        max_tokens,
        stop_tokens: tok.eos.map(|e| vec![e as usize]).unwrap_or_default(),
        cancel: None,
    };
    let mut dec = tok.decoder();
    print!("{prompt}");
    let out = &mut std::io::stdout();
    let stats = m
        .generate(&mut st, &ids, &mut sampler, &gen_opts, &mut |t| {
            let s = dec.push(t as u32);
            let _ = write!(out, "{s}");
            let _ = out.flush();
            true
        })
        .unwrap_or_else(|e| die(e));
    println!("{}", dec.finish());
    eprintln!(
        "prefill {} tokens at {:.1} tok/s, decode {} tokens at {:.1} tok/s ({:?}), KV cache {:.2} MB",
        stats.prompt_tokens,
        stats.prefill_tps(),
        stats.generated,
        stats.decode_tps(),
        stats.stop,
        st.kv_bytes() as f64 / 1e6
    );
}

fn check(s: &Support, family: &str) {
    if let Support::Planned(why) = s {
        die(format!("`{family}` is not supported yet: {why}"));
    }
}
