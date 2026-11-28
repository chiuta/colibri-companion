//! Models that do not go through the dense loader: state-space and recurrent models (Mamba, Mamba-2,
//! RWKV-6/7) and attention/state-space hybrids, opened with `family::load_any`.
//!
//! What works here: `generate` and `chat` on a model folder or `.gguf` with a `tokenizer.json` (or a
//! tokenizer inside the GGUF), sampling with --temperature, --top-k, --top-p, --seed, --max-tokens.
//! What does not: the chat template (the text is continued as plain text, chat keeps the recurrent
//! state between turns), structured output, speculative decoding, receipts, sessions, KV policies,
//! layer streaming, the mesh, `serve`, and MLA / DeepSeek models (the engine has the attention
//! variant but not the whole model). Native RWKV checkpoints that ship a word-list vocabulary instead
//! of `tokenizer.json` cannot be tokenised here.
use crate::args::Args;
use crate::load::{Source, resolve};
use crate::runner::GenOut;
use crate::template::{Builtin, ChatTemplate};
use companion_engine::arch::{self, Support};
use companion_engine::family::{self, AnyModel, AnyState, LanguageModel, classify_family};
use companion_engine::loader::GgufFile;
use companion_engine::model::{GenerateOptions, Sampler, StopReason};
use companion_engine::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// Does this model need the family loader rather than the dense one?
pub fn wants(path: &Path) -> bool {
    match resolve(path) {
        Ok(Source::HfDir(d)) => {
            let Ok(t) = std::fs::read_to_string(d.join("config.json")) else {
                return false;
            };
            let Ok(j) = serde_json::from_str::<serde_json::Value>(&t) else {
                return false;
            };
            let mt = j.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
            classify_family(mt).is_some()
                || matches!(
                    arch::classify(mt),
                    Support::Recurrent | Support::Hybrid | Support::Native(_)
                )
        }
        Ok(Source::Gguf(f)) => GgufFile::open(&f)
            .ok()
            .and_then(|g| {
                g.gguf.architecture().map(|a| {
                    classify_family(a).is_some()
                        || matches!(
                            arch::classify(a),
                            Support::Recurrent | Support::Hybrid | Support::Native(_)
                        )
                })
            })
            .unwrap_or(false),
        Err(_) => false,
    }
}

pub struct Fam {
    pub model: AnyModel,
    pub tok: Tokenizer,
    pub stop: Vec<usize>,
    pub name: String,
}

/// Just the tokenizer (and a display name) of a model folder or GGUF.
pub fn load_tokenizer(path: &Path) -> Result<(Tokenizer, String), String> {
    Ok(match resolve(path)? {
        Source::HfDir(d) => {
            let tp = d.join("tokenizer.json");
            if !tp.is_file() {
                return Err("tokenizer.json is missing (native RWKV word-list vocabularies are not supported)".into());
            }
            (
                Tokenizer::from_file(&tp).map_err(|e| format!("tokenizer.json: {e}"))?,
                d.file_name()
                    .map_or("model".into(), |n| n.to_string_lossy().into_owned()),
            )
        }
        Source::Gguf(f) => {
            let g = GgufFile::open(&f).map_err(|e| e.to_string())?;
            let beside = f.with_file_name("tokenizer.json");
            let tok = match Tokenizer::from_gguf(&g.gguf) {
                Ok(t) => t,
                // a GGUF without its own vocabulary can use a tokenizer.json lying next to it
                Err(e) if beside.is_file() => Tokenizer::from_file(&beside)
                    .map_err(|e2| format!("tokenizer in the GGUF: {e}; tokenizer.json: {e2}"))?,
                Err(e) => return Err(format!("tokenizer in the GGUF: {e}")),
            };
            (
                tok,
                f.file_stem()
                    .map_or("model".into(), |n| n.to_string_lossy().into_owned()),
            )
        }
    })
}

pub fn open(path: &Path) -> Result<Fam, String> {
    let (tok, name) = load_tokenizer(path)?;
    let mut model = family::load_any(path).map_err(|e| e.to_string())?;
    crate::gpu::accelerate_family(&mut model);
    let vocab = model.vocab_size();
    let mut stop = vec![];
    if let Some(e) = tok.eos {
        stop.push(e as usize);
    }
    if let Ok(Source::Gguf(p)) = resolve(path)
        && let Ok(g) = GgufFile::open(&p)
    {
        for k in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
            if let Some(i) = g.gguf.meta.get(k).and_then(|v| v.as_u64())
                && !stop.contains(&(i as usize))
            {
                stop.push(i as usize);
            }
        }
    }
    for t in [
        "<|endoftext|>",
        "<|im_end|>",
        "</s>",
        "<end_of_turn>",
        "<turn|>",
        "<|return|>",
        "<|call|>",
        "<|eot_id|>",
    ] {
        if let Some(i) = tok.token_to_id(t)
            && !stop.contains(&(i as usize))
        {
            stop.push(i as usize);
        }
    }
    stop.retain(|&i| vocab == 0 || i < vocab);
    Ok(Fam {
        model,
        tok,
        stop,
        name,
    })
}

/// The chat template of a model folder or GGUF, the text of its beginning and end tokens, any
/// warnings, and the architecture name.
#[allow(clippy::type_complexity)]
pub fn template_of(
    path: &Path,
    tok: &Tokenizer,
) -> Result<(ChatTemplate, String, String, Vec<String>, String), String> {
    let (src, arch, cfg): (Option<String>, String, Option<serde_json::Value>) = match resolve(path)?
    {
        Source::Gguf(f) => {
            let g = GgufFile::open(&f).map_err(|e| e.to_string())?;
            (
                g.gguf
                    .meta
                    .get("tokenizer.chat_template")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                g.gguf.architecture().unwrap_or("").to_string(),
                None,
            )
        }
        Source::HfDir(d) => {
            let read = |n: &str| {
                std::fs::read_to_string(d.join(n))
                    .ok()
                    .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            };
            let cfg = read("tokenizer_config.json");
            let mut t = cfg.as_ref().and_then(crate::load::template_from_config);
            if t.is_none() {
                t = std::fs::read_to_string(d.join("chat_template.jinja")).ok();
            }
            let arch = read("config.json")
                .and_then(|j| {
                    j.get("model_type")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                })
                .unwrap_or_default();
            (t, arch, cfg)
        }
    };
    let text = |key: &str, id: Option<u32>| -> String {
        cfg.as_ref()
            .and_then(|c| c.get(key))
            .and_then(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Object(o) => o
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                _ => None,
            })
            .or_else(|| id.and_then(|i| tok.id_to_token(i).map(str::to_string)))
            .unwrap_or_default()
    };
    let (bos, eos) = (text("bos_token", tok.bos), text("eos_token", tok.eos));
    let builtin = Builtin::detect(&arch, &|t| tok.token_to_id(t).is_some());
    let (template, warn) = ChatTemplate::from_source(src.as_deref(), builtin);
    Ok((template, bos, eos, warn.into_iter().collect(), arch))
}

fn sampler_of(args: &Args) -> Result<(Sampler, usize), String> {
    let p = crate::runner::sampler_params(args, 0.7)?;
    let s = Sampler::new(p.temperature, p.top_k, p.seed.unwrap_or(1)).with_top_p(p.top_p);
    Ok((s, args.num("max-tokens")?.unwrap_or(512)))
}

fn model_path(args: &Args, what: &str) -> Result<PathBuf, String> {
    args.pos
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{what}: give a model folder or .gguf file"))
}

/// One generation on `st`: returns the text, the tokens and how it ended.
pub fn run(
    f: &Fam,
    st: &mut AnyState,
    ids: &[usize],
    sampler: &mut Sampler,
    max_tokens: usize,
    cancel: &AtomicBool,
    mut live: impl FnMut(&str),
) -> Result<GenOut, String> {
    let opts = GenerateOptions {
        max_tokens,
        stop_tokens: f.stop.clone(),
        cancel: Some(cancel),
    };
    let mut dec = f.tok.decoder();
    dec.skip_special = true;
    let mut tokens = vec![];
    let mut text = String::new();
    let stats = family::generate(&f.model, st, ids, sampler, &opts, &mut |t| {
        tokens.push(t);
        let piece = dec.push(t as u32);
        if !piece.is_empty() {
            live(&piece);
            text.push_str(&piece);
        }
        true
    })
    .map_err(|e| e.to_string())?;
    let tail = dec.finish();
    if !tail.is_empty() {
        live(&tail);
        text.push_str(&tail);
    }
    let finish = match stats.stop {
        StopReason::StopToken => "stop",
        StopReason::Cancelled => "cancelled",
        StopReason::ContextFull => "context",
        _ => "length",
    };
    Ok(GenOut {
        text,
        tokens,
        finish,
        prefill_secs: stats.prefill_secs,
        decode_secs: stats.decode_secs,
        prompt_tokens: stats.prompt_tokens,
    })
}

fn prompt_of(args: &Args) -> Result<String, String> {
    match (args.get("prompt"), args.get("prompt-file")) {
        (Some(p), _) => Ok(p.to_string()),
        (None, Some(f)) => std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}")),
        _ => args
            .pos
            .get(1)
            .cloned()
            .ok_or_else(|| "generate: give --prompt TEXT (or --prompt-file F)".to_string()),
    }
}

fn refuse(args: &Args) -> Result<(), String> {
    crate::fast::refuse(args, false, "state-space and hybrid models")?;
    for f in [
        "lookahead",
        "self-spec",
        "no-lock",
        "keep-steps",
        "leaves",
        "compress",
    ] {
        if args.flag(f) {
            return Err(format!(
                "--{f} is not available for state-space and hybrid models"
            ));
        }
    }
    for k in [
        "draft",
        "skip-layers",
        "lora",
        "json-schema",
        "regex",
        "grammar",
        "ram-budget",
        "kv-budget",
        "citations",
        "kv-sinks",
        "receipt-topk",
        "session-cap",
    ] {
        if args.get(k).is_some() {
            return Err(format!(
                "--{k} is not available for state-space and hybrid models"
            ));
        }
    }
    Ok(())
}

pub fn generate(args: &Args) -> Result<(), String> {
    refuse(args)?;
    let path = model_path(args, "generate")?;
    let f = open(&path)?;
    eprintln!(
        "{}: {} model through the family loader (no chat template: the prompt is continued as plain text)",
        f.name,
        match &f.model {
            AnyModel::Ssm(m) => format!("{:?}", m.kind()),
            _ => "hybrid / attention".into(),
        }
    );
    let prompt = prompt_of(args)?;
    let ids: Vec<usize> = f
        .tok
        .encode(&prompt, true)
        .into_iter()
        .map(|i| i as usize)
        .collect();
    if ids.is_empty() {
        return Err("the prompt is empty".into());
    }
    let (mut sampler, max_tokens) = sampler_of(args)?;
    let mut st = f.model.new_state();
    let cancel = AtomicBool::new(false);
    let o = run(&f, &mut st, &ids, &mut sampler, max_tokens, &cancel, |s| {
        crate::out!("{s}");
    })?;
    crate::outln!();
    if !args.flag("quiet") {
        eprintln!("{}", crate::cmds::finish_line(&o));
    }
    Ok(())
}

pub fn chat(args: &Args) -> Result<(), String> {
    use std::io::BufRead;
    refuse(args)?;
    let path = model_path(args, "chat")?;
    let f = open(&path)?;
    let (mut sampler, max_tokens) = sampler_of(args)?;
    let mut st = f.model.new_state();
    let cancel = AtomicBool::new(false);
    let mut first = true;
    eprintln!(
        "Plain-text chat (no chat template). The model's recurrent state carries the conversation. /reset clears it, /exit quits."
    );
    let stdin = std::io::stdin();
    loop {
        eprint!("\n> ");
        let mut line = String::new();
        if stdin
            .lock()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?
            == 0
        {
            break;
        }
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                st = f.model.new_state();
                first = true;
                sampler.clear_history();
                eprintln!("(conversation cleared)");
                continue;
            }
            l => {
                let text = format!(
                    "{}User: {l}\nAssistant:",
                    if first {
                        args.get("system")
                            .map_or(String::new(), |s| format!("{s}\n\n"))
                    } else {
                        "\n".into()
                    }
                );
                let ids: Vec<usize> = f
                    .tok
                    .encode(&text, first)
                    .into_iter()
                    .map(|i| i as usize)
                    .collect();
                first = false;
                let mut shown = String::new();
                let mut stop_at_user = false;
                let r = run(&f, &mut st, &ids, &mut sampler, max_tokens, &cancel, |s| {
                    if stop_at_user {
                        return;
                    }
                    shown.push_str(s);
                    if let Some(i) = shown.find("\nUser:") {
                        let keep = shown[..i].to_string();
                        stop_at_user = true;
                        crate::out!(
                            "{}",
                            &keep[keep.len().saturating_sub(s.len().min(keep.len()))..]
                        );
                    } else {
                        crate::out!("{s}");
                    }
                });
                crate::outln!();
                if let Err(e) = r {
                    eprintln!("{e}");
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;

    #[test]
    fn only_state_space_and_hybrid_models_take_this_path() {
        let dense = tmpdir("fam-dense");
        write_model(&dense, true);
        assert!(!wants(&dense));
        let ssm = tmpdir("fam-ssm");
        std::fs::write(
            ssm.join("config.json"),
            r#"{"model_type":"mamba","hidden_size":16}"#,
        )
        .unwrap();
        assert!(wants(&ssm));
        let jamba = tmpdir("fam-jamba");
        std::fs::write(jamba.join("config.json"), r#"{"model_type":"jamba"}"#).unwrap();
        assert!(wants(&jamba) || !wants(&jamba)); // classified by the engine; must not panic either way
        assert!(!wants(Path::new("/no/such/model")));
        std::fs::remove_dir_all(dense).unwrap();
        std::fs::remove_dir_all(ssm).unwrap();
        std::fs::remove_dir_all(jamba).unwrap();
    }

    #[test]
    fn missing_pieces_are_reported_not_panicked() {
        let ssm = tmpdir("fam-ssm2");
        std::fs::write(ssm.join("config.json"), r#"{"model_type":"mamba"}"#).unwrap();
        let a = Args::parse(&[
            ssm.to_string_lossy().to_string(),
            "--prompt".into(),
            "hi".into(),
        ]);
        assert!(generate(&a).unwrap_err().contains("tokenizer"));
        let bad = Args::parse(&[
            ssm.to_string_lossy().to_string(),
            "--prompt".into(),
            "hi".into(),
            "--min-p".into(),
            "0.1".into(),
        ]);
        assert!(generate(&bad).unwrap_err().contains("--min-p"));
        let spec = Args::parse(&[
            ssm.to_string_lossy().to_string(),
            "--prompt".into(),
            "hi".into(),
            "--lookahead".into(),
        ]);
        assert!(generate(&spec).unwrap_err().contains("--lookahead"));
        // options of the attention-model path are refused, never silently ignored
        for (opt, val) in [
            ("--citations", Some("x")),
            ("--kv-sinks", Some("4")),
            ("--receipt-topk", Some("2")),
            ("--session-cap", Some("1m")),
            ("--no-lock", None),
            ("--keep-steps", None),
            ("--leaves", None),
            ("--compress", None),
        ] {
            let mut v = vec![
                ssm.to_string_lossy().to_string(),
                "--prompt".into(),
                "hi".into(),
                opt.to_string(),
            ];
            v.extend(val.map(String::from));
            let e = generate(&Args::parse(&v)).unwrap_err();
            assert!(e.contains(opt), "{opt}: {e}");
        }
        std::fs::remove_dir_all(ssm).unwrap();
    }
}

#[cfg(test)]
mod gemma4_route_tests {
    use super::wants;
    use companion_engine::gguf_write::{GgufWriter, MetaValue, TensorDesc};
    use companion_engine::quant::DType;

    fn file(arch: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("famrun-g4-{}-{arch}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("m.gguf");
        let meta = vec![("general.architecture".to_string(), MetaValue::str(arch))];
        let t = vec![TensorDesc::new("x", &[1], DType::F32)];
        GgufWriter::write_all(std::fs::File::create(&p).unwrap(), &meta, t, |_, _| {
            Ok(vec![0u8; 4])
        })
        .unwrap();
        p
    }

    #[test]
    fn gemma4_files_go_to_the_family_loader_and_other_dense_ones_do_not() {
        assert!(wants(&file("gemma4")));
        assert!(wants(&file("mamba")));
        assert!(!wants(&file("llama")));
        assert!(!wants(&file("gemma3")));
    }
}
