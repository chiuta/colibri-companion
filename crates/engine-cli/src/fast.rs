//! Faster decoding that keeps the answer the same: speculative decoding with a small draft model
//! (`--draft MODEL`), with the prompt itself as the source of guesses (`--lookahead`), or with the
//! model's own early layers (`--self-spec`).
//!
//! What is guaranteed, and what is not:
//! * Greedy (`--temperature 0`): the tokens are exactly those plain greedy decoding gives. The
//!   engine's tests check this bit for bit on synthetic models; on a real model a near-tie in the
//!   logits can flip a token because a batched pass rounds slightly differently from a single step.
//! * Sampled (`--temperature > 0`): the distribution of the output is the one plain sampling with the
//!   same temperature, top-k and top-p gives (standard speculative rejection sampling). Individual
//!   runs differ from plain sampling for the same seed.
//! * Only temperature, top-k, top-p and the seed are supported. The other sampler options, structured
//!   output, quote locks, receipts, sessions, KV eviction, streaming from disk and the mesh are
//!   refused rather than silently ignored.
//! * Text appears when the run ends, not token by token.
//! * Speed-up depends on how well the guesses match. Nothing here was timed on a real model.
use crate::args::Args;
use crate::cmds;
use crate::load::{self, Prepared};
use crate::runner::GenOut;
use companion_engine::sampling::SamplerParams;
use companion_engine::search::{self, LookupConfig, ModelVerifier, SeqModel};
use companion_engine::selfspec::{self, CalibrateOptions, SelfSpecConfig, SkipSet};
use companion_engine::spec::{self, AdaptiveLen, DraftModel, SamplingCfg, SpecConfig};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Which kind of speculation the command line asks for.
#[derive(Clone, Debug, PartialEq)]
pub enum Fast {
    /// A separate, smaller model that shares the tokenizer.
    Draft(PathBuf),
    /// Prompt lookup: guesses are copied from text already seen.
    Lookahead,
    /// The model's own layers, with some skipped, draft; the full model verifies.
    SelfSpec(Skip),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Skip {
    Auto,
    Layers(Vec<usize>),
}

/// `--skip-layers auto` or `--skip-layers 4,6,8-11` (layer numbers start at 0).
pub fn parse_skip(s: &str) -> Result<Skip, String> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("auto") {
        return Ok(Skip::Auto);
    }
    let mut v = vec![];
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let bad = || format!("--skip-layers: `{part}` is not a layer number or a range like 8-11");
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) = (
                    a.trim().parse().map_err(|_| bad())?,
                    b.trim().parse().map_err(|_| bad())?,
                );
                if a > b || b - a > 4096 {
                    return Err(bad());
                }
                v.extend(a..=b);
            }
            None => v.push(part.parse().map_err(|_| bad())?),
        }
    }
    if v.is_empty() {
        return Err("--skip-layers: give `auto` or a list such as 4,6,8-11".into());
    }
    v.sort_unstable();
    v.dedup();
    Ok(Skip::Layers(v))
}

/// The speculation the options ask for, if any (at most one kind).
pub fn from_args(a: &Args) -> Result<Option<Fast>, String> {
    let (d, l, s) = (
        a.get("draft"),
        a.flag("lookahead"),
        a.flag("self-spec") || a.get("skip-layers").is_some(),
    );
    if [d.is_some(), l, s].iter().filter(|x| **x).count() > 1 {
        return Err(
            "--draft, --lookahead and --self-spec are three different ways to speculate; use one"
                .into(),
        );
    }
    if let Some(p) = d {
        return Ok(Some(Fast::Draft(PathBuf::from(p))));
    }
    if l {
        return Ok(Some(Fast::Lookahead));
    }
    if s {
        return Ok(Some(Fast::SelfSpec(match a.get("skip-layers") {
            Some(x) => parse_skip(x)?,
            None => Skip::Auto,
        })));
    }
    for k in ["draft-len", "ngram-max", "ngram-min"] {
        if a.get(k).is_some() {
            return Err(format!(
                "--{k} only makes sense with --draft, --lookahead or --self-spec"
            ));
        }
    }
    Ok(None)
}

/// Options that speculation cannot honour: it refuses them instead of ignoring them.
pub fn check_compatible(a: &Args, constrained: bool) -> Result<(), String> {
    refuse(a, constrained, "speculative decoding")
}

/// Refuse options a simplified generation loop cannot honour (`what` names the loop in the message).
pub fn refuse(a: &Args, constrained: bool, what: &str) -> Result<(), String> {
    const SAMPLER: [&str; 11] = [
        "min-p",
        "typical-p",
        "repeat-penalty",
        "presence-penalty",
        "frequency-penalty",
        "mirostat",
        "mirostat-tau",
        "mirostat-eta",
        "dry-multiplier",
        "xtc-probability",
        "sampler-json",
    ];
    if let Some(k) = SAMPLER.iter().find(|k| a.get(k).is_some()) {
        return Err(format!(
            "--{k} cannot be combined with {what}, which supports --temperature, --top-k, --top-p sampling only"
        ));
    }
    const OTHER: [&str; 12] = [
        "receipt",
        "session",
        "kv-policy",
        "workers",
        "sources",
        "lang",
        "json-schema",
        "regex",
        "grammar",
        "watermark",
        "steer",
        "lora-none",
    ];
    if let Some(k) = OTHER.iter().find(|k| a.get(k).is_some()) {
        return Err(format!("--{k} cannot be combined with {what}"));
    }
    for f in ["stream-layers", "discover", "governor", "json"] {
        if a.flag(f) {
            return Err(format!("--{f} cannot be combined with {what}"));
        }
    }
    if constrained {
        return Err(format!("structured output cannot be combined with {what}"));
    }
    Ok(())
}

fn sampling_of(s: &SamplerParams) -> SamplingCfg {
    if s.temperature <= 0.0 || !s.temperature.is_finite() {
        SamplingCfg::greedy()
    } else {
        SamplingCfg::sampled(s.temperature, s.top_k, s.top_p)
    }
}

/// What a speculative run did, for the line printed on stderr.
#[derive(Clone, Debug, Default)]
pub struct FastStats {
    pub drafted: usize,
    pub accepted: usize,
    pub passes: usize,
    pub emitted: usize,
    pub note: String,
}
impl FastStats {
    pub fn line(&self) -> String {
        let rate = if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 * 100.0 / self.drafted as f64
        };
        let per = if self.passes == 0 {
            0.0
        } else {
            self.emitted as f64 / self.passes as f64
        };
        format!(
            "speculation: {} of {} guessed tokens accepted ({rate:.0}%), {per:.2} tokens per full pass{}",
            self.accepted,
            self.drafted,
            if self.note.is_empty() {
                String::new()
            } else {
                format!(" · {}", self.note)
            }
        )
    }
}

fn draft_len(a: &Args, default: usize) -> Result<usize, String> {
    let n = a.num::<usize>("draft-len")?.unwrap_or(default);
    if !(1..=selfspec::MAX_DRAFT).contains(&n) {
        return Err(format!(
            "--draft-len must be between 1 and {}",
            selfspec::MAX_DRAFT
        ));
    }
    Ok(n)
}

/// A loaded draft model is only useful when it speaks the same token ids.
fn same_tokens(target: &Prepared, draft: &Prepared) -> Result<(), String> {
    if target.spec.vocab != draft.spec.vocab {
        return Err(format!(
            "the draft model has {} tokens but the main model has {}: they must share a tokenizer",
            draft.spec.vocab, target.spec.vocab
        ));
    }
    for t in ["Hello, world! 123 — ăâîșț", "def f(x):\n    return x * 2"] {
        if target.tok.encode(t, false) != draft.tok.encode(t, false) {
            return Err("the draft model tokenises text differently from the main model: they must share a tokenizer".into());
        }
    }
    Ok(())
}

/// Remove a final stop token (the speculative loops include it in their output) and say how the run ended.
fn finish(
    mut tokens: Vec<usize>,
    stop_ids: &[usize],
    max_tokens: usize,
) -> (Vec<usize>, &'static str) {
    if tokens.last().is_some_and(|t| stop_ids.contains(t)) {
        tokens.pop();
        return (tokens, "stop");
    }
    let f = if tokens.len() >= max_tokens {
        "length"
    } else {
        "context"
    };
    (tokens, f)
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    args: &Args,
    prep: &Prepared,
    cfg: &load::LoadCfg,
    fast: &Fast,
    ids: &[usize],
    sampler: &SamplerParams,
    max_tokens: usize,
    stop_ids: &[usize],
) -> Result<(GenOut, FastStats), String> {
    let model = cmds::build_model_with_threads(prep, cfg, args)?;
    let prompt32: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let stop32: Vec<u32> = stop_ids.iter().map(|&i| i as u32).collect();
    let sampling = sampling_of(sampler);
    let seed = sampler.seed.unwrap_or(1);
    let t0 = Instant::now();
    let (tokens, mut stats): (Vec<usize>, FastStats) = match fast {
        Fast::Lookahead => {
            let c = LookupConfig {
                max_new: max_tokens,
                stop: stop32,
                sampling,
                seed,
                draft_len: draft_len(args, 6)?,
                adaptive: true,
                n_min: args.num("ngram-min")?.unwrap_or(1),
                n_max: args.num("ngram-max")?.unwrap_or(4),
                corpus: vec![],
            };
            if c.n_min == 0 || c.n_max < c.n_min {
                return Err("--ngram-min must be at least 1 and not above --ngram-max".into());
            }
            let o = search::lookup_generate(&model, &prompt32, &c).map_err(|e| e.to_string())?;
            let s = FastStats {
                drafted: o.stats.drafted,
                accepted: o.stats.accepted,
                passes: o.stats.passes,
                emitted: o.stats.emitted,
                note: "guesses copied from the prompt and the text so far".into(),
            };
            (o.tokens.into_iter().map(|t| t as usize).collect(), s)
        }
        Fast::Draft(path) => {
            let mut dcfg = cfg.clone();
            dcfg.lora.clear();
            dcfg.ram_budget = None;
            let dprep = load::prepare(path, &dcfg)
                .map_err(|e| format!("--draft {}: {e}", path.display()))?;
            same_tokens(prep, &dprep)?;
            let mut dmodel = dprep.build_model(&dcfg)?;
            if let Some(n) = args.num::<usize>("threads")? {
                dmodel.set_threads(n);
            }
            eprintln!(
                "draft model {} ({} layers) guesses for {} ({} layers)",
                dprep.name, dprep.spec.layers, prep.name, prep.spec.layers
            );
            let k = draft_len(args, 4)?;
            let sc = SpecConfig {
                max_new: max_tokens,
                stop: stop32,
                sampling,
                seed,
                draft_len: k,
                adaptive: Some(AdaptiveLen::new(1, selfspec::MAX_DRAFT.min(12), k)),
                ctx_limit: Some(model.context()),
            };
            let mut target = ModelVerifier::new(&model);
            let mut drafter = DraftModel::new(ModelVerifier::new(&dmodel));
            let r = spec::generate(&mut target, &mut drafter, &prompt32, &sc);
            let o = match r {
                Ok(o) => o,
                Err(e) => {
                    return Err(target
                        .take_error()
                        .map_or_else(|| e.to_string(), |m| m.to_string()));
                }
            };
            let s = FastStats {
                drafted: o.stats.drafted,
                accepted: o.stats.accepted,
                passes: o.stats.passes,
                emitted: o.stats.emitted,
                note: format!("draft model {}", dprep.name),
            };
            (o.tokens.into_iter().map(|t| t as usize).collect(), s)
        }
        Fast::SelfSpec(skip) => {
            let layers = model.spec.layers as usize;
            let set = match skip {
                Skip::Layers(v) => SkipSet::from_layers(layers, v).map_err(|e| e.to_string())?,
                Skip::Auto => {
                    eprintln!("choosing which layers to skip on your prompt (up to 60 s) …");
                    let o = CalibrateOptions {
                        deadline: Some(Duration::from_secs(60)),
                        ..CalibrateOptions::default()
                    };
                    let c = selfspec::calibrate(&model, std::slice::from_ref(&ids.to_vec()), &o)
                        .map_err(|e| e.to_string())?;
                    if c.worthwhile {
                        eprintln!(
                            "skipping {} of {layers} layers while drafting {:?}; guessed agreement {:.0}%, estimated speed-up ×{:.2} (an estimate, not a timing)",
                            c.skip.skipped_count(),
                            c.skip.skipped(),
                            c.alpha * 100.0,
                            c.est_speedup
                        );
                    } else {
                        eprintln!(
                            "no set of layers beat plain decoding on this prompt; running without drafting"
                        );
                    }
                    c.skip
                }
            };
            let mut sc = SelfSpecConfig::new(set.clone());
            sc.sampling = sampling;
            sc.seed = seed;
            sc.max_tokens = max_tokens;
            sc.stop_tokens = stop_ids.to_vec();
            sc.start_draft = draft_len(args, 4)?.min(sc.max_draft);
            let mut st = model.new_state();
            let o = if set.is_empty() {
                selfspec::generate_plain(&model, &mut st, ids, &sc)
            } else {
                selfspec::generate(&model, &mut st, ids, &sc)
            }
            .map_err(|e| e.to_string())?;
            let passes = o.stats.rounds + o.stats.plain_steps;
            let s = FastStats {
                drafted: o.stats.drafted,
                accepted: o.stats.accepted,
                passes,
                emitted: o.stats.emitted,
                note: if set.is_empty() {
                    "no layers skipped".into()
                } else {
                    format!("{} layers skipped while drafting", set.skipped_count())
                },
            };
            (o.tokens, s)
        }
    };
    let secs = t0.elapsed().as_secs_f64();
    let (tokens, finish) = finish(tokens, stop_ids, max_tokens);
    stats.note = stats.note.trim().to_string();
    let mut dec = prep.tok.decoder();
    dec.skip_special = true;
    let mut text = String::new();
    for &t in &tokens {
        text.push_str(&dec.push(t as u32));
    }
    text.push_str(&dec.finish());
    Ok((
        GenOut {
            text,
            tokens,
            finish,
            prefill_secs: 0.0,
            decode_secs: secs,
            prompt_tokens: ids.len(),
        },
        stats,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;

    fn a(s: &str) -> Args {
        Args::parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>())
    }

    #[test]
    fn skip_lists_parse() {
        assert_eq!(parse_skip("auto").unwrap(), Skip::Auto);
        assert_eq!(
            parse_skip("4, 6,8-10,6").unwrap(),
            Skip::Layers(vec![4, 6, 8, 9, 10])
        );
        assert!(parse_skip("").is_err());
        assert!(parse_skip("x").is_err());
        assert!(parse_skip("5-2").is_err());
        assert!(parse_skip("0-999999").is_err());
    }

    #[test]
    fn the_options_pick_one_kind_of_speculation() {
        assert_eq!(from_args(&a("m")).unwrap(), None);
        assert_eq!(
            from_args(&a("m --lookahead")).unwrap(),
            Some(Fast::Lookahead)
        );
        assert_eq!(
            from_args(&a("m --draft small")).unwrap(),
            Some(Fast::Draft("small".into()))
        );
        assert_eq!(
            from_args(&a("m --self-spec")).unwrap(),
            Some(Fast::SelfSpec(Skip::Auto))
        );
        assert_eq!(
            from_args(&a("m --skip-layers 1,2")).unwrap(),
            Some(Fast::SelfSpec(Skip::Layers(vec![1, 2])))
        );
        assert!(from_args(&a("m --lookahead --self-spec")).is_err());
        assert!(from_args(&a("m --draft x --lookahead")).is_err());
        assert!(from_args(&a("m --draft-len 4")).is_err());
    }

    #[test]
    fn unsupported_options_are_refused_not_ignored() {
        assert!(check_compatible(&a("m --temperature 0"), false).is_ok());
        assert!(
            check_compatible(&a("m --min-p 0.1"), false)
                .unwrap_err()
                .contains("--min-p")
        );
        assert!(check_compatible(&a("m --receipt r.json"), false).is_err());
        assert!(check_compatible(&a("m --stream-layers"), false).is_err());
        assert!(check_compatible(&a("m"), true).is_err());
    }

    fn greedy_plain(prep: &Prepared, cfg: &load::LoadCfg, ids: &[usize], n: usize) -> Vec<usize> {
        let model = prep.build_model(cfg).unwrap();
        let mut st = model.new_state();
        let mut s = companion_engine::model::Sampler::new(0.0, 1, 1);
        let mut got = vec![];
        let opts = companion_engine::model::GenerateOptions {
            max_tokens: n,
            stop_tokens: vec![],
            cancel: None,
        };
        model
            .generate(&mut st, ids, &mut s, &opts, &mut |t| {
                got.push(t);
                true
            })
            .unwrap();
        got
    }

    #[test]
    fn every_kind_of_speculation_gives_the_greedy_answer() {
        let d = tmpdir("fast");
        write_model_ctx(&d, true, 96);
        let draft = tmpdir("fast-draft");
        write_model_ctx(&draft, true, 96);
        let base = d.to_string_lossy().to_string();
        let args0 = Args::parse(std::slice::from_ref(&base));
        let (prep, cfg) = cmds::open_prepared(&args0).unwrap();
        let ids = cmds::prompt_ids(
            &prep,
            &[crate::template::Message::new("user", "abcabcabcabc")],
            None,
        )
        .unwrap();
        let want = greedy_plain(&prep, &cfg, &ids, 12);
        let sp = SamplerParams {
            temperature: 0.0,
            seed: Some(1),
            ..Default::default()
        };
        let cases: Vec<(&str, Fast)> = vec![
            ("lookahead", Fast::Lookahead),
            ("draft", Fast::Draft(draft.clone())),
            ("self-spec list", Fast::SelfSpec(Skip::Layers(vec![]))),
        ];
        for (name, f) in cases {
            let (o, st) = run(&args0, &prep, &cfg, &f, &ids, &sp, 12, &[])
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(o.tokens, want, "{name} must give the greedy tokens");
            assert!(
                st.emitted >= o.tokens.len().saturating_sub(1),
                "{name}: {st:?}"
            );
            assert!(!st.line().is_empty());
        }
        // skipping a layer while drafting still gives the same tokens (the full model decides)
        if let Ok((o, _)) = run(
            &args0,
            &prep,
            &cfg,
            &Fast::SelfSpec(Skip::Layers(vec![1])),
            &ids,
            &sp,
            12,
            &[],
        ) {
            assert_eq!(o.tokens, want, "skipped layer");
        }
        std::fs::remove_dir_all(d).unwrap();
        std::fs::remove_dir_all(draft).unwrap();
    }

    #[test]
    fn a_draft_with_another_tokenizer_is_refused() {
        let d = tmpdir("fast2");
        write_model_ctx(&d, true, 96);
        let (prep, cfg) =
            cmds::open_prepared(&Args::parse(&[d.to_string_lossy().to_string()])).unwrap();
        let bad = tmpdir("fast2-bad");
        std::fs::write(bad.join("config.json"), "{}").unwrap();
        let sp = SamplerParams {
            temperature: 0.0,
            ..Default::default()
        };
        let e = run(
            &Args::parse(&[]),
            &prep,
            &cfg,
            &Fast::Draft(bad.clone()),
            &[65, 66],
            &sp,
            4,
            &[],
        )
        .unwrap_err();
        assert!(e.contains("--draft"), "{e}");
        std::fs::remove_dir_all(d).unwrap();
        std::fs::remove_dir_all(bad).unwrap();
    }
}
