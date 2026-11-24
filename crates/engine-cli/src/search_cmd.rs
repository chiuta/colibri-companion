//! `corbul search MODEL --prompt TEXT --method beam|best-of-n|vote`: spend more computation for a
//! better answer. All of these run the model several times, so they are slower than plain generation.
//!
//! * `beam`: keeps the `--width` most probable continuations at every step and prints the best (and,
//!   with `--all`, the others). Deterministic. Favours likely text, not necessarily better text.
//! * `best-of-n`: draws `--samples` continuations at the given temperature and keeps the one with the
//!   best mean log-probability per token. Log-probability is the model's own confidence, not a check of
//!   truth.
//! * `vote`: self-consistency. Draws `--samples` answers, extracts a final answer from each (`--answer
//!   number` takes the last number, `text` the whole normalised text) and prints the most common one
//!   with its agreement. Useful for questions with one short answer.
//!
//! The prompt is used as plain text after the chat template. No structured output, no receipts.
use crate::args::Args;
use crate::cmds;
use crate::template::Message;
use companion_engine::search::{self, BeamConfig, MeanLogprob, SampleConfig};
use companion_engine::spec::SamplingCfg;

pub fn cmd(args: &Args) -> Result<(), String> {
    cmds_apply(args);
    crate::fast::refuse(
        &Args {
            pos: vec![],
            opts: args
                .opts
                .iter()
                .filter(|(k, _)| k != "json")
                .cloned()
                .collect(),
        },
        false,
        "search",
    )?;
    let (prep, cfg) = cmds::open_prepared(args)?;
    let prompt = match (args.get("prompt"), args.get("prompt-file")) {
        (Some(p), _) => p.to_string(),
        (None, Some(f)) => std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?,
        _ => args
            .pos
            .get(1)
            .cloned()
            .ok_or("search: give --prompt TEXT (or --prompt-file F)")?,
    };
    let method = args.get("method").unwrap_or("beam");
    let ids = cmds::prompt_ids(
        &prep,
        &[Message::new("user", &prompt)],
        args.flag("raw").then_some(prompt.as_str()),
    )?;
    let prompt32: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let stop: Vec<u32> = prep.stop_ids.iter().map(|&i| i as u32).collect();
    let max_new: usize = args.num("max-tokens")?.unwrap_or(128);
    let model = cmds::build_model_with_threads(&prep, &cfg, args)?;
    let text = |t: &[u32]| prep.tok.decode_with(t, true);
    let seed: u64 = args.num("seed")?.unwrap_or(1);
    let temp: f32 = args.num("temperature")?.unwrap_or(0.8);
    let sampling = if temp <= 0.0 {
        SamplingCfg::greedy()
    } else {
        SamplingCfg::sampled(
            temp,
            args.num("top-k")?.unwrap_or(40),
            args.num("top-p")?.unwrap_or(0.95),
        )
    };
    let json = args.flag("json");
    let mut j = serde_json::json!({"method": method});
    match method {
        "beam" => {
            let c = BeamConfig {
                width: args.num("width")?.unwrap_or(4),
                max_new,
                stop,
                length_penalty: args.num("length-penalty")?.unwrap_or(0.0),
                min_new: 0,
            };
            if c.width == 0 {
                return Err("--width must be at least 1".into());
            }
            let hyp = search::beam_search(&model, &prompt32, &c).map_err(|e| e.to_string())?;
            let shown = if args.flag("all") { hyp.len() } else { 1 };
            let list: Vec<_> = hyp.iter().take(shown).map(|h| serde_json::json!({"text": text(&h.tokens), "logprob": h.logprob, "score": h.score, "finished": h.stopped})).collect();
            if json {
                j["results"] = list.into();
            } else {
                for (i, h) in hyp.iter().take(shown).enumerate() {
                    if shown > 1 {
                        crate::outln!("--- beam {} (log-probability {:.2}) ---", i + 1, h.logprob);
                    }
                    crate::outln!("{}", text(&h.tokens));
                }
                if let Some(h) = hyp.first() {
                    eprintln!(
                        "[beam width {} · {} tokens · log-probability {:.2}]",
                        c.width,
                        h.tokens.len(),
                        h.logprob
                    );
                }
            }
        }
        "best-of-n" | "vote" => {
            let n: usize = args.num("samples")?.unwrap_or(5);
            if !(1..=64).contains(&n) {
                return Err("--samples must be between 1 and 64".into());
            }
            let c = SampleConfig {
                max_new,
                stop,
                sampling,
                seed,
            };
            let samples = search::sample_n(&model, &prompt32, n, &c).map_err(|e| e.to_string())?;
            if method == "best-of-n" {
                let best =
                    search::best_of_n(&samples, &MeanLogprob).ok_or("no samples were drawn")?;
                if json {
                    j["results"] = serde_json::json!([{"text": text(&best.tokens), "logprob": best.logprob, "samples": n}]);
                } else {
                    crate::outln!("{}", text(&best.tokens));
                    eprintln!(
                        "[best of {n} by mean log-probability {:.3}]",
                        best.logprob / best.tokens.len().max(1) as f64
                    );
                }
            } else {
                let kind = args.get("answer").unwrap_or("number");
                if !matches!(kind, "number" | "text") {
                    return Err("--answer must be `number` or `text`".into());
                }
                let key = |s: &search::Sample| {
                    let t = text(&s.tokens);
                    if kind == "number" {
                        search::last_number(&t)
                    } else {
                        let k = search::normalise_answer(&t);
                        (!k.is_empty()).then_some(k)
                    }
                };
                let votes = search::self_consistency(&samples, &key, Some(&MeanLogprob));
                let Some(top) = votes.first() else {
                    return Err(format!(
                        "none of the {n} samples contained an answer of kind `{kind}`"
                    ));
                };
                let agree = search::agreement(&votes);
                let best = &samples[top.best];
                if json {
                    j["results"] = serde_json::json!([{"answer": top.answer, "agreement": agree, "text": text(&best.tokens), "votes": votes.iter().map(|v| serde_json::json!({"answer": v.answer, "count": v.count})).collect::<Vec<_>>()}]);
                } else {
                    crate::outln!("{}", top.answer);
                    eprintln!(
                        "[{} of {} samples agree ({:.0}%){}]",
                        top.count,
                        votes.iter().map(|v| v.count).sum::<usize>(),
                        agree * 100.0,
                        if agree < 0.5 {
                            " — low agreement: do not trust this answer"
                        } else {
                            ""
                        }
                    );
                    if args.flag("explain") {
                        crate::outln!(
                            "\nreasoning behind the winning answer:\n{}",
                            text(&best.tokens)
                        );
                    }
                }
            }
        }
        other => {
            return Err(format!(
                "--method `{other}` is not one of beam, best-of-n, vote"
            ));
        }
    }
    if json {
        crate::outln!("{}", serde_json::to_string_pretty(&j).unwrap_or_default());
    }
    Ok(())
}

fn cmds_apply(args: &Args) {
    cmds::apply_global(args);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;
    fn a(s: &str) -> Args {
        Args::parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>())
    }

    #[test]
    fn every_method_runs_on_a_tiny_model() {
        let d = tmpdir("search");
        write_model_ctx(&d, true, 96);
        let m = d.to_string_lossy().to_string();
        for extra in [
            "--method beam --width 2",
            "--method beam --width 2 --all --json",
            "--method best-of-n --samples 3",
            "--method vote --samples 3 --answer text",
            "--method vote --samples 3 --json",
        ] {
            let r = cmd(&a(&format!("{m} --prompt abc --max-tokens 4 {extra}")));
            if let Err(e) = &r {
                assert!(e.contains("none of the"), "{extra}: {e}");
            }
        }
        assert!(cmd(&a(&format!("{m} --prompt abc --method nope"))).is_err());
        assert!(cmd(&a(&format!("{m} --prompt abc --samples 0 --method vote"))).is_err());
        assert!(cmd(&a(&format!("{m} --prompt abc --min-p 0.1"))).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }
}
