//! `--watermark KEY` on generate, and `corbul watermark detect`.
//!
//! A statistical green-list mark (see `companion_engine::watermark`): a keyed hash of the previous
//! token splits the vocabulary, and green tokens get a small logit bonus before sampling. Detection
//! counts green tokens and reports a z-score and a p-value. It is evidence, not proof: short or
//! low-entropy text (code, quotes) carries little, paraphrasing or translating removes it, and anyone
//! with the key can forge it. The context the hash sees starts at the first generated token, so
//! detection of generated text needs no prompt.
use crate::args::Args;
use crate::genloop::Constraint;
use companion_engine::watermark::{Watermark, WatermarkConfig};

/// A secret string becomes the 64-bit key of the hash (FNV-1a, then mixed). Same text, same key.
pub fn key_of(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^= h >> 29;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^ (h >> 32)
}

/// Parameters from `--watermark KEY --wm-gamma G --wm-delta D --wm-context N`.
pub fn config(a: &Args, key: &str, ignore: Vec<u32>) -> Result<WatermarkConfig, String> {
    let mut c = WatermarkConfig {
        key: key_of(key),
        ignore,
        ..WatermarkConfig::default()
    };
    if let Some(v) = a.num("wm-gamma")? {
        c.gamma = v;
    }
    if let Some(v) = a.num("wm-delta")? {
        c.delta = v;
    }
    if let Some(v) = a.num("wm-context")? {
        c.context = v;
    }
    Ok(c)
}

pub fn build(a: &Args, key: &str, ignore: Vec<u32>) -> Result<Watermark, String> {
    Watermark::new(config(a, key, ignore)?).map_err(|e| e.to_string())
}

/// Biases the logits of green tokens; remembers what was chosen so the next split sees it.
pub struct WatermarkConstraint {
    wm: Watermark,
    prev: Vec<u32>,
}
impl WatermarkConstraint {
    pub fn new(wm: Watermark) -> Self {
        WatermarkConstraint { wm, prev: vec![] }
    }
}
impl Constraint for WatermarkConstraint {
    fn apply(&mut self, logits: &mut [f32]) {
        self.wm.apply_to_logits(&self.prev, logits);
    }
    fn accept(&mut self, token: usize) {
        self.prev.push(token as u32);
    }
}

/// Run several constraints in order (masks first, the watermark last, so it never revives a masked token).
pub struct Chain(pub Vec<Box<dyn Constraint>>);
impl Constraint for Chain {
    fn apply(&mut self, logits: &mut [f32]) {
        for c in &mut self.0 {
            c.apply(logits);
        }
    }
    fn accept(&mut self, token: usize) {
        for c in &mut self.0 {
            c.accept(token);
        }
    }
}

/// Add the watermark after whatever constraint is already there.
pub fn wrap(existing: Option<Box<dyn Constraint>>, wm: Watermark) -> Box<dyn Constraint> {
    let w: Box<dyn Constraint> = Box::new(WatermarkConstraint::new(wm));
    match existing {
        Some(c) => Box::new(Chain(vec![c, w])),
        None => w,
    }
}

/// `corbul watermark detect MODEL --key K (--text T | --file F)`.
pub fn cmd(a: &Args) -> Result<(), String> {
    match a.pos.first().map(String::as_str) {
        Some("detect") => detect(a),
        _ => Err("watermark: use `corbul watermark detect MODEL --key K (--text TEXT | --file F)`; to mark text, add --watermark KEY to `generate`".into()),
    }
}

fn detect(a: &Args) -> Result<(), String> {
    let model = a.pos.get(1).ok_or("watermark detect: give the model folder or .gguf the text was written with (its tokenizer is needed)")?;
    let key = a
        .get("key")
        .ok_or("watermark detect: --key SECRET is required")?;
    let text = match (a.get("text"), a.get("file")) {
        (Some(t), _) => t.to_string(),
        (None, Some(f)) => std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?,
        _ => return Err("watermark detect: give --text TEXT or --file F".into()),
    };
    let (tok, _) = crate::famrun::load_tokenizer(std::path::Path::new(model))?;
    let ignore: Vec<u32> = tok.eos.into_iter().collect();
    let wm = build(a, key, ignore)?;
    let ids = tok.encode(&text, false);
    let dedup = !a.flag("no-dedup");
    let d = wm.score(&[], &ids, dedup);
    let verdict = |p: f64| {
        if d.scored < 25 {
            "too short to say anything"
        } else if p < 1e-4 {
            "strong evidence of this key's mark"
        } else if p < 0.01 {
            "some evidence"
        } else {
            "no evidence"
        }
    };
    if a.flag("json") {
        let mut j = serde_json::json!({"tokens": ids.len(), "scored": d.scored, "green": d.green, "expected": d.expected, "z": d.z, "p_value": d.p_value, "green_fraction": d.green_fraction, "verdict": verdict(d.p_value)});
        if let Some(w) = a
            .num::<usize>("window")?
            .and_then(|w| wm.scan(&[], &ids, w, dedup))
        {
            j["best_window"] = serde_json::json!({"start": w.start, "z": w.detection.z, "adjusted_p": w.adjusted_p});
        }
        crate::outln!("{}", serde_json::to_string_pretty(&j).unwrap_or_default());
        return Ok(());
    }
    crate::outln!("tokens        {} ({} counted)", ids.len(), d.scored);
    crate::outln!(
        "green         {} ({:.1}%, {:.1}% expected without the mark)",
        d.green,
        d.green_fraction * 100.0,
        wm.gamma() * 100.0
    );
    crate::outln!("z-score       {:.2}", d.z);
    crate::outln!("p-value       {:.3e}", d.p_value);
    if let Some(w) = a
        .num::<usize>("window")?
        .and_then(|w| wm.scan(&[], &ids, w, dedup))
    {
        crate::outln!(
            "best window   from token {} of {} windows: z {:.2}, adjusted p {:.3e}",
            w.start,
            w.windows,
            w.detection.z,
            w.adjusted_p
        );
    }
    crate::outln!("verdict       {}", verdict(d.p_value));
    crate::outln!(
        "note          a statistical signal, not proof: paraphrasing removes it, short or low-entropy text carries little, and the key holder can forge it"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn a(s: &str) -> Args {
        Args::parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>())
    }

    #[test]
    fn keys_are_stable_and_differ() {
        assert_eq!(key_of("abc"), key_of("abc"));
        assert_ne!(key_of("abc"), key_of("abd"));
    }

    #[test]
    fn marked_logits_are_detected_and_unmarked_are_not() {
        let wm = build(&a("x"), "secret", vec![]).unwrap();
        let mut c = WatermarkConstraint::new(wm.clone());
        let mut seed = 7u64;
        let mut out = vec![];
        for _ in 0..300 {
            // a flat-ish distribution: the sampler has freedom, so the bias shows
            let mut logits: Vec<f32> = (0..64)
                .map(|i| ((i * 31 + out.len() * 17) % 11) as f32 * 0.1)
                .collect();
            c.apply(&mut logits);
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
            let w: Vec<f64> = logits
                .iter()
                .map(|l| ((*l - mx) as f64 * 3.0).exp())
                .collect();
            let mut r = (seed >> 11) as f64 / (1u64 << 53) as f64 * w.iter().sum::<f64>();
            let mut pick = 63;
            for (i, x) in w.iter().enumerate() {
                if r < *x {
                    pick = i;
                    break;
                }
                r -= x;
            }
            c.accept(pick);
            out.push(pick as u32);
        }
        assert!(wm.score(&[], &out, true).p_value < 1e-3);
        let other = build(&a("x"), "another key", vec![]).unwrap();
        assert!(other.score(&[], &out, true).p_value > 1e-3);
    }

    #[test]
    fn generate_with_a_mark_and_the_detector_run_end_to_end() {
        use crate::load::testkit::*;
        let d = tmpdir("wm-e2e");
        write_model_ctx(&d, true, 96);
        let m = d.to_string_lossy().to_string();
        crate::cmds::generate(&a(&format!(
            "{m} --prompt abc --max-tokens 6 --temperature 0 --watermark k1 --quiet"
        )))
        .unwrap();
        assert!(
            crate::cmds::generate(&a(&format!(
                "{m} --prompt abc --watermark k1 --receipt r.json"
            )))
            .is_err()
        );
        assert!(
            crate::cmds::generate(&a(&format!("{m} --prompt abc --watermark k1 --wm-gamma 2")))
                .is_err()
        );
        cmd(&a(&format!("detect {m} --key k1 --text abcabcabcabc"))).unwrap();
        cmd(&a(&format!(
            "detect {m} --key k1 --text abcabcabcabc --json --window 4"
        )))
        .unwrap();
        assert!(cmd(&a(&format!("detect {m} --text x"))).is_err());
        // speculative decoding end to end through the command
        crate::cmds::generate(&a(&format!(
            "{m} --prompt abcabc --max-tokens 6 --temperature 0 --lookahead --quiet"
        )))
        .unwrap();
        assert!(
            crate::cmds::generate(&a(&format!("{m} --prompt abc --lookahead --watermark k")))
                .is_err()
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn bad_parameters_are_refused() {
        assert!(build(&a("x --wm-gamma 1.5"), "k", vec![]).is_err());
        assert!(build(&a("x --wm-context 0"), "k", vec![]).is_err());
        assert!(cmd(&a("nope")).is_err());
    }
}
