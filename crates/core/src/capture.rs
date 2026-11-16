//! Capturing a real routing trace and reading from it what would help on this computer.
//!
//! Colibri writes a routing stream when the environment variable `ROUTE_TRACE=<path>` is set
//! (see colibri's `docs/routing-telemetry.md`): one line per (moe call, batch row):
//! `<call> <row> <layer> <expert>:<gate> ...`. The file is opened for writing (truncated) when
//! the engine starts and grows with every token, so the caller must watch its size
//! ([`over_cap`]) and stop the engine, or ignore the tail, once it is large enough.
//! `deepseek_v4` keeps the history but writes no rows to this stream (its own banner says so).
//!
//! The analysis replays the capture through the `expert-streamer` cost models. Those are
//! models, not measurements of speed: they say how many expert loads a policy would save.

use expert_streamer::policy::{DecayFreq, Lru, belady, replay};
use expert_streamer::prefetch;
use expert_streamer::speculative;
use expert_streamer::trace::parse_trace;
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::Path;

/// Default cap on a capture: about ten thousand tokens on a 40-layer model.
pub const DEFAULT_CAP_BYTES: u64 = 64 * 1024 * 1024;
/// Below this many tokens the numbers are only indicative.
pub const MIN_TOKENS: usize = 200;

/// The environment to start colibri with so that it records its routing to `path`.
pub fn capture_env(path: &Path) -> Vec<(String, String)> {
    vec![("ROUTE_TRACE".into(), path.to_string_lossy().into_owned())]
}

/// True when the capture has reached `cap_bytes` (or the file cannot be read).
pub fn over_cap(path: &Path, cap_bytes: u64) -> bool {
    fs::metadata(path).map_or(true, |m| m.len() >= cap_bytes)
}

/// Reads at most `cap_bytes` from the start of the file, cut at the last whole line.
fn read_capped(path: &Path, cap_bytes: u64) -> std::io::Result<(String, bool)> {
    let f = fs::File::open(path)?;
    let mut buf = Vec::new();
    f.take(cap_bytes).read_to_end(&mut buf)?;
    let truncated = buf.len() as u64 >= cap_bytes;
    if truncated && let Some(i) = buf.iter().rposition(|b| *b == b'\n') {
        buf.truncate(i + 1);
    }
    Ok((String::from_utf8_lossy(&buf).into_owned(), truncated))
}

/// What a capture contains.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Summary {
    pub bytes: u64,
    pub lines: usize,
    /// Distinct moe calls seen (a prompt pass counts as one call carrying many rows).
    pub calls: usize,
    /// Tokens: distinct (row, layer-wrap) steps, the unit the analysis replays.
    pub tokens: usize,
    pub layers: u32,
    pub experts_seen: usize,
    pub max_expert: u32,
    pub per_layer_top_k: usize,
}

/// A guess at the number of experts per layer from the largest id seen: the next of the sizes
/// models actually use. Pass the real number whenever it is known (the model's config).
pub fn guess_experts(max_expert: u32) -> u32 {
    [8, 16, 32, 64, 128, 160, 192, 256, 384, 512, 1024]
        .into_iter()
        .find(|n| *n > max_expert)
        .unwrap_or(max_expert + 1)
}

/// Counts what is in `text` without needing the model's dimensions.
pub fn summarize_text(text: &str, bytes: u64) -> Summary {
    let mut calls = HashSet::new();
    let (mut lines, mut layers, mut max_e, mut top_k) = (0usize, 0u32, 0u32, 0usize);
    let mut seen = HashSet::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(c), Some(_row), Some(l)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        let (Ok(c), Ok(l)) = (c.parse::<u64>(), l.parse::<u32>()) else {
            continue;
        };
        let ids: Vec<u32> = it
            .filter_map(|t| t.split(':').next()?.parse().ok())
            .collect();
        if ids.is_empty() {
            continue;
        }
        lines += 1;
        calls.insert(c);
        layers = layers.max(l + 1);
        top_k = top_k.max(ids.len());
        for e in ids {
            max_e = max_e.max(e);
            seen.insert((l, e));
        }
    }
    let tokens = parse_trace(text, guess_experts(max_e)).len();
    Summary {
        bytes,
        lines,
        calls: calls.len(),
        tokens,
        layers,
        experts_seen: seen.len(),
        max_expert: max_e,
        per_layer_top_k: top_k,
    }
}

/// Hit rates of the policies at one cache size (a share of every expert of the model).
#[derive(Debug, Clone, PartialEq)]
pub struct CachePoint {
    pub cache_pct: u32,
    pub lru: f64,
    pub decay_freq: f64,
    /// A fixed set of the most used experts: the best a pinned cache could do on this capture.
    pub pinned: f64,
    /// Sees the future: a ceiling, not something an engine can run.
    pub optimal: f64,
}

/// Share of expert loads a next-layer predictor would have had ready, at `mult` times the
/// experts the layer really used, against plain popularity with the same budget.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefetchPoint {
    pub mult: usize,
    pub predicted: f64,
    pub popularity_only: f64,
}

/// Expert loads per accepted token when `k` drafted tokens are verified in one pass.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecPoint {
    pub k: usize,
    pub acceptance: f64,
    pub loads_per_token: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub summary: Summary,
    pub n_experts: u32,
    /// The file was longer than the cap and only its start was read.
    pub truncated: bool,
    pub too_short: bool,
    pub cache: Vec<CachePoint>,
    pub prefetch: Vec<PrefetchPoint>,
    pub speculative: Vec<SpecPoint>,
    /// Plain English, most important first. Always starts with the honesty flag when too short.
    pub verdicts: Vec<String>,
}

fn pinned_cover(trace: &[Vec<u32>], cap: usize) -> f64 {
    let mut count = std::collections::HashMap::<u32, u64>::new();
    let mut total = 0u64;
    for k in trace.iter().flatten() {
        *count.entry(*k).or_default() += 1;
        total += 1;
    }
    let mut v: Vec<u64> = count.into_values().collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    if total == 0 {
        0.0
    } else {
        // the first use of each pinned expert still has to load it
        let top = v.iter().take(cap);
        (top.clone().sum::<u64>() - top.count() as u64) as f64 / total as f64
    }
}

/// Replays a capture text through the cost models. `n_experts` is the number of experts per
/// layer (use [`guess_experts`] only when the model's config is unknown).
pub fn analyze_text(text: &str, bytes: u64, n_experts: u32, truncated: bool) -> Report {
    let summary = summarize_text(text, bytes);
    let trace = parse_trace(text, n_experts);
    let universe = (summary.layers.max(1) * n_experts) as usize;
    let too_short = trace.len() < MIN_TOKENS;

    let cache: Vec<CachePoint> = [5u32, 10, 25, 50]
        .into_iter()
        .map(|pct| {
            let cap = (universe * pct as usize / 100).max(1);
            CachePoint {
                cache_pct: pct,
                lru: replay(&trace, &mut Lru::new(cap)),
                decay_freq: replay(&trace, &mut DecayFreq::new(cap, 300.0)),
                pinned: pinned_cover(&trace, cap),
                optimal: belady(&trace, cap),
            }
        })
        .collect();

    let prefetch: Vec<PrefetchPoint> = [1usize, 2, 4]
        .into_iter()
        .map(|mult| {
            let s = prefetch::evaluate(&trace, n_experts, 0.5, mult);
            PrefetchPoint {
                mult,
                predicted: s.share(),
                popularity_only: s.baseline(),
            }
        })
        .collect();

    let cap10 = (universe / 10).max(1);
    let mut speculative = vec![SpecPoint {
        k: 1,
        acceptance: 1.0,
        loads_per_token: speculative::loads_per_token(&trace, 1, 1.0, cap10),
    }];
    for k in [4usize, 8] {
        for a in [0.5, 0.7, 0.9] {
            speculative.push(SpecPoint {
                k,
                acceptance: a,
                loads_per_token: speculative::loads_per_token(&trace, k, a, cap10),
            });
        }
    }

    let verdicts = verdicts(
        &summary,
        &cache,
        &prefetch,
        &speculative,
        too_short,
        truncated,
    );
    Report {
        summary,
        n_experts,
        truncated,
        too_short,
        cache,
        prefetch,
        speculative,
        verdicts,
    }
}

fn pct(x: f64) -> String {
    format!("{:.0}%", 100.0 * x)
}

fn verdicts(
    s: &Summary,
    cache: &[CachePoint],
    pre: &[PrefetchPoint],
    spec: &[SpecPoint],
    too_short: bool,
    truncated: bool,
) -> Vec<String> {
    let mut v = Vec::new();
    if s.tokens == 0 {
        v.push("The capture holds no routing rows: this engine may not write them, or the run was too short.".into());
        return v;
    }
    if too_short {
        v.push(format!(
            "Only {} tokens were captured (at least {MIN_TOKENS} are needed): treat every number below as indicative.",
            s.tokens
        ));
    }
    if truncated {
        v.push("The capture was longer than the size limit: only its start was analysed.".into());
    }
    if let Some(c) = cache.iter().find(|c| c.cache_pct == 10) {
        if c.pinned > c.lru + 0.05 {
            v.push(format!(
                "Pinning the most used experts would serve {} of loads from a 10% cache against {} for plain LRU: the learning cache (PIN) is worth using.",
                pct(c.pinned),
                pct(c.lru)
            ));
        } else {
            v.push(format!(
                "LRU already serves {} of loads from a 10% cache and pinning adds little ({}): the cache policy is not the bottleneck.",
                pct(c.lru),
                pct(c.pinned)
            ));
        }
        let gap = c.optimal - c.lru.max(c.pinned);
        if gap > 0.10 {
            v.push(format!(
                "Even the best of those leaves {} points to the theoretical ceiling at a 10% cache: a smarter policy or prefetching could still gain, but the ceiling cannot be reached.",
                (gap * 100.0).round()
            ));
        }
    }
    if let Some(p) = pre.iter().find(|p| p.mult == 2) {
        if p.predicted - p.popularity_only > 0.10 {
            v.push(format!(
                "Layers are correlated: a next-layer predictor would have {} of loads ready in time against {} from popularity alone, so prefetching is worth building.",
                pct(p.predicted),
                pct(p.popularity_only)
            ));
        } else {
            v.push(format!(
                "Little correlation between layers was found ({} against {} from popularity alone): prefetching by prediction would not help much.",
                pct(p.predicted),
                pct(p.popularity_only)
            ));
        }
    }
    let base = spec.iter().find(|p| p.k == 1).map(|p| p.loads_per_token);
    let best = spec
        .iter()
        .filter(|p| p.k > 1 && p.acceptance >= 0.7)
        .min_by(|a, b| a.loads_per_token.total_cmp(&b.loads_per_token));
    if let (Some(base), Some(b)) = (base, best) {
        if base > 0.0 && b.loads_per_token < 0.9 * base {
            v.push(format!(
                "Verifying {} drafted tokens per pass at {} acceptance would cut expert loads per token from {:.0} to {:.0}: speculative decoding could pay off.",
                b.k,
                pct(b.acceptance),
                base,
                b.loads_per_token
            ));
        } else {
            v.push("Verifying several drafted tokens per pass would not reduce expert loads on this capture: speculative decoding is not a gain for the drive.".into());
        }
    }
    v.push("These are cost models over your own routing, not speed measurements.".into());
    v
}

/// Reads a capture file (at most `cap_bytes`) and analyses it.
pub fn analyze_file(path: &Path, n_experts: Option<u32>, cap_bytes: u64) -> Result<Report, String> {
    let bytes = fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    let (text, truncated) = read_capped(path, cap_bytes).map_err(|e| e.to_string())?;
    let ne = n_experts.unwrap_or_else(|| guess_experts(summarize_text(&text, bytes).max_expert));
    Ok(analyze_text(&text, bytes, ne, truncated))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tokens` tokens over `layers` layers, `k` experts each, drawn from a small hot set.
    fn trace_text(tokens: usize, layers: u32, k: u32, hot: u32) -> String {
        let mut s = String::new();
        let mut x = 7u64;
        for t in 0..tokens {
            for l in 0..layers {
                s += &format!("{t} 0 {l}");
                for j in 0..k {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let e = (x >> 33) as u32 % hot;
                    s += &format!(" {}:0.1250", (e + j) % 64);
                }
                s.push('\n');
            }
        }
        s
    }

    #[test]
    fn the_environment_names_the_variable_colibri_reads() {
        let e = capture_env(Path::new("/tmp/t.txt"));
        assert_eq!(
            e,
            vec![("ROUTE_TRACE".to_string(), "/tmp/t.txt".to_string())]
        );
    }

    #[test]
    fn summarizes_tokens_layers_and_experts() {
        let t = trace_text(10, 4, 2, 8);
        let s = summarize_text(&t, t.len() as u64);
        assert_eq!(
            (s.tokens, s.layers, s.per_layer_top_k, s.lines),
            (10, 4, 2, 40)
        );
        assert_eq!(s.calls, 10);
        assert!(s.experts_seen > 0 && s.max_expert < 64);
    }

    #[test]
    fn guesses_the_next_real_expert_count() {
        assert_eq!(guess_experts(0), 8);
        assert_eq!(guess_experts(255), 256);
        assert_eq!(guess_experts(256), 384);
    }

    #[test]
    fn a_short_capture_is_flagged_first() {
        let t = trace_text(20, 4, 2, 8);
        let r = analyze_text(&t, t.len() as u64, 64, false);
        assert!(r.too_short);
        assert!(r.verdicts[0].contains("Only 20 tokens"));
    }

    #[test]
    fn a_hot_set_is_served_by_pinning_and_the_ceiling_is_respected() {
        let t = trace_text(400, 6, 2, 6);
        let r = analyze_text(&t, t.len() as u64, 64, false);
        assert!(!r.too_short);
        for c in &r.cache {
            assert!(
                c.optimal + 1e-9 >= c.lru && c.optimal + 1e-9 >= c.pinned,
                "{c:?}"
            );
        }
        let c10 = r.cache.iter().find(|c| c.cache_pct == 10).unwrap();
        assert!(c10.lru > 0.5, "a hot set fits a 10% cache: {c10:?}");
        assert!(r.verdicts.iter().any(|v| v.contains("cost models")));
    }

    #[test]
    fn an_empty_capture_says_so() {
        let r = analyze_text("", 0, 64, false);
        assert_eq!(r.summary.tokens, 0);
        assert!(r.verdicts[0].contains("no routing rows"));
    }

    #[test]
    fn reads_a_file_up_to_the_cap_and_cuts_at_a_whole_line() {
        let dir = std::env::temp_dir().join(format!("coli-capture-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("trace.txt");
        let t = trace_text(300, 3, 2, 6);
        fs::write(&p, &t).unwrap();
        assert!(over_cap(&p, 100));
        assert!(!over_cap(&p, t.len() as u64 + 1));
        let full = analyze_file(&p, Some(64), DEFAULT_CAP_BYTES).unwrap();
        assert!(!full.truncated && full.summary.tokens == 300);
        let cut = analyze_file(&p, Some(64), 2000).unwrap();
        assert!(cut.truncated && cut.summary.tokens < 300);
        assert!(cut.verdicts.iter().any(|v| v.contains("size limit")));
        assert!(analyze_file(&dir.join("missing"), None, 10).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
