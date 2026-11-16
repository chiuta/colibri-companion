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
/// Longest analysed capture, in tokens. The policy replays are linear (the optimal one and the
/// predictor's tables are the heavy ones), and this keeps the time and memory of a 64 MB file
/// to seconds and a few hundred MB.
pub const MAX_TOKENS: usize = 6000;
/// Bounds on what a trace line may say. A real model stays far inside them; a corrupt or hostile
/// file does not get to overflow an index or allocate without limit.
pub const MAX_LAYERS: u32 = 512;
pub const MAX_EXPERTS: u32 = 4096;
/// The decayed-frequency policy scans its whole cache at every miss (measured: about 20 ns per
/// access and cached expert), so it is replayed only while accesses times cache size stay under
/// this bound (about a second). On a real capture that leaves it out; it is not used in the
/// verdicts.
const DECAY_MAX_WORK: usize = 50_000_000;
/// The next-layer predictor is trained again for every budget; this many tokens are enough for
/// its share estimate and keep it to a couple of seconds.
const PREFETCH_TOKENS: usize = 2000;

/// The environment to start colibri with so that it records its routing to `path`.
pub fn capture_env(path: &Path) -> Vec<(String, String)> {
    vec![("ROUTE_TRACE".into(), path.to_string_lossy().into_owned())]
}

/// True when the capture has reached `cap_bytes`. A file that does not exist yet is not over the
/// cap (the engine creates it a moment after it starts); any other failure to read it is.
pub fn over_cap(path: &Path, cap_bytes: u64) -> bool {
    match fs::metadata(path) {
        Ok(m) => m.len() >= cap_bytes,
        Err(e) => e.kind() != std::io::ErrorKind::NotFound,
    }
}

/// Reads at most `cap_bytes` from the start of the file and keeps only whole lines: the engine
/// may still be writing, so the last line can be cut anywhere (an expert id cut after its first
/// digit would be a different expert). The flag says the file held more than `cap_bytes`.
fn read_capped(path: &Path, cap_bytes: u64) -> std::io::Result<(String, bool)> {
    let f = fs::File::open(path)?;
    let mut buf = Vec::new();
    // one byte more than the cap tells a file of exactly `cap_bytes` from a longer one
    f.take(cap_bytes.saturating_add(1)).read_to_end(&mut buf)?;
    let truncated = buf.len() as u64 > cap_bytes;
    if truncated {
        buf.truncate(cap_bytes as usize);
    }
    match buf.iter().rposition(|b| *b == b'\n') {
        Some(i) => buf.truncate(i + 1),
        None => buf.clear(),
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
    /// Lines that were not a valid routing row inside [`MAX_LAYERS`] and [`MAX_EXPERTS`].
    pub skipped: usize,
}

/// A guess at the number of experts per layer from the largest id seen: the next of the sizes
/// models actually use. Pass the real number whenever it is known (the model's config).
pub fn guess_experts(max_expert: u32) -> u32 {
    [8, 16, 32, 64, 128, 160, 192, 256, 384, 512, 1024]
        .into_iter()
        .find(|n| *n > max_expert)
        .unwrap_or(max_expert.saturating_add(1))
}

/// Keeps the lines that are a valid routing row: `<call> <row> <layer> <id>[:<gate>]...` with at
/// least one expert, the layer below [`MAX_LAYERS`] and every id below [`MAX_EXPERTS`]. Returns
/// the kept text and how many non-blank lines were dropped.
fn clean(text: &str) -> (String, usize) {
    let mut out = String::with_capacity(text.len());
    let mut skipped = 0usize;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let head = (it.next(), it.next(), it.next());
        let ok = match head {
            (Some(c), Some(r), Some(l)) => {
                c.parse::<u64>().is_ok()
                    && r.parse::<u32>().is_ok()
                    && l.parse::<u32>().is_ok_and(|l| l < MAX_LAYERS)
                    && {
                        let mut n = 0usize;
                        let all = it.all(|t| {
                            n += 1;
                            t.split(':')
                                .next()
                                .and_then(|x| x.parse::<u32>().ok())
                                .is_some_and(|e| e < MAX_EXPERTS)
                        });
                        all && n > 0
                    }
            }
            _ => false,
        };
        if ok {
            out.push_str(line);
            out.push('\n');
        } else {
            skipped += 1;
        }
    }
    (out, skipped)
}

/// Counts what is in `text` without needing the model's dimensions.
pub fn summarize_text(text: &str, bytes: u64) -> Summary {
    let (cleaned, skipped) = clean(text);
    summarize_clean(&cleaned, bytes, skipped)
}

fn summarize_clean(text: &str, bytes: u64, skipped: usize) -> Summary {
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
        layers = layers.max(l + 1); // l < MAX_LAYERS after `clean`
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
        skipped,
    }
}

/// Hit rates of the policies at one cache size (a share of every expert of the model).
#[derive(Debug, Clone, PartialEq)]
pub struct CachePoint {
    pub cache_pct: u32,
    pub lru: f64,
    /// `None` when the cache is too large for this policy to be replayed in reasonable time.
    pub decay_freq: Option<f64>,
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
    /// Tokens actually replayed (at most [`MAX_TOKENS`]).
    pub analysed_tokens: usize,
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
    let (cleaned, skipped) = clean(text);
    let summary = summarize_clean(&cleaned, bytes, skipped);
    let mut notes: Vec<String> = Vec::new();
    // a count of experts the capture contradicts would silently drop its experts
    let mut n_experts = n_experts.clamp(1, MAX_EXPERTS);
    if summary.tokens > 0 && n_experts <= summary.max_expert {
        let guessed = guess_experts(summary.max_expert).min(MAX_EXPERTS);
        notes.push(format!(
            "The capture uses expert {} but {} experts per layer were given: {} were assumed instead.",
            summary.max_expert, n_experts, guessed
        ));
        n_experts = guessed;
    }
    let mut trace = parse_trace(&cleaned, n_experts);
    let analysed_tokens = trace.len().min(MAX_TOKENS);
    trace.truncate(MAX_TOKENS);
    let universe = summary.layers.max(1) as usize * n_experts as usize;
    let too_short = trace.len() < MIN_TOKENS;

    let accesses: usize = trace.iter().map(Vec::len).sum();
    let cache: Vec<CachePoint> = [5u32, 10, 25, 50]
        .into_iter()
        .map(|pct| {
            let cap = (universe * pct as usize / 100).max(1);
            CachePoint {
                cache_pct: pct,
                lru: replay(&trace, &mut Lru::new(cap)),
                decay_freq: (accesses.saturating_mul(cap) <= DECAY_MAX_WORK)
                    .then(|| replay(&trace, &mut DecayFreq::new(cap, 300.0))),
                pinned: pinned_cover(&trace, cap),
                optimal: belady(&trace, cap),
            }
        })
        .collect();

    let prefetch: Vec<PrefetchPoint> = [1usize, 2, 4]
        .into_iter()
        .map(|mult| {
            let s = prefetch::evaluate(
                &trace[..trace.len().min(PREFETCH_TOKENS)],
                n_experts,
                0.5,
                mult,
            );
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

    let mut verdicts = verdicts(
        &summary,
        &cache,
        &prefetch,
        &speculative,
        too_short,
        truncated,
    );
    if summary.skipped > 0 {
        notes.push(format!(
            "{} line(s) were not valid routing rows and were ignored.",
            summary.skipped
        ));
    }
    if summary.tokens > analysed_tokens {
        notes.push(format!(
            "Only the first {analysed_tokens} of {} tokens were analysed.",
            summary.tokens
        ));
    }
    if summary.tokens > 0 {
        // after the honesty flag, before the findings
        let at = usize::from(too_short && !verdicts.is_empty());
        for (i, n) in notes.into_iter().enumerate() {
            verdicts.insert(at + i, n);
        }
    }
    Report {
        summary,
        n_experts,
        analysed_tokens,
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
                "Layers are correlated: a next-layer predictor would have {} of loads ready in time against {} from popularity alone, so cross-layer prefetching can pay off: colibri already has it (PILOT, or COUPLE with a table built from this capture by c/tools/route_pairs.py).",
                pct(p.predicted),
                pct(p.popularity_only)
            ));
        } else {
            v.push(format!(
                "Little correlation between layers was found ({} against {} from popularity alone): prefetching by prediction (PILOT, COUPLE) would not help much.",
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
    let md = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !md.is_file() {
        // a pipe or device would block the read forever
        return Err(format!("{} is not a regular file", path.display()));
    }
    let (text, truncated) = read_capped(path, cap_bytes).map_err(|e| e.to_string())?;
    let ne = n_experts.unwrap_or_else(|| guess_experts(summarize_text(&text, md.len()).max_expert));
    Ok(analyze_text(&text, md.len(), ne, truncated))
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

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("coli-capture-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn hostile_numbers_neither_panic_nor_allocate() {
        let t = "0 0 4294967295 1:0.5\n0 0 0 4294967295:0.5\n0 4294967295 1 1:1\n\
                 18446744073709551615 0 1 2:1\n-1 0 0 3:1\n0 0 99999999999 1:1\n";
        let s = summarize_text(t, t.len() as u64);
        assert_eq!(s.lines, 2, "{s:?}"); // the huge row and call ids are valid; the rest is not
        let r = analyze_text(t, t.len() as u64, u32::MAX, false);
        assert!(r.n_experts <= MAX_EXPERTS);
        for n in [0, 1, u32::MAX] {
            let _ = analyze_text(t, t.len() as u64, n, false);
        }
    }

    #[test]
    fn random_garbage_never_panics() {
        let mut x = 0x2545F4914F6CDD1Du64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let atoms = [
            "0",
            "1",
            "7",
            "255",
            "4294967295",
            "-3",
            "1.5",
            "a",
            ":",
            "3:0.1",
            "9:",
            ":4",
            "",
            "\u{1F980}",
            "99999999999999999999",
        ];
        for _ in 0..300 {
            let mut t = String::new();
            for _ in 0..(rnd() % 40) {
                for _ in 0..(rnd() % 7) {
                    t += atoms[(rnd() % atoms.len() as u64) as usize];
                    t.push(if rnd() % 5 == 0 { '\t' } else { ' ' });
                }
                t.push('\n');
            }
            let n = (rnd() % 600) as u32;
            let r = analyze_text(&t, t.len() as u64, n, false);
            assert!(r.n_experts >= 1 && r.n_experts <= MAX_EXPERTS);
            assert!(!r.verdicts.is_empty());
        }
    }

    #[test]
    fn a_file_of_exactly_the_cap_is_not_called_truncated() {
        let d = scratch("exact");
        let p = d.join("t.txt");
        let t = trace_text(5, 2, 2, 4);
        fs::write(&p, &t).unwrap();
        let r = analyze_file(&p, Some(64), t.len() as u64).unwrap();
        assert!(!r.truncated, "the whole file was read");
        assert_eq!(r.summary.tokens, 5);
        let cut = analyze_file(&p, Some(64), t.len() as u64 - 1).unwrap();
        assert!(cut.truncated);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_line_the_engine_is_still_writing_is_dropped_not_misread() {
        let d = scratch("partial");
        let p = d.join("t.txt");
        // the last line stops after the first digit of expert 45
        fs::write(&p, "0 0 0 1:0.5 2:0.5\n0 0 1 3:0.5 4").unwrap();
        let r = analyze_file(&p, Some(64), DEFAULT_CAP_BYTES).unwrap();
        assert_eq!(r.summary.lines, 1, "{:?}", r.summary);
        assert_eq!(r.summary.max_expert, 2);
        // one huge line with no newline inside the cap leaves nothing to read
        fs::write(&p, "0 0 0 1:0.5 2:0.5 3:0.5 4:0.5 5:0.5").unwrap();
        let r = analyze_file(&p, Some(64), 20).unwrap();
        assert_eq!(r.summary.tokens, 0);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn bytes_that_are_not_utf8_do_not_stop_the_analysis() {
        let d = scratch("utf8");
        let p = d.join("t.txt");
        let mut b = b"0 0 0 1:0.5 2:0.5\n\xff\xfe\x00garbage\n".to_vec();
        b.extend_from_slice(b"1 0 0 1:0.5 2:0.5\n");
        fs::write(&p, &b).unwrap();
        let r = analyze_file(&p, Some(64), DEFAULT_CAP_BYTES).unwrap();
        assert_eq!(r.summary.lines, 2);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_missing_file_is_not_over_the_cap_but_a_directory_is_refused() {
        let d = scratch("missing");
        assert!(
            !over_cap(&d.join("not-yet"), 10),
            "the engine has not created it yet"
        );
        assert!(
            analyze_file(&d, Some(64), 100).is_err(),
            "a directory is not a capture"
        );
        let empty = d.join("empty");
        fs::write(&empty, b"").unwrap();
        let r = analyze_file(&empty, None, 100).unwrap();
        assert_eq!(r.summary.tokens, 0);
        assert!(r.verdicts[0].contains("no routing rows"));
        let _ = fs::remove_dir_all(&d);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_capture_is_followed() {
        let d = scratch("link");
        let real = d.join("real.txt");
        fs::write(&real, trace_text(3, 2, 2, 4)).unwrap();
        let link = d.join("link.txt");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let r = analyze_file(&link, Some(64), DEFAULT_CAP_BYTES).unwrap();
        assert_eq!(r.summary.tokens, 3);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_wrong_expert_count_is_corrected_and_said() {
        let t = trace_text(250, 3, 2, 40);
        // the real ids reach well above 8
        let r = analyze_text(&t, t.len() as u64, 8, false);
        assert!(r.n_experts > r.summary.max_expert);
        assert!(
            r.verdicts
                .iter()
                .any(|v| v.contains("were assumed instead"))
        );
        assert!(r.summary.tokens > 0);
    }

    #[test]
    fn a_long_capture_is_cut_to_a_bounded_replay_and_says_so() {
        let t = trace_text(MAX_TOKENS + 500, 2, 2, 8);
        let r = analyze_text(&t, t.len() as u64, 64, false);
        assert_eq!(r.analysed_tokens, MAX_TOKENS);
        assert!(r.verdicts.iter().any(|v| v.contains("Only the first")));
    }

    #[test]
    fn the_slow_policy_is_skipped_for_a_large_cache() {
        let small = trace_text(300, 4, 2, 8);
        let r = analyze_text(&small, small.len() as u64, 64, false);
        assert!(r.cache.iter().all(|c| c.decay_freq.is_some()));
        let big = trace_text(300, 40, 8, 64);
        let r = analyze_text(&big, big.len() as u64, 256, false);
        assert!(r.cache.iter().any(|c| c.decay_freq.is_none()));
    }
}
