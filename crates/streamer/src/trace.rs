//! Real routing data. Two inputs, both plain text:
//! * a routing trace, one line per (call, row, layer): `<call> <row> <layer> <expert>:<gate> ...`
//! * an expert history, `<layer> <expert> <count>` per line after two negative header records.
//!
//! A trace has the order of events, so recency and prefetching can be judged on it. A history has
//! only popularity: a trace sampled from it has no temporal locality at all, so it understates
//! what recency-based policies reach on real conversations.

use crate::policy::Step;
use std::collections::HashMap;

/// Dense id of an expert across all layers: `layer * n_experts + expert`. `None` when the pair is
/// not a real expert (`expert >= n_experts`, which would alias an expert of the next layer, or
/// `n_experts == 0`) or when the id does not fit a u32.
pub fn try_dense(layer: u32, expert: u32, n_experts: u32) -> Option<u32> {
    if expert >= n_experts {
        return None;
    }
    u32::try_from(layer as u64 * n_experts as u64 + expert as u64).ok()
}

/// Splits a dense id back into `(layer, expert)`; `None` when `n_experts` is 0.
pub fn split_dense(id: u32, n_experts: u32) -> Option<(u32, u32)> {
    if n_experts == 0 {
        return None;
    }
    Some((id / n_experts, id % n_experts))
}

/// Parses a routing trace into tokens. A row starts a new token when its layer index does not
/// increase, since a token visits the layers in order. Lines that do not parse, experts outside
/// `0..n_experts` and layers whose ids would not fit a u32 are skipped.
pub fn parse_trace(text: &str, n_experts: u32) -> Vec<Step> {
    let mut cur: HashMap<u32, (u32, Step)> = HashMap::new();
    let mut out = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(_call), Some(row), Some(layer)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        let (Ok(row), Ok(layer)) = (row.parse::<u32>(), layer.parse::<u32>()) else {
            continue;
        };
        let ids: Vec<u32> = it
            .filter_map(|t| t.split(':').next()?.parse::<u32>().ok())
            .filter(|e| *e < n_experts)
            .filter_map(|e| try_dense(layer, e, n_experts))
            .collect();
        let slot = cur.entry(row).or_insert((layer, Vec::new()));
        if layer <= slot.0 && !slot.1.is_empty() {
            out.push(std::mem::take(&mut slot.1));
        }
        slot.0 = layer;
        slot.1.extend(ids);
    }
    let mut rest: Vec<_> = cur.into_iter().collect();
    rest.sort_by_key(|(r, _)| *r);
    out.extend(
        rest.into_iter()
            .map(|(_, (_, s))| s)
            .filter(|s| !s.is_empty()),
    );
    out
}

/// Largest input file read whole (1 GiB, about 300 000 tokens of a 40-layer trace).
pub const MAX_FILE_BYTES: u64 = 1 << 30;

/// Reads a text file of at most `MAX_FILE_BYTES`. Only a regular file is accepted: a device or a
/// pipe (`/dev/zero`) has no end and would be read until memory runs out. The read itself is
/// bounded too, so a file that keeps growing while it is read cannot get past the limit.
pub fn read_text(path: &str) -> Result<String, String> {
    read_text_limit(path, MAX_FILE_BYTES)
}

fn read_text_limit(path: &str, limit: u64) -> Result<String, String> {
    use std::io::Read;
    let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let meta = f.metadata().map_err(|e| format!("{path}: {e}"))?;
    if !meta.is_file() {
        return Err(format!("{path}: not a regular file"));
    }
    let mut text = String::new();
    f.take(limit + 1)
        .read_to_string(&mut text)
        .map_err(|e| format!("{path}: {e}"))?;
    if text.len() as u64 > limit {
        return Err(format!("{path}: larger than {limit} bytes"));
    }
    Ok(text)
}

/// Reads and parses a routing-trace file; an error (never a panic) for an unreadable, oversized
/// or non-UTF-8 file, and for one that holds no tokens (wrong file, or wrong `n_experts`).
pub fn read_trace_file(path: &str, n_experts: u32) -> Result<Vec<Step>, String> {
    let text = read_text(path)?;
    let t = parse_trace(&text, n_experts);
    if t.is_empty() {
        return Err(format!(
            "{path}: no routing tokens found (is it a ROUTE_TRACE file, and is {n_experts} the right number of experts?)"
        ));
    }
    Ok(t)
}

/// Reads and parses a `.coli_usage` file with the same error behaviour.
pub fn read_usage_file(path: &str) -> Result<Usage, String> {
    let text = read_text(path)?;
    parse_usage(&text).map_err(|e| format!("{path}: {e}"))
}

/// A history larger than this many (layer, expert) cells is refused rather than allocated.
pub const MAX_CELLS: u64 = 1 << 24;

/// Per-expert counts from a history file.
#[derive(Debug, Default, PartialEq)]
pub struct Usage {
    pub n_layers: u32,
    pub n_experts: u32,
    /// `counts[layer * n_experts + expert]`
    pub counts: Vec<u64>,
}

pub fn parse_usage(text: &str) -> Result<Usage, String> {
    let mut head: Option<(u32, u32)> = None;
    let mut rows: Vec<(u32, u32, u64)> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 3 {
            continue;
        }
        let (Ok(a), Ok(b), Ok(c)) = (
            f[0].parse::<i64>(),
            f[1].parse::<i64>(),
            f[2].parse::<u64>(),
        ) else {
            continue;
        };
        match a {
            -1 => {
                head = match (u32::try_from(b), u32::try_from(c)) {
                    (Ok(l), Ok(e)) if l > 0 && e > 0 => Some((l, e)),
                    _ => return Err(format!("bad header record `-1 {b} {c}`")),
                };
            }
            l if l >= 0 && b >= 0 => {
                if let (Ok(l), Ok(e)) = (u32::try_from(l), u32::try_from(b)) {
                    rows.push((l, e, c));
                }
            }
            _ => {}
        }
    }
    let Some((nl, ne)) = head else {
        return Err("no `-1 <layers> <experts>` header record".into());
    };
    let cells = nl as u64 * ne as u64;
    if cells > MAX_CELLS {
        return Err(format!(
            "{nl} layers x {ne} experts is over the {MAX_CELLS}-cell limit"
        ));
    }
    let mut counts = vec![0u64; cells as usize];
    for (l, e, c) in rows {
        if l < nl && e < ne {
            let cell = &mut counts[l as usize * ne as usize + e as usize];
            *cell = cell.saturating_add(c);
        }
    }
    Ok(Usage {
        n_layers: nl,
        n_experts: ne,
        counts,
    })
}

impl Usage {
    pub fn total(&self) -> u64 {
        self.counts.iter().fold(0u64, |a, c| a.saturating_add(*c))
    }

    /// Share of all uses a fixed set of the `cap` most used experts would serve: what pinning
    /// the hottest experts (colibri's learning cache) reaches at best on this history. 0.0 for a
    /// history with no uses at all (nothing is served; check `total()` to tell the cases apart).
    pub fn static_coverage(&self, cap: usize) -> f64 {
        let all: f64 = self.counts.iter().map(|x| *x as f64).sum(); // f64: u64 sums would saturate
        let cap = cap.min(self.counts.len());
        if cap == 0 || all <= 0.0 {
            return 0.0;
        }
        let mut c = self.counts.clone();
        if cap < c.len() {
            // the `cap` largest end up in c[..cap], unordered: O(n) instead of a full sort
            c.select_nth_unstable_by(cap - 1, |a, b| b.cmp(a));
        }
        let top: f64 = c[..cap].iter().map(|x| *x as f64).sum();
        (top / all).min(1.0)
    }

    /// A trace of `tokens` steps: in every layer, `k` distinct experts drawn by popularity
    /// (every expert has weight count + 1, so a zero count is still possible; for counts above
    /// 2^53 that +1 is below f64 resolution and such an expert is no longer drawn). Independent
    /// draws, so it carries no recency structure. Always terminates: after a bounded number of
    /// draws a layer is completed with the lowest-numbered experts not yet chosen.
    ///
    /// An error when the history is inconsistent or the result would hold more than
    /// `MAX_TRACE_IDS` ids (memory is bounded by the caller's request, so it must be refused).
    pub fn try_sample_trace(
        &self,
        k: usize,
        tokens: usize,
        seed: u64,
    ) -> Result<Vec<Step>, String> {
        let (nl, ne) = (self.n_layers as usize, self.n_experts as usize);
        if nl == 0 || ne == 0 {
            return Ok(vec![Vec::new(); tokens.min(MAX_TRACE_IDS as usize)]);
        }
        if nl as u64 * ne as u64 > MAX_CELLS || self.counts.len() != nl * ne {
            return Err(format!(
                "inconsistent history: {nl} layers x {ne} experts, {} counts",
                self.counts.len()
            ));
        }
        let want = k.min(ne);
        let ids = (tokens as u128) * (nl as u128) * (want as u128);
        if ids > MAX_TRACE_IDS as u128 {
            return Err(format!(
                "{tokens} tokens x {nl} layers x {want} experts is {ids} ids, over the {MAX_TRACE_IDS} limit"
            ));
        }
        let mut s = seed | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        // cumulative weights in f64: no integer overflow, however large the counts
        let cum: Vec<Vec<f64>> = (0..nl)
            .map(|l| {
                let mut acc = 0.0f64;
                (0..ne)
                    .map(|e| {
                        acc += self.counts[l * ne + e] as f64 + 1.0;
                        acc
                    })
                    .collect()
            })
            .collect();
        Ok((0..tokens)
            .map(|_| {
                let mut step = Vec::with_capacity(nl * want);
                for (l, c) in cum.iter().enumerate() {
                    let tot = c[ne - 1];
                    let mut got: Vec<u32> = Vec::with_capacity(want);
                    let mut tries = 0usize;
                    while got.len() < want && tries < 64 * want {
                        tries += 1;
                        let x = rnd() * tot;
                        let e = c.partition_point(|v| *v <= x).min(ne - 1) as u32;
                        if !got.contains(&e) {
                            got.push(e);
                        }
                    }
                    let mut e = 0u32;
                    while got.len() < want {
                        if !got.contains(&e) {
                            got.push(e);
                        }
                        e += 1;
                    }
                    // cannot be None: nl * ne <= MAX_CELLS fits a u32 id and e < ne
                    step.extend(
                        got.into_iter()
                            .filter_map(|e| try_dense(l as u32, e, self.n_experts)),
                    );
                }
                step
            })
            .collect())
    }

    /// `try_sample_trace`, with an error turned into an empty trace (check `.is_empty()`).
    pub fn sample_trace(&self, k: usize, tokens: usize, seed: u64) -> Vec<Step> {
        self.try_sample_trace(k, tokens, seed).unwrap_or_default()
    }
}

/// A sampled trace may hold at most this many expert ids (2^26, 256 MB of u32).
pub const MAX_TRACE_IDS: u64 = 1 << 26;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trace_splits_into_tokens_where_the_layer_index_wraps() {
        let t = "0 0 0 1:0.6 2:0.4\n1 0 1 3:1.0\n2 0 0 1:0.5 4:0.5\n3 0 1 3:1.0\n";
        let steps = parse_trace(t, 10);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0], vec![1, 2, 13]);
        assert_eq!(steps[1], vec![1, 4, 13]);
    }

    #[test]
    fn reads_a_history_and_ignores_the_header_records_it_does_not_need() {
        let u = parse_usage("-1 2 4\n-2 1 99\n0 0 5\n0 3 1\n1 2 10\n").unwrap();
        assert_eq!((u.n_layers, u.n_experts, u.total()), (2, 4, 16));
        assert_eq!(u.counts[3], 1);
        assert_eq!(u.counts[6], 10);
        assert!(parse_usage("0 0 5\n").is_err());
    }

    #[test]
    fn pinning_the_hottest_covers_what_it_should() {
        let u = parse_usage("-1 1 4\n0 0 70\n0 1 20\n0 2 6\n0 3 4\n").unwrap();
        assert!((u.static_coverage(1) - 0.7).abs() < 1e-9);
        assert!((u.static_coverage(2) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn sampling_follows_popularity() {
        let u = parse_usage("-1 1 4\n0 0 1000\n0 1 1\n0 2 1\n0 3 1\n").unwrap();
        let tr = u.sample_trace(1, 500, 3);
        let zeros = tr.iter().filter(|s| s[0] == 0).count();
        assert!(zeros > 450, "{zeros}");
    }

    #[test]
    fn a_huge_header_is_an_error_not_a_capacity_overflow() {
        assert!(parse_usage("-1 4000000000 4000000000\n").is_err());
        assert!(parse_usage("-1 -3 4\n").is_err());
        assert!(parse_usage("-1 0 4\n").is_err());
        assert!(parse_usage("-1 4 99999999999\n").is_err());
    }

    #[test]
    fn counts_at_the_u64_limit_neither_panic_nor_hang_when_sampled() {
        let u = parse_usage(&format!(
            "-1 2 4\n0 0 {m}\n0 0 {m}\n0 1 {m}\n1 3 {m}\n7 7 5\n-4 0 1\n0 -2 1\n",
            m = u64::MAX
        ))
        .unwrap();
        assert_eq!(u.counts[0], u64::MAX); // duplicates saturate
        assert_eq!(u.total(), u64::MAX);
        let tr = u.sample_trace(4, 20, 5);
        assert_eq!(tr.len(), 20);
        for s in &tr {
            assert_eq!(s.len(), 8); // 4 distinct experts in each of 2 layers
            assert!(s.iter().all(|id| *id < 8));
        }
        // three cells hold u64::MAX each: pinning one of them covers a third, not everything
        assert!(
            (u.static_coverage(1) - 1.0 / 3.0).abs() < 1e-9,
            "{}",
            u.static_coverage(1)
        );
    }

    #[test]
    fn asking_for_more_experts_than_exist_terminates_with_all_of_them() {
        let u = parse_usage("-1 1 3\n0 0 1000000000000\n").unwrap();
        let tr = u.sample_trace(50, 3, 1);
        for s in &tr {
            let mut v = s.clone();
            v.sort_unstable();
            assert_eq!(v, vec![0, 1, 2]);
        }
        assert_eq!(
            Usage::default().sample_trace(3, 2, 1),
            vec![Vec::<u32>::new(); 2]
        );
    }

    #[test]
    fn a_history_with_no_uses_covers_nothing() {
        let u = parse_usage("-1 2 2\n").unwrap();
        assert_eq!(u.total(), 0);
        assert_eq!(u.static_coverage(3), 0.0);
    }

    #[test]
    fn a_layer_index_that_overflows_u32_ids_is_skipped_not_wrapped() {
        // 4_000_000_000 * 256 wraps in u32 and would alias a real expert
        let t = "0 0 4000000000 3:1.0\n1 0 0 5:1.0\n";
        let steps = parse_trace(t, 256);
        assert_eq!(steps, vec![vec![5]]);
        assert_eq!(try_dense(4_000_000_000, 3, 256), None);
        assert_eq!(try_dense(2, 3, 256), Some(515));
    }

    #[test]
    fn odd_lines_are_skipped() {
        let t = "\n   \n0\n0 0\n0 0 0 NaN:x -1:0.5 9999999999:1 3:0.5\nx y z\n0 0 0 3:1.0\n";
        let steps = parse_trace(t, 8);
        // the second `0 0 0` line repeats layer 0 for row 0, so it starts a new token
        assert_eq!(steps, vec![vec![3], vec![3]]);
    }

    #[test]
    fn reading_a_missing_non_utf8_or_empty_file_is_an_error() {
        let dir = std::env::temp_dir().join(format!("tr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("bin.txt");
        std::fs::write(&bin, [0xffu8, 0xfe, b'\n']).unwrap();
        assert!(read_trace_file(bin.to_str().unwrap(), 8).is_err());
        assert!(read_usage_file(bin.to_str().unwrap()).is_err());
        assert!(read_trace_file("/nonexistent/x", 8).is_err());
        let junk = dir.join("junk.txt");
        std::fs::write(&junk, "hello world\n").unwrap();
        let e = read_trace_file(junk.to_str().unwrap(), 8).unwrap_err();
        assert!(e.contains("no routing tokens"), "{e}");
        let _ = std::fs::remove_dir_all(dir);
    }

    // --- independent checks of the parsers (random text, brute-force references) ---

    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    #[test]
    fn random_text_never_panics_either_parser_and_yields_only_valid_ids() {
        let mut r = xorshift(2718);
        let words = [
            "0",
            "1",
            "2",
            "7",
            "255",
            "256",
            "-1",
            "-2",
            "4294967295",
            "4294967296",
            "18446744073709551615",
            "18446744073709551616",
            "NaN",
            "inf",
            "1e9",
            "a:b",
            ":",
            "::",
            "3:0.5",
            "255:1",
            "256:1",
            "-3:1",
            "0x10",
            "1:1:1",
            "",
            "é",
            "\t",
            "99999999999:2",
        ];
        for _ in 0..3000 {
            let lines = r() % 8;
            let mut text = String::new();
            for _ in 0..lines {
                for _ in 0..(r() % 7) {
                    text.push_str(words[(r() % words.len() as u64) as usize]);
                    text.push(if r() % 5 == 0 { '\t' } else { ' ' });
                }
                text.push('\n');
            }
            for ne in [0u32, 1, 256, u32::MAX] {
                for step in parse_trace(&text, ne) {
                    assert!(!step.is_empty());
                    for id in step {
                        let (l, e) = split_dense(id, ne).expect("ne > 0 when an id exists");
                        assert!(e < ne, "expert {e} of {ne} in {text:?}");
                        assert_eq!(try_dense(l, e, ne), Some(id));
                    }
                }
            }
            if let Ok(u) = parse_usage(&text) {
                assert_eq!(
                    u.counts.len() as u64,
                    u.n_layers as u64 * u.n_experts as u64
                );
                assert!(u.counts.len() as u64 <= MAX_CELLS);
            }
        }
    }

    #[test]
    fn a_wellformed_multi_row_trace_parses_to_exactly_the_tokens_that_wrote_it() {
        let mut r = xorshift(31337);
        for _ in 0..300 {
            let ne = 2 + (r() % 30) as u32;
            let rows = 1 + (r() % 4) as u32;
            let mut expected: Vec<Vec<u32>> = Vec::new();
            // per row: a list of tokens, each a list of (layer, experts)
            type Token = Vec<(u32, Vec<u32>)>;
            let mut per_row: Vec<Vec<Token>> = vec![Vec::new(); rows as usize];
            for row in 0..rows {
                for _ in 0..(1 + r() % 6) {
                    let mut token = Vec::new();
                    let mut ids = Vec::new();
                    let mut layer = 0u32; // a new token must restart at a layer not above the last one
                    for _ in 0..(1 + r() % 5) {
                        let mut es: Vec<u32> = (0..(1 + r() % 3))
                            .map(|_| (r() % ne as u64) as u32)
                            .collect();
                        es.sort_unstable();
                        es.dedup();
                        ids.extend(es.iter().map(|e| layer * ne + e));
                        token.push((layer, es));
                        layer += 1 + (r() % 2) as u32;
                    }
                    per_row[row as usize].push(token);
                    expected.push(ids);
                }
            }
            // interleave the rows' lines in a random order, each row keeping its own order
            let mut cursor = vec![(0usize, 0usize); rows as usize];
            let mut text = String::new();
            let mut call = 0;
            loop {
                let live: Vec<usize> = (0..rows as usize)
                    .filter(|&x| cursor[x].0 < per_row[x].len())
                    .collect();
                if live.is_empty() {
                    break;
                }
                let x = live[(r() % live.len() as u64) as usize];
                let (t, l) = cursor[x];
                let (layer, es) = &per_row[x][t][l];
                let gates: Vec<String> = es.iter().map(|e| format!("{e}:0.5")).collect();
                text.push_str(&format!("{call} {x} {layer} {}\n", gates.join(" ")));
                call += 1;
                cursor[x] = if l + 1 < per_row[x][t].len() {
                    (t, l + 1)
                } else {
                    (t + 1, 0)
                };
            }
            let mut got = parse_trace(&text, ne);
            got.sort();
            expected.sort();
            assert_eq!(got, expected, "{text}");
        }
    }

    #[test]
    fn parse_usage_matches_a_brute_force_count_including_duplicates_and_strays() {
        let mut r = xorshift(99);
        for _ in 0..500 {
            let (nl, ne) = (1 + (r() % 5) as u32, 1 + (r() % 7) as u32);
            let mut text = format!("-1 {nl} {ne}\n-2 1 777\n");
            let mut want = vec![0u64; (nl * ne) as usize];
            for _ in 0..(r() % 40) {
                let (l, e, c) = ((r() % 8) as i64, (r() % 10) as i64 - 1, r() % 1000);
                text.push_str(&format!("{l} {e} {c}\n"));
                if (0..nl as i64).contains(&l) && (0..ne as i64).contains(&e) {
                    want[(l as u32 * ne + e as u32) as usize] += c;
                }
            }
            let u = parse_usage(&text).unwrap();
            assert_eq!((u.n_layers, u.n_experts), (nl, ne));
            assert_eq!(u.counts, want, "{text}");
            assert_eq!(u.total(), want.iter().sum::<u64>());
        }
    }

    #[test]
    fn static_coverage_matches_sorting_everything() {
        let mut r = xorshift(8);
        for _ in 0..500 {
            let n = 1 + (r() % 40) as usize;
            let counts: Vec<u64> = (0..n)
                .map(|_| if r() % 4 == 0 { 0 } else { r() % 1000 })
                .collect();
            let u = Usage {
                n_layers: 1,
                n_experts: n as u32,
                counts: counts.clone(),
            };
            let mut sorted = counts.clone();
            sorted.sort_unstable_by(|a, b| b.cmp(a));
            let all: u64 = counts.iter().sum();
            for cap in [0usize, 1, 2, n / 2, n - 1, n, n + 5, usize::MAX] {
                let top: u64 = sorted.iter().take(cap).sum();
                let want = if all == 0 {
                    0.0
                } else {
                    top as f64 / all as f64
                };
                let got = u.static_coverage(cap);
                assert!(
                    (got - want).abs() < 1e-12,
                    "cap {cap} {counts:?}: {got} vs {want}"
                );
                assert!((0.0..=1.0).contains(&got));
            }
        }
    }

    #[test]
    fn sampled_experts_follow_their_weights() {
        // weights are count + 1: 1001 : 101 : 11 : 2
        let u = parse_usage("-1 1 4\n0 0 1000\n0 1 100\n0 2 10\n0 3 1\n").unwrap();
        let tokens = 40_000;
        let tr = u.try_sample_trace(1, tokens, 7).unwrap();
        let mut seen = [0usize; 4];
        for s in &tr {
            assert_eq!(s.len(), 1);
            seen[s[0] as usize] += 1;
        }
        let total = 1001.0 + 101.0 + 11.0 + 2.0;
        for (e, w) in [1001.0, 101.0, 11.0, 2.0].iter().enumerate() {
            let got = seen[e] as f64 / tokens as f64;
            assert!(
                (got - w / total).abs() < 0.01,
                "expert {e}: {got} vs {}",
                w / total
            );
        }
    }

    #[test]
    fn a_sampled_step_has_k_distinct_experts_in_every_layer_and_nowhere_else() {
        let u = parse_usage("-1 3 6\n0 0 50\n1 2 7\n2 5 9\n").unwrap();
        for k in [1usize, 2, 5, 6, 40] {
            for s in u.try_sample_trace(k, 30, k as u64).unwrap() {
                let want = k.min(6);
                assert_eq!(s.len(), 3 * want);
                for (l, chunk) in s.chunks(want).enumerate() {
                    let mut v: Vec<u32> = chunk.to_vec();
                    v.sort_unstable();
                    v.dedup();
                    assert_eq!(v.len(), want, "k {k}");
                    assert!(v.iter().all(|id| *id / 6 == l as u32));
                }
            }
        }
    }

    #[test]
    fn a_sampling_request_that_would_not_fit_memory_is_refused() {
        let u = parse_usage("-1 40 256\n").unwrap();
        assert!(u.try_sample_trace(256, 10_000, 1).is_err()); // 102M ids
        assert!(u.sample_trace(256, 10_000, 1).is_empty());
        assert!(u.try_sample_trace(8, usize::MAX, 1).is_err());
        assert!(
            u.try_sample_trace(usize::MAX, 10, 1).is_ok()
                || u.try_sample_trace(usize::MAX, 1, 1).is_err()
        );
        let bad = Usage {
            n_layers: 2,
            n_experts: 3,
            counts: vec![1; 5],
        };
        assert!(bad.try_sample_trace(1, 1, 1).is_err());
        let huge = Usage {
            n_layers: u32::MAX,
            n_experts: u32::MAX,
            counts: Vec::new(),
        };
        assert!(huge.try_sample_trace(1, 1, 1).is_err());
        assert_eq!(Usage::default().try_sample_trace(1, 3, 1).unwrap().len(), 3);
    }

    #[test]
    fn only_a_regular_file_within_the_limit_is_read() {
        let dir = std::env::temp_dir().join(format!("rt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("ten.txt");
        std::fs::write(&f, "0123456789").unwrap();
        let p = f.to_str().unwrap();
        assert_eq!(read_text_limit(p, 10).unwrap(), "0123456789");
        assert!(read_text_limit(p, 9).unwrap_err().contains("larger than"));
        assert!(read_text(dir.to_str().unwrap()).is_err()); // a directory
        #[cfg(unix)]
        {
            let e = read_text("/dev/zero").unwrap_err();
            assert!(e.contains("not a regular file"), "{e}");
            assert!(read_text("/dev/null").is_err());
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}
