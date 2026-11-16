//! Real routing data. Two inputs, both plain text:
//! * a routing trace, one line per (call, row, layer): `<call> <row> <layer> <expert>:<gate> ...`
//! * an expert history, `<layer> <expert> <count>` per line after two negative header records.
//!
//! A trace has the order of events, so recency and prefetching can be judged on it. A history has
//! only popularity: a trace sampled from it has no temporal locality at all, so it understates
//! what recency-based policies reach on real conversations.

use crate::policy::Step;
use std::collections::HashMap;

/// Dense id of an expert across all layers. Saturates instead of wrapping: a layer index from a
/// corrupt file must never alias a real expert (see `try_dense`).
pub fn dense(layer: u32, expert: u32, n_experts: u32) -> u32 {
    try_dense(layer, expert, n_experts).unwrap_or(u32::MAX)
}

/// `layer * n_experts + expert` when it fits a u32 id.
pub fn try_dense(layer: u32, expert: u32, n_experts: u32) -> Option<u32> {
    let id = layer as u64 * n_experts as u64 + expert as u64;
    u32::try_from(id).ok()
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

/// Reads and parses a routing-trace file; an error (never a panic) for an unreadable or non-UTF-8
/// file, and for one that holds no tokens (wrong file, or wrong `n_experts`).
pub fn read_trace_file(path: &str, n_experts: u32) -> Result<Vec<Step>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
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
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
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
        let mut c = self.counts.clone();
        c.sort_unstable_by(|a, b| b.cmp(a));
        // summed in f64: u64 sums of huge counts would saturate and report 100%
        let top: f64 = c.iter().take(cap).map(|x| *x as f64).sum();
        let all: f64 = self.counts.iter().map(|x| *x as f64).sum();
        if all > 0.0 { top / all } else { 0.0 }
    }

    /// A trace of `tokens` steps: in every layer, `k` distinct experts drawn by popularity
    /// (every expert keeps weight count + 1, so none is unreachable). Independent draws, so it
    /// carries no recency structure. Always terminates: after a bounded number of draws a layer
    /// is completed with the lowest-numbered experts not yet chosen.
    pub fn sample_trace(&self, k: usize, tokens: usize, seed: u64) -> Vec<Step> {
        let (nl, ne) = (self.n_layers as usize, self.n_experts as usize);
        if nl == 0 || ne == 0 {
            return vec![Vec::new(); tokens];
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
                        let c = self.counts.get(l * ne + e).copied().unwrap_or(0);
                        acc += c as f64 + 1.0;
                        acc
                    })
                    .collect()
            })
            .collect();
        let want = k.min(ne);
        (0..tokens)
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
                    step.extend(got.into_iter().map(|e| dense(l as u32, e, self.n_experts)));
                }
                step
            })
            .collect()
    }
}

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
        assert_eq!(dense(4_000_000_000, 3, 256), u32::MAX);
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
}
