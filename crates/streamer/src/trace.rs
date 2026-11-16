//! Real routing data. Two inputs, both plain text:
//! * a routing trace, one line per (call, row, layer): `<call> <row> <layer> <expert>:<gate> ...`
//! * an expert history, `<layer> <expert> <count>` per line after two negative header records.
//!
//! A trace has the order of events, so recency and prefetching can be judged on it. A history has
//! only popularity: a trace sampled from it has no temporal locality at all, so it understates
//! what recency-based policies reach on real conversations.

use crate::policy::Step;
use std::collections::HashMap;

/// Dense id of an expert across all layers.
pub fn dense(layer: u32, expert: u32, n_experts: u32) -> u32 {
    layer * n_experts + expert
}

/// Parses a routing trace into tokens. A row starts a new token when its layer index does not
/// increase, since a token visits the layers in order.
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
            .map(|e| dense(layer, e, n_experts))
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

/// Per-expert counts from a history file.
#[derive(Debug, Default, PartialEq)]
pub struct Usage {
    pub n_layers: u32,
    pub n_experts: u32,
    /// `counts[layer * n_experts + expert]`
    pub counts: Vec<u64>,
}

pub fn parse_usage(text: &str) -> Result<Usage, String> {
    let (mut nl, mut ne) = (0u32, 0u32);
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
                nl = b as u32;
                ne = c as u32;
            }
            l if l >= 0 && b >= 0 => rows.push((l as u32, b as u32, c)),
            _ => {}
        }
    }
    if nl == 0 || ne == 0 {
        return Err("no `-1 <layers> <experts>` header record".into());
    }
    let mut counts = vec![0u64; nl as usize * ne as usize];
    for (l, e, c) in rows {
        if l < nl && e < ne {
            counts[(l * ne + e) as usize] += c;
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
        self.counts.iter().sum()
    }

    /// Share of all uses a fixed set of the `cap` most used experts would serve: what pinning
    /// the hottest experts (colibri's learning cache) reaches at best on this history.
    pub fn static_coverage(&self, cap: usize) -> f64 {
        let mut c = self.counts.clone();
        c.sort_unstable_by(|a, b| b.cmp(a));
        let top: u64 = c.iter().take(cap).sum();
        match self.total() {
            0 => 1.0,
            t => top as f64 / t as f64,
        }
    }

    /// A trace of `tokens` steps: in every layer, `k` distinct experts drawn by popularity.
    /// Independent draws, so it carries no recency structure.
    pub fn sample_trace(&self, k: usize, tokens: usize, seed: u64) -> Vec<Step> {
        let mut s = seed | 1;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let ne = self.n_experts as usize;
        let cum: Vec<Vec<u64>> = (0..self.n_layers as usize)
            .map(|l| {
                let mut acc = 0;
                self.counts[l * ne..(l + 1) * ne]
                    .iter()
                    .map(|c| {
                        acc += c + 1; // +1 keeps every expert reachable
                        acc
                    })
                    .collect()
            })
            .collect();
        (0..tokens)
            .map(|_| {
                let mut step = Vec::new();
                for (l, c) in cum.iter().enumerate() {
                    let tot = *c.last().unwrap();
                    let mut got: Vec<u32> = Vec::new();
                    while got.len() < k.min(ne) {
                        let x = (rnd() * tot as f64) as u64;
                        let e = c.partition_point(|v| *v <= x) as u32;
                        if !got.contains(&e) {
                            got.push(e);
                        }
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
}
