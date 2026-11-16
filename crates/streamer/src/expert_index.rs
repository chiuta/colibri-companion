//! Where every routed expert lives. Mixture-of-experts checkpoints name their expert weights like
//! `model.layers.12.mlp.experts.37.up_proj.weight`; all tensors that share a (layer, expert) pair
//! form one unit that is read together.

use crate::safetensors::Tensor;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertId {
    pub layer: u32,
    pub expert: u32,
}

/// `(layer, expert)` when `name` is a per-expert tensor (`...layers.12.mlp.experts.37.up_proj`).
pub fn expert_of(name: &str) -> Option<ExpertId> {
    let parts: Vec<&str> = name.split('.').collect();
    for w in 0..parts.len().saturating_sub(3) {
        if parts[w] == "layers" && parts.get(w + 3) == Some(&"experts") {
            let layer = parts[w + 1].parse().ok()?;
            let expert = parts.get(w + 4)?.parse().ok()?;
            return Some(ExpertId { layer, expert });
        }
    }
    None
}

/// The layer when `name` is a PACKED expert tensor: one tensor holding all of a layer's experts
/// stacked along its first dimension, as in `...layers.12.mlp.experts.gate_up_proj` (newer
/// transformers layouts, gpt-oss, Qwen3 MoE checkpoints re-exported that way). The first
/// dimension of its shape is the expert count; that is the convention, it cannot be verified from
/// the name, so a layout with the experts on another axis would be split wrongly.
pub fn packed_layer_of(name: &str) -> Option<u32> {
    let parts: Vec<&str> = name.split('.').collect();
    for w in 0..parts.len().saturating_sub(3) {
        if parts[w] == "layers" && parts.get(w + 3) == Some(&"experts") {
            let layer = parts[w + 1].parse().ok()?;
            let tail = parts.get(w + 4)?;
            // a number there is the per-expert layout, handled by `expert_of`
            return tail.parse::<u32>().is_err().then_some(layer);
        }
    }
    None
}

/// A packed tensor claiming more experts than this is not split (it is counted as resident).
pub const MAX_PACKED_EXPERTS: u64 = 1 << 16;

/// The index holds at most this many experts; tensors that would add more are counted as
/// resident, so a hostile header cannot make `build` allocate without bound.
pub const MAX_EXPERT_ENTRIES: usize = 1 << 22;

/// The byte ranges of one expert and their total size.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExpertSpan {
    pub ranges: Vec<(u64, u64)>,
    pub bytes: u64,
}

#[derive(Debug, Default)]
pub struct ExpertIndex {
    pub experts: BTreeMap<ExpertId, ExpertSpan>,
    /// Bytes of tensors that are not routed experts (attention, embeddings, shared parts).
    pub resident_bytes: u64,
}

impl ExpertIndex {
    pub fn build(tensors: &[Tensor]) -> Self {
        Self::build_with_limit(tensors, MAX_EXPERT_ENTRIES)
    }

    pub(crate) fn build_with_limit(tensors: &[Tensor], max_entries: usize) -> Self {
        let mut idx = Self::default();
        for t in tensors {
            if let Some(id) = expert_of(&t.name) {
                if idx.experts.contains_key(&id) || idx.experts.len() < max_entries {
                    idx.add(id, t.start, t.end);
                    continue;
                }
            } else if let Some(layer) = packed_layer_of(&t.name) {
                if idx.add_packed(layer, t, max_entries) {
                    continue;
                }
            }
            idx.resident_bytes = idx.resident_bytes.saturating_add(t.len());
        }
        for s in idx.experts.values_mut() {
            s.ranges.sort_unstable();
        }
        idx
    }

    fn add(&mut self, id: ExpertId, start: u64, end: u64) {
        let s = self.experts.entry(id).or_default();
        s.ranges.push((start, end));
        s.bytes = s.bytes.saturating_add(end.saturating_sub(start));
    }

    /// Splits a packed tensor into one equal slice per expert. False (and nothing added) when it
    /// cannot be split: no shape, an expert count of 0 or above `MAX_PACKED_EXPERTS`, bytes that
    /// do not divide evenly, an empty or inverted range, or more entries than the limit allows.
    fn add_packed(&mut self, layer: u32, t: &Tensor, max_entries: usize) -> bool {
        let n = match t.shape.first() {
            Some(n) if (1..=MAX_PACKED_EXPERTS).contains(n) => *n,
            _ => return false,
        };
        let len = t.len();
        if len == 0 || len % n != 0 {
            return false;
        }
        let new = (0..n as u32)
            .filter(|e| !self.experts.contains_key(&ExpertId { layer, expert: *e }))
            .count();
        if self.experts.len() + new > max_entries {
            return false;
        }
        let each = len / n;
        for e in 0..n {
            let (Some(a), Some(b)) = (
                e.checked_mul(each).and_then(|o| t.start.checked_add(o)),
                (e + 1)
                    .checked_mul(each)
                    .and_then(|o| t.start.checked_add(o)),
            ) else {
                return false;
            };
            self.add(
                ExpertId {
                    layer,
                    expert: e as u32,
                },
                a,
                b,
            );
        }
        true
    }

    pub fn expert_bytes(&self) -> u64 {
        self.experts
            .values()
            .fold(0u64, |a, s| a.saturating_add(s.bytes))
    }

    /// Mean size of one expert, 0 when there are none.
    pub fn mean_expert_bytes(&self) -> u64 {
        match self.experts.len() as u64 {
            0 => 0,
            n => self.expert_bytes() / n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, s: u64, e: u64) -> Tensor {
        Tensor {
            name: name.into(),
            dtype: "U8".into(),
            shape: vec![e - s],
            start: s,
            end: e,
        }
    }

    #[test]
    fn names() {
        assert_eq!(
            expert_of("model.layers.12.mlp.experts.37.up_proj.weight"),
            Some(ExpertId {
                layer: 12,
                expert: 37
            })
        );
        assert_eq!(expert_of("model.layers.3.self_attn.q_proj.weight"), None);
        assert_eq!(expert_of("model.layers.3.mlp.experts.x.w"), None);
    }

    #[test]
    fn groups_an_experts_tensors_and_counts_the_rest_as_resident() {
        let idx = ExpertIndex::build(&[
            t("model.layers.0.mlp.experts.1.up.weight", 100, 150),
            t("model.layers.0.mlp.experts.1.down.weight", 150, 200),
            t("model.layers.0.mlp.experts.2.up.weight", 200, 230),
            t("model.embed_tokens.weight", 0, 100),
        ]);
        assert_eq!(idx.experts.len(), 2);
        assert_eq!(
            idx.experts[&ExpertId {
                layer: 0,
                expert: 1
            }]
                .bytes,
            100
        );
        assert_eq!(idx.resident_bytes, 100);
        assert_eq!(idx.expert_bytes(), 130);
        assert_eq!(idx.mean_expert_bytes(), 65);
    }

    fn packed(name: &str, shape: &[u64], s: u64, e: u64) -> Tensor {
        Tensor {
            name: name.into(),
            dtype: "U8".into(),
            shape: shape.to_vec(),
            start: s,
            end: e,
        }
    }

    fn id(layer: u32, expert: u32) -> ExpertId {
        ExpertId { layer, expert }
    }

    #[test]
    fn packed_names_are_told_apart_from_per_expert_ones() {
        assert_eq!(
            packed_layer_of("model.layers.4.mlp.experts.gate_up_proj"),
            Some(4)
        );
        assert_eq!(
            packed_layer_of("model.layers.4.mlp.experts.gate_up_proj_blocks"),
            Some(4)
        );
        assert_eq!(
            packed_layer_of("model.layers.4.mlp.experts.7.up_proj.weight"),
            None
        );
        assert_eq!(packed_layer_of("model.layers.4.mlp.gate.weight"), None);
        assert_eq!(
            packed_layer_of("model.layers.4.mlp.shared_experts.up_proj"),
            None
        );
        assert_eq!(
            packed_layer_of("model.layers.x.mlp.experts.gate_up_proj"),
            None
        );
        assert_eq!(packed_layer_of("model.layers.4.mlp.experts"), None);
        assert_eq!(packed_layer_of(""), None);
        assert_eq!(expert_of("model.layers.4.mlp.experts.gate_up_proj"), None);
    }

    #[test]
    fn a_packed_tensor_becomes_one_equal_slice_per_expert() {
        // 4 experts of 25 bytes in one tensor starting at byte 100, plus a down projection
        let idx = ExpertIndex::build(&[
            packed(
                "model.layers.2.mlp.experts.gate_up_proj",
                &[4, 5, 5],
                100,
                200,
            ),
            packed("model.layers.2.mlp.experts.down_proj", &[4, 10], 200, 240),
            packed("model.embed_tokens.weight", &[10], 0, 100),
        ]);
        assert_eq!(idx.experts.len(), 4);
        assert_eq!(idx.resident_bytes, 100);
        for e in 0..4u32 {
            let span = &idx.experts[&id(2, e)];
            let (a, b) = (100 + e as u64 * 25, 200 + e as u64 * 10);
            assert_eq!(span.ranges, vec![(a, a + 25), (b, b + 10)]);
            assert_eq!(span.bytes, 35);
        }
        assert_eq!(idx.expert_bytes(), 140);
        assert_eq!(idx.mean_expert_bytes(), 35);
    }

    #[test]
    fn packed_slices_tile_the_tensor_exactly_whatever_the_sizes() {
        let mut s = 1u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..500 {
            let n = 1 + r() % 40;
            let each = 1 + r() % 1000;
            let start = r() % 1_000_000;
            let t = packed(
                "a.layers.0.mlp.experts.w",
                &[n, each],
                start,
                start + n * each,
            );
            let idx = ExpertIndex::build(&[t]);
            assert_eq!(idx.experts.len() as u64, n);
            let mut next = start;
            for e in 0..n as u32 {
                let span = &idx.experts[&id(0, e)];
                assert_eq!(span.ranges, vec![(next, next + each)]);
                next += each;
            }
            assert_eq!(next, start + n * each);
            assert_eq!(idx.resident_bytes, 0);
        }
    }

    #[test]
    fn a_packed_tensor_that_cannot_be_split_is_resident_not_a_guess() {
        let len = |t: &Tensor| t.len();
        for t in [
            packed("l.layers.0.mlp.experts.w", &[], 0, 100), // no shape
            packed("l.layers.0.mlp.experts.w", &[0, 5], 0, 100), // zero experts
            packed("l.layers.0.mlp.experts.w", &[3, 5], 0, 100), // 100 bytes do not divide by 3
            packed("l.layers.0.mlp.experts.w", &[4, 0], 50, 50), // empty
            packed("l.layers.0.mlp.experts.w", &[2, 3], 60, 10), // inverted range
            packed(
                "l.layers.0.mlp.experts.w",
                &[4_000_000_000, 1],
                0,
                4_000_000_000,
            ), // absurd
            packed(
                "l.layers.0.mlp.experts.w",
                &[MAX_PACKED_EXPERTS + 1, 1],
                0,
                MAX_PACKED_EXPERTS + 1,
            ),
        ] {
            let idx = ExpertIndex::build(std::slice::from_ref(&t));
            assert!(idx.experts.is_empty(), "{:?}", t.shape);
            assert_eq!(idx.resident_bytes, len(&t), "{:?}", t.shape);
        }
        // the largest accepted count still splits
        let n = MAX_PACKED_EXPERTS;
        let t = packed("l.layers.0.mlp.experts.w", &[n, 2], 0, 2 * n);
        assert_eq!(ExpertIndex::build(&[t]).experts.len() as u64, n);
    }

    #[test]
    fn offsets_near_u64_max_never_overflow_while_splitting() {
        let t = packed("l.layers.0.mlp.experts.w", &[2, 1], u64::MAX - 1, u64::MAX); // 1 byte, 2 experts: does not divide
        assert!(ExpertIndex::build(&[t]).experts.is_empty());
        let t = packed(
            "l.layers.0.mlp.experts.w",
            &[2, 1],
            u64::MAX - 4,
            u64::MAX - 2,
        );
        let idx = ExpertIndex::build(&[t]);
        assert_eq!(idx.experts.len(), 2);
        assert_eq!(
            idx.experts[&id(0, 1)].ranges,
            vec![(u64::MAX - 3, u64::MAX - 2)]
        );
    }

    #[test]
    fn the_entry_limit_bounds_the_index_whatever_the_header_says() {
        let many: Vec<Tensor> = (0..50u32)
            .map(|l| {
                packed(
                    &format!("m.layers.{l}.mlp.experts.w"),
                    &[8, 4],
                    l as u64 * 32,
                    l as u64 * 32 + 32,
                )
            })
            .collect();
        let idx = ExpertIndex::build_with_limit(&many, 20);
        assert!(idx.experts.len() <= 20, "{}", idx.experts.len());
        // what did not fit is counted as resident, so no bytes are lost
        assert_eq!(idx.expert_bytes() + idx.resident_bytes, 50 * 32);
        let per_expert: Vec<Tensor> = (0..50u32)
            .map(|e| {
                packed(
                    &format!("m.layers.0.mlp.experts.{e}.w"),
                    &[4],
                    e as u64 * 4,
                    e as u64 * 4 + 4,
                )
            })
            .collect();
        let idx = ExpertIndex::build_with_limit(&per_expert, 10);
        assert_eq!(idx.experts.len(), 10);
        assert_eq!(idx.expert_bytes() + idx.resident_bytes, 200);
    }

    #[test]
    fn per_expert_and_packed_layouts_can_share_a_checkpoint() {
        let idx = ExpertIndex::build(&[
            packed("m.layers.0.mlp.experts.0.up.weight", &[10], 0, 10),
            packed("m.layers.0.mlp.experts.1.up.weight", &[10], 10, 20),
            packed("m.layers.1.mlp.experts.up", &[2, 10], 20, 40),
        ]);
        assert_eq!(idx.experts.len(), 4);
        assert_eq!(idx.experts[&id(1, 1)].ranges, vec![(30, 40)]);
    }
}
