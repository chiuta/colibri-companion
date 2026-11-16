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

/// `(layer, expert)` when `name` is an expert tensor.
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
        let mut idx = Self::default();
        for t in tensors {
            match expert_of(&t.name) {
                Some(id) => {
                    let s = idx.experts.entry(id).or_default();
                    s.ranges.push((t.start, t.end));
                    s.bytes += t.len();
                }
                None => idx.resident_bytes += t.len(),
            }
        }
        for s in idx.experts.values_mut() {
            s.ranges.sort_unstable();
        }
        idx
    }

    pub fn expert_bytes(&self) -> u64 {
        self.experts.values().map(|s| s.bytes).sum()
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
}
