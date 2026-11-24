//! Memory planner: decides what stays in RAM and what is streamed, for any model and budget.
//! The rule is the one that makes huge mixture-of-experts models usable on small machines:
//! attention, norms and embeddings are always resident; routed experts live on disk and a
//! cache holds the hot ones.
use crate::gguf::TensorInfo;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role { Embedding, Attention, Norm, Dense, SharedExpert, RoutedExpert, Other }

pub fn role_of(name: &str) -> Role {
    let n = name.to_ascii_lowercase();
    if n.contains("shared_expert") || n.contains("_shexp") { Role::SharedExpert }
    else if n.contains(".experts.") || n.contains("_exps") || n.contains("block_sparse_moe.experts") { Role::RoutedExpert }
    else if n.contains("embed") || n.contains("token_embd") || n.starts_with("output.") || n.contains("lm_head") { Role::Embedding }
    else if n.contains("norm") || n.contains("ln_") { Role::Norm }
    else if n.contains("attn") || n.contains("attention") || n.contains("self_attn") { Role::Attention }
    else if n.contains("mlp") || n.contains("ffn") || n.contains("feed_forward") { Role::Dense }
    else { Role::Other }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub resident: u64,
    pub streamed: u64,
    pub kv_cache: u64,
    /// Bytes left for the expert cache after everything resident is counted.
    pub expert_cache: u64,
    pub fits: Verdict,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Everything fits in RAM.
    AllInMemory,
    /// Core fits; experts stream from disk through a cache.
    Streaming,
    /// Even the resident core plus KV cache is larger than the budget.
    TooSmall { missing: u64 },
}

pub fn plan(tensors: &[TensorInfo], ram_budget: u64, kv_cache: u64) -> Plan {
    let (mut resident, mut streamed) = (0u64, 0u64);
    for t in tensors {
        let b = t.bytes().unwrap_or(0);
        if role_of(&t.name) == Role::RoutedExpert { streamed = streamed.saturating_add(b) } else { resident = resident.saturating_add(b) }
    }
    let core = resident.saturating_add(kv_cache);
    let (fits, expert_cache) = if core.saturating_add(streamed) <= ram_budget {
        (Verdict::AllInMemory, streamed)
    } else if core <= ram_budget {
        (Verdict::Streaming, (ram_budget - core).min(streamed))
    } else {
        (Verdict::TooSmall { missing: core - ram_budget }, 0)
    };
    Plan { resident, streamed, kv_cache, expert_cache, fits }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::DType;

    fn t(name: &str, n: u64) -> TensorInfo { TensorInfo { name: name.into(), dims: vec![n], dtype: DType::F16, offset: 0 } }

    #[test]
    fn names_from_both_ecosystems_are_classified() {
        assert_eq!(role_of("model.layers.3.mlp.experts.17.down_proj.weight"), Role::RoutedExpert);
        assert_eq!(role_of("blk.3.ffn_down_exps.weight"), Role::RoutedExpert);
        assert_eq!(role_of("model.layers.3.block_sparse_moe.experts.0.w1.weight"), Role::RoutedExpert);
        assert_eq!(role_of("blk.3.ffn_gate_shexp.weight"), Role::SharedExpert);
        assert_eq!(role_of("model.embed_tokens.weight"), Role::Embedding);
        assert_eq!(role_of("blk.0.attn_q.weight"), Role::Attention);
        assert_eq!(role_of("model.layers.0.input_layernorm.weight"), Role::Norm);
        assert_eq!(role_of("model.layers.0.mlp.up_proj.weight"), Role::Dense);
    }

    #[test]
    fn three_verdicts() {
        let ts = [t("model.embed_tokens.weight", 1000), t("model.layers.0.mlp.experts.0.w.weight", 10_000)];
        let (core, exp) = (2000, 20_000);
        assert_eq!(plan(&ts, 100_000, 0).fits, Verdict::AllInMemory);
        let p = plan(&ts, 5000, 1000);
        assert_eq!((p.fits, p.resident, p.streamed, p.expert_cache), (Verdict::Streaming, core, exp, 2000));
        assert_eq!(plan(&ts, 1500, 0).fits, Verdict::TooSmall { missing: 500 });
    }

    #[test]
    fn overflowing_sizes_do_not_panic() {
        let big = TensorInfo { name: "x".into(), dims: vec![u64::MAX, 2], dtype: DType::F32, offset: 0 };
        assert_eq!(plan(&[big], 10, u64::MAX).resident, 0);
    }
}
