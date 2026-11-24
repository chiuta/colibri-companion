//! Architecture registry: one [`Spec`] for every model family, from Hugging Face `config.json`
//! or from GGUF metadata, plus an honest statement of how far each family is supported.
use crate::gguf::Gguf;
use serde_json::Value;

/// How far this engine goes for a family. Never claims more than is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    /// Plain decoder: grouped-query attention, RoPE, gated MLP — the shared forward pass covers it.
    Standard,
    /// Standard plus routed experts (streamed from disk by the planner).
    Moe,
    /// Needs an attention or state-space variant that is not implemented yet.
    Planned(&'static str),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    pub family: String,
    pub support: Support,
    pub layers: u64,
    pub hidden: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub ffn: u64,
    pub vocab: u64,
    pub context: u64,
    pub rope_theta: f64,
    pub norm_eps: f64,
    pub experts: u64,
    pub experts_used: u64,
    pub shared_experts: u64,
    pub tied_embeddings: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum SpecError { Missing(&'static str), NotAnObject }
impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecError::Missing(k) => write!(f, "the model description has no `{k}`"),
            SpecError::NotAnObject => write!(f, "the model description is not a JSON object"),
        }
    }
}
impl std::error::Error for SpecError {}

/// Classify a `model_type` / GGUF architecture name.
pub fn classify(name: &str) -> Support {
    let n = name.to_ascii_lowercase();
    match n.as_str() {
        "llama" | "mistral" | "qwen" | "qwen2" | "qwen3" | "gemma" | "gemma2" | "gemma3" | "gemma3_text"
        | "phi" | "phi3" | "olmo" | "olmo2" | "granite" | "cohere" | "cohere2" | "command-r" | "internlm2"
        | "yi" | "smollm3" | "stablelm" | "starcoder2" | "exaone" | "glm" | "glm4" | "baichuan" | "minicpm"
        | "falcon" | "gpt_neox" | "gptneox" | "gpt2" | "gptj" | "opt" | "bloom" | "mpt" => Support::Standard,
        "mixtral" | "qwen2_moe" | "qwen3_moe" | "qwen3moe" | "olmoe" | "granitemoe" | "phimoe" | "dbrx"
        | "arctic" | "jamba_moe" | "gpt_oss" | "gpt-oss" | "llama4" | "glm4_moe" | "hunyuan_moe" | "ernie4_5_moe" => Support::Moe,
        "deepseek_v2" | "deepseek_v3" | "deepseek2" | "kimi_k2" | "minimax" => Support::Planned("multi-head latent attention"),
        "mamba" | "mamba2" | "rwkv" | "rwkv6" | "rwkv7" | "falcon_mamba" | "jamba" | "zamba" | "bamba" => Support::Planned("state-space / recurrent layers"),
        "t5" | "mt5" | "bart" | "whisper" | "bert" | "roberta" => Support::Planned("encoder / encoder-decoder models"),
        _ => Support::Planned("unknown architecture"),
    }
}

fn first_u64(c: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| c.get(*k).and_then(Value::as_u64))
}
fn first_f64(c: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| c.get(*k).and_then(Value::as_f64))
}

/// Build a [`Spec`] from a Hugging Face `config.json`. Multimodal wrappers keep the language
/// model under `text_config`; that is followed automatically.
pub fn from_hf_config(json: &str) -> Result<Spec, SpecError> {
    let root: Value = serde_json::from_str(json).map_err(|_| SpecError::NotAnObject)?;
    if !root.is_object() { return Err(SpecError::NotAnObject); }
    let c = root.get("text_config").filter(|t| t.is_object()).unwrap_or(&root);
    let family = c.get("model_type").or_else(|| root.get("model_type")).and_then(Value::as_str).unwrap_or("unknown").to_string();
    let hidden = first_u64(c, &["hidden_size", "n_embd", "d_model"]).ok_or(SpecError::Missing("hidden_size"))?;
    let layers = first_u64(c, &["num_hidden_layers", "n_layer", "num_layers"]).ok_or(SpecError::Missing("num_hidden_layers"))?;
    let heads = first_u64(c, &["num_attention_heads", "n_head"]).ok_or(SpecError::Missing("num_attention_heads"))?;
    if heads == 0 { return Err(SpecError::Missing("num_attention_heads")); }
    let kv_heads = first_u64(c, &["num_key_value_heads", "n_head_kv", "multi_query_group_num"]).unwrap_or(heads);
    let experts = first_u64(c, &["num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"]).unwrap_or(0);
    let support = match classify(&family) {
        Support::Standard if experts > 1 => Support::Moe,
        s => s,
    };
    Ok(Spec {
        family,
        support,
        layers, hidden, heads,
        kv_heads: kv_heads.max(1),
        head_dim: first_u64(c, &["head_dim"]).unwrap_or(hidden / heads),
        ffn: first_u64(c, &["intermediate_size", "ffn_dim", "n_inner"]).unwrap_or(hidden * 4),
        vocab: first_u64(c, &["vocab_size"]).unwrap_or(0),
        context: first_u64(c, &["max_position_embeddings", "n_positions", "seq_length", "max_seq_len"]).unwrap_or(4096),
        rope_theta: first_f64(c, &["rope_theta", "rotary_emb_base"]).unwrap_or(10_000.0),
        norm_eps: first_f64(c, &["rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps", "norm_eps"]).unwrap_or(1e-5),
        experts,
        experts_used: first_u64(c, &["num_experts_per_tok", "num_experts_per_token", "moe_top_k", "top_k"]).unwrap_or(if experts > 0 { 2 } else { 0 }),
        shared_experts: first_u64(c, &["n_shared_experts", "num_shared_experts"]).unwrap_or(0),
        tied_embeddings: c.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false),
    })
}

/// Build a [`Spec`] from GGUF metadata.
pub fn from_gguf(g: &Gguf) -> Result<Spec, SpecError> {
    let family = g.architecture().ok_or(SpecError::Missing("general.architecture"))?.to_string();
    let layers = g.get_u64("block_count").ok_or(SpecError::Missing("block_count"))?;
    let hidden = g.get_u64("embedding_length").ok_or(SpecError::Missing("embedding_length"))?;
    let heads = g.get_u64("attention.head_count").filter(|h| *h > 0).ok_or(SpecError::Missing("attention.head_count"))?;
    let experts = g.get_u64("expert_count").unwrap_or(0);
    let support = match classify(&family) { Support::Standard if experts > 1 => Support::Moe, s => s };
    let vocab = match g.meta.get("tokenizer.ggml.tokens") {
        Some(crate::gguf::Value::Array(a)) => a.len() as u64,
        _ => 0,
    };
    Ok(Spec {
        family, support, layers, hidden, heads,
        kv_heads: g.get_u64("attention.head_count_kv").unwrap_or(heads).max(1),
        head_dim: g.get_u64("attention.key_length").unwrap_or(hidden / heads),
        ffn: g.get_u64("feed_forward_length").unwrap_or(hidden * 4),
        vocab,
        context: g.get_u64("context_length").unwrap_or(4096),
        rope_theta: g.get_f64("rope.freq_base").unwrap_or(10_000.0),
        norm_eps: g.get_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
        experts,
        experts_used: g.get_u64("expert_used_count").unwrap_or(if experts > 0 { 2 } else { 0 }),
        shared_experts: g.get_u64("expert_shared_count").unwrap_or(0),
        tied_embeddings: !g.tensors.iter().any(|t| t.name == "output.weight"),
    })
}

impl Spec {
    /// Bytes of KV cache per token (f16 keys and values).
    pub fn kv_bytes_per_token(&self) -> u64 { 2 * self.layers * self.kv_heads * self.head_dim * 2 }
    pub fn kv_bytes(&self, tokens: u64) -> u64 { self.kv_bytes_per_token().saturating_mul(tokens) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA: &str = r#"{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,
        "num_key_value_heads":8,"intermediate_size":14336,"vocab_size":128256,"rope_theta":500000.0,"rms_norm_eps":1e-5}"#;

    #[test]
    fn llama_style_config() {
        let s = from_hf_config(LLAMA).unwrap();
        assert_eq!((s.layers, s.hidden, s.heads, s.kv_heads, s.head_dim), (32, 4096, 32, 8, 128));
        assert_eq!(s.support, Support::Standard);
        assert_eq!(s.kv_bytes_per_token(), 2 * 32 * 8 * 128 * 2);
    }

    #[test]
    fn moe_is_detected_even_under_a_dense_family_name() {
        let s = from_hf_config(r#"{"model_type":"mixtral","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,
            "num_local_experts":8,"num_experts_per_tok":2}"#).unwrap();
        assert_eq!((s.support, s.experts, s.experts_used), (Support::Moe, 8, 2));
        let s = from_hf_config(r#"{"model_type":"llama","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_local_experts":4}"#).unwrap();
        assert_eq!(s.support, Support::Moe);
    }

    #[test]
    fn multimodal_wrapper_uses_the_text_config() {
        let s = from_hf_config(r#"{"model_type":"gemma3","text_config":{"model_type":"gemma3_text","hidden_size":2560,
            "num_hidden_layers":34,"num_attention_heads":8,"num_key_value_heads":4,"head_dim":256}}"#).unwrap();
        assert_eq!((s.family.as_str(), s.hidden, s.head_dim), ("gemma3_text", 2560, 256));
    }

    #[test]
    fn unsupported_families_say_why() {
        assert!(matches!(classify("deepseek_v3"), Support::Planned(w) if w.contains("latent")));
        assert!(matches!(classify("mamba2"), Support::Planned(_)));
        assert!(matches!(classify("never-heard-of-it"), Support::Planned("unknown architecture")));
    }

    #[test]
    fn broken_configs_are_errors() {
        assert_eq!(from_hf_config("[1]"), Err(SpecError::NotAnObject));
        assert_eq!(from_hf_config("{}"), Err(SpecError::Missing("hidden_size")));
        assert!(from_hf_config(r#"{"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":0}"#).is_err());
    }

    #[test]
    fn gguf_metadata_gives_the_same_shape() {
        let g = Gguf::parse(&crate::gguf::tests::sample()[..]).unwrap();
        // the sample has no embedding_length: must be a clean error
        assert_eq!(from_gguf(&g), Err(SpecError::Missing("embedding_length")));
    }
}
