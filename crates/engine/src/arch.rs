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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind { Rms, Layer }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act { Silu, GeluTanh, Gelu }

/// Position-embedding scaling used to stretch the context window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeScaling {
    None,
    Linear { factor: f64 },
    Llama3 { factor: f64, low_freq_factor: f64, high_freq_factor: f64, original_max: f64 },
    Yarn { factor: f64, original_max: f64, beta_fast: f64, beta_slow: f64, mscale: f64 },
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
    // ---- variants (defaults describe a plain Llama) ----
    pub norm: NormKind,
    /// Gemma-style RMSNorm: the stored weight is applied as `1 + w`.
    pub norm_offset: bool,
    pub act: Act,
    /// Gated MLP (`act(gate) * up`) versus plain `act(up)`.
    pub gated_mlp: bool,
    /// Multiplier applied to the token embeddings (Gemma: `sqrt(hidden)`).
    pub embed_scale: f64,
    /// Per-head RMSNorm on queries and keys (Qwen3, Gemma 3).
    pub qk_norm: bool,
    /// Q/K/V projections must carry biases (Qwen2). Biases that exist are always used.
    pub qkv_bias: bool,
    /// Extra norms after attention and after the MLP (Gemma 2/3).
    pub post_norms: bool,
    /// GPT-NeoX / Falcon / Phi style: attention and MLP both read the block input.
    pub parallel_residual: bool,
    /// `cap * tanh(x / cap)` on attention scores; 0 = off.
    pub attn_softcap: f64,
    /// Same on the final logits; 0 = off.
    pub final_softcap: f64,
    pub sliding_window: u64,
    /// `sliding_layers[i]` = layer i only attends to the last `sliding_window` tokens. Empty = none.
    pub sliding_layers: Vec<bool>,
    /// RoPE base for sliding layers when it differs from `rope_theta` (Gemma 3); 0 = same.
    pub rope_theta_local: f64,
    /// Number of dimensions per head that are rotated (partial rotary); even, `<= head_dim`.
    pub rotary_dim: u64,
    pub rope_scaling: RopeScaling,
    /// Attention score multiplier; 0 = `1/sqrt(head_dim)`.
    pub attn_scale: f64,
}

impl Spec {
    /// A plain dense Llama-style decoder; variants are switched on by assigning fields afterwards.
    #[allow(clippy::too_many_arguments)]
    pub fn dense(family: &str, layers: u64, hidden: u64, heads: u64, kv_heads: u64, head_dim: u64, ffn: u64, vocab: u64, context: u64) -> Spec {
        Spec {
            family: family.into(), support: classify(family), layers, hidden, heads, kv_heads, head_dim, ffn, vocab, context,
            rope_theta: 10_000.0, norm_eps: 1e-5, experts: 0, experts_used: 0, shared_experts: 0, tied_embeddings: false,
            norm: NormKind::Rms, norm_offset: false, act: Act::Silu, gated_mlp: true, embed_scale: 1.0, qk_norm: false,
            qkv_bias: false, post_norms: false, parallel_residual: false, attn_softcap: 0.0, final_softcap: 0.0,
            sliding_window: 0, sliding_layers: vec![], rope_theta_local: 0.0, rotary_dim: head_dim,
            rope_scaling: RopeScaling::None, attn_scale: 0.0,
        }
    }
    /// Does layer `i` use the sliding window?
    pub fn is_sliding(&self, i: usize) -> bool { self.sliding_window > 0 && self.sliding_layers.get(i).copied().unwrap_or(false) }
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
        | "yi" | "smollm3" | "stablelm" | "starcoder2" | "exaone" | "glm" | "glm4" | "baichuan" | "minicpm" => Support::Standard,
        "falcon" | "gpt_neox" | "gptneox" => Support::Planned("fused interleaved QKV tensor layout"),
        "gpt2" | "gptj" | "opt" | "bloom" | "mpt" => Support::Planned("learned or ALiBi position scheme"),
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

/// Per-family behaviour that the config file does not spell out.
struct Traits { layer_norm: bool, gated: bool, parallel: bool, gemma: bool, qk_norm: bool, post_norms: bool, qkv_bias: bool, act: Act, sliding_pattern: u64 }
fn traits(family: &str) -> Traits {
    let mut t = Traits { layer_norm: false, gated: true, parallel: false, gemma: false, qk_norm: false, post_norms: false, qkv_bias: false, act: Act::Silu, sliding_pattern: 0 };
    match family {
        "gemma" => { t.gemma = true; t.act = Act::GeluTanh; }
        "gemma2" => { t.gemma = true; t.act = Act::GeluTanh; t.post_norms = true; t.sliding_pattern = 2; }
        "gemma3" | "gemma3_text" => { t.gemma = true; t.act = Act::GeluTanh; t.post_norms = true; t.qk_norm = true; t.sliding_pattern = 6; }
        "qwen2" | "qwen2_moe" => t.qkv_bias = true,
        "qwen3" | "qwen3_moe" | "qwen3moe" => t.qk_norm = true,
        "starcoder2" => { t.layer_norm = true; t.gated = false; t.act = Act::GeluTanh; }
        "gpt_neox" | "gptneox" => { t.layer_norm = true; t.gated = false; t.parallel = true; t.act = Act::Gelu; }
        "falcon" => { t.layer_norm = true; t.gated = false; t.act = Act::Gelu; }
        "phi" => { t.layer_norm = true; t.gated = false; t.parallel = true; t.act = Act::GeluTanh; }
        "stablelm" => t.layer_norm = true,
        _ => {}
    }
    t
}

fn parse_act(s: &str) -> Option<Act> {
    match s {
        "silu" | "swish" => Some(Act::Silu),
        "gelu" => Some(Act::Gelu),
        "gelu_new" | "gelu_pytorch_tanh" | "gelu_fast" | "gelu_tanh" => Some(Act::GeluTanh),
        _ => None,
    }
}

/// Reads `rope_scaling` / `rope_parameters`. Returns the scaling and, for kinds we cannot do
/// faithfully, a reason (the caller then downgrades the support level instead of guessing).
fn parse_rope_scaling(c: &Value, root: &Value, orig_default: f64) -> (RopeScaling, Option<&'static str>) {
    let Some(r) = ["rope_scaling", "rope_parameters"].iter().find_map(|k| c.get(*k).or_else(|| root.get(*k)).filter(|v| v.is_object())) else {
        return (RopeScaling::None, None);
    };
    let kind = r.get("rope_type").or_else(|| r.get("type")).and_then(Value::as_str).unwrap_or("default");
    let f = |k: &str| r.get(k).and_then(Value::as_f64);
    let factor = f("factor").unwrap_or(1.0);
    let orig = f("original_max_position_embeddings").unwrap_or(orig_default);
    match kind {
        "default" => (RopeScaling::None, None),
        "linear" => (RopeScaling::Linear { factor }, None),
        "llama3" => (RopeScaling::Llama3 { factor, low_freq_factor: f("low_freq_factor").unwrap_or(1.0),
            high_freq_factor: f("high_freq_factor").unwrap_or(4.0), original_max: f("original_max_position_embeddings").unwrap_or(8192.0) }, None),
        "yarn" => (RopeScaling::Yarn { factor, original_max: orig, beta_fast: f("beta_fast").unwrap_or(32.0), beta_slow: f("beta_slow").unwrap_or(1.0),
            mscale: f("attention_factor").unwrap_or(if factor > 1.0 { 0.1 * factor.ln() + 1.0 } else { 1.0 }) }, None),
        // dynamic NTK only changes behaviour beyond the original window, so short contexts are exact
        "dynamic" => (RopeScaling::None, None),
        "longrope" | "su" => (RopeScaling::None, Some("LongRoPE scaling")),
        _ => (RopeScaling::None, Some("unknown RoPE scaling")),
    }
}

fn layers_from_types(c: &Value, n: u64) -> Option<Vec<bool>> {
    let a = c.get("layer_types")?.as_array()?;
    (a.len() as u64 == n).then(|| a.iter().map(|v| v.as_str() == Some("sliding_attention")).collect())
}
fn layers_from_pattern(n: u64, window: u64, pattern: u64) -> Vec<bool> {
    if window == 0 { return vec![]; }
    (0..n).map(|i| pattern == 0 || (i + 1) % pattern != 0).collect()
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
    let head_dim = first_u64(c, &["head_dim"]).filter(|d| *d > 0).unwrap_or(hidden / heads);
    let t = traits(&family);
    let mut s = Spec::dense(&family, layers, hidden, heads, kv_heads.max(1), head_dim,
        first_u64(c, &["intermediate_size", "ffn_dim", "n_inner"]).unwrap_or(hidden * 4),
        first_u64(c, &["vocab_size"]).unwrap_or(0),
        first_u64(c, &["max_position_embeddings", "n_positions", "seq_length", "max_seq_len"]).unwrap_or(4096));
    s.support = match classify(&family) { Support::Standard if experts > 1 => Support::Moe, x => x };
    let rope_obj = c.get("rope_parameters").filter(|v| v.is_object());
    s.rope_theta = first_f64(c, &["rope_theta", "rotary_emb_base"])
        .or_else(|| rope_obj.and_then(|r| r.get("rope_theta")).and_then(Value::as_f64))
        .unwrap_or(if t.gemma && family.starts_with("gemma3") { 1_000_000.0 } else { 10_000.0 });
    s.norm_eps = first_f64(c, &["rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps", "norm_eps", "norm_epsilon"]).unwrap_or(1e-5);
    s.experts = experts;
    s.experts_used = first_u64(c, &["num_experts_per_tok", "num_experts_per_token", "moe_top_k", "top_k"]).unwrap_or(if experts > 0 { 2 } else { 0 });
    s.shared_experts = first_u64(c, &["n_shared_experts", "num_shared_experts"]).unwrap_or(0);
    s.tied_embeddings = c.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(t.gemma);

    s.norm = if t.layer_norm { NormKind::Layer } else { NormKind::Rms };
    s.norm_offset = t.gemma;
    s.gated_mlp = t.gated;
    s.act = ["hidden_activation", "hidden_act", "activation_function"].iter()
        .find_map(|k| c.get(*k).and_then(Value::as_str)).and_then(parse_act).unwrap_or(t.act);
    s.embed_scale = if t.gemma { (hidden as f64).sqrt() } else { 1.0 };
    s.qk_norm = t.qk_norm;
    s.qkv_bias = t.qkv_bias || c.get("attention_bias").and_then(Value::as_bool).unwrap_or(false) || c.get("use_bias").and_then(Value::as_bool).unwrap_or(false);
    s.post_norms = t.post_norms;
    s.parallel_residual = c.get("use_parallel_residual").or_else(|| c.get("parallel_attn")).and_then(Value::as_bool).unwrap_or(t.parallel);
    s.attn_softcap = c.get("attn_logit_softcapping").and_then(Value::as_f64).unwrap_or(0.0);
    s.final_softcap = c.get("final_logit_softcapping").and_then(Value::as_f64).unwrap_or(0.0);
    let qwen_off = matches!(family.as_str(), "qwen2" | "qwen2_moe" | "qwen3" | "qwen3_moe") && c.get("use_sliding_window").and_then(Value::as_bool) != Some(true);
    s.sliding_window = if qwen_off { 0 } else { c.get("sliding_window").and_then(Value::as_u64).unwrap_or(0) };
    s.sliding_layers = if s.sliding_window == 0 { vec![] } else {
        layers_from_types(c, layers).unwrap_or_else(|| layers_from_pattern(layers, s.sliding_window, c.get("sliding_window_pattern").and_then(Value::as_u64).unwrap_or(t.sliding_pattern)))
    };
    s.rope_theta_local = c.get("rope_local_base_freq").and_then(Value::as_f64).unwrap_or(0.0);
    let pf = first_f64(c, &["partial_rotary_factor", "rotary_pct"]).or_else(|| rope_obj.and_then(|r| r.get("partial_rotary_factor")).and_then(Value::as_f64)).unwrap_or(1.0);
    s.rotary_dim = (((head_dim as f64 * pf) as u64) & !1).clamp(2, head_dim.max(2));
    let (rs, unsupported) = parse_rope_scaling(c, &root, s.context as f64);
    s.rope_scaling = rs;
    if let Some(why) = unsupported { s.support = Support::Planned(why); }
    s.attn_scale = c.get("query_pre_attn_scalar").and_then(Value::as_f64).filter(|v| *v > 0.0).map_or(0.0, |v| 1.0 / v.sqrt());
    Ok(s)
}

/// Build a [`Spec`] from GGUF metadata.
pub fn from_gguf(g: &Gguf) -> Result<Spec, SpecError> {
    let family = g.architecture().ok_or(SpecError::Missing("general.architecture"))?.to_string();
    let layers = g.get_u64("block_count").ok_or(SpecError::Missing("block_count"))?;
    let hidden = g.get_u64("embedding_length").ok_or(SpecError::Missing("embedding_length"))?;
    let heads = g.get_u64("attention.head_count").filter(|h| *h > 0).ok_or(SpecError::Missing("attention.head_count"))?;
    let experts = g.get_u64("expert_count").unwrap_or(0);
    let vocab = match g.meta.get("tokenizer.ggml.tokens") {
        Some(crate::gguf::Value::Array(a)) => a.len() as u64,
        _ => 0,
    };
    let head_dim = g.get_u64("attention.key_length").filter(|d| *d > 0).unwrap_or(hidden / heads);
    let t = traits(&family);
    let mut s = Spec::dense(&family, layers, hidden, heads, g.get_u64("attention.head_count_kv").unwrap_or(heads).max(1), head_dim,
        g.get_u64("feed_forward_length").unwrap_or(hidden * 4), vocab, g.get_u64("context_length").unwrap_or(4096));
    s.support = match classify(&family) { Support::Standard if experts > 1 => Support::Moe, x => x };
    s.rope_theta = g.get_f64("rope.freq_base").unwrap_or(10_000.0);
    s.norm_eps = g.get_f64("attention.layer_norm_rms_epsilon").or_else(|| g.get_f64("attention.layer_norm_epsilon")).unwrap_or(1e-5);
    s.experts = experts;
    s.experts_used = g.get_u64("expert_used_count").unwrap_or(if experts > 0 { 2 } else { 0 });
    s.shared_experts = g.get_u64("expert_shared_count").unwrap_or(0);
    s.tied_embeddings = !g.tensors.iter().any(|x| x.name == "output.weight");
    s.norm = if t.layer_norm { NormKind::Layer } else { NormKind::Rms };
    // llama.cpp's converter already adds 1 to Gemma norm weights, so no offset at run time.
    s.norm_offset = false;
    s.gated_mlp = t.gated;
    s.act = t.act;
    s.embed_scale = if t.gemma { (hidden as f64).sqrt() } else { 1.0 };
    s.qk_norm = t.qk_norm;
    s.qkv_bias = t.qkv_bias;
    s.post_norms = t.post_norms;
    s.parallel_residual = t.parallel;
    s.attn_softcap = g.get_f64("attn_logit_softcapping").unwrap_or(0.0);
    s.final_softcap = g.get_f64("final_logit_softcapping").unwrap_or(0.0);
    s.sliding_window = g.get_u64("attention.sliding_window").unwrap_or(0);
    s.sliding_layers = layers_from_pattern(layers, s.sliding_window, g.get_u64("attention.sliding_window_pattern").unwrap_or(t.sliding_pattern));
    s.rope_theta_local = g.get_f64("rope.freq_base_swa").unwrap_or(if t.gemma && family.starts_with("gemma3") { 10_000.0 } else { 0.0 });
    s.rotary_dim = g.get_u64("rope.dimension_count").map_or(head_dim, |d| (d & !1).clamp(2, head_dim.max(2)));
    let factor = g.get_f64("rope.scaling.factor").unwrap_or(1.0);
    s.rope_scaling = match g.meta.get(&format!("{family}.rope.scaling.type")).and_then(crate::gguf::Value::as_str) {
        Some("linear") => RopeScaling::Linear { factor },
        Some("yarn") => RopeScaling::Yarn { factor, original_max: g.get_f64("rope.scaling.original_context_length").unwrap_or(s.context as f64),
            beta_fast: 32.0, beta_slow: 1.0, mscale: if factor > 1.0 { 0.1 * factor.ln() + 1.0 } else { 1.0 } },
        _ => RopeScaling::None,
    };
    Ok(s)
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

    #[test]
    fn gemma2_config_switches_on_every_gemma_behaviour() {
        let s = from_hf_config(r#"{"model_type":"gemma2","hidden_size":2304,"num_hidden_layers":26,"num_attention_heads":8,"num_key_value_heads":4,
            "head_dim":256,"intermediate_size":9216,"vocab_size":256000,"sliding_window":4096,"attn_logit_softcapping":50.0,
            "final_logit_softcapping":30.0,"query_pre_attn_scalar":256,"hidden_activation":"gelu_pytorch_tanh","rms_norm_eps":1e-6}"#).unwrap();
        assert_eq!((s.norm, s.norm_offset, s.act, s.post_norms, s.tied_embeddings), (NormKind::Rms, true, Act::GeluTanh, true, true));
        assert!((s.embed_scale - 48.0).abs() < 1e-9);
        assert_eq!((s.attn_softcap, s.final_softcap, s.sliding_window), (50.0, 30.0, 4096));
        assert!((s.attn_scale - 1.0 / 16.0).abs() < 1e-12);
        assert_eq!(&s.sliding_layers[..4], &[true, false, true, false]);
        assert!(s.is_sliding(0) && !s.is_sliding(1));
        assert_eq!(s.support, Support::Standard);
    }

    #[test]
    fn gemma3_uses_the_pattern_local_theta_and_scaling() {
        let s = from_hf_config(r#"{"model_type":"gemma3","text_config":{"model_type":"gemma3_text","hidden_size":2560,"num_hidden_layers":12,
            "num_attention_heads":8,"num_key_value_heads":4,"head_dim":256,"sliding_window":1024,"rope_theta":1000000.0,
            "rope_local_base_freq":10000.0,"rope_scaling":{"rope_type":"linear","factor":8.0}}}"#).unwrap();
        assert!(s.qk_norm && s.post_norms);
        assert_eq!(s.sliding_layers.iter().map(|b| *b as u8).collect::<Vec<_>>(), [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]);
        assert_eq!((s.rope_theta, s.rope_theta_local), (1_000_000.0, 10_000.0));
        assert_eq!(s.rope_scaling, RopeScaling::Linear { factor: 8.0 });
        // explicit layer_types win over the pattern
        let s = from_hf_config(r#"{"model_type":"gemma3_text","hidden_size":64,"num_hidden_layers":3,"num_attention_heads":4,"sliding_window":8,
            "layer_types":["full_attention","sliding_attention","sliding_attention"]}"#).unwrap();
        assert_eq!(s.sliding_layers, [false, true, true]);
    }

    #[test]
    fn qwen_configs() {
        let q2 = from_hf_config(r#"{"model_type":"qwen2","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"sliding_window":4096,"use_sliding_window":false}"#).unwrap();
        assert!(q2.qkv_bias && !q2.qk_norm);
        assert_eq!((q2.sliding_window, q2.sliding_layers.len()), (0, 0));
        let q3 = from_hf_config(r#"{"model_type":"qwen3","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"head_dim":32}"#).unwrap();
        assert!(q3.qk_norm && !q3.qkv_bias);
        assert_eq!(q3.head_dim, 32);
        let m = from_hf_config(r#"{"model_type":"mistral","hidden_size":64,"num_hidden_layers":3,"num_attention_heads":4,"sliding_window":128}"#).unwrap();
        assert_eq!(m.sliding_layers, [true, true, true]);
    }

    #[test]
    fn layer_norm_families_and_partial_rotary() {
        let n = from_hf_config(r#"{"model_type":"gpt_neox","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"rotary_pct":0.25,"hidden_act":"gelu"}"#).unwrap();
        assert_eq!((n.norm, n.gated_mlp, n.parallel_residual, n.act, n.rotary_dim), (NormKind::Layer, false, true, Act::Gelu, 4));
        assert!(matches!(n.support, Support::Planned(_)));
        let sc = from_hf_config(r#"{"model_type":"starcoder2","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"use_bias":true,"norm_epsilon":1e-5,"hidden_act":"gelu_pytorch_tanh"}"#).unwrap();
        assert_eq!((sc.norm, sc.gated_mlp, sc.parallel_residual, sc.qkv_bias, sc.act, sc.support), (NormKind::Layer, false, false, true, Act::GeluTanh, Support::Standard));
        let p = from_hf_config(r#"{"model_type":"phi3","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"partial_rotary_factor":0.5}"#).unwrap();
        assert_eq!((p.rotary_dim, p.norm, p.gated_mlp), (8, NormKind::Rms, true));
    }

    #[test]
    fn rope_scaling_kinds_from_config() {
        let cfg = |r: &str| format!(r#"{{"model_type":"llama","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"max_position_embeddings":131072,"rope_scaling":{r}}}"#);
        let s = from_hf_config(&cfg(r#"{"rope_type":"llama3","factor":8.0,"low_freq_factor":1.0,"high_freq_factor":4.0,"original_max_position_embeddings":8192}"#)).unwrap();
        assert_eq!(s.rope_scaling, RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max: 8192.0 });
        let s = from_hf_config(&cfg(r#"{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}"#)).unwrap();
        match s.rope_scaling { RopeScaling::Yarn { factor, original_max, beta_fast, mscale, .. } => {
            assert_eq!((factor, original_max, beta_fast), (4.0, 32768.0, 32.0));
            assert!((mscale - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-12);
        } o => panic!("{o:?}") }
        assert_eq!(from_hf_config(&cfg(r#"{"rope_type":"dynamic","factor":2.0}"#)).unwrap().rope_scaling, RopeScaling::None);
        assert!(matches!(from_hf_config(&cfg(r#"{"rope_type":"longrope","long_factor":[1.0]}"#)).unwrap().support, Support::Planned(w) if w.contains("LongRoPE")));
        assert!(matches!(from_hf_config(&cfg(r#"{"rope_type":"mystery"}"#)).unwrap().support, Support::Planned(_)));
        // newer configs nest theta and scaling in rope_parameters
        let s = from_hf_config(r#"{"model_type":"llama","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"rope_parameters":{"rope_type":"linear","factor":2.0,"rope_theta":5000.0}}"#).unwrap();
        assert_eq!((s.rope_theta, s.rope_scaling), (5000.0, RopeScaling::Linear { factor: 2.0 }));
    }

    #[test]
    fn plain_llama_keeps_every_default() {
        let s = from_hf_config(LLAMA).unwrap();
        assert_eq!((s.norm, s.norm_offset, s.act, s.gated_mlp, s.embed_scale), (NormKind::Rms, false, Act::Silu, true, 1.0));
        assert_eq!((s.qk_norm, s.qkv_bias, s.post_norms, s.parallel_residual, s.attn_softcap, s.sliding_window), (false, false, false, false, 0.0, 0));
        assert_eq!((s.rotary_dim, s.rope_scaling, s.attn_scale, s.rope_theta_local), (128, RopeScaling::None, 0.0, 0.0));
    }

    #[test]
    fn gguf_gemma_does_not_double_the_norm_offset() {
        let mut g = Gguf::parse(&crate::gguf::tests::sample()[..]).unwrap();
        g.meta.insert("general.architecture".into(), crate::gguf::Value::Str("gemma2".into()));
        for (k, v) in [("gemma2.block_count", 6u64), ("gemma2.embedding_length", 64), ("gemma2.attention.head_count", 4), ("gemma2.attention.sliding_window", 16)] {
            g.meta.insert(k.into(), crate::gguf::Value::U64(v));
        }
        g.meta.insert("gemma2.final_logit_softcapping".into(), crate::gguf::Value::F64(30.0));
        let s = from_gguf(&g).unwrap();
        assert!(!s.norm_offset, "llama.cpp already added 1 to Gemma norm weights");
        assert_eq!((s.embed_scale, s.final_softcap, s.sliding_window), (8.0, 30.0, 16));
        assert_eq!(s.sliding_layers, [true, false, true, false, true, false]);
    }
}
