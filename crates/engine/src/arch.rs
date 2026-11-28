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
    /// Mamba / Mamba-2 / RWKV-6 / RWKV-7: no attention, no KV cache. Runs through
    /// [`crate::family::SsmModel`] (`family::load_any`), not through the dense forward pass.
    Recurrent,
    /// Attention layers interleaved with state-space layers (Jamba, Bamba). Runs through
    /// [`crate::hybrid::HybridModel`] (`family::load_any`).
    Hybrid,
    /// A model with its own implementation behind `family::load_any` (the text says which): it does
    /// not fit the shared decoder, so it has a dedicated forward pass.
    Native(&'static str),
    /// Some of the building blocks exist but the whole model cannot be assembled and run yet; the
    /// text says what is there and what is missing.
    Partial(&'static str),
    /// Needs an attention or state-space variant that is not implemented yet.
    Planned(&'static str),
}

impl Support {
    /// Can the engine run a model of this support level today (through whichever model type fits)?
    pub fn runnable(&self) -> bool {
        !matches!(self, Support::Partial(_) | Support::Planned(_))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    Rms,
    Layer,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Silu,
    GeluTanh,
    Gelu,
}

/// Position-embedding scaling used to stretch the context window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeScaling {
    None,
    Linear {
        factor: f64,
    },
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max: f64,
    },
    Yarn {
        factor: f64,
        original_max: f64,
        beta_fast: f64,
        beta_slow: f64,
        mscale: f64,
    },
    /// LongRoPE (Phi-3 family): one divisor per rotated pair, one list for windows up to
    /// `original_max` (`Spec::rope_short`) and one for longer windows (`Spec::rope_long`), plus an
    /// attention magnitude factor. The tables live in the [`Spec`]; use [`Spec::rope_freqs`].
    LongRope {
        factor: f64,
        original_max: f64,
        mscale: f64,
    },
    /// No rotation at all: attention without any position encoding (Jamba's attention layers).
    Nope,
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
    /// LongRoPE frequency divisors for windows up to the original size (`rotary_dim / 2` values), else empty.
    pub rope_short: Vec<f64>,
    /// LongRoPE frequency divisors for longer windows, else empty.
    pub rope_long: Vec<f64>,
    /// Attention score multiplier; 0 = `1/sqrt(head_dim)`.
    pub attn_scale: f64,
    /// Granite: multiplier on what attention and the MLP add to the residual stream; 1 = none.
    pub residual_scale: f64,
    /// Granite: the final logits are divided by this; 1 = none.
    pub logit_scale: f64,
    // ---- mixture of experts ----
    /// Intermediate size of one routed expert; 0 = same as `ffn` (Mixtral).
    pub moe_ffn: u64,
    /// Divide the selected routing weights by their sum (Mixtral, `norm_topk_prob`).
    pub norm_topk_prob: bool,
    /// Multiplier on the routing weights (`routed_scaling_factor`); 1 = none.
    pub routed_scale: f64,
    /// Router scores are sigmoids, not a softmax (DeepSeek-V3, GLM-4.5).
    pub moe_sigmoid: bool,
    /// `(groups, groups kept)` for group-limited routing; `(0, 0)` = none.
    pub moe_groups: (u64, u64),
}

impl Spec {
    /// A plain dense Llama-style decoder; variants are switched on by assigning fields afterwards.
    #[allow(clippy::too_many_arguments)]
    pub fn dense(
        family: &str,
        layers: u64,
        hidden: u64,
        heads: u64,
        kv_heads: u64,
        head_dim: u64,
        ffn: u64,
        vocab: u64,
        context: u64,
    ) -> Spec {
        Spec {
            family: family.into(),
            support: classify(family),
            layers,
            hidden,
            heads,
            kv_heads,
            head_dim,
            ffn,
            vocab,
            context,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            experts: 0,
            experts_used: 0,
            shared_experts: 0,
            tied_embeddings: false,
            norm: NormKind::Rms,
            norm_offset: false,
            act: Act::Silu,
            gated_mlp: true,
            embed_scale: 1.0,
            qk_norm: false,
            qkv_bias: false,
            post_norms: false,
            parallel_residual: false,
            attn_softcap: 0.0,
            final_softcap: 0.0,
            sliding_window: 0,
            sliding_layers: vec![],
            rope_theta_local: 0.0,
            rotary_dim: head_dim,
            rope_scaling: RopeScaling::None,
            rope_short: vec![],
            rope_long: vec![],
            attn_scale: 0.0,
            residual_scale: 1.0,
            logit_scale: 1.0,
            moe_ffn: 0,
            norm_topk_prob: false,
            routed_scale: 1.0,
            moe_sigmoid: false,
            moe_groups: (0, 0),
        }
    }
    /// Width of one routed expert.
    pub fn expert_ffn(&self) -> u64 {
        if self.moe_ffn > 0 {
            self.moe_ffn
        } else {
            self.ffn
        }
    }
    /// Inverse frequencies and cos/sin magnitude factor of the global RoPE. Identical to
    /// `model::rope_inv_freq` except for LongRoPE, whose per-pair divisors come from this spec
    /// (the long list when the configured window exceeds the original one, otherwise the short list).
    pub fn rope_freqs(&self) -> (Vec<f64>, f64) {
        let rd = self.rotary_dim as usize;
        if self.rope_scaling == RopeScaling::Nope {
            return (vec![0.0; rd / 2], 1.0);
        }
        if let RopeScaling::LongRope {
            original_max,
            mscale,
            ..
        } = self.rope_scaling
        {
            let table = if self.context as f64 > original_max {
                &self.rope_long
            } else {
                &self.rope_short
            };
            if table.len() == rd / 2 && table.iter().all(|f| f.is_finite() && *f > 0.0) {
                let inv = (0..rd / 2)
                    .map(|i| {
                        1.0 / (table[i]
                            * crate::simd::rmath::powf64(
                                self.rope_theta,
                                (2.0 * i as f64) / rd as f64,
                            ))
                    })
                    .collect();
                return (inv, mscale);
            }
        }
        crate::model::rope_inv_freq(self.rope_theta, rd, self.rope_scaling)
    }
    /// Is this a LongRoPE spec whose factor tables are still missing (GGUF keeps them in tensors)?
    pub fn needs_rope_factors(&self) -> bool {
        matches!(self.rope_scaling, RopeScaling::LongRope { .. })
            && self.rope_short.len() != self.rotary_dim as usize / 2
    }
    /// Install LongRoPE tables read from `rope_factors_short.weight` / `rope_factors_long.weight`.
    pub fn set_rope_factors(&mut self, short: Vec<f64>, long: Vec<f64>) -> Result<(), SpecError> {
        let n = self.rotary_dim as usize / 2;
        if short.len() != n
            || long.len() != n
            || short
                .iter()
                .chain(&long)
                .any(|f| !(f.is_finite() && *f > 0.0))
        {
            return Err(SpecError::Absurd("rope factors"));
        }
        self.rope_short = short;
        self.rope_long = long;
        if let Support::Partial(_) = self.support {
            self.support = classify(&self.family);
        }
        Ok(())
    }
    /// Does layer `i` use the sliding window?
    pub fn is_sliding(&self, i: usize) -> bool {
        self.sliding_window > 0 && self.sliding_layers.get(i).copied().unwrap_or(false)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum SpecError {
    Missing(&'static str),
    NotAnObject,
    Absurd(&'static str),
}
impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecError::Missing(k) => write!(f, "the model description has no `{k}`"),
            SpecError::NotAnObject => write!(f, "the model description is not a JSON object"),
            SpecError::Absurd(k) => write!(f, "the model description has an unreasonable `{k}`"),
        }
    }
}
impl std::error::Error for SpecError {}

/// Classify a `model_type` / GGUF architecture name. The text of `Partial` and `Planned` says what
/// exists and what is missing; nothing is claimed that is not built.
pub fn classify(name: &str) -> Support {
    let n = name.to_ascii_lowercase();
    match n.as_str() {
        "llama" | "mistral" | "qwen" | "qwen2" | "qwen3" | "gemma" | "gemma2" | "gemma3"
        | "gemma3_text" | "phi" | "phi3" | "olmo" | "olmo2" | "granite" | "cohere" | "cohere2"
        | "command-r" | "internlm2" | "yi" | "smollm3" | "stablelm" | "starcoder2" | "exaone"
        | "glm" | "glm4" | "baichuan" | "minicpm" | "granitemoe" | "glm4moe" | "glm4_moe" => {
            Support::Standard
        }
        "falcon" | "gpt_neox" | "gptneox" | "gpt2" | "gptj" | "bloom" | "mpt" => {
            Support::Native("GPT-2 / GPT-J / GPT-NeoX / Falcon / BLOOM / MPT (own model, GGUF)")
        }
        "opt" => Support::Planned("OPT has no GGUF conversion"),
        "mixtral" | "qwen2_moe" | "qwen2moe" | "qwen3_moe" | "qwen3moe" => Support::Moe,
        "olmoe" => Support::Planned("OLMoE normalises whole q/k projections, not heads"),
        "gpt_oss" | "gpt-oss" | "gptoss" => Support::Native("gpt-oss (own model, GGUF)"),
        "phimoe" | "dbrx" | "arctic" | "jamba_moe" | "llama4" | "hunyuan_moe" | "ernie4_5_moe"
        | "grok" => Support::Planned("this mixture-of-experts layout"),
        // multi-head latent attention exists as a module (mla.rs), the DeepSeek model around it does not
        "deepseek_v2" | "deepseek_v3" | "deepseek2" | "kimi_k2" => Support::Partial(
            "multi-head latent attention is implemented (mla.rs, absorbed and naive), but the full DeepSeek model — MLA layers, dense first layers, shared and routed experts — is not assembled or loadable yet",
        ),
        "minimax" => Support::Planned("lightning attention and this mixture-of-experts layout"),
        // state-space and recurrent families: run through family::SsmModel
        "mamba" | "falcon_mamba" | "falcon-mamba" | "mamba2" | "rwkv6" | "rwkv-6" | "rwkv7"
        | "rwkv-7" => Support::Recurrent,
        "rwkv" | "rwkv4" | "rwkv5" => {
            Support::Planned("only RWKV-6 and RWKV-7 are implemented, not RWKV-4/5")
        }
        // attention interleaved with state-space layers: run through hybrid::HybridModel
        "jamba" | "bamba" => Support::Hybrid,
        "zamba" | "zamba2" => Support::Partial(
            "the Mamba mixers exist, but Zamba's shared attention block (fed with the embeddings concatenated to the hidden state, with per-use LoRA adapters) is not implemented",
        ),
        "granitemoehybrid" | "granite_moe_hybrid" | "falcon_h1" | "nemotron_h" | "lfm2" => {
            Support::Planned("this hybrid layout (parallel or MoE feed-forward blocks)")
        }
        "t5" | "mt5" | "bart" | "whisper" | "bert" | "roberta" => {
            Support::Planned("encoder / encoder-decoder models")
        }
        "gemma4" | "gemma4_text" => Support::Native("Gemma 4 (own model, GGUF)"),
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
struct Traits {
    layer_norm: bool,
    gated: bool,
    parallel: bool,
    gemma: bool,
    qk_norm: bool,
    post_norms: bool,
    qkv_bias: bool,
    act: Act,
    sliding_pattern: u64,
}
fn traits(family: &str) -> Traits {
    let mut t = Traits {
        layer_norm: false,
        gated: true,
        parallel: false,
        gemma: false,
        qk_norm: false,
        post_norms: false,
        qkv_bias: false,
        act: Act::Silu,
        sliding_pattern: 0,
    };
    match family {
        "gemma" => {
            t.gemma = true;
            t.act = Act::GeluTanh;
        }
        "gemma2" => {
            t.gemma = true;
            t.act = Act::GeluTanh;
            t.post_norms = true;
            t.sliding_pattern = 2;
        }
        "gemma3" | "gemma3_text" => {
            t.gemma = true;
            t.act = Act::GeluTanh;
            t.post_norms = true;
            t.qk_norm = true;
            t.sliding_pattern = 6;
        }
        "qwen2" | "qwen2_moe" | "qwen2moe" => t.qkv_bias = true,
        "qwen3" | "qwen3_moe" | "qwen3moe" => t.qk_norm = true,
        "starcoder2" => {
            t.layer_norm = true;
            t.gated = false;
            t.act = Act::GeluTanh;
        }
        "gpt_neox" | "gptneox" => {
            t.layer_norm = true;
            t.gated = false;
            t.parallel = true;
            t.act = Act::Gelu;
        }
        "falcon" => {
            t.layer_norm = true;
            t.gated = false;
            t.act = Act::Gelu;
        }
        "phi" => {
            t.layer_norm = true;
            t.gated = false;
            t.parallel = true;
            t.act = Act::GeluTanh;
        }
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

/// LongRoPE tables found in a config: (short divisors, long divisors).
type RopeTables = Option<(Vec<f64>, Vec<f64>)>;

/// Reads `rope_scaling` / `rope_parameters`. Returns the scaling and, for kinds we cannot do
/// faithfully, a reason (the caller then downgrades the support level instead of guessing), plus the
/// LongRoPE factor lists when there are any.
fn parse_rope_scaling(
    c: &Value,
    root: &Value,
    orig_default: f64,
) -> (RopeScaling, Option<&'static str>, RopeTables) {
    let Some(r) = ["rope_scaling", "rope_parameters"]
        .iter()
        .find_map(|k| c.get(*k).or_else(|| root.get(*k)).filter(|v| v.is_object()))
    else {
        return (RopeScaling::None, None, None);
    };
    let kind = r
        .get("rope_type")
        .or_else(|| r.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("default");
    let f = |k: &str| r.get(k).and_then(Value::as_f64);
    let factor = f("factor").unwrap_or(1.0);
    let orig = f("original_max_position_embeddings").unwrap_or(orig_default);
    let none = |s: RopeScaling, why: Option<&'static str>| (s, why, None);
    match kind {
        "default" => none(RopeScaling::None, None),
        "linear" => none(RopeScaling::Linear { factor }, None),
        "llama3" => none(
            RopeScaling::Llama3 {
                factor,
                low_freq_factor: f("low_freq_factor").unwrap_or(1.0),
                high_freq_factor: f("high_freq_factor").unwrap_or(4.0),
                original_max: f("original_max_position_embeddings").unwrap_or(8192.0),
            },
            None,
        ),
        "yarn" => none(
            RopeScaling::Yarn {
                factor,
                original_max: orig,
                beta_fast: f("beta_fast").unwrap_or(32.0),
                beta_slow: f("beta_slow").unwrap_or(1.0),
                mscale: f("attention_factor").unwrap_or(if factor > 1.0 {
                    0.1 * factor.ln() + 1.0
                } else {
                    1.0
                }),
            },
            None,
        ),
        // dynamic NTK only changes behaviour beyond the original window, so short contexts are exact
        "dynamic" => none(RopeScaling::None, None),
        // Qwen2-VL "multimodal" RoPE: text tokens use the same position on all three axes, which is plain RoPE
        "mrope" => none(RopeScaling::None, None),
        "longrope" | "su" => {
            let list = |k: &str| -> Option<Vec<f64>> {
                let v: Vec<f64> = r
                    .get(k)?
                    .as_array()?
                    .iter()
                    .map(|x| x.as_f64().unwrap_or(f64::NAN))
                    .collect();
                (!v.is_empty() && v.iter().all(|x| x.is_finite() && *x > 0.0)).then_some(v)
            };
            let (Some(short), Some(long)) = (list("short_factor"), list("long_factor")) else {
                return none(
                    RopeScaling::None,
                    Some("LongRoPE scaling without valid short_factor / long_factor lists"),
                );
            };
            if short.len() != long.len() {
                return none(
                    RopeScaling::None,
                    Some("LongRoPE short_factor and long_factor differ in length"),
                );
            }
            // Hugging Face: when the config names an original window, factor = window / original window
            let given = c
                .get("original_max_position_embeddings")
                .and_then(Value::as_f64)
                .or_else(|| {
                    root.get("original_max_position_embeddings")
                        .and_then(Value::as_f64)
                })
                .or_else(|| f("original_max_position_embeddings"));
            let orig = given.unwrap_or(orig_default);
            let factor = if given.is_some() && orig > 0.0 {
                orig_default / orig
            } else {
                factor
            };
            if !(orig.is_finite() && orig > 1.0 && factor.is_finite() && factor > 0.0) {
                return none(
                    RopeScaling::None,
                    Some("LongRoPE with an unusable original window"),
                );
            }
            let mscale = f("attention_factor").unwrap_or(if factor <= 1.0 {
                1.0
            } else {
                (1.0 + factor.ln() / orig.ln()).sqrt()
            });
            (
                RopeScaling::LongRope {
                    factor,
                    original_max: orig,
                    mscale,
                },
                None,
                Some((short, long)),
            )
        }
        _ => none(RopeScaling::None, Some("unknown RoPE scaling")),
    }
}

/// No real model has more layers than this; larger values would only size per-layer tables.
const MAX_LAYERS: u64 = 1 << 16;

fn layers_from_types(c: &Value, n: u64) -> Option<Vec<bool>> {
    let a = c.get("layer_types")?.as_array()?;
    (a.len() as u64 == n).then(|| {
        a.iter()
            .map(|v| v.as_str() == Some("sliding_attention"))
            .collect()
    })
}
fn layers_from_pattern(n: u64, window: u64, pattern: u64) -> Vec<bool> {
    if window == 0 {
        return vec![];
    }
    (0..n)
        .map(|i| pattern == 0 || (i + 1) % pattern != 0)
        .collect()
}

/// Build a [`Spec`] from a Hugging Face `config.json`. Multimodal wrappers keep the language
/// model under `text_config`; that is followed automatically.
pub fn from_hf_config(json: &str) -> Result<Spec, SpecError> {
    let root: Value = serde_json::from_str(json).map_err(|_| SpecError::NotAnObject)?;
    if !root.is_object() {
        return Err(SpecError::NotAnObject);
    }
    let c = root
        .get("text_config")
        .filter(|t| t.is_object())
        .unwrap_or(&root);
    let family = c
        .get("model_type")
        .or_else(|| root.get("model_type"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let hidden = first_u64(c, &["hidden_size", "n_embd", "d_model"])
        .ok_or(SpecError::Missing("hidden_size"))?;
    let layers = first_u64(c, &["num_hidden_layers", "n_layer", "num_layers"])
        .ok_or(SpecError::Missing("num_hidden_layers"))?;
    if layers > MAX_LAYERS {
        return Err(SpecError::Absurd("num_hidden_layers"));
    }
    let heads = first_u64(c, &["num_attention_heads", "n_head"])
        .ok_or(SpecError::Missing("num_attention_heads"))?;
    if heads == 0 {
        return Err(SpecError::Missing("num_attention_heads"));
    }
    let kv_heads = first_u64(
        c,
        &["num_key_value_heads", "n_head_kv", "multi_query_group_num"],
    )
    .unwrap_or(heads);
    let experts = first_u64(
        c,
        &[
            "num_local_experts",
            "num_experts",
            "n_routed_experts",
            "moe_num_experts",
        ],
    )
    .unwrap_or(0);
    let head_dim = first_u64(c, &["head_dim"])
        .filter(|d| *d > 0)
        .unwrap_or(hidden / heads);
    let t = traits(&family);
    let mut s = Spec::dense(
        &family,
        layers,
        hidden,
        heads,
        kv_heads.max(1),
        head_dim,
        first_u64(c, &["intermediate_size", "ffn_dim", "n_inner"])
            .unwrap_or(hidden.saturating_mul(4)),
        first_u64(c, &["vocab_size"]).unwrap_or(0),
        first_u64(
            c,
            &[
                "max_position_embeddings",
                "n_positions",
                "seq_length",
                "max_seq_len",
            ],
        )
        .unwrap_or(4096),
    );
    s.support = match classify(&family) {
        Support::Standard if family == "glm4_moe" => {
            Support::Planned("GLM-4.5 safetensors are not read; use the GGUF")
        }
        Support::Standard if family == "granitemoe" => Support::Planned(
            "GraniteMoE safetensors keep the experts fused (input_linear); convert the model to GGUF",
        ),
        Support::Standard if experts > 1 => Support::Moe,
        x => x,
    };
    let rope_obj = c.get("rope_parameters").filter(|v| v.is_object());
    s.rope_theta = first_f64(c, &["rope_theta", "rotary_emb_base"])
        .or_else(|| {
            rope_obj
                .and_then(|r| r.get("rope_theta"))
                .and_then(Value::as_f64)
        })
        .unwrap_or(if t.gemma && family.starts_with("gemma3") {
            1_000_000.0
        } else {
            10_000.0
        });
    s.norm_eps = first_f64(
        c,
        &[
            "rms_norm_eps",
            "layer_norm_epsilon",
            "layer_norm_eps",
            "norm_eps",
            "norm_epsilon",
        ],
    )
    .unwrap_or(1e-5);
    s.experts = experts;
    s.experts_used = first_u64(
        c,
        &[
            "num_experts_per_tok",
            "num_experts_per_token",
            "moe_top_k",
            "top_k",
        ],
    )
    .unwrap_or(if experts > 0 { 2 } else { 0 });
    s.shared_experts = first_u64(c, &["n_shared_experts", "num_shared_experts"]).unwrap_or(0);
    s.moe_ffn = first_u64(c, &["moe_intermediate_size", "expert_intermediate_size"]).unwrap_or(0);
    // Mixtral always renormalises; the others say so in their config (default off, as in transformers)
    s.norm_topk_prob = c
        .get("norm_topk_prob")
        .and_then(Value::as_bool)
        .unwrap_or(family == "mixtral");
    s.routed_scale = first_f64(c, &["routed_scaling_factor"])
        .filter(|v| *v > 0.0)
        .unwrap_or(1.0);
    s.tied_embeddings = c
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(t.gemma);

    s.norm = if t.layer_norm {
        NormKind::Layer
    } else {
        NormKind::Rms
    };
    s.norm_offset = t.gemma;
    s.gated_mlp = t.gated;
    s.act = ["hidden_activation", "hidden_act", "activation_function"]
        .iter()
        .find_map(|k| c.get(*k).and_then(Value::as_str))
        .and_then(parse_act)
        .unwrap_or(t.act);
    s.embed_scale = if t.gemma { (hidden as f64).sqrt() } else { 1.0 };
    s.qk_norm = t.qk_norm;
    s.qkv_bias = t.qkv_bias
        || c.get("attention_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        || c.get("use_bias").and_then(Value::as_bool).unwrap_or(false);
    s.post_norms = t.post_norms;
    s.parallel_residual = c
        .get("use_parallel_residual")
        .or_else(|| c.get("parallel_attn"))
        .and_then(Value::as_bool)
        .unwrap_or(t.parallel);
    s.attn_softcap = c
        .get("attn_logit_softcapping")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    s.final_softcap = c
        .get("final_logit_softcapping")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let qwen_off = matches!(
        family.as_str(),
        "qwen2" | "qwen2_moe" | "qwen3" | "qwen3_moe"
    ) && c.get("use_sliding_window").and_then(Value::as_bool) != Some(true);
    s.sliding_window = if qwen_off {
        0
    } else {
        c.get("sliding_window").and_then(Value::as_u64).unwrap_or(0)
    };
    s.sliding_layers = if s.sliding_window == 0 {
        vec![]
    } else {
        layers_from_types(c, layers).unwrap_or_else(|| {
            layers_from_pattern(
                layers,
                s.sliding_window,
                c.get("sliding_window_pattern")
                    .and_then(Value::as_u64)
                    .unwrap_or(t.sliding_pattern),
            )
        })
    };
    s.rope_theta_local = c
        .get("rope_local_base_freq")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let pf = first_f64(c, &["partial_rotary_factor", "rotary_pct"])
        .or_else(|| {
            rope_obj
                .and_then(|r| r.get("partial_rotary_factor"))
                .and_then(Value::as_f64)
        })
        .unwrap_or(1.0);
    s.rotary_dim = (((head_dim as f64 * pf) as u64) & !1).clamp(2, head_dim.max(2));
    let (rs, unsupported, tables) = parse_rope_scaling(c, &root, s.context as f64);
    s.rope_scaling = rs;
    if let Some((short, long)) = tables {
        // the lists have one divisor per rotated pair; anything else would silently rotate wrongly
        if short.len() as u64 == s.rotary_dim / 2 {
            s.rope_short = short;
            s.rope_long = long;
        } else {
            s.rope_scaling = RopeScaling::None;
            s.support =
                Support::Planned("LongRoPE factor lists that do not match the rotary dimension");
        }
    }
    if let Some(why) = unsupported {
        s.support = Support::Planned(why);
    }
    s.attn_scale = c
        .get("query_pre_attn_scalar")
        .and_then(Value::as_f64)
        .filter(|v| *v > 0.0)
        .map_or(0.0, |v| 1.0 / v.sqrt());
    granite_multipliers_hf(c, &mut s);
    Ok(s)
}

/// Granite's four multipliers (`embedding_multiplier`, `attention_multiplier`,
/// `residual_multiplier`, `logits_scaling`), from a `config.json`.
fn granite_multipliers_hf(c: &Value, s: &mut Spec) {
    let pos = |k: &str| {
        c.get(k)
            .and_then(Value::as_f64)
            .filter(|v| *v > 0.0 && v.is_finite())
    };
    if let Some(v) = pos("embedding_multiplier") {
        s.embed_scale = v;
    }
    if let Some(v) = pos("attention_multiplier") {
        s.attn_scale = v;
    }
    if let Some(v) = pos("residual_multiplier") {
        s.residual_scale = v;
    }
    if let Some(v) = pos("logits_scaling") {
        s.logit_scale = v;
    }
}

/// Build a [`Spec`] from GGUF metadata.
pub fn from_gguf(g: &Gguf) -> Result<Spec, SpecError> {
    let family = g
        .architecture()
        .ok_or(SpecError::Missing("general.architecture"))?
        .to_string();
    let layers = g
        .get_u64("block_count")
        .ok_or(SpecError::Missing("block_count"))?;
    // GLM-4.5 appends next-token-prediction blocks to the file; they are not part of the decoder
    let layers = layers.saturating_sub(g.get_u64("nextn_predict_layers").unwrap_or(0));
    if layers > MAX_LAYERS {
        return Err(SpecError::Absurd("block_count"));
    }
    let hidden = g
        .get_u64("embedding_length")
        .ok_or(SpecError::Missing("embedding_length"))?;
    let heads = g
        .get_u64("attention.head_count")
        .filter(|h| *h > 0)
        .ok_or(SpecError::Missing("attention.head_count"))?;
    let experts = g.get_u64("expert_count").unwrap_or(0);
    let vocab = match g.meta.get("tokenizer.ggml.tokens") {
        Some(crate::gguf::Value::Array(a)) => a.len() as u64,
        _ => 0,
    };
    let head_dim = g
        .get_u64("attention.key_length")
        .filter(|d| *d > 0)
        .unwrap_or(hidden / heads);
    let t = traits(&family);
    let mut s = Spec::dense(
        &family,
        layers,
        hidden,
        heads,
        g.get_u64("attention.head_count_kv").unwrap_or(heads).max(1),
        head_dim,
        g.get_u64("feed_forward_length")
            .unwrap_or(hidden.saturating_mul(4)),
        vocab,
        g.get_u64("context_length").unwrap_or(4096),
    );
    s.support = match classify(&family) {
        Support::Standard if experts > 1 => Support::Moe,
        x => x,
    };
    s.rope_theta = g.get_f64("rope.freq_base").unwrap_or(10_000.0);
    s.norm_eps = g
        .get_f64("attention.layer_norm_rms_epsilon")
        .or_else(|| g.get_f64("attention.layer_norm_epsilon"))
        .unwrap_or(1e-5);
    s.experts = experts;
    s.experts_used = g
        .get_u64("expert_used_count")
        .unwrap_or(if experts > 0 { 2 } else { 0 });
    s.shared_experts = g.get_u64("expert_shared_count").unwrap_or(0);
    s.moe_ffn = g.get_u64("expert_feed_forward_length").unwrap_or(0);
    // llama.cpp renormalises for Mixtral-style `llama` files unless the file says otherwise
    s.norm_topk_prob = match g.meta.get(&format!("{family}.expert_weights_norm")) {
        Some(crate::gguf::Value::Bool(b)) => *b,
        _ => matches!(family.as_str(), "llama" | "granitemoe") && experts > 1,
    };
    s.routed_scale = g
        .get_f64("expert_weights_scale")
        .filter(|v| *v > 0.0)
        .unwrap_or(1.0);
    s.tied_embeddings = !g.tensors.iter().any(|x| x.name == "output.weight");
    s.norm = if t.layer_norm {
        NormKind::Layer
    } else {
        NormKind::Rms
    };
    // llama.cpp's converter already adds 1 to Gemma norm weights, so no offset at run time.
    s.norm_offset = false;
    s.gated_mlp = t.gated;
    s.act = t.act;
    s.embed_scale = if t.gemma { (hidden as f64).sqrt() } else { 1.0 };
    s.qk_norm = t.qk_norm
        || (family == "glm4moe"
            && g.tensors
                .iter()
                .any(|x| x.name.ends_with("attn_q_norm.weight")));
    s.qkv_bias = t.qkv_bias || family == "glm4moe";
    s.post_norms = t.post_norms;
    s.parallel_residual = t.parallel;
    s.moe_sigmoid = g.get_u64("expert_gating_func") == Some(2);
    if let (Some(n), Some(k)) = (
        g.get_u64("expert_group_count"),
        g.get_u64("expert_group_used_count"),
    ) && n > 1
        && k > 0
    {
        s.moe_groups = (n, k);
    }
    let pos = |k: &str| g.get_f64(k).filter(|v| *v > 0.0 && v.is_finite());
    if let Some(v) = pos("embedding_scale") {
        s.embed_scale = v;
    }
    if let Some(v) = pos("attention.scale") {
        s.attn_scale = v;
    }
    if let Some(v) = pos("residual_scale") {
        s.residual_scale = v;
    }
    if let Some(v) = pos("logit_scale") {
        s.logit_scale = v;
    }
    s.attn_softcap = g.get_f64("attn_logit_softcapping").unwrap_or(0.0);
    s.final_softcap = g.get_f64("final_logit_softcapping").unwrap_or(0.0);
    s.sliding_window = g.get_u64("attention.sliding_window").unwrap_or(0);
    s.sliding_layers = layers_from_pattern(
        layers,
        s.sliding_window,
        g.get_u64("attention.sliding_window_pattern")
            .unwrap_or(t.sliding_pattern),
    );
    s.rope_theta_local =
        g.get_f64("rope.freq_base_swa")
            .unwrap_or(if t.gemma && family.starts_with("gemma3") {
                10_000.0
            } else {
                0.0
            });
    s.rotary_dim = g
        .get_u64("rope.dimension_count")
        .map_or(head_dim, |d| (d & !1).clamp(2, head_dim.max(2)));
    let factor = g.get_f64("rope.scaling.factor").unwrap_or(1.0);
    s.rope_scaling = match g
        .meta
        .get(&format!("{family}.rope.scaling.type"))
        .and_then(crate::gguf::Value::as_str)
    {
        Some("linear") => RopeScaling::Linear { factor },
        Some("yarn") => RopeScaling::Yarn {
            factor,
            original_max: g
                .get_f64("rope.scaling.original_context_length")
                .unwrap_or(s.context as f64),
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: if factor > 1.0 {
                0.1 * factor.ln() + 1.0
            } else {
                1.0
            },
        },
        _ => RopeScaling::None,
    };
    // LongRoPE (Phi-3 family): the divisors are tensors, read by `load_rope_factors` when the model loads
    let has_factors = g
        .tensors
        .iter()
        .any(|x| x.name == "rope_factors_long.weight")
        && g.tensors
            .iter()
            .any(|x| x.name == "rope_factors_short.weight");
    let longrope_declared = g
        .meta
        .get(&format!("{family}.rope.scaling.type"))
        .and_then(crate::gguf::Value::as_str)
        == Some("longrope");
    if has_factors || longrope_declared {
        let orig = g
            .get_f64("rope.scaling.original_context_length")
            .filter(|o| *o > 1.0)
            .unwrap_or(s.context as f64);
        let f = if orig > 0.0 {
            s.context as f64 / orig
        } else {
            1.0
        };
        let mscale = g
            .get_f64("rope.scaling.attn_factor")
            .filter(|m| *m > 0.0)
            .unwrap_or(if f <= 1.0 {
                1.0
            } else {
                (1.0 + f.ln() / orig.ln()).sqrt()
            });
        if has_factors {
            s.rope_scaling = RopeScaling::LongRope {
                factor: f,
                original_max: orig,
                mscale,
            };
        }
        if longrope_declared && !has_factors {
            s.support =
                Support::Planned("LongRoPE without rope_factors_short / rope_factors_long tensors");
        }
    }
    Ok(s)
}

/// Read LongRoPE's `rope_factors_short.weight` / `rope_factors_long.weight` into `spec` (GGUF stores
/// them as tensors). Does nothing when the spec does not need them.
pub fn load_rope_factors(
    spec: &mut Spec,
    src: &dyn crate::loader::TensorSource,
) -> Result<(), String> {
    if !spec.needs_rope_factors() {
        return Ok(());
    }
    let rd = |n: &str| {
        src.read(n)
            .map(|v| v.into_iter().map(f64::from).collect::<Vec<f64>>())
            .map_err(|e| format!("{n}: {e}"))
    };
    let (short, long) = (
        rd("rope_factors_short.weight")?,
        rd("rope_factors_long.weight")?,
    );
    spec.set_rope_factors(short, long)
        .map_err(|e| e.to_string())
}

impl Spec {
    /// Bytes of KV cache per token (f16 keys and values).
    pub fn kv_bytes_per_token(&self) -> u64 {
        4u64.saturating_mul(self.layers)
            .saturating_mul(self.kv_heads)
            .saturating_mul(self.head_dim)
    }
    pub fn kv_bytes(&self, tokens: u64) -> u64 {
        self.kv_bytes_per_token().saturating_mul(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA: &str = r#"{"model_type":"llama","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,
        "num_key_value_heads":8,"intermediate_size":14336,"vocab_size":128256,"rope_theta":500000.0,"rms_norm_eps":1e-5}"#;

    #[test]
    fn llama_style_config() {
        let s = from_hf_config(LLAMA).unwrap();
        assert_eq!(
            (s.layers, s.hidden, s.heads, s.kv_heads, s.head_dim),
            (32, 4096, 32, 8, 128)
        );
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
        assert_eq!(
            (s.family.as_str(), s.hidden, s.head_dim),
            ("gemma3_text", 2560, 256)
        );
    }

    #[test]
    fn unsupported_families_say_why() {
        assert!(
            matches!(classify("deepseek_v3"), Support::Partial(w) if w.contains("latent") && w.contains("not assembled"))
        );
        assert!(
            matches!(classify("zamba2"), Support::Partial(w) if w.contains("shared attention"))
        );
        assert!(matches!(classify("rwkv5"), Support::Planned(w) if w.contains("RWKV-6")));
        assert!(matches!(classify("minimax"), Support::Planned(_)));
        assert!(matches!(
            classify("never-heard-of-it"),
            Support::Planned("unknown architecture")
        ));
    }

    #[test]
    fn broken_configs_are_errors() {
        assert_eq!(from_hf_config("[1]"), Err(SpecError::NotAnObject));
        assert_eq!(from_hf_config("{}"), Err(SpecError::Missing("hidden_size")));
        assert!(
            from_hf_config(r#"{"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":0}"#)
                .is_err()
        );
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
        assert_eq!(
            (
                s.norm,
                s.norm_offset,
                s.act,
                s.post_norms,
                s.tied_embeddings
            ),
            (NormKind::Rms, true, Act::GeluTanh, true, true)
        );
        assert!((s.embed_scale - 48.0).abs() < 1e-9);
        assert_eq!(
            (s.attn_softcap, s.final_softcap, s.sliding_window),
            (50.0, 30.0, 4096)
        );
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
        assert_eq!(
            s.sliding_layers
                .iter()
                .map(|b| *b as u8)
                .collect::<Vec<_>>(),
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
        );
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
        assert_eq!(
            (
                n.norm,
                n.gated_mlp,
                n.parallel_residual,
                n.act,
                n.rotary_dim
            ),
            (NormKind::Layer, false, true, Act::Gelu, 4)
        );
        assert!(matches!(n.support, Support::Native(_)));
        let sc = from_hf_config(r#"{"model_type":"starcoder2","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"use_bias":true,"norm_epsilon":1e-5,"hidden_act":"gelu_pytorch_tanh"}"#).unwrap();
        assert_eq!(
            (
                sc.norm,
                sc.gated_mlp,
                sc.parallel_residual,
                sc.qkv_bias,
                sc.act,
                sc.support
            ),
            (
                NormKind::Layer,
                false,
                false,
                true,
                Act::GeluTanh,
                Support::Standard
            )
        );
        let p = from_hf_config(r#"{"model_type":"phi3","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"partial_rotary_factor":0.5}"#).unwrap();
        assert_eq!(
            (p.rotary_dim, p.norm, p.gated_mlp),
            (8, NormKind::Rms, true)
        );
    }

    #[test]
    fn rope_scaling_kinds_from_config() {
        let cfg = |r: &str| {
            format!(
                r#"{{"model_type":"llama","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"max_position_embeddings":131072,"rope_scaling":{r}}}"#
            )
        };
        let s = from_hf_config(&cfg(r#"{"rope_type":"llama3","factor":8.0,"low_freq_factor":1.0,"high_freq_factor":4.0,"original_max_position_embeddings":8192}"#)).unwrap();
        assert_eq!(
            s.rope_scaling,
            RopeScaling::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max: 8192.0
            }
        );
        let s = from_hf_config(&cfg(
            r#"{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}"#,
        ))
        .unwrap();
        match s.rope_scaling {
            RopeScaling::Yarn {
                factor,
                original_max,
                beta_fast,
                mscale,
                ..
            } => {
                assert_eq!((factor, original_max, beta_fast), (4.0, 32768.0, 32.0));
                assert!((mscale - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-12);
            }
            o => panic!("{o:?}"),
        }
        assert_eq!(
            from_hf_config(&cfg(r#"{"rope_type":"dynamic","factor":2.0}"#))
                .unwrap()
                .rope_scaling,
            RopeScaling::None
        );
        assert!(
            matches!(from_hf_config(&cfg(r#"{"rope_type":"longrope","long_factor":[1.0]}"#)).unwrap().support, Support::Planned(w) if w.contains("LongRoPE"))
        );
        // Qwen2-VL's multimodal RoPE is plain RoPE for text
        assert_eq!(
            from_hf_config(&cfg(r#"{"type":"mrope","mrope_section":[1,1,2]}"#))
                .unwrap()
                .rope_scaling,
            RopeScaling::None
        );
        assert!(matches!(
            from_hf_config(&cfg(r#"{"rope_type":"mystery"}"#))
                .unwrap()
                .support,
            Support::Planned(_)
        ));
        // newer configs nest theta and scaling in rope_parameters
        let s = from_hf_config(r#"{"model_type":"llama","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"rope_parameters":{"rope_type":"linear","factor":2.0,"rope_theta":5000.0}}"#).unwrap();
        assert_eq!(
            (s.rope_theta, s.rope_scaling),
            (5000.0, RopeScaling::Linear { factor: 2.0 })
        );
    }

    #[test]
    fn registry_tells_the_truth_about_each_family() {
        for m in ["mamba", "falcon_mamba", "mamba2", "rwkv6", "rwkv7"] {
            assert_eq!(classify(m), Support::Recurrent, "{m}");
        }
        for m in ["jamba", "bamba"] {
            assert_eq!(classify(m), Support::Hybrid, "{m}");
        }
        assert!(
            Support::Recurrent.runnable()
                && Support::Hybrid.runnable()
                && Support::Standard.runnable()
                && Support::Moe.runnable()
        );
        assert!(!Support::Partial("x").runnable() && !Support::Planned("x").runnable());
        // a Mamba config has no attention heads, so it never becomes a dense Spec by accident
        assert!(
            from_hf_config(r#"{"model_type":"mamba","hidden_size":64,"num_hidden_layers":2}"#)
                .is_err()
        );
    }

    const PHI3_LONG: &str = r#"{"model_type":"phi3","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"max_position_embeddings":16384,
        "original_max_position_embeddings":4096,"rope_scaling":{"type":"longrope","short_factor":[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7],
        "long_factor":[2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]}}"#;

    #[test]
    fn longrope_parses_factors_window_ratio_and_attention_scale() {
        let s = from_hf_config(PHI3_LONG).unwrap();
        assert_eq!(s.support, Support::Standard);
        assert_eq!(s.rotary_dim, 16);
        let RopeScaling::LongRope {
            factor,
            original_max,
            mscale,
        } = s.rope_scaling
        else {
            panic!("{:?}", s.rope_scaling)
        };
        assert_eq!((factor, original_max), (4.0, 4096.0));
        assert!((mscale - (1.0 + 4f64.ln() / 4096f64.ln()).sqrt()).abs() < 1e-12);
        assert_eq!((s.rope_short.len(), s.rope_long.len()), (8, 8));
        // legacy name and an explicit attention factor
        let s = from_hf_config(&PHI3_LONG.replace("longrope", "su").replace(
            "\"short_factor\"",
            "\"attention_factor\":1.25,\"short_factor\"",
        ))
        .unwrap();
        assert!(matches!(s.rope_scaling, RopeScaling::LongRope { mscale, .. } if mscale == 1.25));
    }

    #[test]
    fn longrope_frequencies_follow_the_formula_and_pick_the_list_by_window() {
        let mut s = from_hf_config(PHI3_LONG).unwrap();
        // window 16384 > 4096: long factors
        let (inv, ms) = s.rope_freqs();
        for (i, f) in inv.iter().enumerate() {
            let want = 1.0 / (s.rope_long[i] * s.rope_theta.powf(2.0 * i as f64 / 16.0));
            assert!((f - want).abs() <= 1e-12 * want.abs(), "{i}: {f} vs {want}");
        }
        assert!(ms > 1.0);
        // a window within the original one switches to the short list
        s.context = 4096;
        let (inv, _) = s.rope_freqs();
        let want = 1.0 / (s.rope_short[3] * s.rope_theta.powf(6.0 / 16.0));
        assert!((inv[3] - want).abs() <= 1e-12 * want);
        // plain RoPE is unchanged for every other scaling
        let plain = from_hf_config(LLAMA).unwrap();
        assert_eq!(
            plain.rope_freqs(),
            crate::model::rope_inv_freq(
                plain.rope_theta,
                plain.rotary_dim as usize,
                RopeScaling::None
            )
        );
    }

    #[test]
    fn longrope_drives_the_real_forward_pass() {
        use crate::model::tests::{base_spec, model_of, reference_logits};
        let toks = [3usize, 7, 1, 12, 5, 9];
        let run = |sp: Spec| {
            let m = model_of(sp, false);
            let mut st = m.new_state();
            (
                m.forward_batch(&mut st, &toks).unwrap(),
                reference_logits(&m, &toks),
            )
        };
        let plain = run(base_spec(2, 1, false)).0;
        let long = |short: Vec<f64>, long: Vec<f64>, ctx: u64, ms: f64| {
            let mut sp = base_spec(2, 1, false);
            sp.context = ctx;
            sp.rope_scaling = RopeScaling::LongRope {
                factor: ctx as f64 / 16.0,
                original_max: 16.0,
                mscale: ms,
            };
            sp.rope_short = short;
            sp.rope_long = long;
            sp
        };
        // all-ones factors and no attention scaling are exactly plain RoPE, in either window regime
        for ctx in [16u64, 64] {
            let (got, want) = run(long(vec![1.0; 4], vec![1.0; 4], ctx, 1.0));
            for i in 0..got.len() {
                assert!((got[i] - plain[i]).abs() < 1e-5, "ctx {ctx} logit {i}");
                assert!((got[i] - want[i]).abs() < 1e-4);
            }
        }
        // real factors change the answer, the cached forward pass agrees with the cache-free reference,
        // and the short and long lists are chosen by the window
        let (a, ra) = run(long(
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 2.0, 4.0, 8.0],
            64,
            1.1,
        ));
        let (b, _) = run(long(
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 2.0, 4.0, 8.0],
            16,
            1.1,
        ));
        assert!(a.iter().zip(&ra).all(|(x, y)| (x - y).abs() < 1e-4));
        assert!(
            a.iter().zip(&plain).any(|(x, y)| (x - y).abs() > 1e-3),
            "long factors must matter"
        );
        assert!(
            b.iter().zip(&plain).any(|(x, y)| (x - y).abs() > 1e-4),
            "the attention scale alone must matter"
        );
        assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-4));
    }

    #[test]
    fn gguf_longrope_takes_its_factors_from_tensors() {
        struct Src;
        impl crate::loader::TensorSource for Src {
            fn read(&self, n: &str) -> Result<Vec<f32>, crate::loader::LoadError> {
                match n {
                    "rope_factors_short.weight" => Ok(vec![1.0; 4]),
                    "rope_factors_long.weight" => Ok(vec![1.0, 2.0, 3.0, 4.0]),
                    _ => Err(crate::loader::LoadError::Missing(n.into())),
                }
            }
            fn has(&self, n: &str) -> bool {
                n.starts_with("rope_factors_")
            }
        }
        let mut g = Gguf::parse(&crate::gguf::tests::sample()[..]).unwrap();
        g.meta.insert(
            "general.architecture".into(),
            crate::gguf::Value::Str("phi3".into()),
        );
        for (k, v) in [
            ("phi3.block_count", 2u64),
            ("phi3.embedding_length", 32),
            ("phi3.attention.head_count", 4),
            ("phi3.context_length", 8192),
            ("phi3.rope.scaling.original_context_length", 2048),
        ] {
            g.meta.insert(k.into(), crate::gguf::Value::U64(v));
        }
        // without the tensors nothing changes
        assert_eq!(from_gguf(&g).unwrap().rope_scaling, RopeScaling::None);
        for n in ["rope_factors_short.weight", "rope_factors_long.weight"] {
            g.tensors.push(crate::gguf::TensorInfo {
                name: n.into(),
                dims: vec![4],
                dtype: crate::quant::DType::F32,
                offset: 0,
            });
        }
        let mut s = from_gguf(&g).unwrap();
        assert!(
            matches!(s.rope_scaling, RopeScaling::LongRope { factor, original_max, .. } if factor == 4.0 && original_max == 2048.0)
        );
        assert!(s.needs_rope_factors());
        load_rope_factors(&mut s, &Src).unwrap();
        assert!(!s.needs_rope_factors());
        assert_eq!(s.rope_long, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!((s.rotary_dim, s.rope_freqs().0.len()), (8, 4));
    }

    #[test]
    fn longrope_with_wrong_lists_is_refused_not_guessed() {
        let bad = PHI3_LONG
            .replace("[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7]", "[1.0,1.1,1.2]")
            .replace("[2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]", "[2.0,3.0,4.0]");
        let s = from_hf_config(&bad).unwrap();
        assert!(matches!(s.support, Support::Planned(w) if w.contains("rotary dimension")));
        let neg = PHI3_LONG.replace(
            "[2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]",
            "[2.0,3.0,4.0,5.0,6.0,7.0,8.0,-9.0]",
        );
        assert!(
            matches!(from_hf_config(&neg).unwrap().support, Support::Planned(w) if w.contains("LongRoPE"))
        );
        let mut s = from_hf_config(PHI3_LONG).unwrap();
        assert!(s.set_rope_factors(vec![1.0; 3], vec![1.0; 3]).is_err());
        assert!(s.set_rope_factors(vec![f64::NAN; 8], vec![1.0; 8]).is_err());
    }

    #[test]
    fn plain_llama_keeps_every_default() {
        let s = from_hf_config(LLAMA).unwrap();
        assert_eq!(
            (s.norm, s.norm_offset, s.act, s.gated_mlp, s.embed_scale),
            (NormKind::Rms, false, Act::Silu, true, 1.0)
        );
        assert_eq!(
            (
                s.qk_norm,
                s.qkv_bias,
                s.post_norms,
                s.parallel_residual,
                s.attn_softcap,
                s.sliding_window
            ),
            (false, false, false, false, 0.0, 0)
        );
        assert_eq!(
            (
                s.rotary_dim,
                s.rope_scaling,
                s.attn_scale,
                s.rope_theta_local
            ),
            (128, RopeScaling::None, 0.0, 0.0)
        );
    }

    #[test]
    fn gguf_gemma_does_not_double_the_norm_offset() {
        let mut g = Gguf::parse(&crate::gguf::tests::sample()[..]).unwrap();
        g.meta.insert(
            "general.architecture".into(),
            crate::gguf::Value::Str("gemma2".into()),
        );
        for (k, v) in [
            ("gemma2.block_count", 6u64),
            ("gemma2.embedding_length", 64),
            ("gemma2.attention.head_count", 4),
            ("gemma2.attention.sliding_window", 16),
        ] {
            g.meta.insert(k.into(), crate::gguf::Value::U64(v));
        }
        g.meta.insert(
            "gemma2.final_logit_softcapping".into(),
            crate::gguf::Value::F64(30.0),
        );
        let s = from_gguf(&g).unwrap();
        assert!(
            !s.norm_offset,
            "llama.cpp already added 1 to Gemma norm weights"
        );
        assert_eq!(
            (s.embed_scale, s.final_softcap, s.sliding_window),
            (8.0, 30.0, 16)
        );
        assert_eq!(s.sliding_layers, [true, false, true, false, true, false]);
    }

    #[test]
    fn granite_multipliers_are_read_from_config_and_gguf() {
        let s = from_hf_config(r#"{"model_type":"granite","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"intermediate_size":128,"vocab_size":100,"embedding_multiplier":12.0,"attention_multiplier":0.015625,"residual_multiplier":0.22,"logits_scaling":8.0}"#).unwrap();
        assert_eq!(
            (s.embed_scale, s.attn_scale, s.residual_scale, s.logit_scale),
            (12.0, 0.015625, 0.22, 8.0)
        );
        let plain = from_hf_config(r#"{"model_type":"llama","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"intermediate_size":128,"vocab_size":100}"#).unwrap();
        assert_eq!((plain.residual_scale, plain.logit_scale), (1.0, 1.0));
    }

    #[test]
    fn gguf_granitemoe_is_a_moe_with_the_granite_multipliers() {
        let mut g = Gguf::parse(&crate::gguf::tests::sample()[..]).unwrap();
        g.meta.insert(
            "general.architecture".into(),
            crate::gguf::Value::Str("granitemoe".into()),
        );
        for (k, v) in [
            ("block_count", 4u64),
            ("embedding_length", 64),
            ("attention.head_count", 4),
            ("expert_count", 8),
            ("expert_used_count", 2),
        ] {
            g.meta
                .insert(format!("granitemoe.{k}"), crate::gguf::Value::U64(v));
        }
        for (k, v) in [
            ("embedding_scale", 12.0),
            ("residual_scale", 0.22),
            ("logit_scale", 8.0),
            ("attention.scale", 0.015625),
        ] {
            g.meta
                .insert(format!("granitemoe.{k}"), crate::gguf::Value::F64(v));
        }
        let s = from_gguf(&g).unwrap();
        assert_eq!(s.support, Support::Moe);
        assert!(s.norm_topk_prob);
        assert_eq!(
            (s.embed_scale, s.residual_scale, s.logit_scale, s.attn_scale),
            (12.0, 0.22, 8.0, 0.015625)
        );
    }

    #[test]
    fn gguf_glm4moe_reads_sigmoid_groups_and_skips_nextn_blocks() {
        let mut g = Gguf::parse(&crate::gguf::tests::sample()[..]).unwrap();
        g.meta.insert(
            "general.architecture".into(),
            crate::gguf::Value::Str("glm4moe".into()),
        );
        for (k, v) in [
            ("block_count", 5u64),
            ("nextn_predict_layers", 1),
            ("embedding_length", 64),
            ("attention.head_count", 4),
            ("expert_count", 8),
            ("expert_used_count", 2),
            ("expert_gating_func", 2),
            ("expert_group_count", 4),
            ("expert_group_used_count", 2),
        ] {
            g.meta
                .insert(format!("glm4moe.{k}"), crate::gguf::Value::U64(v));
        }
        let s = from_gguf(&g).unwrap();
        assert_eq!(s.layers, 4);
        assert_eq!(s.support, Support::Moe);
        assert!(s.moe_sigmoid && s.qkv_bias);
        assert_eq!(s.moe_groups, (4, 2));
        assert!(!s.qk_norm);
    }
}
