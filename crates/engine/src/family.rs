//! Complete models for the non-attention families (Mamba-1, Mamba-2, RWKV-6, RWKV-7) on top of the
//! blocks in [`crate::ssm`], plus the pieces that let the rest of the engine treat every family alike:
//!
//! * [`SsmModel`] – embedding → N × (pre-norm + mixer + residual) → final norm → head. The state
//!   ([`SsmState`]) has a fixed size: it does not grow with the context.
//! * [`LanguageModel`] – the small trait both the dense [`Model`] and [`SsmModel`] implement, and a
//!   generic [`generate`] that mirrors [`Model::generate`].
//! * [`AnyModel`] / [`load_any`] – open a folder or file and pick the right family from `config.json`
//!   or from the GGUF architecture.
//!
//! Tensor names (all written from memory of the public checkpoints, none checked against a real file):
//!
//! | Family | Naming | Embedding | Layer norm | Final norm | Head |
//! |---|---|---|---|---|---|
//! | Mamba-1/2 | Hugging Face / mamba-ssm | `backbone.embeddings.weight` (or `backbone.embedding.weight`) | `backbone.layers.N.norm.weight` | `backbone.norm_f.weight` | `lm_head.weight`, else tied |
//! | Mamba-1/2 | llama.cpp GGUF | `token_embd.weight` | `blk.N.attn_norm.weight` | `output_norm.weight` | `output.weight`, else tied |
//! | RWKV-6, RWKV-7 | native (BlinkDL) | `emb.weight` | inside the block (`ln1`, `ln2`, `ln0` on `blocks.0`) | `ln_out.weight/.bias` | `head.weight` |
//! | RWKV-7 | flash-linear-attention | `model.embeddings.weight` | inside the block (`attn_norm`, `ffn_norm`, `pre_norm` on layer 0) | `model.norm.weight/.bias` | `lm_head.weight` |
//!
//! Not covered: RWKV-6 in the flash-linear-attention layout, RWKV in GGUF (different tensor names),
//! Falcon-Mamba's extra GGUF norms, Zamba/Zamba2 (shared attention block), mixture-of-experts Jamba and RWKV-4/5; Jamba and Bamba hybrids are in `hybrid`. Weights stay
//! dense f32 (no quantised path yet).
use crate::arch;
use crate::gguf::Gguf;
use crate::loader::{
    GgufFile, LoadError, LoadOptions, MoeOptions, Naming, SafeTensors, TensorSource, load_model,
};
use crate::model::{GenerateOptions, GenerateStats, Model, ModelError, Sampler, State, StopReason};
use crate::ops::{matvec, rmsnorm};
use crate::ssm::{
    Mamba1, Mamba1Config, Mamba1State, Mamba2, Mamba2Config, Mamba2State, MambaNames, Rwkv6Config,
    Rwkv6Layer, Rwkv6Names, Rwkv6State, Rwkv7Config, Rwkv7Layer, Rwkv7Names, Rwkv7State, SsmError,
};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

const MAX_LAYERS: usize = 1 << 14;
const DEFAULT_CONTEXT: usize = 1 << 20;

// ------------------------------------------------------------------------------------- errors

#[derive(Debug)]
pub enum FamilyError {
    /// The model description is missing, malformed or out of range.
    Config(String),
    /// A tensor is missing, has the wrong size or the file is damaged.
    Load(String),
    /// A recognised model that this engine cannot run (yet); the text says why.
    Unsupported(String),
}
impl std::fmt::Display for FamilyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FamilyError::Config(s) | FamilyError::Load(s) => write!(f, "{s}"),
            FamilyError::Unsupported(s) => write!(f, "not supported yet: {s}"),
        }
    }
}
impl std::error::Error for FamilyError {}
impl From<SsmError> for FamilyError {
    fn from(e: SsmError) -> Self {
        match e {
            SsmError::Config(_) => FamilyError::Config(e.to_string()),
            _ => FamilyError::Load(e.to_string()),
        }
    }
}
impl From<LoadError> for FamilyError {
    fn from(e: LoadError) -> Self {
        FamilyError::Load(e.to_string())
    }
}
impl From<ModelError> for FamilyError {
    fn from(e: ModelError) -> Self {
        FamilyError::Load(e.to_string())
    }
}
impl From<arch::SpecError> for FamilyError {
    fn from(e: arch::SpecError) -> Self {
        FamilyError::Config(e.to_string())
    }
}
impl From<std::io::Error> for FamilyError {
    fn from(e: std::io::Error) -> Self {
        FamilyError::Load(e.to_string())
    }
}

fn cfg_err<T>(s: impl Into<String>) -> Result<T, FamilyError> {
    Err(FamilyError::Config(s.into()))
}
fn load_err<T>(s: impl Into<String>) -> Result<T, FamilyError> {
    Err(FamilyError::Load(s.into()))
}
fn to_model_err(e: SsmError) -> ModelError {
    ModelError::Shape(e.to_string())
}

// ------------------------------------------------------------------------------------ configs

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FamilyKind {
    Mamba1,
    Mamba2,
    Rwkv6,
    Rwkv7,
}

/// Which recurrent family a `model_type` / GGUF architecture name belongs to. `None` for everything
/// else (dense and mixture-of-experts transformers, RWKV-4/5, hybrids).
pub fn classify_family(model_type: &str) -> Option<FamilyKind> {
    match model_type.to_ascii_lowercase().as_str() {
        "mamba" | "falcon_mamba" | "falcon-mamba" => Some(FamilyKind::Mamba1),
        "mamba2" => Some(FamilyKind::Mamba2),
        "rwkv6" | "rwkv-6" => Some(FamilyKind::Rwkv6),
        "rwkv7" | "rwkv-7" => Some(FamilyKind::Rwkv7),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MixerConfig {
    M1(Mamba1Config),
    M2(Mamba2Config),
    R6(Rwkv6Config),
    R7(Rwkv7Config),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FamilyConfig {
    pub kind: FamilyKind,
    pub layers: usize,
    /// Vocabulary size the description claims, `0` if unknown. The embedding table decides the real one.
    pub vocab: usize,
    pub context: usize,
    /// Epsilon of the RMS pre-norms (Mamba) and the final norm.
    pub norm_eps: f32,
    pub mixer: MixerConfig,
}
impl FamilyConfig {
    pub fn hidden(&self) -> usize {
        match &self.mixer {
            MixerConfig::M1(c) => c.d_model,
            MixerConfig::M2(c) => c.d_model,
            MixerConfig::R6(c) => c.hidden,
            MixerConfig::R7(c) => c.hidden,
        }
    }
    fn validate(&self) -> Result<(), FamilyError> {
        if self.layers == 0 || self.layers > MAX_LAYERS {
            return cfg_err(format!("{} layers is out of range", self.layers));
        }
        if self.hidden() == 0 {
            return cfg_err("hidden size is zero");
        }
        if !(self.norm_eps.is_finite() && self.norm_eps >= 0.0) {
            return cfg_err("norm epsilon is not a valid number");
        }
        match &self.mixer {
            MixerConfig::M1(c) => c.validate()?,
            MixerConfig::M2(c) => c.validate()?,
            MixerConfig::R6(c) => c.validate()?,
            MixerConfig::R7(c) => c.validate()?,
        }
        Ok(())
    }

    /// From a Hugging Face `config.json` of a Mamba, Mamba-2, Falcon-Mamba, RWKV-6 or RWKV-7 model.
    pub fn from_hf_config(json: &str) -> Result<FamilyConfig, FamilyError> {
        let c: Value = serde_json::from_str(json)
            .map_err(|e| FamilyError::Config(format!("config.json: {e}")))?;
        if !c.is_object() {
            return cfg_err("config.json is not a JSON object");
        }
        let mt = c.get("model_type").and_then(Value::as_str).unwrap_or("");
        let from_arch = || -> Option<FamilyKind> {
            let a = c
                .get("architectures")?
                .as_array()?
                .first()?
                .as_str()?
                .to_ascii_lowercase();
            if a.contains("mamba2") {
                Some(FamilyKind::Mamba2)
            } else if a.contains("mamba") {
                Some(FamilyKind::Mamba1)
            } else if a.contains("rwkv7") {
                Some(FamilyKind::Rwkv7)
            } else if a.contains("rwkv6") {
                Some(FamilyKind::Rwkv6)
            } else {
                None
            }
        };
        let kind = classify_family(mt).or_else(from_arch).ok_or_else(|| {
            FamilyError::Unsupported(format!(
                "model type `{mt}` is not a Mamba or RWKV-6/7 model"
            ))
        })?;
        let u = |keys: &[&str]| {
            keys.iter()
                .find_map(|k| c.get(*k).and_then(Value::as_u64))
                .map(|v| v as usize)
        };
        let layers = u(&["num_hidden_layers", "n_layer", "num_layers"])
            .ok_or_else(|| FamilyError::Config("no num_hidden_layers".into()))?;
        let norm_eps = [
            "layer_norm_epsilon",
            "norm_epsilon",
            "rms_norm_eps",
            "norm_eps",
        ]
        .iter()
        .find_map(|k| c.get(*k).and_then(Value::as_f64))
        .unwrap_or(1e-5) as f32;
        let mixer = match kind {
            FamilyKind::Mamba1 => MixerConfig::M1(Mamba1Config::from_hf_config(json)?),
            FamilyKind::Mamba2 => MixerConfig::M2(Mamba2Config::from_hf_config(json)?),
            FamilyKind::Rwkv6 => MixerConfig::R6(Rwkv6Config::from_hf_config(json)?),
            FamilyKind::Rwkv7 => MixerConfig::R7(Rwkv7Config::from_hf_config(json)?),
        };
        let cfg = FamilyConfig {
            kind,
            layers,
            vocab: u(&["vocab_size"]).unwrap_or(0),
            context: u(&["max_position_embeddings", "n_positions", "context_length"])
                .unwrap_or(DEFAULT_CONTEXT)
                .clamp(1, usize::MAX / 4),
            norm_eps,
            mixer,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// From the metadata of a llama.cpp GGUF file (`mamba` and `mamba2` only).
    pub fn from_gguf(g: &Gguf) -> Result<FamilyConfig, FamilyError> {
        let arch = g.architecture().ok_or_else(|| {
            FamilyError::Config("the GGUF file has no general.architecture".into())
        })?;
        let kind = classify_family(arch).ok_or_else(|| {
            FamilyError::Unsupported(format!("architecture `{arch}` is not a recurrent family"))
        })?;
        let get = |k: &str| {
            g.get_u64(k)
                .map(|v| v as usize)
                .ok_or_else(|| FamilyError::Config(format!("the GGUF file has no {arch}.{k}")))
        };
        let hidden = get("embedding_length")?;
        let layers = get("block_count")?;
        let inner = get("ssm.inner_size")?;
        let norm_eps = g
            .get_f64("attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5) as f32;
        let mixer = match kind {
            FamilyKind::Mamba1 => MixerConfig::M1(Mamba1Config {
                d_model: hidden,
                d_inner: inner,
                d_state: get("ssm.state_size")?,
                d_conv: get("ssm.conv_kernel")?,
                dt_rank: get("ssm.time_step_rank")?,
                conv_bias: true,
                bias: false,
                bcdt_rms_eps: None,
            }),
            FamilyKind::Mamba2 => {
                let heads = get("ssm.time_step_rank")?;
                if heads == 0 || inner % heads != 0 {
                    return cfg_err("ssm.inner_size is not a multiple of the head count");
                }
                MixerConfig::M2(Mamba2Config {
                    d_model: hidden,
                    n_heads: heads,
                    head_dim: inner / heads,
                    d_state: get("ssm.state_size")?,
                    n_groups: get("ssm.group_count").unwrap_or(1),
                    d_conv: get("ssm.conv_kernel")?,
                    conv_bias: true,
                    bias: false,
                    norm_eps,
                    norm_before_gate: false,
                    chunk_size: 64,
                })
            }
            _ => {
                return Err(FamilyError::Unsupported(
                    "RWKV tensor names in GGUF files are not implemented".into(),
                ));
            }
        };
        let cfg = FamilyConfig {
            kind,
            layers,
            vocab: 0,
            context: g
                .get_u64("context_length")
                .map_or(DEFAULT_CONTEXT, |c| (c as usize).max(1)),
            norm_eps,
            mixer,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

// ------------------------------------------------------------------------------------- model

/// Which tensor names a checkpoint uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SsmNaming {
    Hf,
    Native,
    Gguf,
}

enum Mixer {
    M1(Mamba1),
    M2(Mamba2),
}
enum Stack {
    Mamba {
        norms: Vec<Vec<f32>>,
        mixers: Vec<Mixer>,
    },
    R6(Vec<Rwkv6Layer>),
    R7(Vec<Rwkv7Layer>),
}
enum FinalNorm {
    Rms(Vec<f32>),
    Layer(Vec<f32>, Vec<f32>),
}

pub struct SsmModel {
    /// Declared first so it is dropped before the weights it points at.
    reg: crate::slicegpu::Registration,
    cfg: FamilyConfig,
    hidden: usize,
    vocab: usize,
    embed: Vec<f32>,
    /// RWKV-6's layer norm on the embeddings (`blocks.0.ln0`).
    embed_norm: Option<(Vec<f32>, Vec<f32>)>,
    stack: Stack,
    final_norm: FinalNorm,
    /// `None` = tied to the embedding table.
    head: Option<Vec<f32>>,
}

enum Layers {
    M1(Vec<Mamba1State>),
    M2(Vec<Mamba2State>),
    R6(Vec<Rwkv6State>),
    R7(Vec<Rwkv7State>),
}

/// Per-conversation state of an [`SsmModel`]. Its size depends on the model only, never on how many
/// tokens were processed.
pub struct SsmState {
    layers: Layers,
    pub pos: usize,
}
impl SsmState {
    pub fn bytes(&self) -> usize {
        match &self.layers {
            Layers::M1(v) => v.iter().map(Mamba1State::bytes).sum(),
            Layers::M2(v) => v.iter().map(Mamba2State::bytes).sum(),
            Layers::R6(v) => v.iter().map(Rwkv6State::bytes).sum(),
            Layers::R7(v) => v.iter().map(Rwkv7State::bytes).sum(),
        }
    }
}

fn layer_norm_into(x: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let k = 1.0 / (var + eps).sqrt();
    for (i, o) in out.iter_mut().enumerate() {
        *o = (x[i] - mean) * k * w[i] + b[i];
    }
}

fn first_present(src: &dyn TensorSource, names: &[&str]) -> Option<String> {
    names.iter().find(|n| src.has(n)).map(|n| n.to_string())
}
fn need(src: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, FamilyError> {
    let v = src.read(name)?;
    if v.len() != len {
        return load_err(format!("{name}: {} values, expected {len}", v.len()));
    }
    Ok(v)
}

/// Pick the naming from the tensors that are present.
pub fn detect_naming(src: &dyn TensorSource) -> Option<SsmNaming> {
    if src.has("emb.weight") {
        Some(SsmNaming::Native)
    } else if src.has("backbone.embeddings.weight")
        || src.has("backbone.embedding.weight")
        || src.has("model.embeddings.weight")
    {
        Some(SsmNaming::Hf)
    } else if src.has("token_embd.weight") {
        Some(SsmNaming::Gguf)
    } else {
        None
    }
}

impl SsmModel {
    /// Put the big matrices on `be`; never fails, the CPU stays the fall-back.
    pub fn attach_backend(
        &mut self,
        be: Arc<dyn crate::backend::Backend>,
    ) -> crate::accel::AttachReport {
        self.reg = Default::default();
        let h = self.hidden;
        let mut mats: Vec<(&[f32], usize)> = vec![(self.head.as_deref().unwrap_or(&self.embed), h)];
        match &self.stack {
            Stack::Mamba { mixers, .. } => {
                for m in mixers {
                    match m {
                        Mixer::M1(x) => mats.extend(x.matrices()),
                        Mixer::M2(x) => mats.extend(x.matrices()),
                    }
                }
            }
            Stack::R6(ls) => ls.iter().for_each(|l| mats.extend(l.matrices())),
            Stack::R7(ls) => ls.iter().for_each(|l| mats.extend(l.matrices())),
        }
        let (reg, rep) = crate::slicegpu::register(be, &mats);
        self.reg = reg;
        rep
    }

    pub fn config(&self) -> &FamilyConfig {
        &self.cfg
    }
    pub fn kind(&self) -> FamilyKind {
        self.cfg.kind
    }
    pub fn hidden(&self) -> usize {
        self.hidden
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Load every layer of the model described by `cfg` from `src` (dense f32).
    pub fn load(
        cfg: FamilyConfig,
        src: &dyn TensorSource,
        naming: SsmNaming,
    ) -> Result<SsmModel, FamilyError> {
        cfg.validate()?;
        let h = cfg.hidden();
        let unsupported = |what: &str| FamilyError::Unsupported(what.to_string());
        let (embed_names, final_w, final_b, head_names): (&[&str], &str, Option<&str>, &[&str]) =
            match (cfg.kind, naming) {
                (FamilyKind::Mamba1 | FamilyKind::Mamba2, SsmNaming::Hf) => (
                    &["backbone.embeddings.weight", "backbone.embedding.weight"],
                    "backbone.norm_f.weight",
                    None,
                    &["lm_head.weight"],
                ),
                (FamilyKind::Mamba1 | FamilyKind::Mamba2, SsmNaming::Gguf) => (
                    &["token_embd.weight"],
                    "output_norm.weight",
                    None,
                    &["output.weight"],
                ),
                (FamilyKind::Mamba1 | FamilyKind::Mamba2, SsmNaming::Native) => {
                    return Err(unsupported(
                        "Mamba checkpoints with RWKV-style native names",
                    ));
                }
                (FamilyKind::Rwkv6, SsmNaming::Native) | (FamilyKind::Rwkv7, SsmNaming::Native) => {
                    (
                        &["emb.weight"],
                        "ln_out.weight",
                        Some("ln_out.bias"),
                        &["head.weight"],
                    )
                }
                (FamilyKind::Rwkv7, SsmNaming::Hf) => (
                    &["model.embeddings.weight"],
                    "model.norm.weight",
                    Some("model.norm.bias"),
                    &["lm_head.weight"],
                ),
                (FamilyKind::Rwkv6, _) => {
                    return Err(unsupported(
                        "RWKV-6 is only loadable from the native (BlinkDL) tensor names",
                    ));
                }
                (FamilyKind::Rwkv7, SsmNaming::Gguf) => {
                    return Err(unsupported("RWKV-7 in GGUF files"));
                }
            };
        let en = first_present(src, embed_names).ok_or_else(|| {
            FamilyError::Load(format!("the model has no tensor `{}`", embed_names[0]))
        })?;
        let embed = src.read(&en)?;
        if embed.is_empty() || embed.len() % h != 0 {
            return load_err(format!(
                "{en}: {} values is not a whole number of {h}-wide rows",
                embed.len()
            ));
        }
        let vocab = embed.len() / h;
        if cfg.vocab > vocab {
            return load_err(format!(
                "the configuration has a vocabulary of {} but the embedding table has only {vocab} rows",
                cfg.vocab
            ));
        }
        let head = match first_present(src, head_names) {
            Some(n) => Some(need(src, &n, vocab * h)?),
            None => None,
        };
        let final_norm = match final_b {
            None => FinalNorm::Rms(need(src, final_w, h)?),
            Some(b) => FinalNorm::Layer(need(src, final_w, h)?, need(src, b, h)?),
        };
        let mut embed_norm = None;
        let stack = match &cfg.mixer {
            MixerConfig::M1(_) | MixerConfig::M2(_) => {
                let (mut norms, mut mixers) = (
                    Vec::with_capacity(cfg.layers),
                    Vec::with_capacity(cfg.layers),
                );
                for i in 0..cfg.layers {
                    let nn = if naming == SsmNaming::Gguf {
                        format!("blk.{i}.attn_norm.weight")
                    } else {
                        format!("backbone.layers.{i}.norm.weight")
                    };
                    norms.push(need(src, &nn, h)?);
                    let negated = naming == SsmNaming::Gguf;
                    mixers.push(match &cfg.mixer {
                        MixerConfig::M1(c) => Mixer::M1(Mamba1::load(
                            c.clone(),
                            src,
                            &if negated {
                                MambaNames::gguf(i)
                            } else {
                                MambaNames::hf_mamba1(i)
                            },
                            negated,
                        )?),
                        MixerConfig::M2(c) => Mixer::M2(Mamba2::load(
                            c.clone(),
                            src,
                            &if negated {
                                MambaNames::gguf(i)
                            } else {
                                MambaNames::hf_mamba2(i)
                            },
                            negated,
                        )?),
                        _ => unreachable!(),
                    });
                }
                Stack::Mamba { norms, mixers }
            }
            MixerConfig::R6(c) => {
                if src.has("blocks.0.ln0.weight") {
                    embed_norm = Some((
                        need(src, "blocks.0.ln0.weight", h)?,
                        need(src, "blocks.0.ln0.bias", h)?,
                    ));
                }
                Stack::R6(
                    (0..cfg.layers)
                        .map(|i| Rwkv6Layer::load(c.clone(), src, &Rwkv6Names::native(i)))
                        .collect::<Result<_, _>>()?,
                )
            }
            MixerConfig::R7(c) => {
                let names = |i: usize| {
                    if naming == SsmNaming::Native {
                        Rwkv7Names::native(i)
                    } else {
                        Rwkv7Names::fla(i)
                    }
                };
                Stack::R7(
                    (0..cfg.layers)
                        .map(|i| Rwkv7Layer::load(c.clone(), src, &names(i), i))
                        .collect::<Result<_, _>>()?,
                )
            }
        };
        Ok(SsmModel {
            reg: Default::default(),
            cfg,
            hidden: h,
            vocab,
            embed,
            embed_norm,
            stack,
            final_norm,
            head,
        })
    }

    pub fn new_state(&self) -> SsmState {
        let layers = match &self.stack {
            Stack::Mamba { mixers, .. } => {
                if matches!(mixers.first(), Some(Mixer::M2(_))) {
                    Layers::M2(
                        mixers
                            .iter()
                            .map(|m| {
                                if let Mixer::M2(m) = m {
                                    m.new_state()
                                } else {
                                    unreachable!()
                                }
                            })
                            .collect(),
                    )
                } else {
                    Layers::M1(
                        mixers
                            .iter()
                            .map(|m| {
                                if let Mixer::M1(m) = m {
                                    m.new_state()
                                } else {
                                    unreachable!()
                                }
                            })
                            .collect(),
                    )
                }
            }
            Stack::R6(l) => Layers::R6(l.iter().map(Rwkv6Layer::new_state).collect()),
            Stack::R7(l) => Layers::R7(l.iter().map(Rwkv7Layer::new_state).collect()),
        };
        SsmState { layers, pos: 0 }
    }

    pub fn forward(&self, st: &mut SsmState, token: usize) -> Result<Vec<f32>, ModelError> {
        self.forward_batch(st, &[token])
    }

    /// Process `tokens` (continuing from `st`) and return the logits of the last one. Equal, within
    /// rounding, to calling [`SsmModel::forward`] for each token.
    pub fn forward_batch(
        &self,
        st: &mut SsmState,
        tokens: &[usize],
    ) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        if let Some(&t) = tokens.iter().find(|&&t| t >= self.vocab) {
            return Err(ModelError::Token(t));
        }
        let (h, t) = (self.hidden, tokens.len());
        let mut xs = Vec::with_capacity(t * h);
        for &tok in tokens {
            xs.extend_from_slice(&self.embed[tok * h..(tok + 1) * h]);
        }
        let wrong = || ModelError::Shape("the state belongs to a different model".into());
        let eps = self.cfg.norm_eps;
        match (&self.stack, &mut st.layers) {
            (Stack::Mamba { norms, mixers }, layers) => {
                let mut normed = vec![0.0; t * h];
                for (li, (nw, mx)) in norms.iter().zip(mixers).enumerate() {
                    for (x, o) in xs.chunks_exact(h).zip(normed.chunks_exact_mut(h)) {
                        rmsnorm(x, nw, eps, o);
                    }
                    let out = match (mx, &mut *layers) {
                        (Mixer::M1(m), Layers::M1(s)) => {
                            m.prefill(s.get_mut(li).ok_or_else(wrong)?, &normed)
                        }
                        (Mixer::M2(m), Layers::M2(s)) => {
                            m.prefill(s.get_mut(li).ok_or_else(wrong)?, &normed)
                        }
                        _ => return Err(wrong()),
                    }
                    .map_err(to_model_err)?;
                    xs.iter_mut().zip(&out).for_each(|(x, o)| *x += o);
                }
            }
            (Stack::R6(ls), Layers::R6(s)) => {
                if let Some((w, b)) = &self.embed_norm {
                    let mut tmp = vec![0.0; h];
                    for x in xs.chunks_exact_mut(h) {
                        layer_norm_into(x, w, b, 1e-5, &mut tmp);
                        x.copy_from_slice(&tmp);
                    }
                }
                for (l, s) in ls.iter().zip(s.iter_mut()) {
                    xs = l.prefill(s, &xs).map_err(to_model_err)?;
                }
            }
            (Stack::R7(ls), Layers::R7(s)) => {
                let mut v_first = vec![0.0; t * h];
                for (l, s) in ls.iter().zip(s.iter_mut()) {
                    xs = l.prefill(s, &xs, &mut v_first).map_err(to_model_err)?;
                }
            }
            _ => return Err(wrong()),
        }
        st.pos += t;
        let last = &xs[(t - 1) * h..];
        let mut nx = vec![0.0; h];
        match &self.final_norm {
            FinalNorm::Rms(w) => rmsnorm(last, w, eps, &mut nx),
            FinalNorm::Layer(w, b) => layer_norm_into(last, w, b, 1e-5, &mut nx),
        }
        let mut logits = vec![0.0; self.vocab];
        matvec(
            self.head.as_deref().unwrap_or(&self.embed),
            h,
            &nx,
            &mut logits,
        );
        Ok(logits)
    }
}

// ------------------------------------------------------------------ the common model interface

/// What [`generate`] needs from a model, whatever its architecture.
pub trait LanguageModel {
    type State;
    fn new_state(&self) -> Self::State;
    fn forward(&self, st: &mut Self::State, token: usize) -> Result<Vec<f32>, ModelError> {
        self.forward_batch(st, &[token])
    }
    /// Process `tokens` and return the logits of the last one.
    fn forward_batch(&self, st: &mut Self::State, tokens: &[usize])
    -> Result<Vec<f32>, ModelError>;
    fn vocab_size(&self) -> usize;
    /// Longest sequence the model should be asked for (recurrent models have no hard limit).
    fn context(&self) -> usize;
    /// Tokens already processed in `st`.
    fn position(&self, st: &Self::State) -> usize;
}

impl LanguageModel for SsmModel {
    type State = SsmState;
    fn new_state(&self) -> SsmState {
        SsmModel::new_state(self)
    }
    fn forward_batch(&self, st: &mut SsmState, tokens: &[usize]) -> Result<Vec<f32>, ModelError> {
        SsmModel::forward_batch(self, st, tokens)
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn context(&self) -> usize {
        self.cfg.context
    }
    fn position(&self, st: &SsmState) -> usize {
        st.pos
    }
}

impl LanguageModel for crate::gemma4::Gemma4Model {
    type State = crate::gemma4::Gemma4State;
    fn new_state(&self) -> Self::State {
        crate::gemma4::Gemma4Model::new_state(self)
    }
    fn forward_batch(
        &self,
        st: &mut Self::State,
        tokens: &[usize],
    ) -> Result<Vec<f32>, ModelError> {
        crate::gemma4::Gemma4Model::forward_batch(self, st, tokens)
    }
    fn vocab_size(&self) -> usize {
        self.vocab()
    }
    fn context(&self) -> usize {
        self.config().context
    }
    fn position(&self, st: &Self::State) -> usize {
        st.pos
    }
}

impl LanguageModel for Model {
    type State = State;
    fn new_state(&self) -> State {
        Model::new_state(self)
    }
    fn forward(&self, st: &mut State, token: usize) -> Result<Vec<f32>, ModelError> {
        Model::forward(self, st, token)
    }
    fn forward_batch(&self, st: &mut State, tokens: &[usize]) -> Result<Vec<f32>, ModelError> {
        Model::forward_batch(self, st, tokens)
    }
    fn vocab_size(&self) -> usize {
        self.spec.vocab as usize
    }
    fn context(&self) -> usize {
        self.spec.context as usize
    }
    fn position(&self, st: &State) -> usize {
        st.pos
    }
}

/// [`Model::generate`] for any [`LanguageModel`]: prefill `prompt`, then sample until a stop token,
/// `max_tokens`, cancellation, a `false` from `on_token` or a full context.
pub fn generate<M: LanguageModel + ?Sized>(
    m: &M,
    st: &mut M::State,
    prompt: &[usize],
    sampler: &mut Sampler,
    opts: &GenerateOptions,
    on_token: &mut dyn FnMut(usize) -> bool,
) -> Result<GenerateStats, ModelError> {
    let mut stats = GenerateStats {
        prompt_tokens: prompt.len(),
        generated: 0,
        prefill_secs: 0.0,
        decode_secs: 0.0,
        stop: StopReason::MaxTokens,
    };
    let t0 = Instant::now();
    let mut logits = m.forward_batch(st, prompt)?;
    stats.prefill_secs = t0.elapsed().as_secs_f64();
    sampler.observe_all(prompt);
    let t1 = Instant::now();
    loop {
        if stats.generated >= opts.max_tokens {
            stats.stop = StopReason::MaxTokens;
            break;
        }
        if opts.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            stats.stop = StopReason::Cancelled;
            break;
        }
        let tok = sampler.sample(&logits);
        sampler.observe(tok);
        if opts.stop_tokens.contains(&tok) {
            stats.stop = StopReason::StopToken;
            break;
        }
        stats.generated += 1;
        if !on_token(tok) {
            stats.stop = StopReason::Callback;
            break;
        }
        if stats.generated >= opts.max_tokens {
            stats.stop = StopReason::MaxTokens;
            break;
        }
        if m.position(st) >= m.context() {
            stats.stop = StopReason::ContextFull;
            break;
        }
        logits = m.forward(st, tok)?;
    }
    stats.decode_secs = t1.elapsed().as_secs_f64();
    Ok(stats)
}

// ---------------------------------------------------------------------------------- AnyModel

#[allow(clippy::large_enum_variant)]
pub enum AnyModel {
    Dense(Model),
    Ssm(SsmModel),
    Hybrid(crate::hybrid::HybridModel),
    Gemma4(Box<crate::gemma4::Gemma4Model>),
    Legacy(Box<crate::legacy::LegacyModel>),
    GptOss(Box<crate::gptoss::GptOssModel>),
}
#[allow(clippy::large_enum_variant)]
pub enum AnyState {
    Dense(State),
    Ssm(SsmState),
    Hybrid(crate::hybrid::HybridState),
    Gemma4(crate::gemma4::Gemma4State),
    Legacy(crate::legacy::LegacyState),
    GptOss(crate::gptoss::GptOssState),
}

impl AnyModel {
    /// The recurrent family, or `None` for a dense / mixture-of-experts transformer.
    pub fn family(&self) -> Option<FamilyKind> {
        if let AnyModel::Ssm(m) = self {
            Some(m.kind())
        } else {
            None
        }
    }
}
impl AnyState {
    /// Bytes held by the per-conversation state (the KV cache for transformers).
    pub fn bytes(&self) -> usize {
        match self {
            AnyState::Dense(s) => s.kv_bytes(),
            AnyState::Ssm(s) => s.bytes(),
            AnyState::Hybrid(s) => s.bytes(),
            AnyState::Gemma4(s) => s.bytes(),
            AnyState::Legacy(s) => s.bytes(),
            AnyState::GptOss(s) => s.bytes(),
        }
    }
}
impl LanguageModel for AnyModel {
    type State = AnyState;
    fn new_state(&self) -> AnyState {
        match self {
            AnyModel::Dense(m) => AnyState::Dense(m.new_state()),
            AnyModel::Ssm(m) => AnyState::Ssm(m.new_state()),
            AnyModel::Hybrid(m) => AnyState::Hybrid(m.new_state()),
            AnyModel::Gemma4(m) => AnyState::Gemma4(m.new_state()),
            AnyModel::Legacy(m) => AnyState::Legacy(m.new_state()),
            AnyModel::GptOss(m) => AnyState::GptOss(m.new_state()),
        }
    }
    fn forward_batch(&self, st: &mut AnyState, tokens: &[usize]) -> Result<Vec<f32>, ModelError> {
        match (self, st) {
            (AnyModel::Dense(m), AnyState::Dense(s)) => m.forward_batch(s, tokens),
            (AnyModel::Ssm(m), AnyState::Ssm(s)) => m.forward_batch(s, tokens),
            (AnyModel::Hybrid(m), AnyState::Hybrid(s)) => m.forward_batch(s, tokens),
            (AnyModel::Gemma4(m), AnyState::Gemma4(s)) => m.forward_batch(s, tokens),
            (AnyModel::Legacy(m), AnyState::Legacy(s)) => m.forward_batch(s, tokens),
            (AnyModel::GptOss(m), AnyState::GptOss(s)) => m.forward_batch(s, tokens),
            _ => Err(ModelError::Shape(
                "the state belongs to a different model".into(),
            )),
        }
    }
    fn vocab_size(&self) -> usize {
        match self {
            AnyModel::Dense(m) => LanguageModel::vocab_size(m),
            AnyModel::Ssm(m) => m.vocab,
            AnyModel::Hybrid(m) => m.vocab(),
            AnyModel::Gemma4(m) => m.vocab(),
            AnyModel::Legacy(m) => m.vocab(),
            AnyModel::GptOss(m) => m.vocab(),
        }
    }
    fn context(&self) -> usize {
        match self {
            AnyModel::Dense(m) => LanguageModel::context(m),
            AnyModel::Ssm(m) => m.cfg.context,
            AnyModel::Hybrid(m) => LanguageModel::context(m),
            AnyModel::Gemma4(m) => m.config().context,
            AnyModel::Legacy(m) => m.config().context,
            AnyModel::GptOss(m) => m.config().context,
        }
    }
    fn position(&self, st: &AnyState) -> usize {
        match st {
            AnyState::Dense(s) => s.pos,
            AnyState::Ssm(s) => s.pos,
            AnyState::Hybrid(s) => s.pos,
            AnyState::Gemma4(s) => s.pos,
            AnyState::Legacy(s) => s.pos,
            AnyState::GptOss(s) => s.pos,
        }
    }
}

/// Work out the configuration of a native (BlinkDL) RWKV-6 or RWKV-7 checkpoint that has no
/// `config.json`, from the tensor shapes alone.
pub fn infer_native_rwkv(src: &dyn TensorSource) -> Result<FamilyConfig, FamilyError> {
    let infos = src.infos();
    let dims = |name: &str| {
        infos
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.dims.clone())
    };
    let kind = if src.has("blocks.0.att.x_r") {
        FamilyKind::Rwkv7
    } else if src.has("blocks.0.att.time_maa_x") {
        FamilyKind::Rwkv6
    } else {
        return Err(FamilyError::Config(
            "no config.json, and the tensors are not a native RWKV-6/7 checkpoint".into(),
        ));
    };
    let d1 = |name: &str| -> Result<usize, FamilyError> {
        dims(name)
            .and_then(|d| d.first().copied())
            .map(|v| v as usize)
            .ok_or_else(|| FamilyError::Load(format!("the model has no tensor `{name}`")))
    };
    let hidden = d1("blocks.0.ln1.weight")?;
    let ffn = d1("blocks.0.att.key.weight").and(d1("blocks.0.ffn.key.weight"))?;
    let layers = (0..MAX_LAYERS)
        .take_while(|i| src.has(&format!("blocks.{i}.ln1.weight")))
        .count();
    let head_size = match dims(if kind == FamilyKind::Rwkv7 {
        "blocks.0.att.r_k"
    } else {
        "blocks.0.att.time_faaaa"
    }) {
        Some(d) if d.len() == 2 && d[1] > 0 => d[1] as usize,
        _ => 64,
    };
    let mixer = if kind == FamilyKind::Rwkv7 {
        MixerConfig::R7(Rwkv7Config {
            hidden,
            head_size,
            ffn,
            ln_eps: 1e-5,
            decay_rank: 0,
            a_rank: 0,
            v_rank: 0,
            gate_rank: 0,
            value_residual: true,
        })
    } else {
        MixerConfig::R6(Rwkv6Config {
            hidden,
            head_size,
            ffn,
            ln_eps: 1e-5,
        })
    };
    let cfg = FamilyConfig {
        kind,
        layers,
        vocab: 0,
        context: DEFAULT_CONTEXT,
        norm_eps: 1e-5,
        mixer,
    };
    cfg.validate()?;
    Ok(cfg)
}

fn model_type_of(json: &str) -> Result<(String, Value), FamilyError> {
    let v: Value =
        serde_json::from_str(json).map_err(|e| FamilyError::Config(format!("config.json: {e}")))?;
    if !v.is_object() {
        return cfg_err("config.json is not a JSON object");
    }
    let root = v.get("text_config").filter(|t| t.is_object()).unwrap_or(&v);
    let mt = root
        .get("model_type")
        .or_else(|| v.get("model_type"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    Ok((mt, v))
}

fn load_safetensors(st: SafeTensors, config: Option<String>) -> Result<AnyModel, FamilyError> {
    let Some(json) = config else {
        let cfg = infer_native_rwkv(&st)?;
        return Ok(AnyModel::Ssm(SsmModel::load(cfg, &st, SsmNaming::Native)?));
    };
    let (mt, _) = model_type_of(&json)?;
    if classify_family(&mt).is_some() || FamilyConfig::from_hf_config(&json).is_ok() {
        let cfg = FamilyConfig::from_hf_config(&json)?;
        let naming = detect_naming(&st).ok_or_else(|| {
            FamilyError::Load("none of the usual embedding tensors is in the file".into())
        })?;
        return Ok(AnyModel::Ssm(SsmModel::load(cfg, &st, naming)?));
    }
    match arch::classify(&mt) {
        arch::Support::Hybrid => {
            let cfg = crate::hybrid::HybridConfig::from_hf_config(&json)?;
            return Ok(AnyModel::Hybrid(crate::hybrid::HybridModel::load(
                cfg,
                &st,
                LoadOptions::default(),
            )?));
        }
        arch::Support::Native(_) => {
            return Err(FamilyError::Unsupported(format!(
                "`{mt}` is only read from GGUF files here; use the .gguf of the model"
            )));
        }
        arch::Support::Partial(why) => {
            return Err(FamilyError::Unsupported(format!("`{mt}`: {why}")));
        }
        arch::Support::Planned(why)
            if why.contains("state-space")
                || why.contains("recurrent")
                || why.contains("hybrid")
                || why.contains("RWKV") =>
        {
            return Err(FamilyError::Unsupported(format!("`{mt}`: {why}")));
        }
        _ => {}
    }
    let spec = arch::from_hf_config(&json)?;
    Ok(AnyModel::Dense(load_model(
        spec,
        Arc::new(st),
        Naming::Hf,
        false,
        LoadOptions::default(),
        MoeOptions::default(),
    )?))
}

/// Open a model folder (with `config.json` and `.safetensors` files), a single `.safetensors` file
/// (with a `config.json` next to it, or a native RWKV-6/7 checkpoint without one) or a `.gguf` file,
/// and return the right kind of model.
pub fn load_any(path: &Path) -> Result<AnyModel, FamilyError> {
    let is_gguf = path.is_file()
        && path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    if is_gguf {
        let f = GgufFile::open(path)?;
        let arch_name = f
            .gguf
            .architecture()
            .ok_or_else(|| FamilyError::Config("the GGUF file has no general.architecture".into()))?
            .to_string();
        if classify_family(&arch_name).is_some() {
            let cfg = FamilyConfig::from_gguf(&f.gguf)?;
            return Ok(AnyModel::Ssm(SsmModel::load(cfg, &f, SsmNaming::Gguf)?));
        }
        if crate::gptoss::handles(&arch_name) {
            let cfg = crate::gptoss::GptOssConfig::from_gguf(&f.gguf)
                .map_err(|e| FamilyError::Config(e.to_string()))?;
            let src: Arc<dyn crate::loader::WeightFile> = Arc::new(f);
            let opts = LoadOptions {
                prefer_quantized: true,
                mmap: true,
                ..LoadOptions::default()
            };
            let m = crate::gptoss::GptOssModel::load(cfg, &src, opts)
                .map_err(|e| FamilyError::Load(e.to_string()))?;
            return Ok(AnyModel::GptOss(Box::new(m)));
        }
        if crate::legacy::handles(&arch_name) {
            let cfg = crate::legacy::LegacyConfig::from_gguf(&f.gguf)
                .map_err(|e| FamilyError::Config(e.to_string()))?;
            let src: Arc<dyn crate::loader::WeightFile> = Arc::new(f);
            let opts = LoadOptions {
                prefer_quantized: true,
                mmap: true,
                ..LoadOptions::default()
            };
            let m = crate::legacy::LegacyModel::load(cfg, &src, opts)
                .map_err(|e| FamilyError::Load(e.to_string()))?;
            return Ok(AnyModel::Legacy(Box::new(m)));
        }
        if matches!(arch::classify(&arch_name), arch::Support::Native(_)) {
            let cfg = crate::gemma4::G4Config::from_gguf(&f.gguf)
                .map_err(|e| FamilyError::Config(e.to_string()))?;
            let src: Arc<dyn crate::loader::WeightFile> = Arc::new(f);
            let opts = LoadOptions {
                prefer_quantized: true,
                mmap: true,
                ..LoadOptions::default()
            };
            let m = crate::gemma4::Gemma4Model::load(cfg, &src, opts)
                .map_err(|e| FamilyError::Load(e.to_string()))?;
            return Ok(AnyModel::Gemma4(Box::new(m)));
        }
        let spec = arch::from_gguf(&f.gguf)?;
        let permuted = arch_name == "llama";
        return Ok(AnyModel::Dense(load_model(
            spec,
            Arc::new(f),
            Naming::Gguf,
            permuted,
            LoadOptions::default(),
            MoeOptions::default(),
        )?));
    }
    let dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    let cfg_path = dir.join("config.json");
    let config = if cfg_path.is_file() {
        let meta = std::fs::metadata(&cfg_path)?;
        if meta.len() > (1 << 24) {
            return cfg_err("config.json is implausibly large");
        }
        Some(
            String::from_utf8(std::fs::read(&cfg_path)?)
                .map_err(|_| FamilyError::Config("config.json is not valid UTF-8".into()))?,
        )
    } else {
        None
    };
    if !path.is_dir() && !path.is_file() {
        return load_err(format!("{} does not exist", path.display()));
    }
    let st = SafeTensors::open(path)?;
    load_safetensors(st, config)
}

// ------------------------------------------------------------------------------------- tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StopReason;
    use crate::model::tests::tiny;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering as O};

    fn tmpdir() -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "engine-family-{}-{}",
            std::process::id(),
            N.fetch_add(1, O::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    type T = (String, Vec<usize>, Vec<f32>);
    struct Gen {
        s: u64,
        out: Vec<T>,
    }
    impl Gen {
        fn new(seed: u64) -> Gen {
            Gen {
                s: seed | 1,
                out: vec![],
            }
        }
        fn r(&mut self) -> f32 {
            self.s ^= self.s << 13;
            self.s ^= self.s >> 7;
            self.s ^= self.s << 17;
            (self.s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }
        fn rnd(&mut self, name: &str, shape: &[usize], scale: f32) {
            let n = shape.iter().product();
            let v = (0..n).map(|_| self.r() * scale).collect();
            self.out.push((name.into(), shape.to_vec(), v));
        }
        fn one(&mut self, name: &str, shape: &[usize]) {
            let n = shape.iter().product();
            let v = (0..n).map(|_| 1.0 + self.r() * 0.2).collect();
            self.out.push((name.into(), shape.to_vec(), v));
        }
        fn alog(&mut self, name: &str, shape: &[usize]) {
            let n: usize = shape.iter().product();
            self.out.push((
                name.into(),
                shape.to_vec(),
                (0..n).map(|i| ((1 + i % 4) as f32).ln()).collect(),
            ));
        }
        fn decay(&mut self, name: &str, shape: &[usize]) {
            let n: usize = shape.iter().product();
            let v = (0..n).map(|_| -1.0 + self.r()).collect();
            self.out.push((name.into(), shape.to_vec(), v));
        }
    }

    fn write_st(path: &Path, ts: &[T]) {
        let mut hdr = serde_json::Map::new();
        let mut off = 0;
        let mut body = vec![];
        for (n, shape, d) in ts {
            let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
            hdr.insert(n.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [off, off + b.len()]}));
            off += b.len();
            body.extend(b);
        }
        let h = serde_json::to_vec(&hdr).unwrap();
        let mut f = (h.len() as u64).to_le_bytes().to_vec();
        f.extend(h);
        f.extend(body);
        std::fs::write(path, f).unwrap();
    }

    const V: usize = 24;
    const D: usize = 16;

    fn mamba1(g: &mut Gen) {
        let (di, n, k, r, layers) = (32usize, 4usize, 3usize, 2usize, 2usize);
        g.rnd("backbone.embeddings.weight", &[V, D], 0.6);
        g.one("backbone.norm_f.weight", &[D]);
        for i in 0..layers {
            let p = format!("backbone.layers.{i}.");
            g.one(&format!("{p}norm.weight"), &[D]);
            g.rnd(&format!("{p}mixer.in_proj.weight"), &[2 * di, D], 0.5);
            g.rnd(&format!("{p}mixer.conv1d.weight"), &[di, 1, k], 0.6);
            g.rnd(&format!("{p}mixer.conv1d.bias"), &[di], 0.2);
            g.rnd(&format!("{p}mixer.x_proj.weight"), &[r + 2 * n, di], 0.4);
            g.rnd(&format!("{p}mixer.dt_proj.weight"), &[di, r], 0.6);
            g.rnd(&format!("{p}mixer.dt_proj.bias"), &[di], 0.4);
            g.alog(&format!("{p}mixer.A_log"), &[di, n]);
            g.one(&format!("{p}mixer.D"), &[di]);
            g.rnd(&format!("{p}mixer.out_proj.weight"), &[D, di], 0.4);
        }
    }
    const MAMBA1_CFG: &str = r#"{"model_type":"mamba","hidden_size":16,"num_hidden_layers":2,"vocab_size":24,"state_size":4,"conv_kernel":3,"expand":2,"time_step_rank":2,"layer_norm_epsilon":1e-5,"tie_word_embeddings":true}"#;

    fn mamba2(g: &mut Gen) {
        let (nh, hd, ng, n, k, layers) = (4usize, 8usize, 2usize, 4usize, 3usize, 2usize);
        let di = nh * hd;
        let conv = di + 2 * ng * n;
        g.rnd("backbone.embeddings.weight", &[V, D], 0.6);
        g.one("backbone.norm_f.weight", &[D]);
        g.rnd("lm_head.weight", &[V, D], 0.5);
        for i in 0..layers {
            let p = format!("backbone.layers.{i}.");
            g.one(&format!("{p}norm.weight"), &[D]);
            g.rnd(
                &format!("{p}mixer.in_proj.weight"),
                &[di + conv + nh, D],
                0.5,
            );
            g.rnd(&format!("{p}mixer.conv1d.weight"), &[conv, 1, k], 0.6);
            g.rnd(&format!("{p}mixer.conv1d.bias"), &[conv], 0.2);
            g.rnd(&format!("{p}mixer.dt_bias"), &[nh], 0.4);
            g.alog(&format!("{p}mixer.A_log"), &[nh]);
            g.one(&format!("{p}mixer.D"), &[nh]);
            g.one(&format!("{p}mixer.norm.weight"), &[di]);
            g.rnd(&format!("{p}mixer.out_proj.weight"), &[D, di], 0.4);
        }
    }
    const MAMBA2_CFG: &str = r#"{"model_type":"mamba2","hidden_size":16,"num_heads":4,"head_dim":8,"state_size":4,"n_groups":2,"conv_kernel":3,"chunk_size":4,"num_hidden_layers":2,"vocab_size":24,"layer_norm_epsilon":1e-5}"#;

    fn rwkv6(g: &mut Gen) {
        let (hs, ffn, e1, e2, layers) = (8usize, 32usize, 4usize, 4usize, 2usize);
        let nh = D / hs;
        g.rnd("emb.weight", &[V, D], 0.6);
        g.rnd("head.weight", &[V, D], 0.5);
        g.one("ln_out.weight", &[D]);
        g.rnd("ln_out.bias", &[D], 0.1);
        g.one("blocks.0.ln0.weight", &[D]);
        g.rnd("blocks.0.ln0.bias", &[D], 0.1);
        for i in 0..layers {
            let p = format!("blocks.{i}.");
            for ln in ["ln1", "ln2"] {
                g.one(&format!("{p}{ln}.weight"), &[D]);
                g.rnd(&format!("{p}{ln}.bias"), &[D], 0.1);
            }
            g.rnd(&format!("{p}att.time_maa_x"), &[D], 1.0);
            for m in ["w", "k", "v", "r", "g"] {
                g.rnd(&format!("{p}att.time_maa_{m}"), &[D], 1.0);
            }
            g.rnd(&format!("{p}att.time_maa_w1"), &[D, 5 * e1], 0.4);
            g.rnd(&format!("{p}att.time_maa_w2"), &[5, e1, D], 0.4);
            g.decay(&format!("{p}att.time_decay"), &[D]);
            g.rnd(&format!("{p}att.time_decay_w1"), &[D, e2], 0.4);
            g.rnd(&format!("{p}att.time_decay_w2"), &[e2, D], 0.4);
            g.rnd(&format!("{p}att.time_faaaa"), &[nh, hs], 0.5);
            for w in ["receptance", "key", "value", "gate", "output"] {
                g.rnd(&format!("{p}att.{w}.weight"), &[D, D], 0.4);
            }
            g.one(&format!("{p}att.ln_x.weight"), &[D]);
            g.rnd(&format!("{p}att.ln_x.bias"), &[D], 0.1);
            g.rnd(&format!("{p}ffn.time_maa_k"), &[D], 1.0);
            g.rnd(&format!("{p}ffn.time_maa_r"), &[D], 1.0);
            g.rnd(&format!("{p}ffn.key.weight"), &[ffn, D], 0.4);
            g.rnd(&format!("{p}ffn.receptance.weight"), &[D, D], 0.4);
            g.rnd(&format!("{p}ffn.value.weight"), &[D, ffn], 0.4);
        }
    }
    const RWKV6_CFG: &str = r#"{"model_type":"rwkv6","hidden_size":16,"head_size":8,"intermediate_size":32,"num_hidden_layers":2,"vocab_size":24}"#;

    fn rwkv7_native(g: &mut Gen) {
        let (hs, ffn, rk, layers) = (8usize, 32usize, 4usize, 2usize);
        let nh = D / hs;
        g.rnd("emb.weight", &[V, D], 0.6);
        g.rnd("head.weight", &[V, D], 0.5);
        g.one("ln_out.weight", &[D]);
        g.rnd("ln_out.bias", &[D], 0.1);
        g.one("blocks.0.ln0.weight", &[D]);
        g.rnd("blocks.0.ln0.bias", &[D], 0.1);
        for i in 0..layers {
            let p = format!("blocks.{i}.");
            for ln in ["ln1", "ln2"] {
                g.one(&format!("{p}{ln}.weight"), &[D]);
                g.rnd(&format!("{p}{ln}.bias"), &[D], 0.1);
            }
            for m in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"] {
                g.rnd(&format!("{p}att.{m}"), &[D], 1.0);
            }
            g.rnd(&format!("{p}att.w0"), &[D], 1.0);
            g.rnd(&format!("{p}att.w1"), &[D, rk], 0.4);
            g.rnd(&format!("{p}att.w2"), &[rk, D], 0.4);
            g.rnd(&format!("{p}att.a0"), &[D], 1.0);
            g.rnd(&format!("{p}att.a1"), &[D, rk], 0.4);
            g.rnd(&format!("{p}att.a2"), &[rk, D], 0.4);
            if i > 0 {
                g.rnd(&format!("{p}att.v0"), &[D], 1.0);
                g.rnd(&format!("{p}att.v1"), &[D, rk], 0.4);
                g.rnd(&format!("{p}att.v2"), &[rk, D], 0.4);
            }
            g.rnd(&format!("{p}att.g1"), &[D, rk], 0.4);
            g.rnd(&format!("{p}att.g2"), &[rk, D], 0.4);
            g.one(&format!("{p}att.k_k"), &[D]);
            g.rnd(&format!("{p}att.k_a"), &[D], 1.0);
            g.rnd(&format!("{p}att.r_k"), &[nh, hs], 0.5);
            for w in ["receptance", "key", "value", "output"] {
                g.rnd(&format!("{p}att.{w}.weight"), &[D, D], 0.4);
            }
            g.one(&format!("{p}att.ln_x.weight"), &[D]);
            g.rnd(&format!("{p}att.ln_x.bias"), &[D], 0.1);
            g.rnd(&format!("{p}ffn.x_k"), &[D], 1.0);
            g.rnd(&format!("{p}ffn.key.weight"), &[ffn, D], 0.4);
            g.rnd(&format!("{p}ffn.value.weight"), &[D, ffn], 0.4);
        }
    }

    fn rwkv7_fla(g: &mut Gen) {
        let (hs, ffn, rk, layers) = (8usize, 32usize, 4usize, 2usize);
        let nh = D / hs;
        g.rnd("model.embeddings.weight", &[V, D], 0.6);
        g.rnd("lm_head.weight", &[V, D], 0.5);
        g.one("model.norm.weight", &[D]);
        g.rnd("model.norm.bias", &[D], 0.1);
        g.one("model.layers.0.pre_norm.weight", &[D]);
        g.rnd("model.layers.0.pre_norm.bias", &[D], 0.1);
        for i in 0..layers {
            let p = format!("model.layers.{i}.");
            for ln in ["attn_norm", "ffn_norm"] {
                g.one(&format!("{p}{ln}.weight"), &[D]);
                g.rnd(&format!("{p}{ln}.bias"), &[D], 0.1);
            }
            for m in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"] {
                g.rnd(&format!("{p}attn.{m}"), &[D], 1.0);
            }
            let lora = |g: &mut Gen, nm: &str, bias: bool| {
                g.rnd(&format!("{p}attn.{nm}.lora.0.weight"), &[rk, D], 0.4);
                g.rnd(&format!("{p}attn.{nm}.lora.2.weight"), &[D, rk], 0.4);
                if bias {
                    g.rnd(&format!("{p}attn.{nm}.lora.2.bias"), &[D], 1.0);
                }
            };
            lora(g, "w_lora", true);
            lora(g, "a_lora", true);
            lora(g, "g_lora", false);
            if i > 0 {
                lora(g, "v_lora", true);
            }
            g.one(&format!("{p}attn.k_k"), &[D]);
            g.rnd(&format!("{p}attn.k_a"), &[D], 1.0);
            g.rnd(&format!("{p}attn.r_k"), &[nh, hs], 0.5);
            for w in ["r_proj", "k_proj", "v_proj", "o_proj"] {
                g.rnd(&format!("{p}attn.{w}.weight"), &[D, D], 0.4);
            }
            g.one(&format!("{p}attn.g_norm.weight"), &[D]);
            g.rnd(&format!("{p}attn.g_norm.bias"), &[D], 0.1);
            g.rnd(&format!("{p}ffn.x_k"), &[D], 1.0);
            g.rnd(&format!("{p}ffn.key.weight"), &[ffn, D], 0.4);
            g.rnd(&format!("{p}ffn.value.weight"), &[D, ffn], 0.4);
        }
    }
    const RWKV7_CFG: &str = r#"{"model_type":"rwkv7","hidden_size":16,"head_dim":8,"intermediate_size":32,"num_hidden_layers":2,"vocab_size":24,
        "decay_low_rank_dim":4,"a_low_rank_dim":4,"v_low_rank_dim":4,"gate_low_rank_dim":4,"value_residual":true}"#;

    fn make(kind: &str, seed: u64) -> (PathBuf, FamilyKind) {
        let d = tmpdir();
        let mut g = Gen::new(seed);
        let (cfg, k) = match kind {
            "mamba1" => {
                mamba1(&mut g);
                (Some(MAMBA1_CFG), FamilyKind::Mamba1)
            }
            "mamba2" => {
                mamba2(&mut g);
                (Some(MAMBA2_CFG), FamilyKind::Mamba2)
            }
            "rwkv6" => {
                rwkv6(&mut g);
                (Some(RWKV6_CFG), FamilyKind::Rwkv6)
            }
            "rwkv7-native" => {
                rwkv7_native(&mut g);
                (None, FamilyKind::Rwkv7)
            }
            "rwkv7-fla" => {
                rwkv7_fla(&mut g);
                (Some(RWKV7_CFG), FamilyKind::Rwkv7)
            }
            _ => unreachable!(),
        };
        write_st(&d.join("model.safetensors"), &g.out);
        if let Some(c) = cfg {
            std::fs::write(d.join("config.json"), c).unwrap();
        }
        (d, k)
    }

    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                x.is_finite() && y.is_finite(),
                "logit {i} not finite: {x} {y}"
            );
            assert!(
                (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())),
                "logit {i}: {x} vs {y}"
            );
        }
    }

    fn state_bytes(s: &AnyState) -> usize {
        s.bytes()
    }

    fn exercise(path: &Path, want: FamilyKind) {
        let m = load_any(path).unwrap();
        assert_eq!(m.family(), Some(want));
        assert_eq!(m.vocab_size(), V);
        let toks = [3usize, 7, 1, 19, 0, 7, 12, 5, 23, 2];
        // one token at a time
        let mut a = m.new_state();
        let mut last = vec![];
        let mut sizes = vec![];
        for &t in &toks {
            last = m.forward(&mut a, t).unwrap();
            sizes.push(state_bytes(&a));
        }
        assert!(
            sizes.iter().all(|&s| s == sizes[0] && s > 0),
            "state size changed: {sizes:?}"
        );
        assert_eq!(m.position(&a), toks.len());
        // the same tokens in one batch
        let mut b = m.new_state();
        close(&last, &m.forward_batch(&mut b, &toks).unwrap(), 2e-3);
        // a batch that continues from a saved state
        let mut c = m.new_state();
        m.forward_batch(&mut c, &toks[..3]).unwrap();
        close(&last, &m.forward_batch(&mut c, &toks[3..]).unwrap(), 2e-3);
        // mixed: batch then single steps
        let mut d = m.new_state();
        m.forward_batch(&mut d, &toks[..6]).unwrap();
        let mut l = vec![];
        for &t in &toks[6..] {
            l = m.forward(&mut d, t).unwrap();
        }
        close(&last, &l, 2e-3);
        // state does not grow over a long run
        let before = state_bytes(&a);
        for i in 0..200 {
            m.forward(&mut a, i % V).unwrap();
        }
        assert_eq!(state_bytes(&a), before);
        // greedy generation is deterministic and matches a hand-rolled loop
        let run = || {
            let mut st = m.new_state();
            let mut out = vec![];
            let opts = GenerateOptions {
                max_tokens: 12,
                stop_tokens: vec![],
                cancel: None,
            };
            let stats = generate(
                &m,
                &mut st,
                &[3, 7, 1],
                &mut Sampler::new(0.0, 0, 1),
                &opts,
                &mut |t| {
                    out.push(t);
                    true
                },
            )
            .unwrap();
            assert_eq!(stats.generated, 12);
            assert_eq!(stats.stop, StopReason::MaxTokens);
            out
        };
        let first = run();
        assert_eq!(first, run());
        let mut st = m.new_state();
        let mut lg = m.forward_batch(&mut st, &[3, 7, 1]).unwrap();
        let mut hand = vec![];
        for _ in 0..12 {
            let t = Sampler::new(0.0, 0, 1).sample(&lg);
            hand.push(t);
            lg = m.forward(&mut st, t).unwrap();
        }
        assert_eq!(first, hand);
    }

    #[test]
    fn mamba1_loads_and_agrees_with_itself() {
        let (d, k) = make("mamba1", 11);
        exercise(&d, k);
        std::fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn mamba2_loads_and_agrees_with_itself() {
        let (d, k) = make("mamba2", 12);
        exercise(&d, k);
        std::fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn rwkv6_native_names_with_a_config() {
        let (d, k) = make("rwkv6", 13);
        exercise(&d, k);
        std::fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn rwkv7_native_names_without_a_config_are_inferred_from_shapes() {
        let (d, k) = make("rwkv7-native", 14);
        assert!(!d.join("config.json").exists());
        exercise(&d.join("model.safetensors"), k);
        exercise(&d, k);
        let AnyModel::Ssm(m) = load_any(&d).unwrap() else {
            panic!()
        };
        let MixerConfig::R7(c) = &m.config().mixer else {
            panic!()
        };
        assert_eq!(
            (c.hidden, c.head_size, c.ffn, m.config().layers),
            (16, 8, 32, 2)
        );
        std::fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn rwkv7_fla_names_with_a_config() {
        let (d, k) = make("rwkv7-fla", 15);
        exercise(&d, k);
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn rwkv6_without_a_config_is_inferred_too() {
        let (d, _) = make("rwkv6", 16);
        std::fs::remove_file(d.join("config.json")).unwrap();
        exercise(&d, FamilyKind::Rwkv6);
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn different_weights_give_different_models_and_the_same_weights_the_same() {
        let (a, _) = make("mamba2", 1);
        let (b, _) = make("mamba2", 2);
        let (ma, mb) = (load_any(&a).unwrap(), load_any(&b).unwrap());
        let (mut sa, mut sb, mut sa2) = (ma.new_state(), mb.new_state(), ma.new_state());
        let la = ma.forward(&mut sa, 4).unwrap();
        assert_ne!(la, mb.forward(&mut sb, 4).unwrap());
        assert_eq!(la, ma.forward(&mut sa2, 4).unwrap());
        std::fs::remove_dir_all(a).unwrap();
        std::fs::remove_dir_all(b).unwrap();
    }

    #[test]
    fn dense_models_go_through_the_same_interface() {
        let m = tiny(4, 2, false);
        let toks = [3usize, 7, 1, 19];
        let mut a = LanguageModel::new_state(&m);
        let mut b = m.new_state();
        assert_eq!(
            LanguageModel::forward_batch(&m, &mut a, &toks).unwrap(),
            m.forward_batch(&mut b, &toks).unwrap()
        );
        assert_eq!(
            (
                LanguageModel::vocab_size(&m),
                LanguageModel::context(&m),
                LanguageModel::position(&m, &a)
            ),
            (20, 32, 4)
        );
        // the generic generate equals Model::generate
        let opts = GenerateOptions {
            max_tokens: 6,
            ..Default::default()
        };
        let (mut o1, mut o2) = (vec![], vec![]);
        generate(
            &m,
            &mut LanguageModel::new_state(&m),
            &[2, 5],
            &mut Sampler::new(0.0, 0, 7),
            &opts,
            &mut |t| {
                o1.push(t);
                true
            },
        )
        .unwrap();
        m.generate(
            &mut m.new_state(),
            &[2, 5],
            &mut Sampler::new(0.0, 0, 7),
            &opts,
            &mut |t| {
                o2.push(t);
                true
            },
        )
        .unwrap();
        assert_eq!(o1, o2);
        // and an AnyModel wrapper behaves like the model inside it
        let any = AnyModel::Dense(tiny(4, 2, false));
        let mut st = any.new_state();
        assert_eq!(
            any.forward_batch(&mut st, &toks).unwrap(),
            m.forward_batch(&mut m.new_state(), &toks).unwrap()
        );
        assert!(any.family().is_none() && st.bytes() > 0);
        // a state from another model is refused
        let (d, _) = make("mamba1", 3);
        let ssm = load_any(&d).unwrap();
        assert!(ssm.forward(&mut st, 1).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_dense_folder_is_loaded_as_a_transformer() {
        let m = tiny(4, 2, false);
        let d = tmpdir();
        let mut ts: Vec<T> = vec![
            (
                "model.embed_tokens.weight".into(),
                vec![20, 16],
                m.w.embed.to_f32(),
            ),
            ("model.norm.weight".into(), vec![16], m.w.final_norm.clone()),
            (
                "lm_head.weight".into(),
                vec![20, 16],
                m.w.lm_head.as_ref().unwrap().to_f32(),
            ),
        ];
        for (i, l) in m.w.layers.iter().enumerate() {
            let p = format!("model.layers.{i}.");
            ts.push((
                format!("{p}input_layernorm.weight"),
                vec![16],
                l.attn_norm.clone(),
            ));
            ts.push((
                format!("{p}post_attention_layernorm.weight"),
                vec![16],
                l.ffn_norm.clone(),
            ));
            for (n, rows, cols, w) in [
                ("self_attn.q_proj", 32, 16, &l.wq),
                ("self_attn.k_proj", 16, 16, &l.wk),
                ("self_attn.v_proj", 16, 16, &l.wv),
                ("self_attn.o_proj", 16, 32, &l.wo),
                ("mlp.gate_proj", 24, 16, &l.w_gate),
                ("mlp.up_proj", 24, 16, &l.w_up),
                ("mlp.down_proj", 16, 24, &l.w_down),
            ] {
                ts.push((format!("{p}{n}.weight"), vec![rows, cols], w.to_f32()));
            }
        }
        write_st(&d.join("model.safetensors"), &ts);
        std::fs::write(d.join("config.json"), r#"{"model_type":"llama","hidden_size":16,"num_hidden_layers":3,"num_attention_heads":4,
            "num_key_value_heads":2,"head_dim":8,"intermediate_size":24,"vocab_size":20,"max_position_embeddings":32,"rope_theta":10000.0,
            "rms_norm_eps":1e-5,"tie_word_embeddings":false}"#).unwrap();
        let any = load_any(&d).unwrap();
        assert!(matches!(any, AnyModel::Dense(_)));
        let toks = [3usize, 7, 1, 19];
        close(
            &any.forward_batch(&mut any.new_state(), &toks).unwrap(),
            &m.forward_batch(&mut m.new_state(), &toks).unwrap(),
            1e-4,
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn classification_and_config_parsing() {
        assert_eq!(classify_family("mamba"), Some(FamilyKind::Mamba1));
        assert_eq!(classify_family("Falcon_Mamba"), Some(FamilyKind::Mamba1));
        assert_eq!(classify_family("mamba2"), Some(FamilyKind::Mamba2));
        assert_eq!(classify_family("rwkv6"), Some(FamilyKind::Rwkv6));
        assert_eq!(classify_family("rwkv7"), Some(FamilyKind::Rwkv7));
        for no in ["rwkv", "rwkv5", "llama", "jamba", "zamba2", "bamba", ""] {
            assert_eq!(classify_family(no), None, "{no}");
        }
        let c = FamilyConfig::from_hf_config(MAMBA2_CFG).unwrap();
        assert_eq!(
            (c.kind, c.layers, c.vocab, c.hidden()),
            (FamilyKind::Mamba2, 2, 24, 16)
        );
        let c = FamilyConfig::from_hf_config(r#"{"architectures":["Mamba2ForCausalLM"],"hidden_size":16,"num_heads":4,"head_dim":8,"num_hidden_layers":3}"#).unwrap();
        assert_eq!((c.kind, c.layers), (FamilyKind::Mamba2, 3));
        assert_eq!(
            FamilyConfig::from_hf_config(RWKV7_CFG).unwrap().kind,
            FamilyKind::Rwkv7
        );
        assert!(matches!(
            FamilyConfig::from_hf_config(
                r#"{"model_type":"rwkv","hidden_size":16,"num_hidden_layers":2}"#
            ),
            Err(FamilyError::Unsupported(_))
        ));
    }

    #[test]
    fn broken_descriptions_are_errors() {
        for bad in [
            "",
            "[]",
            "{",
            "null",
            r#"{"model_type":"mamba"}"#,
            r#"{"model_type":"mamba","hidden_size":0,"num_hidden_layers":2}"#,
            r#"{"model_type":"mamba","hidden_size":16,"num_hidden_layers":0}"#,
            r#"{"model_type":"mamba","hidden_size":16,"num_hidden_layers":99999999}"#,
            r#"{"model_type":"mamba","hidden_size":16,"num_hidden_layers":2,"state_size":0}"#,
            r#"{"model_type":"mamba","hidden_size":99999999999,"num_hidden_layers":2,"expand":99999999999}"#,
            r#"{"model_type":"mamba2","hidden_size":16,"num_hidden_layers":2,"num_heads":3,"n_groups":2}"#,
            r#"{"model_type":"rwkv6","hidden_size":15,"head_size":8,"num_hidden_layers":2}"#,
            r#"{"model_type":"rwkv6","hidden_size":16,"num_hidden_layers":2}"#,
            r#"{"model_type":"rwkv7","hidden_size":16,"num_hidden_layers":2}"#,
        ] {
            assert!(
                FamilyConfig::from_hf_config(bad).is_err(),
                "accepted: {bad}"
            );
        }
    }

    #[test]
    fn hostile_files_are_errors_not_panics() {
        // missing tensor, named in the message
        let (d, _) = make("mamba1", 5);
        let mut g = Gen::new(5);
        mamba1(&mut g);
        g.out.retain(|t| t.0 != "backbone.norm_f.weight");
        write_st(&d.join("model.safetensors"), &g.out);
        let e = load_any(&d).err().unwrap().to_string();
        assert!(e.contains("backbone.norm_f.weight"), "{e}");
        // wrong tensor size
        let mut g = Gen::new(5);
        mamba1(&mut g);
        for t in &mut g.out {
            if t.0 == "backbone.layers.1.mixer.in_proj.weight" {
                t.2.pop();
                t.1 = vec![t.2.len()];
            }
        }
        write_st(&d.join("model.safetensors"), &g.out);
        assert!(load_any(&d).is_err());
        // truncated file at every few cut points
        let mut g = Gen::new(5);
        mamba1(&mut g);
        write_st(&d.join("model.safetensors"), &g.out);
        let full = std::fs::read(d.join("model.safetensors")).unwrap();
        for cut in [0, 3, 8, 40, full.len() / 3, full.len() - 1] {
            std::fs::write(d.join("model.safetensors"), &full[..cut]).unwrap();
            assert!(load_any(&d).is_err(), "cut at {cut}");
        }
        // an embedding table whose size is not a multiple of the hidden size
        let mut g = Gen::new(5);
        mamba1(&mut g);
        g.out[0].2.truncate(V * D - 3);
        g.out[0].1 = vec![V * D - 3];
        write_st(&d.join("model.safetensors"), &g.out);
        assert!(load_any(&d).is_err());
        // configuration promises more tokens than the table has
        let mut g = Gen::new(5);
        mamba1(&mut g);
        write_st(&d.join("model.safetensors"), &g.out);
        std::fs::write(
            d.join("config.json"),
            MAMBA1_CFG.replace("\"vocab_size\":24", "\"vocab_size\":9999"),
        )
        .unwrap();
        assert!(load_any(&d).is_err());
        // garbage and an unsupported recurrent family
        std::fs::write(d.join("config.json"), "not json").unwrap();
        assert!(load_any(&d).is_err());
        std::fs::write(
            d.join("config.json"),
            r#"{"model_type":"rwkv","hidden_size":16,"num_hidden_layers":2}"#,
        )
        .unwrap();
        assert!(load_any(&d).is_err());
        // missing path and an empty folder
        assert!(load_any(&d.join("nope")).is_err());
        let e = tmpdir();
        assert!(load_any(&e).is_err());
        std::fs::remove_dir_all(d).unwrap();
        std::fs::remove_dir_all(e).unwrap();
    }

    #[test]
    fn state_space_models_use_the_accelerator_and_fall_back() {
        use crate::accel::testbe::Odd;
        let _g = crate::slicegpu::test_lock();
        for kind in ["mamba1", "mamba2", "rwkv6", "rwkv7-fla"] {
            let (d, _) = make(kind, 4);
            let AnyModel::Ssm(mut m) = load_any(&d).unwrap() else {
                panic!("{kind}")
            };
            let toks = [1usize, 5, 2, 9, 3, 7];
            let cpu = m.forward_batch(&mut m.new_state(), &toks).unwrap();
            let rep = m.attach_backend(Odd::new(usize::MAX, 1.0, None));
            assert!(rep.active, "{kind}: {rep:?}");
            let on = m.forward_batch(&mut m.new_state(), &toks).unwrap();
            for (a, b) in on.iter().zip(&cpu) {
                assert!((a - b).abs() < 1e-3 * (1.0 + b.abs()), "{kind}: {a} {b}");
            }
            let rep = m.attach_backend(Odd::new(3, 1.0, None));
            assert!(rep.active);
            let on = m.forward_batch(&mut m.new_state(), &toks).unwrap();
            assert!(crate::slicegpu::failure().is_some(), "{kind}");
            for (a, b) in on.iter().zip(&cpu) {
                assert!((a - b).abs() < 1e-3 * (1.0 + b.abs()), "{kind}: {a} {b}");
            }
            drop(m);
            std::fs::remove_dir_all(d).unwrap();
        }
    }

    #[test]
    fn bad_tokens_and_foreign_states_are_errors() {
        let (d, _) = make("rwkv7-fla", 6);
        let AnyModel::Ssm(m) = load_any(&d).unwrap() else {
            panic!()
        };
        let mut st = m.new_state();
        assert!(matches!(m.forward(&mut st, V), Err(ModelError::Token(t)) if t == V));
        assert!(matches!(
            m.forward_batch(&mut st, &[]),
            Err(ModelError::NoTokens)
        ));
        assert!(matches!(
            m.forward_batch(&mut st, &[1, 2, usize::MAX]),
            Err(ModelError::Token(_))
        ));
        assert_eq!(st.pos, 0, "a refused call must not advance the state");
        let (d2, _) = make("mamba1", 6);
        let AnyModel::Ssm(other) = load_any(&d2).unwrap() else {
            panic!()
        };
        assert!(other.forward(&mut st, 1).is_err());
        std::fs::remove_dir_all(d).unwrap();
        std::fs::remove_dir_all(d2).unwrap();
    }

    #[test]
    fn generate_stops_on_stop_tokens_cancel_and_callback() {
        let (d, _) = make("mamba1", 8);
        let m = load_any(&d).unwrap();
        let free = {
            let mut out = vec![];
            generate(
                &m,
                &mut m.new_state(),
                &[1, 2],
                &mut Sampler::new(0.0, 0, 1),
                &GenerateOptions {
                    max_tokens: 5,
                    ..Default::default()
                },
                &mut |t| {
                    out.push(t);
                    true
                },
            )
            .unwrap();
            out
        };
        let stop = GenerateOptions {
            max_tokens: 50,
            stop_tokens: vec![free[2]],
            cancel: None,
        };
        let mut got = vec![];
        let s = generate(
            &m,
            &mut m.new_state(),
            &[1, 2],
            &mut Sampler::new(0.0, 0, 1),
            &stop,
            &mut |t| {
                got.push(t);
                true
            },
        )
        .unwrap();
        assert_eq!(s.stop, StopReason::StopToken);
        assert_eq!(
            got,
            free[..2]
                .to_vec()
                .iter()
                .copied()
                .take_while(|t| *t != free[2])
                .collect::<Vec<_>>()
        );
        let flag = std::sync::atomic::AtomicBool::new(true);
        let s = generate(
            &m,
            &mut m.new_state(),
            &[1],
            &mut Sampler::new(0.0, 0, 1),
            &GenerateOptions {
                max_tokens: 9,
                stop_tokens: vec![],
                cancel: Some(&flag),
            },
            &mut |_| true,
        )
        .unwrap();
        assert_eq!((s.stop, s.generated), (StopReason::Cancelled, 0));
        let s = generate(
            &m,
            &mut m.new_state(),
            &[1],
            &mut Sampler::new(0.0, 0, 1),
            &GenerateOptions::default(),
            &mut |_| false,
        )
        .unwrap();
        assert_eq!((s.stop, s.generated), (StopReason::Callback, 1));
        std::fs::remove_dir_all(d).unwrap();
    }

    /// Long, repeated runs stay finite for every family (the blocks are stable; the model must not break that).
    #[test]
    fn long_runs_stay_finite() {
        for (kind, seed) in [
            ("mamba1", 21),
            ("mamba2", 22),
            ("rwkv6", 23),
            ("rwkv7-fla", 24),
        ] {
            let (d, _) = make(kind, seed);
            let m = load_any(&d).unwrap();
            let mut st = m.new_state();
            let mut tok = 1;
            for i in 0..600 {
                let lg = m.forward(&mut st, tok).unwrap();
                assert!(lg.iter().all(|v| v.is_finite()), "{kind} step {i}");
                tok = Sampler::new(0.0, 0, 1).sample(&lg);
            }
            std::fs::remove_dir_all(d).unwrap();
        }
    }

    /// A GGUF Mamba file with llama.cpp tensor names loads through `load_any` and matches the HF layout.
    #[test]
    fn gguf_mamba_matches_the_same_weights_in_hf_names() {
        use crate::gguf::tests::put_str;
        let (d, _) = make("mamba1", 31);
        let hf = load_any(&d).unwrap();
        // re-key the same tensors with GGUF names (ssm_a holds -exp(A_log))
        let mut g = Gen::new(31);
        mamba1(&mut g);
        let mut ts: Vec<T> = vec![];
        for (n, shape, data) in g.out {
            let name = match n.as_str() {
                "backbone.embeddings.weight" => "token_embd.weight".to_string(),
                "backbone.norm_f.weight" => "output_norm.weight".to_string(),
                s => {
                    let rest = s.strip_prefix("backbone.layers.").unwrap();
                    let (i, tail) = rest.split_once('.').unwrap();
                    let t = match tail {
                        "norm.weight" => "attn_norm.weight",
                        "mixer.in_proj.weight" => "ssm_in.weight",
                        "mixer.conv1d.weight" => "ssm_conv1d.weight",
                        "mixer.conv1d.bias" => "ssm_conv1d.bias",
                        "mixer.x_proj.weight" => "ssm_x.weight",
                        "mixer.dt_proj.weight" => "ssm_dt.weight",
                        "mixer.dt_proj.bias" => "ssm_dt.bias",
                        "mixer.A_log" => "ssm_a",
                        "mixer.D" => "ssm_d",
                        "mixer.out_proj.weight" => "ssm_out.weight",
                        o => panic!("{o}"),
                    };
                    format!("blk.{i}.{t}")
                }
            };
            let data = if name.ends_with("ssm_a") {
                data.iter().map(|a| -a.exp()).collect()
            } else {
                data
            };
            ts.push((name, shape, data));
        }
        // minimal GGUF writer: F32 tensors, dims reversed as the format stores them
        let mut v = b"GGUF".to_vec();
        v.extend(3u32.to_le_bytes());
        v.extend((ts.len() as u64).to_le_bytes());
        let kvs: [(&str, u32, u64); 8] = [
            ("mamba.embedding_length", 4, 16),
            ("mamba.block_count", 4, 2),
            ("mamba.context_length", 4, 4096),
            ("mamba.ssm.inner_size", 4, 32),
            ("mamba.ssm.state_size", 4, 4),
            ("mamba.ssm.conv_kernel", 4, 3),
            ("mamba.ssm.time_step_rank", 4, 2),
            ("zz.dummy", 4, 0),
        ];
        v.extend(((kvs.len() + 1) as u64).to_le_bytes());
        put_str(&mut v, "general.architecture");
        v.extend(8u32.to_le_bytes());
        put_str(&mut v, "mamba");
        for (k, ty, val) in kvs {
            put_str(&mut v, k);
            v.extend(ty.to_le_bytes());
            v.extend((val as u32).to_le_bytes());
        }
        let (mut info, mut data) = (vec![], vec![]);
        for (n, shape, d) in &ts {
            while data.len() % 32 != 0 {
                data.push(0);
            }
            put_str(&mut info, n);
            // GGUF keeps conv1d as [k, channels]; our flat layout is the same
            let dims: Vec<usize> = if shape.len() == 3 {
                vec![shape[2], shape[0]]
            } else {
                shape.clone()
            };
            info.extend((dims.len() as u32).to_le_bytes());
            for x in dims.iter().rev().rev() {
                info.extend((*x as u64).to_le_bytes());
            }
            info.extend(0u32.to_le_bytes());
            info.extend((data.len() as u64).to_le_bytes());
            data.extend(d.iter().flat_map(|x| x.to_le_bytes()));
        }
        v.extend(info);
        while !v.len().is_multiple_of(32) {
            v.push(0);
        }
        v.extend(data);
        let path = d.join("m.gguf");
        std::fs::write(&path, v).unwrap();
        let gg = load_any(&path).unwrap();
        assert_eq!(gg.family(), Some(FamilyKind::Mamba1));
        let toks = [3usize, 7, 1, 19, 0];
        close(
            &gg.forward_batch(&mut gg.new_state(), &toks).unwrap(),
            &hf.forward_batch(&mut hf.new_state(), &toks).unwrap(),
            1e-3,
        );
        // a truncated GGUF is a clean error
        let b = std::fs::read(&path).unwrap();
        std::fs::write(&path, &b[..b.len() - 50]).unwrap();
        assert!(load_any(&path).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }
}
