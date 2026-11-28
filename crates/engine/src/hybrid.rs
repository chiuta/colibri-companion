//! Hybrid models: attention layers interleaved with state-space layers (Jamba, Bamba).
//!
//! Every layer is `x += mixer(norm(x))` followed by `x += mlp(norm(x))`; the mixer is either
//! attention (with a KV cache) or a Mamba / Mamba-2 block (with a constant-size state). Which layer
//! is which comes from the configuration (`attn_layer_period`/`attn_layer_offset` for Jamba,
//! `attn_layer_indices` for Bamba).
//!
//! The attention layers are not re-implemented: they form an ordinary [`Model`] that holds only the
//! attention layers (its tensors are found through a name-translating view of the checkpoint), and
//! the hybrid model calls it one layer at a time ([`Model::run_layers_batch`]). The state-space
//! layers use the mixers of [`crate::ssm`]. A hybrid state therefore holds a KV cache for the
//! attention layers only plus one small recurrent state per state-space layer.
//!
//! What is **not** here:
//! * Zamba / Zamba2 (a single shared attention block fed with the embeddings concatenated to the
//!   hidden state, plus per-use LoRA adapters): the registry marks them `Partial`.
//! * Mixture-of-experts feed-forward layers (most of Jamba's checkpoints use them): refused with a
//!   clear error rather than run wrongly.
//! * Quantised state-space weights: the mixers hold dense f32. The attention layers and the MLPs
//!   follow [`LoadOptions`].
//! * Rewinding: a recurrent state cannot be truncated, so there is no prefix reuse; start a new state.
//!
//! Provenance: the Hugging Face tensor names and configuration keys below are written from
//! memory of the public `JambaForCausalLM` / `BambaForCausalLM` layouts and were checked only
//! against synthetic checkpoints made with the same names; no real Jamba or Bamba file was available.
use crate::arch::{RopeScaling, Spec, Support};
use crate::family::{FamilyError, LanguageModel};
use crate::loader::{
    LoadError, LoadOptions, Naming, TensorSource, build_layer, build_outer, load_weight,
};
use crate::model::{Model, ModelError, State, Weight};
use crate::ops::{rmsnorm, silu_mul};
use crate::ssm::{
    Mamba1, Mamba1Config, Mamba1State, Mamba2, Mamba2Config, Mamba2State, MambaNames,
};
use serde_json::Value;
use std::sync::Arc;

/// Tokens per pass through the layers (bounds the scratch memory of a long prompt).
const CHUNK: usize = 64;
const MAX_LAYERS: usize = 4096;
const MAX_DIM: u64 = 1 << 22;

/// Which checkpoint family the tensor names and configuration follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavor {
    Jamba,
    Bamba,
}

/// The mixer of one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Attention,
    Ssm,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SsmConfig {
    M1(Mamba1Config),
    M2(Mamba2Config),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HybridConfig {
    pub flavor: Flavor,
    /// The attention side: `layers` is the TOTAL layer count here; the attention-only model made at
    /// load time gets a copy with `layers` = the number of attention layers.
    pub spec: Spec,
    /// One entry per layer.
    pub kinds: Vec<Kind>,
    pub ssm: SsmConfig,
    /// Epsilon of the RMS norms in front of the mixer and the MLP.
    pub eps: f32,
}

fn cfg_err<T>(s: impl Into<String>) -> Result<T, FamilyError> {
    Err(FamilyError::Config(s.into()))
}
fn uget(c: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| c.get(*k).and_then(Value::as_u64))
}
fn fget(c: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| c.get(*k).and_then(Value::as_f64))
}
fn bget(c: &Value, k: &str, d: bool) -> bool {
    c.get(k).and_then(Value::as_bool).unwrap_or(d)
}
fn dim(name: &str, v: u64) -> Result<u64, FamilyError> {
    if v == 0 || v > MAX_DIM {
        cfg_err(format!("`{name}` = {v} is out of range"))
    } else {
        Ok(v)
    }
}

impl HybridConfig {
    /// From a Hugging Face `config.json` of a Jamba or Bamba model.
    pub fn from_hf_config(json: &str) -> Result<HybridConfig, FamilyError> {
        let root: Value = serde_json::from_str(json)
            .map_err(|e| FamilyError::Config(format!("config.json: {e}")))?;
        if !root.is_object() {
            return cfg_err("config.json is not a JSON object");
        }
        let c = root
            .get("text_config")
            .filter(|t| t.is_object())
            .unwrap_or(&root);
        let mt = c
            .get("model_type")
            .or_else(|| root.get("model_type"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        let flavor = match mt.as_str() {
            "jamba" => Flavor::Jamba,
            "bamba" => Flavor::Bamba,
            other => {
                return Err(FamilyError::Unsupported(format!(
                    "model type `{other}` is not a Jamba or Bamba model"
                )));
            }
        };
        let need = |keys: &[&str]| {
            uget(c, keys).ok_or_else(|| {
                FamilyError::Config(format!("the model description has no `{}`", keys[0]))
            })
        };
        let layers = need(&["num_hidden_layers"])? as usize;
        if layers == 0 || layers > MAX_LAYERS {
            return cfg_err(format!("{layers} layers is out of range"));
        }
        let hidden = dim("hidden_size", need(&["hidden_size"])?)?;
        let heads = dim("num_attention_heads", need(&["num_attention_heads"])?)?;
        let kv = dim(
            "num_key_value_heads",
            uget(c, &["num_key_value_heads"]).unwrap_or(heads),
        )?;
        if !heads.is_multiple_of(kv) {
            return cfg_err("num_attention_heads is not a multiple of num_key_value_heads");
        }
        let head_dim = uget(c, &["head_dim"])
            .filter(|d| *d > 0)
            .unwrap_or(hidden / heads);
        if head_dim == 0 || (uget(c, &["head_dim"]).is_none() && !hidden.is_multiple_of(heads)) {
            return cfg_err("hidden_size is not a multiple of num_attention_heads");
        }
        let ffn = dim("intermediate_size", need(&["intermediate_size"])?)?;
        let vocab = uget(c, &["vocab_size"]).unwrap_or(0);
        let context = uget(c, &["max_position_embeddings"]).unwrap_or(4096).max(1);
        let eps = fget(c, &["rms_norm_eps"]).unwrap_or(1e-6) as f32;
        if !(eps.is_finite() && eps >= 0.0) {
            return cfg_err("rms_norm_eps is not a valid number");
        }

        let mut kinds = vec![Kind::Ssm; layers];
        match flavor {
            Flavor::Jamba => {
                let period = uget(c, &["attn_layer_period"]).unwrap_or(8).max(1) as usize;
                let offset = uget(c, &["attn_layer_offset"]).unwrap_or(4) as usize;
                for (i, k) in kinds.iter_mut().enumerate() {
                    if i % period == offset {
                        *k = Kind::Attention;
                    }
                }
                let experts = uget(c, &["num_experts"]).unwrap_or(16);
                let (ep, eo) = (
                    uget(c, &["expert_layer_period"]).unwrap_or(2).max(1) as usize,
                    uget(c, &["expert_layer_offset"]).unwrap_or(1) as usize,
                );
                if experts > 1
                    && let Some(i) = (0..layers).find(|i| i % ep == eo)
                {
                    return Err(FamilyError::Unsupported(format!(
                        "this Jamba checkpoint has mixture-of-experts feed-forward layers (layer {i} of {layers}); only dense feed-forward layers are implemented"
                    )));
                }
            }
            Flavor::Bamba => {
                if let Some(a) = c.get("attn_layer_indices").and_then(Value::as_array) {
                    for v in a {
                        let i = v
                            .as_u64()
                            .filter(|i| (*i as usize) < layers)
                            .ok_or_else(|| {
                                FamilyError::Config(
                                    "attn_layer_indices names a layer that does not exist".into(),
                                )
                            })? as usize;
                        kinds[i] = Kind::Attention;
                    }
                }
            }
        }

        let mut spec = Spec::dense(
            &mt,
            layers as u64,
            hidden,
            heads,
            kv,
            head_dim,
            ffn,
            vocab,
            context,
        );
        spec.support = Support::Hybrid;
        spec.norm_eps = eps as f64;
        spec.tied_embeddings = bget(c, "tie_word_embeddings", false);
        let expand = uget(c, &["mamba_expand"]).unwrap_or(2);
        let ssm = match flavor {
            Flavor::Jamba => {
                spec.rope_scaling = RopeScaling::Nope; // Jamba's attention has no position encoding
                let dt_rank = match c.get("mamba_dt_rank") {
                    Some(Value::Number(n)) => n.as_u64().unwrap_or(0) as usize,
                    _ => (hidden as usize).div_ceil(16),
                };
                let cfg = Mamba1Config {
                    d_model: hidden as usize,
                    d_inner: (hidden * expand) as usize,
                    d_state: uget(c, &["mamba_d_state"]).unwrap_or(16) as usize,
                    d_conv: uget(c, &["mamba_d_conv"]).unwrap_or(4) as usize,
                    dt_rank,
                    conv_bias: bget(c, "mamba_conv_bias", true),
                    bias: bget(c, "mamba_proj_bias", false),
                    bcdt_rms_eps: Some(eps),
                };
                cfg.validate()?;
                SsmConfig::M1(cfg)
            }
            Flavor::Bamba => {
                spec.rope_theta = fget(c, &["rope_theta"]).unwrap_or(10_000.0);
                let pf = fget(c, &["partial_rotary_factor"])
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
                spec.rotary_dim = (((head_dim as f64 * pf) as u64) & !1).clamp(2, head_dim.max(2));
                let n_heads = uget(c, &["mamba_n_heads"]).unwrap_or(128) as usize;
                let head_dim_m = match c.get("mamba_d_head") {
                    Some(Value::Number(n)) => n.as_u64().unwrap_or(0) as usize,
                    _ => (hidden * expand) as usize / n_heads.max(1),
                };
                let cfg = Mamba2Config {
                    d_model: hidden as usize,
                    n_heads,
                    head_dim: head_dim_m,
                    d_state: uget(c, &["mamba_d_state"]).unwrap_or(128) as usize,
                    n_groups: uget(c, &["mamba_n_groups"]).unwrap_or(1) as usize,
                    d_conv: uget(c, &["mamba_d_conv"]).unwrap_or(4) as usize,
                    conv_bias: bget(c, "mamba_conv_bias", true),
                    bias: bget(c, "mamba_proj_bias", false),
                    norm_eps: eps,
                    norm_before_gate: false,
                    chunk_size: uget(c, &["mamba_chunk_size"]).unwrap_or(256) as usize,
                };
                cfg.validate()?;
                SsmConfig::M2(cfg)
            }
        };
        Ok(HybridConfig {
            flavor,
            spec,
            kinds,
            ssm,
            eps,
        })
    }

    pub fn attention_layers(&self) -> usize {
        self.kinds.iter().filter(|k| **k == Kind::Attention).count()
    }
    pub fn ssm_layers(&self) -> usize {
        self.kinds.len() - self.attention_layers()
    }

    fn validate(&self) -> Result<(), FamilyError> {
        if self.kinds.len() as u64 != self.spec.layers
            || self.kinds.is_empty()
            || self.kinds.len() > MAX_LAYERS
        {
            return cfg_err(format!(
                "{} layer kinds for {} layers",
                self.kinds.len(),
                self.spec.layers
            ));
        }
        let h = self.spec.hidden as usize;
        let mixer_h = match &self.ssm {
            SsmConfig::M1(c) => c.d_model,
            SsmConfig::M2(c) => c.d_model,
        };
        if mixer_h != h {
            return cfg_err("the state-space mixer and the model disagree on the hidden size");
        }
        match &self.ssm {
            SsmConfig::M1(c) => c.validate()?,
            SsmConfig::M2(c) => c.validate()?,
        }
        if !(self.eps.is_finite() && self.eps >= 0.0) {
            return cfg_err("the norm epsilon is not a valid number");
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------------ checkpoint view

/// Presents the checkpoint to the dense loader as if it held only the attention layers, numbered
/// `0..n`, with Llama-style sub-names (`mlp.*`, `post_attention_layernorm`).
struct AttentionView<'a> {
    inner: &'a dyn TensorSource,
    /// `real[a]` = the layer index (in the whole model) of the `a`-th attention layer.
    real: Vec<usize>,
}
impl AttentionView<'_> {
    fn map(&self, name: &str) -> String {
        let Some(rest) = name.strip_prefix("model.layers.") else {
            return name.to_string();
        };
        let Some((n, tail)) = rest.split_once('.') else {
            return name.to_string();
        };
        let Some(real) = n.parse::<usize>().ok().and_then(|a| self.real.get(a)) else {
            return name.to_string();
        };
        let tail = if let Some(t) = tail.strip_prefix("post_attention_layernorm.") {
            format!("pre_ff_layernorm.{t}")
        } else if let Some(t) = tail.strip_prefix("mlp.") {
            format!("feed_forward.{t}")
        } else {
            tail.to_string()
        };
        format!("model.layers.{real}.{tail}")
    }
}
impl TensorSource for AttentionView<'_> {
    fn read(&self, name: &str) -> Result<Vec<f32>, LoadError> {
        self.inner.read(&self.map(name))
    }
    fn has(&self, name: &str) -> bool {
        self.inner.has(&self.map(name))
    }
    fn read_raw(
        &self,
        name: &str,
        mmap: bool,
    ) -> Result<Option<crate::loader::RawTensor>, LoadError> {
        self.inner.read_raw(&self.map(name), mmap)
    }
}

// ------------------------------------------------------------------------------------- model

enum Mixer {
    M1(Box<Mamba1>),
    M2(Box<Mamba2>),
}
struct SsmLayer {
    norm: Vec<f32>,
    mixer: Mixer,
    ff_norm: Vec<f32>,
    gate: Weight,
    up: Weight,
    down: Weight,
}

pub struct HybridModel {
    /// Declared first so it is dropped before the weights it points at.
    reg: crate::slicegpu::Registration,
    cfg: HybridConfig,
    /// The attention layers (with their MLPs), the embedding table, the final norm and the head.
    attn: Model,
    /// `true` when a failed upload consumed the attention weights (see `attach_backend`).
    attn_lost: bool,
    /// One entry per SSM layer, in layer order.
    ssm: Vec<SsmLayer>,
    hidden: usize,
}

/// Device-side KV positions given to the attention layers of a hybrid (there are few of them).
const ATTN_KV_CAP: usize = 16384;

/// Matrix `w` of row length `cols`, as the backend takes it.
fn weight_ref(w: &Weight, cols: usize) -> crate::backend::WeightRef<'_> {
    use crate::backend::WeightRef;
    match w {
        Weight::Dense(v) => WeightRef::dense(v.len() / cols.max(1), cols, v),
        Weight::Quant(q) => WeightRef::quant(q.dtype, q.rows, q.cols, q.bytes.as_slice()),
    }
}

enum SsmLayerState {
    M1(Mamba1State),
    M2(Mamba2State),
}

/// Per-conversation state: a KV cache for the attention layers and a recurrent state per
/// state-space layer. After an `Err` from a `forward` call the state must be discarded.
pub struct HybridState {
    kv: State,
    ssm: Vec<SsmLayerState>,
    pub pos: usize,
}
impl HybridState {
    /// Bytes held: grows with the context for the attention layers only.
    pub fn bytes(&self) -> usize {
        self.kv.kv_bytes()
            + self
                .ssm
                .iter()
                .map(|s| match s {
                    SsmLayerState::M1(s) => s.bytes(),
                    SsmLayerState::M2(s) => s.bytes(),
                })
                .sum::<usize>()
    }
    /// Bytes of the recurrent part, which never grows.
    pub fn ssm_bytes(&self) -> usize {
        self.bytes() - self.kv.kv_bytes()
    }
    /// An independent copy (for branching conversations, beam search).
    pub fn fork(&self) -> Result<HybridState, ModelError> {
        let ssm = self
            .ssm
            .iter()
            .map(|s| match s {
                SsmLayerState::M1(s) => SsmLayerState::M1(s.clone()),
                SsmLayerState::M2(s) => SsmLayerState::M2(s.clone()),
            })
            .collect();
        Ok(HybridState {
            kv: self.kv.fork()?,
            ssm,
            pos: self.pos,
        })
    }
}

fn load_err(e: impl std::fmt::Display) -> FamilyError {
    FamilyError::Load(e.to_string())
}

impl HybridModel {
    /// Put the state-space layers' matrices (and their dense MLPs) and the attention layers on
    /// `be`; never fails — the report says what happened and whatever the device does not take
    /// stays on the CPU. The attention half is a dense [`Model`] and is offloaded through it, so
    /// its layers keep their KV cache on the device (up to 16384 positions; a state made before
    /// the call cannot be continued after it, and a state with a device cache cannot be forked).
    /// If the device fails *while* the attention layers run, that call returns an error (the
    /// device-side cache cannot be rebuilt on the host); the state-space products fall back alone.
    pub fn attach_backend(
        &mut self,
        be: Arc<dyn crate::backend::Backend>,
    ) -> crate::accel::AttachReport {
        self.reg = Default::default();
        let mut mats: Vec<(&[f32], usize)> = vec![];
        for l in &self.ssm {
            match &l.mixer {
                Mixer::M1(x) => mats.extend(x.matrices()),
                Mixer::M2(x) => mats.extend(x.matrices()),
            }
            let h = self.hidden;
            for (w, cols) in [(&l.gate, h), (&l.up, h)] {
                if let Weight::Dense(v) = w {
                    mats.push((v, cols));
                }
            }
            if let Weight::Dense(v) = &l.down {
                mats.push((v, v.len() / h.max(1)));
            }
        }
        let (reg, mut rep) = crate::slicegpu::register(be.clone(), &mats);
        self.reg = reg;
        self.attach_attention(be, &mut rep);
        rep
    }

    fn attach_attention(
        &mut self,
        be: Arc<dyn crate::backend::Backend>,
        rep: &mut crate::accel::AttachReport,
    ) {
        use crate::model::{AccelOptions, Weights};
        let n = self.attn.w.layers.len();
        if n == 0 || self.attn_lost {
            return;
        }
        let h = self.hidden;
        let say = |rep: &mut crate::accel::AttachReport, t: String| {
            if !rep.note.is_empty() {
                rep.note.push_str("; ");
            }
            rep.note.push_str(&t);
        };
        // A failed upload would consume the weights, so a small upload is tried first.
        let probe = || -> Result<(), String> {
            let l = &self.attn.w.layers[0];
            be.alloc(h).map_err(|e| e.to_string())?;
            be.upload(weight_ref(&l.wq, h)).map_err(|e| e.to_string())?;
            Ok(())
        };
        if let Err(e) = probe() {
            say(rep, format!("attention layers stay on the CPU ({e})"));
            return;
        }
        // an empty stand-in for the model while the real one is handed to the device
        let mut spec = self.attn.spec.clone();
        spec.layers = 0;
        spec.vocab = 0;
        let empty = Weights {
            final_norm: vec![0.0; h],
            ..Default::default()
        };
        let Ok(stand_in) = Model::new(spec, empty) else {
            say(rep, "attention layers stay on the CPU".into());
            return;
        };
        let old = std::mem::replace(&mut self.attn, stand_in);
        let per_layer: Vec<u64> = old
            .w
            .layers
            .iter()
            .map(|l| {
                [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]
                    .iter()
                    .map(|w| w.stored_bytes() as u64)
                    .sum()
            })
            .collect();
        let opts = AccelOptions {
            kv_capacity: (self.cfg.spec.context as usize).clamp(1, ATTN_KV_CAP),
            ..AccelOptions::default()
        };
        match old.with_backend_opts(be, opts) {
            Err(e) => {
                self.attn_lost = true;
                say(
                    rep,
                    format!(
                        "the device refused the attention layers ({e}) and they were lost; reload the model"
                    ),
                );
            }
            Ok(mut m) => {
                let on: Vec<bool> = m.offloaded();
                let k = on.iter().filter(|b| **b).count();
                rep.total += n * 7;
                if k == 0 {
                    // no layer went: a model without a device cache has no position limit
                    let w = std::mem::take(&mut m.w);
                    let threads = m.threads();
                    match Model::new(m.spec.clone(), w) {
                        Ok(mut plain) => {
                            plain.set_threads(threads);
                            self.attn = plain;
                        }
                        Err(_) => {
                            self.attn_lost = true;
                        }
                    }
                    say(rep, "the device took no attention layer".into());
                } else {
                    rep.matrices += k * 7;
                    rep.bytes += on
                        .iter()
                        .zip(&per_layer)
                        .filter(|(o, _)| **o)
                        .map(|(_, b)| *b)
                        .sum::<u64>();
                    rep.active = true;
                    self.attn = m;
                    if k < n {
                        say(rep, format!("{k}/{n} attention layers on the device"));
                    }
                }
            }
        }
    }

    pub fn config(&self) -> &HybridConfig {
        &self.cfg
    }
    pub fn hidden(&self) -> usize {
        self.hidden
    }
    pub fn vocab(&self) -> usize {
        self.attn.spec.vocab as usize
    }
    /// The attention layers as an ordinary model (layer `a` of it is the `a`-th attention layer).
    pub fn attention_model(&self) -> &Model {
        &self.attn
    }

    /// Load every layer from a Hugging Face style checkpoint.
    pub fn load(
        cfg: HybridConfig,
        src: &dyn TensorSource,
        opts: LoadOptions,
    ) -> Result<HybridModel, FamilyError> {
        cfg.validate()?;
        let h = cfg.spec.hidden as usize;
        let real: Vec<usize> = cfg
            .kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == Kind::Attention)
            .map(|(i, _)| i)
            .collect();
        let mut aspec = cfg.spec.clone();
        aspec.layers = real.len() as u64;
        aspec.support = Support::Standard;
        let view = AttentionView {
            inner: src,
            real: real.clone(),
        };
        let mut layers = Vec::with_capacity(real.len());
        #[allow(clippy::needless_range_loop)]
        for a in 0..real.len() {
            let (l, is_moe, _) =
                build_layer(&aspec, &view, Naming::Hf, false, opts, a).map_err(load_err)?;
            if is_moe {
                return Err(FamilyError::Unsupported(format!(
                    "layer {} has a mixture-of-experts feed-forward",
                    real[a]
                )));
            }
            layers.push(l);
        }
        let weights = build_outer(&aspec, &view, Naming::Hf, opts, layers).map_err(load_err)?;
        let vocab = weights.embed.len() / h.max(1);
        if vocab == 0 || weights.embed.len() % h != 0 {
            return Err(load_err(
                "the embedding table is not a whole number of rows",
            ));
        }
        aspec.vocab = vocab as u64;
        let mut attn = Model::new(aspec, weights).map_err(load_err)?;
        attn.set_threads(opts.threads);

        let ff = cfg.spec.ffn as usize;
        let mut ssm = Vec::with_capacity(cfg.kinds.len() - real.len());
        for (i, k) in cfg.kinds.iter().enumerate() {
            if *k != Kind::Ssm {
                continue;
            }
            let p = format!("model.layers.{i}.");
            let read = |s: &str, want: usize| -> Result<Vec<f32>, FamilyError> {
                let v = src.read(&format!("{p}{s}")).map_err(load_err)?;
                if v.len() != want {
                    return Err(load_err(format!(
                        "{p}{s}: {} values, expected {want}",
                        v.len()
                    )));
                }
                Ok(v)
            };
            let w = |s: &str, rows: usize, cols: usize| -> Result<Weight, FamilyError> {
                let w = load_weight(src, &format!("{p}{s}"), opts).map_err(load_err)?;
                if w.len() != rows * cols {
                    return Err(load_err(format!(
                        "{p}{s}: {} values, expected {rows}×{cols}",
                        w.len()
                    )));
                }
                Ok(w)
            };
            let n = |s: &str| format!("{p}{s}");
            let mixer = match &cfg.ssm {
                SsmConfig::M1(c) => {
                    let names = MambaNames {
                        in_proj: n("mamba.in_proj.weight"),
                        in_bias: n("mamba.in_proj.bias"),
                        conv_w: n("mamba.conv1d.weight"),
                        conv_b: n("mamba.conv1d.bias"),
                        x_proj: n("mamba.x_proj.weight"),
                        dt_proj: n("mamba.dt_proj.weight"),
                        dt_bias: n("mamba.dt_proj.bias"),
                        a: n("mamba.A_log"),
                        d: n("mamba.D"),
                        out_proj: n("mamba.out_proj.weight"),
                        out_bias: n("mamba.out_proj.bias"),
                        norm: String::new(),
                    };
                    let m = Mamba1::load(c.clone(), src, &names, false)?.with_bcdt_weights(
                        read("mamba.dt_layernorm.weight", c.dt_rank)?,
                        read("mamba.b_layernorm.weight", c.d_state)?,
                        read("mamba.c_layernorm.weight", c.d_state)?,
                    )?;
                    Mixer::M1(Box::new(m))
                }
                SsmConfig::M2(c) => {
                    let names = MambaNames {
                        in_proj: n("mamba.in_proj.weight"),
                        in_bias: n("mamba.in_proj.bias"),
                        conv_w: n("mamba.conv1d.weight"),
                        conv_b: n("mamba.conv1d.bias"),
                        x_proj: String::new(),
                        dt_proj: String::new(),
                        dt_bias: n("mamba.dt_bias"),
                        a: n("mamba.A_log"),
                        d: n("mamba.D"),
                        out_proj: n("mamba.out_proj.weight"),
                        out_bias: n("mamba.out_proj.bias"),
                        norm: n("mamba.norm.weight"),
                    };
                    Mixer::M2(Box::new(Mamba2::load(c.clone(), src, &names, false)?))
                }
            };
            ssm.push(SsmLayer {
                norm: read("input_layernorm.weight", h)?,
                mixer,
                ff_norm: read("pre_ff_layernorm.weight", h)?,
                gate: w("feed_forward.gate_proj.weight", ff, h)?,
                up: w("feed_forward.up_proj.weight", ff, h)?,
                down: w("feed_forward.down_proj.weight", h, ff)?,
            });
        }
        Ok(HybridModel {
            reg: Default::default(),
            cfg,
            attn,
            attn_lost: false,
            ssm,
            hidden: h,
        })
    }

    pub fn new_state(&self) -> HybridState {
        let ssm = self
            .ssm
            .iter()
            .map(|l| match &l.mixer {
                Mixer::M1(m) => SsmLayerState::M1(m.new_state()),
                Mixer::M2(m) => SsmLayerState::M2(m.new_state()),
            })
            .collect();
        HybridState {
            kv: self.attn.new_state(),
            ssm,
            pos: 0,
        }
    }

    pub fn forward(&self, st: &mut HybridState, token: usize) -> Result<Vec<f32>, ModelError> {
        self.forward_batch(st, &[token])
    }

    /// Process `tokens` (continuing from `st`) and return the logits of the last one. Equal, within
    /// rounding, to calling [`HybridModel::forward`] for each token.
    pub fn forward_batch(
        &self,
        st: &mut HybridState,
        tokens: &[usize],
    ) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        if self.attn_lost {
            return Err(ModelError::Backend(
                "the attention layers were lost when the device refused them; reload the model"
                    .into(),
            ));
        }
        let vocab = self.vocab();
        if let Some(&t) = tokens.iter().find(|&&t| t >= vocab) {
            return Err(ModelError::Token(t));
        }
        if st.ssm.len() != self.ssm.len() || st.kv.layer_count() != self.attn.spec.layers as usize {
            return Err(ModelError::Shape(
                "the state belongs to a different model".into(),
            ));
        }
        let ctx = self.cfg.spec.context as usize;
        if self.attn.spec.layers > 0 && st.pos + tokens.len() > ctx {
            return Err(ModelError::ContextFull(ctx));
        }
        let mut last = vec![];
        for c in tokens.chunks(CHUNK) {
            last = self.run_chunk(st, c)?;
        }
        Ok(last)
    }

    fn run_chunk(&self, st: &mut HybridState, toks: &[usize]) -> Result<Vec<f32>, ModelError> {
        let (h, b) = (self.hidden, toks.len());
        let mut x = vec![0.0f32; b * h];
        for (t, &tok) in toks.iter().enumerate() {
            x[t * h..(t + 1) * h].copy_from_slice(&self.attn.embed_token(tok)?);
        }
        let pos0 = st.kv.pos;
        let (mut ai, mut si) = (0, 0);
        for k in &self.cfg.kinds {
            match k {
                Kind::Attention => {
                    self.attn
                        .run_layers_batch(&mut st.kv, ai..ai + 1, &mut x, pos0)?;
                    ai += 1;
                }
                Kind::Ssm => {
                    self.ssm_layer(&self.ssm[si], &mut st.ssm[si], &mut x, b)?;
                    si += 1;
                }
            }
        }
        self.attn.finish_batch(&mut st.kv, toks);
        st.pos += b;
        let logits = self.attn.logits_from_hidden(&x[(b - 1) * h..]);
        if logits.is_empty() {
            return Err(ModelError::Shape(
                "the output head rejected the hidden state".into(),
            ));
        }
        Ok(logits)
    }

    fn ssm_layer(
        &self,
        l: &SsmLayer,
        s: &mut SsmLayerState,
        x: &mut [f32],
        b: usize,
    ) -> Result<(), ModelError> {
        let (h, eps, ff) = (self.hidden, self.cfg.eps, self.cfg.spec.ffn as usize);
        let nt = self.attn.threads();
        let mut normed = vec![0.0f32; b * h];
        for (xr, nr) in x.chunks_exact(h).zip(normed.chunks_exact_mut(h)) {
            rmsnorm(xr, &l.norm, eps, nr);
        }
        let out = match (&l.mixer, s) {
            (Mixer::M1(m), SsmLayerState::M1(s)) => m.prefill(s, &normed),
            (Mixer::M2(m), SsmLayerState::M2(s)) => m.prefill(s, &normed),
            _ => {
                return Err(ModelError::Shape(
                    "the state belongs to a different model".into(),
                ));
            }
        }
        .map_err(|e| ModelError::Shape(e.to_string()))?;
        x.iter_mut().zip(&out).for_each(|(a, o)| *a += o);
        // feed-forward
        for (xr, nr) in x.chunks_exact(h).zip(normed.chunks_exact_mut(h)) {
            rmsnorm(xr, &l.ff_norm, eps, nr);
        }
        let (mut g, mut u, mut yt) = (vec![0.0f32; b * ff], vec![0.0f32; b * ff], Vec::new());
        mm(&l.gate, h, &normed, b, &mut g, &mut yt, nt)?;
        mm(&l.up, h, &normed, b, &mut u, &mut yt, nt)?;
        silu_mul(&mut g, &u);
        let mut d = vec![0.0f32; b * h];
        mm(&l.down, ff, &g, b, &mut d, &mut yt, nt)?;
        x.iter_mut().zip(&d).for_each(|(a, o)| *a += o);
        Ok(())
    }
}

/// `out = W · x` for `b` rows: on the device when the matrix is registered and the device works,
/// otherwise exactly [`Weight::matmat_nt`].
fn mm(
    w: &Weight,
    cols: usize,
    x: &[f32],
    b: usize,
    out: &mut [f32],
    yt: &mut Vec<f32>,
    nt: usize,
) -> Result<(), ModelError> {
    if let Weight::Dense(v) = w
        && crate::slicegpu::active()
        && crate::slicegpu::try_rows(v, cols, x, b, None, out)
    {
        return Ok(());
    }
    w.matmat_nt(cols, x, b, out, yt, nt)
}

impl LanguageModel for HybridModel {
    type State = HybridState;
    fn new_state(&self) -> HybridState {
        HybridModel::new_state(self)
    }
    fn forward_batch(
        &self,
        st: &mut HybridState,
        tokens: &[usize],
    ) -> Result<Vec<f32>, ModelError> {
        HybridModel::forward_batch(self, st, tokens)
    }
    fn vocab_size(&self) -> usize {
        self.vocab()
    }
    fn context(&self) -> usize {
        self.cfg.spec.context as usize
    }
    fn position(&self, st: &HybridState) -> usize {
        st.pos
    }
}

// ------------------------------------------------------------------------------------- tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct MapSource(HashMap<String, Vec<f32>>);
    impl TensorSource for MapSource {
        fn read(&self, name: &str) -> Result<Vec<f32>, LoadError> {
            self.0
                .get(name)
                .cloned()
                .ok_or_else(|| LoadError::Missing(name.into()))
        }
        fn has(&self, name: &str) -> bool {
            self.0.contains_key(name)
        }
    }

    fn rnd(s: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
            })
            .collect()
    }
    fn norm_w(s: &mut u64, n: usize) -> Vec<f32> {
        rnd(s, n, 0.4).into_iter().map(|v| 1.0 + v).collect()
    }

    const JAMBA: &str = r#"{"model_type":"jamba","hidden_size":16,"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2,
        "intermediate_size":24,"vocab_size":20,"max_position_embeddings":160,"attn_layer_period":4,"attn_layer_offset":1,"num_experts":1,
        "mamba_d_state":4,"mamba_d_conv":3,"mamba_expand":2,"mamba_dt_rank":2,"rms_norm_eps":1e-6}"#;
    const BAMBA: &str = r#"{"model_type":"bamba","hidden_size":16,"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2,
        "intermediate_size":24,"vocab_size":20,"max_position_embeddings":160,"attn_layer_indices":[1,3],"mamba_n_heads":4,"mamba_d_head":8,
        "mamba_d_state":4,"mamba_n_groups":1,"mamba_d_conv":3,"mamba_expand":2,"mamba_chunk_size":4,"rms_norm_eps":1e-5,"partial_rotary_factor":0.5}"#;

    /// A random checkpoint with exactly the tensors `cfg` asks for.
    fn synth(cfg: &HybridConfig, seed: u64) -> MapSource {
        let mut s = seed;
        let mut m = HashMap::new();
        let sp = &cfg.spec;
        let (h, ff, v) = (sp.hidden as usize, sp.ffn as usize, sp.vocab as usize);
        let (qd, kd) = (
            (sp.heads * sp.head_dim) as usize,
            (sp.kv_heads * sp.head_dim) as usize,
        );
        let mut put = |k: String, d: Vec<f32>| {
            m.insert(k, d);
        };
        put("model.embed_tokens.weight".into(), rnd(&mut s, v * h, 0.6));
        put("model.final_layernorm.weight".into(), norm_w(&mut s, h));
        put("lm_head.weight".into(), rnd(&mut s, v * h, 0.6));
        for (i, k) in cfg.kinds.iter().enumerate() {
            let p = format!("model.layers.{i}.");
            put(format!("{p}input_layernorm.weight"), norm_w(&mut s, h));
            put(format!("{p}pre_ff_layernorm.weight"), norm_w(&mut s, h));
            put(
                format!("{p}feed_forward.gate_proj.weight"),
                rnd(&mut s, ff * h, 0.6),
            );
            put(
                format!("{p}feed_forward.up_proj.weight"),
                rnd(&mut s, ff * h, 0.6),
            );
            put(
                format!("{p}feed_forward.down_proj.weight"),
                rnd(&mut s, h * ff, 0.6),
            );
            match k {
                Kind::Attention => {
                    put(
                        format!("{p}self_attn.q_proj.weight"),
                        rnd(&mut s, qd * h, 0.6),
                    );
                    put(
                        format!("{p}self_attn.k_proj.weight"),
                        rnd(&mut s, kd * h, 0.6),
                    );
                    put(
                        format!("{p}self_attn.v_proj.weight"),
                        rnd(&mut s, kd * h, 0.6),
                    );
                    put(
                        format!("{p}self_attn.o_proj.weight"),
                        rnd(&mut s, h * qd, 0.6),
                    );
                }
                Kind::Ssm => match &cfg.ssm {
                    SsmConfig::M1(c) => {
                        let (di, n, r, kk) = (c.d_inner, c.d_state, c.dt_rank, c.d_conv);
                        put(
                            format!("{p}mamba.in_proj.weight"),
                            rnd(&mut s, 2 * di * h, 0.6),
                        );
                        put(format!("{p}mamba.conv1d.weight"), rnd(&mut s, di * kk, 0.8));
                        put(format!("{p}mamba.conv1d.bias"), rnd(&mut s, di, 0.2));
                        put(
                            format!("{p}mamba.x_proj.weight"),
                            rnd(&mut s, (r + 2 * n) * di, 0.6),
                        );
                        put(format!("{p}mamba.dt_proj.weight"), rnd(&mut s, di * r, 0.6));
                        put(format!("{p}mamba.dt_proj.bias"), rnd(&mut s, di, 0.6));
                        put(format!("{p}mamba.A_log"), rnd(&mut s, di * n, 1.0));
                        put(format!("{p}mamba.D"), norm_w(&mut s, di));
                        put(
                            format!("{p}mamba.out_proj.weight"),
                            rnd(&mut s, h * di, 0.6),
                        );
                        put(format!("{p}mamba.dt_layernorm.weight"), norm_w(&mut s, r));
                        put(format!("{p}mamba.b_layernorm.weight"), norm_w(&mut s, n));
                        put(format!("{p}mamba.c_layernorm.weight"), norm_w(&mut s, n));
                    }
                    SsmConfig::M2(c) => {
                        put(
                            format!("{p}mamba.in_proj.weight"),
                            rnd(&mut s, c.proj_dim() * h, 0.6),
                        );
                        put(
                            format!("{p}mamba.conv1d.weight"),
                            rnd(&mut s, c.conv_dim() * c.d_conv, 0.8),
                        );
                        put(
                            format!("{p}mamba.conv1d.bias"),
                            rnd(&mut s, c.conv_dim(), 0.2),
                        );
                        put(format!("{p}mamba.dt_bias"), rnd(&mut s, c.n_heads, 0.6));
                        put(format!("{p}mamba.A_log"), rnd(&mut s, c.n_heads, 1.0));
                        put(format!("{p}mamba.D"), norm_w(&mut s, c.n_heads));
                        put(format!("{p}mamba.norm.weight"), norm_w(&mut s, c.d_inner()));
                        put(
                            format!("{p}mamba.out_proj.weight"),
                            rnd(&mut s, h * c.d_inner(), 0.6),
                        );
                    }
                },
            }
        }
        MapSource(m)
    }

    fn model(json: &str, seed: u64) -> HybridModel {
        let cfg = HybridConfig::from_hf_config(json).unwrap();
        let src = synth(&cfg, seed);
        HybridModel::load(cfg, &src, LoadOptions::default()).unwrap()
    }
    fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
    }
    const TOKS: [usize; 9] = [3, 7, 1, 12, 5, 9, 19, 0, 4];

    #[test]
    fn hybrid_models_use_the_accelerator_and_fall_back() {
        use crate::accel::testbe::Odd;
        let _g = crate::slicegpu::test_lock();
        for json in [JAMBA, BAMBA] {
            let mut m = model(json, 3);
            let cpu = m.forward_batch(&mut m.new_state(), &TOKS).unwrap();
            let rep = m.attach_backend(Odd::new(usize::MAX, 1.0, None));
            assert!(rep.active, "{rep:?}");
            let on = m.forward_batch(&mut m.new_state(), &TOKS).unwrap();
            assert!(close(&on, &cpu, 1e-3));
            m.attach_backend(Odd::new(2, 1.0, None));
            let on = m.forward_batch(&mut m.new_state(), &TOKS).unwrap();
            assert!(close(&on, &cpu, 1e-3));
            assert!(crate::slicegpu::failure().is_some());
        }
    }

    #[test]
    fn the_attention_layers_go_to_the_device_and_the_cpu_keeps_what_the_device_refuses() {
        use crate::accel::testbe::Odd;
        let _g = crate::slicegpu::test_lock();
        // full rotary dimension: the attention layers are of the kind a device can take
        let json = BAMBA.replace(
            "\"partial_rotary_factor\":0.5",
            "\"partial_rotary_factor\":1.0",
        );
        let mut m = model(&json, 3);
        let cpu = m.forward_batch(&mut m.new_state(), &TOKS).unwrap();
        let cpu_steps: Vec<Vec<f32>> = {
            let mut st = m.new_state();
            TOKS.iter()
                .map(|&t| m.forward(&mut st, t).unwrap())
                .collect()
        };
        let rep = m.attach_backend(Odd::new(usize::MAX, 1.0, None));
        assert!(rep.active, "{rep:?}");
        let on = m.attention_model().offloaded();
        assert_eq!(on, [true, true], "{rep:?}");
        assert!(rep.total >= 14 && rep.matrices >= 14, "{rep:?}");
        let got = m.forward_batch(&mut m.new_state(), &TOKS).unwrap();
        assert!(close(&got, &cpu, 2e-3));
        let mut st = m.new_state();
        for (&t, want) in TOKS.iter().zip(&cpu_steps) {
            assert!(close(&m.forward(&mut st, t).unwrap(), want, 2e-3));
        }
        assert_eq!(st.pos, TOKS.len());
        // a device with no memory takes no attention layer: plain CPU numerics again
        let rep = m.attach_backend(Odd::new(usize::MAX, 1.0, Some(0)));
        assert!(m.attention_model().offloaded().is_empty(), "{rep:?}");
        assert!(
            rep.note.contains("no attention layer") || !rep.note.is_empty(),
            "{rep:?}"
        );
        let back = m.forward_batch(&mut m.new_state(), &TOKS).unwrap();
        assert!(close(&back, &cpu, 2e-3));
        // the state-space products still fell back or worked; nothing was lost
        let rep = m.attach_backend(Odd::new(usize::MAX, 1.0, None));
        assert!(rep.active);
        assert_eq!(m.attention_model().offloaded(), [true, true]);
        // a device that dies while the attention layers run gives an error, never wrong numbers
        let rep = m.attach_backend(Odd::new(40, 1.0, None));
        assert!(rep.active);
        let mut dead = false;
        for _ in 0..4 {
            match m.forward_batch(&mut m.new_state(), &TOKS) {
                Ok(l) => assert!(close(&l, &cpu, 2e-3)),
                Err(ModelError::Backend(_)) => dead = true,
                Err(e) => panic!("{e}"),
            }
        }
        assert!(dead || crate::slicegpu::failure().is_some());
        // reattaching to a working device recovers
        m.attach_backend(Odd::new(usize::MAX, 1.0, None));
        let again = m.forward_batch(&mut m.new_state(), &TOKS).unwrap();
        assert!(close(&again, &cpu, 2e-3));
    }

    #[test]
    fn jamba_config_gives_the_layer_pattern_and_no_rope() {
        let c = HybridConfig::from_hf_config(JAMBA).unwrap();
        assert_eq!(c.kinds, [Kind::Ssm, Kind::Attention, Kind::Ssm, Kind::Ssm]);
        assert_eq!((c.attention_layers(), c.ssm_layers()), (1, 3));
        assert_eq!(c.spec.support, Support::Hybrid);
        assert_eq!(c.spec.rope_scaling, RopeScaling::Nope);
        let SsmConfig::M1(m) = &c.ssm else { panic!() };
        assert_eq!(
            (
                m.d_model,
                m.d_inner,
                m.d_state,
                m.d_conv,
                m.dt_rank,
                m.bcdt_rms_eps
            ),
            (16, 32, 4, 3, 2, Some(1e-6))
        );
        // the documented Jamba defaults: one attention layer in eight, offset 4
        let d = HybridConfig::from_hf_config(r#"{"model_type":"jamba","hidden_size":32,"num_hidden_layers":16,"num_attention_heads":4,"intermediate_size":64,"num_experts":1}"#).unwrap();
        assert_eq!(
            d.kinds
                .iter()
                .enumerate()
                .filter(|(_, k)| **k == Kind::Attention)
                .map(|(i, _)| i)
                .collect::<Vec<_>>(),
            [4, 12]
        );
        let SsmConfig::M1(m) = &d.ssm else { panic!() };
        assert_eq!((m.d_state, m.d_conv, m.dt_rank), (16, 4, 2));
    }

    #[test]
    fn bamba_config_uses_indices_and_mamba2() {
        let c = HybridConfig::from_hf_config(BAMBA).unwrap();
        assert_eq!(
            c.kinds,
            [Kind::Ssm, Kind::Attention, Kind::Ssm, Kind::Attention]
        );
        let SsmConfig::M2(m) = &c.ssm else { panic!() };
        assert_eq!(
            (m.n_heads, m.head_dim, m.d_inner(), m.d_state, m.chunk_size),
            (4, 8, 32, 4, 4)
        );
        assert_eq!((c.spec.rotary_dim, c.spec.head_dim), (2, 4));
        assert_eq!(c.spec.rope_scaling, RopeScaling::None);
        let auto = BAMBA.replace(r#""mamba_d_head":8,"#, "");
        let SsmConfig::M2(m) = &HybridConfig::from_hf_config(&auto).unwrap().ssm else {
            panic!()
        };
        assert_eq!(
            m.head_dim, 8,
            "head dimension defaults to expand·hidden / heads"
        );
    }

    #[test]
    fn unusable_configs_are_errors_with_reasons() {
        let moe = JAMBA.replace(r#""num_experts":1"#, r#""num_experts":8"#);
        assert!(
            matches!(HybridConfig::from_hf_config(&moe), Err(FamilyError::Unsupported(w)) if w.contains("mixture-of-experts"))
        );
        assert!(matches!(
            HybridConfig::from_hf_config(
                r#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":1}"#
            ),
            Err(FamilyError::Unsupported(_))
        ));
        assert!(HybridConfig::from_hf_config("[1]").is_err());
        assert!(HybridConfig::from_hf_config(&BAMBA.replace("[1,3]", "[1,99]")).is_err());
        assert!(
            HybridConfig::from_hf_config(
                &JAMBA.replace(r#""num_hidden_layers":4"#, r#""num_hidden_layers":0"#)
            )
            .is_err()
        );
        assert!(
            HybridConfig::from_hf_config(
                &JAMBA.replace(r#""num_attention_heads":4"#, r#""num_attention_heads":5"#)
            )
            .is_err()
        );
        assert!(
            HybridConfig::from_hf_config(
                &JAMBA.replace(r#""hidden_size":16"#, r#""hidden_size":99999999999"#)
            )
            .is_err()
        );
        assert!(
            HybridConfig::from_hf_config(
                &JAMBA.replace(r#""mamba_d_state":4"#, r#""mamba_d_state":0"#)
            )
            .is_err()
        );
        // a bad layer-kind map is caught at load time
        let mut c = HybridConfig::from_hf_config(JAMBA).unwrap();
        c.kinds.pop();
        assert!(HybridModel::load(c, &MapSource(HashMap::new()), LoadOptions::default()).is_err());
    }

    #[test]
    fn prefill_equals_token_by_token_and_spans_chunk_boundaries() {
        for json in [JAMBA, BAMBA] {
            let m = model(json, 7);
            let toks: Vec<usize> = (0..150).map(|i| (i * 7 + 3) % 20).collect();
            let mut a = m.new_state();
            let la = m.forward_batch(&mut a, &toks).unwrap();
            let mut b = m.new_state();
            let mut lb = vec![];
            for &t in &toks {
                lb = m.forward(&mut b, t).unwrap();
            }
            assert!(close(&la, &lb, 2e-3), "{json:.30}");
            assert_eq!((a.pos, b.pos), (150, 150));
            // continuing after a batch equals continuing after single steps
            let (na, nb) = (m.forward(&mut a, 6).unwrap(), m.forward(&mut b, 6).unwrap());
            assert!(close(&na, &nb, 2e-3));
            assert!(la.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn all_attention_matches_the_cache_free_reference() {
        use crate::model::tests::reference_logits;
        let mut cfg = HybridConfig::from_hf_config(BAMBA).unwrap();
        cfg.kinds = vec![Kind::Attention; 4];
        let src = synth(&cfg, 11);
        let m = HybridModel::load(cfg, &src, LoadOptions::default()).unwrap();
        let mut st = m.new_state();
        let got = m.forward_batch(&mut st, &TOKS).unwrap();
        assert!(close(
            &got,
            &reference_logits(m.attention_model(), &TOKS),
            1e-4
        ));
        assert_eq!(st.ssm_bytes(), 0);
    }

    #[test]
    fn all_ssm_matches_a_plain_step_by_step_composition() {
        let mut cfg = HybridConfig::from_hf_config(JAMBA).unwrap();
        cfg.kinds = vec![Kind::Ssm; 4];
        let src = synth(&cfg, 5);
        let m = HybridModel::load(cfg.clone(), &src, LoadOptions::default()).unwrap();
        let mut st = m.new_state();
        let got = m.forward_batch(&mut st, &TOKS).unwrap();
        assert_eq!(st.kv.kv_bytes(), 0);

        // independent oracle: the tensors straight from the source, the mixers stepped one token at a time
        let SsmConfig::M1(c) = &cfg.ssm else { panic!() };
        let (h, ff) = (16usize, 24usize);
        let rd = |n: &str| src.read(n).unwrap();
        let mixers: Vec<Mamba1> = (0..4)
            .map(|i| {
                let p = format!("model.layers.{i}.mamba.");
                let names = MambaNames {
                    in_proj: format!("{p}in_proj.weight"),
                    in_bias: format!("{p}in_proj.bias"),
                    conv_w: format!("{p}conv1d.weight"),
                    conv_b: format!("{p}conv1d.bias"),
                    x_proj: format!("{p}x_proj.weight"),
                    dt_proj: format!("{p}dt_proj.weight"),
                    dt_bias: format!("{p}dt_proj.bias"),
                    a: format!("{p}A_log"),
                    d: format!("{p}D"),
                    out_proj: format!("{p}out_proj.weight"),
                    out_bias: format!("{p}out_proj.bias"),
                    norm: String::new(),
                };
                Mamba1::load(c.clone(), &src, &names, false)
                    .unwrap()
                    .with_bcdt_weights(
                        rd(&format!("{p}dt_layernorm.weight")),
                        rd(&format!("{p}b_layernorm.weight")),
                        rd(&format!("{p}c_layernorm.weight")),
                    )
                    .unwrap()
            })
            .collect();
        let mut states: Vec<Mamba1State> = mixers.iter().map(|m| m.new_state()).collect();
        let embed = rd("model.embed_tokens.weight");
        let mut last = vec![];
        for &t in &TOKS {
            let mut x = embed[t * h..(t + 1) * h].to_vec();
            for (i, (mx, stt)) in mixers.iter().zip(states.iter_mut()).enumerate() {
                let p = format!("model.layers.{i}.");
                let mut nx = vec![0.0; h];
                rmsnorm(
                    &x,
                    &rd(&format!("{p}input_layernorm.weight")),
                    1e-6,
                    &mut nx,
                );
                let o = mx.step(stt, &nx).unwrap();
                x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
                rmsnorm(
                    &x,
                    &rd(&format!("{p}pre_ff_layernorm.weight")),
                    1e-6,
                    &mut nx,
                );
                let mv = |w: &[f32], cols: usize, v: &[f32]| -> Vec<f32> {
                    w.chunks(cols)
                        .map(|r| r.iter().zip(v).map(|(a, b)| a * b).sum())
                        .collect()
                };
                let (g, u) = (
                    mv(&rd(&format!("{p}feed_forward.gate_proj.weight")), h, &nx),
                    mv(&rd(&format!("{p}feed_forward.up_proj.weight")), h, &nx),
                );
                let act: Vec<f32> = (0..ff)
                    .map(|j| g[j] / (1.0 + (-g[j]).exp()) * u[j])
                    .collect();
                let d = mv(&rd(&format!("{p}feed_forward.down_proj.weight")), ff, &act);
                x.iter_mut().zip(&d).for_each(|(a, b)| *a += b);
            }
            last = x;
        }
        let mut nx = vec![0.0; h];
        rmsnorm(&last, &rd("model.final_layernorm.weight"), 1e-6, &mut nx);
        let head = rd("lm_head.weight");
        let want: Vec<f32> = head
            .chunks(h)
            .map(|r| r.iter().zip(&nx).map(|(a, b)| a * b).sum())
            .collect();
        assert!(close(&got, &want, 1e-3), "{got:?}\n{want:?}");
    }

    #[test]
    fn state_sizes_grow_with_attention_layers_only() {
        let m = model(JAMBA, 1);
        let mut st = m.new_state();
        let before = st.ssm_bytes();
        assert!(before > 0);
        m.forward_batch(&mut st, &TOKS[..4]).unwrap();
        let kv4 = st.bytes() - st.ssm_bytes();
        m.forward_batch(&mut st, &TOKS[4..]).unwrap();
        let kv9 = st.bytes() - st.ssm_bytes();
        assert_eq!(st.ssm_bytes(), before, "the recurrent state never grows");
        assert!(kv9 >= kv4);
        // 1 attention layer of 4: the cache is a quarter of what a pure transformer of the same size holds
        let mut cfg = HybridConfig::from_hf_config(JAMBA).unwrap();
        cfg.kinds = vec![Kind::Attention; 4];
        let src = synth(&cfg, 2);
        let full = HybridModel::load(cfg, &src, LoadOptions::default()).unwrap();
        let mut fs = full.new_state();
        full.forward_batch(&mut fs, &TOKS).unwrap();
        assert!(fs.kv.kv_bytes() >= 3 * st.kv.kv_bytes());
    }

    #[test]
    fn fork_branches_independently_and_reset_means_a_new_state() {
        let m = model(BAMBA, 3);
        let mut a = m.new_state();
        m.forward_batch(&mut a, &TOKS[..5]).unwrap();
        let mut b = a.fork().unwrap();
        let (la, lb) = (m.forward(&mut a, 2).unwrap(), m.forward(&mut b, 2).unwrap());
        assert_eq!(la, lb);
        let lb2 = m.forward(&mut b, 8).unwrap();
        let la2 = m.forward(&mut a, 11).unwrap();
        assert_ne!(la2, lb2);
        // the original is untouched by what the fork did
        let mut c = m.new_state();
        m.forward_batch(
            &mut c,
            &[TOKS[0], TOKS[1], TOKS[2], TOKS[3], TOKS[4], 2, 11],
        )
        .unwrap();
        assert!(close(
            &m.forward(&mut c, 1).unwrap(),
            &m.forward(&mut a, 1).unwrap(),
            2e-3
        ));
    }

    #[test]
    fn nope_ignores_the_rope_base_and_differs_from_rope() {
        let run = |theta: f64, nope: bool| {
            let mut cfg = HybridConfig::from_hf_config(JAMBA).unwrap();
            cfg.spec.rope_theta = theta;
            if !nope {
                cfg.spec.rope_scaling = RopeScaling::None;
            }
            let src = synth(&cfg, 9);
            let m = HybridModel::load(cfg, &src, LoadOptions::default()).unwrap();
            let mut st = m.new_state();
            m.forward_batch(&mut st, &TOKS).unwrap()
        };
        assert_eq!(
            run(10_000.0, true),
            run(500_000.0, true),
            "no position encoding: the base cannot matter"
        );
        assert!(
            run(10_000.0, true)
                .iter()
                .zip(run(10_000.0, false))
                .any(|(a, b)| (a - b).abs() > 1e-4)
        );
    }

    #[test]
    fn bad_input_and_damaged_checkpoints_are_errors_not_panics() {
        let m = model(JAMBA, 4);
        let mut st = m.new_state();
        assert_eq!(m.forward_batch(&mut st, &[]), Err(ModelError::NoTokens));
        assert_eq!(
            m.forward_batch(&mut st, &[1, 999]),
            Err(ModelError::Token(999))
        );
        let long: Vec<usize> = vec![1; 161];
        assert_eq!(
            m.forward_batch(&mut m.new_state(), &long),
            Err(ModelError::ContextFull(160))
        );
        // a state from another model
        let other = model(BAMBA, 4);
        assert!(m.forward(&mut other.new_state(), 1).is_err());

        let cfg = HybridConfig::from_hf_config(JAMBA).unwrap();
        let good = synth(&cfg, 4);
        for victim in [
            "model.layers.2.mamba.A_log",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.0.feed_forward.down_proj.weight",
            "model.layers.3.mamba.c_layernorm.weight",
            "model.final_layernorm.weight",
            "model.embed_tokens.weight",
        ] {
            let mut missing = MapSource(good.0.clone());
            missing.0.remove(victim);
            let err = HybridModel::load(cfg.clone(), &missing, LoadOptions::default())
                .err()
                .unwrap()
                .to_string();
            assert!(
                err.contains(victim.rsplit('.').nth(1).unwrap_or(victim)) || err.contains("model."),
                "{victim}: {err}"
            );
            let mut short = MapSource(good.0.clone());
            short.0.get_mut(victim).unwrap().pop();
            assert!(
                HybridModel::load(cfg.clone(), &short, LoadOptions::default()).is_err(),
                "truncated {victim}"
            );
        }
        // non-finite weights flow through as NaN logits, never a panic
        let mut nan = MapSource(good.0.clone());
        nan.0
            .get_mut("model.layers.0.mamba.in_proj.weight")
            .unwrap()[0] = f32::NAN;
        let mn = HybridModel::load(cfg, &nan, LoadOptions::default()).unwrap();
        let _ = mn.forward_batch(&mut mn.new_state(), &TOKS);
    }

    #[test]
    fn works_through_the_common_trait() {
        use crate::family::generate;
        use crate::model::{GenerateOptions, Sampler};
        let m = model(BAMBA, 6);
        let mut st = LanguageModel::new_state(&m);
        let mut out = vec![];
        let mut sampler = Sampler::new(0.0, 1, 1);
        let stats = generate(
            &m,
            &mut st,
            &TOKS[..3],
            &mut sampler,
            &GenerateOptions {
                max_tokens: 6,
                stop_tokens: vec![],
                cancel: None,
            },
            &mut |t| {
                out.push(t);
                true
            },
        )
        .unwrap();
        assert_eq!((stats.generated, out.len()), (6, 6));
        assert_eq!(LanguageModel::position(&m, &st), 3 + 5);
        assert_eq!(
            (LanguageModel::vocab_size(&m), LanguageModel::context(&m)),
            (20, 160)
        );
    }
}
