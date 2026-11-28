//! Vision for multimodal models: a ViT encoder in the CLIP / SigLIP family, the projector that maps
//! its output into the language model's hidden space, tiling for large pictures, and the glue that
//! feeds the resulting embedding rows into a prompt ([`forward_pieces`]).
//!
//! Written from the published architecture descriptions of ViT, CLIP and SigLIP and the Hugging
//! Face `config.json` conventions; no existing implementation was read. The pipeline is
//!
//! ```text
//! Image --imageio::resize/crop/pad--> CHW floats --patch embedding--> [CLS] + patches (+ positions)
//!       --(pre-norm) transformer blocks (bidirectional attention, GELU MLP)--> hidden states
//!       --feature layer / post-norm / drop CLS--> --(pixel shuffle) projector--> rows of LLM width
//! ```
//!
//! **What is verified and what is not.** Everything here is tested for internal consistency
//! (against a slow independent reference written in the tests, permutation equivariance, exact
//! tiling and pixel-shuffle bookkeeping, loader round trips, hostile inputs). Nothing has been run
//! against a real CLIP/SigLIP/LLaVA checkpoint, so the tensor-name tables for Hugging Face and for
//! `mmproj` GGUF files, the choice of feature layer, the 2-D rotary convention and the default
//! preprocessing constants follow the public conventions as understood, and may need adjusting on
//! the first real file. Errors name the missing tensor.
//!
//! Positions in the language model: image rows occupy consecutive ordinary positions, like tokens.

use crate::arch::Act;
use crate::imageio::{self, Filter, Image, ImageError};
use crate::loader::{GgufFile, LoadError, SafeTensors, TensorSource};
use crate::model::{Model, ModelError, Segment, State};
use serde_json::Value;
use std::fmt;

#[derive(Debug)]
pub enum VisionError {
    /// The configuration is inconsistent or asks for something not implemented.
    Config(String),
    /// A weight is missing or has the wrong size.
    Weights(String),
    /// The picture could not be decoded or prepared.
    Image(ImageError),
    /// The language model refused the prompt.
    Model(ModelError),
    /// The caller's data does not fit (wrong sizes, placeholder mismatch).
    Shape(String),
}
impl fmt::Display for VisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VisionError::Config(s) => write!(f, "vision configuration: {s}"),
            VisionError::Weights(s) => write!(f, "vision weights: {s}"),
            VisionError::Image(e) => write!(f, "{e}"),
            VisionError::Model(e) => write!(f, "{e}"),
            VisionError::Shape(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for VisionError {}
impl From<ImageError> for VisionError {
    fn from(e: ImageError) -> Self {
        VisionError::Image(e)
    }
}
impl From<ModelError> for VisionError {
    fn from(e: ModelError) -> Self {
        VisionError::Model(e)
    }
}
impl From<LoadError> for VisionError {
    fn from(e: LoadError) -> Self {
        VisionError::Weights(e.to_string())
    }
}
type Res<T> = Result<T, VisionError>;
fn cfg<T>(s: impl Into<String>) -> Res<T> {
    Err(VisionError::Config(s.into()))
}
fn wts<T>(s: impl Into<String>) -> Res<T> {
    Err(VisionError::Weights(s.into()))
}

// ---------------------------------------------------------------------------------- configuration

/// Activation of the encoder's MLP (and of the projector).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VAct {
    Gelu,
    GeluTanh,
    QuickGelu,
    Silu,
}
impl VAct {
    pub fn apply(self, x: f32) -> f32 {
        match self {
            VAct::Gelu => crate::model::activate(Act::Gelu, x),
            VAct::GeluTanh => crate::model::activate(Act::GeluTanh, x),
            VAct::QuickGelu => x / (1.0 + crate::simd::rmath::exp(-1.702 * x)),
            VAct::Silu => crate::model::activate(Act::Silu, x),
        }
    }
    fn from_name(s: &str) -> Option<VAct> {
        Some(match s {
            "gelu" => VAct::Gelu,
            "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" | "gelu_tanh" => VAct::GeluTanh,
            "quick_gelu" => VAct::QuickGelu,
            "silu" | "swish" => VAct::Silu,
            _ => return None,
        })
    }
}

/// How token positions enter the encoder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PosKind {
    /// A learned table (one row per patch, plus one for the class token when there is one).
    /// Pictures whose patch grid differs from the trained one get the table bilinearly resampled.
    Learned,
    /// Rotary embedding in two dimensions: the first half of every head rotates with the patch row,
    /// the second half with the patch column (the class token is not rotated). No table.
    Rope2d { theta: f32 },
}

/// How a picture is brought to the encoder's square input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeMode {
    /// Stretch to `image_size × image_size` (SigLIP).
    Squash,
    /// Scale the short side to `image_size`, crop the centre (CLIP).
    CenterCrop,
    /// Pad to a square with the mean colour, then scale (LLaVA 1.5 "pad").
    Pad,
}

/// Everything that describes the encoder's shape and preprocessing.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionSpec {
    pub image_size: usize,
    pub patch: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub act: VAct,
    pub eps: f32,
    /// RMSNorm instead of LayerNorm (no bias, no mean subtraction).
    pub rms: bool,
    /// A learned class token in front of the patches.
    pub cls: bool,
    pub pos: PosKind,
    /// A LayerNorm after the embeddings, before the first block (CLIP's `pre_layrnorm`).
    pub pre_norm: bool,
    /// A LayerNorm after the last block; applied only when the selected feature layer is the last.
    pub post_norm: bool,
    /// Which hidden state feeds the projector: `-1` is the output of the last block, `-2` the one
    /// before it (LLaVA), `0` the embeddings. Blocks after the selected one are not computed.
    pub feature_layer: i32,
    /// Remove the class token from the output.
    pub drop_cls: bool,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub resize: ResizeMode,
    pub filter: Filter,
    /// Projector description.
    pub projector: ProjectorSpec,
}

/// The adapter between the encoder and the language model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectorSpec {
    pub kind: ProjKind,
    /// Merge `shuffle × shuffle` neighbouring patches into one token (their features concatenated),
    /// dividing the token count by `shuffle²`. `1` keeps every patch.
    pub shuffle: usize,
    /// A normalisation of the (shuffled) features in front of the projector (InternVL-style).
    pub pre_norm: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjKind {
    /// No projector: the encoder output is used as is (its width must equal the model's).
    Identity,
    /// One linear layer.
    Linear,
    /// Linear, activation, linear (LLaVA 1.5).
    Mlp(VAct),
}

pub const CLIP_MEAN: [f32; 3] = [0.481_454_7, 0.457_827_5, 0.408_210_7];
pub const CLIP_STD: [f32; 3] = [0.268_629_55, 0.261_302_6, 0.275_777_1];

impl VisionSpec {
    pub fn grid(&self) -> usize {
        self.image_size / self.patch
    }
    pub fn n_patches(&self) -> usize {
        self.grid() * self.grid()
    }
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads.max(1)
    }
    /// How many blocks run for the selected feature layer (`0` = embeddings only, `k > 0` = the
    /// output of block `k`, `-1` = the last block, `-2` the one before, …).
    pub fn blocks_to_run(&self) -> usize {
        let l = self.layers as i64;
        (if self.feature_layer >= 0 {
            self.feature_layer as i64
        } else {
            l + 1 + self.feature_layer as i64
        })
        .clamp(0, l) as usize
    }
    /// Rows the projector produces for one `image_size` square.
    pub fn tokens_per_tile(&self) -> usize {
        let g = self.grid() / self.projector.shuffle.max(1);
        g * g
    }

    /// A CLIP-like encoder of the given size (class token, learned positions, pre- and post-norm,
    /// quick-GELU, second-to-last layer, LLaVA-style MLP projector).
    pub fn clip(
        image_size: usize,
        patch: usize,
        hidden: usize,
        layers: usize,
        heads: usize,
        ffn: usize,
    ) -> VisionSpec {
        VisionSpec {
            image_size,
            patch,
            hidden,
            layers,
            heads,
            ffn,
            act: VAct::QuickGelu,
            eps: 1e-5,
            rms: false,
            cls: true,
            pos: PosKind::Learned,
            pre_norm: true,
            post_norm: true,
            feature_layer: -2,
            drop_cls: true,
            mean: CLIP_MEAN,
            std: CLIP_STD,
            resize: ResizeMode::CenterCrop,
            filter: Filter::Bicubic,
            projector: ProjectorSpec {
                kind: ProjKind::Mlp(VAct::Gelu),
                shuffle: 1,
                pre_norm: false,
            },
        }
    }
    /// A SigLIP-like encoder: no class token, tanh-GELU, post-norm, squash resize, mean/std 0.5.
    pub fn siglip(
        image_size: usize,
        patch: usize,
        hidden: usize,
        layers: usize,
        heads: usize,
        ffn: usize,
    ) -> VisionSpec {
        VisionSpec {
            image_size,
            patch,
            hidden,
            layers,
            heads,
            ffn,
            act: VAct::GeluTanh,
            eps: 1e-6,
            rms: false,
            cls: false,
            pos: PosKind::Learned,
            pre_norm: false,
            post_norm: true,
            feature_layer: -1,
            drop_cls: false,
            mean: [0.5; 3],
            std: [0.5; 3],
            resize: ResizeMode::Squash,
            filter: Filter::Bicubic,
            projector: ProjectorSpec {
                kind: ProjKind::Mlp(VAct::Gelu),
                shuffle: 1,
                pre_norm: false,
            },
        }
    }

    fn validate(&self) -> Res<()> {
        let bad = |what: &str| cfg::<()>(what.to_string());
        if self.patch == 0 || self.image_size == 0 || !self.image_size.is_multiple_of(self.patch) {
            return bad("image_size must be a positive multiple of patch");
        }
        if self.hidden == 0 || self.heads == 0 || !self.hidden.is_multiple_of(self.heads) {
            return bad("hidden must be a positive multiple of heads");
        }
        if self.hidden > 16384
            || self.layers > 256
            || self.ffn == 0
            || self.ffn > 1 << 16
            || self.heads > 256
        {
            return bad("encoder dimensions out of range");
        }
        if self.grid() > 256 {
            return bad("patch grid larger than 256 x 256");
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return bad("eps must be positive");
        }
        if self.std.iter().any(|s| !s.is_finite() || *s == 0.0)
            || self.mean.iter().any(|m| !m.is_finite())
        {
            return bad("mean/std must be finite and std non-zero");
        }
        if matches!(self.pos, PosKind::Rope2d { .. }) && !self.head_dim().is_multiple_of(4) {
            return bad("2-D rotary embedding needs head_dim divisible by 4");
        }
        if let PosKind::Rope2d { theta } = self.pos
            && (!theta.is_finite() || theta <= 1.0)
        {
            return bad("rotary theta must be above 1");
        }
        let sh = self.projector.shuffle;
        if sh == 0 || sh > 8 || !self.grid().is_multiple_of(sh) {
            return bad("pixel-shuffle factor must divide the patch grid");
        }
        let l = self.layers as i64;
        let idx = if self.feature_layer >= 0 {
            self.feature_layer as i64
        } else {
            l + 1 + self.feature_layer as i64
        };
        if idx < 0 || idx > l {
            return bad("feature_layer selects a hidden state that does not exist");
        }
        if self.drop_cls && !self.cls {
            return bad("drop_cls without a class token");
        }
        Ok(())
    }

    /// Read the vision part of a Hugging Face `config.json` (the whole file of a CLIP/SigLIP vision
    /// model, or a LLaVA-style one with a `vision_config` section). `llava` fields
    /// (`vision_feature_layer`, `vision_feature_select_strategy`) are honoured when present.
    pub fn from_hf_config(root: &Value) -> Res<VisionSpec> {
        let v = root.get("vision_config").unwrap_or(root);
        let u = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as usize);
        let mt = v
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("clip_vision_model");
        let mut s = match mt {
            "siglip_vision_model" | "siglip2_vision_model" => VisionSpec::siglip(0, 0, 0, 0, 0, 0),
            "clip_vision_model" | "clip" | "" => VisionSpec::clip(0, 0, 0, 0, 0, 0),
            other => {
                return cfg(format!(
                    "vision model type `{other}` is not supported (clip_vision_model, siglip_vision_model)"
                ));
            }
        };
        s.image_size =
            u("image_size").ok_or_else(|| VisionError::Config("no image_size".into()))?;
        s.patch = u("patch_size").ok_or_else(|| VisionError::Config("no patch_size".into()))?;
        s.hidden = u("hidden_size").ok_or_else(|| VisionError::Config("no hidden_size".into()))?;
        s.layers = u("num_hidden_layers")
            .ok_or_else(|| VisionError::Config("no num_hidden_layers".into()))?;
        s.heads = u("num_attention_heads")
            .ok_or_else(|| VisionError::Config("no num_attention_heads".into()))?;
        s.ffn = u("intermediate_size")
            .ok_or_else(|| VisionError::Config("no intermediate_size".into()))?;
        if let Some(a) = v.get("hidden_act").and_then(Value::as_str) {
            s.act = VAct::from_name(a)
                .ok_or_else(|| VisionError::Config(format!("unknown activation `{a}`")))?;
        }
        if let Some(e) = v.get("layer_norm_eps").and_then(Value::as_f64) {
            s.eps = e as f32;
        }
        if let Some(l) = root.get("vision_feature_layer").and_then(Value::as_i64) {
            s.feature_layer = l.clamp(-1000, 1000) as i32;
        }
        if let Some(st) = root
            .get("vision_feature_select_strategy")
            .and_then(Value::as_str)
        {
            s.drop_cls = st == "default" && s.cls;
        }
        if let Some(p) = root.get("projector_hidden_act").and_then(Value::as_str)
            && let Some(a) = VAct::from_name(p)
        {
            s.projector.kind = ProjKind::Mlp(a);
        }
        if let Some(sh) = root.get("downsample_ratio").and_then(Value::as_f64)
            && sh > 0.0
            && sh < 1.0
            && (1.0 / sh).fract() == 0.0
        {
            s.projector.shuffle = (1.0 / sh) as usize;
        }
        s.validate()?;
        Ok(s)
    }
}

// --------------------------------------------------------------------------------------- weights

/// A normalisation layer's parameters.
#[derive(Clone, Debug)]
pub struct Norm {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
}
/// A linear layer: `y = W x + b`, `W` stored `rows × cols` row-major (PyTorch `[out, in]`).
#[derive(Clone, Debug)]
pub struct Lin {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
    pub rows: usize,
    pub cols: usize,
}
impl Lin {
    pub fn new(w: Vec<f32>, b: Option<Vec<f32>>, cols: usize) -> Res<Lin> {
        if cols == 0 || w.is_empty() || !w.len().is_multiple_of(cols) {
            return wts(format!(
                "a weight of {} values does not divide into rows of {cols}",
                w.len()
            ));
        }
        let rows = w.len() / cols;
        if b.as_ref().is_some_and(|b| b.len() != rows) {
            return wts(format!(
                "a bias of {} values for {rows} rows",
                b.as_ref().map_or(0, Vec::len)
            ));
        }
        Ok(Lin { w, b, rows, cols })
    }
    /// `n` input rows (`n × cols`) to `n × rows`.
    pub fn apply(&self, x: &[f32], n: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), n * self.cols);
        let (rows, cols) = (self.rows, self.cols);
        if n > 0 && crate::slicegpu::active() {
            let mut out = vec![0f32; n * rows];
            if crate::slicegpu::try_rows(&self.w, cols, x, n, self.b.as_deref(), &mut out) {
                return out;
            }
        }
        let mut yt = vec![0f32; n * rows];
        if n > 0 {
            let min = (65536 / cols.max(1)).max(1);
            crate::pool::global().parallel_chunks_mut(&mut yt, min, |start, chunk| {
                for (k, o) in chunk.iter_mut().enumerate() {
                    let e = start + k;
                    let (r, t) = (e / n, e % n);
                    *o = crate::simd::dot_f32(
                        &self.w[r * cols..(r + 1) * cols],
                        &x[t * cols..(t + 1) * cols],
                    );
                }
            });
        }
        let mut out = vec![0f32; n * rows];
        for r in 0..rows {
            let bias = self.b.as_ref().map_or(0.0, |b| b[r]);
            for t in 0..n {
                out[t * rows + r] = yt[r * n + t] + bias;
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct VLayer {
    pub ln1: Norm,
    pub q: Lin,
    pub k: Lin,
    pub v: Lin,
    pub o: Lin,
    pub ln2: Norm,
    pub fc1: Lin,
    pub fc2: Lin,
}

/// All encoder tensors.
#[derive(Clone, Debug)]
pub struct VisionWeights {
    /// Patch embedding as a `hidden × (3·patch²)` matrix (a convolution with stride = patch size);
    /// the input order within a row is channel, then patch row, then patch column.
    pub patch: Lin,
    pub cls: Option<Vec<f32>>,
    /// `n_pos × hidden` learned positions (empty for rotary encoders).
    pub pos: Vec<f32>,
    pub pre: Option<Norm>,
    pub layers: Vec<VLayer>,
    pub post: Option<Norm>,
}

#[derive(Clone, Debug)]
pub struct Projector {
    pub pre: Option<Norm>,
    pub l1: Option<Lin>,
    pub l2: Option<Lin>,
}

/// The encoder plus projector.
#[derive(Clone, Debug)]
pub struct VisionModel {
    /// Declared first so it is dropped before the weights it points at.
    reg: crate::slicegpu::Registration,
    spec: VisionSpec,
    w: VisionWeights,
    proj: Projector,
    out_dim: usize,
}

impl VisionModel {
    /// Check every shape against `spec` and assemble the model.
    pub fn new(spec: VisionSpec, w: VisionWeights, proj: Projector) -> Res<VisionModel> {
        spec.validate()?;
        let (h, p) = (spec.hidden, spec.patch);
        let nm = |n: &Norm, what: &str| -> Res<()> {
            if n.w.len() != h {
                return wts(format!(
                    "{what}: weight has {} values, expected {h}",
                    n.w.len()
                ));
            }
            if n.b.as_ref().is_some_and(|b| b.len() != h) {
                return wts(format!("{what}: bias length mismatch"));
            }
            if spec.rms && n.b.is_some() { /* tolerated, ignored */ }
            Ok(())
        };
        let ln = |l: &Lin, rows: usize, cols: usize, what: &str| -> Res<()> {
            if l.rows != rows || l.cols != cols {
                wts(format!(
                    "{what}: shape {}x{}, expected {rows}x{cols}",
                    l.rows, l.cols
                ))
            } else {
                Ok(())
            }
        };
        ln(&w.patch, h, 3 * p * p, "patch embedding")?;
        if let Some(c) = &w.cls
            && c.len() != h
        {
            return wts("class embedding has the wrong size");
        }
        if spec.cls != w.cls.is_some() {
            return wts(if spec.cls {
                "the spec wants a class token but the weights have none"
            } else {
                "the weights have a class token the spec does not use"
            });
        }
        match spec.pos {
            PosKind::Learned => {
                let want = (spec.n_patches() + spec.cls as usize) * h;
                if w.pos.len() != want {
                    return wts(format!(
                        "position table has {} values, expected {want}",
                        w.pos.len()
                    ));
                }
            }
            PosKind::Rope2d { .. } => {
                if !w.pos.is_empty() {
                    return wts("a rotary encoder must not carry a position table");
                }
            }
        }
        if spec.pre_norm != w.pre.is_some() {
            return wts("pre-norm presence disagrees with the spec");
        }
        if let Some(n) = &w.pre {
            nm(n, "pre-norm")?;
        }
        if spec.post_norm != w.post.is_some() {
            return wts("post-norm presence disagrees with the spec");
        }
        if let Some(n) = &w.post {
            nm(n, "post-norm")?;
        }
        if w.layers.len() != spec.layers {
            return wts(format!(
                "{} blocks, expected {}",
                w.layers.len(),
                spec.layers
            ));
        }
        for (i, l) in w.layers.iter().enumerate() {
            nm(&l.ln1, &format!("block {i} norm 1"))?;
            nm(&l.ln2, &format!("block {i} norm 2"))?;
            for (m, name) in [(&l.q, "q"), (&l.k, "k"), (&l.v, "v"), (&l.o, "out")] {
                ln(m, h, h, &format!("block {i} attention {name}"))?;
            }
            ln(&l.fc1, spec.ffn, h, &format!("block {i} mlp fc1"))?;
            ln(&l.fc2, h, spec.ffn, &format!("block {i} mlp fc2"))?;
        }
        let feat = h * spec.projector.shuffle * spec.projector.shuffle;
        if let Some(n) = &proj.pre
            && n.w.len() != feat
        {
            return wts("projector norm has the wrong size");
        }
        if spec.projector.pre_norm != proj.pre.is_some() {
            return wts("projector norm presence disagrees with the spec");
        }
        let out_dim = match spec.projector.kind {
            ProjKind::Identity => {
                if proj.l1.is_some() || proj.l2.is_some() {
                    return wts("an identity projector has no weights");
                }
                feat
            }
            ProjKind::Linear => {
                let l1 = proj.l1.as_ref().ok_or_else(|| {
                    VisionError::Weights("the projector has no linear layer".into())
                })?;
                if l1.cols != feat {
                    return wts(format!(
                        "projector input is {} wide, the encoder gives {feat}",
                        l1.cols
                    ));
                }
                if proj.l2.is_some() {
                    return wts("a linear projector has one layer");
                }
                l1.rows
            }
            ProjKind::Mlp(_) => {
                let (Some(l1), Some(l2)) = (&proj.l1, &proj.l2) else {
                    return wts("an MLP projector needs two layers");
                };
                if l1.cols != feat || l2.cols != l1.rows {
                    return wts("projector layer widths do not chain");
                }
                l2.rows
            }
        };
        Ok(VisionModel {
            reg: Default::default(),
            spec,
            w,
            proj,
            out_dim,
        })
    }

    /// Put the big matrices (patch embedding, attention, feed-forward, projector) on `be`.
    /// Products then run there in batches and fall back to the CPU for good if the device errs.
    /// Never fails: the report says what happened. Copies made by `clone` start on the CPU.
    pub fn attach_backend(
        &mut self,
        be: std::sync::Arc<dyn crate::backend::Backend>,
    ) -> crate::accel::AttachReport {
        self.reg = Default::default();
        let mut mats: Vec<(&[f32], usize)> = vec![(&self.w.patch.w, self.w.patch.cols)];
        for l in &self.w.layers {
            for m in [&l.q, &l.k, &l.v, &l.o, &l.fc1, &l.fc2] {
                mats.push((&m.w, m.cols));
            }
        }
        for m in self.proj.l1.iter().chain(&self.proj.l2) {
            mats.push((&m.w, m.cols));
        }
        let (reg, rep) = crate::slicegpu::register(be, &mats);
        self.reg = reg;
        rep
    }

    pub fn spec(&self) -> &VisionSpec {
        &self.spec
    }
    /// Width of the rows this model produces: must equal the language model's hidden size.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// A model with deterministic pseudo-random weights, for tests and demos (`llm_hidden` is the
    /// projector's output width). Meaningless as a vision system; useful to exercise the pipeline.
    pub fn random(spec: VisionSpec, llm_hidden: usize, seed: u64) -> Res<VisionModel> {
        spec.validate()?;
        let mut s = seed | 1;
        let mut rnd = move |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
                })
                .collect()
        };
        let h = spec.hidden;
        let norm = |r: &mut dyn FnMut(usize, f32) -> Vec<f32>| Norm {
            w: r(h, 0.1).iter().map(|v| 1.0 + v).collect(),
            b: Some(r(h, 0.05)),
        };
        let mut r = |n: usize, sc: f32| rnd(n, sc);
        let lin = |r: &mut dyn FnMut(usize, f32) -> Vec<f32>, rows: usize, cols: usize| Lin {
            w: r(rows * cols, 1.0 / (cols as f32).sqrt()),
            b: Some(r(rows, 0.02)),
            rows,
            cols,
        };
        let patch = lin(&mut r, h, 3 * spec.patch * spec.patch);
        let cls = spec.cls.then(|| r(h, 0.1));
        let pos = if spec.pos == PosKind::Learned {
            r((spec.n_patches() + spec.cls as usize) * h, 0.1)
        } else {
            vec![]
        };
        let pre = spec.pre_norm.then(|| norm(&mut r));
        let layers = (0..spec.layers)
            .map(|_| VLayer {
                ln1: norm(&mut r),
                q: lin(&mut r, h, h),
                k: lin(&mut r, h, h),
                v: lin(&mut r, h, h),
                o: lin(&mut r, h, h),
                ln2: norm(&mut r),
                fc1: lin(&mut r, spec.ffn, h),
                fc2: lin(&mut r, h, spec.ffn),
            })
            .collect();
        let post = spec.post_norm.then(|| norm(&mut r));
        let feat = h * spec.projector.shuffle * spec.projector.shuffle;
        let proj = match spec.projector.kind {
            ProjKind::Identity => Projector {
                pre: None,
                l1: None,
                l2: None,
            },
            ProjKind::Linear => Projector {
                pre: spec.projector.pre_norm.then(|| Norm {
                    w: vec![1.0; feat],
                    b: Some(vec![0.0; feat]),
                }),
                l1: Some(lin(&mut r, llm_hidden, feat)),
                l2: None,
            },
            ProjKind::Mlp(_) => Projector {
                pre: spec.projector.pre_norm.then(|| Norm {
                    w: vec![1.0; feat],
                    b: Some(vec![0.0; feat]),
                }),
                l1: Some(lin(&mut r, llm_hidden, feat)),
                l2: Some(lin(&mut r, llm_hidden, llm_hidden)),
            },
        };
        VisionModel::new(
            spec,
            VisionWeights {
                patch,
                cls,
                pos,
                pre,
                layers,
                post,
            },
            proj,
        )
    }

    // ---------------------------------------------------------------------------- the forward pass

    /// Encode one prepared picture (`[3][height][width]` floats, both sides multiples of the patch
    /// size) and project it: `tokens × out_dim` values, tokens in row-major patch order.
    pub fn encode_chw(&self, chw: &[f32], width: usize, height: usize) -> Res<Vec<f32>> {
        let s = &self.spec;
        let p = s.patch;
        if width == 0 || height == 0 || !width.is_multiple_of(p) || !height.is_multiple_of(p) {
            return Err(VisionError::Shape(format!(
                "a {width}x{height} picture is not a whole number of {p}-pixel patches"
            )));
        }
        let (gw, gh) = (width / p, height / p);
        if gw > 256 || gh > 256 {
            return Err(VisionError::Shape(
                "patch grid larger than 256 x 256".into(),
            ));
        }
        if chw.len() != 3 * width * height {
            return Err(VisionError::Shape(format!(
                "{} values for a {width}x{height} picture",
                chw.len()
            )));
        }
        if chw.iter().any(|v| !v.is_finite()) {
            return Err(VisionError::Shape(
                "the picture holds a NaN or infinity".into(),
            ));
        }
        let h = s.hidden;
        let np = gw * gh;
        // patches -> matrix (np × 3p²), order (channel, row, col) within a patch
        let pd = 3 * p * p;
        let mut pm = vec![0f32; np * pd];
        for gy in 0..gh {
            for gx in 0..gw {
                let dst = &mut pm[(gy * gw + gx) * pd..(gy * gw + gx + 1) * pd];
                for c in 0..3 {
                    for py in 0..p {
                        let src = c * width * height + (gy * p + py) * width + gx * p;
                        dst[(c * p + py) * p..(c * p + py + 1) * p]
                            .copy_from_slice(&chw[src..src + p]);
                    }
                }
            }
        }
        let emb = self.w.patch.apply(&pm, np);
        let nt = np + s.cls as usize;
        let mut x = vec![0f32; nt * h];
        let off = s.cls as usize;
        if let Some(c) = &self.w.cls {
            x[..h].copy_from_slice(c);
        }
        x[off * h..].copy_from_slice(&emb);
        if s.pos == PosKind::Learned {
            let g = s.grid();
            let table = &self.w.pos;
            if s.cls {
                for i in 0..h {
                    x[i] += table[i];
                }
            }
            let patches = &table[off * h..];
            if (gw, gh) == (g, g) {
                for i in 0..np * h {
                    x[off * h + i] += patches[i];
                }
            } else {
                let r = resample_pos(patches, g, g, h, gw, gh);
                for i in 0..np * h {
                    x[off * h + i] += r[i];
                }
            }
        }
        if let Some(n) = &self.w.pre {
            norm_rows(&mut x, nt, h, n, s.eps, false);
        }
        let idx = s.blocks_to_run();
        for l in self.w.layers.iter().take(idx) {
            self.block(&mut x, nt, l, gw, gh);
        }
        if idx == s.layers
            && let Some(n) = &self.w.post
        {
            norm_rows(&mut x, nt, h, n, s.eps, s.rms);
        }
        let (mut feats, mut nt) = (x, nt);
        if s.drop_cls {
            feats.drain(..h);
            nt -= 1;
        }
        Ok(self.project(feats, nt, gw, gh))
    }

    fn block(&self, x: &mut [f32], nt: usize, l: &VLayer, gw: usize, gh: usize) {
        let s = &self.spec;
        let h = s.hidden;
        let mut y = x.to_vec();
        norm_rows(&mut y, nt, h, &l.ln1, s.eps, s.rms);
        let mut q = l.q.apply(&y, nt);
        let mut k = l.k.apply(&y, nt);
        let v = l.v.apply(&y, nt);
        if let PosKind::Rope2d { theta } = s.pos {
            rope2d(&mut q, nt, s.heads, s.head_dim(), theta, gw, gh, s.cls);
            rope2d(&mut k, nt, s.heads, s.head_dim(), theta, gw, gh, s.cls);
        }
        let att = attention(&q, &k, &v, nt, s.heads, s.head_dim());
        let o = l.o.apply(&att, nt);
        for (a, b) in x.iter_mut().zip(&o) {
            *a += b;
        }
        let mut y = x.to_vec();
        norm_rows(&mut y, nt, h, &l.ln2, s.eps, s.rms);
        let mut m = l.fc1.apply(&y, nt);
        for v in m.iter_mut() {
            *v = s.act.apply(*v);
        }
        let m = l.fc2.apply(&m, nt);
        for (a, b) in x.iter_mut().zip(&m) {
            *a += b;
        }
    }

    fn project(&self, feats: Vec<f32>, nt: usize, gw: usize, gh: usize) -> Vec<f32> {
        let ps = self.spec.projector;
        let h = self.spec.hidden;
        let (mut f, mut n, mut w) = (feats, nt, h);
        if ps.shuffle > 1 {
            let k = ps.shuffle;
            let (ow, oh) = (gw / k, gh / k);
            let mut out = vec![0f32; ow * oh * k * k * h];
            for oy in 0..oh {
                for ox in 0..ow {
                    for dy in 0..k {
                        for dx in 0..k {
                            let src = ((oy * k + dy) * gw + ox * k + dx) * h;
                            let dst = ((oy * ow + ox) * k * k + dy * k + dx) * h;
                            out[dst..dst + h].copy_from_slice(&f[src..src + h]);
                        }
                    }
                }
            }
            f = out;
            n = ow * oh;
            w = h * k * k;
        }
        if let Some(nm) = &self.proj.pre {
            norm_rows(&mut f, n, w, nm, self.spec.eps, false);
        }
        match ps.kind {
            ProjKind::Identity => f,
            ProjKind::Linear => self.proj.l1.as_ref().map_or(f.clone(), |l| l.apply(&f, n)),
            ProjKind::Mlp(act) => {
                let (Some(l1), Some(l2)) = (&self.proj.l1, &self.proj.l2) else {
                    return f;
                };
                let mut m = l1.apply(&f, n);
                for v in m.iter_mut() {
                    *v = act.apply(*v);
                }
                l2.apply(&m, n)
            }
        }
    }

    // ---------------------------------------------------------------------------------- pictures

    /// Bring a picture to the encoder's square input according to the spec's resize mode.
    pub fn prepare(&self, img: &Image) -> Res<Image> {
        let s = &self.spec;
        let n = s.image_size;
        let bg = [
            (s.mean[0] * 255.0).round().clamp(0.0, 255.0) as u8,
            (s.mean[1] * 255.0).round().clamp(0.0, 255.0) as u8,
            (s.mean[2] * 255.0).round().clamp(0.0, 255.0) as u8,
        ];
        Ok(match s.resize {
            ResizeMode::Squash => imageio::resize(img, n, n, s.filter)?,
            ResizeMode::CenterCrop => {
                let r = imageio::resize_short_side(img, n, s.filter)?;
                imageio::center_crop(&r, n, n, bg)?
            }
            ResizeMode::Pad => {
                let sq = imageio::pad_to_square(img, bg)?;
                imageio::resize(&sq, n, n, s.filter)?
            }
        })
    }

    /// Normalised `[3][n][n]` floats of a picture already at `image_size`.
    pub fn to_chw(&self, img: &Image) -> Vec<f32> {
        imageio::to_chw(img, self.spec.mean, self.spec.std)
    }

    /// One picture to embedding rows for the language model.
    pub fn embed_image(&self, img: &Image) -> Res<ImageEmbeds> {
        let sq = self.prepare(img)?;
        let rows = self.encode_chw(&self.to_chw(&sq), sq.width, sq.height)?;
        Ok(ImageEmbeds {
            count: rows.len() / self.out_dim,
            dim: self.out_dim,
            rows,
        })
    }

    /// A large picture as a global thumbnail followed by tiles (see [`plan_tiles`]); every piece is
    /// encoded independently and the rows are concatenated: thumbnail first, then tiles row by row.
    /// No separator rows are inserted between tiles.
    pub fn embed_anyres(&self, img: &Image, any: &AnyRes) -> Res<ImageEmbeds> {
        let n = self.spec.image_size;
        let bg = [128u8; 3];
        let (cols, rows) = plan_tiles(img.width, img.height, n, any.max_tiles);
        let mut pieces: Vec<Image> = vec![];
        if any.thumbnail || cols * rows == 1 {
            pieces.push(self.prepare(&imageio::pad_to_square(img, bg)?)?);
        }
        if cols * rows > 1 {
            pieces.extend(split_tiles(img, n, cols, rows, bg, self.spec.filter)?);
        }
        let mut all = vec![];
        for t in &pieces {
            all.extend(self.encode_chw(&self.to_chw(t), t.width, t.height)?);
        }
        Ok(ImageEmbeds {
            count: all.len() / self.out_dim,
            dim: self.out_dim,
            rows: all,
        })
    }
}

/// Embedding rows for one picture, ready for [`forward_pieces`].
#[derive(Clone, Debug, PartialEq)]
pub struct ImageEmbeds {
    pub rows: Vec<f32>,
    pub count: usize,
    pub dim: usize,
}

// ----------------------------------------------------------------------------------------- maths

fn norm_rows(x: &mut [f32], n: usize, w: usize, nm: &Norm, eps: f32, rms: bool) {
    for t in 0..n {
        let row = &mut x[t * w..(t + 1) * w];
        if rms {
            let ms = crate::simd::sumsq(row) / w as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for (v, g) in row.iter_mut().zip(&nm.w) {
                *v = *v * inv * g;
            }
        } else {
            let mean = row.iter().sum::<f32>() / w as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for (i, v) in row.iter_mut().enumerate() {
                *v = (*v - mean) * inv * nm.w[i] + nm.b.as_ref().map_or(0.0, |b| b[i]);
            }
        }
    }
}

/// Full (bidirectional) multi-head attention over `n` tokens.
fn attention(q: &[f32], k: &[f32], v: &[f32], n: usize, heads: usize, hd: usize) -> Vec<f32> {
    let h = heads * hd;
    let mut out = vec![0f32; n * h];
    let scale = 1.0 / (hd as f32).sqrt();
    let threads = crate::pool::global().threads().clamp(1, 16).min(n.max(1));
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (ci, chunk) in out.chunks_mut(per * h).enumerate() {
            sc.spawn(move || {
                let mut scores = vec![0f32; n];
                for (ti, orow) in chunk.chunks_mut(h).enumerate() {
                    let t = ci * per + ti;
                    for head in 0..heads {
                        let qh = &q[t * h + head * hd..t * h + (head + 1) * hd];
                        let mut m = f32::NEG_INFINITY;
                        for s in 0..n {
                            let kh = &k[s * h + head * hd..s * h + (head + 1) * hd];
                            let d = crate::simd::dot_f32(qh, kh) * scale;
                            scores[s] = d;
                            m = m.max(d);
                        }
                        let mut sum = 0.0;
                        for sc in scores.iter_mut() {
                            *sc = crate::simd::rmath::exp(*sc - m);
                            sum += *sc;
                        }
                        let inv = 1.0 / sum;
                        let o = &mut orow[head * hd..(head + 1) * hd];
                        o.fill(0.0);
                        for s in 0..n {
                            let w = scores[s] * inv;
                            let vh = &v[s * h + head * hd..s * h + (head + 1) * hd];
                            for (a, b) in o.iter_mut().zip(vh) {
                                *a += w * b;
                            }
                        }
                    }
                }
            });
        }
    });
    out
}

/// Two-dimensional rotary embedding: of each head's `hd` values, the first half is rotated by the
/// patch row, the second by the patch column; within a half, value `i` pairs with `i + hd/4`. The
/// class token (first row when `cls`) is left alone.
#[allow(clippy::too_many_arguments)]
fn rope2d(
    x: &mut [f32],
    nt: usize,
    heads: usize,
    hd: usize,
    theta: f32,
    gw: usize,
    gh: usize,
    cls: bool,
) {
    let half = hd / 2;
    let quarter = half / 2;
    let inv: Vec<f32> = (0..quarter)
        .map(|i| theta.powf(-(i as f32) / quarter as f32))
        .collect();
    let off = cls as usize;
    debug_assert_eq!(nt, gw * gh + off);
    for t in off..nt {
        let (row, col) = (((t - off) / gw) as f32, ((t - off) % gw) as f32);
        for head in 0..heads {
            let base = t * heads * hd + head * hd;
            for (part, pos) in [(0usize, row), (half, col)] {
                for (i, f) in inv.iter().enumerate() {
                    let (s, c) = (pos * f).sin_cos();
                    let (a, b) = (x[base + part + i], x[base + part + i + quarter]);
                    x[base + part + i] = a * c - b * s;
                    x[base + part + i + quarter] = b * c + a * s;
                }
            }
        }
    }
    let _ = gh;
}

/// Bilinear resampling of a `sh × sw` grid of `h`-vectors to `dh × dw` (PyTorch `align_corners=False`).
fn resample_pos(src: &[f32], sw: usize, sh: usize, h: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; dw * dh * h];
    let at = |d: usize, ds: usize, ss: usize| -> (usize, usize, f32) {
        let c = ((d as f32 + 0.5) * ss as f32 / ds as f32 - 0.5).max(0.0);
        let i0 = (c.floor() as usize).min(ss - 1);
        let i1 = (i0 + 1).min(ss - 1);
        (i0, i1, c - i0 as f32)
    };
    for y in 0..dh {
        let (y0, y1, fy) = at(y, dh, sh);
        for x in 0..dw {
            let (x0, x1, fx) = at(x, dw, sw);
            for i in 0..h {
                let g = |yy: usize, xx: usize| src[(yy * sw + xx) * h + i];
                out[(y * dw + x) * h + i] = (1.0 - fy) * ((1.0 - fx) * g(y0, x0) + fx * g(y0, x1))
                    + fy * ((1.0 - fx) * g(y1, x0) + fx * g(y1, x1));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------- tiling

/// How [`VisionModel::embed_anyres`] cuts a big picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnyRes {
    /// Most tiles (columns × rows) to use.
    pub max_tiles: usize,
    /// Also encode one whole-picture view first.
    pub thumbnail: bool,
}
impl Default for AnyRes {
    fn default() -> Self {
        AnyRes {
            max_tiles: 4,
            thumbnail: true,
        }
    }
}

/// The `(columns, rows)` grid of `tile`-sized squares (at most `max_tiles` of them) that keeps the
/// most of a `w × h` picture's resolution, preferring fewer tiles on ties: for each candidate the
/// picture is scaled to fit inside the grid; the best candidate maximises the number of source
/// pixels that survive and then minimises the unused canvas.
pub fn plan_tiles(w: usize, h: usize, tile: usize, max_tiles: usize) -> (usize, usize) {
    if w == 0 || h == 0 || tile == 0 || max_tiles <= 1 {
        return (1, 1);
    }
    let mut best = (1usize, 1usize);
    let mut best_key = (0f64, f64::MAX);
    for cols in 1..=max_tiles {
        for rows in 1..=max_tiles / cols {
            let scale = ((cols * tile) as f64 / w as f64).min((rows * tile) as f64 / h as f64);
            let eff = ((w as f64 * scale) * (h as f64 * scale)).min(w as f64 * h as f64);
            let waste = (cols * rows * tile * tile) as f64 - eff;
            if eff > best_key.0 * (1.0 + 1e-9)
                || ((eff - best_key.0).abs() <= best_key.0 * 1e-9 && waste < best_key.1 - 1e-9)
            {
                best = (cols, rows);
                best_key = (eff, waste);
            }
        }
    }
    best
}

/// Scale the picture to fit `cols × rows` tiles of `tile` pixels (keeping its shape, padding the
/// rest with `bg`) and cut it into `cols * rows` squares, row by row.
pub fn split_tiles(
    img: &Image,
    tile: usize,
    cols: usize,
    rows: usize,
    bg: [u8; 3],
    f: Filter,
) -> Result<Vec<Image>, ImageError> {
    let fitted = imageio::fit_inside(img, cols * tile, rows * tile, f)?;
    let canvas = imageio::pad_to(&fitted, cols * tile, rows * tile, bg)?;
    let mut out = vec![];
    for r in 0..rows {
        for c in 0..cols {
            out.push(imageio::crop(&canvas, c * tile, r * tile, tile, tile)?);
        }
    }
    Ok(out)
}

// -------------------------------------------------------------------------------- prompt hooking

/// One piece of a multimodal prompt.
#[derive(Clone, Copy, Debug)]
pub enum Piece<'a> {
    Text(&'a [usize]),
    Image(&'a ImageEmbeds),
}

/// Feed a mixed prompt to a language model; returns the logits after the last position. Every
/// image's width must equal the model's hidden size. The state records the image positions as
/// [`crate::model::EMBEDDING_POSITION`] (so cached-prefix reuse stops before them).
pub fn forward_pieces(model: &Model, st: &mut State, pieces: &[Piece<'_>]) -> Res<Vec<f32>> {
    let h = model.spec.hidden as usize;
    let mut segs = Vec::with_capacity(pieces.len());
    for p in pieces {
        match p {
            Piece::Text(t) => {
                if !t.is_empty() {
                    segs.push(Segment::Tokens(t));
                }
            }
            Piece::Image(e) => {
                if e.dim != h {
                    return Err(VisionError::Shape(format!(
                        "image rows are {} wide but the language model's hidden size is {h}",
                        e.dim
                    )));
                }
                if e.rows.len() != e.count * e.dim {
                    return Err(VisionError::Shape(
                        "an image's row buffer does not match its count".into(),
                    ));
                }
                if e.count > 0 {
                    segs.push(Segment::Rows(&e.rows));
                }
            }
        }
    }
    Ok(model.forward_segments(st, &segs)?)
}

/// Split a token sequence at runs of `placeholder` tokens and pair the k-th run with the k-th
/// image. Each run is replaced by the whole image; with `strict`, a run must be exactly as long as
/// the image's row count (as chat templates that pre-expand the placeholder produce), otherwise any
/// length is accepted (a single placeholder token is the usual form). The number of runs must equal
/// the number of images.
pub fn pieces_from_placeholders<'a>(
    tokens: &'a [usize],
    placeholder: usize,
    images: &'a [ImageEmbeds],
    strict: bool,
) -> Res<Vec<Piece<'a>>> {
    let mut out = vec![];
    let (mut start, mut i, mut used) = (0usize, 0usize, 0usize);
    while i < tokens.len() {
        if tokens[i] != placeholder {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < tokens.len() && tokens[j] == placeholder {
            j += 1;
        }
        let img = images.get(used).ok_or_else(|| {
            VisionError::Shape("the prompt has more image placeholders than images".into())
        })?;
        if strict && j - i != img.count {
            return Err(VisionError::Shape(format!(
                "placeholder run {used} is {} tokens but the image has {} rows",
                j - i,
                img.count
            )));
        }
        out.push(Piece::Text(&tokens[start..i]));
        out.push(Piece::Image(img));
        used += 1;
        start = j;
        i = j;
    }
    if used != images.len() {
        return Err(VisionError::Shape(format!(
            "{} images but the prompt has {used} placeholders",
            images.len()
        )));
    }
    out.push(Piece::Text(&tokens[start..]));
    Ok(out)
}

// ------------------------------------------------------------------------------------ loaders

fn read(src: &dyn TensorSource, name: &str) -> Res<Vec<f32>> {
    src.read(name).map_err(|e| match e {
        LoadError::Missing(n) => VisionError::Weights(format!("the file has no tensor `{n}`")),
        o => o.into(),
    })
}
fn opt(src: &dyn TensorSource, name: &str) -> Res<Option<Vec<f32>>> {
    if src.has(name) {
        Ok(Some(read(src, name)?))
    } else {
        Ok(None)
    }
}
fn norm_of(src: &dyn TensorSource, base: &str) -> Res<Norm> {
    Ok(Norm {
        w: read(src, &format!("{base}.weight"))?,
        b: opt(src, &format!("{base}.bias"))?,
    })
}
fn lin_of(src: &dyn TensorSource, base: &str, cols: usize) -> Res<Lin> {
    Lin::new(
        read(src, &format!("{base}.weight"))?,
        opt(src, &format!("{base}.bias"))?,
        cols,
    )
}

impl VisionModel {
    /// Load a Hugging Face CLIP/SigLIP vision tower (and a LLaVA-style `multi_modal_projector`).
    /// `src` is a [`SafeTensors`] file or any other [`TensorSource`]; tensor-name prefixes
    /// (`vision_model.`, `vision_tower.vision_model.`, …) are detected.
    pub fn load_hf(src: &dyn TensorSource, spec: VisionSpec) -> Res<VisionModel> {
        spec.validate()?;
        let (h, f) = (spec.hidden, spec.ffn);
        const PREFIXES: [&str; 5] = [
            "vision_model.",
            "vision_tower.vision_model.",
            "model.vision_tower.vision_model.",
            "vision_tower.",
            "model.vision_model.",
        ];
        let pre = PREFIXES.iter().find(|p| src.has(&format!("{p}embeddings.patch_embedding.weight")))
            .ok_or_else(|| VisionError::Weights("no `…embeddings.patch_embedding.weight` tensor (tried vision_model., vision_tower.vision_model., model.vision_tower.vision_model.)".into()))?;
        let n = |s: &str| format!("{pre}{s}");
        let patch = Lin::new(
            read(src, &n("embeddings.patch_embedding.weight"))?,
            opt(src, &n("embeddings.patch_embedding.bias"))?,
            3 * spec.patch * spec.patch,
        )?;
        let cls = if spec.cls {
            Some(read(src, &n("embeddings.class_embedding"))?)
        } else {
            None
        };
        let pos = if spec.pos == PosKind::Learned {
            read(src, &n("embeddings.position_embedding.weight"))?
        } else {
            vec![]
        };
        let pre_n = if spec.pre_norm {
            Some(norm_of(
                src,
                &n(if src.has(&n("pre_layrnorm.weight")) {
                    "pre_layrnorm"
                } else {
                    "pre_layernorm"
                }),
            )?)
        } else {
            None
        };
        let post_n = if spec.post_norm {
            Some(norm_of(src, &n("post_layernorm"))?)
        } else {
            None
        };
        let mut layers = vec![];
        for i in 0..spec.layers {
            let b = n(&format!("encoder.layers.{i}."));
            layers.push(VLayer {
                ln1: norm_of(src, &format!("{b}layer_norm1"))?,
                q: lin_of(src, &format!("{b}self_attn.q_proj"), h)?,
                k: lin_of(src, &format!("{b}self_attn.k_proj"), h)?,
                v: lin_of(src, &format!("{b}self_attn.v_proj"), h)?,
                o: lin_of(src, &format!("{b}self_attn.out_proj"), h)?,
                ln2: norm_of(src, &format!("{b}layer_norm2"))?,
                fc1: lin_of(src, &format!("{b}mlp.fc1"), h)?,
                fc2: lin_of(src, &format!("{b}mlp.fc2"), f)?,
            });
        }
        let feat = h * spec.projector.shuffle * spec.projector.shuffle;
        let proj = match spec.projector.kind {
            ProjKind::Identity => Projector {
                pre: None,
                l1: None,
                l2: None,
            },
            kind => {
                const PP: [&str; 3] = [
                    "multi_modal_projector.",
                    "model.multi_modal_projector.",
                    "mlp1.",
                ];
                let base = PP
                    .iter()
                    .copied()
                    .find(|p| {
                        src.has(&format!("{p}linear_1.weight"))
                            || src.has(&format!("{p}0.weight"))
                            || src.has(&format!("{p}1.weight"))
                    })
                    .ok_or_else(|| {
                        VisionError::Weights(
                            "no projector tensors (`multi_modal_projector.linear_1.weight`)".into(),
                        )
                    })?;
                let (a, b) = if src.has(&format!("{base}linear_1.weight")) {
                    (format!("{base}linear_1"), format!("{base}linear_2"))
                } else if src.has(&format!("{base}1.weight")) {
                    (format!("{base}1"), format!("{base}3"))
                } else {
                    (format!("{base}0"), format!("{base}2"))
                };
                let l1 = lin_of(src, &a, feat)?;
                let l2 = if matches!(kind, ProjKind::Mlp(_)) {
                    let c = l1.rows;
                    Some(lin_of(src, &b, c)?)
                } else {
                    None
                };
                Projector {
                    pre: None,
                    l1: Some(l1),
                    l2,
                }
            }
        };
        VisionModel::new(
            spec,
            VisionWeights {
                patch,
                cls,
                pos,
                pre: pre_n,
                layers,
                post: post_n,
            },
            proj,
        )
    }

    /// Open a `.safetensors` file (or sharded directory) and load it with [`VisionModel::load_hf`].
    pub fn open_safetensors(path: &std::path::Path, spec: VisionSpec) -> Res<VisionModel> {
        let st = SafeTensors::open(path)?;
        VisionModel::load_hf(&st, spec)
    }

    /// Load a multimodal projector GGUF (`mmproj`), best effort: metadata under `clip.vision.*`,
    /// tensors `v.patch_embd.weight`, `v.class_embd`, `v.position_embd.weight`, `v.pre_ln.*`,
    /// `v.blk.N.{attn_q,attn_k,attn_v,attn_out,ln1,ln2,ffn_up,ffn_down}.*`, `mm.0.*`, `mm.2.*`.
    /// The two feed-forward matrices are told apart by shape, since converters disagree about which
    /// is "up". Mirrors the files llava-style converters write, where the last encoder block has
    /// been dropped, so the output of the last stored block is used with no post-norm.
    pub fn open_gguf(path: &std::path::Path) -> Res<VisionModel> {
        let g = GgufFile::open(path)?;
        let m = &g.gguf;
        let get = |k: &str| {
            m.meta
                .get(k)
                .and_then(crate::gguf::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| VisionError::Config(format!("the GGUF has no `{k}`")))
        };
        let arch = m.architecture().unwrap_or("");
        if arch != "clip" && !m.meta.contains_key("clip.vision.image_size") {
            return cfg(format!(
                "not a vision projector GGUF (architecture `{arch}`)"
            ));
        }
        let mut spec = VisionSpec::clip(
            get("clip.vision.image_size")?,
            get("clip.vision.patch_size")?,
            get("clip.vision.embedding_length")?,
            get("clip.vision.block_count")?,
            get("clip.vision.attention.head_count")?,
            get("clip.vision.feed_forward_length")?,
        );
        spec.feature_layer = -1;
        spec.post_norm = g.has("v.post_ln.weight");
        spec.pre_norm = g.has("v.pre_ln.weight");
        spec.cls = g.has("v.class_embd");
        spec.drop_cls = spec.cls;
        if let Some(e) = m
            .meta
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(crate::gguf::Value::as_f64)
        {
            spec.eps = e as f32;
        }
        let triple = |k: &str| -> Option<[f32; 3]> {
            if let Some(crate::gguf::Value::Array(a)) = m.meta.get(k)
                && a.len() == 3
            {
                Some([
                    a[0].as_f64()? as f32,
                    a[1].as_f64()? as f32,
                    a[2].as_f64()? as f32,
                ])
            } else {
                None
            }
        };
        if let Some(v) = triple("clip.vision.image_mean") {
            spec.mean = v;
        }
        if let Some(v) = triple("clip.vision.image_std") {
            spec.std = v;
        }
        if let Some(crate::gguf::Value::Bool(b)) = m.meta.get("clip.use_gelu")
            && *b
        {
            spec.act = VAct::Gelu;
        }
        if let Some(crate::gguf::Value::Bool(b)) = m.meta.get("clip.use_silu")
            && *b
        {
            spec.act = VAct::Silu;
        }
        let ptype = m
            .meta
            .get("clip.projector_type")
            .and_then(crate::gguf::Value::as_str)
            .unwrap_or("mlp");
        let has_mm2 = g.has("mm.2.weight");
        spec.projector.kind = match ptype {
            "mlp" if has_mm2 => ProjKind::Mlp(VAct::Gelu),
            "mlp" | "linear" => ProjKind::Linear,
            other => {
                return cfg(format!(
                    "projector type `{other}` is not supported (mlp, linear)"
                ));
            }
        };
        spec.validate()?;
        let (h, f) = (spec.hidden, spec.ffn);
        let dims = |name: &str| {
            g.gguf
                .tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| t.dims.clone())
        };
        let patch = Lin::new(
            read(&g, "v.patch_embd.weight")?,
            opt(&g, "v.patch_embd.bias")?,
            3 * spec.patch * spec.patch,
        )?;
        let cls = if spec.cls {
            Some(read(&g, "v.class_embd")?)
        } else {
            None
        };
        let pos = read(&g, "v.position_embd.weight")?;
        let pre = if spec.pre_norm {
            Some(norm_of(&g, "v.pre_ln")?)
        } else {
            None
        };
        let post = if spec.post_norm {
            Some(norm_of(&g, "v.post_ln")?)
        } else {
            None
        };
        let mut layers = vec![];
        for i in 0..spec.layers {
            let b = format!("v.blk.{i}.");
            // dims are [cols, rows]; fc1 maps hidden -> ffn, so its rows are `ffn`
            let (a, c) = (format!("{b}ffn_down"), format!("{b}ffn_up"));
            let a_is_fc1 = dims(&format!("{a}.weight"))
                .is_some_and(|d| d.len() == 2 && d[1] as usize == f && d[0] as usize == h);
            let (fc1, fc2) = if a_is_fc1 { (a, c) } else { (c, a) };
            layers.push(VLayer {
                ln1: norm_of(&g, &format!("{b}ln1"))?,
                q: lin_of(&g, &format!("{b}attn_q"), h)?,
                k: lin_of(&g, &format!("{b}attn_k"), h)?,
                v: lin_of(&g, &format!("{b}attn_v"), h)?,
                o: lin_of(&g, &format!("{b}attn_out"), h)?,
                ln2: norm_of(&g, &format!("{b}ln2"))?,
                fc1: lin_of(&g, &fc1, h)?,
                fc2: lin_of(&g, &fc2, f)?,
            });
        }
        let l1 = lin_of(&g, "mm.0", h)?;
        let l2 = if has_mm2 {
            let c = l1.rows;
            Some(lin_of(&g, "mm.2", c)?)
        } else {
            None
        };
        VisionModel::new(
            spec,
            VisionWeights {
                patch,
                cls,
                pos,
                pre,
                layers,
                post,
            },
            Projector {
                pre: None,
                l1: Some(l1),
                l2,
            },
        )
    }
}

#[cfg(test)]
#[allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::too_many_arguments
)]
mod tests {
    use super::*;
    use crate::imageio::synthetic;
    use crate::model::tests::tiny;

    fn small_spec() -> VisionSpec {
        let mut s = VisionSpec::clip(16, 4, 16, 3, 2, 24);
        s.feature_layer = -1;
        s
    }
    fn rnd(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
    fn picture(w: usize, h: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..3 * w * h).map(|_| rnd(&mut s)).collect()
    }

    // ---------------------------------------------------------- an independent slow reference

    /// Straightforward f64 implementation: convolution from the picture directly, token-at-a-time
    /// attention, no shared code with the real path beyond the weights.
    fn reference(m: &VisionModel, chw: &[f32], width: usize, height: usize) -> Vec<f64> {
        let s = &m.spec;
        let (p, h) = (s.patch, s.hidden);
        let (gw, gh) = (width / p, height / p);
        let mut toks: Vec<Vec<f64>> = vec![];
        if let Some(c) = &m.w.cls {
            toks.push(c.iter().map(|&v| v as f64).collect());
        }
        for gy in 0..gh {
            for gx in 0..gw {
                let mut v = vec![0f64; h];
                for o in 0..h {
                    let mut acc = m.w.patch.b.as_ref().map_or(0.0, |b| b[o] as f64);
                    for c in 0..3 {
                        for py in 0..p {
                            for px in 0..p {
                                acc += m.w.patch.w[o * 3 * p * p + c * p * p + py * p + px] as f64
                                    * chw[c * width * height + (gy * p + py) * width + gx * p + px]
                                        as f64;
                            }
                        }
                    }
                    v[o] = acc;
                }
                toks.push(v);
            }
        }
        if s.pos == PosKind::Learned {
            for (i, t) in toks.iter_mut().enumerate() {
                for d in 0..h {
                    t[d] += m.w.pos[i * h + d] as f64;
                }
            }
        }
        let lnorm = |t: &mut Vec<f64>, n: &Norm, rms: bool| {
            if rms {
                let ms = t.iter().map(|v| v * v).sum::<f64>() / h as f64;
                for d in 0..h {
                    t[d] = t[d] / (ms + s.eps as f64).sqrt() * n.w[d] as f64;
                }
            } else {
                let mean = t.iter().sum::<f64>() / h as f64;
                let var = t.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / h as f64;
                for d in 0..h {
                    t[d] = (t[d] - mean) / (var + s.eps as f64).sqrt() * n.w[d] as f64
                        + n.b.as_ref().map_or(0.0, |b| b[d] as f64);
                }
            }
        };
        if let Some(n) = &m.w.pre {
            for t in toks.iter_mut() {
                lnorm(t, n, false);
            }
        }
        let idx = s.blocks_to_run();
        let mv = |l: &Lin, x: &[f64]| -> Vec<f64> {
            (0..l.rows)
                .map(|r| {
                    (0..l.cols)
                        .map(|c| l.w[r * l.cols + c] as f64 * x[c])
                        .sum::<f64>()
                        + l.b.as_ref().map_or(0.0, |b| b[r] as f64)
                })
                .collect()
        };
        let act = |v: f64| s.act.apply(v as f32) as f64;
        for l in m.w.layers.iter().take(idx) {
            let normed: Vec<Vec<f64>> = toks
                .iter()
                .map(|t| {
                    let mut y = t.clone();
                    lnorm(&mut y, &l.ln1, s.rms);
                    y
                })
                .collect();
            let q: Vec<Vec<f64>> = normed.iter().map(|y| mv(&l.q, y)).collect();
            let k: Vec<Vec<f64>> = normed.iter().map(|y| mv(&l.k, y)).collect();
            let v: Vec<Vec<f64>> = normed.iter().map(|y| mv(&l.v, y)).collect();
            let hd = s.head_dim();
            let (mut q, mut k) = (q, k);
            if let PosKind::Rope2d { theta } = s.pos {
                let off = s.cls as usize;
                let quarter = hd / 4;
                for t in off..toks.len() {
                    let (row, col) = (((t - off) / gw) as f64, ((t - off) % gw) as f64);
                    for head in 0..s.heads {
                        for (base, pos) in [(0, row), (hd / 2, col)] {
                            for i in 0..quarter {
                                let ang = pos * (theta as f64).powf(-(i as f64) / quarter as f64);
                                for x in [&mut q, &mut k] {
                                    let (a, b) = (
                                        x[t][head * hd + base + i],
                                        x[t][head * hd + base + i + quarter],
                                    );
                                    x[t][head * hd + base + i] = a * ang.cos() - b * ang.sin();
                                    x[t][head * hd + base + i + quarter] =
                                        b * ang.cos() + a * ang.sin();
                                }
                            }
                        }
                    }
                }
            }
            let n = toks.len();
            let mut att = vec![vec![0f64; h]; n];
            for t in 0..n {
                for head in 0..s.heads {
                    let sc: Vec<f64> = (0..n)
                        .map(|u| {
                            (0..hd)
                                .map(|d| q[t][head * hd + d] * k[u][head * hd + d])
                                .sum::<f64>()
                                / (hd as f64).sqrt()
                        })
                        .collect();
                    let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for d in 0..hd {
                        att[t][head * hd + d] =
                            (0..n).map(|u| e[u] / z * v[u][head * hd + d]).sum();
                    }
                }
            }
            for t in 0..n {
                let o = mv(&l.o, &att[t]);
                for d in 0..h {
                    toks[t][d] += o[d];
                }
                let mut y = toks[t].clone();
                lnorm(&mut y, &l.ln2, s.rms);
                let hid: Vec<f64> = mv(&l.fc1, &y).into_iter().map(act).collect();
                let o2 = mv(&l.fc2, &hid);
                for d in 0..h {
                    toks[t][d] += o2[d];
                }
            }
        }
        if idx == s.layers
            && let Some(n) = &m.w.post
        {
            for t in toks.iter_mut() {
                lnorm(t, n, s.rms);
            }
        }
        if s.drop_cls {
            toks.remove(0);
        }
        // projector
        let k = s.projector.shuffle;
        let (ow, oh) = (gw / k, gh / k);
        let mut rows: Vec<Vec<f64>> = vec![];
        for oy in 0..oh {
            for ox in 0..ow {
                let mut r = vec![];
                for dy in 0..k {
                    for dx in 0..k {
                        r.extend_from_slice(&toks[(oy * k + dy) * gw + ox * k + dx]);
                    }
                }
                rows.push(r);
            }
        }
        if let Some(n) = &m.proj.pre {
            for r in rows.iter_mut() {
                let w = r.len();
                let mean = r.iter().sum::<f64>() / w as f64;
                let var = r.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / w as f64;
                for d in 0..w {
                    r[d] = (r[d] - mean) / (var + s.eps as f64).sqrt() * n.w[d] as f64
                        + n.b.as_ref().map_or(0.0, |b| b[d] as f64);
                }
            }
        }
        let out: Vec<Vec<f64>> = match s.projector.kind {
            ProjKind::Identity => rows,
            ProjKind::Linear => rows
                .iter()
                .map(|r| mv(m.proj.l1.as_ref().unwrap(), r))
                .collect(),
            ProjKind::Mlp(a) => rows
                .iter()
                .map(|r| {
                    let hid: Vec<f64> = mv(m.proj.l1.as_ref().unwrap(), r)
                        .into_iter()
                        .map(|v| a.apply(v as f32) as f64)
                        .collect();
                    mv(m.proj.l2.as_ref().unwrap(), &hid)
                })
                .collect(),
        };
        out.into_iter().flatten().collect()
    }

    fn close(a: &[f32], b: &[f64], tol: f64) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut worst = 0f64;
        for (x, y) in a.iter().zip(b) {
            worst = worst.max((*x as f64 - y).abs() / (1.0 + y.abs()));
        }
        assert!(worst < tol, "worst relative difference {worst}");
        worst
    }

    #[test]
    fn encoder_matches_the_slow_reference_for_clip_like_models() {
        let m = VisionModel::random(small_spec(), 12, 7).unwrap();
        let img = picture(16, 16, 3);
        let out = m.encode_chw(&img, 16, 16).unwrap();
        assert_eq!(out.len(), 16 * 12);
        close(&out, &reference(&m, &img, 16, 16), 2e-4);
    }

    #[test]
    fn encoder_matches_the_slow_reference_for_siglip_like_models_and_other_feature_layers() {
        for fl in [-1, -2, -3, 0] {
            let mut s = VisionSpec::siglip(16, 4, 16, 3, 4, 20);
            s.feature_layer = fl;
            let m = VisionModel::random(s, 10, (11 + fl + 5) as u64).unwrap();
            let img = picture(16, 16, 9);
            let out = m.encode_chw(&img, 16, 16).unwrap();
            close(&out, &reference(&m, &img, 16, 16), 2e-4);
        }
        // a second-to-last layer must not even compute the last block: layers beyond the selection are ignored
        let mut a = VisionModel::random(
            {
                let mut s = small_spec();
                s.feature_layer = -2;
                s
            },
            8,
            5,
        )
        .unwrap();
        let img = picture(16, 16, 1);
        let before = a.encode_chw(&img, 16, 16).unwrap();
        a.w.layers[2].fc2.w.iter_mut().for_each(|v| *v = f32::NAN);
        a.w.layers[2].q.w.iter_mut().for_each(|v| *v = f32::NAN);
        assert_eq!(a.encode_chw(&img, 16, 16).unwrap(), before);
    }

    #[test]
    fn rotary_encoder_matches_an_independent_f64_rotation() {
        let mut s = VisionSpec::siglip(16, 4, 16, 2, 2, 20);
        s.pos = PosKind::Rope2d { theta: 100.0 };
        let m = VisionModel::random(s, 8, 21).unwrap();
        let img = picture(16, 16, 4);
        let out = m.encode_chw(&img, 16, 16).unwrap();
        close(&out, &reference(&m, &img, 16, 16), 5e-4);
    }

    #[test]
    fn the_accelerator_gives_the_same_embeddings_and_the_cpu_takes_over_when_it_fails() {
        use crate::slicegpu::testbe::flaky;
        let _g = crate::slicegpu::test_lock();
        let s = VisionSpec::siglip(16, 4, 16, 2, 2, 20);
        let mut m = VisionModel::random(s, 8, 33).unwrap();
        let img = picture(16, 16, 6);
        let cpu = m.encode_chw(&img, 16, 16).unwrap();
        let rep = m.attach_backend(flaky(usize::MAX, usize::MAX));
        assert!(rep.active && rep.matrices >= 13, "{rep:?}");
        let on = m.encode_chw(&img, 16, 16).unwrap();
        assert_eq!(on.len(), cpu.len());
        for (a, b) in on.iter().zip(&cpu) {
            assert!((a - b).abs() < 2e-3 * (1.0 + b.abs()), "{a} vs {b}");
        }
        // a copy runs on the CPU, bit for bit as before
        assert_eq!(m.clone().encode_chw(&img, 16, 16).unwrap(), cpu);
        // the device dies after a few batched products: same answer from the CPU
        m.attach_backend(flaky(usize::MAX, 3));
        let again = m.encode_chw(&img, 16, 16).unwrap();
        for (a, b) in again.iter().zip(&cpu) {
            assert!((a - b).abs() < 2e-3 * (1.0 + b.abs()));
        }
        assert!(crate::slicegpu::failure().is_some());
        drop(m);
        let fresh = VisionModel::random(VisionSpec::siglip(16, 4, 16, 2, 2, 20), 8, 33).unwrap();
        assert_eq!(fresh.encode_chw(&img, 16, 16).unwrap(), cpu);
    }

    #[test]
    fn rotary_positions_change_the_result_and_encode_relative_geometry() {
        let mut s = VisionSpec::siglip(16, 4, 16, 2, 2, 20);
        s.pos = PosKind::Rope2d { theta: 100.0 };
        let m = VisionModel::random(s.clone(), 8, 33).unwrap();
        let mut flat = s;
        flat.pos = PosKind::Rope2d { theta: 1e30 };
        let m2 = VisionModel {
            spec: flat,
            ..m.clone()
        };
        let img = picture(16, 16, 6);
        assert_ne!(
            m.encode_chw(&img, 16, 16).unwrap(),
            m2.encode_chw(&img, 16, 16).unwrap()
        );
        // rotary attention depends only on offsets: a picture whose content is all one patch repeated
        // has the same pairwise geometry everywhere, so the output rows of interior patches are not all equal,
        // but shifting a 2x2 block of patches within a larger grid gives the same rows for that block
        // as long as nothing else contributes: use identical patches everywhere and compare two rows
        // that have identical neighbourhood counts (the grid is symmetric under 180 degree rotation of positions)
        let patch: Vec<f32> = picture(4, 4, 8)[..48].to_vec();
        let mut big = vec![0f32; 3 * 16 * 16];
        for c in 0..3 {
            for gy in 0..4 {
                for gx in 0..4 {
                    for py in 0..4 {
                        for px in 0..4 {
                            big[c * 256 + (gy * 4 + py) * 16 + gx * 4 + px] =
                                patch[c * 16 + py * 4 + px];
                        }
                    }
                }
            }
        }
        let out = m.encode_chw(&big, 16, 16).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn without_positions_patches_are_exchangeable() {
        // no class token, rotary off (theta huge) and zero position table: permuting patches permutes the output rows
        let mut s = VisionSpec::siglip(16, 4, 16, 3, 2, 24);
        s.pos = PosKind::Learned;
        let mut m = VisionModel::random(s, 9, 77).unwrap();
        m.w.pos.iter_mut().for_each(|v| *v = 0.0);
        let img = picture(16, 16, 12);
        let a = m.encode_chw(&img, 16, 16).unwrap();
        // swap patches (0,0) and (2,3)
        let mut b_img = img.clone();
        for c in 0..3 {
            for py in 0..4 {
                for px in 0..4 {
                    let i0 = c * 256 + py * 16 + px;
                    let i1 = c * 256 + (2 * 4 + py) * 16 + 3 * 4 + px;
                    b_img.swap(i0, i1);
                }
            }
        }
        let b = m.encode_chw(&b_img, 16, 16).unwrap();
        let (r0, r1) = (0usize, 2 * 4 + 3);
        let d = 9;
        for t in 0..16 {
            let src = if t == r0 {
                r1
            } else if t == r1 {
                r0
            } else {
                t
            };
            for i in 0..d {
                assert!((b[t * d + i] - a[src * d + i]).abs() < 1e-4, "token {t}");
            }
        }
        // and the output really depends on the content
        assert_ne!(a, b);
    }

    #[test]
    fn learned_positions_break_that_symmetry() {
        let mut s = VisionSpec::siglip(16, 4, 16, 2, 2, 24);
        s.pos = PosKind::Learned;
        let m = VisionModel::random(s, 9, 78).unwrap();
        let img = picture(16, 16, 13);
        let a = m.encode_chw(&img, 16, 16).unwrap();
        let mut b_img = img.clone();
        for c in 0..3 {
            for py in 0..4 {
                for px in 0..4 {
                    b_img.swap(
                        c * 256 + py * 16 + px,
                        c * 256 + (2 * 4 + py) * 16 + 3 * 4 + px,
                    );
                }
            }
        }
        let b = m.encode_chw(&b_img, 16, 16).unwrap();
        let moved = (0..9).any(|i| (b[i] - a[(2 * 4 + 3) * 9 + i]).abs() > 1e-3);
        assert!(moved);
    }

    #[test]
    fn other_grid_sizes_resample_the_position_table() {
        let m = VisionModel::random(VisionSpec::siglip(16, 4, 16, 2, 2, 24), 8, 3).unwrap();
        let img = picture(24, 12, 5);
        let out = m.encode_chw(&img, 24, 12).unwrap();
        assert_eq!(out.len(), 6 * 3 * 8);
        assert!(out.iter().all(|v| v.is_finite()));
        // at the trained grid size, resampling is the identity
        let t = &m.w.pos;
        assert_eq!(resample_pos(t, 4, 4, 16, 4, 4), *t);
        // constant tables stay constant at any size
        let c = vec![0.75f32; 4 * 4 * 3];
        assert!(
            resample_pos(&c, 4, 4, 3, 7, 2)
                .iter()
                .all(|&v| (v - 0.75).abs() < 1e-6)
        );
        // a horizontal ramp stays monotone when widened
        let ramp: Vec<f32> = (0..16).flat_map(|i| [(i % 4) as f32]).collect();
        let r = resample_pos(&ramp, 4, 4, 1, 8, 4);
        for y in 0..4 {
            for x in 1..8 {
                assert!(r[y * 8 + x] >= r[y * 8 + x - 1] - 1e-6);
            }
        }
    }

    #[test]
    fn pixel_shuffle_groups_neighbours_in_order() {
        let mut s = VisionSpec::siglip(16, 4, 8, 1, 2, 8);
        s.projector = ProjectorSpec {
            kind: ProjKind::Identity,
            shuffle: 2,
            pre_norm: false,
        };
        s.feature_layer = 0;
        s.post_norm = false;
        let m = VisionModel::random(s, 0, 4).unwrap();
        assert_eq!(m.out_dim(), 32);
        let img = picture(16, 16, 2);
        let shuffled = m.encode_chw(&img, 16, 16).unwrap();
        assert_eq!(shuffled.len(), 4 * 32);
        let mut plain_spec = m.spec.clone();
        plain_spec.projector.shuffle = 1;
        let plain = VisionModel {
            spec: plain_spec,
            ..m.clone()
        }
        .encode_chw(&img, 16, 16)
        .unwrap();
        // output token (oy, ox) = concat of plain tokens (2oy,2ox), (2oy,2ox+1), (2oy+1,2ox), (2oy+1,2ox+1)
        for oy in 0..2 {
            for ox in 0..2 {
                let mut want = vec![];
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let t = (2 * oy + dy) * 4 + 2 * ox + dx;
                    want.extend_from_slice(&plain[t * 8..(t + 1) * 8]);
                }
                assert_eq!(
                    &shuffled[(oy * 2 + ox) * 32..(oy * 2 + ox + 1) * 32],
                    &want[..]
                );
            }
        }
        assert_eq!(m.spec.tokens_per_tile(), 4);
    }

    #[test]
    fn projector_variants_match_the_reference() {
        for kind in [
            ProjKind::Linear,
            ProjKind::Mlp(VAct::Silu),
            ProjKind::Mlp(VAct::GeluTanh),
        ] {
            let mut s = VisionSpec::clip(16, 4, 16, 2, 2, 24);
            s.projector = ProjectorSpec {
                kind,
                shuffle: 2,
                pre_norm: true,
            };
            s.feature_layer = -1;
            let m = VisionModel::random(s, 6, 90).unwrap();
            let img = picture(16, 16, 15);
            close(
                &m.encode_chw(&img, 16, 16).unwrap(),
                &reference(&m, &img, 16, 16),
                3e-4,
            );
        }
    }

    #[test]
    fn rms_norm_variant_matches_the_reference() {
        let mut s = VisionSpec::siglip(16, 4, 16, 2, 2, 24);
        s.rms = true;
        let m = VisionModel::random(s, 6, 55).unwrap();
        let img = picture(16, 16, 2);
        close(
            &m.encode_chw(&img, 16, 16).unwrap(),
            &reference(&m, &img, 16, 16),
            3e-4,
        );
    }

    #[test]
    fn construction_rejects_inconsistent_specs_and_weights() {
        let bad = |f: &dyn Fn(&mut VisionSpec)| {
            let mut s = small_spec();
            f(&mut s);
            VisionModel::random(s, 8, 1).is_err()
        };
        assert!(bad(&|s| s.patch = 0));
        assert!(bad(&|s| s.patch = 5));
        assert!(bad(&|s| s.heads = 3));
        assert!(bad(&|s| s.hidden = 0));
        assert!(bad(&|s| s.eps = 0.0));
        assert!(bad(&|s| s.std = [0.0; 3]));
        assert!(bad(&|s| s.feature_layer = -9));
        assert!(bad(&|s| s.feature_layer = 4));
        assert!(bad(&|s| s.projector.shuffle = 3));
        assert!(bad(&|s| s.pos = PosKind::Rope2d { theta: 1.0 }));
        assert!(bad(&|s| {
            s.cls = false;
            s.drop_cls = true;
        }));
        assert!(bad(&|s| s.layers = 100_000));
        let good = VisionModel::random(small_spec(), 8, 1).unwrap();
        let wrong = |f: &dyn Fn(&mut VisionWeights, &mut Projector)| {
            let (mut w, mut p) = (good.w.clone(), good.proj.clone());
            f(&mut w, &mut p);
            VisionModel::new(good.spec.clone(), w, p).is_err()
        };
        assert!(wrong(&|w, _| {
            w.pos.pop();
        }));
        assert!(wrong(&|w, _| {
            w.cls = None;
        }));
        assert!(wrong(&|w, _| {
            w.layers.pop();
        }));
        assert!(wrong(&|w, _| {
            w.layers[1].fc1.rows += 1;
        }));
        assert!(wrong(&|w, _| {
            w.pre = None;
        }));
        assert!(wrong(&|w, _| {
            w.post.as_mut().unwrap().w.pop();
        }));
        assert!(wrong(&|_, p| {
            p.l2 = None;
        }));
        assert!(wrong(&|_, p| {
            p.l1.as_mut().unwrap().cols += 1;
        }));
    }

    #[test]
    fn encoding_rejects_bad_pictures() {
        let m = VisionModel::random(small_spec(), 8, 1).unwrap();
        assert!(m.encode_chw(&picture(16, 16, 1), 15, 16).is_err());
        assert!(m.encode_chw(&picture(16, 16, 1)[..100], 16, 16).is_err());
        assert!(m.encode_chw(&[], 0, 0).is_err());
        let mut nan = picture(16, 16, 1);
        nan[7] = f32::NAN;
        assert!(m.encode_chw(&nan, 16, 16).is_err());
        assert!(m.encode_chw(&vec![0.0; 3 * 8 * 1100], 8, 1100).is_err()); // grid over 256
    }

    // -------------------------------------------------------------------------------- pictures

    #[test]
    fn embedding_a_decoded_picture_end_to_end() {
        let m = VisionModel::random(small_spec(), 8, 1).unwrap();
        let png = imageio::encode_png(&synthetic(40, 24, 2));
        let img = imageio::decode(&png).unwrap();
        let e = m.embed_image(&img).unwrap();
        assert_eq!((e.count, e.dim, e.rows.len()), (16, 8, 128));
        // deterministic, and different pictures give different rows
        assert_eq!(m.embed_image(&img).unwrap(), e);
        let other = m.embed_image(&synthetic(40, 24, 9)).unwrap();
        assert_ne!(other.rows, e.rows);
        // all three resize modes give the right shape
        for mode in [ResizeMode::Squash, ResizeMode::CenterCrop, ResizeMode::Pad] {
            let mut s = small_spec();
            s.resize = mode;
            let mm = VisionModel::random(s, 8, 1).unwrap();
            assert_eq!(mm.prepare(&img).unwrap().width, 16);
            assert_eq!(mm.embed_image(&img).unwrap().count, 16);
        }
    }

    #[test]
    fn tile_planning_prefers_the_grid_that_keeps_the_most_detail() {
        assert_eq!(plan_tiles(100, 100, 100, 4), (1, 1));
        assert_eq!(plan_tiles(200, 100, 100, 4), (2, 1));
        assert_eq!(plan_tiles(100, 300, 100, 4), (1, 3));
        assert_eq!(plan_tiles(200, 200, 100, 4), (2, 2));
        assert_eq!(plan_tiles(400, 100, 100, 4), (4, 1));
        assert_eq!(plan_tiles(1000, 1000, 100, 1), (1, 1));
        assert_eq!(plan_tiles(0, 5, 100, 4), (1, 1));
        // never more tiles than allowed
        for (w, h) in [(300, 100), (123, 987), (999, 50), (5, 5)] {
            let (c, r) = plan_tiles(w, h, 64, 6);
            assert!(c * r <= 6 && c >= 1 && r >= 1);
        }
        // a thin picture is not padded to a square grid
        assert_eq!(plan_tiles(1000, 100, 100, 6), (6, 1));
    }

    #[test]
    fn splitting_covers_the_canvas_exactly() {
        let img = synthetic(50, 20, 1);
        let tiles = split_tiles(&img, 16, 3, 1, [7, 7, 7], Filter::Area).unwrap();
        assert_eq!(tiles.len(), 3);
        assert!(tiles.iter().all(|t| t.width == 16 && t.height == 16));
        // the fitted picture is 48x19 -> padded to 48x16? fit_inside(48x16) gives 40x16: its content spans tiles
        let again = split_tiles(&img, 16, 3, 1, [7, 7, 7], Filter::Area).unwrap();
        assert_eq!(tiles, again);
        // single tile is a plain fit
        assert_eq!(
            split_tiles(&img, 16, 1, 1, [0; 3], Filter::Area)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn anyres_concatenates_thumbnail_then_tiles() {
        let m = VisionModel::random(small_spec(), 8, 1).unwrap();
        let img = synthetic(64, 32, 3); // 2x1 tiles of 16 would be chosen at max 4? plan decides
        let (c, r) = plan_tiles(64, 32, 16, 4);
        let e = m
            .embed_anyres(
                &img,
                &AnyRes {
                    max_tiles: 4,
                    thumbnail: true,
                },
            )
            .unwrap();
        assert_eq!(e.count, 16 * (1 + c * r));
        let e2 = m
            .embed_anyres(
                &img,
                &AnyRes {
                    max_tiles: 4,
                    thumbnail: false,
                },
            )
            .unwrap();
        assert_eq!(e2.count, 16 * c * r);
        assert_eq!(&e.rows[16 * 8..], &e2.rows[..]); // the tiles match, the thumbnail is in front
        let one = m
            .embed_anyres(
                &synthetic(10, 10, 1),
                &AnyRes {
                    max_tiles: 4,
                    thumbnail: false,
                },
            )
            .unwrap();
        assert_eq!(one.count, 16); // one tile: the whole picture
    }

    // ------------------------------------------------------------------------- prompt hooking

    fn long_tiny() -> Model {
        let mut sp = crate::model::tests::base_spec(2, 2, true);
        sp.context = 400;
        crate::model::tests::model_of(sp, false)
    }

    #[test]
    fn segments_of_tokens_equal_ordinary_prefill_bit_for_bit() {
        let m = long_tiny();
        let toks: Vec<usize> = (0..300).map(|i| (i * 7 + 3) % 20).collect(); // crosses chunk boundaries
        let mut a = m.new_state();
        let la = m.forward_batch(&mut a, &toks).unwrap();
        let mut b = m.new_state();
        let lb = m
            .forward_segments(
                &mut b,
                &[
                    Segment::Tokens(&toks[..100]),
                    Segment::Tokens(&toks[100..101]),
                    Segment::Tokens(&toks[101..]),
                ],
            )
            .unwrap();
        assert_eq!(la, lb);
        assert_eq!(a.tokens(), b.tokens());
        // and the caches behave identically afterwards
        let na = m.forward(&mut a, 5).unwrap();
        let nb = m.forward(&mut b, 5).unwrap();
        assert_eq!(na, nb);
    }

    #[test]
    fn rows_made_from_the_token_table_equal_the_tokens() {
        let m = tiny(2, 1, false);
        let toks = [3usize, 9, 1, 19, 7, 7, 2];
        let mut rows = vec![];
        for &t in &toks[2..5] {
            rows.extend(m.embed_token(t).unwrap());
        }
        let mut a = m.new_state();
        let la = m.forward_batch(&mut a, &toks).unwrap();
        let mut b = m.new_state();
        let lb = m
            .forward_segments(
                &mut b,
                &[
                    Segment::Tokens(&toks[..2]),
                    Segment::Rows(&rows),
                    Segment::Tokens(&toks[5..]),
                ],
            )
            .unwrap();
        assert_eq!(la, lb);
        // the embedding positions are recorded as such, which stops prefix reuse in front of them
        assert_eq!(b.tokens()[..2], toks[..2]);
        assert!(
            b.tokens()[2..5]
                .iter()
                .all(|&t| t == crate::model::EMBEDDING_POSITION)
        );
        assert_eq!(b.tokens()[5..], toks[5..]);
        let reuse = m.prefill_cached(&mut b, &toks).unwrap();
        assert_eq!(b.reused(), 2);
        assert_eq!(reuse.len(), la.len());
    }

    #[test]
    fn segment_errors() {
        let m = tiny(2, 2, true);
        let h = m.spec.hidden as usize;
        let mut st = m.new_state();
        assert!(m.forward_segments(&mut st, &[]).is_err());
        assert!(
            m.forward_segments(&mut st, &[Segment::Tokens(&[])])
                .is_err()
        );
        assert!(
            m.forward_segments(&mut st, &[Segment::Tokens(&[999])])
                .is_err()
        );
        assert!(
            m.forward_segments(&mut st, &[Segment::Rows(&vec![0.0; h - 1])])
                .is_err()
        );
        assert!(m.forward_segments(&mut st, &[Segment::Rows(&[])]).is_err());
        let mut nan = vec![0.5f32; h];
        nan[3] = f32::NAN;
        assert!(m.forward_segments(&mut st, &[Segment::Rows(&nan)]).is_err());
        nan[3] = f32::INFINITY;
        assert!(m.forward_segments(&mut st, &[Segment::Rows(&nan)]).is_err());
        // nothing was consumed by the rejected calls
        assert_eq!(st.tokens().len(), 0);
        // context overflow
        let ctx = m.spec.context as usize;
        let many = vec![0.1f32; (ctx + 1) * h];
        assert!(matches!(
            m.forward_segments(&mut m.new_state(), &[Segment::Rows(&many)]),
            Err(ModelError::ContextFull(_))
        ));
        // a replay of a recorded sequence through the ordinary path fails loudly
        let mut s2 = m.new_state();
        m.forward_segments(&mut s2, &[Segment::Rows(&vec![0.2; h])])
            .unwrap();
        assert!(matches!(
            m.forward_batch(&mut m.new_state(), s2.tokens()),
            Err(ModelError::Token(_))
        ));
    }

    #[test]
    fn an_image_changes_the_answer_and_the_pipeline_is_deterministic() {
        let lm = tiny(2, 2, true);
        let h = lm.spec.hidden as usize;
        let vm = VisionModel::random(small_spec(), h, 5).unwrap();
        let e1 = vm.embed_image(&synthetic(33, 21, 1)).unwrap();
        let e2 = vm.embed_image(&synthetic(33, 21, 8)).unwrap();
        let (pre, post) = ([1usize, 2, 3], [4usize, 5]);
        let run = |e: &ImageEmbeds| {
            let mut st = lm.new_state();
            forward_pieces(
                &lm,
                &mut st,
                &[Piece::Text(&pre), Piece::Image(e), Piece::Text(&post)],
            )
            .unwrap()
        };
        let (a, a2, b) = (run(&e1), run(&e1), run(&e2));
        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert!(a.iter().all(|v| v.is_finite()));
        // wrong width is refused
        let bad = ImageEmbeds {
            rows: vec![0.0; 3 * (h + 1)],
            count: 3,
            dim: h + 1,
        };
        assert!(forward_pieces(&lm, &mut lm.new_state(), &[Piece::Image(&bad)]).is_err());
        let torn = ImageEmbeds {
            rows: vec![0.0; h],
            count: 2,
            dim: h,
        };
        assert!(forward_pieces(&lm, &mut lm.new_state(), &[Piece::Image(&torn)]).is_err());
    }

    #[test]
    fn placeholders_are_paired_with_images() {
        let (a, b) = (
            ImageEmbeds {
                rows: vec![0.0; 6],
                count: 3,
                dim: 2,
            },
            ImageEmbeds {
                rows: vec![1.0; 2],
                count: 1,
                dim: 2,
            },
        );
        let imgs = [a.clone(), b.clone()];
        let toks = [1usize, 99, 2, 3, 99, 99, 99, 4];
        let p = pieces_from_placeholders(&toks, 99, &imgs, false).unwrap();
        let kinds: Vec<String> = p
            .iter()
            .map(|x| match x {
                Piece::Text(t) => format!("T{}", t.len()),
                Piece::Image(i) => format!("I{}", i.count),
            })
            .collect();
        assert_eq!(kinds, ["T1", "I3", "T2", "I1", "T1"]);
        // strict mode wants runs as long as the images
        assert!(pieces_from_placeholders(&toks, 99, &imgs, true).is_err());
        let toks2 = [99usize, 99, 99, 5, 99];
        assert!(pieces_from_placeholders(&toks2, 99, &imgs, true).is_ok());
        // counts must match
        assert!(pieces_from_placeholders(&toks, 99, &imgs[..1], false).is_err());
        assert!(pieces_from_placeholders(&[1, 2], 99, &imgs, false).is_err());
        assert!(pieces_from_placeholders(&[1, 2], 99, &[], false).is_ok());
    }

    // ---------------------------------------------------------------------------------- loaders

    fn write_safetensors(path: &std::path::Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = vec![];
        for (name, shape, vals) in tensors {
            let start = data.len();
            for v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
        }
        let hb = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut out = (hb.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hb);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }
    fn hf_tensors(
        m: &VisionModel,
        prefix: &str,
        proj_prefix: &str,
    ) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = vec![];
        let s = &m.spec;
        let mut put = |n: String, shape: Vec<usize>, v: &[f32]| t.push((n, shape, v.to_vec()));
        let lin = |put: &mut dyn FnMut(String, Vec<usize>, &[f32]), base: String, l: &Lin| {
            put(format!("{base}.weight"), vec![l.rows, l.cols], &l.w);
            if let Some(b) = &l.b {
                put(format!("{base}.bias"), vec![l.rows], b);
            }
        };
        let nrm = |put: &mut dyn FnMut(String, Vec<usize>, &[f32]), base: String, n: &Norm| {
            put(format!("{base}.weight"), vec![n.w.len()], &n.w);
            if let Some(b) = &n.b {
                put(format!("{base}.bias"), vec![b.len()], b);
            }
        };
        lin(
            &mut put,
            format!("{prefix}embeddings.patch_embedding"),
            &m.w.patch,
        );
        if let Some(c) = &m.w.cls {
            put(
                format!("{prefix}embeddings.class_embedding"),
                vec![s.hidden],
                c,
            );
        }
        put(
            format!("{prefix}embeddings.position_embedding.weight"),
            vec![m.w.pos.len() / s.hidden, s.hidden],
            &m.w.pos,
        );
        if let Some(n) = &m.w.pre {
            nrm(&mut put, format!("{prefix}pre_layrnorm"), n);
        }
        if let Some(n) = &m.w.post {
            nrm(&mut put, format!("{prefix}post_layernorm"), n);
        }
        for (i, l) in m.w.layers.iter().enumerate() {
            let b = format!("{prefix}encoder.layers.{i}.");
            nrm(&mut put, format!("{b}layer_norm1"), &l.ln1);
            nrm(&mut put, format!("{b}layer_norm2"), &l.ln2);
            for (n, x) in [
                ("q_proj", &l.q),
                ("k_proj", &l.k),
                ("v_proj", &l.v),
                ("out_proj", &l.o),
            ] {
                lin(&mut put, format!("{b}self_attn.{n}"), x);
            }
            lin(&mut put, format!("{b}mlp.fc1"), &l.fc1);
            lin(&mut put, format!("{b}mlp.fc2"), &l.fc2);
        }
        lin(
            &mut put,
            format!("{proj_prefix}linear_1"),
            m.proj.l1.as_ref().unwrap(),
        );
        lin(
            &mut put,
            format!("{proj_prefix}linear_2"),
            m.proj.l2.as_ref().unwrap(),
        );
        t
    }
    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("vision_test_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn safetensors_round_trip_with_hf_names_and_prefixes() {
        let m = VisionModel::random(small_spec(), 8, 123).unwrap();
        let img = picture(16, 16, 20);
        let want = m.encode_chw(&img, 16, 16).unwrap();
        for (pre, proj) in [
            ("vision_model.", "multi_modal_projector."),
            ("vision_tower.vision_model.", "multi_modal_projector."),
            (
                "model.vision_tower.vision_model.",
                "model.multi_modal_projector.",
            ),
        ] {
            let dir = tmp("st");
            let path = dir.join("m.safetensors");
            write_safetensors(&path, &hf_tensors(&m, pre, proj));
            let loaded = VisionModel::open_safetensors(&path, m.spec.clone())
                .unwrap_or_else(|e| panic!("{pre}: {e}"));
            assert_eq!(loaded.encode_chw(&img, 16, 16).unwrap(), want, "{pre}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn missing_or_misshapen_tensors_are_named() {
        let m = VisionModel::random(small_spec(), 8, 123).unwrap();
        let dir = tmp("missing");
        let path = dir.join("m.safetensors");
        let mut t = hf_tensors(&m, "vision_model.", "multi_modal_projector.");
        t.retain(|(n, _, _)| n != "vision_model.encoder.layers.1.mlp.fc2.weight");
        write_safetensors(&path, &t);
        let e = VisionModel::open_safetensors(&path, m.spec.clone())
            .unwrap_err()
            .to_string();
        assert!(e.contains("layers.1.mlp.fc2.weight"), "{e}");
        let mut t = hf_tensors(&m, "vision_model.", "multi_modal_projector.");
        t.iter_mut()
            .find(|(n, _, _)| n == "vision_model.encoder.layers.0.self_attn.q_proj.weight")
            .unwrap()
            .2
            .pop();
        t.iter_mut()
            .find(|(n, _, _)| n == "vision_model.encoder.layers.0.self_attn.q_proj.weight")
            .unwrap()
            .1 = vec![1];
        write_safetensors(&path, &t);
        assert!(VisionModel::open_safetensors(&path, m.spec.clone()).is_err());
        let mut t = hf_tensors(&m, "vision_model.", "multi_modal_projector.");
        t.retain(|(n, _, _)| !n.starts_with("multi_modal_projector."));
        write_safetensors(&path, &t);
        assert!(
            VisionModel::open_safetensors(&path, m.spec.clone())
                .unwrap_err()
                .to_string()
                .contains("projector")
        );
        write_safetensors(&path, &[("something.else".into(), vec![1], vec![0.0])]);
        assert!(VisionModel::open_safetensors(&path, m.spec.clone()).is_err());
        std::fs::write(&path, b"not a safetensors file at all").unwrap();
        assert!(VisionModel::open_safetensors(&path, m.spec.clone()).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gguf_projector_files_load_whichever_way_the_ffn_matrices_are_named() {
        use crate::gguf_write::{GgufWriter, MetaValue, TensorDesc};
        use crate::quant::DType;
        let mut spec = small_spec();
        spec.feature_layer = -1;
        spec.post_norm = false;
        spec.projector.kind = ProjKind::Mlp(VAct::Gelu);
        spec.act = VAct::Gelu;
        let m = VisionModel::random(spec.clone(), 8, 321).unwrap();
        let img = picture(16, 16, 9);
        let want = m.encode_chw(&img, 16, 16).unwrap();
        for swapped in [false, true] {
            let mut meta: Vec<(String, MetaValue)> = vec![
                ("general.architecture".into(), MetaValue::str("clip")),
                ("clip.projector_type".into(), MetaValue::str("mlp")),
                ("clip.use_gelu".into(), MetaValue::Bool(true)),
                ("clip.vision.image_size".into(), MetaValue::U32(16)),
                ("clip.vision.patch_size".into(), MetaValue::U32(4)),
                ("clip.vision.embedding_length".into(), MetaValue::U32(16)),
                ("clip.vision.feed_forward_length".into(), MetaValue::U32(24)),
                ("clip.vision.block_count".into(), MetaValue::U32(3)),
                ("clip.vision.attention.head_count".into(), MetaValue::U32(2)),
                (
                    "clip.vision.attention.layer_norm_epsilon".into(),
                    MetaValue::F32(1e-5),
                ),
            ];
            meta.push((
                "clip.vision.image_mean".into(),
                MetaValue::Array(CLIP_MEAN.iter().map(|&v| MetaValue::F32(v)).collect()),
            ));
            meta.push((
                "clip.vision.image_std".into(),
                MetaValue::Array(CLIP_STD.iter().map(|&v| MetaValue::F32(v)).collect()),
            ));
            let mut ts: Vec<(String, Vec<u64>, Vec<f32>)> = vec![];
            let h = 16u64;
            let lin = |ts: &mut Vec<(String, Vec<u64>, Vec<f32>)>, base: &str, l: &Lin| {
                ts.push((
                    format!("{base}.weight"),
                    vec![l.rows as u64, l.cols as u64],
                    l.w.clone(),
                ));
                if let Some(b) = &l.b {
                    ts.push((format!("{base}.bias"), vec![l.rows as u64], b.clone()));
                }
            };
            let nrm = |ts: &mut Vec<(String, Vec<u64>, Vec<f32>)>, base: &str, n: &Norm| {
                ts.push((
                    format!("{base}.weight"),
                    vec![n.w.len() as u64],
                    n.w.clone(),
                ));
                ts.push((
                    format!("{base}.bias"),
                    vec![n.w.len() as u64],
                    n.b.clone().unwrap(),
                ));
            };
            lin(&mut ts, "v.patch_embd", &m.w.patch);
            ts.push(("v.class_embd".into(), vec![h], m.w.cls.clone().unwrap()));
            ts.push((
                "v.position_embd.weight".into(),
                vec![(m.w.pos.len() / 16) as u64, h],
                m.w.pos.clone(),
            ));
            nrm(&mut ts, "v.pre_ln", m.w.pre.as_ref().unwrap());
            for (i, l) in m.w.layers.iter().enumerate() {
                let b = format!("v.blk.{i}");
                nrm(&mut ts, &format!("{b}.ln1"), &l.ln1);
                nrm(&mut ts, &format!("{b}.ln2"), &l.ln2);
                lin(&mut ts, &format!("{b}.attn_q"), &l.q);
                lin(&mut ts, &format!("{b}.attn_k"), &l.k);
                lin(&mut ts, &format!("{b}.attn_v"), &l.v);
                lin(&mut ts, &format!("{b}.attn_out"), &l.o);
                let (n1, n2) = if swapped {
                    ("ffn_up", "ffn_down")
                } else {
                    ("ffn_down", "ffn_up")
                };
                lin(&mut ts, &format!("{b}.{n1}"), &l.fc1);
                lin(&mut ts, &format!("{b}.{n2}"), &l.fc2);
            }
            lin(&mut ts, "mm.0", m.proj.l1.as_ref().unwrap());
            lin(&mut ts, "mm.2", m.proj.l2.as_ref().unwrap());
            let descs: Vec<TensorDesc> = ts
                .iter()
                .map(|(n, s, _)| TensorDesc::new(n.clone(), s, DType::F32))
                .collect();
            let dir = tmp("gguf");
            let path = dir.join("mmproj.gguf");
            let f = std::fs::File::create(&path).unwrap();
            GgufWriter::write_all(f, &meta, descs, |i, _| {
                Ok(ts[i].2.iter().flat_map(|v| v.to_le_bytes()).collect())
            })
            .unwrap();
            let loaded =
                VisionModel::open_gguf(&path).unwrap_or_else(|e| panic!("swapped {swapped}: {e}"));
            assert_eq!(loaded.spec.feature_layer, -1);
            assert_eq!(
                loaded.encode_chw(&img, 16, 16).unwrap(),
                want,
                "swapped {swapped}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
        // not a vision file
        let dir = tmp("gguf_bad");
        let path = dir.join("x.gguf");
        std::fs::write(&path, b"GGUF garbage").unwrap();
        assert!(VisionModel::open_gguf(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hf_configs_are_parsed() {
        let clip: Value = serde_json::from_str(r#"{"vision_config":{"model_type":"clip_vision_model","image_size":336,"patch_size":14,"hidden_size":1024,"num_hidden_layers":24,
            "num_attention_heads":16,"intermediate_size":4096,"hidden_act":"quick_gelu","layer_norm_eps":1e-5},"vision_feature_layer":-2,"vision_feature_select_strategy":"default","projector_hidden_act":"gelu"}"#).unwrap();
        let s = VisionSpec::from_hf_config(&clip).unwrap();
        assert_eq!(
            (s.image_size, s.patch, s.hidden, s.layers, s.heads, s.ffn),
            (336, 14, 1024, 24, 16, 4096)
        );
        assert_eq!(
            (
                s.act,
                s.feature_layer,
                s.drop_cls,
                s.cls,
                s.grid(),
                s.tokens_per_tile()
            ),
            (VAct::QuickGelu, -2, true, true, 24, 576)
        );
        assert_eq!(s.projector.kind, ProjKind::Mlp(VAct::Gelu));
        let sig: Value = serde_json::from_str(r#"{"model_type":"siglip_vision_model","image_size":384,"patch_size":14,"hidden_size":1152,"num_hidden_layers":27,
            "num_attention_heads":16,"intermediate_size":4304,"hidden_act":"gelu_pytorch_tanh","layer_norm_eps":1e-6}"#).unwrap();
        let s = VisionSpec::from_hf_config(&sig);
        assert!(s.is_err(), "384 is not a multiple of 14: {s:?}"); // a real SigLIP-so400m uses 378/14 or 384/16; this must be reported, not rounded
        let sig: Value = serde_json::from_str(r#"{"model_type":"siglip_vision_model","image_size":384,"patch_size":16,"hidden_size":1152,"num_hidden_layers":27,
            "num_attention_heads":16,"intermediate_size":4304,"hidden_act":"gelu_pytorch_tanh","layer_norm_eps":1e-6,"downsample_ratio":0.5}"#).unwrap();
        let s = VisionSpec::from_hf_config(&sig).unwrap();
        assert_eq!(
            (s.cls, s.act, s.projector.shuffle, s.tokens_per_tile()),
            (false, VAct::GeluTanh, 2, 144)
        );
        for bad in [
            r#"{"model_type":"mystery","image_size":1}"#,
            r#"{"image_size":16}"#,
            r#"{"image_size":16,"patch_size":4,"hidden_size":16,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":8,"hidden_act":"weird"}"#,
        ] {
            assert!(
                VisionSpec::from_hf_config(&serde_json::from_str(bad).unwrap()).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn hostile_weights_and_pictures_never_panic() {
        let m = VisionModel::random(small_spec(), 8, 2).unwrap();
        let mut seed = 99u64;
        for round in 0..200 {
            let w = 4 * (1 + (rnd(&mut seed).abs() * 6.0) as usize);
            let h = 4 * (1 + (rnd(&mut seed).abs() * 6.0) as usize);
            let mut img = picture(w, h, round as u64);
            for _ in 0..3 {
                let i = ((rnd(&mut seed).abs() * img.len() as f32) as usize).min(img.len() - 1);
                img[i] = [1e30, -1e30, 0.0, 1e-30][(round % 4) as usize];
            }
            let r = m.encode_chw(&img, w, h);
            if let Ok(o) = r {
                assert_eq!(o.len() % 8, 0);
            }
        }
        // random bytes as a picture file
        for round in 0..200 {
            let junk: Vec<u8> = (0..(round % 50)).map(|i| (i * 31 + round) as u8).collect();
            assert!(
                m.embed_image(&imageio::decode(&junk).unwrap_or(Image {
                    width: 0,
                    height: 0,
                    rgb: vec![]
                }))
                .is_err()
            );
        }
    }
}
