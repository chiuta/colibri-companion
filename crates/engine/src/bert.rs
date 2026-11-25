//! BERT-family encoders: sentence embeddings and cross-encoder relevance scores.
//!
//! Covers the original BERT layout (post-layer-norm blocks, learned absolute positions, segment
//! embeddings, exact GELU) as stored by Hugging Face `transformers` (`BertModel`,
//! `BertForSequenceClassification`): MiniLM, bge-small/base/large v1.5, e5 (English), the ms-marco
//! cross-encoders and their relatives. Not covered: RoBERTa/XLM-R layouts (a different tokenizer and
//! position offsets), DistilBERT (different tensor names), relative-position variants.
//!
//! Tensor names are the `transformers` ones (`embeddings.word_embeddings.weight`,
//! `encoder.layer.N.attention.self.query.weight`, …, with or without a leading `bert.`), as
//! remembered from the library's public documentation. The maths is checked against a plain
//! f64 reference on small random models; no real checkpoint has been run.
//!
//! Weights are held as f32 (about four bytes per parameter).
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::loader::{LoadError, SafeTensors, TensorSource};
use crate::ops;
use serde_json::Value;
use std::path::Path;

#[derive(Debug, PartialEq, Eq)]
pub enum BertError {
    Load(String),
    Shape(String),
    Token(u32),
    TooLong { have: usize, max: usize },
    Empty,
    NoClassifier,
}
impl std::fmt::Display for BertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BertError::Load(s) => write!(f, "cannot load the encoder: {s}"),
            BertError::Shape(s) => write!(f, "the encoder does not match its configuration: {s}"),
            BertError::Token(t) => write!(f, "token {t} is outside the vocabulary"),
            BertError::TooLong { have, max } => {
                write!(f, "{have} tokens exceed the model's {max}-token window")
            }
            BertError::Empty => write!(f, "no tokens were given"),
            BertError::NoClassifier => write!(
                f,
                "this model has no classification head, so it cannot score a query against a document"
            ),
        }
    }
}
impl std::error::Error for BertError {}
impl From<LoadError> for BertError {
    fn from(e: LoadError) -> Self {
        BertError::Load(e.to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Gelu,
    GeluNew,
    Relu,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BertConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub max_pos: usize,
    pub type_vocab: usize,
    pub eps: f32,
    pub act: Act,
}

/// Most hidden units / layers / vocabulary accepted from a configuration (guards allocations).
const MAX_HIDDEN: u64 = 16_384;
const MAX_LAYERS: u64 = 128;
const MAX_VOCAB: u64 = 1 << 22;
const MAX_FFN: u64 = 1 << 16;

impl BertConfig {
    pub fn from_json(s: &str) -> Result<BertConfig, BertError> {
        let v: Value = serde_json::from_str(s).map_err(|e| BertError::Load(e.to_string()))?;
        let mt = v
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("bert");
        if mt != "bert" {
            return Err(BertError::Load(format!(
                "model type `{mt}` is not supported (BERT only)"
            )));
        }
        if let Some(p) = v.get("position_embedding_type").and_then(Value::as_str) {
            if p != "absolute" {
                return Err(BertError::Load(format!(
                    "position embedding type `{p}` is not supported"
                )));
            }
        }
        let n = |k: &str, max: u64| -> Result<usize, BertError> {
            let x = v
                .get(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| BertError::Load(format!("config has no `{k}`")))?;
            if x == 0 || x > max {
                return Err(BertError::Load(format!("`{k}` = {x} is out of range")));
            }
            Ok(x as usize)
        };
        let hidden = n("hidden_size", MAX_HIDDEN)?;
        let heads = n("num_attention_heads", 1024)?;
        if hidden % heads != 0 {
            return Err(BertError::Load(format!(
                "hidden_size {hidden} is not a multiple of num_attention_heads {heads}"
            )));
        }
        let act = match v
            .get("hidden_act")
            .and_then(Value::as_str)
            .unwrap_or("gelu")
        {
            "gelu" => Act::Gelu,
            "gelu_new" | "gelu_pytorch_tanh" | "gelu_fast" => Act::GeluNew,
            "relu" => Act::Relu,
            o => {
                return Err(BertError::Load(format!(
                    "activation `{o}` is not supported"
                )));
            }
        };
        Ok(BertConfig {
            vocab: n("vocab_size", MAX_VOCAB)?,
            hidden,
            layers: n("num_hidden_layers", MAX_LAYERS)?,
            heads,
            ffn: n("intermediate_size", MAX_FFN)?,
            max_pos: n("max_position_embeddings", 1 << 20)?,
            type_vocab: v
                .get("type_vocab_size")
                .and_then(Value::as_u64)
                .unwrap_or(2)
                .clamp(1, 64) as usize,
            eps: v
                .get("layer_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-12) as f32,
            act,
        })
    }
}

#[derive(Clone, Debug)]
struct Lin {
    w: Vec<f32>,
    b: Vec<f32>,
    rows: usize,
    cols: usize,
}
#[derive(Clone, Debug)]
struct Ln {
    g: Vec<f32>,
    b: Vec<f32>,
}
#[derive(Clone, Debug)]
struct Layer {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln1: Ln,
    up: Lin,
    down: Lin,
    ln2: Ln,
}

/// How a sequence of hidden states becomes one vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    /// The first token's final hidden state (`[CLS]`): bge, BERT sentence classifiers.
    Cls,
    /// The mean over all tokens: MiniLM, e5, most sentence-transformers models.
    Mean,
}

#[derive(Debug, Clone)]
pub struct Bert {
    pub cfg: BertConfig,
    word: Vec<f32>,
    pos: Vec<f32>,
    typ: Vec<f32>,
    ln_e: Ln,
    layers: Vec<Layer>,
    pooler: Option<Lin>,
    classifier: Option<Lin>,
}

fn layer_norm(x: &mut [f32], ln: &Ln, eps: f32) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var = x
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n;
    let inv = 1.0 / (var + f64::from(eps)).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((f64::from(*v) - mean) * inv) as f32 * ln.g[i] + ln.b[i];
    }
}

fn activate(a: Act, x: f32) -> f32 {
    match a {
        Act::Gelu => crate::audio::gelu(x),
        Act::GeluNew => {
            let x = f64::from(x);
            (0.5 * x
                * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh()))
                as f32
        }
        Act::Relu => x.max(0.0),
    }
}

/// `out[t] = W · x[t] + b` for every row `t` of `x` (`n` rows), split over threads by rows.
fn linear(l: &Lin, x: &[f32], n: usize, out: &mut [f32]) {
    let (r, c) = (l.rows, l.cols);
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let work = n.saturating_mul(r).saturating_mul(c);
    let one = |xs: &[f32], os: &mut [f32]| {
        for (xt, ot) in xs.chunks(c).zip(os.chunks_mut(r)) {
            ops::matvec(&l.w, c, xt, ot);
            for (o, b) in ot.iter_mut().zip(&l.b) {
                *o += b;
            }
        }
    };
    if threads <= 1 || n < 2 || work < (1 << 22) {
        return one(x, out);
    }
    let per = n.div_ceil(threads);
    std::thread::scope(|s| {
        for (xs, os) in x.chunks(per * c).zip(out.chunks_mut(per * r)) {
            let one = &one;
            s.spawn(move || one(xs, os));
        }
    });
}

impl Bert {
    /// Load `config.json` and the safetensors files of a Hugging Face folder.
    pub fn open(dir: &Path) -> Result<Bert, BertError> {
        let cfg = BertConfig::from_json(
            &std::fs::read_to_string(dir.join("config.json"))
                .map_err(|e| BertError::Load(format!("config.json: {e}")))?,
        )?;
        let st = SafeTensors::open(dir)?;
        Bert::from_source(cfg, &st)
    }

    pub fn from_source(cfg: BertConfig, t: &dyn TensorSource) -> Result<Bert, BertError> {
        let pre = if t.has("bert.embeddings.word_embeddings.weight") {
            "bert."
        } else {
            ""
        };
        let get = |name: &str, len: usize| -> Result<Vec<f32>, BertError> {
            let v = t.read(&format!("{pre}{name}"))?;
            if v.len() != len {
                return Err(BertError::Shape(format!(
                    "`{name}` has {} values, expected {len}",
                    v.len()
                )));
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err(BertError::Shape(format!(
                    "`{name}` holds non-finite values"
                )));
            }
            Ok(v)
        };
        let get_any = |names: &[&str], len: usize| -> Result<Vec<f32>, BertError> {
            for n in names {
                if t.has(&format!("{pre}{n}")) {
                    return get(n, len);
                }
            }
            Err(BertError::Load(format!(
                "the model has no tensor `{}`",
                names[0]
            )))
        };
        let h = cfg.hidden;
        let lin = |base: &str, rows: usize, cols: usize| -> Result<Lin, BertError> {
            Ok(Lin {
                w: get(&format!("{base}.weight"), rows * cols)?,
                b: get(&format!("{base}.bias"), rows)?,
                rows,
                cols,
            })
        };
        let ln = |base: &str| -> Result<Ln, BertError> {
            Ok(Ln {
                g: get_any(&[&format!("{base}.weight"), &format!("{base}.gamma")], h)?,
                b: get_any(&[&format!("{base}.bias"), &format!("{base}.beta")], h)?,
            })
        };
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("encoder.layer.{i}");
            layers.push(Layer {
                q: lin(&format!("{p}.attention.self.query"), h, h)?,
                k: lin(&format!("{p}.attention.self.key"), h, h)?,
                v: lin(&format!("{p}.attention.self.value"), h, h)?,
                o: lin(&format!("{p}.attention.output.dense"), h, h)?,
                ln1: ln(&format!("{p}.attention.output.LayerNorm"))?,
                up: lin(&format!("{p}.intermediate.dense"), cfg.ffn, h)?,
                down: lin(&format!("{p}.output.dense"), h, cfg.ffn)?,
                ln2: ln(&format!("{p}.output.LayerNorm"))?,
            });
        }
        let pooler = if t.has(&format!("{pre}pooler.dense.weight")) {
            Some(lin("pooler.dense", h, h)?)
        } else {
            None
        };
        // the classification head of BertForSequenceClassification is stored without the `bert.` prefix
        let classifier = if t.has("classifier.weight") {
            let w = t.read("classifier.weight")?;
            if w.is_empty() || w.len() % h != 0 {
                return Err(BertError::Shape(
                    "classifier.weight has a wrong size".into(),
                ));
            }
            let rows = w.len() / h;
            let b = t.read("classifier.bias")?;
            if b.len() != rows {
                return Err(BertError::Shape("classifier.bias has a wrong size".into()));
            }
            Some(Lin {
                w,
                b,
                rows,
                cols: h,
            })
        } else {
            None
        };
        Ok(Bert {
            word: get("embeddings.word_embeddings.weight", cfg.vocab * h)?,
            pos: get("embeddings.position_embeddings.weight", cfg.max_pos * h)?,
            typ: get(
                "embeddings.token_type_embeddings.weight",
                cfg.type_vocab * h,
            )?,
            ln_e: ln("embeddings.LayerNorm")?,
            layers,
            pooler,
            classifier,
            cfg,
        })
    }

    pub fn has_classifier(&self) -> bool {
        self.classifier.is_some()
    }

    /// Final hidden states, `ids.len() × hidden` row-major. `types` gives the segment id of every
    /// token (all zero when `None`). No padding mask is needed: one sequence at a time.
    pub fn forward(&self, ids: &[u32], types: Option<&[u32]>) -> Result<Vec<f32>, BertError> {
        let c = &self.cfg;
        let (n, h) = (ids.len(), c.hidden);
        if n == 0 {
            return Err(BertError::Empty);
        }
        if n > c.max_pos {
            return Err(BertError::TooLong {
                have: n,
                max: c.max_pos,
            });
        }
        if types.is_some_and(|t| t.len() != n) {
            return Err(BertError::Shape(
                "one segment id per token is needed".into(),
            ));
        }
        let mut x = vec![0.0f32; n * h];
        for (t, &id) in ids.iter().enumerate() {
            if id as usize >= c.vocab {
                return Err(BertError::Token(id));
            }
            let ty = types.map_or(0, |v| v[t]) as usize;
            if ty >= c.type_vocab {
                return Err(BertError::Shape(format!("segment id {ty} is out of range")));
            }
            let row = &mut x[t * h..(t + 1) * h];
            let (w, p, s) = (
                &self.word[id as usize * h..(id as usize + 1) * h],
                &self.pos[t * h..(t + 1) * h],
                &self.typ[ty * h..(ty + 1) * h],
            );
            for i in 0..h {
                row[i] = w[i] + p[i] + s[i];
            }
            layer_norm(row, &self.ln_e, c.eps);
        }
        let (nh, dh) = (c.heads, h / c.heads);
        let scale = 1.0 / (dh as f32).sqrt();
        let (mut q, mut k, mut v) = (
            vec![0.0f32; n * h],
            vec![0.0f32; n * h],
            vec![0.0f32; n * h],
        );
        let (mut ctx, mut tmp) = (vec![0.0f32; n * h], vec![0.0f32; n * h]);
        let mut ff = vec![0.0f32; n * c.ffn];
        let mut sc = vec![0.0f32; n];
        for l in &self.layers {
            linear(&l.q, &x, n, &mut q);
            linear(&l.k, &x, n, &mut k);
            linear(&l.v, &x, n, &mut v);
            for hd in 0..nh {
                let o = hd * dh;
                for i in 0..n {
                    let qi = &q[i * h + o..i * h + o + dh];
                    for j in 0..n {
                        let kj = &k[j * h + o..j * h + o + dh];
                        sc[j] = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                    }
                    ops::softmax(&mut sc);
                    let out = &mut ctx[i * h + o..i * h + o + dh];
                    out.fill(0.0);
                    for j in 0..n {
                        ops::axpy(sc[j], &v[j * h + o..j * h + o + dh], out);
                    }
                }
            }
            linear(&l.o, &ctx, n, &mut tmp);
            for t in 0..n {
                let row = &mut tmp[t * h..(t + 1) * h];
                for i in 0..h {
                    row[i] += x[t * h + i];
                }
                layer_norm(row, &l.ln1, c.eps);
            }
            x.copy_from_slice(&tmp);
            linear(&l.up, &x, n, &mut ff);
            ff.iter_mut().for_each(|f| *f = activate(c.act, *f));
            linear(&l.down, &ff, n, &mut tmp);
            for t in 0..n {
                let row = &mut tmp[t * h..(t + 1) * h];
                for i in 0..h {
                    row[i] += x[t * h + i];
                }
                layer_norm(row, &l.ln2, c.eps);
            }
            x.copy_from_slice(&tmp);
        }
        Ok(x)
    }

    /// One vector for a token sequence (`Pool::Cls` or `Pool::Mean`), optionally scaled to unit length.
    pub fn embed(
        &self,
        ids: &[u32],
        types: Option<&[u32]>,
        pool: Pool,
        normalize: bool,
    ) -> Result<Vec<f32>, BertError> {
        let h = self.cfg.hidden;
        let x = self.forward(ids, types)?;
        let mut v = match pool {
            Pool::Cls => x[..h].to_vec(),
            Pool::Mean => {
                let mut m = vec![0.0f64; h];
                for row in x.chunks(h) {
                    for (a, b) in m.iter_mut().zip(row) {
                        *a += f64::from(*b);
                    }
                }
                m.iter().map(|a| (*a / ids.len() as f64) as f32).collect()
            }
        };
        if normalize {
            let n = v
                .iter()
                .map(|a| f64::from(*a) * f64::from(*a))
                .sum::<f64>()
                .sqrt();
            if n > 0.0 && n.is_finite() {
                v.iter_mut().for_each(|a| *a = (f64::from(*a) / n) as f32);
            }
        }
        if v.iter().any(|a| !a.is_finite()) {
            return Err(BertError::Shape("the embedding is not finite".into()));
        }
        Ok(v)
    }

    /// The raw classification logit(s) for a (query, document) pair already encoded as
    /// `[CLS] q [SEP] d [SEP]` with its segment ids: `tanh(pooler(h[CLS]))` through the classifier.
    /// A single-label cross-encoder returns one relevance score (higher is more relevant).
    pub fn classify(&self, ids: &[u32], types: &[u32]) -> Result<Vec<f32>, BertError> {
        let cl = self.classifier.as_ref().ok_or(BertError::NoClassifier)?;
        let h = self.cfg.hidden;
        let x = self.forward(ids, Some(types))?;
        let mut cls = x[..h].to_vec();
        if let Some(p) = &self.pooler {
            let mut o = vec![0.0f32; h];
            linear(p, &cls, 1, &mut o);
            cls = o.into_iter().map(f32::tanh).collect();
        }
        let mut out = vec![0.0f32; cl.rows];
        linear(cl, &cls, 1, &mut out);
        if out.iter().any(|a| !a.is_finite()) {
            return Err(BertError::Shape("the score is not finite".into()));
        }
        Ok(out)
    }
}

/// A BERT encoder together with its WordPiece tokenizer: text in, vectors and scores out.
#[derive(Debug, Clone)]
pub struct Encoder {
    pub bert: Bert,
    pub tok: crate::wordpiece::WordPiece,
    /// The pooling the model was trained with (`1_Pooling/config.json` when the folder has one, else mean).
    pub pool: Pool,
    pub name: String,
}

impl Encoder {
    /// A Hugging Face folder: `config.json`, `tokenizer.json`, safetensors, optionally `1_Pooling/config.json`.
    pub fn open(dir: &Path) -> Result<Encoder, BertError> {
        let bert = Bert::open(dir)?;
        let tok = crate::wordpiece::WordPiece::from_file(&dir.join("tokenizer.json"))
            .map_err(|e| BertError::Load(format!("tokenizer.json: {e}")))?;
        if tok.vocab_size() > bert.cfg.vocab {
            return Err(BertError::Load(format!(
                "the tokenizer has {} tokens but the model's vocabulary is {}",
                tok.vocab_size(),
                bert.cfg.vocab
            )));
        }
        let pool = std::fs::read_to_string(dir.join("1_Pooling").join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .map(|v| {
                if v.get("pooling_mode_cls_token").and_then(Value::as_bool) == Some(true) {
                    Pool::Cls
                } else {
                    Pool::Mean
                }
            })
            .unwrap_or(Pool::Mean);
        let name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("encoder")
            .to_string();
        Ok(Encoder {
            bert,
            tok,
            pool,
            name,
        })
    }

    /// `[CLS] text [SEP]`, cut to the model's window (the end of the text is dropped).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = self.tok.encode(text);
        let max = self.bert.cfg.max_pos;
        if ids.len() > max {
            ids.truncate(max);
            if let (Some(sep), Some(last)) = (self.tok.sep, ids.last_mut()) {
                *last = sep;
            }
        }
        ids
    }

    /// Token count of `text` as `embed` would see it.
    pub fn count(&self, text: &str) -> usize {
        self.encode(text).len()
    }

    pub fn embed_text(
        &self,
        text: &str,
        pool: Option<Pool>,
        normalize: bool,
    ) -> Result<Vec<f32>, BertError> {
        self.bert.embed(
            &self.encode(text),
            None,
            pool.unwrap_or(self.pool),
            normalize,
        )
    }

    /// Relevance of `doc` to `query` (needs a model with a classification head): the first logit.
    pub fn score(&self, query: &str, doc: &str) -> Result<f32, BertError> {
        let (ids, types) = self.tok.encode_pair(query, doc, self.bert.cfg.max_pos);
        Ok(self.bert.classify(&ids, &types)?[0])
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6
        }
        fn v(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.f()).collect()
        }
        fn ones(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| 1.0 + self.f() * 0.2).collect()
        }
    }

    pub fn tiny_cfg() -> BertConfig {
        BertConfig {
            vocab: 20,
            hidden: 8,
            layers: 2,
            heads: 2,
            ffn: 16,
            max_pos: 12,
            type_vocab: 2,
            eps: 1e-12,
            act: Act::Gelu,
        }
    }

    pub struct Mem(pub HashMap<String, Vec<f32>>);
    impl TensorSource for Mem {
        fn read(&self, n: &str) -> Result<Vec<f32>, LoadError> {
            self.0
                .get(n)
                .cloned()
                .ok_or_else(|| LoadError::Missing(n.into()))
        }
        fn has(&self, n: &str) -> bool {
            self.0.contains_key(n)
        }
    }

    pub fn tiny_tensors(c: &BertConfig, seed: u64, classifier: bool, prefix: &str) -> Mem {
        let mut r = Rng(seed);
        let mut m = HashMap::new();
        let h = c.hidden;
        let mut put = |k: String, v: Vec<f32>| {
            m.insert(format!("{prefix}{k}"), v);
        };
        put("embeddings.word_embeddings.weight".into(), r.v(c.vocab * h));
        put(
            "embeddings.position_embeddings.weight".into(),
            r.v(c.max_pos * h),
        );
        put(
            "embeddings.token_type_embeddings.weight".into(),
            r.v(c.type_vocab * h),
        );
        put("embeddings.LayerNorm.weight".into(), r.ones(h));
        put("embeddings.LayerNorm.bias".into(), r.v(h));
        for i in 0..c.layers {
            let p = format!("encoder.layer.{i}");
            for (n, rows, cols) in [
                ("attention.self.query", h, h),
                ("attention.self.key", h, h),
                ("attention.self.value", h, h),
                ("attention.output.dense", h, h),
                ("intermediate.dense", c.ffn, h),
                ("output.dense", h, c.ffn),
            ] {
                put(format!("{p}.{n}.weight"), r.v(rows * cols));
                put(format!("{p}.{n}.bias"), r.v(rows));
            }
            for n in ["attention.output.LayerNorm", "output.LayerNorm"] {
                put(format!("{p}.{n}.weight"), r.ones(h));
                put(format!("{p}.{n}.bias"), r.v(h));
            }
        }
        put("pooler.dense.weight".into(), r.v(h * h));
        put("pooler.dense.bias".into(), r.v(h));
        if classifier {
            m.insert("classifier.weight".into(), r.v(h));
            m.insert("classifier.bias".into(), r.v(1));
        }
        Mem(m)
    }

    /// A direct f64 transcription of the BERT equations, independent of the production code.
    fn reference(c: &BertConfig, t: &Mem, ids: &[u32], types: &[u32], pre: &str) -> Vec<Vec<f64>> {
        let g = |n: &str| -> Vec<f64> {
            t.0[&format!("{pre}{n}")]
                .iter()
                .map(|&x| f64::from(x))
                .collect()
        };
        let h = c.hidden;
        let ln = |x: &[f64], gm: &[f64], bt: &[f64]| -> Vec<f64> {
            let m = x.iter().sum::<f64>() / h as f64;
            let v = x.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / h as f64;
            (0..h)
                .map(|i| (x[i] - m) / (v + f64::from(c.eps)).sqrt() * gm[i] + bt[i])
                .collect()
        };
        let lin = |w: &[f64], b: &[f64], x: &[f64], rows: usize| -> Vec<f64> {
            (0..rows)
                .map(|r| (0..x.len()).map(|k| w[r * x.len() + k] * x[k]).sum::<f64>() + b[r])
                .collect()
        };
        let gelu = |x: f64| -> f64 {
            // erf via a high-order series/continued evaluation: use the library-independent identity
            // gelu(x) = x * Phi(x), Phi from numerical integration of the normal density
            let steps = 4000;
            let (lo, hi) = (-9.0f64, x);
            let dx = (hi - lo) / steps as f64;
            let mut s = 0.0;
            for i in 0..=steps {
                let u = lo + dx * i as f64;
                let wgt = if i == 0 || i == steps { 0.5 } else { 1.0 };
                s += wgt * (-0.5 * u * u).exp();
            }
            x * s * dx / (2.0 * std::f64::consts::PI).sqrt()
        };
        let (word, posn, typ) = (
            g("embeddings.word_embeddings.weight"),
            g("embeddings.position_embeddings.weight"),
            g("embeddings.token_type_embeddings.weight"),
        );
        let mut x: Vec<Vec<f64>> = ids
            .iter()
            .enumerate()
            .map(|(p, &id)| {
                let e: Vec<f64> = (0..h)
                    .map(|i| {
                        word[id as usize * h + i] + posn[p * h + i] + typ[types[p] as usize * h + i]
                    })
                    .collect();
                ln(
                    &e,
                    &g("embeddings.LayerNorm.weight"),
                    &g("embeddings.LayerNorm.bias"),
                )
            })
            .collect();
        let (nh, dh) = (c.heads, h / c.heads);
        for l in 0..c.layers {
            let p = format!("encoder.layer.{l}");
            let proj = |n: &str, x: &Vec<Vec<f64>>, rows: usize| -> Vec<Vec<f64>> {
                let (w, b) = (g(&format!("{p}.{n}.weight")), g(&format!("{p}.{n}.bias")));
                x.iter().map(|r| lin(&w, &b, r, rows)).collect()
            };
            let (q, k, v) = (
                proj("attention.self.query", &x, h),
                proj("attention.self.key", &x, h),
                proj("attention.self.value", &x, h),
            );
            let n = x.len();
            let mut ctx = vec![vec![0.0f64; h]; n];
            for hd in 0..nh {
                for i in 0..n {
                    let s: Vec<f64> = (0..n)
                        .map(|j| {
                            (0..dh)
                                .map(|d| q[i][hd * dh + d] * k[j][hd * dh + d])
                                .sum::<f64>()
                                / (dh as f64).sqrt()
                        })
                        .collect();
                    let mx = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = s.iter().map(|a| (a - mx).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for d in 0..dh {
                        ctx[i][hd * dh + d] = (0..n).map(|j| e[j] / z * v[j][hd * dh + d]).sum();
                    }
                }
            }
            let o = proj("attention.output.dense", &ctx, h);
            let x1: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let r: Vec<f64> = (0..h).map(|a| o[i][a] + x[i][a]).collect();
                    ln(
                        &r,
                        &g(&format!("{p}.attention.output.LayerNorm.weight")),
                        &g(&format!("{p}.attention.output.LayerNorm.bias")),
                    )
                })
                .collect();
            let up = proj("intermediate.dense", &x1, c.ffn);
            let act: Vec<Vec<f64>> = up
                .iter()
                .map(|r| r.iter().map(|&a| gelu(a)).collect())
                .collect();
            let (w, b) = (
                g(&format!("{p}.output.dense.weight")),
                g(&format!("{p}.output.dense.bias")),
            );
            x = (0..n)
                .map(|i| {
                    let d = lin(&w, &b, &act[i], h);
                    let r: Vec<f64> = (0..h).map(|a| d[a] + x1[i][a]).collect();
                    ln(
                        &r,
                        &g(&format!("{p}.output.LayerNorm.weight")),
                        &g(&format!("{p}.output.LayerNorm.bias")),
                    )
                })
                .collect();
        }
        x
    }

    #[test]
    fn matches_a_plain_f64_reference_with_and_without_the_bert_prefix() {
        let c = tiny_cfg();
        for pre in ["", "bert."] {
            let t = tiny_tensors(&c, 7, false, pre);
            let m = Bert::from_source(c.clone(), &t).unwrap();
            let ids = [3u32, 11, 4, 4, 19, 0];
            let ty = [0u32, 0, 0, 1, 1, 1];
            let got = m.forward(&ids, Some(&ty)).unwrap();
            let want = reference(&c, &t, &ids, &ty, pre);
            for (t, row) in want.iter().enumerate() {
                for (i, w) in row.iter().enumerate() {
                    let g = f64::from(got[t * c.hidden + i]);
                    assert!(
                        (g - w).abs() < 2e-4,
                        "prefix {pre:?} token {t} dim {i}: {g} vs {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn embeddings_pool_normalise_and_depend_on_the_input() {
        let c = tiny_cfg();
        let m = Bert::from_source(c.clone(), &tiny_tensors(&c, 3, false, "")).unwrap();
        let ids = [2u32, 5, 9];
        let cls = m.embed(&ids, None, Pool::Cls, true).unwrap();
        let mean = m.embed(&ids, None, Pool::Mean, true).unwrap();
        for v in [&cls, &mean] {
            let n: f64 = v.iter().map(|x| f64::from(*x).powi(2)).sum();
            assert!((n - 1.0).abs() < 1e-4);
        }
        assert_ne!(cls, mean);
        assert_eq!(cls, m.embed(&ids, None, Pool::Cls, true).unwrap());
        assert_ne!(cls, m.embed(&[2, 5, 8], None, Pool::Cls, true).unwrap());
        // the CLS pooling equals the first row of the hidden states (not normalised)
        let raw = m.embed(&ids, None, Pool::Cls, false).unwrap();
        assert_eq!(raw, m.forward(&ids, None).unwrap()[..c.hidden]);
    }

    #[test]
    fn classifier_scores_pairs_and_is_required() {
        let c = tiny_cfg();
        let m = Bert::from_source(c.clone(), &tiny_tensors(&c, 5, true, "")).unwrap();
        assert!(m.has_classifier());
        let a = m.classify(&[1, 4, 2, 6, 2], &[0, 0, 0, 1, 1]).unwrap();
        let b = m.classify(&[1, 4, 2, 7, 2], &[0, 0, 0, 1, 1]).unwrap();
        assert_eq!(a.len(), 1);
        assert_ne!(a, b);
        // by hand: tanh(pooler(h0)) through the classifier
        let t = tiny_tensors(&c, 5, true, "");
        let x = m.forward(&[1, 4, 2, 6, 2], Some(&[0, 0, 0, 1, 1])).unwrap();
        let (pw, pb) = (&t.0["pooler.dense.weight"], &t.0["pooler.dense.bias"]);
        let pooled: Vec<f64> = (0..c.hidden)
            .map(|r| {
                let dot: f64 = (0..c.hidden)
                    .map(|k| f64::from(pw[r * c.hidden + k]) * f64::from(x[k]))
                    .sum();
                (dot + f64::from(pb[r])).tanh()
            })
            .collect();
        let score: f64 = pooled
            .iter()
            .zip(&t.0["classifier.weight"])
            .map(|(a, w)| a * f64::from(*w))
            .sum::<f64>()
            + f64::from(t.0["classifier.bias"][0]);
        assert!(
            (f64::from(a[0]) - score).abs() < 1e-4,
            "{} vs {score}",
            a[0]
        );
        let plain = Bert::from_source(c.clone(), &tiny_tensors(&c, 5, false, "")).unwrap();
        assert_eq!(
            plain.classify(&[1, 2], &[0, 0]),
            Err(BertError::NoClassifier)
        );
    }

    fn write_st(path: &Path, m: &Mem) {
        let mut names: Vec<&String> = m.0.keys().collect();
        names.sort();
        let mut hdr = serde_json::Map::new();
        let mut data: Vec<u8> = vec![];
        for n in names {
            let v = &m.0[n];
            let (a, b) = (data.len(), data.len() + v.len() * 4);
            data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            hdr.insert(
                n.clone(),
                serde_json::json!({"dtype": "F32", "shape": [v.len()], "data_offsets": [a, b]}),
            );
        }
        let h = Value::Object(hdr).to_string();
        let mut f = (h.len() as u64).to_le_bytes().to_vec();
        f.extend(h.as_bytes());
        f.extend(data);
        std::fs::write(path, f).unwrap();
    }

    #[test]
    fn a_folder_loads_embeds_text_and_scores_pairs() {
        let c = BertConfig {
            vocab: 18,
            ..tiny_cfg()
        };
        let d = std::env::temp_dir().join(format!("bert-folder-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("1_Pooling")).unwrap();
        std::fs::write(
            d.join("config.json"),
            r#"{"model_type":"bert","hidden_size":8,"num_attention_heads":2,"num_hidden_layers":2,
                "intermediate_size":16,"vocab_size":18,"max_position_embeddings":12}"#,
        )
        .unwrap();
        let toks = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "un", "##aff",
            "##able", "!", ",", "a", "##s", "cafe", "sa", "##rut", "##a",
        ];
        let vocab: serde_json::Map<String, Value> = toks
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i.into()))
            .collect();
        std::fs::write(
            d.join("tokenizer.json"),
            serde_json::json!({"model": {"type": "WordPiece", "vocab": vocab, "unk_token": "[UNK]"}})
                .to_string(),
        )
        .unwrap();
        std::fs::write(
            d.join("1_Pooling/config.json"),
            r#"{"pooling_mode_cls_token": true}"#,
        )
        .unwrap();
        write_st(
            &d.join("model.safetensors"),
            &tiny_tensors(&c, 9, true, "bert."),
        );
        let e = Encoder::open(&d).unwrap();
        assert_eq!(e.pool, Pool::Cls);
        let v = e.embed_text("hello world", None, true).unwrap();
        assert_eq!(v.len(), 8);
        assert_ne!(v, e.embed_text("world hello", None, true).unwrap());
        assert_ne!(
            v,
            e.embed_text("hello world", Some(Pool::Mean), true).unwrap()
        );
        // text longer than the 12-token window is cut and still closes with [SEP]
        let long = "hello ".repeat(40);
        let ids = e.encode(&long);
        assert_eq!(ids.len(), 12);
        assert_eq!(*ids.last().unwrap(), e.tok.sep.unwrap());
        assert!(e.embed_text(&long, None, true).is_ok());
        let (s1, s2) = (
            e.score("hello", "world").unwrap(),
            e.score("hello", "hello").unwrap(),
        );
        assert!(s1.is_finite() && s2.is_finite() && s1 != s2);
        // a vocabulary larger than the model's is refused
        std::fs::write(
            d.join("config.json"),
            r#"{"model_type":"bert","hidden_size":8,"num_attention_heads":2,"num_hidden_layers":2,
                "intermediate_size":16,"vocab_size":10,"max_position_embeddings":12}"#,
        )
        .unwrap();
        assert!(Encoder::open(&d).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn bad_input_and_bad_weights_are_errors() {
        let c = tiny_cfg();
        let t = tiny_tensors(&c, 1, false, "");
        let m = Bert::from_source(c.clone(), &t).unwrap();
        assert_eq!(m.forward(&[], None), Err(BertError::Empty));
        assert_eq!(m.forward(&[99], None), Err(BertError::Token(99)));
        assert!(matches!(
            m.forward(&[1; 13], None),
            Err(BertError::TooLong { have: 13, max: 12 })
        ));
        assert!(m.forward(&[1, 2], Some(&[0])).is_err());
        assert!(m.forward(&[1, 2], Some(&[0, 5])).is_err());
        // a missing tensor, a wrong-size tensor and a non-finite tensor
        let mut miss = t.0.clone();
        miss.remove("encoder.layer.1.output.dense.bias");
        assert!(Bert::from_source(c.clone(), &Mem(miss)).is_err());
        let mut sz = t.0.clone();
        sz.get_mut("embeddings.word_embeddings.weight")
            .unwrap()
            .pop();
        assert!(Bert::from_source(c.clone(), &Mem(sz)).is_err());
        let mut nan = t.0.clone();
        nan.get_mut("embeddings.LayerNorm.bias").unwrap()[0] = f32::NAN;
        assert!(Bert::from_source(c, &Mem(nan)).is_err());
    }

    #[test]
    fn config_parsing_rejects_other_architectures_and_absurd_sizes() {
        let ok = r#"{"model_type":"bert","hidden_size":8,"num_attention_heads":2,"num_hidden_layers":2,
            "intermediate_size":16,"vocab_size":20,"max_position_embeddings":12}"#;
        let c = BertConfig::from_json(ok).unwrap();
        assert_eq!((c.hidden, c.type_vocab, c.act), (8, 2, Act::Gelu));
        for bad in [
            r#"{"model_type":"roberta"}"#.to_string(),
            ok.replace("\"hidden_size\":8", "\"hidden_size\":7"),
            ok.replace("\"hidden_size\":8", "\"hidden_size\":99999999"),
            ok.replace("\"num_hidden_layers\":2", "\"num_hidden_layers\":0"),
            ok.replace("}", ",\"hidden_act\":\"swish\"}"),
            ok.replace("}", ",\"position_embedding_type\":\"relative_key\"}"),
            "{".to_string(),
        ] {
            assert!(BertConfig::from_json(&bad).is_err(), "{bad}");
        }
    }
}
