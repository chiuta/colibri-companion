//! Loads a model folder (Hugging Face layout) or a GGUF file together with everything needed to talk
//! to it: tokenizer, chat template, special tokens and stop tokens.
use crate::lorasrc::{self, BaseKind, LoraSource};
use crate::template::{Builtin, ChatTemplate};
use companion_engine::arch::{Spec, Support, from_gguf, from_hf_config};
use companion_engine::loader::{
    GgufFile, LoadOptions, MoeOptions, Naming, SafeTensors, TensorSource, WeightFile, load_model,
};
use companion_engine::model::{KvKind, Model};
use companion_engine::tokenizer::Tokenizer;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A LoRA adapter to merge while loading (`--lora PATH[:SCALE]`).
#[derive(Clone, Debug, PartialEq)]
pub struct LoraSpec {
    pub path: PathBuf,
    pub scale: f32,
}

#[derive(Clone, Debug)]
pub struct LoadCfg {
    pub prefer_quantized: bool,
    pub mmap: bool,
    pub kv: KvKind,
    /// RAM the model may use. Only mixture-of-experts models act on it here: routed experts that do not
    /// fit are streamed from disk. `None`: everything is read up front.
    pub ram_budget: Option<u64>,
    /// Adapters merged into the weights as they are read.
    pub lora: Vec<LoraSpec>,
}
impl Default for LoadCfg {
    fn default() -> LoadCfg {
        LoadCfg {
            prefer_quantized: true,
            mmap: true,
            kv: KvKind::F32,
            ram_budget: None,
            lora: vec![],
        }
    }
}

pub struct Loaded {
    pub model: Model,
    pub tok: Tokenizer,
    pub name: String,
    pub path: PathBuf,
    pub template: ChatTemplate,
    pub builtin: Builtin,
    pub bos: String,
    pub eos: String,
    /// Tokens that end a reply.
    pub stop_ids: Vec<usize>,
    pub kv: KvKind,
    pub warnings: Vec<String>,
}
impl std::fmt::Debug for Loaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Loaded({})", self.name)
    }
}

/// What kind of thing is at `path`.
#[derive(Debug, PartialEq, Eq)]
pub enum Source {
    HfDir(PathBuf),
    Gguf(PathBuf),
}

pub fn resolve(path: &Path) -> Result<Source, String> {
    if path.is_file() {
        return if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            Ok(Source::Gguf(path.to_path_buf()))
        } else if path.file_name().is_some_and(|n| n == "config.json") {
            Ok(Source::HfDir(
                path.parent().unwrap_or(Path::new(".")).to_path_buf(),
            ))
        } else {
            Err(format!(
                "{}: expected a .gguf file or a model folder",
                path.display()
            ))
        };
    }
    if !path.is_dir() {
        return Err(format!("{}: no such file or folder", path.display()));
    }
    if path.join("config.json").is_file() {
        return Ok(Source::HfDir(path.to_path_buf()));
    }
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        })
        .collect();
    ggufs.sort();
    match ggufs.len() {
        0 => Err(format!(
            "{}: no config.json and no .gguf file in this folder",
            path.display()
        )),
        _ => Ok(Source::Gguf(ggufs.swap_remove(0))),
    }
}

fn read_json(p: &Path) -> Option<Value> {
    let len = std::fs::metadata(p).ok()?.len();
    if len > 64 << 20 {
        return None;
    }
    serde_json::from_slice(&std::fs::read(p).ok()?).ok()
}

fn token_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content").and_then(Value::as_str).map(str::to_string),
        _ => None,
    }
}

/// The chat template string from `tokenizer_config.json` (a string, or a list of named templates).
pub fn template_from_config(cfg: &Value) -> Option<String> {
    match cfg.get("chat_template")? {
        Value::String(s) => Some(s.clone()),
        Value::Array(a) => {
            let pick = a
                .iter()
                .find(|x| x.get("name").and_then(Value::as_str) == Some("default"))
                .or_else(|| a.first())?;
            pick.get("template")
                .and_then(Value::as_str)
                .map(str::to_string)
        }
        _ => None,
    }
}

fn check_support(spec: &Spec) -> Result<(), String> {
    match spec.support {
        Support::Planned(why) => Err(format!(
            "`{}` models need {why}, which the engine does not support yet",
            spec.family
        )),
        Support::Partial(why) => Err(format!("`{}` is only partly supported: {why}", spec.family)),
        Support::Recurrent | Support::Hybrid => Err(format!(
            "`{}` models do not run through the dense loader; use family::load_any",
            spec.family
        )),
        _ => Ok(()),
    }
}

/// Everything about a model that is known before its weights are placed in memory: where they come
/// from, the architecture, tokenizer and chat template. [`Prepared::build_model`] turns it into a
/// [`Model`]; the layer-streaming runner uses `src` directly.
pub struct Prepared {
    pub spec: Spec,
    pub src: Arc<dyn WeightFile>,
    pub naming: Naming,
    pub permuted: bool,
    pub tok: Tokenizer,
    pub name: String,
    pub path: PathBuf,
    pub template: ChatTemplate,
    pub builtin: Builtin,
    pub bos: String,
    pub eos: String,
    pub stop_ids: Vec<usize>,
    pub warnings: Vec<String>,
    /// Bytes of tensor data on disk.
    pub disk_bytes: u64,
}

impl Prepared {
    pub fn options(&self, cfg: &LoadCfg) -> LoadOptions {
        LoadOptions {
            prefer_quantized: cfg.prefer_quantized,
            mmap: cfg.mmap,
            ..Default::default()
        }
    }

    /// Place the weights in memory (mixture-of-experts models stream what does not fit in `cfg.ram_budget`).
    pub fn build_model(&self, cfg: &LoadCfg) -> Result<Model, String> {
        let moe = MoeOptions {
            ram_budget: cfg.ram_budget,
            expert_cache: None,
            prefetch: true,
        };
        load_model(
            self.spec.clone(),
            self.src.clone(),
            self.naming,
            self.permuted,
            self.options(cfg),
            moe,
        )
        .map_err(|e| e.to_string())
    }

    pub fn into_loaded(self, cfg: &LoadCfg) -> Result<Loaded, String> {
        let model = self.build_model(cfg)?;
        Ok(Loaded {
            model,
            tok: self.tok,
            name: self.name,
            path: self.path,
            template: self.template,
            builtin: self.builtin,
            bos: self.bos,
            eos: self.eos,
            stop_ids: self.stop_ids,
            kv: cfg.kv,
            warnings: self.warnings,
        })
    }
}

fn with_lora(
    src: Arc<dyn WeightFile>,
    naming: Naming,
    base: BaseKind,
    spec: &Spec,
    cfg: &LoadCfg,
    warnings: &mut Vec<String>,
) -> Result<Arc<dyn WeightFile>, String> {
    if cfg.lora.is_empty() {
        return Ok(src);
    }
    let mut list = vec![];
    for l in &cfg.lora {
        let a = lorasrc::prepare_adapter(&l.path, base, spec)?;
        warnings.push(format!("adapter `{}` ({} matrices, scale {}) is merged into the weights; the matrices it changes are held as f32", a.name, a.pairs.len(), l.scale));
        list.push((Arc::new(a), l.scale));
    }
    Ok(Arc::new(LoraSource::new(src, naming, list)))
}

/// Open a model folder or file without placing its weights in memory yet.
pub fn prepare(path: &Path, cfg: &LoadCfg) -> Result<Prepared, String> {
    let mut warnings = vec![];
    #[allow(clippy::type_complexity)]
    let (spec, src, naming, permuted, tok, name, tcfg, tmpl_src, gguf_eot, disk_bytes): (
        Spec,
        Arc<dyn WeightFile>,
        Naming,
        bool,
        Tokenizer,
        String,
        Option<Value>,
        Option<String>,
        Vec<u32>,
        u64,
    );
    let resolved = resolve(path)?;
    let dir = match &resolved {
        Source::HfDir(d) => Some(d.clone()),
        Source::Gguf(_) => None,
    };
    match resolved {
        Source::HfDir(dir) => {
            let c = std::fs::read_to_string(dir.join("config.json"))
                .map_err(|e| format!("config.json: {e}"))?;
            spec = from_hf_config(&c).map_err(|e| e.to_string())?;
            check_support(&spec)?;
            let st = SafeTensors::open(&dir).map_err(|e| e.to_string())?;
            disk_bytes = st.infos().iter().filter_map(|t| t.bytes()).sum::<u64>();
            naming = Naming::Hf;
            permuted = false;
            src = with_lora(
                Arc::new(st),
                naming,
                BaseKind::Hf,
                &spec,
                cfg,
                &mut warnings,
            )?;
            let tp = dir.join("tokenizer.json");
            if !tp.is_file() {
                return Err(if dir.join("tokenizer.model").is_file() {
                    "this folder only has a SentencePiece `tokenizer.model`; a `tokenizer.json` is needed".into()
                } else {
                    "tokenizer.json is missing".into()
                });
            }
            tok = Tokenizer::from_file(&tp).map_err(|e| format!("tokenizer.json: {e}"))?;
            tcfg = read_json(&dir.join("tokenizer_config.json"));
            let mut t = tcfg.as_ref().and_then(template_from_config);
            if t.is_none() {
                t = std::fs::read_to_string(dir.join("chat_template.jinja")).ok();
            }
            if t.is_none() {
                t = read_json(&dir.join("chat_template.json"))
                    .as_ref()
                    .and_then(template_from_config);
            }
            tmpl_src = t;
            gguf_eot = vec![];
            name = dir
                .file_name()
                .map_or_else(|| "model".to_string(), |n| n.to_string_lossy().into_owned());
        }
        Source::Gguf(file) => {
            let f = GgufFile::open(&file).map_err(|e| e.to_string())?;
            spec = from_gguf(&f.gguf).map_err(|e| e.to_string())?;
            check_support(&spec)?;
            permuted = f.gguf.architecture() == Some("llama");
            naming = Naming::Gguf;
            tok =
                Tokenizer::from_gguf(&f.gguf).map_err(|e| format!("tokenizer in the GGUF: {e}"))?;
            tcfg = None;
            tmpl_src = f
                .gguf
                .meta
                .get("tokenizer.chat_template")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            gguf_eot = ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"]
                .iter()
                .filter_map(|k| f.gguf.meta.get(*k).and_then(|v| v.as_u64()))
                .map(|v| v as u32)
                .collect();
            name = file
                .file_stem()
                .map_or_else(|| "model".to_string(), |n| n.to_string_lossy().into_owned());
            disk_bytes = f.gguf.total_tensor_bytes();
            let kind = if permuted {
                BaseKind::GgufLlama
            } else {
                BaseKind::GgufOther
            };
            src = with_lora(Arc::new(f), naming, kind, &spec, cfg, &mut warnings)?;
        }
    }
    let tok_text = |key: &str, id: Option<u32>| -> String {
        tcfg.as_ref()
            .and_then(|c| c.get(key))
            .and_then(token_text)
            .or_else(|| id.and_then(|i| tok.id_to_token(i).map(str::to_string)))
            .unwrap_or_default()
    };
    let (bos, eos) = (
        tok_text("bos_token", tok.bos),
        tok_text("eos_token", tok.eos),
    );
    let builtin = Builtin::detect(&spec.family, &|t| tok.token_to_id(t).is_some());
    let (template, warn) = ChatTemplate::from_source(tmpl_src.as_deref(), builtin);
    warnings.extend(warn);

    // tokens that end a reply
    let vocab = spec.vocab as usize;
    let mut stop: Vec<usize> = vec![];
    let mut add = |id: u32| {
        let i = id as usize;
        if (vocab == 0 || i < vocab) && !stop.contains(&i) {
            stop.push(i);
        }
    };
    if let Some(e) = tok.eos {
        add(e);
    }
    gguf_eot.into_iter().for_each(&mut add);
    for t in [
        "<|im_end|>",
        "<|eot_id|>",
        "<end_of_turn>",
        "<|endoftext|>",
        "<|end|>",
    ] {
        if let Some(i) = tok.token_to_id(t) {
            add(i);
        }
    }
    if let Some(d) = &dir {
        for f in ["generation_config.json", "config.json"] {
            if let Some(j) = read_json(&d.join(f)) {
                match j.get("eos_token_id") {
                    Some(Value::Number(n)) => {
                        if let Some(i) = n.as_u64() {
                            add(i as u32);
                        }
                    }
                    Some(Value::Array(a)) => a
                        .iter()
                        .filter_map(Value::as_u64)
                        .for_each(|i| add(i as u32)),
                    _ => {}
                }
            }
        }
    }
    Ok(Prepared {
        spec,
        src,
        naming,
        permuted,
        tok,
        name,
        path: path.to_path_buf(),
        template,
        builtin,
        bos,
        eos,
        stop_ids: stop,
        warnings,
        disk_bytes,
    })
}

pub fn load(path: &Path, cfg: &LoadCfg) -> Result<Loaded, String> {
    prepare(path, cfg)?.into_loaded(cfg)
}

#[cfg(test)]
pub(crate) mod testkit {
    //! Writes a tiny random Llama-style model (safetensors + config + byte-level tokenizer) to a folder.
    use std::path::{Path, PathBuf};

    fn rng(s: &mut u64) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.8
    }
    fn mat(n: usize, s: &mut u64) -> Vec<f32> {
        (0..n).map(|_| rng(s)).collect()
    }
    fn ones(n: usize, s: &mut u64) -> Vec<f32> {
        (0..n).map(|_| 1.0 + rng(s) * 0.1).collect()
    }

    pub fn tmpdir(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "corbul-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn byte_chars() -> Vec<char> {
        let mut bs: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut table = vec!['\0'; 256];
        for (b, c) in bs.into_iter().zip(cs) {
            table[b as usize] = char::from_u32(c).unwrap();
        }
        table
    }

    pub const IM_START: usize = 256;
    pub const IM_END: usize = 257;
    pub const VOCAB: usize = 258;

    /// A 2-layer model, hidden 16, byte-level vocabulary of 256 bytes + ChatML specials.
    pub fn write_model(dir: &Path, with_template: bool) {
        write_model_ctx(dir, with_template, 96)
    }

    /// Like [`write_model`] with a chosen context window.
    pub fn write_model_ctx(dir: &Path, with_template: bool, ctx: usize) {
        write_model_opts(dir, with_template, ctx, 0)
    }

    /// A Mixtral-style mixture-of-experts variant (`experts` ≥ 2, two used per token).
    pub fn write_moe_model(dir: &Path, experts: usize) {
        write_model_opts(dir, true, 96, experts)
    }

    /// Write a PEFT LoRA adapter for the q projection of layer 0 (rank 2, alpha 2).
    pub fn write_adapter(dir: &Path, seed: u64) {
        let (h, out, r) = (16usize, 16usize, 2usize);
        let mut s = seed | 1;
        let a = mat(r * h, &mut s);
        let b = mat(out * r, &mut s);
        let (mut header, mut data) = (serde_json::Map::new(), Vec::<u8>::new());
        for (n, shape, d) in [
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                vec![r, h],
                a,
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                vec![out, r],
                b,
            ),
        ] {
            let start = data.len();
            for x in &d {
                data.extend(x.to_le_bytes());
            }
            header.insert(n.to_string(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
        }
        let hb = serde_json::to_vec(&header).unwrap();
        let mut file = (hb.len() as u64).to_le_bytes().to_vec();
        file.extend(hb);
        file.extend(data);
        std::fs::write(dir.join("adapter_model.safetensors"), file).unwrap();
        std::fs::write(dir.join("adapter_config.json"), r#"{"peft_type":"LORA","r":2,"lora_alpha":2,"target_modules":["q_proj"],"base_model_name_or_path":"x"}"#).unwrap();
    }

    /// Like [`write_model`] with other random weights (same architecture and vocabulary).
    pub fn write_model_seeded(dir: &Path, seed: u64) {
        write_model_full(dir, false, 96, 0, seed)
    }

    fn write_model_opts(dir: &Path, with_template: bool, ctx: usize, experts: usize) {
        write_model_full(dir, with_template, ctx, experts, 0x9e37_79b9_7f4a_7c15)
    }

    fn write_model_full(dir: &Path, with_template: bool, ctx: usize, experts: usize, seed: u64) {
        let (h, layers, heads, kv, hd, ff, v) =
            (16usize, 2usize, 4usize, 2usize, 4usize, 24usize, VOCAB);
        let mut s = seed;
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
            (
                "model.embed_tokens.weight".into(),
                vec![v, h],
                mat(v * h, &mut s),
            ),
            ("model.norm.weight".into(), vec![h], ones(h, &mut s)),
        ];
        for i in 0..layers {
            let p = |n: &str| format!("model.layers.{i}.{n}");
            tensors.push((p("input_layernorm.weight"), vec![h], ones(h, &mut s)));
            tensors.push((
                p("self_attn.q_proj.weight"),
                vec![heads * hd, h],
                mat(heads * hd * h, &mut s),
            ));
            tensors.push((
                p("self_attn.k_proj.weight"),
                vec![kv * hd, h],
                mat(kv * hd * h, &mut s),
            ));
            tensors.push((
                p("self_attn.v_proj.weight"),
                vec![kv * hd, h],
                mat(kv * hd * h, &mut s),
            ));
            tensors.push((
                p("self_attn.o_proj.weight"),
                vec![h, heads * hd],
                mat(h * heads * hd, &mut s),
            ));
            tensors.push((
                p("post_attention_layernorm.weight"),
                vec![h],
                ones(h, &mut s),
            ));
            if experts > 0 {
                tensors.push((
                    p("block_sparse_moe.gate.weight"),
                    vec![experts, h],
                    mat(experts * h, &mut s),
                ));
                for e in 0..experts {
                    tensors.push((
                        p(&format!("block_sparse_moe.experts.{e}.w1.weight")),
                        vec![ff, h],
                        mat(ff * h, &mut s),
                    ));
                    tensors.push((
                        p(&format!("block_sparse_moe.experts.{e}.w2.weight")),
                        vec![h, ff],
                        mat(h * ff, &mut s),
                    ));
                    tensors.push((
                        p(&format!("block_sparse_moe.experts.{e}.w3.weight")),
                        vec![ff, h],
                        mat(ff * h, &mut s),
                    ));
                }
            } else {
                tensors.push((p("mlp.gate_proj.weight"), vec![ff, h], mat(ff * h, &mut s)));
                tensors.push((p("mlp.up_proj.weight"), vec![ff, h], mat(ff * h, &mut s)));
                tensors.push((p("mlp.down_proj.weight"), vec![h, ff], mat(h * ff, &mut s)));
            }
        }
        let (mut header, mut data) = (serde_json::Map::new(), Vec::<u8>::new());
        for (n, shape, d) in &tensors {
            let start = data.len();
            for x in d {
                data.extend(x.to_le_bytes());
            }
            header.insert(n.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
        }
        let hb = serde_json::to_vec(&header).unwrap();
        let mut file = (hb.len() as u64).to_le_bytes().to_vec();
        file.extend(hb);
        file.extend(data);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        let mut config = serde_json::json!({
            "model_type": if experts > 0 { "mixtral" } else { "llama" }, "hidden_size": h, "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv,
            "intermediate_size": ff, "vocab_size": v, "max_position_embeddings": ctx, "rope_theta": 10000.0, "rms_norm_eps": 1e-5, "eos_token_id": IM_END
        });
        if experts > 0 {
            config["num_local_experts"] = experts.into();
            config["num_experts_per_tok"] = 2.into();
        }
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let table = byte_chars();
        let vocab: serde_json::Map<String, serde_json::Value> = table
            .iter()
            .enumerate()
            .map(|(i, c)| (c.to_string(), serde_json::json!(i)))
            .collect();
        std::fs::write(dir.join("tokenizer.json"), serde_json::json!({
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "added_tokens": [{"id": IM_START, "content": "<|im_start|>", "special": true}, {"id": IM_END, "content": "<|im_end|>", "special": true}],
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false}, "decoder": {"type": "ByteLevel"}
        }).to_string()).unwrap();
        let mut tc = serde_json::json!({"bos_token": null, "eos_token": "<|im_end|>"});
        if with_template {
            tc["chat_template"] = serde_json::json!(
                "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
            );
        }
        std::fs::write(dir.join("tokenizer_config.json"), tc.to_string()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::testkit::*;
    use super::*;

    #[test]
    fn a_model_folder_loads_with_its_template_and_stop_tokens() {
        let d = tmpdir("load");
        write_model(&d, true);
        let l = load(&d, &LoadCfg::default()).unwrap();
        assert_eq!(l.model.spec.layers, 2);
        assert!(matches!(l.template, ChatTemplate::Jinja(_)));
        assert_eq!(l.builtin, Builtin::ChatMl);
        assert!(l.stop_ids.contains(&IM_END));
        assert_eq!(l.eos, "<|im_end|>");
        assert!(l.warnings.is_empty());
        // text without a template falls back to the detected built-in
        let d2 = tmpdir("load2");
        write_model(&d2, false);
        let l2 = load(&d2, &LoadCfg::default()).unwrap();
        assert!(matches!(
            l2.template,
            ChatTemplate::Builtin(Builtin::ChatMl)
        ));
        std::fs::remove_dir_all(d).unwrap();
        std::fs::remove_dir_all(d2).unwrap();
    }

    #[test]
    fn bad_inputs_give_plain_errors() {
        let d = tmpdir("bad");
        assert!(
            load(&d, &LoadCfg::default())
                .unwrap_err()
                .contains("no config.json")
        );
        assert!(
            load(&d.join("missing"), &LoadCfg::default())
                .unwrap_err()
                .contains("no such file")
        );
        std::fs::write(d.join("config.json"), r#"{"model_type":"deepseek_v3","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2}"#).unwrap();
        assert!(
            load(&d, &LoadCfg::default())
                .unwrap_err()
                .contains("latent attention")
        );
        // mixture-of-experts configs are accepted now; without weights the failure is about the files
        std::fs::write(d.join("config.json"), r#"{"model_type":"mixtral","hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_local_experts":4}"#).unwrap();
        assert!(
            !load(&d, &LoadCfg::default())
                .unwrap_err()
                .contains("mixture-of-experts")
        );
        std::fs::write(d.join("config.json"), "not json").unwrap();
        assert!(load(&d, &LoadCfg::default()).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }

    fn greedy(l: &mut Loaded, n: usize) -> Vec<usize> {
        let ids: Vec<usize> = l
            .tok
            .encode("hello there", true)
            .into_iter()
            .map(|i| i as usize)
            .collect();
        let mut st = l.model.new_state();
        let mut s = companion_engine::model::Sampler::new(0.0, 1, 1);
        let mut got = vec![];
        let opts = companion_engine::model::GenerateOptions {
            max_tokens: n,
            stop_tokens: vec![],
            cancel: None,
        };
        l.model
            .generate(&mut st, &ids, &mut s, &opts, &mut |t| {
                got.push(t);
                true
            })
            .unwrap();
        got
    }

    #[test]
    fn a_lora_adapter_changes_the_model_and_scale_zero_does_not() {
        let d = tmpdir("lora");
        write_model(&d, true);
        let ad = tmpdir("lora-ad");
        write_adapter(&ad, 77);
        let base = load(&d, &LoadCfg::default()).unwrap();
        let logits = |l: &Loaded| {
            let mut st = l.model.new_state();
            l.model.forward_batch(&mut st, &[65, 66, 67, 68]).unwrap()
        };
        let lb = logits(&base);
        let with = |scale: f32| {
            load(
                &d,
                &LoadCfg {
                    lora: vec![LoraSpec {
                        path: ad.clone(),
                        scale,
                    }],
                    ..LoadCfg::default()
                },
            )
            .unwrap()
        };
        let l1 = with(1.0);
        assert!(l1.warnings.iter().any(|w| w.contains("merged")));
        let (a, b) = (logits(&l1), lb.clone());
        assert!(
            a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-4),
            "the adapter had no effect"
        );
        assert_eq!(logits(&with(0.0)), lb, "scale 0 must be the base model");
        // doubling the scale moves further from the base than the single scale does
        let dist = |x: &[f32]| x.iter().zip(&lb).map(|(p, q)| (p - q).abs()).sum::<f32>();
        assert!(dist(&logits(&with(2.0))) > dist(&a));
        // an adapter folder without its files is a plain error
        assert!(
            load(
                &d,
                &LoadCfg {
                    lora: vec![LoraSpec {
                        path: d.join("missing"),
                        scale: 1.0
                    }],
                    ..LoadCfg::default()
                }
            )
            .is_err()
        );
        std::fs::remove_dir_all(d).unwrap();
        std::fs::remove_dir_all(ad).unwrap();
    }

    #[test]
    fn mixture_of_experts_models_load_and_run() {
        let d = tmpdir("moe");
        write_moe_model(&d, 4);
        let mut l = load(&d, &LoadCfg::default()).unwrap();
        assert!(l.model.spec.experts > 1);
        let a = greedy(&mut l, 6);
        assert_eq!(a.len(), 6);
        // a RAM budget that holds everything gives the same answer; an impossible one is an error, not a crash
        let mut l2 = load(
            &d,
            &LoadCfg {
                ram_budget: Some(1 << 30),
                ..LoadCfg::default()
            },
        )
        .unwrap();
        assert_eq!(greedy(&mut l2, 6), a);
        let e = load(
            &d,
            &LoadCfg {
                ram_budget: Some(64),
                ..LoadCfg::default()
            },
        )
        .unwrap_err();
        assert!(e.contains("fit") || e.contains("RAM"), "{e}");
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn template_lists_pick_the_default() {
        let j: Value = serde_json::json!({"chat_template": [{"name": "tool_use", "template": "A"}, {"name": "default", "template": "B"}]});
        assert_eq!(template_from_config(&j).as_deref(), Some("B"));
        assert_eq!(
            template_from_config(&serde_json::json!({"chat_template": "X"})).as_deref(),
            Some("X")
        );
        assert_eq!(template_from_config(&serde_json::json!({})), None);
    }
}
