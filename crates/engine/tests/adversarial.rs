#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]
//! Adversarial inputs for the engine: mutated valid files, absurd headers, nasty text, NaN/inf
//! numbers and extreme sizes. Every public parser and kernel must answer with a value or an error,
//! never panic, never hang, and never allocate in proportion to a size it merely read from the
//! input (see `hostile_gen` for the guards). `ADVERSARIAL_N` sets the number of random cases
//! (default 1500; 40000 for the long run).
mod hostile_gen;
use companion_engine::arch::{self, Spec};
use companion_engine::gguf::Gguf;
use companion_engine::loader::{GgufFile, LoadOptions, MoeOptions, Naming, SafeTensors, TensorSource, WeightFile, load_dense_with, load_model};
use companion_engine::model::{Sampler, State, Weights};
use companion_engine::quant::{self, DType};
use companion_engine::tokenizer::Tokenizer;
use companion_engine::{hub, hw, mla, moe, ops, plan, pool, qmatmul, simd, ssm};
use hostile_gen::*;
use std::path::Path;
use std::sync::Arc;

// ------------------------------------------------------------------------------- helpers

fn run_model(m: &companion_engine::model::Model) {
    let mut st: State = m.new_state();
    for tok in [0usize, 5, 19, 20, usize::MAX] { let _ = m.forward(&mut st, tok); }
    let mut st = m.new_state();
    let _ = m.forward_batch(&mut st, &[]);
    let _ = m.forward_batch(&mut st, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let mut st = m.new_state_with(companion_engine::model::KvKind::Q8);
    let _ = m.forward_batch(&mut st, &vec![3usize; 40]);
}

fn exercise_gguf(g: &Gguf) {
    let _ = g.architecture();
    let _ = g.get_u64("block_count");
    let _ = g.get_f64("rope.freq_base");
    let _ = g.total_tensor_bytes();
    if let Ok(s) = arch::from_gguf(g) { exercise_spec(&s); }
    let _ = plan::plan(&g.tensors, u64::MAX / 3, u64::MAX);
    let _ = plan::plan(&g.tensors, 1 << 20, 1 << 10);
    for t in &g.tensors { let _ = (t.elements(), t.bytes(), plan::role_of(&t.name)); }
    if let Ok(t) = Tokenizer::from_gguf(g) { exercise_tokenizer(&t); }
}

fn exercise_spec(s: &Spec) {
    let _ = s.kv_bytes_per_token();
    let _ = s.kv_bytes(u64::MAX);
    let _ = s.kv_bytes(1 << 40);
    let _ = s.expert_ffn();
    let _ = s.is_sliding(usize::MAX);
    let _ = s.is_sliding(0);
    let _ = arch::classify(&s.family);
}

/// A spec that claims huge sizes meets weights that are empty: an error, not an overflow.
fn spec_meets_empty_weights(s: &Spec) { let _ = companion_engine::model::Model::new(s.clone(), Weights::default()); }

#[test]

fn model_new_with_huge_spec_is_an_error() {
    hammer_rng("Model::new(hostile spec)", |r| {
        let cfg = wild_config(CONFIGS[r.below(CONFIGS.len() - 4)], r);
        if let Ok(sp) = arch::from_hf_config(&cfg) { spec_meets_empty_weights(&sp); }
    });
}

fn exercise_tokenizer(t: &Tokenizer) {
    let _ = t.vocab_size();
    for text in ["", "hello", "hello world", " \n\t", "😀 é e\u{301} 한글 <|endoftext|> <s> </s> ▁ Ġ <0x41>", "\0\u{feff}\u{202e}", "12345 6789", "'s 'RE n't"] {
        let ids = t.encode(text, false);
        let _ = t.encode(text, true);
        let _ = t.decode(&ids);
        let _ = t.decode_with(&ids, true);
        let mut d = t.decoder();
        for &i in &ids { let _ = d.push(i); }
        let _ = d.finish();
        let _ = t.pre_tokenize(text);
    }
    let junk = [u32::MAX, 0, 1, u32::MAX - 1, 255, 256, 257, 300_000, 1 << 31];
    let _ = t.decode(&junk);
    let mut d = t.decoder();
    for &i in &junk { let _ = d.push(i); }
    let _ = d.finish();
    let _ = (t.token_to_id("a"), t.id_to_token(u32::MAX), t.is_special(u32::MAX), t.warnings().len());
}

const KITCHEN_SINK: &str = r#"{"model_type":"qwen3_moe","hidden_size":2048,"num_hidden_layers":48,"num_attention_heads":32,"num_key_value_heads":4,
 "head_dim":128,"intermediate_size":6144,"moe_intermediate_size":768,"vocab_size":151936,"max_position_embeddings":40960,"rope_theta":1000000.0,
 "rms_norm_eps":1e-6,"num_experts":128,"num_experts_per_tok":8,"n_shared_experts":1,"n_routed_experts":128,"tie_word_embeddings":false,
 "sliding_window":4096,"layer_types":["sliding_attention","full_attention"],"rope_local_base_freq":10000.0,"attn_logit_softcapping":50.0,
 "final_logit_softcapping":30.0,"partial_rotary_factor":0.5,"norm_topk_prob":true,"routed_scaling_factor":2.5,"first_k_dense_replace":3,
 "decoder_sparse_step":1,"mlp_only_layers":[0,1],"rope_scaling":{"type":"yarn","factor":4.0,"original_max_position_embeddings":4096,
 "beta_fast":32.0,"beta_slow":1.0,"low_freq_factor":1.0,"high_freq_factor":4.0},"query_pre_attn_scalar":256,"hidden_act":"silu"}"#;

const CONFIGS: &[&str] = &[
    CONFIG_JSON,
    KITCHEN_SINK,
    r#"{"model_type":"mixtral","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"intermediate_size":14336,"vocab_size":32000,"num_local_experts":8,"num_experts_per_tok":2}"#,
    r#"{"model_type":"gemma3","text_config":{"model_type":"gemma3_text","hidden_size":2560,"num_hidden_layers":34,"num_attention_heads":8,"num_key_value_heads":4,"head_dim":256,"sliding_window":1024,"rope_local_base_freq":10000}}"#,
    r#"{"model_type":"phi3","hidden_size":3072,"num_hidden_layers":32,"num_attention_heads":32,"rope_scaling":{"type":"longrope","short_factor":[1.0,2.0],"long_factor":[3.0]}}"#,
    r#"{"model_type":"deepseek_v3","hidden_size":7168,"num_hidden_layers":61,"num_attention_heads":128,"kv_lora_rank":512,"q_lora_rank":1536,"qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,"n_routed_experts":256,"n_shared_experts":1,"num_experts_per_tok":8}"#,
    r#"{"model_type":"falcon","hidden_size":4544,"n_layer":32,"n_head":71,"multi_query":true}"#,
    r#"{"model_type":"llama","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"rope_scaling":{"rope_type":"llama3","factor":8.0,"low_freq_factor":1.0,"high_freq_factor":4.0,"original_max_position_embeddings":8192}}"#,
    "[1,2,3]", "null", "{}", "\"x\"",
];

// ------------------------------------------------------------------------------- gguf / files

#[test]
fn gguf_parse_mutated_files() {
    let seeds = vec![tiny_gguf(1, false), tiny_gguf(2, true), tokenizer_gguf("gpt2"), tokenizer_gguf("llama")];
    hammer_bytes("gguf::parse", &seeds, |b| {
        if let Ok(g) = Gguf::parse(b) { exercise_gguf(&g); }
    });
}

fn header(version: u32, tensors: u64, kvs: u64) -> Vec<u8> {
    let mut v = b"GGUF".to_vec();
    v.extend(version.to_le_bytes());
    v.extend(tensors.to_le_bytes());
    v.extend(kvs.to_le_bytes());
    v
}
fn s(v: &mut Vec<u8>, text: &str) { v.extend((text.len() as u64).to_le_bytes()); v.extend(text.as_bytes()); }

#[test]
fn gguf_absurd_headers_are_refused_without_allocating() {
    let mut cases: Vec<(&str, Vec<u8>)> = vec![];
    for (v, name) in [(0u32, "v0"), (1, "v1"), (4, "v4"), (u32::MAX, "vmax")] { cases.push((name, header(v, 0, 0))); }
    for n in [1u64 << 24, (1 << 24) + 1, 1 << 40, 1 << 60, u64::MAX] {
        cases.push(("n_tensors", header(3, n, 0)));
        cases.push(("n_kv", header(3, 0, n)));
    }
    // string length 2^60 and u64::MAX, array length 2^60, nested arrays, huge dims, absurd alignment
    for len in [1u64 << 24, (1 << 24) + 1, 1 << 40, 1 << 60, u64::MAX] {
        let mut v = header(3, 0, 1);
        v.extend(len.to_le_bytes());
        v.extend([b'k'; 16]);
        cases.push(("key length", v));
        let mut v = header(3, 0, 1);
        s(&mut v, "k");
        v.extend(8u32.to_le_bytes());
        v.extend(len.to_le_bytes());
        cases.push(("string value length", v));
        let mut v = header(3, 0, 1);
        s(&mut v, "k");
        v.extend(9u32.to_le_bytes());
        v.extend(0u32.to_le_bytes());
        v.extend(len.to_le_bytes());
        cases.push(("u8 array length", v));
        let mut v = header(3, 0, 1);
        s(&mut v, "k");
        v.extend(9u32.to_le_bytes());
        v.extend(9u32.to_le_bytes());
        v.extend(len.to_le_bytes());
        v.extend(vec![0u8; 4096]);
        cases.push(("array of arrays length", v));
    }
    let mut v = header(3, 0, 1);
    s(&mut v, "k");
    for _ in 0..200 { v.extend(9u32.to_le_bytes()); v.extend(9u32.to_le_bytes()); v.extend(1u64.to_le_bytes()); }
    cases.push(("deep nesting", v));
    for dims in [vec![u64::MAX, 2], vec![1 << 32, 1 << 32, 1 << 32], vec![1 << 63], vec![3; 9], vec![0, u64::MAX]] {
        for ty in [0u32, 1, 2, 8, 12, 14, 30, 99, u32::MAX] {
            let mut v = header(3, 1, 0);
            s(&mut v, "t");
            v.extend((dims.len() as u32).to_le_bytes());
            for d in &dims { v.extend(d.to_le_bytes()); }
            v.extend(ty.to_le_bytes());
            v.extend(u64::MAX.to_le_bytes());
            v.extend(vec![0u8; 64]);
            cases.push(("tensor dims", v));
        }
    }
    for al in [0u32, 1, 3, 31, 33, 1 << 30, u32::MAX] {
        let mut v = header(3, 0, 1);
        s(&mut v, "general.alignment");
        v.extend(4u32.to_le_bytes());
        v.extend(al.to_le_bytes());
        cases.push(("alignment", v));
    }
    for al in [0u64, 1 << 62, 1 << 63, u64::MAX] {
        let mut v = header(3, 0, 1);
        s(&mut v, "general.alignment");
        v.extend(10u32.to_le_bytes());
        v.extend(al.to_le_bytes());
        cases.push(("alignment u64", v));
    }
    for (name, bytes) in cases {
        once(name, || {
            if let Ok(g) = Gguf::parse(&bytes[..]) { exercise_gguf(&g); }
        });
    }
}

fn open_and_load_gguf(path: &Path) {
    let Ok(f) = GgufFile::open(path) else { return };
    for t in f.infos() {
        let _ = f.read(&t.name);
        let _ = f.read_raw(&t.name, false);
        let _ = f.read_raw(&t.name, true);
    }
    let _ = f.read("does.not.exist");
    exercise_gguf(&f.gguf);
    let spec = tiny_spec();
    for opts in [LoadOptions::default(), LoadOptions { prefer_quantized: true, mmap: false, ..Default::default() }, LoadOptions { prefer_quantized: true, mmap: true, ..Default::default() }] {
        for llama in [true, false] {
            if let Ok(m) = load_dense_with(spec.clone(), &f, Naming::Gguf, llama, opts) { run_model(&m); }
        }
    }
    let arc: Arc<dyn WeightFile> = Arc::new(f);
    for moe in [MoeOptions::default(), MoeOptions { ram_budget: Some(1 << 10), expert_cache: None, prefetch: true }, MoeOptions { ram_budget: None, expert_cache: Some(0), prefetch: false }] {
        if let Ok(m) = load_model(spec.clone(), arc.clone(), Naming::Gguf, true, LoadOptions::default(), moe) { run_model(&m); }
    }
}

#[test]
fn gguf_files_read_and_load_mutated_models() {
    let tmp = Tmp::new("gguf");
    let seeds = vec![tiny_gguf(1, false), tiny_gguf(2, true)];
    hammer_bytes("GgufFile::open+load", &seeds, |b| {
        let p = tmp.file("m.gguf", b);
        open_and_load_gguf(&p);
    });
}

fn open_and_load_st(path: &Path) {
    let Ok(st) = SafeTensors::open(path) else { return };
    let names: Vec<String> = st.names().map(str::to_string).collect();
    for n in names.iter().take(64) { let _ = st.read(n); let _ = st.read_raw(n, true); let _ = st.has(n); }
    let _ = st.read("nope");
    let spec = tiny_spec();
    for opts in [LoadOptions::default(), LoadOptions { prefer_quantized: true, mmap: true, ..Default::default() }] {
        if let Ok(m) = load_dense_with(spec.clone(), &st, Naming::Hf, false, opts) { run_model(&m); }
    }
    let arc: Arc<dyn WeightFile> = Arc::new(st);
    if let Ok(m) = load_model(spec, arc, Naming::Hf, false, LoadOptions::default(), MoeOptions { ram_budget: Some(1 << 12), expert_cache: None, prefetch: false }) { run_model(&m); }
}

#[test]
fn safetensors_files_read_and_load_mutated_models() {
    let tmp = Tmp::new("st");
    let seeds = vec![safetensors_bytes(1, false), safetensors_bytes(2, true)];
    hammer_bytes("SafeTensors::open+load", &seeds, |b| {
        let p = tmp.file("model.safetensors", b);
        open_and_load_st(&p);
    });
}

#[test]
fn safetensors_absurd_headers() {
    let tmp = Tmp::new("stabs");
    let mut cases: Vec<Vec<u8>> = vec![];
    for n in [0u64, 1, 7, 1 << 27, (1 << 27) + 1, 1 << 40, u64::MAX, u64::MAX - 7] {
        let mut v = n.to_le_bytes().to_vec();
        v.extend(vec![b'{'; 64]);
        cases.push(v);
    }
    for shape in ["[18446744073709551615,2]", "[4294967296,4294967296]", "[0]", "[]", "[-1]", "[1.5]", "\"x\"", "[1,2,3,4,5,6,7,8,9,10,11,12]"] {
        for off in ["[0,0]", "[0,4]", "[4,0]", "[0,18446744073709551615]", "[18446744073709551615,18446744073709551615]", "[0]", "null"] {
            let h = format!(r#"{{"a":{{"dtype":"F32","shape":{shape},"data_offsets":{off}}},"__metadata__":{{"x":"y"}}}}"#);
            let mut v = (h.len() as u64).to_le_bytes().to_vec();
            v.extend(h.as_bytes());
            v.extend([0u8; 64]);
            cases.push(v);
        }
    }
    for dt in ["F32", "F16", "BF16", "I8", "F64", "F8_E4M3", "", "\u{0}", "f32"] {
        let h = format!(r#"{{"a":{{"dtype":"{dt}","shape":[2],"data_offsets":[0,8]}}}}"#);
        let mut v = (h.len() as u64).to_le_bytes().to_vec();
        v.extend(h.as_bytes());
        v.extend([0u8; 16]);
        cases.push(v);
    }
    for (i, c) in cases.iter().enumerate() {
        once(&format!("safetensors case {i}"), || {
            let p = tmp.file("x.safetensors", c);
            open_and_load_st(&p);
        });
    }
}

#[test]
fn safetensors_index_files_cannot_escape_the_folder() {
    let tmp = Tmp::new("stidx");
    let outside = tmp.0.join("outside.safetensors");
    std::fs::write(&outside, safetensors_bytes(3, false)).unwrap();
    let dir = tmp.0.join("model");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.safetensors"), safetensors_bytes(4, false)).unwrap();
    let seed = r#"{"metadata":{"total_size":1},"weight_map":{"model.embed_tokens.weight":"model.safetensors","lm_head.weight":"model.safetensors"}}"#;
    hammer_str("SafeTensors::open(dir+index)", &[seed], |idx| {
        std::fs::write(dir.join("model.safetensors.index.json"), idx).unwrap();
        if let Ok(st) = SafeTensors::open(&dir) {
            // the real file inside the folder is fine; the one next to it must never be reached
            let n: Vec<String> = st.names().map(str::to_string).collect();
            let _ = n.len();
        }
    });
    for evil in ["../outside.safetensors", "/etc/passwd", "..\\outside.safetensors", "a/../../outside.safetensors", "", ".", "..", "\0x", "C:\\x", "//host/share/x"] {
        let idx = format!(r#"{{"weight_map":{{"a":{}}}}}"#, serde_json::to_string(evil).unwrap());
        std::fs::write(dir.join("model.safetensors.index.json"), idx).unwrap();
        if let Ok(st) = SafeTensors::open(&dir) {
            assert!(!st.names().any(|n| n == "model.embed_tokens.weight" && evil.contains("outside")),
                "an index entry {evil:?} reached a file outside the model folder");
        }
    }
}

// ------------------------------------------------------------------------------- architecture

#[test]
fn arch_from_hf_config_text() {
    hammer_str("arch::from_hf_config", CONFIGS, |s| {
        if let Ok(sp) = arch::from_hf_config(s) { exercise_spec(&sp); }
    });
}

fn wild_value(r: &mut Rng) -> serde_json::Value {
    use serde_json::Value;
    match r.below(10) {
        0 => Value::Null,
        1 => Value::Bool(r.chance(2)),
        2 => Value::from(r.extreme_u64()),
        3 => Value::from(-(r.extreme_u64() as i64 >> 1)),
        4 => serde_json::Number::from_f64([1e308, -1e308, 1e-320, 0.5, 4.0, 1e30][r.below(6)]).map(Value::Number).unwrap_or(Value::Null),
        5 => Value::String(random_text(r)),
        6 => Value::Array((0..r.below(5)).map(|_| Value::from(r.extreme_u64())).collect()),
        7 => serde_json::json!({"type": "yarn", "factor": 1e300, "original_max_position_embeddings": 0}),
        8 => Value::from(r.below(70)),
        _ => Value::from(r.extreme_u64() % 100_000),
    }
}

fn wild_config(seed: &str, r: &mut Rng) -> String {
    let mut v: serde_json::Value = serde_json::from_str(seed).unwrap_or(serde_json::Value::Null);
    fn walk(v: &mut serde_json::Value, r: &mut Rng) {
        match v {
            serde_json::Value::Object(m) => {
                for (_, x) in m.iter_mut() {
                    if r.chance(6) { *x = wild_value(r); } else { walk(x, r); }
                }
                if r.chance(8) { m.insert(random_text(r).chars().take(12).collect(), wild_value(r)); }
            }
            serde_json::Value::Array(a) => { for x in a.iter_mut() { if r.chance(4) { *x = wild_value(r); } } }
            _ => {}
        }
    }
    walk(&mut v, r);
    v.to_string()
}

#[test]
fn arch_from_hf_config_wild_numbers() {
    hammer_rng("arch::from_hf_config(structured)", |r| {
        let seed = CONFIGS[r.below(CONFIGS.len() - 4)];
        let cfg = wild_config(seed, r);
        if let Ok(sp) = arch::from_hf_config(&cfg) { exercise_spec(&sp); }
    });
}

// ------------------------------------------------------------------------------- hub

const TREE: &str = r#"[{"type":"file","path":"config.json","size":700,"oid":"a"},{"type":"file","path":"model-00001-of-00002.safetensors","size":4000000000,"lfs":{"oid":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","size":4000000000}},{"type":"directory","path":"sub"},{"type":"file","path":"x-Q4_K_M-00001-of-00003.gguf","size":18446744073709551615}]"#;
const INFO: &str = r#"{"id":"a/b","siblings":[{"rfilename":"config.json"},{"rfilename":"w-Q8_0.gguf","size":5},{"rfilename":"../x"}]}"#;
const MODELSCOPE: &str = r#"{"Data":{"Files":[{"Path":"a.gguf","Size":1,"Sha256":"00"},{"Name":"b","Type":"blob"}]}}"#;

fn files_from(r: &mut Rng) -> Vec<hub::RemoteFile> {
    const PARTS: &[&str] = &["model", "-00001-of-00002", "-00002-of-00002", "-00000-of-00000", "-99999-of-00001", ".safetensors", ".gguf", "Q4_K_M", "Q8_0", "F16", "BF16", "-",
        "_", ".", "/", "..", "é", "日本", "😀", "mmproj-", "imatrix", "config.json", "tokenizer.json", "model.safetensors.index.json", "a", "IQ4_XS", "\u{0301}", "\0", "\\", " ", "-of-"];
    (0..r.below(12)).map(|_| {
        let mut path = String::new();
        for _ in 0..1 + r.below(6) { path.push_str(PARTS[r.below(PARTS.len())]); }
        if r.chance(2) { path.push_str(if r.chance(2) { ".gguf" } else { ".safetensors" }); }
        hub::RemoteFile { path, size: r.extreme_u64(), sha256: r.chance(3).then(|| "0".repeat(r.below(70))) }
    }).collect()
}

#[test]
fn hub_parsers_and_selection() {
    hammer_str("hub::parse_tree", &[TREE, MODELSCOPE, INFO], |s| { let _ = hub::parse_tree(s); });
    hammer_str("hub::parse_model_info", &[INFO, TREE, MODELSCOPE], |s| { let _ = hub::parse_model_info(s); });
    hammer_str("hub::RepoSpec::parse", &["org/name", "org/name@rev", "hf://models/org/name@main", "https://huggingface.co/org/name/tree/main", "https://x/a/b/resolve/v1/c.gguf", "name"], |s| {
        if let Ok(spec) = hub::RepoSpec::parse(s) {
            for h in [hub::Hub::HuggingFace, hub::Hub::ModelScope, hub::Hub::HfCompatible("https://mirror.example".into()), hub::Hub::HfCompatible(s.into())] {
                let _ = (h.tree_url(&spec), h.info_url(&spec), h.file_url(&spec, "a/b c.gguf"), h.file_url(&spec, s));
            }
        }
    });
    hammer_rng("hub::select_files", |r| {
        let files = files_from(r);
        for format in [hub::Format::Auto, hub::Format::Safetensors, hub::Format::Gguf] {
            let mut sel = hub::Select { format, quants: vec![] };
            if r.chance(2) { sel.quants = (0..r.below(5)).map(|_| random_text(r)).collect(); } else { sel = hub::Select { format, ..Default::default() }; }
            if let Ok(chosen) = hub::select_files(&files, &sel) {
                let _ = hub::estimate_bytes(&chosen);
                for f in &chosen { let _ = hub::safe_join(Path::new("/tmp/dest"), &f.path); }
            }
        }
        let _ = hub::estimate_bytes(&files);
    });
}

#[test]
fn hub_safe_join_never_leaves_the_destination() {
    let dest = Path::new("/srv/models/dest");
    let check = |name: &str| {
        if let Ok(p) = hub::safe_join(dest, name) {
            assert!(p.starts_with(dest), "{name:?} gave {p:?} outside the destination");
            assert!(p != dest, "{name:?} resolved to the destination itself");
            assert!(!p.components().any(|c| matches!(c, std::path::Component::ParentDir | std::path::Component::CurDir)), "{name:?} gave {p:?}");
            assert!(!name.contains('\\') && !name.contains('\0'), "{name:?} accepted");
        }
    };
    for name in ["a", "a/b", "../a", "a/../b", "a/..", "/a", "//a", "a//b", "a/", "./a", "a/./b", "..\\a", "a\\b", "C:\\a", "C:a", "c:/a", "\\\\?\\C:\\a", "a\0b", "", ".", "..", "...", "a/...", "a./b", "a /b", "CON", "con.txt", "NUL", "a/COM1", "aux", "LPT9.gguf",
        "%2e%2e/a", "..%2fa", "a:stream", "a::$DATA", "\u{202e}a", "a\u{0}", "~/a", "$HOME/a", "a/\u{fe}/b"] {
        check(name);
    }
    hammer_str("hub::safe_join", &["a/b/c.gguf", "../../etc/passwd", "model/../../x"], check);
}

#[test]
fn hub_misc_pure_functions() {
    hammer_str("hub::classify", &["curl: (22) The requested URL returned error: 401", "curl: (6) Could not resolve host", "No space left on device", "SSL certificate problem"], |s| {
        for code in [None, Some(0), Some(1), Some(22), Some(-1), Some(i32::MAX), Some(i32::MIN)] {
            let _ = hub::classify(&hub::Tool::Curl("curl".into()), code, s);
            let _ = hub::classify(&hub::Tool::Wget("wget".into()), code, s);
        }
    });
    // SHA-256: hashing in random pieces equals hashing in one go
    hammer_rng("hub::Sha256 chunking", |r| {
        let data: Vec<u8> = (0..r.below(700)).map(|_| r.next_u64() as u8).collect();
        let one = hub::sha256_hex(&data);
        let mut h = hub::Sha256::new();
        let mut rest = &data[..];
        while !rest.is_empty() {
            let k = 1 + r.below(130).min(rest.len() - 1);
            h.update(&rest[..k.min(rest.len())]);
            rest = &rest[k.min(rest.len())..];
        }
        assert_eq!(hub::hex(&h.finish()), one);
    });
}

// ------------------------------------------------------------------------------- tokenizer

#[test]
fn tokenizer_from_json() {
    let (bl, spm) = (byte_level_tokenizer_json(), spm_tokenizer_json());
    hammer_str_cap("Tokenizer::from_json", std::time::Duration::from_secs(10), &[&bl, &spm, "{}", "[]", r#"{"model":{"type":"BPE","vocab":{},"merges":[]}}"#, r#"{"model":{"type":"WordPiece","vocab":{"a":0}}}"#], |s| {
        if let Ok(t) = Tokenizer::from_json(s) { exercise_tokenizer(&t); }
    });
}

#[test]
fn tokenizer_structured_hostile_json() {
    let bl = byte_level_tokenizer_json();
    hammer_rng("Tokenizer::from_json(structured)", |r| {
        let mut v: serde_json::Value = serde_json::from_str(&(if r.chance(2) { bl.clone() } else { spm_tokenizer_json() })).unwrap();
        // replace random values deep inside (ids, merges, added tokens, normalisers) with wild ones
        fn walk(v: &mut serde_json::Value, r: &mut Rng, depth: usize) {
            if depth > 6 { return; }
            match v {
                serde_json::Value::Object(m) => for (_, x) in m.iter_mut() { if r.chance(12) { *x = wild_value(r); } else { walk(x, r, depth + 1); } },
                serde_json::Value::Array(a) => {
                    for x in a.iter_mut() { if r.chance(30) { *x = wild_value(r); } else { walk(x, r, depth + 1); } }
                    if r.chance(10) { a.truncate(r.below(a.len() + 1)); }
                }
                _ => {}
            }
        }
        walk(&mut v, r, 0);
        if let Ok(t) = Tokenizer::from_json(&v.to_string()) { exercise_tokenizer(&t); }
    });
}

#[test]
fn tokenizer_from_gguf_mutated() {
    let seeds = vec![tokenizer_gguf("gpt2"), tokenizer_gguf("llama"), tokenizer_gguf("bert"), tiny_gguf(1, false)];
    hammer_bytes_cap("Tokenizer::from_gguf", std::time::Duration::from_secs(10), &seeds, |b| {
        if let Ok(g) = Gguf::parse(b) && let Ok(t) = Tokenizer::from_gguf(&g) { exercise_tokenizer(&t); }
    });
}

#[test]
fn byte_level_tokenizer_round_trips_any_text() {
    let t = Tokenizer::from_json(&byte_level_tokenizer_json()).expect("the byte-level test tokenizer loads");
    let check = |s: &str| {
        let ids = t.encode(s, false);
        assert_eq!(t.decode(&ids), s, "decode(encode(x)) != x for {:?}", s.chars().take(80).collect::<String>());
        // the streaming decoder gives the same text, whatever the token boundaries
        let mut d = t.decoder();
        let mut out = String::new();
        for &i in &ids { out.push_str(&d.push(i)); }
        out.push_str(&d.finish());
        assert_eq!(out, s, "streaming decode differs for {:?}", s.chars().take(80).collect::<String>());
        assert_eq!(t.encode(s, false), ids, "encoding is not deterministic");
    };
    for s in ["", " ", "hello", "hello hello  hello\n\n", "the then there", "😀😀", "e\u{301}", "한글", "\u{feff}\u{202e}x", "\0\0", "<|endoftext|>", "a<|im_start|>b", "\r\n\t", "   leading", "trailing   ", "1234567890", "don't I'm we're"] { check(s); }
    check(&"a".repeat(100_000));
    check(&"hello ".repeat(20_000));
    check(&"😀".repeat(20_000));
    hammer_str_cap("byte-level round trip", std::time::Duration::from_secs(10), &["hello world", "the 😀 thing"], check);
}

// ------------------------------------------------------------------------------- quant kernels

#[test]
fn quant_dequantize_and_dot_with_random_bytes() {
    // every half float decodes without panicking
    for h in 0..=u16::MAX { let _ = quant::f16_to_f32(h); }
    hammer_rng("quant::dequantize", |r| {
        let t = DType::from_id([0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 30, 99, u32::MAX, 4, 5, 9][r.below(19)]);
        let n: usize = match r.below(6) { 0 => 0, 1 => 1 + r.below(70), 2 => 256 * (1 + r.below(4)), 3 => 32 * (1 + r.below(9)), 4 => r.extreme_u64() as usize, _ => r.below(1000) };
        let len = match r.below(4) {
            0 => t.bytes_for(n as u64).map_or(r.below(300), |b| (b as usize).min(1 << 20)),
            1 => r.below(300),
            2 => t.bytes_for(n as u64).map_or(0, |b| (b as usize).saturating_sub(1).min(1 << 20)),
            _ => 0,
        };
        let src: Vec<u8> = (0..len).map(|_| r.next_u64() as u8).collect();
        let mut out = vec![1.0f32; r.below(3)];
        if let Ok(()) = quant::dequantize(t, &src, n, &mut out) { assert_eq!(out.len(), n, "{t:?}: decoded {} values, asked for {n}", out.len()) }
        let x: Vec<f32> = (0..n.min(2048)).map(|_| r.nasty_f32()).collect();
        let _ = quant::dot_row(t, &src, &x);
    });
}

#[test]
fn qmatmul_with_hostile_shapes_and_values() {
    hammer_rng("qmatmul::qmatvec", |r| {
        let types = [DType::F32, DType::F16, DType::BF16, DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::Q5_1, DType::Q8_0, DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K, DType::Q6K, DType::Q8K, DType::Other(77)];
        let t = types[r.below(types.len())];
        let (e, _) = t.geometry().unwrap_or((32, 1));
        let (rows, cols) = match r.below(5) {
            0 => (r.below(40), e * r.below(6)),
            1 => (r.extreme_u64() as usize, r.extreme_u64() as usize),
            2 => (r.below(5), r.below(700)),
            3 => (usize::MAX, 1 + r.below(8)),
            _ => (1 + r.below(24), e * (1 + r.below(5))),
        };
        let wlen = match (r.below(3), t.bytes_for(cols as u64)) {
            (0, Some(rb)) => (rb as usize).saturating_mul(rows).min(1 << 20),
            (1, Some(rb)) => (rb as usize).saturating_mul(rows).saturating_sub(1).min(1 << 20),
            _ => r.below(500),
        };
        let w: Vec<u8> = (0..wlen).map(|_| r.next_u64() as u8).collect();
        let xl = if r.chance(5) { r.below(50) } else { cols.min(4096) };
        let x: Vec<f32> = (0..xl).map(|_| r.nasty_f32()).collect();
        let mut out = vec![0.0f32; if r.chance(5) { r.below(9) } else { rows.min(256) }];
        let _ = qmatmul::qmatvec(t, &w, rows, cols, &x, &mut out);
        let _ = qmatmul::qmatvec_threads(t, &w, rows, cols, &x, &mut out, [0, 1, 3, 64, usize::MAX][r.below(5)]);
        let _ = qmatmul::qmatvec_exact(t, &w, rows, cols, &x, &mut out, [0, 1, 2][r.below(3)]);
        if let Ok(m) = qmatmul::QMatrix::new(t, rows, cols, w.clone()) {
            let _ = m.matvec(&x, &mut out);
            let _ = m.matvec_exact(&x, &mut out);
        }
    });
}

// ------------------------------------------------------------------------------- ssm / mla / moe

#[test]
fn ssm_config_parsers() {
    let seeds: &[&str] = &[
        r#"{"model_type":"mamba","hidden_size":768,"state_size":16,"conv_kernel":4,"expand":2,"time_step_rank":48}"#,
        r#"{"model_type":"falcon_mamba","hidden_size":4096,"intermediate_size":8192,"mixer_rms_eps":1e-6}"#,
        r#"{"model_type":"mamba2","hidden_size":2560,"num_heads":80,"head_dim":64,"state_size":128,"n_groups":1,"chunk_size":256}"#,
        r#"{"model_type":"rwkv6","hidden_size":2048,"head_size":64,"intermediate_size":7168,"layer_norm_epsilon":1e-5}"#,
        r#"{"model_type":"rwkv7","hidden_size":768,"head_dim":64,"decay_low_rank_dim":64,"a_low_rank_dim":64,"v_low_rank_dim":32,"gate_low_rank_dim":128,"value_residual":true}"#,
    ];
    hammer_str("ssm configs", seeds, |s| {
        let _ = ssm::Mamba1Config::from_hf_config(s);
        let _ = ssm::Mamba2Config::from_hf_config(s);
        let _ = ssm::Rwkv6Config::from_hf_config(s);
        let _ = ssm::Rwkv7Config::from_hf_config(s);
    });
    hammer_rng("ssm configs(structured)", |r| {
        let seed = seeds[r.below(seeds.len())];
        let c = wild_config(seed, r);
        let _ = ssm::Mamba1Config::from_hf_config(&c);
        let _ = ssm::Mamba2Config::from_hf_config(&c);
        let _ = ssm::Rwkv6Config::from_hf_config(&c);
        let _ = ssm::Rwkv7Config::from_hf_config(&c);
        // structs built directly with extreme fields must be validated, not overflow
        let mut u = || r.extreme_u64() as usize;
        let c1 = ssm::Mamba1Config { d_model: u(), d_inner: u(), d_state: u(), d_conv: u(), dt_rank: u(), conv_bias: true, bias: false, bcdt_rms_eps: None };
        let _ = c1.validate();
        let _ = ssm::Mamba1::new(c1, rand_m1_weights(0, 0, 0, 0, 0));
        let c2 = ssm::Mamba2Config { d_model: u(), n_heads: u(), head_dim: u(), d_state: u(), n_groups: u(), d_conv: u(), conv_bias: false, bias: false, norm_eps: 1e-5, norm_before_gate: false, chunk_size: u() };
        let _ = c2.validate();
        let c6 = ssm::Rwkv6Config { hidden: u(), head_size: u(), ffn: u(), ln_eps: 1e-5 };
        let _ = c6.validate();
    });
}

fn rand_vec(n: usize, r: &mut Rng) -> Vec<f32> { (0..n).map(|_| r.f32()).collect() }

fn rand_m1_weights(dm: usize, di: usize, n: usize, k: usize, rank: usize) -> ssm::Mamba1Weights {
    let mut r = Rng::new(7);
    ssm::Mamba1Weights {
        in_proj: rand_vec(2 * di * dm, &mut r), in_bias: None, conv_w: rand_vec(di * k, &mut r), conv_b: Some(rand_vec(di, &mut r)),
        x_proj: rand_vec((rank + 2 * n) * di, &mut r), dt_proj: rand_vec(di * rank, &mut r), dt_bias: rand_vec(di, &mut r),
        a_log: rand_vec(di * n, &mut r), d: rand_vec(di, &mut r), out_proj: rand_vec(dm * di, &mut r), out_bias: None,
    }
}

fn hostile_input(r: &mut Rng, n: usize) -> Vec<f32> {
    match r.below(4) {
        0 => (0..n).map(|_| r.nasty_f32()).collect(),
        1 => vec![f32::NAN; n],
        2 => vec![1e30; n],
        _ => rand_vec(n, r),
    }
}

#[test]
fn ssm_blocks_survive_nan_inf_and_wrong_lengths() {
    hammer_rng("ssm blocks", |r| {
        let (dm, di, n, k, rank) = (1 + r.below(8), 1 + r.below(10), 1 + r.below(5), 1 + r.below(5), 1 + r.below(4));
        let cfg = ssm::Mamba1Config { d_model: dm, d_inner: di, d_state: n, d_conv: k, dt_rank: rank, conv_bias: true, bias: false, bcdt_rms_eps: r.chance(3).then_some(1e-6) };
        let mut w = rand_m1_weights(dm, di, n, k, rank);
        if r.chance(6) { w.x_proj.pop(); }
        if r.chance(6) { w.a_log.push(1.0); }
        if let Ok(b) = ssm::Mamba1::new(cfg, w) {
            let mut st = b.new_state();
            for _ in 0..1 + r.below(6) {
                let len = if r.chance(5) { r.below(2 * dm + 2) } else { dm };
                let x = hostile_input(r, len);
                let _ = b.step(&mut st, &x);
                let nn = dm * r.below(5) + if r.chance(4) { 1 } else { 0 };
                let xs = hostile_input(r, nn);
                let _ = b.prefill(&mut st, &xs);
            }
            let _ = st.bytes();
        }
        let (nh, hd, ns, ng, kk) = (1 + r.below(4), 1 + r.below(4), 1 + r.below(4), 1 + r.below(2), 1 + r.below(4));
        let c2 = ssm::Mamba2Config { d_model: dm, n_heads: nh * ng, head_dim: hd * 2, d_state: ns, n_groups: ng, d_conv: kk, conv_bias: true, bias: false, norm_eps: 1e-5, norm_before_gate: r.chance(2), chunk_size: 1 + r.below(5) };
        if c2.validate().is_ok() {
            let (di2, pd, cd) = (c2.d_inner(), c2.proj_dim(), c2.conv_dim());
            let w2 = ssm::Mamba2Weights {
                in_proj: rand_vec(pd * dm, r), in_bias: None, conv_w: rand_vec(cd * kk, r), conv_b: None, dt_bias: rand_vec(c2.n_heads, r),
                a_log: rand_vec(c2.n_heads, r), d: rand_vec(c2.n_heads, r), norm_w: rand_vec(di2, r), out_proj: rand_vec(dm * di2, r), out_bias: None,
            };
            if let Ok(b) = ssm::Mamba2::new(c2, w2) {
                let mut st = b.new_state();
                for _ in 0..1 + r.below(5) {
                    let _ = b.step(&mut st, &hostile_input(r, dm));
                    let _ = b.step(&mut st, &hostile_input(r, dm + 1));
                    let nn = dm * r.below(9);
                    let _ = b.prefill(&mut st, &hostile_input(r, nn));
                }
            }
        }
    });
}

const MLA_CFG: &str = r#"{"hidden_size":7168,"num_attention_heads":128,"q_lora_rank":1536,"kv_lora_rank":512,"qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,"rms_norm_eps":1e-6,"rope_theta":10000,
 "rope_scaling":{"type":"yarn","factor":40,"original_max_position_embeddings":4096,"beta_fast":32,"beta_slow":1,"mscale":1.0,"mscale_all_dim":1.0}}"#;

#[test]
fn mla_config_and_block() {
    hammer_str("mla::from_hf_config", &[MLA_CFG, r#"{"text_config":{"hidden_size":8,"num_attention_heads":2,"kv_lora_rank":4,"qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4}}"#], |s| {
        let _ = mla::MlaConfig::from_hf_config(s);
    });
    hammer_rng("mla block", |r| {
        let c = wild_config(MLA_CFG, r);
        if let Ok(cfg) = mla::MlaConfig::from_hf_config(&c) { let _ = (cfg.latent_bytes_per_token(), cfg.full_bytes_per_token()); }
        let cfg = mla::MlaConfig {
            hidden: 1 + r.below(10), heads: 1 + r.below(4), q_lora_rank: r.chance(2).then(|| 1 + r.below(5)), kv_lora_rank: 1 + r.below(6),
            qk_nope: 1 + r.below(5), qk_rope: 2 * (1 + r.below(3)), v_head: 1 + r.below(5), eps: 1e-6, rope_theta: 10_000.0,
            yarn: r.chance(2).then(|| mla::Yarn { factor: 1.0 + r.below(40) as f64, original_max_position: 64.0, beta_fast: 32.0, beta_slow: 1.0, mscale: 1.0, mscale_all_dim: r.below(2) as f64 }),
        };
        let q = cfg.qk_nope + cfg.qk_rope;
        let mut w = mla::MlaWeights {
            kv_a: rand_vec((cfg.kv_lora_rank + cfg.qk_rope) * cfg.hidden, r), kv_a_norm: rand_vec(cfg.kv_lora_rank, r),
            kv_b: rand_vec(cfg.heads * (cfg.qk_nope + cfg.v_head) * cfg.kv_lora_rank, r), o: rand_vec(cfg.hidden * cfg.heads * cfg.v_head, r), ..Default::default()
        };
        match cfg.q_lora_rank {
            Some(rank) => { w.q_a = rand_vec(rank * cfg.hidden, r); w.q_a_norm = rand_vec(rank, r); w.q_b = rand_vec(cfg.heads * q * rank, r); }
            None => w.q_proj = rand_vec(cfg.heads * q * cfg.hidden, r),
        }
        if r.chance(6) { w.o.pop(); }
        if r.chance(8) { w.kv_b.clear(); }
        let hidden = cfg.hidden;
        if let Ok(m) = mla::Mla::new(cfg, w) {
            let (mut c1, mut c2) = (mla::MlaCache::new(), mla::NaiveCache::new());
            for pos in 0..1 + r.below(5) {
                let hl = if r.chance(6) { hidden + 1 } else { hidden };
                let x = hostile_input(r, hl);
                let p = if r.chance(5) { r.extreme_u64() as usize } else { pos };
                let _ = m.forward(&mut c1, &x, p);
                let _ = m.forward_naive(&mut c2, &x, p);
            }
            let nn = hidden * 3 + r.below(2);
            let _ = m.forward_batch(&mut c1, &hostile_input(r, nn));
            let _ = (c1.bytes(), c2.bytes(), c1.len());
        }
    });
}

#[test]
fn moe_router_and_block_with_hostile_inputs() {
    use moe::ExpertProvider;
    hammer_rng("moe::top_k", |r| {
        let scores: Vec<f32> = (0..r.below(40)).map(|_| r.nasty_f32()).collect();
        let k = [0usize, 1, 2, 5, 100, usize::MAX][r.below(6)];
        let got = moe::top_k(&scores, k);
        assert!(got.len() <= k.min(scores.len()));
        assert!(got.iter().all(|&i| i < scores.len()), "index out of range");
        let mut d = got.clone();
        d.sort_unstable();
        d.dedup();
        assert_eq!(d.len(), got.len(), "duplicate expert chosen");
    });
    hammer_rng("moe::Router/MoeBlock", |r| {
        let (hidden, experts) = (1 + r.below(6), 1 + r.below(8));
        let mut cfg = moe::RouterConfig::softmax(hidden, experts, r.below(experts + 3));
        if r.chance(2) { cfg.scoring = moe::Scoring::Sigmoid; }
        cfg.renormalise = r.chance(2);
        cfg.routed_scale = r.nasty_f32();
        if r.chance(3) { cfg.groups = Some((r.below(5), r.below(5))); }
        if r.chance(3) { cfg.selection_bias = Some(rand_vec(if r.chance(5) { experts + 1 } else { experts }, r)); }
        let wlen = if r.chance(8) { r.below(50) } else { hidden * experts };
        let router = match moe::Router::new(cfg.clone(), (0..wlen).map(|_| r.nasty_f32()).collect()) { Ok(x) => x, Err(_) => return };
        let ffn = 1 + r.below(6);
        let mk = |r: &mut Rng, bad: bool| moe::ExpertWeights::dense(rand_vec(hidden * ffn, r), rand_vec(hidden * ffn + bad as usize, r), rand_vec(hidden * ffn, r));
        let nl = experts - r.below(2).min(experts - 1);
        let layer: Vec<_> = (0..nl).map(|_| { let b = r.chance(10); mk(r, b) }).collect();
        let shared: Vec<_> = if r.chance(3) { let b = r.chance(5); vec![mk(r, b)] } else { vec![] };
        let Ok(block) = moe::MoeBlock::new(0, router, shared) else { return };
        let prov = moe::InMemoryExperts::new(vec![layer]);
        for _ in 0..3 {
            let n = r.below(4);
            let nn = hidden * n + if r.chance(6) { 1 } else { 0 };
            let xs = hostile_input(r, nn);
            let _ = block.forward_batch(&prov, &xs, n);
            let _ = block.forward_batch(&prov, &xs, usize::MAX);
            let _ = block.forward(&prov, &hostile_input(r, hidden));
        }
        prov.prefetch(usize::MAX, &[usize::MAX, 0]);
        let _ = prov.expert(usize::MAX, usize::MAX);
    });
}

// ------------------------------------------------------------------------------- hw / plan

#[test]
fn hw_parsers_accept_any_text() {
    let seeds: &[&str] = &[
        "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Core(TM) i7\ncpu cores\t: 4\nflags\t\t: fpu avx2 fma\n",
        "MemTotal:       16318484 kB\nMemAvailable:   10243992 kB\n",
        "0-3,8-11", "max 100000", "100000 100000", "9223372036854775807", "512K", "32768K",
        "index, name, memory.total [MiB]\n0, NVIDIA GeForce RTX 4090, 24564 MiB\n",
        "GPU[0]\t\t: Card series:\t\tAMD Radeon\nGPU[0]\t\t: VRAM Total Memory (B): 17163091968\n",
        "Graphics/Displays:\n\n    Apple M2:\n      Chipset Model: Apple M2\n      VRAM (Total): 8 GB\n",
        "hw.memsize: 17179869184\nhw.perflevel0.logicalcpu: 4\n", "Pages free: 12345.\nPage size: 16384 bytes", "Name=Intel\nNumberOfCores=8\nNumberOfLogicalProcessors=16\n",
        "Filesystem 1K-blocks Used Available Use% Mounted on\n/dev/nvme0n1p2 100 50 50 50% /\n", "nvme0n1p2", "[always] madvise never", "0::/user.slice/x.scope",
    ];
    hammer_str("hw parsers", seeds, |s| {
        let _ = (hw::parse_cpuinfo(s), hw::parse_meminfo(s), hw::parse_cpulist(s), hw::parse_size(s), hw::parse_cgroup_v2_cpu_max(s), hw::parse_cgroup_mem(s));
        let _ = (hw::parse_cgroup_v1_cpu(s, s), hw::parse_cgroup_v1_cpu(s, "0"), hw::parse_cgroup_v1_cpu("-1", s), hw::parse_proc_self_cgroup(s), hw::parse_thp(s));
        let lines: Vec<String> = s.lines().map(str::to_string).collect();
        let _ = (hw::physical_from_sibling_lists(&lines), hw::count_numa_nodes(&lines));
        let _ = (hw::parse_nvidia_smi(s), hw::parse_rocm_smi(s), hw::parse_system_profiler(s), hw::parse_sysctl_pairs(s), hw::parse_vm_stat(s), hw::parse_kv_lines(s));
        let _ = (hw::parse_win_cpu(s), hw::parse_win_mem(s), hw::parse_win_gpus(s), hw::parse_df_device(s), hw::block_base_name(s), hw::parse_rotational(s), hw::gpu_kind_from_pci_vendor(s));
        let entries: Vec<(u32, String, String)> = lines.iter().enumerate().map(|(i, l)| ((i as u32).wrapping_mul(1_000_000_007), l.clone(), l.clone())).collect();
        let _ = hw::caches_from_entries(&entries);
    });
    hammer_rng("hw numbers", |r| {
        let caps: Vec<u32> = (0..r.below(20)).map(|_| r.extreme_u64() as u32).collect();
        let _ = hw::hybrid_from_capacities(&caps);
        let quota = [None, Some(f64::NAN), Some(-1.0), Some(0.0), Some(1e300), Some(f64::INFINITY), Some(0.3)][r.below(7)];
        let t = hw::choose_threads([None, Some(0), Some(1), Some(usize::MAX)][r.below(4)], r.extreme_u64() as usize, quota);
        assert!(t >= 1, "choose_threads returned {t}");
        let mut p = hw::HardwareProfile::default();
        p.total_ram = r.chance(2).then(|| r.extreme_u64());
        p.available_ram = r.chance(2).then(|| r.extreme_u64());
        p.limits = hw::Limits { cpu_cores: quota, memory: r.chance(2).then(|| r.extreme_u64()) };
        p.usable_threads = r.extreme_u64() as usize;
        p.numa_nodes = r.extreme_u64() as usize;
        p.cpu.logical = r.extreme_u64() as usize;
        p.cpu.physical = r.chance(2).then(|| r.extreme_u64() as usize);
        let t = hw::Tuning::recommend(&p, r.extreme_u64(), r.extreme_u64());
        assert!(t.threads >= 1 && t.prefill_batch >= 1, "unusable tuning: {t:?}");
        let _ = t.explain();
    });
}

#[test]
fn plan_with_extreme_sizes() {
    hammer_rng("plan::plan", |r| {
        let tensors: Vec<companion_engine::gguf::TensorInfo> = (0..r.below(8)).map(|_| companion_engine::gguf::TensorInfo {
            name: ["model.layers.0.mlp.experts.3.w", "blk.0.ffn_down_exps.weight", "token_embd.weight", "x", "blk.1.attn_q.weight"][r.below(5)].to_string(),
            dims: (0..r.below(5)).map(|_| r.extreme_u64()).collect(),
            dtype: DType::from_id(r.below(40) as u32),
            offset: r.extreme_u64(),
        }).collect();
        let p = plan::plan(&tensors, r.extreme_u64(), r.extreme_u64());
        // the plan never claims more cache than there is streamed data
        assert!(p.expert_cache <= p.streamed, "expert cache {} exceeds the streamed bytes {}", p.expert_cache, p.streamed);
    });
    hammer_str("plan::role_of", &["model.layers.3.mlp.experts.17.down_proj.weight", "blk.3.ffn_down_exps.weight"], |s| { let _ = plan::role_of(s); });
}

// ------------------------------------------------------------------------------- sampler / kernels

#[test]
fn sampler_with_nan_inf_and_extreme_settings() {
    hammer_rng("Sampler::sample", |r| {
        let temp = [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e-30, 1.0, 100.0, 1e30][r.below(9)];
        let mut s = Sampler::new(temp, [0usize, 1, 3, 1000, usize::MAX][r.below(5)], r.next_u64());
        if r.chance(2) { s = s.with_top_p([0.0, -1.0, f32::NAN, 0.5, 0.95, 1.0, 5.0][r.below(7)]); }
        if r.chance(2) { s = s.with_repetition_penalty([0.0, -1.0, f32::NAN, 1.0, 1.3, f32::INFINITY][r.below(6)], [0usize, 1, 64, usize::MAX][r.below(4)]); }
        for _ in 0..r.below(20) { s.observe([0usize, 1, 5, usize::MAX, 1 << 40][r.below(5)]); }
        for _ in 0..3 {
            let n = [0usize, 1, 2, 7, 300][r.below(5)];
            let logits: Vec<f32> = (0..n).map(|_| if r.chance(2) { r.nasty_f32() } else { r.f32() * 10.0 }).collect();
            let i = s.sample(&logits);
            assert!(n == 0 || i < n, "sampled {i} from {n} logits (temp {temp}, logits {logits:?})");
        }
        s.clear_history();
    });
}

#[test]
fn pool_with_extreme_arguments() {
    let p = pool::Pool::new(4);
    for (n, min) in [(0usize, 0usize), (1, 0), (10, 0), (10, usize::MAX), (1000, 1), (1000, 333), (usize::MAX, usize::MAX), (12345, 7), (1 << 20, 1 << 10)] {
        once("pool::parallel_for", || {
            let sum = std::sync::atomic::AtomicUsize::new(0);
            let calls = std::sync::atomic::AtomicUsize::new(0);
            p.parallel_for(n, min, |range| {
                assert!(range.start <= range.end && range.end <= n, "chunk {range:?} outside 0..{n}");
                sum.fetch_add(range.len(), std::sync::atomic::Ordering::Relaxed);
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            });
            assert_eq!(sum.load(std::sync::atomic::Ordering::Relaxed), n, "n={n} min_chunk={min}: items not covered exactly once");
        });
    }
    for len in [0usize, 1, 5, 1000] {
        for min in [0usize, 1, 7, usize::MAX] {
            let mut v = vec![0u32; len];
            p.parallel_chunks_mut(&mut v, min, |start, c| { for (i, x) in c.iter_mut().enumerate() { *x += (start + i) as u32 + 1; } });
            assert!(v.iter().enumerate().all(|(i, &x)| x == i as u32 + 1), "len={len} min={min}: {v:?}");
        }
    }
    for n in [0usize, 1, 2, 64] { let _ = pool::Pool::new(n).threads(); }
    // a panic in a chunk reaches the caller and leaves the pool usable
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.parallel_for(1000, 1, |r| { if r.contains(&500) { panic!("boom") } })));
    assert!(r.is_err());
    let c = std::sync::atomic::AtomicUsize::new(0);
    p.parallel_for(100, 1, |r| { c.fetch_add(r.len(), std::sync::atomic::Ordering::Relaxed); });
    assert_eq!(c.load(std::sync::atomic::Ordering::Relaxed), 100);
}

#[test]
fn ops_and_simd_kernels_with_nasty_values() {
    hammer_rng("ops/simd", |r| {
        let n = [0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 1000][r.below(18)];
        let a: Vec<f32> = (0..n).map(|_| r.nasty_f32()).collect();
        let b: Vec<f32> = (0..n).map(|_| r.nasty_f32()).collect();
        let w: Vec<f32> = (0..n).map(|_| r.nasty_f32()).collect();
        let _ = (simd::dot_f32(&a, &b), simd::sumsq(&a), simd::max(&a));
        let mut y = b.clone();
        simd::axpy(r.nasty_f32(), &a, &mut y);
        simd::scale(&mut y, r.nasty_f32());
        let mut out = vec![0.0; n];
        simd::mul3(&a, &w, r.nasty_f32(), &mut out);
        let mut g = a.clone();
        simd::silu_mul(&mut g, &b);
        let mut e = a.clone();
        let _ = simd::exp_sub_sum(&mut e, r.nasty_f32());
        let mut o2 = vec![0.0; n];
        ops::rmsnorm(&a, &w, [0.0, 1e-5, f32::NAN, -1.0][r.below(4)], &mut o2);
        let mut sm = a.clone();
        ops::softmax(&mut sm);
        let mut sm2 = a.clone();
        simd::set_precise(r.chance(2));
        ops::softmax(&mut sm2);
        simd::set_precise(false);
        let mut head = a.clone();
        ops::rope(&mut head, [0usize, 1, 1 << 20, usize::MAX][r.below(4)], [10_000.0, 0.0, -5.0, f64::NAN, 1e300, 1.0][r.below(6)]);
        let ia: Vec<i8> = (0..n).map(|_| r.next_u64() as i8).collect();
        let ib: Vec<i8> = (0..n).map(|_| r.next_u64() as i8).collect();
        let want: i32 = ia.iter().zip(&ib).map(|(x, y)| *x as i32 * *y as i32).sum();
        assert_eq!(simd::dot_i8_i8(&ia, &ib), want, "dot_i8_i8 wrong for n={n}");
        let half: Vec<u16> = (0..n).map(|_| r.next_u64() as u16).collect();
        let mut f = vec![0.0; n];
        simd::f16_to_f32(&half, &mut f);
        for (h, x) in half.iter().zip(&f) {
            let want = quant::f16_to_f32(*h);
            assert!(x.to_bits() == want.to_bits() || (x.is_nan() && want.is_nan()), "f16 {h:#06x}: simd {x} vs scalar {want}");
        }
        simd::bf16_to_f32(&half, &mut f);
        let rows = 1 + r.below(5);
        let m: Vec<f32> = (0..rows * n).map(|_| r.nasty_f32()).collect();
        let mut mo = vec![0.0; rows];
        if n > 0 { ops::matvec(&m, n, &a, &mut mo); }
    });
}
