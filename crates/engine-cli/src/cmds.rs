//! The commands `corbul` adds around loading and generating: `plan`, `doctor`, `capabilities`, `worker`,
//! `run`, `verify-receipt`, `replay`, and the shared generation path used by `generate`, `chat` and `run`.
use crate::args::{Args, parse_size};
use crate::constraint::{self, Spec as OutSpec};
use crate::load::{self, LoadCfg, LoraSpec, Prepared};
use crate::quotes;
use crate::runner::{
    self, Engine, GenCfg, GenOut, GovernorLoop, KvPolicyCfg, LocalEngine, MeshEngine, StreamEngine,
};
use crate::template::Message;
use companion_engine::arch::Spec;
use companion_engine::governor::{
    Autopilot, Calibration, Meter, ModelInfo, Recommendation, Sensors, SystemSensors, Verdict,
};
use companion_engine::hw::{HardwareProfile, KvPrecision};
use companion_engine::lang::{self, Lang, LangLock};
use companion_engine::loader::LoadOptions;
use companion_engine::mesh::{self, Mesh, MeshConfig, Wire};
use companion_engine::model::{KvKind, Model};
use companion_engine::receipt::{self, GenerateSpec, Receipt, RecordOptions};
use companion_engine::simd;
use companion_engine::stream_layers::{StreamedModel, plan_layers};
use std::io::Write;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

pub fn gib(b: u64) -> String {
    format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
}

// ------------------------------------------------------------------------------ shared options

pub fn kv_kind(args: &Args) -> Result<KvKind, String> {
    match args.get("kv") {
        None | Some("f32") => Ok(KvKind::F32),
        Some("f16") => Ok(KvKind::F16),
        Some("q8") => Ok(KvKind::Q8),
        Some(o) => Err(format!("--kv: `{o}` (use f32, f16 or q8)")),
    }
}

/// `--lora PATH` or `--lora PATH:SCALE` (repeatable). A drive letter such as `C:\x` is not a scale.
pub fn parse_lora(s: &str) -> Result<LoraSpec, String> {
    if let Some((p, sc)) = s.rsplit_once(':')
        && !p.is_empty()
        && let Ok(v) = sc.parse::<f32>()
    {
        if !v.is_finite() {
            return Err(format!("--lora {s}: the scale must be a finite number"));
        }
        return Ok(LoraSpec {
            path: p.into(),
            scale: v,
        });
    }
    Ok(LoraSpec {
        path: s.into(),
        scale: 1.0,
    })
}

pub fn load_cfg(args: &Args) -> Result<LoadCfg, String> {
    Ok(LoadCfg {
        prefer_quantized: true,
        mmap: !args.flag("no-mmap"),
        kv: kv_kind(args)?,
        ram_budget: args.get("ram-budget").map(parse_size).transpose()?,
        lora: args
            .all("lora")
            .into_iter()
            .map(parse_lora)
            .collect::<Result<_, _>>()?,
    })
}

/// Settings that act on the whole process: `--reproducible` (bit-identical results on every machine).
pub fn apply_global(args: &Args) {
    if args.flag("reproducible") {
        simd::set_reproducible(true);
        eprintln!(
            "reproducible mode: results are bit-identical on every machine and thread count, at some cost in speed"
        );
    }
}

fn weights_bytes_of(prep: &Prepared) -> u64 {
    prep.disk_bytes
}

/// The prompt's tokens: the chat template applied to the messages, or the text as it is with `--raw`.
pub fn prompt_ids(
    prep: &Prepared,
    msgs: &[Message],
    raw: Option<&str>,
) -> Result<Vec<usize>, String> {
    let ids = match raw {
        Some(t) => prep.tok.encode(t, true),
        None => {
            let text = prep
                .template
                .render(msgs, true, &prep.bos, &prep.eos)
                .map_err(|e| e.to_string())?;
            prep.tok.encode(&text, false)
        }
    };
    Ok(ids.into_iter().map(|i| i as usize).collect())
}

pub(crate) fn read_sources(args: &Args) -> Result<Vec<String>, String> {
    args.all("sources")
        .into_iter()
        .map(|f| std::fs::read_to_string(f).map_err(|e| format!("--sources {f}: {e}")))
        .collect()
}

// ------------------------------------------------------------------------------ engines

/// Error out on options (`keys` take a value, `flags` do not) that the loop named by `what` would silently ignore.
pub(crate) fn refuse_ignored(
    a: &Args,
    what: &str,
    keys: &[&str],
    flags: &[&str],
) -> Result<(), String> {
    if let Some(k) = keys.iter().find(|k| a.get(k).is_some()) {
        return Err(format!(
            "--{k} cannot be combined with {what} (it would be ignored)"
        ));
    }
    if let Some(f) = flags.iter().find(|f| a.flag(f)) {
        return Err(format!(
            "--{f} cannot be combined with {what} (it would be ignored)"
        ));
    }
    Ok(())
}

/// What the receipt recorder honours is temperature, top-k, top-p, repetition penalty and seed: nothing else.
const RECEIPT_IGNORES: [&str; 17] = [
    "min-p",
    "typical-p",
    "presence-penalty",
    "frequency-penalty",
    "mirostat",
    "mirostat-tau",
    "mirostat-eta",
    "dry-multiplier",
    "xtc-probability",
    "sampler-json",
    "lang",
    "draft",
    "skip-layers",
    "draft-len",
    "ngram-max",
    "ngram-min",
    "citations",
];

pub(crate) enum Mode {
    Local,
    Stream,
    Mesh,
}

pub(crate) fn mode_of(args: &Args) -> Result<Mode, String> {
    let (stream, mesh) = (
        args.flag("stream-layers"),
        args.get("workers").is_some() || args.flag("discover"),
    );
    if stream && mesh {
        return Err("--stream-layers and --workers/--discover cannot be combined".into());
    }
    Ok(if stream {
        Mode::Stream
    } else if mesh {
        Mode::Mesh
    } else {
        Mode::Local
    })
}

/// A warning (not an error) for a short shared secret; `None` when it is long enough.
pub fn weak_secret_warning(what: &str, s: &str) -> Option<String> {
    (s.chars().count() < 16).then(|| format!("warning: the {what} is shorter than 16 characters; use a long random one (for example `openssl rand -hex 24`)"))
}

fn mesh_key(args: &Args) -> Result<Vec<u8>, String> {
    let k = args.secret("key", "key-file", Some("CORBUL_MESH_KEY"))?.ok_or("--key SECRET, --key-file FILE or CORBUL_MESH_KEY is required: it is how machines prove they belong to the same mesh")?;
    if k.len() < 8 {
        return Err("the mesh key must be at least 8 characters".into());
    }
    if let Some(w) = weak_secret_warning("mesh key", &k) {
        eprintln!("{w}");
    }
    Ok(k.into_bytes())
}

fn resolve_addrs(list: &str) -> Result<Vec<SocketAddr>, String> {
    list.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.to_socket_addrs()
                .map_err(|e| format!("{s}: {e}"))
                .and_then(|mut a| a.next().ok_or_else(|| format!("{s}: no address")))
        })
        .collect()
}

pub(crate) fn build_model_with_threads(
    prep: &Prepared,
    cfg: &LoadCfg,
    args: &Args,
) -> Result<Model, String> {
    let t = std::time::Instant::now();
    let mut m = prep.build_model(cfg)?;
    if let Some(n) = args.num::<usize>("threads")? {
        m.set_threads(n);
    }
    if let Some(c) = args.num::<u64>("ctx")?.filter(|c| *c > 0) {
        m.spec.context = m.spec.context.min(c as _);
    }
    eprintln!(
        "loaded {} in {:.1}s ({}, context {}, template: {})",
        prep.name,
        t.elapsed().as_secs_f64(),
        gib(m.weight_bytes() as u64),
        m.spec.context,
        prep.builtin.name()
    );
    Ok(m)
}

pub(crate) fn open_prepared(args: &Args) -> Result<(Prepared, LoadCfg), String> {
    let path = args
        .pos
        .first()
        .ok_or("give a model folder or .gguf file")?;
    let cfg = load_cfg(args)?;
    let prep = load::prepare(Path::new(path), &cfg)?;
    for w in &prep.warnings {
        eprintln!("warning: {w}");
    }
    Ok((prep, cfg))
}

/// Open the layer-streaming engine; falls back to `None` (load normally) when everything fits.
fn open_stream(
    prep: &Prepared,
    cfg: &LoadCfg,
    args: &Args,
) -> Result<Option<StreamEngine>, String> {
    let hp = HardwareProfile::detect(Some(&prep.path));
    let budget = cfg
        .ram_budget
        .or(hp.available_ram)
        .ok_or("--stream-layers needs --ram-budget SIZE (the free RAM could not be read)")?;
    let opts = LoadOptions {
        prefer_quantized: true,
        mmap: cfg.mmap,
        threads: args.num("threads")?.unwrap_or(0),
    };
    let (spec, src) = (prep.spec.clone(), prep.src.clone());
    let probe = StreamedModel::open(
        spec.clone(),
        src.clone(),
        prep.naming,
        prep.permuted,
        opts,
        1,
    )
    .map_err(|e| e.to_string())?;
    let kv = spec.kv_bytes(spec.context.min(8192));
    let disk = args.num::<f64>("disk-mbps")?.unwrap_or(500.0);
    let plan = plan_layers(
        spec.layers as usize,
        probe.layer_bytes() as u64,
        probe.outer_bytes() as u64,
        kv,
        budget,
        disk,
        args.num("stream-window")?.unwrap_or(4),
    )
    .map_err(|missing| {
        format!(
            "not even one layer fits the {} RAM budget (short by {})",
            gib(budget),
            gib(missing)
        )
    })?;
    if plan.all_resident && args.get("stream-window").is_none() {
        eprintln!(
            "note: the whole model fits the {} budget; loading it normally instead of streaming",
            gib(budget)
        );
        return Ok(None);
    }
    let m = if plan.window_layers > 1 {
        drop(probe);
        StreamedModel::open(
            spec,
            src,
            prep.naming,
            prep.permuted,
            opts,
            plan.window_layers,
        )
        .map_err(|e| e.to_string())?
    } else {
        probe
    };
    eprintln!("layer streaming: {} layers at once ({}), {} always resident{}", plan.window_layers, gib(plan.budget_bytes), gib(plan.resident_bytes),
        plan.decode_tps.map_or(String::new(), |t| format!(" · about {t:.2} tokens/s if the disk gives {disk:.0} MB/s (assumed; --disk-mbps to change)")));
    Ok(Some(StreamEngine::new(m, cfg.kv)))
}

// ------------------------------------------------------------------------------ generate

fn print_notes(e: &mut dyn Engine) {
    for n in e.take_notes() {
        eprintln!("{n}");
    }
}

/// One reply, with the engine chosen by the options.
#[allow(clippy::too_many_arguments)]
fn generate_with(
    args: &Args,
    prep: &Prepared,
    cfg: &LoadCfg,
    ids: &[usize],
    gcfg: &GenCfg,
    cons: Option<Box<dyn crate::genloop::Constraint>>,
    cancel: &AtomicBool,
    echo: bool,
    session: Option<&mut ChatEngine>,
) -> Result<GenOut, String> {
    let mut sink = |p: &str| {
        if echo {
            let o = std::io::stdout();
            let mut o = o.lock();
            let _ = o.write_all(p.as_bytes());
            let _ = o.flush();
        }
    };
    let mut gov = if args.flag("governor") {
        let hp = HardwareProfile::detect(Some(&prep.path));
        Some(GovernorLoop::new(
            &hp,
            weights_bytes_of(prep),
            prep.spec.kv_bytes(prep.spec.context.min(8192)),
            args.flag("verbose"),
        ))
    } else {
        None
    };
    if let Some(ChatEngine::Local { model, st }) = session {
        // a chat keeps its model and cache between turns, so the next prompt reuses what was already computed
        let state = st
            .take()
            .ok_or("the chat state was lost after an earlier error; use /reset")?;
        let mut e = LocalEngine::new(model, state, None, None);
        let r = runner::run(
            &mut e,
            &prep.tok,
            ids,
            gcfg,
            cons,
            gov.as_mut(),
            cancel,
            &mut sink,
        );
        *st = Some(e.st);
        return r;
    }
    match mode_of(args)? {
        Mode::Local => {
            let mut model = build_model_with_threads(prep, cfg, args)?;
            let store = runner::open_sessions(args)?;
            let st = model.new_state_with(cfg.kv);
            let mut e = LocalEngine::new(&mut model, st, KvPolicyCfg::from_args(args)?, store);
            let r = runner::run(
                &mut e,
                &prep.tok,
                ids,
                gcfg,
                cons,
                gov.as_mut(),
                cancel,
                &mut sink,
            );
            print_notes(&mut e);
            r
        }
        Mode::Stream => match open_stream(prep, cfg, args)? {
            Some(mut e) => {
                let r = runner::run(
                    &mut e,
                    &prep.tok,
                    ids,
                    gcfg,
                    cons,
                    gov.as_mut(),
                    cancel,
                    &mut sink,
                );
                print_notes(&mut e);
                r
            }
            None => {
                let mut model = build_model_with_threads(prep, cfg, args)?;
                let st = model.new_state_with(cfg.kv);
                let mut e = LocalEngine::new(&mut model, st, None, None);
                runner::run(
                    &mut e,
                    &prep.tok,
                    ids,
                    gcfg,
                    cons,
                    gov.as_mut(),
                    cancel,
                    &mut sink,
                )
            }
        },
        Mode::Mesh => {
            let model = Arc::new(build_model_with_threads(prep, cfg, args)?);
            let mut e = mesh_engine(args, model)?;
            runner::run(
                &mut e,
                &prep.tok,
                ids,
                gcfg,
                cons,
                gov.as_mut(),
                cancel,
                &mut sink,
            )
        }
    }
}

/// Mesh options from the command line: `--wire-f16` (half-size hidden states), `--local-speed`.
fn mesh_config(args: &Args, key: Vec<u8>) -> Result<MeshConfig, String> {
    let mut mc = MeshConfig::with_key(key);
    if args.flag("wire-f16") {
        mc.wire = Wire::F16;
    }
    if let Some(s) = args.num::<f64>("local-speed")? {
        mc.local_speed = Some(s);
    }
    Ok(mc)
}

fn mesh_engine(args: &Args, model: Arc<Model>) -> Result<MeshEngine, String> {
    let key = mesh_key(args)?;
    let mut workers = match args.get("workers") {
        Some(w) => resolve_addrs(w)?,
        None => vec![],
    };
    if args.flag("discover") {
        let wait = Duration::from_millis(args.num("discover-wait-ms")?.unwrap_or(2000));
        let target = args
            .get("discover-target")
            .unwrap_or("255.255.255.255:47813");
        let t = resolve_addrs(target)?;
        let found = mesh::discover(&t, wait, Some(mesh::model_fingerprint(&model)));
        eprintln!(
            "discovery: {} worker(s) with this model answered",
            found.len()
        );
        for f in found {
            if !workers.contains(&f.addr) {
                workers.push(f.addr);
            }
        }
    }
    if workers.is_empty() {
        return Err("no workers: give --workers host:port,host:port or --discover".into());
    }
    let mc = mesh_config(args, key)?;
    let ctx = model.spec.context as usize;
    let m = Mesh::connect(model, &workers, mc).map_err(|e| e.to_string())?;
    for n in m.layout() {
        eprintln!(
            "mesh: {} runs layers {}..{} ({} threads)",
            n.name, n.layers.start, n.layers.end, n.threads
        );
    }
    eprintln!("mesh: traffic is authenticated but not encrypted; use a trusted LAN or a VPN");
    Ok(MeshEngine::new(m, ctx))
}

/// A chat's engine, kept between turns.
pub enum ChatEngine {
    Local {
        model: Box<Model>,
        st: Option<companion_engine::model::State>,
    },
}

pub fn finish_line(o: &GenOut) -> String {
    format!(
        "[{} prompt tokens @ {:.1} t/s · {} generated @ {:.1} t/s · {}]",
        o.prompt_tokens,
        o.prefill_tps(),
        o.tokens.len(),
        o.decode_tps(),
        match o.finish {
            "length" => "length",
            "stop" => "stop",
            "context" => "context full",
            "cancelled" => "cancelled",
            o => o,
        }
    )
}

pub fn generate(args: &Args) -> Result<(), String> {
    apply_global(args);
    if let Some(p) = args.pos.first()
        && crate::famrun::wants(std::path::Path::new(p))
    {
        return crate::famrun::generate(args);
    }
    let (prep, cfg) = open_prepared(args)?;
    let raw_prompt = match (args.get("prompt"), args.get("prompt-file")) {
        (Some(p), _) => p.to_string(),
        (None, Some(f)) => std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?,
        _ => args
            .pos
            .get(1)
            .cloned()
            .ok_or("generate: give --prompt TEXT (or --prompt-file F)")?,
    };
    let sources = read_sources(args)?;
    generate_run(args, &prep, &cfg, raw_prompt, sources)
}

/// Generation for one prompt and its (optional) quote-locked sources; `generate` and `rag ask` share it.
pub(crate) fn generate_run(
    args: &Args,
    prep: &Prepared,
    cfg: &LoadCfg,
    raw_prompt: String,
    sources: Vec<String>,
) -> Result<(), String> {
    let out_spec = OutSpec::from_args(args)?;
    if !sources.is_empty() && out_spec.is_some() {
        return Err(
            "--sources (quote lock) cannot be combined with --json/--json-schema/--regex/--grammar"
                .into(),
        );
    }
    let mut msgs = vec![];
    if !sources.is_empty() {
        msgs.push(Message::new("system", &quotes::instructions(&sources)));
    }
    msgs.push(Message::new("user", &raw_prompt));
    let ids = prompt_ids(prep, &msgs, args.flag("raw").then_some(raw_prompt.as_str()))?;
    let sampler = runner::sampler_params(args, 0.7)?;
    let gcfg = GenCfg {
        sampler: sampler.clone(),
        max_tokens: args.num("max-tokens")?.unwrap_or(512),
        stop_ids: prep.stop_ids.clone(),
    };
    let cancel = AtomicBool::new(false);

    if let Some(path) = args.get("receipt") {
        if args.get("watermark").is_some() {
            return Err("--watermark cannot be combined with --receipt (the receipt would not describe the biased sampling)".into());
        }
        if out_spec.is_some()
            || !sources.is_empty()
            || args.get("kv-policy").is_some()
            || args.get("session").is_some()
            || !matches!(mode_of(args)?, Mode::Local)
        {
            return Err("--receipt records the plain generation loop: it cannot be combined with structured output, --sources, --kv-policy, --session or --stream-layers/--workers".into());
        }
        refuse_ignored(
            args,
            "--receipt",
            &RECEIPT_IGNORES,
            &["lookahead", "self-spec", "governor", "no-lock"],
        )?;
        return generate_receipt(args, prep, cfg, &ids, &sampler, gcfg.max_tokens, path);
    }

    if let Some(f) = crate::fast::from_args(args)? {
        crate::fast::check_compatible(args, out_spec.is_some() || !sources.is_empty())?;
        if !matches!(mode_of(args)?, Mode::Local) {
            return Err("speculative decoding runs the model in this process: it cannot be combined with --stream-layers or --workers".into());
        }
        let (o, st) = crate::fast::run(
            args,
            prep,
            cfg,
            &f,
            &ids,
            &sampler,
            gcfg.max_tokens,
            &gcfg.stop_ids,
        )?;
        outln!("{}", o.text);
        if !args.flag("quiet") {
            eprintln!("{}\n{}", st.line(), finish_line(&o));
        }
        return Ok(());
    }

    let (mut cons, index) = build_constraint(args, prep, out_spec.as_ref(), &sources)?;
    if let Some(key) = args.get("watermark") {
        let wm = crate::wm::build(args, key, prep.stop_ids.iter().map(|&i| i as u32).collect())?;
        eprintln!(
            "watermark on (gamma {}, delta {}, context {}); check text with `corbul watermark detect`",
            wm.gamma(),
            wm.delta(),
            wm.context()
        );
        cons = Some(crate::wm::wrap(cons, wm));
    }
    let o = generate_with(args, prep, cfg, &ids, &gcfg, cons, &cancel, true, None)?;
    outln!();
    if let Some(idx) = index {
        let rep = idx.verify(&o.text);
        eprintln!(
            "quotes: {} verbatim, {} wrong source, {} truncated, {} altered, {} merged, {} unsupported",
            rep.verbatim, rep.wrong_source, rep.truncated, rep.altered, rep.merged, rep.unsupported
        );
        if args.flag("json") || args.get("citations").is_some() {
            let j = serde_json::to_string_pretty(&quotes::report_json(&rep)).unwrap_or_default();
            match args.get("citations") {
                Some(f) => std::fs::write(f, j).map_err(|e| format!("--citations {f}: {e}"))?,
                None => eprintln!("{j}"),
            }
        }
    }
    if !args.flag("quiet") {
        eprintln!("{}", finish_line(&o));
    }
    Ok(())
}

/// The steering constraint (if any) and the quote index to verify against afterwards.
pub(crate) type Steering = (
    Option<Box<dyn crate::genloop::Constraint>>,
    Option<std::sync::Arc<companion_engine::quotelock::QuoteIndex>>,
);

/// The one constraint that steers a plain generation (`--lang`, structured output or the quote lock)
/// and the quote index to verify against afterwards. Exactly one steering constraint may be active.
pub(crate) fn build_constraint(
    args: &Args,
    prep: &Prepared,
    out_spec: Option<&OutSpec>,
    sources: &[String],
) -> Result<Steering, String> {
    let mut cons = None;
    let mut index = None;
    let lang_lock = lang_constraint(args, prep)?;
    if lang_lock.is_some() && (out_spec.is_some() || !sources.is_empty()) {
        return Err("--lang cannot be combined with structured output or --sources (only one constraint can steer the reply)".into());
    }
    if let Some(l) = lang_lock {
        cons = Some(l);
    }
    if let Some(s) = out_spec {
        cons = Some(constraint::build_standalone(&prep.tok, &prep.stop_ids, s)?);
    }
    if !sources.is_empty() {
        let idx = quotes::index(sources)?;
        if !args.flag("no-lock") {
            cons = Some(quotes::constraint(idx.clone(), quotes::vocab(&prep.tok)));
        }
        index = Some(idx);
    }
    Ok((cons, index))
}

pub(crate) fn generate_receipt(
    args: &Args,
    prep: &Prepared,
    cfg: &LoadCfg,
    ids: &[usize],
    s: &companion_engine::sampling::SamplerParams,
    max_tokens: usize,
    path: &str,
) -> Result<(), String> {
    let model = build_model_with_threads(prep, cfg, args)?;
    let sp = receipt::SamplerParams {
        temperature: s.temperature,
        top_k: s.top_k as u64,
        top_p: s.top_p,
        repeat_penalty: s.repeat_penalty,
        repeat_last_n: s.repeat_last_n.max(0) as u64,
        seed: s.seed.unwrap_or(1),
    };
    let g = GenerateSpec {
        sampler: sp,
        max_tokens,
        stop_token: prep.stop_ids.first().copied(),
    };
    let o = RecordOptions {
        logit_topk: args.num("receipt-topk")?.unwrap_or(0),
        keep_steps: args.flag("keep-steps"),
        merkle_leaves: args.flag("leaves"),
        tokenizer_fingerprint: Some(receipt::fingerprint_tokenizer(&prep.tok)),
        timestamp: None,
        key: args
            .secret("receipt-key", "receipt-key-file", None)?
            .map(|k| k.into_bytes()),
    };
    if !simd::is_reproducible() {
        eprintln!(
            "note: without --reproducible this receipt can only be replayed on the same machine and thread count"
        );
    }
    let (tokens, rc) =
        receipt::generate_with_receipt(&model, ids, &g, &o).map_err(|e| e.to_string())?;
    let text = prep
        .tok
        .decode(&tokens.iter().map(|&t| t as u32).collect::<Vec<_>>());
    outln!("{text}");
    std::fs::write(path, rc.to_json()).map_err(|e| format!("{path}: {e}"))?;
    let side = format!("{path}.prompt");
    std::fs::write(&side, serde_json::to_string(ids).unwrap_or_default())
        .map_err(|e| format!("{side}: {e}"))?;
    eprintln!("receipt written to {path} (prompt tokens in {side}, needed by `corbul replay`)");
    Ok(())
}

// ------------------------------------------------------------------------------ language tools

struct LangMask {
    mask: companion_engine::sampling::TokenMask,
}
impl crate::genloop::Constraint for LangMask {
    fn apply(&mut self, logits: &mut [f32]) {
        let mut open = false;
        for (i, l) in logits.iter_mut().enumerate() {
            if self.mask.allows(i as u32) {
                open = true;
            } else {
                *l = f32::NEG_INFINITY;
            }
        }
        // never leave the sampler with nothing to choose from
        if !open {
            logits.iter_mut().for_each(|l| {
                if *l == f32::NEG_INFINITY {
                    *l = -1.0e30;
                }
            });
        }
    }
    fn accept(&mut self, _token: usize) {}
}

/// `--lang ro|en|…`: tokens with letters of a script the language does not use are never chosen
/// (Romanian also never gets the old cedilla ş ţ).
pub fn lang_constraint(
    args: &Args,
    prep: &Prepared,
) -> Result<Option<Box<dyn crate::genloop::Constraint>>, String> {
    let Some(code) = args.get("lang") else {
        return Ok(None);
    };
    let l = Lang::from_code(code).ok_or_else(|| format!("--lang: unknown language `{code}` (try ro, en, fr, de, es, it, pt, hu, pl, ru, uk, el, tr, ar, he, zh, ja, ko …)"))?;
    let lock = LangLock::from_tokenizer(&prep.tok, l).deny_ro_cedilla(true);
    eprintln!(
        "language lock: {} of {} tokens are off limits for {}",
        lock.forbidden_count(),
        lock.vocab_size(),
        l.code()
    );
    Ok(Some(Box::new(LangMask { mask: lock.mask() })))
}

/// `corbul lang detect TEXT | --file F` and `corbul lang normalize TEXT | --file F`.
pub fn lang_cmd(args: &Args) -> Result<(), String> {
    let sub = args
        .pos
        .first()
        .map(String::as_str)
        .ok_or("lang: use `lang detect` or `lang normalize`")?;
    let text = match args.get("file") {
        Some(f) => std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?,
        None if args.pos.len() > 1 => args.pos[1..].join(" "),
        None => return Err("lang: give the text or --file F".into()),
    };
    match sub {
        "detect" => {
            let d = lang::detect(&text);
            match d.lang {
                Some(l) => outln!(
                    "{} (confidence {:.2}{})",
                    l.code(),
                    d.confidence,
                    if d.reliable {
                        ""
                    } else {
                        ", not reliable: too little text or too ambiguous"
                    }
                ),
                None => outln!("unknown (no evidence)"),
            }
            for (l, p) in &d.scores {
                outln!("  {:<4} {:.2}", l.code(), p);
            }
            Ok(())
        }
        "normalize" => {
            let n = lang::normalize(&text);
            if args.flag("verbose") {
                eprintln!("{}", n.report());
            }
            out!("{}", n.text);
            if !n.text.ends_with('\n') {
                outln!();
            }
            eprintln!(
                "{} change(s){}",
                n.changes.len(),
                if n.is_unchanged() {
                    ""
                } else {
                    " (--verbose lists them)"
                }
            );
            Ok(())
        }
        o => Err(format!("lang: `{o}` is not detect or normalize")),
    }
}

// ------------------------------------------------------------------------------ chat

pub fn chat(args: &Args) -> Result<(), String> {
    use std::io::BufRead;
    apply_global(args);
    if let Some(p) = args.pos.first()
        && crate::famrun::wants(std::path::Path::new(p))
    {
        return crate::famrun::chat(args);
    }
    if !matches!(mode_of(args)?, Mode::Local) {
        return Err(
            "chat runs in this process; use `generate` with --stream-layers or --workers".into(),
        );
    }
    refuse_ignored(
        args,
        "chat",
        &[
            "receipt",
            "session",
            "kv-policy",
            "sources",
            "lang",
            "watermark",
            "draft",
            "skip-layers",
            "draft-len",
            "ngram-max",
            "ngram-min",
            "prompt",
            "prompt-file",
            "citations",
        ],
        &["lookahead", "self-spec", "no-lock"],
    )?;
    let (prep, cfg) = open_prepared(args)?;
    let model = build_model_with_threads(&prep, &cfg, args)?;
    let st = model.new_state_with(cfg.kv);
    let mut engine = ChatEngine::Local {
        model: Box::new(model),
        st: Some(st),
    };
    let gcfg = GenCfg {
        sampler: runner::sampler_params(args, 0.7)?,
        max_tokens: args.num("max-tokens")?.unwrap_or(512),
        stop_ids: prep.stop_ids.clone(),
    };
    let out_spec = OutSpec::from_args(args)?;
    let mut history: Vec<Message> = vec![];
    if let Some(sys) = args.get("system") {
        history.push(Message::new("system", sys));
    }
    eprintln!("Type a message and press Enter. /reset clears the conversation, /exit quits.");
    let stdin = std::io::stdin();
    let cancel = AtomicBool::new(false);
    loop {
        eprint!("\n> ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        if stdin
            .lock()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?
            == 0
        {
            break;
        }
        let line = line.trim();
        match line {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                history.retain(|m| m.role == "system");
                let ChatEngine::Local { model, st } = &mut engine;
                *st = Some(model.new_state_with(cfg.kv));
                eprintln!("(conversation cleared)");
                continue;
            }
            _ => {}
        }
        history.push(Message::new("user", line));
        let ids = prompt_ids(&prep, &history, None)?;
        let cons = match &out_spec {
            Some(s) => Some(constraint::build_standalone(&prep.tok, &prep.stop_ids, s)?),
            None => None,
        };
        match generate_with(
            args,
            &prep,
            &cfg,
            &ids,
            &gcfg,
            cons,
            &cancel,
            true,
            Some(&mut engine),
        ) {
            Ok(o) => {
                outln!();
                history.push(Message::new("assistant", o.text.trim()));
            }
            Err(e) => {
                eprintln!("\n{e}");
                history.pop();
                if let Some(i) = history.iter().position(|m| m.role != "system") {
                    history.drain(i..(i + 2).min(history.len()));
                }
                let ChatEngine::Local { model, st } = &mut engine;
                *st = Some(model.new_state_with(cfg.kv));
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------ mesh worker

pub fn worker(args: &Args) -> Result<(), String> {
    apply_global(args);
    let (prep, cfg) = open_prepared(args)?;
    let key = mesh_key(args)?;
    let model = Arc::new(build_model_with_threads(&prep, &cfg, args)?);
    let bind = args.get("bind").unwrap_or("0.0.0.0:7471");
    // UDP discovery is opt-in: it answers probes from anyone on the network segment.
    let disc = if args.flag("no-discovery")
        || !(args.flag("discovery") || args.get("discovery-port").is_some())
    {
        None
    } else {
        Some(
            args.num::<u16>("discovery-port")?
                .unwrap_or(mesh::DISCOVERY_PORT),
        )
    };
    let w = mesh::spawn_worker(model, bind, MeshConfig::with_key(key), disc)
        .map_err(|e| e.to_string())?;
    crate::shutdown::install();
    eprintln!(
        "worker for {} listening on {}{}",
        prep.name,
        w.addr(),
        w.discovery_addr()
            .map_or(String::new(), |d| format!(" (discovery on UDP {d})"))
    );
    eprintln!(
        "traffic is authenticated with the shared key but not encrypted; use a trusted LAN or a VPN"
    );
    while !crate::shutdown::requested() {
        std::thread::sleep(Duration::from_millis(100));
    }
    eprintln!("stopping the worker …");
    drop(w);
    Ok(())
}

// ------------------------------------------------------------------------------ receipts

fn read_receipt(path: &str) -> Result<Receipt, String> {
    let len = std::fs::metadata(path)
        .map_err(|e| format!("{path}: {e}"))?
        .len();
    if len > receipt::MAX_JSON_BYTES as u64 {
        return Err(format!("{path}: implausibly large for a receipt"));
    }
    Receipt::from_json(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)
        .map_err(|e| format!("{path}: {e}"))
}

pub fn verify_receipt(args: &Args) -> Result<(), String> {
    let path = args
        .pos
        .first()
        .ok_or("verify-receipt: give the receipt file")?;
    let r = read_receipt(path)?;
    let key = args.secret("key", "key-file", None)?;
    let key = key.as_deref().map(str::as_bytes);
    let v = r.verify(key).map_err(|e| format!("INVALID: {e}"))?;
    outln!(
        "receipt is intact: digest ok{}",
        if v.authenticated {
            ", HMAC verified with the key"
        } else if r.mac.is_some() {
            " (it carries an HMAC; give --key to check it)"
        } else {
            " (no HMAC: anyone could have made it)"
        }
    );
    outln!(
        "engine {} · {} · reproducible {} · prompt {} tokens → output {} tokens",
        r.engine_version,
        r.hardware,
        r.reproducible,
        r.prompt_len,
        r.output_len
    );
    outln!("model fingerprint {}", r.model_fingerprint);
    if let Some(m) = args.get("model") {
        apply_global(args);
        let cfg = load_cfg(args)?;
        let prep = load::prepare(Path::new(m), &cfg)?;
        let model = prep.build_model(&cfg)?;
        receipt::verify_weights(&model, &r)
            .map_err(|e| format!("the weights in {m} do NOT match: {e}"))?;
        outln!("the weights in {m} match the receipt");
    } else {
        outln!("(give --model PATH to check that a model's weights match)");
    }
    Ok(())
}

pub fn replay(args: &Args) -> Result<(), String> {
    let path = args.pos.first().ok_or("replay: give the receipt file")?;
    let model_path = args.get("model").ok_or("replay: give --model PATH")?;
    let r = read_receipt(path)?;
    if r.reproducible {
        simd::set_reproducible(true);
    }
    let cfg = load_cfg(args)?;
    let prep = load::prepare(Path::new(model_path), &cfg)?;
    let ids: Vec<usize> = match args.get("prompt-tokens") {
        Some(f) => {
            serde_json::from_str(&std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?)
                .map_err(|e| format!("{f}: {e}"))?
        }
        None => {
            let side = format!("{path}.prompt");
            match (args.get("prompt"), std::fs::read_to_string(&side)) {
                (Some(p), _) => prompt_ids(
                    &prep,
                    &[Message::new("user", p)],
                    args.flag("raw").then_some(p),
                )?,
                (None, Ok(t)) => serde_json::from_str(&t).map_err(|e| format!("{side}: {e}"))?,
                _ => {
                    return Err(format!(
                        "replay: no prompt: {side} is missing; give --prompt TEXT (and --raw if it was raw) or --prompt-tokens FILE"
                    ));
                }
            }
        }
    };
    let model = prep.build_model(&cfg)?;
    let key = args.secret("key", "key-file", None)?;
    let rep = receipt::replay(&r, &model, &ids, key.as_deref().map(str::as_bytes))
        .map_err(|e| format!("cannot replay: {e}"))?;
    if rep.identical {
        outln!(
            "IDENTICAL: the same {} tokens, the same hash chain",
            rep.tokens.len()
        );
        Ok(())
    } else {
        Err(format!(
            "DIFFERENT output{}",
            rep.first_divergence.map_or(String::new(), |i| format!(
                " (first difference at output token {i})"
            ))
        ))
    }
}

// ------------------------------------------------------------------------------ plan / doctor

/// Spec and tensor bytes of a model folder or file without loading the weights.
pub fn spec_and_bytes(path: &Path) -> Result<(Spec, u64, std::path::PathBuf), String> {
    match load::resolve(path)? {
        load::Source::HfDir(dir) => {
            let cfg =
                std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?;
            let spec = companion_engine::arch::from_hf_config(&cfg).map_err(|e| e.to_string())?;
            let bytes: u64 = std::fs::read_dir(&dir)
                .map_err(|e| e.to_string())?
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum();
            Ok((spec, bytes, dir))
        }
        load::Source::Gguf(f) => {
            let g = {
                let fh = std::fs::File::open(&f).map_err(|e| e.to_string())?;
                let n = fh.metadata().map_err(|e| e.to_string())?.len();
                companion_engine::gguf::Gguf::parse_with_len(std::io::BufReader::new(fh), n)
                    .map_err(|e| e.to_string())?
            };
            let spec = companion_engine::arch::from_gguf(&g).map_err(|e| e.to_string())?;
            Ok((
                spec,
                g.total_tensor_bytes(),
                f.parent().map(Path::to_path_buf).unwrap_or_default(),
            ))
        }
    }
}

pub fn recommend(
    hp: &HardwareProfile,
    cal: Calibration,
    spec: &Spec,
    weights: u64,
    context: u64,
) -> Recommendation {
    let params = ModelInfo::from_spec(spec, 16.0).params_total.max(1);
    let bpw = (weights as f64 * 8.0 / params as f64).clamp(1.0, 32.0);
    Autopilot::with_calibration(hp, cal).recommend(&ModelInfo::from_spec(spec, bpw), context)
}

fn verdict_text(v: Verdict) -> &'static str {
    match v {
        Verdict::Fits => "FITS — everything stays in RAM with room to spare",
        Verdict::Tight => "TIGHT — fits, but with little headroom",
        Verdict::Streams => "STREAMS — does not fit; runs by reading from disk",
        Verdict::TooSlow => "TOO SLOW — would run, but below about 0.2 tokens/s",
        Verdict::WontRun => "WON'T RUN on this machine as it is",
    }
}

fn calibration(hp: &HardwareProfile, args: &Args) -> Calibration {
    if args.flag("calibrate") {
        eprintln!("measuring memory and compute speed (about a second) …");
        Calibration::measure(hp, Duration::from_millis(400))
    } else {
        Calibration::heuristic(hp)
    }
}

pub fn plan(args: &Args) -> Result<(), String> {
    let path = args
        .pos
        .first()
        .ok_or("plan: give a model folder or .gguf file")?;
    let (spec, weights, dir) = spec_and_bytes(Path::new(path))?;
    let hp = HardwareProfile::detect(Some(&dir));
    let cal = calibration(&hp, args);
    let ctx = args
        .num::<u64>("context")?
        .unwrap_or(spec.context.min(8192));
    let mut hp2 = hp.clone();
    if let Some(b) = args.get("ram-budget") {
        let b = parse_size(b)?;
        hp2.available_ram = Some(b);
        hp2.total_ram = Some(hp.total_ram.unwrap_or(b).max(b));
    }
    let r = recommend(&hp2, cal, &spec, weights, ctx);
    outln!(
        "{}  ({} weights on disk, {} experts, context asked {})",
        path,
        gib(weights),
        if spec.experts > 1 {
            spec.experts.to_string()
        } else {
            "no".into()
        },
        ctx
    );
    outln!("verdict       {}", verdict_text(r.verdict));
    outln!(
        "speed         about {:.2} tokens/s decoding, {:.1} tokens/s prompt processing",
        r.decode_tps,
        r.prefill_tps
    );
    outln!(
        "memory        needs {} of a {} budget · largest context that stays resident: {}",
        gib(r.ram_needed),
        gib(r.ram_budget),
        r.max_context
    );
    outln!(
        "advice        {} threads · prefill batch {} · KV {:?}{}",
        r.tuning.threads,
        r.tuning.prefill_batch,
        r.kv_precision,
        if r.disk_fraction > 0.0 {
            format!(
                " · {:.0}% of the weight traffic comes from disk",
                r.disk_fraction * 100.0
            )
        } else {
            String::new()
        }
    );
    for n in &r.notes {
        outln!("note          {n}");
    }
    for s in &r.suggestions {
        outln!("suggestion    {s}");
    }
    outln!(
        "honesty       speeds are {} estimates, not measurements of this model{}",
        if cal.source == companion_engine::governor::CalSource::Measured {
            "calibrated"
        } else {
            "heuristic"
        },
        if args.flag("calibrate") {
            ""
        } else {
            " (add --calibrate to time this machine)"
        }
    );
    if matches!(r.kv_precision, KvPrecision::Q8) {
        outln!("              (a Q8 KV cache is lossy: pass --kv q8 to use it)");
    }
    Ok(())
}

pub fn doctor(args: &Args) -> Result<(), String> {
    let hp = HardwareProfile::detect(args.pos.first().map(Path::new));
    outln!("corbul {}", env!("CARGO_PKG_VERSION"));
    outln!(
        "system        {} {} · {} usable threads · NUMA nodes {}",
        hp.os,
        hp.arch,
        hp.usable_threads,
        hp.numa_nodes
    );
    outln!(
        "cpu           {}",
        if hp.cpu.isa.is_empty() {
            "baseline instructions only".to_string()
        } else {
            hp.cpu.isa.join(" ")
        }
    );
    outln!(
        "kernels       {} (reproducible mode {})",
        simd::active(),
        if simd::is_reproducible() { "on" } else { "off" }
    );
    outln!(
        "memory        {} free of {}",
        hp.available_ram.map_or("?".into(), gib),
        hp.total_ram.map_or("?".into(), gib)
    );
    outln!("disk          {:?}", hp.disk);
    if hp.gpus.is_empty() {
        outln!("gpu           none detected");
    } else {
        for g in &hp.gpus {
            outln!("gpu           {g:?}");
        }
    }
    outln!(
        "              (the engine computes on the CPU; the wgpu GPU backend is a separate crate and is not used by corbul)"
    );
    let r = SystemSensors.read();
    outln!(
        "sensors       temperature {} · battery {} · memory {}",
        r.cpu_temp_c.map_or("n/a".into(), |t| format!("{t:.0} °C")),
        r.battery_pct.map_or("n/a".into(), |p| format!(
            "{p:.0}%{}",
            if r.on_battery == Some(true) {
                " (on battery)"
            } else {
                ""
            }
        )),
        r.mem_available.map_or("n/a".into(), gib)
    );
    outln!(
        "energy meter  {}",
        if Meter::discover_rapl().is_some() {
            "RAPL counter readable"
        } else {
            "no readable counter; energy would be an assumption"
        }
    );
    let cal = calibration(&hp, args);
    outln!(
        "calibration   memory {:.1} GB/s · compute {:.1} GFLOP/s · disk {:.0} MB/s ({})",
        cal.mem_bw / 1e9,
        cal.flops / 1e9,
        cal.disk_bw / 1e6,
        if cal.source == companion_engine::governor::CalSource::Measured {
            "measured"
        } else {
            "heuristic"
        }
    );
    if let Some(m) = args.pos.first() {
        outln!();
        return plan(&Args {
            pos: vec![m.clone()],
            opts: args.opts.clone(),
        });
    }
    outln!("\ntip           `corbul doctor MODEL` also says whether that model will run here");
    Ok(())
}

// ------------------------------------------------------------------------------ capabilities

pub const CAPABILITIES: &[(&str, &str, &str)] = &[
    (
        "dense decoder models (Llama, Qwen, Gemma, Mistral, Phi…)",
        "yes",
        "safetensors and GGUF; quantised weights stay quantised",
    ),
    (
        "mixture-of-experts (Mixtral, Qwen-MoE, GGUF stacked experts)",
        "yes",
        "--ram-budget streams routed experts from disk",
    ),
    (
        "models larger than RAM (dense)",
        "yes",
        "--stream-layers --ram-budget SIZE; decoding is disk-bound; MoE layers are not streamed this way",
    ),
    (
        "state-space and hybrid models (Mamba, Mamba-2, RWKV-6/7, Jamba-style)",
        "yes",
        "generate and chat through the family loader: plain-text continuation (no chat template), temperature/top-k/top-p only, no serve; native RWKV word-list vocabularies need a tokenizer.json; unverified on real checkpoints",
    ),
    (
        "MLA / DeepSeek",
        "no",
        "the attention variant exists in the engine library, the whole model is not assembled; corbul refuses it with a message",
    ),
    (
        "LoRA adapters",
        "yes",
        "--lora PATH[:SCALE], merged while loading; matrices an adapter touches are held as f32",
    ),
    (
        "structured output (JSON, schema, regex, GBNF)",
        "yes",
        "--json --json-schema --regex --grammar; server: response_format, grammar",
    ),
    (
        "quote-locked citations",
        "yes",
        "--sources FILE (repeatable); server: `sources`, POST /v1/quotes/verify; streaming responses carry no citations block",
    ),
    (
        "sampler chain",
        "yes",
        "--min-p --typical-p --mirostat --dry-multiplier --xtc-probability --sampler-json …",
    ),
    (
        "saved sessions",
        "yes",
        "--session DIR reuses the longest saved prompt prefix; not with --kv-policy",
    ),
    (
        "KV-cache eviction",
        "yes",
        "--kv-policy streaming|h2o|snapkv --kv-budget N; lossy by design",
    ),
    (
        "receipts and reproducible mode",
        "yes",
        "--reproducible, generate --receipt F, verify-receipt, replay; plain loop only, no MoE",
    ),
    (
        "distributed inference over a LAN",
        "yes",
        "worker / generate --workers or --discover --key; authenticated, not encrypted; tested on loopback only",
    ),
    (
        "language tools",
        "yes",
        "corbul lang detect|normalize; --lang CODE forbids tokens of foreign scripts (Romanian: no ş ţ); not in the server",
    ),
    (
        "hardware governor",
        "yes",
        "--governor; reads temperature, memory and battery on Linux",
    ),
    (
        "speculative decoding",
        "yes",
        "generate --draft MODEL | --lookahead | --self-spec [--skip-layers auto|LIST] [--draft-len N]; greedy output is the plain greedy output (up to float rounding), sampled output follows the same distribution; text appears at the end; no structured output, receipts, sessions or mesh with it",
    ),
    (
        "search (beam, best-of-N, voting)",
        "yes",
        "corbul search MODEL --method beam|best-of-n|vote; several model runs per answer, so slower",
    ),
    (
        "automatic mixed-precision quantisation",
        "yes",
        "corbul autoquant MODEL --budget … writes a recipe and a report, corbul quantize --recipe writes the GGUF; damage is estimated from the weights, not measured on outputs; HF folders only",
    ),
    (
        "model merging",
        "yes",
        "corbul merge CONFIG.json (linear, slerp, task, TIES, DARE, passthrough/slices); writes weights only",
    ),
    (
        "activation steering and logit lens",
        "yes",
        "corbul steer train|apply|lens; dense models only; tested on synthetic models",
    ),
    (
        "local document library (RAG)",
        "yes",
        "corbul rag add|search|ask; re-indexes on every call, hashing embedder by default; answers are quote-locked; no PDF/Office parsing",
    ),
    (
        "text watermark and detector",
        "yes",
        "generate --watermark KEY, corbul watermark detect; a statistical signal that paraphrasing removes; not with --receipt or speculation",
    ),
    (
        "web interface",
        "yes",
        "corbul serve opens it at / and /ui: chat, model and sampler panel, sources, receipts, hardware plan; needs the same token as the API",
    ),
    (
        "GPU",
        "no",
        "the wgpu crate exists but corbul does not use it",
    ),
    ("vision / audio inputs", "no", "text only"),
    (
        "embeddings, rerank endpoints",
        "no",
        "answered with `not implemented`",
    ),
];

pub fn capabilities(args: &Args) -> Result<(), String> {
    if args.flag("json") {
        let v: Vec<_> = CAPABILITIES
            .iter()
            .map(
                |(n, s, d)| serde_json::json!({"feature": n, "supported": *s == "yes", "notes": d}),
            )
            .collect();
        outln!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        return Ok(());
    }
    for (n, s, d) in CAPABILITIES {
        outln!(
            "{:<4} {n}\n     {d}",
            if *s == "yes" { "yes" } else { "NO" }
        );
    }
    outln!(
        "\nNothing here has been run on a real downloaded model by the authors of this build: the engine was tested on synthetic checkpoints."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;

    fn args(s: &str) -> Args {
        Args::parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>())
    }

    #[test]
    fn lora_option_parsing() {
        assert_eq!(
            parse_lora("a/b").unwrap(),
            LoraSpec {
                path: "a/b".into(),
                scale: 1.0
            }
        );
        assert_eq!(
            parse_lora("a/b:0.5").unwrap(),
            LoraSpec {
                path: "a/b".into(),
                scale: 0.5
            }
        );
        assert_eq!(parse_lora(r"C:\adapters\x").unwrap().scale, 1.0);
        assert_eq!(parse_lora("x:-1").unwrap().scale, -1.0);
    }

    #[test]
    fn load_options() {
        let c = load_cfg(&args(
            "m --ram-budget 2g --lora a:2 --lora b --kv q8 --no-mmap",
        ))
        .unwrap();
        assert_eq!(
            (c.ram_budget, c.lora.len(), c.kv, c.mmap),
            (Some(2 << 30), 2, KvKind::Q8, false)
        );
        assert!(load_cfg(&args("m --ram-budget lots")).is_err());
        assert!(load_cfg(&args("m --kv f64")).is_err());
    }

    #[test]
    fn plan_and_doctor_run_on_a_tiny_model() {
        let d = tmpdir("plan");
        write_model(&d, true);
        let p = d.to_string_lossy().to_string();
        plan(&Args::parse(&[p.clone(), "--context".into(), "64".into()])).unwrap();
        doctor(&Args::parse(std::slice::from_ref(&p))).unwrap();
        let (spec, bytes, _) = spec_and_bytes(&d).unwrap();
        assert!(bytes > 0);
        let hp = HardwareProfile::detect(None);
        let r = recommend(&hp, Calibration::heuristic(&hp), &spec, bytes, 64);
        assert!(matches!(r.verdict, Verdict::Fits | Verdict::Tight));
        assert!(plan(&Args::parse(&["/no/such/model".into()])).is_err());
        capabilities(&args("")).unwrap();
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_huge_model_is_not_promised_to_fit() {
        let hp = HardwareProfile::detect(None);
        let spec = companion_engine::arch::from_hf_config(r#"{"model_type":"llama","hidden_size":8192,"num_hidden_layers":80,"num_attention_heads":64,"num_key_value_heads":8,"intermediate_size":28672,"vocab_size":128256,"max_position_embeddings":8192}"#).unwrap();
        let mut hp = hp;
        hp.available_ram = Some(4 << 30);
        hp.total_ram = Some(8 << 30);
        let r = recommend(&hp, Calibration::heuristic(&hp), &spec, 140 << 30, 4096);
        assert!(
            matches!(
                r.verdict,
                Verdict::Streams | Verdict::TooSlow | Verdict::WontRun
            ),
            "{:?}",
            r.verdict
        );
    }

    #[test]
    fn receipts_round_trip_through_the_commands() {
        let d = tmpdir("rcpt");
        write_model_ctx(&d, true, 96);
        let out = tmpdir("rcpt-out");
        let rc = out.join("r.json").to_string_lossy().to_string();
        let model = d.to_string_lossy().to_string();
        let run = |extra: &[&str]| {
            let mut v: Vec<String> = vec![
                model.clone(),
                "--prompt".into(),
                "hi".into(),
                "--max-tokens".into(),
                "6".into(),
                "--temperature".into(),
                "0".into(),
                "--receipt".into(),
                rc.clone(),
                "--quiet".into(),
            ];
            v.extend(extra.iter().map(|s| s.to_string()));
            generate(&Args::parse(&v))
        };
        // options the recorder would silently drop are refused up front
        for bad in [
            &["--min-p", "0.1"][..],
            &["--lang", "ro"],
            &["--lookahead"],
            &["--self-spec"],
            &["--governor"],
            &["--mirostat", "1"],
            &["--sampler-json", "{}"],
        ] {
            let e = run(bad).unwrap_err();
            assert!(
                e.contains("--receipt") && e.contains("ignored"),
                "{bad:?}: {e}"
            );
        }
        assert!(
            !std::path::Path::new(&rc).exists(),
            "a refused run must not leave a receipt behind"
        );
        run(&["--receipt-key", "secret-key"]).unwrap();
        let rcv = |extra: &[&str]| {
            let mut v = vec![rc.clone()];
            v.extend(extra.iter().map(|s| s.to_string()));
            Args::parse(&v)
        };
        verify_receipt(&rcv(&["--key", "secret-key", "--model", &model])).unwrap();
        assert!(verify_receipt(&rcv(&["--key", "wrong-key"])).is_err());
        replay(&rcv(&["--model", &model, "--key", "secret-key"])).unwrap();
        // tampering is caught
        let text = std::fs::read_to_string(&rc).unwrap().replacen(
            "\"output_len\":6",
            "\"output_len\":5",
            1,
        );
        std::fs::write(&rc, text).unwrap();
        assert!(verify_receipt(&rcv(&["--key", "secret-key"])).is_err());
        std::fs::remove_dir_all(d).unwrap();
        std::fs::remove_dir_all(out).unwrap();
    }

    #[test]
    fn reproducible_flag_switches_the_process_into_reproducible_mode() {
        let before = simd::is_reproducible();
        apply_global(&Args::parse(&["m".into()]));
        assert_eq!(
            simd::is_reproducible(),
            before,
            "without the flag nothing changes"
        );
        apply_global(&Args::parse(&["m".into(), "--reproducible".into()]));
        assert!(simd::is_reproducible());
        simd::set_reproducible(before);
    }

    #[test]
    fn wire_f16_and_local_speed_reach_the_mesh_config() {
        let key = b"0123456789abcdef".to_vec();
        let c = mesh_config(
            &Args::parse(&["--workers".into(), "x:1".into()]),
            key.clone(),
        )
        .unwrap();
        assert_eq!((c.wire, c.local_speed), (Wire::F32, None));
        let c = mesh_config(
            &Args::parse(&["--wire-f16".into(), "--local-speed".into(), "2.5".into()]),
            key.clone(),
        )
        .unwrap();
        assert_eq!((c.wire, c.local_speed), (Wire::F16, Some(2.5)));
        assert!(mesh_config(&Args::parse(&["--local-speed".into(), "fast".into()]), key).is_err());
    }

    #[test]
    fn lang_and_sources_build_the_one_steering_constraint() {
        let d = tmpdir("constraint");
        write_model_ctx(&d, true, 96);
        let m = d.to_string_lossy().to_string();
        let src = d.join("src.txt");
        std::fs::write(
            &src,
            "The quick brown fox jumps over the lazy dog near the river bank.",
        )
        .unwrap();
        let (sp, mp) = (src.to_string_lossy().to_string(), m.clone());
        let prep = |extra: &[&str]| {
            let mut v = vec![mp.clone()];
            v.extend(extra.iter().map(|s| s.to_string()));
            let a = Args::parse(&v);
            (open_prepared(&a).unwrap().0, a)
        };
        let none = |a: &Args, p: &Prepared, s: &[String]| build_constraint(a, p, None, s);

        let (p, a) = prep(&[]);
        let (c, i) = none(&a, &p, &[]).unwrap();
        assert!(c.is_none() && i.is_none());

        // --lang: a constraint that really removes tokens, and an unknown code is an error
        let (p, a) = prep(&["--lang", "ro"]);
        let (c, i) = none(&a, &p, &[]).unwrap();
        assert!(i.is_none());
        let mut logits = vec![0.0f32; p.tok.vocab_size()];
        c.expect("--lang builds a constraint").apply(&mut logits);
        assert!(
            logits.contains(&f32::NEG_INFINITY),
            "the Romanian lock must forbid something"
        );
        let (p, a) = prep(&["--lang", "xx-nonsense"]);
        assert!(
            none(&a, &p, &[])
                .err()
                .expect("unknown language")
                .contains("unknown language")
        );

        // --sources: constraint + index; --no-lock keeps the index (verification) but drops the lock
        let srcs =
            read_sources(&Args::parse(&[m.clone(), "--sources".into(), sp.clone()])).unwrap();
        assert_eq!(srcs.len(), 1);
        assert!(srcs[0].starts_with("The quick brown fox"));
        let (p, a) = prep(&["--sources", &sp]);
        let (c, i) = none(&a, &p, &srcs).unwrap();
        assert!(c.is_some() && i.is_some());
        let (p, a) = prep(&["--sources", &sp, "--no-lock"]);
        let (c, i) = none(&a, &p, &srcs).unwrap();
        assert!(c.is_none() && i.is_some());
        // only one constraint may steer
        let (p, a) = prep(&["--sources", &sp, "--lang", "ro"]);
        assert!(
            none(&a, &p, &srcs)
                .err()
                .expect("conflict")
                .contains("only one constraint")
        );
        assert!(read_sources(&Args::parse(&["--sources".into(), "/no/such/file".into()])).is_err());
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn generate_with_sources_writes_the_citation_report() {
        let d = tmpdir("citations");
        write_model_ctx(&d, true, 1024);
        let (src, cit) = (d.join("src.txt"), d.join("cit.json"));
        std::fs::write(
            &src,
            "The quick brown fox jumps over the lazy dog near the river bank.",
        )
        .unwrap();
        let v: Vec<String> = [
            &d.to_string_lossy()[..],
            "--prompt",
            "hi",
            "--max-tokens",
            "4",
            "--temperature",
            "0",
            "--quiet",
            "--sources",
            &src.to_string_lossy(),
            "--citations",
            &cit.to_string_lossy(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        generate(&Args::parse(&v)).unwrap();
        let j: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cit).unwrap()).unwrap();
        assert!(
            j["counts"]["verbatim"].is_u64()
                && j["quotes"].is_array()
                && j["all_verbatim"].is_boolean(),
            "{j}"
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn receipt_keys_can_come_from_files() {
        let d = tmpdir("rcpt-keyfile");
        write_model_ctx(&d, true, 96);
        let out = tmpdir("rcpt-keyfile-out");
        let (rc, kf) = (
            out.join("r.json").to_string_lossy().to_string(),
            out.join("key").to_string_lossy().to_string(),
        );
        std::fs::write(&kf, "a-long-random-receipt-key\n").unwrap();
        let model = d.to_string_lossy().to_string();
        let v: Vec<String> = [
            &model[..],
            "--prompt",
            "hi",
            "--max-tokens",
            "4",
            "--temperature",
            "0",
            "--quiet",
            "--receipt",
            &rc,
            "--receipt-key-file",
            &kf,
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        generate(&Args::parse(&v)).unwrap();
        verify_receipt(&Args::parse(&[rc.clone(), "--key-file".into(), kf.clone()])).unwrap();
        assert!(
            verify_receipt(&Args::parse(&[
                rc.clone(),
                "--key".into(),
                "wrong-key-value".into()
            ]))
            .is_err()
        );
        assert!(
            verify_receipt(&Args::parse(&[
                rc.clone(),
                "--key".into(),
                "x".into(),
                "--key-file".into(),
                kf
            ]))
            .is_err(),
            "value and file together are ambiguous"
        );
        std::fs::remove_dir_all(d).unwrap();
        std::fs::remove_dir_all(out).unwrap();
    }

    #[test]
    fn chat_refuses_options_it_cannot_honour() {
        let d = tmpdir("chat-refuse");
        write_model_ctx(&d, true, 96);
        let m = d.to_string_lossy().to_string();
        for extra in [
            &["--receipt", "r.json"][..],
            &["--session", "s"],
            &["--kv-policy", "h2o"],
            &["--sources", "f"],
            &["--lang", "ro"],
            &["--watermark", "k"],
            &["--lookahead"],
            &["--draft", "m"],
            &["--prompt", "hi"],
        ] {
            let mut v = vec![m.clone()];
            v.extend(extra.iter().map(|s| s.to_string()));
            let e = chat(&Args::parse(&v)).unwrap_err();
            assert!(e.contains("cannot be combined with chat"), "{extra:?}: {e}");
        }
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn mesh_keys_warn_when_short_and_read_from_files() {
        assert!(
            weak_secret_warning("mesh key", "short-key-1")
                .unwrap()
                .contains("16")
        );
        assert!(weak_secret_warning("token", "0123456789abcdef").is_none());
        assert!(
            mesh_key(&Args::parse(&["--key".into(), "abc".into()])).is_err(),
            "under 8 characters stays an error"
        );
        let p = std::env::temp_dir().join(format!("corbul-mesh-key-{}", std::process::id()));
        std::fs::write(&p, "file-based-mesh-key-0123\n").unwrap();
        assert_eq!(
            mesh_key(&Args::parse(&[
                "--key-file".into(),
                p.to_string_lossy().to_string()
            ]))
            .unwrap(),
            b"file-based-mesh-key-0123"
        );
        let _ = std::fs::remove_file(p);
    }

    fn greedy_cfg(n: usize) -> GenCfg {
        GenCfg {
            sampler: companion_engine::sampling::SamplerParams {
                temperature: 0.0,
                seed: Some(1),
                ..Default::default()
            },
            max_tokens: n,
            stop_ids: vec![],
        }
    }

    #[test]
    fn layer_streaming_matches_the_normal_engine() {
        let d = tmpdir("stream");
        write_model_ctx(&d, true, 96);
        let a = Args::parse(&[
            d.to_string_lossy().to_string(),
            "--stream-layers".into(),
            "--stream-window".into(),
            "1".into(),
            "--ram-budget".into(),
            "1g".into(),
        ]);
        let (prep, cfg) = open_prepared(&a).unwrap();
        let ids = prompt_ids(&prep, &[Message::new("user", "hi")], None).unwrap();
        let cancel = AtomicBool::new(false);
        let mut se = open_stream(&prep, &cfg, &a)
            .unwrap()
            .expect("streaming was forced by --stream-window");
        let streamed = runner::run(
            &mut se,
            &prep.tok,
            &ids,
            &greedy_cfg(8),
            None,
            None,
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        let mut model = prep.build_model(&cfg).unwrap();
        let st = model.new_state();
        let mut le = LocalEngine::new(&mut model, st, None, None);
        let local = runner::run(
            &mut le,
            &prep.tok,
            &ids,
            &greedy_cfg(8),
            None,
            None,
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(streamed.tokens, local.tokens);
        assert!(se.take_notes()[0].contains("peak"));
        // without a forced window a model that fits is loaded normally
        let b = Args::parse(&[
            d.to_string_lossy().to_string(),
            "--stream-layers".into(),
            "--ram-budget".into(),
            "1g".into(),
        ]);
        assert!(open_stream(&prep, &cfg, &b).unwrap().is_none());
        // a budget too small for even one layer is an error with numbers in it
        let c = Args::parse(&[
            d.to_string_lossy().to_string(),
            "--stream-layers".into(),
            "--ram-budget".into(),
            "64".into(),
        ]);
        let mut cfg2 = cfg.clone();
        cfg2.ram_budget = Some(64);
        assert!(
            open_stream(&prep, &cfg2, &c)
                .err()
                .unwrap()
                .contains("not even one layer")
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_two_machine_mesh_gives_the_same_tokens_as_one() {
        let d = tmpdir("mesh");
        write_model_ctx(&d, true, 96);
        let base = Args::parse(&[d.to_string_lossy().to_string()]);
        let (prep, cfg) = open_prepared(&base).unwrap();
        let ids = prompt_ids(&prep, &[Message::new("user", "hi")], None).unwrap();
        let cancel = AtomicBool::new(false);
        let worker_model = Arc::new(prep.build_model(&cfg).unwrap());
        let w = mesh::spawn_worker(
            worker_model,
            "127.0.0.1:0",
            MeshConfig::with_key(b"test-secret-key".to_vec()),
            None,
        )
        .unwrap();
        let a = Args::parse(&[
            d.to_string_lossy().to_string(),
            "--workers".into(),
            w.addr().to_string(),
            "--key".into(),
            "test-secret-key".into(),
        ]);
        let mut me = mesh_engine(&a, Arc::new(prep.build_model(&cfg).unwrap())).unwrap();
        let meshed = runner::run(
            &mut me,
            &prep.tok,
            &ids,
            &greedy_cfg(8),
            None,
            None,
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        let mut model = prep.build_model(&cfg).unwrap();
        let st = model.new_state();
        let mut le = LocalEngine::new(&mut model, st, None, None);
        let local = runner::run(
            &mut le,
            &prep.tok,
            &ids,
            &greedy_cfg(8),
            None,
            None,
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(meshed.tokens, local.tokens);
        // a wrong key is refused, a missing key is explained
        let bad = Args::parse(&[
            d.to_string_lossy().to_string(),
            "--workers".into(),
            w.addr().to_string(),
            "--key".into(),
            "another-secret!".into(),
        ]);
        assert!(mesh_engine(&bad, Arc::new(prep.build_model(&cfg).unwrap())).is_err());
        let nokey = Args::parse(&[
            d.to_string_lossy().to_string(),
            "--workers".into(),
            w.addr().to_string(),
        ]);
        assert!(
            mesh_engine(&nokey, Arc::new(prep.build_model(&cfg).unwrap()))
                .err()
                .unwrap()
                .contains("--key")
        );
        assert!(
            mesh_engine(
                &Args::parse(&[
                    d.to_string_lossy().to_string(),
                    "--key".into(),
                    "test-secret-key".into(),
                    "--workers".into(),
                    "".into()
                ]),
                Arc::new(prep.build_model(&cfg).unwrap())
            )
            .is_err()
        );
        drop(w);
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn constrained_and_quote_locked_generation_obeys_the_constraint() {
        let d = tmpdir("cons");
        write_model_ctx(&d, true, 96);
        let (prep, cfg) = open_prepared(&Args::parse(&[d.to_string_lossy().to_string()])).unwrap();
        let ids = prompt_ids(&prep, &[Message::new("user", "hi")], None).unwrap();
        let cancel = AtomicBool::new(false);
        // a regex: the reply can only be digits
        let cons = constraint::build_standalone(
            &prep.tok,
            &prep.stop_ids,
            &OutSpec::Regex("[0-9]{3,6}".into()),
        )
        .unwrap();
        let mut model = prep.build_model(&cfg).unwrap();
        let st = model.new_state();
        let mut e = LocalEngine::new(&mut model, st, None, None);
        let mut g = greedy_cfg(10);
        g.stop_ids = prep.stop_ids.clone();
        let o = runner::run(
            &mut e,
            &prep.tok,
            &ids,
            &g,
            Some(cons),
            None,
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        assert!(
            !o.text.is_empty() && o.text.chars().all(|c| c.is_ascii_digit()),
            "{:?}",
            o.text
        );
        assert!(o.text.len() <= 6);
        // options: at most one structured-output flag; a schema may be given inline
        assert!(
            OutSpec::from_args(&Args::parse(&[
                "--json".into(),
                "--regex".into(),
                "a".into()
            ]))
            .is_err()
        );
        assert!(matches!(
            OutSpec::from_args(&Args::parse(&[
                "--json-schema".into(),
                r#"{"type":"object"}"#.into()
            ]))
            .unwrap(),
            Some(OutSpec::Schema(_))
        ));
        assert!(OutSpec::from_args(&Args::parse(&["--json-schema".into(), "[1]".into()])).is_err());
        assert!(
            OutSpec::from_args(&Args::parse(&["--grammar".into(), "/no/such/file".into()]))
                .is_err()
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn language_tools() {
        lang_cmd(&Args::parse(&[
            "detect".into(),
            "Aceasta este o propoziție scrisă în limba română, cu diacritice.".into(),
        ]))
        .unwrap();
        lang_cmd(&Args::parse(&[
            "normalize".into(),
            "Ştiinţa şi şcoala".into(),
        ]))
        .unwrap();
        assert!(lang_cmd(&Args::parse(&["detect".into()])).is_err());
        assert!(lang_cmd(&Args::parse(&["fly".into(), "x".into()])).is_err());
        let n = lang::normalize("Ştiinţa şi şcoala");
        assert_eq!(
            n.text, "Știința și școala",
            "cedilla forms become comma-below"
        );
        let d = lang::detect("Aceasta este o propoziție scrisă în limba română, cu diacritice.");
        assert_eq!(d.lang, Some(Lang::Ro));
        // the mask forbids a Cyrillic token for a Romanian reply and leaves Latin ones alone
        let lock = LangLock::from_token_bytes(
            &[
                "a".as_bytes().to_vec(),
                "я".as_bytes().to_vec(),
                " ".as_bytes().to_vec(),
                "ş".as_bytes().to_vec(),
            ],
            Lang::Ro,
        )
        .deny_ro_cedilla(true);
        let mut c = LangMask { mask: lock.mask() };
        let mut l = vec![1.0f32; 4];
        crate::genloop::Constraint::apply(&mut c, &mut l);
        assert_eq!(l[0], 1.0);
        assert_eq!(l[1], f32::NEG_INFINITY);
        assert_eq!(l[3], f32::NEG_INFINITY);
    }
}
