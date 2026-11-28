//! `corbul` — run, inspect, download and serve language models with the clean-room engine.
use companion_engine::arch::{self, Spec, Support};
use companion_engine::gguf::Gguf;
use companion_engine::hub::{self, Downloader, Hub, RepoSpec, Select};
use companion_engine::hw::{HardwareProfile, Tuning};
use corbul::args::{Args, parse_size};
use corbul::auth::KeyStore;
use corbul::cmds;
use corbul::load::{self, LoadCfg, Source, resolve};
use corbul::outln;
use corbul::registry::{self, RegCfg, Registry};
use corbul::serve::{self, App, LogSink, ServeConfig};
use corbul::shutdown;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

const HELP: &str = "corbul — local language-model runner (clean-room engine)

USAGE
  corbul inspect  <folder|file.gguf|owner/repo[@rev]>
  corbul download <owner/repo[@rev]> [--dest DIR] [--quant Q4_K_M] [--format gguf|safetensors]
  corbul generate <model> [--prompt TEXT | --prompt-file F] [--raw] [options]
  corbul chat     <model> [--system TEXT] [options]
  corbul serve    [<model>] [--models-dir DIR] [--host 127.0.0.1] [--port 8080] [--token T] [options]
  corbul plan     <model> [--context N] [--ram-budget 8g] [--calibrate]   will it run here, and how fast (estimate)
  corbul doctor   [<model>] [--calibrate]                                 this machine: kernels, memory, sensors
  corbul capabilities [--json]                                            what works, what does not
  corbul worker   <model> --key SECRET [--bind 0.0.0.0:7471]              serve layers to a mesh coordinator
  corbul run      <model> --workers h:p,h:p | --discover --key SECRET     generate across machines (= generate)
  corbul verify-receipt <receipt.json> [--key K|--key-file F] [--model M]            check a generation receipt
  corbul replay   <receipt.json> --model M [--key K]                      re-run it and compare
  corbul lang     detect|normalize TEXT | --file F                        language id; Romanian diacritics, mojibake, quotes
  corbul search   <model> --prompt T --method beam|best-of-n|vote         spend more computation for a better answer
  corbul autoquant <folder> --budget 4.5bpw|6g|Q4_K [--recipe F --report F] choose a format per tensor (weight-error estimate)
  corbul quantize <in> <out.gguf> --type Q4_K_M | --recipe F [--imatrix F] write a quantised GGUF
  corbul merge    <config.json> [--out F.safetensors] [--dry-run]         merge models (linear, slerp, task, TIES, DARE, slices)
  corbul steer    train|apply|lens <model> …                              control vectors and a logit lens
  corbul packs    MODEL --session DIR --pack FILE… [--question Q] [--check]   reuse the work done on texts that come back in a new order
  corbul kvfit    MODEL --text FILE --question Q [--budget 0.25]          measure what shrinking the cache costs, with and without refitting values
  corbul map-fit  SRC DST --text FILE --out MAP [--rows 1024]            learn a map from the cache rows of one model to another
  corbul map-check SRC DST --map MAP --text FILE --question Q             measure what the mapped cache does to the answer
  corbul rag      add LIB FILE… | search LIB --query Q | ask LIB MODEL --question Q   local document library, quote-locked answers
  corbul watermark detect <model> --key K --text T | --file F             test a text for the mark made by generate --watermark
  corbul embed    <bert-folder> TEXT… [--file F --pool cls|mean --no-normalize --json]   sentence vectors from a BERT encoder
  corbul rerank   <bert-folder> --query Q DOC… [--file F --top N --json]  relevance scores from a BERT cross-encoder

OPTIONS (generate / chat / serve)
  --max-tokens N     --temperature T   --top-p P   --top-k K   --seed S
  --kv f32|f16|q8    --no-mmap         --model-name NAME
  --threads N        limit the threads used for the heavy products (default: all)
  --ctx N            cap the context window at N tokens
  --quiet            (generate: no statistics on stderr)
  --ram-budget SIZE  RAM the model may use (8g, 512m); mixture-of-experts models stream the experts that do not fit
  --lora PATH[:S]    merge a LoRA adapter (PEFT folder or .gguf), scale S (repeatable)

OPTIONS (generate / chat)
  --min-p P --typical-p P --repeat-penalty R --presence-penalty P --frequency-penalty P
  --mirostat 1|2 --mirostat-tau T --mirostat-eta E --dry-multiplier M --xtc-probability P
  --sampler-json '{\"top_k\": 20}'   any sampler field as JSON, applied before the options above
  --json | --json-schema FILE|JSON | --regex PATTERN | --grammar FILE.gbnf   constrain the output
  --sources FILE     (repeatable) quote-locked answer: text between « » is copied verbatim from a source,
                     then checked and reported (--no-lock: only check; --citations FILE: save the report)
  --session DIR      keep the prompt cache on disk and reuse the longest saved prefix (--session-cap 2g, --compress)
  --kv-policy streaming|h2o|snapkv --kv-budget N    evict the KV cache beyond N tokens (lossy; --kv-sinks 4)
  --stream-layers    models larger than RAM: read layers from disk as needed (--ram-budget, --disk-mbps, --stream-window)
  --workers h:p,...  --discover [--key SECRET] [--wire-f16]   spread layers over machines running `corbul worker`
  --lang CODE        keep the reply in one language (hard mask on foreign scripts; Romanian: no ş ţ)
  --governor         slow down or drop threads when the machine runs hot or low on memory (--verbose shows why)
  --reproducible     bit-identical results on every machine (slower)
  --receipt FILE     sign what was generated (--receipt-key K | --receipt-key-file F, --receipt-topk N, --keep-steps, --leaves)
  --draft MODEL | --lookahead | --self-spec [--skip-layers auto|4,6,8-11] [--draft-len N]   speculative decoding (generate only):
                     same greedy answer, same sampling distribution, fewer full passes; text appears at the end;
                     only --temperature/--top-k/--top-p; not with structured output, receipts, sessions or the mesh
  --watermark KEY    mark the text statistically (--wm-gamma G --wm-delta D --wm-context N); check with `watermark detect`

State-space and hybrid models (Mamba, RWKV, Jamba-style) run through `generate` and `chat` as plain-text continuation
(no chat template, no structured output, no serve). See `corbul capabilities`.

SERVE
  Open http://HOST:PORT/ in a browser for the web interface (also /ui): chat, models and sampling, sources, receipts, hardware.
  The page is public; every call it makes needs the token. GET /v1/capabilities lists what works.
  --models-dir DIR        serve every model folder / .gguf / *.Modelfile in DIR (loaded on first request)
  --config-dir DIR        where `POST /api/create` keeps Modelfiles (default ~/.config/corbul)
  --alias NAME=MODEL      another name for a model (repeatable)
  --max-loaded N          keep at most N models in memory (least recently used go first)
  --sleep-idle-seconds S  unload a model after S seconds without requests
  --token T               one shared bearer token (or CORBUL_TOKEN); --token-file FILE reads it from a file (keeps it out of `ps`)
  --encoder FOLDER        also serve a BERT-family encoder: /v1/embeddings, /api/embed, and /v1/rerank for a cross-encoder
  --allow-admin           let API-key holders use /api/create, /api/delete, /models/load, /models/unload (default: master token only)
  --max-kv SIZE           refuse requests whose KV cache would exceed SIZE (default 8g)   --max-heavy N  concurrent tokenize/template/verify requests (4)
  --keys-file FILE        API keys with per-key model lists and rate limits (JSON)
  --cors ORIGINS          allow browsers from these origins (comma separated, or `*`)
  --log-prompts           include prompt text in the request log (off by default)
  --no-log                no request log
  --drain-seconds S       how long Ctrl-C waits for running requests (default 10)
  Requests may carry `sources` for quote-locked answers; POST /v1/quotes/verify checks any text against sources.

Hugging Face access: set HF_TOKEN for gated repos, HF_ENDPOINT for a mirror.
A model is a folder with config.json + *.safetensors + tokenizer.json, or a .gguf file.";

fn gib(b: u64) -> String {
    format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let code = match run(&argv) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("corbul: {e}");
            1
        }
    };
    std::process::exit(code);
}

fn run(argv: &[String]) -> Result<(), String> {
    let Some(cmd) = argv.first() else {
        outln!("{HELP}");
        return Ok(());
    };
    let args = Args::parse(&argv[1..]);
    if args.flag("help") {
        outln!("{HELP}");
        return Ok(());
    }
    match cmd.as_str() {
        "help" | "-h" | "--help" => {
            outln!("{HELP}");
            Ok(())
        }
        "version" | "-V" | "--version" => {
            outln!("corbul {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "inspect" => inspect(&args),
        "download" => download(&args),
        "generate" | "run" => cmds::generate(&args),
        "chat" => cmds::chat(&args),
        "serve" => serve_cmd(&args),
        "plan" => cmds::plan(&args),
        "doctor" => cmds::doctor(&args),
        "capabilities" | "features" => cmds::capabilities(&args),
        "worker" => cmds::worker(&args),
        "verify-receipt" => cmds::verify_receipt(&args),
        "replay" => cmds::replay(&args),
        "lang" => cmds::lang_cmd(&args),
        "autoquant" => corbul::quant_cmd::autoquant(&args),
        "quantize" => corbul::quant_cmd::quantize_cmd(&args),
        "steer" => corbul::steer_cmd::cmd(&args),
        "merge" => corbul::merge_cmd::cmd(&args),
        "packs" => corbul::packs_cmd::cmd(&args),
        "kvfit" => corbul::kvfit_cmd::cmd(&args),
        "map-fit" => corbul::kvmap_cmd::fit_cmd(&args),
        "map-check" => corbul::kvmap_cmd::check_cmd(&args),
        "rag" => corbul::rag_cmd::cmd(&args),
        "search" => corbul::search_cmd::cmd(&args),
        "watermark" => corbul::wm::cmd(&args),
        "embed" => corbul::enc_cmd::embed(&args),
        "rerank" => corbul::enc_cmd::rerank(&args),
        other => Err(format!("unknown command `{other}` (try `corbul help`)")),
    }
}

// ------------------------------------------------------------------------------ inspect

fn describe(spec: &Spec, weights: Option<u64>, budget_from: Option<&Path>) {
    outln!("family        {}", spec.family);
    match &spec.support {
        Support::Standard => outln!("support       full"),
        Support::Moe => outln!("support       mixture-of-experts — runs through the family loader"),
        Support::Recurrent => outln!(
            "support       recurrent (no attention, no KV cache) — runs through the family loader"
        ),
        Support::Hybrid => {
            outln!("support       hybrid attention + state-space — runs through the family loader")
        }
        Support::Native(what) => outln!("support       {what} — runs through the family loader"),
        Support::Partial(why) => outln!("support       partial — {why}"),
        Support::Planned(why) => outln!("support       planned — {why}"),
    }
    outln!(
        "layers        {}  hidden {}  heads {} (kv {})  head_dim {}  ffn {}",
        spec.layers,
        spec.hidden,
        spec.heads,
        spec.kv_heads,
        spec.head_dim,
        spec.ffn
    );
    outln!("vocab         {}  context {}", spec.vocab, spec.context);
    if spec.experts > 1 {
        outln!(
            "experts       {} (used {}, shared {})",
            spec.experts,
            spec.experts_used,
            spec.shared_experts
        );
    }
    let kv = spec.kv_bytes(spec.context.min(8192));
    outln!(
        "kv cache      {} per {} tokens (f16)",
        gib(kv),
        spec.context.min(8192)
    );
    if let Some(w) = weights {
        outln!("weights       {}", gib(w));
        let hp = HardwareProfile::detect(budget_from);
        let t = Tuning::recommend(&hp, w, kv);
        outln!(
            "\nthis machine  {} {} · {} threads · RAM {} free of {}",
            hp.os,
            hp.arch,
            hp.usable_threads,
            hp.available_ram.map_or("?".into(), gib),
            hp.total_ram.map_or("?".into(), gib)
        );
        outln!(
            "isa           {}",
            if hp.cpu.isa.is_empty() {
                "baseline".to_string()
            } else {
                hp.cpu.isa.join(" ")
            }
        );
        outln!(
            "advice        threads {} · prefill batch {} · kv {:?} · mmap {} · RAM budget {}",
            t.threads,
            t.prefill_batch,
            t.kv_precision,
            t.use_mmap,
            gib(t.ram_budget)
        );
        for n in t.explain() {
            outln!("              - {n}");
        }
        if w > t.ram_budget {
            outln!(
                "note          the weights are larger than the RAM budget; with mmap they stream from disk and run slowly"
            );
        }
    }
}

fn inspect(args: &Args) -> Result<(), String> {
    let target = args
        .pos
        .first()
        .ok_or("inspect: give a folder, a .gguf file or owner/repo")?;
    let p = Path::new(target);
    if p.exists() {
        match resolve(p)? {
            Source::HfDir(dir) => {
                let cfg =
                    std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?;
                let spec = arch::from_hf_config(&cfg).map_err(|e| e.to_string())?;
                let bytes: u64 = std::fs::read_dir(&dir)
                    .map_err(|e| e.to_string())?
                    .filter_map(Result::ok)
                    .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum();
                outln!("{}", dir.display());
                describe(&spec, Some(bytes), Some(&dir));
            }
            Source::Gguf(f) => {
                let g = {
                    let fh = std::fs::File::open(&f).map_err(|e| e.to_string())?;
                    let n = fh.metadata().map_err(|e| e.to_string())?.len();
                    Gguf::parse_with_len(std::io::BufReader::new(fh), n)
                        .map_err(|e| e.to_string())?
                };
                let spec = arch::from_gguf(&g).map_err(|e| e.to_string())?;
                outln!("{}", f.display());
                describe(&spec, Some(g.total_tensor_bytes()), f.parent());
            }
        }
        return Ok(());
    }
    // a repository on the hub
    let spec = RepoSpec::parse(target).map_err(|e| e.to_string())?;
    let dl = downloader()?;
    let cancel = AtomicBool::new(false);
    let files = dl.list_files(&spec, &cancel).map_err(|e| e.to_string())?;
    outln!(
        "{}@{}  ({} files, {})",
        spec.repo,
        spec.revision,
        files.len(),
        gib(hub::estimate_bytes(&files))
    );
    let sel = Select {
        format: parse_format(args)?,
        quants: args
            .get("quant")
            .map(|q| vec![q.to_string()])
            .unwrap_or_else(|| Select::default().quants),
    };
    let chosen = hub::select_files(&files, &sel).map_err(|e| e.to_string())?;
    outln!(
        "would download {} files, {}:",
        chosen.len(),
        gib(hub::estimate_bytes(&chosen))
    );
    for f in chosen.iter().take(40) {
        outln!("  {:>10}  {}", gib(f.size), f.path);
    }
    if chosen.len() > 40 {
        outln!("  … and {} more", chosen.len() - 40);
    }
    Ok(())
}

fn parse_format(args: &Args) -> Result<hub::Format, String> {
    match args.get("format") {
        None | Some("auto") => Ok(hub::Format::Auto),
        Some("gguf") => Ok(hub::Format::Gguf),
        Some("safetensors") => Ok(hub::Format::Safetensors),
        Some(o) => Err(format!("--format: `{o}` (use auto, gguf or safetensors)")),
    }
}

fn downloader() -> Result<Downloader, String> {
    let token = std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|t| !t.is_empty());
    Ok(Downloader::new(Hub::from_env())
        .map_err(|e| e.to_string())?
        .with_token(token))
}

// ------------------------------------------------------------------------------ download

fn download(args: &Args) -> Result<(), String> {
    let target = args.pos.first().ok_or("download: give owner/repo[@rev]")?;
    let spec = RepoSpec::parse(target).map_err(|e| e.to_string())?;
    let dl = downloader()?;
    let cancel = AtomicBool::new(false);
    eprintln!("listing {}@{} …", spec.repo, spec.revision);
    let files = dl.list_files(&spec, &cancel).map_err(|e| e.to_string())?;
    let sel = Select {
        format: parse_format(args)?,
        quants: args
            .get("quant")
            .map(|q| vec![q.to_string()])
            .unwrap_or_else(|| Select::default().quants),
    };
    let chosen = hub::select_files(&files, &sel).map_err(|e| e.to_string())?;
    let dest: PathBuf = match args.get("dest") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from("models").join(spec.repo.replace('/', "--")),
    };
    eprintln!(
        "{} files, {} → {}",
        chosen.len(),
        gib(hub::estimate_bytes(&chosen)),
        dest.display()
    );
    let mut last = std::time::Instant::now() - std::time::Duration::from_secs(1);
    let paths = dl
        .download_all(&spec, &chosen, &dest, &cancel, &mut |p| {
            if last.elapsed().as_millis() >= 250
                || (p.file_total > 0 && p.file_done >= p.file_total)
            {
                last = std::time::Instant::now();
                let pct = if p.total > 0 {
                    p.done as f64 * 100.0 / p.total as f64
                } else {
                    0.0
                };
                eprint!(
                    "\r  {pct:5.1}%  {}  {}   ",
                    gib(p.done),
                    p.file
                        .chars()
                        .rev()
                        .take(48)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<String>()
                );
            }
        })
        .map_err(|e| e.to_string())?;
    eprintln!("\ndone: {} files in {}", paths.len(), dest.display());
    outln!("{}", dest.display());
    Ok(())
}

// ------------------------------------------------------------------------------ loading and sampling options

struct Sampling {
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
}
fn sampling(args: &Args, default_temp: f32) -> Result<Sampling, String> {
    Ok(Sampling {
        max_tokens: args.num("max-tokens")?.unwrap_or(512),
        temperature: args.num("temperature")?.unwrap_or(default_temp),
        top_p: args.num("top-p")?.unwrap_or(0.95),
        top_k: args.num("top-k")?.unwrap_or(40),
    })
}

fn open_model(args: &Args) -> Result<load::Loaded, String> {
    let path = args
        .pos
        .first()
        .ok_or("give a model folder or .gguf file")?;
    let cfg = cmds::load_cfg(args)?;
    let t = std::time::Instant::now();
    let mut l = load::load(Path::new(path), &cfg)?;
    if let Some(n) = args.num::<usize>("threads")? {
        l.model.set_threads(n);
    }
    if let Some(c) = args.num::<u64>("ctx")?.filter(|c| *c > 0) {
        l.model.spec.context = l.model.spec.context.min(c as _);
    }
    eprintln!(
        "loaded {} in {:.1}s ({}, context {}, template: {})",
        l.name,
        t.elapsed().as_secs_f64(),
        gib(l.model.weight_bytes() as u64),
        l.model.spec.context,
        l.builtin.name()
    );
    for w in &l.warnings {
        eprintln!("warning: {w}");
    }
    Ok(l)
}

// ------------------------------------------------------------------------------ serve

fn serve_cmd(args: &Args) -> Result<(), String> {
    if let Some(p) = args.pos.first()
        && corbul::famrun::wants(Path::new(p))
    {
        return corbul::famserve::serve(args);
    }
    let load_cfg: LoadCfg = cmds::load_cfg(args)?;
    let reg_cfg = RegCfg {
        load: load_cfg,
        max_loaded: args.num("max-loaded")?.unwrap_or(0),
        idle: args
            .num::<u64>("sleep-idle-seconds")?
            .filter(|s| *s > 0)
            .map(Duration::from_secs),
        threads: args.num("threads")?,
        ctx: args.num::<usize>("ctx")?.filter(|c| *c > 0),
    };
    let models_dir = args.get("models-dir").map(PathBuf::from);
    if args.pos.is_empty() && models_dir.is_none() && args.get("encoder").is_none() {
        return Err("serve: give a model, --models-dir DIR, or --encoder FOLDER".into());
    }
    let mut cfg = ServeConfig::default();
    if let Some(h) = args.get("host") {
        cfg.host = h.to_string();
    }
    if let Some(p) = args.num("port")? {
        cfg.port = p;
    }
    cfg.token = args.secret("token", "token-file", Some("CORBUL_TOKEN"))?;
    if let Some(t) = &cfg.token
        && let Some(w) = cmds::weak_secret_warning("token", t)
    {
        eprintln!("{w}");
    }
    cfg.allow_admin = args.flag("allow-admin");
    if let Some(n) = args.num::<usize>("max-heavy")? {
        cfg.max_heavy = n.max(1);
    }
    if let Some(s) = args.get("max-kv") {
        cfg.max_kv_bytes = parse_size(s)?;
    }
    if let Some(f) = args.get("keys-file") {
        cfg.keys = Some(KeyStore::load(Path::new(f))?);
    }
    if let Some(c) = args.get("cors") {
        cfg.cors = Some(
            c.split(',')
                .map(|o| o.trim().trim_end_matches('/').to_string())
                .filter(|o| !o.is_empty())
                .collect(),
        );
    }
    let s = sampling(args, 0.7)?;
    (cfg.max_tokens, cfg.temperature, cfg.top_p, cfg.top_k) =
        (s.max_tokens, s.temperature, s.top_p, s.top_k);
    cfg.model_id = args.get("model-name").map(str::to_string);
    cfg.log_prompts = args.flag("log-prompts");
    cfg.packs_dir = args.get("packs").map(std::path::PathBuf::from);
    cfg.packs_compress = args.flag("packs-compress");
    cfg.packs_f16 = match args.get("packs-kv") {
        None | Some("f32") => false,
        Some("f16") => true,
        Some(o) => return Err(format!("--packs-kv: {o} is not f32 or f16")),
    };
    if let Some(c) = args.get("packs-cap") {
        cfg.packs_cap = parse_size(c)?;
    }
    if let Some(d) = args.num::<u64>("drain-seconds")? {
        cfg.drain = Duration::from_secs(d);
    }
    if !args.flag("no-log") {
        cfg.log = Some(LogSink(Arc::new(|l: &str| eprintln!("{l}"))));
    }
    let loopback = matches!(cfg.host.as_str(), "127.0.0.1" | "localhost" | "::1");
    if !loopback && cfg.token.is_none() && cfg.keys.is_none() {
        eprintln!(
            "warning: listening on {} without --token or --keys-file; anyone who can reach this port can use the models",
            cfg.host
        );
    }
    let encoder = match args.get("encoder") {
        Some(p) => {
            let mut e = companion_engine::bert::Encoder::open(Path::new(p))
                .map_err(|e| format!("--encoder: {e}"))?;
            corbul::gpu::accelerate_encoder(&mut e);
            eprintln!(
                "encoder {} ({} layers, hidden {}, pooling {}, cross-encoder head {})",
                e.name,
                e.bert.cfg.layers,
                e.bert.cfg.hidden,
                match e.pool {
                    companion_engine::bert::Pool::Cls => "cls",
                    companion_engine::bert::Pool::Mean => "mean",
                },
                if e.bert.has_classifier() { "yes" } else { "no" }
            );
            Some(e)
        }
        None => None,
    };
    let registry = match args.pos.first() {
        Some(_) => {
            let l = open_model(args)?;
            let name = cfg.model_id.clone().unwrap_or_else(|| l.name.clone());
            Registry::single(l, &name, reg_cfg)
        }
        None => Registry::new(reg_cfg),
    };
    if let Some(d) = &models_dir {
        let n = registry
            .scan_dir(d)
            .map_err(|e| format!("--models-dir: {e}"))?;
        eprintln!("found {n} model(s) in {}", d.display());
    }
    let config_dir = args
        .get("config-dir")
        .map(PathBuf::from)
        .or_else(registry::default_config_dir);
    if let Some(c) = &config_dir
        && let Ok(n) = registry.set_config_dir(c)
        && n > 0
    {
        eprintln!("found {n} Modelfile(s) in {}", c.display());
    }
    for a in args.all("alias") {
        let (alias, target) = a
            .split_once('=')
            .ok_or_else(|| format!("--alias: `{a}` is not NAME=MODEL"))?;
        registry.alias(alias.trim(), target.trim())?;
    }
    if registry.is_empty() && encoder.is_none() {
        return Err("no models found".into());
    }
    if let Some(d) = args.get("default-model") {
        registry.set_default(d);
    }
    let mut names = registry.names();
    if let Some(e) = &encoder {
        names.push(e.name.clone());
    }
    let mut app = App::with_registry(registry, cfg);
    if let Some(e) = encoder {
        app = app.with_encoder(e);
    }
    let app = Arc::new(app);
    let h = serve::spawn(app).map_err(|e| format!("cannot listen: {e}"))?;
    shutdown::install();
    eprintln!(
        "serving {} on http://{}",
        if names.len() == 1 {
            names[0].clone()
        } else {
            format!("{} models", names.len())
        },
        h.addr
    );
    eprintln!(
        "  OpenAI: /v1/chat/completions /v1/completions /v1/models · Anthropic: /v1/messages · Ollama: /api/chat /api/generate /api/tags · /metrics /props /slots /tokenize"
    );
    while !shutdown::requested() {
        std::thread::sleep(Duration::from_millis(100));
    }
    eprintln!("shutting down (waiting for running requests) …");
    h.stop();
    Ok(())
}
