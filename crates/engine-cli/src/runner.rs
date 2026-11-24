//! The token loop behind `generate`, `chat` and `run`, over three ways of holding a model: all in this
//! process ([`LocalEngine`], with prompt-prefix reuse, saved sessions and KV-cache eviction), layer by
//! layer from disk ([`StreamEngine`], for models larger than RAM) or spread over machines ([`MeshEngine`]).
use crate::args::{Args, parse_size};
use crate::genloop::Constraint;
use companion_engine::governor::{
    Governor, Policy as GovPolicy, Sensors, SharedSettings, SystemSensors,
};
use companion_engine::hw::{HardwareProfile, Tuning};
use companion_engine::kvpolicy::{self, H2o, LayerBudgets, Policy, SnapKv, StreamingLlm};
use companion_engine::mesh::{Mesh, MeshState};
use companion_engine::model::{Model, State};
use companion_engine::sampling::{SamplerChain, SamplerParams};
use companion_engine::snapshot::{SnapOptions, SnapshotStore};
use companion_engine::stream_layers::StreamedModel;
use companion_engine::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// What an inference engine must offer the token loop.
pub trait Engine {
    fn context(&self) -> usize;
    /// Tokens currently in the cache.
    fn pos(&self) -> usize;
    /// Feed the prompt (reusing whatever the cache already holds) and return the logits after it.
    fn prefill(&mut self, ids: &[usize]) -> Result<Vec<f32>, String>;
    fn step(&mut self, token: usize) -> Result<Vec<f32>, String>;
    /// The governor wants a different thread count (engines that cannot change it ignore this).
    fn set_threads(&mut self, _n: usize) {}
    /// Messages about what the engine did on its own (cache eviction, sessions).
    fn take_notes(&mut self) -> Vec<String> {
        vec![]
    }
}

// ------------------------------------------------------------------------------ KV policy options

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyKind {
    Streaming,
    H2o,
    SnapKv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPolicyCfg {
    pub kind: PolicyKind,
    pub budget: usize,
    pub sinks: usize,
}

impl KvPolicyCfg {
    /// `--kv-policy streaming|h2o|snapkv --kv-budget N [--kv-sinks 4]`.
    pub fn from_args(a: &Args) -> Result<Option<KvPolicyCfg>, String> {
        let kind = match a.get("kv-policy") {
            None | Some("none" | "off") => {
                if a.get("kv-budget").is_some() {
                    return Err("--kv-budget needs --kv-policy streaming|h2o|snapkv".into());
                }
                return Ok(None);
            }
            Some("streaming" | "streamingllm") => PolicyKind::Streaming,
            Some("h2o") => PolicyKind::H2o,
            Some("snapkv") => PolicyKind::SnapKv,
            Some(o) => return Err(format!("--kv-policy: `{o}` (use streaming, h2o or snapkv)")),
        };
        let budget: usize = a
            .num("kv-budget")?
            .ok_or("--kv-policy needs --kv-budget N (tokens kept in the cache)")?;
        if budget < 8 {
            return Err("--kv-budget must be at least 8 tokens".into());
        }
        let sinks = a.num("kv-sinks")?.unwrap_or(4);
        Ok(Some(KvPolicyCfg {
            kind,
            budget,
            sinks,
        }))
    }
    fn policy(&self) -> Box<dyn Policy> {
        match self.kind {
            PolicyKind::Streaming => Box::new(StreamingLlm { sinks: self.sinks }),
            PolicyKind::H2o => Box::new(H2o::default()),
            PolicyKind::SnapKv => Box::new(SnapKv {
                window: 32,
                pool: 7,
            }),
        }
    }
}

/// `--session DIR [--session-cap 2g] [--compress]`.
pub fn open_sessions(a: &Args) -> Result<Option<(SnapshotStore, SnapOptions)>, String> {
    let Some(dir) = a.get("session") else {
        return Ok(None);
    };
    let cap = match a.get("session-cap") {
        Some(c) => parse_size(c)?,
        None => 2 << 30,
    };
    let store = SnapshotStore::open(PathBuf::from(dir), cap)
        .map_err(|e| format!("--session {dir}: {e}"))?;
    Ok(Some((
        store,
        SnapOptions {
            compress: a.flag("compress"),
        },
    )))
}

// ------------------------------------------------------------------------------ local

pub struct LocalEngine<'a> {
    pub model: &'a mut Model,
    pub st: State,
    kv: Option<KvPolicyCfg>,
    policy: Option<Box<dyn Policy>>,
    store: Option<(SnapshotStore, SnapOptions)>,
    notes: Vec<String>,
    saved: bool,
}

impl<'a> LocalEngine<'a> {
    pub fn new(
        model: &'a mut Model,
        st: State,
        kv: Option<KvPolicyCfg>,
        store: Option<(SnapshotStore, SnapOptions)>,
    ) -> LocalEngine<'a> {
        let policy = kv.map(|c| c.policy());
        let mut e = LocalEngine {
            model,
            st,
            kv,
            policy,
            store,
            notes: vec![],
            saved: false,
        };
        if e.policy.as_ref().is_some_and(|p| p.needs_scores()) {
            e.st.track_scores(true);
        }
        e
    }

    fn evict(&mut self, force: bool) -> Result<(), String> {
        let (Some(cfg), Some(p)) = (self.kv, self.policy.as_ref()) else {
            return Ok(());
        };
        if cfg.kind == PolicyKind::SnapKv && !force {
            return Ok(());
        }
        let slack = (cfg.budget / 16).max(8);
        if self.st.pos < cfg.budget + slack && !force {
            return Ok(());
        }
        let c = self
            .st
            .apply_policy(p.as_ref(), cfg.budget, &LayerBudgets::Uniform)
            .map_err(|e| e.to_string())?;
        if c.evicted > 0 {
            self.notes.push(format!("kv cache: {} evicted, {} kept ({} → {} KiB, positions continue past the gap of {})", c.evicted, c.after, c.bytes_before / 1024, c.bytes_after / 1024, c.rope_gap));
        }
        Ok(())
    }
}

impl Engine for LocalEngine<'_> {
    fn context(&self) -> usize {
        self.model.spec.context as usize
    }
    fn pos(&self) -> usize {
        self.st.pos
    }

    fn prefill(&mut self, ids: &[usize]) -> Result<Vec<f32>, String> {
        if self.kv.is_none()
            && self.st.pos == 0
            && let Some((store, _)) = &self.store
        {
            match store.best_prefix(self.model, ids) {
                Ok(Some((st, n))) if n > 0 => {
                    self.notes.push(format!(
                        "session: reusing {n} of {} prompt tokens from a saved cache",
                        ids.len()
                    ));
                    self.st = st;
                }
                Ok(_) => {}
                Err(e) => self.notes.push(format!(
                    "session: could not read a saved cache ({e}); starting fresh"
                )),
            }
        }
        let logits = match self.kv {
            Some(c) if c.kind == PolicyKind::SnapKv && self.st.pos == 0 && ids.len() > c.budget => {
                let sk = SnapKv {
                    window: 32,
                    pool: 7,
                };
                let (lg, comp) = kvpolicy::prefill_snapkv(
                    self.model,
                    &mut self.st,
                    ids,
                    &sk,
                    c.budget,
                    &LayerBudgets::Uniform,
                )
                .map_err(|e| e.to_string())?;
                self.notes.push(format!(
                    "snapkv: prompt of {} tokens compressed to {}",
                    comp.before, comp.after
                ));
                lg
            }
            _ => self
                .model
                .prefill_cached(&mut self.st, ids)
                .map_err(|e| e.to_string())?,
        };
        // keep the prompt's cache for the next run (not after eviction: a compacted cache is not a prefix of any prompt)
        if self.kv.is_none()
            && !self.saved
            && let Some((store, opts)) = &self.store
        {
            self.saved = true;
            match store.put(self.model, &self.st, *opts) {
                Ok(true) => self.notes.push(format!(
                    "session: saved the cache of the {}-token prompt",
                    self.st.pos
                )),
                Ok(false) => {}
                Err(e) => self.notes.push(format!("session: not saved ({e})")),
            }
        }
        self.evict(false)?;
        Ok(logits)
    }

    fn step(&mut self, token: usize) -> Result<Vec<f32>, String> {
        let l = self
            .model
            .forward(&mut self.st, token)
            .map_err(|e| e.to_string())?;
        self.evict(false)?;
        Ok(l)
    }

    fn set_threads(&mut self, n: usize) {
        self.model.set_threads(n);
    }
    fn take_notes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notes)
    }
}

// ------------------------------------------------------------------------------ layer streaming

pub struct StreamEngine {
    pub m: StreamedModel,
    pub st: State,
}
impl StreamEngine {
    pub fn new(m: StreamedModel, kv: companion_engine::model::KvKind) -> StreamEngine {
        let st = m.new_state_with(kv);
        StreamEngine { m, st }
    }
}
impl Engine for StreamEngine {
    fn context(&self) -> usize {
        self.m.spec().context as usize
    }
    fn pos(&self) -> usize {
        self.st.pos
    }
    fn prefill(&mut self, ids: &[usize]) -> Result<Vec<f32>, String> {
        self.m
            .forward_batch(&mut self.st, ids)
            .map_err(|e| e.to_string())
    }
    fn step(&mut self, token: usize) -> Result<Vec<f32>, String> {
        self.m
            .forward(&mut self.st, token)
            .map_err(|e| e.to_string())
    }
    fn take_notes(&mut self) -> Vec<String> {
        let s = self.m.stats();
        vec![format!(
            "layer streaming: {:.0} MB/s from disk, peak {:.0} MiB resident (budget {:.0} MiB)",
            s.disk_mbps(),
            self.m.peak_resident_bytes() as f64 / 1048576.0,
            self.m.pool_budget() as f64 / 1048576.0
        )]
    }
}

// ------------------------------------------------------------------------------ mesh

pub struct MeshEngine {
    pub mesh: Mesh,
    pub st: MeshState,
    ctx: usize,
}
impl MeshEngine {
    pub fn new(mesh: Mesh, ctx: usize) -> MeshEngine {
        let st = mesh.new_state();
        MeshEngine { mesh, st, ctx }
    }
}
impl Engine for MeshEngine {
    fn context(&self) -> usize {
        self.ctx
    }
    fn pos(&self) -> usize {
        self.st.position()
    }
    fn prefill(&mut self, ids: &[usize]) -> Result<Vec<f32>, String> {
        self.mesh
            .prefill_cached(&mut self.st, ids)
            .map_err(|e| e.to_string())
    }
    fn step(&mut self, token: usize) -> Result<Vec<f32>, String> {
        self.mesh
            .try_forward_batch(&mut self.st, &[token])
            .map_err(|e| e.to_string())
    }
}

// ------------------------------------------------------------------------------ governor

/// Reads the machine's sensors about once a second while tokens are generated and turns the governor's
/// advice into a thread count and a pause between tokens.
pub struct GovernorLoop {
    gov: Governor,
    sensors: Box<dyn Sensors>,
    pub shared: SharedSettings,
    start: Instant,
    last: Option<Instant>,
    every: Duration,
    verbose: bool,
}

impl GovernorLoop {
    pub fn new(
        hp: &HardwareProfile,
        weight_bytes: u64,
        kv_bytes: u64,
        verbose: bool,
    ) -> GovernorLoop {
        let t = Tuning::recommend(hp, weight_bytes, kv_bytes);
        let policy = GovPolicy::from_tuning(&t);
        GovernorLoop {
            gov: Governor::new(policy),
            sensors: Box::new(SystemSensors),
            shared: SharedSettings::new(t.threads, t.prefill_batch),
            start: Instant::now(),
            last: None,
            every: Duration::from_secs(1),
            verbose,
        }
    }
    #[cfg(test)]
    pub fn with_sensors(mut self, s: Box<dyn Sensors>, every: Duration) -> GovernorLoop {
        self.sensors = s;
        self.every = every;
        self
    }

    /// Call between tokens. Returns the pause to take now.
    pub fn tick(&mut self, engine: &mut dyn Engine) -> Duration {
        let due = self.last.is_none_or(|l| l.elapsed() >= self.every);
        if due {
            self.last = Some(Instant::now());
            let r = self.sensors.read();
            let a = self.gov.step(&r, self.start.elapsed().as_secs_f64());
            if a.changed {
                if a.threads != self.shared.threads() {
                    engine.set_threads(a.threads);
                }
                if self.verbose {
                    for why in &a.reasons {
                        eprintln!("governor: {why}");
                    }
                }
            }
            self.shared.apply(&a);
        }
        self.shared.throttle()
    }
}

// ------------------------------------------------------------------------------ the loop

pub struct GenCfg {
    pub sampler: SamplerParams,
    pub max_tokens: usize,
    pub stop_ids: Vec<usize>,
}

#[derive(Debug)]
pub struct GenOut {
    pub text: String,
    pub tokens: Vec<usize>,
    pub finish: &'static str,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub prompt_tokens: usize,
}

impl GenOut {
    pub fn decode_tps(&self) -> f64 {
        if self.decode_secs > 0.0 {
            self.tokens.len().saturating_sub(1) as f64 / self.decode_secs
        } else {
            0.0
        }
    }
    pub fn prefill_tps(&self) -> f64 {
        if self.prefill_secs > 0.0 {
            self.prompt_tokens as f64 / self.prefill_secs
        } else {
            0.0
        }
    }
}

/// Sampler settings from the command line on top of the llama.cpp defaults.
pub fn sampler_params(a: &Args, default_temp: f32) -> Result<SamplerParams, String> {
    let mut p = SamplerParams {
        temperature: default_temp,
        ..SamplerParams::default()
    };
    if let Some(j) = a.get("sampler-json") {
        p = p
            .overlay(&serde_json::from_str(j).map_err(|e| format!("--sampler-json: {e}"))?)
            .map_err(|e| format!("--sampler-json: {e}"))?;
    }
    if let Some(v) = a.num("temperature")? {
        p.temperature = v;
    }
    if let Some(v) = a.num("top-p")? {
        p.top_p = v;
    }
    if let Some(v) = a.num("top-k")? {
        p.top_k = v;
    }
    if let Some(v) = a.num("min-p")? {
        p.min_p = v;
    }
    if let Some(v) = a.num("typical-p")? {
        p.typical_p = v;
    }
    if let Some(v) = a.num("repeat-penalty")? {
        p.repeat_penalty = v;
    }
    if let Some(v) = a.num("presence-penalty")? {
        p.presence_penalty = v;
    }
    if let Some(v) = a.num("frequency-penalty")? {
        p.frequency_penalty = v;
    }
    if let Some(v) = a.num("mirostat")? {
        p.mirostat = v;
    }
    if let Some(v) = a.num("mirostat-tau")? {
        p.mirostat_tau = v;
    }
    if let Some(v) = a.num("mirostat-eta")? {
        p.mirostat_eta = v;
    }
    if let Some(v) = a.num("dry-multiplier")? {
        p.dry_multiplier = v;
    }
    if let Some(v) = a.num("xtc-probability")? {
        p.xtc_probability = v;
    }
    if let Some(v) = a.num("seed")? {
        p.seed = Some(v);
    } else if p.seed.is_none() {
        p.seed = Some(crate::serve::now_secs() as u64 ^ 0x5eed);
    }
    if p.mirostat > 2 {
        return Err("--mirostat must be 0, 1 or 2".into());
    }
    Ok(p)
}

/// Generate a reply. `on_piece` gets text as soon as it is complete (never half a UTF-8 character).
#[allow(clippy::too_many_arguments)]
pub fn run(
    engine: &mut dyn Engine,
    tok: &Tokenizer,
    ids: &[usize],
    cfg: &GenCfg,
    mut cons: Option<Box<dyn Constraint>>,
    mut gov: Option<&mut GovernorLoop>,
    cancel: &AtomicBool,
    on_piece: &mut dyn FnMut(&str),
) -> Result<GenOut, String> {
    let ctx = engine.context();
    if ids.is_empty() {
        return Err("the prompt is empty".into());
    }
    if ids.len() >= ctx {
        return Err(format!(
            "the prompt is {} tokens but the context window is {ctx}",
            ids.len()
        ));
    }
    let t0 = Instant::now();
    let mut logits = engine.prefill(ids)?;
    let prefill_secs = t0.elapsed().as_secs_f64();
    let mut chain = SamplerChain::new(cfg.sampler.clone());
    chain.set_eos(cfg.stop_ids.first().map(|i| *i as u32));
    let mut history: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let mut dec = tok.decoder();
    dec.skip_special = true;
    let (mut text, mut tokens) = (String::new(), vec![]);
    let t1 = Instant::now();
    let finish = loop {
        if cancel.load(Ordering::Relaxed) {
            break "cancelled";
        }
        if tokens.len() >= cfg.max_tokens {
            break "length";
        }
        if let Some(c) = cons.as_mut() {
            c.apply(&mut logits);
        }
        let s = chain.sample(&logits, &history);
        if s.empty {
            break "no_token";
        }
        let t = s.token as usize;
        if cfg.stop_ids.contains(&t) {
            break "stop";
        }
        if let Some(c) = cons.as_mut() {
            c.accept(t);
        }
        history.push(s.token);
        tokens.push(t);
        let piece = dec.push(s.token);
        if !piece.is_empty() {
            text.push_str(&piece);
            on_piece(&piece);
        }
        if engine.pos() >= ctx {
            break "context";
        }
        if let Some(g) = gov.as_deref_mut() {
            let pause = g.tick(engine);
            if !pause.is_zero() {
                std::thread::sleep(pause);
            }
        }
        logits = engine.step(t)?;
    };
    let tail = dec.finish();
    if !tail.is_empty() {
        text.push_str(&tail);
        on_piece(&tail);
    }
    let decode_secs = t1.elapsed().as_secs_f64();
    Ok(GenOut {
        text,
        tokens,
        finish,
        prefill_secs,
        decode_secs,
        prompt_tokens: ids.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;
    use crate::load::{LoadCfg, load};
    use companion_engine::governor::Reading;

    fn args(s: &str) -> Args {
        Args::parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>())
    }

    fn tiny() -> (crate::load::Loaded, std::path::PathBuf) {
        let d = tmpdir("runner");
        write_model_ctx(&d, true, 160);
        (load(&d, &LoadCfg::default()).unwrap(), d)
    }

    fn cfg(max: usize) -> GenCfg {
        GenCfg {
            sampler: SamplerParams {
                temperature: 0.0,
                seed: Some(1),
                ..SamplerParams::default()
            },
            max_tokens: max,
            stop_ids: vec![],
        }
    }

    #[test]
    fn option_parsing() {
        assert_eq!(KvPolicyCfg::from_args(&args("")).unwrap(), None);
        assert_eq!(
            KvPolicyCfg::from_args(&args("--kv-policy h2o --kv-budget 64")).unwrap(),
            Some(KvPolicyCfg {
                kind: PolicyKind::H2o,
                budget: 64,
                sinks: 4
            })
        );
        assert!(KvPolicyCfg::from_args(&args("--kv-policy h2o")).is_err());
        assert!(KvPolicyCfg::from_args(&args("--kv-budget 64")).is_err());
        assert!(KvPolicyCfg::from_args(&args("--kv-policy nope --kv-budget 64")).is_err());
        assert!(KvPolicyCfg::from_args(&args("--kv-policy h2o --kv-budget 2")).is_err());
        let p = sampler_params(
            &args("--temperature 0.3 --min-p 0.1 --mirostat 2 --seed 9"),
            0.7,
        )
        .unwrap();
        assert_eq!(
            (p.temperature, p.min_p, p.mirostat, p.seed),
            (0.3, 0.1, 2, Some(9))
        );
        assert!(sampler_params(&args("--mirostat 5"), 0.7).is_err());
        assert!(sampler_params(&args("--sampler-json {nope"), 0.7).is_err());
        assert_eq!(
            sampler_params(&args(r#"--sampler-json {"top_k":7}"#), 0.7)
                .unwrap()
                .top_k,
            7
        );
    }

    #[test]
    fn greedy_generation_is_deterministic_and_matches_the_engines_own_loop() {
        let (mut l, d) = tiny();
        let ids: Vec<usize> = l
            .tok
            .encode("hello", true)
            .into_iter()
            .map(|i| i as usize)
            .collect();
        let cancel = AtomicBool::new(false);
        let run_once = |l: &mut crate::load::Loaded| {
            let st = l.model.new_state();
            let mut e = LocalEngine::new(&mut l.model, st, None, None);
            run(
                &mut e,
                &l.tok,
                &ids,
                &cfg(12),
                None,
                None,
                &cancel,
                &mut |_| {},
            )
            .unwrap()
        };
        let a = run_once(&mut l);
        let b = run_once(&mut l);
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.tokens.len(), 12);
        assert_eq!(a.finish, "length");
        // the same greedy continuation from the model's own generate
        let mut st = l.model.new_state();
        let mut s = companion_engine::model::Sampler::new(0.0, 1, 1);
        let mut got = vec![];
        let opts = companion_engine::model::GenerateOptions {
            max_tokens: 12,
            stop_tokens: vec![],
            cancel: None,
        };
        l.model
            .generate(&mut st, &ids, &mut s, &opts, &mut |t| {
                got.push(t);
                true
            })
            .unwrap();
        assert_eq!(a.tokens, got);
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_cancelled_run_stops_at_once() {
        let (mut l, d) = tiny();
        let ids: Vec<usize> = l
            .tok
            .encode("hi", true)
            .into_iter()
            .map(|i| i as usize)
            .collect();
        let cancel = AtomicBool::new(true);
        let st = l.model.new_state();
        let mut e = LocalEngine::new(&mut l.model, st, None, None);
        let o = run(
            &mut e,
            &l.tok,
            &ids,
            &cfg(50),
            None,
            None,
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!((o.finish, o.tokens.len()), ("cancelled", 0));
        assert!(
            run(
                &mut e,
                &l.tok,
                &[],
                &cfg(5),
                None,
                None,
                &cancel,
                &mut |_| {}
            )
            .is_err()
        );
        let long = vec![65usize; 400];
        assert!(
            run(
                &mut e,
                &l.tok,
                &long,
                &cfg(5),
                None,
                None,
                &cancel,
                &mut |_| {}
            )
            .unwrap_err()
            .contains("context window")
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn eviction_lets_generation_run_past_the_cache_budget() {
        let (mut l, d) = tiny();
        let ids: Vec<usize> = l
            .tok
            .encode("some prompt text", true)
            .into_iter()
            .map(|i| i as usize)
            .collect();
        let cancel = AtomicBool::new(false);
        let st = l.model.new_state();
        let kv = Some(KvPolicyCfg {
            kind: PolicyKind::Streaming,
            budget: 24,
            sinks: 4,
        });
        let mut e = LocalEngine::new(&mut l.model, st, kv, None);
        // more tokens than the 160-token context could ever hold without eviction
        let o = run(
            &mut e,
            &l.tok,
            &ids,
            &cfg(300),
            None,
            None,
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!((o.finish, o.tokens.len()), ("length", 300));
        assert!(
            e.pos() <= 24 + 8 + 1 + 24 / 16,
            "cache holds {} slots",
            e.pos()
        );
        assert!(e.take_notes().iter().any(|n| n.contains("evicted")));
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_saved_session_is_reused_and_gives_the_same_continuation() {
        let (mut l, d) = tiny();
        let sess = tmpdir("sess");
        let ids: Vec<usize> = l
            .tok
            .encode("the same long prompt for both runs", true)
            .into_iter()
            .map(|i| i as usize)
            .collect();
        let cancel = AtomicBool::new(false);
        let mut outs = vec![];
        let mut notes = vec![];
        for _ in 0..2 {
            let store = open_sessions(&args(&format!("--session {}", sess.display()))).unwrap();
            let st = l.model.new_state();
            let mut e = LocalEngine::new(&mut l.model, st, None, store);
            outs.push(
                run(
                    &mut e,
                    &l.tok,
                    &ids,
                    &cfg(8),
                    None,
                    None,
                    &cancel,
                    &mut |_| {},
                )
                .unwrap()
                .tokens,
            );
            notes.push(e.take_notes());
        }
        assert_eq!(outs[0], outs[1]);
        assert!(
            notes[0].iter().any(|n| n.contains("saved the cache")),
            "{:?}",
            notes[0]
        );
        assert!(
            notes[1].iter().any(|n| n.contains("reusing")),
            "{:?}",
            notes[1]
        );
        std::fs::remove_dir_all(d).unwrap();
        std::fs::remove_dir_all(sess).unwrap();
    }

    struct Scripted(Vec<Reading>);
    impl Sensors for Scripted {
        fn read(&mut self) -> Reading {
            if self.0.len() > 1 {
                self.0.remove(0)
            } else {
                self.0[0].clone()
            }
        }
    }

    #[test]
    fn the_governor_lowers_the_thread_count_and_the_loop_pauses() {
        let (mut l, d) = tiny();
        let hp = HardwareProfile::detect(None);
        let hot = Reading {
            cpu_temp_c: Some(120.0),
            ..Default::default()
        };
        let mut g = GovernorLoop::new(&hp, 1 << 20, 1 << 10, false)
            .with_sensors(Box::new(Scripted(vec![hot])), Duration::ZERO);
        let ids: Vec<usize> = l
            .tok
            .encode("x", true)
            .into_iter()
            .map(|i| i as usize)
            .collect();
        let cancel = AtomicBool::new(false);
        let st = l.model.new_state();
        let mut e = LocalEngine::new(&mut l.model, st, None, None);
        let t = Instant::now();
        let o = run(
            &mut e,
            &l.tok,
            &ids,
            &cfg(3),
            None,
            Some(&mut g),
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(o.tokens.len(), 3);
        assert_eq!(g.shared.threads(), 1);
        assert!(g.shared.throttle() >= Duration::from_millis(250));
        assert!(
            t.elapsed() >= Duration::from_millis(400),
            "two pauses of 250 ms were expected"
        );
        std::fs::remove_dir_all(d).unwrap();
    }
}
