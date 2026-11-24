//! Several models behind one server: discovered in a `--models-dir` and in the config dir (Modelfiles),
//! loaded on first use, unloaded when idle or when `--max-loaded` is exceeded (least recently used first).
//!
//! Names: a folder or `.gguf` file in the models dir is served as its name (without `.gguf`); a
//! `NAME.Modelfile` (or `NAME.modelfile`) next to them, or in `<config dir>/models/`, defines `NAME` on top of
//! the model its `FROM` names (a path, or another registered model), with its own system prompt, template and defaults.
use crate::load::{LoadCfg, Loaded, load, resolve};
use crate::modelfile::Modelfile;
use crate::serve::now_secs;
use crate::template::{ChatTemplate, Message};
use companion_engine::model::State;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default)]
pub struct RegCfg {
    pub load: LoadCfg,
    /// Most models in memory at once (0 = no limit).
    pub max_loaded: usize,
    /// Unload a model after this long without a request.
    pub idle: Option<Duration>,
    pub threads: Option<usize>,
    /// Cap the context window (tokens).
    pub ctx: Option<usize>,
}

/// A loaded model plus what a Modelfile layered on it.
pub struct Slot {
    pub name: String,
    pub loaded: Loaded,
    /// Held while generating; holds the conversation state so the next turn can reuse the prompt prefix.
    pub gate: Mutex<Option<State>>,
    pub system: Option<String>,
    pub preface: Vec<Message>,
    /// Request defaults from `PARAMETER` lines.
    pub options: Map<String, Value>,
    /// Vocabulary trie for grammar-constrained decoding (built on first use) and compiled grammars by source.
    pub trie: std::sync::OnceLock<Arc<companion_engine::grammar::TokenTrie>>,
    pub grammars: Mutex<std::collections::HashMap<String, Arc<companion_engine::grammar::Grammar>>>,
    /// Token table for the quote lock (built on first use).
    pub quote_vocab: std::sync::OnceLock<Arc<companion_engine::quotelock::QuoteVocab>>,
    last_used: AtomicU64,
}
impl Slot {
    pub fn touch(&self) {
        self.last_used.store(now_secs() as u64, Ordering::Relaxed);
    }
    pub fn idle_secs(&self) -> u64 {
        (now_secs() as u64).saturating_sub(self.last_used.load(Ordering::Relaxed))
    }
}

pub struct Entry {
    pub name: String,
    pub aliases: Vec<String>,
    pub path: PathBuf,
    pub modelfile: Option<Modelfile>,
    pub bytes: u64,
    slot: Mutex<Option<Arc<Slot>>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RegError {
    NotFound(String),
    Load(String),
    Busy,
}
impl std::fmt::Display for RegError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegError::NotFound(n) => write!(f, "the model `{n}` does not exist"),
            RegError::Load(e) => write!(f, "could not load the model: {e}"),
            RegError::Busy => write!(f, "every model slot is busy; try again later"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Info {
    pub name: String,
    pub aliases: Vec<String>,
    pub path: PathBuf,
    pub loaded: bool,
    pub bytes: u64,
    pub idle_secs: Option<u64>,
    pub system: Option<String>,
    pub modelfile: Option<String>,
}

pub struct Registry {
    pub cfg: RegCfg,
    entries: RwLock<Vec<Arc<Entry>>>,
    default: RwLock<Option<String>>,
    dirs: Mutex<Vec<PathBuf>>,
    config_dir: Mutex<Option<PathBuf>>,
    last_scan: Mutex<Option<Instant>>,
    pub loads: AtomicU64,
}

fn dir_bytes(p: &Path) -> u64 {
    if p.is_file() {
        return std::fs::metadata(p).map_or(0, |m| m.len());
    }
    std::fs::read_dir(p).map_or(0, |d| {
        d.filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .filter(|m| m.is_file())
            .map(|m| m.len())
            .sum()
    })
}

fn valid_name(n: &str) -> bool {
    !n.is_empty()
        && n.len() <= 128
        && !n.starts_with('.')
        && n.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b':' | b'/' | b'@' | b'+')
        })
}

impl Registry {
    pub fn new(cfg: RegCfg) -> Registry {
        Registry {
            cfg,
            entries: RwLock::default(),
            default: RwLock::default(),
            dirs: Mutex::default(),
            config_dir: Mutex::default(),
            last_scan: Mutex::default(),
            loads: AtomicU64::new(0),
        }
    }

    /// One already-loaded model (what `corbul serve MODEL` uses).
    pub fn single(loaded: Loaded, name: &str, cfg: RegCfg) -> Registry {
        let r = Registry::new(cfg);
        let e = Arc::new(Entry {
            name: name.to_string(),
            aliases: vec![],
            path: loaded.path.clone(),
            modelfile: None,
            bytes: 0,
            slot: Mutex::new(None),
        });
        let slot = r.make_slot(name, loaded, None);
        *e.slot.lock().unwrap_or_else(|x| x.into_inner()) = Some(Arc::new(slot));
        r.entries.write().unwrap_or_else(|x| x.into_inner()).push(e);
        *r.default.write().unwrap_or_else(|x| x.into_inner()) = Some(name.to_string());
        r
    }

    fn make_slot(&self, name: &str, mut loaded: Loaded, mf: Option<&Modelfile>) -> Slot {
        if let Some(n) = self.cfg.threads {
            loaded.model.set_threads(n);
        }
        if let Some(c) = self.cfg.ctx.filter(|c| *c > 0) {
            loaded.model.spec.context = loaded.model.spec.context.min(c as _);
        }
        let (mut system, mut preface, mut options) = (None, vec![], Map::new());
        if let Some(m) = mf {
            if let Some(t) = &m.template {
                let (tpl, warn) = ChatTemplate::from_source(Some(t), loaded.builtin);
                loaded.template = tpl;
                loaded.warnings.extend(warn);
            }
            system = m.system.clone();
            preface = m.messages.iter().map(|(r, c)| Message::new(r, c)).collect();
            options = m.options();
            if !m.adapters.is_empty() {
                loaded.warnings.push("ADAPTER lines are recorded but this build does not apply adapters at load time".into());
            }
        }
        let s = Slot {
            name: name.to_string(),
            loaded,
            gate: Mutex::new(None),
            system,
            preface,
            options,
            trie: Default::default(),
            grammars: Default::default(),
            quote_vocab: Default::default(),
            last_used: AtomicU64::new(0),
        };
        s.touch();
        s
    }

    pub fn set_default(&self, name: &str) {
        *self.default.write().unwrap_or_else(|x| x.into_inner()) = Some(name.to_string());
    }
    pub fn default_name(&self) -> Option<String> {
        self.default
            .read()
            .unwrap_or_else(|x| x.into_inner())
            .clone()
            .or_else(|| {
                self.entries
                    .read()
                    .unwrap_or_else(|x| x.into_inner())
                    .first()
                    .map(|e| e.name.clone())
            })
    }

    pub fn add(
        &self,
        name: &str,
        path: &Path,
        aliases: Vec<String>,
        modelfile: Option<Modelfile>,
    ) -> Result<(), String> {
        if !valid_name(name) {
            return Err(format!("`{name}` is not a valid model name"));
        }
        for a in &aliases {
            if !valid_name(a) {
                return Err(format!("`{a}` is not a valid alias"));
            }
        }
        let mut es = self.entries.write().unwrap_or_else(|x| x.into_inner());
        if es
            .iter()
            .any(|e| e.name == name || e.aliases.iter().any(|a| a == name))
        {
            return Err(format!("a model named `{name}` already exists"));
        }
        es.push(Arc::new(Entry {
            name: name.to_string(),
            aliases,
            path: path.to_path_buf(),
            modelfile,
            bytes: dir_bytes(path),
            slot: Mutex::new(None),
        }));
        Ok(())
    }

    /// `alias` now also names `target` (a model or another alias).
    pub fn alias(&self, alias: &str, target: &str) -> Result<(), String> {
        if !valid_name(alias) {
            return Err(format!("`{alias}` is not a valid alias"));
        }
        let mut es = self.entries.write().unwrap_or_else(|x| x.into_inner());
        if es
            .iter()
            .any(|e| e.name == alias || e.aliases.iter().any(|a| a == alias))
        {
            return Err(format!("`{alias}` is already taken"));
        }
        let i = es
            .iter()
            .position(|e| e.name == target || e.aliases.iter().any(|a| a == target))
            .ok_or_else(|| format!("no model named `{target}`"))?;
        let old = &es[i];
        let mut aliases = old.aliases.clone();
        aliases.push(alias.to_string());
        let slot = old.slot.lock().unwrap_or_else(|x| x.into_inner()).clone();
        es[i] = Arc::new(Entry {
            name: old.name.clone(),
            aliases,
            path: old.path.clone(),
            modelfile: old.modelfile.clone(),
            bytes: old.bytes,
            slot: Mutex::new(slot),
        });
        Ok(())
    }

    pub fn remove(&self, name: &str) -> bool {
        let mut es = self.entries.write().unwrap_or_else(|x| x.into_inner());
        let n = es.len();
        es.retain(|e| e.name != name);
        es.len() != n
    }

    /// Find models in `dir` (sub-folders with a model, `.gguf` files, `.Modelfile` files). Returns how many were added.
    pub fn scan_dir(&self, dir: &Path) -> Result<usize, String> {
        {
            let mut d = self.dirs.lock().unwrap_or_else(|x| x.into_inner());
            if !d.iter().any(|x| x == dir) {
                d.push(dir.to_path_buf());
            }
        }
        self.rescan_one(dir, false)
    }

    /// `<dir>/models/*.Modelfile` is where `POST /api/create` keeps what it creates.
    pub fn set_config_dir(&self, dir: &Path) -> Result<usize, String> {
        *self.config_dir.lock().unwrap_or_else(|x| x.into_inner()) = Some(dir.to_path_buf());
        let m = dir.join("models");
        if !m.is_dir() {
            return Ok(0);
        }
        self.rescan_one(&m, true)
    }

    pub fn config_dir(&self) -> Option<PathBuf> {
        self.config_dir
            .lock()
            .unwrap_or_else(|x| x.into_inner())
            .clone()
    }

    fn rescan_one(&self, dir: &Path, only_modelfiles: bool) -> Result<usize, String> {
        let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut items: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
        items.sort();
        let known = |n: &str| {
            self.entries
                .read()
                .unwrap_or_else(|x| x.into_inner())
                .iter()
                .any(|e| e.name == n)
        };
        let mut added = 0;
        let mut modelfiles = vec![];
        for p in items {
            let fname = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if fname.starts_with('.') {
                continue;
            }
            let lower = fname.to_ascii_lowercase();
            if p.is_file() && lower.ends_with(".modelfile") {
                modelfiles.push((fname[..fname.len() - ".modelfile".len()].to_string(), p));
            } else if !only_modelfiles {
                let stem = if p.is_file() && lower.ends_with(".gguf") {
                    fname[..fname.len() - 5].to_string()
                } else if p.is_dir() && resolve(&p).is_ok() {
                    fname.clone()
                } else {
                    continue;
                };
                if !known(&stem) && self.add(&stem, &p, vec![], None).is_ok() {
                    added += 1;
                }
            }
        }
        // Modelfiles last: their FROM may name a model found above
        for (name, p) in modelfiles {
            if known(&name) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            let Ok(mf) = Modelfile::parse(&text) else {
                continue;
            };
            let base = p.parent().unwrap_or(Path::new("."));
            let from = Path::new(&mf.from);
            let target = if from.is_absolute() {
                from.to_path_buf()
            } else {
                base.join(from)
            };
            let path = if target.exists() {
                Some(target)
            } else {
                self.entries
                    .read()
                    .unwrap_or_else(|x| x.into_inner())
                    .iter()
                    .find(|e| e.name == mf.from || e.aliases.contains(&mf.from))
                    .map(|e| e.path.clone())
            };
            if let Some(path) = path
                && self.add(&name, &path, vec![], Some(mf)).is_ok()
            {
                added += 1;
            }
        }
        Ok(added)
    }

    /// Re-read the directories at most every few seconds (new downloads show up without a restart).
    fn refresh(&self) {
        {
            let mut l = self.last_scan.lock().unwrap_or_else(|x| x.into_inner());
            if l.is_some_and(|t| t.elapsed() < Duration::from_secs(3)) {
                return;
            }
            *l = Some(Instant::now());
        }
        let dirs = self.dirs.lock().unwrap_or_else(|x| x.into_inner()).clone();
        for d in dirs {
            let _ = self.rescan_one(&d, false);
        }
        if let Some(c) = self.config_dir() {
            let _ = self.rescan_one(&c.join("models"), true);
        }
    }

    fn find(&self, name: &str) -> Option<Arc<Entry>> {
        self.entries
            .read()
            .unwrap_or_else(|x| x.into_inner())
            .iter()
            .find(|e| {
                e.name == name
                    || e.aliases.iter().any(|a| a == name)
                    || e.name.strip_suffix(":latest") == Some(name)
                    || name.strip_suffix(":latest") == Some(e.name.as_str())
            })
            .cloned()
    }

    /// Exact lookup (no leniency): the model or alias `name`, loaded or not.
    pub fn info(&self, name: &str) -> Option<Info> {
        self.infos()
            .into_iter()
            .find(|i| i.name == name || i.aliases.iter().any(|a| a == name))
    }

    pub fn len(&self) -> usize {
        self.entries.read().unwrap_or_else(|x| x.into_inner()).len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn names(&self) -> Vec<String> {
        self.entries
            .read()
            .unwrap_or_else(|x| x.into_inner())
            .iter()
            .map(|e| e.name.clone())
            .collect()
    }

    /// The canonical name a request's `model` field refers to, if any (without loading anything).
    pub fn canonical(&self, requested: Option<&str>) -> Result<String, RegError> {
        let req = requested.map(str::trim).filter(|s| !s.is_empty());
        if let Some(n) = req {
            if let Some(e) = self.find(n) {
                return Ok(e.name.clone());
            }
            self.refresh();
            if let Some(e) = self.find(n) {
                return Ok(e.name.clone());
            }
            // a lone model answers to any name (like llama-server)
            if self.len() == 1 && self.default_name().is_some() {
                return Ok(self.default_name().unwrap_or_default());
            }
            return Err(RegError::NotFound(n.to_string()));
        }
        self.default_name()
            .ok_or_else(|| RegError::NotFound("(none)".into()))
    }

    pub fn infos(&self) -> Vec<Info> {
        self.refresh();
        self.entries
            .read()
            .unwrap_or_else(|x| x.into_inner())
            .iter()
            .map(|e| {
                let slot = e.slot.try_lock().ok().and_then(|g| g.clone());
                Info {
                    name: e.name.clone(),
                    aliases: e.aliases.clone(),
                    path: e.path.clone(),
                    loaded: slot.is_some(),
                    bytes: e.bytes,
                    idle_secs: slot.as_ref().map(|s| s.idle_secs()),
                    system: e.modelfile.as_ref().and_then(|m| m.system.clone()),
                    modelfile: e.modelfile.as_ref().map(|m| m.from.clone()),
                }
            })
            .collect()
    }

    pub fn loaded_count(&self) -> usize {
        self.entries
            .read()
            .unwrap_or_else(|x| x.into_inner())
            .iter()
            .filter(|e| e.slot.try_lock().map_or(true, |g| g.is_some()))
            .count()
    }

    pub fn loaded_slots(&self) -> Vec<Arc<Slot>> {
        self.entries
            .read()
            .unwrap_or_else(|x| x.into_inner())
            .iter()
            .filter_map(|e| e.slot.try_lock().ok().and_then(|g| g.clone()))
            .collect()
    }

    /// The model for a request, loading it when needed.
    pub fn get(&self, requested: Option<&str>) -> Result<Arc<Slot>, RegError> {
        let name = self.canonical(requested)?;
        let entry = self
            .find(&name)
            .ok_or_else(|| RegError::NotFound(name.clone()))?;
        {
            let g = entry.slot.lock().unwrap_or_else(|x| x.into_inner());
            if let Some(s) = g.as_ref() {
                s.touch();
                return Ok(s.clone());
            }
        }
        self.make_room(&entry)?;
        // re-lock: loading is serialised per model so two requests do not load it twice
        let mut g = entry.slot.lock().unwrap_or_else(|x| x.into_inner());
        if let Some(s) = g.as_ref() {
            s.touch();
            return Ok(s.clone());
        }
        let loaded = load(&entry.path, &self.cfg.load).map_err(RegError::Load)?;
        self.loads.fetch_add(1, Ordering::Relaxed);
        let s = Arc::new(self.make_slot(&entry.name, loaded, entry.modelfile.as_ref()));
        *g = Some(s.clone());
        Ok(s)
    }

    /// Unload least-recently-used models until one more fits under `max_loaded`.
    fn make_room(&self, wanted: &Arc<Entry>) -> Result<(), RegError> {
        let max = self.cfg.max_loaded;
        if max == 0 {
            return Ok(());
        }
        loop {
            let es = self
                .entries
                .read()
                .unwrap_or_else(|x| x.into_inner())
                .clone();
            let loaded: Vec<(Arc<Entry>, Arc<Slot>)> = es
                .iter()
                .filter(|e| !Arc::ptr_eq(e, wanted))
                .filter_map(|e| {
                    e.slot
                        .try_lock()
                        .ok()
                        .and_then(|g| g.clone())
                        .map(|s| (e.clone(), s))
                })
                .collect();
            if loaded.len() < max {
                return Ok(());
            }
            // the oldest one nobody is using (strong count: the entry's + our clone)
            let victim = loaded
                .iter()
                .filter(|(_, s)| Arc::strong_count(s) <= 2 && s.gate.try_lock().is_ok())
                .max_by_key(|(_, s)| s.idle_secs());
            let Some((e, _)) = victim else {
                return Err(RegError::Busy);
            };
            *e.slot.lock().unwrap_or_else(|x| x.into_inner()) = None;
        }
    }

    pub fn unload(&self, name: &str) -> bool {
        let Some(e) = self.find(name) else {
            return false;
        };
        let mut g = e.slot.lock().unwrap_or_else(|x| x.into_inner());
        match g.as_ref() {
            Some(s) if Arc::strong_count(s) <= 1 => {
                *g = None;
                true
            }
            _ => false,
        }
    }

    /// Unload models idle for longer than the configured time. Returns how many.
    pub fn sweep_idle(&self) -> usize {
        let Some(idle) = self.cfg.idle else { return 0 };
        let mut n = 0;
        for e in self
            .entries
            .read()
            .unwrap_or_else(|x| x.into_inner())
            .clone()
        {
            let Ok(mut g) = e.slot.try_lock() else {
                continue;
            };
            if let Some(s) = g.as_ref()
                && Arc::strong_count(s) <= 1
                && s.idle_secs() >= idle.as_secs()
                && s.gate.try_lock().is_ok()
            {
                *g = None;
                n += 1;
            }
        }
        n
    }

    /// Register (and persist, when a config dir is set) a model defined by a Modelfile text.
    /// `FROM` may be any existing path: for trusted callers (the command line).
    pub fn create(&self, name: &str, modelfile_text: &str) -> Result<(), String> {
        self.create_with(name, modelfile_text, false)
    }

    /// Like [`Registry::create`] for callers that arrive over the network: `FROM` must be a model the server
    /// already serves or a path inside one of the directories it serves (symlinks resolved), never an
    /// arbitrary file system path. The error does not echo or reveal what exists outside.
    pub fn create_restricted(&self, name: &str, modelfile_text: &str) -> Result<(), String> {
        self.create_with(name, modelfile_text, true)
    }

    /// Is `p` (after resolving symlinks and `..`) inside a served directory or a registered model's own path?
    fn confined(&self, p: &Path) -> bool {
        let Ok(c) = p.canonicalize() else {
            return false;
        };
        let inside = |root: &Path| root.canonicalize().is_ok_and(|r| c.starts_with(&r));
        if self
            .dirs
            .lock()
            .unwrap_or_else(|x| x.into_inner())
            .iter()
            .any(|d| inside(d))
        {
            return true;
        }
        self.entries
            .read()
            .unwrap_or_else(|x| x.into_inner())
            .iter()
            .any(|e| inside(&e.path))
    }

    fn create_with(
        &self,
        name: &str,
        modelfile_text: &str,
        restricted: bool,
    ) -> Result<(), String> {
        let mf = Modelfile::parse(modelfile_text)?;
        if !valid_name(name) || name.contains('/') || name.contains(':') {
            return Err(format!("`{name}` is not a valid model name"));
        }
        let path = {
            let from = Path::new(&mf.from);
            if restricted {
                if let Some(e) = self.find(&mf.from) {
                    e.path.clone()
                } else if from.exists() && self.confined(from) {
                    from.to_path_buf()
                } else {
                    return Err("FROM must name a model this server already serves (or a path inside its model directories)".into());
                }
            } else if from.exists() {
                from.to_path_buf()
            } else {
                self.find(&mf.from).map(|e| e.path.clone()).ok_or_else(|| {
                    format!("FROM `{}` is neither a path nor a known model", mf.from)
                })?
            }
        };
        self.add(name, &path, vec![], Some(mf))?;
        if let Some(c) = self.config_dir() {
            let d = c.join("models");
            let w = std::fs::create_dir_all(&d)
                .and_then(|_| std::fs::write(d.join(format!("{name}.Modelfile")), modelfile_text));
            if let Err(e) = w {
                self.remove(name);
                return Err(format!("cannot save the Modelfile: {e}"));
            }
        }
        Ok(())
    }

    /// Forget a model made with [`Registry::create`] (and its saved Modelfile).
    pub fn delete(&self, name: &str) -> Result<(), String> {
        let e = self
            .find(name)
            .ok_or_else(|| format!("no model named `{name}`"))?;
        if e.modelfile.is_none() {
            return Err("only models made from a Modelfile can be deleted through the API".into());
        }
        self.remove(&e.name);
        if let Some(c) = self.config_dir() {
            for ext in ["Modelfile", "modelfile"] {
                let _ = std::fs::remove_file(c.join("models").join(format!("{}.{ext}", e.name)));
            }
        }
        Ok(())
    }
}

/// `$XDG_CONFIG_HOME/corbul`, `~/.config/corbul`, or `%APPDATA%\corbul`.
pub fn default_config_dir() -> Option<PathBuf> {
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    env("CORBUL_CONFIG_DIR")
        .or_else(|| env("XDG_CONFIG_HOME").map(|p| p.join("corbul")))
        .or_else(|| env("APPDATA").map(|p| p.join("corbul")))
        .or_else(|| env("HOME").map(|p| p.join(".config").join("corbul")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::testkit::*;

    fn models_dir() -> PathBuf {
        let root = tmpdir("reg");
        for n in ["alpha", "beta"] {
            let d = root.join(n);
            std::fs::create_dir_all(&d).unwrap();
            write_model(&d, true);
        }
        std::fs::create_dir_all(root.join("not-a-model")).unwrap();
        root
    }

    #[test]
    fn discovers_and_loads_lazily() {
        let root = models_dir();
        let r = Registry::new(RegCfg::default());
        assert_eq!(r.scan_dir(&root).unwrap(), 2);
        assert_eq!(r.names(), vec!["alpha", "beta"]);
        assert_eq!(r.loaded_count(), 0);
        let s = r.get(Some("alpha")).unwrap();
        assert_eq!(s.name, "alpha");
        assert_eq!(r.loaded_count(), 1);
        assert_eq!(r.loads.load(Ordering::Relaxed), 1);
        r.get(Some("alpha")).unwrap();
        assert_eq!(
            r.loads.load(Ordering::Relaxed),
            1,
            "second use must not reload"
        );
        assert_eq!(
            r.get(None).unwrap().name,
            "alpha",
            "first model is the default"
        );
        assert_eq!(
            r.get(Some("nope")).err(),
            Some(RegError::NotFound("nope".into()))
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn aliases_and_latest_suffix() {
        let root = models_dir();
        let r = Registry::new(RegCfg::default());
        r.scan_dir(&root).unwrap();
        r.alias("gpt-4", "beta").unwrap();
        assert_eq!(r.canonical(Some("gpt-4")).unwrap(), "beta");
        assert_eq!(r.canonical(Some("beta:latest")).unwrap(), "beta");
        assert!(
            r.alias("gpt-4", "alpha").is_err()
                && r.alias("x", "missing").is_err()
                && r.alias("bad name", "alpha").is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn max_loaded_evicts_the_least_recently_used_idle_model() {
        let root = models_dir();
        let r = Registry::new(RegCfg {
            max_loaded: 1,
            ..RegCfg::default()
        });
        r.scan_dir(&root).unwrap();
        drop(r.get(Some("alpha")).unwrap());
        drop(r.get(Some("beta")).unwrap());
        assert_eq!(r.loaded_count(), 1);
        let infos = r.infos();
        assert!(!infos.iter().find(|i| i.name == "alpha").unwrap().loaded);
        assert!(infos.iter().find(|i| i.name == "beta").unwrap().loaded);
        // a busy model cannot be evicted
        let held = r.get(Some("beta")).unwrap();
        assert_eq!(r.get(Some("alpha")).err(), Some(RegError::Busy));
        drop(held);
        r.get(Some("alpha")).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn idle_models_are_unloaded() {
        let root = models_dir();
        let r = Registry::new(RegCfg {
            idle: Some(Duration::from_secs(0)),
            ..RegCfg::default()
        });
        r.scan_dir(&root).unwrap();
        let held = r.get(Some("alpha")).unwrap();
        assert_eq!(r.sweep_idle(), 0, "in use");
        drop(held);
        assert_eq!(r.sweep_idle(), 1);
        assert_eq!(r.loaded_count(), 0);
        r.get(Some("alpha")).unwrap();
        assert_eq!(r.loads.load(Ordering::Relaxed), 2);
        let none = Registry::new(RegCfg::default());
        assert_eq!(none.sweep_idle(), 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn modelfiles_layer_on_a_base_model() {
        let root = models_dir();
        std::fs::write(root.join("terse.Modelfile"), "FROM alpha\nSYSTEM \"\"\"Be terse.\"\"\"\nPARAMETER temperature 0\nPARAMETER stop END\nMESSAGE user hi\nMESSAGE assistant yo\n").unwrap();
        std::fs::write(root.join("broken.Modelfile"), "FROM nowhere\n").unwrap();
        let r = Registry::new(RegCfg::default());
        assert_eq!(r.scan_dir(&root).unwrap(), 3);
        assert!(
            r.names().contains(&"terse".to_string()) && !r.names().contains(&"broken".to_string())
        );
        let s = r.get(Some("terse")).unwrap();
        assert_eq!(s.system.as_deref(), Some("Be terse."));
        assert_eq!(s.options["temperature"], serde_json::json!(0));
        assert_eq!(s.options["stop"], serde_json::json!(["END"]));
        assert_eq!(s.preface.len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn create_and_delete_persist_in_the_config_dir() {
        let root = models_dir();
        let cfg = tmpdir("regcfg");
        let r = Registry::new(RegCfg::default());
        r.scan_dir(&root).unwrap();
        r.set_config_dir(&cfg).unwrap();
        r.create("mine", "FROM beta\nSYSTEM hello").unwrap();
        assert!(cfg.join("models/mine.Modelfile").is_file());
        assert!(
            r.create("mine", "FROM beta").is_err()
                && r.create("x", "FROM nowhere").is_err()
                && r.create("../evil", "FROM beta").is_err()
        );
        // a fresh registry finds it again
        let r2 = Registry::new(RegCfg::default());
        r2.scan_dir(&root).unwrap();
        r2.set_config_dir(&cfg).unwrap();
        assert!(r2.names().contains(&"mine".to_string()));
        assert!(r.delete("beta").is_err());
        r.delete("mine").unwrap();
        // network callers: no arbitrary paths, nothing revealed about what exists elsewhere
        let outside = tmpdir("outside");
        let a = r
            .create_restricted("viaapi", &format!("FROM {}", outside.display()))
            .unwrap_err();
        let b = r
            .create_restricted("viaapi", "FROM /definitely/not/here")
            .unwrap_err();
        assert_eq!(a, b, "existing and missing paths must be indistinguishable");
        assert!(!a.contains(outside.to_str().unwrap()));
        assert!(
            r.create_restricted("viaapi", "FROM /etc").is_err()
                && r.create_restricted("viaapi", "FROM ../../etc").is_err()
        );
        r.create_restricted("viaapi", "FROM beta").unwrap();
        r.create_restricted("viaapi2", &format!("FROM {}", root.join("beta").display()))
            .unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside, root.join("escape")).unwrap();
            assert!(
                r.create_restricted(
                    "viaapi3",
                    &format!("FROM {}", root.join("escape").display())
                )
                .is_err(),
                "a symlink must not escape the served directory"
            );
        }
        let _ = std::fs::remove_dir_all(outside);
        assert!(
            !cfg.join("models/mine.Modelfile").exists() && !r.names().contains(&"mine".to_string())
        );
        std::fs::remove_dir_all(root).unwrap();
        std::fs::remove_dir_all(cfg).unwrap();
    }

    #[test]
    fn a_lone_model_answers_to_any_name() {
        let root = tmpdir("reg1");
        let d = root.join("only");
        std::fs::create_dir_all(&d).unwrap();
        write_model(&d, true);
        let r = Registry::new(RegCfg::default());
        r.scan_dir(&root).unwrap();
        assert_eq!(r.canonical(Some("whatever")).unwrap(), "only");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ctx_and_threads_are_applied() {
        let root = models_dir();
        let r = Registry::new(RegCfg {
            ctx: Some(32),
            threads: Some(1),
            ..RegCfg::default()
        });
        r.scan_dir(&root).unwrap();
        let s = r.get(Some("alpha")).unwrap();
        assert_eq!(s.loaded.model.spec.context, 32);
        assert_eq!(s.loaded.model.threads(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }
}
