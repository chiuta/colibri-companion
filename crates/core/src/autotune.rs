//! Per-machine autotuner: a small, bounded search over the few settings colibri documents
//! (`PIN`/`PIN_GB`, `OMP_NUM_THREADS`, `--ctx`), scored by measured tokens per second.
//!
//! It invents no flags. `coli tune` stays the first step (it knows colibri's quality gates); this
//! module only explores around the result and remembers the winner for this computer and model.
//! The way a trial is run is behind [`Runner`], so the search is tested without any engine.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// The settings that are tried. `None` leaves colibri's own choice untouched.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Knobs {
    /// Hot-store budget in GB (`PIN=auto` + `PIN_GB`).
    pub pin_gb: Option<f64>,
    /// OpenMP team size (`OMP_NUM_THREADS`; setting it hands the sizing to the user).
    pub threads: Option<u32>,
    /// Context length (`--ctx`).
    pub ctx: Option<u32>,
}

/// Largest hot store, thread team and context that are passed on to colibri: a profile read back
/// from a file the user can edit, or a space computed from odd measurements, must not turn into
/// `PIN_GB=-5`, `OMP_NUM_THREADS=0` or a number that does not fit.
const MAX_PIN_GB: f64 = 4096.0;
const MAX_THREADS: u32 = 1024;

impl Knobs {
    /// The same settings with every value that colibri could not use dropped (left to its own
    /// choice) or brought into range.
    pub fn sanitized(&self) -> Knobs {
        Knobs {
            pin_gb: self
                .pin_gb
                .filter(|g| g.is_finite() && *g > 0.0)
                .map(|g| g.min(MAX_PIN_GB)),
            threads: self.threads.filter(|t| *t >= 1).map(|t| t.min(MAX_THREADS)),
            ctx: self.ctx,
        }
    }

    /// Environment for a run. Values are rounded so equal settings give equal text.
    pub fn env(&self) -> Vec<(String, String)> {
        let this = self.sanitized();
        let mut v = Vec::new();
        if let Some(gb) = this.pin_gb {
            v.push(("PIN".to_string(), "auto".to_string()));
            v.push(("PIN_GB".to_string(), format!("{gb:.1}")));
        }
        if let Some(t) = this.threads {
            v.push(("OMP_NUM_THREADS".to_string(), t.to_string()));
        }
        v
    }

    pub fn ctx_arg(&self) -> Option<String> {
        self.ctx.map(|c| c.to_string())
    }

    /// A stable text form, used to avoid repeating a trial.
    fn key(&self) -> String {
        format!(
            "{:?}|{:?}|{:?}",
            self.pin_gb.map(|g| (g * 10.0).round() as i64),
            self.threads,
            self.ctx
        )
    }
}

/// What the machine offers; the candidates come from here.
#[derive(Clone, Debug, PartialEq)]
pub struct Space {
    /// RAM free for colibri, after the system's share.
    pub free_ram_gb: f64,
    /// RAM the conversation itself takes at the planned context (from `coli plan`).
    pub state_gb: f64,
    pub physical_cores: u32,
    /// Context lengths worth trying, from the planner. Empty: leave the context alone.
    pub contexts: Vec<u32>,
}

impl Space {
    pub fn pin_options(&self) -> Vec<Option<f64>> {
        let room = (self.free_ram_gb - self.state_gb - 2.0).max(0.0);
        let mut out = vec![None];
        for frac in [0.25, 0.5, 0.7] {
            let gb = (room * frac * 10.0).round() / 10.0;
            if gb >= 0.5 {
                out.push(Some(gb));
            }
        }
        out
    }

    pub fn thread_options(&self) -> Vec<Option<u32>> {
        let c = self.physical_cores.clamp(1, MAX_THREADS);
        let mut out = vec![None];
        for t in [c / 2, c.saturating_sub(1), c] {
            if t >= 1 && !out.contains(&Some(t)) {
                out.push(Some(t));
            }
        }
        out
    }

    pub fn ctx_options(&self) -> Vec<Option<u32>> {
        let mut out = vec![None];
        out.extend(self.contexts.iter().map(|c| Some(*c)));
        out
    }
}

/// One run's result.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub tok_s: f64,
    /// Swap in use while it ran; paging makes a fast-looking run a bad one.
    pub swap_used_gb: f64,
}

/// Higher is better. Swap use above 1 GB divides the speed down: a setting that pages is not a win.
pub fn score(m: &Measurement) -> f64 {
    if !m.tok_s.is_finite() || m.tok_s <= 0.0 {
        return 0.0;
    }
    let excess = (m.swap_used_gb - 1.0).max(0.0);
    m.tok_s / (1.0 + excess)
}

/// The last "N tok/s" figure in colibri's output (ANSI colours tolerated). A figure that is
/// negative, not finite or above [`MAX_TOK_S`] is not a measurement and is skipped.
pub fn parse_tok_s(log: &str) -> Option<f64> {
    let mut found = None;
    for line in log.lines() {
        let clean: String = {
            let mut s = String::new();
            let mut it = line.chars();
            while let Some(c) = it.next() {
                if c == '\u{1b}' {
                    for n in it.by_ref() {
                        if n.is_ascii_alphabetic() {
                            break;
                        }
                    }
                } else {
                    s.push(c);
                }
            }
            s
        };
        let mut rest = clean.as_str();
        while let Some(i) = rest.find("tok/s") {
            let head = rest[..i].trim_end();
            let num: String = head
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let negative = head[..head.len() - num.len()].ends_with('-');
            if let Ok(v) = num.parse::<f64>() {
                if !negative && v.is_finite() && v <= MAX_TOK_S {
                    found = Some(v);
                }
            }
            rest = &rest[i + 5..];
        }
    }
    found
}

/// No local run of a language model reaches this; a larger figure is a misread.
const MAX_TOK_S: f64 = 100_000.0;

/// Runs one trial. The implementation starts colibri with the knobs, waits at most `budget`
/// and returns what it measured. It should stop early when `cancel` is set.
pub trait Runner {
    fn run(
        &mut self,
        knobs: &Knobs,
        budget: Duration,
        cancel: &AtomicBool,
    ) -> Result<Measurement, String>;
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Hard cap on trials, the starting point included.
    pub max_runs: usize,
    /// Time allowed for one trial (never more than an hour, whatever is asked).
    pub budget: Duration,
    /// Time allowed for the whole search; no trial starts after it.
    pub total: Duration,
}

/// Hard ceilings, so a careless configuration cannot make the search run for days.
const MAX_RUNS: usize = 200;
const MAX_TRIAL: Duration = Duration::from_secs(3600);

impl Default for Config {
    fn default() -> Self {
        Self {
            max_runs: 12,
            budget: Duration::from_secs(120),
            total: Duration::from_secs(30 * 60),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Trial {
    pub knobs: Knobs,
    pub result: Result<Measurement, String>,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub best: Knobs,
    pub best_score: f64,
    /// Score of the starting point (colibri's own choice unless a start was given).
    pub start_score: f64,
    pub trials: Vec<Trial>,
    pub aborted: bool,
    /// The total time ran out before the search was done.
    pub timed_out: bool,
}

impl Outcome {
    /// Relative gain over the start, 0 when it did not improve or there is no baseline.
    pub fn gain(&self) -> f64 {
        if self.start_score > 0.0 && self.best_score > self.start_score {
            self.best_score / self.start_score - 1.0
        } else {
            0.0
        }
    }
}

/// Coordinate descent: one setting at a time (hot store, threads, context), keeping what wins,
/// until a full round brings nothing or `max_runs` is reached. Trials already done are not repeated.
pub fn search(
    space: &Space,
    start: Knobs,
    runner: &mut dyn Runner,
    cfg: Config,
    cancel: &AtomicBool,
) -> Outcome {
    let mut trials: Vec<Trial> = Vec::new();
    let mut seen: HashMap<String, f64> = HashMap::new();
    let mut aborted = false;
    let mut timed_out = false;
    let started = Instant::now();
    let max_runs = cfg.max_runs.min(MAX_RUNS);
    let budget = cfg.budget.min(MAX_TRIAL);
    let start = start.sanitized();

    let mut try_one = |k: &Knobs, trials: &mut Vec<Trial>, aborted: &mut bool| -> Option<f64> {
        if let Some(s) = seen.get(&k.key()) {
            return Some(*s);
        }
        if trials.len() >= max_runs {
            return None;
        }
        if cancel.load(Ordering::Relaxed) {
            *aborted = true;
            return None;
        }
        if started.elapsed() >= cfg.total {
            timed_out = true;
            return None;
        }
        let result = runner.run(k, budget, cancel);
        if cancel.load(Ordering::Relaxed) {
            // a run that was cut short measured nothing worth keeping
            *aborted = true;
            trials.push(Trial {
                knobs: k.clone(),
                result: Err("cancelled".into()),
                score: 0.0,
            });
            return None;
        }
        let s = result.as_ref().map(score).unwrap_or(0.0);
        seen.insert(k.key(), s);
        trials.push(Trial {
            knobs: k.clone(),
            result,
            score: s,
        });
        Some(s)
    };

    let mut best = start.clone();
    let Some(start_score) = try_one(&best, &mut trials, &mut aborted) else {
        return Outcome {
            best,
            best_score: 0.0,
            start_score: 0.0,
            trials,
            aborted,
            timed_out,
        };
    };
    let mut best_score = start_score;

    'rounds: loop {
        let mut improved = false;
        for dim in 0..3 {
            let candidates: Vec<Knobs> = match dim {
                0 => space
                    .pin_options()
                    .into_iter()
                    .map(|p| Knobs {
                        pin_gb: p,
                        ..best.clone()
                    })
                    .collect(),
                1 => space
                    .thread_options()
                    .into_iter()
                    .map(|t| Knobs {
                        threads: t,
                        ..best.clone()
                    })
                    .collect(),
                _ => space
                    .ctx_options()
                    .into_iter()
                    .map(|c| Knobs {
                        ctx: c,
                        ..best.clone()
                    })
                    .collect(),
            };
            for cand in candidates {
                if cand == best {
                    continue;
                }
                match try_one(&cand, &mut trials, &mut aborted) {
                    Some(s) if s > best_score * 1.02 => {
                        // a 2% margin keeps noise from steering the search
                        best = cand;
                        best_score = s;
                        improved = true;
                    }
                    Some(_) => {}
                    None => break 'rounds,
                }
            }
        }
        if !improved {
            break;
        }
    }

    Outcome {
        best,
        best_score,
        start_score,
        trials,
        aborted,
        timed_out,
    }
}

// ------------------------------------------------------------------ memory of the winner

/// The usual sizes of memory; the amount the system reports (15.7, 31.2...) is rounded to the
/// nearest one, so the same computer always gets the same key.
const RAM_TIERS: [u64; 15] = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512];

/// The standard memory size nearest to `gb`; above the table, multiples of 128 GB.
pub fn ram_tier(gb: f64) -> u64 {
    if !gb.is_finite() || gb <= 0.0 {
        return 0;
    }
    if gb > 640.0 {
        return ((gb.min(1e9) / 128.0).round() as u64).max(5) * 128;
    }
    RAM_TIERS
        .iter()
        .copied()
        .min_by(|a, b| (*a as f64 - gb).abs().total_cmp(&(*b as f64 - gb).abs()))
        .unwrap_or(0)
}

/// `cpu`, `cuda`, `vulkan` or `metal`; anything else (including empty) is `auto`.
pub fn backend_name(backend: &str) -> &'static str {
    match backend.trim().to_ascii_lowercase().as_str() {
        "cpu" => "cpu",
        "cuda" => "cuda",
        "vulkan" => "vulkan",
        "metal" => "metal",
        _ => "auto",
    }
}

/// Identifies this computer, backend and model: another memory size, core count or backend is a
/// different case (the best settings for the GPU are not those for the processor).
pub fn machine_key(total_ram_gb: f64, physical_cores: u32, backend: &str, model: &str) -> String {
    format!(
        "{}gb-{}c-{}/{}",
        ram_tier(total_ram_gb),
        physical_cores,
        backend_name(backend),
        model
    )
}

/// A saved result older than this is measured again before it is trusted (drivers, background
/// load and the model files change).
pub const STALE_AFTER_SECS: u64 = 21 * 24 * 3600;
/// How many past measurements are kept per key.
pub const HISTORY_LEN: usize = 8;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub knobs: Knobs,
    pub score: f64,
    pub at: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub knobs: Knobs,
    /// The last measured score of `knobs` (a fresh measurement may be lower than an earlier one).
    pub score: f64,
    pub saved_at: u64,
    /// The latest measurements, oldest first, at most [`HISTORY_LEN`].
    #[serde(default)]
    pub history: Vec<Point>,
}

impl Profile {
    pub fn is_stale(&self, now: u64) -> bool {
        now.saturating_sub(self.saved_at) > STALE_AFTER_SECS
    }
}

#[derive(Default)]
struct Book {
    profiles: HashMap<String, Profile>,
}

/// Reads the saved winners. One damaged entry is dropped, not the whole book: an entry whose
/// settings or score are not usable (hand-edited, or written by a bug) is skipped.
fn read_book(path: &Path) -> Book {
    let Some(v) = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
    else {
        return Book::default();
    };
    let mut profiles = HashMap::new();
    if let Some(map) = v.get("profiles").and_then(|p| p.as_object()) {
        for (k, e) in map {
            if let Ok(mut p) = serde_json::from_value::<Profile>(e.clone()) {
                if p.score.is_finite() && p.score > 0.0 {
                    p.knobs = p.knobs.sanitized();
                    p.history.retain(|h| h.score.is_finite() && h.score > 0.0);
                    for h in &mut p.history {
                        h.knobs = h.knobs.sanitized();
                    }
                    let extra = p.history.len().saturating_sub(HISTORY_LEN);
                    p.history.drain(..extra);
                    profiles.insert(k.clone(), p);
                }
            }
        }
    }
    Book { profiles }
}

fn book_json(book: &Book) -> Result<String, std::io::Error> {
    let mut sorted: Vec<(&String, &Profile)> = book.profiles.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0)); // stable text for the same content
    let map: serde_json::Map<String, serde_json::Value> = sorted
        .into_iter()
        .map(|(k, p)| {
            Ok((
                k.clone(),
                serde_json::to_value(p).map_err(std::io::Error::other)?,
            ))
        })
        .collect::<Result<_, std::io::Error>>()?;
    serde_json::to_string_pretty(&serde_json::json!({ "profiles": map }))
        .map_err(std::io::Error::other)
}

/// Writes through a temporary file named for this process and call, then renames it over the
/// target: two programs saving at once never write into the same temporary file, and a crash
/// leaves the old file whole.
fn write_atomic_unique(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::sync::atomic::AtomicU64;
    static N: AtomicU64 = AtomicU64::new(0);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let name = path.file_name().map_or_else(
        || "profiles".to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let tmp = path.with_file_name(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let res = (|| {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// Runs `f` holding a lock file next to `path`, so two programs do not both read the book, add
/// their winner and write it back, losing one of them. A lock older than half a minute is a
/// leftover from a crash and is taken over; after ten seconds of waiting `f` runs anyway (a lost
/// update is better than a save that never happens).
fn with_lock<T>(path: &Path, f: impl FnOnce() -> T) -> T {
    let name = path.file_name().map_or_else(
        || "profiles".to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let lock = path.with_file_name(format!(".{name}.lock"));
    if let Some(dir) = lock.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let started = Instant::now();
    let mut held = false;
    // "access denied" with no lock file to be seen: a deletion has just finished (Windows) or the
    // folder really is read-only; retry briefly before giving up on the lock
    let mut denied = 0u32;
    while started.elapsed() < Duration::from_secs(10) {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock)
        {
            Ok(_) => {
                held = true;
                break;
            }
            // Windows answers "access denied" while another thread's lock file is still being
            // deleted: that is a busy lock, not a folder we cannot write to
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::PermissionDenied
                ) && (e.kind() == std::io::ErrorKind::AlreadyExists || lock.exists()) =>
            {
                let stale = std::fs::metadata(&lock)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.elapsed().ok())
                    .is_some_and(|age| age > Duration::from_secs(30));
                if stale {
                    let _ = std::fs::remove_file(&lock);
                } else {
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied && denied < 50 => {
                denied += 1;
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(_) => break, // cannot lock here (read-only folder...): the write will say so
        }
    }
    let out = f();
    if held {
        let _ = std::fs::remove_file(&lock);
    }
    out
}

/// Remembers a measured result for `key`. Rules: the same settings measured again replace the old
/// figure (a lucky first run must not stand forever); a stored result older than
/// [`STALE_AFTER_SECS`] is replaced; otherwise another setting replaces it only when better.
/// Every measurement goes into a short history. A score that is not a positive finite number is
/// refused (it would corrupt the file for every model).
pub fn save_profile(path: &Path, key: &str, knobs: &Knobs, score: f64) -> std::io::Result<()> {
    save_profile_at(path, key, knobs, score, crate::store::now_secs())
}

fn save_profile_at(
    path: &Path,
    key: &str,
    knobs: &Knobs,
    score: f64,
    now: u64,
) -> std::io::Result<()> {
    if !score.is_finite() || score <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "the score must be a positive number",
        ));
    }
    with_lock(path, || {
        // keep a copy of a file that exists but cannot be read, instead of silently replacing it
        if let Ok(t) = std::fs::read_to_string(path) {
            if serde_json::from_str::<serde_json::Value>(&t).is_err() {
                let _ = std::fs::write(path.with_extension("json.bad"), t);
            }
        }
        let mut book = read_book(path);
        let knobs = knobs.sanitized();
        let point = Point {
            knobs: knobs.clone(),
            score,
            at: now,
        };
        let entry = match book.profiles.remove(key) {
            None => Profile {
                knobs: knobs.clone(),
                score,
                saved_at: now,
                history: vec![point],
            },
            Some(mut old) => {
                old.history.push(point);
                let extra = old.history.len().saturating_sub(HISTORY_LEN);
                old.history.drain(..extra);
                if old.knobs == knobs || old.is_stale(now) || score > old.score {
                    old.knobs = knobs;
                    old.score = score;
                    old.saved_at = now;
                }
                old
            }
        };
        book.profiles.insert(key.to_string(), entry);
        write_atomic_unique(path, book_json(&book)?.as_bytes())
    })
}

/// Measures the stored best settings of `key` again and records the figure (it may be lower than
/// before). Returns the profile now stored, or `None` when nothing is stored for `key`. A failed
/// or cancelled measurement changes nothing and returns the error.
pub fn revalidate(
    path: &Path,
    key: &str,
    runner: &mut dyn Runner,
    budget: Duration,
    cancel: &AtomicBool,
) -> Result<Option<Profile>, String> {
    let Some(p) = load_profile(path, key) else {
        return Ok(None);
    };
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    let m = runner.run(&p.knobs, budget.min(Duration::from_secs(3600)), cancel)?;
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    let s = score(&m);
    if s <= 0.0 {
        return Err("the measurement was not usable".into());
    }
    save_profile(path, key, &p.knobs, s).map_err(|e| e.to_string())?;
    Ok(load_profile(path, key))
}

pub fn load_profile(path: &Path, key: &str) -> Option<Profile> {
    read_book(path).profiles.remove(key)
}

/// Where the profiles live: next to the settings file.
pub fn profiles_file() -> std::path::PathBuf {
    crate::paths::data_dir().join("autotune.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Speed is a known function of the knobs, so the optimum is known.
    struct Fake {
        calls: usize,
        swap_at_big_pin: bool,
    }

    impl Runner for Fake {
        fn run(&mut self, k: &Knobs, _b: Duration, _c: &AtomicBool) -> Result<Measurement, String> {
            self.calls += 1;
            let pin = k.pin_gb.unwrap_or(0.0);
            let thr = k.threads.unwrap_or(4) as f64;
            let ctx = k.ctx.unwrap_or(8192) as f64;
            let tok_s = 2.0 + pin * 0.5 + thr * 0.25 - ctx / 100_000.0;
            let swap = if self.swap_at_big_pin && pin > 7.0 {
                6.0
            } else {
                0.0
            };
            Ok(Measurement {
                tok_s,
                swap_used_gb: swap,
            })
        }
    }

    fn space() -> Space {
        Space {
            free_ram_gb: 14.0,
            state_gb: 1.0,
            physical_cores: 8,
            contexts: vec![4096, 16384],
        }
    }

    #[test]
    fn candidates_come_from_the_machine() {
        let s = space();
        assert_eq!(s.pin_options()[0], None);
        assert!(s.pin_options().len() >= 3);
        assert!(s.pin_options().iter().flatten().all(|g| *g <= 11.0));
        assert_eq!(s.thread_options(), vec![None, Some(4), Some(7), Some(8)]);
        assert_eq!(s.ctx_options(), vec![None, Some(4096), Some(16384)]);
    }

    #[test]
    fn env_uses_only_documented_variables() {
        let k = Knobs {
            pin_gb: Some(5.0),
            threads: Some(6),
            ctx: Some(4096),
        };
        let e = k.env();
        assert!(e.contains(&("PIN".into(), "auto".into())));
        assert!(e.contains(&("PIN_GB".into(), "5.0".into())));
        assert!(e.contains(&("OMP_NUM_THREADS".into(), "6".into())));
        assert_eq!(k.ctx_arg().as_deref(), Some("4096"));
        assert!(Knobs::default().env().is_empty());
    }

    #[test]
    fn finds_the_better_setting_and_stays_within_the_budget() {
        let mut r = Fake {
            calls: 0,
            swap_at_big_pin: false,
        };
        let cancel = AtomicBool::new(false);
        let out = search(
            &space(),
            Knobs::default(),
            &mut r,
            Config::default(),
            &cancel,
        );
        assert!(out.best_score > out.start_score);
        assert!(out.gain() > 0.0);
        assert!(r.calls <= 12 && out.trials.len() == r.calls);
        assert!(out.best.pin_gb.is_some() && out.best.threads == Some(8));
        assert!(!out.aborted);
    }

    #[test]
    fn a_setting_that_swaps_does_not_win() {
        let mut r = Fake {
            calls: 0,
            swap_at_big_pin: true,
        };
        let cancel = AtomicBool::new(false);
        let out = search(
            &space(),
            Knobs::default(),
            &mut r,
            Config::default(),
            &cancel,
        );
        assert!(out.best.pin_gb.is_none_or(|g| g <= 7.0));
    }

    #[test]
    fn never_repeats_a_trial_and_honours_max_runs() {
        let mut r = Fake {
            calls: 0,
            swap_at_big_pin: false,
        };
        let cancel = AtomicBool::new(false);
        let cfg = Config {
            max_runs: 3,
            budget: Duration::from_secs(1),
            total: Duration::from_secs(60),
        };
        let out = search(&space(), Knobs::default(), &mut r, cfg, &cancel);
        assert_eq!(r.calls, 3);
        let mut keys: Vec<_> = out.trials.iter().map(|t| t.knobs.key()).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), out.trials.len());
    }

    #[test]
    fn abort_stops_before_any_run() {
        let mut r = Fake {
            calls: 0,
            swap_at_big_pin: false,
        };
        let cancel = AtomicBool::new(true);
        let out = search(
            &space(),
            Knobs::default(),
            &mut r,
            Config::default(),
            &cancel,
        );
        assert!(out.aborted);
        assert_eq!(r.calls, 0);
    }

    #[test]
    fn a_failed_trial_scores_zero_and_the_search_goes_on() {
        struct Flaky(usize);
        impl Runner for Flaky {
            fn run(
                &mut self,
                k: &Knobs,
                _: Duration,
                _: &AtomicBool,
            ) -> Result<Measurement, String> {
                self.0 += 1;
                if k.threads == Some(4) {
                    return Err("timeout".into());
                }
                Ok(Measurement {
                    tok_s: 3.0,
                    swap_used_gb: 0.0,
                })
            }
        }
        let cancel = AtomicBool::new(false);
        let out = search(
            &space(),
            Knobs::default(),
            &mut Flaky(0),
            Config::default(),
            &cancel,
        );
        assert!(
            out.trials
                .iter()
                .any(|t| t.result.is_err() && t.score == 0.0)
        );
        assert!(out.best_score > 0.0);
    }

    #[test]
    fn score_penalises_swap_and_rejects_nonsense() {
        let ok = Measurement {
            tok_s: 8.0,
            swap_used_gb: 0.5,
        };
        let paging = Measurement {
            tok_s: 8.0,
            swap_used_gb: 5.0,
        };
        assert_eq!(score(&ok), 8.0);
        assert!((score(&paging) - 8.0 / 5.0).abs() < 1e-9);
        assert_eq!(
            score(&Measurement {
                tok_s: f64::NAN,
                swap_used_gb: 0.0
            }),
            0.0
        );
    }

    #[test]
    fn reads_the_last_speed_figure_even_with_colours() {
        let log = "warm 3.1 tok/s\n\u{1b}[32mdecode 12.75 tok/s\u{1b}[0m done";
        assert_eq!(parse_tok_s(log), Some(12.75));
        assert_eq!(parse_tok_s("nothing here"), None);
    }

    #[test]
    fn the_winner_is_remembered_per_machine_and_model() {
        let dir = std::env::temp_dir().join(format!("autotune-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let file = dir.join("autotune.json");
        let key = machine_key(15.7, 8, "CPU", "qwen36-35b");
        assert_eq!(key, "16gb-8c-cpu/qwen36-35b");
        let a = Knobs {
            pin_gb: Some(5.0),
            threads: Some(8),
            ctx: None,
        };
        save_profile(&file, &key, &a, 6.0).unwrap();
        assert_eq!(load_profile(&file, &key).unwrap().knobs, a);
        // a worse run does not overwrite
        save_profile(&file, &key, &Knobs::default(), 2.0).unwrap();
        assert_eq!(load_profile(&file, &key).unwrap().score, 6.0);
        assert!(load_profile(&file, "other").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    struct Scratch(std::path::PathBuf);
    impl Scratch {
        fn new(name: &str) -> Self {
            let d = std::env::temp_dir().join(format!("autotune-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            Scratch(d)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn knobs(pin: Option<f64>, threads: Option<u32>) -> Knobs {
        Knobs {
            pin_gb: pin,
            threads,
            ctx: None,
        }
    }

    #[test]
    fn values_colibri_could_not_use_never_reach_its_environment() {
        for bad in [f64::NAN, f64::INFINITY, -5.0, 0.0] {
            assert!(knobs(Some(bad), None).env().is_empty(), "{bad}");
        }
        let huge = knobs(Some(1e300), Some(u32::MAX)).env();
        assert!(
            huge.contains(&("PIN_GB".into(), "4096.0".into())),
            "{huge:?}"
        );
        assert!(huge.contains(&("OMP_NUM_THREADS".into(), "1024".into())));
        assert!(knobs(None, Some(0)).env().is_empty());
        let s = Space {
            free_ram_gb: f64::NAN,
            state_gb: f64::INFINITY,
            physical_cores: u32::MAX,
            contexts: vec![],
        };
        assert!(s.pin_options().iter().flatten().all(|g| g.is_finite()));
        assert!(s.thread_options().iter().flatten().all(|t| *t <= 1024));
    }

    #[test]
    fn a_speed_figure_that_is_not_a_measurement_is_skipped() {
        assert_eq!(parse_tok_s("-3.5 tok/s"), None);
        assert_eq!(
            parse_tok_s("5 tok/s then 99999999999999999999 tok/s"),
            Some(5.0)
        );
        assert_eq!(parse_tok_s("1.2.3 tok/s"), None);
        assert_eq!(parse_tok_s("tok/s tok/s \u{1F980}tok/s"), None);
        assert_eq!(parse_tok_s("\u{1b}[32m7.5 tok/s"), Some(7.5));
        assert_eq!(parse_tok_s(""), None);
    }

    #[test]
    fn a_run_cut_short_by_cancelling_is_not_taken_as_a_result() {
        struct Cuts<'a>(&'a AtomicBool);
        impl Runner for Cuts<'_> {
            fn run(
                &mut self,
                _: &Knobs,
                _: Duration,
                c: &AtomicBool,
            ) -> Result<Measurement, String> {
                // the user presses stop while this trial runs; the figure it returns is partial
                c.store(true, Ordering::Relaxed);
                self.0.store(true, Ordering::Relaxed);
                Ok(Measurement {
                    tok_s: 999.0,
                    swap_used_gb: 0.0,
                })
            }
        }
        let cancel = AtomicBool::new(false);
        let out = search(
            &space(),
            Knobs::default(),
            &mut Cuts(&cancel),
            Config::default(),
            &cancel,
        );
        assert!(out.aborted);
        assert_eq!(out.best_score, 0.0, "{out:?}");
        assert_eq!(out.best, Knobs::default());
        assert!(out.trials.iter().all(|t| t.score == 0.0));
    }

    #[test]
    fn when_every_trial_fails_the_search_still_ends_and_changes_nothing() {
        struct Dead(usize);
        impl Runner for Dead {
            fn run(
                &mut self,
                _: &Knobs,
                _: Duration,
                _: &AtomicBool,
            ) -> Result<Measurement, String> {
                self.0 += 1;
                Err("no server".into())
            }
        }
        let mut r = Dead(0);
        let out = search(
            &space(),
            Knobs::default(),
            &mut r,
            Config::default(),
            &AtomicBool::new(false),
        );
        assert_eq!(out.best, Knobs::default());
        assert_eq!(out.gain(), 0.0);
        assert!(r.0 <= 12 && !out.trials.is_empty());
    }

    #[test]
    fn nonsense_measurements_never_win() {
        struct Odd;
        impl Runner for Odd {
            fn run(
                &mut self,
                k: &Knobs,
                _: Duration,
                _: &AtomicBool,
            ) -> Result<Measurement, String> {
                let tok_s = match k.threads {
                    Some(4) => f64::INFINITY,
                    Some(7) => f64::NAN,
                    Some(_) => -1.0,
                    None => 3.0,
                };
                Ok(Measurement {
                    tok_s,
                    swap_used_gb: f64::NAN,
                })
            }
        }
        let out = search(
            &space(),
            Knobs::default(),
            &mut Odd,
            Config::default(),
            &AtomicBool::new(false),
        );
        assert_eq!(out.best.threads, None);
        assert!(out.best_score.is_finite());
    }

    #[test]
    fn the_total_time_and_the_trial_time_are_bounded() {
        struct Spy(Vec<Duration>);
        impl Runner for Spy {
            fn run(
                &mut self,
                _: &Knobs,
                b: Duration,
                _: &AtomicBool,
            ) -> Result<Measurement, String> {
                self.0.push(b);
                Ok(Measurement {
                    tok_s: 1.0,
                    swap_used_gb: 0.0,
                })
            }
        }
        let cfg = Config {
            max_runs: usize::MAX,
            budget: Duration::MAX,
            total: Duration::ZERO,
        };
        let mut spy = Spy(vec![]);
        let out = search(
            &space(),
            Knobs::default(),
            &mut spy,
            cfg,
            &AtomicBool::new(false),
        );
        assert!(out.timed_out && spy.0.is_empty());
        let cfg = Config {
            max_runs: usize::MAX,
            budget: Duration::MAX,
            total: Duration::from_secs(60),
        };
        let mut spy = Spy(vec![]);
        let out = search(
            &space(),
            Knobs::default(),
            &mut spy,
            cfg,
            &AtomicBool::new(false),
        );
        assert!(!spy.0.is_empty() && spy.0.iter().all(|b| *b <= Duration::from_secs(3600)));
        assert!(out.trials.len() <= MAX_RUNS);
    }

    #[test]
    fn a_bad_score_is_refused_and_does_not_corrupt_the_file() {
        let d = Scratch::new("bad score");
        let file = d.0.join("with space").join("autotune.json");
        save_profile(&file, "a", &knobs(Some(2.0), None), 4.0).unwrap();
        for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            assert!(save_profile(&file, "b", &knobs(None, None), bad).is_err());
        }
        assert_eq!(load_profile(&file, "a").unwrap().score, 4.0);
    }

    #[test]
    fn a_damaged_entry_costs_one_profile_not_all_and_a_damaged_file_is_kept_aside() {
        let d = Scratch::new("damaged");
        let file = d.0.join("autotune.json");
        std::fs::write(
            &file,
            r#"{"profiles":{"good":{"knobs":{"pin_gb":3.0,"threads":4,"ctx":null},"score":5.0,"saved_at":1},
                "nan":{"knobs":{"pin_gb":null,"threads":null,"ctx":null},"score":null,"saved_at":1},
                "neg":{"knobs":{"pin_gb":-9.0,"threads":0,"ctx":null},"score":-2.0,"saved_at":1}}}"#,
        )
        .unwrap();
        assert!(load_profile(&file, "good").is_some());
        assert!(load_profile(&file, "nan").is_none() && load_profile(&file, "neg").is_none());
        std::fs::write(&file, "{ not json").unwrap();
        assert!(load_profile(&file, "good").is_none());
        save_profile(&file, "x", &knobs(None, Some(2)), 1.5).unwrap();
        assert_eq!(load_profile(&file, "x").unwrap().score, 1.5);
        assert_eq!(
            std::fs::read_to_string(file.with_extension("json.bad")).unwrap(),
            "{ not json"
        );
    }

    #[test]
    fn an_edited_profile_cannot_smuggle_values_colibri_rejects() {
        let d = Scratch::new("edited");
        let file = d.0.join("autotune.json");
        std::fs::write(
            &file,
            r#"{"profiles":{"k":{"knobs":{"pin_gb":-9.0,"threads":0,"ctx":null},"score":2.0,"saved_at":1}}}"#,
        )
        .unwrap();
        let p = load_profile(&file, "k").unwrap();
        assert_eq!(p.knobs, Knobs::default());
    }

    #[test]
    fn saves_from_several_threads_all_land_and_leave_no_litter() {
        let d = Scratch::new("threads");
        let file = std::sync::Arc::new(d.0.join("autotune.json"));
        let hs: Vec<_> = (0..8)
            .map(|i| {
                let f = file.clone();
                std::thread::spawn(move || {
                    for r in 0..5 {
                        save_profile(
                            &f,
                            &format!("k{i}"),
                            &knobs(None, Some(1 + r)),
                            1.0 + f64::from(r),
                        )
                        .unwrap();
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        for i in 0..8 {
            let p = load_profile(&file, &format!("k{i}")).unwrap_or_else(|| panic!("k{i} lost"));
            assert_eq!(p.score, 5.0);
        }
        let left: Vec<_> = std::fs::read_dir(&d.0)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "autotune.json")
            .collect();
        assert!(left.is_empty(), "{left:?}");
    }

    #[test]
    fn the_machine_key_is_stable_and_total() {
        assert_eq!(
            machine_key(15.7, 8, "cpu", "m"),
            machine_key(15.69, 8, "cpu", "m")
        );
        assert_eq!(
            machine_key(15.2, 8, "cpu", "m"),
            machine_key(16.0, 8, "cpu", "m")
        );
        assert_ne!(
            machine_key(15.7, 8, "cpu", "m"),
            machine_key(31.6, 8, "cpu", "m")
        );
        assert_ne!(
            machine_key(15.7, 8, "cpu", "m"),
            machine_key(15.7, 8, "cuda", "m")
        );
        assert_eq!(
            machine_key(15.7, 8, "weird", "m"),
            machine_key(15.7, 8, "", "m")
        );
        let _ = machine_key(f64::NAN, 0, "", "");
        let _ = machine_key(f64::INFINITY, u32::MAX, "x", "é/ ü");
        let _ = machine_key(1e300, 1, "x", "m");
        assert_eq!(ram_tier(1000.0), 1024);
        assert_eq!(ram_tier(-3.0), 0);
    }

    struct Fixed(f64, usize);
    impl Runner for Fixed {
        fn run(&mut self, _: &Knobs, _: Duration, _: &AtomicBool) -> Result<Measurement, String> {
            self.1 += 1;
            Ok(Measurement {
                tok_s: self.0,
                swap_used_gb: 0.0,
            })
        }
    }

    #[test]
    fn a_lucky_score_does_not_block_honest_later_saves() {
        let d = Scratch::new("lucky");
        let file = d.0.join("autotune.json");
        let (k1, k2) = (knobs(Some(2.0), Some(4)), knobs(Some(3.0), Some(6)));
        save_profile(&file, "k", &k1, 12.0).unwrap();
        // a better-looking rival does not win while the stored figure is fresh and higher
        save_profile(&file, "k", &k2, 9.0).unwrap();
        assert_eq!(load_profile(&file, "k").unwrap().knobs, k1);
        // measuring the stored best again shows it was luck
        let mut r = Fixed(8.0, 0);
        let p = revalidate(
            &file,
            "k",
            &mut r,
            Duration::from_secs(5),
            &AtomicBool::new(false),
        )
        .unwrap()
        .unwrap();
        assert_eq!((p.knobs.clone(), p.score, r.1), (k1.clone(), 8.0, 1));
        save_profile(&file, "k", &k2, 9.0).unwrap();
        let p = load_profile(&file, "k").unwrap();
        assert_eq!((p.knobs, p.score), (k2, 9.0));
        assert!(p.history.len() >= 3 && p.history.len() <= HISTORY_LEN);
    }

    #[test]
    fn a_stale_result_is_replaced_and_the_history_is_bounded() {
        let d = Scratch::new("stale");
        let file = d.0.join("autotune.json");
        let (k1, k2) = (knobs(Some(2.0), None), knobs(Some(3.0), None));
        save_profile_at(&file, "k", &k1, 10.0, 1_000).unwrap();
        save_profile_at(&file, "k", &k2, 5.0, 2_000).unwrap();
        assert_eq!(load_profile(&file, "k").unwrap().knobs, k1);
        save_profile_at(&file, "k", &k2, 5.0, 1_000 + STALE_AFTER_SECS + 1).unwrap();
        assert_eq!(load_profile(&file, "k").unwrap().knobs, k2);
        // a clock that went backwards is not "stale"
        save_profile_at(&file, "k", &k1, 1.0, 5).unwrap();
        assert_eq!(load_profile(&file, "k").unwrap().knobs, k2);
        for i in 0..50 {
            save_profile_at(&file, "k", &k2, 5.0 + f64::from(i), 9_000_000).unwrap();
        }
        assert_eq!(load_profile(&file, "k").unwrap().history.len(), HISTORY_LEN);
    }

    #[test]
    fn revalidating_fails_safely() {
        let d = Scratch::new("reval");
        let file = d.0.join("autotune.json");
        let go = AtomicBool::new(false);
        let b = Duration::from_secs(1);
        assert!(
            revalidate(&file, "none", &mut Fixed(1.0, 0), b, &go)
                .unwrap()
                .is_none()
        );
        save_profile(&file, "k", &knobs(None, Some(2)), 4.0).unwrap();
        assert!(revalidate(&file, "k", &mut Fixed(f64::NAN, 0), b, &go).is_err());
        let stop = AtomicBool::new(true);
        let mut r = Fixed(9.0, 0);
        assert!(revalidate(&file, "k", &mut r, b, &stop).is_err() && r.1 == 0);
        assert_eq!(load_profile(&file, "k").unwrap().score, 4.0);
    }
}
