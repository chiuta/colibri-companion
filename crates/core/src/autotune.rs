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
use std::time::Duration;

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

impl Knobs {
    /// Environment for a run. Values are rounded so equal settings give equal text.
    pub fn env(&self) -> Vec<(String, String)> {
        let mut v = Vec::new();
        if let Some(gb) = self.pin_gb {
            v.push(("PIN".to_string(), "auto".to_string()));
            v.push(("PIN_GB".to_string(), format!("{gb:.1}")));
        }
        if let Some(t) = self.threads {
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
        let c = self.physical_cores.max(1);
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

/// The last "N tok/s" figure in colibri's output (ANSI colours tolerated).
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
            if let Ok(v) = num.parse::<f64>() {
                found = Some(v);
            }
            rest = &rest[i + 5..];
        }
    }
    found
}

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
    /// Time allowed for one trial.
    pub budget: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_runs: 12,
            budget: Duration::from_secs(120),
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

    let mut try_one = |k: &Knobs, trials: &mut Vec<Trial>, aborted: &mut bool| -> Option<f64> {
        if let Some(s) = seen.get(&k.key()) {
            return Some(*s);
        }
        if trials.len() >= cfg.max_runs {
            return None;
        }
        if cancel.load(Ordering::Relaxed) {
            *aborted = true;
            return None;
        }
        let result = runner.run(k, cfg.budget, cancel);
        let s = result.as_ref().map(score).unwrap_or(0.0);
        if cancel.load(Ordering::Relaxed) {
            *aborted = true;
        }
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
    }
}

// ------------------------------------------------------------------ memory of the winner

/// Identifies this computer and model; a different amount of RAM or core count is a different case.
pub fn machine_key(total_ram_gb: f64, physical_cores: u32, model: &str) -> String {
    format!(
        "{}gb-{}c/{}",
        total_ram_gb.round() as i64,
        physical_cores,
        model
    )
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub knobs: Knobs,
    pub score: f64,
    pub saved_at: u64,
}

#[derive(Default, Serialize, Deserialize)]
struct Book {
    profiles: HashMap<String, Profile>,
}

fn read_book(path: &Path) -> Book {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Remembers the winner for `key`. A worse score never replaces a better saved one.
pub fn save_profile(path: &Path, key: &str, knobs: &Knobs, score: f64) -> std::io::Result<()> {
    let mut book = read_book(path);
    if book.profiles.get(key).is_some_and(|p| p.score >= score) {
        return Ok(());
    }
    book.profiles.insert(
        key.to_string(),
        Profile {
            knobs: knobs.clone(),
            score,
            saved_at: crate::store::now_secs(),
        },
    );
    let text = serde_json::to_string_pretty(&book).map_err(std::io::Error::other)?;
    crate::store::write_atomic(path, text.as_bytes())
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
        let key = machine_key(15.7, 8, "qwen36-35b");
        assert_eq!(key, "16gb-8c/qwen36-35b");
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
}
