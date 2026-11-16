//! The real [`Runner`] for [`crate::autotune`]: it measures one setting by starting colibri with
//! that setting, asking the model for a short fixed answer, reading the speed and stopping colibri.
//!
//! **Not exercised against a real model.** Everything that does not need a process (order of the
//! steps, stopping after every outcome, timeouts, cancelling, the refusal to touch a server that was
//! already running) is covered by tests with a fake [`Host`]. [`ColiHost`] is the thin layer that
//! talks to the real `coli` and has been checked only against the documented command line.
//!
//! Safety rules: a server that is already running is never stopped or used (it may be doing the
//! user's work); a server this runner started is always stopped, after success, failure, timeout,
//! cancelling or a panic; if it cannot be stopped, no further trial is started.

use crate::autotune::{Knobs, Measurement, Runner};
use crate::chat::{ChatClient, ChatMessage, ChatRequest, Thinking};
use crate::coli::Coli;
use crate::http::CancelToken;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// The question every trial asks: the same text, so the figures are comparable.
pub const PROMPT: &str = "Write one paragraph, about eighty words, about why the sea is salty.";
const WARMUP_TOKENS: u32 = 16;
const MEASURE_TOKENS: u32 = 64;
/// Fewer tokens than this is not a speed measurement.
const MIN_TOKENS: u64 = 8;
const MAX_TRIAL: Duration = Duration::from_secs(3600);
const MAX_CTX: u32 = 1 << 22;

#[derive(Clone, Debug, PartialEq)]
pub struct Endpoint {
    pub base: String,
    pub model: String,
    pub pid: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Generation {
    pub tok_s: f64,
    pub tokens: u64,
}

/// What the runner needs from the outside world; a fake in tests, [`ColiHost`] for real.
pub trait Host {
    /// Is a colibri server loading or running (not started by this runner)?
    fn server_running(&mut self) -> Result<bool, String>;
    /// Sets the context length for the next start; 0 means colibri's automatic choice.
    fn set_ctx(&mut self, ctx: u32) -> Result<(), String>;
    /// Starts the server in the background with this extra environment.
    fn start(&mut self, env: &[(String, String)]) -> Result<(), String>;
    fn wait_ready(&mut self, deadline: Instant, cancel: &CancelToken) -> Result<Endpoint, String>;
    fn generate(
        &mut self,
        ep: &Endpoint,
        max_tokens: u32,
        deadline: Instant,
        cancel: &CancelToken,
    ) -> Result<Generation, String>;
    fn swap_used_gb(&mut self) -> f64;
    /// Stops the server and does not return `Ok` until it is really gone.
    fn stop(&mut self, ep: Option<&Endpoint>) -> Result<(), String>;
}

pub struct HostRunner<H: Host> {
    host: H,
    ctx: u32,
    /// A server of ours may be running (set before starting, cleared after a successful stop).
    running: bool,
    poisoned: Option<String>,
}

impl<H: Host> HostRunner<H> {
    pub fn new(host: H) -> Self {
        Self {
            host,
            ctx: 0,
            running: false,
            poisoned: None,
        }
    }

    /// The context length colibri's setting was last changed to (0: never changed, or automatic).
    /// The search leaves the last tried value in colibri's settings; the caller restores its own.
    pub fn ctx_now(&self) -> u32 {
        self.ctx
    }

    pub fn restore_ctx(&mut self, ctx: u32) -> Result<(), String> {
        if ctx != self.ctx {
            self.host.set_ctx(ctx)?;
            self.ctx = ctx;
        }
        Ok(())
    }

    pub fn host(&self) -> &H {
        &self.host
    }

    fn trial(
        &mut self,
        k: &Knobs,
        deadline: Instant,
        token: &CancelToken,
    ) -> Result<Measurement, String> {
        if self.host.server_running()? {
            return Err("a colibri server is already running; stop it before tuning".into());
        }
        let want = match k.ctx {
            None | Some(0) => 0,
            Some(n) if n <= MAX_CTX => n,
            Some(n) => return Err(format!("unusable context length {n}")),
        };
        if want != self.ctx {
            self.host.set_ctx(want)?;
            self.ctx = want;
        }
        self.running = true;
        let mut ep = None;
        let res = match self.host.start(&k.env()) {
            Err(e) => Err(e),
            Ok(()) => (|| {
                let e = self.host.wait_ready(deadline, token)?;
                ep = Some(e.clone());
                self.host.generate(&e, WARMUP_TOKENS, deadline, token)?;
                let g = self.host.generate(&e, MEASURE_TOKENS, deadline, token)?;
                Ok((g, self.host.swap_used_gb()))
            })(),
        };
        if let Err(e) = self.host.stop(ep.as_ref()) {
            self.poisoned = Some(format!("could not stop colibri after a trial: {e}"));
            return Err(self.poisoned.clone().unwrap_or_default());
        }
        self.running = false;
        let (g, swap) = res?;
        if token.is_cancelled() || Instant::now() > deadline {
            return Err("cancelled or out of time".into());
        }
        if !g.tok_s.is_finite() || g.tok_s <= 0.0 || g.tokens < MIN_TOKENS {
            return Err(format!(
                "no usable speed ({} tokens, {} tok/s)",
                g.tokens, g.tok_s
            ));
        }
        Ok(Measurement {
            tok_s: g.tok_s,
            swap_used_gb: if swap.is_finite() { swap.max(0.0) } else { 0.0 },
        })
    }
}

impl<H: Host> Runner for HostRunner<H> {
    fn run(
        &mut self,
        k: &Knobs,
        budget: Duration,
        cancel: &AtomicBool,
    ) -> Result<Measurement, String> {
        if let Some(p) = &self.poisoned {
            return Err(p.clone());
        }
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let deadline = Instant::now() + budget.min(MAX_TRIAL);
        let token = CancelToken::new();
        let done = AtomicBool::new(false);
        std::thread::scope(|s| {
            // wakes the blocked network calls when the user stops the search or time runs out
            s.spawn(|| {
                while !done.load(Ordering::Relaxed) {
                    if cancel.load(Ordering::Relaxed) || Instant::now() >= deadline {
                        token.cancel();
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
            });
            let r = self.trial(k, deadline, &token);
            done.store(true, Ordering::Relaxed);
            r
        })
    }
}

impl<H: Host> Drop for HostRunner<H> {
    fn drop(&mut self) {
        if self.running {
            let _ = self.host.stop(None);
        }
    }
}

// ------------------------------------------------------------------ the real host

/// Drives the real `coli`. The model folder and backend are the ones already in colibri's settings.
pub struct ColiHost {
    coli: Coli,
    sampler: crate::resources::Sampler,
}

impl ColiHost {
    pub fn new(coli: Coli) -> Self {
        Self {
            coli,
            sampler: crate::resources::Sampler::new(),
        }
    }

    fn state(&self) -> Result<crate::coli::StatusReport, String> {
        self.coli.status()
    }

    fn wait_stopped(&self, limit: Duration) -> bool {
        let t = Instant::now();
        while t.elapsed() < limit {
            if self.state().is_ok_and(|s| s.server_state() == "stopped") {
                return true;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        false
    }

    /// Ends a process and everything it started. Only a plausible process id is ever used.
    fn kill_tree(&mut self, pid: u32) {
        if !(2..=i32::MAX as u32).contains(&pid) {
            return;
        }
        let pids = self.sampler.tree_pids(pid);
        for p in pids.into_iter().rev() {
            if !(2..=i32::MAX as u32).contains(&p) || p == std::process::id() {
                continue;
            }
            #[cfg(unix)]
            let mut c = {
                let mut c = std::process::Command::new("kill");
                c.args(["-KILL", &p.to_string()]);
                c
            };
            #[cfg(windows)]
            let mut c = {
                let mut c =
                    std::process::Command::new(crate::proc::system_exe(r"System32\taskkill.exe"));
                c.args(["/PID", &p.to_string(), "/T", "/F"]);
                c
            };
            crate::proc::hide_window(&mut c);
            let _ = c
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
    }
}

impl Host for ColiHost {
    fn server_running(&mut self) -> Result<bool, String> {
        Ok(self.state()?.server_state() != "stopped")
    }

    fn set_ctx(&mut self, ctx: u32) -> Result<(), String> {
        let st = self.state()?;
        let setup = st.setup.ok_or("colibri is not set up")?;
        let dir = setup.model_dir.ok_or("colibri has no model folder")?;
        let log = crate::paths::data_dir().join("autotune-setup.log");
        let mut child = self
            .coli
            .spawn_use_dir(
                std::path::Path::new(&dir),
                setup.backend.as_deref(),
                (ctx > 0).then_some(ctx),
                &log,
            )
            .map_err(|e| e.to_string())?;
        let t = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(s)) if s.success() => return Ok(()),
                Ok(Some(_)) => {
                    return Err("coli setup --ctx failed (see autotune-setup.log)".into());
                }
                Ok(None) if t.elapsed() < Duration::from_secs(180) => {
                    std::thread::sleep(Duration::from_millis(200));
                }
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err("coli setup --ctx took too long".into());
                }
            }
        }
    }

    fn start(&mut self, env: &[(String, String)]) -> Result<(), String> {
        let mut c = self.coli.clone();
        c.env.extend(env.iter().cloned()); // later entries win
        c.start_server().map(|_| ())
    }

    fn wait_ready(&mut self, deadline: Instant, cancel: &CancelToken) -> Result<Endpoint, String> {
        let mut stopped_since: Option<Instant> = None;
        loop {
            if cancel.is_cancelled() || Instant::now() >= deadline {
                return Err("cancelled or out of time while loading".into());
            }
            if let Ok(st) = self.state() {
                let srv = st.server.clone().unwrap_or_default();
                match srv.state.as_str() {
                    "ready" if !srv.urls.openai_base_url.is_empty() => {
                        return Ok(Endpoint {
                            base: srv.urls.openai_base_url.clone(),
                            model: srv.model_id.or(srv.model).unwrap_or_default(),
                            pid: srv.pid,
                        });
                    }
                    "stopped" => {
                        let since = *stopped_since.get_or_insert_with(Instant::now);
                        if since.elapsed() > Duration::from_secs(15) {
                            return Err("the server stopped while loading".into());
                        }
                    }
                    _ => stopped_since = None,
                }
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    fn generate(
        &mut self,
        ep: &Endpoint,
        max_tokens: u32,
        _deadline: Instant,
        cancel: &CancelToken,
    ) -> Result<Generation, String> {
        let client = ChatClient::new(&ep.base, None);
        let req = ChatRequest {
            model: ep.model.clone(),
            messages: vec![ChatMessage::new("user", PROMPT)],
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: Some(max_tokens),
            thinking: Thinking::Off,
        };
        let s = client
            .stream_chat(&req, cancel, |_| {})
            .map_err(|e| e.to_string())?;
        if s.cancelled || cancel.is_cancelled() {
            return Err("cancelled or out of time".into());
        }
        let tokens = s.completion_tokens.unwrap_or(0);
        let tok_s = s
            .tokens_per_second()
            .or_else(|| {
                // colibri's own figure for its last answer
                self.state()
                    .ok()
                    .and_then(|st| st.server.and_then(|v| v.tokens_per_second))
            })
            .ok_or("the answer was too short to measure")?;
        Ok(Generation { tok_s, tokens })
    }

    fn swap_used_gb(&mut self) -> f64 {
        self.sampler.sample(&[], None).swap_used_gb
    }

    fn stop(&mut self, ep: Option<&Endpoint>) -> Result<(), String> {
        let pid = self
            .state()
            .ok()
            .and_then(|s| s.server.and_then(|v| v.pid))
            .or(ep.and_then(|e| e.pid));
        let _ = self.coli.stop_server();
        if self.wait_stopped(Duration::from_secs(30)) {
            return Ok(());
        }
        if let Some(p) = pid {
            self.kill_tree(p);
        }
        let _ = self.coli.stop_server();
        if self.wait_stopped(Duration::from_secs(15)) {
            Ok(())
        } else {
            Err("the server did not stop".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Script {
        calls: Vec<String>,
        running: bool,
        fail_start: bool,
        fail_stop: bool,
        fail_ready: bool,
        answer: Option<Generation>,
        cancel_during_generate: Option<Arc<AtomicBool>>,
        env_seen: Vec<Vec<(String, String)>>,
    }
    #[derive(Clone)]
    struct Fake(Arc<Mutex<Script>>);
    impl Fake {
        fn new() -> Self {
            let s = Script {
                answer: Some(Generation {
                    tok_s: 5.0,
                    tokens: 64,
                }),
                ..Default::default()
            };
            Fake(Arc::new(Mutex::new(s)))
        }
        fn calls(&self) -> Vec<String> {
            self.0.lock().unwrap().calls.clone()
        }
    }
    impl Host for Fake {
        fn server_running(&mut self) -> Result<bool, String> {
            let mut s = self.0.lock().unwrap();
            s.calls.push("running?".into());
            Ok(s.running)
        }
        fn set_ctx(&mut self, ctx: u32) -> Result<(), String> {
            self.0.lock().unwrap().calls.push(format!("ctx {ctx}"));
            Ok(())
        }
        fn start(&mut self, env: &[(String, String)]) -> Result<(), String> {
            let mut s = self.0.lock().unwrap();
            s.calls.push("start".into());
            s.env_seen.push(env.to_vec());
            if s.fail_start {
                Err("no python".into())
            } else {
                Ok(())
            }
        }
        fn wait_ready(&mut self, _: Instant, _: &CancelToken) -> Result<Endpoint, String> {
            let mut s = self.0.lock().unwrap();
            s.calls.push("ready".into());
            if s.fail_ready {
                return Err("never ready".into());
            }
            Ok(Endpoint {
                base: "http://127.0.0.1:1".into(),
                model: "m".into(),
                pid: Some(1234),
            })
        }
        fn generate(
            &mut self,
            _: &Endpoint,
            n: u32,
            _: Instant,
            _: &CancelToken,
        ) -> Result<Generation, String> {
            let s = self.0.lock().unwrap();
            let (flag, g) = (s.cancel_during_generate.clone(), s.answer);
            drop(s);
            self.0.lock().unwrap().calls.push(format!("gen {n}"));
            if let Some(f) = flag {
                f.store(true, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(200)); // the watchdog reacts
            }
            g.ok_or_else(|| "too short".to_string())
        }
        fn swap_used_gb(&mut self) -> f64 {
            0.25
        }
        fn stop(&mut self, _: Option<&Endpoint>) -> Result<(), String> {
            let mut s = self.0.lock().unwrap();
            s.calls.push("stop".into());
            if s.fail_stop {
                Err("stuck".into())
            } else {
                Ok(())
            }
        }
    }

    fn go(r: &mut HostRunner<Fake>, k: &Knobs) -> Result<Measurement, String> {
        r.run(k, Duration::from_secs(30), &AtomicBool::new(false))
    }

    #[test]
    fn a_trial_checks_starts_warms_up_measures_and_stops() {
        let f = Fake::new();
        let mut r = HostRunner::new(f.clone());
        let k = Knobs {
            pin_gb: Some(4.0),
            threads: Some(6),
            ctx: None,
        };
        let m = go(&mut r, &k).unwrap();
        assert_eq!((m.tok_s, m.swap_used_gb), (5.0, 0.25));
        assert_eq!(
            f.calls(),
            ["running?", "start", "ready", "gen 16", "gen 64", "stop"]
        );
        let env = f.0.lock().unwrap().env_seen[0].clone();
        assert!(env.contains(&("PIN_GB".into(), "4.0".into())));
        assert!(env.contains(&("OMP_NUM_THREADS".into(), "6".into())));
    }

    #[test]
    fn a_server_that_was_already_running_is_left_alone() {
        let f = Fake::new();
        f.0.lock().unwrap().running = true;
        let mut r = HostRunner::new(f.clone());
        assert!(
            go(&mut r, &Knobs::default())
                .unwrap_err()
                .contains("already")
        );
        assert_eq!(f.calls(), ["running?"]); // no start, and above all no stop
        drop(r);
        assert_eq!(f.calls(), ["running?"]);
    }

    #[test]
    fn the_server_is_stopped_after_every_kind_of_failure() {
        for (start, ready, answer) in [
            (true, false, true),
            (false, true, true),
            (false, false, false),
        ] {
            let f = Fake::new();
            {
                let mut s = f.0.lock().unwrap();
                s.fail_start = start;
                s.fail_ready = ready;
                if !answer {
                    s.answer = None;
                }
            }
            let mut r = HostRunner::new(f.clone());
            assert!(go(&mut r, &Knobs::default()).is_err());
            assert_eq!(
                f.calls().last().map(String::as_str),
                Some("stop"),
                "{start} {ready}"
            );
            // the runner is still usable afterwards
            f.0.lock().unwrap().fail_start = false;
            f.0.lock().unwrap().fail_ready = false;
            f.0.lock().unwrap().answer = Some(Generation {
                tok_s: 2.0,
                tokens: 64,
            });
            assert!(go(&mut r, &Knobs::default()).is_ok());
        }
    }

    #[test]
    fn if_the_server_cannot_be_stopped_nothing_else_is_started() {
        let f = Fake::new();
        f.0.lock().unwrap().fail_stop = true;
        let mut r = HostRunner::new(f.clone());
        assert!(go(&mut r, &Knobs::default()).is_err());
        let n = f.calls().len();
        assert!(go(&mut r, &Knobs::default()).unwrap_err().contains("stop"));
        assert_eq!(f.calls().len(), n, "no call after poisoning");
    }

    #[test]
    fn the_context_is_changed_only_when_it_differs() {
        let f = Fake::new();
        let mut r = HostRunner::new(f.clone());
        let with = |c| Knobs {
            ctx: c,
            ..Knobs::default()
        };
        for c in [None, Some(8192), Some(8192), Some(0), None] {
            go(&mut r, &with(c)).unwrap();
        }
        let ctxs: Vec<_> = f
            .calls()
            .into_iter()
            .filter(|c| c.starts_with("ctx"))
            .collect();
        assert_eq!(ctxs, ["ctx 8192", "ctx 0"]);
        assert!(go(&mut r, &with(Some(u32::MAX))).is_err());
        assert_eq!(r.ctx_now(), 0);
    }

    #[test]
    fn a_stop_pressed_during_a_trial_gives_no_result_and_stops_the_server() {
        let f = Fake::new();
        let flag = Arc::new(AtomicBool::new(false));
        f.0.lock().unwrap().cancel_during_generate = Some(flag.clone());
        let mut r = HostRunner::new(f.clone());
        let e = r
            .run(&Knobs::default(), Duration::from_secs(30), &flag)
            .unwrap_err();
        assert!(e.contains("cancel"), "{e}");
        assert_eq!(f.calls().last().map(String::as_str), Some("stop"));
        // already cancelled: not even started
        let n = f.calls().len();
        assert!(
            r.run(&Knobs::default(), Duration::from_secs(1), &flag)
                .is_err()
        );
        assert_eq!(f.calls().len(), n);
    }

    #[test]
    fn running_out_of_time_is_not_a_result() {
        let f = Fake::new();
        let mut r = HostRunner::new(f.clone());
        // a zero budget: the watchdog cancels at once, so whatever the host returns is discarded
        let e = r
            .run(&Knobs::default(), Duration::ZERO, &AtomicBool::new(false))
            .unwrap_err();
        assert!(e.contains("cancel") || e.contains("time"), "{e}");
        assert_eq!(f.calls().last().map(String::as_str), Some("stop"));
    }

    #[test]
    fn nonsense_speeds_are_refused() {
        for g in [
            Generation {
                tok_s: f64::NAN,
                tokens: 64,
            },
            Generation {
                tok_s: 0.0,
                tokens: 64,
            },
            Generation {
                tok_s: -1.0,
                tokens: 64,
            },
            Generation {
                tok_s: 9.0,
                tokens: 2,
            },
        ] {
            let f = Fake::new();
            f.0.lock().unwrap().answer = Some(g);
            let mut r = HostRunner::new(f);
            assert!(go(&mut r, &Knobs::default()).is_err(), "{g:?}");
        }
    }

    #[test]
    fn dropping_a_runner_that_left_a_server_behind_stops_it() {
        let f = Fake::new();
        let mut r = HostRunner::new(f.clone());
        r.running = true; // as after a panic inside the host
        drop(r);
        assert_eq!(f.calls(), ["stop"]);
    }
}
