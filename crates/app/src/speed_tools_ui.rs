//! Three speed tools on top of the finished core modules: the "will it be fast?" estimate, the
//! routing-trace recording with its analysis, and the per-computer settings search.
//!
//! Nothing here runs against a real model unless the user presses a button; every long job runs on
//! a worker thread and reports back through [`Event`]. The pure parts (number formatting, the
//! recording state machine, the summaries) are at the top and are unit-tested.

use crate::app::{CompanionApp, Event, ServerAction};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::advisor;
use companion_core::autotune::{self, Config, Knobs, Measurement, Runner, Space};
use companion_core::autotune_runner::{ColiHost, Host, HostRunner};
use companion_core::capture::{self, Report};
use companion_core::coli::ModelInfo;
use companion_core::estimate::{self, Bottleneck, Drive, Estimate, ModelFacts, Speed};
use companion_core::paths;
use egui::{RichText, Ui};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

/// Context length the estimate assumes (the planner's usual default for chat).
pub const ESTIMATE_CTX: u32 = 8192;
/// The recording stops growing its analysis at this size (the core's own default cap).
const CAP_BYTES: u64 = capture::DEFAULT_CAP_BYTES;
/// Texts the search thread can fail with; `engine_text` translates them where they are shown.
const MSG_NOT_STOPPED: &str = "The model did not stop, so the search was not started.";
const MSG_PANIC: &str = "The settings search stopped unexpectedly; the model was started again.";
const MSG_CTX_PREFIX: &str = "Could not restore the context length: {}";
/// How often the recording file is looked at.
const POLL: Duration = Duration::from_secs(2);

// ------------------------------------------------------------------ pure view-model helpers

fn grid(v: f64) -> f64 {
    if v >= 10.0 {
        1.0
    } else if v >= 1.0 {
        0.1
    } else {
        0.01
    }
}

fn fmt_one(v: f64) -> String {
    if v >= 10.0 {
        format!("{v:.0}")
    } else if v >= 1.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// A range of tokens per second as text, always two different figures ("0.3–1.2"): the low end
/// rounded down and the high end rounded up, so the text never claims more precision than the
/// estimate has. A range that is not usable (NaN, infinite, not positive) is a dash.
pub fn fmt_rate_range(low: f64, high: f64) -> String {
    if !(low.is_finite() && high.is_finite()) || high <= 0.0 {
        return "—".to_string();
    }
    let (lo, hi) = if low <= high {
        (low, high)
    } else {
        (high, low)
    };
    let lo = lo.max(0.0);
    let a = (((lo / grid(lo)) + 1e-9).floor() * grid(lo)).max(0.01);
    let mut b = (((hi / grid(hi)) - 1e-9).ceil() * grid(hi)).max(a + 0.01);
    if fmt_one(a) == fmt_one(b) {
        b += grid(b);
    }
    format!("{}–{}", fmt_one(a), fmt_one(b))
}

/// A share (0..1) as a whole percent, or `None` when it is not a number.
pub fn share(x: f64) -> Option<String> {
    x.is_finite()
        .then(|| format!("{:.0}%", 100.0 * x.clamp(0.0, 1.0)))
}

/// The drive input of the estimate: the measured speed when the drive test has run.
pub fn drive_of(test: Option<&Result<f64, String>>) -> Drive {
    match test {
        Some(Ok(mb)) if mb.is_finite() && *mb > 0.0 => Drive::Measured(*mb),
        _ => Drive::Unknown,
    }
}

/// The estimate for a model on a computer with `ram_gb` of memory. Without a usable memory
/// reading there is no estimate at all: guessing zero would call every model too slow.
pub fn estimate_at(ram_gb: f64, drive: Drive, m: &ModelInfo) -> Option<Estimate> {
    (ram_gb.is_finite() && ram_gb > 0.0)
        .then(|| estimate::estimate(&ModelFacts::from(m), ram_gb, drive, ESTIMATE_CTX))
}

/// Where a routing recording is.
#[derive(Debug, Clone, Default)]
pub enum CaptureState {
    #[default]
    Idle,
    /// Asked for; the file appears when the model next starts.
    Armed,
    Recording,
    Analysing,
    Done(Box<Report>),
    Failed(String),
}

impl CaptureState {
    /// True while the model must be started with the recording switched on.
    pub fn is_on(&self) -> bool {
        matches!(self, Self::Armed | Self::Recording)
    }
}

/// What the periodic look at the recording file decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureStep {
    Stay,
    /// The file exists: recording has really begun.
    Begin,
    /// The file reached its size cap: stop and analyse.
    Full,
}

/// `len` is the size of the trace file now (`None`: no such file yet). `stale` is true while the
/// file is one left from before the recording was asked for (it could not be removed because the
/// model still holds it open): until the model restarts it says nothing about the new recording.
pub fn capture_step(state: &CaptureState, len: Option<u64>, cap: u64, stale: bool) -> CaptureStep {
    match (state, len) {
        (CaptureState::Armed, Some(_)) if stale => CaptureStep::Stay,
        (CaptureState::Armed | CaptureState::Recording, Some(n)) if n >= cap => CaptureStep::Full,
        (CaptureState::Armed, Some(_)) => CaptureStep::Begin,
        _ => CaptureStep::Stay,
    }
}

/// The report as plain English text, for pasting into a message or an issue. NaN figures read
/// "no data".
pub fn report_text(r: &Report) -> String {
    let pct = |x: f64| share(x).unwrap_or_else(|| "no data".into());
    let mut s = String::new();
    s.push_str(&format!(
        "Routing recording: {} tokens analysed, {} layers, {} experts per layer.\n",
        r.analysed_tokens, r.summary.layers, r.n_experts
    ));
    s.push_str("\nCache (share of all experts) | LRU | hottest pinned | optimal ceiling\n");
    for c in &r.cache {
        s.push_str(&format!(
            "{}% | {} | {} | {}\n",
            c.cache_pct,
            pct(c.lru),
            pct(c.pinned),
            pct(c.optimal)
        ));
    }
    if !r.prefetch.is_empty() {
        s.push_str("\nNext-layer prediction (budget x the experts a layer uses): predicted | popularity alone\n");
        for p in &r.prefetch {
            s.push_str(&format!(
                "x{} | {} | {}\n",
                p.mult,
                pct(p.predicted),
                pct(p.popularity_only)
            ));
        }
    }
    if !r.speculative.is_empty() {
        s.push_str(
            "\nVerifying k drafted tokens per pass: acceptance | expert loads per accepted token\n",
        );
        for p in &r.speculative {
            let loads = if p.loads_per_token.is_finite() {
                format!("{:.0}", p.loads_per_token)
            } else {
                "no data".into()
            };
            s.push_str(&format!("k={} | {} | {}\n", p.k, pct(p.acceptance), loads));
        }
    }
    s.push('\n');
    for v in &r.verdicts {
        s.push_str(&format!("- {v}\n"));
    }
    s
}

/// Settings as `NAME=value` text, for showing what is applied.
pub fn knobs_text(k: &Knobs) -> String {
    let env = k.env();
    let mut parts: Vec<String> = env
        .into_iter()
        .filter(|(n, _)| n != "PIN")
        .map(|(n, v)| format!("{n}={v}"))
        .collect();
    if let Some(c) = k.ctx_arg() {
        parts.push(format!("--ctx {c}"));
    }
    if parts.is_empty() {
        "—".to_string()
    } else {
        parts.join(", ")
    }
}

/// What the search found, in numbers the screen can show.
#[derive(Debug, Clone, PartialEq)]
pub struct TuneSummary {
    pub best: Knobs,
    /// Relative gain over the automatic settings (0.12 = 12% faster), 0 when none.
    pub gain: f64,
    pub best_tok_s: f64,
    pub trials: usize,
    pub ok_trials: usize,
    pub aborted: bool,
    pub timed_out: bool,
    pub saved: bool,
    pub first_error: Option<String>,
}

pub fn summarize_outcome(o: &autotune::Outcome, saved: bool) -> TuneSummary {
    let ok_trials = o.trials.iter().filter(|t| t.result.is_ok()).count();
    let first_error = o
        .trials
        .iter()
        .find_map(|t| t.result.as_ref().err().cloned());
    let best_tok_s = o
        .trials
        .iter()
        .filter_map(|t| t.result.as_ref().ok())
        .map(|m| m.tok_s)
        .fold(0.0, f64::max);
    TuneSummary {
        best: o.best.clone(),
        gain: o.gain(),
        best_tok_s,
        trials: o.trials.len(),
        ok_trials,
        aborted: o.aborted,
        timed_out: o.timed_out,
        saved,
        first_error,
    }
}

/// Where the settings search is.
#[derive(Debug, Clone, Default)]
pub enum TuneUi {
    #[default]
    Idle,
    /// The warning is on screen, waiting for a decision.
    Confirm,
    Running {
        done: usize,
        max: usize,
        best: Option<f64>,
        cancelling: bool,
    },
    Done(TuneSummary),
    Failed(String),
}

impl TuneUi {
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }
}

/// Everything these tools keep between frames.
pub struct ToolsState {
    pub capture: CaptureState,
    /// The recording may still be written by the model (until it restarts).
    pub writing: bool,
    pub last_poll: Option<Instant>,
    pub bytes_now: u64,
    pub tune: TuneUi,
    pub cancel: Arc<AtomicBool>,
    /// Start the model with the settings the search found.
    pub apply_tuned: bool,
    /// The recording file was still there (held open by the model) when a new recording was asked
    /// for; it is ignored until the model restarts.
    pub stale_file: bool,
    /// Set by the search thread when it has finished for good (also after a panic).
    pub worker_done: Arc<AtomicBool>,
    /// The saved settings for this computer, read from disk now and then, not on every frame.
    pub tuned_cache: Option<(Instant, Option<String>, Option<Knobs>)>,
}

impl Default for ToolsState {
    fn default() -> Self {
        Self {
            capture: CaptureState::Idle,
            writing: false,
            last_poll: None,
            bytes_now: 0,
            tune: TuneUi::Idle,
            cancel: Arc::new(AtomicBool::new(false)),
            apply_tuned: true,
            stale_file: false,
            worker_done: Arc::new(AtomicBool::new(true)),
            tuned_cache: None,
        }
    }
}

fn trace_file() -> PathBuf {
    paths::data_dir().join("route-trace.txt")
}

fn ctx_file() -> PathBuf {
    paths::data_dir().join("context-tokens.txt")
}

/// The context length the user chose through the app, when there is one.
pub fn saved_ctx() -> Option<u32> {
    std::fs::read_to_string(ctx_file())
        .ok()?
        .trim()
        .parse()
        .ok()
        .filter(|n| *n > 0)
}

pub fn save_ctx(n: u32) {
    let _ = std::fs::write(ctx_file(), n.to_string());
}

fn mb(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
}

/// Wraps the real runner to report progress after every trial.
struct ProgressRunner<H: Host> {
    inner: HostRunner<H>,
    tx: Sender<Event>,
    ui: egui::Context,
    done: usize,
    max: usize,
    best: f64,
}

impl<H: Host> Runner for ProgressRunner<H> {
    fn run(
        &mut self,
        k: &Knobs,
        budget: Duration,
        cancel: &AtomicBool,
    ) -> Result<Measurement, String> {
        let r = self.inner.run(k, budget, cancel);
        self.done += 1;
        if let Ok(m) = &r {
            self.best = self.best.max(m.tok_s);
        }
        let _ = self.tx.send(Event::AutoTuneProgress {
            done: self.done,
            max: self.max,
            best: (self.best > 0.0).then_some(self.best),
        });
        self.ui.request_repaint();
        r
    }
}

// ------------------------------------------------------------------ the app side

impl CompanionApp {
    fn known_ram_gb(&self) -> f64 {
        let live = self.res.as_ref().map_or(0.0, |v| v.snap.ram_total_gb);
        if live > 0.0 {
            live
        } else {
            self.hardware.ready().map_or(0.0, |h| h.ram_total_gb)
        }
    }

    /// The catalogue entry of the model colibri is set up for, when known.
    fn active_model_info(&self) -> Option<ModelInfo> {
        let id = self.active_model_id()?;
        self.catalog.ready()?.iter().find(|m| m.id == id).cloned()
    }

    pub fn active_model_id(&self) -> Option<String> {
        let st = self.status.as_ref()?;
        st.server
            .as_ref()
            .and_then(|s| s.model_id.clone())
            .or_else(|| st.setup.as_ref()?.model.as_ref()?.id.clone())
            .or_else(|| self.model_id.clone())
    }

    fn estimate_for(&self, m: &ModelInfo) -> Option<Estimate> {
        estimate_at(self.known_ram_gb(), drive_of(self.disk_speed.as_ref()), m)
    }

    fn speed_class(&self, s: Speed) -> &'static str {
        match s {
            Speed::Fast => self.t("Rapid", "Fast"),
            Speed::Usable => self.t("Utilizabil", "Usable"),
            Speed::Slow => self.t("Lent", "Slow"),
            Speed::TooSlow => self.t("Prea lent", "Too slow to be useful"),
        }
    }

    fn speed_colour(&self, s: Speed) -> egui::Color32 {
        match s {
            Speed::Fast | Speed::Usable => self.p.accent,
            Speed::Slow => self.p.warn,
            Speed::TooSlow => self.p.bad,
        }
    }

    fn estimate_headline(&self, e: &Estimate) -> String {
        self.t(
            "Viteză estimată: ≈ {} jetoane/s · {}",
            "Estimated speed: ≈ {} tokens/s · {}",
        )
        .replacen(
            "{}",
            &crate::widgets::loc(fmt_rate_range(e.tok_s_low, e.tok_s_high)),
            1,
        )
        .replacen("{}", self.speed_class(e.speed), 1)
    }

    /// The line under a catalogue card, before anything is downloaded.
    pub fn estimate_card_line(&mut self, ui: &mut Ui, m: &ModelInfo) {
        if !m.is_chat_model() || m.status == "unsupported" {
            return;
        }
        let Some(e) = self.estimate_for(m) else {
            return;
        };
        ui.label(
            RichText::new(self.estimate_headline(&e))
                .color(self.speed_colour(e.speed))
                .semibold(),
        );
        if e.drive_assumed && e.bottleneck != Bottleneck::RamResident {
            ui.horizontal_wrapped(|ui| {
                let p = self.p;
                muted(
                    ui,
                    &p,
                    self.t(
                        "Viteza discului nu este măsurată: s-a presupus un SSD obișnuit, deci intervalul este larg.",
                        "The drive speed is not measured: a typical SSD is assumed, so the range is wide.",
                    ),
                );
                if !self.disk_busy && ui.link(self.t("Testează discul", "Test the drive")).clicked()
                {
                    self.measure_disk();
                }
            });
        }
    }

    /// The estimate for the model in use, in the speed section of the resources window.
    fn estimate_block(&mut self, ui: &mut Ui) {
        let p = self.p;
        let Some(m) = self.active_model_info() else {
            return;
        };
        let Some(e) = self.estimate_for(&m) else {
            return;
        };
        ui.add_space(4.0);
        ui.label(
            RichText::new(self.estimate_headline(&e))
                .color(self.speed_colour(e.speed))
                .semibold(),
        );
        let why = match e.bottleneck {
            Bottleneck::RamResident => self.t(
                "Totul încape în memorie: viteza memoriei stabilește ritmul.",
                "Everything fits in memory: memory speed sets the pace.",
            ),
            Bottleneck::SsdBound => self.t(
                "Experții se citesc de pe disc: discul stabilește ritmul.",
                "Experts are read from the drive: the drive sets the pace.",
            ),
            Bottleneck::SwapRisk => self.t(
                "Nici partea necesară permanent nu încape în memorie: sistemul ar folosi swap.",
                "Not even the always-needed part fits in memory: the system would swap.",
            ),
        };
        muted(ui, &p, why);
        if e.drive_assumed && e.bottleneck != Bottleneck::RamResident {
            muted(
                ui,
                &p,
                self.t(
                    "Viteza discului nu este măsurată: s-a presupus un SSD obișnuit, deci intervalul este larg.",
                    "The drive speed is not measured: a typical SSD is assumed, so the range is wide.",
                ),
            );
            if !self.disk_busy
                && ui
                    .button(self.t("Testează discul", "Test the drive"))
                    .clicked()
            {
                self.measure_disk();
            }
        }
        muted(
            ui,
            &p,
            self.t(
                "Estimare din mărimea memoriei și viteza discului, nu o măsurătoare.",
                "An estimate from the memory size and the drive speed, not a measurement.",
            ),
        );
    }

    // -------------------------------------------------------------- the tuned settings

    /// The key that identifies this computer, backend and model for the saved settings.
    fn tune_key(&self) -> Option<String> {
        let ram = self.known_ram_gb();
        let model = self.active_model_id()?;
        let cores = self.hardware.ready().map(|h| h.cores)?;
        if ram <= 0.0 || cores == 0 {
            return None;
        }
        let backend = self
            .status
            .as_ref()
            .and_then(|st| st.server.as_ref()?.backend.clone())
            .or_else(|| self.status.as_ref()?.setup.as_ref()?.backend.clone())
            .unwrap_or_else(|| self.s.backend.as_arg().unwrap_or("auto").to_string());
        Some(autotune::machine_key(ram, cores, &backend, &model))
    }

    /// The saved result of the search for this computer, when it is recent enough to trust.
    pub fn tuned_knobs(&self) -> Option<Knobs> {
        let key = self.tune_key()?;
        let prof = autotune::load_profile(&autotune::profiles_file(), &key)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        (!prof.is_stale(now)).then_some(prof.knobs)
    }

    /// [`Self::tuned_knobs`] for drawing: the profile file is read at most every few seconds, not
    /// on every frame.
    fn tuned_knobs_cached(&mut self) -> Option<Knobs> {
        let key = self.tune_key();
        if let Some((at, k, v)) = &self.tools.tuned_cache {
            if *k == key && at.elapsed() < Duration::from_secs(5) {
                return v.clone();
            }
        }
        let v = self.tuned_knobs();
        self.tools.tuned_cache = Some((Instant::now(), key, v.clone()));
        v
    }

    /// The extra environment these tools add when the model starts: the recording switch and the
    /// settings the search found.
    pub fn tools_env(&self) -> Vec<(String, String)> {
        let mut env = Vec::new();
        if self.tools.capture.is_on() {
            env.extend(capture::capture_env(&trace_file()));
        }
        if self.tools.apply_tuned && !self.tools.tune.is_running() {
            if let Some(k) = self.tuned_knobs() {
                env.extend(k.env());
            }
        }
        env
    }

    // -------------------------------------------------------------- the recording

    fn restart_model(&mut self) {
        let Some(mut coli) = self.coli.clone() else {
            return;
        };
        if self.chat.generating.is_some() || self.tune_blocks_server() {
            return;
        }
        coli.env = self.accel_env();
        self.starting_since = Some(Instant::now());
        self.busy = Some(self.t("Repornesc modelul…", "Restarting the model…").into());
        self.spawn(move |tx| {
            let _ = coli.stop_server();
            let result = coli.start_server().map(|_| ());
            let _ = tx.send(Event::ServerAction {
                what: ServerAction::Start,
                result,
            });
        });
    }

    /// True (with a note on screen) while the settings search owns the model: nothing else may
    /// stop, start or restart it meanwhile.
    pub fn tune_blocks_server(&mut self) -> bool {
        if !self.tools.tune.is_running() {
            return false;
        }
        let text = self.t(
            "Căutarea de setări rulează: oprește-o mai întâi.",
            "The settings search is running: stop it first.",
        );
        self.toast(text);
        true
    }

    fn arm_capture(&mut self) {
        let _ = std::fs::remove_file(trace_file());
        // on Windows a file the model holds open cannot be removed: it must not pass for the
        // new recording
        self.tools.stale_file = trace_file().exists();
        self.tools.bytes_now = 0;
        self.tools.capture = CaptureState::Armed;
    }

    fn analyse_capture(&mut self) {
        self.tools.capture = CaptureState::Analysing;
        let path = trace_file();
        self.spawn(move |tx| {
            let r = capture::analyze_file(&path, None, CAP_BYTES).map(Box::new);
            let _ = tx.send(Event::Captured(r));
        });
    }

    fn delete_capture(&mut self) {
        let _ = std::fs::remove_file(trace_file());
        self.tools.capture = CaptureState::Idle;
        self.tools.bytes_now = 0;
    }

    pub fn captured(&mut self, r: Result<Box<Report>, String>) {
        self.tools.capture = match r {
            Ok(rep) => CaptureState::Done(rep),
            Err(e) => CaptureState::Failed(e),
        };
    }

    /// Called every frame: looks at the recording file now and then.
    pub fn tools_tick(&mut self) {
        if !self.tools.capture.is_on() {
            return;
        }
        self.ctx.request_repaint_after(POLL);
        if self.tools.last_poll.is_some_and(|t| t.elapsed() < POLL) {
            return;
        }
        self.tools.last_poll = Some(Instant::now());
        let len = std::fs::metadata(trace_file()).ok().map(|m| m.len());
        self.tools.bytes_now = len.unwrap_or(0);
        match capture_step(&self.tools.capture, len, CAP_BYTES, self.tools.stale_file) {
            CaptureStep::Stay => {}
            CaptureStep::Begin => {
                self.tools.capture = CaptureState::Recording;
                self.tools.writing = true;
            }
            CaptureStep::Full => {
                self.tools.writing = true;
                self.analyse_capture();
            }
        }
    }

    /// The model has just been restarted or stopped: it no longer writes unless a recording is on.
    pub fn tools_server_changed(&mut self) {
        if !self.tools.capture.is_on() {
            self.tools.writing = false;
        }
        // a restart makes the model create the file afresh
        self.tools.stale_file = false;
    }

    pub fn remember_context(&mut self, n: u32) {
        save_ctx(n);
    }

    fn capture_section(&mut self, ui: &mut Ui) {
        let p = self.p;
        ui.add_space(10.0);
        ui.label(
            RichText::new(self.t(
                "Înregistrează rutarea experților",
                "Record the expert routing",
            ))
            .size(15.0)
            .semibold(),
        );
        muted(
            ui,
            &p,
            self.t(
                "Colibri notează ce experți alege pentru fiecare jeton; aplicația îi analizează local și arată ce ar ajuta cu adevărat pe calculatorul tău.",
                "Colibri writes down which experts it picks for every token; the app analyses them locally and shows what would really help on your computer.",
            ),
        );
        let generating = self.chat.generating.is_some();
        match self.tools.capture.clone() {
            CaptureState::Idle => {
                let ready = self.coli.is_some()
                    && self.active_model_dir().is_some()
                    && !self.tools.tune.is_running();
                if ui
                    .add_enabled(
                        ready,
                        egui::Button::new(self.t("Începe înregistrarea", "Start recording")),
                    )
                    .clicked()
                {
                    self.arm_capture();
                }
                if !ready && !self.tools.tune.is_running() {
                    muted(
                        ui,
                        &p,
                        self.t("Mai întâi configurează un model.", "Set up a model first."),
                    );
                }
            }
            CaptureState::Armed => {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Înregistrarea începe la următoarea pornire a modelului.",
                        "Recording starts the next time the model starts.",
                    ),
                );
                ui.horizontal(|ui| {
                    if self.server_ready()
                        && ui
                            .add_enabled(
                                !generating,
                                egui::Button::new(
                                    self.t("Repornește modelul acum", "Restart the model now"),
                                ),
                            )
                            .clicked()
                    {
                        self.restart_model();
                    }
                    if ui.button(self.t("Renunț", "Cancel")).clicked() {
                        self.delete_capture();
                    }
                });
                if generating && self.server_ready() {
                    muted(
                        ui,
                        &p,
                        self.t(
                            "Așteaptă sfârșitul răspunsului curent.",
                            "Wait for the current answer to finish.",
                        ),
                    );
                }
            }
            CaptureState::Recording => {
                ui.label(
                    self.t(
                        "Se înregistrează: {} din {}. Folosește modelul ca de obicei, apoi analizează.",
                        "Recording: {} of {}. Use the model as usual, then analyse.",
                    )
                    .replacen("{}", &mb(self.tools.bytes_now), 1)
                    .replacen("{}", &mb(CAP_BYTES), 1),
                );
                ui.horizontal(|ui| {
                    if ui
                        .button(self.t("Analizează acum", "Analyse now"))
                        .clicked()
                    {
                        self.analyse_capture();
                    }
                    if ui
                        .button(self.t("Oprește fără analiză", "Stop without analysing"))
                        .clicked()
                    {
                        self.delete_capture();
                    }
                });
            }
            CaptureState::Analysing => {
                ui.horizontal(|ui| {
                    crate::widgets::spinner(ui);
                    ui.label(self.t("Se analizează înregistrarea…", "Analysing the recording…"));
                });
            }
            CaptureState::Failed(e) => {
                ui.label(
                    RichText::new(
                        self.t("Analiza a eșuat: {}", "The analysis failed: {}")
                            .replacen("{}", &e, 1),
                    )
                    .color(p.bad),
                );
                if ui
                    .button(self.t("Înregistrează din nou", "Record again"))
                    .clicked()
                {
                    self.delete_capture();
                }
            }
            CaptureState::Done(rep) => {
                self.report_view(ui, &rep);
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui
                        .button(self.t(
                            "Copiază raportul (în engleză)",
                            "Copy the report (in English)",
                        ))
                        .clicked()
                    {
                        ui.ctx().copy_text(report_text(&rep));
                    }
                    if ui
                        .button(self.t("Șterge înregistrarea", "Delete the recording"))
                        .clicked()
                    {
                        self.delete_capture();
                    }
                    if ui
                        .button(self.t("Înregistrează din nou", "Record again"))
                        .clicked()
                    {
                        self.arm_capture();
                    }
                });
            }
        }
        if self.tools.writing && !self.tools.capture.is_on() && self.server_ready() {
            ui.add_space(4.0);
            muted(
                ui,
                &p,
                self.t(
                    "Modelul continuă să scrie în fișier până la repornire.",
                    "The model keeps writing to the file until it is restarted.",
                ),
            );
            if ui
                .add_enabled(
                    !generating,
                    egui::Button::new(self.t("Repornește modelul acum", "Restart the model now")),
                )
                .clicked()
            {
                self.restart_model();
            }
        }
    }

    fn report_view(&self, ui: &mut Ui, r: &Report) {
        let p = self.p;
        ui.add_space(4.0);
        ui.label(
            RichText::new(self.t("Ce arată înregistrarea", "What the recording shows")).semibold(),
        );
        let none = self.t("fără date", "no data");
        let cell = |x: f64| share(x).unwrap_or_else(|| none.to_string());
        if !r.cache.is_empty() {
            egui::Grid::new("capture_cache")
                .num_columns(4)
                .show(ui, |ui| {
                    ui.label(RichText::new(self.t("Cache", "Cache")).semibold());
                    ui.label(RichText::new("LRU").semibold());
                    ui.label(
                        RichText::new(self.t("Cei mai folosiți, fixați", "Hottest pinned"))
                            .semibold(),
                    );
                    ui.label(
                        RichText::new(self.t("Optim (limită)", "Optimal (ceiling)")).semibold(),
                    );
                    ui.end_row();
                    for c in &r.cache {
                        ui.label(format!("{}%", c.cache_pct));
                        ui.label(cell(c.lru));
                        ui.label(cell(c.pinned));
                        ui.label(cell(c.optimal));
                        ui.end_row();
                    }
                });
        }
        for pr in &r.prefetch {
            muted(
                ui,
                &p,
                &self
                    .t(
                        "Predicție pentru stratul următor, buget ×{}: {} pregătite la timp, {} doar după popularitate",
                        "Next-layer prediction, budget ×{}: {} ready in time, {} from popularity alone",
                    )
                    .replacen("{}", &pr.mult.to_string(), 1)
                    .replacen("{}", &cell(pr.predicted), 1)
                    .replacen("{}", &cell(pr.popularity_only), 1),
            );
        }
        for sp in &r.speculative {
            let loads = if sp.loads_per_token.is_finite() {
                format!("{:.0}", sp.loads_per_token)
            } else {
                none.to_string()
            };
            muted(
                ui,
                &p,
                &self
                    .t(
                        "{} jetoane verificate odată, acceptare {}: {} încărcări pe jeton",
                        "{} tokens verified at once, {} acceptance: {} loads per token",
                    )
                    .replacen("{}", &sp.k.to_string(), 1)
                    .replacen("{}", &cell(sp.acceptance), 1)
                    .replacen("{}", &loads, 1),
            );
        }
        ui.add_space(4.0);
        for v in &r.verdicts {
            ui.label(format!("• {}", crate::i18n::engine_text(self.s.lang, v)));
        }
    }

    // -------------------------------------------------------------- the settings search

    fn start_autotune(&mut self) {
        let (Some(dir), Some(coli)) = (self.active_model_dir(), self.coli.clone()) else {
            return;
        };
        let Some(key) = self.tune_key() else {
            return;
        };
        let _ = dir;
        let ram = self.known_ram_gb();
        let cores = self.hardware.ready().map_or(1, |h| h.cores.max(1));
        let state_gb = match &self.plan {
            Some(Ok(pl)) => pl.state_gb,
            _ => 0.0,
        };
        let space = Space {
            free_ram_gb: advisor::usable_ram_gb(ram),
            state_gb,
            physical_cores: cores,
            // the conversation length is the user's to choose: it is never changed here
            contexts: Vec::new(),
        };
        let cfg = Config {
            max_runs: 8,
            budget: Duration::from_secs(15 * 60),
            total: Duration::from_secs(60 * 60),
        };
        let base_env = self.accel_env_base();
        let apply = self.tools.apply_tuned;
        self.tools.cancel.store(false, Ordering::Relaxed);
        self.tools.tune = TuneUi::Running {
            done: 0,
            max: cfg.max_runs,
            best: None,
            cancelling: false,
        };
        let cancel = self.tools.cancel.clone();
        let (tx, ui_ctx) = (self.tx.clone(), self.ctx.clone());
        let user_ctx = saved_ctx();
        let finished = Arc::new(AtomicBool::new(false));
        self.tools.worker_done = finished.clone();
        std::thread::spawn(move || {
            let mut coli = coli;
            coli.env = base_env.clone();
            // a panic must neither leave the screen stuck on "running" nor the model stopped
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_search(&coli, &space, cfg, &cancel, &tx, &ui_ctx, &key, user_ctx)
            }))
            .unwrap_or_else(|_| Err(MSG_PANIC.to_string()));
            // always bring the model back, with the new settings when they won
            let mut env = base_env;
            if apply {
                if let Ok(s) = &r {
                    if s.gain > 0.0 {
                        env.extend(s.best.env());
                    }
                }
            }
            coli.env = env;
            let _ = coli.start_server();
            finished.store(true, Ordering::SeqCst);
            let _ = tx.send(Event::AutoTuneDone(r));
            ui_ctx.request_repaint();
        });
    }

    /// The window is closing: a running search is told to stop, and given a moment to put the
    /// model back, so no trial server is left running with experimental settings.
    pub fn tools_on_exit(&mut self) {
        if !self.tools.tune.is_running() {
            return;
        }
        self.tools.cancel.store(true, Ordering::SeqCst);
        let t = Instant::now();
        while !self.tools.worker_done.load(Ordering::SeqCst)
            && t.elapsed() < Duration::from_secs(45)
        {
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    pub fn autotune_progress(&mut self, done: usize, max: usize, best: Option<f64>) {
        if let TuneUi::Running { cancelling, .. } = &self.tools.tune {
            let cancelling = *cancelling;
            self.tools.tune = TuneUi::Running {
                done,
                max,
                best,
                cancelling,
            };
        }
    }

    pub fn autotune_done(&mut self, r: Result<TuneSummary, String>) {
        self.tools.tuned_cache = None;
        self.plan = None;
        self.plan_asked = None;
        self.tools.tune = match r {
            Ok(s) => TuneUi::Done(s),
            Err(e) => TuneUi::Failed(e),
        };
    }

    fn autotune_section(&mut self, ui: &mut Ui) {
        let p = self.p;
        ui.add_space(10.0);
        ui.label(
            RichText::new(self.t(
                "Caută cele mai bune setări pentru acest calculator",
                "Find the best settings for this computer",
            ))
            .size(15.0)
            .semibold(),
        );
        muted(
            ui,
            &p,
            self.t(
                "Încearcă pe rând combinații de memorie fixată și fire de calcul, măsoară viteza reală și le păstrează pe cele mai rapide. Nu schimbă răspunsurile și nici lungimea conversației.",
                "Tries combinations of pinned memory and compute threads one at a time, measures the real speed and keeps the fastest. It changes neither the answers nor the conversation length.",
            ),
        );
        let reason: Option<&'static str> =
            if self.coli.is_none() || self.active_model_dir().is_none() {
                Some(self.t("Mai întâi configurează un model.", "Set up a model first."))
            } else if self.known_ram_gb() <= 0.0 || self.tune_key().is_none() {
                Some(self.t(
                    "Memoria calculatorului nu a fost măsurată încă.",
                    "The computer's memory has not been measured yet.",
                ))
            } else if self.tools.capture.is_on() {
                Some(self.t(
                    "Oprește mai întâi înregistrarea rutării.",
                    "Stop the routing recording first.",
                ))
            } else if self.chat.generating.is_some() {
                Some(self.t(
                    "Așteaptă sfârșitul răspunsului curent.",
                    "Wait for the current answer to finish.",
                ))
            } else if self.tuning || self.ctx_busy {
                Some(self.t(
                    "O altă optimizare este în curs.",
                    "Another optimisation is running.",
                ))
            } else {
                None
            };
        match self.tools.tune.clone() {
            TuneUi::Idle | TuneUi::Done(_) | TuneUi::Failed(_) => {
                if ui
                    .add_enabled(
                        reason.is_none(),
                        egui::Button::new(self.t("Începe căutarea…", "Start the search…")),
                    )
                    .clicked()
                {
                    self.tools.tune = TuneUi::Confirm;
                }
                if let Some(r) = reason {
                    muted(ui, &p, r);
                }
                self.tune_result_view(ui);
            }
            TuneUi::Confirm => {
                tinted(ui, p.warn_soft, p.warn, |ui| {
                    ui.label(self.t(
                        "Atenție: modelul pornit acum este oprit și repornit de mai multe ori, iar cât timp durează nu poți conversa. Durează câteva minute, iar la modelele mari până la o oră. Poți anula oricând.",
                        "Careful: the running model is stopped and restarted several times, and you cannot chat meanwhile. It takes several minutes, and up to an hour for big models. You can cancel at any time.",
                    ));
                });
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            reason.is_none(),
                            egui::Button::new(self.t("Pornește căutarea", "Start the search")),
                        )
                        .clicked()
                    {
                        self.start_autotune();
                    }
                    if ui.button(self.t("Renunț", "Cancel")).clicked() {
                        self.tools.tune = TuneUi::Idle;
                    }
                });
            }
            TuneUi::Running {
                done,
                max,
                best,
                cancelling,
            } => {
                ui.horizontal(|ui| {
                    crate::widgets::spinner(ui);
                    let best_txt = best.map_or("—".to_string(), |b| format!("{b:.1}"));
                    ui.label(
                        self.t(
                            "Testul {} din cel mult {} · cea mai bună viteză până acum: {} jetoane/s",
                            "Test {} of at most {} · best speed so far: {} tokens/s",
                        )
                        .replacen("{}", &(done + 1).min(max).to_string(), 1)
                        .replacen("{}", &max.to_string(), 1)
                        .replacen("{}", &best_txt, 1),
                    );
                });
                if cancelling {
                    muted(
                        ui,
                        &p,
                        self.t(
                            "Se oprește după testul curent…",
                            "Stopping after the current test…",
                        ),
                    );
                } else if ui
                    .button(self.t("Oprește căutarea", "Stop the search"))
                    .clicked()
                {
                    self.tools.cancel.store(true, Ordering::Relaxed);
                    self.tools.tune = TuneUi::Running {
                        done,
                        max,
                        best,
                        cancelling: true,
                    };
                }
            }
        }
        let label = self.t(
            "Pornește modelul cu setările găsite",
            "Start the model with the settings found",
        );
        ui.checkbox(&mut self.tools.apply_tuned, label);
        if self.tools.apply_tuned {
            if let Some(k) = self.tuned_knobs_cached() {
                muted(
                    ui,
                    &p,
                    &self
                        .t("Setări aplicate: {}", "Settings in use: {}")
                        .replacen("{}", &knobs_text(&k), 1),
                );
            }
        }
    }

    fn tune_result_view(&self, ui: &mut Ui) {
        let p = self.p;
        match &self.tools.tune {
            TuneUi::Failed(e) => {
                ui.label(RichText::new(crate::i18n::engine_text(self.s.lang, e)).color(p.bad));
            }
            TuneUi::Done(s) => {
                if s.ok_trials == 0 {
                    let why = crate::i18n::engine_text(
                        self.s.lang,
                        &s.first_error.clone().unwrap_or_default(),
                    );
                    ui.label(
                        RichText::new(
                            self.t(
                                "Nicio măsurătoare nu a reușit: {}",
                                "No measurement succeeded: {}",
                            )
                            .replacen("{}", &why, 1),
                        )
                        .color(p.bad),
                    );
                } else if s.gain > 0.0 {
                    ui.label(
                        RichText::new(
                            self.t(
                                "Câștig: +{}% față de setările automate. Se aplică la următoarea pornire a modelului.",
                                "Gain: +{}% over the automatic settings. It applies from the next time the model starts.",
                            )
                            .replacen("{}", &format!("{:.0}", s.gain * 100.0), 1),
                        )
                        .semibold(),
                    );
                    muted(ui, &p, &knobs_text(&s.best));
                } else {
                    ui.label(self.t(
                        "Nicio îmbunătățire: setările automate ale lui colibri rămân cele mai bune.",
                        "No improvement: colibri's automatic settings stay the best.",
                    ));
                }
                if s.aborted {
                    muted(
                        ui,
                        &p,
                        self.t(
                            "Căutarea a fost oprită; s-a păstrat ce s-a măsurat până atunci.",
                            "The search was stopped; what was measured so far is kept.",
                        ),
                    );
                }
                if s.timed_out {
                    muted(
                        ui,
                        &p,
                        self.t(
                            "Timpul total s-a terminat înainte de sfârșitul căutării.",
                            "The total time ran out before the search was done.",
                        ),
                    );
                }
                muted(
                    ui,
                    &p,
                    &self
                        .t("{} teste, {} reușite", "{} tests, {} succeeded")
                        .replacen("{}", &s.trials.to_string(), 1)
                        .replacen("{}", &s.ok_trials.to_string(), 1),
                );
            }
            _ => {}
        }
    }

    /// Appended to the speed section of the resources window.
    pub fn speed_tools(&mut self, ui: &mut Ui) {
        self.estimate_block(ui);
        self.capture_section(ui);
        self.autotune_section(ui);
    }
}

/// The blocking part of the search: stops the model, measures, restores, saves.
#[allow(clippy::too_many_arguments)]
fn run_search(
    coli: &companion_core::coli::Coli,
    space: &Space,
    cfg: Config,
    cancel: &AtomicBool,
    tx: &Sender<Event>,
    ui_ctx: &egui::Context,
    key: &str,
    user_ctx: Option<u32>,
) -> Result<TuneSummary, String> {
    let _ = coli.stop_server();
    let t = Instant::now();
    loop {
        let stopped = coli
            .status()
            .map(|s| s.server_state() == "stopped")
            .unwrap_or(false);
        if stopped {
            break;
        }
        if t.elapsed() > Duration::from_secs(40) {
            return Err(MSG_NOT_STOPPED.into());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    let mut runner = ProgressRunner {
        inner: HostRunner::new(ColiHost::new(coli.clone())),
        tx: tx.clone(),
        ui: ui_ctx.clone(),
        done: 0,
        max: cfg.max_runs,
        best: 0.0,
    };
    let outcome = autotune::search(space, Knobs::default(), &mut runner, cfg, cancel);
    let saved = outcome.best_score > 0.0
        && autotune::save_profile(
            &autotune::profiles_file(),
            key,
            &outcome.best,
            outcome.best_score,
        )
        .is_ok();
    // the search never changes the context here, but if it did, the user's own value comes back
    if runner.inner.ctx_now() != 0 {
        runner
            .inner
            .restore_ctx(user_ctx.unwrap_or(0))
            .map_err(|e| MSG_CTX_PREFIX.replacen("{}", &e.to_string(), 1))?;
    }
    drop(runner);
    Ok(summarize_outcome(&outcome, saved))
}

#[cfg(test)]
mod tests {
    use super::*;
    use companion_core::autotune::{Outcome, Trial};
    use companion_core::capture::{CachePoint, PrefetchPoint, SpecPoint, Summary};

    #[test]
    fn a_range_is_always_two_different_figures() {
        assert_eq!(fmt_rate_range(0.3, 1.2), "0.30–1.2");
        assert_eq!(fmt_rate_range(12.4, 31.2), "12–32");
        assert_eq!(fmt_rate_range(2.0, 2.0), "2.0–2.1");
        assert_eq!(fmt_rate_range(30.0, 30.0), "30–31");
        // reversed input is put in order
        assert_eq!(fmt_rate_range(5.0, 2.0), "2.0–5.0");
        for (a, b) in [(0.0, 0.004), (0.004, 0.004), (999.0, 1000.0), (0.99, 1.0)] {
            let t = fmt_rate_range(a, b);
            let (x, y) = t.split_once('–').unwrap();
            assert_ne!(x, y, "{t}");
        }
    }

    #[test]
    fn a_range_that_is_not_a_number_is_a_dash() {
        for (a, b) in [
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::INFINITY, 1.0),
            (1.0, f64::NEG_INFINITY),
            (0.0, 0.0),
            (-1.0, -0.5),
        ] {
            assert_eq!(fmt_rate_range(a, b), "—", "{a} {b}");
        }
    }

    #[test]
    fn the_range_never_understates_the_high_end_or_overstates_the_low_end() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..2000 {
            let lo = rnd().powi(3) * 500.0 + 0.011;
            let hi = lo * (1.0 + rnd() * 5.0);
            let t = fmt_rate_range(lo, hi);
            let (x, y) = t.split_once('–').unwrap();
            let (x, y): (f64, f64) = (x.parse().unwrap(), y.parse().unwrap());
            assert!(x <= lo + 1e-9, "{t} low {lo}");
            assert!(y >= hi - 1e-9, "{t} high {hi}");
            assert!(x < y, "{t}");
        }
    }

    #[test]
    fn shares_that_are_not_numbers_have_no_text() {
        assert_eq!(share(0.236).as_deref(), Some("24%"));
        assert_eq!(share(f64::NAN), None);
        assert_eq!(share(f64::INFINITY), None);
        assert_eq!(share(1.7).as_deref(), Some("100%"));
        assert_eq!(share(-0.2).as_deref(), Some("0%"));
    }

    #[test]
    fn no_memory_reading_means_no_estimate_not_too_slow() {
        let m = ModelInfo {
            download_gb: 128.0,
            ..Default::default()
        };
        for ram in [0.0, -4.0, f64::NAN, f64::INFINITY] {
            assert!(estimate_at(ram, Drive::Unknown, &m).is_none(), "{ram}");
        }
        let e = estimate_at(16.0, Drive::Unknown, &m).expect("a real reading gives an estimate");
        assert!(matches!(e.speed, Speed::Slow | Speed::TooSlow));
        assert!(e.tok_s_low <= e.tok_s_high);
    }

    #[test]
    fn the_drive_is_measured_only_after_a_good_test() {
        assert_eq!(drive_of(None), Drive::Unknown);
        assert_eq!(drive_of(Some(&Err("x".into()))), Drive::Unknown);
        assert_eq!(drive_of(Some(&Ok(0.0))), Drive::Unknown);
        assert_eq!(drive_of(Some(&Ok(f64::NAN))), Drive::Unknown);
        assert_eq!(drive_of(Some(&Ok(1800.0))), Drive::Measured(1800.0));
    }

    #[test]
    fn the_recording_state_machine() {
        let cap = 100;
        // armed: nothing until the file appears
        assert_eq!(
            capture_step(&CaptureState::Armed, None, cap, false),
            CaptureStep::Stay
        );
        assert_eq!(
            capture_step(&CaptureState::Armed, Some(0), cap, false),
            CaptureStep::Begin
        );
        assert_eq!(
            capture_step(&CaptureState::Armed, Some(99), cap, false),
            CaptureStep::Begin
        );
        // full beats begin, at exactly the cap
        assert_eq!(
            capture_step(&CaptureState::Armed, Some(100), cap, false),
            CaptureStep::Full
        );
        assert_eq!(
            capture_step(&CaptureState::Recording, Some(99), cap, false),
            CaptureStep::Stay
        );
        assert_eq!(
            capture_step(&CaptureState::Recording, Some(100), cap, false),
            CaptureStep::Full
        );
        assert_eq!(
            capture_step(&CaptureState::Recording, None, cap, false),
            CaptureStep::Stay
        );
        // a file left from before (the model still holds it open) is never the new recording,
        // however big it is, until the model has restarted
        for len in [Some(0), Some(50), Some(100), Some(10_000)] {
            assert_eq!(
                capture_step(&CaptureState::Armed, len, cap, true),
                CaptureStep::Stay,
                "{len:?}"
            );
        }
        // once recording has really begun the flag no longer matters
        assert_eq!(
            capture_step(&CaptureState::Recording, Some(100), cap, true),
            CaptureStep::Full
        );
        // nothing to do in the other states
        for st in [
            CaptureState::Idle,
            CaptureState::Analysing,
            CaptureState::Failed("x".into()),
        ] {
            assert_eq!(capture_step(&st, Some(1000), cap, false), CaptureStep::Stay);
        }
        assert!(CaptureState::Armed.is_on() && CaptureState::Recording.is_on());
        assert!(!CaptureState::Idle.is_on() && !CaptureState::Analysing.is_on());
    }

    fn report() -> Report {
        Report {
            summary: Summary {
                layers: 40,
                tokens: 500,
                ..Default::default()
            },
            n_experts: 256,
            analysed_tokens: 500,
            truncated: false,
            too_short: false,
            cache: vec![CachePoint {
                cache_pct: 10,
                lru: 0.2,
                decay_freq: None,
                pinned: f64::NAN,
                optimal: 0.5,
            }],
            prefetch: vec![PrefetchPoint {
                mult: 2,
                predicted: 0.4,
                popularity_only: f64::NAN,
            }],
            speculative: vec![SpecPoint {
                k: 4,
                acceptance: 0.7,
                loads_per_token: f64::NAN,
            }],
            verdicts: vec!["First.".into(), "Second.".into()],
        }
    }

    #[test]
    fn the_copied_report_says_no_data_instead_of_nan() {
        let t = report_text(&report());
        assert!(t.contains("10% | 20% | no data | 50%"), "{t}");
        assert!(t.contains("x2 | 40% | no data"), "{t}");
        assert!(t.contains("k=4 | 70% | no data"), "{t}");
        assert!(t.contains("- First.") && t.contains("- Second."));
        assert!(!t.to_lowercase().contains("nan"), "{t}");
    }

    #[test]
    fn settings_read_as_plain_text() {
        assert_eq!(knobs_text(&Knobs::default()), "—");
        let k = Knobs {
            pin_gb: Some(3.5),
            threads: Some(6),
            ctx: Some(8192),
        };
        assert_eq!(knobs_text(&k), "PIN_GB=3.5, OMP_NUM_THREADS=6, --ctx 8192");
    }

    fn trial(k: Knobs, tok_s: Option<f64>) -> Trial {
        Trial {
            knobs: k,
            result: tok_s
                .map(|t| {
                    Ok(Measurement {
                        tok_s: t,
                        swap_used_gb: 0.0,
                    })
                })
                .unwrap_or_else(|| Err("boom".to_string())),
            score: tok_s.unwrap_or(0.0),
        }
    }

    #[test]
    fn the_summary_reports_gain_failures_and_stops() {
        let fast = Knobs {
            threads: Some(8),
            ..Default::default()
        };
        let o = Outcome {
            best: fast.clone(),
            best_score: 6.0,
            start_score: 5.0,
            trials: vec![
                trial(Knobs::default(), Some(5.0)),
                trial(fast.clone(), Some(6.0)),
                trial(Knobs::default(), None),
            ],
            aborted: true,
            timed_out: false,
        };
        let s = summarize_outcome(&o, true);
        assert_eq!((s.trials, s.ok_trials), (3, 2));
        assert!((s.gain - 0.2).abs() < 1e-9);
        assert_eq!(s.best_tok_s, 6.0);
        assert!(s.aborted && !s.timed_out && s.saved);
        assert_eq!(s.first_error.as_deref(), Some("boom"));
        // nothing measured: no gain, no best speed
        let none = Outcome {
            best: Knobs::default(),
            best_score: 0.0,
            start_score: 0.0,
            trials: vec![trial(Knobs::default(), None)],
            aborted: false,
            timed_out: true,
        };
        let s = summarize_outcome(&none, false);
        assert_eq!((s.ok_trials, s.gain, s.best_tok_s), (0, 0.0, 0.0));
        assert!(s.timed_out && !s.saved);
    }

    #[test]
    fn only_a_running_search_blocks_other_work() {
        assert!(!TuneUi::Idle.is_running());
        assert!(!TuneUi::Confirm.is_running());
        assert!(
            !TuneUi::Done(summarize_outcome(
                &Outcome {
                    best: Knobs::default(),
                    best_score: 0.0,
                    start_score: 0.0,
                    trials: vec![],
                    aborted: false,
                    timed_out: false
                },
                false
            ))
            .is_running()
        );
        assert!(
            TuneUi::Running {
                done: 0,
                max: 1,
                best: None,
                cancelling: false
            }
            .is_running()
        );
    }

    #[test]
    fn every_verdict_the_analysis_can_write_is_translated() {
        use companion_core::Lang;
        let samples = [
            "The capture holds no routing rows: this engine may not write them, or the run was too short.",
            "Only 20 tokens were captured (at least 200 are needed): treat every number below as indicative.",
            "The capture was longer than the size limit: only its start was analysed.",
            "Pinning the most used experts would serve 61% of loads from a 10% cache against 40% for plain LRU: the learning cache (PIN) is worth using.",
            "LRU already serves 61% of loads from a 10% cache and pinning adds little (62%): the cache policy is not the bottleneck.",
            "Even the best of those leaves 15 points to the theoretical ceiling at a 10% cache: a smarter policy or prefetching could still gain, but the ceiling cannot be reached.",
            "Layers are correlated: a next-layer predictor would have 70% of loads ready in time against 40% from popularity alone, so cross-layer prefetching can pay off: colibri already has it (PILOT, or COUPLE with a table built from this capture by c/tools/route_pairs.py).",
            "Little correlation between layers was found (41% against 40% from popularity alone): prefetching by prediction (PILOT, COUPLE) would not help much.",
            "Verifying 4 drafted tokens per pass at 75% acceptance would cut expert loads per token from 120 to 80: speculative decoding could pay off.",
            "Verifying several drafted tokens per pass would not reduce expert loads on this capture: speculative decoding is not a gain for the drive.",
            "These are cost models over your own routing, not speed measurements.",
            "The capture uses expert 300 but 256 experts per layer were given: 512 were assumed instead.",
            "3 line(s) were not valid routing rows and were ignored.",
            "Only the first 6000 of 9000 tokens were analysed.",
            MSG_NOT_STOPPED,
            MSG_PANIC,
            "Could not restore the context length: disk full",
        ];
        for lang in [
            Lang::Ro,
            Lang::Fr,
            Lang::It,
            Lang::Es,
            Lang::Pt,
            Lang::De,
            Lang::Hu,
        ] {
            for v in samples {
                assert_ne!(crate::i18n::engine_text(lang, v), v, "{lang:?}: {v}");
            }
        }
    }

    #[test]
    fn the_defaults_are_safe() {
        let s = ToolsState::default();
        assert!(!s.capture.is_on() && !s.writing && !s.tune.is_running());
        assert!(s.apply_tuned);
        assert!(!s.cancel.load(Ordering::Relaxed));
    }

    /// Draws the three screens in every interface language, at phone to wide-screen widths and
    /// in every state, headless: it must not panic, loop or ask for an endless repaint.
    #[test]
    fn the_screens_draw_without_panic_in_every_language_width_and_state() {
        use companion_core::Lang;
        let ctx = egui::Context::default();
        let mut app = CompanionApp::new(
            ctx.clone(),
            companion_core::store::Settings::default(),
            Some("http://127.0.0.1:1".into()),
            false,
        );
        let long = "x".repeat(4000);
        let captures = [
            CaptureState::Idle,
            CaptureState::Armed,
            CaptureState::Recording,
            CaptureState::Analysing,
            CaptureState::Done(Box::new(report())),
            CaptureState::Failed(long.clone()),
        ];
        let best = Knobs::default();
        let summary = |ok: usize, gain: f64, err: Option<String>| TuneSummary {
            best: best.clone(),
            gain,
            best_tok_s: 1.5,
            trials: 4,
            ok_trials: ok,
            aborted: true,
            timed_out: true,
            saved: true,
            first_error: err,
        };
        let tunes = [
            TuneUi::Idle,
            TuneUi::Confirm,
            TuneUi::Running {
                done: 99,
                max: 0,
                best: Some(f64::NAN),
                cancelling: false,
            },
            TuneUi::Running {
                done: 0,
                max: 8,
                best: None,
                cancelling: true,
            },
            TuneUi::Done(summary(3, 0.25, None)),
            TuneUi::Done(summary(0, 0.0, Some(long.clone()))),
            TuneUi::Done(summary(2, f64::NAN, None)),
            TuneUi::Failed(long.clone()),
        ];
        let langs = [
            Lang::Ro,
            Lang::En,
            Lang::Fr,
            Lang::It,
            Lang::Es,
            Lang::Pt,
            Lang::De,
            Lang::Hu,
        ];
        // the first frames of a new context always ask for a repaint (fonts, layout): let them pass
        for _ in 0..4 {
            let mut out = ctx.run_ui(egui::RawInput::default(), |ui| app.speed_tools(ui));
            out.textures_delta.clear();
        }
        for lang in langs {
            app.s.lang = lang;
            for width in [320.0_f32, 480.0, 800.0, 1600.0] {
                // every capture state beside an idle search, and every search state beside an
                // idle capture (the two sections do not depend on each other's layout)
                let pairs = captures
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| (ci, c, 0usize, &tunes[0]))
                    .chain(
                        tunes
                            .iter()
                            .enumerate()
                            .map(|(ti, t)| (0usize, &captures[0], ti, t)),
                    );
                {
                    for (ci, cap, ti, tune) in pairs {
                        app.tools.capture = cap.clone();
                        app.tools.tune = tune.clone();
                        app.tools.bytes_now = u64::MAX;
                        let raw = |t: usize| egui::RawInput {
                            time: Some(t as f64 * 0.1),
                            screen_rect: Some(egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::vec2(width, 900.0),
                            )),
                            ..Default::default()
                        };
                        // egui needs a few frames to settle a new layout; it must then rest
                        let mut out = None;
                        for t in 0..12 {
                            let mut f = ctx.run_ui(raw(t), |ui| {
                                egui::ScrollArea::vertical().show(ui, |ui| app.speed_tools(ui));
                            });
                            f.textures_delta.clear();
                            let settled = f
                                .viewport_output
                                .get(&egui::ViewportId::ROOT)
                                .is_some_and(|v| v.repaint_delay != Duration::ZERO);
                            out = Some(f);
                            if settled {
                                break;
                            }
                        }
                        let out = out.expect("at least one frame");
                        // a screen at rest (no spinner) never asks for an endless repaint
                        let spinning = matches!(cap, CaptureState::Analysing)
                            || matches!(tune, TuneUi::Running { .. });
                        let delay = out
                            .viewport_output
                            .get(&egui::ViewportId::ROOT)
                            .map(|v| v.repaint_delay);
                        if !spinning {
                            assert_ne!(
                                delay,
                                Some(Duration::ZERO),
                                "{lang:?} {width} capture#{ci} tune#{ti}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn headless_app() -> CompanionApp {
        CompanionApp::new(
            egui::Context::default(),
            companion_core::store::Settings::default(),
            Some("http://127.0.0.1:1".into()),
            false,
        )
    }

    #[test]
    fn while_a_search_runs_the_model_belongs_to_it() {
        let mut app = headless_app();
        assert!(!app.tune_blocks_server());
        assert!(app.toasts.is_empty());
        app.tools.tune = TuneUi::Running {
            done: 1,
            max: 8,
            best: None,
            cancelling: false,
        };
        assert!(app.tune_blocks_server());
        assert_eq!(app.toasts.len(), 1, "the user is told why nothing happened");
        // closing the window asks the search to stop and does not wait for a finished worker
        app.tools.worker_done.store(true, Ordering::SeqCst);
        let t = Instant::now();
        app.tools_on_exit();
        assert!(app.tools.cancel.load(Ordering::SeqCst));
        assert!(t.elapsed() < Duration::from_secs(2));
        // with no search running, closing does nothing at all
        app.tools.tune = TuneUi::Idle;
        app.tools.cancel.store(false, Ordering::SeqCst);
        app.tools.worker_done.store(false, Ordering::SeqCst);
        let t = Instant::now();
        app.tools_on_exit();
        assert!(!app.tools.cancel.load(Ordering::SeqCst));
        assert!(t.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn a_recording_asked_for_again_ignores_a_file_the_model_still_holds() {
        let mut app = headless_app();
        app.tools.stale_file = true;
        app.tools.capture = CaptureState::Armed;
        assert_eq!(
            capture_step(
                &app.tools.capture,
                Some(CAP_BYTES),
                CAP_BYTES,
                app.tools.stale_file
            ),
            CaptureStep::Stay
        );
        // the model restarting makes it write the file afresh
        app.tools_server_changed();
        assert!(!app.tools.stale_file);
        assert_eq!(
            capture_step(
                &app.tools.capture,
                Some(10),
                CAP_BYTES,
                app.tools.stale_file
            ),
            CaptureStep::Begin
        );
    }

    #[test]
    fn the_error_texts_the_search_thread_can_write_are_translated() {
        use companion_core::Lang;
        let ctx_msg = MSG_CTX_PREFIX.replacen("{}", "disk full", 1);
        for lang in [
            Lang::Ro,
            Lang::Fr,
            Lang::It,
            Lang::Es,
            Lang::Pt,
            Lang::De,
            Lang::Hu,
        ] {
            for m in [MSG_NOT_STOPPED, MSG_PANIC, ctx_msg.as_str()] {
                assert_ne!(crate::i18n::engine_text(lang, m), m, "{lang:?}: {m}");
            }
        }
        assert!(crate::i18n::engine_text(Lang::De, &ctx_msg).contains("disk full"));
    }
}
