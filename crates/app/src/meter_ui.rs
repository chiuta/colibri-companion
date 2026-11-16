//! A side panel next to the answer: what it costs while it is being written (tokens, speed,
//! context, processor, memory, graphics card) and, afterwards, what it cost in the end. The
//! figures come from `companion_core::meter`; a machine reading is taken once a second while
//! an answer is being written, and the thread stops with the answer.

use crate::app::{CompanionApp, Event};
use crate::theme::Semibold;
use crate::widgets::loc;
use companion_core::chat::StreamSummary;
use companion_core::meter::{AnswerStats, Live, Meter, Sample};
use companion_core::resources::Sampler;
use egui::{RichText, Ui, vec2};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct MeterState {
    pub open: bool,
    /// The person closed the panel: the next answer does not open it again by itself. Pressing
    /// "Usage" under an answer opens it and clears this.
    pub dismissed: bool,
    /// The machine reading of the answer being written means something: the server is on this
    /// computer. For a server on another computer only speed and tokens are measured.
    pub machine: bool,
    /// The answer being written: its turn, the meter, when it began.
    pub run: Option<(u64, Meter, Instant)>,
    /// What to show once nothing is being written (the last answer, or one picked from the chat).
    pub shown: Option<AnswerStats>,
    /// Speed history of the answer that just ended (not kept in the conversation file).
    pub shown_history: Vec<f32>,
    /// The sampler thread runs only while this equals its own turn.
    gate: Arc<AtomicU64>,
}

impl CompanionApp {
    /// An answer is about to be written: start measuring it.
    pub fn begin_meter(&mut self, turn: u64) {
        let ctx_tokens = self.facts().and_then(|f| f.context_tokens);
        self.meter.run = Some((turn, Meter::new(ctx_tokens), Instant::now()));
        self.meter.open = !self.meter.dismissed;
        self.meter.shown = None;
        self.meter.shown_history.clear();
        // Only a server on this computer is read: this computer's processor and memory say
        // nothing about a server somewhere else.
        let local = companion_core::net::is_local_base(&self.base());
        self.meter.machine = local;
        self.meter.gate.store(turn, Ordering::SeqCst);
        if !local {
            return;
        }
        let engine_pid = self
            .status
            .as_ref()
            .and_then(|s| s.server.as_ref())
            .and_then(|s| s.pid);
        let (gate, tx, ctx) = (self.meter.gate.clone(), self.tx.clone(), self.ctx.clone());
        let _ = thread::Builder::new()
            .name("meter-sampler".into())
            .spawn(move || sample_while(&gate, turn, engine_pid, &tx, &ctx));
    }

    pub fn meter_delta(&mut self, turn: u64, kind: companion_core::meter::Kind, text: &str) {
        if let Some((t, m, started)) = self.meter.run.as_mut()
            && *t == turn
        {
            m.delta(started.elapsed().as_millis() as u64, kind, text);
        }
    }

    pub fn meter_sample(&mut self, turn: u64, sample: Sample) {
        if let Some((t, m, _)) = self.meter.run.as_mut()
            && *t == turn
        {
            m.sample(sample);
        }
    }

    /// The answer ended (or was stopped): keep the figures with it and stop the sampler.
    pub fn finish_meter(&mut self, summary: Option<&StreamSummary>) {
        self.meter.gate.store(0, Ordering::SeqCst);
        let Some((_, m, started)) = self.meter.run.take() else {
            return;
        };
        let end = started.elapsed().as_millis() as u64;
        let live = m.live(end);
        let stats = m.finish(end, summary);
        if let Some(last) = self
            .chat
            .conv
            .messages
            .last_mut()
            .filter(|x| x.role == "assistant")
        {
            last.stats = Some(stats.clone());
        }
        self.meter.shown = Some(stats);
        self.meter.shown_history = live.tps_history;
    }

    pub fn show_meter(&self) -> bool {
        self.meter.open && (self.meter.run.is_some() || self.meter.shown.is_some())
    }

    pub fn meter_panel(&mut self, ui: &mut Ui) {
        // a named region, so a screen reader can find the panel and say what it is
        let name = self.t("Consum pentru acest răspuns", "Usage for this answer");
        ui.scope_builder(
            egui::UiBuilder::new()
                .accessibility_role(egui::accesskit::Role::Complementary)
                .accessibility_label(name),
            |ui| self.meter_panel_contents(ui, name),
        );
    }

    fn meter_panel_contents(&mut self, ui: &mut Ui, title: &str) {
        let p = self.p;
        let live: Option<Live> = self
            .meter
            .run
            .as_ref()
            .map(|(_, m, started)| m.live(started.elapsed().as_millis() as u64));
        let writing = live.is_some();
        if writing {
            ui.ctx().request_repaint_after(Duration::from_millis(250));
        }
        let (stats, history, now_tps, last) = match &live {
            Some(l) => (l.stats.clone(), l.tps_history.clone(), l.tps_now, l.last),
            None => (
                self.meter.shown.clone().unwrap_or_default(),
                self.meter.shown_history.clone(),
                None,
                None,
            ),
        };
        // the machine rows are shown when this answer was measured on this computer
        let machine = stats.samples > 0 || (writing && self.meter.machine);
        // the title wraps instead of running under the close button, in every language
        ui.horizontal_wrapped(|ui| {
            let close = self.t("Închide", "Close");
            let b = ui.small_button("×").on_hover_text(close);
            // the glyph alone is read as "multiplication sign": name the button instead
            ui.ctx()
                .accesskit_node_builder(b.id, |n| n.set_label(close));
            if b.clicked() {
                self.meter.open = false;
                self.meter.dismissed = true;
            }
            ui.label(RichText::new(title).semibold());
        });
        ui.add_space(4.0);
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            // Each figure is a small muted name with its value under it, not a two-column grid:
            // a long name (German, Hungarian) wraps inside the panel instead of pushing the
            // value out of sight, and everything that follows keeps the panel's width.
            let row = |ui: &mut Ui, label: &str, value: String| {
                ui.label(RichText::new(label).small().color(p.muted));
                ui.label(value);
                ui.add_space(3.0);
            };
            ui.scope(|ui| {
                row(ui, self.t("Timp scurs", "Time elapsed"), secs(stats.total_ms));
                row(
                    ui,
                    self.t("Primul token după", "First token after"),
                    stats.first_token_ms.map(secs).unwrap_or_else(dash),
                );
                row(
                    ui,
                    self.t("Tokeni la intrare (întrebare și context)", "Tokens in (question and context)"),
                    stats.prompt_tokens.map(|n| n.to_string()).unwrap_or_else(dash),
                );
                row(
                    ui,
                    self.t("Tokeni produși", "Tokens out"),
                    stats
                        .completion_tokens
                        .map(|n| format!("{}{n}", if stats.tokens_estimated { "≈ " } else { "" }))
                        .unwrap_or_else(dash),
                );
                if writing {
                    row(ui, self.t("Viteză acum", "Speed now"), tps(now_tps));
                }
                row(ui, self.t("Viteză medie", "Average speed"), tps(stats.avg_tps));
                if let Some(pk) = stats.peak_tps {
                    row(ui, self.t("Viteză maximă", "Peak speed"), tps(Some(pk)));
                }
                if let Some(share) = stats.reasoning_share() {
                    row(
                        ui,
                        self.t("Parte din text care e «gândire»", "Share of the text that is thinking"),
                        format!("{:.0} %", share * 100.0),
                    );
                }
            });
            if let Some(fill) = stats.context_fill() {
                ui.add_space(6.0);
                let l = ui.label(RichText::new(self.t("Context folosit", "Context used")).color(p.muted));
                ui.add(
                    egui::ProgressBar::new(fill.clamp(0.0, 1.0))
                        .text(format!("{:.0} %", fill * 100.0))
                        .animate(false),
                )
                .labelled_by(l.id);
            }
            if history.len() > 1 {
                ui.add_space(6.0);
                let name = self.t("Viteza în timp (tokeni/s)", "Speed over time (tokens/s)");
                ui.label(RichText::new(name).color(p.muted));
                sparkline(ui, &history, p.accent, name);
            }
            if machine {
                ui.add_space(8.0);
                ui.label(RichText::new(self.t("Calculatorul cât se scrie", "The machine while writing")).semibold());
                ui.add_space(2.0);
                ui.scope(|ui| {
                    if let Some(l) = last {
                        // the figures at this moment, while the answer is being written
                        if let Some(c) = l.engine_cores {
                            row(ui, self.t("Procesor motor acum (nuclee ocupate)", "Engine processor now (cores busy)"), loc(format!("{c:.1}")));
                        }
                        row(
                            ui,
                            self.t("Memorie folosită acum", "Memory used now"),
                            loc(format!("{:.1} / {:.1} GB", l.ram_used_gb, l.ram_total_gb)),
                        );
                    }
                    row(
                        ui,
                        self.t("Procesor, tot calculatorul (medie / vârf)", "Processor, whole machine (average / peak)"),
                        pair(stats.cpu_avg, stats.cpu_peak, "%"),
                    );
                    row(
                        ui,
                        self.t("Procesor motor, vârf (nuclee)", "Engine processor, peak (cores)"),
                        stats.engine_cores_peak.map(|c| loc(format!("{c:.1}"))).unwrap_or_else(dash),
                    );
                    row(
                        ui,
                        self.t("Memorie motor, vârf", "Engine memory, peak"),
                        stats.engine_rss_peak_gb.map(gb).unwrap_or_else(dash),
                    );
                    row(
                        ui,
                        self.t("Memorie folosită, vârf", "Memory used, peak"),
                        stats.ram_peak_gb.map(gb).unwrap_or_else(dash),
                    );
                    row(
                        ui,
                        self.t("Swap folosit, vârf", "Swap used, peak"),
                        stats.swap_peak_gb.map(gb).unwrap_or_else(dash),
                    );
                    if stats.gpu_peak.is_some() || stats.vram_peak_gb.is_some() {
                        row(
                            ui,
                            self.t("Placa video (medie / vârf)", "Graphics card (average / peak)"),
                            pair(stats.gpu_avg, stats.gpu_peak, "%"),
                        );
                        row(
                            ui,
                            self.t("Memorie video, vârf", "Graphics memory, peak"),
                            match (stats.vram_peak_gb, stats.vram_total_gb) {
                                (Some(u), Some(t)) => loc(format!("{u:.1} / {t:.1} GB")),
                                (Some(u), None) => gb(u),
                                _ => dash(),
                            },
                        );
                    }
                });
            }
            ui.add_space(8.0);
            let mut notes = vec![self.t(
                "Încărcarea calculatorului se citește aici o dată pe secundă, doar pentru un server de pe acest calculator; partea motorului se vede doar pentru un motor pornit de această aplicație. Un server de pe alt calculator se măsoară doar prin viteză și tokeni.",
                "The machine's load is read here once a second, only for a server on this computer; the engine's own share shows only for an engine this app started. A server on another computer is measured by speed and tokens only.",
            )];
            if stats.tokens_estimated && stats.completion_tokens.is_some() {
                notes.push(self.t(
                    "Cifrele cu ≈ sunt estimate din bucățile primite; serverul nu le-a raportat.",
                    "Figures with ≈ are estimated from the pieces received; the server did not report them.",
                ));
            }
            for n in notes {
                ui.label(RichText::new(n).small().color(p.muted));
            }
        });
    }
}

/// Reads the machine about once a second for as long as `gate` still holds `turn`. The wait is
/// cut into short slices, so the thread ends within a fraction of a second after the answer does.
fn sample_while(
    gate: &AtomicU64,
    turn: u64,
    engine_pid: Option<u32>,
    tx: &std::sync::mpsc::Sender<Event>,
    ctx: &egui::Context,
) {
    let live = || gate.load(Ordering::SeqCst) == turn;
    let pause = |ms: u64| {
        let until = Instant::now() + Duration::from_millis(ms);
        while live() && Instant::now() < until {
            thread::sleep(Duration::from_millis(50));
        }
    };
    let mut sampler = Sampler::new();
    pause(300);
    while live() {
        let targets: Vec<(String, u32)> = engine_pid
            .map(|p| vec![("engine".to_string(), p)])
            .unwrap_or_default();
        let snap = sampler.sample(&targets, None);
        // the answer may have ended while the reading was taken: say nothing then
        if !live() {
            return;
        }
        let sample = Sample::from_snapshot(&snap, engine_pid);
        if tx.send(Event::MeterSample { turn, sample }).is_err() {
            return;
        }
        ctx.request_repaint();
        pause(1000);
    }
}

fn dash() -> String {
    "—".to_string()
}

fn secs(ms: u64) -> String {
    loc(format!("{:.1} s", ms as f64 / 1000.0))
}

fn tps(v: Option<f64>) -> String {
    v.map(|t| loc(format!("{t:.1} tok/s"))).unwrap_or_else(dash)
}

fn gb(v: f64) -> String {
    loc(format!("{v:.1} GB"))
}

fn pair(avg: Option<f32>, peak: Option<f32>, unit: &str) -> String {
    match (avg, peak) {
        (Some(a), Some(p)) => format!("{a:.0}{unit} / {p:.0}{unit}"),
        _ => dash(),
    }
}

/// A small line of the speed over time, drawn with shapes. `name` and a summary of the figures are
/// what a screen reader gets, since the drawing itself has no text.
fn sparkline(ui: &mut Ui, values: &[f32], colour: egui::Color32, name: &str) {
    let (rect, resp) = ui.allocate_exact_size(
        vec2(ui.available_width().min(260.0), 44.0),
        egui::Sense::hover(),
    );
    let top = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0_f32, f32::max);
    let last = values
        .last()
        .copied()
        .filter(|v| v.is_finite())
        .unwrap_or(0.0);
    resp.widget_info(|| {
        egui::WidgetInfo::labeled(
            egui::accesskit::Role::Image,
            true,
            loc(format!("{name}: {last:.1}, max {top:.1}")),
        )
    });
    let max = top.max(0.1);
    let n = values.len().max(2) - 1;
    let pts: Vec<egui::Pos2> = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let x = rect.left() + rect.width() * i as f32 / n as f32;
            let v = if v.is_finite() { *v } else { 0.0 };
            let y = rect.bottom() - 2.0 - (rect.height() - 4.0) * (v / max).clamp(0.0, 1.0);
            egui::pos2(x, y)
        })
        .collect();
    ui.painter()
        .add(egui::Shape::line(pts, egui::Stroke::new(1.6, colour)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use companion_core::resources::Snapshot;

    #[test]
    fn small_formatters_never_panic_on_odd_values() {
        assert_eq!(secs(1500), "1.5 s");
        assert_eq!(tps(None), "—");
        assert_eq!(pair(Some(12.4), Some(80.0), "%"), "12% / 80%");
        assert_eq!(pair(None, Some(1.0), "%"), "—");
        assert_eq!(gb(f64::NAN), "NaN GB");
    }

    #[test]
    fn a_snapshot_without_the_engine_reads_machine_figures_only() {
        let snap = Snapshot {
            cpu_percent: 40.0,
            ram_used_gb: 8.0,
            ram_total_gb: 16.0,
            ..Snapshot::default()
        };
        let s = Sample::from_snapshot(&snap, None);
        assert_eq!(s.engine_cores, None);
        assert_eq!(s.ram_total_gb, 16.0);
    }

    use crate::widgets::testkit;
    use companion_core::Lang;
    use companion_core::meter::Kind;
    use companion_core::store::Settings;

    fn app(lang: Lang) -> CompanionApp {
        let s = Settings {
            lang,
            ..Settings::default()
        };
        CompanionApp::new(
            egui::Context::default(),
            s,
            Some("http://127.0.0.1:1".into()),
            false,
        )
    }

    /// A busy answer: pieces, server figures, a machine reading with GPU and engine.
    fn metered() -> Meter {
        let mut m = Meter::new(Some(4096));
        for i in 0..60 {
            m.delta(100 + i * 60, Kind::Answer, "word ");
        }
        m.delta(4000, Kind::Reasoning, "hm");
        m.sample(Sample {
            cpu_percent: 55.0,
            engine_cores: Some(6.5),
            engine_rss_gb: Some(9.0),
            ram_used_gb: 12.0,
            ram_total_gb: 16.0,
            swap_used_gb: 0.5,
            gpu_util_percent: Some(30.0),
            vram_used_gb: Some(2.0),
            vram_total_gb: Some(8.0),
        });
        m
    }

    fn draw_panel(a: &mut CompanionApp, w: f32) -> Vec<testkit::Drawn> {
        let ctx = a.ctx.clone();
        let frame = |a: &mut CompanionApp| {
            testkit::draw(&ctx, w + 80.0, 700.0, |ui| {
                egui::Panel::right("meter")
                    .exact_size(w)
                    .show(ui, |ui| a.meter_panel(ui));
            })
        };
        frame(a);
        frame(a)
    }

    #[test]
    fn the_panel_fits_its_column_in_every_language_while_writing_and_afterwards() {
        let mut problems = Vec::new();
        for lang in Lang::ALL {
            for width in [220.0_f32, 290.0, 400.0] {
                let mut a = app(lang);
                let m = metered();
                let sum = StreamSummary {
                    prompt_tokens: Some(1234),
                    completion_tokens: Some(61),
                    first_token_ms: Some(90),
                    total_ms: 4100,
                    ..StreamSummary::default()
                };
                a.meter.open = true;
                a.meter.machine = true;
                a.meter.run = Some((1, m.clone(), Instant::now()));
                let writing = draw_panel(&mut a, width);
                a.meter.run = None;
                a.meter.shown = Some(m.finish(4100, Some(&sum)));
                let done = draw_panel(&mut a, width);
                for (what, drawn) in [("writing", writing), ("done", done)] {
                    assert!(drawn.len() > 10, "{lang:?} {what}: {} texts", drawn.len());
                    for d in testkit::clipped(&drawn) {
                        problems.push(format!("{lang:?} @{width} {what}: '{}' is cut off", d.text));
                    }
                    for (x, y) in testkit::overlapping(&drawn) {
                        problems.push(format!(
                            "{lang:?} @{width} {what}: '{}' overlaps '{}' ({:?} / {:?})",
                            x.text, y.text, x.rect, y.rect
                        ));
                    }
                }
            }
        }
        problems.sort();
        problems.dedup();
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    #[test]
    fn a_server_on_another_computer_shows_no_machine_rows() {
        let mut a = app(Lang::En);
        let mut m = metered();
        a.meter.open = true;
        // remote: no machine readings were taken and none are claimed
        a.meter.machine = false;
        let remote = Meter::new(None);
        a.meter.run = Some((1, remote, Instant::now()));
        let drawn = draw_panel(&mut a, 290.0);
        assert!(
            !drawn
                .iter()
                .any(|d| d.text.contains("The machine while writing"))
        );
        // local: the same panel does
        a.meter.machine = true;
        a.meter.run = Some((2, std::mem::take(&mut m), Instant::now()));
        let drawn = draw_panel(&mut a, 290.0);
        assert!(
            drawn
                .iter()
                .any(|d| d.text.contains("The machine while writing"))
        );
    }

    #[test]
    fn the_estimate_note_appears_only_when_an_estimate_is_shown() {
        let mut a = app(Lang::En);
        a.meter.open = true;
        // stopped before any piece arrived: nothing is estimated, so no note about it
        a.meter.shown = Some(Meter::new(None).finish(10, None));
        let none = draw_panel(&mut a, 400.0);
        assert!(!none.iter().any(|d| d.text.contains("estimated")));
        a.meter.shown = Some(metered().finish(5000, None));
        let some = draw_panel(&mut a, 400.0);
        assert!(some.iter().any(|d| d.text.contains("estimated")));
    }

    #[test]
    fn the_panel_and_its_buttons_have_names_for_a_screen_reader() {
        use egui::accesskit::Role;
        let mut a = app(Lang::En);
        a.meter.open = true;
        a.meter.machine = true;
        a.meter.run = Some((1, metered(), Instant::now()));
        a.meter.shown = Some(metered().finish(5000, None));
        let ctx = a.ctx.clone();
        ctx.enable_accesskit();
        let mut update = None;
        for _ in 0..2 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    vec2(700.0, 600.0),
                )),
                ..Default::default()
            };
            let mut out = ctx.run_ui(input, |ui| {
                egui::Panel::right("meter")
                    .exact_size(290.0)
                    .show(ui, |ui| a.meter_panel(ui));
            });
            out.textures_delta.clear();
            update = out.platform_output.accesskit_update;
        }
        let nodes = update.expect("an accessibility tree").nodes;
        let named = |role: Role, label: &str| {
            nodes
                .iter()
                .any(|(_, n)| n.role() == role && n.label() == Some(label))
        };
        assert!(
            named(Role::Complementary, "Usage for this answer"),
            "the panel is a named region"
        );
        assert!(
            named(Role::Button, "Close"),
            "the × button is called Close, not 'multiplication'"
        );
        assert!(
            nodes.iter().any(|(_, n)| n.role() == Role::Image
                && n.label().is_some_and(|l| l.starts_with("Speed over time"))),
            "the speed drawing is described"
        );
    }

    #[test]
    fn closing_the_panel_keeps_it_closed_for_the_next_answer() {
        let mut a = app(Lang::En);
        a.begin_meter(1);
        assert!(a.meter.open);
        a.meter.open = false;
        a.meter.dismissed = true;
        a.finish_meter(None);
        a.begin_meter(2);
        assert!(
            !a.meter.open,
            "the person closed it; it does not pop up again"
        );
        a.finish_meter(None);
        a.meter.dismissed = false; // pressing "Usage" under an answer
        a.begin_meter(3);
        assert!(a.meter.open);
        a.finish_meter(None);
    }

    #[test]
    fn the_sampler_thread_ends_with_the_answer_and_a_stale_turn_is_ignored() {
        let gate = Arc::new(AtomicU64::new(7));
        let (tx, rx) = std::sync::mpsc::channel();
        let g = gate.clone();
        let ctx = egui::Context::default();
        let t = Instant::now();
        let h = thread::spawn(move || sample_while(&g, 7, None, &tx, &ctx));
        // wait for the first reading, then end the answer
        let first = rx.recv_timeout(Duration::from_secs(20)).expect("a reading");
        assert!(matches!(first, Event::MeterSample { turn: 7, .. }));
        gate.store(0, Ordering::SeqCst);
        let ended = Instant::now();
        h.join().unwrap();
        assert!(
            ended.elapsed() < Duration::from_millis(1500),
            "the thread outlived the answer by {:?}",
            ended.elapsed()
        );
        assert!(t.elapsed() < Duration::from_secs(30));
        // nothing arrives after the thread ended
        while let Ok(extra) = rx.try_recv() {
            assert!(matches!(extra, Event::MeterSample { turn: 7, .. }));
        }
        // a reading of an old turn does not touch the answer being written now
        let mut a = app(Lang::En);
        a.meter.run = Some((9, Meter::new(None), Instant::now()));
        a.meter_sample(
            7,
            Sample {
                cpu_percent: 99.0,
                ..Sample::default()
            },
        );
        let live = a.meter.run.as_ref().unwrap().1.live(0);
        assert_eq!(live.stats.samples, 0);
        a.meter_sample(
            9,
            Sample {
                cpu_percent: 50.0,
                ..Sample::default()
            },
        );
        assert_eq!(a.meter.run.as_ref().unwrap().1.live(0).stats.samples, 1);
    }

    #[test]
    fn finishing_stores_the_figures_with_the_last_answer_and_stops_the_gate() {
        let mut a = app(Lang::En);
        a.chat
            .conv
            .messages
            .push(companion_core::store::StoredMessage {
                role: "assistant".into(),
                ..Default::default()
            });
        a.begin_meter(4);
        assert_eq!(a.meter.gate.load(Ordering::SeqCst), 4);
        a.meter_delta(4, Kind::Answer, "hello");
        a.meter_delta(3, Kind::Answer, "stale turn");
        a.finish_meter(None);
        assert_eq!(a.meter.gate.load(Ordering::SeqCst), 0);
        let st = a.chat.conv.messages[0]
            .stats
            .clone()
            .expect("stats kept with the answer");
        assert_eq!(
            st.completion_tokens,
            Some(1),
            "the stale piece was not counted"
        );
        assert!(a.meter.run.is_none() && a.meter.shown.is_some());
        // a second finish (stop, then the stream's own end) changes nothing
        a.finish_meter(None);
        assert!(a.meter.shown.is_some());
    }
}
