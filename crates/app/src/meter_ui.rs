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
        self.meter.open = true;
        self.meter.shown = None;
        self.meter.shown_history.clear();
        self.meter.gate.store(turn, Ordering::SeqCst);
        let engine_pid = self
            .status
            .as_ref()
            .and_then(|s| s.server.as_ref())
            .and_then(|s| s.pid);
        let (gate, tx, ctx) = (self.meter.gate.clone(), self.tx.clone(), self.ctx.clone());
        thread::spawn(move || {
            let mut sampler = Sampler::new();
            thread::sleep(Duration::from_millis(300));
            while gate.load(Ordering::SeqCst) == turn {
                let targets: Vec<(String, u32)> = engine_pid
                    .map(|p| vec![("engine".to_string(), p)])
                    .unwrap_or_default();
                let snap = sampler.sample(&targets, None);
                let s = Sample::from_snapshot(&snap, engine_pid);
                if tx.send(Event::MeterSample { turn, sample: s }).is_err() {
                    return;
                }
                ctx.request_repaint();
                thread::sleep(Duration::from_millis(1000));
            }
        });
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
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(self.t("Consum pentru acest răspuns", "Usage for this answer"))
                    .semibold(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("×")
                    .on_hover_text(self.t("Închide", "Close"))
                    .clicked()
                {
                    self.meter.open = false;
                }
            });
        });
        ui.add_space(4.0);
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            egui::Grid::new("meter-grid").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
                let row = |ui: &mut Ui, label: &str, value: String| {
                    ui.label(RichText::new(label).color(p.muted));
                    ui.label(value);
                    ui.end_row();
                };
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
                ui.label(RichText::new(self.t("Context folosit", "Context used")).color(p.muted));
                ui.add(
                    egui::ProgressBar::new(fill.clamp(0.0, 1.0))
                        .text(format!("{:.0} %", fill * 100.0))
                        .animate(false),
                );
            }
            if history.len() > 1 {
                ui.add_space(6.0);
                ui.label(RichText::new(self.t("Viteza în timp (tokeni/s)", "Speed over time (tokens/s)")).color(p.muted));
                sparkline(ui, &history, p.accent);
            }
            ui.add_space(8.0);
            ui.label(RichText::new(self.t("Calculatorul cât se scrie", "The machine while writing")).semibold());
            ui.add_space(2.0);
            egui::Grid::new("meter-machine").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
                let row = |ui: &mut Ui, label: &str, value: String| {
                    ui.label(RichText::new(label).color(p.muted));
                    ui.label(value);
                    ui.end_row();
                };
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
            ui.add_space(8.0);
            let mut notes = vec![self.t(
                "Încărcarea calculatorului se citește aici o dată pe secundă. Un server de pe alt calculator se măsoară doar prin viteză și tokeni.",
                "The machine's load is read here once a second. A server on another computer is measured by speed and tokens only.",
            )];
            if stats.tokens_estimated {
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

/// A small line of the speed over time, drawn with shapes.
fn sparkline(ui: &mut Ui, values: &[f32], colour: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(
        vec2(ui.available_width().min(260.0), 44.0),
        egui::Sense::hover(),
    );
    let max = values.iter().copied().fold(0.0_f32, f32::max).max(0.1);
    let n = values.len().max(2) - 1;
    let pts: Vec<egui::Pos2> = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let x = rect.left() + rect.width() * i as f32 / n as f32;
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
}
