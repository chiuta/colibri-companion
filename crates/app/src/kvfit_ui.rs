//! "Measure cache shrinking": runs `corbul kvfit` on the model corbul is running, for a text file
//! and a question, and shows what it printed (see `docs/KV-FIT.md`). It loads the model a second
//! time, so it says so; nothing is changed in the running model.
use crate::app::{CompanionApp, Event};
use crate::widgets::*;
use egui::{Align2, vec2};
use std::path::PathBuf;

pub struct KvfitUi {
    pub open: bool,
    pub text: String,
    pub question: String,
    pub budget: f32,
    pub running: bool,
    pub output: String,
}
impl Default for KvfitUi {
    fn default() -> Self {
        KvfitUi {
            open: false,
            text: String::new(),
            question: String::new(),
            budget: 0.25,
            running: false,
            output: String::new(),
        }
    }
}

/// Why the inputs cannot be used, or `None` when they can.
pub fn check_inputs(text: &str, question: &str, budget: f32) -> Option<&'static str> {
    let t = text.trim().trim_matches('"');
    if t.is_empty() || !std::path::Path::new(t).is_file() {
        return Some("text");
    }
    if question.trim().is_empty() {
        return Some("question");
    }
    if !(budget > 0.0 && budget < 1.0) {
        return Some("budget");
    }
    None
}

impl CompanionApp {
    pub fn run_kvfit(&mut self) {
        let (Some(bin), Some(model)) = (self.corbul_bin.clone(), self.corbul_path.clone()) else {
            return;
        };
        if self.kvfit.running || self.corbul.is_none() {
            return;
        }
        let text = PathBuf::from(self.kvfit.text.trim().trim_matches('"'));
        let (q, b) = (
            self.kvfit.question.trim().to_string(),
            self.kvfit.budget as f64,
        );
        self.kvfit.running = true;
        self.kvfit.output.clear();
        self.spawn(move |tx| {
            let r = companion_core::corbul::run_kvfit(&bin, &model, &text, &q, b)
                .map_err(|e| e.to_string());
            let _ = tx.send(Event::KvfitDone(r));
        });
    }

    pub fn kvfit_window(&mut self, ctx: &egui::Context) {
        if !self.kvfit.open {
            return;
        }
        let p = self.p;
        let mut open = true;
        let mut run = false;
        let name = self.corbul_name.clone();
        let hint_text = self.t(
            "Calea unui fișier text (UTF-8, cel mult 8 MiB)",
            "Path to a text file (UTF-8, at most 8 MiB)",
        );
        let hint_q = self.t("Întrebarea", "The question");
        egui::Window::new(self.t(
            "Măsoară micșorarea cache-ului",
            "Measure cache shrinking",
        ))
        .bounded(ctx)
        .open(&mut open)
        .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
        .default_size(fit(ctx, 680.0, 520.0))
        .collapsible(false)
        .show(ctx, |ui| {
            muted(
                ui,
                &p,
                self.t(
                    "Păstrează doar o parte din rândurile cache-ului unui text și compară răspunsul cu cel din cache-ul întreg, cu și fără refacerea valorilor. Încarcă modelul încă o dată (memorie în plus) și poate dura. Nu schimbă nimic în modelul care rulează.",
                    "Keeps only part of a text's cache rows and compares the answer with the full cache, with and without refitting the values. It loads the model a second time (extra memory) and can take a while. Nothing changes in the model that is running.",
                ),
            );
            ui.add_space(4.0);
            muted(ui, &p, &format!("{} {}", self.t("Model:", "Model:"), name));
            ui.add(
                egui::TextEdit::singleline(&mut self.kvfit.text)
                    .hint_text(hint_text)
                    .desired_width(f32::INFINITY),
            );
            ui.add(
                egui::TextEdit::singleline(&mut self.kvfit.question)
                    .hint_text(hint_q)
                    .desired_width(f32::INFINITY),
            );
            ui.horizontal(|ui| {
                ui.label(self.t("Rânduri păstrate", "Rows kept"));
                ui.add(
                    egui::Slider::new(&mut self.kvfit.budget, 0.05..=0.9)
                        .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                );
            });
            let why = crate::kvfit_ui::check_inputs(
                &self.kvfit.text,
                &self.kvfit.question,
                self.kvfit.budget,
            );
            if let Some(w) = why {
                let msg = match w {
                    "text" => self.t(
                        "Scrie calea unui fișier text existent.",
                        "Type the path of an existing text file.",
                    ),
                    "question" => self.t("Scrie întrebarea.", "Type the question."),
                    _ => self.t(
                        "Alege între 5% și 90%.",
                        "Pick between 5% and 90%.",
                    ),
                };
                muted(ui, &p, msg);
            }
            ui.horizontal(|ui| {
                ui.add_enabled_ui(why.is_none() && !self.kvfit.running, |ui| {
                    if primary_button(ui, &p, self.t("Măsoară", "Measure")).clicked() {
                        run = true;
                    }
                });
                if self.kvfit.running {
                    ui.spinner();
                }
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(&self.kvfit.output).monospace())
                        .wrap(),
                );
            });
        });
        if run {
            self.run_kvfit();
        }
        if !open {
            self.kvfit.open = false;
        }
    }

    pub fn kvfit_done(&mut self, r: Result<(bool, String), String>) {
        self.kvfit.running = false;
        self.kvfit.output = match r {
            Ok((true, s)) => s,
            Ok((false, s)) => format!("{}\n{s}", self.t("Nu a reușit:", "It did not work:")),
            Err(e) => format!(
                "{}: {e}",
                self.t("corbul nu a pornit", "corbul could not be started")
            ),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inputs_are_checked_before_anything_runs() {
        let f = std::env::temp_dir().join(format!("kvfit-ui-{}.txt", std::process::id()));
        std::fs::write(&f, "text").unwrap();
        let p = f.display().to_string();
        assert_eq!(check_inputs("", "q", 0.25), Some("text"));
        assert_eq!(check_inputs("/no/such/file", "q", 0.25), Some("text"));
        assert_eq!(check_inputs(&p, "  ", 0.25), Some("question"));
        assert_eq!(check_inputs(&p, "q", 0.0), Some("budget"));
        assert_eq!(check_inputs(&p, "q", 1.0), Some("budget"));
        assert_eq!(check_inputs(&p, "q", f32::NAN), Some("budget"));
        assert_eq!(check_inputs(&format!("\"{p}\""), "q", 0.25), None);
        let _ = std::fs::remove_file(&f);
    }
}
