//! The accessibility window, reading answers aloud, and the taskbar flash.

use crate::app::CompanionApp;
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::access::{self, Access, Preset, Targets};
use companion_core::{Theme, prompts::Style};
use egui::{Align2, RichText, vec2};
use std::process::Child;

#[derive(Default)]
pub struct AccessState {
    pub open: bool,
    /// Speech programs found on this computer, probed once.
    pub found: Option<Vec<String>>,
    pub speaking: Option<(Child, access::Speech)>,
}

fn preset_name(app: &CompanionApp, p: Preset) -> (&'static str, &'static str) {
    match p {
        Preset::LowVision => (
            app.t("Văd puțin", "Low vision"),
            app.t(
                "Text mare, contrast mare, font special, butoane mari.",
                "Big text, high contrast, a special typeface, big buttons.",
            ),
        ),
        Preset::ScreenReader => (
            app.t("Nu văd (cititor de ecran)", "Blind (screen reader)"),
            app.t(
                "Răspunsurile se citesc cu voce, fără mișcare, meniuri puține.",
                "Answers are read aloud, no movement, few menus.",
            ),
        ),
        Preset::Motor => (
            app.t("Mișcări greu de controlat", "Hard to control movements"),
            app.t(
                "Butoane foarte mari și spațiu între ele.",
                "Very big buttons with space between them.",
            ),
        ),
        Preset::Dyslexia => (
            app.t("Dislexie", "Dyslexia"),
            app.t(
                "Font lizibil, text mărit, rânduri mai rare, fără animații.",
                "A readable typeface, bigger text, looser lines, no animation.",
            ),
        ),
        Preset::Cognitive => (
            app.t(
                "Citesc greu / îmi place simplu",
                "Easier reading / keep it simple",
            ),
            app.t(
                "Meniuri puține, spațiu mult, fără animații și răspunsuri în limbaj simplu.",
                "Few menus, lots of space, no animation, and answers in plain language.",
            ),
        ),
        Preset::ColourBlind => (
            app.t("Daltonism", "Colour blindness"),
            app.t(
                "Albastru și portocaliu în loc de verde și roșu.",
                "Blue and orange instead of green and red.",
            ),
        ),
        Preset::Deaf => (
            app.t("Aud puțin sau deloc", "Deaf or hard of hearing"),
            app.t(
                "Aplicația nu folosește sunete; te anunță vizual când un răspuns este gata.",
                "The app uses no sounds; it tells you visually when an answer is ready.",
            ),
        ),
    }
}

impl CompanionApp {
    pub fn open_access(&mut self) {
        self.access.open = true;
        self.probe_speech();
    }

    fn probe_speech(&mut self) {
        if self.access.found.is_none() {
            let mut v = Vec::new();
            if std::env::consts::OS == "linux" {
                for p in ["espeak-ng", "espeak", "spd-say"] {
                    if companion_core::proc::program_exists(p) {
                        v.push(p.to_string());
                    }
                }
            }
            self.access.found = Some(v);
        }
    }

    pub fn is_speaking(&mut self) -> bool {
        if let Some((c, _)) = self.access.speaking.as_mut() {
            if matches!(c.try_wait(), Ok(None)) {
                return true;
            }
            self.access.speaking = None;
        }
        false
    }

    pub fn stop_speaking(&mut self) {
        if let Some((mut c, s)) = self.access.speaking.take() {
            access::stop(&mut c, Some(&s));
        }
    }

    pub fn speak(&mut self, markdown: &str) {
        self.stop_speaking();
        self.probe_speech();
        let found = self.access.found.clone().unwrap_or_default();
        let text = access::clean_for_speech(
            markdown,
            self.t("Aici este un bloc de cod", "A code block is here"),
        );
        if text.is_empty() {
            return;
        }
        let have = |p: &str| found.iter().any(|f| f == p);
        match access::speech_plan(
            &text,
            self.s.lang,
            self.s.access.speech_rate,
            std::env::consts::OS,
            &have,
        ) {
            Some(plan) => match access::start(&plan) {
                Ok(child) => self.access.speaking = Some((child, plan)),
                Err(e) => {
                    let m = format!(
                        "{}: {e}",
                        self.t("Nu pot citi cu voce", "Cannot read aloud")
                    );
                    self.toast(m);
                }
            },
            None => self.toast(self.t(
                "Nu a fost găsit un program de citire cu voce. Pe Linux, instalează espeak-ng.",
                "No speech program found. On Linux, install espeak-ng.",
            )),
        }
    }

    /// Called when an answer is complete.
    pub fn answer_ready(&mut self, text: &str) {
        if self.s.access.read_aloud_auto {
            self.speak(text);
        }
        if self.s.access.flash_when_done
            && !self.ctx.input(|i| i.viewport().focused.unwrap_or(true))
        {
            self.ctx
                .send_viewport_cmd(egui::ViewportCommand::RequestUserAttention(
                    egui::UserAttentionType::Informational,
                ));
        }
    }

    fn apply_preset(&mut self, p: Preset) {
        let (a, scale, theme) = Access::preset(p);
        self.s.access = a;
        self.s.font_scale = scale;
        if let Some(t) = theme {
            self.s.theme = t;
        } else if self.s.theme == Theme::HighContrast {
            self.s.theme = Theme::Light;
        }
        if p == Preset::Cognitive {
            self.prompts.style = Style::Simple;
        }
        let _ = self.s.save();
    }

    pub fn access_window(&mut self, ctx: &egui::Context) {
        if !self.access.open {
            return;
        }
        let p = self.p;
        let mut open = true;
        let before = self.s.clone();
        egui::Window::new(self.t("Accesibilitate", "Accessibility"))
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_size(fit(ctx, 680.0, 640.0))
            .collapsible(false)
            .vscroll(true)
            .show(ctx, |ui| {
                muted(ui, &p, self.t(
                    "Alege ce ți se potrivește. Totul se modifică pe loc și poate fi anulat oricând. Poți combina oricâte opțiuni.",
                    "Pick what suits you. Everything changes at once and can be undone any time. You can combine as many options as you like.",
                ));
                ui.add_space(6.0);
                ui.label(RichText::new(self.t("Pornire rapidă", "Quick start")).semibold());
                for pr in Preset::ALL {
                    let (name, what) = preset_name(self, pr);
                    ui.horizontal_wrapped(|ui| {
                        if ui.button(name).clicked() {
                            self.apply_preset(pr);
                        }
                        muted(ui, &p, what);
                    });
                }
                if ui.button(self.t("Înapoi la setările obișnuite", "Back to the usual settings")).clicked() {
                    self.s.access = Access::default();
                    self.s.font_scale = 1.1;
                    self.s.theme = Theme::Light;
                    self.prompts.style = Style::Normal;
                }
                ui.separator();

                ui.label(RichText::new(self.t("Vedere", "Seeing")).semibold());
                ui.horizontal(|ui| {
                    let (l, d, h) = (self.t("Luminos", "Light"), self.t("Întunecat", "Dark"), self.t("Contrast mare", "High contrast"));
                    ui.selectable_value(&mut self.s.theme, Theme::Light, l);
                    ui.selectable_value(&mut self.s.theme, Theme::Dark, d);
                    ui.selectable_value(&mut self.s.theme, Theme::HighContrast, h);
                });
                ui.horizontal(|ui| {
                    ui.label(self.t("Mărimea textului", "Text size"));
                    ui.add(egui::Slider::new(&mut self.s.font_scale, 0.8..=3.0).show_value(false));
                    ui.label(format!("{:.0}%", self.s.font_scale * 100.0));
                });
                ui.horizontal(|ui| {
                    ui.label(self.t("Spațiu între rânduri și butoane", "Space between lines and buttons"));
                    ui.add(egui::Slider::new(&mut self.s.access.spacing, 1.0..=2.0).show_value(false));
                });
                let l = self.t("Font creat pentru vedere slabă (Atkinson Hyperlegible)", "Typeface made for low vision (Atkinson Hyperlegible)");
                ui.checkbox(&mut self.s.access.readable_font, l);
                let l = self.t("Culori pentru daltonism (albastru și portocaliu în loc de verde și roșu)", "Colours for colour blindness (blue and orange instead of green and red)");
                ui.checkbox(&mut self.s.access.colour_blind, l);
                let l = self.t("Contur gros în jurul elementului selectat cu tastatura", "A thick outline around the item the keyboard is on");
                ui.checkbox(&mut self.s.access.strong_focus, l);
                ui.add_space(6.0);

                ui.label(RichText::new(self.t("Citire cu voce", "Reading aloud")).semibold());
                let l = self.t("Citește automat fiecare răspuns, când este gata", "Read every answer aloud by itself when it is done");
                ui.checkbox(&mut self.s.access.read_aloud_auto, l);
                ui.horizontal(|ui| {
                    ui.label(self.t("Viteză", "Speed"));
                    let unit = self.t(" cuv./min", " wpm");
                    ui.add(egui::Slider::new(&mut self.s.access.speech_rate, 100..=300).suffix(unit));
                });
                ui.horizontal(|ui| {
                    if ui.button(self.t("Încearcă vocea", "Try the voice")).clicked() {
                        let t = self.t("Bună ziua! Așa sună vocea. Poți modifica viteza.", "Hello! This is how the voice sounds. You can change the speed.");
                        self.speak(t);
                    }
                    if self.is_speaking() && ui.button(self.t("Oprește citirea", "Stop reading")).clicked() {
                        self.stop_speaking();
                    }
                });
                let none = self.access.found.as_ref().is_some_and(|f| f.is_empty()) && std::env::consts::OS == "linux";
                if none {
                    ui.label(RichText::new(self.t(
                        "Nu a fost găsit niciun program de voce. Instalează espeak-ng (de ex. «sudo apt install espeak-ng»).",
                        "No speech program found. Install espeak-ng (for example “sudo apt install espeak-ng”).",
                    )).color(p.warn));
                }
                muted(ui, &p, self.t(
                    "Se utilizează vocea sistemului tău; pe Windows și macOS, limba vocii este cea aleasă în sistem. Fiecare răspuns are și un buton «Citește». Scurtătură: Ctrl+Shift+S.",
                    "It uses your system's own voice; on Windows and macOS the voice language is the one set in the system. Every answer also has a “Read aloud” button. Shortcut: Ctrl+Shift+S.",
                ));
                ui.add_space(6.0);

                ui.label(RichText::new(self.t("Mișcare și mâini", "Movement and hands")).semibold());
                ui.horizontal(|ui| {
                    ui.label(self.t("Mărimea butoanelor", "Button size"));
                    let (a, b, c) = (self.t("Normal", "Normal"), self.t("Mare", "Large"), self.t("Foarte mare", "Huge"));
                    ui.selectable_value(&mut self.s.access.targets, Targets::Normal, a);
                    ui.selectable_value(&mut self.s.access.targets, Targets::Large, b);
                    ui.selectable_value(&mut self.s.access.targets, Targets::Huge, c);
                });
                muted(ui, &p, self.t(
                    "Aplicația poate fi utilizată doar cu tastatura: Tab și Shift+Tab trec de la un element la altul, Spațiu sau Enter acționează, Esc oprește un răspuns. Nu sunt necesare mouse-ul, tragerea sau mișcări precise. Scurtături: Ctrl+N conversație nouă, Ctrl+, setări, Ctrl+P prompturi, Ctrl+K căutare, Ctrl+Shift+A această fereastră, Ctrl+Shift+S citește ultimul răspuns.",
                    "The app can be used with the keyboard alone: Tab and Shift+Tab move between items, Space or Enter presses, Esc stops an answer. No mouse, dragging or precise movement is needed. Shortcuts: Ctrl+N new chat, Ctrl+, settings, Ctrl+P prompts, Ctrl+K search, Ctrl+Shift+A this window, Ctrl+Shift+S read the last answer.",
                ));
                muted(ui, &p, self.t(
                    "Pentru a dicta în loc să scrii, utilizează dictarea sistemului: Windows Win+H, macOS Fn de două ori, Linux în funcție de mediu. Funcționează în orice câmp de text de aici.",
                    "To dictate instead of typing, use your system's dictation: Windows Win+H, macOS press Fn twice, Linux depends on the desktop. It works in any text box here.",
                ));
                ui.add_space(6.0);

                ui.label(RichText::new(self.t("Auz", "Hearing")).semibold());
                let l = self.t("Clipește bara de activități când un răspuns e gata și fereastra e în fundal", "Flash the taskbar when an answer is ready and the window is in the background");
                ui.checkbox(&mut self.s.access.flash_when_done, l);
                muted(ui, &p, self.t(
                    "Aplicația nu scoate niciun sunet și nu cere să asculți nimic: tot ce apare e text.",
                    "The app makes no sound and never asks you to listen: everything is shown as text.",
                ));
                ui.add_space(6.0);

                ui.label(RichText::new(self.t("Citit și înțeles", "Reading and understanding")).semibold());
                let l = self.t("Fără animații și fără rotițe care se învârt", "No animation and no spinning wheels");
                ui.checkbox(&mut self.s.access.reduce_motion, l);
                let l = self.t("Meniuri simple: ascunde ferestrele avansate", "Simple menus: hide the advanced windows");
                ui.checkbox(&mut self.s.access.simple_menus, l);
                ui.horizontal_wrapped(|ui| {
                    let on = self.prompts.style == Style::Simple;
                    if ui.selectable_label(on, self.t("Răspunsuri în limbaj simplu", "Answers in plain language")).clicked() {
                        self.prompts.style = if on { Style::Normal } else { Style::Simple };
                    }
                    muted(ui, &p, self.t(
                        "Propoziții scurte, cuvinte obișnuite, o idee pe propoziție. Modelul o respectă cât poate.",
                        "Short sentences, everyday words, one idea per sentence. The model follows it as well as it can.",
                    ));
                });
                ui.add_space(6.0);

                ui.label(RichText::new(self.t("Cititoare de ecran", "Screen readers")).semibold());
                muted(ui, &p, self.t(
                    "Aplicația își descrie conținutul sistemului (prin AccessKit), astfel încât NVDA, JAWS, Narrator, VoiceOver și Orca pot citi butoanele și textele. Aplicația nu a putut fi încă testată cu un cititor de ecran real: dacă ceva nu se citește corect, spune-ne ce, iar noi vom corecta. Alternativ, butonul «Citește» de mai sus nu necesită cititor de ecran.",
                    "The app describes its contents to the system (through AccessKit), so NVDA, JAWS, Narrator, VoiceOver and Orca can read the buttons and texts. I have not yet been able to try the app with a real screen reader: if something is not read well, tell me what and I will fix it. The “Read aloud” button above does not need a screen reader.",
                ));
                ui.add_space(4.0);
                muted(ui, &p, self.t(
                    "Ajutor din sistem: Windows Narrator Win+Ctrl+Enter, Lupă Win+Plus; macOS VoiceOver Cmd+F5, Zoom Opțiune+Cmd+8; Linux Orca Super+Alt+S.",
                    "System help: Windows Narrator Win+Ctrl+Enter, Magnifier Win+Plus; macOS VoiceOver Cmd+F5, Zoom Option+Cmd+8; Linux Orca Super+Alt+S.",
                ));
            });
        self.access.open = open;
        if self.s != before {
            let _ = self.s.save();
        }
    }
}
