//! Top bar, status bar, settings, about, logs and the little pop-up notes.

use crate::app::{CompanionApp, Phase};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::chat::Health;
use companion_core::{Lang, Theme, locate, paths, proc};
use egui::{Align2, CornerRadius, Frame, Margin, Panel, RichText, Ui, vec2};

const THIRD_PARTY: &str = include_str!("../../../THIRD_PARTY_LICENSES.md");

impl CompanionApp {
    pub fn top_bar(&mut self, ui: &mut Ui) {
        let p = self.p;
        Panel::top("topbar")
            .frame(
                Frame::new()
                    .fill(p.panel)
                    .inner_margin(Margin::symmetric(14, 8)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Colibri Companion")
                            .size(19.0)
                            .semibold()
                            .color(p.accent),
                    );
                    ui.add_space(8.0);
                    let (text, fg, bg) = self.state_pill();
                    pill(ui, &text, fg, bg);
                    right_to_left(ui, |ui| {
                        // The language is always one click away, written in its own name.
                        let before = self.s.lang;
                        egui::ComboBox::from_id_salt("top-language")
                            .selected_text(self.s.lang.native_name())
                            .width(110.0)
                            .height(420.0)
                            .show_ui(ui, |ui| {
                                for l in Lang::ALL {
                                    ui.selectable_value(&mut self.s.lang, l, l.native_name());
                                }
                            })
                            .response
                            .on_hover_text(self.t("Limba", "Language"));
                        if self.s.lang != before {
                            let _ = self.s.save();
                        }
                        if ui.button(self.t("Resurse", "Resources")).clicked() {
                            self.set_resources_open(true);
                        }
                        ui.menu_button(self.t("Mai multe ▾", "More ▾"), |ui| {
                            for (label, which) in [
                                (self.t("Programe locale", "Local programs"), 0),
                                (self.t("Compară modele", "Compare models"), 1),
                                (self.t("Explorează", "Explore"), 2),
                                (self.t("Sfaturi", "Tips"), 3),
                                (self.t("Legal", "Legal"), 4),
                                (self.t("Prompturi", "Prompts"), 5),
                                (self.t("Conexiuni online", "Online connections"), 6),
                                (self.t("Catalog de platforme", "Platform catalogue"), 7),
                                (self.t("Modele creative", "Creative models"), 8),
                                (self.t("Automatizări", "Automations"), 9),
                                (
                                    self.t("Memorie și constelație", "Memory and constellation"),
                                    10,
                                ),
                                (self.t("Dueluri între modele", "Model duels"), 11),
                                (self.t("Ghid de început", "Getting started"), 12),
                                (self.t("Documente", "Documents"), 13),
                                (self.t("Unelte MCP", "MCP tools"), 14),
                            ] {
                                if self.s.access.simple_menus
                                    && matches!(which, 0 | 1 | 6 | 7 | 8 | 9 | 11 | 14)
                                {
                                    continue;
                                }
                                if ui.button(label).clicked() {
                                    match which {
                                        0 => self.sheets.local = true,
                                        1 => self.sheets.compare = true,
                                        2 => self.guide.explore = true,
                                        3 => self.guide.tips = true,
                                        5 => self.open_prompts(),
                                        6 => self.open_online(),
                                        7 => self.open_catalog(),
                                        8 => self.open_studio(),
                                        9 => self.open_flows(),
                                        10 => self.open_memory(),
                                        11 => self.open_duel(),
                                        12 => self.guide.tour = Some(0),
                                        13 => self.open_docs(),
                                        14 => self.open_mcp(),
                                        _ => self.guide.legal = true,
                                    }
                                    ui.close();
                                }
                            }
                        });
                        if ui.button(self.t("Despre", "About")).clicked() {
                            self.show_about = true;
                        }
                        if ui.button(self.t("Setări", "Settings")).clicked() {
                            self.show_settings = true;
                        }
                        if ui
                            .button(self.t("Accesibilitate", "Accessibility"))
                            .clicked()
                        {
                            self.open_access();
                        }
                        if self.coli.is_some() && !self.demo {
                            if self.server_ready()
                                && ui
                                    .button(self.t("Oprește modelul", "Stop the model"))
                                    .clicked()
                                && !self.tune_blocks_server()
                            {
                                self.stop_server();
                            }
                            if ui.button(self.t("Modele", "Models")).clicked() {
                                self.show_models = true;
                                self.refresh_installed();
                                self.refresh_engine_info();
                            }
                        }
                        if let Some(msg) = self.busy.clone() {
                            ui.label(RichText::new(msg).color(p.muted));
                            ui.add(egui::Spinner::new());
                        }
                    });
                });
            });
    }

    /// Escape closes the open tool windows, one at a time (when no text box has the keyboard).
    pub fn close_one_window(&mut self) {
        // The welcome tour is closed by Esc too, and counts as seen.
        if self.guide.tour.take().is_some() {
            let _ = self.s.save();
            return;
        }
        let flags: [&mut bool; 18] = [
            &mut self.guide.tips,
            &mut self.guide.explore,
            &mut self.guide.legal,
            &mut self.guide.shortcuts,
            &mut self.show_about,
            &mut self.sheets.local,
            &mut self.sheets.compare,
            &mut self.flows.open,
            &mut self.memory.open,
            &mut self.docs.open,
            &mut self.mcp.open,
            &mut self.studio.open,
            &mut self.duel.open,
            &mut self.access.open,
            &mut self.prompts.open,
            &mut self.online.open_catalog,
            &mut self.online.open_conn,
            &mut self.show_settings,
        ];
        for f in flags {
            if *f {
                *f = false;
                return;
            }
        }
    }

    fn state_pill(&self) -> (String, egui::Color32, egui::Color32) {
        let p = self.p;
        match &self.health {
            Health::Ready(_) if self.demo => (
                self.t("Demonstrație (fără model real)", "Demo (no real model)")
                    .into(),
                p.warn,
                p.warn_soft,
            ),
            Health::Ready(_) => (
                format!(
                    "{} · {}",
                    self.t("Pregătit", "Ready"),
                    self.model_id.clone().unwrap_or_else(|| "colibri".into())
                ),
                p.ok,
                p.accent_soft,
            ),
            Health::Loading(_) => (
                self.t("Se încarcă modelul…", "Loading the model…").into(),
                p.warn,
                p.warn_soft,
            ),
            Health::Down if self.phase() == Phase::Installing => (
                self.t("Se instalează…", "Installing…").into(),
                p.warn,
                p.warn_soft,
            ),
            Health::Down => (
                self.t("Modelul nu rulează", "Model not running").into(),
                p.muted,
                p.panel,
            ),
        }
    }

    pub fn status_bar(&mut self, ui: &mut Ui) {
        let p = self.p;
        Panel::bottom("statusbar")
            .frame(
                Frame::new()
                    .fill(p.panel)
                    .inner_margin(Margin::symmetric(14, 4)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let base = self.base();
                    let local = companion_core::net::is_local_base(&base);
                    let privacy = if local {
                        self.t(
                            "Totul rămâne pe acest calculator",
                            "Everything stays on this computer",
                        )
                        .to_string()
                    } else {
                        format!(
                            "{}: {}",
                            self.t("Server la distanță", "Remote server"),
                            base
                        )
                    };
                    if ui
                        .link(RichText::new(format!("🔒 {privacy}")).small())
                        .on_hover_text(self.t(
                            "Vezi resursele utilizate și dovada că nimic nu iese din calculator",
                            "See the resources in use and the proof that nothing leaves your computer",
                        ))
                        .clicked()
                    {
                        self.set_resources_open(true);
                    }
                    right_to_left(ui, |ui| {
                        if let Some(f) = self.facts() {
                            let mut bits = Vec::new();
                            if let Some(c) = f.context_tokens {
                                bits.push(format!(
                                    "{} {c} {}",
                                    self.t("context", "context"),
                                    self.t("tokeni", "tokens")
                                ));
                            }
                            if let Some(r) = f.ram_used_gb {
                                bits.push(format!("RAM {r:.1} GB"));
                            }
                            ui.label(RichText::new(bits.join("  ·  ")).small().color(p.muted));
                        }
                    });
                });
            });
    }

    pub fn toasts(&mut self, ctx: &egui::Context) {
        let p = self.p;
        for (i, t) in self.toasts.iter().enumerate() {
            egui::Area::new(egui::Id::unique(("toast", i)))
                .anchor(Align2::CENTER_BOTTOM, vec2(0.0, -70.0 - 46.0 * i as f32))
                .show(ctx, |ui| {
                    Frame::new()
                        .fill(p.text)
                        .corner_radius(CornerRadius::same(10))
                        .inner_margin(Margin::symmetric(16, 10))
                        .show(ui, |ui| {
                            ui.label(RichText::new(&t.text).color(p.bg));
                        });
                });
        }
    }

    pub fn windows(&mut self, ctx: &egui::Context) {
        self.settings_window(ctx);
        self.models_window(ctx);
        self.resources_window(ctx);
        self.guide_windows(ctx);
        self.prompt_window(ctx);
        self.online_windows(ctx);
        self.studio_window(ctx);
        self.duel_window(ctx);
        self.access_window(ctx);
        self.flows_window(ctx);
        self.memory_window(ctx);
        self.docs_window(ctx);
        self.mcp_windows(ctx);
        self.sheet_windows(ctx);
        self.about_window(ctx);
        self.logs_window(ctx);
    }

    fn settings_window(&mut self, ctx: &egui::Context) {
        if !self.show_settings {
            return;
        }
        let mut open = true;
        let before = self.s.clone();
        egui::Window::new(self.t("Setări", "Settings")).open(&mut open).anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0)).default_width(600.0).default_height(560.0).collapsible(false).vscroll(true).show(ctx, |ui| {
            let p = self.p;
            ui.label(RichText::new(self.t("Limbă", "Language")).semibold());
            ui.horizontal_wrapped(|ui| {
                for l in Lang::ALL {
                    ui.selectable_value(&mut self.s.lang, l, l.native_name());
                }
            });
            ui.add_space(6.0);
            ui.label(RichText::new(self.t("Aspect", "Appearance")).semibold());
            ui.horizontal(|ui| {
                let (l, d, h) = (self.t("Luminos", "Light"), self.t("Întunecat", "Dark"), self.t("Contrast mare", "High contrast"));
                ui.selectable_value(&mut self.s.theme, Theme::Light, l);
                ui.selectable_value(&mut self.s.theme, Theme::Dark, d);
                ui.selectable_value(&mut self.s.theme, Theme::HighContrast, h);
            });
            ui.horizontal(|ui| {
                ui.label(self.t("Mărimea textului", "Text size"));
                ui.add(egui::Slider::new(&mut self.s.font_scale, 0.8..=1.9).show_value(false));
                ui.label(format!("{:.0}%", self.s.font_scale * 100.0));
                if ui.button(self.t("Normal", "Reset")).clicked() {
                    self.s.font_scale = 1.1;
                }
            });
            ui.separator();
            ui.label(RichText::new(self.t("Cum răspunde modelul", "How the model answers")).semibold());
            let lbl = self.t("Arată «gândirea» modelului, când există", "Show the model's “thinking”, when there is any");
            ui.checkbox(&mut self.s.show_thinking, lbl);
            ui.label(self.t("Instrucțiuni permanente (opțional), de ex. «Răspunde scurt și în română»:", "Standing instructions (optional), e.g. “Answer briefly”:"));
            ui.add(egui::TextEdit::multiline(&mut self.s.system_prompt).desired_rows(3).desired_width(f32::INFINITY));
            let mut limit = self.s.max_tokens.is_some();
            if ui.checkbox(&mut limit, self.t("Limitează lungimea unui răspuns", "Limit the length of an answer")).changed() {
                self.s.max_tokens = limit.then_some(1024);
            }
            let tok = self.t("tokeni", "tokens");
            if let Some(n) = self.s.max_tokens.as_mut() {
                ui.add(egui::Slider::new(n, 64..=8192).text(tok));
            }
            ui.separator();
            ui.label(RichText::new(self.t("Server", "Server")).semibold());
            ui.label(RichText::new(self.t(
                "Lasă adresa implicită pentru a utiliza colibri instalat de aici. O poți modifica pentru orice server compatibil OpenAI (doar http://).",
                "Leave the default to use the colibri installed from here. Change it for any OpenAI-compatible server (http:// only).",
            )).color(p.muted));
            ui.horizontal(|ui| {
                ui.label(self.t("Adresă", "Address"));
                ui.add(egui::TextEdit::singleline(&mut self.s.server_url).desired_width(260.0));
            });
            ui.horizontal(|ui| {
                ui.label(self.t("Cheie API", "API key"));
                let hint = self.t("(opțional)", "(optional)");
                ui.add(egui::TextEdit::singleline(&mut self.s.api_key).password(true).desired_width(260.0).hint_text(hint));
            });
            ui.label(match &self.health {
                Health::Ready(_) => RichText::new(self.t("✔ Conectat", "✔ Connected")).color(p.ok),
                Health::Loading(_) => RichText::new(self.t("Serverul răspunde, dar se încarcă", "The server answers but is still loading")).color(p.warn),
                Health::Down => RichText::new(self.t("Nu răspunde nimeni la această adresă", "Nothing answers at this address")).color(p.muted),
            });
            let auto = self.t("Pornește modelul automat la deschiderea aplicației", "Start the model automatically when the app opens");
            ui.checkbox(&mut self.s.start_server_automatically, auto);
            let upd = self.t(
                "Verifică o dată pe zi dacă există versiuni noi (o cerere mică către GitHub)",
                "Check once a day for new versions (one small request to GitHub)",
            );
            ui.checkbox(&mut self.s.check_updates, upd);
            ui.horizontal(|ui| {
                if secondary_button(ui, self.t("Verifică acum", "Check now")).clicked() {
                    self.check_updates(true);
                }
                if self.upd.checking {
                    ui.add(egui::Spinner::new());
                }
            });
            ui.separator();
            ui.label(RichText::new(self.t("Foldere", "Folders")).semibold());
            let mut root = self.s.colibri_root.as_ref().map(|p| p.display().to_string()).unwrap_or_default();
            ui.horizontal(|ui| {
                ui.label("colibri");
                if ui.add(egui::TextEdit::singleline(&mut root).desired_width(300.0).hint_text(self.t("(găsit automat)", "(found automatically)"))).changed() {
                    self.s.colibri_root = (!root.trim().is_empty()).then(|| root.trim().into());
                }
                if ui.button(self.t("Caută din nou", "Search again")).clicked() {
                    self.detect_environment();
                }
            });
            let mut models = self.s.models_dir.as_ref().map(|p| p.display().to_string()).unwrap_or_default();
            ui.horizontal(|ui| {
                ui.label(self.t("Modele", "Models"));
                if ui.add(egui::TextEdit::singleline(&mut models).desired_width(300.0).hint_text("~/colibri-models")).changed() {
                    self.s.models_dir = (!models.trim().is_empty()).then(|| models.trim().into());
                }
            });
            ui.horizontal(|ui| {
                if ui.button(self.t("Deschide folderul conversațiilor", "Open the conversations folder")).clicked() {
                    let dir = paths::conversations_dir();
                    let _ = std::fs::create_dir_all(&dir);
                    proc::open_externally(&dir.display().to_string());
                }
            });
        });
        if self.s != before {
            let _ = self.s.save();
        }
        self.show_settings = open;
    }

    fn about_window(&mut self, ctx: &egui::Context) {
        if !self.show_about {
            return;
        }
        let mut open = true;
        egui::Window::new(self.t("Despre Colibri Companion", "About Colibri Companion")).open(&mut open).anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0)).default_width(620.0).collapsible(false).vscroll(true).show(ctx, |ui| {
            let p = self.p;
            ui.label(RichText::new(format!("Colibri Companion {}", env!("CARGO_PKG_VERSION"))).size(20.0).semibold());
            ui.label(self.t(
                "O interfață prietenoasă, neoficială, pentru motorul colibri. Nu este creată, susținută sau aprobată de autorii colibri.",
                "A friendly, unofficial interface for the colibri engine. It is not made, supported or endorsed by colibri's authors.",
            ));
            ui.add_space(6.0);
            ui.label(RichText::new(self.t("Licențe", "Licences")).semibold());
            ui.label(self.t(
                "• Această aplicație: Apache License 2.0 (fișierele LICENSE și NOTICE).\n• colibri (proiect separat): Apache License 2.0. Aplicația nu îi conține și nu îi modifică codul; îl pornește și comunică prin API-ul lui public.\n• Modelele au licențele lor proprii, afișate înainte de descărcare.\n• Bibliotecile folosite sunt listate mai jos.",
                "• This app: Apache License 2.0 (see LICENSE and NOTICE).\n• colibri (a separate project): Apache License 2.0. This app contains and modifies none of its code; it runs it and talks to its public API.\n• Models carry their own licences, shown before download.\n• The libraries used are listed below.",
            ));
            ui.horizontal(|ui| {
                if ui.link(self.t("colibri pe GitHub", "colibri on GitHub")).clicked() {
                    proc::open_externally(locate::COLIBRI_GIT_URL);
                }
                if ui.link("Apache-2.0").clicked() {
                    proc::open_externally("https://www.apache.org/licenses/LICENSE-2.0");
                }
            });
            ui.add_space(6.0);
            ui.label(self.t("Fonturi incluse: Inter și JetBrains Mono, sub licența SIL Open Font License 1.1.", "Bundled fonts: Inter and JetBrains Mono, under the SIL Open Font License 1.1."));
            ui.add_space(4.0);
            ui.label(RichText::new(self.t("Biblioteci terțe", "Third-party libraries")).semibold());
            // Only the table of crates: the explanation around it is already said above, in this language.
            let table = THIRD_PARTY.find("| Crate").map(|i| &THIRD_PARTY[i..]).unwrap_or(THIRD_PARTY);
            egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                ui.label(RichText::new(table).monospace().size(12.0).color(p.muted));
            });
        });
        self.show_about = open;
    }

    fn logs_window(&mut self, ctx: &egui::Context) {
        let Some(text) = self.logs_view.clone() else {
            return;
        };
        let mut open = true;
        egui::Window::new(self.t("Jurnal", "Log"))
            .open(&mut open)
            .default_width(640.0)
            .show(ctx, |ui| {
                if ui.button(self.t("Copiază", "Copy")).clicked() {
                    ui.ctx().copy_text(text.clone());
                }
                egui::ScrollArea::vertical()
                    .max_height(360.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(if text.trim().is_empty() {
                                self.t("(gol)", "(empty)")
                            } else {
                                &text
                            })
                            .monospace()
                            .size(12.0),
                        );
                    });
            });
        if !open {
            self.logs_view = None;
        }
    }
}
