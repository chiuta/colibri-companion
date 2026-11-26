//! Top bar, status bar, settings, about, logs and the little pop-up notes.

use crate::app::{CompanionApp, Phase};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::chat::Health;
use companion_core::{Lang, Theme, locate, paths, proc};
use egui::{Align2, CornerRadius, Frame, Margin, Panel, RichText, Ui, vec2};

const THIRD_PARTY: &str = include_str!("../../../THIRD_PARTY_LICENSES.md");

/// A menu entry: Romanian label, English label, which tool it opens.
type MenuItem = (&'static str, &'static str, u8);
/// A titled group of menu entries.
type MenuGroup = (&'static str, &'static str, &'static [MenuItem]);

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
                // Below this width the less used buttons move into the "More" menu, so the bar never
                // overlaps itself (long words in German or Hungarian need more room than English).
                let narrow = ui.available_width() < 1380.0;
                // The buttons on the right keep their room; the title, the state and the status
                // message share what is left and are cut with "…" when it is not enough, so the two
                // sides can never be drawn over each other, whatever the language or the window size.
                // The width the buttons took is known from the previous frame.
                let right_id = ui.scope_id().with("topbar-right-width");
                let right_w: f32 = ui.data(|d| d.get_temp(right_id)).unwrap_or(0.0);
                let mut right_now = right_w;
                ui.horizontal(|ui| {
                    let left_w = (ui.available_width() - right_w - 16.0).max(100.0);
                    ui.allocate_ui_with_layout(
                        vec2(left_w, 0.0),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            ui.add(
                                egui::Label::new(
                                    RichText::new("Colibri Companion")
                                        .size(19.0)
                                        .semibold()
                                        .color(p.accent),
                                )
                                .truncate(),
                            );
                            ui.add_space(8.0);
                            let (text, fg, bg) = self.state_pill();
                            pill(ui, &text, fg, bg);
                            if let Some(msg) = self.busy.clone() {
                                // on a narrow bar the spinner alone says "working"
                                if left_w >= 340.0 {
                                    ui.add(
                                        egui::Label::new(RichText::new(msg).color(p.muted))
                                            .truncate(),
                                    );
                                }
                                crate::widgets::spinner(ui);
                            }
                        },
                    );
                    let right =
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
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
                            if !narrow && ui.button(self.t("Resurse", "Resources")).clicked() {
                                self.set_resources_open(true);
                            }
                            ui.menu_button(self.t("Mai multe ▾", "More ▾"), |ui| {
                                if narrow {
                                    if ui.button(self.t("Resurse", "Resources")).clicked() {
                                        self.set_resources_open(true);
                                        ui.close();
                                    }
                                    if ui
                                        .button(self.t("Accesibilitate", "Accessibility"))
                                        .clicked()
                                    {
                                        self.open_access();
                                        ui.close();
                                    }
                                    if ui.button(self.t("Despre", "About")).clicked() {
                                        self.show_about = true;
                                        ui.close();
                                    }
                                    ui.separator();
                                }
                                // grouped by what the person is doing; a heading and a line
                                // between groups, the most used entries first
                                let groups: [MenuGroup; 5] = [
                                    (
                                        "Primii pași",
                                        "First steps",
                                        &[
                                            ("Ghid de început", "Getting started", 12),
                                            ("Explorează", "Explore", 2),
                                            ("Sfaturi", "Tips", 3),
                                        ],
                                    ),
                                    (
                                        "Munca ta",
                                        "Your work",
                                        &[
                                            ("Documente", "Documents", 13),
                                            ("Prompturi", "Prompts", 5),
                                            ("Unelte MCP", "MCP tools", 14),
                                            ("Automatizări", "Automations", 9),
                                            (
                                                "Memorie și constelație",
                                                "Memory and constellation",
                                                10,
                                            ),
                                        ],
                                    ),
                                    (
                                        "Modele",
                                        "Models",
                                        &[
                                            ("Compară modele", "Compare models", 1),
                                            ("Dueluri între modele", "Model duels", 11),
                                            ("Modele creative", "Creative models", 8),
                                            ("Catalog de platforme", "Platform catalogue", 7),
                                        ],
                                    ),
                                    (
                                        "Conexiuni",
                                        "Connections",
                                        &[
                                            ("Conexiuni online", "Online connections", 6),
                                            ("Programe locale", "Local programs", 0),
                                        ],
                                    ),
                                    ("Informații", "Information", &[("Legal", "Legal", 4)]),
                                ];
                                let mut first_group = true;
                                for (g_ro, g_en, items) in groups {
                                    let shown: Vec<_> = items
                                        .iter()
                                        .filter(|(_, _, which)| {
                                            !(self.s.access.simple_menus
                                                && matches!(which, 0 | 1 | 6 | 7 | 8 | 9 | 11 | 14))
                                        })
                                        .collect();
                                    if shown.is_empty() {
                                        continue;
                                    }
                                    if !first_group {
                                        ui.separator();
                                    }
                                    first_group = false;
                                    ui.label(
                                        RichText::new(self.t(g_ro, g_en))
                                            .small()
                                            .color(self.p.muted),
                                    );
                                    for (l_ro, l_en, which) in shown {
                                        let label = self.t(l_ro, l_en);
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
                                }
                            });
                            if !narrow && ui.button(self.t("Despre", "About")).clicked() {
                                self.show_about = true;
                            }
                            if ui.button(self.t("Setări", "Settings")).clicked() {
                                self.show_settings = true;
                            }
                            if !narrow
                                && ui
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
                        });
                    right_now = right.response.rect.width();
                });
                if (right_now - right_w).abs() > 1.0 {
                    ui.data_mut(|d| d.insert_temp(right_id, right_now));
                    ui.ctx().request_repaint();
                }
            });
    }

    /// Escape closes the open tool windows, one at a time (when no text box has the keyboard).
    /// Whether Esc has a window to close.
    pub fn has_open_window(&self) -> bool {
        self.guide.tour.is_some()
            || [
                self.guide.tips,
                self.guide.explore,
                self.guide.legal,
                self.guide.shortcuts,
                self.show_about,
                self.sheets.local,
                self.sheets.compare,
                self.flows.open,
                self.memory.open,
                self.docs.open,
                self.mcp.open,
                self.studio.open,
                self.duel.open,
                self.access.open,
                self.prompts.open,
                self.online.open_catalog,
                self.online.open_conn,
                self.show_settings,
            ]
            .contains(&true)
    }

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
                                bits.push(crate::widgets::loc(format!("RAM {r:.1} GB")));
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
        egui::Window::new(self.t("Setări", "Settings")).bounded(ctx).open(&mut open).anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0)).default_width(fit(ctx, 600.0, 0.0).x).default_height(fit(ctx, 0.0, 560.0).y).collapsible(false).vscroll(true).show(ctx, |ui| {
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
                    crate::widgets::spinner(ui);
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
        egui::Window::new(self.t("Despre Colibri Companion", "About Colibri Companion")).bounded(ctx).open(&mut open).anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0)).default_width(fit(ctx, 620.0, 0.0).x).collapsible(false).vscroll(true).show(ctx, |ui| {
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
            .bounded(ctx)
            .open(&mut open)
            .default_width(fit(ctx, 640.0, 0.0).x)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widgets::testkit;
    use companion_core::chat::ServerFacts;
    use companion_core::locate::PythonCmd;
    use companion_core::store::Settings;

    /// The widest the bar gets: a model is running (Stop and Models buttons), its name is long,
    /// and a status message with a spinner is showing.
    fn busy_app(lang: Lang) -> CompanionApp {
        let s = Settings {
            lang,
            ..Settings::default()
        };
        let mut a = CompanionApp::new(
            egui::Context::default(),
            s,
            Some("http://127.0.0.1:1".into()),
            false,
        );
        a.coli = Some(companion_core::coli::Coli::new(
            std::env::temp_dir(),
            PythonCmd {
                program: "python3".into(),
                pre_args: vec![],
                version: (3, 12),
            },
        ));
        a.demo = false;
        a.health = Health::Ready(ServerFacts::default());
        a.model_id = Some("qwen3.5-4b-instruct-q4_k_m".into());
        a.busy = Some(a.t("Se descarcă modelul…", "Downloading the model…").into());
        a
    }

    /// Text drawn by the top bar at `width`, after the layout has settled (second frame).
    fn bar(a: &mut CompanionApp, width: f32) -> Vec<testkit::Drawn> {
        let ctx = a.ctx.clone();
        testkit::draw(&ctx, width, 200.0, |ui| a.top_bar(ui));
        testkit::draw(&ctx, width, 200.0, |ui| a.top_bar(ui))
    }

    #[test]
    fn the_top_bar_never_overlaps_or_clips_in_any_language_at_any_width() {
        let mut problems = Vec::new();
        for lang in Lang::ALL {
            for width in [
                760.0_f32, 900.0, 1100.0, 1280.0, 1379.0, 1380.0, 1400.0, 1920.0,
            ] {
                let mut a = busy_app(lang);
                let drawn = bar(&mut a, width);
                assert!(drawn.len() >= 4, "{lang:?} {width}: the bar drew {drawn:?}");
                for (x, y) in testkit::overlapping(&drawn) {
                    problems.push(format!(
                        "{lang:?} @{width}: '{}' overlaps '{}'",
                        x.text, y.text
                    ));
                }
                for d in testkit::clipped(&drawn) {
                    problems.push(format!("{lang:?} @{width}: '{}' is clipped", d.text));
                }
                for d in &drawn {
                    if d.rect.right() > width + 1.0 || d.rect.left() < -1.0 {
                        problems.push(format!(
                            "{lang:?} @{width}: '{}' leaves the window ({:?})",
                            d.text, d.rect
                        ));
                    }
                }
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    /// Every window, opened alone, on screens from a phone to a desktop, in two languages: it
    /// must lie wholly inside the screen, so its title bar and close button can always be reached.
    #[test]
    fn every_window_fits_every_screen() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let d = std::env::temp_dir().join(format!("cc-win-home-{}", std::process::id()));
            let _ = std::fs::create_dir_all(&d);
            // SAFETY: set once, before any test of this module reads it, always to this value.
            unsafe { std::env::set_var("COLIBRI_COMPANION_HOME", d) };
        });
        let open = |a: &mut CompanionApp, i: usize| match i {
            0 => a.guide.tips = true,
            1 => a.guide.explore = true,
            2 => a.guide.legal = true,
            3 => a.guide.shortcuts = true,
            4 => a.show_about = true,
            5 => a.sheets.local = true,
            6 => a.sheets.compare = true,
            7 => a.flows.open = true,
            8 => a.memory.open = true,
            9 => a.docs.open = true,
            10 => a.mcp.open = true,
            11 => a.studio.open = true,
            12 => a.duel.open = true,
            13 => a.access.open = true,
            14 => a.prompts.open = true,
            15 => a.online.open_catalog = true,
            16 => a.online.open_conn = true,
            _ => a.show_settings = true,
        };
        for lang in [Lang::Ro, Lang::De] {
            for (w, h) in [
                (1920.0, 1080.0),
                (1366.0, 768.0),
                (1024.0, 600.0),
                (800.0, 480.0),
                (390.0, 740.0),
            ] {
                for i in 0..18 {
                    let mut a = busy_app(lang);
                    open(&mut a, i);
                    let ctx = egui::Context::default();
                    let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), vec2(w, h));
                    for _ in 0..3 {
                        let raw = egui::RawInput {
                            screen_rect: Some(screen),
                            ..Default::default()
                        };
                        let mut out = ctx.run_ui(raw, |ui| a.windows(ui.ctx()));
                        out.textures_delta.clear();
                    }
                    let areas: Vec<_> =
                        ctx.memory(|m| m.areas().visible_layer_ids().into_iter().collect());
                    for layer in areas {
                        if layer.order != egui::Order::Middle {
                            continue;
                        }
                        let Some(r) = ctx.memory(|m| m.area_rect(layer.id)) else {
                            continue;
                        };
                        assert!(
                            r.min.x >= -1.0
                                && r.min.y >= -1.0
                                && r.max.x <= w + 1.0
                                && r.max.y <= h + 1.0,
                            "window {i} ({lang:?}) on {w}x{h} lies at {r:?}"
                        );
                    }
                }
            }
        }
    }

    /// A conversation full of hostile text (broken markdown, a million-character line, control and
    /// right-to-left characters, fake data URLs, absurd nesting) must draw on every screen without
    /// a panic, and each frame must stay quick.
    #[test]
    fn hostile_conversation_draws_everywhere() {
        use companion_core::store::StoredMessage;
        let nasty: Vec<String> = vec![
            "x".repeat(300_000),
            "word ".repeat(40_000),
            "# ".repeat(5_000) + "title",
            "> ".repeat(3_000) + "quote",
            "- ".repeat(3_000) + "item",
            "1. ".repeat(3_000) + "n",
            "```\n".repeat(2_000),
            "```rust\nfn main() {".to_string(),
            "**".repeat(20_000),
            "[a](".repeat(5_000),
            "![img](data:image/png;base64,AAAA)".repeat(200),
            "| a | b |\n|---|---|\n".to_string() + &"| 1 | 2 | 3 | 4 |\n".repeat(5_000),
            "\u{202e}\u{200b}\u{0}\u{1b}[31m\u{feff}e\u{301}\u{301}\u{301}😀👨\u{200d}👩\u{200d}👧"
                .repeat(2_000),
            "$$ ".repeat(5_000),
            "<script>alert(1)</script><img src=x onerror=1>".repeat(500),
            "\n".repeat(50_000),
            String::new(),
            "ا".repeat(30_000),
            "漢字かな".repeat(30_000),
        ];
        // Each message alone, so a slow one is named in the failure; the first frame lays out
        // everything, the second reuses what was measured.
        let mut slow: Vec<String> = Vec::new();
        for lang in [Lang::Ro] {
            for (w, h) in [(1920.0, 1080.0), (390.0, 740.0)] {
                for (i, t) in nasty.iter().enumerate() {
                    let mut a = busy_app(lang);
                    a.busy = None;
                    a.chat.conv.messages.push(StoredMessage {
                        role: if i % 2 == 0 { "user" } else { "assistant" }.into(),
                        content: t.clone(),
                        reasoning: t.chars().take(5_000).collect(),
                        display: if i % 3 == 0 {
                            t.chars().take(10_000).collect()
                        } else {
                            String::new()
                        },
                        sources: vec![t.chars().take(300).collect(); 40],
                        images: vec![
                            "data:image/png;base64,####".into(),
                            "data:,".into(),
                            String::new(),
                        ],
                        ..StoredMessage::default()
                    });
                    assert_eq!(
                        a.phase(),
                        crate::app::Phase::Chat,
                        "the test must draw the chat"
                    );
                    a.chat.input = t.chars().take(100_000).collect();
                    let ctx = egui::Context::default();
                    let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), vec2(w, h));
                    for frame in 0..2 {
                        let t0 = std::time::Instant::now();
                        let raw = egui::RawInput {
                            screen_rect: Some(screen),
                            ..Default::default()
                        };
                        let mut out = ctx.run_ui(raw, |ui| a.show_chat(ui));
                        out.textures_delta.clear();
                        let took = t0.elapsed();
                        if took > std::time::Duration::from_secs(30) {
                            let head: String = t.chars().take(12).collect();
                            slow.push(format!("message {i} ({} chars, {head:?}) frame {frame} on {w}x{h} ({lang:?}): {took:?}", t.chars().count()));
                            break;
                        }
                    }
                }
            }
        }
        assert!(slow.is_empty(), "slow frames:\n{}", slow.join("\n"));
    }
}
