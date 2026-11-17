//! The prompt library window, the answer-style picker and the prompt editor.

use crate::app::CompanionApp;
use crate::widgets::*;
use companion_core::prompts::{self, Library, Prompt, Style};
use egui::{Align2, RichText, Ui, vec2};

#[derive(Default)]
pub struct PromptState {
    pub open: bool,
    pub query: String,
    /// Empty = all, "★" = favourites, "mine" handled by `own_only`.
    pub category: String,
    pub favorites_only: bool,
    pub own_only: bool,
    pub selected: Option<String>,
    pub values: Vec<(String, String)>,
    pub lib: Library,
    pub edit: Option<Prompt>,
    pub import_path: String,
    pub style: Style,
    pub loaded: bool,
    pub focus_query: bool,
}

fn style_name(app: &CompanionApp, s: Style) -> &'static str {
    match s {
        Style::Normal => app.t("Obișnuit", "Normal"),
        Style::Exact => app.t("Exact", "Exact"),
        Style::Didactic => app.t("Didactic", "Didactic"),
        Style::Socratic => app.t("Socratic", "Socratic"),
        Style::Contrarian => app.t("Contrarian", "Contrarian"),
        Style::Brainstorm => app.t("Idei multe", "Brainstorm"),
        Style::Telegraphic => app.t("Telegrafic", "Telegraphic"),
        Style::Coach => app.t("Coach", "Coach"),
        Style::Simple => app.t("Limbaj simplu", "Plain language"),
    }
}

fn library_file() -> std::path::PathBuf {
    companion_core::paths::data_dir().join("prompts.json")
}

impl CompanionApp {
    /// The built-in prompts in the current language: Romanian and English are written out,
    /// the other six come from the translation table by their English text.
    pub fn builtin_prompts(&self) -> Vec<companion_core::prompts::Prompt> {
        use companion_core::store::Lang;
        let mut v = prompts::builtin(self.s.lang == Lang::Ro);
        if matches!(self.s.lang, Lang::Ro | Lang::En) {
            return v;
        }
        let tr = |en: &str| {
            crate::i18n::lookup(self.s.lang, en).map_or_else(|| en.to_string(), str::to_string)
        };
        v = prompts::builtin(false);
        for p in &mut v {
            p.category = tr(&p.category);
            p.title = tr(&p.title);
            p.text = tr(&p.text);
        }
        v
    }

    pub fn open_prompts(&mut self) {
        if !self.prompts.loaded {
            self.prompts.lib = Library::load(&library_file());
            self.prompts.loaded = true;
        }
        self.prompts.open = true;
        self.prompts.focus_query = true;
    }

    fn save_prompts(&mut self) {
        if let Err(e) = self.prompts.lib.save(&library_file()) {
            self.toast(format!(
                "{}: {e}",
                self.t(
                    "Nu am putut salva prompturile",
                    "Could not save the prompts"
                )
            ));
        }
    }

    /// A small picker for how the answers should be written, shown above the composer.
    pub fn style_bar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new(self.t("Stil răspuns:", "Answer style:")).small());
            let current = style_name(self, self.prompts.style);
            egui::ComboBox::from_id_salt("answer-style")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for s in Style::ALL {
                        let label = style_name(self, s);
                        ui.selectable_value(&mut self.prompts.style, s, label);
                    }
                });
            if ui
                .small_button(self.t("Prompturi (Ctrl+P)", "Prompts (Ctrl+P)"))
                .clicked()
            {
                self.open_prompts();
            }
            self.docs_inline(ui);
        });
    }

    pub fn prompt_window(&mut self, ctx: &egui::Context) {
        if !self.prompts.open {
            return;
        }
        let p = self.p;
        let mut open = true;
        let mut all = self.builtin_prompts();
        all.extend(self.prompts.lib.own.clone());
        let h_search = self.t("Caută…", "Search…");
        let h_fav = self.t("★ Favorite", "★ Favourites");
        let h_mine = self.t("Ale mele", "Mine");
        let h_path = self.t(
            "Calea unui fișier JSON de importat",
            "Path of a JSON file to import",
        );
        let mut use_text: Option<String> = None;
        let mut fav_toggle: Option<String> = None;
        let mut delete: Option<String> = None;
        let mut edit_new: Option<Prompt> = None;
        egui::Window::new(self.t("Biblioteca de prompturi", "Prompt library"))
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_size(fit(ctx, 760.0, 520.0))
            .collapsible(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.prompts.query)
                            .hint_text(h_search)
                            .desired_width(240.0),
                    );
                    if self.prompts.focus_query {
                        r.request_focus();
                        self.prompts.focus_query = false;
                    }
                    ui.checkbox(&mut self.prompts.favorites_only, h_fav);
                    ui.checkbox(&mut self.prompts.own_only, h_mine);
                    if ui.button(self.t("+ Prompt nou", "+ New prompt")).clicked() {
                        edit_new = Some(Prompt {
                            id: String::new(),
                            title: String::new(),
                            category: String::new(),
                            text: String::new(),
                            own: true,
                        });
                    }
                });
                let mut cats: Vec<String> = all.iter().map(|p| p.category.clone()).collect();
                cats.sort();
                cats.dedup();
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .selectable_label(self.prompts.category.is_empty(), self.t("Toate", "All"))
                        .clicked()
                    {
                        self.prompts.category.clear();
                    }
                    for c in &cats {
                        if ui.selectable_label(&self.prompts.category == c, c).clicked() {
                            self.prompts.category = c.clone();
                        }
                    }
                });
                ui.separator();
                let shown: Vec<&Prompt> = all
                    .iter()
                    .filter(|x| self.prompts.category.is_empty() || x.category == self.prompts.category)
                    .filter(|x| !self.prompts.favorites_only || self.prompts.lib.is_favorite(&x.id))
                    .filter(|x| !self.prompts.own_only || x.own)
                    .filter(|x| prompts::matches(x, &self.prompts.query))
                    .collect();
                ui.columns(2, |cols| {
                    egui::ScrollArea::vertical()
                        .id_salt("prompt-list")
                        .show(&mut cols[0], |ui| {
                            if shown.is_empty() {
                                muted(ui, &p, self.t(
                                    "Niciun prompt nu se potrivește. Încearcă alt cuvânt sau altă categorie.",
                                    "No prompt matches. Try another word or category.",
                                ));
                            }
                            for x in &shown {
                                ui.horizontal(|ui| {
                                    let star = if self.prompts.lib.is_favorite(&x.id) { "★" } else { "☆" };
                                    if ui.small_button(star).clicked() {
                                        fav_toggle = Some(x.id.clone());
                                    }
                                    let sel = self.prompts.selected.as_deref() == Some(x.id.as_str());
                                    if ui.selectable_label(sel, &x.title).clicked() {
                                        self.prompts.selected = Some(x.id.clone());
                                        self.prompts.values = prompts::placeholders(&x.text)
                                            .into_iter()
                                            .map(|n| (n, String::new()))
                                            .collect();
                                    }
                                });
                            }
                        });
                    let ui = &mut cols[1];
                    let chosen = self
                        .prompts
                        .selected
                        .as_ref()
                        .and_then(|id| all.iter().find(|x| &x.id == id));
                    match chosen {
                        None => muted(ui, &p, self.t(
                            "Alege un prompt din listă. Completează spațiile libere, apoi îl plasezi în mesaj.",
                            "Pick a prompt from the list. Fill in the blanks, then put it in your message.",
                        )),
                        Some(x) => {
                            ui.label(RichText::new(&x.title).strong().size(16.0));
                            muted(ui, &p, &x.category);
                            ui.add_space(4.0);
                            for (name, v) in &mut self.prompts.values {
                                ui.label(RichText::new(name.as_str()).small());
                                ui.add(
                                    egui::TextEdit::multiline(v)
                                        .desired_rows(1)
                                        .desired_width(f32::INFINITY),
                                );
                            }
                            let text = prompts::fill(&x.text, &self.prompts.values);
                            ui.add_space(4.0);
                            egui::ScrollArea::vertical().id_salt("prompt-preview").max_height(150.0).show(ui, |ui| {
                                ui.label(RichText::new(&text).color(p.muted));
                            });
                            ui.horizontal(|ui| {
                                if ui.button(self.t("Plasează în mesaj", "Put in message")).clicked() {
                                    use_text = Some(text.clone());
                                }
                                if ui.button(self.t("Copiază", "Copy")).clicked() {
                                    ui.ctx().copy_text(text.clone());
                                }
                                if x.own {
                                    if ui.button(self.t("Editează", "Edit")).clicked() {
                                        edit_new = Some((*x).clone());
                                    }
                                    if ui.button(self.t("Șterge", "Delete")).clicked() {
                                        delete = Some(x.id.clone());
                                    }
                                }
                            });
                        }
                    }
                });
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button(self.t("Exportă prompturile mele (JSON)", "Export my prompts (JSON)")).clicked() {
                        let path = companion_core::paths::downloads_dir().join("prompturi-colibri.json");
                        let _ = std::fs::create_dir_all(companion_core::paths::downloads_dir());
                        let msg = match std::fs::write(&path, self.prompts.lib.export_json()) {
                            Ok(()) => format!("{} {}", self.t("Salvat în", "Saved to"), path.display()),
                            Err(e) => e.to_string(),
                        };
                        self.toast(msg);
                    }
                    ui.add(
                        egui::TextEdit::singleline(&mut self.prompts.import_path)
                            .hint_text(h_path)
                            .desired_width(220.0),
                    );
                    if ui.button(self.t("Importă", "Import")).clicked() {
                        let path = self.prompts.import_path.trim().trim_matches('"').to_string();
                        let msg = match std::fs::read_to_string(&path) {
                            Ok(t) => match self.prompts.lib.import_json(&t) {
                                Ok(n) => format!("{} {n}", self.t("Prompturi noi:", "New prompts:")),
                                Err(e) => format!("{}: {e}", self.t("Fișier nepotrivit", "Not a prompt file")),
                            },
                            Err(_) => self.t("Nu pot citi fișierul", "I can't read the file").to_string(),
                        };
                        self.toast(msg);
                        self.save_prompts();
                    }
                });
            });
        self.prompts.open = open;
        if let Some(id) = fav_toggle {
            self.prompts.lib.toggle_favorite(&id);
            self.save_prompts();
        }
        if let Some(id) = delete {
            self.prompts.lib.remove(&id);
            self.prompts.selected = None;
            self.save_prompts();
        }
        if let Some(t) = use_text {
            if !self.chat.input.is_empty() && !self.chat.input.ends_with('\n') {
                self.chat.input.push('\n');
            }
            self.chat.input.push_str(&t);
            self.chat.focus_input = true;
            self.prompts.open = false;
        }
        if edit_new.is_some() {
            self.prompts.edit = edit_new;
        }
        self.prompt_editor(ctx);
    }

    fn prompt_editor(&mut self, ctx: &egui::Context) {
        let Some(mut e) = self.prompts.edit.take() else {
            return;
        };
        let p = self.p;
        let mut keep_open = true;
        let mut save = false;
        egui::Window::new(self.t("Prompt propriu", "My own prompt"))
            .open(&mut keep_open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_width(fit(ctx, 480.0, 0.0).x)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.label(self.t("Titlu", "Title"));
                ui.add(egui::TextEdit::singleline(&mut e.title).desired_width(f32::INFINITY));
                ui.label(self.t("Categorie", "Category"));
                ui.add(egui::TextEdit::singleline(&mut e.category).desired_width(f32::INFINITY));
                ui.label(self.t(
                    "Text (pune [CUVINTE] pentru spațiile de completat)",
                    "Text (use [WORDS] for the blanks)",
                ));
                ui.add(
                    egui::TextEdit::multiline(&mut e.text)
                        .desired_rows(8)
                        .desired_width(f32::INFINITY),
                );
                if e.title.trim().is_empty() || e.text.trim().is_empty() {
                    muted(
                        ui,
                        &p,
                        self.t(
                            "Are nevoie de un titlu și un text.",
                            "It needs a title and a text.",
                        ),
                    );
                } else if ui.button(self.t("Salvează", "Save")).clicked() {
                    save = true;
                }
            });
        if save {
            let id = e.id.clone();
            self.prompts.lib.upsert(e);
            self.save_prompts();
            let _ = id;
        } else if keep_open {
            self.prompts.edit = Some(e);
        }
    }
}
