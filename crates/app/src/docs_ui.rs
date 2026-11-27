//! Chat with your documents: a window to build collections of files, and the glue that puts the
//! best passages in front of the model when a collection is switched on. Everything is local.

use crate::app::{CompanionApp, Event, Probe};
use crate::widgets::*;
use companion_core::attach::{self, AttachError, Extracted};
use companion_core::paths;
use companion_core::rag::{self, AddError, Library, Searcher};
use egui::{Align2, RichText, Ui, vec2};
use std::path::PathBuf;

#[derive(Default)]
pub struct DocsState {
    pub open: bool,
    pub lib: Library,
    pub loaded: bool,
    /// Bumped on every change, so the cached search index knows when it is stale.
    pub rev: u64,
    pub selected: Option<String>,
    pub new_name: String,
    pub path_input: String,
    /// Files still being read in the background.
    pub busy: usize,
    pub message: String,
    pub verify: crate::verify_ui::VerifyState,
    pub bib: crate::bib_ui::BibState,
    pub ocr: crate::ocr_ui::OcrState,
    pub research: crate::research_ui::ResearchState,
    searcher: Option<(String, u64, Searcher)>,
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl CompanionApp {
    pub fn open_docs(&mut self) {
        self.load_docs();
        if self.docs.selected.is_none() {
            self.docs.selected = self
                .s
                .docs_active
                .clone()
                .or_else(|| self.docs.lib.collections.first().map(|c| c.id.clone()));
        }
        self.docs.open = true;
        self.ocr_probe();
    }

    pub fn load_docs(&mut self) {
        if !self.docs.loaded {
            self.docs.lib = Library::load(&paths::documents_file());
            self.docs.loaded = true;
            self.docs.rev += 1;
        }
    }

    fn save_docs(&mut self) {
        self.docs.rev += 1;
        if let Err(e) = self.docs.lib.save(&paths::documents_file()) {
            self.toast(format!(
                "{}: {e}",
                self.t(
                    "Nu am putut salva documentele",
                    "Could not save the documents"
                )
            ));
        }
    }

    /// Files or folders chosen for a collection: each file is read in the background.
    pub fn docs_add_paths(&mut self, coll: &str, roots: &[PathBuf]) {
        let found = attach::collect(roots, rag::MAX_DOCS_PER_COLLECTION);
        if found.files.is_empty() {
            self.docs.message = self
                .t("Nu am găsit fișiere acolo.", "I found no files there.")
                .to_string();
            return;
        }
        if found.too_many {
            self.toast(self.t(
                "Sunt prea multe fișiere: le iau doar pe primele.",
                "There are too many files: taking only the first ones.",
            ));
        }
        let python = match &self.python {
            Probe::Found(p) => Some(p.clone()),
            _ => None,
        };
        let shelf_files: Vec<PathBuf> = found
            .files
            .iter()
            .filter(|p| attach::kind_of(p) != attach::Kind::Image)
            .cloned()
            .collect();
        if !shelf_files.is_empty() {
            let coll = coll.to_string();
            self.spawn(move |tx| {
                let file =
                    companion_scholar::shelf::shelf_file(&crate::research_ui::shelves_dir(), &coll);
                let results = companion_scholar::shelf::add_files(&file, &shelf_files);
                let _ = tx.send(Event::ShelfDone { coll, results });
            });
        }
        for path in found.files {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if attach::kind_of(&path) == attach::Kind::Image {
                continue;
            }
            self.docs.busy += 1;
            let (py, coll, p2) = (python.clone(), coll.to_string(), path.clone());
            self.spawn(move |tx| {
                let result = attach::extract_text(py.as_ref(), &p2);
                let _ = tx.send(Event::DocsRead { coll, name, result });
            });
        }
    }

    pub fn docs_read(
        &mut self,
        coll: String,
        name: String,
        result: Result<Extracted, AttachError>,
    ) {
        self.docs.busy = self.docs.busy.saturating_sub(1);
        let why = match result {
            Ok(ex) => match self.docs.lib.add_document(&coll, &name, &ex.text) {
                Ok(_) => {
                    self.save_docs();
                    return;
                }
                Err(AddError::Empty) => self.t("nu are text", "it has no text").to_string(),
                Err(AddError::TooManyDocs) => self
                    .t("colecția e plină", "the collection is full")
                    .to_string(),
                Err(AddError::TooManyChunks) => self
                    .t(
                        "colecția are prea mult text",
                        "the collection has too much text",
                    )
                    .to_string(),
                Err(AddError::NoSuchCollection) => return,
            },
            Err(e) => {
                self.docs.message = self.attach_error(&name, &e);
                return;
            }
        };
        self.docs.message = format!("«{name}»: {why}");
    }

    /// The passages for a question, from the collection that is switched on. `None` when no
    /// collection is on, or nothing in it matches.
    pub fn docs_context(&mut self, question: &str) -> Option<rag::Context> {
        let id = self.s.docs_active.clone()?;
        self.load_docs();
        let rev = self.docs.rev;
        let col = self.docs.lib.get(&id)?;
        let stale = !matches!(&self.docs.searcher, Some((i, r, _)) if *i == id && *r == rev);
        if stale {
            let s = Searcher::new(col);
            self.docs.searcher = Some((id.clone(), rev, s));
        }
        let (_, _, searcher) = self.docs.searcher.as_ref()?;
        let hits = searcher.search(question, rag::TOP_K);
        rag::build_context(self.docs.lib.get(&id)?, &hits)
    }

    /// A line above the composer: answer from documents, and which collection.
    pub fn docs_inline(&mut self, ui: &mut Ui) {
        self.load_docs();
        if self.docs.lib.collections.is_empty() && self.s.docs_active.is_none() {
            return;
        }
        {
            ui.separator();
            ui.label(RichText::new(self.t("Documente:", "Documents:")).small());
            let current = self
                .s
                .docs_active
                .as_ref()
                .and_then(|id| self.docs.lib.get(id))
                .map(|c| c.name.clone())
                .unwrap_or_else(|| self.t("oprit", "off").to_string());
            let mut pick: Option<Option<String>> = None;
            egui::ComboBox::from_id_salt("docs-active")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(self.s.docs_active.is_none(), self.t("oprit", "off"))
                        .clicked()
                    {
                        pick = Some(None);
                    }
                    for c in &self.docs.lib.collections {
                        let on = self.s.docs_active.as_deref() == Some(c.id.as_str());
                        if ui.selectable_label(on, &c.name).clicked() {
                            pick = Some(Some(c.id.clone()));
                        }
                    }
                });
            if let Some(p) = pick {
                self.s.docs_active = p;
                let _ = self.s.save();
            }
            if ui.small_button(self.t("Gestionează", "Manage")).clicked() {
                self.open_docs();
            }
            if self.s.docs_active.is_some() {
                let tip = self.t(
                    "Doar cu corbul pornit cu --packs: lucrul făcut pe fiecare pasaj se reține și se refolosește, în orice ordine. Cu alt motor se folosește chatul obișnuit.",
                    "Only with corbul started with --packs: the work done on each passage is kept and reused, in any order. With another engine the ordinary chat is used.",
                );
                let mut on = self.s.context_packs;
                if ui
                    .checkbox(
                        &mut on,
                        self.t("Refolosește calculul", "Reuse earlier work"),
                    )
                    .on_hover_text(tip)
                    .changed()
                {
                    self.s.context_packs = on;
                    let _ = self.s.save();
                }
            }
        }
    }

    pub fn docs_window(&mut self, ctx: &egui::Context) {
        if !self.docs.open {
            return;
        }
        let p = self.p;
        let mut open = true;
        let mut add: Option<Vec<PathBuf>> = None;
        let mut create = false;
        let mut remove_coll: Option<String> = None;
        let mut remove_doc: Option<(String, String)> = None;
        let hint_new = self.t("Numele unei colecții noi", "Name of a new collection");
        let hint_path = self.t(
            "Calea către un fișier sau un dosar (poți și trage fișiere aici)",
            "Path to a file or folder (you can also drop files here)",
        );
        egui::Window::new(self.t("Documente", "Documents")).bounded(ctx)
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_size(fit(ctx, 760.0, 620.0))
            .collapsible(false)
            .show(ctx, |ui| {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Pune fișiere într-o colecție, apoi alege colecția deasupra casetei de scris. Răspunsurile se formulează din pasajele găsite, cu numerele lor. Totul se caută pe acest calculator, fără internet și fără alt model.",
                        "Put files in a collection, then pick the collection above the message box. Answers are written from the passages found, with their numbers. Everything is searched on this computer, with no internet and no second model.",
                    ),
                );
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.docs.new_name)
                            .hint_text(hint_new)
                            .desired_width(240.0),
                    );
                    if secondary_button(ui, self.t("Creează", "Create")).clicked() {
                        create = true;
                    }
                });
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    for c in &self.docs.lib.collections {
                        let on = self.docs.selected.as_deref() == Some(c.id.as_str());
                        if ui.selectable_label(on, &c.name).clicked() {
                            self.docs.selected = Some(c.id.clone());
                        }
                    }
                });
                ui.separator();
                let Some(id) = self.docs.selected.clone() else {
                    muted(
                        ui,
                        &p,
                        self.t(
                            "Creează o colecție ca să începi.",
                            "Create a collection to begin.",
                        ),
                    );
                    return;
                };
                // only the short list of documents is copied, never the passages
                let Some((doc_list, total_chunks)) = self
                    .docs
                    .lib
                    .get(&id)
                    .map(|c| (c.docs.clone(), c.chunks.len()))
                else {
                    return;
                };
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.docs.path_input)
                            .hint_text(hint_path)
                            .desired_width(ui.available_width() - 190.0),
                    );
                    if primary_button(ui, &p, self.t("Adaugă", "Add")).clicked()
                        && !self.docs.path_input.trim().is_empty()
                    {
                        add = Some(vec![PathBuf::from(self.docs.path_input.trim())]);
                        self.docs.path_input.clear();
                    }
                });
                ui.label(
                    RichText::new(
                        self.t("documente: {} · pasaje: {}", "documents: {} · passages: {}")
                            .replacen("{}", &doc_list.len().to_string(), 1)
                            .replacen("{}", &total_chunks.to_string(), 1),
                    )
                    .color(p.muted),
                );
                if self.docs.busy > 0 {
                    ui.horizontal(|ui| {
                        crate::widgets::spinner(ui);
                        ui.label(
                            self.t("Citesc {} fișiere…", "Reading {} files…")
                                .replacen("{}", &self.docs.busy.to_string(), 1),
                        );
                    });
                }
                if !self.docs.message.is_empty() {
                    ui.label(RichText::new(&self.docs.message).color(p.bad));
                }
                egui::ScrollArea::vertical()
                    .auto_shrink([false, true])
                    .max_height(360.0)
                    .show(ui, |ui| {
                        for d in &doc_list {
                            ui.horizontal(|ui| {
                                ui.label(&d.name);
                                ui.label(
                                    RichText::new(
                                        self.t("pasaje: {}", "passages: {}")
                                            .replacen("{}", &d.chunks.to_string(), 1),
                                    )
                                    .small()
                                    .color(p.muted),
                                );
                                if ui.small_button(self.t("Scoate", "Remove")).clicked() {
                                    remove_doc = Some((id.clone(), d.name.clone()));
                                }
                            });
                        }
                    });
                ui.add_space(6.0);
                self.cite_section(ui, &doc_list);
                ui.add_space(6.0);
                self.verify_section(ui, &id);
                ui.add_space(6.0);
                self.research_section(ui, &id);
                ui.add_space(6.0);
                self.ocr_section(ui, &id);
                ui.add_space(6.0);
                self.bib_section(ui);
                ui.add_space(6.0);
                if ui
                    .button(self.t("Șterge această colecție", "Delete this collection"))
                    .clicked()
                {
                    remove_coll = Some(id.clone());
                }
            });
        // files dropped while the window is open go into the selected collection
        if self.docs.open {
            let dropped: Vec<PathBuf> = ctx.input(|i| {
                i.raw
                    .dropped_files
                    .iter()
                    .filter(|f| !f.path().as_os_str().is_empty())
                    .map(|f| f.path().to_path_buf())
                    .collect()
            });
            let (for_docs, for_ocr) = crate::ocr_ui::route_dropped(&dropped, self.docs.ocr.found());
            if let Some(p) = for_ocr {
                self.docs.ocr.path = p.to_string_lossy().into_owned();
            }
            if !for_docs.is_empty() {
                add = Some(for_docs);
            }
        }
        if create {
            let name = std::mem::take(&mut self.docs.new_name);
            if let Some(id) = self.docs.lib.create(&name, now()) {
                self.docs.selected = Some(id);
                self.save_docs();
            }
        }
        if let (Some(paths), Some(id)) = (add, self.docs.selected.clone()) {
            self.docs.message.clear();
            self.docs_add_paths(&id, &paths);
        }
        if let Some((c, d)) = remove_doc {
            {
                let (coll, name) = (c.clone(), d.clone());
                self.spawn(move |tx| {
                    let file = companion_scholar::shelf::shelf_file(
                        &crate::research_ui::shelves_dir(),
                        &coll,
                    );
                    companion_scholar::shelf::remove_file(&file, &name);
                    let _ = tx.send(Event::ShelfDone {
                        coll,
                        results: Vec::new(),
                    });
                });
            }
            if let Some(col) = self.docs.lib.collections.iter_mut().find(|x| x.id == c) {
                col.remove_doc(&d);
            }
            self.save_docs();
        }
        if let Some(c) = remove_coll {
            companion_scholar::shelf::delete(&companion_scholar::shelf::shelf_file(
                &crate::research_ui::shelves_dir(),
                &c,
            ));
            self.docs.research.drop_shelf();
            self.docs.lib.remove(&c);
            if self.s.docs_active.as_deref() == Some(c.as_str()) {
                self.s.docs_active = None;
                let _ = self.s.save();
            }
            self.docs.selected = self.docs.lib.collections.first().map(|c| c.id.clone());
            self.save_docs();
        }
        if !open {
            self.docs.open = false;
        }
    }
}
