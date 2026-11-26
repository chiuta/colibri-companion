//! "Research tools" in the Documents window: what the scholar library (companion-scholar, ported
//! from TEZAUR) does beyond the plain collection. Passages come with their page numbers, the
//! citation check can use those pages, an answer can be sealed in a provenance receipt and the
//! receipt checked again later, the receipts can be turned into an AI-use declaration in eight
//! languages, and the retrieval can be measured on questions of your own. Nothing leaves the
//! computer; files go to the `exports` folder of the app's data folder.

use crate::app::CompanionApp;
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::embed;
use companion_core::http::CancelToken;
use companion_core::paths;
use companion_scholar::declare::{self, Lang as DLang, Locality};
use companion_scholar::receipt::Status;
use companion_scholar::retrieve::Passage;
use companion_scholar::shelf::{self, Added, Shelf};
use egui::{RichText, Ui};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};

/// Receipts read from the exports folder when a declaration is written (the rest are ignored).
const MAX_RECEIPT_FILES: usize = 2000;
/// Largest receipt file read (bytes); a real receipt is a few kilobytes.
const MAX_RECEIPT_READ: u64 = 4 << 20;
/// Characters of a passage shown in the list.
const SNIPPET_CHARS: usize = 320;

/// What the embedding thread reports.
pub enum EmbMsg {
    Progress(usize, usize),
    Passages(Vec<Vec<f32>>),
    Query(Vec<f32>),
    Failed(String),
}

#[derive(Default)]
pub struct ResearchState {
    shelf: Option<(String, Shelf)>,
    pub emb_url: String,
    pub emb_model: String,
    pub meaning: bool,
    emb_rx: Option<Receiver<EmbMsg>>,
    emb_cancel: Option<CancelToken>,
    emb_progress: (usize, usize),
    pending_query: String,
    /// Collection and model the running embedding job was started for.
    emb_job: Option<(String, String)>,
    /// Collection and model whose passages carry embeddings in the loaded shelf.
    emb_ready: Option<(String, String)>,
    pub query: String,
    pub hits: Vec<Passage>,
    pub searched: bool,
    pub answer: String,
    pub model: String,
    pub key: String,
    pub receipt: String,
    pub check_text: String,
    pub check_answer: String,
    pub check_lines: Vec<(Status, String)>,
    pub decl_lang: Option<usize>,
    pub decl_loc: usize,
    pub decl_extra: String,
    pub decl: String,
    pub cases: String,
    pub eval: String,
    pub message: String,
}

impl ResearchState {
    /// Forget the loaded shelf (a file was added or removed); it is read again when needed.
    pub fn drop_shelf(&mut self) {
        self.stop_embedding();
        self.emb_ready = None;
        self.shelf = None;
        self.hits.clear();
        self.searched = false;
    }

    /// The shelf of this collection when it is loaded.
    pub fn shelf_of(&self, coll: &str) -> Option<&Shelf> {
        self.shelf
            .as_ref()
            .filter(|(c, _)| c == coll)
            .map(|(_, s)| s)
    }

    fn shelf_mut(&mut self, coll: &str) -> Option<&mut Shelf> {
        self.shelf
            .as_mut()
            .filter(|(c, _)| c == coll)
            .map(|(_, s)| s)
    }

    pub fn embedding_busy(&self) -> bool {
        self.emb_rx.is_some()
    }

    fn stop_embedding(&mut self) {
        if let Some(c) = self.emb_cancel.take() {
            c.cancel();
        }
        self.emb_rx = None;
        self.emb_job = None;
    }

    /// Embeds `texts` (batches) in a thread; `query` is the one question to embed instead.
    fn start_embedding(
        &mut self,
        ctx: &egui::Context,
        coll: &str,
        texts: Vec<String>,
        is_query: bool,
    ) {
        self.stop_embedding();
        let (tx, rx) = mpsc::channel();
        let cancel = CancelToken::new();
        let (url, model) = (
            self.emb_url.trim().to_string(),
            self.emb_model.trim().to_string(),
        );
        let (c2, ctx2) = (cancel.clone(), ctx.clone());
        self.emb_progress = (0, texts.len());
        let job = (coll.to_string(), model.clone());
        let spawned = std::thread::Builder::new()
            .name("embeddings".into())
            .spawn(move || {
                let total = texts.len();
                let mut all: Vec<Vec<f32>> = Vec::with_capacity(total);
                for chunk in texts.chunks(embed::BATCH) {
                    if c2.is_cancelled() {
                        return;
                    }
                    match embed::embed_batch(&url, &model, None, chunk, Some(&c2)) {
                        Ok(v) => all.extend(v),
                        Err(e) => {
                            let _ = tx.send(EmbMsg::Failed(e));
                            ctx2.request_repaint();
                            return;
                        }
                    }
                    let _ = tx.send(EmbMsg::Progress(all.len(), total));
                    ctx2.request_repaint();
                }
                let msg = if is_query {
                    all.into_iter().next().map_or(
                        EmbMsg::Failed("no embedding came back".into()),
                        EmbMsg::Query,
                    )
                } else {
                    EmbMsg::Passages(all)
                };
                let _ = tx.send(msg);
                ctx2.request_repaint();
            });
        if spawned.is_ok() {
            self.emb_rx = Some(rx);
            self.emb_cancel = Some(cancel);
            self.emb_job = Some(job);
        }
    }

    pub fn ensure(&mut self, coll: &str) {
        if self.shelf.as_ref().is_none_or(|(c, _)| c != coll) {
            let s = Shelf::load(&shelf::shelf_file(&shelves_dir(), coll));
            self.shelf = Some((coll.to_string(), s));
            self.hits.clear();
            self.searched = false;
            self.stop_embedding();
            self.emb_ready = None;
        }
    }
}

pub fn shelves_dir() -> PathBuf {
    paths::data_dir().join("shelves")
}

pub fn exports_dir() -> PathBuf {
    paths::data_dir().join("exports")
}

/// Writes `content` to a new file in `dir`, named `<stem>-<unix seconds>[-n].<ext>`; the stem is
/// reduced to safe characters. Never overwrites. Returns the path.
pub fn save_text(dir: &Path, stem: &str, ext: &str, content: &str) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let safe: String = stem
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(40)
        .collect();
    let ext: String = ext
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(8)
        .collect();
    let now = companion_scholar::receipt::now_secs();
    for n in 0..1000 {
        let name = if n == 0 {
            format!("{safe}-{now}.{ext}")
        } else {
            format!("{safe}-{now}-{n}.{ext}")
        };
        let path = dir.join(name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => {
                use std::io::Write;
                f.write_all(content.as_bytes()).map_err(|e| e.to_string())?;
                return Ok(path);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.to_string()),
        }
    }
    Err("too many files with the same name".into())
}

/// Every `receipt-*.json` in `dir`, as text (bounded in count and size).
pub fn read_receipts(dir: &Path) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().is_some_and(|n| {
                let n = n.to_string_lossy();
                n.starts_with("receipt-") && n.ends_with(".json")
            })
        })
        .collect();
    names.sort();
    names
        .iter()
        .take(MAX_RECEIPT_FILES)
        .filter(|p| std::fs::metadata(p).is_ok_and(|m| m.len() <= MAX_RECEIPT_READ))
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .collect()
}

/// The declaration for the given receipts, and how many receipt texts could not be read.
pub fn declaration_for(
    texts: &[String],
    lang: DLang,
    locality: Locality,
    tool: &str,
) -> (String, usize) {
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let (u, skipped) = declare::Use::from_receipts(&refs, tool, locality);
    (declare::declaration(&u, lang), skipped)
}

fn tool_name() -> String {
    format!("Colibri Companion {}", env!("CARGO_PKG_VERSION"))
}

fn snippet(s: &str) -> String {
    let one: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() <= SNIPPET_CHARS {
        one
    } else {
        let cut: String = one.chars().take(SNIPPET_CHARS).collect();
        format!("{cut}…")
    }
}

/// The text of what became of each file added to the shelf: how many could not be read.
pub fn failures(results: &[(String, Added)]) -> usize {
    results
        .iter()
        .filter(|(_, r)| matches!(r, Added::Failed(_) | Added::NoText))
        .count()
}

impl CompanionApp {
    /// The shelf finished changing in the background.
    pub fn shelf_done(&mut self, coll: &str, results: &[(String, Added)]) {
        if self.docs.selected.as_deref() == Some(coll)
            || self.docs.research.shelf_of(coll).is_some()
        {
            self.docs.research.drop_shelf();
        }
        let n = failures(results);
        if n > 0 {
            self.docs.research.message = self
                .t(
                    "{} fișiere nu au numere de pagină (nu s-a putut citi textul).",
                    "{} files have no page numbers (their text could not be read).",
                )
                .replacen("{}", &n.to_string(), 1);
        }
    }

    fn save_note(&mut self, r: Result<PathBuf, String>) {
        self.docs.research.message = match r {
            Ok(p) => {
                self.t("Salvat în: {}", "Saved to: {}")
                    .replacen("{}", &p.display().to_string(), 1)
            }
            Err(e) => self
                .t("Nu am putut salva: {}", "Could not save: {}")
                .replacen("{}", &e, 1),
        };
    }

    pub fn research_section(&mut self, ui: &mut Ui, coll_id: &str) {
        egui::CollapsingHeader::new(
            RichText::new(self.t("Instrumente de cercetare", "Research tools")).semibold(),
        )
        .id_salt("research_section")
        .show(ui, |ui| self.research_body(ui, coll_id));
    }

    fn research_body(&mut self, ui: &mut Ui, coll_id: &str) {
        let p = self.p;
        self.docs.research.ensure(coll_id);
        muted(
            ui,
            &p,
            self.t(
                "Pasaje cu numere de pagină, chitanțe de proveniență pentru răspunsuri, declarație de folosire a AI și evaluarea căutării. Totul rămâne pe calculatorul tău. Fișierele iau forma lor cu pagini când le adaugi în colecție; cele adăugate înainte de această versiune trebuie adăugate din nou.",
                "Passages with page numbers, provenance receipts for answers, an AI-use declaration and a measure of the search. Everything stays on your computer. Files get their pages when you add them to the collection; files added before this version need to be added again.",
            ),
        );
        let (docs, passages, paged) = self
            .docs
            .research
            .shelf_of(coll_id)
            .map(Shelf::counts)
            .unwrap_or((0, 0, 0));
        let line = self
            .t(
                "Raftul cu pagini: {} documente, {} pasaje, {} cu pagini reale",
                "Page shelf: {} documents, {} passages, {} with real pages",
            )
            .replacen("{}", &docs.to_string(), 1)
            .replacen("{}", &passages.to_string(), 1)
            .replacen("{}", &paged.to_string(), 1);
        muted(ui, &p, &line);
        if !self.docs.research.message.is_empty() {
            let m = self.docs.research.message.clone();
            ui.label(RichText::new(m).color(p.warn));
        }
        if passages == 0 {
            return;
        }
        self.research_search(ui);
        self.research_receipt(ui, coll_id);
        self.research_verify_receipt(ui, coll_id);
        self.research_declaration(ui);
        self.research_eval(ui, coll_id);
    }

    /// Takes in what the embedding thread has sent since the last frame.
    fn poll_embeddings(&mut self) {
        let coll = self.docs.selected.clone().unwrap_or_default();
        let job = self.docs.research.emb_job.clone();
        // a result that belongs to another collection is never applied to this one
        let same = job.as_ref().is_some_and(|(c, _)| *c == coll);
        let mut msgs = Vec::new();
        if let Some(rx) = &self.docs.research.emb_rx {
            while let Ok(m) = rx.try_recv() {
                msgs.push(m);
            }
        }
        let mut finished = false;
        for m in msgs {
            match m {
                EmbMsg::Progress(a, b) => self.docs.research.emb_progress = (a, b),
                EmbMsg::Passages(v) => {
                    let mut ok = 0;
                    let total = v.len();
                    let fits = same
                        && self
                            .docs
                            .research
                            .shelf_of(&coll)
                            .is_some_and(|s| s.counts().1 == total);
                    if fits && let Some(sh) = self.docs.research.shelf_mut(&coll) {
                        for (i, e) in v.into_iter().enumerate() {
                            if sh.set_embedding(i, e).is_ok() {
                                ok += 1;
                            }
                        }
                    }
                    if fits && ok == total {
                        if let Some((c, m)) = job.clone() {
                            self.docs.research.emb_ready = Some((c, m));
                        }
                    } else {
                        self.docs.research.emb_ready = None;
                    }
                    self.docs.research.message = if !fits {
                        self.t(
                            "Raftul s-a schimbat cât timp se pregătea; apasă din nou „Pregătește raftul”.",
                            "The shelf changed while it was being prepared; press “Prepare the shelf” again.",
                        )
                        .to_string()
                    } else {
                        format!(
                            "{} {ok}",
                            self.t(
                                "Pasaje pregătite pentru căutarea după sens:",
                                "Passages ready for search by meaning:"
                            )
                        )
                    };
                    finished = true;
                }
                EmbMsg::Query(q) => {
                    let query = self.docs.research.pending_query.clone();
                    let hits = if same {
                        self.docs
                            .research
                            .shelf_of(&coll)
                            .map(|s| s.search_by_meaning(&query, 8, &q).passages)
                            .unwrap_or_default()
                    } else {
                        Vec::new()
                    };
                    self.docs.research.hits = hits;
                    finished = true;
                }
                EmbMsg::Failed(e) => {
                    let query = self.docs.research.pending_query.clone();
                    if !query.is_empty() {
                        let hits = self
                            .docs
                            .research
                            .shelf_of(&coll)
                            .map(|s| s.search(&query, 8).passages)
                            .unwrap_or_default();
                        self.docs.research.hits = hits;
                    }
                    self.docs.research.message = format!(
                        "{} {e}",
                        self.t(
                            "Căutarea după sens a eșuat; folosesc cuvintele cheie:",
                            "Search by meaning failed; using keywords:"
                        )
                    );
                    finished = true;
                }
            }
        }
        if finished {
            self.docs.research.emb_rx = None;
            self.docs.research.emb_cancel = None;
            self.docs.research.pending_query.clear();
        }
    }

    fn research_meaning(&mut self, ui: &mut Ui, coll: &str) {
        let p = self.p;
        egui::CollapsingHeader::new(self.t("Caută după sens", "Search by meaning"))
            .id_salt("research_meaning")
            .show(ui, |ui| {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Folosește un server de embeddings pornit de tine (de exemplu corbul serve --encoder). Se trimite doar textul pasajelor și al întrebării, la adresa de mai jos; folosește o adresă de pe acest calculator, dacă nu ai încredere în server.",
                        "Uses an embeddings server you start yourself (for example corbul serve --encoder). Only the text of the passages and of your question is sent, to the address below; use an address on this computer unless you trust the server.",
                    ),
                );
                ui.add(
                    egui::TextEdit::singleline(&mut self.docs.research.emb_url)
                        .hint_text("http://127.0.0.1:8080")
                        .char_limit(300)
                        .desired_width(f32::INFINITY),
                );
                let hint = self.t("Numele modelului", "Model name");
                ui.add(
                    egui::TextEdit::singleline(&mut self.docs.research.emb_model)
                        .hint_text(hint)
                        .char_limit(200)
                        .desired_width(f32::INFINITY),
                );
                let busy = self.docs.research.embedding_busy();
                let have_url = !self.docs.research.emb_url.trim().is_empty();
                let n = self
                    .docs
                    .research
                    .shelf_of(coll)
                    .map(|s| (s.embedded(), s.counts().1))
                    .unwrap_or((0, 0));
                ui.horizontal(|ui| {
                    if !busy {
                        if ui
                            .add_enabled(
                                have_url && n.1 > 0,
                                egui::Button::new(self.t("Pregătește raftul", "Prepare the shelf")),
                            )
                            .clicked()
                        {
                            let texts = self
                                .docs
                                .research
                                .shelf_of(coll)
                                .map(|s| s.passage_texts())
                                .unwrap_or_default();
                            let ctx = ui.ctx().clone();
                            self.docs.research.start_embedding(&ctx, coll, texts, false);
                        }
                    } else if secondary_button(ui, self.t("Oprește", "Stop")).clicked() {
                        self.docs.research.stop_embedding();
                    }
                    let (a, b) = self.docs.research.emb_progress;
                    if busy {
                        muted(ui, &p, &format!("{a} / {b}"));
                    } else {
                        muted(
                            ui,
                            &p,
                            &format!(
                                "{} {} / {}",
                                self.t("pasaje pregătite:", "passages ready:"),
                                n.0,
                                n.1
                            ),
                        );
                    }
                });
                let label = self.t(
                    "Folosește sensul la căutare",
                    "Use meaning in the search",
                );
                ui.add_enabled(
                    n.0 > 0,
                    egui::Checkbox::new(&mut self.docs.research.meaning, label),
                );
                let ready = self.docs.research.emb_ready
                    == Some((
                        coll.to_string(),
                        self.docs.research.emb_model.trim().to_string(),
                    ));
                if self.docs.research.meaning && n.0 > 0 && !ready && !busy {
                    muted(
                        ui,
                        &p,
                        self.t(
                            "Modelul s-a schimbat: pregătește raftul din nou, altfel se caută doar după cuvinte.",
                            "The model changed: prepare the shelf again, or only the words are searched.",
                        ),
                    );
                }
            });
    }

    fn research_search(&mut self, ui: &mut Ui) {
        self.poll_embeddings();
        let coll_now = self.docs.selected.clone().unwrap_or_default();
        self.research_meaning(ui, &coll_now);
        let p = self.p;
        egui::CollapsingHeader::new(
            self.t("Caută cu numere de pagină", "Search with page numbers"),
        )
        .id_salt("research_search")
        .show(ui, |ui| {
            let hint = self.t(
                "Întrebarea ta (ghilimele pentru o expresie exactă)",
                "Your question (quotes for an exact phrase)",
            );
            let enter = ui
                .add(
                    egui::TextEdit::singleline(&mut self.docs.research.query)
                        .hint_text(hint)
                        .char_limit(2000)
                        .desired_width(f32::INFINITY),
                )
                .lost_focus()
                && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let go = secondary_button(ui, self.t("Caută", "Search")).clicked() || enter;
            if go {
                self.docs.research.searched = true;
                let q = self.docs.research.query.clone();
                let coll = self.docs.selected.clone().unwrap_or_default();
                let by_meaning = self.docs.research.meaning
                    && !q.trim().is_empty()
                    && self
                        .docs
                        .research
                        .shelf_of(&coll)
                        .is_some_and(|s| s.embedded() > 0)
                    && self.docs.research.emb_ready
                        == Some((
                            coll.clone(),
                            self.docs.research.emb_model.trim().to_string(),
                        ));
                if by_meaning {
                    self.docs.research.pending_query = q.clone();
                    let ctx = ui.ctx().clone();
                    self.docs
                        .research
                        .start_embedding(&ctx, &coll, vec![q], true);
                } else {
                    let hits = self
                        .docs
                        .research
                        .shelf_of(&coll)
                        .map(|s| s.search(&q, 8).passages)
                        .unwrap_or_default();
                    self.docs.research.hits = hits;
                }
            }
            if self.docs.research.searched && self.docs.research.hits.is_empty() {
                muted(ui, &p, self.t("Nimic găsit.", "Nothing found."));
            }
            let hits = self.docs.research.hits.clone();
            for h in &hits {
                let page = if h.page > 0 {
                    format!("p. {}", h.page)
                } else {
                    self.t("fără pagini", "no pages").to_string()
                };
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(format!("{} · {page}", h.doc_title)).semibold());
                    ui.label(RichText::new(format!("{:.0}%", h.coverage * 100.0)).color(p.muted));
                    if ui
                        .small_button(self.t("Copiază citarea", "Copy citation"))
                        .clicked()
                    {
                        ui.ctx().copy_text(Shelf::citation(h));
                    }
                });
                muted(ui, &p, &snippet(&h.text));
                ui.add_space(3.0);
            }
        });
    }

    fn research_receipt(&mut self, ui: &mut Ui, coll_id: &str) {
        let p = self.p;
        egui::CollapsingHeader::new(self.t("Chitanță de proveniență", "Provenance receipt"))
            .id_salt("research_receipt")
            .show(ui, |ui| {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Leagă un răspuns de pasajele găsite mai sus (ultima căutare): întrebarea, modelul, amprenta răspunsului și a fiecărui pasaj, cu pagina. Cu o cheie secretă, chitanța e sigilată: oricine are cheia poate verifica că nu a fost schimbată. Fără cheie, ea dovedește doar că nimic nu s-a stricat din greșeală.",
                        "Ties an answer to the passages found above (the last search): the question, the model, a fingerprint of the answer and of every passage, with its page. With a secret key the receipt is sealed, so anyone with the key can check it was not changed. Without a key it only shows that nothing was damaged by accident.",
                    ),
                );
                ui.horizontal(|ui| {
                    ui.label(self.t("Model", "Model"));
                    ui.add(
                        egui::TextEdit::singleline(&mut self.docs.research.model)
                            .char_limit(200)
                            .desired_width(180.0),
                    );
                    ui.label(self.t("Cheie (opțional)", "Key (optional)"));
                    ui.add(
                        egui::TextEdit::singleline(&mut self.docs.research.key)
                            .password(true)
                            .char_limit(200)
                            .desired_width(140.0),
                    );
                });
                let hint = self.t("Răspunsul de legat de chitanță", "The answer to tie to the receipt");
                ui.add(
                    egui::TextEdit::multiline(&mut self.docs.research.answer)
                        .hint_text(hint)
                        .char_limit(1 << 20)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY),
                );
                if secondary_button(ui, self.t("Emite chitanța", "Issue receipt")).clicked() {
                    self.issue_receipt(coll_id);
                }
                if !self.docs.research.receipt.is_empty() {
                    let r = self.docs.research.receipt.clone();
                    ui.horizontal(|ui| {
                        if secondary_button(ui, self.t("Copiază chitanța", "Copy receipt")).clicked()
                        {
                            ui.ctx().copy_text(r);
                        }
                    });
                }
            });
    }

    fn issue_receipt(&mut self, coll_id: &str) {
        let st = &self.docs.research;
        if st.hits.is_empty() {
            self.docs.research.message = self
                .t(
                    "Caută întâi: chitanța se leagă de pasajele găsite.",
                    "Search first: the receipt is tied to the passages found.",
                )
                .to_string();
            return;
        }
        let Some(shelf) = st.shelf_of(coll_id) else {
            return;
        };
        let model = if st.model.trim().is_empty() {
            "unspecified"
        } else {
            st.model.trim()
        };
        let key = st.key.as_bytes();
        let made = shelf.issue_receipt(
            &st.query,
            model,
            &st.answer,
            &st.hits,
            (!key.is_empty()).then_some(key),
            &tool_name(),
        );
        match made {
            Ok(text) => {
                let saved = save_text(&exports_dir(), "receipt", "json", &text);
                self.docs.research.receipt = text;
                self.docs.research.message.clear();
                self.save_note(saved);
            }
            Err(e) => self.docs.research.message = e,
        }
    }

    fn research_verify_receipt(&mut self, ui: &mut Ui, coll_id: &str) {
        let p = self.p;
        egui::CollapsingHeader::new(self.t("Verifică o chitanță", "Check a receipt"))
            .id_salt("research_check")
            .show(ui, |ui| {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Lipește o chitanță. Se verifică amprenta, sigiliul (dacă dai cheia), răspunsul (dacă îl lipești) și faptul că fiecare pasaj există încă, neschimbat, în această colecție.",
                        "Paste a receipt. It checks the fingerprint, the seal (if you give the key), the answer (if you paste it) and that every passage still exists unchanged in this collection.",
                    ),
                );
                let hint = self.t("Chitanța (JSON)", "The receipt (JSON)");
                ui.add(
                    egui::TextEdit::multiline(&mut self.docs.research.check_text)
                        .hint_text(hint)
                        .char_limit(4 << 20)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY),
                );
                let hint = self.t("Răspunsul (opțional)", "The answer (optional)");
                ui.add(
                    egui::TextEdit::multiline(&mut self.docs.research.check_answer)
                        .hint_text(hint)
                        .char_limit(1 << 20)
                        .desired_rows(2)
                        .desired_width(f32::INFINITY),
                );
                if secondary_button(ui, self.t("Verifică", "Check")).clicked() {
                    let st = &self.docs.research;
                    let key = st.key.as_bytes();
                    let lines = st
                        .shelf_of(coll_id)
                        .map(|s| {
                            let v = s.verify_receipt(
                                &st.check_text,
                                (!key.is_empty()).then_some(key),
                                Some(st.check_answer.as_str()),
                            );
                            v.checks
                                .iter()
                                .map(|c| (c.status, c.detail.clone()))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    self.docs.research.check_lines = lines;
                }
                for (status, detail) in &self.docs.research.check_lines {
                    let (mark, colour) = match status {
                        Status::Pass => ("✓", p.ok),
                        Status::Fail => ("✗", p.bad),
                        Status::Note => ("•", p.muted),
                    };
                    ui.label(RichText::new(format!("{mark} {detail}")).color(colour));
                }
            });
    }

    fn research_declaration(&mut self, ui: &mut Ui) {
        let p = self.p;
        egui::CollapsingHeader::new(self.t("Declarație de folosire a AI", "AI-use declaration"))
            .id_salt("research_decl")
            .show(ui, |ui| {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Un text pentru lucrarea ta, făcut din chitanțele salvate (folderul exports) și, dacă vrei, încă una lipită mai jos: ce unealtă și ce modele ai folosit, între ce date, câte întrebări și câte documente. Nu spune nimic despre ce n-ai ales să spui.",
                        "A text for your paper, made from the saved receipts (the exports folder) and, if you like, one more pasted below: which tool and models you used, between which dates, how many questions and how many documents. It says nothing you did not choose to say.",
                    ),
                );
                let cur = DLang::from_code(self.s.lang.code()).unwrap_or(DLang::En);
                let idx = self
                    .docs
                    .research
                    .decl_lang
                    .unwrap_or_else(|| DLang::ALL.iter().position(|l| *l == cur).unwrap_or(0))
                    .min(DLang::ALL.len() - 1);
                let mut idx_new = idx;
                let locals = [
                    self.t("Modelul a rulat pe calculatorul meu", "The model ran on my computer"),
                    self.t("Text poate fi plecat la un serviciu extern", "Text may have gone to an external service"),
                    self.t("Nu spun unde a rulat", "Do not say where it ran"),
                ];
                let mut loc = self.docs.research.decl_loc.min(2);
                ui.horizontal_wrapped(|ui| {
                    ui.label(self.t("Limba", "Language"));
                    egui::ComboBox::from_id_salt("decl_lang")
                        .selected_text(DLang::ALL[idx].code())
                        .show_ui(ui, |ui| {
                            for (i, l) in DLang::ALL.iter().enumerate() {
                                ui.selectable_value(&mut idx_new, i, l.code());
                            }
                        });
                    egui::ComboBox::from_id_salt("decl_loc")
                        .selected_text(locals[loc])
                        .show_ui(ui, |ui| {
                            for (i, t) in locals.iter().enumerate() {
                                ui.selectable_value(&mut loc, i, *t);
                            }
                        });
                });
                self.docs.research.decl_lang = Some(idx_new);
                self.docs.research.decl_loc = loc;
                let hint = self.t("O chitanță în plus (opțional)", "One more receipt (optional)");
                ui.add(
                    egui::TextEdit::multiline(&mut self.docs.research.decl_extra)
                        .hint_text(hint)
                        .char_limit(4 << 20)
                        .desired_rows(2)
                        .desired_width(f32::INFINITY),
                );
                if secondary_button(ui, self.t("Scrie declarația", "Write the declaration")).clicked()
                {
                    let mut texts = read_receipts(&exports_dir());
                    if !self.docs.research.decl_extra.trim().is_empty() {
                        texts.push(self.docs.research.decl_extra.clone());
                    }
                    let locality = [Locality::Local, Locality::External, Locality::Unstated][loc];
                    let (text, skipped) =
                        declaration_for(&texts, DLang::ALL[idx_new], locality, &tool_name());
                    self.docs.research.decl = text;
                    if texts.is_empty() {
                        self.docs.research.message = self
                            .t(
                                "Nu există nicio chitanță: emite una mai sus.",
                                "There is no receipt yet: issue one above.",
                            )
                            .to_string();
                    } else if skipped > 0 {
                        self.docs.research.message = self
                            .t(
                                "{} chitanțe nu s-au putut citi.",
                                "{} receipts could not be read.",
                            )
                            .replacen("{}", &skipped.to_string(), 1);
                    } else {
                        self.docs.research.message.clear();
                    }
                }
                if !self.docs.research.decl.is_empty() {
                    let d = self.docs.research.decl.clone();
                    ui.label(&d);
                    ui.horizontal(|ui| {
                        if secondary_button(ui, self.t("Copiază", "Copy")).clicked() {
                            ui.ctx().copy_text(d.clone());
                        }
                        if secondary_button(ui, self.t("Salvează ca fișier", "Save as a file"))
                            .clicked()
                        {
                            let r = save_text(&exports_dir(), "declaration", "md", &d);
                            self.save_note(r);
                        }
                    });
                }
            });
    }

    fn research_eval(&mut self, ui: &mut Ui, coll_id: &str) {
        let p = self.p;
        egui::CollapsingHeader::new(self.t("Măsoară căutarea", "Measure the search"))
            .id_salt("research_eval")
            .show(ui, |ui| {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Scrie întrebări cu documentul (și pagina) care le răspunde, câte una pe linie, în forma {\"q\": \"întrebarea\", \"doc\": \"parte din titlu\", \"page\": 3}. Vezi cât de des apare pasajul potrivit printre primele rezultate. Măsoară căutarea, nu calitatea răspunsurilor.",
                        "Write questions with the document (and page) that answers them, one per line, like {\"q\": \"the question\", \"doc\": \"part of the title\", \"page\": 3}. See how often the right passage comes first. It measures the search, not the quality of answers.",
                    ),
                );
                ui.add(
                    egui::TextEdit::multiline(&mut self.docs.research.cases)
                        .char_limit(1 << 20)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY),
                );
                if secondary_button(ui, self.t("Măsoară", "Measure")).clicked() {
                    let st = &self.docs.research;
                    let out = st
                        .shelf_of(coll_id)
                        .map(|s| s.evaluate(&st.cases))
                        .unwrap_or_else(|| Err(String::new()));
                    self.docs.research.eval = match out {
                        Ok(r) => r.markdown(),
                        Err(e) => e,
                    };
                }
                if !self.docs.research.eval.is_empty() {
                    let e = self.docs.research.eval.clone();
                    ui.label(&e);
                    if secondary_button(ui, self.t("Copiază raportul", "Copy report")).clicked() {
                        ui.ctx().copy_text(e);
                    }
                }
            });
    }

    /// "Save as <format>" buttons for a list of entries (the same six formats as "Copy as").
    pub fn save_formats_row(
        &mut self,
        ui: &mut Ui,
        stem: &str,
        entries: &[companion_scholar::bib::Entry],
    ) {
        let save_t = self.t("Salvează ca {}", "Save as {}");
        let mut chosen: Option<(String, &'static str, String)> = None;
        ui.horizontal_wrapped(|ui| {
            for (name, make) in &crate::bib_ui::formats() {
                if secondary_button(ui, &save_t.replacen("{}", name, 1)).clicked() {
                    let ext = match *name {
                        "BibTeX" => "bib",
                        "RIS" => "ris",
                        "CSL JSON" => "json",
                        _ => "txt",
                    };
                    chosen = Some((name.to_string(), ext, make(entries)));
                }
            }
        });
        if let Some((name, ext, text)) = chosen {
            let file = format!("{stem}-{}", name.to_lowercase().replace(' ', "-"));
            let r = save_text(&exports_dir(), &file, ext, &text);
            self.save_note(r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("research-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn saving_never_overwrites_and_cleans_the_name() {
        let d = tmp("save");
        let a = save_text(&d, "../evil name", "js/on", "one").unwrap();
        let b = save_text(&d, "../evil name", "js/on", "two").unwrap();
        assert_ne!(a, b);
        assert_eq!(a.parent().unwrap(), d.as_path());
        assert!(
            a.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("___evil_name-")
        );
        assert!(a.extension().is_some_and(|e| e == "json"));
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "one");
        assert_eq!(std::fs::read_to_string(&b).unwrap(), "two");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn an_issued_receipt_becomes_a_declaration_in_every_language() {
        let d = tmp("decl");
        let s = Shelf::from_bytes(&[(
            "law.txt",
            "txt",
            b"The burden of proof lies with the authority.",
        )]);
        let hits = s.search("burden of proof", 3).passages;
        let r = s
            .issue_receipt(
                "Who must prove?",
                "tiny-model",
                "The authority.",
                &hits,
                None,
                "Test 1",
            )
            .unwrap();
        save_text(&d, "receipt", "json", &r).unwrap();
        save_text(&d, "other", "json", "{}").unwrap(); // not a receipt file: ignored
        let texts = read_receipts(&d);
        assert_eq!(texts.len(), 1);
        for l in DLang::ALL {
            let (text, skipped) = declaration_for(&texts, l, Locality::Local, "Test 1");
            assert_eq!(skipped, 0);
            assert!(text.contains("tiny-model"), "{l:?}: {text}");
        }
        let (_, skipped) =
            declaration_for(&["not json".into()], DLang::En, Locality::Unstated, "t");
        assert_eq!(skipped, 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn snippets_are_one_line_and_bounded() {
        let long = "word ".repeat(500);
        let s = snippet(&format!("a\n\n b\t{long}"));
        assert!(!s.contains('\n'));
        assert!(s.chars().count() <= SNIPPET_CHARS + 1);
        assert_eq!(snippet("short  text"), "short text");
    }

    fn app() -> crate::app::CompanionApp {
        crate::app::CompanionApp::new(
            egui::Context::default(),
            companion_core::store::Settings::default(),
            Some("http://127.0.0.1:1".into()),
            false,
        )
    }

    fn frame(a: &mut crate::app::CompanionApp, id: &str) {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900.0, 900.0),
            )),
            ..Default::default()
        };
        let ctx = a.ctx.clone();
        let mut out = ctx.run_ui(input, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                a.research_body(ui, id);
            });
        });
        out.textures_delta.clear();
    }

    #[test]
    fn the_section_draws_with_and_without_a_shelf_in_every_language() {
        for lang in companion_core::Lang::ALL {
            let mut a = app();
            a.s.lang = lang;
            // an empty shelf for this collection: only the explanation is shown
            a.docs.research.shelf = Some(("c".into(), Shelf::default()));
            frame(&mut a, "c");
            // a shelf with documents and a finished search: every sub-section draws
            let s = Shelf::from_bytes(&[(
                "law.txt",
                "txt",
                b"The burden of proof lies with the authority.",
            )]);
            a.docs.research.hits = s.search("burden of proof", 3).passages;
            a.docs.research.searched = true;
            a.docs.research.shelf = Some(("c".into(), s));
            a.docs.research.decl = "A declaration.".into();
            a.docs.research.eval = "# Retrieval evaluation".into();
            a.docs.research.receipt = "{}".into();
            a.docs.research.check_lines = vec![
                (Status::Pass, "ok".into()),
                (Status::Fail, "bad".into()),
                (Status::Note, "note".into()),
            ];
            for _ in 0..3 {
                frame(&mut a, "c");
            }
            // another collection starts with its own, empty shelf from the file system
            a.docs.research.drop_shelf();
            assert!(a.docs.research.shelf_of("c").is_none());
        }
    }

    #[test]
    fn the_bibliography_tools_draw_with_entries() {
        let mut a = app();
        a.docs.bib.text =
            "@book{a, author={X, Y}, year={2020}, title={T}, isbn={9780306406157}}".into();
        a.docs.bib.read_text();
        a.docs.bib.ids = vec![(
            "9780306406157".into(),
            companion_scholar::ids::Verdict::Valid,
        )];
        a.docs.bib.ids_done = true;
        a.docs.bib.cite_out = "text".into();
        a.docs.bib.cite_refs = "refs".into();
        a.docs.bib.dups = "0 duplicates".into();
        let ctx = a.ctx.clone();
        let mut out = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900.0, 900.0),
                )),
                ..Default::default()
            },
            |ui| {
                egui::CentralPanel::default().show(ui, |ui| a.bib_body(ui));
            },
        );
        out.textures_delta.clear();
    }

    #[test]
    fn failures_counts_unreadable_files_only() {
        let r = vec![
            ("a".to_string(), Added::Passages(3)),
            ("b".to_string(), Added::Same),
            ("c".to_string(), Added::NoText),
            ("d".to_string(), Added::Failed("x".into())),
        ];
        assert_eq!(failures(&r), 2);
    }
}
