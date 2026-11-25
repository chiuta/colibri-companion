//! "Check citations": paste a text whose sentences cite the documents of a collection, like
//! `[@report, p. 3]`, and see for each citation whether the cited passage backs the sentence.
//! It compares words and figures with the source (companion-scholar, ported from TEZAUR); it does
//! not understand meaning, so "supported" is a pointer to where to look, not proof.

use crate::app::CompanionApp;
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::rag::Collection;
use companion_scholar::audit::{
    self, AuditOptions, AuditReport, Passage, Passages, SourceInfo, Verdict,
};
use egui::{RichText, Ui};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Duration;

#[derive(Default)]
pub struct VerifyState {
    pub text: String,
    pub report: Option<AuditReport>,
    /// The collection the report (or the check still running) is about.
    report_coll: String,
    /// The check is running on its own thread; its report arrives here.
    run: Option<Receiver<AuditReport>>,
    /// The last check ended without a report (its thread stopped unexpectedly).
    failed: bool,
    /// Read the pages of the research shelf instead of the plain collection.
    pub pages: bool,
}

/// Longest text the box takes; the check itself reads at most 256 KiB of it.
const MAX_INPUT_CHARS: usize = 256 << 10;

/// Reads a collection the way the audit wants it: one source per document, its passages in order.
/// The collection has no page numbers, so every passage is page 0 ("no pages").
struct CollectionSources<'a> {
    col: &'a Collection,
    by_doc: Vec<Vec<usize>>,
}

impl<'a> CollectionSources<'a> {
    fn new(col: &'a Collection) -> Self {
        let mut by_doc = vec![Vec::new(); col.docs.len()];
        for (i, c) in col.chunks.iter().enumerate() {
            if let Some(v) = by_doc.get_mut(c.doc) {
                v.push(i);
            }
        }
        CollectionSources { col, by_doc }
    }
}

/// `report.pdf` is cited as `report` (and as `report.pdf`). A dot followed by a number is part of
/// the name, not an extension: "Chapter 2.5" and "Chapter 2" are two different documents.
pub(crate) fn stem(name: &str) -> String {
    companion_scholar::shelf::stem(name)
}

impl Passages for CollectionSources<'_> {
    fn source_count(&self) -> usize {
        self.col.docs.len()
    }
    fn source(&self, id: usize) -> Option<SourceInfo> {
        let d = self.col.docs.get(id)?;
        Some(SourceInfo {
            key: stem(&d.name),
            aliases: vec![d.name.clone()],
            title: d.name.clone(),
            ..SourceInfo::default()
        })
    }
    fn passages(&self, id: usize) -> Vec<Passage<'_>> {
        self.by_doc
            .get(id)
            .map(|v| {
                v.iter()
                    .map(|&i| Passage {
                        page: 0,
                        text: self.col.chunks[i].text.as_str(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Smaller limits than the library's defaults: a check should stay a matter of seconds. It runs
/// on its own thread (see [`start_check`]), never on the window thread: measured on a collection
/// of 60 000 passages and a hostile 256 KiB text, the same work held the window for 2.7 s
/// (release build; 19 s unoptimised).
fn options() -> AuditOptions {
    AuditOptions {
        max_input_bytes: 256 << 10,
        work_budget: 30_000_000,
        ..AuditOptions::default()
    }
}

#[cfg(test)]
fn check(col: &Collection, text: &str) -> AuditReport {
    check_with(col, text, &options())
}

fn check_with(col: &Collection, text: &str, opts: &AuditOptions) -> AuditReport {
    audit::audit(&CollectionSources::new(col), text, opts)
}

/// Runs the check on a new thread and returns at once; the report comes through the receiver and
/// `wake` is called when it is ready (to redraw the window). The collection is copied first, so the
/// person can go on adding documents while the check reads the copy. Dropping the receiver
/// abandons the check: the thread ends by itself, because the work is bounded by the budget.
fn start_check(
    col: &Collection,
    text: &str,
    opts: AuditOptions,
    wake: impl FnOnce() + Send + 'static,
) -> Receiver<AuditReport> {
    let (col, text) = (col.clone(), text.to_string());
    start_run(move || check_with(&col, &text, &opts), wake)
}

/// Like [`start_check`], for any check: `work` runs on its own thread.
fn start_run(
    work: impl FnOnce() -> AuditReport + Send + 'static,
    wake: impl FnOnce() + Send + 'static,
) -> Receiver<AuditReport> {
    let (tx, rx) = mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("verify-citations".into())
        .spawn(move || {
            let report = work();
            let _ = tx.send(report);
            wake();
        });
    // no thread: the sender was dropped with the closure, the receiver reports "disconnected"
    drop(spawned);
    rx
}

impl CompanionApp {
    fn verdict_label(&self, v: Verdict) -> &'static str {
        match v {
            Verdict::Supported => self.t("susținut", "supported"),
            Verdict::Weak => self.t("slab", "weak"),
            Verdict::Unsupported => self.t("nesusținut", "unsupported"),
            Verdict::WrongPage => self.t("pagină greșită", "wrong page"),
            Verdict::NumberMismatch => self.t("cifră diferită", "number differs"),
            Verdict::UnknownSource => self.t("sursă necunoscută", "unknown source"),
            Verdict::PageOutOfRange => self.t("pagină inexistentă", "page does not exist"),
            Verdict::BadRange => self.t("interval greșit", "bad page range"),
            Verdict::Uncited => self.t("fără citare", "no citation"),
            Verdict::QuoteVerbatim => self.t("citat exact", "exact quotation"),
            Verdict::QuoteElsewhere => {
                self.t("citat din altă sursă", "quotation from another source")
            }
            Verdict::QuoteNotFound => self.t("citat negăsit", "quotation not found"),
            Verdict::Contradicted => self.t("contrazis", "contradicted"),
        }
    }

    pub fn verify_section(&mut self, ui: &mut Ui, coll_id: &str) {
        egui::CollapsingHeader::new(
            RichText::new(self.t("Verifică citările", "Check citations")).semibold(),
        )
        .show(ui, |ui| self.verify_body(ui, coll_id));
    }

    fn verify_body(&mut self, ui: &mut Ui, coll_id: &str) {
        let p = self.p;
        {
            muted(
                ui,
                &p,
                self.t(
                    "Lipește un text ale cărui propoziții citează documentele din această colecție, de forma [@fișier, p. 3], și vezi dacă pasajul citat susține propoziția. Se compară cuvinte și cifre, nu sensul: «susținut» arată unde să te uiți, nu e o dovadă.",
                    "Paste a text whose sentences cite the documents in this collection, like [@file, p. 3], and see whether the cited passage backs the sentence. It compares words and figures, not meaning: “supported” points to where to look and is not proof.",
                ),
            );
            // the pasted text is capped at what the check reads, so a huge paste cannot freeze the
            // text box either
            let hint = self.t(
                "Lipește aici textul de verificat",
                "Paste the text to check here",
            );
            ui.add(
                egui::TextEdit::multiline(&mut self.docs.verify.text)
                    .hint_text(hint)
                    .char_limit(MAX_INPUT_CHARS)
                    .desired_rows(5)
                    .desired_width(f32::INFINITY),
            );
            self.docs.research.ensure(coll_id);
            let paged = self
                .docs
                .research
                .shelf_of(coll_id)
                .is_some_and(|s| s.counts().2 > 0);
            if paged {
                let label = self.t(
                    "Folosește numerele de pagină reale (raftul de cercetare)",
                    "Use the real page numbers (research shelf)",
                );
                ui.checkbox(&mut self.docs.verify.pages, label);
            }
            // take the report if the thread is done
            let mut waiting = false;
            if let Some(rx) = &self.docs.verify.run {
                match rx.try_recv() {
                    Ok(r) => {
                        self.docs.verify.report = Some(r);
                        self.docs.verify.run = None;
                    }
                    Err(TryRecvError::Empty) => waiting = true,
                    Err(TryRecvError::Disconnected) => {
                        self.docs.verify.failed = true;
                        self.docs.verify.run = None;
                    }
                }
            }
            if waiting {
                ui.ctx().request_repaint_after(Duration::from_millis(150));
            }
            ui.horizontal(|ui| {
                if waiting {
                    spinner(ui);
                    ui.label(self.t("Se verifică…", "Checking…"));
                } else if secondary_button(ui, self.t("Verifică", "Check")).clicked() {
                    self.docs.verify.report = None;
                    self.docs.verify.failed = false;
                    self.docs.verify.report_coll = coll_id.to_string();
                    let ctx = ui.ctx().clone();
                    if let Some(sh) = self
                        .docs
                        .research
                        .shelf_of(coll_id)
                        .filter(|_| paged && self.docs.verify.pages)
                    {
                        let (src, text) = (sh.sources(), self.docs.verify.text.clone());
                        self.docs.verify.run = Some(start_run(
                            move || audit::audit(&src, &text, &options()),
                            move || ctx.request_repaint(),
                        ));
                    } else if let Some(c) = self.docs.lib.get(coll_id) {
                        self.docs.verify.run = Some(start_check(
                            c,
                            &self.docs.verify.text,
                            options(),
                            move || ctx.request_repaint(),
                        ));
                    }
                }
                if let Some(r) = &self.docs.verify.report
                    && self.docs.verify.report_coll == coll_id
                    && secondary_button(ui, self.t("Copiază raportul", "Copy report")).clicked()
                {
                    ui.ctx().copy_text(r.markdown());
                }
            });
            if self.docs.verify.failed {
                ui.label(
                    RichText::new(self.t(
                        "Verificarea s-a oprit pe neașteptate. Încearcă din nou.",
                        "The check stopped unexpectedly. Try again.",
                    ))
                    .color(p.bad),
                );
            }
            let Some(r) = &self.docs.verify.report else {
                return;
            };
            // a report belongs to the collection it was made for
            if self.docs.verify.report_coll != coll_id {
                return;
            }
            if r.findings.is_empty() {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Nu am găsit nicio citare de forma [@fișier, p. 3] și niciun citat între ghilimele.",
                        "No citation like [@file, p. 3] and no quotation was found.",
                    ),
                );
            }
            let problems = r.findings.iter().filter(|f| f.is_problem()).count();
            ui.label(format!(
                "{} {} · {} {}",
                r.findings.len(),
                self.t("verificate", "checked"),
                problems,
                self.t("de citit", "to read")
            ));
            if r.truncated() {
                ui.label(
                    RichText::new(self.t(
                        "Raport incomplet: s-a atins o limită, deci nu e verificat tot textul.",
                        "Incomplete report: a limit was reached, so not all of the text was checked.",
                    ))
                    .color(p.warn),
                );
            }
            egui::ScrollArea::vertical()
                .id_salt("verify-findings")
                .max_height(260.0)
                .show(ui, |ui| {
                    for f in r.findings.iter().take(500) {
                        let colour = if f.is_problem() { p.bad } else { p.muted };
                        ui.label(
                            RichText::new(format!(
                                "{}{}",
                                self.verdict_label(f.verdict),
                                f.cite
                                    .as_ref()
                                    .map(|c| format!(" · {c}"))
                                    .unwrap_or_default()
                            ))
                            .color(colour)
                            .semibold(),
                        );
                        ui.label(&f.sentence);
                        if !f.note.is_empty() {
                            muted(ui, &p, &f.note);
                        }
                        ui.add_space(4.0);
                    }
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use companion_core::Lang;
    use companion_core::rag::Library;
    use companion_core::store::Settings;
    use std::time::Instant;

    fn library() -> (Library, String) {
        let mut lib = Library::default();
        let id = lib.create("test", 1).expect("a collection");
        lib.add_document(
            &id,
            "report.txt",
            "The reactor produced 42 megawatts of power during the autumn trial. \
             Engineers recorded stable temperatures throughout the whole trial.",
        )
        .expect("added");
        (lib, id)
    }

    /// Many documents of pseudo-random terms: reading them is what costs time.
    fn big_library(docs: usize, words: usize) -> (Library, String) {
        let mut lib = Library::default();
        let id = lib.create("big", 1).expect("a collection");
        for d in 0..docs {
            let mut x = (d as u64 + 1).wrapping_mul(6364136223846793005);
            let mut t = String::new();
            for _ in 0..words {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                t.push_str(&format!("term{} ", (x >> 33) % 20_000));
                if x % 17 == 0 {
                    t.push_str(". ");
                }
            }
            lib.add_document(&id, &format!("doc{d}.txt"), &t)
                .expect("added");
        }
        (lib, id)
    }

    /// 256 KiB of citations that each point at a different document.
    fn hostile_text(docs: usize) -> String {
        let mut t: String = (0..40_000)
            .map(|i| {
                format!(
                    "Sentence {i} term{} term{} [@doc{}]. ",
                    i % 20_000,
                    (i * 7) % 20_000,
                    i % docs
                )
            })
            .collect();
        t.truncate(256 << 10);
        t
    }

    #[test]
    fn a_cited_sentence_is_compared_with_the_cited_document() {
        let (lib, id) = library();
        let col = lib.get(&id).unwrap();
        let r = check(
            col,
            "The reactor produced 42 megawatts of power in the autumn trial [@report].",
        );
        assert_eq!(r.findings.len(), 1);
        assert_eq!(
            r.findings[0].verdict,
            Verdict::Supported,
            "{:?}",
            r.findings[0]
        );
        let r = check(
            col,
            "The reactor produced 90 megawatts of power in the autumn trial [@report].",
        );
        assert!(r.findings[0].is_problem(), "{:?}", r.findings[0]);
        let r = check(col, "Anything at all about the reactor power [@nothing].");
        assert_eq!(r.findings[0].verdict, Verdict::UnknownSource);
    }

    #[test]
    fn file_name_with_extension_also_resolves_and_too_long_input_is_reported_as_cut() {
        let (lib, id) = library();
        let col = lib.get(&id).unwrap();
        let r = check(
            col,
            "Stable temperatures were recorded in the trial [@report.txt].",
        );
        assert_ne!(r.findings[0].verdict, Verdict::UnknownSource);
        // 1.1 MB of input: only the first 256 KiB is read, and the report says so
        let big = "[@x, p. 1] ".repeat(100_000);
        let r = check(col, &big);
        assert!(r.truncated(), "a cut must be announced, not silent");
    }

    #[test]
    fn a_dot_before_a_number_is_not_an_extension() {
        assert_eq!(stem("report.pdf"), "report");
        assert_eq!(stem("notes.TXT"), "notes");
        assert_eq!(stem("Chapter 2.5"), "Chapter 2.5");
        assert_eq!(stem("minutes 2024.11"), "minutes 2024.11");
        assert_eq!(stem(".hidden"), ".hidden");
        assert_eq!(stem("no extension"), "no extension");
        // two documents that used to share the key "Chapter 2"
        let mut lib = Library::default();
        let id = lib.create("c", 1).unwrap();
        lib.add_document(
            &id,
            "Chapter 2.pdf",
            "The first text is about apples and pears.",
        )
        .unwrap();
        lib.add_document(
            &id,
            "Chapter 2.5",
            "The second text is about quantum chromodynamics.",
        )
        .unwrap();
        let col = lib.get(&id).unwrap();
        let src = CollectionSources::new(col);
        assert_eq!(src.source(0).unwrap().key, "Chapter 2");
        assert_eq!(src.source(1).unwrap().key, "Chapter 2.5");
    }

    #[test]
    fn the_check_never_holds_the_window_thread() {
        // The work is real (about 2.7 s in a release build on 60 000 passages and the worst
        // 256 KiB text); starting it must take a few milliseconds whatever the size. A smaller
        // budget keeps the test short: the start time does not depend on it.
        let (lib, id) = big_library(150, 15_000);
        let col = lib.get(&id).unwrap();
        let opts = AuditOptions {
            work_budget: 3_000_000,
            ..options()
        };
        let text = hostile_text(150);
        // how long the real work takes on this machine, so the limit scales with the runner
        let t = Instant::now();
        let direct = check_with(col, &text, &opts);
        let work = t.elapsed();
        let t = Instant::now();
        let rx = start_check(col, &text, opts.clone(), || {});
        let started_in = t.elapsed();
        assert!(
            started_in * 4 < work,
            "starting took {started_in:?}, the work {work:?}"
        );
        let report = rx
            .recv_timeout(Duration::from_secs(120))
            .expect("the report arrives");
        // same answer as the direct call
        assert_eq!(report.findings.len(), direct.findings.len());
        assert_eq!(report.truncated(), direct.truncated());
        assert!(
            report.truncated(),
            "the budget must have been the limit here"
        );
    }

    fn app() -> crate::app::CompanionApp {
        crate::app::CompanionApp::new(
            egui::Context::default(),
            Settings::default(),
            Some("http://127.0.0.1:1".into()),
            false,
        )
    }

    fn frame(a: &mut crate::app::CompanionApp, id: &str) {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900.0, 700.0),
            )),
            ..Default::default()
        };
        let ctx = a.ctx.clone();
        let mut out = ctx.run_ui(input, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                // what the Documents window shows when the person unfolds the section
                a.verify_body(ui, id);
            });
        });
        out.textures_delta.clear();
    }

    #[test]
    fn the_window_picks_up_the_report_when_the_thread_is_done() {
        let (lib, id) = library();
        let mut a = app();
        a.docs.lib = lib;
        a.docs.verify.text = "Stable temperatures were recorded in the trial [@report].".into();
        let col = a.docs.lib.get(&id).unwrap().clone();
        a.docs.verify.report_coll = id.clone();
        a.docs.verify.run = Some(start_check(
            &col,
            &a.docs.verify.text.clone(),
            options(),
            || {},
        ));
        let t = Instant::now();
        while a.docs.verify.report.is_none() && t.elapsed() < Duration::from_secs(30) {
            frame(&mut a, &id);
            std::thread::sleep(Duration::from_millis(20));
        }
        let r = a.docs.verify.report.as_ref().expect("a report");
        assert_eq!(r.findings.len(), 1);
        assert!(a.docs.verify.run.is_none());
        // a report is not shown under another collection
        frame(&mut a, "another");
        assert!(!a.docs.verify.failed);
    }

    #[test]
    fn a_check_whose_thread_died_is_reported_not_waited_for_forever() {
        let (lib, id) = library();
        let mut a = app();
        a.docs.lib = lib;
        let (tx, rx) = mpsc::channel::<AuditReport>();
        drop(tx); // the thread ended without sending a report
        a.docs.verify.run = Some(rx);
        a.docs.verify.report_coll = id.clone();
        frame(&mut a, &id);
        assert!(a.docs.verify.run.is_none(), "no endless spinner");
        assert!(a.docs.verify.failed, "the person is told");
        assert!(a.docs.verify.report.is_none());
    }

    #[test]
    fn every_verdict_has_a_label_in_every_language() {
        let mut a = app();
        for lang in Lang::ALL {
            a.s.lang = lang;
            for v in Verdict::ALL {
                let label = a.verdict_label(v);
                assert!(!label.is_empty(), "{lang:?} {v:?}");
                if lang != Lang::En {
                    // translated: not the English word left in place
                    let mut en = app();
                    en.s.lang = Lang::En;
                    assert_ne!(label, en.verdict_label(v), "{lang:?} {v:?}");
                }
            }
        }
    }

    /// `cargo test --release -p colibri-companion perf_ -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn perf_worst_case_check_in_the_background_thread() {
        let (lib, id) = big_library(370, 15_000);
        let col = lib.get(&id).unwrap();
        let text = hostile_text(370);
        let t = Instant::now();
        let rx = start_check(col, &text, options(), || {});
        println!("window thread blocked for {} ms", t.elapsed().as_millis());
        let r = rx.recv().unwrap();
        println!(
            "check itself (other thread): {} ms, findings {}, truncated {}",
            t.elapsed().as_millis(),
            r.findings.len(),
            r.truncated()
        );
    }
}
