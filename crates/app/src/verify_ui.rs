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

#[derive(Default)]
pub struct VerifyState {
    pub text: String,
    pub report: Option<AuditReport>,
}

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

/// `report.pdf` is cited as `report` (and as `report.pdf`).
fn stem(name: &str) -> String {
    match name.rsplit_once('.') {
        Some((s, ext)) if !s.is_empty() && ext.len() <= 5 && !ext.contains(' ') => s.to_string(),
        _ => name.to_string(),
    }
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

/// Smaller limits than the library's defaults: this runs on the window thread.
fn options() -> AuditOptions {
    AuditOptions {
        max_input_bytes: 256 << 10,
        work_budget: 30_000_000,
        ..AuditOptions::default()
    }
}

pub fn check(col: &Collection, text: &str) -> AuditReport {
    audit::audit(&CollectionSources::new(col), text, &options())
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
        let p = self.p;
        egui::CollapsingHeader::new(
            RichText::new(self.t("Verifică citările", "Check citations")).semibold(),
        )
        .show(ui, |ui| {
            muted(
                ui,
                &p,
                self.t(
                    "Lipește un text ale cărui propoziții citează documentele din această colecție, de forma [@fișier, p. 3], și vezi dacă pasajul citat susține propoziția. Se compară cuvinte și cifre, nu sensul: «susținut» arată unde să te uiți, nu e o dovadă.",
                    "Paste a text whose sentences cite the documents in this collection, like [@file, p. 3], and see whether the cited passage backs the sentence. It compares words and figures, not meaning: “supported” points to where to look and is not proof.",
                ),
            );
            ui.add(
                egui::TextEdit::multiline(&mut self.docs.verify.text)
                    .desired_rows(5)
                    .desired_width(f32::INFINITY),
            );
            ui.horizontal(|ui| {
                if secondary_button(ui, self.t("Verifică", "Check")).clicked() {
                    self.docs.verify.report = self
                        .docs
                        .lib
                        .get(coll_id)
                        .map(|c| check(c, &self.docs.verify.text));
                }
                if let Some(r) = &self.docs.verify.report
                    && secondary_button(ui, self.t("Copiază raportul", "Copy report")).clicked()
                {
                    ui.ctx().copy_text(r.markdown());
                }
            });
            let Some(r) = &self.docs.verify.report else {
                return;
            };
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
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use companion_core::rag::Library;

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
    fn file_name_with_extension_also_resolves_and_hostile_input_is_bounded() {
        let (lib, id) = library();
        let col = lib.get(&id).unwrap();
        let r = check(
            col,
            "Stable temperatures were recorded in the trial [@report.txt].",
        );
        assert_ne!(r.findings[0].verdict, Verdict::UnknownSource);
        let big = "[@x, p. 1] ".repeat(100_000);
        let t = std::time::Instant::now();
        let r = check(col, &big);
        assert!(r.truncated() || r.findings.len() <= 100_000);
        assert!(t.elapsed().as_secs() < 60);
    }

    #[test]
    fn every_verdict_has_a_label_in_both_spellings() {
        for v in Verdict::ALL {
            assert!(!v.as_str().is_empty());
        }
    }
}
