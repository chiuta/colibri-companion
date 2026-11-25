//! "Citations": the documents of a collection as citable sources. Each one gets a citation
//! (`[@name]`, exactly what "Check citations" recognises) and the whole list can be copied out as
//! BibTeX, RIS, CSL JSON or a formatted reference list. A bibliography pasted in the section
//! below lends its authors, year and title to the documents it describes.

use crate::app::CompanionApp;
use crate::bib_ui::formats;
use crate::theme::Semibold;
use crate::verify_ui::stem;
use crate::widgets::*;
use companion_core::rag::DocInfo;
use companion_scholar::bib::Entry;
use companion_scholar::keys::KeySet;
use egui::{RichText, Ui};

/// The citation text a document is cited with.
pub fn cite_text(doc_name: &str) -> String {
    format!("[@{}]", stem(doc_name))
}

fn same(a: &str, b: &str) -> bool {
    !a.trim().is_empty() && a.trim().to_lowercase() == b.trim().to_lowercase()
}

/// One entry per document: the matching pasted entry (its `file` or `title` equals the document's
/// name or stem) when there is one, a bare entry named after the file otherwise. Keys are unique.
/// The second value is how many documents found a pasted entry.
pub fn entries_for(docs: &[DocInfo], known: &[Entry]) -> (Vec<Entry>, usize) {
    let mut keys = KeySet::new();
    let mut used = 0;
    let mut out = Vec::with_capacity(docs.len());
    for d in docs {
        let s = stem(&d.name);
        let hit = known.iter().find(|k| {
            same(&k.file, &d.name)
                || same(&k.file, &s)
                || same(&k.title, &s)
                || same(&k.title, &d.name)
        });
        let mut e = match hit {
            Some(k) => {
                used += 1;
                k.clone()
            }
            None => Entry {
                title: s.clone(),
                file: d.name.clone(),
                ..Entry::default()
            },
        };
        e.key = keys.unique(&e, &e.key.clone());
        if e.file.is_empty() {
            e.file = d.name.clone();
        }
        out.push(e);
    }
    (out, used)
}

impl CompanionApp {
    pub fn cite_section(&mut self, ui: &mut Ui, docs: &[DocInfo]) {
        egui::CollapsingHeader::new(RichText::new(self.t("Citări", "Citations")).semibold())
            .id_salt("cite_section")
            .show(ui, |ui| {
                let p = self.p;
                muted(
                    ui,
                    &p,
                    self.t(
                        "Documentele colecției ca surse citabile. Copiază citarea ([@nume]) în textul tău sau exportă toată lista. Verificarea de mai jos recunoaște exact aceste citări.",
                        "The collection's documents as citable sources. Copy the citation ([@name]) into your text or export the whole list. The check below recognises exactly these citations.",
                    ),
                );
                if docs.is_empty() {
                    muted(
                        ui,
                        &p,
                        self.t("Colecția nu are încă documente.", "The collection has no documents yet."),
                    );
                    return;
                }
                let known: &[Entry] = if self.docs.bib.read { &self.docs.bib.entries } else { &[] };
                let (entries, used) = entries_for(docs, known);
                if used > 0 {
                    let line = self
                        .t(
                            "Intrări din bibliografia lipită folosite: {}",
                            "Entries from the pasted bibliography used: {}",
                        )
                        .replacen("{}", &used.to_string(), 1);
                    muted(ui, &p, &line);
                }
                for d in docs {
                    ui.horizontal(|ui| {
                        ui.label(&d.name);
                        if ui.small_button(self.t("Copiază citarea", "Copy citation")).clicked() {
                            ui.ctx().copy_text(cite_text(&d.name));
                        }
                    });
                }
                let copy_t = self.t("Copiază ca {}", "Copy as {}");
                ui.horizontal_wrapped(|ui| {
                    for (name, make) in &formats() {
                        if secondary_button(ui, &copy_t.replacen("{}", name, 1)).clicked() {
                            ui.ctx().copy_text(make(&entries));
                        }
                    }
                });
                self.save_formats_row(ui, "sources", &entries);
                if !self.docs.research.message.is_empty() {
                    let m = self.docs.research.message.clone();
                    ui.label(RichText::new(m).color(p.warn));
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use companion_scholar::bib;

    fn doc(n: &str) -> DocInfo {
        DocInfo {
            name: n.into(),
            chars: 10,
            chunks: 1,
        }
    }

    #[test]
    fn every_document_gets_a_unique_key_and_all_exports_read_back() {
        let docs = [doc("report.pdf"), doc("Report.PDF"), doc("notes 2.5.txt")];
        let (e, used) = entries_for(&docs, &[]);
        assert_eq!(used, 0);
        let keys: std::collections::HashSet<_> = e.iter().map(|x| x.key.to_lowercase()).collect();
        assert_eq!(keys.len(), 3, "keys must differ: {e:?}");
        for (name, make) in formats() {
            assert!(!make(&e).trim().is_empty(), "{name} is empty");
        }
        let back = bib::parse_any("", &formats()[0].1(&e)).unwrap().0;
        assert_eq!(back.len(), 3);
        let back = bib::parse_any("", &formats()[2].1(&e)).unwrap().0;
        assert_eq!(back.len(), 3);
    }

    #[test]
    fn a_pasted_entry_lends_its_details_to_the_document_it_describes() {
        let known = bib::parse(
            "@book{x, author={Ionescu, Ana}, year={2020}, title={Teza}, file={teza.pdf}}",
        );
        let (e, used) = entries_for(&[doc("teza.pdf"), doc("altceva.txt")], &known);
        assert_eq!(used, 1);
        assert_eq!(e[0].year, "2020");
        assert!(e[0].author.contains("Ionescu"));
        assert_eq!(e[1].title, "altceva");
    }

    #[test]
    fn the_citation_text_is_what_the_check_recognises() {
        assert_eq!(cite_text("report.pdf"), "[@report]");
        assert_eq!(cite_text("Chapter 2.5"), "[@Chapter 2.5]");
    }
}
