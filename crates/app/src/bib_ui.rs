//! "Bibliography": paste a BibTeX, RIS or CSL JSON bibliography and copy it back out as any of
//! the three, or as a formatted reference list (companion-scholar, ported from TEZAUR). Nothing
//! leaves the computer. The reader keeps what it understands and counts what it could not.

use crate::app::CompanionApp;
use crate::theme::Semibold;
use crate::widgets::*;
use companion_scholar::bib::{self, Entry};
use companion_scholar::csljson;
use companion_scholar::dedup;
use companion_scholar::export::{self, Style};
use companion_scholar::ids;
use egui::{RichText, Ui};

pub(crate) type Maker = Box<dyn Fn(&[Entry]) -> String>;

#[derive(Default)]
pub struct BibState {
    pub text: String,
    pub entries: Vec<Entry>,
    pub warnings: usize,
    pub read: bool,
    pub failed: Option<String>,
    /// The duplicate report of the last "Find duplicates".
    pub dups: String,
    /// Identifiers found in the pasted text: (display text, verdict).
    pub ids: Vec<(String, companion_scholar::ids::Verdict)>,
    pub ids_done: bool,
    /// Text with `[@key, p. 3]` citations, the style chosen and the result.
    pub cite_in: String,
    pub cite_style: usize,
    pub cite_out: String,
    pub cite_refs: String,
}

/// Largest paste the box takes (characters).
const MAX_INPUT_CHARS: usize = 4 << 20;

pub fn as_bibtex(entries: &[Entry]) -> String {
    entries
        .iter()
        .map(|e| e.to_bibtex())
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn as_ris(entries: &[Entry]) -> String {
    entries
        .iter()
        .map(|e| e.to_ris())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The eight ways a list of entries can be copied out.
pub(crate) fn formats() -> [(&'static str, Maker); 8] {
    [
        ("BibTeX", Box::new(as_bibtex)),
        ("RIS", Box::new(as_ris)),
        ("CSL JSON", Box::new(csljson::to_json)),
        ("APA", Box::new(|e| export::bibliography(e, Style::Apa))),
        (
            "Harvard",
            Box::new(|e| export::bibliography(e, Style::AuthorYear)),
        ),
        (
            "IEEE",
            Box::new(|e| export::bibliography(e, Style::Numeric)),
        ),
        (
            "Chicago",
            Box::new(|e| export::bibliography(e, Style::Chicago)),
        ),
        (
            "Vancouver",
            Box::new(|e| export::bibliography(e, Style::Vancouver)),
        ),
    ]
}

impl BibState {
    /// Reads `self.text`; whatever cannot be read is counted, never fatal.
    pub fn read_text(&mut self) {
        self.failed = None;
        match bib::parse_any("", &self.text) {
            Ok((entries, warnings)) => {
                self.entries = entries;
                self.warnings = warnings.len();
                self.read = true;
            }
            Err(e) => {
                self.entries.clear();
                self.warnings = 0;
                self.read = false;
                self.failed = Some(e);
            }
        }
    }
}

impl CompanionApp {
    pub fn bib_section(&mut self, ui: &mut Ui) {
        egui::CollapsingHeader::new(
            RichText::new(self.t("Bibliografie", "Bibliography")).semibold(),
        )
        .show(ui, |ui| self.bib_body(ui));
    }

    pub fn bib_body(&mut self, ui: &mut Ui) {
        let p = self.p;
        muted(
            ui,
            &p,
            self.t(
                "Lipește o bibliografie (BibTeX, RIS sau CSL JSON) ca s-o convertești sau s-o tipărești ca listă de referințe. Nimic nu iese din calculator. Câmpurile neînțelese sunt numărate, nu ghicite.",
                "Paste a bibliography (BibTeX, RIS or CSL JSON) to convert it or print it as a reference list. Nothing leaves your computer. Fields it does not understand are counted, not guessed.",
            ),
        );
        let hint = self.t("Lipește aici bibliografia", "Paste the bibliography here");
        ui.add(
            egui::TextEdit::multiline(&mut self.docs.bib.text)
                .hint_text(hint)
                .char_limit(MAX_INPUT_CHARS)
                .desired_rows(5)
                .desired_width(f32::INFINITY),
        );
        if secondary_button(ui, self.t("Citește intrările", "Read entries")).clicked() {
            self.docs.bib.read_text();
        }
        if let Some(e) = self.docs.bib.failed.clone() {
            ui.label(RichText::new(e).color(p.bad));
        }
        if !self.docs.bib.read {
            return;
        }
        let n = self.docs.bib.entries.len();
        let w = self.docs.bib.warnings;
        let line = self
            .t(
                "Intrări citite: {}; avertismente: {}",
                "Entries read: {}; warnings: {}",
            )
            .replacen("{}", &n.to_string(), 1)
            .replacen("{}", &w.to_string(), 1);
        muted(ui, &p, &line);
        if n == 0 {
            return;
        }
        let copy_t = self.t("Copiază ca {}", "Copy as {}");
        let outputs = formats();
        ui.horizontal_wrapped(|ui| {
            for (name, make) in &outputs {
                if secondary_button(ui, &copy_t.replacen("{}", name, 1)).clicked() {
                    let s = make(&self.docs.bib.entries);
                    ui.ctx().copy_text(s);
                }
            }
        });
        let entries = self.docs.bib.entries.clone();
        self.save_formats_row(ui, "bibliography", &entries);
        if !self.docs.research.message.is_empty() {
            let m = self.docs.research.message.clone();
            ui.label(RichText::new(m).color(p.warn));
        }
        self.bib_tools(ui);
    }

    /// Duplicates, identifiers and citations in a text, on the entries read above.
    fn bib_tools(&mut self, ui: &mut Ui) {
        let p = self.p;
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            if secondary_button(ui, self.t("Caută dubluri", "Find duplicates")).clicked() {
                let b = &mut self.docs.bib;
                b.dups = dedup::analyze(&b.entries).render(b.entries.len());
            }
            if secondary_button(
                ui,
                self.t("Unifică dublurile sigure", "Merge the sure duplicates"),
            )
            .clicked()
            {
                let b = &mut self.docs.bib;
                let before = b.entries.len();
                let (merged, rep) = dedup::unify(&b.entries);
                b.entries = merged;
                b.dups = rep.render(before);
            }
            if secondary_button(ui, self.t("Verifică identificatorii", "Check identifiers"))
                .clicked()
            {
                let b = &mut self.docs.bib;
                b.ids = ids::extract(&b.text)
                    .into_iter()
                    .take(2000)
                    .map(|f| (f.display, f.verdict))
                    .collect();
                b.ids_done = true;
            }
        });
        if !self.docs.bib.dups.is_empty() {
            muted(ui, &p, &self.docs.bib.dups.clone());
        }
        if self.docs.bib.ids_done {
            if self.docs.bib.ids.is_empty() {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Niciun DOI, ISBN, ISSN sau ORCID în text.",
                        "No DOI, ISBN, ISSN or ORCID in the text.",
                    ),
                );
            }
            let ids = self.docs.bib.ids.clone();
            for (shown, v) in &ids {
                let (word, colour) = match v {
                    ids::Verdict::Valid => (self.t("valid", "valid"), p.ok),
                    ids::Verdict::BadChecksum => (
                        self.t("cifră de control greșită", "wrong check digit"),
                        p.bad,
                    ),
                    ids::Verdict::BadStructure => (self.t("formă greșită", "malformed"), p.bad),
                };
                ui.label(RichText::new(format!("{shown} · {word}")).color(colour));
            }
        }
        egui::CollapsingHeader::new(self.t("Citări în text", "Citations in text"))
            .id_salt("bib_cite_text")
            .show(ui, |ui| {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Lipește un text cu citări [@cheie, p. 3] și primești citările scrise în stilul ales, plus lista de referințe în ordinea citării. O cheie fără intrare rămâne neschimbată.",
                        "Paste a text with [@key, p. 3] citations and get the citations written in the chosen style, plus the reference list in order of citing. A key with no entry is left as written.",
                    ),
                );
                ui.add(
                    egui::TextEdit::multiline(&mut self.docs.bib.cite_in)
                        .char_limit(MAX_INPUT_CHARS)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY),
                );
                let styles = [("APA", Style::Apa), ("Harvard", Style::AuthorYear), ("IEEE", Style::Numeric), ("Chicago", Style::Chicago), ("Vancouver", Style::Vancouver)];
                ui.horizontal(|ui| {
                    let cur = self.docs.bib.cite_style.min(4);
                    egui::ComboBox::from_id_salt("bib_cite_style")
                        .selected_text(styles[cur].0)
                        .show_ui(ui, |ui| {
                            for (i, (n, _)) in styles.iter().enumerate() {
                                ui.selectable_value(&mut self.docs.bib.cite_style, i, *n);
                            }
                        });
                    if secondary_button(ui, self.t("Scrie citările", "Write the citations")).clicked() {
                        let b = &mut self.docs.bib;
                        let style = styles[b.cite_style.min(4)].1;
                        let (out, cited) = export::render_citations(&b.cite_in, &b.entries, style);
                        let list: Vec<Entry> = cited.into_iter().cloned().collect();
                        b.cite_refs = export::bibliography(&list, style);
                        b.cite_out = out;
                    }
                });
                if !self.docs.bib.cite_out.is_empty() {
                    let (out, refs) = (self.docs.bib.cite_out.clone(), self.docs.bib.cite_refs.clone());
                    ui.label(&out);
                    if !refs.trim().is_empty() {
                        muted(ui, &p, &refs);
                    }
                    ui.horizontal(|ui| {
                        if secondary_button(ui, self.t("Copiază textul", "Copy the text")).clicked() {
                            ui.ctx().copy_text(out.clone());
                        }
                        if secondary_button(ui, self.t("Copiază lista", "Copy the list")).clicked() {
                            ui.ctx().copy_text(refs.clone());
                        }
                    });
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BIB: &str = "@article{smith2019, author={Smith, Jane and Doe, Alan}, title={A study}, journal={Journal of Things}, year={2019}, volume={3}, pages={1--9}}\n";

    #[test]
    fn reads_bibtex_and_writes_all_formats() {
        let mut s = BibState {
            text: BIB.into(),
            ..Default::default()
        };
        s.read_text();
        assert!(s.read && s.entries.len() == 1);
        assert!(as_bibtex(&s.entries).contains("smith2019"));
        assert!(as_ris(&s.entries).contains("TY  -"));
        assert!(csljson::to_json(&s.entries).contains("A study"));
        let apa = export::bibliography(&s.entries, Style::Apa);
        assert!(apa.contains("Smith") && apa.contains("2019"), "{apa}");
    }

    #[test]
    fn round_trips_through_ris_and_csl() {
        let mut s = BibState {
            text: BIB.into(),
            ..Default::default()
        };
        s.read_text();
        for text in [as_ris(&s.entries), csljson::to_json(&s.entries)] {
            let mut t = BibState {
                text,
                ..Default::default()
            };
            t.read_text();
            assert_eq!(t.entries.len(), 1);
            assert_eq!(t.entries[0].title, "A study");
        }
    }

    #[test]
    fn garbage_gives_zero_entries_not_a_panic() {
        let mut s = BibState {
            text: "this is not a bibliography \u{0}\u{1}".into(),
            ..Default::default()
        };
        s.read_text();
        assert!(s.entries.is_empty());
    }
}
