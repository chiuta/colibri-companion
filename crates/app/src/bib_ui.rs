//! "Bibliography": paste a BibTeX, RIS or CSL JSON bibliography and copy it back out as any of
//! the three, or as a formatted reference list (companion-scholar, ported from TEZAUR). Nothing
//! leaves the computer. The reader keeps what it understands and counts what it could not.

use crate::app::CompanionApp;
use crate::theme::Semibold;
use crate::widgets::*;
use companion_scholar::bib::{self, Entry};
use companion_scholar::csljson;
use companion_scholar::export::{self, Style};
use egui::{RichText, Ui};

type Maker = Box<dyn Fn(&[Entry]) -> String>;

#[derive(Default)]
pub struct BibState {
    pub text: String,
    pub entries: Vec<Entry>,
    pub warnings: usize,
    pub read: bool,
    pub failed: Option<String>,
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
        let outputs: [(&str, Maker); 6] = [
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
        ];
        ui.horizontal_wrapped(|ui| {
            for (name, make) in &outputs {
                if secondary_button(ui, &copy_t.replacen("{}", name, 1)).clicked() {
                    let s = make(&self.docs.bib.entries);
                    ui.ctx().copy_text(s);
                }
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
