//! "Light models": small models that fit the memory of this computer, with what their own
//! pages say about local agents. The catalogue is `companion_core::light_models`; nothing
//! here downloads or starts a model, it only shows the facts and copies the start command.

use crate::app::CompanionApp;
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::light_models as lm;
use egui::{RichText, Ui};

/// The largest catalogue tier that this much memory reaches. The 0.6 GB slack is because a
/// "16 GB" computer reports about 15.9 GB.
fn tier_of(ram_gb: f64) -> u32 {
    lm::TIERS
        .iter()
        .copied()
        .filter(|t| f64::from(*t) <= ram_gb + 0.6)
        .max()
        .unwrap_or(0)
}

impl CompanionApp {
    /// What the model's own pages document for local agents, in the interface language. The
    /// catalogue's free-text note stays as the vendor printed it (English).
    fn agent_line(&self, m: &lm::LightModel) -> String {
        let word = |s: lm::Support| match s {
            lm::Support::Documented(_) => self.t("documentat", "documented"),
            lm::Support::Unverified => self.t("neverificat", "unverified"),
            lm::Support::No => self.t("nesuportat", "not supported"),
        };
        format!(
            "{}: {}; {}: {}; {}: {}. {}",
            self.t("Unelte", "Tools"),
            word(m.agent.tool_calling),
            self.t("Programare agentică", "Agentic coding"),
            word(m.agent.agentic_coding),
            self.t("Ieșire structurată", "Structured output"),
            word(m.agent.structured_output),
            m.agent.notes
        )
    }

    pub fn light_models_section(&mut self, ui: &mut Ui) {
        egui::CollapsingHeader::new(
            RichText::new(self.t(
                "Modele ușoare pentru memoria ta",
                "Light models for your memory",
            ))
            .semibold(),
        )
        .show(ui, |ui| self.light_models_body(ui));
    }

    fn light_models_body(&mut self, ui: &mut Ui) {
        let p = self.p;
        let ram = self.hardware.ready().map(|h| h.ram_total_gb).unwrap_or(0.0);
        let tier = if ram > 0.0 { tier_of(ram) } else { 16 };
        {
            muted(
                ui,
                &p,
                self.t(
                    "Modele mici, care încap confortabil în memoria calculatorului. Rulează printr-un server precum Ollama; adresa lui se pune în Setări.",
                    "Small models that fit comfortably in this computer's memory. They run through a server such as Ollama; set its address in Settings.",
                ),
            );
            ui.label(format!(
                "{} {} GB",
                self.t("Treapta de memorie folosită:", "Memory tier used:"),
                tier
            ));
            if ram <= 0.0 {
                // nothing was measured: say that the 16 GB tier is an assumption, not a finding
                muted(
                    ui,
                    &p,
                    self.t(
                        "Memoria nu a fost detectată încă: se arată treapta de 16 GB.",
                        "Memory not detected yet: showing the 16 GB tier.",
                    ),
                );
            }
            let mut only = self.light_agents_only;
            ui.checkbox(
                &mut only,
                self.t(
                    "Doar modele cu apel de unelte documentat (pentru agenți locali)",
                    "Only models with documented tool calling (for local agents)",
                ),
            );
            self.light_agents_only = only;
            let list = if only {
                lm::for_agents(tier)
            } else {
                lm::for_ram(tier)
            };
            if list.is_empty() {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Niciun model ușor din catalog nu se potrivește acestei trepte de memorie.",
                        "No light model in the catalogue fits this memory tier.",
                    ),
                );
            }
            for m in list {
                ui.add_space(4.0);
                ui.group(|ui| {
                    let active = m.params_active.map(|a| format!(" / {a}")).unwrap_or_default();
                    ui.label(
                        RichText::new(format!("{} · {}", m.name, m.vendor)).semibold(),
                    );
                    // the sizes in the catalogue ("5.1B", "256K (262,144)") are printed as the vendor
                    // wrote them; only the download size is a number of ours
                    ui.label(format!(
                        "{}{} · {} {} · {} {}",
                        m.params_total,
                        active,
                        self.t("descărcare", "download"),
                        gb1(m.size_gb),
                        self.t("context", "context"),
                        m.context
                    ));
                    // Catalogue wording is kept as the vendors printed it (English).
                    muted(ui, &p, m.strength);
                    muted(ui, &p, &self.agent_line(m));
                    if m.engine == lm::Engine::Colibri {
                        muted(
                            ui,
                            &p,
                            self.t(
                                "Rulează cu colibri, după un pas de conversie; vezi README-ul lui.",
                                "Runs with colibri after a conversion step; see its README.",
                            ),
                        );
                    } else if lm::ollama_command(m).is_none() {
                        muted(
                            ui,
                            &p,
                            self.t(
                                "Aici nu există comandă de pornire: ia fișierele de la sursă.",
                                "No start command here: get the files from the source link.",
                            ),
                        );
                    }
                    if !m.permissive {
                        ui.label(
                            RichText::new(self.t(
                                "Licența nu este una permisivă obișnuită: citește-o înainte să folosești modelul.",
                                "The licence is not a standard permissive one: read it before use.",
                            ))
                            .color(p.bad),
                        );
                    }
                    ui.horizontal(|ui| {
                        if let Some(cmd) = lm::ollama_command(m)
                            && secondary_button(
                                ui,
                                self.t("Copiază comanda de pornire", "Copy start command"),
                            )
                            .clicked()
                        {
                            ui.ctx().copy_text(cmd);
                            self.toast(self.t(
                                "Comanda e copiată. Rulează-o într-un terminal, apoi indică serverul în Setări.",
                                "Command copied. Run it in a terminal, then point the app to the server in Settings.",
                            ));
                        }
                        if let Some(src) = m.sources.first() {
                            ui.hyperlink_to(self.t("Sursa", "Source"), *src);
                        }
                    });
                });
            }
            ui.add_space(4.0);
            muted(
                ui,
                &p,
                &format!(
                    "{} {}",
                    self.t(
                        "Cifrele sunt citite de pe paginile furnizorilor la data de",
                        "Figures were read from the vendors' own pages on",
                    ),
                    lm::CHECKED
                ),
            );
            muted(
                ui,
                &p,
                self.t(
                    "Nimic de aici nu a fost rulat în această aplicație, iar o insignă de unelte nu face un model mic fiabil.",
                    "Nothing here was run in this app, and a tools badge does not make a small model reliable.",
                ),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tier_of;
    use crate::app::CompanionApp;
    use crate::widgets::testkit;
    use companion_core::Lang;
    use companion_core::light_models as lm;
    use companion_core::store::Settings;

    fn app(lang: Lang) -> CompanionApp {
        CompanionApp::new(
            egui::Context::default(),
            Settings {
                lang,
                ..Settings::default()
            },
            Some("http://127.0.0.1:1".into()),
            false,
        )
    }

    fn draw(a: &mut CompanionApp, w: f32) -> Vec<testkit::Drawn> {
        let ctx = a.ctx.clone();
        let f = |a: &mut CompanionApp| {
            testkit::draw(&ctx, w, 4000.0, |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| a.light_models_body(ui));
                });
            })
        };
        f(a);
        f(a)
    }

    #[test]
    fn a_sixteen_gb_computer_reaches_the_sixteen_tier() {
        assert_eq!(tier_of(15.9), 16);
        assert_eq!(tier_of(16.0), 16);
        assert_eq!(tier_of(7.3), 4);
        assert_eq!(tier_of(64.0), 32);
        assert_eq!(tier_of(2.0), 0);
    }

    #[test]
    fn the_section_fits_and_is_translated_in_every_language() {
        let mut problems = Vec::new();
        for lang in Lang::ALL {
            let mut a = app(lang);
            a.light_agents_only = false;
            let drawn = draw(&mut a, 560.0);
            assert!(drawn.len() > 20, "{lang:?}: {} lines", drawn.len());
            for d in testkit::clipped(&drawn) {
                problems.push(format!("{lang:?}: '{}' is cut off", d.text));
            }
            // the capability line is ours, so it is in the interface language, not English
            let line = drawn
                .iter()
                .find(|d| d.text.contains(';') && d.text.contains(':'));
            if lang != Lang::En {
                let en = ["documented", "unverified", "Agentic coding"];
                if let Some(l) = line {
                    assert!(
                        !en.iter().any(|w| l.text.contains(w)),
                        "{lang:?}: English left in '{}'",
                        l.text
                    );
                }
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    #[test]
    fn an_unknown_memory_size_is_said_not_silently_assumed() {
        let mut a = app(Lang::En);
        let drawn = draw(&mut a, 560.0);
        assert!(
            drawn
                .iter()
                .any(|d| d.text.contains("Memory not detected yet")),
            "the 16 GB tier is an assumption and must say so"
        );
    }

    #[test]
    fn every_listed_model_says_how_to_start_it_or_why_there_is_no_command() {
        let mut a = app(Lang::En);
        a.light_agents_only = false;
        let drawn = draw(&mut a, 560.0);
        let text: String = drawn
            .iter()
            .map(|d| d.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        for m in lm::for_ram(16) {
            if m.engine == lm::Engine::Colibri {
                assert!(text.contains("conversion step"), "{}", m.id);
            } else if lm::ollama_command(m).is_none() {
                assert!(text.contains("No start command here"), "{}", m.id);
            }
        }
    }
}
