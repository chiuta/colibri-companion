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
    pub fn light_models_section(&mut self, ui: &mut Ui) {
        let p = self.p;
        let ram = self.hardware.ready().map(|h| h.ram_total_gb).unwrap_or(0.0);
        let tier = if ram > 0.0 { tier_of(ram) } else { 16 };
        egui::CollapsingHeader::new(
            RichText::new(self.t("Modele ușoare pentru memoria ta", "Light models for your memory"))
                .semibold(),
        )
        .show(ui, |ui| {
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
            let mut only = self.light_agents_only;
            ui.checkbox(
                &mut only,
                self.t(
                    "Doar modele cu apel de unelte documentat (pentru agenți locali)",
                    "Only models with documented tool calling (for local agents)",
                ),
            );
            self.light_agents_only = only;
            let list = if only { lm::for_agents(tier) } else { lm::for_ram(tier) };
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
                    ui.label(format!(
                        "{}{} · {} {:.1} GB · {} {}",
                        m.params_total,
                        active,
                        self.t("descărcare", "download"),
                        m.size_gb,
                        self.t("context", "context"),
                        m.context
                    ));
                    // Catalogue wording is kept as the vendors printed it (English).
                    muted(ui, &p, m.strength);
                    muted(ui, &p, &lm::agent_note(m));
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
        });
    }
}

#[cfg(test)]
mod tests {
    use super::tier_of;

    #[test]
    fn a_sixteen_gb_computer_reaches_the_sixteen_tier() {
        assert_eq!(tier_of(15.9), 16);
        assert_eq!(tier_of(16.0), 16);
        assert_eq!(tier_of(7.3), 4);
        assert_eq!(tier_of(64.0), 32);
        assert_eq!(tier_of(2.0), 0);
    }
}
