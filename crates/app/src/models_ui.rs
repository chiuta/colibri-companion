//! "My models": every model on this computer, switching between them, adding more and
//! freeing space. Switching and adding both go through colibri's own `setup --model`.

use crate::app::{CompanionApp, ModelAsk, Phase};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::Backend;
use companion_core::coli::{self, Engine, ModelInfo};
use companion_core::models::InstalledModel;
use companion_core::proc;
use egui::{Align2, RichText, Ui, vec2};

impl CompanionApp {
    pub fn models_window(&mut self, ctx: &egui::Context) {
        if !self.show_models {
            return;
        }
        self.refresh_installed_if_stale();
        let mut open = true;
        egui::Window::new(self.t("Modelele mele", "My models"))
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_width(640.0)
            .default_height(600.0)
            .collapsible(false)
            .vscroll(true)
            .show(ctx, |ui| self.models_body(ui));
        if !open {
            self.show_models = false;
            self.model_ask = None;
        }
    }

    fn models_body(&mut self, ui: &mut Ui) {
        let p = self.p;
        let busy = self.phase() == Phase::Installing || self.install_spawning;
        ui.label(self.t(
            "Poți păstra mai multe modele pe calculator și poți trece de la unul la altul oricând. Rulează câte unul singur, pentru a nu încărca memoria.",
            "You can keep several models on this computer and switch between them any time. Only one runs at a time, so memory isn't overloaded.",
        ));
        ui.add_space(6.0);
        self.light_models_section(ui);
        ui.add_space(6.0);
        egui::CollapsingHeader::new(
            RichText::new(self.t("Am deja un model descărcat", "I already have a model downloaded"))
                .semibold(),
        )
        .show(ui, |ui| {
            muted(
                ui,
                &p,
                self.t(
                    "Dacă ai fișiere .gguf (de ex. «…-00001-of-00008.gguf»), alege primul fișier; celelalte părți trebuie să se afle în același folder. colibri le convertește într-un folder de model alăturat (poate dura și ocupă spațiu suplimentar), apoi îl pornește. Poți alege și un folder de model colibri deja pregătit.",
                    "If you have .gguf files (for example “…-00001-of-00008.gguf”), pick the first file; the other parts must be in the same folder. colibri converts them into a model folder beside them (this can take a while and needs extra space), then I start it. You can also pick a ready-made colibri model folder.",
                ),
            );
            let hint = self.t(
                "Calea primului fișier .gguf sau a folderului",
                "Path of the first .gguf file or of the folder",
            );
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.existing_path)
                        .desired_width(340.0)
                        .hint_text(hint),
                );
                ui.add_enabled_ui(!busy && !self.existing_path.trim().is_empty(), |ui| {
                    if ui.button(self.t("Utilizează acest model", "Use this model")).clicked() {
                        let path = self.existing_path.trim().trim_matches('"').into();
                        self.use_existing_model(path);
                    }
                });
            });
            muted(
                ui,
                &p,
                self.t(
                    "Poți, de asemenea, să tragi fișierul în fereastra aplicației.",
                    "You can also drag the file onto the app window.",
                ),
            );
        });
        ui.add_space(6.0);
        if busy {
            tinted(ui, p.warn_soft, p.warn, |ui| {
                ui.label(
                    RichText::new(self.t(
                        "Se instalează un model chiar acum. Urmărește progresul în fereastra principală.",
                        "A model is installing right now. Follow its progress in the main window.",
                    ))
                    .color(p.warn),
                );
            });
            ui.add_space(6.0);
        }

        self.performance_panel(ui);
        let have_ids: Vec<String> = self.installed.iter().map(|i| i.id.clone()).collect();
        let top = self.strongest_card(ui, true);
        let _ = have_ids;

        // ---- installed
        ui.label(
            RichText::new(self.t(
                "Instalate pe acest calculator",
                "Installed on this computer",
            ))
            .size(17.0)
            .semibold(),
        );
        let installed = self.installed.clone();
        if installed.is_empty() {
            muted(
                ui,
                &p,
                self.t(
                    "Nu am găsit încă niciun model descărcat.",
                    "I haven't found any downloaded model yet.",
                ),
            );
        }
        let active_id = self
            .status
            .as_ref()
            .and_then(|s| s.setup.as_ref())
            .and_then(|s| s.model.as_ref())
            .and_then(|m| m.id.clone());
        for m in &installed {
            self.installed_row(ui, m, active_id.as_deref(), busy);
            ui.add_space(6.0);
        }

        // ---- add
        ui.add_space(8.0);
        ui.label(
            RichText::new(self.t("Adaugă un alt model", "Add another model"))
                .size(17.0)
                .semibold(),
        );
        self.model_filter_bar(ui);
        match self.catalog.clone() {
            crate::app::Loadable::Ready(models) => {
                let have: Vec<&str> = installed.iter().map(|i| i.id.as_str()).collect();
                let (fit, nofit): (Vec<_>, Vec<_>) = models
                    .iter()
                    .filter(|m| m.is_chat_model() && !have.contains(&m.id.as_str()))
                    .filter(|m| top.as_deref() != Some(m.id.as_str()))
                    .filter(|m| self.model_shown(m))
                    .partition(|m| m.fits);
                if fit.is_empty() && nofit.is_empty() && top.is_none() && !self.models_filtered() {
                    muted(
                        ui,
                        &p,
                        self.t(
                            "Ai deja toate modelele de conversație din listă.",
                            "You already have every chat model on the list.",
                        ),
                    );
                }
                for m in &fit {
                    self.add_row(ui, m, busy);
                    ui.add_space(6.0);
                }
                if !nofit.is_empty() {
                    let label = format!(
                        "{} ({})",
                        self.t(
                            "Modele care nu se potrivesc acum",
                            "Models that don't fit right now"
                        ),
                        nofit.len()
                    );
                    egui::CollapsingHeader::new(label).show(ui, |ui| {
                        for m in &nofit {
                            self.add_row(ui, m, busy);
                            ui.add_space(6.0);
                        }
                    });
                }
            }
            crate::app::Loadable::Failed(e) => {
                ui.label(RichText::new(e).color(p.bad));
                if secondary_button(ui, self.t("Încearcă din nou", "Try again")).clicked() {
                    self.refresh_engine_info();
                }
            }
            _ => {
                ui.horizontal(|ui| {
                    crate::widgets::spinner(ui);
                    ui.label(self.t("Citesc lista de modele…", "Reading the model list…"));
                });
            }
        }
        ui.add_space(6.0);
        muted(
            ui,
            &p,
            self.t(
                "Modelele vin de pe Hugging Face și au fiecare propria licență.",
                "Models come from Hugging Face and each has its own licence.",
            ),
        );
    }

    fn installed_row(&mut self, ui: &mut Ui, m: &InstalledModel, active: Option<&str>, busy: bool) {
        let p = self.p;
        let name = self
            .catalog
            .ready()
            .and_then(|c| c.iter().find(|x| x.id == m.id))
            .map(|x| x.name.clone())
            .unwrap_or_else(|| m.id.clone());
        let is_active = active == Some(m.id.as_str());
        let running = is_active && self.server_ready() && !self.demo;
        let gb = m.total_bytes as f64 / 1e9;
        card(ui, &p, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(&name).size(18.0).semibold());
                if running {
                    pill(
                        ui,
                        self.t("Rulează acum", "Running now"),
                        p.on_accent,
                        p.accent,
                    );
                } else if is_active && m.complete {
                    pill(ui, self.t("Ales", "Selected"), p.ok, p.accent_soft);
                }
                if !m.complete {
                    pill(
                        ui,
                        &format!(
                            "{} {:.0}%",
                            self.t("Descărcare neterminată", "Unfinished download"),
                            m.fraction() * 100.0
                        ),
                        p.warn,
                        p.warn_soft,
                    );
                }
            });
            ui.label(
                RichText::new(format!("{} · {}", fmt_gb(gb), m.dir.display()))
                    .small()
                    .color(p.muted),
            );
            if self.model_ask == Some(ModelAsk::Delete(m.id.clone())) {
                ui.add_space(4.0);
                tinted(ui, p.bad_soft, p.bad, |ui| {
                    let msg = self
                        .t(
                            "Șterge «{}»? Se eliberează {}. Îl poți descărca din nou oricând.",
                            "Delete “{}”? This frees {}. You can download it again any time.",
                        )
                        .replacen("{}", &name, 1)
                        .replacen("{}", &fmt_gb(gb), 1);
                    ui.label(msg);
                    ui.horizontal(|ui| {
                        if secondary_button(ui, self.t("Da, șterge", "Yes, delete")).clicked() {
                            self.delete_model(&m.id);
                            self.model_ask = None;
                        }
                        if secondary_button(ui, self.t("Renunț", "Cancel")).clicked() {
                            self.model_ask = None;
                        }
                    });
                });
                return;
            }
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!busy && !running, |ui| {
                    let label = if !m.complete {
                        self.t("Continuă descărcarea", "Resume the download")
                    } else if is_active {
                        self.t("Pornește acest model", "Start this model")
                    } else {
                        self.t("Utilizează acest model", "Use this model")
                    };
                    if primary_button(ui, &p, label).clicked() {
                        self.switch_model(&m.id);
                    }
                });
                ui.add_enabled_ui(!busy && !is_active, |ui| {
                    if secondary_button(ui, self.t("Șterge", "Delete")).clicked() {
                        self.model_ask = Some(ModelAsk::Delete(m.id.clone()));
                    }
                });
                if secondary_button(ui, self.t("Deschide folderul", "Open folder")).clicked() {
                    proc::open_externally(&m.dir.display().to_string());
                }
            });
        });
    }

    fn add_row(&mut self, ui: &mut Ui, m: &ModelInfo, busy: bool) {
        let p = self.p;
        let asked = self.model_ask == Some(ModelAsk::Install(m.id.clone()));
        card(ui, &p, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(&m.name).size(18.0).semibold());
                if m.recommended {
                    pill(
                        ui,
                        self.t("Recomandat pentru tine", "Recommended for you"),
                        p.on_accent,
                        p.accent,
                    );
                }
                let (txt, fg, bg) = self.status_badge(m);
                pill(ui, txt, fg, bg);
            });
            ui.label(RichText::new(self.blurb(m)).color(p.muted));
            ui.label(format!(
                "{} {} · RAM: {} {} / {} {}",
                self.t("Descărcare:", "Download:"),
                fmt_gb(m.download_gb),
                self.t("minim", "min"),
                fmt_gb(m.ram_min_gb),
                self.t("recomandat", "recommended"),
                fmt_gb(m.ram_good_gb)
            ));
            if m.fits {
                self.estimate_card_line(ui, m);
            }
            if !m.fits && !m.reason.is_empty() {
                ui.label(
                    RichText::new(crate::i18n::engine_text(self.s.lang, &m.reason))
                        .size(13.0)
                        .color(p.bad),
                );
                return;
            }
            if !m.license_note.is_empty() {
                ui.label(
                    RichText::new(format!(
                        "{} {}",
                        self.t("Licență:", "Licence:"),
                        crate::i18n::engine_text(self.s.lang, &m.license_note)
                    ))
                    .color(p.warn)
                    .semibold(),
                );
            }
            if !asked {
                ui.add_enabled_ui(!busy, |ui| {
                    if secondary_button(ui, self.t("Instalează acest model", "Install this model"))
                        .clicked()
                    {
                        self.model_ask = Some(ModelAsk::Install(m.id.clone()));
                        self.license_ok = false;
                    }
                });
                return;
            }
            ui.separator();
            let free = self
                .hardware
                .ready()
                .map(|h| h.disk_free_gb)
                .unwrap_or(f64::MAX);
            let room = free >= m.download_gb * 1.05;
            if !room {
                ui.label(
                    RichText::new(self.t(
                        "Spațiu pe disc prea mic pentru acest model.",
                        "Not enough free disk space for this model.",
                    ))
                    .color(p.bad)
                    .semibold(),
                );
            }
            if self.server_ready() && !self.demo {
                ui.label(
                    RichText::new(self.t(
                        "Modelul care rulează în prezent va fi oprit cât timp se descarcă acesta. Rămâne instalat și îl poți relua oricând.",
                        "The model that is running now will stop while this one downloads. It stays installed and you can switch back any time.",
                    ))
                    .color(p.muted),
                );
            }
            ui.horizontal_wrapped(|ui| {
                ui.label(self.t(
                    "Modelul are propria licență.",
                    "The model has its own licence.",
                ));
                if ui
                    .link(self.t("Vezi pagina modelului", "See the model page"))
                    .clicked()
                {
                    proc::open_externally(&m.hub_url());
                }
            });
            let agree = self.t(
                "Am văzut licența modelului și sunt de acord cu ea",
                "I've seen the model's licence and agree to it",
            );
            ui.checkbox(&mut self.license_ok, agree);
            ui.horizontal(|ui| {
                ui.add_enabled_ui(self.license_ok && room, |ui| {
                    if primary_button(ui, &p, self.t("Descarcă și utilizează", "Download and use"))
                        .clicked()
                    {
                        self.install_additional(m);
                    }
                });
                if secondary_button(ui, self.t("Renunț", "Cancel")).clicked() {
                    self.model_ask = None;
                }
            });
        });
    }

    /// What speeds the model up on this computer, and the choice to override it.
    pub fn performance_panel(&mut self, ui: &mut Ui) {
        let p = self.p;
        let Some(hw) = self.hardware.ready().cloned() else {
            return;
        };
        let engine = hw.auto_engine();
        let (line, good) = match &engine {
            Engine::Cuda(name) => {
                let vram = hw.nvidia.first().map(|c| c.vram_gb).unwrap_or(0.0);
                (
                    format!(
                        "{} {name} ({}) — {}",
                        self.t("Placă video NVIDIA găsită:", "NVIDIA graphics card found:"),
                        fmt_gb(vram),
                        self.t(
                            "o folosesc la maximum cu CUDA",
                            "I use it to the fullest with CUDA"
                        )
                    ),
                    true,
                )
            }
            Engine::Vulkan(name) => (
                format!(
                    "{} {name} — {}",
                    self.t("Cip grafic găsit:", "Graphics chip found:"),
                    self.t(
                        "îl folosesc prin Vulkan pentru accelerare",
                        "I use it through Vulkan for acceleration"
                    )
                ),
                true,
            ),
            Engine::Metal => (
                self.t(
                    "Mac cu Apple Silicon: folosesc cipul grafic al Mac-ului prin Metal (memorie unificată).",
                    "Apple Silicon Mac: I use the Mac's graphics chip through Metal (unified memory).",
                )
                .to_string(),
                true,
            ),
            Engine::Cpu => (
                self.t(
                    "Nu am găsit o placă video cu suport de calcul: modelul rulează pe procesor. Merge și așa, doar mai încet.",
                    "No graphics card with compute support found: the model runs on the processor. That works, just more slowly.",
                )
                .to_string(),
                false,
            ),
        };
        let before = self.s.backend;
        egui::CollapsingHeader::new(
            RichText::new(self.t("Viteză și accelerare", "Speed and acceleration")).semibold(),
        )
        .default_open(false)
        .show(ui, |ui| {
            ui.label(RichText::new(line).color(if good { p.ok } else { p.muted }));
            ui.add_space(4.0);
            ui.label(self.t("Cum rulează modelul:", "How the model runs:"));
            ui.horizontal_wrapped(|ui| {
                for b in Backend::ALL {
                    let name = match b {
                        Backend::Auto => self.t(
                            "Automat (cel mai bun disponibil)",
                            "Automatic (best available)",
                        ),
                        Backend::Cuda => self.t("NVIDIA CUDA", "NVIDIA CUDA"),
                        Backend::Vulkan => self.t(
                            "Vulkan (AMD, Intel, NVIDIA)",
                            "Vulkan (AMD, Intel, NVIDIA)",
                        ),
                        Backend::Cpu => self.t("Doar procesorul", "Processor only"),
                    };
                    ui.selectable_value(&mut self.s.backend, b, name);
                }
            });
            muted(
                ui,
                &p,
                self.t(
                    "«Automat» alege singur CUDA pentru o placă NVIDIA, altfel Vulkan, altfel procesorul, și reconstruiește motorul dacă instalezi ulterior o placă video. Modificarea se aplică la următoarea instalare sau comutare de model.",
                    "“Automatic” picks CUDA for an NVIDIA card, otherwise Vulkan, otherwise the processor, and rebuilds the engine if you add a graphics card later. A change applies the next time you install or switch a model.",
                ),
            );
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                let label = self.t(
                    "Utilizează cipul grafic al Mac-ului (Metal, experimental în colibri)",
                    "Use the Mac's graphics chip (Metal, experimental in colibri)",
                );
                ui.checkbox(&mut self.s.apple_gpu, label);
            }
            match self.npu.clone() {
                Some(name) if companion_core::npu::colibri_can_use(&name) => {
                    ui.label(format!(
                        "{} {name}",
                        self.t("Cip de AI găsit:", "AI chip found:")
                    ));
                    let label = self.t(
                        "Utilizează cipul de AI (NPU), experimental: precizie redusă, răspunsurile pot diferi",
                        "Use the AI chip (NPU), experimental: reduced precision, answers may differ",
                    );
                    ui.checkbox(&mut self.s.use_npu, label);
                    muted(
                        ui,
                        &p,
                        self.t(
                            "Funcționează doar cu modele GLM și cere pachetul XDNA al lui colibri; dacă nu e utilizabil, colibri continuă pe calea obișnuită.",
                            "It only works with GLM models and needs colibri's XDNA package; if it isn't usable, colibri carries on the normal way.",
                        ),
                    );
                }
                Some(name) => {
                    ui.label(format!(
                        "{} {name}",
                        self.t("Cip de AI găsit:", "AI chip found:")
                    ));
                    muted(
                        ui,
                        &p,
                        self.t(
                            "colibri nu poate folosi încă acest cip; modelul rulează pe procesor sau pe placa video.",
                            "colibri can't use this chip yet; the model runs on the processor or the graphics card.",
                        ),
                    );
                }
                None => muted(
                    ui,
                    &p,
                    self.t(
                        "Nu a fost găsit un cip de AI dedicat (NPU). Dacă există unul, colibri îl poate folosi doar experimental, iar aplicația nu îl activează pe ascuns.",
                        "No dedicated AI chip (NPU) found. If there is one, colibri only supports it experimentally, and this app doesn't switch it on behind your back.",
                    ),
                ),
            }
            let active = self
                .status
                .as_ref()
                .and_then(|s| s.setup.as_ref())
                .filter(|s| s.ready)
                .and_then(|s| s.model.as_ref())
                .and_then(|m| m.id.clone());
            if let Some(id) = active {
                let busy = self.phase() == Phase::Installing || self.install_spawning;
                ui.add_enabled_ui(!busy && !self.demo, |ui| {
                    if secondary_button(
                        ui,
                        self.t(
                            "Aplică acum modelului curent",
                            "Apply now to the current model",
                        ),
                    )
                    .clicked()
                    {
                        self.switch_model(&id);
                    }
                });
            }
        });
        if self.s.backend != before {
            let _ = self.s.save();
        }
        ui.add_space(6.0);
    }

    /// The most sophisticated model this computer can run, with an explanation of how a small
    /// computer manages it. Returns its id so the caller does not list it a second time.
    /// A search box and quick filters for the model list.
    pub fn model_filter_bar(&mut self, ui: &mut Ui) {
        use companion_core::coli::ModelFilter as F;
        let hint = self.t("Caută un model…", "Search for a model…");
        ui.add(
            egui::TextEdit::singleline(&mut self.model_query)
                .hint_text(hint)
                .desired_width(280.0),
        );
        ui.horizontal_wrapped(|ui| {
            for f in F::ALL {
                let label = match f {
                    F::All => self.t("Toate", "All"),
                    F::Immediate => self.t("Imediat", "Ready now"),
                    F::Top => self.t("Recomandate", "Recommended"),
                    F::Rapid => self.t("Rapide", "Fast"),
                    F::Code => self.t("Cod", "Code"),
                    F::Creative => self.t("Creative", "Creative"),
                    F::Math => self.t("Matematică și logică", "Math and reasoning"),
                    F::Multilingual => self.t("Multilingve", "Multilingual"),
                };
                if ui.selectable_label(self.model_filter == f, label).clicked() {
                    self.model_filter = f;
                }
            }
        });
        if !matches!(self.model_filter, F::All | F::Immediate | F::Top | F::Rapid) {
            muted(
                ui,
                &self.p,
                self.t(
                    "Filtrul se bazează pe cuvintele din descrierea modelului; un model poate lipsi dacă descrierea nu menționează acest aspect.",
                    "This filter goes by the words in the model's description; a model can be missing if its description says nothing about it.",
                ),
            );
        }
    }

    fn models_filtered(&self) -> bool {
        !self.model_query.trim().is_empty()
            || self.model_filter != companion_core::coli::ModelFilter::All
    }

    pub fn model_shown(&self, m: &ModelInfo) -> bool {
        m.passes(self.model_filter) && m.matches_query(&self.model_query)
    }

    pub fn strongest_card(&mut self, ui: &mut Ui, in_window: bool) -> Option<String> {
        let p = self.p;
        let models = self.catalog.ready()?.clone();
        let best = coli::strongest_that_fits(&models)?.clone();
        if in_window && self.installed.iter().any(|i| i.id == best.id) {
            return Some(best.id);
        }
        let ram = self.hardware.ready().map(|h| h.ram_total_gb).unwrap_or(0.0);
        let small = ram > 0.0 && ram < 17.5;
        let (ro, en) = if best.streams_from_disk() && small {
            (
                "Calculatorul tău are puțină memorie, dar modelul acesta rulează oricum: partea mică și deasă rămâne în RAM, iar restul se citește din SSD pe măsură ce e nevoie. Răspunde mai încet decât un model mic, dar este cel mai capabil pe care îl poți folosi. Ai nevoie de un SSD rapid (NVMe) și de spațiu liber pe disc.",
                "Your computer has little memory, yet this model still runs: the small, dense part stays in RAM and the rest is read from the SSD as needed. It answers more slowly than a small model, but it is the most capable one you can use. You need a fast SSD (NVMe) and free disk space.",
            )
        } else if best.streams_from_disk() {
            (
                "Modelul acesta ține partea deasă în RAM și citește restul din SSD pe măsură ce e nevoie; un SSD rapid (NVMe) ajută mult.",
                "This model keeps its dense part in RAM and reads the rest from the SSD as needed; a fast NVMe drive helps a lot.",
            )
        } else {
            (
                "Se potrivește în memoria acestui calculator, deci va răspunde repede.",
                "It fits in this computer's memory, so it will answer quickly.",
            )
        };
        tinted(ui, p.accent_soft, p.accent, |ui| {
            ui.label(
                RichText::new(self.t(
                    "★ Cel mai capabil model pentru acest calculator",
                    "★ The most capable model for this computer",
                ))
                .semibold()
                .size(17.0)
                .color(p.accent),
            );
            ui.label(self.t(ro, en));
        });
        ui.add_space(4.0);
        if in_window {
            let busy = self.phase() == Phase::Installing || self.install_spawning;
            self.add_row(ui, &best, busy);
        } else {
            self.model_card(ui, &best);
        }
        ui.add_space(8.0);
        Some(best.id)
    }
}
