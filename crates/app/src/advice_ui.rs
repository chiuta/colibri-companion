//! "Your computer, analysed": the graphics cards found, what the engine uses, and plain advice
//! drawn from the live measurements. Nothing here changes a setting by itself.

use crate::app::{CompanionApp, ResView};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::advisor::{self, Tip, Verdict};
use companion_core::gpu::GpuInfo;
use egui::{RichText, Ui};

impl CompanionApp {
    /// Size in GB of the model that is loaded or was last set up, when it is known.
    fn active_model_gb(&self) -> Option<f64> {
        let st = self.status.as_ref()?;
        let id = st
            .server
            .as_ref()
            .and_then(|s| s.model_id.clone())
            .or_else(|| st.setup.as_ref()?.model.as_ref()?.id.clone())
            .or_else(|| self.model_id.clone())?;
        let m = self.installed.iter().find(|m| m.id == id && m.complete)?;
        Some(m.total_bytes as f64 / 1_073_741_824.0)
    }

    fn gpu_line(&self, g: &GpuInfo) -> String {
        let mem = match g.vram_gb {
            Some(v) if !g.integrated => format!(" · {v:.0} GB VRAM"),
            _ if g.integrated => self
                .t(" · folosește memoria RAM", " · shares main memory")
                .to_string(),
            _ => String::new(),
        };
        format!("{}{}", g.name, mem)
    }

    /// The biggest chat model of the catalog that fits in memory without reading from the disk.
    fn faster_alternative(&self, ram_total_gb: f64) -> Option<String> {
        let models = self.catalog.ready()?;
        let usable = advisor::usable_ram_gb(ram_total_gb);
        models
            .iter()
            .filter(|m| m.is_chat_model() && m.status != "unsupported" && !m.streams_from_disk())
            .filter(|m| m.download_gb <= usable)
            .max_by(|a, b| a.download_gb.total_cmp(&b.download_gb))
            .map(|m| format!("{} ({:.0} GB)", m.name, m.download_gb))
    }

    pub fn advice_section(&mut self, ui: &mut Ui, view: &ResView) {
        let p = self.p;
        let s = &view.snap;
        ui.label(
            RichText::new(self.t(
                "Calculatorul dumneavoastră, analizat",
                "Your computer, analysed",
            ))
            .size(17.0)
            .semibold(),
        );
        // ---- graphics cards
        match self.gpus.clone() {
            None => {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new());
                    ui.label(self.t("Caut plăcile video…", "Looking for graphics cards…"));
                });
            }
            Some(list) if list.is_empty() => muted(
                ui,
                &p,
                self.t(
                    "Nu am găsit nicio placă video. Modelul rulează pe procesor.",
                    "No graphics card was found. The model runs on the processor.",
                ),
            ),
            Some(list) => {
                for g in &list {
                    ui.label(format!("• {}", self.gpu_line(g)));
                }
            }
        }
        if let Some(h) = self.hardware.ready() {
            let engine = match h.auto_engine() {
                companion_core::coli::Engine::Cuda(n) => format!("CUDA · {n}"),
                companion_core::coli::Engine::Vulkan(n) => format!("Vulkan · {n}"),
                companion_core::coli::Engine::Metal => "Metal".to_string(),
                companion_core::coli::Engine::Cpu => self.t("procesor", "processor").to_string(),
            };
            muted(
                ui,
                &p,
                &format!(
                    "{}: {}",
                    self.t("Motorul alege automat", "The engine picks automatically"),
                    engine
                ),
            );
        }
        // ---- the loaded model against memory
        let gpus = self.gpus.clone().unwrap_or_default();
        let model_gb = self.active_model_gb();
        if let Some(m) = model_gb {
            if s.ram_total_gb > 0.0 {
                let v = advisor::assess(m, s.ram_total_gb, advisor::best_vram_gb(&gpus));
                let (ro, en, ok) = match v {
                    Verdict::FitsGpu => (
                        "Modelul încape în memoria plăcii video: va răspunde cel mai repede.",
                        "The model fits in the graphics card's memory: the fastest way to run it.",
                        true,
                    ),
                    Verdict::FitsRam => (
                        "Modelul încape în RAM: va răspunde repede.",
                        "The model fits in memory: it will answer quickly.",
                        true,
                    ),
                    Verdict::StreamsFromDisk => (
                        "Modelul e mai mare decât RAM-ul disponibil: o parte se citește de pe disc la fiecare răspuns, deci va fi lent.",
                        "The model is larger than the memory available: part of it is read from the disk for every answer, so it will be slow.",
                        false,
                    ),
                };
                let (fill, stroke) = if ok {
                    (p.accent_soft, p.accent)
                } else {
                    (p.warn_soft, p.warn)
                };
                tinted(ui, fill, stroke, |ui| {
                    ui.label(format!("{} · {:.1} GB", self.t(ro, en), m));
                });
            }
        }
        // ---- advice
        let backend = self
            .status
            .as_ref()
            .and_then(|st| st.server.as_ref()?.backend.clone())
            .or_else(|| self.status.as_ref()?.setup.as_ref()?.backend.clone());
        let tips = advisor::tips(&advisor::TipInput {
            model_gb,
            ram_total_gb: s.ram_total_gb,
            ram_used_gb: s.ram_used_gb,
            swap_used_gb: s.swap_used_gb,
            gpus: &gpus,
            backend_in_use: backend.as_deref(),
            others: &s.others,
            disk_free_gb: s.disk_free_gb,
        });
        let mut open_models = false;
        for tip in tips {
            ui.add_space(4.0);
            match tip {
                Tip::SwapHeavy { swap_gb } => {
                    ui.label(
                        self.t(
                            "Windows mută memorie pe disc ({} GB în swap). Aceasta încetinește totul: închideți programe sau alegeți un model mai mic.",
                            "The system is moving memory to the disk ({} GB of swap). That slows everything down: close programs or pick a smaller model.",
                        )
                        .replacen("{}", &format!("{swap_gb:.1}"), 1),
                    );
                }
                Tip::ModelBiggerThanRam {
                    model_gb,
                    usable_gb,
                } => {
                    ui.label(
                        self.t(
                            "Modelul ({} GB) depășește memoria utilizabilă ({} GB). Un model care încape în RAM răspunde de multe ori mai repede.",
                            "The model ({} GB) is larger than the usable memory ({} GB). A model that fits in memory answers many times faster.",
                        )
                        .replacen("{}", &format!("{model_gb:.0}"), 1)
                        .replacen("{}", &format!("{usable_gb:.0}"), 1),
                    );
                    if let Some(alt) = self.faster_alternative(s.ram_total_gb) {
                        ui.label(
                            RichText::new(
                                self.t("Alternativă rapidă: {}", "A faster alternative: {}")
                                    .replacen("{}", &alt, 1),
                            )
                            .semibold(),
                        );
                    }
                    if ui
                        .button(self.t("Deschideți lista de modele", "Open the model list"))
                        .clicked()
                    {
                        open_models = true;
                    }
                }
                Tip::CloseApps(list) => {
                    let names = list
                        .iter()
                        .map(|(n, g)| format!("{n} ({g:.1} GB)"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    ui.label(
                        self.t(
                            "Alte programe rețin multă memorie: {}. Închideți-le înainte de a genera.",
                            "Other programs hold a lot of memory: {}. Close them before generating.",
                        )
                        .replacen("{}", &names, 1),
                    );
                }
                Tip::GpuNotUsed { name } => {
                    ui.label(
                        self.t(
                            "Placa {} nu este folosită: motorul rulează pe procesor. Actualizează driverul plăcii (cu suport Vulkan) și repornește modelul.",
                            "The card {} is not used: the engine runs on the processor. Update the graphics driver (with Vulkan support) and restart the model.",
                        )
                        .replacen("{}", &name, 1),
                    );
                }
                Tip::IntegratedOnly { name } => {
                    muted(
                        ui,
                        &p,
                        &self
                            .t(
                                "Doar {} (folosește memoria RAM): nu aduce un câștig separat pentru modele mari.",
                                "Only {} (it shares main memory): it adds no separate speed-up for large models.",
                            )
                            .replacen("{}", &name, 1),
                    );
                }
                Tip::DiskNearlyFull { free_gb } => {
                    ui.label(
                        self.t(
                            "Mai aveți doar {} GB liberi pe disc: lăsați loc pentru modele și pentru swap.",
                            "Only {} GB of disk space is free: leave room for models and swap.",
                        )
                        .replacen("{}", &format!("{free_gb:.0}"), 1),
                    );
                }
            }
        }
        self.speed_section(ui);
        if open_models {
            self.set_resources_open(false);
            self.show_models = true;
        }
    }

    /// colibri's own plan for the model that is set up, and the way to have it measure itself.
    fn speed_section(&mut self, ui: &mut Ui) {
        let p = self.p;
        if self.active_model_dir().is_none() {
            return;
        }
        self.fetch_plan_once();
        ui.add_space(8.0);
        ui.label(
            RichText::new(self.t("Viteza pe acest calculator", "Speed on this computer"))
                .size(15.0)
                .semibold(),
        );
        match self.plan.clone() {
            None => {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new());
                    ui.label(self.t(
                        "colibri calculează planul…",
                        "colibri is working out its plan…",
                    ));
                });
            }
            Some(Err(_)) => muted(
                ui,
                &p,
                self.t(
                    "Planul colibri nu a putut fi citit pentru acest model.",
                    "colibri's plan could not be read for this model.",
                ),
            ),
            Some(Ok(plan)) => {
                if plan.reads_disk() {
                    ui.label(
                        self.t(
                            "{} GB din experții modelului rămân pe disc; {} GB de memorie sunt rezervați pentru cache-ul de experți. În medie, {}% dintre experții necesari se găsesc deja în memorie.",
                            "{} GB of the model's experts stay on the disk; {} GB of memory is reserved for the expert cache. On average {}% of the experts needed are already in memory.",
                        )
                        .replacen("{}", &format!("{:.0}", plan.cold_gb), 1)
                        .replacen("{}", &format!("{:.1}", plan.cache_gb), 1)
                        .replacen("{}", &format!("{:.0}", plan.hit_rate * 100.0), 1),
                    );
                } else {
                    muted(
                        ui,
                        &p,
                        self.t(
                            "Modelul încape în memorie: nu se citește nimic de pe disc în timpul răspunsului.",
                            "The model fits in memory: nothing is read from the disk while it answers.",
                        ),
                    );
                }
                for w in plan.warnings.iter().take(3) {
                    muted(ui, &p, w);
                }
            }
        }
        ui.add_space(4.0);
        let busy = self.tuning;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !busy,
                    egui::Button::new(self.t(
                        "Măsoară și optimizează (câteva minute)",
                        "Measure and optimise (a few minutes)",
                    )),
                )
                .clicked()
            {
                self.run_tune();
            }
            if busy {
                ui.add(egui::Spinner::new());
            }
        });
        if busy {
            muted(
                ui,
                &p,
                self.t(
                    "colibri încearcă pe rând setările sigure (citire directă, suprapunerea citirilor, fire, dimensiunea cache-ului) și le păstrează doar pe cele mai rapide, fără să schimbe răspunsurile. Modelul este oprit cât durează și repornit la sfârșit.",
                    "colibri tries its safe settings one by one (direct reads, overlapped reads, threads, cache size) and keeps only the fastest, without changing any answer. The model is stopped meanwhile and restarted at the end.",
                ),
            );
        }
        if let Some(note) = self.tune_note.clone() {
            ui.label(note);
        }
        let label = self.t(
            "Adaptează cache-ul de experți la conversația curentă (politica «balanced» din colibri; nu schimbă răspunsurile)",
            "Adapt the expert cache to the current conversation (colibri's “balanced” policy; it does not change answers)",
        );
        if ui.checkbox(&mut self.s.adaptive_cache, label).changed() {
            self.plan = None;
            self.plan_asked = None;
        }
        muted(
            ui,
            &p,
            self.t(
                "Se aplică la următoarea pornire a modelului.",
                "Applies the next time the model starts.",
            ),
        );
    }
}
