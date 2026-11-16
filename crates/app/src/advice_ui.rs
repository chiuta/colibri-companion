//! "Your computer, analysed": the graphics cards found, what the engine uses, and plain advice
//! drawn from the live measurements. Nothing here changes a setting by itself.

use crate::app::{CompanionApp, ResView};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::advisor::{self, Tip, Verdict};
use companion_core::gpu::GpuInfo;
use egui::{RichText, Ui};

impl CompanionApp {
    fn latest_ram_total(&self) -> f64 {
        self.res.as_ref().map_or(0.0, |v| v.snap.ram_total_gb)
    }

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
            Some(v) if !g.integrated => crate::widgets::loc(format!(" · {v:.0} GB VRAM")),
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
            .map(|m| {
                format!(
                    "{} ({})",
                    m.name,
                    crate::widgets::loc(format!("{:.0} GB", m.download_gb))
                )
            })
    }

    pub fn advice_section(&mut self, ui: &mut Ui, view: &ResView) {
        let p = self.p;
        let s = &view.snap;
        ui.label(
            RichText::new(self.t("Calculatorul tău, analizat", "Your computer, analysed"))
                .size(17.0)
                .semibold(),
        );
        // ---- graphics cards
        match self.gpus.clone() {
            None => {
                ui.horizontal(|ui| {
                    crate::widgets::spinner(ui);
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
                    ui.label(format!("{} · {}", self.t(ro, en), crate::widgets::gb1(m)));
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
                            "Windows mută memorie pe disc ({} GB în swap). Aceasta încetinește totul: închide programe sau alege un model mai mic.",
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
                        .button(self.t("Deschide lista de modele", "Open the model list"))
                        .clicked()
                    {
                        open_models = true;
                    }
                }
                Tip::CloseApps(list) => {
                    let names = list
                        .iter()
                        .map(|(n, g)| format!("{n} ({})", crate::widgets::gb1(*g)))
                        .collect::<Vec<_>>()
                        .join(", ");
                    ui.label(
                        self.t(
                            "Alte programe rețin multă memorie: {}. Închide-le înainte de a genera.",
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
                            "Mai ai doar {} GB liberi pe disc: lasă loc pentru modele și pentru swap.",
                            "Only {} GB of disk space is free: leave room for models and swap.",
                        )
                        .replacen("{}", &format!("{free_gb:.0}"), 1),
                    );
                }
            }
        }
        self.speed_section(ui);
        self.capacity_section(ui);
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
                    crate::widgets::spinner(ui);
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
                crate::widgets::spinner(ui);
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
        self.context_table(ui);
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
        self.speed_tools(ui);
    }

    /// What a shorter conversation memory would buy: colibri's planner at several context lengths.
    fn context_table(&mut self, ui: &mut Ui) {
        let p = self.p;
        ui.add_space(6.0);
        ui.label(
            RichText::new(self.t(
                "Context mai scurt, model mai rapid",
                "A shorter context, a faster model",
            ))
            .semibold(),
        );
        muted(
            ui,
            &p,
            self.t(
                "Fiecare jeton de context ocupă memorie care altfel ar ține experții modelului. Un context mai scurt lasă mai mult loc experților și reduce citirile de pe disc, dar conversațiile foarte lungi pierd începutul mai devreme.",
                "Every token of context takes memory that would otherwise hold the model's experts. A shorter context leaves more room for experts and cuts disk reads, but very long conversations lose their beginning sooner.",
            ),
        );
        if self.ctx_rows.is_none() {
            let reads_disk = matches!(&self.plan, Some(Ok(pl)) if pl.reads_disk());
            if reads_disk && !self.ctx_busy && !self.ctx_auto_done {
                self.ctx_auto_done = true;
                self.compare_contexts();
            }
            ui.horizontal(|ui| {
                let label = self.t("Calculează pentru 4k–64k", "Work it out for 4k–64k");
                if ui
                    .add_enabled(!self.ctx_busy, egui::Button::new(label))
                    .clicked()
                {
                    self.compare_contexts();
                }
                if self.ctx_busy {
                    crate::widgets::spinner(ui);
                }
            });
            return;
        }
        let rows = self.ctx_rows.clone().unwrap_or_default();
        let best = recommended_context(&rows);
        let mut apply = None;
        egui::Grid::new("ctx_plan_rows")
            .num_columns(4)
            .show(ui, |ui| {
                for (n, plan) in &rows {
                    ui.label(format!("{n}"));
                    match plan {
                        Ok(pl) => {
                            ui.label(format!(
                                "{} {}",
                                self.t("cache de experți", "expert cache"),
                                crate::widgets::gb1(pl.cache_gb)
                            ));
                            ui.label(format!(
                                "{:.0}% {}",
                                pl.hit_rate * 100.0,
                                self.t("în memorie", "in memory")
                            ));
                            ui.horizontal(|ui| {
                                if ui
                                    .add_enabled(
                                        !self.ctx_busy,
                                        egui::Button::new(self.t("Utilizează", "Use")),
                                    )
                                    .clicked()
                                {
                                    apply = Some(*n);
                                }
                                if best == Some(*n) {
                                    ui.label(
                                        RichText::new(self.t("Recomandat", "Recommended"))
                                            .semibold(),
                                    );
                                }
                            });
                        }
                        Err(_) => {
                            ui.label("—");
                            ui.label("");
                            ui.label("");
                        }
                    }
                    ui.end_row();
                }
            });
        if let Some(n) = apply {
            self.apply_context(n);
        }
    }
    /// The drive's speed, and what 16, 32 and 64 GB of memory would let each catalogue model do.
    fn capacity_section(&mut self, ui: &mut Ui) {
        use companion_core::advisor::{TierFit, tier_fit};
        use companion_core::diskbench::{DiskClass, classify};
        let p = self.p;
        ui.add_space(10.0);
        ui.label(
            RichText::new(self.t("Discul și memoria", "The drive and the memory"))
                .size(15.0)
                .semibold(),
        );
        muted(
            ui,
            &p,
            self.t(
                "Modelele uriașe își citesc experții de pe disc, așa că viteza lor depinde mai ales de disc. Un test scurt de citire aleatorie (aproximativ 4 secunde) arată ce poți aștepta.",
                "Huge models read their experts from the drive, so their speed depends mostly on the drive. A short random-read test (about 4 seconds) shows what to expect.",
            ),
        );
        ui.horizontal(|ui| {
            let label = self.t("Măsoară discul modelelor", "Test the models' drive");
            if ui
                .add_enabled(!self.disk_busy, egui::Button::new(label))
                .clicked()
            {
                self.measure_disk();
            }
            if self.disk_busy {
                crate::widgets::spinner(ui);
            }
        });
        match self.disk_speed.clone() {
            Some(Ok(mb)) => {
                let (ro, en) = match classify(mb) {
                    DiskClass::Fast => (
                        "Disc rapid: modelele care își citesc experții de pe disc pot rula acceptabil.",
                        "A fast drive: models that read their experts from it can run acceptably.",
                    ),
                    DiskClass::Fair => (
                        "Disc mediu: modelele uriașe vor fi lente; preferă modele care încap în memorie.",
                        "A middling drive: huge models will be slow; prefer models that fit in memory.",
                    ),
                    DiskClass::Slow => (
                        "Disc lent: modelele care citesc de pe disc vor fi foarte lente. Alege doar modele care încap în memorie, sau folosește un SSD NVMe.",
                        "A slow drive: models that read from it will be very slow. Pick only models that fit in memory, or use an NVMe SSD.",
                    ),
                };
                ui.label(format!(
                    "{} · {}",
                    crate::widgets::loc(format!("{mb:.0} MB/s")),
                    self.t(ro, en)
                ));
            }
            Some(Err(_)) => muted(
                ui,
                &p,
                self.t(
                    "Testul nu a găsit un fișier mare de citit în folderul modelelor.",
                    "The test found no large file to read in the models folder.",
                ),
            ),
            None => {}
        }
        let Some(models) = self.catalog.ready() else {
            return;
        };
        let mut rows: Vec<_> = models
            .iter()
            .filter(|m| m.is_chat_model() && m.status != "unsupported")
            .collect();
        rows.sort_by(|a, b| a.download_gb.total_cmp(&b.download_gb));
        if let Some((m, why)) = advisor::recommend(
            models,
            self.latest_ram_total(),
            self.disk_speed
                .as_ref()
                .and_then(|r| r.as_ref().ok())
                .map(|mb| classify(*mb)),
        ) {
            let (ro, en) = match why {
                advisor::Why::FromMemory => (
                    "Cel mai capabil model care rulează integral din memorie pe acest calculator: {} ({} GB).",
                    "The most capable model that runs entirely from memory on this computer: {} ({} GB).",
                ),
                advisor::Why::FastDrive => (
                    "Discul este rapid și memoria ajunge: cel mai capabil model potrivit este {} ({} GB).",
                    "The drive is fast and the memory is enough: the most capable fitting model is {} ({} GB).",
                ),
            };
            ui.add_space(4.0);
            tinted(ui, p.accent_soft, p.accent, |ui| {
                ui.label(
                    RichText::new(self.t(ro, en).replacen("{}", &m.name, 1).replacen(
                        "{}",
                        &format!("{:.0}", m.download_gb),
                        1,
                    ))
                    .semibold(),
                );
            });
        }
        ui.add_space(6.0);
        ui.label(
            RichText::new(self.t(
                "Ce permite 16, 32 sau 64 GB de memorie",
                "What 16, 32 or 64 GB of memory allows",
            ))
            .semibold(),
        );
        egui::Grid::new("tier_rows").num_columns(5).show(ui, |ui| {
            ui.label("");
            ui.label(RichText::new("GB").weak());
            for t in ["16", "32", "64"] {
                ui.label(RichText::new(t).semibold());
            }
            ui.end_row();
            for m in rows {
                ui.label(&m.name);
                ui.label(RichText::new(format!("{:.0}", m.download_gb)).weak());
                for ram in [16.0, 32.0, 64.0] {
                    let (mark, tip) = match tier_fit(ram, m.ram_min_gb, m.ram_good_gb) {
                        TierFit::Good => ("●", self.t("Merge bine", "Runs well")),
                        TierFit::Tight => (
                            "◐",
                            self.t(
                                "Merge, cu mai multe citiri de pe disc",
                                "Runs, with more reading from the drive",
                            ),
                        ),
                        TierFit::No => ("○", self.t("Nu încape", "Does not fit")),
                    };
                    ui.label(mark).on_hover_text(tip);
                }
                ui.end_row();
            }
        });
        muted(
            ui,
            &p,
            self.t(
                "● merge bine · ◐ merge, dar citește mult de pe disc · ○ nu încape. Cifrele vin din catalogul colibri; viteza reală depinde de disc.",
                "● runs well · ◐ runs, but reads a lot from the drive · ○ does not fit. The figures come from colibri's catalogue; the real speed depends on the drive.",
            ),
        );
    }
}

/// The longest context that still keeps nearly all of the best in-memory share (within three
/// points), never below 8 192 tokens: the most conversation memory for almost the full speed.
pub fn recommended_context(
    rows: &[(u32, Result<companion_core::coli::PlanSummary, String>)],
) -> Option<u32> {
    let ok: Vec<(u32, f64)> = rows
        .iter()
        .filter_map(|(n, p)| p.as_ref().ok().map(|p| (*n, p.hit_rate)))
        .collect();
    let top = ok.iter().map(|(_, h)| *h).fold(f64::MIN, f64::max);
    if ok.is_empty() || top < 0.0 {
        return None;
    }
    ok.iter()
        .filter(|(n, h)| *n >= 8192 && *h >= top - 0.03)
        .map(|(n, _)| *n)
        .max()
        .or_else(|| {
            ok.iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(n, _)| *n)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use companion_core::coli::PlanSummary;

    fn row(n: u32, hit: f64) -> (u32, Result<PlanSummary, String>) {
        (
            n,
            Ok(PlanSummary {
                hit_rate: hit,
                ..Default::default()
            }),
        )
    }

    #[test]
    fn picks_the_longest_context_that_keeps_the_speed() {
        let rows = [
            row(4096, 0.80),
            row(8192, 0.79),
            row(16384, 0.70),
            row(32768, 0.50),
        ];
        assert_eq!(recommended_context(&rows), Some(8192));
        let flat = [row(8192, 1.0), row(16384, 1.0), row(32768, 1.0)];
        assert_eq!(recommended_context(&flat), Some(32768));
        assert_eq!(recommended_context(&[]), None);
    }
}
