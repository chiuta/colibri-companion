//! The guided path from "nothing installed" to "ready to chat": one calm screen
//! per step, plain words, one obvious next action.

use crate::app::{CompanionApp, Loadable, Phase};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::coli::{self, ModelInfo};
use companion_core::locate;
use companion_core::proc;
use egui::{RichText, Spinner, Ui};
use std::time::Duration;

impl CompanionApp {
    pub fn show_phase(&mut self, ui: &mut Ui, phase: Phase) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_space(18.0);
                let p = self.p;
                column(ui, 820.0, |ui| match phase {
                    Phase::Detecting => self.screen_detecting(ui),
                    Phase::NeedPython => self.screen_need_python(ui),
                    Phase::NeedColibri => self.screen_need_colibri(ui),
                    Phase::ChooseModel => self.screen_choose_model(ui),
                    Phase::Installing => self.screen_installing(ui),
                    Phase::InstallProblem => self.screen_install_problem(ui),
                    Phase::StartServer => self.screen_start_server(ui),
                    Phase::Loading => self.screen_loading(ui),
                    Phase::Chat => {}
                });
                let _ = p;
                ui.add_space(24.0);
            });
    }

    fn step_header(&self, ui: &mut Ui, step: u8, title: &str, subtitle: &str) {
        let p = self.p;
        ui.horizontal(|ui| {
            pill(
                ui,
                &format!("{} {step}/4", self.t("Pasul", "Step")),
                p.on_accent,
                p.accent,
            );
        });
        ui.add_space(4.0);
        heading(ui, title);
        if !subtitle.is_empty() {
            ui.label(RichText::new(subtitle).size(16.0).color(p.muted));
        }
        ui.add_space(10.0);
    }

    fn screen_detecting(&mut self, ui: &mut Ui) {
        ui.add_space(80.0);
        ui.vertical_centered(|ui| {
            ui.add(Spinner::new().size(36.0));
            ui.add_space(12.0);
            ui.label(
                RichText::new(self.t(
                    "Mă uit ce am la dispoziție pe acest calculator…",
                    "Looking at what this computer has…",
                ))
                .size(18.0),
            );
            if self.s.colibri_root.is_none() {
                ui.add_space(6.0);
                if secondary_button(
                    ui,
                    self.t(
                        "Vreau să mă conectez la un server deja pornit",
                        "I want to connect to a server that's already running",
                    ),
                )
                .clicked()
                {
                    self.show_settings = true;
                }
            }
        });
    }

    fn screen_need_python(&mut self, ui: &mut Ui) {
        let p = self.p;
        self.step_header(
            ui,
            1,
            self.t("Mai întâi, ai nevoie de Python", "First, you need Python"),
            self.t(
                "colibri își pornește treburile cu Python 3.10 sau mai nou. Nu l-am găsit pe acest calculator.",
                "colibri uses Python 3.10 or newer to set itself up. I couldn't find it on this computer.",
            ),
        );
        card(ui, &p, |ui| {
            ui.label(self.t(
                "1. Descarcă și instalează Python (gratuit).",
                "1. Download and install Python (free).",
            ));
            ui.label(self.t(
                "2. Pe Windows, bifează «Add python.exe to PATH» în instalator.",
                "2. On Windows, tick “Add python.exe to PATH” in the installer.",
            ));
            ui.label(self.t(
                "3. Revino aici și apasă «Caută din nou».",
                "3. Come back here and press “Check again”.",
            ));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if primary_button(
                    ui,
                    &p,
                    self.t("Deschide pagina Python", "Open the Python page"),
                )
                .clicked()
                {
                    proc::open_externally(locate::PYTHON_DOWNLOAD_URL);
                }
                if secondary_button(ui, self.t("Caută din nou", "Check again")).clicked() {
                    self.detect_environment();
                }
            });
        });
        ui.add_space(8.0);
        muted(
            ui,
            &p,
            self.t(
                "Python se instalează o singură dată. Nu trebuie să scrii nicio comandă.",
                "Python is installed once. You never have to type a command.",
            ),
        );
    }

    fn screen_need_colibri(&mut self, ui: &mut Ui) {
        let p = self.p;
        self.step_header(
            ui,
            1,
            self.t("Unde se află colibri?", "Where is colibri?"),
            self.t(
                "colibri este motorul care rulează modelele pe calculatorul tău. Îl pot descărca eu sau îl poți arăta tu.",
                "colibri is the engine that runs models on your computer. I can download it for you, or you can point me to it.",
            ),
        );

        card(ui, &p, |ui| {
            ui.label(
                RichText::new(self.t("Varianta ușoară", "The easy way"))
                    .semibold()
                    .size(17.0),
            );
            ui.label(self.t(
                "Descarc colibri de pe GitHub în folderul tău personal (~/colibri), fără să ai nevoie de git. Ocupă puțin spațiu.",
                "I download colibri from GitHub into your home folder (~/colibri), no git needed. It's small.",
            ));
            if self.cloning {
                ui.label(
                    RichText::new(self.t(
                        "Nu am găsit colibri pe acest calculator, așa că îl descarc acum.",
                        "I couldn't find colibri on this computer, so I'm downloading it now.",
                    ))
                    .color(p.accent)
                    .semibold(),
                );
            }
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!self.cloning, |ui| {
                    if primary_button(
                        ui,
                        &p,
                        self.t("Descarcă colibri pentru mine", "Download colibri for me"),
                    )
                    .clicked()
                    {
                        self.clone_colibri();
                    }
                });
                if self.cloning && self.fetch_progress.is_none() && !self.fetch_unpacking {
                    ui.add(Spinner::new());
                }
            });
            if self.cloning {
                if self.fetch_unpacking {
                    progress_bar(ui, &p, None, self.t("Dezarhivez…", "Unpacking…"));
                } else if let Some((done, total)) = self.fetch_progress {
                    let frac = (total > 0).then(|| (done as f64 / total as f64) as f32);
                    let label = format!("{:.1} MB", done as f64 / 1e6);
                    progress_bar(ui, &p, frac, &label);
                }
            }
            if let Some(last) = self.clone_lines.last() {
                ui.label(RichText::new(last).small().color(p.muted).monospace());
            }
            if let Some(msg) = &self.root_msg {
                ui.label(RichText::new(msg).color(p.bad));
            }
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(self.t("Nu merge?", "Not working?")).color(p.muted));
                if ui
                    .link(self.t(
                        "Deschide descărcarea ZIP în browser",
                        "Open the ZIP download in your browser",
                    ))
                    .clicked()
                {
                    proc::open_externally(locate::COLIBRI_ZIP_URL);
                }
            });
        });
        ui.add_space(8.0);
        card(ui, &p, |ui| {
            ui.label(
                RichText::new(self.t("L-am descărcat deja", "I already have it"))
                    .semibold()
                    .size(17.0),
            );
            ui.label(self.t(
                "Trage folderul colibri în această fereastră sau lipește calea lui.",
                "Drag the colibri folder into this window, or paste its path.",
            ));
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 190.0).max(160.0);
                ui.add(
                    egui::TextEdit::singleline(&mut self.root_input)
                        .desired_width(w)
                        .hint_text("/home/…/colibri"),
                );
                if secondary_button(ui, self.t("Folosește acest folder", "Use this folder"))
                    .clicked()
                {
                    let text = self.root_input.clone();
                    self.use_root_from_text(&text);
                }
            });
            if let Some(msg) = &self.root_msg {
                ui.label(RichText::new(msg).color(p.bad));
            }
            if secondary_button(
                ui,
                self.t("Caută din nou automat", "Search again automatically"),
            )
            .clicked()
            {
                self.detect_environment();
            }
        });
        ui.add_space(8.0);
        muted(
            ui,
            &p,
            self.t(
                "colibri este un proiect separat (Apache-2.0), creat de JustVugg. Această aplicație nu îi modifică codul: doar îl pornește și vorbește cu el.",
                "colibri is a separate project (Apache-2.0) by JustVugg. This app never changes its code: it only runs it and talks to it.",
            ),
        );
    }

    // ------------------------------------------------------------ choose a model

    fn screen_choose_model(&mut self, ui: &mut Ui) {
        let p = self.p;
        self.step_header(
            ui,
            2,
            self.t("Alege un model", "Choose a model"),
            self.t(
                "Un model este «creierul» care răspunde. Ți-am marcat ce se potrivește acestui calculator.",
                "A model is the “brain” that answers. I've marked what suits this computer.",
            ),
        );

        // What this computer has.
        if let Some(hw) = self.hardware.ready() {
            let gpu = if hw.gpus.is_empty() {
                self.t("fără placă video dedicată", "no dedicated graphics card")
                    .to_string()
            } else {
                hw.gpus.join(", ")
            };
            tinted(ui, p.accent_soft, p.border, |ui| {
                ui.label(RichText::new(self.t("Calculatorul tău", "Your computer")).semibold());
                ui.label(format!(
                    "{}: {} · RAM: {} ({} {}) · {}: {} {} · {}",
                    self.t("Procesor", "CPU"),
                    if hw.cpu_name.is_empty() {
                        "—"
                    } else {
                        &hw.cpu_name
                    },
                    fmt_gb(hw.ram_total_gb),
                    fmt_gb(hw.ram_free_gb),
                    self.t("liberi", "free"),
                    self.t("Disc", "Disk"),
                    fmt_gb(hw.disk_free_gb),
                    self.t("liberi", "free"),
                    gpu
                ));
                for w in &hw.warnings {
                    ui.label(RichText::new(crate::i18n::engine_text(self.s.lang, w)).color(p.warn));
                }
            });
            ui.add_space(6.0);
        } else if matches!(self.hardware, Loadable::Loading) {
            ui.horizontal(|ui| {
                ui.add(Spinner::new());
                ui.label(self.t("Măsor calculatorul…", "Measuring your computer…"));
            });
        }

        self.performance_panel(ui);
        let top = self.strongest_card(ui, false);
        match self.catalog.clone() {
            Loadable::Idle | Loadable::Loading => {
                ui.horizontal(|ui| {
                    ui.add(Spinner::new());
                    ui.label(self.t("Citesc lista de modele…", "Reading the model list…"));
                });
            }
            Loadable::Failed(e) => {
                tinted(ui, p.bad_soft, p.bad, |ui| {
                    ui.label(
                        RichText::new(self.t(
                            "Nu am putut citi lista de modele.",
                            "I couldn't read the model list.",
                        ))
                        .semibold()
                        .color(p.bad),
                    );
                    ui.label(&e);
                });
                if secondary_button(ui, self.t("Încearcă din nou", "Try again")).clicked() {
                    self.refresh_engine_info();
                }
            }
            Loadable::Ready(models) => {
                let (fit, nofit): (Vec<_>, Vec<_>) = models
                    .iter()
                    .filter(|m| top.as_deref() != Some(m.id.as_str()))
                    .partition(|m| m.fits && m.is_chat_model());
                if fit.is_empty() && top.is_none() {
                    tinted(ui, p.warn_soft, p.warn, |ui| {
                        ui.label(
                            RichText::new(self.t(
                                "Niciun model nu se potrivește acum.",
                                "No model fits right now.",
                            ))
                            .semibold()
                            .color(p.warn),
                        );
                        ui.label(self.t(
                            "Eliberează spațiu pe disc sau închide programe care folosesc multă memorie, apoi măsoară din nou. Mai jos poți vedea de ce nu se potrivesc.",
                            "Free some disk space or close memory-hungry programs, then measure again. Below you can see why each one doesn't fit.",
                        ));
                    });
                    ui.add_space(6.0);
                }
                for m in &fit {
                    self.model_card(ui, m);
                    ui.add_space(8.0);
                }
                if !nofit.is_empty() {
                    ui.add_space(4.0);
                    let label = format!(
                        "{} ({})",
                        self.t(
                            "Modele care nu se potrivesc acum",
                            "Models that don't fit right now"
                        ),
                        nofit.len()
                    );
                    ui.checkbox(&mut self.show_unfit, label);
                    if self.show_unfit {
                        for m in &nofit {
                            self.model_card(ui, m);
                            ui.add_space(8.0);
                        }
                    }
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if secondary_button(ui, self.t("Măsoară din nou", "Measure again")).clicked()
                    {
                        self.refresh_engine_info();
                    }
                    if self.status.as_ref().is_some_and(|s| s.setup_ready())
                        && secondary_button(
                            ui,
                            self.t("Înapoi la modelul instalat", "Back to the installed model"),
                        )
                        .clicked()
                    {
                        self.show_model_picker = false;
                    }
                });
            }
        }
    }

    pub(crate) fn model_card(&mut self, ui: &mut Ui, m: &ModelInfo) {
        let p = self.p;
        let selected = self.selected_model.as_deref() == Some(m.id.as_str());
        let can_install = m.fits && m.is_chat_model();
        egui::Frame::new()
            .fill(p.card)
            .stroke(egui::Stroke::new(if selected { 2.5 } else { 1.0 }, if selected { p.accent } else { p.border }))
            .corner_radius(egui::CornerRadius::same(12))
            .inner_margin(egui::Margin::same(14))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(&m.name).size(19.0).semibold());
                    if m.recommended {
                        pill(ui, self.t("Recomandat pentru tine", "Recommended for you"), p.on_accent, p.accent);
                    }
                    let (txt, fg, bg) = self.status_badge(m);
                    pill(ui, txt, fg, bg);
                });
                ui.label(RichText::new(self.blurb(m)).color(p.muted));
                ui.add_space(2.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("{} {}", self.t("Descărcare:", "Download:"), fmt_gb(m.download_gb)));
                    ui.label("·");
                    ui.label(format!(
                        "RAM: {} {} / {} {}",
                        self.t("minim", "min"),
                        fmt_gb(m.ram_min_gb),
                        self.t("recomandat", "recommended"),
                        fmt_gb(m.ram_good_gb)
                    ));
                });
                ui.label(RichText::new(self.speed_note(m)).size(14.0));
                if !m.fits && !m.reason.is_empty() {
                    ui.label(RichText::new(crate::i18n::engine_text(self.s.lang, &m.reason)).size(13.0).color(p.bad));
                }
                if !m.license_note.is_empty() {
                    ui.label(RichText::new(format!("{} {}", self.t("Licență:", "Licence:"), crate::i18n::engine_text(self.s.lang, &m.license_note))).color(p.warn).semibold());
                }
                if !m.is_chat_model() {
                    ui.label(RichText::new(self.t("Acest model generează imagini; aplicația de față este pentru conversație.", "This model makes pictures; this app is for conversation.")).color(p.muted));
                }
                self.sheet_panel(ui, m);
                if can_install {
                    ui.add_space(4.0);
                    if !selected {
                        if secondary_button(ui, self.t("Alege acest model", "Choose this model")).clicked() {
                            self.selected_model = Some(m.id.clone());
                            self.license_ok = false;
                        }
                    } else {
                        self.install_panel(ui, m);
                    }
                }
            });
    }

    /// Where new models are saved: next to this program, in the home folder, on any disk, or typed.
    pub(crate) fn location_picker(&mut self, ui: &mut Ui, needed_gb: f64) {
        let p = self.p;
        let current = self
            .s
            .models_dir
            .clone()
            .or_else(companion_core::models::default_models_root);
        ui.label(
            RichText::new(self.t("Unde se salvează modelele", "Where models are saved")).semibold(),
        );
        if let Some(c) = &current {
            let free = companion_core::models::free_gb_at(c);
            let line = match free {
                Some(f) => format!(
                    "{} · {} {}",
                    c.display(),
                    fmt_gb(f),
                    self.t("liberi", "free")
                ),
                None => c.display().to_string(),
            };
            let bad = free.is_some_and(|f| f < needed_gb * 1.05);
            ui.label(RichText::new(line).color(if bad { p.bad } else { p.muted }));
        }
        let mut choose: Option<std::path::PathBuf> = None;
        ui.horizontal_wrapped(|ui| {
            if let Some(b) = companion_core::models::beside_app() {
                if ui
                    .button(self.t("Lângă aplicație", "Next to this app"))
                    .on_hover_text(b.display().to_string())
                    .clicked()
                {
                    choose = Some(b);
                }
            }
            if let Some(d) = companion_core::models::default_models_root() {
                if ui
                    .button(self.t("În folderul meu personal", "In my home folder"))
                    .on_hover_text(d.display().to_string())
                    .clicked()
                {
                    choose = Some(d);
                }
            }
        });
        let drives = companion_core::models::drives();
        if !drives.is_empty() {
            ui.label(RichText::new(self.t("Pe alt disc:", "On another disk:")).color(p.muted));
            ui.horizontal_wrapped(|ui| {
                for d in &drives {
                    let label = format!(
                        "{} ({} {})",
                        d.mount.display(),
                        fmt_gb(d.free_gb),
                        self.t("liberi", "free")
                    );
                    if ui
                        .button(label)
                        .on_hover_text(d.models_folder().display().to_string())
                        .clicked()
                    {
                        choose = Some(d.models_folder());
                    }
                }
            });
        }
        let mut typed = self
            .s
            .models_dir
            .as_ref()
            .map(|d| d.display().to_string())
            .unwrap_or_default();
        ui.horizontal(|ui| {
            ui.label(self.t("sau orice folder:", "or any folder:"));
            let r = ui.add(
                egui::TextEdit::singleline(&mut typed)
                    .desired_width(280.0)
                    .hint_text("~/colibri-models"),
            );
            if r.lost_focus() && r.changed() {
                let t = typed.trim().trim_matches('"');
                if !t.is_empty() {
                    choose = Some(t.into());
                }
            }
        });
        if let Some(dir) = choose {
            if companion_core::models::writable(&dir) {
                self.s.models_dir = Some(dir);
                let _ = self.s.save();
                self.refresh_engine_info();
            } else {
                let m = self
                    .t(
                        "Nu pot scrie în «{}». Alege alt folder sau alt disc.",
                        "I can't write to «{}». Pick another folder or disk.",
                    )
                    .replacen("{}", &dir.display().to_string(), 1);
                self.toast(m);
            }
        }
        muted(
            ui,
            &p,
            self.t(
                "Modelele deja descărcate rămân unde sunt și apar în listă. Dacă alegi «Lângă aplicație», un model de zeci de GB ocupă discul pe care e instalată aplicația.",
                "Models you already downloaded stay where they are and still show in the list. If you pick “Next to this app”, a model of tens of GB uses the disk the app is installed on.",
            ),
        );
    }

    pub(crate) fn install_panel(&mut self, ui: &mut Ui, m: &ModelInfo) {
        let p = self.p;
        ui.separator();
        let free = self
            .hardware
            .ready()
            .map(|h| h.disk_free_gb)
            .unwrap_or(f64::MAX);
        ui.label(format!(
            "{} {}",
            self.t("Se descarcă", "Will download"),
            fmt_gb(m.download_gb),
        ));
        self.location_picker(ui, m.download_gb);
        let free = self
            .s
            .models_dir
            .clone()
            .or_else(companion_core::models::default_models_root)
            .and_then(|d| companion_core::models::free_gb_at(&d))
            .unwrap_or(free);
        if free < m.download_gb * 1.05 {
            ui.label(
                RichText::new(self.t(
                    "Spațiu pe disc prea mic pentru acest model.",
                    "Not enough free disk space for this model.",
                ))
                .color(p.bad)
                .semibold(),
            );
        }
        ui.label(
            RichText::new(self.t(
                "Poți întrerupe și relua oricând; descărcarea continuă chiar dacă închizi fereastra.",
                "You can pause and resume at any time; it keeps going even if you close this window.",
            ))
            .color(p.muted),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label(self.t(
                "Modelul vine de pe Hugging Face și are propria licență.",
                "The model comes from Hugging Face and has its own licence.",
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
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let ready = self.license_ok && free >= m.download_gb * 1.05;
            ui.add_enabled_ui(ready, |ui| {
                if primary_button(
                    ui,
                    &p,
                    self.t("Descarcă și instalează", "Download and install"),
                )
                .clicked()
                {
                    self.start_install(m);
                }
            });
            if secondary_button(ui, self.t("Renunț", "Cancel")).clicked() {
                self.selected_model = None;
            }
        });
    }

    pub(crate) fn status_badge(
        &self,
        m: &ModelInfo,
    ) -> (&'static str, egui::Color32, egui::Color32) {
        let p = self.p;
        match m.status.as_str() {
            "good" => (
                self.t("Merge foarte bine", "Runs well"),
                p.ok,
                p.accent_soft,
            ),
            "tight" => (
                self.t("Merge, la limită", "Runs, but tight"),
                p.warn,
                p.warn_soft,
            ),
            "needs-ram" => (
                self.t("Prea puțină memorie RAM", "Not enough RAM"),
                p.bad,
                p.bad_soft,
            ),
            "needs-disk" => (
                self.t("Prea puțin spațiu pe disc", "Not enough disk space"),
                p.bad,
                p.bad_soft,
            ),
            _ => (
                self.t("Nu merge aici", "Not supported here"),
                p.bad,
                p.bad_soft,
            ),
        }
    }

    pub(crate) fn speed_note(&self, m: &ModelInfo) -> &'static str {
        if m.streams_from_disk() {
            self.t(
                "Citește părți din model de pe SSD: mai lent, dar foarte capabil. Un SSD rapid ajută mult.",
                "Reads parts of the model from your SSD: slower, but very capable. A fast SSD helps a lot.",
            )
        } else {
            self.t(
                "Rulează din memoria RAM: răspunsuri mai rapide.",
                "Runs from RAM: faster answers.",
            )
        }
    }

    pub(crate) fn blurb(&self, m: &ModelInfo) -> String {
        match m.id.as_str() {
            "qwen36-35b" => self.t("Conversație generală, cu «gândire» și instrumente. O alegere bună pentru început.", "General conversation with thinking and tools. A good place to start.").into(),
            "qwen3-coder-30b" => self.t("Specializat în cod și instrumente, fără pasul de «gândire». Se descurcă și la întrebări obișnuite.", "Specialised in code and tools, without the “thinking” step. Handles everyday questions too.").into(),
            _ if m.summary.is_empty() => String::new(),
            _ => crate::i18n::engine_text(self.s.lang, &m.summary),
        }
    }

    // ------------------------------------------------------------ install, start, load

    fn screen_installing(&mut self, ui: &mut Ui) {
        let p = self.p;
        let inst = self
            .status
            .as_ref()
            .and_then(|s| s.install.clone())
            .filter(|i| i.running);
        let model = self
            .status
            .as_ref()
            .and_then(|s| s.setup.as_ref())
            .and_then(|s| s.model.as_ref())
            .and_then(|m| m.name.clone())
            .or_else(|| {
                let id = self.selected_model.as_deref()?;
                let name = self
                    .catalog
                    .ready()?
                    .iter()
                    .find(|m| m.id == id)
                    .map(|m| m.name.clone());
                name.or_else(|| Some(id.to_string()))
            })
            .unwrap_or_default();
        self.step_header(
            ui,
            3,
            self.t("Se pregătește totul", "Getting everything ready"),
            &model,
        );
        card(ui, &p, |ui| {
            let phase = inst.as_ref().map(|i| i.phase.as_str()).unwrap_or("build");
            let what = match phase {
                "build" => self.t(
                    "Pregătesc motorul pentru calculatorul tău…",
                    "Preparing the engine for your computer…",
                ),
                "download" => self.t("Descarc modelul…", "Downloading the model…"),
                "starting" => self.t("Pornesc…", "Starting…"),
                _ => self.t("Lucrez…", "Working…"),
            };
            ui.horizontal(|ui| {
                ui.add(Spinner::new());
                ui.label(RichText::new(what).size(18.0).semibold());
            });
            if let Some(i) = &inst {
                if let Some(f) = i.fraction() {
                    progress_bar(ui, &p, Some(f), &format!("{:.0}%", f * 100.0));
                    let of = self.t("din", "of");
                    let mut bits = vec![format!(
                        "{} {of} {}",
                        fmt_gb(i.done.unwrap_or(0.0) / coli::GB),
                        fmt_gb(i.total.unwrap_or(0.0) / coli::GB)
                    )];
                    if let Some(r) = i.rate.filter(|r| *r > 0.0) {
                        bits.push(fmt_rate(r));
                    }
                    if let Some(eta) = i.eta_seconds() {
                        let d = fmt_duration(eta, self.is_ro());
                        bits.push(self.t("încă ~{}", "~{} left").replacen("{}", &d, 1));
                    }
                    ui.label(bits.join("  ·  "));
                } else {
                    progress_bar(ui, &p, None, "");
                }
            } else {
                progress_bar(ui, &p, None, "");
            }
            ui.add_space(4.0);
            ui.label(RichText::new(self.t(
                "Poți lăsa fereastra deschisă și face altceva. Dacă o închizi, descărcarea continuă în fundal.",
                "Feel free to do something else. If you close this window, the download keeps going in the background.",
            )).color(p.muted));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if secondary_button(ui, self.t("Pune pe pauză", "Pause")).clicked() {
                    self.pause_install();
                }
            });
        });
        self.technical_details(ui);
    }

    fn technical_details(&mut self, ui: &mut Ui) {
        let p = self.p;
        if let Some(cmd) = coli::find_package_hint(&self.install_log) {
            ui.add_space(8.0);
            tinted(ui, p.warn_soft, p.warn, |ui| {
                ui.label(
                    RichText::new(self.t(
                        "Un pachet opțional ar face totul mai rapid",
                        "An optional package would make this faster",
                    ))
                    .semibold()
                    .color(p.warn),
                );
                ui.label(self.t(
                    "Pentru a folosi placa video, rulează această comandă într-un terminal (eu nu instalez niciodată pachete de sistem singură), apoi apasă «Reia».",
                    "To use your graphics card, run this command in a terminal (I never install system packages myself), then press “Resume”.",
                ));
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&cmd).monospace());
                    if ui.button(self.t("Copiază", "Copy")).clicked() {
                        ui.ctx().copy_text(cmd.clone());
                        self.toast(self.t("Copiat", "Copied"));
                    }
                });
            });
        }
        ui.add_space(8.0);
        egui::CollapsingHeader::new(self.t("Detalii tehnice", "Technical details")).show(
            ui,
            |ui| {
                egui::ScrollArea::vertical()
                    .max_height(220.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(if self.install_log.is_empty() {
                                "…"
                            } else {
                                &self.install_log
                            })
                            .monospace()
                            .size(12.0),
                        );
                    });
            },
        );
    }

    fn screen_install_problem(&mut self, ui: &mut Ui) {
        let p = self.p;
        let (phase, msg) = self
            .status
            .as_ref()
            .and_then(|s| s.install.as_ref())
            .map(|i| (i.phase.clone(), i.message.clone().unwrap_or_default()))
            .unwrap_or_default();
        let paused = phase == "interrupted";
        self.step_header(
            ui,
            3,
            if paused {
                self.t("Instalarea este pe pauză", "The setup is paused")
            } else {
                self.t("Instalarea s-a oprit", "The setup stopped")
            },
            self.t(
                "Nu s-a pierdut nimic: reluarea continuă de unde a rămas.",
                "Nothing is lost: resuming continues where it stopped.",
            ),
        );
        if !paused && !msg.is_empty() {
            tinted(ui, p.bad_soft, p.bad, |ui| {
                ui.label(RichText::new(&msg).color(p.bad));
            });
            ui.add_space(6.0);
        }
        self.technical_details(ui);
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let model = self
                .status
                .as_ref()
                .and_then(|s| s.install.as_ref())
                .and_then(|i| i.model.clone())
                .or_else(|| {
                    self.status
                        .as_ref()
                        .and_then(|s| s.setup.as_ref())
                        .and_then(|s| s.model.clone())
                        .and_then(|m| m.id)
                });
            if primary_button(ui, &p, self.t("Reia", "Resume")).clicked() {
                if let (Some(id), Loadable::Ready(models)) = (model.as_deref(), &self.catalog) {
                    if let Some(m) = models.iter().find(|m| m.id == id).cloned() {
                        self.start_install(&m);
                    }
                } else {
                    self.show_model_picker = true;
                }
            }
            if secondary_button(ui, self.t("Alege alt model", "Choose another model")).clicked() {
                self.show_model_picker = true;
            }
            if secondary_button(ui, self.t("Arată jurnalul", "Show the log")).clicked() {
                self.fetch_logs(true);
            }
        });
    }

    fn screen_start_server(&mut self, ui: &mut Ui) {
        let p = self.p;
        let name = self
            .status
            .as_ref()
            .and_then(|s| s.setup.as_ref())
            .and_then(|s| s.model.as_ref())
            .and_then(|m| m.name.clone())
            .unwrap_or_default();
        self.step_header(
            ui,
            4,
            self.t("Totul este instalat", "Everything is installed"),
            &name,
        );
        card(ui, &p, |ui| {
            ui.label(self.t(
                "Modelul trebuie încărcat în memorie înainte de prima conversație. La modelele mari poate dura câteva minute.",
                "The model has to be loaded into memory before your first chat. Big models can take a few minutes.",
            ));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if primary_button(ui, &p, self.t("Pornește modelul", "Start the model")).clicked()
                {
                    self.start_server();
                }
                if secondary_button(ui, self.t("Alege alt model", "Choose another model")).clicked()
                {
                    self.show_model_picker = true;
                }
            });
            let auto = self.t("Pornește singur data viitoare", "Start by itself next time");
            if ui
                .checkbox(&mut self.s.start_server_automatically, auto)
                .changed()
            {
                let _ = self.s.save();
            }
        });
        ui.add_space(8.0);
        if !self.chat.list.is_empty()
            && secondary_button(
                ui,
                self.t("Vezi conversațiile salvate", "See saved conversations"),
            )
            .clicked()
        {
            self.browse_history = true;
        }
    }

    fn screen_loading(&mut self, ui: &mut Ui) {
        let p = self.p;
        self.step_header(ui, 4, self.t("Se încarcă modelul", "Loading the model"), "");
        card(ui, &p, |ui| {
            ui.horizontal(|ui| {
                ui.add(Spinner::new().size(28.0));
                let secs = self
                    .starting_since
                    .map(|t| t.elapsed().as_secs_f64())
                    .unwrap_or(0.0);
                ui.label(
                    RichText::new(format!(
                        "{} {}",
                        self.t("Se încarcă… de", "Loading… for"),
                        fmt_duration(secs, self.is_ro())
                    ))
                    .size(18.0),
                );
            });
            ui.label(RichText::new(self.t(
                "Modelul se citește de pe disc în memorie. La prima pornire poate dura mai mult. Imediat ce e gata, se deschide conversația.",
                "The model is being read from disk into memory. The first start can take longer. The chat opens by itself as soon as it's ready.",
            )).color(p.muted));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if secondary_button(ui, self.t("Arată jurnalul", "Show the log")).clicked() {
                    self.fetch_logs(false);
                }
                if secondary_button(ui, self.t("Oprește", "Stop")).clicked() {
                    self.stop_server();
                }
            });
        });
        if !self.chat.list.is_empty() {
            ui.add_space(8.0);
            if secondary_button(
                ui,
                self.t(
                    "Între timp, vezi conversațiile salvate",
                    "Meanwhile, see saved conversations",
                ),
            )
            .clicked()
            {
                self.browse_history = true;
            }
        }
        ui.ctx().request_repaint_after(Duration::from_millis(500));
    }
}
