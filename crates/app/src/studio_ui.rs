//! The creative models window: a map of open models for pictures, video, voice and music, and a
//! small "make a picture" box that talks to any OpenAI-style server on this computer or network
//! (colibri with Qwen-Image-2.1, or ComfyUI/LocalAI behind an images endpoint).

use crate::app::{CompanionApp, Event};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::studio::{self, Commercial, Modality, Where};
use egui::{Align2, RichText, vec2};

pub struct StudioState {
    pub open: bool,
    pub tab: Modality,
    pub commercial_only: bool,
    pub prompt: String,
    pub endpoint: String,
    pub size: String,
    pub busy: bool,
    pub result: String,
    pub last: Option<std::path::PathBuf>,
}

impl Default for StudioState {
    fn default() -> Self {
        StudioState {
            open: false,
            tab: Modality::Image,
            commercial_only: false,
            prompt: String::new(),
            endpoint: String::new(),
            size: "1024x1024".into(),
            busy: false,
            result: String::new(),
            last: None,
        }
    }
}

fn modality_name(app: &CompanionApp, m: Modality) -> &'static str {
    match m {
        Modality::Image => app.t("Imagini", "Pictures"),
        Modality::Video => app.t("Video", "Video"),
        Modality::Speak => app.t("Voce (text → sunet)", "Voice (text → sound)"),
        Modality::Listen => app.t("Ascultare (sunet → text)", "Listening (sound → text)"),
        Modality::Music => app.t("Muzică", "Music"),
    }
}

impl CompanionApp {
    pub fn open_studio(&mut self) {
        if self.studio.endpoint.is_empty() {
            self.studio.endpoint = self.base();
        }
        self.studio.open = true;
    }

    fn make_picture(&mut self) {
        let endpoint = companion_core::chat::normalize_base(&self.studio.endpoint);
        // Only this computer or this network: the picture request is sent in plain http.
        let local = companion_core::net::is_local_base(&endpoint)
            || companion_core::connect::check_url(&endpoint).is_ok()
                && companion_core::http::parse_authority(&endpoint).is_ok_and(|(h, _)| {
                    h.parse::<std::net::Ipv4Addr>()
                        .is_ok_and(|ip| ip.is_private())
                });
        if !local {
            self.studio.result = self
                .t(
                    "Adresa serverului nu e bună: aici merg doar servere http:// de pe acest calculator sau din rețeaua ta.",
                    "The server address is not valid: only http:// servers on this computer or your own network work here.",
                )
                .into();
            return;
        }
        let prompt = self.studio.prompt.trim().to_string();
        if prompt.is_empty() {
            return;
        }
        let body = studio::image_request(
            &prompt,
            &self.studio.size,
            &self.model_id.clone().unwrap_or_else(|| "colibri".into()),
        );
        self.studio.busy = true;
        self.studio.result = self
            .t(
                "Generez… poate dura câteva minute.",
                "Making it… this can take a few minutes.",
            )
            .into();
        self.spawn(move |tx| {
            let opts = companion_core::http::RequestOptions {
                connect_timeout: std::time::Duration::from_secs(5),
                read_timeout: Some(std::time::Duration::from_secs(1800)),
                bearer: None,
                cancel: None,
            };
            let result = companion_core::http::send(
                &endpoint,
                "POST",
                "/v1/images/generations",
                Some(&body),
                &opts,
            )
            .map_err(|e| e.to_string())
            .and_then(|r| {
                let status = r.status;
                let text = r.text();
                if status >= 400 {
                    studio::image_from_response(&text).and(Err(format!("HTTP {status}")))
                } else {
                    studio::image_from_response(&text)
                }
            })
            .and_then(|bytes| {
                let dir = companion_core::paths::downloads_dir();
                std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                let ext = studio::extension_of(&bytes);
                let mut path = dir.join(format!("colibri-imagine.{ext}"));
                let mut n = 2;
                while path.exists() {
                    path = dir.join(format!("colibri-imagine-{n}.{ext}"));
                    n += 1;
                }
                std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
                Ok(path)
            });
            let _ = tx.send(Event::Picture(result));
        });
    }

    pub fn picture_done(&mut self, r: Result<std::path::PathBuf, String>) {
        self.studio.busy = false;
        match r {
            Ok(p) => {
                self.studio.result = format!("{} {}", self.t("Salvat în", "Saved to"), p.display());
                self.studio.last = Some(p);
            }
            Err(e) => {
                self.studio.result = format!("{}: {e}", self.t("Nu a mers", "It failed"));
            }
        }
    }

    pub fn studio_window(&mut self, ctx: &egui::Context) {
        if !self.studio.open {
            return;
        }
        let p = self.p;
        let mut open = true;
        let l_commercial = self.t(
            "Doar ce se poate folosi comercial",
            "Only what can be used commercially",
        );
        let h_prompt = self.t("Descrie imaginea…", "Describe the picture…");
        egui::Window::new(self.t("Modele creative", "Creative models"))
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_size(fit(ctx, 780.0, 600.0))
            .collapsible(false)
            .show(ctx, |ui| {
                muted(ui, &p, self.t(
                    "Modele deschise care fac imagini, video, voce și muzică sau înțeleg sunetul, căutate online în noiembrie 2026. colibri servește singur doar Qwen-Image-2.1; celelalte rulează în alt program (ComfyUI, LocalAI), către care poți îndrepta această aplicație. Licențele diferă: verifică pagina fiecărui model.",
                    "Open models that make pictures, video, voice and music or understand sound, looked up online in November 2026. colibri itself serves only Qwen-Image-2.1; the others run in another program (ComfyUI, LocalAI) that you can point this app at. Licences differ: check each model's page.",
                ));
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    for m in Modality::ALL {
                        if ui.selectable_label(self.studio.tab == m, modality_name(self, m)).clicked() {
                            self.studio.tab = m;
                        }
                    }
                });
                ui.checkbox(&mut self.studio.commercial_only, l_commercial);
                ui.separator();
                egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                    for x in studio::of(self.studio.tab) {
                        if self.studio.commercial_only && x.commercial != Commercial::Yes {
                            continue;
                        }
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new(x.name).semibold());
                            muted(ui, &p, x.maker);
                            if x.run == Where::Colibri {
                                ui.label(RichText::new(self.t("servit de colibri", "served by colibri")).small().color(p.ok));
                            }
                            let (t, c) = match x.commercial {
                                Commercial::Yes => (self.t("comercial: da", "commercial: yes"), p.ok),
                                Commercial::No => (self.t("comercial: nu", "commercial: no"), p.bad),
                                Commercial::Check => (self.t("comercial: verifică", "commercial: check"), p.warn),
                            };
                            ui.label(RichText::new(t).small().color(c));
                        });
                        muted(ui, &p, &format!("{} · {}", x.licence, x.needs));
                        ui.label(self.t(x.ro, x.en));
                        ui.add_space(6.0);
                    }
                    ui.separator();
                    ui.label(RichText::new(self.t("Programe care le rulează", "Programs that run them")).strong());
                    for r in studio::RUNNERS {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new(r.name).semibold());
                            if ui.small_button("↗").clicked() {
                                companion_core::proc::open_externally(r.url);
                            }
                        });
                        muted(ui, &p, self.t(r.ro, r.en));
                    }
                });
                ui.separator();
                ui.label(RichText::new(self.t("Fă o imagine", "Make a picture")).strong());
                ui.horizontal(|ui| {
                    ui.label(self.t("Server:", "Server:"));
                    ui.add(egui::TextEdit::singleline(&mut self.studio.endpoint).desired_width(260.0));
                    egui::ComboBox::from_id_salt("img-size")
                        .selected_text(self.studio.size.clone())
                        .show_ui(ui, |ui| {
                            for s in ["512x512", "768x768", "1024x1024", "1024x1536", "1536x1024"] {
                                ui.selectable_value(&mut self.studio.size, s.to_string(), s);
                            }
                        });
                });
                ui.add(egui::TextEdit::multiline(&mut self.studio.prompt).hint_text(h_prompt).desired_rows(2).desired_width(f32::INFINITY));
                ui.horizontal(|ui| {
                    let ok = !self.studio.busy && !self.studio.prompt.trim().is_empty();
                    if ui.add_enabled(ok, egui::Button::new(self.t("Generează", "Generate"))).clicked() {
                        self.make_picture();
                    }
                    if self.studio.busy {
                        crate::theme::busy(ui);
                    }
                    if let Some(path) = self.studio.last.clone() {
                        if ui.button(self.t("Deschide imaginea", "Open the picture")).clicked() {
                            companion_core::proc::open_externally(&path.display().to_string());
                        }
                    }
                });
                if !self.studio.result.is_empty() {
                    muted(ui, &p, &self.studio.result);
                }
                muted(ui, &p, self.t(
                    "Neîncercat încă pe un server colibri real: formatul cererii urmează convenția OpenAI. Cu un model altul decât Qwen-Image-2.1 serverul poate răspunde cu o eroare.",
                    "Not yet tried against a real colibri server: the request follows the OpenAI convention. With a model other than Qwen-Image-2.1 the server may answer with an error.",
                ));
            });
        self.studio.open = open;
    }
}
