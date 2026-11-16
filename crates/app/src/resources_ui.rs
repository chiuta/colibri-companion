//! "Resources and privacy": what the model is using on this computer, and the evidence that
//! nothing talks to the outside world when it should not.

use crate::app::{CompanionApp, Phase, ResView};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::net::{self, Kind};
use egui::{Align2, Color32, RichText, Stroke, Ui, vec2};
use std::collections::VecDeque;

/// A label above a plain bar, so the text stays readable at any fill level.
fn gauge(ui: &mut Ui, p: &crate::theme::Palette, fraction: f32, label: &str) {
    ui.label(label);
    progress_bar(ui, p, Some(fraction.clamp(0.0, 1.0)), "");
}

fn sparkline(ui: &mut Ui, hist: &VecDeque<f32>, color: Color32, bg: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 38.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 6.0, bg);
    if hist.len() < 2 {
        return;
    }
    let n = hist.len() as f32 - 1.0;
    let pts: Vec<egui::Pos2> = hist
        .iter()
        .enumerate()
        .map(|(i, v)| {
            egui::pos2(
                rect.left() + rect.width() * i as f32 / n,
                rect.bottom() - 3.0 - (rect.height() - 6.0) * (v / 100.0).clamp(0.0, 1.0),
            )
        })
        .collect();
    ui.painter()
        .add(egui::Shape::line(pts, Stroke::new(2.0, color)));
}

impl CompanionApp {
    pub fn resources_window(&mut self, ctx: &egui::Context) {
        if !self.show_resources {
            return;
        }
        let mut open = true;
        egui::Window::new(self.t("Resurse și confidențialitate", "Resources and privacy"))
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_width(680.0)
            .default_height(640.0)
            .collapsible(false)
            .vscroll(true)
            .show(ctx, |ui| self.resources_body(ui));
        if !open {
            self.set_resources_open(false);
        }
    }

    fn resources_body(&mut self, ui: &mut Ui) {
        let p = self.p;
        let Some(view) = self.res.clone() else {
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new());
                ui.label(self.t("Măsor…", "Measuring…"));
            });
            return;
        };
        self.privacy_section(ui, &view);
        ui.add_space(10.0);
        self.advice_section(ui, &view);
        ui.add_space(10.0);

        // ---- resources
        ui.label(
            RichText::new(self.t(
                "Ce folosește acum calculatorul",
                "What your computer is using now",
            ))
            .size(17.0)
            .semibold(),
        );
        let s = &view.snap;
        let cpu = format!(
            "{} {:.0}% · {} {}",
            self.t("Procesor", "Processor"),
            s.cpu_percent,
            s.cores,
            self.t("nuclee", "cores")
        );
        gauge(ui, &p, s.cpu_percent / 100.0, &cpu);
        sparkline(ui, &self.cpu_hist, p.accent, p.panel);
        ui.add_space(4.0);
        let ram_frac = if s.ram_total_gb > 0.0 {
            (s.ram_used_gb / s.ram_total_gb) as f32
        } else {
            0.0
        };
        let ram = format!(
            "RAM {:.1} / {:.1} GB{}",
            s.ram_used_gb,
            s.ram_total_gb,
            if s.swap_used_gb > 0.2 {
                format!(" · swap {:.1} GB", s.swap_used_gb)
            } else {
                String::new()
            }
        );
        gauge(ui, &p, ram_frac, &ram);
        sparkline(ui, &self.ram_hist, p.ok, p.panel);
        ui.add_space(4.0);
        if s.gpus.is_empty() {
            muted(
                ui,
                &p,
                self.t(
                    "Placa video: nu pot citi încărcarea (fără NVIDIA sau fără nvidia-smi). Modelul poate totuși folosi un cip grafic prin Vulkan.",
                    "Graphics card: its load can't be read (no NVIDIA card or no nvidia-smi). The model may still use a graphics chip through Vulkan.",
                ),
            );
        }
        for g in &s.gpus {
            let util = g.util_percent.unwrap_or(0.0);
            let label = format!(
                "{} · {:.0}%{}",
                g.name,
                util,
                g.temp_c.map(|t| format!(" · {t:.0}°C")).unwrap_or_default()
            );
            gauge(ui, &p, util / 100.0, &label);
            let vf = if g.vram_total_gb > 0.0 {
                (g.vram_used_gb / g.vram_total_gb) as f32
            } else {
                0.0
            };
            gauge(
                ui,
                &p,
                vf,
                &format!("VRAM {:.1} / {:.1} GB", g.vram_used_gb, g.vram_total_gb),
            );
        }
        if let (Some(free), Some(total)) = (s.disk_free_gb, s.disk_total_gb) {
            ui.add_space(4.0);
            let used = if total > 0.0 {
                ((total - free) / total) as f32
            } else {
                0.0
            };
            gauge(
                ui,
                &p,
                used,
                &format!(
                    "{} {:.0} GB {} {:.0} GB",
                    self.t("Disc (modele): liber", "Disk (models): free"),
                    free,
                    self.t("din", "of"),
                    total
                ),
            );
        }
        ui.add_space(8.0);
        ui.label(RichText::new(self.t("Cine consumă", "Who is using it")).semibold());
        for pr in &s.procs {
            ui.label(format!(
                "{} — {}% {} · {:.2} GB RAM · {} {}",
                pr.label,
                pr.cpu_percent.round(),
                self.t("din nucleu", "of a core"),
                pr.rss_gb,
                pr.processes,
                self.t("procese", "processes")
            ));
        }
        if s.procs.len() < 2 {
            muted(
                ui,
                &p,
                self.t(
                    "Motorul colibri nu rulează acum, deci nu consumă nimic.",
                    "The colibri engine isn't running now, so it uses nothing.",
                ),
            );
        }
    }

    fn privacy_section(&mut self, ui: &mut Ui, view: &ResView) {
        let p = self.p;
        let external: Vec<_> = view
            .conns
            .iter()
            .filter(|(_, c)| c.leaves_computer())
            .collect();
        let exposed: Vec<_> = view
            .conns
            .iter()
            .filter(|(_, c)| c.exposed_listener())
            .collect();
        let expected = self.phase() == Phase::Installing || self.install_spawning || self.cloning;
        let remote_server = !net::is_local_base(&self.base());

        if let Some(err) = &view.conns_error {
            tinted(ui, p.warn_soft, p.warn, |ui| {
                ui.label(
                    RichText::new(self.t(
                        "Nu pot citi acum conexiunile sistemului.",
                        "I can't read the system's connections right now.",
                    ))
                    .semibold()
                    .color(p.warn),
                );
                ui.label(RichText::new(err).small());
            });
        } else if external.is_empty() && !remote_server {
            tinted(ui, p.accent_soft, p.ok, |ui| {
                ui.label(
                    RichText::new(self.t(
                        "✔ Nicio conexiune către exterior în acest moment",
                        "✔ No connection leaves this computer right now",
                    ))
                    .size(18.0)
                    .semibold()
                    .color(p.ok),
                );
                ui.label(self.t(
                    "Aplicația și motorul colibri vorbesc doar între ele, în interiorul calculatorului tău. Conversațiile tale nu sunt trimise nicăieri.",
                    "The app and the colibri engine only talk to each other, inside your computer. Your conversations are not sent anywhere.",
                ));
            });
        } else if remote_server {
            tinted(ui, p.warn_soft, p.warn, |ui| {
                ui.label(
                    RichText::new(self.t(
                        "Folosești un server la distanță",
                        "You are using a remote server",
                    ))
                    .semibold()
                    .color(p.warn),
                );
                ui.label(format!(
                    "{} {}",
                    self.t("Mesajele se trimit la:", "Your messages are sent to:"),
                    self.base()
                ));
                ui.label(self.t(
                    "Așa ai ales în Setări. Revino la adresa locală (http://127.0.0.1:8000) ca totul să rămână pe acest calculator.",
                    "You chose this in Settings. Go back to the local address (http://127.0.0.1:8000) to keep everything on this computer.",
                ));
            });
        } else {
            let (fill, edge) = if expected {
                (p.warn_soft, p.warn)
            } else {
                (p.bad_soft, p.bad)
            };
            tinted(ui, fill, edge, |ui| {
                ui.label(
                    RichText::new(if expected {
                        self.t(
                            "Se descarcă ceva, la cererea ta",
                            "Something is downloading, because you asked for it",
                        )
                    } else {
                        self.t(
                            "Atenție: o conexiune către exterior",
                            "Heads up: a connection to the outside",
                        )
                    })
                    .semibold()
                    .color(edge),
                );
                for (label, c) in &external {
                    if let Some(r) = c.remote {
                        ui.label(RichText::new(format!("{label}  →  {r}")).monospace());
                    }
                }
                if expected {
                    ui.label(self.t(
                        "Descărcarea unui model sau a lui colibri are nevoie de internet. Se oprește singură când termină.",
                        "Downloading a model or colibri needs the internet. It stops by itself when done.",
                    ));
                }
            });
        }

        if !exposed.is_empty() {
            ui.add_space(6.0);
            tinted(ui, p.warn_soft, p.warn, |ui| {
                ui.label(
                    RichText::new(self.t(
                        "Un server ascultă și pentru alte calculatoare din rețea",
                        "A server is also listening for other computers on your network",
                    ))
                    .semibold()
                    .color(p.warn),
                );
                for (label, c) in &exposed {
                    ui.label(RichText::new(format!("{label}  ·  {}", c.local)).monospace());
                }
            });
        }

        ui.add_space(6.0);
        let inspected: usize = view.snap.procs.iter().map(|p| p.processes).sum();
        ui.label(
            RichText::new(
                self.t(
                    "Verificat acum: {} procese (aplicația și motorul colibri), {} socket-uri deschise.",
                    "Checked just now: {} processes (the app and the colibri engine), {} open sockets.",
                )
                .replacen("{}", &inspected.to_string(), 1)
                .replacen("{}", &view.conns.len().to_string(), 1),
            )
            .color(p.muted),
        );
        // what the app itself opened
        let log = &view.applog;
        if log.external.is_empty() {
            ui.label(
                RichText::new(
                    self.t(
                        "De la pornire, aplicația a deschis {} conexiuni, toate către acest calculator.",
                        "Since it started, the app opened {} connections, all to this computer.",
                    )
                    .replacen("{}", &log.total.to_string(), 1),
                )
                .color(p.muted),
            );
        } else {
            ui.label(
                RichText::new(
                    self.t(
                        "De la pornire, aplicația a deschis {} conexiuni; {} către alte adrese:",
                        "Since it started, the app opened {} connections; {} to other addresses:",
                    )
                    .replacen("{}", &log.total.to_string(), 1)
                    .replacen("{}", &(log.total - log.local).to_string(), 1),
                )
                .color(p.warn),
            );
            for (ip, n) in &log.external {
                ui.label(RichText::new(format!("{ip}  ×{n}")).monospace());
            }
        }

        // what went out to the internet (only host names are ever kept)
        self.egress_block(ui, &view.egress);

        // the evidence itself
        egui::CollapsingHeader::new(self.t("Arată dovada (toate conexiunile)", "Show the evidence (every connection)"))
            .default_open(false)
            .show(ui, |ui| {
                if view.conns.is_empty() {
                    muted(ui, &p, self.t("Niciun socket deschis de aceste procese.", "These processes have no open sockets."));
                }
                egui::Grid::new("conns").striped(true).show(ui, |ui| {
                    for h in [
                        self.t("Proces", "Process"),
                        self.t("Tip", "Kind"),
                        self.t("Local", "Local"),
                        self.t("Către", "To"),
                        "",
                    ] {
                        ui.label(RichText::new(h).semibold());
                    }
                    ui.end_row();
                    for (label, c) in &view.conns {
                        ui.label(label);
                        ui.label(c.proto);
                        ui.label(RichText::new(c.local.to_string()).monospace());
                        match (c.kind, c.remote) {
                            (Kind::Connected, Some(r)) => ui.label(RichText::new(r.to_string()).monospace()),
                            _ => ui.label(RichText::new(self.t("ascultă", "listening")).color(p.muted)),
                        };
                        if c.leaves_computer() {
                            ui.label(RichText::new(self.t("exterior", "external")).color(p.warn).semibold());
                        } else {
                            ui.label(RichText::new(self.t("local", "local")).color(p.ok));
                        }
                        ui.end_row();
                    }
                });
                ui.add_space(6.0);
                ui.label(self.t(
                    "Poți verifica și singur, cu uneltele sistemului tău, fără să te bazezi pe această aplicație:",
                    "You can also check for yourself with your system's own tools, without relying on this app:",
                ));
                ui.horizontal(|ui| {
                    ui.label(RichText::new(net::verify_command()).monospace());
                    if secondary_button(ui, self.t("Copiază", "Copy")).clicked() {
                        ui.ctx().copy_text(net::verify_command().to_string());
                    }
                });
                muted(
                    ui,
                    &p,
                    self.t(
                        "Aceasta este o fotografie a momentului de față. Singurele situații în care internetul este folosit sunt cele cerute de tine: descărcarea lui colibri, a unui model, verificarea de actualizări sau platformele conectate de tine.",
                        "This is a snapshot of right now. The only times the internet is used are the ones you ask for: downloading colibri, downloading a model, checking for updates or the platforms you connected.",
                    ),
                );
            });
    }
}

impl CompanionApp {
    fn egress_block(&self, ui: &mut egui::Ui, log: &[net::Egress]) {
        let p = self.p;
        if log.is_empty() {
            ui.label(
                RichText::new(self.t(
                    "De la pornire, nimic nu a plecat spre internet (platforme, verificări de actualizări, descărcări).",
                    "Since it started, nothing has gone out to the internet (platforms, update checks, downloads).",
                ))
                .color(p.ok),
            );
            return;
        }
        ui.label(
            RichText::new(
                self.t(
                    "De la pornire, {} cereri au plecat spre internet, către:",
                    "Since it started, {} requests went out to the internet, to:",
                )
                .replacen("{}", &log.len().to_string(), 1),
            )
            .color(p.warn),
        );
        for (host, n) in net::egress_hosts(log) {
            ui.label(RichText::new(format!("{host}  ×{n}")).monospace());
        }
        egui::CollapsingHeader::new(self.t("Jurnalul cererilor către internet", "Journal of requests to the internet"))
            .id_salt("egress-journal")
            .show(ui, |ui| {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Se păstrează doar numele gazdei și felul cererii, niciodată calea, parametrii sau jetoanele. Jurnalul se șterge când închizi aplicația.",
                        "Only the host name and the kind of request are kept, never the path, the parameters or tokens. The journal is erased when you close the app.",
                    ),
                );
                egui::Grid::new("egress-grid").striped(true).show(ui, |ui| {
                    for e in log.iter().rev().take(100) {
                        let secs = e.at.elapsed().map(|d| d.as_secs()).unwrap_or(0);
                        ui.label(RichText::new(format!("{}s", secs)).color(p.muted));
                        ui.label(RichText::new(&e.host).monospace());
                        ui.label(e.method);
                        ui.label(match e.purpose {
                            "message" => self.t("trimitere", "sending"),
                            "update check" => self.t("verificare actualizări", "update check"),
                            "download" => self.t("descărcare", "download"),
                            _ => self.t("citire", "reading"),
                        });
                        if e.ok {
                            ui.label(RichText::new("✔").color(p.ok));
                        } else {
                            ui.label(RichText::new(self.t("eșuat", "failed")).color(p.bad));
                        }
                        ui.end_row();
                    }
                });
            });
    }

    /// A slim bar under the top bar when colibri or this app has something new.
    pub fn update_banner(&mut self, ui: &mut Ui) {
        if self.upd.dismissed || self.demo || (self.upd.colibri.is_none() && self.upd.app.is_none())
        {
            return;
        }
        let p = self.p;
        egui::Panel::top("updatebar")
            .frame(
                egui::Frame::new()
                    .fill(p.accent_soft)
                    .inner_margin(egui::Margin::symmetric(14, 6)),
            )
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if let Some(u) = self.upd.colibri.clone() {
                        ui.label(RichText::new("⬆").color(p.accent));
                        let msg = self
                            .t(
                                "colibri are o versiune nouă ({}). Actualizarea aduce viteză și modele noi.",
                                "colibri has a new version ({}). Updating brings speed-ups and new models.",
                            )
                            .replacen("{}", u.title.trim(), 1);
                        ui.label(msg);
                        let label = if self.server_ready() {
                            self.t("Oprește modelul și actualizează", "Stop the model and update")
                        } else {
                            self.t("Actualizează colibri", "Update colibri")
                        };
                        ui.add_enabled_ui(!self.upd.updating && !self.install_spawning, |ui| {
                            if primary_button(ui, &p, label).clicked() {
                                self.update_colibri();
                            }
                        });
                        if self.upd.updating {
                            ui.add(egui::Spinner::new());
                        }
                    }
                    if let Some(r) = self.upd.app.clone() {
                        ui.label(RichText::new("⬆").color(p.accent));
                        ui.label(
                            self.t(
                                "Colibri Companion {} este disponibil.",
                                "Colibri Companion {} is available.",
                            )
                            .replacen("{}", &r.tag, 1),
                        );
                        if secondary_button(ui, self.t("Vezi descărcarea", "See the download")).clicked() {
                            companion_core::proc::open_externally(&r.url);
                        }
                    }
                    if secondary_button(ui, self.t("Mai târziu", "Later")).clicked() {
                        self.upd.dismissed = true;
                    }
                });
            });
    }
}
