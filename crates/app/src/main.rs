//! Colibri Companion: a friendly desktop interface for the colibri engine.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod chat_ui;
mod demo;
mod guide_ui;
mod i18n;
mod models_ui;
mod resources_ui;
mod screens;
mod sheets_ui;
mod theme;
mod widgets;
mod windows;

use app::{CompanionApp, Phase};
use companion_core::{Lang, Settings};

/// Command-line help in the interface language: [tagline, "USAGE", --demo, --lang, --version, unknown option].
fn help_words(lang: Lang) -> [&'static str; 6] {
    match lang.code() {
        "en" => [
            "an unofficial, friendly interface for colibri",
            "USAGE",
            "try the interface with a built-in fake model (nothing is downloaded)",
            "start in English, Romanian, French, Italian, Spanish, Portuguese, German or Hungarian",
            "print the version",
            "unknown option",
        ],
        "ro" => [
            "o interfață prietenoasă, neoficială, pentru colibri",
            "UTILIZARE",
            "încearcă interfața cu un model fals încorporat (nu se descarcă nimic)",
            "pornește în engleză, română, franceză, italiană, spaniolă, portugheză, germană sau maghiară",
            "afișează versiunea",
            "opțiune necunoscută",
        ],
        "fr" => [
            "une interface conviviale et non officielle pour colibri",
            "UTILISATION",
            "essayer l'interface avec un faux modèle intégré (rien n'est téléchargé)",
            "démarrer en anglais, roumain, français, italien, espagnol, portugais, allemand ou hongrois",
            "afficher la version",
            "option inconnue",
        ],
        "it" => [
            "un'interfaccia amichevole e non ufficiale per colibri",
            "USO",
            "prova l'interfaccia con un modello finto incorporato (non si scarica nulla)",
            "avvia in inglese, rumeno, francese, italiano, spagnolo, portoghese, tedesco o ungherese",
            "mostra la versione",
            "opzione sconosciuta",
        ],
        "es" => [
            "una interfaz amigable y no oficial para colibri",
            "USO",
            "probar la interfaz con un modelo falso integrado (no se descarga nada)",
            "iniciar en inglés, rumano, francés, italiano, español, portugués, alemán o húngaro",
            "mostrar la versión",
            "opción desconocida",
        ],
        "pt" => [
            "uma interface amigável e não oficial para o colibri",
            "UTILIZAÇÃO",
            "experimentar a interface com um modelo falso integrado (nada é transferido)",
            "iniciar em inglês, romeno, francês, italiano, espanhol, português, alemão ou húngaro",
            "mostrar a versão",
            "opção desconhecida",
        ],
        "de" => [
            "eine freundliche, inoffizielle Oberfläche für colibri",
            "VERWENDUNG",
            "die Oberfläche mit einem eingebauten Schein-Modell ausprobieren (es wird nichts heruntergeladen)",
            "in Englisch, Rumänisch, Französisch, Italienisch, Spanisch, Portugiesisch, Deutsch oder Ungarisch starten",
            "die Version ausgeben",
            "unbekannte Option",
        ],
        "hu" => [
            "barátságos, nem hivatalos felület a colibrihez",
            "HASZNÁLAT",
            "a felület kipróbálása beépített álmodellel (semmi nem töltődik le)",
            "indítás angolul, románul, franciául, olaszul, spanyolul, portugálul, németül vagy magyarul",
            "a verzió kiírása",
            "ismeretlen kapcsoló",
        ],
        _ => unreachable!(),
    }
}

fn help_text(lang: Lang) -> String {
    let w = help_words(lang);
    format!(
        "Colibri Companion — {}\n\n{}: colibri-companion [--demo] [--lang en|ro|fr|it|es|pt|de|hu] [--version]\n\n  --demo       {}\n  --lang       {}\n  --version    {}\n",
        w[0], w[1], w[2], w[3], w[4]
    )
}

impl eframe::App for CompanionApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let started = std::time::Instant::now();
        self.build_ui(ui);
        // `COLIBRI_PERF=1` prints how long laying out the interface takes (for tuning only).
        if self.perf_log {
            eprintln!("ui {:.2} ms", started.elapsed().as_secs_f64() * 1000.0);
        }
    }
}

impl CompanionApp {
    fn build_ui(&mut self, ui: &mut egui::Ui) {
        self.tick();
        let phase = self.phase();
        self.top_bar(ui);
        self.update_banner(ui);
        self.status_bar(ui);
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(self.p.bg))
            .show(ui, |ui| {
                if phase == Phase::Chat {
                    self.show_chat(ui);
                } else {
                    self.show_phase(ui, phase);
                }
            });
        let ctx = ui.ctx().clone();
        self.handle_drops(&ctx);
        self.windows(&ctx);
        self.toasts(&ctx);
        self.shortcuts(&ctx);
    }
}

impl CompanionApp {
    /// Dropping the colibri folder onto the window is the friendliest folder picker there is.
    fn handle_drops(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        let paths: Vec<std::path::PathBuf> = dropped
            .iter()
            .map(|f| f.path().to_path_buf())
            .filter(|p| !p.as_os_str().is_empty())
            .collect();
        if paths.is_empty() {
            return;
        }
        if self.coli.is_none() {
            self.root_input = paths[0].display().to_string();
            let text = self.root_input.clone();
            self.use_root_from_text(&text);
        } else if self.server_ready() {
            self.attach_files(&paths);
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        use egui::{Key, Modifiers};
        if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::N)) {
            self.new_conversation();
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::Comma)) {
            self.show_settings = !self.show_settings;
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::K)) {
            self.chat.focus_search = true;
        }
        if self.chat.generating.is_some()
            && ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape))
        {
            self.stop_generation();
        }
    }
}

fn app_icon() -> egui::IconData {
    // A teal disc with a lighter wing shape, drawn pixel by pixel: no image file to ship.
    let n = 64usize;
    let mut rgba = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let (dx, dy) = (x as f32 - 31.5, y as f32 - 31.5);
            let r = (dx * dx + dy * dy).sqrt();
            let i = (y * n + x) * 4;
            if r <= 30.0 {
                let wing = (dx + 6.0).powi(2) / 220.0 + (dy - 2.0).powi(2) / 70.0 < 1.0 && dx < 8.0;
                let c = if wing {
                    [236, 250, 246]
                } else {
                    [12, 122, 105]
                };
                rgba[i..i + 3].copy_from_slice(&c);
                rgba[i + 3] = 255;
            }
        }
    }
    egui::IconData {
        rgba,
        width: n as u32,
        height: n as u32,
    }
}

fn main() -> eframe::Result {
    let mut demo = false;
    let mut lang_override = None;
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let help_lang = argv
        .windows(2)
        .find(|w| w[0] == "--lang")
        .and_then(|w| Lang::from_code(&w[1]))
        .unwrap_or_else(Lang::from_env);
    let mut args = argv.into_iter();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--demo" => demo = true,
            "--lang" => lang_override = args.next(),
            "--version" | "-V" => {
                println!("colibri-companion {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--help" | "-h" => {
                print!("{}", help_text(help_lang));
                return Ok(());
            }
            other => {
                eprintln!(
                    "{}: {other}\n\n{}",
                    help_words(help_lang)[5],
                    help_text(help_lang)
                );
                std::process::exit(2);
            }
        }
    }
    let mut settings = Settings::load();
    if let Some(l) = lang_override.as_deref().and_then(Lang::from_code) {
        settings.lang = l;
    }
    let demo_base = if demo {
        match demo::start(settings.lang) {
            Ok(s) => Some(s.base),
            Err(e) => {
                eprintln!("could not start the demo server: {e}");
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Colibri Companion")
            .with_inner_size([1120.0, 780.0])
            .with_min_inner_size([760.0, 520.0])
            .with_icon(app_icon()),
        ..Default::default()
    };
    eframe::run_native(
        "Colibri Companion",
        options,
        Box::new(move |cc| {
            theme::install_fonts(&cc.egui_ctx);
            Ok(Box::new(CompanionApp::new(
                cc.egui_ctx.clone(),
                settings,
                demo_base,
            )))
        }),
    )
}
