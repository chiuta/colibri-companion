//! Scanned files in the Documents window: if the independent program LegatorOCR (MIT) is
//! installed, run it on a scan (a file or a folder of page images) in a background thread and put
//! the recognised text into the selected collection. Nothing is downloaded or started
//! automatically: the program is only looked for on disk when the window opens, and only run when
//! the user presses the button. The text is OCR, not a verified transcription.

use crate::app::{CompanionApp, Event};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::Lang;
use companion_core::attach;
use companion_core::legatorocr::{self as lgr, OcrError};
use egui::{RichText, Ui};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Where to get the program (shown as text and opened only when the user asks).
pub const HOME_URL: &str = "https://github.com/chiuta/LegatorOCR";
/// Longest path accepted in the field (a longer one cannot be a real path).
const MAX_PATH_CHARS: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Tool {
    /// Not looked for yet.
    #[default]
    Unknown,
    /// The search runs in the background.
    Probing,
    Found {
        bin: PathBuf,
        version: String,
    },
    Missing,
}

#[derive(Clone)]
pub struct Running {
    pub cancel: Arc<AtomicBool>,
    pub started: Instant,
}

#[derive(Default)]
pub struct OcrState {
    pub tool: Tool,
    pub path: String,
    /// Language of the scan; the app language until the user picks one.
    pub lang: Option<Lang>,
    pub run: Option<Running>,
    pub message: String,
}

impl OcrState {
    pub fn found(&self) -> bool {
        matches!(self.tool, Tool::Found { .. })
    }
}

/// Text typed or pasted into the path field, made into a path: no control characters, no
/// surrounding quotes or spaces (a path copied from a file manager often has them), bounded.
pub fn clean_path_input(s: &str) -> Option<PathBuf> {
    let t: String = s.chars().filter(|c| !c.is_control()).collect();
    let t = t.trim();
    let t = t
        .strip_prefix('"')
        .and_then(|x| x.strip_suffix('"'))
        .or_else(|| t.strip_prefix('\'').and_then(|x| x.strip_suffix('\'')))
        .unwrap_or(t)
        .trim();
    if t.is_empty() || t.chars().count() > MAX_PATH_CHARS {
        return None;
    }
    Some(PathBuf::from(t))
}

/// Dropped files: page images go to the scan field when LegatorOCR is there (the first one);
/// everything else keeps going into the collection as before.
pub fn route_dropped(dropped: &[PathBuf], found: bool) -> (Vec<PathBuf>, Option<PathBuf>) {
    let is_image = |p: &Path| {
        lgr::is_scan_file(p)
            && !p
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
    };
    if !found {
        return (dropped.to_vec(), None);
    }
    let ocr = dropped.iter().find(|p| is_image(p)).cloned();
    let rest = dropped
        .iter()
        .filter(|p| Some(*p) != ocr.as_ref())
        .cloned()
        .collect();
    (rest, ocr)
}

/// A folder for one run's output, unique per run.
fn work_dir() -> PathBuf {
    use std::sync::atomic::AtomicU64;
    static N: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "colibri-ocr-{}-{}-{nanos}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ))
}

/// The whole job, for the background thread: read the scan, then clean up the work folder.
pub fn run_job(
    bin: &Path,
    scan: &Path,
    lang: Lang,
    cancel: &AtomicBool,
    timeout: std::time::Duration,
) -> Result<String, OcrError> {
    let work = work_dir();
    let r = lgr::read_scan_cancel(bin, scan, lgr::ocr_lang(lang), &work, timeout, cancel);
    let _ = std::fs::remove_dir_all(&work);
    r
}

impl CompanionApp {
    /// Looks for an installed LegatorOCR once, in the background (it asks the program for its
    /// version, which must never hold the window up).
    pub fn ocr_probe(&mut self) {
        if !matches!(self.docs.ocr.tool, Tool::Unknown) {
            return;
        }
        self.docs.ocr.tool = Tool::Probing;
        self.spawn(|tx| {
            let found = lgr::locate(&[]).and_then(|bin| lgr::version(&bin).map(|v| (bin, v)));
            let _ = tx.send(Event::OcrProbe(found));
        });
    }

    pub fn ocr_probed(&mut self, found: Option<(PathBuf, String)>) {
        self.docs.ocr.tool = match found {
            Some((bin, version)) => Tool::Found { bin, version },
            None => Tool::Missing,
        };
    }

    fn ocr_start(&mut self, coll: &str) {
        let Tool::Found { bin, .. } = self.docs.ocr.tool.clone() else {
            self.docs.ocr.message = self.ocr_error_text(&OcrError::NotInstalled);
            return;
        };
        if self.docs.ocr.run.is_some() {
            return;
        }
        let Some(scan) = clean_path_input(&self.docs.ocr.path) else {
            self.docs.ocr.message = self.ocr_error_text(&OcrError::NoInput);
            return;
        };
        self.docs.ocr.message.clear();
        let lang = self.docs.ocr.lang.unwrap_or(self.s.lang);
        let cancel = Arc::new(AtomicBool::new(false));
        self.docs.ocr.run = Some(Running {
            cancel: cancel.clone(),
            started: Instant::now(),
        });
        let (coll, name) = (coll.to_string(), lgr::doc_name(&scan));
        self.spawn(move |tx| {
            let result = run_job(&bin, &scan, lang, &cancel, lgr::DEFAULT_TIMEOUT);
            let _ = tx.send(Event::OcrDone { coll, name, result });
        });
    }

    pub fn ocr_done(&mut self, coll: String, name: String, result: Result<String, OcrError>) {
        self.docs.ocr.run = None;
        match result {
            Ok(text) => {
                let ex = attach::limit(text);
                if ex.truncated {
                    self.docs.ocr.message = self
                        .t(
                            "Textul era foarte lung: am păstrat doar începutul.",
                            "The text was very long: only the beginning was kept.",
                        )
                        .to_string();
                }
                // the same road as every other file; it counts one file as read
                self.docs.busy += 1;
                self.docs_read(coll, name, Ok(ex));
            }
            Err(OcrError::Cancelled) => {
                self.docs.ocr.message = self.ocr_error_text(&OcrError::Cancelled);
            }
            Err(e) => self.docs.ocr.message = self.ocr_error_text(&e),
        }
    }

    /// What went wrong, in plain words in the current language.
    pub fn ocr_error_text(&self, e: &OcrError) -> String {
        match e {
            OcrError::NotInstalled => self
                .t(
                    "LegatorOCR nu este instalat sau nu l-am găsit.",
                    "LegatorOCR is not installed, or I could not find it.",
                )
                .to_string(),
            OcrError::NoInput => self
                .t(
                    "Nu am găsit nicio imagine sau niciun PDF acolo. Dă-mi o imagine, un PDF sau un dosar cu pagini scanate.",
                    "I found no page image or PDF there. Give me an image, a PDF or a folder of scanned pages.",
                )
                .to_string(),
            OcrError::NoText => self
                .t(
                    "LegatorOCR a terminat, dar n-a găsit text (pagini goale sau limbă fără model de recunoaștere).",
                    "LegatorOCR finished but found no text (blank pages, or no recognition data for this language).",
                )
                .to_string(),
            OcrError::Cancelled => self.t("Am oprit citirea.", "Reading was stopped.").to_string(),
            OcrError::Failed(m) => format!(
                "{} {}",
                self.t(
                    "LegatorOCR nu a putut citi scanarea. Poate lipsește modelul pentru limba aleasă (vezi «legatorocr --doctor»). Mesajul programului:",
                    "LegatorOCR could not read the scan. The data for the chosen language may be missing (see “legatorocr --doctor”). The program said:",
                ),
                one_line(m)
            ),
            OcrError::Run(m) if m.contains("too long") => self
                .t(
                    "LegatorOCR a depășit timpul îngăduit și l-am oprit.",
                    "LegatorOCR took longer than allowed, so I stopped it.",
                )
                .to_string(),
            OcrError::Run(m) => format!(
                "{} {}",
                self.t(
                    "Nu am putut porni LegatorOCR:",
                    "I could not start LegatorOCR:",
                ),
                one_line(m)
            ),
        }
    }

    /// The scanned-files part of the Documents window. Without the program it is only a hint.
    pub fn ocr_section(&mut self, ui: &mut Ui, coll: &str) {
        let p = self.p;
        match self.docs.ocr.tool.clone() {
            Tool::Unknown | Tool::Probing => {}
            Tool::Missing => {
                egui::CollapsingHeader::new(
                    RichText::new(self.t("Fișiere scanate (LegatorOCR)", "Scanned files (LegatorOCR)"))
                        .semibold(),
                )
                .default_open(true)
                .show(ui, |ui| {
                    muted(
                        ui,
                        &p,
                        self.t(
                            "Pentru scanări și poze de pagini îți trebuie programul LegatorOCR (gratuit, licență MIT), care nu e instalat aici. Nu descarc și nu pornesc nimic singură: îl instalezi tu de la adresa de mai jos, apoi apeși „Caută din nou”.",
                            "To read scans and photos of pages you need the program LegatorOCR (free, MIT licence), which is not installed here. I never download or start anything by myself: you install it from the address below, then press “Look again”.",
                        ),
                    );
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(HOME_URL).monospace());
                        if secondary_button(ui, self.t("Caută din nou", "Look again")).clicked() {
                            self.docs.ocr.tool = Tool::Unknown;
                            self.ocr_probe();
                        }
                    });
                });
            }
            Tool::Found { version, .. } => self.ocr_found(ui, coll, &version),
        }
    }

    fn ocr_found(&mut self, ui: &mut Ui, coll: &str, version: &str) {
        let p = self.p;
        let hint = self.t(
            "Calea către o scanare (imagine, PDF) sau un dosar cu pagini",
            "Path to a scan (image, PDF) or a folder of pages",
        );
        egui::CollapsingHeader::new(
            RichText::new(self.t("Fișiere scanate (LegatorOCR)", "Scanned files (LegatorOCR)"))
                .semibold(),
        )
        .default_open(true)
        .show(ui, |ui| {
            muted(
                ui,
                &p,
                self.t(
                    "LegatorOCR citește scanarea pe acest calculator, iar textul recunoscut intră în colecția aleasă, sub numele fișierului urmat de „(OCR)”. E recunoaștere optică, nu o transcriere verificată: pot fi greșeli. Citațiile arată scanarea, nu pagina. Pe o carte întreagă poate dura minute.",
                    "LegatorOCR reads the scan on this computer and the recognised text goes into the chosen collection, under the file name followed by “(OCR)”. It is optical recognition, not a checked transcription: there can be mistakes. Citations name the scan, not the page. A whole book can take minutes.",
                ),
            );
            muted(ui, &p, version);
            let running = self.docs.ocr.run.clone();
            ui.add_enabled_ui(running.is_none(), |ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.docs.ocr.path)
                        .hint_text(hint)
                        .char_limit(MAX_PATH_CHARS)
                        .desired_width(f32::INFINITY),
                );
                ui.horizontal(|ui| {
                    ui.label(self.t("Limba scanării:", "Language of the scan:"));
                    let cur = self.docs.ocr.lang.unwrap_or(self.s.lang);
                    egui::ComboBox::from_id_salt("ocr-lang")
                        .selected_text(cur.native_name())
                        .show_ui(ui, |ui| {
                            for l in Lang::ALL {
                                if ui.selectable_label(l == cur, l.native_name()).clicked() {
                                    self.docs.ocr.lang = Some(l);
                                }
                            }
                        });
                });
            });
            muted(
                ui,
                &p,
                self.t(
                    "Limba trebuie să aibă modelul ei de recunoaștere instalat în LegatorOCR; «legatorocr --doctor» le arată pe cele disponibile.",
                    "The language needs its recognition data installed for LegatorOCR; “legatorocr --doctor” lists the ones available.",
                ),
            );
            match running {
                None => {
                    if primary_button(ui, &p, self.t("Citește scanarea", "Read the scan")).clicked() {
                        self.ocr_start(coll);
                    }
                }
                Some(r) => {
                    ui.horizontal(|ui| {
                        spinner(ui);
                        ui.label(
                            self.t(
                                "LegatorOCR citește… {} s (nu dă un procent; poate dura minute)",
                                "LegatorOCR is reading… {} s (it gives no percentage; this can take minutes)",
                            )
                            .replacen("{}", &r.started.elapsed().as_secs().to_string(), 1),
                        );
                        if secondary_button(ui, self.t("Oprește", "Stop")).clicked() {
                            r.cancel.store(true, Ordering::SeqCst);
                        }
                    });
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(500));
                }
            }
            if !self.docs.ocr.message.is_empty() {
                ui.label(RichText::new(&self.docs.ocr.message).color(p.bad));
            }
        });
    }
}

/// A program's message on one short line (it may be long or hold line breaks).
fn one_line(m: &str) -> String {
    let s: String = m
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.chars().count() > 300 {
        let cut: String = s.chars().take(300).collect();
        format!("{cut}…")
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use companion_core::rag::Library;
    use std::time::Duration;

    fn temp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("cc-ocrui-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Tests never write into the real data folder of the person running them.
    fn sandbox() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let d = temp("home");
            // SAFETY: set once, before the tests of this module read it, and only ever to this value.
            unsafe { std::env::set_var("COLIBRI_COMPANION_HOME", d) };
        });
    }

    fn headless_app() -> CompanionApp {
        sandbox();
        CompanionApp::new(
            egui::Context::default(),
            companion_core::store::Settings::default(),
            Some("http://127.0.0.1:1".into()),
            false,
        )
    }

    #[cfg(unix)]
    fn fake_bin(dir: &Path, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join("legatorocr");
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    /// Understands `--version` and `--txt FILE`; `$BODY` is what it writes.
    #[cfg(unix)]
    fn fake_writing(dir: &Path, write: &str) -> PathBuf {
        fake_bin(
            dir,
            &format!(
                r#"if [ "$1" = "--version" ]; then echo "legatorocr 0.0-test"; exit 0; fi
while [ $# -gt 0 ]; do if [ "$1" = "--txt" ]; then shift; out="$1"; fi; shift; done
{write}"#
            ),
        )
    }

    const NEVER: Duration = Duration::from_secs(60);

    #[test]
    fn typed_paths_are_cleaned_and_bounded() {
        assert_eq!(
            clean_path_input("  /a/b c/é.png \n"),
            Some("/a/b c/é.png".into())
        );
        assert_eq!(
            clean_path_input("\"/a b/c.pdf\""),
            Some("/a b/c.pdf".into())
        );
        assert_eq!(clean_path_input("'x y.png'"), Some("x y.png".into()));
        assert_eq!(
            clean_path_input(r"C:\Users\Ana Pop\Scanări\pagina 1.png"),
            Some(r"C:\Users\Ana Pop\Scanări\pagina 1.png".into())
        );
        assert_eq!(clean_path_input("a\0b\x1b[31m"), Some("ab[31m".into()));
        assert_eq!(clean_path_input("   "), None);
        assert_eq!(clean_path_input("\"\""), None);
        assert_eq!(clean_path_input(&"x".repeat(MAX_PATH_CHARS + 1)), None);
    }

    #[test]
    fn dropped_images_go_to_the_scan_field_only_when_the_program_is_there() {
        let files: Vec<PathBuf> = ["a.png", "b.txt", "c.PDF", "d.JPG"]
            .iter()
            .map(PathBuf::from)
            .collect();
        let (docs, ocr) = route_dropped(&files, true);
        assert_eq!(ocr, Some("a.png".into()));
        assert_eq!(
            docs,
            vec![PathBuf::from("b.txt"), "c.PDF".into(), "d.JPG".into()]
        );
        let (docs, ocr) = route_dropped(&files, false);
        assert_eq!((docs.len(), ocr), (4, None));
    }

    #[test]
    fn program_messages_become_one_short_line() {
        let long = format!("a\nb\x07c\r\n{}", "z".repeat(5000));
        let s = one_line(&long);
        assert!(s.starts_with("a b c z"));
        assert!(s.chars().count() <= 301);
        assert!(!s.contains('\n'));
    }

    #[test]
    fn every_error_has_a_text_in_every_language() {
        let mut app = headless_app();
        let errors = [
            OcrError::NotInstalled,
            OcrError::NoInput,
            OcrError::NoText,
            OcrError::Cancelled,
            OcrError::Failed("boom\nline two".into()),
            OcrError::Run("the command took too long".into()),
            OcrError::Run("No such file".into()),
        ];
        for lang in Lang::ALL {
            app.s.lang = lang;
            for e in &errors {
                let t = app.ocr_error_text(e);
                assert!(t.len() > 8 && !t.contains('\n'), "{lang:?} {e:?}: {t}");
            }
        }
        app.s.lang = Lang::En;
        assert!(
            app.ocr_error_text(&OcrError::Run("the command took too long".into()))
                .contains("longer than allowed")
        );
    }

    fn texts_of(shapes: &[egui::epaint::ClippedShape]) -> String {
        fn walk(s: &egui::epaint::Shape, out: &mut String) {
            match s {
                egui::epaint::Shape::Text(t) => {
                    out.push_str(t.galley.text());
                    out.push('\n');
                }
                egui::epaint::Shape::Vec(v) => v.iter().for_each(|x| walk(x, out)),
                _ => {}
            }
        }
        let mut out = String::new();
        shapes.iter().for_each(|c| walk(&c.shape, &mut out));
        out
    }

    /// One headless frame sequence of the section; returns every text drawn in the last frame.
    fn render(app: &mut CompanionApp, width: f32) -> String {
        let ctx = app.ctx.clone();
        let mut last = String::new();
        for i in 0..4 {
            let raw = egui::RawInput {
                time: Some(f64::from(i) * 0.1),
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width, 900.0),
                )),
                ..Default::default()
            };
            let mut out = ctx.run_ui(raw, |ui| app.ocr_section(ui, "c1"));
            out.textures_delta.clear();
            last = texts_of(&out.shapes);
        }
        last
    }

    fn found() -> Tool {
        Tool::Found {
            bin: "/x/legatorocr".into(),
            version: "legatorocr 0.0-test".into(),
        }
    }

    /// Every state, language and width, headless: no panic, drawing alone never starts a run.
    #[test]
    fn the_section_renders_in_every_state() {
        let mut app = headless_app();
        for lang in Lang::ALL {
            app.s.lang = lang;
            for st in [Tool::Unknown, Tool::Probing, Tool::Missing, found()] {
                for running in [false, true] {
                    for width in [320.0_f32, 800.0] {
                        app.docs.ocr.tool = st.clone();
                        app.docs.ocr.run = running.then(|| Running {
                            cancel: Arc::new(AtomicBool::new(false)),
                            started: Instant::now(),
                        });
                        app.docs.ocr.message = "Something went wrong".into();
                        let _ = render(&mut app, width);
                        assert_eq!(app.docs.ocr.run.is_some(), running);
                    }
                }
            }
        }
    }

    #[test]
    fn not_found_shows_the_address_and_no_run_button_found_shows_the_controls() {
        let mut app = headless_app();
        app.s.lang = Lang::En;
        app.docs.ocr.tool = Tool::Missing;
        let t = render(&mut app, 800.0);
        assert!(t.contains("Scanned files (LegatorOCR)"), "{t}");
        assert!(t.contains("github.com/chiuta/LegatorOCR"), "{t}");
        assert!(t.contains("Look again"), "{t}");
        assert!(!t.contains("Read the scan"), "{t}");
        app.docs.ocr.tool = found();
        let t = render(&mut app, 800.0);
        assert!(t.contains("Read the scan"), "{t}");
        assert!(t.contains("Language of the scan"), "{t}");
        assert!(t.contains("legatorocr 0.0-test"), "{t}");
        assert!(!t.contains("github.com/chiuta"), "{t}");
        app.docs.ocr.run = Some(Running {
            cancel: Arc::new(AtomicBool::new(false)),
            started: Instant::now(),
        });
        let t = render(&mut app, 800.0);
        assert!(t.contains("Stop") && !t.contains("Read the scan"), "{t}");
        // still searching: nothing is drawn yet
        app.docs.ocr.run = None;
        app.docs.ocr.tool = Tool::Probing;
        assert!(!render(&mut app, 800.0).contains("Scanned files"));
        // another language is really translated
        app.docs.ocr.tool = Tool::Missing;
        app.s.lang = Lang::De;
        let t = render(&mut app, 800.0);
        assert!(
            !t.contains("Scanned files") && t.contains("LegatorOCR"),
            "{t}"
        );
    }

    #[test]
    fn the_hint_texts_are_honest() {
        // the address is the project's public page, never a download link
        assert_eq!(HOME_URL, "https://github.com/chiuta/LegatorOCR");
    }

    #[cfg(unix)]
    #[test]
    fn happy_path_with_spaces_and_unicode_in_the_path() {
        let d = temp("happy");
        let bin = fake_writing(
            &d,
            "printf 'The harbour office opened at nine.\\n\\nFees were paid in silver coins.\\n' > \"$out\"",
        );
        let dir = d.join("Scanări vechi (1)");
        std::fs::create_dir_all(&dir).unwrap();
        let scan = dir.join("pagină -1.png");
        std::fs::write(&scan, b"not an image").unwrap();
        let text = run_job(&bin, &scan, Lang::En, &AtomicBool::new(false), NEVER).unwrap();
        assert!(text.contains("silver coins"));
        // a folder works too
        let text = run_job(&bin, &dir, Lang::Hu, &AtomicBool::new(false), NEVER).unwrap();
        assert!(text.contains("harbour"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[cfg(unix)]
    #[test]
    fn the_text_lands_in_the_selected_collection_through_the_docs_flow() {
        let d = temp("flow");
        let bin = fake_writing(
            &d,
            "printf 'The harbour office opened at nine.\\n\\nFees were paid in silver coins.\\n' > \"$out\"",
        );
        let scan = d.join("ledger.png");
        std::fs::write(&scan, b"x").unwrap();
        let mut app = headless_app();
        app.docs.lib = Library::default();
        let id = app.docs.lib.create("Scans", 1).unwrap();
        app.docs.loaded = true;
        app.docs.ocr.tool = Tool::Found {
            bin: bin.clone(),
            version: "legatorocr 0.0-test".into(),
        };
        app.docs.ocr.path = format!("  \"{}\" ", scan.display());
        app.ocr_start(&id);
        assert!(app.docs.ocr.run.is_some(), "started in the background");
        // a second press while it runs does nothing
        app.ocr_start(&id);
        // the result arrives as an event, never by blocking this thread
        let t = Instant::now();
        while app.docs.ocr.run.is_some() {
            app.tick();
            assert!(t.elapsed() < Duration::from_secs(30), "no result");
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(app.docs.ocr.run.is_none());
        assert!(app.docs.ocr.message.is_empty(), "{}", app.docs.ocr.message);
        let col = app.docs.lib.get(&id).unwrap();
        assert_eq!(col.docs[0].name, "ledger.png (OCR)");
        assert!(!col.chunks.is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[cfg(unix)]
    #[test]
    fn failures_come_back_as_plain_errors() {
        let d = temp("fails");
        let scan = d.join("a.png");
        std::fs::write(&scan, b"x").unwrap();
        let go = |bin: &Path, scan: &Path, t: Duration| {
            run_job(bin, scan, Lang::En, &AtomicBool::new(false), t)
        };
        let bad = fake_bin(&d, "echo 'no OCR data for eng' 1>&2; exit 2");
        assert_eq!(
            go(&bad, &scan, NEVER),
            Err(OcrError::Failed("no OCR data for eng".into()))
        );
        let empty = fake_bin(&d, "exit 0");
        assert_eq!(go(&empty, &scan, NEVER), Err(OcrError::NoText));
        assert_eq!(
            go(&empty, &d.join("none.png"), NEVER),
            Err(OcrError::NoInput)
        );
        assert_eq!(
            go(&empty, Path::new(r"C:\Users\x\scan 1.png"), NEVER),
            Err(OcrError::NoInput)
        );
        assert!(matches!(
            go(&d.join("absent"), &scan, NEVER),
            Err(OcrError::Run(_))
        ));
        let t = Instant::now();
        let slow = fake_bin(&d, "sleep 30");
        let r = go(&slow, &scan, Duration::from_millis(400));
        assert!(
            matches!(&r, Err(OcrError::Run(m)) if m.contains("too long")),
            "{r:?}"
        );
        assert!(t.elapsed() < Duration::from_secs(15));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[cfg(unix)]
    #[test]
    fn stop_ends_the_program_quickly() {
        let d = temp("stop");
        let scan = d.join("a.png");
        std::fs::write(&scan, b"x").unwrap();
        let slow = fake_bin(&d, "sleep 60");
        let cancel = Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        let t = Instant::now();
        let h = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            c2.store(true, Ordering::SeqCst);
        });
        let r = run_job(&slow, &scan, Lang::En, &cancel, NEVER);
        h.join().unwrap();
        assert_eq!(r, Err(OcrError::Cancelled));
        assert!(t.elapsed() < Duration::from_secs(10));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[cfg(unix)]
    #[test]
    fn hostile_output_cannot_hang_or_crash() {
        let d = temp("hostile");
        let scan = d.join("a.png");
        std::fs::write(&scan, b"x").unwrap();
        // not UTF-8, with a NUL and a lone surrogate-like byte run
        let bin = fake_writing(
            &d,
            "printf 'ok text \\377\\376\\000 more words\\n' > \"$out\"",
        );
        let t = run_job(&bin, &scan, Lang::En, &AtomicBool::new(false), NEVER).unwrap();
        assert!(t.contains("ok text") && t.contains('\u{FFFD}'));
        // a flood on both streams and 40 MB of text with no spaces
        let bin = fake_writing(
            &d,
            "head -c 80000000 /dev/zero | tr '\\0' 'x' 1>&2 &\nhead -c 80000000 /dev/zero | tr '\\0' 'y'\nhead -c 40000000 /dev/zero | tr '\\0' 'a' > \"$out\"\nwait",
        );
        let start = Instant::now();
        let t = run_job(&bin, &scan, Lang::En, &AtomicBool::new(false), NEVER).unwrap();
        assert!(t.len() as u64 <= lgr::MAX_TEXT_BYTES);
        assert!(start.elapsed() < Duration::from_secs(60));
        // and the app takes it without freezing: it is cut, not refused, and the library copes
        let mut app = headless_app();
        app.docs.lib = Library::default();
        let id = app.docs.lib.create("Scans", 1).unwrap();
        app.docs.loaded = true;
        let start = Instant::now();
        app.ocr_done(id.clone(), "huge.png (OCR)".into(), Ok(t));
        assert!(start.elapsed() < Duration::from_secs(60));
        assert!(app.docs.ocr.run.is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_failed_run_is_shown_not_added() {
        let mut app = headless_app();
        app.s.lang = Lang::En;
        app.docs.lib = Library::default();
        let id = app.docs.lib.create("Scans", 1).unwrap();
        app.docs.loaded = true;
        app.docs.ocr.run = Some(Running {
            cancel: Arc::new(AtomicBool::new(false)),
            started: Instant::now(),
        });
        app.ocr_done(id.clone(), "a.png (OCR)".into(), Err(OcrError::NoText));
        assert!(app.docs.ocr.run.is_none());
        assert!(app.docs.ocr.message.contains("no text"));
        assert!(app.docs.lib.get(&id).unwrap().docs.is_empty());
        // empty text that slipped through is reported by the library flow
        app.ocr_done(id.clone(), "b.png (OCR)".into(), Ok("   ".into()));
        assert!(app.docs.message.contains("b.png"));
    }

    #[test]
    fn pressing_read_without_the_program_or_path_only_explains() {
        let mut app = headless_app();
        app.s.lang = Lang::En;
        app.docs.ocr.tool = Tool::Missing;
        app.ocr_start("c");
        assert!(app.docs.ocr.message.contains("not installed"));
        assert!(app.docs.ocr.run.is_none());
        app.docs.ocr.tool = Tool::Found {
            bin: "/nope/legatorocr".into(),
            version: "legatorocr 1".into(),
        };
        app.docs.ocr.path = "   ".into();
        app.ocr_start("c");
        assert!(app.docs.ocr.message.contains("no page image"));
        assert!(app.docs.ocr.run.is_none());
    }
}
