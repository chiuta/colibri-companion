//! Things inspired by AI Liber (ai-liber.chiuta.ro): a Markdown formatting toolbar for the
//! message box, plus Tips, Explore, Legal and Shortcuts windows. Everything here is local:
//! the Explore list only opens links when the person clicks them.

use crate::app::CompanionApp;
use crate::theme::Semibold;
use crate::widgets::*;
use egui::{Align2, RichText, Ui, vec2};

/// Free tools that run on your own computer, like colibri; opened only on click.
const LOCAL_TOOLS: &[(&str, &str, &str)] = &[
    ("Ollama", "https://ollama.com", "ollama"),
    ("LM Studio", "https://lmstudio.ai", "lmstudio"),
    (
        "llama.cpp",
        "https://github.com/ggml-org/llama.cpp",
        "llamacpp",
    ),
    ("Jan", "https://jan.ai", "jan"),
    ("GPT4All", "https://gpt4all.io", "gpt4all"),
    ("Open WebUI", "https://openwebui.com", "openwebui"),
];

const EMOJI: &[&str] = &[
    "🙂", "😀", "😄", "😉", "🤔", "👍", "👏", "🙏", "❤️", "🎉", "✅", "⚠️", "💡", "🔥", "⭐", "📌",
];

#[derive(Default)]
pub struct GuideState {
    pub tips: bool,
    pub explore: bool,
    pub legal: bool,
    pub shortcuts: bool,
    pub emoji: bool,
    pub table: bool,
    pub path_box: bool,
    pub path: String,
    /// The step of the welcome tour that is showing, if any.
    pub tour: Option<usize>,
}

impl CompanionApp {
    /// Replaces the selection (or inserts at the cursor) in the message box.
    fn md_wrap(&mut self, ctx: &egui::Context, id: egui::Id, before: &str, after: &str) {
        use egui::text::{CCursor, CCursorRange};
        let chars: Vec<char> = self.chat.input.chars().collect();
        let mut state = egui::TextEdit::load_state(ctx, id).unwrap_or_default();
        let (a, b) = state
            .cursor
            .char_range()
            .map(|r| {
                let (x, y) = (r.primary.index.0, r.secondary.index.0);
                (x.min(y).min(chars.len()), x.max(y).min(chars.len()))
            })
            .unwrap_or((chars.len(), chars.len()));
        let sel: String = chars[a..b].iter().collect();
        let head: String = chars[..a].iter().collect();
        let tail: String = chars[b..].iter().collect();
        self.chat.input = format!("{head}{before}{sel}{after}{tail}");
        let at = a + before.chars().count() + sel.chars().count() + after.chars().count();
        state
            .cursor
            .set_char_range(Some(CCursorRange::one(CCursor::new(
                egui::text::CharIndex(at),
            ))));
        state.store(ctx, id);
        self.chat.focus_input = true;
    }

    /// Line-start markers (headings, lists, quotes) go at the start of the current line.
    fn md_line(&mut self, ctx: &egui::Context, id: egui::Id, marker: &str) {
        let state = egui::TextEdit::load_state(ctx, id).unwrap_or_default();
        let chars: Vec<char> = self.chat.input.chars().collect();
        let at = state
            .cursor
            .char_range()
            .map_or(chars.len(), |r| r.primary.index.0.min(chars.len()));
        let start = chars[..at]
            .iter()
            .rposition(|c| *c == '\n')
            .map_or(0, |i| i + 1);
        let head: String = chars[..start].iter().collect();
        let tail: String = chars[start..].iter().collect();
        self.chat.input = format!("{head}{marker}{tail}");
        self.chat.focus_input = true;
    }

    pub fn format_toolbar(&mut self, ui: &mut Ui, id: egui::Id) {
        let p = self.p;
        let ctx = ui.ctx().clone();
        let tip = |ui: &mut Ui, text: &str, hint: &str| -> bool {
            ui.add(egui::Button::new(RichText::new(text).monospace()).small())
                .on_hover_text(hint)
                .clicked()
        };
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            if tip(ui, "B", self.t("Aldin", "Bold")) {
                self.md_wrap(&ctx, id, "**", "**");
            }
            if tip(ui, "I", self.t("Cursiv", "Italic")) {
                self.md_wrap(&ctx, id, "*", "*");
            }
            if tip(ui, "x²", self.t("Exponent", "Superscript")) {
                self.md_wrap(&ctx, id, "<sup>", "</sup>");
            }
            if tip(ui, "x₂", self.t("Indice", "Subscript")) {
                self.md_wrap(&ctx, id, "<sub>", "</sub>");
            }
            if tip(ui, "`", self.t("Cod în rând", "Inline code")) {
                self.md_wrap(&ctx, id, "`", "`");
            }
            if tip(ui, "{ }", self.t("Bloc de cod", "Code block")) {
                self.md_wrap(&ctx, id, "\n```\n", "\n```\n");
            }
            for (label, marker, hint_ro, hint_en) in [
                ("H1", "# ", "Titlu 1", "Heading 1"),
                ("H2", "## ", "Titlu 2", "Heading 2"),
                ("H3", "### ", "Titlu 3", "Heading 3"),
            ] {
                if tip(ui, label, self.t(hint_ro, hint_en)) {
                    self.md_line(&ctx, id, marker);
                }
            }
            if tip(ui, "•", self.t("Listă cu buline", "Bulleted list")) {
                self.md_line(&ctx, id, "- ");
            }
            if tip(ui, "1.", self.t("Listă numerotată", "Numbered list")) {
                self.md_line(&ctx, id, "1. ");
            }
            if tip(ui, "☑", self.t("Listă de bifat", "Checkbox list")) {
                self.md_line(&ctx, id, "- [ ] ");
            }
            if tip(ui, "❝", self.t("Citat", "Quote")) {
                self.md_line(&ctx, id, "> ");
            }
            if tip(ui, "—", self.t("Linie despărțitoare", "Dash line")) {
                self.md_wrap(&ctx, id, "\n\n---\n\n", "");
            }
            if tip(ui, "[ ]( )", self.t("Legătură", "Link")) {
                self.md_wrap(&ctx, id, "[", "](https://)");
            }
            if tip(ui, "|||", self.t("Tabel", "Table")) {
                self.guide.table = !self.guide.table;
            }
            if tip(ui, "∑", self.t("Formulă matematică", "Math")) {
                self.md_wrap(&ctx, id, "$$", "$$");
            }
            if tip(ui, ":)", self.t("Emoji", "Emoji")) {
                self.guide.emoji = !self.guide.emoji;
            }
            if tip(ui, "+ file", self.t("Fișier", "File")) {
                self.guide.path_box = !self.guide.path_box;
            }
        });
        if self.guide.emoji {
            ui.horizontal_wrapped(|ui| {
                for e in EMOJI {
                    if ui.small_button(*e).clicked() {
                        self.md_wrap(&ctx, id, e, "");
                        self.guide.emoji = false;
                    }
                }
            });
        }
        if self.guide.table {
            ui.horizontal(|ui| {
                ui.label(RichText::new(self.t("Mărime tabel:", "Table size:")).color(p.muted));
                for (c, r) in [(2, 2), (3, 3), (4, 3), (4, 5), (5, 6)] {
                    if ui.small_button(format!("{c}×{r}")).clicked() {
                        let mut t = String::from("\n");
                        t.push_str(&format!("|{}\n", " | ".repeat(c)));
                        t.push_str(&format!("|{}\n", " --- |".repeat(c)));
                        for _ in 0..r {
                            t.push_str(&format!("|{}\n", " | ".repeat(c)));
                        }
                        self.md_wrap(&ctx, id, &t, "");
                        self.guide.table = false;
                    }
                }
            });
        }
        if self.guide.path_box {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(self.t(
                        "Trage un fișier în fereastră sau scrie calea lui:",
                        "Drop a file on the window, or type its path:",
                    ))
                    .color(p.muted),
                );
                ui.add(egui::TextEdit::singleline(&mut self.guide.path).desired_width(260.0));
                if ui.button(self.t("Adaugă", "Add")).clicked()
                    && !self.guide.path.trim().is_empty()
                {
                    let path = std::path::PathBuf::from(self.guide.path.trim().trim_matches('"'));
                    self.guide.path.clear();
                    self.guide.path_box = false;
                    self.attach_files(&[path]);
                }
            });
        }
    }

    pub fn guide_windows(&mut self, ctx: &egui::Context) {
        let p = self.p;
        self.tour_window(ctx);
        if self.guide.tips {
            let mut open = true;
            egui::Window::new(self.t("Sfaturi", "Tips"))
                .open(&mut open)
                .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
                .default_width(520.0)
                .collapsible(false)
                .vscroll(true)
                .show(ctx, |ui| {
                    for (ro, en) in [
                        ("Fii precis: spune ce vrei, pentru cine și în ce formă.", "Be specific: say what you want, for whom, and in what form."),
                        ("Cere rescrieri: «mai scurt», «mai simplu», «mai cald».", "Ask for rewrites: “shorter”, “simpler”, “warmer”."),
                        ("Dă-i un rol: «ești profesor de istorie care explică unui elev».", "Give it a role: “you are a history teacher explaining to a student”."),
                        ("Verifică informațiile importante: un model local poate greși.", "Check important facts: a local model can be wrong."),
                        ("Compară modele: instalează două din fereastra Modele și încearcă-le pe aceeași întrebare.", "Compare models: install two from the Models window and try the same question on both."),
                    ] {
                        ui.label(RichText::new(format!("• {}", self.t(ro, en))).size(15.0));
                        ui.add_space(4.0);
                    }
                    ui.separator();
                    muted(ui, &p, self.t(
                        "Modelele AI pot greși. Pentru probleme medicale, juridice sau financiare, întreabă un specialist.",
                        "AI models can make mistakes. For medical, legal or financial matters, ask a specialist.",
                    ));
                });
            self.guide.tips = open;
        }
        if self.guide.explore {
            let mut open = true;
            egui::Window::new(self.t("Explorează", "Explore"))
                .open(&mut open)
                .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
                .default_width(520.0)
                .collapsible(false)
                .vscroll(true)
                .show(ctx, |ui| {
                    muted(ui, &p, self.t(
                        "Alte programe gratuite care rulează AI pe calculatorul tău, ca și colibri. Se deschid în browser doar când apeși. Verifică singur site-ul fiecăruia; lista nu e un catalog complet.",
                        "Other free programs that run AI on your own computer, like colibri. They open in your browser only when you click. Check each site yourself; this is not a complete catalogue.",
                    ));
                    ui.add_space(6.0);
                    for (name, url, _) in LOCAL_TOOLS {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(*name).semibold());
                            if ui.link(*url).clicked() {
                                companion_core::proc::open_externally(url);
                            }
                        });
                    }
                });
            self.guide.explore = open;
        }
        if self.guide.shortcuts {
            let mut open = true;
            egui::Window::new(self.t("Scurtături", "Shortcuts"))
                .open(&mut open)
                .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
                .collapsible(false)
                .show(ctx, |ui| {
                    for (keys, ro, en) in [
                        ("Enter", "trimite mesajul", "send the message"),
                        ("Shift+Enter", "rând nou", "new line"),
                        ("Ctrl/⌘+N", "conversație nouă", "new conversation"),
                        ("Ctrl/⌘+,", "setări", "settings"),
                    ] {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(keys).monospace().semibold());
                            ui.label(self.t(ro, en));
                        });
                    }
                });
            self.guide.shortcuts = open;
        }
        if self.guide.legal {
            let mut open = true;
            egui::Window::new(self.t("Confidențialitate și condiții", "Privacy and terms"))
                .open(&mut open)
                .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
                .default_width(560.0)
                .collapsible(false)
                .vscroll(true)
                .show(ctx, |ui| {
                    for (head_ro, head_en, ro, en) in [
                        ("Confidențialitate / GDPR", "Privacy / GDPR",
                         "Aplicația nu are conturi, nu trimite telemetrie și nu cheamă servere de AI din exterior. Conversațiile rămân în folderul de date al aplicației, pe calculatorul tău. Singura conexiune opțională la internet este verificarea de actualizări pe GitHub, pe care o poți opri în Setări; fereastra Resurse arată dovada conexiunilor.",
                         "The app has no accounts, sends no telemetry and calls no outside AI servers. Conversations stay in the app's data folder on your computer. The only optional internet connection is the update check on GitHub, which you can switch off in Settings; the Resources window shows the evidence of connections."),
                        ("Cookie-uri", "Cookies",
                         "Aplicația desktop nu folosește cookie-uri.",
                         "The desktop app uses no cookies."),
                        ("Condiții", "Terms",
                         "Programul este oferit așa cum este, fără garanții, sub licența Apache-2.0. Răspunsurile modelelor pot fi greșite; tu răspunzi de cum le folosești.",
                         "The program is provided as is, without warranty, under the Apache-2.0 licence. Model answers can be wrong; you are responsible for how you use them."),
                    ] {
                        ui.label(RichText::new(self.t(head_ro, head_en)).semibold().color(p.accent));
                        ui.label(self.t(ro, en));
                        ui.add_space(8.0);
                    }
                });
            self.guide.legal = open;
        }
    }
}

/// The tour: what each step says, in Romanian and English, and where it points.
pub(crate) const TOUR: &[(&str, &str, &str, &str)] = &[
    (
        "Bun venit la Colibri Companion",
        "Welcome to Colibri Companion",
        "Aici vorbești cu un model AI care rulează pe calculatorul tău. Conversațiile, notele și fișierele rămân aici: aplicația nu are conturi, nu trimite telemetrie și nu cheamă servere AI din exterior. Alege mai jos limba în care vrei să o folosești.",
        "Here you talk to an AI model that runs on your own computer. Conversations, notes and files stay here: the app has no accounts, sends no telemetry and calls no outside AI servers. Pick below the language you want to use.",
    ),
    (
        "1. Un model care să răspundă",
        "1. A model to answer you",
        "Prima dată ai nevoie de un model. Aplicația te duce pas cu pas: verifică ce are calculatorul, îți arată modele potrivite și le descarcă. Dacă ai deja un fișier de model, îl poți indica din fereastra Modele.",
        "The first time you need a model. The app guides you step by step: it checks what your computer has, shows suitable models and downloads them. If you already have a model file, you can point to it in the Models window.",
    ),
    (
        "2. Conversații cu memorie",
        "2. Conversations with memory",
        "Scrie în caseta de jos. Poți atașa fișiere și dosare, poți ramifica o conversație dintr-un mesaj anterior și poți ține minte un răspuns. Fereastra «Memorie și constelație» arată harta conversațiilor tale, legate după subiect.",
        "Write in the box at the bottom. You can attach files and folders, branch a conversation from an earlier message and keep an answer in memory. The “Memory and constellation” window shows the map of your conversations, linked by topic.",
    ),
    (
        "3. Idei și unelte",
        "3. Ideas and tools",
        "În meniul «Mai mult» găsești biblioteca de prompturi, automatizările (lanțuri de pași pe care le poți și testa), modelele creative și dueluri între modele, unde votezi orbește răspunsul mai bun.",
        "The “More” menu holds the prompt library, automations (chains of steps you can also test), creative models and model duels, where you vote blindly for the better answer.",
    ),
    (
        "4. Internetul doar când vrei tu",
        "4. The internet only when you want it",
        "Conectarea la GitHub, GitLab și alte platforme e oprită până o pornești tu. Fereastra «Resurse» arată dovada: ce conexiuni există și un jurnal al fiecărei cereri către internet, doar cu numele gazdei.",
        "Connecting to GitHub, GitLab and other platforms is off until you switch it on. The “Resources” window shows the evidence: which connections exist and a journal of every request to the internet, with host names only.",
    ),
    (
        "5. Pentru toți",
        "5. For everyone",
        "Din «Accesibilitate» alegi un profil (vedere slabă, dislexie, cititor de ecran, o singură mână și altele). Limba o schimbi oricând din bara de sus. Apasă Esc ca să închizi o fereastră. Succes!",
        "In “Accessibility” you choose a profile (low vision, dyslexia, screen reader, one hand and more). You can change the language any time from the top bar. Press Esc to close a window. Enjoy!",
    ),
];

impl CompanionApp {
    pub fn tour_window(&mut self, ctx: &egui::Context) {
        let Some(step) = self.guide.tour else {
            return;
        };
        let step = step.min(TOUR.len() - 1);
        let p = self.p;
        let mut open = true;
        let mut next: Option<usize> = None;
        let mut finish = false;
        let (l_back, l_next, l_skip, l_start) = (
            self.t("Înapoi", "Back"),
            self.t("Mai departe", "Next"),
            self.t("Sari peste", "Skip"),
            self.t("Începem", "Let's start"),
        );
        let (ro, en, tro, ten) = TOUR[step];
        egui::Window::new(self.t("Ghid de început", "Getting started"))
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_width(560.0)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.label(RichText::new(self.t(ro, en)).heading().semibold());
                ui.add_space(6.0);
                ui.label(RichText::new(self.t(tro, ten)).size(16.0));
                if step == 0 {
                    ui.add_space(8.0);
                    ui.horizontal_wrapped(|ui| {
                        for l in companion_core::store::Lang::ALL {
                            if ui
                                .selectable_label(self.s.lang == l, l.native_name())
                                .clicked()
                            {
                                self.s.lang = l;
                                let _ = self.s.save();
                            }
                        }
                    });
                }
                ui.add_space(10.0);
                muted(ui, &p, &format!("{} / {}", step + 1, TOUR.len()));
                ui.horizontal(|ui| {
                    if step > 0 && ui.button(l_back).clicked() {
                        next = Some(step - 1);
                    }
                    if step + 1 < TOUR.len() {
                        if primary_button(ui, &p, l_next).clicked() {
                            next = Some(step + 1);
                        }
                        if ui.button(l_skip).clicked() {
                            finish = true;
                        }
                    } else if primary_button(ui, &p, l_start).clicked() {
                        finish = true;
                    }
                });
            });
        if finish || !open {
            self.guide.tour = None;
            // Written now, so the tour does not come back on the next start.
            let _ = self.s.save();
        } else if let Some(n) = next {
            self.guide.tour = Some(n);
        }
    }
}
