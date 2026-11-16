//! Model sheets, the comparison with online assistants, and the "Local programs" window that
//! hands other software on this computer the settings to use colibri.

use crate::app::CompanionApp;
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::coli::ModelInfo;
use companion_core::sheets::{self, Cap, Gpu, Know};
use egui::{Align2, RichText, Ui, vec2};

#[derive(Default)]
pub struct SheetState {
    pub compare: bool,
    pub local: bool,
}

impl CompanionApp {
    fn cap_name(&self, c: Cap) -> &'static str {
        match c {
            Cap::Chat => self.t("Conversație", "Chat"),
            Cap::Thinking => self.t("Gândire pas cu pas", "Step-by-step thinking"),
            Cap::Tools => self.t("Utilizează unelte", "Uses tools"),
            Cap::Vision => self.t("Vede imagini", "Sees pictures"),
            Cap::Code => self.t("Specializat pe cod", "Specialised in code"),
            Cap::ImageOut => self.t("Generează imagini", "Makes pictures"),
            Cap::Decisions => self.t(
                "Decizii (etichete, probabilități)",
                "Decisions (labels, probabilities)",
            ),
        }
    }

    fn know_mark(&self, k: Know) -> (&'static str, egui::Color32) {
        let p = self.p;
        match k {
            Know::Yes => ("✔", p.ok),
            Know::No => ("✘", p.bad),
            Know::Unstated => ("?", p.muted),
        }
    }

    /// The detailed sheet of one catalog model, folded under its card.
    pub fn sheet_panel(&mut self, ui: &mut Ui, m: &ModelInfo) {
        let p = self.p;
        let Some(sh) = sheets::by_id(&m.id) else {
            return;
        };
        egui::CollapsingHeader::new(RichText::new(self.t("Fișă detaliată", "Detailed sheet")).semibold())
            .id_salt(("sheet", &m.id))
            .show(ui, |ui| {
                ui.label(RichText::new(self.t("Ce știe să facă", "What it can do")).semibold());
                for c in Cap::ALL {
                    let k = sh.know(c);
                    if k == Know::Unstated && c != Cap::Vision && c != Cap::Tools {
                        continue;
                    }
                    let (mark, col) = self.know_mark(k);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(mark).color(col).strong());
                        ui.label(self.cap_name(c));
                        if k == Know::Unstated {
                            ui.label(RichText::new(self.t("(colibri nu spune)", "(colibri doesn't say)")).color(p.muted));
                        }
                    });
                }
                if let Some(ctx) = sh.context {
                    ui.label(format!("{} {ctx}", self.t("Context (tokeni):", "Context (tokens):")));
                }
                ui.add_space(4.0);
                ui.label(RichText::new(self.t("Fișiere pe care le poate primi în această aplicație", "Files it can take in this app")).semibold());
                ui.label(self.t(
                    "Text, cod, CSV, JSON, HTML, PDF, Word, Excel, PowerPoint și OpenDocument: le citesc eu și le pun în mesaj.",
                    "Text, code, CSV, JSON, HTML, PDF, Word, Excel, PowerPoint and OpenDocument: I read them and put the text in your message.",
                ));
                ui.label(if sh.accepts_images() {
                    self.t("Imagini (PNG, JPG, GIF, WebP, BMP): modelul le poate privi.", "Pictures (PNG, JPG, GIF, WebP, BMP): the model can look at them.")
                } else {
                    self.t("Imagini: nu, sau colibri nu indică faptul că le poate analiza.", "Pictures: no, or colibri doesn't say it can look at them.")
                });
                ui.add_space(4.0);
                ui.label(RichText::new(self.t("Ce poate produce", "What it can produce")).semibold());
                ui.label(if sh.makes_images() {
                    self.t("Imagini din text (prin colibri, nu în conversația de aici).", "Pictures from text (through colibri, not in the chat here).")
                } else {
                    self.t(
                        "Text. Din răspuns salvezi ca fișier orice bloc de cod sau de tabel: .py, .rs, .html, .csv, .json, .md, .svg și altele.",
                        "Text. From an answer you can save any code or table block as a file: .py, .rs, .html, .csv, .json, .md, .svg and more.",
                    )
                });
                ui.add_space(4.0);
                let gpus: Vec<&str> = sh.gpus.iter().map(|g| match g {
                    Gpu::Cuda => "CUDA",
                    Gpu::Vulkan => "Vulkan",
                    Gpu::Cpu => "CPU",
                }).collect();
                ui.label(format!("{} {}", self.t("Placă video / procesor:", "Graphics / processor:"), gpus.join(", ")));
                if let Some(l) = sh.license {
                    ui.label(format!("{} {l}", self.t("Licență:", "Licence:")));
                }
                if !sh.maker_inputs.is_empty() || sh.maker_context.is_some() || sh.maker_license.is_some() {
                    ui.add_space(4.0);
                    ui.label(RichText::new(self.t("Ce spune producătorul modelului", "What the model's maker says")).semibold());
                    if !sh.maker_inputs.is_empty() {
                        let ins: Vec<&str> = sh.maker_inputs.iter().map(|i| match i {
                            sheets::Input::Text => self.t("text", "text"),
                            sheets::Input::Image => self.t("imagini", "pictures"),
                            sheets::Input::Video => self.t("video", "video"),
                            sheets::Input::Audio => self.t("audio", "audio"),
                        }).collect();
                        ui.label(format!("{} {}", self.t("Primește:", "Takes in:"), ins.join(", ")));
                    }
                    if let Some(c) = sh.maker_context {
                        ui.label(format!("{} {c}", self.t("Context maxim (tokeni):", "Maximum context (tokens):")));
                    }
                    if let Some(a) = sh.maker_active {
                        ui.label(format!("{} {a}", self.t("Parametri activi la fiecare cuvânt:", "Parameters active per token:")));
                    }
                    if let Some(l) = sh.maker_license {
                        ui.label(format!("{} {l}", self.t("Licență (producător):", "Licence (maker):")));
                    }
                    muted(ui, &p, self.t(
                        "Atenție: modelul poate face mai mult decât utilizează colibri în prezent. Aici contează ce arată bifele de mai sus (din pagina colibri); ce spune producătorul indică doar potențialul. Date căutate online în noiembrie 2026; verificați pe pagina modelului.",
                        "Note: the model may be able to do more than colibri uses today. What counts here is the ticks above (from the colibri page); the maker's claims show only the potential. Data looked up online in November 2026; check the model's own page.",
                    ));
                }
                ui.add_space(4.0);
                if ui.button(self.t("Comparați cu AI-urile online", "Compare with online AIs")).clicked() {
                    self.sheets.compare = true;
                }
            });
    }

    pub fn sheet_windows(&mut self, ctx: &egui::Context) {
        let p = self.p;
        if self.sheets.compare {
            let mut open = true;
            egui::Window::new(self.t("Comparație: modele locale și AI-uri online", "Comparison: local models and online AIs"))
                .open(&mut open)
                .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
                .default_width(760.0)
                .default_height(560.0)
                .collapsible(false)
                .vscroll(true)
                .show(ctx, |ui| {
                    muted(ui, &p, self.t(
                        "Modelele locale: datele provin din pagina colibri; «?» înseamnă că pagina nu precizează. AI-urile online: doar funcții prezentate chiar de site-urile lor, fără scoruri sau teste; verificați pe site, deoarece se schimbă des. Diferența sigură: modelele locale rulează pe calculatorul dumneavoastră, fără cont și fără trimiterea conversației către terți.",
                        "Local models: the data comes from the colibri page; “?” means the page doesn't say. Online AIs: only features their own sites advertise, no scores or tests; check the site, they change often. The sure difference: local models run on your computer, with no account and without sending your conversation to anyone else.",
                    ));
                    ui.add_space(6.0);
                    ui.label(RichText::new(self.t("Modele locale (colibri)", "Local models (colibri)")).semibold().color(p.accent));
                    egui::Grid::new("cmp-local").striped(true).spacing([10.0, 4.0]).show(ui, |ui| {
                        ui.label("");
                        for c in [Cap::Chat, Cap::Thinking, Cap::Tools, Cap::Vision, Cap::Code, Cap::ImageOut] {
                            ui.label(RichText::new(self.cap_name(c)).small().semibold());
                        }
                        ui.label(RichText::new(self.t("Descărcare", "Download")).small().semibold());
                        ui.end_row();
                        for sh in sheets::all() {
                            ui.label(RichText::new(sh.name).semibold());
                            for c in [Cap::Chat, Cap::Thinking, Cap::Tools, Cap::Vision, Cap::Code, Cap::ImageOut] {
                                let (mark, col) = self.know_mark(sh.know(c));
                                ui.label(RichText::new(mark).color(col).strong());
                            }
                            ui.label(sh.download_gb.map_or("?".to_string(), fmt_gb));
                            ui.end_row();
                        }
                    });
                    ui.add_space(10.0);
                    ui.label(RichText::new(self.t("AI-uri online", "Online AIs")).semibold().color(p.accent));
                    egui::Grid::new("cmp-online").striped(true).spacing([10.0, 4.0]).show(ui, |ui| {
                        ui.label("");
                        for h in [
                            self.t("Fișiere", "Files in"),
                            self.t("Imagini primite", "Pictures in"),
                            self.t("Generează imagini", "Makes pictures"),
                            self.t("Căutare pe web", "Web search"),
                            self.t("Rulează local", "Runs on your PC"),
                        ] {
                            ui.label(RichText::new(h).small().semibold());
                        }
                        ui.end_row();
                        for o in sheets::ONLINE {
                            if ui.link(o.name).clicked() {
                                companion_core::proc::open_externally(o.site);
                            }
                            for b in [o.files_in, o.images_in, o.makes_images, o.web_search, false] {
                                let (mark, col) = self.know_mark(if b { Know::Yes } else { Know::No });
                                ui.label(RichText::new(mark).color(col).strong());
                            }
                            ui.end_row();
                        }
                        ui.label(RichText::new("colibri").semibold());
                        let web = Know::No;
                        for k in [Know::Yes, Know::Yes, Know::Yes, web, Know::Yes] {
                            let (mark, col) = self.know_mark(k);
                            ui.label(RichText::new(mark).color(col).strong());
                        }
                        ui.end_row();
                    });
                    muted(ui, &p, self.t(
                        "Pentru colibri: fișierele și imaginile depind de modelul ales (consultați fișa lui); imaginile se generează doar cu Qwen-Image-2.1; aplicația nu caută pe web.",
                        "For colibri: files and pictures depend on the model you pick (see its sheet); pictures are made only with Qwen-Image-2.1; this app does not search the web.",
                    ));
                });
            self.sheets.compare = open;
        }
        if self.sheets.local {
            let mut open = true;
            egui::Window::new(self.t("Programe locale", "Local programs"))
                .open(&mut open)
                .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
                .default_width(660.0)
                .default_height(560.0)
                .collapsible(false)
                .vscroll(true)
                .show(ctx, |ui| self.local_body(ui));
            self.sheets.local = open;
        }
    }

    fn local_body(&mut self, ui: &mut Ui) {
        let p = self.p;
        let base = self.base().trim_end_matches('/').to_string();
        let model = self.model_id.clone().unwrap_or_else(|| "colibri".into());
        ui.label(self.t(
            "Orice program de pe acest calculator care utilizează «OpenAI» sau «Anthropic» poate folosi colibri prin aceeași adresă locală. Nimic nu iese din calculator. Copiați setările de mai jos în programul respectiv.",
            "Any program on this computer that speaks “OpenAI” or “Anthropic” can use colibri through the same local address. Nothing leaves the computer. Copy the settings below into that program.",
        ));
        ui.add_space(6.0);
        let mut copy: Option<String> = None;
        let mut block = |ui: &mut Ui, title: &str, hint: &str, text: String| {
            ui.separator();
            ui.label(RichText::new(title).semibold().color(p.accent));
            if !hint.is_empty() {
                muted(ui, &p, hint);
            }
            ui.add(egui::Label::new(RichText::new(&text).monospace()).selectable(true));
            if ui.small_button(self.t("Copiați", "Copy")).clicked() {
                copy = Some(text);
            }
        };
        block(
            ui,
            self.t(
                "Programe care acceptă un «furnizor OpenAI» (editoare de cod, agenți, interfețe)",
                "Programs with an “OpenAI provider” setting (code editors, agents, interfaces)",
            ),
            self.t(
                "Introduceți adresa de bază, numele modelului și orice cheie nenulă (nu se verifică).",
                "Enter the base URL, the model name and any non-empty key (it isn't checked).",
            ),
            format!("Base URL: {base}/v1\nModel:    {model}\nAPI key:  local"),
        );
        block(
            ui,
            "Claude Code / Anthropic SDK",
            self.t(
                "colibri răspunde și pe adresa compatibilă Anthropic (/v1/messages).",
                "colibri also answers on the Anthropic-compatible address (/v1/messages).",
            ),
            format!(
                "ANTHROPIC_BASE_URL={base}\nANTHROPIC_AUTH_TOKEN=local\nANTHROPIC_MODEL={model}"
            ),
        );
        block(
            ui,
            "Python (openai)",
            "",
            format!(
                "from openai import OpenAI\nc = OpenAI(base_url=\"{base}/v1\", api_key=\"local\")\nr = c.chat.completions.create(model=\"{model}\", messages=[{{\"role\": \"user\", \"content\": \"Salut!\"}}])\nprint(r.choices[0].message.content)"
            ),
        );
        block(
            ui,
            "curl",
            "",
            format!(
                "curl {base}/v1/chat/completions -H \"Content-Type: application/json\" -d '{{\"model\":\"{model}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Salut!\"}}]}}'"
            ),
        );
        block(
            ui,
            "MCP",
            self.t(
                "Programele care folosesc MCP (de ex. Claude Desktop sau Claude Code) pot cere colibri să recomande, instaleze, pornească sau oprească modele: comanda este «coli mcp».",
                "Programs that use MCP (for example Claude Desktop or Claude Code) can ask colibri to recommend, install, start or stop models: the command is “coli mcp”.",
            ),
            "coli mcp".to_string(),
        );
        if let Some(t) = copy {
            ui.ctx().copy_text(t);
            self.toast(self.t("Copiat în clipboard", "Copied to the clipboard"));
        }
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            if ui
                .button(self.t(
                    "Deschideți panoul colibri în browser",
                    "Open the colibri dashboard in the browser",
                ))
                .clicked()
            {
                companion_core::proc::open_externally(&format!("{base}/"));
            }
            if ui
                .button(self.t(
                    "Deschideți folderul conversațiilor",
                    "Open the conversations folder",
                ))
                .clicked()
            {
                let dir = companion_core::paths::conversations_dir();
                let _ = std::fs::create_dir_all(&dir);
                companion_core::proc::open_externally(&dir.display().to_string());
            }
            if ui
                .button(self.t(
                    "Deschideți folderul Descărcări",
                    "Open the Downloads folder",
                ))
                .clicked()
            {
                companion_core::proc::open_externally(
                    &companion_core::paths::downloads_dir().display().to_string(),
                );
            }
        });
        muted(ui, &p, self.t(
            "Programele merg doar cât timp modelul rulează aici. Aplicația nu garantează că fiecare program terț funcționează: colibri a testat oficial SDK-urile OpenAI și Anthropic; restul sunt programe care folosesc același protocol.",
            "Programs work only while a model is running here. This app can't promise every third-party program works: colibri officially tested the OpenAI and Anthropic SDKs; the others use the same protocol.",
        ));
    }
}
