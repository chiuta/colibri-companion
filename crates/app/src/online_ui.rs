//! Online access: the switch, saved connections, browsing a repository, reading a web page,
//! and the catalogue of platforms by category and capability level.

use crate::app::{CompanionApp, Event, Probe};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::connect::{self, ConnectError, Connections, Entry, Kind, Op, Repo, Service};
use companion_core::platforms::{self, Auth, Category, Level, Status};
use egui::{Align2, RichText, vec2};

#[derive(Clone, Debug)]
pub enum OnlineWhat {
    Test,
    Repos,
    Tree(String),
    File(String),
    Issues(String),
    Page(String),
}

pub struct OnlineState {
    pub open_conn: bool,
    pub open_catalog: bool,
    pub conn: Connections,
    pub loaded: bool,
    pub sel: usize,
    pub new_name: String,
    pub new_kind: Kind,
    pub new_base: String,
    pub new_token: String,
    pub new_remember: bool,
    pub repos: Vec<Repo>,
    pub repo: String,
    pub path: String,
    pub entries: Vec<Entry>,
    pub status: String,
    pub busy: bool,
    pub page_url: String,
    pub cat: Option<Category>,
    pub max_level: Level,
    pub ready_only: bool,
}

impl Default for OnlineState {
    fn default() -> Self {
        OnlineState {
            open_conn: false,
            open_catalog: false,
            conn: Connections::default(),
            loaded: false,
            sel: 0,
            new_name: String::new(),
            new_kind: Kind::Github,
            new_base: Kind::Github.default_base().into(),
            new_token: String::new(),
            new_remember: false,
            repos: Vec::new(),
            repo: String::new(),
            path: String::new(),
            entries: Vec::new(),
            status: String::new(),
            busy: false,
            page_url: String::new(),
            cat: None,
            max_level: Level::Act,
            ready_only: false,
        }
    }
}

fn file() -> std::path::PathBuf {
    companion_core::paths::data_dir().join("connections.json")
}

fn category_name(app: &CompanionApp, c: Category) -> &'static str {
    match c {
        Category::Code => app.t("Cod", "Code"),
        Category::Docs => app.t("Documente și notițe", "Documents and notes"),
        Category::Projects => app.t("Proiecte și sarcini", "Projects and tasks"),
        Category::Chat => app.t("Mesaje și notificări", "Messages and alerts"),
        Category::Mail => app.t("Poștă și calendar", "Mail and calendar"),
        Category::Web => app.t("Web și căutare", "Web and search"),
        Category::Data => app.t("Date și fișiere", "Data and files"),
        Category::Automation => app.t(
            "Automatizări și casă inteligentă",
            "Automation and smart home",
        ),
        Category::Knowledge => app.t("Cunoaștere și știință", "Knowledge and science"),
        Category::DevOps => app.t("Operațiuni și unelte", "Operations and tools"),
    }
}

fn level_short(app: &CompanionApp, l: Level) -> &'static str {
    match l {
        Level::Read => app.t("Citire", "Read"),
        Level::Search => app.t("Căutare", "Search"),
        Level::Write => app.t("Scriere", "Write"),
        Level::Notify => app.t("Notificare", "Notify"),
        Level::Act => app.t("Acțiune", "Act"),
    }
}

impl CompanionApp {
    pub fn open_online(&mut self) {
        self.load_online();
        self.online.open_conn = true;
    }

    pub fn open_catalog(&mut self) {
        self.online.open_catalog = true;
    }

    fn load_online(&mut self) {
        if !self.online.loaded {
            self.online.conn = Connections::load(&file());
            self.online.loaded = true;
        }
    }

    fn save_online(&mut self) {
        if let Err(e) = self.online.conn.save(&file()) {
            self.toast(format!(
                "{}: {e}",
                self.t("Nu am putut salva", "Could not save")
            ));
        }
    }

    /// Sends one request in the background. Refuses while online access is off.
    fn online_go(&mut self, kind: Kind, req: connect::Request, what: OnlineWhat) {
        if !self.online.conn.internet {
            self.toast(self.t(
                "Accesul la internet e oprit. Pornește-l mai întâi.",
                "Online access is off. Turn it on first.",
            ));
            return;
        }
        let python = match &self.python {
            Probe::Found(p) => Some(p.clone()),
            _ => None,
        };
        self.online.busy = true;
        self.online.status = self
            .t("Aștept răspunsul…", "Waiting for the answer…")
            .into();
        self.spawn(move |tx| {
            let result = connect::send(python.as_ref(), &req);
            let _ = tx.send(Event::Online { what, kind, result });
        });
    }

    pub fn online_done(
        &mut self,
        what: OnlineWhat,
        kind: Kind,
        result: Result<Vec<u8>, ConnectError>,
    ) {
        self.online.busy = false;
        let ro = self.is_ro();
        let bytes = match result {
            Ok(b) => b,
            Err(e) => {
                self.online.status = connect::explain(&e, ro);
                return;
            }
        };
        self.online.status.clear();
        match what {
            OnlineWhat::Test => {
                let who = serde_json::from_slice::<serde_json::Value>(&bytes)
                    .ok()
                    .and_then(|v| {
                        ["login", "username", "name"]
                            .iter()
                            .find_map(|k| v.get(k).and_then(|x| x.as_str()).map(str::to_string))
                    })
                    .unwrap_or_default();
                self.online.status = format!("{} {who}", self.t("Conectat ca", "Connected as"));
            }
            OnlineWhat::Repos => match connect::parse_repos(kind, &bytes) {
                Ok(r) => {
                    self.online.status =
                        format!("{} {}", r.len(), self.t("depozite", "repositories"));
                    self.online.repos = r;
                }
                Err(e) => self.online.status = connect::explain(&e, ro),
            },
            OnlineWhat::Tree(path) => match connect::parse_tree(kind, &bytes) {
                Ok(t) => {
                    self.online.path = path;
                    self.online.entries = t;
                }
                Err(e) => self.online.status = connect::explain(&e, ro),
            },
            OnlineWhat::File(name) => {
                let text = String::from_utf8_lossy(&bytes).into_owned();
                if bytes.contains(&0) || text.trim().is_empty() {
                    self.online.status = self.t("Nu e un fișier text.", "Not a text file.").into();
                } else {
                    let ex = companion_core::attach::limit(text);
                    self.chat.pending_docs.push((name.clone(), ex.text));
                    self.toast(
                        self.t(
                            "Am adăugat «{}» din depozit",
                            "Added «{}» from the repository",
                        )
                        .replacen("{}", &name, 1),
                    );
                }
            }
            OnlineWhat::Issues(repo) => match connect::parse_issues(kind, &bytes) {
                Ok(i) if !i.is_empty() => {
                    self.chat.pending_docs.push((
                        format!("issues-{}.md", repo.replace('/', "-")),
                        connect::issues_text(&repo, &i),
                    ));
                    self.toast(self.t("Am adăugat problemele deschise", "Added the open issues"));
                }
                Ok(_) => {
                    self.online.status = self
                        .t("Nu sunt probleme deschise.", "No open issues.")
                        .into()
                }
                Err(e) => self.online.status = connect::explain(&e, ro),
            },
            OnlineWhat::Page(url) => match connect::page_text(&bytes) {
                Ok(t) => {
                    let ex = companion_core::attach::limit(t);
                    self.chat
                        .pending_docs
                        .push((format!("{}.txt", connect::host_of(&url)), ex.text));
                    self.toast(self.t("Am adăugat pagina web", "Added the web page"));
                }
                Err(e) => self.online.status = connect::explain(&e, ro),
            },
        }
    }

    pub fn online_windows(&mut self, ctx: &egui::Context) {
        if self.online.open_conn {
            self.connections_window(ctx);
        }
        if self.online.open_catalog {
            self.catalog_window(ctx);
        }
    }

    fn connections_window(&mut self, ctx: &egui::Context) {
        let p = self.p;
        let mut open = true;
        let mut changed = false;
        let l_keep = self.t(
            "Păstrează tokenul pe acest calculator",
            "Keep the token on this computer",
        );
        let l_allow = self.t(
            "Permite accesul la internet",
            "Allow access to the internet",
        );
        let h_name = self.t("Nume (de ex. GitHub personal)", "Name (e.g. my GitHub)");
        let h_page = self.t(
            "https://… adresa unei pagini",
            "https://… address of a page",
        );
        egui::Window::new(self.t("Conexiuni online", "Online connections"))
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_size(vec2(680.0, 560.0))
            .collapsible(false)
            .vscroll(true)
            .show(ctx, |ui| {
                if ui
                    .checkbox(
                        &mut self.online.conn.internet,
                        l_allow,
                    )
                    .changed()
                {
                    changed = true;
                }
                muted(ui, &p, self.t(
                    "Oprit implicit. Cât e oprit, programul vorbește doar cu serverul colibri de pe acest calculator. Când e pornit, trimite cereri doar când apeși tu un buton, la adresa aleasă; nu trimite conversațiile tale nicăieri.",
                    "Off by default. While off, the program only talks to the colibri server on this computer. When on, it sends requests only when you press a button, to the address you chose; it never sends your conversations anywhere.",
                ));
                if !matches!(self.python, Probe::Found(_)) {
                    ui.colored_label(p.warn, self.t("Îmi trebuie Python pentru acces la internet.", "I need Python for internet access."));
                }
                ui.add_enabled_ui(self.online.conn.internet, |ui| {
                    ui.separator();
                    ui.label(RichText::new(self.t("Citește o pagină web", "Read a web page")).strong());
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.online.page_url).hint_text(h_page).desired_width(380.0));
                        if ui.button(self.t("Citește", "Read")).clicked() {
                            let url = self.online.page_url.trim().to_string();
                            match connect::check_url(&url) {
                                Ok(()) => self.online_go(Kind::Generic, connect::page(&url, ""), OnlineWhat::Page(url)),
                                Err(e) => self.online.status = connect::explain(&e, self.is_ro()),
                            }
                        }
                    });
                    ui.separator();
                    ui.label(RichText::new(self.t("Platforme conectate", "Connected platforms")).strong());
                    let mut remove = None;
                    let services = self.online.conn.services.clone();
                    for (i, sv) in services.iter().enumerate() {
                        ui.horizontal(|ui| {
                            if ui.selectable_label(self.online.sel == i, format!("{} · {}", sv.name, sv.kind.name())).clicked() {
                                self.online.sel = i;
                            }
                            let token = if sv.token.is_empty() {
                                self.t("fără token", "no token")
                            } else if sv.remember {
                                self.t("token salvat", "token saved")
                            } else {
                                self.t("token doar în această sesiune", "token for this session only")
                            };
                            muted(ui, &p, token);
                            if ui.small_button("✕").clicked() {
                                remove = Some(i);
                            }
                        });
                    }
                    if let Some(i) = remove {
                        self.online.conn.services.remove(i);
                        self.online.sel = 0;
                        changed = true;
                    }
                    if self.online.conn.services.is_empty() {
                        muted(ui, &p, self.t("Nicio platformă încă. Adaugă una mai jos.", "No platform yet. Add one below."));
                    }
                    ui.collapsing(self.t("Adaugă o platformă", "Add a platform"), |ui| {
                        egui::ComboBox::from_id_salt("new-kind")
                            .selected_text(self.online.new_kind.name())
                            .show_ui(ui, |ui| {
                                for k in Kind::ALL {
                                    if ui.selectable_label(self.online.new_kind == k, k.name()).clicked() {
                                        self.online.new_kind = k;
                                        self.online.new_base = k.default_base().into();
                                    }
                                }
                            });
                        ui.add(egui::TextEdit::singleline(&mut self.online.new_name).hint_text(h_name).desired_width(f32::INFINITY));
                        ui.label(self.t("Adresa API (schimb-o pentru serverul tău)", "API address (change it for your own server)"));
                        ui.add(egui::TextEdit::singleline(&mut self.online.new_base).desired_width(f32::INFINITY));
                        ui.label(self.t("Token de acces (opțional pentru conținut public)", "Access token (optional for public content)"));
                        ui.add(egui::TextEdit::singleline(&mut self.online.new_token).password(true).desired_width(f32::INFINITY));
                        let tp = self.online.new_kind.token_page();
                        if !tp.is_empty() {
                            ui.horizontal(|ui| {
                                muted(ui, &p, self.t("Creează un token cu drept doar de citire aici:", "Make a read-only token here:"));
                                if ui.link(tp).clicked() {
                                    companion_core::proc::open_externally(tp);
                                }
                            });
                        }
                        ui.checkbox(
                            &mut self.online.new_remember,
                            l_keep,
                        );
                        muted(ui, &p, self.t(
                            "Dacă nu bifezi, tokenul rămâne doar în memorie până închizi programul. Dacă bifezi, e salvat necriptat într-un fișier care poate fi citit doar de contul tău. Folosește un token cu drepturi minime.",
                            "If unticked, the token stays in memory until you close the program. If ticked, it is saved unencrypted in a file only your account can read. Use a token with the fewest rights.",
                        ));
                        let ok = !self.online.new_name.trim().is_empty() && connect::check_url(&self.online.new_base).is_ok();
                        if ui.add_enabled(ok, egui::Button::new(self.t("Adaugă", "Add"))).clicked() {
                            self.online.conn.services.push(Service {
                                name: self.online.new_name.trim().to_string(),
                                kind: self.online.new_kind,
                                base: self.online.new_base.trim().to_string(),
                                token: self.online.new_token.trim().to_string(),
                                remember: self.online.new_remember,
                            });
                            self.online.sel = self.online.conn.services.len() - 1;
                            self.online.new_token.clear();
                            self.online.new_name.clear();
                            changed = true;
                        }
                    });
                    if let Some(sv) = self.online.conn.services.get(self.online.sel).cloned() {
                        ui.separator();
                        ui.label(RichText::new(format!("{} — {}", sv.name, sv.kind.name())).strong());
                        ui.horizontal_wrapped(|ui| {
                            if ui.button(self.t("Testează", "Test")).clicked() {
                                let r = connect::build(sv.kind, &sv.base, &sv.token, &Op::Whoami);
                                self.online_go(sv.kind, r, OnlineWhat::Test);
                            }
                            if sv.kind != Kind::Generic && ui.button(self.t("Arată depozitele mele", "List my repositories")).clicked() {
                                let r = connect::build(sv.kind, &sv.base, &sv.token, &Op::Repos);
                                self.online_go(sv.kind, r, OnlineWhat::Repos);
                            }
                        });
                        let repos = self.online.repos.clone();
                        if !repos.is_empty() {
                            egui::ScrollArea::vertical().id_salt("repos").max_height(110.0).show(ui, |ui| {
                                for r in &repos {
                                    let label = format!("{}{}", if r.private { "🔒 " } else { "" }, r.id);
                                    if ui.selectable_label(self.online.repo == r.id, label).on_hover_text(&r.description).clicked() {
                                        self.online.repo = r.id.clone();
                                        let req = connect::build(sv.kind, &sv.base, &sv.token, &Op::Tree { repo: &r.id, path: "" });
                                        self.online_go(sv.kind, req, OnlineWhat::Tree(String::new()));
                                    }
                                }
                            });
                        }
                        ui.horizontal(|ui| {
                            ui.label(self.t("Depozit:", "Repository:"));
                            ui.add(egui::TextEdit::singleline(&mut self.online.repo).hint_text("owner/name").desired_width(220.0));
                            if ui.button(self.t("Deschide", "Open")).clicked() && !self.online.repo.trim().is_empty() {
                                let repo = self.online.repo.trim().to_string();
                                let req = connect::build(sv.kind, &sv.base, &sv.token, &Op::Tree { repo: &repo, path: "" });
                                self.online_go(sv.kind, req, OnlineWhat::Tree(String::new()));
                            }
                            if ui.button(self.t("Probleme deschise → chat", "Open issues → chat")).clicked() && !self.online.repo.trim().is_empty() {
                                let repo = self.online.repo.trim().to_string();
                                let req = connect::build(sv.kind, &sv.base, &sv.token, &Op::Issues { repo: &repo });
                                self.online_go(sv.kind, req, OnlineWhat::Issues(repo));
                            }
                        });
                        if !self.online.entries.is_empty() {
                            ui.label(RichText::new(format!("/{}", self.online.path)).monospace());
                            if !self.online.path.is_empty() && ui.small_button("⬆ ..").clicked() {
                                let up = self.online.path.rsplit_once('/').map(|x| x.0).unwrap_or("").to_string();
                                let repo = self.online.repo.clone();
                                let req = connect::build(sv.kind, &sv.base, &sv.token, &Op::Tree { repo: &repo, path: &up });
                                self.online_go(sv.kind, req, OnlineWhat::Tree(up));
                            }
                            let entries = self.online.entries.clone();
                            egui::ScrollArea::vertical().id_salt("entries").max_height(180.0).show(ui, |ui| {
                                for e in &entries {
                                    ui.horizontal(|ui| {
                                        if e.is_dir {
                                            if ui.button(format!("📁 {}", e.name)).clicked() {
                                                let repo = self.online.repo.clone();
                                                let req = connect::build(sv.kind, &sv.base, &sv.token, &Op::Tree { repo: &repo, path: &e.path });
                                                self.online_go(sv.kind, req, OnlineWhat::Tree(e.path.clone()));
                                            }
                                        } else {
                                            ui.label(format!("📄 {}", e.name));
                                            if let Some(s) = e.size {
                                                muted(ui, &p, &format!("{} KB", s / 1024 + 1));
                                            }
                                            if ui.small_button(self.t("Adaugă în chat", "Add to chat")).clicked() {
                                                let repo = self.online.repo.clone();
                                                let req = connect::build(sv.kind, &sv.base, &sv.token, &Op::File { repo: &repo, path: &e.path });
                                                self.online_go(sv.kind, req, OnlineWhat::File(e.name.clone()));
                                            }
                                        }
                                    });
                                }
                            });
                        }
                    }
                });
                if self.online.busy {
                    crate::theme::busy(ui);
                }
                if !self.online.status.is_empty() {
                    ui.label(RichText::new(&self.online.status).color(p.accent));
                }
                ui.separator();
                if ui.button(self.t("Catalog de platforme…", "Platform catalogue…")).clicked() {
                    self.online.open_catalog = true;
                }
            });
        self.online.open_conn = open;
        if changed {
            self.save_online();
        }
    }

    fn catalog_window(&mut self, ctx: &egui::Context) {
        let p = self.p;
        let mut open = true;
        let ro = self.is_ro();
        let l_ready = self.t("Doar ce merge acum", "Only what works now");
        egui::Window::new(self.t("Catalog de platforme", "Platform catalogue"))
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_size(vec2(760.0, 560.0))
            .collapsible(false)
            .show(ctx, |ui| {
                muted(ui, &p, self.t(
                    "Unde se poate conecta un model local și cât poate face acolo, de la cel mai prudent (citire) la cel mai îndrăzneț (acțiune). «Gata» înseamnă că programul face asta acum; «prin web» merge acum printr-o adresă sau un webhook; «planificat» încă nu există. Verifică singur site-ul fiecărei platforme.",
                    "Where a local model can connect and how much it can do there, from the most careful (read) to the boldest (act). “Ready” means this program does it today; “via web” works now through an address or a webhook; “planned” does not exist yet. Check each platform's own site.",
                ));
                ui.horizontal_wrapped(|ui| {
                    if ui.selectable_label(self.online.cat.is_none(), self.t("Toate", "All")).clicked() {
                        self.online.cat = None;
                    }
                    for c in Category::ALL {
                        if ui.selectable_label(self.online.cat == Some(c), category_name(self, c)).clicked() {
                            self.online.cat = Some(c);
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(self.t("Până la nivelul:", "Up to level:"));
                    egui::ComboBox::from_id_salt("lvl")
                        .selected_text(level_short(self, self.online.max_level))
                        .show_ui(ui, |ui| {
                            for l in Level::ALL {
                                let t = level_short(self, l);
                                ui.selectable_value(&mut self.online.max_level, l, t);
                            }
                        });
                    ui.checkbox(&mut self.online.ready_only, l_ready);
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for c in Category::ALL {
                        if self.online.cat.is_some_and(|x| x != c) {
                            continue;
                        }
                        let items: Vec<_> = platforms::in_category(c)
                            .filter(|x| x.min <= self.online.max_level)
                            .filter(|x| !self.online.ready_only || x.status != Status::Planned)
                            .collect();
                        if items.is_empty() {
                            continue;
                        }
                        ui.add_space(6.0);
                        ui.label(RichText::new(category_name(self, c)).strong().size(16.0));
                        for x in items {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(RichText::new(x.name).semibold());
                                let (txt, col) = match x.status {
                                    Status::Ready => (self.t("gata", "ready"), p.ok),
                                    Status::ViaWeb => (self.t("prin web", "via web"), p.accent),
                                    Status::Planned => (self.t("planificat", "planned"), p.muted),
                                };
                                ui.label(RichText::new(txt).small().color(col));
                                let range = if x.min == x.max {
                                    level_short(self, x.min).to_string()
                                } else {
                                    format!("{} → {}", level_short(self, x.min), level_short(self, x.max))
                                };
                                ui.label(RichText::new(range).small().color(p.muted));
                                let auth = match x.auth {
                                    Auth::None => self.t("fără cont", "no account"),
                                    Auth::Token => self.t("token", "token"),
                                    Auth::OAuth => self.t("autentificare", "sign-in"),
                                    Auth::Local => self.t("local", "local"),
                                    Auth::WebhookUrl => self.t("adresă webhook", "webhook address"),
                                };
                                ui.label(RichText::new(auth).small().color(p.muted));
                                if !x.url.is_empty() && ui.small_button("↗").clicked() {
                                    companion_core::proc::open_externally(x.url);
                                }
                            });
                            muted(ui, &p, if ro { x.ro } else { x.en });
                        }
                    }
                    ui.add_space(8.0);
                    ui.label(RichText::new(self.t("Nivelurile", "The levels")).strong());
                    for l in Level::ALL {
                        muted(ui, &p, platforms::level_text(l, ro));
                    }
                });
            });
        self.online.open_catalog = open;
    }
}
