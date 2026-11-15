//! Application state, background work and the flows between screens.
//!
//! The UI thread never blocks: everything slow (probing Python, calling `coli`,
//! cloning, installing, streaming a chat answer) runs on a worker thread and
//! reports back through one channel of [`Event`]s.

use crate::theme::{self, Palette};
use companion_core::chat::{
    ChatClient, ChatError, ChatMessage, ChatRequest, ErrorKind, Health, ServerFacts, StreamEvent,
    StreamSummary,
};
use companion_core::coli::{self, Coli, Hardware, ModelInfo, StatusReport};
use companion_core::http::CancelToken;
use companion_core::locate::{self, PythonCmd};
use companion_core::{Conversation, ConversationStore, Lang, Settings, StoredMessage, paths, proc};
use egui_commonmark::CommonMarkCache;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub const DEFAULT_SERVER: &str = "http://127.0.0.1:8000";

pub enum Event {
    Python(Option<PythonCmd>),
    Root(Option<PathBuf>),
    Catalog(Result<Vec<ModelInfo>, String>),
    Hardware(Result<Hardware, String>),
    Status(Result<StatusReport, String>),
    Health(Health),
    ModelIds(Vec<String>),
    CloneLine(String),
    CloneDone(Result<PathBuf, String>),
    InstallSpawned(Result<(), String>),
    InstallEnded(Option<i32>),
    ServerAction {
        what: ServerAction,
        result: Result<(), String>,
    },
    Logs(String),
    ChatDelta {
        turn: u64,
        ev: StreamEvent,
    },
    ChatDone {
        turn: u64,
        result: Result<StreamSummary, ChatError>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServerAction {
    Start,
    Stop,
}

#[derive(Clone, Debug, Default)]
pub enum Loadable<T> {
    #[default]
    Idle,
    Loading,
    Ready(T),
    Failed(String),
}

impl<T> Loadable<T> {
    pub fn ready(&self) -> Option<&T> {
        match self {
            Loadable::Ready(t) => Some(t),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Probe<T> {
    Pending,
    Found(T),
    Missing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Detecting,
    NeedPython,
    NeedColibri,
    ChooseModel,
    Installing,
    InstallProblem,
    StartServer,
    Loading,
    Chat,
}

pub struct Generation {
    pub id: u64,
    pub cancel: CancelToken,
    pub started: Instant,
}

pub struct ChatState {
    pub conv: Conversation,
    pub list: Vec<Conversation>,
    pub input: String,
    pub generating: Option<Generation>,
    pub counter: u64,
    pub error: Option<ChatError>,
    pub cache: CommonMarkCache,
    /// Measured height of each message (and what it was measured for), so off-screen ones are skipped.
    pub heights: Vec<Option<(HeightKey, f32)>>,
    pub heights_for: String,
    pub focus_input: bool,
    pub confirm_delete: Option<String>,
    /// Sidebar search text, and the conversation being renamed (id + new name).
    pub search: String,
    pub focus_search: bool,
    pub renaming: Option<(String, String)>,
}

/// What a message's height depends on; when it changes the message is drawn (and measured) again.
#[derive(Clone, Copy, PartialEq)]
pub struct HeightKey {
    pub len: usize,
    pub reasoning: usize,
    pub width: u32,
    pub flags: u8,
}

pub struct PollCfg {
    pub coli: Option<Coli>,
    pub base: String,
    pub api_key: String,
    pub fast: bool,
}

pub struct Toast {
    pub text: String,
    pub until: Instant,
}

pub struct CompanionApp {
    pub perf_log: bool,
    pub ctx: egui::Context,
    pub s: Settings,
    pub p: Palette,
    pub tx: Sender<Event>,
    rx: Receiver<Event>,
    poll: Arc<Mutex<PollCfg>>,
    last_poll_sync: (Option<PathBuf>, String, String, bool),
    applied_look: Option<(companion_core::Theme, u32)>,

    // environment
    pub python: Probe<PythonCmd>,
    pub root_searched: bool,
    pub coli: Option<Coli>,
    /// A colibri folder that was found before Python was (the two probes race).
    found_root: Option<PathBuf>,
    pub root_input: String,
    pub root_msg: Option<String>,
    pub git_available: Option<bool>,
    pub clone_lines: Vec<String>,
    pub cloning: bool,

    // engine
    pub catalog: Loadable<Vec<ModelInfo>>,
    pub hardware: Loadable<Hardware>,
    pub status: Option<StatusReport>,
    pub status_error: Option<String>,
    pub health: Health,
    pub model_id: Option<String>,
    pub override_base: Option<String>,

    // flows
    pub selected_model: Option<String>,
    pub license_ok: bool,
    pub show_unfit: bool,
    pub show_model_picker: bool,
    pub installing_since: Option<Instant>,
    pub install_spawning: bool,
    pub install_log: String,
    install_log_read: Option<Instant>,
    pub install_exit: Option<i32>,
    pub busy: Option<String>,
    pub starting_since: Option<Instant>,
    pub auto_started: bool,
    pub browse_history: bool,
    pub demo: bool,

    // windows
    pub show_settings: bool,
    pub show_about: bool,
    pub logs_view: Option<String>,

    pub chat: ChatState,
    pub store: ConversationStore,
    pub toasts: Vec<Toast>,
}

impl CompanionApp {
    pub fn new(ctx: egui::Context, mut settings: Settings, demo_base: Option<String>) -> Self {
        let (tx, rx) = channel();
        let poll = Arc::new(Mutex::new(PollCfg {
            coli: None,
            base: DEFAULT_SERVER.into(),
            api_key: String::new(),
            fast: false,
        }));
        let store = ConversationStore::new();
        let demo = demo_base.is_some();
        if demo {
            settings.colibri_root = None;
        }
        let mut app = Self {
            perf_log: std::env::var_os("COLIBRI_PERF").is_some(),
            ctx: ctx.clone(),
            p: theme::palette(settings.theme),
            root_input: settings
                .colibri_root
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            s: settings,
            tx,
            rx,
            poll,
            last_poll_sync: (None, String::new(), String::new(), false),
            applied_look: None,
            python: Probe::Pending,
            root_searched: false,
            coli: None,
            found_root: None,
            root_msg: None,
            git_available: None,
            clone_lines: Vec::new(),
            cloning: false,
            catalog: Loadable::Idle,
            hardware: Loadable::Idle,
            status: None,
            status_error: None,
            health: Health::Down,
            model_id: None,
            override_base: demo_base,
            selected_model: None,
            license_ok: false,
            show_unfit: false,
            show_model_picker: false,
            installing_since: None,
            install_spawning: false,
            install_log: String::new(),
            install_log_read: None,
            install_exit: None,
            busy: None,
            starting_since: None,
            auto_started: false,
            browse_history: false,
            demo,
            show_settings: false,
            show_about: false,
            logs_view: None,
            chat: ChatState {
                conv: Conversation::new(),
                list: store.summaries(),
                input: String::new(),
                generating: None,
                counter: 0,
                error: None,
                cache: CommonMarkCache::default(),
                heights: Vec::new(),
                heights_for: String::new(),
                focus_input: true,
                confirm_delete: None,
                search: String::new(),
                focus_search: false,
                renaming: None,
            },
            store,
            toasts: Vec::new(),
        };
        app.spawn_poller();
        if !demo {
            app.detect_environment();
        }
        app
    }

    // ------------------------------------------------------------ helpers

    /// Romanian and English sit next to the code; the other six languages are looked up
    /// by the English text (see i18n.rs) and fall back to English.
    pub fn t(&self, ro: &'static str, en: &'static str) -> &'static str {
        match self.s.lang {
            Lang::Ro => ro,
            Lang::En => en,
            other => crate::i18n::lookup(other, en).unwrap_or(en),
        }
    }

    pub fn is_ro(&self) -> bool {
        self.s.lang == Lang::Ro
    }

    pub fn toast(&mut self, text: impl Into<String>) {
        self.toasts.push(Toast {
            text: text.into(),
            until: Instant::now() + Duration::from_secs(4),
        });
    }

    pub fn spawn<F>(&self, work: F)
    where
        F: FnOnce(&Sender<Event>) + Send + 'static,
    {
        let (tx, ctx) = (self.tx.clone(), self.ctx.clone());
        thread::spawn(move || {
            work(&tx);
            ctx.request_repaint();
        });
    }

    /// Where the OpenAI-compatible server is right now.
    pub fn base(&self) -> String {
        if let Some(b) = &self.override_base {
            return b.clone();
        }
        let custom = self.s.server_url.trim().trim_end_matches('/') != DEFAULT_SERVER;
        if !custom {
            if let Some(u) = self
                .status
                .as_ref()
                .and_then(|s| s.server.as_ref())
                .map(|s| s.urls.anthropic_base_url.clone())
                .filter(|u| !u.is_empty())
            {
                return u;
            }
        }
        self.s.server_url.clone()
    }

    pub fn client(&self) -> ChatClient {
        ChatClient::new(&self.base(), Some(self.s.api_key.clone()))
    }

    pub fn facts(&self) -> Option<&ServerFacts> {
        match &self.health {
            Health::Ready(f) => Some(f),
            _ => None,
        }
    }

    pub fn server_ready(&self) -> bool {
        matches!(self.health, Health::Ready(_))
    }

    // ------------------------------------------------------------ environment

    pub fn detect_environment(&mut self) {
        self.python = Probe::Pending;
        self.root_searched = false;
        let hint = self.s.colibri_root.clone();
        self.spawn(|tx| {
            let _ = tx.send(Event::Python(locate::find_python()));
        });
        self.spawn(move |tx| {
            let _ = tx.send(Event::Root(locate::find_colibri(hint.as_deref())));
        });
        self.spawn(|tx| {
            let _ = tx.send(Event::CloneLine(
                String::from("\u{0}git:")
                    + if proc::program_exists("git") {
                        "yes"
                    } else {
                        "no"
                    },
            ));
        });
    }

    fn try_assemble_coli(&mut self, root: Option<PathBuf>) {
        let python = match &self.python {
            Probe::Found(p) => Some(p.clone()),
            _ => None,
        };
        if let Some(r) = &root {
            self.found_root = Some(r.clone());
        }
        let root = root.or_else(|| self.found_root.clone());
        if let (Some(py), Some(root)) = (python, root) {
            if self.coli.as_ref().map(|c| &c.root) != Some(&root) {
                self.s.colibri_root = Some(root.clone());
                let _ = self.s.save();
                self.coli = Some(Coli::new(root, py));
                self.refresh_engine_info();
            }
        }
    }

    pub fn use_root_from_text(&mut self, text: &str) {
        let path = PathBuf::from(text.trim().trim_matches('"'));
        match locate::normalize_root(&path) {
            Some(root) => {
                self.root_msg = None;
                self.root_searched = true;
                self.try_assemble_coli(Some(root));
            }
            None => {
                self.root_msg = Some(
                    self.t(
                        "Folderul acesta nu pare a fi colibri (lipsește c/coli). Alege folderul care conține «c», «docs» și «README.md».",
                        "That folder doesn't look like colibri (c/coli is missing). Pick the folder that contains «c», «docs» and «README.md».",
                    )
                    .into(),
                );
            }
        }
    }

    pub fn refresh_engine_info(&mut self) {
        let Some(coli) = self.coli.clone() else {
            return;
        };
        self.catalog = Loadable::Loading;
        self.hardware = Loadable::Loading;
        let dir = self.s.models_dir.clone();
        let c2 = coli.clone();
        self.spawn(move |tx| {
            let _ = tx.send(Event::Catalog(coli.list_models().map(|mut m| {
                coli::sort_for_display(&mut m);
                m
            })));
        });
        self.spawn(move |tx| {
            let _ = tx.send(Event::Hardware(c2.hardware(dir.as_deref())));
        });
    }

    pub fn clone_colibri(&mut self) {
        if self.cloning {
            return;
        }
        self.cloning = true;
        self.clone_lines.clear();
        let dest = locate::default_clone_dir();
        self.spawn(move |tx| {
            use std::io::Read;
            use std::process::{Command, Stdio};
            let mut cmd = Command::new("git");
            cmd.args([
                "clone",
                "--depth",
                "1",
                "--progress",
                locate::COLIBRI_GIT_URL,
            ])
            .arg(&dest);
            proc::hide_window(&mut cmd);
            cmd.stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());
            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => return drop(tx.send(Event::CloneDone(Err(e.to_string())))),
            };
            let mut err = child.stderr.take().unwrap();
            let (mut buf, mut line) = ([0u8; 512], Vec::new());
            while let Ok(n) = err.read(&mut buf) {
                if n == 0 {
                    break;
                }
                for &b in &buf[..n] {
                    if b == b'\n' || b == b'\r' {
                        if !line.is_empty() {
                            let _ = tx.send(Event::CloneLine(
                                String::from_utf8_lossy(&line).into_owned(),
                            ));
                            line.clear();
                        }
                    } else {
                        line.push(b);
                    }
                }
            }
            let ok = child.wait().map(|s| s.success()).unwrap_or(false);
            let _ = tx.send(Event::CloneDone(if ok {
                Ok(dest)
            } else {
                Err("git clone failed".into())
            }));
        });
    }

    // ------------------------------------------------------------ phases

    pub fn phase(&self) -> Phase {
        if self.server_ready() || self.browse_history {
            return Phase::Chat;
        }
        let Some(_) = &self.coli else {
            return match (&self.python, self.root_searched) {
                (Probe::Missing, _) => Phase::NeedPython,
                (Probe::Pending, _) | (_, false) => Phase::Detecting,
                _ => Phase::NeedColibri,
            };
        };
        let status = self.status.as_ref();
        if self.install_spawning || status.is_some_and(StatusReport::install_running) {
            return Phase::Installing;
        }
        if self.show_model_picker {
            return Phase::ChooseModel;
        }
        if status.is_some_and(StatusReport::setup_ready) {
            let loading = matches!(self.health, Health::Loading(_))
                || status.is_some_and(|s| s.server_state() == "loading")
                || self.starting_since.is_some();
            return if loading {
                Phase::Loading
            } else {
                Phase::StartServer
            };
        }
        let broken = status
            .and_then(|s| s.install.as_ref())
            .is_some_and(|i| matches!(i.phase.as_str(), "error" | "interrupted"));
        if broken {
            return Phase::InstallProblem;
        }
        Phase::ChooseModel
    }

    // ------------------------------------------------------------ install and server

    pub fn start_install(&mut self, model: &ModelInfo) {
        let Some(coli) = self.coli.clone() else {
            return;
        };
        if !model.license_note.is_empty() || !model.repo.is_empty() {
            let key = model.id.clone();
            if !self.s.accepted_licenses.contains(&key) {
                self.s.accepted_licenses.push(key);
                let _ = self.s.save();
            }
        }
        self.install_spawning = true;
        self.installing_since = Some(Instant::now());
        self.install_exit = None;
        self.install_log.clear();
        self.show_model_picker = false;
        let (id, dir, log) = (
            model.id.clone(),
            self.s.models_dir.clone(),
            paths::logs_dir().join("install.log"),
        );
        self.spawn(
            move |tx| match coli.spawn_install(&id, dir.as_deref(), &log) {
                Ok(mut child) => {
                    let _ = tx.send(Event::InstallSpawned(Ok(())));
                    let code = child.wait().ok().and_then(|s| s.code());
                    let _ = tx.send(Event::InstallEnded(code));
                }
                Err(e) => {
                    let _ = tx.send(Event::InstallSpawned(Err(e.to_string())));
                }
            },
        );
    }

    pub fn pause_install(&mut self) {
        let pid = self
            .status
            .as_ref()
            .and_then(|s| s.install.as_ref())
            .and_then(|i| i.pid);
        if let (Some(coli), Some(pid)) = (&self.coli, pid) {
            let _ = coli.pause_install(pid);
            self.toast(self.t(
                "Descărcarea a fost oprită. O poți relua oricând.",
                "The download was paused. You can resume it any time.",
            ));
        }
    }

    pub fn start_server(&mut self) {
        let Some(coli) = self.coli.clone() else {
            return;
        };
        if self.starting_since.is_some() {
            return;
        }
        self.starting_since = Some(Instant::now());
        self.busy = Some(self.t("Pornesc serverul…", "Starting the server…").into());
        self.spawn(move |tx| {
            let result = coli.start_server().map(|_| ());
            let _ = tx.send(Event::ServerAction {
                what: ServerAction::Start,
                result,
            });
        });
    }

    pub fn stop_server(&mut self) {
        let Some(coli) = self.coli.clone() else {
            return;
        };
        self.busy = Some(self.t("Opresc serverul…", "Stopping the server…").into());
        self.stop_generation();
        self.spawn(move |tx| {
            let result = coli.stop_server().map(|_| ());
            let _ = tx.send(Event::ServerAction {
                what: ServerAction::Stop,
                result,
            });
        });
    }

    pub fn fetch_logs(&mut self, install: bool) {
        let Some(coli) = self.coli.clone() else {
            return;
        };
        self.spawn(move |tx| {
            let _ = tx.send(Event::Logs(coli.logs(install, 120)));
        });
    }

    fn spawn_poller(&mut self) {
        let (cfg, tx, ctx) = (self.poll.clone(), self.tx.clone(), self.ctx.clone());
        thread::spawn(move || {
            let mut tick = 0u64;
            let mut last_ids_for: Option<String> = None;
            let mut last_health: Option<String> = None;
            let mut last_status: Option<String> = None;
            loop {
                let (coli, base, key, fast) = {
                    let c = cfg.lock().unwrap();
                    (c.coli.clone(), c.base.clone(), c.api_key.clone(), c.fast)
                };
                let client = ChatClient::new(&base, Some(key));
                let health = client.health();
                if matches!(health, Health::Ready(_)) && last_ids_for.as_deref() != Some(&base) {
                    if let Ok(ids) = client.model_ids() {
                        last_ids_for = Some(base.clone());
                        let _ = tx.send(Event::ModelIds(ids));
                    }
                }
                if !matches!(health, Health::Ready(_)) {
                    last_ids_for = None;
                }
                // Wake the interface only when something it shows has changed: a quiet app uses no CPU.
                let mut changed = false;
                let health_dbg = format!("{health:?}");
                if last_health.as_deref() != Some(health_dbg.as_str()) {
                    last_health = Some(health_dbg);
                    changed = true;
                }
                if tx.send(Event::Health(health)).is_err() {
                    return;
                }
                // `coli status` starts a Python process, so it is asked for rarely when nothing is going on.
                let every = if fast { 1 } else { 10 };
                if let (Some(coli), true) = (coli, tick % every == 0) {
                    let status = coli.status();
                    let status_dbg = format!("{status:?}");
                    if last_status.as_deref() != Some(status_dbg.as_str()) {
                        last_status = Some(status_dbg);
                        changed = true;
                    }
                    if tx.send(Event::Status(status)).is_err() {
                        return;
                    }
                }
                if changed {
                    ctx.request_repaint();
                }
                tick += 1;
                thread::sleep(Duration::from_millis(if fast { 1200 } else { 3000 }));
            }
        });
    }

    fn sync_poller(&mut self) {
        let fast = matches!(self.phase(), Phase::Installing | Phase::Loading)
            || self.starting_since.is_some();
        let key = (
            self.coli.as_ref().map(|c| c.root.clone()),
            self.base(),
            self.s.api_key.clone(),
            fast,
        );
        if key != self.last_poll_sync {
            let mut c = self.poll.lock().unwrap();
            c.coli = self.coli.clone();
            c.base = key.1.clone();
            c.api_key = key.2.clone();
            c.fast = fast;
            drop(c);
            self.last_poll_sync = key;
        }
    }

    pub fn refresh_install_log(&mut self) {
        if self
            .install_log_read
            .is_some_and(|t| t.elapsed() < Duration::from_millis(900))
        {
            return;
        }
        self.install_log_read = Some(Instant::now());
        if let Ok(text) = std::fs::read_to_string(paths::logs_dir().join("install.log")) {
            let tail: Vec<&str> = text.lines().rev().take(40).collect();
            self.install_log = tail.into_iter().rev().collect::<Vec<_>>().join("\n");
        }
    }

    // ------------------------------------------------------------ chat

    pub fn new_conversation(&mut self) {
        self.stop_generation();
        self.persist_conversation();
        self.chat.conv = Conversation::new();
        self.chat.error = None;
        self.chat.focus_input = true;
    }

    pub fn open_conversation(&mut self, id: &str) {
        if self.chat.conv.id == id {
            return;
        }
        self.stop_generation();
        self.persist_conversation();
        if let Some(c) = self.store.load(id) {
            self.chat.conv = c;
            self.chat.error = None;
        }
    }

    pub fn delete_conversation(&mut self, id: &str) {
        let _ = self.store.delete(id);
        if self.chat.conv.id == id {
            self.stop_generation();
            self.chat.conv = Conversation::new();
        }
        self.chat.list = self.store.summaries();
    }

    pub fn persist_conversation(&mut self) {
        if self.chat.conv.messages.iter().any(|m| m.role == "user") {
            self.chat.conv.updated = companion_core::store::now_secs();
            if let Err(e) = self.store.save(&self.chat.conv) {
                self.toast(format!(
                    "{}: {e}",
                    self.t(
                        "Nu am putut salva conversația",
                        "Could not save the conversation"
                    )
                ));
            }
            self.chat.list = self.store.summaries();
        }
    }

    pub fn send_message(&mut self) {
        let text = self.chat.input.trim().to_string();
        if text.is_empty() || self.chat.generating.is_some() || !self.server_ready() {
            return;
        }
        self.chat.input.clear();
        self.chat.error = None;
        self.chat.conv.messages.push(StoredMessage {
            role: "user".into(),
            content: text,
            ..Default::default()
        });
        self.chat.conv.messages.push(StoredMessage {
            role: "assistant".into(),
            ..Default::default()
        });
        self.chat.conv.retitle();
        self.start_generation();
    }

    /// Takes the last question back into the composer, dropping it and its answer.
    pub fn edit_last_message(&mut self) {
        if self.chat.generating.is_some() {
            return;
        }
        let Some(pos) = self
            .chat
            .conv
            .messages
            .iter()
            .rposition(|m| m.role == "user")
        else {
            return;
        };
        let text = self.chat.conv.messages[pos].content.clone();
        self.chat.conv.messages.truncate(pos);
        self.chat.input = text;
        self.chat.error = None;
        self.chat.focus_input = true;
        if self.chat.conv.messages.is_empty() {
            let _ = self.store.delete(&self.chat.conv.id);
            self.chat.list = self.store.summaries();
        } else {
            self.persist_conversation();
        }
    }

    pub fn rename_conversation(&mut self, id: &str, name: &str) {
        let mut conv = if self.chat.conv.id == id {
            self.chat.conv.clone()
        } else {
            match self.store.load(id) {
                Some(c) => c,
                None => return,
            }
        };
        conv.rename(name);
        if let Err(e) = self.store.save(&conv) {
            self.toast(format!(
                "{}: {e}",
                self.t(
                    "Nu am putut salva conversația",
                    "Could not save the conversation"
                )
            ));
        }
        if self.chat.conv.id == id {
            self.chat.conv.title = conv.title;
            self.chat.conv.title_custom = conv.title_custom;
        }
        self.chat.list = self.store.summaries();
    }

    /// Appends dropped text files to the composer; says why when one can't be used.
    pub fn attach_files(&mut self, paths: &[std::path::PathBuf]) {
        for path in paths {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            match companion_core::attach::read_text_file(path) {
                Some(text) => {
                    if !self.chat.input.is_empty() && !self.chat.input.ends_with('\n') {
                        self.chat.input.push('\n');
                    }
                    self.chat
                        .input
                        .push_str(&companion_core::attach::fenced(&name, &text));
                    self.chat.focus_input = true;
                    let msg = self
                        .t("Am adăugat fișierul «{}»", "Added file «{}»")
                        .replacen("{}", &name, 1);
                    self.toast(msg);
                }
                None => {
                    let msg = self
                        .t(
                            "«{}» nu este un fișier text sau e prea mare (max. 256 KB)",
                            "«{}» is not a text file or is too large (max 256 KB)",
                        )
                        .replacen("{}", &name, 1);
                    self.toast(msg);
                }
            }
        }
    }

    pub fn export_conversation(&mut self) {
        let md = self.chat.conv.to_markdown(self.t("Eu", "Me"), "Colibri");
        match self
            .chat
            .conv
            .export_to(&companion_core::paths::downloads_dir(), &md)
        {
            Ok(path) => {
                let msg = self.t("Salvat în {}", "Saved to {}").replacen(
                    "{}",
                    &path.display().to_string(),
                    1,
                );
                self.toast(msg);
                if let Some(dir) = path.parent() {
                    companion_core::proc::open_externally(&dir.display().to_string());
                }
            }
            Err(e) => self.toast(format!(
                "{}: {e}",
                self.t("Nu am putut salva fișierul", "Could not save the file")
            )),
        }
    }

    pub fn regenerate(&mut self) {
        if self.chat.generating.is_some() || !self.server_ready() {
            return;
        }
        if self
            .chat
            .conv
            .messages
            .last()
            .is_some_and(|m| m.role == "assistant")
        {
            self.chat.conv.messages.pop();
        }
        if self
            .chat
            .conv
            .messages
            .last()
            .is_some_and(|m| m.role == "user")
        {
            self.chat.conv.messages.push(StoredMessage {
                role: "assistant".into(),
                ..Default::default()
            });
            self.chat.error = None;
            self.start_generation();
        }
    }

    pub fn start_generation(&mut self) {
        self.chat.counter += 1;
        let turn = self.chat.counter;
        let cancel = CancelToken::new();
        let mut messages = Vec::new();
        if !self.s.system_prompt.trim().is_empty() {
            messages.push(ChatMessage::new("system", self.s.system_prompt.trim()));
        }
        let n = self.chat.conv.messages.len().saturating_sub(1); // all but the empty placeholder
        messages.extend(
            self.chat.conv.messages[..n]
                .iter()
                .map(|m| ChatMessage::new(&m.role, m.content.clone())),
        );
        let req = ChatRequest {
            model: self.model_id.clone().unwrap_or_else(|| "colibri".into()),
            messages,
            temperature: self.s.style.temperature(),
            top_p: self.s.style.top_p(),
            max_tokens: self.s.max_tokens,
            thinking: self.s.thinking,
        };
        self.chat.generating = Some(Generation {
            id: turn,
            cancel: cancel.clone(),
            started: Instant::now(),
        });
        let (client, tx, ctx) = (self.client(), self.tx.clone(), self.ctx.clone());
        thread::spawn(move || {
            let result = client.stream_chat(&req, &cancel, |ev| {
                let _ = tx.send(Event::ChatDelta { turn, ev });
                ctx.request_repaint();
            });
            let _ = tx.send(Event::ChatDone { turn, result });
            ctx.request_repaint();
        });
    }

    /// Stop now: keep what has been written so far.
    pub fn stop_generation(&mut self) {
        if let Some(g) = self.chat.generating.take() {
            g.cancel.cancel();
            if let Some(last) = self
                .chat
                .conv
                .messages
                .last_mut()
                .filter(|m| m.role == "assistant")
            {
                last.stopped = true;
            }
            self.finish_assistant_message();
        }
    }

    fn finish_assistant_message(&mut self) {
        if self.chat.conv.messages.last().is_some_and(|m| {
            m.role == "assistant" && m.content.is_empty() && m.reasoning.is_empty()
        }) {
            self.chat.conv.messages.pop();
        }
        self.persist_conversation();
    }

    // ------------------------------------------------------------ events

    fn handle(&mut self, ev: Event) {
        match ev {
            Event::Python(p) => {
                self.python = p.map(Probe::Found).unwrap_or(Probe::Missing);
                self.try_assemble_coli(None);
            }
            Event::Root(r) => {
                self.root_searched = true;
                if let Some(root) = r {
                    self.root_input = root.display().to_string();
                    self.try_assemble_coli(Some(root));
                } else if let Some(h) = self.s.colibri_root.take() {
                    let _ = h; // a remembered folder that is gone: forget it
                }
            }
            Event::Catalog(r) => {
                self.catalog = r.map(Loadable::Ready).unwrap_or_else(Loadable::Failed)
            }
            Event::Hardware(r) => {
                self.hardware = r.map(Loadable::Ready).unwrap_or_else(Loadable::Failed)
            }
            Event::Status(r) => match r {
                Ok(st) => {
                    self.status_error = None;
                    if self.starting_since.is_some() && st.server_state() != "stopped" {
                        // the server is up (loading or ready): the poller takes it from here
                    }
                    self.status = Some(st);
                }
                Err(e) => self.status_error = Some(e),
            },
            Event::Health(h) => {
                if matches!(h, Health::Ready(_)) {
                    self.starting_since = None;
                    self.busy = None;
                }
                self.health = h;
            }
            Event::ModelIds(ids) => self.model_id = ids.into_iter().next(),
            Event::CloneLine(line) => {
                if let Some(v) = line.strip_prefix("\u{0}git:") {
                    self.git_available = Some(v == "yes");
                } else {
                    self.clone_lines.push(line);
                    if self.clone_lines.len() > 200 {
                        self.clone_lines.remove(0);
                    }
                }
            }
            Event::CloneDone(r) => {
                self.cloning = false;
                match r {
                    Ok(dest) => {
                        self.root_searched = true;
                        self.try_assemble_coli(Some(dest));
                    }
                    Err(e) => {
                        self.root_msg = Some(format!(
                            "{}: {e}",
                            self.t("Descărcarea nu a reușit", "The download failed")
                        ))
                    }
                }
            }
            Event::InstallSpawned(r) => {
                if let Err(e) = r {
                    self.install_spawning = false;
                    self.toast(format!(
                        "{}: {e}",
                        self.t("Nu am putut porni instalarea", "Could not start the setup")
                    ));
                }
            }
            Event::InstallEnded(code) => {
                self.install_spawning = false;
                self.install_exit = code;
                self.refresh_install_log();
                if let Some(coli) = self.coli.clone() {
                    self.spawn(move |tx| {
                        let _ = tx.send(Event::Status(coli.status()));
                    });
                }
            }
            Event::ServerAction { what, result } => {
                self.busy = None;
                if what == ServerAction::Stop {
                    self.health = Health::Down;
                    self.starting_since = None;
                    self.auto_started = true; // do not undo a deliberate stop
                }
                if let Err(e) = result {
                    self.starting_since = None;
                    self.toast(e);
                }
            }
            Event::Logs(text) => self.logs_view = Some(text),
            Event::ChatDelta { turn, ev } => {
                if self.chat.generating.as_ref().is_some_and(|g| g.id == turn) {
                    if let Some(m) = self.chat.conv.messages.last_mut() {
                        match ev {
                            StreamEvent::Content(t) => m.content.push_str(&t),
                            StreamEvent::Reasoning(t) => m.reasoning.push_str(&t),
                        }
                    }
                }
            }
            Event::ChatDone { turn, result } => {
                if self.chat.generating.as_ref().is_some_and(|g| g.id == turn) {
                    self.chat.generating = None;
                    match result {
                        Ok(summary) => {
                            if let Some(m) = self.chat.conv.messages.last_mut() {
                                m.tokens_per_second = summary.tokens_per_second();
                                m.completion_tokens = summary.completion_tokens;
                                m.stopped = summary.cancelled;
                            }
                        }
                        Err(e) => {
                            if e.kind == ErrorKind::Unreachable {
                                self.health = Health::Down;
                            }
                            self.chat.error = Some(e);
                        }
                    }
                    self.finish_assistant_message();
                }
            }
        }
    }

    // ------------------------------------------------------------ per frame

    pub fn tick(&mut self) {
        while let Ok(ev) = self.rx.try_recv() {
            self.handle(ev);
        }
        let look = (self.s.theme, (self.s.font_scale * 100.0) as u32);
        if self.applied_look != Some(look) {
            self.p = theme::palette(self.s.theme);
            theme::apply(&self.ctx, self.s.theme, self.s.font_scale);
            self.applied_look = Some(look);
        }
        self.sync_poller();
        self.toasts.retain(|t| t.until > Instant::now());

        // A server start that never became ready should not spin forever.
        if self
            .starting_since
            .is_some_and(|t| t.elapsed() > Duration::from_secs(30 * 60))
        {
            self.starting_since = None;
        }
        // Start the model on its own, once, when everything is installed.
        if self.phase() == Phase::StartServer
            && self.s.start_server_automatically
            && !self.auto_started
            && self.status.is_some()
        {
            self.auto_started = true;
            self.start_server();
        }
        if self.phase() == Phase::Installing {
            self.refresh_install_log();
        }
        // Redraw on a timer only while something is moving; otherwise egui redraws on input and on events.
        if self.chat.generating.is_some()
            || self.starting_since.is_some()
            || self.phase() == Phase::Installing
        {
            self.ctx.request_repaint_after(Duration::from_millis(250));
        }
        if let Some(next) = self.toasts.iter().map(|t| t.until).min() {
            self.ctx.request_repaint_after(
                next.saturating_duration_since(Instant::now()) + Duration::from_millis(30),
            );
        }
    }
}
