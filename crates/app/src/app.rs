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
use companion_core::models::{self, InstalledModel};
use companion_core::net::{self, Conn, LogSummary};
use companion_core::npu;
use companion_core::resources::{Sampler, Snapshot};
use companion_core::update::{self, Release, Upstream};
use companion_core::{Conversation, ConversationStore, Lang, Settings, StoredMessage, paths, proc};
use egui_commonmark::CommonMarkCache;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// What the sampler threads leave for the interface: only the newest of each, so a window that is
/// hidden (and does not drain its channel) cannot make the queue grow.
#[derive(Default)]
pub struct Mail {
    pub health: companion_core::mailbox::Latest<Health>,
    pub status: companion_core::mailbox::Latest<Result<StatusReport, String>>,
    pub resources: companion_core::mailbox::Latest<Box<ResView>>,
}

pub const DEFAULT_SERVER: &str = "http://127.0.0.1:8000";

#[allow(clippy::large_enum_variant)]
pub enum Event {
    Python(Option<PythonCmd>),
    Root(Option<PathBuf>),
    Catalog(Result<Vec<ModelInfo>, String>),
    Hardware(Result<Hardware, String>),
    Gpus(Vec<companion_core::gpu::GpuInfo>),
    Plan {
        dir: PathBuf,
        result: Result<companion_core::coli::PlanSummary, String>,
    },
    Tuned(Result<companion_core::coli::TuneOutcome, String>),
    Captured(Result<Box<companion_core::capture::Report>, String>),
    AutoTuneProgress {
        done: usize,
        max: usize,
        best: Option<f64>,
    },
    AutoTuneDone(Result<crate::speed_tools_ui::TuneSummary, String>),
    CtxPlans {
        dir: PathBuf,
        rows: Vec<(u32, Result<companion_core::coli::PlanSummary, String>)>,
    },
    CtxApplied(Result<u32, String>),
    DiskSpeed(Result<f64, String>),
    Status(Result<StatusReport, String>),
    Health(Health),
    /// A sampler left a newer value in [`Mail`]; read it with `take`.
    Mail,
    ModelIds(Vec<String>),
    CloneLine(String),
    CloneDone(Result<PathBuf, String>),
    FetchProgress {
        done: u64,
        total: u64,
    },
    FetchUnpacking,
    Online {
        what: crate::online_ui::OnlineWhat,
        kind: companion_core::connect::Kind,
        result: Result<Vec<u8>, companion_core::connect::ConnectError>,
    },
    Picture(Result<PathBuf, String>),
    Memory(Box<(companion_core::memory::Index, companion_core::memory::Graph)>),
    McpStarted {
        name: String,
        result: Result<Box<(companion_core::mcp::Client, Vec<companion_core::mcp::Tool>)>, String>,
    },
    McpResult {
        req: companion_core::mcp::ToolRequest,
        result: Result<String, String>,
    },
    McpManual(Result<String, String>),
    DocsRead {
        coll: String,
        name: String,
        result: Result<companion_core::attach::Extracted, companion_core::attach::AttachError>,
    },
    Summary {
        conv: String,
        upto: usize,
        result: Result<String, String>,
    },
    FlowLog(String),
    FlowNotify(String),
    FlowTestProgress(usize, usize),
    FlowTestDone(String, companion_core::flow_eval::Report, usize),
    FlowApprove(String, std::sync::mpsc::Sender<bool>),
    DuelPiece {
        round: u64,
        side: usize,
        piece: String,
    },
    DuelEnd {
        round: u64,
        side: usize,
        err: Option<String>,
    },
    FlowDone {
        ok: bool,
        output: String,
    },
    Attached {
        name: String,
        result: Result<companion_core::attach::Extracted, companion_core::attach::AttachError>,
    },
    Installed(Vec<InstalledModel>),
    Resources(Box<ResView>),
    Npu(Option<String>),
    UpdateChecked {
        colibri: Option<Upstream>,
        app: Option<Release>,
        error: Option<String>,
    },
    ColibriUpdated(Result<(), String>),
    ModelDeleted(Result<String, String>),
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
    MeterSample {
        turn: u64,
        sample: companion_core::meter::Sample,
    },
}

/// One reading for the "Resources and privacy" window.
#[derive(Clone)]
pub struct ResView {
    pub snap: Snapshot,
    /// Open sockets of this app and of colibri's processes, with the name of the process.
    pub conns: Vec<(String, Conn)>,
    pub conns_error: Option<String>,
    pub applog: LogSummary,
    /// Requests that went to the internet through the helper script (host names only).
    pub egress: Vec<net::Egress>,
}

#[derive(Clone, Debug, Default)]
pub struct UpdateState {
    pub checking: bool,
    /// A newer colibri exists upstream.
    pub colibri: Option<Upstream>,
    /// A newer release of this app exists.
    pub app: Option<Release>,
    pub dismissed: bool,
    pub updating: bool,
    pub error: Option<String>,
}

/// A question put to the user inside the "My models" window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelAsk {
    Install(String),
    Delete(String),
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
    /// (message, block) whose active preview (HTML/SVG) awaits the user's confirmation.
    pub preview_ask: Option<(usize, usize)>,
    pub focus_input: bool,
    /// Pictures waiting to go with the next message: (file name, data URL, bytes).
    pub pending_images: Vec<(String, String, usize)>,
    /// Documents read and waiting to go with the next message: (file name, text).
    pub pending_docs: Vec<(String, String)>,
    /// Further parts of a big message, last one first; sent as each answer finishes.
    pub send_queue: Vec<String>,
    pub confirm_delete: Option<String>,
    /// Sidebar search text, and the conversation being renamed (id + new name).
    pub search: String,
    pub focus_search: bool,
    pub renaming: Option<(String, String)>,
    /// What the sidebar search found for the text last searched, so the files are not read again
    /// on every redraw while the text stays the same.
    pub search_cache: Option<SearchCache>,
}

/// The result of one sidebar search, and what it is valid for.
pub struct SearchCache {
    pub query: String,
    /// Fingerprint of the conversation list it was made from: a saved or deleted conversation
    /// changes it, so the result is searched again.
    pub stamp: (usize, u64),
    pub at: Instant,
    pub hits: std::collections::HashSet<String>,
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

/// The longest the screen waits to show newly streamed text (about 30 frames a second).
pub const STREAM_REPAINT: Duration = Duration::from_millis(33);

pub struct CompanionApp {
    pub perf_log: bool,
    pub ctx: egui::Context,
    pub s: Settings,
    pub p: Palette,
    pub tx: Sender<Event>,
    pub mail: Arc<Mail>,
    rx: Receiver<Event>,
    poll: Arc<Mutex<PollCfg>>,
    last_poll_sync: (Option<PathBuf>, String, String, bool),
    applied_look: Option<(companion_core::Theme, u32, companion_core::access::Access)>,

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
    /// Bytes downloaded and expected while fetching colibri's ZIP; `unpacking` once it is whole.
    pub fetch_progress: Option<(u64, u64)>,
    pub fetch_unpacking: bool,
    auto_fetch_tried: bool,

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
    pub model_filter: companion_core::coli::ModelFilter,
    pub model_query: String,
    pub show_model_picker: bool,
    pub installing_since: Option<Instant>,
    pub install_spawning: bool,
    /// "Light models" list: only those documented for tool calling.
    pub light_agents_only: bool,
    pub install_log: String,
    install_log_read: Option<Instant>,
    pub install_exit: Option<i32>,
    pub busy: Option<String>,
    pub starting_since: Option<Instant>,
    pub auto_started: bool,
    /// Start the model as soon as a switch or an added model is ready, whatever the setting says.
    pub force_autostart: bool,
    /// Set when a setup ended; cleared by the next status, so the old status never starts the old model.
    awaiting_status: bool,
    switch_after_stop: Option<String>,
    pub browse_history: bool,
    pub demo: bool,

    // windows
    pub show_settings: bool,
    pub show_about: bool,
    pub guide: crate::guide_ui::GuideState,
    pub prompts: crate::prompts_ui::PromptState,
    pub online: crate::online_ui::OnlineState,
    pub studio: crate::studio_ui::StudioState,
    pub duel: crate::duel_ui::DuelState,
    pub access: crate::access_ui::AccessState,
    pub flows: crate::flows_ui::FlowState,
    pub memory: crate::memory_ui::MemoryState,
    pub docs: crate::docs_ui::DocsState,
    pub mcp: crate::mcp_ui::McpState,
    pub sheets: crate::sheets_ui::SheetState,
    pub existing_path: String,
    pub show_models: bool,
    pub show_resources: bool,
    /// The dedicated AI chip, when there is one.
    pub npu: Option<String>,
    /// Graphics cards the operating system reports; `None` while still looking.
    pub gpus: Option<Vec<companion_core::gpu::GpuInfo>>,
    /// colibri's placement plan for the model that is set up, and the folder it was asked for.
    pub plan: Option<Result<companion_core::coli::PlanSummary, String>>,
    pub plan_asked: Option<PathBuf>,
    pub tuning: bool,
    pub tune_note: Option<String>,
    /// What colibri's planner gives at several context lengths: (tokens, plan).
    pub ctx_rows: Option<Vec<(u32, Result<companion_core::coli::PlanSummary, String>)>>,
    pub ctx_busy: bool,
    pub ctx_auto_done: bool,
    pub disk_speed: Option<Result<f64, String>>,
    pub disk_busy: bool,
    pub tools: crate::speed_tools_ui::ToolsState,
    pub upd: UpdateState,
    update_after_stop: bool,
    pub res: Option<ResView>,
    pub cpu_hist: VecDeque<f32>,
    pub ram_hist: VecDeque<f32>,
    /// Generation of the resource sampler: a thread runs only while it is still the newest.
    res_gen: Arc<AtomicU64>,
    res_targets: Arc<Mutex<Vec<(String, u32)>>>,
    pub installed: Vec<InstalledModel>,
    installed_read: Option<Instant>,
    pub model_ask: Option<ModelAsk>,
    pub logs_view: Option<String>,

    pub chat: ChatState,
    pub meter: crate::meter_ui::MeterState,
    pub store: ConversationStore,
    pub toasts: Vec<Toast>,
}

impl CompanionApp {
    pub fn new(
        ctx: egui::Context,
        mut settings: Settings,
        demo_base: Option<String>,
        first_run: bool,
    ) -> Self {
        let show_tour = first_run && demo_base.is_none();
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
            p: theme::palette_for(settings.theme, settings.access.colour_blind),
            root_input: settings
                .colibri_root
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            s: settings,
            tx,
            mail: Arc::default(),
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
            fetch_progress: None,
            fetch_unpacking: false,
            auto_fetch_tried: false,
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
            model_filter: Default::default(),
            model_query: String::new(),
            show_model_picker: false,
            installing_since: None,
            install_spawning: false,
            light_agents_only: false,
            install_log: String::new(),
            install_log_read: None,
            install_exit: None,
            busy: None,
            starting_since: None,
            auto_started: false,
            force_autostart: false,
            awaiting_status: false,
            switch_after_stop: None,
            browse_history: false,
            demo,
            show_settings: false,
            show_about: false,
            guide: crate::guide_ui::GuideState {
                // The very first start shows the short tour once; the demo never does.
                tour: show_tour.then_some(0),
                ..Default::default()
            },
            prompts: Default::default(),
            online: Default::default(),
            studio: Default::default(),
            duel: Default::default(),
            access: Default::default(),
            flows: Default::default(),
            memory: Default::default(),
            docs: Default::default(),
            mcp: Default::default(),
            sheets: Default::default(),
            existing_path: String::new(),
            show_models: false,
            show_resources: false,
            npu: None,
            gpus: None,
            plan: None,
            plan_asked: None,
            tuning: false,
            tune_note: None,
            ctx_rows: None,
            ctx_busy: false,
            ctx_auto_done: false,
            disk_speed: None,
            disk_busy: false,
            tools: Default::default(),
            upd: UpdateState::default(),
            update_after_stop: false,
            res: None,
            cpu_hist: VecDeque::new(),
            ram_hist: VecDeque::new(),
            res_gen: Arc::new(AtomicU64::new(0)),
            res_targets: Arc::new(Mutex::new(Vec::new())),
            installed: Vec::new(),
            installed_read: None,
            model_ask: None,
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
                preview_ask: None,
                focus_input: true,
                pending_images: Vec::new(),
                pending_docs: Vec::new(),
                send_queue: Vec::new(),
                confirm_delete: None,
                search: String::new(),
                focus_search: false,
                renaming: None,
                search_cache: None,
            },
            meter: crate::meter_ui::MeterState::default(),
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

    /// An error from the online connection, in the current language.
    pub fn explain(&self, e: &companion_core::connect::ConnectError) -> String {
        explain_in(self.s.lang, e)
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
                        "Acest folder nu pare a fi colibri (lipsește c/coli). Alege folderul care conține «c», «docs» și «README.md».",
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
        self.spawn(|tx| {
            let _ = tx.send(Event::Npu(npu::detect()));
        });
        self.spawn(|tx| {
            let _ = tx.send(Event::Gpus(companion_core::gpu::detect()));
        });
    }

    pub fn clone_colibri(&mut self) {
        if self.cloning {
            return;
        }
        self.cloning = true;
        self.clone_lines.clear();
        self.fetch_progress = None;
        self.fetch_unpacking = false;
        self.root_msg = None;
        let dest = locate::fresh_clone_dir();
        let git = self.git_available == Some(true);
        let python = match &self.python {
            Probe::Found(p) => Some(p.clone()),
            _ => None,
        };
        let ctx = self.ctx.clone();
        self.spawn(move |tx| {
            let mut why = String::new();
            if git {
                match clone_with_git(&dest, tx) {
                    Ok(()) => return drop(tx.send(Event::CloneDone(Ok(dest)))),
                    Err(e) => {
                        why = e;
                        let _ = std::fs::remove_dir_all(&dest); // ours: `fresh_clone_dir` never exists beforehand
                    }
                }
            }
            let result = match python {
                Some(py) => zip_with_python(&py, &dest, tx, &ctx),
                None => Err("Python was not found".to_string()),
            };
            let _ = tx.send(Event::CloneDone(match result {
                Ok(()) => Ok(dest),
                Err(e) if why.is_empty() => Err(e),
                Err(e) => Err(format!("{why}; {e}")),
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
        if self.coli.is_none() {
            return;
        }
        if !model.license_note.is_empty() || !model.repo.is_empty() {
            let key = model.id.clone();
            if !self.s.accepted_licenses.contains(&key) {
                self.s.accepted_licenses.push(key);
                let _ = self.s.save();
            }
        }
        self.begin_setup(&model.id);
    }

    /// Runs colibri's own `setup --model <id>`: downloads what is missing (nothing, for a model
    /// that is already whole) and makes it the model colibri is set up for.
    pub fn begin_setup(&mut self, id: &str) {
        let Some(coli) = self.coli.clone() else {
            return;
        };
        self.install_spawning = true;
        self.installing_since = Some(Instant::now());
        self.install_exit = None;
        self.install_log.clear();
        self.show_model_picker = false;
        self.show_models = false;
        let (id, dir, backend, log) = (
            id.to_string(),
            self.s.models_dir.clone(),
            self.s.backend.as_arg(),
            paths::logs_dir().join("install.log"),
        );
        self.spawn(
            move |tx| match coli.spawn_install(&id, dir.as_deref(), backend, &log) {
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

    /// A model the person already downloaded: a split GGUF (give the first part) is converted
    /// with `coli import`, a ready model folder is used as it is; then colibri is set up for it.
    pub fn use_existing_model(&mut self, path: std::path::PathBuf) {
        use companion_core::coli::{imported_dir, is_first_gguf, missing_gguf_parts};
        let Some(coli) = self.coli.clone() else {
            return;
        };
        if self.phase() == Phase::Installing || self.install_spawning {
            return;
        }
        if path.is_file() {
            if !is_first_gguf(&path) {
                self.toast(self.t(
                    "Alege primul fișier (cel cu «-00001-of-…» în nume) sau un fișier .gguf întreg.",
                    "Pick the first file (the one with “-00001-of-…” in its name) or a single .gguf file.",
                ));
                return;
            }
            let missing = missing_gguf_parts(&path);
            if !missing.is_empty() {
                let m = self
                    .t(
                        "Lipsesc din același folder: {}",
                        "Missing from the same folder: {}",
                    )
                    .replacen("{}", &missing.join(", "), 1);
                self.toast(m);
                return;
            }
        } else if !path.join("config.json").is_file() {
            self.toast(self.t(
                "Folderul nu pare un model colibri (lipsește config.json).",
                "That folder doesn't look like a colibri model (config.json is missing).",
            ));
            return;
        }
        self.install_spawning = true;
        self.installing_since = Some(Instant::now());
        self.install_exit = None;
        self.install_log.clear();
        self.show_models = false;
        self.force_autostart = true;
        self.auto_started = false;
        let (backend, log) = (
            self.s.backend.as_arg(),
            paths::logs_dir().join("install.log"),
        );
        self.spawn(move |tx| {
            let started = std::time::SystemTime::now() - Duration::from_secs(2);
            let mut dir = path.clone();
            if path.is_file() {
                match coli.spawn_import(&path, &log) {
                    Ok(mut child) => {
                        let _ = tx.send(Event::InstallSpawned(Ok(())));
                        let ok = child.wait().map(|s| s.success()).unwrap_or(false);
                        match imported_dir(&path, started).filter(|_| ok) {
                            Some(d) => dir = d,
                            None => {
                                let _ = tx.send(Event::InstallEnded(Some(1)));
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Event::InstallSpawned(Err(e.to_string())));
                        return;
                    }
                }
            }
            match coli.spawn_use_dir(&dir, backend, None, &log) {
                Ok(mut child) => {
                    let _ = tx.send(Event::InstallSpawned(Ok(())));
                    let code = child.wait().ok().and_then(|s| s.code());
                    let _ = tx.send(Event::InstallEnded(code));
                }
                Err(e) => {
                    let _ = tx.send(Event::InstallSpawned(Err(e.to_string())));
                }
            }
        });
    }

    /// Switch to a model that is already on this computer: stop the running one, set colibri up for
    /// the other (quick, nothing to download), start it.
    pub fn switch_model(&mut self, id: &str) {
        if self.tune_blocks_server() {
            return;
        }
        if self.coli.is_none() || self.phase() == Phase::Installing || self.install_spawning {
            return;
        }
        self.browse_history = false;
        self.show_models = false;
        self.force_autostart = true;
        self.auto_started = false;
        let running = self.server_ready()
            || matches!(self.health, Health::Loading(_))
            || self.starting_since.is_some();
        if running {
            self.switch_after_stop = Some(id.to_string());
            self.stop_server();
        } else {
            self.begin_setup(id);
        }
    }

    /// Download one more model and use it as soon as it is there. The model that runs now stays
    /// installed; it is only stopped while the new one downloads.
    pub fn install_additional(&mut self, model: &ModelInfo) {
        if self.tune_blocks_server() {
            return;
        }
        if self.phase() == Phase::Installing || self.install_spawning {
            return;
        }
        self.model_ask = None;
        self.browse_history = false;
        self.force_autostart = true;
        self.auto_started = false;
        let running = self.server_ready()
            || matches!(self.health, Health::Loading(_))
            || self.starting_since.is_some();
        if !model.license_note.is_empty() || !model.repo.is_empty() {
            let key = model.id.clone();
            if !self.s.accepted_licenses.contains(&key) {
                self.s.accepted_licenses.push(key);
                let _ = self.s.save();
            }
        }
        if running && !self.demo {
            self.switch_after_stop = Some(model.id.clone());
            self.show_models = false;
            self.stop_server();
        } else {
            self.begin_setup(&model.id);
        }
    }

    /// Opening the window starts a sampler thread; closing it (or the app) stops it, so a closed
    /// window costs nothing.
    pub fn set_resources_open(&mut self, open: bool) {
        self.show_resources = open;
        let mine = self.res_gen.fetch_add(1, Ordering::SeqCst) + 1;
        if !open {
            return; // the newer generation makes the running thread stop
        }
        let (run, targets, tx, ctx, mail) = (
            self.res_gen.clone(),
            self.res_targets.clone(),
            self.tx.clone(),
            self.ctx.clone(),
            self.mail.clone(),
        );
        let disk = self
            .s
            .models_dir
            .clone()
            .or_else(models::default_models_root)
            .filter(|p| p.exists())
            .or_else(|| Some(paths::data_dir()));
        thread::spawn(move || {
            let mut sampler = Sampler::new();
            thread::sleep(Duration::from_millis(400));
            while run.load(Ordering::SeqCst) == mine {
                let t = targets.lock().map(|t| t.clone()).unwrap_or_default();
                let snap = sampler.sample(&t, disk.as_deref());
                // which of our processes owns which socket
                let mut pids: Vec<(u32, String)> = Vec::new();
                for (label, root) in &t {
                    for pid in sampler.tree_pids(*root) {
                        if !pids.iter().any(|(p, _)| *p == pid) {
                            pids.push((pid, label.clone()));
                        }
                    }
                }
                let list: Vec<u32> = pids.iter().map(|(p, _)| *p).collect();
                let (conns, conns_error) = match net::connections_of(&list) {
                    Ok(c) => (c, None),
                    Err(e) => (Vec::new(), Some(e)),
                };
                let conns = conns
                    .into_iter()
                    .map(|c| {
                        let label = pids
                            .iter()
                            .find(|(p, _)| *p == c.pid)
                            .map(|(_, l)| l.clone())
                            .unwrap_or_default();
                        (label, c)
                    })
                    .collect();
                let view = ResView {
                    snap,
                    conns,
                    conns_error,
                    applog: net::log_summary(),
                    egress: net::egress_log(),
                };
                if mail.resources.put(Box::new(view)) && tx.send(Event::Mail).is_err() {
                    return;
                }
                ctx.request_repaint();
                thread::sleep(Duration::from_millis(1500));
            }
        });
    }

    fn sync_resource_targets(&mut self) {
        let mut t = vec![("Colibri Companion".to_string(), std::process::id())];
        if let Some(pid) = self
            .status
            .as_ref()
            .and_then(|s| s.server.as_ref())
            .and_then(|s| s.pid)
        {
            t.push((self.t("Motorul colibri", "colibri engine").to_string(), pid));
        }
        if let Some(pid) = self
            .status
            .as_ref()
            .and_then(|s| s.install.as_ref())
            .filter(|i| i.running)
            .and_then(|i| i.pid)
        {
            t.push((
                self.t("Instalare colibri", "colibri setup").to_string(),
                pid,
            ));
        }
        if let Ok(mut g) = self.res_targets.lock() {
            *g = t;
        }
    }

    /// One small request to GitHub, through Python, for colibri's latest commit and this app's
    /// latest release. `manual` ignores the once-a-day limit.
    pub fn check_updates(&mut self, manual: bool) {
        let Probe::Found(py) = &self.python else {
            return;
        };
        if self.upd.checking || self.demo {
            return;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if !manual && now.saturating_sub(self.s.last_update_check) < 24 * 3600 {
            return;
        }
        self.s.last_update_check = now;
        let _ = self.s.save();
        self.upd.checking = true;
        self.upd.error = None;
        let py = py.clone();
        let root = self.coli.as_ref().map(|c| c.root.clone());
        self.spawn(move |tx| {
            let mut error = None;
            let colibri = match update::http_get(&py, update::COLIBRI_COMMIT_API) {
                Ok(text) => update::parse_commit(&text).filter(|u| {
                    root.as_ref()
                        .and_then(|r| update::colibri_is_outdated(r, u))
                        .unwrap_or(false)
                }),
                Err(e) => {
                    error = Some(e);
                    None
                }
            };
            let app = update::http_get(&py, update::COMPANION_RELEASES_API)
                .ok()
                .and_then(|t| update::parse_latest_release(&t))
                .filter(|r| update::is_newer(&r.tag, env!("CARGO_PKG_VERSION")));
            let _ = tx.send(Event::UpdateChecked {
                colibri,
                app,
                error,
            });
        });
    }

    /// Bring colibri up to date: `git pull` for a git checkout, otherwise a fresh ZIP that
    /// replaces the folder (the old one is kept next to it, renamed). The model must be stopped.
    pub fn update_colibri(&mut self) {
        if self.tune_blocks_server() {
            return;
        }
        let Some(coli) = self.coli.clone() else {
            return;
        };
        if self.upd.updating {
            return;
        }
        if self.server_ready() || matches!(self.health, Health::Loading(_)) {
            self.update_after_stop = true;
            self.stop_server();
            return;
        }
        self.upd.updating = true;
        let git = self.git_available == Some(true) && coli.root.join(".git").exists();
        let python = match &self.python {
            Probe::Found(p) => Some(p.clone()),
            _ => None,
        };
        let ctx = self.ctx.clone();
        self.spawn(move |tx| {
            let root = coli.root.clone();
            let result = if git {
                let mut c = std::process::Command::new("git");
                c.args(["-C"]).arg(&root).args(["pull", "--ff-only"]);
                proc::hide_window(&mut c);
                match proc::run_capture(c, Duration::from_secs(300)) {
                    Ok(o) if o.status.success() => Ok(()),
                    Ok(o) => Err(String::from_utf8_lossy(&o.stderr).trim().to_string()),
                    Err(e) => Err(e.to_string()),
                }
            } else if let Some(py) = python {
                let name = root
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "colibri".into());
                let fresh = root.with_file_name(format!("{name}-update"));
                update::prepare_update_dir(&fresh)
                    .and_then(|_| zip_with_python(&py, &fresh, tx, &ctx))
                    .and_then(|_| {
                        let stamp = update::iso_now().replace([':', '-'], "");
                        let old = root.with_file_name(format!("{name}-old-{stamp}"));
                        std::fs::rename(&root, &old).map_err(|e| e.to_string())?;
                        if let Err(e) = std::fs::rename(&fresh, &root) {
                            let _ = std::fs::rename(&old, &root); // put the old one back
                            return Err(e.to_string());
                        }
                        update::write_fetched_marker(&root);
                        Ok(())
                    })
            } else {
                Err("Python was not found".into())
            };
            let _ = tx.send(Event::ColibriUpdated(result));
        });
    }

    pub fn refresh_installed(&mut self) {
        self.installed_read = Some(Instant::now());
        let active = self
            .status
            .as_ref()
            .and_then(|s| s.setup.as_ref())
            .and_then(|s| s.model_dir.clone())
            .map(PathBuf::from);
        let roots = models::roots(self.s.models_dir.as_deref(), active.as_deref());
        self.spawn(move |tx| {
            let _ = tx.send(Event::Installed(models::scan(&roots)));
        });
    }

    pub fn refresh_installed_if_stale(&mut self) {
        if self
            .installed_read
            .is_none_or(|t| t.elapsed() > Duration::from_secs(4))
        {
            self.refresh_installed();
        }
    }

    pub fn delete_model(&mut self, id: &str) {
        let Some(m) = self.installed.iter().find(|m| m.id == id).cloned() else {
            return;
        };
        let active = self
            .status
            .as_ref()
            .and_then(|s| s.setup.as_ref())
            .and_then(|s| s.model_dir.clone())
            .map(PathBuf::from);
        let roots = models::roots(self.s.models_dir.as_deref(), active.as_deref());
        self.spawn(move |tx| {
            let r = models::delete(&m.dir, &roots, active.as_deref())
                .map(|_| m.id.clone())
                .map_err(|e| e.to_string());
            let _ = tx.send(Event::ModelDeleted(r));
        });
    }

    pub fn pause_install(&mut self) {
        // Only a process the status says is running: a pid left in a state file may belong to
        // another program by now.
        let pid = self
            .status
            .as_ref()
            .and_then(|s| s.install.as_ref())
            .filter(|i| i.running)
            .and_then(|i| i.pid);
        if let (Some(coli), Some(pid)) = (&self.coli, pid) {
            let _ = coli.pause_install(pid);
            self.toast(self.t(
                "Descărcarea a fost oprită. O poți relua oricând.",
                "The download was paused. You can resume it any time.",
            ));
        }
    }

    /// The switches that turn on colibri's Metal backend (Apple Silicon) or its experimental NPU
    /// path (AMD, Windows). Both are colibri's own opt-ins, passed as it documents.
    pub fn accel_env(&self) -> Vec<(String, String)> {
        let mut env = self.accel_env_base();
        env.extend(self.tools_env());
        env
    }

    /// The switches without the recording and the searched settings (the search's baseline).
    pub fn accel_env_base(&self) -> Vec<(String, String)> {
        let mut env = Vec::new();
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) && self.s.apple_gpu {
            env.push(("COLI_METAL".to_string(), "1".to_string()));
            env.push(("K3_METAL".to_string(), "1".to_string()));
        }
        if cfg!(windows) && self.s.use_npu && self.npu.as_deref().is_some_and(npu::colibri_can_use)
        {
            env.push(("COLI_XDNA".to_string(), "1".to_string()));
        }
        if self.s.adaptive_cache {
            env.push(("COLI_POLICY".to_string(), "balanced".to_string()));
        }
        env
    }

    /// The folder of the model colibri is set up for, when there is one.
    pub fn active_model_dir(&self) -> Option<PathBuf> {
        self.status
            .as_ref()?
            .setup
            .as_ref()?
            .model_dir
            .clone()
            .map(PathBuf::from)
    }

    /// Asks colibri for its placement plan once per model folder.
    pub fn fetch_plan_once(&mut self) {
        let (Some(dir), Some(mut coli)) = (self.active_model_dir(), self.coli.clone()) else {
            return;
        };
        if self.plan_asked.as_deref() == Some(dir.as_path()) {
            return;
        }
        coli.env = self.accel_env();
        self.plan_asked = Some(dir.clone());
        self.plan = None;
        self.spawn(move |tx| {
            let result = coli.plan(&dir);
            let _ = tx.send(Event::Plan { dir, result });
        });
    }

    /// A few seconds of random reads on the drive that holds the models.
    pub fn measure_disk(&mut self) {
        // the search measures the model's speed: the drive must not be shared with another test
        if self.disk_busy || self.tools.tune.is_running() {
            return;
        }
        let dir = self
            .active_model_dir()
            .or_else(|| self.s.models_dir.clone())
            .or_else(companion_core::models::default_models_root);
        let Some(dir) = dir else { return };
        self.disk_busy = true;
        self.spawn(move |tx| {
            let r = companion_core::diskbench::random_read_speed(
                &dir,
                std::time::Duration::from_secs(4),
            );
            let _ = tx.send(Event::DiskSpeed(r));
        });
    }

    /// Asks colibri's planner what each of a few context lengths would leave for the expert cache.
    pub fn compare_contexts(&mut self) {
        let (Some(dir), Some(mut coli)) = (self.active_model_dir(), self.coli.clone()) else {
            return;
        };
        if self.ctx_busy || self.tools.tune.is_running() {
            return;
        }
        coli.env = self.accel_env();
        self.ctx_busy = true;
        self.ctx_rows = None;
        self.spawn(move |tx| {
            let rows = [4096u32, 8192, 16384, 32768, 65536]
                .into_iter()
                .map(|n| (n, coli.plan_ctx(&dir, n)))
                .collect();
            let _ = tx.send(Event::CtxPlans { dir, rows });
        });
    }

    /// Keeps `ctx` as the context colibri starts with (its own `setup --ctx`), then restarts it.
    pub fn apply_context(&mut self, ctx: u32) {
        let (Some(dir), Some(mut coli)) = (self.active_model_dir(), self.coli.clone()) else {
            return;
        };
        if self.ctx_busy || self.tuning || self.tune_blocks_server() {
            return;
        }
        coli.env = self.accel_env();
        let backend = self.s.backend.as_arg();
        self.ctx_busy = true;
        self.stop_generation();
        let log = paths::logs_dir().join("context.log");
        self.spawn(move |tx| {
            let _ = coli.stop_server();
            let result = match coli.spawn_use_dir(&dir, backend, Some(ctx), &log) {
                Ok(mut child) => match child.wait() {
                    Ok(st) if st.success() => Ok(ctx),
                    _ => Err(coli_last_log(&log)),
                },
                Err(e) => Err(e.to_string()),
            };
            let _ = coli.start_server();
            let _ = tx.send(Event::CtxApplied(result));
        });
    }

    /// Stops the server, lets colibri measure its quality-preserving settings on this model and
    /// computer (a few minutes), then starts the server again so the saved profile applies.
    pub fn run_tune(&mut self) {
        let (Some(dir), Some(mut coli)) = (self.active_model_dir(), self.coli.clone()) else {
            return;
        };
        if self.tuning || self.tune_blocks_server() {
            return;
        }
        coli.env = self.accel_env();
        self.tuning = true;
        self.tune_note = None;
        self.stop_generation();
        let log = paths::logs_dir().join("tune.log");
        self.spawn(move |tx| {
            let _ = coli.stop_server();
            let result = match coli.spawn_tune(&dir, &log) {
                Ok(mut child) => {
                    let _ = child.wait();
                    let text = std::fs::read_to_string(&log).unwrap_or_default();
                    Ok(companion_core::coli::parse_tune_log(&text))
                }
                Err(e) => Err(e.to_string()),
            };
            let _ = coli.start_server();
            let _ = tx.send(Event::Tuned(result));
        });
    }

    pub fn start_server(&mut self) {
        let Some(mut coli) = self.coli.clone() else {
            return;
        };
        if self.tune_blocks_server() {
            return;
        }
        coli.env = self.accel_env();
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
        let (cfg, tx, ctx, mail) = (
            self.poll.clone(),
            self.tx.clone(),
            self.ctx.clone(),
            self.mail.clone(),
        );
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
                if mail.health.put(health) && tx.send(Event::Mail).is_err() {
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
                    if mail.status.put(status) && tx.send(Event::Mail).is_err() {
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
        // only the end of the file: an install log can grow to many megabytes, and this runs
        // on the interface thread about once a second while an install is going
        if let Ok(text) = read_tail(&paths::logs_dir().join("install.log"), INSTALL_LOG_TAIL) {
            let tail: Vec<&str> = text.lines().rev().take(40).collect();
            self.install_log = tail.into_iter().rev().collect::<Vec<_>>().join("\n");
        }
    }

    // ------------------------------------------------------------ chat

    pub fn new_conversation(&mut self) {
        self.stop_generation();
        self.persist_conversation();
        self.chat.conv = Conversation::new();
        self.mcp.trusted.clear();
        self.mcp.rounds = 0;
        self.mcp.approval = None;
        self.chat.error = None;
        self.chat.focus_input = true;
    }

    /// Continue from message `upto` in a copy, leaving the original conversation as it was.
    pub fn branch_conversation(&mut self, upto: usize) {
        self.stop_generation();
        self.persist_conversation();
        let b = self.chat.conv.branch_at(upto, "(↳)");
        self.chat.conv = b;
        self.mcp.trusted.clear();
        self.mcp.rounds = 0;
        self.mcp.approval = None;
        self.chat.error = None;
        self.chat.focus_input = true;
        self.persist_conversation();
        self.toast(
            self.t(
                "Ramificație nouă: originalul a rămas neschimbat",
                "New branch: the original is unchanged",
            )
            .to_string(),
        );
    }

    pub fn open_conversation(&mut self, id: &str) {
        if self.chat.conv.id == id {
            return;
        }
        self.stop_generation();
        self.persist_conversation();
        if let Some(c) = self.store.load(id) {
            self.chat.conv = c;
            self.mcp.trusted.clear();
            self.mcp.rounds = 0;
            self.mcp.approval = None;
            self.chat.error = None;
        }
    }

    pub fn delete_conversation(&mut self, id: &str) {
        let _ = self.store.delete(id);
        if self.chat.conv.id == id {
            self.stop_generation();
            self.chat.conv = Conversation::new();
            self.mcp.trusted.clear();
            self.mcp.rounds = 0;
            self.mcp.approval = None;
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
        if (text.is_empty() && self.chat.pending_docs.is_empty())
            || self.chat.generating.is_some()
            || !self.server_ready()
        {
            return;
        }
        self.chat.input.clear();
        self.chat.error = None;
        self.mcp.rounds = 0;
        let images = std::mem::take(&mut self.chat.pending_images)
            .into_iter()
            .map(|(_, url, _)| url)
            .collect();
        let docs = std::mem::take(&mut self.chat.pending_docs);
        let budget = self.context_budget();
        let mut parts = companion_core::attach::plan_parts(&text, &docs, budget);
        if parts.len() > 1 {
            self.toast(
                self.t(
                    "Fișierele sunt mari: le trimit în {} părți, pe rând.",
                    "The files are big: sending them in {} parts, one after another.",
                )
                .replacen("{}", &parts.len().to_string(), 1),
            );
        }
        let first = parts.remove(0);
        parts.reverse();
        self.chat.send_queue = parts;
        self.push_user_turn(first, images);
    }

    pub fn push_user_turn(&mut self, content: String, images: Vec<String>) {
        self.chat.conv.messages.push(StoredMessage {
            role: "user".into(),
            content,
            images,
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
        let images = self.chat.conv.messages[pos].images.clone();
        self.chat.conv.messages.truncate(pos);
        self.chat.input = text;
        self.chat.pending_images = images
            .into_iter()
            .enumerate()
            .map(|(i, u)| (format!("#{}", i + 1), u.clone(), u.len() * 3 / 4))
            .collect();
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

    /// Files dropped or chosen: pictures wait beside the composer, documents are read in the
    /// background and their text lands in the message box.
    pub fn attach_files(&mut self, paths: &[std::path::PathBuf]) {
        use companion_core::attach::{self, AttachError, Kind};
        if let Some(g) = paths.iter().find(|p| {
            p.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        }) {
            self.existing_path = g.display().to_string();
            self.show_models = true;
            self.toast(self.t(
                "Fișier de model găsit. Apasă «Folosește acest model» în fereastra Modele.",
                "Model file found. Press “Use this model” in the Models window.",
            ));
            return;
        }
        let found = attach::collect(paths, 300);
        if found.files.is_empty() {
            self.toast(self.t(
                "Nu am găsit fișiere de citit",
                "No files to read were found",
            ));
            return;
        }
        if found.files.len() > 1 || found.skipped > 0 || found.too_many {
            let mut m = self.t("Citesc {} fișiere", "Reading {} files").replacen(
                "{}",
                &found.files.len().to_string(),
                1,
            );
            if found.too_many {
                m.push_str(self.t(
                    " (doar primele; sunt mai multe)",
                    " (only the first ones; there are more)",
                ));
            }
            self.toast(m);
        }
        for path in &found.files {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if attach::kind_of(path) == Kind::Image {
                if self.chat.pending_images.len() >= 3 {
                    self.toast(self.t("Cel mult 3 imagini odată", "At most 3 pictures at a time"));
                    continue;
                }
                match attach::read_image(path) {
                    Ok((mime, bytes)) => {
                        let size = bytes.len();
                        self.chat.pending_images.push((
                            name.clone(),
                            attach::data_url(mime, &bytes),
                            size,
                        ));
                        self.chat.focus_input = true;
                        self.toast(
                            self.t(
                                "Imaginea «{}» pleacă odată cu mesajul. Merge doar cu modele care văd imagini.",
                                "Picture «{}» goes with your message. It only works with models that can see pictures.",
                            )
                            .replacen("{}", &name, 1),
                        );
                    }
                    Err(e) => {
                        let m = self.attach_error(&name, &e);
                        self.toast(m);
                    }
                }
                continue;
            }
            let python = match &self.python {
                Probe::Found(p) => Some(p.clone()),
                _ => None,
            };
            let (p2, n2) = (path.clone(), name.clone());
            self.toast(
                self.t("Citesc «{}»…", "Reading «{}»…")
                    .replacen("{}", &name, 1),
            );
            self.spawn(move |tx| {
                let result = attach::extract_text(python.as_ref(), &p2);
                let _ = tx.send(Event::Attached { name: n2, result });
            });
            let _ = AttachError::Unreadable;
        }
    }

    pub fn attach_error(&self, name: &str, e: &companion_core::attach::AttachError) -> String {
        use companion_core::attach::AttachError as E;
        let why = match e {
            E::Unreadable => self
                .t("nu pot citi fișierul", "I can't read the file")
                .to_string(),
            E::TooLarge => self.t("este prea mare", "it is too large").to_string(),
            E::NotText => self
                .t(
                    "nu este text sau imagine pe care o știu citi",
                    "it is not text or a picture I can read",
                )
                .to_string(),
            E::NeedsTool(what) => self
                .t("îmi trebuie {} ca să-l citesc", "I need {} to read it")
                .replacen("{}", what, 1),
            E::Failed(m) => format!("{}: {m}", self.t("nu a mers", "it failed")),
        };
        format!("«{name}»: {why}")
    }

    fn attached(
        &mut self,
        name: String,
        result: Result<companion_core::attach::Extracted, companion_core::attach::AttachError>,
    ) {
        match result {
            Ok(ex) => {
                self.chat.pending_docs.push((name.clone(), ex.text));
                self.chat.focus_input = true;
                let msg = if ex.truncated {
                    self.t(
                        "Am adăugat «{}» (textul lung a fost tăiat)",
                        "Added «{}» (the long text was cut)",
                    )
                } else {
                    self.t("Am adăugat fișierul «{}»", "Added file «{}»")
                }
                .replacen("{}", &name, 1);
                self.toast(msg);
            }
            Err(e) => {
                let m = self.attach_error(&name, &e);
                self.toast(m);
            }
        }
    }

    /// Saves a code block from an answer into the downloads folder, never over another file.
    pub fn save_block(&mut self, b: &companion_core::blocks::Block) {
        let dir = companion_core::paths::downloads_dir();
        let _ = std::fs::create_dir_all(&dir);
        let ext = b.extension();
        let mut path = dir.join(format!("colibri.{ext}"));
        let mut n = 2;
        while path.exists() {
            path = dir.join(format!("colibri-{n}.{ext}"));
            n += 1;
        }
        match std::fs::write(&path, &b.code) {
            Ok(()) => {
                let msg = self.t("Salvat în {}", "Saved to {}").replacen(
                    "{}",
                    &path.display().to_string(),
                    1,
                );
                self.toast(msg);
                companion_core::proc::open_externally(&dir.display().to_string());
            }
            Err(e) => self.toast(format!(
                "{}: {e}",
                self.t("Nu am putut salva fișierul", "Could not save the file")
            )),
        }
    }

    /// Shows a block in the system's own viewer (browser for HTML/SVG/Markdown).
    pub fn preview_block(&mut self, b: &companion_core::blocks::Block) {
        let dir = companion_core::paths::data_dir().join("preview");
        let _ = std::fs::create_dir_all(&dir);
        // HTML and SVG are written as a page that opens with a restrictive policy (see `preview_file`).
        let (ext, text) = b.preview_file();
        let path = dir.join(format!("preview.{ext}"));
        match std::fs::write(&path, text) {
            // Scripts (.py, .sh, .js…) are refused by the opener: a preview must never run code.
            Ok(()) if companion_core::proc::open_externally(&path.display().to_string()) => {}
            Ok(()) => self.toast(
                self.t(
                    "Previzualizarea nu a putut fi deschisă",
                    "Could not open the preview",
                )
                .to_string(),
            ),
            Err(e) => self.toast(format!(
                "{}: {e}",
                self.t(
                    "Previzualizarea nu a putut fi deschisă",
                    "Could not open the preview"
                )
            )),
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
        let system = companion_core::prompts::with_style(&self.s.system_prompt, self.prompts.style);
        let system_chars = system.chars().count();
        let (extra, rest, _dropped) = self.memory_messages(system_chars);
        // Passages from the person's documents, when a collection is switched on.
        let question = self
            .chat
            .conv
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.chars().take(800).collect::<String>())
            .unwrap_or_default();
        let doc_ctx = self.docs_context(&question);
        if let Some(last) = self.chat.conv.messages.last_mut() {
            if last.role == "assistant" {
                last.sources = doc_ctx
                    .as_ref()
                    .map(|c| c.sources.clone())
                    .unwrap_or_default();
            }
        }
        let extra = match doc_ctx {
            Some(c) if extra.is_empty() => c.text,
            Some(c) => format!("{extra}\n\n{}", c.text),
            None => extra,
        };
        let tools_note = self.mcp_system_note();
        let extra = match (extra.is_empty(), tools_note.is_empty()) {
            (_, true) => extra,
            (true, false) => tools_note,
            (false, false) => format!("{extra}\n\n{tools_note}"),
        };
        let system = match (system.is_empty(), extra.is_empty()) {
            (_, true) => system,
            (true, false) => extra,
            (false, false) => format!("{system}\n\n{extra}"),
        };
        if !system.is_empty() {
            messages.push(ChatMessage::new("system", system));
        }
        messages.extend(rest);
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
        self.begin_meter(turn);
        let (client, tx, ctx) = (self.client(), self.tx.clone(), self.ctx.clone());
        thread::spawn(move || {
            let result = client.stream_chat(&req, &cancel, |ev| {
                let _ = tx.send(Event::ChatDelta { turn, ev });
                // At most about 30 redraws a second while text streams in: a fast model sends
                // hundreds of pieces a second and nobody can read faster than the screen refreshes.
                ctx.request_repaint_after(STREAM_REPAINT);
            });
            let _ = tx.send(Event::ChatDone { turn, result });
            ctx.request_repaint();
        });
    }

    /// Stop now: keep what has been written so far.
    pub fn stop_generation(&mut self) {
        if let Some(g) = self.chat.generating.take() {
            g.cancel.cancel();
            self.finish_meter(None);
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
            Event::Gpus(list) => self.gpus = Some(list),
            Event::Plan { dir, result } => {
                if self.plan_asked.as_deref() == Some(dir.as_path()) {
                    self.plan = Some(result);
                }
            }
            Event::CtxPlans { dir, rows } => {
                if self.active_model_dir().as_deref() == Some(dir.as_path()) {
                    self.ctx_rows = Some(rows);
                }
                self.ctx_busy = false;
            }
            Event::DiskSpeed(r) => {
                self.disk_busy = false;
                self.disk_speed = Some(r);
            }
            Event::Captured(r) => self.captured(r),
            Event::AutoTuneProgress { done, max, best } => self.autotune_progress(done, max, best),
            Event::AutoTuneDone(r) => self.autotune_done(r),
            Event::CtxApplied(r) => {
                self.ctx_busy = false;
                if let Ok(n) = &r {
                    self.remember_context(*n);
                }
                self.ctx_rows = None;
                self.ctx_auto_done = true;
                self.plan = None;
                self.plan_asked = None;
                self.tune_note = Some(match r {
                    Ok(n) => self
                        .t(
                            "Contextul este acum de {} de jetoane; modelul a fost repornit.",
                            "The context is now {} tokens; the model was restarted.",
                        )
                        .replacen("{}", &n.to_string(), 1),
                    Err(e) => e,
                });
            }
            Event::Tuned(r) => {
                use companion_core::coli::TuneOutcome;
                self.tuning = false;
                self.plan = None;
                self.plan_asked = None;
                self.tune_note = Some(match r {
                    Ok(TuneOutcome::Faster { before, after }) => self
                        .t(
                            "Măsurat: {} → {} jetoane/s. Setarea mai rapidă este salvată și se aplică la pornire.",
                            "Measured: {} → {} tokens/s. The faster setting is saved and applied at start.",
                        )
                        .replacen("{}", &format!("{before:.1}"), 1)
                        .replacen("{}", &format!("{after:.1}"), 1),
                    Ok(TuneOutcome::Unchanged) => self
                        .t(
                            "Nicio setare nu a depășit pragul de siguranță și câștig: nu s-a schimbat nimic.",
                            "No setting cleared the safety and gain thresholds: nothing was changed.",
                        )
                        .to_string(),
                    Ok(TuneOutcome::Unknown) => self
                        .t(
                            "Măsurarea s-a încheiat, dar rezultatul nu a putut fi citit. Consultă jurnalul.",
                            "Tuning finished, but the result could not be read. Check the log.",
                        )
                        .to_string(),
                    Err(e) => e,
                });
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
                    self.awaiting_status = false;
                }
                Err(e) => {
                    self.status_error = Some(e);
                    self.awaiting_status = false;
                }
            },
            Event::Mail => {
                if let Some(h) = self.mail.health.take() {
                    self.handle(Event::Health(h));
                }
                if let Some(st) = self.mail.status.take() {
                    self.handle(Event::Status(st));
                }
                if let Some(v) = self.mail.resources.take() {
                    self.handle(Event::Resources(v));
                }
            }
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
            Event::FetchProgress { done, total } => {
                self.fetch_progress = Some((done, total));
            }
            Event::FetchUnpacking => self.fetch_unpacking = true,
            Event::UpdateChecked {
                colibri,
                app,
                error,
            } => {
                self.upd.checking = false;
                self.upd.colibri = colibri;
                self.upd.app = app;
                self.upd.error = error;
                self.upd.dismissed = false;
            }
            Event::ColibriUpdated(r) => {
                self.upd.updating = false;
                match r {
                    Ok(()) => {
                        self.upd.colibri = None;
                        self.toast(self.t(
                            "colibri este la zi. Verifică lista de modele: pot exista modele noi.",
                            "colibri is up to date. Check the model list: there may be new models.",
                        ));
                        self.refresh_engine_info();
                    }
                    Err(e) => self.toast(format!(
                        "{}: {e}",
                        self.t("Actualizarea nu a reușit", "The update failed")
                    )),
                }
            }
            Event::Installed(list) => self.installed = list,
            Event::Npu(n) => self.npu = n,
            Event::Attached { name, result } => self.attached(name, result),
            Event::Picture(r) => self.picture_done(r),
            Event::DuelPiece { round, side, piece } => self.duel_piece(round, side, piece),
            Event::DuelEnd { round, side, err } => self.duel_end(round, side, err),
            Event::Memory(b) => self.memory_built(*b),
            Event::DocsRead { coll, name, result } => self.docs_read(coll, name, result),
            Event::McpStarted { name, result } => self.mcp_started(name, result),
            Event::McpResult { req, result } => self.mcp_result(req, result),
            Event::McpManual(r) => self.mcp_manual_result(r),
            Event::Summary { conv, upto, result } => self.summary_done(conv, upto, result),
            Event::FlowLog(l) => {
                let l = crate::i18n::engine_text(self.s.lang, &l);
                self.flows.log.push(l);
                let n = self.flows.log.len();
                if n > 300 {
                    self.flows.log.drain(..n - 300);
                }
            }
            Event::FlowNotify(t) => {
                self.flows.result = t;
            }
            Event::FlowApprove(text, tx) => {
                self.flows.open = true;
                self.flows.approval = Some((text, tx));
            }
            Event::FlowDone { ok, output } => self.flow_done(ok, output),
            Event::FlowTestProgress(d, t) => self.flows.test_progress = (d, t),
            Event::FlowTestDone(n, r, c) => self.flow_test_done(n, r, c),
            Event::Online { what, kind, result } => self.online_done(what, kind, result),
            Event::Resources(v) => {
                for (hist, val) in [
                    (&mut self.cpu_hist, v.snap.cpu_percent),
                    (
                        &mut self.ram_hist,
                        if v.snap.ram_total_gb > 0.0 {
                            (v.snap.ram_used_gb / v.snap.ram_total_gb * 100.0) as f32
                        } else {
                            0.0
                        },
                    ),
                ] {
                    hist.push_back(val);
                    while hist.len() > 60 {
                        hist.pop_front();
                    }
                }
                self.res = Some(*v);
            }
            Event::ModelDeleted(r) => {
                match r {
                    Ok(id) => {
                        let msg = self
                            .t("Am șters «{}»", "Deleted “{}”")
                            .replacen("{}", &id, 1);
                        self.toast(msg);
                    }
                    Err(e) => self.toast(format!(
                        "{}: {e}",
                        self.t("Modelul nu a putut fi șters", "Could not delete the model")
                    )),
                }
                self.refresh_installed();
            }
            Event::InstallEnded(code) => {
                self.install_spawning = false;
                self.awaiting_status = true;
                if code != Some(0) {
                    self.force_autostart = false;
                }
                self.install_exit = code;
                self.refresh_installed();
                self.refresh_install_log();
                if let Some(coli) = self.coli.clone() {
                    self.spawn(move |tx| {
                        let _ = tx.send(Event::Status(coli.status()));
                    });
                }
            }
            Event::ServerAction { what, result } => {
                self.busy = None;
                self.tools_server_changed();
                if what == ServerAction::Stop {
                    self.health = Health::Down;
                    self.starting_since = None;
                    self.auto_started = true; // do not undo a deliberate stop
                    if self.update_after_stop {
                        self.update_after_stop = false;
                        if result.is_ok() {
                            self.update_colibri();
                        }
                    }
                    if let Some(id) = self.switch_after_stop.take() {
                        if result.is_ok() {
                            self.auto_started = false;
                            self.begin_setup(&id);
                        } else {
                            self.force_autostart = false;
                        }
                    }
                }
                if let Err(e) = result {
                    self.starting_since = None;
                    self.toast(e);
                }
            }
            Event::Logs(text) => self.logs_view = Some(text),
            Event::MeterSample { turn, sample } => self.meter_sample(turn, sample),
            Event::ChatDelta { turn, ev } => {
                if self.chat.generating.as_ref().is_some_and(|g| g.id == turn) {
                    if let Some(m) = self.chat.conv.messages.last_mut() {
                        match &ev {
                            StreamEvent::Content(t) => m.content.push_str(t),
                            StreamEvent::Reasoning(t) => m.reasoning.push_str(t),
                        }
                    }
                    match &ev {
                        StreamEvent::Content(t) => {
                            self.meter_delta(turn, companion_core::meter::Kind::Answer, t)
                        }
                        StreamEvent::Reasoning(t) => {
                            self.meter_delta(turn, companion_core::meter::Kind::Reasoning, t)
                        }
                    }
                }
            }
            Event::ChatDone { turn, result } => {
                if self.chat.generating.as_ref().is_some_and(|g| g.id == turn) {
                    self.chat.generating = None;
                    self.finish_meter(result.as_ref().ok());
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
                    self.memory.dirty = true;
                    if self.chat.error.is_none() {
                        self.maybe_summarise();
                    }
                    if self.chat.error.is_none() {
                        if let Some(m) = self.chat.conv.messages.last() {
                            if m.role == "assistant" && !m.stopped && !m.content.is_empty() {
                                let t = m.content.clone();
                                self.answer_ready(&t);
                            }
                        }
                        self.mcp_after_answer();
                    }
                    if self.chat.error.is_some()
                        || self
                            .chat
                            .conv
                            .messages
                            .iter()
                            .rev()
                            .nth(0)
                            .is_some_and(|m| m.stopped)
                    {
                        self.chat.send_queue.clear();
                    } else if let Some(next) = self.chat.send_queue.pop() {
                        self.push_user_turn(next, Vec::new());
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------ per frame

    pub fn tick(&mut self) {
        while let Ok(ev) = self.rx.try_recv() {
            self.handle(ev);
        }
        let look = (
            self.s.theme,
            (self.s.font_scale * 100.0) as u32,
            self.s.access.clone(),
        );
        if self.applied_look.as_ref() != Some(&look) {
            self.p = theme::palette_for(self.s.theme, self.s.access.colour_blind);
            theme::apply(&self.ctx, self.s.theme, self.s.font_scale, &self.s.access);
            self.applied_look = Some(look);
        }
        self.flows_tick();
        self.tools_tick();
        self.memory_tick();
        self.sync_poller();
        if self.show_resources {
            self.sync_resource_targets();
        }
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
            && (self.s.start_server_automatically || self.force_autostart)
            && !self.auto_started
            && !self.awaiting_status
            && self.status.is_some()
        {
            self.auto_started = true;
            self.force_autostart = false;
            self.start_server();
        }
        if self.s.check_updates
            && self.coli.is_some()
            && matches!(self.python, Probe::Found(_))
            && matches!(
                self.phase(),
                Phase::Chat | Phase::StartServer | Phase::ChooseModel
            )
        {
            self.check_updates(false);
        }
        // colibri was not found and Python is there: fetch it without asking, once.
        if self.phase() == Phase::NeedColibri
            && !self.cloning
            && !self.auto_fetch_tried
            && !self.demo
            && self.git_available.is_some()
        {
            self.auto_fetch_tried = true;
            self.clone_colibri();
        }
        if self.show_models && self.phase() == Phase::Installing {
            self.refresh_installed_if_stale();
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

fn clone_with_git(dest: &std::path::Path, tx: &Sender<Event>) -> Result<(), String> {
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
    .arg(dest);
    proc::hide_window(&mut cmd);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let mut err = child.stderr.take().ok_or("no output from git")?;
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
    match child.wait() {
        Ok(s) if s.success() => Ok(()),
        _ => Err("git clone failed".into()),
    }
}

/// Download colibri's ZIP and unpack it with Python: works without git.
fn zip_with_python(
    python: &PythonCmd,
    dest: &std::path::Path,
    tx: &Sender<Event>,
    ctx: &egui::Context,
) -> Result<(), String> {
    use companion_core::fetch::{self, FetchLine};
    use std::io::{BufRead, BufReader, Read};
    let mut child = fetch::spawn_zip_download(python, locate::COLIBRI_ZIP_URL, dest)
        .map_err(|e| e.to_string())?;
    let mut stderr = child.stderr.take();
    let reader = thread::spawn(move || {
        let mut text = String::new();
        if let Some(e) = stderr.as_mut() {
            let _ = e.read_to_string(&mut text);
        }
        text
    });
    let mut last_sent = Instant::now() - Duration::from_secs(1);
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            match fetch::parse_line(&line) {
                Some(FetchLine::Progress { done, total }) => {
                    if last_sent.elapsed() > Duration::from_millis(120) {
                        last_sent = Instant::now();
                        let _ = tx.send(Event::FetchProgress { done, total });
                        ctx.request_repaint();
                    }
                }
                Some(FetchLine::Unpacking) => {
                    let _ = tx.send(Event::FetchUnpacking);
                    ctx.request_repaint();
                }
                _ => {}
            }
        }
    }
    let ok = child.wait().map(|s| s.success()).unwrap_or(false);
    let err = reader.join().unwrap_or_default();
    if ok {
        update::write_fetched_marker(dest);
        Ok(())
    } else {
        Err(err
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("the download failed")
            .trim()
            .to_string())
    }
}

/// An error from the online connection in the given language: Romanian and English are written
/// out in the core, the other six come from the translation table by the English text.
pub fn explain_in(lang: Lang, e: &companion_core::connect::ConnectError) -> String {
    use companion_core::connect;
    match lang {
        Lang::Ro => connect::explain(e, true),
        Lang::En => connect::explain(e, false),
        other => {
            let (key, detail) = connect::explain_key(e);
            let detail = crate::i18n::lookup(other, &detail).map_or(detail, str::to_string);
            crate::i18n::lookup(other, key)
                .unwrap_or(key)
                .replace("{}", &detail)
        }
    }
}

/// How much of the end of the install log is looked at: far more than 40 lines of any real log.
const INSTALL_LOG_TAIL: u64 = 64 * 1024;

/// The last `max` bytes of a text file (all of it when it is smaller), as text. When the file is
/// cut, the first, probably partial, line is dropped, so only whole lines come back.
fn read_tail(path: &std::path::Path, max: u64) -> std::io::Result<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let cut = len > max;
    if cut {
        f.seek(SeekFrom::Start(len - max))?;
    }
    let mut bytes = Vec::with_capacity(len.min(max) as usize);
    f.take(max).read_to_end(&mut bytes)?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    if cut {
        return Ok(text
            .split_once('\n')
            .map_or(String::new(), |(_, rest)| rest.to_string()));
    }
    Ok(text)
}

fn coli_last_log(log: &std::path::Path) -> String {
    let t = std::fs::read_to_string(log).unwrap_or_default();
    companion_core::coli::last_lines(&t, 4)
}

#[cfg(test)]
mod tail_tests {
    use super::read_tail;
    use std::io::Write;

    fn file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("cc-tail-{}-{name}", std::process::id()));
        std::fs::File::create(&p).unwrap().write_all(bytes).unwrap();
        p
    }

    fn last40(text: &str) -> Vec<&str> {
        let mut v: Vec<&str> = text.lines().rev().take(40).collect();
        v.reverse();
        v
    }

    #[test]
    fn a_small_file_is_read_whole_and_a_big_one_by_its_end() {
        let small = file("small", b"one\ntwo\nthree");
        assert_eq!(read_tail(&small, 1024).unwrap(), "one\ntwo\nthree");
        let _ = std::fs::remove_file(small);

        let lines: Vec<String> = (0..5000).map(|i| format!("line {i} ünïcode ✓")).collect();
        let text = lines.join("\n") + "\n";
        let big = file("big", text.as_bytes());
        // a limit that lands inside a multi-byte character must still give whole, valid lines
        for max in [100u64, 1000, 4097, 65536] {
            let tail = read_tail(&big, max).unwrap();
            assert!(!tail.contains('\u{FFFD}'), "max {max}: broken character");
            let (a, b) = (last40(&tail), last40(&text));
            // a short tail holds fewer than 40 lines: they must be the file's last whole lines
            assert!(b.ends_with(&a), "max {max}");
            if max >= 4097 {
                assert_eq!(a, b, "max {max}");
            }
        }
        let _ = std::fs::remove_file(big);
    }

    #[test]
    fn empty_missing_and_line_less_files() {
        let empty = file("empty", b"");
        assert_eq!(read_tail(&empty, 10).unwrap(), "");
        let _ = std::fs::remove_file(empty);
        assert!(read_tail(std::path::Path::new("/definitely/not/here.log"), 10).is_err());
        let one_long = file("long", "x".repeat(5000).as_bytes());
        assert_eq!(read_tail(&one_long, 100).unwrap(), ""); // only a partial line: nothing whole
        let _ = std::fs::remove_file(one_long);
    }

    #[test]
    #[ignore]
    fn perf_install_log_tail() {
        let line = "Collecting some-package==1.2.3 (from requirements) Downloading 12.3 MB\n";
        let p = file("perf", line.repeat(400_000).as_bytes()); // about 28 MB
        let time = |f: &dyn Fn() -> usize| {
            let t = std::time::Instant::now();
            let n = (0..5).map(|_| f()).sum::<usize>();
            (t.elapsed().as_secs_f64() * 1000.0 / 5.0, n)
        };
        let old = time(&|| {
            let text = std::fs::read_to_string(&p).unwrap();
            text.lines().rev().take(40).count()
        });
        let new = time(&|| {
            let text = read_tail(&p, super::INSTALL_LOG_TAIL).unwrap();
            text.lines().rev().take(40).count()
        });
        println!(
            "perf_install_log 28 MB: OLD {:.2} ms, NEW {:.3} ms",
            old.0, new.0
        );
        let _ = std::fs::remove_file(p);
    }
}
