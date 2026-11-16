//! A typed wrapper around colibri's own launcher (`c/coli`). Everything goes
//! through the documented, machine-readable commands (`setup --list --json`,
//! `status --json`, `start`, `stop`, `logs`), so no colibri code is copied or
//! linked: this program only talks to it, the way any other client would.

use crate::locate::PythonCmd;
use crate::proc::{hide_window, interrupt_pid, run_capture};
use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

pub const GB: f64 = 1_000_000_000.0;

// ------------------------------------------------------------------ catalog

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub family: String,
    pub repo: String,
    pub download_gb: f64,
    pub ram_min_gb: f64,
    pub ram_good_gb: f64,
    pub dense_gb: f64,
    /// `text` (chat) or `image` (text to picture); decision models may appear later.
    pub kind: String,
    /// `from RAM` or `streams experts from the SSD`.
    pub runs: String,
    pub summary: String,
    /// `good`, `tight`, `needs-ram`, `needs-disk`, `unsupported`.
    pub status: String,
    pub fits: bool,
    pub reason: String,
    pub recommended: bool,
    pub license_note: String,
    pub doc: String,
}

/// A quick way to narrow the model list. Several are approximations drawn from the catalogue's own
/// words (name, family, summary), so a model can be missing from "Code" if its description never says so.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ModelFilter {
    #[default]
    All,
    /// Fits this computer and is ready to download and run.
    Immediate,
    /// The catalogue's own recommendation.
    Top,
    /// Runs from RAM (no streaming from the SSD) and is a modest download.
    Rapid,
    Code,
    Creative,
    Math,
    Multilingual,
}

impl ModelFilter {
    pub const ALL: [ModelFilter; 8] = [
        ModelFilter::All,
        ModelFilter::Immediate,
        ModelFilter::Top,
        ModelFilter::Rapid,
        ModelFilter::Code,
        ModelFilter::Creative,
        ModelFilter::Math,
        ModelFilter::Multilingual,
    ];

    fn words(self) -> &'static [&'static str] {
        match self {
            ModelFilter::Code => &["code", "coder", "coding", "programming", "developer"],
            ModelFilter::Creative => &[
                "creative", "story", "stories", "roleplay", "writing", "fiction",
            ],
            ModelFilter::Math => &["math", "reasoning", "thinking", "logic", "science"],
            ModelFilter::Multilingual => &["multilingual", "languages", "translation"],
            _ => &[],
        }
    }
}

impl ModelInfo {
    fn text_blob(&self) -> String {
        format!("{} {} {} {}", self.id, self.name, self.family, self.summary).to_lowercase()
    }

    pub fn passes(&self, f: ModelFilter) -> bool {
        match f {
            ModelFilter::All => true,
            ModelFilter::Immediate => self.fits && self.status == "good",
            ModelFilter::Top => self.recommended,
            ModelFilter::Rapid => {
                self.fits && !self.streams_from_disk() && self.download_gb <= 10.0
            }
            other => {
                let blob = self.text_blob();
                other.words().iter().any(|w| blob.contains(w))
            }
        }
    }

    /// Every word typed must appear in the name, family, summary or id.
    pub fn matches_query(&self, query: &str) -> bool {
        let blob = self.text_blob();
        query
            .split_whitespace()
            .all(|w| blob.contains(&w.to_lowercase()))
    }
}

impl ModelInfo {
    /// This app is a chat interface: only text models can be used in it.
    pub fn is_chat_model(&self) -> bool {
        self.kind == "text" || self.kind.is_empty()
    }
    pub fn streams_from_disk(&self) -> bool {
        self.runs.contains("SSD")
    }
    pub fn hub_url(&self) -> String {
        format!("https://huggingface.co/{}", self.repo)
    }
}

/// Parse the output of `coli setup --list --json`.
pub fn parse_catalog(json: &str) -> Result<Vec<ModelInfo>, String> {
    serde_json::from_str(json).map_err(|e| format!("could not read the model list: {e}"))
}

/// Order models for display: what fits and chats first, the recommendation on top,
/// then by download size so the lightest options come before the heaviest.
pub fn sort_for_display(models: &mut [ModelInfo]) {
    models.sort_by(|a, b| {
        let key = |m: &ModelInfo| (!m.recommended, !m.fits, !m.is_chat_model());
        key(a).cmp(&key(b)).then(
            a.download_gb
                .partial_cmp(&b.download_gb)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
}

/// The most capable chat model this computer can actually run: among the models that fit
/// (memory, disk and a supported engine), the largest one. Bigger checkpoints are the more
/// sophisticated ones; models that stream their experts from the SSD are what lets a
/// computer with 8 or 16 GB of RAM and no strong graphics card run them at all.
pub fn strongest_that_fits(models: &[ModelInfo]) -> Option<&ModelInfo> {
    models
        .iter()
        .filter(|m| m.fits && m.is_chat_model() && m.status != "unsupported")
        .max_by(|a, b| {
            a.download_gb
                .partial_cmp(&b.download_gb)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

// ------------------------------------------------------------------ hardware

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hardware {
    pub cpu_name: String,
    pub cores: u32,
    pub ram_total_gb: f64,
    pub ram_free_gb: f64,
    pub disk_free_gb: f64,
    pub gpus: Vec<String>,
    /// NVIDIA cards (the ones CUDA can use), with their memory.
    pub nvidia: Vec<NvidiaCard>,
    /// A Vulkan-capable graphics chip, integrated or not.
    pub vulkan: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NvidiaCard {
    pub name: String,
    pub vram_gb: f64,
    pub compute_cap: Option<String>,
}

/// What `Backend::Auto` will end up using on this computer, for showing in words.
#[derive(Clone, Debug, PartialEq)]
pub enum Engine {
    Cuda(String),
    Vulkan(String),
    /// The unified-memory GPU of an Apple Silicon Mac, through colibri's opt-in Metal backend.
    Metal,
    Cpu,
}

impl Hardware {
    pub fn auto_engine(&self) -> Engine {
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return Engine::Metal;
        }
        if let Some(c) = self.nvidia.first() {
            Engine::Cuda(c.name.clone())
        } else if let Some(v) = &self.vulkan {
            Engine::Vulkan(v.clone())
        } else {
            Engine::Cpu
        }
    }
}

impl Hardware {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("hardware report: {e}"))?;
        let num = |p: &str| -> f64 {
            p.split('.')
                .try_fold(&v, |cur, k| cur.get(k))
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
        };
        let mut gpus = Vec::new();
        if let Some(list) = v.pointer("/gpu/nvidia").and_then(Value::as_array) {
            for g in list {
                if let Some(name) = g.get("name").and_then(Value::as_str) {
                    gpus.push(name.to_string());
                }
            }
        }
        if let Some(vk) = v.pointer("/gpu/vulkan").filter(|x| x.is_object()) {
            let name = ["name", "device_name", "deviceName"]
                .iter()
                .find_map(|k| vk.get(*k).and_then(Value::as_str));
            if let Some(name) = name {
                if !gpus.iter().any(|g| g == name) {
                    gpus.push(name.to_string());
                }
            }
        }
        let nvidia = v
            .pointer("/gpu/nvidia")
            .and_then(Value::as_array)
            .map(|list| {
                list.iter()
                    .filter_map(|g| {
                        Some(NvidiaCard {
                            name: g.get("name")?.as_str()?.to_string(),
                            vram_gb: g.get("total_bytes").and_then(Value::as_f64).unwrap_or(0.0)
                                / GB,
                            compute_cap: g
                                .get("compute_cap")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let vulkan = v
            .pointer("/gpu/vulkan")
            .filter(|x| x.is_object())
            .and_then(|vk| {
                ["name", "device_name", "deviceName"]
                    .iter()
                    .find_map(|k| vk.get(*k).and_then(Value::as_str))
            })
            .map(str::to_string);
        let warnings = v
            .pointer("/disk/warnings")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|w| w.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Hardware {
            cpu_name: v
                .pointer("/cpu/name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            cores: num("cpu.physical_cores").max(num("cpu.logical_cores")) as u32,
            ram_total_gb: num("memory.total") / GB,
            ram_free_gb: num("memory.available") / GB,
            disk_free_gb: num("disk.free_bytes") / GB,
            gpus,
            nvidia,
            vulkan,
            warnings,
        })
    }
}

// ------------------------------------------------------------------ status

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelRef {
    pub id: Option<String>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct SetupInfo {
    pub status: Option<String>,
    pub model: Option<ModelRef>,
    pub model_dir: Option<String>,
    pub backend: Option<String>,
    pub engine: Option<String>,
    pub ready: bool,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Urls {
    pub browser: String,
    pub openai_base_url: String,
    pub anthropic_base_url: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct ServerInfo {
    /// `stopped`, `loading` or `ready`.
    pub state: String,
    pub pid: Option<u32>,
    pub urls: Urls,
    pub model: Option<String>,
    pub backend: Option<String>,
    pub model_id: Option<String>,
    pub tokens_per_second: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct InstallState {
    /// `build`, `download`, `ready`, `starting`, `started`, `error`, `interrupted`.
    pub phase: String,
    pub running: bool,
    pub pid: Option<u32>,
    pub done: Option<f64>,
    pub total: Option<f64>,
    pub rate: Option<f64>,
    pub message: Option<String>,
    pub model: Option<String>,
}

impl InstallState {
    pub fn fraction(&self) -> Option<f32> {
        match (self.done, self.total) {
            (Some(d), Some(t)) if t > 0.0 => Some((d / t).clamp(0.0, 1.0) as f32),
            _ => None,
        }
    }
    /// Seconds left at the current rate, when it can be known.
    pub fn eta_seconds(&self) -> Option<f64> {
        let (d, t, r) = (self.done?, self.total?, self.rate?);
        (r > 1.0 && t > d).then(|| (t - d) / r)
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct StatusReport {
    pub configured: bool,
    pub config_path: Option<String>,
    pub setup: Option<SetupInfo>,
    pub server: Option<ServerInfo>,
    pub install: Option<InstallState>,
}

impl StatusReport {
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| format!("could not read the status: {e}"))
    }
    pub fn setup_ready(&self) -> bool {
        self.setup.as_ref().is_some_and(|s| s.ready)
    }
    pub fn install_running(&self) -> bool {
        self.install.as_ref().is_some_and(|i| i.running)
    }
    pub fn server_state(&self) -> &str {
        self.server
            .as_ref()
            .map(|s| s.state.as_str())
            .unwrap_or("stopped")
    }
}

// ------------------------------------------------------------------ colibri's own placement plan

/// What `coli plan --json` says about this model on this computer: where the weights are kept
/// and what limits the speed. Read, never decided here: colibri's planner does the work.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlanSummary {
    /// colibri's own wording of the limit, e.g. "disk expert misses".
    pub bottleneck: String,
    /// `disk`, `memory`, `compute` or `mixed`.
    pub class: String,
    /// Share of expert lookups served from memory instead of the disk (0..1).
    pub hit_rate: f64,
    /// Expert bytes that stay on the disk.
    pub cold_gb: f64,
    /// Memory colibri reserves for the expert cache.
    pub cache_gb: f64,
    pub warnings: Vec<String>,
}

impl PlanSummary {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let start = text.find('{').ok_or("colibri returned no plan")?;
        let v: serde_json::Value = serde_json::from_str(&text[start..])
            .map_err(|e| format!("could not read the plan: {e}"))?;
        let gb = |p: &str| {
            v.pointer(p)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0)
                / 1_000_000_000.0
        };
        let text_at = |p: &str| {
            v.pointer(p)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        Ok(Self {
            bottleneck: text_at("/expected_bottleneck"),
            class: text_at("/bottleneck_class"),
            hit_rate: v
                .pointer("/projected_hit_rate")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0),
            cold_gb: gb("/tiers/disk/cold_expert_bytes"),
            cache_gb: gb("/tiers/ram/expert_cache_bytes"),
            warnings: v
                .pointer("/warnings")
                .and_then(serde_json::Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|w| w.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
        })
    }

    /// Part of the experts is read from the disk while answering.
    pub fn reads_disk(&self) -> bool {
        self.class == "disk" && self.cold_gb > 0.05
    }
}

/// How a `coli tune` run ended.
#[derive(Clone, Debug, PartialEq)]
pub enum TuneOutcome {
    /// A faster setting passed colibri's quality and safety gates (tokens per second, before → after).
    Faster {
        before: f64,
        after: f64,
    },
    /// No candidate cleared the gates; nothing was changed.
    Unchanged,
    Unknown,
}

fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for n in chars.by_ref() {
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Reads the closing lines of `coli tune`: "accepted NAME: 3.00 → 4.00 tok/s" or "baseline retained".
pub fn parse_tune_log(log: &str) -> TuneOutcome {
    let log = strip_ansi(log);
    for line in log.lines().rev() {
        if line.contains("baseline retained") {
            return TuneOutcome::Unchanged;
        }
        if line.contains("accepted") && line.contains("tok/s") {
            let nums: Vec<f64> = line
                .split(|c: char| !(c.is_ascii_digit() || c == '.'))
                .filter_map(|t| t.parse::<f64>().ok())
                .collect();
            if nums.len() >= 2 {
                return TuneOutcome::Faster {
                    before: nums[0],
                    after: nums[1],
                };
            }
        }
    }
    TuneOutcome::Unknown
}

// ------------------------------------------------------------------ the wrapper

#[derive(Clone, Debug)]
pub struct Coli {
    pub root: PathBuf,
    pub python: PythonCmd,
    /// Extra environment for every `coli` run (the switches that turn on Metal or the NPU).
    pub env: Vec<(String, String)>,
}

/// Something the user can copy and run themselves. The app never runs `sudo`,
/// `apt` or `winget` on its own: it shows the exact command and lets the person decide.
pub fn find_package_hint(log: &str) -> Option<String> {
    for line in log.lines().rev() {
        let line = line.trim();
        if let Some(pos) = line.find("first run:") {
            let cmd = line[pos + "first run:".len()..].trim();
            if !cmd.is_empty() {
                return Some(cmd.to_string());
            }
        }
    }
    None
}

impl Coli {
    pub fn new(root: PathBuf, python: PythonCmd) -> Self {
        Self {
            root,
            python,
            env: Vec::new(),
        }
    }

    fn script(&self) -> PathBuf {
        self.root.join("c").join("coli")
    }

    fn base(&self) -> Command {
        let mut c = self.python.command();
        c.current_dir(&self.root)
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUTF8", "1")
            .env("PYTHONUNBUFFERED", "1");
        for (k, v) in &self.env {
            c.env(k, v);
        }
        c
    }

    fn coli_cmd(&self, args: &[&str]) -> Command {
        let mut c = self.base();
        c.arg(self.script()).args(args);
        c
    }

    fn run(&self, args: &[&str], timeout: Duration) -> Result<String, String> {
        let out = run_capture(self.coli_cmd(args), timeout).map_err(|e| e.to_string())?;
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        if out.status.success() || !text.trim().is_empty() {
            Ok(text)
        } else {
            Err(last_lines(&String::from_utf8_lossy(&out.stderr), 6))
        }
    }

    pub fn version(&self) -> Option<String> {
        self.run(&["--version"], Duration::from_secs(20))
            .ok()
            .map(|s| s.trim().to_string())
    }

    pub fn list_models(&self) -> Result<Vec<ModelInfo>, String> {
        // `--all` includes the models that do not fit, so the app can say why.
        let text = self.run(
            &["setup", "--list", "--all", "--json"],
            Duration::from_secs(60),
        )?;
        parse_catalog(&text)
    }

    pub fn hardware(&self, models_dir: Option<&Path>) -> Result<Hardware, String> {
        let mut c = self.base();
        c.arg(self.root.join("c").join("setup_hw.py")).arg("--json");
        if let Some(d) = models_dir {
            c.arg(d);
        }
        let out = run_capture(c, Duration::from_secs(60)).map_err(|e| e.to_string())?;
        Hardware::from_json(&String::from_utf8_lossy(&out.stdout))
    }

    pub fn status(&self) -> Result<StatusReport, String> {
        let text = self.run(&["status", "--json"], Duration::from_secs(30))?;
        StatusReport::from_json(&text)
    }

    /// Start `coli setup` as a detached background process whose output goes to `log`.
    /// It keeps running if this window closes; running it again resumes a download.
    pub fn spawn_install(
        &self,
        model_id: &str,
        models_dir: Option<&Path>,
        backend: Option<&str>,
        log: &Path,
    ) -> io::Result<Child> {
        if let Some(dir) = log.parent() {
            fs::create_dir_all(dir)?;
        }
        let out = fs::File::create(log)?;
        let err = out.try_clone()?;
        let mut c = self.coli_cmd(&[
            "setup",
            "--yes",
            "--no-start",
            "--no-browser",
            "--model",
            model_id,
        ]);
        if let Some(d) = models_dir {
            c.arg("--dir").arg(d);
        }
        if let Some(b) = backend {
            c.arg("--backend").arg(b);
        }
        c.stdin(Stdio::null()).stdout(out).stderr(err);
        hide_window(&mut c);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            c.process_group(0); // not tied to this program's terminal or Ctrl+C
        }
        c.spawn()
    }

    /// Ask a running install to stop; it can be resumed later with no loss.
    pub fn pause_install(&self, pid: u32) -> io::Result<()> {
        interrupt_pid(pid)
    }

    /// `coli import FIRST.gguf`: converts a GGUF you already have (other parts beside it) into a
    /// model folder written next to it. Output goes to `log`; the child is returned to wait on.
    pub fn spawn_import(&self, first_gguf: &Path, log: &Path) -> io::Result<Child> {
        if let Some(dir) = log.parent() {
            fs::create_dir_all(dir)?;
        }
        let out = fs::File::create(log)?;
        let err = out.try_clone()?;
        let mut c = self.coli_cmd(&["import"]);
        c.arg(first_gguf)
            .stdin(Stdio::null())
            .stdout(out)
            .stderr(err);
        hide_window(&mut c);
        c.spawn()
    }

    /// `coli setup --model-dir DIR`: use a model folder that is already on this computer.
    pub fn spawn_use_dir(
        &self,
        model_dir: &Path,
        backend: Option<&str>,
        log: &Path,
    ) -> io::Result<Child> {
        if let Some(dir) = log.parent() {
            fs::create_dir_all(dir)?;
        }
        let out = fs::OpenOptions::new().create(true).append(true).open(log)?;
        let err = out.try_clone()?;
        let mut c = self.coli_cmd(&[
            "setup",
            "--yes",
            "--no-start",
            "--no-browser",
            "--model-dir",
        ]);
        c.arg(model_dir);
        if let Some(b) = backend {
            c.arg("--backend").arg(b);
        }
        c.stdin(Stdio::null()).stdout(out).stderr(err);
        hide_window(&mut c);
        c.spawn()
    }

    /// `coli plan --model DIR --json`: colibri's own placement plan for this computer.
    pub fn plan(&self, model_dir: &Path) -> Result<PlanSummary, String> {
        let dir = model_dir.to_string_lossy();
        let text = self.run(
            &["plan", "--model", dir.as_ref(), "--json"],
            Duration::from_secs(90),
        )?;
        PlanSummary::from_json(&text)
    }

    /// `coli tune --model DIR`: colibri measures its quality-preserving settings on this exact
    /// model and computer and saves the fastest; `--auto-tier` (what `coli start` passes) then
    /// loads that profile. Output goes to `log`; the child is returned to wait on.
    pub fn spawn_tune(&self, model_dir: &Path, log: &Path) -> io::Result<Child> {
        if let Some(dir) = log.parent() {
            fs::create_dir_all(dir)?;
        }
        let out = fs::File::create(log)?;
        let err = out.try_clone()?;
        let mut c = self.coli_cmd(&["tune", "--model"]);
        c.arg(model_dir)
            .stdin(Stdio::null())
            .stdout(out)
            .stderr(err);
        hide_window(&mut c);
        c.spawn()
    }

    pub fn start_server(&self) -> Result<String, String> {
        self.run(
            &["start", "--background", "--no-browser"],
            Duration::from_secs(120),
        )
    }

    pub fn stop_server(&self) -> Result<String, String> {
        self.run(&["stop"], Duration::from_secs(90))
    }

    pub fn logs(&self, install: bool, lines: u32) -> String {
        let n = lines.to_string();
        let mut args = vec!["logs", "-n", n.as_str()];
        if install {
            args.push("--install");
        }
        self.run(&args, Duration::from_secs(20)).unwrap_or_default()
    }
}

pub fn last_lines(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// Is this the first part of a split GGUF (`name-00001-of-0000N.gguf`), or a lone `.gguf`?
pub fn is_first_gguf(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if !name.ends_with(".gguf") {
        return false;
    }
    match name.find("-of-") {
        Some(i) => name[..i].ends_with("-00001"),
        None => true,
    }
}

/// Every part listed by a split GGUF's first file name that is missing beside it.
pub fn missing_gguf_parts(first: &Path) -> Vec<String> {
    let name = first
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let Some(i) = name.find("-of-") else {
        return Vec::new();
    };
    let (head, tail) = (&name[..i], &name[i + 4..]);
    let Some(stem) = head.strip_suffix("-00001") else {
        return Vec::new();
    };
    let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
    let Ok(total) = digits.parse::<u32>() else {
        return Vec::new();
    };
    let rest = &tail[digits.len()..];
    let dir = first.parent().unwrap_or(Path::new("."));
    (1..=total)
        .map(|n| format!("{stem}-{n:05}-of-{digits}{rest}"))
        .filter(|f| !dir.join(f).is_file())
        .collect()
}

/// The model folder `coli import` wrote beside the GGUF: the newest folder there that holds a
/// `config.json`, changed since `after`.
pub fn imported_dir(first: &Path, after: std::time::SystemTime) -> Option<PathBuf> {
    let parent = first.parent()?;
    fs::read_dir(parent)
        .ok()?
        .flatten()
        .filter(|e| e.path().join("config.json").is_file())
        .filter_map(|e| {
            let t = e.metadata().ok()?.modified().ok()?;
            (t >= after).then_some((t, e.path()))
        })
        .max_by_key(|(t, _)| *t)
        .map(|(_, p)| p)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CATALOG: &str = r#"[
      {"id":"a","name":"A","download_gb":19.4,"ram_min_gb":8,"ram_good_gb":18,"dense_gb":3.6,"kind":"text",
       "runs":"from RAM","summary":"s","status":"tight","fits":true,"reason":"r","recommended":true,
       "license_note":"","doc":"d","repo":"x/y","unknown_future_field":1},
      {"id":"b","name":"B","download_gb":300,"kind":"text","runs":"streams experts from the SSD","fits":false},
      {"id":"c","name":"C","download_gb":33,"kind":"image","license_note":"non-commercial","fits":true}
    ]"#;

    #[test]
    fn catalog_parses_and_sorts() {
        let mut m = parse_catalog(CATALOG).unwrap();
        assert_eq!(m.len(), 3);
        sort_for_display(&mut m);
        assert_eq!(m[0].id, "a");
        assert!(m[0].recommended);
        assert!(m[1].fits && !m[1].is_chat_model()); // fits, but not chat: before the one that does not fit? check order below
        assert!(m[2].streams_from_disk());
        assert_eq!(m[0].hub_url(), "https://huggingface.co/x/y");
    }

    #[test]
    fn status_parses_all_shapes() {
        let idle = StatusReport::from_json(
            r#"{"configured":false,"config_path":"/x","setup":null,"server":null,"install":null}"#,
        )
        .unwrap();
        assert!(!idle.configured && !idle.setup_ready());
        assert_eq!(idle.server_state(), "stopped");

        let busy = StatusReport::from_json(
            r#"{"configured":true,"setup":{"status":"x","model":{"id":"a","name":"A"},"ready":false},
                "server":null,
                "install":{"phase":"download","running":true,"pid":42,"done":5e9,"total":2.0e10,"rate":2.5e7}}"#,
        )
        .unwrap();
        let i = busy.install.as_ref().unwrap();
        assert!(busy.install_running());
        assert!((i.fraction().unwrap() - 0.25).abs() < 1e-6);
        assert_eq!(i.eta_seconds().unwrap().round() as i64, 600);

        let up = StatusReport::from_json(
            r#"{"configured":true,"setup":{"ready":true},"server":{"state":"ready","pid":7,
                "urls":{"browser":"http://127.0.0.1:8000/","openai_base_url":"http://127.0.0.1:8000/v1","anthropic_base_url":"http://127.0.0.1:8000"},
                "model_id":"qwen","tokens_per_second":6.5}}"#,
        )
        .unwrap();
        assert_eq!(up.server_state(), "ready");
        assert_eq!(up.server.unwrap().model_id.as_deref(), Some("qwen"));
    }

    #[test]
    fn hardware_parses_real_shape() {
        let hw = Hardware::from_json(
            r#"{"cpu":{"name":"Xeon","logical_cores":4,"physical_cores":2},
                "memory":{"total":8413868032,"available":6421245952},
                "disk":{"free_bytes":32155578368,"warnings":["slow disk"]},
                "gpu":{"vulkan":null,"nvidia":[{"name":"RTX 3070","compute_cap":"8.6"}]}}"#,
        )
        .unwrap();
        assert_eq!(hw.cores, 4);
        assert_eq!(hw.nvidia[0].compute_cap.as_deref(), Some("8.6"));
        assert!((hw.ram_total_gb - 8.41).abs() < 0.01);
        assert_eq!(hw.gpus, vec!["RTX 3070"]);
        assert_eq!(hw.warnings, vec!["slow disk"]);
    }

    #[test]
    fn engine_follows_the_best_hardware_and_strongest_model_fits() {
        let hw = Hardware::from_json(
            r#"{"gpu":{"vulkan":{"name":"Intel UHD"},"nvidia":[{"name":"RTX 4090","total_bytes":25757220000,"compute_cap":"8.9"}]}}"#,
        )
        .unwrap();
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return;
        }
        assert_eq!(hw.auto_engine(), Engine::Cuda("RTX 4090".into()));
        assert!((hw.nvidia[0].vram_gb - 25.76).abs() < 0.01);
        let igpu =
            Hardware::from_json(r#"{"gpu":{"vulkan":{"name":"Intel UHD"},"nvidia":[]}}"#).unwrap();
        assert_eq!(igpu.auto_engine(), Engine::Vulkan("Intel UHD".into()));
        assert_eq!(Hardware::default().auto_engine(), Engine::Cpu);

        let m = |id: &str, gb: f64, fits: bool, kind: &str| ModelInfo {
            id: id.into(),
            download_gb: gb,
            fits,
            kind: kind.into(),
            ..Default::default()
        };
        let list = vec![
            m("small", 20.0, true, "text"),
            m("huge", 400.0, false, "text"),
            m("big", 85.0, true, "text"),
            m("pic", 300.0, true, "image"),
        ];
        assert_eq!(strongest_that_fits(&list).unwrap().id, "big");
        assert!(strongest_that_fits(&list[1..2]).is_none());
    }

    #[test]
    fn finds_the_package_hint_the_setup_prints() {
        let log = "Engine: qwen36 with CPU\n  to use the GPU through VULKAN, first run:  sudo apt install libvulkan-dev glslc\nnext";
        assert_eq!(
            find_package_hint(log).as_deref(),
            Some("sudo apt install libvulkan-dev glslc")
        );
        assert_eq!(find_package_hint("nothing"), None);
    }

    #[test]
    fn split_gguf_parts_are_checked() {
        assert!(is_first_gguf(Path::new(
            "GLM-5.3-Flash-Q8_0-00001-of-00008.gguf"
        )));
        assert!(!is_first_gguf(Path::new(
            "GLM-5.3-Flash-Q8_0-00002-of-00008.gguf"
        )));
        assert!(is_first_gguf(Path::new("Flint-UD-Q4_K_XL.gguf")));
        assert!(!is_first_gguf(Path::new("model.safetensors")));
        let dir = std::env::temp_dir().join(format!("cc-gguf-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for n in [1, 2, 3, 5] {
            fs::write(dir.join(format!("M-Q8_0-{n:05}-of-00005.gguf")), b"x").unwrap();
        }
        let first = dir.join("M-Q8_0-00001-of-00005.gguf");
        assert_eq!(
            missing_gguf_parts(&first),
            vec!["M-Q8_0-00004-of-00005.gguf"]
        );
        let before = std::time::SystemTime::now() - Duration::from_secs(5);
        assert!(imported_dir(&first, before).is_none());
        fs::create_dir_all(dir.join("M.colibri")).unwrap();
        fs::write(dir.join("M.colibri/config.json"), b"{}").unwrap();
        assert_eq!(imported_dir(&first, before), Some(dir.join("M.colibri")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn filters_and_search_narrow_the_list() {
        let m =
            |id: &str, summary: &str, fits: bool, status: &str, gb: f64, runs: &str| ModelInfo {
                id: id.into(),
                name: id.into(),
                summary: summary.into(),
                fits,
                status: status.into(),
                download_gb: gb,
                runs: runs.into(),
                ..Default::default()
            };
        let coder = m(
            "qwen-coder",
            "A coding model",
            true,
            "good",
            5.0,
            "from RAM",
        );
        let big = m(
            "big",
            "Reasoning and math",
            true,
            "tight",
            80.0,
            "streams experts from the SSD",
        );
        assert!(coder.passes(ModelFilter::All));
        assert!(coder.passes(ModelFilter::Immediate));
        assert!(!big.passes(ModelFilter::Immediate));
        assert!(coder.passes(ModelFilter::Rapid));
        assert!(!big.passes(ModelFilter::Rapid));
        assert!(coder.passes(ModelFilter::Code) && !big.passes(ModelFilter::Code));
        assert!(big.passes(ModelFilter::Math));
        assert!(!coder.passes(ModelFilter::Top));
        assert!(coder.matches_query("QWEN coding"));
        assert!(!coder.matches_query("qwen math"));
        assert!(coder.matches_query("   "));
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;

    #[test]
    fn reads_the_plan() {
        let j = r#"{"version":2,"expected_bottleneck":"disk expert misses","bottleneck_class":"disk",
          "projected_hit_rate":0.41,"tiers":{"disk":{"cold_expert_bytes":14000000000},
          "ram":{"expert_cache_bytes":6500000000}},"warnings":["swap is in use"]}"#;
        let p = PlanSummary::from_json(&format!("noise\n{j}")).unwrap();
        assert_eq!(p.class, "disk");
        assert!(p.reads_disk());
        assert!((p.cold_gb - 14.0).abs() < 1e-9 && (p.cache_gb - 6.5).abs() < 1e-9);
        assert!((p.hit_rate - 0.41).abs() < 1e-9);
        assert_eq!(p.warnings, vec!["swap is in use".to_string()]);
        assert!(PlanSummary::from_json("no json here").is_err());
    }

    #[test]
    fn a_model_in_memory_does_not_read_the_disk() {
        let p = PlanSummary::from_json(
            r#"{"bottleneck_class":"memory","projected_hit_rate":1.0,"tiers":{"disk":{"cold_expert_bytes":0}}}"#,
        )
        .unwrap();
        assert!(!p.reads_disk());
    }

    #[test]
    fn reads_how_tuning_ended() {
        let ok = "x\n  \u{1b}[32m✓ accepted\u{1b}[0m pipe+direct: 3.00 → 4.50 tok/s (+50.0%)\n  profile: /p\n";
        assert_eq!(
            parse_tune_log(ok),
            TuneOutcome::Faster {
                before: 3.0,
                after: 4.5
            }
        );
        assert_eq!(
            parse_tune_log("  ✓ baseline retained: no candidate cleared the 3.0% gain"),
            TuneOutcome::Unchanged
        );
        assert_eq!(parse_tune_log("tuning failed"), TuneOutcome::Unknown);
    }
}
