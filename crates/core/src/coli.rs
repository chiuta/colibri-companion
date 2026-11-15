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

// ------------------------------------------------------------------ hardware

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hardware {
    pub cpu_name: String,
    pub cores: u32,
    pub ram_total_gb: f64,
    pub ram_free_gb: f64,
    pub disk_free_gb: f64,
    pub gpus: Vec<String>,
    pub warnings: Vec<String>,
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

// ------------------------------------------------------------------ the wrapper

#[derive(Clone, Debug)]
pub struct Coli {
    pub root: PathBuf,
    pub python: PythonCmd,
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
        Self { root, python }
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
        assert!((hw.ram_total_gb - 8.41).abs() < 0.01);
        assert_eq!(hw.gpus, vec!["RTX 3070"]);
        assert_eq!(hw.warnings, vec!["slow disk"]);
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
}
