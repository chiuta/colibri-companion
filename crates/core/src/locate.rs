//! Finding the two things colibri needs on this machine: Python (>= 3.10) and a
//! copy of the colibri repository.

use crate::paths::home_dir;
use crate::proc::run_capture;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

pub const COLIBRI_GIT_URL: &str = "https://github.com/JustVugg/colibri";
pub const COLIBRI_ZIP_URL: &str = "https://github.com/JustVugg/colibri/archive/refs/heads/main.zip";
pub const PYTHON_DOWNLOAD_URL: &str = "https://www.python.org/downloads/";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PythonCmd {
    pub program: String,
    pub pre_args: Vec<String>,
    pub version: (u32, u32),
}

impl PythonCmd {
    pub fn command(&self) -> Command {
        let mut c = Command::new(&self.program);
        c.args(&self.pre_args);
        c
    }
    pub fn display(&self) -> String {
        let mut s = self.program.clone();
        for a in &self.pre_args {
            s.push(' ');
            s.push_str(a);
        }
        s
    }
}

/// Parse "3 12" as printed by the probe script.
pub fn parse_version(text: &str) -> Option<(u32, u32)> {
    let mut it = text.split_whitespace();
    let major = it.next()?.parse().ok()?;
    let minor = it.next()?.parse().ok()?;
    Some((major, minor))
}

pub fn find_python() -> Option<PythonCmd> {
    let candidates: [(&str, &[&str]); 3] = [("python3", &[]), ("python", &[]), ("py", &["-3"])];
    for (program, pre) in candidates {
        let mut cmd = Command::new(program);
        cmd.args(pre);
        cmd.args([
            "-c",
            "import sys; print(sys.version_info[0], sys.version_info[1])",
        ]);
        let Ok(out) = run_capture(cmd, Duration::from_secs(10)) else {
            continue;
        };
        if !out.status.success() {
            continue;
        }
        let Some(version) = parse_version(&String::from_utf8_lossy(&out.stdout)) else {
            continue;
        };
        if version >= (3, 10) {
            return Some(PythonCmd {
                program: program.to_string(),
                pre_args: pre.iter().map(|s| s.to_string()).collect(),
                version,
            });
        }
    }
    None
}

/// A folder is a colibri checkout when the launcher and the setup code are there.
pub fn is_colibri_root(path: &Path) -> bool {
    path.join("c").join("coli").is_file() && path.join("c").join("setup_flow.py").is_file()
}

/// If the user picked a sub-folder (for example `colibri/c`), step up to the root.
pub fn normalize_root(path: &Path) -> Option<PathBuf> {
    if is_colibri_root(path) {
        return Some(path.to_path_buf());
    }
    let parent = path.parent()?;
    if path.file_name().is_some_and(|n| n == "c") && is_colibri_root(parent) {
        return Some(parent.to_path_buf());
    }
    // A zip unpacked into a folder of the same name: colibri-main/colibri-main/
    let mut inner = Vec::new();
    if let Ok(rd) = std::fs::read_dir(path) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() && is_colibri_root(&p) {
                inner.push(p);
            }
        }
    }
    if inner.len() == 1 { inner.pop() } else { None }
}

/// Places where a colibri checkout usually is, most specific first.
pub fn candidate_roots(hint: Option<&Path>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(h) = hint {
        out.push(h.to_path_buf());
    }
    if let Some(v) = env::var_os("COLIBRI_HOME") {
        out.push(PathBuf::from(v));
    }
    if let Ok(cwd) = env::current_dir() {
        out.extend(cwd.ancestors().take(4).map(Path::to_path_buf));
    }
    if let Ok(exe) = env::current_exe() {
        out.extend(exe.ancestors().skip(1).take(5).map(Path::to_path_buf));
    }
    if let Some(home) = home_dir() {
        for rel in [
            "colibri",
            "colibri-main",
            "Documents/colibri",
            "Desktop/colibri",
            "src/colibri",
        ] {
            out.push(home.join(rel));
        }
        // An unpacked GitHub zip usually lands in Downloads as colibri-main.
        for folder in ["Downloads", "Desktop", "Documents"] {
            if let Ok(rd) = std::fs::read_dir(home.join(folder)) {
                for entry in rd.flatten() {
                    let name = entry.file_name().to_string_lossy().to_lowercase();
                    if name.starts_with("colibri") && entry.path().is_dir() {
                        out.push(entry.path());
                    }
                }
            }
        }
    }
    out
}

pub fn find_colibri(hint: Option<&Path>) -> Option<PathBuf> {
    candidate_roots(hint).iter().find_map(|c| normalize_root(c))
}

/// Where "Download colibri for me" puts it.
pub fn default_clone_dir() -> PathBuf {
    home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("colibri")
}

/// A folder name for a fresh copy of colibri that does not exist yet: `~/colibri`, or
/// `~/colibri-2`, `~/colibri-3`… when something already sits there.
pub fn fresh_clone_dir() -> PathBuf {
    let base = default_clone_dir();
    if !base.exists() {
        return base;
    }
    (2..)
        .map(|n| base.with_file_name(format!("colibri-{n}")))
        .find(|p| !p.exists())
        .unwrap_or(base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp(name: &str) -> PathBuf {
        let p = env::temp_dir().join(format!("cc-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn fake_root(p: &Path) {
        fs::create_dir_all(p.join("c")).unwrap();
        fs::write(p.join("c").join("coli"), "#!/usr/bin/env python3\n").unwrap();
        fs::write(p.join("c").join("setup_flow.py"), "").unwrap();
    }

    #[test]
    fn version_parsing() {
        assert_eq!(parse_version("3 12\n"), Some((3, 12)));
        assert_eq!(parse_version("garbage"), None);
    }

    #[test]
    fn recognises_root_subfolder_and_nested_zip() {
        let base = temp("root");
        let root = base.join("colibri-main");
        fake_root(&root);
        assert!(is_colibri_root(&root));
        assert_eq!(normalize_root(&root.join("c")), Some(root.clone()));
        // Folder that merely contains the checkout (zip unpacked into its own folder).
        let wrapper = base.join("wrapper");
        fs::create_dir_all(&wrapper).unwrap();
        fake_root(&wrapper.join("colibri-main"));
        assert_eq!(normalize_root(&wrapper), Some(wrapper.join("colibri-main")));
        assert_eq!(normalize_root(&base.join("nope")), None);
        let _ = fs::remove_dir_all(&base);
    }
}
