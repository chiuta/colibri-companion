//! Running a model with corbul, Companion's own engine, for files colibri cannot import (a GGUF of
//! a family colibri does not implement, Gemma for one). corbul reads a `.gguf` directly.
//!
//! The program is looked for next to this one, then on the PATH. It listens on the loopback only,
//! on a free port, and is stopped when the guard is dropped (the app closes, another model starts).

use crate::proc::hide_window;
use std::fs;
use std::io;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

fn program_name() -> &'static str {
    if cfg!(windows) {
        "corbul.exe"
    } else {
        "corbul"
    }
}

/// `corbul` beside the running program, or on the PATH.
pub fn find() -> Option<PathBuf> {
    let name = program_name();
    let beside = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.join(name)))
        .filter(|p| p.is_file());
    beside.or_else(|| find_in(std::env::var_os("PATH").as_deref(), name))
}

/// The first `name` file in the directories of a PATH-style list.
pub fn find_in(path: Option<&std::ffi::OsStr>, name: &str) -> Option<PathBuf> {
    std::env::split_paths(path?)
        .map(|d| d.join(name))
        .find(|p| p.is_file())
}

/// A port nothing listens on right now (the loopback; another program could still take it before
/// corbul does, in which case corbul exits and the caller reports it).
pub fn free_port() -> io::Result<u16> {
    Ok(TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port())
}

/// The arguments of `corbul serve` for one model file. Never listens beyond the loopback.
pub fn serve_args(model: &Path, port: u16, packs: &Path) -> Vec<std::ffi::OsString> {
    // A relative name starting with "-" would be read as an option: always hand over a path.
    let model = if model.is_absolute() {
        model.to_path_buf()
    } else {
        Path::new(".").join(model)
    };
    let mut a: Vec<std::ffi::OsString> = vec!["serve".into(), model.into()];
    for (k, v) in [
        ("--host", "127.0.0.1".to_string()),
        ("--port", port.to_string()),
    ] {
        a.push(k.into());
        a.push(v.into());
    }
    a.push("--packs".into());
    a.push(packs.into());
    a
}

/// A running corbul. Dropping it stops the program.
pub struct Running {
    child: Child,
    pub port: u16,
}

impl Running {
    pub fn base(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
    /// `Some(exit code)` once it has stopped (`-1` when it was ended by a signal).
    pub fn exited(&mut self) -> Option<i32> {
        self.child
            .try_wait()
            .ok()
            .flatten()
            .map(|s| s.code().unwrap_or(-1))
    }
    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Start `corbul serve MODEL` on a free port. Its output goes to `log` (replaced each time).
pub fn start(bin: &Path, model: &Path, packs: &Path, log: &Path) -> io::Result<Running> {
    if let Some(d) = log.parent() {
        fs::create_dir_all(d)?;
    }
    fs::create_dir_all(packs)?;
    let port = free_port()?;
    let out = fs::File::create(log)?;
    let err = out.try_clone()?;
    let mut c = Command::new(bin);
    c.args(serve_args(model, port, packs))
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err);
    hide_window(&mut c);
    Ok(Running {
        child: c.spawn()?,
        port,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_program_on_a_path_list() {
        let d = std::env::temp_dir().join(format!("corbul-find-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("a")).unwrap();
        fs::create_dir_all(d.join("b")).unwrap();
        fs::write(d.join("b").join("tool"), b"x").unwrap();
        let list = std::env::join_paths([d.join("a"), d.join("b")]).unwrap();
        assert_eq!(find_in(Some(&list), "tool"), Some(d.join("b").join("tool")));
        assert_eq!(find_in(Some(&list), "nope"), None);
        assert_eq!(find_in(None, "tool"), None);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn serve_arguments_stay_on_the_loopback_and_keep_odd_paths_whole() {
        let a = serve_args(
            Path::new("/m/my model; rm -rf.gguf"),
            4321,
            Path::new("/p q"),
        );
        let s: Vec<String> = a.iter().map(|x| x.to_string_lossy().into_owned()).collect();
        assert_eq!(s[0], "serve");
        assert_eq!(s[1], "/m/my model; rm -rf.gguf");
        assert!(s.windows(2).any(|w| w == ["--host", "127.0.0.1"]));
        assert!(s.windows(2).any(|w| w == ["--port", "4321"]));
        assert!(s.windows(2).any(|w| w == ["--packs", "/p q"]));
        assert!(!s.iter().any(|x| x.contains("0.0.0.0")));
        let r = serve_args(Path::new("--port"), 1, Path::new("/p"));
        assert_eq!(
            r[1],
            std::ffi::OsString::from(Path::new(".").join("--port"))
        );
    }

    #[test]
    fn a_free_port_is_usable_and_a_program_that_ends_is_noticed_and_stopped() {
        let p = free_port().unwrap();
        assert!(p > 0);
        #[cfg(unix)]
        {
            let d = std::env::temp_dir().join(format!("corbul-run-{}", std::process::id()));
            let _ = fs::remove_dir_all(&d);
            fs::create_dir_all(&d).unwrap();
            let fake = d.join("corbul");
            fs::write(&fake, "#!/bin/sh\necho started $@\nexit 3\n").unwrap();
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
            let mut r = start(
                &fake,
                Path::new("/x.gguf"),
                &d.join("packs"),
                &d.join("l/log.txt"),
            )
            .unwrap();
            let mut code = None;
            for _ in 0..100 {
                code = r.exited();
                if code.is_some() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
            assert_eq!(code, Some(3));
            assert!(
                fs::read_to_string(d.join("l/log.txt"))
                    .unwrap()
                    .contains("--packs")
            );
            // a long-running one is killed when dropped
            fs::write(&fake, "#!/bin/sh\nsleep 60\n").unwrap();
            let r = start(
                &fake,
                Path::new("/x.gguf"),
                &d.join("packs"),
                &d.join("l/log2.txt"),
            )
            .unwrap();
            drop(r);
            let _ = fs::remove_dir_all(&d);
        }
    }
}
