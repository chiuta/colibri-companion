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

/// How the shelf of context packs is kept (see `docs/CONTEXT-PACKS.md`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackOptions {
    /// Keep packs in F16: half the size, F16 rounding on every row.
    pub f16: bool,
    /// Compress the pack files (lossless).
    pub compress: bool,
}

/// The arguments of `corbul serve` for one model file. Never listens beyond the loopback.
pub fn serve_args(
    model: &Path,
    port: u16,
    packs: &Path,
    opts: PackOptions,
) -> Vec<std::ffi::OsString> {
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
    if opts.f16 {
        a.push("--packs-kv".into());
        a.push("f16".into());
    }
    if opts.compress {
        a.push("--packs-compress".into());
    }
    a
}

/// What `corbul kvfit` is asked, and the arguments that say it. A path that starts with `-` is
/// given as `./path`, so it cannot be read as an option.
pub fn kvfit_args(
    model: &Path,
    text: &Path,
    question: &str,
    budget: f64,
) -> Vec<std::ffi::OsString> {
    let safe = |p: &Path| -> std::ffi::OsString {
        if p.is_absolute() {
            p.into()
        } else {
            Path::new(".").join(p).into()
        }
    };
    vec![
        "kvfit".into(),
        safe(model),
        "--text".into(),
        safe(text),
        "--question".into(),
        question.into(),
        "--budget".into(),
        format!("{budget}").into(),
    ]
}

/// Run `corbul kvfit` and wait for it. Returns whether it succeeded and what it printed (at most
/// the last 8 KB of the two streams together, so a runaway program cannot fill the memory).
pub fn run_kvfit(
    bin: &Path,
    model: &Path,
    text: &Path,
    question: &str,
    budget: f64,
) -> io::Result<(bool, String)> {
    let mut c = Command::new(bin);
    c.args(kvfit_args(model, text, question, budget))
        .stdin(Stdio::null());
    hide_window(&mut c);
    let out = c.output()?;
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    if !out.stderr.is_empty() {
        s.push('\n');
        s.push_str(&String::from_utf8_lossy(&out.stderr));
    }
    let keep = 8 * 1024;
    if s.len() > keep {
        let mut cut = s.len() - keep;
        while !s.is_char_boundary(cut) {
            cut += 1;
        }
        s = s[cut..].to_string();
    }
    Ok((out.status.success(), s))
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
pub fn start(
    bin: &Path,
    model: &Path,
    packs: &Path,
    log: &Path,
    opts: PackOptions,
) -> io::Result<Running> {
    if let Some(d) = log.parent() {
        fs::create_dir_all(d)?;
    }
    fs::create_dir_all(packs)?;
    let port = free_port()?;
    let out = fs::File::create(log)?;
    let err = out.try_clone()?;
    let mut c = Command::new(bin);
    c.args(serve_args(model, port, packs, opts))
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
            PackOptions::default(),
        );
        let s: Vec<String> = a.iter().map(|x| x.to_string_lossy().into_owned()).collect();
        assert_eq!(s[0], "serve");
        assert_eq!(s[1], "/m/my model; rm -rf.gguf");
        assert!(s.windows(2).any(|w| w == ["--host", "127.0.0.1"]));
        assert!(s.windows(2).any(|w| w == ["--port", "4321"]));
        assert!(s.windows(2).any(|w| w == ["--packs", "/p q"]));
        assert!(!s.iter().any(|x| x.contains("0.0.0.0")));
        assert!(
            !s.iter()
                .any(|x| x.contains("packs-kv") || x.contains("compress"))
        );
        let r = serve_args(
            Path::new("--port"),
            1,
            Path::new("/p"),
            PackOptions::default(),
        );
        assert_eq!(
            r[1],
            std::ffi::OsString::from(Path::new(".").join("--port"))
        );
        let o = serve_args(
            Path::new("/m.gguf"),
            1,
            Path::new("/p"),
            PackOptions {
                f16: true,
                compress: true,
            },
        );
        let o: Vec<String> = o.iter().map(|x| x.to_string_lossy().into_owned()).collect();
        assert!(o.windows(2).any(|w| w == ["--packs-kv", "f16"]));
        assert!(o.iter().any(|x| x == "--packs-compress"));
    }

    #[test]
    fn kvfit_arguments_keep_odd_names_whole_and_the_program_output_is_capped() {
        let a = kvfit_args(
            Path::new("-m.gguf"),
            Path::new("/t x.txt"),
            "why; rm -rf /?",
            0.25,
        );
        let s: Vec<String> = a.iter().map(|x| x.to_string_lossy().into_owned()).collect();
        assert_eq!(s[0], "kvfit");
        assert_eq!(s[1], Path::new(".").join("-m.gguf").to_string_lossy());
        assert!(s.windows(2).any(|w| w == ["--text", "/t x.txt"]));
        assert!(s.windows(2).any(|w| w == ["--question", "why; rm -rf /?"]));
        assert!(s.windows(2).any(|w| w == ["--budget", "0.25"]));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let d = std::env::temp_dir().join(format!("corbul-kvfit-{}", std::process::id()));
            let _ = fs::remove_dir_all(&d);
            fs::create_dir_all(&d).unwrap();
            let fake = d.join("corbul");
            fs::write(
                &fake,
                "#!/bin/sh\nhead -c 100000 /dev/zero | tr '\\0' 'x'\necho done\nexit 0\n",
            )
            .unwrap();
            fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
            let (ok, out) = run_kvfit(&fake, Path::new("/m"), Path::new("/t"), "q", 0.5).unwrap();
            assert!(ok && out.len() <= 8 * 1024 && out.trim_end().ends_with("done"));
            fs::write(&fake, "#!/bin/sh\necho bad >&2\nexit 2\n").unwrap();
            let (ok, out) = run_kvfit(&fake, Path::new("/m"), Path::new("/t"), "q", 0.5).unwrap();
            assert!(!ok && out.contains("bad"));
            let _ = fs::remove_dir_all(&d);
        }
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
                PackOptions::default(),
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
                PackOptions::default(),
            )
            .unwrap();
            drop(r);
            let _ = fs::remove_dir_all(&d);
        }
    }
}
