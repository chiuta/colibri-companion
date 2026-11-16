//! Small process helpers: run a command with a timeout, hide console windows on
//! Windows, signal a process by pid, open a URL in the default browser.

use std::io::{self, Read};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// Keep a console window from flashing up when a GUI program spawns a child.
pub fn hide_window(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    {
        let _ = cmd;
    }
}

/// Run `cmd` to completion, killing it when `timeout` passes. Output is captured. On a timeout
/// the helpers it started are ended too (on Unix through its process group).
pub fn run_capture(cmd: Command, timeout: Duration) -> io::Result<Output> {
    run_capture_opts(cmd, None, timeout, true)
}

/// [`run_capture`] for a program that is meant to leave something running (`coli start
/// --background`): on a timeout only the program itself is ended, never a server it launched.
pub fn run_capture_keep(cmd: Command, timeout: Duration) -> io::Result<Output> {
    run_capture_opts(cmd, None, timeout, false)
}

/// Most bytes kept from one stream of a captured command; the rest is read and thrown away so
/// the program never blocks on a full pipe, and this one never grows without limit.
const CAPTURE_CAP: u64 = 64 * 1024 * 1024;
/// After the program has ended, how long to wait for its streams to close. A program that left a
/// background helper holding the pipes open (a daemon started with `--background`) must not hang us.
const STREAM_GRACE: Duration = Duration::from_millis(1500);

type Shared = std::sync::Arc<std::sync::Mutex<Vec<u8>>>;

fn drain<R: Read + Send + 'static>(stream: Option<R>) -> (thread::JoinHandle<()>, Shared) {
    let buf: Shared = Default::default();
    let sink = buf.clone();
    let handle = thread::spawn(move || {
        let Some(mut r) = stream else { return };
        let mut chunk = [0u8; 16 * 1024];
        let mut kept = 0u64;
        loop {
            let n = match r.read(&mut chunk) {
                Ok(n) => n,
                // a signal interrupted the read: not the end of the stream
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            };
            if n == 0 {
                break;
            }
            if kept < CAPTURE_CAP {
                let take = n.min((CAPTURE_CAP - kept) as usize);
                if let Ok(mut b) = sink.lock() {
                    b.extend_from_slice(&chunk[..take]);
                }
                kept += take as u64;
            }
        }
    });
    (handle, buf)
}

fn collect(handle: thread::JoinHandle<()>, buf: Shared, until: Instant) -> Vec<u8> {
    while !handle.is_finished() && Instant::now() < until {
        thread::sleep(Duration::from_millis(10));
    }
    // A reader still blocked here is left to end by itself when the pipe closes.
    buf.lock()
        .map(|mut b| std::mem::take(&mut *b))
        .unwrap_or_default()
}

/// Like `run_capture`, and feeds `input` to the program's standard input (from a separate
/// thread, so a program that does not read it cannot block this one).
pub fn run_capture_input(
    cmd: Command,
    input: Option<Vec<u8>>,
    timeout: Duration,
) -> io::Result<Output> {
    run_capture_opts(cmd, input, timeout, true)
}

fn run_capture_opts(
    mut cmd: Command,
    input: Option<Vec<u8>>,
    timeout: Duration,
    tree: bool,
) -> io::Result<Output> {
    hide_window(&mut cmd);
    #[cfg(unix)]
    if tree {
        // Its own process group, so a timeout can end the helpers it started too, not only the
        // program itself (a Python that spawned a child would otherwise be left running).
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.stdin(if input.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    })
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    if let (Some(data), Some(mut stdin)) = (input, child.stdin.take()) {
        thread::spawn(move || {
            use std::io::Write;
            let _ = stdin.write_all(&data);
        });
    }
    let (t_out, out) = drain(child.stdout.take());
    let (t_err, err) = drain(child.stderr.take());
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() > timeout {
            if tree {
                kill_group(&child);
            }
            let _ = child.kill();
            let _ = child.wait();
            // Not joined: a helper the program started may still hold the pipes open.
            drop((t_out, t_err));
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "the command took too long",
            ));
        }
        thread::sleep(Duration::from_millis(15));
    };
    let until = Instant::now() + STREAM_GRACE; // one grace period for both streams
    Ok(Output {
        status,
        stdout: collect(t_out, out, until),
        stderr: collect(t_err, err, until),
    })
}

/// A program of the operating system by its full path under `%SystemRoot%` (Windows), so a program
/// of the same name placed in the folder of this app or in the working folder is never the one
/// that runs. `rel` is relative to the Windows folder, e.g. `System32\\taskkill.exe`. When
/// `SystemRoot` is unknown or the file is not there, the plain name is used, as before.
#[cfg_attr(not(windows), allow(dead_code))]
pub fn system_exe(rel: &str) -> std::ffi::OsString {
    join_system_root(std::env::var_os("SystemRoot").as_deref(), rel)
}

#[cfg_attr(not(windows), allow(dead_code))]
fn join_system_root(root: Option<&std::ffi::OsStr>, rel: &str) -> std::ffi::OsString {
    let plain = rel.rsplit(['\\', '/']).next().unwrap_or(rel);
    match root {
        Some(r) if !r.is_empty() => {
            let full = std::path::Path::new(r).join(rel.replace('\\', "/"));
            if full.is_file() {
                full.into_os_string()
            } else {
                plain.into()
            }
        }
        _ => plain.into(),
    }
}

/// Ends the process group `child` leads (see `process_group(0)` above). Only ever called for a
/// child that was started as its own group leader, so the pid is that group's id.
#[cfg(unix)]
fn kill_group(child: &std::process::Child) {
    let pid = child.id();
    if (2..=i32::MAX as u32).contains(&pid) {
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}
#[cfg(not(unix))]
fn kill_group(_child: &std::process::Child) {}

/// Ask a process to stop the polite way (SIGINT on Unix, which Python turns into
/// `KeyboardInterrupt`, so colibri's setup records "interrupted" and can resume).
/// On Windows there is no polite signal for a hidden console process, so the
/// process tree is ended; the setup resumes from the bytes already on disk anyway.
pub fn interrupt_pid(pid: u32) -> io::Result<()> {
    // 0 would be this program's own process group, 1 is init, and anything above i32::MAX turns
    // into -1 inside `kill`, which means "every process I may signal". A pid read from a state
    // file is never trusted into those.
    if !(2..=i32::MAX as u32).contains(&pid) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "not a usable process id",
        ));
    }
    #[cfg(unix)]
    let mut cmd = {
        let mut c = Command::new("kill");
        c.args(["-INT", &pid.to_string()]);
        c
    };
    #[cfg(windows)]
    let mut cmd = {
        let mut c = Command::new(system_exe(r"System32\taskkill.exe"));
        c.args(["/PID", &pid.to_string(), "/T", "/F"]);
        c
    };
    hide_window(&mut cmd);
    cmd.stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|_| ())
}

/// File types that would run a program, install something or start a macro when "opened". Never
/// handed to the system's opener.
const RUNNABLE: &[&str] = &[
    "exe",
    "bat",
    "cmd",
    "com",
    "scr",
    "msi",
    "msp",
    "lnk",
    "pif",
    "ps1",
    "psm1",
    "ps1xml",
    "ps2",
    "vbs",
    "vbe",
    "vb",
    "js",
    "jse",
    "wsf",
    "wsh",
    "ws",
    "hta",
    "jar",
    "reg",
    "dll",
    "cpl",
    "sh",
    "command",
    "app",
    "desktop",
    "appimage",
    "run",
    "bin",
    "py",
    "pyw",
    "pyz",
    "pl",
    "rb",
    "url",
    "chm",
    "inf",
    "msc",
    "scf",
    "jnlp",
    "gadget",
    "msh",
    "msh1",
    "msh2",
    "mshxml",
    "application",
    "appref-ms",
    "settingcontent-ms",
    "library-ms",
    "docm",
    "dotm",
    "xlsm",
    "xlam",
    "xll",
    "wll",
    "pptm",
    "potm",
    "ppam",
    "ppsm",
    "sldm",
    "appx",
    "msix",
    "appxbundle",
    "msixbundle",
    "apk",
    "pkg",
    "dmg",
    "deb",
    "rpm",
    "war",
];

/// The longest address or path handed to the opener.
const MAX_TARGET: usize = 8192;

/// Is this a path component Windows would read as something else? A trailing dot or space is
/// dropped there (`evil.exe.` is `evil.exe`), and a colon names an alternate data stream.
fn odd_windows_component(path: &std::path::Path) -> bool {
    path.components().any(|c| match c {
        std::path::Component::Normal(n) => n
            .to_str()
            .is_none_or(|n| n.ends_with(['.', ' ']) || n.contains(':')),
        _ => false,
    })
}

/// `\\?\C:\dir` (what `canonicalize` returns on Windows) as `C:\dir`; other forms stay as they are.
/// The Explorer is handed the plain form.
fn strip_verbatim(p: &str) -> &str {
    match p.strip_prefix(r"\\?\") {
        Some(rest) if rest.as_bytes().get(1) == Some(&b':') => rest,
        _ => p,
    }
}

/// What to hand to the system's opener for `target`, or `None` when it must not be opened: a web
/// address (http, https or mailto) without control characters, quotes, angle brackets or other
/// characters a Windows command line gives a meaning, or a path that exists, is a folder or a
/// document, and is given in its resolved, absolute form (so it can never be read as an option).
/// Other schemes (`ms-msdt:`, `file:`, `javascript:`…), a leading `-`, network (UNC) paths,
/// programs, scripts, shortcuts and anything with an execute bit are refused: addresses can come
/// from web pages and catalogues, paths from text a model wrote.
fn resolve_target(target: &str) -> Option<String> {
    let t = target.trim();
    if t.is_empty()
        || t.len() > MAX_TARGET
        || t.starts_with('-')
        || t.chars().any(|c| c.is_control())
    {
        return None;
    }
    let lower = t.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("mailto:")
    {
        let bad = |c: char| c.is_whitespace() || "\"<>\\^|`".contains(c);
        return (!t.chars().any(bad)).then(|| t.to_string());
    }
    if t.starts_with("\\\\") || t.starts_with("//") {
        return None; // a network share or a device path
    }
    let given = std::path::Path::new(t);
    if cfg!(windows) && odd_windows_component(given) {
        return None;
    }
    // the real file, after links and `..`: its name and mode are what counts
    let real = std::fs::canonicalize(given).ok()?;
    let text = real.to_str()?;
    let text = strip_verbatim(text);
    if text.starts_with("\\\\") {
        return None; // resolved onto a network share
    }
    if real.is_dir() {
        return Some(text.to_string());
    }
    let ext_ok = |p: &std::path::Path| {
        !p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| RUNNABLE.contains(&e.to_ascii_lowercase().as_str()))
    };
    #[cfg(unix)]
    let executable = {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(&real).is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
    };
    #[cfg(not(unix))]
    let executable = false;
    (real.is_file() && ext_ok(given) && ext_ok(&real) && !executable).then(|| text.to_string())
}

/// Is this something it is safe to hand to the system's opener? See [`resolve_target`].
pub fn is_safe_target(target: &str) -> bool {
    resolve_target(target).is_some()
}

/// Open a URL (or a folder) with the system's default handler. Refuses anything that
/// [`is_safe_target`] does not accept; returns whether it was handed over.
pub fn open_externally(target: &str) -> bool {
    let Some(target) = resolve_target(target) else {
        return false;
    };
    let target = target.as_str();
    #[cfg(target_os = "linux")]
    let mut cmd = {
        let mut c = Command::new("xdg-open");
        c.arg(target);
        c
    };
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = Command::new("open");
        c.arg(target);
        c
    };
    // Not `cmd /C start`: that runs the text through the command interpreter, where `&` ends the
    // address and starts another command. The Explorer takes the argument as it is.
    #[cfg(windows)]
    let mut cmd = {
        let mut c = Command::new(system_exe("explorer.exe"));
        c.arg(target);
        c
    };
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    let mut cmd = Command::new("true");
    hide_window(&mut cmd);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|mut child| {
            // Reaped in the background so no zombie is left behind.
            thread::spawn(move || {
                let _ = child.wait();
            });
        })
        .is_ok()
}

/// The file a bare command name stands for on Windows, where `npx` is really `npx.cmd`: Rust only
/// tries `.exe` there, so tools installed as scripts (`npx`, `uvx`, `npm`) were "not found". Each
/// folder of `path` is tried with each extension of `pathext` (default `.COM;.EXE;.BAT;.CMD`).
/// A command that already has a folder in it, or an extension, is left alone (`None`).
pub fn resolve_program(
    command: &str,
    path: Option<&std::ffi::OsStr>,
    pathext: Option<&str>,
) -> Option<std::path::PathBuf> {
    if command.is_empty() || command.contains(['/', '\\', ':']) {
        return None;
    }
    if std::path::Path::new(command).extension().is_some() {
        return None;
    }
    let exts: Vec<String> = pathext
        .filter(|e| !e.trim().is_empty())
        .unwrap_or(".COM;.EXE;.BAT;.CMD")
        .split(';')
        .filter(|e| e.starts_with('.') && e.len() > 1)
        .map(str::to_string)
        .collect();
    for dir in std::env::split_paths(path?) {
        for ext in &exts {
            let candidate = dir.join(format!("{command}{ext}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Is a program on the PATH? (Runs `<program> --version`.)
pub fn program_exists(program: &str) -> bool {
    let mut cmd = Command::new(program);
    cmd.arg("--version");
    run_capture(cmd, Duration::from_secs(8))
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(unix)]
    fn captures_output_and_times_out() {
        let mut c = Command::new("sh");
        c.args(["-c", "echo hello; echo oops 1>&2"]);
        let out = run_capture(c, Duration::from_secs(5)).unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hello");
        assert_eq!(String::from_utf8_lossy(&out.stderr).trim(), "oops");

        let mut slow = Command::new("sh");
        slow.args(["-c", "sleep 5"]);
        let err = run_capture(slow, Duration::from_millis(150)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    }

    #[test]
    #[cfg(unix)]
    fn a_helper_holding_the_pipes_cannot_hang_the_caller() {
        // `sh` ends at once, its background `sleep` keeps stdout open for 3 s.
        let mut c = Command::new("sh");
        c.args(["-c", "sleep 3 & echo started"]);
        let t = Instant::now();
        let out = run_capture(c, Duration::from_secs(10)).unwrap();
        assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "started");

        // and when the timeout fires while a helper holds the pipe
        let mut slow = Command::new("sh");
        slow.args(["-c", "sleep 3 & wait"]);
        let t = Instant::now();
        let err = run_capture(slow, Duration::from_millis(150)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
    }

    #[test]
    fn unusable_process_ids_are_refused() {
        for pid in [0, 1, u32::MAX, i32::MAX as u32 + 1] {
            assert_eq!(
                interrupt_pid(pid).unwrap_err().kind(),
                io::ErrorKind::InvalidInput,
                "{pid}"
            );
        }
    }

    #[test]
    fn only_addresses_documents_and_folders_are_opened() {
        assert!(is_safe_target("https://huggingface.co/x/y?a=1&b=2"));
        assert!(is_safe_target("mailto:a@b.ro"));
        assert!(!is_safe_target("https://x.ro/a b"));
        assert!(!is_safe_target("https://x.ro/\u{0}a"));
        for bad in [
            "",
            "-flag",
            "file:///etc/passwd",
            "ms-msdt:/id x",
            "javascript:alert(1)",
            "calc.exe",
            "http://a\nb",
        ] {
            assert!(!is_safe_target(bad), "{bad:?}");
        }
        let dir = std::env::temp_dir().join(format!("cc-open-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.md"), b"x").unwrap();
        std::fs::write(dir.join("run.sh"), b"x").unwrap();
        std::fs::write(dir.join("go.EXE"), b"x").unwrap();
        assert!(is_safe_target(&dir.display().to_string()));
        assert!(is_safe_target(&dir.join("a.md").display().to_string()));
        assert!(!is_safe_target(&dir.join("run.sh").display().to_string()));
        assert!(!is_safe_target(&dir.join("go.EXE").display().to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(unix)]
    fn a_timeout_ends_the_helpers_the_program_started_too() {
        let dir = std::env::temp_dir().join(format!("cc-grp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pidfile = dir.join("child.pid");
        let mut c = Command::new("sh");
        c.args([
            "-c",
            &format!("sleep 60 & echo $! > '{}'; wait", pidfile.display()),
        ]);
        let err = run_capture(c, Duration::from_millis(400)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        let pid: u32 = std::fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        // /proc is Linux-only; elsewhere `kill -0` says whether the process is still there
        let mut gone = false;
        for _ in 0..40 {
            let alive = Command::new("kill")
                .args(["-0", &pid.to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success());
            // a zombie waiting for its parent still answers: look at its state when we can
            let zombie = std::fs::read_to_string(format!("/proc/{pid}/stat"))
                .is_ok_and(|t| t.contains(") Z "));
            if !alive || zombie {
                gone = true;
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(gone, "the helper {pid} is still running");
    }

    #[test]
    fn an_interrupted_read_is_not_the_end_of_the_stream() {
        struct Flaky(u8);
        impl Read for Flaky {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                self.0 += 1;
                match self.0 {
                    1 | 3 => Err(io::Error::from(io::ErrorKind::Interrupted)),
                    2 => {
                        buf[..3].copy_from_slice(b"abc");
                        Ok(3)
                    }
                    4 => {
                        buf[..2].copy_from_slice(b"de");
                        Ok(2)
                    }
                    _ => Ok(0),
                }
            }
        }
        let (h, buf) = drain(Some(Flaky(0)));
        h.join().unwrap();
        assert_eq!(&*buf.lock().unwrap(), b"abcde");
    }

    #[test]
    fn what_the_opener_gets_is_resolved_absolute_and_never_a_program() {
        for bad in [
            "https://x.ro/a\"b",
            "https://x.ro/<b>",
            "https://x.ro/a\\b",
            "https://x.ro/a|b",
            "https://x.ro/a^b",
            "https://x.ro/`id`",
            r"\\server\share\a.md",
            "//server/share/a.md",
        ] {
            assert!(!is_safe_target(bad), "{bad:?}");
        }
        let long = format!("https://x.ro/{}", "a".repeat(MAX_TARGET));
        assert!(!is_safe_target(&long));
        let dir = std::env::temp_dir().join(format!("cc-open2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("a.md"), b"x").unwrap();
        std::fs::write(dir.join("short.url"), b"[InternetShortcut]").unwrap();
        std::fs::write(dir.join("macro.XLSM"), b"x").unwrap();
        // `..` is resolved, so what is opened is the real path
        let via = format!("{}/sub/../a.md", dir.display());
        let got = resolve_target(&via).unwrap();
        assert!(!got.contains(".."), "{got}");
        assert!(std::path::Path::new(&got).is_absolute());
        assert!(!is_safe_target(
            &dir.join("short.url").display().to_string()
        ));
        assert!(!is_safe_target(
            &dir.join("macro.XLSM").display().to_string()
        ));
        assert!(!is_safe_target(
            &dir.join("missing.md").display().to_string()
        ));
        #[cfg(unix)]
        {
            use std::os::unix::fs::{PermissionsExt, symlink};
            // a program with no extension, and a harmless-looking name that points at a script
            std::fs::write(dir.join("tool"), b"#!/bin/sh\n").unwrap();
            std::fs::set_permissions(dir.join("tool"), std::fs::Permissions::from_mode(0o755))
                .unwrap();
            assert!(!is_safe_target(&dir.join("tool").display().to_string()));
            std::fs::write(dir.join("run.sh"), b"x").unwrap();
            symlink(dir.join("run.sh"), dir.join("note.md")).unwrap();
            assert!(!is_safe_target(&dir.join("note.md").display().to_string()));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn windows_name_tricks_are_recognised_on_every_system() {
        use std::path::Path;
        assert!(odd_windows_component(Path::new("/a/evil.exe.")));
        assert!(odd_windows_component(Path::new("/a/evil.exe ")));
        assert!(odd_windows_component(Path::new("/a/note.md:evil.exe")));
        assert!(!odd_windows_component(Path::new("/a/b/note.md")));
        assert_eq!(strip_verbatim(r"\\?\C:\Users\a"), r"C:\Users\a");
        assert_eq!(strip_verbatim(r"\\?\UNC\srv\share"), r"\\?\UNC\srv\share");
        assert_eq!(strip_verbatim("/usr/bin"), "/usr/bin");
    }

    #[test]
    fn system_tools_come_from_the_windows_folder_when_it_has_them() {
        use std::ffi::OsStr;
        let root = std::env::temp_dir().join(format!("cc-sysroot-{}", std::process::id()));
        std::fs::create_dir_all(root.join("System32")).unwrap();
        std::fs::write(root.join("System32").join("taskkill.exe"), b"x").unwrap();
        let got = join_system_root(Some(root.as_os_str()), r"System32\taskkill.exe");
        assert_eq!(
            std::path::Path::new(&got),
            root.join("System32/taskkill.exe")
        );
        // a missing file, an unknown or empty root: the plain name, as before
        assert_eq!(
            join_system_root(Some(root.as_os_str()), "explorer.exe"),
            OsStr::new("explorer.exe")
        );
        assert_eq!(
            join_system_root(None, r"System32\taskkill.exe"),
            OsStr::new("taskkill.exe")
        );
        assert_eq!(
            join_system_root(Some(OsStr::new("")), "explorer.exe"),
            OsStr::new("explorer.exe")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_script_command_is_found_with_the_extensions_windows_tries() {
        let root = std::env::temp_dir().join(format!("cc-resolve-{}", std::process::id()));
        let (a, b) = (root.join("a"), root.join("b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(b.join("npx.cmd"), b"x").unwrap();
        std::fs::write(a.join("npx.txt"), b"x").unwrap();
        let path = std::env::join_paths([&a, &b]).unwrap();
        let got = resolve_program("npx", Some(&path), Some(".com;.exe;.cmd")).unwrap();
        assert_eq!(got, b.join("npx.cmd"));
        // not found, already has an extension or a folder, odd pathext, no PATH
        assert!(resolve_program("uvx", Some(&path), None).is_none());
        assert!(resolve_program("npx.cmd", Some(&path), None).is_none());
        assert!(resolve_program("sub/npx", Some(&path), None).is_none());
        assert!(resolve_program("C:npx", Some(&path), None).is_none());
        assert!(resolve_program("npx", None, None).is_none());
        assert_eq!(
            resolve_program("npx", Some(&path), Some(";;bad;.cmd")).unwrap(),
            b.join("npx.cmd")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[cfg(unix)]
    fn a_keep_timeout_leaves_a_server_the_program_started_running() {
        let dir = std::env::temp_dir().join(format!("cc-keep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pidfile = dir.join("child.pid");
        let mut c = Command::new("sh");
        c.args([
            "-c",
            &format!("sleep 5 & echo $! > '{}'; wait", pidfile.display()),
        ]);
        let _ = run_capture_keep(c, Duration::from_millis(300)).unwrap_err();
        let pid = std::fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .to_string();
        let alive = Command::new("kill")
            .args(["-0", &pid])
            .status()
            .is_ok_and(|s| s.success());
        let _ = Command::new("kill").args(["-KILL", &pid]).status();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(alive, "the launched server was ended by the timeout");
    }
}
