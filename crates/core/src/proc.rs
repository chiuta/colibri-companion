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

/// Run `cmd` to completion, killing it when `timeout` passes. Output is captured.
pub fn run_capture(cmd: Command, timeout: Duration) -> io::Result<Output> {
    run_capture_input(cmd, None, timeout)
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
        while let Ok(n) = r.read(&mut chunk) {
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
    mut cmd: Command,
    input: Option<Vec<u8>>,
    timeout: Duration,
) -> io::Result<Output> {
    hide_window(&mut cmd);
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
        let mut c = Command::new("taskkill");
        c.args(["/PID", &pid.to_string(), "/T", "/F"]);
        c
    };
    hide_window(&mut cmd);
    cmd.stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|_| ())
}

/// File types that would run a program when "opened". Never handed to the system's opener.
const RUNNABLE: &[&str] = &[
    "exe", "bat", "cmd", "com", "scr", "msi", "msp", "lnk", "pif", "ps1", "psm1", "vbs", "vbe",
    "js", "jse", "wsf", "wsh", "hta", "jar", "reg", "dll", "cpl", "sh", "command", "app",
    "desktop", "appimage", "run", "bin", "py", "pl", "rb",
];

/// Is this something it is safe to hand to the system's opener? A web address (http, https or
/// mailto) without control characters, or a path that exists and is a folder or a document.
/// Other schemes (`ms-msdt:`, `file:`, `javascript:`…), a leading `-` (an option for the opener)
/// and programs are refused: addresses can come from web pages and catalogues.
pub fn is_safe_target(target: &str) -> bool {
    let t = target.trim();
    if t.is_empty() || t.starts_with('-') || t.chars().any(|c| c.is_control()) {
        return false;
    }
    let lower = t.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("mailto:")
    {
        return !t.contains(char::is_whitespace);
    }
    let path = std::path::Path::new(t);
    if path.is_dir() {
        return true;
    }
    path.is_file()
        && !path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| RUNNABLE.contains(&e.to_ascii_lowercase().as_str()))
}

/// Open a URL (or a folder) with the system's default handler. Refuses anything that
/// [`is_safe_target`] does not accept; returns whether it was handed over.
pub fn open_externally(target: &str) -> bool {
    if !is_safe_target(target) {
        return false;
    }
    let target = target.trim();
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
        let mut c = Command::new("explorer.exe");
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
}
