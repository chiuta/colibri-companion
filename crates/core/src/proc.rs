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
pub fn run_capture(mut cmd: Command, timeout: Duration) -> io::Result<Output> {
    hide_window(&mut cmd);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let mut out = child.stdout.take();
    let mut err = child.stderr.take();
    let t_out = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(o) = out.as_mut() {
            let _ = o.read_to_end(&mut buf);
        }
        buf
    });
    let t_err = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(e) = err.as_mut() {
            let _ = e.read_to_end(&mut buf);
        }
        buf
    });
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = t_out.join();
            let _ = t_err.join();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "the command took too long",
            ));
        }
        thread::sleep(Duration::from_millis(15));
    };
    Ok(Output {
        status,
        stdout: t_out.join().unwrap_or_default(),
        stderr: t_err.join().unwrap_or_default(),
    })
}

/// Ask a process to stop the polite way (SIGINT on Unix, which Python turns into
/// `KeyboardInterrupt`, so colibri's setup records "interrupted" and can resume).
/// On Windows there is no polite signal for a hidden console process, so the
/// process tree is ended; the setup resumes from the bytes already on disk anyway.
pub fn interrupt_pid(pid: u32) -> io::Result<()> {
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

/// Open a URL (or a folder) with the system's default handler.
pub fn open_externally(target: &str) {
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
    #[cfg(windows)]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.args(["/C", "start", "", target]);
        c
    };
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    let mut cmd = Command::new("true");
    hide_window(&mut cmd);
    let _ = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
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
}
