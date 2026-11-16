//! Is there a dedicated AI chip (NPU) in this computer? Only to tell the person; colibri's own
//! NPU support is experimental, Windows-only and AMD-only, and is switched on by a setting.

#[cfg(windows)]
use crate::proc::{hide_window, run_capture};
#[cfg(windows)]
use std::process::Command;
#[cfg(windows)]
use std::time::Duration;

/// The NPU's name when one is present.
pub fn detect() -> Option<String> {
    #[cfg(windows)]
    {
        let mut c = Command::new(crate::proc::system_exe(
            r"System32\WindowsPowerShell\v1.0\powershell.exe",
        ));
        c.args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | Where-Object { $_.FriendlyName -match 'NPU|Ryzen AI|IPU|Neural' } | Select-Object -First 1 -ExpandProperty FriendlyName",
        ]);
        hide_window(&mut c);
        let out = run_capture(c, Duration::from_secs(12)).ok()?;
        let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
        return (!name.is_empty()).then_some(name);
    }
    #[cfg(target_os = "linux")]
    {
        return linux_accel();
    }
    #[allow(unreachable_code)]
    None
}

#[cfg(target_os = "linux")]
fn linux_accel() -> Option<String> {
    let rd = std::fs::read_dir("/sys/class/accel").ok()?;
    let e = rd.flatten().next()?;
    let driver = std::fs::read_link(e.path().join("device").join("driver"))
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
    Some(match driver.as_deref() {
        Some("amdxdna") => "AMD XDNA NPU".to_string(),
        Some("intel_vpu") | Some("ivpu") => "Intel NPU".to_string(),
        Some(other) => format!("AI accelerator ({other})"),
        None => "AI accelerator".to_string(),
    })
}

/// Can colibri use it? Its XDNA path is for Windows and AMD only.
pub fn colibri_can_use(name: &str) -> bool {
    cfg!(windows) && (name.contains("Ryzen") || name.contains("XDNA") || name.contains("AMD"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_amd_on_windows_is_usable() {
        assert_eq!(colibri_can_use("AMD IPU Device"), cfg!(windows));
        assert!(!colibri_can_use("Intel(R) AI Boost"));
        // detection never panics and answers quickly, whatever this machine has
        let _ = detect();
    }
}
