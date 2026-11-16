//! Finds the graphics cards of this computer by itself, on Windows, Linux and macOS, with the
//! operating system's own tools. Read-only. colibri has its own report; this one also sees
//! cards colibri cannot use, so the app can say why a card sits idle.

use crate::proc::{hide_window, run_capture};
use serde_json::Value;
use std::process::Command;
use std::time::Duration;

const GB: f64 = 1_073_741_824.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: Vendor,
    /// Dedicated video memory, when the system tells it. `None` for a chip that shares main memory.
    pub vram_gb: Option<f64>,
    /// Shares main memory instead of having its own.
    pub integrated: bool,
}

pub fn vendor_of(name: &str) -> Vendor {
    let n = name.to_ascii_lowercase();
    if n.contains("nvidia") || n.contains("geforce") || n.contains("quadro") || n.contains("rtx ") {
        Vendor::Nvidia
    } else if n.contains("amd") || n.contains("radeon") || n.contains("ati ") {
        Vendor::Amd
    } else if n.contains("apple") {
        Vendor::Apple
    } else if n.contains("intel") || n.contains("uhd") || n.contains("iris") || n.contains("arc ") {
        Vendor::Intel
    } else {
        Vendor::Other
    }
}

/// A chip that borrows main memory: Apple silicon, Intel UHD/Iris/HD, AMD "Radeon Graphics"
/// or Vega without its own memory. Intel Arc and anything with several GB of its own is discrete.
pub fn looks_integrated(name: &str, vendor: Vendor, vram_gb: Option<f64>) -> bool {
    let n = name.to_ascii_lowercase();
    match vendor {
        Vendor::Apple => true,
        Vendor::Intel => !n.contains("arc"),
        Vendor::Nvidia => false,
        Vendor::Amd | Vendor::Other => vram_gb.is_none_or(|v| v < 1.5),
    }
}

fn make(name: &str, vram_gb: Option<f64>) -> GpuInfo {
    let vendor = vendor_of(name);
    let vram_gb = vram_gb.filter(|v| *v > 0.05);
    GpuInfo {
        name: name.trim().to_string(),
        vendor,
        integrated: looks_integrated(name, vendor, vram_gb),
        vram_gb,
    }
}

/// Removes the virtual adapters that remote desktops and hypervisors add.
fn is_virtual(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    [
        "microsoft basic",
        "microsoft remote",
        "virtual",
        "vmware",
        "parsec",
        "displaylink",
        "meta virtual",
        "indirect display",
    ]
    .iter()
    .any(|k| n.contains(k))
}

// ---------------------------------------------------------------- Windows

/// One JSON array of `{Name, Mem}` objects. `Mem` comes from the driver's registry entry, which
/// is 64-bit; the usual `AdapterRAM` stops at 4 GB.
pub const WINDOWS_SCRIPT: &str = r#"
$ErrorActionPreference='SilentlyContinue'
$o=@()
$base='HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
foreach($k in (Get-ChildItem $base)){
  $p=Get-ItemProperty $k.PSPath
  if($p.DriverDesc){
    $m=$p.'HardwareInformation.qwMemorySize'
    if(-not $m){$m=$p.'HardwareInformation.MemorySize'}
    $n=0.0
    if($m -is [byte[]]){ $b=New-Object byte[] 8; [Array]::Copy($m,$b,[Math]::Min(8,$m.Length)); $n=[double][BitConverter]::ToUInt64($b,0) }
    elseif($m){ $n=[double]$m }
    $o+=[pscustomobject]@{Name=[string]$p.DriverDesc;Mem=$n}
  }
}
if($o.Count -eq 0){
  foreach($c in (Get-CimInstance Win32_VideoController)){ $o+=[pscustomobject]@{Name=[string]$c.Name;Mem=[double]$c.AdapterRAM} }
}
ConvertTo-Json -InputObject @($o) -Compress
"#;

pub fn parse_windows(json: &str) -> Vec<GpuInfo> {
    let Ok(v) = serde_json::from_str::<Value>(json.trim()) else {
        return Vec::new();
    };
    let list = match v {
        Value::Array(a) => a,
        o @ Value::Object(_) => vec![o],
        _ => return Vec::new(),
    };
    let mut out: Vec<GpuInfo> = Vec::new();
    for g in list {
        let Some(name) = g.get("Name").and_then(Value::as_str) else {
            continue;
        };
        if name.trim().is_empty() || is_virtual(name) {
            continue;
        }
        let mem = g.get("Mem").and_then(Value::as_f64).map(|b| b / GB);
        out.push(make(name, mem));
    }
    dedupe(out)
}

// ---------------------------------------------------------------- Linux

/// `lspci -mm` lines for display adapters: `00:02.0 "VGA compatible controller" "Intel" "..."`.
pub fn parse_lspci(text: &str) -> Vec<GpuInfo> {
    let mut out = Vec::new();
    for line in text.lines() {
        let q: Vec<&str> = line.split('"').collect();
        // q = [addr, class, _, vendor, _, device, ...]
        if q.len() < 6 {
            continue;
        }
        let class = q[1].to_ascii_lowercase();
        if !(class.contains("vga") || class.contains("3d controller") || class.contains("display"))
        {
            continue;
        }
        let name = format!("{} {}", q[3].trim(), q[5].trim());
        if !is_virtual(&name) {
            out.push(make(&name, None));
        }
    }
    dedupe(out)
}

/// Dedicated memory of AMD/Intel cards that expose it in sysfs (`mem_info_vram_total`).
#[cfg(target_os = "linux")]
fn sysfs_vram() -> Vec<f64> {
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/sys/class/drm") {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if n.starts_with("card") && !n.contains('-') {
                if let Ok(t) = std::fs::read_to_string(e.path().join("device/mem_info_vram_total"))
                {
                    if let Ok(b) = t.trim().parse::<f64>() {
                        v.push(b / GB);
                    }
                }
            }
        }
    }
    v
}

// ---------------------------------------------------------------- macOS

pub fn parse_macos(json: &str) -> Vec<GpuInfo> {
    let Ok(v) = serde_json::from_str::<Value>(json) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for g in v
        .get("SPDisplaysDataType")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = g
            .get("sppci_model")
            .or_else(|| g.get("_name"))
            .and_then(Value::as_str);
        if let Some(name) = name {
            let vram = ["spdisplays_vram", "spdisplays_vram_shared"]
                .iter()
                .find_map(|k| g.get(*k).and_then(Value::as_str))
                .and_then(parse_size_gb);
            out.push(make(name, vram));
        }
    }
    dedupe(out)
}

/// "8 GB", "1536 MB" → gigabytes.
fn parse_size_gb(s: &str) -> Option<f64> {
    let mut it = s.split_whitespace();
    let n: f64 = it.next()?.parse().ok()?;
    match it.next()?.to_ascii_uppercase().as_str() {
        "GB" => Some(n),
        "MB" => Some(n / 1024.0),
        _ => None,
    }
}

// ---------------------------------------------------------------- shared

fn dedupe(mut v: Vec<GpuInfo>) -> Vec<GpuInfo> {
    let mut seen: Vec<String> = Vec::new();
    v.retain(|g| {
        let k = g.name.to_ascii_lowercase();
        if seen.contains(&k) {
            false
        } else {
            seen.push(k);
            true
        }
    });
    v
}

fn capture(mut c: Command, secs: u64) -> Option<String> {
    hide_window(&mut c);
    match run_capture(c, Duration::from_secs(secs)) {
        Ok(o) if o.status.success() => Some(String::from_utf8_lossy(&o.stdout).into_owned()),
        _ => None,
    }
}

/// `nvidia-smi` knows the exact memory of NVIDIA cards, so it overrides the guess for those.
fn nvidia_memory() -> Vec<(String, f64)> {
    let mut c = Command::new("nvidia-smi");
    c.args([
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]);
    capture(c, 5)
        .map(|t| {
            t.lines()
                .filter_map(|l| {
                    let (n, m) = l.split_once(',')?;
                    Some((
                        n.trim().to_string(),
                        m.trim().parse::<f64>().ok()? * 1_048_576.0 / GB,
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Every graphics card the operating system reports, best first (dedicated, most memory).
pub fn detect() -> Vec<GpuInfo> {
    let mut list = detect_os();
    for (name, gb) in nvidia_memory() {
        match list.iter_mut().find(|g| {
            g.vendor == Vendor::Nvidia
                && (g.name.eq_ignore_ascii_case(&name)
                    || name
                        .to_ascii_lowercase()
                        .contains(&g.name.to_ascii_lowercase()))
        }) {
            Some(g) => {
                g.vram_gb = Some(gb);
                g.integrated = false;
            }
            None => list.push(make(&name, Some(gb))),
        }
    }
    list.sort_by(|a, b| {
        a.integrated.cmp(&b.integrated).then(
            b.vram_gb
                .unwrap_or(0.0)
                .partial_cmp(&a.vram_gb.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    list
}

#[cfg(target_os = "windows")]
fn detect_os() -> Vec<GpuInfo> {
    let mut c = Command::new("powershell");
    c.args(["-NoProfile", "-NonInteractive", "-Command", WINDOWS_SCRIPT]);
    capture(c, 15)
        .map(|t| parse_windows(&t))
        .unwrap_or_default()
}

#[cfg(target_os = "macos")]
fn detect_os() -> Vec<GpuInfo> {
    let mut c = Command::new("system_profiler");
    c.args(["SPDisplaysDataType", "-json"]);
    capture(c, 15).map(|t| parse_macos(&t)).unwrap_or_default()
}

#[cfg(target_os = "linux")]
fn detect_os() -> Vec<GpuInfo> {
    let mut c = Command::new("lspci");
    c.arg("-mm");
    let mut list = capture(c, 5).map(|t| parse_lspci(&t)).unwrap_or_default();
    let mem = sysfs_vram();
    // sysfs cards come in the same order for AMD/Intel; give each discrete non-NVIDIA card its memory
    let mut it = mem.into_iter().filter(|m| *m > 0.05);
    for g in list.iter_mut().filter(|g| g.vendor != Vendor::Nvidia) {
        if let Some(m) = it.next() {
            *g = make(&g.name, Some(m));
        }
    }
    list
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn detect_os() -> Vec<GpuInfo> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_report_with_discrete_and_integrated() {
        let j = r#"[{"Name":"NVIDIA GeForce RTX 4070","Mem":12884901888},
                    {"Name":"Intel(R) UHD Graphics 770","Mem":0},
                    {"Name":"Microsoft Basic Display Adapter","Mem":0}]"#;
        let g = parse_windows(j);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].vendor, Vendor::Nvidia);
        assert!((g[0].vram_gb.unwrap() - 12.0).abs() < 0.01);
        assert!(!g[0].integrated);
        assert_eq!(g[1].vendor, Vendor::Intel);
        assert!(g[1].integrated && g[1].vram_gb.is_none());
    }

    #[test]
    fn windows_single_object_and_garbage() {
        assert_eq!(
            parse_windows(r#"{"Name":"AMD Radeon RX 6700 XT","Mem":12884901888}"#).len(),
            1
        );
        assert!(parse_windows("not json").is_empty());
        assert!(parse_windows("").is_empty());
    }

    #[test]
    fn amd_without_memory_is_integrated() {
        let g = parse_windows(r#"[{"Name":"AMD Radeon(TM) Graphics","Mem":536870912}]"#);
        assert!(g[0].integrated);
        let d = parse_windows(r#"[{"Name":"AMD Radeon RX 7800 XT","Mem":17179869184}]"#);
        assert!(!d[0].integrated);
    }

    #[test]
    fn intel_arc_is_discrete() {
        assert!(!looks_integrated(
            "Intel(R) Arc(TM) A770 Graphics",
            Vendor::Intel,
            Some(16.0)
        ));
        assert!(looks_integrated(
            "Intel(R) Iris(R) Xe Graphics",
            Vendor::Intel,
            None
        ));
    }

    #[test]
    fn lspci_lines() {
        let t = "00:02.0 \"VGA compatible controller\" \"Intel Corporation\" \"Raptor Lake-S GT1 [UHD Graphics 770]\" -r04 \"ASRock\" \"x\"\n\
                 01:00.0 \"VGA compatible controller\" \"NVIDIA Corporation\" \"AD104 [GeForce RTX 4070]\" -ra1 \"x\" \"y\"\n\
                 00:1f.3 \"Audio device\" \"Intel Corporation\" \"x\" \"y\"\n";
        let g = parse_lspci(t);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].vendor, Vendor::Intel);
        assert_eq!(g[1].vendor, Vendor::Nvidia);
    }

    #[test]
    fn macos_apple_silicon_and_sizes() {
        let j = r#"{"SPDisplaysDataType":[{"sppci_model":"Apple M2 Pro","spdisplays_vram_shared":"16 GB"}]}"#;
        let g = parse_macos(j);
        assert_eq!(g[0].vendor, Vendor::Apple);
        assert!(g[0].integrated);
        assert_eq!(parse_size_gb("1536 MB"), Some(1.5));
        assert_eq!(parse_size_gb("nonsense"), None);
    }

    #[test]
    fn duplicates_collapse() {
        let g = parse_windows(r#"[{"Name":"Intel UHD","Mem":0},{"Name":"intel uhd","Mem":0}]"#);
        assert_eq!(g.len(), 1);
    }
}
