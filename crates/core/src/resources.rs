//! A live picture of what this computer is spending on the model: processor, memory, graphics
//! card, disk, and what colibri's own processes use. Read-only; nothing is changed.

use crate::proc::{hide_window, run_capture};
use std::path::Path;
use std::process::Command;
use std::time::Duration;
use sysinfo::{Disks, Pid, ProcessesToUpdate, System};

const GB: f64 = 1_073_741_824.0;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuUse {
    pub name: String,
    pub util_percent: Option<f32>,
    pub vram_used_gb: f64,
    pub vram_total_gb: f64,
    pub temp_c: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProcUse {
    pub label: String,
    pub pid: u32,
    /// Percent of one core, summed over the process and its children (can pass 100).
    pub cpu_percent: f32,
    pub rss_gb: f64,
    pub processes: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    /// Whole-machine processor use, 0..100.
    pub cpu_percent: f32,
    pub cores: usize,
    pub ram_used_gb: f64,
    pub ram_total_gb: f64,
    pub swap_used_gb: f64,
    pub gpus: Vec<GpuUse>,
    pub disk_free_gb: Option<f64>,
    pub disk_total_gb: Option<f64>,
    pub procs: Vec<ProcUse>,
}

pub struct Sampler {
    sys: Option<System>,
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler {
    pub fn new() -> Self {
        let mut sys = System::new().ok();
        if let Some(s) = sys.as_mut() {
            s.refresh_cpu_usage(); // the first reading is a baseline; the next one is a real percentage
        }
        Self { sys }
    }

    /// `targets` are the root processes to report on (label, pid); each one includes its children.
    pub fn sample(&mut self, targets: &[(String, u32)], disk_path: Option<&Path>) -> Snapshot {
        let (disk_free_gb, disk_total_gb) = disk_path.map_or((None, None), disk_space);
        let gpus = nvidia_use();
        let Some(sys) = self.sys.as_mut() else {
            return Snapshot {
                gpus,
                disk_free_gb,
                disk_total_gb,
                ..Default::default()
            };
        };
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        sys.refresh_processes(ProcessesToUpdate::All, true);
        let sys = &*sys;
        let procs = targets
            .iter()
            .filter_map(|(label, root)| tree_use(sys, label, *root))
            .collect();
        Snapshot {
            cpu_percent: sys.global_cpu_usage(),
            cores: sys.cpus().len(),
            ram_used_gb: sys.used_memory() as f64 / GB,
            ram_total_gb: sys.total_memory() as f64 / GB,
            swap_used_gb: sys.used_swap() as f64 / GB,
            gpus,
            disk_free_gb,
            disk_total_gb,
            procs,
        }
    }

    /// The root process and every process descended from it.
    pub fn tree_pids(&self, root: u32) -> Vec<u32> {
        self.sys
            .as_ref()
            .map_or_else(|| vec![root], |s| tree_pids(s, root))
    }
}

fn tree_pids(sys: &System, root: u32) -> Vec<u32> {
    let mut out = vec![root];
    let mut i = 0;
    while i < out.len() {
        let parent = Pid::from_u32(out[i]);
        for (pid, p) in sys.processes() {
            if p.parent() == Some(parent) && !out.contains(&pid.as_u32()) {
                out.push(pid.as_u32());
            }
        }
        i += 1;
    }
    out
}

fn tree_use(sys: &System, label: &str, root: u32) -> Option<ProcUse> {
    sys.process(Pid::from_u32(root))?;
    let (mut cpu, mut bytes, mut n) = (0.0f32, 0u64, 0usize);
    for pid in tree_pids(sys, root) {
        if let Some(p) = sys.process(Pid::from_u32(pid)) {
            cpu += p.cpu_usage();
            bytes += p.memory();
            n += 1;
        }
    }
    Some(ProcUse {
        label: label.to_string(),
        pid: root,
        cpu_percent: cpu,
        rss_gb: bytes as f64 / GB,
        processes: n,
    })
}

fn disk_space(path: &Path) -> (Option<f64>, Option<f64>) {
    let Ok(disks) = Disks::new_with_refreshed_list() else {
        return (None, None);
    };
    let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    // Windows spells a canonical path `\\?\C:\…`, which never starts with the mount point `C:\`
    let plain = path.to_string_lossy();
    let path = std::path::PathBuf::from(plain.strip_prefix(r"\\?\").unwrap_or(&plain));
    disks
        .iter()
        .filter(|d| path.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map_or((None, None), |d| {
            (
                Some(d.available_space() as f64 / GB),
                Some(d.total_space() as f64 / GB),
            )
        })
}

/// `nvidia-smi` is the one tool that reports a graphics card's load the same way everywhere.
fn nvidia_use() -> Vec<GpuUse> {
    let mut c = Command::new("nvidia-smi");
    c.args([
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]);
    hide_window(&mut c);
    match run_capture(c, Duration::from_secs(5)) {
        Ok(out) if out.status.success() => parse_nvidia_smi(&String::from_utf8_lossy(&out.stdout)),
        _ => Vec::new(),
    }
}

pub fn parse_nvidia_smi(text: &str) -> Vec<GpuUse> {
    text.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() < 5 {
                return None;
            }
            let mib = |s: &str| s.parse::<f64>().ok().map(|v| v * 1_048_576.0 / GB);
            Some(GpuUse {
                name: f[0].to_string(),
                util_percent: f[1].parse().ok(),
                vram_used_gb: mib(f[2]).unwrap_or(0.0),
                vram_total_gb: mib(f[3]).unwrap_or(0.0),
                temp_c: f[4].parse().ok(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_nvidia_smi() {
        let g = parse_nvidia_smi("NVIDIA GeForce RTX 4070, 37, 5120, 12282, 54\n");
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].util_percent, Some(37.0));
        assert!((g[0].vram_used_gb - 5.0).abs() < 0.01);
        assert!((g[0].vram_total_gb - 11.99).abs() < 0.01);
        assert_eq!(g[0].temp_c, Some(54.0));
        let na = parse_nvidia_smi("Card, [N/A], 100, 200, [N/A]\n");
        assert_eq!(na[0].util_percent, None);
        assert!(parse_nvidia_smi("garbage").is_empty());
    }

    #[test]
    fn samples_this_process() {
        let mut s = Sampler::new();
        std::thread::sleep(Duration::from_millis(300));
        let me = std::process::id();
        let snap = s.sample(&[("test".into(), me)], Some(&std::env::temp_dir()));
        assert!(snap.cores > 0 && snap.ram_total_gb > 0.0);
        assert_eq!(snap.procs.len(), 1);
        assert!(snap.procs[0].rss_gb > 0.0);
        assert!(snap.disk_free_gb.is_some());
        assert!(s.tree_pids(me).contains(&me));
    }
}
