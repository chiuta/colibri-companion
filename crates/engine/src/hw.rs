//! Hardware detection and tuning, std only.
//!
//! Every probe is failure tolerant (a missing file or tool gives `None`, never an error) and
//! time bounded. All text parsing lives in small pure functions that the tests feed with real
//! samples from x86, ARM, Apple and Windows machines, so the logic is verified without that hardware.
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
const PROBE_LIMIT: Duration = Duration::from_secs(3);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cpu {
    pub vendor: Option<String>,
    pub model: Option<String>,
    pub logical: usize,
    pub physical: Option<usize>,
    /// Hybrid machines (Intel P+E, ARM big.LITTLE, Apple P+E): logical CPUs of each class.
    pub hybrid: Option<Hybrid>,
    /// Instruction-set extensions the running CPU reports (`avx2`, `avx512f`, `neon`, `dotprod`, ...).
    pub isa: Vec<&'static str>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hybrid { pub performance: usize, pub efficiency: usize }

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Caches { pub l1d: Option<u64>, pub l2: Option<u64>, pub l3: Option<u64> }

/// Container limits (cgroup v1 and v2).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Limits { pub cpu_cores: Option<f64>, pub memory: Option<u64> }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuKind { Nvidia, Amd, Intel, Apple, Other }
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gpu { pub name: String, pub vram: Option<u64>, pub kind: GpuKind }

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DiskKind { Ssd, Hdd, #[default] Unknown }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThpMode { Always, Madvise, Never }

#[derive(Clone, Debug, Default, PartialEq)]
pub struct HardwareProfile {
    pub os: &'static str,
    pub arch: &'static str,
    pub cpu: Cpu,
    pub caches: Caches,
    pub total_ram: Option<u64>,
    pub available_ram: Option<u64>,
    pub limits: Limits,
    /// Threads the process may really run on (affinity and cgroup aware).
    pub usable_threads: usize,
    pub numa_nodes: usize,
    pub thp: Option<ThpMode>,
    pub gpus: Vec<Gpu>,
    /// CPU and GPU share one memory pool (Apple Silicon).
    pub unified_memory: bool,
    pub disk: DiskKind,
}

// ------------------------------------------------------------------ process helper

/// Run a program with a hard time limit; `None` on any failure, non-zero exit or timeout.
pub fn run_cmd(prog: &str, args: &[&str], limit: Duration) -> Option<String> {
    let mut child = Command::new(prog).args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
    let mut out = child.stdout.take()?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || { let mut b = Vec::new(); let _ = out.read_to_end(&mut b); let _ = tx.send(b); });
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(st)) => {
                if !st.success() { return None; }
                let b = rx.recv_timeout(Duration::from_millis(500)).ok()?;
                return Some(String::from_utf8_lossy(&b).into_owned());
            }
            Ok(None) => {
                if start.elapsed() > limit { let _ = child.kill(); let _ = child.wait(); return None; }
                std::thread::sleep(Duration::from_millis(4));
            }
            Err(_) => return None,
        }
    }
}
fn read(p: &str) -> Option<String> { std::fs::read_to_string(p).ok() }
fn clean(s: &str) -> String { s.replace(['\0', '\r'], "") }
fn num(s: &str) -> Option<u64> { s.trim().parse().ok() }

// ------------------------------------------------------------------ pure parsers

#[derive(Debug, Default, PartialEq)]
pub struct CpuInfo { pub vendor: Option<String>, pub model: Option<String>, pub logical: usize, pub physical: Option<usize>, pub parts: Vec<String>, pub features: Vec<String> }

/// `/proc/cpuinfo` (x86 and ARM layouts).
pub fn parse_cpuinfo(text: &str) -> CpuInfo {
    let mut ci = CpuInfo::default();
    let mut cores: HashSet<(String, String)> = HashSet::new();
    let (mut cpu_cores_per_socket, mut sockets) = (None::<usize>, HashSet::new());
    let (mut hw_model, mut got_features) = (None::<String>, false);
    for block in text.split("\n\n") {
        let (mut pid, mut cid, mut is_cpu) = (None::<String>, None::<String>, false);
        for line in block.lines() {
            let Some((k, v)) = line.split_once(':') else { continue };
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
            match k.as_str() {
                "processor" => { is_cpu = true; ci.logical += 1; }
                "vendor_id" if ci.vendor.is_none() => ci.vendor = Some(v.to_string()),
                "model name" if ci.model.is_none() && !v.is_empty() => ci.model = Some(v.to_string()),
                "hardware" | "model" if hw_model.is_none() && !v.is_empty() => hw_model = Some(v.to_string()),
                "physical id" => pid = Some(v.to_string()),
                "core id" => cid = Some(v.to_string()),
                "cpu cores" => cpu_cores_per_socket = num(v).map(|n| n as usize),
                "cpu part" => { if !ci.parts.iter().any(|p| p == v) { ci.parts.push(v.to_string()); } }
                "flags" | "features" if !got_features => { ci.features = v.split_whitespace().map(str::to_string).collect(); got_features = true; }
                _ => {}
            }
        }
        if is_cpu {
            if let Some(p) = &pid { sockets.insert(p.clone()); }
            if let (Some(p), Some(c)) = (pid, cid) { cores.insert((p, c)); }
        }
    }
    if ci.model.is_none() { ci.model = hw_model; }
    ci.physical = if !cores.is_empty() { Some(cores.len()) } else { cpu_cores_per_socket.map(|c| c * sockets.len().max(1)) };
    ci
}

/// `/proc/meminfo` → (total, available) in bytes.
pub fn parse_meminfo(text: &str) -> (Option<u64>, Option<u64>) {
    let get = |key: &str| text.lines().find_map(|l| { let (k, v) = l.split_once(':')?; (k == key).then(|| num(v.trim().trim_end_matches("kB"))).flatten().map(|kb| kb * 1024) });
    let total = get("MemTotal");
    let avail = get("MemAvailable").or_else(|| Some(get("MemFree")? + get("Cached").unwrap_or(0) + get("Buffers").unwrap_or(0)));
    (total, avail)
}

/// `0-3,8,10-11` → 7. `None` for anything malformed.
pub fn parse_cpulist(s: &str) -> Option<usize> {
    let s = s.trim();
    if s.is_empty() { return Some(0); }
    let mut n = 0usize;
    for part in s.split(',') {
        match part.split_once('-') {
            Some((a, b)) => { let (a, b): (usize, usize) = (a.trim().parse().ok()?, b.trim().parse().ok()?); if b < a { return None; } n = n.checked_add(b - a + 1)?; }
            None => { part.trim().parse::<usize>().ok()?; n += 1; }
        }
    }
    Some(n)
}

/// Number of distinct `thread_siblings_list` values = physical cores.
pub fn physical_from_sibling_lists(lists: &[String]) -> Option<usize> {
    let set: HashSet<&str> = lists.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    (!set.is_empty()).then_some(set.len())
}

/// ARM `cpu_capacity` values → (performance, efficiency) CPU counts; `None` when all cores are alike.
pub fn hybrid_from_capacities(caps: &[u32]) -> Option<Hybrid> {
    let max = *caps.iter().max()?;
    let min = *caps.iter().min()?;
    if max == min { return None; }
    let perf = caps.iter().filter(|&&c| c as u64 * 10 >= max as u64 * 6).count();
    Some(Hybrid { performance: perf, efficiency: caps.len() - perf })
}

/// `32K`, `1024K`, `8M`, `1G`, `512` (bytes).
pub fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let (digits, mult) = match s.chars().last()? { 'K' | 'k' => (&s[..s.len() - 1], 1024), 'M' | 'm' => (&s[..s.len() - 1], MIB), 'G' | 'g' => (&s[..s.len() - 1], GIB), _ => (s, 1) };
    num(digits)?.checked_mul(mult)
}

/// sysfs cache entries `(level, type, size)` → data/unified cache sizes per level.
pub fn caches_from_entries(entries: &[(u32, String, String)]) -> Caches {
    let mut c = Caches::default();
    for (level, kind, size) in entries {
        let kind = kind.trim().to_ascii_lowercase();
        if kind == "instruction" { continue; }
        let Some(sz) = parse_size(size) else { continue };
        match level { 1 => c.l1d = c.l1d.or(Some(sz)), 2 => c.l2 = c.l2.or(Some(sz)), 3 => c.l3 = c.l3.or(Some(sz)), _ => {} }
    }
    c
}

/// cgroup v2 `cpu.max`: `max 100000` or `200000 100000`.
pub fn parse_cgroup_v2_cpu_max(t: &str) -> Option<f64> {
    let mut it = t.split_whitespace();
    let (q, p) = (it.next()?, it.next()?);
    if q == "max" { return None; }
    let (q, p): (f64, f64) = (q.parse().ok()?, p.parse().ok()?);
    (p > 0.0 && q > 0.0).then_some(q / p)
}
/// cgroup v1 `cpu.cfs_quota_us` and `cpu.cfs_period_us`.
pub fn parse_cgroup_v1_cpu(quota: &str, period: &str) -> Option<f64> {
    let (q, p): (f64, f64) = (quota.trim().parse().ok()?, period.trim().parse().ok()?);
    (q > 0.0 && p > 0.0).then_some(q / p)
}
/// `memory.max` / `memory.limit_in_bytes`; "max" and the v1 "unlimited" sentinel give `None`.
pub fn parse_cgroup_mem(t: &str) -> Option<u64> {
    let v = num(t.trim())?;
    (v < (1 << 60)).then_some(v)
}
/// Path of this process in the unified hierarchy from `/proc/self/cgroup`.
pub fn parse_proc_self_cgroup(t: &str) -> Option<String> { t.lines().find_map(|l| l.strip_prefix("0::")).map(|p| p.trim().to_string()) }

pub fn parse_thp(t: &str) -> Option<ThpMode> {
    let w = t.split('[').nth(1)?.split(']').next()?;
    match w { "always" => Some(ThpMode::Always), "madvise" => Some(ThpMode::Madvise), "never" => Some(ThpMode::Never), _ => None }
}
/// NUMA nodes from directory names such as `node0`, `node1`, `possible`.
pub fn count_numa_nodes(names: &[String]) -> usize {
    names.iter().filter(|n| n.strip_prefix("node").is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))).count().max(1)
}

/// `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`
pub fn parse_nvidia_smi(t: &str) -> Vec<Gpu> {
    t.lines().filter_map(|l| {
        let (name, mem) = l.rsplit_once(',')?;
        let name = name.trim();
        if name.is_empty() { return None; }
        let vram = num(mem.trim().trim_end_matches("MiB").trim()).map(|m| m * MIB); // "[N/A]" on Jetson → None
        Some(Gpu { name: name.to_string(), vram, kind: GpuKind::Nvidia })
    }).collect()
}
/// `rocm-smi --showmeminfo vram --csv`
pub fn parse_rocm_smi(t: &str) -> Vec<Gpu> {
    let mut lines = t.lines();
    let Some(head) = lines.next() else { return vec![] };
    let Some(col) = head.split(',').position(|h| h.contains("VRAM Total Memory")) else { return vec![] };
    lines.filter_map(|l| {
        let f: Vec<&str> = l.split(',').collect();
        let id = f.first()?.trim();
        if !id.starts_with("card") { return None; }
        Some(Gpu { name: format!("AMD GPU ({id})"), vram: f.get(col).and_then(|v| num(v)), kind: GpuKind::Amd })
    }).collect()
}
pub fn gpu_kind_from_pci_vendor(v: &str) -> GpuKind {
    match v.trim().to_ascii_lowercase().as_str() { "0x10de" => GpuKind::Nvidia, "0x1002" => GpuKind::Amd, "0x8086" => GpuKind::Intel, _ => GpuKind::Other }
}
fn parse_gb_mb(v: &str) -> Option<u64> {
    let mut it = v.split_whitespace();
    let n: u64 = it.next()?.parse().ok()?;
    match it.next()?.to_ascii_uppercase().as_str() { "GB" => Some(n * GIB), "MB" => Some(n * MIB), _ => None }
}
/// macOS `system_profiler SPDisplaysDataType`
pub fn parse_system_profiler(t: &str) -> Vec<Gpu> {
    let mut gpus: Vec<Gpu> = vec![];
    for l in t.lines() {
        let l = l.trim();
        if let Some(n) = l.strip_prefix("Chipset Model:") {
            let name = n.trim().to_string();
            let kind = if name.contains("Apple") { GpuKind::Apple } else if name.contains("NVIDIA") { GpuKind::Nvidia } else if name.contains("AMD") || name.contains("Radeon") { GpuKind::Amd } else if name.contains("Intel") { GpuKind::Intel } else { GpuKind::Other };
            gpus.push(Gpu { name, vram: None, kind });
        } else if let Some(v) = l.strip_prefix("VRAM (Total):").or_else(|| l.strip_prefix("VRAM (Dynamic, Max):")) {
            if let Some(g) = gpus.last_mut() { g.vram = parse_gb_mb(v); }
        }
    }
    gpus
}
/// `sysctl a b c` output: `key: value` per line.
pub fn parse_sysctl_pairs(t: &str) -> HashMap<String, String> {
    t.lines().filter_map(|l| { let (k, v) = l.split_once(':').or_else(|| l.split_once('='))?; Some((k.trim().to_string(), v.trim().to_string())) }).collect()
}
/// `vm_stat` → approximately reclaimable bytes (free + inactive + speculative + purgeable).
pub fn parse_vm_stat(t: &str) -> Option<u64> {
    let page: u64 = t.lines().next()?.split("page size of").nth(1)?.split_whitespace().next()?.parse().ok()?;
    let pages = |key: &str| t.lines().find_map(|l| { let (k, v) = l.split_once(':')?; (k.trim() == key).then(|| num(v.trim().trim_end_matches('.'))).flatten() });
    Some((pages("Pages free")? + pages("Pages inactive").unwrap_or(0) + pages("Pages speculative").unwrap_or(0) + pages("Pages purgeable").unwrap_or(0)) * page)
}
/// `key=value` lines (wmic `/value`, or the PowerShell snippets below); UTF-16 NULs and CRs are removed.
pub fn parse_kv_lines(t: &str) -> Vec<(String, String)> {
    clean(t).lines().filter_map(|l| { let (k, v) = l.split_once('=')?; let k = k.trim(); (!k.is_empty()).then(|| (k.to_string(), v.trim().to_string())) }).collect()
}
#[derive(Debug, Default, PartialEq, Eq)]
pub struct WinCpu { pub name: Option<String>, pub vendor: Option<String>, pub cores: usize, pub logical: usize, pub l2_kb: Option<u64>, pub l3_kb: Option<u64> }
pub fn parse_win_cpu(t: &str) -> WinCpu {
    let mut w = WinCpu::default();
    for (k, v) in parse_kv_lines(t) {
        match k.as_str() {
            "Name" if w.name.is_none() => w.name = Some(v),
            "Manufacturer" if w.vendor.is_none() => w.vendor = Some(v),
            "NumberOfCores" => w.cores += num(&v).unwrap_or(0) as usize,
            "NumberOfLogicalProcessors" => w.logical += num(&v).unwrap_or(0) as usize,
            "L2CacheSize" => w.l2_kb = w.l2_kb.or(num(&v).filter(|n| *n > 0)),
            "L3CacheSize" => w.l3_kb = w.l3_kb.or(num(&v).filter(|n| *n > 0)),
            _ => {}
        }
    }
    w
}
/// Windows OS memory (`TotalVisibleMemorySize`, `FreePhysicalMemory`, both in kB) → (total, free) bytes.
pub fn parse_win_mem(t: &str) -> (Option<u64>, Option<u64>) {
    let kv = parse_kv_lines(t);
    let get = |k: &str| kv.iter().find(|(a, _)| a == k).and_then(|(_, v)| num(v)).map(|kb| kb * 1024);
    (get("TotalVisibleMemorySize"), get("FreePhysicalMemory"))
}
pub fn parse_win_gpus(t: &str) -> Vec<Gpu> {
    let mut g: Vec<Gpu> = vec![];
    for (k, v) in parse_kv_lines(t) {
        match k.as_str() {
            "Name" if !v.is_empty() => {
                let kind = if v.contains("NVIDIA") { GpuKind::Nvidia } else if v.contains("AMD") || v.contains("Radeon") { GpuKind::Amd } else if v.contains("Intel") { GpuKind::Intel } else { GpuKind::Other };
                g.push(Gpu { name: v, vram: None, kind });
            }
            "AdapterRAM" => if let Some(l) = g.last_mut() { l.vram = num(&v).filter(|n| *n > 0); },
            _ => {}
        }
    }
    g
}
/// `df -P path` → device node of the file system.
pub fn parse_df_device(t: &str) -> Option<String> { t.lines().nth(1)?.split_whitespace().next().filter(|d| d.starts_with('/')).map(str::to_string) }
/// `/dev/nvme0n1p2` → `nvme0n1`, `/dev/sda1` → `sda`, `/dev/mmcblk0p1` → `mmcblk0`, `/dev/dm-0` stays.
pub fn block_base_name(dev: &str) -> Option<String> {
    let n = dev.strip_prefix("/dev/")?;
    if n.is_empty() || n.contains('/') { return None; }
    if n.starts_with("dm-") || n.starts_with("md") || n.starts_with("loop") { return Some(n.to_string()); }
    if n.starts_with("nvme") || n.starts_with("mmcblk") {
        if let Some(p) = n.rfind('p') { if p > 0 && n[p + 1..].bytes().all(|b| b.is_ascii_digit()) && p + 1 < n.len() { return Some(n[..p].to_string()); } }
        return Some(n.to_string());
    }
    Some(n.trim_end_matches(|c: char| c.is_ascii_digit()).to_string()).filter(|s| !s.is_empty())
}
pub fn parse_rotational(t: &str) -> DiskKind { match t.trim() { "1" => DiskKind::Hdd, "0" => DiskKind::Ssd, _ => DiskKind::Unknown } }

// ------------------------------------------------------------------ ISA

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_isa() -> Vec<&'static str> {
    let mut v = Vec::new();
    macro_rules! chk { ($($f:tt),*) => { $( if std::arch::is_x86_feature_detected!($f) { v.push($f); } )* } }
    chk!("sse4.2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512vl", "avx512vnni", "avxvnni");
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{__cpuid, __cpuid_count};
        #[allow(unused_unsafe)]
        // SAFETY: CPUID exists on every x86_64 CPU.
        let (max, l7) = unsafe { (__cpuid(0).eax, if __cpuid(0).eax >= 7 { Some(__cpuid_count(7, 0)) } else { None }) };
        if let (true, Some(l7)) = (max >= 7, l7) { // reported by the CPU; the OS must still enable the tile state
            if l7.edx >> 24 & 1 == 1 { v.push("amx-tile"); }
            if l7.edx >> 22 & 1 == 1 { v.push("amx-bf16"); }
            if l7.edx >> 25 & 1 == 1 { v.push("amx-int8"); }
        }
    }
    v
}
#[cfg(target_arch = "aarch64")]
fn detect_isa() -> Vec<&'static str> {
    let mut v = vec!["neon"];
    macro_rules! chk { ($($f:tt),*) => { $( if std::arch::is_aarch64_feature_detected!($f) { v.push($f); } )* } }
    chk!("dotprod", "i8mm", "fp16", "sve", "bf16");
    v
}
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_isa() -> Vec<&'static str> { Vec::new() }

// ------------------------------------------------------------------ probes

fn linux_cpu_dirs() -> Vec<String> {
    let Ok(rd) = std::fs::read_dir("/sys/devices/system/cpu") else { return vec![] };
    let mut v: Vec<String> = rd.filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.strip_prefix("cpu").is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))).collect();
    v.sort();
    v
}

fn probe_linux(p: &mut HardwareProfile) {
    let info = read("/proc/cpuinfo").map(|t| parse_cpuinfo(&t)).unwrap_or_default();
    p.cpu.vendor = info.vendor.clone();
    p.cpu.model = info.model.clone();
    if info.logical > 0 { p.cpu.logical = info.logical; }
    let dirs = linux_cpu_dirs();
    let sib: Vec<String> = dirs.iter().filter_map(|d| read(&format!("/sys/devices/system/cpu/{d}/topology/thread_siblings_list"))).collect();
    p.cpu.physical = physical_from_sibling_lists(&sib).or(info.physical);
    let (core, atom) = (read("/sys/devices/cpu_core/cpus").and_then(|t| parse_cpulist(&t)), read("/sys/devices/cpu_atom/cpus").and_then(|t| parse_cpulist(&t)));
    p.cpu.hybrid = match (core, atom) {
        (Some(c), Some(a)) if c > 0 && a > 0 => Some(Hybrid { performance: c, efficiency: a }),
        _ => {
            let caps: Vec<u32> = dirs.iter().filter_map(|d| read(&format!("/sys/devices/system/cpu/{d}/cpu_capacity"))).filter_map(|t| t.trim().parse().ok()).collect();
            hybrid_from_capacities(&caps)
        }
    };
    let entries: Vec<(u32, String, String)> = (0..8).filter_map(|i| {
        let b = format!("/sys/devices/system/cpu/cpu0/cache/index{i}");
        Some((read(&format!("{b}/level"))?.trim().parse().ok()?, read(&format!("{b}/type"))?, read(&format!("{b}/size"))?))
    }).collect();
    p.caches = caches_from_entries(&entries);
    if let Some(m) = read("/proc/meminfo") { let (t, a) = parse_meminfo(&m); p.total_ram = t; p.available_ram = a; }
    p.limits = linux_cgroup_limits();
    if let Ok(rd) = std::fs::read_dir("/sys/devices/system/node") {
        let names: Vec<String> = rd.filter_map(|e| e.ok()?.file_name().into_string().ok()).collect();
        p.numa_nodes = count_numa_nodes(&names);
    }
    p.thp = read("/sys/kernel/mm/transparent_hugepage/enabled").and_then(|t| parse_thp(&t));
}

fn linux_cgroup_limits() -> Limits {
    let mut l = Limits::default();
    let min_f = |a: Option<f64>, b: Option<f64>| match (a, b) { (Some(x), Some(y)) => Some(x.min(y)), (x, None) => x, (None, y) => y };
    let min_u = |a: Option<u64>, b: Option<u64>| match (a, b) { (Some(x), Some(y)) => Some(x.min(y)), (x, None) => x, (None, y) => y };
    if let Some(own) = read("/proc/self/cgroup").and_then(|t| parse_proc_self_cgroup(&t)) {
        let comps: Vec<&str> = own.split('/').filter(|c| !c.is_empty() && *c != "..").collect();
        for k in (0..=comps.len()).rev() {
            let dir = format!("/sys/fs/cgroup/{}", comps[..k].join("/"));
            l.cpu_cores = min_f(l.cpu_cores, read(&format!("{dir}/cpu.max")).and_then(|t| parse_cgroup_v2_cpu_max(&t)));
            l.memory = min_u(l.memory, read(&format!("{dir}/memory.max")).and_then(|t| parse_cgroup_mem(&t)));
        }
    }
    for base in ["/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"] {
        if let (Some(q), Some(p)) = (read(&format!("{base}/cpu.cfs_quota_us")), read(&format!("{base}/cpu.cfs_period_us"))) { l.cpu_cores = min_f(l.cpu_cores, parse_cgroup_v1_cpu(&q, &p)); }
    }
    l.memory = min_u(l.memory, read("/sys/fs/cgroup/memory/memory.limit_in_bytes").and_then(|t| parse_cgroup_mem(&t)));
    l
}

fn probe_macos(p: &mut HardwareProfile) {
    let keys = ["hw.memsize", "hw.physicalcpu", "hw.logicalcpu", "hw.perflevel0.logicalcpu", "hw.perflevel1.logicalcpu", "hw.l1dcachesize", "hw.l2cachesize", "hw.l3cachesize",
        "hw.perflevel0.l2cachesize", "machdep.cpu.brand_string", "machdep.cpu.vendor"];
    let m = run_cmd("sysctl", &keys, PROBE_LIMIT).map(|t| parse_sysctl_pairs(&t)).unwrap_or_default();
    let n = |k: &str| m.get(k).and_then(|v| num(v));
    p.cpu.model = m.get("machdep.cpu.brand_string").cloned();
    p.cpu.vendor = m.get("machdep.cpu.vendor").cloned().or_else(|| p.cpu.model.as_ref().filter(|s| s.contains("Apple")).map(|_| "Apple".into()));
    if let Some(l) = n("hw.logicalcpu") { p.cpu.logical = l as usize; }
    p.cpu.physical = n("hw.physicalcpu").map(|v| v as usize);
    p.cpu.hybrid = match (n("hw.perflevel0.logicalcpu"), n("hw.perflevel1.logicalcpu")) { (Some(a), Some(b)) if a > 0 && b > 0 => Some(Hybrid { performance: a as usize, efficiency: b as usize }), _ => None };
    p.caches = Caches { l1d: n("hw.l1dcachesize"), l2: n("hw.l2cachesize").or(n("hw.perflevel0.l2cachesize")), l3: n("hw.l3cachesize") };
    p.total_ram = n("hw.memsize");
    p.available_ram = run_cmd("vm_stat", &[], PROBE_LIMIT).and_then(|t| parse_vm_stat(&t));
}

const PS_CPU: &str = "Get-CimInstance Win32_Processor | ForEach-Object { \"Name=$($_.Name)\"; \"Manufacturer=$($_.Manufacturer)\"; \"NumberOfCores=$($_.NumberOfCores)\"; \"NumberOfLogicalProcessors=$($_.NumberOfLogicalProcessors)\"; \"L2CacheSize=$($_.L2CacheSize)\"; \"L3CacheSize=$($_.L3CacheSize)\" }";
const PS_MEM: &str = "Get-CimInstance Win32_OperatingSystem | ForEach-Object { \"TotalVisibleMemorySize=$($_.TotalVisibleMemorySize)\"; \"FreePhysicalMemory=$($_.FreePhysicalMemory)\" }";
const PS_GPU: &str = "Get-CimInstance Win32_VideoController | ForEach-Object { \"Name=$($_.Name)\"; \"AdapterRAM=$($_.AdapterRAM)\" }";

fn powershell(script: &str) -> Option<String> { run_cmd("powershell", &["-NoProfile", "-NonInteractive", "-Command", script], Duration::from_secs(6)) }

fn probe_windows(p: &mut HardwareProfile) {
    if let Some(n) = std::env::var("NUMBER_OF_PROCESSORS").ok().and_then(|v| v.trim().parse::<usize>().ok()) { p.cpu.logical = n; }
    p.cpu.model = std::env::var("PROCESSOR_IDENTIFIER").ok();
    let cpu = powershell(PS_CPU).or_else(|| run_cmd("wmic", &["cpu", "get", "Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,L2CacheSize,L3CacheSize", "/value"], PROBE_LIMIT));
    if let Some(t) = cpu {
        let w = parse_win_cpu(&t);
        if w.name.is_some() { p.cpu.model = w.name; }
        p.cpu.vendor = w.vendor;
        if w.cores > 0 { p.cpu.physical = Some(w.cores); }
        if w.logical > 0 { p.cpu.logical = w.logical; }
        p.caches.l2 = w.l2_kb.map(|k| k * 1024);
        p.caches.l3 = w.l3_kb.map(|k| k * 1024);
    }
    let mem = powershell(PS_MEM).or_else(|| run_cmd("wmic", &["OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/value"], PROBE_LIMIT));
    if let Some(t) = mem { let (a, b) = parse_win_mem(&t); p.total_ram = a; p.available_ram = b; }
}

fn probe_gpus(p: &mut HardwareProfile) {
    let mut gpus = run_cmd("nvidia-smi", &["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], PROBE_LIMIT).map(|t| parse_nvidia_smi(&t)).unwrap_or_default();
    if let Some(t) = run_cmd("rocm-smi", &["--showmeminfo", "vram", "--csv"], PROBE_LIMIT) { gpus.extend(parse_rocm_smi(&t)); }
    match p.os {
        "macos" => if let Some(t) = run_cmd("system_profiler", &["SPDisplaysDataType"], Duration::from_secs(6)) { gpus.extend(parse_system_profiler(&t)); },
        "windows" => if let Some(t) = powershell(PS_GPU) { for g in parse_win_gpus(&t) { if !(g.kind == GpuKind::Nvidia && gpus.iter().any(|x| x.kind == GpuKind::Nvidia)) { gpus.push(g); } } },
        "linux" => if let Ok(rd) = std::fs::read_dir("/sys/class/drm") {
            let mut cards: Vec<String> = rd.filter_map(|e| e.ok()?.file_name().into_string().ok()).filter(|n| n.strip_prefix("card").is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))).collect();
            cards.sort();
            for c in cards {
                let Some(v) = read(&format!("/sys/class/drm/{c}/device/vendor")) else { continue };
                let kind = gpu_kind_from_pci_vendor(&v);
                if gpus.iter().any(|g| g.kind == kind) || kind == GpuKind::Other { continue; }
                gpus.push(Gpu { name: format!("{kind:?} GPU ({c})"), vram: None, kind });
            }
        },
        _ => {}
    }
    p.unified_memory = p.os == "macos" && p.arch == "aarch64";
    p.gpus = gpus;
}

/// Disk type of the file system holding `path` (Linux; macOS on Apple Silicon is always flash).
pub fn disk_kind_for(path: &Path) -> DiskKind {
    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") { return DiskKind::Ssd; }
    if !cfg!(target_os = "linux") { return DiskKind::Unknown; }
    let Some(p) = path.to_str() else { return DiskKind::Unknown };
    let Some(dev) = run_cmd("df", &["-P", p], PROBE_LIMIT).and_then(|t| parse_df_device(&t)) else { return DiskKind::Unknown };
    let dev = std::fs::canonicalize(&dev).ok().and_then(|d| d.to_str().map(str::to_string)).unwrap_or(dev);
    let Some(base) = block_base_name(&dev) else { return DiskKind::Unknown };
    let mut candidates = vec![base.clone()];
    if let Ok(rd) = std::fs::read_dir(format!("/sys/block/{base}/slaves")) { candidates.extend(rd.filter_map(|e| e.ok()?.file_name().into_string().ok())); }
    for c in candidates {
        let k = read(&format!("/sys/block/{c}/queue/rotational")).map(|t| parse_rotational(&t)).unwrap_or(DiskKind::Unknown);
        if k != DiskKind::Unknown { return k; }
        // a partition sits below its disk: /sys/block/<disk>/<part>
        if let Some(parent) = c.rfind(|ch: char| !ch.is_ascii_digit()).map(|i| c[..=i].trim_end_matches('p').to_string()) {
            if let Some(t) = read(&format!("/sys/block/{parent}/queue/rotational")) { let k = parse_rotational(&t); if k != DiskKind::Unknown { return k; } }
        }
    }
    DiskKind::Unknown
}

impl HardwareProfile {
    /// Probe this machine. `model_dir` (optional) is used to classify the disk the weights live on.
    pub fn detect(model_dir: Option<&Path>) -> HardwareProfile {
        let mut p = HardwareProfile { os: std::env::consts::OS, arch: std::env::consts::ARCH, numa_nodes: 1, ..Default::default() };
        p.usable_threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        p.cpu.logical = p.usable_threads;
        p.cpu.isa = detect_isa();
        match p.os { "linux" => probe_linux(&mut p), "macos" => probe_macos(&mut p), "windows" => probe_windows(&mut p), _ => {} }
        if p.cpu.logical == 0 { p.cpu.logical = p.usable_threads; }
        probe_gpus(&mut p);
        if let Some(d) = model_dir { p.disk = disk_kind_for(d); }
        p
    }
    pub fn has_isa(&self, f: &str) -> bool { self.cpu.isa.contains(&f) }
}

// ------------------------------------------------------------------ tuning

/// Worker threads for matrix–vector products.
pub fn choose_threads(physical: Option<usize>, usable: usize, quota: Option<f64>) -> usize {
    let usable = usable.max(1);
    let mut cores = physical.unwrap_or(usable).clamp(1, usable);
    if let Some(q) = quota { cores = cores.min((q.ceil() as usize).max(1)); }
    else if (3..=4).contains(&cores) { cores -= 1; } // leave one core to the OS and the UI on small machines
    cores.max(1)
}

/// Cheap thread count for the global pool: no subprocesses. `COLIBRI_THREADS` overrides it.
pub fn quick_threads() -> usize {
    if let Some(n) = std::env::var("COLIBRI_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|n| *n > 0) { return n.min(1024); }
    let usable = std::thread::available_parallelism().map_or(1, |n| n.get());
    let (mut physical, mut quota) = (None, None);
    if cfg!(target_os = "linux") {
        let sib: Vec<String> = linux_cpu_dirs().iter().filter_map(|d| read(&format!("/sys/devices/system/cpu/{d}/topology/thread_siblings_list"))).collect();
        physical = physical_from_sibling_lists(&sib);
        quota = linux_cgroup_limits().cpu_cores;
    }
    choose_threads(physical, usable, quota)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPrecision { F32, F16, Q8 }

#[derive(Clone, Debug, PartialEq)]
pub struct Tuning {
    pub threads: usize,
    pub prefill_batch: usize,
    pub use_mmap: bool,
    /// `madvise(MADV_HUGEPAGE)` on the weight mapping is worthwhile (Linux, large model, THP not disabled).
    pub hugepage_hint: bool,
    pub numa_interleave: bool,
    /// Bytes to hand to [`crate::plan::plan`].
    pub ram_budget: u64,
    pub kv_precision: KvPrecision,
    notes: Vec<String>,
}

fn gib(b: u64) -> String { format!("{:.1} GiB", b as f64 / GIB as f64) }

impl Tuning {
    pub fn explain(&self) -> &[String] { &self.notes }

    pub fn recommend(p: &HardwareProfile, model_bytes: u64, kv_bytes: u64) -> Tuning {
        let mut notes = Vec::new();
        // threads
        let threads = choose_threads(p.cpu.physical, p.usable_threads.max(1), p.limits.cpu_cores);
        notes.push(format!("{threads} worker threads ({} physical / {} logical cores{}).", p.cpu.physical.map_or("?".to_string(), |n| n.to_string()), p.cpu.logical,
            p.limits.cpu_cores.map_or(String::new(), |q| format!(", container limit {q:.1} CPUs"))));
        if let Some(h) = p.cpu.hybrid { notes.push(format!("Hybrid CPU ({} performance + {} efficiency logical cores): work is handed out in small chunks so fast cores take more of it.", h.performance, h.efficiency)); }
        // memory
        let total = match (p.total_ram, p.limits.memory) { (Some(t), Some(l)) => Some(t.min(l)), (t, l) => t.or(l) };
        let avail = match (p.available_ram, p.limits.memory) { (Some(a), Some(l)) => Some(a.min(l)), (a, l) => a.or(l) };
        let (total_v, avail_v) = match (total, avail) {
            (Some(t), Some(a)) => (t, a.min(t)),
            (Some(t), None) => (t, t / 2),
            (None, Some(a)) => (a * 2, a),
            (None, None) => { notes.push("Memory size unknown: assuming 4 GiB.".into()); (4 * GIB, 2 * GIB) }
        };
        let (reserve, frac) = if total_v <= 4 * GIB { (512 * MIB, 0.6) } else if total_v <= 8 * GIB { (768 * MIB, 0.7) } else { (GIB, 0.8) };
        let ram_budget = (avail_v.saturating_sub(reserve) as f64 * frac) as u64;
        notes.push(format!("RAM budget {} of {} available ({} total; {} kept for the system, {:.0}% usable).", gib(ram_budget), gib(avail_v), gib(total_v), gib(reserve), frac * 100.0));
        if p.unified_memory { notes.push("Unified memory: CPU and GPU share the same pool, so the budget is not doubled by the GPU.".into()); }
        // KV cache precision
        let pressure = (model_bytes as f64 + kv_bytes as f64) / ram_budget.max(1) as f64;
        let kv_precision = if pressure <= 0.6 { KvPrecision::F32 } else if pressure <= 1.0 { KvPrecision::F16 } else { KvPrecision::Q8 };
        notes.push(format!("KV cache {:?}: model plus cache need {:.0}% of the budget.", kv_precision, pressure * 100.0));
        // mmap
        let model_streams = model_bytes > ram_budget;
        let use_mmap = !(p.disk == DiskKind::Hdd && model_streams) && p.arch != "x86" && p.arch != "arm";
        notes.push(match (use_mmap, p.disk) {
            (false, DiskKind::Hdd) => "Weights are bigger than the budget and live on a spinning disk: read them in large sequential blocks instead of memory-mapping (random page faults would crawl).".into(),
            (false, _) => "32-bit address space: memory mapping is not safe for large models, read in blocks instead.".into(),
            (true, DiskKind::Unknown) => "Memory-map the weights (disk type unknown).".into(),
            (true, k) => format!("Memory-map the weights ({k:?}); the page cache keeps hot parts in RAM."),
        });
        let hugepage_hint = p.os == "linux" && model_bytes >= GIB && p.thp != Some(ThpMode::Never) && use_mmap;
        if hugepage_hint { notes.push("Ask for transparent huge pages on the weight mapping (fewer TLB misses).".into()); }
        let numa_interleave = p.numa_nodes > 1 && model_bytes >= GIB;
        if numa_interleave { notes.push(format!("{} NUMA nodes: interleave the weight pages across nodes so no single memory controller is the bottleneck.", p.numa_nodes)); }
        // prefill batch
        let class = if p.has_isa("amx-tile") || p.has_isa("avx512f") { 3 } else if (p.has_isa("avx2") && p.has_isa("fma")) || (p.has_isa("neon") && p.has_isa("dotprod")) || p.unified_memory { 2 } else { 1 };
        let mut prefill_batch = match class { 3 => 32, 2 => 16, _ => 4 };
        if ram_budget < 2 * GIB { prefill_batch = prefill_batch.min(4); } else if ram_budget < 6 * GIB { prefill_batch = prefill_batch.min(8); }
        if threads == 1 { prefill_batch = prefill_batch.min(8); }
        notes.push(format!("Prompt processing in batches of {prefill_batch} tokens ({}).", match class { 3 => "wide vector units", 2 => "256-bit SIMD / dot-product units", _ => "baseline SIMD" }));
        if let Some(g) = p.gpus.first() { notes.push(format!("GPU detected: {}{}. The CPU engine does not offload to it yet.", g.name, g.vram.map_or(String::new(), |v| format!(" ({} VRAM)", gib(v))))); }
        Tuning { threads, prefill_batch, use_mmap, hugepage_hint, numa_interleave, ram_budget, kv_precision, notes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X86_CPUINFO: &str = "processor\t: 0\nvendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel name\t: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz\nphysical id\t: 0\nsiblings\t: 8\ncore id\t: 0\ncpu cores\t: 4\nflags\t\t: fpu sse4_2 avx avx2 fma\n\nprocessor\t: 1\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz\nphysical id\t: 0\ncore id\t: 1\ncpu cores\t: 4\n\nprocessor\t: 2\nvendor_id\t: GenuineIntel\nphysical id\t: 0\ncore id\t: 0\ncpu cores\t: 4\n\nprocessor\t: 3\nphysical id\t: 0\ncore id\t: 1\ncpu cores\t: 4\n\n";
    const ARM_CPUINFO: &str = "processor\t: 0\nBogoMIPS\t: 38.40\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp\nCPU implementer\t: 0x41\nCPU part\t: 0xd0b\n\nprocessor\t: 1\nFeatures\t: fp asimd\nCPU implementer\t: 0x41\nCPU part\t: 0xd05\n\nHardware\t: Raspberry Pi 5\nModel\t: Raspberry Pi 5 Model B Rev 1.0\n";
    const MEMINFO: &str = "MemTotal:       16384000 kB\nMemFree:         1000000 kB\nMemAvailable:    9000000 kB\nBuffers:          100000 kB\nCached:          2000000 kB\n";

    #[test]
    fn x86_cpuinfo_counts_logical_and_physical_cores() {
        let c = parse_cpuinfo(X86_CPUINFO);
        assert_eq!((c.logical, c.physical), (4, Some(2)));
        assert_eq!(c.vendor.as_deref(), Some("GenuineIntel"));
        assert!(c.model.unwrap().contains("i7-8550U"));
        assert!(c.features.contains(&"avx2".to_string()));
    }
    #[test]
    fn arm_cpuinfo_has_parts_features_and_a_board_name() {
        let c = parse_cpuinfo(ARM_CPUINFO);
        assert_eq!((c.logical, c.physical), (2, None));
        assert_eq!(c.parts, vec!["0xd0b", "0xd05"]);
        assert!(c.features.contains(&"asimddp".to_string()));
        assert_eq!(c.model.as_deref(), Some("Raspberry Pi 5"));
        assert_eq!(parse_cpuinfo(""), CpuInfo::default());
        assert_eq!(parse_cpuinfo("processor : 0\ncpu cores : 8\n\nprocessor : 1\ncpu cores : 8\n").physical, Some(8)); // VM without ids
    }
    #[test]
    fn meminfo_with_and_without_memavailable() {
        assert_eq!(parse_meminfo(MEMINFO), (Some(16_384_000 * 1024), Some(9_000_000 * 1024)));
        let old = "MemTotal: 1000 kB\nMemFree: 100 kB\nBuffers: 10 kB\nCached: 20 kB\n";
        assert_eq!(parse_meminfo(old), (Some(1_024_000), Some(130 * 1024)));
        assert_eq!(parse_meminfo("garbage"), (None, None));
    }
    #[test]
    fn cpu_lists_and_topology() {
        assert_eq!(parse_cpulist("0-3,8,10-11"), Some(7));
        assert_eq!(parse_cpulist("0-15\n"), Some(16));
        assert_eq!(parse_cpulist(""), Some(0));
        assert_eq!(parse_cpulist("3-1"), None);
        assert_eq!(parse_cpulist("a-b"), None);
        let sib: Vec<String> = ["0,4\n", "1,5\n", "0,4\n", "1,5\n", "2,6", "3,7"].iter().map(|s| s.to_string()).collect();
        assert_eq!(physical_from_sibling_lists(&sib), Some(4));
        assert_eq!(physical_from_sibling_lists(&[]), None);
    }
    #[test]
    fn big_little_from_capacities() {
        assert_eq!(hybrid_from_capacities(&[1024, 1024, 1024, 1024, 430, 430, 430, 430]), Some(Hybrid { performance: 4, efficiency: 4 }));
        assert_eq!(hybrid_from_capacities(&[1024, 871, 871, 871, 446, 446, 446, 446]), Some(Hybrid { performance: 4, efficiency: 4 }));
        assert_eq!(hybrid_from_capacities(&[1024; 8]), None);
        assert_eq!(hybrid_from_capacities(&[]), None);
    }
    #[test]
    fn cache_sizes() {
        assert_eq!(parse_size("32K"), Some(32768));
        assert_eq!(parse_size("8M\n"), Some(8 << 20));
        assert_eq!(parse_size("1G"), Some(1 << 30));
        assert_eq!(parse_size("512"), Some(512));
        assert_eq!(parse_size("xK"), None);
        let e = vec![(1, "Data".into(), "32K".into()), (1, "Instruction".into(), "64K".into()), (2, "Unified".into(), "1024K".into()), (3, "Unified".into(), "32M".into())];
        assert_eq!(caches_from_entries(&e), Caches { l1d: Some(32 << 10), l2: Some(1 << 20), l3: Some(32 << 20) });
    }
    #[test]
    fn cgroup_files_v1_and_v2() {
        assert_eq!(parse_cgroup_v2_cpu_max("max 100000\n"), None);
        assert_eq!(parse_cgroup_v2_cpu_max("250000 100000\n"), Some(2.5));
        assert_eq!(parse_cgroup_v2_cpu_max("100000 0"), None);
        assert_eq!(parse_cgroup_v1_cpu("-1\n", "100000\n"), None);
        assert_eq!(parse_cgroup_v1_cpu("400000", "100000"), Some(4.0));
        assert_eq!(parse_cgroup_mem("max\n"), None);
        assert_eq!(parse_cgroup_mem("2147483648\n"), Some(2 << 30));
        assert_eq!(parse_cgroup_mem("9223372036854771712\n"), None);
        assert_eq!(parse_proc_self_cgroup("12:memory:/x\n0::/user.slice/app.scope\n").as_deref(), Some("/user.slice/app.scope"));
        assert_eq!(parse_proc_self_cgroup("4:cpu:/docker\n"), None);
    }
    #[test]
    fn thp_and_numa() {
        assert_eq!(parse_thp("always [madvise] never\n"), Some(ThpMode::Madvise));
        assert_eq!(parse_thp("[always] madvise never"), Some(ThpMode::Always));
        assert_eq!(parse_thp("always madvise [never]"), Some(ThpMode::Never));
        assert_eq!(parse_thp("nonsense"), None);
        let n: Vec<String> = ["node0", "node1", "possible", "online", "has_cpu"].iter().map(|s| s.to_string()).collect();
        assert_eq!(count_numa_nodes(&n), 2);
        assert_eq!(count_numa_nodes(&[]), 1);
    }
    #[test]
    fn nvidia_and_rocm_output() {
        let g = parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564\nNVIDIA A100-SXM4-80GB, 81920\n");
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].vram, Some(24564 * MIB));
        assert_eq!(g[1].name, "NVIDIA A100-SXM4-80GB");
        let jet = parse_nvidia_smi("Orin (nvgpu), [N/A]\n");
        assert_eq!((jet[0].name.as_str(), jet[0].vram), ("Orin (nvgpu)", None));
        assert!(parse_nvidia_smi("No devices were found").is_empty());
        let r = parse_rocm_smi("device,VRAM Total Memory (B),VRAM Total Used Memory (B)\ncard0,17163091968,123456\ncard1,8573157376,0\n");
        assert_eq!((r.len(), r[0].vram, r[1].kind), (2, Some(17_163_091_968), GpuKind::Amd));
        assert!(parse_rocm_smi("nothing").is_empty());
        assert_eq!(gpu_kind_from_pci_vendor("0x10de\n"), GpuKind::Nvidia);
        assert_eq!(gpu_kind_from_pci_vendor("0x1002"), GpuKind::Amd);
        assert_eq!(gpu_kind_from_pci_vendor("0x1234"), GpuKind::Other);
    }
    #[test]
    fn apple_sysctl_vmstat_and_system_profiler() {
        let s = "hw.memsize: 17179869184\nhw.physicalcpu: 8\nhw.logicalcpu: 8\nhw.perflevel0.logicalcpu: 4\nhw.perflevel1.logicalcpu: 4\nhw.l1dcachesize: 65536\nhw.perflevel0.l2cachesize: 4194304\nmachdep.cpu.brand_string: Apple M2\n";
        let m = parse_sysctl_pairs(s);
        assert_eq!(m["hw.memsize"], "17179869184");
        assert_eq!(m["machdep.cpu.brand_string"], "Apple M2");
        let vm = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               20000.\nPages active:                            500000.\nPages inactive:                          100000.\nPages speculative:                         5000.\nPages purgeable:                           1000.\n";
        assert_eq!(parse_vm_stat(vm), Some((20000 + 100000 + 5000 + 1000) * 16384));
        assert_eq!(parse_vm_stat("junk"), None);
        let sp = "Graphics/Displays:\n\n    Apple M2:\n\n      Chipset Model: Apple M2\n      Type: GPU\n      Total Number of Cores: 10\n      Vendor: Apple (0x106b)\n\n    AMD Radeon Pro 5500M:\n\n      Chipset Model: AMD Radeon Pro 5500M\n      VRAM (Total): 8 GB\n";
        let g = parse_system_profiler(sp);
        assert_eq!((g.len(), g[0].kind, g[0].vram, g[1].kind, g[1].vram), (2, GpuKind::Apple, None, GpuKind::Amd, Some(8 * GIB)));
    }
    #[test]
    fn windows_powershell_and_wmic_output() {
        let cpu = "Name=AMD Ryzen 9 7950X 16-Core Processor            \r\nManufacturer=AuthenticAMD\r\nNumberOfCores=16\r\nNumberOfLogicalProcessors=32\r\nL2CacheSize=16384\r\nL3CacheSize=65536\r\n";
        let w = parse_win_cpu(cpu);
        assert_eq!((w.cores, w.logical, w.l2_kb, w.l3_kb), (16, 32, Some(16384), Some(65536)));
        assert!(w.name.unwrap().starts_with("AMD Ryzen 9"));
        let dual = "NumberOfCores=12\nNumberOfLogicalProcessors=24\nNumberOfCores=12\nNumberOfLogicalProcessors=24\n";
        assert_eq!((parse_win_cpu(dual).cores, parse_win_cpu(dual).logical), (24, 48));
        let utf16ish = "T\0o\0t\0a\0l\0V\0i\0s\0i\0b\0l\0e\0M\0e\0m\0o\0r\0y\0S\0i\0z\0e\0=\x003\x002\x000\x000\x000\x000\x000\x000\x00\r\0\nFreePhysicalMemory=100\r\n";
        assert_eq!(parse_win_mem(utf16ish), (Some(32_000_000 * 1024), Some(100 * 1024)));
        let gpu = "Name=NVIDIA GeForce RTX 3060\r\nAdapterRAM=4293918720\r\nName=Intel(R) UHD Graphics\r\nAdapterRAM=1073741824\r\n";
        let g = parse_win_gpus(gpu);
        assert_eq!((g.len(), g[0].kind, g[1].kind, g[1].vram), (2, GpuKind::Nvidia, GpuKind::Intel, Some(GIB)));
        assert_eq!(parse_win_mem("nothing"), (None, None));
    }
    #[test]
    fn disk_helpers() {
        assert_eq!(parse_df_device("Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/nvme0n1p2 100 1 99 1% /\n").as_deref(), Some("/dev/nvme0n1p2"));
        assert_eq!(parse_df_device("Filesystem\ntmpfs 1 1 1 1% /tmp\n"), None);
        for (d, want) in [("/dev/nvme0n1p2", "nvme0n1"), ("/dev/nvme0n1", "nvme0n1"), ("/dev/sda1", "sda"), ("/dev/sdb", "sdb"), ("/dev/mmcblk0p1", "mmcblk0"), ("/dev/vda3", "vda"), ("/dev/dm-0", "dm-0"), ("/dev/xvda1", "xvda")] {
            assert_eq!(block_base_name(d).as_deref(), Some(want), "{d}");
        }
        assert_eq!(block_base_name("/dev/mapper/x"), None);
        assert_eq!(block_base_name("tmpfs"), None);
        assert_eq!((parse_rotational("1\n"), parse_rotational("0"), parse_rotational("?")), (DiskKind::Hdd, DiskKind::Ssd, DiskKind::Unknown));
    }
    #[test]
    fn thread_choice_rules() {
        assert_eq!(choose_threads(Some(16), 32, None), 16);
        assert_eq!(choose_threads(Some(4), 8, None), 3); // small machine: leave one
        assert_eq!(choose_threads(Some(2), 4, None), 2);
        assert_eq!(choose_threads(Some(1), 1, None), 1);
        assert_eq!(choose_threads(Some(8), 8, Some(2.5)), 3); // container quota wins, nothing left free
        assert_eq!(choose_threads(Some(8), 8, Some(0.2)), 1);
        assert_eq!(choose_threads(Some(16), 6, None), 6); // affinity mask
        assert_eq!(choose_threads(None, 12, None), 12);
        assert_eq!(choose_threads(None, 0, None), 1);
    }

    fn base() -> HardwareProfile { HardwareProfile { os: "linux", arch: "x86_64", numa_nodes: 1, ..Default::default() } }
    fn with(f: impl FnOnce(&mut HardwareProfile)) -> HardwareProfile { let mut p = base(); f(&mut p); p }

    #[test]
    fn four_gb_laptop_is_conservative() {
        let p = with(|p| { p.cpu = Cpu { logical: 4, physical: Some(2), isa: vec!["sse4.2"], ..Default::default() }; p.usable_threads = 4; p.total_ram = Some(4 * GIB); p.available_ram = Some(2 * GIB); p.disk = DiskKind::Hdd; });
        let t = Tuning::recommend(&p, 2 * GIB, 100 * MIB);
        assert_eq!(t.threads, 2);
        assert!(t.ram_budget < GIB, "{}", t.ram_budget);
        assert_eq!(t.prefill_batch, 4);
        assert_eq!(t.kv_precision, KvPrecision::Q8);
        assert!(!t.use_mmap, "HDD and a model larger than the budget must not be memory-mapped");
        assert!(t.explain().iter().any(|n| n.contains("spinning")));
    }
    #[test]
    fn big_workstation_goes_wide() {
        let p = with(|p| { p.cpu = Cpu { logical: 64, physical: Some(32), isa: vec!["avx2", "fma", "avx512f"], ..Default::default() }; p.usable_threads = 64;
            p.total_ram = Some(256 * GIB); p.available_ram = Some(200 * GIB); p.numa_nodes = 2; p.thp = Some(ThpMode::Madvise); p.disk = DiskKind::Ssd; });
        let t = Tuning::recommend(&p, 40 * GIB, 2 * GIB);
        assert_eq!((t.threads, t.prefill_batch, t.kv_precision), (32, 32, KvPrecision::F32));
        assert!(t.use_mmap && t.hugepage_hint && t.numa_interleave);
        assert_eq!(t.ram_budget, ((200 * GIB - GIB) as f64 * 0.8) as u64);
    }
    #[test]
    fn container_limits_win_over_the_host() {
        let p = with(|p| { p.cpu = Cpu { logical: 64, physical: Some(32), isa: vec!["avx2", "fma"], ..Default::default() }; p.usable_threads = 64;
            p.total_ram = Some(256 * GIB); p.available_ram = Some(200 * GIB); p.limits = Limits { cpu_cores: Some(2.0), memory: Some(8 * GIB) }; });
        let t = Tuning::recommend(&p, 3 * GIB, GIB / 2);
        assert_eq!(t.threads, 2);
        assert!(t.ram_budget <= 8 * GIB && t.ram_budget >= 4 * GIB, "{}", t.ram_budget);
        assert!(t.explain()[0].contains("container limit 2.0"));
    }
    #[test]
    fn apple_silicon_unified_memory_and_missing_data() {
        let p = with(|p| { p.os = "macos"; p.arch = "aarch64"; p.cpu = Cpu { logical: 8, physical: Some(8), hybrid: Some(Hybrid { performance: 4, efficiency: 4 }), isa: vec!["neon", "dotprod", "i8mm"], ..Default::default() };
            p.usable_threads = 8; p.total_ram = Some(16 * GIB); p.available_ram = Some(10 * GIB); p.unified_memory = true; p.disk = DiskKind::Ssd;
            p.gpus = vec![Gpu { name: "Apple M2".into(), vram: None, kind: GpuKind::Apple }]; });
        let t = Tuning::recommend(&p, 5 * GIB, GIB);
        assert_eq!((t.threads, t.prefill_batch, t.use_mmap, t.hugepage_hint), (8, 16, true, false));
        let text = t.explain().join("\n");
        assert!(text.contains("Unified memory") && text.contains("Hybrid CPU") && text.contains("GPU detected"));
        let t = Tuning::recommend(&HardwareProfile::default(), 0, 0);
        assert!(t.threads >= 1 && t.ram_budget > 0 && t.prefill_batch >= 1);
        assert!(t.explain().iter().any(|n| n.contains("unknown")));
    }
    #[test]
    fn hdd_with_a_model_that_fits_still_maps() {
        let p = with(|p| { p.usable_threads = 8; p.cpu = Cpu { logical: 8, physical: Some(8), isa: vec!["avx2", "fma"], ..Default::default() }; p.total_ram = Some(32 * GIB); p.available_ram = Some(24 * GIB); p.disk = DiskKind::Hdd; });
        assert!(Tuning::recommend(&p, 4 * GIB, GIB / 4).use_mmap);
        assert!(!Tuning::recommend(&p, 60 * GIB, GIB).use_mmap);
    }
    #[test]
    fn real_detection_is_sane_and_does_not_hang() {
        let t = Instant::now();
        let p = HardwareProfile::detect(Some(Path::new(".")));
        assert!(t.elapsed() < Duration::from_secs(60));
        assert!(p.cpu.logical >= 1 && p.usable_threads >= 1 && p.numa_nodes >= 1);
        assert_eq!(p.os, std::env::consts::OS);
        assert!(quick_threads() >= 1);
        let tu = Tuning::recommend(&p, 4 * GIB, GIB / 4);
        assert!(tu.threads >= 1 && !tu.explain().is_empty());
        assert!(run_cmd("definitely-not-a-program-xyz", &[], Duration::from_secs(1)).is_none());
    }
    #[test]
    fn a_hung_tool_is_killed_at_the_limit() {
        if cfg!(unix) {
            let t = Instant::now();
            assert!(run_cmd("sleep", &["5"], Duration::from_millis(200)).is_none());
            assert!(t.elapsed() < Duration::from_secs(3));
        }
    }
}
