//! Reads the measurements (memory, swap, graphics cards, which programs hold memory) and turns
//! them into plain advice. Pure functions: the interface chooses the words.

use crate::gpu::{GpuInfo, Vendor};

/// How a model of a given size will behave on this computer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Fits in the dedicated memory of a graphics card.
    FitsGpu,
    /// Fits in main memory with room left for the system.
    FitsRam,
    /// Larger than main memory: parts are read from the disk while answering.
    StreamsFromDisk,
}

/// Memory kept for the operating system and the other programs.
pub fn reserve_gb(ram_total_gb: f64) -> f64 {
    (ram_total_gb * 0.25).clamp(2.0, 6.0)
}

pub fn usable_ram_gb(ram_total_gb: f64) -> f64 {
    (ram_total_gb - reserve_gb(ram_total_gb)).max(0.5)
}

pub fn best_vram_gb(gpus: &[GpuInfo]) -> Option<f64> {
    gpus.iter()
        .filter(|g| !g.integrated)
        .filter_map(|g| g.vram_gb)
        .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |x| x.max(v))))
}

pub fn assess(model_gb: f64, ram_total_gb: f64, vram_gb: Option<f64>) -> Verdict {
    if vram_gb.is_some_and(|v| model_gb * 1.1 <= v) {
        Verdict::FitsGpu
    } else if model_gb <= usable_ram_gb(ram_total_gb) {
        Verdict::FitsRam
    } else {
        Verdict::StreamsFromDisk
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Tip {
    /// Windows is paging memory to the disk: everything slows down.
    SwapHeavy {
        swap_gb: f64,
    },
    /// The loaded model is bigger than the memory it can use.
    ModelBiggerThanRam {
        model_gb: f64,
        usable_gb: f64,
    },
    /// Other programs hold a lot of memory (name, GB).
    CloseApps(Vec<(String, f64)>),
    /// A dedicated card exists but the engine runs on the processor.
    GpuNotUsed {
        name: String,
    },
    /// Only a chip that shares main memory: no separate speed-up for large models.
    IntegratedOnly {
        name: String,
    },
    DiskNearlyFull {
        free_gb: f64,
    },
}

pub struct TipInput<'a> {
    pub model_gb: Option<f64>,
    pub ram_total_gb: f64,
    pub ram_used_gb: f64,
    pub swap_used_gb: f64,
    pub gpus: &'a [GpuInfo],
    /// The backend the running server reports (`cpu`, `vulkan`, `cuda`, `metal`…), if it runs.
    pub backend_in_use: Option<&'a str>,
    /// Programs that are not ours, largest first.
    pub others: &'a [(String, f64)],
    pub disk_free_gb: Option<f64>,
}

pub fn tips(i: &TipInput) -> Vec<Tip> {
    let mut out = Vec::new();
    let tight = i.ram_total_gb > 0.0 && i.ram_used_gb / i.ram_total_gb > 0.85;
    if i.swap_used_gb > 2.0 && tight {
        out.push(Tip::SwapHeavy {
            swap_gb: i.swap_used_gb,
        });
    }
    if let Some(m) = i.model_gb {
        let usable = usable_ram_gb(i.ram_total_gb);
        if i.ram_total_gb > 0.0
            && assess(m, i.ram_total_gb, best_vram_gb(i.gpus)) == Verdict::StreamsFromDisk
        {
            out.push(Tip::ModelBiggerThanRam {
                model_gb: m,
                usable_gb: usable,
            });
        }
    }
    let big: Vec<(String, f64)> = i
        .others
        .iter()
        .filter(|(_, g)| *g >= 0.5)
        .take(3)
        .cloned()
        .collect();
    if tight && !big.is_empty() {
        out.push(Tip::CloseApps(big));
    }
    let on_cpu = i
        .backend_in_use
        .is_some_and(|b| b.eq_ignore_ascii_case("cpu"));
    if on_cpu {
        if let Some(g) = i
            .gpus
            .iter()
            .find(|g| !g.integrated && g.vendor != Vendor::Other)
        {
            out.push(Tip::GpuNotUsed {
                name: g.name.clone(),
            });
        }
    }
    if !i.gpus.is_empty() && i.gpus.iter().all(|g| g.integrated) {
        out.push(Tip::IntegratedOnly {
            name: i.gpus[0].name.clone(),
        });
    }
    if let Some(f) = i.disk_free_gb {
        if f < 20.0 {
            out.push(Tip::DiskNearlyFull { free_gb: f });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuInfo, looks_integrated};

    fn gpu(name: &str, vendor: Vendor, vram: Option<f64>) -> GpuInfo {
        GpuInfo {
            name: name.into(),
            vendor,
            vram_gb: vram,
            integrated: looks_integrated(name, vendor, vram),
        }
    }

    #[test]
    fn verdicts() {
        assert_eq!(assess(23.0, 15.7, None), Verdict::StreamsFromDisk);
        assert_eq!(assess(8.0, 15.7, None), Verdict::FitsRam);
        assert_eq!(assess(8.0, 16.0, Some(12.0)), Verdict::FitsGpu);
        assert_eq!(assess(11.0, 16.0, Some(12.0)), Verdict::FitsRam);
    }

    #[test]
    fn reserve_is_bounded() {
        assert_eq!(reserve_gb(4.0), 2.0);
        assert_eq!(reserve_gb(16.0), 4.0);
        assert_eq!(reserve_gb(128.0), 6.0);
        assert!(usable_ram_gb(1.0) >= 0.5);
    }

    #[test]
    fn integrated_card_does_not_count_as_vram() {
        let g = [gpu("Intel UHD", Vendor::Intel, None)];
        assert_eq!(best_vram_gb(&g), None);
        let d = [gpu("RTX 4070", Vendor::Nvidia, Some(12.0))];
        assert_eq!(best_vram_gb(&d), Some(12.0));
    }

    #[test]
    fn the_reported_case_gets_the_right_advice() {
        // 15.7 GB RAM, 13.8 used, 14.8 swap, a 23 GB model, browser and Teams holding memory
        let others = vec![
            ("chrome.exe".to_string(), 3.1),
            ("Teams.exe".to_string(), 1.4),
        ];
        let gpus = [gpu("Intel UHD Graphics", Vendor::Intel, None)];
        let t = tips(&TipInput {
            model_gb: Some(23.0),
            ram_total_gb: 15.7,
            ram_used_gb: 13.8,
            swap_used_gb: 14.8,
            gpus: &gpus,
            backend_in_use: Some("cpu"),
            others: &others,
            disk_free_gb: Some(62.0),
        });
        assert!(t.iter().any(|x| matches!(x, Tip::SwapHeavy { .. })));
        assert!(
            t.iter()
                .any(|x| matches!(x, Tip::ModelBiggerThanRam { .. }))
        );
        assert!(
            t.iter()
                .any(|x| matches!(x, Tip::CloseApps(a) if a.len() == 2))
        );
        assert!(t.iter().any(|x| matches!(x, Tip::IntegratedOnly { .. })));
        assert!(!t.iter().any(|x| matches!(x, Tip::GpuNotUsed { .. })));
        assert!(!t.iter().any(|x| matches!(x, Tip::DiskNearlyFull { .. })));
    }

    #[test]
    fn idle_discrete_card_is_reported() {
        let gpus = [gpu("AMD Radeon RX 6700 XT", Vendor::Amd, Some(12.0))];
        let t = tips(&TipInput {
            model_gb: Some(6.0),
            ram_total_gb: 32.0,
            ram_used_gb: 8.0,
            swap_used_gb: 0.0,
            gpus: &gpus,
            backend_in_use: Some("cpu"),
            others: &[],
            disk_free_gb: Some(500.0),
        });
        assert_eq!(
            t,
            vec![Tip::GpuNotUsed {
                name: "AMD Radeon RX 6700 XT".into()
            }]
        );
    }

    #[test]
    fn a_healthy_machine_gets_no_advice() {
        let gpus = [gpu("RTX 4070", Vendor::Nvidia, Some(12.0))];
        let t = tips(&TipInput {
            model_gb: Some(6.0),
            ram_total_gb: 32.0,
            ram_used_gb: 10.0,
            swap_used_gb: 0.0,
            gpus: &gpus,
            backend_in_use: Some("cuda"),
            others: &[("chrome.exe".into(), 2.0)],
            disk_free_gb: Some(300.0),
        });
        assert!(t.is_empty());
    }
}
