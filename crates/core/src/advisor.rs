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

/// How a catalogue model would run with a given amount of memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierFit {
    /// Memory reaches what colibri calls good for this model.
    Good,
    /// Above the minimum but below good: it runs, with less cache, so more reading from the disk.
    Tight,
    /// Below the minimum.
    No,
}

/// Uses the catalogue's own memory figures (`ram_min_gb`, `ram_good_gb`).
pub fn tier_fit(ram_gb: f64, min_gb: f64, good_gb: f64) -> TierFit {
    if ram_gb >= good_gb {
        TierFit::Good
    } else if ram_gb >= min_gb {
        TierFit::Tight
    } else {
        TierFit::No
    }
}

#[cfg(test)]
mod tier_tests {
    use super::*;

    #[test]
    fn tiers_follow_the_catalogue_figures() {
        // DeepSeek V4 Flash: min 16, good 32
        assert_eq!(tier_fit(16.0, 16.0, 32.0), TierFit::Tight);
        assert_eq!(tier_fit(32.0, 16.0, 32.0), TierFit::Good);
        assert_eq!(tier_fit(8.0, 16.0, 32.0), TierFit::No);
    }
}

/// Why a model is recommended for this computer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// It runs entirely from memory: the fastest way.
    FromMemory,
    /// It reads experts from the drive, and the drive tested fast enough for that.
    FastDrive,
}

/// The most capable catalogue chat model this computer can run well.
/// Without a fast drive (or before the drive was tested) only models that run from memory are
/// offered; with a fast drive a streaming model is offered when memory reaches colibri's "good".
/// "Most capable" is taken as the largest download, the only size the catalogue gives.
pub fn recommend<'a>(
    models: &'a [crate::coli::ModelInfo],
    ram_total_gb: f64,
    drive: Option<crate::diskbench::DiskClass>,
) -> Option<(&'a crate::coli::ModelInfo, Why)> {
    let usable = usable_ram_gb(ram_total_gb);
    let chat = |m: &&crate::coli::ModelInfo| m.is_chat_model() && m.status != "unsupported";
    let in_memory = models
        .iter()
        .filter(chat)
        .filter(|m| !m.streams_from_disk() && m.download_gb <= usable)
        .max_by(|a, b| a.download_gb.total_cmp(&b.download_gb));
    let streaming = if drive == Some(crate::diskbench::DiskClass::Fast) {
        models
            .iter()
            .filter(chat)
            .filter(|m| {
                m.streams_from_disk()
                    && tier_fit(ram_total_gb, m.ram_min_gb, m.ram_good_gb) == TierFit::Good
            })
            .max_by(|a, b| a.download_gb.total_cmp(&b.download_gb))
    } else {
        None
    };
    match (streaming, in_memory) {
        (Some(m), _) => Some((m, Why::FastDrive)),
        (None, Some(m)) => Some((m, Why::FromMemory)),
        _ => None,
    }
}

#[cfg(test)]
mod recommend_tests {
    use super::*;
    use crate::coli::ModelInfo;
    use crate::diskbench::DiskClass;

    fn m(id: &str, dl: f64, min: f64, good: f64, ssd: bool) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            download_gb: dl,
            ram_min_gb: min,
            ram_good_gb: good,
            kind: "text".into(),
            runs: if ssd {
                "streams experts from the SSD".into()
            } else {
                "from RAM".into()
            },
            ..Default::default()
        }
    }

    #[test]
    fn follows_memory_and_drive() {
        let cat = [
            m("small", 8.0, 6.0, 10.0, false),
            m("mid", 19.0, 8.0, 18.0, false),
            m("giant", 167.0, 16.0, 32.0, true),
        ];
        // 16 GB: nothing streaming is "good" yet, and the in-memory pick must fit
        assert_eq!(recommend(&cat, 16.0, None).unwrap().0.id, "small");
        assert_eq!(
            recommend(&cat, 16.0, Some(DiskClass::Fast)).unwrap().0.id,
            "small"
        );
        // 32 GB with a fast drive: the giant is good
        let (g, why) = recommend(&cat, 32.0, Some(DiskClass::Fast)).unwrap();
        assert_eq!((g.id.as_str(), why), ("giant", Why::FastDrive));
        // 32 GB with a slow drive: stay in memory
        assert_eq!(
            recommend(&cat, 32.0, Some(DiskClass::Slow)).unwrap().0.id,
            "mid"
        );
    }
}
