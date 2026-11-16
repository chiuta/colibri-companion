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
        .filter(|v| v.is_finite() && *v > 0.0)
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

/// The context the recommendation is judged at (tokens). An ordinary chat, not a long document.
const RECOMMEND_CTX: u32 = 8192;

/// The most capable catalogue chat model this computer can run well.
///
/// "Run well" means the speed estimate ([`crate::estimate::estimate`]) puts the model in the
/// usable or fast class on this computer, so a recommendation and the estimate shown next to it
/// can never contradict each other. A model that runs from memory is judged with everything in
/// memory; a model that streams its experts is offered only when the drive tested fast, memory
/// reaches colibri's "good" figure and the estimate, judged at the slowest speed of the fast
/// class, is still at least usable. Colibri's own README measures its 167 GB DeepSeek V4 Flash at
/// about 1 token per second on 32 GB, so on ordinary machines this offers memory-resident models.
/// "Most capable" is taken as the largest download, the only size the catalogue gives; when a
/// streaming and an in-memory model are both eligible the larger one wins and a tie goes to the
/// one that runs from memory.
pub fn recommend(
    models: &[crate::coli::ModelInfo],
    ram_total_gb: f64,
    drive: Option<crate::diskbench::DiskClass>,
) -> Option<(&crate::coli::ModelInfo, Why)> {
    use crate::estimate::{Bottleneck, Drive, ModelFacts, Speed, estimate};
    // a catalogue entry with no usable size (missing, zero, NaN) cannot be judged, so it is not offered
    let chat = |m: &&crate::coli::ModelInfo| {
        m.is_chat_model()
            && m.status != "unsupported"
            && m.download_gb.is_finite()
            && m.download_gb > 0.0
    };
    let drive_figure = match drive {
        Some(c) if c == crate::diskbench::DiskClass::Fast => Drive::Measured(c.floor_mbs()),
        _ => Drive::Unknown,
    };
    // the estimate's own verdict, so the two can never disagree; `want` is where it must run
    let runs_well = |m: &crate::coli::ModelInfo, want: Bottleneck, drive: Drive| {
        let e = estimate(&ModelFacts::from(m), ram_total_gb, drive, RECOMMEND_CTX);
        matches!(e.speed, Speed::Usable | Speed::Fast)
            && e.bottleneck == want
            && e.tok_s_high >= crate::estimate::USABLE_TOK_S
    };
    // Whatever the catalogue says about how it runs, a model that fits in memory whole runs from
    // memory, and the drive does not matter once it is loaded.
    let in_memory = models
        .iter()
        .filter(chat)
        .filter(|m| runs_well(m, Bottleneck::RamResident, Drive::Unknown))
        .max_by(|a, b| a.download_gb.total_cmp(&b.download_gb));
    let streaming = if drive == Some(crate::diskbench::DiskClass::Fast) {
        models
            .iter()
            .filter(chat)
            .filter(|m| {
                m.streams_from_disk()
                    && tier_fit(ram_total_gb, m.ram_min_gb, m.ram_good_gb) == TierFit::Good
                    && runs_well(m, Bottleneck::SsdBound, drive_figure)
            })
            .max_by(|a, b| a.download_gb.total_cmp(&b.download_gb))
    } else {
        None
    };
    match (streaming, in_memory) {
        (Some(s), Some(m)) if m.download_gb >= s.download_gb => Some((m, Why::FromMemory)),
        (Some(s), _) => Some((s, Why::FastDrive)),
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

    /// The shipped catalogue's chat models (c/setup_catalog.py, 2026-11): id, download, dense,
    /// memory minimum and good figure, streams from the drive.
    const CATALOGUE: [(&str, f64, f64, f64, f64, bool); 11] = [
        ("qwen36-35b", 23.1, 3.5, 10.0, 20.0, false),
        ("qwen3-coder-30b", 19.4, 3.6, 8.0, 18.0, false),
        ("deepseek-v4-flash-reap", 84.7, 7.9, 16.0, 32.0, true),
        ("deepseek-v4-flash", 166.9, 7.9, 16.0, 32.0, true),
        ("qwen38-flash-next", 185.6, 8.0, 24.0, 32.0, true),
        ("mimo-v2.6-flash", 171.8, 24.0, 32.0, 52.0, true),
        ("glm-5.2", 429.3, 9.9, 16.0, 24.0, true),
        ("glm-5.3", 419.3, 9.9, 16.0, 24.0, true),
        ("mimo-v2.6-pro", 563.6, 32.4, 54.0, 64.0, true),
        ("inkling", 514.1, 49.4, 120.0, 128.0, true),
        ("kimi-k3", 1561.0, 20.0, 32.0, 64.0, true),
    ];

    fn real(id: &str, dl: f64, dense: f64, min: f64, good: f64, ssd: bool) -> ModelInfo {
        ModelInfo {
            dense_gb: dense,
            ..m(id, dl, min, good, ssd)
        }
    }

    fn catalogue() -> Vec<ModelInfo> {
        CATALOGUE
            .iter()
            .map(|&(id, dl, de, mn, gd, ssd)| real(id, dl, de, mn, gd, ssd))
            .collect()
    }

    /// What `recommend` promises, checked against the estimate it must agree with.
    fn agrees_with_estimate(pick: (&ModelInfo, Why), ram: f64, drive: Option<DiskClass>) {
        use crate::estimate::{Bottleneck, Drive, ModelFacts, Speed, USABLE_TOK_S, estimate};
        let (model, why) = pick;
        // memory-resident picks are judged without the drive (it does not matter once loaded),
        // streaming picks at the slowest speed of the class the drive test gave
        let figure = match (why, drive) {
            (Why::FastDrive, Some(c)) => Drive::Measured(c.floor_mbs()),
            _ => Drive::Unknown,
        };
        let e = estimate(&ModelFacts::from(model), ram, figure, RECOMMEND_CTX);
        assert!(
            matches!(e.speed, Speed::Usable | Speed::Fast),
            "{} on {ram} GB {drive:?}: recommended but estimated {e:?}",
            model.id
        );
        assert!(e.tok_s_high >= USABLE_TOK_S, "{} {e:?}", model.id);
        match why {
            Why::FromMemory => {
                assert_eq!(e.bottleneck, Bottleneck::RamResident, "{}", model.id);
            }
            Why::FastDrive => {
                assert!(model.streams_from_disk() && drive == Some(DiskClass::Fast));
                assert_eq!(e.bottleneck, Bottleneck::SsdBound, "{}", model.id);
                assert_eq!(
                    tier_fit(ram, model.ram_min_gb, model.ram_good_gb),
                    TierFit::Good
                );
            }
        }
    }

    #[test]
    fn follows_memory_and_drive() {
        let cat = catalogue();
        // 16 GB: not even the smallest catalogue model fits whole, and nothing that streams is
        // usable there (colibri's README: 5 tok/s at best with a reduced cache), so no offer
        for d in [None, Some(DiskClass::Slow), Some(DiskClass::Fast)] {
            assert!(recommend(&cat, 16.0, d).is_none(), "{d:?}");
        }
        // 32 GB with a fast drive: DeepSeek V4 Flash "runs" (colibri measured 0.93 tok/s) but the
        // estimate does not call that usable, so the memory-resident model is offered instead
        for d in [None, Some(DiskClass::Slow), Some(DiskClass::Fast)] {
            let (p, why) = recommend(&cat, 32.0, d).unwrap();
            assert_eq!(
                (p.id.as_str(), why),
                ("qwen36-35b", Why::FromMemory),
                "{d:?}"
            );
        }
    }

    #[test]
    fn a_streaming_model_is_offered_when_the_estimate_supports_it() {
        // a small streaming entry: 40 GB download on 64 GB with a fast drive
        let cat = [
            real("small", 8.0, 1.5, 6.0, 10.0, false),
            real("streamer", 60.0, 4.0, 12.0, 24.0, true),
        ];
        let (p, why) = recommend(&cat, 64.0, Some(DiskClass::Fast)).unwrap();
        assert_eq!((p.id.as_str(), why), ("streamer", Why::FastDrive));
        agrees_with_estimate((p, why), 64.0, Some(DiskClass::Fast));
        // the same machine without a drive test gets the memory-resident model
        let (p, why) = recommend(&cat, 64.0, None).unwrap();
        assert_eq!((p.id.as_str(), why), ("small", Why::FromMemory));
    }

    #[test]
    fn a_recommendation_never_contradicts_the_estimate_over_the_whole_catalogue() {
        let cat = catalogue();
        for ram in [
            8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 512.0,
        ] {
            for d in [
                None,
                Some(DiskClass::Slow),
                Some(DiskClass::Fair),
                Some(DiskClass::Fast),
            ] {
                if let Some(pick) = recommend(&cat, ram, d) {
                    agrees_with_estimate(pick, ram, d);
                }
            }
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next()
        }
    }

    #[test]
    fn property_random_catalogues_never_contradict_and_more_memory_never_offers_less() {
        let mut r = Rng(0xD1B54A32D192ED03);
        for _ in 0..1500 {
            let n = 1 + (r.next() * 8.0) as usize;
            let cat: Vec<ModelInfo> = (0..n)
                .map(|i| {
                    let dl = r.range(1.0, 700.0);
                    let dense = if r.next() < 0.2 {
                        0.0
                    } else {
                        r.range(0.5, dl * 0.5)
                    };
                    let min = r.range(4.0, 64.0);
                    real(
                        &format!("m{i}"),
                        dl,
                        dense,
                        min,
                        min + r.range(0.0, 64.0),
                        r.next() < 0.6,
                    )
                })
                .collect();
            let d = match (r.next() * 4.0) as u8 {
                0 => None,
                1 => Some(DiskClass::Slow),
                2 => Some(DiskClass::Fair),
                _ => Some(DiskClass::Fast),
            };
            let ram = r.range(4.0, 400.0);
            let base = recommend(&cat, ram, d);
            if let Some(p) = base {
                agrees_with_estimate(p, ram, d);
            }
            let more = ram + r.range(0.5, 200.0);
            let bigger = recommend(&cat, more, d);
            if let Some(p) = bigger {
                agrees_with_estimate(p, more, d);
            }
            let size = |x: Option<(&ModelInfo, Why)>| x.map_or(0.0, |(m, _)| m.download_gb);
            assert!(
                size(bigger) >= size(base) - 1e-9,
                "{ram} -> {more} GB, {d:?}: {:?} then {:?}",
                base.map(|b| &b.0.id),
                bigger.map(|b| &b.0.id)
            );
        }
    }

    #[test]
    fn an_entry_without_a_usable_size_is_never_recommended() {
        for bad in [f64::NAN, f64::INFINITY, 0.0, -4.0] {
            let cat = [
                m("broken", bad, 6.0, 10.0, true),
                m("ok", 8.0, 6.0, 10.0, false),
            ];
            let (pick, _) = recommend(&cat, 64.0, Some(DiskClass::Fast)).unwrap();
            assert_eq!(pick.id, "ok", "{bad}");
        }
        assert!(recommend(&[], 64.0, None).is_none());
        assert!(recommend(&[m("x", 8.0, 6.0, 10.0, false)], f64::NAN, None).is_none());
    }

    #[test]
    fn a_nonsense_graphics_memory_figure_is_ignored() {
        let g = |v| GpuInfo {
            name: "x".into(),
            vendor: Vendor::Nvidia,
            vram_gb: Some(v),
            integrated: false,
        };
        assert_eq!(
            best_vram_gb(&[g(f64::NAN), g(8.0), g(-3.0), g(f64::INFINITY)]),
            Some(8.0)
        );
        assert_eq!(best_vram_gb(&[g(f64::NAN)]), None);
    }
}
