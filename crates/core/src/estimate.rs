//! "Will it be fast?" before a download: a coarse speed ESTIMATE for a catalogue model on this
//! computer. A cost model, not a benchmark. Every figure that is not measured is an assumption
//! and widens the range. The only measured inputs are the RAM size and, when the drive test has
//! run, the drive speed.
//!
//! The model: the dense part of the network stays in RAM; the experts are cached in what is left
//! of RAM, and a token reads only a small share of them. Each token costs the time to read the
//! experts missing from the cache from the drive, plus the time to stream the active bytes from RAM.

use crate::advisor::{TierFit, tier_fit, usable_ram_gb};
use crate::coli::ModelInfo;

/// How fast the drive is known to be.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Drive {
    /// The app's drive test, in MB/s.
    Measured(f64),
    /// No measurement: a typical SSD is assumed and the range is widened.
    Unknown,
}

/// The catalogue figures the estimate needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelFacts {
    pub download_gb: f64,
    /// Part that always stays in RAM.
    pub dense_gb: f64,
    pub ram_min_gb: f64,
    pub ram_good_gb: f64,
}

impl From<&ModelInfo> for ModelFacts {
    fn from(m: &ModelInfo) -> Self {
        Self {
            download_gb: m.download_gb,
            dense_gb: m.dense_gb,
            ram_min_gb: m.ram_min_gb,
            ram_good_gb: m.ram_good_gb,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speed {
    Fast,
    Usable,
    Slow,
    TooSlow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bottleneck {
    /// Everything fits in RAM; memory bandwidth sets the speed.
    RamResident,
    /// Experts stream from the drive; the drive sets the speed.
    SsdBound,
    /// Even the always-resident part does not fit: the system would swap.
    SwapRisk,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Estimate {
    /// Tokens per second, low and high. Always an estimate.
    pub tok_s_low: f64,
    pub tok_s_high: f64,
    pub speed: Speed,
    pub bottleneck: Bottleneck,
    /// True when the drive speed was assumed, not measured.
    pub drive_assumed: bool,
    /// One English sentence for the reason.
    pub reason: String,
}

const ASSUMED_DRIVE_MBS: f64 = 1000.0;
/// Memory taken by the conversation per token of context (an assumption, in GB).
const STATE_GB_PER_TOKEN: f64 = 6.0e-5;
/// Share of the expert bytes one token reads: low and high assumptions.
const ACTIVE_SHARE: (f64, f64) = (0.02, 0.06);
/// RAM streaming speed assumptions, GB/s: slow and fast.
const RAM_GBS: (f64, f64) = (12.0, 40.0);

/// Share of expert lookups a cache of `frac` of all expert bytes serves. Points measured on one
/// shipped profile (qwen36-35b): an LRU cache (`low`) and the hottest experts pinned (`high`).
fn hit_rate(frac: f64) -> (f64, f64) {
    const LRU: [(f64, f64); 6] = [
        (0.0, 0.0),
        (0.05, 0.06),
        (0.10, 0.166),
        (0.25, 0.414),
        (0.50, 0.702),
        (1.0, 1.0),
    ];
    const PIN: [(f64, f64); 6] = [
        (0.0, 0.0),
        (0.05, 0.212),
        (0.10, 0.342),
        (0.25, 0.593),
        (0.50, 0.835),
        (1.0, 1.0),
    ];
    let f = frac.clamp(0.0, 1.0);
    (interp(&LRU, f), interp(&PIN, f))
}

fn interp(pts: &[(f64, f64)], x: f64) -> f64 {
    for w in pts.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if x <= x1 {
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    pts[pts.len() - 1].1
}

fn classify(low: f64, high: f64) -> Speed {
    let mid = (low.max(1e-6) * high.max(1e-6)).sqrt();
    if mid >= 10.0 {
        Speed::Fast
    } else if mid >= 3.0 {
        Speed::Usable
    } else if mid >= 1.0 {
        Speed::Slow
    } else {
        Speed::TooSlow
    }
}

/// Estimates the speed of `m` on a computer with `ram_total_gb` of memory.
pub fn estimate(m: &ModelFacts, ram_total_gb: f64, drive: Drive, ctx_tokens: u32) -> Estimate {
    let usable = usable_ram_gb(ram_total_gb);
    let state = f64::from(ctx_tokens) * STATE_GB_PER_TOKEN;
    let dense = m.dense_gb.max(0.0);
    let experts = (m.download_gb - dense).max(0.0);
    let (drive_mbs, drive_assumed) = match drive {
        Drive::Measured(v) if v > 0.0 => (v, false),
        _ => (ASSUMED_DRIVE_MBS, true),
    };
    // Unmeasured drive: a typical SSD could be four times slower or two times faster.
    let (d_lo, d_hi) = if drive_assumed {
        (drive_mbs * 0.25 / 1000.0, drive_mbs * 2.0 / 1000.0)
    } else {
        (drive_mbs * 0.8 / 1000.0, drive_mbs * 1.2 / 1000.0)
    };
    let cache = usable - dense - state;

    if cache < 0.0 {
        // Not even the dense part and the conversation fit: the system pages them in and out.
        let high = 0.3;
        let low = 0.02;
        return Estimate {
            tok_s_low: low,
            tok_s_high: high,
            speed: classify(low, high),
            bottleneck: Bottleneck::SwapRisk,
            drive_assumed,
            reason: format!(
                "About {:.1} GB must stay in memory but only {:.1} GB is usable here, so the system would swap.",
                dense + state,
                usable
            ),
        };
    }

    let fits_all = cache >= experts;
    let ram_time = |active_share: f64, ram_gbs: f64| (dense + experts * active_share) / ram_gbs;

    if fits_all {
        let low = 1.0 / ram_time(ACTIVE_SHARE.1, RAM_GBS.0);
        let high = 1.0 / ram_time(ACTIVE_SHARE.0, RAM_GBS.1);
        return Estimate {
            tok_s_low: low,
            tok_s_high: high,
            speed: classify(low, high),
            bottleneck: Bottleneck::RamResident,
            drive_assumed,
            reason: format!(
                "The whole model fits in memory ({:.0} GB of {:.1} GB usable), so memory speed sets the pace.",
                m.download_gb + state,
                usable
            ),
        };
    }

    let frac = if experts > 0.0 { cache / experts } else { 1.0 };
    let (hit_lo, hit_hi) = hit_rate(frac);
    // Pessimistic: low hit rate, large active share, slow drive, slow RAM.
    let t_lo =
        experts * ACTIVE_SHARE.1 * (1.0 - hit_lo) / d_lo + ram_time(ACTIVE_SHARE.1, RAM_GBS.0);
    let t_hi =
        experts * ACTIVE_SHARE.0 * (1.0 - hit_hi) / d_hi + ram_time(ACTIVE_SHARE.0, RAM_GBS.1);
    let (low, high) = (1.0 / t_lo, 1.0 / t_hi);
    let tier = tier_fit(ram_total_gb, m.ram_min_gb, m.ram_good_gb);
    let tail = match tier {
        TierFit::Good => "",
        TierFit::Tight => " Memory is below what the catalogue calls good for this model.",
        TierFit::No => " Memory is below the catalogue's minimum for this model.",
    };
    Estimate {
        tok_s_low: low,
        tok_s_high: high,
        speed: classify(low, high),
        bottleneck: Bottleneck::SsdBound,
        drive_assumed,
        reason: format!(
            "Only about {:.0}% of the experts fit in the memory cache, so the rest is read from the drive{}.{}",
            frac * 100.0,
            if drive_assumed {
                " (drive speed assumed, run the drive test)"
            } else {
                ""
            },
            tail
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(download: f64, dense: f64, min: f64, good: f64) -> ModelFacts {
        ModelFacts {
            download_gb: download,
            dense_gb: dense,
            ram_min_gb: min,
            ram_good_gb: good,
        }
    }

    #[test]
    fn a_small_model_in_plenty_of_memory_is_fast_or_usable() {
        let e = estimate(
            &facts(8.0, 2.0, 8.0, 12.0),
            32.0,
            Drive::Measured(3000.0),
            8192,
        );
        assert_eq!(e.bottleneck, Bottleneck::RamResident);
        assert!(matches!(e.speed, Speed::Fast | Speed::Usable), "{e:?}");
        assert!(e.tok_s_low < e.tok_s_high);
    }

    #[test]
    fn a_128_gb_model_on_16_gb_is_slow_or_worse_even_on_a_fast_drive() {
        let e = estimate(
            &facts(128.0, 10.0, 24.0, 96.0),
            16.0,
            Drive::Measured(3000.0),
            8192,
        );
        assert!(matches!(e.speed, Speed::Slow | Speed::TooSlow), "{e:?}");
        assert!(matches!(
            e.bottleneck,
            Bottleneck::SsdBound | Bottleneck::SwapRisk
        ));
        assert!(e.tok_s_high < 3.0, "{e:?}");
    }

    #[test]
    fn a_23_gb_model_on_15_7_gb_streams_and_the_drive_decides() {
        let m = facts(23.0, 3.6, 8.0, 24.0);
        let fast = estimate(&m, 15.7, Drive::Measured(3000.0), 8192);
        let slow = estimate(&m, 15.7, Drive::Measured(400.0), 8192);
        assert_eq!(fast.bottleneck, Bottleneck::SsdBound);
        assert!(fast.tok_s_high > slow.tok_s_high);
        assert!(slow.tok_s_low < fast.tok_s_low);
    }

    #[test]
    fn when_the_dense_part_does_not_fit_the_system_swaps() {
        let e = estimate(
            &facts(60.0, 14.0, 16.0, 40.0),
            16.0,
            Drive::Measured(3000.0),
            8192,
        );
        assert_eq!(e.bottleneck, Bottleneck::SwapRisk);
        assert_eq!(e.speed, Speed::TooSlow);
    }

    #[test]
    fn an_assumed_drive_widens_the_range_and_says_so() {
        let m = facts(23.0, 3.6, 8.0, 24.0);
        let measured = estimate(&m, 15.7, Drive::Measured(1000.0), 8192);
        let assumed = estimate(&m, 15.7, Drive::Unknown, 8192);
        assert!(assumed.drive_assumed && !measured.drive_assumed);
        assert!(assumed.tok_s_high / assumed.tok_s_low > measured.tok_s_high / measured.tok_s_low);
        assert!(assumed.reason.contains("drive test"));
    }

    #[test]
    fn a_longer_context_never_makes_it_faster() {
        let m = facts(23.0, 3.6, 8.0, 24.0);
        let short = estimate(&m, 15.7, Drive::Measured(1500.0), 4096);
        let long = estimate(&m, 15.7, Drive::Measured(1500.0), 65536);
        assert!(long.tok_s_high <= short.tok_s_high + 1e-9);
    }

    #[test]
    fn hit_rate_grows_with_the_cache_and_pinning_beats_lru() {
        let mut prev = (0.0, 0.0);
        for i in 0..=20 {
            let (l, p) = hit_rate(f64::from(i) / 20.0);
            assert!(l >= prev.0 && p >= prev.1 && p >= l - 1e-9);
            prev = (l, p);
        }
    }

    #[test]
    fn reads_the_figures_from_a_catalogue_entry() {
        let info = ModelInfo {
            download_gb: 19.4,
            dense_gb: 3.6,
            ram_min_gb: 8.0,
            ram_good_gb: 18.0,
            ..Default::default()
        };
        assert_eq!(ModelFacts::from(&info), facts(19.4, 3.6, 8.0, 18.0));
    }
}
