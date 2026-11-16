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
/// No drive reads faster than this (PCIe 5.0 x4 NVMe, MB/s). A larger "measurement" is the page
/// cache answering, not the drive, so it is not believed.
const MAX_DRIVE_MBS: f64 = 14_000.0;
/// Ceiling on any estimate, tokens per second: memory bandwidth cannot feed more than this even
/// for a toy model, and it keeps a zero-size catalogue entry from producing infinity.
const MAX_TOK_S: f64 = 1000.0;
/// A model whose catalogue entry has no dense size is assumed to keep this share of its download
/// resident (the shipped catalogue ranges from 5% to 45%).
const ASSUMED_DENSE_SHARE: f64 = 0.15;
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

fn finite_or(x: f64, default: f64) -> f64 {
    if x.is_finite() { x } else { default }
}

/// Tokens per second for a time per token, never above [`MAX_TOK_S`] and never infinite.
fn rate(t: f64) -> f64 {
    (1.0 / t.max(1.0 / MAX_TOK_S)).min(MAX_TOK_S)
}

/// The streaming case: `frac` of the expert bytes are cached, the rest is read from the drive.
/// Returns (low, high) tokens per second.
fn streaming_range(experts: f64, dense: f64, frac: f64, (d_lo, d_hi): (f64, f64)) -> (f64, f64) {
    let ram_time = |share: f64, gbs: f64| (dense + experts * share) / gbs;
    let (hit_lo, hit_hi) = hit_rate(frac);
    // Pessimistic: low hit rate, large active share, slow drive, slow RAM.
    let t_lo =
        experts * ACTIVE_SHARE.1 * (1.0 - hit_lo) / d_lo + ram_time(ACTIVE_SHARE.1, RAM_GBS.0);
    let t_hi =
        experts * ACTIVE_SHARE.0 * (1.0 - hit_hi) / d_hi + ram_time(ACTIVE_SHARE.0, RAM_GBS.1);
    (rate(t_lo), rate(t_hi))
}

/// Estimates the speed of `m` on a computer with `ram_total_gb` of memory.
///
/// Never panics and never returns NaN or infinity, whatever the inputs: a non-finite figure is
/// treated as unknown. Guarantees (checked by property tests): `0 < low <= high <= 1000`; more
/// memory, a faster measured drive or a smaller model never lower the estimate; more context or
/// a bigger resident part never raise it.
pub fn estimate(m: &ModelFacts, ram_total_gb: f64, drive: Drive, ctx_tokens: u32) -> Estimate {
    let ram_total_gb = finite_or(ram_total_gb, 0.0).clamp(0.0, 1.0e5);
    let usable = usable_ram_gb(ram_total_gb);
    let state = f64::from(ctx_tokens) * STATE_GB_PER_TOKEN;
    let download = finite_or(m.download_gb, 0.0).clamp(0.0, 1.0e5);
    let dense = if m.dense_gb.is_finite() && m.dense_gb > 0.0 {
        m.dense_gb
    } else {
        download * ASSUMED_DENSE_SHARE
    }
    .min(download);
    let experts = (download - dense).max(0.0);
    let (drive_mbs, drive_assumed) = match drive {
        Drive::Measured(v) if v.is_finite() && v > 0.0 => (v.min(MAX_DRIVE_MBS), false),
        _ => (ASSUMED_DRIVE_MBS, true),
    };
    // Unmeasured drive: from a slow SATA-class SSD (a quarter of the assumed figure) to a fast
    // PCIe 4.0 NVMe (six times it). The upper end is set by colibri's own README, which measures
    // 0.93 tok/s for DeepSeek V4 Flash on 32 GB: a drive capped at 2 GB/s cannot reach that.
    let d = if drive_assumed {
        (drive_mbs * 0.25 / 1000.0, drive_mbs * 6.0 / 1000.0)
    } else {
        (drive_mbs * 0.8 / 1000.0, drive_mbs * 1.2 / 1000.0)
    };
    let cache = usable - dense - state;

    if cache < 0.0 {
        // Not even the dense part and the conversation fit: the system pages them in and out.
        // Paging is worse than streaming with an empty cache, so the range is capped by what
        // that case gives, and a slow drive makes paging slow too.
        let (lo0, hi0) = streaming_range(experts, dense, 0.0, d);
        let (low, high) = (0.02f64.min(lo0), 0.3f64.min(hi0));
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

    if cache >= experts {
        let ram_time = |share: f64, gbs: f64| (dense + experts * share) / gbs;
        let low = rate(ram_time(ACTIVE_SHARE.1, RAM_GBS.0));
        let high = rate(ram_time(ACTIVE_SHARE.0, RAM_GBS.1));
        return Estimate {
            tok_s_low: low,
            tok_s_high: high,
            speed: classify(low, high),
            bottleneck: Bottleneck::RamResident,
            drive_assumed,
            reason: format!(
                "The whole model fits in memory ({:.0} GB of {:.1} GB usable), so memory speed sets the pace.",
                download + state,
                usable
            ),
        };
    }

    let frac = cache / experts; // experts > cache >= 0 here, so experts > 0
    let (low, high) = streaming_range(experts, dense, frac, d);
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

    fn sane(e: &Estimate) {
        assert!(e.tok_s_low.is_finite() && e.tok_s_high.is_finite(), "{e:?}");
        assert!(e.tok_s_low > 0.0, "{e:?}");
        assert!(e.tok_s_low <= e.tok_s_high + 1e-12, "{e:?}");
        assert!(e.tok_s_high <= MAX_TOK_S, "{e:?}");
    }

    fn le(a: f64, b: f64) -> bool {
        a <= b * (1.0 + 1e-9) + 1e-12
    }

    /// `worse` must not be faster than `better` at either end of the range.
    fn not_faster(worse: &Estimate, better: &Estimate, what: &str) {
        assert!(
            le(worse.tok_s_low, better.tok_s_low) && le(worse.tok_s_high, better.tok_s_high),
            "{what}: {worse:?} vs {better:?}"
        );
    }

    #[test]
    fn nonsense_inputs_never_give_nan_infinity_or_a_panic() {
        let bad = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -5.0, 0.0, 1e300];
        for &a in &bad {
            for &b in &bad {
                for &c in &bad {
                    for drive in [Drive::Measured(a), Drive::Unknown] {
                        let e = estimate(&facts(a, b, c, a), c, drive, u32::MAX);
                        sane(&e);
                        sane(&estimate(&facts(a, b, c, a), b, drive, 0));
                    }
                }
            }
        }
    }

    #[test]
    fn a_zero_size_entry_does_not_give_infinite_speed() {
        let e = estimate(
            &facts(0.0, 0.0, 0.0, 0.0),
            16.0,
            Drive::Measured(3000.0),
            8192,
        );
        sane(&e);
        assert!(e.tok_s_high <= MAX_TOK_S);
    }

    #[test]
    fn a_cached_drive_figure_is_not_believed_beyond_the_hardware() {
        let m = facts(167.0, 7.9, 16.0, 32.0);
        let real = estimate(&m, 16.0, Drive::Measured(MAX_DRIVE_MBS), 8192);
        let absurd = estimate(&m, 16.0, Drive::Measured(5.0e7), 8192);
        assert_eq!(real, absurd);
    }

    #[test]
    fn a_slow_drive_makes_paging_slower_too() {
        // dense part does not fit: SwapRisk. A 5 MB/s drive must not look better than a fast one,
        // and the swap case must never beat the same model with just enough memory.
        let m = facts(60.0, 14.0, 16.0, 40.0);
        let slow = estimate(&m, 16.0, Drive::Measured(5.0), 8192);
        let fast = estimate(&m, 16.0, Drive::Measured(3000.0), 8192);
        assert_eq!(slow.bottleneck, Bottleneck::SwapRisk);
        not_faster(&slow, &fast, "slow drive");
        let roomy = estimate(&m, 24.0, Drive::Measured(5.0), 8192);
        not_faster(&slow, &roomy, "more memory with a slow drive");
    }

    #[test]
    fn a_missing_dense_size_is_not_read_as_a_model_that_is_all_experts() {
        let unknown = estimate(&facts(8.0, 0.0, 6.0, 10.0), 32.0, Drive::Unknown, 8192);
        let known = estimate(&facts(8.0, 1.2, 6.0, 10.0), 32.0, Drive::Unknown, 8192);
        assert_eq!(unknown, known);
    }

    #[test]
    fn property_monotone_in_memory_drive_size_context_and_resident_part() {
        let mut r = Rng(0x9E3779B97F4A7C15);
        for _ in 0..4000 {
            let download = r.range(1.0, 600.0);
            let dense = r.range(0.5, download.min(60.0) * 0.99);
            let (min, good) = (r.range(4.0, 64.0), r.range(8.0, 128.0));
            let m = facts(download, dense, min, good);
            let ram = r.range(2.0, 192.0);
            let mbs = r.range(50.0, 12000.0);
            let ctx = (r.range(512.0, 200_000.0)) as u32;
            let base = estimate(&m, ram, Drive::Measured(mbs), ctx);
            sane(&base);
            let more_ram = estimate(&m, ram + r.range(0.1, 64.0), Drive::Measured(mbs), ctx);
            not_faster(&base, &more_ram, "more memory");
            let faster = estimate(&m, ram, Drive::Measured(mbs * r.range(1.01, 4.0)), ctx);
            not_faster(&base, &faster, "faster drive");
            let longer = estimate(
                &m,
                ram,
                Drive::Measured(mbs),
                ctx + r.range(1.0, 100_000.0) as u32,
            );
            not_faster(&longer, &base, "longer context");
            let bigger = facts(download + r.range(0.1, 300.0), dense, min, good);
            not_faster(
                &estimate(&bigger, ram, Drive::Measured(mbs), ctx),
                &base,
                "bigger model",
            );
            let heavier = facts(
                download + r.range(0.1, 30.0),
                dense + r.range(0.1, 30.0),
                min,
                good,
            );
            // a bigger resident part with the same experts: only the resident part grows
            let fixed_experts = facts(
                heavier.dense_gb + (download - dense),
                heavier.dense_gb,
                min,
                good,
            );
            not_faster(
                &estimate(&fixed_experts, ram, Drive::Measured(mbs), ctx),
                &base,
                "bigger resident part",
            );
        }
    }

    #[test]
    fn agrees_with_colibris_own_measurements_of_deepseek_v4_flash() {
        // README: 0.93 tok/s with 32 GB (Ryzen 7 5800X, CPU only), 1.24 with 63 GB (Ryzen 9 5950X).
        // 167 GB download, 7.9 GB resident, catalogue memory 16 / 32 GB. The drives are not stated,
        // so the drive is unknown and the range is the widest one.
        let m = facts(166.9, 7.9, 16.0, 32.0);
        for (ram, measured) in [(32.0, 0.93), (63.0, 1.24)] {
            let e = estimate(&m, ram, Drive::Unknown, 16384);
            assert!(
                e.tok_s_low <= measured && measured <= e.tok_s_high,
                "{ram} GB: measured {measured} outside {:.2}-{:.2}",
                e.tok_s_low,
                e.tok_s_high
            );
        }
    }
}
