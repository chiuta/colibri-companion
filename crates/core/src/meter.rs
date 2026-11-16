//! What one answer cost: tokens, speed and the machine's load while it was written.
//!
//! The meter is fed with the pieces of the answer as they arrive and with a machine reading
//! about once a second. It keeps only a few numbers (peaks, sums, a short speed history), so a
//! very long answer costs no more memory than a short one. All times are milliseconds since the
//! answer was asked for, passed in by the caller, so the arithmetic is testable without a clock.
//!
//! Limits, stated plainly: a server that sends no usage figures gives only an estimate of the
//! tokens (one piece is about one token); a server on another machine is measured by speed and
//! tokens only, because its processor and memory are not ours to read.

use crate::chat::StreamSummary;
use crate::resources::Snapshot;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Speed is averaged over this many milliseconds to show "now".
const WINDOW_MS: u64 = 2_000;
/// One point of the speed history per this many milliseconds; the history is capped.
const BIN_MS: u64 = 500;
const MAX_BINS: usize = 240;
/// Recent pieces kept for the rolling speed (older ones are dropped).
const MAX_RECENT: usize = 4_096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Answer,
    Reasoning,
}

/// One reading of the machine, reduced to what the meter shows.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sample {
    /// The whole machine, 0..100.
    pub cpu_percent: f32,
    /// Cores kept busy by the engine (100 % = one core); `None` when the engine is not ours to read.
    pub engine_cores: Option<f32>,
    pub engine_rss_gb: Option<f64>,
    pub ram_used_gb: f64,
    pub ram_total_gb: f64,
    pub swap_used_gb: f64,
    pub gpu_util_percent: Option<f32>,
    pub vram_used_gb: Option<f64>,
    pub vram_total_gb: Option<f64>,
}

impl Sample {
    /// The same reading with every figure made usable: a NaN, an infinity or a negative number
    /// (a driver or a hand-built value can give one) becomes "unknown" or zero, and percentages
    /// stay within 0..100, so one bad reading can never poison a peak for the rest of the answer.
    pub fn sanitized(self) -> Sample {
        fn pct(x: f32) -> f32 {
            if x.is_finite() {
                x.clamp(0.0, 100.0)
            } else {
                0.0
            }
        }
        fn opt32(x: Option<f32>) -> Option<f32> {
            x.filter(|v| v.is_finite() && *v >= 0.0)
        }
        fn opt64(x: Option<f64>) -> Option<f64> {
            x.filter(|v| v.is_finite() && *v >= 0.0)
        }
        Sample {
            cpu_percent: pct(self.cpu_percent),
            engine_cores: opt32(self.engine_cores),
            engine_rss_gb: opt64(self.engine_rss_gb),
            ram_used_gb: finite(self.ram_used_gb),
            ram_total_gb: finite(self.ram_total_gb),
            swap_used_gb: finite(self.swap_used_gb),
            gpu_util_percent: opt32(self.gpu_util_percent).map(|g| g.min(100.0)),
            vram_used_gb: opt64(self.vram_used_gb),
            vram_total_gb: opt64(self.vram_total_gb),
        }
    }

    /// `engine_pid` is the process of the local engine, if there is one.
    pub fn from_snapshot(s: &Snapshot, engine_pid: Option<u32>) -> Sample {
        let engine = engine_pid.and_then(|pid| s.procs.iter().find(|p| p.pid == pid));
        let gpu = s.gpus.first();
        Sample {
            cpu_percent: s.cpu_percent,
            engine_cores: engine.map(|p| p.cpu_percent / 100.0),
            engine_rss_gb: engine.map(|p| p.rss_gb),
            ram_used_gb: s.ram_used_gb,
            ram_total_gb: s.ram_total_gb,
            swap_used_gb: s.swap_used_gb,
            gpu_util_percent: gpu.and_then(|g| g.util_percent),
            vram_used_gb: gpu.map(|g| g.vram_used_gb),
            vram_total_gb: gpu.map(|g| g.vram_total_gb),
        }
    }
}

/// What is kept with the answer once it is done.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnswerStats {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    /// `true` when `completion_tokens` was counted from the pieces received, not reported by the server.
    pub tokens_estimated: bool,
    pub answer_chars: u64,
    pub reasoning_chars: u64,
    pub first_token_ms: Option<u64>,
    pub total_ms: u64,
    pub avg_tps: Option<f64>,
    pub peak_tps: Option<f64>,
    pub context_tokens: Option<u64>,
    pub samples: u32,
    pub cpu_avg: Option<f32>,
    pub cpu_peak: Option<f32>,
    pub engine_cores_peak: Option<f32>,
    pub engine_rss_peak_gb: Option<f64>,
    pub ram_peak_gb: Option<f64>,
    pub swap_peak_gb: Option<f64>,
    pub gpu_avg: Option<f32>,
    pub gpu_peak: Option<f32>,
    pub vram_peak_gb: Option<f64>,
    pub vram_total_gb: Option<f64>,
}

impl AnswerStats {
    /// How much of the engine's context window this answer used (prompt plus answer), 0..1+.
    pub fn context_fill(&self) -> Option<f32> {
        // a hand-edited file can hold any number; adding must never overflow
        let used = self
            .prompt_tokens?
            .saturating_add(self.completion_tokens.unwrap_or(0));
        let ctx = self.context_tokens.filter(|c| *c > 0)?;
        Some(used as f32 / ctx as f32)
    }

    /// Share of the written text that was "thinking", 0..1.
    pub fn reasoning_share(&self) -> Option<f32> {
        let all = self.answer_chars.saturating_add(self.reasoning_chars);
        (all > 0).then(|| self.reasoning_chars as f32 / all as f32)
    }
}

/// The figures to draw while the answer is still being written.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Live {
    pub elapsed_ms: u64,
    pub first_token_ms: Option<u64>,
    pub tokens_out: u64,
    pub tps_now: Option<f64>,
    pub tps_avg: Option<f64>,
    pub tps_history: Vec<f32>,
    pub last: Option<Sample>,
    pub stats: AnswerStats,
}

#[derive(Clone, Debug, Default)]
pub struct Meter {
    context_tokens: Option<u64>,
    prompt_tokens: Option<u64>,
    first_token_ms: Option<u64>,
    last_ms: u64,
    pieces: u64,
    answer_chars: u64,
    reasoning_chars: u64,
    /// (time, pieces so far), newest last.
    recent: VecDeque<(u64, u64)>,
    bins: VecDeque<f32>,
    bin_start_ms: u64,
    bin_pieces: u64,
    peak_tps: f64,
    samples: u32,
    cpu_sum: f64,
    gpu_sum: f64,
    gpu_n: u32,
    last_sample: Option<Sample>,
    peaks: Peaks,
}

#[derive(Clone, Copy, Debug, Default)]
struct Peaks {
    cpu: f32,
    cores: Option<f32>,
    rss: Option<f64>,
    ram: f64,
    swap: f64,
    gpu: Option<f32>,
    vram: Option<f64>,
    vram_total: Option<f64>,
}

fn max_opt<T: PartialOrd + Copy>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(x), Some(y)) => Some(if y > x { y } else { x }),
        (x, None) => x,
        (None, y) => y,
    }
}

fn finite(x: f64) -> f64 {
    if x.is_finite() && x >= 0.0 { x } else { 0.0 }
}

impl Meter {
    /// `context_tokens`: the engine's context size, when it said.
    pub fn new(context_tokens: Option<u64>) -> Self {
        Meter {
            context_tokens,
            ..Meter::default()
        }
    }

    /// The server counted the prompt (some do before the first piece).
    pub fn set_prompt_tokens(&mut self, n: u64) {
        self.prompt_tokens = Some(n);
    }

    /// One piece of the answer arrived.
    pub fn delta(&mut self, now_ms: u64, kind: Kind, text: &str) {
        let now_ms = now_ms.max(self.last_ms);
        self.last_ms = now_ms;
        let chars = text.chars().count() as u64;
        match kind {
            Kind::Answer => self.answer_chars = self.answer_chars.saturating_add(chars),
            Kind::Reasoning => self.reasoning_chars = self.reasoning_chars.saturating_add(chars),
        }
        if self.first_token_ms.is_none() {
            self.first_token_ms = Some(now_ms);
            self.bin_start_ms = now_ms;
        }
        self.pieces += 1;
        self.recent.push_back((now_ms, self.pieces));
        while self.recent.len() > MAX_RECENT
            || self
                .recent
                .front()
                .is_some_and(|(t, _)| now_ms.saturating_sub(*t) > WINDOW_MS * 2)
        {
            self.recent.pop_front();
        }
        // Close every finished history bin; a long silence gives zero-speed bins. A huge gap is
        // handled by arithmetic, never by looping over it.
        let gap = now_ms.saturating_sub(self.bin_start_ms);
        if gap >= BIN_MS {
            let n = gap / BIN_MS;
            let tps = self.bin_pieces as f64 * 1000.0 / BIN_MS as f64;
            self.peak_tps = self.peak_tps.max(tps);
            self.push_bin(tps as f32);
            for _ in 1..n.min(MAX_BINS as u64) {
                self.push_bin(0.0);
            }
            self.bin_pieces = 0;
            self.bin_start_ms = self.bin_start_ms.saturating_add(n.saturating_mul(BIN_MS));
        }
        // this piece belongs to the bin that is open now, not to the one it just closed
        self.bin_pieces += 1;
    }

    fn push_bin(&mut self, v: f32) {
        if self.bins.len() == MAX_BINS {
            self.bins.pop_front();
        }
        self.bins.push_back(v);
    }

    /// A machine reading.
    pub fn sample(&mut self, s: Sample) {
        let s = s.sanitized();
        self.samples = self.samples.saturating_add(1);
        self.cpu_sum += f64::from(s.cpu_percent);
        let p = &mut self.peaks;
        p.cpu = p.cpu.max(s.cpu_percent);
        p.cores = max_opt(p.cores, s.engine_cores);
        p.rss = max_opt(p.rss, s.engine_rss_gb);
        p.ram = p.ram.max(s.ram_used_gb);
        p.swap = p.swap.max(s.swap_used_gb);
        if let Some(g) = s.gpu_util_percent {
            self.gpu_sum += f64::from(g);
            self.gpu_n = self.gpu_n.saturating_add(1);
            p.gpu = max_opt(p.gpu, Some(g));
        }
        p.vram = max_opt(p.vram, s.vram_used_gb);
        p.vram_total = max_opt(p.vram_total, s.vram_total_gb);
        self.last_sample = Some(s);
    }

    /// Speed over the last couple of seconds, from the pieces received.
    fn tps_now(&self, now_ms: u64) -> Option<f64> {
        let (t_last, n_last) = *self.recent.back()?;
        let from = now_ms.saturating_sub(WINDOW_MS);
        let (t0, n0) = self
            .recent
            .iter()
            .find(|(t, _)| *t >= from)
            .copied()
            .unwrap_or((t_last, n_last));
        // pieces after the first one in the window, over the time they took
        let dt = now_ms.max(t_last).saturating_sub(t0) as f64 / 1000.0;
        let dn = (n_last - n0) as f64;
        (dn >= 1.0 && dt >= 0.05).then(|| dn / dt)
    }

    fn tps_avg(&self, end_ms: u64) -> Option<f64> {
        let first = self.first_token_ms?;
        let dt = end_ms.saturating_sub(first) as f64 / 1000.0;
        (self.pieces > 1 && dt >= 0.05).then(|| (self.pieces - 1) as f64 / dt)
    }

    fn stats(&self, end_ms: u64) -> AnswerStats {
        let n = f64::from(self.samples.max(1));
        AnswerStats {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: (self.pieces > 0).then_some(self.pieces),
            tokens_estimated: true,
            answer_chars: self.answer_chars,
            reasoning_chars: self.reasoning_chars,
            first_token_ms: self.first_token_ms,
            total_ms: end_ms,
            avg_tps: self.tps_avg(end_ms),
            peak_tps: (self.peak_tps > 0.0).then_some(self.peak_tps),
            context_tokens: self.context_tokens,
            samples: self.samples,
            cpu_avg: (self.samples > 0).then(|| (self.cpu_sum / n) as f32),
            cpu_peak: (self.samples > 0).then_some(self.peaks.cpu),
            engine_cores_peak: self.peaks.cores,
            engine_rss_peak_gb: self.peaks.rss,
            ram_peak_gb: (self.samples > 0).then_some(self.peaks.ram),
            swap_peak_gb: (self.samples > 0).then_some(self.peaks.swap),
            gpu_avg: (self.gpu_n > 0).then(|| (self.gpu_sum / f64::from(self.gpu_n)) as f32),
            gpu_peak: self.peaks.gpu,
            vram_peak_gb: self.peaks.vram,
            vram_total_gb: self.peaks.vram_total,
        }
    }

    pub fn live(&self, now_ms: u64) -> Live {
        let now_ms = now_ms.max(self.last_ms);
        Live {
            elapsed_ms: now_ms,
            first_token_ms: self.first_token_ms,
            tokens_out: self.pieces,
            tps_now: self.tps_now(now_ms),
            tps_avg: self.tps_avg(now_ms),
            tps_history: self.bins.iter().copied().collect(),
            last: self.last_sample,
            stats: self.stats(now_ms),
        }
    }

    /// The answer ended. Figures the server reported win over the estimates.
    pub fn finish(&self, end_ms: u64, summary: Option<&StreamSummary>) -> AnswerStats {
        let end_ms = end_ms.max(self.last_ms);
        let mut s = self.stats(end_ms);
        if let Some(sum) = summary {
            if sum.prompt_tokens.is_some() {
                s.prompt_tokens = sum.prompt_tokens;
            }
            if let Some(c) = sum.completion_tokens {
                // the stream code says whether it counted pieces itself (a server without usage)
                s.tokens_estimated = sum.completion_estimated;
                s.completion_tokens = Some(c);
            }
            if sum.first_token_ms.is_some() {
                s.first_token_ms = sum.first_token_ms;
            }
            if sum.total_ms > 0 {
                s.total_ms = sum.total_ms;
            }
            if let Some(t) = sum.tokens_per_second() {
                s.avg_tps = Some(t);
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(m: &mut Meter, from_ms: u64, count: u64, step_ms: u64) {
        for i in 0..count {
            m.delta(from_ms + i * step_ms, Kind::Answer, "ab");
        }
    }

    #[test]
    fn speed_is_pieces_over_time_not_a_constant() {
        let mut m = Meter::new(Some(1000));
        feed(&mut m, 1000, 51, 100); // 10 pieces a second for 5 s
        let l = m.live(6000);
        assert_eq!(l.tokens_out, 51);
        let avg = l.tps_avg.expect("avg");
        assert!((avg - 10.0).abs() < 0.5, "{avg}");
        let now = l.tps_now.expect("now");
        assert!((now - 10.0).abs() < 1.0, "{now}");
        assert_eq!(l.first_token_ms, Some(1000));
        assert!(!l.tps_history.is_empty());
    }

    #[test]
    fn a_stall_shows_in_the_speed_now() {
        let mut m = Meter::new(None);
        feed(&mut m, 0, 50, 20);
        let fast = m.live(1000).tps_now.expect("fast");
        // nothing for ten seconds: the speed "now" must not still claim the old speed
        let slow = m.live(11_000).tps_now;
        assert!(slow.is_none_or(|s| s < fast / 5.0), "{slow:?} vs {fast}");
    }

    #[test]
    fn no_pieces_means_no_speed_and_no_division_by_zero() {
        let m = Meter::new(None);
        let l = m.live(0);
        assert_eq!((l.tps_now, l.tps_avg, l.first_token_ms), (None, None, None));
        let s = m.finish(0, None);
        assert_eq!(s.completion_tokens, None);
        let mut one = Meter::new(None);
        one.delta(5, Kind::Answer, "x");
        assert_eq!(one.live(5).tps_avg, None);
        // time going backwards is clamped
        one.delta(1, Kind::Answer, "y");
        assert!(one.live(0).elapsed_ms >= 5);
    }

    #[test]
    fn server_figures_replace_estimates() {
        let mut m = Meter::new(Some(2000));
        feed(&mut m, 100, 10, 50);
        let sum = StreamSummary {
            prompt_tokens: Some(300),
            completion_tokens: Some(12),
            first_token_ms: Some(80),
            total_ms: 700,
            ..StreamSummary::default()
        };
        let s = m.finish(700, Some(&sum));
        assert_eq!(
            (s.prompt_tokens, s.completion_tokens),
            (Some(300), Some(12))
        );
        assert!(!s.tokens_estimated);
        assert_eq!(s.first_token_ms, Some(80));
        let f = s.context_fill().expect("fill");
        assert!((f - 312.0 / 2000.0).abs() < 1e-6);
        // without usage the count stays an estimate
        let s = m.finish(700, None);
        assert!(s.tokens_estimated);
        assert_eq!(s.completion_tokens, Some(10));
        assert_eq!(s.context_fill(), None); // prompt tokens unknown
    }

    #[test]
    fn reasoning_share_counts_characters_of_each_kind() {
        let mut m = Meter::new(None);
        m.delta(0, Kind::Reasoning, "abcd");
        m.delta(10, Kind::Answer, "ef");
        let s = m.finish(10, None);
        assert_eq!((s.reasoning_chars, s.answer_chars), (4, 2));
        assert!((s.reasoning_share().unwrap() - 4.0 / 6.0).abs() < 1e-6);
        assert_eq!(AnswerStats::default().reasoning_share(), None);
    }

    #[test]
    fn machine_readings_keep_peaks_and_averages() {
        let mut m = Meter::new(None);
        for (cpu, gpu, vram) in [(10.0, 20.0, 3.0), (90.0, 80.0, 5.0), (50.0, 40.0, 4.0)] {
            m.sample(Sample {
                cpu_percent: cpu,
                engine_cores: Some(cpu / 10.0),
                engine_rss_gb: Some(9.0 + f64::from(cpu) / 100.0),
                ram_used_gb: 10.0 + f64::from(cpu) / 10.0,
                ram_total_gb: 16.0,
                swap_used_gb: 0.5,
                gpu_util_percent: Some(gpu),
                vram_used_gb: Some(vram),
                vram_total_gb: Some(8.0),
            });
        }
        let s = m.finish(1000, None);
        assert_eq!(s.samples, 3);
        assert_eq!(s.cpu_peak, Some(90.0));
        assert!((s.cpu_avg.unwrap() - 50.0).abs() < 0.01);
        assert_eq!(s.gpu_peak, Some(80.0));
        assert!((s.gpu_avg.unwrap() - 46.666).abs() < 0.1);
        assert_eq!(s.vram_peak_gb, Some(5.0));
        assert_eq!(s.engine_cores_peak, Some(9.0));
        assert!((s.engine_rss_peak_gb.unwrap() - 9.9).abs() < 1e-9);
        assert!((s.ram_peak_gb.unwrap() - 19.0).abs() < 1e-9);
    }

    #[test]
    fn no_readings_means_unknown_not_zero() {
        let s = Meter::new(None).finish(10, None);
        assert_eq!(
            (s.cpu_avg, s.cpu_peak, s.ram_peak_gb, s.gpu_peak),
            (None, None, None, None)
        );
    }

    #[test]
    fn hostile_numbers_and_endless_answers_stay_bounded() {
        let mut m = Meter::new(Some(0));
        m.sample(Sample {
            cpu_percent: f32::NAN,
            ram_used_gb: f64::NAN,
            swap_used_gb: f64::INFINITY,
            ..Sample::default()
        });
        let s = m.finish(1, None);
        assert!(s.ram_peak_gb.is_some_and(f64::is_finite));
        assert_eq!(s.context_fill(), None);
        let mut m = Meter::new(None);
        feed(&mut m, 0, 200_000, 7);
        assert!(m.recent.len() <= MAX_RECENT);
        assert!(m.bins.len() <= MAX_BINS);
        // a huge gap between two pieces does not loop for ever
        m.delta(u64::MAX / 2, Kind::Answer, "z");
        assert!(m.bins.len() <= MAX_BINS);
    }

    #[test]
    fn a_steady_ten_a_second_stream_peaks_at_ten_not_more() {
        // the piece that closes a half-second bin belongs to the next bin: peak must not read 12
        let mut m = Meter::new(None);
        feed(&mut m, 0, 101, 100);
        let s = m.finish(10_000, None);
        let peak = s.peak_tps.expect("peak");
        assert!((peak - 10.0).abs() < 0.01, "{peak}");
        assert!(m.live(10_000).tps_history.iter().all(|v| *v <= 10.01));
    }

    #[test]
    fn one_bad_reading_cannot_poison_the_peaks() {
        let mut m = Meter::new(None);
        m.sample(Sample {
            cpu_percent: f32::INFINITY,
            engine_cores: Some(f32::NAN),
            engine_rss_gb: Some(f64::NAN),
            gpu_util_percent: Some(f32::NAN),
            vram_used_gb: Some(f64::INFINITY),
            vram_total_gb: Some(-4.0),
            ..Sample::default()
        });
        m.sample(Sample {
            cpu_percent: 40.0,
            engine_cores: Some(2.0),
            engine_rss_gb: Some(3.0),
            gpu_util_percent: Some(50.0),
            vram_used_gb: Some(1.5),
            vram_total_gb: Some(8.0),
            ..Sample::default()
        });
        let s = m.finish(1000, None);
        assert_eq!(s.cpu_peak, Some(40.0));
        assert_eq!(s.engine_cores_peak, Some(2.0));
        assert_eq!(s.engine_rss_peak_gb, Some(3.0));
        assert_eq!(s.gpu_peak, Some(50.0));
        assert_eq!(s.vram_peak_gb, Some(1.5));
        assert_eq!(s.vram_total_gb, Some(8.0));
        // a percentage over 100 is clamped, never shown as 250 %
        let over = Sample {
            cpu_percent: 250.0,
            ..Sample::default()
        }
        .sanitized();
        assert_eq!(over.cpu_percent, 100.0);
    }

    #[test]
    fn numbers_from_a_hand_edited_file_never_overflow() {
        let s = AnswerStats {
            prompt_tokens: Some(u64::MAX),
            completion_tokens: Some(u64::MAX),
            context_tokens: Some(10),
            answer_chars: u64::MAX,
            reasoning_chars: u64::MAX,
            ..AnswerStats::default()
        };
        assert!(s.context_fill().is_some_and(f32::is_finite));
        assert!(s.reasoning_share().is_some_and(f32::is_finite));
    }

    #[test]
    fn server_count_that_equals_the_piece_count_is_still_the_servers() {
        let mut m = Meter::new(None);
        feed(&mut m, 0, 5, 10);
        let reported = StreamSummary {
            completion_tokens: Some(5),
            total_ms: 100,
            ..StreamSummary::default()
        };
        assert!(!m.finish(100, Some(&reported)).tokens_estimated);
        let counted = StreamSummary {
            completion_estimated: true,
            ..reported
        };
        assert!(m.finish(100, Some(&counted)).tokens_estimated);
    }

    #[test]
    fn stats_survive_a_round_trip_and_old_files_without_them_load() {
        let s = Meter::new(Some(10)).finish(5, None);
        let j = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<AnswerStats>(&j).unwrap(), s);
        assert_eq!(
            serde_json::from_str::<AnswerStats>("{}").unwrap(),
            AnswerStats::default()
        );
    }
}
