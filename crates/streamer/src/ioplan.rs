//! A small analytic model of streaming experts from a drive. A cost model, not an I/O engine:
//! every default below is an ASSUMPTION about typical hardware, to be replaced by a measurement
//! (the app's drive test gives a MB/s figure).
//!
//! One request of `expert_bytes` takes `latency + expert_bytes / seq_bw`. With `q` requests in
//! flight the drive delivers at most `rand_bw` in total, so a wave of `q` requests lasts
//! `max(single request time, q * expert_bytes / rand_bw)`.
//!
//! The "overlapped" figures are a CEILING, not a prediction: they assume the experts of layer L+1
//! are already known while layer L computes (a perfect prefetcher). In a real engine they are
//! known only once the router of layer L+1 has run, unless a predictor guesses them (see
//! `prefetch`, which measures how often a guess would be right). The "serial" figures need no
//! such assumption.

/// Drive and model parameters. Bandwidths in bytes per second, times in seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub expert_bytes: f64,
    /// Speed of one large request on its own.
    pub seq_bw: f64,
    /// Ceiling the whole drive reaches with many requests in flight.
    pub rand_bw: f64,
    pub latency: f64,
    pub layers: u32,
    /// Experts read per layer per token.
    pub experts_per_layer: u32,
    /// Share of experts already in RAM (no read needed), 0..1.
    pub hit_rate: f64,
    /// Compute time per layer.
    pub compute: f64,
}

impl Params {
    /// Assumed typical NVMe drive (about 3 GB/s, 100 us latency).
    pub fn nvme() -> Self {
        Self::with_drive(3.0e9, 3.0e9, 100e-6)
    }

    /// Assumed typical SATA SSD (about 500 MB/s, 150 us latency).
    pub fn sata() -> Self {
        Self::with_drive(500e6, 500e6, 150e-6)
    }

    /// A drive known only by a measured MB/s (the app's drive test): taken as both the single
    /// request speed and the ceiling, which is the cautious reading.
    pub fn measured(mb_per_s: f64) -> Self {
        Self::with_drive(mb_per_s * 1e6, mb_per_s * 1e6, 100e-6)
    }

    fn with_drive(seq_bw: f64, rand_bw: f64, latency: f64) -> Self {
        Self {
            expert_bytes: 8.0e6,
            seq_bw,
            rand_bw,
            latency,
            layers: 40,
            experts_per_layer: 8,
            hit_rate: 0.0,
            compute: 2e-3,
        }
    }

    /// Why these numbers cannot describe a drive, if they cannot. The model functions never
    /// panic or return NaN on bad input, but their answer is only meaningful when this is `Ok`.
    pub fn validate(&self) -> Result<(), String> {
        let pos = |v: f64, what: &str| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(format!("{what} must be a positive number, got {v}"))
            }
        };
        pos(self.expert_bytes, "expert size")?;
        pos(self.seq_bw, "single-request bandwidth")?;
        pos(self.rand_bw, "bandwidth ceiling")?;
        if !(self.latency.is_finite() && self.latency >= 0.0) {
            return Err(format!("latency must be 0 or more, got {}", self.latency));
        }
        if !(self.compute.is_finite() && self.compute >= 0.0) {
            return Err(format!(
                "compute time must be 0 or more, got {}",
                self.compute
            ));
        }
        if !(0.0..=1.0).contains(&self.hit_rate) {
            return Err(format!(
                "hit rate must be within 0..=1, got {}",
                self.hit_rate
            ));
        }
        if self.layers == 0 {
            return Err("layers must be at least 1".into());
        }
        if self.experts_per_layer == 0 {
            return Err("experts per layer must be at least 1".into());
        }
        Ok(())
    }

    /// Time of one request on an idle drive; infinite when the drive has no usable speed.
    fn request_time(&self) -> f64 {
        if usable(self.seq_bw) {
            self.latency + self.expert_bytes / self.seq_bw
        } else {
            f64::INFINITY
        }
    }

    /// Experts that must be read per layer. The small epsilon keeps float noise such as
    /// `10 * (1 - 0.7) = 3.0000000000000004` from costing a whole extra read.
    fn reads(&self) -> u32 {
        let hit = if self.hit_rate.is_finite() {
            self.hit_rate.clamp(0.0, 1.0)
        } else {
            0.0 // unknown: assume no help
        };
        (self.experts_per_layer as f64 * (1.0 - hit) - 1e-9)
            .ceil()
            .max(0.0) as u32
    }

    /// Drive throughput in bytes per second with `q` requests in flight.
    pub fn throughput(&self, q: u32) -> f64 {
        let t = self.request_time();
        if !t.is_finite() || t <= 0.0 || !usable(self.rand_bw) {
            return 0.0;
        }
        (q.max(1) as f64 * self.expert_bytes / t).min(self.rand_bw)
    }

    /// Smallest queue depth (1..=64) that reaches 95% of the ceiling; `None` when even 64
    /// requests in flight stay below it (a drive that is slow per request and has a high
    /// ceiling), or when the drive has no usable speed.
    pub fn saturation_depth(&self) -> Option<u32> {
        if !usable(self.rand_bw) {
            return None;
        }
        (1..=64).find(|&q| self.throughput(q) >= 0.95 * self.rand_bw)
    }

    /// Read time for one layer at queue depth `q`.
    pub fn layer_io(&self, q: u32) -> f64 {
        let q = q.max(1);
        let reads = self.reads();
        if reads == 0 {
            return 0.0;
        }
        if !usable(self.rand_bw) || !self.request_time().is_finite() {
            return f64::INFINITY; // a drive with no speed never finishes; avoids 0 * inf = NaN
        }
        let full = reads / q;
        let rest = reads % q;
        let wave = |n: u32| {
            self.request_time()
                .max(n as f64 * self.expert_bytes / self.rand_bw)
        };
        let mut t = 0.0;
        if full > 0 {
            t += full as f64 * wave(q);
        }
        if rest > 0 {
            t += wave(rest);
        }
        t
    }

    /// Seconds per token, reading and computing one layer after the other.
    pub fn token_time_serial(&self, q: u32) -> f64 {
        if self.layers == 0 {
            return 0.0;
        }
        self.layers as f64 * (self.layer_io(q) + self.compute)
    }

    /// Seconds per token when the reads of layer L+1 run during the compute of layer L.
    pub fn token_time_overlapped(&self, q: u32) -> f64 {
        if self.layers == 0 {
            return 0.0;
        }
        let io = self.layer_io(q);
        let n = self.layers as f64;
        io + (n - 1.0) * io.max(self.compute) + self.compute
    }

    /// Tokens per second; 0 when the time per token is zero or unbounded (no usable estimate).
    pub fn tokens_per_s_serial(&self, q: u32) -> f64 {
        rate(self.token_time_serial(q))
    }

    pub fn tokens_per_s_overlapped(&self, q: u32) -> f64 {
        rate(self.token_time_overlapped(q))
    }
}

/// A bandwidth that can be divided by: finite and positive.
fn usable(v: f64) -> bool {
    v.is_finite() && v > 0.0
}

fn rate(t: f64) -> f64 {
    if t.is_finite() && t > 0.0 {
        1.0 / t
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_grows_with_depth_up_to_the_ceiling() {
        let p = Params::nvme();
        assert!(p.throughput(2) > p.throughput(1));
        assert!((p.throughput(64) - p.rand_bw).abs() < 1.0);
    }

    #[test]
    fn saturation_depth_is_one_when_a_single_request_fills_the_drive() {
        let mut p = Params::sata();
        p.latency = 0.0;
        assert_eq!(p.saturation_depth(), Some(1));
        let mut n = Params::nvme();
        n.seq_bw = 1.0e9; // one request reaches a third of the ceiling
        assert!(n.saturation_depth().unwrap() >= 3);
    }

    #[test]
    fn deeper_queues_never_slow_reading_down() {
        let p = Params::nvme();
        assert!(p.layer_io(8) <= p.layer_io(1));
    }

    #[test]
    fn overlap_helps_and_is_bounded_by_the_slower_part() {
        let p = Params::nvme();
        for q in [1, 4, 16] {
            assert!(p.token_time_overlapped(q) <= p.token_time_serial(q) + 1e-12);
        }
        let mut io_bound = Params::sata();
        io_bound.compute = 1e-6;
        let io = io_bound.layer_io(8);
        let t = io_bound.token_time_overlapped(8);
        assert!(t >= io * io_bound.layers as f64 - 1e-9);
    }

    #[test]
    fn no_reads_means_compute_only() {
        let mut p = Params::nvme();
        p.hit_rate = 1.0;
        assert_eq!(p.layer_io(4), 0.0);
        assert!((p.token_time_serial(4) - 40.0 * 2e-3).abs() < 1e-12);
    }

    #[test]
    fn a_slower_drive_is_slower() {
        assert!(Params::sata().tokens_per_s_serial(8) < Params::nvme().tokens_per_s_serial(8));
        assert!(
            Params::measured(200.0).tokens_per_s_serial(8) < Params::sata().tokens_per_s_serial(8)
        );
    }

    #[test]
    fn a_drive_with_no_speed_gives_zero_tokens_per_second_never_nan() {
        for (seq, rand) in [
            (0.0, 3e9),
            (3e9, 0.0),
            (-1.0, 3e9),
            (f64::NAN, 3e9),
            (3e9, f64::NAN),
        ] {
            let mut p = Params::nvme();
            (p.seq_bw, p.rand_bw) = (seq, rand);
            for q in [0, 1, 8, 64] {
                for v in [
                    p.tokens_per_s_serial(q),
                    p.tokens_per_s_overlapped(q),
                    p.throughput(q),
                ] {
                    assert!(
                        v.is_finite() && v >= 0.0,
                        "seq {seq} rand {rand} q {q}: {v}"
                    );
                }
                assert!(!p.layer_io(q).is_nan());
            }
            assert!(p.validate().is_err());
        }
        assert!(Params::measured(0.0).validate().is_err());
    }

    #[test]
    fn no_layers_is_no_estimate_in_both_modes() {
        let mut p = Params::nvme();
        p.layers = 0;
        assert_eq!(p.tokens_per_s_serial(4), 0.0);
        assert_eq!(p.tokens_per_s_overlapped(4), 0.0);
        assert!(p.validate().is_err());
    }

    #[test]
    fn float_noise_does_not_add_a_read() {
        let mut p = Params::nvme();
        p.experts_per_layer = 10;
        p.hit_rate = 0.7; // 10 * (1 - 0.7) = 3.0000000000000004 in f64: a plain ceil() says 4
        assert_eq!(p.reads(), 3);
        p.hit_rate = f64::NAN;
        assert_eq!(p.reads(), 10);
        p.hit_rate = 7.0; // out of range: clamped, not a negative read count
        assert_eq!(p.reads(), 0);
    }

    #[test]
    fn the_defaults_are_valid() {
        for p in [Params::nvme(), Params::sata(), Params::measured(1800.0)] {
            assert!(p.validate().is_ok());
        }
    }

    #[test]
    fn a_drive_that_never_reaches_its_ceiling_does_not_claim_a_saturation_depth() {
        let mut p = Params::nvme();
        p.seq_bw = 20e6; // one request moves 20 MB/s, 64 of them 1.28 GB/s: below 95% of 3 GB/s
        assert_eq!(p.saturation_depth(), None);
        let mut p = Params::nvme();
        p.rand_bw = f64::NAN;
        assert_eq!(p.saturation_depth(), None);
        // and the depth it reports, when it reports one, really reaches the 95% mark first
        for seq in [50e6, 100e6, 400e6, 3e9] {
            let mut p = Params::nvme();
            p.seq_bw = seq;
            if let Some(q) = p.saturation_depth() {
                assert!(p.throughput(q) >= 0.95 * p.rand_bw);
                assert!(q == 1 || p.throughput(q - 1) < 0.95 * p.rand_bw);
            }
        }
    }

    #[test]
    fn zero_experts_per_layer_is_refused() {
        let mut p = Params::nvme();
        p.experts_per_layer = 0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn more_hits_more_depth_a_faster_drive_or_less_compute_never_lower_the_speed() {
        // monotonicity over a grid of random valid parameter sets
        let mut s = 31u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..2000 {
            let mut p = Params::nvme();
            p.seq_bw = 50e6 + r() * 3e9;
            p.rand_bw = p.seq_bw + r() * 3e9;
            p.latency = r() * 1e-3;
            p.expert_bytes = 1e5 + r() * 5e7;
            p.experts_per_layer = 1 + (r() * 16.0) as u32;
            p.layers = 1 + (r() * 80.0) as u32;
            p.hit_rate = r() * 0.9;
            p.compute = r() * 1e-2;
            assert!(p.validate().is_ok());
            for q in [1u32, 4, 16, 64] {
                let base = p.tokens_per_s_overlapped(q);
                let mut better = p.clone();
                better.hit_rate = (p.hit_rate + 0.05).min(1.0);
                assert!(better.tokens_per_s_overlapped(q) >= base - 1e-9, "hit rate");
                let mut faster = p.clone();
                faster.rand_bw *= 1.5;
                faster.seq_bw *= 1.5;
                assert!(faster.tokens_per_s_overlapped(q) >= base - 1e-9, "drive");
                let mut lighter = p.clone();
                lighter.compute *= 0.5;
                assert!(lighter.tokens_per_s_overlapped(q) >= base - 1e-9, "compute");
                assert!(p.tokens_per_s_overlapped(q) >= p.tokens_per_s_serial(q) - 1e-9);
            }
            assert!(
                p.tokens_per_s_serial(8) >= p.tokens_per_s_serial(1) - 1e-9,
                "depth"
            );
            assert!(p.throughput(8) >= p.throughput(1) - 1e-6);
            assert!(p.throughput(64) <= p.rand_bw + 1e-6);
        }
    }
}
