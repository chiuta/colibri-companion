//! Adaptive runtime governor and "will it run?" autopilot, plus an energy / carbon meter.
//!
//! * [`Autopilot`] answers, *before* a model is loaded: will it fit, how fast will it be, which
//!   KV precision / context / quantisation should I use? It is an analytic roofline-style model
//!   (memory bandwidth vs compute vs disk) fed by a [`Calibration`] that is either a hardware
//!   heuristic or a short real measurement. Numbers are estimates, never promises.
//! * [`Governor`] watches temperature, free memory and battery while a model runs and returns
//!   [`Advice`] (threads, prefill batch, throttle, shed load) with hysteresis so it never flaps.
//!   [`SharedSettings`] lets the inference loop pick the advice up without locking.
//! * [`Meter`] turns an energy counter (Linux RAPL, with wrap-around handling) or an assumed
//!   wattage into joules per token and grams of CO2 for an assumed grid intensity.
//!
//! Pure functions and injected inputs everywhere, so the logic is testable without sensors.

use crate::arch::Spec;
use crate::hw::{self, DiskKind, HardwareProfile, KvPrecision, Tuning};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const GIB: f64 = (1u64 << 30) as f64;

// ------------------------------------------------------------------ model description

/// What the autopilot needs to know about a model.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    pub params_total: u64,
    /// Parameters touched per generated token (MoE: only the routed experts).
    pub params_active: u64,
    /// Expert parameters in total / used per token (0 for dense models).
    pub expert_params_total: u64,
    pub expert_params_active: u64,
    pub bits_per_weight: f64,
    /// KV bytes per token with an f16 cache.
    pub kv_bytes_per_token: u64,
    pub context: u64,
    pub layers: u64,
}

impl ModelInfo {
    /// Estimate parameter counts from an architecture spec. Embeddings are counted once or twice.
    pub fn from_spec(s: &Spec, bits_per_weight: f64) -> ModelInfo {
        let attn_dim = s.heads.saturating_mul(s.head_dim);
        let kv_dim = s.kv_heads.saturating_mul(s.head_dim);
        let attn = s.hidden.saturating_mul(
            attn_dim
                .saturating_mul(2)
                .saturating_add(kv_dim.saturating_mul(2)),
        );
        let mats = if s.gated_mlp { 3 } else { 2 };
        let ffn_w = |w: u64| mats * s.hidden.saturating_mul(w);
        let (mlp_total, mlp_active, exp_total, exp_active) = if s.experts > 1 {
            let e = ffn_w(s.expert_ffn());
            let shared = s.shared_experts.saturating_mul(e);
            let router = s.hidden.saturating_mul(s.experts);
            (
                s.experts
                    .saturating_mul(e)
                    .saturating_add(shared)
                    .saturating_add(router),
                s.experts_used
                    .min(s.experts)
                    .saturating_mul(e)
                    .saturating_add(shared)
                    .saturating_add(router),
                s.experts.saturating_mul(e).saturating_mul(s.layers),
                s.experts_used
                    .min(s.experts)
                    .saturating_mul(e)
                    .saturating_mul(s.layers),
            )
        } else {
            let d = ffn_w(s.ffn);
            (d, d, 0, 0)
        };
        let embed = s.vocab.saturating_mul(s.hidden);
        let embed_all = if s.tied_embeddings {
            embed
        } else {
            embed.saturating_mul(2)
        };
        let body = attn.saturating_add(mlp_total).saturating_mul(s.layers);
        let body_active = attn.saturating_add(mlp_active).saturating_mul(s.layers);
        ModelInfo {
            params_total: body.saturating_add(embed_all),
            // Only one embedding row is read per token, but the output projection is a full matvec.
            params_active: body_active.saturating_add(embed),
            expert_params_total: exp_total,
            expert_params_active: exp_active,
            bits_per_weight: bits_per_weight.clamp(1.0, 32.0),
            kv_bytes_per_token: s.kv_bytes_per_token(),
            context: s.context,
            layers: s.layers,
        }
    }

    pub fn is_moe(&self) -> bool {
        self.expert_params_total > 0
    }
    fn bytes(&self, params: u64) -> u64 {
        (params as f64 * self.bits_per_weight / 8.0) as u64
    }
    pub fn weight_bytes(&self) -> u64 {
        self.bytes(self.params_total)
    }
    pub fn active_bytes(&self) -> u64 {
        self.bytes(self.params_active)
    }
    pub fn expert_bytes(&self) -> u64 {
        self.bytes(self.expert_params_total)
    }
    /// Weights that are always needed (everything but the routed experts).
    pub fn core_bytes(&self) -> u64 {
        self.weight_bytes().saturating_sub(self.expert_bytes())
    }
    /// The same model at another bits-per-weight.
    pub fn at_bpw(&self, bpw: f64) -> ModelInfo {
        ModelInfo {
            bits_per_weight: bpw.clamp(1.0, 32.0),
            ..self.clone()
        }
    }
}

/// KV cache bytes per token for a precision (the spec figure is f16).
pub fn kv_per_token(info: &ModelInfo, p: &KvPrecision) -> u64 {
    match p {
        KvPrecision::F32 => info.kv_bytes_per_token.saturating_mul(2),
        KvPrecision::F16 => info.kv_bytes_per_token,
        // Q8_0: 34 bytes per 32 values vs 64 for f16.
        KvPrecision::Q8 => (info.kv_bytes_per_token as f64 * 34.0 / 64.0).ceil() as u64,
    }
}

/// Common quantisation levels and their bits per weight, best quality first.
pub const QUANT_LADDER: &[(&str, f64)] = &[
    ("Q8_0", 8.5),
    ("Q6_K", 6.56),
    ("Q5_K_M", 5.7),
    ("Q4_K_M", 4.85),
    ("Q3_K_M", 3.9),
    ("Q2_K", 2.63),
];

// ------------------------------------------------------------------ calibration

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalSource {
    Heuristic,
    Measured,
}

/// Machine speed figures used by the throughput model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    /// Sustained memory bandwidth, bytes/s.
    pub mem_bw: f64,
    /// Sustained f32 throughput, FLOP/s.
    pub flops: f64,
    /// Sequential read speed of the disk holding the model, bytes/s.
    pub disk_bw: f64,
    pub source: CalSource,
}

impl Calibration {
    /// A guess from the hardware profile alone. Deliberately conservative.
    pub fn heuristic(p: &HardwareProfile) -> Calibration {
        let threads = p.usable_threads.max(1) as f64;
        let mem_bw = if p.unified_memory {
            100e9
        } else {
            (threads * 5e9).min(40e9) * (p.numa_nodes.max(1) as f64).min(8.0)
        };
        let per_core = if p.cpu.isa.iter().any(|i| i.starts_with("avx512")) {
            48e9
        } else if p.cpu.isa.contains(&"avx2") {
            24e9
        } else if p.cpu.isa.contains(&"neon") {
            12e9
        } else {
            4e9
        };
        let disk_bw = match p.disk {
            DiskKind::Ssd => 1.5e9,
            DiskKind::Hdd => 0.12e9,
            DiskKind::Unknown => 0.5e9,
        };
        Calibration {
            mem_bw,
            flops: per_core * threads,
            disk_bw,
            source: CalSource::Heuristic,
        }
    }

    /// Fold in one single-thread streaming measurement (`bytes` read in `secs`). Bandwidth scales
    /// with up to four threads (a documented assumption); a nonsensical measurement is ignored.
    pub fn with_stream_measurement(self, bytes: f64, secs: f64, threads: usize) -> Calibration {
        if !(bytes.is_finite() && secs.is_finite()) || bytes <= 0.0 || secs <= 0.0 {
            return self;
        }
        let single = bytes / secs;
        let bw = single * threads.clamp(1, 4) as f64;
        if !(1e8..=2e12).contains(&bw) {
            return self;
        }
        Calibration {
            mem_bw: bw,
            source: CalSource::Measured,
            ..self
        }
    }

    /// Fold in a measured f32 compute rate for the whole machine.
    pub fn with_flops_measurement(self, flop: f64, secs: f64, threads: usize) -> Calibration {
        if !(flop.is_finite() && secs.is_finite()) || flop <= 0.0 || secs <= 0.0 {
            return self;
        }
        let f = flop / secs * threads.max(1) as f64;
        if !(1e8..=1e15).contains(&f) {
            return self;
        }
        Calibration {
            flops: f,
            source: CalSource::Measured,
            ..self
        }
    }

    /// Run a short real `matvec` benchmark (about `budget`) and fold it in.
    pub fn measure(p: &HardwareProfile, budget: Duration) -> Calibration {
        let base = Calibration::heuristic(p);
        let (rows, cols) = (2048usize, 2048usize); // 16 MiB: larger than most L2/L3 slices
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 251) as f32 - 125.0) * 1e-3)
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 * 0.1).collect();
        let mut out = vec![0f32; rows];
        crate::ops::matvec(&w, cols, &x, &mut out); // warm up
        let start = Instant::now();
        let mut n = 0u64;
        while start.elapsed() < budget.max(Duration::from_millis(5)) && n < 10_000 {
            crate::ops::matvec(&w, cols, &x, &mut out);
            n += 1;
        }
        let secs = start.elapsed().as_secs_f64();
        std::hint::black_box(&out);
        let bytes = (rows * cols * 4) as f64 * n as f64;
        let flop = 2.0 * (rows * cols) as f64 * n as f64;
        base.with_stream_measurement(bytes, secs, p.usable_threads)
            .with_flops_measurement(flop, secs, 1)
    }
}

// ------------------------------------------------------------------ autopilot

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Everything resident with headroom.
    Fits,
    /// Resident but with little headroom; expect the OS to push back under load.
    Tight,
    /// Does not fit; runs by streaming from disk at a usable speed.
    Streams,
    /// Would run, but below ~0.2 tokens/s.
    TooSlow,
    WontRun,
}

#[derive(Clone, Debug)]
pub struct Recommendation {
    pub verdict: Verdict,
    pub tuning: Tuning,
    pub kv_precision: KvPrecision,
    /// Context length the numbers below assume.
    pub context: u64,
    /// Largest context that stays resident with the chosen KV precision (0 if weights do not fit).
    pub max_context: u64,
    pub ram_needed: u64,
    pub ram_budget: u64,
    pub decode_tps: f64,
    pub prefill_tps: f64,
    /// Fraction of per-token weight traffic that has to come from disk (0 when resident).
    pub disk_fraction: f64,
    pub notes: Vec<String>,
    pub suggestions: Vec<String>,
}

pub struct Autopilot<'a> {
    pub hw: &'a HardwareProfile,
    pub cal: Calibration,
}

/// Fraction of nominal speed real code reaches.
const EFFICIENCY: f64 = 0.7;

impl<'a> Autopilot<'a> {
    pub fn new(hw: &'a HardwareProfile) -> Self {
        Autopilot {
            hw,
            cal: Calibration::heuristic(hw),
        }
    }
    pub fn with_calibration(hw: &'a HardwareProfile, cal: Calibration) -> Self {
        Autopilot { hw, cal }
    }

    /// Decode speed with `resident_core` and `resident_experts` bytes held in RAM.
    fn decode_tps(
        &self,
        info: &ModelInfo,
        kv_per_tok: u64,
        ctx: u64,
        resident_core: u64,
        resident_experts: u64,
    ) -> (f64, f64) {
        let c = &self.cal;
        let kv_read = kv_per_tok as f64 * ctx as f64;
        let active = info.active_bytes() as f64;
        // Disk traffic per token: the part of the working set that is not resident.
        let (core, exp) = (info.core_bytes() as f64, info.expert_bytes() as f64);
        let core_miss = if core > 0.0 {
            (1.0 - resident_core as f64 / core).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let exp_miss = if exp > 0.0 {
            (1.0 - resident_experts as f64 / exp).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let exp_active = (info.bytes(info.expert_params_active)) as f64;
        let core_active = (active - exp_active).max(0.0);
        let disk_bytes = core_active * core_miss + exp_active * exp_miss;
        let t_mem = (active + kv_read) / c.mem_bw.max(1.0);
        let t_cmp = 2.0 * info.params_active as f64 / c.flops.max(1.0);
        let t_disk = disk_bytes / c.disk_bw.max(1.0);
        let t = t_mem.max(t_cmp).max(t_disk) / EFFICIENCY;
        let frac = if active > 0.0 {
            (disk_bytes / active).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (if t > 0.0 { 1.0 / t } else { 0.0 }, frac)
    }

    fn prefill_tps(&self, info: &ModelInfo) -> f64 {
        let t = 2.0 * info.params_active as f64 / (self.cal.flops.max(1.0) * 0.5);
        if t > 0.0 { 1.0 / t } else { 0.0 }
    }

    /// Judge `info` for a desired context length.
    pub fn recommend(&self, info: &ModelInfo, desired_context: u64) -> Recommendation {
        let ctx = desired_context.max(1).min(info.context.max(1));
        let weights = info.weight_bytes();
        let tuning = Tuning::recommend(
            self.hw,
            weights,
            kv_per_token(info, &KvPrecision::F16).saturating_mul(ctx),
        );
        let budget = tuning.ram_budget;
        let mut notes = Vec::new();
        let mut suggestions = Vec::new();

        // Prefer the highest-precision KV cache that leaves the weights resident.
        let mut kvp = KvPrecision::Q8;
        for p in [KvPrecision::F16, KvPrecision::Q8] {
            if weights.saturating_add(kv_per_token(info, &p).saturating_mul(ctx)) as f64
                <= budget as f64 * 0.92
            {
                kvp = p;
                break;
            }
        }
        let kvt = kv_per_token(info, &kvp);
        let kv = kvt.saturating_mul(ctx);
        let need = weights.saturating_add(kv);
        let max_context = if weights < budget && kvt > 0 {
            ((budget - weights) / kvt).min(info.context)
        } else {
            0
        };

        let prefill = self.prefill_tps(info);
        let mut verdict;
        let (decode, disk_frac, ram_needed);
        if (need as f64) <= budget as f64 {
            let (d, f) = self.decode_tps(info, kvt, ctx, info.core_bytes(), info.expert_bytes());
            (decode, disk_frac, ram_needed) = (d, f, need);
            verdict = if need as f64 <= budget as f64 * 0.85 {
                Verdict::Fits
            } else {
                Verdict::Tight
            };
            if verdict == Verdict::Tight {
                notes.push(
                    "Little RAM headroom: other programs may force the model out of memory.".into(),
                );
            }
        } else {
            // Streaming: keep the KV cache and as much of the weights as fit.
            let free = budget.saturating_sub(kv);
            let core = info.core_bytes();
            let min_core = core.min((core as f64 * 0.5) as u64);
            if kv >= budget || free < min_core.max(1) {
                verdict = Verdict::WontRun;
                (decode, disk_frac, ram_needed) = (0.0, 1.0, need);
            } else {
                let res_core = free.min(core);
                let res_exp = (free - res_core).min(info.expert_bytes());
                let (d, f) = self.decode_tps(info, kvt, ctx, res_core, res_exp);
                (decode, disk_frac, ram_needed) = (d, f, res_core + res_exp + kv);
                verdict = if d >= 0.2 {
                    Verdict::Streams
                } else {
                    Verdict::TooSlow
                };
                if info.is_moe() {
                    notes.push(format!(
                        "MoE: {:.0}% of the experts stay cached; the rest stream from disk.",
                        100.0 * res_exp as f64 / info.expert_bytes().max(1) as f64
                    ));
                } else {
                    notes.push("Dense model larger than RAM: every token re-reads the missing weights from disk.".into());
                }
            }
        }
        match self.hw.disk {
            DiskKind::Hdd if disk_frac > 0.0 => notes
                .push("Model storage looks like a spinning disk; streaming will be slow.".into()),
            _ => {}
        }
        if matches!(
            verdict,
            Verdict::TooSlow | Verdict::WontRun | Verdict::Streams | Verdict::Tight
        ) {
            self.suggest(info, ctx, budget, &mut suggestions);
            if verdict == Verdict::WontRun && suggestions.is_empty() {
                suggestions.push("Free memory, add RAM, or choose a smaller model.".into());
            }
        }
        if self.cal.source == CalSource::Heuristic {
            notes.push("Speeds are estimates from the hardware profile; run a calibration for better numbers.".into());
        }
        if verdict == Verdict::WontRun {
            verdict = Verdict::WontRun;
        }
        Recommendation {
            verdict,
            tuning,
            kv_precision: kvp,
            context: ctx,
            max_context,
            ram_needed,
            ram_budget: budget,
            decode_tps: decode,
            prefill_tps: prefill,
            disk_fraction: disk_frac,
            notes,
            suggestions,
        }
    }

    fn suggest(&self, info: &ModelInfo, ctx: u64, budget: u64, out: &mut Vec<String>) {
        // Highest-quality quantisation that is fully resident.
        for (name, bpw) in QUANT_LADDER {
            if *bpw >= info.bits_per_weight {
                continue;
            }
            let m = info.at_bpw(*bpw);
            let need = m
                .weight_bytes()
                .saturating_add(kv_per_token(&m, &KvPrecision::Q8).saturating_mul(ctx));
            if need as f64 <= budget as f64 * 0.9 {
                out.push(format!(
                    "Use {name} (~{bpw} bits/weight): about {:.1} GiB and fully resident.",
                    need as f64 / GIB
                ));
                break;
            }
        }
        let kvt = kv_per_token(info, &KvPrecision::Q8);
        if info.weight_bytes() < budget && kvt > 0 {
            let max = ((budget - info.weight_bytes()) / kvt).min(info.context);
            if max > 0 && max < ctx {
                out.push(format!(
                    "Reduce the context to about {max} tokens to keep everything resident."
                ));
            }
        }
    }
}

/// Convenience: heuristic autopilot for a spec at a given bits-per-weight.
pub fn will_it_run(
    hw: &HardwareProfile,
    spec: &Spec,
    bits_per_weight: f64,
    context: u64,
) -> Recommendation {
    Autopilot::new(hw).recommend(&ModelInfo::from_spec(spec, bits_per_weight), context)
}

// ------------------------------------------------------------------ sensors

/// One snapshot of the machine. Every field is optional: a missing sensor never blocks the governor.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reading {
    pub cpu_temp_c: Option<f64>,
    pub mem_available: Option<u64>,
    pub mem_total: Option<u64>,
    pub on_battery: Option<bool>,
    pub battery_pct: Option<f64>,
}

pub trait Sensors {
    fn read(&mut self) -> Reading;
}

/// `/sys/class/thermal/*/temp` is in millidegrees; accept plain degrees too. Implausible → `None`.
pub fn parse_thermal(text: &str) -> Option<f64> {
    let v: f64 = text.trim().parse().ok()?;
    let c = if v.abs() >= 1000.0 { v / 1000.0 } else { v };
    (c.is_finite() && (-30.0..=150.0).contains(&c)).then_some(c)
}

/// Battery capacity percentage, `0..=100` only.
pub fn parse_percent(text: &str) -> Option<f64> {
    let v: f64 = text.trim().parse().ok()?;
    (v.is_finite() && (0.0..=100.0).contains(&v)).then_some(v)
}

/// `Discharging` → on battery; `Charging`/`Full`/`Not charging` → mains. Anything else → `None`.
pub fn parse_battery_status(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "discharging" => Some(true),
        "charging" | "full" | "not charging" => Some(false),
        _ => None,
    }
}

/// Reads real sensors on Linux; other platforms report nothing (the governor then does nothing).
#[derive(Default)]
pub struct SystemSensors;

impl Sensors for SystemSensors {
    fn read(&mut self) -> Reading {
        let mut r = Reading::default();
        if cfg!(target_os = "linux") {
            if let Ok(rd) = std::fs::read_dir("/sys/class/thermal") {
                r.cpu_temp_c = rd
                    .flatten()
                    .filter(|e| e.file_name().to_string_lossy().starts_with("thermal_zone"))
                    .filter_map(|e| std::fs::read_to_string(e.path().join("temp")).ok())
                    .filter_map(|t| parse_thermal(&t))
                    .fold(None, |m: Option<f64>, t| Some(m.map_or(t, |x| x.max(t))));
            }
            if let Ok(t) = std::fs::read_to_string("/proc/meminfo") {
                let (total, avail) = hw::parse_meminfo(&t);
                r.mem_total = total;
                r.mem_available = avail;
            }
            if let Ok(rd) = std::fs::read_dir("/sys/class/power_supply") {
                for e in rd.flatten() {
                    let p = e.path();
                    if std::fs::read_to_string(p.join("type"))
                        .map(|t| t.trim() == "Battery")
                        .unwrap_or(false)
                    {
                        r.on_battery = std::fs::read_to_string(p.join("status"))
                            .ok()
                            .and_then(|t| parse_battery_status(&t));
                        r.battery_pct = std::fs::read_to_string(p.join("capacity"))
                            .ok()
                            .and_then(|t| parse_percent(&t));
                        break;
                    }
                }
            }
        }
        r
    }
}

// ------------------------------------------------------------------ governor

#[derive(Clone, Debug)]
pub struct Policy {
    pub temp_hot_c: f64,
    pub temp_cool_c: f64,
    pub temp_critical_c: f64,
    /// Free-memory fraction below which the prefill batch is cut / above which it is restored.
    pub mem_low_frac: f64,
    pub mem_ok_frac: f64,
    pub mem_critical_frac: f64,
    pub battery_low_pct: f64,
    pub min_threads: usize,
    pub max_threads: usize,
    pub max_prefill_batch: usize,
    /// Minimum seconds between two changes in the same direction (prevents flapping).
    pub dwell_secs: f64,
}

impl Policy {
    pub fn from_tuning(t: &Tuning) -> Policy {
        Policy {
            temp_hot_c: 88.0,
            temp_cool_c: 78.0,
            temp_critical_c: 98.0,
            mem_low_frac: 0.06,
            mem_ok_frac: 0.12,
            mem_critical_frac: 0.02,
            battery_low_pct: 20.0,
            min_threads: 1,
            max_threads: t.threads.max(1),
            max_prefill_batch: t.prefill_batch.max(1),
            dwell_secs: 8.0,
        }
    }
}

/// What the inference loop should do now.
#[derive(Clone, Debug, PartialEq)]
pub struct Advice {
    pub threads: usize,
    pub prefill_batch: usize,
    /// Sleep this long between generated tokens (0 = full speed).
    pub throttle_ms: u64,
    /// Stop accepting new requests / sequences.
    pub shed_load: bool,
    /// Suggest dropping the KV cache to Q8 / evicting caches.
    pub reduce_cache: bool,
    pub changed: bool,
    pub reasons: Vec<String>,
}

pub struct Governor {
    pub policy: Policy,
    threads: usize,
    batch: usize,
    throttle_ms: u64,
    last_down: f64,
    last_up: f64,
    hot_since: Option<f64>,
    cool_since: Option<f64>,
    mem_state: u8, // 0 ok, 1 low, 2 critical
}

impl Governor {
    pub fn new(policy: Policy) -> Governor {
        let (threads, batch) = (policy.max_threads.max(1), policy.max_prefill_batch.max(1));
        Governor {
            policy,
            threads,
            batch,
            throttle_ms: 0,
            last_down: f64::NEG_INFINITY,
            last_up: f64::NEG_INFINITY,
            hot_since: None,
            cool_since: None,
            mem_state: 0,
        }
    }

    /// Feed one reading taken at `now` (seconds, any monotone origin). Unknown sensors are skipped.
    pub fn step(&mut self, r: &Reading, now: f64) -> Advice {
        let p = self.policy.clone();
        let before = (self.threads, self.batch, self.throttle_ms, self.mem_state);
        let mut reasons = Vec::new();
        let max_t = p.max_threads.max(p.min_threads).max(1);
        let min_t = p.min_threads.max(1).min(max_t);

        // Thermal: hot for a full dwell → shed 25% of the threads; critical → immediately throttle.
        let mut hot = false;
        if let Some(t) = r.cpu_temp_c.filter(|t| t.is_finite()) {
            if t >= p.temp_critical_c {
                hot = true;
                self.throttle_ms = 250;
                self.threads = min_t;
                self.last_down = now;
                reasons.push(format!(
                    "CPU {t:.0} °C is critical: minimum threads and a pause between tokens."
                ));
            } else if t >= p.temp_hot_c {
                hot = true;
                self.cool_since = None;
                let since = *self.hot_since.get_or_insert(now);
                if now - since >= p.dwell_secs.min(2.0) && now - self.last_down >= p.dwell_secs {
                    let cut = (self.threads / 4).max(1);
                    self.threads = self.threads.saturating_sub(cut).max(min_t);
                    self.last_down = now;
                    reasons.push(format!(
                        "CPU {t:.0} °C is hot: reducing threads to {}.",
                        self.threads
                    ));
                }
            } else if t <= p.temp_cool_c {
                self.hot_since = None;
                let since = *self.cool_since.get_or_insert(now);
                if now - since >= p.dwell_secs && now - self.last_up >= p.dwell_secs {
                    if self.throttle_ms > 0 {
                        self.throttle_ms = 0;
                        self.last_up = now;
                        reasons.push("CPU cooled down: token pause removed.".into());
                    } else if self.threads < max_t {
                        self.threads += 1;
                        self.last_up = now;
                        reasons.push(format!("CPU is cool: threads back up to {}.", self.threads));
                    }
                }
            } else {
                // Dead band between cool and hot: hold.
                self.hot_since = None;
                self.cool_since = None;
            }
        } else {
            self.hot_since = None;
        }

        // Battery.
        let mut cap = max_t;
        match (r.on_battery, r.battery_pct.filter(|b| b.is_finite())) {
            (Some(true), Some(b)) if b <= p.battery_low_pct => {
                cap = (max_t / 2).max(min_t);
                reasons.push(format!("Battery {b:.0}%: limited to {cap} threads."));
            }
            _ => {}
        }
        if self.threads > cap {
            self.threads = cap;
        }

        // Memory.
        let mut shed = false;
        let mut reduce_cache = false;
        if let (Some(avail), Some(total)) = (r.mem_available, r.mem_total.filter(|t| *t > 0)) {
            {
                let f = avail as f64 / total as f64;
                self.mem_state = match self.mem_state {
                    _ if f < p.mem_critical_frac => 2,
                    2 if f < p.mem_low_frac => 2,
                    _ if f < p.mem_low_frac => 1,
                    1 | 2 if f < p.mem_ok_frac => 1,
                    _ => 0,
                };
                match self.mem_state {
                    2 => {
                        self.batch = 1;
                        shed = true;
                        reduce_cache = true;
                        reasons.push(format!(
                            "Only {:.1}% of memory free: new work paused, batch 1.",
                            f * 100.0
                        ));
                    }
                    1 => {
                        if self.batch > 1 && now - self.last_down >= p.dwell_secs.min(1.0) {
                            self.batch = (self.batch / 2).max(1);
                            self.last_down = now;
                        }
                        reduce_cache = true;
                        reasons.push(format!(
                            "Memory is low ({:.1}% free): smaller prefill batch ({}).",
                            f * 100.0,
                            self.batch
                        ));
                    }
                    _ => {
                        if self.batch < p.max_prefill_batch && now - self.last_up >= p.dwell_secs {
                            self.batch = (self.batch * 2).min(p.max_prefill_batch);
                            self.last_up = now;
                            reasons.push(format!(
                                "Memory recovered: prefill batch back to {}.",
                                self.batch
                            ));
                        }
                    }
                }
            }
        }
        let _ = hot;
        let after = (self.threads, self.batch, self.throttle_ms, self.mem_state);
        Advice {
            threads: self.threads,
            prefill_batch: self.batch,
            throttle_ms: self.throttle_ms,
            shed_load: shed,
            reduce_cache,
            changed: before != after,
            reasons,
        }
    }
}

/// Lock-free settings the inference loop reads between tokens.
#[derive(Clone)]
pub struct SharedSettings(Arc<SettingsInner>);
struct SettingsInner {
    threads: AtomicUsize,
    batch: AtomicUsize,
    throttle_ms: AtomicU64,
    shed: AtomicBool,
}

impl SharedSettings {
    pub fn new(threads: usize, batch: usize) -> Self {
        SharedSettings(Arc::new(SettingsInner {
            threads: AtomicUsize::new(threads.max(1)),
            batch: AtomicUsize::new(batch.max(1)),
            throttle_ms: AtomicU64::new(0),
            shed: AtomicBool::new(false),
        }))
    }
    pub fn apply(&self, a: &Advice) {
        self.0.threads.store(a.threads.max(1), Ordering::Relaxed);
        self.0
            .batch
            .store(a.prefill_batch.max(1), Ordering::Relaxed);
        self.0.throttle_ms.store(a.throttle_ms, Ordering::Relaxed);
        self.0.shed.store(a.shed_load, Ordering::Relaxed);
    }
    pub fn threads(&self) -> usize {
        self.0.threads.load(Ordering::Relaxed)
    }
    pub fn prefill_batch(&self) -> usize {
        self.0.batch.load(Ordering::Relaxed)
    }
    pub fn throttle(&self) -> Duration {
        Duration::from_millis(self.0.throttle_ms.load(Ordering::Relaxed))
    }
    pub fn shedding(&self) -> bool {
        self.0.shed.load(Ordering::Relaxed)
    }
}

// ------------------------------------------------------------------ energy & carbon

/// Energy counter delta with wrap-around (`max` is the counter's range, e.g. RAPL
/// `max_energy_range_uj`). A counter that goes backwards by more than half its range is a reset.
pub fn counter_delta(prev: u64, cur: u64, max: u64) -> u64 {
    if cur >= prev {
        cur - prev
    } else if max > 0 && prev <= max && cur < max {
        (max - prev).saturating_add(cur)
    } else {
        0
    }
}

/// Average grid intensity used when the caller does not know better (g CO2 per kWh). An assumption.
pub const DEFAULT_GRID_G_PER_KWH: f64 = 400.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnergyReport {
    pub joules: f64,
    pub seconds: f64,
    pub tokens: u64,
    pub joules_per_token: f64,
    pub avg_watts: f64,
    pub grams_co2: f64,
    /// `true` when a hardware counter was read; `false` when derived from an assumed wattage.
    pub measured: bool,
}

#[derive(Clone, Debug)]
enum Source {
    Counter {
        range: u64,
        last: Option<(u64, f64)>,
        joules: f64,
        first_t: Option<f64>,
    },
    Assumed {
        watts: f64,
        t0: f64,
        t1: f64,
    },
}

pub struct Meter {
    src: Source,
    tokens: u64,
    pub grid_g_per_kwh: f64,
}

impl Meter {
    /// A meter over a microjoule counter (RAPL style) with the given wrap range.
    pub fn counter(range_uj: u64) -> Meter {
        Meter {
            src: Source::Counter {
                range: range_uj,
                last: None,
                joules: 0.0,
                first_t: None,
            },
            tokens: 0,
            grid_g_per_kwh: DEFAULT_GRID_G_PER_KWH,
        }
    }
    /// No counter: energy = assumed watts × time. Reported as `measured: false`.
    pub fn assumed(watts: f64) -> Meter {
        Meter {
            src: Source::Assumed {
                watts: if watts.is_finite() {
                    watts.max(0.0)
                } else {
                    0.0
                },
                t0: 0.0,
                t1: 0.0,
            },
            tokens: 0,
            grid_g_per_kwh: DEFAULT_GRID_G_PER_KWH,
        }
    }

    /// Read the Linux RAPL package counter if present and readable.
    pub fn discover_rapl() -> Option<Meter> {
        let dir = "/sys/class/powercap/intel-rapl:0";
        let range: u64 = std::fs::read_to_string(format!("{dir}/max_energy_range_uj"))
            .ok()?
            .trim()
            .parse()
            .ok()?;
        read_rapl_uj()?;
        Some(Meter::counter(range))
    }

    /// Record the counter value (µJ) at time `t` seconds. For an assumed meter only time matters.
    pub fn sample(&mut self, counter_uj: u64, t: f64) {
        match &mut self.src {
            Source::Counter {
                range,
                last,
                joules,
                first_t,
            } => {
                if first_t.is_none() {
                    *first_t = Some(t);
                }
                if let Some((prev, _)) = *last {
                    *joules += counter_delta(prev, counter_uj, *range) as f64 / 1e6;
                }
                *last = Some((counter_uj, t));
            }
            Source::Assumed { t0, t1, .. } => {
                if *t1 == 0.0 && *t0 == 0.0 {
                    *t0 = t;
                }
                *t1 = t;
            }
        }
    }

    pub fn add_tokens(&mut self, n: u64) {
        self.tokens = self.tokens.saturating_add(n);
    }

    pub fn report(&self) -> EnergyReport {
        let (joules, seconds, measured) = match &self.src {
            Source::Counter {
                last,
                joules,
                first_t,
                ..
            } => (
                *joules,
                last.zip(*first_t)
                    .map_or(0.0, |((_, t), f)| (t - f).max(0.0)),
                true,
            ),
            Source::Assumed { watts, t0, t1 } => {
                let s = (t1 - t0).max(0.0);
                (watts * s, s, false)
            }
        };
        let kwh = joules / 3.6e6;
        EnergyReport {
            joules,
            seconds,
            tokens: self.tokens,
            joules_per_token: if self.tokens > 0 {
                joules / self.tokens as f64
            } else {
                0.0
            },
            avg_watts: if seconds > 0.0 { joules / seconds } else { 0.0 },
            grams_co2: kwh * self.grid_g_per_kwh.max(0.0),
            measured,
        }
    }
}

/// Current RAPL package energy in µJ, if readable (often root-only on recent kernels).
pub fn read_rapl_uj() -> Option<u64> {
    std::fs::read_to_string("/sys/class/powercap/intel-rapl:0/energy_uj")
        .ok()?
        .trim()
        .parse()
        .ok()
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw::{Caches, Cpu, Limits};

    const G: u64 = 1 << 30;

    fn hw(
        threads: usize,
        ram_gib: u64,
        avail_gib: u64,
        disk: DiskKind,
        isa: Vec<&'static str>,
    ) -> HardwareProfile {
        HardwareProfile {
            os: "linux",
            arch: "x86_64",
            cpu: Cpu {
                logical: threads,
                physical: Some(threads),
                isa,
                ..Default::default()
            },
            caches: Caches::default(),
            total_ram: Some(ram_gib * G),
            available_ram: Some(avail_gib * G),
            limits: Limits::default(),
            usable_threads: threads,
            numa_nodes: 1,
            thp: None,
            gpus: vec![],
            unified_memory: false,
            disk,
        }
    }
    fn laptop() -> HardwareProfile {
        hw(8, 4, 3, DiskKind::Ssd, vec!["avx2"])
    }
    fn desktop() -> HardwareProfile {
        hw(16, 32, 28, DiskKind::Ssd, vec!["avx2"])
    }
    fn workstation() -> HardwareProfile {
        let mut h = hw(64, 512, 480, DiskKind::Ssd, vec!["avx512f"]);
        h.numa_nodes = 4;
        h
    }

    fn dense(layers: u64, hidden: u64, ffn: u64, vocab: u64, ctx: u64) -> Spec {
        let mut s = Spec::dense(
            "llama",
            layers,
            hidden,
            hidden / 128,
            (hidden / 128 / 4).max(1),
            128,
            ffn,
            vocab,
            ctx,
        );
        s.gated_mlp = true;
        s.tied_embeddings = false;
        s
    }
    fn small() -> Spec {
        dense(24, 2048, 5632, 32000, 8192)
    } // ~1.5 B
    fn huge() -> Spec {
        dense(80, 8192, 28672, 128000, 32768)
    } // ~70 B
    fn big_moe() -> Spec {
        let mut s = dense(60, 7168, 18432, 129000, 32768);
        s.experts = 256;
        s.experts_used = 8;
        s.shared_experts = 1;
        s.moe_ffn = 2048;
        s
    }

    #[test]
    fn param_formulas() {
        let s = small();
        let m = ModelInfo::from_spec(&s, 16.0);
        // attn = hidden * (2*attn_dim + 2*kv_dim); mlp = 3*hidden*ffn
        let attn = 2048u64 * (2 * 2048 + 2 * 4 * 128);
        let mlp = 3 * 2048u64 * 5632;
        let embed = 32000u64 * 2048;
        assert_eq!(m.params_total, (attn + mlp) * 24 + 2 * embed);
        assert_eq!(m.params_active, (attn + mlp) * 24 + embed);
        assert!(!m.is_moe());
        assert_eq!(m.weight_bytes(), m.params_total * 2);
        let moe = ModelInfo::from_spec(&big_moe(), 4.0);
        assert!(moe.is_moe());
        assert!(
            moe.params_total > 10 * moe.params_active,
            "{} vs {}",
            moe.params_total,
            moe.params_active
        );
        assert_eq!(moe.expert_params_total, 256 * 3 * 7168 * 2048 * 60);
        assert_eq!(moe.expert_params_active * 256, moe.expert_params_total * 8);
        assert_eq!(moe.core_bytes() + moe.expert_bytes(), moe.weight_bytes());
    }

    #[test]
    fn kv_precision_sizes() {
        let m = ModelInfo::from_spec(&small(), 4.0);
        assert_eq!(
            kv_per_token(&m, &KvPrecision::F32),
            2 * m.kv_bytes_per_token
        );
        let q = kv_per_token(&m, &KvPrecision::Q8);
        assert!(q < m.kv_bytes_per_token && q * 2 > m.kv_bytes_per_token);
    }

    #[test]
    fn verdict_matrix() {
        let (l, d, w) = (laptop(), desktop(), workstation());
        let small_q4 = ModelInfo::from_spec(&small(), 4.85);
        let huge_q4 = ModelInfo::from_spec(&huge(), 4.85);
        let moe_q4 = ModelInfo::from_spec(&big_moe(), 4.85);
        // Small model: runs everywhere that has a GB.
        assert!(matches!(
            Autopilot::new(&l).recommend(&small_q4, 2048).verdict,
            Verdict::Fits | Verdict::Tight
        ));
        assert_eq!(
            Autopilot::new(&d).recommend(&small_q4, 4096).verdict,
            Verdict::Fits
        );
        assert_eq!(
            Autopilot::new(&w).recommend(&small_q4, 4096).verdict,
            Verdict::Fits
        );
        // 70B at ~4.85 bpw (~40 GB): laptop cannot, desktop streams or is too slow, workstation fits.
        let r = Autopilot::new(&l).recommend(&huge_q4, 2048);
        assert!(
            matches!(r.verdict, Verdict::WontRun | Verdict::TooSlow),
            "{:?}",
            r.verdict
        );
        let r = Autopilot::new(&d).recommend(&huge_q4, 2048);
        assert!(
            matches!(r.verdict, Verdict::Streams | Verdict::TooSlow),
            "{:?}",
            r.verdict
        );
        assert!(r.disk_fraction > 0.0);
        let r = Autopilot::new(&w).recommend(&huge_q4, 8192);
        assert_eq!(r.verdict, Verdict::Fits);
        assert_eq!(r.disk_fraction, 0.0);
        // ~400 B MoE (~230 GB at 4.85 bpw): laptop no, desktop streams experts, workstation fits.
        let r = Autopilot::new(&l).recommend(&moe_q4, 2048);
        assert_eq!(r.verdict, Verdict::WontRun);
        assert!(!r.suggestions.is_empty());
        let r = Autopilot::new(&d).recommend(&moe_q4, 2048);
        assert!(
            matches!(r.verdict, Verdict::Streams | Verdict::TooSlow),
            "{:?}",
            r.verdict
        );
        assert!(r.notes.iter().any(|n| n.contains("MoE")));
        // The sample MoE has ~676 B parameters: 410 GB at 4.85 bpw streams a little even on 512 GB,
        // but at ~2.6 bpw (~220 GB) it is fully resident.
        let r = Autopilot::new(&w).recommend(&moe_q4.at_bpw(2.63), 4096);
        assert_eq!(r.verdict, Verdict::Fits, "{:?}", r.verdict);
    }

    #[test]
    fn moe_decodes_faster_than_dense_of_same_size_when_streaming() {
        // Same machine, similar total bytes: the MoE only reads its active experts.
        let d = desktop();
        let ap = Autopilot::new(&d);
        let moe = ModelInfo::from_spec(&big_moe(), 2.63);
        let dense_info = ModelInfo::from_spec(&huge(), 8.5);
        let (a, b) = (ap.recommend(&moe, 2048), ap.recommend(&dense_info, 2048));
        assert!(
            a.decode_tps > b.decode_tps,
            "{} vs {}",
            a.decode_tps,
            b.decode_tps
        );
    }

    #[test]
    fn speed_formula_matches_hand_calculation() {
        let d = desktop();
        let cal = Calibration {
            mem_bw: 20e9,
            flops: 1e12,
            disk_bw: 1e9,
            source: CalSource::Measured,
        };
        let ap = Autopilot::with_calibration(&d, cal);
        let info = ModelInfo::from_spec(&small(), 8.0);
        let r = ap.recommend(&info, 1);
        // Bandwidth bound: (active bytes + 1 token of KV) / bw, derated by EFFICIENCY.
        let kvt = kv_per_token(&info, &r.kv_precision) as f64;
        let t = (info.active_bytes() as f64 + kvt) / 20e9 / 0.7;
        assert!(
            (r.decode_tps - 1.0 / t).abs() / (1.0 / t) < 1e-9,
            "{} vs {}",
            r.decode_tps,
            1.0 / t
        );
        // Doubling bandwidth doubles speed for a memory-bound model.
        let r2 = Autopilot::with_calibration(
            &d,
            Calibration {
                mem_bw: 40e9,
                ..cal
            },
        )
        .recommend(&info, 1);
        assert!((r2.decode_tps / r.decode_tps - 2.0).abs() < 0.01);
    }

    #[test]
    fn heavy_quant_suggestion_makes_it_fit() {
        let d = desktop();
        let info = ModelInfo::from_spec(&dense(48, 6144, 16384, 128000, 32768), 16.0); // ~20 B params, ~41 GB: too big for 28 GB free
        let r = Autopilot::new(&d).recommend(&info, 2048);
        assert!(!matches!(r.verdict, Verdict::Fits));
        assert!(
            r.suggestions
                .iter()
                .any(|s| s.starts_with("Use Q") && s.contains("fully resident")),
            "{:?}",
            r.suggestions
        );
    }

    #[test]
    fn context_clamped_and_max_context_reported() {
        let w = workstation();
        let info = ModelInfo::from_spec(&small(), 4.85);
        let r = Autopilot::new(&w).recommend(&info, 10_000_000);
        assert_eq!(r.context, 8192);
        assert_eq!(r.max_context, 8192);
        let l = laptop();
        let r = Autopilot::new(&l).recommend(&ModelInfo::from_spec(&small(), 4.85), 8192);
        assert!(r.max_context <= 8192);
    }

    #[test]
    fn calibration_with_injected_measurements() {
        let h = desktop();
        let base = Calibration::heuristic(&h);
        assert_eq!(base.source, CalSource::Heuristic);
        // 1 GiB in 0.1 s on one thread = ~10.7 GB/s, scaled by min(threads,4)=4.
        let c = base.with_stream_measurement(G as f64, 0.1, 16);
        assert_eq!(c.source, CalSource::Measured);
        assert!((c.mem_bw - G as f64 / 0.1 * 4.0).abs() < 1.0);
        // Garbage measurements are ignored.
        for (b, s) in [
            (0.0, 1.0),
            (1e9, 0.0),
            (f64::NAN, 1.0),
            (1e9, f64::INFINITY),
            (1.0, 1.0),
            (-5.0, 1.0),
        ] {
            assert_eq!(base.with_stream_measurement(b, s, 4), base, "{b} {s}");
        }
        assert_eq!(base.with_flops_measurement(f64::NAN, 1.0, 1), base);
        let f = base.with_flops_measurement(2e9, 1.0, 8);
        assert!((f.flops - 16e9).abs() < 1.0);
    }

    #[test]
    fn real_measurement_is_sane() {
        let h = desktop();
        let c = Calibration::measure(&h, Duration::from_millis(30));
        assert!(c.mem_bw > 1e8 && c.flops > 1e8, "{c:?}");
        assert_eq!(c.source, CalSource::Measured);
    }

    #[test]
    fn heuristics_scale_with_hardware() {
        let a = Calibration::heuristic(&laptop());
        let b = Calibration::heuristic(&workstation());
        assert!(b.mem_bw > a.mem_bw && b.flops > a.flops);
        assert!(Calibration::heuristic(&hw(8, 16, 12, DiskKind::Hdd, vec![])).disk_bw < a.disk_bw);
    }

    // ----- governor -----

    fn pol() -> Policy {
        Policy {
            max_threads: 8,
            max_prefill_batch: 32,
            dwell_secs: 8.0,
            ..Policy::from_tuning(&Tuning::recommend(&desktop(), 4 * G, 0))
        }
    }
    fn temp(t: f64) -> Reading {
        Reading {
            cpu_temp_c: Some(t),
            ..Default::default()
        }
    }
    fn mem(avail: u64) -> Reading {
        Reading {
            mem_available: Some(avail),
            mem_total: Some(1000),
            ..Default::default()
        }
    }

    #[test]
    fn hot_cpu_sheds_threads_gradually_then_recovers() {
        let mut g = Governor::new(Policy {
            max_threads: 8,
            ..pol()
        });
        assert_eq!(g.step(&temp(60.0), 0.0).threads, 8);
        let mut now = 1.0;
        let mut last = 8;
        let mut drops = 0;
        for _ in 0..60 {
            let a = g.step(&temp(92.0), now);
            assert!(a.threads <= last);
            if a.threads < last {
                drops += 1;
            }
            last = a.threads;
            now += 1.0;
        }
        assert!(
            drops >= 2 && (1..8).contains(&last),
            "last {last} drops {drops}"
        );
        // Dead band holds.
        let hold = g.step(&temp(82.0), now).threads;
        assert_eq!(hold, last);
        // Cool for a long time: climbs back one at a time, never above max.
        let mut prev = last;
        for _ in 0..200 {
            now += 1.0;
            let a = g.step(&temp(55.0), now);
            assert!(a.threads >= prev && a.threads <= prev + 1);
            prev = a.threads;
        }
        assert_eq!(prev, 8);
    }

    #[test]
    fn no_flapping_around_threshold() {
        let mut g = Governor::new(Policy {
            max_threads: 8,
            ..pol()
        });
        let mut changes = 0;
        let mut last = 8;
        // Oscillate across the hot threshold every second for 5 minutes.
        for i in 0..300 {
            let t = if i % 2 == 0 { 89.0 } else { 86.0 };
            let a = g.step(&temp(t), i as f64);
            if a.threads != last {
                changes += 1;
                last = a.threads;
            }
        }
        // At most one drop per dwell (8 s) and never any increase (never cooled below 78).
        assert!(changes <= 300 / 8 + 1, "{changes}");
        assert!(last <= 8);
        let mut up = 0;
        let mut g = Governor::new(Policy {
            max_threads: 8,
            ..pol()
        });
        let mut last = 8;
        for i in 0..400 {
            let t = if i % 2 == 0 { 79.0 } else { 77.0 }; // straddles the cool line, never hot
            let a = g.step(&temp(t), i as f64);
            if a.threads > last {
                up += 1;
            }
            last = a.threads;
        }
        assert_eq!(up, 0);
        assert_eq!(last, 8);
    }

    #[test]
    fn critical_temperature_throttles_immediately() {
        let mut g = Governor::new(pol());
        let a = g.step(&temp(105.0), 0.0);
        assert_eq!(a.threads, 1);
        assert!(a.throttle_ms > 0 && a.changed);
        // Recovery removes the pause first, only after the dwell.
        assert!(g.step(&temp(50.0), 1.0).throttle_ms > 0);
        let a = g.step(&temp(50.0), 20.0);
        assert_eq!(a.throttle_ms, 0);
    }

    #[test]
    fn memory_pressure_cuts_batch_and_sheds_load_with_hysteresis() {
        let mut g = Governor::new(pol());
        assert_eq!(g.step(&mem(500), 0.0).prefill_batch, 32);
        let a = g.step(&mem(50), 1.0); // 5% < low 6%
        assert!(a.prefill_batch < 32 && a.reduce_cache && !a.shed_load);
        let a = g.step(&mem(10), 2.0); // 1% critical
        assert!(a.shed_load && a.prefill_batch == 1);
        // 8% is above the critical exit line but below the 12% "ok" line: low state, no restoring.
        let a = g.step(&mem(80), 3.0);
        assert!(!a.shed_load && a.prefill_batch == 1 && a.reduce_cache);
        // 9%: still not "ok".
        assert_eq!(g.step(&mem(90), 40.0).prefill_batch, 1);
        // 30%: recovers stepwise, dwell-limited.
        let mut now = 60.0;
        let mut b = 1;
        for _ in 0..10 {
            b = g.step(&mem(300), now).prefill_batch;
            now += 9.0;
        }
        assert_eq!(b, 32);
    }

    #[test]
    fn low_battery_caps_threads_only_on_battery() {
        let mut g = Governor::new(Policy {
            max_threads: 8,
            ..pol()
        });
        let r = Reading {
            on_battery: Some(true),
            battery_pct: Some(10.0),
            ..Default::default()
        };
        assert_eq!(g.step(&r, 0.0).threads, 4);
        let r = Reading {
            on_battery: Some(false),
            battery_pct: Some(10.0),
            ..Default::default()
        };
        // Plugged in: the cap is lifted gradually only through the cool path; no sensor → holds.
        assert!(g.step(&r, 1.0).threads <= 8);
        let mut g = Governor::new(Policy {
            max_threads: 8,
            ..pol()
        });
        let r = Reading {
            on_battery: Some(true),
            battery_pct: Some(80.0),
            ..Default::default()
        };
        assert_eq!(g.step(&r, 0.0).threads, 8);
    }

    #[test]
    fn missing_or_hostile_readings_do_nothing() {
        let mut g = Governor::new(pol());
        let a = g.step(&Reading::default(), 0.0);
        assert!(!a.changed && a.threads == pol().max_threads);
        for t in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let a = g.step(
                &Reading {
                    cpu_temp_c: Some(t),
                    battery_pct: Some(t),
                    mem_available: Some(0),
                    mem_total: Some(0),
                    on_battery: Some(true),
                },
                1.0,
            );
            assert!(a.threads >= 1 && a.prefill_batch >= 1);
        }
        // mem_total = 0 must not divide by zero; avail > total is fine.
        let _ = g.step(
            &Reading {
                mem_available: Some(u64::MAX),
                mem_total: Some(1),
                ..Default::default()
            },
            2.0,
        );
    }

    #[test]
    fn sensor_parsers_reject_garbage() {
        assert_eq!(parse_thermal("45000\n"), Some(45.0));
        assert_eq!(parse_thermal("45.5"), Some(45.5));
        for bad in [
            "",
            "abc",
            "NaN",
            "inf",
            "-999999",
            "9999999999",
            "4 5",
            "\u{0}",
            "0x10",
            "1e400",
        ] {
            assert_eq!(parse_thermal(bad), None, "{bad:?}");
        }
        assert_eq!(parse_percent("55"), Some(55.0));
        for bad in ["-1", "101", "x", "", "nan"] {
            assert_eq!(parse_percent(bad), None, "{bad:?}");
        }
        assert_eq!(parse_battery_status("Discharging\n"), Some(true));
        assert_eq!(parse_battery_status("Full"), Some(false));
        assert_eq!(parse_battery_status("Unknown"), None);
        assert_eq!(parse_battery_status(&"x".repeat(100_000)), None);
    }

    #[test]
    fn system_sensors_never_panic() {
        let r = SystemSensors.read();
        if let Some(t) = r.cpu_temp_c {
            assert!((-30.0..=150.0).contains(&t));
        }
    }

    #[test]
    fn shared_settings_roundtrip() {
        let s = SharedSettings::new(0, 0);
        assert_eq!((s.threads(), s.prefill_batch()), (1, 1));
        let mut g = Governor::new(Policy {
            max_threads: 6,
            ..pol()
        });
        let a = g.step(&temp(120.0), 0.0);
        let s2 = s.clone();
        s.apply(&a);
        assert_eq!(s2.threads(), 1);
        assert!(s2.throttle() > Duration::ZERO);
        assert!(!s2.shedding());
    }

    // ----- energy -----

    #[test]
    fn counter_wraparound() {
        assert_eq!(counter_delta(100, 250, 1000), 150);
        assert_eq!(counter_delta(900, 50, 1000), 150); // wrapped
        assert_eq!(counter_delta(5, 5, 1000), 0);
        assert_eq!(counter_delta(900, 2000, 1000), 1100); // cur >= prev: plain delta
        assert_eq!(counter_delta(2000, 50, 1000), 0); // prev out of range: reset, ignore
        assert_eq!(counter_delta(10, 5, 0), 0);
    }

    #[test]
    fn meter_with_wrap_and_carbon() {
        let mut m = Meter::counter(1_000_000_000); // 1000 J range
        m.grid_g_per_kwh = 500.0;
        m.sample(900_000_000, 0.0);
        m.sample(100_000_000, 10.0); // wrapped: +200 J
        m.sample(600_000_000, 20.0); // +500 J
        m.add_tokens(70);
        let r = m.report();
        assert!(r.measured);
        assert!((r.joules - 700.0).abs() < 1e-6, "{}", r.joules);
        assert_eq!(r.seconds, 20.0);
        assert!((r.joules_per_token - 10.0).abs() < 1e-9);
        assert!((r.avg_watts - 35.0).abs() < 1e-9);
        // 700 J = 0.000194.. kWh × 500 g/kWh
        assert!((r.grams_co2 - 700.0 / 3.6e6 * 500.0).abs() < 1e-12);
    }

    #[test]
    fn assumed_meter_is_flagged_and_zero_safe() {
        let mut m = Meter::assumed(20.0);
        let r = m.report();
        assert!(!r.measured && r.joules == 0.0 && r.joules_per_token == 0.0 && r.avg_watts == 0.0);
        m.sample(0, 100.0);
        m.sample(0, 110.0);
        m.add_tokens(5);
        let r = m.report();
        assert!((r.joules - 200.0).abs() < 1e-9 && (r.joules_per_token - 40.0).abs() < 1e-9);
        assert!(Meter::assumed(f64::NAN).report().joules == 0.0);
        // Time going backwards never yields negative energy.
        let mut m = Meter::assumed(10.0);
        m.sample(0, 50.0);
        m.sample(0, 40.0);
        assert!(m.report().joules >= 0.0);
    }

    #[test]
    fn rapl_discovery_is_optional() {
        if let Some(mut m) = Meter::discover_rapl() {
            m.sample(read_rapl_uj().unwrap_or(0), 0.0);
            assert!(m.report().joules >= 0.0);
        }
    }
}
