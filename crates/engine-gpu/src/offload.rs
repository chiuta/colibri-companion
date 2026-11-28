//! Memory-aware layer-offload planning for a device.
//!
//! `Model::with_backend_opts` already stops uploading layers when `Backend::device_memory()` runs
//! out. This module answers the question *before* anything is uploaded: how many layers fit, how
//! long a context the device cache can hold, what limits the plan, and roughly what speed to
//! expect, so `doctor` and `plan` can say it in plain words.
//!
//! It mirrors the model's own rules (the same [`backend::offload_plan`] arithmetic, the same
//! per-layer cache size, the same eligibility conditions at the `Spec` level) but is only a
//! prediction: `Model::offloaded()` is authoritative once a model is loaded.
use crate::caps::DeviceClass;
use crate::layout::Format;
use companion_engine::arch::{Act, NormKind, RopeScaling, Spec};
use companion_engine::backend::offload_plan;
use companion_engine::quant::DType;

/// Bytes a `rows × cols` matrix takes on the device (the GPU layouts of [`crate::layout`], not the
/// file layout; `F32` is what dense weights are uploaded as). `None` when the format has no kernel
/// for that row length, in which case the tensor stays on the host.
pub fn device_weight_bytes(dtype: DType, rows: u64, cols: u64) -> Option<u64> {
    let f = Format::from_dtype(dtype)?;
    if !f.accepts_cols(usize::try_from(cols).ok()?) {
        return None;
    }
    rows.checked_mul(f.row_words(cols as usize) as u64)?
        .checked_mul(4)
}

/// Device bytes of the seven matrices of one layer, or `None` if any of them cannot go.
pub fn layer_device_bytes(spec: &Spec, dtype: DType) -> Option<u64> {
    let (h, qd, kd, ff) = (
        spec.hidden,
        spec.heads.saturating_mul(spec.head_dim),
        spec.kv_heads.saturating_mul(spec.head_dim),
        spec.ffn,
    );
    let mats = [
        (qd, h),
        (kd, h),
        (kd, h),
        (h, qd),
        (ff, h),
        (ff, h),
        (h, ff),
    ];
    mats.iter().try_fold(0u64, |acc, &(r, c)| {
        acc.checked_add(device_weight_bytes(dtype, r, c)?)
    })
}

/// Which layers the model would hand to a backend, judged from the architecture alone. The rules
/// are those of `Model::layer_offloadable`; per-tensor details (biases that exist only in the file)
/// are not visible here.
pub fn eligible_layers(s: &Spec) -> Vec<bool> {
    let base = s.norm == NormKind::Rms
        && !s.norm_offset
        && s.act == Act::Silu
        && s.gated_mlp
        && !s.qk_norm
        && !s.qkv_bias
        && !s.post_norms
        && !s.parallel_residual
        && s.residual_scale == 1.0
        && s.attn_softcap == 0.0
        && s.rotary_dim == s.head_dim
        && s.experts <= 1
        && s.heads > 0
        && s.kv_heads > 0
        && s.heads.is_multiple_of(s.kv_heads);
    (0..s.layers as usize)
        .map(|i| {
            let rope_ok = (s.is_sliding(i) && s.rope_theta_local > 0.0)
                || s.rope_scaling == RopeScaling::None;
            base && rope_ok
        })
        .collect()
}

/// Why the plan stops where it does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limiter {
    /// Every eligible layer fits.
    Nothing,
    /// Device memory ran out of weights (and cache) first.
    Memory,
    /// No layer can go to the device (architecture or format), whatever the memory.
    NoEligibleLayer,
    /// The device has no memory to offer.
    NoMemory,
}

#[derive(Clone, Debug)]
pub struct Request<'a> {
    pub spec: &'a Spec,
    /// Format the model's matrices are stored in (dense models: `DType::F32`).
    pub dtype: DType,
    /// Context length wanted on the device.
    pub context: u64,
    /// Bytes of device memory the engine may use.
    pub vram: u64,
    /// Working buffers and driver overhead (the model's default is 256 MiB).
    pub reserve: u64,
    /// Largest storage binding of the adapter ([`crate::caps::Limits::usable_binding`]).
    pub binding: u64,
    pub class: DeviceClass,
    /// Measured or estimated CPU-only decode speed, for the speed estimate.
    pub cpu_decode_tps: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub total_layers: usize,
    /// Layers that will run on the device (eligible ones among the leading run that fits).
    pub gpu_layers: usize,
    /// Length of the leading run of layers the plan covers (eligible or not); what
    /// `AccelOptions::max_layers` takes.
    pub leading_layers: usize,
    pub eligible_layers: usize,
    pub gpu_weight_bytes: u64,
    pub kv_capacity: u64,
    pub kv_bytes_per_layer: u64,
    /// Longest context that would still let every eligible layer sit on the device.
    pub max_context_all_layers: u64,
    pub limiter: Limiter,
    /// Estimated decode speed with this split, if a CPU speed was given. An estimate from class
    /// bandwidth heuristics, not a measurement.
    pub est_decode_tps: Option<f64>,
    pub notes: Vec<String>,
}

impl Plan {
    /// The options that make `Model::with_backend_opts` follow this plan exactly. The model plans
    /// with the host byte size of each weight, which for some quantised formats is up to 6 % less
    /// than the device layout takes (Q8_0 blocks grow from 34 to 36 bytes); passing the planned
    /// layer count and cache length keeps it from taking a layer the device cannot really hold.
    pub fn accel_options(&self, reserve: u64) -> companion_engine::model::AccelOptions {
        companion_engine::model::AccelOptions {
            max_layers: Some(self.leading_layers),
            kv_capacity: usize::try_from(self.kv_capacity).unwrap_or(usize::MAX),
            reserve,
        }
    }
}

/// Longest cache the kernels can address: both the cache buffer and the attention scratch
/// (`heads × cap` floats) must fit one binding.
pub fn kv_capacity_limit(spec: &Spec, binding: u64) -> u64 {
    let kd = spec.kv_heads.saturating_mul(spec.head_dim).max(1);
    let h = spec.heads.max(1);
    (binding / kd.saturating_mul(4).max(1)).min(binding / h.saturating_mul(4).max(1))
}

fn plan_for(
    req: &Request<'_>,
    eligible: &[bool],
    per_layer_dev: u64,
    cap: u64,
) -> companion_engine::backend::Offload {
    let kd = req.spec.kv_heads.saturating_mul(req.spec.head_dim);
    let kv_layer = cap.saturating_mul(kd).saturating_mul(2 * 4);
    let sizes: Vec<u64> = eligible
        .iter()
        .map(|&e| if e { per_layer_dev } else { 0 })
        .collect();
    offload_plan(&sizes, kv_layer, 0, req.vram, req.reserve)
}

/// Predict the split for `req`.
pub fn plan_offload(req: &Request<'_>) -> Plan {
    let s = req.spec;
    let n = s.layers as usize;
    let eligible = eligible_layers(s);
    let n_eligible = eligible.iter().filter(|&&e| e).count();
    let mut notes = Vec::new();
    let per_layer = layer_device_bytes(s, req.dtype);
    let kd = s.kv_heads.saturating_mul(s.head_dim);

    let limit = kv_capacity_limit(s, req.binding);
    let wanted = req.context.max(1).min(s.context.max(1));
    let cap = wanted.min(limit).max(1);
    if cap < wanted {
        notes.push(format!(
            "the device cache is limited to {cap} positions by the adapter's {} MiB binding limit",
            req.binding >> 20
        ));
    }
    let kv_layer = cap.saturating_mul(kd).saturating_mul(8);

    let mut plan = Plan {
        total_layers: n,
        gpu_layers: 0,
        leading_layers: 0,
        eligible_layers: n_eligible,
        gpu_weight_bytes: 0,
        kv_capacity: cap,
        kv_bytes_per_layer: kv_layer,
        max_context_all_layers: 0,
        limiter: Limiter::NoEligibleLayer,
        est_decode_tps: None,
        notes,
    };

    if req.vram == 0 {
        plan.limiter = Limiter::NoMemory;
        plan.notes
            .push("the device reports no memory to use".into());
        return plan;
    }
    let Some(per_dev) = per_layer else {
        plan.notes.push(format!("{:?} has no kernel for this model's row lengths (hidden {}, ffn {}); the layers stay on the CPU", req.dtype, s.hidden, s.ffn));
        return plan;
    };
    if n_eligible == 0 {
        plan.notes.push(explain_ineligible(s));
        return plan;
    }

    let off = plan_for(req, &eligible, per_dev, cap);
    let on_gpu = eligible.iter().take(off.gpu_layers).filter(|&&e| e).count();
    plan.gpu_layers = on_gpu;
    plan.leading_layers = off.gpu_layers;
    plan.gpu_weight_bytes = on_gpu as u64 * per_dev;
    plan.limiter = if on_gpu == n_eligible {
        Limiter::Nothing
    } else {
        Limiter::Memory
    };
    if on_gpu < n_eligible {
        plan.notes.push(format!("{on_gpu} of {n_eligible} eligible layers fit in {} MiB (each takes {} MiB of weights plus {} MiB of cache)", req.vram >> 20, per_dev >> 20, kv_layer >> 20));
    }
    if n_eligible < n {
        plan.notes.push(format!(
            "{} of {n} layers use features the GPU kernels do not cover and stay on the CPU",
            n - n_eligible
        ));
    }

    // Largest context that still holds every eligible layer: binary search on the cache length.
    let (mut lo, mut hi) = (0u64, limit.min(s.context.max(1)));
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let o = plan_for(req, &eligible, per_dev, mid);
        let all = eligible.iter().take(o.gpu_layers).filter(|&&e| e).count() == n_eligible;
        if all {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    plan.max_context_all_layers = lo;
    if lo == 0 {
        plan.notes
            .push("even a one-token cache does not leave room for all eligible layers".into());
    }

    plan.est_decode_tps = req
        .cpu_decode_tps
        .filter(|t| *t > 0.0)
        .map(|cpu| estimate_tps(s, req, per_dev, on_gpu, cpu));
    if let Some(t) = plan.est_decode_tps {
        if req.cpu_decode_tps.is_some_and(|c| t <= c * 1.05) {
            plan.notes
                .push("this split is not expected to beat the CPU alone".into());
        }
        if req.class == DeviceClass::Software {
            plan.notes
                .push("a software adapter is slower than the CPU kernels".into());
        }
    }
    plan
}

fn estimate_tps(s: &Spec, req: &Request<'_>, per_dev: u64, on_gpu: usize, cpu_tps: f64) -> f64 {
    let total_layer_bytes = per_dev.saturating_mul(s.layers.max(1));
    let head = s.vocab.saturating_mul(s.hidden).saturating_mul(4);
    let total = (total_layer_bytes + head).max(1) as f64;
    let gpu_bytes = per_dev as f64 * on_gpu as f64;
    let frac = (gpu_bytes / total).clamp(0.0, 1.0);
    let t_gpu = gpu_bytes / (req.class.bandwidth_guess() * 0.6);
    let t_cpu = (1.0 - frac) / cpu_tps;
    // The hidden state crosses the boundary twice per token; small, but not free.
    let cross = if on_gpu > 0 && frac < 1.0 {
        2.0e-4
    } else {
        0.0
    };
    1.0 / (t_gpu + t_cpu + cross + 1.0e-4)
}

fn explain_ineligible(s: &Spec) -> String {
    let mut why: Vec<&str> = Vec::new();
    if s.norm != NormKind::Rms {
        why.push("LayerNorm");
    }
    if s.norm_offset {
        why.push("offset norm weights");
    }
    if s.act != Act::Silu {
        why.push("a non-SiLU activation");
    }
    if !s.gated_mlp {
        why.push("a non-gated MLP");
    }
    if s.qk_norm {
        why.push("QK-norm");
    }
    if s.qkv_bias {
        why.push("QKV biases");
    }
    if s.post_norms {
        why.push("post-norms");
    }
    if s.parallel_residual {
        why.push("parallel residuals");
    }
    if s.attn_softcap != 0.0 {
        why.push("attention soft-capping");
    }
    if s.rotary_dim != s.head_dim {
        why.push("partial RoPE");
    }
    if s.experts > 1 {
        why.push("mixture-of-experts layers");
    }
    if s.rope_scaling != RopeScaling::None {
        why.push("scaled RoPE");
    }
    if why.is_empty() {
        "the architecture is not supported by the GPU kernels".into()
    } else {
        format!(
            "no layer can be offloaded: this model uses {}",
            why.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama(layers: u64) -> Spec {
        Spec::dense("llama", layers, 4096, 32, 8, 128, 14336, 128_256, 8192)
    }
    fn req<'a>(s: &'a Spec, dtype: DType, vram: u64) -> Request<'a> {
        Request {
            spec: s,
            dtype,
            context: 4096,
            vram,
            reserve: 256 << 20,
            binding: 2 << 30,
            class: DeviceClass::Discrete,
            cpu_decode_tps: Some(4.0),
        }
    }

    #[test]
    fn device_bytes_follow_the_gpu_layouts() {
        // Q4_K stays raw: 144 bytes per 256 weights.
        assert_eq!(device_weight_bytes(DType::Q4K, 10, 512), Some(10 * 2 * 144));
        // Q8_0 widens the scale to f32: 36 bytes per block instead of 34.
        assert_eq!(device_weight_bytes(DType::Q8_0, 1, 64), Some(2 * 36));
        assert_eq!(device_weight_bytes(DType::F32, 3, 5), Some(60));
        assert_eq!(
            device_weight_bytes(DType::Q4K, 1, 100),
            None,
            "row length must be a block multiple"
        );
        assert_eq!(device_weight_bytes(DType::Q5K, 1, 256), None, "no kernel");
        assert_eq!(
            device_weight_bytes(DType::F32, u64::MAX, 4),
            None,
            "no overflow"
        );
    }

    #[test]
    fn everything_fits_on_a_big_card_and_the_plan_says_so() {
        let s = llama(32);
        let p = plan_offload(&req(&s, DType::Q4K, 24 << 30));
        assert_eq!(
            (p.gpu_layers, p.eligible_layers, p.limiter),
            (32, 32, Limiter::Nothing)
        );
        assert_eq!(
            p.max_context_all_layers,
            s.context.min(kv_capacity_limit(&s, 2 << 30))
        );
        assert!(p.est_decode_tps.unwrap() > 4.0, "{:?}", p.est_decode_tps);
    }

    #[test]
    fn a_small_card_takes_a_prefix_and_never_exceeds_its_memory() {
        let s = llama(32);
        for vram in [1u64 << 30, 2 << 30, 4 << 30, 6 << 30, 8 << 30] {
            let r = req(&s, DType::Q4K, vram);
            let p = plan_offload(&r);
            let used = p.gpu_layers as u64
                * (layer_device_bytes(&s, DType::Q4K).unwrap() + p.kv_bytes_per_layer);
            assert!(
                used + r.reserve <= vram || p.gpu_layers == 0,
                "vram {vram}: used {used}"
            );
            if p.gpu_layers < 32 {
                assert_eq!(p.limiter, Limiter::Memory);
                assert!(!p.notes.is_empty());
            }
        }
        let p4 = plan_offload(&req(&s, DType::Q4K, 4 << 30));
        let p8 = plan_offload(&req(&s, DType::Q4K, 8 << 30));
        assert!(p4.gpu_layers < p8.gpu_layers, "more memory, more layers");
    }

    #[test]
    fn a_shorter_context_buys_layers() {
        let s = llama(32);
        let vram = 5u64 << 30;
        let long = plan_offload(&Request {
            context: 8192,
            ..req(&s, DType::Q4K, vram)
        });
        let short = plan_offload(&Request {
            context: 512,
            ..req(&s, DType::Q4K, vram)
        });
        assert!(short.gpu_layers >= long.gpu_layers);
        assert!(
            short.max_context_all_layers == long.max_context_all_layers,
            "the all-layers context does not depend on the request"
        );
    }

    #[test]
    fn max_context_is_the_exact_boundary() {
        let s = llama(8);
        let r = req(&s, DType::Q4K, 3 << 30);
        let p = plan_offload(&r);
        assert!(p.max_context_all_layers > 0);
        let at = plan_offload(&Request {
            context: p.max_context_all_layers,
            ..r.clone()
        });
        assert_eq!(at.gpu_layers, 8);
        if p.max_context_all_layers < s.context.min(kv_capacity_limit(&s, r.binding)) {
            let over = plan_offload(&Request {
                context: p.max_context_all_layers + 1,
                ..r
            });
            assert!(over.gpu_layers < 8, "one more position no longer fits");
        }
    }

    #[test]
    fn the_binding_limit_caps_the_cache() {
        let s = llama(32);
        // Cache rows are 8 KV heads x 128 dims x 4 bytes = 4 KiB per position.
        let r = Request {
            binding: 8 << 20,
            ..req(&s, DType::Q4K, 24 << 30)
        };
        let cap = kv_capacity_limit(&s, r.binding);
        assert_eq!(cap, (8u64 << 20) / (8 * 128 * 4));
        assert_eq!(cap, 2048);
        let p = plan_offload(&r);
        assert_eq!(p.kv_capacity, 2048, "wanted 4096, the binding allows 2048");
        assert!(
            p.notes.iter().any(|n| n.contains("binding limit")),
            "{:?}",
            p.notes
        );
        let short = plan_offload(&Request {
            context: 512,
            ..r.clone()
        });
        assert_eq!(short.kv_capacity, 512);
        assert!(!short.notes.iter().any(|n| n.contains("binding limit")));
        // The spec's own context caps it first when the binding is generous.
        let wide = plan_offload(&Request {
            context: 1_000_000,
            binding: 2 << 30,
            ..r
        });
        assert_eq!(wide.kv_capacity, s.context);
    }

    #[test]
    fn unsupported_architectures_and_formats_stay_on_the_cpu_with_a_reason() {
        let mut s = llama(8);
        s.qk_norm = true;
        let p = plan_offload(&req(&s, DType::Q4K, 24 << 30));
        assert_eq!((p.gpu_layers, p.limiter), (0, Limiter::NoEligibleLayer));
        assert!(
            p.notes.iter().any(|n| n.contains("QK-norm")),
            "{:?}",
            p.notes
        );

        let mut m = llama(8);
        m.experts = 8;
        assert_eq!(eligible_layers(&m), vec![false; 8]);

        let odd = Spec::dense("llama", 4, 4000, 32, 8, 125, 11000, 32000, 2048);
        let p = plan_offload(&req(&odd, DType::Q4K, 24 << 30));
        assert_eq!(p.gpu_layers, 0);
        assert!(p.notes.iter().any(|n| n.contains("no kernel")));
        assert_eq!(
            plan_offload(&req(&odd, DType::F32, 24 << 30)).gpu_layers,
            4,
            "dense weights have no row-length rule"
        );
    }

    #[test]
    fn scaled_rope_blocks_global_layers_only_when_there_is_no_local_theta() {
        let mut s = llama(4);
        s.rope_scaling = RopeScaling::Linear { factor: 4.0 };
        assert_eq!(eligible_layers(&s), vec![false; 4]);
        s.sliding_window = 512;
        s.sliding_layers = vec![true, false, true, false];
        s.rope_theta_local = 10_000.0;
        assert_eq!(eligible_layers(&s), vec![true, false, true, false]);
    }

    #[test]
    fn no_memory_and_software_adapters_are_reported_honestly() {
        let s = llama(8);
        let p = plan_offload(&Request {
            vram: 0,
            ..req(&s, DType::Q4K, 0)
        });
        assert_eq!((p.gpu_layers, p.limiter), (0, Limiter::NoMemory));
        let soft = plan_offload(&Request {
            class: DeviceClass::Software,
            ..req(&s, DType::Q4K, 24 << 30)
        });
        assert!(
            soft.notes.iter().any(|n| n.contains("software")),
            "{:?}",
            soft.notes
        );
        assert!(
            soft.notes
                .iter()
                .any(|n| n.contains("not expected to beat")),
            "{:?}",
            soft.notes
        );
    }

    #[test]
    fn hostile_numbers_do_not_overflow() {
        let s = Spec::dense(
            "llama",
            2,
            u64::MAX / 4,
            u64::MAX / 2,
            1,
            u64::MAX / 2,
            u64::MAX / 3,
            1,
            u64::MAX,
        );
        // Only the arithmetic is under test; the plan itself may be empty.
        let _ = layer_device_bytes(&s, DType::F32);
        let _ = kv_capacity_limit(&s, u64::MAX);
        let small = llama(2);
        let p = plan_offload(&Request {
            context: u64::MAX,
            reserve: u64::MAX,
            ..req(&small, DType::Q4K, u64::MAX)
        });
        assert!(p.gpu_layers <= 2);
        let p = plan_offload(&Request {
            binding: 0,
            ..req(&small, DType::Q4K, 1 << 40)
        });
        let _ = plan_offload(&req(&s, DType::F32, u64::MAX));
        assert!(p.kv_capacity >= 1);
    }
}
