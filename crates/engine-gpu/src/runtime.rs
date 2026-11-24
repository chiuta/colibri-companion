//! Opening the best available backend, with a plain explanation whichever one it is.
//!
//! The rule is the engine's own ([`backend::build`]): a GPU that was asked for but cannot start
//! falls back to the CPU kernels and the description says why, so nothing switches silently.
//! This module adds the part only this crate knows: how to start the device runner, and what to
//! tell the user when the build has none, no adapter exists, or the adapter is too weak.
use crate::caps::Selection;
use companion_engine::backend::{self, Backend, GpuFactory, Preference};
use companion_engine::hw::HardwareProfile;
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct RuntimeOptions {
    pub selection: Selection,
    /// Bytes of device memory the engine may use; `None` = ask the hardware profile, else 4 GiB.
    pub vram_budget: Option<u64>,
}

/// The outcome of [`open`].
pub struct Opened {
    pub backend: Arc<dyn Backend>,
    /// One line: which backend, and why.
    pub description: String,
    /// `true` when the backend is a device (the model may offload layers to it).
    pub gpu: bool,
    /// Extra lines worth showing when the GPU was wanted but not obtained.
    pub diagnostics: Vec<String>,
}

/// Choose and start a backend. `factory` starts the device runner; pass `None` for a build without
/// one. Separate from [`open`] so the fallback paths are testable without a device.
pub fn open_with(
    profile: &HardwareProfile,
    pref: Preference,
    factory: Option<GpuFactory<'_>>,
) -> Opened {
    let has_runner = factory.is_some();
    let (backend, description) = backend::build(profile, pref, factory);
    let gpu = backend.is_accelerator();
    let mut diagnostics = Vec::new();
    if !gpu && pref == Preference::Gpu {
        diagnostics.push(if has_runner { "the GPU was requested but could not be used; running on the CPU".to_string() } else { "the GPU was requested but this build has no GPU runner (rebuild with --features wgpu); running on the CPU".to_string() });
        diagnostics.push(
            "set ENGINE_BACKEND=cpu to silence this, or run `corbul doctor` for the adapter list"
                .to_string(),
        );
    }
    Opened {
        backend: Arc::from(backend),
        description,
        gpu,
        diagnostics,
    }
}

/// Choose and start a backend for this machine.
pub fn open(profile: &HardwareProfile, pref: Preference, opts: &RuntimeOptions) -> Opened {
    #[cfg(feature = "wgpu")]
    {
        let gpu_opts = crate::gpu::GpuOptions {
            selection: opts.selection.clone(),
            vram_budget: opts.vram_budget,
            profile: Some(profile.clone()),
            ..Default::default()
        };
        let factory = move || crate::gpu::create(gpu_opts.clone());
        open_with(profile, pref, Some(&factory))
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = opts;
        open_with(profile, pref, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::TimedBackend;
    use companion_engine::backend::{BackendError, CpuBackend};
    use companion_engine::hw::{Gpu, GpuKind};

    fn big_gpu() -> HardwareProfile {
        HardwareProfile {
            gpus: vec![Gpu {
                name: "RTX".into(),
                vram: Some(16 << 30),
                kind: GpuKind::Nvidia,
            }],
            ..HardwareProfile::default()
        }
    }

    #[test]
    fn a_machine_without_a_gpu_runs_on_the_cpu_and_says_why() {
        let o = open_with(&HardwareProfile::default(), Preference::Auto, None);
        assert!(
            !o.gpu && o.description.starts_with("cpu") && o.description.contains("no GPU found"),
            "{}",
            o.description
        );
        assert!(
            o.diagnostics.is_empty(),
            "Auto asked for nothing, so nothing to complain about"
        );
    }

    #[test]
    fn asking_for_a_gpu_in_a_build_without_the_runner_is_explained() {
        let o = open_with(&big_gpu(), Preference::Gpu, None);
        assert!(!o.gpu);
        assert!(
            o.description.contains("no GPU backend"),
            "{}",
            o.description
        );
        assert!(
            o.diagnostics[0].contains("--features wgpu"),
            "{:?}",
            o.diagnostics
        );
    }

    #[test]
    fn a_runner_that_fails_to_start_falls_back_with_its_error() {
        let failing = || -> Result<Box<dyn Backend>, BackendError> {
            Err(BackendError::Device(
                "only a software adapter was found (llvmpipe)".into(),
            ))
        };
        let o = open_with(&big_gpu(), Preference::Auto, Some(&failing));
        assert!(!o.gpu);
        assert!(
            o.description.contains("llvmpipe") && o.description.contains("failed to start"),
            "{}",
            o.description
        );
        let o = open_with(&big_gpu(), Preference::Gpu, Some(&failing));
        assert!(
            o.diagnostics
                .iter()
                .any(|d| d.contains("could not be used")),
            "{:?}",
            o.diagnostics
        );
    }

    #[test]
    fn a_runner_that_starts_is_used_and_cpu_preference_wins_over_it() {
        // A device stand-in: the CPU backend dressed as an accelerator would need a new type, so the
        // factory returns the timing wrapper, which reports the inner backend honestly (not a GPU).
        let ok = || -> Result<Box<dyn Backend>, BackendError> {
            Ok(Box::new(TimedBackend::new(CpuBackend::new())))
        };
        let o = open_with(&big_gpu(), Preference::Auto, Some(&ok));
        assert!(o.description.starts_with("timed cpu"), "{}", o.description);
        assert!(
            !o.gpu,
            "is_accelerator comes from the backend, not from the request"
        );
        let forced = open_with(&big_gpu(), Preference::Cpu, Some(&ok));
        assert!(
            forced.description.contains("ENGINE_BACKEND=cpu"),
            "{}",
            forced.description
        );
    }

    #[test]
    fn open_never_panics_on_this_machine() {
        let o = open(
            &HardwareProfile::default(),
            Preference::Auto,
            &RuntimeOptions::default(),
        );
        assert!(!o.description.is_empty());
        let o = open(&big_gpu(), Preference::Gpu, &RuntimeOptions::default());
        assert!(!o.description.is_empty());
    }
}
