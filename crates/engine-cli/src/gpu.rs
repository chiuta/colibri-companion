//! Putting models on the GPU, safely.
//!
//! One device is opened per process and shared. Every step can say "no" and the model then simply
//! keeps running on the CPU: no GPU in the build, `ENGINE_BACKEND=cpu`, no adapter, an adapter that
//! fails the start-up self-check, too little memory (partial offload), or an error while running
//! (the model switches itself back to the CPU). The reason goes to stderr in one line.
use companion_engine::backend::Preference;
use companion_engine::family::AnyModel;
use companion_engine::model::Model;

/// What the GPU build may do, as one line for `doctor`-style output.
pub fn available() -> bool {
    cfg!(feature = "gpu")
}

#[cfg(feature = "gpu")]
mod dev {
    use super::*;
    use companion_engine::backend::{Backend, CpuBackend, WeightRef};
    use std::sync::{Arc, OnceLock};

    pub struct Device {
        pub backend: Arc<dyn Backend>,
        pub description: String,
    }

    /// A tiny product on the device against the CPU's: a device that computes wrongly is refused.
    fn selftest(be: &Arc<dyn Backend>) -> Result<(), String> {
        let (rows, cols) = (48usize, 64usize);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 31 % 17) as f32 - 8.0) / 16.0)
            .collect();
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) / 7.0)
            .collect();
        let run = |b: &dyn Backend| -> Result<Vec<f32>, String> {
            let h = b
                .upload(WeightRef::dense(rows, cols, &w))
                .map_err(|e| e.to_string())?;
            let (bx, bo) = (
                b.alloc_from(&x).map_err(|e| e.to_string())?,
                b.alloc(rows).map_err(|e| e.to_string())?,
            );
            b.matvec(&h, &bx, &bo).map_err(|e| e.to_string())?;
            let mut out = vec![0.0; rows];
            b.read(&bo, &mut out).map_err(|e| e.to_string())?;
            Ok(out)
        };
        let (a, b) = (run(be.as_ref())?, run(&CpuBackend::new())?);
        let err = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (p, q)| m.max((p - q).abs()));
        let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        if a.iter().any(|v| !v.is_finite()) || err > 0.02 * scale {
            return Err(format!(
                "it multiplied wrongly in the self-check (error {err:.2e})"
            ));
        }
        Ok(())
    }

    pub fn device() -> Option<&'static Device> {
        static DEV: OnceLock<Option<Device>> = OnceLock::new();
        DEV.get_or_init(|| {
            let pref = Preference::from_env();
            if pref == Preference::Cpu {
                return None;
            }
            let hp = companion_engine::hw::HardwareProfile::detect(None);
            let o = companion_engine_gpu::runtime::open(
                &hp,
                pref,
                &companion_engine_gpu::runtime::RuntimeOptions::default(),
            );
            if !o.gpu {
                if pref == Preference::Gpu {
                    for l in &o.diagnostics {
                        eprintln!("corbul: {l}");
                    }
                }
                return None;
            }
            if let Err(e) = selftest(&o.backend) {
                eprintln!("corbul: the GPU ({}) is not used: {e}", o.description);
                return None;
            }
            Some(Device {
                backend: o.backend,
                description: o.description,
            })
        })
        .as_ref()
    }
}

/// Put what can go on the GPU: Gemma 4 and the older families through their matrices.
pub fn accelerate_family(model: &mut AnyModel) {
    #[cfg(feature = "gpu")]
    {
        let Some(d) = dev::device() else { return };
        let rep = match model {
            AnyModel::Gemma4(g) => g.attach_backend(d.backend.clone()),
            AnyModel::Legacy(g) => g.attach_backend(d.backend.clone()),
            AnyModel::GptOss(g) => g.attach_backend(d.backend.clone()),
            AnyModel::Ssm(g) => g.attach_backend(d.backend.clone()),
            AnyModel::Hybrid(g) => g.attach_backend(d.backend.clone()),
            _ => return,
        };
        say(
            &d.description,
            &rep.note,
            rep.active,
            rep.matrices,
            rep.total,
            rep.bytes,
        );
    }
    #[cfg(not(feature = "gpu"))]
    let _ = model;
}

/// Run `attach` with the shared device (if there is one) and say in one line what came of it.
#[cfg(feature = "gpu")]
fn attach_with(
    attach: impl FnOnce(
        std::sync::Arc<dyn companion_engine::backend::Backend>,
    ) -> companion_engine::accel::AttachReport,
) {
    let Some(d) = dev::device() else { return };
    let rep = attach(d.backend.clone());
    say(
        &d.description,
        &rep.note,
        rep.active,
        rep.matrices,
        rep.total,
        rep.bytes,
    );
}

/// BERT / embedding / reranker models: the attention, feed-forward, pooler and classifier matrices.
pub fn accelerate_encoder(enc: &mut companion_engine::bert::Encoder) {
    #[cfg(feature = "gpu")]
    attach_with(|be| enc.attach_backend(be));
    #[cfg(not(feature = "gpu"))]
    let _ = enc;
}

/// Vision encoders (patch embedding, attention, feed-forward, projector).
pub fn accelerate_vision(model: &mut companion_engine::vision::VisionModel) {
    #[cfg(feature = "gpu")]
    attach_with(|be| model.attach_backend(be));
    #[cfg(not(feature = "gpu"))]
    let _ = model;
}

/// Whisper-style speech models (encoder, decoder, cross-attention, output head).
pub fn accelerate_whisper(model: &mut companion_engine::audio::Whisper) {
    #[cfg(feature = "gpu")]
    attach_with(|be| model.attach_backend(be));
    #[cfg(not(feature = "gpu"))]
    let _ = model;
}

/// The linear layers that map audio-encoder frames to language-model embeddings.
pub fn accelerate_audio_projector(proj: &mut companion_engine::audio::AudioProjector) {
    #[cfg(feature = "gpu")]
    attach_with(|be| proj.attach_backend(be));
    #[cfg(not(feature = "gpu"))]
    let _ = proj;
}

#[cfg(feature = "gpu")]
fn say(desc: &str, note: &str, active: bool, n: usize, total: usize, bytes: u64) {
    if active {
        let extra = if note.is_empty() {
            String::new()
        } else {
            format!(", {note}")
        };
        eprintln!(
            "corbul: GPU ({desc}): {n}/{total} matrices on the device ({} MiB){extra}",
            bytes >> 20
        );
    } else {
        eprintln!("corbul: the GPU is not used ({note}); running on the CPU");
    }
}

/// Offload the layers of a dense / mixture-of-experts transformer. `rebuild` makes a fresh CPU
/// model; it is only called when the backend refuses a model it has already consumed.
pub fn accelerate_dense(
    model: Model,
    rebuild: &dyn Fn() -> Result<Model, String>,
) -> Result<Model, String> {
    #[cfg(feature = "gpu")]
    {
        use companion_engine::model::AccelOptions;
        let Some(d) = dev::device() else {
            return Ok(model);
        };
        let total = model.spec.layers as usize;
        match model.with_backend_opts(d.backend.clone(), AccelOptions::default()) {
            Ok(m) => {
                let on = m.offloaded().iter().filter(|b| **b).count();
                if on == 0 {
                    eprintln!(
                        "corbul: the GPU ({}) took no layer of this model; running on the CPU",
                        d.description
                    );
                } else {
                    eprintln!(
                        "corbul: GPU ({}): {on}/{total} layers on the device, the rest on the CPU",
                        d.description
                    );
                }
                Ok(m)
            }
            Err(e) => {
                eprintln!("corbul: the GPU could not take the model ({e}); running on the CPU");
                rebuild()
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = rebuild;
        Ok(model)
    }
}

/// `true` when the user asked for the CPU only.
pub fn cpu_only() -> bool {
    Preference::from_env() == Preference::Cpu
}
