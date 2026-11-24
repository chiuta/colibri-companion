//! What a device offers, what the kernels need, and how one adapter is chosen among several.
//!
//! Everything here is plain data and arithmetic, so it is tested on machines with no GPU; the
//! `wgpu` runner only fills [`AdapterInfo`] in from the real adapters.
use crate::{plan, shaders};
use std::fmt;

/// The graphics API an adapter is reached through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Api { Vulkan, Metal, Dx12, Gl, Other }

impl Api {
    pub fn parse(s: &str) -> Api {
        match s.trim().to_ascii_lowercase().as_str() {
            "vulkan" | "vk" => Api::Vulkan,
            "metal" | "mtl" => Api::Metal,
            "dx12" | "d3d12" | "directx12" => Api::Dx12,
            "gl" | "opengl" | "gles" => Api::Gl,
            _ => Api::Other,
        }
    }
    pub fn name(self) -> &'static str {
        match self { Api::Vulkan => "Vulkan", Api::Metal => "Metal", Api::Dx12 => "DX12", Api::Gl => "OpenGL", Api::Other => "other" }
    }
    /// The APIs a build for `os` (as in `std::env::consts::OS`) can be expected to reach, best first.
    pub fn expected_on(os: &str) -> Vec<Api> {
        match os {
            "macos" | "ios" => vec![Api::Metal],
            "windows" => vec![Api::Dx12, Api::Vulkan],
            "linux" | "android" | "freebsd" | "netbsd" | "openbsd" => vec![Api::Vulkan],
            _ => vec![Api::Vulkan],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceClass { Discrete, Integrated, Virtual, Software, Unknown }

impl DeviceClass {
    pub fn name(self) -> &'static str {
        match self { DeviceClass::Discrete => "discrete GPU", DeviceClass::Integrated => "integrated GPU", DeviceClass::Virtual => "virtual GPU", DeviceClass::Software => "software (CPU) adapter", DeviceClass::Unknown => "unknown device type" }
    }
    /// Rough device-memory bandwidth in bytes per second, used only for estimates. A class
    /// heuristic, not a measurement.
    pub fn bandwidth_guess(self) -> f64 {
        match self { DeviceClass::Discrete => 250e9, DeviceClass::Integrated => 40e9, DeviceClass::Virtual => 30e9, DeviceClass::Software => 10e9, DeviceClass::Unknown => 30e9 }
    }
}

/// The adapter limits the engine cares about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_storage_binding: u64,
    pub max_buffer: u64,
    pub max_workgroup_size_x: u32,
    pub max_invocations: u32,
    pub max_workgroups_per_dim: u32,
    pub max_storage_buffers_per_stage: u32,
    /// Native 16-bit float arithmetic in shaders (`shader-f16`). The current kernels widen halves
    /// by hand and do not need it; it is reported for the planner and for future kernels.
    pub shader_f16: bool,
    /// Subgroup size range, when the device reports one (0 = unknown).
    pub subgroup_min: u32,
    pub subgroup_max: u32,
}

impl Default for Limits {
    /// The WebGPU baseline: what every conforming adapter guarantees.
    fn default() -> Self {
        Limits { max_storage_binding: 128 << 20, max_buffer: 256 << 20, max_workgroup_size_x: 256, max_invocations: 256, max_workgroups_per_dim: 65_535, max_storage_buffers_per_stage: 8, shader_f16: false, subgroup_min: 0, subgroup_max: 0 }
    }
}

/// What the kernels in [`shaders`] need from a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requirements {
    pub workgroup_size: u32,
    pub invocations: u32,
    pub workgroups_per_dim: u32,
    pub storage_buffers: u32,
    /// Smallest storage binding that still holds one row of the widest supported format at a
    /// useful width (a 16384-column Q6_K row is 3.4 KB, so this is generous).
    pub min_binding: u64,
}

impl Requirements {
    pub fn of_kernels() -> Requirements {
        let wg = shaders::MATVEC_WG.max(shaders::NORM_WG).max(shaders::ATTN_WG).max(shaders::ELEM_WG);
        Requirements { workgroup_size: wg, invocations: wg, workgroups_per_dim: plan::MAX_DIM as u32, storage_buffers: 5, min_binding: 16 << 20 }
    }
}

impl Limits {
    /// Why this device cannot run the kernels; empty when it can.
    pub fn problems(&self) -> Vec<String> {
        let r = Requirements::of_kernels();
        let mut p = Vec::new();
        if self.max_workgroup_size_x < r.workgroup_size { p.push(format!("workgroup size {} is below the {} the kernels use", self.max_workgroup_size_x, r.workgroup_size)); }
        if self.max_invocations < r.invocations { p.push(format!("{} invocations per workgroup is below the {} the kernels use", self.max_invocations, r.invocations)); }
        if self.max_workgroups_per_dim < r.workgroups_per_dim { p.push(format!("{} workgroups per dimension is below the {} the dispatch arithmetic assumes", self.max_workgroups_per_dim, r.workgroups_per_dim)); }
        if self.max_storage_buffers_per_stage < r.storage_buffers { p.push(format!("{} storage buffers per stage is below the {} attention binds", self.max_storage_buffers_per_stage, r.storage_buffers)); }
        if self.max_storage_binding.min(self.max_buffer) < r.min_binding { p.push(format!("storage binding of {} MiB is too small", self.max_storage_binding.min(self.max_buffer) >> 20)); }
        p
    }
    /// The largest single binding the runner will use (it also never goes past 2 GiB, the limit of
    /// the 32-bit offsets the kernels compute).
    pub fn usable_binding(&self) -> u64 { self.max_storage_binding.min(self.max_buffer).min(1 << 31) }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterInfo {
    pub name: String,
    pub api: Api,
    pub class: DeviceClass,
    pub driver: String,
    pub vendor_id: u32,
    pub limits: Limits,
}

impl AdapterInfo {
    pub fn summary(&self) -> String {
        format!("{} [{}, {}], binding {} MiB, buffer {} MiB, workgroup {}, f16 {}", self.name, self.api.name(), self.class.name(), self.limits.max_storage_binding >> 20, self.limits.max_buffer >> 20, self.limits.max_workgroup_size_x, if self.limits.shader_f16 { "yes" } else { "no" })
    }
}

/// What the user asked for (flags or the `ENGINE_GPU*` variables).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    /// Accept a software adapter (llvmpipe, WARP): slower than the engine's own CPU kernels.
    pub allow_software: bool,
    pub api: Option<Api>,
    /// Pick the n-th adapter of the listing, or the first whose name contains this text.
    pub index: Option<usize>,
    pub name_contains: Option<String>,
}

impl Selection {
    /// `ENGINE_GPU` = an index or part of a name, `ENGINE_GPU_API` = vulkan|metal|dx12|gl,
    /// `ENGINE_GPU_SOFTWARE` = 1 to accept software adapters.
    pub fn from_env() -> Selection {
        let mut s = Selection::default();
        if let Ok(v) = std::env::var("ENGINE_GPU") {
            let v = v.trim();
            if let Ok(i) = v.parse::<usize>() { s.index = Some(i); } else if !v.is_empty() { s.name_contains = Some(v.to_string()); }
        }
        if let Ok(v) = std::env::var("ENGINE_GPU_API") { let a = Api::parse(&v); if a != Api::Other { s.api = Some(a); } }
        if let Ok(v) = std::env::var("ENGINE_GPU_SOFTWARE") { s.allow_software = matches!(v.trim(), "1" | "true" | "yes"); }
        s
    }
}

/// Why no adapter could be chosen, with one line per adapter that was looked at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectError { pub lines: Vec<String> }
impl fmt::Display for SelectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.lines.join("; ")) }
}
impl std::error::Error for SelectError {}

fn class_score(c: DeviceClass) -> u32 {
    match c { DeviceClass::Discrete => 40, DeviceClass::Integrated => 30, DeviceClass::Virtual => 20, DeviceClass::Unknown => 10, DeviceClass::Software => 0 }
}

/// Choose the adapter to run on. The same physical GPU is usually listed once per API; the API the
/// platform prefers (`Api::expected_on(os)`) wins among equals. Returns the index into `list`.
pub fn select(list: &[AdapterInfo], sel: &Selection, os: &str) -> Result<usize, SelectError> {
    if list.is_empty() { return Err(SelectError { lines: vec!["no adapter was reported by the graphics drivers (no Vulkan/Metal/DX12 driver, or no device)".into()] }); }
    let order = Api::expected_on(os);
    let mut lines = Vec::new();
    let mut best: Option<(u32, usize)> = None;
    for (i, a) in list.iter().enumerate() {
        let mut why: Vec<String> = Vec::new();
        if let Some(n) = sel.index.filter(|n| *n != i) { why.push(format!("not adapter #{n}")); }
        if let Some(t) = sel.name_contains.as_ref().filter(|t| !a.name.to_ascii_lowercase().contains(&t.to_ascii_lowercase())) { why.push(format!("name does not contain \"{t}\"")); }
        if let Some(api) = sel.api.filter(|api| a.api != *api) { why.push(format!("not reached through {}", api.name())); }
        if a.class == DeviceClass::Software && !sel.allow_software { why.push("software adapter (set ENGINE_GPU_SOFTWARE=1 to allow it)".into()); }
        why.extend(a.limits.problems());
        if !why.is_empty() { lines.push(format!("#{i} {} [{}]: {}", a.name, a.api.name(), why.join(", "))); continue; }
        let api_rank = order.iter().position(|x| *x == a.api).map_or(0, |p| 9 - p as u32);
        let score = class_score(a.class) * 100 + api_rank * 10 + (a.limits.max_storage_binding.min(a.limits.max_buffer) >> 28).min(9) as u32;
        // First of equal scores wins, so the listing order of the driver is the final tie-break.
        if best.is_none_or(|(s, _)| score > s) { best = Some((score, i)); }
    }
    best.map(|(_, i)| i).ok_or(SelectError { lines })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ad(name: &str, api: Api, class: DeviceClass) -> AdapterInfo {
        AdapterInfo { name: name.into(), api, class, driver: String::new(), vendor_id: 0, limits: Limits::default() }
    }

    #[test]
    fn the_baseline_limits_run_every_kernel() {
        assert!(Limits::default().problems().is_empty());
        let r = Requirements::of_kernels();
        assert_eq!(r.workgroup_size, 256);
        assert_eq!(r.workgroups_per_dim as u64, plan::MAX_DIM);
    }

    #[test]
    fn weak_devices_are_rejected_with_reasons() {
        let l = Limits { max_workgroup_size_x: 128, max_storage_buffers_per_stage: 4, max_storage_binding: 1 << 20, ..Limits::default() };
        let p = l.problems();
        assert_eq!(p.len(), 3, "{p:?}");
        assert!(p.iter().any(|s| s.contains("workgroup size 128")));
        assert!(p.iter().any(|s| s.contains("storage buffers")));
    }

    #[test]
    fn discrete_beats_integrated_and_the_platform_api_breaks_ties() {
        let list = vec![
            ad("Intel UHD", Api::Vulkan, DeviceClass::Integrated),
            ad("RTX", Api::Vulkan, DeviceClass::Discrete),
            ad("RTX", Api::Dx12, DeviceClass::Discrete),
        ];
        assert_eq!(select(&list, &Selection::default(), "windows"), Ok(2), "DX12 is preferred on Windows");
        assert_eq!(select(&list, &Selection::default(), "linux"), Ok(1), "Vulkan on Linux");
        let only_igpu = &list[..1];
        assert_eq!(select(only_igpu, &Selection::default(), "linux"), Ok(0));
    }

    #[test]
    fn software_adapters_need_an_explicit_opt_in() {
        let list = vec![ad("llvmpipe", Api::Vulkan, DeviceClass::Software)];
        let e = select(&list, &Selection::default(), "linux").unwrap_err();
        assert!(e.to_string().contains("software adapter"), "{e}");
        assert_eq!(select(&list, &Selection { allow_software: true, ..Default::default() }, "linux"), Ok(0));
    }

    #[test]
    fn explicit_choices_are_honoured_and_explained_when_impossible() {
        let list = vec![ad("Alpha", Api::Vulkan, DeviceClass::Integrated), ad("Beta", Api::Vulkan, DeviceClass::Discrete)];
        assert_eq!(select(&list, &Selection { index: Some(0), ..Default::default() }, "linux"), Ok(0));
        assert_eq!(select(&list, &Selection { name_contains: Some("alp".into()), ..Default::default() }, "linux"), Ok(0));
        let e = select(&list, &Selection { api: Some(Api::Metal), ..Default::default() }, "linux").unwrap_err();
        assert_eq!(e.lines.len(), 2);
        assert!(e.to_string().contains("not reached through Metal"));
        assert!(select(&[], &Selection::default(), "linux").unwrap_err().to_string().contains("no adapter"));
    }

    #[test]
    fn api_names_round_trip() {
        for a in [Api::Vulkan, Api::Metal, Api::Dx12, Api::Gl] { assert_eq!(Api::parse(a.name()), a); }
        assert_eq!(Api::parse("D3D12"), Api::Dx12);
        assert_eq!(Api::parse("banana"), Api::Other);
        assert_eq!(Api::expected_on("macos"), vec![Api::Metal]);
    }

    #[test]
    fn usable_binding_never_passes_32_bit_offsets() {
        let l = Limits { max_storage_binding: 8 << 30, max_buffer: 16 << 30, ..Limits::default() };
        assert_eq!(l.usable_binding(), 1 << 31);
    }
}
