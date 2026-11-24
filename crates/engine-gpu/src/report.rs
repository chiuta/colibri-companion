//! The capability report `doctor` prints: which graphics APIs this build and this machine can
//! reach, which adapters exist, which one would be used and why, and how much memory it may take.
use crate::caps::{select, AdapterInfo, Api, SelectError, Selection};
use companion_engine::hw::HardwareProfile;

#[derive(Clone, Debug)]
pub struct Report {
    /// Was the device runner (`wgpu` feature) compiled in?
    pub compiled: bool,
    pub os: String,
    /// APIs expected on this OS, best first.
    pub expected: Vec<Api>,
    pub adapters: Vec<AdapterInfo>,
    pub selected: Option<usize>,
    pub why_not: Option<SelectError>,
    /// GPUs the hardware profile found by other means, with their memory when known.
    pub profile_gpus: Vec<(String, Option<u64>)>,
    pub unified_memory: bool,
    pub vram_budget: Option<u64>,
    pub notes: Vec<String>,
}

/// Memory the engine may take on the chosen device: 90 % of what the profile reports for a GPU
/// whose name matches, two thirds of RAM for unified memory (Apple's own guidance is of that
/// order), else `None` and the caller picks a default.
pub fn vram_budget(profile: &HardwareProfile, adapter_name: &str) -> Option<u64> {
    if profile.unified_memory { return profile.total_ram.map(|r| r / 3 * 2); }
    let name = adapter_name.to_ascii_lowercase();
    let matched = profile.gpus.iter().find(|g| { let n = g.name.to_ascii_lowercase(); !n.is_empty() && (name.contains(&n) || n.contains(&name)) });
    let g = matched.or(if profile.gpus.len() == 1 { profile.gpus.first() } else { None })?;
    g.vram.map(|v| v / 10 * 9)
}

impl Report {
    pub fn build(adapters: Vec<AdapterInfo>, compiled: bool, os: &str, sel: &Selection, profile: Option<&HardwareProfile>) -> Report {
        let expected = Api::expected_on(os);
        let mut notes = Vec::new();
        let (selected, why_not) = if !compiled {
            notes.push(format!("this build has no GPU runner (rebuild with --features wgpu); on {os} it would use {}", expected.iter().map(|a| a.name()).collect::<Vec<_>>().join(" or ")));
            (None, None)
        } else {
            match select(&adapters, sel, os) { Ok(i) => (Some(i), None), Err(e) => (None, Some(e)) }
        };
        if compiled && adapters.is_empty() {
            notes.push(match os {
                "linux" => "no Vulkan device: install the vendor driver or Mesa's Vulkan package (vulkaninfo should list a device)".to_string(),
                "windows" => "no DX12/Vulkan device: update the graphics driver".to_string(),
                "macos" => "no Metal device was reported".to_string(),
                _ => "no graphics device was reported".to_string(),
            });
        }
        if let Some(a) = selected.and_then(|i| adapters.get(i)) {
            if !a.limits.shader_f16 { notes.push("the device has no native f16 in shaders; the kernels widen halves by hand, so nothing is lost".into()); }
            if a.class == crate::caps::DeviceClass::Software { notes.push("a software adapter is slower than the CPU kernels; use it only for testing".into()); }
        }
        let (profile_gpus, unified_memory, budget) = match profile {
            Some(p) => (p.gpus.iter().map(|g| (g.name.clone(), g.vram)).collect(), p.unified_memory, selected.and_then(|i| adapters.get(i)).and_then(|a| vram_budget(p, &a.name))),
            None => (Vec::new(), false, None),
        };
        Report { compiled, os: os.to_string(), expected, adapters, selected, why_not, profile_gpus, unified_memory, vram_budget: budget, notes }
    }

    pub fn usable(&self) -> bool { self.selected.is_some() }
    pub fn selected_adapter(&self) -> Option<&AdapterInfo> { self.selected.and_then(|i| self.adapters.get(i)) }

    /// A short verdict line for the top of `doctor`.
    pub fn verdict(&self) -> String {
        match (self.selected_adapter(), &self.why_not) {
            (Some(a), _) => format!("GPU compute: usable through {} on {}", a.api.name(), a.name),
            (None, Some(e)) => format!("GPU compute: not usable ({e})"),
            (None, None) if !self.compiled => "GPU compute: not compiled into this build".into(),
            _ => "GPU compute: not usable".into(),
        }
    }

    pub fn text(&self) -> String {
        let mut o = String::new();
        o.push_str(&self.verdict());
        o.push('\n');
        o.push_str(&format!("graphics APIs expected on {}: {}\n", self.os, self.expected.iter().map(|a| a.name()).collect::<Vec<_>>().join(", ")));
        if self.compiled {
            if self.adapters.is_empty() { o.push_str("adapters: none\n"); }
            for (i, a) in self.adapters.iter().enumerate() {
                let mark = if self.selected == Some(i) { "*" } else { " " };
                o.push_str(&format!(" {mark}#{i} {}\n", a.summary()));
                for p in a.limits.problems() { o.push_str(&format!("      cannot run the kernels: {p}\n")); }
            }
        }
        for (n, v) in &self.profile_gpus {
            o.push_str(&format!("hardware profile: {n}{}\n", v.map_or(String::new(), |v| format!(", {} MiB", v >> 20))));
        }
        if self.unified_memory { o.push_str("memory: shared with the CPU (unified)\n"); }
        if let Some(b) = self.vram_budget { o.push_str(&format!("memory the engine may use: {} MiB\n", b >> 20)); }
        for n in &self.notes { o.push_str(&format!("note: {n}\n")); }
        o
    }

    pub fn json(&self) -> String {
        fn q(s: &str) -> String {
            let mut o = String::from("\"");
            for c in s.chars() {
                match c { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"), c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)), c => o.push(c) }
            }
            o.push('"');
            o
        }
        let adapters: Vec<String> = self.adapters.iter().map(|a| format!(
            "{{\"name\":{},\"api\":{},\"class\":{},\"driver\":{},\"max_storage_binding\":{},\"max_buffer\":{},\"max_workgroup_size\":{},\"shader_f16\":{},\"problems\":[{}]}}",
            q(&a.name), q(a.api.name()), q(a.class.name()), q(&a.driver), a.limits.max_storage_binding, a.limits.max_buffer, a.limits.max_workgroup_size_x, a.limits.shader_f16,
            a.limits.problems().iter().map(|p| q(p)).collect::<Vec<_>>().join(","))).collect();
        format!("{{\"compiled\":{},\"os\":{},\"usable\":{},\"selected\":{},\"verdict\":{},\"vram_budget\":{},\"unified_memory\":{},\"adapters\":[{}],\"notes\":[{}]}}",
            self.compiled, q(&self.os), self.usable(), self.selected.map_or("null".into(), |i| i.to_string()), q(&self.verdict()),
            self.vram_budget.map_or("null".into(), |v| v.to_string()), self.unified_memory, adapters.join(","),
            self.notes.iter().map(|n| q(n)).collect::<Vec<_>>().join(","))
    }
}

/// The report for this machine: with the `wgpu` feature the drivers are asked, without it the
/// report says what would be used and why nothing is.
pub fn capability_report(profile: Option<&HardwareProfile>) -> Report {
    capability_report_with(profile, &Selection::from_env())
}

pub fn capability_report_with(profile: Option<&HardwareProfile>, sel: &Selection) -> Report {
    #[cfg(feature = "wgpu")]
    { Report::build(crate::gpu::adapter_infos(wgpu::Backends::all()), true, std::env::consts::OS, sel, profile) }
    #[cfg(not(feature = "wgpu"))]
    { Report::build(Vec::new(), false, std::env::consts::OS, sel, profile) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{DeviceClass, Limits};
    use companion_engine::hw::{Gpu, GpuKind};

    fn ad(name: &str, api: Api, class: DeviceClass) -> AdapterInfo { AdapterInfo { name: name.into(), api, class, driver: "drv \"x\"".into(), vendor_id: 0x10de, limits: Limits::default() } }
    fn profile(gpus: Vec<Gpu>, unified: bool, ram: Option<u64>) -> HardwareProfile {
        HardwareProfile { gpus, unified_memory: unified, total_ram: ram, ..HardwareProfile::default() }
    }

    #[test]
    fn a_normal_machine_reports_the_chosen_adapter_and_its_memory_budget() {
        let p = profile(vec![Gpu { name: "NVIDIA GeForce RTX 4070".into(), vram: Some(12 << 30), kind: GpuKind::Nvidia }], false, Some(32 << 30));
        let list = vec![ad("NVIDIA GeForce RTX 4070", Api::Vulkan, DeviceClass::Discrete), ad("llvmpipe", Api::Vulkan, DeviceClass::Software)];
        let r = Report::build(list, true, "linux", &Selection::default(), Some(&p));
        assert!(r.usable());
        assert_eq!(r.selected, Some(0));
        assert_eq!(r.vram_budget, Some((12u64 << 30) / 10 * 9));
        let t = r.text();
        assert!(t.contains("usable through Vulkan"), "{t}");
        assert!(t.contains("*#0") && t.contains(" #1"), "{t}");
        assert!(t.contains("memory the engine may use"), "{t}");
    }

    #[test]
    fn no_adapter_gives_an_actionable_message_per_os() {
        for (os, needle) in [("linux", "vulkaninfo"), ("windows", "graphics driver"), ("macos", "Metal")] {
            let r = Report::build(vec![], true, os, &Selection::default(), None);
            assert!(!r.usable());
            assert!(r.text().contains(needle), "{os}: {}", r.text());
            assert!(r.verdict().contains("not usable"));
        }
    }

    #[test]
    fn a_build_without_the_runner_says_so_and_names_the_api_it_would_use() {
        let r = Report::build(vec![], false, "macos", &Selection::default(), None);
        assert!(r.verdict().contains("not compiled"));
        assert!(r.text().contains("--features wgpu") && r.text().contains("Metal"), "{}", r.text());
    }

    #[test]
    fn unified_memory_and_name_matching_budgets() {
        let apple = profile(vec![Gpu { name: "Apple M2".into(), vram: None, kind: GpuKind::Apple }], true, Some(24 << 30));
        assert_eq!(vram_budget(&apple, "Apple M2"), Some((24u64 << 30) / 3 * 2));
        let two = profile(vec![Gpu { name: "AMD Radeon".into(), vram: Some(8 << 30), kind: GpuKind::Amd }, Gpu { name: "Intel Arc".into(), vram: Some(4 << 30), kind: GpuKind::Intel }], false, None);
        assert_eq!(vram_budget(&two, "Intel Arc A380 Graphics"), Some((4u64 << 30) / 10 * 9));
        assert_eq!(vram_budget(&two, "unrelated"), None, "two GPUs and no name match: do not guess");
        let one = profile(vec![Gpu { name: "Whatever".into(), vram: Some(1 << 30), kind: GpuKind::Other }], false, None);
        assert!(vram_budget(&one, "something else").is_some(), "a single GPU is the only candidate");
        assert_eq!(vram_budget(&profile(vec![], false, None), "x"), None);
    }

    #[test]
    fn software_only_and_weak_adapters_are_explained() {
        let r = Report::build(vec![ad("llvmpipe", Api::Vulkan, DeviceClass::Software)], true, "linux", &Selection::default(), None);
        assert!(!r.usable());
        assert!(r.text().contains("software adapter"), "{}", r.text());
        let r = Report::build(vec![ad("llvmpipe", Api::Vulkan, DeviceClass::Software)], true, "linux", &Selection { allow_software: true, ..Default::default() }, None);
        assert!(r.usable() && r.text().contains("slower than the CPU"), "{}", r.text());
        let mut weak = ad("Tiny", Api::Vulkan, DeviceClass::Integrated);
        weak.limits.max_workgroup_size_x = 64;
        let r = Report::build(vec![weak], true, "linux", &Selection::default(), None);
        assert!(!r.usable());
        assert!(r.text().contains("cannot run the kernels"), "{}", r.text());
    }

    #[test]
    fn the_json_is_well_formed_and_escaped() {
        let list = vec![ad("Quote \"GPU\"\nline", Api::Dx12, DeviceClass::Discrete)];
        let r = Report::build(list, true, "windows", &Selection::default(), None);
        let j = r.json();
        assert!(j.starts_with('{') && j.ends_with('}'));
        assert!(j.contains("Quote \\\"GPU\\\"\\nline"), "{j}");
        assert!(j.contains("\"usable\":true") && j.contains("\"selected\":0"), "{j}");
        // Balanced braces and brackets outside strings.
        let (mut depth, mut in_str, mut esc) = (0i32, false, false);
        for c in j.chars() {
            if in_str { if esc { esc = false; } else if c == '\\' { esc = true; } else if c == '"' { in_str = false; } continue; }
            match c { '"' => in_str = true, '{' | '[' => depth += 1, '}' | ']' => depth -= 1, _ => {} }
            assert!(depth >= 0);
        }
        assert_eq!((depth, in_str), (0, false));
        let none = Report::build(vec![], true, "linux", &Selection::default(), None).json();
        assert!(none.contains("\"selected\":null") && none.contains("\"vram_budget\":null"), "{none}");
    }

    #[test]
    fn this_machine_report_never_panics() {
        let r = capability_report_with(None, &Selection::default());
        assert!(!r.text().is_empty());
        assert!(r.json().starts_with('{'));
        #[cfg(not(feature = "wgpu"))]
        assert!(!r.compiled);
    }
}
