//! Detailed sheets for the models colibri supports: what each can do, which files go in and
//! come out, and how it compares with the popular online assistants.
//!
//! The model facts come from colibri's own README (a field it does not state stays `None` or
//! empty: nothing here is guessed). The online assistants are described only by the kinds of
//! features their own sites advertise, never by benchmark scores; check each provider's page.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cap {
    Chat,
    Thinking,
    Tools,
    Vision,
    Code,
    ImageOut,
    Decisions,
}

impl Cap {
    pub const ALL: [Cap; 7] = [
        Cap::Chat,
        Cap::Thinking,
        Cap::Tools,
        Cap::Vision,
        Cap::Code,
        Cap::ImageOut,
        Cap::Decisions,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gpu {
    Cuda,
    Vulkan,
    Cpu,
}

/// How sure the README is about a capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Know {
    Yes,
    No,
    /// The README doesn't say.
    Unstated,
}

/// What a model can take in, according to the people who made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Text,
    Image,
    Video,
    Audio,
}

#[derive(Clone, Debug)]
pub struct Sheet {
    /// From the model's maker (model card or docs); colibri may use only part of it.
    pub maker_inputs: &'static [Input],
    pub maker_context: Option<&'static str>,
    pub maker_license: Option<&'static str>,
    pub maker_active: Option<&'static str>,
    pub id: &'static str,
    pub name: &'static str,
    pub params: &'static str,
    pub download_gb: Option<f64>,
    pub ram_min_gb: Option<f64>,
    pub ram_good_gb: Option<f64>,
    pub gpus: &'static [Gpu],
    pub caps: &'static [(Cap, Know)],
    /// Context window in tokens, only where colibri states one.
    pub context: Option<&'static str>,
    pub license: Option<&'static str>,
    /// `true` when the model is installed with `coli setup --model` (otherwise: by hand/GGUF).
    pub one_step: bool,
}

impl Sheet {
    pub fn know(&self, c: Cap) -> Know {
        self.caps
            .iter()
            .find(|(k, _)| *k == c)
            .map_or(Know::Unstated, |(_, v)| *v)
    }

    /// File kinds that can go into a chat with this model.
    pub fn accepts_images(&self) -> bool {
        self.know(Cap::Vision) == Know::Yes
    }

    pub fn makes_images(&self) -> bool {
        self.know(Cap::ImageOut) == Know::Yes
    }
}

use Cap::*;
use Gpu::*;
use Know::{No, Yes};

const fn s(
    id: &'static str,
    name: &'static str,
    params: &'static str,
    download_gb: f64,
    ram: (f64, f64),
    gpus: &'static [Gpu],
    caps: &'static [(Cap, Know)],
) -> Sheet {
    Sheet {
        maker_inputs: &[],
        maker_context: None,
        maker_license: None,
        maker_active: None,
        id,
        name,
        params,
        download_gb: Some(download_gb),
        ram_min_gb: Some(ram.0),
        ram_good_gb: Some(ram.1),
        gpus,
        caps,
        context: None,
        license: None,
        one_step: true,
    }
}

pub fn all() -> Vec<Sheet> {
    let mut v = vec![
        s(
            "qwen36-35b",
            "Qwen3.6-35B-A3B",
            "35B (A3B)",
            23.0,
            (10.0, 20.0),
            &[Cuda, Vulkan],
            &[(Chat, Yes), (Thinking, Yes), (Tools, Yes)],
        ),
        s(
            "qwen3-coder-30b",
            "Qwen3-Coder-30B-A3B",
            "30B (A3B)",
            19.0,
            (8.0, 18.0),
            &[Cuda, Vulkan],
            &[(Chat, Yes), (Code, Yes), (Tools, Yes), (Thinking, No)],
        ),
        Sheet {
            license: Some("non-commercial"),
            ..s(
                "qwen-image-2.1",
                "Qwen-Image-2.1",
                "n/a",
                33.0,
                (12.0, 18.0),
                &[Vulkan],
                &[(ImageOut, Yes), (Chat, No)],
            )
        },
        s(
            "deepseek-v4-flash-reap",
            "DeepSeek V4 Flash REAP 150B",
            "150B",
            85.0,
            (16.0, 32.0),
            &[Cuda, Vulkan],
            &[(Chat, Yes)],
        ),
        s(
            "deepseek-v4-flash",
            "DeepSeek V4 Flash",
            "284B",
            167.0,
            (16.0, 32.0),
            &[Cuda, Vulkan],
            &[(Chat, Yes), (Tools, Yes)],
        ),
        s(
            "mimo-v2.6-flash",
            "MiMo-V2.6 Flash",
            "309B",
            172.0,
            (32.0, 52.0),
            &[Vulkan],
            &[(Chat, Yes), (Vision, Yes), (Tools, Yes)],
        ),
        Sheet {
            context: Some("262,144 (tested)"),
            ..s(
                "qwen38-flash-next",
                "Qwen3.8-Flash-Next",
                "125B + 51B n-gram",
                186.0,
                (24.0, 32.0),
                &[Cuda, Vulkan],
                &[(Chat, Yes), (Vision, Yes), (Tools, Yes)],
            )
        },
        s(
            "glm-5.2",
            "GLM-5.2",
            "744B",
            429.0,
            (16.0, 24.0),
            &[Cuda, Vulkan],
            &[(Chat, Yes), (Tools, Yes)],
        ),
        s(
            "glm-5.3",
            "GLM-5.3",
            "744B",
            419.0,
            (16.0, 24.0),
            &[Cuda, Vulkan],
            &[(Chat, Yes), (Tools, Yes)],
        ),
        s(
            "inkling",
            "Inkling",
            "975B",
            514.0,
            (120.0, 128.0),
            &[Cuda, Vulkan],
            &[(Chat, Yes), (Tools, No)],
        ),
        s(
            "mimo-v2.6-pro",
            "MiMo-V2.6 Pro",
            "1.02T",
            564.0,
            (54.0, 64.0),
            &[Vulkan],
            &[(Chat, Yes), (Vision, Yes), (Tools, Yes)],
        ),
        s(
            "kimi-k3",
            "Kimi K3",
            "2.8T",
            1560.0,
            (32.0, 64.0),
            &[Cuda, Vulkan],
            &[(Chat, Yes), (Tools, Yes)],
        ),
    ];
    // "By hand" and GGUF entries: no one-step id.
    let by_hand = |name: &'static str,
                   params: &'static str,
                   dl: Option<f64>,
                   ram: Option<f64>,
                   gpus: &'static [Gpu],
                   caps: &'static [(Cap, Know)],
                   lic: Option<&'static str>| Sheet {
        maker_inputs: &[],
        maker_context: None,
        maker_license: None,
        maker_active: None,
        id: "",
        name,
        params,
        download_gb: dl,
        ram_min_gb: ram,
        ram_good_gb: None,
        gpus,
        caps,
        context: None,
        license: lic,
        one_step: false,
    };
    v.push(by_hand(
        "OLMoE",
        "7B",
        Some(14.0),
        Some(8.0),
        &[Vulkan],
        &[(Chat, Yes), (Tools, No)],
        None,
    ));
    v.push(by_hand(
        "Qwen3.8-27B (dense)",
        "27B",
        Some(56.0),
        Some(20.0),
        &[Vulkan],
        &[(Chat, Yes), (Vision, Yes)],
        None,
    ));
    v.push(by_hand(
        "GLM-5.3-Flash",
        "321B",
        Some(328.0),
        Some(25.0),
        &[Cuda, Vulkan],
        &[(Chat, Yes), (Vision, Yes), (Tools, Yes)],
        None,
    ));
    v.push(by_hand(
        "DeepSeek V4.1 Flash",
        "552B",
        Some(510.0),
        Some(18.0),
        &[Vulkan],
        &[(Chat, Yes), (Vision, Yes), (Tools, Yes)],
        None,
    ));
    v.push(by_hand(
        "Mistral-Large-4 (GGUF)",
        "1T, 52B active",
        Some(233.0),
        None,
        &[Cuda, Vulkan],
        &[(Chat, Yes), (Thinking, Yes), (Vision, Yes), (Tools, Yes)],
        None,
    ));
    v.push(by_hand(
        "Muse Spark Open (GGUF)",
        "816B, 42B active",
        Some(176.0),
        None,
        &[Cuda, Vulkan],
        &[(Chat, Yes), (Tools, Yes)],
        Some("Meta licence agreement"),
    ));
    v.push(by_hand(
        "Flint (GGUF)",
        "40B, 2.5B active",
        Some(11.4),
        None,
        &[Cuda, Vulkan],
        &[(Chat, Yes), (Tools, Yes)],
        Some("Apache 2.0"),
    ));
    v.push(by_hand(
        "LFM3-VL-2B (GGUF)",
        "2B",
        None,
        None,
        &[Cuda, Vulkan],
        &[(Chat, Yes), (Vision, Yes), (Tools, Yes)],
        Some("LFM Open License"),
    ));
    v.push(by_hand(
        "Laya (decision model)",
        "n/a",
        Some(0.85),
        Some(1.7),
        &[Cpu, Cuda],
        &[(Decisions, Yes)],
        None,
    ));
    v.push(by_hand(
        "GLiNER2.5-Decide (decision model)",
        "n/a",
        Some(1.95),
        Some(1.9),
        &[Cpu, Cuda],
        &[(Decisions, Yes)],
        None,
    ));
    v.push(by_hand(
        "Clef (decision model)",
        "27B",
        Some(55.0),
        Some(19.0),
        &[Cpu],
        &[(Decisions, Yes), (Chat, Yes)],
        None,
    ));
    for sh in &mut v {
        apply_maker(sh);
    }
    v
}

const T: &[Input] = &[Input::Text];
const TI: &[Input] = &[Input::Text, Input::Image];
const TIV: &[Input] = &[Input::Text, Input::Image, Input::Video];
const TIVA: &[Input] = &[Input::Text, Input::Image, Input::Video, Input::Audio];
const TIA: &[Input] = &[Input::Text, Input::Image, Input::Audio];

type Maker = (
    &'static [Input],
    Option<&'static str>,
    Option<&'static str>,
    Option<&'static str>,
);

/// Facts from each model's own card or docs (checked online in November 2026). Where sources
/// disagreed, the field is left empty rather than picked.
fn apply_maker(sh: &mut Sheet) {
    let _ = TI;
    let m: Maker = match sh.id {
        "qwen36-35b" => (
            TIV,
            Some("262,144 (up to ~1M with YaRN)"),
            Some("Apache 2.0"),
            Some("3B"),
        ),
        "qwen3-coder-30b" => (
            T,
            Some("262,144 (up to 1M with YaRN)"),
            Some("Apache 2.0"),
            Some("3.3B"),
        ),
        "qwen-image-2.1" => (
            T,
            None,
            Some("Qwen Research License (non-commercial)"),
            None,
        ),
        "deepseek-v4-flash" => (T, Some("1,048,576"), Some("MIT"), Some("13B")),
        "mimo-v2.6-flash" => (TIVA, Some("1,048,576"), Some("MIT"), Some("15B")),
        "mimo-v2.6-pro" => (TIVA, Some("1,048,576"), Some("MIT"), Some("42B")),
        "qwen38-flash-next" => (
            TIV,
            Some("262,144 (up to 1M with YaRN)"),
            Some("Qwen Community License 1.0"),
            Some("6B"),
        ),
        "glm-5.3" => (
            T,
            Some("1,048,576"),
            Some("GLM-5.3 License (MIT-like, extra clause for very large hosting services)"),
            None,
        ),
        "inkling" => (
            TIA,
            Some("up to 1,000,000"),
            Some("Apache 2.0"),
            Some("41B"),
        ),
        "kimi-k3" => (
            TIV,
            Some("1,048,576"),
            Some("Kimi K3 License (MIT-style, revenue and user-count clauses)"),
            Some("104B"),
        ),
        _ if sh.name == "GLM-5.3-Flash" => (TIV, None, Some("MIT"), Some("18B")),
        _ => (&[], None, None, None),
    };
    sh.maker_inputs = m.0;
    sh.maker_context = m.1;
    sh.maker_license = m.2;
    sh.maker_active = m.3;
}

pub fn by_id(id: &str) -> Option<Sheet> {
    all().into_iter().find(|s| s.id == id && !id.is_empty())
}

/// What goes in and what comes out, the same for every chat model in this app.
pub struct FileSupport {
    pub text_in: &'static [&'static str],
    pub office_in: &'static [&'static str],
}

pub const FILES: FileSupport = FileSupport {
    text_in: &["txt", "md", "csv", "json", "html", "code files", "pdf"],
    office_in: &["docx", "xlsx", "pptx", "odt/ods/odp"],
};

/// One online assistant, described by features its own site advertises (no scores).
pub struct Online {
    pub name: &'static str,
    /// Typical features: (files in, images in, makes images, web search, runs on your machine).
    pub files_in: bool,
    pub images_in: bool,
    pub makes_images: bool,
    pub web_search: bool,
    pub site: &'static str,
}

pub const ONLINE: &[Online] = &[
    Online {
        name: "ChatGPT (OpenAI)",
        files_in: true,
        images_in: true,
        makes_images: true,
        web_search: true,
        site: "https://chatgpt.com",
    },
    Online {
        name: "Claude (Anthropic)",
        files_in: true,
        images_in: true,
        makes_images: false,
        web_search: true,
        site: "https://claude.ai",
    },
    Online {
        name: "Gemini (Google)",
        files_in: true,
        images_in: true,
        makes_images: true,
        web_search: true,
        site: "https://gemini.google.com",
    },
    Online {
        name: "Le Chat (Mistral)",
        files_in: true,
        images_in: true,
        makes_images: true,
        web_search: true,
        site: "https://chat.mistral.ai",
    },
    Online {
        name: "DeepSeek",
        files_in: true,
        images_in: true,
        makes_images: false,
        web_search: true,
        site: "https://chat.deepseek.com",
    },
    Online {
        name: "Copilot (Microsoft)",
        files_in: true,
        images_in: true,
        makes_images: true,
        web_search: true,
        site: "https://copilot.microsoft.com",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_match_colibri_setup_ids_and_are_unique() {
        let ids: Vec<&str> = all().iter().filter(|s| s.one_step).map(|s| s.id).collect();
        assert_eq!(ids.len(), 12);
        let mut u = ids.clone();
        u.sort();
        u.dedup();
        assert_eq!(u.len(), ids.len());
        let q = by_id("qwen38-flash-next").unwrap();
        assert!(q.accepts_images() && q.context.is_some());
        assert!(by_id("qwen-image-2.1").unwrap().makes_images());
        assert_eq!(by_id("inkling").unwrap().know(Cap::Tools), Know::No);
        assert_eq!(by_id("glm-5.2").unwrap().know(Cap::Vision), Know::Unstated);
        assert!(by_id("").is_none());
    }

    #[test]
    fn maker_facts_are_attached_and_differ_from_what_colibri_states() {
        let q = by_id("qwen36-35b").unwrap();
        assert!(q.maker_inputs.contains(&Input::Image));
        assert_eq!(q.know(Cap::Vision), Know::Unstated); // colibri doesn't list it
        assert!(by_id("deepseek-v4-flash").unwrap().maker_inputs == [Input::Text]);
        assert!(
            by_id("inkling")
                .unwrap()
                .maker_inputs
                .contains(&Input::Audio)
        );
        assert_eq!(by_id("inkling").unwrap().know(Cap::Tools), Know::No);
        assert!(by_id("glm-5.2").unwrap().maker_inputs.is_empty());
    }
}
