//! Verified, dated catalogue of *light* local models (fit in 4/8/16/32 GB RAM).
//!
//! "Light" here only means: the download at the stated quantisation fits
//! comfortably in ordinary RAM. It says nothing about quality.
//!
//! Rules of this file:
//! * Only entries whose figures were read on a primary page (vendor, Ollama
//!   library or Hugging Face model card) on `CHECKED` are in [`LIGHT_MODELS`].
//!   Everything else is in [`UNVERIFIED`] with the reason.
//! * Sizes are the figures printed on the primary page. Ollama prints a range
//!   (several quantisations, default not named): `size_lo_gb`..`size_gb`, tiers use the high end.
//! * Engine: no light model is in the colibri catalogue, so all entries are
//!   reached through any OpenAI-compatible server (`settings.server_url`).
//! * RAM tiers are derived, never invented:
//!   `minimum = size_gb + 2.0` (runtime + OS; the colibri README's documented RAM wins when present), `comfortable = size_gb * 1.5 + 2.0`
//!   (adds room for the KV cache / context), each rounded UP to the next tier of
//!   4 / 8 / 16 / 32 GB (above 32 GB: `None`).
//! * Agents. Companion's MCP tools reach the model as plain text (a fenced
//!   `tool` block parsed by `mcp::parse_tool_request`), so no model NEEDS the
//!   OpenAI `tools` field. What a model needs: follow that fenced format,
//!   stream over SSE with `stream_options.include_usage` (`chat.rs`), optional
//!   `reasoning_content` / `<think>` deltas, `image_url` parts only for vision
//!   use, and context for up to `mcp::MAX_TOOL_ROUNDS` (5) rounds of tool output.
//!   [`AgentFit`] records what primary pages document; nothing was run here.
//! * [`for_ram`] ordering is a transparent heuristic (larger download that still
//!   fits comfortably first). It is NOT a quality benchmark.

/// Date on which the figures below were read.
pub const CHECKED: &str = "2026-11-16";

/// RAM tiers in GB.
pub const TIERS: [u32; 4] = [4, 8, 16, 32];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// Documented in the colibri README (needs a conversion step there).
    Colibri,
    /// Any OpenAI-compatible server (Ollama, llama.cpp, LM Studio).
    OpenAiServer,
    /// BF16 + vLLM on a datacentre GPU only; never offered by [`for_ram`].
    GpuVllmOnly,
}

/// Evidence for one agent capability. `Documented` carries the primary-page URL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    Documented(&'static str),
    /// No primary page read says either way.
    Unverified,
    /// A primary page says it is not supported.
    No,
}

/// What a model is documented to do for local agents. Filled ONLY from primary
/// pages (model cards, Ollama capability badges, vendor docs). Nothing here
/// was run against Companion's flows.
#[derive(Clone, Copy, Debug)]
pub struct AgentFit {
    pub tool_calling: Support,
    pub agentic_coding: Support,
    pub structured_output: Support,
    /// Context in K tokens as stated on the page (the same figure as `context`).
    pub min_context_k: Option<u32>,
    pub notes: &'static str,
}

const AGENT_UNKNOWN: AgentFit = AgentFit {
    tool_calling: Support::Unverified,
    agentic_coding: Support::Unverified,
    structured_output: Support::Unverified,
    min_context_k: None,
    notes: "no primary page read documents agent features",
};

const OLLAMA_GEMMA4: &str = "https://registry.ollama.ai/library/gemma4";

#[derive(Clone, Copy, Debug)]
pub struct LightModel {
    pub id: &'static str,
    pub name: &'static str,
    pub vendor: &'static str,
    /// Release month as stated on the primary page, or why it is unknown.
    pub released: &'static str,
    pub params_total: &'static str,
    /// Active parameters for MoE; `None` if dense or not stated on the page.
    pub params_active: Option<&'static str>,
    /// Download size: lowest and highest figure printed on the primary page
    /// (equal when the page prints one figure). RAM tiers use the HIGHEST,
    /// because pages do not say which tag is the default quantisation.
    pub size_lo_gb: f64,
    pub size_gb: f64,
    pub quant: &'static str,
    pub context: &'static str,
    pub modalities: &'static str,
    /// Exact licence string from the page.
    pub licence: &'static str,
    /// `false` = not a standard permissive licence; show a warning.
    pub permissive: bool,
    pub strength: &'static str,
    pub ollama_tag: Option<&'static str>,
    pub hf_repo: Option<&'static str>,
    pub engine: Engine,
    /// RAM documented by the colibri README, used as the minimum when present.
    pub documented_ram_gb: Option<u32>,
    pub note: &'static str,
    pub agent: AgentFit,
    pub sources: &'static [&'static str],
    pub checked: &'static str,
}

pub const LIGHT_MODELS: &[LightModel] = &[
    LightModel {
        id: "gemma4-e2b",
        name: "Gemma 4 E2B",
        vendor: "Google",
        released: "not stated (model card updated 2026-07-30)",
        params_total: "5.1B (2.3B effective)",
        params_active: None,
        size_lo_gb: 4.6,
        size_gb: 7.5,
        quant: "Ollama default tags (page prints 4.6-7.5 GB)",
        context: "128K",
        modalities: "text, image, audio",
        licence: "Apache 2.0",
        permissive: true,
        strength: "Smallest multimodal Gemma 4 for phones and 8 GB laptops",
        ollama_tag: Some("gemma4:e2b"),
        hf_repo: None,
        engine: Engine::OpenAiServer,
        documented_ram_gb: None,
        note: "The Ollama page prints 4.6-7.5 GB for the e2b tag without naming quantisations; 7.5 GB equals the separate e2b-mlx tag, but the page does not say which variant is which. Plan for the high figure.",
        agent: AgentFit {
            tool_calling: Support::Documented(OLLAMA_GEMMA4),
            agentic_coding: Support::Unverified,
            structured_output: Support::Unverified,
            min_context_k: Some(128),
            notes: "Ollama tools badge and 'native function-calling support' on the library page",
        },
        sources: &[
            "https://registry.ollama.ai/library/gemma4",
            "https://ai.google.dev/gemma/docs/core/model_card_4",
        ],
        checked: CHECKED,
    },
    LightModel {
        id: "gemma4-e4b",
        name: "Gemma 4 E4B",
        vendor: "Google",
        released: "not stated (model card updated 2026-07-30)",
        params_total: "8B (4.5B effective)",
        params_active: None,
        size_lo_gb: 6.6,
        size_gb: 9.5,
        quant: "Ollama default tags (page prints 6.6-9.5 GB)",
        context: "128K",
        modalities: "text, image, audio",
        licence: "Apache 2.0",
        permissive: true,
        strength: "Everyday multimodal model for 16 GB machines",
        ollama_tag: Some("gemma4:e4b"),
        hf_repo: None,
        engine: Engine::OpenAiServer,
        documented_ram_gb: None,
        note: "Range printed by Ollama without quantisation names; the page lists input as text+image, the model card adds audio.",
        agent: AgentFit {
            tool_calling: Support::Documented(OLLAMA_GEMMA4),
            agentic_coding: Support::Unverified,
            structured_output: Support::Unverified,
            min_context_k: Some(128),
            notes: "Ollama tools badge and 'native function-calling support' on the library page",
        },
        sources: &[
            "https://registry.ollama.ai/library/gemma4",
            "https://ai.google.dev/gemma/docs/core/model_card_4",
        ],
        checked: CHECKED,
    },
    LightModel {
        id: "gemma4-12b",
        name: "Gemma 4 12B Unified",
        vendor: "Google",
        released: "not stated (technical report published Jul 2026)",
        params_total: "11.95B",
        params_active: None,
        size_lo_gb: 7.7,
        size_gb: 8.0,
        quant: "Ollama default tags (page prints 7.7-8.0 GB)",
        context: "256K",
        modalities: "text, image, audio (video as frames)",
        licence: "Apache 2.0",
        permissive: true,
        strength: "Encoder-free multimodal design, single decoder",
        ollama_tag: Some("gemma4:12b"),
        hf_repo: Some("google/gemma-4-12B"),
        engine: Engine::OpenAiServer,
        documented_ram_gb: None,
        note: "Range printed by Ollama without quantisation names.",
        agent: AgentFit {
            tool_calling: Support::Documented(OLLAMA_GEMMA4),
            agentic_coding: Support::Unverified,
            structured_output: Support::Unverified,
            min_context_k: Some(256),
            notes: "Ollama tools badge and 'native function-calling support' on the library page",
        },
        sources: &[
            "https://registry.ollama.ai/library/gemma4",
            "https://huggingface.co/google/gemma-4-12B",
            "https://ai.google.dev/gemma/docs/core/model_card_4",
        ],
        checked: CHECKED,
    },
    LightModel {
        id: "gemma4-26b-a4b",
        name: "Gemma 4 26B A4B (MoE)",
        vendor: "Google",
        released: "not stated (model card updated 2026-07-30)",
        params_total: "25.2B",
        params_active: Some("3.8B"),
        size_lo_gb: 16.0,
        size_gb: 19.0,
        quant: "Ollama default tags (page prints 16-19 GB)",
        context: "256K",
        modalities: "text, image",
        licence: "Apache 2.0",
        permissive: true,
        strength: "MoE: runs at small-model speed, needs 32 GB comfortably",
        ollama_tag: Some("gemma4:26b"),
        hf_repo: None,
        engine: Engine::OpenAiServer,
        documented_ram_gb: None,
        note: "Range printed by Ollama without quantisation names; MoE, all weights must be resident.",
        agent: AgentFit {
            tool_calling: Support::Documented(OLLAMA_GEMMA4),
            agentic_coding: Support::Unverified,
            structured_output: Support::Unverified,
            min_context_k: Some(256),
            notes: "Ollama tools badge and 'native function-calling support' on the library page",
        },
        sources: &[
            "https://registry.ollama.ai/library/gemma4",
            "https://ai.google.dev/gemma/docs/core/model_card_4",
        ],
        checked: CHECKED,
    },
    LightModel {
        id: "qwen3.5-0.8b",
        name: "Qwen3.5 0.8B",
        vendor: "Alibaba Qwen",
        released: "Feb/Mar 2026 (HF card updated Mar 9)",
        params_total: "0.8B",
        params_active: None,
        size_lo_gb: 1.2,
        size_gb: 1.3,
        quant: "Ollama default tags (page prints 1.2-1.3 GB)",
        context: "256K (262,144)",
        modalities: "text, image",
        licence: "apache-2.0",
        permissive: true,
        strength: "Tiny; fits 4 GB machines, simple tasks and drafts",
        ollama_tag: Some("qwen3.5:0.8b"),
        hf_repo: Some("Qwen/Qwen3.5-0.8B"),
        engine: Engine::OpenAiServer,
        documented_ram_gb: None,
        note: "Range printed by Ollama without quantisation names.",
        agent: AgentFit {
            tool_calling: Support::Documented("https://ollama.com/library/qwen3.5"),
            agentic_coding: Support::Unverified,
            structured_output: Support::Unverified,
            min_context_k: Some(256),
            notes: "Ollama tools badge for the Qwen3.5 family; a 0.8B model will call tools unreliably",
        },
        sources: &[
            "https://ollama.com/library/qwen3.5",
            "https://huggingface.co/Qwen/Qwen3.5-0.8B",
        ],
        checked: CHECKED,
    },
    LightModel {
        id: "gpt-oss-20b",
        name: "gpt-oss 20B",
        vendor: "OpenAI",
        released: "not stated on the Ollama page",
        params_total: "20B",
        params_active: None,
        size_lo_gb: 14.0,
        size_gb: 14.0,
        quant: "MXFP4",
        context: "not stated on the Ollama page",
        modalities: "text",
        licence: "Apache License Version 2.0",
        permissive: true,
        strength: "Page says it runs in as little as 16 GB of memory",
        ollama_tag: Some("gpt-oss:20b"),
        hf_repo: None,
        engine: Engine::OpenAiServer,
        documented_ram_gb: None,
        note: "Single figure; MXFP4 is the only published quantisation on the page.",
        agent: AgentFit {
            tool_calling: Support::Documented("https://ollama.com/library/gpt-oss:20b"),
            agentic_coding: Support::Unverified,
            structured_output: Support::Unverified,
            min_context_k: None,
            notes: "Ollama tools badge; page describes the family as designed for agentic tasks",
        },
        sources: &["https://ollama.com/library/gpt-oss:20b"],
        checked: CHECKED,
    },
    LightModel {
        id: "lfm3-1.7b",
        name: "LFM3 1.7B",
        vendor: "Liquid AI",
        released: "27 Oct 2026 (vendor blog, via roundup)",
        params_total: "1.73B",
        params_active: None,
        size_lo_gb: 1.0,
        size_gb: 1.0,
        quant: "about 1 GB at 4-bit (HF card)",
        context: "256K",
        modalities: "text",
        licence: "LFM Open License v1.0",
        permissive: false,
        strength: "Very small and recent; commercial use limited by revenue",
        ollama_tag: None,
        hf_repo: Some("LiquidAI/LFM3-1.7B-GGUF"),
        engine: Engine::OpenAiServer,
        documented_ram_gb: None,
        note: "Non-standard licence (revenue-limited commercial use): ask before shipping in a product.",
        agent: AGENT_UNKNOWN,
        sources: &[
            "https://huggingface.co/LiquidAI/LFM3-1.7B",
            "https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF",
        ],
        checked: CHECKED,
    },
    LightModel {
        id: "qwen3.5-4b",
        name: "Qwen3.5 4B",
        vendor: "Alibaba Qwen",
        released: "Mar 2026 (Qwen3.5 small series; HF cards updated Mar 9)",
        params_total: "4B (not stated on the page)",
        params_active: None,
        size_lo_gb: 3.3,
        size_gb: 4.0,
        quant: "Ollama default tags (page prints 3.3-4.0 GB)",
        context: "256K",
        modalities: "text, image",
        licence: "apache-2.0",
        permissive: true,
        strength: "Best fit for 8 GB machines in the Qwen3.5 small series",
        ollama_tag: Some("qwen3.5:4b"),
        hf_repo: Some("Qwen/Qwen3.5-4B"),
        engine: Engine::OpenAiServer,
        documented_ram_gb: None,
        note: "HF card shows only a relative commit age; the month comes from the 0.8B card of the same series. 2B and 9B licences were not readable (see UNVERIFIED).",
        agent: AgentFit {
            tool_calling: Support::Documented("https://ollama.com/library/qwen3.5"),
            agentic_coding: Support::Unverified,
            structured_output: Support::Unverified,
            min_context_k: Some(256),
            notes: "Ollama tools badge for the Qwen3.5 family",
        },
        sources: &[
            "https://ollama.com/library/qwen3.5",
            "https://huggingface.co/Qwen/Qwen3.5-4B/tree/main",
        ],
        checked: CHECKED,
    },
    LightModel {
        id: "lfm2.5-2.6b",
        name: "LFM2.5 2.6B",
        vendor: "Liquid AI",
        released: "4 Aug 2026 (vendor blog)",
        params_total: "2.6B (HF listing shows 3B)",
        params_active: None,
        size_lo_gb: 1.67,
        size_gb: 1.67,
        quant: "GGUF Q4_K_M (Q8_0 is 2.87 GB)",
        context: "not stated on the pages read",
        modalities: "text",
        licence: "lfm1.0",
        permissive: false,
        strength: "Vendor reports CPU inference under 2.5 GB",
        ollama_tag: None,
        hf_repo: Some("LiquidAI/LFM2.5-2.6B-GGUF"),
        engine: Engine::OpenAiServer,
        documented_ram_gb: None,
        note: "Non-standard licence. Only entry with a size printed for a named quantisation.",
        agent: AgentFit {
            notes: "vendor blog is titled 'Deploy Agents Everywhere'; no tool-calling statement was read",
            ..AGENT_UNKNOWN
        },
        sources: &[
            "https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF",
            "https://www.liquid.ai/en/blog/lfm2-5-2-6b",
        ],
        checked: CHECKED,
    },
    LightModel {
        id: "olmoe-1b-7b",
        name: "OLMoE 1B-7B (0924)",
        vendor: "Allen AI",
        released: "Sep 2024 (from the 0924 name; legacy)",
        params_total: "6.9B",
        params_active: Some("1.3B"),
        size_lo_gb: 7.0,
        size_gb: 7.0,
        quant: "int8 after conversion (README: 14 GB download, 7 GB converted)",
        context: "not stated on the pages read",
        modalities: "text",
        licence: "Apache-2.0",
        permissive: true,
        strength: "Old but documented: 22-23 tok/s on CPU, good for learning the tools",
        ollama_tag: None,
        hf_repo: Some("allenai/OLMoE-1B-7B-0924"),
        engine: Engine::Colibri,
        documented_ram_gb: Some(8),
        note: "colibri README line 233: 8 GB RAM, Vulkan, 22-23 tok/s; needs the conversion step described there. A GGUF repo also exists (allenai/OLMoE-1B-7B-0924-GGUF).",
        agent: AGENT_UNKNOWN,
        sources: &[
            "https://github.com/allenai/OLMoE",
            "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct/blob/refs%2Fpr%2F3/README.md",
            "colibri README line 233",
        ],
        checked: CHECKED,
    },
];

/// Candidates NOT in the list, with the reason. Never shown as recommendations.
pub const UNVERIFIED: &[(&str, &str)] = &[
    (
        "qwen3.5-2b / 9b",
        "Ollama sizes read (2.7-3.1 / 6.6-7.6 GB) but the HF card licence could not be fetched; 4B and 0.8B are Apache-2.0",
    ),
    (
        "granite-4.2 3B/8B/30B",
        "Apache 2.0 and 25 Aug 2026 from IBM research; no context length, no GB size (Ollama page not fetchable)",
    ),
    (
        "mimo-v2.6-distill-qwen-9b",
        "original XiaomiMiMo HF card not readable; MIT/32K/21 Sep 2026 only from a mirror",
    ),
    (
        "mellum2-12b-a2.5b",
        "HF cards: Apache 2.0, 131,072 ctx, 12B MoE, but no GGUF size and no year; no page mentions 'Mellum2.1'",
    ),
    (
        "nemotron-3.5-bolt-e4b",
        "GPU-vLLM only (BF16 8B, OpenMDW-1.1); no GGUF/Ollama path, page inconsistent: excluded from for_ram",
    ),
    (
        "qwen3.8 <27B",
        "Qwen publishes none (github.com/QwenLM/Qwen3.8 lists only 27B and 2.4T-A95B); empero-ai Qwen3.8-4B is a third-party distill, unread",
    ),
    (
        "flint-40b-a2.5b",
        "colibri README line 252: Apache 2.0, UD-IQ1_M 11.4 GB to BF16 79.6 GB; release date not read",
    ),
    (
        "lfm3-vl-2b",
        "colibri README line 253: LFM Open License; sizes and date not read",
    ),
    (
        "plamo-3-translate 0.8B/2B/8B",
        "translation-only; sizes from a roundup; PLaMo Community License (non-permissive)",
    ),
    (
        "phi-4-mini, granite-4.1, bonsai-2-27b",
        "legacy, secondary-only or custom runtime; not re-read",
    ),
    (
        "gemma4 QAT tags",
        "no QAT tag appears on the Ollama page; not listed",
    ),
];

/// Smallest tier (GB) that holds `need_gb`, or `None` above the largest tier.
pub fn tier_for(need_gb: f64) -> Option<u32> {
    TIERS.iter().copied().find(|t| f64::from(*t) >= need_gb)
}

/// Minimum RAM tier: size plus 2 GB for runtime and OS.
pub fn min_tier(m: &LightModel) -> Option<u32> {
    match m.documented_ram_gb {
        Some(r) => tier_for(f64::from(r)),
        None => tier_for(m.size_gb + 2.0),
    }
}

/// Comfortable RAM tier: 1.5x size plus 2 GB, leaving room for context.
pub fn comfortable_tier(m: &LightModel) -> Option<u32> {
    tier_for(m.size_gb * 1.5 + 2.0)
}

pub fn comfortable_for(m: &LightModel, ram_gb: u32) -> bool {
    m.engine != Engine::GpuVllmOnly && comfortable_tier(m).is_some_and(|t| t <= ram_gb)
}

/// Models that fit comfortably in `ram_gb`, larger download first, ties by id.
/// Release dates are free text on primary pages (often absent), so "newer
/// first" is not applied. This is a heuristic, not a quality benchmark.
pub fn for_ram(ram_gb: u32) -> Vec<&'static LightModel> {
    let mut v: Vec<&LightModel> = LIGHT_MODELS
        .iter()
        .filter(|m| comfortable_for(m, ram_gb))
        .collect();
    v.sort_by(|a, b| {
        b.size_gb
            .partial_cmp(&a.size_gb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.id.cmp(b.id))
    });
    v
}

/// Models documented for tool calling that fit comfortably in `ram_gb` (same
/// fit rule and ordering as [`for_ram`]). A documented badge is not a promise
/// that a small model calls tools reliably.
pub fn for_agents(ram_gb: u32) -> Vec<&'static LightModel> {
    let mut v = for_ram(ram_gb);
    v.retain(|m| matches!(m.agent.tool_calling, Support::Documented(_)));
    v
}

fn support_word(s: Support) -> &'static str {
    match s {
        Support::Documented(_) => "documented",
        Support::Unverified => "unverified",
        Support::No => "no",
    }
}

/// One line for the UI: what is documented, what is not.
pub fn agent_note(m: &LightModel) -> String {
    format!(
        "tools: {}; agentic coding: {}; structured output: {}. {}",
        support_word(m.agent.tool_calling),
        support_word(m.agent.agentic_coding),
        support_word(m.agent.structured_output),
        m.agent.notes
    )
}

fn shell_safe(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | ':' | '/'))
}

/// `ollama run <tag>` (the form printed on the Ollama library pages).
pub fn ollama_command(m: &LightModel) -> Option<String> {
    let tag = m.ollama_tag?;
    shell_safe(tag).then(|| format!("ollama run {tag}"))
}

/// No llama.cpp command was read on a primary page for any entry, so none is
/// emitted; the Hugging Face repo is exposed in `hf_repo` instead.
pub fn llama_cpp_command(_m: &LightModel) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_complete() {
        for m in LIGHT_MODELS {
            assert!(!m.sources.is_empty(), "{}", m.id);
            assert!(m.sources.iter().any(|s| s.starts_with("https://")));
            assert_eq!(m.checked, CHECKED);
            assert!(!m.licence.is_empty());
            assert!(
                m.size_lo_gb <= m.size_gb && m.size_gb > 0.0 && m.size_gb < 32.0,
                "{}",
                m.id
            );
            assert!(m.ollama_tag.is_some() || m.hf_repo.is_some(), "{}", m.id);
        }
    }

    #[test]
    fn documented_support_has_source_and_for_agents_is_consistent() {
        for m in LIGHT_MODELS {
            for s in [
                m.agent.tool_calling,
                m.agent.agentic_coding,
                m.agent.structured_output,
            ] {
                if let Support::Documented(u) = s {
                    assert!(u.starts_with("https://"), "{}", m.id);
                }
            }
        }
        for m in for_agents(32) {
            assert!(matches!(m.agent.tool_calling, Support::Documented(_)));
            assert!(m.size_gb > 0.0 && !m.licence.is_empty());
            assert!(!agent_note(m).is_empty());
        }
        assert!(for_agents(32).iter().any(|m| m.id == "gpt-oss-20b"));
        assert!(for_agents(4).iter().all(|m| m.id != "gpt-oss-20b"));
    }

    #[test]
    fn ids_unique() {
        let mut ids: Vec<_> = LIGHT_MODELS.iter().map(|m| m.id).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(n, ids.len());
    }

    #[test]
    fn tiers_monotone_in_size() {
        let mut v: Vec<_> = LIGHT_MODELS.iter().collect();
        v.sort_by(|a, b| a.size_gb.partial_cmp(&b.size_gb).unwrap());
        for w in v.windows(2) {
            assert!(min_tier(w[0]) <= min_tier(w[1]));
            assert!(comfortable_tier(w[0]) <= comfortable_tier(w[1]));
        }
    }

    #[test]
    fn commands_are_shell_safe() {
        for m in LIGHT_MODELS {
            if let Some(c) = ollama_command(m) {
                assert!(c.starts_with("ollama run "));
                assert!(shell_safe(&c["ollama run ".len()..]));
            }
        }
    }

    #[test]
    fn for_ram_respects_comfort() {
        for m in for_ram(8) {
            assert_ne!(m.engine, Engine::GpuVllmOnly);
            assert!(comfortable_tier(m).unwrap() <= 8);
        }
        assert!(for_ram(4).iter().any(|m| m.id == "qwen3.5-0.8b"));
        assert!(for_ram(2).is_empty());
    }
}
