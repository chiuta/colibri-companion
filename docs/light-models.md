# Light local models (checked 2026-11-17)

"Light" only means the download fits comfortably in 4/8/16/32 GB of RAM. It says
nothing about quality; no ordering here is a benchmark. Source of truth:
`crates/core/src/light_models.rs`. All entries run through any OpenAI-compatible
server (Ollama, llama.cpp, LM Studio) set in `server_url`; none is in the colibri
catalogue.

RAM tiers: minimum = size + 2 GB, comfortable = 1.5 x size + 2 GB, rounded up to
4/8/16/32.

| id | size GB (low-high printed) | context | licence | min / comfortable RAM |
|---|---|---|---|---|
| lfm3-1.7b | ~1.0 (4-bit) | 256K | LFM Open License v1.0 (**non-standard**) | 4 / 4 |
| qwen3.5-0.8b | 1.2-1.3 | 256K | apache-2.0 | 4 / 4 |
| lfm2.5-2.6b | 1.67 (Q4_K_M) | 131,072 | lfm1.0 (**non-standard**) | 4 / 8 |
| qwen3.5-4b | 3.3-4.0 | 256K | apache-2.0 | 8 / 8 |
| qwen3.5-9b | 6.6-7.6 | 256K | apache-2.0 | 16 / 16 |
| olmoe-1b-7b (colibri, legacy 2024) | 7 (int8 converted) | n/s | Apache-2.0 | 8 (README) / 16 |
| gemma4-e2b | 4.6-7.5 | 128K | Apache 2.0 | 16 / 16 |
| gemma4-12b | 7.7-8.0 | 256K | Apache 2.0 | 16 / 16 |
| gemma4-e4b | 6.6-9.5 | 128K | Apache 2.0 | 16 / 16 |
| gpt-oss-20b | 14 (MXFP4) | n/s | Apache 2.0 | 16 / 32 |
| gemma4-26b-a4b | 16-19 | 256K | Apache 2.0 | 32 / 32 |

Ollama pages print a size range per tag and never name the default quantisation,
so tiers use the high end. Only LFM2.5 (Q4_K_M), gpt-oss (MXFP4) and OLMoE (README)
have a size tied to one named format. Gemma 4 E2B's 7.5 GB equals the separate
`e2b-mlx` tag; the page does not say which variant is which.
Nemotron 3.5 Bolt E4B is GPU-vLLM only and never offered by `for_ram`.

## Not verified (not in the list)
- Qwen3.5 2B: Ollama size read, HF licence not readable (0.8B, 4B, 9B are Apache-2.0).
- Granite 4.2 3B/8B/30B: IBM docs say Apache 2.0, 128K, tool calling; date 25 Aug 2026 only from secondary pages; no GB size.
- MiMo-V2.6-Distill-Qwen-9B: original HF card not readable; mirror only.
- Mellum2 12B-A2.5B: Apache 2.0, 131,072 ctx, but no GGUF size or year; no page mentions "Mellum2.1".
- Nemotron 3.5 Bolt E4B: GPU-vLLM only, excluded.
- Qwen3.8: Qwen publishes no model under 27B; empero-ai Qwen3.8-4B is a third-party distill.
- Flint 40B-A2.5B and LFM3-VL-2B (colibri README lines 252-253), PLaMo 3 Translate,
  Phi-4-mini, Granite 4.1, Bonsai 2; Gemma 4 QAT tags do not appear on the Ollama page.
- A search for Sept-Nov 2026 releases returned only third-party ranking pages, none read as primary.
- Not light: Ripple-65B, Kimi K3.1 Pro, ERNIE-5.5-Flash-Base, Qwen3.5 122B, PLaMo 3 31B.

## Sources
Ollama: gemma4, qwen3.5, gpt-oss:20b library pages. Google Gemma 4 model card
(ai.google.dev) and huggingface.co/google/gemma-4-12B. huggingface.co/Qwen/Qwen3.5-0.8B.
huggingface.co/LiquidAI/LFM3-1.7B (+ -GGUF). Not found on any page: Gemma 4
release month, gpt-oss context length, llama.cpp commands (none emitted).

## Agents (tool calling, agentic coding, long context)

What Companion needs from a model (read in `chat.rs`, `mcp.rs`): MCP tools are
offered as plain text, a fenced `tool` block the model may write and
`mcp::parse_tool_request` parses, so **no model needs the OpenAI `tools` field**.
The model must follow that format, stream over SSE with `include_usage`, may emit
`reasoning_content` or `<think>` text (split by `chat.rs`), and needs context for up
to 5 rounds of tool output. `image_url` parts matter only for vision use. A
"tools" badge is therefore a helpful sign, not a requirement.

`for_agents(ram_gb)` returns only models with tool calling documented on a primary
page, using the same fit rule as `for_ram`. Documented sources:

| model | tool calling (source) | agentic coding | structured output |
|---|---|---|---|
| gpt-oss-20b | Ollama `tools` badge; "designed for agentic tasks" | unverified | unverified |
| gemma4 e2b / e4b / 12b / 26b-a4b | Ollama `tools` badge, "native function-calling support" | unverified | unverified |
| qwen3.5 0.8b / 4b | Ollama `tools` badge (family) | unverified | unverified |
| lfm3-1.7b | HF card: "designed for local agents, tool calling" (XML calls) | unverified | HF card: "structured outputs" (unverified as a test) |
| qwen3.5 9b | Ollama `tools` badge | unverified | unverified |
| lfm2.5-2.6b, olmoe-1b-7b | unverified | unverified | unverified |

Not in the agent list (no size or licence on a primary page, or not local):
- Devstral Small 2 24B: Mistral docs say Apache 2.0, 256k, function-calling,
  structured outputs, 2025-12-09, but the API status is Deprecated (retired
  2026-03-31) and no local size is printed.
- Qwen3-Coder-30B-A3B: the Qwen card says tool calling and agentic coding, 262,144
  context, but prints no licence or size; colibri README line 209 lists it.
- Granite 4.2: IBM docs document tool calling and 128K context, no size.
- MiMo distill, Mellum2 (agentic coding claimed only on pages not read),
  Nemotron 3.5 Bolt (vLLM only).

Limits: small models call tools unreliably, and a badge says the chat template
supports tools, not that the model chooses well. Nothing here was run against
Companion's flows.

## Re-check 2026-11-17 (full catalogue)
Re-read today: Ollama gemma4 (all sizes/contexts match), qwen3.5 (sizes, badges; page names no licence), gpt-oss:20b (14 GB, tools, Apache 2.0, no context); HF gemma-4-12B, Qwen3.5-0.8B/4B, LFM3-1.7B, LFM2.5-2.6B; IBM Granite 4.2 docs.
Corrected: LFM2.5 context (131,072, was "not stated"); LFM3 tool calling now documented; Qwen3.5 9B added.
Not re-read: OLMoE (README figures), Gemma 4 E2B/E4B/26B licence (family card read earlier; 12B card read today), LFM3 GGUF size, LFM2.5 Q4_K_M size (HF page links only), release months (no primary page prints them; LFM3's 27 Oct comes from a roundup).
The Ollama gemma4, qwen3.5 and gpt-oss pages print no licence; licences come from the HF/Google cards.
