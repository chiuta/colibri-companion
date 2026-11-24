# The clean-room engine (`companion-engine` and `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Other documents: [INNOVATIONS.md](INNOVATIONS.md) · [QUICKSTART.md](QUICKSTART.md) · [CLEANROOM.md](CLEANROOM.md) · [CHANGES-SECURITY.md](CHANGES-SECURITY.md)

**Status: alpha.** Everything below was tested on small synthetic checkpoints, on tokenizer reference files and on recorded sample text. **No real model downloaded from Hugging Face has been run through this engine by its authors**, no GPU has executed any of it, and the ARM, macOS and Windows code paths have not been run here. Read "Verified and unverified" before you trust a claim. `corbul capabilities` prints the same limits from the binary itself.

## What it is

A local inference engine for language models, written from scratch in Rust from public file-format descriptions and published papers.

| Crate | What it is |
|---|---|
| `crates/engine` (`companion-engine`) | The library: file formats, tokenizer, quantisation, forward passes, sampling, and the extra tools in INNOVATIONS.md. Runtime dependencies: `serde_json` and `memmap2` only. |
| `crates/engine-cli` (`corbul`) | The command line and an OpenAI / Ollama / Anthropic-compatible local server with a built-in web interface. Depends on the library and `serde_json`. |
| `crates/engine-gpu` (`companion-engine-gpu`) | WGSL kernels and a CPU emulation of them. The device runner (`wgpu`, `pollster`) sits behind the cargo feature `wgpu`, which is off by default. **`corbul` does not use this crate.** |

It is separate from, and does not replace, the Colibri Companion desktop app, which still drives JustVugg's colibri. Using this engine as an alternative back end of the app is planned, not done.

## What it supports

**Files.** GGUF v2 and v3 (the header and tensor table are read without touching the weights); a streaming GGUF writer for outputs larger than RAM; safetensors, single or sharded with an index; Hugging Face `config.json`, `tokenizer.json` and chat templates; a model downloader (module `hub`, through `curl`, with `wget` as an anonymous fallback; resume, SHA-256 checks, commit-pinned downloads, size ceilings, refusal of redirects to plain http).

**Quantisation.** Decoded: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K (14 types). Encoders exist for the same types, with importance-matrix weighting and per-tensor recipes. Weights stay quantised in memory; the matrix–vector product never expands the whole matrix. **Not supported:** the i-quants (IQ\*), and any type that is not in the list.

**Architectures.**

| Family | State in `corbul` |
|---|---|
| Dense decoders: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 and similar (RMS or layer norm, GQA/MQA, RoPE with linear, Llama-3, YaRN and LongRoPE scaling and partial rotary, QK-norm, biases, soft-capping, sliding windows, gated or plain MLP) | Loaded and run (chat template, serve, all options) |
| Mixture of experts: Mixtral, Qwen-MoE, GGUF stacked experts, shared experts, routed experts streamed from disk with an LRU cache | Loaded and run |
| State-space: Mamba-1, Mamba-2, RWKV-6, RWKV-7 (module `ssm`) | Run by `generate` and `chat` as **plain-text continuation**: no chat template, no structured output, no `serve`. RWKV in GGUF form is not supported (safetensors only) |
| Hybrid: Jamba and Bamba (module `hybrid`) | Same as the state-space row. Zamba and Zamba2 are not supported |
| Multi-head latent attention (DeepSeek-V2/V3, Kimi-K2; module `mla`) | **Library only.** The block exists and is tested for internal consistency, but `corbul` does not assemble such a model (`corbul capabilities` says no) |
| Vision encoder (ViT, CLIP, SigLIP; module `vision`, decoders in `imageio`) and speech front-end plus a Whisper-layout runner (module `audio`) | **Library only.** `corbul` accepts text only |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, encoder and encoder-decoder models (T5, BERT…), WordPiece tokenizers | Not supported; `corbul` says so instead of guessing |
| Embeddings and reranking endpoints | Answer 501 (not implemented) |

**Tokenizer.** Byte-level BPE and SentencePiece-style models, pre-tokenizer rules for 12 families, added tokens, byte fallback, templates from `tokenizer_config.json` and GGUF metadata. WordPiece/BERT is not supported.

**Sampling.** Temperature, top-k, top-p, min-p, typical-p, repetition / presence / frequency penalties, Mirostat 1 and 2, DRY, XTC, a JSON sampler spec, seeded randomness; constrained decoding by JSON, JSON Schema, regular expression or GBNF grammar (with work budgets so a hostile grammar cannot hang the process); logit bias and stop strings through the server. Speculative decoding: draft model, prompt lookup, self-speculation (see INNOVATIONS.md).

**Server.** `corbul serve` offers `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/messages` and `/v1/messages/count_tokens` (Anthropic style), the Ollama endpoints `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/create`, `/api/delete`, `/api/ps`, `/api/version`, model management `/models`, `/models/load`, `/models/unload`, plus `/metrics`, `/props`, `/slots`, `/tokenize`, `/detokenize`, `/apply-template`, `/health`, `/v1/capabilities`, `/v1/quotes/verify` and the web interface at `/` and `/ui` (eight languages). Tool calling is supported through the chat templates. Embedding and rerank routes (`/v1/embeddings`, `/v1/rerank`, `/api/embed`, `/api/embeddings`) answer 501. Streaming responses carry no quote-citation block.

**Hardware.** CPU kernels are chosen at run time: scalar, avx2+fma, avx2+fma+avxvnni, avx512, avx512+vnni (x86-64), neon and neon+dotprod (aarch64); `ENGINE_SIMD` forces one. A persistent thread pool, hardware and NUMA detection, and a planner (`corbul plan`, `corbul doctor`) decide what stays in RAM. The GPU kernels have never run on a GPU (see below).

## Exactness

- **Dense f32 weights:** batched and single-token paths, the mesh, layer streaming and the caches give bit-identical logits.
- **Quantised weights:** bit-identity between batched and single-token evaluation holds only when `exact_batch` is on. Verification, self-speculative verification rounds and the batching executor switch it on. Ordinary prefill uses a faster dequantise-then-tile path and differs from single-token evaluation by rounding noise only.
- **Reproducible mode** (`--reproducible` or `ENGINE_REPRODUCIBLE=1`): identical results across the x86 kernel variants and thread counts that we could test. ARM was not tested. It costs speed.
- Mirostat 1 uses the number of remaining candidates, not the vocabulary size, in its estimate (documented deviation).

## Verified and unverified

| Area | State |
|---|---|
| Unit, integration, adversarial and fuzz tests on synthetic checkpoints (Linux x86-64). The counts change with every commit; run `cargo test --workspace` for the current number | Passing at the last full run. Hostile and truncated files return errors. |
| Tokenizer against llama.cpp reference files, 12 families | 46 of 46 cases equal (reference files fetched from GitHub; `crates/engine/tests/tokenizer_real.rs`, `ENGINE_VOCAB_DIR`). Only the tokenizer, not whole models. |
| Real Hugging Face model through `corbul` (any family) | **Not run by the authors.** The build sandbox could not reach huggingface.co. A manual CI job runs a tiny llama GGUF (`stories260K`) through the CLI; it has not run yet |
| llama.cpp-style Q/K row permutation, GGUF metadata variants | Implemented from public descriptions; not checked on real files |
| Quantisers and dequantisers | Hand-computed blocks and round trips; not compared with files written by other tools |
| GPU execution | **Not run.** Kernels are validated by `naga` and by a Rust emulation only; `corbul` does not use them |
| ARM / NEON, macOS, Windows | **Not run here.** Probes are tested on recorded sample text. The CI workflow has a matrix for them but has never run |
| Mamba, RWKV, Jamba/Bamba, MLA, vision, audio | Internal-consistency tests only; no real checkpoint was used |
| Mesh | Loopback only; authenticated, not encrypted; no speed measurement |
| Speed | **No reliable numbers.** The build machine had two shared cores. Autopilot speeds are estimates |
| Dependency licences | Checked with `cargo-deny` at commit b496ba5; later changes have not been re-checked |
| Translations of the documents | Machine-assisted; no native review yet |

## Environment variables

| Variable | Effect |
|---|---|
| `COLIBRI_THREADS` | Size of the global thread pool (default: usable cores). Read first by the library |
| `ENGINE_SIMD` | Force a kernel variant by name (benchmarks, bug reports) |
| `ENGINE_REPRODUCIBLE` | `1`, `true` or `on`: reproducible mode from the start |
| `ENGINE_BACKEND` | `cpu`, `gpu` or `auto` (unset = auto). `corbul` runs on the CPU either way |
| `ENGINE_GPU`, `ENGINE_GPU_API`, `ENGINE_GPU_SOFTWARE` | Adapter selection for the GPU crate: index, graphics API, allow a software adapter. Not used by `corbul` |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Access token for gated repositories (`download`, `inspect owner/repo`) |
| `HF_ENDPOINT` | Mirror of the Hugging Face host |
| `CORBUL_TOKEN` | Shared bearer token of `corbul serve` (same as `--token`) |
| `CORBUL_MESH_KEY` | Mesh key for `worker` and `--workers`/`--discover` (same as `--key`) |
| `CORBUL_CONFIG_DIR` | Where Modelfiles created by `POST /api/create` are kept (same as `--config-dir`; default `~/.config/corbul`) |
| `NUMBER_OF_PROCESSORS`, `PROCESSOR_IDENTIFIER` | Read on Windows by the hardware probe; you do not set them |
| `ENGINE_VOCAB_DIR`, `ENGINE_VOCAB_ONLY`, `ENGINE_REQUIRE_VOCAB`, `ENGINE_REQUIRE_TOOLS`, `ENGINE_PRINT_GOLDEN`, `ADVERSARIAL_N` | Test-suite controls: vocabulary folder, restrict families, make a skipped test fail, make missing `curl`/`wget` fail, regenerate golden hashes, size of the adversarial run |

## Clean-room statement

No code from colibri, llama.cpp or any other inference engine was read or copied while writing this engine. The formats and algorithms come from public specifications, papers and file layouts. The model that wrote the code was trained on public text that may include such projects; see [CLEANROOM.md](CLEANROOM.md) for the process and its limits.

## How it compares with other local engines

We ran no head-to-head benchmark, so there are no speed or quality claims here.

- Established engines (llama.cpp, Ollama, mistral.rs, vLLM) have been run on thousands of real checkpoints, have working GPU back ends, cover more architectures and quantisation formats including i-quants, and handle vision models. This engine has none of that evidence yet. If you need something that works today on a specific large model, use one of them.
- What this engine tries to do differently: integrity tools built in (quote-locked citations, reproducible mode with receipts, uncertainty analytics, a planner that says "this won't run" instead of crashing), Romanian-first language tools, models bigger than your RAM or your machine, a small Rust code base with two library dependencies, and permissive licences only.

## Library map

Formats and loading: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`, `imageio`. Models: `arch`, `model`, `moe`, `mla`, `ssm`, `hybrid`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Runtime: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Distinctive features: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Test-time compute, tuning, retrieval: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark`. Multimodal: `vision`, `audio`. Which of these `corbul` reaches is listed in INNOVATIONS.md; library APIs may still change.
