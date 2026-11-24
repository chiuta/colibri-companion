# The clean-room engine (`companion-engine` and `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Other documents: [INNOVATIONS.md](INNOVATIONS.md) · [QUICKSTART.md](QUICKSTART.md) · [CLEANROOM.md](CLEANROOM.md)

**Status: alpha. Everything below was tested on small synthetic checkpoints only.** No real model from Hugging Face has been run through this engine by its authors (the build sandbox could not reach huggingface.co), no GPU has executed any of it, and the ARM, macOS and Windows code paths have never been compiled here. Read the "verified vs unverified" table before you trust any claim.

## What it is

A local inference engine for language models, written from scratch in Rust, from public file-format descriptions and published papers. It has three parts:

| Crate | What it is |
|---|---|
| `crates/engine` (`companion-engine`) | The library: file formats, tokenizer, quantisation, the forward passes, sampling, and the extra tools described in INNOVATIONS.md. Dependencies: `serde_json` and `memmap2` only. |
| `crates/engine-cli` (`corbul`) | A command line and an OpenAI / Ollama / Anthropic-compatible local server on top of the library. |
| `crates/engine-gpu` (`companion-engine-gpu`) | WGSL kernels for the `wgpu` crate, behind a feature that is off by default. Not used by `corbul`. |

It is separate from, and does not replace, the Colibri Companion desktop app: the app still drives JustVugg's colibri. Wiring this engine into the app as an alternative backend is planned, not done.

## What it supports

**Files.** GGUF v2/v3 (header and tensor table are read without touching the weights, so a 400 GB file is inspected in milliseconds); a streaming GGUF writer for models larger than RAM; safetensors, single file or sharded with an index; Hugging Face `config.json` and `tokenizer.json`; a model downloader (through `curl`, with resume and SHA-256 checks).

**Quantisation.** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and the K-quants Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K can be decoded, and there are encoders for all of them (with importance-matrix weighting), Hugging Face → GGUF conversion and GGUF re-quantisation. Weights stay quantised in memory; the matrix–vector product never materialises the matrix. The i-quants (IQ*) are not implemented.

**Architectures.**

| Family | State |
|---|---|
| Dense decoders: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 and similar (RMS or layer norm, GQA/MQA, RoPE with linear / Llama-3 / YaRN scaling and partial rotary, QK-norm, biases, soft-capping, sliding windows, gated or plain MLP) | Implemented; loaded by `corbul` |
| Mixture of experts: Mixtral, Qwen-MoE, GGUF stacked experts, shared experts, routed experts streamed from disk with an LRU cache | Implemented; loaded by `corbul` |
| Multi-head latent attention (DeepSeek-V2/V3, Kimi-K2 style) | Library module only; the architecture registry still marks it "planned" and `corbul` does not load it |
| Mamba-1, Mamba-2, RWKV-6, RWKV-7 | Library module only; same as above |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, encoder and encoder-decoder models (T5, BERT, Whisper…), LongRoPE | Not supported; `corbul` says so instead of guessing |
| Vision and audio inputs | Not supported (text only) |

**Hardware paths.**

- CPU: runtime-dispatched kernels (scalar, AVX2, AVX-512, VNNI on x86-64; NEON on aarch64), a persistent thread pool that suits mixed performance/efficiency cores, hardware and NUMA detection, and a planner that decides what stays in RAM.
- GPU: WGSL kernels checked with `naga` and against a Rust emulation. They have never run on a GPU.
- Models larger than RAM: routed experts from disk (MoE), layer streaming (dense), a LAN mesh over several machines.

## Verified vs unverified

| Area | State |
|---|---|
| Unit, integration and fuzz tests on synthetic checkpoints (Linux, x86-64) | Verified. Hundreds of tests pass; hostile and truncated files return errors instead of crashing. |
| Real Hugging Face checkpoints (any family) | **Not run.** The sandbox could not reach huggingface.co. |
| Tokenizer on real `tokenizer.json` files | Written from the public format descriptions; a test harness exists (`ENGINE_VOCAB_DIR`) but has not been run on real files by us. |
| llama.cpp-style Q/K row permutation in GGUF files | Implemented from the description; not checked on real files. |
| Quantisers and dequantisers | Checked against hand-computed blocks and round trips; not against files written by other tools. |
| GPU execution | **Not run.** Kernels are validated by `naga` and a Rust emulation only. |
| ARM / NEON, macOS, Windows paths | **Never compiled or run here.** Probes are tested on recorded sample text from those systems. |
| Mamba / RWKV / MLA formulas | Written from papers and from memory of the reference descriptions, tested for internal consistency only. |
| Mesh over a network | Tested on loopback only. |
| Speed | **No reliable numbers.** The build machine had two shared cores; benchmarks were noisy. All speeds in the autopilot are estimates. |
| Reproducible mode | Identical across five x86 kernel variants and several thread counts; ARM not tested. |
| Dependency licences | Checked with `cargo-deny` at an earlier commit (b496ba5); later additions have not been re-checked yet. |

## Clean-room statement

No code from colibri, llama.cpp or any other inference engine was read or copied while writing this engine. The formats and algorithms come from public specifications, papers and file layouts. See [CLEANROOM.md](CLEANROOM.md) for the process and its limits.

## How it compares with other local engines

We ran no head-to-head benchmark, so there are no speed or quality claims here. Qualitatively:

- Established engines such as llama.cpp, Ollama, mistral.rs or vLLM have been run on thousands of real checkpoints by many people, have working GPU back ends, cover more architectures and quantisation formats (including i-quants) and handle vision models. This engine has none of that evidence yet. If you need something that works today on a specific large model, use one of them.
- What this engine tries to do differently: integrity and honesty tools built in (quote-locked citations, reproducible mode with sealed (hash-chained, optionally HMAC-protected) receipts, uncertainty analytics, an autopilot that says "this won't run" instead of crashing), Romanian-first language tools, running models bigger than your RAM or your single machine, a single small Rust code base with two dependencies, and permissive licences only.
- Any claim that this engine is faster or more accurate than another would be untested. Please measure on your own hardware and tell us.

## Library map

Formats and loading: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`. Model: `arch`, `model`, `moe`, `mla`, `ssm`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Runtime: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Distinctive features: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Newest wave: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark` — implemented in the library and covered by tests on synthetic models, still being finished, not yet reachable from `corbul`, and their APIs may change.
