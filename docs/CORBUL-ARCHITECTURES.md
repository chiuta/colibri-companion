# Models corbul runs that colibri does not

Companion starts **corbul**, its own engine, for every GGUF that colibri cannot import. Since alpha.18 it goes there **directly** for the families below, without a failed import first.

## What corbul runs
- **Gemma 4** (E2B, E4B, 26B-A4B mixture-of-experts, 31B), from a `.gguf`: mixed sliding and global attention, per-layer head sizes, keys used as values on global layers, proportional RoPE, shared K/V in the last layers, per-layer embeddings, the dense MLP next to the routed experts, final logit soft-capping.
- **GPT-2, GPT-J, GPT-NeoX/Pythia, Falcon, BLOOM, MPT**, from a `.gguf`: LayerNorm, fused or separate QKV, learned / rotary / ALiBi positions, parallel residual. Tested on small synthetic models only. OPT has no GGUF conversion.
- **Gemma 1–3, Phi, OLMo, Granite, Cohere, InternLM2, StableLM, StarCoder2, EXAONE, Baichuan, MiniCPM, SmolLM3**, Llama and Mistral, Qwen and the other mixture-of-experts families the engine already had.
- **Mamba, Mamba-2, RWKV-6/7, Jamba, Bamba** (recurrent and hybrid).

`corbul serve MODEL` answers `/health`, `/v1/models`, `/v1/chat/completions` and `/v1/completions` (also streaming) with the model's own chat template, `stop`, `max_tokens`, `temperature`, `top_p`, `top_k` and `seed`. A conversation that continues the previous one does not read its history again.

## What it does not do for these models
Tool calls, structured output, several answers per request, embeddings, context packs (Companion then uses the ordinary chat), image input, several models at once. Gemma 4 is read from `.gguf` only, not from safetensors folders. A GGUF without its own vocabulary can use a `tokenizer.json` placed next to it.

## What is not checked
All of it is tested on **small synthetic models** (their own invariants: caching equals recomputing, sliding windows, shared K/V, routing, quantised experts, the HTTP replies). **Nothing was compared with a real Gemma 4 file.** If the answers of a real file look wrong, run `corbul serve FILE.gguf` yourself and send its output; the first thing to check is the sliding/global pattern and the RoPE tables.

## Still not supported
OPT, gpt-oss, PhiMoE, DBRX, Llama 4, GLM-4-MoE, Hunyuan, Ernie-MoE, Grok, the full DeepSeek model, MiniMax, Zamba, encoder-decoder models. For them corbul says so by name and does not guess.
