# Clean-room process

[English](CLEANROOM.md) · [Română](CLEANROOM.ro.md) · [Français](CLEANROOM.fr.md) · [Italiano](CLEANROOM.it.md) · [Español](CLEANROOM.es.md) · [Português](CLEANROOM.pt.md) · [Deutsch](CLEANROOM.de.md) · [Magyar](CLEANROOM.hu.md)

A short, honest note on how the engine in `crates/engine` was written and what "clean-room" does and does not mean here.

## The rules we followed

1. **Sources.** Only public specifications and papers: the GGUF and safetensors file layouts, Hugging Face `config.json` and `tokenizer.json` fields, the published descriptions of block-quantisation layouts, and the papers behind each technique (attention variants, MoE routing, Mamba, RWKV, speculative decoding, H2O, StreamingLLM, SnapKV, TIES, DARE, green-list watermarking, HNSW, BM25…).
2. **No peeking.** The source code of colibri, llama.cpp and other inference engines was not read or copied while writing the engine. When a layout was unclear, the work stopped at the public description and the uncertainty is written in the code and in the docs.
3. **Oracles instead of imitation.** Every fast kernel is tested against a plain scalar version written first; every quantiser against hand-computed blocks and round trips; split paths (mesh, streaming, caches) against the single-machine path, bit for bit where that is promised.
4. **Permissive licences only.** The engine depends on `serde_json` and `memmap2` and nothing else; the GPU crate adds `wgpu` behind a feature. Licences were checked with `cargo-deny` at an earlier commit; later changes have not been re-checked yet.
5. **Hostile input is normal.** Every size or offset read from a file is checked against the real file length before anything is allocated; fuzz-style tests feed truncated and corrupt files and must get an error, not a crash.

## What this does not claim

- Clean-room is a process, not a legal guarantee. The model writing the code, like any model, was trained on public text that may include open-source code, and we cannot rule out that it has seen the projects above. We did not consult them, and we tried to make that visible by keeping oracles, tests and sources of each formula in the code.
- Several formulas (Mamba, RWKV-6/7, MLA, imatrix and GGUF extras) were written from the published descriptions and memory of them, and have been checked only for internal consistency, because no real checkpoint was available. They may differ from the reference in details; treat them as unverified until someone runs them on real models.
- Compatibility with other engines' files is a goal, not a verified fact (see the table in [ENGINE.md](ENGINE.md)).

If you find a place where the engine reproduces someone else's code too closely, please open an issue; we will rewrite it.
