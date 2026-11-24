# Clean-room process

[English](CLEANROOM.md) · [Română](CLEANROOM.ro.md) · [Français](CLEANROOM.fr.md) · [Italiano](CLEANROOM.it.md) · [Español](CLEANROOM.es.md) · [Português](CLEANROOM.pt.md) · [Deutsch](CLEANROOM.de.md) · [Magyar](CLEANROOM.hu.md)

A short, honest note on how the engine in `crates/engine` and `crates/engine-cli` was written, and what "clean-room" does and does not mean here. It is a statement of process, not a legal opinion.

## The rules we followed

1. **Sources.** Public specifications and papers: the GGUF and safetensors file layouts, Hugging Face `config.json` and `tokenizer.json` fields, published descriptions of block-quantisation layouts, the format standards behind the image and audio decoders (RFC 1950/1951, the PNG specification, ITU T.81, the BMP and Netpbm descriptions, RIFF/WAVE), and the papers behind each technique (attention variants, MoE routing, Mamba, RWKV, speculative decoding, H2O, StreamingLLM, SnapKV, TIES, DARE, green-list watermarking, HNSW, BM25, ViT, CLIP, SigLIP, Whisper…).
2. **No copying, no consulting of source code.** The source code of colibri, llama.cpp and other inference engines was not opened or copied while writing the engine. When a layout was unclear, work stopped at the public description and the uncertainty was written into the code and the docs.
3. **Oracles instead of imitation.** Every fast kernel is tested against a plain scalar version written first; every quantiser against hand-computed blocks and round trips; split paths (mesh, streaming, caches) against the single-machine path, bit for bit where the docs promise it. For the tokenizer we also compare against **reference data files** (vocabulary files and expected token sequences) published in the llama.cpp repository and fetched as plain data: these are test inputs and expected outputs, not code, and the result (46 of 46 cases, 12 families) says nothing about whole-model behaviour.
4. **Permissive licences only.** The library depends on `serde_json` and `memmap2` at run time. The command line adds nothing beyond the library and `serde_json`. The GPU crate adds `wgpu` and `pollster` behind the `wgpu` feature (off by default) and `naga` for tests. Licences were checked with `cargo-deny` at commit b496ba5; later changes have not been re-checked yet, so the claim "permissive only" is verified for that commit and expected, not proven, for the current tree.
5. **Hostile input is normal.** Sizes and offsets read from files are checked before anything is allocated, against whichever bound applies: the real file length where the format allows it (GGUF parsing takes the file length), a documented ceiling otherwise (image `Limits`, download ceilings, grammar and quote-lock work budgets, tokenizer piece caps, receipt and handshake caps). Fuzz-style and adversarial tests feed truncated and corrupt files and must get an error, not a crash. This lowers the risk; it is not a proof that no input can panic.

## What this does not claim

- Clean-room is a process, not a guarantee. The model that wrote the code, like any model, was trained on public text that may include open-source code, and we cannot rule out that it has seen the projects above. We did not consult them, and we tried to make the process visible by keeping oracles, tests and the source of each formula in the code.
- Several formulas (Mamba, RWKV-6/7, MLA, Jamba/Bamba, imatrix and GGUF extras, vision and audio blocks) were written from published descriptions and memory of them and checked only for internal consistency, because no real checkpoint was available. They may differ from the reference in details; treat them as unverified until someone runs them on real models.
- Compatibility with other engines' files is a goal, not a verified fact (see "Verified and unverified" in [ENGINE.md](ENGINE.md)).
- The statement covers the code, not the models you run: models have their own licences.

If you find a place where the engine reproduces someone else's code too closely, please open an issue; we will rewrite it.
