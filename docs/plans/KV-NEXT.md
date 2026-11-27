# Status of the cache work, and what is left

Implemented: exact re-rotation splice, adaptive repair, the pack shelf (F32/F16, optional lossless compression), eviction (`kvpolicy.rs`: StreamingLLM, H2O, SnapKV), query recording and value refit (`kvfit.rs`, `corbul kvfit`), **shrunk packs** (`corbul packs --shrink`, `shrink` in `/v1/packs/complete`, a checkbox in the Companion) and **cache maps between models** (`kvmap.rs`, `corbul map-fit`, `corbul map-check`, `ctxpack::compose_mapped`). User-facing description: `docs/SHRINK-MAP.md` and `docs/KV-FIT.md`.

How the two newest pieces work, in short:
- A shrunk pack is squeezed (`Model::squeeze_cache`): kept keys are rotated to their new slots, so the state *is* the cache of the shortened text. Normal splice and repair apply; with full repair the answer equals a plain prefill of the shortened text.
- A cache map is a per-layer ridge-regression map from the position-free key/value rows of one model to another's, fitted on paired rows for identical tokens. Imported rows are approximate and go through the same repair, so a poor map costs time only.

## Not measured, and why it matters

Everything was tested on tiny random models, where attention is near-uniform and caches of different models are unrelated. These tests show the code is correct (exactness with full repair, damaged files refused, bad settings rejected). They say nothing about quality or speed on real models. Needed, on real GGUF models:
1. `corbul kvfit` and `corbul packs --shrink` at 50 %, 25 % and 10 %: first-token agreement and answer quality against a full prefill.
2. `corbul map-fit` and `corbul map-check` on a real pair (same family and tokenizer, e.g. a model and its larger sibling, or two quantisations/fine-tunes of one model): error on held-out rows against the mean baseline, and how many rows the repair recomputes at tolerance 0.02.

## Open
- Mapped packs in the server and the Companion (waits for point 2: if no real pair shows a saving, it is not worth it).
- A per-row weight for the mass of dropped rows (needs a per-row bias in attention).
- Sliding-window layers and accelerator caches are refused by splice, squeeze and import.
