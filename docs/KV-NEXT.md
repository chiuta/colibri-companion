# Next steps for the cache work (plan, not implemented)

What exists: exact re-rotation splice, adaptive repair, the pack shelf, and eviction with a position gap (`kvpolicy.rs`: StreamingLLM, H2O, SnapKV).

## 1. Compacting a pack (shrinking its rows, in the style of attention matching)

**Done since this plan was written:** recording the probe queries (`State::record_queries`), the value fit (`kvfit.rs`, tested on synthetic data and on a tiny model), `State::apply_policy_fitted` and the measuring command `corbul kvfit` (see `KV-FIT.md`). **Still open:** the per-row positions in `splice_cache` (a shrunk pack cannot be spliced yet), the weight of dropped rows (needs a per-row bias in attention), and the measurements on a real model.

Goal: keep `m < n` rows of a pack and correct them so attention over the kept rows gives (nearly) what it gave over all `n`.

What is already there: choosing the rows (`State::apply_policy`, `compact`) with RoPE kept exact through the position gap.

What is missing, and why it was **not** written blind:
- **Correcting the kept rows** (refitting values and adding a per-row weight for the dropped mass) needs the queries that will be asked of the pack. The engine does not keep them; they come from every layer's hidden state, and the layers differ by family (MoE, MLA, sliding window). Without a real model there is no way to see whether a fit helps or hurts: a random test model has near-uniform attention, so any selection looks as good as any other.
- **Splicing a compacted pack** needs per-row positions in the target state (`abs`/`gap`) and cannot be repaired with `refresh_slots`, which recomputes rows from contiguous tokens. That is a change in the core of `splice_cache`.

Plan: (a) record queries of the last window of each pack during prefill (opt-in); (b) fit per layer and head by least squares; (c) extend `splice_cache` with per-row positions; (d) measure on a real model: first-token agreement and answer quality against a full prefill, at 50 %, 25 % and 10 % of the rows. Ship only if the numbers justify it.

## 2. Moving a cache between different models

A cache is meaningful only to the weights that wrote it. Between two models a *learned* map from one layer space to the other is needed, trained on paired activations of both models. No such data or model pair is available here, and there is no honest test without them. Same model, different quantisation or different file is the only case that can be considered first (rows differ by rounding only); that case is not implemented either.

## What is needed to go on
A real small model that corbul loads (a GGUF), and a measured baseline: `corbul packs --check` already prints the largest score difference and whether the first token agrees.
