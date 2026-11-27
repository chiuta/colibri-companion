# Shrunk packs and cache maps

Two extensions of the context packs ([CONTEXT-PACKS.md](CONTEXT-PACKS.md), corbul engine only; clean-room, original code).

## Shrunk packs

When the texts and the question together do not fit the model's window, packs are refused. With shrinking they are shortened instead: `corbul packs --shrink auto` (shrink only as much as needed) or `--shrink R` (always keep about the fraction `R`, 0.1 to 1 of each pack's rows). The same goes for the `"shrink": "auto"` (or a number) field of `/v1/packs/complete`; the answer then lists what was shrunk (`"shrunk"`).

How: the last `--shrink-window` tokens of a pack (default 16) are re-read and vote for the rows to keep (SnapKV); the first rows are always kept (attention piles up there); the values that stay are refitted ([KV-FIT.md](KV-FIT.md); `--no-refit` turns that off). The kept keys are then rotated to the positions of their new slots, so the pack becomes *exactly the cache of its shortened text*: it can be spliced and repaired like any other pack, and with full repair the answer equals a plain prefill of the shortened text. The first `--shrink-skip` texts (default 1, normally the instructions) are left whole. Shrinking happens when the packs are put together; the shelf keeps the full packs.

In Colibri Companion: *Documents → Context packs (corbul)* → *Shrink texts that are too long*.

Limits: information is lost, how much depends on the model and the text (measure it with `corbul kvfit` or the *Measure cache shrinking…* window); models with sliding-window layers, Q8 caches and accelerator caches are refused; if the texts do not fit even with every shrinkable text removed, the request fails. Only tested on tiny random models so far.

## Cache maps between different models

A pack written by one model is useless to another. If the two share a tokenizer, their cache rows for the *same tokens* can be related: `corbul map-fit SRC DST --text FILE --out MAP` reads the text with both models and fits, per layer, a linear map (ridge least squares, with the mean) from SRC's key and value rows to DST's. Keys are compared without their position (rotated back to 0 and re-rotated on the way in), so rotary tables and head sizes may differ; layers are paired by relative depth. It prints the error on rows kept out of the fit next to the error of guessing the mean, and warns if the map is no better.

`corbul map-check SRC DST --map MAP --text FILE --question Q` splits the text into two packs written by SRC, brings them into DST through the map and answers three ways: DST's own full prefill (the truth), the mapped cache with the usual repair (`--tolerance`, default 0.02) and the mapped cache with no repair. It prints the rows recomputed and how far each answer is from the truth.

A mapped row is only a guess: imported rows are marked approximate, and the repair recomputes as many as it takes (all of them with tolerance 0, which equals a plain prefill). A bad map therefore costs time, never correctness. The library entry point is `ctxpack::compose_mapped`.

Limits: the tokenizers must give identical ids (checked, otherwise nothing is fitted); the map is for one pair of models (fingerprints are checked); its size is `kd × kd' × layers × 8` bytes; fitting keeps the calibration rows in memory (`--rows`, default 1024). A linear map between the caches of unrelated models may well be poor; whether it pays off for your pair can only be measured. It is not yet in the server or the Companion, and nothing has been measured on a real pair of models.
