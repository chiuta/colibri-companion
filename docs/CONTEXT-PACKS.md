# Context packs

Reuse the work done on a text when it appears again inside a different prompt (corbul engine only; clean-room, original code).

A prefix cache only hits when the shared text comes first. A *pack* is the KV cache of one text computed alone and kept on a shelf (signed, LRU-capped snapshot store).

`compose` puts packs one after another: rows are copied and their keys rotated by the position difference (exact), then the start of each later pack is recomputed in growing windows until the measured change of the rows drops under a tolerance. The question is always computed fresh.

`tolerance = 0` recomputes everything and equals a full prefill. Models with sliding-window layers, evicted caches or accelerator caches are refused and fall back to an ordinary prefill.

Not exact until repaired: a state with moved-in rows reports `is_exact() == false` and is never written to disk.

Limits: JustVugg's colibri engine cannot export caches, so this works only with corbul. Tested on small synthetic models; quality on real models is not yet measured.

## Command line

`corbul packs MODEL --session DIR --pack FILE… --question TEXT [--tolerance 0.02] [--probe 8] [--max-new 64] [--check]` shelves every file once, then answers using the stored packs in any order. `--check` also runs a full prefill and prints the score difference and the time of each.

## Server

`corbul serve MODEL --packs DIR [--packs-cap 2g]` adds `POST /v1/packs/complete` with `{"packs": [text, …], "question": "…", "max_tokens": 64, "tolerance": 0.02, "probe": 8}`. The answer has `text`, `usage` and a `packs` object (which texts were already shelved, rows copied, recomputed, fresh and saved, and the change measured at each join). Plain-text greedy completion; no chat template.

## In Colibri Companion

With documents switched on, tick *Reuse earlier work*. Each retrieved passage becomes a pack labelled by its source (not by its rank), so the same passage is reused whatever the order. It needs corbul started with `--packs`; with any other engine the setting is ignored and the ordinary chat runs. The answer arrives whole, not word by word.
