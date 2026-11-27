# Context packs

Reuse the work done on a text when it appears again inside a different prompt (corbul engine only; clean-room, original code).

A prefix cache only hits when the shared text comes first. A *pack* is the KV cache of one text computed alone and kept on a shelf (signed, LRU-capped snapshot store).

`compose` puts packs one after another: rows are copied and their keys rotated by the position difference (exact), then the start of each later pack is recomputed in growing windows until the measured change of the rows drops under a tolerance. The question is always computed fresh.

`tolerance = 0` recomputes everything and equals a full prefill. Models with sliding-window layers, evicted caches or accelerator caches are refused and fall back to an ordinary prefill.

Not exact until repaired: a state with moved-in rows reports `is_exact() == false` and is never written to disk.

Limits: JustVugg's colibri engine cannot export caches, so this works only with corbul. Tested on small synthetic models; quality on real models is not yet measured.

## Command line

`corbul packs MODEL --session DIR --pack FILE… --question TEXT [--tolerance 0.02] [--probe 8] [--max-new 64] [--check]` shelves every file once, then answers using the stored packs in any order. `--check` also runs a full prefill and prints the score difference and the time of each.
