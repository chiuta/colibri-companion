# Shrinking the cache and refitting the values

`corbul kvfit MODEL --text FILE --question Q` measures what it costs to keep only a fraction of a text's cache rows (SnapKV picks them), with and without **refitting the values** that stay.

The last `--window` tokens of the text are recorded as probe queries. The kept rows' values are then moved the least that makes attention over them equal, on the probes, what attention over all rows gave (a ridge least-squares in the dual form, `crates/engine/src/kvfit.rs`). The fit can never be worse than the plain values *on the probes*; whether it helps on a question nobody probed depends on the model and the text, and only this command, run on your model, tells you.

Options: `--budget 0.25` (fraction of rows kept), `--window 16`, `--ridge 0.01`, `--max-new 32`. It prints the fit error on the probes, the largest difference of the first scores against the full cache, whether the first token agrees and how many of the first tokens of the answer agree.

Limits: values only (keys carry the position and stay; the weight of dropped rows is not modelled), sliding-window layers are left as they are, the fit costs time in proportion to rows × probes per head. Not yet joined to context packs (a shrunk pack cannot be spliced). Nothing measured on a real model yet.

Context packs can also be shelved in half the size: `corbul packs --pack-kv f16` and `corbul serve --packs-kv f16` (F16 rounding on every row), and compressed without loss: `--pack-compress`, `--packs-compress`.
