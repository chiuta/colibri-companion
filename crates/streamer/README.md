# expert-streamer

Clean-room groundwork for an expert-streaming inference engine that would work in tandem with
colibri. Written from the public safetensors layout and from first principles; no colibri code.

**Stage one (this crate):**
- `safetensors`: reads a weight file's header only (never the weights), with bounds checks.
- `expert_index`: finds every routed expert's byte ranges and the resident remainder.
- `policy`: replays a routing trace against LRU, decayed-frequency and the offline-optimal
  (Belady) policy, so the headroom of any smarter cache or prefetcher is measured before it is built.

**Not here yet:** tokenizer, attention, int4 kernels, sampling, an I/O scheduler. This crate does
not run a model.

- `trace`: reads colibri's documented text formats: the `ROUTE_TRACE=<path>` stream (order of
  events) and the `.coli_usage` history (popularity only), and can sample a trace from a history.

`cargo run --release -p expert-streamer --example headroom -- --usage FILE` samples a trace from a
real history: real popularity, but no recency, so it favours pinning the hottest experts and
understates LRU-like policies on real conversations; `-- --trace FILE --experts N` replays a real
`ROUTE_TRACE=` capture and is the measurement that decides whether a prefetcher is worth building.
The offline-optimal column sees the future, so it is a ceiling, not a target.
