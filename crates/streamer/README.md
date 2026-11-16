# expert-streamer

Clean-room groundwork for an expert-streaming inference engine that would work in tandem with
colibri. Written from the public safetensors layout and from first principles; no colibri code.

**Stage one (this crate):**
- `safetensors`: reads a weight file's header only (never the weights), with bounds checks.
- `expert_index`: finds every routed expert's byte ranges and the resident remainder.
- `policy`: replays a routing trace against LRU, decayed-frequency and the offline-optimal
  (Belady) policy, so the headroom of any smarter cache or prefetcher is measured before it is built.

- `trace`: reads colibri's documented text formats: the `ROUTE_TRACE=<path>` stream (order of
  events) and the `.coli_usage` history (popularity only), and can sample a trace from a history.

**Not here yet:** tokenizer, attention, int4 kernels, sampling, an I/O scheduler. This crate does
not run a model.

`cargo run --release -p expert-streamer --example headroom -- --usage FILE` samples a trace from a
real history: real popularity, but no recency, so it favours pinning the hottest experts and
understates LRU-like policies on real conversations; `-- --trace FILE --experts N` replays a real
`ROUTE_TRACE=` capture and is the measurement that decides whether a prefetcher is worth building.
The offline-optimal column sees the future, so it is a ceiling, not a target.

## Measurement models (cost models, not an engine)

- `prefetch`: a next-layer predictor from co-occurrence counts. On a sampled profile it only ties
  with plain popularity, because the sampler has no inter-layer correlation; whether real layers
  correlate is open until a real `ROUTE_TRACE` is replayed (`--example prefetch -- --trace FILE --experts N`).
- `speculative`: loads per accepted token when k drafted tokens are verified in one pass (union of
  their experts). On the sampled profile it does not pay off; a pessimistic bound if real
  consecutive tokens overlap more than sampled ones (unmeasured) (`--example speculative`). A
  trace shorter than one pass gives NaN (no data), never 0.
- `ioplan`: tokens/s from drive speed, expert size and overlap of I/O with compute. Defaults are
  assumptions, not measurements; `--measured-mbs` takes the app's drive-test result (`--example ioplan`).
