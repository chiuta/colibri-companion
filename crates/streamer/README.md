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
  Input files must be regular files of at most 1 GiB; ids outside `0..n_experts` are skipped, and
  sampling requests that would not fit memory are refused.
- `expert_index` also recognises packed expert tensors (`...experts.gate_up_proj` with the experts
  on the leading dimension); a layout that stacks experts on another axis is not handled.
- Safetensors headers with duplicate keys are rejected.
- All replay policies run in O(log n) per access (checked against naive references in the tests).
  Numbers are only as real as the trace: no real colibri capture has been replayed yet.

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
  An optional per-layer expert BUDGET (`verify_pass`, `--budget 1,1.5,2`, multiples of the experts
  one token uses per layer) follows MoE-Spec (arXiv 2602.16052, training-free): each layer loads
  only the experts that most tokens of the window route to, ties to the lower id, and drops the
  tail. Read the loads together with the printed share of routed slots dropped. On a trace whose
  tokens do not overlap, a tight budget "saves" loads only by dropping most of the work
  (e.g. 55-77% of slots at 1x for k=4..8 on the synthetic and correlated traces); that is a
  warning, not a gain. The paper reports 10-30% more throughput at maintained quality on ITS
  models; this model measures neither quality nor how the acceptance rate would change, so
  nothing here supports or refutes that for any model. Sampled traces (`.coli_usage`) overlap
  only through popularity and so understate the window overlap; a real `ROUTE_TRACE` is needed.
  `locality_trace` is a synthetic drifting-pool trace that exists to exercise the code.
- `ioplan`: tokens/s from drive speed, expert size and overlap of I/O with compute. The
  "overlapped" figure is a ceiling that assumes a perfect prefetcher. Defaults are
  assumptions, not measurements; `--measured-mbs` takes the app's drive-test result (`--example ioplan`).

## Measured skew of real histories

`docs/profiles.md` reports concentration, hold-out and per-layer figures measured on the
histories colibri ships (`cargo run --release -p expert-streamer --example profiles`), and
`skew::SKEW_SAMPLES` / `skew::pinned_hit_range` expose the measured spread as constants so other
crates can use a range instead of one number. At 10% of all experts pinned, a fixed hot set
served 34-75% of the uses on the history it was built from but only 11-30% on another history of
the same model; a per-layer budget lost under 1 point against one global ranking. Limits: counts
only (no order, so nothing about recency and no stability test inside a history), two model
families, four distinct histories, all from the colibri authors' own machines and prompts.
