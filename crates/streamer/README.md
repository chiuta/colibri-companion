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

`cargo run -p expert-streamer --example headroom` prints a comparison on a synthetic trace. The
numbers say how the policies relate, not how a real model routes: real traces are needed.
