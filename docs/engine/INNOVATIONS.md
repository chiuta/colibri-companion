# What is different about this engine

[English](INNOVATIONS.md) · [Română](INNOVATIONS.ro.md) · [Français](INNOVATIONS.fr.md) · [Italiano](INNOVATIONS.it.md) · [Español](INNOVATIONS.es.md) · [Português](INNOVATIONS.pt.md) · [Deutsch](INNOVATIONS.de.md) · [Magyar](INNOVATIONS.hu.md)

Plain-language notes on the features that other local engines mostly do not have. Each says what it does, where you reach it, and what it cannot do. All were tested on synthetic models only; none has been run on a real checkpoint by the authors. "In `corbul`" means a command or flag exists today (`corbul help`); "library only" means Rust code without a command. See [ENGINE.md](ENGINE.md) and [QUICKSTART.md](QUICKSTART.md).

## Integrity, hardware and memory

### LAN mesh

Module `mesh`. In `corbul`: `worker`, `generate --workers`, `--discover`, `--key`, `--wire-f16`. A model too big for one machine is cut into layer ranges, one per computer. Only the hidden state travels between them. Machines prove they share a key (mutual HMAC-SHA256), every frame carries a MAC, handshake sizes are capped, a dead worker is dropped and its layers redistributed. Dense f32 logits equal a single machine. UDP discovery is opt-in: a worker answers probes only when started with `--discovery` or `--discovery-port` (`--no-discovery` forces it off), and the coordinator's `--discover` broadcasts a probe and waits for answers from workers holding the same model. *Cannot:* encrypt (use a trusted LAN or a VPN); it was tested on loopback only; speed against one machine was not measured.

### Layer streaming

Module `stream_layers`. In `corbul`: `--stream-layers`, `--ram-budget`, `--stream-window`, `--disk-mbps` (for estimates). Runs a dense model larger than RAM by keeping a small window of layers in memory and prefetching on a background thread. Results equal the in-RAM model. *Cannot:* every token reads every layer, so speed is bounded by disk speed; dense layers only (mixture-of-experts models stream their routed experts instead, with `--ram-budget`).

### Quote-locked citations

Module `quotelock`. In `corbul`: `--sources FILE` (repeatable), `--no-lock`, `--citations FILE`; the server accepts `sources` and offers `POST /v1/quotes/verify`. While decoding, text between « and » must be a verbatim passage of a source (tolerant of case, spacing, quote styles and Romanian diacritics); `[S3]«` limits a quote to source 3. A verifier then classifies each quote (verbatim, wrong source, truncated, altered, merged, unsupported) and scores the sentences outside quotes; the report says when a work limit was hit (`limit_hit`). *Cannot:* make the text around the quotes true; streaming responses carry no citation block.

### Planner, governor and energy meter

Module `governor` (and `plan`, `hw`). In `corbul`: `plan`, `doctor`, `--governor`, `--no-governor`, `--verbose`, `--calibrate`. Before loading, says whether the model fits, streams, runs slowly or cannot run, with an estimated speed. While running, the governor reads temperature, memory and battery (Linux) and advises threads and pauses without oscillating. *Cannot:* speeds are estimates (`--calibrate` times your machine for about a second); energy is an assumption unless a RAPL counter is readable; the carbon intensity (400 g/kWh) is a placeholder.

### KV-cache policies and session snapshots

Modules `kvpolicy`, `snapshot`. In `corbul`: `--kv-policy streaming|h2o|snapkv`, `--kv-budget`, `--kv-sinks`, `--session DIR`, `--session-cap`, `--compress`. Keep long chats inside a memory budget (attention sinks, heavy hitters, prompt compression) and save or resume a conversation from a checksummed file, reusing the longest saved prefix. Snapshots can carry a keyed HMAC. *Cannot:* eviction is lossy by design; `--kv-policy` and `--session` cannot be combined; very long chats can pass the trained context.

### Reproducible mode and receipts

Modules `simd` (switch), `receipt`. In `corbul`: `--reproducible`, `--receipt FILE`, `--receipt-key`, `--receipt-key-file`, `--receipt-topk`, `--keep-steps`, `--leaves`, `verify-receipt`, `replay`. A receipt records the model (a Merkle fingerprint of the weights), tokenizer, sampler settings, prompt and output with a hash chain, sealed with SHA-256 and an optional HMAC key; `replay` re-runs it. Damaged weights are localised to a tensor and chunk. *Cannot:* prove anything to a stranger without the HMAC key or a re-run (no public-key signatures); MoE and state-space models are refused; ARM untested; reproducible mode costs speed.

### Romanian-first language tools

Module `lang`. In `corbul`: `lang detect`, `lang normalize`, `--lang CODE` (not in the server). Normalises ș/ț and ş/ţ, repairs mojibake, identifies 27 languages, forbids foreign scripts during decoding (Romanian: no ş ţ), formats Romanian numbers, money and dates, restores missing diacritics with a flag on every uncertain word. *Cannot:* restoration uses a lexicon of about 650 forms and was not measured on a corpus; ambiguous words are resolved by a few context rules; language identification needs about five words.

### Uncertainty analytics

Module `confidence`. Library only (demo example `confidence_demo`). Per-token, word, sentence and answer statistics for a heat-map, a transparent hallucination-risk heuristic, self-consistency scores and calibration tools. *Cannot:* the risk score is a heuristic, not a truth detector.

## Test-time compute, tuning, retrieval

### Search and lookahead

Module `search`. In `corbul`: `search --method beam|best-of-n|vote` (`--width`, `--samples`, `--answer number|text`, `--length-penalty`, `--all`, `--explain`, `--json`) and `--lookahead` on `generate` (`--ngram-min`, `--ngram-max`). Beam search, best-of-N, self-consistency voting, a conversation tree (library), prompt-lookup speculative decoding. *Cannot:* lookahead helps only when text repeats earlier text; best-of-N needs a good scorer.

### Speculative decoding

Modules `spec`, `selfspec`. In `corbul`: `--draft MODEL`, `--lookahead`, `--self-spec`, `--skip-layers auto|LIST`, `--draft-len`. Same greedy answer and same sampling distribution with fewer full passes; output appears at the end. Only `--temperature`, `--top-k`, `--top-p`; not with structured output, receipts, sessions or the mesh. *Cannot:* promise a speed-up; self-speculation falls back to plain steps for sliding-window rings that would wrap, compacted caches and offloaded layers.

### Automatic mixed quantisation and the quantiser

Modules `autoquant`, `quantize`. In `corbul`: `autoquant FOLDER --budget 4.5bpw|6g|Q4_K` (`--recipe`, `--report`, `--candidates`, `--base`, `--imatrix`, `--sample-rows`) and `quantize IN OUT.gguf --type NAME | --recipe FILE`. Measures weight error per tensor group in each format and picks formats under a size budget (exact knapsack or greedy). *Cannot:* it estimates weight error, not answer quality; it assumes damages add up; i-quants do not exist.

### Steering vectors and logit lens

Module `steer`. In `corbul`: `steer train|apply|lens` (`--positive`, `--negative`, `--method mean-diff|pca`, `--out`, `--vector`, `--strength`, `--layers`, `--positions`, `--no-attention`). Learns a direction from two prompt sets and adds it to the residual stream without changing weights; the lens shows what each layer would predict. *Cannot:* work with state-space models, structured output, speculation, receipts or sessions; a vector from one fine-tune is accepted by another with the same shape.

### Local retrieval

Module `rag`. In `corbul`: `rag add|search|ask` (`--query`, `--question`, `--k`, `--chunk-chars`, `--overlap`, `--embed-model`). Romanian-aware chunking, an HNSW index in plain Rust, BM25 with diacritic folding, rank fusion, passages numbered `[S1]…` feeding quote-locked citations, checksummed files. *Cannot:* the default embedder is a hashing embedder (words and word parts, not meaning); `--embed-model` uses hidden states of a generative model (weak); the CLI re-indexes per call and reads plain text, not PDF or Office files (the library has ingestion helpers).

### Model merging

Module `merge`. In `corbul`: `merge CONFIG.json` (`--out`, `--model`, `--dry-run`, `--max-memory`, `--overwrite`, `--verbose`). Linear, SLERP, task arithmetic, TIES, DARE (seeded) and layer slices, one tensor at a time. Shapes and ranges are checked before writing; symlinks are not followed. *Cannot:* guarantee quality; writes safetensors weights only, never quantised, and you copy `config.json` and the tokenizer yourself.

### Text watermark

Module `watermark`. In `corbul`: `generate --watermark KEY` (`--wm-gamma`, `--wm-delta`, `--wm-context`) and `watermark detect MODEL --key K --text T | --file F` (`--no-dedup`). A keyed "green list" bias and a z-score detector. *Cannot:* give proof; short texts, code and quotes carry little signal; paraphrase removes it; whoever has the key can forge it; no mark does not prove human authorship.

## Multimodal (library only)

### Vision

Modules `vision`, `imageio`. A CLIP/SigLIP-style encoder, projector, tiling, and a `Model::forward_segments` hook for feeding embeddings into a prompt; decoders for PNG, JPEG (baseline and progressive), BMP and PNM written from the specifications, with size limits. *Cannot:* be used from `corbul` (text only); no real checkpoint was run.

### Audio

Module `audio`. WAV reading and writing, resampling, FFT, log-mel features, voice-activity detection, a Whisper-layout runner. *Cannot:* be used from `corbul`; no real Whisper checkpoint was run.

## State-space, hybrid and GPU (see ENGINE.md)

Modules `ssm`, `hybrid`, `mla`, and the `companion-engine-gpu` crate are described, with their limits, in ENGINE.md.
