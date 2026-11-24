# Quick start: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` is the command line of the clean-room engine ([ENGINE.md](ENGINE.md)). Every command and flag on this page exists in `crates/engine-cli` and is printed by `corbul help`; a script (`tools/check_engine_docs.py`) fails if the code has a flag, subcommand or environment variable that this page does not mention. `corbul help` stays the final word.

**Honest warning:** `corbul` has been tested on tiny synthetic models and tokenizer reference files only. No real downloaded model has been run by the authors. Expect rough edges and please report them. `corbul capabilities` prints what works and what does not.

## Build

```bash
cargo build --release -p colibri-rs-cli     # produces target/release/corbul
```

A model is a folder with `config.json` + `*.safetensors` + `tokenizer.json`, or a single `.gguf` file.

## Look, download, run

```bash
corbul inspect  ./my-model                  # or file.gguf or owner/repo[@rev]: architecture, size, support
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors
corbul generate ./my-model --prompt "Hello" --max-tokens 64
corbul generate ./my-model --prompt-file question.txt --raw   # --raw: no chat template
corbul chat     ./my-model --system "You are concise."
corbul run      ./my-model --prompt "Hello"                   # `run` is `generate`
corbul serve    ./my-model --port 8080                        # bound to 127.0.0.1 by default
```

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated repositories and `HF_ENDPOINT` for a mirror. Downloads need `curl` (anonymous `wget` is a fallback).

Common options for `generate`, `chat` and `serve`: `--max-tokens N`, `--temperature T`, `--top-p P`, `--top-k K`, `--seed S`, `--kv f32|f16|q8`, `--no-mmap`, `--model-name NAME`, `--threads N` (also `COLIBRI_THREADS`), `--ctx N`, `--quiet`, `--ram-budget SIZE`, `--lora PATH[:S]` (repeatable; PEFT folder or GGUF adapter with a scale).

Extra sampler options for `generate` and `chat`:

```bash
corbul generate M --prompt "…" --min-p 0.05 --typical-p 0.9 --repeat-penalty 1.1 \
  --presence-penalty 0.2 --frequency-penalty 0.2 --mirostat 2 --mirostat-tau 5 --mirostat-eta 0.1 \
  --dry-multiplier 0.8 --xtc-probability 0.1
corbul generate M --prompt "…" --sampler-json '{"top_k": 20}'   # any sampler field, applied before the flags above
```

State-space and hybrid models (Mamba, RWKV, Jamba, Bamba) run through `generate` and `chat` as plain-text continuation: no chat template, no structured output, no `serve`.

## Will it run on this machine?

```bash
corbul doctor                # CPU instruction sets, chosen kernel, RAM, disk, sensors, GPU probe
corbul plan ./my-model --context 8192          # verdict (fits / tight / streams / too slow / won't run), speed, advice
corbul plan ./my-model --ram-budget 8g --calibrate   # pretend you have 8 GB; time this machine for about a second first
corbul capabilities [--json]                   # what this build can and cannot do (alias: corbul features)
```

Speeds are estimates, not measurements of your model. `ENGINE_SIMD` forces a kernel, `ENGINE_REPRODUCIBLE=1` turns reproducible mode on, `ENGINE_BACKEND=cpu|gpu|auto` is read by the library (`corbul` always computes on the CPU).

## Bigger than RAM

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"     # MoE: routed experts stream from disk
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --disk-mbps 800 --prompt "…"   # dense: layers stream; speed is bounded by the disk; --disk-mbps (default 500) only feeds the speed estimate it prints
```

## Several machines (LAN mesh)

```bash
# on every helper machine (the key needs 8+ characters; or set CORBUL_MESH_KEY)
corbul worker ./my-model --key MESHKEY12345 --bind 0.0.0.0:7471
corbul worker ./my-model --key-file mesh.key --discovery            # also answer UDP discovery probes (opt-in); --discovery-port N; --no-discovery forces it off
# on the main machine
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key MESHKEY12345 --prompt "…"
corbul generate ./my-model --discover --key MESHKEY12345 --prompt "…"      # broadcast a probe, use workers that hold the same model
corbul generate ./my-model --workers h:7471 --key MESHKEY12345 --wire-f16 --local-speed 2.5 --prompt "…"
```

`--wire-f16` halves the hidden-state traffic (and gives up bit-identity); `--local-speed` tells the splitter how fast this machine is relative to the workers; `--discover-target HOST:PORT` and `--discover-wait-ms N` change the probe address (default `255.255.255.255:47813`) and the wait (default 2000 ms). Traffic is authenticated but **not encrypted**: use a trusted network or a VPN. Tested on loopback only.

## Structured output, citations, adapters

```bash
corbul generate M --json --prompt "…"                       # valid JSON
corbul generate M --json-schema schema.json --prompt "…"    # a file or inline JSON
corbul generate M --regex '[A-Z]{3}-[0-9]{4}' --prompt "…"  # regular expression
corbul generate M --grammar rules.gbnf --prompt "…"         # GBNF grammar; use only one of the four
corbul generate M --sources notes.txt --sources paper.txt --prompt "Answer with quotes" --citations report.json   # quote-locked
corbul generate M --sources notes.txt --no-lock --prompt "…"   # only check the quotes afterwards
corbul generate M --lora ./adapter:0.8
```

The server accepts `response_format`, `grammar` and `sources` in requests and offers `POST /v1/quotes/verify`.

## Long chats, sessions, reproducibility

```bash
corbul generate M --session ./sessions --session-cap 2g --prompt "…"   # reuse the longest saved prompt prefix; --compress shrinks saved files (not with --kv-policy)
corbul generate M --kv-policy h2o --kv-budget 2048 --kv-sinks 4 --prompt "…"   # streaming | h2o | snapkv; lossy by design
corbul generate M --governor --verbose --prompt "…"            # adapt to temperature, memory and battery (Linux); --no-governor turns it off
corbul generate M --lang ro --prompt "…"                         # forbid tokens of foreign scripts; Romanian: no ş ţ
corbul generate M --reproducible --temperature 0 --prompt "hi" --receipt r.json --receipt-key-file receipt.key
corbul verify-receipt r.json --key-file receipt.key --model M      # checks the seal (and the weights if --model is given)
corbul replay r.json --model M --key-file receipt.key              # re-runs the generation and compares
corbul replay r.json --model M --key-file receipt.key --prompt "…" --raw     # when the receipt keeps no prompt text; or --prompt-tokens FILE
```

Receipts work with the plain generation loop, not with MoE or state-space models. Optional: `--receipt-key K` (visible in the process list; prefer `--receipt-key-file`), `--receipt-topk N`, `--keep-steps`, `--leaves`.

## Faster decoding

```bash
corbul generate M --draft ./small-model --draft-len 4 --prompt "…"      # a draft model proposes, M verifies
corbul generate M --lookahead --ngram-min 1 --ngram-max 4 --prompt "…"   # prompt-lookup, no extra model
corbul generate M --self-spec --skip-layers auto --prompt "…"            # M drafts with a subset of its layers; or --skip-layers 4,6,8-11
```

Same greedy answer and same sampling distribution; text appears at the end. Only `--temperature`, `--top-k`, `--top-p` apply; not with structured output, receipts, sessions or the mesh.

## Search, quantise, merge, steer, retrieve, watermark, language

```bash
corbul search M --prompt "Q" --method beam --width 4 --all --explain --json     # also --length-penalty L
corbul search M --prompt "2+2?" --method best-of-n --samples 5
corbul search M --prompt "2+2?" --method vote --samples 5 --answer number       # or text
corbul autoquant ./model --budget 4.5bpw --recipe recipe.json --report report.txt   # 4.5bpw, 6g or Q4_K; --candidates Q4_K,Q6_K --base Q8_0 --imatrix I --sample-rows 256 --json
corbul quantize ./model out.gguf --type Q4_K_M --imatrix imatrix.dat    # or --recipe recipe.json; --name NAME
corbul merge merge.json --out merged.safetensors --dry-run              # --model PATH (repeatable) --max-memory 8g --overwrite --verbose
corbul steer train M --positive polite.txt --negative rude.txt --out polite.cvec --method mean-diff   # or pca
corbul steer apply M --vector polite.cvec --strength 1 --layers 8-20 --positions last --prompt "…" --no-attention
corbul steer lens M --prompt "The capital of France is"
corbul rag add ./lib notes.txt paper.txt --chunk-chars 800 --overlap 120 --embed-model ./embedder
corbul rag search ./lib --query "refund policy" --k 5 --json
corbul rag ask ./lib M --question "What is the refund period?"             # answers with quote-locked citations
corbul lang detect "Aceasta este o propoziție."                          # or --file text.txt
corbul lang normalize --file text.txt                                    # ș/ț, mojibake, quotes
corbul generate M --prompt "…" --watermark SECRETKEY --wm-gamma 0.25 --wm-delta 2 --wm-context 4
corbul watermark detect M --key SECRETKEY --file answer.txt --no-dedup   # or --text "…"
```

Not every flag combination works; each command prints a one-line usage error when it does not. Nothing here has been run on a real checkpoint.

## Serving

```bash
corbul serve --models-dir ./models --host 127.0.0.1 --port 8080 --token-file token.txt --default-model qwen
corbul serve --models-dir ./models --alias fast=qwen-0.5b --max-loaded 2 --sleep-idle-seconds 600
corbul serve --models-dir ./models --keys-file keys.json --allow-admin --max-kv 8g --max-heavy 4
corbul serve --models-dir ./models --cors "https://app.example" --log-prompts --drain-seconds 10 --config-dir ./cfg
```

Open `http://127.0.0.1:8080/` for the web interface (also `/ui`). The page is public; every call it makes needs the token. `--token T` is the same as `CORBUL_TOKEN` (visible in the process list, so prefer `--token-file`). `--keys-file` holds API keys with per-key model lists and rate limits. `--allow-admin` lets key holders use `/api/create`, `/api/delete`, `/models/load` and `/models/unload`; by default only the master token may. `--max-kv` refuses requests whose KV cache would exceed the size (default 8g); `--max-heavy` limits concurrent tokenize, template and verify requests (default 4). `--no-log` silences the request log. `GET /v1/capabilities` lists what works. Embedding and rerank routes answer 501.

## Not available in `corbul` (yet)

- Multi-head latent attention (DeepSeek-style) models, RWKV in GGUF form, Zamba/Zamba2, i-quants, WordPiece/BERT models.
- Vision and audio inputs (the library has the blocks), embedding and rerank endpoints, the GPU back end.
- Bit-identity of ordinary quantised prefill (see ENGINE.md, "Exactness").
