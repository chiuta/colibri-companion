# Quick start: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` is the command line of the clean-room engine ([ENGINE.md](ENGINE.md)). This page lists only commands and flags that exist in `crates/engine-cli` as of this writing; the CLI is still being extended, so `corbul help` is the final word. Some flags below are read by the code but not yet printed by `corbul help`; they are marked *(not in help yet)*. Things that do not exist yet are marked **planned**.

**Honest warning:** `corbul` has been tested only on tiny synthetic models. Expect rough edges on real checkpoints and please report them.

## Build

```bash
cargo build --release -p colibri-rs-cli     # produces target/release/corbul
```

A model is either a folder with `config.json` + `*.safetensors` + `tokenizer.json`, or a single `.gguf` file.

## Look, download, run

```bash
corbul inspect  ./my-model                  # or file.gguf or owner/repo[@rev]: architecture, size, support
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors; HF_TOKEN for gated repos, HF_ENDPOINT for a mirror
corbul generate ./my-model --prompt "Hello" --max-tokens 64
corbul chat     ./my-model --system "You are concise."
corbul serve    ./my-model --port 8080      # OpenAI / Ollama / Anthropic-compatible, bound to 127.0.0.1 by default
```

Common options for `generate`, `chat` and `serve`: `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--kv f32|f16|q8`, `--no-mmap`, `--threads N`, `--ctx N`, `--model-name`, `--quiet`. `generate` also takes `--prompt-file F` and `--raw` (no chat template). `corbul run` is an alias of `generate`.

Serving several models: `--models-dir DIR`, `--alias NAME=MODEL`, `--max-loaded N`, `--sleep-idle-seconds S`, `--token T` (or `CORBUL_TOKEN`), `--keys-file FILE`, `--cors ORIGINS`, `--log-prompts`, `--no-log`, `--drain-seconds S`, `--config-dir DIR`.

## Will it run on this machine?

```bash
corbul doctor                # what the engine sees: CPU instruction sets, kernels, RAM, disk, sensors, energy counter
corbul plan ./my-model --context 8192            # verdict: FITS / TIGHT / STREAMS / TOO SLOW / WON'T RUN, speed, advice
corbul plan ./my-model --ram-budget 8g --calibrate   # pretend you have 8 GB; time this machine for a second first
corbul capabilities [--json]                     # what this build can and cannot do
```

The speeds are estimates, not measurements of your model.

## Bigger than RAM

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"                        # MoE: routed experts stream from disk
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --prompt "…"   # dense: layers stream (disk-bound; --stream-window not in help yet)
```

## Several machines (LAN mesh)

```bash
# on every helper machine (the key needs 8+ characters; or set CORBUL_MESH_KEY)
corbul worker ./my-model --key SECRET --bind 0.0.0.0:7471          # --no-discovery to skip UDP discovery
# on the main machine
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key SECRET --prompt "…"
corbul generate ./my-model --discover --key SECRET --prompt "…"      # find workers on the LAN
```

Traffic is authenticated but **not encrypted**: use a trusted network or a VPN. Tested on loopback only.

## Structured output, citations, adapters

```bash
corbul generate M --json --prompt "…"                 # valid JSON output
corbul generate M --json-schema schema.json --prompt "…"      # a file or inline JSON; also --regex PATTERN, --grammar FILE (GBNF); use only one of the four; not in help yet
corbul generate M --sources notes.txt --sources paper.txt --prompt "Answer with quotes" --citations report.json   # quote-locked
corbul generate M --lora ./adapter:0.8                # PEFT or GGUF adapter with a scale
```

The server accepts `response_format`, `grammar` and `sources`, and offers `POST /v1/quotes/verify`.

## Long chats, sessions, reproducibility

```bash
corbul generate M --session ./sessions --prompt "…"             # reuse the longest saved prompt prefix (not with --kv-policy)
corbul generate M --kv-policy h2o --kv-budget 2048 --prompt "…" # streaming | h2o | snapkv; lossy by design
corbul generate M --governor --prompt "…"                       # adapt to temperature, memory and battery (Linux)
corbul generate M --reproducible --temperature 0 --prompt "hi" --receipt r.json --receipt-key SECRET
corbul verify-receipt r.json --key SECRET --model M            # checks the seal (and the weights if --model is given)
corbul replay r.json --model M --key SECRET                    # re-runs the generation and compares
```

Receipts work with the plain generation loop and not with MoE models. Optional: `--receipt-topk N`, `--keep-steps`, `--leaves` *(not in help yet)*.

## Planned (not in `corbul` yet)

- Loading Mamba / RWKV / MLA (DeepSeek-style) models, even though the library has the blocks.
- The Romanian language tools, uncertainty analytics, search and lookahead, self-speculative decoding, steering, retrieval, merging and the watermark (see [INNOVATIONS.md](INNOVATIONS.md)).
- Embedding and rerank endpoints (the server answers "not implemented"), vision and audio inputs.
- The GPU back end, and a web interface.
