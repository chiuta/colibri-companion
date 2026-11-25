# Avvio rapido: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` è la riga di comando del motore clean-room ([ENGINE.it.md](ENGINE.it.md)). Ogni comando e ogni opzione di questa pagina esiste in `crates/engine-cli` ed è stampato da `corbul help`; uno script (`tools/check_engine_docs.py`) fallisce se il codice ha un'opzione, un sottocomando o una variabile d'ambiente che questa pagina non menziona. `corbul help` resta l'ultima parola.

**Avvertenza onesta:** `corbul` è stato provato solo su minuscoli modelli sintetici e su file di riferimento del tokenizer. Nessun modello reale scaricato è stato eseguito dagli autori. Aspettati qualche spigolo e segnalalo, per favore. `corbul capabilities` stampa che cosa funziona e che cosa no.

## Compilare

```bash
cargo build --release -p colibri-rs-cli     # produces target/release/corbul
```

Un modello è una cartella con `config.json` + `*.safetensors` + `tokenizer.json`, oppure un singolo file `.gguf`.

## Guardare, scaricare, eseguire

```bash
corbul inspect  ./my-model                  # or file.gguf or owner/repo[@rev]: architecture, size, support
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors
corbul generate ./my-model --prompt "Hello" --max-tokens 64
corbul generate ./my-model --prompt-file question.txt --raw   # --raw: no chat template
corbul chat     ./my-model --system "You are concise."
corbul run      ./my-model --prompt "Hello"                   # `run` is `generate`
corbul serve    ./my-model --port 8080                        # bound to 127.0.0.1 by default
```

Imposta `HF_TOKEN` (o `HUGGING_FACE_HUB_TOKEN`) per i repository con accesso limitato e `HF_ENDPOINT` per un mirror. I download richiedono `curl` (`wget` anonimo è un ripiego).

Opzioni comuni a `generate`, `chat` e `serve`: `--max-tokens N`, `--temperature T`, `--top-p P`, `--top-k K`, `--seed S`, `--kv f32|f16|q8`, `--no-mmap`, `--model-name NAME`, `--threads N` (anche `COLIBRI_THREADS`), `--ctx N`, `--quiet`, `--ram-budget SIZE`, `--lora PATH[:S]` (ripetibile; cartella PEFT o adattatore GGUF con un fattore di scala).

Opzioni aggiuntive del campionatore per `generate` e `chat`:

```bash
corbul generate M --prompt "…" --min-p 0.05 --typical-p 0.9 --repeat-penalty 1.1 \
  --presence-penalty 0.2 --frequency-penalty 0.2 --mirostat 2 --mirostat-tau 5 --mirostat-eta 0.1 \
  --dry-multiplier 0.8 --xtc-probability 0.1
corbul generate M --prompt "…" --sampler-json '{"top_k": 20}'   # any sampler field, applied before the flags above
```

I modelli a spazi di stato e ibridi (Mamba, RWKV, Jamba, Bamba) passano per `generate` e `chat` come semplice continuazione di testo: niente template di chat, niente output strutturato, niente `serve`.

## Girerà su questa macchina?

```bash
corbul doctor                # CPU instruction sets, chosen kernel, RAM, disk, sensors, GPU probe
corbul plan ./my-model --context 8192          # verdict (fits / tight / streams / too slow / won't run), speed, advice
corbul plan ./my-model --ram-budget 8g --calibrate   # pretend you have 8 GB; time this machine for about a second first
corbul capabilities [--json]                   # what this build can and cannot do (alias: corbul features)
```

Le velocità sono stime, non misure del tuo modello. `ENGINE_SIMD` impone un kernel, `ENGINE_REPRODUCIBLE=1` attiva la modalità riproducibile, `ENGINE_BACKEND=cpu|gpu|auto` è letta dalla libreria (`corbul` calcola sempre sulla CPU).

## Più grande della RAM

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"     # MoE: routed experts stream from disk
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --disk-mbps 800 --prompt "…"   # dense: layers stream; speed is bounded by the disk; --disk-mbps (default 500) only feeds the speed estimate it prints
```

## Più macchine (mesh in rete locale)

```bash
# on every helper machine (the key needs 8+ characters; or set CORBUL_MESH_KEY)
corbul worker ./my-model --key MESHKEY12345 --bind 0.0.0.0:7471
corbul worker ./my-model --key-file mesh.key --discovery            # also answer UDP discovery probes (opt-in); --discovery-port N; --no-discovery forces it off
# on the main machine
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key MESHKEY12345 --prompt "…"
corbul generate ./my-model --discover --key MESHKEY12345 --prompt "…"      # broadcast a probe, use workers that hold the same model
corbul generate ./my-model --workers h:7471 --key MESHKEY12345 --wire-f16 --local-speed 2.5 --prompt "…"
```

`--wire-f16` dimezza il traffico dello stato nascosto (e rinuncia all'identità bit per bit); `--local-speed` dice al suddivisore quanto è veloce questa macchina rispetto ai worker; `--discover-target HOST:PORT` e `--discover-wait-ms N` cambiano l'indirizzo della sonda (predefinito `255.255.255.255:47813`) e l'attesa (predefinita 2000 ms). Il traffico è autenticato ma **non cifrato**: usa una rete fidata o una VPN. Provato solo in loopback.

## Output strutturato, citazioni, adattatori

```bash
corbul generate M --json --prompt "…"                       # valid JSON
corbul generate M --json-schema schema.json --prompt "…"    # a file or inline JSON
corbul generate M --regex '[A-Z]{3}-[0-9]{4}' --prompt "…"  # regular expression
corbul generate M --grammar rules.gbnf --prompt "…"         # GBNF grammar; use only one of the four
corbul generate M --sources notes.txt --sources paper.txt --prompt "Answer with quotes" --citations report.json   # quote-locked
corbul generate M --sources notes.txt --no-lock --prompt "…"   # only check the quotes afterwards
corbul generate M --lora ./adapter:0.8
```

Il server accetta `response_format`, `grammar` e `sources` nelle richieste e offre `POST /v1/quotes/verify`.

## Chat lunghe, sessioni, riproducibilità

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

Le ricevute funzionano con il ciclo di generazione semplice, non con i modelli MoE o a spazi di stato. Facoltativi: `--receipt-key K` (visibile nell'elenco dei processi; preferisci `--receipt-key-file`), `--receipt-topk N`, `--keep-steps`, `--leaves`.

## Decodifica più veloce

```bash
corbul generate M --draft ./small-model --draft-len 4 --prompt "…"      # a draft model proposes, M verifies
corbul generate M --lookahead --ngram-min 1 --ngram-max 4 --prompt "…"   # prompt-lookup, no extra model
corbul generate M --self-spec --skip-layers auto --prompt "…"            # M drafts with a subset of its layers; or --skip-layers 4,6,8-11
```

Stessa risposta greedy e stessa distribuzione di campionamento; il testo compare alla fine. Valgono solo `--temperature`, `--top-k`, `--top-p`; non con output strutturato, ricevute, sessioni o mesh.

## Ricerca, quantizzazione, fusione, guida, recupero, filigrana, lingua

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

Non tutte le combinazioni di opzioni funzionano; ogni comando stampa un errore d'uso di una riga quando non è così. Nulla di tutto questo è stato eseguito su un checkpoint reale.

## Servizio

```bash
corbul serve --models-dir ./models --host 127.0.0.1 --port 8080 --token-file token.txt --default-model qwen
corbul serve --models-dir ./models --alias fast=qwen-0.5b --max-loaded 2 --sleep-idle-seconds 600
corbul serve --models-dir ./models --keys-file keys.json --allow-admin --max-kv 8g --max-heavy 4
corbul serve --models-dir ./models --cors "https://app.example" --log-prompts --drain-seconds 10 --config-dir ./cfg
```

Apri `http://127.0.0.1:8080/` per l'interfaccia web (anche `/ui`). La pagina è pubblica; ogni chiamata che fa richiede il token. `--token T` equivale a `CORBUL_TOKEN` (visibile nell'elenco dei processi, quindi preferisci `--token-file`). `--keys-file` contiene chiavi API con elenchi di modelli e limiti di frequenza per chiave. `--allow-admin` consente ai titolari di chiavi di usare `/api/create`, `/api/delete`, `/models/load` e `/models/unload`; per impostazione predefinita può farlo solo il token principale. `--max-kv` rifiuta le richieste la cui cache KV supererebbe la dimensione indicata (predefinito 8g); `--max-heavy` limita le richieste simultanee di tokenizzazione, template e verifica (predefinito 4). `--no-log` silenzia il log delle richieste. `GET /v1/capabilities` elenca ciò che funziona. Le rotte di embedding funzionano (`/v1/embeddings`, `/api/embed`); le rotte di rerank rispondono 501.

## Non disponibile in `corbul` (per ora)

- Modelli con attenzione latente multi-testa (stile DeepSeek), RWKV in forma GGUF, Zamba/Zamba2, i-quant, modelli WordPiece/BERT.
- Input di visione e audio (la libreria ha i blocchi), endpoint di reranking, il back end GPU.
- Identità bit per bit del prefill quantizzato ordinario (vedi ENGINE.md, «Esattezza»).
