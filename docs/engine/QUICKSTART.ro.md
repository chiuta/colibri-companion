# Pornire rapidă: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` este linia de comandă a motorului clean-room ([ENGINE.md](ENGINE.md)). Fiecare comandă și fiecare flag de pe această pagină există în `crates/engine-cli` și este afișat de `corbul help`; un script (`tools/check_engine_docs.py`) eșuează dacă în cod există un flag, o subcomandă sau o variabilă de mediu pe care această pagină nu le menționează. `corbul help` rămâne cuvântul final.

**Avertisment sincer:** `corbul` a fost testat doar pe modele sintetice minuscule și pe fișiere de referință ale tokenizatorului. Autorii nu au rulat niciun model real descărcat. Așteaptă-te la neajunsuri și te rugăm să le raportezi. `corbul capabilities` afișează ce funcționează și ce nu.

## Construire

```bash
cargo build --release -p colibri-rs-cli     # produces target/release/corbul
```

Un model este un folder cu `config.json` + `*.safetensors` + `tokenizer.json`, sau un singur fișier `.gguf`.

## Privește, descarcă, rulează

```bash
corbul inspect  ./my-model                  # or file.gguf or owner/repo[@rev]: architecture, size, support
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors
corbul generate ./my-model --prompt "Hello" --max-tokens 64
corbul generate ./my-model --prompt-file question.txt --raw   # --raw: no chat template
corbul chat     ./my-model --system "You are concise."
corbul run      ./my-model --prompt "Hello"                   # `run` is `generate`
corbul serve    ./my-model --port 8080                        # bound to 127.0.0.1 by default
```

Setează `HF_TOKEN` (sau `HUGGING_FACE_HUB_TOKEN`) pentru depozite cu acces restricționat și `HF_ENDPOINT` pentru o oglindă. Descărcările au nevoie de `curl` (`wget` anonim este alternativa).

Opțiuni comune pentru `generate`, `chat` și `serve`: `--max-tokens N`, `--temperature T`, `--top-p P`, `--top-k K`, `--seed S`, `--kv f32|f16|q8`, `--no-mmap`, `--model-name NAME`, `--threads N` (și `COLIBRI_THREADS`), `--ctx N`, `--quiet`, `--ram-budget SIZE`, `--lora PATH[:S]` (repetabil; folder PEFT sau adaptor GGUF, cu o scală).

Opțiuni suplimentare de eșantionare pentru `generate` și `chat`:

```bash
corbul generate M --prompt "…" --min-p 0.05 --typical-p 0.9 --repeat-penalty 1.1 \
  --presence-penalty 0.2 --frequency-penalty 0.2 --mirostat 2 --mirostat-tau 5 --mirostat-eta 0.1 \
  --dry-multiplier 0.8 --xtc-probability 0.1
corbul generate M --prompt "…" --sampler-json '{"top_k": 20}'   # any sampler field, applied before the flags above
```

Modelele cu spațiu de stare și cele hibride (Mamba, RWKV, Jamba, Bamba) rulează prin `generate` și `chat` ca simplă continuare de text: fără șablon de conversație, fără ieșire structurată, fără `serve`.

## Va rula pe acest calculator?

```bash
corbul doctor                # CPU instruction sets, chosen kernel, RAM, disk, sensors, GPU probe
corbul plan ./my-model --context 8192          # verdict (fits / tight / streams / too slow / won't run), speed, advice
corbul plan ./my-model --ram-budget 8g --calibrate   # pretend you have 8 GB; time this machine for about a second first
corbul capabilities [--json]                   # what this build can and cannot do (alias: corbul features)
```

Vitezele sunt estimări, nu măsurători ale modelului tău. `ENGINE_SIMD` forțează un nucleu, `ENGINE_REPRODUCIBLE=1` pornește modul reproductibil, `ENGINE_BACKEND=cpu|gpu|auto` este citit de bibliotecă (`corbul` calculează întotdeauna pe procesor).

## Mai mare decât RAM-ul

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"     # MoE: routed experts stream from disk
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --disk-mbps 800 --prompt "…"   # dense: layers stream; speed is bounded by the disk; --disk-mbps (default 500) only feeds the speed estimate it prints
```

## Mai multe calculatoare (mesh în rețeaua locală)

```bash
# on every helper machine (the key needs 8+ characters; or set CORBUL_MESH_KEY)
corbul worker ./my-model --key MESHKEY12345 --bind 0.0.0.0:7471
corbul worker ./my-model --key-file mesh.key --discovery            # also answer UDP discovery probes (opt-in); --discovery-port N; --no-discovery forces it off
# on the main machine
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key MESHKEY12345 --prompt "…"
corbul generate ./my-model --discover --key MESHKEY12345 --prompt "…"      # broadcast a probe, use workers that hold the same model
corbul generate ./my-model --workers h:7471 --key MESHKEY12345 --wire-f16 --local-speed 2.5 --prompt "…"
```

`--wire-f16` înjumătățește traficul stării ascunse (și renunță la identitatea bit cu bit); `--local-speed` îi spune împărțitorului cât de rapid este acest calculator față de workeri; `--discover-target HOST:PORT` și `--discover-wait-ms N` schimbă adresa sondei (implicit `255.255.255.255:47813`) și timpul de așteptare (implicit 2000 ms). Traficul este autentificat, dar **necriptat**: folosește o rețea de încredere sau un VPN. Testat doar pe loopback.

## Ieșire structurată, citări, adaptoare

```bash
corbul generate M --json --prompt "…"                       # valid JSON
corbul generate M --json-schema schema.json --prompt "…"    # a file or inline JSON
corbul generate M --regex '[A-Z]{3}-[0-9]{4}' --prompt "…"  # regular expression
corbul generate M --grammar rules.gbnf --prompt "…"         # GBNF grammar; use only one of the four
corbul generate M --sources notes.txt --sources paper.txt --prompt "Answer with quotes" --citations report.json   # quote-locked
corbul generate M --sources notes.txt --no-lock --prompt "…"   # only check the quotes afterwards
corbul generate M --lora ./adapter:0.8
```

Serverul acceptă `response_format`, `grammar` și `sources` în cereri și oferă `POST /v1/quotes/verify`.

## Conversații lungi, sesiuni, reproductibilitate

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

Chitanțele funcționează cu bucla simplă de generare, nu cu modele MoE sau cu spațiu de stare. Opțional: `--receipt-key K` (vizibil în lista de procese; preferă `--receipt-key-file`), `--receipt-topk N`, `--keep-steps`, `--leaves`.

## Decodare mai rapidă

```bash
corbul generate M --draft ./small-model --draft-len 4 --prompt "…"      # a draft model proposes, M verifies
corbul generate M --lookahead --ngram-min 1 --ngram-max 4 --prompt "…"   # prompt-lookup, no extra model
corbul generate M --self-spec --skip-layers auto --prompt "…"            # M drafts with a subset of its layers; or --skip-layers 4,6,8-11
```

Același răspuns greedy și aceeași distribuție de eșantionare; textul apare la sfârșit. Se aplică doar `--temperature`, `--top-k`, `--top-p`; nu merge cu ieșire structurată, chitanțe, sesiuni sau mesh.

## Căutare, cuantizare, îmbinare, direcționare, regăsire, filigran, limbă

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

Nu orice combinație de flaguri funcționează; fiecare comandă afișează o eroare de utilizare pe o singură linie când nu funcționează. Nimic de aici nu a fost rulat pe un punct de control real.

## Servire

```bash
corbul serve --models-dir ./models --host 127.0.0.1 --port 8080 --token-file token.txt --default-model qwen
corbul serve --models-dir ./models --alias fast=qwen-0.5b --max-loaded 2 --sleep-idle-seconds 600
corbul serve --models-dir ./models --keys-file keys.json --allow-admin --max-kv 8g --max-heavy 4
corbul serve --models-dir ./models --cors "https://app.example" --log-prompts --drain-seconds 10 --config-dir ./cfg
```

Deschide `http://127.0.0.1:8080/` pentru interfața web (și `/ui`). Pagina este publică; fiecare apel pe care îl face cere tokenul. `--token T` este același lucru cu `CORBUL_TOKEN` (vizibil în lista de procese, deci preferă `--token-file`). `--keys-file` conține chei API cu liste de modele și limite de rată pe cheie. `--allow-admin` le permite deținătorilor de chei să folosească `/api/create`, `/api/delete`, `/models/load` și `/models/unload`; implicit le poate folosi doar tokenul principal. `--max-kv` refuză cererile al căror cache KV ar depăși dimensiunea (implicit 8g); `--max-heavy` limitează cererile simultane de tokenizare, de șablon și de verificare (implicit 4). `--no-log` oprește jurnalul cererilor. `GET /v1/capabilities` listează ce funcționează. Rutele de embeddings și rerank răspund 501.

## Indisponibile în `corbul` (încă)

- Modele cu atenție latentă multi-cap (în stil DeepSeek), RWKV în formă GGUF, Zamba/Zamba2, i-quant-uri, modele WordPiece/BERT.
- Intrări de imagine și sunet (biblioteca are blocurile), endpoint-uri de embeddings și rerank, back end-ul GPU.
- Identitatea bit cu bit a prefill-ului cuantizat obișnuit (vezi ENGINE.md, „Exactitate”).
