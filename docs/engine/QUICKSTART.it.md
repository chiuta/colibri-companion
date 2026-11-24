# Avvio rapido: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` è la riga di comando del motore clean-room ([ENGINE.it.md](ENGINE.it.md)). Questa pagina elenca solo i comandi e le opzioni che esistono in `crates/engine-cli` al momento della stesura; la riga di comando è ancora in estensione, quindi l'ultima parola spetta a `corbul help`. Alcune opzioni qui sotto sono lette dal codice ma non ancora stampate da `corbul help`; sono segnate *(non ancora nell'aiuto)*. Ciò che non esiste ancora è segnato **pianificato**.

**Avviso onesto:** `corbul` è stato provato solo su minuscoli modelli sintetici. Aspettati asperità sui checkpoint reali e segnalale.

## Compilare

```bash
cargo build --release -p colibri-rs-cli     # produce target/release/corbul
```

Un modello è o una cartella con `config.json` + `*.safetensors` + `tokenizer.json`, oppure un singolo file `.gguf`.

## Guarda, scarica, esegui

```bash
corbul inspect  ./my-model                  # o file.gguf o owner/repo[@rev]: architettura, dimensione, supporto
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors; HF_TOKEN per i repository ad accesso limitato, HF_ENDPOINT per uno specchio
corbul generate ./my-model --prompt "Ciao" --max-tokens 64
corbul chat     ./my-model --system "Sii conciso."
corbul serve    ./my-model --port 8080      # compatibile OpenAI / Ollama / Anthropic, legato per impostazione predefinita a 127.0.0.1
```

Opzioni comuni a `generate`, `chat` e `serve`: `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--kv f32|f16|q8`, `--no-mmap`, `--threads N`, `--ctx N`, `--model-name`, `--quiet`. `generate` accetta anche `--prompt-file F` e `--raw` (senza template di chat). `corbul run` è un alias di `generate`.

Servire più modelli: `--models-dir DIR`, `--alias NOME=MODELLO`, `--max-loaded N`, `--sleep-idle-seconds S`, `--token T` (o `CORBUL_TOKEN`), `--keys-file FILE`, `--cors ORIGINI`, `--log-prompts`, `--no-log`, `--drain-seconds S`, `--config-dir DIR`.

## Girerà su questa macchina?

```bash
corbul doctor                # che cosa vede il motore: set di istruzioni CPU, kernel, RAM, disco, sensori, contatore di energia
corbul plan ./my-model --context 8192            # verdetto: FITS / TIGHT / STREAMS / TOO SLOW / WON'T RUN, velocità, consigli
corbul plan ./my-model --ram-budget 8g --calibrate   # fai finta di avere 8 GB; prima cronometra la macchina per un secondo
corbul capabilities [--json]                     # che cosa questa versione può e non può fare
```

Le velocità sono stime, non misure del tuo modello.

## Più grande della RAM

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"                        # MoE: gli esperti instradati vengono letti dal disco
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --prompt "…"   # denso: gli strati scorrono (limitato dal disco; --stream-window non ancora nell'aiuto)
```

## Più macchine (mesh LAN)

```bash
# su ogni macchina di supporto (la chiave ha almeno 8 caratteri; oppure imposta CORBUL_MESH_KEY)
corbul worker ./my-model --key SECRET --bind 0.0.0.0:7471          # --no-discovery salta la scoperta UDP
# sulla macchina principale
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key SECRET --prompt "…"
corbul generate ./my-model --discover --key SECRET --prompt "…"      # trova i worker nella LAN
```

Il traffico è autenticato ma **non cifrato**: usa una rete fidata o una VPN. Provato solo in loopback.

## Output strutturato, citazioni, adattatori

```bash
corbul generate M --json --prompt "…"                 # output JSON valido
corbul generate M --json-schema schema.json --prompt "…"      # file o JSON in linea; anche --regex MODELLO, --grammar FILE (GBNF); usane una sola delle quattro; non ancora nell'aiuto
corbul generate M --sources notes.txt --sources paper.txt --prompt "Rispondi con citazioni" --citations report.json   # citazioni bloccate
corbul generate M --lora ./adapter:0.8                # adattatore PEFT o GGUF con un fattore
```

Il server accetta `response_format`, `grammar` e `sources` e offre `POST /v1/quotes/verify`.

## Conversazioni lunghe, sessioni, riproducibilità

```bash
corbul generate M --session ./sessions --prompt "…"             # riusa il più lungo prefisso di prompt salvato (non con --kv-policy)
corbul generate M --kv-policy h2o --kv-budget 2048 --prompt "…" # streaming | h2o | snapkv; con perdita per costruzione
corbul generate M --governor --prompt "…"                       # si adatta a temperatura, memoria e batteria (Linux)
corbul generate M --reproducible --temperature 0 --prompt "hi" --receipt r.json --receipt-key SECRET
corbul verify-receipt r.json --key SECRET --model M            # controlla il sigillo (e i pesi se dai --model)
corbul replay r.json --model M --key SECRET                    # riesegue la generazione e confronta
```

Le ricevute funzionano con il ciclo di generazione semplice, non con i modelli MoE. Facoltativi: `--receipt-topk N`, `--keep-steps`, `--leaves` *(non ancora nell'aiuto)*.

## Pianificato (non ancora in `corbul`)

- Il caricamento di modelli Mamba / RWKV / MLA (stile DeepSeek), anche se la libreria ha i blocchi.
- Gli strumenti per la lingua rumena, l'analisi dell'incertezza, ricerca e lookahead, decodifica auto-speculativa, vettori di controllo, recupero, fusione e filigrana (vedi [INNOVATIONS.it.md](INNOVATIONS.it.md)).
- Gli endpoint di embedding e rerank (il server risponde «not implemented»), gli ingressi visivi e audio.
- Il backend GPU e un'interfaccia web.
