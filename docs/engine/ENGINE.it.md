# Il motore clean-room (`companion-engine` e `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Altri documenti: [INNOVATIONS.it.md](INNOVATIONS.it.md) · [QUICKSTART.it.md](QUICKSTART.it.md) · [CLEANROOM.it.md](CLEANROOM.it.md) · [CHANGES-SECURITY.it.md](CHANGES-SECURITY.it.md)

**Stato: alpha.** Tutto ciò che segue è stato provato su piccoli checkpoint sintetici, su file di riferimento del tokenizer e su testi di esempio registrati. **Nessun modello reale scaricato da Hugging Face è stato eseguito con questo motore dai suoi autori**, nessuna GPU ne ha eseguito alcuna parte, e i percorsi di codice per ARM, macOS e Windows non sono stati eseguiti qui. Leggi «Verificato e non verificato» prima di fidarti di un'affermazione. `corbul capabilities` stampa gli stessi limiti dal binario stesso.

## Che cos'è

Un motore di inferenza locale per modelli linguistici, scritto da zero in Rust a partire da descrizioni pubbliche di formati di file e da articoli pubblicati.

| Crate | Che cos'è |
|---|---|
| `crates/engine` (`companion-engine`) | La libreria: formati di file, tokenizer, quantizzazione, passaggi in avanti, campionamento e gli strumenti aggiuntivi di INNOVATIONS.md. Dipendenze a runtime: solo `serde_json` e `memmap2`. |
| `crates/engine-cli` (`corbul`) | La riga di comando e un server locale compatibile con OpenAI / Ollama / Anthropic con un'interfaccia web integrata. Dipende dalla libreria e da `serde_json`. |
| `crates/engine-gpu` (`companion-engine-gpu`) | Kernel WGSL e un'emulazione su CPU degli stessi. L'esecutore su dispositivo (`wgpu`, `pollster`) sta dietro la feature cargo `wgpu`, disattivata per impostazione predefinita. **`corbul` non gira su di esso; compilato con `--features gpu`, `corbul doctor` stampa soltanto il suo rapporto sulle capacità.** |

È separato dall'app desktop Colibri Companion e non la sostituisce: l'app continua a pilotare il colibri di JustVugg. Usare questo motore come back end alternativo dell'app è previsto, non fatto.

## Che cosa supporta

**File.** GGUF v2 e v3 (l'intestazione e la tabella dei tensori si leggono senza toccare i pesi); uno scrittore GGUF in streaming per output più grandi della RAM; safetensors, singoli o suddivisi con un indice; `config.json`, `tokenizer.json` e template di chat di Hugging Face; un downloader di modelli (modulo `hub`, tramite `curl`, con `wget` come ripiego anonimo; ripresa, controlli SHA-256, download fissati a un commit, limiti di dimensione, rifiuto dei reindirizzamenti verso http in chiaro).

**Quantizzazione.** Decodificati: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ4_NL, IQ4_XS (16 tipi). Esistono encoder per gli stessi tipi, con ponderazione tramite matrice di importanza e ricette per singolo tensore. I pesi restano quantizzati in memoria; il prodotto matrice–vettore non espande mai l'intera matrice. **Non supportati:** le altre i-quant (IQ1, IQ2, IQ3), le cui tabelle di ricerca non sono riprodotte qui, e qualsiasi tipo non presente nell'elenco. IQ4_NL e IQ4_XS sono stati scritti dalla descrizione del formato e verificati con blocchi costruiti a mano; nessun file prodotto da un altro strumento è stato letto.

**Architetture.**

| Famiglia | Stato in `corbul` |
|---|---|
| Decoder densi: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 e simili (normalizzazione RMS o layer norm, GQA/MQA, RoPE con scalatura lineare, Llama-3, YaRN e LongRoPE e rotazione parziale, QK-norm, bias, soft-capping, finestre scorrevoli, MLP con gate o semplice) | Caricati ed eseguiti (template di chat, serve, tutte le opzioni) |
| Mixture of experts: Mixtral, Qwen-MoE, esperti impilati GGUF, esperti condivisi, esperti instradati letti in streaming dal disco con una cache LRU | Caricati ed eseguiti |
| Spazi di stato: Mamba-1, Mamba-2, RWKV-6, RWKV-7 (modulo `ssm`) | Eseguiti da `generate` e `chat` come **continuazione di testo semplice**: niente template di chat, niente output strutturato, niente `serve`. RWKV in forma GGUF non è supportato (solo safetensors) |
| Ibridi: Jamba e Bamba (modulo `hybrid`) | Come la riga degli spazi di stato. Zamba e Zamba2 non sono supportati |
| Attenzione latente multi-testa (DeepSeek-V2/V3, Kimi-K2; modulo `mla`) | **Solo libreria.** Il blocco esiste ed è provato per coerenza interna, ma `corbul` non assembla un modello simile (`corbul capabilities` risponde no) |
| Encoder di visione (ViT, CLIP, SigLIP; modulo `vision`, decoder in `imageio`) e front-end vocale più un esecutore con struttura Whisper (modulo `audio`) | **Solo libreria.** `corbul` accetta solo testo |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, modelli encoder-decoder (T5…), encoder RoBERTa/XLM-R e altri encoder non BERT | Non supportati; `corbul` lo dice invece di tirare a indovinare |
| Encoder della famiglia BERT (BERT, MiniLM, bge, e5, cross-encoder ms-marco; moduli `bert`, `wordpiece`) | `corbul embed`, `corbul rerank`, `corbul serve --encoder`. Solo cartelle Hugging Face. Verificati contro un riferimento f64 semplice, e il tokenizer WordPiece contro il riferimento BERT di llama.cpp (46 casi su 46); nessun checkpoint reale è stato eseguito |
| Endpoint di embedding (`/v1/embeddings`, `/api/embed`, `/api/embeddings`) | Funzionano: stati nascosti finali aggregati (media, ultimo o primo), lunghezza unitaria per impostazione predefinita |
| Endpoint di reranking | Rispondono 501 (non implementati) |

**Tokenizer.** Modelli BPE a livello di byte e in stile SentencePiece, regole di pre-tokenizzazione per 12 famiglie, token aggiunti, ripiego sui byte, template da `tokenizer_config.json` e dai metadati GGUF. WordPiece (famiglia BERT) ha il proprio tokenizer nel modulo `wordpiece`: `tokenizer.json` Hugging Face e vocabolari GGUF `bert`, 46 casi su 46 uguali al riferimento BERT di llama.cpp.

**Campionamento.** Temperatura, top-k, top-p, min-p, typical-p, penalità di ripetizione / presenza / frequenza, Mirostat 1 e 2, DRY, XTC, una specifica JSON del campionatore, casualità con seme; decodifica vincolata da JSON, JSON Schema, espressione regolare o grammatica GBNF (con budget di lavoro perché una grammatica ostile non possa bloccare il processo); bias dei logit e stringhe di arresto tramite il server. Decodifica speculativa: modello bozza, ricerca nel prompt, auto-speculazione (vedi INNOVATIONS.md).

**Server.** `corbul serve` offre `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/messages` e `/v1/messages/count_tokens` (stile Anthropic), gli endpoint Ollama `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/create`, `/api/delete`, `/api/ps`, `/api/version`, la gestione dei modelli `/models`, `/models/load`, `/models/unload`, oltre a `/metrics`, `/props`, `/slots`, `/tokenize`, `/detokenize`, `/apply-template`, `/health`, `/v1/capabilities`, `/v1/quotes/verify` e l'interfaccia web su `/` e `/ui` (otto lingue). La chiamata di strumenti è supportata tramite i template di chat. Le rotte di embedding (`/v1/embeddings`, `/api/embed`, `/api/embeddings`) restituiscono un vettore per ogni input, ricavato dagli stati nascosti normalizzati dell'ultimo strato; il corpo accetta `input` (una stringa, id di token o una lista di entrambi; al massimo 64 input e 65.536 token), `pooling` (`mean` predefinito, `last`, `first`), `normalize` (predefinito true), `dimensions`, `truncate` e `encoding_format` (`float` o `base64`). Sono pensate per modelli decoder, i cui vettori sono utilizzabili ma non addestrati per il recupero. Con `--encoder CARTELLA` (una cartella della famiglia BERT) le stesse rotte rispondono da quell'encoder quando `model` è il nome della cartella o non è caricato alcun decoder, con l'aggregazione con cui è stato addestrato (`pooling` può essere `cls` o `mean`); `/v1/rerank` e `/rerank` valutano allora `documents` rispetto a `query` con un cross-encoder (`top_n`, `return_documents`). Senza `--encoder` le rotte di rerank rispondono 501; un encoder senza testa di classificazione risponde 400. Le risposte in streaming non contengono un blocco di citazioni.

**Hardware.** I kernel CPU sono scelti a runtime: scalare, avx2+fma, avx2+fma+avxvnni, avx512, avx512+vnni (x86-64), neon e neon+dotprod (aarch64); `ENGINE_SIMD` ne impone uno. Un pool di thread persistente, il rilevamento di hardware e NUMA e un pianificatore (`corbul plan`, `corbul doctor`) decidono che cosa resta in RAM. I kernel GPU non sono mai stati eseguiti su una GPU (vedi sotto).

## Esattezza

- **Pesi f32 densi:** i percorsi a lotti e a singolo token, il mesh, lo streaming dei livelli e le cache producono logit identici bit per bit.
- **Pesi quantizzati:** l'identità bit per bit tra valutazione a lotti e a singolo token vale solo quando `exact_batch` è attivo. La verifica, i giri di verifica auto-speculativi e l'esecutore a lotti lo attivano. Il prefill ordinario usa un percorso più veloce (dequantizza e poi suddividi in tessere) e differisce dalla valutazione a singolo token solo per rumore di arrotondamento.
- **Modalità riproducibile** (`--reproducible` o `ENGINE_REPRODUCIBLE=1`): risultati identici tra le varianti di kernel x86 e i numeri di thread che abbiamo potuto provare. ARM non è stato provato. Costa velocità.
- Mirostat 1 usa nella sua stima il numero di candidati rimasti, non la dimensione del vocabolario (scostamento documentato).

## Verificato e non verificato

| Area | Stato |
|---|---|
| Test unitari, di integrazione, avversariali e di fuzzing su checkpoint sintetici (Linux x86-64). I numeri cambiano a ogni commit; esegui `cargo test --workspace` per il valore attuale | Superati nell'ultima esecuzione completa. I file ostili e troncati restituiscono errori. |
| Tokenizer contro i file di riferimento di llama.cpp, 12 famiglie | 46 casi su 46 identici (file di riferimento scaricati da GitHub; `crates/engine/tests/tokenizer_real.rs`, `ENGINE_VOCAB_DIR`). Solo il tokenizer, non modelli interi. |
| Modello Hugging Face reale tramite `corbul` (qualsiasi famiglia) | **Non eseguito dagli autori.** Il sandbox di compilazione non poteva raggiungere huggingface.co. Un job CI manuale esegue un minuscolo GGUF llama (`stories260K`) tramite la CLI; non è ancora stato eseguito |
| Permutazione delle righe Q/K in stile llama.cpp, varianti dei metadati GGUF | Implementate da descrizioni pubbliche; non controllate su file reali |
| Quantizzatori e dequantizzatori | Blocchi calcolati a mano e andata e ritorno; non confrontati con file scritti da altri strumenti |
| Esecuzione su GPU | **Non eseguita.** I kernel sono validati solo da `naga` e da un'emulazione in Rust; `corbul` non li usa |
| ARM / NEON, macOS, Windows | **Non eseguiti qui.** Le sonde sono provate su testi di esempio registrati. Il workflow CI ha una matrice per questi sistemi ma non è mai stato eseguito |
| Mamba, RWKV, Jamba/Bamba, MLA, visione, audio | Solo test di coerenza interna; nessun checkpoint reale è stato usato |
| Mesh | Solo loopback; autenticato, non cifrato; nessuna misura di velocità |
| Velocità | **Nessun numero affidabile.** La macchina di compilazione aveva due core condivisi. Le velocità dell'autopilota sono stime |
| Licenze delle dipendenze | Controllate con `cargo-deny` al commit b496ba5; le modifiche successive non sono state ricontrollate |
| Traduzioni dei documenti | Assistite da macchina; non ancora riviste da madrelingua |

## Variabili d'ambiente

| Variabile | Effetto |
|---|---|
| `COLIBRI_THREADS` | Dimensione del pool di thread globale (predefinito: core utilizzabili). Letta per prima dalla libreria |
| `ENGINE_SIMD` | Impone una variante di kernel per nome (benchmark, segnalazioni di bug) |
| `ENGINE_REPRODUCIBLE` | `1`, `true` o `on`: modalità riproducibile fin dall'avvio |
| `ENGINE_BACKEND` | `cpu`, `gpu` o `auto` (non impostata = auto). `corbul` calcola comunque sulla CPU |
| `ENGINE_GPU`, `ENGINE_GPU_API`, `ENGINE_GPU_SOFTWARE` | Scelta dell'adattatore per il crate GPU: indice, API grafica, consentire un adattatore software. Non usate da `corbul` |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Token di accesso per i repository con accesso limitato (`download`, `inspect owner/repo`) |
| `HF_ENDPOINT` | Mirror dell'host Hugging Face |
| `CORBUL_TOKEN` | Token bearer condiviso di `corbul serve` (equivale a `--token`) |
| `CORBUL_MESH_KEY` | Chiave del mesh per `worker` e `--workers`/`--discover` (equivale a `--key`) |
| `CORBUL_CONFIG_DIR` | Dove sono conservati i Modelfile creati da `POST /api/create` (equivale a `--config-dir`; predefinito `~/.config/corbul`) |
| `NUMBER_OF_PROCESSORS`, `PROCESSOR_IDENTIFIER` | Lette su Windows dalla sonda hardware; non le imposti tu |
| `ENGINE_VOCAB_DIR`, `ENGINE_VOCAB_ONLY`, `ENGINE_REQUIRE_VOCAB`, `ENGINE_REQUIRE_TOOLS`, `ENGINE_PRINT_GOLDEN`, `ADVERSARIAL_N` | Controlli della suite di test: cartella dei vocabolari, restrizione delle famiglie, far fallire un test saltato, far fallire l'assenza di `curl`/`wget`, rigenerare gli hash di riferimento, dimensione dell'esecuzione avversariale |

## Dichiarazione clean-room

Nessun codice di colibri, di llama.cpp o di qualsiasi altro motore di inferenza è stato letto o copiato mentre si scriveva questo motore. I formati e gli algoritmi derivano da specifiche pubbliche, articoli e strutture di file. Il modello che ha scritto il codice è stato addestrato su testi pubblici che possono includere tali progetti; vedi [CLEANROOM.it.md](CLEANROOM.it.md) per il processo e i suoi limiti.

## Confronto con altri motori locali

Non abbiamo eseguito alcun benchmark diretto, quindi qui non si fa alcuna affermazione su velocità o qualità.

- I motori affermati (llama.cpp, Ollama, mistral.rs, vLLM) sono stati eseguiti su migliaia di checkpoint reali, hanno back end GPU funzionanti, coprono più architetture e formati di quantizzazione, i-quant incluse, e gestiscono i modelli di visione. Questo motore non ha ancora nulla di tutto ciò come prova. Se ti serve qualcosa che funzioni oggi su un modello grande specifico, usa uno di quelli.
- Che cosa cerca di fare diversamente questo motore: strumenti di integrità incorporati (citazioni vincolate alla fonte, modalità riproducibile con ricevute, analisi dell'incertezza, un pianificatore che dice «questo non girerà» invece di andare in crash), strumenti linguistici pensati prima di tutto per il rumeno, modelli più grandi della tua RAM o della tua macchina, una piccola base di codice Rust con due dipendenze di libreria e solo licenze permissive.

## Mappa della libreria

Formati e caricamento: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`, `wordpiece`, `imageio`. Modelli: `arch`, `model`, `moe`, `mla`, `ssm`, `hybrid`, `bert`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Esecuzione: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Funzioni distintive: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Calcolo in fase di inferenza, messa a punto, recupero: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark`. Multimodale: `vision`, `audio`. Quali di questi moduli `corbul` raggiunge è elencato in INNOVATIONS.md; le API della libreria possono ancora cambiare.
