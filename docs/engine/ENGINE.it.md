# Il motore «clean-room» (`companion-engine` e `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Altri documenti: [INNOVATIONS.it.md](INNOVATIONS.it.md) · [QUICKSTART.it.md](QUICKSTART.it.md) · [CLEANROOM.it.md](CLEANROOM.it.md)

**Stato: alfa. Tutto ciò che segue è stato provato solo su piccoli checkpoint sintetici.** Gli autori non hanno fatto girare in questo motore nessun modello reale di Hugging Face (l'ambiente di compilazione non raggiungeva huggingface.co), nessuna GPU ne ha eseguito alcuna parte e i percorsi ARM, macOS e Windows non sono mai stati compilati qui. Leggi la tabella «verificato e non verificato» prima di fidarti di qualsiasi affermazione.

## Che cos'è

Un motore di inferenza locale per modelli linguistici, scritto da zero in Rust, a partire da descrizioni pubbliche di formati di file e da articoli pubblicati. Ha tre parti:

| Crate | Che cos'è |
|---|---|
| `crates/engine` (`companion-engine`) | La libreria: formati di file, tokenizer, quantizzazione, passaggi in avanti, campionamento e gli strumenti extra descritti in INNOVATIONS.it.md. Dipendenze: solo `serde_json` e `memmap2`. |
| `crates/engine-cli` (`corbul`) | Una riga di comando e un server locale compatibile con OpenAI / Ollama / Anthropic, costruiti sopra la libreria. |
| `crates/engine-gpu` (`companion-engine-gpu`) | Kernel WGSL per il crate `wgpu`, dietro una funzione disattivata per impostazione predefinita. Non usato da `corbul`. |

È separato dall'app desktop Colibri Companion e non la sostituisce: l'app continua a pilotare colibri di JustVugg. Collegare questo motore all'app come backend alternativo è pianificato, non fatto.

## Che cosa supporta

**File.** GGUF v2/v3 (intestazione e tabella dei tensori si leggono senza toccare i pesi, quindi un file da 400 GB si ispeziona in millisecondi); uno scrittore GGUF in streaming per modelli più grandi della RAM; safetensors, in un file o suddiviso con indice; `config.json` e `tokenizer.json` di Hugging Face; uno scaricatore di modelli (tramite `curl`, con ripresa e controllo SHA-256).

**Quantizzazione.** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 e i K-quant Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K si possono decodificare, e per tutti esistono codificatori (con pesatura tramite matrice di importanza), conversione Hugging Face → GGUF e riquantizzazione GGUF. I pesi restano quantizzati in memoria; il prodotto matrice–vettore non costruisce mai la matrice. Gli i-quant (IQ*) non sono implementati.

**Architetture.**

| Famiglia | Stato |
|---|---|
| Decoder densi: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 e simili (norma RMS o di strato, GQA/MQA, RoPE con scalatura lineare / Llama-3 / YaRN e rotazione parziale, QK-norm, bias, soft-capping, finestre scorrevoli, MLP a porta o semplice) | Implementate; caricate da `corbul` |
| Mixture of experts: Mixtral, Qwen-MoE, esperti impilati GGUF, esperti condivisi, esperti instradati letti dal disco con cache LRU | Implementate; caricate da `corbul` |
| Attenzione latente multi-testa (stile DeepSeek-V2/V3, Kimi-K2) | Solo modulo di libreria; il registro delle architetture la segna ancora «pianificata» e `corbul` non la carica |
| Mamba-1, Mamba-2, RWKV-6, RWKV-7 | Solo modulo di libreria; idem |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, modelli encoder ed encoder-decoder (T5, BERT, Whisper…), LongRoPE | Non supportati; `corbul` lo dice invece di tirare a indovinare |
| Ingressi visivi e audio | Non supportati (solo testo) |

**Percorsi hardware.**

- CPU: kernel scelti a runtime (scalare, AVX2, AVX-512, VNNI su x86-64; NEON su aarch64), un pool di thread persistente adatto a core prestazionali/efficienti misti, rilevamento di hardware e NUMA e un pianificatore che decide cosa resta in RAM.
- GPU: kernel WGSL controllati con `naga` e rispetto a un'emulazione in Rust. Non sono mai girati su una GPU.
- Modelli più grandi della RAM: esperti dal disco (MoE), streaming degli strati (denso), una mesh LAN su più macchine.

## Verificato e non verificato

| Area | Stato |
|---|---|
| Test unitari, di integrazione e fuzz su checkpoint sintetici (Linux, x86-64) | Verificato. Centinaia di test passano; i file ostili o troncati restituiscono errori, non crash. |
| Checkpoint reali di Hugging Face (qualsiasi famiglia) | **Non eseguito.** L'ambiente non raggiungeva huggingface.co. |
| Tokenizer su veri file `tokenizer.json` | Scritto sulle descrizioni pubbliche del formato; esiste un banco di prova (`ENGINE_VOCAB_DIR`) ma non l'abbiamo lanciato su file reali. |
| Permutazione delle righe Q/K in stile llama.cpp nei file GGUF | Implementata dalla descrizione; non verificata su file reali. |
| Quantizzatori e dequantizzatori | Verificati su blocchi calcolati a mano e andata e ritorno; non su file scritti da altri strumenti. |
| Esecuzione su GPU | **Non eseguita.** I kernel sono validati solo con `naga` e un'emulazione in Rust. |
| Percorsi ARM / NEON, macOS, Windows | **Mai compilati né eseguiti qui.** Le sonde sono provate su testo registrato da quei sistemi. |
| Formule Mamba / RWKV / MLA | Scritte da articoli e dal ricordo delle descrizioni di riferimento, provate solo per coerenza interna. |
| Mesh su una rete vera | Provata solo in loopback. |
| Velocità | **Nessun numero affidabile.** La macchina di compilazione aveva due core condivisi; le misure erano rumorose. Tutte le velocità dell'autopilota sono stime. |
| Modalità riproducibile | Identica su cinque varianti di kernel x86 e più numeri di thread; ARM non provato. |
| Licenze delle dipendenze | Controllate con `cargo-deny` a un commit precedente (b496ba5); le aggiunte successive non sono ancora state ricontrollate. |

## Dichiarazione clean-room

Nessun codice di colibri, llama.cpp o altri motori di inferenza è stato letto o copiato scrivendo questo motore. Formati e algoritmi vengono da specifiche pubbliche, articoli e strutture di file. Vedi [CLEANROOM.it.md](CLEANROOM.it.md) per il processo e i suoi limiti.

## Confronto con altri motori locali

Non abbiamo eseguito alcun confronto diretto, quindi nessuna affermazione su velocità o qualità. In modo qualitativo:

- Motori consolidati come llama.cpp, Ollama, mistral.rs o vLLM sono stati eseguiti da molte persone su migliaia di checkpoint reali, hanno backend GPU funzionanti, coprono più architetture e formati di quantizzazione (inclusi gli i-quant) e gestiscono i modelli di visione. Questo motore non ha ancora nulla di tutto ciò. Se ti serve qualcosa che funzioni oggi su un grande modello preciso, usa uno di quelli.
- Che cosa cerca di fare diversamente questo motore: strumenti di integrità e onestà incorporati (citazioni bloccate sulle fonti, modalità riproducibile con ricevute sigillate da catena di hash e HMAC facoltativo, analisi dell'incertezza, un autopilota che dice «non girerà» invece di andare in crash), strumenti linguistici con il rumeno al primo posto, esecuzione di modelli più grandi della tua RAM o di una sola macchina, una piccola base di codice Rust con due dipendenze e solo licenze permissive.
- Qualsiasi affermazione che questo motore sia più veloce o più preciso di un altro sarebbe non verificata. Misura sul tuo hardware e faccelo sapere.

## Mappa della libreria

Formati e caricamento: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`. Modello: `arch`, `model`, `moe`, `mla`, `ssm`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Esecuzione: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Funzioni distintive: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Ultima ondata: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark` — implementate nella libreria e coperte da test su modelli sintetici, ancora in rifinitura, non ancora raggiungibili da `corbul`, e le loro API possono cambiare.
