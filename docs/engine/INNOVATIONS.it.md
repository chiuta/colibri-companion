# Che cosa c'è di diverso in questo motore

[English](INNOVATIONS.md) · [Română](INNOVATIONS.ro.md) · [Français](INNOVATIONS.fr.md) · [Italiano](INNOVATIONS.it.md) · [Español](INNOVATIONS.es.md) · [Português](INNOVATIONS.pt.md) · [Deutsch](INNOVATIONS.de.md) · [Magyar](INNOVATIONS.hu.md)

Note in linguaggio semplice sulle funzioni che la maggior parte degli altri motori locali non ha. Ognuna dice che cosa fa, dove si raggiunge e che cosa non può fare. Tutte sono state provate solo su modelli sintetici; nessuna è stata eseguita dagli autori su un checkpoint reale. «In `corbul`» significa che oggi esiste un comando o un'opzione (`corbul help`); «solo libreria» significa codice Rust senza comando. Vedi [ENGINE.it.md](ENGINE.it.md) e [QUICKSTART.it.md](QUICKSTART.it.md).

## Integrità, hardware e memoria

### Mesh in rete locale

Modulo `mesh`. In `corbul`: `worker`, `generate --workers`, `--discover`, `--key`, `--wire-f16`. Un modello troppo grande per una sola macchina viene diviso in intervalli di livelli, uno per computer. Tra le macchine viaggia solo lo stato nascosto. Le macchine dimostrano di condividere una chiave (HMAC-SHA256 reciproco), ogni frame porta un MAC, le dimensioni dell'handshake sono limitate, un worker morto viene escluso e i suoi livelli ridistribuiti. I logit f32 densi sono uguali a quelli di una singola macchina. La scoperta UDP è opzionale: un worker risponde alle sonde solo se avviato con `--discovery` o `--discovery-port` (`--no-discovery` la forza a spenta), e `--discover` del coordinatore trasmette una sonda e attende le risposte dai worker che hanno lo stesso modello. *Non può:* cifrare (usa una rete locale fidata o una VPN); è stato provato solo in loopback; la velocità rispetto a una singola macchina non è stata misurata.

### Streaming dei livelli

Modulo `stream_layers`. In `corbul`: `--stream-layers`, `--ram-budget`, `--stream-window`, `--disk-mbps` (per le stime). Esegue un modello denso più grande della RAM tenendo in memoria una piccola finestra di livelli e precaricando in un thread in background. I risultati sono uguali a quelli del modello in RAM. *Non può:* evitare che ogni token legga ogni livello, quindi la velocità è limitata da quella del disco; solo livelli densi (i modelli mixture-of-experts leggono invece in streaming i propri esperti instradati, con `--ram-budget`).

### Citazioni vincolate alla fonte

Modulo `quotelock`. In `corbul`: `--sources FILE` (ripetibile), `--no-lock`, `--citations FILE`; il server accetta `sources` e offre `POST /v1/quotes/verify`. Durante la decodifica, il testo tra « e » deve essere un passaggio letterale di una fonte (tollerante a maiuscole, spaziatura, stili di virgolette e segni diacritici rumeni); `[S3]«` limita una citazione alla fonte 3. Un verificatore classifica poi ogni citazione (letterale, fonte sbagliata, troncata, alterata, fusa, non supportata) e dà un punteggio alle frasi fuori dalle citazioni; il rapporto indica quando è stato raggiunto un limite di lavoro (`limit_hit`). *Non può:* rendere vero il testo attorno alle citazioni; le risposte in streaming non contengono un blocco di citazioni.

### Pianificatore, regolatore e contatore di energia

Modulo `governor` (e `plan`, `hw`). In `corbul`: `plan`, `doctor`, `--governor`, `--no-governor`, `--verbose`, `--calibrate`. Prima del caricamento dice se il modello ci sta, va in streaming, gira lentamente o non può girare, con una velocità stimata. In esecuzione, il regolatore legge temperatura, memoria e batteria (Linux) e consiglia thread e pause senza oscillare. *Non può:* dare altro che stime di velocità (`--calibrate` cronometra la tua macchina per circa un secondo); l'energia è un'ipotesi a meno che un contatore RAPL sia leggibile; l'intensità di carbonio (400 g/kWh) è un valore provvisorio.

### Politiche della cache KV e snapshot di sessione

Moduli `kvpolicy`, `snapshot`. In `corbul`: `--kv-policy streaming|h2o|snapkv`, `--kv-budget`, `--kv-sinks`, `--session DIR`, `--session-cap`, `--compress`. Mantengono le chat lunghe entro un budget di memoria (attention sink, heavy hitter, compressione del prompt) e salvano o riprendono una conversazione da un file con checksum, riutilizzando il prefisso salvato più lungo. Gli snapshot possono portare un HMAC a chiave. *Non può:* evitare le perdite (l'espulsione è con perdita per progetto); `--kv-policy` e `--session` non si possono combinare; chat molto lunghe possono superare il contesto di addestramento.

### Modalità riproducibile e ricevute

Moduli `simd` (interruttore), `receipt`. In `corbul`: `--reproducible`, `--receipt FILE`, `--receipt-key`, `--receipt-key-file`, `--receipt-topk`, `--keep-steps`, `--leaves`, `verify-receipt`, `replay`. Una ricevuta registra il modello (un'impronta di Merkle dei pesi), il tokenizer, le impostazioni del campionatore, il prompt e l'output con una catena di hash, sigillata con SHA-256 e una chiave HMAC facoltativa; `replay` la riesegue. I pesi danneggiati vengono localizzati fino a un tensore e a un blocco. *Non può:* dimostrare nulla a un estraneo senza la chiave HMAC o una nuova esecuzione (niente firme a chiave pubblica); i modelli MoE e a spazi di stato sono rifiutati; ARM non provato; la modalità riproducibile costa velocità.

### Strumenti linguistici pensati prima di tutto per il rumeno

Modulo `lang`. In `corbul`: `lang detect`, `lang normalize`, `--lang CODE` (non nel server). Normalizza ș/ț e ş/ţ, ripara il mojibake, identifica 27 lingue, vieta le scritture straniere durante la decodifica (rumeno: niente ş ţ), formatta numeri, importi e date rumeni, ripristina i segni diacritici mancanti segnalando ogni parola incerta. *Non può:* andare oltre un lessico di circa 650 forme per il ripristino, non misurato su un corpus; le parole ambigue sono risolte da poche regole di contesto; l'identificazione della lingua richiede circa cinque parole.

### Analisi dell'incertezza

Modulo `confidence`. Solo libreria (esempio dimostrativo `confidence_demo`). Statistiche per token, parola, frase e risposta per una mappa di calore, un'euristica trasparente del rischio di allucinazione, punteggi di auto-coerenza e strumenti di calibrazione. *Non può:* essere altro che un'euristica: il punteggio di rischio non è un rilevatore di verità.

## Calcolo in fase di inferenza, messa a punto, recupero

### Ricerca e lookahead

Modulo `search`. In `corbul`: `search --method beam|best-of-n|vote` (`--width`, `--samples`, `--answer number|text`, `--length-penalty`, `--all`, `--explain`, `--json`) e `--lookahead` su `generate` (`--ngram-min`, `--ngram-max`). Beam search, best-of-N, votazione per auto-coerenza, un albero di conversazione (libreria), decodifica speculativa per ricerca nel prompt. *Non può:* aiutare se il testo non ripete testo precedente (il lookahead); il best-of-N richiede un buon valutatore.

### Decodifica speculativa

Moduli `spec`, `selfspec`. In `corbul`: `--draft MODEL`, `--lookahead`, `--self-spec`, `--skip-layers auto|LIST`, `--draft-len`. Stessa risposta greedy e stessa distribuzione di campionamento con meno passaggi completi; l'output compare alla fine. Solo `--temperature`, `--top-k`, `--top-p`; non con output strutturato, ricevute, sessioni o mesh. *Non può:* promettere un'accelerazione; l'auto-speculazione ricade su passi semplici per gli anelli a finestra scorrevole che si riavvolgerebbero, le cache compattate e i livelli scaricati.

### Quantizzazione mista automatica e quantizzatore

Moduli `autoquant`, `quantize`. In `corbul`: `autoquant FOLDER --budget 4.5bpw|6g|Q4_K` (`--recipe`, `--report`, `--candidates`, `--base`, `--imatrix`, `--sample-rows`) e `quantize IN OUT.gguf --type NAME | --recipe FILE`. Misura l'errore sui pesi per gruppo di tensori in ciascun formato e sceglie i formati entro un budget di dimensione (zaino esatto o greedy). *Non può:* stimare la qualità delle risposte, solo l'errore sui pesi; presume che i danni si sommino; le i-quant non esistono.

### Vettori di guida e logit lens

Modulo `steer`. In `corbul`: `steer train|apply|lens` (`--positive`, `--negative`, `--method mean-diff|pca`, `--out`, `--vector`, `--strength`, `--layers`, `--positions`, `--no-attention`). Apprende una direzione da due insiemi di prompt e la aggiunge al flusso residuo senza modificare i pesi; la lente mostra che cosa prevedrebbe ciascun livello. *Non può:* funzionare con modelli a spazi di stato, output strutturato, speculazione, ricevute o sessioni; un vettore ottenuto da un fine-tuning viene accettato da un altro con la stessa forma.

### Recupero locale

Modulo `rag`. In `corbul`: `rag add|search|ask` (`--query`, `--question`, `--k`, `--chunk-chars`, `--overlap`, `--embed-model`). Suddivisione adatta al rumeno, indice HNSW in Rust puro, BM25 con ripiegamento dei segni diacritici, fusione dei ranghi, passaggi numerati `[S1]…` che alimentano le citazioni vincolate alla fonte, file con checksum. *Non può:* fare di meglio dell'embedder predefinito, un embedder a hash (parole e parti di parole, non significato); `--embed-model` usa gli stati nascosti di un modello generativo (debole); la CLI reindicizza a ogni chiamata e legge testo semplice, non file PDF o Office (la libreria ha funzioni di supporto per l'ingestione).

### Fusione di modelli

Modulo `merge`. In `corbul`: `merge CONFIG.json` (`--out`, `--model`, `--dry-run`, `--max-memory`, `--overwrite`, `--verbose`). Lineare, SLERP, aritmetica dei compiti, TIES, DARE (con seme) e porzioni di livelli, un tensore alla volta. Forme e intervalli sono controllati prima della scrittura; i link simbolici non vengono seguiti. *Non può:* garantire la qualità; scrive solo pesi safetensors, mai quantizzati, e `config.json` e tokenizer li copi tu.

### Filigrana di testo

Modulo `watermark`. In `corbul`: `generate --watermark KEY` (`--wm-gamma`, `--wm-delta`, `--wm-context`) e `watermark detect MODEL --key K --text T | --file F` (`--no-dedup`). Un bias a «lista verde» con chiave e un rilevatore a punteggio z. *Non può:* fornire una prova; testi brevi, codice e citazioni portano poco segnale; la parafrasi lo rimuove; chi ha la chiave può falsificarlo; l'assenza di marchio non prova una scrittura umana.

## Multimodale (solo libreria)

### Visione

Moduli `vision`, `imageio`. Un encoder in stile CLIP/SigLIP, un proiettore, la suddivisione in tessere e un aggancio `Model::forward_segments` per inserire embedding in un prompt; decoder per PNG, JPEG (baseline e progressivo), BMP e PNM scritti dalle specifiche, con limiti di dimensione. *Non può:* essere usato da `corbul` (solo testo); nessun checkpoint reale è stato eseguito.

### Audio

Modulo `audio`. Lettura e scrittura WAV, ricampionamento, FFT, caratteristiche log-mel, rilevamento dell'attività vocale, un esecutore con struttura Whisper. *Non può:* essere usato da `corbul`; nessun checkpoint Whisper reale è stato eseguito.

## Spazi di stato, ibridi e GPU (vedi ENGINE.md)

I moduli `ssm`, `hybrid`, `mla` e il crate `companion-engine-gpu` sono descritti, con i loro limiti, in ENGINE.md.
