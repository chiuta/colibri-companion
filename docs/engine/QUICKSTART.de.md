# Schnellstart: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` ist die Kommandozeile der Clean-Room-Engine ([ENGINE.md](ENGINE.de.md)). Jeder Befehl und jedes Flag auf dieser Seite existiert in `crates/engine-cli` und wird von `corbul help` ausgegeben; ein Skript (`tools/check_engine_docs.py`) schlägt fehl, wenn der Code ein Flag, einen Unterbefehl oder eine Umgebungsvariable hat, die diese Seite nicht erwähnt. `corbul help` bleibt das letzte Wort.

**Ehrliche Warnung:** `corbul` wurde nur mit winzigen synthetischen Modellen und Tokenizer-Referenzdateien getestet. Die Autoren haben kein echtes heruntergeladenes Modell ausgeführt. Rechne mit Ecken und Kanten und melde sie bitte. `corbul capabilities` gibt aus, was funktioniert und was nicht.

## Bauen

```bash
cargo build --release -p colibri-rs-cli     # erzeugt target/release/corbul
```

Ein Modell ist ein Ordner mit `config.json` + `*.safetensors` + `tokenizer.json` oder eine einzelne `.gguf`-Datei.

## Ansehen, herunterladen, ausführen

```bash
corbul inspect  ./my-model                  # oder file.gguf oder owner/repo[@rev]: Architektur, Größe, Unterstützung
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors
corbul generate ./my-model --prompt "Hello" --max-tokens 64
corbul generate ./my-model --prompt-file question.txt --raw   # --raw: kein Chat-Template
corbul chat     ./my-model --system "You are concise."
corbul run      ./my-model --prompt "Hello"                   # `run` ist `generate`
corbul serve    ./my-model --port 8080                        # standardmäßig an 127.0.0.1 gebunden
```

Setze `HF_TOKEN` (oder `HUGGING_FACE_HUB_TOKEN`) für geschützte Repositories und `HF_ENDPOINT` für einen Spiegel. Downloads brauchen `curl` (anonymes `wget` ist ein Ersatz).

Gemeinsame Optionen für `generate`, `chat` und `serve`: `--max-tokens N`, `--temperature T`, `--top-p P`, `--top-k K`, `--seed S`, `--kv f32|f16|q8`, `--no-mmap`, `--model-name NAME`, `--threads N` (auch `COLIBRI_THREADS`), `--ctx N`, `--quiet`, `--ram-budget SIZE`, `--lora PATH[:S]` (wiederholbar; PEFT-Ordner oder GGUF-Adapter mit einem Skalierungsfaktor).

Zusätzliche Sampler-Optionen für `generate` und `chat`:

```bash
corbul generate M --prompt "…" --min-p 0.05 --typical-p 0.9 --repeat-penalty 1.1 \
  --presence-penalty 0.2 --frequency-penalty 0.2 --mirostat 2 --mirostat-tau 5 --mirostat-eta 0.1 \
  --dry-multiplier 0.8 --xtc-probability 0.1
corbul generate M --prompt "…" --sampler-json '{"top_k": 20}'   # jedes Sampler-Feld, angewendet vor den obigen Flags
```

Zustandsraum- und Hybridmodelle (Mamba, RWKV, Jamba, Bamba) laufen über `generate` und `chat` als reine Textfortsetzung: kein Chat-Template, keine strukturierte Ausgabe, kein `serve`.

## Läuft es auf diesem Rechner?

```bash
corbul doctor                # CPU-Befehlssätze, gewählter Kernel, RAM, Platte, Sensoren, GPU-Sonde
corbul plan ./my-model --context 8192          # Urteil (passt / knapp / streamt / zu langsam / läuft nicht), Geschwindigkeit, Rat
corbul plan ./my-model --ram-budget 8g --calibrate   # tu so, als hättest du 8 GB; miss diesen Rechner vorher etwa eine Sekunde lang
corbul capabilities [--json]                   # was dieser Build kann und was nicht (Alias: corbul features)
```

Geschwindigkeiten sind Schätzungen, keine Messungen deines Modells. `ENGINE_SIMD` erzwingt einen Kernel, `ENGINE_REPRODUCIBLE=1` schaltet den reproduzierbaren Modus ein, `ENGINE_BACKEND=cpu|gpu|auto` wird von der Bibliothek gelesen (`corbul` rechnet immer auf der CPU).

## Größer als der RAM

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"     # MoE: geroutete Experten streamen von der Platte
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --disk-mbps 800 --prompt "…"   # dicht: Layer streamen; die Geschwindigkeit ist durch die Platte begrenzt; --disk-mbps (Standard 500) speist nur die Geschwindigkeitsschätzung, die ausgegeben wird
```

## Mehrere Rechner (LAN-Mesh)

```bash
# auf jedem Helfer-Rechner (der Schlüssel braucht 8+ Zeichen; oder CORBUL_MESH_KEY setzen)
corbul worker ./my-model --key MESHKEY12345 --bind 0.0.0.0:7471
corbul worker ./my-model --key-file mesh.key --discovery            # beantwortet auch UDP-Erkennungssonden (opt-in); --discovery-port N; --no-discovery erzwingt „aus“
# auf dem Hauptrechner
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key MESHKEY12345 --prompt "…"
corbul generate ./my-model --discover --key MESHKEY12345 --prompt "…"      # Sonde per Broadcast senden, Worker nutzen, die dasselbe Modell halten
corbul generate ./my-model --workers h:7471 --key MESHKEY12345 --wire-f16 --local-speed 2.5 --prompt "…"
```

`--wire-f16` halbiert den Verkehr des verborgenen Zustands (und gibt die Bitidentität auf); `--local-speed` sagt dem Aufteiler, wie schnell dieser Rechner im Verhältnis zu den Workern ist; `--discover-target HOST:PORT` und `--discover-wait-ms N` ändern die Sondenadresse (Standard `255.255.255.255:47813`) und die Wartezeit (Standard 2000 ms). Der Verkehr ist authentifiziert, aber **nicht verschlüsselt**: Nimm ein vertrauenswürdiges Netz oder ein VPN. Nur auf Loopback getestet.

## Strukturierte Ausgabe, Quellenangaben, Adapter

```bash
corbul generate M --json --prompt "…"                       # gültiges JSON
corbul generate M --json-schema schema.json --prompt "…"    # eine Datei oder Inline-JSON
corbul generate M --regex '[A-Z]{3}-[0-9]{4}' --prompt "…"  # regulärer Ausdruck
corbul generate M --grammar rules.gbnf --prompt "…"         # GBNF-Grammatik; nutze nur eine der vier
corbul generate M --sources notes.txt --sources paper.txt --prompt "Answer with quotes" --citations report.json   # zitatgesperrt
corbul generate M --sources notes.txt --no-lock --prompt "…"   # Zitate erst nachträglich prüfen
corbul generate M --lora ./adapter:0.8
```

Der Server akzeptiert in Anfragen `response_format`, `grammar` und `sources` und bietet `POST /v1/quotes/verify`.

## Lange Chats, Sitzungen, Reproduzierbarkeit

```bash
corbul generate M --session ./sessions --session-cap 2g --prompt "…"   # das längste gespeicherte Prompt-Präfix wiederverwenden; --compress verkleinert gespeicherte Dateien (nicht mit --kv-policy)
corbul generate M --kv-policy h2o --kv-budget 2048 --kv-sinks 4 --prompt "…"   # streaming | h2o | snapkv; absichtlich verlustbehaftet
corbul generate M --governor --verbose --prompt "…"            # an Temperatur, Speicher und Akku anpassen (Linux); --no-governor schaltet es aus
corbul generate M --lang ro --prompt "…"                         # Tokens fremder Schriften verbieten; Rumänisch: kein ş ţ
corbul generate M --reproducible --temperature 0 --prompt "hi" --receipt r.json --receipt-key-file receipt.key
corbul verify-receipt r.json --key-file receipt.key --model M      # prüft das Siegel (und die Gewichte, wenn --model angegeben ist)
corbul replay r.json --model M --key-file receipt.key              # führt die Generierung erneut aus und vergleicht
corbul replay r.json --model M --key-file receipt.key --prompt "…" --raw     # wenn die Quittung keinen Prompt-Text enthält; oder --prompt-tokens FILE
```

Quittungen funktionieren mit der einfachen Generierungsschleife, nicht mit MoE- oder Zustandsraum-Modellen. Optional: `--receipt-key K` (in der Prozessliste sichtbar; bevorzuge `--receipt-key-file`), `--receipt-topk N`, `--keep-steps`, `--leaves`.

## Schnelleres Dekodieren

```bash
corbul generate M --draft ./small-model --draft-len 4 --prompt "…"      # ein Entwurfsmodell schlägt vor, M verifiziert
corbul generate M --lookahead --ngram-min 1 --ngram-max 4 --prompt "…"   # Prompt-Lookup, kein zusätzliches Modell
corbul generate M --self-spec --skip-layers auto --prompt "…"            # M entwirft mit einer Teilmenge seiner Layer; oder --skip-layers 4,6,8-11
```

Dieselbe gierige Antwort und dieselbe Sampling-Verteilung; der Text erscheint am Ende. Es gelten nur `--temperature`, `--top-k`, `--top-p`; nicht mit strukturierter Ausgabe, Quittungen, Sitzungen oder dem Mesh.

## Suchen, quantisieren, mergen, steuern, abrufen, Wasserzeichen, Sprache

```bash
corbul search M --prompt "Q" --method beam --width 4 --all --explain --json     # auch --length-penalty L
corbul search M --prompt "2+2?" --method best-of-n --samples 5
corbul search M --prompt "2+2?" --method vote --samples 5 --answer number       # oder text
corbul autoquant ./model --budget 4.5bpw --recipe recipe.json --report report.txt   # 4.5bpw, 6g oder Q4_K; --candidates Q4_K,Q6_K --base Q8_0 --imatrix I --sample-rows 256 --json
corbul quantize ./model out.gguf --type Q4_K_M --imatrix imatrix.dat    # oder --recipe recipe.json; --name NAME
corbul merge merge.json --out merged.safetensors --dry-run              # --model PATH (wiederholbar) --max-memory 8g --overwrite --verbose
corbul steer train M --positive polite.txt --negative rude.txt --out polite.cvec --method mean-diff   # oder pca
corbul steer apply M --vector polite.cvec --strength 1 --layers 8-20 --positions last --prompt "…" --no-attention
corbul steer lens M --prompt "The capital of France is"
corbul rag add ./lib notes.txt paper.txt --chunk-chars 800 --overlap 120 --embed-model ./embedder
corbul rag search ./lib --query "refund policy" --k 5 --json
corbul rag ask ./lib M --question "What is the refund period?"             # antwortet mit zitatgesperrten Quellenangaben
corbul lang detect "Aceasta este o propoziție."                          # oder --file text.txt
corbul lang normalize --file text.txt                                    # ș/ț, Mojibake, Anführungszeichen
corbul generate M --prompt "…" --watermark SECRETKEY --wm-gamma 0.25 --wm-delta 2 --wm-context 4
corbul watermark detect M --key SECRETKEY --file answer.txt --no-dedup   # oder --text "…"
corbul embed ./bge-small "first text" "second text" --pool cls --json    # a BERT-family folder; or --file lines.txt, --no-normalize
corbul rerank ./ms-marco-minilm --query "refund policy" "doc one" "doc two" --top 3 --json   # needs a cross-encoder
```

Nicht jede Flag-Kombination funktioniert; jeder Befehl gibt dann einen einzeiligen Aufrufhinweis aus. Nichts davon wurde mit einem echten Checkpoint ausgeführt.

## Bereitstellen (Serving)

```bash
corbul serve --models-dir ./models --host 127.0.0.1 --port 8080 --token-file token.txt --default-model qwen
corbul serve --models-dir ./models --alias fast=qwen-0.5b --max-loaded 2 --sleep-idle-seconds 600
corbul serve --models-dir ./models --keys-file keys.json --allow-admin --max-kv 8g --max-heavy 4
corbul serve --models-dir ./models --cors "https://app.example" --log-prompts --drain-seconds 10 --config-dir ./cfg
corbul serve --encoder ./bge-small --port 8081                                  # embeddings (and /v1/rerank for a cross-encoder) without a decoder
```

Öffne `http://127.0.0.1:8080/` für die Weboberfläche (auch `/ui`). Die Seite ist öffentlich; jeder Aufruf, den sie macht, braucht das Token. `--token T` entspricht `CORBUL_TOKEN` (in der Prozessliste sichtbar, bevorzuge also `--token-file`). `--keys-file` enthält API-Schlüssel mit Modelllisten und Ratenbegrenzungen pro Schlüssel. `--allow-admin` erlaubt Schlüsselinhabern, `/api/create`, `/api/delete`, `/models/load` und `/models/unload` zu nutzen; standardmäßig darf das nur das Master-Token. `--max-kv` lehnt Anfragen ab, deren KV-Cache die Größe überschreiten würde (Standard 8g); `--max-heavy` begrenzt gleichzeitige Tokenize-, Template- und Verify-Anfragen (Standard 4). `--no-log` schaltet das Anfrageprotokoll stumm. `GET /v1/capabilities` listet auf, was funktioniert. Die Embedding-Routen funktionieren (`/v1/embeddings`, `/api/embed`); Rerank braucht `--encoder`.

## In `corbul` (noch) nicht verfügbar

- Modelle mit Multi-Head Latent Attention (im DeepSeek-Stil), RWKV in GGUF-Form, Zamba/Zamba2, die übrigen i-Quants (IQ1–IQ3), RoBERTa/XLM-R-Modelle.
- Vision- und Audio-Eingaben (die Bibliothek hat die Bausteine), das GPU-Back-End.
- Bitidentität des gewöhnlichen quantisierten Prefills (siehe ENGINE.md, „Exaktheit“).
