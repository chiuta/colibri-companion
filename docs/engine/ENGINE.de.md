# Die Clean-Room-Engine (`companion-engine` und `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Weitere Dokumente: [INNOVATIONS.md](INNOVATIONS.de.md) · [QUICKSTART.md](QUICKSTART.de.md) · [CLEANROOM.md](CLEANROOM.de.md) · [CHANGES-SECURITY.md](CHANGES-SECURITY.de.md)

**Status: Alpha.** Alles Folgende wurde mit kleinen synthetischen Checkpoints, mit Tokenizer-Referenzdateien und mit aufgezeichnetem Beispieltext getestet. **Kein echtes, von Hugging Face heruntergeladenes Modell wurde von den Autoren durch diese Engine laufen gelassen**, keine GPU hat etwas davon ausgeführt, und die Codepfade für ARM, macOS und Windows wurden hier nicht ausgeführt. Lies „Verifiziert und nicht verifiziert“, bevor du einer Aussage traust. `corbul capabilities` gibt dieselben Grenzen aus der Binärdatei selbst aus.

## Was es ist

Eine lokale Inferenz-Engine für Sprachmodelle, von Grund auf in Rust geschrieben, auf Basis öffentlicher Dateiformat-Beschreibungen und veröffentlichter Fachartikel.

| Crate | Was es ist |
|---|---|
| `crates/engine` (`companion-engine`) | Die Bibliothek: Dateiformate, Tokenizer, Quantisierung, Vorwärtsdurchläufe, Sampling und die zusätzlichen Werkzeuge aus INNOVATIONS.md. Laufzeitabhängigkeiten: nur `serde_json` und `memmap2`. |
| `crates/engine-cli` (`corbul`) | Die Kommandozeile und ein lokaler Server, kompatibel mit OpenAI, Ollama und Anthropic, mit eingebauter Weboberfläche. Hängt von der Bibliothek und von `serde_json` ab. |
| `crates/engine-gpu` (`companion-engine-gpu`) | WGSL-Kernel und eine CPU-Emulation davon. Der Geräte-Runner (`wgpu`, `pollster`) liegt hinter dem Cargo-Feature `wgpu`, das standardmäßig aus ist. **`corbul` verwendet dieses Crate nicht.** |

Sie ist von der Desktop-App Colibri Companion getrennt und ersetzt sie nicht; die App steuert weiterhin das colibri von JustVugg. Diese Engine als alternatives Back-End der App zu nutzen, ist geplant, aber nicht umgesetzt.

## Was unterstützt wird

**Dateien.** GGUF v2 und v3 (Header und Tensortabelle werden gelesen, ohne die Gewichte zu berühren); ein streamender GGUF-Writer für Ausgaben, die größer als der RAM sind; safetensors, einzeln oder in Shards mit Index; Hugging-Face-`config.json`, `tokenizer.json` und Chat-Templates; ein Modell-Downloader (Modul `hub`, über `curl`, mit `wget` als anonymem Ersatz; Fortsetzen, SHA-256-Prüfungen, an einen Commit gebundene Downloads, Größenobergrenzen, Ablehnung von Weiterleitungen auf reines http).

**Quantisierung.** Dekodiert: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K (14 Typen). Für dieselben Typen gibt es Encoder, mit Gewichtung durch eine Importance-Matrix und mit Rezepten pro Tensor. Die Gewichte bleiben im Speicher quantisiert; das Matrix-Vektor-Produkt entpackt nie die ganze Matrix. **Nicht unterstützt:** die i-Quants (IQ\*) und jeder Typ, der nicht in der Liste steht.

**Architekturen.**

| Familie | Zustand in `corbul` |
|---|---|
| Dichte Decoder: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 und ähnliche (RMS- oder Layer-Norm, GQA/MQA, RoPE mit linearer, Llama-3-, YaRN- und LongRoPE-Skalierung und partieller Rotation, QK-Norm, Biases, Soft-Capping, Gleitfenster, gated oder einfaches MLP) | Geladen und ausgeführt (Chat-Template, serve, alle Optionen) |
| Mixture of Experts: Mixtral, Qwen-MoE, gestapelte GGUF-Experten, geteilte Experten, geroutete Experten, die mit einem LRU-Cache von der Platte gestreamt werden | Geladen und ausgeführt |
| Zustandsraum: Mamba-1, Mamba-2, RWKV-6, RWKV-7 (Modul `ssm`) | Von `generate` und `chat` als **reine Textfortsetzung** ausgeführt: kein Chat-Template, keine strukturierte Ausgabe, kein `serve`. RWKV in GGUF-Form wird nicht unterstützt (nur safetensors) |
| Hybrid: Jamba und Bamba (Modul `hybrid`) | Wie die Zeile zum Zustandsraum. Zamba und Zamba2 werden nicht unterstützt |
| Multi-Head Latent Attention (DeepSeek-V2/V3, Kimi-K2; Modul `mla`) | **Nur Bibliothek.** Der Baustein existiert und ist auf interne Konsistenz getestet, aber `corbul` setzt ein solches Modell nicht zusammen (`corbul capabilities` sagt nein) |
| Vision-Encoder (ViT, CLIP, SigLIP; Modul `vision`, Decoder in `imageio`) und Sprach-Front-End plus ein Runner im Whisper-Layout (Modul `audio`) | **Nur Bibliothek.** `corbul` nimmt nur Text an |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, Encoder- und Encoder-Decoder-Modelle (T5, BERT…), WordPiece-Tokenizer | Nicht unterstützt; `corbul` sagt das, statt zu raten |
| Endpunkte für Embeddings und Reranking | Antworten mit 501 (nicht implementiert) |

**Tokenizer.** Byte-Level-BPE und Modelle im SentencePiece-Stil, Pre-Tokenizer-Regeln für 12 Familien, hinzugefügte Tokens, Byte-Fallback, Templates aus `tokenizer_config.json` und aus GGUF-Metadaten. WordPiece/BERT wird nicht unterstützt.

**Sampling.** Temperatur, Top-k, Top-p, Min-p, Typical-p, Wiederholungs-, Präsenz- und Häufigkeitsstrafen, Mirostat 1 und 2, DRY, XTC, eine JSON-Sampler-Spezifikation, Zufall mit Seed; eingeschränktes Dekodieren per JSON, JSON Schema, regulärem Ausdruck oder GBNF-Grammatik (mit Arbeitsbudgets, damit eine feindliche Grammatik den Prozess nicht aufhängen kann); Logit-Bias und Stopp-Strings über den Server. Spekulatives Dekodieren: Entwurfsmodell, Prompt-Lookup, Selbstspekulation (siehe INNOVATIONS.md).

**Server.** `corbul serve` bietet `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/messages` und `/v1/messages/count_tokens` (Anthropic-Stil), die Ollama-Endpunkte `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/create`, `/api/delete`, `/api/ps`, `/api/version`, die Modellverwaltung `/models`, `/models/load`, `/models/unload`, dazu `/metrics`, `/props`, `/slots`, `/tokenize`, `/detokenize`, `/apply-template`, `/health`, `/v1/capabilities`, `/v1/quotes/verify` und die Weboberfläche unter `/` und `/ui` (acht Sprachen). Tool-Calling wird über die Chat-Templates unterstützt. Die Routen für Embeddings und Rerank (`/v1/embeddings`, `/v1/rerank`, `/api/embed`, `/api/embeddings`) antworten mit 501. Streaming-Antworten enthalten keinen Zitatblock.

**Hardware.** CPU-Kernel werden zur Laufzeit gewählt: scalar, avx2+fma, avx2+fma+avxvnni, avx512, avx512+vnni (x86-64), neon und neon+dotprod (aarch64); `ENGINE_SIMD` erzwingt einen davon. Ein dauerhafter Thread-Pool, Hardware- und NUMA-Erkennung und ein Planer (`corbul plan`, `corbul doctor`) entscheiden, was im RAM bleibt. Die GPU-Kernel sind nie auf einer GPU gelaufen (siehe unten).

## Exaktheit

- **Dichte f32-Gewichte:** Die gebündelten und die Ein-Token-Pfade, das Mesh, das Layer-Streaming und die Caches liefern bitidentische Logits.
- **Quantisierte Gewichte:** Bitidentität zwischen gebündelter und Ein-Token-Auswertung gilt nur, wenn `exact_batch` eingeschaltet ist. Die Verifikation, die Verifikationsrunden der Selbstspekulation und der Batching-Executor schalten es ein. Das normale Prefill nutzt einen schnelleren Pfad (erst dequantisieren, dann kacheln) und unterscheidet sich von der Ein-Token-Auswertung nur durch Rundungsrauschen.
- **Reproduzierbarer Modus** (`--reproducible` oder `ENGINE_REPRODUCIBLE=1`): identische Ergebnisse über die x86-Kernelvarianten und Thread-Zahlen, die wir testen konnten. ARM wurde nicht getestet. Er kostet Geschwindigkeit.
- Mirostat 1 verwendet in seiner Schätzung die Zahl der verbleibenden Kandidaten, nicht die Vokabulargröße (dokumentierte Abweichung).

## Verifiziert und nicht verifiziert

| Bereich | Zustand |
|---|---|
| Unit-, Integrations-, adversariale und Fuzz-Tests mit synthetischen Checkpoints (Linux x86-64). Die Zahlen ändern sich mit jedem Commit; führe `cargo test --workspace` aus, um die aktuelle Zahl zu sehen | Beim letzten vollständigen Lauf bestanden. Feindliche und abgeschnittene Dateien liefern Fehler. |
| Tokenizer gegen llama.cpp-Referenzdateien, 12 Familien | 46 von 46 Fällen stimmen überein (Referenzdateien von GitHub geholt; `crates/engine/tests/tokenizer_real.rs`, `ENGINE_VOCAB_DIR`). Nur der Tokenizer, keine ganzen Modelle. |
| Echtes Hugging-Face-Modell über `corbul` (beliebige Familie) | **Von den Autoren nicht ausgeführt.** Die Build-Sandbox konnte huggingface.co nicht erreichen. Ein manueller CI-Job lässt ein winziges Llama-GGUF (`stories260K`) durch die CLI laufen; er ist noch nicht gelaufen |
| Q/K-Zeilenpermutation im llama.cpp-Stil, GGUF-Metadatenvarianten | Nach öffentlichen Beschreibungen implementiert; an echten Dateien nicht geprüft |
| Quantisierer und Dequantisierer | Von Hand berechnete Blöcke und Rundreisen; nicht mit Dateien anderer Werkzeuge verglichen |
| GPU-Ausführung | **Nicht ausgeführt.** Die Kernel sind nur durch `naga` und durch eine Rust-Emulation validiert; `corbul` verwendet sie nicht |
| ARM / NEON, macOS, Windows | **Hier nicht ausgeführt.** Die Sonden sind an aufgezeichnetem Beispieltext getestet. Der CI-Workflow hat dafür eine Matrix, ist aber nie gelaufen |
| Mamba, RWKV, Jamba/Bamba, MLA, Vision, Audio | Nur Tests auf interne Konsistenz; kein echter Checkpoint wurde verwendet |
| Mesh | Nur Loopback; authentifiziert, nicht verschlüsselt; keine Geschwindigkeitsmessung |
| Geschwindigkeit | **Keine verlässlichen Zahlen.** Die Build-Maschine hatte zwei geteilte Kerne. Die Geschwindigkeiten des Autopiloten sind Schätzungen |
| Lizenzen der Abhängigkeiten | Mit `cargo-deny` bei Commit b496ba5 geprüft; spätere Änderungen wurden nicht erneut geprüft |
| Übersetzungen der Dokumente | Maschinell unterstützt; noch keine Prüfung durch Muttersprachler |

## Umgebungsvariablen

| Variable | Wirkung |
|---|---|
| `COLIBRI_THREADS` | Größe des globalen Thread-Pools (Standard: nutzbare Kerne). Wird zuerst von der Bibliothek gelesen |
| `ENGINE_SIMD` | Erzwingt eine Kernelvariante per Namen (Benchmarks, Fehlerberichte) |
| `ENGINE_REPRODUCIBLE` | `1`, `true` oder `on`: reproduzierbarer Modus von Anfang an |
| `ENGINE_BACKEND` | `cpu`, `gpu` oder `auto` (nicht gesetzt = auto). `corbul` rechnet in beiden Fällen auf der CPU |
| `ENGINE_GPU`, `ENGINE_GPU_API`, `ENGINE_GPU_SOFTWARE` | Adapterwahl für das GPU-Crate: Index, Grafik-API, Software-Adapter erlauben. Von `corbul` nicht verwendet |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Zugriffstoken für geschützte Repositories (`download`, `inspect owner/repo`) |
| `HF_ENDPOINT` | Spiegel des Hugging-Face-Hosts |
| `CORBUL_TOKEN` | Gemeinsames Bearer-Token von `corbul serve` (wie `--token`) |
| `CORBUL_MESH_KEY` | Mesh-Schlüssel für `worker` und `--workers`/`--discover` (wie `--key`) |
| `CORBUL_CONFIG_DIR` | Wo die von `POST /api/create` erzeugten Modelfiles liegen (wie `--config-dir`; Standard `~/.config/corbul`) |
| `NUMBER_OF_PROCESSORS`, `PROCESSOR_IDENTIFIER` | Werden unter Windows von der Hardware-Sonde gelesen; du setzt sie nicht |
| `ENGINE_VOCAB_DIR`, `ENGINE_VOCAB_ONLY`, `ENGINE_REQUIRE_VOCAB`, `ENGINE_REQUIRE_TOOLS`, `ENGINE_PRINT_GOLDEN`, `ADVERSARIAL_N` | Steuerung der Testsuite: Vokabularordner, Familien einschränken, einen übersprungenen Test fehlschlagen lassen, fehlendes `curl`/`wget` fehlschlagen lassen, Golden-Hashes neu erzeugen, Größe des adversarialen Laufs |

## Clean-Room-Erklärung

Beim Schreiben dieser Engine wurde kein Code von colibri, llama.cpp oder einer anderen Inferenz-Engine gelesen oder kopiert. Die Formate und Algorithmen stammen aus öffentlichen Spezifikationen, Fachartikeln und Dateilayouts. Das Modell, das den Code geschrieben hat, wurde mit öffentlichen Texten trainiert, die solche Projekte enthalten können; den Prozess und seine Grenzen beschreibt [CLEANROOM.md](CLEANROOM.de.md).

## Vergleich mit anderen lokalen Engines

Wir haben keinen direkten Benchmark durchgeführt, deshalb gibt es hier keine Aussagen zu Geschwindigkeit oder Qualität.

- Etablierte Engines (llama.cpp, Ollama, mistral.rs, vLLM) wurden mit Tausenden echter Checkpoints betrieben, haben funktionierende GPU-Back-Ends, decken mehr Architekturen und Quantisierungsformate ab, einschließlich i-Quants, und verarbeiten Vision-Modelle. Diese Engine hat davon noch nichts nachzuweisen. Wenn du heute etwas brauchst, das mit einem bestimmten großen Modell funktioniert, nimm eine davon.
- Was diese Engine anders zu machen versucht: eingebaute Integritätswerkzeuge (zitatgesperrte Quellenangaben, reproduzierbarer Modus mit Quittungen, Unsicherheitsanalysen, ein Planer, der „das läuft nicht“ sagt, statt abzustürzen), Sprachwerkzeuge mit Rumänisch an erster Stelle, Modelle, die größer sind als dein RAM oder deine Maschine, eine kleine Rust-Codebasis mit zwei Bibliotheksabhängigkeiten und ausschließlich permissive Lizenzen.

## Bibliothekskarte

Formate und Laden: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`, `imageio`. Modelle: `arch`, `model`, `moe`, `mla`, `ssm`, `hybrid`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Laufzeit: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Besondere Funktionen: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Rechenaufwand zur Laufzeit, Abstimmung, Retrieval: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark`. Multimodal: `vision`, `audio`. Welche davon `corbul` erreicht, steht in INNOVATIONS.md; die Bibliotheks-APIs können sich noch ändern.
