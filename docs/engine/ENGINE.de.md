# Die Clean-Room-Engine (`companion-engine` und `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Weitere Dokumente: [INNOVATIONS.de.md](INNOVATIONS.de.md) · [QUICKSTART.de.md](QUICKSTART.de.md) · [CLEANROOM.de.md](CLEANROOM.de.md)

**Status: Alpha. Alles Folgende wurde nur an kleinen synthetischen Checkpoints getestet.** Die Autoren haben kein echtes Modell von Hugging Face durch diese Engine laufen lassen (die Build-Umgebung kam nicht an huggingface.co heran), keine GPU hat irgendetwas davon ausgeführt, und die Pfade für ARM, macOS und Windows wurden hier nie kompiliert. Lies die Tabelle „verifiziert und nicht verifiziert“, bevor du irgendeiner Aussage vertraust.

## Was es ist

Eine lokale Inferenz-Engine für Sprachmodelle, von Grund auf in Rust geschrieben, aus öffentlichen Beschreibungen von Dateiformaten und veröffentlichten Arbeiten. Sie besteht aus drei Teilen:

| Crate | Was es ist |
|---|---|
| `crates/engine` (`companion-engine`) | Die Bibliothek: Dateiformate, Tokenizer, Quantisierung, Vorwärtsdurchläufe, Sampling und die zusätzlichen Werkzeuge aus INNOVATIONS.de.md. Abhängigkeiten: nur `serde_json` und `memmap2`. |
| `crates/engine-cli` (`corbul`) | Eine Kommandozeile und ein lokaler, zu OpenAI / Ollama / Anthropic kompatibler Server auf der Bibliothek. |
| `crates/engine-gpu` (`companion-engine-gpu`) | WGSL-Kernel für das Crate `wgpu`, hinter einer standardmäßig ausgeschalteten Option. Wird von `corbul` nicht benutzt. |

Sie ist von der Desktop-App Colibri Companion getrennt und ersetzt sie nicht: Die App steuert weiterhin colibri von JustVugg. Diese Engine als alternatives Backend in die App einzubinden ist geplant, nicht erledigt.

## Was sie unterstützt

**Dateien.** GGUF v2/v3 (Header und Tensortabelle werden gelesen, ohne die Gewichte anzufassen, eine 400-GB-Datei wird also in Millisekunden inspiziert); ein streamender GGUF-Writer für Modelle, die größer als der RAM sind; safetensors, als eine Datei oder in Teilen mit Index; `config.json` und `tokenizer.json` von Hugging Face; ein Modell-Downloader (über `curl`, mit Fortsetzung und SHA-256-Prüfung).

**Quantisierung.** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 und die K-Quants Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K lassen sich dekodieren, für alle gibt es Kodierer (mit Gewichtung über eine Wichtigkeitsmatrix), die Umwandlung Hugging Face → GGUF und die Requantisierung von GGUF. Gewichte bleiben im Speicher quantisiert; das Matrix-Vektor-Produkt baut die Matrix nie auf. Die i-Quants (IQ*) sind nicht implementiert.

**Architekturen.**

| Familie | Stand |
|---|---|
| Dichte Decoder: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 und ähnliche (RMS- oder Layer-Norm, GQA/MQA, RoPE mit linearer / Llama-3- / YaRN-Skalierung und partieller Rotation, QK-Norm, Biases, Soft-Capping, Gleitfenster, gated oder einfaches MLP) | Implementiert; von `corbul` geladen |
| Mixture of Experts: Mixtral, Qwen-MoE, gestapelte GGUF-Experten, gemeinsame Experten, geroutete Experten von der Platte mit LRU-Cache | Implementiert; von `corbul` geladen |
| Multi-Head Latent Attention (DeepSeek-V2/V3-, Kimi-K2-Stil) | Nur Bibliotheksmodul; das Architekturregister führt sie noch als „geplant“, und `corbul` lädt sie nicht |
| Mamba-1, Mamba-2, RWKV-6, RWKV-7 | Nur Bibliotheksmodul; ebenso |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, Encoder- und Encoder-Decoder-Modelle (T5, BERT, Whisper …), LongRoPE | Nicht unterstützt; `corbul` sagt das, statt zu raten |
| Bild- und Audioeingaben | Nicht unterstützt (nur Text) |

**Hardware-Pfade.**

- CPU: zur Laufzeit gewählte Kernel (skalar, AVX2, AVX-512, VNNI auf x86-64; NEON auf aarch64), ein dauerhafter Thread-Pool, der zu gemischten Leistungs-/Effizienzkernen passt, Erkennung von Hardware und NUMA und ein Planer, der entscheidet, was im RAM bleibt.
- GPU: WGSL-Kernel, mit `naga` und gegen eine Rust-Emulation geprüft. Sie sind nie auf einer GPU gelaufen.
- Modelle größer als der RAM: Experten von der Platte (MoE), Layer-Streaming (dicht), ein LAN-Mesh über mehrere Rechner.

## Verifiziert und nicht verifiziert

| Bereich | Stand |
|---|---|
| Unit-, Integrations- und Fuzz-Tests an synthetischen Checkpoints (Linux, x86-64) | Verifiziert. Hunderte Tests bestehen; feindliche oder abgeschnittene Dateien liefern Fehler statt Abstürze. |
| Echte Hugging-Face-Checkpoints (beliebige Familie) | **Nicht ausgeführt.** Die Umgebung kam nicht an huggingface.co heran. |
| Tokenizer mit echten `tokenizer.json`-Dateien | Nach den öffentlichen Formatbeschreibungen geschrieben; ein Testaufbau existiert (`ENGINE_VOCAB_DIR`), wurde von uns aber nicht mit echten Dateien ausgeführt. |
| Q/K-Zeilenpermutation nach llama.cpp-Art in GGUF-Dateien | Nach Beschreibung implementiert; an echten Dateien nicht geprüft. |
| Quantisierer und Dequantisierer | Gegen von Hand berechnete Blöcke und Hin-und-zurück-Läufe geprüft; nicht gegen Dateien anderer Werkzeuge. |
| Ausführung auf GPU | **Nicht ausgeführt.** Die Kernel sind nur mit `naga` und einer Rust-Emulation validiert. |
| ARM-/NEON-, macOS-, Windows-Pfade | **Hier nie kompiliert oder ausgeführt.** Die Sonden werden mit aufgezeichnetem Text dieser Systeme getestet. |
| Formeln für Mamba / RWKV / MLA | Aus Veröffentlichungen und der Erinnerung an die Referenzbeschreibungen geschrieben, nur auf interne Konsistenz getestet. |
| Mesh in einem echten Netz | Nur über Loopback getestet. |
| Geschwindigkeit | **Keine verlässlichen Zahlen.** Die Build-Maschine hatte zwei geteilte Kerne; Messungen waren verrauscht. Alle Geschwindigkeiten im Autopiloten sind Schätzungen. |
| Reproduzierbarer Modus | Identisch über fünf x86-Kernel-Varianten und mehrere Thread-Zahlen; ARM ungetestet. |
| Lizenzen der Abhängigkeiten | Mit `cargo-deny` bei einem früheren Commit (b496ba5) geprüft; spätere Ergänzungen wurden noch nicht erneut geprüft. |

## Clean-Room-Erklärung

Beim Schreiben dieser Engine wurde kein Code von colibri, llama.cpp oder einer anderen Inferenz-Engine gelesen oder kopiert. Formate und Algorithmen stammen aus öffentlichen Spezifikationen, Veröffentlichungen und Dateistrukturen. Siehe [CLEANROOM.de.md](CLEANROOM.de.md) für den Prozess und seine Grenzen.

## Vergleich mit anderen lokalen Engines

Wir haben keinen direkten Vergleichstest gemacht, daher gibt es hier keine Aussagen zu Geschwindigkeit oder Qualität. Qualitativ:

- Etablierte Engines wie llama.cpp, Ollama, mistral.rs oder vLLM wurden von vielen Menschen an Tausenden echter Checkpoints betrieben, haben funktionierende GPU-Backends, decken mehr Architekturen und Quantisierungsformate ab (auch i-Quants) und verarbeiten Vision-Modelle. Diese Engine hat noch nichts davon vorzuweisen. Wenn du heute etwas brauchst, das mit einem bestimmten großen Modell funktioniert, nimm eine von ihnen.
- Was diese Engine anders versucht: eingebaute Werkzeuge für Integrität und Ehrlichkeit (an Quellen gebundene Zitate, reproduzierbarer Modus mit versiegelten Quittungen aus Hash-Kette und optionalem HMAC, Unsicherheitsanalyse, ein Autopilot, der „das läuft nicht“ sagt, statt abzustürzen), Sprachwerkzeuge mit Rumänisch an erster Stelle, Modelle jenseits deines RAMs oder einer einzelnen Maschine, eine kleine Rust-Codebasis mit zwei Abhängigkeiten und nur permissive Lizenzen.
- Jede Behauptung, diese Engine sei schneller oder genauer als eine andere, wäre ungetestet. Miss auf deiner Hardware und sag uns Bescheid.

## Bibliothekskarte

Formate und Laden: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`. Modell: `arch`, `model`, `moe`, `mla`, `ssm`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Laufzeit: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Besondere Funktionen: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Neueste Welle: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark` — in der Bibliothek implementiert und durch Tests an synthetischen Modellen abgedeckt, noch im Feinschliff, aus `corbul` noch nicht erreichbar, ihre APIs können sich ändern.
