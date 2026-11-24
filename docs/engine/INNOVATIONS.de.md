# Was an dieser Engine anders ist

[English](INNOVATIONS.md) · [Română](INNOVATIONS.ro.md) · [Français](INNOVATIONS.fr.md) · [Italiano](INNOVATIONS.it.md) · [Español](INNOVATIONS.es.md) · [Português](INNOVATIONS.pt.md) · [Deutsch](INNOVATIONS.de.md) · [Magyar](INNOVATIONS.hu.md)

Notizen in einfacher Sprache zu Funktionen, die andere lokale Engines meist nicht haben. Zu jeder steht, was sie tut, wo du sie erreichst und was sie nicht kann. Alle wurden nur mit synthetischen Modellen getestet; keine wurde von den Autoren mit einem echten Checkpoint ausgeführt. „In `corbul`“ heißt, dass es heute einen Befehl oder ein Flag gibt (`corbul help`); „nur Bibliothek“ heißt Rust-Code ohne Befehl. Siehe [ENGINE.md](ENGINE.de.md) und [QUICKSTART.md](QUICKSTART.de.md).

## Integrität, Hardware und Speicher

### LAN-Mesh

Modul `mesh`. In `corbul`: `worker`, `generate --workers`, `--discover`, `--key`, `--wire-f16`. Ein Modell, das für eine Maschine zu groß ist, wird in Layer-Bereiche zerlegt, einen pro Rechner. Zwischen ihnen wandert nur der verborgene Zustand. Die Maschinen beweisen einander, dass sie einen Schlüssel teilen (gegenseitiges HMAC-SHA256), jeder Frame trägt einen MAC, die Größen im Handshake sind begrenzt, ein ausgefallener Worker wird verworfen und seine Layer werden neu verteilt. Dichte f32-Logits sind gleich wie auf einer einzelnen Maschine. Die UDP-Erkennung ist opt-in: Ein Worker antwortet auf Sonden nur, wenn er mit `--discovery` oder `--discovery-port` gestartet wurde (`--no-discovery` erzwingt „aus“), und das `--discover` des Koordinators sendet eine Sonde per Broadcast und wartet auf Antworten von Workern, die dasselbe Modell halten. *Kann nicht:* verschlüsseln (nimm ein vertrauenswürdiges LAN oder ein VPN); getestet wurde nur auf Loopback; die Geschwindigkeit gegenüber einer einzelnen Maschine wurde nicht gemessen.

### Layer-Streaming

Modul `stream_layers`. In `corbul`: `--stream-layers`, `--ram-budget`, `--stream-window`, `--disk-mbps` (für Schätzungen). Führt ein dichtes Modell aus, das größer als der RAM ist, indem ein kleines Fenster von Layern im Speicher gehalten und in einem Hintergrund-Thread vorab geladen wird. Die Ergebnisse sind gleich wie beim Modell im RAM. *Kann nicht:* Jeder Token liest jeden Layer, die Geschwindigkeit ist also durch die Plattengeschwindigkeit begrenzt; nur dichte Layer (Mixture-of-Experts-Modelle streamen stattdessen ihre gerouteten Experten, mit `--ram-budget`).

### Zitatgesperrte Quellenangaben

Modul `quotelock`. In `corbul`: `--sources FILE` (wiederholbar), `--no-lock`, `--citations FILE`; der Server akzeptiert `sources` und bietet `POST /v1/quotes/verify`. Beim Dekodieren muss Text zwischen « und » eine wörtliche Passage einer Quelle sein (tolerant gegenüber Groß- und Kleinschreibung, Leerraum, Anführungszeichenstilen und rumänischen Diakritika); `[S3]«` beschränkt ein Zitat auf Quelle 3. Ein Prüfer klassifiziert danach jedes Zitat (wörtlich, falsche Quelle, abgeschnitten, verändert, zusammengefügt, nicht belegt) und bewertet die Sätze außerhalb der Zitate; der Bericht sagt, wenn eine Arbeitsgrenze erreicht wurde (`limit_hit`). *Kann nicht:* den Text um die Zitate herum wahr machen; Streaming-Antworten enthalten keinen Zitatblock.

### Planer, Governor und Energiemesser

Modul `governor` (und `plan`, `hw`). In `corbul`: `plan`, `doctor`, `--governor`, `--no-governor`, `--verbose`, `--calibrate`. Sagt vor dem Laden, ob das Modell passt, streamt, langsam läuft oder nicht laufen kann, mit einer geschätzten Geschwindigkeit. Während des Laufs liest der Governor Temperatur, Speicher und Akku (Linux) und rät zu Threads und Pausen, ohne zu schwingen. *Kann nicht:* Geschwindigkeiten sind Schätzungen (`--calibrate` misst deine Maschine etwa eine Sekunde lang); die Energie ist eine Annahme, sofern kein RAPL-Zähler lesbar ist; die Kohlenstoffintensität (400 g/kWh) ist ein Platzhalter.

### KV-Cache-Richtlinien und Sitzungs-Snapshots

Module `kvpolicy`, `snapshot`. In `corbul`: `--kv-policy streaming|h2o|snapkv`, `--kv-budget`, `--kv-sinks`, `--session DIR`, `--session-cap`, `--compress`. Halten lange Chats innerhalb eines Speicherbudgets (Attention-Sinks, Heavy Hitters, Prompt-Kompression) und speichern oder setzen ein Gespräch aus einer Datei mit Prüfsumme fort, wobei das längste gespeicherte Präfix wiederverwendet wird. Snapshots können einen HMAC mit Schlüssel tragen. *Kann nicht:* Die Verdrängung ist absichtlich verlustbehaftet; `--kv-policy` und `--session` lassen sich nicht kombinieren; sehr lange Chats können den trainierten Kontext überschreiten.

### Reproduzierbarer Modus und Quittungen

Module `simd` (Schalter), `receipt`. In `corbul`: `--reproducible`, `--receipt FILE`, `--receipt-key`, `--receipt-key-file`, `--receipt-topk`, `--keep-steps`, `--leaves`, `verify-receipt`, `replay`. Eine Quittung hält Modell (einen Merkle-Fingerabdruck der Gewichte), Tokenizer, Sampler-Einstellungen, Prompt und Ausgabe in einer Hash-Kette fest, versiegelt mit SHA-256 und einem optionalen HMAC-Schlüssel; `replay` führt sie erneut aus. Beschädigte Gewichte werden auf einen Tensor und einen Chunk eingegrenzt. *Kann nicht:* einem Fremden etwas beweisen, ohne den HMAC-Schlüssel oder einen erneuten Lauf (keine Public-Key-Signaturen); MoE- und Zustandsraum-Modelle werden abgelehnt; ARM ungetestet; der reproduzierbare Modus kostet Geschwindigkeit.

### Sprachwerkzeuge mit Rumänisch an erster Stelle

Modul `lang`. In `corbul`: `lang detect`, `lang normalize`, `--lang CODE` (nicht im Server). Normalisiert ș/ț und ş/ţ, repariert Mojibake, erkennt 27 Sprachen, verbietet beim Dekodieren fremde Schriften (Rumänisch: kein ş ţ), formatiert rumänische Zahlen, Geldbeträge und Daten und stellt fehlende Diakritika wieder her, mit einer Markierung an jedem unsicheren Wort. *Kann nicht:* Die Wiederherstellung nutzt ein Lexikon von etwa 650 Formen und wurde nicht an einem Korpus gemessen; mehrdeutige Wörter werden durch einige Kontextregeln aufgelöst; die Spracherkennung braucht etwa fünf Wörter.

### Unsicherheitsanalysen

Modul `confidence`. Nur Bibliothek (Demo-Beispiel `confidence_demo`). Statistiken pro Token, Wort, Satz und Antwort für eine Heatmap, eine transparente Heuristik für das Halluzinationsrisiko, Self-Consistency-Werte und Kalibrierungswerkzeuge. *Kann nicht:* Der Risikowert ist eine Heuristik, kein Wahrheitsdetektor.

## Rechenaufwand zur Laufzeit, Abstimmung, Retrieval

### Suche und Lookahead

Modul `search`. In `corbul`: `search --method beam|best-of-n|vote` (`--width`, `--samples`, `--answer number|text`, `--length-penalty`, `--all`, `--explain`, `--json`) und `--lookahead` bei `generate` (`--ngram-min`, `--ngram-max`). Beam-Suche, Best-of-N, Voting mit Self-Consistency, ein Gesprächsbaum (Bibliothek), spekulatives Dekodieren per Prompt-Lookup. *Kann nicht:* Lookahead hilft nur, wenn sich Text früheren Text wiederholt; Best-of-N braucht einen guten Bewerter.

### Spekulatives Dekodieren

Module `spec`, `selfspec`. In `corbul`: `--draft MODEL`, `--lookahead`, `--self-spec`, `--skip-layers auto|LIST`, `--draft-len`. Dieselbe gierige Antwort und dieselbe Sampling-Verteilung mit weniger vollen Durchläufen; die Ausgabe erscheint am Ende. Nur `--temperature`, `--top-k`, `--top-p`; nicht mit strukturierter Ausgabe, Quittungen, Sitzungen oder dem Mesh. *Kann nicht:* eine Beschleunigung versprechen; die Selbstspekulation fällt auf einfache Schritte zurück bei Gleitfenster-Ringen, die überlaufen würden, bei verdichteten Caches und bei ausgelagerten Layern.

### Automatische gemischte Quantisierung und der Quantisierer

Module `autoquant`, `quantize`. In `corbul`: `autoquant FOLDER --budget 4.5bpw|6g|Q4_K` (`--recipe`, `--report`, `--candidates`, `--base`, `--imatrix`, `--sample-rows`) und `quantize IN OUT.gguf --type NAME | --recipe FILE`. Misst den Gewichtsfehler pro Tensorgruppe in jedem Format und wählt Formate unter einem Größenbudget (exaktes Rucksackproblem oder gierig). *Kann nicht:* Es schätzt den Gewichtsfehler, nicht die Antwortqualität; es nimmt an, dass sich Schäden addieren; i-Quants gibt es nicht.

### Steuerungsvektoren und Logit-Linse

Modul `steer`. In `corbul`: `steer train|apply|lens` (`--positive`, `--negative`, `--method mean-diff|pca`, `--out`, `--vector`, `--strength`, `--layers`, `--positions`, `--no-attention`). Lernt aus zwei Prompt-Mengen eine Richtung und addiert sie zum Residualstrom, ohne die Gewichte zu ändern; die Linse zeigt, was jeder Layer vorhersagen würde. *Kann nicht:* mit Zustandsraum-Modellen, strukturierter Ausgabe, Spekulation, Quittungen oder Sitzungen arbeiten; ein Vektor aus einem Fine-Tune wird von einem anderen mit gleicher Form akzeptiert.

### Lokales Retrieval

Modul `rag`. In `corbul`: `rag add|search|ask` (`--query`, `--question`, `--k`, `--chunk-chars`, `--overlap`, `--embed-model`). Chunking mit Rücksicht auf Rumänisch, ein HNSW-Index in reinem Rust, BM25 mit Diakritika-Faltung, Rangfusion, Passagen mit Nummern `[S1]…`, die in zitatgesperrte Quellenangaben einfließen, Dateien mit Prüfsumme. *Kann nicht:* Der Standard-Embedder ist ein Hashing-Embedder (Wörter und Wortteile, keine Bedeutung); `--embed-model` nutzt verborgene Zustände eines generativen Modells (schwach); die CLI indiziert bei jedem Aufruf neu und liest reinen Text, keine PDF- oder Office-Dateien (die Bibliothek hat Hilfen zum Einlesen).

### Modell-Merging

Modul `merge`. In `corbul`: `merge CONFIG.json` (`--out`, `--model`, `--dry-run`, `--max-memory`, `--overwrite`, `--verbose`). Linear, SLERP, Task-Arithmetik, TIES, DARE (mit Seed) und Layer-Ausschnitte, jeweils ein Tensor nach dem anderen. Formen und Bereiche werden vor dem Schreiben geprüft; Symlinks werden nicht verfolgt. *Kann nicht:* Qualität garantieren; schreibt nur safetensors-Gewichte, nie quantisierte, und `config.json` und den Tokenizer kopierst du selbst.

### Text-Wasserzeichen

Modul `watermark`. In `corbul`: `generate --watermark KEY` (`--wm-gamma`, `--wm-delta`, `--wm-context`) und `watermark detect MODEL --key K --text T | --file F` (`--no-dedup`). Eine „Green-List“-Verzerrung mit Schlüssel und ein z-Score-Detektor. *Kann nicht:* einen Beweis liefern; kurze Texte, Code und Zitate tragen wenig Signal; Umschreiben entfernt es; wer den Schlüssel hat, kann es fälschen; kein Zeichen beweist nicht, dass ein Mensch geschrieben hat.

## Multimodal (nur Bibliothek)

### Vision

Module `vision`, `imageio`. Ein Encoder im CLIP/SigLIP-Stil, ein Projektor, Kachelung und ein Hook `Model::forward_segments`, um Embeddings in einen Prompt einzuspeisen; Decoder für PNG, JPEG (Baseline und progressiv), BMP und PNM, nach den Spezifikationen geschrieben, mit Größengrenzen. *Kann nicht:* aus `corbul` genutzt werden (nur Text); kein echter Checkpoint wurde ausgeführt.

### Audio

Modul `audio`. WAV lesen und schreiben, Resampling, FFT, Log-Mel-Merkmale, Spracherkennung (Voice-Activity-Detection), ein Runner im Whisper-Layout. *Kann nicht:* aus `corbul` genutzt werden; kein echter Whisper-Checkpoint wurde ausgeführt.

## Zustandsraum, Hybrid und GPU (siehe ENGINE.md)

Die Module `ssm`, `hybrid`, `mla` und das Crate `companion-engine-gpu` werden samt ihren Grenzen in ENGINE.md beschrieben.
