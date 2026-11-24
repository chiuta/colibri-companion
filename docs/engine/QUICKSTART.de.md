# Schnellstart: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` ist die Kommandozeile der Clean-Room-Engine ([ENGINE.de.md](ENGINE.de.md)). Diese Seite nennt nur Befehle und Optionen, die zum Zeitpunkt des Schreibens in `crates/engine-cli` existieren; die Kommandozeile wird noch erweitert, das letzte Wort hat also `corbul help`. Einige Optionen unten werden vom Code gelesen, stehen aber noch nicht in `corbul help`; sie sind mit *(noch nicht in der Hilfe)* markiert. Was es noch nicht gibt, ist als **geplant** markiert.

**Ehrlicher Hinweis:** `corbul` wurde nur an winzigen synthetischen Modellen getestet. Rechne mit Ecken und Kanten bei echten Checkpoints und melde sie bitte.

## Bauen

```bash
cargo build --release -p colibri-rs-cli     # erzeugt target/release/corbul
```

Ein Modell ist entweder ein Ordner mit `config.json` + `*.safetensors` + `tokenizer.json` oder eine einzelne `.gguf`-Datei.

## Ansehen, herunterladen, ausführen

```bash
corbul inspect  ./my-model                  # oder file.gguf oder owner/repo[@rev]: Architektur, Größe, Unterstützung
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors; HF_TOKEN für geschützte Repos, HF_ENDPOINT für einen Mirror
corbul generate ./my-model --prompt "Hallo" --max-tokens 64
corbul chat     ./my-model --system "Fasse dich kurz."
corbul serve    ./my-model --port 8080      # OpenAI-/Ollama-/Anthropic-kompatibel, standardmäßig an 127.0.0.1 gebunden
```

Gemeinsame Optionen für `generate`, `chat` und `serve`: `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--kv f32|f16|q8`, `--no-mmap`, `--threads N`, `--ctx N`, `--model-name`, `--quiet`. `generate` nimmt außerdem `--prompt-file F` und `--raw` (ohne Chat-Vorlage). `corbul run` ist ein Alias für `generate`.

Mehrere Modelle bereitstellen: `--models-dir DIR`, `--alias NAME=MODELL`, `--max-loaded N`, `--sleep-idle-seconds S`, `--token T` (oder `CORBUL_TOKEN`), `--keys-file DATEI`, `--cors URSPRÜNGE`, `--log-prompts`, `--no-log`, `--drain-seconds S`, `--config-dir DIR`.

## Läuft es auf diesem Rechner?

```bash
corbul doctor                # was die Engine sieht: CPU-Befehlssätze, Kernel, RAM, Platte, Sensoren, Energiezähler
corbul plan ./my-model --context 8192            # Urteil: FITS / TIGHT / STREAMS / TOO SLOW / WON'T RUN, Geschwindigkeit, Ratschläge
corbul plan ./my-model --ram-budget 8g --calibrate   # tu so, als hättest du 8 GB; stoppt zuerst eine Sekunde lang den Rechner
corbul capabilities [--json]                     # was dieser Build kann und was nicht
```

Die Geschwindigkeiten sind Schätzungen, keine Messungen deines Modells.

## Größer als der RAM

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"                        # MoE: geroutete Experten werden von der Platte gestreamt
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --prompt "…"   # dicht: Schichten werden gestreamt (plattenbegrenzt; --stream-window noch nicht in der Hilfe)
```

## Mehrere Rechner (LAN-Mesh)

```bash
# auf jedem Hilfsrechner (der Schlüssel hat mindestens 8 Zeichen; oder CORBUL_MESH_KEY setzen)
corbul worker ./my-model --key SECRET --bind 0.0.0.0:7471          # --no-discovery lässt die UDP-Erkennung aus
# auf dem Hauptrechner
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key SECRET --prompt "…"
corbul generate ./my-model --discover --key SECRET --prompt "…"      # findet Worker im LAN
```

Der Verkehr ist authentifiziert, aber **nicht verschlüsselt**: Nimm ein vertrauenswürdiges Netz oder ein VPN. Nur über Loopback getestet.

## Strukturierte Ausgabe, Zitate, Adapter

```bash
corbul generate M --json --prompt "…"                 # gültige JSON-Ausgabe
corbul generate M --json-schema schema.json --prompt "…"      # Datei oder Inline-JSON; auch --regex MUSTER, --grammar DATEI (GBNF); nur eine der vier verwenden; noch nicht in der Hilfe
corbul generate M --sources notes.txt --sources paper.txt --prompt "Antworte mit Zitaten" --citations report.json   # Zitat-Sperre
corbul generate M --lora ./adapter:0.8                # PEFT- oder GGUF-Adapter mit Skalierung
```

Der Server akzeptiert `response_format`, `grammar` und `sources` und bietet `POST /v1/quotes/verify`.

## Lange Chats, Sitzungen, Reproduzierbarkeit

```bash
corbul generate M --session ./sessions --prompt "…"             # nutzt das längste gespeicherte Prompt-Präfix wieder (nicht mit --kv-policy)
corbul generate M --kv-policy h2o --kv-budget 2048 --prompt "…" # streaming | h2o | snapkv; absichtlich verlustbehaftet
corbul generate M --governor --prompt "…"                       # passt sich an Temperatur, Speicher und Akku an (Linux)
corbul generate M --reproducible --temperature 0 --prompt "hi" --receipt r.json --receipt-key SECRET
corbul verify-receipt r.json --key SECRET --model M            # prüft das Siegel (und die Gewichte, wenn --model angegeben ist)
corbul replay r.json --model M --key SECRET                    # führt die Erzeugung erneut aus und vergleicht
```

Quittungen funktionieren mit der einfachen Erzeugungsschleife, nicht mit MoE-Modellen. Optional: `--receipt-topk N`, `--keep-steps`, `--leaves` *(noch nicht in der Hilfe)*.

## Geplant (noch nicht in `corbul`)

- Laden von Mamba-/RWKV-/MLA-Modellen (DeepSeek-Stil), obwohl die Bibliothek die Bausteine hat.
- Die rumänischen Sprachwerkzeuge, Unsicherheitsanalyse, Suche und Lookahead, selbst-spekulative Dekodierung, Steuervektoren, Abruf, Fusion und Wasserzeichen (siehe [INNOVATIONS.de.md](INNOVATIONS.de.md)).
- Embedding- und Rerank-Endpunkte (der Server antwortet „not implemented“), Bild- und Audioeingaben.
- Das GPU-Backend und eine Weboberfläche.
