# Leichte lokale Modelle (geprüft am 2026-11-17)

„Leicht“ heißt nur, dass der Download bequem in 4/8/16/32 GB RAM passt. Es sagt nichts über die Qualität; keine Reihenfolge hier ist ein Benchmark. Quelle der Wahrheit: `crates/core/src/light_models.rs`. Alle Einträge laufen über jeden OpenAI-kompatiblen Server (Ollama, llama.cpp, LM Studio), der in `server_url` eingetragen ist; keiner steht im colibri-Katalog.

RAM-Stufen: Minimum = Größe + 2 GB, komfortabel = 1,5 × Größe + 2 GB, aufgerundet auf 4/8/16/32.

| id | Größe GB (niedrig–hoch angegeben) | Kontext | Lizenz | RAM min. / komfortabel |
|---|---|---|---|---|
| lfm3-1.7b | ~1.0 (4 Bit) | 256K | LFM Open License v1.0 (**nicht standardisiert**) | 4 / 4 |
| qwen3.5-0.8b | 1.2-1.3 | 256K | apache-2.0 | 4 / 4 |
| lfm2.5-2.6b | 1.67 (Q4_K_M) | 131,072 | lfm1.0 (**nicht standardisiert**) | 4 / 8 |
| qwen3.5-4b | 3.3-4.0 | 256K | apache-2.0 | 8 / 8 |
| qwen3.5-9b | 6.6-7.6 | 256K | apache-2.0 | 16 / 16 |
| olmoe-1b-7b (colibri, alt 2024) | 7 (int8 umgewandelt) | k. A. | Apache-2.0 | 8 (README) / 16 |
| gemma4-e2b | 4.6-7.5 | 128K | Apache 2.0 | 16 / 16 |
| gemma4-12b | 7.7-8.0 | 256K | Apache 2.0 | 16 / 16 |
| gemma4-e4b | 6.6-9.5 | 128K | Apache 2.0 | 16 / 16 |
| gpt-oss-20b | 14 (MXFP4) | k. A. | Apache 2.0 | 16 / 32 |
| gemma4-26b-a4b | 16-19 | 256K | Apache 2.0 | 32 / 32 |

(k. A. = keine Angabe.) Die Ollama-Seiten geben pro Tag einen Größenbereich an und nennen nie die Standard-Quantisierung, daher nutzen die Stufen das obere Ende. Nur LFM2.5 (Q4_K_M), gpt-oss (MXFP4) und OLMoE (README) haben eine Größe, die an ein benanntes Format gebunden ist. Die 7,5 GB von Gemma 4 E2B entsprechen dem separaten Tag `e2b-mlx`; die Seite sagt nicht, welche Variante welche ist. Nemotron 3.5 Bolt E4B läuft nur auf GPU mit vLLM und wird von `for_ram` nie angeboten.

## Nicht geprüft (nicht in der Liste)
- Qwen3.5 2B: Größe von Ollama gelesen, HF-Lizenz nicht lesbar (0.8B, 4B, 9B sind Apache-2.0).
- Granite 4.2 3B/8B/30B: IBM-Doku nennt Apache 2.0, 128K, Werkzeugaufrufe; Datum 25. Aug. 2026 nur von Sekundärseiten; keine Größe in GB.
- MiMo-V2.6-Distill-Qwen-9B: ursprüngliche HF-Karte nicht lesbar; nur ein Spiegel.
- Mellum2 12B-A2.5B: Apache 2.0, Kontext 131,072, aber keine GGUF-Größe und kein Jahr; keine Seite erwähnt „Mellum2.1“.
- Nemotron 3.5 Bolt E4B: nur GPU mit vLLM, ausgeschlossen.
- Qwen3.8: Qwen veröffentlicht kein Modell unter 27B; empero-ais Qwen3.8-4B ist eine Destillation von Dritten.
- Flint 40B-A2.5B und LFM3-VL-2B (Zeilen 252–253 der colibri-README), PLaMo 3 Translate, Phi-4-mini, Granite 4.1, Bonsai 2; Gemma-4-QAT-Tags erscheinen nicht auf der Ollama-Seite.
- Eine Suche nach Veröffentlichungen von Sept.–Nov. 2026 lieferte nur Ranglistenseiten Dritter, keine als Primärquelle gelesen.
- Nicht leicht: Ripple-65B, Kimi K3.1 Pro, ERNIE-5.5-Flash-Base, Qwen3.5 122B, PLaMo 3 31B.

## Quellen
Ollama: Bibliotheksseiten gemma4, qwen3.5, gpt-oss:20b. Google-Gemma-4-Modellkarte (ai.google.dev) und huggingface.co/google/gemma-4-12B. huggingface.co/Qwen/Qwen3.5-0.8B. huggingface.co/LiquidAI/LFM3-1.7B (+ -GGUF). Auf keiner Seite gefunden: Veröffentlichungsmonat von Gemma 4, Kontextlänge von gpt-oss, llama.cpp-Befehle (keine ausgegeben).

## Agenten (Werkzeugaufrufe, agentisches Programmieren, langer Kontext)

Was Companion von einem Modell braucht (gelesen in `chat.rs`, `mcp.rs`): MCP-Werkzeuge werden als einfacher Text angeboten, ein umzäunter `tool`-Block, den das Modell schreiben kann und den `mcp::parse_tool_request` auswertet; **kein Modell braucht also das OpenAI-Feld `tools`**. Das Modell muss dieses Format befolgen, per SSE mit `include_usage` streamen, darf `reasoning_content` oder `<think>`-Text ausgeben (von `chat.rs` getrennt) und braucht Kontext für bis zu 5 Runden Werkzeugausgabe. `image_url`-Teile zählen nur für Bildverarbeitung. Ein „tools“-Abzeichen ist also ein gutes Zeichen, keine Voraussetzung.

`for_agents(ram_gb)` liefert nur Modelle mit auf einer Primärseite dokumentierten Werkzeugaufrufen, mit derselben Passregel wie `for_ram`. Dokumentierte Quellen:

| Modell | Werkzeugaufrufe (Quelle) | agentisches Programmieren | strukturierte Ausgabe |
|---|---|---|---|
| gpt-oss-20b | Ollama-Abzeichen `tools`; „designed for agentic tasks“ | ungeprüft | ungeprüft |
| gemma4 e2b / e4b / 12b / 26b-a4b | Ollama-Abzeichen `tools`, „native function-calling support“ | ungeprüft | ungeprüft |
| qwen3.5 0.8b / 4b | Ollama-Abzeichen `tools` (Familie) | ungeprüft | ungeprüft |
| lfm3-1.7b | HF-Karte: „designed for local agents, tool calling“ (XML-Aufrufe) | ungeprüft | HF-Karte: „structured outputs“ (als Test ungeprüft) |
| qwen3.5 9b | Ollama-Abzeichen `tools` | ungeprüft | ungeprüft |
| lfm2.5-2.6b, olmoe-1b-7b | ungeprüft | ungeprüft | ungeprüft |

Nicht in der Agentenliste (keine Größe oder Lizenz auf einer Primärseite, oder nicht lokal):
- Devstral Small 2 24B: Mistral-Doku nennt Apache 2.0, 256k, Funktionsaufrufe, strukturierte Ausgaben, 2025-12-09, aber der API-Status ist Deprecated (eingestellt am 2026-03-31) und keine lokale Größe ist angegeben.
- Qwen3-Coder-30B-A3B: die Qwen-Karte nennt Werkzeugaufrufe und agentisches Programmieren, Kontext 262,144, gibt aber weder Lizenz noch Größe an; die colibri-README führt es in Zeile 209.
- Granite 4.2: IBM-Doku dokumentiert Werkzeugaufrufe und 128K Kontext, keine Größe.
- MiMo-Distill, Mellum2 (agentisches Programmieren nur auf ungelesenen Seiten behauptet), Nemotron 3.5 Bolt (nur vLLM).

Grenzen: Kleine Modelle rufen Werkzeuge unzuverlässig auf, und ein Abzeichen sagt, dass die Chatvorlage Werkzeuge unterstützt, nicht dass das Modell gut wählt. Nichts davon wurde mit den Automatisierungen von Companion ausgeführt.

## Nachprüfung 2026-11-17 (gesamter Katalog)
Heute neu gelesen: Ollama gemma4 (alle Größen/Kontexte stimmen), qwen3.5 (Größen, Abzeichen; die Seite nennt keine Lizenz), gpt-oss:20b (14 GB, Werkzeuge, Apache 2.0, kein Kontext); HF gemma-4-12B, Qwen3.5-0.8B/4B, LFM3-1.7B, LFM2.5-2.6B; IBM-Granite-4.2-Doku.
Korrigiert: LFM2.5-Kontext (131,072, war „nicht angegeben“); LFM3-Werkzeugaufrufe jetzt dokumentiert; Qwen3.5 9B hinzugefügt.
Nicht neu gelesen: OLMoE (README-Zahlen), Lizenz von Gemma 4 E2B/E4B/26B (Familienkarte früher gelesen; 12B-Karte heute gelesen), LFM3-GGUF-Größe, LFM2.5-Q4_K_M-Größe (HF-Seite hat nur Links), Veröffentlichungsmonate (keine Primärseite druckt sie; der 27. Okt. von LFM3 stammt aus einer Zusammenfassung).
Die Ollama-Seiten gemma4, qwen3.5 und gpt-oss geben keine Lizenz an; die Lizenzen stammen von den HF-/Google-Karten.
