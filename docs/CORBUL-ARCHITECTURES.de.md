# Modelle, die corbul ausführt und colibri nicht

Companion startet **corbul**, die eigene Engine, für jede GGUF-Datei, die colibri nicht importieren kann. Seit alpha.18 geht es für die folgenden Familien **direkt** dorthin, ohne vorher einen fehlschlagenden Import.

## Was corbul ausführt
- **Gemma 4** (E2B, E4B, 26B-A4B mit Experten, 31B) aus einer `.gguf`: gemischte gleitende und globale Aufmerksamkeit, Kopfgrößen je Schicht, Schlüssel als Werte auf globalen Schichten, proportionales RoPE, geteiltes K/V in den letzten Schichten, Embeddings je Schicht, das dichte MLP neben den gerouteten Experten, Begrenzung der Ausgabe-Logits.
- **GPT-2, GPT-J, GPT-NeoX/Pythia, Falcon, BLOOM, MPT**, aus einer `.gguf`: LayerNorm, fusioniertes oder getrenntes QKV, gelernte / rotierende / ALiBi-Positionen, paralleles Residuum. Nur an kleinen synthetischen Modellen getestet. OPT hat keine GGUF-Konvertierung.
- **Gemma 1–3, Phi, OLMo, Granite, Cohere, InternLM2, StableLM, StarCoder2, EXAONE, Baichuan, MiniCPM, SmolLM3**, Llama und Mistral, Qwen und die übrigen Expertenfamilien, die die Engine schon hatte.
- **Mamba, Mamba-2, RWKV-6/7, Jamba, Bamba** (rekurrent und hybrid).

`corbul serve MODELL` beantwortet `/health`, `/v1/models`, `/v1/chat/completions` und `/v1/completions` (auch als Stream) mit der Chatvorlage des Modells, `stop`, `max_tokens`, `temperature`, `top_p`, `top_k` und `seed`. Eine Unterhaltung, die die vorige fortsetzt, liest ihren Verlauf nicht erneut.

## Was für diese Modelle nicht geht
Werkzeugaufrufe, strukturierte Ausgabe, mehrere Antworten je Anfrage, Embeddings, Kontextpakete (Companion nutzt dann den gewöhnlichen Chat), Bildeingabe, mehrere Modelle gleichzeitig. Gemma 4 wird nur aus `.gguf` gelesen, nicht aus safetensors-Ordnern. Eine GGUF ohne eigenes Vokabular kann eine danebenliegende `tokenizer.json` verwenden.

## Was nicht geprüft ist
Alles ist an **kleinen synthetischen Modellen** getestet (eigene Invarianten: Cache gleich Neuberechnung, gleitende Fenster, geteiltes K/V, Routing, quantisierte Experten, die HTTP-Antworten). **Nichts wurde mit einer echten Gemma-4-Datei verglichen.** Wirken die Antworten einer echten Datei falsch, führe `corbul serve DATEI.gguf` selbst aus und schicke die Ausgabe; zuerst prüft man das Muster gleitend/global und die RoPE-Tabellen.

## Noch nicht unterstützt
OPT, gpt-oss, PhiMoE, DBRX, Llama 4, GLM-4-MoE, Hunyuan, Ernie-MoE, Grok, das vollständige DeepSeek-Modell, MiniMax, Zamba, Encoder-Decoder-Modelle. corbul nennt sie beim Namen und rät nicht.
