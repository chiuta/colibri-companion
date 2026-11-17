# Modelli locali leggeri (verificato il 2026-11-17)

«Leggero» significa soltanto che il download sta comodamente in 4/8/16/32 GB di RAM. Non dice nulla sulla qualità; nessun ordine qui è un benchmark. Fonte di verità: `crates/core/src/light_models.rs`. Tutte le voci girano su qualsiasi server compatibile OpenAI (Ollama, llama.cpp, LM Studio) impostato in `server_url`; nessuna è nel catalogo colibri.

Fasce di RAM: minimo = dimensione + 2 GB, comoda = 1,5 × dimensione + 2 GB, arrotondate per eccesso a 4/8/16/32.

| id | dimensione GB (bassa–alta stampata) | contesto | licenza | RAM min / comoda |
|---|---|---|---|---|
| lfm3-1.7b | ~1.0 (4 bit) | 256K | LFM Open License v1.0 (**non standard**) | 4 / 4 |
| qwen3.5-0.8b | 1.2-1.3 | 256K | apache-2.0 | 4 / 4 |
| lfm2.5-2.6b | 1.67 (Q4_K_M) | 131,072 | lfm1.0 (**non standard**) | 4 / 8 |
| qwen3.5-4b | 3.3-4.0 | 256K | apache-2.0 | 8 / 8 |
| qwen3.5-9b | 6.6-7.6 | 256K | apache-2.0 | 16 / 16 |
| olmoe-1b-7b (colibri, vecchio 2024) | 7 (int8 convertito) | n/s | Apache-2.0 | 8 (README) / 16 |
| gemma4-e2b | 4.6-7.5 | 128K | Apache 2.0 | 16 / 16 |
| gemma4-12b | 7.7-8.0 | 256K | Apache 2.0 | 16 / 16 |
| gemma4-e4b | 6.6-9.5 | 128K | Apache 2.0 | 16 / 16 |
| gpt-oss-20b | 14 (MXFP4) | n/s | Apache 2.0 | 16 / 32 |
| gemma4-26b-a4b | 16-19 | 256K | Apache 2.0 | 32 / 32 |

(n/s = non specificato.) Le pagine Ollama stampano un intervallo di dimensione per tag e non nominano mai la quantizzazione predefinita, perciò le fasce usano il limite alto. Solo LFM2.5 (Q4_K_M), gpt-oss (MXFP4) e OLMoE (README) hanno una dimensione legata a un formato nominato. I 7,5 GB di Gemma 4 E2B coincidono con il tag separato `e2b-mlx`; la pagina non dice quale variante sia quale. Nemotron 3.5 Bolt E4B gira solo su GPU con vLLM e non è mai offerto da `for_ram`.

## Non verificato (non in elenco)
- Qwen3.5 2B: dimensione letta su Ollama, licenza HF non leggibile (0.8B, 4B, 9B sono Apache-2.0).
- Granite 4.2 3B/8B/30B: la documentazione IBM dice Apache 2.0, 128K, chiamata a strumenti; data 25 ago 2026 solo da pagine secondarie; nessuna dimensione in GB.
- MiMo-V2.6-Distill-Qwen-9B: scheda HF originale non leggibile; solo uno specchio.
- Mellum2 12B-A2.5B: Apache 2.0, contesto 131,072, ma senza dimensione GGUF né anno; nessuna pagina cita «Mellum2.1».
- Nemotron 3.5 Bolt E4B: solo GPU con vLLM, escluso.
- Qwen3.8: Qwen non pubblica alcun modello sotto 27B; il Qwen3.8-4B di empero-ai è una distillazione di terzi.
- Flint 40B-A2.5B e LFM3-VL-2B (righe 252–253 del README di colibri), PLaMo 3 Translate, Phi-4-mini, Granite 4.1, Bonsai 2; i tag Gemma 4 QAT non compaiono nella pagina Ollama.
- Una ricerca delle uscite di set.–nov. 2026 ha restituito solo pagine di classifiche di terzi, nessuna letta come fonte primaria.
- Non leggeri: Ripple-65B, Kimi K3.1 Pro, ERNIE-5.5-Flash-Base, Qwen3.5 122B, PLaMo 3 31B.

## Fonti
Ollama: pagine di libreria gemma4, qwen3.5, gpt-oss:20b. Scheda del modello Google Gemma 4 (ai.google.dev) e huggingface.co/google/gemma-4-12B. huggingface.co/Qwen/Qwen3.5-0.8B. huggingface.co/LiquidAI/LFM3-1.7B (+ -GGUF). Non trovati in nessuna pagina: il mese di uscita di Gemma 4, la lunghezza di contesto di gpt-oss, i comandi llama.cpp (nessuno emesso).

## Agenti (chiamata a strumenti, programmazione agentica, contesto lungo)

Cosa serve a Companion da un modello (letto in `chat.rs`, `mcp.rs`): gli strumenti MCP sono offerti come testo semplice, un blocco `tool` recintato che il modello può scrivere e che `mcp::parse_tool_request` analizza, quindi **nessun modello ha bisogno del campo `tools` di OpenAI**. Il modello deve seguire quel formato, trasmettere via SSE con `include_usage`, può emettere `reasoning_content` o testo `<think>` (separato da `chat.rs`) e ha bisogno di contesto per fino a 5 giri di risultati di strumenti. Le parti `image_url` contano solo per la visione. Un badge «tools» è quindi un buon segno, non un requisito.

`for_agents(ram_gb)` restituisce solo modelli con chiamata a strumenti documentata su una pagina primaria, con la stessa regola di adattamento di `for_ram`. Fonti documentate:

| modello | chiamata a strumenti (fonte) | programmazione agentica | output strutturato |
|---|---|---|---|
| gpt-oss-20b | badge `tools` di Ollama; «designed for agentic tasks» | non verificato | non verificato |
| gemma4 e2b / e4b / 12b / 26b-a4b | badge `tools` di Ollama, «native function-calling support» | non verificato | non verificato |
| qwen3.5 0.8b / 4b | badge `tools` di Ollama (famiglia) | non verificato | non verificato |
| lfm3-1.7b | scheda HF: «designed for local agents, tool calling» (chiamate XML) | non verificato | scheda HF: «structured outputs» (non verificato come test) |
| qwen3.5 9b | badge `tools` di Ollama | non verificato | non verificato |
| lfm2.5-2.6b, olmoe-1b-7b | non verificato | non verificato | non verificato |

Non nell'elenco degli agenti (senza dimensione o licenza su una pagina primaria, o non locali):
- Devstral Small 2 24B: la documentazione Mistral dice Apache 2.0, 256k, chiamata di funzioni, output strutturati, 2025-12-09, ma lo stato dell'API è Deprecated (ritirata il 2026-03-31) e non è stampata alcuna dimensione locale.
- Qwen3-Coder-30B-A3B: la scheda Qwen dice chiamata a strumenti e programmazione agentica, contesto 262,144, ma non stampa licenza né dimensione; il README di colibri lo elenca alla riga 209.
- Granite 4.2: la documentazione IBM documenta la chiamata a strumenti e il contesto 128K, senza dimensione.
- MiMo distill, Mellum2 (programmazione agentica dichiarata solo su pagine non lette), Nemotron 3.5 Bolt (solo vLLM).

Limiti: i modelli piccoli chiamano gli strumenti in modo inaffidabile, e un badge dice che il template di chat supporta gli strumenti, non che il modello scelga bene. Nulla di quanto sopra è stato eseguito con le automazioni di Companion.

## Ricontrollo del 2026-11-17 (catalogo completo)
Riletti oggi: Ollama gemma4 (tutte le dimensioni/i contesti coincidono), qwen3.5 (dimensioni, badge; la pagina non nomina alcuna licenza), gpt-oss:20b (14 GB, strumenti, Apache 2.0, nessun contesto); HF gemma-4-12B, Qwen3.5-0.8B/4B, LFM3-1.7B, LFM2.5-2.6B; documentazione IBM Granite 4.2.
Corretto: contesto di LFM2.5 (131,072, era «non indicato»); chiamata a strumenti di LFM3 ora documentata; aggiunto Qwen3.5 9B.
Non riletti: OLMoE (cifre del README), licenza Gemma 4 E2B/E4B/26B (scheda di famiglia letta prima; scheda 12B letta oggi), dimensione GGUF di LFM3, dimensione Q4_K_M di LFM2.5 (la pagina HF ha solo link), mesi di uscita (nessuna pagina primaria li stampa; il 27 ott. di LFM3 viene da un riepilogo).
Le pagine Ollama gemma4, qwen3.5 e gpt-oss non stampano la licenza; le licenze vengono dalle schede HF/Google.
