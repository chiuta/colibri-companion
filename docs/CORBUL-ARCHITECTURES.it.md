# Modelli che corbul esegue e colibri no

Companion avvia **corbul**, il motore proprio, per ogni GGUF che colibri non può importare. Da alpha.18 per le famiglie qui sotto va **direttamente** lì, senza un'importazione fallita prima.

## Cosa esegue corbul
- **Gemma 4** (E2B, E4B, 26B-A4B con esperti, 31B), da un `.gguf`: attenzione mista scorrevole e globale, dimensioni della testa per strato, chiavi usate come valori sugli strati globali, RoPE proporzionale, K/V condiviso negli ultimi strati, embedding per strato, l'MLP denso accanto agli esperti instradati, limite dei logit finali.
- **GPT-2, GPT-J, GPT-NeoX/Pythia, Falcon, BLOOM, MPT**, da un `.gguf`: LayerNorm, QKV fuso o separato, posizioni apprese / rotative / ALiBi, residuo parallelo. Provati solo su piccoli modelli sintetici. OPT non ha conversione GGUF.
- **Gemma 1–3, Phi, OLMo, Granite, Cohere, InternLM2, StableLM, StarCoder2, EXAONE, Baichuan, MiniCPM, SmolLM3**, Llama e Mistral, Qwen e le altre famiglie con esperti che il motore aveva già.
- **Mamba, Mamba-2, RWKV-6/7, Jamba, Bamba** (ricorrenti e ibridi).

`corbul serve MODELLO` risponde a `/health`, `/v1/models`, `/v1/chat/completions` e `/v1/completions` (anche in flusso) con il modello di chat del modello, `stop`, `max_tokens`, `temperature`, `top_p`, `top_k` e `seed`. Una conversazione che continua la precedente non rilegge la sua cronologia.

## Cosa non fa per questi modelli
Chiamate a strumenti, output strutturato, più risposte per richiesta, embedding, pacchetti di contesto (Companion usa allora la chat normale), immagini in ingresso, più modelli insieme. Gemma 4 si legge solo da `.gguf`, non da cartelle safetensors. Un GGUF senza vocabolario proprio può usare un `tokenizer.json` messo accanto.

## Cosa non è verificato
Tutto è provato su **piccoli modelli sintetici** (invarianti proprie: cache uguale al ricalcolo, finestre scorrevoli, K/V condiviso, instradamento, esperti quantizzati, le risposte HTTP). **Nulla è stato confrontato con un vero file Gemma 4.** Se le risposte di un file reale sembrano sbagliate, esegui tu stesso `corbul serve FILE.gguf` e invia l'output; si controllano prima lo schema scorrevole/globale e le tabelle RoPE.

## Ancora non supportati
OPT, gpt-oss, GraniteMoE, PhiMoE, DBRX, Llama 4, GLM-4-MoE, Hunyuan, Ernie-MoE, Grok, il modello DeepSeek completo, MiniMax, Zamba, modelli encoder-decoder. corbul li nomina e non indovina.
