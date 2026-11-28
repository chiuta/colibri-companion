# Modele pe care corbul le rulează și colibri nu

Companion pornește **corbul**, motorul propriu, pentru orice GGUF pe care colibri nu-l poate importa. Din alpha.18 merge **direct** acolo pentru familiile de mai jos, fără un import eșuat înainte.

## Ce rulează corbul
- **Gemma 4** (E2B, E4B, 26B-A4B cu experți, 31B), dintr-un `.gguf`: atenție mixtă glisantă și globală, dimensiuni de cap pe strat, chei folosite ca valori pe straturile globale, RoPE proporțional, K/V partajat în ultimele straturi, embeddinguri pe strat, MLP-ul dens lângă experții rutați, limitarea logiturilor finale.
- **GPT-2, GPT-J, GPT-NeoX/Pythia, Falcon, BLOOM, MPT**, dintr-un `.gguf`: LayerNorm, QKV fuzionat sau separat, poziții învățate / rotative / ALiBi, rezidual paralel. Testate doar pe modele sintetice mici. OPT nu are conversie GGUF.
- **gpt-oss (20B, 120B)**, dintr-un `.gguf` (inclusiv experți MXFP4): sink-uri de atenție, fereastră glisantă, YaRN, activare de expert limitată. Testat doar pe modele sintetice mici. **GraniteMoE, GLM-4.5/4.6 MoE**: tot din `.gguf`.
- **Gemma 1–3, Phi, OLMo, Granite, Cohere, InternLM2, StableLM, StarCoder2, EXAONE, Baichuan, MiniCPM, SmolLM3**, Llama și Mistral, Qwen și celelalte familii cu experți pe care motorul le avea deja.
- **Mamba, Mamba-2, RWKV-6/7, Jamba, Bamba** (recurente și hibride).

`corbul serve MODEL` răspunde la `/health`, `/v1/models`, `/v1/chat/completions` și `/v1/completions` (și în flux), cu șablonul de chat al modelului, `stop`, `max_tokens`, `temperature`, `top_p`, `top_k` și `seed`. O conversație care o continuă pe precedenta nu își recitește istoricul.

## Ce nu face pentru aceste modele
Apeluri de unelte, ieșire structurată, mai multe răspunsuri per cerere, embeddinguri, pachete de context (Companion folosește atunci chatul obișnuit), imagini la intrare, mai multe modele deodată. Gemma 4 se citește doar din `.gguf`, nu din foldere safetensors. Un GGUF fără vocabular propriu poate folosi un `tokenizer.json` pus lângă el.

## Ce nu este verificat
Totul e testat pe **modele sintetice mici** (invariante proprii: cache egal cu recalculare, ferestre glisante, K/V partajat, rutare, experți cuantizați, răspunsurile HTTP). **Nimic nu a fost comparat cu un fișier Gemma 4 real.** Dacă răspunsurile unui fișier real par greșite, rulează singur `corbul serve FIȘIER.gguf` și trimite ieșirea; primul lucru de verificat sunt tiparul glisant/global și tabelele RoPE.

## Încă nesuportate
OPT, PhiMoE, DBRX, Llama 4, Hunyuan, Ernie-MoE, Grok, modelul DeepSeek complet, MiniMax, Zamba, modelele encoder-decoder. Pentru ele corbul spune pe nume că nu merg și nu ghicește.
