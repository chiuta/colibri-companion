# Modele locale ușoare (verificat 2026-11-17)

„Ușor” înseamnă doar că descărcarea încape confortabil în 4/8/16/32 GB de RAM. Nu spune nimic despre calitate; nicio ordonare de aici nu este un test de performanță. Sursa de adevăr: `crates/core/src/light_models.rs`. Toate intrările rulează prin orice server compatibil OpenAI (Ollama, llama.cpp, LM Studio) setat în `server_url`; niciuna nu e în catalogul colibri.

Trepte de RAM: minim = mărime + 2 GB, confortabil = 1,5 × mărime + 2 GB, rotunjite în sus la 4/8/16/32.

| id | mărime GB (mic–mare, cum e tipărită) | context | licență | RAM minim / confortabil |
|---|---|---|---|---|
| lfm3-1.7b | ~1.0 (4-bit) | 256K | LFM Open License v1.0 (**nestandard**) | 4 / 4 |
| qwen3.5-0.8b | 1.2-1.3 | 256K | apache-2.0 | 4 / 4 |
| lfm2.5-2.6b | 1.67 (Q4_K_M) | 131,072 | lfm1.0 (**nestandard**) | 4 / 8 |
| qwen3.5-4b | 3.3-4.0 | 256K | apache-2.0 | 8 / 8 |
| qwen3.5-9b | 6.6-7.6 | 256K | apache-2.0 | 16 / 16 |
| olmoe-1b-7b (colibri, vechi 2024) | 7 (int8 convertit) | n/s | Apache-2.0 | 8 (README) / 16 |
| gemma4-e2b | 4.6-7.5 | 128K | Apache 2.0 | 16 / 16 |
| gemma4-12b | 7.7-8.0 | 256K | Apache 2.0 | 16 / 16 |
| gemma4-e4b | 6.6-9.5 | 128K | Apache 2.0 | 16 / 16 |
| gpt-oss-20b | 14 (MXFP4) | n/s | Apache 2.0 | 16 / 32 |
| gemma4-26b-a4b | 16-19 | 256K | Apache 2.0 | 32 / 32 |

(n/s = nespecificat.) Paginile Ollama tipăresc un interval de mărime per etichetă și nu numesc niciodată cuantizarea implicită, deci treptele folosesc capătul de sus. Doar LFM2.5 (Q4_K_M), gpt-oss (MXFP4) și OLMoE (README) au o mărime legată de un format numit. Cei 7,5 GB ai Gemma 4 E2B sunt egali cu eticheta separată `e2b-mlx`; pagina nu spune care variantă e care. Nemotron 3.5 Bolt E4B rulează doar pe GPU cu vLLM și nu este oferit niciodată de `for_ram`.

## Neverificat (nu e în listă)
- Qwen3.5 2B: mărimea citită din Ollama, licența de pe HF nu poate fi citită (0.8B, 4B, 9B sunt Apache-2.0).
- Granite 4.2 3B/8B/30B: documentația IBM spune Apache 2.0, 128K, apelare de unelte; data 25 aug. 2026 doar din pagini secundare; fără mărime în GB.
- MiMo-V2.6-Distill-Qwen-9B: cardul HF original nu poate fi citit; doar o oglindă.
- Mellum2 12B-A2.5B: Apache 2.0, context 131,072, dar fără mărime GGUF sau an; nicio pagină nu pomenește „Mellum2.1”.
- Nemotron 3.5 Bolt E4B: doar GPU cu vLLM, exclus.
- Qwen3.8: Qwen nu publică niciun model sub 27B; Qwen3.8-4B de la empero-ai este o distilare terță.
- Flint 40B-A2.5B și LFM3-VL-2B (rândurile 252–253 din README-ul colibri), PLaMo 3 Translate, Phi-4-mini, Granite 4.1, Bonsai 2; etichetele Gemma 4 QAT nu apar pe pagina Ollama.
- O căutare a lansărilor din sept.–nov. 2026 a întors doar pagini terțe cu clasamente, niciuna citită ca sursă primară.
- Nu sunt ușoare: Ripple-65B, Kimi K3.1 Pro, ERNIE-5.5-Flash-Base, Qwen3.5 122B, PLaMo 3 31B.

## Surse
Ollama: paginile din bibliotecă gemma4, qwen3.5, gpt-oss:20b. Cardul modelului Google Gemma 4 (ai.google.dev) și huggingface.co/google/gemma-4-12B. huggingface.co/Qwen/Qwen3.5-0.8B. huggingface.co/LiquidAI/LFM3-1.7B (+ -GGUF). Negăsite pe nicio pagină: luna lansării Gemma 4, lungimea de context gpt-oss, comenzi llama.cpp (niciuna emisă).

## Agenți (apelare de unelte, programare agentică, context lung)

Ce îi trebuie Companion de la un model (citit în `chat.rs`, `mcp.rs`): uneltele MCP sunt oferite ca text simplu, un bloc `tool` între garduri pe care modelul îl poate scrie și pe care îl parsează `mcp::parse_tool_request`, deci **niciun model nu are nevoie de câmpul `tools` din OpenAI**. Modelul trebuie să urmeze acel format, să transmită prin SSE cu `include_usage`, poate emite `reasoning_content` sau text `<think>` (separat de `chat.rs`) și are nevoie de context pentru până la 5 runde de rezultate de unelte. Părțile `image_url` contează doar pentru vedere. O insignă „tools” este deci un semn bun, nu o cerință.

`for_agents(ram_gb)` întoarce doar modele cu apelare de unelte documentată pe o pagină primară, cu aceeași regulă de potrivire ca `for_ram`. Surse documentate:

| model | apelare de unelte (sursa) | programare agentică | ieșire structurată |
|---|---|---|---|
| gpt-oss-20b | insigna `tools` din Ollama; „designed for agentic tasks” | neverificat | neverificat |
| gemma4 e2b / e4b / 12b / 26b-a4b | insigna `tools` din Ollama, „native function-calling support” | neverificat | neverificat |
| qwen3.5 0.8b / 4b | insigna `tools` din Ollama (familie) | neverificat | neverificat |
| lfm3-1.7b | card HF: „designed for local agents, tool calling” (apeluri XML) | neverificat | card HF: „structured outputs” (neverificat ca test) |
| qwen3.5 9b | insigna `tools` din Ollama | neverificat | neverificat |
| lfm2.5-2.6b, olmoe-1b-7b | neverificat | neverificat | neverificat |

Nu sunt în lista de agenți (fără mărime sau licență pe o pagină primară, sau nu sunt locale):
- Devstral Small 2 24B: documentația Mistral spune Apache 2.0, 256k, apelare de funcții, ieșiri structurate, 2025-12-09, dar starea API este Deprecated (retras 2026-03-31) și nu se tipărește nicio mărime locală.
- Qwen3-Coder-30B-A3B: cardul Qwen spune apelare de unelte și programare agentică, context 262,144, dar nu tipărește licență sau mărime; README-ul colibri îl listează la rândul 209.
- Granite 4.2: documentația IBM documentează apelarea de unelte și contextul de 128K, fără mărime.
- MiMo distill, Mellum2 (programare agentică susținută doar pe pagini necitite), Nemotron 3.5 Bolt (doar vLLM).

Limite: modelele mici apelează unelte nesigur, iar o insignă spune că șablonul de chat suportă unelte, nu că modelul alege bine. Nimic de aici nu a fost rulat cu automatizările Companion.

## Reverificare 2026-11-17 (tot catalogul)
Recitite azi: Ollama gemma4 (toate mărimile/contextele se potrivesc), qwen3.5 (mărimi, insigne; pagina nu numește nicio licență), gpt-oss:20b (14 GB, unelte, Apache 2.0, fără context); HF gemma-4-12B, Qwen3.5-0.8B/4B, LFM3-1.7B, LFM2.5-2.6B; documentația IBM Granite 4.2.
Corectat: contextul LFM2.5 (131,072, era „nespecificat”); apelarea de unelte LFM3 acum documentată; Qwen3.5 9B adăugat.
Nerecitite: OLMoE (cifre din README), licența Gemma 4 E2B/E4B/26B (cardul familiei citit mai devreme; cardul 12B citit azi), mărimea GGUF LFM3, mărimea LFM2.5 Q4_K_M (pagina HF are doar linkuri), lunile lansărilor (nicio pagină primară nu le tipărește; 27 oct. pentru LFM3 vine dintr-o sinteză).
Paginile Ollama gemma4, qwen3.5 și gpt-oss nu tipăresc licența; licențele vin din cardurile HF/Google.
