# Könnyű helyi modellek (ellenőrizve: 2026-11-17)

A „könnyű” csak annyit jelent, hogy a letöltés kényelmesen elfér 4/8/16/32 GB RAM-ban. A minőségről semmit sem mond; az itteni sorrendek nem teljesítménymérések. Az igazság forrása: `crates/core/src/light_models.rs`. Minden tétel bármely OpenAI-kompatibilis szerveren fut (Ollama, llama.cpp, LM Studio), amelyet a `server_url` mutat; egyik sincs a colibri katalógusában.

RAM-lépcsők: minimum = méret + 2 GB, kényelmes = 1,5 × méret + 2 GB, felfelé kerekítve 4/8/16/32-re.

| id | méret GB (alsó–felső kiírt) | kontextus | licenc | RAM min. / kényelmes |
|---|---|---|---|---|
| lfm3-1.7b | ~1.0 (4 bites) | 256K | LFM Open License v1.0 (**nem szabványos**) | 4 / 4 |
| qwen3.5-0.8b | 1.2-1.3 | 256K | apache-2.0 | 4 / 4 |
| lfm2.5-2.6b | 1.67 (Q4_K_M) | 131,072 | lfm1.0 (**nem szabványos**) | 4 / 8 |
| qwen3.5-4b | 3.3-4.0 | 256K | apache-2.0 | 8 / 8 |
| qwen3.5-9b | 6.6-7.6 | 256K | apache-2.0 | 16 / 16 |
| olmoe-1b-7b (colibri, régi 2024) | 7 (int8-ra alakítva) | n/m | Apache-2.0 | 8 (README) / 16 |
| gemma4-e2b | 4.6-7.5 | 128K | Apache 2.0 | 16 / 16 |
| gemma4-12b | 7.7-8.0 | 256K | Apache 2.0 | 16 / 16 |
| gemma4-e4b | 6.6-9.5 | 128K | Apache 2.0 | 16 / 16 |
| gpt-oss-20b | 14 (MXFP4) | n/m | Apache 2.0 | 16 / 32 |
| gemma4-26b-a4b | 16-19 | 256K | Apache 2.0 | 32 / 32 |

(n/m = nincs megadva.) Az Ollama-oldalak címkénként méretsávot írnak ki, és soha nem nevezik meg az alapértelmezett kvantálást, ezért a lépcsők a felső értéket használják. Csak az LFM2.5 (Q4_K_M), a gpt-oss (MXFP4) és az OLMoE (README) mérete kötődik megnevezett formátumhoz. A Gemma 4 E2B 7,5 GB-ja megegyezik a külön `e2b-mlx` címkével; az oldal nem mondja meg, melyik változat melyik. A Nemotron 3.5 Bolt E4B csak vLLM-mel, GPU-n fut, és a `for_ram` soha nem ajánlja.

## Nem ellenőrzött (nincs a listán)
- Qwen3.5 2B: a méret az Ollamáról olvasva, a HF-licenc nem olvasható (a 0.8B, 4B, 9B Apache-2.0).
- Granite 4.2 3B/8B/30B: az IBM dokumentációja szerint Apache 2.0, 128K, eszközhívás; a 2026. aug. 25-i dátum csak másodlagos oldalakról; GB-méret nincs.
- MiMo-V2.6-Distill-Qwen-9B: az eredeti HF-kártya nem olvasható; csak tükör.
- Mellum2 12B-A2.5B: Apache 2.0, 131,072 kontextus, de nincs GGUF-méret és évszám; egy oldal sem említi a „Mellum2.1”-et.
- Nemotron 3.5 Bolt E4B: csak vLLM-es GPU, kizárva.
- Qwen3.8: a Qwen nem ad ki 27B alatti modellt; az empero-ai Qwen3.8-4B-je harmadik féltől származó desztilláció.
- Flint 40B-A2.5B és LFM3-VL-2B (a colibri README 252–253. sora), PLaMo 3 Translate, Phi-4-mini, Granite 4.1, Bonsai 2; a Gemma 4 QAT címkék nem szerepelnek az Ollama-oldalon.
- A 2026. szept.–nov. kiadásokra irányuló keresés csak harmadik fél rangsoroló oldalait adta, egyiket sem olvasták elsődleges forrásként.
- Nem könnyűek: Ripple-65B, Kimi K3.1 Pro, ERNIE-5.5-Flash-Base, Qwen3.5 122B, PLaMo 3 31B.

## Források
Ollama: a gemma4, qwen3.5, gpt-oss:20b könyvtároldalak. A Google Gemma 4 modellkártyája (ai.google.dev) és a huggingface.co/google/gemma-4-12B. huggingface.co/Qwen/Qwen3.5-0.8B. huggingface.co/LiquidAI/LFM3-1.7B (+ -GGUF). Egyetlen oldalon sem található: a Gemma 4 kiadási hónapja, a gpt-oss kontextushossza, llama.cpp-parancsok (egyet sem adtak ki).

## Ügynökök (eszközhívás, ügynöki programozás, hosszú kontextus)

Amire a Companionnak szüksége van egy modelltől (a `chat.rs` és `mcp.rs` alapján): az MCP-eszközöket egyszerű szövegként kínálja, egy kerített `tool` blokk formájában, amelyet a modell megírhat, és amelyet az `mcp::parse_tool_request` értelmez, így **egyetlen modellnek sincs szüksége az OpenAI `tools` mezőjére**. A modellnek követnie kell ezt a formátumot, SSE-n kell streamelnie `include_usage`-szal, kibocsáthat `reasoning_content`-et vagy `<think>` szöveget (a `chat.rs` szétválasztja), és kontextus kell neki legfeljebb 5 kör eszközkimenethez. Az `image_url` részek csak a látáshoz számítanak. A „tools” jelvény tehát jó jel, nem követelmény.

A `for_agents(ram_gb)` csak olyan modelleket ad vissza, amelyek eszközhívása elsődleges oldalon dokumentált, ugyanazzal az illeszkedési szabállyal, mint a `for_ram`. Dokumentált források:

| modell | eszközhívás (forrás) | ügynöki programozás | strukturált kimenet |
|---|---|---|---|
| gpt-oss-20b | Ollama `tools` jelvény; „designed for agentic tasks” | nem ellenőrzött | nem ellenőrzött |
| gemma4 e2b / e4b / 12b / 26b-a4b | Ollama `tools` jelvény, „native function-calling support” | nem ellenőrzött | nem ellenőrzött |
| qwen3.5 0.8b / 4b | Ollama `tools` jelvény (család) | nem ellenőrzött | nem ellenőrzött |
| lfm3-1.7b | HF-kártya: „designed for local agents, tool calling” (XML-hívások) | nem ellenőrzött | HF-kártya: „structured outputs” (tesztként nem ellenőrzött) |
| qwen3.5 9b | Ollama `tools` jelvény | nem ellenőrzött | nem ellenőrzött |
| lfm2.5-2.6b, olmoe-1b-7b | nem ellenőrzött | nem ellenőrzött | nem ellenőrzött |

Nincs az ügynöklistán (elsődleges oldalon nincs méret vagy licenc, vagy nem helyi):
- Devstral Small 2 24B: a Mistral dokumentációja szerint Apache 2.0, 256k, függvényhívás, strukturált kimenet, 2025-12-09, de az API állapota Deprecated (2026-03-31-én kivezetve), és helyi méret nincs kiírva.
- Qwen3-Coder-30B-A3B: a Qwen kártyája eszközhívást és ügynöki programozást említ, 262,144 kontextus, de nem ír ki licencet vagy méretet; a colibri README a 209. sorban sorolja fel.
- Granite 4.2: az IBM dokumentációja az eszközhívást és a 128K kontextust dokumentálja, méret nélkül.
- MiMo distill, Mellum2 (az ügynöki programozást csak el nem olvasott oldalak állítják), Nemotron 3.5 Bolt (csak vLLM).

Korlátok: a kis modellek megbízhatatlanul hívnak eszközöket, a jelvény pedig azt mondja, hogy a csevegősablon támogatja az eszközöket, nem azt, hogy a modell jól választ. Mindez nem futott a Companion automatizálásaival.

## Újraellenőrzés 2026-11-17 (a teljes katalógus)
Ma újraolvasva: Ollama gemma4 (minden méret/kontextus egyezik), qwen3.5 (méretek, jelvények; az oldal nem nevez meg licencet), gpt-oss:20b (14 GB, eszközök, Apache 2.0, kontextus nélkül); HF gemma-4-12B, Qwen3.5-0.8B/4B, LFM3-1.7B, LFM2.5-2.6B; az IBM Granite 4.2 dokumentációja.
Javítva: az LFM2.5 kontextusa (131,072, korábban „nincs megadva”); az LFM3 eszközhívása már dokumentált; a Qwen3.5 9B hozzáadva.
Nem olvasva újra: OLMoE (README-adatok), a Gemma 4 E2B/E4B/26B licence (a családi kártyát korábban olvasták; a 12B kártyát ma), az LFM3 GGUF-mérete, az LFM2.5 Q4_K_M mérete (a HF-oldalon csak hivatkozások), a kiadási hónapok (elsődleges oldal nem írja ki; az LFM3 okt. 27-i dátuma egy összefoglalóból származik).
Az Ollama gemma4, qwen3.5 és gpt-oss oldalai nem írnak ki licencet; a licencek a HF/Google kártyákból jönnek.
