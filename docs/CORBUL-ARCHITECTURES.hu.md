# Modellek, amelyeket a corbul futtat, a colibri nem

A Companion a **corbult**, a saját motorját indítja minden olyan GGUF-hoz, amelyet a colibri nem tud importálni. Az alpha.18 óta az alábbi családoknál **közvetlenül** oda megy, előzetes, sikertelen import nélkül.

## Mit futtat a corbul
- **Gemma 4** (E2B, E4B, 26B-A4B szakértőkkel, 31B), `.gguf` fájlból: vegyes csúszóablakos és globális figyelem, rétegenkénti fejméretek, globális rétegeken a kulcs mint érték, arányos RoPE, megosztott K/V az utolsó rétegekben, rétegenkénti beágyazások, a sűrű MLP az irányított szakértők mellett, a végső logitok korlátozása.
- **GPT-2, GPT-J, GPT-NeoX/Pythia, Falcon, BLOOM, MPT**, `.gguf` fájlból: LayerNorm, összevont vagy külön QKV, tanult / forgó / ALiBi pozíciók, párhuzamos reziduális. Csak kis szintetikus modelleken tesztelve. Az OPT-nak nincs GGUF-konverziója.
- **`corbul scan [MAPPA…]`** megkeresi egy meghajtó vagy mappa `.gguf` modelljeit, és fájlonként kiírja, mit mond a fejléc (architektúra, méret, kvantálás, kontextus, szakértők, csevegősablon, eszközhívások), futtatja-e a corbul, elfér-e e gép memóriájában és videókártyáján, valamint a dekódolás **becsült** sebességét token/s-ban CPU-n és GPU-n. A fájl adatai pontosak; a sebességek egy memória-sávszélességi modellből jönnek (a CPU-ét helyben méri, a kártyáét név szerinti táblázatból veszi), nem mérések. Ugyanezek az adatok a Modellek ablakban is megjelennek minden fájl alatt.
- **DBRX, Phi-3.5-MoE**, `.gguf` fájlból (saját modell, szakértők tokenenként olvasva). A Phi-3.5-MoE csak a rövid rope-tényezőket használja, a 4096 feletti kontextus még nincs skálázva. Csak szintetikus tesztek.
- **gpt-oss (20B, 120B)**, `.gguf` fájlból (MXFP4 szakértőkkel): figyelem-nyelők, csúszóablak, YaRN, korlátozott szakértői aktiváció. Csak kis szintetikus modelleken tesztelve. **GraniteMoE, GLM-4.5/4.6 MoE**: szintén `.gguf`-ból.
- **Gemma 1–3, Phi, OLMo, Granite, Cohere, InternLM2, StableLM, StarCoder2, EXAONE, Baichuan, MiniCPM, SmolLM3**, Llama és Mistral, Qwen és a motor korábbi szakértős családjai.
- **Mamba, Mamba-2, RWKV-6/7, Jamba, Bamba** (rekurrens és hibrid).

A `corbul serve MODELL` válaszol a `/health`, `/v1/models`, `/v1/chat/completions` és `/v1/completions` címekre (folyamként is), a modell saját csevegősablonjával, `stop`, `max_tokens`, `temperature`, `top_p`, `top_k` és `seed` mellett. Az előzőt folytató beszélgetés nem olvassa újra az előzményeit.

## Mit nem tud ezeknél a modelleknél
Eszközhívás, strukturált kimenet, több válasz kérésenként, beágyazások, kontextuscsomagok (ilyenkor a Companion a szokásos csevegést használja), képbemenet, több modell egyszerre. A Gemma 4-et csak `.gguf`-ból olvassa, safetensors mappából nem. A saját szókincs nélküli GGUF mellé tehető `tokenizer.json` használható.

## Mi nincs ellenőrizve
Minden **kis szintetikus modelleken** van tesztelve (saját invariánsok: a gyorsítótár egyenlő az újraszámolással, csúszóablakok, megosztott K/V, útválasztás, kvantált szakértők, a HTTP-válaszok). **Valódi Gemma 4 fájllal semmit sem hasonlítottunk össze.** Ha egy valódi fájl válaszai hibásnak tűnnek, futtasd magad a `corbul serve FÁJL.gguf` parancsot, és küldd el a kimenetét; először a csúszó/globális mintát és a RoPE-táblákat kell ellenőrizni.

## Még nem támogatott
OPT, Llama 4, Hunyuan, Ernie-MoE, Grok, a teljes DeepSeek modell, MiniMax, Zamba, kódoló–dekódoló modellek. A corbul néven nevezi őket, és nem találgat.
