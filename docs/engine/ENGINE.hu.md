# A clean-room motor (`companion-engine` és `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

További dokumentumok: [INNOVATIONS.md](INNOVATIONS.hu.md) · [QUICKSTART.md](QUICKSTART.hu.md) · [CLEANROOM.md](CLEANROOM.hu.md) · [CHANGES-SECURITY.md](CHANGES-SECURITY.hu.md)

**Állapot: alfa.** Az alábbiak mindegyikét kis szintetikus ellenőrzőpontokon, tokenizáló-referenciafájlokon és rögzített mintaszövegeken teszteltük. **A szerzők nem futtattak át ezen a motoron egyetlen valódi, Hugging Face-ről letöltött modellt sem**, GPU-n semmi sem futott, az ARM, a macOS és a Windows kódútjait pedig itt nem futtattuk. Mielőtt bármelyik állításnak hinnél, olvasd el az „Ellenőrzött és nem ellenőrzött” részt. A `corbul capabilities` ugyanezeket a korlátokat írja ki magából a futtatható fájlból.

## Mi ez

Nyelvi modellek helyi következtető motorja, nulláról, Rustban írva, nyilvános fájlformátum-leírások és közzétett tanulmányok alapján.

| Crate | Micsoda |
|---|---|
| `crates/engine` (`companion-engine`) | A könyvtár: fájlformátumok, tokenizáló, kvantálás, előre menő számítások, mintavételezés és az INNOVATIONS.md kiegészítő eszközei. Futásidejű függőségei: csak a `serde_json` és a `memmap2`. |
| `crates/engine-cli` (`corbul`) | A parancssor és egy OpenAI-, Ollama- és Anthropic-kompatibilis helyi szerver beépített webes felülettel. A könyvtártól és a `serde_json` crate-től függ. |
| `crates/engine-gpu` (`companion-engine-gpu`) | WGSL-kernelek és CPU-s emulációjuk. Az eszközfuttató (`wgpu`, `pollster`) a `wgpu` cargo-képesség mögött van, amely alapértelmezés szerint ki van kapcsolva. **A `corbul` nem használja ezt a crate-et.** |

Különálló a Colibri Companion asztali alkalmazástól, és nem váltja le azt; az alkalmazás továbbra is JustVugg colibrijét vezérli. Hogy ez a motor az alkalmazás alternatív háttérrendszere legyen, tervben van, de nincs kész.

## Mit támogat

**Fájlok.** GGUF v2 és v3 (a fejlécet és a tenzortáblát a súlyok érintése nélkül olvassa); folyamatos GGUF-író a memóriánál nagyobb kimenetekhez; safetensors, egyben vagy indexszel szétosztva; Hugging Face `config.json`, `tokenizer.json` és csevegősablonok; modelletöltő (a `hub` modul, `curl` segítségével, névtelen `wget` tartalékkal mint tartalékkal; folytatás, SHA-256-ellenőrzés, commithoz kötött letöltés, méretplafonok, a sima http-re mutató átirányítások elutasítása).

**Kvantálás.** Dekódolható: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K (14 típus). Ugyanezekhez a típusokhoz vannak kódolók is, fontossági mátrixos súlyozással és tenzoronkénti receptekkel. A súlyok a memóriában kvantáltak maradnak; a mátrix–vektor szorzat sosem bontja ki az egész mátrixot. **Nem támogatott:** az i-kvantumok (IQ\*) és minden olyan típus, amely nincs a listán.

**Architektúrák.**

| Család | Állapot a `corbul` programban |
|---|---|
| Sűrű dekóderek: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 és hasonlók (RMS- vagy réteg-normalizálás, GQA/MQA, RoPE lineáris, Llama-3, YaRN és LongRoPE skálázással és részleges forgatással, QK-norma, torzítások, soft-capping, csúszó ablakok, kapuzott vagy egyszerű MLP) | Betöltve és futtatva (csevegősablon, serve, minden opció) |
| Szakértők keveréke: Mixtral, Qwen-MoE, GGUF egymásra rakott szakértők, megosztott szakértők, lemezről LRU-gyorsítótárral folyamatosan olvasott irányított szakértők | Betöltve és futtatva |
| Állapottér: Mamba-1, Mamba-2, RWKV-6, RWKV-7 (`ssm` modul) | A `generate` és a `chat` **sima szövegfolytatásként** futtatja: nincs csevegősablon, nincs strukturált kimenet, nincs `serve`. A GGUF formátumú RWKV nem támogatott (csak safetensors) |
| Hibrid: Jamba és Bamba (`hybrid` modul) | Ugyanaz, mint az állapottér sora. A Zamba és a Zamba2 nem támogatott |
| Többfejes látens figyelem (DeepSeek-V2/V3, Kimi-K2; `mla` modul) | **Csak könyvtár.** Az építőelem létezik, és belső konzisztenciára tesztelt, de a `corbul` nem állít össze ilyen modellt (a `corbul capabilities` szerint nem) |
| Látásenkóder (ViT, CLIP, SigLIP; `vision` modul, dekóderek az `imageio` modulban) és beszéd-előfeldolgozó plusz Whisper-elrendezésű futtató (`audio` modul) | **Csak könyvtár.** A `corbul` csak szöveget fogad |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, enkóder és enkóder-dekóder modellek (T5, BERT…), WordPiece tokenizálók | Nem támogatott; a `corbul` kimondja, ahelyett hogy találgatna |
| Beágyazás- és újrarangsoroló végpontok | 501-gyel válaszolnak (nincs megvalósítva) |

**Tokenizáló.** Bájtszintű BPE és SentencePiece-stílusú modellek, előtokenizáló szabályok 12 családhoz, hozzáadott tokenek, bájt-tartalék, sablonok a `tokenizer_config.json` fájlból és a GGUF-metaadatokból. A WordPiece/BERT nem támogatott.

**Mintavételezés.** Hőmérséklet, top-k, top-p, min-p, typical-p, ismétlési, jelenléti és gyakorisági büntetések, Mirostat 1 és 2, DRY, XTC, JSON mintavételező-specifikáció, magozott véletlen; korlátozott dekódolás JSON-nal, JSON Schemával, reguláris kifejezéssel vagy GBNF-nyelvtannal (munkakeretekkel, hogy egy ellenséges nyelvtan ne akaszthassa meg a folyamatot); logit-torzítás és leállító sztringek a szerveren át. Spekulatív dekódolás: vázlatmodell, prompt-keresés, önspekuláció (lásd INNOVATIONS.md).

**Szerver.** A `corbul serve` kínálja a `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/messages` és `/v1/messages/count_tokens` (Anthropic-stílus) végpontokat, az Ollama `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/create`, `/api/delete`, `/api/ps`, `/api/version` végpontjait, a modellkezelő `/models`, `/models/load`, `/models/unload` végpontokat, továbbá a `/metrics`, `/props`, `/slots`, `/tokenize`, `/detokenize`, `/apply-template`, `/health`, `/v1/capabilities`, `/v1/quotes/verify` végpontokat és a webes felületet a `/` és a `/ui` címen (nyolc nyelven). Az eszközhívás a csevegősablonokon keresztül támogatott. A beágyazás- és újrarangsoroló útvonalak (`/v1/embeddings`, `/v1/rerank`, `/api/embed`, `/api/embeddings`) 501-gyel válaszolnak. A folyamatos válaszok nem tartalmaznak idézetblokkot.

**Hardver.** A CPU-kerneleket futásidőben választja ki: scalar, avx2+fma, avx2+fma+avxvnni, avx512, avx512+vnni (x86-64), neon és neon+dotprod (aarch64); az `ENGINE_SIMD` valamelyiket kikényszeríti. Állandó szálkészlet, hardver- és NUMA-felismerés, valamint egy tervező (`corbul plan`, `corbul doctor`) dönti el, mi marad a memóriában. A GPU-kernelek még sosem futottak GPU-n (lásd lent).

## Pontosság

- **Sűrű f32 súlyok:** a kötegelt és az egyetlen tokenes utak, a háló, a rétegek folyamatos betöltése és a gyorsítótárak bitazonos logitokat adnak.
- **Kvantált súlyok:** a kötegelt és az egyetlen tokenes kiértékelés közti bitazonosság csak akkor áll fenn, ha az `exact_batch` be van kapcsolva. Az ellenőrzés, az önspekulatív ellenőrzési körök és a kötegelő végrehajtó bekapcsolja. A szokásos előtöltés gyorsabb, dekvantálás-majd-csempézés utat használ, és az egyetlen tokenes kiértékeléstől csak kerekítési zajban tér el.
- **Reprodukálható mód** (`--reproducible` vagy `ENGINE_REPRODUCIBLE=1`): azonos eredmények az általunk tesztelhető x86-kernelváltozatokon és szálszámokon. Az ARM-ot nem teszteltük. Sebességbe kerül.
- A Mirostat 1 a becslésében a fennmaradó jelöltek számát használja, nem a szótár méretét (dokumentált eltérés).

## Ellenőrzött és nem ellenőrzött

| Terület | Állapot |
|---|---|
| Egység-, integrációs, adverzális és fuzz-tesztek szintetikus ellenőrzőpontokon (Linux x86-64). A számok minden commitnál változnak; az aktuális számhoz futtasd a `cargo test --workspace` parancsot | Az utolsó teljes futásnál sikeres. Az ellenséges és csonka fájlok hibát adnak. |
| Tokenizáló a llama.cpp referenciafájljaival szemben, 12 család | 46-ból 46 eset egyezik (a referenciafájlokat a GitHubról töltöttük le; `crates/engine/tests/tokenizer_real.rs`, `ENGINE_VOCAB_DIR`). Csak a tokenizáló, nem teljes modellek. |
| Valódi Hugging Face-modell a `corbul` programmal (bármely család) | **A szerzők nem futtatták.** Az építési homokozó nem érte el a huggingface.co-t. Egy kézi CI-feladat egy apró llama GGUF-et (`stories260K`) futtat a parancssoron át; még nem futott |
| llama.cpp-stílusú Q/K sorpermutáció, GGUF-metaadat-változatok | Nyilvános leírások alapján megvalósítva; valódi fájlokon nem ellenőrizve |
| Kvantálók és dekvantálók | Kézzel kiszámolt blokkok és körutak; más eszközök által írt fájlokkal nem hasonlítottuk össze |
| GPU-futtatás | **Nem futott.** A kerneleket csak a `naga` és egy Rust-emuláció validálja; a `corbul` nem használja őket |
| ARM / NEON, macOS, Windows | **Itt nem futtatva.** A szondákat rögzített mintaszövegen teszteltük. A CI-munkafolyamatban van rájuk mátrix, de sosem futott |
| Mamba, RWKV, Jamba/Bamba, MLA, látás, hang | Csak belső konzisztenciát vizsgáló tesztek; valódi ellenőrzőpontot nem használtunk |
| Háló (mesh) | Csak loopback; hitelesített, nem titkosított; sebességmérés nincs |
| Sebesség | **Nincsenek megbízható számok.** Az építőgépnek két megosztott magja volt. Az autopilot sebességei becslések |
| Függőségek licencei | A `cargo-deny` eszközzel ellenőrizve a b496ba5 commitnál; a későbbi változtatásokat nem ellenőriztük újra |
| A dokumentumok fordításai | Gépi támogatással készültek; anyanyelvi átnézés még nincs |

## Környezeti változók

| Változó | Hatás |
|---|---|
| `COLIBRI_THREADS` | A globális szálkészlet mérete (alapértelmezés: a használható magok). A könyvtár olvassa elsőként |
| `ENGINE_SIMD` | Névvel kikényszerít egy kernelváltozatot (teljesítménymérések, hibajelentések) |
| `ENGINE_REPRODUCIBLE` | `1`, `true` vagy `on`: a reprodukálható mód az elejétől |
| `ENGINE_BACKEND` | `cpu`, `gpu` vagy `auto` (beállítatlan = auto). A `corbul` mindkét esetben a CPU-n számol |
| `ENGINE_GPU`, `ENGINE_GPU_API`, `ENGINE_GPU_SOFTWARE` | Adapterválasztás a GPU-crate-hez: index, grafikus API, szoftveres adapter engedélyezése. A `corbul` nem használja |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Hozzáférési token a védett tárolókhoz (`download`, `inspect owner/repo`) |
| `HF_ENDPOINT` | A Hugging Face gazdagép tükre |
| `CORBUL_TOKEN` | A `corbul serve` közös bearer tokenje (ugyanaz, mint a `--token`) |
| `CORBUL_MESH_KEY` | Hálókulcs a `worker` parancshoz és a `--workers`/`--discover` kapcsolókhoz (ugyanaz, mint a `--key`) |
| `CORBUL_CONFIG_DIR` | Hol tárolódnak a `POST /api/create` által létrehozott Modelfile-ok (ugyanaz, mint a `--config-dir`; alapértelmezés `~/.config/corbul`) |
| `NUMBER_OF_PROCESSORS`, `PROCESSOR_IDENTIFIER` | Windowson a hardverszonda olvassa; nem te állítod be |
| `ENGINE_VOCAB_DIR`, `ENGINE_VOCAB_ONLY`, `ENGINE_REQUIRE_VOCAB`, `ENGINE_REQUIRE_TOOLS`, `ENGINE_PRINT_GOLDEN`, `ADVERSARIAL_N` | A tesztcsomag vezérlői: szótármappa, családok szűkítése, kihagyott teszt megbuktatása, hiányzó `curl`/`wget` megbuktatása, aranyhashek újragenerálása, az adverzális futás mérete |

## Clean-room nyilatkozat

A motor írása közben nem olvastunk és nem másoltunk kódot a colibriból, a llama.cpp-ből vagy bármely más következtető motorból. A formátumok és az algoritmusok nyilvános specifikációkból, tanulmányokból és fájlelrendezésekből származnak. A kódot író modellt nyilvános szövegeken tanították, amelyek tartalmazhatnak ilyen projekteket; a folyamatot és korlátait a [CLEANROOM.md](CLEANROOM.hu.md) írja le.

## Összehasonlítás más helyi motorokkal

Nem futtattunk közvetlen összehasonlító tesztet, ezért itt nincsenek sebességre vagy minőségre vonatkozó állítások.

- A bevált motorokat (llama.cpp, Ollama, mistral.rs, vLLM) több ezer valódi ellenőrzőponton futtatták, működő GPU-háttérrendszerük van, több architektúrát és kvantálási formátumot fednek le, az i-kvantumokat is, és kezelik a látásmodelleket. Ennek a motornak egyelőre egyik sincs bizonyítva. Ha olyasmire van szükséged, ami ma működik egy adott nagy modellel, használj egyet közülük.
- Amit ez a motor másképp próbál csinálni: beépített integritáseszközök (idézetzáras hivatkozások, reprodukálható mód nyugtákkal, bizonytalansági elemzések, egy tervező, amely összeomlás helyett azt mondja, hogy „ez nem fog futni”), román nyelvet előtérbe helyező nyelvi eszközök, a memóriádnál vagy a gépednél nagyobb modellek, egy kis Rust kódbázis két könyvtárfüggőséggel, és kizárólag megengedő licencek.

## Könyvtártérkép

Formátumok és betöltés: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`, `imageio`. Modellek: `arch`, `model`, `moe`, `mla`, `ssm`, `hybrid`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Futásidő: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Megkülönböztető képességek: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Futásidejű számítás, hangolás, visszakeresés: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark`. Multimodális: `vision`, `audio`. Hogy ezek közül melyiket éri el a `corbul`, az INNOVATIONS.md-ben van felsorolva; a könyvtári API-k még változhatnak.
