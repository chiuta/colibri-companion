# A clean-room motor (`companion-engine` és `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

További dokumentumok: [INNOVATIONS.hu.md](INNOVATIONS.hu.md) · [QUICKSTART.hu.md](QUICKSTART.hu.md) · [CLEANROOM.hu.md](CLEANROOM.hu.md)

**Állapot: alfa. Minden, ami következik, csak kis szintetikus ellenőrzőpontokon lett tesztelve.** A szerzők egyetlen valódi Hugging Face-modellt sem futtattak át ezen a motoron (a fordítási környezet nem érte el a huggingface.co-t), egyetlen GPU sem futtatott belőle semmit, és az ARM-, macOS- és Windows-útvonalakat itt soha nem fordították le. Olvasd el az „ellenőrzött és nem ellenőrzött” táblázatot, mielőtt bármilyen állításban megbíznál.

## Mi ez

Helyi következtetési (inferencia-) motor nyelvi modellekhez, a nulláról írva Rustban, nyilvános fájlformátum-leírásokból és publikált tanulmányokból. Három részből áll:

| Crate | Mi ez |
|---|---|
| `crates/engine` (`companion-engine`) | A könyvtár: fájlformátumok, tokenizáló, kvantálás, előre irányú számítás, mintavételezés és az INNOVATIONS.hu.md-ben leírt kiegészítő eszközök. Függőségek: csak `serde_json` és `memmap2`. |
| `crates/engine-cli` (`corbul`) | Parancssor és helyi, OpenAI / Ollama / Anthropic-kompatibilis szerver a könyvtár fölött. |
| `crates/engine-gpu` (`companion-engine-gpu`) | WGSL-magok a `wgpu` crate-hez, alapból kikapcsolt kapcsoló mögött. A `corbul` nem használja. |

Különálló a Colibri Companion asztali alkalmazástól, és nem váltja le: az alkalmazás továbbra is JustVugg colibri-jét vezérli. Ennek a motornak alternatív háttérként való beépítése az alkalmazásba tervezett, nem kész.

## Mit támogat

**Fájlok.** GGUF v2/v3 (a fejléc és a tenzortábla a súlyok érintése nélkül olvasható, így egy 400 GB-os fájl ezredmásodpercek alatt megvizsgálható); streamelő GGUF-író a RAM-nál nagyobb modellekhez; safetensors, egy fájlban vagy indexszel szétosztva; Hugging Face `config.json` és `tokenizer.json`; modelletöltő (a `curl`-lel, folytatással és SHA-256-ellenőrzéssel).

**Kvantálás.** Az F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 és a K-kvantok Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K dekódolhatók, mindegyikhez van kódoló (fontossági mátrixos súlyozással), Hugging Face → GGUF átalakítás és GGUF újrakvantálás. A súlyok a memóriában kvantáltak maradnak; a mátrix–vektor szorzat sosem építi fel a mátrixot. Az i-kvantok (IQ*) nincsenek megvalósítva.

**Architektúrák.**

| Család | Állapot |
|---|---|
| Sűrű dekóderek: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 és hasonlók (RMS- vagy rétegnorma, GQA/MQA, RoPE lineáris / Llama-3 / YaRN skálázással és részleges forgatással, QK-norma, torzítások, soft-capping, csúszó ablakok, kapuzott vagy egyszerű MLP) | Megvalósítva; a `corbul` betölti |
| Szakértők keveréke: Mixtral, Qwen-MoE, GGUF egymásra rakott szakértők, megosztott szakértők, lemezről streamelt irányított szakértők LRU-gyorsítótárral | Megvalósítva; a `corbul` betölti |
| Többfejű látens figyelem (DeepSeek-V2/V3, Kimi-K2 stílus) | Csak könyvtári modul; az architektúra-nyilvántartás még „tervezett”-nek jelöli, a `corbul` nem tölti be |
| Mamba-1, Mamba-2, RWKV-6, RWKV-7 | Csak könyvtári modul; ugyanúgy |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, enkóder és enkóder–dekóder modellek (T5, BERT, Whisper…), LongRoPE | Nem támogatott; a `corbul` ezt kimondja, nem találgat |
| Kép- és hangbemenet | Nem támogatott (csak szöveg) |

**Hardverútvonalak.**

- CPU: futásidőben kiválasztott magok (skaláris, AVX2, AVX-512, VNNI x86-64-en; NEON aarch64-en), tartós szálkészlet, amely vegyes teljesítmény-/hatékonysági magokhoz illik, hardver- és NUMA-észlelés, és egy tervező, amely eldönti, mi marad a RAM-ban.
- GPU: WGSL-magok, `naga`-val és Rust-emulációval ellenőrizve. Soha nem futottak GPU-n.
- A RAM-nál nagyobb modellek: szakértők lemezről (MoE), rétegek streamelése (sűrű), LAN-háló több gépen.

## Ellenőrzött és nem ellenőrzött

| Terület | Állapot |
|---|---|
| Egység-, integrációs és fuzz-tesztek szintetikus ellenőrzőpontokon (Linux, x86-64) | Ellenőrzött. Több száz teszt sikeres; az ellenséges vagy csonkolt fájlok hibát adnak, nem összeomlást. |
| Valódi Hugging Face-ellenőrzőpontok (bármely család) | **Nem futtatva.** A környezet nem érte el a huggingface.co-t. |
| Tokenizáló valódi `tokenizer.json` fájlokkal | A nyilvános formátumleírások alapján írva; van tesztkörnyezet (`ENGINE_VOCAB_DIR`), de valódi fájlokon mi nem futtattuk. |
| llama.cpp-stílusú Q/K sorpermutáció GGUF-fájlokban | A leírás alapján megvalósítva; valódi fájlokon nem ellenőrizve. |
| Kvantálók és dekvantálók | Kézzel számolt blokkok és oda-vissza körök ellen ellenőrizve; más eszközök által írt fájlok ellen nem. |
| GPU-végrehajtás | **Nem futtatva.** A magok csak `naga`-val és Rust-emulációval validáltak. |
| ARM / NEON, macOS, Windows útvonalak | **Itt soha nem fordítva és nem futtatva.** A szondákat az e rendszerekről rögzített szövegen tesztelik. |
| Mamba / RWKV / MLA képletek | Tanulmányokból és a hivatkozási leírások emlékezetéből írva, csak belső következetességre tesztelve. |
| Háló valódi hálózaton | Csak loopbacken tesztelve. |
| Sebesség | **Nincsenek megbízható számok.** A fordítógép két megosztott maggal rendelkezett; a mérések zajosak voltak. Az autopilot minden sebessége becslés. |
| Reprodukálható mód | Azonos öt x86 magvariánson és több szálszámon; ARM nem tesztelve. |
| Függőségek licencei | `cargo-deny`-vel ellenőrizve egy korábbi commitnál (b496ba5); a későbbi kiegészítéseket még nem ellenőriztük újra. |

## Clean-room nyilatkozat

A motor írása során nem olvastunk és nem másoltunk kódot a colibriből, a llama.cpp-ből vagy más következtetési motorból. A formátumok és algoritmusok nyilvános specifikációkból, tanulmányokból és fájlszerkezetekből származnak. A folyamatról és korlátairól lásd: [CLEANROOM.hu.md](CLEANROOM.hu.md).

## Összevetés más helyi motorokkal

Nem futtattunk közvetlen összehasonlítást, ezért sebességre vagy minőségre vonatkozó állítást nem teszünk. Minőségileg:

- A bevett motorokat, mint a llama.cpp, az Ollama, a mistral.rs vagy a vLLM, sokan futtatták ezernyi valódi ellenőrzőponton, működő GPU-hátterük van, több architektúrát és kvantálási formátumot fednek le (az i-kvantokat is), és kezelik a látásmodelleket. Ennek a motornak ebből még semmilyen bizonyítéka nincs. Ha olyasmire van szükséged, ami ma működik egy konkrét nagy modellel, használd valamelyiküket.
- Amit ez a motor másként próbál: beépített integritási és őszinteségi eszközök (forrásokhoz kötött idézetek, reprodukálható mód hash-lánccal és opcionális HMAC-kal lezárt nyugtákkal, bizonytalanság-elemzés, egy autopilot, amely azt mondja: „ez nem fog futni”, ahelyett hogy összeomlana), román-első nyelvi eszközök, a RAM-odnál vagy egyetlen gépnél nagyobb modellek futtatása, egy kis Rust-kódbázis két függőséggel és kizárólag megengedő licencekkel.
- Bármely állítás, hogy ez a motor gyorsabb vagy pontosabb másnál, tesztelés nélküli lenne. Mérj a saját hardveredden, és szólj nekünk.

## A könyvtár térképe

Formátumok és betöltés: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`. Modell: `arch`, `model`, `moe`, `mla`, `ssm`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Futtatás: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Megkülönböztető funkciók: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Legújabb hullám: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark` — a könyvtárban megvalósítva és szintetikus modelleken tesztekkel lefedve, még csiszolás alatt, a `corbul`ból még nem érhető el, és az API-jaik változhatnak.
