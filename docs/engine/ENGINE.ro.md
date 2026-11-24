# Motorul clean-room (`companion-engine` și `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Alte documente: [INNOVATIONS.ro.md](INNOVATIONS.ro.md) · [QUICKSTART.ro.md](QUICKSTART.ro.md) · [CLEANROOM.ro.md](CLEANROOM.ro.md)

**Stare: alfa. Tot ce urmează a fost testat doar pe puncte de control sintetice, mici.** Autorii nu au rulat prin acest motor niciun model real de pe Hugging Face (mediul de construire nu a putut ajunge la huggingface.co), nicio placă video nu a executat nimic din el, iar căile pentru ARM, macOS și Windows nu au fost niciodată compilate aici. Citește tabelul „verificat și neverificat” înainte să te bazezi pe vreo afirmație.

## Ce este

Un motor local de inferență pentru modele de limbaj, scris de la zero în Rust, din descrieri publice ale formatelor de fișiere și din lucrări publicate. Are trei părți:

| Crate | Ce este |
|---|---|
| `crates/engine` (`companion-engine`) | Biblioteca: formate de fișiere, tokenizator, cuantizare, trecerile înainte, eșantionare și uneltele suplimentare din INNOVATIONS.ro.md. Dependențe: doar `serde_json` și `memmap2`. |
| `crates/engine-cli` (`corbul`) | O linie de comandă și un server local compatibil OpenAI / Ollama / Anthropic, construite peste bibliotecă. |
| `crates/engine-gpu` (`companion-engine-gpu`) | Nuclee WGSL pentru crate-ul `wgpu`, sub o opțiune dezactivată implicit. Nu e folosit de `corbul`. |

Este separat de aplicația desktop Colibri Companion și nu o înlocuiește: aplicația încă pilotează colibri al lui JustVugg. Integrarea acestui motor în aplicație ca backend alternativ este planificată, nu făcută.

## Ce susține

**Fișiere.** GGUF v2/v3 (antetul și tabela de tensori se citesc fără a atinge greutățile, deci un fișier de 400 GB se inspectează în milisecunde); un scriitor GGUF în flux pentru modele mai mari decât RAM; safetensors, într-un fișier sau fragmentat cu index; `config.json` și `tokenizer.json` de la Hugging Face; un descărcător de modele (prin `curl`, cu reluare și verificare SHA-256).

**Cuantizare.** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 și K-quants Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K se pot decoda, iar pentru toate există codificatoare (cu ponderare prin matrice de importanță), conversie Hugging Face → GGUF și recuantizare GGUF. Greutățile rămân cuantizate în memorie; produsul matrice–vector nu construiește niciodată matricea. I-quants (IQ*) nu sunt implementate.

**Arhitecturi.**

| Familie | Stare |
|---|---|
| Decodoare dense: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 și similare (normă RMS sau de strat, GQA/MQA, RoPE cu scalare liniară / Llama-3 / YaRN și rotație parțială, QK-norm, bias-uri, soft-capping, ferestre glisante, MLP cu poartă sau simplu) | Implementate; încărcate de `corbul` |
| Amestec de experți: Mixtral, Qwen-MoE, experți stivuiți GGUF, experți partajați, experți rutați transmiși de pe disc cu cache LRU | Implementate; încărcate de `corbul` |
| Atenție latentă multi-cap (stil DeepSeek-V2/V3, Kimi-K2) | Doar modul de bibliotecă; registrul de arhitecturi îl marchează încă „planificat”, iar `corbul` nu îl încarcă |
| Mamba-1, Mamba-2, RWKV-6, RWKV-7 | Doar modul de bibliotecă; la fel |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, modele encoder și encoder-decoder (T5, BERT, Whisper…), LongRoPE | Nesusținute; `corbul` o spune, nu ghicește |
| Intrări vizuale și audio | Nesusținute (doar text) |

**Căi hardware.**

- CPU: nuclee alese la rulare (scalar, AVX2, AVX-512, VNNI pe x86-64; NEON pe aarch64), un grup persistent de fire potrivit pentru nuclee performante/eficiente amestecate, detectarea hardware-ului și a NUMA, și un planificator care decide ce rămâne în RAM.
- GPU: nuclee WGSL verificate cu `naga` și față de o emulare în Rust. Nu au rulat niciodată pe o placă video.
- Modele mai mari decât RAM: experți de pe disc (MoE), transmiterea straturilor (dens), o plasă LAN peste mai multe calculatoare.

## Verificat și neverificat

| Zonă | Stare |
|---|---|
| Teste unitare, de integrare și fuzz pe puncte de control sintetice (Linux, x86-64) | Verificat. Sute de teste trec; fișierele ostile sau trunchiate dau erori, nu căderi. |
| Puncte de control reale Hugging Face (orice familie) | **Nerulat.** Mediul nu a putut ajunge la huggingface.co. |
| Tokenizatorul pe fișiere `tokenizer.json` reale | Scris după descrierile publice ale formatului; există un banc de teste (`ENGINE_VOCAB_DIR`), dar nu l-am rulat pe fișiere reale. |
| Permutarea rândurilor Q/K în stil llama.cpp din fișierele GGUF | Implementată după descriere; neverificată pe fișiere reale. |
| Cuantizatoare și decuantizatoare | Verificate față de blocuri calculate manual și dus-întors; nu față de fișiere scrise de alte unelte. |
| Execuție pe GPU | **Nerulată.** Nucleele sunt validate doar cu `naga` și o emulare în Rust. |
| Căi ARM / NEON, macOS, Windows | **Niciodată compilate sau rulate aici.** Sondele sunt testate pe text înregistrat de pe acele sisteme. |
| Formulele Mamba / RWKV / MLA | Scrise după lucrări și din amintirea descrierilor de referință, testate doar pentru coerență internă. |
| Plasa prin rețea | Testată doar pe loopback. |
| Viteză | **Fără numere de încredere.** Mașina de construire avea două nuclee partajate; măsurătorile au fost zgomotoase. Toate vitezele din autopilot sunt estimări. |
| Modul reproductibil | Identic pe cinci variante de nuclee x86 și mai multe numere de fire; ARM netestat. |
| Licențele dependențelor | Verificate cu `cargo-deny` la o versiune anterioară (b496ba5); adăugirile ulterioare nu au fost reverificate încă. |

## Declarație clean-room

Niciun cod din colibri, llama.cpp sau alt motor de inferență nu a fost citit sau copiat la scrierea acestui motor. Formatele și algoritmii vin din specificații publice, lucrări și structuri de fișiere. Vezi [CLEANROOM.ro.md](CLEANROOM.ro.md) pentru proces și limitele lui.

## Cum se compară cu alte motoare locale

Nu am rulat nicio comparație directă, deci nu facem afirmații despre viteză sau calitate. Calitativ:

- Motoare consacrate precum llama.cpp, Ollama, mistral.rs sau vLLM au fost rulate pe mii de puncte de control reale de mulți oameni, au backend-uri GPU funcționale, acoperă mai multe arhitecturi și formate de cuantizare (inclusiv i-quants) și rulează modele vizuale. Acest motor nu are încă nicio astfel de dovadă. Dacă ai nevoie de ceva care merge azi pe un anumit model mare, folosește unul dintre ele.
- Ce încearcă acest motor să facă altfel: unelte de integritate și onestitate încorporate (citate blocate pe surse, mod reproductibil cu chitanțe sigilate cu lanț de hash-uri și HMAC opțional, analiză a incertitudinii, un autopilot care spune „nu va merge” în loc să cadă), unelte de limbă cu româna pe primul loc, rularea de modele mai mari decât RAM-ul sau decât un singur calculator, un singur cod Rust mic cu două dependențe și doar licențe permisive.
- Orice afirmație că acest motor e mai rapid sau mai precis decât altul ar fi netestată. Te rugăm să măsori pe hardware-ul tău și să ne spui.

## Harta bibliotecii

Formate și încărcare: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`. Model: `arch`, `model`, `moe`, `mla`, `ssm`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Execuție: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Funcții distinctive: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Cel mai nou val: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark` — implementate în bibliotecă și acoperite de teste pe modele sintetice, încă în finalizare, deocamdată inaccesibile din `corbul`, iar API-urile lor se pot schimba.
