# Motorul clean-room (`companion-engine` și `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Alte documente: [INNOVATIONS.md](INNOVATIONS.md) · [QUICKSTART.md](QUICKSTART.md) · [CLEANROOM.md](CLEANROOM.md) · [CHANGES-SECURITY.md](CHANGES-SECURITY.md)

**Stare: alpha.** Tot ce urmează a fost testat pe puncte de control sintetice mici, pe fișiere de referință ale tokenizatorului și pe text de probă înregistrat. **Autorii nu au rulat prin acest motor niciun model real descărcat de pe Hugging Face**, nicio placă grafică nu a executat nimic din el, iar căile de cod pentru ARM, macOS și Windows nu au fost rulate aici. Citește „Verificat și neverificat” înainte să te bazezi pe o afirmație. `corbul capabilities` afișează aceleași limite chiar din executabil.

## Ce este

Un motor local de inferență pentru modele de limbaj, scris de la zero în Rust, pornind de la descrieri publice ale formatelor de fișiere și de la lucrări publicate.

| Crate | Ce este |
|---|---|
| `crates/engine` (`companion-engine`) | Biblioteca: formate de fișiere, tokenizator, cuantizare, treceri înainte, eșantionare și uneltele suplimentare din INNOVATIONS.md. Dependențe la rulare: doar `serde_json` și `memmap2`. |
| `crates/engine-cli` (`corbul`) | Linia de comandă și un server local compatibil OpenAI / Ollama / Anthropic, cu interfață web încorporată. Depinde de bibliotecă și de `serde_json`. |
| `crates/engine-gpu` (`companion-engine-gpu`) | Nuclee WGSL și o emulare a lor pe procesor. Rulatorul pe dispozitiv (`wgpu`, `pollster`) se află în spatele funcției cargo `wgpu`, dezactivată implicit. **`corbul` nu folosește acest crate.** |

Este separat de aplicația desktop Colibri Companion și nu o înlocuiește; aceasta continuă să controleze colibri al lui JustVugg. Folosirea acestui motor ca alternativă de back end a aplicației este planificată, nu făcută.

## Ce suportă

**Fișiere.** GGUF v2 și v3 (antetul și tabela de tensori se citesc fără a atinge ponderile); un scriitor GGUF în flux pentru ieșiri mai mari decât memoria RAM; safetensors, într-un singur fișier sau fragmentat cu index; `config.json`, `tokenizer.json` și șabloanele de conversație Hugging Face; un descărcător de modele (modulul `hub`, prin `curl`, cu `wget` ca alternativă anonimă; reluare, verificări SHA-256, descărcări fixate pe commit, plafoane de dimensiune, refuzul redirecționărilor către http simplu).

**Cuantizare.** Decodate: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K (14 tipuri). Există codificatoare pentru aceleași tipuri, cu ponderare prin matricea de importanță și rețete pe tensor. Ponderile rămân cuantizate în memorie; produsul matrice–vector nu extinde niciodată întreaga matrice. **Nesuportate:** i-quant-urile (IQ\*) și orice tip care nu apare în listă.

**Arhitecturi.**

| Familie | Starea în `corbul` |
|---|---|
| Decodoare dense: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 și similare (normalizare RMS sau pe strat, GQA/MQA, RoPE cu scalare liniară, Llama-3, YaRN și LongRoPE și rotație parțială, QK-norm, biasuri, soft-capping, ferestre glisante, MLP cu poartă sau simplu) | Încărcate și rulate (șablon de conversație, serve, toate opțiunile) |
| Amestec de experți: Mixtral, Qwen-MoE, experți stivuiți GGUF, experți partajați, experți rutați transmiși de pe disc cu o memorie cache LRU | Încărcate și rulate |
| Spațiu de stare: Mamba-1, Mamba-2, RWKV-6, RWKV-7 (modulul `ssm`) | Rulate de `generate` și `chat` ca **simplă continuare de text**: fără șablon de conversație, fără ieșire structurată, fără `serve`. RWKV în formă GGUF nu este suportat (doar safetensors) |
| Hibride: Jamba și Bamba (modulul `hybrid`) | La fel ca rândul spațiului de stare. Zamba și Zamba2 nu sunt suportate |
| Atenție latentă multi-cap (DeepSeek-V2/V3, Kimi-K2; modulul `mla`) | **Doar bibliotecă.** Blocul există și este testat pentru consistență internă, dar `corbul` nu asamblează un astfel de model (`corbul capabilities` spune că nu) |
| Encoder de viziune (ViT, CLIP, SigLIP; modulul `vision`, decodoare în `imageio`) și front-end de vorbire plus un rulator cu structură Whisper (modulul `audio`) | **Doar bibliotecă.** `corbul` acceptă doar text |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, modele encoder și encoder-decoder (T5, BERT…), tokenizatoare WordPiece | Nesuportate; `corbul` o spune, în loc să ghicească |
| Endpoint-uri de embeddings și reranking | Răspund 501 (neimplementate) |

**Tokenizator.** Modele BPE la nivel de octet și de tip SentencePiece, reguli de pre-tokenizare pentru 12 familii, tokeni adăugați, rezervă pe octeți, șabloane din `tokenizer_config.json` și din metadatele GGUF. WordPiece/BERT nu este suportat.

**Eșantionare.** Temperatură, top-k, top-p, min-p, typical-p, penalizări de repetiție / prezență / frecvență, Mirostat 1 și 2, DRY, XTC, o specificație JSON a eșantionatorului, aleatoriu cu sămânță; decodare constrânsă prin JSON, JSON Schema, expresie regulată sau gramatică GBNF (cu bugete de lucru, astfel încât o gramatică ostilă să nu poată bloca procesul); bias pe logiți și șiruri de oprire prin server. Decodare speculativă: model-schiță, căutare în prompt, auto-speculație (vezi INNOVATIONS.md).

**Server.** `corbul serve` oferă `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/messages` și `/v1/messages/count_tokens` (în stil Anthropic), endpoint-urile Ollama `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/create`, `/api/delete`, `/api/ps`, `/api/version`, administrarea modelelor `/models`, `/models/load`, `/models/unload`, plus `/metrics`, `/props`, `/slots`, `/tokenize`, `/detokenize`, `/apply-template`, `/health`, `/v1/capabilities`, `/v1/quotes/verify` și interfața web la `/` și `/ui` (opt limbi). Apelarea de unelte este suportată prin șabloanele de conversație. Rutele de embeddings și rerank (`/v1/embeddings`, `/v1/rerank`, `/api/embed`, `/api/embeddings`) răspund 501. Răspunsurile în flux nu conțin bloc de citări.

**Hardware.** Nucleele pe procesor se aleg la rulare: scalar, avx2+fma, avx2+fma+avxvnni, avx512, avx512+vnni (x86-64), neon și neon+dotprod (aarch64); `ENGINE_SIMD` forțează unul. Un grup persistent de fire, detectarea hardware-ului și a NUMA și un planificator (`corbul plan`, `corbul doctor`) hotărăsc ce rămâne în RAM. Nucleele GPU nu au rulat niciodată pe o placă grafică (vezi mai jos).

## Exactitate

- **Ponderi dense f32:** căile pe loturi și pe un singur token, mesh-ul, transmiterea pe straturi și cache-urile dau logiți identici bit cu bit.
- **Ponderi cuantizate:** identitatea bit cu bit între evaluarea pe loturi și cea pe un singur token se păstrează doar când `exact_batch` este activ. Verificarea, rundele de verificare ale auto-speculației și executorul pe loturi îl activează. Prefill-ul obișnuit folosește o cale mai rapidă, de decuantizare urmată de împărțire în plăci, și diferă de evaluarea pe un singur token doar prin zgomot de rotunjire.
- **Modul reproductibil** (`--reproducible` sau `ENGINE_REPRODUCIBLE=1`): rezultate identice între variantele de nuclee x86 și numerele de fire pe care le-am putut testa. ARM nu a fost testat. Costă viteză.
- Mirostat 1 folosește în estimare numărul de candidați rămași, nu dimensiunea vocabularului (abatere documentată).

## Verificat și neverificat

| Zonă | Stare |
|---|---|
| Teste unitare, de integrare, adversariale și de tip fuzz pe puncte de control sintetice (Linux x86-64). Numerele se schimbă cu fiecare commit; rulează `cargo test --workspace` pentru numărul curent | Trec la ultima rulare completă. Fișierele ostile și cele trunchiate returnează erori. |
| Tokenizator față de fișierele de referință llama.cpp, 12 familii | 46 din 46 de cazuri egale (fișiere de referință luate de pe GitHub; `crates/engine/tests/tokenizer_real.rs`, `ENGINE_VOCAB_DIR`). Doar tokenizatorul, nu modele întregi. |
| Un model real Hugging Face prin `corbul` (orice familie) | **Nerulat de autori.** Mediul de construire nu a putut ajunge la huggingface.co. Un job CI manual rulează prin CLI un mic GGUF llama (`stories260K`); nu a rulat încă |
| Permutarea rândurilor Q/K în stil llama.cpp, variante de metadate GGUF | Implementate după descrieri publice; neverificate pe fișiere reale |
| Cuantizatoare și decuantizatoare | Blocuri calculate de mână și parcursuri dus-întors; necomparate cu fișiere scrise de alte unelte |
| Execuție pe GPU | **Nerulată.** Nucleele sunt validate doar de `naga` și de o emulare în Rust; `corbul` nu le folosește |
| ARM / NEON, macOS, Windows | **Nerulate aici.** Sondele sunt testate pe text de probă înregistrat. Fluxul CI are o matrice pentru ele, dar nu a rulat niciodată |
| Mamba, RWKV, Jamba/Bamba, MLA, viziune, audio | Doar teste de consistență internă; nu s-a folosit niciun punct de control real |
| Mesh | Doar pe loopback; autentificat, necriptat; fără măsurători de viteză |
| Viteză | **Fără cifre de încredere.** Calculatorul de construire avea două nuclee partajate. Vitezele din autopilot sunt estimări |
| Licențele dependențelor | Verificate cu `cargo-deny` la commitul b496ba5; modificările ulterioare nu au fost reverificate |
| Traducerile documentelor | Asistate automat; încă fără revizuire de către vorbitori nativi |

## Variabile de mediu

| Variabilă | Efect |
|---|---|
| `COLIBRI_THREADS` | Dimensiunea grupului global de fire (implicit: nucleele utilizabile). Citită prima de bibliotecă |
| `ENGINE_SIMD` | Forțează o variantă de nucleu după nume (măsurători, rapoarte de erori) |
| `ENGINE_REPRODUCIBLE` | `1`, `true` sau `on`: modul reproductibil de la început |
| `ENGINE_BACKEND` | `cpu`, `gpu` sau `auto` (nesetat = auto). `corbul` rulează pe procesor în ambele cazuri |
| `ENGINE_GPU`, `ENGINE_GPU_API`, `ENGINE_GPU_SOFTWARE` | Alegerea adaptorului pentru crate-ul GPU: indice, API grafic, permite un adaptor software. Nefolosite de `corbul` |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Token de acces pentru depozite cu acces restricționat (`download`, `inspect owner/repo`) |
| `HF_ENDPOINT` | Oglindă a gazdei Hugging Face |
| `CORBUL_TOKEN` | Tokenul bearer partajat al `corbul serve` (la fel ca `--token`) |
| `CORBUL_MESH_KEY` | Cheia mesh pentru `worker` și `--workers`/`--discover` (la fel ca `--key`) |
| `CORBUL_CONFIG_DIR` | Unde se păstrează Modelfile-urile create de `POST /api/create` (la fel ca `--config-dir`; implicit `~/.config/corbul`) |
| `NUMBER_OF_PROCESSORS`, `PROCESSOR_IDENTIFIER` | Citite pe Windows de sonda hardware; nu le setezi tu |
| `ENGINE_VOCAB_DIR`, `ENGINE_VOCAB_ONLY`, `ENGINE_REQUIRE_VOCAB`, `ENGINE_REQUIRE_TOOLS`, `ENGINE_PRINT_GOLDEN`, `ADVERSARIAL_N` | Comenzi ale suitei de teste: folderul cu vocabulare, restrângerea la anumite familii, eșuarea unui test sărit, eșuarea când lipsesc `curl`/`wget`, regenerarea hash-urilor de referință, dimensiunea rulării adversariale |

## Declarație clean-room

Niciun fragment de cod din colibri, llama.cpp sau din alt motor de inferență nu a fost citit sau copiat la scrierea acestui motor. Formatele și algoritmii provin din specificații publice, lucrări și structuri de fișiere. Modelul care a scris codul a fost antrenat pe text public care poate include astfel de proiecte; vezi [CLEANROOM.md](CLEANROOM.md) pentru proces și limitele lui.

## Cum se compară cu alte motoare locale

Nu am făcut nicio comparație directă de performanță, deci aici nu există afirmații despre viteză sau calitate.

- Motoarele consacrate (llama.cpp, Ollama, mistral.rs, vLLM) au rulat mii de puncte de control reale, au back end-uri GPU funcționale, acoperă mai multe arhitecturi și formate de cuantizare, inclusiv i-quant-uri, și se descurcă cu modelele de viziune. Acest motor nu are încă nimic din aceste dovezi. Dacă ai nevoie de ceva care funcționează azi pe un anumit model mare, folosește unul dintre ele.
- Ce încearcă acest motor să facă altfel: unelte de integritate încorporate (citări blocate pe citat, mod reproductibil cu chitanțe, analiză a incertitudinii, un planificator care spune „asta nu va rula” în loc să se prăbușească), unelte lingvistice cu prioritate pentru română, modele mai mari decât RAM-ul sau decât calculatorul tău, un cod Rust mic, cu două dependențe de bibliotecă, și doar licențe permisive.

## Harta bibliotecii

Formate și încărcare: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`, `imageio`. Modele: `arch`, `model`, `moe`, `mla`, `ssm`, `hybrid`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Execuție: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Funcții distinctive: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Calcul la momentul testării, reglaj, regăsire: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark`. Multimodal: `vision`, `audio`. Care dintre ele sunt accesibile prin `corbul` se vede în INNOVATIONS.md; API-urile bibliotecii se pot încă schimba.
