# Procesul clean-room

[English](CLEANROOM.md) · [Română](CLEANROOM.ro.md) · [Français](CLEANROOM.fr.md) · [Italiano](CLEANROOM.it.md) · [Español](CLEANROOM.es.md) · [Português](CLEANROOM.pt.md) · [Deutsch](CLEANROOM.de.md) · [Magyar](CLEANROOM.hu.md)

O notă scurtă și sinceră despre cum a fost scris motorul din `crates/engine` și `crates/engine-cli` și despre ce înseamnă și ce nu înseamnă aici „clean-room”. Este o declarație de proces, nu o opinie juridică.

## Regulile pe care le-am urmat

1. **Surse.** Specificații și lucrări publice: structurile fișierelor GGUF și safetensors, câmpurile `config.json` și `tokenizer.json` din Hugging Face, descrieri publicate ale structurilor de cuantizare pe blocuri, standardele de format din spatele decodoarelor de imagine și sunet (RFC 1950/1951, specificația PNG, ITU T.81, descrierile BMP și Netpbm, RIFF/WAVE) și lucrările din spatele fiecărei tehnici (variante de atenție, rutarea MoE, Mamba, RWKV, decodare speculativă, H2O, StreamingLLM, SnapKV, TIES, DARE, filigran cu listă verde, HNSW, BM25, ViT, CLIP, SigLIP, Whisper…).
2. **Fără copiere, fără consultarea codului sursă.** Codul sursă al colibri, al llama.cpp și al altor motoare de inferență nu a fost deschis și nici copiat la scrierea motorului. Când o structură nu era clară, munca s-a oprit la descrierea publică, iar nesiguranța a fost scrisă în cod și în documente.
3. **Oracole în loc de imitație.** Fiecare nucleu rapid este testat față de o versiune scalară simplă, scrisă prima; fiecare cuantizator, față de blocuri calculate de mână și parcursuri dus-întors; căile împărțite (mesh, transmiterea pe straturi, cache-urile), față de calea pe o singură mașină, bit cu bit acolo unde documentele promit asta. Pentru tokenizator comparăm și cu **fișiere de date de referință** (fișiere de vocabular și secvențe de tokeni așteptate) publicate în depozitul llama.cpp și luate ca simple date: sunt intrări și ieșiri așteptate de test, nu cod, iar rezultatul (46 din 46 de cazuri, 12 familii) nu spune nimic despre comportamentul unor modele întregi.
4. **Doar licențe permisive.** Biblioteca depinde la rulare de `serde_json` și `memmap2`. Linia de comandă nu adaugă nimic în afară de bibliotecă și `serde_json`. Crate-ul GPU adaugă `wgpu` și `pollster` în spatele funcției `wgpu` (dezactivată implicit) și `naga` pentru teste. Licențele au fost verificate cu `cargo-deny` la commitul b496ba5; modificările ulterioare nu au fost încă reverificate, deci afirmația „doar permisive” este verificată pentru acel commit și așteptată, nu dovedită, pentru arborele curent.
5. **Intrarea ostilă este normală.** Dimensiunile și decalajele citite din fișiere sunt verificate înainte să se aloce ceva, față de limita aplicabilă: lungimea reală a fișierului acolo unde formatul permite (parsarea GGUF primește lungimea fișierului), altfel un plafon documentat (`Limits` pentru imagini, plafoanele de descărcare, bugetele de lucru ale gramaticii și ale blocării citatelor, plafoanele pentru piesele tokenizatorului, plafoanele pentru chitanțe și pentru strângerea de mână). Testele de tip fuzz și cele adversariale dau fișiere trunchiate și corupte și trebuie să primească o eroare, nu o prăbușire. Asta reduce riscul; nu este o dovadă că nicio intrare nu poate provoca un panic.

## Ce nu pretinde acest document

- Clean-room este un proces, nu o garanție. Modelul care a scris codul, ca orice model, a fost antrenat pe text public care poate include cod open-source și nu putem exclude faptul că a văzut proiectele de mai sus. Nu le-am consultat și am încercat să facem procesul vizibil păstrând în cod oracolele, testele și sursa fiecărei formule.
- Mai multe formule (Mamba, RWKV-6/7, MLA, Jamba/Bamba, imatrix și completările GGUF, blocurile de viziune și de sunet) au fost scrise după descrieri publicate și după amintirea lor și verificate doar pentru consistență internă, fiindcă nu exista niciun punct de control real. Pot diferi de referință în detalii; consideră-le neverificate până când cineva le rulează pe modele reale.
- Compatibilitatea cu fișierele altor motoare este un obiectiv, nu un fapt verificat (vezi „Verificat și neverificat” în [ENGINE.md](ENGINE.md)).
- Declarația acoperă codul, nu modelele pe care le rulezi: modelele au propriile licențe.

Dacă găsești un loc unde motorul reproduce prea fidel codul altcuiva, te rugăm să deschizi o problemă (issue); îl vom rescrie.
