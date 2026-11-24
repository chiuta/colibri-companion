# Procesul clean-room

[English](CLEANROOM.md) · [Română](CLEANROOM.ro.md) · [Français](CLEANROOM.fr.md) · [Italiano](CLEANROOM.it.md) · [Español](CLEANROOM.es.md) · [Português](CLEANROOM.pt.md) · [Deutsch](CLEANROOM.de.md) · [Magyar](CLEANROOM.hu.md)

O notă scurtă și onestă despre cum a fost scris motorul din `crates/engine` și ce înseamnă și ce nu înseamnă „clean-room” aici.

## Regulile pe care le-am urmat

1. **Surse.** Doar specificații și lucrări publice: structurile fișierelor GGUF și safetensors, câmpurile `config.json` și `tokenizer.json` de la Hugging Face, descrierile publicate ale structurii blocurilor de cuantizare și lucrările din spatele fiecărei tehnici (variante de atenție, rutarea MoE, Mamba, RWKV, decodarea speculativă, H2O, StreamingLLM, SnapKV, TIES, DARE, filigran „green list”, HNSW, BM25…).
2. **Fără să te uiți.** Codul sursă al lui colibri, llama.cpp și al altor motoare de inferență nu a fost citit sau copiat la scrierea motorului. Când o structură nu era clară, lucrul s-a oprit la descrierea publică, iar incertitudinea e scrisă în cod și în documentație.
3. **Oracole în loc de imitație.** Fiecare nucleu rapid e testat față de o versiune scalară simplă scrisă prima; fiecare cuantizator, față de blocuri calculate manual și dus-întors; căile împărțite (plasă, transmitere, cache-uri), față de calea pe o singură mașină, bit cu bit acolo unde se promite asta.
4. **Doar licențe permisive.** Motorul depinde de `serde_json` și `memmap2` și de nimic altceva; crate-ul GPU adaugă `wgpu` sub o opțiune. Licențele au fost verificate cu `cargo-deny` la o versiune anterioară; modificările ulterioare nu au fost reverificate încă.
5. **Intrarea ostilă e normală.** Fiecare mărime sau decalaj citit dintr-un fișier se verifică față de lungimea reală a fișierului înainte de orice alocare; testele de tip fuzz dau fișiere trunchiate și corupte și trebuie să primească o eroare, nu o cădere.

## Ce nu afirmăm

- Clean-room este un proces, nu o garanție juridică. Modelul care a scris codul, ca orice model, a fost antrenat pe text public care poate include cod open-source și nu putem exclude că a văzut proiectele de mai sus. Nu le-am consultat și am încercat să facem asta vizibil păstrând în cod oracolele, testele și sursa fiecărei formule.
- Mai multe formule (Mamba, RWKV-6/7, MLA, extrasele imatrix și GGUF) au fost scrise după descrieri publicate și din amintirea lor și au fost verificate doar pentru coerență internă, fiindcă nu a existat niciun punct de control real. Pot diferi de referință în detalii; tratează-le ca neverificate până când cineva le rulează pe modele reale.
- Compatibilitatea cu fișierele altor motoare este un obiectiv, nu un fapt verificat (vezi tabelul din [ENGINE.ro.md](ENGINE.ro.md)).

Dacă găsești un loc în care motorul reproduce prea fidel codul altcuiva, deschide te rog o problemă (issue); îl rescriem.
