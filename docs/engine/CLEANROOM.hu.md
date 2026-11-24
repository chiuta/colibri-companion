# Clean-room folyamat

[English](CLEANROOM.md) · [Română](CLEANROOM.ro.md) · [Français](CLEANROOM.fr.md) · [Italiano](CLEANROOM.it.md) · [Español](CLEANROOM.es.md) · [Português](CLEANROOM.pt.md) · [Deutsch](CLEANROOM.de.md) · [Magyar](CLEANROOM.hu.md)

Rövid, őszinte feljegyzés arról, hogyan íródott a `crates/engine` és a `crates/engine-cli` motorja, és mit jelent, illetve mit nem jelent itt a „clean-room”. Folyamatleírás, nem jogi vélemény.

## A szabályok, amelyeket követtünk

1. **Források.** Nyilvános specifikációk és tanulmányok: a GGUF és a safetensors fájlelrendezése, a Hugging Face `config.json` és `tokenizer.json` mezői, a blokk-kvantálási elrendezések közzétett leírásai, a kép- és hangdekóderek mögötti formátumszabványok (RFC 1950/1951, a PNG-specifikáció, ITU T.81, a BMP és a Netpbm leírása, RIFF/WAVE), valamint az eljárások mögötti tanulmányok (figyelemváltozatok, MoE-irányítás, Mamba, RWKV, spekulatív dekódolás, H2O, StreamingLLM, SnapKV, TIES, DARE, zöldlistás vízjelezés, HNSW, BM25, ViT, CLIP, SigLIP, Whisper…).
2. **Nincs másolás, nincs forráskód-nézegetés.** A colibri, a llama.cpp és más következtető motorok forráskódját a motor írása közben nem nyitottuk meg és nem másoltuk. Ha egy elrendezés nem volt világos, a munka a nyilvános leírásnál megállt, a bizonytalanságot pedig a kódba és a dokumentációba írtuk.
3. **Orákulumok az utánzás helyett.** Minden gyors kernelt egy előbb megírt egyszerű skaláris változattal szemben tesztelünk; minden kvantálót kézzel kiszámolt blokkokkal és körutakkal; a szétosztott utakat (háló, folyamatos betöltés, gyorsítótárak) az egygépes úttal szemben, bitre pontosan ott, ahol a dokumentáció ezt ígéri. A tokenizálónál **referenciaadat-fájlokkal** (szótárfájlok és várt tokensorozatok) is összevetjük, amelyeket a llama.cpp tárolója közöl, és sima adatként töltünk le: ezek tesztbemenetek és várt kimenetek, nem kód, az eredmény (46 eset 46-ból, 12 család) pedig semmit sem mond a teljes modellek viselkedéséről.
4. **Csak megengedő licencek.** A könyvtár futásidőben a `serde_json` és a `memmap2` crate-től függ. A parancssor a könyvtáron és a `serde_json` crate-en kívül semmit sem ad hozzá. A GPU-crate a `wgpu` és a `pollster` crate-et adja hozzá a `wgpu` képesség mögött (alapértelmezés szerint kikapcsolva), valamint a `naga` crate-et a tesztekhez. A licenceket a `cargo-deny` eszközzel ellenőriztük a b496ba5 commitnál; a későbbi változtatásokat még nem ellenőriztük újra, így a „csak megengedő” állítás arra a commitra ellenőrzött, a jelenlegi fára pedig várt, nem bizonyított.
5. **Az ellenséges bemenet a megszokott eset.** A fájlokból olvasott méreteket és eltolásokat azelőtt ellenőrizzük, hogy bármit lefoglalnánk, az éppen érvényes korláttal szemben: a valódi fájlhosszal, ahol a formátum megengedi (a GGUF-elemzés megkapja a fájl hosszát), egyébként dokumentált plafonnal (a képek `Limits` értékei, letöltési plafonok, a nyelvtan és az idézetzár munkakeretei, tokenizáló-darabplafonok, nyugta- és kézfogás-plafonok). A fuzz-szerű és adverzális tesztek csonka és sérült fájlokat táplálnak be, és hibát kell kapniuk, nem összeomlást. Ez csökkenti a kockázatot; nem bizonyítja, hogy semmilyen bemenet nem okozhat pánikot.

## Mit nem állít ez

- A clean-room folyamat, nem garancia. A kódot író modellt, mint bármelyiket, nyilvános szövegeken tanították, amelyek tartalmazhatnak nyílt forráskódot, és nem zárhatjuk ki, hogy látta a fent említett projekteket. Nem néztük meg őket, és a folyamatot láthatóvá igyekeztünk tenni azzal, hogy az orákulumokat, a teszteket és minden képlet forrását a kódban tartjuk.
- Több képletet (Mamba, RWKV-6/7, MLA, Jamba/Bamba, imatrix és GGUF-kiegészítők, látás- és hangblokkok) közzétett leírások és azok emlékezete alapján írtunk, és csak belső konzisztenciára ellenőriztünk, mert nem állt rendelkezésre valódi ellenőrzőpont. Részletekben eltérhetnek a referenciától; tekintsd őket ellenőrizetlennek, amíg valaki nem futtatja őket valódi modelleken.
- A más motorok fájljaival való kompatibilitás cél, nem ellenőrzött tény (lásd az „Ellenőrzött és nem ellenőrzött” részt az [ENGINE.md](ENGINE.hu.md) fájlban).
- A nyilatkozat a kódra vonatkozik, nem a modellekre, amelyeket futtatsz: a modelleknek saját licencük van.

Ha találsz olyan helyet, ahol a motor túl szorosan adja vissza valaki más kódját, kérlek, nyiss egy hibajegyet; újraírjuk.
