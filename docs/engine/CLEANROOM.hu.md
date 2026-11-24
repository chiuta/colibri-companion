# Clean-room folyamat

[English](CLEANROOM.md) · [Română](CLEANROOM.ro.md) · [Français](CLEANROOM.fr.md) · [Italiano](CLEANROOM.it.md) · [Español](CLEANROOM.es.md) · [Português](CLEANROOM.pt.md) · [Deutsch](CLEANROOM.de.md) · [Magyar](CLEANROOM.hu.md)

Rövid, őszinte jegyzet arról, hogyan íródott a `crates/engine` motor, és mit jelent és mit nem jelent itt a „clean-room”.

## A követett szabályok

1. **Források.** Csak nyilvános specifikációk és tanulmányok: a GGUF és safetensors fájlok szerkezete, a Hugging Face `config.json` és `tokenizer.json` mezői, a kvantálási blokkok szerkezetének publikált leírásai, és az egyes technikák mögötti tanulmányok (figyelemvariánsok, MoE-irányítás, Mamba, RWKV, spekulatív dekódolás, H2O, StreamingLLM, SnapKV, TIES, DARE, zöldlistás vízjel, HNSW, BM25…).
2. **Nem lesünk.** A colibri, a llama.cpp és más következtetési motorok forráskódját a motor írása közben nem olvastuk és nem másoltuk. Ahol egy szerkezet nem volt világos, a munka a nyilvános leírásnál megállt, a bizonytalanság pedig le van írva a kódban és a dokumentációban.
3. **Orákulumok utánzás helyett.** Minden gyors magot egy előbb megírt egyszerű skaláris változat ellen tesztelünk; minden kvantálót kézzel számolt blokkok és oda-vissza körök ellen; az osztott útvonalakat (háló, streamelés, gyorsítótárak) az egygépes útvonal ellen, bitre pontosan, ahol ezt ígérjük.
4. **Csak megengedő licencek.** A motor a `serde_json`-tól és a `memmap2`-től függ, mástól nem; a GPU-crate kapcsoló mögött adja hozzá a `wgpu`-t. A licenceket `cargo-deny`-vel egy korábbi commitnál ellenőriztük; a későbbi változtatásokat még nem ellenőriztük újra.
5. **Az ellenséges bemenet a megszokott.** Minden fájlból olvasott méretet vagy eltolást a fájl valódi hosszához mérünk, mielőtt bármit lefoglalnánk; a fuzz-jellegű tesztek csonkolt és sérült fájlokat adnak, és hibát kell kapniuk, nem összeomlást.

## Amit nem állítunk

- A clean-room folyamat, nem jogi garancia. A kódot író modellt, mint minden modellt, nyilvános szövegen tanították, amely nyílt forráskódot is tartalmazhat, és nem zárhatjuk ki, hogy látta a fent említett projekteket. Nem néztük meg őket, és igyekeztünk ezt láthatóvá tenni azzal, hogy a kódban megtartottuk az orákulumokat, a teszteket és az egyes képletek forrását.
- Több képletet (Mamba, RWKV-6/7, MLA, imatrix- és GGUF-kiegészítők) publikált leírások és azok emlékezete alapján írtunk, és csak belső következetességre ellenőriztük, mert nem állt rendelkezésre valódi ellenőrzőpont. Részletekben eltérhetnek a referenciától; tekintsd őket ellenőrizetlennek, amíg valaki nem futtatja őket valódi modelleken.
- A más motorok fájljaival való kompatibilitás cél, nem ellenőrzött tény (lásd a táblázatot itt: [ENGINE.hu.md](ENGINE.hu.md)).

Ha találsz olyan helyet, ahol a motor túl szorosan követi valaki más kódját, nyiss egy issue-t; újraírjuk.
