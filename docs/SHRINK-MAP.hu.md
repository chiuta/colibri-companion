# Összezsugorított csomagok és gyorsítótár-leképezések modellek között

A kontextuscsomagok két kiterjesztése ([CONTEXT-PACKS.hu.md](CONTEXT-PACKS.hu.md), csak a corbul motorhoz; clean-room, saját kód).

## Összezsugorított csomagok

Ha a szövegek és a kérdés együtt nem férnek a modell ablakába, a csomagokat elutasítja. Zsugorítással megrövidíti őket: `corbul packs --shrink auto` (csak amennyi kell) vagy `--shrink R` (mindig nagyjából az `R` hányadot tartja meg, minden csomag sorainak 0,1–1 részét). Ugyanezt teszi a `/v1/packs/complete` `"shrink": "auto"` (vagy szám) mezője; a válasz felsorolja, mi zsugorodott (`"shrunk"`).

Hogyan: egy csomag utolsó `--shrink-window` tokenjét (alapértelmezés 16) újraolvassa, ezek szavaznak a megtartandó sorokra (SnapKV); az első sorokat mindig megtartja (ott gyűlik a figyelem); a megmaradó értékeket újraillesztik ([KV-FIT.hu.md](KV-FIT.hu.md); a `--no-refit` kikapcsolja). A megtartott kulcsokat ezután az új helyük pozíciójába forgatja, így a csomag *pontosan a rövidített szövegének gyorsítótára* lesz: úgy illeszthető és javítható, mint bármely csomag, teljes javítással a válasz megegyezik a rövidített szöveg egyszerű előtöltésével. Az első `--shrink-skip` szöveg (alapértelmezés 1, általában az utasítások) egészben marad. A zsugorítás a csomagok összeállításakor történik; a polc az egész csomagokat őrzi.

A Colibri Companionban: *Dokumentumok → Kontextuscsomagok (corbul)* → *Túl hosszú szövegek zsugorítása*.

Korlátok: információ vész el, mennyi, az a modelltől és a szövegtől függ (mérd a `corbul kvfit` paranccsal vagy a *Measure cache shrinking…* ablakkal); a csúszóablakos rétegű modelleket, a Q8 és a gyorsítón lévő gyorsítótárakat elutasítja; ha a szövegek az összes zsugorítható szöveg elhagyásával sem férnek el, a kérés meghiúsul. Eddig csak apró, véletlen modelleken próbálták.

## Gyorsítótár-leképezések különböző modellek között

Az egyik modell írta csomag a másiknak nem használ. Ha a kettő tokenizálója azonos, a *ugyanazokhoz a tokenekhez* tartozó gyorsítótársoraik összekapcsolhatók: a `corbul map-fit SRC DST --text FÁJL --out TÉRKÉP` mindkét modellel beolvassa a szöveget, és rétegenként lineáris leképezést illeszt (ridge legkisebb négyzetek, átlaggal) SRC kulcs- és értéksoraiból DST soraiba. A kulcsokat pozíció nélkül hasonlítja össze (0-ra visszaforgatva, importáláskor újra elforgatva), így a forgatási táblák és a fejméretek eltérhetnek; a rétegeket relatív mélység szerint párosítja. Kiírja az illesztésből kihagyott sorokon mért hibát az átlag találgatásának hibája mellett, és figyelmeztet, ha a térkép nem jobb.

A `corbul map-check SRC DST --map TÉRKÉP --text FÁJL --question Q` a szöveget két, SRC által írt csomagra bontja, a térképen át DST-be viszi, és háromféleképpen válaszol: DST saját teljes előtöltése (az igazság), a leképezett gyorsítótár a szokásos javítással (`--tolerance`, alapértelmezés 0,02) és a leképezett gyorsítótár javítás nélkül. Kiírja az újraszámolt sorokat és azt, mennyire tér el az egyes válasz az igazságtól.

A leképezett sor csak tipp: az importált sorok közelítőként vannak megjelölve, a javítás annyit számol újra, amennyi kell (mindet 0 tűréssel, ami az egyszerű előtöltéssel egyenlő). Egy rossz térkép tehát időbe kerül, a helyességet sosem rontja. A könyvtári belépési pont: `ctxpack::compose_mapped`.

Korlátok: a tokenizálóknak azonos azonosítókat kell adniuk (ellenőrzi, különben semmit sem illeszt); a térkép egyetlen modellpárra érvényes (az ujjlenyomatokat ellenőrzi); mérete `kd × kd' × rétegek × 8` bájt; az illesztés a kalibrációs sorokat a memóriában tartja (`--rows`, alapértelmezés 1024). Egymáshoz nem kapcsolódó modellek gyorsítótárai között a lineáris térkép gyenge lehet; hogy a te párodnál megéri-e, csak méréssel derül ki. A szerverben és a Companionban még nincs benne, és valódi modellpáron semmit sem mértek.
