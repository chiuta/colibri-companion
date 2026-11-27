# A gyorsítótár zsugorítása és az értékek újraillesztése

A `corbul kvfit MODEL --text FILE --question Q` azt méri, mibe kerül egy szöveg gyorsítótár-soraiból csak egy részt megtartani (a SnapKV választja ki), a megmaradó **értékek újraillesztésével** és anélkül.

A szöveg utolsó `--window` tokenje próbalekérdezésként kerül rögzítésre. A megtartott sorok értékeit a lehető legkevésbé mozgatja el úgy, hogy a rájuk irányuló figyelem a próbákon azt adja, amit az összes sorra irányuló figyelem adott (regularizált legkisebb négyzetek, duális alak, `crates/engine/src/kvfit.rs`). Az illesztés *a próbákon* sosem lehet rosszabb az egyszerű értékeknél; hogy egy meg nem próbált kérdésnél segít-e, a modelltől és a szövegtől függ, és ezt csak ez a parancs mondja meg a te modelleden.

Beállítások: `--budget 0.25` (megtartott sorok aránya), `--window 16`, `--ridge 0.01`, `--max-new 32`. Kiírja az illesztési hibát a próbákon, az első pontszámok legnagyobb eltérését a teljes gyorsítótárhoz képest, egyezik-e az első token, és a válasz első tokenjei közül hány egyezik.

Korlátok: csak az értékek (a kulcsok hordozzák a pozíciót és maradnak; az elhagyott sorok súlya nincs modellezve), a csúszóablakos rétegek változatlanok, az illesztés ideje arányos a sorok × próbák számával fejenként. Még nincs összekötve a kontextuscsomagokkal (egy összezsugorított csomag nem illeszthető be). Valódi modellen még semmi sincs mérve.

A kontextuscsomagok fél méretben is tárolhatók: `corbul packs --pack-kv f16` és `corbul serve --packs-kv f16` (F16-kerekítés minden soron), és veszteség nélkül tömöríthetők: `--pack-compress`, `--packs-compress`.

A Colibri Companionban: a *Dokumentumok → Kontextuscsomagok (corbul)* alatt ott van az F16 és a tömörítés beállítása (a corbul következő indításakor lép életbe) és a *A gyorsítótár zsugorításának mérése…*, amely ezt a parancsot a corbul által futtatott modellen indítja el (a modellt másodszor is betölti).
