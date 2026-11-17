# A LegatorOCR beépítése

Állapot: könyvtár a `crates/core/src/legatorocr.rs` fájlban, az Iratok ablakhoz a `crates/app/src/ocr_ui.rs` köti („Beolvasott fájlok (LegatorOCR)” szakasz). Egy valódi `legatorocr 0.2.0-pre.65` binárissal tesztelve egyetlen generált, tiszta szintetikus szövegű oldalon; egy valódi könyvbeolvasáson soha.

## Mit tesz

A LegatorOCR (github.com/chiuta/LegatorOCR, **MIT**) a beolvasásokat kereshető PDF-fé vagy szöveggé alakítja. A Colibri Companion (Apache-2.0) soha nem kapcsolja hozzá és nem másolja a kódját; csak külső parancsként futtatja a programot, amit mindkét licenc megenged.

1. `locate` / `locate_in`: `legatorocr` futtatható fájlt keres (a felhasználó által megadott mappa, az alkalmazás mappája, majd a `PATH`). Csak fájlrendszer, hálózat nélkül.
2. `version`: lefuttatja a `legatorocr --version` parancsot; a válasznak `legatorocr`-rel kell kezdődnie, hogy egy azonos nevű, idegen fájlt ne tévesszenek össze a programmal.
3. `collect_inputs`: egy kép vagy PDF, illetve egy mappában közvetlenül található képek és PDF-ek, névsorrendben (legfeljebb 2000).
4. `read_scan`: lefuttatja a `legatorocr <oldalak> --ocr --lang <kód> --txt <fájl> --format none --out-dir <mappa>` parancsot (románhoz még `--clean-text`) a `proc::run_capture` útján, időkorláttal, majd visszaolvassa a szöveget (legfeljebb 32 MB).
5. `ingest_scan`: ezt a szöveget hozzáadja egy dokumentumgyűjteményhez (`rag::Library::add_document`) `<fájl vagy mappa> (OCR)` néven. A helyi modell ezután a beolvasásból válaszol, a szakaszokat pedig ezen a néven idézi, mint bármely más dokumentumot.

Biztonság: az argumentumok egyenként kerülnek átadásra (shell nélkül); a bemenetek abszolút útvonallá válnak, hogy egy `-x.png` nevű fájlt ne olvassanak kapcsolóként; a nyelvkód ellenőrzött (`clean_lang`); a kimenet korlátozott; a be nem fejeződő futást leállítja; a hibák a program hibakimenetének utolsó soraival érkeznek vissza; semmi sem indul el, ha nem hívják.

## Őszinte korlátok

- A szöveg OCR. Nem ellenőrzött átirat, az OCR minősége (és hogy mely nyelvek működnek) a LegatorOCR-tól és a telepített Tesseract nyelvi csomagoktól függ: a `legatorocr --doctor` felsorolja őket (a tesztgépen hiányzott a román és a magyar modell).
- Az oldalszámok elvesznek: az egész beolvasás egyetlen dokumentum, így az idézet a beolvasást nevezi meg, nem az oldalt.
- A könyvtárban a keresés kulcsszavas (BM25), mint minden dokumentumnál.
- Az ablak csak akkor mutatja a szakaszt, ha a program megvan (egyszer keresi, a háttérben, az ablak megnyitásakor; semmit nem tölt le és nem indít el magától). Nélküle csak egy tipp jelenik meg a `github.com/chiuta/LegatorOCR` címmel és egy „Keresés újra” gomb.
- A futás háttérszálon megy Leállítás gombbal; a LegatorOCR nem ad előrehaladási adatot, ezért az ablak az eltelt másodperceket mutatja, nem százalékot. A teljes alkalmazás futás közbeni bezárását külön nem kezeli.
- A nyelvlista az alkalmazás 8 nyelve Tesseract-kódokra leképezve; hogy működik-e egy nyelv, a LegatorOCR-hoz telepített adatoktól függ (`legatorocr --doctor`). Az alkalmazás ezt előre nem ellenőrzi: a hiányzó nyelv a program saját hibaüzeneteként tér vissza.
- Az ablakba húzott oldalképek (a PDF-ek nem) a beolvasási mezőbe kerülnek, ha a program megvan.

## Próbáld ki

```
cargo test -p companion-core legatorocr
LEGATOROCR_BIN=/path/to/legatorocr LEGATOROCR_TEST_PAGE=/path/to/page.png \
  cargo test -p companion-core real_legatorocr -- --nocapture
```
