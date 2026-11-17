# Összekapcsolási ütemterv: a csevegőablaktól a képes asszisztensig

Cél: a helyi modellt összekötni mindennel, ami megsokszorozza, amit tud, miközben az ember marad az irányító. Minden lépés három szabályt tart be: **előbb helyben, semmi sem hagyja el a gépet igen nélkül, minden művelet látható és visszavonható.**

## 0. A biztonsági keret (először épül, minden más használja)
- **Képességi szintek** (lásd `platforms.rs`): Olvasás → Keresés → Írás → Értesítés → Cselekvés. A kapcsolat egy szinten jön létre, és sosem léphet túl rajta.
- **Jóváhagyási kártya** minden íráshoz, küldéshez és műveletre: pontosan megmutatja, mi fog történni, *Egyszer engedélyezem / Ennél az automatizálásnál engedélyezem / Elutasítom* gombokkal. Olvasáshoz nem kell kártya.
- **Nem megbízható tartalom** (weboldalak, hibajegyek, e-mailek, dokumentumok) keretezve és adatként megjelölve érkezik. Az az eszközhívás, amely nem megbízható tartalom után jön, mindig újra rákérdez. Ez a fő védelem a prompt-injekció ellen.
- **Keretek és fékek**: lépések, modellhívások, percek és bájtok futásonként; egy *Mindent leállít* billentyű.
- **Naplózás** minden kérésről, amely elhagyta a számítógépet, és minden végrehajtott műveletről, egyszerű fájlban.
- **Titkok**: a tokenek alapból a memóriában élnek; a mentésük kifejezett döntés.

## 1. Eszközök a modellnek (a szorzó)
A Colibri a legtöbb motoron kínál eszközhívást. A program kis, őszinte szerszámosládát ad a modellnek: weboldal, keresés, tárolófájl, mappakeresés, számológép, Python homokozóban, táblázat-lekérdezés, jegyzetkeresés, fájlmentés, webhook. A modell kér, a program ellenőrzi a szintet és a jóváhagyást, lefuttatja, és az eredményt adatként adja vissza.

## 2. MCP: egy dugasz ezernyi eszközhöz
A Model Context Protocol beszélése kliensként (és a `coli mcp` megtartása szerverként). Bármely MCP-szerver (fájlok, adatbázisok, böngészők, naptárak, tervezőeszközök, otthoni eszközök) ugyanazon jóváhagyási szabályok alatt elérhetővé válik. Ez a legnagyobb hatókör-ugrás a legkevesebb kódért.

## 3. Veled növekvő emlékezet és tudás
- Helyi index (most BM25, később beágyazások) mappák, jegyzetek, mentett csevegések, kód, levélarchívumok fölött.
- A válaszok idézik a felhasznált szakaszt; a „mutasd meg, hol” megnyitja.
- Offline könyvtárak: Wikipédia a Kiwixen át, arXiv- és Wikidata-pillanatképek, kézikönyvek, törvények.
- A *csillagkép*: az emlékek összekötött csillagokként (címkék, emberek, projektek), amelyeket láthatsz és szerkeszthetsz.

## 4. Érzékek és hang (minden helyben)
- Hallás: whisper.cpp diktáláshoz és értekezlet-átiratokhoz.
- Beszéd: Piper vagy a rendszerhang, a felület nyelvének megfelelően.
- Látás: képeket fogadó modellek (a colibri felsorolja, melyek), plusz OCR (Tesseract) beolvasásokhoz és papírfényképekhez, plusz a kamera vagy egy képernyőkép kérésre.
- Alkotás: Qwen-Image a colibrin át, ComfyUI vagy Stable Diffusion képekhez; hang- és videóeszközök, ahogy megjelennek.

## 5. Több elme
- **Párbaj**: vak A/B Elóval, hogy kiderüljön, melyik modell a legjobb *neked*.
- **Tanács**: több modell válaszol, egy bíró modell összefésüli, és jelzi az egyet nem értést.
- **Útválasztó**: kis, gyors modell a könnyű kérdésekre, a nagy a nehezekre, döntési modell választja (a colibri szállítja a Layát, a GLiNER2.5-Decide-ot, a Clefet).
- **Specialisták**: kód, fordítás, látás, mind automatikusan választva, mind cserélhető.

## 6. Pontosság: a modell ellenőrizze magát
- Az általa írt Pythont homokozóban futtatni, és a valódi kimenetet megmutatni.
- Egységtesztek az általa készített kódhoz, felkínálás előtt lefuttatva.
- Önkonzisztencia (kétszer rákérdezni, összevetni), tényekhez kötelező hivatkozás, „mutasd a forrásaidat” kapcsoló és kis kiértékelő keret konfidenciaintervallumokkal (mint a LocalMindban).

## 7. Ügynökök és automatizálások
Mentett lépéssorok (olvas, kérdez, ellenőriz, ment, értesít) indítókkal: kézzel, időzítve, mappaváltozáskor, webhook érkezésekor. Sablonok: reggeli összefoglaló, postaláda-szortírozási vázlat, tárolófigyelő, értekezlet-jegyzet, kutatási összefoglaló, heti áttekintés. Minden lépés mutatja a szintjét.

## 8. Eszközök és a fizikai világ (utoljára, és a legnagyobb óvatossággal)
Home Assistant, MQTT, soros eszközök, 3D-nyomtatók, kamerák. Alapból csak olvasás; a műveletek engedélylistán át; soha zárak, riasztók vagy bármi biztonságkritikus.

## 9. Több számítógép, egy agy
Modell megosztása az otthoni hálózaton (az erős gép szolgál, a laptop használja), promptok, emlékezet és beállítások szinkronizálása a gépek között (titkosítva, egyenrangú kapcsolaton, felhő nélkül), és a munka szétosztása több videokártyára, ahol a motor támogatja.

## 10. Legyen erőfeszítés nélküli
Első indításkor útmutató, amely a hardverhez választ modellt, ötgombos *egyszerű mód*, egykattintásos kapcsolatok a katalógusból, „mit tehetek ezzel?” javaslatok, látható „mit lát most a modell” panel, és minden nyolc nyelven.

## A munka sorrendje
1. Mappák és nagy fájlok, promptkönyvtár, válaszstílusok (kész).
2. Online hozzáférés, GitHub/GitLab/Gitea/weboldalak, a katalógus (folyamatban).
3. Jóváhagyási kártyák, naplózás, keretek (a biztonsági keret).
4. Automatizálás webhookokkal és értesítésekkel.
5. Párbaj, tanács, útválasztó.
6. Eszközök a modellnek, majd MCP-kliens.
7. Index és emlékezet, hang és OCR.
8. Eszközök, több gép.

Amit *nem* ígérünk: ebből semmit sem teszteltek még valódi, eszközhívásos colibri-motorokkal, és minden külső szolgáltatás idővel változtatja az API-ját; a katalógusban „tervezett”-nek jelölt tételek tervek, nem funkciók.
