# Kontextcsomagok

Újrahasznosítja egy szövegen végzett munkát, amikor az másik promptban újra megjelenik (csak a corbul motor; clean-room, saját kód).

Az előtag-gyorsítótár csak akkor talál, ha a közös szöveg áll elöl. A *csomag* egy önállóan kiszámított szöveg KV-gyorsítótára, polcon tárolva (aláírt, LRU-korlátos pillanatkép-tár).

A `compose` egymás után fűzi a csomagokat: a sorokat másolja, a kulcsokat a pozíciókülönbséggel elforgatja (pontos); majd minden további csomag elejét növekvő ablakokban újraszámolja, amíg a mért változás egy tűrés alá nem csökken. A kérdést mindig újonnan számolja.

A `tolerance = 0` mindent újraszámol, és pontosan a teljes prefillt adja. A csúszóablakos rétegű, kiürített vagy gyorsítón élő gyorsítótárú modelleket elutasítja, és a szokásos prefillre vált.

Javításig nem pontos: az áthelyezett sorokat tartalmazó állapot `is_exact() == false`-t jelez, és soha nem íródik lemezre.

Korlátok: JustVugg colibri motorja nem tud gyorsítótárat exportálni, ezért csak a corbullal működik. Kis szintetikus modelleken tesztelve; valódi modelleken a minőséget még nem mértük.

## Parancssor

`corbul packs MODEL --session DIR --pack FÁJL… --question SZÖVEG [--tolerance 0.02] [--probe 8] [--max-new 64] [--check]` minden fájlt egyszer a polcra tesz, majd a tárolt csomagokkal válaszol, tetszőleges sorrendben. A `--check` teljes prefillt is futtat, és kiírja a pontszámkülönbséget és mindkettő idejét.

## Szerver

A `corbul serve MODEL --packs DIR [--packs-cap 2g]` hozzáadja a `POST /v1/packs/complete` végpontot ezzel: `{"packs": [szöveg, …], "question": "…", "max_tokens": 64, "tolerance": 0.02, "probe": 8}`. A válasz tartalmaz `text`, `usage` és `packs` mezőt (már polcon lévő szövegek, másolt, újraszámolt, friss és megspórolt sorok, illesztésenkénti mért változás). Egyszerű, mohó szövegkiegészítés, chat-sablon nélkül.
