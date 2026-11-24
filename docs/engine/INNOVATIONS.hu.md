# Miben más ez a motor

[English](INNOVATIONS.md) · [Română](INNOVATIONS.ro.md) · [Français](INNOVATIONS.fr.md) · [Italiano](INNOVATIONS.it.md) · [Español](INNOVATIONS.es.md) · [Português](INNOVATIONS.pt.md) · [Deutsch](INNOVATIONS.de.md) · [Magyar](INNOVATIONS.hu.md)

Közérthető jegyzetek olyan képességekről, amelyek a legtöbb más helyi motorból hiányoznak. Mindegyik leírja, mit csinál, hol éred el, és mit nem tud. Mindet csak szintetikus modelleken teszteltük; egyiket sem futtatták a szerzők valódi ellenőrzőponton. A „`corbul` esetében” azt jelenti, hogy ma létezik hozzá parancs vagy kapcsoló (`corbul help`); a „csak könyvtár” parancs nélküli Rust-kódot jelent. Lásd az [ENGINE.md](ENGINE.hu.md) és a [QUICKSTART.md](QUICKSTART.hu.md) fájlt.

## Integritás, hardver és memória

### LAN-háló (mesh)

`mesh` modul. A `corbul` esetében: `worker`, `generate --workers`, `--discover`, `--key`, `--wire-f16`. Az egy gépnek túl nagy modellt rétegtartományokra vágja, számítógépenként egyre. Köztük csak a rejtett állapot utazik. A gépek bizonyítják, hogy ugyanazt a kulcsot ismerik (kölcsönös HMAC-SHA256), minden keret MAC-et visel, a kézfogás méretei korlátozottak, egy kiesett munkást elhagy, és a rétegeit újraosztja. A sűrű f32 logitok megegyeznek az egygépes eredménnyel. Az UDP-felderítés bekapcsolásra épül: a munkás csak akkor válaszol szondákra, ha a `--discovery` vagy a `--discovery-port` kapcsolóval indították (a `--no-discovery` kikényszeríti a kikapcsolást), a koordinátor `--discover` kapcsolója pedig szondát szór szét, és megvárja azoknak a munkásoknak a válaszát, amelyek ugyanazt a modellt tartják. *Nem tud:* titkosítani (használj megbízható LAN-t vagy VPN-t); csak loopbacken tesztelték; az egy géphez képesti sebességet nem mérték.

### Rétegek folyamatos betöltése

`stream_layers` modul. A `corbul` esetében: `--stream-layers`, `--ram-budget`, `--stream-window`, `--disk-mbps` (becslésekhez). A memóriánál nagyobb sűrű modellt úgy futtatja, hogy a rétegek kis ablakát tartja memóriában, és egy háttérszálon előre betölt. Az eredmények megegyeznek a memóriában futó modelléval. *Nem tud:* minden token minden réteget beolvas, ezért a sebességet a lemez sebessége korlátozza; csak sűrű rétegek (a szakértők keverékét használó modellek ehelyett az irányított szakértőiket töltik folyamatosan, a `--ram-budget` kapcsolóval).

### Idézetzáras hivatkozások

`quotelock` modul. A `corbul` esetében: `--sources FILE` (ismételhető), `--no-lock`, `--citations FILE`; a szerver elfogadja a `sources` mezőt, és kínálja a `POST /v1/quotes/verify` végpontot. Dekódolás közben a « és » közötti szövegnek egy forrás szó szerinti szakaszának kell lennie (nem érzékeny a kis- és nagybetűre, a szóközökre, az idézőjelstílusokra és a román ékezetekre); a `[S3]«` az idézetet a 3. forrásra korlátozza. Egy ellenőrző ezután egyenként besorolja az idézeteket (szó szerinti, rossz forrás, csonka, módosított, összeolvasztott, alátámasztatlan), és pontozza az idézeteken kívüli mondatokat; a jelentés megmondja, ha munkahatárba ütközött (`limit_hit`). *Nem tud:* igazzá tenni az idézetek körüli szöveget; a folyamatos válaszok nem tartalmaznak idézetblokkot.

### Tervező, szabályozó és energiamérő

`governor` modul (és `plan`, `hw`). A `corbul` esetében: `plan`, `doctor`, `--governor`, `--no-governor`, `--verbose`, `--calibrate`. Betöltés előtt megmondja, hogy a modell elfér-e, folyamatosan töltődik-e, lassan fut-e vagy nem tud futni, becsült sebességgel. Futás közben a szabályozó olvassa a hőmérsékletet, a memóriát és az akkumulátort (Linux), és ingadozás nélkül tanácsol szálszámot és szüneteket. *Nem tud:* a sebességek becslések (a `--calibrate` körülbelül egy másodpercig méri a gépedet); az energia feltevés, hacsak nem olvasható RAPL-számláló; a szén-dioxid-intenzitás (400 g/kWh) helyőrző.

### KV-gyorsítótár-szabályok és munkamenet-pillanatképek

`kvpolicy`, `snapshot` modulok. A `corbul` esetében: `--kv-policy streaming|h2o|snapkv`, `--kv-budget`, `--kv-sinks`, `--session DIR`, `--session-cap`, `--compress`. A hosszú csevegéseket memóriakereten belül tartják (figyelemnyelők, nehéz találatok, promptsűrítés), és egy ellenőrzőösszeggel ellátott fájlból mentik vagy folytatják a beszélgetést, a leghosszabb mentett előtagot felhasználva. A pillanatképek kulcsos HMAC-ot is hordozhatnak. *Nem tud:* a kiszorítás tervezetten veszteséges; a `--kv-policy` és a `--session` nem kombinálható; nagyon hosszú csevegések túlléphetik a betanított környezetet.

### Reprodukálható mód és nyugták

`simd` (kapcsoló), `receipt` modulok. A `corbul` esetében: `--reproducible`, `--receipt FILE`, `--receipt-key`, `--receipt-key-file`, `--receipt-topk`, `--keep-steps`, `--leaves`, `verify-receipt`, `replay`. A nyugta hashlánccal rögzíti a modellt (a súlyok Merkle-ujjlenyomata), a tokenizálót, a mintavételező beállításait, a promptot és a kimenetet, SHA-256-tal és opcionális HMAC-kulccsal pecsételve; a `replay` újra lefuttatja. A sérült súlyokat egy tenzorig és darabig lokalizálja. *Nem tud:* bármit bizonyítani egy idegennek a HMAC-kulcs vagy az újrafuttatás nélkül (nincsenek nyilvános kulcsú aláírások); az MoE- és állapottér-modelleket elutasítja; az ARM nincs tesztelve; a reprodukálható mód sebességbe kerül.

### Román nyelvet előtérbe helyező nyelvi eszközök

`lang` modul. A `corbul` esetében: `lang detect`, `lang normalize`, `--lang CODE` (a szerverben nincs). Normalizálja az ș/ț és ş/ţ jeleket, javítja a mojibake-ot, 27 nyelvet azonosít, dekódolás közben tiltja az idegen írásrendszereket (román: nincs ş ţ), formázza a román számokat, pénzösszegeket és dátumokat, és helyreállítja a hiányzó ékezeteket, minden bizonytalan szót megjelölve. *Nem tud:* a helyreállítás körülbelül 650 alakos lexikont használ, és korpuszon nem mérték; az egyértelműtlen szavakat néhány szövegkörnyezeti szabály oldja fel; a nyelvfelismeréshez körülbelül öt szó kell.

### Bizonytalansági elemzések

`confidence` modul. Csak könyvtár (a `confidence_demo` bemutatópélda). Token-, szó-, mondat- és válaszszintű statisztikák hőtérképhez, átlátható hallucinációkockázat-heurisztika, önkonzisztencia-pontszámok és kalibrációs eszközök. *Nem tud:* a kockázati pontszám heurisztika, nem igazságdetektor.

## Futásidejű számítás, hangolás, visszakeresés

### Keresés és előretekintés

`search` modul. A `corbul` esetében: `search --method beam|best-of-n|vote` (`--width`, `--samples`, `--answer number|text`, `--length-penalty`, `--all`, `--explain`, `--json`) és a `generate` parancs `--lookahead` kapcsolója (`--ngram-min`, `--ngram-max`). Nyalábkeresés, legjobb-az-N-ből, önkonzisztencia-szavazás, beszélgetésfa (könyvtár), prompt-kereséses spekulatív dekódolás. *Nem tud:* az előretekintés csak akkor segít, ha a szöveg korábbi szöveget ismétel; a legjobb-az-N-ből jó pontozót igényel.

### Spekulatív dekódolás

`spec`, `selfspec` modulok. A `corbul` esetében: `--draft MODEL`, `--lookahead`, `--self-spec`, `--skip-layers auto|LIST`, `--draft-len`. Ugyanaz a mohó válasz és ugyanaz a mintavételezési eloszlás kevesebb teljes lefutással; a kimenet a végén jelenik meg. Csak a `--temperature`, `--top-k`, `--top-p` kapcsolók; nem használható strukturált kimenettel, nyugtákkal, munkamenetekkel vagy a hálóval. *Nem tud:* gyorsulást ígérni; az önspekuláció sima lépésekre tér vissza a körbeforduló csúszóablakos gyűrűknél, a tömörített gyorsítótáraknál és a kiszervezett rétegeknél.

### Automatikus vegyes kvantálás és a kvantáló

`autoquant`, `quantize` modulok. A `corbul` esetében: `autoquant FOLDER --budget 4.5bpw|6g|Q4_K` (`--recipe`, `--report`, `--candidates`, `--base`, `--imatrix`, `--sample-rows`) és `quantize IN OUT.gguf --type NAME | --recipe FILE`. Minden formátumban megméri a súlyhibát tenzorcsoportonként, és méretkeret alatt választ formátumokat (pontos hátizsák vagy mohó). *Nem tud:* a súlyhibát becsüli, nem a válaszminőséget; feltételezi, hogy a károk összeadódnak; i-kvantumok nincsenek.

### Terelővektorok és logit-lencse

`steer` modul. A `corbul` esetében: `steer train|apply|lens` (`--positive`, `--negative`, `--method mean-diff|pca`, `--out`, `--vector`, `--strength`, `--layers`, `--positions`, `--no-attention`). Két promptkészletből tanul egy irányt, és hozzáadja a reziduális áramhoz a súlyok módosítása nélkül; a lencse megmutatja, mit jósolna az egyes réteg. *Nem tud:* állapottér-modellekkel, strukturált kimenettel, spekulációval, nyugtákkal vagy munkamenetekkel működni; az egyik finomhangolásból származó vektort egy másik is elfogadja, ha ugyanaz az alakja.

### Helyi visszakeresés

`rag` modul. A `corbul` esetében: `rag add|search|ask` (`--query`, `--question`, `--k`, `--chunk-chars`, `--overlap`, `--embed-model`). Románt figyelembe vevő darabolás, HNSW-index tiszta Rustban, ékezetfeloldó BM25, rangfúzió, `[S1]…` számozású szakaszok, amelyek az idézetzáras hivatkozásokat táplálják, ellenőrzőösszeges fájlok. *Nem tud:* az alapértelmezett beágyazó hash-alapú (szavak és szórészek, nem jelentés); az `--embed-model` egy generatív modell rejtett állapotait használja (gyenge); a parancssor minden híváskor újraindexel, és sima szöveget olvas, nem PDF- vagy Office-fájlokat (a könyvtárban vannak beolvasássegédek).

### Modellösszefésülés

`merge` modul. A `corbul` esetében: `merge CONFIG.json` (`--out`, `--model`, `--dry-run`, `--max-memory`, `--overwrite`, `--verbose`). Lineáris, SLERP, feladataritmetika, TIES, DARE (magozott) és rétegszeletek, egyszerre egy tenzoron. Az alakokat és a tartományokat írás előtt ellenőrzi; a szimbolikus hivatkozásokat nem követi. *Nem tud:* minőséget garantálni; csak safetensors súlyokat ír, kvantáltat soha, a `config.json` fájlt és a tokenizálót magad másolod át.

### Szövegvízjel

`watermark` modul. A `corbul` esetében: `generate --watermark KEY` (`--wm-gamma`, `--wm-delta`, `--wm-context`) és `watermark detect MODEL --key K --text T | --file F` (`--no-dedup`). Kulcsos „zöldlistás” torzítás és z-pontszámos detektor. *Nem tud:* bizonyítékot adni; a rövid szövegek, a kód és az idézetek keveset hordoznak a jelből; a körülírás eltávolítja; aki ismeri a kulcsot, hamisíthatja; a jel hiánya nem bizonyítja az emberi szerzőséget.

## Multimodális (csak könyvtár)

### Látás

`vision`, `imageio` modulok. CLIP/SigLIP-stílusú enkóder, projektor, csempézés és egy `Model::forward_segments` horog beágyazások promptba juttatásához; PNG, JPEG (alap és progresszív), BMP és PNM dekóderek, a specifikációk alapján írva, méretkorlátokkal. *Nem tud:* a `corbul` programból használható lenni (csak szöveg); valódi ellenőrzőpontot nem futtattak.

### Hang

`audio` modul. WAV olvasása és írása, újramintavételezés, FFT, log-mel jellemzők, hangaktivitás-észlelés, Whisper-elrendezésű futtató. *Nem tud:* a `corbul` programból használható lenni; valódi Whisper-ellenőrzőpontot nem futtattak.

## Állapottér, hibrid és GPU (lásd ENGINE.md)

Az `ssm`, `hybrid`, `mla` modult és a `companion-engine-gpu` crate-et korlátaikkal együtt az ENGINE.md írja le.
