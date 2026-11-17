# Colibri Companion

[English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · **Magyar**

**Weboldal: https://colibri-companion.chiuta.ro** — letöltés, videós útmutatók, segítség.

Egyszerű asztali alkalmazás a [colibrihez](https://github.com/JustVugg/colibri): válassz a számítógépedhez illő modellt, töltsd le egy kattintással, majd csevegj — a modell a saját gépeden fut, fiók és felhőszolgáltatás nélkül.

## Indítás

```bash
cargo run --release -p colibri-companion            # a valódi alkalmazás
cargo run --release -p colibri-companion -- --demo  # demó, colibri nélkül
# kapcsolók: --lang en|ro|fr|it|es|pt|de|hu   --version   --help
```

### A négy lépés

(1) megkeresi a Pythont és a colibrit (és felajánlja a colibri letöltését, ha hiányzik), (2) lehetővé teszi a modell kiválasztását (amelyik nem fér a RAM-ba, el van rejtve; az ajánlott meg van jelölve), (3) letölt és telepít folyamatjelzéssel, szünettel és folytatással, (4) elindítja a szervert, és megnyitja a csevegést.

Csevegés: folyamatos válaszok, azonnali leállítás, Markdown, összecsukható gondolkodás, mentett beszélgetések, stílus (Pontos / Kiegyensúlyozott / Kreatív), másolás és újragenerálás. Témák: világos, sötét, nagy kontraszt.

## Hogyan működik

Az alkalmazás nem tartalmazza a colibrit. A colibri JSON-parancsait futtatja (`setup --list --json`, `status --json`, `setup`, `start`, `stop`), és a `127.0.0.1`-en lévő OpenAI-kompatibilis átjárójával beszélget. Soha nem telepít rendszercsomagokat: ha valami hiányzik, megmutatja a másolható parancsot. Az adataid (beállítások, beszélgetések) helyben maradnak a felhasználói adatmappában; a `COLIBRI_COMPANION_HOME` ezt megváltoztathatja.

## Licencek

- Ez a projekt: Apache-2.0 (`LICENSE`, `NOTICE`), mint a colibri.
- Függőségek: csak megengedő licencek, a `cargo deny check licenses` (`deny.toml`) ellenőrzi; a teljes lista a `THIRD_PARTY_LICENSES.md` fájlban. Betűtípusok: Inter és JetBrains Mono (SIL OFL 1.1, a `licenses/` mappában).
- A modelleknek saját licencük van; az alkalmazás megmutatja, és letöltés előtt megerősítést kér.

## Platformok

Linux (x86-64, ARM64; Wayland és X11), Windows 10/11 és macOS (Intel és Apple Silicon) — egyetlen forráskód, OpenGL-megjelenítés (régi videókártyán és virtuális gépben is fut). A `.github/workflows/build.yml` minden változatot lefordít és tesztel, és kiadásra kész archívumokat készít. Linuxon a `packaging/linux/install.sh` felveszi az alkalmazást a menübe. HiDPI-skálázás, Ctrl/⌘+N, Ctrl/⌘+, és Esc billentyűk, képernyőolvasó-támogatás (AccessKit). Az automatikus fordítás és tesztelés mindhárom rendszerre kiterjed; a valódi Windows- és macOS-gépeken végzett kézi tesztelés még korlátozott, és a képernyőolvasó-címkéket (AccessKit) még nem próbáltuk ki valódi képernyőolvasóval.

Régebbi Linux (Debian 8+, Ubuntu 14.04+, CentOS 7+): kiegészítő buildek `linux-x86_64-glibc2.17` és `linux-aarch64-glibc2.17` néven, glibc 2.17-hez linkelve; a CI ellenőrzi a szimbólumalapot, és betölti a programot Ubuntu 16.04-ben. Az ablakkezelő és grafikus könyvtárak továbbra is a te rendszeredből jönnek.

**Dokumentumok és eszközök:** adj hozzá fájlokat (**Továbbiak → Dokumentumok**), és kérdezz róluk: az alkalmazás a számítógépeden keresi őket (kulcsszavak, felhő nélkül), és a válasz alatt megmutatja a felhasznált részeket. A **Továbbiak → MCP-eszközök** az általad hozzáadott és elindított MCP-szervereket köti be; a modell eszközt kérhet, és minden hívás a te jóváhagyásodat kéri (korai funkció, függ attól, hogy a modell követi-e a formátumot).

Modellek: tarts több modellt a számítógépeden, válts közöttük, vagy adj hozzá újakat egyetlen ablakból (**Modellek**); először a számítógéped által futtatható legképesebb modellt javasolja (kevés RAM esetén a szakértőket az SSD-ről olvassa). Sebesség: CUDA az NVIDIA-hoz, Vulkan más grafikus chipekhez, különben a processzor, automatikusan választva. Az **Erőforrások** élőben mutatja a processzor-, memória-, videokártya- és lemezhasználatot, valamint a bizonyítékot, hogy semmi sem hagyja el a számítógépedet (az alkalmazás és a motor minden nyitott kapcsolata, és a parancs, amellyel magad ellenőrizheted). Ha a colibri hiányzik, az alkalmazás letölti neked (git nélkül), és egy napi ellenőrzés (kikapcsolható) jelzi, ha a colibrinek vagy ennek az alkalmazásnak új verziója van.

**A számítógéped elemzése:** az **Erőforrások** ablakban az alkalmazás magától megtalálja a videokártyákat, és egyszerűen megmondja, elfér-e a modell a memóriában, és miért lehet lassú (swap, más programok, kihasználatlan kártya). Csak tanácsot ad; magától nem módosít beállítást.

**Újabb eszközök (mind valódi modell nélkül tesztelve; a weboldal megmondja, mi ellenőrzött és mi nem):** sebességtartomány minden modell alatt letöltés előtt (a memóriádból, a meghajtóból és a katalógusból; sosem mérés) és a ROUTE_TRACE felvétel elemzéssel arról, mi spórolna meghajtóolvasást; a rögzített memória és a szálak automatikus hangolása (**Erőforrások**); **Könnyű modellek a memóriádhoz** a Modellek ablakban (11 kis modell, szűrő a dokumentált eszközhíváshoz, az `ollama run` parancs másolása); **Hivatkozások ellenőrzése** és **Beszkennelt fájlok (LegatorOCR)** a Dokumentumok ablakban (az utóbbihoz kell a különálló LegatorOCR program, MIT licenc; az OCR-szöveg nem ellenőrzött átirat); valamint egy **Fogyasztás** panel minden válasz mellett (idő, első token, bemeneti és kimeneti tokenek, pillanatnyi, átlagos és csúcssebesség, felhasznált kontextus, ennek a számítógépnek a processzora, memóriája, swapja és videókártyája; a ≈ jelű számok becslések). Az Iratok ablak **Irodalomjegyzék** szakasza átalakítja a BibTeX-, RIS- és CSL JSON-fájlokat, és hivatkozási listát másol (APA, Harvard, IEEE); csak generált bejegyzéseken tesztelve. A TEZAUR-átvétel többi része (szakaszok előkeresése, eredetigazolás, MI-használati nyilatkozat) még könyvtár, amelyet az ablak nem használ. Több a `docs/` mappában.

## Nyelvek

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — a Beállításokban választható, a `--lang en|ro|fr|it|es|pt|de|hu` kapcsolóval, vagy első induláskor a rendszernyelv szerint. A fordítások a `crates/app/src/i18n.rs` fájlban vannak (a kulcs az angol szöveg); egy teszt ellenőrzi, hogy a felület minden szövegének megvan az összes fordítása. MI segítségével készültek, anyanyelvi beszélők nem ellenőrizték őket.

## Fejlesztés

```bash
cargo test --workspace   # tartalmaz egy végponttól végpontig tartó tesztet egy hamis `coli`-val (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```

Új: keresés a beszélgetésekben (Ctrl/⌘+K), átnevezés (jobb klikk), az utolsó üzenet szerkesztése, szövegfájlok behúzása az üzenetbe és mentés .md fájlként.
