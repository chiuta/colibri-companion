# Colibri Companion

[English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · **Magyar**

Barátságos asztali alkalmazás a [colibrihez](https://github.com/JustVugg/colibri): válassz a számítógépedhez illő modellt, töltsd le egy kattintással, majd csevegj — minden helyben, fiók és felhő nélkül.

## Indítás

```bash
cargo run --release -p colibri-companion            # a valódi alkalmazás
cargo run --release -p colibri-companion -- --demo  # demó, colibri nélkül
# kapcsolók: --lang en|ro|fr|it|es|pt|de|hu   --version   --help
```

### A négy lépés

(1) megkeresi a Pythont és a colibrit (és felajánlja a colibri letöltését, ha hiányzik), (2) modellt választhatsz (amelyik nem fér a RAM-ba, el van rejtve; az ajánlott meg van jelölve), (3) letölt és telepít folyamatjelzéssel, szünettel és folytatással, (4) elindítja a szervert, és megnyitja a csevegést.

Csevegés: folyamatos válaszok, azonnali leállítás, Markdown, összecsukható gondolkodás, mentett beszélgetések, stílus (Pontos / Kiegyensúlyozott / Kreatív), másolás és újragenerálás. Témák: világos, sötét, nagy kontraszt.

## Hogyan működik

Az alkalmazás nem tartalmazza a colibrit. A colibri JSON-parancsait futtatja (`setup --list --json`, `status --json`, `setup`, `start`, `stop`), és a `127.0.0.1`-en lévő OpenAI-kompatibilis átjárójával beszélget. Soha nem telepít rendszercsomagokat: ha valami hiányzik, megmutatja a másolható parancsot. Az adataid (beállítások, beszélgetések) helyben maradnak a felhasználói adatmappában; a `COLIBRI_COMPANION_HOME` ezt megváltoztathatja.

## Licencek

- Ez a projekt: Apache-2.0 (`LICENSE`, `NOTICE`), mint a colibri.
- Függőségek: csak megengedő licencek, a `cargo deny check licenses` (`deny.toml`) ellenőrzi; a teljes lista a `THIRD_PARTY_LICENSES.md` fájlban. Betűtípusok: Inter és JetBrains Mono (SIL OFL 1.1, a `licenses/` mappában).
- A modelleknek saját licencük van; az alkalmazás megmutatja, és letöltés előtt megerősítést kér.

## Platformok

Linux (x86-64, ARM64; Wayland és X11), Windows 10/11 és macOS (Intel és Apple Silicon) — egyetlen forráskód, OpenGL-megjelenítés (régi videókártyán és virtuális gépben is fut). A `.github/workflows/build.yml` minden változatot lefordít és tesztel, és kiadásra kész archívumokat készít. Linuxon a `packaging/linux/install.sh` felveszi az alkalmazást a menübe. HiDPI-skálázás, Ctrl/⌘+N, Ctrl/⌘+, és Esc billentyűk, képernyőolvasó-támogatás (AccessKit).

**Dokumentumok és eszközök:** adjon hozzá fájlokat (**Továbbiak → Dokumentumok**), és kérdezzen róluk: az alkalmazás a számítógépén keresi őket (kulcsszavak, felhő nélkül), és a válasz alatt megmutatja a felhasznált részeket. A **Továbbiak → MCP-eszközök** az Ön által hozzáadott és elindított MCP-szervereket köti be; a modell eszközt kérhet, és minden hívás az Ön jóváhagyását kéri (korai funkció, függ attól, hogy a modell követi-e a formátumot).

Modellek: tarts több modellt a számítógépen, válts közöttük, vagy adj hozzá újakat egyetlen ablakból (**Modellek**); először a számítógéped által futtatható legképesebb modellt javasolja (kevés RAM esetén a szakértőket az SSD-ről olvassa). Sebesség: CUDA az NVIDIA-hoz, Vulkan más grafikus chipekhez, különben a processzor, automatikusan választva. Az **Erőforrások** élőben mutatja a processzor-, memória-, videokártya- és lemezhasználatot, valamint a bizonyítékot, hogy semmi sem hagyja el a számítógépedet (az alkalmazás és a motor minden nyitott kapcsolata, és a parancs, amivel magad ellenőrizheted). Ha a colibri hiányzik, letöltöm helyetted (git nélkül), és egy napi ellenőrzés (kikapcsolható) szól, ha a colibrinek vagy ennek az alkalmazásnak új verziója van.

**A számítógépe elemzése:** az **Erőforrások** ablakban az alkalmazás magától megtalálja a videokártyákat, és egyszerűen megmondja, elfér-e a modell a memóriában, és miért lehet lassú (swap, más programok, kihasználatlan kártya). Csak tanácsot ad; magától nem módosít beállítást.

## Nyelvek

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — a Beállításokban választható, a `--lang en|ro|fr|it|es|pt|de|hu` kapcsolóval, vagy első induláskor a rendszernyelv szerint. A fordítások a `crates/app/src/i18n.rs` fájlban vannak (a kulcs az angol szöveg); egy teszt ellenőrzi, hogy a felület minden szövegének megvan az összes fordítása. MI segítségével készültek, anyanyelvi beszélők nem ellenőrizték őket.

## Fejlesztés

```bash
cargo test --workspace   # tartalmaz egy végponttól végpontig tartó tesztet egy hamis `coli`-val (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```

Új: keresés a beszélgetésekben (Ctrl/⌘+K), átnevezés (jobb klikk), az utolsó üzenet szerkesztése, szövegfájlok behúzása az üzenetbe és mentés .md fájlként.
