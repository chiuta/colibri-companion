# Akadálymentesség

A Colibri Companion azt akarja, hogy minél többen használhassák. Minden, ami alább következik, a felső sáv **Akadálymentesség** gombja alatt érhető el (vagy a `Ctrl+Shift+A` billentyűvel).

## Mi van benne

| Igény | Mit tesz az alkalmazás |
| --- | --- |
| Gyengénlátás | Szövegméret 80%-tól 300%-ig; nagy kontrasztú megjelenés; az Atkinson Hyperlegible betűtípus (SIL OFL); nagyobb hely a sorok és a gombok között; vastag keret az elem körül, amelyen a billentyűzet áll |
| Vakság | A felület az AccessKiten keresztül elérhető a képernyőolvasóknak; minden válasz felolvasható a számítógép saját hangján (gomb, `Ctrl+Shift+S`, vagy automatikusan) |
| Színtévesztés | Kék/narancs paletta a zöld/piros helyett mind a három megjelenésben |
| Kéz és mozgás | Legfeljebb 64 pont magas gombok; az egész alkalmazás csak billentyűzettel is működik; soha nem kell húzni vagy finom mozdulatokat tenni |
| Hallás | Az alkalmazás soha nem ad hangot, és soha nem kéri, hogy hallgass; a tálca villoghat, ha kész a válasz Az Esc a szövegmezőből is bezárja a nyitott ablakot, és egyetlen ablak sem lóg ki a képernyőről. |
| Diszlexia, figyelem, kognitív terhelés | Jól olvasható betűtípus, szellősebb sorok, nincs animáció, egyszerű menük és *könnyű nyelvű* válaszstílus (rövid mondatok, hétköznapi szavak) |

Hét előbeállítás (gyengénlátás, vakság, nehezen irányítható mozgás, diszlexia, könnyebb olvasás, színtévesztés, hallás) egy kattintással állít be észszerű csomagokat; utána minden beállítás külön is módosítható.

## Hogyan ellenőrzik

* A felület által használt minden szöveg–háttér színpárnak mindhárom megjelenésben, a színtévesztőknek szóló paletta be- és kikapcsolt állapotában is el kell érnie a 4,5 : 1 kontrasztot (WCAG AA normál szövegre). Ez egy egységteszt (`theme::tests::every_look_is_readable`), ezért egy színváltoztatás, amely megsérti, megbuktatja a fordítást.
* A színtévesztőknek szóló paletta „jó/rossz” színeinek fényességben is el kell térniük, hogy szürkeárnyalatban is megkülönböztethetők maradjanak.
* A beszédprogramnak küldött szöveg a szabványos bemeneten vagy környezeti változóban utazik, soha nem a parancssorban, így egy válasz nem téveszthető össze egy kapcsolóval (egységteszttel ellenőrizve).

## Ami **nincs** ellenőrizve

* Az alkalmazást nem próbálták ki valódi képernyőolvasóval (NVDA, JAWS, Narrator, VoiceOver, Orca). A hozzáférhetőségi fa az egui/AccessKitből jön; a csak ikonos gombokat rosszul mondhatják be. A visszajelzést szívesen vesszük.
* A felolvasás Linuxon az `espeak-ng`/`espeak`/`spd-say` programot, macOS-en a `say` parancsot, Windowson a System.Speech-et használja PowerShellen át. Csak a parancs összeállítása tesztelt; magát a beszédet itt nem hallották. macOS-en és Windowson a hang nyelve a rendszerben választott.
* A beszédbevitelt az operációs rendszer diktálására bízzuk (Windows `Win+H`, macOS kétszer `Fn`). Az alkalmazáson belüli offline diktálás (whisper.cpp) a `ROADMAP-INTERCONNECT.md`-ben szerepel tervként.
* A kapcsolós pásztázás és a szemkövetés nincs megvalósítva; az operációs rendszer eszközei a billentyűzettámogatásra épülve működnek.
