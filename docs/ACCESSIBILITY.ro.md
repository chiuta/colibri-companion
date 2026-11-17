# Accesibilitate

Colibri Companion vrea să poată fi folosit de cât mai mulți oameni. Tot ce urmează se află la **Accesibilitate**, în bara de sus (sau cu `Ctrl+Shift+A`).

## Ce există

| Nevoie | Ce face aplicația |
| --- | --- |
| Vedere slabă | Mărimea textului de la 80% la 300%; un aspect cu contrast ridicat; fontul Atkinson Hyperlegible (SIL OFL); mai mult spațiu între rânduri și butoane; un contur gros în jurul elementului pe care se află tastatura |
| Nevăzători | Interfața este expusă cititoarelor de ecran prin AccessKit; orice răspuns poate fi citit cu voce tare cu vocea calculatorului (buton, `Ctrl+Shift+S` sau automat) |
| Daltonism | O paletă albastru/portocaliu în locul celei verde/roșu, în toate cele trei aspecte |
| Mâini și mișcare | Butoane de până la 64 de puncte înălțime; toată aplicația merge doar cu tastatura; nu se cere niciodată tragere sau mișcare fină |
| Auz | Aplicația nu scoate niciun sunet și nu îți cere să asculți; bara de activități poate clipi când un răspuns e gata |
| Dislexie, atenție, încărcare cognitivă | Font ușor de citit, rânduri mai aerisite, fără animații, meniuri simple și un stil de răspuns în *limbaj simplu* (propoziții scurte, cuvinte de zi cu zi) |

Șapte presetări (vedere slabă, nevăzători, mișcări greu de controlat, dislexie, citire mai ușoară, daltonism, auz) pun la punct pachete rezonabile dintr-un clic; apoi fiecare opțiune poate fi schimbată separat.

## Cum se verifică

* Fiecare pereche de culori text–fundal folosită de interfață, în toate cele trei aspecte și cu paleta pentru daltoniști pornită sau oprită, trebuie să atingă contrastul 4,5 : 1 (WCAG AA pentru text normal). Este un test unitar (`theme::tests::every_look_is_readable`), deci o schimbare de culoare care îl încalcă oprește construirea.
* Culorile „bun/rău” din paleta pentru daltoniști trebuie să difere și ca luminozitate, ca să rămână distinctibile în tonuri de gri.
* Textul trimis programului de vorbire circulă pe intrarea standard sau într-o variabilă de mediu, niciodată în linia de comandă, așa că un răspuns nu poate fi luat drept opțiune (testat unitar).

## Ce **nu** a fost verificat

* Aplicația nu a fost încercată cu un cititor de ecran real (NVDA, JAWS, Narrator, VoiceOver, Orca). Arborele de accesibilitate vine din egui/AccessKit; butoanele doar cu pictogramă pot fi anunțate prost. Rapoartele sunt binevenite.
* Citirea cu voce tare folosește `espeak-ng`/`espeak`/`spd-say` pe Linux, `say` pe macOS și System.Speech prin PowerShell pe Windows. Doar construcția comenzii este testată; vorbirea în sine nu a fost auzită aici. Pe macOS și Windows limba vocii este cea aleasă în sistem.
* *Intrarea* vocală este lăsată dictării sistemului de operare (Windows `Win+H`, macOS `Fn` de două ori). Dictarea offline în aplicație (whisper.cpp) este planificată în `ROADMAP-INTERCONNECT.md`.
* Scanarea pentru comutator și urmărirea privirii nu sunt implementate; uneltele sistemului de operare funcționează peste suportul de tastatură.
