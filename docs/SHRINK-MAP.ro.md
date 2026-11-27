# Pachete micșorate și hărți de cache între modele

Două extensii ale pachetelor de context ([CONTEXT-PACKS.ro.md](CONTEXT-PACKS.ro.md), doar motorul corbul; clean-room, cod original).

## Pachete micșorate

Când textele și întrebarea nu încap împreună în fereastra modelului, pachetele sunt refuzate. Cu micșorare ele sunt scurtate: `corbul packs --shrink auto` (doar cât e nevoie) sau `--shrink R` (păstrează mereu aproximativ fracția `R`, între 0,1 și 1 din rândurile fiecărui pachet). Același lucru face câmpul `"shrink": "auto"` (sau un număr) din `/v1/packs/complete`; răspunsul listează ce a fost micșorat (`"shrunk"`).

Cum: ultimele `--shrink-window` tokeni ai unui pachet (implicit 16) sunt recitiți și votează rândurile de păstrat (SnapKV); primele rânduri se păstrează mereu (acolo se adună atenția); valorile rămase sunt reajustate ([KV-FIT.ro.md](KV-FIT.ro.md); `--no-refit` oprește asta). Cheile păstrate sunt apoi rotite la pozițiile noilor locuri, deci pachetul devine *exact cache-ul textului său scurtat*: poate fi îmbinat și reparat ca orice pachet, iar cu reparare completă răspunsul e egal cu o preîncărcare simplă a textului scurtat. Primele `--shrink-skip` texte (implicit 1, de obicei instrucțiunile) rămân întregi. Micșorarea se face la asamblarea pachetelor; raftul păstrează pachetele întregi.

În Colibri Companion: *Documente → Pachete de context (corbul)* → *Micșorează textele prea lungi*.

Limite: se pierde informație, cât depinde de model și de text (măsoară cu `corbul kvfit` sau cu fereastra *Măsoară micșorarea cache-ului…*); modelele cu straturi cu fereastră glisantă, cache-urile Q8 și cele de pe accelerator sunt refuzate; dacă textele nu încap nici după ce toate textele micșorabile sunt scoase, cererea eșuează. Testat până acum doar pe modele mici aleatorii.

## Hărți de cache între modele diferite

Un pachet scris de un model nu folosește altuia. Dacă cele două au același tokenizator, rândurile lor de cache pentru *aceiași tokeni* pot fi puse în relație: `corbul map-fit SRC DST --text FIȘIER --out HARTA` citește textul cu ambele modele și potrivește, pe straturi, o hartă liniară (cele mai mici pătrate cu ridge, cu medie) de la rândurile de chei și valori ale lui SRC la cele ale lui DST. Cheile sunt comparate fără poziție (rotite înapoi la 0 și rotite la loc la import), deci tabelele rotative și mărimile capetelor pot diferi; straturile se împerechează după adâncimea relativă. Afișează eroarea pe rândurile lăsate în afara potrivirii, lângă eroarea ghicirii mediei, și avertizează dacă harta nu e mai bună.

`corbul map-check SRC DST --map HARTA --text FIȘIER --question Q` împarte textul în două pachete scrise de SRC, le aduce în DST prin hartă și răspunde în trei feluri: preîncărcarea completă făcută de DST (adevărul), cache-ul mapat cu repararea obișnuită (`--tolerance`, implicit 0,02) și cache-ul mapat fără reparare. Afișează rândurile recalculate și cât de departe e fiecare răspuns de adevăr.

Un rând mapat e doar o ghicire: rândurile importate sunt marcate aproximative, iar repararea recalculează câte e nevoie (pe toate, cu toleranță 0, ceea ce egalează o preîncărcare simplă). O hartă proastă costă deci timp, niciodată corectitudine. Punctul de intrare din bibliotecă este `ctxpack::compose_mapped`.

Limite: tokenizatoarele trebuie să dea aceiași identificatori (se verifică, altfel nu se potrivește nimic); harta e pentru o singură pereche de modele (amprentele se verifică); mărimea ei este `kd × kd' × straturi × 8` octeți; potrivirea ține rândurile de calibrare în memorie (`--rows`, implicit 1024). O hartă liniară între cache-urile unor modele fără legătură poate fi slabă; dacă merită pentru perechea ta se poate afla doar măsurând. Nu e încă în server sau în Companion și nimic nu a fost măsurat pe o pereche reală de modele.
