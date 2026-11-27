# Micșorarea cache-ului și refacerea valorilor

`corbul kvfit MODEL --text FILE --question Q` măsoară ce costă păstrarea doar a unei fracțiuni din rândurile cache-ului unui text (le alege SnapKV), cu și fără **refacerea valorilor** care rămân.

Ultimele `--window` tokenuri ale textului sunt înregistrate ca interogări de probă. Valorile rândurilor păstrate sunt apoi mutate cât mai puțin, astfel încât atenția asupra lor să dea, pe probe, ce dădea atenția asupra tuturor rândurilor (cele mai mici pătrate cu regularizare, în forma duală, `crates/engine/src/kvfit.rs`). Refacerea nu poate fi mai rea decât valorile simple *pe probe*; dacă ajută la o întrebare care nu a fost probată depinde de model și de text, și doar această comandă, rulată pe modelul tău, îți spune.

Opțiuni: `--budget 0.25` (fracțiunea de rânduri păstrate), `--window 16`, `--ridge 0.01`, `--max-new 32`. Afișează eroarea refacerii pe probe, cea mai mare diferență a primelor scoruri față de cache-ul întreg, dacă primul token coincide și câte dintre primele tokenuri ale răspunsului coincid.

Limite: doar valorile (cheile poartă poziția și rămân; greutatea rândurilor eliminate nu e modelată), straturile cu fereastră glisantă rămân neschimbate, refacerea costă timp proporțional cu rânduri × probe pe cap. Acum legată de pachetele de context: vezi SHRINK-MAP.ro.md (`--shrink`). Nimic măsurat încă pe un model real.

Pachetele de context pot fi păstrate și la jumătate de mărime: `corbul packs --pack-kv f16` și `corbul serve --packs-kv f16` (rotunjire F16 pe fiecare rând), și comprimate fără pierderi: `--pack-compress`, `--packs-compress`.

În Colibri Companion: *Documente → Pachete de context (corbul)* are opțiunile F16 și compresie (se aplică la următoarea pornire a lui corbul) și *Măsoară micșorarea cache-ului…*, care rulează această comandă pe modelul pe care îl rulează corbul (încarcă modelul a doua oară).
