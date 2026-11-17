# Integrarea LegatorOCR

Stare: bibliotecă în `crates/core/src/legatorocr.rs`, legată de fereastra Documente prin `crates/app/src/ocr_ui.rs` (secțiunea „Fișiere scanate (LegatorOCR)”). Testată cu un binar real `legatorocr 0.2.0-pre.65` pe o singură pagină generată cu text sintetic curat; niciodată pe scanarea reală a unei cărți.

## Ce face

LegatorOCR (github.com/chiuta/LegatorOCR, **MIT**) transformă scanările în PDF sau text căutabil. Colibri Companion (Apache-2.0) nu leagă și nu copiază niciodată codul lui; doar rulează programul ca o comandă externă, lucru permis de ambele licențe.

1. `locate` / `locate_in`: caută un executabil `legatorocr` (un dosar dat de utilizator, dosarul aplicației, apoi `PATH`). Doar sistemul de fișiere, fără rețea.
2. `version`: rulează `legatorocr --version`; răspunsul trebuie să înceapă cu `legatorocr`, ca un fișier fără legătură cu același nume să nu fie luat drept program.
3. `collect_inputs`: o imagine sau un PDF, ori imaginile și PDF-urile aflate direct într-un dosar, în ordinea numelor (cel mult 2000).
4. `read_scan`: rulează `legatorocr <pagini> --ocr --lang <cod> --txt <fișier> --format none --out-dir <dosar>` (plus `--clean-text` pentru română) prin `proc::run_capture`, cu limită de timp, apoi citește textul înapoi (cel mult 32 MB).
5. `ingest_scan`: adaugă textul la o colecție de documente (`rag::Library::add_document`) sub numele `<fișier sau dosar> (OCR)`. Modelul local răspunde apoi din scanare, iar pasajele sunt citate cu acel nume, ca orice alt document.

Siguranță: argumentele se transmit pe rând (fără shell); intrările sunt făcute absolute, ca un fișier numit `-x.png` să nu fie citit ca opțiune; codul de limbă este verificat (`clean_lang`); ieșirea este plafonată; o rulare care nu se termină este oprită; erorile se întorc cu ultimele rânduri din ieșirea de eroare a programului; nimic nu pornește dacă nu e chemat.

## Limite spuse deschis

- Textul este OCR. Nu este o transcriere verificată, iar calitatea OCR (și ce limbi funcționează) depinde de LegatorOCR și de pachetele de limbă Tesseract instalate: `legatorocr --doctor` le listează (pe mașina de test lipseau modelele pentru română și maghiară).
- Numerele de pagină se pierd: toată scanarea e un singur document, așa că o citare numește scanarea, nu pagina.
- Căutarea în bibliotecă se face pe cuvinte-cheie (BM25), ca pentru orice document.
- Fereastra arată secțiunea doar când programul este găsit (căutat o dată, în fundal, la deschiderea ferestrei; nimic nu se descarcă și nu pornește singur). Fără el apare doar un indiciu cu `github.com/chiuta/LegatorOCR` și un buton „Caută din nou”.
- Rularea este un fir în fundal cu buton Oprește; LegatorOCR nu dă nicio cifră de progres, așa că fereastra arată secundele scurse, nu un procent. Închiderea întregii aplicații în timpul unei rulări nu este tratată special.
- Lista de limbi este cele 8 limbi ale aplicației mapate pe codurile Tesseract; dacă una funcționează depinde de datele instalate pentru LegatorOCR (`legatorocr --doctor`). Aplicația nu verifică asta dinainte: o limbă lipsă se întoarce ca mesajul de eroare al programului însuși.
- Imaginile de pagină trase în fereastră (nu PDF-urile) ajung în câmpul de scanare când programul este găsit.

## Încearcă

```
cargo test -p companion-core legatorocr
LEGATOROCR_BIN=/path/to/legatorocr LEGATOROCR_TEST_PAGE=/path/to/page.png \
  cargo test -p companion-core real_legatorocr -- --nocapture
```
