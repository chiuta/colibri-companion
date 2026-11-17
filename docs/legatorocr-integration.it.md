# Integrazione di LegatorOCR

Stato: libreria in `crates/core/src/legatorocr.rs`, collegata alla finestra Documenti da `crates/app/src/ocr_ui.rs` (sezione «File scansionati (LegatorOCR)»). Provata con un vero binario `legatorocr 0.2.0-pre.65` su una sola pagina generata di testo sintetico pulito; mai su una scansione reale di un libro.

## Cosa fa

LegatorOCR (github.com/chiuta/LegatorOCR, **MIT**) trasforma le scansioni in PDF o testo ricercabile. Colibri Companion (Apache-2.0) non collega né copia mai il suo codice; esegue soltanto il programma come comando esterno, cosa che entrambe le licenze consentono.

1. `locate` / `locate_in`: cerca un eseguibile `legatorocr` (una cartella indicata dall'utente, la cartella dell'app, poi `PATH`). Solo file system, nessuna rete.
2. `version`: esegue `legatorocr --version`; la risposta deve iniziare con `legatorocr`, così un file estraneo con lo stesso nome non viene scambiato per il programma.
3. `collect_inputs`: un'immagine o un PDF, oppure le immagini e i PDF direttamente in una cartella, in ordine di nome (al massimo 2000).
4. `read_scan`: esegue `legatorocr <pagine> --ocr --lang <codice> --txt <file> --format none --out-dir <cartella>` (più `--clean-text` per il rumeno) tramite `proc::run_capture` con un limite di tempo, poi rilegge il testo (al massimo 32 MB).
5. `ingest_scan`: aggiunge quel testo a una raccolta di documenti (`rag::Library::add_document`) con il nome `<file o cartella> (OCR)`. Il modello locale risponde poi dalla scansione e i passaggi sono citati con quel nome come qualsiasi altro documento.

Sicurezza: gli argomenti sono passati uno per uno (nessuna shell); gli input diventano assoluti, così un file chiamato `-x.png` non viene letto come opzione; il codice lingua è controllato (`clean_lang`); l'output ha un tetto; un'esecuzione che non finisce viene terminata; gli errori tornano con le ultime righe dell'output di errore del programma; niente parte se non viene chiamato.

## Limiti, detti onestamente

- Il testo è OCR. Non è una trascrizione verificata, e la qualità dell'OCR (e quali lingue funzionano) dipende da LegatorOCR e dai pacchetti lingua Tesseract installati: `legatorocr --doctor` li elenca (sulla macchina di prova mancavano i modelli rumeno e ungherese).
- I numeri di pagina si perdono: l'intera scansione è un solo documento, quindi una citazione indica la scansione, non la pagina.
- La ricerca nella libreria è per parole chiave (BM25), come per ogni documento.
- La finestra mostra la sezione solo quando il programma viene trovato (cercato una volta, in background, all'apertura della finestra; niente viene scaricato o avviato da solo). Senza, compaiono solo un suggerimento con `github.com/chiuta/LegatorOCR` e un pulsante «Cerca di nuovo».
- L'esecuzione è un thread in background con un pulsante Ferma; LegatorOCR non dà alcun dato di avanzamento, quindi la finestra mostra i secondi trascorsi, non una percentuale. Chiudere l'intera app durante un'esecuzione non è gestito in modo particolare.
- L'elenco delle lingue sono le 8 lingue dell'app associate ai codici Tesseract; se una funziona dipende dai dati installati per LegatorOCR (`legatorocr --doctor`). L'app non lo controlla in anticipo: una lingua mancante torna come messaggio d'errore del programma stesso.
- Le immagini di pagina trascinate nella finestra (non i PDF) vanno nel campo della scansione quando il programma viene trovato.

## Provalo

```
cargo test -p companion-core legatorocr
LEGATOROCR_BIN=/path/to/legatorocr LEGATOROCR_TEST_PAGE=/path/to/page.png \
  cargo test -p companion-core real_legatorocr -- --nocapture
```
