# LegatorOCR integration

Status: library in `crates/core/src/legatorocr.rs`, wired into the Documents window by
`crates/app/src/ocr_ui.rs` (section "Scanned files (LegatorOCR)"). Tested against a real `legatorocr 0.2.0-pre.65` binary on one generated page of
clean synthetic text; never on a real scan of a book.

## What it does

LegatorOCR (github.com/chiuta/LegatorOCR, **MIT**) turns scans into searchable PDF or text.
Colibri Companion (Apache-2.0) never links or copies its code; it only runs the program as an
external command, which both licences allow.

1. `locate` / `locate_in`: look for a `legatorocr` executable (a folder given by the user, the
   folder of the app, then `PATH`). File system only, no network.
2. `version`: runs `legatorocr --version`; the answer must start with `legatorocr`, so an
   unrelated file of the same name is not taken for it.
3. `collect_inputs`: one image or PDF, or the images and PDFs directly inside a folder, in name
   order (at most 2000).
4. `read_scan`: runs `legatorocr <pages> --ocr --lang <code> --txt <file> --format none
   --out-dir <dir>` (plus `--clean-text` for Romanian) through `proc::run_capture` with a time
   limit, then reads the text back (at most 32 MB).
5. `ingest_scan`: adds that text to a document collection (`rag::Library::add_document`) under the
   name `<file or folder> (OCR)`. The local model then answers from the scan, and the passages are
   cited by that name like any other document.

Safety: arguments are passed one by one (no shell); inputs are made absolute so a file called
`-x.png` cannot be read as an option; the language code is checked (`clean_lang`); output is
capped; a run that does not end is killed; failures are returned with the last lines of the
program's error output; nothing starts unless called.

## Honest limits

- The text is OCR. It is not a verified transcription, and OCR quality (and which languages work)
  depends on LegatorOCR and the Tesseract language packs installed: `legatorocr --doctor` lists
  them (on the test machine the Romanian and Hungarian models were missing).
- Page numbers are lost: the whole scan is one document, so a citation names the scan, not the page.
- The library search is keyword based (BM25), as for every document.
- The window shows the section only when the program is found (searched once, in the
  background, when the window opens; nothing is downloaded or started by itself). Without it only
  a hint with `github.com/chiuta/LegatorOCR` and a "Look again" button appear.
- The run is a background thread with a Stop button; LegatorOCR gives no progress figure, so the
  window shows elapsed seconds, not a percentage. Closing the whole app during a run is not
  handled specially.
- The language list is the 8 app languages mapped to Tesseract codes; whether one works depends
  on the data installed for LegatorOCR (`legatorocr --doctor`). The app does not check this
  beforehand: a missing language comes back as the program's own error message.
- Dropped page images (not PDFs) go to the scan field when the program is found.

## Try it

```
cargo test -p companion-core legatorocr
LEGATOROCR_BIN=/path/to/legatorocr LEGATOROCR_TEST_PAGE=/path/to/page.png \
  cargo test -p companion-core real_legatorocr -- --nocapture
```
