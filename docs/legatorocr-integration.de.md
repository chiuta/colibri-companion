# LegatorOCR-Anbindung

Stand: Bibliothek in `crates/core/src/legatorocr.rs`, über `crates/app/src/ocr_ui.rs` an das Fenster Dokumente angeschlossen (Abschnitt „Gescannte Dateien (LegatorOCR)“). Getestet mit einer echten `legatorocr 0.2.0-pre.65`-Binärdatei an einer einzigen erzeugten Seite mit sauberem synthetischem Text; nie an einem echten Scan eines Buches.

## Was sie tut

LegatorOCR (github.com/chiuta/LegatorOCR, **MIT**) macht aus Scans durchsuchbare PDFs oder Text. Colibri Companion (Apache-2.0) bindet seinen Code nie ein und kopiert ihn nicht; es startet das Programm nur als externen Befehl, was beide Lizenzen erlauben.

1. `locate` / `locate_in`: sucht eine ausführbare Datei `legatorocr` (einen vom Benutzer angegebenen Ordner, den Ordner der App, dann `PATH`). Nur Dateisystem, kein Netzwerk.
2. `version`: führt `legatorocr --version` aus; die Antwort muss mit `legatorocr` beginnen, damit eine fremde Datei gleichen Namens nicht für das Programm gehalten wird.
3. `collect_inputs`: ein Bild oder PDF, oder die Bilder und PDFs direkt in einem Ordner, nach Namen geordnet (höchstens 2000).
4. `read_scan`: führt `legatorocr <Seiten> --ocr --lang <Code> --txt <Datei> --format none --out-dir <Ordner>` aus (für Rumänisch zusätzlich `--clean-text`), über `proc::run_capture` mit Zeitlimit, und liest den Text dann zurück (höchstens 32 MB).
5. `ingest_scan`: fügt diesen Text einer Dokumentensammlung hinzu (`rag::Library::add_document`) unter dem Namen `<Datei oder Ordner> (OCR)`. Das lokale Modell antwortet dann aus dem Scan, und die Textstellen werden wie bei jedem anderen Dokument unter diesem Namen zitiert.

Sicherheit: Argumente werden einzeln übergeben (keine Shell); Eingaben werden absolut gemacht, damit eine Datei namens `-x.png` nicht als Option gelesen wird; der Sprachcode wird geprüft (`clean_lang`); die Ausgabe ist begrenzt; ein Lauf, der nicht endet, wird beendet; Fehler kommen mit den letzten Zeilen der Fehlerausgabe des Programms zurück; nichts startet, wenn es nicht aufgerufen wird.

## Ehrliche Grenzen

- Der Text ist OCR. Er ist keine geprüfte Abschrift, und die OCR-Qualität (und welche Sprachen funktionieren) hängt von LegatorOCR und den installierten Tesseract-Sprachpaketen ab: `legatorocr --doctor` listet sie auf (auf dem Testrechner fehlten die Modelle für Rumänisch und Ungarisch).
- Seitenzahlen gehen verloren: der ganze Scan ist ein Dokument, eine Quellenangabe nennt also den Scan, nicht die Seite.
- Die Suche in der Bibliothek arbeitet mit Stichwörtern (BM25), wie bei jedem Dokument.
- Das Fenster zeigt den Abschnitt nur, wenn das Programm gefunden wird (einmal im Hintergrund gesucht, wenn das Fenster sich öffnet; nichts wird von selbst heruntergeladen oder gestartet). Ohne es erscheinen nur ein Hinweis mit `github.com/chiuta/LegatorOCR` und eine Schaltfläche „Erneut suchen“.
- Der Lauf ist ein Hintergrund-Thread mit einer Schaltfläche Stopp; LegatorOCR liefert keine Fortschrittszahl, das Fenster zeigt daher vergangene Sekunden, keinen Prozentwert. Das Schließen der ganzen App während eines Laufs wird nicht besonders behandelt.
- Die Sprachliste sind die 8 Sprachen der App, abgebildet auf Tesseract-Codes; ob eine funktioniert, hängt von den für LegatorOCR installierten Daten ab (`legatorocr --doctor`). Die App prüft das nicht vorher: eine fehlende Sprache kommt als Fehlermeldung des Programms selbst zurück.
- In das Fenster gezogene Seitenbilder (keine PDFs) landen im Scan-Feld, wenn das Programm gefunden wird.

## Ausprobieren

```
cargo test -p companion-core legatorocr
LEGATOROCR_BIN=/path/to/legatorocr LEGATOROCR_TEST_PAGE=/path/to/page.png \
  cargo test -p companion-core real_legatorocr -- --nocapture
```
