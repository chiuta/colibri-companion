# Eine Version veröffentlichen

Wie eine Version entsteht (für die Betreuenden)

1. **Version.** Ändere `version` in der obersten `Cargo.toml` und `TAG` in `index.html` auf die neue Nummer (zum Beispiel `0.1.0-alpha.12` und `v0.1.0-alpha.12`), führe `cargo update -w --offline` aus, füge `docs/releases/<tag>.md` hinzu und committe und pushe nach `main`.
2. **Bauen.** Warte, bis die Läufe `build` und `engine-ci` für diesen Commit grün sind. Starte dann den Workflow `build` von Hand (Actions, Run workflow) und gib das Tag ein, zum Beispiel `v0.1.0-alpha.12`. Er baut alle Plattformen und legt ein **Entwurf**-Release an.
3. **Veröffentlichen.** Öffne den Entwurf, füge `docs/releases/<tag>.md` in die Beschreibung ein, setze Pre-release und drücke Publish. **Ändere den Tag-Namen im Entwurf nicht**: Die Dateien wurden mit der Version aus `Cargo.toml` gebaut, und ein anderes Tag lässt das Programm eine Version melden und das Release eine andere (und die Update-Prüfung bietet endlos ein Update an). Der Workflow lehnt ein Tag ab, das schon auf einen anderen Commit zeigt.

## Falls eine Nummer versehentlich veröffentlicht wurde

Verwende sie nicht wieder. Lösche dieses Release und sein Tag von Hand und veröffentliche die nächste Nummer.

## Das Release nie von Hand anlegen

Der `build`-Workflow legt den Entwurf selbst an, mit allen Dateien im Anhang. Ein von Hand angelegtes Release mit demselben Tag (auch als Entwurf) ist leer und lässt den letzten Schritt scheitern. Falls es passiert: das leere Release löschen (den Tag behalten) und beim `build`-Lauf **Re-run failed jobs** benutzen.
