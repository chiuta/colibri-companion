# Änderungen für Sicherheit und Robustheit

[English](CHANGES-SECURITY.md) · [Română](CHANGES-SECURITY.ro.md) · [Français](CHANGES-SECURITY.fr.md) · [Italiano](CHANGES-SECURITY.it.md) · [Español](CHANGES-SECURITY.es.md) · [Português](CHANGES-SECURITY.pt.md) · [Deutsch](CHANGES-SECURITY.de.md) · [Magyar](CHANGES-SECURITY.hu.md)

Weitere Dokumente: [ENGINE.md](ENGINE.de.md) · [QUICKSTART.md](QUICKSTART.de.md)

Diese Seite listet auf, was die adversariale Prüfung der Engine und von `corbul` gefunden hat und was dagegen getan wurde, damit du beurteilen kannst, wie weit du einer Funktion trauen kannst. Sie ist eine Liste von Korrekturen und Grenzen, keine Sicherheitszertifizierung: nichts davon wurde von Dritten geprüft.

## Server

- **Seitenübergreifende Anfragen.** Zustandsändernde Aufrufe prüfen `Origin` und verlangen einen JSON-`Content-Type`, damit eine Webseite, die du besuchst, deinen Browser nicht dazu bringen kann, an ein lokales `corbul serve` zu senden. Browser sind nur von den Origins erlaubt, die du mit `--cors` aufführst.
- **Admin-Routen.** `/api/create`, `/api/delete`, `/models/load` und `/models/unload` brauchen das Master-Token. Inhaber von API-Schlüsseln dürfen sie nur nutzen, wenn du den Server mit `--allow-admin` startest.
- **Geheimnisse nicht auf der Kommandozeile.** `--token-file`, `--key-file`, `--receipt-key-file` und die Variablen `CORBUL_TOKEN` und `CORBUL_MESH_KEY` halten Geheimnisse aus der Prozessliste fern. Die einfachen `--token`, `--key` und `--receipt-key` funktionieren weiterhin und sind für andere Benutzer der Maschine sichtbar.
- **Ressourcengrenzen.** `--max-kv` lehnt Anfragen ab, deren KV-Cache zu groß würde (Standard 8g); `--max-heavy` begrenzt gleichzeitige Tokenize-, Template- und Verify-Anfragen (Standard 4); die Ratenbegrenzungen aus `--keys-file` gelten für jede Route, die Rechenaufwand kostet; der Decoder für Chunked-Bodys arbeitet linear; `POST /v1/quotes/verify` hat Obergrenzen für Text- und Quellengrößen.
- **Protokolle.** Der Prompt-Text wird nicht protokolliert, außer `--log-prompts` ist gesetzt; `--no-log` schaltet das Protokoll aus.
- **Weboberfläche.** Die Seite selbst ist öffentlich (sie enthält keine Geheimnisse); jeder Aufruf, den sie macht, braucht das Token.
- **Verbleibende Grenze:** Der Server spricht reines HTTP. Binde ihn an `127.0.0.1` (der Standard) oder setze TLS davor.

## Mesh

- Gegenseitiger HMAC-SHA256-Handshake mit einem gemeinsamen Schlüssel von mindestens 8 Zeichen, ein MAC auf jedem Frame, Obergrenzen für Handshake- und Frame-Größen.
- Die UDP-Erkennung ist opt-in (`--discovery`, `--discovery-port`), und Antworten sind nicht größer als die Anfrage, sodass ein Worker nicht als Verstärker missbraucht werden kann.
- **Verbleibende Grenze:** Der Verkehr ist authentifiziert, aber nicht verschlüsselt; der Schlüssel wird von allen im Mesh geteilt; nur auf Loopback getestet.

## Dateien, Downloads und Merges

- Das GGUF-Parsing erhält die tatsächliche Dateilänge (`parse_with_len`); der Aufbau des Tokenizers nutzt einen Trie mit einer Obergrenze für die Stückgröße; abgeschnittene und beschädigte Dateien liefern in den adversarialen Tests Fehler.
- Downloads: Bindung an einen Commit-SHA, Größenobergrenzen, Ablehnung von Weiterleitungen auf reines http, anonymer `wget`-Ersatz (das Token wird ihm nie übergeben), SHA-256-Prüfungen.
- Merges folgen keinen Symlinks und weigern sich, eine Ausgabe zu überschreiben, außer `--overwrite` ist gesetzt; Formen und Bereiche werden vor dem Schreiben geprüft.
- Die Bilddecoder erzwingen `Limits` vor dem Allokieren und begrenzen die Deflate-Expansion; der WAV-Leser hat Grenzen für feindliche Dateien.

## Dekodieren und Verifikation

- Das Dekodieren per Grammatik, regulärem Ausdruck und JSON-Schema hat Arbeitsbudgets; die Zitatsperre arbeitet linear mit Obergrenzen und meldet `limit_hit`, wenn eine Obergrenze sie stoppt.
- Quittungen haben Größenobergrenzen; Snapshots können mit einem HMAC mit Schlüssel versiegelt werden; Quittungen und Snapshots nutzen SHA-256-Ketten.
- In der Prüfung gefundene Korrekturen der Korrektheit: veraltete Wahrscheinlichkeiten nach Top-k/Top-p (jetzt renormalisiert), der Alias `typ_p`, Dezimalkommas bei der Zahlenverarbeitung, Lookahead-Grenzen jenseits des Kontexts, der Gruppenscore von DeepSeek-V2, Exaktheit quantisierter Batches (`exact_batch`).

## Nicht getan oder nicht verifiziert

- Kein Audit durch Dritte, keine Fuzzing-Kampagne über die adversarialen Tests des Repositorys hinaus, kein Penetrationstest von `corbul serve`.
- `ToolStream::push` kann seinen Puffer wiederholt neu durchsuchen (quadratisch in der Ausgabelänge bis `--max-tokens`).
- Der `GrammarCache` existiert, wird von der CLI aber noch nicht verwendet.
- Die Lizenzen der Abhängigkeiten wurden zuletzt mit `cargo-deny` bei Commit b496ba5 geprüft.
- Die CI-Workflows sind nie gelaufen; die Codepfade für ARM, macOS und Windows wurden hier nicht ausgeführt.
