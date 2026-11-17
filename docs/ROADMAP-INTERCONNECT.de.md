# Fahrplan der Vernetzung: vom Chatfenster zum fähigen Assistenten

Ziel: das lokale Modell mit allem verbinden, was vervielfacht, was es kann, während der Mensch die Kontrolle behält. Jeder Schritt hält drei Regeln ein: **zuerst lokal, nichts verlässt den Rechner ohne ein Ja, jede Aktion lässt sich ansehen und rückgängig machen.**

## 0. Der Sicherheitsrahmen (zuerst gebaut, von allem Folgenden benutzt)
- **Fähigkeitsstufen** (siehe `platforms.rs`): Lesen → Suchen → Schreiben → Benachrichtigen → Handeln. Eine Verbindung wird auf einer Stufe angelegt und kann sie nie überschreiten.
- **Freigabekarte** für jedes Schreiben, Senden und Handeln: zeigt genau, was geschehen wird, mit *Einmal erlauben / Für diese Automatisierung erlauben / Ablehnen*. Lesen braucht keine Karte.
- **Nicht vertrauenswürdige Inhalte** (Webseiten, Tickets, E-Mails, Dokumente) werden eingerahmt und als Daten gekennzeichnet. Ein Werkzeugaufruf, der auf nicht vertrauenswürdigen Inhalt folgt, fragt immer erneut. Das ist die wichtigste Abwehr gegen Prompt-Injection.
- **Budgets und Bremsen**: Schritte, Modellaufrufe, Minuten und Bytes pro Lauf; eine Taste *Alles stoppen*.
- **Prüfprotokoll** jeder Anfrage, die den Rechner verlassen hat, und jeder ausgeführten Aktion, in einer einfachen Datei.
- **Geheimnisse**: Tokens liegen standardmäßig im Arbeitsspeicher; sie zu speichern ist eine ausdrückliche Entscheidung.

## 1. Werkzeuge für das Modell (der Vervielfacher)
Colibri bietet auf den meisten Engines Werkzeugaufrufe an. Das Programm gibt dem Modell einen kleinen, ehrlichen Werkzeugkasten: Webseite, Suche, Repository-Datei, Ordnersuche, Taschenrechner, Python in einer Sandbox, Tabellenabfrage, Notizsuche, Datei speichern, Webhook. Das Modell fragt an, das Programm prüft Stufe und Freigabe, führt aus und gibt das Ergebnis als Daten zurück.

## 2. MCP: ein Stecker für Tausende Werkzeuge
Das Model Context Protocol als Client sprechen (und `coli mcp` als Server behalten). Jeder MCP-Server (Dateien, Datenbanken, Browser, Kalender, Designwerkzeuge, Heimgeräte) wird unter denselben Freigaberegeln verfügbar. Das ist der größte Sprung an Reichweite für den geringsten Code.

## 3. Gedächtnis und Wissen, die mit dir wachsen
- Lokaler Index (jetzt BM25, später Embeddings) über Ordner, Notizen, gespeicherte Chats, Code, Mailarchive.
- Antworten zitieren die verwendete Textstelle; „zeig mir, wo“ öffnet sie.
- Offline-Bibliotheken: Wikipedia über Kiwix, arXiv- und Wikidata-Abzüge, Handbücher, Gesetze.
- Die *Konstellation*: Erinnerungen als verbundene Sterne (Tags, Personen, Projekte), die du sehen und bearbeiten kannst.

## 4. Sinne und Stimme (alles lokal)
- Hören: whisper.cpp für Diktat und Besprechungsprotokolle.
- Sprechen: Piper oder die Systemstimme, je nach Sprache der Oberfläche.
- Sehen: Modelle, die Bilder annehmen (colibri listet, welche), dazu OCR (Tesseract) für Scans und Fotos von Papier, dazu Kamera oder Bildschirmfoto auf Abruf.
- Erzeugen: Qwen-Image über colibri, ComfyUI oder Stable Diffusion für Bilder; Audio- und Videowerkzeuge, sobald sie erscheinen.

## 5. Viele Köpfe
- **Duell**: blindes A/B mit Elo, um herauszufinden, welches Modell *für dich* das beste ist.
- **Rat**: mehrere Modelle antworten, ein Richtermodell führt zusammen und markiert Widersprüche.
- **Weiche**: ein kleines schnelles Modell für leichte Fragen, das große für schwere, gewählt von einem Entscheidungsmodell (colibri liefert Laya, GLiNER2.5-Decide, Clef).
- **Spezialisten**: Code, Übersetzung, Bild, jeweils automatisch gewählt, jeweils austauschbar.

## 6. Genauigkeit: das Modell prüft sich selbst
- Den Python-Code, den es schreibt, in einer Sandbox ausführen und die echte Ausgabe zeigen.
- Unit-Tests für den erzeugten Code, vor dem Anbieten ausgeführt.
- Selbstkonsistenz (zweimal fragen, vergleichen), Quellenpflicht für Fakten, ein Schalter „zeig deine Quellen“ und ein kleiner Auswertungsrahmen mit Konfidenzintervallen (wie bei LocalMind).

## 7. Agenten und Automatisierungen
Gespeicherte Schrittketten (lesen, fragen, prüfen, speichern, benachrichtigen) mit Auslösern: von Hand, nach Zeitplan, wenn sich ein Ordner ändert, wenn ein Webhook eintrifft. Vorlagen: Morgenübersicht, Entwurf zur Posteingangssortierung, Repository-Beobachter, Besprechungsnotizen, Recherchebericht, Wochenrückblick. Jeder Schritt zeigt seine Stufe.

## 8. Geräte und die physische Welt (zuletzt und mit größter Vorsicht)
Home Assistant, MQTT, serielle Geräte, 3D-Drucker, Kameras. Standardmäßig nur lesend; Aktionen über eine Erlaubnisliste; niemals Schlösser, Alarme oder irgendetwas Sicherheitskritisches.

## 9. Mehrere Rechner, ein Gehirn
Ein Modell im Heimnetz teilen (ein starker PC bedient, ein Laptop nutzt), Prompts, Gedächtnis und Einstellungen zwischen Rechnern abgleichen (verschlüsselt, Peer-to-Peer, ohne Cloud) und die Arbeit auf mehrere GPUs verteilen, wo die Engine es unterstützt.

## 10. Mühelos machen
Eine Einrichtungshilfe beim ersten Start, die ein Modell passend zur Hardware wählt, ein *einfacher Modus* mit fünf Schaltflächen, Ein-Klick-Verbindungen aus dem Katalog, Vorschläge „was kann ich damit tun?“, ein sichtbares Feld „was das Modell gerade sehen kann“ und alles in acht Sprachen.

## Reihenfolge der Arbeit
1. Ordner und große Dateien, Prompt-Bibliothek, Antwortstile (erledigt).
2. Online-Zugang, GitHub/GitLab/Gitea/Webseiten, der Katalog (in Arbeit).
3. Freigabekarten, Prüfprotokoll, Budgets (der Sicherheitsrahmen).
4. Automatisierungen mit Webhooks und Benachrichtigungen.
5. Duell, Rat, Weiche.
6. Werkzeuge für das Modell, dann MCP-Client.
7. Index und Gedächtnis, Stimme und OCR.
8. Geräte, mehrere Rechner.

Was *nicht* versprochen wird: Nichts davon wurde bisher mit echten colibri-Engines mit Werkzeugaufrufen getestet, und jeder externe Dienst ändert seine API mit der Zeit; im Katalog als „geplant“ markierte Einträge sind Pläne, keine Funktionen.
