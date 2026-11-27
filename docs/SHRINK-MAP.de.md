# Verkleinerte Pakete und Cache-Abbildungen zwischen Modellen

Zwei Erweiterungen der Kontextpakete ([CONTEXT-PACKS.de.md](CONTEXT-PACKS.de.md), nur corbul-Engine; Clean-Room, eigener Code).

## Verkleinerte Pakete

Passen die Texte und die Frage zusammen nicht in das Fenster des Modells, werden die Pakete abgelehnt. Mit Verkleinerung werden sie gekürzt: `corbul packs --shrink auto` (nur so viel wie nötig) oder `--shrink R` (behält immer etwa den Anteil `R`, 0,1 bis 1 der Zeilen jedes Pakets). Dasselbe bewirkt das Feld `"shrink": "auto"` (oder eine Zahl) von `/v1/packs/complete`; die Antwort listet, was verkleinert wurde (`"shrunk"`).

Wie: Die letzten `--shrink-window` Tokens eines Pakets (Standard 16) werden erneut gelesen und stimmen über die zu behaltenden Zeilen ab (SnapKV); die ersten Zeilen bleiben immer (dort sammelt sich die Aufmerksamkeit); die verbleibenden Werte werden neu angepasst ([KV-FIT.de.md](KV-FIT.de.md); `--no-refit` schaltet das ab). Die behaltenen Schlüssel werden dann auf die Positionen ihrer neuen Plätze gedreht, sodass das Paket *genau der Cache seines gekürzten Textes* wird: Es lässt sich wie jedes Paket einspleißen und reparieren, und bei vollständiger Reparatur gleicht die Antwort einem einfachen Vorbefüllen des gekürzten Textes. Die ersten `--shrink-skip` Texte (Standard 1, meist die Anweisungen) bleiben ganz. Verkleinert wird beim Zusammensetzen; das Regal behält die vollen Pakete.

In Colibri Companion: *Dokumente → Kontextpakete (corbul)* → *Zu lange Texte verkleinern*.

Grenzen: Information geht verloren, wie viel, hängt von Modell und Text ab (messen mit `corbul kvfit` oder dem Fenster *Measure cache shrinking…*); Modelle mit Gleitfenster-Schichten, Q8-Caches und Beschleuniger-Caches werden abgelehnt; passen die Texte selbst ohne alle verkleinerbaren Texte nicht, schlägt die Anfrage fehl. Bisher nur an winzigen Zufallsmodellen getestet.

## Cache-Abbildungen zwischen verschiedenen Modellen

Ein von einem Modell geschriebenes Paket nützt einem anderen nichts. Teilen beide den Tokenizer, lassen sich ihre Cache-Zeilen für *dieselben Tokens* aufeinander beziehen: `corbul map-fit SRC DST --text DATEI --out KARTE` liest den Text mit beiden Modellen und passt je Schicht eine lineare Abbildung an (Ridge-Kleinste-Quadrate, mit Mittelwert), von den Schlüssel- und Wertzeilen von SRC auf die von DST. Schlüssel werden ohne Position verglichen (auf 0 zurückgedreht und beim Import wieder gedreht), daher dürfen Rotationstabellen und Kopfgrößen abweichen; Schichten werden nach relativer Tiefe gepaart. Ausgegeben wird der Fehler auf nicht angepassten Zeilen neben dem des bloßen Mittelwerts, mit Warnung, wenn die Abbildung nicht besser ist.

`corbul map-check SRC DST --map KARTE --text DATEI --question Q` teilt den Text in zwei von SRC geschriebene Pakete, bringt sie über die Abbildung in DST und antwortet auf drei Arten: volles Vorbefüllen durch DST (die Wahrheit), abgebildeter Cache mit der üblichen Reparatur (`--tolerance`, Standard 0,02) und abgebildeter Cache ohne Reparatur. Ausgegeben werden die neu berechneten Zeilen und der Abstand jeder Antwort zur Wahrheit.

Eine abgebildete Zeile ist nur eine Vermutung: Importierte Zeilen gelten als ungefähr, die Reparatur berechnet so viele neu, wie nötig sind (alle bei Toleranz 0, was einem einfachen Vorbefüllen entspricht). Eine schlechte Abbildung kostet also Zeit, nie Richtigkeit. Einstieg in der Bibliothek: `ctxpack::compose_mapped`.

Grenzen: Die Tokenizer müssen dieselben IDs liefern (wird geprüft, sonst wird nichts angepasst); die Abbildung gilt für ein einziges Modellpaar (Fingerabdrücke werden geprüft); ihre Größe ist `kd × kd' × Schichten × 8` Bytes; die Anpassung hält die Kalibrierzeilen im Speicher (`--rows`, Standard 1024). Eine lineare Abbildung zwischen den Caches nicht verwandter Modelle kann schlecht sein; ob sie sich für Ihr Paar lohnt, zeigt nur eine Messung. Noch nicht im Server oder im Companion, und auf einem echten Modellpaar wurde nichts gemessen.
