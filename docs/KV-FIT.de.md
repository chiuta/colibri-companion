# Den Cache verkleinern und die Werte nachpassen

`corbul kvfit MODEL --text FILE --question Q` misst, was es kostet, nur einen Teil der Cache-Zeilen eines Textes zu behalten (SnapKV wählt sie), mit und ohne **Nachpassen der Werte**, die bleiben.

Die letzten `--window` Tokens des Textes werden als Sondenanfragen aufgezeichnet. Die Werte der behaltenen Zeilen werden dann so wenig wie möglich verschoben, dass die Attention über sie auf den Sonden ergibt, was die Attention über alle Zeilen ergab (regularisierte kleinste Quadrate in der dualen Form, `crates/engine/src/kvfit.rs`). Die Anpassung kann *auf den Sonden* nie schlechter sein als die einfachen Werte; ob sie bei einer nicht geprobten Frage hilft, hängt von Modell und Text ab, und nur dieser Befehl, auf deinem Modell ausgeführt, sagt es dir.

Optionen: `--budget 0.25` (Anteil behaltener Zeilen), `--window 16`, `--ridge 0.01`, `--max-new 32`. Ausgegeben werden der Anpassungsfehler auf den Sonden, die größte Abweichung der ersten Scores gegenüber dem vollen Cache, ob das erste Token übereinstimmt und wie viele der ersten Antwort-Tokens übereinstimmen.

Grenzen: nur Werte (Schlüssel tragen die Position und bleiben; das Gewicht entfernter Zeilen wird nicht modelliert), Schichten mit Gleitfenster bleiben unverändert, die Anpassung kostet Zeit proportional zu Zeilen × Sonden je Kopf. Noch nicht mit Kontextpaketen verbunden (ein verkleinertes Paket lässt sich nicht einspleißen). Auf einem echten Modell noch nichts gemessen.

Kontextpakete lassen sich auch halb so groß ablegen: `corbul packs --pack-kv f16` und `corbul serve --packs-kv f16` (F16-Rundung in jeder Zeile) und verlustfrei komprimieren: `--pack-compress`, `--packs-compress`.
