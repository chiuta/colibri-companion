# Clean-Room-Prozess

[English](CLEANROOM.md) · [Română](CLEANROOM.ro.md) · [Français](CLEANROOM.fr.md) · [Italiano](CLEANROOM.it.md) · [Español](CLEANROOM.es.md) · [Português](CLEANROOM.pt.md) · [Deutsch](CLEANROOM.de.md) · [Magyar](CLEANROOM.hu.md)

Eine kurze, ehrliche Notiz dazu, wie die Engine in `crates/engine` geschrieben wurde und was „Clean-Room“ hier bedeutet und was nicht.

## Die Regeln, denen wir gefolgt sind

1. **Quellen.** Nur öffentliche Spezifikationen und Veröffentlichungen: die Strukturen der GGUF- und safetensors-Dateien, die Felder von `config.json` und `tokenizer.json` bei Hugging Face, die veröffentlichten Beschreibungen der Blockstrukturen der Quantisierung und die Arbeiten hinter jeder Technik (Attention-Varianten, MoE-Routing, Mamba, RWKV, spekulative Dekodierung, H2O, StreamingLLM, SnapKV, TIES, DARE, Grünlisten-Wasserzeichen, HNSW, BM25 …).
2. **Kein Spicken.** Der Quellcode von colibri, llama.cpp und anderen Inferenz-Engines wurde beim Schreiben der Engine weder gelesen noch kopiert. Wo eine Struktur unklar war, hörte die Arbeit bei der öffentlichen Beschreibung auf, und die Unsicherheit steht im Code und in der Dokumentation.
3. **Orakel statt Nachahmung.** Jeder schnelle Kernel wird gegen eine zuerst geschriebene einfache skalare Fassung getestet; jeder Quantisierer gegen von Hand berechnete Blöcke und Hin-und-zurück-Läufe; geteilte Pfade (Mesh, Streaming, Caches) gegen den Ein-Maschinen-Pfad, bitgleich, wo das versprochen wird.
4. **Nur permissive Lizenzen.** Die Engine hängt von `serde_json` und `memmap2` ab und von nichts sonst; das GPU-Crate fügt `wgpu` hinter einer Option hinzu. Die Lizenzen wurden bei einem früheren Commit mit `cargo-deny` geprüft; spätere Änderungen wurden noch nicht erneut geprüft.
5. **Feindliche Eingaben sind normal.** Jede aus einer Datei gelesene Größe oder jeder Offset wird vor jeder Allokation gegen die echte Dateilänge geprüft; Fuzz-artige Tests füttern abgeschnittene und beschädigte Dateien und müssen einen Fehler erhalten, keinen Absturz.

## Was wir nicht behaupten

- Clean-Room ist ein Prozess, keine rechtliche Garantie. Das Modell, das den Code schrieb, wurde wie jedes Modell mit öffentlichem Text trainiert, der Open-Source-Code enthalten kann, und wir können nicht ausschließen, dass es die oben genannten Projekte gesehen hat. Wir haben sie nicht konsultiert und versucht, das sichtbar zu machen, indem Orakel, Tests und die Quelle jeder Formel im Code bleiben.
- Mehrere Formeln (Mamba, RWKV-6/7, MLA, imatrix- und GGUF-Zusätze) wurden nach veröffentlichten Beschreibungen und der Erinnerung daran geschrieben und nur auf interne Konsistenz geprüft, weil kein echter Checkpoint verfügbar war. Sie können in Details von der Referenz abweichen; betrachte sie als unverifiziert, bis jemand sie an echten Modellen ausführt.
- Kompatibilität mit Dateien anderer Engines ist ein Ziel, keine verifizierte Tatsache (siehe die Tabelle in [ENGINE.de.md](ENGINE.de.md)).

Wenn du eine Stelle findest, an der die Engine fremden Code zu nah wiedergibt, öffne bitte ein Issue; wir schreiben sie neu.
