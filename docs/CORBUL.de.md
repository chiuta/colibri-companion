# GGUF-Dateien mit corbul ausführen

Colibri importiert nur die Modellfamilien, die es implementiert (Qwen3.x, DeepSeek, GLM, Mistral-Large, MiMo und einige weitere). Ein GGUF einer anderen Familie, etwa Gemma, lässt sich nicht importieren. Colibri Companion versucht dann die eigene Engine **corbul**, die die `.gguf`-Datei direkt liest.

- Unter *Meine Modelle → GGUF-Modelle aus einem Ordner laden* hat jede Datei die Schaltfläche *Mit corbul ausführen* (sichtbar, wenn `corbul` neben der App oder im PATH liegt; die Release-Archive enthalten es).
- Scheitert ein Import, nennt die Meldung die Datei, ihre erklärte Architektur und die letzten Logzeilen; corbul wird automatisch versucht.
- corbul lauscht nur auf dem Loopback, an einem freien Port, mit dem Regal der Kontextpakete im Datenordner der App (siehe CONTEXT-PACKS). Es endet, wenn du das Modell stoppst, ein anderes startest oder die App schließt. Das Log zeigt *Protokoll anzeigen*.
- Grenzen: corbul ist bei sehr großen Modellen langsamer als colibri, streamt keine Experten von der Platte und unterstützt die Familien der eigenen Engine; kann es die Datei nicht laden, sagt die Meldung warum.
