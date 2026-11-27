# Kontextpakete

Verwendet die Arbeit an einem Text wieder, wenn er in einem anderen Prompt erneut auftaucht (nur corbul-Engine; Clean-Room, eigener Code).

Ein Präfix-Cache greift nur, wenn der gemeinsame Text zuerst steht. Ein *Paket* ist der KV-Cache eines allein berechneten Textes, abgelegt in einem Regal (signierter Snapshot-Speicher mit LRU-Obergrenze).

`compose` reiht Pakete aneinander: Zeilen werden kopiert und ihre Schlüssel um die Positionsdifferenz gedreht (exakt); danach wird der Anfang jedes weiteren Pakets in wachsenden Fenstern neu berechnet, bis die gemessene Änderung unter eine Toleranz fällt. Die Frage wird immer frisch berechnet.

`tolerance = 0` berechnet alles neu und entspricht einem vollständigen Prefill. Modelle mit Gleitfenster, verdrängtem Cache oder Beschleuniger-Cache werden abgelehnt und fallen auf den normalen Prefill zurück.

Bis zur Reparatur nicht exakt: Ein Zustand mit verschobenen Zeilen meldet `is_exact() == false` und wird nie auf die Platte geschrieben.

Grenzen: JustVuggs colibri-Engine kann keine Caches exportieren, daher nur mit corbul. Mit kleinen synthetischen Modellen getestet; die Qualität bei echten Modellen ist noch nicht gemessen.
