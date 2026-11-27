# GGUF-fájlok futtatása a corbullal

A colibri csak az általa megvalósított modellcsaládokat importálja (Qwen3.x, DeepSeek, GLM, Mistral-Large, MiMo és még néhány). Más család GGUF-ja, például a Gemma, nem importálható. A Colibri Companion ilyenkor a saját motorját, a **corbult** próbálja, amely közvetlenül olvassa a `.gguf` fájlt.

- A *Modelljeim → GGUF-modellek betöltése mappából* alatt minden fájlnál ott van a *Futtatás corbullal* gomb (akkor látszik, ha a `corbul` az alkalmazás mellett vagy a PATH-ban van; a kiadási csomagok tartalmazzák).
- Ha egy importálás meghiúsul, az üzenet megnevezi a fájlt, a bejelentett architektúrát és a napló utolsó sorait, és a corbul automatikusan megpróbálódik.
- A corbul csak a visszacsatoláson (loopback) hallgat, szabad porton, a kontextuscsomagok polcával az alkalmazás adatmappájában (lásd CONTEXT-PACKS). Leáll, ha leállítod a modellt, másikat indítasz vagy bezárod az alkalmazást. Naplója a *Napló megjelenítése* alatt látszik.
- Korlátok: a corbul nagyon nagy modelleknél lassabb a colibrinél, nem streameli a szakértőket lemezről, és a saját motorja családjait támogatja; ha nem tudja betölteni a fájlt, az üzenet megmondja, miért.
