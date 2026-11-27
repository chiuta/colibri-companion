# Reducir la caché y reajustar los valores

`corbul kvfit MODEL --text FILE --question Q` mide lo que cuesta conservar solo una fracción de las filas de la caché de un texto (las elige SnapKV), con y sin **reajuste de los valores** que quedan.

Los últimos `--window` tokens del texto se registran como consultas de sondeo. Los valores de las filas conservadas se mueven lo mínimo para que la atención sobre ellas dé, en las sondas, lo que daba la atención sobre todas las filas (mínimos cuadrados regularizados, forma dual, `crates/engine/src/kvfit.rs`). El ajuste nunca puede ser peor que los valores simples *en las sondas*; si ayuda en una pregunta no sondeada depende del modelo y del texto, y solo este comando, ejecutado en tu modelo, te lo dice.

Opciones: `--budget 0.25` (fracción de filas conservadas), `--window 16`, `--ridge 0.01`, `--max-new 32`. Muestra el error del ajuste en las sondas, la mayor diferencia de las primeras puntuaciones respecto a la caché completa, si el primer token coincide y cuántos de los primeros tokens de la respuesta coinciden.

Límites: solo valores (las claves llevan la posición y se quedan; el peso de las filas eliminadas no se modela), las capas de ventana deslizante quedan como están, el ajuste cuesta tiempo proporcional a filas × sondas por cabeza. Ahora unido a los paquetes de contexto: véase SHRINK-MAP.es.md (`--shrink`). Nada medido en un modelo real.

Los paquetes de contexto también pueden guardarse a mitad de tamaño: `corbul packs --pack-kv f16` y `corbul serve --packs-kv f16` (redondeo F16 en cada fila), y comprimirse sin pérdida: `--pack-compress`, `--packs-compress`.

En Colibri Companion: *Documentos → Paquetes de contexto (corbul)* tiene las opciones F16 y compresión (se aplican la próxima vez que corbul arranque) y *Medir la reducción de la caché…*, que ejecuta este comando sobre el modelo que corbul ejecuta (carga el modelo una segunda vez).
