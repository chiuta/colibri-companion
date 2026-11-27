# Paquetes reducidos y mapas de caché entre modelos

Dos extensiones de los paquetes de contexto ([CONTEXT-PACKS.es.md](CONTEXT-PACKS.es.md), solo motor corbul; clean-room, código original).

## Paquetes reducidos

Cuando los textos y la pregunta juntos no caben en la ventana del modelo, los paquetes se rechazan. Con la reducción se acortan: `corbul packs --shrink auto` (solo lo necesario) o `--shrink R` (conserva siempre aproximadamente la fracción `R`, de 0,1 a 1 de las filas de cada paquete). Lo mismo hace el campo `"shrink": "auto"` (o un número) de `/v1/packs/complete`; la respuesta lista lo reducido (`"shrunk"`).

Cómo: los últimos `--shrink-window` tokens de un paquete (16 por defecto) se releen y votan las filas que se conservan (SnapKV); las primeras filas se conservan siempre (ahí se acumula la atención); los valores que quedan se reajustan ([KV-FIT.es.md](KV-FIT.es.md); `--no-refit` lo desactiva). Las claves conservadas se rotan luego a las posiciones de sus nuevos huecos, así que el paquete pasa a ser *exactamente la caché de su texto acortado*: se empalma y se repara como cualquier paquete y, con reparación completa, la respuesta es igual a un simple prellenado del texto acortado. Los primeros `--shrink-skip` textos (1 por defecto, normalmente las instrucciones) quedan enteros. La reducción ocurre al juntar los paquetes; el estante guarda los paquetes enteros.

En Colibri Companion: *Documentos → Paquetes de contexto (corbul)* → *Reducir los textos demasiado largos*.

Límites: se pierde información, cuánta depende del modelo y del texto (mídelo con `corbul kvfit` o la ventana *Measure cache shrinking…*); los modelos con capas de ventana deslizante, las cachés Q8 y las de acelerador se rechazan; si los textos no caben ni quitando todos los textos reducibles, la petición falla. Probado hasta ahora solo en modelos diminutos aleatorios.

## Mapas de caché entre modelos distintos

Un paquete escrito por un modelo no le sirve a otro. Si ambos comparten tokenizador, sus filas de caché para los *mismos tokens* se pueden relacionar: `corbul map-fit SRC DST --text ARCHIVO --out MAPA` lee el texto con ambos modelos y ajusta, por capa, un mapa lineal (mínimos cuadrados con ridge, con la media) de las filas de claves y valores de SRC a las de DST. Las claves se comparan sin su posición (devueltas a 0 y rotadas de nuevo al importar), así que las tablas rotatorias y los tamaños de cabeza pueden diferir; las capas se emparejan por profundidad relativa. Muestra el error en las filas apartadas del ajuste junto al de adivinar la media y avisa si el mapa no es mejor.

`corbul map-check SRC DST --map MAPA --text ARCHIVO --question Q` divide el texto en dos paquetes escritos por SRC, los lleva a DST mediante el mapa y responde de tres maneras: el prellenado completo de DST (la verdad), la caché mapeada con la reparación habitual (`--tolerance`, 0,02 por defecto) y la caché mapeada sin reparación. Muestra las filas recalculadas y cuánto se aleja cada respuesta de la verdad.

Una fila mapeada es solo una conjetura: las filas importadas se marcan como aproximadas y la reparación recalcula las que haga falta (todas con tolerancia 0, lo que iguala un prellenado simple). Un mal mapa cuesta, pues, tiempo, nunca corrección. Punto de entrada de la biblioteca: `ctxpack::compose_mapped`.

Límites: los tokenizadores deben dar los mismos identificadores (se comprueba; si no, no se ajusta nada); el mapa vale para un solo par de modelos (se comprueban las huellas); su tamaño es `kd × kd' × capas × 8` bytes; el ajuste guarda en memoria las filas de calibración (`--rows`, 1024 por defecto). Un mapa lineal entre cachés de modelos sin relación puede ser pobre; si compensa para tu par solo se sabe midiendo. Aún no está en el servidor ni en el Companion y no se ha medido nada en un par real de modelos.
