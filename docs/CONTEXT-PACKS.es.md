# Paquetes de contexto

Reutiliza el trabajo hecho sobre un texto cuando reaparece en otro prompt (solo motor corbul; clean-room, código original).

Una caché de prefijo solo acierta si el texto común va primero. Un *paquete* es la caché KV de un texto calculado solo y guardado en un estante (almacén de instantáneas firmado, con tope LRU).

`compose` pone los paquetes uno tras otro: las filas se copian y sus claves se rotan por la diferencia de posición (exacto); luego el inicio de cada paquete siguiente se recalcula en ventanas crecientes hasta que el cambio medido baja de una tolerancia. La pregunta siempre se calcula de nuevo.

`tolerance = 0` recalcula todo e iguala un prefill completo. Los modelos con ventana deslizante, caché expulsada o en acelerador se rechazan y vuelven al prefill normal.

No exacto hasta repararse: un estado con filas movidas informa `is_exact() == false` y nunca se escribe en disco.

Límites: el motor colibri de JustVugg no puede exportar cachés, así que solo funciona con corbul. Probado con modelos sintéticos pequeños; la calidad con modelos reales aún no se ha medido.

## Línea de comandos

`corbul packs MODEL --session DIR --pack ARCHIVO… --question TEXTO [--tolerance 0.02] [--probe 8] [--max-new 64] [--check]` guarda cada archivo una sola vez y luego responde con los paquetes guardados, en cualquier orden. `--check` también ejecuta un prefill completo y muestra la diferencia de puntuación y el tiempo de cada uno.
