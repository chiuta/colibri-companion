# Qué tiene de distinto este motor

[English](INNOVATIONS.md) · [Română](INNOVATIONS.ro.md) · [Français](INNOVATIONS.fr.md) · [Italiano](INNOVATIONS.it.md) · [Español](INNOVATIONS.es.md) · [Português](INNOVATIONS.pt.md) · [Deutsch](INNOVATIONS.de.md) · [Magyar](INNOVATIONS.hu.md)

Notas en lenguaje llano sobre las funciones que la mayoría de los demás motores locales no tienen. Cada una dice qué hace, cómo se llega a ella y qué no puede hacer. Todas se probaron solo con modelos sintéticos; los autores no han ejecutado ninguna con un checkpoint real. «En `corbul`» significa que hoy existe un comando o una opción (`corbul help`); «solo biblioteca» significa código Rust sin comando. Véanse [ENGINE.es.md](ENGINE.es.md) y [QUICKSTART.es.md](QUICKSTART.es.md).

## Integridad, hardware y memoria

### Malla en LAN

Módulo `mesh`. En `corbul`: `worker`, `generate --workers`, `--discover`, `--key`, `--wire-f16`. Un modelo demasiado grande para una máquina se corta en rangos de capas, uno por ordenador. Entre ellos solo viaja el estado oculto. Las máquinas prueban que comparten una clave (HMAC-SHA256 mutuo), cada trama lleva un MAC, los tamaños del handshake están acotados, un worker caído se descarta y sus capas se redistribuyen. Los logits f32 densos coinciden con los de una sola máquina. El descubrimiento UDP es opcional: un worker solo responde a las sondas si se inició con `--discovery` o `--discovery-port` (`--no-discovery` lo fuerza a apagado), y el `--discover` del coordinador emite una sonda y espera las respuestas de los workers que tengan el mismo modelo. *No puede:* cifrar (usa una LAN de confianza o una VPN); solo se probó en loopback; no se midió la velocidad frente a una sola máquina.

### Streaming de capas

Módulo `stream_layers`. En `corbul`: `--stream-layers`, `--ram-budget`, `--stream-window`, `--disk-mbps` (para estimaciones). Ejecuta un modelo denso mayor que la RAM manteniendo en memoria una pequeña ventana de capas y precargando en un hilo de fondo. Los resultados coinciden con los del modelo en RAM. *No puede:* evitar que cada token lea todas las capas, así que la velocidad queda limitada por la del disco; solo capas densas (los modelos de mezcla de expertos transmiten en cambio sus expertos enrutados, con `--ram-budget`).

### Citas bloqueadas a la cita textual

Módulo `quotelock`. En `corbul`: `--sources FILE` (repetible), `--no-lock`, `--citations FILE`; el servidor acepta `sources` y ofrece `POST /v1/quotes/verify`. Durante la decodificación, el texto entre « y » debe ser un pasaje literal de una fuente (tolerante a mayúsculas, espacios, estilos de comillas y diacríticos rumanos); `[S3]«` limita una cita a la fuente 3. Después, un verificador clasifica cada cita (literal, fuente equivocada, truncada, alterada, fusionada, sin respaldo) y puntúa las frases que quedan fuera de las citas; el informe indica cuándo se alcanzó un límite de trabajo (`limit_hit`). *No puede:* hacer verdadero el texto que rodea a las citas; las respuestas en streaming no llevan bloque de citas.

### Planificador, gobernador y medidor de energía

Módulo `governor` (y `plan`, `hw`). En `corbul`: `plan`, `doctor`, `--governor`, `--no-governor`, `--verbose`, `--calibrate`. Antes de cargar, dice si el modelo cabe, se transmite, va lento o no puede ejecutarse, con una velocidad estimada. Durante la ejecución, el gobernador lee temperatura, memoria y batería (Linux) y aconseja hilos y pausas sin oscilar. *No puede:* dar velocidades que no sean estimaciones (`--calibrate` mide tu máquina durante un segundo aproximadamente); la energía es una suposición salvo que se pueda leer un contador RAPL; la intensidad de carbono (400 g/kWh) es un valor provisional.

### Políticas de caché KV e instantáneas de sesión

Módulos `kvpolicy`, `snapshot`. En `corbul`: `--kv-policy streaming|h2o|snapkv`, `--kv-budget`, `--kv-sinks`, `--session DIR`, `--session-cap`, `--compress`. Mantienen las conversaciones largas dentro de un presupuesto de memoria (sumideros de atención, heavy hitters, compresión del prompt) y guardan o reanudan una conversación desde un archivo con suma de comprobación, reutilizando el prefijo guardado más largo. Las instantáneas pueden llevar un HMAC con clave. *No puede:* evitar que la expulsión sea con pérdida por diseño; `--kv-policy` y `--session` no se pueden combinar; las conversaciones muy largas pueden superar el contexto de entrenamiento.

### Modo reproducible y recibos

Módulos `simd` (interruptor), `receipt`. En `corbul`: `--reproducible`, `--receipt FILE`, `--receipt-key`, `--receipt-key-file`, `--receipt-topk`, `--keep-steps`, `--leaves`, `verify-receipt`, `replay`. Un recibo registra el modelo (una huella de Merkle de los pesos), el tokenizador, los ajustes del muestreador, el prompt y la salida con una cadena de hashes, sellado con SHA-256 y una clave HMAC opcional; `replay` lo vuelve a ejecutar. Los pesos dañados se localizan hasta un tensor y un fragmento. *No puede:* demostrar nada a un desconocido sin la clave HMAC o una nueva ejecución (no hay firmas de clave pública); se rechazan los modelos MoE y de espacio de estados; ARM sin probar; el modo reproducible cuesta velocidad.

### Herramientas de lenguaje con el rumano primero

Módulo `lang`. En `corbul`: `lang detect`, `lang normalize`, `--lang CODE` (no en el servidor). Normaliza ș/ț y ş/ţ, repara mojibake, identifica 27 idiomas, prohíbe escrituras ajenas durante la decodificación (rumano: sin ş ţ), da formato a números, dinero y fechas rumanos, restaura diacríticos ausentes marcando cada palabra incierta. *No puede:* ir más allá de un léxico de unas 650 formas, cuya restauración no se midió con ningún corpus; las palabras ambiguas se resuelven con unas pocas reglas de contexto; la identificación del idioma necesita unas cinco palabras.

### Analítica de incertidumbre

Módulo `confidence`. Solo biblioteca (ejemplo de demostración `confidence_demo`). Estadísticas por token, palabra, frase y respuesta para un mapa de calor, una heurística transparente de riesgo de alucinación, puntuaciones de autoconsistencia y herramientas de calibración. *No puede:* ser más que una heurística; la puntuación de riesgo no es un detector de verdad.

## Cómputo en tiempo de prueba, ajuste, recuperación

### Búsqueda y lookahead

Módulo `search`. En `corbul`: `search --method beam|best-of-n|vote` (`--width`, `--samples`, `--answer number|text`, `--length-penalty`, `--all`, `--explain`, `--json`) y `--lookahead` en `generate` (`--ngram-min`, `--ngram-max`). Búsqueda en haz, best-of-N, votación por autoconsistencia, un árbol de conversación (biblioteca), decodificación especulativa por búsqueda en el prompt. *No puede:* ayudar con el lookahead salvo cuando el texto repite texto anterior; best-of-N necesita un buen evaluador.

### Decodificación especulativa

Módulos `spec`, `selfspec`. En `corbul`: `--draft MODEL`, `--lookahead`, `--self-spec`, `--skip-layers auto|LIST`, `--draft-len`. Misma respuesta voraz y misma distribución de muestreo con menos pasadas completas; la salida aparece al final. Solo `--temperature`, `--top-k`, `--top-p`; no con salida estructurada, recibos, sesiones ni la malla. *No puede:* prometer una aceleración; la autoespeculación vuelve a pasos simples con anillos de ventana deslizante que darían la vuelta, cachés compactadas y capas descargadas.

### Cuantización mixta automática y el cuantizador

Módulos `autoquant`, `quantize`. En `corbul`: `autoquant FOLDER --budget 4.5bpw|6g|Q4_K` (`--recipe`, `--report`, `--candidates`, `--base`, `--imatrix`, `--sample-rows`) y `quantize IN OUT.gguf --type NAME | --recipe FILE`. Mide el error de pesos por grupo de tensores en cada formato y elige formatos dentro de un presupuesto de tamaño (mochila exacta o voraz). *No puede:* estimar la calidad de las respuestas, solo el error de los pesos; supone que los daños se suman; las i-quants no existen.

### Vectores de dirección y lente de logits

Módulo `steer`. En `corbul`: `steer train|apply|lens` (`--positive`, `--negative`, `--method mean-diff|pca`, `--out`, `--vector`, `--strength`, `--layers`, `--positions`, `--no-attention`). Aprende una dirección a partir de dos conjuntos de prompts y la suma al flujo residual sin cambiar los pesos; la lente muestra lo que predeciría cada capa. *No puede:* funcionar con modelos de espacio de estados, salida estructurada, especulación, recibos ni sesiones; un vector de un ajuste fino lo acepta otro con la misma forma.

### Recuperación local

Módulo `rag`. En `corbul`: `rag add|search|ask` (`--query`, `--question`, `--k`, `--chunk-chars`, `--overlap`, `--embed-model`). Troceado que conoce el rumano, un índice HNSW en Rust puro, BM25 con plegado de diacríticos, fusión de rankings, pasajes numerados `[S1]…` que alimentan las citas bloqueadas a la cita textual, archivos con suma de comprobación. *No puede:* ir más allá del incrustador por defecto, que es un incrustador por hash (palabras y fragmentos de palabras, no significado); `--embed-model` usa estados ocultos de un modelo generativo (débil); la CLI reindexa en cada llamada y lee texto plano, no PDF ni archivos de Office (la biblioteca tiene ayudantes de ingesta).

### Fusión de modelos

Módulo `merge`. En `corbul`: `merge CONFIG.json` (`--out`, `--model`, `--dry-run`, `--max-memory`, `--overwrite`, `--verbose`). Lineal, SLERP, aritmética de tareas, TIES, DARE (con semilla) y cortes de capas, un tensor cada vez. Las formas y los rangos se comprueban antes de escribir; no se siguen enlaces simbólicos. *No puede:* garantizar la calidad; solo escribe pesos safetensors, nunca cuantizados, y `config.json` y el tokenizador los copias tú.

### Marca de agua de texto

Módulo `watermark`. En `corbul`: `generate --watermark KEY` (`--wm-gamma`, `--wm-delta`, `--wm-context`) y `watermark detect MODEL --key K --text T | --file F` (`--no-dedup`). Un sesgo de «lista verde» con clave y un detector de puntuación z. *No puede:* dar una prueba; los textos cortos, el código y las citas llevan poca señal; la paráfrasis la elimina; quien tenga la clave puede falsificarla; que no haya marca no prueba autoría humana.

## Multimodal (solo biblioteca)

### Visión

Módulos `vision`, `imageio`. Un codificador de estilo CLIP/SigLIP, un proyector, teselado y un gancho `Model::forward_segments` para introducir embeddings en un prompt; decodificadores de PNG, JPEG (base y progresivo), BMP y PNM escritos a partir de las especificaciones, con límites de tamaño. *No puede:* usarse desde `corbul` (solo texto); no se ejecutó ningún checkpoint real.

### Audio

Módulo `audio`. Lectura y escritura de WAV, remuestreo, FFT, características log-mel, detección de actividad de voz, un ejecutor con la disposición de Whisper. *No puede:* usarse desde `corbul`; no se ejecutó ningún checkpoint real de Whisper.

## Espacio de estados, híbridos y GPU (véase ENGINE.es.md)

Los módulos `ssm`, `hybrid`, `mla` y el crate `companion-engine-gpu` se describen, con sus límites, en ENGINE.es.md.
