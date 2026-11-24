# El motor «clean-room» (`companion-engine` y `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Otros documentos: [INNOVATIONS.es.md](INNOVATIONS.es.md) · [QUICKSTART.es.md](QUICKSTART.es.md) · [CLEANROOM.es.md](CLEANROOM.es.md)

**Estado: alfa. Todo lo que sigue se probó solo con pequeños puntos de control sintéticos.** Los autores no han ejecutado ningún modelo real de Hugging Face con este motor (el entorno de compilación no podía llegar a huggingface.co), ninguna GPU ha ejecutado nada de él y las rutas de ARM, macOS y Windows nunca se han compilado aquí. Lee la tabla de «verificado y no verificado» antes de fiarte de cualquier afirmación.

## Qué es

Un motor de inferencia local para modelos de lenguaje, escrito desde cero en Rust a partir de descripciones públicas de formatos de archivo y de artículos publicados. Tiene tres partes:

| Crate | Qué es |
|---|---|
| `crates/engine` (`companion-engine`) | La biblioteca: formatos de archivo, tokenizador, cuantización, pasadas hacia delante, muestreo y las herramientas extra descritas en INNOVATIONS.es.md. Dependencias: solo `serde_json` y `memmap2`. |
| `crates/engine-cli` (`corbul`) | Una línea de comandos y un servidor local compatible con OpenAI / Ollama / Anthropic sobre la biblioteca. |
| `crates/engine-gpu` (`companion-engine-gpu`) | Núcleos WGSL para el crate `wgpu`, tras una opción desactivada por defecto. No lo usa `corbul`. |

Es independiente de la aplicación de escritorio Colibri Companion y no la sustituye: la aplicación sigue manejando colibri de JustVugg. Conectar este motor a la aplicación como backend alternativo está planeado, no hecho.

## Qué admite

**Archivos.** GGUF v2/v3 (la cabecera y la tabla de tensores se leen sin tocar los pesos, así que un archivo de 400 GB se inspecciona en milisegundos); un escritor GGUF en flujo para modelos mayores que la RAM; safetensors, en un archivo o fragmentado con índice; `config.json` y `tokenizer.json` de Hugging Face; un descargador de modelos (mediante `curl`, con reanudación y comprobación SHA-256).

**Cuantización.** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 y los K-quants Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K se pueden decodificar, y hay codificadores para todos (con ponderación por matriz de importancia), conversión Hugging Face → GGUF y recuantización de GGUF. Los pesos siguen cuantizados en memoria; el producto matriz–vector nunca construye la matriz. Los i-quants (IQ*) no están implementados.

**Arquitecturas.**

| Familia | Estado |
|---|---|
| Decodificadores densos: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 y similares (norma RMS o de capa, GQA/MQA, RoPE con escalado lineal / Llama-3 / YaRN y rotación parcial, QK-norm, sesgos, soft-capping, ventanas deslizantes, MLP con puerta o simple) | Implementadas; las carga `corbul` |
| Mezcla de expertos: Mixtral, Qwen-MoE, expertos apilados GGUF, expertos compartidos, expertos enrutados leídos desde disco con caché LRU | Implementadas; las carga `corbul` |
| Atención latente multicabeza (estilo DeepSeek-V2/V3, Kimi-K2) | Solo módulo de biblioteca; el registro de arquitecturas aún la marca «planeada» y `corbul` no la carga |
| Mamba-1, Mamba-2, RWKV-6, RWKV-7 | Solo módulo de biblioteca; igual |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, modelos encoder y encoder-decoder (T5, BERT, Whisper…), LongRoPE | No admitidos; `corbul` lo dice en lugar de adivinar |
| Entradas de imagen y audio | No admitidas (solo texto) |

**Rutas de hardware.**

- CPU: núcleos elegidos en ejecución (escalar, AVX2, AVX-512, VNNI en x86-64; NEON en aarch64), un pool de hilos persistente apto para núcleos de rendimiento y eficiencia mezclados, detección de hardware y NUMA y un planificador que decide qué se queda en RAM.
- GPU: núcleos WGSL comprobados con `naga` y frente a una emulación en Rust. Nunca se han ejecutado en una GPU.
- Modelos mayores que la RAM: expertos desde disco (MoE), transmisión de capas (denso), una malla LAN sobre varias máquinas.

## Verificado y no verificado

| Área | Estado |
|---|---|
| Pruebas unitarias, de integración y fuzz con puntos de control sintéticos (Linux, x86-64) | Verificado. Pasan cientos de pruebas; los archivos hostiles o truncados devuelven errores, no caídas. |
| Puntos de control reales de Hugging Face (cualquier familia) | **No ejecutado.** El entorno no podía llegar a huggingface.co. |
| Tokenizador con archivos `tokenizer.json` reales | Escrito según las descripciones públicas del formato; hay un banco de pruebas (`ENGINE_VOCAB_DIR`) pero no lo hemos ejecutado con archivos reales. |
| Permutación de filas Q/K al estilo llama.cpp en archivos GGUF | Implementada según la descripción; no verificada con archivos reales. |
| Cuantizadores y descuantizadores | Verificados frente a bloques calculados a mano y ida y vuelta; no frente a archivos escritos por otras herramientas. |
| Ejecución en GPU | **No ejecutada.** Los núcleos solo están validados con `naga` y una emulación en Rust. |
| Rutas ARM / NEON, macOS, Windows | **Nunca compiladas ni ejecutadas aquí.** Las sondas se prueban con texto grabado de esos sistemas. |
| Fórmulas de Mamba / RWKV / MLA | Escritas a partir de artículos y del recuerdo de las descripciones de referencia, probadas solo por coherencia interna. |
| Malla en una red real | Probada solo en loopback. |
| Velocidad | **Sin cifras fiables.** La máquina de compilación tenía dos núcleos compartidos; las mediciones fueron ruidosas. Todas las velocidades del piloto automático son estimaciones. |
| Modo reproducible | Idéntico en cinco variantes de núcleos x86 y varios números de hilos; ARM sin probar. |
| Licencias de las dependencias | Comprobadas con `cargo-deny` en un commit anterior (b496ba5); las incorporaciones posteriores aún no se han vuelto a comprobar. |

## Declaración clean-room

No se leyó ni copió código de colibri, llama.cpp ni de ningún otro motor de inferencia al escribir este motor. Los formatos y algoritmos proceden de especificaciones públicas, artículos y estructuras de archivo. Consulta [CLEANROOM.es.md](CLEANROOM.es.md) para el proceso y sus límites.

## Comparación con otros motores locales

No hemos hecho ninguna comparación directa, así que no afirmamos nada sobre velocidad o calidad. De forma cualitativa:

- Motores consolidados como llama.cpp, Ollama, mistral.rs o vLLM los ha ejecutado mucha gente con miles de puntos de control reales, tienen backends de GPU que funcionan, cubren más arquitecturas y formatos de cuantización (incluidos los i-quants) y manejan modelos de visión. Este motor aún no tiene nada de esa evidencia. Si necesitas algo que funcione hoy con un modelo grande concreto, usa uno de ellos.
- Lo que este motor intenta hacer distinto: herramientas de integridad y honestidad integradas (citas bloqueadas a las fuentes, modo reproducible con recibos sellados por cadena de hashes y HMAC opcional, análisis de incertidumbre, un piloto automático que dice «esto no va a funcionar» en vez de caerse), herramientas de idioma con el rumano primero, ejecutar modelos mayores que tu RAM o que una sola máquina, una base de código Rust pequeña con dos dependencias y solo licencias permisivas.
- Cualquier afirmación de que este motor es más rápido o más preciso que otro sería no probada. Mide en tu hardware y cuéntanoslo.

## Mapa de la biblioteca

Formatos y carga: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`. Modelo: `arch`, `model`, `moe`, `mla`, `ssm`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Ejecución: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Funciones distintivas: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Última oleada: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark` — implementadas en la biblioteca y cubiertas por pruebas con modelos sintéticos, aún en acabado, todavía no accesibles desde `corbul`, y sus API pueden cambiar.
