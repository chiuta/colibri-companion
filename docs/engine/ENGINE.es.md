# El motor clean-room (`companion-engine` y `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Otros documentos: [INNOVATIONS.es.md](INNOVATIONS.es.md) · [QUICKSTART.es.md](QUICKSTART.es.md) · [CLEANROOM.es.md](CLEANROOM.es.md) · [CHANGES-SECURITY.es.md](CHANGES-SECURITY.es.md)

**Estado: alfa.** Todo lo que sigue se probó con pequeños checkpoints sintéticos, con archivos de referencia de tokenizadores y con texto de muestra grabado. **Sus autores no han ejecutado con este motor ningún modelo real descargado de Hugging Face**, ninguna GPU ha ejecutado nada de esto, y las rutas de código de ARM, macOS y Windows no se han ejecutado aquí. Lee «Verificado y no verificado» antes de fiarte de una afirmación. `corbul capabilities` imprime los mismos límites desde el propio binario.

## Qué es

Un motor local de inferencia para modelos de lenguaje, escrito desde cero en Rust a partir de descripciones públicas de formatos de archivo y de artículos publicados.

| Crate | Qué es |
|---|---|
| `crates/engine` (`companion-engine`) | La biblioteca: formatos de archivo, tokenizador, cuantización, pasadas hacia delante, muestreo y las herramientas adicionales de INNOVATIONS.es.md. Dependencias en tiempo de ejecución: solo `serde_json` y `memmap2`. |
| `crates/engine-cli` (`corbul`) | La línea de comandos y un servidor local compatible con OpenAI / Ollama / Anthropic con interfaz web integrada. Depende de la biblioteca y de `serde_json`. |
| `crates/engine-gpu` (`companion-engine-gpu`) | Kernels WGSL y una emulación de ellos en CPU. El ejecutor en dispositivo (`wgpu`, `pollster`) queda tras la característica de cargo `wgpu`, desactivada por defecto. **`corbul` no se ejecuta sobre él; compilado con `--features gpu`, `corbul doctor` solo imprime su informe de capacidades.** |

Es independiente de la aplicación de escritorio Colibri Companion y no la sustituye; esta sigue manejando el colibri de JustVugg. Usar este motor como back end alternativo de la aplicación está planeado, no hecho.

## Qué admite

**Archivos.** GGUF v2 y v3 (la cabecera y la tabla de tensores se leen sin tocar los pesos); un escritor GGUF en streaming para salidas mayores que la RAM; safetensors, único o fragmentado con índice; `config.json`, `tokenizer.json` y plantillas de chat de Hugging Face; un descargador de modelos (módulo `hub`, mediante `curl`, con `wget` anónimo como alternativa; reanudación, comprobaciones SHA-256, descargas fijadas a un commit, límites de tamaño, rechazo de redirecciones a http sin cifrar).

**Cuantización.** Se decodifican: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K (14 tipos). Existen codificadores para los mismos tipos, con ponderación por matriz de importancia y recetas por tensor. Los pesos permanecen cuantizados en memoria; el producto matriz–vector nunca expande la matriz entera. **No se admiten:** las i-quants (IQ\*) ni ningún tipo que no esté en la lista.

**Arquitecturas.**

| Familia | Estado en `corbul` |
|---|---|
| Decodificadores densos: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 y similares (norma RMS o de capa, GQA/MQA, RoPE con escalado lineal, Llama-3, YaRN y LongRoPE y rotación parcial, QK-norm, sesgos, soft-capping, ventanas deslizantes, MLP con o sin compuerta) | Cargados y ejecutados (plantilla de chat, serve, todas las opciones) |
| Mezcla de expertos: Mixtral, Qwen-MoE, expertos apilados de GGUF, expertos compartidos, expertos enrutados transmitidos desde disco con caché LRU | Cargados y ejecutados |
| Espacio de estados: Mamba-1, Mamba-2, RWKV-6, RWKV-7 (módulo `ssm`) | Ejecutados por `generate` y `chat` como **continuación de texto plano**: sin plantilla de chat, sin salida estructurada, sin `serve`. RWKV en formato GGUF no se admite (solo safetensors) |
| Híbridos: Jamba y Bamba (módulo `hybrid`) | Igual que la fila de espacio de estados. Zamba y Zamba2 no se admiten |
| Atención latente multicabezal (DeepSeek-V2/V3, Kimi-K2; módulo `mla`) | **Solo biblioteca.** El bloque existe y se prueba por coherencia interna, pero `corbul` no ensambla un modelo así (`corbul capabilities` dice que no) |
| Codificador de visión (ViT, CLIP, SigLIP; módulo `vision`, decodificadores en `imageio`) y front-end de voz más un ejecutor con la disposición de Whisper (módulo `audio`) | **Solo biblioteca.** `corbul` solo acepta texto |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, modelos de codificador y de codificador-decodificador (T5, BERT…), tokenizadores WordPiece | No se admiten; `corbul` lo dice en lugar de adivinar |
| Endpoints de embeddings (`/v1/embeddings`, `/api/embed`, `/api/embeddings`) | Funcionan: estados ocultos finales agregados (media, último o primero), longitud unitaria por defecto |
| Endpoints de reranking | Responden 501 (no implementado) |

**Tokenizador.** BPE a nivel de byte y modelos de estilo SentencePiece, reglas de pretokenización para 12 familias, tokens añadidos, retroceso a bytes, plantillas desde `tokenizer_config.json` y desde los metadatos GGUF. WordPiece/BERT no se admite.

**Muestreo.** Temperatura, top-k, top-p, min-p, typical-p, penalizaciones de repetición / presencia / frecuencia, Mirostat 1 y 2, DRY, XTC, una especificación JSON del muestreador, aleatoriedad con semilla; decodificación restringida por JSON, JSON Schema, expresión regular o gramática GBNF (con presupuestos de trabajo para que una gramática hostil no pueda colgar el proceso); sesgo de logits y cadenas de parada a través del servidor. Decodificación especulativa: modelo borrador, búsqueda en el prompt, autoespeculación (véase INNOVATIONS.es.md).

**Servidor.** `corbul serve` ofrece `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/messages` y `/v1/messages/count_tokens` (estilo Anthropic), los endpoints de Ollama `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/create`, `/api/delete`, `/api/ps`, `/api/version`, la gestión de modelos `/models`, `/models/load`, `/models/unload`, además de `/metrics`, `/props`, `/slots`, `/tokenize`, `/detokenize`, `/apply-template`, `/health`, `/v1/capabilities`, `/v1/quotes/verify` y la interfaz web en `/` y `/ui` (ocho idiomas). La llamada a herramientas se admite a través de las plantillas de chat. Las rutas de embeddings (`/v1/embeddings`, `/api/embed`, `/api/embeddings`) devuelven un vector por entrada, tomado de los estados ocultos normalizados de la última capa; el cuerpo admite `input` (un texto, ids de tokens o una lista de cualquiera de ellos; como máximo 64 entradas y 65 536 tokens), `pooling` (`mean` por defecto, `last`, `first`), `normalize` (true por defecto), `dimensions`, `truncate` y `encoding_format` (`float` o `base64`). Están pensadas para modelos decodificadores, cuyos vectores son utilizables pero no están entrenados para recuperación. Las rutas de rerank (`/v1/rerank`, `/rerank`) responden 501. Las respuestas en streaming no llevan bloque de citas de citas textuales.

**Hardware.** Los kernels de CPU se eligen en tiempo de ejecución: escalar, avx2+fma, avx2+fma+avxvnni, avx512, avx512+vnni (x86-64), neon y neon+dotprod (aarch64); `ENGINE_SIMD` fuerza uno. Un pool de hilos persistente, detección de hardware y de NUMA, y un planificador (`corbul plan`, `corbul doctor`) deciden qué se queda en RAM. Los kernels de GPU nunca se han ejecutado en una GPU (véase más abajo).

## Exactitud

- **Pesos f32 densos:** las rutas por lotes y de un solo token, la malla, el streaming de capas y las cachés dan logits idénticos bit a bit.
- **Pesos cuantizados:** la identidad bit a bit entre la evaluación por lotes y la de un solo token solo se cumple con `exact_batch` activado. La verificación, las rondas de verificación autoespeculativa y el ejecutor por lotes lo activan. El prefill ordinario usa una ruta más rápida de decuantizar y luego dividir en teselas, y difiere de la evaluación de un solo token solo por ruido de redondeo.
- **Modo reproducible** (`--reproducible` o `ENGINE_REPRODUCIBLE=1`): resultados idénticos entre las variantes de kernel x86 y los números de hilos que pudimos probar. ARM no se probó. Cuesta velocidad.
- Mirostat 1 usa en su estimación el número de candidatos restantes, no el tamaño del vocabulario (desviación documentada).

## Verificado y no verificado

| Área | Estado |
|---|---|
| Pruebas unitarias, de integración, adversarias y de fuzzing con checkpoints sintéticos (Linux x86-64). Los recuentos cambian con cada commit; ejecuta `cargo test --workspace` para el número actual | Pasan en la última ejecución completa. Los archivos hostiles y truncados devuelven errores. |
| Tokenizador frente a los archivos de referencia de llama.cpp, 12 familias | 46 de 46 casos iguales (archivos de referencia obtenidos de GitHub; `crates/engine/tests/tokenizer_real.rs`, `ENGINE_VOCAB_DIR`). Solo el tokenizador, no modelos completos. |
| Modelo real de Hugging Face a través de `corbul` (cualquier familia) | **No lo han ejecutado los autores.** El sandbox de compilación no podía llegar a huggingface.co. Un job manual de CI ejecuta un llama GGUF diminuto (`stories260K`) por la CLI; todavía no se ha ejecutado |
| Permutación de filas Q/K al estilo llama.cpp, variantes de metadatos GGUF | Implementadas a partir de descripciones públicas; no comprobadas con archivos reales |
| Cuantizadores y decuantizadores | Bloques calculados a mano y ciclos de ida y vuelta; no comparados con archivos escritos por otras herramientas |
| Ejecución en GPU | **No ejecutada.** Los kernels se validan solo con `naga` y con una emulación en Rust; `corbul` no los usa |
| ARM / NEON, macOS, Windows | **No ejecutados aquí.** Las sondas se prueban con texto de muestra grabado. El workflow de CI tiene una matriz para ellos pero nunca se ha ejecutado |
| Mamba, RWKV, Jamba/Bamba, MLA, visión, audio | Solo pruebas de coherencia interna; no se usó ningún checkpoint real |
| Malla | Solo loopback; autenticada, no cifrada; sin medición de velocidad |
| Velocidad | **Sin cifras fiables.** La máquina de compilación tenía dos núcleos compartidos. Las velocidades del piloto automático son estimaciones |
| Licencias de las dependencias | Comprobadas con `cargo-deny` en el commit b496ba5; los cambios posteriores no se han vuelto a comprobar |
| Traducciones de los documentos | Asistidas por máquina; todavía sin revisión por hablantes nativos |

## Variables de entorno

| Variable | Efecto |
|---|---|
| `COLIBRI_THREADS` | Tamaño del pool de hilos global (por defecto: núcleos utilizables). La biblioteca la lee primero |
| `ENGINE_SIMD` | Fuerza una variante de kernel por nombre (benchmarks, informes de errores) |
| `ENGINE_REPRODUCIBLE` | `1`, `true` o `on`: modo reproducible desde el principio |
| `ENGINE_BACKEND` | `cpu`, `gpu` o `auto` (sin definir = auto). `corbul` se ejecuta en la CPU en ambos casos |
| `ENGINE_GPU`, `ENGINE_GPU_API`, `ENGINE_GPU_SOFTWARE` | Selección de adaptador para el crate de GPU: índice, API gráfica, permitir un adaptador por software. No los usa `corbul` |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Token de acceso para repositorios restringidos (`download`, `inspect owner/repo`) |
| `HF_ENDPOINT` | Espejo del host de Hugging Face |
| `CORBUL_TOKEN` | Token bearer compartido de `corbul serve` (igual que `--token`) |
| `CORBUL_MESH_KEY` | Clave de malla para `worker` y `--workers`/`--discover` (igual que `--key`) |
| `CORBUL_CONFIG_DIR` | Dónde se guardan los Modelfiles creados por `POST /api/create` (igual que `--config-dir`; por defecto `~/.config/corbul`) |
| `NUMBER_OF_PROCESSORS`, `PROCESSOR_IDENTIFIER` | La sonda de hardware las lee en Windows; tú no las defines |
| `ENGINE_VOCAB_DIR`, `ENGINE_VOCAB_ONLY`, `ENGINE_REQUIRE_VOCAB`, `ENGINE_REQUIRE_TOOLS`, `ENGINE_PRINT_GOLDEN`, `ADVERSARIAL_N` | Controles de la batería de pruebas: carpeta de vocabularios, restringir familias, hacer fallar una prueba omitida, hacer fallar la ausencia de `curl`/`wget`, regenerar los hashes de referencia, tamaño de la ejecución adversaria |

## Declaración clean-room

No se leyó ni se copió código de colibri, de llama.cpp ni de ningún otro motor de inferencia al escribir este motor. Los formatos y los algoritmos proceden de especificaciones públicas, artículos y disposiciones de archivo. El modelo que escribió el código se entrenó con texto público que puede incluir esos proyectos; véase [CLEANROOM.es.md](CLEANROOM.es.md) para el proceso y sus límites.

## Comparación con otros motores locales

No hicimos ningún benchmark cara a cara, así que aquí no hay afirmaciones de velocidad ni de calidad.

- Los motores establecidos (llama.cpp, Ollama, mistral.rs, vLLM) se han ejecutado con miles de checkpoints reales, tienen back ends de GPU que funcionan, cubren más arquitecturas y formatos de cuantización, incluidas las i-quants, y manejan modelos de visión. Este motor aún no tiene nada de esa evidencia. Si necesitas algo que funcione hoy con un modelo grande concreto, usa uno de ellos.
- Lo que este motor intenta hacer de otra manera: herramientas de integridad incorporadas (citas bloqueadas a la cita textual, modo reproducible con recibos, analítica de incertidumbre, un planificador que dice «esto no se ejecutará» en lugar de caerse), herramientas de lenguaje con el rumano en primer lugar, modelos mayores que tu RAM o que tu máquina, una base de código pequeña en Rust con dos dependencias de biblioteca, y solo licencias permisivas.

## Mapa de la biblioteca

Formatos y carga: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`, `imageio`. Modelos: `arch`, `model`, `moe`, `mla`, `ssm`, `hybrid`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Ejecución: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Funciones distintivas: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Cómputo en tiempo de prueba, ajuste, recuperación: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark`. Multimodal: `vision`, `audio`. En INNOVATIONS.es.md se indica cuáles de ellos alcanza `corbul`; las API de la biblioteca aún pueden cambiar.
