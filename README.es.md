# Colibri Companion

[English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · **Español** · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

**Sitio web: https://colibri-companion.chiuta.ro** — descarga, videotutoriales, ayuda.

Una aplicación de escritorio sencilla para [colibri](https://github.com/JustVugg/colibri): elige un modelo adecuado a tu ordenador, descárgalo con un clic y conversa — el modelo funciona en tu ordenador, sin cuenta y sin servicios en la nube.

## Ejecución

```bash
cargo run --release -p colibri-companion            # la aplicación real
cargo run --release -p colibri-companion -- --demo  # demo, sin colibri
# opciones: --lang en|ro|fr|it|es|pt|de|hu   --version   --help
```

### Los cuatro pasos

(1) encuentra Python y colibri (y ofrece descargar colibri si falta), (2) te permite elegir un modelo (los que no caben en la RAM se ocultan; el recomendado está marcado), (3) descarga e instala con progreso, pausa y reanudación, (4) inicia el servidor y abre el chat.

Chat: respuestas en flujo, Detener al instante, Markdown, razonamiento plegable, conversaciones guardadas, estilo (Preciso / Equilibrado / Creativo), copiar y regenerar. Temas: claro, oscuro, alto contraste.

## Cómo funciona

La aplicación no incluye colibri. Ejecuta los comandos JSON de colibri (`setup --list --json`, `status --json`, `setup`, `start`, `stop`) y se comunica con su pasarela compatible con OpenAI en `127.0.0.1`. Nunca instala paquetes del sistema: si falta algo, te muestra el comando para copiar. Tus datos (ajustes, conversaciones) se quedan en local, en la carpeta de datos del usuario; `COLIBRI_COMPANION_HOME` permite cambiarla.

## corbul: el motor clean-room (alfa)

Además de la aplicación de escritorio, el repositorio contiene un motor de inferencia escrito desde cero en Rust, y `corbul`, su línea de comandos y servidor local (estilos OpenAI, Ollama y Anthropic, con una interfaz web en `/`). No sustituye a colibri en la aplicación.

```bash
cargo build --release -p colibri-rs-cli      # target/release/corbul
corbul capabilities                          # what works and what does not
corbul generate ./model --prompt "Hello"     # a folder with safetensors or a .gguf file
```

**Estado honesto:** probado solo con modelos sintéticos diminutos y con archivos de referencia de tokenizadores (12 familias de modelos, 46 de 46 casos iguales a los datos de referencia de llama.cpp). Ningún modelo real descargado, ninguna GPU y ninguna máquina Windows, macOS o ARM lo ha ejecutado todavía; las velocidades son estimaciones. Se cargan modelos densos y de mezcla de expertos; Mamba, RWKV y Jamba/Bamba se ejecutan como continuación de texto plano; la atención al estilo DeepSeek, las imágenes, el sonido y la GPU no están disponibles en `corbul`; los embeddings funcionan, y el reranking funciona con un cross-encoder BERT mediante `--encoder`. Las i-quants distintas de IQ4_NL e IQ4_XS y los codificadores de tipo RoBERTa no se admiten.

Documentación: [qué es y qué no hace](docs/engine/ENGINE.es.md) · [inicio rápido y todas las opciones](docs/engine/QUICKSTART.es.md) · [funciones distintas](docs/engine/INNOVATIONS.es.md) · [proceso clean-room](docs/engine/CLEANROOM.es.md) · [cambios de seguridad](docs/engine/CHANGES-SECURITY.es.md).

## Licencias

- Este proyecto: Apache-2.0 (`LICENSE`, `NOTICE`), como colibri.
- Dependencias: solo licencias permisivas, comprobadas con `cargo deny check licenses` (`deny.toml`); lista completa en `THIRD_PARTY_LICENSES.md`. Fuentes: Inter y JetBrains Mono (SIL OFL 1.1, en `licenses/`).
- Los modelos tienen sus propias licencias; la aplicación las muestra y pide confirmación antes de descargar.

## Plataformas

Linux (x86-64, ARM64; Wayland y X11), Windows 10/11 y macOS (Intel y Apple Silicon) — un solo código fuente, renderizado OpenGL (funciona en tarjetas gráficas antiguas y en máquinas virtuales). `.github/workflows/build.yml` compila y prueba todas las variantes y genera archivos listos para distribuir. En Linux, `packaging/linux/install.sh` añade la aplicación al menú. Escalado HiDPI, atajos Ctrl/⌘+N, Ctrl/⌘+, y Esc, y compatibilidad con lectores de pantalla (AccessKit). Las compilaciones y pruebas automáticas cubren los tres sistemas; las pruebas manuales en ordenadores reales con Windows y macOS siguen siendo limitadas, y las etiquetas para lectores de pantalla (AccessKit) aún no se han probado con lectores reales.

Linux más antiguos (Debian 8+, Ubuntu 14.04+, CentOS 7+): compilaciones adicionales `linux-x86_64-glibc2.17` y `linux-aarch64-glibc2.17`, enlazadas con glibc 2.17; la CI comprueba la base de símbolos y carga el programa en Ubuntu 16.04. Las bibliotecas de ventanas y gráficos siguen viniendo de tu sistema.

**Documentos y herramientas:** añade archivos (**Más → Documentos**) y haz preguntas: la aplicación los busca en tu ordenador (palabras clave, sin nube) y muestra bajo cada respuesta los pasajes utilizados. **Más → Herramientas MCP** conecta servidores MCP que tú añades e inicias; el modelo puede solicitar una herramienta y cada llamada requiere tu aprobación (función temprana, depende de que el modelo siga el formato).

Modelos: conserva varios en el ordenador, cambia entre ellos o añade más desde una sola ventana (**Modelos**); primero se sugiere el más capaz que tu ordenador puede ejecutar (con poca RAM, los expertos se leen del SSD). Velocidad: CUDA para NVIDIA, Vulkan para otros chips gráficos, si no el procesador, elegidos automáticamente. **Recursos** muestra en directo procesador, memoria, tarjeta gráfica y disco, y la prueba de que nada sale de tu ordenador (todas las conexiones abiertas de la aplicación y del motor, con el comando para comprobarlo por ti mismo). Si falta colibri se descarga por ti (sin git), y una comprobación diaria (se puede desactivar) avisa cuando colibri o esta aplicación tienen una versión nueva.

**Tu ordenador, analizado:** en **Recursos**, la aplicación encuentra sola tus tarjetas gráficas y dice con claridad si el modelo cabe en memoria y por qué puede ir lento (swap, otros programas, tarjeta sin usar). Solo aconseja; no cambia ningún ajuste por sí sola.

**Herramientas más nuevas (todas probadas sin un modelo real; el sitio dice qué está verificado y qué no):** un rango de velocidad bajo cada modelo antes de descargarlo (según tu memoria, tu disco y el catálogo; nunca una medición) y la grabación ROUTE_TRACE con un análisis de lo que ahorraría lecturas de disco; el ajuste automático de memoria fijada e hilos (**Recursos**); **Modelos ligeros para tu memoria** en la ventana Modelos (11 modelos pequeños, filtro de llamada a herramientas documentada, copia del comando `ollama run`); **Comprobar citas** y **Archivos escaneados (LegatorOCR)** en la ventana Documentos (el segundo necesita el programa independiente LegatorOCR, licencia MIT; el texto OCR no es una transcripción verificada); y un panel **Consumo** junto a cada respuesta (tiempo, primer token, tokens de entrada y salida, velocidad actual, media y máxima, contexto usado, procesador, memoria, swap y tarjeta gráfica de este ordenador; las cifras con ≈ son estimaciones). La sección **Bibliografía** de la ventana Documentos convierte BibTeX, RIS y CSL JSON y copia una lista de referencias (APA, Harvard, IEEE); probada solo con entradas generadas. La misma ventana tiene ahora **Herramientas de investigación**: búsqueda con números de página, comprobación de citas con páginas reales, recibos de procedencia y su comprobación, declaración de uso de IA en 8 idiomas y medición de la búsqueda; la Bibliografía también guarda archivos, une duplicados, comprueba DOI/ISBN/ISSN/ORCID y escribe citas en el texto. Solo la búsqueda por significado (el servidor colibri no tiene endpoint de embeddings) sigue siendo una biblioteca que la ventana no usa. Todo está probado con archivos pequeños generados, no con tesis reales. Más en `docs/`, cada archivo en los ocho idiomas (`NOMBRE.md`, `NOMBRE.ro.md`, `NOMBRE.fr.md`, `NOMBRE.it.md`, `NOMBRE.es.md`, `NOMBRE.pt.md`, `NOMBRE.de.md`, `NOMBRE.hu.md`). Novedad en alpha.7: una sección **Citas** en la ventana Documentos da a cada documento una cita (`[@nombre]`) que la comprobación de citas reconoce, y copia toda la lista como BibTeX, RIS, CSL JSON, APA, Harvard o IEEE; las ventanas ya no se salen de la pantalla (el contenido se desplaza y la barra de título con el botón de cierre sigue a mano) y Esc cierra la ventana abierta incluso desde un cuadro de texto.

## Idiomas

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — se eligen en Ajustes, con `--lang en|ro|fr|it|es|pt|de|hu`, o según el idioma del sistema en el primer inicio. Las traducciones están en `crates/app/src/i18n.rs` (la clave es el texto en inglés); una prueba comprueba que cada texto de la interfaz tenga todas las traducciones. Se hicieron con ayuda de una IA y no las han revisado hablantes nativos.

## Desarrollo

```bash
cargo test --workspace   # incluye una prueba de extremo a extremo con un `coli` falso (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```

Nuevo: búsqueda en las conversaciones (Ctrl/⌘+K), renombrar (clic derecho), editar el último mensaje, archivos de texto arrastrados al mensaje y guardado como .md.
