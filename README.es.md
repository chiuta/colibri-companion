# Colibri Companion

[English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · **Español** · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

Una aplicación de escritorio amigable para [colibri](https://github.com/JustVugg/colibri): elige un modelo adecuado a tu ordenador, descárgalo con un clic y conversa — todo en local, sin cuenta y sin nube.

## Ejecución

```bash
cargo run --release -p colibri-companion            # la aplicación real
cargo run --release -p colibri-companion -- --demo  # demo, sin colibri
# opciones: --lang en|ro|fr|it|es|pt|de|hu   --version   --help
```

### Los cuatro pasos

(1) encuentra Python y colibri (y ofrece descargar colibri si falta), (2) te deja elegir un modelo (los que no caben en la RAM se ocultan; el recomendado está marcado), (3) descarga e instala con progreso, pausa y reanudación, (4) inicia el servidor y abre el chat.

Chat: respuestas en flujo, Detener al instante, Markdown, razonamiento plegable, conversaciones guardadas, estilo (Preciso / Equilibrado / Creativo), copiar y regenerar. Temas: claro, oscuro, alto contraste.

## Cómo funciona

La aplicación no incluye colibri. Ejecuta los comandos JSON de colibri (`setup --list --json`, `status --json`, `setup`, `start`, `stop`) y se comunica con su pasarela compatible con OpenAI en `127.0.0.1`. Nunca instala paquetes del sistema: si falta algo, te muestra el comando para copiar. Tus datos (ajustes, conversaciones) se quedan en local, en la carpeta de datos del usuario; `COLIBRI_COMPANION_HOME` permite cambiarla.

## Licencias

- Este proyecto: Apache-2.0 (`LICENSE`, `NOTICE`), como colibri.
- Dependencias: solo licencias permisivas, comprobadas con `cargo deny check licenses` (`deny.toml`); lista completa en `THIRD_PARTY_LICENSES.md`. Fuentes: Inter y JetBrains Mono (SIL OFL 1.1, en `licenses/`).
- Los modelos tienen sus propias licencias; la aplicación las muestra y pide confirmación antes de descargar.

## Plataformas

Linux (x86-64, ARM64; Wayland y X11), Windows 10/11 y macOS (Intel y Apple Silicon) — un solo código fuente, renderizado OpenGL (funciona en tarjetas gráficas antiguas y en máquinas virtuales). `.github/workflows/build.yml` compila y prueba todas las variantes y genera archivos listos para distribuir. En Linux, `packaging/linux/install.sh` añade la aplicación al menú. Escalado HiDPI, atajos Ctrl/⌘+N, Ctrl/⌘+, y Esc, y compatibilidad con lectores de pantalla (AccessKit).

## Idiomas

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — se eligen en Ajustes, con `--lang en|ro|fr|it|es|pt|de|hu`, o según el idioma del sistema en el primer inicio. Las traducciones están en `crates/app/src/i18n.rs` (la clave es el texto en inglés); una prueba comprueba que cada texto de la interfaz tenga todas las traducciones. Se hicieron con ayuda de una IA y no las han revisado hablantes nativos.

## Desarrollo

```bash
cargo test --workspace   # incluye una prueba de extremo a extremo con un `coli` falso (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```

Nuevo: búsqueda en las conversaciones (Ctrl/⌘+K), renombrar (clic derecho), editar el último mensaje, archivos de texto arrastrados al mensaje y guardado como .md.
