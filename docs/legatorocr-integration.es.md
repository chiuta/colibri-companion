# Integración de LegatorOCR

Estado: biblioteca en `crates/core/src/legatorocr.rs`, conectada a la ventana Documentos por `crates/app/src/ocr_ui.rs` (sección «Archivos escaneados (LegatorOCR)»). Probada con un binario real `legatorocr 0.2.0-pre.65` en una sola página generada de texto sintético limpio; nunca con el escaneo real de un libro.

## Qué hace

LegatorOCR (github.com/chiuta/LegatorOCR, **MIT**) convierte escaneos en PDF o texto con búsqueda. Colibri Companion (Apache-2.0) nunca enlaza ni copia su código; solo ejecuta el programa como un comando externo, algo que permiten ambas licencias.

1. `locate` / `locate_in`: busca un ejecutable `legatorocr` (una carpeta indicada por el usuario, la carpeta de la aplicación, luego `PATH`). Solo el sistema de archivos, sin red.
2. `version`: ejecuta `legatorocr --version`; la respuesta debe empezar por `legatorocr`, para que un archivo ajeno con el mismo nombre no se tome por el programa.
3. `collect_inputs`: una imagen o un PDF, o las imágenes y PDF que estén directamente en una carpeta, en orden de nombre (como máximo 2000).
4. `read_scan`: ejecuta `legatorocr <páginas> --ocr --lang <código> --txt <archivo> --format none --out-dir <carpeta>` (más `--clean-text` para rumano) mediante `proc::run_capture` con límite de tiempo, y luego vuelve a leer el texto (como máximo 32 MB).
5. `ingest_scan`: añade ese texto a una colección de documentos (`rag::Library::add_document`) con el nombre `<archivo o carpeta> (OCR)`. El modelo local responde entonces desde el escaneo y los pasajes se citan con ese nombre como cualquier otro documento.

Seguridad: los argumentos se pasan uno a uno (sin shell); las entradas se vuelven absolutas para que un archivo llamado `-x.png` no se lea como opción; el código de idioma se comprueba (`clean_lang`); la salida tiene tope; una ejecución que no termina se mata; los fallos vuelven con las últimas líneas de la salida de error del programa; nada arranca si no se llama.

## Límites, con franqueza

- El texto es OCR. No es una transcripción verificada, y la calidad del OCR (y qué idiomas funcionan) depende de LegatorOCR y de los paquetes de idioma de Tesseract instalados: `legatorocr --doctor` los lista (en la máquina de pruebas faltaban los modelos de rumano y húngaro).
- Los números de página se pierden: todo el escaneo es un solo documento, así que una cita nombra el escaneo, no la página.
- La búsqueda en la biblioteca es por palabras clave (BM25), como en cualquier documento.
- La ventana muestra la sección solo cuando se encuentra el programa (se busca una vez, en segundo plano, al abrir la ventana; nada se descarga ni se inicia solo). Sin él solo aparecen una pista con `github.com/chiuta/LegatorOCR` y un botón «Buscar de nuevo».
- La ejecución es un hilo en segundo plano con un botón Detener; LegatorOCR no da cifra de progreso, así que la ventana muestra los segundos transcurridos, no un porcentaje. Cerrar toda la aplicación durante una ejecución no se trata de forma especial.
- La lista de idiomas son los 8 idiomas de la aplicación asignados a códigos de Tesseract; que uno funcione depende de los datos instalados para LegatorOCR (`legatorocr --doctor`). La aplicación no lo comprueba antes: un idioma que falta vuelve como el mensaje de error del propio programa.
- Las imágenes de página soltadas en la ventana (no los PDF) van al campo de escaneo cuando se encuentra el programa.

## Pruébalo

```
cargo test -p companion-core legatorocr
LEGATOROCR_BIN=/path/to/legatorocr LEGATOROCR_TEST_PAGE=/path/to/page.png \
  cargo test -p companion-core real_legatorocr -- --nocapture
```
