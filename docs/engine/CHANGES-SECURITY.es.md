# Cambios de seguridad y robustez

[English](CHANGES-SECURITY.md) · [Română](CHANGES-SECURITY.ro.md) · [Français](CHANGES-SECURITY.fr.md) · [Italiano](CHANGES-SECURITY.it.md) · [Español](CHANGES-SECURITY.es.md) · [Português](CHANGES-SECURITY.pt.md) · [Deutsch](CHANGES-SECURITY.de.md) · [Magyar](CHANGES-SECURITY.hu.md)

Otros documentos: [ENGINE.es.md](ENGINE.es.md) · [QUICKSTART.es.md](QUICKSTART.es.md)

Esta página enumera lo que encontró la revisión adversaria del motor y de `corbul`, y lo que se hizo al respecto, para que puedas juzgar hasta qué punto fiarte de una función. Es una lista de correcciones y límites, no una certificación de seguridad: nada de esto fue auditado por un tercero.

## Servidor

- **Peticiones entre sitios.** Las llamadas que cambian estado comprueban `Origin` y exigen un `Content-Type` JSON, de modo que una página web que visites no pueda hacer que tu navegador envíe peticiones a un `corbul serve` local. Los navegadores solo se admiten desde los orígenes que indiques con `--cors`.
- **Rutas de administración.** `/api/create`, `/api/delete`, `/models/load` y `/models/unload` necesitan el token maestro. Los titulares de claves de API solo pueden usarlas si inicias el servidor con `--allow-admin`.
- **Secretos fuera de la línea de comandos.** `--token-file`, `--key-file`, `--receipt-key-file` y las variables `CORBUL_TOKEN` y `CORBUL_MESH_KEY` mantienen los secretos fuera de la lista de procesos. Los simples `--token`, `--key` y `--receipt-key` siguen funcionando y son visibles para otros usuarios de la máquina.
- **Límites de recursos.** `--max-kv` rechaza las peticiones cuya caché KV sería demasiado grande (por defecto 8g); `--max-heavy` limita las peticiones simultáneas de tokenización, plantilla y verificación (por defecto 4); los límites de frecuencia de `--keys-file` cubren todas las rutas que cuestan cómputo; el decodificador de cuerpos fragmentados es lineal; `POST /v1/quotes/verify` tiene topes en el tamaño del texto y de las fuentes.
- **Registros.** El texto de los prompts no se registra salvo que se indique `--log-prompts`; `--no-log` desactiva el registro.
- **Interfaz web.** La página en sí es pública (no contiene secretos); cada llamada que hace necesita el token.
- **Límite que permanece:** el servidor habla HTTP sin cifrar. Enlázalo a `127.0.0.1` (el valor por defecto) o pon TLS delante.

## Malla

- Handshake HMAC-SHA256 mutuo con una clave compartida de al menos 8 caracteres, un MAC en cada trama, topes en los tamaños de handshake y de trama.
- El descubrimiento UDP es opcional (`--discovery`, `--discovery-port`) y las respuestas no son mayores que la sonda, de modo que un worker no puede usarse como amplificador.
- **Límite que permanece:** el tráfico está autenticado pero no cifrado; la clave la comparte todo el mundo en la malla; probado solo en loopback.

## Archivos, descargas y fusiones

- El análisis de GGUF recibe la longitud real del archivo (`parse_with_len`); la construcción del tokenizador usa un trie con un tope en el tamaño de las piezas; los archivos truncados y corruptos devuelven errores en las pruebas adversarias.
- Descargas: fijación al SHA del commit, límites de tamaño, rechazo de redirecciones a http sin cifrar, `wget` anónimo como alternativa (el token nunca se le pasa), comprobaciones SHA-256.
- Las fusiones no siguen enlaces simbólicos y se niegan a sobrescribir una salida salvo que se indique `--overwrite`; las formas y los rangos se comprueban antes de escribir.
- Los decodificadores de imagen aplican `Limits` antes de reservar memoria y acotan la expansión de deflate; el lector de WAV tiene límites frente a archivos hostiles.

## Decodificación y verificación

- La decodificación por gramática, expresión regular y esquema JSON tiene presupuestos de trabajo; el bloqueo de citas es lineal con topes e informa `limit_hit` cuando un tope lo detiene.
- Los recibos tienen topes de tamaño; las instantáneas pueden sellarse con un HMAC con clave; los recibos y las instantáneas usan cadenas SHA-256.
- Correcciones de exactitud halladas en la revisión: probabilidades obsoletas tras top-k/top-p (ahora renormalizadas), el alias `typ_p`, comas decimales en el tratamiento de números, límites de lookahead más allá del contexto, la puntuación de grupo de DeepSeek-V2, la exactitud por lotes cuantizada (`exact_batch`).

## No hecho o no verificado

- Sin auditoría de terceros, sin campaña de fuzzing más allá de las pruebas adversarias del repositorio, sin prueba de penetración de `corbul serve`.
- `ToolStream::push` puede volver a recorrer su búfer repetidamente (cuadrático en la longitud de la salida hasta `--max-tokens`).
- El `GrammarCache` existe pero la CLI todavía no lo usa.
- Las licencias de las dependencias se comprobaron por última vez con `cargo-deny` en el commit b496ba5.
- Los workflows de CI nunca se han ejecutado; las rutas de código de ARM, macOS y Windows no se han ejecutado aquí.
