# Publicar una versión

Cómo se hace una versión (para el mantenedor)

1. **Versión.** Cambia `version` en el `Cargo.toml` raíz y `TAG` en `index.html` por el número nuevo (por ejemplo `0.1.0-alpha.12` y `v0.1.0-alpha.12`), ejecuta `cargo update -w --offline`, añade `docs/releases/<tag>.md`, y haz commit y push a `main`.
2. **Compilación.** Espera a que las ejecuciones `build` y `engine-ci` de ese commit estén en verde. Luego lanza a mano el workflow `build` (Actions, Run workflow) y escribe la etiqueta, por ejemplo `v0.1.0-alpha.12`. Compila todas las plataformas y crea una versión **borrador**.
3. **Publicación.** Abre el borrador, pega `docs/releases/<tag>.md` en la descripción, marca pre-release y pulsa Publish. **No cambies el nombre de la etiqueta en el borrador**: los archivos se compilaron con la versión de `Cargo.toml`, y otra etiqueta hace que el programa informe de una versión y la publicación de otra (y la comprobación de actualizaciones ofrece una sin fin). El workflow rechaza una etiqueta que ya apunta a otro commit.

## Si se publicó un número por error

No lo reutilices. Borra a mano esa versión y su etiqueta y publica el número siguiente.

## No crees nunca la versión a mano

El workflow `build` crea él mismo el borrador, con todos los archivos adjuntos. Una versión creada a mano con la misma etiqueta (incluso como borrador) está vacía y hace fallar el paso final. Si ocurre: borra la versión vacía (conserva la etiqueta) y usa **Re-run failed jobs** en la ejecución `build`.
