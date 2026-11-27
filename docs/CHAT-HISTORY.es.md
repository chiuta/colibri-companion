# Archivo del historial de conversaciones

El historial de conversaciones es un único archivo JSON sencillo, para que Colibri Companion, CartaColibri y cualquier script puedan intercambiarlo. En la aplicación: la barra lateral, «Historial: exportar / importar».

```json
{
  "format": "colibri-chat-history",
  "version": 1,
  "app": "colibri-companion",
  "exported": 1790000000,
  "conversations": [
    {
      "id": "c1790000000-000000001",
      "title": "…",
      "created": 1790000000,
      "updated": 1790000500,
      "messages": [
        { "role": "user", "content": "…" },
        { "role": "assistant", "content": "…", "reasoning": "…", "sources": ["…"] }
      ]
    }
  ]
}
```

## Reglas para quien escribe y quien lee

- `format` debe ser `colibri-chat-history`; `version` es `1`. Un lector rechaza una versión mayor.
- Los tiempos son segundos desde 1970 (UTC). `id` solo usa letras, cifras, `-` y `_` (máximo 80); lo demás se quita.
- `role` es `user`, `assistant` o `system`. Campos opcionales del mensaje: `reasoning`, `display`, `sources` (lista de textos), `stopped` (true), `images` (solo URL `data:image/…`, y solo al exportar «con imágenes»).
- Los campos desconocidos se ignoran, así que otros programas pueden añadir los suyos.
- La importación añade los `id` nuevos y solo reemplaza un `id` conocido si `updated` del archivo es más reciente. Las conversaciones sin mensajes se omiten.
- Límites al leer: archivo 128 MB, 20.000 conversaciones, 100.000 mensajes por conversación.

Arrastrar un archivo de historial sobre la ventana lo importa.
