# Ejecutar archivos GGUF con corbul

Colibri solo importa las familias de modelos que implementa (Qwen3.x, DeepSeek, GLM, Mistral-Large, MiMo y alguna más). Un GGUF de otra familia, Gemma por ejemplo, no se importa. Colibri Companion prueba entonces su propio motor, **corbul**, que lee el `.gguf` directamente.

- En *Mis modelos → Cargar modelos GGUF desde una carpeta* cada archivo tiene el botón *Ejecutar con corbul* (aparece cuando `corbul` está junto a la aplicación o en el PATH; los archivos de versión lo incluyen).
- Cuando una importación falla, el mensaje nombra el archivo, su arquitectura declarada y las últimas líneas del registro, y se prueba corbul automáticamente.
- corbul escucha solo en el loopback, en un puerto libre, con el estante de paquetes de contexto en la carpeta de datos de la aplicación (véase CONTEXT-PACKS). Se detiene cuando detienes el modelo, inicias otro o cierras la aplicación. El registro se ve con *Mostrar el registro*.
- Límites: corbul es más lento que colibri en modelos muy grandes, sin streaming de expertos desde disco, y admite las familias de su propio motor; si no puede cargar el archivo, el mensaje dice por qué.
