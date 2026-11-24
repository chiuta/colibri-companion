# Proceso clean-room

[English](CLEANROOM.md) · [Română](CLEANROOM.ro.md) · [Français](CLEANROOM.fr.md) · [Italiano](CLEANROOM.it.md) · [Español](CLEANROOM.es.md) · [Português](CLEANROOM.pt.md) · [Deutsch](CLEANROOM.de.md) · [Magyar](CLEANROOM.hu.md)

Una nota breve y honesta sobre cómo se escribió el motor de `crates/engine` y qué significa y qué no significa aquí «clean-room».

## Las reglas que seguimos

1. **Fuentes.** Solo especificaciones y artículos públicos: las estructuras de los archivos GGUF y safetensors, los campos `config.json` y `tokenizer.json` de Hugging Face, las descripciones publicadas de las estructuras de bloques de cuantización y los artículos detrás de cada técnica (variantes de atención, enrutamiento MoE, Mamba, RWKV, decodificación especulativa, H2O, StreamingLLM, SnapKV, TIES, DARE, marca de agua de lista verde, HNSW, BM25…).
2. **Sin mirar.** El código fuente de colibri, llama.cpp y otros motores de inferencia no se leyó ni se copió al escribir el motor. Cuando una estructura no estaba clara, el trabajo se detuvo en la descripción pública y la incertidumbre está escrita en el código y en la documentación.
3. **Oráculos en lugar de imitación.** Cada núcleo rápido se prueba frente a una versión escalar simple escrita primero; cada cuantizador frente a bloques calculados a mano y ida y vuelta; las rutas repartidas (malla, transmisión, cachés) frente a la ruta de una sola máquina, bit a bit donde se promete.
4. **Solo licencias permisivas.** El motor depende de `serde_json` y `memmap2` y de nada más; el crate de GPU añade `wgpu` tras una opción. Las licencias se comprobaron con `cargo-deny` en un commit anterior; los cambios posteriores aún no se han vuelto a comprobar.
5. **La entrada hostil es lo normal.** Cada tamaño o desplazamiento leído de un archivo se comprueba frente a la longitud real del archivo antes de reservar nada; las pruebas tipo fuzz aportan archivos truncados y corruptos y deben recibir un error, no una caída.

## Lo que no afirmamos

- Clean-room es un proceso, no una garantía legal. El modelo que escribió el código, como cualquier modelo, se entrenó con texto público que puede incluir código abierto, y no podemos descartar que haya visto los proyectos citados. No los consultamos e intentamos hacerlo visible conservando en el código los oráculos, las pruebas y la fuente de cada fórmula.
- Varias fórmulas (Mamba, RWKV-6/7, MLA, extras de imatrix y GGUF) se escribieron a partir de descripciones publicadas y de su recuerdo, y se comprobaron solo por coherencia interna, porque no había ningún punto de control real. Pueden diferir de la referencia en detalles; trátalas como no verificadas hasta que alguien las ejecute con modelos reales.
- La compatibilidad con los archivos de otros motores es un objetivo, no un hecho verificado (consulta la tabla de [ENGINE.es.md](ENGINE.es.md)).

Si encuentras un lugar donde el motor reproduce demasiado de cerca el código de otro, abre una issue; lo reescribiremos.
