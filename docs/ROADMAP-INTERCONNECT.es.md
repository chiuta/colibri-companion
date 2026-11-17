# Hoja de ruta de la interconexión: de una ventana de chat a un asistente capaz

Objetivo: conectar el modelo local con todo lo que multiplica lo que puede hacer, mientras la persona sigue al mando. Cada paso mantiene tres reglas: **primero lo local, nada sale sin un sí, toda acción puede verse y deshacerse.**

## 0. El marco de seguridad (se construye primero, lo usa todo lo demás)
- **Niveles de capacidad** (véase `platforms.rs`): Leer → Buscar → Escribir → Notificar → Actuar. Una conexión se crea en un nivel y nunca puede superarlo.
- **Tarjeta de aprobación** para cada escritura, envío y acción: muestra exactamente qué ocurrirá, con *Permitir una vez / Permitir para esta automatización / Rechazar*. Leer no necesita tarjeta.
- **Contenido no fiable** (páginas web, incidencias, correos, documentos) se encierra y se etiqueta como datos. Una llamada a una herramienta que sigue a contenido no fiable siempre vuelve a preguntar. Es la defensa principal contra la inyección de prompts.
- **Presupuestos y frenos**: pasos, llamadas al modelo, minutos y bytes por ejecución; una tecla *Detenerlo todo*.
- **Registro de auditoría** de cada petición que salió del ordenador y de cada acción realizada, en un archivo sencillo.
- **Secretos**: los tokens viven en memoria por defecto; guardarlos es una decisión explícita.

## 1. Herramientas para el modelo (el multiplicador)
Colibri expone la llamada a herramientas en la mayoría de los motores. El programa ofrece al modelo una caja pequeña y honesta: página web, búsqueda, archivo de repositorio, búsqueda en carpeta, calculadora, Python en un entorno aislado, consulta de tablas, búsqueda en notas, guardar archivo, webhook. El modelo pide, el programa comprueba nivel y aprobación, ejecuta y devuelve el resultado como datos.

## 2. MCP: un solo enchufe para miles de herramientas
Hablar el Model Context Protocol como cliente (y mantener `coli mcp` como servidor). Cualquier servidor MCP (archivos, bases de datos, navegadores, calendarios, herramientas de diseño, dispositivos del hogar) pasa a estar disponible bajo las mismas reglas de aprobación. Es el mayor salto de alcance con el menor código.

## 3. Memoria y conocimiento que crecen contigo
- Índice local (BM25 ahora, embeddings después) sobre carpetas, notas, chats guardados, código, archivos de correo.
- Las respuestas citan el pasaje que usaron; «muéstrame dónde» lo abre.
- Bibliotecas sin conexión: Wikipedia mediante Kiwix, instantáneas de arXiv y Wikidata, manuales, leyes.
- La *constelación*: recuerdos como estrellas enlazadas (etiquetas, personas, proyectos) que puedes ver y editar.

## 4. Sentidos y voz (todo local)
- Oído: whisper.cpp para dictado y transcripciones de reuniones.
- Habla: Piper o la voz del sistema, según el idioma de la interfaz.
- Vista: modelos que aceptan imágenes (colibri indica cuáles), más OCR (Tesseract) para escaneos y fotos de papel, más la cámara o una captura de pantalla bajo demanda.
- Creación: Qwen-Image mediante colibri, ComfyUI o Stable Diffusion para imágenes; herramientas de audio y vídeo según vayan apareciendo.

## 5. Varias mentes
- **Duelo**: A/B a ciegas con Elo, para saber qué modelo es el mejor *para ti*.
- **Consejo**: varios modelos responden, un modelo juez los fusiona y señala los desacuerdos.
- **Enrutador**: un modelo pequeño y rápido para las preguntas fáciles, el grande para las difíciles, elegido por un modelo de decisión (colibri incluye Laya, GLiNER2.5-Decide, Clef).
- **Especialistas**: código, traducción, visión, cada uno elegido automáticamente, cada uno reemplazable.

## 6. Exactitud: que el modelo se compruebe a sí mismo
- Ejecutar en un entorno aislado el Python que escribe y mostrar la salida real.
- Pruebas unitarias del código que produce, ejecutadas antes de ofrecerlo.
- Autoconsistencia (preguntar dos veces, comparar), citas obligatorias para los hechos, un interruptor «muestra tus fuentes» y un pequeño banco de evaluación con intervalos de confianza (como hace LocalMind).

## 7. Agentes y automatizaciones
Cadenas guardadas de pasos (leer, preguntar, comprobar, guardar, notificar) con disparadores: a mano, según un horario, cuando cambia una carpeta, cuando llega un webhook. Plantillas: resumen de la mañana, borrador de triaje de la bandeja de entrada, vigilante de repositorio, notas de reunión, informe de investigación, revisión semanal. Cada paso muestra su nivel.

## 8. Dispositivos y mundo físico (al final, y con el mayor cuidado)
Home Assistant, MQTT, dispositivos serie, impresoras 3D, cámaras. Solo lectura por defecto; acciones mediante una lista de permitidos; nunca cerraduras, alarmas ni nada crítico para la seguridad.

## 9. Varios ordenadores, un solo cerebro
Compartir un modelo por la red doméstica (un PC potente lo sirve, un portátil lo usa), sincronizar prompts, memoria y ajustes entre máquinas (cifrado, de igual a igual, sin nube) y repartir el trabajo entre varias GPU donde el motor lo permita.

## 10. Que sea sin esfuerzo
Una guía en el primer arranque que elige un modelo para el hardware, un *modo sencillo* con cinco botones, conexiones de un clic desde el catálogo, sugerencias «¿qué puedo hacer con esto?», un panel visible «lo que el modelo ve ahora» y todo en ocho idiomas.

## Orden de trabajo
1. Carpetas y archivos grandes, biblioteca de prompts, estilos de respuesta (hecho).
2. Acceso en línea, GitHub/GitLab/Gitea/páginas web, el catálogo (en curso).
3. Tarjetas de aprobación, registro de auditoría, presupuestos (el marco de seguridad).
4. Automatizaciones con webhooks y notificaciones.
5. Duelo, consejo, enrutador.
6. Herramientas para el modelo, luego cliente MCP.
7. Índice y memoria, voz y OCR.
8. Dispositivos, varias máquinas.

Lo que *no* se promete: nada de esto se ha probado aún con motores colibri reales con llamada a herramientas, y cada servicio externo cambia su API con el tiempo; las entradas marcadas «previsto» en el catálogo son planes, no funciones.
