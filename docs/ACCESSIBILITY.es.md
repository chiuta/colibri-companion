# Accesibilidad

Colibri Companion quiere poder ser usado por el mayor número posible de personas. Todo lo que sigue está en **Accesibilidad**, en la barra superior (o con `Ctrl+Mayús+A`).

## Qué hay

| Necesidad | Qué hace la aplicación |
| --- | --- |
| Baja visión | Tamaño del texto del 80 % al 300 %; un aspecto de alto contraste; la tipografía Atkinson Hyperlegible (SIL OFL); más espacio entre líneas y botones; un contorno grueso alrededor del elemento donde está el teclado |
| Ceguera | La interfaz se expone a los lectores de pantalla mediante AccessKit; cada respuesta se puede leer en voz alta con la voz del propio ordenador (botón, `Ctrl+Mayús+S` o automáticamente) |
| Daltonismo | Una paleta azul/naranja en lugar de verde/rojo en los tres aspectos |
| Manos y movimiento | Botones de hasta 64 puntos de alto; toda la aplicación funciona solo con el teclado; nunca se exige arrastrar ni hacer movimientos finos |
| Audición | La aplicación nunca emite sonidos ni te pide que escuches; la barra de tareas puede parpadear cuando una respuesta está lista Esc cierra la ventana abierta, incluso desde un cuadro de texto, y ninguna ventana se sale de la pantalla. |
| Dislexia, atención, carga cognitiva | Tipografía legible, líneas más espaciadas, sin animaciones, menús sencillos y un estilo de respuesta en *lenguaje sencillo* (frases cortas, palabras de todos los días) |

Siete preajustes (baja visión, ceguera, movimientos difíciles de controlar, dislexia, lectura más fácil, daltonismo, audición) fijan conjuntos razonables con un clic; después cada opción se puede cambiar por separado.

## Cómo se comprueba

* Cada pareja de colores de texto y fondo que usa la interfaz, en los tres aspectos y con la paleta para daltónicos activada o no, debe alcanzar un contraste de 4,5 : 1 (WCAG AA para texto normal). Es una prueba unitaria (`theme::tests::every_look_is_readable`), así que un cambio de color que la rompa hace fallar la compilación.
* Los colores «bueno/malo» de la paleta para daltónicos también deben diferir en luminosidad, para seguir siendo distinguibles en escala de grises.
* El texto enviado al programa de voz viaja por la entrada estándar o en una variable de entorno, nunca en la línea de comandos, así que una respuesta no puede confundirse con una opción (probado de forma unitaria).

## Lo que **no** se ha comprobado

* La aplicación no se ha probado con un lector de pantalla real (NVDA, JAWS, Narrator, VoiceOver, Orca). El árbol de accesibilidad viene de egui/AccessKit; los botones con solo icono pueden anunciarse mal. Los informes son bienvenidos.
* La lectura en voz alta usa `espeak-ng`/`espeak`/`spd-say` en Linux, `say` en macOS y System.Speech mediante PowerShell en Windows. Solo se prueba la construcción del comando; la voz en sí no se ha oído aquí. En macOS y Windows el idioma de la voz es el elegido en el sistema.
* La *entrada* de voz se deja al dictado del sistema operativo (Windows `Win+H`, macOS `Fn` dos veces). El dictado sin conexión dentro de la aplicación (whisper.cpp) está previsto en `ROADMAP-INTERCONNECT.md`.
* El barrido por pulsador y el seguimiento ocular no están implementados; las herramientas del sistema operativo funcionan sobre la compatibilidad con el teclado.
