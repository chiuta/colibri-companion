# Barrierefreiheit

Colibri Companion soll von möglichst vielen Menschen benutzt werden können. Alles Folgende erreichst du über **Barrierefreiheit** in der oberen Leiste (oder mit `Strg+Umschalt+A`).

## Was es gibt

| Bedarf | Was die App tut |
| --- | --- |
| Sehschwäche | Textgröße von 80 % bis 300 %; ein kontrastreiches Erscheinungsbild; die Schrift Atkinson Hyperlegible (SIL OFL); mehr Abstand zwischen Zeilen und Schaltflächen; eine dicke Umrandung um das Element, auf dem die Tastatur steht |
| Blindheit | Die Oberfläche wird Screenreadern über AccessKit bereitgestellt; jede Antwort kann mit der Stimme des Computers vorgelesen werden (Schaltfläche, `Strg+Umschalt+S` oder automatisch) |
| Farbenblindheit | Eine blau/orange Palette statt grün/rot in allen drei Erscheinungsbildern |
| Hände und Bewegung | Schaltflächen bis 64 Punkte Höhe; die ganze App funktioniert allein mit der Tastatur; es wird nie Ziehen oder Feinmotorik verlangt |
| Hören | Die App macht nie ein Geräusch und verlangt nie, dass du zuhörst; die Taskleiste kann blinken, wenn eine Antwort fertig ist |
| Legasthenie, Aufmerksamkeit, kognitive Belastung | Gut lesbare Schrift, lockerere Zeilen, keine Animation, einfache Menüs und ein Antwortstil in *einfacher Sprache* (kurze Sätze, Alltagswörter) |

Sieben Voreinstellungen (Sehschwäche, Blindheit, schwer kontrollierbare Bewegungen, Legasthenie, leichteres Lesen, Farbenblindheit, Hören) stellen sinnvolle Bündel mit einem Klick ein; danach lässt sich jede Option einzeln ändern.

## Wie es geprüft wird

* Jedes Paar aus Text- und Hintergrundfarbe, das die Oberfläche verwendet, muss in allen drei Erscheinungsbildern und mit ein- oder ausgeschalteter Palette für Farbenblinde das Kontrastverhältnis 4,5 : 1 erreichen (WCAG AA für normalen Text). Das ist ein Unit-Test (`theme::tests::every_look_is_readable`); eine Farbänderung, die ihn bricht, lässt den Build scheitern.
* Die Farben „gut/schlecht“ der Palette für Farbenblinde müssen sich auch in der Helligkeit unterscheiden, damit sie in Graustufen unterscheidbar bleiben.
* Der Text, der an das Sprachprogramm geht, läuft über die Standardeingabe oder eine Umgebungsvariable, nie über die Befehlszeile; eine Antwort kann daher nicht für eine Option gehalten werden (per Unit-Test geprüft).

## Was **nicht** überprüft wurde

* Die App wurde nicht mit einem echten Screenreader (NVDA, JAWS, Narrator, VoiceOver, Orca) ausprobiert. Der Barrierefreiheitsbaum kommt von egui/AccessKit; Schaltflächen nur mit Symbol werden womöglich schlecht angesagt. Rückmeldungen sind willkommen.
* Das Vorlesen nutzt `espeak-ng`/`espeak`/`spd-say` unter Linux, `say` unter macOS und System.Speech über PowerShell unter Windows. Nur der Aufbau des Befehls ist getestet; die Sprache selbst wurde hier nicht gehört. Unter macOS und Windows ist die Stimmensprache die im System gewählte.
* Spracheingabe überlassen wir dem Diktat des Betriebssystems (Windows `Win+H`, macOS zweimal `Fn`). Offline-Diktat in der App (whisper.cpp) ist in `ROADMAP-INTERCONNECT.md` geplant.
* Switch-Scanning und Eye-Tracking sind nicht umgesetzt; die Werkzeuge des Betriebssystems arbeiten auf der Tastaturunterstützung auf.
