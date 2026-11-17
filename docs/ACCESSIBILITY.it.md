# Accessibilità

Colibri Companion vuole essere utilizzabile da quante più persone possibile. Tutto quanto segue si trova in **Accessibilità**, nella barra in alto (o con `Ctrl+Maiusc+A`).

## Cosa c'è

| Esigenza | Cosa fa l'app |
| --- | --- |
| Ipovisione | Dimensione del testo da 80% a 300%; un aspetto ad alto contrasto; il carattere Atkinson Hyperlegible (SIL OFL); più spazio tra righe e pulsanti; un contorno spesso attorno all'elemento su cui si trova la tastiera |
| Cecità | L'interfaccia è esposta agli screen reader tramite AccessKit; ogni risposta può essere letta ad alta voce con la voce del computer (pulsante, `Ctrl+Maiusc+S` o in automatico) |
| Daltonismo | Una tavolozza blu/arancione al posto di verde/rosso in tutti e tre gli aspetti |
| Mani e movimento | Pulsanti alti fino a 64 punti; tutta l'app funziona solo con la tastiera; non servono mai trascinamenti o movimenti fini |
| Udito | L'app non emette mai suoni e non ti chiede mai di ascoltare; la barra delle applicazioni può lampeggiare quando una risposta è pronta |
| Dislessia, attenzione, carico cognitivo | Carattere leggibile, righe più distanziate, nessuna animazione, menu semplici e uno stile di risposta in *linguaggio semplice* (frasi brevi, parole di ogni giorno) |

Sette preset (ipovisione, cecità, movimenti difficili da controllare, dislessia, lettura più facile, daltonismo, udito) impostano insiemi sensati con un clic; poi ogni opzione si può cambiare da sola.

## Come viene verificata

* Ogni coppia di colori testo e sfondo usata dall'interfaccia, in tutti e tre gli aspetti e con la tavolozza per daltonici attiva o no, deve raggiungere il contrasto 4,5 : 1 (WCAG AA per il testo normale). È un test unitario (`theme::tests::every_look_is_readable`): una modifica di colore che lo rompe fa fallire la compilazione.
* I colori «buono/cattivo» della tavolozza per daltonici devono differire anche per luminosità, così restano distinguibili in scala di grigi.
* Il testo inviato al programma di sintesi vocale viaggia sullo standard input o in una variabile d'ambiente, mai nella riga di comando, quindi una risposta non può essere scambiata per un'opzione (testato).

## Cosa **non** è stato verificato

* L'app non è stata provata con un vero screen reader (NVDA, JAWS, Narrator, VoiceOver, Orca). L'albero di accessibilità viene da egui/AccessKit; i pulsanti con la sola icona possono essere annunciati male. I segnalamenti sono benvenuti.
* La lettura ad alta voce usa `espeak-ng`/`espeak`/`spd-say` su Linux, `say` su macOS e System.Speech tramite PowerShell su Windows. È testata solo la costruzione del comando; la voce in sé non è stata ascoltata qui. Su macOS e Windows la lingua della voce è quella scelta nel sistema.
* L'*input* vocale è lasciato alla dettatura del sistema operativo (Windows `Win+H`, macOS `Fn` due volte). La dettatura offline nell'app (whisper.cpp) è prevista in `ROADMAP-INTERCONNECT.md`.
* La scansione per interruttore e il tracciamento dello sguardo non sono implementati; gli strumenti del sistema operativo funzionano sopra il supporto da tastiera.
