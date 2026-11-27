# Pacchetti ridotti e mappe di cache tra modelli

Due estensioni dei pacchetti di contesto ([CONTEXT-PACKS.it.md](CONTEXT-PACKS.it.md), solo motore corbul; clean-room, codice originale).

## Pacchetti ridotti

Quando i testi e la domanda insieme non stanno nella finestra del modello, i pacchetti vengono rifiutati. Con la riduzione vengono accorciati: `corbul packs --shrink auto` (solo quanto serve) oppure `--shrink R` (tiene sempre circa la frazione `R`, da 0,1 a 1 delle righe di ogni pacchetto). Lo stesso vale per il campo `"shrink": "auto"` (o un numero) di `/v1/packs/complete`; la risposta elenca ciò che è stato ridotto (`"shrunk"`).

Come: gli ultimi `--shrink-window` token di un pacchetto (16 di default) vengono riletti e votano le righe da tenere (SnapKV); le prime righe si tengono sempre (lì si accumula l'attenzione); i valori rimasti vengono rifittati ([KV-FIT.it.md](KV-FIT.it.md); `--no-refit` lo disattiva). Le chiavi tenute sono poi ruotate alle posizioni dei nuovi posti, così il pacchetto diventa *esattamente la cache del suo testo accorciato*: si innesta e si ripara come ogni pacchetto e, con riparazione completa, la risposta è uguale a un semplice precaricamento del testo accorciato. I primi `--shrink-skip` testi (1 di default, di solito le istruzioni) restano interi. La riduzione avviene quando i pacchetti vengono messi insieme; lo scaffale conserva i pacchetti interi.

In Colibri Companion: *Documenti → Pacchetti di contesto (corbul)* → *Riduci i testi troppo lunghi*.

Limiti: si perde informazione, quanta dipende dal modello e dal testo (misurala con `corbul kvfit` o con la finestra *Measure cache shrinking…*); i modelli con livelli a finestra scorrevole, le cache Q8 e quelle su acceleratore sono rifiutati; se i testi non stanno nemmeno togliendo tutti i testi riducibili, la richiesta fallisce. Finora provato solo su piccoli modelli casuali.

## Mappe di cache tra modelli diversi

Un pacchetto scritto da un modello è inutile a un altro. Se i due condividono il tokenizzatore, le loro righe di cache per gli *stessi token* si possono mettere in relazione: `corbul map-fit SRC DST --text FILE --out MAPPA` legge il testo con entrambi i modelli e adatta, per livello, una mappa lineare (minimi quadrati con ridge, con la media) dalle righe di chiavi e valori di SRC a quelle di DST. Le chiavi sono confrontate senza posizione (riportate a 0 e ruotate di nuovo all'ingresso), quindi tabelle rotative e dimensioni delle teste possono differire; i livelli sono accoppiati per profondità relativa. Stampa l'errore sulle righe tenute fuori dall'adattamento accanto a quello di indovinare la media e avverte se la mappa non è migliore.

`corbul map-check SRC DST --map MAPPA --text FILE --question Q` divide il testo in due pacchetti scritti da SRC, li porta in DST attraverso la mappa e risponde in tre modi: il precaricamento completo di DST (la verità), la cache mappata con la riparazione solita (`--tolerance`, 0,02 di default) e la cache mappata senza riparazione. Stampa le righe ricalcolate e quanto ogni risposta dista dalla verità.

Una riga mappata è solo un'ipotesi: le righe importate sono segnate come approssimate e la riparazione ne ricalcola quante servono (tutte con tolleranza 0, il che equivale a un precaricamento semplice). Una mappa scadente costa quindi tempo, mai correttezza. Punto d'ingresso della libreria: `ctxpack::compose_mapped`.

Limiti: i tokenizzatori devono dare gli stessi identificatori (verificato, altrimenti non si adatta nulla); la mappa vale per una sola coppia di modelli (impronte verificate); la sua dimensione è `kd × kd' × livelli × 8` byte; l'adattamento tiene in memoria le righe di calibrazione (`--rows`, 1024 di default). Una mappa lineare tra cache di modelli non imparentati può essere scarsa; se convenga per la tua coppia si può sapere solo misurando. Non è ancora nel server né nel Companion e nulla è stato misurato su una coppia reale di modelli.
