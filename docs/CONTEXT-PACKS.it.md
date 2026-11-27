# Pacchetti di contesto

Riusa il lavoro fatto su un testo quando ricompare in un prompt diverso (solo motore corbul; clean-room, codice originale).

Una cache di prefisso funziona solo se il testo comune viene per primo. Un *pacchetto* è la cache KV di un testo calcolato da solo e tenuto su uno scaffale (archivio di snapshot firmato, con limite LRU).

`compose` mette i pacchetti uno dopo l'altro: le righe si copiano e le chiavi si ruotano della differenza di posizione (esatto); poi l'inizio di ogni pacchetto successivo viene ricalcolato in finestre crescenti finché il cambiamento misurato scende sotto una tolleranza. La domanda è sempre calcolata da zero.

`tolerance = 0` ricalcola tutto ed equivale a un prefill completo. I modelli con finestra scorrevole, cache espulsa o su acceleratore vengono rifiutati e tornano al prefill normale.

Non esatto finché non riparato: uno stato con righe spostate dichiara `is_exact() == false` e non viene mai scritto su disco.

Limiti: il motore colibri di JustVugg non può esportare le cache, quindi funziona solo con corbul. Provato su piccoli modelli sintetici; la qualità su modelli reali non è ancora misurata.

## Riga di comando

`corbul packs MODEL --session DIR --pack FILE… --question TESTO [--tolerance 0.02] [--probe 8] [--max-new 64] [--check]` mette ogni file sullo scaffale una sola volta, poi risponde con i pacchetti salvati, in qualsiasi ordine. `--check` esegue anche un prefill completo e mostra la differenza di punteggio e il tempo di ciascuno.

## Server

`corbul serve MODEL --packs DIR [--packs-cap 2g]` aggiunge `POST /v1/packs/complete` con `{"packs": [testo, …], "question": "…", "max_tokens": 64, "tolerance": 0.02, "probe": 8}`. La risposta ha `text`, `usage` e un oggetto `packs` (testi già sullo scaffale, righe copiate, ricalcolate, nuove e risparmiate, cambiamento misurato a ogni giunzione). Completamento di testo greedy, senza template di chat.

## In Colibri Companion

Con i documenti attivi, spunta *Riusa il calcolo*. Ogni brano trovato diventa un pacchetto etichettato dalla fonte (non dalla posizione), quindi lo stesso brano è riusato in qualsiasi ordine. Serve corbul avviato con `--packs`; con un altro motore l'opzione è ignorata e funziona la chat normale. La risposta arriva tutta insieme.
