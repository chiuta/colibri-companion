# Ridurre la cache e rimettere a punto i valori

`corbul kvfit MODEL --text FILE --question Q` misura quanto costa tenere solo una frazione delle righe della cache di un testo (scelte da SnapKV), con e senza **messa a punto dei valori** che restano.

Gli ultimi `--window` token del testo vengono registrati come interrogazioni di sonda. I valori delle righe tenute vengono poi spostati il meno possibile perché l'attenzione su di esse dia, sulle sonde, ciò che dava l'attenzione su tutte le righe (minimi quadrati regolarizzati, forma duale, `crates/engine/src/kvfit.rs`). L'aggiustamento non può mai essere peggiore dei valori semplici *sulle sonde*; se aiuti su una domanda non sondata dipende dal modello e dal testo, e solo questo comando, eseguito sul tuo modello, te lo dice.

Opzioni: `--budget 0.25` (frazione di righe tenute), `--window 16`, `--ridge 0.01`, `--max-new 32`. Mostra l'errore sulle sonde, la massima differenza dei primi punteggi rispetto alla cache intera, se il primo token coincide e quanti dei primi token della risposta coincidono.

Limiti: solo i valori (le chiavi portano la posizione e restano; il peso delle righe tolte non è modellato), i livelli a finestra scorrevole restano com'erano, l'aggiustamento costa tempo proporzionale a righe × sonde per testa. Ora unito ai pacchetti di contesto: vedi SHRINK-MAP.it.md (`--shrink`). Nulla misurato su un modello reale.

I pacchetti di contesto si possono anche tenere a metà dimensione: `corbul packs --pack-kv f16` e `corbul serve --packs-kv f16` (arrotondamento F16 su ogni riga), e comprimere senza perdite: `--pack-compress`, `--packs-compress`.

In Colibri Companion: *Documenti → Pacchetti di contesto (corbul)* ha le opzioni F16 e compressione (applicate al prossimo avvio di corbul) e *Misura la riduzione della cache…*, che esegue questo comando sul modello che corbul sta eseguendo (carica il modello una seconda volta).
