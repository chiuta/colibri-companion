# Pubblicare una versione

Come si fa una versione (per il manutentore)

1. **Versione.** Cambia `version` nel `Cargo.toml` principale e `TAG` in `index.html` con il nuovo numero (ad esempio `0.1.0-alpha.12` e `v0.1.0-alpha.12`), esegui `cargo update -w --offline`, aggiungi `docs/releases/<tag>.md`, poi commit e push su `main`.
2. **Compilazione.** Aspetta che le esecuzioni `build` e `engine-ci` di quel commit siano verdi. Poi avvia a mano il workflow `build` (Actions, Run workflow) e scrivi il tag, ad esempio `v0.1.0-alpha.12`. Compila tutte le piattaforme e crea una release **bozza**.
3. **Pubblicazione.** Apri la bozza, incolla `docs/releases/<tag>.md` nella descrizione, spunta pre-release e premi Publish. **Non cambiare il nome del tag nella bozza**: i file sono stati compilati con la versione di `Cargo.toml`, e un altro tag fa sì che il programma dichiari una versione e la release un'altra (e il controllo aggiornamenti ne propone una all'infinito). Il workflow rifiuta un tag che punta già a un altro commit.

## Se un numero è stato pubblicato per errore

Non riutilizzarlo. Elimina a mano quella release e il suo tag e pubblica il numero successivo.

## Non creare mai la release a mano

Il workflow `build` crea da solo la bozza, con tutti i file allegati. Una release creata a mano con lo stesso tag (anche come bozza) è vuota e fa fallire il passo finale. Se succede: elimina la release vuota (tieni il tag) e usa **Re-run failed jobs** sull'esecuzione `build`.
