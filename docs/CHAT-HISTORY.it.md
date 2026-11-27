# File della cronologia delle conversazioni

La cronologia delle conversazioni è un semplice file JSON, così Colibri Companion, CartaColibri e qualsiasi script possono scambiarsela. Nell'app: la barra laterale, «Cronologia: esporta / importa».

```json
{
  "format": "colibri-chat-history",
  "version": 1,
  "app": "colibri-companion",
  "exported": 1790000000,
  "conversations": [
    {
      "id": "c1790000000-000000001",
      "title": "…",
      "created": 1790000000,
      "updated": 1790000500,
      "messages": [
        { "role": "user", "content": "…" },
        { "role": "assistant", "content": "…", "reasoning": "…", "sources": ["…"] }
      ]
    }
  ]
}
```

## Regole per chi scrive e chi legge

- `format` deve essere `colibri-chat-history`; `version` è `1`. Un lettore rifiuta una versione più alta.
- I tempi sono secondi dal 1970 (UTC). `id` usa solo lettere, cifre, `-` e `_` (al massimo 80); il resto viene tolto.
- `role` è `user`, `assistant` o `system`. Campi facoltativi del messaggio: `reasoning`, `display`, `sources` (elenco di testi), `stopped` (true), `images` (solo URL `data:image/…`, e solo con l'esportazione «con immagini»).
- I campi sconosciuti vengono ignorati, quindi altri programmi possono aggiungere i propri.
- L'importazione aggiunge gli `id` nuovi e sostituisce un `id` noto solo se `updated` del file è più recente. Le conversazioni senza messaggi vengono saltate.
- Limiti in lettura: file 128 MB, 20.000 conversazioni, 100.000 messaggi per conversazione.

Trascinare un file di cronologia sulla finestra lo importa.
