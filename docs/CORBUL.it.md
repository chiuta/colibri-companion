# Eseguire file GGUF con corbul

Colibri importa solo le famiglie di modelli che implementa (Qwen3.x, DeepSeek, GLM, Mistral-Large, MiMo e poche altre). Un GGUF di un'altra famiglia, per esempio Gemma, non si importa. Colibri Companion prova allora il proprio motore, **corbul**, che legge direttamente il `.gguf`.

- In *I miei modelli → Carica modelli GGUF da una cartella* ogni file ha il pulsante *Esegui con corbul* (appare quando `corbul` è accanto all'app o nel PATH; gli archivi di rilascio lo includono).
- Quando un'importazione fallisce, il messaggio indica il file, l'architettura dichiarata e le ultime righe del registro, e corbul viene provato automaticamente.
- corbul ascolta solo sul loopback, su una porta libera, con lo scaffale dei pacchetti di contesto nella cartella dati dell'app (vedi CONTEXT-PACKS). Si ferma quando fermi il modello, ne avvii un altro o chiudi l'app. Il registro si vede con *Mostra il registro*.
- Limiti: corbul è più lento di colibri sui modelli molto grandi, senza streaming degli esperti dal disco, e supporta le famiglie del proprio motore; se non riesce a caricare il file, il messaggio dice perché.
