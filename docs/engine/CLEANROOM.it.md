# Processo clean-room

[English](CLEANROOM.md) · [Română](CLEANROOM.ro.md) · [Français](CLEANROOM.fr.md) · [Italiano](CLEANROOM.it.md) · [Español](CLEANROOM.es.md) · [Português](CLEANROOM.pt.md) · [Deutsch](CLEANROOM.de.md) · [Magyar](CLEANROOM.hu.md)

Una nota breve e onesta su come è stato scritto il motore in `crates/engine` e su che cosa «clean-room» significa e non significa qui.

## Le regole seguite

1. **Fonti.** Solo specifiche e articoli pubblici: le strutture dei file GGUF e safetensors, i campi `config.json` e `tokenizer.json` di Hugging Face, le descrizioni pubblicate delle strutture dei blocchi di quantizzazione e gli articoli dietro ogni tecnica (varianti di attenzione, instradamento MoE, Mamba, RWKV, decodifica speculativa, H2O, StreamingLLM, SnapKV, TIES, DARE, filigrana a lista verde, HNSW, BM25…).
2. **Niente sbirciate.** Il codice sorgente di colibri, llama.cpp e altri motori di inferenza non è stato letto né copiato scrivendo il motore. Quando una struttura non era chiara, il lavoro si è fermato alla descrizione pubblica e l'incertezza è scritta nel codice e nella documentazione.
3. **Oracoli invece di imitazione.** Ogni kernel veloce è provato contro una versione scalare semplice scritta per prima; ogni quantizzatore contro blocchi calcolati a mano e andata e ritorno; i percorsi divisi (mesh, streaming, cache) contro il percorso su una sola macchina, bit per bit dove lo si promette.
4. **Solo licenze permissive.** Il motore dipende da `serde_json` e `memmap2` e da nient'altro; il crate GPU aggiunge `wgpu` dietro una funzione. Le licenze sono state controllate con `cargo-deny` a un commit precedente; le modifiche successive non sono ancora state ricontrollate.
5. **L'input ostile è normale.** Ogni dimensione o offset letto da un file viene controllato contro la lunghezza reale del file prima di qualsiasi allocazione; test di tipo fuzz forniscono file troncati e corrotti e devono ricevere un errore, non un crash.

## Che cosa non affermiamo

- Il clean-room è un processo, non una garanzia legale. Il modello che ha scritto il codice, come ogni modello, è stato addestrato su testo pubblico che può includere codice open source, e non possiamo escludere che abbia visto i progetti citati. Non li abbiamo consultati e abbiamo cercato di renderlo visibile tenendo nel codice oracoli, test e fonte di ogni formula.
- Diverse formule (Mamba, RWKV-6/7, MLA, extra di imatrix e GGUF) sono state scritte da descrizioni pubblicate e dal loro ricordo, e controllate solo per coerenza interna, perché non c'era alcun checkpoint reale. Possono differire dal riferimento nei dettagli; trattale come non verificate finché qualcuno non le esegue su modelli reali.
- La compatibilità con i file di altri motori è un obiettivo, non un fatto verificato (vedi la tabella in [ENGINE.it.md](ENGINE.it.md)).

Se trovi un punto in cui il motore riproduce troppo da vicino il codice di qualcun altro, apri una issue; lo riscriveremo.
