# Colibri Companion

[English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · **Italiano** · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

**Sito web: https://colibri-companion.chiuta.ro** — download, video tutorial, aiuto.

Un'app desktop semplice per [colibri](https://github.com/JustVugg/colibri): scegli un modello adatto al tuo computer, scaricalo con un clic, poi conversa — tutto in locale, senza account e senza cloud.

## Avvio

```bash
cargo run --release -p colibri-companion            # l'app vera
cargo run --release -p colibri-companion -- --demo  # demo, senza colibri
# opzioni: --lang en|ro|fr|it|es|pt|de|hu   --version   --help
```

### I quattro passi

(1) trova Python e colibri (e propone di scaricare colibri se manca), (2) ti fa scegliere un modello (quelli che non stanno nella RAM sono nascosti; quello consigliato è segnalato), (3) scarica e installa con avanzamento, pausa e ripresa, (4) avvia il server e apre la chat.

Chat: risposte in streaming, Stop immediato, Markdown, ragionamento comprimibile, conversazioni salvate, stile (Preciso / Equilibrato / Creativo), copia e rigenera. Temi: chiaro, scuro, alto contrasto.

## Come funziona

L'app non include colibri. Esegue i comandi JSON di colibri (`setup --list --json`, `status --json`, `setup`, `start`, `stop`) e dialoga con il suo gateway compatibile OpenAI su `127.0.0.1`. Non installa mai pacchetti di sistema: se manca qualcosa, ti mostra il comando da copiare. I tuoi dati (impostazioni, conversazioni) restano in locale nella cartella dati dell'utente; `COLIBRI_COMPANION_HOME` può cambiarla.

## Licenze

- Questo progetto: Apache-2.0 (`LICENSE`, `NOTICE`), come colibri.
- Dipendenze: solo licenze permissive, verificate con `cargo deny check licenses` (`deny.toml`); elenco completo in `THIRD_PARTY_LICENSES.md`. Caratteri: Inter e JetBrains Mono (SIL OFL 1.1, in `licenses/`).
- I modelli hanno licenze proprie; l'app le mostra e chiede conferma prima del download.

## Piattaforme

Linux (x86-64, ARM64; Wayland e X11), Windows 10/11 e macOS (Intel e Apple Silicon) — un solo codice sorgente, rendering OpenGL (funziona anche su vecchie schede grafiche e in macchine virtuali). `.github/workflows/build.yml` compila e testa ogni variante e produce archivi pronti da distribuire. Su Linux, `packaging/linux/install.sh` aggiunge l'app al menu. Scalatura HiDPI, scorciatoie Ctrl/⌘+N, Ctrl/⌘+, ed Esc, supporto per lettori di schermo (AccessKit). Compilazioni e test automatici coprono tutti e tre i sistemi; le prove manuali su veri computer Windows e macOS sono ancora limitate e le etichette per i lettori di schermo (AccessKit) non sono state ancora provate con lettori reali.

**Documenti e strumenti:** aggiungi file (**Altro → Documenti**) e poni domande: l'app li cerca sul tuo computer (parole chiave, senza cloud) e mostra sotto ogni risposta i passaggi utilizzati. **Altro → Strumenti MCP** collega server MCP che tu aggiungi e avvii; il modello può richiedere uno strumento e ogni chiamata richiede la tua approvazione (funzione iniziale, dipende da quanto il modello segue il formato).

Modelli: conservane più d'uno sul computer, passa dall'uno all'altro o aggiungine altri da un'unica finestra (**Modelli**); per primo viene proposto il più capace che il tuo computer può eseguire (con poca RAM gli esperti vengono letti dall'SSD). Velocità: CUDA per NVIDIA, Vulkan per gli altri chip grafici, altrimenti il processore, scelti automaticamente. **Risorse** mostra in tempo reale processore, memoria, scheda grafica e disco, più la prova che nulla esce dal tuo computer (tutte le connessioni aperte dell'app e del motore, con il comando per verificare in autonomia). Se colibri manca viene scaricato per te (senza git), e un controllo giornaliero (disattivabile) ti avvisa quando colibri o questa app hanno una nuova versione.

**Il tuo computer, analizzato:** in **Risorse**, l'app trova da sola le schede video e dice in modo semplice se il modello sta in memoria e perché può essere lento (swap, altri programmi, scheda inutilizzata). Si limita a consigliare; non cambia impostazioni da sola.

## Lingue

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — scelte dalle Impostazioni, con `--lang en|ro|fr|it|es|pt|de|hu`, o in base alla lingua del sistema al primo avvio. Le traduzioni sono in `crates/app/src/i18n.rs` (la chiave è il testo inglese); un test verifica che ogni testo dell'interfaccia abbia tutte le traduzioni. Sono state fatte con l'aiuto di un'IA e non sono state riviste da madrelingua.

## Sviluppo

```bash
cargo test --workspace   # include un test end-to-end con un `coli` finto (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```

Nuovo: ricerca nelle conversazioni (Ctrl/⌘+K), rinomina (clic destro), modifica dell'ultimo messaggio, file di testo trascinati nel messaggio e salvataggio in .md.
