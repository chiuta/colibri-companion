# Colibri Companion

[English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · **Italiano** · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

**Sito web: https://colibri-companion.chiuta.ro** — download, video tutorial, aiuto.

Un'app desktop semplice per [colibri](https://github.com/JustVugg/colibri): scegli un modello adatto al tuo computer, scaricalo con un clic, poi conversa — il modello gira sul tuo computer, senza account e senza servizi cloud.

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

## corbul: il motore clean-room (alpha)

Oltre all'app desktop, il repository contiene un motore di inferenza scritto da zero in Rust e `corbul`, la sua riga di comando e il suo server locale (stili OpenAI, Ollama e Anthropic, con un'interfaccia web su `/`). Non sostituisce colibri nell'app.

```bash
cargo build --release -p colibri-rs-cli      # target/release/corbul
corbul capabilities                          # what works and what does not
corbul generate ./model --prompt "Hello"     # a folder with safetensors or a .gguf file
```

**Stato, in tutta onestà:** provato solo su minuscoli modelli sintetici e su file di riferimento del tokenizer (12 famiglie di modelli, 46 casi su 46 uguali ai dati di riferimento di llama.cpp). Nessun modello reale scaricato, nessuna GPU e nessuna macchina Windows, macOS o ARM lo ha ancora eseguito; le velocità sono stime. I modelli densi e mixture-of-experts vengono caricati; Mamba, RWKV e Jamba/Bamba girano come semplice continuazione; l'attenzione in stile DeepSeek, le immagini, l'audio e la GPU non sono disponibili in `corbul`; gli embedding funzionano, e il reranking funziona con un cross-encoder BERT tramite `--encoder`. Le i-quant diverse da IQ4_NL e IQ4_XS e gli encoder in stile RoBERTa non sono supportati.

Documentazione: [che cos'è e che cosa non fa](docs/engine/ENGINE.it.md) · [avvio rapido e tutte le opzioni](docs/engine/QUICKSTART.it.md) · [funzioni diverse](docs/engine/INNOVATIONS.it.md) · [processo clean-room](docs/engine/CLEANROOM.it.md) · [modifiche di sicurezza](docs/engine/CHANGES-SECURITY.it.md).

## Licenze

- Questo progetto: Apache-2.0 (`LICENSE`, `NOTICE`), come colibri.
- Dipendenze: solo licenze permissive, verificate con `cargo deny check licenses` (`deny.toml`); elenco completo in `THIRD_PARTY_LICENSES.md`. Caratteri: Inter e JetBrains Mono (SIL OFL 1.1, in `licenses/`).
- I modelli hanno licenze proprie; l'app le mostra e chiede conferma prima del download.

## Piattaforme

Linux (x86-64, ARM64; Wayland e X11), Windows 10/11 e macOS (Intel e Apple Silicon) — un solo codice sorgente, rendering OpenGL (funziona anche su vecchie schede grafiche e in macchine virtuali). `.github/workflows/build.yml` compila e testa ogni variante e produce archivi pronti da distribuire. Su Linux, `packaging/linux/install.sh` aggiunge l'app al menu. Scalatura HiDPI, scorciatoie Ctrl/⌘+N, Ctrl/⌘+, ed Esc, supporto per lettori di schermo (AccessKit). Compilazioni e test automatici coprono tutti e tre i sistemi; le prove manuali su veri computer Windows e macOS sono ancora limitate e le etichette per i lettori di schermo (AccessKit) non sono state ancora provate con lettori reali.

Linux più vecchi (Debian 8+, Ubuntu 14.04+, CentOS 7+): build aggiuntive `linux-x86_64-glibc2.17` e `linux-aarch64-glibc2.17`, collegate a glibc 2.17; la CI verifica la base dei simboli e carica il programma in Ubuntu 16.04. Le librerie di finestre e grafica arrivano comunque dal tuo sistema.

**Documenti e strumenti:** aggiungi file (**Altro → Documenti**) e poni domande: l'app li cerca sul tuo computer (parole chiave, senza cloud) e mostra sotto ogni risposta i passaggi utilizzati. **Altro → Strumenti MCP** collega server MCP che tu aggiungi e avvii; il modello può richiedere uno strumento e ogni chiamata richiede la tua approvazione (funzione iniziale, dipende da quanto il modello segue il formato).

Modelli: conservane più d'uno sul computer, passa dall'uno all'altro o aggiungine altri da un'unica finestra (**Modelli**); per primo viene proposto il più capace che il tuo computer può eseguire (con poca RAM gli esperti vengono letti dall'SSD). Velocità: CUDA per NVIDIA, Vulkan per gli altri chip grafici, altrimenti il processore, scelti automaticamente. **Risorse** mostra in tempo reale processore, memoria, scheda grafica e disco, più la prova che nulla esce dal tuo computer (tutte le connessioni aperte dell'app e del motore, con il comando per verificare in autonomia). Se colibri manca viene scaricato per te (senza git), e un controllo giornaliero (disattivabile) ti avvisa quando colibri o questa app hanno una nuova versione.

**Il tuo computer, analizzato:** in **Risorse**, l'app trova da sola le schede video e dice in modo semplice se il modello sta in memoria e perché può essere lento (swap, altri programmi, scheda inutilizzata). Si limita a consigliare; non cambia impostazioni da sola.

**Strumenti più recenti (tutti provati senza un vero modello; il sito dice cosa è verificato e cosa no):** un intervallo di velocità sotto ogni modello prima di scaricarlo (dalla tua memoria, dal disco e dal catalogo; mai una misura) e la registrazione ROUTE_TRACE con un'analisi di cosa farebbe risparmiare letture dal disco; la regolazione automatica di memoria fissata e thread (**Risorse**); **Modelli leggeri per la tua memoria** nella finestra Modelli (11 piccoli modelli, filtro per chiamata di strumenti documentata, copia del comando `ollama run`); **Verifica le citazioni** e **File scansionati (LegatorOCR)** nella finestra Documenti (il secondo richiede il programma separato LegatorOCR, licenza MIT; il testo OCR non è una trascrizione verificata); e un pannello **Consumo** accanto a ogni risposta (tempo, primo token, token in ingresso e in uscita, velocità attuale, media e massima, contesto usato, processore, memoria, swap e scheda video di questo computer; le cifre con ≈ sono stime). La sezione **Bibliografia** della finestra Documenti converte BibTeX, RIS e CSL JSON e copia un elenco di riferimenti (APA, Harvard, IEEE); provata solo su voci generate. Il resto del porting da TEZAUR (recupero di passaggi, ricevuta di provenienza, dichiarazione d'uso dell'IA) è ancora una libreria che la finestra non usa. Altro in `docs/`, ogni file in tutte e otto le lingue (`NOME.md`, `NOME.ro.md`, `NOME.fr.md`, `NOME.it.md`, `NOME.es.md`, `NOME.pt.md`, `NOME.de.md`, `NOME.hu.md`). Novità in alpha.7: una sezione **Citazioni** nella finestra Documenti dà a ogni documento una citazione (`[@nome]`) che la verifica delle citazioni riconosce, e copia l'intero elenco come BibTeX, RIS, CSL JSON, APA, Harvard o IEEE; le finestre non superano più lo schermo (il contenuto scorre e la barra del titolo con il pulsante di chiusura resta raggiungibile) ed Esc chiude la finestra aperta anche da una casella di testo.

## Lingue

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — scelte dalle Impostazioni, con `--lang en|ro|fr|it|es|pt|de|hu`, o in base alla lingua del sistema al primo avvio. Le traduzioni sono in `crates/app/src/i18n.rs` (la chiave è il testo inglese); un test verifica che ogni testo dell'interfaccia abbia tutte le traduzioni. Sono state fatte con l'aiuto di un'IA e non sono state riviste da madrelingua.

## Sviluppo

```bash
cargo test --workspace   # include un test end-to-end con un `coli` finto (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```

Nuovo: ricerca nelle conversazioni (Ctrl/⌘+K), rinomina (clic destro), modifica dell'ultimo messaggio, file di testo trascinati nel messaggio e salvataggio in .md.
