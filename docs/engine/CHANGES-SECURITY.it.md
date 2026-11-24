# Modifiche di sicurezza e robustezza

[English](CHANGES-SECURITY.md) · [Română](CHANGES-SECURITY.ro.md) · [Français](CHANGES-SECURITY.fr.md) · [Italiano](CHANGES-SECURITY.it.md) · [Español](CHANGES-SECURITY.es.md) · [Português](CHANGES-SECURITY.pt.md) · [Deutsch](CHANGES-SECURITY.de.md) · [Magyar](CHANGES-SECURITY.hu.md)

Altri documenti: [ENGINE.it.md](ENGINE.it.md) · [QUICKSTART.it.md](QUICKSTART.it.md)

Questa pagina elenca ciò che la revisione avversariale del motore e di `corbul` ha trovato e ciò che è stato fatto, così da poter giudicare quanto fidarti di una funzione. È un elenco di correzioni e limiti, non una certificazione di sicurezza: nulla di quanto segue è stato verificato da terzi.

## Server

- **Richieste cross-site.** Le chiamate che modificano lo stato controllano `Origin` e richiedono un `Content-Type` JSON, così una pagina web che visiti non può far eseguire al tuo browser un POST verso un `corbul serve` locale. I browser sono ammessi solo dalle origini che indichi con `--cors`.
- **Rotte di amministrazione.** `/api/create`, `/api/delete`, `/models/load` e `/models/unload` richiedono il token principale. I titolari di chiavi API possono usarle solo se avvii il server con `--allow-admin`.
- **Segreti fuori dalla riga di comando.** `--token-file`, `--key-file`, `--receipt-key-file` e le variabili `CORBUL_TOKEN` e `CORBUL_MESH_KEY` tengono i segreti fuori dall'elenco dei processi. Le opzioni semplici `--token`, `--key` e `--receipt-key` funzionano ancora e sono visibili agli altri utenti della macchina.
- **Limiti di risorse.** `--max-kv` rifiuta le richieste la cui cache KV sarebbe troppo grande (predefinito 8g); `--max-heavy` limita le richieste simultanee di tokenizzazione, template e verifica (predefinito 4); i limiti di frequenza di `--keys-file` coprono ogni rotta che costa calcolo; il decoder del corpo a blocchi è lineare; `POST /v1/quotes/verify` pone limiti alla dimensione di testo e fonti.
- **Log.** Il testo dei prompt non viene registrato a meno che non si passi `--log-prompts`; `--no-log` disattiva il log.
- **Interfaccia web.** La pagina stessa è pubblica (non contiene segreti); ogni chiamata che fa richiede il token.
- **Limite che resta:** il server parla http in chiaro. Associalo a `127.0.0.1` (il valore predefinito) oppure metti TLS davanti.

## Mesh

- Handshake HMAC-SHA256 reciproco con una chiave condivisa di almeno 8 caratteri, un MAC su ogni frame, limiti sulle dimensioni di handshake e frame.
- La scoperta UDP è opzionale (`--discovery`, `--discovery-port`) e le risposte non sono più grandi della sonda, quindi un worker non può essere usato come amplificatore.
- **Limite che resta:** il traffico è autenticato ma non cifrato; la chiave è condivisa da tutti nel mesh; provato solo in loopback.

## File, download e fusioni

- Il parsing GGUF riceve la lunghezza reale del file (`parse_with_len`); la costruzione del tokenizer usa un trie con un limite sulla dimensione dei pezzi; i file troncati e corrotti restituiscono errori nei test avversariali.
- Download: fissaggio al SHA del commit, limiti di dimensione, rifiuto dei reindirizzamenti verso http in chiaro, ripiego anonimo su `wget` (il token non gli viene mai passato), controlli SHA-256.
- Le fusioni non seguono i link simbolici e rifiutano di sovrascrivere un output salvo `--overwrite`; forme e intervalli sono controllati prima della scrittura.
- I decoder di immagini applicano `Limits` prima di allocare e limitano l'espansione deflate; il lettore WAV ha limiti per file ostili.

## Decodifica e verifica

- La decodifica con grammatica, espressione regolare e schema JSON ha budget di lavoro; il vincolo delle citazioni è lineare con limiti e segnala `limit_hit` quando un limite lo ferma.
- Le ricevute hanno limiti di dimensione; gli snapshot possono essere sigillati con un HMAC a chiave; ricevute e snapshot usano catene SHA-256.
- Correzioni di correttezza trovate in revisione: probabilità obsolete dopo top-k/top-p (ora rinormalizzate), l'alias `typ_p`, virgole decimali nella gestione dei numeri, limiti del lookahead oltre il contesto, punteggio di gruppo DeepSeek-V2, esattezza dei lotti quantizzati (`exact_batch`).

## Non fatto o non verificato

- Nessun audit di terze parti, nessuna campagna di fuzzing oltre ai test avversariali del repository, nessun penetration test di `corbul serve`.
- `ToolStream::push` può riscansionare ripetutamente il proprio buffer (quadratico nella lunghezza dell'output fino a `--max-tokens`).
- `GrammarCache` esiste ma la CLI non lo usa ancora.
- Le licenze delle dipendenze sono state controllate l'ultima volta con `cargo-deny` al commit b496ba5.
- I workflow CI non sono mai stati eseguiti; i percorsi di codice per ARM, macOS e Windows non sono stati eseguiti qui.
