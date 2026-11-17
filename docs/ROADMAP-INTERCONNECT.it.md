# Tabella di marcia dell'interconnessione: da una finestra di chat a un assistente capace

Obiettivo: collegare il modello locale a tutto ciò che moltiplica ciò che può fare, mentre la persona resta al comando. Ogni passo mantiene tre regole: **prima il locale, niente esce senza un sì, ogni azione si può vedere e annullare.**

## 0. La cornice di sicurezza (costruita per prima, usata da tutto il resto)
- **Livelli di capacità** (vedi `platforms.rs`): Leggere → Cercare → Scrivere → Notificare → Agire. Una connessione nasce a un livello e non può mai superarlo.
- **Scheda di approvazione** per ogni scrittura, invio e azione: mostra esattamente cosa accadrà, con *Consenti una volta / Consenti per questa automazione / Rifiuta*. Leggere non richiede la scheda.
- **Contenuto non fidato** (pagine web, issue, e-mail, documenti) è racchiuso e etichettato come dato. Una chiamata a strumento che segue contenuto non fidato chiede sempre di nuovo. È la difesa principale contro la prompt injection.
- **Budget e freni**: passi, chiamate al modello, minuti e byte per esecuzione; un tasto *Ferma tutto*.
- **Registro di audit** di ogni richiesta uscita dal computer e di ogni azione compiuta, in un semplice file.
- **Segreti**: i token restano in memoria per impostazione predefinita; salvarli è una scelta esplicita.

## 1. Strumenti per il modello (il moltiplicatore)
Colibri espone la chiamata a strumenti sulla maggior parte dei motori. Il programma offre al modello una piccola cassetta onesta: pagina web, ricerca, file di un repository, ricerca in una cartella, calcolatrice, Python in una sandbox, interrogazione di tabelle, ricerca nelle note, salvataggio di file, webhook. Il modello chiede, il programma controlla livello e approvazione, esegue e restituisce il risultato come dato.

## 2. MCP: una sola presa per migliaia di strumenti
Parlare il Model Context Protocol come client (e tenere `coli mcp` come server). Qualsiasi server MCP (file, database, browser, calendari, strumenti di design, dispositivi di casa) diventa disponibile con le stesse regole di approvazione. È il salto di portata più grande con meno codice.

## 3. Memoria e conoscenza che crescono con te
- Indice locale (BM25 ora, embedding poi) su cartelle, note, chat salvate, codice, archivi di posta.
- Le risposte citano il passaggio usato; «mostrami dove» lo apre.
- Biblioteche offline: Wikipedia tramite Kiwix, istantanee di arXiv e Wikidata, manuali, leggi.
- La *costellazione*: i ricordi come stelle collegate (etichette, persone, progetti) che puoi vedere e modificare.

## 4. Sensi e voce (tutto in locale)
- Udito: whisper.cpp per dettatura e trascrizioni di riunioni.
- Parola: Piper o la voce di sistema, nella lingua dell'interfaccia.
- Vista: modelli che accettano immagini (colibri indica quali), più OCR (Tesseract) per scansioni e foto di carta, più la fotocamera o uno screenshot su richiesta.
- Creazione: Qwen-Image tramite colibri, ComfyUI o Stable Diffusion per le immagini; strumenti audio e video man mano che arrivano.

## 5. Più menti
- **Duello**: A/B alla cieca con Elo, per scoprire quale modello è il migliore *per te*.
- **Consiglio**: più modelli rispondono, un modello giudice unisce e segnala i disaccordi.
- **Instradatore**: un modello piccolo e veloce per le domande facili, quello grande per le difficili, scelto da un modello di decisione (colibri include Laya, GLiNER2.5-Decide, Clef).
- **Specialisti**: codice, traduzione, visione, ciascuno scelto automaticamente, ciascuno sostituibile.

## 6. Esattezza: far controllare il modello da sé
- Eseguire in una sandbox il Python che scrive e mostrare l'output reale.
- Test unitari per il codice che produce, eseguiti prima di proporlo.
- Autoconsistenza (chiedere due volte, confrontare), citazioni obbligatorie per i fatti, un interruttore «mostra le fonti» e un piccolo banco di valutazione con intervalli di confidenza (come fa LocalMind).

## 7. Agenti e automazioni
Catene salvate di passi (leggi, chiedi, controlla, salva, notifica) con inneschi: a mano, a orario, quando cambia una cartella, quando arriva un webhook. Modelli: sintesi del mattino, bozza di smistamento della posta, osservatore di repository, appunti di riunione, nota di ricerca, revisione settimanale. Ogni passo mostra il suo livello.

## 8. Dispositivi e mondo fisico (per ultimo, e con la massima cautela)
Home Assistant, MQTT, dispositivi seriali, stampanti 3D, telecamere. Sola lettura per impostazione predefinita; azioni tramite una lista di consentiti; mai serrature, allarmi o qualsiasi cosa critica per la sicurezza.

## 9. Più computer, un solo cervello
Condividere un modello sulla rete di casa (un PC potente lo serve, un portatile lo usa), sincronizzare prompt, memoria e impostazioni tra le macchine (cifrato, da pari a pari, senza cloud) e dividere il lavoro tra più GPU dove il motore lo consente.

## 10. Renderlo senza fatica
Una guida al primo avvio che sceglie un modello per l'hardware, una *modalità semplice* con cinque pulsanti, connessioni a un clic dal catalogo, suggerimenti «cosa posso fare con questo?», un pannello visibile «cosa vede il modello adesso» e tutto in otto lingue.

## Ordine dei lavori
1. Cartelle e file grandi, libreria di prompt, stili di risposta (fatto).
2. Accesso online, GitHub/GitLab/Gitea/pagine web, il catalogo (in corso).
3. Schede di approvazione, registro di audit, budget (la cornice di sicurezza).
4. Automazioni con webhook e notifiche.
5. Duello, consiglio, instradatore.
6. Strumenti per il modello, poi client MCP.
7. Indice e memoria, voce e OCR.
8. Dispositivi, più macchine.

Cosa *non* è promesso: nulla di tutto ciò è ancora stato provato con veri motori colibri con chiamata a strumenti, e ogni servizio esterno cambia API nel tempo; le voci segnate «previsto» nel catalogo sono piani, non funzioni.
