# Foaie de parcurs a interconectării: de la o fereastră de chat la un asistent capabil

Scop: să legăm modelul local de tot ce multiplică ce poate face, în timp ce omul rămâne la comandă. Fiecare pas păstrează trei reguli: **întâi local, nimic nu pleacă fără un da, orice acțiune poate fi văzută și anulată.**

## 0. Cadrul de siguranță (construit primul, folosit de tot ce urmează)
- **Niveluri de capabilitate** (vezi `platforms.rs`): Citește → Caută → Scrie → Notifică → Acționează. O conexiune este creată la un nivel și nu îl poate depăși niciodată.
- **Card de aprobare** pentru fiecare scriere, trimitere și acțiune: arată exact ce se va întâmpla, cu *Permite o dată / Permite pentru această automatizare / Refuză*. Citirea nu cere card.
- **Conținut nesigur** (pagini web, probleme, e-mailuri, documente) este încadrat și etichetat ca date. Un apel de unealtă care urmează după conținut nesigur cere întotdeauna din nou. Aceasta este principala apărare împotriva injectării de prompt.
- **Bugete și frâne**: pași, apeluri de model, minute și octeți pe rulare; o tastă *Oprește tot*.
- **Jurnal de audit** pentru fiecare cerere care a părăsit calculatorul și fiecare acțiune făcută, într-un fișier simplu.
- **Secrete**: jetoanele stau implicit în memorie; salvarea lor este o alegere explicită.

## 1. Unelte pentru model (multiplicatorul)
Colibri expune apelarea de unelte pe majoritatea motoarelor. Programul oferă modelului o trusă mică și cinstită: pagină web, căutare, fișier din depozit, căutare în dosar, calculator, Python într-un mediu izolat, interogare de tabel, căutare în note, salvare de fișier, webhook. Modelul cere, programul verifică nivelul și aprobarea, rulează unealta și întoarce rezultatul ca date.

## 2. MCP: o singură priză pentru mii de unelte
Vorbim Model Context Protocol ca client (și păstrăm `coli mcp` ca server). Orice server MCP (fișiere, baze de date, browsere, calendare, unelte de design, dispozitive de acasă) devine disponibil sub aceleași reguli de aprobare. Este cel mai mare salt de acoperire pentru cel mai puțin cod.

## 3. Memorie și cunoștințe care cresc odată cu tine
- Index local (BM25 acum, embeddinguri mai târziu) peste dosare, note, conversații salvate, cod, arhive de poștă.
- Răspunsurile citează pasajul folosit; „arată-mi unde” îl deschide.
- Biblioteci offline: Wikipedia prin Kiwix, arhive arXiv și Wikidata, manuale, legi.
- *Constelația*: amintiri ca stele legate (etichete, oameni, proiecte) pe care le poți vedea și edita.

## 4. Simțuri și voce (totul local)
- Auz: whisper.cpp pentru dictare și transcrieri de ședințe.
- Vorbire: Piper sau vocea sistemului, în limba interfeței.
- Vedere: modele care primesc imagini (colibri le listează), plus OCR (Tesseract) pentru scanări și fotografii de hârtie, plus camera sau o captură de ecran la cerere.
- Creare: Qwen-Image prin colibri, ComfyUI sau Stable Diffusion pentru imagini; unelte audio și video pe măsură ce apar.

## 5. Mai multe minți
- **Duel**: A/B orb cu Elo, ca să afli care model e cel mai bun *pentru tine*.
- **Consiliu**: mai multe modele răspund, un model judecător le îmbină și semnalează dezacordul.
- **Dirijor**: un model mic și rapid pentru întrebări ușoare, cel mare pentru cele grele, ales de un model de decizie (colibri vine cu Laya, GLiNER2.5-Decide, Clef).
- **Specialiști**: cod, traducere, vedere, fiecare ales automat, fiecare înlocuibil.

## 6. Exactitate: modelul își verifică singur munca
- Rulează în mediu izolat Python-ul pe care îl scrie și arată rezultatul real.
- Teste unitare pentru codul produs, rulate înainte de a fi oferit.
- Autoconsistență (întreabă de două ori, compară), citări obligatorii pentru fapte, un comutator „arată-ți sursele” și un mic cadru de evaluare cu intervale de încredere (ca la LocalMind).

## 7. Agenți și automatizări
Lanțuri salvate de pași (citește, întreabă, verifică, salvează, notifică) cu declanșatoare: manual, după orar, când se schimbă un dosar, când sosește un webhook. Șabloane: buletin de dimineață, ciornă de triere a căsuței de e-mail, observator de depozit, notițe de ședință, notă de cercetare, recapitulare săptămânală. Fiecare pas își arată nivelul.

## 8. Dispozitive și lumea fizică (la sfârșit și cu cea mai mare grijă)
Home Assistant, MQTT, dispozitive seriale, imprimante 3D, camere. Doar citire implicit; acțiuni printr-o listă de permisiuni; niciodată încuietori, alarme sau altceva critic pentru siguranță.

## 9. Mai multe calculatoare, un singur creier
Împarte un model în rețeaua de acasă (un PC puternic servește, un laptop îl folosește), sincronizează prompturi, memorie și setări între mașini (criptat, de la egal la egal, fără cloud) și împarte munca pe mai multe plăci video unde motorul o permite.

## 10. Fă totul fără efort
Ghid la prima pornire care alege un model pentru hardware, un *mod simplu* cu cinci butoane, conexiuni dintr-un clic din catalog, sugestii „ce pot face cu asta?”, un panou vizibil „ce poate vedea modelul acum” și totul în opt limbi.

## Ordinea lucrului
1. Dosare și fișiere mari, bibliotecă de prompturi, stiluri de răspuns (făcut).
2. Acces online, GitHub/GitLab/Gitea/pagini web, catalogul (în curs).
3. Carduri de aprobare, jurnal de audit, bugete (cadrul de siguranță).
4. Automatizări cu webhooks și notificări.
5. Duel, consiliu, dirijor.
6. Unelte pentru model, apoi client MCP.
7. Index și memorie, voce și OCR.
8. Dispozitive, mai multe mașini.

Ce *nu* se promite: nimic de aici nu a fost încă testat cu motoare colibri reale cu apelare de unelte, iar fiecare serviciu extern își schimbă API-ul în timp; intrările marcate „planificat” din catalog sunt planuri, nu funcții.
