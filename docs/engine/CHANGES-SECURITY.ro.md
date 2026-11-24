# Modificări de securitate și robustețe

[English](CHANGES-SECURITY.md) · [Română](CHANGES-SECURITY.ro.md) · [Français](CHANGES-SECURITY.fr.md) · [Italiano](CHANGES-SECURITY.it.md) · [Español](CHANGES-SECURITY.es.md) · [Português](CHANGES-SECURITY.pt.md) · [Deutsch](CHANGES-SECURITY.de.md) · [Magyar](CHANGES-SECURITY.hu.md)

Alte documente: [ENGINE.md](ENGINE.md) · [QUICKSTART.md](QUICKSTART.md)

Această pagină enumeră ce a găsit revizuirea adversarială a motorului și a `corbul` și ce s-a făcut în privința lor, ca să poți judeca cât să te încrezi într-o funcție. Este o listă de remedieri și limite, nu o certificare de securitate: nimic de aici nu a fost auditat de o terță parte.

## Server

- **Cereri între site-uri.** Apelurile care schimbă starea verifică `Origin` și cer un `Content-Type` JSON, astfel încât o pagină web pe care o vizitezi nu poate face browserul tău să trimită cereri către un `corbul serve` local. Browserele sunt permise doar de la originile pe care le enumeri cu `--cors`.
- **Rute de administrare.** `/api/create`, `/api/delete`, `/models/load` și `/models/unload` cer tokenul principal. Deținătorii de chei API le pot folosi doar dacă pornești serverul cu `--allow-admin`.
- **Secrete în afara liniei de comandă.** `--token-file`, `--key-file`, `--receipt-key-file` și variabilele `CORBUL_TOKEN` și `CORBUL_MESH_KEY` țin secretele departe de lista de procese. Variantele simple `--token`, `--key` și `--receipt-key` merg în continuare și sunt vizibile altor utilizatori ai calculatorului.
- **Limite de resurse.** `--max-kv` refuză cererile al căror cache KV ar fi prea mare (implicit 8g); `--max-heavy` limitează cererile simultane de tokenizare, de șablon și de verificare (implicit 4); limitele de rată din `--keys-file` acoperă fiecare rută care costă calcul; decodorul corpului „chunked” este liniar; `POST /v1/quotes/verify` are plafoane pentru dimensiunea textului și a surselor.
- **Jurnale.** Textul prompturilor nu este jurnalizat decât dacă se dă `--log-prompts`; `--no-log` oprește jurnalul.
- **Interfața web.** Pagina în sine este publică (nu conține secrete); fiecare apel pe care îl face cere tokenul.
- **Limita care rămâne:** serverul vorbește HTTP simplu. Leagă-l de `127.0.0.1` (implicit) sau pune TLS în fața lui.

## Mesh

- Strângere de mână HMAC-SHA256 reciprocă, cu o cheie partajată de cel puțin 8 caractere, un MAC pe fiecare cadru, plafoane pentru dimensiunile strângerii de mână și ale cadrelor.
- Descoperirea UDP este opțională (`--discovery`, `--discovery-port`), iar răspunsurile nu sunt mai mari decât sonda, deci un worker nu poate fi folosit ca amplificator.
- **Limita care rămâne:** traficul este autentificat, dar necriptat; cheia este partajată de toți cei din mesh; testat doar pe loopback.

## Fișiere, descărcări și îmbinări

- Parsarea GGUF primește lungimea reală a fișierului (`parse_with_len`); construcția tokenizatorului folosește un trie cu plafon pentru dimensiunea pieselor; fișierele trunchiate și corupte returnează erori în testele adversariale.
- Descărcări: fixare pe SHA de commit, plafoane de dimensiune, refuzul redirecționărilor către http simplu, alternativa `wget` anonim (tokenul nu i se dă niciodată), verificări SHA-256.
- Îmbinările nu urmează legături simbolice și refuză să suprascrie o ieșire dacă nu se dă `--overwrite`; formele și intervalele sunt verificate înainte de scriere.
- Decodoarele de imagine impun `Limits` înainte de alocare și limitează expandarea deflate; cititorul WAV are limite pentru fișiere ostile.

## Decodare și verificare

- Decodarea cu gramatică, cu expresie regulată și cu schemă JSON are bugete de lucru; blocarea citatelor este liniară, cu plafoane, și raportează `limit_hit` când un plafon o oprește.
- Chitanțele au plafoane de dimensiune; instantaneele pot fi sigilate cu un HMAC cu cheie; chitanțele și instantaneele folosesc lanțuri SHA-256.
- Corecturi de corectitudine găsite la revizuire: probabilități rămase vechi după top-k/top-p (acum renormalizate), aliasul `typ_p`, virgulele zecimale la tratarea numerelor, limitele lookahead dincolo de context, scorul de grup DeepSeek-V2, exactitatea loturilor cuantizate (`exact_batch`).

## Nefăcut sau neverificat

- Fără audit de la o terță parte, fără o campanie de fuzzing dincolo de testele adversariale din depozit, fără test de penetrare pentru `corbul serve`.
- `ToolStream::push` poate rescana de mai multe ori bufferul (pătratic în lungimea ieșirii, până la `--max-tokens`).
- `GrammarCache` există, dar nu este încă folosit de CLI.
- Licențele dependențelor au fost verificate ultima oară cu `cargo-deny` la commitul b496ba5.
- Fluxurile CI nu au rulat niciodată; căile de cod pentru ARM, macOS și Windows nu au fost executate aici.
