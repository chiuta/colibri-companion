# Ce este diferit la acest motor

[English](INNOVATIONS.md) · [Română](INNOVATIONS.ro.md) · [Français](INNOVATIONS.fr.md) · [Italiano](INNOVATIONS.it.md) · [Español](INNOVATIONS.es.md) · [Português](INNOVATIONS.pt.md) · [Deutsch](INNOVATIONS.de.md) · [Magyar](INNOVATIONS.hu.md)

Note în limbaj simplu despre funcțiile pe care alte motoare locale nu le au, în mare parte. Fiecare spune ce face, unde ajungi la ea și ce nu poate face. Toate au fost testate doar pe modele sintetice; autorii nu au rulat niciuna pe un punct de control real. „În `corbul`” înseamnă că o comandă sau un flag există azi (`corbul help`); „doar bibliotecă” înseamnă cod Rust fără comandă. Vezi [ENGINE.md](ENGINE.md) și [QUICKSTART.md](QUICKSTART.md).

## Integritate, hardware și memorie

### Mesh în rețeaua locală

Modulul `mesh`. În `corbul`: `worker`, `generate --workers`, `--discover`, `--key`, `--wire-f16`. Un model prea mare pentru o singură mașină este tăiat în intervale de straturi, câte unul pentru fiecare calculator. Între ele circulă doar starea ascunsă. Mașinile dovedesc că au aceeași cheie (HMAC-SHA256 reciproc), fiecare cadru poartă un MAC, dimensiunile strângerii de mână sunt plafonate, un worker căzut este scos, iar straturile lui sunt redistribuite. Logiții denși f32 sunt egali cu cei de pe o singură mașină. Descoperirea UDP este opțională: un worker răspunde la sonde doar dacă a fost pornit cu `--discovery` sau `--discovery-port` (`--no-discovery` o oprește forțat), iar `--discover` al coordonatorului trimite o sondă în difuzare și așteaptă răspunsuri de la workerii care au același model. *Nu poate:* cripta (folosește o rețea locală de încredere sau un VPN); a fost testat doar pe loopback; viteza față de o singură mașină nu a fost măsurată.

### Transmiterea pe straturi

Modulul `stream_layers`. În `corbul`: `--stream-layers`, `--ram-budget`, `--stream-window`, `--disk-mbps` (pentru estimări). Rulează un model dens mai mare decât RAM-ul, ținând în memorie o fereastră mică de straturi și preîncărcând pe un fir din fundal. Rezultatele sunt egale cu cele ale modelului aflat în RAM. *Nu poate:* ocoli faptul că fiecare token citește fiecare strat, deci viteza este limitată de viteza discului; doar straturi dense (modelele cu amestec de experți își transmit în schimb experții rutați, cu `--ram-budget`).

### Citări blocate pe citat

Modulul `quotelock`. În `corbul`: `--sources FILE` (repetabil), `--no-lock`, `--citations FILE`; serverul acceptă `sources` și oferă `POST /v1/quotes/verify`. În timpul decodării, textul dintre « și » trebuie să fie un pasaj întocmai dintr-o sursă (tolerant la majuscule, spațiere, stiluri de ghilimele și diacriticele românești); `[S3]«` limitează un citat la sursa 3. Apoi un verificator clasifică fiecare citat (întocmai, sursă greșită, trunchiat, modificat, îmbinat, nesusținut) și notează propozițiile din afara citatelor; raportul spune când s-a atins o limită de lucru (`limit_hit`). *Nu poate:* face adevărat textul din jurul citatelor; răspunsurile în flux nu conțin bloc de citări.

### Planificator, guvernor și contor de energie

Modulul `governor` (și `plan`, `hw`). În `corbul`: `plan`, `doctor`, `--governor`, `--no-governor`, `--verbose`, `--calibrate`. Înainte de încărcare, spune dacă modelul încape, se transmite pe straturi, rulează încet sau nu poate rula, cu o viteză estimată. În timpul rulării, guvernorul citește temperatura, memoria și bateria (Linux) și recomandă fire și pauze, fără oscilații. *Nu poate:* da viteze exacte, sunt estimări (`--calibrate` cronometrează calculatorul tău cam o secundă); energia este o presupunere, în afară de cazul în care se poate citi un contor RAPL; intensitatea carbonului (400 g/kWh) este o valoare provizorie.

### Politici pentru cache-ul KV și instantanee de sesiune

Modulele `kvpolicy`, `snapshot`. În `corbul`: `--kv-policy streaming|h2o|snapkv`, `--kv-budget`, `--kv-sinks`, `--session DIR`, `--session-cap`, `--compress`. Țin conversațiile lungi într-un buget de memorie (ancore de atenție, „heavy hitters”, comprimarea promptului) și salvează sau reiau o conversație dintr-un fișier cu sumă de control, refolosind cel mai lung prefix salvat. Instantaneele pot purta un HMAC cu cheie. *Nu poate:* evita pierderile, eliminarea pierde informație prin construcție; `--kv-policy` și `--session` nu pot fi combinate; conversațiile foarte lungi pot depăși contextul de antrenare.

### Mod reproductibil și chitanțe

Modulele `simd` (comutatorul), `receipt`. În `corbul`: `--reproducible`, `--receipt FILE`, `--receipt-key`, `--receipt-key-file`, `--receipt-topk`, `--keep-steps`, `--leaves`, `verify-receipt`, `replay`. O chitanță înregistrează modelul (o amprentă Merkle a ponderilor), tokenizatorul, setările eșantionatorului, promptul și ieșirea, cu un lanț de hash-uri, sigilate cu SHA-256 și o cheie HMAC opțională; `replay` o rulează din nou. Ponderile deteriorate sunt localizate până la tensor și bucată. *Nu poate:* dovedi nimic unui străin fără cheia HMAC sau o nouă rulare (nu există semnături cu cheie publică); modelele MoE și cele cu spațiu de stare sunt refuzate; ARM netestat; modul reproductibil costă viteză.

### Unelte lingvistice cu prioritate pentru română

Modulul `lang`. În `corbul`: `lang detect`, `lang normalize`, `--lang CODE` (nu și în server). Normalizează ș/ț și ş/ţ, repară mojibake, recunoaște 27 de limbi, interzice scripturile străine în timpul decodării (română: fără ş ţ), formatează numere, sume de bani și date în stil românesc, restabilește diacriticele lipsă cu un semn pe fiecare cuvânt nesigur. *Nu poate:* restaurarea folosește un lexicon de aproximativ 650 de forme și nu a fost măsurată pe un corpus; cuvintele ambigue sunt rezolvate cu câteva reguli de context; recunoașterea limbii are nevoie de cam cinci cuvinte.

### Analiză a incertitudinii

Modulul `confidence`. Doar bibliotecă (exemplu demonstrativ `confidence_demo`). Statistici pe token, cuvânt, propoziție și răspuns pentru o hartă termică, o euristică transparentă de risc de halucinație, scoruri de autoconsistență și unelte de calibrare. *Nu poate:* scorul de risc este o euristică, nu un detector de adevăr.

## Calcul la momentul testării, reglaj, regăsire

### Căutare și lookahead

Modulul `search`. În `corbul`: `search --method beam|best-of-n|vote` (`--width`, `--samples`, `--answer number|text`, `--length-penalty`, `--all`, `--explain`, `--json`) și `--lookahead` la `generate` (`--ngram-min`, `--ngram-max`). Căutare beam, best-of-N, vot de autoconsistență, un arbore de conversație (bibliotecă), decodare speculativă prin căutare în prompt. *Nu poate:* lookahead ajută doar când textul repetă text anterior; best-of-N are nevoie de un bun evaluator.

### Decodare speculativă

Modulele `spec`, `selfspec`. În `corbul`: `--draft MODEL`, `--lookahead`, `--self-spec`, `--skip-layers auto|LIST`, `--draft-len`. Același răspuns greedy și aceeași distribuție de eșantionare, cu mai puține treceri complete; ieșirea apare la sfârșit. Doar `--temperature`, `--top-k`, `--top-p`; nu merge cu ieșire structurată, chitanțe, sesiuni sau mesh. *Nu poate:* promite o accelerare; auto-speculația revine la pași simpli pentru inelele cu fereastră glisantă care s-ar înfășura, pentru cache-uri compactate și pentru straturi descărcate.

### Cuantizare mixtă automată și cuantizatorul

Modulele `autoquant`, `quantize`. În `corbul`: `autoquant FOLDER --budget 4.5bpw|6g|Q4_K` (`--recipe`, `--report`, `--candidates`, `--base`, `--imatrix`, `--sample-rows`) și `quantize IN OUT.gguf --type NAME | --recipe FILE`. Măsoară eroarea ponderilor pe grupuri de tensori în fiecare format și alege formate într-un buget de dimensiune (rucsac exact sau lacom). *Nu poate:* estima calitatea răspunsurilor, doar eroarea ponderilor; presupune că daunele se adună; i-quant-urile nu există.

### Vectori de direcționare și lentila de logiți

Modulul `steer`. În `corbul`: `steer train|apply|lens` (`--positive`, `--negative`, `--method mean-diff|pca`, `--out`, `--vector`, `--strength`, `--layers`, `--positions`, `--no-attention`). Învață o direcție din două seturi de prompturi și o adaugă la fluxul rezidual fără a schimba ponderile; lentila arată ce ar prezice fiecare strat. *Nu poate:* funcționa cu modele cu spațiu de stare, ieșire structurată, speculație, chitanțe sau sesiuni; un vector de la un fine-tuning este acceptat de altul cu aceeași formă.

### Regăsire locală

Modulul `rag`. În `corbul`: `rag add|search|ask` (`--query`, `--question`, `--k`, `--chunk-chars`, `--overlap`, `--embed-model`). Împărțire în bucăți atentă la limba română, un index HNSW în Rust simplu, BM25 cu ignorarea diacriticelor, fuziune de clasamente, pasaje numerotate `[S1]…` care alimentează citările blocate pe citat, fișiere cu sumă de control. *Nu poate:* implicit, embedderul este unul prin hashing (cuvinte și părți de cuvinte, nu sens); `--embed-model` folosește stările ascunse ale unui model generativ (slab); CLI reindexează la fiecare apel și citește text simplu, nu PDF sau fișiere Office (biblioteca are ajutoare de ingestie).

### Îmbinarea modelelor

Modulul `merge`. În `corbul`: `merge CONFIG.json` (`--out`, `--model`, `--dry-run`, `--max-memory`, `--overwrite`, `--verbose`). Liniar, SLERP, aritmetica sarcinilor, TIES, DARE (cu sămânță) și felii de straturi, câte un tensor pe rând. Formele și intervalele sunt verificate înainte de scriere; legăturile simbolice nu sunt urmate. *Nu poate:* garanta calitatea; scrie doar ponderi safetensors, niciodată cuantizate, iar `config.json` și tokenizatorul le copiezi tu.

### Filigran de text

Modulul `watermark`. În `corbul`: `generate --watermark KEY` (`--wm-gamma`, `--wm-delta`, `--wm-context`) și `watermark detect MODEL --key K --text T | --file F` (`--no-dedup`). Un bias de „listă verde” cu cheie și un detector cu scor z. *Nu poate:* da o dovadă; textele scurte, codul și citatele poartă puțin semnal; parafrazarea îl elimină; cine are cheia îl poate falsifica; lipsa semnului nu dovedește că autorul este om.

## Multimodal (doar bibliotecă)

### Viziune

Modulele `vision`, `imageio`. Un encoder de tip CLIP/SigLIP, proiector, tăiere în plăci și un cârlig `Model::forward_segments` pentru a introduce embeddings într-un prompt; decodoare pentru PNG, JPEG (de bază și progresiv), BMP și PNM scrise după specificații, cu limite de dimensiune. *Nu poate:* fi folosit din `corbul` (doar text); nu a rulat niciun punct de control real.

### Audio

Modulul `audio`. Citire și scriere WAV, reeșantionare, FFT, trăsături log-mel, detectarea activității vocale, un rulator cu structură Whisper. *Nu poate:* fi folosit din `corbul`; nu a rulat niciun punct de control Whisper real.

## Spațiu de stare, hibride și GPU (vezi ENGINE.md)

Modulele `ssm`, `hybrid`, `mla` și crate-ul `companion-engine-gpu` sunt descrise, cu limitele lor, în ENGINE.md.
