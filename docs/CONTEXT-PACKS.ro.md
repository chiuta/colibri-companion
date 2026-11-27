# Pachete de context

Refolosește munca făcută pe un text când apare din nou într-un prompt diferit (doar motorul corbul; clean-room, cod original).

Un cache de prefix funcționează doar când textul comun e primul. Un *pachet* e cache-ul KV al unui text calculat singur și ținut pe un raft (magazin de snapshot-uri semnat, cu plafon LRU).

`compose` pune pachetele unul după altul: rândurile se copiază, iar cheile se rotesc cu diferența de poziție (exact); apoi începutul fiecărui pachet următor se recalculează în ferestre tot mai mari până când schimbarea măsurată scade sub o toleranță. Întrebarea se calculează mereu proaspăt.

`tolerance = 0` recalculează tot și dă exact ce dă un prefill complet. Modelele cu straturi cu fereastră glisantă, cache evacuat sau cache pe accelerator sunt refuzate și revin la prefill obișnuit.

Neexact până la reparare: o stare cu rânduri mutate raportează `is_exact() == false` și nu se scrie niciodată pe disc.

Limite: motorul colibri al lui JustVugg nu poate exporta cache-uri, deci merge doar cu corbul. Testat pe modele sintetice mici; calitatea pe modele reale nu e încă măsurată.

## Linia de comandă

`corbul packs MODEL --session DIR --pack FIȘIER… --question TEXT [--tolerance 0.02] [--probe 8] [--max-new 64] [--check]` pune fiecare fișier pe raft o singură dată, apoi răspunde folosind pachetele păstrate, în orice ordine. `--check` rulează și un prefill complet și afișează diferența de scoruri și timpul fiecăruia.
