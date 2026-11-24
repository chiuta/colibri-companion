# Pornire rapidă: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` este linia de comandă a motorului clean-room ([ENGINE.ro.md](ENGINE.ro.md)). Pagina enumeră doar comenzile și opțiunile care există în `crates/engine-cli` în momentul scrierii; linia de comandă încă se extinde, deci ultimul cuvânt îl are `corbul help`. Unele opțiuni de mai jos sunt citite de cod, dar încă nu apar în `corbul help`; sunt marcate *(încă nu e în help)*. Ce nu există încă e marcat **planificat**.

**Avertisment onest:** `corbul` a fost testat doar pe modele sintetice minuscule. Te așteaptă asperități pe puncte de control reale; te rugăm să le raportezi.

## Construire

```bash
cargo build --release -p colibri-rs-cli     # produce target/release/corbul
```

Un model este fie un folder cu `config.json` + `*.safetensors` + `tokenizer.json`, fie un singur fișier `.gguf`.

## Privește, descarcă, rulează

```bash
corbul inspect  ./my-model                  # sau file.gguf sau owner/repo[@rev]: arhitectură, mărime, suport
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors; HF_TOKEN pentru depozite cu acces restricționat, HF_ENDPOINT pentru o oglindă
corbul generate ./my-model --prompt "Bună" --max-tokens 64
corbul chat     ./my-model --system "Fii concis."
corbul serve    ./my-model --port 8080      # compatibil OpenAI / Ollama / Anthropic, legat implicit la 127.0.0.1
```

Opțiuni comune pentru `generate`, `chat` și `serve`: `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--kv f32|f16|q8`, `--no-mmap`, `--threads N`, `--ctx N`, `--model-name`, `--quiet`. `generate` mai primește `--prompt-file F` și `--raw` (fără șablon de chat). `corbul run` este un alias pentru `generate`.

Servirea mai multor modele: `--models-dir DIR`, `--alias NUME=MODEL`, `--max-loaded N`, `--sleep-idle-seconds S`, `--token T` (sau `CORBUL_TOKEN`), `--keys-file FIȘIER`, `--cors ORIGINI`, `--log-prompts`, `--no-log`, `--drain-seconds S`, `--config-dir DIR`.

## Va rula pe această mașină?

```bash
corbul doctor                # ce vede motorul: seturi de instrucțiuni CPU, nuclee, RAM, disc, senzori, contor de energie
corbul plan ./my-model --context 8192            # verdict: FITS / TIGHT / STREAMS / TOO SLOW / WON'T RUN, viteză, sfaturi
corbul plan ./my-model --ram-budget 8g --calibrate   # prefă-te că ai 8 GB; mai întâi cronometrează mașina o secundă
corbul capabilities [--json]                     # ce poate și ce nu poate această versiune
```

Vitezele sunt estimări, nu măsurători ale modelului tău.

## Mai mare decât RAM-ul

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"                        # MoE: experții rutați se citesc de pe disc
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --prompt "…"   # dens: straturile se transmit (limitat de disc; --stream-window încă nu e în help)
```

## Mai multe calculatoare (plasă LAN)

```bash
# pe fiecare calculator ajutor (cheia are minimum 8 caractere; sau setează CORBUL_MESH_KEY)
corbul worker ./my-model --key SECRET --bind 0.0.0.0:7471          # --no-discovery sare peste descoperirea UDP
# pe calculatorul principal
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key SECRET --prompt "…"
corbul generate ./my-model --discover --key SECRET --prompt "…"      # găsește workerii din LAN
```

Traficul este autentificat, dar **nu e criptat**: folosește o rețea de încredere sau un VPN. Testat doar pe loopback.

## Ieșire structurată, citate, adaptoare

```bash
corbul generate M --json --prompt "…"                 # ieșire JSON validă
corbul generate M --json-schema schema.json --prompt "…"      # fișier sau JSON direct; și --regex MODEL, --grammar FIȘIER (GBNF); folosește doar una din cele patru; încă nu e în help
corbul generate M --sources notes.txt --sources paper.txt --prompt "Răspunde cu citate" --citations report.json   # citate blocate pe surse
corbul generate M --lora ./adapter:0.8                # adaptor PEFT sau GGUF cu o scară
```

Serverul acceptă `response_format`, `grammar` și `sources` și oferă `POST /v1/quotes/verify`.

## Conversații lungi, sesiuni, reproductibilitate

```bash
corbul generate M --session ./sessions --prompt "…"             # refolosește cel mai lung prefix de prompt salvat (nu împreună cu --kv-policy)
corbul generate M --kv-policy h2o --kv-budget 2048 --prompt "…" # streaming | h2o | snapkv; cu pierdere prin construcție
corbul generate M --governor --prompt "…"                       # se adaptează la temperatură, memorie și baterie (Linux)
corbul generate M --reproducible --temperature 0 --prompt "hi" --receipt r.json --receipt-key SECRET
corbul verify-receipt r.json --key SECRET --model M            # verifică sigiliul (și greutățile dacă dai --model)
corbul replay r.json --model M --key SECRET                    # reia generarea și compară
```

Chitanțele merg cu bucla simplă de generare, nu cu modelele MoE. Opționale: `--receipt-topk N`, `--keep-steps`, `--leaves` *(încă nu e în help)*.

## Planificat (încă nu e în `corbul`)

- Încărcarea modelelor Mamba / RWKV / MLA (în stil DeepSeek), deși biblioteca are blocurile.
- Uneltele de limbă română, analiza incertitudinii, căutarea și lookahead-ul, decodarea auto-speculativă, vectorii de control, regăsirea, îmbinarea și filigranul (vezi [INNOVATIONS.ro.md](INNOVATIONS.ro.md)).
- Endpoint-uri de embedding și rerank (serverul răspunde „not implemented”), intrări vizuale și audio.
- Backend-ul GPU și o interfață web.
