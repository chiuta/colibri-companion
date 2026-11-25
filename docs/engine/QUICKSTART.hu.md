# Gyorsindítás: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

A `corbul` a clean-room motor parancssora ([ENGINE.md](ENGINE.hu.md)). Az ezen az oldalon szereplő minden parancs és kapcsoló létezik a `crates/engine-cli` könyvtárban, és kiírja a `corbul help`; egy szkript (`tools/check_engine_docs.py`) hibával leáll, ha a kódban van olyan kapcsoló, alparancs vagy környezeti változó, amelyet ez az oldal nem említ. A végső szó a `corbul help` kimenetéé marad.

**Őszinte figyelmeztetés:** a `corbul` programot csak apró szintetikus modelleken és tokenizáló-referenciafájlokon teszteltük. A szerzők egyetlen valódi, letöltött modellt sem futtattak. Számíts hiányosságokra, és kérjük, jelezd őket. A `corbul capabilities` kiírja, mi működik és mi nem.

## Fordítás

```bash
cargo build --release -p colibri-rs-cli     # létrehozza a target/release/corbul fájlt
```

Egy modell egy mappa `config.json` + `*.safetensors` + `tokenizer.json` fájlokkal, vagy egyetlen `.gguf` fájl.

## Megnézni, letölteni, futtatni

```bash
corbul inspect  ./my-model                  # vagy file.gguf vagy owner/repo[@rev]: architektúra, méret, támogatottság
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors
corbul generate ./my-model --prompt "Hello" --max-tokens 64
corbul generate ./my-model --prompt-file question.txt --raw   # --raw: nincs csevegősablon
corbul chat     ./my-model --system "You are concise."
corbul run      ./my-model --prompt "Hello"                   # a `run` ugyanaz, mint a `generate`
corbul serve    ./my-model --port 8080                        # alapértelmezés szerint a 127.0.0.1 címhez kötve
```

Állítsd be a `HF_TOKEN` (vagy a `HUGGING_FACE_HUB_TOKEN`) változót a védett tárolókhoz, és a `HF_ENDPOINT` változót tükörhöz. A letöltésekhez `curl` kell (a névtelen `wget` tartalék).

A `generate`, a `chat` és a `serve` közös opciói: `--max-tokens N`, `--temperature T`, `--top-p P`, `--top-k K`, `--seed S`, `--kv f32|f16|q8`, `--no-mmap`, `--model-name NAME`, `--threads N` (a `COLIBRI_THREADS` változó is), `--ctx N`, `--quiet`, `--ram-budget SIZE`, `--lora PATH[:S]` (ismételhető; PEFT-mappa vagy GGUF-adapter skálázással).

További mintavételező opciók a `generate` és a `chat` parancshoz:

```bash
corbul generate M --prompt "…" --min-p 0.05 --typical-p 0.9 --repeat-penalty 1.1 \
  --presence-penalty 0.2 --frequency-penalty 0.2 --mirostat 2 --mirostat-tau 5 --mirostat-eta 0.1 \
  --dry-multiplier 0.8 --xtc-probability 0.1
corbul generate M --prompt "…" --sampler-json '{"top_k": 20}'   # bármely mintavételező-mező, a fenti kapcsolók előtt alkalmazva
```

Az állapottér- és hibridmodellek (Mamba, RWKV, Jamba, Bamba) a `generate` és a `chat` parancson át sima szövegfolytatásként futnak: nincs csevegősablon, nincs strukturált kimenet, nincs `serve`.

## Fut ezen a gépen?

```bash
corbul doctor                # CPU-utasításkészletek, választott kernel, RAM, lemez, érzékelők, GPU-szonda
corbul plan ./my-model --context 8192          # ítélet (elfér / szűk / folyamatosan töltődik / túl lassú / nem fut), sebesség, tanács
corbul plan ./my-model --ram-budget 8g --calibrate   # tégy úgy, mintha 8 GB-od lenne; előbb mérd meg ezt a gépet körülbelül egy másodpercig
corbul capabilities [--json]                   # mit tud és mit nem tud ez a build (álnév: corbul features)
```

A sebességek becslések, nem a te modelled mérései. Az `ENGINE_SIMD` kikényszerít egy kernelt, az `ENGINE_REPRODUCIBLE=1` bekapcsolja a reprodukálható módot, az `ENGINE_BACKEND=cpu|gpu|auto` változót a könyvtár olvassa (a `corbul` mindig a CPU-n számol).

## Nagyobb a memóriánál

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"     # MoE: az irányított szakértők a lemezről töltődnek folyamatosan
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --disk-mbps 800 --prompt "…"   # sűrű: a rétegek folyamatosan töltődnek; a sebességet a lemez korlátozza; a --disk-mbps (alapértelmezés 500) csak a kiírt sebességbecslést táplálja
```

## Több gép (LAN-háló)

```bash
# minden segédgépen (a kulcsnak 8+ karakteresnek kell lennie; vagy állítsd be a CORBUL_MESH_KEY változót)
corbul worker ./my-model --key MESHKEY12345 --bind 0.0.0.0:7471
corbul worker ./my-model --key-file mesh.key --discovery            # az UDP-felderítési szondákra is válaszol (bekapcsolásra épül); --discovery-port N; a --no-discovery kikényszeríti a kikapcsolást
# a fő gépen
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key MESHKEY12345 --prompt "…"
corbul generate ./my-model --discover --key MESHKEY12345 --prompt "…"      # szondát szór szét, azokat a munkásokat használja, amelyek ugyanazt a modellt tartják
corbul generate ./my-model --workers h:7471 --key MESHKEY12345 --wire-f16 --local-speed 2.5 --prompt "…"
```

A `--wire-f16` felezi a rejtett állapot forgalmát (és lemond a bitazonosságról); a `--local-speed` megmondja a felosztónak, mennyire gyors ez a gép a munkásokhoz képest; a `--discover-target HOST:PORT` és a `--discover-wait-ms N` a szonda címét (alapértelmezés `255.255.255.255:47813`) és a várakozást (alapértelmezés 2000 ms) változtatja. A forgalom hitelesített, de **nem titkosított**: használj megbízható hálózatot vagy VPN-t. Csak loopbacken tesztelve.

## Strukturált kimenet, hivatkozások, adapterek

```bash
corbul generate M --json --prompt "…"                       # érvényes JSON
corbul generate M --json-schema schema.json --prompt "…"    # egy fájl vagy beágyazott JSON
corbul generate M --regex '[A-Z]{3}-[0-9]{4}' --prompt "…"  # reguláris kifejezés
corbul generate M --grammar rules.gbnf --prompt "…"         # GBNF-nyelvtan; a négyből csak egyet használj
corbul generate M --sources notes.txt --sources paper.txt --prompt "Answer with quotes" --citations report.json   # idézetzáras
corbul generate M --sources notes.txt --no-lock --prompt "…"   # az idézeteket csak utólag ellenőrzi
corbul generate M --lora ./adapter:0.8
```

A szerver elfogadja kérésekben a `response_format`, a `grammar` és a `sources` mezőt, és kínálja a `POST /v1/quotes/verify` végpontot.

## Hosszú csevegések, munkamenetek, reprodukálhatóság

```bash
corbul generate M --session ./sessions --session-cap 2g --prompt "…"   # a leghosszabb mentett promptelőtagot használja újra; a --compress kicsinyíti a mentett fájlokat (a --kv-policy mellett nem)
corbul generate M --kv-policy h2o --kv-budget 2048 --kv-sinks 4 --prompt "…"   # streaming | h2o | snapkv; tervezetten veszteséges
corbul generate M --governor --verbose --prompt "…"            # alkalmazkodik a hőmérséklethez, a memóriához és az akkumulátorhoz (Linux); a --no-governor kikapcsolja
corbul generate M --lang ro --prompt "…"                         # idegen írásrendszerű tokenek tiltása; román: nincs ş ţ
corbul generate M --reproducible --temperature 0 --prompt "hi" --receipt r.json --receipt-key-file receipt.key
corbul verify-receipt r.json --key-file receipt.key --model M      # ellenőrzi a pecsétet (és a súlyokat, ha a --model meg van adva)
corbul replay r.json --model M --key-file receipt.key              # újra lefuttatja a generálást, és összehasonlítja
corbul replay r.json --model M --key-file receipt.key --prompt "…" --raw     # ha a nyugta nem őrzi a prompt szövegét; vagy --prompt-tokens FILE
```

A nyugták a sima generálási ciklussal működnek, MoE- vagy állapottér-modellekkel nem. Opcionális: `--receipt-key K` (látszik a folyamatlistában; inkább a `--receipt-key-file` kapcsolót használd), `--receipt-topk N`, `--keep-steps`, `--leaves`.

## Gyorsabb dekódolás

```bash
corbul generate M --draft ./small-model --draft-len 4 --prompt "…"      # egy vázlatmodell javasol, M ellenőriz
corbul generate M --lookahead --ngram-min 1 --ngram-max 4 --prompt "…"   # prompt-keresés, nincs extra modell
corbul generate M --self-spec --skip-layers auto --prompt "…"            # M a rétegei egy részhalmazával vázlatol; vagy --skip-layers 4,6,8-11
```

Ugyanaz a mohó válasz és ugyanaz a mintavételezési eloszlás; a szöveg a végén jelenik meg. Csak a `--temperature`, `--top-k`, `--top-p` kapcsolók érvényesek; nem használható strukturált kimenettel, nyugtákkal, munkamenetekkel vagy a hálóval.

## Keresés, kvantálás, összefésülés, terelés, visszakeresés, vízjel, nyelv

```bash
corbul search M --prompt "Q" --method beam --width 4 --all --explain --json     # még: --length-penalty L
corbul search M --prompt "2+2?" --method best-of-n --samples 5
corbul search M --prompt "2+2?" --method vote --samples 5 --answer number       # vagy text
corbul autoquant ./model --budget 4.5bpw --recipe recipe.json --report report.txt   # 4.5bpw, 6g vagy Q4_K; --candidates Q4_K,Q6_K --base Q8_0 --imatrix I --sample-rows 256 --json
corbul quantize ./model out.gguf --type Q4_K_M --imatrix imatrix.dat    # vagy --recipe recipe.json; --name NAME
corbul merge merge.json --out merged.safetensors --dry-run              # --model PATH (ismételhető) --max-memory 8g --overwrite --verbose
corbul steer train M --positive polite.txt --negative rude.txt --out polite.cvec --method mean-diff   # vagy pca
corbul steer apply M --vector polite.cvec --strength 1 --layers 8-20 --positions last --prompt "…" --no-attention
corbul steer lens M --prompt "The capital of France is"
corbul rag add ./lib notes.txt paper.txt --chunk-chars 800 --overlap 120 --embed-model ./embedder
corbul rag search ./lib --query "refund policy" --k 5 --json
corbul rag ask ./lib M --question "What is the refund period?"             # idézetzáras hivatkozásokkal válaszol
corbul lang detect "Aceasta este o propoziție."                          # vagy --file text.txt
corbul lang normalize --file text.txt                                    # ș/ț, mojibake, idézőjelek
corbul generate M --prompt "…" --watermark SECRETKEY --wm-gamma 0.25 --wm-delta 2 --wm-context 4
corbul watermark detect M --key SECRETKEY --file answer.txt --no-dedup   # vagy --text "…"
corbul embed ./bge-small "first text" "second text" --pool cls --json    # a BERT-family folder; or --file lines.txt, --no-normalize
corbul rerank ./ms-marco-minilm --query "refund policy" "doc one" "doc two" --top 3 --json   # needs a cross-encoder
```

Nem minden kapcsolókombináció működik; ilyenkor minden parancs egysoros használati hibát ír ki. Ebből semmit sem futtattak valódi ellenőrzőponton.

## Kiszolgálás

```bash
corbul serve --models-dir ./models --host 127.0.0.1 --port 8080 --token-file token.txt --default-model qwen
corbul serve --models-dir ./models --alias fast=qwen-0.5b --max-loaded 2 --sleep-idle-seconds 600
corbul serve --models-dir ./models --keys-file keys.json --allow-admin --max-kv 8g --max-heavy 4
corbul serve --models-dir ./models --cors "https://app.example" --log-prompts --drain-seconds 10 --config-dir ./cfg
corbul serve --encoder ./bge-small --port 8081                                  # embeddings (and /v1/rerank for a cross-encoder) without a decoder
```

A webes felülethez nyisd meg a `http://127.0.0.1:8080/` címet (a `/ui` is jó). Az oldal nyilvános; minden hívásához token kell. A `--token T` megegyezik a `CORBUL_TOKEN` változóval (látszik a folyamatlistában, ezért inkább a `--token-file` kapcsolót használd). A `--keys-file` API-kulcsokat tartalmaz kulcsonkénti modellistákkal és sebességkorlátokkal. Az `--allow-admin` lehetővé teszi a kulcsbirtokosoknak a `/api/create`, `/api/delete`, `/models/load` és `/models/unload` használatát; alapértelmezés szerint csak a mestertoken használhatja. A `--max-kv` elutasítja azokat a kéréseket, amelyeknél a KV-gyorsítótár meghaladná a megadott méretet (alapértelmezés 8g); a `--max-heavy` az egyidejű tokenizálási, sablon- és ellenőrzési kéréseket korlátozza (alapértelmezés 4). A `--no-log` elnémítja a kérésnaplót. A `GET /v1/capabilities` felsorolja, mi működik. A beágyazási útvonalak működnek (`/v1/embeddings`, `/api/embed`); az újrarangsoroláshoz `--encoder` kell.

## A `corbul` programban (még) nem érhető el

- Többfejes látens figyelmet használó (DeepSeek-stílusú) modellek, GGUF formátumú RWKV, Zamba/Zamba2, a többi i-kvantum (IQ1–IQ3), RoBERTa/XLM-R-modellek.
- Látás- és hangbemenetek (a könyvtárban megvannak az építőelemek), a GPU-háttérrendszer.
- A szokásos kvantált előtöltés bitazonossága (lásd az ENGINE.md „Pontosság” szakaszát).
