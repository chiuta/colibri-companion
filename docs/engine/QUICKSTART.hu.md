# Gyorsindító: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

A `corbul` a clean-room motor parancssora ([ENGINE.hu.md](ENGINE.hu.md)). Ez az oldal csak azokat a parancsokat és kapcsolókat sorolja fel, amelyek az írás pillanatában léteznek a `crates/engine-cli`-ben; a parancssor még bővül, így a végső szó a `corbul help`-é. Az alábbi kapcsolók egy részét a kód olvassa, de a `corbul help` még nem írja ki; ezeket *(még nincs a súgóban)* jelöli. Ami még nem létezik, az **tervezett**.

**Őszinte figyelmeztetés:** a `corbul`t csak apró szintetikus modelleken teszteltük. Számíts szögletes élekre valódi ellenőrzőpontokon, és jelezd őket.

## Fordítás

```bash
cargo build --release -p colibri-rs-cli     # elkészíti: target/release/corbul
```

Egy modell vagy egy mappa `config.json` + `*.safetensors` + `tokenizer.json` fájlokkal, vagy egyetlen `.gguf` fájl.

## Megnézés, letöltés, futtatás

```bash
corbul inspect  ./my-model                  # vagy file.gguf vagy owner/repo[@rev]: architektúra, méret, támogatás
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors; HF_TOKEN a zárt tárolókhoz, HF_ENDPOINT tükörhöz
corbul generate ./my-model --prompt "Szia" --max-tokens 64
corbul chat     ./my-model --system "Légy tömör."
corbul serve    ./my-model --port 8080      # OpenAI / Ollama / Anthropic-kompatibilis, alapból a 127.0.0.1-hez kötve
```

A `generate`, `chat` és `serve` közös kapcsolói: `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--kv f32|f16|q8`, `--no-mmap`, `--threads N`, `--ctx N`, `--model-name`, `--quiet`. A `generate` ezen felül elfogadja a `--prompt-file F` és a `--raw` (csevegősablon nélkül) kapcsolót. A `corbul run` a `generate` álneve.

Több modell kiszolgálása: `--models-dir DIR`, `--alias NÉV=MODELL`, `--max-loaded N`, `--sleep-idle-seconds S`, `--token T` (vagy `CORBUL_TOKEN`), `--keys-file FÁJL`, `--cors EREDETEK`, `--log-prompts`, `--no-log`, `--drain-seconds S`, `--config-dir DIR`.

## Fut majd ezen a gépen?

```bash
corbul doctor                # amit a motor lát: CPU-utasításkészletek, magok, RAM, lemez, érzékelők, energiaszámláló
corbul plan ./my-model --context 8192            # ítélet: FITS / TIGHT / STREAMS / TOO SLOW / WON'T RUN, sebesség, tanácsok
corbul plan ./my-model --ram-budget 8g --calibrate   # tégy úgy, mintha 8 GB-od lenne; előbb egy másodpercig méri a gépet
corbul capabilities [--json]                     # mit tud és mit nem ez a build
```

A sebességek becslések, nem a te modelled mérései.

## A RAM-nál nagyobb

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"                        # MoE: az irányított szakértők lemezről streamelődnek
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --prompt "…"   # sűrű: a rétegek streamelődnek (lemezkorlátos; a --stream-window még nincs a súgóban)
```

## Több gép (LAN-háló)

```bash
# minden segédgépen (a kulcs legalább 8 karakter; vagy állítsd be a CORBUL_MESH_KEY-t)
corbul worker ./my-model --key SECRET --bind 0.0.0.0:7471          # a --no-discovery kihagyja az UDP-felderítést
# a fő gépen
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key SECRET --prompt "…"
corbul generate ./my-model --discover --key SECRET --prompt "…"      # megkeresi a workereket a LAN-on
```

A forgalom hitelesített, de **nem titkosított**: használj megbízható hálózatot vagy VPN-t. Csak loopbacken tesztelve.

## Strukturált kimenet, idézetek, adapterek

```bash
corbul generate M --json --prompt "…"                 # érvényes JSON-kimenet
corbul generate M --json-schema schema.json --prompt "…"      # fájl vagy beágyazott JSON; továbbá --regex MINTA, --grammar FÁJL (GBNF); a négyből csak egyet használj; még nincs a súgóban
corbul generate M --sources notes.txt --sources paper.txt --prompt "Válaszolj idézetekkel" --citations report.json   # idézetzár
corbul generate M --lora ./adapter:0.8                # PEFT- vagy GGUF-adapter skálával
```

A szerver elfogadja a `response_format`, `grammar` és `sources` mezőket, és kínálja a `POST /v1/quotes/verify` végpontot.

## Hosszú beszélgetések, munkamenetek, reprodukálhatóság

```bash
corbul generate M --session ./sessions --prompt "…"             # újrahasználja a leghosszabb mentett prompt-előtagot (nem a --kv-policy-val)
corbul generate M --kv-policy h2o --kv-budget 2048 --prompt "…" # streaming | h2o | snapkv; szándékosan veszteséges
corbul generate M --governor --prompt "…"                       # alkalmazkodik a hőmérséklethez, memóriához és akkumulátorhoz (Linux)
corbul generate M --reproducible --temperature 0 --prompt "hi" --receipt r.json --receipt-key SECRET
corbul verify-receipt r.json --key SECRET --model M            # ellenőrzi a pecsétet (és a súlyokat, ha adsz --model-t)
corbul replay r.json --model M --key SECRET                    # újrafuttatja a generálást és összehasonlít
```

A nyugták az egyszerű generálási ciklussal működnek, MoE-modellekkel nem. Opcionális: `--receipt-topk N`, `--keep-steps`, `--leaves` *(még nincs a súgóban)*.

## Tervezett (még nincs a `corbul`ban)

- Mamba / RWKV / MLA (DeepSeek-stílusú) modellek betöltése, bár a könyvtárban megvannak az építőelemek.
- A román nyelvi eszközök, a bizonytalanság-elemzés, keresés és lookahead, önspekulatív dekódolás, irányítóvektorok, visszakeresés, egyesítés és vízjel (lásd: [INNOVATIONS.hu.md](INNOVATIONS.hu.md)).
- Beágyazási és rerank végpontok (a szerver „not implemented”-et válaszol), kép- és hangbemenet.
- A GPU-háttér és egy webes felület.
