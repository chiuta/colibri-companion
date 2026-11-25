# Inicio rápido: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` es la línea de comandos del motor clean-room ([ENGINE.es.md](ENGINE.es.md)). Cada comando y opción de esta página existe en `crates/engine-cli` y lo imprime `corbul help`; un script (`tools/check_engine_docs.py`) falla si el código tiene una opción, un subcomando o una variable de entorno que esta página no menciona. `corbul help` sigue teniendo la última palabra.

**Aviso honesto:** `corbul` se ha probado solo con modelos sintéticos diminutos y con archivos de referencia de tokenizadores. Los autores no han ejecutado ningún modelo real descargado. Espera asperezas y por favor infórmalas. `corbul capabilities` imprime lo que funciona y lo que no.

## Compilar

```bash
cargo build --release -p colibri-rs-cli     # produces target/release/corbul
```

Un modelo es una carpeta con `config.json` + `*.safetensors` + `tokenizer.json`, o un único archivo `.gguf`.

## Mirar, descargar, ejecutar

```bash
corbul inspect  ./my-model                  # or file.gguf or owner/repo[@rev]: architecture, size, support
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors
corbul generate ./my-model --prompt "Hello" --max-tokens 64
corbul generate ./my-model --prompt-file question.txt --raw   # --raw: no chat template
corbul chat     ./my-model --system "You are concise."
corbul run      ./my-model --prompt "Hello"                   # `run` is `generate`
corbul serve    ./my-model --port 8080                        # bound to 127.0.0.1 by default
```

Define `HF_TOKEN` (o `HUGGING_FACE_HUB_TOKEN`) para repositorios restringidos y `HF_ENDPOINT` para un espejo. Las descargas necesitan `curl` (`wget` anónimo es la alternativa).

Opciones comunes de `generate`, `chat` y `serve`: `--max-tokens N`, `--temperature T`, `--top-p P`, `--top-k K`, `--seed S`, `--kv f32|f16|q8`, `--no-mmap`, `--model-name NAME`, `--threads N` (también `COLIBRI_THREADS`), `--ctx N`, `--quiet`, `--ram-budget SIZE`, `--lora PATH[:S]` (repetible; carpeta PEFT o adaptador GGUF con una escala).

Opciones adicionales del muestreador para `generate` y `chat`:

```bash
corbul generate M --prompt "…" --min-p 0.05 --typical-p 0.9 --repeat-penalty 1.1 \
  --presence-penalty 0.2 --frequency-penalty 0.2 --mirostat 2 --mirostat-tau 5 --mirostat-eta 0.1 \
  --dry-multiplier 0.8 --xtc-probability 0.1
corbul generate M --prompt "…" --sampler-json '{"top_k": 20}'   # any sampler field, applied before the flags above
```

Los modelos de espacio de estados e híbridos (Mamba, RWKV, Jamba, Bamba) se ejecutan con `generate` y `chat` como continuación de texto plano: sin plantilla de chat, sin salida estructurada, sin `serve`.

## ¿Se ejecutará en esta máquina?

```bash
corbul doctor                # CPU instruction sets, chosen kernel, RAM, disk, sensors, GPU probe
corbul plan ./my-model --context 8192          # verdict (fits / tight / streams / too slow / won't run), speed, advice
corbul plan ./my-model --ram-budget 8g --calibrate   # pretend you have 8 GB; time this machine for about a second first
corbul capabilities [--json]                   # what this build can and cannot do (alias: corbul features)
```

Las velocidades son estimaciones, no mediciones de tu modelo. `ENGINE_SIMD` fuerza un kernel, `ENGINE_REPRODUCIBLE=1` activa el modo reproducible, `ENGINE_BACKEND=cpu|gpu|auto` lo lee la biblioteca (`corbul` siempre calcula en la CPU).

## Más grande que la RAM

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"     # MoE: routed experts stream from disk
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --disk-mbps 800 --prompt "…"   # dense: layers stream; speed is bounded by the disk; --disk-mbps (default 500) only feeds the speed estimate it prints
```

## Varias máquinas (malla en LAN)

```bash
# on every helper machine (the key needs 8+ characters; or set CORBUL_MESH_KEY)
corbul worker ./my-model --key MESHKEY12345 --bind 0.0.0.0:7471
corbul worker ./my-model --key-file mesh.key --discovery            # also answer UDP discovery probes (opt-in); --discovery-port N; --no-discovery forces it off
# on the main machine
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key MESHKEY12345 --prompt "…"
corbul generate ./my-model --discover --key MESHKEY12345 --prompt "…"      # broadcast a probe, use workers that hold the same model
corbul generate ./my-model --workers h:7471 --key MESHKEY12345 --wire-f16 --local-speed 2.5 --prompt "…"
```

`--wire-f16` reduce a la mitad el tráfico del estado oculto (y renuncia a la identidad bit a bit); `--local-speed` indica al divisor lo rápida que es esta máquina en relación con los workers; `--discover-target HOST:PORT` y `--discover-wait-ms N` cambian la dirección de la sonda (por defecto `255.255.255.255:47813`) y la espera (por defecto 2000 ms). El tráfico está autenticado pero **no cifrado**: usa una red de confianza o una VPN. Probado solo en loopback.

## Salida estructurada, citas, adaptadores

```bash
corbul generate M --json --prompt "…"                       # valid JSON
corbul generate M --json-schema schema.json --prompt "…"    # a file or inline JSON
corbul generate M --regex '[A-Z]{3}-[0-9]{4}' --prompt "…"  # regular expression
corbul generate M --grammar rules.gbnf --prompt "…"         # GBNF grammar; use only one of the four
corbul generate M --sources notes.txt --sources paper.txt --prompt "Answer with quotes" --citations report.json   # quote-locked
corbul generate M --sources notes.txt --no-lock --prompt "…"   # only check the quotes afterwards
corbul generate M --lora ./adapter:0.8
```

El servidor acepta `response_format`, `grammar` y `sources` en las peticiones y ofrece `POST /v1/quotes/verify`.

## Conversaciones largas, sesiones, reproducibilidad

```bash
corbul generate M --session ./sessions --session-cap 2g --prompt "…"   # reuse the longest saved prompt prefix; --compress shrinks saved files (not with --kv-policy)
corbul generate M --kv-policy h2o --kv-budget 2048 --kv-sinks 4 --prompt "…"   # streaming | h2o | snapkv; lossy by design
corbul generate M --governor --verbose --prompt "…"            # adapt to temperature, memory and battery (Linux); --no-governor turns it off
corbul generate M --lang ro --prompt "…"                         # forbid tokens of foreign scripts; Romanian: no ş ţ
corbul generate M --reproducible --temperature 0 --prompt "hi" --receipt r.json --receipt-key-file receipt.key
corbul verify-receipt r.json --key-file receipt.key --model M      # checks the seal (and the weights if --model is given)
corbul replay r.json --model M --key-file receipt.key              # re-runs the generation and compares
corbul replay r.json --model M --key-file receipt.key --prompt "…" --raw     # when the receipt keeps no prompt text; or --prompt-tokens FILE
```

Los recibos funcionan con el bucle de generación simple, no con modelos MoE ni de espacio de estados. Opcionales: `--receipt-key K` (visible en la lista de procesos; prefiere `--receipt-key-file`), `--receipt-topk N`, `--keep-steps`, `--leaves`.

## Decodificación más rápida

```bash
corbul generate M --draft ./small-model --draft-len 4 --prompt "…"      # a draft model proposes, M verifies
corbul generate M --lookahead --ngram-min 1 --ngram-max 4 --prompt "…"   # prompt-lookup, no extra model
corbul generate M --self-spec --skip-layers auto --prompt "…"            # M drafts with a subset of its layers; or --skip-layers 4,6,8-11
```

Misma respuesta voraz y misma distribución de muestreo; el texto aparece al final. Solo se aplican `--temperature`, `--top-k`, `--top-p`; no con salida estructurada, recibos, sesiones ni la malla.

## Búsqueda, cuantizar, fusionar, dirigir, recuperar, marca de agua, idioma

```bash
corbul search M --prompt "Q" --method beam --width 4 --all --explain --json     # also --length-penalty L
corbul search M --prompt "2+2?" --method best-of-n --samples 5
corbul search M --prompt "2+2?" --method vote --samples 5 --answer number       # or text
corbul autoquant ./model --budget 4.5bpw --recipe recipe.json --report report.txt   # 4.5bpw, 6g or Q4_K; --candidates Q4_K,Q6_K --base Q8_0 --imatrix I --sample-rows 256 --json
corbul quantize ./model out.gguf --type Q4_K_M --imatrix imatrix.dat    # or --recipe recipe.json; --name NAME
corbul merge merge.json --out merged.safetensors --dry-run              # --model PATH (repeatable) --max-memory 8g --overwrite --verbose
corbul steer train M --positive polite.txt --negative rude.txt --out polite.cvec --method mean-diff   # or pca
corbul steer apply M --vector polite.cvec --strength 1 --layers 8-20 --positions last --prompt "…" --no-attention
corbul steer lens M --prompt "The capital of France is"
corbul rag add ./lib notes.txt paper.txt --chunk-chars 800 --overlap 120 --embed-model ./embedder
corbul rag search ./lib --query "refund policy" --k 5 --json
corbul rag ask ./lib M --question "What is the refund period?"             # answers with quote-locked citations
corbul lang detect "Aceasta este o propoziție."                          # or --file text.txt
corbul lang normalize --file text.txt                                    # ș/ț, mojibake, quotes
corbul generate M --prompt "…" --watermark SECRETKEY --wm-gamma 0.25 --wm-delta 2 --wm-context 4
corbul watermark detect M --key SECRETKEY --file answer.txt --no-dedup   # or --text "…"
corbul embed ./bge-small "first text" "second text" --pool cls --json    # a BERT-family folder; or --file lines.txt, --no-normalize
corbul rerank ./ms-marco-minilm --query "refund policy" "doc one" "doc two" --top 3 --json   # needs a cross-encoder
```

No toda combinación de opciones funciona; cada comando imprime un error de uso de una línea cuando no es así. Nada de esto se ha ejecutado con un checkpoint real.

## Servir

```bash
corbul serve --models-dir ./models --host 127.0.0.1 --port 8080 --token-file token.txt --default-model qwen
corbul serve --models-dir ./models --alias fast=qwen-0.5b --max-loaded 2 --sleep-idle-seconds 600
corbul serve --models-dir ./models --keys-file keys.json --allow-admin --max-kv 8g --max-heavy 4
corbul serve --models-dir ./models --cors "https://app.example" --log-prompts --drain-seconds 10 --config-dir ./cfg
corbul serve --encoder ./bge-small --port 8081                                  # embeddings (and /v1/rerank for a cross-encoder) without a decoder
```

Abre `http://127.0.0.1:8080/` para la interfaz web (también `/ui`). La página es pública; cada llamada que hace necesita el token. `--token T` equivale a `CORBUL_TOKEN` (visible en la lista de procesos, así que prefiere `--token-file`). `--keys-file` contiene claves de API con listas de modelos y límites de frecuencia por clave. `--allow-admin` permite a los titulares de claves usar `/api/create`, `/api/delete`, `/models/load` y `/models/unload`; por defecto solo puede el token maestro. `--max-kv` rechaza las peticiones cuya caché KV superaría el tamaño indicado (por defecto 8g); `--max-heavy` limita las peticiones simultáneas de tokenización, plantilla y verificación (por defecto 4). `--no-log` silencia el registro de peticiones. `GET /v1/capabilities` enumera lo que funciona. Las rutas de embeddings funcionan (`/v1/embeddings`, `/api/embed`); el rerank necesita `--encoder`.

## No disponible en `corbul` (todavía)

- Modelos de atención latente multicabezal (al estilo DeepSeek), RWKV en formato GGUF, Zamba/Zamba2, las demás i-quants (IQ1–IQ3), modelos RoBERTa/XLM-R.
- Entradas de visión y audio (la biblioteca tiene los bloques), el back end de GPU.
- Identidad bit a bit del prefill cuantizado ordinario (véase ENGINE.es.md, «Exactitud»).
