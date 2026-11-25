# Início rápido: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` é a linha de comandos do motor clean-room ([ENGINE.pt.md](ENGINE.pt.md)). Cada comando e opção desta página existe em `crates/engine-cli` e é impresso por `corbul help`; um script (`tools/check_engine_docs.py`) falha se o código tiver uma opção, um subcomando ou uma variável de ambiente que esta página não mencione. `corbul help` continua a ter a última palavra.

**Aviso honesto:** `corbul` foi testado apenas com modelos sintéticos minúsculos e com ficheiros de referência de tokenizadores. Os autores não executaram nenhum modelo real descarregado. Conta com arestas por limar e, por favor, reporta-as. `corbul capabilities` imprime o que funciona e o que não funciona.

## Compilar

```bash
cargo build --release -p colibri-rs-cli     # produces target/release/corbul
```

Um modelo é uma pasta com `config.json` + `*.safetensors` + `tokenizer.json`, ou um único ficheiro `.gguf`.

## Ver, descarregar, executar

```bash
corbul inspect  ./my-model                  # or file.gguf or owner/repo[@rev]: architecture, size, support
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors
corbul generate ./my-model --prompt "Hello" --max-tokens 64
corbul generate ./my-model --prompt-file question.txt --raw   # --raw: no chat template
corbul chat     ./my-model --system "You are concise."
corbul run      ./my-model --prompt "Hello"                   # `run` is `generate`
corbul serve    ./my-model --port 8080                        # bound to 127.0.0.1 by default
```

Define `HF_TOKEN` (ou `HUGGING_FACE_HUB_TOKEN`) para repositórios restritos e `HF_ENDPOINT` para um espelho. Os descarregamentos precisam de `curl` (`wget` anónimo é a alternativa).

Opções comuns de `generate`, `chat` e `serve`: `--max-tokens N`, `--temperature T`, `--top-p P`, `--top-k K`, `--seed S`, `--kv f32|f16|q8`, `--no-mmap`, `--model-name NAME`, `--threads N` (também `COLIBRI_THREADS`), `--ctx N`, `--quiet`, `--ram-budget SIZE`, `--lora PATH[:S]` (repetível; pasta PEFT ou adaptador GGUF com uma escala).

Opções adicionais do amostrador para `generate` e `chat`:

```bash
corbul generate M --prompt "…" --min-p 0.05 --typical-p 0.9 --repeat-penalty 1.1 \
  --presence-penalty 0.2 --frequency-penalty 0.2 --mirostat 2 --mirostat-tau 5 --mirostat-eta 0.1 \
  --dry-multiplier 0.8 --xtc-probability 0.1
corbul generate M --prompt "…" --sampler-json '{"top_k": 20}'   # any sampler field, applied before the flags above
```

Os modelos de espaço de estados e híbridos (Mamba, RWKV, Jamba, Bamba) correm com `generate` e `chat` como continuação de texto simples: sem modelo de conversa, sem saída estruturada, sem `serve`.

## Vai correr nesta máquina?

```bash
corbul doctor                # CPU instruction sets, chosen kernel, RAM, disk, sensors, GPU probe
corbul plan ./my-model --context 8192          # verdict (fits / tight / streams / too slow / won't run), speed, advice
corbul plan ./my-model --ram-budget 8g --calibrate   # pretend you have 8 GB; time this machine for about a second first
corbul capabilities [--json]                   # what this build can and cannot do (alias: corbul features)
```

As velocidades são estimativas, não medições do teu modelo. `ENGINE_SIMD` força um kernel, `ENGINE_REPRODUCIBLE=1` ativa o modo reprodutível, `ENGINE_BACKEND=cpu|gpu|auto` é lido pela biblioteca (`corbul` calcula sempre na CPU).

## Maior do que a RAM

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"     # MoE: routed experts stream from disk
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --disk-mbps 800 --prompt "…"   # dense: layers stream; speed is bounded by the disk; --disk-mbps (default 500) only feeds the speed estimate it prints
```

## Várias máquinas (malha em LAN)

```bash
# on every helper machine (the key needs 8+ characters; or set CORBUL_MESH_KEY)
corbul worker ./my-model --key MESHKEY12345 --bind 0.0.0.0:7471
corbul worker ./my-model --key-file mesh.key --discovery            # also answer UDP discovery probes (opt-in); --discovery-port N; --no-discovery forces it off
# on the main machine
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key MESHKEY12345 --prompt "…"
corbul generate ./my-model --discover --key MESHKEY12345 --prompt "…"      # broadcast a probe, use workers that hold the same model
corbul generate ./my-model --workers h:7471 --key MESHKEY12345 --wire-f16 --local-speed 2.5 --prompt "…"
```

`--wire-f16` reduz a metade o tráfego do estado oculto (e abdica da identidade bit a bit); `--local-speed` diz ao divisor a rapidez desta máquina relativamente aos workers; `--discover-target HOST:PORT` e `--discover-wait-ms N` alteram o endereço da sonda (por defeito `255.255.255.255:47813`) e a espera (por defeito 2000 ms). O tráfego é autenticado mas **não cifrado**: usa uma rede de confiança ou uma VPN. Testado apenas em loopback.

## Saída estruturada, citações, adaptadores

```bash
corbul generate M --json --prompt "…"                       # valid JSON
corbul generate M --json-schema schema.json --prompt "…"    # a file or inline JSON
corbul generate M --regex '[A-Z]{3}-[0-9]{4}' --prompt "…"  # regular expression
corbul generate M --grammar rules.gbnf --prompt "…"         # GBNF grammar; use only one of the four
corbul generate M --sources notes.txt --sources paper.txt --prompt "Answer with quotes" --citations report.json   # quote-locked
corbul generate M --sources notes.txt --no-lock --prompt "…"   # only check the quotes afterwards
corbul generate M --lora ./adapter:0.8
```

O servidor aceita `response_format`, `grammar` e `sources` nos pedidos e oferece `POST /v1/quotes/verify`.

## Conversas longas, sessões, reprodutibilidade

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

Os recibos funcionam com o ciclo de geração simples, não com modelos MoE nem de espaço de estados. Opcionais: `--receipt-key K` (visível na lista de processos; prefere `--receipt-key-file`), `--receipt-topk N`, `--keep-steps`, `--leaves`.

## Descodificação mais rápida

```bash
corbul generate M --draft ./small-model --draft-len 4 --prompt "…"      # a draft model proposes, M verifies
corbul generate M --lookahead --ngram-min 1 --ngram-max 4 --prompt "…"   # prompt-lookup, no extra model
corbul generate M --self-spec --skip-layers auto --prompt "…"            # M drafts with a subset of its layers; or --skip-layers 4,6,8-11
```

Mesma resposta gulosa e mesma distribuição de amostragem; o texto aparece no fim. Aplicam-se apenas `--temperature`, `--top-k`, `--top-p`; não com saída estruturada, recibos, sessões nem a malha.

## Pesquisa, quantizar, fundir, orientar, recuperar, marca de água, língua

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

Nem toda a combinação de opções funciona; cada comando imprime um erro de utilização de uma linha quando assim é. Nada disto foi executado com um checkpoint real.

## Servir

```bash
corbul serve --models-dir ./models --host 127.0.0.1 --port 8080 --token-file token.txt --default-model qwen
corbul serve --models-dir ./models --alias fast=qwen-0.5b --max-loaded 2 --sleep-idle-seconds 600
corbul serve --models-dir ./models --keys-file keys.json --allow-admin --max-kv 8g --max-heavy 4
corbul serve --models-dir ./models --cors "https://app.example" --log-prompts --drain-seconds 10 --config-dir ./cfg
corbul serve --encoder ./bge-small --port 8081                                  # embeddings (and /v1/rerank for a cross-encoder) without a decoder
```

Abre `http://127.0.0.1:8080/` para a interface web (também `/ui`). A página é pública; cada chamada que faz precisa do token. `--token T` equivale a `CORBUL_TOKEN` (visível na lista de processos, por isso prefere `--token-file`). `--keys-file` contém chaves de API com listas de modelos e limites de taxa por chave. `--allow-admin` permite aos titulares de chaves usar `/api/create`, `/api/delete`, `/models/load` e `/models/unload`; por defeito só o token mestre pode. `--max-kv` recusa pedidos cuja cache KV excederia o tamanho indicado (por defeito 8g); `--max-heavy` limita os pedidos simultâneos de tokenização, modelo e verificação (por defeito 4). `--no-log` silencia o registo de pedidos. `GET /v1/capabilities` enumera o que funciona. As rotas de embeddings funcionam (`/v1/embeddings`, `/api/embed`); o rerank precisa de `--encoder`.

## Não disponível em `corbul` (ainda)

- Modelos de atenção latente multicabeça (ao estilo DeepSeek), RWKV em formato GGUF, Zamba/Zamba2, as restantes i-quants (IQ1–IQ3), modelos RoBERTa/XLM-R.
- Entradas de visão e áudio (a biblioteca tem os blocos), o back end de GPU.
- Identidade bit a bit do prefill quantizado normal (ver ENGINE.pt.md, «Exatidão»).
