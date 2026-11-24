# Início rápido: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` é a linha de comandos do motor clean-room ([ENGINE.pt.md](ENGINE.pt.md)). Esta página lista apenas os comandos e opções que existem em `crates/engine-cli` no momento da escrita; a linha de comandos ainda está a ser alargada, por isso a última palavra é de `corbul help`. Algumas opções abaixo são lidas pelo código mas ainda não aparecem em `corbul help`; estão marcadas *(ainda não na ajuda)*. O que ainda não existe está marcado como **planeado**.

**Aviso honesto:** o `corbul` só foi testado em modelos sintéticos minúsculos. Espera arestas em pontos de controlo reais e comunica-as.

## Compilar

```bash
cargo build --release -p colibri-rs-cli     # produz target/release/corbul
```

Um modelo é uma pasta com `config.json` + `*.safetensors` + `tokenizer.json`, ou um único ficheiro `.gguf`.

## Ver, descarregar, executar

```bash
corbul inspect  ./my-model                  # ou file.gguf ou owner/repo[@rev]: arquitetura, tamanho, suporte
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors; HF_TOKEN para repositórios de acesso restrito, HF_ENDPOINT para um espelho
corbul generate ./my-model --prompt "Olá" --max-tokens 64
corbul chat     ./my-model --system "Sê conciso."
corbul serve    ./my-model --port 8080      # compatível com OpenAI / Ollama / Anthropic, ligado por omissão a 127.0.0.1
```

Opções comuns a `generate`, `chat` e `serve`: `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--kv f32|f16|q8`, `--no-mmap`, `--threads N`, `--ctx N`, `--model-name`, `--quiet`. `generate` aceita ainda `--prompt-file F` e `--raw` (sem modelo de chat). `corbul run` é um alias de `generate`.

Servir vários modelos: `--models-dir DIR`, `--alias NOME=MODELO`, `--max-loaded N`, `--sleep-idle-seconds S`, `--token T` (ou `CORBUL_TOKEN`), `--keys-file FICHEIRO`, `--cors ORIGENS`, `--log-prompts`, `--no-log`, `--drain-seconds S`, `--config-dir DIR`.

## Vai correr nesta máquina?

```bash
corbul doctor                # o que o motor vê: conjuntos de instruções do CPU, núcleos, RAM, disco, sensores, contador de energia
corbul plan ./my-model --context 8192            # veredicto: FITS / TIGHT / STREAMS / TOO SLOW / WON'T RUN, velocidade, conselhos
corbul plan ./my-model --ram-budget 8g --calibrate   # finge que tens 8 GB; primeiro cronometra a máquina durante um segundo
corbul capabilities [--json]                     # o que esta versão pode e não pode fazer
```

As velocidades são estimativas, não medições do teu modelo.

## Maior do que a RAM

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"                        # MoE: os especialistas encaminhados são lidos do disco
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --prompt "…"   # denso: as camadas são transmitidas (limitado pelo disco; --stream-window ainda não na ajuda)
```

## Várias máquinas (malha LAN)

```bash
# em cada máquina auxiliar (a chave tem pelo menos 8 caracteres; ou define CORBUL_MESH_KEY)
corbul worker ./my-model --key SECRET --bind 0.0.0.0:7471          # --no-discovery dispensa a descoberta UDP
# na máquina principal
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key SECRET --prompt "…"
corbul generate ./my-model --discover --key SECRET --prompt "…"      # encontra workers na LAN
```

O tráfego é autenticado mas **não cifrado**: usa uma rede de confiança ou uma VPN. Testado apenas em loopback.

## Saída estruturada, citações, adaptadores

```bash
corbul generate M --json --prompt "…"                 # saída JSON válida
corbul generate M --json-schema schema.json --prompt "…"      # ficheiro ou JSON em linha; também --regex PADRÃO, --grammar FICHEIRO (GBNF); usa só uma das quatro; ainda não na ajuda
corbul generate M --sources notes.txt --sources paper.txt --prompt "Responde com citações" --citations report.json   # citações bloqueadas
corbul generate M --lora ./adapter:0.8                # adaptador PEFT ou GGUF com um fator
```

O servidor aceita `response_format`, `grammar` e `sources`, e oferece `POST /v1/quotes/verify`.

## Conversas longas, sessões, reprodutibilidade

```bash
corbul generate M --session ./sessions --prompt "…"             # reutiliza o prefixo de prompt guardado mais longo (não com --kv-policy)
corbul generate M --kv-policy h2o --kv-budget 2048 --prompt "…" # streaming | h2o | snapkv; com perda por construção
corbul generate M --governor --prompt "…"                       # adapta-se à temperatura, memória e bateria (Linux)
corbul generate M --reproducible --temperature 0 --prompt "hi" --receipt r.json --receipt-key SECRET
corbul verify-receipt r.json --key SECRET --model M            # verifica o selo (e os pesos se deres --model)
corbul replay r.json --model M --key SECRET                    # repete a geração e compara
```

Os recibos funcionam com o ciclo de geração simples, não com modelos MoE. Opcionais: `--receipt-topk N`, `--keep-steps`, `--leaves` *(ainda não na ajuda)*.

## Planeado (ainda não no `corbul`)

- Carregar modelos Mamba / RWKV / MLA (estilo DeepSeek), embora a biblioteca tenha os blocos.
- As ferramentas de língua romena, a análise de incerteza, pesquisa e lookahead, descodificação autoespeculativa, vetores de direção, recuperação, fusão e marca de água (vê [INNOVATIONS.pt.md](INNOVATIONS.pt.md)).
- Endpoints de embedding e rerank (o servidor responde «not implemented»), entradas de imagem e áudio.
- O backend de GPU e uma interface web.
