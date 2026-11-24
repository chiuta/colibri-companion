# O que este motor tem de diferente

[English](INNOVATIONS.md) · [Română](INNOVATIONS.ro.md) · [Français](INNOVATIONS.fr.md) · [Italiano](INNOVATIONS.it.md) · [Español](INNOVATIONS.es.md) · [Português](INNOVATIONS.pt.md) · [Deutsch](INNOVATIONS.de.md) · [Magyar](INNOVATIONS.hu.md)

Notas em linguagem simples sobre as funcionalidades que a maioria dos outros motores locais não tem. Cada uma diz o que faz, como se lá chega e o que não consegue fazer. Todas foram testadas apenas com modelos sintéticos; os autores não executaram nenhuma com um checkpoint real. «Em `corbul`» significa que hoje existe um comando ou uma opção (`corbul help`); «apenas biblioteca» significa código Rust sem comando. Ver [ENGINE.pt.md](ENGINE.pt.md) e [QUICKSTART.pt.md](QUICKSTART.pt.md).

## Integridade, hardware e memória

### Malha em LAN

Módulo `mesh`. Em `corbul`: `worker`, `generate --workers`, `--discover`, `--key`, `--wire-f16`. Um modelo demasiado grande para uma máquina é cortado em intervalos de camadas, um por computador. Entre eles viaja apenas o estado oculto. As máquinas provam que partilham uma chave (HMAC-SHA256 mútuo), cada trama leva um MAC, os tamanhos do handshake são limitados, um worker caído é descartado e as suas camadas redistribuídas. Os logits f32 densos coincidem com os de uma só máquina. A descoberta UDP é opcional: um worker só responde às sondas se tiver sido iniciado com `--discovery` ou `--discovery-port` (`--no-discovery` força-o a desligado), e o `--discover` do coordenador difunde uma sonda e espera as respostas dos workers que tenham o mesmo modelo. *Não consegue:* cifrar (usa uma LAN de confiança ou uma VPN); só foi testada em loopback; a velocidade face a uma só máquina não foi medida.

### Streaming de camadas

Módulo `stream_layers`. Em `corbul`: `--stream-layers`, `--ram-budget`, `--stream-window`, `--disk-mbps` (para estimativas). Executa um modelo denso maior do que a RAM mantendo em memória uma pequena janela de camadas e pré-carregando numa thread de fundo. Os resultados coincidem com os do modelo em RAM. *Não consegue:* evitar que cada token leia todas as camadas, pelo que a velocidade fica limitada pela do disco; apenas camadas densas (os modelos de mistura de especialistas transmitem antes os seus especialistas encaminhados, com `--ram-budget`).

### Citações bloqueadas à citação literal

Módulo `quotelock`. Em `corbul`: `--sources FILE` (repetível), `--no-lock`, `--citations FILE`; o servidor aceita `sources` e oferece `POST /v1/quotes/verify`. Durante a descodificação, o texto entre « e » tem de ser uma passagem literal de uma fonte (tolerante a maiúsculas, espaços, estilos de aspas e diacríticos romenos); `[S3]«` limita uma citação à fonte 3. Depois, um verificador classifica cada citação (literal, fonte errada, truncada, alterada, fundida, sem suporte) e pontua as frases fora das citações; o relatório indica quando se atingiu um limite de trabalho (`limit_hit`). *Não consegue:* tornar verdadeiro o texto que rodeia as citações; as respostas em streaming não trazem bloco de citações.

### Planeador, governador e medidor de energia

Módulo `governor` (e `plan`, `hw`). Em `corbul`: `plan`, `doctor`, `--governor`, `--no-governor`, `--verbose`, `--calibrate`. Antes de carregar, diz se o modelo cabe, é transmitido, corre devagar ou não pode correr, com uma velocidade estimada. Durante a execução, o governador lê temperatura, memória e bateria (Linux) e aconselha threads e pausas sem oscilar. *Não consegue:* dar velocidades que não sejam estimativas (`--calibrate` mede a tua máquina durante cerca de um segundo); a energia é uma suposição a menos que um contador RAPL seja legível; a intensidade de carbono (400 g/kWh) é um valor provisório.

### Políticas de cache KV e instantâneos de sessão

Módulos `kvpolicy`, `snapshot`. Em `corbul`: `--kv-policy streaming|h2o|snapkv`, `--kv-budget`, `--kv-sinks`, `--session DIR`, `--session-cap`, `--compress`. Mantêm as conversas longas dentro de um orçamento de memória (sumidouros de atenção, heavy hitters, compressão do prompt) e guardam ou retomam uma conversa a partir de um ficheiro com soma de verificação, reutilizando o prefixo guardado mais longo. Os instantâneos podem levar um HMAC com chave. *Não consegue:* evitar que a remoção seja com perdas por desenho; `--kv-policy` e `--session` não se podem combinar; conversas muito longas podem ultrapassar o contexto de treino.

### Modo reprodutível e recibos

Módulos `simd` (interruptor), `receipt`. Em `corbul`: `--reproducible`, `--receipt FILE`, `--receipt-key`, `--receipt-key-file`, `--receipt-topk`, `--keep-steps`, `--leaves`, `verify-receipt`, `replay`. Um recibo regista o modelo (uma impressão de Merkle dos pesos), o tokenizador, as definições do amostrador, o prompt e a saída com uma cadeia de hashes, selado com SHA-256 e uma chave HMAC opcional; `replay` volta a executá-lo. Os pesos danificados são localizados até um tensor e um fragmento. *Não consegue:* provar o que quer que seja a um desconhecido sem a chave HMAC ou uma nova execução (não há assinaturas de chave pública); os modelos MoE e de espaço de estados são recusados; ARM por testar; o modo reprodutível custa velocidade.

### Ferramentas de língua com o romeno primeiro

Módulo `lang`. Em `corbul`: `lang detect`, `lang normalize`, `--lang CODE` (não no servidor). Normaliza ș/ț e ş/ţ, repara mojibake, identifica 27 línguas, proíbe escritas estrangeiras durante a descodificação (romeno: sem ş ţ), formata números, dinheiro e datas romenos, restaura diacríticos em falta sinalizando cada palavra incerta. *Não consegue:* ir além de um léxico de cerca de 650 formas, cuja restauração não foi medida em nenhum corpus; as palavras ambíguas são resolvidas por algumas regras de contexto; a identificação da língua precisa de cerca de cinco palavras.

### Análise de incerteza

Módulo `confidence`. Apenas biblioteca (exemplo de demonstração `confidence_demo`). Estatísticas por token, palavra, frase e resposta para um mapa de calor, uma heurística transparente de risco de alucinação, pontuações de autoconsistência e ferramentas de calibração. *Não consegue:* ser mais do que uma heurística; a pontuação de risco não é um detetor de verdade.

## Computação em tempo de teste, ajuste, recuperação

### Pesquisa e lookahead

Módulo `search`. Em `corbul`: `search --method beam|best-of-n|vote` (`--width`, `--samples`, `--answer number|text`, `--length-penalty`, `--all`, `--explain`, `--json`) e `--lookahead` em `generate` (`--ngram-min`, `--ngram-max`). Pesquisa em feixe, best-of-N, votação por autoconsistência, uma árvore de conversa (biblioteca), descodificação especulativa por procura no prompt. *Não consegue:* ajudar com o lookahead exceto quando o texto repete texto anterior; best-of-N precisa de um bom avaliador.

### Descodificação especulativa

Módulos `spec`, `selfspec`. Em `corbul`: `--draft MODEL`, `--lookahead`, `--self-spec`, `--skip-layers auto|LIST`, `--draft-len`. Mesma resposta gulosa e mesma distribuição de amostragem com menos passagens completas; a saída aparece no fim. Apenas `--temperature`, `--top-k`, `--top-p`; não com saída estruturada, recibos, sessões nem a malha. *Não consegue:* prometer uma aceleração; a autoespeculação recorre a passos simples com anéis de janela deslizante que dariam a volta, caches compactadas e camadas descarregadas.

### Quantização mista automática e o quantizador

Módulos `autoquant`, `quantize`. Em `corbul`: `autoquant FOLDER --budget 4.5bpw|6g|Q4_K` (`--recipe`, `--report`, `--candidates`, `--base`, `--imatrix`, `--sample-rows`) e `quantize IN OUT.gguf --type NAME | --recipe FILE`. Mede o erro dos pesos por grupo de tensores em cada formato e escolhe formatos dentro de um orçamento de tamanho (mochila exata ou gulosa). *Não consegue:* estimar a qualidade das respostas, apenas o erro dos pesos; supõe que os danos se somam; as i-quants não existem.

### Vetores de direção e lente de logits

Módulo `steer`. Em `corbul`: `steer train|apply|lens` (`--positive`, `--negative`, `--method mean-diff|pca`, `--out`, `--vector`, `--strength`, `--layers`, `--positions`, `--no-attention`). Aprende uma direção a partir de dois conjuntos de prompts e soma-a ao fluxo residual sem alterar os pesos; a lente mostra o que cada camada preveria. *Não consegue:* funcionar com modelos de espaço de estados, saída estruturada, especulação, recibos nem sessões; um vetor de um ajuste fino é aceite por outro com a mesma forma.

### Recuperação local

Módulo `rag`. Em `corbul`: `rag add|search|ask` (`--query`, `--question`, `--k`, `--chunk-chars`, `--overlap`, `--embed-model`). Divisão em partes que conhece o romeno, um índice HNSW em Rust puro, BM25 com normalização de diacríticos, fusão de classificações, passagens numeradas `[S1]…` que alimentam as citações bloqueadas à citação literal, ficheiros com soma de verificação. *Não consegue:* ir além do incorporador por defeito, que é um incorporador por hash (palavras e partes de palavras, não significado); `--embed-model` usa estados ocultos de um modelo generativo (fraco); a CLI reindexa a cada chamada e lê texto simples, não PDF nem ficheiros do Office (a biblioteca tem auxiliares de ingestão).

### Fusão de modelos

Módulo `merge`. Em `corbul`: `merge CONFIG.json` (`--out`, `--model`, `--dry-run`, `--max-memory`, `--overwrite`, `--verbose`). Linear, SLERP, aritmética de tarefas, TIES, DARE (com semente) e cortes de camadas, um tensor de cada vez. As formas e os intervalos são verificados antes de escrever; as ligações simbólicas não são seguidas. *Não consegue:* garantir a qualidade; só escreve pesos safetensors, nunca quantizados, e `config.json` e o tokenizador copias tu.

### Marca de água de texto

Módulo `watermark`. Em `corbul`: `generate --watermark KEY` (`--wm-gamma`, `--wm-delta`, `--wm-context`) e `watermark detect MODEL --key K --text T | --file F` (`--no-dedup`). Um viés de «lista verde» com chave e um detetor de pontuação z. *Não consegue:* dar uma prova; os textos curtos, o código e as citações levam pouco sinal; a paráfrase elimina-a; quem tiver a chave pode falsificá-la; não haver marca não prova autoria humana.

## Multimodal (apenas biblioteca)

### Visão

Módulos `vision`, `imageio`. Um codificador ao estilo CLIP/SigLIP, um projetor, mosaico e um gancho `Model::forward_segments` para introduzir embeddings num prompt; descodificadores de PNG, JPEG (base e progressivo), BMP e PNM escritos a partir das especificações, com limites de tamanho. *Não consegue:* ser usada a partir de `corbul` (apenas texto); não foi executado nenhum checkpoint real.

### Áudio

Módulo `audio`. Leitura e escrita de WAV, reamostragem, FFT, características log-mel, deteção de atividade de voz, um executor com a disposição do Whisper. *Não consegue:* ser usado a partir de `corbul`; não foi executado nenhum checkpoint real do Whisper.

## Espaço de estados, híbridos e GPU (ver ENGINE.pt.md)

Os módulos `ssm`, `hybrid`, `mla` e o crate `companion-engine-gpu` são descritos, com os seus limites, em ENGINE.pt.md.
