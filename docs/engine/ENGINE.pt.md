# O motor clean-room (`companion-engine` e `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Outros documentos: [INNOVATIONS.pt.md](INNOVATIONS.pt.md) · [QUICKSTART.pt.md](QUICKSTART.pt.md) · [CLEANROOM.pt.md](CLEANROOM.pt.md) · [CHANGES-SECURITY.pt.md](CHANGES-SECURITY.pt.md)

**Estado: alfa.** Tudo o que se segue foi testado com pequenos checkpoints sintéticos, com ficheiros de referência de tokenizadores e com texto de amostra gravado. **Os autores não executaram neste motor nenhum modelo real descarregado do Hugging Face**, nenhuma GPU executou nada disto, e os caminhos de código de ARM, macOS e Windows não foram executados aqui. Lê «Verificado e não verificado» antes de confiares numa afirmação. `corbul capabilities` imprime os mesmos limites a partir do próprio binário.

## O que é

Um motor local de inferência para modelos de linguagem, escrito de raiz em Rust a partir de descrições públicas de formatos de ficheiro e de artigos publicados.

| Crate | O que é |
|---|---|
| `crates/engine` (`companion-engine`) | A biblioteca: formatos de ficheiro, tokenizador, quantização, passagens para a frente, amostragem e as ferramentas adicionais de INNOVATIONS.pt.md. Dependências em tempo de execução: apenas `serde_json` e `memmap2`. |
| `crates/engine-cli` (`corbul`) | A linha de comandos e um servidor local compatível com OpenAI / Ollama / Anthropic com interface web integrada. Depende da biblioteca e de `serde_json`. |
| `crates/engine-gpu` (`companion-engine-gpu`) | Kernels WGSL e uma emulação deles em CPU. O executor em dispositivo (`wgpu`, `pollster`) fica atrás da funcionalidade de cargo `wgpu`, desativada por defeito. **`corbul` não usa este crate.** |

É independente da aplicação de ambiente de trabalho Colibri Companion e não a substitui; esta continua a controlar o colibri de JustVugg. Usar este motor como back end alternativo da aplicação está planeado, não feito.

## O que suporta

**Ficheiros.** GGUF v2 e v3 (o cabeçalho e a tabela de tensores são lidos sem tocar nos pesos); um escritor GGUF em streaming para saídas maiores do que a RAM; safetensors, único ou fragmentado com índice; `config.json`, `tokenizer.json` e modelos de conversa do Hugging Face; um descarregador de modelos (módulo `hub`, através de `curl`, com `wget` anónimo como alternativa; retoma, verificações SHA-256, descarregamentos fixados a um commit, limites de tamanho, recusa de redirecionamentos para http sem cifra).

**Quantização.** Descodificam-se: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K (14 tipos). Existem codificadores para os mesmos tipos, com ponderação por matriz de importância e receitas por tensor. Os pesos permanecem quantizados em memória; o produto matriz–vetor nunca expande a matriz inteira. **Não suportados:** as i-quants (IQ\*) e qualquer tipo que não esteja na lista.

**Arquiteturas.**

| Família | Estado em `corbul` |
|---|---|
| Descodificadores densos: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 e semelhantes (norma RMS ou de camada, GQA/MQA, RoPE com escala linear, Llama-3, YaRN e LongRoPE e rotação parcial, QK-norm, vieses, soft-capping, janelas deslizantes, MLP com ou sem porta) | Carregados e executados (modelo de conversa, serve, todas as opções) |
| Mistura de especialistas: Mixtral, Qwen-MoE, especialistas empilhados de GGUF, especialistas partilhados, especialistas encaminhados transmitidos do disco com cache LRU | Carregados e executados |
| Espaço de estados: Mamba-1, Mamba-2, RWKV-6, RWKV-7 (módulo `ssm`) | Executados por `generate` e `chat` como **continuação de texto simples**: sem modelo de conversa, sem saída estruturada, sem `serve`. RWKV em formato GGUF não é suportado (apenas safetensors) |
| Híbridos: Jamba e Bamba (módulo `hybrid`) | Igual à linha do espaço de estados. Zamba e Zamba2 não são suportados |
| Atenção latente multicabeça (DeepSeek-V2/V3, Kimi-K2; módulo `mla`) | **Apenas biblioteca.** O bloco existe e é testado quanto à coerência interna, mas `corbul` não monta um modelo assim (`corbul capabilities` diz que não) |
| Codificador de visão (ViT, CLIP, SigLIP; módulo `vision`, descodificadores em `imageio`) e front-end de voz mais um executor com a disposição do Whisper (módulo `audio`) | **Apenas biblioteca.** `corbul` aceita apenas texto |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, modelos de codificador e de codificador-descodificador (T5, BERT…), tokenizadores WordPiece | Não suportados; `corbul` di-lo em vez de adivinhar |
| Endpoints de embeddings e de reranking | Respondem 501 (não implementado) |

**Tokenizador.** BPE ao nível do byte e modelos ao estilo SentencePiece, regras de pré-tokenização para 12 famílias, tokens adicionados, recurso a bytes, modelos a partir de `tokenizer_config.json` e dos metadados GGUF. WordPiece/BERT não é suportado.

**Amostragem.** Temperatura, top-k, top-p, min-p, typical-p, penalizações de repetição / presença / frequência, Mirostat 1 e 2, DRY, XTC, uma especificação JSON do amostrador, aleatoriedade com semente; descodificação restringida por JSON, JSON Schema, expressão regular ou gramática GBNF (com orçamentos de trabalho para que uma gramática hostil não possa bloquear o processo); viés de logits e cadeias de paragem através do servidor. Descodificação especulativa: modelo rascunho, procura no prompt, autoespeculação (ver INNOVATIONS.pt.md).

**Servidor.** `corbul serve` oferece `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/messages` e `/v1/messages/count_tokens` (estilo Anthropic), os endpoints do Ollama `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/create`, `/api/delete`, `/api/ps`, `/api/version`, a gestão de modelos `/models`, `/models/load`, `/models/unload`, além de `/metrics`, `/props`, `/slots`, `/tokenize`, `/detokenize`, `/apply-template`, `/health`, `/v1/capabilities`, `/v1/quotes/verify` e a interface web em `/` e `/ui` (oito idiomas). A chamada de ferramentas é suportada através dos modelos de conversa. As rotas de embeddings e rerank (`/v1/embeddings`, `/v1/rerank`, `/api/embed`, `/api/embeddings`) respondem 501. As respostas em streaming não trazem bloco de citações de citações literais.

**Hardware.** Os kernels de CPU são escolhidos em tempo de execução: escalar, avx2+fma, avx2+fma+avxvnni, avx512, avx512+vnni (x86-64), neon e neon+dotprod (aarch64); `ENGINE_SIMD` força um. Um pool de threads persistente, deteção de hardware e de NUMA, e um planeador (`corbul plan`, `corbul doctor`) decidem o que fica na RAM. Os kernels de GPU nunca foram executados numa GPU (ver abaixo).

## Exatidão

- **Pesos f32 densos:** os caminhos em lote e de um só token, a malha, o streaming de camadas e as caches dão logits idênticos bit a bit.
- **Pesos quantizados:** a identidade bit a bit entre a avaliação em lote e a de um só token só se verifica com `exact_batch` ativo. A verificação, as rondas de verificação autoespeculativa e o executor em lote ativam-no. O prefill normal usa um caminho mais rápido de descodificar e depois dividir em blocos, e difere da avaliação de um só token apenas por ruído de arredondamento.
- **Modo reprodutível** (`--reproducible` ou `ENGINE_REPRODUCIBLE=1`): resultados idênticos entre as variantes de kernel x86 e os números de threads que pudemos testar. ARM não foi testado. Custa velocidade.
- Mirostat 1 usa na sua estimativa o número de candidatos restantes, não o tamanho do vocabulário (desvio documentado).

## Verificado e não verificado

| Área | Estado |
|---|---|
| Testes unitários, de integração, adversariais e de fuzzing com checkpoints sintéticos (Linux x86-64). As contagens mudam a cada commit; executa `cargo test --workspace` para o número atual | Passam na última execução completa. Os ficheiros hostis e truncados devolvem erros. |
| Tokenizador contra os ficheiros de referência do llama.cpp, 12 famílias | 46 de 46 casos iguais (ficheiros de referência obtidos do GitHub; `crates/engine/tests/tokenizer_real.rs`, `ENGINE_VOCAB_DIR`). Apenas o tokenizador, não modelos completos. |
| Modelo real do Hugging Face através de `corbul` (qualquer família) | **Não executado pelos autores.** O sandbox de compilação não conseguia chegar a huggingface.co. Um job manual de CI executa um llama GGUF minúsculo (`stories260K`) pela CLI; ainda não foi executado |
| Permutação de linhas Q/K ao estilo llama.cpp, variantes de metadados GGUF | Implementadas a partir de descrições públicas; não verificadas com ficheiros reais |
| Quantizadores e descodificadores de quantização | Blocos calculados à mão e ciclos de ida e volta; não comparados com ficheiros escritos por outras ferramentas |
| Execução em GPU | **Não executada.** Os kernels são validados apenas por `naga` e por uma emulação em Rust; `corbul` não os usa |
| ARM / NEON, macOS, Windows | **Não executados aqui.** As sondas são testadas com texto de amostra gravado. O workflow de CI tem uma matriz para eles mas nunca foi executado |
| Mamba, RWKV, Jamba/Bamba, MLA, visão, áudio | Apenas testes de coerência interna; não foi usado nenhum checkpoint real |
| Malha | Apenas loopback; autenticada, não cifrada; sem medição de velocidade |
| Velocidade | **Sem números fiáveis.** A máquina de compilação tinha dois núcleos partilhados. As velocidades do piloto automático são estimativas |
| Licenças das dependências | Verificadas com `cargo-deny` no commit b496ba5; as alterações posteriores não foram reverificadas |
| Traduções dos documentos | Assistidas por máquina; ainda sem revisão por falantes nativos |

## Variáveis de ambiente

| Variável | Efeito |
|---|---|
| `COLIBRI_THREADS` | Tamanho do pool de threads global (por defeito: núcleos utilizáveis). A biblioteca lê-a primeiro |
| `ENGINE_SIMD` | Força uma variante de kernel pelo nome (benchmarks, relatórios de erros) |
| `ENGINE_REPRODUCIBLE` | `1`, `true` ou `on`: modo reprodutível desde o início |
| `ENGINE_BACKEND` | `cpu`, `gpu` ou `auto` (não definida = auto). `corbul` corre na CPU em ambos os casos |
| `ENGINE_GPU`, `ENGINE_GPU_API`, `ENGINE_GPU_SOFTWARE` | Seleção de adaptador para o crate de GPU: índice, API gráfica, permitir um adaptador por software. Não usadas por `corbul` |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Token de acesso para repositórios restritos (`download`, `inspect owner/repo`) |
| `HF_ENDPOINT` | Espelho do anfitrião do Hugging Face |
| `CORBUL_TOKEN` | Token bearer partilhado de `corbul serve` (igual a `--token`) |
| `CORBUL_MESH_KEY` | Chave da malha para `worker` e `--workers`/`--discover` (igual a `--key`) |
| `CORBUL_CONFIG_DIR` | Onde ficam guardados os Modelfiles criados por `POST /api/create` (igual a `--config-dir`; por defeito `~/.config/corbul`) |
| `NUMBER_OF_PROCESSORS`, `PROCESSOR_IDENTIFIER` | A sonda de hardware lê-as no Windows; tu não as defines |
| `ENGINE_VOCAB_DIR`, `ENGINE_VOCAB_ONLY`, `ENGINE_REQUIRE_VOCAB`, `ENGINE_REQUIRE_TOOLS`, `ENGINE_PRINT_GOLDEN`, `ADVERSARIAL_N` | Controlos da bateria de testes: pasta de vocabulários, restringir famílias, fazer falhar um teste ignorado, fazer falhar a ausência de `curl`/`wget`, regenerar os hashes de referência, tamanho da execução adversarial |

## Declaração clean-room

Não foi lido nem copiado código do colibri, do llama.cpp nem de qualquer outro motor de inferência ao escrever este motor. Os formatos e os algoritmos vêm de especificações públicas, artigos e disposições de ficheiro. O modelo que escreveu o código foi treinado com texto público que pode incluir esses projetos; ver [CLEANROOM.pt.md](CLEANROOM.pt.md) para o processo e os seus limites.

## Comparação com outros motores locais

Não fizemos nenhum benchmark frente a frente, por isso aqui não há afirmações de velocidade nem de qualidade.

- Os motores estabelecidos (llama.cpp, Ollama, mistral.rs, vLLM) foram executados com milhares de checkpoints reais, têm back ends de GPU que funcionam, cobrem mais arquiteturas e formatos de quantização, incluindo as i-quants, e tratam modelos de visão. Este motor ainda não tem nada dessa evidência. Se precisas de algo que funcione hoje com um modelo grande concreto, usa um deles.
- O que este motor tenta fazer de outra maneira: ferramentas de integridade incorporadas (citações bloqueadas à citação literal, modo reprodutível com recibos, análise de incerteza, um planeador que diz «isto não vai correr» em vez de falhar), ferramentas de língua com o romeno em primeiro lugar, modelos maiores do que a tua RAM ou do que a tua máquina, uma base de código pequena em Rust com duas dependências de biblioteca, e apenas licenças permissivas.

## Mapa da biblioteca

Formatos e carregamento: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`, `imageio`. Modelos: `arch`, `model`, `moe`, `mla`, `ssm`, `hybrid`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Execução: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Funcionalidades distintivas: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Computação em tempo de teste, ajuste, recuperação: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark`. Multimodal: `vision`, `audio`. Em INNOVATIONS.pt.md indica-se quais destes `corbul` alcança; as API da biblioteca ainda podem mudar.
