# O motor «clean-room» (`companion-engine` e `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Outros documentos: [INNOVATIONS.pt.md](INNOVATIONS.pt.md) · [QUICKSTART.pt.md](QUICKSTART.pt.md) · [CLEANROOM.pt.md](CLEANROOM.pt.md)

**Estado: alfa. Tudo o que se segue foi testado apenas em pequenos pontos de controlo sintéticos.** Os autores não correram nenhum modelo real do Hugging Face neste motor (o ambiente de compilação não conseguia chegar a huggingface.co), nenhuma GPU executou qualquer parte dele e os caminhos ARM, macOS e Windows nunca foram compilados aqui. Lê a tabela «verificado e não verificado» antes de confiares em qualquer afirmação.

## O que é

Um motor de inferência local para modelos de linguagem, escrito de raiz em Rust, a partir de descrições públicas de formatos de ficheiro e de artigos publicados. Tem três partes:

| Crate | O que é |
|---|---|
| `crates/engine` (`companion-engine`) | A biblioteca: formatos de ficheiro, tokenizador, quantização, passagens em frente, amostragem e as ferramentas extra descritas em INNOVATIONS.pt.md. Dependências: apenas `serde_json` e `memmap2`. |
| `crates/engine-cli` (`corbul`) | Uma linha de comandos e um servidor local compatível com OpenAI / Ollama / Anthropic, por cima da biblioteca. |
| `crates/engine-gpu` (`companion-engine-gpu`) | Núcleos WGSL para o crate `wgpu`, atrás de uma opção desativada por omissão. Não é usado pelo `corbul`. |

É independente da aplicação de ambiente de trabalho Colibri Companion e não a substitui: a aplicação continua a comandar o colibri de JustVugg. Ligar este motor à aplicação como backend alternativo está planeado, não feito.

## O que suporta

**Ficheiros.** GGUF v2/v3 (o cabeçalho e a tabela de tensores leem-se sem tocar nos pesos, por isso um ficheiro de 400 GB inspeciona-se em milissegundos); um escritor GGUF em fluxo para modelos maiores do que a RAM; safetensors, num ficheiro ou fragmentado com índice; `config.json` e `tokenizer.json` do Hugging Face; um descarregador de modelos (através do `curl`, com retoma e verificação SHA-256).

**Quantização.** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 e os K-quants Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K podem ser descodificados, e há codificadores para todos (com ponderação por matriz de importância), conversão Hugging Face → GGUF e requantização de GGUF. Os pesos ficam quantizados em memória; o produto matriz–vetor nunca constrói a matriz. Os i-quants (IQ*) não estão implementados.

**Arquiteturas.**

| Família | Estado |
|---|---|
| Descodificadores densos: Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 e semelhantes (norma RMS ou de camada, GQA/MQA, RoPE com escala linear / Llama-3 / YaRN e rotação parcial, QK-norm, vieses, soft-capping, janelas deslizantes, MLP com porta ou simples) | Implementadas; carregadas pelo `corbul` |
| Mistura de especialistas: Mixtral, Qwen-MoE, especialistas empilhados GGUF, especialistas partilhados, especialistas encaminhados lidos do disco com cache LRU | Implementadas; carregadas pelo `corbul` |
| Atenção latente multicabeça (estilo DeepSeek-V2/V3, Kimi-K2) | Apenas módulo de biblioteca; o registo de arquiteturas ainda a marca «planeada» e o `corbul` não a carrega |
| Mamba-1, Mamba-2, RWKV-6, RWKV-7 | Apenas módulo de biblioteca; idem |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, modelos encoder e encoder-decoder (T5, BERT, Whisper…), LongRoPE | Não suportados; o `corbul` diz isso em vez de adivinhar |
| Entradas de imagem e áudio | Não suportadas (apenas texto) |

**Caminhos de hardware.**

- CPU: núcleos escolhidos em execução (escalar, AVX2, AVX-512, VNNI em x86-64; NEON em aarch64), um pool de threads persistente adequado a núcleos de desempenho e eficiência misturados, deteção de hardware e NUMA e um planeador que decide o que fica na RAM.
- GPU: núcleos WGSL verificados com `naga` e face a uma emulação em Rust. Nunca correram numa GPU.
- Modelos maiores do que a RAM: especialistas a partir do disco (MoE), transmissão de camadas (denso), uma malha LAN sobre várias máquinas.

## Verificado e não verificado

| Área | Estado |
|---|---|
| Testes unitários, de integração e fuzz em pontos de controlo sintéticos (Linux, x86-64) | Verificado. Centenas de testes passam; ficheiros hostis ou truncados devolvem erros, não falhas. |
| Pontos de controlo reais do Hugging Face (qualquer família) | **Não executado.** O ambiente não chegava a huggingface.co. |
| Tokenizador com ficheiros `tokenizer.json` reais | Escrito segundo as descrições públicas do formato; existe uma bancada de testes (`ENGINE_VOCAB_DIR`), mas não a corremos com ficheiros reais. |
| Permutação das linhas Q/K ao estilo llama.cpp em ficheiros GGUF | Implementada segundo a descrição; não verificada com ficheiros reais. |
| Quantizadores e desquantizadores | Verificados face a blocos calculados à mão e ida e volta; não face a ficheiros escritos por outras ferramentas. |
| Execução em GPU | **Não executada.** Os núcleos só estão validados com `naga` e uma emulação em Rust. |
| Caminhos ARM / NEON, macOS, Windows | **Nunca compilados nem executados aqui.** As sondas são testadas com texto gravado desses sistemas. |
| Fórmulas de Mamba / RWKV / MLA | Escritas a partir de artigos e da memória das descrições de referência, testadas apenas quanto à coerência interna. |
| Malha numa rede real | Testada apenas em loopback. |
| Velocidade | **Sem números fiáveis.** A máquina de compilação tinha dois núcleos partilhados; as medições foram ruidosas. Todas as velocidades do piloto automático são estimativas. |
| Modo reprodutível | Idêntico em cinco variantes de núcleos x86 e vários números de threads; ARM não testado. |
| Licenças das dependências | Verificadas com `cargo-deny` num commit anterior (b496ba5); as adições posteriores ainda não foram reverificadas. |

## Declaração clean-room

Nenhum código do colibri, do llama.cpp ou de qualquer outro motor de inferência foi lido ou copiado ao escrever este motor. Os formatos e algoritmos vêm de especificações públicas, artigos e estruturas de ficheiro. Vê [CLEANROOM.pt.md](CLEANROOM.pt.md) para o processo e os seus limites.

## Comparação com outros motores locais

Não fizemos nenhuma comparação direta, por isso não afirmamos nada sobre velocidade ou qualidade. Em termos qualitativos:

- Motores consolidados como llama.cpp, Ollama, mistral.rs ou vLLM foram executados por muita gente em milhares de pontos de controlo reais, têm backends de GPU a funcionar, cobrem mais arquiteturas e formatos de quantização (incluindo i-quants) e tratam modelos de visão. Este motor ainda não tem nada dessa evidência. Se precisas de algo que funcione hoje num modelo grande concreto, usa um deles.
- O que este motor tenta fazer de forma diferente: ferramentas de integridade e honestidade incorporadas (citações bloqueadas às fontes, modo reprodutível com recibos selados por cadeia de hashes e HMAC opcional, análise de incerteza, um piloto automático que diz «isto não vai correr» em vez de falhar), ferramentas de língua com o romeno em primeiro lugar, correr modelos maiores do que a tua RAM ou do que uma só máquina, uma base de código Rust pequena com duas dependências e apenas licenças permissivas.
- Qualquer afirmação de que este motor é mais rápido ou mais preciso do que outro seria não testada. Mede no teu hardware e diz-nos.

## Mapa da biblioteca

Formatos e carregamento: `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`. Modelo: `arch`, `model`, `moe`, `mla`, `ssm`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Execução: `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Funções distintivas: `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Última vaga: `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark` — implementadas na biblioteca e cobertas por testes em modelos sintéticos, ainda em acabamento, ainda não acessíveis a partir do `corbul`, e as suas APIs podem mudar.
