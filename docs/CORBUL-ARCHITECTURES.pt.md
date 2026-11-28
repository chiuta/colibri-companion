# Modelos que o corbul executa e o colibri não

O Companion inicia o **corbul**, o motor próprio, para todo GGUF que o colibri não consegue importar. Desde a alpha.18 vai **diretamente** para lá nas famílias abaixo, sem uma importação falhada antes.

## O que o corbul executa
- **Gemma 4** (E2B, E4B, 26B-A4B com especialistas, 31B), a partir de um `.gguf`: atenção mista deslizante e global, tamanhos de cabeça por camada, chaves usadas como valores nas camadas globais, RoPE proporcional, K/V partilhado nas últimas camadas, embeddings por camada, o MLP denso ao lado dos especialistas encaminhados, limite dos logits finais.
- **GPT-2, GPT-J, GPT-NeoX/Pythia, Falcon, BLOOM, MPT**, a partir de um `.gguf`: LayerNorm, QKV fundido ou separado, posições aprendidas / rotativas / ALiBi, residual paralelo. Testados apenas em pequenos modelos sintéticos. O OPT não tem conversão GGUF.
- **gpt-oss (20B, 120B)**, a partir de um `.gguf` (incluindo especialistas MXFP4): sinks de atenção, janela deslizante, YaRN, ativação de especialista limitada. Testado apenas em pequenos modelos sintéticos. **GraniteMoE, GLM-4.5/4.6 MoE**: também a partir de `.gguf`.
- **Gemma 1–3, Phi, OLMo, Granite, Cohere, InternLM2, StableLM, StarCoder2, EXAONE, Baichuan, MiniCPM, SmolLM3**, Llama e Mistral, Qwen e as outras famílias com especialistas que o motor já tinha.
- **Mamba, Mamba-2, RWKV-6/7, Jamba, Bamba** (recorrentes e híbridos).

`corbul serve MODELO` responde a `/health`, `/v1/models`, `/v1/chat/completions` e `/v1/completions` (também em fluxo) com o modelo de chat do modelo, `stop`, `max_tokens`, `temperature`, `top_p`, `top_k` e `seed`. Uma conversa que continua a anterior não volta a ler o seu histórico.

## O que não faz com estes modelos
Chamadas de ferramentas, saída estruturada, várias respostas por pedido, embeddings, pacotes de contexto (o Companion usa então o chat normal), imagens de entrada, vários modelos ao mesmo tempo. O Gemma 4 lê-se apenas de `.gguf`, não de pastas safetensors. Um GGUF sem vocabulário próprio pode usar um `tokenizer.json` colocado ao lado.

## O que não está verificado
Tudo está testado em **pequenos modelos sintéticos** (invariantes próprias: cache igual ao recálculo, janelas deslizantes, K/V partilhado, encaminhamento, especialistas quantizados, as respostas HTTP). **Nada foi comparado com um ficheiro Gemma 4 real.** Se as respostas de um ficheiro real parecerem erradas, executa tu mesmo `corbul serve FICHEIRO.gguf` e envia a saída; verifica-se primeiro o padrão deslizante/global e as tabelas RoPE.

## Ainda não suportados
OPT, PhiMoE, DBRX, Llama 4, Hunyuan, Ernie-MoE, Grok, o modelo DeepSeek completo, MiniMax, Zamba, modelos codificador-decodificador. O corbul nomeia-os e não adivinha.
