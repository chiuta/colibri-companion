# Modelos locais leves (verificado em 2026-11-17)

«Leve» só significa que o descarregamento cabe confortavelmente em 4/8/16/32 GB de RAM. Não diz nada sobre a qualidade; nenhuma ordem aqui é um benchmark. Fonte de verdade: `crates/core/src/light_models.rs`. Todas as entradas funcionam com qualquer servidor compatível com OpenAI (Ollama, llama.cpp, LM Studio) definido em `server_url`; nenhuma está no catálogo do colibri.

Escalões de RAM: mínimo = tamanho + 2 GB, confortável = 1,5 × tamanho + 2 GB, arredondados para cima a 4/8/16/32.

| id | tamanho GB (baixo–alto impresso) | contexto | licença | RAM mín. / confortável |
|---|---|---|---|---|
| lfm3-1.7b | ~1.0 (4 bits) | 256K | LFM Open License v1.0 (**não padrão**) | 4 / 4 |
| qwen3.5-0.8b | 1.2-1.3 | 256K | apache-2.0 | 4 / 4 |
| lfm2.5-2.6b | 1.67 (Q4_K_M) | 131,072 | lfm1.0 (**não padrão**) | 4 / 8 |
| qwen3.5-4b | 3.3-4.0 | 256K | apache-2.0 | 8 / 8 |
| qwen3.5-9b | 6.6-7.6 | 256K | apache-2.0 | 16 / 16 |
| olmoe-1b-7b (colibri, antigo 2024) | 7 (int8 convertido) | n/e | Apache-2.0 | 8 (README) / 16 |
| gemma4-e2b | 4.6-7.5 | 128K | Apache 2.0 | 16 / 16 |
| gemma4-12b | 7.7-8.0 | 256K | Apache 2.0 | 16 / 16 |
| gemma4-e4b | 6.6-9.5 | 128K | Apache 2.0 | 16 / 16 |
| gpt-oss-20b | 14 (MXFP4) | n/e | Apache 2.0 | 16 / 32 |
| gemma4-26b-a4b | 16-19 | 256K | Apache 2.0 | 32 / 32 |

(n/e = não especificado.) As páginas do Ollama imprimem um intervalo de tamanho por etiqueta e nunca nomeiam a quantização por omissão, por isso os escalões usam o extremo alto. Só o LFM2.5 (Q4_K_M), o gpt-oss (MXFP4) e o OLMoE (README) têm um tamanho ligado a um formato nomeado. Os 7,5 GB do Gemma 4 E2B igualam a etiqueta separada `e2b-mlx`; a página não diz qual variante é qual. O Nemotron 3.5 Bolt E4B só corre em GPU com vLLM e nunca é oferecido por `for_ram`.

## Não verificado (não está na lista)
- Qwen3.5 2B: tamanho lido no Ollama, licença do HF ilegível (0.8B, 4B, 9B são Apache-2.0).
- Granite 4.2 3B/8B/30B: a documentação da IBM diz Apache 2.0, 128K, chamada a ferramentas; data de 25 ago 2026 só em páginas secundárias; sem tamanho em GB.
- MiMo-V2.6-Distill-Qwen-9B: a ficha original do HF não é legível; só um espelho.
- Mellum2 12B-A2.5B: Apache 2.0, contexto 131,072, mas sem tamanho GGUF nem ano; nenhuma página menciona «Mellum2.1».
- Nemotron 3.5 Bolt E4B: só GPU com vLLM, excluído.
- Qwen3.8: a Qwen não publica nenhum modelo abaixo de 27B; o Qwen3.8-4B da empero-ai é uma destilação de terceiros.
- Flint 40B-A2.5B e LFM3-VL-2B (linhas 252–253 do README do colibri), PLaMo 3 Translate, Phi-4-mini, Granite 4.1, Bonsai 2; as etiquetas Gemma 4 QAT não aparecem na página do Ollama.
- Uma pesquisa de lançamentos de set.–nov. 2026 só devolveu páginas de classificações de terceiros, nenhuma lida como fonte primária.
- Não são leves: Ripple-65B, Kimi K3.1 Pro, ERNIE-5.5-Flash-Base, Qwen3.5 122B, PLaMo 3 31B.

## Fontes
Ollama: páginas de biblioteca gemma4, qwen3.5, gpt-oss:20b. Ficha do modelo Google Gemma 4 (ai.google.dev) e huggingface.co/google/gemma-4-12B. huggingface.co/Qwen/Qwen3.5-0.8B. huggingface.co/LiquidAI/LFM3-1.7B (+ -GGUF). Não encontrados em nenhuma página: o mês de lançamento do Gemma 4, o comprimento de contexto do gpt-oss, comandos do llama.cpp (nenhum emitido).

## Agentes (chamada a ferramentas, programação agêntica, contexto longo)

O que o Companion precisa de um modelo (lido em `chat.rs`, `mcp.rs`): as ferramentas MCP são oferecidas como texto simples, um bloco `tool` delimitado que o modelo pode escrever e que `mcp::parse_tool_request` analisa, por isso **nenhum modelo precisa do campo `tools` da OpenAI**. O modelo tem de seguir esse formato, transmitir por SSE com `include_usage`, pode emitir `reasoning_content` ou texto `<think>` (separado pelo `chat.rs`) e precisa de contexto para até 5 rondas de resultados de ferramentas. As partes `image_url` só importam para visão. Um selo «tools» é, portanto, um bom sinal, não um requisito.

`for_agents(ram_gb)` só devolve modelos com chamada a ferramentas documentada numa página primária, com a mesma regra de ajuste que `for_ram`. Fontes documentadas:

| modelo | chamada a ferramentas (fonte) | programação agêntica | saída estruturada |
|---|---|---|---|
| gpt-oss-20b | selo `tools` do Ollama; «designed for agentic tasks» | por verificar | por verificar |
| gemma4 e2b / e4b / 12b / 26b-a4b | selo `tools` do Ollama, «native function-calling support» | por verificar | por verificar |
| qwen3.5 0.8b / 4b | selo `tools` do Ollama (família) | por verificar | por verificar |
| lfm3-1.7b | ficha HF: «designed for local agents, tool calling» (chamadas XML) | por verificar | ficha HF: «structured outputs» (por verificar como teste) |
| qwen3.5 9b | selo `tools` do Ollama | por verificar | por verificar |
| lfm2.5-2.6b, olmoe-1b-7b | por verificar | por verificar | por verificar |

Fora da lista de agentes (sem tamanho ou licença numa página primária, ou não locais):
- Devstral Small 2 24B: a documentação da Mistral diz Apache 2.0, 256k, chamada de funções, saídas estruturadas, 2025-12-09, mas o estado da API é Deprecated (retirada em 2026-03-31) e não se imprime tamanho local.
- Qwen3-Coder-30B-A3B: a ficha da Qwen diz chamada a ferramentas e programação agêntica, contexto 262,144, mas não imprime licença nem tamanho; o README do colibri lista-o na linha 209.
- Granite 4.2: a documentação da IBM documenta a chamada a ferramentas e o contexto de 128K, sem tamanho.
- MiMo distill, Mellum2 (programação agêntica afirmada só em páginas não lidas), Nemotron 3.5 Bolt (só vLLM).

Limites: os modelos pequenos chamam ferramentas de forma pouco fiável, e um selo diz que o modelo de chat suporta ferramentas, não que o modelo escolha bem. Nada disto foi executado com as automatizações do Companion.

## Reverificação de 2026-11-17 (catálogo completo)
Relido hoje: Ollama gemma4 (todos os tamanhos/contextos coincidem), qwen3.5 (tamanhos, selos; a página não nomeia licença), gpt-oss:20b (14 GB, ferramentas, Apache 2.0, sem contexto); HF gemma-4-12B, Qwen3.5-0.8B/4B, LFM3-1.7B, LFM2.5-2.6B; documentação da IBM Granite 4.2.
Corrigido: contexto do LFM2.5 (131,072, antes «não indicado»); a chamada a ferramentas do LFM3 está agora documentada; Qwen3.5 9B acrescentado.
Não relido: OLMoE (números do README), licença do Gemma 4 E2B/E4B/26B (ficha da família lida antes; ficha 12B lida hoje), tamanho GGUF do LFM3, tamanho Q4_K_M do LFM2.5 (a página do HF só tem ligações), meses de lançamento (nenhuma página primária os imprime; o 27 de out. do LFM3 vem de um resumo).
As páginas do Ollama gemma4, qwen3.5 e gpt-oss não imprimem a licença; as licenças vêm das fichas do HF/Google.
