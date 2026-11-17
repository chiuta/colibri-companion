# Modelos locales ligeros (comprobado el 2026-11-17)

«Ligero» solo significa que la descarga cabe cómodamente en 4/8/16/32 GB de RAM. No dice nada de la calidad; ningún orden de aquí es un benchmark. Fuente de verdad: `crates/core/src/light_models.rs`. Todas las entradas funcionan con cualquier servidor compatible con OpenAI (Ollama, llama.cpp, LM Studio) configurado en `server_url`; ninguna está en el catálogo de colibri.

Tramos de RAM: mínimo = tamaño + 2 GB, cómodo = 1,5 × tamaño + 2 GB, redondeados hacia arriba a 4/8/16/32.

| id | tamaño GB (bajo–alto impreso) | contexto | licencia | RAM mín. / cómoda |
|---|---|---|---|---|
| lfm3-1.7b | ~1.0 (4 bits) | 256K | LFM Open License v1.0 (**no estándar**) | 4 / 4 |
| qwen3.5-0.8b | 1.2-1.3 | 256K | apache-2.0 | 4 / 4 |
| lfm2.5-2.6b | 1.67 (Q4_K_M) | 131,072 | lfm1.0 (**no estándar**) | 4 / 8 |
| qwen3.5-4b | 3.3-4.0 | 256K | apache-2.0 | 8 / 8 |
| qwen3.5-9b | 6.6-7.6 | 256K | apache-2.0 | 16 / 16 |
| olmoe-1b-7b (colibri, antiguo 2024) | 7 (int8 convertido) | n/e | Apache-2.0 | 8 (README) / 16 |
| gemma4-e2b | 4.6-7.5 | 128K | Apache 2.0 | 16 / 16 |
| gemma4-12b | 7.7-8.0 | 256K | Apache 2.0 | 16 / 16 |
| gemma4-e4b | 6.6-9.5 | 128K | Apache 2.0 | 16 / 16 |
| gpt-oss-20b | 14 (MXFP4) | n/e | Apache 2.0 | 16 / 32 |
| gemma4-26b-a4b | 16-19 | 256K | Apache 2.0 | 32 / 32 |

(n/e = no especificado.) Las páginas de Ollama imprimen un rango de tamaño por etiqueta y nunca nombran la cuantización por defecto, así que los tramos usan el extremo alto. Solo LFM2.5 (Q4_K_M), gpt-oss (MXFP4) y OLMoE (README) tienen un tamaño ligado a un formato nombrado. Los 7,5 GB de Gemma 4 E2B igualan la etiqueta aparte `e2b-mlx`; la página no dice qué variante es cuál. Nemotron 3.5 Bolt E4B es solo GPU con vLLM y `for_ram` nunca lo ofrece.

## No comprobado (no está en la lista)
- Qwen3.5 2B: tamaño leído en Ollama, licencia de HF ilegible (0.8B, 4B, 9B son Apache-2.0).
- Granite 4.2 3B/8B/30B: la documentación de IBM dice Apache 2.0, 128K, llamada a herramientas; fecha del 25 ago 2026 solo en páginas secundarias; sin tamaño en GB.
- MiMo-V2.6-Distill-Qwen-9B: la ficha original de HF no se puede leer; solo un espejo.
- Mellum2 12B-A2.5B: Apache 2.0, contexto 131,072, pero sin tamaño GGUF ni año; ninguna página menciona «Mellum2.1».
- Nemotron 3.5 Bolt E4B: solo GPU con vLLM, excluido.
- Qwen3.8: Qwen no publica ningún modelo por debajo de 27B; el Qwen3.8-4B de empero-ai es una destilación de terceros.
- Flint 40B-A2.5B y LFM3-VL-2B (líneas 252–253 del README de colibri), PLaMo 3 Translate, Phi-4-mini, Granite 4.1, Bonsai 2; las etiquetas Gemma 4 QAT no aparecen en la página de Ollama.
- Una búsqueda de lanzamientos de sept.–nov. 2026 solo devolvió páginas de clasificaciones de terceros, ninguna leída como fuente primaria.
- No ligeros: Ripple-65B, Kimi K3.1 Pro, ERNIE-5.5-Flash-Base, Qwen3.5 122B, PLaMo 3 31B.

## Fuentes
Ollama: páginas de biblioteca gemma4, qwen3.5, gpt-oss:20b. Ficha del modelo Google Gemma 4 (ai.google.dev) y huggingface.co/google/gemma-4-12B. huggingface.co/Qwen/Qwen3.5-0.8B. huggingface.co/LiquidAI/LFM3-1.7B (+ -GGUF). No encontrados en ninguna página: el mes de lanzamiento de Gemma 4, la longitud de contexto de gpt-oss, comandos de llama.cpp (ninguno emitido).

## Agentes (llamada a herramientas, programación agéntica, contexto largo)

Lo que Companion necesita de un modelo (leído en `chat.rs`, `mcp.rs`): las herramientas MCP se ofrecen como texto plano, un bloque `tool` delimitado que el modelo puede escribir y que `mcp::parse_tool_request` analiza, así que **ningún modelo necesita el campo `tools` de OpenAI**. El modelo debe seguir ese formato, transmitir por SSE con `include_usage`, puede emitir `reasoning_content` o texto `<think>` (separado por `chat.rs`) y necesita contexto para hasta 5 rondas de resultados de herramientas. Las partes `image_url` solo importan para visión. Una insignia «tools» es, pues, una buena señal, no un requisito.

`for_agents(ram_gb)` solo devuelve modelos con llamada a herramientas documentada en una página primaria, con la misma regla de ajuste que `for_ram`. Fuentes documentadas:

| modelo | llamada a herramientas (fuente) | programación agéntica | salida estructurada |
|---|---|---|---|
| gpt-oss-20b | insignia `tools` de Ollama; «designed for agentic tasks» | sin verificar | sin verificar |
| gemma4 e2b / e4b / 12b / 26b-a4b | insignia `tools` de Ollama, «native function-calling support» | sin verificar | sin verificar |
| qwen3.5 0.8b / 4b | insignia `tools` de Ollama (familia) | sin verificar | sin verificar |
| lfm3-1.7b | ficha HF: «designed for local agents, tool calling» (llamadas XML) | sin verificar | ficha HF: «structured outputs» (sin verificar como prueba) |
| qwen3.5 9b | insignia `tools` de Ollama | sin verificar | sin verificar |
| lfm2.5-2.6b, olmoe-1b-7b | sin verificar | sin verificar | sin verificar |

No están en la lista de agentes (sin tamaño o licencia en una página primaria, o no son locales):
- Devstral Small 2 24B: la documentación de Mistral dice Apache 2.0, 256k, llamada a funciones, salidas estructuradas, 2025-12-09, pero el estado de la API es Deprecated (retirada el 2026-03-31) y no se imprime tamaño local.
- Qwen3-Coder-30B-A3B: la ficha de Qwen dice llamada a herramientas y programación agéntica, contexto 262,144, pero no imprime licencia ni tamaño; el README de colibri lo lista en la línea 209.
- Granite 4.2: la documentación de IBM documenta la llamada a herramientas y el contexto de 128K, sin tamaño.
- MiMo distill, Mellum2 (programación agéntica afirmada solo en páginas no leídas), Nemotron 3.5 Bolt (solo vLLM).

Límites: los modelos pequeños llaman a las herramientas de forma poco fiable, y una insignia dice que la plantilla de chat admite herramientas, no que el modelo elija bien. Nada de esto se ha ejecutado con las automatizaciones de Companion.

## Revisión del 2026-11-17 (catálogo completo)
Releído hoy: Ollama gemma4 (todos los tamaños/contextos coinciden), qwen3.5 (tamaños, insignias; la página no nombra licencia), gpt-oss:20b (14 GB, herramientas, Apache 2.0, sin contexto); HF gemma-4-12B, Qwen3.5-0.8B/4B, LFM3-1.7B, LFM2.5-2.6B; documentación de IBM Granite 4.2.
Corregido: contexto de LFM2.5 (131,072, antes «no indicado»); la llamada a herramientas de LFM3 ya está documentada; añadido Qwen3.5 9B.
No releído: OLMoE (cifras del README), licencia de Gemma 4 E2B/E4B/26B (ficha de familia leída antes; ficha 12B leída hoy), tamaño GGUF de LFM3, tamaño Q4_K_M de LFM2.5 (la página de HF solo tiene enlaces), meses de lanzamiento (ninguna página primaria los imprime; el 27 oct. de LFM3 sale de un resumen).
Las páginas de Ollama gemma4, qwen3.5 y gpt-oss no imprimen la licencia; las licencias salen de las fichas de HF/Google.
