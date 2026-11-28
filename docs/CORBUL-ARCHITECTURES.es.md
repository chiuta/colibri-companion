# Modelos que corbul ejecuta y colibri no

Companion arranca **corbul**, su propio motor, para todo GGUF que colibri no puede importar. Desde alpha.18 va **directamente** allí para las familias de abajo, sin un import fallido antes.

## Qué ejecuta corbul
- **Gemma 4** (E2B, E4B, 26B-A4B con expertos, 31B), desde un `.gguf`: atención mixta deslizante y global, tamaños de cabeza por capa, claves usadas como valores en las capas globales, RoPE proporcional, K/V compartido en las últimas capas, embeddings por capa, el MLP denso junto a los expertos enrutados, límite de los logits finales.
- **GPT-2, GPT-J, GPT-NeoX/Pythia, Falcon, BLOOM, MPT**, desde un `.gguf`: LayerNorm, QKV fusionado o separado, posiciones aprendidas / rotatorias / ALiBi, residual paralelo. Probados solo con modelos sintéticos pequeños. OPT no tiene conversión GGUF.
- **Gemma 1–3, Phi, OLMo, Granite, Cohere, InternLM2, StableLM, StarCoder2, EXAONE, Baichuan, MiniCPM, SmolLM3**, Llama y Mistral, Qwen y las demás familias con expertos que el motor ya tenía.
- **Mamba, Mamba-2, RWKV-6/7, Jamba, Bamba** (recurrentes e híbridos).

`corbul serve MODELO` responde a `/health`, `/v1/models`, `/v1/chat/completions` y `/v1/completions` (también en flujo) con la plantilla de chat del modelo, `stop`, `max_tokens`, `temperature`, `top_p`, `top_k` y `seed`. Una conversación que continúa la anterior no vuelve a leer su historial.

## Qué no hace con estos modelos
Llamadas a herramientas, salida estructurada, varias respuestas por petición, embeddings, paquetes de contexto (Companion usa entonces el chat normal), imágenes de entrada, varios modelos a la vez. Gemma 4 se lee solo de `.gguf`, no de carpetas safetensors. Un GGUF sin vocabulario propio puede usar un `tokenizer.json` colocado a su lado.

## Qué no está comprobado
Todo está probado con **modelos sintéticos pequeños** (invariantes propias: caché igual a recalcular, ventanas deslizantes, K/V compartido, enrutado, expertos cuantizados, las respuestas HTTP). **Nada se ha comparado con un archivo Gemma 4 real.** Si las respuestas de un archivo real parecen erróneas, ejecuta tú mismo `corbul serve ARCHIVO.gguf` y envía la salida; lo primero que se revisa es el patrón deslizante/global y las tablas RoPE.

## Aún no soportado
OPT, gpt-oss, GraniteMoE, PhiMoE, DBRX, Llama 4, GLM-4-MoE, Hunyuan, Ernie-MoE, Grok, el modelo DeepSeek completo, MiniMax, Zamba, modelos codificador-decodificador. corbul los nombra y no adivina.
