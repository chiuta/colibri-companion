# Inicio rápido: `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` es la línea de comandos del motor clean-room ([ENGINE.es.md](ENGINE.es.md)). Esta página lista solo los comandos y opciones que existen en `crates/engine-cli` en el momento de escribirla; la línea de comandos aún se está ampliando, así que la última palabra la tiene `corbul help`. Algunas opciones de abajo las lee el código pero `corbul help` aún no las muestra; van marcadas *(aún no en la ayuda)*. Lo que todavía no existe va marcado como **planeado**.

**Aviso honesto:** `corbul` solo se ha probado con modelos sintéticos diminutos. Espera aristas en puntos de control reales e infórmanos.

## Compilar

```bash
cargo build --release -p colibri-rs-cli     # produce target/release/corbul
```

Un modelo es una carpeta con `config.json` + `*.safetensors` + `tokenizer.json`, o un único archivo `.gguf`.

## Mirar, descargar, ejecutar

```bash
corbul inspect  ./my-model                  # o file.gguf o owner/repo[@rev]: arquitectura, tamaño, soporte
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors; HF_TOKEN para repositorios de acceso restringido, HF_ENDPOINT para un espejo
corbul generate ./my-model --prompt "Hola" --max-tokens 64
corbul chat     ./my-model --system "Sé conciso."
corbul serve    ./my-model --port 8080      # compatible con OpenAI / Ollama / Anthropic, enlazado por defecto a 127.0.0.1
```

Opciones comunes de `generate`, `chat` y `serve`: `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--kv f32|f16|q8`, `--no-mmap`, `--threads N`, `--ctx N`, `--model-name`, `--quiet`. `generate` también acepta `--prompt-file F` y `--raw` (sin plantilla de chat). `corbul run` es un alias de `generate`.

Servir varios modelos: `--models-dir DIR`, `--alias NOMBRE=MODELO`, `--max-loaded N`, `--sleep-idle-seconds S`, `--token T` (o `CORBUL_TOKEN`), `--keys-file ARCHIVO`, `--cors ORÍGENES`, `--log-prompts`, `--no-log`, `--drain-seconds S`, `--config-dir DIR`.

## ¿Funcionará en esta máquina?

```bash
corbul doctor                # lo que ve el motor: conjuntos de instrucciones de CPU, núcleos, RAM, disco, sensores, contador de energía
corbul plan ./my-model --context 8192            # veredicto: FITS / TIGHT / STREAMS / TOO SLOW / WON'T RUN, velocidad, consejos
corbul plan ./my-model --ram-budget 8g --calibrate   # finge que tienes 8 GB; antes cronometra la máquina un segundo
corbul capabilities [--json]                     # lo que esta versión puede y no puede hacer
```

Las velocidades son estimaciones, no mediciones de tu modelo.

## Mayor que la RAM

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"                        # MoE: los expertos enrutados se leen desde disco
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --prompt "…"   # denso: las capas se transmiten (limitado por el disco; --stream-window aún no en la ayuda)
```

## Varias máquinas (malla LAN)

```bash
# en cada máquina auxiliar (la clave tiene 8 caracteres como mínimo; o define CORBUL_MESH_KEY)
corbul worker ./my-model --key SECRET --bind 0.0.0.0:7471          # --no-discovery omite el descubrimiento UDP
# en la máquina principal
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key SECRET --prompt "…"
corbul generate ./my-model --discover --key SECRET --prompt "…"      # encuentra workers en la LAN
```

El tráfico se autentica pero **no se cifra**: usa una red de confianza o una VPN. Probado solo en loopback.

## Salida estructurada, citas, adaptadores

```bash
corbul generate M --json --prompt "…"                 # salida JSON válida
corbul generate M --json-schema schema.json --prompt "…"      # archivo o JSON en línea; también --regex PATRÓN, --grammar ARCHIVO (GBNF); usa solo una de las cuatro; aún no en la ayuda
corbul generate M --sources notes.txt --sources paper.txt --prompt "Responde con citas" --citations report.json   # citas bloqueadas
corbul generate M --lora ./adapter:0.8                # adaptador PEFT o GGUF con un factor
```

El servidor acepta `response_format`, `grammar` y `sources`, y ofrece `POST /v1/quotes/verify`.

## Conversaciones largas, sesiones, reproducibilidad

```bash
corbul generate M --session ./sessions --prompt "…"             # reutiliza el prefijo de prompt guardado más largo (no con --kv-policy)
corbul generate M --kv-policy h2o --kv-budget 2048 --prompt "…" # streaming | h2o | snapkv; con pérdida por diseño
corbul generate M --governor --prompt "…"                       # se adapta a temperatura, memoria y batería (Linux)
corbul generate M --reproducible --temperature 0 --prompt "hi" --receipt r.json --receipt-key SECRET
corbul verify-receipt r.json --key SECRET --model M            # comprueba el sello (y los pesos si das --model)
corbul replay r.json --model M --key SECRET                    # repite la generación y compara
```

Los recibos funcionan con el bucle de generación simple, no con modelos MoE. Opcionales: `--receipt-topk N`, `--keep-steps`, `--leaves` *(aún no en la ayuda)*.

## Planeado (aún no en `corbul`)

- Cargar modelos Mamba / RWKV / MLA (estilo DeepSeek), aunque la biblioteca tiene los bloques.
- Las herramientas de idioma rumano, el análisis de incertidumbre, búsqueda y lookahead, decodificación autoespeculativa, vectores de dirección, recuperación, fusión y marca de agua (consulta [INNOVATIONS.es.md](INNOVATIONS.es.md)).
- Endpoints de embedding y rerank (el servidor responde «not implemented»), entradas de imagen y audio.
- El backend de GPU y una interfaz web.
