# Démarrage rapide : `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` est la ligne de commande du moteur clean-room ([ENGINE.fr.md](ENGINE.fr.md)). Chaque commande et chaque option de cette page existe dans `crates/engine-cli` et est affichée par `corbul help` ; un script (`tools/check_engine_docs.py`) échoue si le code a une option, une sous-commande ou une variable d'environnement que cette page ne mentionne pas. `corbul help` reste le dernier mot.

**Avertissement honnête :** `corbul` n'a été testé que sur de minuscules modèles synthétiques et des fichiers de référence de tokenizer. Aucun vrai modèle téléchargé n'a été exécuté par les auteurs. Attends-toi à des aspérités et signale-les, s'il te plaît. `corbul capabilities` affiche ce qui fonctionne et ce qui ne fonctionne pas.

## Construire

```bash
cargo build --release -p colibri-rs-cli     # produces target/release/corbul
```

Un modèle est un dossier contenant `config.json` + `*.safetensors` + `tokenizer.json`, ou un seul fichier `.gguf`.

## Regarder, télécharger, lancer

```bash
corbul inspect  ./my-model                  # or file.gguf or owner/repo[@rev]: architecture, size, support
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors
corbul generate ./my-model --prompt "Hello" --max-tokens 64
corbul generate ./my-model --prompt-file question.txt --raw   # --raw: no chat template
corbul chat     ./my-model --system "You are concise."
corbul run      ./my-model --prompt "Hello"                   # `run` is `generate`
corbul serve    ./my-model --port 8080                        # bound to 127.0.0.1 by default
```

Définis `HF_TOKEN` (ou `HUGGING_FACE_HUB_TOKEN`) pour les dépôts à accès restreint et `HF_ENDPOINT` pour un miroir. Les téléchargements demandent `curl` (`wget` anonyme sert de repli).

Options communes à `generate`, `chat` et `serve` : `--max-tokens N`, `--temperature T`, `--top-p P`, `--top-k K`, `--seed S`, `--kv f32|f16|q8`, `--no-mmap`, `--model-name NAME`, `--threads N` (aussi `COLIBRI_THREADS`), `--ctx N`, `--quiet`, `--ram-budget SIZE`, `--lora PATH[:S]` (répétable ; dossier PEFT ou adaptateur GGUF avec un facteur d'échelle).

Options d'échantillonnage supplémentaires pour `generate` et `chat` :

```bash
corbul generate M --prompt "…" --min-p 0.05 --typical-p 0.9 --repeat-penalty 1.1 \
  --presence-penalty 0.2 --frequency-penalty 0.2 --mirostat 2 --mirostat-tau 5 --mirostat-eta 0.1 \
  --dry-multiplier 0.8 --xtc-probability 0.1
corbul generate M --prompt "…" --sampler-json '{"top_k": 20}'   # any sampler field, applied before the flags above
```

Les modèles à espaces d'états et hybrides (Mamba, RWKV, Jamba, Bamba) passent par `generate` et `chat` comme simple continuation de texte : pas de modèle de chat, pas de sortie structurée, pas de `serve`.

## Est-ce que ça tournera sur cette machine ?

```bash
corbul doctor                # CPU instruction sets, chosen kernel, RAM, disk, sensors, GPU probe
corbul plan ./my-model --context 8192          # verdict (fits / tight / streams / too slow / won't run), speed, advice
corbul plan ./my-model --ram-budget 8g --calibrate   # pretend you have 8 GB; time this machine for about a second first
corbul capabilities [--json]                   # what this build can and cannot do (alias: corbul features)
```

Les vitesses sont des estimations, pas des mesures de ton modèle. `ENGINE_SIMD` impose un noyau, `ENGINE_REPRODUCIBLE=1` active le mode reproductible, `ENGINE_BACKEND=cpu|gpu|auto` est lu par la bibliothèque (`corbul` calcule toujours sur le CPU).

## Plus grand que la RAM

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"     # MoE: routed experts stream from disk
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --disk-mbps 800 --prompt "…"   # dense: layers stream; speed is bounded by the disk; --disk-mbps (default 500) only feeds the speed estimate it prints
```

## Plusieurs machines (mesh en réseau local)

```bash
# on every helper machine (the key needs 8+ characters; or set CORBUL_MESH_KEY)
corbul worker ./my-model --key MESHKEY12345 --bind 0.0.0.0:7471
corbul worker ./my-model --key-file mesh.key --discovery            # also answer UDP discovery probes (opt-in); --discovery-port N; --no-discovery forces it off
# on the main machine
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key MESHKEY12345 --prompt "…"
corbul generate ./my-model --discover --key MESHKEY12345 --prompt "…"      # broadcast a probe, use workers that hold the same model
corbul generate ./my-model --workers h:7471 --key MESHKEY12345 --wire-f16 --local-speed 2.5 --prompt "…"
```

`--wire-f16` divise par deux le trafic d'états cachés (et renonce à l'identité bit à bit) ; `--local-speed` indique au répartiteur la vitesse de cette machine par rapport aux workers ; `--discover-target HOST:PORT` et `--discover-wait-ms N` changent l'adresse de la sonde (par défaut `255.255.255.255:47813`) et l'attente (par défaut 2000 ms). Le trafic est authentifié mais **non chiffré** : utilise un réseau de confiance ou un VPN. Testé uniquement en boucle locale.

## Sortie structurée, citations, adaptateurs

```bash
corbul generate M --json --prompt "…"                       # valid JSON
corbul generate M --json-schema schema.json --prompt "…"    # a file or inline JSON
corbul generate M --regex '[A-Z]{3}-[0-9]{4}' --prompt "…"  # regular expression
corbul generate M --grammar rules.gbnf --prompt "…"         # GBNF grammar; use only one of the four
corbul generate M --sources notes.txt --sources paper.txt --prompt "Answer with quotes" --citations report.json   # quote-locked
corbul generate M --sources notes.txt --no-lock --prompt "…"   # only check the quotes afterwards
corbul generate M --lora ./adapter:0.8
```

Le serveur accepte `response_format`, `grammar` et `sources` dans les requêtes et propose `POST /v1/quotes/verify`.

## Longues conversations, sessions, reproductibilité

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

Les reçus fonctionnent avec la boucle de génération simple, pas avec les modèles MoE ni à espaces d'états. Optionnel : `--receipt-key K` (visible dans la liste des processus ; préfère `--receipt-key-file`), `--receipt-topk N`, `--keep-steps`, `--leaves`.

## Décodage plus rapide

```bash
corbul generate M --draft ./small-model --draft-len 4 --prompt "…"      # a draft model proposes, M verifies
corbul generate M --lookahead --ngram-min 1 --ngram-max 4 --prompt "…"   # prompt-lookup, no extra model
corbul generate M --self-spec --skip-layers auto --prompt "…"            # M drafts with a subset of its layers; or --skip-layers 4,6,8-11
```

Même réponse gloutonne et même distribution d'échantillonnage ; le texte apparaît à la fin. Seuls `--temperature`, `--top-k`, `--top-p` s'appliquent ; pas avec la sortie structurée, les reçus, les sessions ni le mesh.

## Recherche, quantification, fusion, pilotage, récupération, filigrane, langue

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
```

Toutes les combinaisons d'options ne fonctionnent pas ; chaque commande affiche une erreur d'usage d'une ligne quand ce n'est pas le cas. Rien de tout cela n'a été exécuté sur un vrai point de contrôle.

## Service

```bash
corbul serve --models-dir ./models --host 127.0.0.1 --port 8080 --token-file token.txt --default-model qwen
corbul serve --models-dir ./models --alias fast=qwen-0.5b --max-loaded 2 --sleep-idle-seconds 600
corbul serve --models-dir ./models --keys-file keys.json --allow-admin --max-kv 8g --max-heavy 4
corbul serve --models-dir ./models --cors "https://app.example" --log-prompts --drain-seconds 10 --config-dir ./cfg
```

Ouvre `http://127.0.0.1:8080/` pour l'interface web (aussi `/ui`). La page est publique ; chaque appel qu'elle fait exige le jeton. `--token T` équivaut à `CORBUL_TOKEN` (visible dans la liste des processus, donc préfère `--token-file`). `--keys-file` contient des clés d'API avec des listes de modèles et des limites de débit par clé. `--allow-admin` permet aux détenteurs de clés d'utiliser `/api/create`, `/api/delete`, `/models/load` et `/models/unload` ; par défaut, seul le jeton maître le peut. `--max-kv` refuse les requêtes dont le cache KV dépasserait la taille indiquée (8g par défaut) ; `--max-heavy` limite les requêtes simultanées de tokenisation, de modèle et de vérification (4 par défaut). `--no-log` fait taire le journal des requêtes. `GET /v1/capabilities` liste ce qui fonctionne. Les routes d'embeddings et de reranking répondent 501.

## Non disponible dans `corbul` (pour l'instant)

- Modèles à attention latente multi-têtes (style DeepSeek), RWKV au format GGUF, Zamba/Zamba2, i-quants, modèles WordPiece/BERT.
- Entrées de vision et d'audio (la bibliothèque a les blocs), points d'accès d'embeddings et de reranking, le back end GPU.
- Identité bit à bit du prefill quantifié ordinaire (voir ENGINE.md, « Exactitude »).
