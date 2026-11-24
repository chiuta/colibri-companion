# Démarrage rapide : `corbul`

[English](QUICKSTART.md) · [Română](QUICKSTART.ro.md) · [Français](QUICKSTART.fr.md) · [Italiano](QUICKSTART.it.md) · [Español](QUICKSTART.es.md) · [Português](QUICKSTART.pt.md) · [Deutsch](QUICKSTART.de.md) · [Magyar](QUICKSTART.hu.md)

`corbul` est la ligne de commande du moteur clean-room ([ENGINE.fr.md](ENGINE.fr.md)). Cette page ne liste que les commandes et options qui existent dans `crates/engine-cli` au moment de la rédaction ; la ligne de commande s'étoffe encore, donc `corbul help` a le dernier mot. Certaines options ci-dessous sont lues par le code mais pas encore affichées par `corbul help` ; elles sont marquées *(pas encore dans l'aide)*. Ce qui n'existe pas encore est marqué **prévu**.

**Avertissement honnête :** `corbul` n'a été testé que sur de minuscules modèles synthétiques. Attends-toi à des aspérités sur de vrais points de contrôle et signale-les.

## Construire

```bash
cargo build --release -p colibri-rs-cli     # produit target/release/corbul
```

Un modèle est soit un dossier avec `config.json` + `*.safetensors` + `tokenizer.json`, soit un seul fichier `.gguf`.

## Regarder, télécharger, lancer

```bash
corbul inspect  ./my-model                  # ou file.gguf ou owner/repo[@rev] : architecture, taille, prise en charge
corbul download owner/repo --quant Q4_K_M --dest ./models     # --format gguf|safetensors ; HF_TOKEN pour les dépôts à accès restreint, HF_ENDPOINT pour un miroir
corbul generate ./my-model --prompt "Bonjour" --max-tokens 64
corbul chat     ./my-model --system "Sois concis."
corbul serve    ./my-model --port 8080      # compatible OpenAI / Ollama / Anthropic, lié par défaut à 127.0.0.1
```

Options communes à `generate`, `chat` et `serve` : `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--kv f32|f16|q8`, `--no-mmap`, `--threads N`, `--ctx N`, `--model-name`, `--quiet`. `generate` accepte aussi `--prompt-file F` et `--raw` (sans modèle de chat). `corbul run` est un alias de `generate`.

Servir plusieurs modèles : `--models-dir DIR`, `--alias NOM=MODÈLE`, `--max-loaded N`, `--sleep-idle-seconds S`, `--token T` (ou `CORBUL_TOKEN`), `--keys-file FICHIER`, `--cors ORIGINES`, `--log-prompts`, `--no-log`, `--drain-seconds S`, `--config-dir DIR`.

## Est-ce que ça tournera sur cette machine ?

```bash
corbul doctor                # ce que voit le moteur : jeux d'instructions CPU, noyaux, RAM, disque, capteurs, compteur d'énergie
corbul plan ./my-model --context 8192            # verdict : FITS / TIGHT / STREAMS / TOO SLOW / WON'T RUN, vitesse, conseils
corbul plan ./my-model --ram-budget 8g --calibrate   # fais comme si tu avais 8 Go ; chronomètre d'abord la machine une seconde
corbul capabilities [--json]                     # ce que cette version peut et ne peut pas faire
```

Les vitesses sont des estimations, pas des mesures de ton modèle.

## Plus gros que la RAM

```bash
corbul generate ./big-moe --ram-budget 16g --prompt "…"                        # MoE : les experts routés sont lus depuis le disque
corbul generate ./big-dense --stream-layers --ram-budget 16g --stream-window 4 --prompt "…"   # dense : les couches sont diffusées (limité par le disque ; --stream-window pas encore dans l'aide)
```

## Plusieurs machines (maillage LAN)

```bash
# sur chaque machine d'appoint (la clé fait 8 caractères minimum ; ou définis CORBUL_MESH_KEY)
corbul worker ./my-model --key SECRET --bind 0.0.0.0:7471          # --no-discovery évite la découverte UDP
# sur la machine principale
corbul generate ./my-model --workers 192.168.1.20:7471,192.168.1.21:7471 --key SECRET --prompt "…"
corbul generate ./my-model --discover --key SECRET --prompt "…"      # trouve les workers sur le LAN
```

Le trafic est authentifié mais **non chiffré** : utilise un réseau de confiance ou un VPN. Testé en boucle locale seulement.

## Sortie structurée, citations, adaptateurs

```bash
corbul generate M --json --prompt "…"                 # sortie JSON valide
corbul generate M --json-schema schema.json --prompt "…"      # fichier ou JSON en ligne ; aussi --regex MOTIF, --grammar FICHIER (GBNF) ; une seule des quatre ; pas encore dans l'aide
corbul generate M --sources notes.txt --sources paper.txt --prompt "Réponds avec des citations" --citations report.json   # citations verrouillées
corbul generate M --lora ./adapter:0.8                # adaptateur PEFT ou GGUF avec un facteur
```

Le serveur accepte `response_format`, `grammar` et `sources`, et offre `POST /v1/quotes/verify`.

## Longues conversations, sessions, reproductibilité

```bash
corbul generate M --session ./sessions --prompt "…"             # réutilise le plus long préfixe de prompt sauvegardé (pas avec --kv-policy)
corbul generate M --kv-policy h2o --kv-budget 2048 --prompt "…" # streaming | h2o | snapkv ; avec perte par construction
corbul generate M --governor --prompt "…"                       # s'adapte à la température, à la mémoire et à la batterie (Linux)
corbul generate M --reproducible --temperature 0 --prompt "hi" --receipt r.json --receipt-key SECRET
corbul verify-receipt r.json --key SECRET --model M            # vérifie le sceau (et les poids si --model est donné)
corbul replay r.json --model M --key SECRET                    # rejoue la génération et compare
```

Les reçus fonctionnent avec la boucle de génération simple, pas avec les modèles MoE. Facultatif : `--receipt-topk N`, `--keep-steps`, `--leaves` *(pas encore dans l'aide)*.

## Prévu (pas encore dans `corbul`)

- Le chargement de modèles Mamba / RWKV / MLA (style DeepSeek), bien que la bibliothèque ait les blocs.
- Les outils de langue roumaine, l'analyse d'incertitude, la recherche et le lookahead, le décodage auto-spéculatif, le pilotage, la recherche documentaire, la fusion et le filigrane (voir [INNOVATIONS.fr.md](INNOVATIONS.fr.md)).
- Les points d'accès d'embedding et de rerank (le serveur répond « not implemented »), les entrées image et audio.
- Le backend GPU et une interface web.
