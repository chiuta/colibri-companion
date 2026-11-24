# Le moteur clean-room (`companion-engine` et `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Autres documents : [INNOVATIONS.fr.md](INNOVATIONS.fr.md) · [QUICKSTART.fr.md](QUICKSTART.fr.md) · [CLEANROOM.fr.md](CLEANROOM.fr.md) · [CHANGES-SECURITY.fr.md](CHANGES-SECURITY.fr.md)

**Statut : alpha.** Tout ce qui suit a été testé sur de petits points de contrôle synthétiques, sur des fichiers de référence du tokenizer et sur des textes d'exemple enregistrés. **Aucun vrai modèle téléchargé depuis Hugging Face n'a été exécuté avec ce moteur par ses auteurs**, aucun GPU n'en a exécuté la moindre partie, et les chemins de code ARM, macOS et Windows n'ont pas été exécutés ici. Lis « Vérifié et non vérifié » avant de croire une affirmation. `corbul capabilities` affiche les mêmes limites à partir du binaire lui-même.

## Ce que c'est

Un moteur d'inférence local pour modèles de langage, écrit de zéro en Rust à partir de descriptions publiques de formats de fichiers et d'articles publiés.

| Crate | Ce que c'est |
|---|---|
| `crates/engine` (`companion-engine`) | La bibliothèque : formats de fichiers, tokenizer, quantification, passes avant, échantillonnage et les outils supplémentaires d'INNOVATIONS.md. Dépendances à l'exécution : `serde_json` et `memmap2` uniquement. |
| `crates/engine-cli` (`corbul`) | La ligne de commande et un serveur local compatible OpenAI / Ollama / Anthropic avec une interface web intégrée. Dépend de la bibliothèque et de `serde_json`. |
| `crates/engine-gpu` (`companion-engine-gpu`) | Des noyaux WGSL et une émulation CPU de ces noyaux. L'exécuteur sur périphérique (`wgpu`, `pollster`) se trouve derrière la feature cargo `wgpu`, désactivée par défaut. **`corbul` n'utilise pas ce crate.** |

Il est distinct de l'application de bureau Colibri Companion, qu'il ne remplace pas : celle-ci pilote toujours le colibri de JustVugg. Utiliser ce moteur comme autre back end de l'application est prévu, pas réalisé.

## Ce qu'il prend en charge

**Fichiers.** GGUF v2 et v3 (l'en-tête et la table des tenseurs sont lus sans toucher aux poids) ; un écrivain GGUF en flux pour les sorties plus grandes que la RAM ; safetensors, unique ou fragmenté avec un index ; `config.json`, `tokenizer.json` et modèles de chat de Hugging Face ; un téléchargeur de modèles (module `hub`, via `curl`, avec `wget` en repli anonyme ; reprise, vérifications SHA-256, téléchargements épinglés sur un commit, plafonds de taille, refus des redirections vers du http en clair).

**Quantification.** Décodés : F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K (14 types). Des encodeurs existent pour les mêmes types, avec pondération par matrice d'importance et recettes par tenseur. Les poids restent quantifiés en mémoire ; le produit matrice–vecteur ne développe jamais la matrice entière. **Non pris en charge :** les i-quants (IQ\*), et tout type absent de la liste.

**Architectures.**

| Famille | État dans `corbul` |
|---|---|
| Décodeurs denses : Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 et similaires (normalisation RMS ou layer norm, GQA/MQA, RoPE avec mise à l'échelle linéaire, Llama-3, YaRN et LongRoPE et rotation partielle, QK-norm, biais, soft-capping, fenêtres glissantes, MLP à portes ou simple) | Chargés et exécutés (modèle de chat, serve, toutes les options) |
| Mélange d'experts : Mixtral, Qwen-MoE, experts empilés GGUF, experts partagés, experts routés lus en flux depuis le disque avec un cache LRU | Chargés et exécutés |
| Espaces d'états : Mamba-1, Mamba-2, RWKV-6, RWKV-7 (module `ssm`) | Exécutés par `generate` et `chat` en **continuation de texte brut** : pas de modèle de chat, pas de sortie structurée, pas de `serve`. RWKV au format GGUF n'est pas pris en charge (safetensors uniquement) |
| Hybrides : Jamba et Bamba (module `hybrid`) | Comme la ligne des espaces d'états. Zamba et Zamba2 ne sont pas pris en charge |
| Attention latente multi-têtes (DeepSeek-V2/V3, Kimi-K2 ; module `mla`) | **Bibliothèque uniquement.** Le bloc existe et sa cohérence interne est testée, mais `corbul` n'assemble pas un tel modèle (`corbul capabilities` répond non) |
| Encodeur de vision (ViT, CLIP, SigLIP ; module `vision`, décodeurs dans `imageio`) et front-end vocal avec un exécuteur de structure Whisper (module `audio`) | **Bibliothèque uniquement.** `corbul` n'accepte que du texte |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, modèles encodeur et encodeur-décodeur (T5, BERT…), tokenizers WordPiece | Non pris en charge ; `corbul` le dit au lieu de deviner |
| Points d'accès d'embeddings et de reranking | Répondent 501 (non implémentés) |

**Tokenizer.** Modèles BPE au niveau de l'octet et de style SentencePiece, règles de pré-tokenisation pour 12 familles, jetons ajoutés, repli sur les octets, modèles issus de `tokenizer_config.json` et des métadonnées GGUF. WordPiece/BERT n'est pas pris en charge.

**Échantillonnage.** Température, top-k, top-p, min-p, typical-p, pénalités de répétition / de présence / de fréquence, Mirostat 1 et 2, DRY, XTC, une spécification d'échantillonneur JSON, aléa à graine ; décodage contraint par JSON, schéma JSON, expression régulière ou grammaire GBNF (avec des budgets de travail pour qu'une grammaire hostile ne puisse pas bloquer le processus) ; biais de logits et chaînes d'arrêt via le serveur. Décodage spéculatif : modèle brouillon, recherche dans le prompt, auto-spéculation (voir INNOVATIONS.md).

**Serveur.** `corbul serve` propose `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/messages` et `/v1/messages/count_tokens` (style Anthropic), les points d'accès Ollama `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/create`, `/api/delete`, `/api/ps`, `/api/version`, la gestion des modèles `/models`, `/models/load`, `/models/unload`, ainsi que `/metrics`, `/props`, `/slots`, `/tokenize`, `/detokenize`, `/apply-template`, `/health`, `/v1/capabilities`, `/v1/quotes/verify` et l'interface web sur `/` et `/ui` (huit langues). L'appel d'outils est pris en charge via les modèles de chat. Les routes d'embeddings et de reranking (`/v1/embeddings`, `/v1/rerank`, `/api/embed`, `/api/embeddings`) répondent 501. Les réponses en flux ne portent pas de bloc de citations.

**Matériel.** Les noyaux CPU sont choisis à l'exécution : scalaire, avx2+fma, avx2+fma+avxvnni, avx512, avx512+vnni (x86-64), neon et neon+dotprod (aarch64) ; `ENGINE_SIMD` en impose un. Un pool de threads persistant, la détection du matériel et du NUMA, et un planificateur (`corbul plan`, `corbul doctor`) décident ce qui reste en RAM. Les noyaux GPU n'ont jamais tourné sur un GPU (voir plus bas).

## Exactitude

- **Poids f32 denses :** les chemins par lots et à un seul jeton, le mesh, le streaming de couches et les caches donnent des logits identiques bit à bit.
- **Poids quantifiés :** l'identité bit à bit entre l'évaluation par lots et l'évaluation à un seul jeton n'est vraie que lorsque `exact_batch` est activé. La vérification, les tours de vérification auto-spéculatifs et l'exécuteur par lots l'activent. Le prefill ordinaire utilise un chemin plus rapide (déquantifier puis découper en tuiles) et ne diffère de l'évaluation à un seul jeton que par du bruit d'arrondi.
- **Mode reproductible** (`--reproducible` ou `ENGINE_REPRODUCIBLE=1`) : résultats identiques d'une variante de noyau x86 et d'un nombre de threads à l'autre, dans la limite de ce que nous avons pu tester. ARM n'a pas été testé. Il coûte en vitesse.
- Mirostat 1 utilise dans son estimation le nombre de candidats restants, et non la taille du vocabulaire (écart documenté).

## Vérifié et non vérifié

| Domaine | État |
|---|---|
| Tests unitaires, d'intégration, adversariaux et de fuzzing sur des points de contrôle synthétiques (Linux x86-64). Les nombres changent à chaque commit ; lance `cargo test --workspace` pour le chiffre actuel | Réussis lors de la dernière exécution complète. Les fichiers hostiles et tronqués renvoient des erreurs. |
| Tokenizer face aux fichiers de référence de llama.cpp, 12 familles | 46 cas sur 46 identiques (fichiers de référence récupérés sur GitHub ; `crates/engine/tests/tokenizer_real.rs`, `ENGINE_VOCAB_DIR`). Seulement le tokenizer, pas des modèles entiers. |
| Vrai modèle Hugging Face via `corbul` (toute famille) | **Non exécuté par les auteurs.** Le bac à sable de compilation n'a pas pu joindre huggingface.co. Un job CI manuel exécute un minuscule GGUF llama (`stories260K`) via la CLI ; il n'a pas encore tourné |
| Permutation des lignes Q/K à la manière de llama.cpp, variantes des métadonnées GGUF | Implémentées d'après des descriptions publiques ; non vérifiées sur de vrais fichiers |
| Quantificateurs et déquantificateurs | Blocs calculés à la main et allers-retours ; non comparés à des fichiers écrits par d'autres outils |
| Exécution GPU | **Non exécutée.** Les noyaux sont validés par `naga` et par une émulation Rust uniquement ; `corbul` ne les utilise pas |
| ARM / NEON, macOS, Windows | **Non exécutés ici.** Les sondes sont testées sur des textes d'exemple enregistrés. Le workflow CI a une matrice pour eux mais n'a jamais tourné |
| Mamba, RWKV, Jamba/Bamba, MLA, vision, audio | Tests de cohérence interne uniquement ; aucun vrai point de contrôle n'a été utilisé |
| Mesh | Boucle locale uniquement ; authentifié, non chiffré ; aucune mesure de vitesse |
| Vitesse | **Aucun chiffre fiable.** La machine de compilation avait deux cœurs partagés. Les vitesses de l'autopilote sont des estimations |
| Licences des dépendances | Vérifiées avec `cargo-deny` au commit b496ba5 ; les changements ultérieurs n'ont pas été revérifiés |
| Traductions des documents | Assistées par machine ; pas encore de relecture par un locuteur natif |

## Variables d'environnement

| Variable | Effet |
|---|---|
| `COLIBRI_THREADS` | Taille du pool de threads global (par défaut : cœurs utilisables). Lue en premier par la bibliothèque |
| `ENGINE_SIMD` | Impose une variante de noyau par son nom (benchmarks, rapports de bogues) |
| `ENGINE_REPRODUCIBLE` | `1`, `true` ou `on` : mode reproductible dès le départ |
| `ENGINE_BACKEND` | `cpu`, `gpu` ou `auto` (non définie = auto). `corbul` calcule sur le CPU dans tous les cas |
| `ENGINE_GPU`, `ENGINE_GPU_API`, `ENGINE_GPU_SOFTWARE` | Choix de l'adaptateur pour le crate GPU : index, API graphique, autoriser un adaptateur logiciel. Non utilisées par `corbul` |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Jeton d'accès aux dépôts à accès restreint (`download`, `inspect owner/repo`) |
| `HF_ENDPOINT` | Miroir de l'hôte Hugging Face |
| `CORBUL_TOKEN` | Jeton bearer partagé de `corbul serve` (équivaut à `--token`) |
| `CORBUL_MESH_KEY` | Clé du mesh pour `worker` et `--workers`/`--discover` (équivaut à `--key`) |
| `CORBUL_CONFIG_DIR` | Où sont conservés les Modelfiles créés par `POST /api/create` (équivaut à `--config-dir` ; par défaut `~/.config/corbul`) |
| `NUMBER_OF_PROCESSORS`, `PROCESSOR_IDENTIFIER` | Lues sous Windows par la sonde matérielle ; tu ne les définis pas |
| `ENGINE_VOCAB_DIR`, `ENGINE_VOCAB_ONLY`, `ENGINE_REQUIRE_VOCAB`, `ENGINE_REQUIRE_TOOLS`, `ENGINE_PRINT_GOLDEN`, `ADVERSARIAL_N` | Réglages de la suite de tests : dossier des vocabulaires, restriction des familles, faire échouer un test ignoré, faire échouer l'absence de `curl`/`wget`, régénérer les empreintes de référence, taille de l'exécution adversariale |

## Déclaration clean-room

Aucun code de colibri, de llama.cpp ni d'aucun autre moteur d'inférence n'a été lu ou copié pendant l'écriture de ce moteur. Les formats et les algorithmes viennent de spécifications publiques, d'articles et de dispositions de fichiers. Le modèle qui a écrit le code a été entraîné sur des textes publics qui peuvent inclure de tels projets ; voir [CLEANROOM.fr.md](CLEANROOM.fr.md) pour le processus et ses limites.

## Comparaison avec les autres moteurs locaux

Nous n'avons réalisé aucun benchmark comparatif, donc aucune affirmation de vitesse ou de qualité n'est faite ici.

- Les moteurs établis (llama.cpp, Ollama, mistral.rs, vLLM) ont été exécutés sur des milliers de vrais points de contrôle, disposent de back ends GPU fonctionnels, couvrent plus d'architectures et de formats de quantification, i-quants compris, et gèrent les modèles de vision. Ce moteur n'a encore aucune de ces preuves. Si tu as besoin de quelque chose qui fonctionne dès aujourd'hui sur un gros modèle précis, utilise l'un d'eux.
- Ce que ce moteur essaie de faire autrement : des outils d'intégrité intégrés (citations verrouillées sur la source, mode reproductible avec reçus, analyse d'incertitude, un planificateur qui dit « ça ne tournera pas » au lieu de planter), des outils linguistiques pensés d'abord pour le roumain, des modèles plus grands que ta RAM ou que ta machine, une petite base de code Rust avec deux dépendances de bibliothèque, et uniquement des licences permissives.

## Carte de la bibliothèque

Formats et chargement : `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`, `imageio`. Modèles : `arch`, `model`, `moe`, `mla`, `ssm`, `hybrid`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Exécution : `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Fonctions distinctives : `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Calcul à l'inférence, réglage, recherche : `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark`. Multimodal : `vision`, `audio`. Ce que `corbul` atteint parmi ces modules est indiqué dans INNOVATIONS.md ; les API de la bibliothèque peuvent encore changer.
