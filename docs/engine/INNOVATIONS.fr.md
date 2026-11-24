# Ce qui est différent dans ce moteur

[English](INNOVATIONS.md) · [Română](INNOVATIONS.ro.md) · [Français](INNOVATIONS.fr.md) · [Italiano](INNOVATIONS.it.md) · [Español](INNOVATIONS.es.md) · [Português](INNOVATIONS.pt.md) · [Deutsch](INNOVATIONS.de.md) · [Magyar](INNOVATIONS.hu.md)

Notes en langage simple sur les fonctions que la plupart des autres moteurs locaux n'ont pas. Chacune dit ce qu'elle fait, où y accéder et ce qu'elle ne peut pas faire. Toutes ont été testées uniquement sur des modèles synthétiques ; aucune n'a été exécutée par les auteurs sur un vrai point de contrôle. « Dans `corbul` » signifie qu'une commande ou une option existe aujourd'hui (`corbul help`) ; « bibliothèque uniquement » signifie du code Rust sans commande. Voir [ENGINE.fr.md](ENGINE.fr.md) et [QUICKSTART.fr.md](QUICKSTART.fr.md).

## Intégrité, matériel et mémoire

### Mesh en réseau local

Module `mesh`. Dans `corbul` : `worker`, `generate --workers`, `--discover`, `--key`, `--wire-f16`. Un modèle trop gros pour une seule machine est découpé en plages de couches, une par ordinateur. Seul l'état caché circule entre elles. Les machines prouvent qu'elles partagent une clé (HMAC-SHA256 mutuel), chaque trame porte un MAC, les tailles de poignée de main sont plafonnées, un worker mort est écarté et ses couches sont redistribuées. Les logits f32 denses sont identiques à ceux d'une seule machine. La découverte UDP est optionnelle : un worker ne répond aux sondes que s'il a été démarré avec `--discovery` ou `--discovery-port` (`--no-discovery` la désactive de force), et `--discover` du coordinateur diffuse une sonde puis attend les réponses des workers qui détiennent le même modèle. *Ne peut pas :* chiffrer (utilise un réseau local de confiance ou un VPN) ; testé uniquement en boucle locale ; la vitesse par rapport à une seule machine n'a pas été mesurée.

### Streaming de couches

Module `stream_layers`. Dans `corbul` : `--stream-layers`, `--ram-budget`, `--stream-window`, `--disk-mbps` (pour les estimations). Exécute un modèle dense plus grand que la RAM en gardant une petite fenêtre de couches en mémoire et en préchargeant dans un thread d'arrière-plan. Les résultats sont identiques à ceux du modèle en RAM. *Ne peut pas :* éviter que chaque jeton lise chaque couche, donc la vitesse est bornée par celle du disque ; couches denses uniquement (les modèles à mélange d'experts lisent plutôt leurs experts routés en flux, avec `--ram-budget`).

### Citations verrouillées sur la source

Module `quotelock`. Dans `corbul` : `--sources FILE` (répétable), `--no-lock`, `--citations FILE` ; le serveur accepte `sources` et propose `POST /v1/quotes/verify`. Pendant le décodage, le texte entre « et » doit être un passage textuel d'une source (tolérant à la casse, aux espaces, aux styles de guillemets et aux signes diacritiques roumains) ; `[S3]«` limite une citation à la source 3. Un vérificateur classe ensuite chaque citation (textuelle, mauvaise source, tronquée, altérée, fusionnée, non étayée) et note les phrases en dehors des citations ; le rapport indique quand une limite de travail a été atteinte (`limit_hit`). *Ne peut pas :* rendre vrai le texte autour des citations ; les réponses en flux ne portent pas de bloc de citations.

### Planificateur, régulateur et compteur d'énergie

Module `governor` (et `plan`, `hw`). Dans `corbul` : `plan`, `doctor`, `--governor`, `--no-governor`, `--verbose`, `--calibrate`. Avant le chargement, dit si le modèle tient, passe en flux, tourne lentement ou ne peut pas tourner, avec une vitesse estimée. Pendant l'exécution, le régulateur lit la température, la mémoire et la batterie (Linux) et conseille le nombre de threads et des pauses sans osciller. *Ne peut pas :* donner autre chose que des estimations de vitesse (`--calibrate` chronomètre ta machine pendant environ une seconde) ; l'énergie est une hypothèse sauf si un compteur RAPL est lisible ; l'intensité carbone (400 g/kWh) est une valeur provisoire.

### Politiques de cache KV et instantanés de session

Modules `kvpolicy`, `snapshot`. Dans `corbul` : `--kv-policy streaming|h2o|snapkv`, `--kv-budget`, `--kv-sinks`, `--session DIR`, `--session-cap`, `--compress`. Garde les longues conversations dans un budget mémoire (puits d'attention, gros contributeurs, compression du prompt) et enregistre ou reprend une conversation depuis un fichier à somme de contrôle, en réutilisant le plus long préfixe enregistré. Les instantanés peuvent porter un HMAC à clé. *Ne peut pas :* éviter les pertes (l'éviction est avec perte par conception) ; `--kv-policy` et `--session` ne se combinent pas ; de très longues conversations peuvent dépasser le contexte d'entraînement.

### Mode reproductible et reçus

Modules `simd` (commutateur), `receipt`. Dans `corbul` : `--reproducible`, `--receipt FILE`, `--receipt-key`, `--receipt-key-file`, `--receipt-topk`, `--keep-steps`, `--leaves`, `verify-receipt`, `replay`. Un reçu consigne le modèle (une empreinte de Merkle des poids), le tokenizer, les réglages de l'échantillonneur, le prompt et la sortie dans une chaîne de hachages, scellés par SHA-256 et une clé HMAC optionnelle ; `replay` le rejoue. Des poids endommagés sont localisés jusqu'à un tenseur et un fragment. *Ne peut pas :* prouver quoi que ce soit à un tiers sans la clé HMAC ni une nouvelle exécution (pas de signatures à clé publique) ; les modèles MoE et à espaces d'états sont refusés ; ARM non testé ; le mode reproductible coûte en vitesse.

### Outils linguistiques pensés d'abord pour le roumain

Module `lang`. Dans `corbul` : `lang detect`, `lang normalize`, `--lang CODE` (pas dans le serveur). Normalise ș/ț et ş/ţ, répare le mojibake, identifie 27 langues, interdit les écritures étrangères pendant le décodage (roumain : pas de ş ţ), formate les nombres, les montants et les dates roumains, restitue les signes diacritiques manquants en signalant chaque mot incertain. *Ne peut pas :* aller au-delà d'un lexique d'environ 650 formes pour la restitution, non mesuré sur un corpus ; les mots ambigus sont tranchés par quelques règles de contexte ; l'identification de la langue demande environ cinq mots.

### Analyse d'incertitude

Module `confidence`. Bibliothèque uniquement (exemple de démonstration `confidence_demo`). Statistiques par jeton, mot, phrase et réponse pour une carte de chaleur, une heuristique transparente de risque d'hallucination, des scores d'auto-cohérence et des outils d'étalonnage. *Ne peut pas :* être autre chose qu'une heuristique : le score de risque n'est pas un détecteur de vérité.

## Calcul à l'inférence, réglage, recherche

### Recherche et lookahead

Module `search`. Dans `corbul` : `search --method beam|best-of-n|vote` (`--width`, `--samples`, `--answer number|text`, `--length-penalty`, `--all`, `--explain`, `--json`) et `--lookahead` sur `generate` (`--ngram-min`, `--ngram-max`). Recherche en faisceau, meilleur de N, vote d'auto-cohérence, arbre de conversation (bibliothèque), décodage spéculatif par recherche dans le prompt. *Ne peut pas :* aider si le texte ne répète pas du texte antérieur (le lookahead) ; le meilleur de N exige un bon évaluateur.

### Décodage spéculatif

Modules `spec`, `selfspec`. Dans `corbul` : `--draft MODEL`, `--lookahead`, `--self-spec`, `--skip-layers auto|LIST`, `--draft-len`. Même réponse gloutonne et même distribution d'échantillonnage avec moins de passes complètes ; la sortie apparaît à la fin. Seulement `--temperature`, `--top-k`, `--top-p` ; pas avec la sortie structurée, les reçus, les sessions ni le mesh. *Ne peut pas :* promettre une accélération ; l'auto-spéculation revient à des pas simples pour les anneaux à fenêtre glissante qui reboucleraient, les caches compactés et les couches déportées.

### Quantification mixte automatique et quantificateur

Modules `autoquant`, `quantize`. Dans `corbul` : `autoquant FOLDER --budget 4.5bpw|6g|Q4_K` (`--recipe`, `--report`, `--candidates`, `--base`, `--imatrix`, `--sample-rows`) et `quantize IN OUT.gguf --type NAME | --recipe FILE`. Mesure l'erreur sur les poids par groupe de tenseurs dans chaque format et choisit les formats sous un budget de taille (sac à dos exact ou glouton). *Ne peut pas :* estimer la qualité des réponses, seulement l'erreur sur les poids ; il suppose que les dégâts s'additionnent ; les i-quants n'existent pas.

### Vecteurs de pilotage et logit lens

Module `steer`. Dans `corbul` : `steer train|apply|lens` (`--positive`, `--negative`, `--method mean-diff|pca`, `--out`, `--vector`, `--strength`, `--layers`, `--positions`, `--no-attention`). Apprend une direction à partir de deux jeux de prompts et l'ajoute au flux résiduel sans modifier les poids ; la lentille montre ce que chaque couche prédirait. *Ne peut pas :* fonctionner avec les modèles à espaces d'états, la sortie structurée, la spéculation, les reçus ni les sessions ; un vecteur issu d'un fine-tuning est accepté par un autre de même forme.

### Recherche locale

Module `rag`. Dans `corbul` : `rag add|search|ask` (`--query`, `--question`, `--k`, `--chunk-chars`, `--overlap`, `--embed-model`). Découpage adapté au roumain, index HNSW en Rust pur, BM25 avec repli des signes diacritiques, fusion de rangs, passages numérotés `[S1]…` alimentant les citations verrouillées sur la source, fichiers à somme de contrôle. *Ne peut pas :* faire mieux que l'encodeur par défaut, un encodeur par hachage (mots et fragments de mots, pas le sens) ; `--embed-model` utilise les états cachés d'un modèle génératif (faible) ; la CLI réindexe à chaque appel et lit du texte brut, pas des fichiers PDF ou Office (la bibliothèque a des aides d'ingestion).

### Fusion de modèles

Module `merge`. Dans `corbul` : `merge CONFIG.json` (`--out`, `--model`, `--dry-run`, `--max-memory`, `--overwrite`, `--verbose`). Linéaire, SLERP, arithmétique de tâches, TIES, DARE (à graine) et tranches de couches, un tenseur à la fois. Les formes et les plages sont vérifiées avant l'écriture ; les liens symboliques ne sont pas suivis. *Ne peut pas :* garantir la qualité ; n'écrit que des poids safetensors, jamais quantifiés, et tu copies toi-même `config.json` et le tokenizer.

### Filigrane de texte

Module `watermark`. Dans `corbul` : `generate --watermark KEY` (`--wm-gamma`, `--wm-delta`, `--wm-context`) et `watermark detect MODEL --key K --text T | --file F` (`--no-dedup`). Un biais de « liste verte » à clé et un détecteur à score z. *Ne peut pas :* donner une preuve ; les textes courts, le code et les citations portent peu de signal ; la paraphrase l'efface ; quiconque a la clé peut le falsifier ; l'absence de marque ne prouve pas une écriture humaine.

## Multimodal (bibliothèque uniquement)

### Vision

Modules `vision`, `imageio`. Un encodeur de style CLIP/SigLIP, un projecteur, le découpage en tuiles et un point d'accroche `Model::forward_segments` pour injecter des embeddings dans un prompt ; des décodeurs PNG, JPEG (de base et progressif), BMP et PNM écrits d'après les spécifications, avec des limites de taille. *Ne peut pas :* être utilisé depuis `corbul` (texte uniquement) ; aucun vrai point de contrôle n'a été exécuté.

### Audio

Module `audio`. Lecture et écriture WAV, rééchantillonnage, FFT, caractéristiques log-mel, détection d'activité vocale, un exécuteur de structure Whisper. *Ne peut pas :* être utilisé depuis `corbul` ; aucun vrai point de contrôle Whisper n'a été exécuté.

## Espaces d'états, hybrides et GPU (voir ENGINE.md)

Les modules `ssm`, `hybrid`, `mla` et le crate `companion-engine-gpu` sont décrits, avec leurs limites, dans ENGINE.md.
