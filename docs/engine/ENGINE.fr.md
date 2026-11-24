# Le moteur « clean-room » (`companion-engine` et `corbul`)

[English](ENGINE.md) · [Română](ENGINE.ro.md) · [Français](ENGINE.fr.md) · [Italiano](ENGINE.it.md) · [Español](ENGINE.es.md) · [Português](ENGINE.pt.md) · [Deutsch](ENGINE.de.md) · [Magyar](ENGINE.hu.md)

Autres documents : [INNOVATIONS.fr.md](INNOVATIONS.fr.md) · [QUICKSTART.fr.md](QUICKSTART.fr.md) · [CLEANROOM.fr.md](CLEANROOM.fr.md)

**État : alpha. Tout ce qui suit n'a été testé que sur de petits points de contrôle synthétiques.** Les auteurs n'ont fait passer aucun vrai modèle de Hugging Face dans ce moteur (l'environnement de construction n'atteignait pas huggingface.co), aucun GPU n'en a exécuté la moindre partie, et les chemins ARM, macOS et Windows n'ont jamais été compilés ici. Lis le tableau « vérifié et non vérifié » avant de te fier à une affirmation.

## Ce que c'est

Un moteur d'inférence local pour modèles de langage, écrit de zéro en Rust, à partir de descriptions publiques de formats de fichiers et d'articles publiés. Trois parties :

| Crate | Ce que c'est |
|---|---|
| `crates/engine` (`companion-engine`) | La bibliothèque : formats de fichiers, tokenizer, quantification, passes avant, échantillonnage et les outils supplémentaires décrits dans INNOVATIONS.fr.md. Dépendances : `serde_json` et `memmap2` seulement. |
| `crates/engine-cli` (`corbul`) | Une ligne de commande et un serveur local compatible OpenAI / Ollama / Anthropic, au-dessus de la bibliothèque. |
| `crates/engine-gpu` (`companion-engine-gpu`) | Des noyaux WGSL pour le crate `wgpu`, derrière une option désactivée par défaut. Non utilisé par `corbul`. |

Il est distinct de l'application de bureau Colibri Companion et ne la remplace pas : l'application pilote toujours colibri de JustVugg. Brancher ce moteur dans l'application comme backend alternatif est prévu, pas fait.

## Ce qu'il prend en charge

**Fichiers.** GGUF v2/v3 (l'en-tête et la table des tenseurs se lisent sans toucher aux poids : un fichier de 400 Go s'inspecte en millisecondes) ; un écrivain GGUF en flux pour les modèles plus grands que la RAM ; safetensors, en un fichier ou fragmenté avec index ; `config.json` et `tokenizer.json` de Hugging Face ; un téléchargeur de modèles (via `curl`, avec reprise et contrôle SHA-256).

**Quantification.** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 et les K-quants Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K se décodent, et il existe des encodeurs pour tous (avec pondération par matrice d'importance), la conversion Hugging Face → GGUF et la re-quantification GGUF. Les poids restent quantifiés en mémoire ; le produit matrice–vecteur ne construit jamais la matrice. Les i-quants (IQ*) ne sont pas implémentés.

**Architectures.**

| Famille | État |
|---|---|
| Décodeurs denses : Llama, Mistral, Qwen 2/3, Gemma 1–3, Phi-3, StarCoder2 et similaires (norme RMS ou de couche, GQA/MQA, RoPE avec mise à l'échelle linéaire / Llama-3 / YaRN et rotation partielle, QK-norm, biais, soft-capping, fenêtres glissantes, MLP à porte ou simple) | Implémentées ; chargées par `corbul` |
| Mélange d'experts : Mixtral, Qwen-MoE, experts empilés GGUF, experts partagés, experts routés lus depuis le disque avec cache LRU | Implémentées ; chargées par `corbul` |
| Attention latente multi-têtes (style DeepSeek-V2/V3, Kimi-K2) | Module de bibliothèque seulement ; le registre d'architectures la marque encore « prévue » et `corbul` ne la charge pas |
| Mamba-1, Mamba-2, RWKV-6, RWKV-7 | Module de bibliothèque seulement ; idem |
| Falcon, GPT-2, GPT-J, OPT, BLOOM, MPT, OLMoE, gpt-oss, modèles encodeur et encodeur-décodeur (T5, BERT, Whisper…), LongRoPE | Non pris en charge ; `corbul` le dit au lieu de deviner |
| Entrées image et audio | Non prises en charge (texte seulement) |

**Chemins matériels.**

- CPU : noyaux choisis à l'exécution (scalaire, AVX2, AVX-512, VNNI sur x86-64 ; NEON sur aarch64), un pool de threads persistant adapté aux cœurs performants/économes mélangés, détection du matériel et du NUMA, et un planificateur qui décide ce qui reste en RAM.
- GPU : noyaux WGSL vérifiés avec `naga` et face à une émulation en Rust. Ils n'ont jamais tourné sur un GPU.
- Modèles plus grands que la RAM : experts depuis le disque (MoE), diffusion des couches (dense), un maillage LAN sur plusieurs machines.

## Vérifié et non vérifié

| Domaine | État |
|---|---|
| Tests unitaires, d'intégration et de fuzzing sur points de contrôle synthétiques (Linux, x86-64) | Vérifié. Des centaines de tests passent ; les fichiers hostiles ou tronqués renvoient des erreurs, pas des plantages. |
| Vrais points de contrôle Hugging Face (toute famille) | **Non exécuté.** L'environnement n'atteignait pas huggingface.co. |
| Tokenizer sur de vrais fichiers `tokenizer.json` | Écrit d'après les descriptions publiques du format ; un banc de test existe (`ENGINE_VOCAB_DIR`) mais nous ne l'avons pas lancé sur de vrais fichiers. |
| Permutation des lignes Q/K à la manière de llama.cpp dans les fichiers GGUF | Implémentée d'après la description ; non vérifiée sur de vrais fichiers. |
| Quantificateurs et déquantificateurs | Vérifiés face à des blocs calculés à la main et par aller-retour ; pas face à des fichiers écrits par d'autres outils. |
| Exécution GPU | **Non exécutée.** Les noyaux ne sont validés que par `naga` et une émulation en Rust. |
| Chemins ARM / NEON, macOS, Windows | **Jamais compilés ni exécutés ici.** Les sondes sont testées sur du texte enregistré de ces systèmes. |
| Formules Mamba / RWKV / MLA | Écrites d'après les articles et le souvenir des descriptions de référence, testées pour la cohérence interne seulement. |
| Maillage sur un vrai réseau | Testé en boucle locale seulement. |
| Vitesse | **Aucun chiffre fiable.** La machine de construction avait deux cœurs partagés ; les mesures étaient bruitées. Toutes les vitesses de l'autopilote sont des estimations. |
| Mode reproductible | Identique sur cinq variantes de noyaux x86 et plusieurs nombres de threads ; ARM non testé. |
| Licences des dépendances | Vérifiées avec `cargo-deny` à un commit antérieur (b496ba5) ; les ajouts ultérieurs n'ont pas encore été revérifiés. |

## Déclaration clean-room

Aucun code de colibri, de llama.cpp ou d'un autre moteur d'inférence n'a été lu ni copié pour écrire ce moteur. Les formats et algorithmes viennent de spécifications publiques, d'articles et de structures de fichiers. Voir [CLEANROOM.fr.md](CLEANROOM.fr.md) pour le processus et ses limites.

## Comparaison avec d'autres moteurs locaux

Nous n'avons lancé aucun comparatif direct : pas d'affirmation de vitesse ou de qualité. Qualitativement :

- Les moteurs établis comme llama.cpp, Ollama, mistral.rs ou vLLM ont été exécutés sur des milliers de vrais points de contrôle par de nombreuses personnes, ont des backends GPU fonctionnels, couvrent plus d'architectures et de formats de quantification (y compris les i-quants) et gèrent les modèles de vision. Ce moteur n'a encore aucune de ces preuves. Si tu as besoin de quelque chose qui marche aujourd'hui sur un gros modèle précis, utilise l'un d'eux.
- Ce que ce moteur essaie de faire autrement : des outils d'intégrité et d'honnêteté intégrés (citations verrouillées sur les sources, mode reproductible avec reçus scellés par chaîne de hachages et HMAC optionnel, analyse d'incertitude, un autopilote qui dit « ça ne tournera pas » au lieu de planter), des outils linguistiques où le roumain passe en premier, l'exécution de modèles plus grands que ta RAM ou qu'une seule machine, une petite base de code Rust avec deux dépendances et uniquement des licences permissives.
- Toute affirmation que ce moteur serait plus rapide ou plus précis qu'un autre serait non testée. Mesure sur ton matériel et dis-le-nous.

## Carte de la bibliothèque

Formats et chargement : `gguf`, `gguf_write`, `loader`, `hub`, `quant`, `quantize`, `tokenizer`. Modèle : `arch`, `model`, `moe`, `mla`, `ssm`, `family`, `ops`, `qmatmul`, `simd`, `pool`, `backend`. Exécution : `plan`, `hw`, `sched`, `sampling`, `grammar`, `spec`, `lora`, `eval`. Fonctions distinctives : `mesh`, `stream_layers`, `quotelock`, `governor`, `kvpolicy`, `snapshot`, `receipt`, `lang`, `confidence`. Dernière vague : `search`, `selfspec`, `autoquant`, `steer`, `rag`, `merge`, `watermark` — implémentées dans la bibliothèque et couvertes par des tests sur modèles synthétiques, encore en finition, pas encore accessibles depuis `corbul`, et leurs API peuvent changer.
