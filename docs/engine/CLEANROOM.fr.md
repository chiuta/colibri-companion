# Processus « clean-room »

[English](CLEANROOM.md) · [Română](CLEANROOM.ro.md) · [Français](CLEANROOM.fr.md) · [Italiano](CLEANROOM.it.md) · [Español](CLEANROOM.es.md) · [Português](CLEANROOM.pt.md) · [Deutsch](CLEANROOM.de.md) · [Magyar](CLEANROOM.hu.md)

Une note brève et honnête sur la façon dont le moteur de `crates/engine` a été écrit, et sur ce que « clean-room » veut et ne veut pas dire ici.

## Les règles suivies

1. **Sources.** Uniquement des spécifications et articles publics : les structures des fichiers GGUF et safetensors, les champs `config.json` et `tokenizer.json` de Hugging Face, les descriptions publiées des structures de blocs de quantification et les articles derrière chaque technique (variantes d'attention, routage MoE, Mamba, RWKV, décodage spéculatif, H2O, StreamingLLM, SnapKV, TIES, DARE, filigrane à liste verte, HNSW, BM25…).
2. **Pas de coup d'œil.** Le code source de colibri, de llama.cpp et d'autres moteurs d'inférence n'a été ni lu ni copié pendant l'écriture. Quand une structure n'était pas claire, le travail s'est arrêté à la description publique, et l'incertitude est écrite dans le code et dans la documentation.
3. **Des oracles plutôt que de l'imitation.** Chaque noyau rapide est testé face à une version scalaire simple écrite d'abord ; chaque quantificateur face à des blocs calculés à la main et à des allers-retours ; les chemins répartis (maillage, diffusion, caches) face au chemin mono-machine, bit à bit là où c'est promis.
4. **Licences permissives uniquement.** Le moteur dépend de `serde_json` et `memmap2` et de rien d'autre ; le crate GPU ajoute `wgpu` derrière une option. Les licences ont été vérifiées avec `cargo-deny` à un commit antérieur ; les changements ultérieurs n'ont pas encore été revérifiés.
5. **L'entrée hostile est normale.** Chaque taille ou décalage lu dans un fichier est contrôlé face à la longueur réelle du fichier avant toute allocation ; des tests de type fuzzing fournissent des fichiers tronqués et corrompus et doivent recevoir une erreur, pas un plantage.

## Ce que nous n'affirmons pas

- Le clean-room est un processus, pas une garantie juridique. Le modèle qui a écrit le code, comme tout modèle, a été entraîné sur du texte public qui peut inclure du code open source, et nous ne pouvons pas exclure qu'il ait vu les projets ci-dessus. Nous ne les avons pas consultés et avons cherché à le rendre visible en gardant dans le code les oracles, les tests et la source de chaque formule.
- Plusieurs formules (Mamba, RWKV-6/7, MLA, extras imatrix et GGUF) ont été écrites d'après des descriptions publiées et leur souvenir, et vérifiées seulement pour la cohérence interne, faute de point de contrôle réel. Elles peuvent différer de la référence dans les détails ; considère-les comme non vérifiées tant que personne ne les a lancées sur de vrais modèles.
- La compatibilité avec les fichiers d'autres moteurs est un objectif, pas un fait vérifié (voir le tableau dans [ENGINE.fr.md](ENGINE.fr.md)).

Si tu trouves un endroit où le moteur reproduit de trop près le code de quelqu'un d'autre, ouvre une issue ; nous le réécrirons.
