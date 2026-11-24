# Changements de sécurité et de robustesse

[English](CHANGES-SECURITY.md) · [Română](CHANGES-SECURITY.ro.md) · [Français](CHANGES-SECURITY.fr.md) · [Italiano](CHANGES-SECURITY.it.md) · [Español](CHANGES-SECURITY.es.md) · [Português](CHANGES-SECURITY.pt.md) · [Deutsch](CHANGES-SECURITY.de.md) · [Magyar](CHANGES-SECURITY.hu.md)

Autres documents : [ENGINE.fr.md](ENGINE.fr.md) · [QUICKSTART.fr.md](QUICKSTART.fr.md)

Cette page liste ce que la revue adversariale du moteur et de `corbul` a trouvé, et ce qui a été fait, pour que tu puisses juger jusqu'où faire confiance à une fonction. C'est une liste de corrections et de limites, pas une certification de sécurité : rien ici n'a été audité par un tiers.

## Serveur

- **Requêtes inter-sites.** Les appels qui modifient l'état vérifient `Origin` et exigent un `Content-Type` JSON, de sorte qu'une page web que tu visites ne puisse pas faire poster ton navigateur vers un `corbul serve` local. Les navigateurs ne sont autorisés que depuis les origines que tu indiques avec `--cors`.
- **Routes d'administration.** `/api/create`, `/api/delete`, `/models/load` et `/models/unload` exigent le jeton maître. Les détenteurs de clés d'API ne peuvent les utiliser que si tu démarres le serveur avec `--allow-admin`.
- **Secrets hors de la ligne de commande.** `--token-file`, `--key-file`, `--receipt-key-file` et les variables `CORBUL_TOKEN` et `CORBUL_MESH_KEY` gardent les secrets hors de la liste des processus. Les options simples `--token`, `--key` et `--receipt-key` fonctionnent toujours et sont visibles des autres utilisateurs de la machine.
- **Limites de ressources.** `--max-kv` refuse les requêtes dont le cache KV serait trop grand (8g par défaut) ; `--max-heavy` borne les requêtes simultanées de tokenisation, de modèle et de vérification (4 par défaut) ; les limites de débit de `--keys-file` couvrent chaque route qui coûte du calcul ; le décodeur de corps en morceaux est linéaire ; `POST /v1/quotes/verify` plafonne la taille du texte et des sources.
- **Journaux.** Le texte des prompts n'est pas journalisé sauf avec `--log-prompts` ; `--no-log` désactive le journal.
- **Interface web.** La page elle-même est publique (elle ne contient aucun secret) ; chaque appel qu'elle fait exige le jeton.
- **Limite qui subsiste :** le serveur parle en http en clair. Lie-le à `127.0.0.1` (la valeur par défaut) ou place du TLS devant lui.

## Mesh

- Poignée de main HMAC-SHA256 mutuelle avec une clé partagée d'au moins 8 caractères, un MAC sur chaque trame, des plafonds sur la taille de la poignée de main et des trames.
- La découverte UDP est optionnelle (`--discovery`, `--discovery-port`) et les réponses ne sont pas plus grandes que la sonde, de sorte qu'un worker ne peut pas servir d'amplificateur.
- **Limite qui subsiste :** le trafic est authentifié mais non chiffré ; la clé est partagée par tous les membres du mesh ; testé uniquement en boucle locale.

## Fichiers, téléchargements et fusions

- L'analyse GGUF prend la vraie longueur du fichier (`parse_with_len`) ; la construction du tokenizer utilise un trie avec un plafond sur la taille des pièces ; les fichiers tronqués et corrompus renvoient des erreurs dans les tests adversariaux.
- Téléchargements : épinglage sur un SHA de commit, plafonds de taille, refus des redirections vers du http en clair, repli anonyme sur `wget` (le jeton ne lui est jamais transmis), vérifications SHA-256.
- Les fusions ne suivent pas les liens symboliques et refusent d'écraser une sortie sauf avec `--overwrite` ; les formes et les plages sont vérifiées avant l'écriture.
- Les décodeurs d'image appliquent `Limits` avant d'allouer et bornent l'expansion deflate ; le lecteur WAV a des limites pour les fichiers hostiles.

## Décodage et vérification

- Le décodage par grammaire, expression régulière et schéma JSON a des budgets de travail ; le verrou de citations est linéaire avec des plafonds et signale `limit_hit` quand un plafond l'arrête.
- Les reçus ont des plafonds de taille ; les instantanés peuvent être scellés avec un HMAC à clé ; reçus et instantanés utilisent des chaînes SHA-256.
- Corrections de justesse trouvées en revue : probabilités périmées après top-k/top-p (désormais renormalisées), l'alias `typ_p`, virgules décimales dans le traitement des nombres, limites de lookahead au-delà du contexte, score de groupe DeepSeek-V2, exactitude des lots quantifiés (`exact_batch`).

## Non fait ou non vérifié

- Pas d'audit par un tiers, pas de campagne de fuzzing au-delà des tests adversariaux du dépôt, pas de test d'intrusion de `corbul serve`.
- `ToolStream::push` peut réanalyser son tampon à répétition (quadratique en la longueur de la sortie, jusqu'à `--max-tokens`).
- `GrammarCache` existe mais la CLI ne l'utilise pas encore.
- Les licences des dépendances ont été vérifiées pour la dernière fois avec `cargo-deny` au commit b496ba5.
- Les workflows CI n'ont jamais tourné ; les chemins de code ARM, macOS et Windows n'ont pas été exécutés ici.
