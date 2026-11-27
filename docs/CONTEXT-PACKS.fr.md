# Paquets de contexte

Réutilise le travail fait sur un texte lorsqu'il revient dans un autre prompt (moteur corbul uniquement ; clean-room, code original).

Un cache de préfixe ne sert que si le texte commun vient en premier. Un *paquet* est le cache KV d'un texte calculé seul et rangé sur une étagère (magasin de snapshots signé, plafonné en LRU).

`compose` met les paquets bout à bout : les lignes sont copiées et leurs clés tournées de la différence de position (exact), puis le début de chaque paquet suivant est recalculé par fenêtres croissantes jusqu'à ce que le changement mesuré passe sous une tolérance. La question est toujours calculée à neuf.

`tolerance = 0` recalcule tout et égale un prefill complet. Les modèles à fenêtre glissante, à cache évincé ou sur accélérateur sont refusés et retombent sur un prefill ordinaire.

Inexact tant qu'il n'est pas réparé : un état avec des lignes déplacées renvoie `is_exact() == false` et n'est jamais écrit sur disque.

Limites : le moteur colibri de JustVugg ne peut pas exporter ses caches ; cela ne marche donc qu'avec corbul. Testé sur de petits modèles synthétiques ; la qualité sur de vrais modèles n'est pas encore mesurée.

## Ligne de commande

`corbul packs MODEL --session DIR --pack FICHIER… --question TEXTE [--tolerance 0.02] [--probe 8] [--max-new 64] [--check]` range chaque fichier une seule fois, puis répond avec les paquets stockés, dans n'importe quel ordre. `--check` lance aussi un prefill complet et affiche l'écart de scores et la durée de chacun.

## Serveur

`corbul serve MODEL --packs DIR [--packs-cap 2g]` ajoute `POST /v1/packs/complete` avec `{"packs": [texte, …], "question": "…", "max_tokens": 64, "tolerance": 0.02, "probe": 8}`. La réponse contient `text`, `usage` et un objet `packs` (textes déjà rangés, lignes copiées, recalculées, neuves et économisées, changement mesuré à chaque jonction). Complétion de texte gloutonne, sans modèle de chat.
