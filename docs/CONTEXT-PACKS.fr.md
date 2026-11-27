# Paquets de contexte

Réutilise le travail fait sur un texte lorsqu'il revient dans un autre prompt (moteur corbul uniquement ; clean-room, code original).

Un cache de préfixe ne sert que si le texte commun vient en premier. Un *paquet* est le cache KV d'un texte calculé seul et rangé sur une étagère (magasin de snapshots signé, plafonné en LRU).

`compose` met les paquets bout à bout : les lignes sont copiées et leurs clés tournées de la différence de position (exact), puis le début de chaque paquet suivant est recalculé par fenêtres croissantes jusqu'à ce que le changement mesuré passe sous une tolérance. La question est toujours calculée à neuf.

`tolerance = 0` recalcule tout et égale un prefill complet. Les modèles à fenêtre glissante, à cache évincé ou sur accélérateur sont refusés et retombent sur un prefill ordinaire.

Inexact tant qu'il n'est pas réparé : un état avec des lignes déplacées renvoie `is_exact() == false` et n'est jamais écrit sur disque.

Limites : le moteur colibri de JustVugg ne peut pas exporter ses caches ; cela ne marche donc qu'avec corbul. Testé sur de petits modèles synthétiques ; la qualité sur de vrais modèles n'est pas encore mesurée.
