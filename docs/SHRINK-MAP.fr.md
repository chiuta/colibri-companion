# Paquets réduits et cartes de cache entre modèles

Deux extensions des paquets de contexte ([CONTEXT-PACKS.fr.md](CONTEXT-PACKS.fr.md), moteur corbul uniquement ; clean-room, code original).

## Paquets réduits

Quand les textes et la question ne tiennent pas ensemble dans la fenêtre du modèle, les paquets sont refusés. Avec la réduction, ils sont raccourcis : `corbul packs --shrink auto` (seulement autant que nécessaire) ou `--shrink R` (garde toujours environ la fraction `R`, de 0,1 à 1 des lignes de chaque paquet). Le champ `"shrink": "auto"` (ou un nombre) de `/v1/packs/complete` fait de même ; la réponse liste ce qui a été réduit (`"shrunk"`).

Comment : les derniers `--shrink-window` jetons d'un paquet (16 par défaut) sont relus et votent pour les lignes à garder (SnapKV) ; les premières lignes sont toujours gardées (l'attention s'y accumule) ; les valeurs restantes sont réajustées ([KV-FIT.fr.md](KV-FIT.fr.md) ; `--no-refit` le désactive). Les clés gardées sont ensuite tournées vers les positions de leurs nouveaux emplacements : le paquet devient *exactement le cache de son texte raccourci*, il peut être raccordé et réparé comme tout paquet, et avec une réparation complète la réponse égale un simple préremplissage du texte raccourci. Les premiers `--shrink-skip` textes (1 par défaut, d'ordinaire les consignes) restent entiers. La réduction se fait à l'assemblage ; l'étagère garde les paquets entiers.

Dans Colibri Companion : *Documents → Paquets de contexte (corbul)* → *Réduire les textes trop longs*.

Limites : de l'information est perdue, combien dépend du modèle et du texte (mesurez avec `corbul kvfit` ou la fenêtre *Measure cache shrinking…*) ; les modèles à couches à fenêtre glissante, les caches Q8 et ceux d'un accélérateur sont refusés ; si les textes ne tiennent pas même une fois tous les textes réductibles retirés, la requête échoue. Seulement testé sur de minuscules modèles aléatoires.

## Cartes de cache entre modèles différents

Un paquet écrit par un modèle est inutile à un autre. Si les deux partagent un tokeniseur, leurs lignes de cache pour les *mêmes jetons* peuvent être mises en relation : `corbul map-fit SRC DST --text FICHIER --out CARTE` lit le texte avec les deux modèles et ajuste, par couche, une carte linéaire (moindres carrés avec ridge, avec la moyenne) des lignes de clés et de valeurs de SRC vers celles de DST. Les clés sont comparées sans leur position (ramenées à 0 puis retournées à l'import), donc les tables rotatives et les tailles de tête peuvent différer ; les couches sont appariées par profondeur relative. La commande affiche l'erreur sur les lignes laissées hors de l'ajustement à côté de celle d'une simple moyenne, et avertit si la carte n'est pas meilleure.

`corbul map-check SRC DST --map CARTE --text FICHIER --question Q` coupe le texte en deux paquets écrits par SRC, les amène dans DST par la carte et répond de trois façons : le préremplissage complet de DST (la vérité), le cache cartographié avec la réparation habituelle (`--tolerance`, 0,02 par défaut) et le cache cartographié sans réparation. Elle affiche les lignes recalculées et l'écart de chaque réponse à la vérité.

Une ligne cartographiée n'est qu'une supposition : les lignes importées sont marquées approximatives et la réparation en recalcule autant qu'il faut (toutes avec une tolérance 0, ce qui égale un préremplissage simple). Une mauvaise carte coûte donc du temps, jamais de la justesse. Point d'entrée de la bibliothèque : `ctxpack::compose_mapped`.

Limites : les tokeniseurs doivent donner les mêmes identifiants (vérifié, sinon rien n'est ajusté) ; la carte vaut pour une seule paire de modèles (empreintes vérifiées) ; sa taille est `kd × kd' × couches × 8` octets ; l'ajustement garde les lignes de calibration en mémoire (`--rows`, 1024 par défaut). Une carte linéaire entre les caches de modèles sans lien peut être médiocre ; seule une mesure dit si elle vaut le coup pour votre paire. Pas encore dans le serveur ni le Companion, et rien n'a été mesuré sur une vraie paire de modèles.
