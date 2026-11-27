# Réduire le cache et réajuster les valeurs

`corbul kvfit MODEL --text FILE --question Q` mesure ce que coûte la conservation d'une fraction seulement des lignes du cache d'un texte (choisies par SnapKV), avec et sans **réajustement des valeurs** conservées.

Les `--window` derniers jetons du texte sont enregistrés comme requêtes de sonde. Les valeurs des lignes conservées sont ensuite déplacées le moins possible pour que l'attention sur elles donne, sur les sondes, ce que donnait l'attention sur toutes les lignes (moindres carrés régularisés, forme duale, `crates/engine/src/kvfit.rs`). L'ajustement ne peut jamais être pire que les valeurs simples *sur les sondes* ; qu'il aide sur une question non sondée dépend du modèle et du texte, et seule cette commande, lancée sur votre modèle, vous le dit.

Options : `--budget 0.25` (fraction de lignes conservées), `--window 16`, `--ridge 0.01`, `--max-new 32`. Elle affiche l'erreur d'ajustement sur les sondes, la plus grande différence des premiers scores avec le cache complet, si le premier jeton concorde et combien des premiers jetons de la réponse concordent.

Limites : valeurs seulement (les clés portent la position et restent ; le poids des lignes retirées n'est pas modélisé), les couches à fenêtre glissante restent telles quelles, l'ajustement coûte du temps proportionnel à lignes × sondes par tête. Pas encore relié aux paquets de contexte (un paquet réduit ne peut pas être raccordé). Rien de mesuré sur un vrai modèle.

Les paquets de contexte peuvent aussi être rangés à moitié taille : `corbul packs --pack-kv f16` et `corbul serve --packs-kv f16` (arrondi F16 sur chaque ligne), et compressés sans perte : `--pack-compress`, `--packs-compress`.

Dans Colibri Companion : *Documents → Paquets de contexte (corbul)* propose les options F16 et compression (appliquées au prochain démarrage de corbul) et *Mesurer la réduction du cache…*, qui lance cette commande sur le modèle que corbul exécute (il charge le modèle une seconde fois).
