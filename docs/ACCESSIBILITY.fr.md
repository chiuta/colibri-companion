# Accessibilité

Colibri Companion veut être utilisable par le plus grand nombre. Tout ce qui suit se trouve sous **Accessibilité**, dans la barre du haut (ou avec `Ctrl+Maj+A`).

## Ce qui existe

| Besoin | Ce que fait l'application |
| --- | --- |
| Basse vision | Taille du texte de 80 % à 300 % ; un aspect à fort contraste ; la police Atkinson Hyperlegible (SIL OFL) ; plus d'espace entre les lignes et les boutons ; un contour épais autour de l'élément où se trouve le clavier |
| Cécité | L'interface est exposée aux lecteurs d'écran via AccessKit ; chaque réponse peut être lue à voix haute avec la voix de l'ordinateur (bouton, `Ctrl+Maj+S`, ou automatiquement) |
| Daltonisme | Une palette bleu/orange à la place du vert/rouge dans les trois aspects |
| Mains et mouvement | Boutons jusqu'à 64 points de haut ; toute l'application fonctionne au clavier seul ; aucun glisser ni geste précis n'est jamais exigé |
| Audition | L'application ne fait jamais de bruit et ne te demande jamais d'écouter ; la barre des tâches peut clignoter quand une réponse est prête Échap ferme la fenêtre ouverte, même depuis une zone de texte, et aucune fenêtre ne dépasse l'écran. |
| Dyslexie, attention, charge cognitive | Police lisible, lignes plus aérées, aucune animation, menus simples et un style de réponse en *langage simple* (phrases courtes, mots de tous les jours) |

Sept préréglages (basse vision, cécité, mouvements difficiles à contrôler, dyslexie, lecture plus facile, daltonisme, audition) règlent des ensembles raisonnables en un clic ; chaque option peut ensuite être changée séparément.

## Comment c'est vérifié

* Chaque paire couleur de texte / couleur de fond utilisée par l'interface, dans les trois aspects et avec la palette pour daltoniens activée ou non, doit atteindre un contraste de 4,5 : 1 (WCAG AA pour le texte normal). C'est un test unitaire (`theme::tests::every_look_is_readable`) : un changement de couleur qui le casse fait échouer la compilation.
* Les couleurs « bon/mauvais » de la palette pour daltoniens doivent aussi différer en luminosité, pour rester distinguables en niveaux de gris.
* Le texte envoyé au programme de synthèse vocale passe par l'entrée standard ou une variable d'environnement, jamais dans la ligne de commande : une réponse ne peut donc pas être prise pour une option (testé unitairement).

## Ce qui n'a **pas** été vérifié

* L'application n'a pas été essayée avec un vrai lecteur d'écran (NVDA, JAWS, Narrator, VoiceOver, Orca). L'arbre d'accessibilité vient d'egui/AccessKit ; les boutons à icône seule peuvent être mal annoncés. Les retours sont bienvenus.
* La lecture à voix haute utilise `espeak-ng`/`espeak`/`spd-say` sous Linux, `say` sous macOS et System.Speech via PowerShell sous Windows. Seule la construction de la commande est testée ; la voix elle-même n'a pas été entendue ici. Sous macOS et Windows, la langue de la voix est celle choisie dans le système.
* La *saisie* vocale est laissée à la dictée du système d'exploitation (Windows `Win+H`, macOS `Fn` deux fois). La dictée hors ligne dans l'application (whisper.cpp) est prévue dans `ROADMAP-INTERCONNECT.md`.
* Le balayage par contacteur et le suivi du regard ne sont pas implémentés ; les outils du système d'exploitation fonctionnent par-dessus la prise en charge du clavier.
