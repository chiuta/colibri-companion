# Lancer des fichiers GGUF avec corbul

Colibri n'importe que les familles de modèles qu'il implémente (Qwen3.x, DeepSeek, GLM, Mistral-Large, MiMo et quelques autres). Un GGUF d'une autre famille, Gemma par exemple, échoue à l'import. Colibri Companion essaie alors son propre moteur, **corbul**, qui lit directement le `.gguf`.

- Dans *Mes modèles → Charger des modèles GGUF depuis un dossier*, chaque fichier a un bouton *Lancer avec corbul* (affiché quand `corbul` est à côté de l'application ou dans le PATH ; les archives de version l'incluent).
- Quand un import échoue, le message nomme le fichier, son architecture déclarée et les dernières lignes du journal, et corbul est essayé automatiquement.
- corbul n'écoute que sur la boucle locale, sur un port libre, avec l'étagère des paquets de contexte dans le dossier de données de l'application (voir CONTEXT-PACKS). Il s'arrête quand vous arrêtez le modèle, en lancez un autre ou fermez l'application. Son journal s'affiche avec *Afficher le journal*.
- Limites : corbul est plus lent que colibri sur de très grands modèles, sans diffusion d'experts depuis le disque, et ne gère que les familles de son propre moteur ; s'il ne peut pas charger le fichier, le message dit pourquoi.
