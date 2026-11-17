# Feuille de route de l'interconnexion : d'une fenêtre de chat à un assistant capable

Objectif : relier le modèle local à tout ce qui multiplie ce qu'il peut faire, tandis que la personne garde la main. Chaque étape respecte trois règles : **local d'abord, rien ne sort sans un oui, chaque action peut être vue et annulée.**

## 0. Le cadre de sécurité (construit en premier, utilisé par tout le reste)
- **Niveaux de capacité** (voir `platforms.rs`) : Lire → Chercher → Écrire → Notifier → Agir. Une connexion est créée à un niveau et ne peut jamais le dépasser.
- **Carte d'approbation** pour chaque écriture, envoi et action : elle montre exactement ce qui va se passer, avec *Autoriser une fois / Autoriser pour cette automatisation / Refuser*. Lire n'exige pas de carte.
- **Contenu non fiable** (pages web, tickets, e-mails, documents) : encadré et étiqueté comme données. Un appel d'outil qui suit un contenu non fiable redemande toujours. C'est la principale défense contre l'injection de prompt.
- **Budgets et freins** : étapes, appels au modèle, minutes et octets par exécution ; une touche *Tout arrêter*.
- **Journal d'audit** de chaque requête sortie de l'ordinateur et de chaque action menée, dans un simple fichier.
- **Secrets** : les jetons restent en mémoire par défaut ; les enregistrer est un choix explicite.

## 1. Des outils pour le modèle (le multiplicateur)
Colibri expose l'appel d'outils sur la plupart des moteurs. Le programme offre au modèle une petite boîte à outils honnête : page web, recherche, fichier de dépôt, recherche dans un dossier, calculatrice, Python en bac à sable, requête sur tableau, recherche dans les notes, enregistrement de fichier, webhook. Le modèle demande, le programme vérifie le niveau et l'approbation, exécute et renvoie le résultat comme donnée.

## 2. MCP : une seule prise pour des milliers d'outils
Parler le Model Context Protocol en client (et garder `coli mcp` comme serveur). Tout serveur MCP (fichiers, bases de données, navigateurs, agendas, outils de design, appareils domestiques) devient disponible sous les mêmes règles d'approbation. C'est le plus grand bond de portée pour le moins de code.

## 3. Une mémoire et des connaissances qui grandissent avec toi
- Index local (BM25 maintenant, embeddings plus tard) sur dossiers, notes, discussions enregistrées, code, archives de courrier.
- Les réponses citent le passage utilisé ; « montre-moi où » l'ouvre.
- Bibliothèques hors ligne : Wikipédia via Kiwix, instantanés arXiv et Wikidata, manuels, lois.
- La *constellation* : des souvenirs comme étoiles reliées (étiquettes, personnes, projets) que tu peux voir et modifier.

## 4. Sens et voix (tout en local)
- Ouïe : whisper.cpp pour la dictée et les transcriptions de réunions.
- Parole : Piper ou la voix du système, selon la langue de l'interface.
- Vue : modèles qui acceptent des images (colibri indique lesquels), plus OCR (Tesseract) pour les numérisations et photos de papier, plus la caméra ou une capture d'écran à la demande.
- Création : Qwen-Image via colibri, ComfyUI ou Stable Diffusion pour les images ; outils audio et vidéo à mesure qu'ils apparaissent.

## 5. Plusieurs esprits
- **Duel** : A/B à l'aveugle avec Elo, pour savoir quel modèle est le meilleur *pour toi*.
- **Conseil** : plusieurs modèles répondent, un modèle juge fusionne et signale les désaccords.
- **Aiguilleur** : un petit modèle rapide pour les questions faciles, le gros pour les difficiles, choisi par un modèle de décision (colibri fournit Laya, GLiNER2.5-Decide, Clef).
- **Spécialistes** : code, traduction, vision, chacun choisi automatiquement, chacun remplaçable.

## 6. Exactitude : laisser le modèle se vérifier
- Exécuter en bac à sable le Python qu'il écrit et montrer la vraie sortie.
- Tests unitaires pour le code qu'il produit, lancés avant de le proposer.
- Auto-cohérence (demander deux fois, comparer), citations exigées pour les faits, un interrupteur « montre tes sources » et un petit banc d'évaluation avec intervalles de confiance (comme dans LocalMind).

## 7. Agents et automatisations
Des chaînes d'étapes enregistrées (lire, demander, vérifier, enregistrer, notifier) avec des déclencheurs : à la main, selon un horaire, quand un dossier change, quand un webhook arrive. Modèles : résumé du matin, brouillon de tri de la boîte de réception, veille d'un dépôt, notes de réunion, note de recherche, bilan hebdomadaire. Chaque étape affiche son niveau.

## 8. Appareils et monde physique (en dernier, et avec le plus de prudence)
Home Assistant, MQTT, appareils série, imprimantes 3D, caméras. Lecture seule par défaut ; actions via une liste d'autorisations ; jamais de serrures, d'alarmes ni rien de critique pour la sécurité.

## 9. Plusieurs ordinateurs, un seul cerveau
Partager un modèle sur le réseau domestique (un PC puissant le sert, un portable l'utilise), synchroniser prompts, mémoire et réglages entre machines (chiffré, de pair à pair, sans cloud) et répartir le travail sur plusieurs GPU là où le moteur le permet.

## 10. Rendre tout sans effort
Un guide au premier lancement qui choisit un modèle pour le matériel, un *mode simple* à cinq boutons, des connexions en un clic depuis le catalogue, des suggestions « que puis-je faire avec ça ? », un panneau visible « ce que le modèle voit en ce moment » et le tout en huit langues.

## Ordre de travail
1. Dossiers et gros fichiers, bibliothèque de prompts, styles de réponse (fait).
2. Accès en ligne, GitHub/GitLab/Gitea/pages web, le catalogue (en cours).
3. Cartes d'approbation, journal d'audit, budgets (le cadre de sécurité).
4. Automatisations avec webhooks et notifications.
5. Duel, conseil, aiguilleur.
6. Outils pour le modèle, puis client MCP.
7. Index et mémoire, voix et OCR.
8. Appareils, plusieurs machines.

Ce qui n'est *pas* promis : rien de tout cela n'a encore été testé avec de vrais moteurs colibri à appel d'outils, et chaque service externe change d'API avec le temps ; les entrées marquées « prévu » du catalogue sont des projets, pas des fonctions.
