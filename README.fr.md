# Colibri Companion

[English](README.md) · [Română](README.ro.md) · **Français** · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

**Site web : https://colibri-companion.chiuta.ro** — téléchargement, tutoriels vidéo, aide.

Une application de bureau conviviale pour [colibri](https://github.com/JustVugg/colibri) : choisis un modèle adapté à ton ordinateur, télécharge-le en un clic, puis discute — le modèle tourne sur ton ordinateur, sans compte ni service cloud.

## Lancement

```bash
cargo run --release -p colibri-companion            # l'application réelle
cargo run --release -p colibri-companion -- --demo  # démo, sans colibri
# options: --lang en|ro|fr|it|es|pt|de|hu   --version   --help
```

### Les quatre étapes

(1) trouve Python et colibri (et propose de télécharger colibri s'il manque), (2) te laisse choisir un modèle (ceux qui ne tiennent pas en RAM sont masqués ; le recommandé est marqué), (3) télécharge et installe avec progression, pause et reprise, (4) démarre le serveur et ouvre la conversation.

Conversation : réponses en flux, Arrêt immédiat, Markdown, raisonnement repliable, conversations enregistrées, style (Précis / Équilibré / Créatif), copie et régénération. Thèmes : clair, sombre, contraste élevé.

## Fonctionnement

L'application n'inclut pas colibri. Elle lance les commandes JSON de colibri (`setup --list --json`, `status --json`, `setup`, `start`, `stop`) et dialogue avec sa passerelle compatible OpenAI sur `127.0.0.1`. Elle n'installe jamais de paquets système : s'il manque quelque chose, elle affiche la commande à copier. Tes données (réglages, conversations) restent en local dans le dossier de données de l'utilisateur ; `COLIBRI_COMPANION_HOME` permet de le changer.

## Licences

- Ce projet : Apache-2.0 (`LICENSE`, `NOTICE`), comme colibri.
- Dépendances : licences permissives uniquement, vérifiées avec `cargo deny check licenses` (`deny.toml`) ; liste complète dans `THIRD_PARTY_LICENSES.md`. Polices : Inter et JetBrains Mono (SIL OFL 1.1, dans `licenses/`).
- Les modèles ont leurs propres licences ; l'application les affiche et demande confirmation avant le téléchargement.

## Plateformes

Linux (x86-64, ARM64 ; Wayland et X11), Windows 10/11 et macOS (Intel et Apple Silicon) — un seul code source, rendu OpenGL (fonctionne sur de vieilles cartes graphiques et en machine virtuelle). `.github/workflows/build.yml` construit et teste toutes les variantes et produit des archives prêtes à distribuer. Sous Linux, `packaging/linux/install.sh` ajoute l'application au menu. Mise à l'échelle HiDPI, raccourcis Ctrl/⌘+N, Ctrl/⌘+, et Échap, prise en charge des lecteurs d'écran (AccessKit). Les compilations et tests automatiques couvrent les trois systèmes ; les essais manuels sur de vrais ordinateurs Windows et macOS restent limités, et les étiquettes pour lecteurs d'écran (AccessKit) n'ont pas encore été essayées avec de vrais lecteurs.

Linux plus ancien (Debian 8+, Ubuntu 14.04+, CentOS 7+) : versions supplémentaires `linux-x86_64-glibc2.17` et `linux-aarch64-glibc2.17`, liées à glibc 2.17 ; la CI vérifie la base des symboles et charge le programme dans Ubuntu 16.04. Les bibliothèques de fenêtrage et de graphisme viennent toujours de ton système.

**Documents et outils :** ajoute des fichiers (**Plus → Documents**) et pose des questions : l'application les cherche sur ton ordinateur (mots-clés, sans cloud) et montre sous chaque réponse les passages utilisés. **Plus → Outils MCP** connecte des serveurs MCP que tu ajoutes et lances ; le modèle peut demander un outil et chaque appel exige ton accord (fonction précoce, dépend du respect du format par le modèle).

Modèles : garde-en plusieurs sur l'ordinateur, passe de l'un à l'autre ou ajoutes-en depuis une seule fenêtre (**Modèles**) ; le plus capable que ton ordinateur peut exécuter est proposé en premier (avec peu de RAM, les experts sont lus depuis le SSD). Vitesse : CUDA pour NVIDIA, Vulkan pour les autres puces graphiques, sinon le processeur, choisis automatiquement. **Ressources** montre en direct processeur, mémoire, carte graphique et disque, avec la preuve que rien ne quitte ton ordinateur (toutes les connexions ouvertes de l'application et du moteur, avec la commande pour vérifier toi-même). Si colibri manque, il est téléchargé pour toi (sans git), et une vérification quotidienne (désactivable) signale une nouvelle version de colibri ou de cette application.

**Ton ordinateur, analysé :** dans **Ressources**, l'application trouve seule tes cartes graphiques et dit simplement si le modèle tient en mémoire et pourquoi il peut être lent (swap, autres programmes, carte inutilisée). Elle conseille seulement ; elle ne change aucun réglage seule.

**Outils plus récents (tous testés sans vrai modèle ; le site dit ce qui est vérifié ou non) :** une fourchette de vitesse sous chaque modèle avant le téléchargement (d'après ta mémoire, ton disque et le catalogue ; jamais une mesure) et l'enregistrement ROUTE_TRACE avec une analyse de ce qui économiserait des lectures disque ; le réglage automatique de la mémoire épinglée et des fils (**Ressources**) ; **Modèles légers pour ta mémoire** dans la fenêtre Modèles (11 petits modèles, filtre pour l'appel d'outils documenté, copie de la commande `ollama run`) ; **Vérifier les citations** et **Fichiers numérisés (LegatorOCR)** dans la fenêtre Documents (le second demande le programme séparé LegatorOCR, licence MIT ; le texte OCR n'est pas une transcription vérifiée) ; et un panneau **Consommation** à côté de chaque réponse (temps, premier jeton, jetons en entrée et en sortie, vitesse actuelle, moyenne et maximale, contexte utilisé, processeur, mémoire, swap et carte graphique de cet ordinateur ; les chiffres avec ≈ sont estimés). La section **Bibliographie** de la fenêtre Documents convertit BibTeX, RIS et CSL JSON et copie une liste de références (APA, Harvard, IEEE) ; testée seulement sur des entrées générées. Le reste du portage de TEZAUR (recherche de passages, reçu de provenance, déclaration d'usage de l'IA) reste une bibliothèque que la fenêtre n'utilise pas. Plus dans `docs/`, chaque fichier dans les huit langues (`NOM.md`, `NOM.ro.md`, `NOM.fr.md`, `NOM.it.md`, `NOM.es.md`, `NOM.pt.md`, `NOM.de.md`, `NOM.hu.md`). Nouveau dans alpha.7 : une section **Citations** dans la fenêtre Documents donne à chaque document une citation (`[@nom]`) que la vérification des citations reconnaît, et copie toute la liste en BibTeX, RIS, CSL JSON, APA, Harvard ou IEEE ; les fenêtres ne dépassent plus l'écran (le contenu défile et la barre de titre avec le bouton de fermeture reste accessible) et Échap ferme la fenêtre ouverte, même depuis une zone de texte.

## Langues

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — choisies dans les Réglages, avec `--lang en|ro|fr|it|es|pt|de|hu`, ou selon la langue du système au premier lancement. Les traductions sont dans `crates/app/src/i18n.rs` (la clé est le texte anglais) ; un test vérifie que chaque texte de l'interface a toutes ses traductions. Elles ont été réalisées avec l'aide d'une IA et n'ont pas été relues par des locuteurs natifs.

## Développement

```bash
cargo test --workspace   # comprend un test de bout en bout avec un faux `coli` (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```

Nouveau : recherche dans les conversations (Ctrl/⌘+K), renommage (clic droit), modification du dernier message, fichiers texte glissés dans le message et enregistrement en .md.
