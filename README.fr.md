# Colibri Companion

[English](README.md) · [Română](README.ro.md) · **Français** · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

Une application de bureau conviviale pour [colibri](https://github.com/JustVugg/colibri) : choisissez un modèle adapté à votre ordinateur, téléchargez-le en un clic, puis discutez — tout en local, sans compte ni cloud.

## Lancement

```bash
cargo run --release -p colibri-companion            # l'application réelle
cargo run --release -p colibri-companion -- --demo  # démo, sans colibri
# options: --lang en|ro|fr|it|es|pt|de|hu   --version   --help
```

### Les quatre étapes

(1) trouve Python et colibri (et propose de télécharger colibri s'il manque), (2) vous laisse choisir un modèle (ceux qui ne tiennent pas en RAM sont masqués ; le recommandé est marqué), (3) télécharge et installe avec progression, pause et reprise, (4) démarre le serveur et ouvre la conversation.

Conversation : réponses en flux, Arrêt immédiat, Markdown, raisonnement repliable, conversations enregistrées, style (Précis / Équilibré / Créatif), copie et régénération. Thèmes : clair, sombre, contraste élevé.

## Fonctionnement

L'application n'inclut pas colibri. Elle lance les commandes JSON de colibri (`setup --list --json`, `status --json`, `setup`, `start`, `stop`) et dialogue avec sa passerelle compatible OpenAI sur `127.0.0.1`. Elle n'installe jamais de paquets système : s'il manque quelque chose, elle affiche la commande à copier. Vos données (réglages, conversations) restent en local dans le dossier de données de l'utilisateur ; `COLIBRI_COMPANION_HOME` permet de le changer.

## Licences

- Ce projet : Apache-2.0 (`LICENSE`, `NOTICE`), comme colibri.
- Dépendances : licences permissives uniquement, vérifiées avec `cargo deny check licenses` (`deny.toml`) ; liste complète dans `THIRD_PARTY_LICENSES.md`. Polices : Inter et JetBrains Mono (SIL OFL 1.1, dans `licenses/`).
- Les modèles ont leurs propres licences ; l'application les affiche et demande confirmation avant le téléchargement.

## Plateformes

Linux (x86-64, ARM64 ; Wayland et X11), Windows 10/11 et macOS (Intel et Apple Silicon) — un seul code source, rendu OpenGL (fonctionne sur de vieilles cartes graphiques et en machine virtuelle). `.github/workflows/build.yml` construit et teste toutes les variantes et produit des archives prêtes à distribuer. Sous Linux, `packaging/linux/install.sh` ajoute l'application au menu. Mise à l'échelle HiDPI, raccourcis Ctrl/⌘+N, Ctrl/⌘+, et Échap, prise en charge des lecteurs d'écran (AccessKit).

**Documents et outils :** ajoutez des fichiers (**Plus → Documents**) et posez des questions : l'application les cherche sur votre ordinateur (mots-clés, sans cloud) et montre sous chaque réponse les passages utilisés. **Plus → Outils MCP** connecte des serveurs MCP que vous ajoutez et lancez ; le modèle peut demander un outil et chaque appel exige votre accord (fonction précoce, dépend du respect du format par le modèle).

Modèles : gardez-en plusieurs sur l'ordinateur, passez de l'un à l'autre ou ajoutez-en depuis une seule fenêtre (**Modèles**) ; le plus capable que votre ordinateur peut exécuter est proposé en premier (avec peu de RAM, les experts sont lus depuis le SSD). Vitesse : CUDA pour NVIDIA, Vulkan pour les autres puces graphiques, sinon le processeur, choisis automatiquement. **Ressources** montre en direct processeur, mémoire, carte graphique et disque, avec la preuve que rien ne quitte votre ordinateur (toutes les connexions ouvertes de l'application et du moteur, avec la commande pour vérifier vous-même). Si colibri manque, il est téléchargé pour vous (sans git), et une vérification quotidienne (désactivable) signale une nouvelle version de colibri ou de cette application.

## Langues

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — choisies dans les Réglages, avec `--lang en|ro|fr|it|es|pt|de|hu`, ou selon la langue du système au premier lancement. Les traductions sont dans `crates/app/src/i18n.rs` (la clé est le texte anglais) ; un test vérifie que chaque texte de l'interface a toutes ses traductions. Elles ont été réalisées avec l'aide d'une IA et n'ont pas été relues par des locuteurs natifs.

## Développement

```bash
cargo test --workspace   # comprend un test de bout en bout avec un faux `coli` (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```

Nouveau : recherche dans les conversations (Ctrl/⌘+K), renommage (clic droit), modification du dernier message, fichiers texte glissés dans le message et enregistrement en .md.
