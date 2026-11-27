# Publier une version

Comment faire une version (pour le mainteneur)

1. **Version.** Change `version` dans le `Cargo.toml` racine et `TAG` dans `index.html` pour le nouveau numéro (par exemple `0.1.0-alpha.12` et `v0.1.0-alpha.12`), lance `cargo update -w --offline`, ajoute `docs/releases/<tag>.md`, puis commit et push sur `main`.
2. **Construction.** Attends que les exécutions `build` et `engine-ci` de ce commit soient vertes. Lance ensuite à la main le workflow `build` (Actions, Run workflow) et saisis le tag, par exemple `v0.1.0-alpha.12`. Il construit toutes les plateformes et crée une version **brouillon**.
3. **Publication.** Ouvre le brouillon, colle `docs/releases/<tag>.md` dans la description, coche pré-version et appuie sur Publish. **Ne change pas le nom du tag dans le brouillon** : les fichiers ont été construits avec la version de `Cargo.toml`, et un autre tag fait que le programme annonce une version et la publication une autre (la vérification des mises à jour en propose alors une sans fin). Le workflow refuse un tag qui pointe déjà vers un autre commit.

## Si un numéro a été publié par erreur

Ne le réutilise pas. Supprime à la main cette version et son tag, puis publie le numéro suivant.

## Ne créez jamais la version à la main

Le workflow `build` crée lui-même le brouillon, avec tous les fichiers joints. Une version créée à la main avec le même tag (même en brouillon) est vide et fait échouer l'étape finale. Si cela arrive : supprimez la version vide (gardez le tag) et utilisez **Re-run failed jobs** sur l'exécution `build`.
