# Intégration de LegatorOCR

État : bibliothèque dans `crates/core/src/legatorocr.rs`, reliée à la fenêtre Documents par `crates/app/src/ocr_ui.rs` (section « Fichiers numérisés (LegatorOCR) »). Testée avec un vrai binaire `legatorocr 0.2.0-pre.65` sur une seule page générée de texte synthétique propre ; jamais sur la numérisation réelle d'un livre.

## Ce qu'elle fait

LegatorOCR (github.com/chiuta/LegatorOCR, **MIT**) transforme des numérisations en PDF ou en texte interrogeable. Colibri Companion (Apache-2.0) ne lie ni ne copie jamais son code ; il lance seulement le programme comme commande externe, ce que les deux licences permettent.

1. `locate` / `locate_in` : cherche un exécutable `legatorocr` (un dossier donné par l'utilisateur, le dossier de l'application, puis `PATH`). Système de fichiers seulement, aucun réseau.
2. `version` : lance `legatorocr --version` ; la réponse doit commencer par `legatorocr`, pour qu'un fichier sans rapport portant le même nom ne soit pas pris pour le programme.
3. `collect_inputs` : une image ou un PDF, ou les images et PDF directement dans un dossier, dans l'ordre des noms (2000 au plus).
4. `read_scan` : lance `legatorocr <pages> --ocr --lang <code> --txt <fichier> --format none --out-dir <dossier>` (plus `--clean-text` pour le roumain) via `proc::run_capture`, avec une limite de temps, puis relit le texte (32 Mo au plus).
5. `ingest_scan` : ajoute ce texte à une collection de documents (`rag::Library::add_document`) sous le nom `<fichier ou dossier> (OCR)`. Le modèle local répond alors à partir de la numérisation, et les passages sont cités sous ce nom comme n'importe quel autre document.

Sécurité : les arguments sont passés un par un (pas de shell) ; les entrées sont rendues absolues pour qu'un fichier nommé `-x.png` ne soit pas lu comme une option ; le code de langue est vérifié (`clean_lang`) ; la sortie est plafonnée ; une exécution qui ne finit pas est tuée ; les échecs reviennent avec les dernières lignes de la sortie d'erreur du programme ; rien ne démarre sans appel.

## Limites, en toute honnêteté

- Le texte est de l'OCR. Ce n'est pas une transcription vérifiée, et la qualité de l'OCR (et les langues qui marchent) dépend de LegatorOCR et des paquets de langue Tesseract installés : `legatorocr --doctor` les liste (sur la machine de test, les modèles roumain et hongrois manquaient).
- Les numéros de page sont perdus : toute la numérisation forme un seul document, donc une citation nomme la numérisation, pas la page.
- La recherche dans la bibliothèque se fait par mots-clés (BM25), comme pour tout document.
- La fenêtre n'affiche la section que si le programme est trouvé (cherché une fois, en arrière-plan, à l'ouverture de la fenêtre ; rien n'est téléchargé ni lancé tout seul). Sans lui, seuls un indice avec `github.com/chiuta/LegatorOCR` et un bouton « Chercher encore » apparaissent.
- L'exécution est un fil en arrière-plan avec un bouton Arrêter ; LegatorOCR ne donne aucun chiffre d'avancement, la fenêtre montre donc les secondes écoulées, pas un pourcentage. Fermer toute l'application pendant une exécution n'est pas géré spécialement.
- La liste des langues est celle des 8 langues de l'application, associées aux codes Tesseract ; qu'une langue marche dépend des données installées pour LegatorOCR (`legatorocr --doctor`). L'application ne le vérifie pas à l'avance : une langue manquante revient sous la forme du message d'erreur du programme lui-même.
- Les images de page déposées dans la fenêtre (pas les PDF) vont dans le champ de numérisation quand le programme est trouvé.

## À essayer

```
cargo test -p companion-core legatorocr
LEGATOROCR_BIN=/path/to/legatorocr LEGATOROCR_TEST_PAGE=/path/to/page.png \
  cargo test -p companion-core real_legatorocr -- --nocapture
```
