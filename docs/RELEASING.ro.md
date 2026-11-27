# Publicarea unei versiuni

Cum se face o versiune (pentru administrator)

1. **Versiunea.** Schimbă `version` în `Cargo.toml` din rădăcină și `TAG` în `index.html` cu numărul nou (de exemplu `0.1.0-alpha.12` și `v0.1.0-alpha.12`), rulează `cargo update -w --offline`, adaugă `docs/releases/<tag>.md`, apoi commit și push pe `main`.
2. **Construirea.** Așteaptă ca rulările `build` și `engine-ci` pe acel commit să fie verzi. Apoi pornește manual workflow-ul `build` (Actions, Run workflow) și scrie tag-ul, de exemplu `v0.1.0-alpha.12`. Construiește toate platformele și creează un release **draft**.
3. **Publicarea.** Deschide draftul, lipește `docs/releases/<tag>.md` în descriere, bifează pre-release și apasă Publish. **Nu schimba numele tag-ului în draft**: fișierele dinăuntru au fost construite cu versiunea din `Cargo.toml`, iar un alt tag face ca programul să raporteze o versiune și release-ul alta (iar verificarea de actualizări oferă o actualizare la nesfârșit). Workflow-ul refuză un tag care arată deja spre alt commit.

## Dacă s-a publicat un număr din greșeală

Nu îl refolosi. Șterge manual acel release și tag-ul lui și publică numărul următor.

## Nu crea niciodată release-ul de mână

Workflow-ul `build` creează singur ciorna, cu toate fișierele atașate. Un release făcut de mână cu același tag (chiar și ciornă) e gol și face să eșueze pasul final. Dacă s-a întâmplat: șterge release-ul gol (păstrează tag-ul) și apasă **Re-run failed jobs** pe rularea `build`.
