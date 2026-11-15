# Colibri Companion

[English](README.md) · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · **Deutsch** · [Magyar](README.hu.md)

Eine freundliche Desktop-App für [colibri](https://github.com/JustVugg/colibri): Wähle ein Modell, das zu deinem Computer passt, lade es mit einem Klick herunter und chatte — alles lokal, ohne Konto und ohne Cloud.

## Starten

```bash
cargo run --release -p colibri-companion            # die echte App
cargo run --release -p colibri-companion -- --demo  # Demo, ohne colibri
# Optionen: --lang en|ro|fr|it|es|pt|de|hu   --version   --help
```

### Die vier Schritte

(1) findet Python und colibri (und bietet an, colibri herunterzuladen, falls es fehlt), (2) lässt dich ein Modell wählen (was nicht in den RAM passt, ist ausgeblendet; das empfohlene ist markiert), (3) lädt herunter und installiert mit Fortschritt, Pause und Fortsetzen, (4) startet den Server und öffnet den Chat.

Chat: Antworten im Datenstrom, sofortiger Stopp, Markdown, einklappbares Denken, gespeicherte Unterhaltungen, Stil (Präzise / Ausgewogen / Kreativ), Kopieren und Neu generieren. Designs: hell, dunkel, hoher Kontrast.

## So funktioniert es

Die App enthält colibri nicht. Sie führt die JSON-Befehle von colibri aus (`setup --list --json`, `status --json`, `setup`, `start`, `stop`) und spricht mit dessen OpenAI-kompatiblem Gateway auf `127.0.0.1`. Sie installiert nie Systempakete: Fehlt etwas, zeigt sie dir den Befehl zum Kopieren. Deine Daten (Einstellungen, Unterhaltungen) bleiben lokal im Datenordner des Benutzers; `COLIBRI_COMPANION_HOME` kann ihn ändern.

## Lizenzen

- Dieses Projekt: Apache-2.0 (`LICENSE`, `NOTICE`), wie colibri.
- Abhängigkeiten: nur freizügige Lizenzen, geprüft mit `cargo deny check licenses` (`deny.toml`); vollständige Liste in `THIRD_PARTY_LICENSES.md`. Schriften: Inter und JetBrains Mono (SIL OFL 1.1, in `licenses/`).
- Modelle haben eigene Lizenzen; die App zeigt sie an und bittet vor dem Download um Bestätigung.

## Plattformen

Linux (x86-64, ARM64; Wayland und X11), Windows 10/11 und macOS (Intel und Apple Silicon) — eine Codebasis, OpenGL-Darstellung (läuft auch auf alten Grafikkarten und in virtuellen Maschinen). `.github/workflows/build.yml` baut und testet alle Varianten und erzeugt fertige Archive. Unter Linux fügt `packaging/linux/install.sh` die App dem Menü hinzu. HiDPI-Skalierung, Kürzel Strg/⌘+N, Strg/⌘+, und Esc sowie Unterstützung für Screenreader (AccessKit).

## Sprachen

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — gewählt in den Einstellungen, mit `--lang en|ro|fr|it|es|pt|de|hu` oder beim ersten Start nach der Systemsprache. Die Übersetzungen stehen in `crates/app/src/i18n.rs` (der Schlüssel ist der englische Text); ein Test prüft, dass jeder Text der Oberfläche alle Übersetzungen hat. Sie entstanden mit KI-Hilfe und wurden nicht von Muttersprachlern geprüft.

## Entwicklung

```bash
cargo test --workspace   # enthält einen Ende-zu-Ende-Test mit einem falschen `coli` (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```

Neu: Suche in Unterhaltungen (Strg/⌘+K), Umbenennen (Rechtsklick), letzte Nachricht bearbeiten, Textdateien in die Nachricht ziehen und als .md speichern.
