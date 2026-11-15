# Colibri Companion

**English** · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

**Website: https://colibri-companion.chiuta.ro** — download, video tutorials, help.

A friendly desktop app for [colibri](https://github.com/JustVugg/colibri): pick a model that fits your computer, download it in one click, then chat — all local, no account, no cloud.

## Run

```bash
cargo run --release -p colibri-companion            # the real app
cargo run --release -p colibri-companion -- --demo  # demo, no colibri needed
# options: --lang en|ro|fr|it|es|pt|de|hu   --version   --help
```

### The four steps

(1) finds Python and colibri (and offers to download colibri if it is missing), (2) lets you choose a model (those that do not fit your RAM are hidden; the recommended one is marked), (3) downloads and installs with progress, pause and resume, (4) starts the server and opens the chat.

Chat: streaming answers, instant Stop, Markdown, collapsible reasoning, saved conversations, style (Precise / Balanced / Creative), copy and regenerate; search your conversations (Ctrl/⌘+K), rename them (right-click), edit your last message, drop text files into the message, and save a conversation as a .md file. Themes: light, dark, high contrast.

## How it works

The app does not include colibri. It runs colibri's JSON commands (`setup --list --json`, `status --json`, `setup`, `start`, `stop`) and talks to its OpenAI-compatible gateway on `127.0.0.1`. It never installs system packages: if something is missing, it shows you the command to copy. Your data (settings, conversations) stays local in the user data folder; `COLIBRI_COMPANION_HOME` can change it.

## Licences

- This project: Apache-2.0 (`LICENSE`, `NOTICE`), like colibri.
- Dependencies: permissive licences only, checked with `cargo deny check licenses` (`deny.toml`); full list in `THIRD_PARTY_LICENSES.md`. Fonts: Inter and JetBrains Mono (SIL OFL 1.1, in `licenses/`).
- Models have their own licences; the app shows them and asks for confirmation before downloading.

## Platforms

Linux (x86-64, ARM64; Wayland and X11), Windows 10/11 and macOS (Intel and Apple Silicon) — one code base, OpenGL rendering (works on old graphics cards and in virtual machines). `.github/workflows/build.yml` builds and tests every variant and produces ready-to-ship archives. On Linux, `packaging/linux/install.sh` adds the app to the menu. HiDPI scaling, Ctrl/⌘+N, Ctrl/⌘+K, Ctrl/⌘+, and Esc shortcuts, and screen-reader support (AccessKit).

## Languages

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — chosen in Settings, with `--lang en|ro|fr|it|es|pt|de|hu`, or by the system language on first launch. Translations live in `crates/app/src/i18n.rs` (the key is the English text); a test checks that every interface text has all translations. They were made with AI help and have not been reviewed by native speakers.

## Development

```bash
cargo test --workspace   # includes an end-to-end test with a fake `coli` (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```
