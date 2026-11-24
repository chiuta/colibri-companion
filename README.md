# Colibri Companion

**English** · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

**Website: https://colibri-companion.chiuta.ro** — download, video tutorials, help.

A friendly desktop app for [colibri](https://github.com/JustVugg/colibri): pick a model that fits your computer, download it in one click, then chat — the model runs on your computer; no account, no cloud service.

## Run

```bash
cargo run --release -p colibri-companion            # the real app
cargo run --release -p colibri-companion -- --demo  # demo, no colibri needed
# options: --lang en|ro|fr|it|es|pt|de|hu   --version   --help
```

### The four steps

(1) finds Python and colibri (and offers to download colibri if it is missing), (2) lets you choose a model (those that do not fit your RAM are hidden; the recommended one is marked), (3) downloads and installs with progress, pause and resume, (4) starts the server and opens the chat.

Chat: streaming answers, instant Stop, Markdown, collapsible reasoning, saved conversations, style (Precise / Balanced / Creative), copy and regenerate; search your conversations (Ctrl/⌘+K), rename them (right-click), edit your last message, drop text files into the message, and save a conversation as a .md file. Themes: light, dark, high contrast.

**Documents and tools:** add files (**More → Documents**) and ask questions about them: the app searches them on your computer (keyword search, no cloud) and shows the passages it used under each answer. **More → MCP tools** connects MCP servers that you add and start yourself; the model can ask to use a tool, and every call needs your approval (early feature, depends on the model following the format).

Models: keep several on the computer, switch between them or add more from one window (**Models**), with the most capable model your computer can run suggested first (small-RAM computers stream experts from the SSD). Speed: CUDA for NVIDIA, Vulkan for other graphics chips, else the processor, chosen automatically. **Resources** shows processor, memory, graphics card and disk use live, plus the evidence that nothing leaves your computer (every open connection of the app and the engine, with the command to check it yourself). If colibri is missing it is downloaded for you (no git needed), and a once-a-day check (can be turned off) tells you when colibri or this app has a new version.

**Your computer, analysed:** in **Resources**, the app finds your graphics cards by itself and says in plain words whether the model fits in memory and why it may be slow (swap, other programs, an unused card). It only advises; it changes no setting on its own.

**Newer tools (all tested without a real model, see the website for what is and is not verified):** a speed range under each model before you download it (from your memory, drive and the catalogue; never a measurement) and the ROUTE_TRACE recording with an analysis of what would save drive reads; auto-tune of pinned memory and threads (**Resources**); **Light models for your memory** in the Models window (11 small models, filter for documented tool calling, copy the `ollama run` command); **Check citations** and **Scanned files (LegatorOCR)** in the Documents window (the second needs the separate LegatorOCR program, MIT licence; OCR text is not a verified transcription); and a **Usage** panel beside each answer (time, first token, tokens in and out, speed now, average and peak, context used, processor, memory, swap and graphics card of this computer; figures with ≈ are estimates). The **Bibliography** section of the Documents window converts BibTeX, RIS and CSL JSON and copies a reference list (APA, Harvard, IEEE); tested on generated entries only. The rest of the TEZAUR port (passage retrieval, provenance receipt, AI-use declaration) is still a library the window does not use. More in `docs/`, each file in all eight languages (`NAME.md`, `NAME.ro.md`, `NAME.fr.md`, `NAME.it.md`, `NAME.es.md`, `NAME.pt.md`, `NAME.de.md`, `NAME.hu.md`). New in alpha.7: a **Citations** section in the Documents window gives every document a citation (`[@name]`) that Check citations recognises, and copies the whole list as BibTeX, RIS, CSL JSON, APA, Harvard or IEEE; windows never grow past the screen (the content scrolls and the title bar with the close button stays reachable) and Esc closes the open window even from a text box.

## How it works

The app does not include colibri. It runs colibri's JSON commands (`setup --list --json`, `status --json`, `setup`, `start`, `stop`) and talks to its OpenAI-compatible gateway on `127.0.0.1`. It never installs system packages: if something is missing, it shows you the command to copy. Your data (settings, conversations) stays local in the user data folder; `COLIBRI_COMPANION_HOME` can change it.

## corbul: the clean-room engine (alpha)

Besides the desktop app, the repository contains an inference engine written from scratch in Rust, and `corbul`, its command line and local server (OpenAI, Ollama and Anthropic styles, with a web interface at `/`). It does not replace colibri in the app.

```bash
cargo build --release -p colibri-rs-cli      # target/release/corbul
corbul capabilities                          # what works and what does not
corbul generate ./model --prompt "Hello"     # a folder with safetensors or a .gguf file
```

**Honest status:** tested on tiny synthetic models and on tokenizer reference files only (12 model families, 46 of 46 cases equal to llama.cpp's reference data). No real downloaded model, no GPU and no Windows, macOS or ARM machine has run it yet; speeds are estimates. Dense and mixture-of-experts models are loaded; Mamba, RWKV and Jamba/Bamba run as plain continuation; DeepSeek-style attention, images, sound, embeddings and the GPU are not available in `corbul`. i-quants and BERT-style models are not supported.

Documentation (English for now, translations in preparation): [what it is and what it does not do](docs/engine/ENGINE.md) · [quick start and every flag](docs/engine/QUICKSTART.md) · [features that are different](docs/engine/INNOVATIONS.md) · [clean-room process](docs/engine/CLEANROOM.md) · [security changes](docs/engine/CHANGES-SECURITY.md).

## Licences

- This project: Apache-2.0 (`LICENSE`, `NOTICE`), like colibri.
- Dependencies: permissive licences only, checked with `cargo deny check licenses` (`deny.toml`); full list in `THIRD_PARTY_LICENSES.md`. Fonts: Inter, JetBrains Mono and Atkinson Hyperlegible (SIL OFL 1.1, in `licenses/`).
- Models have their own licences; the app shows them and asks for confirmation before downloading.

## Platforms

Linux (x86-64, ARM64; Wayland and X11), Windows 10/11 and macOS (Intel and Apple Silicon) — one code base, OpenGL rendering (works on old graphics cards and in virtual machines). `.github/workflows/build.yml` builds and tests every variant and produces ready-to-ship archives. On Linux, `packaging/linux/install.sh` adds the app to the menu. HiDPI scaling, Ctrl/⌘+N, Ctrl/⌘+K, Ctrl/⌘+, and Esc shortcuts, and screen-reader support (AccessKit). Automated builds and tests cover all three systems; manual testing on real Windows and macOS computers is still limited, and the screen-reader labels (AccessKit) have not been tried with real screen readers.

Older Linux (Debian 8+, Ubuntu 14.04+, CentOS 7+): extra builds `linux-x86_64-glibc2.17` and `linux-aarch64-glibc2.17` are linked against glibc 2.17; CI checks the symbol baseline and loads the program in Ubuntu 16.04. The windowing and graphics libraries still come from your system.

## Languages

English, Română, Français, Italiano, Español, Português, Deutsch, Magyar — chosen in Settings, with `--lang en|ro|fr|it|es|pt|de|hu`, or by the system language on first launch. Translations live in `crates/app/src/i18n.rs` (the key is the English text); a test checks that every interface text has all translations. They were made with AI help and have not been reviewed by native speakers.

## Development

```bash
cargo test --workspace   # includes an end-to-end test with a fake `coli` (tests/fixtures/fake_colibri)
cargo clippy --workspace --all-targets
```
