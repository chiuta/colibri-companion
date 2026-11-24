# Packaging the engine (`corbul`)

This note covers the command-line front end of the clean-room engine. The desktop application has its own packaging in `build.yml` and `packaging/linux/`.

## What is built

| Workflow | When | What it does |
|---|---|---|
| `.github/workflows/engine-ci.yml` | push to `main`, pull requests, manual | licence check, clippy with `-D warnings`, MSRV 1.89 build, debug tests on six runners, kernel tests per SIMD variant, release-mode adversarial/fuzz suites, optional real-checkpoint validation |
| `.github/workflows/engine-release.yml` | tag `engine-v<version>`, manual | builds `corbul` for every platform, packages it, writes `SHA256SUMS`, creates a **draft** pre-release |

Neither workflow had been run when it was written. The first run is the first test; expect to fix small things.

## Releasing

1. Make sure `version` in the root `Cargo.toml` is the version you want (the engine crates share the workspace version).
2. Push a tag named `engine-v<that version>`, for example `engine-v0.1.0-alpha.8`. A different tag, or one that disagrees with `Cargo.toml`, stops the release job.
3. Check the draft release (archives, `SHA256SUMS`), then press Publish yourself. The workflow never publishes.

Plain `v…` tags belong to the desktop application (`build.yml`). Both can exist for the same commit.

## Archives

| Name | Contents |
|---|---|
| `corbul-<version>-linux-x86_64.tar.gz`, `…-linux-aarch64.tar.gz` | glibc builds from Ubuntu 26.04, so they need a recent glibc |
| `…-linux-x86_64-musl.tar.gz`, `…-linux-aarch64-musl.tar.gz` | statically linked (experimental) |
| `…-macos-universal.tar.gz` | one binary for Apple Silicon and Intel |
| `…-windows-x86_64.zip`, `…-windows-arm64.zip` | MSVC builds (ARM64 is experimental) |

Each archive holds `corbul`, `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md`, `licenses/`, `README.md`, `docs/engine/` and this note. Nothing is built with `target-cpu=native`: the engine picks its SIMD kernels (AVX2, AVX-512, NEON, dot-product) at run time, so one binary per platform is enough.

Not done: code signing and notarisation (macOS, Windows), `.deb`/`.rpm`/Homebrew/winget packages, a glibc-2.17 build like the one the desktop application has. macOS and Windows binaries will show the usual unsigned-program warnings.

## Real-checkpoint validation

The manual `engine-ci` run with `real_model` ticked downloads one small public GGUF file (default `stories260K.gguf` from `ggml-org/tiny-llamas`; repository and file are inputs), then runs `corbul doctor`, `inspect`, `plan`, a reproducible `generate` with a signed receipt, and `replay` of that receipt under each forced SIMD variant (`ENGINE_SIMD=scalar|avx2+fma|avx512|neon|neon+dotprod`). A variant the CPU does not have falls back to the default one, so read the `doctor` output in `replay.log` to see what really ran. Logs and the receipt are uploaded as an artifact per platform. This is the first test of the engine against a file it was not generated for; until it has run and been read, every claim about real models stays unverified.

## Things to know

* The engine crates are not formatted with rustfmt. The `fmt` job in `engine-ci` only reports. `build.yml` runs `cargo fmt --all --check` for the whole workspace, which will fail until the engine crates are formatted or excluded there.
* Runner labels used (checked against GitHub's documentation): `ubuntu-26.04`, `ubuntu-26.04-arm`, `macos-latest` (Apple Silicon), `macos-15-intel`, `windows-latest`, `windows-11-arm`. `macos-13` no longer appears in the documented list and is not used.
* Node.js is not used by these workflows, so the "Node ≥ 24" rule has nothing to apply to; the actions themselves run on the runner's own Node.
