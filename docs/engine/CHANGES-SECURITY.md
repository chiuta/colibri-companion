# Security and robustness changes

[English](CHANGES-SECURITY.md) · [Română](CHANGES-SECURITY.ro.md) · [Français](CHANGES-SECURITY.fr.md) · [Italiano](CHANGES-SECURITY.it.md) · [Español](CHANGES-SECURITY.es.md) · [Português](CHANGES-SECURITY.pt.md) · [Deutsch](CHANGES-SECURITY.de.md) · [Magyar](CHANGES-SECURITY.hu.md)

Other documents: [ENGINE.md](ENGINE.md) · [QUICKSTART.md](QUICKSTART.md)

This page lists what was found by adversarial review of the engine and `corbul`, and what was done about it, so that you can judge how far to trust a feature. It is a list of fixes and limits, not a security certification: nothing here was audited by a third party.

## Server

- **Cross-site requests.** State-changing calls check `Origin` and require a JSON `Content-Type`, so a web page you visit cannot make your browser post to a local `corbul serve`. Browsers are allowed only from the origins you list with `--cors`.
- **Admin routes.** `/api/create`, `/api/delete`, `/models/load` and `/models/unload` need the master token. API-key holders may use them only if you start the server with `--allow-admin`.
- **Secrets off the command line.** `--token-file`, `--key-file`, `--receipt-key-file` and the variables `CORBUL_TOKEN` and `CORBUL_MESH_KEY` keep secrets out of the process list. The plain `--token`, `--key` and `--receipt-key` still work and are visible to other users of the machine.
- **Resource limits.** `--max-kv` refuses requests whose KV cache would be too large (default 8g); `--max-heavy` bounds concurrent tokenize, template and verify requests (default 4); rate limits from `--keys-file` cover every route that costs computation; the chunked-body decoder is linear; `POST /v1/quotes/verify` has caps on text and source sizes.
- **Logs.** Prompt text is not logged unless `--log-prompts` is given; `--no-log` turns the log off.
- **Web interface.** The page itself is public (it holds no secrets); every call it makes needs the token.
- **Limit that remains:** the server speaks plain HTTP. Bind it to `127.0.0.1` (the default) or put TLS in front of it.

## Mesh

- Mutual HMAC-SHA256 handshake with a shared key of at least 8 characters, a MAC on every frame, caps on handshake and frame sizes.
- UDP discovery is opt-in (`--discovery`, `--discovery-port`) and answers are not larger than the probe, so a worker cannot be used as an amplifier.
- **Limit that remains:** traffic is authenticated but not encrypted; the key is shared by everyone in the mesh; tested on loopback only.

## Files, downloads and merges

- GGUF parsing takes the real file length (`parse_with_len`); tokenizer construction uses a trie with a cap on piece size; truncated and corrupt files return errors in the adversarial tests.
- Downloads: commit-SHA pinning, size ceilings, refusal of redirects to plain http, anonymous `wget` fallback (the token is never passed to it), SHA-256 checks.
- Merges do not follow symlinks and refuse to overwrite an output unless `--overwrite` is given; shapes and ranges are checked before writing.
- Image decoders enforce `Limits` before allocating and bound deflate expansion; the WAV reader has hostile-file limits.

## Decoding and verification

- Grammar, regular-expression and JSON-schema decoding have work budgets; the quote lock is linear with caps and reports `limit_hit` when a cap stops it.
- Receipts have size caps; snapshots can be sealed with a keyed HMAC; receipts and snapshots use SHA-256 chains.
- Correctness fixes found in review: stale probabilities after top-k/top-p (now renormalised), the `typ_p` alias, decimal commas in number handling, lookahead limits past the context, DeepSeek-V2 group score, quantised batch exactness (`exact_batch`).

## Not done or not verified

- No third-party audit, no fuzzing campaign beyond the repository's adversarial tests, no penetration test of `corbul serve`.
- `ToolStream::push` can rescan its buffer repeatedly (quadratic in the output length up to `--max-tokens`).
- The `GrammarCache` exists but is not used by the CLI yet.
- Dependency licences were last checked with `cargo-deny` at commit b496ba5.
- The CI workflows have never run; the ARM, macOS and Windows code paths have not been executed here.
