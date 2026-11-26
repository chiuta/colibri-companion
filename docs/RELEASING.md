# Releasing

How a release is made (for the maintainer)

1. **Version.** Change `version` in the root `Cargo.toml` and `TAG` in `index.html` to the new number (for example `0.1.0-alpha.12` and `v0.1.0-alpha.12`), run `cargo update -w --offline`, add `docs/releases/<tag>.md`, commit and push to `main`.
2. **Build.** Wait for the `build` and `engine-ci` runs on that commit to be green. Then start the `build` workflow by hand (Actions, Run workflow) and type the tag, for example `v0.1.0-alpha.12`. It builds every platform and creates a **draft** release.
3. **Publish.** Open the draft, paste `docs/releases/<tag>.md` into the description, tick pre-release and press Publish. **Do not change the tag name in the draft**: the files inside were built as the version in `Cargo.toml`, and a different tag makes the program report one version while the release says another (so the update check offers an update forever). The workflow refuses a tag that already points at another commit.

## If a number was published by mistake

Do not reuse it. Delete that release and its tag by hand and publish the next number.
