# Running GGUF files with corbul

Colibri imports only the model families it implements (Qwen3.x, DeepSeek, GLM, Mistral-Large, MiMo and a few more). A GGUF of another family, Gemma for one, fails to import. Colibri Companion then tries its own engine, **corbul**, which reads the `.gguf` directly.

- In *My models → Load GGUF models from folder* every file has a *Run with corbul* button (shown when `corbul` sits next to the app or on the PATH; the release archives include it).
- When an import fails, the message names the file, its declared architecture and the last log lines, and corbul is tried automatically.
- corbul listens on the loopback only, on a free port, with the context packs shelf in the app's data folder (see CONTEXT-PACKS). It stops when you stop the model, start another or close the app. Its log is shown by *Show the log*.
- Limits: corbul is slower than colibri on very large models, has no expert streaming, and supports the families its own engine implements; if it cannot load the file, the message says why.
