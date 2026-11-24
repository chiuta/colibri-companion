#!/usr/bin/env bash
# Download the llama.cpp vocab-only GGUF files and their .inp/.out reference pairs (MIT licence)
# used by crates/engine/tests/tokenizer_real.rs.
#   tools/fetch_vocab.sh [DIR] [REF]      DIR default: ./vocab   REF default: master
# Then: ENGINE_VOCAB_DIR=DIR ENGINE_REQUIRE_VOCAB=1 cargo test -p companion-engine --test tokenizer_real
set -euo pipefail
dir="${1:-vocab}"
ref="${2:-master}"
base="https://raw.githubusercontent.com/ggml-org/llama.cpp/${ref}/models"
families="llama-bpe qwen2 gpt-2 llama-spm deepseek-llm deepseek-coder phi-3 falcon starcoder mpt command-r refact"
mkdir -p "$dir"
fail=0
for f in $families; do
  for ext in gguf gguf.inp gguf.out; do
    name="ggml-vocab-$f.$ext"
    if [ -s "$dir/$name" ]; then continue; fi
    if ! curl -fsSL --retry 3 --retry-delay 2 --proto '=https' -o "$dir/$name.part" "$base/$name"; then
      echo "FAILED: $base/$name" >&2; rm -f "$dir/$name.part"; fail=1; continue
    fi
    [ -s "$dir/$name.part" ] || { echo "EMPTY: $name" >&2; rm -f "$dir/$name.part"; fail=1; continue; }
    mv "$dir/$name.part" "$dir/$name"
  done
done
echo "fetched into $dir: $(ls "$dir" | wc -l) files"
exit "$fail"
