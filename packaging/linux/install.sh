#!/bin/sh
# Installs the binary, menu entry and icon for the current user (no root needed).
set -e
BIN="${1:-target/release/colibri-companion}"
install -Dm755 "$BIN" "$HOME/.local/bin/colibri-companion"
install -Dm644 packaging/linux/colibri-companion.desktop "$HOME/.local/share/applications/colibri-companion.desktop"
install -Dm644 packaging/icon-256.png "$HOME/.local/share/icons/hicolor/256x256/apps/colibri-companion.png"
echo "Installed. Find “Colibri Companion” in your application menu."
