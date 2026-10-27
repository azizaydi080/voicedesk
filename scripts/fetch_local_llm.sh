#!/usr/bin/env bash
# Downloads a pinned llama.cpp server build and a small multilingual GGUF model
# into .local/ so the whole stack can run offline without API keys.
set -euo pipefail

LLAMA_TAG="${LLAMA_TAG:-b11929}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-qwen2.5-1.5b-instruct-q4_k_m.gguf}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/.local"
mkdir -p "$DEST/models"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) ASSET="llama-${LLAMA_TAG}-bin-macos-arm64.tar.gz" ;;
  Darwin-x86_64) ASSET="llama-${LLAMA_TAG}-bin-macos-x64.tar.gz" ;;
  Linux-x86_64) ASSET="llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz" ;;
  *) echo "Unsupported platform; use the Docker llama.cpp image instead." >&2; exit 1 ;;
esac

if [ ! -x "$DEST/llama/llama-server" ]; then
  echo "Fetching $ASSET"
  curl -fsSL "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${ASSET}" -o "$DEST/llama.tar.gz"
  mkdir -p "$DEST/llama"
  tar -xzf "$DEST/llama.tar.gz" -C "$DEST/llama" --strip-components=1
  rm "$DEST/llama.tar.gz"
fi

if [ ! -f "$DEST/models/$MODEL_FILE" ]; then
  echo "Fetching $MODEL_REPO/$MODEL_FILE"
  curl -fsSL "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}" -o "$DEST/models/$MODEL_FILE"
fi
echo "Ready: $DEST/llama/llama-server, $DEST/models/$MODEL_FILE"
