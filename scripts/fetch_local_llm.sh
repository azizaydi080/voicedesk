#!/usr/bin/env bash
# Downloads the default local LLM (a small multilingual GGUF) into .local/models so the
# stack runs offline without API keys. Override MODEL_REPO / MODEL_FILE for another model,
# e.g. MODEL_REPO=Qwen/Qwen2.5-3B-Instruct-GGUF MODEL_FILE=qwen2.5-3b-instruct-q4_k_m.gguf
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-qwen2.5-1.5b-instruct-q4_k_m.gguf}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/.local/models"
mkdir -p "$DEST"

if [ ! -f "$DEST/$MODEL_FILE" ]; then
  echo "Fetching $MODEL_REPO/$MODEL_FILE"
  curl -fsSL "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}" -o "$DEST/$MODEL_FILE.part"
  mv "$DEST/$MODEL_FILE.part" "$DEST/$MODEL_FILE"
fi
echo "Ready: $DEST/$MODEL_FILE"
