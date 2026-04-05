#!/bin/sh
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_ROOT/.venv-prune"

if [ ! -f "$VENV/bin/python3" ]; then
  echo "Creating prune venv at $VENV..."
  python3 -m venv "$VENV"
fi

echo "Installing prune dependencies..."
"$VENV/bin/pip" install --quiet safetensors huggingface_hub transformers torch
echo "Prune venv ready at $VENV"
