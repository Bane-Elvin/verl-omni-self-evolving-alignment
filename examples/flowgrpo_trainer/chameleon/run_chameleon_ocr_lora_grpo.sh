#!/usr/bin/env bash
# Chameleon OCR preflight.
#
# The public facebook/chameleon-7b checkpoint is image-text-to-text and does not
# expose image output decoding in Transformers. This script intentionally fails
# fast unless MODEL_PATH points to a Chameleon-family checkpoint with image-token
# generation and image decoding support.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN=$REPO_ROOT/.venv/bin/python
    elif [[ -n "${PYTHON:-}" ]]; then
        PYTHON_BIN=$PYTHON
    else
        PYTHON_BIN=$(command -v python)
    fi
fi

MODEL_PATH=${MODEL_PATH:-/home/elvin/Models/chameleon-7b}

exec "$PYTHON_BIN" "$SCRIPT_DIR/check_chameleon_ocr_support.py" --model-path "$MODEL_PATH"
