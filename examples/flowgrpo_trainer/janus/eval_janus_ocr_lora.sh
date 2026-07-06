#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JANUS_MODEL_SIZE=${JANUS_MODEL_SIZE:-7B}
janus_model_size_slug=${JANUS_MODEL_SIZE,,}

if [[ -z "${ADAPTER_PATH:-}" ]]; then
    echo "Set ADAPTER_PATH to a saved Janus LoRA adapter directory." >&2
    exit 1
fi

EVAL_ONLY=1 \
EVAL_AT_END=0 \
SAVE_EVAL_IMAGES=${SAVE_EVAL_IMAGES:-1} \
WANDB_RUN_NAME=${WANDB_RUN_NAME:-janus_pro_${janus_model_size_slug}_ocr_eval_$(date +%m%d_%H%M)} \
bash "$SCRIPT_DIR/run_janus_ocr_lora_grpo.sh" "$@"
