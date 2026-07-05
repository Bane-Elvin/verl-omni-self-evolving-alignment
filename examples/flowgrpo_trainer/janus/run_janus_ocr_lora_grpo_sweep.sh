#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

SWEEP_TAG=${SWEEP_TAG:-janus_ocr_seen172_$(date +%m%d_%H%M)}
WANDB_PROJECT=${WANDB_PROJECT:-verl-janus}
MODEL_PATH=${MODEL_PATH:-/home/Models/Janus-Pro-1B}
REWARD_URL=${REWARD_URL:-http://127.0.0.1:8000/v1/chat/completions}
REWARD_MODEL=${REWARD_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-${NPROC_PER_NODE:-4}}
LAUNCHER=${LAUNCHER:-ray}

for spec in \
    "1 16 172" \
    "1 8 172" \
    "1 4 172" \
    "2 8 86" \
    "2 4 86" \
    "4 4 43"
do
    read -r train_bsz rollout_n steps <<< "$spec"
    run_name="${SWEEP_TAG}_b${train_bsz}_n${rollout_n}_s${steps}"
    echo "==== ${run_name} ===="

    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_GROUP="$SWEEP_TAG" \
    WANDB_RUN_NAME="$run_name" \
    MODEL_PATH="$MODEL_PATH" \
    NUM_GPUS_ACTOR_ROLLOUT_REWARD="$NUM_GPUS_ACTOR_ROLLOUT_REWARD" \
    LAUNCHER="$LAUNCHER" \
    TRAIN_BATCH_SIZE="$train_bsz" \
    ROLLOUT_N="$rollout_n" \
    TOTAL_TRAINING_STEPS="$steps" \
    SAVE_FREQ="$steps" \
    TEST_FREQ="$steps" \
    REWARD_MODE="${REWARD_MODE:-openai_ocr}" \
    REWARD_URL="$REWARD_URL" \
    REWARD_MODEL="$REWARD_MODEL" \
    EVAL_AT_END=1 \
    EVAL_SAMPLES="${EVAL_SAMPLES:-32}" \
    EVAL_ROLLOUT_N="${EVAL_ROLLOUT_N:-1}" \
    SAVE_EVAL_IMAGES="${SAVE_EVAL_IMAGES:-0}" \
    bash "$SCRIPT_DIR/run_janus_ocr_lora_grpo.sh"
done
