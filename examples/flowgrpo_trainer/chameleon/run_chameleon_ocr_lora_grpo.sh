#!/usr/bin/env bash
# Anole/Chameleon LoRA RL, Ray-distributed AR-GRPO.
#
# This mirrors the Janus OCR launcher shape: the dataset paths, W&B fields,
# GPU-count variable, save/test frequency knobs, and trainer logger names use
# the same environment variables.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
case ":${PYTHONPATH:-}:" in
    *":$REPO_ROOT:"*) ;;
    *) export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" ;;
esac

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN=$REPO_ROOT/.venv/bin/python
    elif [[ -n "${PYTHON:-}" ]]; then
        PYTHON_BIN=$PYTHON
    else
        PYTHON_BIN=$(command -v python)
    fi
fi

prepend_ld_library_path() {
    local lib_dir=$1
    if [[ -d "$lib_dir" ]]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$lib_dir:"*) ;;
            *) export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}" ;;
        esac
    fi
}

abs_path() {
    local path=$1
    if [[ "$path" == /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$REPO_ROOT" "$path"
    fi
}

PYTHON_SITE_PACKAGES=$("$PYTHON_BIN" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)
prepend_ld_library_path "$PYTHON_SITE_PACKAGES/nvidia/cu13/lib"

if [[ -z "${WORKSPACE:-}" ]]; then
    if [[ -f "$HOME/Datasets/data/ocr/bagel/train.parquet" ]]; then
        WORKSPACE=$HOME/Datasets
    else
        WORKSPACE=$HOME
    fi
fi
ocr_train_path=${TRAIN_FILE:-$WORKSPACE/data/ocr/bagel/train.parquet}
ocr_test_path=${VAL_FILE:-$WORKSPACE/data/ocr/bagel/test.parquet}

if [[ -z "${MODEL_PATH:-}" ]]; then
    model_name=
    for candidate in \
        "$HOME/Models/Anole-7b-v0.1-hf" \
        "$HOME/Models/chameleon-7b" \
        "/home/Models/Anole-7b-v0.1-hf" \
        "/home/Models/chameleon-7b"; do
        if [[ -d "$candidate" ]]; then
            model_name=$candidate
            break
        fi
    done
else
    model_name=$MODEL_PATH
fi
VQVAE_PATH=${VQVAE_PATH:-$HOME/Models/Anole-7b-v0.1-vqvae-hf}
reward_model_name=${REWARD_MODEL_PATH:-${REWARD_MODEL:-Qwen/Qwen3-VL-8B-Instruct}}
reward_url=${REWARD_URL:-http://127.0.0.1:8000/v1/chat/completions}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-${NPROC_PER_NODE:-4}}
LAUNCHER=${LAUNCHER:-ray}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-${LOGPROB_MICRO_BATCH_SIZE:-1}}
ROLLOUT_N=${ROLLOUT_N:-4}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
SAVE_FREQ=${SAVE_FREQ:-$TOTAL_TRAINING_STEPS}
TEST_FREQ=${TEST_FREQ:-$TOTAL_TRAINING_STEPS}
WANDB_PROJECT=${WANDB_PROJECT:-verl-chameleon}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-anole_7b_ocr_lora}
WANDB_GROUP=${WANDB_GROUP:-}
WANDB_MODE=${WANDB_MODE:-online}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}

REWARD_MODE=${REWARD_MODE:-openai_ocr}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-bfloat16}
VQVAE_DTYPE=${VQVAE_DTYPE:-float32}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.9}
MAX_IMAGE_TOKENS=${MAX_IMAGE_TOKENS:-1024}
TOKEN_CHUNK_SIZE=${TOKEN_CHUNK_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
PPO_EPOCHS=${PPO_EPOCHS:-1}
POLICY_LOSS=${POLICY_LOSS:-stage_grpo}
EVAL_AT_END=${EVAL_AT_END:-1}
EVAL_SAMPLES=${EVAL_SAMPLES:-$VAL_BATCH_SIZE}
EVAL_ROLLOUT_N=${EVAL_ROLLOUT_N:-1}
SAVE_EVAL_IMAGES=${SAVE_EVAL_IMAGES:-1}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-8}
ADAPTER_PATH=${ADAPTER_PATH:-}
EVAL_ONLY=${EVAL_ONLY:-0}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/anole_7b_ocr_grpo}
OUTPUT_DIR=$(abs_path "$OUTPUT_DIR")
if [[ -n "$ADAPTER_PATH" ]]; then
    ADAPTER_PATH=$(abs_path "$ADAPTER_PATH")
fi
RAY_STORAGE_PATH=${RAY_STORAGE_PATH:-$OUTPUT_DIR/ray}
RAY_STORAGE_PATH=$(abs_path "$RAY_STORAGE_PATH")
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}

if [[ -z "$model_name" || ! -d "$model_name" ]]; then
    echo "Missing Chameleon-family model directory: ${model_name:-<empty>}" >&2
    echo "Set MODEL_PATH to /home/elvin/Models/Anole-7b-v0.1-hf or another local Chameleon-family checkpoint." >&2
    exit 1
fi
if [[ ! -d "$VQVAE_PATH" ]]; then
    echo "Missing Chameleon VQVAE directory: $VQVAE_PATH" >&2
    echo "Set VQVAE_PATH to /home/elvin/Models/Anole-7b-v0.1-vqvae-hf." >&2
    exit 1
fi
if [[ ! -f "$ocr_train_path" ]]; then
    echo "Missing OCR train parquet: $ocr_train_path" >&2
    echo "Set WORKSPACE to the directory containing data/ocr/bagel/train.parquet." >&2
    exit 1
fi
if [[ ! -f "$ocr_test_path" ]]; then
    echo "Missing OCR test parquet: $ocr_test_path" >&2
    echo "Set WORKSPACE to the directory containing data/ocr/bagel/test.parquet." >&2
    exit 1
fi

if [[ "$TOTAL_TRAINING_STEPS" == "full" || "$TOTAL_TRAINING_STEPS" == "epoch" ]]; then
    TOTAL_TRAINING_STEPS=$("$PYTHON_BIN" - "$ocr_train_path" "$TRAIN_BATCH_SIZE" <<'PY'
import sys

import pandas as pd

num_rows = len(pd.read_parquet(sys.argv[1]))
batch_size = int(sys.argv[2])
print(num_rows // batch_size)
PY
)
fi

"$PYTHON_BIN" - "$NUM_GPUS_ACTOR_ROLLOUT_REWARD" "$LAUNCHER" "$TRAINER_LOGGER" <<'PY'
import json
import os
import sys

import torch

expected_gpus = int(sys.argv[1])
launcher = sys.argv[2]
try:
    json.loads(sys.argv[3])
except json.JSONDecodeError as exc:
    raise SystemExit(f"TRAINER_LOGGER must be JSON, got {sys.argv[3]!r}") from exc
if sys.version_info < (3, 12):
    raise SystemExit(f"Expected Python >= 3.12, got {sys.version.split()[0]} at {sys.executable}")
try:
    import Levenshtein  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing OCR reward dependency Levenshtein in {sys.executable}. "
        "Install it with: uv pip install --python .venv/bin/python Levenshtein"
    ) from exc
if not torch.cuda.is_available():
    raise SystemExit(f"CUDA is not available. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
if launcher in {"ray", "torchrun"} and torch.cuda.device_count() < expected_gpus:
    raise SystemExit(
        f"Expected {expected_gpus} visible CUDA devices, got {torch.cuda.device_count()}. "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
    )
PY

python_launcher=$LAUNCHER
if [[ "$LAUNCHER" == "torchrun" ]]; then
    python_launcher=local
fi

args=(
    "$SCRIPT_DIR/anole_ocr_grpo.py"
    --launcher "$python_launcher"
    --ray-num-workers "$NUM_GPUS_ACTOR_ROLLOUT_REWARD"
    --ray-storage-path "$RAY_STORAGE_PATH"
    --model-path "$model_name"
    --vqvae-path "$VQVAE_PATH"
    --train-file "$ocr_train_path"
    --val-file "$ocr_test_path"
    --output-dir "$OUTPUT_DIR"
    --train-batch-size "$TRAIN_BATCH_SIZE"
    --ppo-mini-batch-size "$PPO_MINI_BATCH_SIZE"
    --ppo-micro-batch-size-per-gpu "$PPO_MICRO_BATCH_SIZE_PER_GPU"
    --rollout-n "$ROLLOUT_N"
    --total-training-steps "$TOTAL_TRAINING_STEPS"
    --ppo-epochs "$PPO_EPOCHS"
    --clip-ratio "${CLIP_RATIO:-0.2}"
    --policy-loss "$POLICY_LOSS"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --max-grad-norm "$MAX_GRAD_NORM"
    --lora-rank "$LORA_RANK"
    --lora-alpha "$LORA_ALPHA"
    --lora-dropout "$LORA_DROPOUT"
    --lora-target-modules "$LORA_TARGET_MODULES"
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$WANDB_RUN_NAME"
    --wandb-mode "$WANDB_MODE"
    --trainer-logger "$TRAINER_LOGGER"
    --save-freq "$SAVE_FREQ"
    --test-freq "$TEST_FREQ"
    --total-epochs "$TOTAL_EPOCHS"
    --reward-mode "$REWARD_MODE"
    --reward-url "$reward_url"
    --reward-model "$reward_model_name"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --vqvae-dtype "$VQVAE_DTYPE"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --max-image-tokens "$MAX_IMAGE_TOKENS"
    --token-chunk-size "$TOKEN_CHUNK_SIZE"
    --log-prob-micro-batch-size-per-gpu "$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    --eval-samples "$EVAL_SAMPLES"
    --eval-rollout-n "$EVAL_ROLLOUT_N"
    --log-val-generations "$LOG_VAL_GENERATIONS"
)

if [[ -n "$WANDB_GROUP" ]]; then
    args+=(--wandb-group "$WANDB_GROUP")
fi
if [[ -n "$ADAPTER_PATH" ]]; then
    args+=(--adapter-path "$ADAPTER_PATH")
fi
if [[ "$EVAL_AT_END" == "1" ]]; then
    args+=(--eval-at-end)
fi
if [[ "$SAVE_EVAL_IMAGES" == "1" ]]; then
    args+=(--save-eval-images)
fi
if [[ "$EVAL_ONLY" == "1" ]]; then
    args+=(--eval-only)
fi

case "$LAUNCHER" in
    ray|local)
        exec "$PYTHON_BIN" "${args[@]}" "$@"
        ;;
    torchrun)
        exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NUM_GPUS_ACTOR_ROLLOUT_REWARD" "${args[@]}" "$@"
        ;;
    *)
        echo "Unsupported LAUNCHER=$LAUNCHER. Expected ray, torchrun, or local." >&2
        exit 1
        ;;
esac
