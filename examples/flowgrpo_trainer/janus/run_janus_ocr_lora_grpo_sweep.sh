#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN=$REPO_ROOT/.venv/bin/python
    elif [[ -n "${PYTHON:-}" ]]; then
        PYTHON_BIN=$PYTHON
    else
        PYTHON_BIN=$(command -v python)
    fi
fi

SWEEP_TAG=${SWEEP_TAG:-janus_ocr_seen172_$(date +%m%d_%H%M)}
WANDB_PROJECT=${WANDB_PROJECT:-verl-janus}
MODEL_PATH=${MODEL_PATH:-/home/Models/Janus-Pro-1B}
REWARD_MODEL=${REWARD_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
LAUNCHER=${LAUNCHER:-ray}
TRAIN_GPU_POOL=${TRAIN_GPU_POOL:-0,1,2,3}
AUTO_START_REWARD_SERVER=${AUTO_START_REWARD_SERVER:-0}
REWARD_SERVER_HOST=${REWARD_SERVER_HOST:-127.0.0.1}
REWARD_SERVER_PORT=${REWARD_SERVER_PORT:-8000}
REWARD_URL=${REWARD_URL:-http://$REWARD_SERVER_HOST:$REWARD_SERVER_PORT/v1/chat/completions}
REWARD_SERVER_CUDA_VISIBLE_DEVICES=${REWARD_SERVER_CUDA_VISIBLE_DEVICES:-0,1,2,3}
REWARD_SERVER_TP=${REWARD_SERVER_TP:-4}
REWARD_SERVER_GPU_MEMORY_UTILIZATION=${REWARD_SERVER_GPU_MEMORY_UTILIZATION:-0.20}
REWARD_SERVER_MAX_MODEL_LEN=${REWARD_SERVER_MAX_MODEL_LEN:-8192}
REWARD_SERVER_LOG=${REWARD_SERVER_LOG:-$REPO_ROOT/outputs/janus_ocr_grpo/${SWEEP_TAG}_reward_server.log}
SWEEP_SPECS=${SWEEP_SPECS:-$'1 16 172\n1 8 172\n1 4 172\n2 8 86\n2 4 86\n4 4 43'}

select_gpus() {
    local count=$1
    local pool=$2
    "$PYTHON_BIN" - "$count" "$pool" <<'PY'
import sys

count = int(sys.argv[1])
gpus = [item for item in sys.argv[2].split(",") if item]
if len(gpus) < count:
    raise SystemExit(f"Need {count} GPUs from TRAIN_GPU_POOL, got {sys.argv[2]!r}")
print(",".join(gpus[:count]))
PY
}

wait_reward_server() {
    "$PYTHON_BIN" - "$REWARD_SERVER_HOST" "$REWARD_SERVER_PORT" <<'PY'
import sys
import time
import urllib.request

host, port = sys.argv[1], sys.argv[2]
url = f"http://{host}:{port}/v1/models"
deadline = time.time() + 900
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print(f"OCR reward server ready: {url}")
                raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
    time.sleep(5)
raise SystemExit(f"OCR reward server did not become ready: {last_error}")
PY
}

reward_server_pid=""
cleanup_reward_server() {
    if [[ -n "$reward_server_pid" ]] && kill -0 "$reward_server_pid" 2>/dev/null; then
        kill "$reward_server_pid" 2>/dev/null || true
        wait "$reward_server_pid" 2>/dev/null || true
    fi
}

if [[ "$AUTO_START_REWARD_SERVER" == "1" && "${REWARD_MODE:-openai_ocr}" == "openai_ocr" ]]; then
    mkdir -p "$(dirname "$REWARD_SERVER_LOG")"
    VLLM_BIN=${VLLM_BIN:-$REPO_ROOT/.venv/bin/vllm}
    CUDA_VISIBLE_DEVICES="$REWARD_SERVER_CUDA_VISIBLE_DEVICES" \
        "$VLLM_BIN" serve "$REWARD_MODEL" \
        --host "$REWARD_SERVER_HOST" \
        --port "$REWARD_SERVER_PORT" \
        --dtype bfloat16 \
        --tensor-parallel-size "$REWARD_SERVER_TP" \
        --gpu-memory-utilization "$REWARD_SERVER_GPU_MEMORY_UTILIZATION" \
        --max-model-len "$REWARD_SERVER_MAX_MODEL_LEN" \
        --trust-remote-code \
        --disable-log-stats \
        >"$REWARD_SERVER_LOG" 2>&1 &
    reward_server_pid=$!
    trap cleanup_reward_server EXIT INT TERM
    wait_reward_server
fi

while read -r train_bsz rollout_n steps; do
    if [[ -z "${train_bsz:-}" || "$train_bsz" == \#* ]]; then
        continue
    fi
    run_name="${SWEEP_TAG}_b${train_bsz}_n${rollout_n}_s${steps}"
    train_gpus=$(select_gpus "$train_bsz" "$TRAIN_GPU_POOL")
    echo "==== ${run_name} ===="

    CUDA_VISIBLE_DEVICES="$train_gpus" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_GROUP="$SWEEP_TAG" \
    WANDB_RUN_NAME="$run_name" \
    MODEL_PATH="$MODEL_PATH" \
    NUM_GPUS_ACTOR_ROLLOUT_REWARD="$train_bsz" \
    LAUNCHER="$LAUNCHER" \
    TRAIN_BATCH_SIZE="$train_bsz" \
    ROLLOUT_N="$rollout_n" \
    LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-4}" \
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
done <<< "$SWEEP_SPECS"
