#!/usr/bin/env bash
# Show-o2 LoRA RL, vllm_omni rollout (FlowGRPO)
#
# Uses the same OCR parquet files as the BAGEL/Janus examples by default:
#   $HOME/Datasets/data/ocr/bagel/{train,test}.parquet

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

if [[ -z "${SHOW_O2_CODE_PATH:-}" ]]; then
    if [[ -f "$HOME/Projects/Show-o/show-o2/models/modeling_showo2_qwen2_5.py" ]]; then
        SHOW_O2_CODE_PATH=$HOME/Projects/Show-o/show-o2
    else
        SHOW_O2_CODE_PATH=/tmp/show_o_ref/show-o2
    fi
fi
SHOW_O2_CONFIG_PATH=${SHOW_O2_CONFIG_PATH:-$SHOW_O2_CODE_PATH/configs/showo2_1.5b_demo_432x432.yaml}
if [[ -z "${SHOW_O2_VAE_PATH:-}" ]]; then
    for candidate in \
        "$SHOW_O2_CODE_PATH/Wan2.1_VAE.pth" \
        "$HOME/Projects/Bagel-test/Wan2.1_VAE.pth" \
        "$HOME/models/Wan2.1_VAE.pth" \
        "$HOME/Models/Wan2.1_VAE.pth"; do
        if [[ -f "$candidate" ]]; then
            SHOW_O2_VAE_PATH=$candidate
            break
        fi
    done
fi
export SHOW_O2_CODE_PATH SHOW_O2_CONFIG_PATH SHOW_O2_VAE_PATH

show_o2_model_name=${MODEL_PATH:-showlab/show-o2-1.5B}
SHOW_O2_MODEL_PATH=${SHOW_O2_MODEL_PATH:-$show_o2_model_name}
SHOW_O2_LLM_MODEL_PATH=${SHOW_O2_LLM_MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
vllm_config_model_name=${VLLM_CONFIG_MODEL_PATH:-$SHOW_O2_LLM_MODEL_PATH}
tokenizer_name=${TOKENIZER_PATH:-$SHOW_O2_LLM_MODEL_PATH}
export SHOW_O2_LLM_MODEL_PATH SHOW_O2_MODEL_PATH

reward_model_name=${REWARD_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
reward_function_path=${REWARD_FUNCTION_PATH:-$REPO_ROOT/verl_omni/utils/reward_score/genrm_ocr.py}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
REWARD_TP=${REWARD_TP:-4}
ENGINE=${ENGINE:-vllm_omni}
REWARD_ENGINE=${REWARD_ENGINE:-vllm}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_NUM_WORKERS=${ROLLOUT_NUM_WORKERS:-$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP))}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP))}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
WANDB_PROJECT=${WANDB_PROJECT:-verl-show-o2}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-show_o2_ocr_lora}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}

LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
HEIGHT=${HEIGHT:-432}
WIDTH=${WIDTH:-432}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-15}
VAL_NUM_INFERENCE_STEPS=${VAL_NUM_INFERENCE_STEPS:-50}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-1024}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}
NOISE_LEVEL=${NOISE_LEVEL:-0.7}
SDE_TYPE=${SDE_TYPE:-sde}
SDE_WINDOW_SIZE=${SDE_WINDOW_SIZE:-2}
SDE_WINDOW_RANGE=${SDE_WINDOW_RANGE:-"[0,7]"}

ATTN_BACKEND=${ATTN_BACKEND:-native}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-TORCH_SDPA}
export SHOW_O2_HEIGHT=$HEIGHT
export SHOW_O2_WIDTH=$WIDTH

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

if [[ ! -f "$ocr_train_path" ]]; then
    echo "Missing OCR train parquet: $ocr_train_path" >&2
    echo "Set WORKSPACE or TRAIN_FILE to the directory/file containing the BAGEL OCR parquet." >&2
    exit 1
fi
if [[ ! -f "$ocr_test_path" ]]; then
    echo "Missing OCR test parquet: $ocr_test_path" >&2
    echo "Set WORKSPACE or VAL_FILE to the directory/file containing the BAGEL OCR parquet." >&2
    exit 1
fi
if [[ ! -f "$SHOW_O2_CODE_PATH/models/modeling_showo2_qwen2_5.py" ]]; then
    echo "Missing Show-o2 source at SHOW_O2_CODE_PATH=$SHOW_O2_CODE_PATH" >&2
    exit 1
fi
if [[ ! -f "$SHOW_O2_CONFIG_PATH" ]]; then
    echo "Missing Show-o2 config: $SHOW_O2_CONFIG_PATH" >&2
    exit 1
fi
if [[ -z "${SHOW_O2_VAE_PATH:-}" || ! -f "$SHOW_O2_VAE_PATH" ]]; then
    echo "Missing Wan2.1 VAE file. Set SHOW_O2_VAE_PATH=/path/to/Wan2.1_VAE.pth" >&2
    exit 1
fi
"$PYTHON_BIN" - "$SHOW_O2_VAE_PATH" <<'PY'
import sys
import zipfile
from pathlib import Path

vae_path = Path(sys.argv[1])
if not zipfile.is_zipfile(vae_path):
    raise SystemExit(
        f"Wan2.1 VAE checkpoint is not a complete PyTorch zip archive: {vae_path}. "
        "Download it from Wan-AI/Wan2.1-T2V-14B as Wan2.1_VAE.pth."
    )
PY
if [[ "$SHOW_O2_MODEL_PATH" == /* || "$SHOW_O2_MODEL_PATH" == ./* ]]; then
    if [[ ! -e "$SHOW_O2_MODEL_PATH" ]]; then
        echo "Missing local Show-o2 model path: $SHOW_O2_MODEL_PATH" >&2
        exit 1
    fi
fi
if [[ "$vllm_config_model_name" == /* || "$vllm_config_model_name" == ./* ]]; then
    if [[ ! -e "$vllm_config_model_name" ]]; then
        echo "Missing local vLLM config model path: $vllm_config_model_name" >&2
        exit 1
    fi
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

"$PYTHON_BIN" - <<'PY'
import os
import sys
import torch

visible = os.environ.get("CUDA_VISIBLE_DEVICES")
if sys.version_info < (3, 12):
    raise SystemExit(f"Expected Python >= 3.12 for vLLM-Omni, got {sys.version.split()[0]} at {sys.executable}")
try:
    import Levenshtein  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing OCR reward dependency Levenshtein in {sys.executable}. "
        "Install it with: uv pip install --python .venv/bin/python Levenshtein"
    ) from exc
if not torch.cuda.is_available():
    raise SystemExit(f"CUDA is not available. CUDA_VISIBLE_DEVICES={visible!r}")
expected = len((visible or "").split(",")) if visible else 4
if torch.cuda.device_count() < expected:
    raise SystemExit(
        f"Expected {expected} visible CUDA devices, got {torch.cuda.device_count()}. "
        f"CUDA_VISIBLE_DEVICES={visible!r}"
    )
PY

"$PYTHON_BIN" -m verl_omni.trainer.main_diffusion \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.val_max_samples=$VAL_MAX_SAMPLES \
    data.dataloader_num_workers=$DATALOADER_NUM_WORKERS \
    data.max_prompt_length=256 \
    data.trust_remote_code=True \
    algorithm.global_std=False \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=$vllm_config_model_name \
    actor_rollout_ref.model.tokenizer_path=$tokenizer_name \
    +actor_rollout_ref.model.architecture=Showo2Qwen2_5 \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.external_lib=verl_omni.pipelines.show_o2_flow_grpo \
    ++actor_rollout_ref.model.attn_backend=$ATTN_BACKEND \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['showo.model.layers.']" \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.model.pipeline.height=$HEIGHT \
    actor_rollout_ref.model.pipeline.width=$WIDTH \
    actor_rollout_ref.model.pipeline.num_inference_steps=$NUM_INFERENCE_STEPS \
    actor_rollout_ref.model.pipeline.max_sequence_length=$MAX_SEQUENCE_LENGTH \
    actor_rollout_ref.model.pipeline.guidance_scale=$GUIDANCE_SCALE \
    actor_rollout_ref.model.algo.noise_level=$NOISE_LEVEL \
    actor_rollout_ref.model.algo.sde_type="$SDE_TYPE" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.external_lib=verl_omni.pipelines.show_o2_flow_grpo \
    ++actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_NUM_WORKERS \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.height=$HEIGHT \
    actor_rollout_ref.rollout.pipeline.width=$WIDTH \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=$NUM_INFERENCE_STEPS \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_SEQUENCE_LENGTH \
    actor_rollout_ref.rollout.pipeline.guidance_scale=$GUIDANCE_SCALE \
    actor_rollout_ref.rollout.algo.noise_level=$NOISE_LEVEL \
    actor_rollout_ref.rollout.algo.sde_type="$SDE_TYPE" \
    actor_rollout_ref.rollout.algo.sde_window_size=$SDE_WINDOW_SIZE \
    actor_rollout_ref.rollout.algo.sde_window_range="$SDE_WINDOW_RANGE" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=$VAL_NUM_INFERENCE_STEPS \
    actor_rollout_ref.rollout.val_kwargs.pipeline.guidance_scale=$GUIDANCE_SCALE \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    reward.num_workers=$REWARD_NUM_WORKERS \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_RUN_NAME \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    "$@"
