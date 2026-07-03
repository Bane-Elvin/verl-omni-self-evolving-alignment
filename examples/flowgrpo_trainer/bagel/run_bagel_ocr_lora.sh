#!/usr/bin/env bash
# Bagel LoRA RL, vllm_omni rollout (FlowGRPO)
#
# Prerequisite: preprocess the OCR dataset for BAGEL:
# python examples/flowgrpo_trainer/data_process/bagel_ocr.py \
#   --model_path ~/models/ByteDance-Seed/BAGEL-7B-MoT \
#   --input_dir ~/data/ocr \
#   --output_dir ~/data/ocr/bagel

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
ocr_train_path=$WORKSPACE/data/ocr/bagel/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/bagel/test.parquet

BAGEL_DEPLOY_CONFIG=${BAGEL_DEPLOY_CONFIG:-"$SCRIPT_DIR/bagel_deploy_config.yaml"}
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d "$HOME/Projects/Bagel-test/models/BAGEL-7B-MoT" ]]; then
        model_name=$HOME/Projects/Bagel-test/models/BAGEL-7B-MoT
    else
        model_name=$HOME/models/ByteDance-Seed/BAGEL-7B-MoT
    fi
else
    model_name=$MODEL_PATH
fi
tokenizer_name=${TOKENIZER_PATH:-$model_name}
reward_model_name=${REWARD_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
reward_function_path=${REWARD_FUNCTION_PATH:-$REPO_ROOT/verl_omni/utils/reward_score/genrm_ocr.py}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-4}
ROLLOUT_TP=${ROLLOUT_TP:-4}
REWARD_TP=${REWARD_TP:-4}
ENGINE=${ENGINE:-vllm_omni}
REWARD_ENGINE=${REWARD_ENGINE:-vllm}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_N=${ROLLOUT_N:-2}
# BAGEL rollout replicas each keep a large diffusion stack in host memory.
# With TP=4, the 4 local GPUs form one rollout replica instead of four
# independent replicas; keep one agent worker for the same startup profile.
ROLLOUT_NUM_WORKERS=${ROLLOUT_NUM_WORKERS:-1}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP))}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
WANDB_PROJECT=${WANDB_PROJECT:-verl-bagel}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-bagel_ocr_lora}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}

# FA3/FlashAttention is faster when fully available. On this machine the actor
# path can fall back to native while rollout stays FLASH_ATTN, which fails the
# consistency check before training starts. Use the conservative matched pair by
# default; override both env vars together if FA3 is installed and verified.
ATTN_BACKEND=${ATTN_BACKEND:-native}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-TORCH_SDPA}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

if [[ ! -f "$ocr_train_path" ]]; then
    echo "Missing BAGEL OCR train parquet: $ocr_train_path" >&2
    echo "Set WORKSPACE to the directory containing data/ocr/bagel/train.parquet." >&2
    exit 1
fi
if [[ ! -f "$ocr_test_path" ]]; then
    echo "Missing BAGEL OCR test parquet: $ocr_test_path" >&2
    echo "Set WORKSPACE to the directory containing data/ocr/bagel/test.parquet." >&2
    exit 1
fi
if [[ ! -d "$model_name" ]]; then
    echo "Missing BAGEL model directory: $model_name" >&2
    echo "Set MODEL_PATH to the local BAGEL-7B-MoT directory." >&2
    exit 1
fi
if [[ ! -f "$BAGEL_DEPLOY_CONFIG" ]]; then
    echo "Missing BAGEL deploy config: $BAGEL_DEPLOY_CONFIG" >&2
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
if torch.cuda.device_count() < 4:
    raise SystemExit(
        f"Expected 4 visible CUDA devices, got {torch.cuda.device_count()}. "
        f"CUDA_VISIBLE_DEVICES={visible!r}"
    )
PY

"$PYTHON_BIN" -m verl_omni.trainer.main_diffusion \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.dataloader_num_workers=$DATALOADER_NUM_WORKERS \
    data.max_prompt_length=256 \
    data.trust_remote_code=True \
    algorithm.global_std=False \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$tokenizer_name \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=True \
    ++actor_rollout_ref.model.attn_backend=$ATTN_BACKEND \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.target_modules="['q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen','mlp_moe_gen.gate_proj','mlp_moe_gen.up_proj','mlp_moe_gen.down_proj']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    ++actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_NUM_WORKERS \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=15 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=0.7 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,7]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config=$BAGEL_DEPLOY_CONFIG \
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
