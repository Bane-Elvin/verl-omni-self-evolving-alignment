#!/usr/bin/env python
"""Anole/Chameleon OCR GRPO training.

Anole is a Chameleon-family checkpoint that keeps the Chameleon autoregressive
image-token interface usable for text-to-image generation. This runner mirrors
the Janus OCR GRPO runner while replacing Janus-specific CFG/image-token code
with Chameleon image-only sampling and a standalone VQVAE decoder.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, ChameleonForConditionalGeneration

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chameleon_vqvae import load_vqvae, pixel_values_to_pil  # noqa: E402
from examples.flowgrpo_trainer.janus.janus_ocr_grpo import (  # noqa: E402
    ActorUpdateStats,
    DistributedState,
    PromptRow,
    RolloutBatch,
    all_gather_rollout_group,
    cleanup_distributed,
    compute_training_metrics,
    distributed_barrier,
    dummy_scores,
    group_advantages,
    init_distributed,
    is_main_process,
    load_rows,
    log_eval_generations_to_wandb,
    log_training_step,
    policy_loss_and_stats,
    reduce_float,
    rollout_shard_bounds,
    score_images_openai_ocr,
    seed_everything,
    trainable_parameters,
    update_weights,
)


def base_chameleon_model(model):
    backbone = model.model
    if isinstance(backbone, DDP):
        return backbone.module
    return backbone


def make_chameleon_prompt(processor, prompt_text: str) -> torch.Tensor:
    batch = processor(text=prompt_text, return_tensors="pt")
    return batch["input_ids"][0]


def chameleon_token_ids(model) -> tuple[int, int, torch.Tensor]:
    backbone = base_chameleon_model(model)
    mapping = backbone.vocabulary_mapping
    boi_token_id = getattr(model.config, "boi_token_id", None) or model.config.vocabulary_map["<racm3:break>"]
    eoi_token_id = getattr(model.config, "eoi_token_id", None) or model.config.vocabulary_map["<eoss>"]
    image_token_ids = torch.tensor(mapping.image_tokens, dtype=torch.long, device=model.lm_head.weight.device)
    return int(boi_token_id), int(eoi_token_id), image_token_ids


def mask_to_image_tokens(logits: torch.Tensor, image_token_ids: torch.Tensor) -> torch.Tensor:
    masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
    masked.index_copy_(dim=-1, index=image_token_ids, source=logits.index_select(dim=-1, index=image_token_ids))
    return masked


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits.float(), dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(logits.dtype).min)
    return torch.full_like(logits, torch.finfo(logits.dtype).min).scatter(-1, sorted_indices, sorted_logits)


def convert_bpe_to_vq_tokens(model, bpe_tokens: torch.Tensor) -> torch.Tensor:
    mapping = base_chameleon_model(model).vocabulary_mapping
    max_bpe_id = max(mapping.bpe2img)
    table = torch.full((max_bpe_id + 1,), -1, dtype=torch.long, device=bpe_tokens.device)
    for bpe_id, image_id in mapping.bpe2img.items():
        table[int(bpe_id)] = int(image_id)
    if int(bpe_tokens.max().item()) >= table.numel():
        raise ValueError("Generated token id is outside the Chameleon image-token vocabulary.")
    image_tokens = table[bpe_tokens]
    if bool((image_tokens < 0).any()):
        raise ValueError("Generated sequence contains non-image tokens.")
    return image_tokens


def decode_images(model, vqvae, processor, generated_tokens: torch.Tensor) -> list[Image.Image]:
    image_tokens = convert_bpe_to_vq_tokens(model, generated_tokens)
    with torch.no_grad():
        pixel_values = vqvae.decode(image_tokens)
    return pixel_values_to_pil(pixel_values, rescale_factor=processor.image_processor.rescale_factor)


@torch.no_grad()
def generate_group(
    model,
    processor,
    vqvae,
    prompt_text: str,
    n: int,
    temperature: float,
    top_p: float,
    max_image_tokens: int,
    device: torch.device,
    decode: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[Image.Image]]:
    model.eval()
    prompt_ids = make_chameleon_prompt(processor, prompt_text).to(device)
    input_ids = prompt_ids.unsqueeze(0).repeat(n, 1)
    attention_mask = torch.ones_like(input_ids)
    boi_token_id, _, image_token_ids = chameleon_token_ids(model)

    generated_tokens = torch.zeros((n, max_image_tokens), dtype=torch.long, device=device)
    old_logprobs = torch.zeros((n, max_image_tokens), dtype=torch.float32, device=device)

    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    next_ids = torch.full((n, 1), boi_token_id, dtype=torch.long, device=device)
    outputs = model.model(
        input_ids=next_ids,
        past_key_values=outputs.past_key_values,
        use_cache=True,
        return_dict=True,
    )

    for token_idx in range(max_image_tokens):
        logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
        logits = mask_to_image_tokens(logits, image_token_ids)
        logits = logits / max(temperature, 1e-6)
        sample_logits = apply_top_p(logits, top_p)
        sample_log_probs = torch.log_softmax(sample_logits.float(), dim=-1)
        next_ids = torch.multinomial(sample_log_probs.exp(), num_samples=1)
        generated_tokens[:, token_idx] = next_ids.squeeze(-1)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        old_logprobs[:, token_idx] = log_probs.gather(1, next_ids).squeeze(1)
        outputs = model.model(
            input_ids=next_ids,
            past_key_values=outputs.past_key_values,
            use_cache=True,
            return_dict=True,
        )

    images: list[Image.Image] = []
    if decode:
        if max_image_tokens != 1024:
            raise ValueError(f"Decoding Chameleon images requires 1024 image tokens, got {max_image_tokens}.")
        images = decode_images(model, vqvae, processor, generated_tokens)
    return generated_tokens.detach().clone(), old_logprobs.detach().clone(), images


def chameleon_logprobs(
    model,
    processor,
    prompt_text: str,
    generated_tokens: torch.Tensor,
    temperature: float,
    top_p: float,
    token_chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    n, t = generated_tokens.shape
    prompt_ids = make_chameleon_prompt(processor, prompt_text).to(device)
    boi_token_id, _, image_token_ids = chameleon_token_ids(model)
    boi = torch.full((1,), boi_token_id, dtype=torch.long, device=device)
    prompt_boi = torch.cat([prompt_ids, boi], dim=0)
    prompt_len = prompt_boi.numel()

    prompt_batch = prompt_boi.unsqueeze(0).repeat(n, 1)
    if t > 1:
        input_ids = torch.cat([prompt_batch, generated_tokens[:, :-1].to(device)], dim=1)
    else:
        input_ids = prompt_batch
    attention_mask = torch.ones_like(input_ids)

    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state[:, prompt_len - 1 : prompt_len - 1 + t, :]
    logprob_chunks = []
    for start in range(0, t, token_chunk_size):
        end = min(t, start + token_chunk_size)
        logits = model.lm_head(hidden[:, start:end, :])
        logits = mask_to_image_tokens(logits, image_token_ids)
        logits = logits / max(temperature, 1e-6)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        targets = generated_tokens[:, start:end].to(device)
        logprob_chunks.append(log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1))
    return torch.cat(logprob_chunks, dim=1)


def setup_lora(model, args):
    from peft import LoraConfig, PeftModel, get_peft_model  # noqa: PLC0415

    for param in model.parameters():
        param.requires_grad = False

    if args.adapter_path:
        model.model = PeftModel.from_pretrained(
            model.model,
            args.adapter_path,
            is_trainable=not args.eval_only,
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.lora_target_modules.split(","),
        )
        model.model = get_peft_model(model.model, lora_config)
    return model


def make_output_dir(args) -> Path:
    run_name = args.wandb_run_name or f"anole_ocr_b{args.train_batch_size}_n{args.rollout_n}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def compute_advantage(
    local_rewards: torch.Tensor,
    args: argparse.Namespace,
    state: DistributedState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    adv_start = time.perf_counter()
    rewards = all_gather_rollout_group(local_rewards, state)
    advantages = group_advantages(rewards)
    _, rollout_start_idx, rollout_end_idx = rollout_shard_bounds(args.rollout_n, state)
    advantages_per_gpu = advantages[:, rollout_start_idx:rollout_end_idx]
    return rewards, advantages, advantages_per_gpu, time.perf_counter() - adv_start


def evaluate(model, processor, vqvae, rows: list[PromptRow], args, device: torch.device, output_dir: Path, step: int):
    eval_dir = output_dir / f"eval_step_{step}"
    if args.save_eval_images:
        eval_dir.mkdir(parents=True, exist_ok=True)

    all_scores: list[float] = []
    jsonl_path = eval_dir / "results.jsonl"
    fh = jsonl_path.open("w", encoding="utf-8") if args.save_eval_images else None
    try:
        for row in tqdm(rows[: args.eval_samples], desc="eval"):
            tokens, _, images = generate_group(
                model=model,
                processor=processor,
                vqvae=vqvae,
                prompt_text=row.prompt,
                n=args.eval_rollout_n,
                temperature=args.temperature,
                top_p=args.top_p,
                max_image_tokens=args.max_image_tokens,
                device=device,
                decode=True,
            )
            del tokens
            if args.reward_mode == "dummy":
                scores, texts = dummy_scores(args.eval_rollout_n, row.ground_truth, row.index)
            else:
                scores, texts = score_images_openai_ocr(
                    images,
                    row.ground_truth,
                    args.reward_url,
                    args.reward_model,
                    args.reward_timeout,
                )
            all_scores.extend(scores)
            if args.save_eval_images and fh is not None:
                sample_dir = eval_dir / f"{row.index:06d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                for image_idx, image in enumerate(images):
                    image.save(sample_dir / f"{image_idx:03d}.png")
                fh.write(
                    json.dumps(
                        {
                            "index": row.index,
                            "prompt": row.prompt,
                            "ground_truth": row.ground_truth,
                            "scores": scores,
                            "ocr_texts": texts,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if fh is not None:
            fh.close()

    score = float(np.mean(all_scores)) if all_scores else 0.0
    return {
        "eval/score_mean": score,
        "eval/score_max": float(np.max(all_scores)) if all_scores else 0.0,
        "eval/score_min": float(np.min(all_scores)) if all_scores else 0.0,
        "eval/num_images": len(all_scores),
    }


def wandb_config(args) -> dict:
    config = vars(args).copy()
    config["trainer"] = {
        "project_name": args.wandb_project,
        "experiment_name": args.wandb_run_name,
        "logger": json.loads(args.trainer_logger),
        "n_gpus_per_node": args.ray_num_workers,
        "total_training_steps": args.total_training_steps,
    }
    config["actor_rollout_ref"] = {
        "actor": {
            "rollout_n": args.rollout_n,
            "ppo_mini_batch_size": args.ppo_mini_batch_size,
            "ppo_micro_batch_size_per_gpu": args.ppo_micro_batch_size_per_gpu,
            "policy_loss": args.policy_loss,
        },
        "model": {
            "path": args.model_path,
            "vqvae_path": args.vqvae_path,
            "architecture": "ChameleonForConditionalGeneration",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.lora_target_modules.split(","),
        },
        "rollout": {
            "n": args.rollout_n,
            "max_image_tokens": args.max_image_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    return config


def trainer_logger_backends(args) -> list[str]:
    value = args.trainer_logger
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def init_wandb(args, output_dir: Path):
    if "wandb" not in trainer_logger_backends(args) or args.wandb_mode == "disabled":
        return None
    import wandb  # noqa: PLC0415

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        group=args.wandb_group or None,
        mode=args.wandb_mode,
        dir=str(output_dir),
        config=wandb_config(args),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Anole/Chameleon on OCR prompts with AR-GRPO.")
    parser.add_argument("--model-path", default="/home/elvin/Models/Anole-7b-v0.1-hf")
    parser.add_argument("--vqvae-path", default="/home/elvin/Models/Anole-7b-v0.1-vqvae-hf")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--output-dir", default="outputs/anole_ocr_grpo")
    parser.add_argument("--train-file", default="/home/elvin/Datasets/data/ocr/bagel/train.parquet")
    parser.add_argument("--val-file", default="/home/elvin/Datasets/data/ocr/bagel/test.parquet")
    parser.add_argument("--eval-only", action="store_true")

    parser.add_argument("--launcher", choices=["local", "ray"], default="local")
    parser.add_argument("--ray-num-workers", type=int, default=1)
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", ""))
    parser.add_argument("--ray-storage-path", default="")

    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=1)
    parser.add_argument("--ppo-micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--total-training-steps", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument(
        "--policy-loss",
        choices=["stage_grpo", "sequence_grpo", "token_ppo"],
        default="stage_grpo",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-image-tokens", type=int, default=1024)
    parser.add_argument("--token-chunk-size", type=int, default=64)
    parser.add_argument(
        "--log-prob-micro-batch-size-per-gpu",
        "--logprob-micro-batch-size",
        dest="log_prob_micro_batch_size_per_gpu",
        type=int,
        default=1,
    )

    parser.add_argument("--reward-mode", choices=["openai_ocr", "dummy"], default="openai_ocr")
    parser.add_argument("--reward-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--reward-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--reward-timeout", type=float, default=600.0)

    parser.add_argument("--eval-at-end", action="store_true")
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--eval-rollout-n", type=int, default=1)
    parser.add_argument("--save-eval-images", action="store_true")
    parser.add_argument("--log-val-generations", type=int, default=8)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--vqvae-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--wandb-project", default="verl-chameleon")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--trainer-logger", default='["console","wandb"]')
    parser.add_argument("--save-freq", type=int, default=20)
    parser.add_argument("--test-freq", type=int, default=20)
    parser.add_argument("--total-epochs", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def validate_training_args(args: argparse.Namespace, state: DistributedState) -> None:
    if args.max_image_tokens != 1024 and args.reward_mode != "dummy":
        raise SystemExit("Non-dummy OCR reward requires full 1024-token Chameleon images.")
    if args.ppo_mini_batch_size != args.train_batch_size:
        raise SystemExit(
            "Chameleon OCR currently updates one global prompt batch per PPO epoch. "
            f"Set PPO_MINI_BATCH_SIZE equal to TRAIN_BATCH_SIZE ({args.train_batch_size}), "
            f"got {args.ppo_mini_batch_size}."
        )
    if state.enabled and args.rollout_n % state.world_size != 0:
        raise SystemExit(
            "Ray/DDP Chameleon shards each prompt's rollout group across workers. "
            f"ROLLOUT_N must be divisible by world_size={state.world_size}, got {args.rollout_n}."
        )
    if not Path(args.model_path).expanduser().exists():
        raise SystemExit(f"Missing Chameleon-family model path: {args.model_path}")
    if not Path(args.vqvae_path).expanduser().exists():
        raise SystemExit(f"Missing Chameleon-family VQVAE path: {args.vqvae_path}")


def load_model_and_processor(args: argparse.Namespace, state: DistributedState):
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    vqvae_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.vqvae_dtype]

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model = ChameleonForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        dtype=dtype,
    ).to(device)
    model = setup_lora(model, args).to(device)
    vqvae = load_vqvae(args.vqvae_path, device=device, dtype=vqvae_dtype)
    if state.enabled and not args.eval_only:
        model.model = DDP(
            model.model,
            device_ids=[state.local_rank],
            output_device=state.local_rank,
            find_unused_parameters=False,
        )
    model.train()
    if state.enabled:
        seed_everything(args.seed + state.rank * 100_003)
    return model, processor, vqvae, device


def select_training_batch(train_rows: list[PromptRow], step: int, train_batch_size: int):
    global_offset = (step - 1) * train_batch_size
    rows = [train_rows[(global_offset + i) % len(train_rows)] for i in range(train_batch_size)]
    return rows, global_offset


def generate_rollout_batch(
    model,
    processor,
    vqvae,
    rows: list[PromptRow],
    args: argparse.Namespace,
    state: DistributedState,
    device: torch.device,
) -> RolloutBatch:
    rollout_start = time.perf_counter()
    rollout_n_per_gpu, _, _ = rollout_shard_bounds(args.rollout_n, state)

    group_tokens: list[torch.Tensor] = []
    group_old_logprobs: list[torch.Tensor] = []
    reward_rows: list[list[float]] = []
    gen_times: list[float] = []
    reward_times: list[float] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []

    for row in rows:
        prompt_lengths.append(int(make_chameleon_prompt(processor, row.prompt).numel()))
        gen_one_start = time.perf_counter()
        with torch.no_grad():
            tokens, old_logprobs, images = generate_group(
                model=model,
                processor=processor,
                vqvae=vqvae,
                prompt_text=row.prompt,
                n=rollout_n_per_gpu,
                temperature=args.temperature,
                top_p=args.top_p,
                max_image_tokens=args.max_image_tokens,
                device=device,
                decode=args.reward_mode != "dummy",
            )
        response_lengths.append(int(tokens.shape[-1]))
        gen_times.append(time.perf_counter() - gen_one_start)

        reward_one_start = time.perf_counter()
        if args.reward_mode == "dummy":
            scores, _ = dummy_scores(rollout_n_per_gpu, row.ground_truth, row.index + state.rank * 1_000_000)
        else:
            scores, _ = score_images_openai_ocr(
                images,
                row.ground_truth,
                args.reward_url,
                args.reward_model,
                args.reward_timeout,
            )
        reward_times.append(time.perf_counter() - reward_one_start)

        group_tokens.append(tokens.clone())
        group_old_logprobs.append(old_logprobs.clone())
        reward_rows.append(scores)

    local_rewards = (
        torch.tensor(reward_rows, dtype=torch.float32, device=device)
        if reward_rows
        else torch.empty((0, rollout_n_per_gpu), dtype=torch.float32, device=device)
    )
    rollout_time = time.perf_counter() - rollout_start
    rewards, advantages, advantages_per_gpu, adv_time = compute_advantage(local_rewards, args, state)
    return RolloutBatch(
        rows=rows,
        tokens=group_tokens,
        old_logprobs=group_old_logprobs,
        rewards=rewards,
        advantages=advantages,
        advantages_per_gpu=advantages_per_gpu,
        gen_times=gen_times,
        reward_times=reward_times,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        rollout_time=rollout_time,
        adv_time=adv_time,
    )


def update_actor(
    model,
    processor,
    optimizer: torch.optim.Optimizer,
    rollout_batch: RolloutBatch,
    args: argparse.Namespace,
    device: torch.device,
) -> ActorUpdateStats:
    update_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    actor_losses = []
    actor_loss_count = 0
    ratio_values = []
    ratio_std_values = []
    clip_values = []
    clip_high_values = []
    clip_low_values = []

    for _ in range(args.ppo_epochs):
        local_items = zip(
            rollout_batch.rows,
            rollout_batch.tokens,
            rollout_batch.old_logprobs,
            rollout_batch.advantages_per_gpu,
            strict=True,
        )
        for row, tokens, old_logprobs, adv in local_items:
            tokens = tokens.to(device)
            old_logprobs = old_logprobs.to(device)
            adv = adv.to(device)
            micro_batch_size = (
                tokens.shape[0]
                if args.ppo_micro_batch_size_per_gpu <= 0
                else min(args.ppo_micro_batch_size_per_gpu, tokens.shape[0])
            )
            total_loss_count = max(1, int(tokens.numel() if args.policy_loss == "token_ppo" else tokens.shape[0]))
            for micro_start in range(0, tokens.shape[0], micro_batch_size):
                micro_end = min(tokens.shape[0], micro_start + micro_batch_size)
                micro_tokens = tokens[micro_start:micro_end].contiguous()
                micro_old_logprobs = old_logprobs[micro_start:micro_end].contiguous()
                micro_adv = adv[micro_start:micro_end].contiguous()
                new_logprobs = chameleon_logprobs(
                    model=model,
                    processor=processor,
                    prompt_text=row.prompt,
                    generated_tokens=micro_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    token_chunk_size=args.token_chunk_size,
                    device=device,
                )
                raw_loss, ratio_float, micro_loss_count = policy_loss_and_stats(
                    new_logprobs=new_logprobs,
                    old_logprobs=micro_old_logprobs,
                    advantages=micro_adv,
                    policy_loss=args.policy_loss,
                    clip_ratio=args.clip_ratio,
                )
                loss_scale = micro_loss_count / total_loss_count
                loss = raw_loss * loss_scale / args.train_batch_size / args.ppo_epochs
                loss.backward()

                actor_losses.append(raw_loss.detach().float() * micro_loss_count)
                actor_loss_count += micro_loss_count
                ratio_values.append(ratio_float.mean() * micro_loss_count)
                ratio_std_values.append(ratio_float.std(unbiased=False) * micro_loss_count)
                clip_values.append((torch.abs(ratio_float - 1.0) > args.clip_ratio).float().mean() * micro_loss_count)
                clip_high_values.append((ratio_float > 1.0 + args.clip_ratio).float().mean() * micro_loss_count)
                clip_low_values.append((ratio_float < 1.0 - args.clip_ratio).float().mean() * micro_loss_count)

    return ActorUpdateStats(
        loss_sum=float(torch.stack(actor_losses).sum().item()) if actor_losses else 0.0,
        ratio_sum=float(torch.stack(ratio_values).sum().item()) if ratio_values else 0.0,
        ratio_std_sum=float(torch.stack(ratio_std_values).sum().item()) if ratio_std_values else 0.0,
        clipfrac_sum=float(torch.stack(clip_values).sum().item()) if clip_values else 0.0,
        clipfrac_higher_sum=float(torch.stack(clip_high_values).sum().item()) if clip_high_values else 0.0,
        clipfrac_lower_sum=float(torch.stack(clip_low_values).sum().item()) if clip_low_values else 0.0,
        loss_count=actor_loss_count,
        update_actor_time=time.perf_counter() - update_start,
    )


def save_adapter_and_evaluate(
    model,
    processor,
    vqvae,
    val_rows: list[PromptRow],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    run,
) -> None:
    adapter_dir = output_dir / "adapter"
    base_chameleon_model(model).save_pretrained(adapter_dir)
    print(f"saved adapter: {adapter_dir}")

    if args.eval_at_end:
        eval_metrics = evaluate(model, processor, vqvae, val_rows, args, device, output_dir, step=args.total_training_steps)
        print("final_eval:" + json.dumps(eval_metrics, ensure_ascii=False))
        if run is not None:
            run.log(eval_metrics, step=args.total_training_steps)
            log_eval_generations_to_wandb(
                run,
                output_dir,
                step=args.total_training_steps,
                limit=args.log_val_generations,
            )


def run_worker(args: argparse.Namespace) -> None:
    state = init_distributed(args)
    run = None
    try:
        validate_training_args(args, state)
        model, processor, vqvae, device = load_model_and_processor(args, state)

        train_limit = args.max_train_rows if args.max_train_rows > 0 else None
        train_rows = load_rows(Path(args.train_file), limit=train_limit)
        val_rows = load_rows(Path(args.val_file), limit=args.eval_samples)
        output_dir = make_output_dir(args)
        run = init_wandb(args, output_dir) if is_main_process(state) else None

        if args.eval_only:
            if is_main_process(state):
                metrics = evaluate(model, processor, vqvae, val_rows, args, device, output_dir, step=0)
                print(json.dumps(metrics, indent=2))
                if run is not None:
                    run.log(metrics, step=0)
            if state.enabled:
                distributed_barrier(state)
            return

        params = trainable_parameters(model)
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        if is_main_process(state):
            print(f"trainable parameters: {sum(p.numel() for p in params):,}")
            if state.enabled:
                print(f"distributed world_size: {state.world_size}")

        for step in range(1, args.total_training_steps + 1):
            step_start = time.perf_counter()
            batch_rows, global_offset = select_training_batch(train_rows, step, args.train_batch_size)
            rollout_batch = generate_rollout_batch(
                model=model,
                processor=processor,
                vqvae=vqvae,
                rows=batch_rows,
                args=args,
                state=state,
                device=device,
            )
            actor_stats = update_actor(
                model=model,
                processor=processor,
                optimizer=optimizer,
                rollout_batch=rollout_batch,
                args=args,
                device=device,
            )
            weight_stats = update_weights(params=params, optimizer=optimizer, args=args)
            metrics = compute_training_metrics(
                step=step,
                global_offset=global_offset,
                train_rows=train_rows,
                rollout_batch=rollout_batch,
                actor_stats=actor_stats,
                weight_stats=weight_stats,
                optimizer=optimizer,
                args=args,
                state=state,
                device=device,
                step_time=time.perf_counter() - step_start,
            )
            log_training_step(run, metrics, step, args, state)

        model.model = base_chameleon_model(model)
        if is_main_process(state):
            save_adapter_and_evaluate(model, processor, vqvae, val_rows, args, device, output_dir, run)
        if state.enabled:
            distributed_barrier(state)
    finally:
        if run is not None:
            run.finish()
        cleanup_distributed(state)


def ray_train_loop(config: dict) -> None:
    run_worker(argparse.Namespace(**config))


def run_ray(args: argparse.Namespace) -> None:
    import ray  # noqa: PLC0415
    from ray.train import RunConfig, ScalingConfig  # noqa: PLC0415
    from ray.train.torch import TorchTrainer  # noqa: PLC0415

    if args.ray_num_workers < 1:
        raise SystemExit("--ray-num-workers must be >= 1")
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    run_name = args.wandb_run_name or f"anole_ocr_b{args.train_batch_size}_n{args.rollout_n}"
    storage_path = args.ray_storage_path or str((Path(args.output_dir).resolve() / "ray").absolute())
    try:
        trainer = TorchTrainer(
            ray_train_loop,
            train_loop_config=vars(args),
            scaling_config=ScalingConfig(num_workers=args.ray_num_workers, use_gpu=True),
            run_config=RunConfig(name=run_name, storage_path=storage_path),
        )
        trainer.fit()
    finally:
        ray.shutdown()


def main() -> None:
    args = parse_args()
    if args.launcher == "ray":
        run_ray(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
