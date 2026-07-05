#!/usr/bin/env python
"""Janus-Pro OCR GRPO training.

This is a small standalone AR-GRPO runner for Janus-Pro image-token generation.
It is intentionally separate from ``verl_omni.trainer.main_diffusion`` because
Janus-Pro generates discrete image tokens autoregressively, while Flow-GRPO in
this repo is wired around diffusion/flow-matching trajectories.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig


DEFAULT_GRM_PROMPT = "Please output only the text content from the image without any additional descriptions or formatting."


@dataclass
class PromptRow:
    prompt: str
    ground_truth: str
    index: int


@dataclass
class DistributedState:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    owns_process_group: bool


def init_distributed(args) -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    owns_process_group = False
    if enabled:
        if not torch.cuda.is_available():
            raise SystemExit("Distributed Janus training requires CUDA.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            owns_process_group = True
        args.device = f"cuda:{local_rank}"
    return DistributedState(
        enabled=enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        owns_process_group=owns_process_group,
    )


def cleanup_distributed(state: DistributedState) -> None:
    if state.enabled and state.owns_process_group and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(state: DistributedState) -> None:
    if state.enabled:
        dist.barrier(device_ids=[state.local_rank])


def is_main_process(state: DistributedState) -> bool:
    return state.rank == 0


def base_language_model(model):
    language_model = model.language_model
    if isinstance(language_model, DDP):
        return language_model.module
    return language_model


def reduce_float(value: float, state: DistributedState, device: torch.device, op=dist.ReduceOp.SUM) -> float:
    if not state.enabled:
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=op)
    if op == dist.ReduceOp.SUM:
        tensor /= state.world_size
    return float(tensor.item())


def patch_transformers_for_janus() -> None:
    """Keep DeepSeek Janus official code importable with newer transformers.

    Recent transformers dataclass-wraps all ``PretrainedConfig`` subclasses.
    Janus official code defines mutable class defaults on config classes, which
    trips Python 3.12 dataclass validation. Janus already implements explicit
    ``__init__`` methods, so skipping the dataclass wrapper is sufficient.
    """

    def _compat_init_subclass(cls, *args, **kwargs):
        super(PretrainedConfig, cls).__init_subclass__(*args, **kwargs)

    PretrainedConfig.__init_subclass__ = classmethod(_compat_init_subclass)


def import_janus(janus_code_path: str):
    patch_transformers_for_janus()
    if janus_code_path and janus_code_path not in sys.path:
        sys.path.insert(0, janus_code_path)
    from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa: PLC0415

    if not hasattr(MultiModalityCausalLM, "all_tied_weights_keys"):
        MultiModalityCausalLM.all_tied_weights_keys = {}

    return VLChatProcessor


def clean_config_fields(config) -> None:
    """Replace dataclasses.Field defaults left by newer transformers configs."""

    for key in dir(config):
        if key.startswith("_"):
            continue
        try:
            value = getattr(config, key)
        except Exception:
            continue
        if isinstance(value, dataclasses.Field):
            if value.default is not dataclasses.MISSING:
                setattr(config, key, value.default)
        elif isinstance(value, PretrainedConfig):
            clean_config_fields(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_prompt(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        if not value:
            return ""
        first = value[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value)


def extract_ground_truth(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("ground_truth", ""))
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return str(parsed.get("ground_truth", ""))
        except json.JSONDecodeError:
            return value
    return str(value)


def load_rows(path: Path, limit: int | None = None) -> list[PromptRow]:
    df = pd.read_parquet(path)
    if limit is not None:
        df = df.head(limit)
    rows: list[PromptRow] = []
    for i, row in df.iterrows():
        rows.append(
            PromptRow(
                prompt=extract_prompt(row["prompt"]),
                ground_truth=extract_ground_truth(row["reward_model"]),
                index=int(row.get("index", i)) if "index" in row else int(i),
            )
        )
    return rows


def make_janus_prompt(processor, prompt_text: str) -> torch.Tensor:
    conversation = [
        {"role": "<|User|>", "content": prompt_text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    prompt = sft + processor.image_start_tag
    return torch.LongTensor(processor.tokenizer.encode(prompt))


def make_cond_uncond_prompt(input_ids: torch.Tensor, pad_id: int, n: int, device: torch.device) -> torch.Tensor:
    tokens = torch.zeros((n * 2, input_ids.numel()), dtype=torch.long, device=device)
    input_ids = input_ids.to(device)
    for i in range(n * 2):
        tokens[i] = input_ids
        if i % 2 == 1 and input_ids.numel() > 2:
            tokens[i, 1:-1] = pad_id
    return tokens


def decode_images(model, generated_tokens: torch.Tensor, image_size: int, patch_size: int) -> list[Image.Image]:
    decoded = model.gen_vision_model.decode_code(
        generated_tokens,
        shape=[
            generated_tokens.shape[0],
            8,
            image_size // patch_size,
            image_size // patch_size,
        ],
    )
    array = decoded.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    array = np.clip((array + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return [Image.fromarray(item) for item in array]


@torch.inference_mode()
def generate_group(
    model,
    processor,
    prompt_text: str,
    n: int,
    cfg_weight: float,
    temperature: float,
    max_image_tokens: int,
    image_size: int,
    patch_size: int,
    device: torch.device,
    decode: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[Image.Image]]:
    model.eval()
    input_ids = make_janus_prompt(processor, prompt_text)
    prompt_tokens = make_cond_uncond_prompt(input_ids, processor.pad_id, n, device)
    inputs_embeds = base_language_model(model).get_input_embeddings()(prompt_tokens)

    generated_tokens = torch.zeros((n, max_image_tokens), dtype=torch.long, device=device)
    old_logprobs = torch.zeros((n, max_image_tokens), dtype=torch.float32, device=device)
    outputs = None

    for token_idx in range(max_image_tokens):
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if outputs is not None else None,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        logits = model.gen_head(hidden[:, -1, :])
        cond = logits[0::2]
        uncond = logits[1::2]
        guided = uncond + cfg_weight * (cond - uncond)
        log_probs = torch.log_softmax(guided.float() / temperature, dim=-1)
        next_token = torch.multinomial(log_probs.exp(), num_samples=1)
        generated_tokens[:, token_idx] = next_token.squeeze(-1)
        old_logprobs[:, token_idx] = log_probs.gather(1, next_token).squeeze(1)

        doubled = torch.cat([next_token, next_token], dim=1).view(-1)
        next_embeds = model.prepare_gen_img_embeds(doubled)
        inputs_embeds = next_embeds.unsqueeze(1)

    images: list[Image.Image] = []
    if decode:
        if max_image_tokens != processor.num_image_tokens:
            raise ValueError(
                f"Decoding Janus images requires {processor.num_image_tokens} image tokens, got {max_image_tokens}."
            )
        images = decode_images(model, generated_tokens, image_size=image_size, patch_size=patch_size)
    return generated_tokens.detach().clone(), old_logprobs.detach().clone(), images


def guided_logprobs(
    model,
    processor,
    prompt_text: str,
    generated_tokens: torch.Tensor,
    cfg_weight: float,
    temperature: float,
    token_chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    n, t = generated_tokens.shape
    input_ids = make_janus_prompt(processor, prompt_text)
    prompt_tokens = make_cond_uncond_prompt(input_ids, processor.pad_id, n, device)
    prompt_len = prompt_tokens.shape[1]

    prompt_embeds = base_language_model(model).get_input_embeddings()(prompt_tokens)
    if t > 1:
        prev_tokens = generated_tokens[:, :-1].repeat_interleave(2, dim=0).to(device)
        prev_embeds = model.prepare_gen_img_embeds(prev_tokens)
        inputs_embeds = torch.cat([prompt_embeds, prev_embeds], dim=1)
    else:
        inputs_embeds = prompt_embeds

    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1][:, prompt_len - 1 : prompt_len - 1 + t, :]
    logprob_chunks = []
    for start in range(0, t, token_chunk_size):
        end = min(t, start + token_chunk_size)
        logits = model.gen_head(hidden[:, start:end, :])
        cond = logits[0::2]
        uncond = logits[1::2]
        guided = uncond + cfg_weight * (cond - uncond)
        log_probs = torch.log_softmax(guided.float() / temperature, dim=-1)
        targets = generated_tokens[:, start:end].to(device)
        logprob_chunks.append(log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1))
    return torch.cat(logprob_chunks, dim=1)


def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def levenshtein_score(text: str, ground_truth: str) -> float:
    import Levenshtein  # noqa: PLC0415

    import re

    gt = re.sub(r"\s+", "", ground_truth).lower()
    pred = re.sub(r"\s+", "", text).lower()
    if gt in pred:
        dist = 0
    else:
        dist = Levenshtein.distance(pred, gt)
    dist = min(dist, len(gt))
    if len(gt) > 0:
        return 1.0 - dist / len(gt)
    return 1.0 if len(pred) == 0 else 0.0


def score_images_openai_ocr(
    images: list[Image.Image],
    ground_truth: str,
    reward_url: str,
    reward_model: str,
    timeout: float,
) -> tuple[list[float], list[str]]:
    scores = []
    texts = []
    for image in images:
        payload = {
            "model": reward_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": pil_to_data_url(image)}},
                        {"type": "text", "text": DEFAULT_GRM_PROMPT},
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        resp = requests.post(reward_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
        texts.append(text)
        scores.append(levenshtein_score(text, ground_truth))
    return scores, texts


def dummy_scores(n: int, ground_truth: str, prompt_index: int) -> tuple[list[float], list[str]]:
    rng = random.Random(hash((ground_truth, prompt_index)) & 0xFFFFFFFF)
    scores = [rng.random() for _ in range(n)]
    return scores, ["<dummy>"] * n


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    adv = (rewards - mean) / (std + 1e-6)
    return torch.where(std > 0, adv, torch.zeros_like(adv))


def setup_lora(model, args):
    from peft import LoraConfig, PeftModel, get_peft_model  # noqa: PLC0415

    for param in model.parameters():
        param.requires_grad = False

    if args.adapter_path:
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            args.adapter_path,
            is_trainable=not args.eval_only,
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
    return model


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def make_output_dir(args) -> Path:
    run_name = args.wandb_run_name or f"janus_ocr_b{args.train_batch_size}_n{args.rollout_n}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def evaluate(model, processor, rows: list[PromptRow], args, device: torch.device, output_dir: Path, step: int) -> dict:
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
                prompt_text=row.prompt,
                n=args.eval_rollout_n,
                cfg_weight=args.cfg_weight,
                temperature=args.temperature,
                max_image_tokens=args.max_image_tokens,
                image_size=args.image_size,
                patch_size=args.patch_size,
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
                for i, image in enumerate(images):
                    image.save(sample_dir / f"{i:03d}.png")
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


def log_eval_generations_to_wandb(run, output_dir: Path, step: int, limit: int) -> None:
    if run is None or limit <= 0:
        return
    eval_dir = output_dir / f"eval_step_{step}"
    jsonl_path = eval_dir / "results.jsonl"
    if not jsonl_path.exists():
        return

    import wandb  # noqa: PLC0415

    table = wandb.Table(columns=["index", "prompt", "ground_truth", "score", "ocr_text", "image"])
    logged = 0
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if logged >= limit:
                break
            item = json.loads(line)
            sample_dir = eval_dir / f"{int(item['index']):06d}"
            image_paths = sorted(sample_dir.glob("*.png"))
            for image_idx, image_path in enumerate(image_paths):
                if logged >= limit:
                    break
                scores = item.get("scores", [])
                texts = item.get("ocr_texts", [])
                table.add_data(
                    item["index"],
                    item["prompt"],
                    item["ground_truth"],
                    scores[image_idx] if image_idx < len(scores) else None,
                    texts[image_idx] if image_idx < len(texts) else "",
                    wandb.Image(str(image_path)),
                )
                logged += 1
    if logged > 0:
        run.log({"eval/generations": table}, step=step)


def wandb_config(args) -> dict:
    config = vars(args).copy()
    config["data"] = {
        "train_files": args.train_file,
        "val_files": args.val_file,
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.eval_samples,
    }
    config["actor_rollout_ref"] = {
        "model": {
            "path": args.model_path,
            "architecture": "Janus-Pro",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.lora_target_modules.split(","),
        },
        "actor": {
            "optim": {"lr": args.learning_rate, "weight_decay": args.weight_decay},
            "ppo_mini_batch_size": args.train_batch_size,
            "ppo_micro_batch_size_per_gpu": 1,
        },
        "rollout": {
            "name": "janus_ar",
            "n": args.rollout_n,
            "max_image_tokens": args.max_image_tokens,
            "cfg_weight": args.cfg_weight,
            "temperature": args.temperature,
        },
    }
    config["reward"] = {
        "reward_model": {
            "enable": args.reward_mode == "openai_ocr",
            "model_path": args.reward_model,
            "rollout": {"name": "vllm_openai_compatible"},
        },
        "custom_reward_function": {"name": args.reward_mode},
    }
    config["trainer"] = {
        "project_name": args.wandb_project,
        "experiment_name": args.wandb_run_name,
        "logger": json.loads(args.trainer_logger),
        "log_val_generations": args.log_val_generations,
        "n_gpus_per_node": args.ray_num_workers,
        "nnodes": 1,
        "save_freq": args.save_freq,
        "test_freq": args.test_freq,
        "total_epochs": args.total_epochs,
        "total_training_steps": args.total_training_steps,
    }
    return config


def init_wandb(args, output_dir: Path):
    if args.wandb_mode == "disabled":
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
    parser = argparse.ArgumentParser(description="Train Janus-Pro on OCR prompts with AR-GRPO.")
    parser.add_argument("--janus-code-path", default="/home/elvin/Projects/Bagel-test/.cache/janus_official")
    parser.add_argument("--model-path", default="/home/Models/Janus-Pro-1B")
    parser.add_argument("--train-file", default="/home/elvin/Datasets/data/ocr/bagel/train.parquet")
    parser.add_argument("--val-file", default="/home/elvin/Datasets/data/ocr/bagel/test.parquet")
    parser.add_argument("--output-dir", default="outputs/janus_ocr_grpo")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--launcher", choices=["local", "ray"], default="local")
    parser.add_argument("--ray-num-workers", type=int, default=1)
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", ""))
    parser.add_argument("--ray-storage-path", default="")

    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--total-training-steps", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
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

    parser.add_argument("--cfg-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-image-tokens", type=int, default=576)
    parser.add_argument("--token-chunk-size", type=int, default=64)

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
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--wandb-project", default="verl-janus")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--trainer-logger", default='["console","wandb"]')
    parser.add_argument("--save-freq", type=int, default=20)
    parser.add_argument("--test-freq", type=int, default=20)
    parser.add_argument("--total-epochs", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def run_worker(args: argparse.Namespace) -> None:
    if args.max_image_tokens != 576 and args.reward_mode != "dummy":
        raise SystemExit("Non-dummy OCR reward requires full 576-token Janus images.")

    state = init_distributed(args)
    run = None
    try:
        seed_everything(args.seed)
        VLChatProcessor = import_janus(args.janus_code_path)
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
        clean_config_fields(config)

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
        processor = VLChatProcessor.from_pretrained(args.model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        model = setup_lora(model, args).to(device)
        if state.enabled and not args.eval_only:
            model.language_model = DDP(
                model.language_model,
                device_ids=[state.local_rank],
                output_device=state.local_rank,
                find_unused_parameters=False,
            )
        model.train()

        train_limit = args.max_train_rows if args.max_train_rows > 0 else None
        train_rows = load_rows(Path(args.train_file), limit=train_limit)
        val_rows = load_rows(Path(args.val_file), limit=args.eval_samples)
        output_dir = make_output_dir(args)
        run = init_wandb(args, output_dir) if is_main_process(state) else None

        if args.eval_only:
            if is_main_process(state):
                metrics = evaluate(model, processor, val_rows, args, device, output_dir, step=0)
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
            rollout_start = time.perf_counter()
            batch_rows = []
            global_offset = (step - 1) * args.train_batch_size * state.world_size
            rank_offset = global_offset + state.rank * args.train_batch_size
            for i in range(args.train_batch_size):
                batch_rows.append(train_rows[(rank_offset + i) % len(train_rows)])

            group_tokens: list[torch.Tensor] = []
            group_old_logprobs: list[torch.Tensor] = []
            reward_rows: list[list[float]] = []
            ocr_text_rows: list[list[str]] = []

            for row in batch_rows:
                decode = args.reward_mode != "dummy"
                tokens, old_logprobs, images = generate_group(
                    model=model,
                    processor=processor,
                    prompt_text=row.prompt,
                    n=args.rollout_n,
                    cfg_weight=args.cfg_weight,
                    temperature=args.temperature,
                    max_image_tokens=args.max_image_tokens,
                    image_size=args.image_size,
                    patch_size=args.patch_size,
                    device=device,
                    decode=decode,
                )
                if args.reward_mode == "dummy":
                    scores, texts = dummy_scores(args.rollout_n, row.ground_truth, row.index)
                else:
                    scores, texts = score_images_openai_ocr(
                        images,
                        row.ground_truth,
                        args.reward_url,
                        args.reward_model,
                        args.reward_timeout,
                    )
                group_tokens.append(tokens.clone())
                group_old_logprobs.append(old_logprobs.clone())
                reward_rows.append(scores)
                ocr_text_rows.append(texts)
            rollout_time = time.perf_counter() - rollout_start

            rewards = torch.tensor(reward_rows, dtype=torch.float32, device=device)
            advantages = group_advantages(rewards)

            update_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            actor_losses = []
            ratio_values = []
            clip_values = []
            for _ in range(args.ppo_epochs):
                for row, tokens, old_logprobs, adv in zip(batch_rows, group_tokens, group_old_logprobs, advantages):
                    tokens = tokens.to(device)
                    old_logprobs = old_logprobs.to(device)
                    adv = adv.to(device)
                    new_logprobs = guided_logprobs(
                        model=model,
                        processor=processor,
                        prompt_text=row.prompt,
                        generated_tokens=tokens,
                        cfg_weight=args.cfg_weight,
                        temperature=args.temperature,
                        token_chunk_size=args.token_chunk_size,
                        device=device,
                    )
                    ratio = torch.exp(new_logprobs - old_logprobs)
                    adv_tokens = adv[:, None].expand_as(ratio)
                    unclipped = ratio * adv_tokens
                    clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * adv_tokens
                    loss = -torch.min(unclipped, clipped).mean() / args.train_batch_size / args.ppo_epochs
                    loss.backward()
                    actor_losses.append(loss.detach().float() * args.train_batch_size * args.ppo_epochs)
                    ratio_values.append(ratio.detach().float().mean())
                    clip_values.append((torch.abs(ratio.detach().float() - 1.0) > args.clip_ratio).float().mean())

            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            update_time = time.perf_counter() - update_start
            step_time = time.perf_counter() - step_start

            rewards_np = rewards.detach().cpu().numpy()
            per_prompt_std = rewards_np.std(axis=1)
            local_step_time = float(step_time)
            global_step_time = reduce_float(local_step_time, state, device, op=dist.ReduceOp.MAX)
            global_images = int(args.train_batch_size * args.rollout_n * state.world_size)
            metrics = {
                "training/global_step": step,
                "training/seen_prompts": int(step * args.train_batch_size * state.world_size),
                "actor/loss": reduce_float(
                    float(torch.stack(actor_losses).mean().item()) if actor_losses else 0.0,
                    state,
                    device,
                ),
                "actor/ratio_mean": reduce_float(
                    float(torch.stack(ratio_values).mean().item()) if ratio_values else 0.0,
                    state,
                    device,
                ),
                "actor/pg_clipfrac": reduce_float(
                    float(torch.stack(clip_values).mean().item()) if clip_values else 0.0,
                    state,
                    device,
                ),
                "actor/grad_norm": reduce_float(
                    float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                    state,
                    device,
                ),
                "actor/lr": optimizer.param_groups[0]["lr"],
                "critic/rewards/mean": reduce_float(float(rewards_np.mean()), state, device),
                "critic/rewards/max": reduce_float(float(rewards_np.max()), state, device, op=dist.ReduceOp.MAX),
                "critic/rewards/min": reduce_float(float(rewards_np.min()), state, device, op=dist.ReduceOp.MIN),
                "critic/rewards/std_mean": reduce_float(float(per_prompt_std.mean()), state, device),
                "critic/rewards/zero_std_ratio": reduce_float(float(np.mean(per_prompt_std == 0)), state, device),
                "critic/rewards/group_size": float(args.rollout_n),
                "perf/total_num_images": global_images,
                "perf/time_per_step": global_step_time,
                "perf/throughput": float(global_images / global_step_time),
                "timing_s/gen_reward": reduce_float(float(rollout_time), state, device, op=dist.ReduceOp.MAX),
                "timing_s/update_actor": reduce_float(float(update_time), state, device, op=dist.ReduceOp.MAX),
                "timing_s/step": global_step_time,
            }

            if run is not None:
                run.log(metrics, step=step)
            if is_main_process(state) and step % args.log_every == 0:
                print("step:" + str(step) + " - " + " - ".join(f"{k}:{v}" for k, v in metrics.items()))

        model.language_model = base_language_model(model)
        if is_main_process(state):
            adapter_dir = output_dir / "adapter"
            model.language_model.save_pretrained(adapter_dir)
            print(f"saved adapter: {adapter_dir}")

            if args.eval_at_end:
                eval_metrics = evaluate(model, processor, val_rows, args, device, output_dir, step=args.total_training_steps)
                print("final_eval:" + json.dumps(eval_metrics, ensure_ascii=False))
                if run is not None:
                    run.log(eval_metrics, step=args.total_training_steps)
                    log_eval_generations_to_wandb(
                        run,
                        output_dir,
                        step=args.total_training_steps,
                        limit=args.log_val_generations,
                    )

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

    run_name = args.wandb_run_name or f"janus_ocr_b{args.train_batch_size}_n{args.rollout_n}"
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
