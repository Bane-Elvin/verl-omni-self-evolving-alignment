# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Show-o2 rollout-side adapter for FlowGRPO."""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import (
    SHOW_O2_ARCHITECTURE,
    build_show_o2_model,
    build_show_o2_runtime,
    build_show_o2_vae,
    copy_weights_into_module,
    decode_show_o2_latents,
    lora_config_from_dict,
    maybe_to_cpu,
    prepare_generation_inputs,
    setup_show_o2_sigmas,
    show_o2_time_from_sigma,
)

logger = logging.getLogger(__name__)

_CHAT_MARKERS = (
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


def _to_token_list(token_ids: Any) -> list[int] | None:
    if token_ids is None:
        return None
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _extract_prompt_text(decoded: str) -> str:
    if "<|im_start|>" in decoded:
        user_chunks = []
        for segment in decoded.split("<|im_start|>"):
            if not segment.startswith("user"):
                continue
            content = segment[len("user") :].lstrip("\n")
            content = content.split("<|im_end|>", 1)[0]
            user_chunks.append(content)
        if user_chunks:
            decoded = user_chunks[-1]

    for marker in _CHAT_MARKERS:
        decoded = decoded.replace(marker, "")
    return decoded.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()


def _coalesce_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _default_image_size(runtime) -> tuple[int, int]:
    height = int(os.environ.get("SHOW_O2_HEIGHT", runtime.latent_height * runtime.patch_size * 8))
    width = int(os.environ.get("SHOW_O2_WIDTH", runtime.latent_width * runtime.patch_size * 8))
    return height, width


def _pick_sde_window(
    timesteps_len: int,
    window_size: Optional[int],
    window_range: Optional[Any],
    generator: torch.Generator | None,
    device: torch.device,
) -> tuple[int, int]:
    if window_size is None or int(window_size) <= 0:
        return (0, timesteps_len)
    window_size = int(window_size)
    if window_range is None:
        low, high = 0, timesteps_len
    else:
        low, high = int(window_range[0]), int(window_range[1])

    high_inclusive = min(high, timesteps_len) - window_size
    if high_inclusive < low:
        return (low, min(low + window_size, timesteps_len - 1))

    start = torch.randint(low, high_inclusive + 1, (1,), generator=generator, device=device).item()
    return (int(start), int(start) + window_size)


@VllmOmniPipelineBase.register(SHOW_O2_ARCHITECTURE, algorithm="flow_grpo")
class ShowO2PipelineWithLogProb(torch.nn.Module):
    """Show-o2 pipeline variant for RL rollouts with verl-omni."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.device = torch.device("cpu")
        self.dtype = od_config.dtype
        self.runtime = build_show_o2_runtime(od_config.model, torch_dtype=od_config.dtype)
        self.transformer = build_show_o2_model(od_config.model, torch_dtype=od_config.dtype, device="cpu")
        self.scheduler = FlowMatchSDEDiscreteScheduler()
        self.vae = None
        self.interrupt = False
        logger.info("ShowO2PipelineWithLogProb: SDE scheduler enabled for RL rollouts")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if args:
            first = args[0]
            if isinstance(first, torch.dtype):
                dtype = first
            else:
                device = first
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        if self.device.type != "cpu" and self.vae is None:
            self.vae = build_show_o2_vae(self.runtime, self.device)
        return self

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = list(weights)
        expected = {name for name, _ in self.named_parameters()} | {name for name, _ in self.named_buffers()}
        if not weights:
            return expected
        has_transformer_prefix = any(name.startswith("transformer.") for name, _ in weights)
        loaded = copy_weights_into_module(self, weights)
        return loaded if has_transformer_prefix else expected

    def load_lora_weights(self, weights: dict[str, torch.Tensor], peft_config: dict) -> set[str]:
        if "default" not in getattr(self.transformer, "peft_config", {}):
            self.transformer.add_adapter(lora_config_from_dict(peft_config), adapter_name="default")
            self.transformer.set_adapter("default")
        return copy_weights_into_module(self, weights.items())

    def _decode_token_prompt(self, token_ids: Any) -> str | None:
        token_list = _to_token_list(token_ids)
        if not token_list:
            return None
        decoded = self.runtime.text_tokenizer.decode(token_list, skip_special_tokens=False)
        return _extract_prompt_text(decoded)

    def _get_prompt_text(self, custom_prompt: dict[str, Any]) -> str:
        prompt = custom_prompt.get("prompt")
        if prompt:
            return str(prompt)
        decoded = self._decode_token_prompt(custom_prompt.get("prompt_token_ids"))
        return decoded if decoded is not None else ""

    def _predict_velocity(
        self,
        latents: torch.Tensor,
        model_t: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        modality_positions: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        return self.transformer(
            text_tokens=text_tokens,
            image_latents=latents.to(self.dtype),
            t=model_t,
            attention_mask=attention_mask,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=self.runtime.max_seq_len,
            guidance_scale=guidance_scale,
        )[1]

    def diffuse(
        self,
        latents: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        modality_positions: torch.Tensor,
        negative_text_tokens: torch.Tensor,
        negative_attention_mask: torch.Tensor,
        negative_modality_positions: torch.Tensor,
        timesteps: torch.Tensor,
        guidance_scale: float,
        noise_level: float,
        sde_window: tuple[int, int],
        sde_type: str,
        generator: torch.Generator | None,
        logprobs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        all_latents = []
        all_log_probs = []
        all_timesteps = []
        self.scheduler.set_begin_index(0)
        for i, timestep_value in enumerate(timesteps):
            if self.interrupt:
                continue

            in_window = sde_window[0] <= i < sde_window[1]
            if i == sde_window[0]:
                all_latents.append(latents.float())
            cur_noise_level = noise_level if in_window else 0.0

            model_t = show_o2_time_from_sigma(timestep_value).expand(latents.shape[0]).to(latents.device)
            noise_pred = self._predict_velocity(
                latents,
                model_t,
                text_tokens,
                attention_mask,
                modality_positions,
                guidance_scale,
            )
            if guidance_scale > 0:
                negative_noise_pred = self._predict_velocity(
                    latents,
                    model_t,
                    negative_text_tokens,
                    negative_attention_mask,
                    negative_modality_positions,
                    guidance_scale,
                )
                noise_pred = negative_noise_pred + guidance_scale * (noise_pred - negative_noise_pred)

            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.float(),
                timestep_value,
                latents.float(),
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs and in_window,
                return_dict=False,
            )

            if in_window:
                all_latents.append(latents.float())
                all_log_probs.append(log_prob)
                all_timesteps.append(timestep_value)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        all_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1)
        return latents, all_latents, all_log_probs, all_timesteps

    def forward(
        self,
        req: OmniDiffusionRequest,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 7),
        sde_type: str = "sde",
        logprobs: bool = True,
    ) -> DiffusionOutput:
        custom_prompt = req.prompts[0] if req.prompts and isinstance(req.prompts[0], dict) else {}
        if not custom_prompt:
            return DiffusionOutput(output=None, custom_output={})

        sampling_params = req.sampling_params
        default_height, default_width = _default_image_size(self.runtime)
        height = height or default_height
        width = width or default_width
        if height != width or height != 432:
            logger.warning("Show-o2 official 1.5B config is calibrated for 432x432; got %sx%s", height, width)

        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        guidance_scale = _coalesce_not_none(
            sampling_params.guidance_scale if sampling_params.guidance_scale_provided else None,
            guidance_scale,
            self.runtime.guidance_scale,
        )
        noise_level = _coalesce_not_none(sampling_params.extra_args.get("noise_level"), noise_level)
        sde_window_size = _coalesce_not_none(sampling_params.extra_args.get("sde_window_size"), sde_window_size)
        sde_window_range = _coalesce_not_none(sampling_params.extra_args.get("sde_window_range"), sde_window_range)
        sde_type = _coalesce_not_none(sampling_params.extra_args.get("sde_type"), sde_type)
        logprobs = _coalesce_not_none(sampling_params.extra_args.get("logprobs"), logprobs)

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)

        prompt = self._get_prompt_text(custom_prompt)
        batch_size = 1
        prepared = prepare_generation_inputs([prompt], self.runtime, self.device, self.dtype)
        text_tokens = prepared["text_tokens"]
        attention_mask = prepared["attention_mask"]
        modality_positions = prepared["modality_positions"]
        negative_text_tokens = prepared["negative_text_tokens"]
        negative_attention_mask = prepared["negative_attention_mask"]
        negative_modality_positions = prepared["negative_modality_positions"]

        if latents is None:
            latents = torch.randn(
                (
                    batch_size,
                    self.runtime.image_latent_dim,
                    self.runtime.latent_height * self.runtime.patch_size,
                    self.runtime.latent_width * self.runtime.patch_size,
                ),
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )

        setup_show_o2_sigmas(self.scheduler, num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        sde_window = _pick_sde_window(
            len(timesteps),
            sde_window_size,
            sde_window_range,
            generator,
            self.device,
        )
        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            latents,
            text_tokens,
            attention_mask,
            modality_positions,
            negative_text_tokens,
            negative_attention_mask,
            negative_modality_positions,
            timesteps,
            float(guidance_scale),
            float(noise_level),
            sde_window,
            str(sde_type),
            generator,
            bool(logprobs),
        )

        if output_type == "latent":
            image = latents
        else:
            if self.vae is None:
                self.vae = build_show_o2_vae(self.runtime, self.device)
            image = decode_show_o2_latents(self.vae, latents)

        output_image = image
        if isinstance(image, torch.Tensor) and image.dim() == 4 and image.shape[0] == 1:
            output_image = image[0]

        return DiffusionOutput(
            output=maybe_to_cpu(output_image),
            custom_output={
                "all_latents": maybe_to_cpu(all_latents),
                "all_log_probs": maybe_to_cpu(all_log_probs),
                "all_timesteps": maybe_to_cpu(all_timesteps),
                "show_o2_text_tokens": maybe_to_cpu(text_tokens),
                "show_o2_attention_mask": maybe_to_cpu(attention_mask),
                "show_o2_modality_positions": maybe_to_cpu(modality_positions),
                "show_o2_negative_text_tokens": maybe_to_cpu(negative_text_tokens),
                "show_o2_negative_attention_mask": maybe_to_cpu(negative_attention_mask),
                "show_o2_negative_modality_positions": maybe_to_cpu(negative_modality_positions),
            },
            to_cpu=True,
        )
