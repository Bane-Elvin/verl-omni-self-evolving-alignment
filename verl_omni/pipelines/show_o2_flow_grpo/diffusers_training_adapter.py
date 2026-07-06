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

"""Show-o2 training-side adapter for FlowGRPO."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    SHOW_O2_ARCHITECTURE,
    build_show_o2_model,
    setup_show_o2_sigmas,
    show_o2_time_from_sigma,
)

logger = logging.getLogger(__name__)


@DiffusionModelBase.register(SHOW_O2_ARCHITECTURE, algorithm="flow_grpo")
class ShowO2Diffusion(DiffusionModelBase):
    """DiffusionModelBase wrapper for official Show-o2."""

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype):
        logger.info("Loading Show-o2 from %s", model_config.local_path)
        return build_show_o2_model(model_config.local_path, torch_dtype=torch_dtype, device="cpu")

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = FlowMatchSDEDiscreteScheduler()
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        setup_show_o2_sigmas(scheduler, model_config.pipeline.num_inference_steps, device=device)

    @staticmethod
    def _guidance_scale(model_config: DiffusionModelConfig, module) -> float:
        if model_config.pipeline.guidance_scale is not None:
            return float(model_config.pipeline.guidance_scale)
        runtime = getattr(module, "show_o2_runtime", None)
        if runtime is not None:
            return float(runtime.guidance_scale)
        return 5.0

    @staticmethod
    def _attention_mask(mask: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if mask.dtype == torch.bool:
            return mask.to(device=device)
        return mask.to(device=device, dtype=dtype)

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, dict]:
        device = latents.device
        hidden_states = latents[:, step]
        timestep = show_o2_time_from_sigma(timesteps[:, step])
        max_seq_len = getattr(
            getattr(module, "show_o2_runtime", None),
            "max_seq_len",
            model_config.pipeline.max_sequence_length,
        )

        model_inputs = {
            "text_tokens": micro_batch["show_o2_text_tokens"].to(device=device, dtype=torch.long),
            "image_latents": hidden_states,
            "t": timestep,
            "attention_mask": cls._attention_mask(
                micro_batch["show_o2_attention_mask"],
                device,
                hidden_states.dtype,
            ),
            "modality_positions": micro_batch["show_o2_modality_positions"].to(device=device, dtype=torch.long),
            "output_hidden_states": True,
            "max_seq_len": max_seq_len,
            "guidance_scale": cls._guidance_scale(model_config, module),
        }
        negative_model_inputs = {
            "text_tokens": micro_batch["show_o2_negative_text_tokens"].to(device=device, dtype=torch.long),
            "image_latents": hidden_states,
            "t": timestep,
            "attention_mask": cls._attention_mask(
                micro_batch["show_o2_negative_attention_mask"],
                device,
                hidden_states.dtype,
            ),
            "modality_positions": micro_batch["show_o2_negative_modality_positions"].to(
                device=device,
                dtype=torch.long,
            ),
            "output_hidden_states": True,
            "max_seq_len": max_seq_len,
            "guidance_scale": cls._guidance_scale(model_config, module),
        }
        return model_inputs, negative_model_inputs

    @classmethod
    def forward(cls, module, model_config: DiffusionModelConfig, model_inputs: dict, negative_model_inputs=None):
        return module(**model_inputs)[1]

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = cls.forward(module, model_config, model_inputs)
        guidance_scale = cls._guidance_scale(model_config, module)
        if guidance_scale > 0:
            assert negative_model_inputs is not None
            negative_noise_pred = cls.forward(module, model_config, negative_model_inputs)
            noise_pred = negative_noise_pred + guidance_scale * (noise_pred - negative_noise_pred)

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
