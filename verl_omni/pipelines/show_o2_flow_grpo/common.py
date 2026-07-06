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

"""Shared utilities for Show-o2 FlowGRPO adapters."""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

import torch
from omegaconf import OmegaConf

from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

SHOW_O2_ARCHITECTURE = "Showo2Qwen2_5"
DEFAULT_SHOW_O2_CONFIG = "configs/showo2_1.5b_demo_432x432.yaml"
SHOW_O2_TIMESTEP_SHIFT = 3.0


@dataclass(frozen=True)
class ShowO2Runtime:
    config: Any
    text_tokenizer: Any
    showo_token_ids: dict[str, int]
    num_t2i_image_tokens: int
    max_seq_len: int
    max_text_len: int
    image_latent_dim: int
    patch_size: int
    latent_width: int
    latent_height: int
    pad_id: int
    bos_id: int
    eos_id: int
    boi_id: int
    eoi_id: int
    img_pad_id: int
    guidance_scale: float
    weight_dtype: torch.dtype


def maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def lora_config_from_dict(config: dict):
    from peft import LoraConfig

    if isinstance(config, LoraConfig):
        return config
    if hasattr(LoraConfig, "from_dict"):
        return LoraConfig.from_dict(config)
    return LoraConfig(**dict(config))


def get_show_o2_code_path() -> str:
    candidates = [
        os.environ.get("SHOW_O2_CODE_PATH"),
        str(Path.home() / "Projects/Show-o/show-o2"),
        "/tmp/show_o_ref/show-o2",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "models/modeling_showo2_qwen2_5.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find Show-o2 source. Set SHOW_O2_CODE_PATH to the directory containing "
        "models/modeling_showo2_qwen2_5.py."
    )


def ensure_show_o2_code_path() -> str:
    code_path = get_show_o2_code_path()
    if code_path not in sys.path:
        sys.path.insert(0, code_path)
    return code_path


def get_show_o2_config_path(code_path: str) -> str:
    config_path = os.environ.get("SHOW_O2_CONFIG_PATH")
    if config_path is None:
        config_path = str(Path(code_path) / DEFAULT_SHOW_O2_CONFIG)
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Missing Show-o2 config: {config_path}")
    return config_path


def _resolve_llm_name(llm_model_path: str) -> str:
    if "llama" in llm_model_path.lower():
        return "llama3"
    return "qwen2_5"


def load_show_o2_config(model_path: str | None = None):
    code_path = ensure_show_o2_code_path()
    config = OmegaConf.load(get_show_o2_config_path(code_path))
    pretrained_model_path = os.environ.get("SHOW_O2_MODEL_PATH") or model_path
    if pretrained_model_path is not None:
        config.model.showo.pretrained_model_path = pretrained_model_path
    if os.environ.get("SHOW_O2_LLM_MODEL_PATH"):
        config.model.showo.llm_model_path = os.environ["SHOW_O2_LLM_MODEL_PATH"]
    if os.environ.get("SHOW_O2_VAE_PATH"):
        config.model.vae_model.pretrained_model_path = os.environ["SHOW_O2_VAE_PATH"]
    if os.environ.get("SHOW_O2_CLIP_MODEL_PATH"):
        config.model.showo.clip_pretrained_model_path = os.environ["SHOW_O2_CLIP_MODEL_PATH"]

    if bool(config.model.showo.add_time_embeds) and not getattr(config, "_verl_time_embed_tokens_added", False):
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
        config._verl_time_embed_tokens_added = True
    return config


def get_show_o2_weight_dtype(config) -> torch.dtype:
    weight_type = str(config.model.weight_type)
    if weight_type == "bfloat16":
        return torch.bfloat16
    if weight_type == "float16":
        return torch.float16
    if weight_type == "float32":
        return torch.float32
    raise NotImplementedError(f"Unsupported Show-o2 weight_type={weight_type!r}")


def _patch_from_pretrained_without_empty_param_hook(model_cls) -> None:
    patch_flag = "_verl_show_o2_from_pretrained_patched"
    if getattr(model_cls, patch_flag, False):
        return

    from transformers import AutoConfig

    def _from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        attn_implementation = kwargs.pop("attn_implementation", None)
        torch_dtype = kwargs.pop("torch_dtype", kwargs.pop("dtype", None))
        kwargs.pop("low_cpu_mem_usage", None)
        kwargs.pop("device_map", None)
        kwargs.pop("use_safetensors", None)

        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        elif attn_implementation is not None:
            config._attn_implementation = attn_implementation

        model = cls(config, *model_args)
        state_dict = load_hf_state_dict(pretrained_model_name_or_path)
        model.load_state_dict(state_dict, strict=False)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        model.eval()
        return model

    model_cls.from_pretrained = classmethod(_from_pretrained)
    setattr(model_cls, patch_flag, True)


def patch_show_o2_transformers_compat() -> None:
    ensure_show_o2_code_path()

    from models import modeling_utils as show_o2_modeling_utils
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            return cls(ddp_cache_data=past_key_values)

        DynamicCache.from_legacy_cache = _from_legacy_cache

    if not hasattr(DynamicCache, "to_legacy_cache"):

        def _to_legacy_cache(self):
            return tuple(
                (layer.keys, layer.values)
                for layer in self.layers
                if getattr(layer, "keys", None) is not None and getattr(layer, "values", None) is not None
            )

        DynamicCache.to_legacy_cache = _to_legacy_cache

    if "default" not in ROPE_INIT_FUNCTIONS:

        def _compute_default_rope_parameters(
            config=None,
            device: torch.device | None = None,
            seq_len: int | None = None,
            layer_type: str | None = None,
            **rope_kwargs,
        ):
            del seq_len
            if config is None:
                base = rope_kwargs["base"]
                dim = rope_kwargs["dim"]
            else:
                config.standardize_rope_params()
                rope_parameters = config.rope_parameters[layer_type] if layer_type else config.rope_parameters
                base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
                partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
                dim = int(head_dim * partial_rotary_factor)

            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    if not hasattr(Qwen2Config, "rope_theta"):

        def _rope_theta(self):
            rope_parameters = getattr(self, "rope_parameters", None) or {}
            return rope_parameters.get("rope_theta", 10000.0)

        Qwen2Config.rope_theta = property(_rope_theta)

    try:
        from models.qwen2 import Qwen2ForCausalLM
    except ImportError:
        return

    if isinstance(getattr(Qwen2ForCausalLM, "_tied_weights_keys", None), list):
        Qwen2ForCausalLM._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _patch_from_pretrained_without_empty_param_hook(Qwen2ForCausalLM)

    from models.modeling_siglip import SiglipModel

    _patch_from_pretrained_without_empty_param_hook(SiglipModel)

    if not getattr(show_o2_modeling_utils.load_state_dict, "_verl_show_o2_variant_patched", False):
        original_load_state_dict = show_o2_modeling_utils.load_state_dict

        def _load_state_dict_without_variant(*args, **kwargs):
            kwargs.pop("variant", None)
            return original_load_state_dict(*args, **kwargs)

        _load_state_dict_without_variant._verl_show_o2_variant_patched = True
        show_o2_modeling_utils.load_state_dict = _load_state_dict_without_variant


def patch_qwen2_config_compat() -> None:
    patch_show_o2_transformers_compat()


def build_show_o2_runtime(model_path: str | None = None, torch_dtype: torch.dtype | None = None) -> ShowO2Runtime:
    ensure_show_o2_code_path()
    from models.misc import get_text_tokenizer

    config = load_show_o2_config(model_path)
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=_resolve_llm_name(config.model.showo.llm_model_path),
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)
    num_t2i_image_tokens = config.dataset.preprocessing.num_t2i_image_tokens
    max_seq_len = config.dataset.preprocessing.max_seq_length
    max_text_len = max_seq_len - num_t2i_image_tokens - 4
    image_latent_dim = config.model.showo.image_latent_dim
    patch_size = config.model.showo.patch_size
    latent_width = config.dataset.preprocessing.latent_width
    latent_height = config.dataset.preprocessing.latent_height
    pad_id = text_tokenizer.pad_token_id
    bos_id = showo_token_ids["bos_id"]
    eos_id = showo_token_ids["eos_id"]
    boi_id = showo_token_ids["boi_id"]
    eoi_id = showo_token_ids["eoi_id"]
    img_pad_id = showo_token_ids["img_pad_id"]
    guidance_scale = config.transport.guidance_scale
    return ShowO2Runtime(
        config=config,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        num_t2i_image_tokens=int(num_t2i_image_tokens),
        max_seq_len=int(max_seq_len),
        max_text_len=int(max_text_len),
        image_latent_dim=int(image_latent_dim),
        patch_size=int(patch_size),
        latent_width=int(latent_width),
        latent_height=int(latent_height),
        pad_id=int(pad_id),
        bos_id=int(bos_id),
        eos_id=int(eos_id),
        boi_id=int(boi_id),
        eoi_id=int(eoi_id),
        img_pad_id=int(img_pad_id),
        guidance_scale=float(guidance_scale),
        weight_dtype=torch_dtype or get_show_o2_weight_dtype(config),
    )


def build_show_o2_model(model_path: str, torch_dtype: torch.dtype, device: torch.device | str = "cpu"):
    ensure_show_o2_code_path()
    patch_show_o2_transformers_compat()
    from models import Showo2Qwen2_5

    runtime = build_show_o2_runtime(model_path, torch_dtype=torch_dtype)
    config = runtime.config
    if bool(config.model.showo.load_from_showo):
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path,
            use_safetensors=False,
            low_cpu_mem_usage=False,
        )
    else:
        model = Showo2Qwen2_5(**config.model.showo)
        model.load_state_dict(load_show_o2_state_dict(config.model_path))
    model.to(device=device, dtype=torch_dtype)
    model.show_o2_runtime = runtime
    attach_show_o2_fsdp_hints(model)
    attach_show_o2_lora_methods(model)
    return model


def load_show_o2_state_dict(model_path: str):
    if model_path.endswith(".bin"):
        return torch.load(model_path, map_location="cpu")

    checkpoint_files = sorted(str(path) for path in Path(model_path).iterdir() if path.name.endswith(".bin"))
    state_dict = OrderedDict()
    for checkpoint_file in checkpoint_files:
        state_dict.update(torch.load(checkpoint_file, map_location="cpu"))
    return state_dict


def load_hf_state_dict(model_path: str):
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(model_path))

    safetensors_index = model_dir / "model.safetensors.index.json"
    if safetensors_index.is_file():
        with safetensors_index.open() as f:
            weight_map = json.load(f)["weight_map"]
        checkpoint_files = sorted({model_dir / filename for filename in weight_map.values()})
        state_dict = OrderedDict()
        from safetensors.torch import load_file

        for checkpoint_file in checkpoint_files:
            state_dict.update(load_file(str(checkpoint_file), device="cpu"))
        return state_dict

    safetensors_file = model_dir / "model.safetensors"
    if safetensors_file.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensors_file), device="cpu")

    torch_index = model_dir / "pytorch_model.bin.index.json"
    if torch_index.is_file():
        with torch_index.open() as f:
            weight_map = json.load(f)["weight_map"]
        checkpoint_files = sorted({model_dir / filename for filename in weight_map.values()})
        state_dict = OrderedDict()
        for checkpoint_file in checkpoint_files:
            state_dict.update(torch.load(checkpoint_file, map_location="cpu"))
        return state_dict

    torch_file = model_dir / "pytorch_model.bin"
    if torch_file.is_file():
        return torch.load(torch_file, map_location="cpu")

    raise FileNotFoundError(f"Could not find HF model weights in {model_dir}")


def attach_show_o2_fsdp_hints(model):
    model._no_split_modules = ["Qwen2DecoderLayer"]
    return model


def attach_show_o2_lora_methods(model):
    if hasattr(model, "add_adapter") and hasattr(model, "set_adapter"):
        return model

    def add_adapter(self, adapter_config, adapter_name: str = "default") -> None:
        from peft import inject_adapter_in_model

        if not hasattr(self, "peft_config"):
            self.peft_config = {}
        self.peft_config[adapter_name] = adapter_config

        injected_showo = inject_adapter_in_model(adapter_config, self.showo, adapter_name=adapter_name)
        if injected_showo is not None:
            self.showo = injected_showo
        for name, parameter in self.named_parameters():
            parameter.requires_grad = "lora_" in name

    def load_lora_adapter(self, adapter_path: str, adapter_name: str = "default") -> None:
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file as safetensors_load_file

        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.isfile(adapter_config_path):
            raise FileNotFoundError(f"LoRA adapter config not found at {adapter_config_path}")
        if not os.path.isfile(adapter_weights_path):
            raise FileNotFoundError(f"LoRA adapter weights not found at {adapter_weights_path}")

        with open(adapter_config_path) as f:
            adapter_config = lora_config_from_dict(json.load(f))
        self.add_adapter(adapter_config, adapter_name=adapter_name)

        adapter_state_dict = safetensors_load_file(adapter_weights_path)
        current_state = get_peft_model_state_dict(self.showo, adapter_name=adapter_name)
        for key, value in current_state.items():
            if key in adapter_state_dict:
                value.copy_(adapter_state_dict[key])

    def set_adapter(self, adapter_name: str) -> None:
        from peft.tuners.tuners_utils import BaseTunerLayer

        for submodule in self.showo.modules():
            if isinstance(submodule, BaseTunerLayer):
                submodule.set_adapter(adapter_name)

    def disable_adapters(self) -> None:
        from peft.tuners.tuners_utils import BaseTunerLayer

        for submodule in self.showo.modules():
            if isinstance(submodule, BaseTunerLayer):
                submodule.enable_adapters(False)

    def enable_adapters(self) -> None:
        from peft.tuners.tuners_utils import BaseTunerLayer

        for submodule in self.showo.modules():
            if isinstance(submodule, BaseTunerLayer):
                submodule.enable_adapters(True)

    model.add_adapter = MethodType(add_adapter, model)
    model.load_lora_adapter = MethodType(load_lora_adapter, model)
    model.set_adapter = MethodType(set_adapter, model)
    model.disable_adapters = MethodType(disable_adapters, model)
    model.enable_adapters = MethodType(enable_adapters, model)
    return model


def build_show_o2_vae(runtime: ShowO2Runtime, device: torch.device | str):
    ensure_show_o2_code_path()
    from models import WanVAE

    if str(runtime.config.model.vae_model.type) != "wan21":
        raise NotImplementedError(f"Unsupported Show-o2 VAE type={runtime.config.model.vae_model.type!r}")
    return WanVAE(
        vae_pth=runtime.config.model.vae_model.pretrained_model_path,
        dtype=runtime.weight_dtype,
        device=device,
    )


def show_o2_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    return t / (t + shift - shift * t)


def setup_show_o2_sigmas(
    scheduler: FlowMatchSDEDiscreteScheduler,
    num_steps: int,
    shift: float = SHOW_O2_TIMESTEP_SHIFT,
    device: str | torch.device | None = None,
) -> list[float]:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    schedule_points = max(num_steps + 1, 2)
    t = torch.linspace(0, 1, schedule_points, dtype=torch.float32, device=device or "cpu")
    if shift and shift != 1.0:
        t = show_o2_time_shift(float(shift), t)
    sigmas = (1.0 - t[:-1]).tolist()

    scheduler.set_shift(1.0)
    if device is not None:
        scheduler.set_timesteps(sigmas=sigmas, timesteps=sigmas, device=device)
    else:
        scheduler.set_timesteps(sigmas=sigmas, timesteps=sigmas)
    scheduler.set_begin_index(0)
    return sigmas


def show_o2_time_from_sigma(sigma: torch.Tensor) -> torch.Tensor:
    return 1.0 - sigma.float()


def build_attention_mask(
    batch_size: int,
    max_seq_len: int,
    modality_positions: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    ensure_show_o2_code_path()
    from models import omni_attn_mask_naive

    return omni_attn_mask_naive(batch_size, max_seq_len, modality_positions, device).to(dtype)


def prepare_generation_inputs(prompts: list[str], runtime: ShowO2Runtime, device, dtype) -> dict[str, torch.Tensor]:
    ensure_show_o2_code_path()
    from models.misc import prepare_gen_input

    text_tokens, null_text_tokens, modality_positions, null_modality_positions = prepare_gen_input(
        prompts,
        runtime.text_tokenizer,
        runtime.num_t2i_image_tokens,
        runtime.bos_id,
        runtime.eos_id,
        runtime.boi_id,
        runtime.eoi_id,
        runtime.pad_id,
        runtime.img_pad_id,
        runtime.max_text_len,
        device,
    )
    attention_mask = build_attention_mask(len(prompts), runtime.max_seq_len, modality_positions, device, dtype)
    null_attention_mask = build_attention_mask(
        len(prompts),
        runtime.max_seq_len,
        null_modality_positions,
        device,
        dtype,
    )
    return {
        "text_tokens": text_tokens.long(),
        "attention_mask": attention_mask,
        "modality_positions": modality_positions.long(),
        "negative_text_tokens": null_text_tokens.long(),
        "negative_attention_mask": null_attention_mask,
        "negative_modality_positions": null_modality_positions.long(),
    }


def decode_show_o2_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    samples = latents.to(vae.dtype if hasattr(vae, "dtype") else latents.dtype).unsqueeze(2)
    images = vae.batch_decode(samples).squeeze(2)
    return torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)


def copy_weights_into_module(module: torch.nn.Module, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    loaded: set[str] = set()
    for name, tensor in weights:
        target = params.get(name)
        load_name = name
        if target is None and not name.startswith("transformer."):
            load_name = f"transformer.{name}"
            target = params.get(load_name)
        if target is None:
            target = buffers.get(load_name)
        if target is None:
            continue
        with torch.no_grad():
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        loaded.add(load_name)
    return loaded
