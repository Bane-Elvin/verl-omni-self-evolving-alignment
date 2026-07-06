"""Minimal Chameleon VQ decoder used by the Anole OCR GRPO runner.

The installed Transformers Chameleon VQVAE exposes image token encoding only.
Anole publishes a separate HF VQVAE checkpoint with decoder weights, so this
module keeps just the decode path needed for generated image-token sequences.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file


class ChameleonVQVAEVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embed_dim
        self.quant_state_dims = [config.resolution // 2 ** (len(config.channel_multiplier) - 1)] * 2
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def get_codebook_entry(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        batch_size = image_tokens.shape[0]
        emb_dim = self.embedding.weight.shape[-1]
        hidden_state_quant = self.embedding(image_tokens)
        hidden_state_quant = hidden_state_quant.view((batch_size, *self.quant_state_dims, emb_dim))
        return hidden_state_quant.permute(0, 3, 1, 2).contiguous()


class ChameleonVQVAEDecoderConvUpsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)


class ChameleonVQVAEResnetBlock(nn.Module):
    def __init__(self, config, in_channels: int, out_channels: int | None = None, conv_shortcut: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states * torch.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states * torch.sigmoid(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return residual + hidden_states


class ChameleonVQVAEAttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)

        batch_size, channels, height, width = query_states.shape
        query_states = query_states.reshape(batch_size, channels, height * width).permute(0, 2, 1)
        key_states = key_states.reshape(batch_size, channels, height * width)
        attn_weights = torch.bmm(query_states, key_states) * (int(channels) ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=2)

        value_states = value_states.reshape(batch_size, channels, height * width)
        attn_output = torch.bmm(value_states, attn_weights.permute(0, 2, 1))
        attn_output = attn_output.reshape(batch_size, channels, height, width)
        return residual + self.proj_out(attn_output)


class ChameleonVQVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        resolution = config.resolution
        latent_channels = config.latent_channels
        out_channels = config.out_channels

        block_in = base_channels * config.channel_multiplier[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv2d(latent_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ChameleonVQVAEResnetBlock(config=config, in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = ChameleonVQVAEAttnBlock(block_in) if config.attn_type == "vanilla" else nn.Identity()
        self.mid.block_2 = ChameleonVQVAEResnetBlock(config=config, in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = base_channels * config.channel_multiplier[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ChameleonVQVAEResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
                if curr_res in (config.attn_resolutions or []) and config.attn_type == "vanilla":
                    attn.append(ChameleonVQVAEAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = ChameleonVQVAEDecoderConvUpsample(block_in)
                curr_res *= 2
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv_in(hidden_state)
        hidden_state = self.mid.block_1(hidden_state)
        hidden_state = self.mid.attn_1(hidden_state)
        hidden_state = self.mid.block_2(hidden_state)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hidden_state = self.up[i_level].block[i_block](hidden_state)
                if len(self.up[i_level].attn) > 0:
                    hidden_state = self.up[i_level].attn[i_block](hidden_state)
            if i_level != 0:
                hidden_state = self.up[i_level].upsample(hidden_state)

        hidden_state = self.norm_out(hidden_state)
        hidden_state = hidden_state * torch.sigmoid(hidden_state)
        return self.conv_out(hidden_state)


class ChameleonVQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = ChameleonVQVAEDecoder(config)
        self.quantize = ChameleonVQVAEVectorQuantizer(config)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, config.latent_channels, 1)

    def decode(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        expected = self.quantize.quant_state_dims[0] * self.quantize.quant_state_dims[1]
        if image_tokens.shape[1] != expected:
            raise ValueError(f"Expected {expected} image tokens, got {image_tokens.shape[1]}.")
        codebook_entry = self.quantize.get_codebook_entry(image_tokens)
        hidden_states = self.post_quant_conv(codebook_entry)
        return self.decoder(hidden_states)


def load_vqvae(vqvae_path: str | Path, device: torch.device, dtype: torch.dtype) -> ChameleonVQVAE:
    vqvae_path = Path(vqvae_path).expanduser()
    config_path = vqvae_path / "config.json"
    weights_path = vqvae_path / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing VQVAE config: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing VQVAE weights: {weights_path}")

    with config_path.open() as fh:
        config_dict = json.load(fh)
    config = SimpleNamespace(**config_dict)
    model = ChameleonVQVAE(config)
    state_dict = load_file(str(weights_path), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if key.startswith(("decoder.", "quantize.", "post_quant_conv."))]
    if missing or unexpected:
        raise RuntimeError(f"Unexpected VQVAE state dict mismatch. missing={missing}, unexpected={unexpected}")
    model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def pixel_values_to_pil(pixel_values: torch.Tensor, rescale_factor: float = 0.0078) -> list[Image.Image]:
    array = pixel_values.detach().float().cpu().numpy()
    array = np.transpose(array, (0, 2, 3, 1))
    array = np.clip((array + 1.0) * (1.0 / rescale_factor), 0, 255).astype(np.uint8)
    return [Image.fromarray(item) for item in array]
