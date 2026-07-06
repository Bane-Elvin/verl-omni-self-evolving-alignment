#!/usr/bin/env python
"""Check whether a Chameleon checkpoint can support OCR image-generation RL."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from transformers import AutoConfig, AutoProcessor
from transformers.models.chameleon.modeling_chameleon import (
    ChameleonForConditionalGeneration,
    ChameleonVQVAE,
)


def load_weight_keys(model_path: Path) -> list[str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        return []
    with index_path.open() as fh:
        index = json.load(fh)
    return sorted(index.get("weight_map", {}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/home/elvin/Models/chameleon-7b")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise SystemExit(f"Missing Chameleon model path: {model_path}")

    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    weight_keys = load_weight_keys(model_path)
    forward_source = inspect.getsource(ChameleonForConditionalGeneration.forward)

    has_vq_decode = any(hasattr(ChameleonVQVAE, name) for name in ("decode", "decode_code"))
    masks_image_logits = "logits[:, :, image_tokens]" in forward_source and "torch.finfo" in forward_source
    decoder_weight_keys = [key for key in weight_keys if "decoder" in key.lower() or "decode" in key.lower()]

    print(f"model_path: {model_path}")
    print(f"model_type: {getattr(config, 'model_type', None)}")
    print(f"architectures: {getattr(config, 'architectures', None)}")
    print(f"processor: {type(processor).__name__}")
    print(f"image_token: {getattr(processor, 'image_token', None)}")
    print(f"vq_decode_method: {has_vq_decode}")
    print(f"masks_image_logits: {masks_image_logits}")
    print(f"decoder_weight_keys: {len(decoder_weight_keys)}")

    issues: list[str] = []
    if not has_vq_decode:
        issues.append("Transformers ChameleonVQVAE has no decode/decode_code method.")
    if masks_image_logits:
        issues.append("ChameleonForConditionalGeneration masks image-token logits during forward().")
    if weight_keys and not decoder_weight_keys:
        issues.append("The local safetensors index contains no decoder image weights.")

    if issues:
        print("\nChameleon OCR image-generation RL is not supported by this checkpoint:")
        for issue in issues:
            print(f"- {issue}")
        print(
            "\nUse a Chameleon-family checkpoint that exposes image-token generation and "
            "image decoding, then add the AR-GRPO runner here."
        )
        return 1

    print("\nThis checkpoint exposes the pieces needed for an OCR image-generation runner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
