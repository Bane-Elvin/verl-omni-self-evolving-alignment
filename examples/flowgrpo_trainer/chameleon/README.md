# Chameleon OCR GRPO

This directory is a preflight placeholder for Chameleon-family OCR image-generation
RL. The local `/home/elvin/Models/chameleon-7b` checkpoint cannot be used for the
same OCR loop as BAGEL or Janus because it does not expose image output decoding.

Observed blockers:

- The local model card declares `pipeline_tag: image-text-to-text`.
- `ChameleonForConditionalGeneration.forward()` masks image-token logits.
- `ChameleonVQVAE` in Transformers has no `decode` or `decode_code` method.
- The local safetensors index has VQ encoder and quantizer weights, but no
  decoder image weights.

Run the preflight check:

```bash
cd /home/elvin/Projects/verl-omni-self-evolving-alignment
MODEL_PATH=/home/elvin/Models/chameleon-7b \
bash examples/flowgrpo_trainer/chameleon/run_chameleon_ocr_lora_grpo.sh
```

To train OCR with this folder, first provide a Chameleon-family checkpoint that
can generate image tokens and decode those tokens into images. Then this folder
can be extended with an AR-GRPO runner following the Janus implementation.
