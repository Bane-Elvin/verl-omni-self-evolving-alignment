# Chameleon OCR GRPO

This directory contains a standalone AR-GRPO OCR runner for Chameleon-family
image-token generation. The default checkpoint is Anole 7B because the public
`facebook/chameleon-7b` Transformers path masks image-token logits and does not
ship an image decoder.

Expected local paths:

```bash
/home/elvin/Models/Anole-7b-v0.1-hf
/home/elvin/Models/Anole-7b-v0.1-vqvae-hf
```

Single run:

```bash
cd /home/elvin/Projects/verl-omni-self-evolving-alignment

WANDB_PROJECT=verl-chameleon \
MODEL_PATH=/home/elvin/Models/Anole-7b-v0.1-hf \
VQVAE_PATH=/home/elvin/Models/Anole-7b-v0.1-vqvae-hf \
TRAIN_BATCH_SIZE=4 \
PPO_MINI_BATCH_SIZE=4 \
ROLLOUT_N=4 \
TOTAL_TRAINING_STEPS=43 \
NUM_GPUS_ACTOR_ROLLOUT_REWARD=4 \
bash examples/flowgrpo_trainer/chameleon/run_chameleon_ocr_lora_grpo.sh
```

Sweep matching the BAGEL/Janus sample-count settings:

```bash
cd /home/elvin/Projects/verl-omni-self-evolving-alignment

WANDB_PROJECT=verl-chameleon \
MODEL_PATH=/home/elvin/Models/Anole-7b-v0.1-hf \
VQVAE_PATH=/home/elvin/Models/Anole-7b-v0.1-vqvae-hf \
TRAIN_NUM_GPUS=4 \
TRAIN_GPU_POOL=0,1,2,3 \
SWEEP_SPECS=$'1 16 172\n1 8 172\n1 4 172\n2 8 86\n2 4 86\n4 4 43' \
bash examples/flowgrpo_trainer/chameleon/run_chameleon_ocr_lora_grpo_sweep.sh
```

The runner logs metrics under the same high-level groups as the Janus runner:
`actor/*`, `critic/*`, `perf/*`, `timing_s/*`, and `timing_per_image_ms/*`.
