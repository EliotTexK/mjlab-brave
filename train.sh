#!/usr/bin/bash
source .venv/bin/activate
WANDB_MODE=disabled uv run train Mjlab-Distill-LocoManip-BRAVE-G1 --gpu-ids all

