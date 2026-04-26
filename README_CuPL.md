# CuPL Prompt Evaluation with CLIP

This module implements the CuPL-style prompt evaluation for CLIP.

## Methods
1. Handcrafted CLIP prompts
2. CuPL-style descriptive prompts

## Model
CLIP ViT-B/16

## Datasets
- EuroSAT
- DTD
- Caltech101
- Oxfordpets
- flowers102

## Commands

Generate prompts:

```bash
python generate_cupl_prompts.py --dataset eurosat
python generate_cupl_prompts.py --dataset dtd
python generate_cupl_prompts.py --dataset caltech101
python generate_cupl_prompts.py --dataset oxfordpets
python generate_cupl_prompts.py --dataset flowers102
