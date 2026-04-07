#!/bin/bash

# 4-Shot MaPLe Comparison: Standard vs MIPRO-Initialized
# This script compares standard MaPLe (random init) vs MIPRO-initialized MaPLe
# on EuroSAT with 4 shots per class (40 total training samples)

SEED=$1
if [ -z "$SEED" ]; then
    SEED=1
fi

echo "=========================================="
echo "4-Shot MaPLe Comparison - Seed $SEED"
echo "=========================================="

# Standard MaPLe (random initialization)
echo ""
echo "Training Standard MaPLe (random init)..."
python train.py \
    --root ./data --seed $SEED --trainer MaPLe \
    --dataset-config-file configs/datasets/eurosat.yaml \
    --config-file configs/maple/vit_b16_c2_ep5_batch4_2ctx.yaml \
    --output-dir output/eurosat/shots_4/maple_standard/seed${SEED} \
    DATASET.NUM_SHOTS 4

# MIPRO-initialized MaPLe
echo ""
echo "Training MIPRO-initialized MaPLe..."
python train.py \
    --root ./data --seed $SEED --trainer MaPLeMIPRO \
    --dataset-config-file configs/datasets/eurosat.yaml \
    --config-file configs/maple/vit_b16_c2_ep5_batch4_2ctx_mipro.yaml \
    --output-dir output/eurosat/shots_4/maple_mipro_init/seed${SEED} \
    DATASET.NUM_SHOTS 4

echo ""
echo "=========================================="
echo "Comparison complete for seed $SEED!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Standard MaPLe: output/eurosat/shots_4/maple_standard/seed${SEED}/"
echo "  - MIPRO MaPLe:    output/eurosat/shots_4/maple_mipro_init/seed${SEED}/"
