#!/bin/bash

# 1-Shot MaPLe Comparison: Standard vs MIPRO-Initialized
# This is the extreme low-data regime (10 total samples) where initialization matters most

SEED=$1
if [ -z "$SEED" ]; then
    SEED=1
fi

echo "=========================================="
echo "1-Shot MaPLe Comparison - Seed $SEED"
echo "Training samples: 10 (1 per class)"
echo "=========================================="

# Standard MaPLe (random initialization) - 1 shot
echo ""
echo "Training Standard MaPLe (random init) with 1-shot..."
/home/zeus/miniconda3/envs/cloudspace/bin/python3 train.py \
    --root ./data --seed $SEED --trainer MaPLe \
    --dataset-config-file configs/datasets/eurosat.yaml \
    --config-file configs/maple/vit_b16_c2_ep5_batch4_2ctx.yaml \
    --output-dir output/eurosat/shots_1/maple_standard/seed${SEED} \
    DATASET.NUM_SHOTS 1

# MIPRO-initialized MaPLe - 1 shot
echo ""
echo "Training MIPRO-initialized MaPLe with 1-shot..."
/home/zeus/miniconda3/envs/cloudspace/bin/python3 train.py \
    --root ./data --seed $SEED --trainer MaPLeMIPRO \
    --dataset-config-file configs/datasets/eurosat.yaml \
    --config-file configs/maple/vit_b16_c2_ep5_batch4_2ctx_mipro.yaml \
    --output-dir output/eurosat/shots_1/maple_mipro_init/seed${SEED} \
    DATASET.NUM_SHOTS 1

echo ""
echo "=========================================="
echo "Comparison complete for seed $SEED!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Standard MaPLe: output/eurosat/shots_1/maple_standard/seed${SEED}/"
echo "  - MIPRO MaPLe:    output/eurosat/shots_1/maple_mipro_init/seed${SEED}/"
