"""
config.py — single place for every knob.
Swap parse_args() for yaml.safe_load() later without touching other files.
"""
import argparse, os
from datasets import DATASETS


def parse_args():
    p = argparse.ArgumentParser(
        description='Zero-shot CLIP vs vs CoCoOp ')

    # Data
    p.add_argument('--data_root',    default='./data')
    p.add_argument('--dataset',      default='oxford_pets',
                   choices=list(DATASETS.keys()))
    p.add_argument('--backbone',     default='ViT-B/16')
    p.add_argument('--shots',        type=int,   default=16)

    # Training
    p.add_argument('--epochs',       type=int,   default=20)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--n_ctx',        type=int,   default=16)
    p.add_argument('--lr',           type=float, default=2e-3)
    p.add_argument('--hidden_ratio', type=float, default=0.125)
    p.add_argument('--eval_freq',    type=int,   default=10)

    # Hardware
    p.add_argument('--cuda_device',  type=int, default=0)
    p.add_argument('--num_workers',  type=int,
                   default=0 if os.name == 'nt' else 2)

    # Pipeline switches
    p.add_argument('--skip_zeroshot', action='store_true')
    p.add_argument('--skip_cocoop',   action='store_true')
    p.add_argument('--hp_sweep',      action='store_true')
    p.add_argument('--merge_csv',     action='store_true')

    # Visualisation
    p.add_argument('--visualize',     action='store_true',
                   help='Attention heatmap — needs trained CoCoOp weights')
    p.add_argument('--vis_class_idx', type=int, default=0)
    p.add_argument('--vis_image',     default='sample.jpg')

    return p.parse_args()
