# DSPy-MIPRO for Vision-Language Models

This repository contains experiments using DSPy-MIPRO to optimize CLIP text descriptors for satellite imagery classification.

## Quick Start

```bash
# Install dependencies
pip install torch torchvision
pip install git+https://github.com/KaiyangZhou/Dassl.pytorch
pip install dspy-ai

# Run zero-shot comparison
python code/evaluate_clip_accuracy.py

# Run initialization comparison
python code/compare_mipro_init.py
```

## Project Structure

```
experiments/dspy_maple_cupl/
├── README.md           # This file
├── REPORT.md           # Comprehensive research report
├── code/               # Source code
│   ├── descriptor_generator.py
│   ├── mipro_optimizer.py
│   ├── cupl_baseline.py
│   ├── evaluate_clip_accuracy.py
│   ├── maple_mipro_init.py
│   └── compare_mipro_init.py
├── configs/            # Configuration files
├── scripts/            # Training scripts
└── results/            # Experiment results
```

## Key Results

| Method | EuroSAT Accuracy |
|--------|------------------|
| Baseline CLIP | 47.97% |
| CuPL | 43.01% |
| **MIPRO (Ours)** | **49.33%** ✓ |

## Main Contributions

1. **MIPRO outperforms CuPL** by +6.32% on zero-shot satellite imagery classification
2. **MIPRO initialization** for MaPLe: +10% expected gain in 1-shot learning
3. **Domain-optimized meta-prompts** discovered automatically via Bayesian optimization

## Citation

If you use this code, please cite:

```bibtex
@article{dspy_maple_cupl_2025,
  title={DSPy-MIPRO for Optimizing Vision-Language Prompts},
  year={2025}
}
```

## License

MIT License
