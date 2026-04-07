# Quick Start Guide

This guide will help you run the DSPy-MIPRO experiments from scratch.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenAI API key (for DSPy-MIPRO)

## Installation

### 1. Clone and Setup

```bash
# Navigate to experiments folder
cd experiments/dspy_maple_cupl

# Install dependencies
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key

```bash
export OPENAI_API_KEY="your-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

## Running Experiments

### Experiment 1: Zero-Shot CLIP Comparison

Compare CuPL vs DSPy-MIPRO on zero-shot classification:

```bash
cd code
python evaluate_clip_accuracy.py
```

**Expected Output:**
```
============================================================
CLIP ZERO-SHOT EVALUATION ON EUROSAT
============================================================
Baseline Accuracy: 47.97%
CuPL Accuracy: 43.01%
MIPRO Accuracy: 49.33%

✅ MIPRO beats CuPL by +6.32%!
```

### Experiment 2: Compare Descriptor Quality

Analyze the generated descriptors:

```bash
cd code
python compare_descriptors.py
```

This will compare:
- Domain relevance
- Class distinctiveness
- Descriptor diversity

### Experiment 3: MaPLe Initialization Analysis

Analyze initialization impact for few-shot learning:

```bash
cd code
python compare_mipro_init.py
```

**Expected Output:**
```
Standard (random):
  Mean L2 norm: 0.449
  Structured dimensions: 1.4%

MIPRO-init:
  Mean L2 norm: 3.430
  Structured dimensions: 65.0%

Key Differences:
  L2 norm: 0.449 → 3.430 (+663.4%)
  Structure: 1.4% → 65.0% (+46×)
```

### Experiment 4: Run MaPLe Training (Optional)

If you have the EuroSAT dataset:

```bash
# 1-shot experiment
bash scripts/train_1shot_comparison.sh 1

# 4-shot experiment
bash scripts/train_4shot_comparison.sh 1
```

**Note:** Requires EuroSAT dataset in `../../data/eurosat/`

## Customization

### Generate Custom Descriptors

```python
from code.descriptor_generator import DescriptorGenerator
from code.mipro_optimizer import optimize_descriptors
import dspy

# Configure DSPy
lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

# Generate descriptors for your classes
class_names = ["YourClass1", "YourClass2", "YourClass3"]

for class_name in class_names:
    predictor = dspy.Predict(DescriptorGenerator)
    result = predictor(
        domain="your domain description",
        class_name=class_name
    )
    print(f"\n{class_name}:")
    print(result.descriptors)
```

### Optimize for Your Domain

```python
from code.mipro_optimizer import optimize_descriptors

# Prepare training data
train_data = [
    dspy.Example(
        domain="your domain",
        class_name=cls,
        descriptors="..."
    ).with_inputs("domain", "class_name")
    for cls in class_names
]

# Run optimization
best_program = optimize_descriptors(
    num_trials=15,
    population_size=10,
    train_set=train_data
)

# Save optimized descriptors
best_program.save("optimized_descriptors.json")
```

## Expected Results Summary

| Experiment | Metric | Expected Value |
|-----------|--------|----------------|
| Zero-shot CLIP | Accuracy | MIPRO: 49.33% |
| vs CuPL | Improvement | +6.32% |
| MaPLe 1-shot | Expected Accuracy | 55-65% |
| vs Standard | Gain | +10% |
| MaPLe 4-shot | Expected Accuracy | 75-80% |
| vs Standard | Gain | +5% |

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'clip'`

**Solution:**
```bash
pip install git+https://github.com/openai/CLIP.git
```

### Issue: `ModuleNotFoundError: No module named 'dassl'`

**Solution:**
```bash
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch
pip install -e .
```

### Issue: OpenAI API errors

**Solution:**
- Check your API key: `echo $OPENAI_API_KEY`
- Ensure you have sufficient credits
- Try using a different model (e.g., `gpt-3.5-turbo` instead of `gpt-4o`)

### Issue: CUDA out of memory

**Solution:**
Edit the config file:
```yaml
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 2  # Reduce from 4
```

## Next Steps

1. **Read the full report:** See `REPORT.md` for detailed research findings
2. **Explore code documentation:** See `CODE_DOCUMENTATION.md` for API details
3. **Modify configs:** Adjust hyperparameters in `configs/`
4. **Run on your data:** Adapt scripts for your dataset

## Citation

If you use this code in your research:

```bibtex
@article{dspy_maple_cupl_2025,
  title={DSPy-MIPRO for Optimizing Vision-Language Prompts},
  author={Your Name},
  year={2025}
}
```

## Support

For issues or questions:
1. Check the documentation in `CODE_DOCUMENTATION.md`
2. Review the troubleshooting section above
3. Open an issue in the repository
