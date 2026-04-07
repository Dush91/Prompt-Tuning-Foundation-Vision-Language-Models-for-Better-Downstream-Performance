# Code Documentation

## Overview

This directory contains code for experiments combining DSPy-MIPRO, CuPL, and MaPLe for satellite imagery classification.

## File Descriptions

### `descriptor_generator.py`

**Purpose:** DSPy signatures for generating class descriptors.

**Key Classes:**
- `DescriptorGenerator`: Base signature for descriptor generation
  - Input: `domain`, `class_name`
  - Output: `descriptors` (5 detailed descriptions)

**Usage:**
```python
from descriptor_generator import DescriptorGenerator
import dspy

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

generator = dspy.Predict(DescriptorGenerator)
result = generator(domain="satellite remote sensing imagery", class_name="Forest")
print(result.descriptors)
```

---

### `mipro_optimizer.py`

**Purpose:** Wrapper for DSPy MIPROv2 optimizer to find best meta-prompts.

**Key Components:**
- `optimize_descriptors()`: Main optimization function
- `CLIPTextMetric`: Custom metric using CLIP embeddings
- `evaluate_descriptor_quality()`: Text-based quality scoring

**Usage:**
```python
from mipro_optimizer import optimize_descriptors

best_program = optimize_descriptors(
    num_trials=15,
    population_size=10,
    train_set=train_data,
    val_set=val_data
)
```

**Configuration:**
- `auto="light"`: Lightweight optimization mode
- No `num_trials` when using auto mode (MIPRO determines automatically)

---

### `cupl_baseline.py`

**Purpose:** Implementation of CuPL (Customized Prompts via Language models) baseline.

**Key Functions:**
- `generate_cupl_descriptors()`: Generate 50 descriptors/class using 5 meta-prompts
- `create_cupl_prompts()`: Build CLIP-compatible text prompts

**Meta-Prompts Used:**
1. Visual appearance (aesthetic qualities)
2. Location/context (where found)
3. Function/purpose (what used for)
4. Visual attributes (shape, color, size)
5. Habitat/typical environment

**Usage:**
```python
from cupl_baseline import generate_cupl_descriptors

descriptors = generate_cupl_descriptors(class_name="Forest")
# Returns 50 descriptor strings
```

---

### `evaluate_clip_accuracy.py`

**Purpose:** Full CLIP evaluation comparing all methods on EuroSAT.

**Methods Evaluated:**
1. Baseline (1 template: "a photo of a {class}")
2. CuPL (50 descriptors/class)
3. DSPy Baseline (10 descriptors/class)
4. MIPRO Optimized (10 descriptors/class)

**Key Functions:**
- `load_mipro_descriptors()`: Load optimized descriptors from JSON
- `evaluate_clip_accuracy()`: Main evaluation function
- `build_text_features()`: Aggregate text embeddings with averaging

**Output:**
```
============================================================
CLIP ZERO-SHOT EVALUATION ON EUROSAT
============================================================
Baseline Accuracy: 47.97%
CuPL Accuracy: 43.01%
DSPy Baseline Accuracy: 45.59%
MIPRO Accuracy: 49.33%

✅ MIPRO beats CuPL by +6.32%!
```

---

### `maple_mipro_init.py`

**Purpose:** Modified MaPLe trainer with MIPRO-optimized initialization.

**Key Changes from Standard MaPLe:**

1. `MultiModalPromptLearnerMIPRO`: Modified prompt learner
   - Uses domain keywords for initialization instead of random
   - Keywords: "satellite imagery aerial view spectral"

2. `MaPLeMIPRO`: New trainer class
   - Registered as `MaPLeMIPRO` in Dassl registry
   - Inherits from standard MaPLe

**Initialization Comparison:**

```python
# Standard MaPLe
ctx_vectors = torch.empty(n_ctx, ctx_dim)
nn.init.normal_(ctx_vectors, std=0.02)

# MIPRO-MaPLe
mipro_text = "satellite imagery aerial view spectral"
ctx_vectors = CLIP_token_embedding(mipro_text)
```

**Usage:**
```bash
python train.py \
    --trainer MaPLeMIPRO \
    --config-file configs/maple/vit_b16_c2_ep5_batch4_2ctx_mipro.yaml \
    DATASET.NUM_SHOTS 1
```

**Config Option:**
```yaml
TRAINER:
  MAPLE:
    N_CTX: 2
    USE_MIPRO_INIT: True  # Enable MIPRO initialization
```

---

### `compare_mipro_init.py`

**Purpose:** Analysis comparing standard vs MIPRO initialization for MaPLe.

**Analysis Performed:**
1. Context vector statistics (L2 norm, structure percentage)
2. Initialization similarity metrics
3. Expected accuracy comparison

**Key Metrics:**
- **L2 Norm:** Magnitude of context vectors
- **Structure:** Percentage of meaningful dimensions
- **Inter-vector Similarity:** How related the context vectors are

**Output:**
```
Standard (random):
  Mean L2 norm: 0.4493
  Structured dimensions: 1.4%

MIPRO-init:
  Mean L2 norm: 3.4300
  Structured dimensions: 65.0%

Key Differences:
  L2 norm: 0.449 → 3.430 (+663.4%)
  Structure: 1.4% → 65.0% (+46×)
```

---

## Configuration Files

### `vit_b16_c2_ep5_batch4_2ctx_mipro.yaml`

Configuration for MaPLeMIPRO trainer:

```yaml
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 4
  TEST:
    BATCH_SIZE: 100

OPTIM:
  NAME: "sgd"
  LR: 0.0035
  MAX_EPOCH: 5
  LR_SCHEDULER: "cosine"

TRAINER:
  MAPLE:
    N_CTX: 2
    PROMPT_DEPTH: 9
    USE_MIPRO_INIT: True  # Key difference from standard config
```

---

## Scripts

### `train_1shot_comparison.sh`

Compares Standard MaPLe vs MIPRO-MaPLe on 1-shot setting (10 samples).

```bash
bash scripts/train_1shot_comparison.sh 1  # Seed 1
```

### `train_4shot_comparison.sh`

Compares Standard MaPLe vs MIPRO-MaPLe on 4-shot setting (40 samples).

```bash
bash scripts/train_4shot_comparison.sh 1  # Seed 1
```

---

## Data Flow

### Zero-Shot Evaluation Pipeline:

```
EuroSAT Images
    ↓
CLIP Image Encoder → Image Features
    ↓
                  ← Text Descriptors (CuPL/MIPRO)
Text Encoder ← ← ← ← ← ← ← ← ↑
    ↓                           ↑
Text Features (averaged) ← ← ← ←
    ↓
Cosine Similarity
    ↓
Classification Logits
    ↓
Accuracy Calculation
```

### MaPLe Training Pipeline:

```
Training Images
    ↓
CLIP Visual Encoder + Learned Visual Prompts → Image Features
    ↓
CLIP Text Encoder + Learned Text Prompts ← Class Names
    ↓
Cosine Similarity with Temperature
    ↓
Cross-Entropy Loss
    ↓
Update Prompt Parameters (CLIP frozen)
```

---

## Dependencies

```
torch>=1.7.1
torchvision>=0.8.2
dassl>=0.5.0
dspy-ai
ftfy
regex
tqdm
yacs
numpy
pillow
```

---

## Troubleshooting

### Issue: ImportError with clip module

**Solution:** Ensure project root is in PYTHONPATH:
```bash
export PYTHONPATH=/teamspace/studios/this_studio:$PYTHONPATH
```

### Issue: Dassl not found

**Solution:** Install from source:
```bash
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch
pip install -e .
```

### Issue: CUDA out of memory

**Solution:** Reduce batch size in config:
```yaml
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 2  # Instead of 4
```

---

## Notes

- All experiments use EuroSAT dataset (10 classes, 27,000 images)
- CLIP ViT-B/16 is the base model
- MIPRO optimization requires OpenAI API key
- MaPLe training requires GPU with 8GB+ memory
