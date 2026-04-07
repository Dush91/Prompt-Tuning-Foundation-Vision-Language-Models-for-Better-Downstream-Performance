# Project Index

Complete file guide for DSPy-MIPRO experiments on vision-language prompt learning.

## 📁 Directory Structure

```
experiments/dspy_maple_cupl/
│
├── 📄 README.md                      # Project overview and quick start
├── 📄 REPORT.md                      # Full research report with findings
├── 📄 QUICKSTART.md                 # Step-by-step guide to run experiments
├── 📄 CODE_DOCUMENTATION.md         # Detailed API documentation
├── 📄 INDEX.md                      # This file - complete navigation guide
├── 📄 requirements.txt              # Python dependencies
├── 📄 .gitignore                    # Git ignore patterns
│
├── 📁 code/                         # Source code
│   ├── __init__.py                 # Package initialization
│   ├── descriptor_generator.py     # DSPy signatures for descriptors
│   ├── mipro_optimizer.py          # MIPRO optimization wrapper
│   ├── cupl_baseline.py            # CuPL baseline implementation
│   ├── evaluate_clip_accuracy.py   # Zero-shot CLIP evaluation
│   ├── maple_mipro_init.py         # MaPLe with MIPRO initialization
│   ├── compare_mipro_init.py       # Initialization analysis
│   ├── compare_descriptors.py      # Descriptor quality comparison
│   ├── evaluate.py                 # Generic evaluation utilities
│   └── visualize_results.py        # Result visualization scripts
│
├── 📁 configs/                      # Configuration files
│   └── vit_b16_c2_ep5_batch4_2ctx_mipro.yaml  # MaPLeMIPRO config
│
├── 📁 scripts/                      # Training scripts
│   ├── train_1shot_comparison.sh   # 1-shot MaPLe comparison
│   └── train_4shot_comparison.sh   # 4-shot MaPLe comparison
│
└── 📁 results/                      # Experiment results
    └── mipro_init_comparison.txt  # Initialization metrics
```

## 📖 Documentation Files

### 1. README.md
**Purpose:** Entry point - project overview  
**Read this first** if you're new to the project

**Contents:**
- Project description
- Quick results summary
- Installation instructions
- Basic usage examples
- Citation information

---

### 2. REPORT.md
**Purpose:** Comprehensive research report  
**Read this** to understand the full research context

**Contents:**
- Initial research background
- Methodology details
- Experimental results
- Discussion and analysis
- References

**Key Sections:**
- Section 1: Initial Research
- Section 2: Methodology
- Section 3: Experiments
- Section 4: Key Findings
- Section 5: Discussion

---

### 3. QUICKSTART.md
**Purpose:** Step-by-step execution guide  
**Read this** when you want to run experiments

**Contents:**
- Prerequisites
- Installation steps
- Running each experiment
- Troubleshooting
- Expected outputs

**Experiments Covered:**
1. Zero-shot CLIP comparison
2. Descriptor quality analysis
3. MaPLe initialization study
4. Full MaPLe training

---

### 4. CODE_DOCUMENTATION.md
**Purpose:** API reference  
**Read this** when coding or extending

**Contents:**
- Module descriptions
- Function signatures
- Usage examples
- Data flow diagrams
- Dependencies

**Modules Documented:**
- `descriptor_generator.py`
- `mipro_optimizer.py`
- `cupl_baseline.py`
- `evaluate_clip_accuracy.py`
- `maple_mipro_init.py`
- `compare_mipro_init.py`

---

## 💻 Code Files

### Core Experiment Files

#### `descriptor_generator.py`
**Purpose:** DSPy signatures for descriptor generation  
**Key Classes:**
- `DescriptorGenerator`: Main signature
- `FewShotDescriptorGenerator`: With examples

**Usage:**
```python
from descriptor_generator import DescriptorGenerator
import dspy

predictor = dspy.Predict(DescriptorGenerator)
result = predictor(domain="satellite", class_name="Forest")
```

---

#### `mipro_optimizer.py`
**Purpose:** MIPRO optimization wrapper  
**Key Functions:**
- `optimize_descriptors()`: Main optimization
- `CLIPTextMetric`: Custom metric

**Usage:**
```python
from mipro_optimizer import optimize_descriptors

best_program = optimize_descriptors(
    num_trials=15,
    train_set=train_data,
    val_set=val_data
)
```

---

#### `cupl_baseline.py`
**Purpose:** CuPL baseline implementation  
**Key Functions:**
- `generate_cupl_descriptors()`: Generate 50 descs/class
- `create_cupl_prompts()`: Build CLIP prompts

**Usage:**
```python
from cupl_baseline import generate_cupl_descriptors

descriptors = generate_cupl_descriptors("Forest")
```

---

#### `evaluate_clip_accuracy.py`
**Purpose:** Full zero-shot evaluation  
**Key Functions:**
- `evaluate_clip_accuracy()`: Main evaluation
- `build_text_features()`: Aggregate embeddings

**Run:**
```bash
python evaluate_clip_accuracy.py
```

---

#### `maple_mipro_init.py`
**Purpose:** Modified MaPLe trainer  
**Key Classes:**
- `MaPLeMIPRO`: New trainer
- `MultiModalPromptLearnerMIPRO`: Modified prompt learner

**Usage:**
```bash
python train.py --trainer MaPLeMIPRO ...
```

---

#### `compare_mipro_init.py`
**Purpose:** Initialization analysis  
**Run:**
```bash
python compare_mipro_init.py
```

**Output:**
- L2 norm comparison
- Structure percentage
- Expected accuracy

---

#### `visualize_results.py`
**Purpose:** Generate result plots  
**Run:**
```bash
python visualize_results.py
```

**Generates:**
- `clip_comparison.png`
- `initialization_comparison.png`
- `class_performance.png`

---

## ⚙️ Configuration Files

### `vit_b16_c2_ep5_batch4_2ctx_mipro.yaml`
**Purpose:** MaPLeMIPRO configuration  

**Key Settings:**
```yaml
TRAINER:
  MAPLE:
    N_CTX: 2
    PROMPT_DEPTH: 9
    USE_MIPRO_INIT: True  # Enable MIPRO
```

---

## 🔧 Script Files

### `train_1shot_comparison.sh`
**Purpose:** Compare MaPLe vs MaPLeMIPRO on 1-shot  
**Usage:**
```bash
bash train_1shot_comparison.sh 1  # seed 1
```

**What it does:**
1. Trains standard MaPLe with 1 sample/class
2. Trains MIPRO-MaPLe with 1 sample/class
3. Compares results

---

### `train_4shot_comparison.sh`
**Purpose:** Compare MaPLe vs MaPLeMIPRO on 4-shot  
**Usage:**
```bash
bash train_4shot_comparison.sh 1  # seed 1
```

---

## 📊 Results

### Generated Files

| File | Description | Generated By |
|------|-------------|--------------|
| `mipro_init_comparison.json` | Initialization metrics | `compare_mipro_init.py` |
| `clip_comparison.png` | Bar chart of accuracies | `visualize_results.py` |
| `initialization_comparison.png` | Init quality plots | `visualize_results.py` |
| `class_performance.png` | Per-class accuracy | `visualize_results.py` |

---

## 🚀 Quick Navigation

### I want to...

#### ...understand the research
→ Read **REPORT.md**

#### ...run experiments quickly
→ Follow **QUICKSTART.md**

#### ...implement my own descriptors
→ Check **CODE_DOCUMENTATION.md**

#### ...see the results
→ Look in **results/** folder

#### ...modify configurations
→ Edit **configs/*.yaml**

#### ...add new experiments
→ Create scripts in **scripts/**

---

## 📈 Results Summary

### Zero-Shot CLIP (27,000 test images)

| Method | Accuracy | vs CuPL |
|--------|----------|---------|
| Baseline | 47.97% | +4.96% |
| CuPL | 43.01% | - |
| DSPy Baseline | 45.59% | +2.58% |
| **MIPRO** | **49.33%** | **+6.32%** ✓ |

### MaPLe Initialization (Expected)

| Setting | Standard | MIPRO | Gain |
|---------|----------|-------|------|
| 1-shot | 50% | 60% | +10% |
| 4-shot | 72.5% | 77.5% | +5% |
| 16-shot | 85% | 86.5% | +1.5% |

---

## 📝 Citation

```bibtex
@article{dspy_maple_cupl_2025,
  title={DSPy-MIPRO for Optimizing Vision-Language Prompts},
  year={2025}
}
```

---

## 🔗 Related Files in Main Repository

- `/algorithms/dspy_clip/` - Original DSPy experiments
- `/algorithms/maple/maple_mipro_init.py` - MaPLe modification
- `/configs/maple/vit_b16_c2_ep5_batch4_2ctx_mipro.yaml` - Config
- `/output/eurosat/` - Full experiment outputs

---

**Last Updated:** April 2025  
**Version:** 0.1.0
