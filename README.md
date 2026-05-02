# CLIP LoRA Fine-tuning — EEEM068 Applied Machine Learning

> Parameter-efficient fine-tuning of CLIP ViT-B/16 using LoRA (Low-Rank Adaptation)  
> on Caltech-101 and EuroSAT with generalisation tests on Oxford Pets.

---

## Overview

This repository implements LoRA applied to CLIP's ViT-B/16 vision encoder for downstream 
image classification. Two classification approaches are compared — cosine similarity 
(CLIP-style) and a linear classifier head — across multiple LoRA ranks to analyse the 
parameter efficiency vs accuracy tradeoff. Training is performed on two datasets 
(Caltech-101 and EuroSAT) with cross-dataset generalisation evaluation.

---

## Methods Implemented

- **Zero-shot CLIP** — baseline using hand-crafted prompts
- **LoRA + Cosine Similarity** — low-rank adaptation with CLIP text prototypes
- **LoRA + Classifier Head** — low-rank adaptation with linear classification layer
- Generalisation evaluation on unseen datasets (Oxford Pets, EuroSAT, Caltech-101)

---

## Results

### Caltech-101 Training — Cosine Similarity (Rank Sweep)

| Method | Params | Caltech Test | Oxford Pets | EuroSAT |
|---|---|---|---|---|
| Zero-shot CLIP | 0 | 84.65% | 84.51% | 43.84% |
| LoRA r=2 | 73,728 (0.04%) | 94.93% | 71.60% | 21.87% |
| LoRA r=4 | 147,456 (0.09%) | 95.40% | 72.26% | 19.33% |
| LoRA r=8 | 294,912 (0.18%) | 94.78% | 73.42% | 17.19% |
| LoRA r=16 | 589,824 (0.36%) | 95.47% | 73.12% | 17.97% |

### EuroSAT Training — Cosine Similarity (Rank Sweep)

| Method | Params | EuroSAT Test | Oxford Pets | Caltech-101 |
|---|---|---|---|---|
| Zero-shot CLIP | 0 | 43.84% | 84.51% | 85.96% |
| LoRA r=2 | 73,728 (0.04%) | 98.00% | 40.92% | 34.61% |
| LoRA r=4 | 147,456 (0.09%) | 97.65% | 43.80% | 33.15% |
| LoRA r=8 | 294,912 (0.18%) | 97.46% | 45.24% | 33.77% |
| LoRA r=16 | 589,824 (0.36%) | 98.10% | 47.45% | 41.90% |

### Cosine vs Classifier Head Comparison

| Dataset | Method | Test Acc | Pets | Generalisation |
|---|---|---|---|---|
| Caltech | LoRA r=4 + Cosine | 95.40% | 72.26% | Good |
| Caltech | LoRA r=4 + Classifier | 61.09% | 17.83% | Poor |
| EuroSAT | LoRA r=4 + Cosine | 97.65% | 43.80% | Moderate |
| EuroSAT | LoRA r=4 + Classifier | 97.75% | 6.82% | Very Poor |

---

## Key Findings

**1. LoRA dramatically improves in-domain accuracy:**
- Caltech: +10% gain over zero-shot (84.65% → 95.40%)
- EuroSAT: +54% gain over zero-shot (43.84% → 98.00%)
- EuroSAT benefits more because zero-shot CLIP struggles with satellite imagery

**2. Rank sweep — r=2 is the most parameter efficient:**
- r=2 achieves 94.93% on Caltech with only 73,728 params (0.04%)
- Higher ranks (r=8, r=16) add parameters but do not consistently improve accuracy
- r=16 gives marginal gains on EuroSAT generalisation (47.45% Pets vs 40.92% for r=2)

**3. Cosine similarity vastly outperforms classifier head for generalisation:**
- Caltech classifier: Pets drops from 84.51% to 17.83% (-66%)
- EuroSAT classifier: Pets drops from 84.51% to 6.82% (-77%)
- Cosine preserves CLIP's pretrained knowledge; classifier destroys it

**4. Domain gap matters:**
- LoRA trained on Caltech performs poorly on EuroSAT (19%) — natural vs satellite images
- LoRA trained on EuroSAT performs poorly on Caltech (33%) — satellite vs natural images
- Oxford Pets drop is smaller when trained on Caltech as it shares the natural image domain

---

## t-SNE Visualisation

### Caltech-101
Both zero-shot and LoRA fine-tuned CLIP show similar feature separation on Caltech. 
The improvement is subtle because CLIP already has strong zero-shot representations 
(84.65%). LoRA refines rather than dramatically restructures the feature space, 
consistent with the moderate +10% accuracy gain.

### EuroSAT — Most Striking Result
The EuroSAT t-SNE plots provide the most compelling visual evidence of LoRA's 
effectiveness:

- **Zero-shot CLIP (left):** All 10 classes heavily overlapping and scattered — 
  consistent with the poor 43.84% zero-shot accuracy. CLIP was not trained on 
  satellite imagery so features are not discriminative.

- **LoRA + Cosine (right):** All 10 classes form tight, completely separated clusters 
  with clear empty space between them — directly explaining the 98% test accuracy.

- **LoRA + Classifier (right):** Fewer distinct clusters despite similar accuracy, 
  explaining the catastrophic generalisation failure (6.82% on Pets). The classifier 
  memorises class patterns without maintaining transferable representations.

This contrast between cosine and classifier t-SNE plots is the strongest visual 
evidence that classification head choice matters as much as the fine-tuning method itself.

---

## Project Structure

```
clip_lora/
├── main.py           # entry point with argparse
├── data.py           # dataset loading (Caltech101, Oxford Pets, EuroSAT)
├── model.py          # LoRA layer definition and injection into CLIP
├── train.py          # training loop with early stopping
├── test.py           # testing, generalisation evaluation and t-SNE
├── config.py         # default hyperparameters
└── requirements.txt  # dependencies
```

---

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

## Usage

```bash
# Train on Caltech-101 with cosine similarity
python3 main.py --rank 4 --method cosine --dataset caltech

# Train on EuroSAT
python3 main.py --rank 4 --method cosine --dataset eurosat

# Train with classifier head
python3 main.py --rank 4 --method classifier --dataset caltech

# Rank sweep on Caltech
python3 main.py --rank 2  --method cosine --dataset caltech
python3 main.py --rank 4  --method cosine --dataset caltech
python3 main.py --rank 8  --method cosine --dataset caltech
python3 main.py --rank 16 --method cosine --dataset caltech

# With t-SNE visualisation
python3 main.py --rank 4 --method cosine --tsne

# Custom hyperparameters
python3 main.py --rank 4 --method cosine --lr 1e-5 --epochs 20 --dropout 0.1
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--rank` | 4 | LoRA rank r (2, 4, 8, 16) |
| `--method` | cosine | cosine similarity or classifier head |
| `--dataset` | caltech | caltech or eurosat |
| `--epochs` | 10 | maximum training epochs |
| `--lr` | 1e-4 | learning rate |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--patience` | 3 | early stopping patience |
| `--dropout` | 0.0 | LoRA dropout |
| `--batch_size` | 32 | batch size |
| `--tsne` | False | generate t-SNE visualisation |
| `--data_root` | ./data | path to dataset root |

---

## How LoRA Works

LoRA injects low-rank matrices A and B into the Q and V projections
of every attention block in CLIP's vision encoder:

```
output = xW  +  x(AB) × (alpha/rank)
          ↑          ↑
       frozen      only A and B
       CLIP        are trained
```

- **A** is initialised randomly, **B** is initialised to zero
- So A@B = 0 at start — model behaves identically to original CLIP
- Only 0.04%–0.36% of total parameters are trained depending on rank

---

## References

1. Radford et al. "Learning Transferable Visual Models from Natural Language Supervision" (CLIP), ICML 2021
2. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022

---


