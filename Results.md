================================================================================
EuroSAT 16-Shot Comparison: CoOp vs CoCoOp vs MaPLe vs Zero-shot CLIP
================================================================================

Dataset: EuroSAT (satellite imagery, 10 classes)
Test Set: 8,100 samples
Training: 16 samples per class (160 total)
Backbone: ViT-B/16

================================================================================
RESULTS
================================================================================

| Method          | Seed 1 | Seed 2 | Seed 3 |  Mean  | Std Dev |
|-----------------|--------|--------|--------|--------|---------|
| MaPLe           | 85.7%  | 86.6%  | 82.7%  | 85.0%  | 2.0%    |
| CoOp            | 74.2%  | 73.4%  | 78.3%  | 75.3%  | 2.6%    |
| CoCoOp          | 67.3%  | 76.6%  | 68.1%  | 70.7%  | 5.0%    |
| Zero-shot CLIP  |   -    |   -    |   -    | 48.4%  |   -     |

================================================================================
KEY FINDINGS
================================================================================

1. MaPLe dominates on EuroSAT (+9.7% over CoOp, +14.3% over CoCoOp)
   - Mean: 85.0% (best performing method)
   - Most stable (lowest variance: 2.0%)
   - Multi-modal prompting is crucial for satellite imagery

2. CoOp outperforms CoCoOp (+4.6%)
   - Class-shared context works better than instance-conditional on EuroSAT
   - More stable training (lower variance)

3. All prompt learning methods significantly improve over zero-shot (+36.6% for MaPLe)

4. EuroSAT is challenging for standard VLM adaptation:
   - Requires domain-specific prompt learning
   - Multi-modal (vision + language) prompting essential

================================================================================
CONFIGURATION DETAILS
================================================================================

MaPLe:
  - Context tokens: 2
  - Prompt depth: 9 (multi-modal)
  - Epochs: 5
  - Batch size: 4
  - LR: 0.0035

CoOp:
  - Context tokens: 4
  - Epochs: 10
  - Batch size: 1
  - LR: 0.002

CoCoOp:
  - Context tokens: 4
  - Epochs: 10
  - Batch size: 1
  - LR: 0.002
  - Meta network: 2-layer MLP

Zero-shot:
  - Hand-crafted prompts: "a centered satellite photo of {class}"

================================================================================
CHECKPOINTS
================================================================================

output/eurosat/shots_16/
├── MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed{1,2,3}/
├── CoOp/vit_b16_c4_ep10_batch1/seed{1,2,3}/
├── CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/seed{1,2,3}/
└── zeroshot_clip/

================================================================================
CITATION
================================================================================

CoOp:     Learning to Prompt for Vision-Language Models (Zhou et al., IJCV 2022)
CoCoOp:   Conditional Prompt Learning for Vision-Language Models (Zhou et al., CVPR 2022)
MaPLe:    MaPLe: Multi-modal Prompt Learning (Khattak et al., CVPR 2023)

Date: 2026-03-27
================================================================================
