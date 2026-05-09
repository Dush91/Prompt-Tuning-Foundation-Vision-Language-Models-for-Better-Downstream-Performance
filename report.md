# Vision-Language Prompt Learning
## Comparing Zero-Shot CLIP vs CoCoOp on Image Classification

---

## 1. Introduction

Large vision-language models like CLIP achieve strong zero-shot classification by matching image embeddings against text embeddings of class descriptions. However, performance is sensitive to the exact wording of prompts — a problem known as **prompt sensitivity**.

This project investigates whether **learned prompt tuning** can close the gap between hand-engineered prompts and optimal text representations, using only a small number of labelled examples (16-shot).

**Methods compared:**

| Method | What it learns | Parameters trained |
|--------|---------------|-------------------|
| ZS-handcrafted | ensemble of 80 fixed templates | 0 |
| ZS-LLM | 7 task-specific prompts | 0 |
| CoCoOp | image-conditioned context tokens via MetaNet | ~ctx_dim + MetaNet |

---

## 2. Setup

**Backbone:** ViT-B/16 (CLIP, frozen throughout)  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4.3 GB VRAM)  
**Few-shot budget:** 16 labelled examples per class  
**Training:** 20 epochs, SGD with cosine LR + linear warmup, AMP mixed precision  
**Context tokens:** n_ctx = 16

**Datasets:**

| Dataset | Classes | Test size | Domain |
|---------|---------|-----------|--------|
| Oxford Pets | 37 | 3,669 | Fine-grained pet breeds |
| EuroSAT | 10 | 27,000 | Satellite / aerial imagery |
| DTD | 47 | 1,880 | Texture classification |

---

## 3. Method Details

### 3.1 Zero-Shot CLIP

CLIP classifies images by computing cosine similarity between the image embedding and text embeddings of each class name wrapped in a template (e.g. `"a photo of a {class}"`).

**ZS-handcrafted** ensembles 80 generic ImageNet templates.  
**ZS-LLM** uses 7 domain-specific prompts written to match each dataset's visual characteristics (e.g. `"a satellite image of {}"` for EuroSAT).

### 3.2 CoCoOp (Conditional Context Optimisation)

CoCoOp extends CoOp by conditioning the learnable context tokens on each input image via a lightweight **MetaNet** (2-layer MLP):

```
delta       = MetaNet(img_feat)              # (B, ctx_dim)
ctx_shifted = ctx + delta                    # image-specific context
text_feats  = encode_prompts(ctx_shifted)    # (B, C, D)
logits      = img_feat · text_feats          # (B, C)
```

Key implementation details:
- **Feature caching:** frozen CLIP image encoder runs once before training; subsequent epochs operate only on cached features → ~20× speedup
- **Batched text encoding:** all B×C prompts encoded in a single transformer call
- **EOT alignment:** context tokens are physically inserted into the token sequence; EOT position is precomputed after the shift

---

## 4. Results

### 4.1 Oxford Pets

| Method | Top-1 (%) | Top-5 (%) |
|--------|-----------|-----------|
| ZS-handcrafted | 88.1 | 97.5 |
| ZS-LLM | 90.2 | 99.3 |
| CoCoOp (16-shot) | **93.5** | **99.9** |

### 4.2 EuroSAT

| Method | Top-1 (%) | Top-5 (%) |
|--------|-----------|-----------|
| ZS-handcrafted | 54.3 | 91.0 |
| ZS-LLM | 52.8 | 94.7 |
| CoCoOp (16-shot) | **74.2** | **98.0** |

### 4.3 DTD

| Method | Top-1 (%) | Top-5 (%) |
|--------|-----------|-----------|
| ZS-handcrafted | 44.2 | 74.6 |
| ZS-LLM | 45.7 | 76.3 |
| CoCoOp (16-shot) | **57.4** | **86.8** |

### 4.4 Zero-Shot Template Sensitivity

Per-template accuracy across 80 ImageNet templates:

| Dataset | Mean (%) | Std (%) | Min (%) | Max (%) | Ensemble (%) |
|---------|----------|---------|---------|---------|--------------|
| Oxford Pets | 85.3 | 3.6 | 71.5 | 88.9 | 88.1 |
| EuroSAT | 49.1 | 4.7 | 38.5 | 57.5 | 54.3 |
| DTD | 41.4 | 2.6 | 31.8 | 45.5 | 44.2 |

> **Observation:** EuroSAT shows the highest template sensitivity (std=4.7%, range ~19 pp), confirming zero-shot CLIP is brittle on out-of-distribution satellite imagery — generic ImageNet language matches some templates better than others. Oxford Pets is more stable (std=3.6%) thanks to natural-image priors in CLIP pretraining. DTD is the least variable (std=2.6%) but has the lowest absolute accuracy, suggesting all templates are equally inadequate for texture semantics. Across all datasets, the ensemble beats the per-template mean by 2–5 pp, and LLM prompts outperform the ensemble on Oxford Pets (+2.1 pp) and DTD (+1.5 pp) — but the ensemble wins on EuroSAT (+1.5 pp vs LLM), the one case where domain-specific top-1 phrasing hurt class ranking despite lifting Top-5.

---

## 5. Analysis

### 5.1 LLM Prompts vs Handcrafted Ensemble

LLM prompts win on Oxford Pets (+2.1 pp, 90.2% vs 88.1%) and DTD (+1.5 pp, 45.7% vs 44.2%), confirming that domain-specific phrasing outperforms an 80-template generic ensemble on natural-image and texture domains. EuroSAT is the exception: ZS-handcrafted (54.3%) beats ZS-LLM (52.8%) on Top-1 by 1.5 pp, though ZS-LLM leads on Top-5 (94.7% vs 91.0%). This inversion suggests the LLM prompts correctly identify the right land-cover class but place it slightly lower in the ranking — likely because satellite class names like "Highway or Road" are visually ambiguous and the LLM prompts introduce terminology that shifts similarity scores across confusable classes. The EuroSAT variance plot (std=4.7%) shows the template choice matters most on this dataset, and the 7 LLM prompts do not reliably sample the high end of that distribution.

### 5.2 CoCoOp vs Zero-Shot

CoCoOp delivers consistent gains across all three datasets:

| Dataset | Best ZS Top-1 (%) | CoCoOp Top-1 (%) | Gain (pp) |
|---------|-------------------|-------------------|-----------|
| Oxford Pets | 90.2 | 93.5 | +3.3 |
| EuroSAT | 54.3 | 74.2 | +19.9 |
| DTD | 45.7 | 57.4 | +11.7 |

EuroSAT yields the largest absolute gain (+19.9 pp), consistent with the large domain gap between satellite imagery and CLIP's natural-image pretraining. Oxford Pets yields the smallest gain (+3.3 pp) where the ZS baseline is already strong — the training curves (loss 0.31→0.25, Top-1 93.0%→93.5% at epochs 5–10) show stable improvement but diminishing returns near saturation. All three training curves are monotonically improving at epoch 10, suggesting further gains are possible with more training.

### 5.3 Top-5 vs Top-1 Gap

| Dataset | Method | Top-1 (%) | Top-5 (%) | Gap (pp) |
|---------|--------|-----------|-----------|----------|
| Oxford Pets | ZS-handcrafted | 88.1 | 97.5 | 9.4 |
| Oxford Pets | ZS-LLM | 90.2 | 99.3 | 9.1 |
| Oxford Pets | CoCoOp | 93.5 | 99.9 | 6.4 |
| EuroSAT | ZS-handcrafted | 54.3 | 91.0 | 36.7 |
| EuroSAT | ZS-LLM | 52.8 | 94.7 | 41.9 |
| EuroSAT | CoCoOp | 74.2 | 98.0 | 23.8 |
| DTD | ZS-handcrafted | 44.2 | 74.6 | 30.4 |
| DTD | ZS-LLM | 45.7 | 76.3 | 30.6 |
| DTD | CoCoOp | 57.4 | 86.8 | 29.4 |

EuroSAT has the largest Top-5/Top-1 gap under zero-shot (up to 41.9 pp for ZS-LLM), confirming that CLIP recognises the correct land-cover category but fails to rank it first — a prompt disambiguation problem. CoCoOp dramatically closes this gap on EuroSAT (41.9 → 23.8 pp), showing that image-conditioned context tokens are especially effective at resolving class-ranking ambiguity in out-of-distribution domains. Oxford Pets has the smallest gap and CoCoOp closes it further (9.4 → 6.4 pp), consistent with a well-behaved natural-image domain near saturation.

---

## 6. Attention Heatmaps

Patch-level cosine similarity between text prompt features and ViT image patch features, comparing ZS vs CoCoOp for the same input.

**Oxford Pets — Target: beagle**



Both ZS and CoCoOp attention concentrate on the dog's body, but CoCoOp shifts focus more toward the background and torso region while ZS centres more on the face. Both correctly localise the animal, consistent with Oxford Pets being a well-suited domain for CLIP's natural-image pretraining. The small Top-1/Top-5 gap (6–9 pp) reflects this: the model reliably finds the dog but occasionally confuses similar breeds.

**EuroSAT — Target: highway or road**



CoCoOp attention aligns more continuously along the road network structure compared to ZS attention, which fires on scattered high-contrast patches (buildings, intersections) with less spatial coherence. This is the clearest example of CoCoOp learning a structurally meaningful prompt: the MetaNet conditions on the aerial image and shifts context toward linear infrastructure features, directly explaining the large Top-1 gain (+19.9 pp) on this dataset.

**DTD — Target: banded**


Both maps are distributed globally — expected for texture, where the discriminative signal is spatially repetitive rather than localised. CoCoOp attention shows slightly stronger emphasis along horizontal band structures vs the more scattered ZS response, but the difference is subtle, consistent with texture being an image-wide property.

---

## 7. Conclusions

All experiments are complete. Results across three datasets confirm and refine the draft conclusions:

1. **LLM prompts beat the handcrafted ensemble on natural-image and texture domains** (Oxford Pets +2.1 pp, DTD +1.5 pp) with zero labelled data. EuroSAT is the exception — the ensemble edges out LLM prompts on Top-1 (54.3% vs 52.8%), likely due to class-ranking ambiguity in satellite imagery that domain-specific phrasing does not resolve.

2. **CoCoOp with 16 shots delivers the largest gains where zero-shot is weakest.** EuroSAT: +19.9 pp (54.3% → 74.2%); DTD: +11.7 pp (45.7% → 57.4%); Oxford Pets: +3.3 pp (90.2% → 93.5%). All training curves are still improving at epoch 10, suggesting further gains remain available.

3. **The Top-5/Top-1 gap is largest on EuroSAT under zero-shot (up to 41.9 pp), and CoCoOp closes it most aggressively there (→ 23.8 pp).** This confirms that prompt tuning primarily helps with class ranking — the model already recognises the correct category but struggles to rank it first without image-conditioned context. Oxford Pets and DTD show smaller but consistent gap reductions.

4. **Feature caching makes CoCoOp training practical on 4 GB VRAM** with ~20× speedup. The low loss values at epoch 10 (Oxford Pets: 0.25; EuroSAT: 0.98; DTD: 1.22) reflect the relative difficulty of each domain and confirm stable convergence across all three.

---

## 8. References

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML.
- Zhou et al. (2022). *Learning to Prompt for Vision-Language Models (CoOp).* IJCV.
- Zhou et al. (2022). *Conditional Prompt Learning for Vision-Language Models (CoCoOp).* CVPR.
- Nie et al. (2023). *Visual Descriptions Are Worth Tokens: Improving Cross-Modal Alignment (KgCoOp).* ICCV.
