# test.py — final testing and generalisation on Oxford Pets and EuroSAT

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import clip
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader


def build_text_features_for(model, class_names, prompt_template, device):
    """Build normalised text features for any dataset using a prompt template."""
    with torch.no_grad():
        tokens = clip.tokenize(
            [prompt_template.format(c) for c in class_names]
        ).to(device)
        feats = model.encode_text(tokens).float()
        return F.normalize(feats, dim=-1)


def _eval_generalisation(model, loader, text_feats, name, device):
    """Evaluate a model on a loader using cosine similarity."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            feats  = F.normalize(model.encode_image(images).float(), dim=-1)
            logits = model.logit_scale.exp() * feats @ text_feats.T
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    acc = 100.0 * correct / total
    """print(f"  {name}: {acc:.2f}%")"""
    return acc


def test_caltech(clip_model, test_loader, text_features,
                 classifier, method, device):
    """Run final test on Caltech-101 test set."""
    from train import evaluate
    clip_model.eval()
    if classifier is not None:
        classifier.eval()
    acc = evaluate(clip_model, test_loader, text_features,
                   classifier, method, device)
    """print(f"  LoRA Test Accuracy ({method}): {acc:.2f}%")"""
    return acc


def test_pets(clip_model, clip_original, data, device):
    """
    Generalisation test on Oxford Pets.
    Compares LoRA-trained CLIP vs original frozen CLIP.
    Uses prompt: 'a photo of a {class}.'
    """
    pets_loader  = data["pets"]["test"]
    pets_classes = data["pets"]["classes"]

    """print("\nOxford Pets Generalisation:")"""
    lora_feats = build_text_features_for(
        clip_model,    pets_classes, "a photo of a {}.", device)
    orig_feats = build_text_features_for(
        clip_original, pets_classes, "a photo of a {}.", device)

    lora_acc = _eval_generalisation(clip_model,    pets_loader, lora_feats, "LoRA-trained CLIP", device)
    orig_acc = _eval_generalisation(clip_original, pets_loader, orig_feats, "Original CLIP    ", device)

    return lora_acc, orig_acc


def test_eurosat(clip_model, clip_original, data, device):
    """
    Generalisation test on EuroSAT satellite images.
    Compares LoRA-trained CLIP vs original frozen CLIP.
    Uses satellite-specific prompt for better zero-shot accuracy.
    """
    eurosat_loader   = data["eurosat"]["test"]
    eurosat_classes  = data["eurosat"]["classes"]

    """print("\nEuroSAT Generalisation:")"""
    lora_feats = build_text_features_for(
        clip_model,    eurosat_classes, "a satellite photo of {}.", device)
    orig_feats = build_text_features_for(
        clip_original, eurosat_classes, "a satellite photo of {}.", device)

    lora_acc = _eval_generalisation(clip_model,    eurosat_loader, lora_feats, "LoRA-trained CLIP", device)
    orig_acc = _eval_generalisation(clip_original, eurosat_loader, orig_feats, "Original CLIP    ", device)

    return lora_acc, orig_acc


@torch.no_grad()
def extract_features(model, loader, device, max_batches=20):
    """Extract image features and labels from a dataloader."""
    model.eval()
    all_feats, all_labels = [], []
    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        feats  = F.normalize(model.encode_image(images).float(), dim=-1)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


def plot_tsne(clip_model, clip_original, test_loader, device,
              save_path="tsne_comparison.png"):
    """
    Plot t-SNE of image features for LoRA vs original CLIP side by side.
    Tight separated clusters = better learned representations.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("Install scikit-learn: pip install scikit-learn")
        return

    print("\nExtracting features for t-SNE...")
    lora_feats, labels = extract_features(clip_model,    test_loader, device)
    orig_feats, _      = extract_features(clip_original, test_loader, device)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)

    print("Running t-SNE on original features...")
    orig_2d = tsne.fit_transform(orig_feats)
    print("Running t-SNE on LoRA features...")
    lora_2d = tsne.fit_transform(lora_feats)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, feats_2d, title in zip(
        axes,
        [orig_2d, lora_2d],
        ["Original CLIP (Zero-shot)", "LoRA Fine-tuned CLIP"]
    ):
        sc = ax.scatter(feats_2d[:, 0], feats_2d[:, 1],
                        c=labels, cmap="tab20", alpha=0.6, s=8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, label="Class index")
    fig.suptitle("t-SNE of Image Features — Caltech-101", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot: {save_path}")
