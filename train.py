# train.py — training loop with early stopping and evaluate function

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_text_features(model, class_names, device):
    """Build and cache normalised text prototype features."""
    with torch.no_grad():
        import clip
        tokens = clip.tokenize([f"a photo of a {c}." for c in class_names]).to(device)
        feats  = model.encode_text(tokens).float()
        return F.normalize(feats, dim=-1)


def evaluate(clip_model, loader, text_features, classifier, method, device):
    """
    Evaluate accuracy on a dataloader.
    method: 'cosine' uses CLIP similarity, 'classifier' uses linear head.
    """
    clip_model.eval()
    if classifier is not None:
        classifier.eval()

    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            feats  = F.normalize(clip_model.encode_image(images).float(), dim=-1)

            if method == "cosine":
                logits = clip_model.logit_scale.exp() * feats @ text_features.T
            else:
                logits = classifier(feats)

            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)

    return 100.0 * correct / total


def train(clip_model, train_loader, val_loader,
          text_features, classifier, method,
          epochs, lr, weight_decay, patience, device,
          save_path="best_model.pt"):
    """
    Training loop with early stopping.
    Saves best checkpoint based on val accuracy.

    Args:
        clip_model    : CLIP model with LoRA injected
        train_loader  : training dataloader
        val_loader    : validation dataloader
        text_features : precomputed text prototypes (for cosine method)
        classifier    : nn.Linear head (for classifier method, else None)
        method        : 'cosine' or 'classifier'
        epochs        : max training epochs
        lr            : learning rate
        weight_decay  : AdamW weight decay
        patience      : early stopping patience
        device        : cuda or cpu
        save_path     : path to save best checkpoint

    Returns:
        best_val_acc  : best validation accuracy achieved
    """
    criterion  = nn.CrossEntropyLoss()

    # Collect trainable parameters
    lora_params = [p for p in clip_model.parameters() if p.requires_grad]
    all_params  = lora_params + (list(classifier.parameters()) if classifier else [])
    optimizer   = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    best_val   = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        clip_model.train()
        if classifier is not None:
            classifier.train()

        total_loss = correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            feats = F.normalize(clip_model.encode_image(images).float(), dim=-1)

            if method == "cosine":
                logits = clip_model.logit_scale.exp() * feats @ text_features.T
            else:
                logits = classifier(feats)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        val_acc = evaluate(clip_model, val_loader, text_features,
                           classifier, method, device)

        print(f"Epoch {epoch:>2}/{epochs}  "
              f"loss={total_loss/len(train_loader):.3f}  "
              f"train={100.*correct/total:.1f}%  "
              f"val={val_acc:.1f}%  "
              f"best={best_val:.1f}%")

        # Save best checkpoint
        if val_acc > best_val:
            best_val   = val_acc
            no_improve = 0
            ckpt = {"clip_model": clip_model.state_dict()}
            if classifier is not None:
                ckpt["classifier"] = classifier.state_dict()
            torch.save(ckpt, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    return best_val
