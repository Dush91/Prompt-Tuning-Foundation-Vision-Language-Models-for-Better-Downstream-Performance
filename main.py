# main.py — entry point with argparse for all hyperparameters
#
# Usage examples:
#   python3 main.py                                      # defaults
#   python3 main.py --rank 4  --method cosine            # LoRA cosine
#   python3 main.py --rank 4  --method classifier        # LoRA classifier
#   python3 main.py --rank 8  --epochs 20 --lr 1e-5      # custom hyperparams
#   python3 main.py --rank 4  --tsne                     # include t-SNE plot
#   python3 main.py --rank 16 --dropout 0.1              # with dropout

import argparse
import torch
import torch.nn as nn
import clip
import sys
import os

from data  import get_all_dataloaders
from model import inject_lora
from train import train, evaluate, build_text_features
from test  import test_caltech, test_pets, test_eurosat, plot_tsne

# ── Auto logging ──────────────────────────────────────────────────────────────


class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()




# ── Argument parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="CLIP LoRA Fine-tuning on Caltech-101")

    # Data
    parser.add_argument("--data_root",    type=str,   default="./data",
                        help="Path to dataset root")
    parser.add_argument("--batch_size",   type=int,   default=32,
                        help="Batch size for all loaders")

    # LoRA
    parser.add_argument("--rank",         type=int,   default=4,
                        help="LoRA rank r (try 2, 4, 8, 16)")
    parser.add_argument("--alpha",        type=float, default=1.0,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--dropout",      type=float, default=0.0,
                        help="LoRA dropout (0.0 = off)")

    # Training
    parser.add_argument("--epochs",       type=int,   default=10,
                        help="Max training epochs")
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--patience",     type=int,   default=3,
                        help="Early stopping patience")

    # Method
    parser.add_argument("--method",       type=str,   default="cosine",
                        choices=["cosine", "classifier"],
                        help="cosine: CLIP similarity | classifier: linear head")

    # Extras
    parser.add_argument("--tsne",         action="store_true",
                        help="Generate t-SNE plot after training")
    parser.add_argument("--clip_model",   type=str,   default="ViT-B/16",
                        help="CLIP backbone (default: ViT-B/16)")

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    log_path = os.path.expanduser(f"~/clip_lora/logs/log_r{args.rank}_{args.method}.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = Logger(log_path)

    print(f"\n{'='*52}")
    print(f"  Device  : {DEVICE}")
    print(f"  Method  : {args.method}")
    print(f"  Rank    : {args.rank}")
    print(f"  Epochs  : {args.epochs}  LR: {args.lr}  WD: {args.weight_decay}")
    print(f"{'='*52}\n")

    # ── 1. Load CLIP ──────────────────────────────────────────────────────────
    clip_model, preprocess = clip.load(args.clip_model, device=DEVICE)
    clip_model = clip_model.float()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # ── 2. Load all datasets at once ─────────────────────────────────────────
    data = get_all_dataloaders(
        preprocess, batch_size=args.batch_size, data_root=args.data_root
    )

    train_loader = data["caltech"]["train"]
    val_loader   = data["caltech"]["val"]
    test_loader  = data["caltech"]["test"]
    CLASS_NAMES  = data["caltech"]["classes"]
    NUM_CLASSES  = len(CLASS_NAMES)

    # ── 3. Zero-shot baseline (before LoRA injection) ─────────────────────────
    text_features = build_text_features(clip_model, CLASS_NAMES, DEVICE)
    zs_acc = evaluate(clip_model, test_loader, text_features,
                      None, "cosine", DEVICE)
    """print(f"\nZero-shot Accuracy on Caltech : {zs_acc:.2f}%\n")"""

    # ── 4. Inject LoRA ────────────────────────────────────────────────────────
    inject_lora(clip_model, rank=args.rank, alpha=args.alpha,
                dropout=args.dropout, device=DEVICE)

    # ── 5. Classifier head (only if method=classifier) ────────────────────────
    classifier = None
    if args.method == "classifier":
        EMBED_DIM  = clip_model.visual.output_dim
        classifier = nn.Linear(EMBED_DIM, NUM_CLASSES).to(DEVICE)

    # Trainable parameter count
    trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    if classifier is not None:
        trainable += sum(p.numel() for p in classifier.parameters())
    total = sum(p.numel() for p in clip_model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)\n")

    # Rebuild text features after LoRA injection
    text_features = build_text_features(clip_model, CLASS_NAMES, DEVICE)

    # ── 6. Train ──────────────────────────────────────────────────────────────
    save_path = f"best_model_r{args.rank}_{args.method}.pt"
    best_val  = train(
        clip_model, train_loader, val_loader,
        text_features, classifier, args.method,
        epochs       = args.epochs,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        patience     = args.patience,
        device       = DEVICE,
        save_path    = save_path,
    )

    # ── 7. Load best checkpoint ───────────────────────────────────────────────
    ckpt = torch.load(save_path)
    clip_model.load_state_dict(ckpt["clip_model"])
    if classifier is not None and "classifier" in ckpt:
        classifier.load_state_dict(ckpt["classifier"])

    # Rebuild text features from best checkpoint
    text_features = build_text_features(clip_model, CLASS_NAMES, DEVICE)

    # ── 8. Test on Caltech ────────────────────────────────────────────────────
    """print("\nCaltech-101 Test:")"""
    test_acc = test_caltech(clip_model, test_loader, text_features,
                            classifier, args.method, DEVICE)

    # ── 9. Load fresh CLIP for fair generalisation comparison ─────────────────
    clip_original, _ = clip.load(args.clip_model, device=DEVICE)
    clip_original = clip_original.float()
    for p in clip_original.parameters():
        p.requires_grad_(False)

    # ── 10. Generalisation tests ──────────────────────────────────────────────
    lora_pets_acc,    orig_pets_acc    = test_pets(clip_model, clip_original, data, DEVICE)
    lora_eurosat_acc, orig_eurosat_acc = test_eurosat(clip_model, clip_original, data, DEVICE)

    # ── 11. Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  Config  : rank={args.rank}  method={args.method}  lr={args.lr}")
    print(f"  Params  : {trainable:,} trainable / {total:,} total")
    print(f"{'='*52}")
    print(f"  Caltech Zero-shot        : {zs_acc:.2f}%")
    print(f"  Caltech LoRA Test        : {test_acc:.2f}%")
    print(f"  LoRA gain over zero-shot : {test_acc - zs_acc:+.2f}%")
    print(f"  Best Val Accuracy        : {best_val:.2f}%")
    print(f"{'='*52}")
    print(f"  Oxford Pets")
    print(f"    LoRA-trained CLIP      : {lora_pets_acc:.2f}%")
    print(f"    Original CLIP          : {orig_pets_acc:.2f}%")
    print(f"    Generalisation drop    : {lora_pets_acc - orig_pets_acc:+.2f}%")
    print(f"{'='*52}")
    print(f"  EuroSAT")
    print(f"    LoRA-trained CLIP      : {lora_eurosat_acc:.2f}%")
    print(f"    Original CLIP          : {orig_eurosat_acc:.2f}%")
    print(f"    Generalisation drop    : {lora_eurosat_acc - orig_eurosat_acc:+.2f}%")
    print(f"{'='*52}\n")

    # ── 12. Optional t-SNE ────────────────────────────────────────────────────
    if args.tsne:
        plot_tsne(clip_model, clip_original, test_loader, DEVICE,
                  save_path=f"tsne_r{args.rank}_{args.method}.png")


if __name__ == "__main__":
    main()
