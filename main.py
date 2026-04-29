import torch
import clip

from config import parse_args
from datasets import build_dataloader
from engine import evaluate_handcrafted_clip, train_coop, save_results
from visualize import plot_training_curves, plot_comparison_all

def main():
    args = parse_args()

    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.float()

    for param in clip_model.parameters():
        param.requires_grad = False

    train_loader, test_loader, classnames = build_dataloader(
        args.dataset,
        preprocess,
        args.batch_size
    )

    print("Dataset:", args.dataset)
    print("Number of classes:", len(classnames))

    clip_acc = evaluate_handcrafted_clip(
        clip_model,
        test_loader,
        classnames,
        args.dataset,
        device
    )

    print(f"Handcrafted CLIP Accuracy: {clip_acc:.2f}%")

    coop_acc, train_losses, train_accs = train_coop(
        clip_model,
        train_loader,
        test_loader,
        classnames,
        args.dataset,
        args,
        device
    )

    csv_path = save_results(
        args.dataset,
        clip_acc,
        coop_acc,
        train_losses,
        train_accs
    )

    plot_training_curves(
        args.dataset,
        train_losses,
        train_accs
    )

    plot_comparison_all()

    print("Saved:", csv_path)
    print("Done.")

if __name__ == "__main__":
    main()