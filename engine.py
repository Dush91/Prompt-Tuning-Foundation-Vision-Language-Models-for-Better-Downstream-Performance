import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import clip

from models import TextEncoder, PromptLearner, PROMPTS

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_handcrafted_clip(clip_model, test_loader, classnames, dataset_name, device):
    templates = PROMPTS[dataset_name]

    prompts = []
    for name in classnames:
        for template in templates:
            prompts.append(template.format(name))

    tokens = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_features = text_features.view(len(classnames), len(templates), -1)
        text_features = text_features.mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Handcrafted CLIP"):
            images = images.to(device)
            labels = labels.to(device)

            image_features = clip_model.encode_image(images).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ text_features.t()
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def train_coop(clip_model, train_loader, test_loader, classnames, dataset_name, args, device):
    text_encoder = TextEncoder(clip_model).to(device)
    prompt_learner = PromptLearner(
        classnames,
        clip_model,
        args.n_ctx,
        dataset_name,
        device
    ).to(device)

    optimizer = optim.AdamW(prompt_learner.parameters(), lr=args.lr, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []

    for epoch in range(args.epochs):
        prompt_learner.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"CoOp Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(images).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            prompts = prompt_learner()
            text_features = text_encoder(prompts, prompt_learner.tokenized_prompts)

            text_features = text_features.view(
                prompt_learner.n_cls,
                prompt_learner.num_templates,
                -1
            )

            text_features = text_features.mean(dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ text_features.t()
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        train_losses.append(avg_loss)
        train_accs.append(train_acc)

        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} Accuracy: {train_acc:.2f}%")

    coop_acc = evaluate_coop(clip_model, test_loader, text_encoder, prompt_learner, device)

    return coop_acc, train_losses, train_accs


def evaluate_coop(clip_model, test_loader, text_encoder, prompt_learner, device):
    prompt_learner.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        prompts = prompt_learner()
        text_features = text_encoder(prompts, prompt_learner.tokenized_prompts)

        text_features = text_features.view(
            prompt_learner.n_cls,
            prompt_learner.num_templates,
            -1
        )

        text_features = text_features.mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for images, labels in tqdm(test_loader, desc="CoOp Testing"):
            images = images.to(device)
            labels = labels.to(device)

            image_features = clip_model.encode_image(images).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ text_features.t()
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def save_results(dataset_name, clip_acc, coop_acc, train_losses, train_accs):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    mean_acc = np.mean(train_accs)
    var_acc = np.var(train_accs, ddof=1)

    mean_loss = np.mean(train_losses)
    var_loss = np.var(train_losses, ddof=1)

    improvement = coop_acc - clip_acc

    print("\n===== FINAL RESULTS =====")
    print(f"Handcrafted CLIP Accuracy: {clip_acc:.2f}%")
    print(f"CoOp Accuracy: {coop_acc:.2f}%")
    print(f"Improvement: {improvement:.2f}%")

    print("\n===== MEAN AND VARIANCE =====")
    print(f"Mean Training Accuracy: {mean_acc:.2f}%")
    print(f"Variance Training Accuracy: {var_acc:.4f}")
    print(f"Mean Training Loss: {mean_loss:.4f}")
    print(f"Variance Training Loss: {var_loss:.4f}")

    csv_path = os.path.join(RESULTS_DIR, f"{dataset_name}_results.csv")

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Dataset", dataset_name])
        writer.writerow(["Handcrafted CLIP Accuracy", clip_acc])
        writer.writerow(["CoOp Accuracy", coop_acc])
        writer.writerow(["Improvement", improvement])

        writer.writerow([])
        writer.writerow(["Mean Training Accuracy", mean_acc])
        writer.writerow(["Variance Training Accuracy", var_acc])
        writer.writerow(["Mean Training Loss", mean_loss])
        writer.writerow(["Variance Training Loss", var_loss])

        writer.writerow([])
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy"])

        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], train_accs[i]])

    print("CSV saved at:", csv_path)

    return csv_path