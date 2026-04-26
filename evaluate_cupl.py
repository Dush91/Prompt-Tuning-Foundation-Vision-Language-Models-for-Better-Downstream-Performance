import argparse
import json
import torch
import clip
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset


def clean(name):
    return name.replace("_", " ").replace("-", " ")


def handcrafted_prompt_templates(dataset):
    if dataset == "eurosat":
        return [
            "a satellite photo of {}.",
            "a centered satellite photo of {}.",
            "a remote sensing image of {}.",
            "an aerial image of {}.",
            "a land cover satellite image of {}.",
            "a Sentinel-2 satellite image showing {}.",
            "an overhead view of {}.",
            "a top-down aerial photograph of {}."
        ]
    if dataset == "oxfordpets":
        return [
            "a photo of a {} pet.",
            "a close-up image showing the face and fur of a {}.",
            "a photo of a {} showing its breed-specific features.",
            "a pet image showing the coat colour and body shape of a {}.",
            "a realistic image of a {} cat or dog.",
            "a photograph showing the ears, eyes, and fur texture of a {}.",
            "a typical {} pet in a natural setting.",
            "a high quality image of a {} breed."
        ]
    if dataset == "flowers102":
        return [
            "a photo of a {} flower.",
            "a close-up image of a {} flower.",
            "a photo showing the petals and colours of a {} flower.",
            "a realistic image of a {} flower in bloom.",
            "a photograph showing the shape and texture of a {} flower.",
            "a typical {} flower in a natural setting.",
            "a high quality image of a {} flower."
        ]        

    if dataset == "dtd":
        return [
            "{} texture.",
            "a photo of a {} texture.",
            "a close-up photo of a {} texture.",
            "a photo of a {} pattern.",
            "a material surface that looks {}.",
            "a detailed texture image of {}."
        ]

    return [
        "a photo of a {}.",
        "a photo of the {}.",
        "a close-up photo of a {}.",
        "a clear photo of a {}.",
        "a centered photo of a {}.",
        "a high quality photo of a {}."
    ]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_single_prompt_template(model, class_names, template, device):
    prompts = [template.format(clean(cls)) for cls in class_names]
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        features = model.encode_text(tokens)

    features = features / features.norm(dim=-1, keepdim=True)
    return features


def encode_cupl_prompt_set(model, class_names, cupl_prompts, prompt_index, device):
    prompts = []

    for cls in class_names:
        class_prompts = cupl_prompts[cls]
        selected_prompt = class_prompts[prompt_index % len(class_prompts)]
        prompts.append(selected_prompt)

    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        features = model.encode_text(tokens)

    features = features / features.norm(dim=-1, keepdim=True)
    return features


def evaluate(model, dataloader, text_features, device):
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ text_features.T
            preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["eurosat", "dtd", "caltech101", "oxfordpets", "flowers102"], help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", default="ViT-B/16")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Loading CLIP:", args.model)

    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    _, test_dataset, class_names = load_dataset(args.dataset, preprocess)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print("Dataset:", args.dataset)
    print("Classes:", len(class_names))
    print("Test images:", len(test_dataset))

    # -------------------------------
    # Handcrafted prompt mean/variance
    # -------------------------------
    templates = handcrafted_prompt_templates(args.dataset)
    handcrafted_accuracies = []

    print("\nEvaluating handcrafted prompt templates...")

    for i, template in enumerate(templates):
        print(f"\nHandcrafted Prompt {i + 1}/{len(templates)}: {template}")

        text_features = encode_single_prompt_template(
            model,
            class_names,
            template,
            device
        )

        acc = evaluate(model, test_loader, text_features, device)
        handcrafted_accuracies.append(acc)

        print(f"Accuracy: {acc:.2f}%")

    handcrafted_mean = np.mean(handcrafted_accuracies)
    handcrafted_variance = np.var(handcrafted_accuracies)

    # -------------------------------
    # CuPL prompt mean/variance
    # -------------------------------
    cupl_path = f"prompts/{args.dataset}_cupl_prompts.json"
    cupl_prompts = load_json(cupl_path)

    max_prompts = min(len(prompts) for prompts in cupl_prompts.values())
    cupl_accuracies = []

    print("\nEvaluating CuPL prompt sets...")

    for i in range(max_prompts):
        print(f"\nCuPL Prompt Set {i + 1}/{max_prompts}")

        text_features = encode_cupl_prompt_set(
            model,
            class_names,
            cupl_prompts,
            i,
            device
        )

        acc = evaluate(model, test_loader, text_features, device)
        cupl_accuracies.append(acc)

        print(f"Accuracy: {acc:.2f}%")

    cupl_mean = np.mean(cupl_accuracies)
    cupl_variance = np.var(cupl_accuracies)

    result = {
        "dataset": args.dataset,
        "model": args.model,

        "handcrafted_accuracies": [round(x, 2) for x in handcrafted_accuracies],
        "handcrafted_mean": round(float(handcrafted_mean), 2),
        "handcrafted_variance": round(float(handcrafted_variance), 4),

        "cupl_accuracies": [round(x, 2) for x in cupl_accuracies],
        "cupl_mean": round(float(cupl_mean), 2),
        "cupl_variance": round(float(cupl_variance), 4),

        "mean_improvement": round(float(cupl_mean - handcrafted_mean), 2)
    }

    print("\nFINAL RESULT")
    print("-" * 50)
    print(json.dumps(result, indent=4))

    json_path = f"results/{args.dataset}_mean_variance_results.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    csv_path = "results/all_mean_variance_results.csv"

    row = {
        "dataset": args.dataset,
        "model": args.model,
        "handcrafted_mean": result["handcrafted_mean"],
        "handcrafted_variance": result["handcrafted_variance"],
        "cupl_mean": result["cupl_mean"],
        "cupl_variance": result["cupl_variance"],
        "mean_improvement": result["mean_improvement"]
    }

    try:
        old_df = pd.read_csv(csv_path)
        old_df = old_df[old_df["dataset"] != args.dataset]
        new_df = pd.concat([old_df, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        new_df = pd.DataFrame([row])

    new_df.to_csv(csv_path, index=False)

    print(f"\nSaved JSON: {json_path}")
    print(f"Updated CSV: {csv_path}")


if __name__ == "__main__":
    main()