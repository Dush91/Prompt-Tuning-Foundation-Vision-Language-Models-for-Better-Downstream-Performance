"""
Evaluation script for DSPy-CLIP descriptors
Compares: Baseline templates vs CuPL vs DSPy-MIPRO
"""

import os
import json
import torch
import torch.nn.functional as F
from typing import List, Dict
from tqdm import tqdm

from clip import clip
from dassl.data import build_data_loader
from dassl.config import get_cfg_default
from datasets.eurosat import EuroSAT

from algorithms.dspy_clip.cupl_baseline import load_cupl_descriptors, CuPLGenerator


class ZeroShotCLIPWithDescriptors:
    """
    Zero-shot CLIP classifier using custom descriptors.
    """

    def __init__(self, clip_model, device="cuda"):
        self.clip_model = clip_model
        self.device = device
        self.clip_model.to(device)
        self.clip_model.eval()
        self.text_features = None
        self.class_names = None

    def build_text_features(
        self,
        class_descriptors: Dict[str, List[str]],
    ):
        """
        Build text features from class descriptors.

        Args:
            class_descriptors: Dict mapping class name to list of descriptions
        """
        self.class_names = list(class_descriptors.keys())
        all_features = []

        print("Building text features...")
        for class_name in tqdm(self.class_names):
            descriptions = class_descriptors[class_name]

            # Tokenize all descriptions for this class
            tokens = torch.cat([
                clip.tokenize(desc) for desc in descriptions
            ]).to(self.device)

            # Get CLIP text features
            with torch.no_grad():
                features = self.clip_model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                # Average across all descriptions for this class
                mean_features = features.mean(dim=0)
                mean_features = mean_features / mean_features.norm()

            all_features.append(mean_features)

        self.text_features = torch.stack(all_features)

    def classify(self, images: torch.Tensor) -> torch.Tensor:
        """
        Classify images.

        Args:
            images: Batch of images

        Returns:
            Logits tensor
        """
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ self.text_features.T

        return logits


def evaluate_on_dataset(
    classifier: ZeroShotCLIPWithDescriptors,
    data_loader,
    device="cuda",
) -> Dict[str, float]:
    """
    Evaluate classifier on a dataset.

    Returns dict with accuracy and per-class accuracy.
    """
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    print("Evaluating...")
    for batch in tqdm(data_loader):
        images = batch["img"].to(device)
        labels = batch["label"].to(device)

        logits = classifier.classify(images)
        predictions = logits.argmax(dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Per-class accuracy
        for pred, label in zip(predictions, labels):
            label_idx = label.item()
            if label_idx not in class_correct:
                class_correct[label_idx] = 0
                class_total[label_idx] = 0
            class_total[label_idx] += 1
            if pred == label:
                class_correct[label_idx] += 1

    accuracy = 100.0 * correct / total

    per_class_acc = {
        class_idx: 100.0 * class_correct.get(class_idx, 0) / class_total[class_idx]
        for class_idx in class_total.keys()
    }

    return {
        "overall_accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "correct": correct,
        "total": total,
    }


def create_baseline_descriptors(class_names: List[str]) -> Dict[str, List[str]]:
    """Create baseline descriptors using simple templates."""
    templates = [
        "a centered satellite photo of {}.",
        "satellite imagery of {}.",
        "aerial view of {}.",
    ]

    descriptors = {}
    for class_name in class_names:
        class_descriptions = [
            template.format(class_name) for template in templates
        ]
        descriptors[class_name] = class_descriptions

    return descriptors


def main():
    """Run full evaluation comparing all methods."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data", help="Data root")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "cupl"],
        choices=["baseline", "cupl", "dspy_mipro"],
        help="Methods to evaluate",
    )
    parser.add_argument(
        "--cupl_descriptors",
        default="output/cupl_descriptors/eurosat_descriptors.json",
        help="Path to CuPL descriptors",
    )
    parser.add_argument(
        "--dspy_descriptors",
        default="output/dspy_mipro/eurosat_descriptors.json",
        help="Path to DSPy-MIPRO descriptors",
    )
    args = parser.parse_args()

    # Load CLIP
    print("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/16", device=args.device)

    # Get EuroSAT class names
    eurosat_classes = [
        "Annual Crop Land",
        "Forest",
        "Herbaceous Vegetation Land",
        "Highway or Road",
        "Industrial Buildings",
        "Pasture Land",
        "Permanent Crop Land",
        "Residential Buildings",
        "River",
        "Sea or Lake",
    ]

    # Initialize classifier
    classifier = ZeroShotCLIPWithDescriptors(clip_model, args.device)

    # Setup data (minimal - just for structure)
    cfg = get_cfg_default()
    cfg.DATASET.NAME = "EuroSAT"
    cfg.DATASET.ROOT = args.root
    cfg.DATASET.NUM_SHOTS = 16

    # Results storage
    results = {}

    # Evaluate each method
    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method.upper()}")
        print("="*60)

        if method == "baseline":
            descriptors = create_baseline_descriptors(eurosat_classes)

        elif method == "cupl":
            if not os.path.exists(args.cupl_descriptors):
                print(f"CuPL descriptors not found at {args.cupl_descriptors}")
                print("Generating with CuPL baseline...")
                generator = CuPLGenerator()
                descriptors = generator.generate_for_dataset(
                    eurosat_classes,
                    output_path=args.cupl_descriptors,
                )
            else:
                print(f"Loading CuPL descriptors from {args.cupl_descriptors}")
                descriptors = load_cupl_descriptors(args.cupl_descriptors)

        elif method == "dspy_mipro":
            if not os.path.exists(args.dspy_descriptors):
                print(f"DSPy-MIPRO descriptors not found at {args.dspy_descriptors}")
                print("Please run MIPRO optimization first")
                continue
            print(f"Loading DSPy-MIPRO descriptors from {args.dspy_descriptors}")
            descriptors = load_cupl_descriptors(args.dspy_descriptors)

        # Build text features
        classifier.build_text_features(descriptors)

        # Note: Full evaluation requires dataset setup
        # For now, we just print descriptor statistics
        print(f"\nDescriptor Statistics:")
        print(f"  Classes: {len(descriptors)}")
        total_descs = sum(len(d) for d in descriptors.values())
        avg_descs = total_descs / len(descriptors)
        print(f"  Total descriptions: {total_descs}")
        print(f"  Avg descriptions per class: {avg_descs:.1f}")

        # Show sample descriptors
        print(f"\nSample descriptors for '{eurosat_classes[0]}':")
        for desc in descriptors[eurosat_classes[0]][:3]:
            print(f"  - {desc}")

        results[method] = {
            "num_classes": len(descriptors),
            "total_descriptions": total_descs,
            "avg_per_class": avg_descs,
        }

    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print("="*60)
    for method, stats in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Classes: {stats['num_classes']}")
        print(f"  Total descriptions: {stats['total_descriptions']}")
        print(f"  Avg per class: {stats['avg_per_class']:.1f}")


if __name__ == "__main__":
    main()
