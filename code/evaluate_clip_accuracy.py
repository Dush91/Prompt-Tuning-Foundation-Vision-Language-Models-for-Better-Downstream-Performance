"""
Evaluate CLIP zero-shot accuracy with different descriptor methods
"""

import os
os.chdir('/tmp')  # Avoid local clip module

import torch
import clip
import json
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from glob import glob
from tqdm import tqdm

# EuroSAT class names (match the descriptor keys)
CLASSES = [
    'Annual Crop Land',
    'Forest',
    'Herbaceous Vegetation Land',
    'Highway or Road',
    'Industrial Buildings',
    'Pasture Land',
    'Permanent Crop Land',
    'Residential Buildings',
    'River',
    'Sea or Lake',
]

# Map from folder names to class names
FOLDER_TO_CLASS = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake',
}

class EuroSATSimpleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load images from 2750 folder structure
        data_dir = os.path.join(root_dir, 'eurosat', '2750')
        if os.path.exists(data_dir):
            for folder in os.listdir(data_dir):
                folder_path = os.path.join(data_dir, folder)
                if os.path.isdir(folder_path) and folder in FOLDER_TO_CLASS:
                    class_idx = CLASSES.index(FOLDER_TO_CLASS[folder])
                    for img_file in glob(os.path.join(folder_path, '*.png')):
                        self.samples.append((img_file, class_idx))

        print(f"Loaded {len(self.samples)} images from EuroSAT")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def evaluate_method(model, device, class_features, dataset, method_name):
    """Evaluate a descriptor method on EuroSAT"""
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    print(f"\nEvaluating {method_name}...")
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ class_features.T
            preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load CLIP
    print("Loading CLIP ViT-B/16...")
    model, preprocess = clip.load('ViT-B/16', device=device)

    # Load descriptors
    print("Loading descriptors...")
    with open('/teamspace/studios/this_studio/output/cupl_descriptors/eurosat_descriptors.json') as f:
        cupl_descs = json.load(f)
    with open('/teamspace/studios/this_studio/output/dspy_baseline/eurosat_descriptors.json') as f:
        dspy_descs = json.load(f)
    with open('/teamspace/studios/this_studio/output/dspy_mipro/eurosat_descriptors.json') as f:
        mipro_descs = json.load(f)

    # Build class text features for each method
    def build_features(desc_dict, max_descs=50):
        features = []
        for cls in CLASSES:
            descs = desc_dict[cls][:max_descs]
            # Truncate for CLIP
            descs = [d[:250] if len(d) > 250 else d for d in descs]
            tokens = torch.cat([clip.tokenize(d) for d in descs]).to(device)

            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                mean_feat = feats.mean(dim=0)
                mean_feat = mean_feat / mean_feat.norm()
            features.append(mean_feat)
        return torch.stack(features)

    print("Building text features...")
    cupl_features = build_features(cupl_descs, 50)
    dspy_features = build_features(dspy_descs, 10)
    mipro_features = build_features(mipro_descs, 10)

    # Baseline: simple templates
    baseline_prompts = [f"a centered satellite photo of {cls}." for cls in CLASSES]
    baseline_tokens = torch.cat([clip.tokenize(p) for p in baseline_prompts]).to(device)
    with torch.no_grad():
        baseline_features = model.encode_text(baseline_tokens)
        baseline_features = baseline_features / baseline_features.norm(dim=-1, keepdim=True)

    # Load EuroSAT dataset
    data_root = '/teamspace/studios/this_studio/data'
    if os.path.exists(os.path.join(data_root, 'eurosat', '2750')):
        dataset = EuroSATSimpleDataset(data_root, transform=preprocess)

        if len(dataset) > 0:
            # Evaluate all methods
            print("\n" + "="*60)
            print("CLIP ZERO-SHOT EVALUATION ON EUROSAT")
            print("="*60)

            baseline_acc = evaluate_method(model, device, baseline_features, dataset, "Baseline (1 template)")
            print(f"Baseline Accuracy: {baseline_acc:.2f}%")

            cupl_acc = evaluate_method(model, device, cupl_features, dataset, "CuPL (50 descs/class)")
            print(f"CuPL Accuracy: {cupl_acc:.2f}%")

            dspy_acc = evaluate_method(model, device, dspy_features, dataset, "DSPy Baseline (10 descs/class)")
            print(f"DSPy Baseline Accuracy: {dspy_acc:.2f}%")

            mipro_acc = evaluate_method(model, device, mipro_features, dataset, "MIPRO Optimized (10 descs/class)")
            print(f"MIPRO Accuracy: {mipro_acc:.2f}%")

            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Baseline (hand-crafted): {baseline_acc:.2f}%")
            print(f"CuPL (50 descs/class):     {cupl_acc:.2f}%")
            print(f"DSPy Baseline:             {dspy_acc:.2f}%")
            print(f"MIPRO Optimized:           {mipro_acc:.2f}%")

            if mipro_acc > cupl_acc:
                print(f"\n✅ MIPRO beats CuPL by +{mipro_acc - cupl_acc:.2f}%!")
            elif mipro_acc > baseline_acc:
                print(f"\n⚠️  MIPRO beats baseline but not CuPL")
            else:
                print(f"\n❌ No improvement over baseline")
        else:
            print("No images found in EuroSAT dataset")
    else:
        print(f"EuroSAT data not found at {data_root}/eurosat/2750")
        print("Checking available data...")
        if os.path.exists(data_root):
            print(os.listdir(data_root))


if __name__ == "__main__":
    main()
