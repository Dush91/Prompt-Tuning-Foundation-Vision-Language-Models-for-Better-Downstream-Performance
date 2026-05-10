"""
datasets.py
"""

import os, random
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet, EuroSAT, DTD

# Windows safe num_workers
NUM_WORKERS = 0 if os.name == 'nt' else 2

def clip_transform(size=224):
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),
            std =(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

def train_transform(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),
            std =(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

def few_shot_indices(dataset, shots, seed=42):
    rng = random.Random(seed)
    buckets = {}
    for idx, (_, label) in enumerate(dataset):
        buckets.setdefault(label, []).append(idx)
    selected = []
    for idxs in buckets.values():
        selected.extend(rng.sample(idxs, min(shots, len(idxs))))
    return selected
    
def val_split_indices(dataset, val_frac=0.2, seed=42):
    """
    Stratified validation split from the training set.
    Returns (train_indices, val_indices) — no class left out.
    val_frac: fraction of each class held out for validation (default 20%).
    """
    rng = random.Random(seed)
    buckets = {}
    for idx, (_, label) in enumerate(dataset):
        buckets.setdefault(label, []).append(idx)
    train_idx, val_idx = [], []
    for idxs in buckets.values():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_frac))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx
    
class OxfordPetsDataset(Dataset):
    def __init__(self, root, split='test', transform=None, download=True):
        # 'train' and 'val' both load from 'trainval' — indices split by val_split_indices
        _split = 'trainval' if split in ('train', 'val') else 'test'
        self.base = OxfordIIITPet(root=root, split=_split,
                                   target_types='category',
                                   transform=None, download=download)
        self.transform = transform or clip_transform()
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label
    @property
    def class_names(self):
        return [c.lower().replace('_', ' ') for c in self.base.classes]

class EuroSATDataset(Dataset):
    CLASS_NAMES = [
        'annual crop land', 'forest', 'brushland or shrubland',
        'highway or road', 'industrial buildings', 'pasture land',
        'permanent crop land', 'residential buildings', 'river', 'sea or lake',
    ]
    def __init__(self, root, split='test', transform=None, download=True):
        # EuroSAT has no native split — val indices carved via val_split_indices
        self.base = EuroSAT(root=root, transform=None, download=download)
        self.transform = transform or clip_transform()
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label
    @property
    def class_names(self): return self.CLASS_NAMES

class DTDDataset(Dataset):
    def __init__(self, root, split='test', transform=None, download=True):
        # DTD has a native 'val' split — use it directly
        _split = split if split in ('train', 'val', 'test') else 'test'
        self.base = DTD(root=root, split=_split,
                        transform=None, download=download)
        self.transform = transform or clip_transform()
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label
    @property
    def class_names(self):
        return [c.replace('_', ' ') for c in self.base.classes]

DATASETS = {
    'oxford_pets':   OxfordPetsDataset,
    'eurosat':       EuroSATDataset,
    'dtd':           DTDDataset,
}

def build_dataloader(name, root, split='test', batch_size=16,
                     num_workers=None, shots=None, train=False,
                     val_frac=0.2):
    import torch
    nw  = NUM_WORKERS if num_workers is None else num_workers
    tf  = train_transform() if train else clip_transform()

    if split in ('train', 'val'):
        # 1. Load the raw training pool
        base_ds = DATASETS[name](root=root, split='train', transform=tf, download=True)
        train_idx, val_idx = val_split_indices(base_ds, val_frac=val_frac)
        if split == 'val':
            ds = Subset(base_ds, val_idx)
        else:
            # 2. We are in the training split
            if shots:
                # Create a temporary subset to find labels within the training split
                train_subset = Subset(base_ds, train_idx)
                # few_shot_indices gives us positions relative to train_subset (0 to len(train_idx))
                sub_indices = few_shot_indices(train_subset, shots)
                final_indices = [train_idx[i] for i in sub_indices]
                ds = Subset(base_ds, final_indices)
            else:
                ds = Subset(base_ds, train_idx)
    else:
        # Standard test split
        ds = DATASETS[name](root=root, split='test', transform=tf, download=True)
    curr = ds
    while isinstance(curr, Subset):
        curr = curr.dataset
        
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        num_workers=nw, pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )
    return loader, curr.class_names
