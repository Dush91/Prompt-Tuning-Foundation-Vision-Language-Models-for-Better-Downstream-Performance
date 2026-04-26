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

class OxfordPetsDataset(Dataset):
    def __init__(self, root, split='test', transform=None, download=True):
        _split = 'trainval' if split == 'train' else 'test'
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
        _split = 'test' if split == 'test' else 'train'
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
                     num_workers=None, shots=None, train=False):
    import torch
    nw  = NUM_WORKERS if num_workers is None else num_workers
    tf  = train_transform() if train else clip_transform()
    ds  = DATASETS[name](root=root, split=split, transform=tf, download=True)

    if shots:
        ds = Subset(ds, few_shot_indices(ds, shots))

    base   = ds.dataset if isinstance(ds, Subset) else ds
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        num_workers=nw, pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )
    return loader, base.class_names