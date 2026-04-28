# data.py — loads all datasets at once: Caltech101, Oxford Pets, EuroSAT

from torch.utils.data import DataLoader, random_split
from torchvision import datasets


def get_all_dataloaders(preprocess, batch_size=32, data_root="./data"):
    """
    Loads all datasets at once and returns a single dictionary.

    Caltech-101 : split 70/15/15 into train/val/test
    Oxford Pets : full dataset used for generalisation testing only
    EuroSAT     : full dataset used for generalisation testing only

    Returns:
        dict with keys: 'caltech', 'pets', 'eurosat'
        each containing 'test' loader, 'classes' list
        caltech also has 'train' and 'val' loaders
    """

    # ── Caltech-101 ───────────────────────────────────────────────────────────
    caltech   = datasets.Caltech101(root=data_root, download=True, transform=preprocess)
    n         = len(caltech)
    n_train   = int(0.70 * n)
    n_val     = int(0.15 * n)
    n_test    = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(caltech, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    caltech_classes = caltech.categories
    """print(f"Caltech101 — Train: {len(train_ds)}  Val: {len(val_ds)}  "
          f"Test: {len(test_ds)}  Classes: {len(caltech_classes)}")"""

    # ── Oxford Pets ───────────────────────────────────────────────────────────
    pets         = datasets.OxfordIIITPet(root=data_root, download=True, transform=preprocess)
    pets_loader  = DataLoader(pets, batch_size=batch_size, shuffle=False, num_workers=2)
    pets_classes = pets.classes
    """print(f"Oxford Pets — Total: {len(pets)}  Classes: {len(pets_classes)}")"""

    # ── EuroSAT ───────────────────────────────────────────────────────────────
    eurosat         = datasets.EuroSAT(root=data_root, download=True, transform=preprocess)
    eurosat_loader  = DataLoader(eurosat, batch_size=batch_size, shuffle=False, num_workers=2)
    eurosat_classes = eurosat.classes
    """"print(f"EuroSAT    — Total: {len(eurosat)}  Classes: {len(eurosat_classes)}")"""

    return {
        "caltech": {
            "train":   train_loader,
            "val":     val_loader,
            "test":    test_loader,
            "classes": caltech_classes,
        },
        "pets": {
            "test":    pets_loader,
            "classes": pets_classes,
        },
        "eurosat": {
            "test":    eurosat_loader,
            "classes": eurosat_classes,
        },
    }
