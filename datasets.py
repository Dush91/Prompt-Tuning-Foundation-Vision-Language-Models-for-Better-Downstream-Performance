import torch
from torch.utils.data import random_split, Dataset
from torchvision import datasets


class LabelRemapDataset(Dataset):
    def __init__(self, dataset, keep_indices, label_map):
        self.dataset = dataset
        self.keep_indices = keep_indices
        self.label_map = label_map

    def __len__(self):
        return len(self.keep_indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.keep_indices[idx]]
        return image, self.label_map[label]


def get_class_names(dataset):
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    if hasattr(dataset, "categories"):
        return list(dataset.categories)
    raise ValueError("Cannot find class names.")


def remove_caltech_background(dataset):
    class_names = get_class_names(dataset)

    if "BACKGROUND_Google" not in class_names:
        return dataset, class_names

    bg_index = class_names.index("BACKGROUND_Google")

    new_class_names = []
    label_map = {}

    new_label = 0
    for old_label, name in enumerate(class_names):
        if old_label == bg_index:
            continue
        label_map[old_label] = new_label
        new_class_names.append(name)
        new_label += 1

    keep_indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label != bg_index:
            keep_indices.append(i)

    filtered_dataset = LabelRemapDataset(dataset, keep_indices, label_map)
    return filtered_dataset, new_class_names


def load_dataset(dataset_name, preprocess, root="data", seed=42):
    dataset_name = dataset_name.lower()

    if dataset_name == "eurosat":
        full_dataset = datasets.EuroSAT(
            root=root,
            transform=preprocess,
            download=True
        )
        class_names = get_class_names(full_dataset)

    elif dataset_name == "dtd":
        train_dataset = datasets.DTD(
            root=root,
            split="train",
            transform=preprocess,
            download=True
        )
        test_dataset = datasets.DTD(
            root=root,
            split="test",
            transform=preprocess,
            download=True
        )
        class_names = get_class_names(train_dataset)
        return train_dataset, test_dataset, class_names

    elif dataset_name == "caltech101":
        full_dataset = datasets.Caltech101(
            root=root,
            target_type="category",
            transform=preprocess,
            download=True
        )
        full_dataset, class_names = remove_caltech_background(full_dataset)
    
    elif dataset_name == "oxfordpets":
        train_dataset = datasets.OxfordIIITPet(
            root=root,
            split="trainval",
            target_types="category",
            transform=preprocess,
            download=True
        )
        test_dataset = datasets.OxfordIIITPet(
            root=root,
            split="test",
            target_types="category",
            transform=preprocess,
            download=True
        )
        class_names = get_class_names(train_dataset)
        return train_dataset, test_dataset, class_names
    elif dataset_name == "flowers102":
        train_dataset = datasets.Flowers102(
            root=root,
            split="train",
            transform=preprocess,
            download=True
        )
        test_dataset = datasets.Flowers102(
            root=root,
            split="test",
            transform=preprocess,
            download=True
        )
        class_names = get_class_names(train_dataset)
        return train_dataset, test_dataset, class_names  

    else:
        raise ValueError("Use one of: eurosat, dtd, caltech101, oxfordpets, flowers102")

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=generator
    )

    return train_dataset, test_dataset, class_names