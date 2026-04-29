from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import torch

FLOWERS102_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise",
    "monkshood", "globe thistle", "snapdragon", "colt's foot", "king protea",
    "spear thistle", "yellow iris", "globe-flower", "purple coneflower",
    "peruvian lily", "balloon flower", "giant white arum lily", "fire lily",
    "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke",
    "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy",
    "daffodil", "sword lily", "poinsettia", "bolero deep blue",
    "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium",
    "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily"
]

def build_dataloader(dataset_name, preprocess, batch_size):
    data_dir = "./data"

    if dataset_name == "stl10":
        train_dataset = datasets.STL10(
            root=data_dir,
            split="train",
            download=True,
            transform=preprocess
        )

        test_dataset = datasets.STL10(
            root=data_dir,
            split="test",
            download=True,
            transform=preprocess
        )

        classnames = train_dataset.classes

    elif dataset_name == "eurosat":
        dataset = datasets.EuroSAT(
            root=data_dir,
            download=True,
            transform=preprocess
        )

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        classnames = dataset.classes

    elif dataset_name == "caltech101":
        dataset = datasets.Caltech101(
            root=data_dir,
            download=True,
            transform=preprocess
        )

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        classnames = dataset.categories

    elif dataset_name == "flowers102":
        train_dataset = datasets.Flowers102(
            root=data_dir,
            split="train",
            download=True,
            transform=preprocess
        )

        test_dataset = datasets.Flowers102(
            root=data_dir,
            split="test",
            download=True,
            transform=preprocess
        )

        classnames = FLOWERS102_CLASSES

    elif dataset_name == "oxfordpets":
        train_dataset = datasets.OxfordIIITPet(
            root=data_dir,
            split="trainval",
            download=True,
            transform=preprocess
        )

        test_dataset = datasets.OxfordIIITPet(
            root=data_dir,
            split="test",
            download=True,
            transform=preprocess
        )

        classnames = train_dataset.classes

    else:
        raise ValueError("Unsupported dataset name")

    classnames = [name.replace("_", " ") for name in classnames]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader, classnames
