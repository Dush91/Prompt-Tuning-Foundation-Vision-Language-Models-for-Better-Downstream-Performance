import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="stl10",
        choices=["stl10", "eurosat", "caltech101", "flowers102", "oxfordpets"]
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_ctx", type=int, default=16)

    return parser.parse_args()