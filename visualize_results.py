import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    csv_path = "results/all_mean_variance_results.csv"
    df = pd.read_csv(csv_path)

    print(df)

    datasets = df["dataset"].tolist()

    handcrafted_mean = df["handcrafted_mean"].values
    cupl_mean = df["cupl_mean"].values

    handcrafted_std = np.sqrt(df["handcrafted_variance"].values)
    cupl_std = np.sqrt(df["cupl_variance"].values)

    x = np.arange(len(datasets))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        handcrafted_mean,
        width,
        yerr=handcrafted_std,
        capsize=5,
        label="Handcrafted CLIP"
    )

    plt.bar(
        x + width / 2,
        cupl_mean,
        width,
        yerr=cupl_std,
        capsize=5,
        label="CuPL"
    )

    plt.xticks(x, datasets)
    plt.ylabel("Mean Accuracy (%)")
    plt.title("Mean Accuracy and Variance: Handcrafted Prompts vs CuPL")
    plt.legend()
    plt.tight_layout()

    save_path = "results/mean_variance_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved graph: {save_path}")


if __name__ == "__main__":
    main()