import os
import csv
import matplotlib.pyplot as plt

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_training_curves(dataset_name, train_losses, train_accs):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    mean_loss = sum(train_losses) / len(train_losses)
    mean_acc = sum(train_accs) / len(train_accs)

    loss_path = os.path.join(RESULTS_DIR, f"{dataset_name}_coop_loss_curve.png")
    acc_path = os.path.join(RESULTS_DIR, f"{dataset_name}_coop_accuracy_curve.png")

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Training Loss")
    plt.axhline(y=mean_loss, linestyle="--", label="Mean Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"CoOp Training Loss — {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_path)
    plt.close()

    print("Loss graph saved:", loss_path)

    plt.figure()
    plt.plot(range(1, len(train_accs) + 1), train_accs, marker="o", label="Training Accuracy")
    plt.axhline(y=mean_acc, linestyle="--", label="Mean Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"CoOp Training Accuracy — {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(acc_path)
    plt.close()

    print("Accuracy graph saved:", acc_path)


def plot_comparison_all():
    result_files = [
        "stl10_results.csv",
        "eurosat_results.csv",
        "caltech101_results.csv",
        "flowers102_results.csv",
        "oxfordpets_results.csv"
    ]

    datasets = []
    clip_accs = []
    coop_accs = []

    for file in result_files:
        path = os.path.join(RESULTS_DIR, file)

        if not os.path.exists(path):
            continue

        with open(path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        dataset = rows[0][1]
        clip_acc = float(rows[1][1])
        coop_acc = float(rows[2][1])

        datasets.append(dataset)
        clip_accs.append(clip_acc)
        coop_accs.append(coop_acc)

    if len(datasets) == 0:
        print("No result files available yet.")
        return

    x = range(len(datasets))
    width = 0.35

    graph_path = os.path.join(RESULTS_DIR, "handcrafted_clip_vs_coop.png")

    plt.figure(figsize=(12, 6))

    plt.bar(
        [i - width / 2 for i in x],
        clip_accs,
        width,
        label="Handcrafted CLIP"
    )

    plt.bar(
        [i + width / 2 for i in x],
        coop_accs,
        width,
        label="CoOp"
    )

    plt.xlabel("Dataset")
    plt.ylabel("Accuracy (%)")
    plt.title("Handcrafted CLIP vs CoOp")
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.savefig(graph_path)
    plt.close()

    print("Comparison graph saved:", graph_path)

def generate_comparison_table():
    result_files = [
        "stl10_results.csv",
        "eurosat_results.csv",
        "caltech101_results.csv",
        "flowers102_results.csv",
        "oxfordpets_results.csv"
    ]

    rows_for_csv = [
        ["dataset", "model", "handcrafted_accuracy", "coop_accuracy", "improvement"]
    ]

    for file in result_files:
        path = os.path.join(RESULTS_DIR, file)

        if not os.path.exists(path):
            print(f"Missing file: {file}")
            continue

        with open(path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        dataset = rows[0][1]
        handcrafted_acc = float(rows[1][1])
        coop_acc = float(rows[2][1])
        improvement = coop_acc - handcrafted_acc

        rows_for_csv.append([
            dataset,
            "ViT-B/32",
            round(handcrafted_acc, 2),
            round(coop_acc, 2),
            round(improvement, 2)
        ])

    table_path = os.path.join(RESULTS_DIR, "final_comparison_table.csv")

    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_for_csv)

    print("Final comparison table saved:", table_path)