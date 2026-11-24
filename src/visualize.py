"""
visualize.py
Generates all 12 mandatory publication-quality visualizations for the report.
Plots are saved to outputs/figures/ with clear labels, legends, and styling.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from data_loader import load_and_validate_mnist
from data_split import create_semi_supervised_splits

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_01_mnist_samples(mnist_data: Dict[str, Any]):
    """1. Grid of representative MNIST images with their digit labels."""
    images = mnist_data["train_images"]
    labels = mnist_data["train_labels"]

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    fig.suptitle("Representative MNIST Handwritten Digit Samples (0-9)", fontsize=14, weight="bold")

    # Pick first occurrence of each digit
    for digit in range(10):
        idx = np.where(labels == digit)[0][0]
        ax = axes[digit // 5, digit % 5]
        ax.imshow(images[idx], cmap="gray_r")
        ax.set_title(f"Digit: {digit}", fontsize=12, weight="bold")
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_mnist_samples_grid.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_02_train_class_distribution(mnist_data: Dict[str, Any]):
    """2. Training-set class distribution (bar chart)."""
    labels = mnist_data["train_labels"]
    digits, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(digits, counts, color="#2b5c8f", width=0.6, edgecolor="black", alpha=0.85)
    ax.set_title("Training Set Class Distribution (N = 60,000)", fontsize=13, weight="bold")
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(digits)
    ax.set_ylim(0, max(counts) * 1.15)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 80, f"{h:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_train_class_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_03_data_split_summary(splits: Dict[str, Any]):
    """3. Labeled versus unlabeled data split (chart or clear visual summary)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.2]})

    # Donut chart of Overall Split
    sizes_overall = [4800, 1200, 54000]
    labels_overall = ["Labeled Train\n(4,800 / 8%)", "Validation\n(1,200 / 2%)", "Unlabeled Pool\n(54,000 / 90%)"]
    colors_overall = ["#1f77b4", "#aec7e8", "#ff7f0e"]

    wedges, texts, autotexts = ax1.pie(
        sizes_overall,
        labels=labels_overall,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors_overall,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_weight("bold")
    ax1.set_title("Total Training Set Split (60,000 Images)", weight="bold", fontsize=12)

    # Stratified breakdown across digits for labeled train vs val
    digits = np.arange(10)
    tr_y = splits["y_train_labeled"]
    val_y = splits["y_val"]
    _, tr_counts = np.unique(tr_y, return_counts=True)
    _, val_counts = np.unique(val_y, return_counts=True)

    ax2.bar(digits - 0.18, tr_counts, width=0.35, label="Labeled Train (4,800)", color="#1f77b4", edgecolor="black")
    ax2.bar(digits + 0.18, val_counts, width=0.35, label="Validation (1,200)", color="#aec7e8", edgecolor="black")
    ax2.set_title("Stratified Digit Distribution in Labeled Pool", weight="bold", fontsize=12)
    ax2.set_xlabel("Digit Class")
    ax2.set_ylabel("Count")
    ax2.set_xticks(digits)
    ax2.legend(frameon=True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_data_split_summary.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_04_05_learning_curves(baseline_hist: Dict[str, Any], ssl_hist: Dict[str, Any]):
    """4. Loss curves & 5. Accuracy curves."""
    # 4. Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Baseline curves
    ax1.plot(baseline_hist["epoch"], baseline_hist["train_loss"], marker="o", label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(baseline_hist["epoch"], baseline_hist["val_loss"], marker="s", label="Val Loss", color="#d62728", lw=2)
    ax1.set_title("Supervised Baseline: Loss Curves (4,800 samples)", weight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend(frameon=True)
    ax1.set_ylim(bottom=0)

    # SSL curves
    ssl_epochs = ssl_hist["epoch_curves"]["epoch"]
    ax2.plot(ssl_epochs, ssl_hist["epoch_curves"]["train_loss"], marker="o", label="Train Loss (Joint)", color="#2ca02c", lw=2)
    ax2.plot(ssl_epochs, ssl_hist["epoch_curves"]["val_loss"], marker="s", label="Val Loss (Held-out 1,200)", color="#d62728", lw=2)
    # Add vertical lines for self-training rounds
    round_size = ssl_hist["config"]["epochs_per_round"]
    for r in range(1, len(ssl_hist["rounds"]) + 1):
        ax2.axvline(x=r * round_size, color="gray", linestyle="--", alpha=0.7)
        ax2.text(r * round_size - round_size/2, max(ssl_hist["epoch_curves"]["val_loss"]) * 0.9, f"Round {r}", ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax2.set_title("Semi-Supervised: Iterative Loss Curves", weight="bold")
    ax2.set_xlabel("Cumulative Epoch")
    ax2.set_ylabel("Weighted Cross Entropy Loss")
    ax2.legend(frameon=True)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    path_loss = os.path.join(OUTPUT_DIR, "04_training_val_loss_curves.png")
    plt.savefig(path_loss, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_loss}")

    # 5. Accuracy curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(baseline_hist["epoch"], [a * 100 for a in baseline_hist["train_acc"]], marker="o", label="Train Accuracy", color="#1f77b4", lw=2)
    ax1.plot(baseline_hist["epoch"], [a * 100 for a in baseline_hist["val_acc"]], marker="s", label="Val Accuracy", color="#d62728", lw=2)
    ax1.set_title("Supervised Baseline: Accuracy Curves", weight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(85, 101)
    ax1.legend(frameon=True, loc="lower right")

    ax2.plot(ssl_epochs, [a * 100 for a in ssl_hist["epoch_curves"]["train_acc"]], marker="o", label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(ssl_epochs, [a * 100 for a in ssl_hist["epoch_curves"]["val_acc"]], marker="s", label="Val Accuracy", color="#d62728", lw=2)
    for r in range(1, len(ssl_hist["rounds"]) + 1):
        ax2.axvline(x=r * round_size, color="gray", linestyle="--", alpha=0.7)
        ax2.text(r * round_size - round_size/2, 88, f"Round {r}", ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax2.set_title("Semi-Supervised: Iterative Accuracy Curves", weight="bold")
    ax2.set_xlabel("Cumulative Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(85, 101)
    ax2.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    path_acc = os.path.join(OUTPUT_DIR, "05_training_val_accuracy_curves.png")
    plt.savefig(path_acc, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_acc}")


def plot_06_pseudolabel_acceptance(ssl_hist: Dict[str, Any]):
    """6. Pseudo-label acceptance count by self-training round."""
    rounds = [f"Round {r['round']}" for r in ssl_hist["rounds"]]
    accepted = [r["num_accepted"] for r in ssl_hist["rounds"]]
    pcts = [r["pct_accepted_of_total"] for r in ssl_hist["rounds"]]
    precisions = [r["post_hoc_accuracy"] for r in ssl_hist["rounds"]]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color1 = "#1f77b4"
    bars = ax1.bar(rounds, accepted, color=color1, width=0.45, edgecolor="black", alpha=0.85, label="Accepted Samples")
    ax1.set_ylabel("Accepted Samples Count", color=color1, weight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, max(accepted) * 1.25)
    ax1.set_title("Pseudo-Label Acceptance Count & Precision by Round", fontsize=13, weight="bold")

    for bar, pct in zip(bars, pcts):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 500, f"{h:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, weight="bold")

    # Second axis for Post-hoc Precision
    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.plot(rounds, precisions, color=color2, marker="D", markersize=8, lw=2.5, label="Post-hoc Precision vs Hidden GT (%)")
    ax2.set_ylabel("Post-Hoc Diagnostic Accuracy (%)", color=color2, weight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(90, 102)

    for i, prec in enumerate(precisions):
        ax2.text(i, prec + 0.6, f"{prec:.2f}%", color=color2, ha="center", weight="bold", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_pseudolabel_acceptance_by_round.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_07_pseudolabel_class_distribution(ssl_hist: Dict[str, Any]):
    """7. Pseudo-label class distribution for each round."""
    rounds = ssl_hist["rounds"]
    digits = np.arange(10)
    num_rounds = len(rounds)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / num_rounds
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, r in enumerate(rounds):
        dist = r["class_distribution"]
        counts = [dist.get(str(d), dist.get(d, 0)) for d in digits]
        offset = (i - (num_rounds - 1) / 2) * width
        ax.bar(digits + offset, counts, width=width, label=f"Round {r['round']} (Total: {r['num_accepted']:,})", color=colors[i % len(colors)], edgecolor="black", alpha=0.85)

    ax.set_title("Pseudo-Label Class Distribution by Self-Training Round", weight="bold", fontsize=13)
    ax.set_xlabel("Predicted Digit Class")
    ax.set_ylabel("Accepted Pseudo-Labels Count")
    ax.set_xticks(digits)
    ax.legend(frameon=True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "07_pseudolabel_class_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_08_confidence_distribution(ssl_hist: Dict[str, Any]):
    """8. Confidence-score distribution for accepted pseudo-labels."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, r in enumerate(ssl_hist["rounds"]):
        confs = r.get("confidences_sample", [])
        if confs:
            sns.kdeplot(confs, ax=ax, label=f"Round {r['round']} (Mean: {r['mean_confidence']:.4f})", color=colors[i % len(colors)], lw=2, fill=True, alpha=0.2)

    threshold = ssl_hist["config"]["confidence_threshold"]
    ax.axvline(x=threshold, color="black", linestyle="--", lw=1.5, label=f"Confidence Threshold ({threshold})")
    ax.set_title(f"Confidence-Score Distribution of Accepted Pseudo-Labels (Threshold = {threshold})", weight="bold")
    ax.set_xlabel("Predicted Probability (Confidence)")
    ax.set_ylabel("Density")
    ax.set_xlim(threshold - 0.01, 1.002)
    ax.legend(frameon=True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "08_confidence_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_09_test_accuracy_comparison(eval_results: Dict[str, Any]):
    """9. Test accuracy comparison between supervised baseline, semi-supervised, and optional upper bound."""
    models = []
    accuracies = []
    colors = []

    if "baseline" in eval_results:
        models.append("Supervised Baseline\n(10% Labeled: 4,800)")
        accuracies.append(eval_results["baseline"]["accuracy"] * 100)
        colors.append("#1f77b4")

    if "semi_supervised" in eval_results:
        models.append("Semi-Supervised Model\n(10% Labeled + Pseudo-Labels)")
        accuracies.append(eval_results["semi_supervised"]["accuracy"] * 100)
        colors.append("#2ca02c")

    if "upper_bound" in eval_results:
        models.append("Upper-Bound Benchmark\n(100% Labeled: 48,000)")
        accuracies.append(eval_results["upper_bound"]["accuracy"] * 100)
        colors.append("#7f7f7f")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, accuracies, color=colors, width=0.5, edgecolor="black", alpha=0.9)
    ax.set_title("Test Accuracy Comparison on Official Test Set (N = 10,000)", weight="bold", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(90, 100.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.15, f"{h:.2f}%", ha="center", va="bottom", fontsize=11, weight="bold")

    # Annotation of gain
    if len(accuracies) >= 2:
        diff = accuracies[1] - accuracies[0]
        ax.annotate(
            f"Pseudo-labeling Gain:\n{diff:+.2f}% absolute",
            xy=(0.5, (accuracies[0] + accuracies[1]) / 2),
            xycoords=("axes fraction", "data"),
            xytext=(0.5, 93.5),
            ha="center",
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f4f8", edgecolor="#1f77b4", lw=1.5)
        )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "09_test_accuracy_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_10_confusion_matrices(eval_results: Dict[str, Any]):
    """10. Confusion matrices for the supervised baseline and final semi-supervised model."""
    cm_base = np.array(eval_results["baseline"]["confusion_matrix"])
    cm_ssl = np.array(eval_results["semi_supervised"]["confusion_matrix"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", ax=ax1, cbar=False, annot_kws={"size": 9})
    ax1.set_title(f"Baseline CNN Confusion Matrix (Acc: {eval_results['baseline']['accuracy']*100:.2f}%)", weight="bold")
    ax1.set_xlabel("Predicted Digit")
    ax1.set_ylabel("True Digit")

    sns.heatmap(cm_ssl, annot=True, fmt="d", cmap="Greens", ax=ax2, cbar=False, annot_kws={"size": 9})
    ax2.set_title(f"Semi-Supervised CNN Confusion Matrix (Acc: {eval_results['semi_supervised']['accuracy']*100:.2f}%)", weight="bold")
    ax2.set_xlabel("Predicted Digit")
    ax2.set_ylabel("True Digit")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "10_confusion_matrices.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_11_per_class_f1(eval_results: Dict[str, Any]):
    """11. Per-class F1-score comparison."""
    digits = np.arange(10)
    base_f1 = [eval_results["baseline"]["per_class"][str(d)]["f1"] * 100 for d in digits]
    ssl_f1 = [eval_results["semi_supervised"]["per_class"][str(d)]["f1"] * 100 for d in digits]

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.38
    b1 = ax.bar(digits - w/2, base_f1, width=w, label="Supervised Baseline (10%)", color="#1f77b4", edgecolor="black")
    b2 = ax.bar(digits + w/2, ssl_f1, width=w, label="Semi-Supervised (Self-Training)", color="#2ca02c", edgecolor="black")

    ax.set_title("Per-Class F1-Score Comparison on Test Set", weight="bold", fontsize=13)
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("F1-Score (%)")
    ax.set_xticks(digits)
    ax.set_ylim(94, 101)
    ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "11_per_class_f1_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_12_error_analysis(mnist_data: Dict[str, Any], eval_results: Dict[str, Any]):
    """12. A grid of representative test errors from the final model, showing true label, predicted label, and confidence."""
    test_images = mnist_data["test_images"]
    errors = eval_results["semi_supervised"]["error_samples"]

    # Show up to 10 representative errors
    num_errors_to_show = min(10, len(errors))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
    fig.suptitle(f"Semi-Supervised Model: Representative Test Misclassifications (Total Errors: {eval_results['semi_supervised']['total_errors']})", weight="bold", fontsize=13)

    for i in range(10):
        ax = axes[i // 5, i % 5]
        if i < num_errors_to_show:
            err = errors[i]
            img = test_images[err["index"]]
            ax.imshow(img, cmap="gray_r")
            ax.set_title(f"True: {err['true_label']} | Pred: {err['predicted_label']}\nConf: {err['confidence']:.2%}", fontsize=9, color="#b30000", weight="bold")
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "12_test_error_analysis.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def generate_all_visualizations():
    mnist_data = load_and_validate_mnist()
    splits = create_semi_supervised_splits(
        mnist_data["train_images"],
        mnist_data["train_labels"],
        labeled_ratio=0.10,
        val_ratio=0.20,
        random_seed=42
    )

    with open("outputs/metrics/baseline_history.json") as f:
        baseline_hist = json.load(f)
    with open("outputs/metrics/self_training_history.json") as f:
        ssl_hist = json.load(f)
    with open("outputs/metrics/evaluation_metrics.json") as f:
        eval_results = json.load(f)

    print("Generating all 12 mandatory visualizations...")
    plot_01_mnist_samples(mnist_data)
    plot_02_train_class_distribution(mnist_data)
    plot_03_data_split_summary(splits)
    plot_04_05_learning_curves(baseline_hist, ssl_hist)
    plot_06_pseudolabel_acceptance(ssl_hist)
    plot_07_pseudolabel_class_distribution(ssl_hist)
    plot_08_confidence_distribution(ssl_hist)
    plot_09_test_accuracy_comparison(eval_results)
    plot_10_confusion_matrices(eval_results)
    plot_11_per_class_f1(eval_results)
    plot_12_error_analysis(mnist_data, eval_results)
    print("All visualizations successfully generated in outputs/figures/!")


if __name__ == "__main__":
    generate_all_visualizations()
