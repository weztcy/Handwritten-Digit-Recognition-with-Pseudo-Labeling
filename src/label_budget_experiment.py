"""
label_budget_experiment.py
Runs the optional label-efficiency experiment comparing Supervised Baseline vs Pseudo-Labeling
across multiple labeled-data budgets: 1% (600), 5% (3,000), 10% (6,000), 20% (12,000).
Evaluates on the fixed 10,000-image MNIST test set.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from data_loader import load_and_validate_mnist
from data_split import create_semi_supervised_splits
from model import DigitCNN
from self_training import run_self_training
from train_supervised import train_supervised_model, set_seed


def evaluate_test_acc(model, test_images, test_labels, device):
    model.eval()
    x_test_t = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1) / 255.0
    y_test_t = torch.tensor(test_labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=256, shuffle=False)
    
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.argmax(logits, dim=1)
            preds.append(p.cpu().numpy())
            targets.append(y.numpy())
    return float(accuracy_score(np.concatenate(targets), np.concatenate(preds)))


def run_budget_experiments(budgets=[0.01, 0.05, 0.20], seed=42):
    mnist = load_and_validate_mnist()
    test_imgs = mnist["test_images"]
    test_lbls = mnist["test_labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    
    # Load primary 10% results if available
    eval_metric_path = "outputs/metrics/evaluation_metrics.json"
    if os.path.exists(eval_metric_path):
        with open(eval_metric_path) as f:
            eval_data = json.load(f)
            if "baseline" in eval_data and "semi_supervised" in eval_data:
                results["10%"] = {
                    "budget_ratio": 0.10,
                    "labeled_count": 4800,
                    "baseline_test_acc": eval_data["baseline"]["accuracy"],
                    "ssl_test_acc": eval_data["semi_supervised"]["accuracy"]
                }

    for b in budgets:
        pct_str = f"{int(b*100)}%"
        print(f"\n==========================================")
        print(f"Running Label Budget Experiment: {pct_str} Labeled Data")
        print(f"==========================================")
        
        splits = create_semi_supervised_splits(
            mnist["train_images"],
            mnist["train_labels"],
            labeled_ratio=b,
            val_ratio=0.20,
            random_seed=seed
        )
        
        # 1. Supervised baseline
        save_base_path = f"outputs/models/baseline_{pct_str}.pt"
        base_model, _ = train_supervised_model(
            splits["x_train_labeled"],
            splits["y_train_labeled"],
            splits["x_val"],
            splits["y_val"],
            num_epochs=12,
            batch_size=64,
            lr=1e-3,
            save_path=save_base_path,
            seed=seed
        )
        base_acc = evaluate_test_acc(base_model, test_imgs, test_lbls, device)
        print(f"[{pct_str}] Baseline Test Accuracy: {base_acc*100:.2f}%")

        # 2. Semi-supervised self-training
        save_ssl_path = f"outputs/models/ssl_{pct_str}.pt"
        ssl_hist_path = f"outputs/metrics/ssl_{pct_str}_history.json"
        run_self_training(
            splits=splits,
            max_rounds=2,
            confidence_threshold=0.95,
            pseudo_loss_weight=0.4,
            epochs_per_round=8,
            batch_size=64,
            lr=5e-4,
            seed=seed,
            save_path=save_ssl_path,
            history_path=ssl_hist_path
        )
        
        ssl_model = DigitCNN().to(device)
        ssl_model.load_state_dict(torch.load(save_ssl_path, map_location=device, weights_only=True))
        ssl_acc = evaluate_test_acc(ssl_model, test_imgs, test_lbls, device)
        print(f"[{pct_str}] Semi-Supervised Test Accuracy: {ssl_acc*100:.2f}%")

        results[pct_str] = {
            "budget_ratio": b,
            "labeled_count": len(splits["x_train_labeled"]),
            "baseline_test_acc": float(base_acc),
            "ssl_test_acc": float(ssl_acc),
            "gain_pct": float((ssl_acc - base_acc) * 100)
        }

    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/label_budget_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot budget comparison
    plot_budget_comparison(results)
    return results


def plot_budget_comparison(results):
    # Sort by budget ratio
    sorted_items = sorted(results.items(), key=lambda x: x[1]["budget_ratio"])
    labels = [k for k, v in sorted_items]
    base_accs = [v["baseline_test_acc"] * 100 for k, v in sorted_items]
    ssl_accs = [v["ssl_test_acc"] * 100 for k, v in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    w = 0.35

    ax.bar(x - w/2, base_accs, width=w, label="Supervised Baseline", color="#1f77b4", edgecolor="black")
    ax.bar(x + w/2, ssl_accs, width=w, label="Semi-Supervised (Self-Training)", color="#2ca02c", edgecolor="black")

    ax.set_title("Test Accuracy Across Labeled Data Budgets (1%, 5%, 10%, 20%)", weight="bold", fontsize=13)
    ax.set_xlabel("Labeled Data Budget (% of Training Pool)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n({results[k]['labeled_count']:,} samples)" for k in labels])
    ax.set_ylim(min(base_accs) - 3, 100.5)
    ax.legend(frameon=True, loc="lower right")

    # Add gain annotations
    for i in range(len(labels)):
        diff = ssl_accs[i] - base_accs[i]
        h = max(ssl_accs[i], base_accs[i])
        ax.text(x[i] + w/2, h + 0.3, f"{diff:+.2f}%", ha="center", va="bottom", fontsize=9, weight="bold", color="#1b5e20")

    plt.tight_layout()
    path = "outputs/figures/13_label_budget_comparison.png"
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    run_budget_experiments()
