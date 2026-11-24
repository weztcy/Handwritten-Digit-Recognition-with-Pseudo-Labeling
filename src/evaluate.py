"""
evaluate.py
Evaluates models (Supervised Baseline, Semi-Supervised, and Upper-Bound Reference)
on the official 10,000-image MNIST test set.
Computes accuracy, macro/per-class precision, recall, F1, and confusion matrices.
Extracts misclassified examples for qualitative error analysis.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any

from data_loader import load_and_validate_mnist
from model import DigitCNN


def evaluate_model_on_test(
    model: torch.nn.Module,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    batch_size: int = 256
) -> Dict[str, Any]:
    model.eval()
    x_test_t = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1) / 255.0
    y_test_t = torch.tensor(test_labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    all_confs = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.numpy())
            all_confs.append(confs.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    confs = np.concatenate(all_confs)
    probs = np.concatenate(all_probs, axis=0)

    acc = float(accuracy_score(targets, preds))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(targets, preds, average="macro")
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(targets, preds, average=None)
    cm = confusion_matrix(targets, preds)

    # Identify errors
    error_indices = np.where(preds != targets)[0]
    errors = []
    for idx in error_indices[:100]:  # store up to 100 sample errors
        errors.append({
            "index": int(idx),
            "true_label": int(targets[idx]),
            "predicted_label": int(preds[idx]),
            "confidence": float(confs[idx]),
            "probabilities": [float(p) for p in probs[idx]]
        })

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": {
            str(i): {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i])
            } for i in range(10)
        },
        "confusion_matrix": cm.tolist(),
        "total_errors": int(len(error_indices)),
        "error_samples": errors
    }


def main():
    mnist_data = load_and_validate_mnist()
    test_images = mnist_data["test_images"]
    test_labels = mnist_data["test_labels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models on test set ({len(test_images)} images) using {device}...")

    results = {}

    # 1. Supervised Baseline
    baseline_path = "outputs/models/baseline_model.pt"
    if os.path.exists(baseline_path):
        baseline_model = DigitCNN().to(device)
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))
        baseline_results = evaluate_model_on_test(baseline_model, test_images, test_labels, device)
        results["baseline"] = baseline_results
        print(f"Supervised Baseline Test Acc: {baseline_results['accuracy']:.4f} | Macro F1: {baseline_results['macro_f1']:.4f}")

    # 2. Semi-Supervised Model
    ssl_path = "outputs/models/semisupervised_model.pt"
    if os.path.exists(ssl_path):
        ssl_model = DigitCNN().to(device)
        ssl_model.load_state_dict(torch.load(ssl_path, map_location=device, weights_only=True))
        ssl_results = evaluate_model_on_test(ssl_model, test_images, test_labels, device)
        results["semi_supervised"] = ssl_results
        print(f"Semi-Supervised Test Acc:    {ssl_results['accuracy']:.4f} | Macro F1: {ssl_results['macro_f1']:.4f}")

    # 3. Upper Bound Supervised Benchmark (if available)
    upper_path = "outputs/models/upper_bound_model.pt"
    if os.path.exists(upper_path):
        upper_model = DigitCNN().to(device)
        upper_model.load_state_dict(torch.load(upper_path, map_location=device, weights_only=True))
        upper_results = evaluate_model_on_test(upper_model, test_images, test_labels, device)
        results["upper_bound"] = upper_results
        print(f"Upper-Bound Benchmark Test Acc: {upper_results['accuracy']:.4f} | Macro F1: {upper_results['macro_f1']:.4f}")

    # Compare Baseline vs Semi-Supervised
    if "baseline" in results and "semi_supervised" in results:
        base_acc = results["baseline"]["accuracy"]
        ssl_acc = results["semi_supervised"]["accuracy"]
        delta_acc = ssl_acc - base_acc
        rel_gain = (delta_acc / base_acc) * 100
        base_err = results["baseline"]["total_errors"]
        ssl_err = results["semi_supervised"]["total_errors"]
        err_reduction = ((base_err - ssl_err) / base_err) * 100

        results["comparison"] = {
            "absolute_acc_diff": float(delta_acc),
            "relative_acc_gain_pct": float(rel_gain),
            "baseline_error_count": base_err,
            "ssl_error_count": ssl_err,
            "error_reduction_pct": float(err_reduction)
        }
        print("\n" + "="*50)
        print("COMPARISON: Supervised Baseline vs Semi-Supervised")
        print("="*50)
        print(f"Baseline Accuracy:        {base_acc*100:.2f}% (Errors: {base_err})")
        print(f"Semi-Supervised Accuracy: {ssl_acc*100:.2f}% (Errors: {ssl_err})")
        print(f"Absolute Accuracy Change: {delta_acc*100:+.2f}%")
        print(f"Relative Gain:            {rel_gain:+.2f}%")
        print(f"Test Error Reduction:     {err_reduction:+.2f}%")

    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved evaluation metrics to outputs/metrics/evaluation_metrics.json")


if __name__ == "__main__":
    main()
