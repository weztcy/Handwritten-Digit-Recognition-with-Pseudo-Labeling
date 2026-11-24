"""
self_training.py
Iterative Confidence-Based Pseudo-Labeling / Self-Training pipeline.
Trains semi-supervised model using 4,800 labeled + 54,000 unlabeled images.
Follows strict label-isolation rules and saves extensive diagnostic metrics.
"""

import os
import json
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, Any, List, Tuple

from data_loader import load_and_validate_mnist
from data_split import create_semi_supervised_splits
from model import DigitCNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WeightedDataset(Dataset):
    """
    Dataset that handles both genuine labeled data (weight=1.0)
    and pseudo-labeled data (weight=alpha).
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray):
        # x is (N, 28, 28) uint8 -> convert to float tensor [0, 1] with shape (N, 1, 28, 28)
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1) / 255.0
        self.y = torch.tensor(y, dtype=torch.long)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.weights[idx]


def predict_unlabeled_pool(
    model: nn.Module,
    x_unlabeled: np.ndarray,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu")
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts probabilities and class labels for unlabeled pool.
    Returns:
    - max_probs: 1D array of maximum class probabilities
    - pred_labels: 1D array of predicted class indices
    """
    model.eval()
    x_tensor = torch.tensor(x_unlabeled, dtype=torch.float32).unsqueeze(1) / 255.0
    loader = DataLoader(x_tensor, batch_size=batch_size, shuffle=False)

    all_max_probs = []
    all_preds = []

    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            max_p, preds = torch.max(probs, dim=1)
            all_max_probs.append(max_p.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_max_probs), np.concatenate(all_preds)


def train_weighted_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y, w in loader:
        x, y, w = x.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model(x)
        unweighted_loss = criterion(logits, y)
        loss = (unweighted_loss * w).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def evaluate_val(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def run_self_training(
    splits: Dict[str, Any],
    max_rounds: int = 3,
    confidence_threshold: float = 0.95,
    pseudo_loss_weight: float = 0.4,
    epochs_per_round: int = 6,
    batch_size: int = 128,
    lr: float = 5e-4,
    seed: int = 42,
    save_path: str = "outputs/models/semisupervised_model.pt",
    history_path: str = "outputs/metrics/self_training_history.json"
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\nStarting Self-Training with Confidence-Based Pseudo-Labeling\n{'='*60}")
    print(f"Device: {device} | Max Rounds: {max_rounds} | Threshold: {confidence_threshold} | Pseudo Weight: {pseudo_loss_weight}")

    # Labeled data
    x_train_labeled = splits["x_train_labeled"]
    y_train_labeled = splits["y_train_labeled"]

    # Validation data
    x_val = splits["x_val"]
    y_val = splits["y_val"]
    val_tensor_x = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1) / 255.0
    val_tensor_y = torch.tensor(y_val, dtype=torch.long)
    val_loader = DataLoader(TensorDataset(val_tensor_x, val_tensor_y), batch_size=batch_size, shuffle=False)

    # Unlabeled pool
    # Remaining unlabeled data tracked with copies
    x_unlabeled_pool = splits["x_unlabeled"].copy()
    y_unlabeled_hidden_pool = splits["y_unlabeled_hidden"].copy() # STRICTLY ISOLATED FROM TRAINING

    total_unlabeled_initial = len(x_unlabeled_pool)

    # Initialize model from scratch or load trained baseline
    baseline_path = "outputs/models/baseline_model.pt"
    model = DigitCNN().to(device)
    if os.path.exists(baseline_path):
        print(f"Initializing self-training model from baseline weights: {baseline_path}")
        model.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))
    else:
        print("Warning: Baseline weights not found. Initializing random weights.")

    # Measure initial baseline validation score
    init_val_loss, init_val_acc = evaluate_val(model, val_loader, device)
    print(f"Initial Baseline Validation Loss: {init_val_loss:.4f} | Acc: {init_val_acc:.4f}")

    best_val_acc = init_val_acc
    best_weights = copy.deepcopy(model.state_dict())

    # Accumulated pseudo-labeled dataset
    x_accumulated_pseudo = np.empty((0, 28, 28), dtype=np.uint8)
    y_accumulated_pseudo = np.empty((0,), dtype=np.int64)

    history = {
        "config": {
            "max_rounds": max_rounds,
            "confidence_threshold": confidence_threshold,
            "pseudo_loss_weight": pseudo_loss_weight,
            "epochs_per_round": epochs_per_round,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "initial_labeled_count": len(x_train_labeled),
            "initial_unlabeled_count": total_unlabeled_initial
        },
        "initial_val_acc": float(init_val_acc),
        "initial_val_loss": float(init_val_loss),
        "rounds": [],
        "epoch_curves": {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
    }

    cumulative_epoch = 0

    for r in range(1, max_rounds + 1):
        print(f"\n--- Round {r}/{max_rounds} ---")
        if len(x_unlabeled_pool) == 0:
            print("No remaining unlabeled samples. Stopping self-training.")
            break

        # Step 1: Predict probabilities for remaining unlabeled pool
        max_probs, pred_labels = predict_unlabeled_pool(model, x_unlabeled_pool, batch_size=256, device=device)

        # Step 2: Threshold filtering
        accepted_mask = max_probs >= confidence_threshold
        num_accepted = int(np.sum(accepted_mask))
        num_remaining_before = len(x_unlabeled_pool)
        pct_accepted_total = float(num_accepted / total_unlabeled_initial * 100)

        print(f"Unlabeled candidates: {num_remaining_before:,}")
        print(f"Accepted with confidence >= {confidence_threshold}: {num_accepted:,} ({pct_accepted_total:.2f}% of total initial unlabeled pool)")

        if num_accepted == 0:
            print("No new pseudo-labels met the confidence threshold. Stopping self-training early.")
            break

        # Extract accepted samples and their predicted labels
        accepted_x = x_unlabeled_pool[accepted_mask]
        accepted_y = pred_labels[accepted_mask]
        accepted_confs = max_probs[accepted_mask]

        # Post-hoc diagnostic: check precision against hidden labels
        # Strictly for logging/diagnostic analysis, not used anywhere in training!
        hidden_ground_truth = y_unlabeled_hidden_pool[accepted_mask]
        post_hoc_correct = int(np.sum(accepted_y == hidden_ground_truth))
        post_hoc_accuracy = float(post_hoc_correct / num_accepted * 100)
        print(f"[POST-HOC DIAGNOSTIC] Pseudo-label accuracy vs hidden ground truth: {post_hoc_accuracy:.2f}% ({post_hoc_correct}/{num_accepted})")

        # Class distribution of accepted pseudo-labels
        unique_classes, counts = np.unique(accepted_y, return_counts=True)
        class_dist = {int(c): int(cnt) for c, cnt in zip(unique_classes, counts)}
        # Ensure all 10 digits appear in dict
        for d in range(10):
            if d not in class_dist:
                class_dist[d] = 0
        print(f"Accepted class distribution: {class_dist}")

        # Remove accepted from unlabeled pool to prevent re-selection
        x_unlabeled_pool = x_unlabeled_pool[~accepted_mask]
        y_unlabeled_hidden_pool = y_unlabeled_hidden_pool[~accepted_mask]
        num_remaining_after = len(x_unlabeled_pool)
        print(f"Remaining unlabeled samples for subsequent rounds: {num_remaining_after:,}")

        # Accumulate into pseudo-labeled training set
        x_accumulated_pseudo = np.concatenate([x_accumulated_pseudo, accepted_x], axis=0) if len(x_accumulated_pseudo) > 0 else accepted_x
        y_accumulated_pseudo = np.concatenate([y_accumulated_pseudo, accepted_y], axis=0) if len(y_accumulated_pseudo) > 0 else accepted_y

        # Form combined training set
        total_train_x = np.concatenate([x_train_labeled, x_accumulated_pseudo], axis=0)
        total_train_y = np.concatenate([y_train_labeled, y_accumulated_pseudo], axis=0)

        weights_labeled = np.ones(len(x_train_labeled), dtype=np.float32)
        weights_pseudo = np.full(len(x_accumulated_pseudo), pseudo_loss_weight, dtype=np.float32)
        total_weights = np.concatenate([weights_labeled, weights_pseudo], axis=0)

        print(f"Joint Training Pool: {len(x_train_labeled):,} genuine labeled (w=1.0) + {len(x_accumulated_pseudo):,} pseudo-labeled (w={pseudo_loss_weight}) = {len(total_train_x):,} total")

        train_dataset = WeightedDataset(total_train_x, total_train_y, total_weights)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Fine-tune model on joint dataset
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        round_best_val_acc = -1.0
        round_best_weights = None

        for ep in range(1, epochs_per_round + 1):
            cumulative_epoch += 1
            tr_loss, tr_acc = train_weighted_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc = evaluate_val(model, val_loader, device)

            history["epoch_curves"]["epoch"].append(cumulative_epoch)
            history["epoch_curves"]["train_loss"].append(round(tr_loss, 5))
            history["epoch_curves"]["train_acc"].append(round(tr_acc, 5))
            history["epoch_curves"]["val_loss"].append(round(val_loss, 5))
            history["epoch_curves"]["val_acc"].append(round(val_acc, 5))

            if val_acc > round_best_val_acc:
                round_best_val_acc = val_acc
                round_best_weights = copy.deepcopy(model.state_dict())

            print(f"  Round {r} Ep {ep:02d}/{epochs_per_round:02d} | Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} {'*' if val_acc == round_best_val_acc else ''}")

        # Check if overall validation accuracy improved
        if round_best_val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.4f} to {round_best_val_acc:.4f}! Updating best checkpoint.")
            best_val_acc = round_best_val_acc
            best_weights = copy.deepcopy(round_best_weights)
            model.load_state_dict(round_best_weights)
        else:
            print(f"Round {r} peak val acc ({round_best_val_acc:.4f}) did not exceed previous best ({best_val_acc:.4f}). Reverting to best checkpoint.")
            model.load_state_dict(best_weights)

        # Record round diagnostics
        round_info = {
            "round": r,
            "threshold": confidence_threshold,
            "num_candidates_before": num_remaining_before,
            "num_accepted": num_accepted,
            "pct_accepted_of_total": pct_accepted_total,
            "num_remaining_after": num_remaining_after,
            "total_accumulated_pseudo": len(x_accumulated_pseudo),
            "class_distribution": class_dist,
            "mean_confidence": float(np.mean(accepted_confs)),
            "confidences_sample": [round(float(c), 4) for c in accepted_confs[:1000]], # sample for histogram plotting
            "post_hoc_correct": post_hoc_correct,
            "post_hoc_accuracy": post_hoc_accuracy,
            "val_acc_after_round": round(float(round_best_val_acc), 5)
        }
        history["rounds"].append(round_info)

    # Save final best model based on validation
    model.load_state_dict(best_weights)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_weights, save_path)
    print(f"\nFinal Best Semi-Supervised Model saved to {save_path} with Val Acc: {best_val_acc:.4f}")

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


def main():
    mnist_data = load_and_validate_mnist()
    splits = create_semi_supervised_splits(
        mnist_data["train_images"],
        mnist_data["train_labels"],
        labeled_ratio=0.10,
        val_ratio=0.20,
        random_seed=42
    )

    run_self_training(
        splits=splits,
        max_rounds=3,
        confidence_threshold=0.95,
        pseudo_loss_weight=0.4,
        epochs_per_round=6,
        batch_size=128,
        lr=5e-4,
        seed=42
    )


if __name__ == "__main__":
    main()
