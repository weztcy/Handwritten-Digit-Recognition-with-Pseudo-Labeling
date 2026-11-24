"""
train_supervised.py
Trains the supervised baseline CNN using only the 4,800 labeled training examples.
Also supports training the upper-bound reference benchmark using all available labels.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_loader import load_and_validate_mnist
from data_split import create_semi_supervised_splits
from model import DigitCNN


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def train_supervised_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str = "outputs/models/baseline_model.pt",
    seed: int = 42
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training supervised model on {device} with {len(x_train)} samples...")

    # Normalize to [0, 1] and add channel dimension (N, 1, 28, 28)
    x_tr_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0
    y_tr_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1) / 255.0
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(x_tr_tensor, y_tr_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = -1.0
    best_weights = None

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(tr_loss, 5))
        history["train_acc"].append(round(tr_acc, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 5))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()

        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} {'*' if val_acc == best_val_acc else ''}")

    # Load best weights
    model.load_state_dict(best_weights)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_weights, save_path)
    print(f"Best model saved to {save_path} with Val Acc: {best_val_acc:.4f}")

    return model, history


def main():
    mnist_data = load_and_validate_mnist()
    splits = create_semi_supervised_splits(
        mnist_data["train_images"],
        mnist_data["train_labels"],
        labeled_ratio=0.10,
        val_ratio=0.20,
        random_seed=42
    )

    # 1. Train Baseline (10% labeled pool: 4800 train, 1200 val)
    print("\n" + "="*50)
    print("STEP 1: Training Supervised Baseline (10% Labeled Budget)")
    print("="*50)
    baseline_model, baseline_history = train_supervised_model(
        splits["x_train_labeled"],
        splits["y_train_labeled"],
        splits["x_val"],
        splits["y_val"],
        num_epochs=15,
        batch_size=64,
        lr=1e-3,
        save_path="outputs/models/baseline_model.pt",
        seed=42
    )

    with open("outputs/metrics/baseline_history.json", "w") as f:
        json.dump(baseline_history, f, indent=2)

    # 2. Train Upper-Bound Benchmark (Full 60,000 dataset: 48,000 train, 12,000 val for fair comparison)
    print("\n" + "="*50)
    print("STEP 2: Training Upper-Bound Supervised Benchmark (100% Labeled Data)")
    print("="*50)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=12000, random_state=42)
    full_tr_idx, full_val_idx = next(sss.split(mnist_data["train_images"], mnist_data["train_labels"]))
    
    upper_bound_model, upper_history = train_supervised_model(
        mnist_data["train_images"][full_tr_idx],
        mnist_data["train_labels"][full_tr_idx],
        mnist_data["train_images"][full_val_idx],
        mnist_data["train_labels"][full_val_idx],
        num_epochs=15,
        batch_size=128,
        lr=1e-3,
        save_path="outputs/models/upper_bound_model.pt",
        seed=42
    )

    with open("outputs/metrics/upper_bound_history.json", "w") as f:
        json.dump(upper_history, f, indent=2)


if __name__ == "__main__":
    main()
