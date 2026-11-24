"""
data_split.py
Reproducible, stratified splitting for semi-supervised MNIST experiments.
Separates labeled training (4,800), validation (1,200), and unlabeled pool (54,000).
Isolates hidden labels from training/selection.
"""

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, Any, Tuple


def create_semi_supervised_splits(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    labeled_ratio: float = 0.10,
    val_ratio: float = 0.20,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Performs stratified split:
    1. Splits 60,000 into labeled_pool (6,000) and unlabeled_pool (54,000).
    2. Splits labeled_pool (6,000) into labeled_train (4,800) and val (1,200).
    
    Returns a dictionary containing:
    - x_train_labeled (4800, 28, 28)
    - y_train_labeled (4800,)
    - x_val (1200, 28, 28)
    - y_val (1200,)
    - x_unlabeled (54000, 28, 28)
    - y_unlabeled_hidden (54000,) [HIDDEN - for post-hoc diagnostics only]
    - split_indices (dict of integer indices)
    """
    num_total = len(train_labels)
    num_labeled_pool = int(num_total * labeled_ratio)
    
    # Stratified split for labeled pool vs unlabeled pool
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=num_labeled_pool, random_state=random_seed)
    labeled_pool_idx, unlabeled_pool_idx = next(sss1.split(train_images, train_labels))

    # Stratified split of labeled pool into train_labeled and val
    pool_labels = train_labels[labeled_pool_idx]
    num_val = int(len(labeled_pool_idx) * val_ratio)
    num_train_labeled = len(labeled_pool_idx) - num_val

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=num_val, random_state=random_seed)
    train_sub_idx, val_sub_idx = next(sss2.split(labeled_pool_idx, pool_labels))

    train_labeled_idx = labeled_pool_idx[train_sub_idx]
    val_idx = labeled_pool_idx[val_sub_idx]

    splits = {
        "x_train_labeled": train_images[train_labeled_idx],
        "y_train_labeled": train_labels[train_labeled_idx],
        "x_val": train_images[val_idx],
        "y_val": train_labels[val_idx],
        "x_unlabeled": train_images[unlabeled_pool_idx],
        "y_unlabeled_hidden": train_labels[unlabeled_pool_idx],  # Strictly hidden during training
        "indices": {
            "train_labeled": train_labeled_idx,
            "val": val_idx,
            "unlabeled": unlabeled_pool_idx
        },
        "random_seed": random_seed,
        "labeled_ratio": labeled_ratio,
        "val_ratio": val_ratio
    }

    return splits


if __name__ == "__main__":
    from data_loader import load_and_validate_mnist
    data = load_and_validate_mnist()
    splits = create_semi_supervised_splits(data["train_images"], data["train_labels"])
    print("Splits created successfully:")
    print("x_train_labeled:", splits["x_train_labeled"].shape, "y_train_labeled:", splits["y_train_labeled"].shape)
    print("x_val:", splits["x_val"].shape, "y_val:", splits["y_val"].shape)
    print("x_unlabeled:", splits["x_unlabeled"].shape)
    
    # Verify stratification
    classes, counts_tr = np.unique(splits["y_train_labeled"], return_counts=True)
    _, counts_val = np.unique(splits["y_val"], return_counts=True)
    print("Train labeled class counts:", dict(zip(classes.tolist(), counts_tr.tolist())))
    print("Val class counts:", dict(zip(classes.tolist(), counts_val.tolist())))
