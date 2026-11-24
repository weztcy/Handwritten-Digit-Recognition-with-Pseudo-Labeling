"""
data_loader.py
IDX file reader and validator for MNIST dataset.
Supports reading compressed (.gz) IDX files directly.
"""

import gzip
import struct
import numpy as np
import os
from typing import Tuple, Dict, Any


def read_idx_images(filepath: str) -> np.ndarray:
    """
    Reads an IDX image file (.gz) and returns uint8 numpy array of shape (N, 28, 28).
    Magic number for 3D image IDX is 2051 (0x00000803).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with gzip.open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {filepath}, expected 2051.")
        
        buffer = f.read(num_images * rows * cols)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape((num_images, rows, cols))
        return data


def read_idx_labels(filepath: str) -> np.ndarray:
    """
    Reads an IDX label file (.gz) and returns uint8 numpy array of shape (N,).
    Magic number for 1D label IDX is 2049 (0x00000801).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with gzip.open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {filepath}, expected 2049.")
        
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels


def load_and_validate_mnist(data_dir: str = "dataset") -> Dict[str, Any]:
    """
    Loads training and test sets and performs comprehensive programmatic validation.
    Returns dictionary with data arrays and validation metadata.
    """
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    train_images = read_idx_images(train_images_path)
    train_labels = read_idx_labels(train_labels_path)
    test_images = read_idx_images(test_images_path)
    test_labels = read_idx_labels(test_labels_path)

    # Validations
    assert train_images.shape == (60000, 28, 28), f"Unexpected train_images shape: {train_images.shape}"
    assert train_labels.shape == (60000,), f"Unexpected train_labels shape: {train_labels.shape}"
    assert test_images.shape == (10000, 28, 28), f"Unexpected test_images shape: {test_images.shape}"
    assert test_labels.shape == (10000,), f"Unexpected test_labels shape: {test_labels.shape}"

    assert train_images.dtype == np.uint8, f"Unexpected train_images dtype: {train_images.dtype}"
    assert train_labels.dtype == np.uint8, f"Unexpected train_labels dtype: {train_labels.dtype}"
    assert test_images.dtype == np.uint8, f"Unexpected test_images dtype: {test_images.dtype}"
    assert test_labels.dtype == np.uint8, f"Unexpected test_labels dtype: {test_labels.dtype}"

    assert np.min(train_images) >= 0 and np.max(train_images) <= 255
    assert np.min(test_images) >= 0 and np.max(test_images) <= 255
    assert set(np.unique(train_labels)) == set(range(10)), f"Unexpected labels in train: {np.unique(train_labels)}"
    assert set(np.unique(test_labels)) == set(range(10)), f"Unexpected labels in test: {np.unique(test_labels)}"

    # Class distribution
    train_classes, train_counts = np.unique(train_labels, return_counts=True)
    test_classes, test_counts = np.unique(test_labels, return_counts=True)

    info = {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "train_counts": dict(zip(train_classes.tolist(), train_counts.tolist())),
        "test_counts": dict(zip(test_classes.tolist(), test_counts.tolist())),
        "train_shape": train_images.shape,
        "test_shape": test_images.shape,
        "pixel_min": int(np.min(train_images)),
        "pixel_max": int(np.max(train_images)),
    }
    return info


if __name__ == "__main__":
    mnist_info = load_and_validate_mnist()
    print("MNIST successfully loaded and validated:")
    print(f"Train images: {mnist_info['train_shape']}, min={mnist_info['pixel_min']}, max={mnist_info['pixel_max']}")
    print(f"Test images: {mnist_info['test_shape']}")
    print(f"Train class distribution: {mnist_info['train_counts']}")
    print(f"Test class distribution: {mnist_info['test_counts']}")
