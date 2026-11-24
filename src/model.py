"""
model.py
CNN architecture for Handwritten Digit Recognition (MNIST).
Used identically for Supervised Baseline and Pseudo-Labeling Self-Training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25):
        super(DigitCNN, self).__init__()
        
        # Block 1: 1 -> 32 channels, 28x28 -> 14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: 32 -> 64 channels, 14x14 -> 7x7
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(dropout_rate)

        # Dense Classifier: 64 * 7 * 7 = 3136 -> 128 -> 10
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.drop_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Dense
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        logits = self.fc2(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes softmax probabilities for input tensor."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs


if __name__ == "__main__":
    net = DigitCNN()
    dummy = torch.randn(4, 1, 28, 28)
    out = net(dummy)
    probs = net.predict_proba(dummy)
    print("Model initialized successfully.")
    print("Output shape:", out.shape)
    print("Probabilities sum:", probs.sum(dim=1).numpy())
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
