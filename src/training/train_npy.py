import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from src.datasets.gait_dataset_npy import GaitDatasetNPY
from src.models.cnn_model_attention import CNN

# =====================================================
# RANDOM SEED
# =====================================================

SEED = 186

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# DATASET
# =====================================================

IMAGES = "data/npy/oumvlp_images.npy"
LABELS = "data/npy/oumvlp_labels.npy"

train_dataset = GaitDatasetNPY(IMAGES, LABELS,
    np.load("data/npy/idx_train.npy"))

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_dataset = GaitDatasetNPY(IMAGES, LABELS,
    np.load("data/npy/idx_val.npy"))

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# =====================================================
# MODEL
# =====================================================
# 3Conv baseline
# lr=0.001

# Attention + BatchNorm
# lr=0.0001
num_classes = 10307

model = CNN(
    num_classes=num_classes,
    dropout=0.1
).to(device)

model = torch.nn.DataParallel(model)

# =====================================================
# LOSS + OPTIMIZER
# =====================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# =====================================================
# TRAIN CONFIG
# =====================================================

epochs = 15

# =====================================================
# LOGGING
# =====================================================

os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

log_file = "logs/training_log.csv"
model_file = "outputs/model.pth"
best_val_accuracy = 0.0
save_start_epoch = epochs // 2

with open(log_file, mode="w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy"
    ])

# =====================================================
# TRAIN LOOP
# =====================================================

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    avg_loss = (
        running_loss /
        len(train_loader)
    )

    accuracy = (
        100 * correct / total
    )

    model.eval()

    val_running_loss = 0.0

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_avg_loss = (
        val_running_loss /
        len(val_loader)
    )

    val_accuracy = (
        100 * val_correct / val_total
    )

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {avg_loss:.4f} "
        f"Train Accuracy: {accuracy:.2f}% "
        f"Val Loss: {val_avg_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    if epoch >= save_start_epoch and val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            model_file
        )

        print(
            f"Model saved "
            f"(best Val Accuracy: {best_val_accuracy:.2f}%)"
        )

    with open(
        log_file,
        mode="a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            epoch + 1,
            avg_loss,
            accuracy,
            val_avg_loss,
            val_accuracy
        ])

# =====================================================
# SAVE MODEL
# =====================================================

print("Training finished")

print(
    f"Best model saved to {model_file} "
    f"with Val Accuracy: {best_val_accuracy:.2f}%"
)
