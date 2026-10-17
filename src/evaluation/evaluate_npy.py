import torch
import numpy as np

from torch.utils.data import DataLoader

from src.datasets.gait_dataset_npy import GaitDatasetNPY
from src.models.cnn_model_attention import CNN

# =====================================================
# DATASET
# =====================================================

IMAGES = "data/npy/oumvlp_images.npy"
LABELS = "data/npy/oumvlp_labels.npy"

test_dataset = GaitDatasetNPY(
    IMAGES,
    LABELS,
    np.load("data/npy/idx_test.npy")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# =====================================================
# MODEL
# =====================================================

model = CNN(num_classes=10307, num_conv_blocks=3, dropout=0.3)

state_dict = torch.load("outputs/model.pth", map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.eval()

# =====================================================
# EVALUATION
# =====================================================

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

# =====================================================
# RESULT
# =====================================================

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")