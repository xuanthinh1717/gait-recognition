import torch

from torch.utils.data import DataLoader

from src.datasets.gait_dataset import GaitDataset
from src.models.cnn_model_attention import CNN

# =====================================================
# DATASET
# =====================================================

test_dataset = GaitDataset(
    "data/splits/test.csv"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# =====================================================
# MODEL
# =====================================================

model = CNN(
    num_classes=10307,
    num_conv_blocks=2,
    dropout=0.1
)

model.load_state_dict(
    torch.load("outputs/model.pth")
)

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