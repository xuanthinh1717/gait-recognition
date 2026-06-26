import pandas as pd
import cv2
import torch

from torch.utils.data import Dataset

class GaitDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        path = row["path"]
        label = row["label"]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = img.astype("float32") / 255.0

        img = torch.tensor(img).unsqueeze(0)

        label = torch.tensor(label)

        return img, label