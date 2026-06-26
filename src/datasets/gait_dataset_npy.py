import numpy as np
import torch

from torch.utils.data import Dataset


class GaitDatasetNPY(Dataset):

    def __init__(self, images_path, labels_path, indices):

        # mmap_mode='r': không load hết vào RAM
        self.images = np.load(images_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')
        self.indices = indices

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        i = self.indices[idx]

        img = self.images[i].astype("float32") / 255.0
        img = torch.tensor(img).unsqueeze(0)

        label = torch.tensor(int(self.labels[i]))

        return img, label