"""
gait_dataset_v2.py

Hai dataset doc cac file csv cua create_split_openset_v2.py:

    SeqDataset : tra ve chuoi (T, 1, S, S) cho cau hinh B / E (CNN-LSTM, mean pooling)
    GEIDataset : tra ve anh GEI (1, S, S) cho cau hinh D (CNN+CBAM tren GEI)

Ca hai tra ve (x, label). Voi val/test label = -1 (khong dung).
Thu tu mau = thu tu dong trong csv (shuffle=False), nen evaluate co the
doi chieu truc tiep voi dataset.df de lay subject / cond / angle.
"""
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CSV_DTYPES = {"subj_dir": str, "angle_dir": str}


class SeqDataset(Dataset):
    """
    train=True : lay mot doan lien tuc T frame o vi tri ngau nhien.
    train=False: lay T frame cach deu nhau tren toan video (co dinh, lap lai duoc).
    Video ngan hon T: tai su dung frame theo vong (chi so modulo).
    """

    def __init__(self, csv_file, frames_path="data/npy/casiab_v2/frames.npy",
                 T=30, train=False):
        self.df = pd.read_csv(csv_file, dtype=CSV_DTYPES)
        self.starts = self.df["start"].to_numpy()
        self.lengths = self.df["length"].to_numpy()
        self.labels = self.df["label"].to_numpy()
        self.frames_path = frames_path
        self.T = T
        self.train = train
        self._frames = None  # mo mmap o moi worker, khong pickle qua process

    def __len__(self):
        return len(self.df)

    def _get_frames(self):
        if self._frames is None:
            self._frames = np.load(self.frames_path, mmap_mode="r")
        return self._frames

    def _pick(self, length):
        T = self.T
        if self.train:
            if length >= T:
                off = np.random.randint(0, length - T + 1)
                return np.arange(off, off + T)
            return np.arange(T) % length
        if length >= T:
            return np.linspace(0, length - 1, T).round().astype(int)
        return np.arange(T) % length

    def __getitem__(self, idx):
        start, length = int(self.starts[idx]), int(self.lengths[idx])
        rel = self._pick(length)
        seq = np.asarray(self._get_frames()[start + rel], dtype=np.float32) / 255.0
        x = torch.from_numpy(seq).unsqueeze(1)          # (T, 1, S, S)
        return x, torch.tensor(int(self.labels[idx]), dtype=torch.long)


class GEIDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, dtype=CSV_DTYPES)
        self.paths = self.df["gei_path"].tolist()
        self.labels = self.df["label"].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.paths[idx])
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)  # (1, S, S)
        return x, torch.tensor(int(self.labels[idx]), dtype=torch.long)
