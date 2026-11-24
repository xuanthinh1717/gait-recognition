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
    train=True : lay mot doan lien tuc T frame o vi tri ngau nhien (uu tien vung giua video
                 khi video dai de tranh cac frame dau/cuoi bi khuat nguoi).
    train=False: lay T frame lien tuc o chinh giua video (center clip) de dong bo
                 toc do chuyen dong (stride=1) voi luc train cho LSTM.
    ff_gei_m > 0: bien moi frame i thanh anh ff-GEI (trung binh 2*m + 1 frame quanh i
                  theo bai bao Wang & Yan 2019: F_i = 1/(2m+1) * sum_{t=i-m}^{i+m} I_t).
    Video ngan hon T: tai su dung frame theo vong (chi so modulo).
    """

    def __init__(self, csv_file, frames_path="data/npy/casiab_v2/frames.npy",
                 T=30, train=False, eval_sample="center", ff_gei_m=0, eval_clips=1,
                 return_uint8=False):
        assert eval_sample in ("center", "linspace")
        assert ff_gei_m >= 0
        assert eval_clips >= 1
        self.df = pd.read_csv(csv_file, dtype=CSV_DTYPES)
        self.starts = self.df["start"].to_numpy()
        self.lengths = self.df["length"].to_numpy()
        self.labels = self.df["label"].to_numpy()
        self.frames_path = frames_path
        self.T = T
        self.train = train
        self.eval_sample = eval_sample
        self.ff_gei_m = int(ff_gei_m)
        self.eval_clips = int(eval_clips)
        self.return_uint8 = bool(return_uint8)
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
                off = int(np.random.randint(0, length - T + 1))
                return slice(off, off + T)
            return np.arange(T) % length
        if length >= T:
            if self.eval_sample == "center":
                off = (length - T) // 2
                return slice(off, off + T)
            return np.linspace(0, length - 1, T).round().astype(int)
        return np.arange(T) % length

    def _extract_clip(self, frames, start, length, rel):
        if self.ff_gei_m > 0:
            rel_idx = np.arange(rel.start, rel.stop) if isinstance(rel, slice) else rel
            offsets = np.arange(-self.ff_gei_m, self.ff_gei_m + 1)
            win_idx = np.clip(rel_idx[:, None] + offsets[None, :], 0, length - 1)
            lo, hi = int(win_idx.min()), int(win_idx.max()) + 1
            block = np.asarray(frames[start + lo : start + hi], dtype=np.float32)
            seq = block[win_idx - lo].mean(axis=1)
            if self.return_uint8:
                return torch.from_numpy(seq.round().clip(0, 255).astype(np.uint8)).unsqueeze(1)
            return torch.from_numpy(seq / 255.0).unsqueeze(1)

        if isinstance(rel, slice):
            raw = np.array(frames[start + rel.start : start + rel.stop], copy=True)
        else:
            raw = np.asarray(frames[start + rel])
        if self.return_uint8:
            if raw.dtype != np.uint8:
                raw = raw.astype(np.uint8, copy=False)
            return torch.from_numpy(raw).unsqueeze(1)          # (T, 1, S, S) uint8
        seq = raw.astype(np.float32) * (1.0 / 255.0)
        return torch.from_numpy(seq).unsqueeze(1)              # (T, 1, S, S) float32

    def __getitem__(self, idx):
        start, length = int(self.starts[idx]), int(self.lengths[idx])
        frames = self._get_frames()
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        if not self.train and self.eval_clips > 1:
            if length > self.T:
                offs = np.linspace(0, length - self.T, self.eval_clips).round().astype(int)
                clips = [self._extract_clip(frames, start, length, slice(int(o), int(o) + self.T)) for o in offs]
            else:
                rel = np.arange(self.T) % length
                clip = self._extract_clip(frames, start, length, rel)
                clips = [clip] * self.eval_clips
            return torch.stack(clips, dim=0), label      # (K, T, 1, S, S)

        rel = self._pick(length)
        return self._extract_clip(frames, start, length, rel), label


class GEIDataset(Dataset):
    def __init__(self, csv_file, frames_path=None, return_uint8=False):
        import os
        self.df = pd.read_csv(csv_file, dtype=CSV_DTYPES)
        self.paths = self.df["gei_path"].tolist()
        self.labels = self.df["label"].to_numpy()
        self.npy_idx = self.df["npy_idx"].to_numpy() if "npy_idx" in self.df.columns else None
        self.frames_path = frames_path
        self.return_uint8 = bool(return_uint8)
        self.use_npy = (self.npy_idx is not None and frames_path is not None
                        and os.path.exists(frames_path))
        self._frames = None
        self._cache = {} if (not self.use_npy and len(self.df) <= 20000) else None

    def __len__(self):
        return len(self.df)

    def _get_frames(self):
        if self._frames is None:
            self._frames = np.load(self.frames_path, mmap_mode="r")
        return self._frames

    def __getitem__(self, idx):
        if self._cache is not None and idx in self._cache:
            x = self._cache[idx]
            if not self.return_uint8 and x.dtype == torch.uint8:
                x = x.float().mul_(1.0 / 255.0)
            return x, torch.tensor(int(self.labels[idx]), dtype=torch.long)
        if self.use_npy:
            img = np.array(self._get_frames()[int(self.npy_idx[idx])], copy=True)
        else:
            img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(self.paths[idx])
        if self.return_uint8 or self._cache is not None:
            if img.dtype != np.uint8:
                img = img.astype(np.uint8, copy=False)
            x_u8 = torch.from_numpy(img).unsqueeze(0)        # (1, S, S) uint8
            if self._cache is not None:
                self._cache[idx] = x_u8
            x = x_u8 if self.return_uint8 else x_u8.float().mul_(1.0 / 255.0)
        else:
            x = torch.from_numpy(np.asarray(img, dtype=np.float32) * (1.0 / 255.0)).unsqueeze(0)
        return x, torch.tensor(int(self.labels[idx]), dtype=torch.long)
