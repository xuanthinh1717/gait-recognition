"""
openset_utils_v2.py

Ham dung chung cho train_v2.py va evaluate_openset_v2.py:
    build_model, make_dataset, extract_embeddings, rank1_openset
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.cnn_lstm_v2 import CNNLSTM, GEIEmbedCNN
from src.datasets.gait_dataset_v2 import GEIDataset, SeqDataset

CONDS = ("nm", "bg", "cl")
DEFAULT_DROPOUT = {"gei": 0.1, "lstm": 0.3, "mean": 0.3}


def build_model(name, num_classes, dropout=None, pool="mean"):
    dropout = DEFAULT_DROPOUT[name] if dropout is None else dropout
    if name == "gei":
        return GEIEmbedCNN(num_classes=num_classes, num_conv_blocks=3, dropout=dropout)
    return CNNLSTM(num_classes, temporal=name, dropout=dropout, pool=pool)


def make_dataset(name, csv_file, frames_path, T, train):
    if name == "gei":
        return GEIDataset(csv_file)
    return SeqDataset(csv_file, frames_path=frames_path, T=T, train=train)


@torch.no_grad()
def extract_embeddings(model, dataset, device, batch_size=32, num_workers=0):
    """Tra ve (N, D) theo dung thu tu dong cua dataset.df."""
    net = model.module if hasattr(model, "module") else model
    net.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    out = []
    for x, _ in loader:
        out.append(net.embed(x.to(device)).float().cpu())
    return torch.cat(out).numpy()


def _l2norm(a):
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)


def rank1_openset(g_df, g_emb, p_df, p_emb):
    """
    Rank-1 chuan gallery-probe, khoang cach cosine.
    Voi moi (dieu kien probe, goc probe): gallery = moi mau gallery o goc KHAC goc probe;
    probe dung neu gallery gan nhat cung subject. Rank-1 cua mot dieu kien =
    trung binh qua cac goc probe.

    Tra ve (per_cond, per_angle):
        per_cond  = {"nm":..., "bg":..., "cl":..., "mean":...}   (don vi %)
        per_angle = {cond: {angle: acc}}
    """
    g, p = _l2norm(g_emb), _l2norm(p_emb)
    g_sub, g_ang = g_df["subject"].to_numpy(), g_df["angle"].to_numpy()
    p_sub, p_ang = p_df["subject"].to_numpy(), p_df["angle"].to_numpy()
    p_cond = p_df["cond"].to_numpy()

    per_angle = {}
    for cond in CONDS:
        cm = p_cond == cond
        if not cm.any():
            continue
        res = {}
        for ang in sorted(set(p_ang[cm])):
            idx = np.where(cm & (p_ang == ang))[0]
            if not (g_ang != ang).any():            # khong con gallery nao khac goc
                continue
            sim = p[idx] @ g.T
            sim[:, g_ang == ang] = -np.inf          # bo cung goc
            nn = sim.argmax(axis=1)
            res[int(ang)] = float((g_sub[nn] == p_sub[idx]).mean() * 100)
        per_angle[cond] = res

    per_cond = {c: float(np.mean(list(r.values()))) for c, r in per_angle.items()}
    per_cond["mean"] = float(np.mean([per_cond[c] for c in per_angle]))
    return per_cond, per_angle
