"""
openset_utils_v2.py

Ham dung chung cho train_v2.py va evaluate_openset_v2.py:
    build_model, make_dataset, extract_embeddings, rank1_openset
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.cnn_gei_v2 import GEIEmbedCNN
from src.models.cnn_lstm_v2 import CNNLSTM
from src.datasets.gait_dataset_v2 import GEIDataset, SeqDataset

CONDS = ("nm", "bg", "cl")
DEFAULT_DROPOUT = {"gei": 0.1, "lstm": 0.3, "mean": 0.3}
DEFAULT_FEAT_DIM = {"gei": 1024, "lstm": 256, "mean": 256}


def build_model(name, num_classes, dropout=None, pool="mean", feat_dim=None, lstm_layers=1):
    dropout = DEFAULT_DROPOUT[name] if dropout is None else dropout
    feat_dim = DEFAULT_FEAT_DIM[name] if feat_dim is None else feat_dim
    if name == "gei":
        return GEIEmbedCNN(num_classes=num_classes, feat_dim=feat_dim, dropout=dropout)
    return CNNLSTM(num_classes, temporal=name, feat_dim=feat_dim, dropout=dropout,
                   pool=pool, lstm_layers=lstm_layers)


def make_dataset(name, csv_file, frames_path, T, train, eval_sample="center",
                 ff_gei_m=0, eval_clips=1, return_uint8=False):
    if name == "gei":
        return GEIDataset(csv_file, frames_path=frames_path, return_uint8=return_uint8)
    return SeqDataset(csv_file, frames_path=frames_path, T=T, train=train,
                      eval_sample=eval_sample, ff_gei_m=ff_gei_m,
                      eval_clips=eval_clips, return_uint8=return_uint8)


def make_loader(dataset, batch_size, num_workers=0, pin_memory=False,
                shuffle=False, drop_last=False, batch_sampler=None, generator=None,
                worker_init_fn=None, persistent=False):
    """Tao DataLoader; bat persistent_workers khi tai su dung qua nhieu epoch."""
    kw = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0 and persistent:
        kw["persistent_workers"] = True
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler,
                          worker_init_fn=worker_init_fn, generator=generator, **kw)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      worker_init_fn=worker_init_fn, generator=generator, **kw)


def to_float_input(x, device):
    """Chuyen tensor (ke ca uint8 tu DataLoader) len GPU va chuan hoa [0, 1]."""
    x = x.to(device, non_blocking=True)
    return x.float().mul_(1.0 / 255.0) if x.dtype == torch.uint8 else x


@torch.no_grad()
def extract_embeddings(model, dataset, device, batch_size=32, num_workers=0,
                       loader=None, amp=False):
    """Tra ve (N, D) theo dung thu tu dong cua dataset.df."""
    net = model.module if hasattr(model, "module") else model
    net.eval()
    if loader is None:
        loader = make_loader(dataset, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=(device.type == "cuda"), shuffle=False)
    use_amp = amp and device.type == "cuda"
    out = []
    for x, _ in loader:
        with torch.amp.autocast("cuda", enabled=use_amp):
            if x.dim() == 6:  # Multi-clip averaging: (B, K, T, 1, S, S)
                B, K = x.shape[:2]
                xf = to_float_input(x.flatten(0, 1), device)
                emb = net.embed(xf).float()
                emb = F.normalize(emb, p=2, dim=1).view(B, K, -1).mean(dim=1).cpu()
            else:
                xf = to_float_input(x, device)
                emb = net.embed(xf).float().cpu()
        out.append(emb)
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
