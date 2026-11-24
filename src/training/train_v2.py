"""
train_v2.py

Train bang softmax tren cac nguoi train; CHON EPOCH bang Rank-1 open-set
(embedding + cosine) tren cac nguoi validation giu rieng. Khong dung nguoi test.

Vi du:
    python -m src.training.train_v2 --model gei  --seed 1
    python -m src.training.train_v2 --model lstm --seed 1
    python -m src.training.train_v2 --model lstm --seed 1 --tag v2 \
        --clip_grad 1.0 --weight_decay 1e-4 --dropout 0.3 --incremental_loss

Tuy chon moi (MAC DINH TAT, nen ket qua cu van tai tao duoc):
    --tag               them hau to vao ten run, de cac bien the khong de len nhau
    --clip_grad X       cat chuan gradient (0 = tat)
    --weight_decay X    >0 thi dung AdamW (0 = Adam nhu cu)
    --warmup_epochs N   tang lr tuyen tinh trong N epoch dau (tinh theo tung iteration)
    --incremental_loss  (chi lstm) loss o MOI buoc thoi gian, trong so t^2 (kieu GaitNet)
    --emb_pool last     (chi lstm) embedding lay o buoc cuoi thay vi trung binh
    --triplet_weight X  trong so Batch-Hard Cosine Triplet Loss (>0 se bat PKSampler)
    --triplet_margin M  bien do margin cho Triplet Loss (mac dinh 0.2)
    --pk_k K            so video moi nguoi trong 1 batch khi dung Triplet Loss (mac dinh 4)
    --label_smoothing S label smoothing cho CrossEntropyLoss (mac dinh 0.0)
    --decay_acc A       khi train_acc >= A (%) thi bat dau giam lr theo cosine ve min_lr (0 = tat)
    --min_lr L          lr toi thieu khi giam lr (mac dinh 1e-5)
    --ff_gei_m M        ban kinh cua so truot ff-GEI (2*M+1 frame quanh moi buoc, 0 = frame thuong)
    --lstm_layers N     so tang LSTM chong len nhau (mac dinh 1; bai bao Conv-LSTM dung 3)
    --optimizer OPT     bo toi uu: adam (mac dinh), sam, hoac asam
    --metric_loss LOSS  ham metric loss khi triplet_weight > 0: triplet (mac dinh) hoac supcon
    --arcface           dung ArcFace / Sub-Center ArcFace thay cho Linear head thuong
    --arcface_s S       ban kinh mat cau s cua ArcFace (mac dinh 30.0)
    --arcface_m M       bien do goc margin m (radian) cua ArcFace (mac dinh 0.35)
    --arcface_k K       so tam con moi lop cho Sub-Center ArcFace (mac dinh 1 = ArcFace chuan)

Dau ra (khong de len ban cu):
    outputs_v2/casiab_<model>[_<tag>]_seed<N>.pth
    logs_v2/casiab_<model>[_<tag>]_seed<N>.csv
"""
import argparse
from collections import defaultdict
import csv
import json
import math
import os
import random
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Sampler

from src.evaluation.openset_utils_v2 import (build_model, extract_embeddings, make_dataset,
                                              make_loader, rank1_openset, to_float_input)
from src.training.sam import SAM, disable_running_stats, enable_running_stats
from src.training.sophia import SophiaG


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


class PKSampler(Sampler):
    """
    Moi batch gom P nguoi x K video/nguoi (batch_size = P * K).
    Dam bao trong batch luon co cac cap cung nguoi (positive) va khac nguoi (negative)
    de tinh Triplet Loss.
    """

    def __init__(self, labels, batch_size, k=4, seed=1):
        assert batch_size % k == 0, f"batch_size ({batch_size}) phai chia het cho pk_k ({k})"
        self.p = batch_size // k
        self.k = k
        self.seed = seed
        self.epoch = 0
        self.by_label = defaultdict(list)
        for idx, y in enumerate(labels):
            self.by_label[int(y)].append(idx)
        assert len(self.by_label) >= self.p, "So lop train it hon P nguoi moi batch"
        self.num_batches = len(labels) // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1009)
        self.epoch += 1
        pools = {}
        for y, idxs in self.by_label.items():
            buf = list(idxs)
            rng.shuffle(buf)
            pools[y] = buf

        classes = list(self.by_label.keys())
        for _ in range(self.num_batches):
            chosen = rng.sample(classes, self.p)
            batch = []
            for y in chosen:
                while len(pools[y]) < self.k:
                    buf = list(self.by_label[y])
                    rng.shuffle(buf)
                    pools[y].extend(buf)
                batch.extend(pools[y][:self.k])
                del pools[y][:self.k]
            yield batch


def batch_hard_triplet_loss(emb, y, margin=0.2):
    """
    Batch-Hard Triplet Loss tren khoang cach Cosine: d(u, v) = 1 - cos(u, v).
    Voi moi anchor i:
        hard_pos = max_{j: y_j == y_i} d(i, j)
        hard_neg = min_{k: y_k != y_i} d(i, k)
        loss_i   = relu(hard_pos - hard_neg + margin)
    """
    with torch.amp.autocast("cuda", enabled=False):
        z = F.normalize(emb.float(), p=2, dim=1)
        dist = (1.0 - z @ z.T).clamp(min=0.0)
        eye = torch.eye(y.size(0), dtype=torch.bool, device=y.device)
        same = y.unsqueeze(0) == y.unsqueeze(1)
        pos_mask = same & ~eye
        neg_mask = ~same

        if not pos_mask.any(dim=1).all() or not neg_mask.any(dim=1).all():
            return z.new_tensor(0.0)

        # Hardest positive: gan -inf cho cac vi tri khong phai positive
        hard_pos = dist.masked_fill(~pos_mask, -1e9).max(dim=1).values

        # Hardest negative: gan +inf cho cac vi tri cung lop
        hard_neg = dist.masked_fill(~neg_mask, 1e9).min(dim=1).values

        return F.relu(hard_pos - hard_neg + margin).mean()


def supcon_loss(emb, y, temperature=0.1):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020) tren Cosine similarity.
    Keo TAT CA cac mau cung nguoi (positive) trong batch lai gan nhau,
    dong thoi day tat ca cac mau khac nguoi (negative) ra xa.
    """
    with torch.amp.autocast("cuda", enabled=False):
        z = F.normalize(emb.float(), p=2, dim=1)
        sim = (z @ z.T) / temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        eye = torch.eye(y.size(0), dtype=torch.bool, device=y.device)
        same = y.unsqueeze(0) == y.unsqueeze(1)
        pos_mask = same & ~eye

        if not pos_mask.any(dim=1).all():
            return z.new_tensor(0.0)

        exp_sim = torch.exp(sim).masked_fill(eye, 0.0)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        mean_log_prob_pos = log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_mask.sum(dim=1)
        return -mean_log_prob_pos.mean()


def arcface_logits(emb, weight, y, s=30.0, m=0.35, k=1, return_clean=False):
    """
    ArcFace / Sub-Center ArcFace (Deng et al., CVPR 2019 / 2020).
    Chuan hoa L2 ca embedding va trong so lop W, sau do cong bien do goc m (radian)
    vao goc cua lop nhan dung y: cos(theta_y + m), roi nhan voi ban kinh mat cau s.
    Neu k > 1 (Sub-Center ArcFace): moi lop co k tam con, lay tam con co cosine lon nhat.
    Neu return_clean=True: tra ve (logits_co_margin, clean_cosine_logits) de tinh train_acc
    trung thuc tren mat cau ma khong bi bien do m lamlech nguong decay_acc.
    """
    with torch.amp.autocast("cuda", enabled=False):
        x_norm = F.normalize(emb.float(), p=2, dim=1)
        w_norm = F.normalize(weight.float(), p=2, dim=1)
        cosine = F.linear(x_norm, w_norm)
        if k > 1:
            cosine = cosine.view(emb.size(0), -1, k).max(dim=2).values
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        clean_logits = cosine * s

        if m == 0.0:
            return (clean_logits, clean_logits) if return_clean else clean_logits

        cos_m = math.cos(m)
        sin_m = math.sin(m)
        th = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m

        sine = torch.sqrt((1.0 - cosine * cosine).clamp(min=1e-12))
        phi = cosine * cos_m - sine * sin_m
        phi = torch.where(cosine > th, phi, cosine - mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1.0)
        logits = ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * s
        return (logits, clean_logits) if return_clean else logits


class ModelEMA:
    """
    Exponential Moving Average (EMA) cua trong so mo hinh.
    Lam muot quy dao cap nhat o day cuc tieu (dac biet hieu qua voi Sophia va SAM).
    """

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model):
        if self.backup:
            model.load_state_dict(self.backup, strict=True)
            self.backup = {}


def incremental_loss(model, x, y, label_smoothing=0.0,
                     arcface=False, arcface_s=30.0, arcface_m=0.35, arcface_k=1):
    """
    Loss tai moi buoc thoi gian t=1..T voi trong so w_t = t^2 (chuan hoa tong = 1).
    Tra ve (loss, logits cua buoc cuoi de tinh accuracy, emb).
    """
    if arcface:
        f = model.step_features(x)
        B, T, D = f.shape
        f_bn = F.batch_norm(
            f.reshape(B * T, D), None, None, model.bn.weight, model.bn.bias,
            training=True, eps=model.bn.eps
        )
        y_rep = y.repeat_interleave(T)
        steps_flat, clean_flat = arcface_logits(
            f_bn, model.head.weight, y_rep,
            s=arcface_s, m=arcface_m, k=arcface_k, return_clean=True
        )
        C = steps_flat.size(1)
        steps = steps_flat.view(B, T, C)
        clean_steps = clean_flat.view(B, T, C)
        pooled = f[:, -1] if model.pool == "last" else f.mean(dim=1)
        emb = model.bn(pooled)
    else:
        steps, emb = model.forward_steps(x, return_embed=True)  # (B, T, C), (B, D)
        clean_steps = steps
        B, T, C = steps.shape
        y_rep = y.repeat_interleave(T)
        steps_flat = steps.reshape(B * T, C)

    w = torch.arange(1, T + 1, device=x.device, dtype=torch.float32) ** 2
    w = w / w.sum()
    ce = F.cross_entropy(steps_flat.float(), y_rep,
                         reduction="none", label_smoothing=label_smoothing).view(B, T)
    return (ce * w).sum(dim=1).mean(), clean_steps[:, -1], emb


def compute_batch_loss(model, x, y, criterion, args, active_arcface=None, cur_m=None):
    use_arcface = args.arcface if active_arcface is None else active_arcface
    m_val = (args.arcface_m if cur_m is None else cur_m) if use_arcface else 0.0
    if args.incremental_loss:
        loss, out, emb = incremental_loss(
            model, x, y, args.label_smoothing,
            arcface=use_arcface, arcface_s=args.arcface_s,
            arcface_m=m_val, arcface_k=args.arcface_k,
        )
    elif args.arcface and (args.switch_arcface_acc > 0 or args.arcface_warmup > 0):
        # Smooth Sub-Center Curriculum: Giu nhánh Linear Sub-Center (co Dropout + tu do norm)
        # va pha tron muot voi nhánh Angular Margin ArcFace sau khi switch de khong bi soc hinh hoc hay overfit.
        out_lin, emb = model(x, return_embed=True)
        if args.arcface_k > 1:
            out_lin = out_lin.view(out_lin.size(0), -1, args.arcface_k).max(dim=2).values
        loss_lin = criterion(out_lin.float(), y)
        if use_arcface and m_val > 0.0:
            logits_arc = arcface_logits(
                emb, model.head.weight, y,
                s=args.arcface_s, m=m_val, k=args.arcface_k, return_clean=False
            )
            loss_arc = criterion(logits_arc, y)
            prog = min(1.0, m_val / max(args.arcface_m, 1e-6))
            w_arc = 0.15 * prog
            loss = (1.0 - w_arc) * loss_lin + w_arc * loss_arc
        else:
            loss = loss_lin
        out = out_lin
    elif use_arcface:
        emb = model.embed(x)
        logits, clean_out = arcface_logits(
            emb, model.head.weight, y,
            s=args.arcface_s, m=m_val, k=args.arcface_k, return_clean=True
        )
        loss = criterion(logits, y)
        out = clean_out
    else:
        out, emb = model(x, return_embed=True)
        if args.arcface and args.arcface_k > 1:
            out = out.view(out.size(0), -1, args.arcface_k).max(dim=2).values
        loss = criterion(out.float(), y)
    if args.triplet_weight > 0:
        if args.metric_loss == "supcon":
            m_loss = supcon_loss(emb, y)
        else:
            m_loss = batch_hard_triplet_loss(emb, y, margin=args.triplet_margin)
        loss = loss + args.triplet_weight * m_loss
    return loss, out, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gei", "lstm", "mean"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=None, help="mac dinh: gei=128, lstm/mean=16")
    ap.add_argument("--dropout", type=float, default=None, help="mac dinh: gei=0.1, lstm/mean=0.3")
    ap.add_argument("--feat_dim", type=int, default=None,
                    help="so chieu embedding (mac dinh: gei=1024, lstm/mean=256; OUMVLP gei=2048)")
    ap.add_argument("--T", type=int, default=30, help="so frame moi chuoi (lstm/mean)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--splits", default="data/splits_v2")
    ap.add_argument("--frames", default="data/npy/casiab_v2/frames.npy")
    ap.add_argument("--out_dir", default="outputs_v2")
    ap.add_argument("--log_dir", default="logs_v2")
    ap.add_argument("--overwrite", action="store_true")
    # ---- tuy chon moi (mac dinh tat) ----
    ap.add_argument("--tag", default="", help="hau to ten run (chu va so), vd v1, v2")
    ap.add_argument("--clip_grad", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_epochs", type=float, default=0.0)
    ap.add_argument("--incremental_loss", action="store_true")
    ap.add_argument("--emb_pool", default="mean", choices=["mean", "last"])
    ap.add_argument("--eval_sample", default="center", choices=["center", "linspace"],
                    help="cach lay T frame khi val/test (center: T frame lien tuc o giua video)")
    ap.add_argument("--triplet_weight", type=float, default=0.0,
                    help="trong so Batch-Hard Cosine Triplet Loss (>0 se bat PKSampler)")
    ap.add_argument("--triplet_margin", type=float, default=0.2)
    ap.add_argument("--pk_k", type=int, default=4,
                    help="so mau moi nguoi trong 1 batch khi triplet_weight > 0")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--decay_acc", type=float, default=0.0,
                    help="nguong train_acc (%%) bat dau giam lr theo cosine ve min_lr (vd 99.0; 0=tat)")
    ap.add_argument("--min_lr", type=float, default=1e-5,
                    help="lr toi thieu khi giam lr (mac dinh 1e-5)")
    ap.add_argument("--ff_gei_m", type=int, default=0,
                    help="ban kinh cua so truot ff-GEI (2*m+1 frame quanh moi buoc, 0=tat)")
    ap.add_argument("--lstm_layers", type=int, default=1,
                    help="so tang LSTM chong len nhau (mac dinh 1)")
    ap.add_argument("--optimizer", default="adam",
                    choices=["adam", "sam", "asam", "sophia", "samsophia", "asamsophia"],
                    help="bo toi uu: adam (mac dinh), sam, asam, sophia, samsophia, hoac asamsophia")
    ap.add_argument("--rho", type=float, default=None,
                    help="ban kinh rho cho SAM/ASAM/Sophia (mac dinh: sam=0.05, asam=0.5, sophia=0.04, samsophia=0.02)")
    ap.add_argument("--sam_rho", type=float, default=None,
                    help="ban kinh nhieu SAM khi dung samsophia/asamsophia (mac dinh: samsophia=0.05, asamsophia=0.5)")
    ap.add_argument("--rho_schedule", default="none", choices=["none", "cos", "damped"],
                    help="lich thay doi ban kinh rho cua SAM/ASAM: none (co dinh), cos (+a -> -a), damped (a*cos(pi*t/T)*(1-t/T))")
    ap.add_argument("--metric_loss", default="triplet", choices=["triplet", "supcon"],
                    help="ham metric loss khi triplet_weight > 0: triplet (mac dinh) hoac supcon")
    ap.add_argument("--arcface", action="store_true",
                    help="dung ArcFace / Sub-Center ArcFace thay cho Linear head thuong")
    ap.add_argument("--arcface_s", type=float, default=30.0,
                    help="he so scale s (ban kinh mat cau) cua ArcFace (mac dinh 30.0)")
    ap.add_argument("--arcface_m", type=float, default=0.35,
                    help="bien do goc margin m (radian) cua ArcFace (mac dinh 0.35)")
    ap.add_argument("--arcface_k", type=int, default=1,
                    help="so tam con moi lop cho Sub-Center ArcFace (mac dinh 1 = ArcFace chuan)")
    ap.add_argument("--switch_arcface_acc", type=float, default=0.0,
                    help="nguong train_acc (%%) bat dau chuyen sang ArcFace (0 = tat, dung arcface tu dau neu co --arcface)")
    ap.add_argument("--arcface_warmup", type=float, default=0.0,
                    help="so epoch tang dan bien do goc m tu 0 -> arcface_m (0 = bat ngay m)")
    ap.add_argument("--sophia_k", type=int, default=10,
                    help="chu ky cap nhat Hessian cua SophiaG (mac dinh 10 buoc)")
    ap.add_argument("--sophia_ef", type=float, default=0.0,
                    help="ty trong pha tron do cong Empirical Fisher toan phan (CE+Triplet) vao Sophia [0..1]")
    ap.add_argument("--ema_decay", type=float, default=0.0,
                    help="he so lam muot trong so Model EMA (vd 0.995; 0 = tat)")
    ap.add_argument("--min_save_epoch", type=int, default=0,
                    help="chi bat dau xet luu checkpoint tot nhat tu epoch nay tro di (tranh luu nham dinh nhieu som)")
    ap.add_argument("--val_every", type=int, default=1,
                    help="chu ky validate (epoch) khi epoch < min_save_epoch; tu min_save_epoch luon validate moi epoch")
    ap.add_argument("--amp", action="store_true",
                    help="bat mixed precision FP16 (torch.amp.autocast) khi train tren CUDA")
    args = ap.parse_args()

    if args.tag and not re.fullmatch(r"[A-Za-z0-9]+", args.tag):
        sys.exit("--tag chi gom chu va so (khong dau gach).")
    if args.model != "lstm" and (args.incremental_loss or args.emb_pool != "mean"
                                 or args.lstm_layers != 1):
        sys.exit("--incremental_loss, --emb_pool last, --lstm_layers chi dung voi --model lstm.")
    if args.arcface_k < 1:
        sys.exit("--arcface_k phai >= 1.")
    if args.val_every < 1:
        sys.exit("--val_every phai >= 1.")

    if args.batch_size is None:
        args.batch_size = 128 if args.model == "gei" else 16
    if args.rho is None and args.optimizer in ("sam", "asam", "sophia", "samsophia", "asamsophia"):
        args.rho = {
            "sam": 0.05, "asam": 0.5, "sophia": 0.04,
            "samsophia": 0.02, "asamsophia": 0.02,
        }[args.optimizer]

    ds_prefix = "oumvlp" if "oumvlp" in args.splits.lower() else "casiab"
    tag = f"{ds_prefix}_{args.model}" + (f"_{args.tag}" if args.tag else "") + f"_seed{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{tag}.pth")
    log_path = os.path.join(args.log_dir, f"{tag}.csv")
    if (os.path.exists(ckpt_path) or os.path.exists(log_path)) and not args.overwrite:
        sys.exit(f"{tag} da ton tai. Dung --overwrite neu muon ghi de.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print("Device:", device, "| run:", tag, "| amp:", use_amp)
    print("Cau hinh:", {k: v for k, v in vars(args).items()
                        if k in ("lr", "dropout", "feat_dim", "batch_size", "T", "epochs",
                                 "clip_grad", "weight_decay", "warmup_epochs", "incremental_loss",
                                 "emb_pool", "eval_sample", "triplet_weight",
                                 "triplet_margin", "pk_k", "label_smoothing",
                                 "decay_acc", "min_lr", "ff_gei_m", "lstm_layers",
                                 "optimizer", "rho", "sam_rho", "rho_schedule", "metric_loss",
                                 "arcface", "arcface_s", "arcface_m", "arcface_k",
                                 "switch_arcface_acc", "arcface_warmup",
                                 "sophia_k", "sophia_ef", "ema_decay", "val_every", "amp")})

    sp = args.splits
    num_classes = json.load(open(os.path.join(sp, "split_info.json")))["num_train_classes"]
    print("num_classes:", num_classes)

    train_ds = make_dataset(args.model, f"{sp}/train.csv", args.frames, args.T, train=True,
                            eval_sample=args.eval_sample, ff_gei_m=args.ff_gei_m, return_uint8=True)
    val_g = make_dataset(args.model, f"{sp}/val_gallery.csv", args.frames, args.T, train=False,
                         eval_sample=args.eval_sample, ff_gei_m=args.ff_gei_m, return_uint8=True)
    val_p = make_dataset(args.model, f"{sp}/val_probe.csv", args.frames, args.T, train=False,
                         eval_sample=args.eval_sample, ff_gei_m=args.ff_gei_m, return_uint8=True)

    gen = torch.Generator().manual_seed(args.seed)
    persist = args.num_workers > 0
    pin_mem = device.type == "cuda"
    if args.triplet_weight > 0:
        pk_sampler = PKSampler(train_ds.labels, batch_size=args.batch_size,
                               k=args.pk_k, seed=args.seed)
        train_loader = make_loader(
            train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=pin_mem, batch_sampler=pk_sampler,
            worker_init_fn=seed_worker, generator=gen, persistent=persist,
        )
    else:
        train_loader = make_loader(
            train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=pin_mem, shuffle=True, drop_last=True,
            worker_init_fn=seed_worker, generator=gen, persistent=persist,
        )
    val_bs = 64 if args.model == "gei" else 16
    val_g_loader = make_loader(val_g, batch_size=val_bs, num_workers=args.num_workers,
                               pin_memory=pin_mem, shuffle=False, persistent=persist)
    val_p_loader = make_loader(val_p, batch_size=val_bs, num_workers=args.num_workers,
                               pin_memory=pin_mem, shuffle=False, persistent=persist)

    head_classes = num_classes * args.arcface_k if args.arcface else num_classes
    model = build_model(args.model, head_classes, args.dropout, args.emb_pool,
                        feat_dim=args.feat_dim, lstm_layers=args.lstm_layers).to(device)
    if args.arcface and args.switch_arcface_acc == 0.0 and args.arcface_warmup == 0.0 and model.head.bias is not None:
        model.head.bias.requires_grad_(False)
    print("Tham so:", sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    base_opt_cls = optim.AdamW if args.weight_decay > 0 else optim.Adam
    opt_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay} if args.weight_decay > 0 else {"lr": args.lr}
    if args.optimizer == "sam":
        optimizer = SAM(model.parameters(), base_opt_cls, rho=args.rho, adaptive=False, **opt_kwargs)
    elif args.optimizer == "asam":
        optimizer = SAM(model.parameters(), base_opt_cls, rho=args.rho, adaptive=True, **opt_kwargs)
    elif args.optimizer == "sophia":
        optimizer = SophiaG(model.parameters(), lr=args.lr, rho=args.rho,
                            weight_decay=args.weight_decay, ef_weight=args.sophia_ef)
    elif args.optimizer in ("samsophia", "asamsophia"):
        is_adaptive = (args.optimizer == "asamsophia")
        sam_rho = args.sam_rho if args.sam_rho is not None else (0.5 if is_adaptive else 0.05)
        ef_w = args.sophia_ef if args.sophia_ef > 0.0 else 1.0
        optimizer = SAM(
            model.parameters(), SophiaG, rho=sam_rho, adaptive=is_adaptive,
            base_rho=args.rho, lr=args.lr, betas=(0.9, 0.999),
            weight_decay=args.weight_decay, ef_weight=ef_w,
        )
    else:
        optimizer = base_opt_cls(model.parameters(), **opt_kwargs)

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0.0 else None

    scheduler = None
    warm_iters = int(args.warmup_epochs * len(train_loader))
    if warm_iters > 0:
        sched_opt = (optimizer.base_optimizer
                     if args.optimizer in ("sam", "asam", "samsophia", "asamsophia")
                     else optimizer)
        scheduler = optim.lr_scheduler.LambdaLR(
            sched_opt, lambda it: min(1.0, (it + 1) / warm_iters))

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_nm", "val_bg", "val_cl", "val_mean"])

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = -1.0
    best_epoch = 0
    decay_start_epoch = None
    global_step = 0
    active_arcface = args.arcface and (args.switch_arcface_acc == 0.0)
    arcface_start_epoch = 1 if active_arcface else None
    base_sam_rho = (optimizer.param_groups[0].get("sam_rho", args.rho)
                    if args.optimizer in ("sam", "asam", "samsophia", "asamsophia") else None)
    for epoch in range(1, args.epochs + 1):
        if decay_start_epoch is not None:
            scheduler = None
            rem = max(1, args.epochs - decay_start_epoch)
            prog = min(1.0, (epoch - decay_start_epoch) / rem)
            cur_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * prog))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

        cur_sam_rho = base_sam_rho
        if base_sam_rho is not None and args.rho_schedule != "none":
            if args.rho_schedule == "cos":
                u = (epoch - 1) / max(1, args.epochs - 1)
                cur_sam_rho = base_sam_rho * math.cos(math.pi * u)
            elif args.rho_schedule == "damped":
                u = (epoch - 1) / max(1, args.epochs)
                cur_sam_rho = base_sam_rho * math.cos(math.pi * u) * (1.0 - u)
            for pg in optimizer.param_groups:
                pg["sam_rho"] = cur_sam_rho

        if active_arcface:
            if args.arcface_warmup > 0 and arcface_start_epoch is not None:
                m_prog = min(1.0, (epoch - arcface_start_epoch + 1) / args.arcface_warmup)
                cur_m = args.arcface_m * m_prog
            else:
                cur_m = args.arcface_m
        else:
            cur_m = 0.0

        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x = to_float_input(x, device)
            y = y.to(device, non_blocking=True)
            if args.optimizer in ("sam", "asam", "samsophia", "asamsophia"):
                enable_running_stats(model)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, out, emb = compute_batch_loss(model, x, y, criterion, args,
                                                        active_arcface=active_arcface, cur_m=cur_m)
                scaler.scale(loss).backward()
                if use_amp:
                    scaler.unscale_(optimizer)
                    if not torch.isfinite(optimizer._grad_norm()):
                        optimizer.zero_grad()
                        scaler.update()
                        continue
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.first_step(zero_grad=True)
                if use_amp:
                    scaler.update()

                disable_running_stats(model)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss2, _, _ = compute_batch_loss(model, x, y, criterion, args,
                                                     active_arcface=active_arcface, cur_m=cur_m)
                scaler.scale(loss2).backward()
                if use_amp:
                    scaler.unscale_(optimizer)
                    if not torch.isfinite(optimizer._grad_norm()):
                        for pg in optimizer.param_groups:
                            for p in pg["params"]:
                                if "old_p" in optimizer.state[p]:
                                    p.data.copy_(optimizer.state[p]["old_p"])
                        optimizer.zero_grad()
                        scaler.update()
                        continue
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                if args.optimizer in ("samsophia", "asamsophia"):
                    optimizer.second_step(zero_grad=True, bs=x.size(0))
                else:
                    optimizer.second_step(zero_grad=True)
                if use_amp:
                    scaler.update()
            elif args.optimizer == "sophia":
                enable_running_stats(model)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, out, emb = compute_batch_loss(model, x, y, criterion, args,
                                                        active_arcface=active_arcface, cur_m=cur_m)
                scaler.scale(loss).backward()
                if use_amp:
                    scaler.unscale_(optimizer)
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer, bs=x.size(0))
                scaler.update()
                optimizer.zero_grad()

                if global_step % args.sophia_k == 0:
                    disable_running_stats(model)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        if args.arcface:
                            emb_h = model.embed(x)
                            out_h = arcface_logits(emb_h, model.head.weight, y,
                                                   s=args.arcface_s, m=0.0, k=args.arcface_k)
                        else:
                            out_h = model(x)
                        out_h_f = out_h.float()
                        with torch.no_grad():
                            y_hat = torch.distributions.Categorical(logits=out_h_f.detach()).sample()
                        loss_h = F.cross_entropy(out_h_f, y_hat)
                    loss_h.backward()
                    optimizer.update_hessian()
                    optimizer.zero_grad()
                    enable_running_stats(model)
                global_step += 1
            else:
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, out, emb = compute_batch_loss(model, x, y, criterion, args,
                                                        active_arcface=active_arcface, cur_m=cur_m)
                scaler.scale(loss).backward()
                if use_amp:
                    scaler.unscale_(optimizer)
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()

            if ema is not None:
                ema.update(model)
            if scheduler is not None:
                scheduler.step()
            run_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        tr_loss = run_loss / max(1, len(train_loader))
        tr_acc = 100 * correct / max(1, total)

        if args.arcface and args.switch_arcface_acc > 0 and not active_arcface and tr_acc >= args.switch_arcface_acc:
            active_arcface = True
            arcface_start_epoch = epoch + 1
            print(f"  -> train_acc {tr_acc:.2f}% >= {args.switch_arcface_acc:.1f}%, "
                  f"BAT ArcFace (m={args.arcface_m}, s={args.arcface_s}, warmup={args.arcface_warmup}) tu epoch {epoch + 1}")

        cur_lr = optimizer.param_groups[0]["lr"]
        m_str = f" m={cur_m:.3f}" if args.arcface else ""
        rho_str = f" rho={cur_sam_rho:+.4f}" if (cur_sam_rho is not None and args.rho_schedule != "none") else ""
        do_val = (epoch >= args.min_save_epoch) or (epoch % args.val_every == 0) or (epoch == args.epochs)

        if do_val:
            # --- validation open-set: embedding + cosine tren nguoi chua tung thay ---
            if ema is not None:
                ema.apply_shadow(model)
            g_emb = extract_embeddings(model, val_g, device, val_bs, args.num_workers,
                                       loader=val_g_loader, amp=use_amp)
            p_emb = extract_embeddings(model, val_p, device, val_bs, args.num_workers,
                                       loader=val_p_loader, amp=use_amp)
            res, _ = rank1_openset(val_g.df, g_emb, val_p.df, p_emb)

            r_nm = res.get("nm", float("nan"))
            r_bg = res.get("bg", float("nan"))
            r_cl = res.get("cl", float("nan"))
            r_mean = res["mean"]
            cond_str = (f"NM {r_nm:.1f} BG {r_bg:.1f} CL {r_cl:.1f}"
                        if "bg" in res or "cl" in res else f"NM {r_nm:.2f}")
            print(f"Epoch [{epoch}/{args.epochs}] lr {cur_lr:.2e}{rho_str}{m_str} loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
                  f"val Rank-1 {cond_str} mean {r_mean:.2f}")

            can_save = (epoch >= args.min_save_epoch) or (epoch == args.epochs and best_epoch == 0)
            if can_save and r_mean > best:
                best, best_epoch = r_mean, epoch
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "num_classes": num_classes, "epoch": epoch,
                            "val_mean": best}, ckpt_path)
                print(f"  -> saved (best val mean Rank-1 {best:.2f})")

            if ema is not None:
                ema.restore(model)
        else:
            r_nm = r_bg = r_cl = r_mean = float("nan")
            print(f"Epoch [{epoch}/{args.epochs}] lr {cur_lr:.2e}{rho_str}{m_str} loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
                  f"val (bo qua, val_every={args.val_every})")

        if args.decay_acc > 0 and decay_start_epoch is None and tr_acc >= args.decay_acc:
            decay_start_epoch = epoch
            print(f"  -> train_acc {tr_acc:.2f}% >= {args.decay_acc:.1f}%, "
                  f"bat dau giam lr theo cosine tu epoch {epoch + 1} ve {args.min_lr:.1e}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, tr_loss, tr_acc, r_nm, r_bg, r_cl, r_mean])

    print(f"Xong. Best epoch {best_epoch}, val mean Rank-1 {best:.2f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
