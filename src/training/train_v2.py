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

Dau ra (khong de len ban cu):
    outputs_v2/casiab_<model>[_<tag>]_seed<N>.pth
    logs_v2/casiab_<model>[_<tag>]_seed<N>.csv
"""
import argparse
import csv
import json
import os
import random
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.evaluation.openset_utils_v2 import (build_model, extract_embeddings, make_dataset,
                                              rank1_openset)


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


def incremental_loss(model, x, y):
    """
    Loss tai moi buoc thoi gian t=1..T voi trong so w_t = t^2 (chuan hoa tong = 1).
    Tra ve (loss, logits cua buoc cuoi de tinh accuracy).
    """
    steps = model.forward_steps(x)                         # (B, T, C)
    B, T, C = steps.shape
    w = torch.arange(1, T + 1, device=x.device, dtype=steps.dtype) ** 2
    w = w / w.sum()
    ce = F.cross_entropy(steps.reshape(B * T, C), y.repeat_interleave(T),
                         reduction="none").view(B, T)
    return (ce * w).sum(dim=1).mean(), steps[:, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gei", "lstm", "mean"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=None, help="mac dinh: gei=128, lstm/mean=16")
    ap.add_argument("--dropout", type=float, default=None, help="mac dinh: gei=0.1, lstm/mean=0.3")
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
    args = ap.parse_args()

    if args.tag and not re.fullmatch(r"[A-Za-z0-9]+", args.tag):
        sys.exit("--tag chi gom chu va so (khong dau gach).")
    if args.model != "lstm" and (args.incremental_loss or args.emb_pool != "mean"):
        sys.exit("--incremental_loss va --emb_pool last chi dung voi --model lstm.")

    if args.batch_size is None:
        args.batch_size = 128 if args.model == "gei" else 16

    tag = f"casiab_{args.model}" + (f"_{args.tag}" if args.tag else "") + f"_seed{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{tag}.pth")
    log_path = os.path.join(args.log_dir, f"{tag}.csv")
    if (os.path.exists(ckpt_path) or os.path.exists(log_path)) and not args.overwrite:
        sys.exit(f"{tag} da ton tai. Dung --overwrite neu muon ghi de.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| run:", tag)
    print("Cau hinh:", {k: v for k, v in vars(args).items()
                        if k in ("lr", "dropout", "batch_size", "T", "epochs", "clip_grad",
                                 "weight_decay", "warmup_epochs", "incremental_loss",
                                 "emb_pool", "eval_sample")})

    sp = args.splits
    num_classes = json.load(open(os.path.join(sp, "split_info.json")))["num_train_classes"]
    print("num_classes:", num_classes)

    train_ds = make_dataset(args.model, f"{sp}/train.csv", args.frames, args.T, train=True,
                            eval_sample=args.eval_sample)
    val_g = make_dataset(args.model, f"{sp}/val_gallery.csv", args.frames, args.T, train=False,
                         eval_sample=args.eval_sample)
    val_p = make_dataset(args.model, f"{sp}/val_probe.csv", args.frames, args.T, train=False,
                         eval_sample=args.eval_sample)

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=gen,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model, num_classes, args.dropout, args.emb_pool).to(device)
    print("Tham so:", sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    if args.weight_decay > 0:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = None
    warm_iters = int(args.warmup_epochs * len(train_loader))
    if warm_iters > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda it: min(1.0, (it + 1) / warm_iters))

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_nm", "val_bg", "val_cl", "val_mean"])

    best = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if args.incremental_loss:
                loss, out = incremental_loss(model, x, y)
            else:
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            run_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        tr_loss = run_loss / max(1, len(train_loader))
        tr_acc = 100 * correct / max(1, total)

        # --- validation open-set: embedding + cosine tren nguoi chua tung thay ---
        bs = 64 if args.model == "gei" else 16
        g_emb = extract_embeddings(model, val_g, device, bs, args.num_workers)
        p_emb = extract_embeddings(model, val_p, device, bs, args.num_workers)
        res, _ = rank1_openset(val_g.df, g_emb, val_p.df, p_emb)

        print(f"Epoch [{epoch}/{args.epochs}] loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
              f"val Rank-1 NM {res['nm']:.1f} BG {res['bg']:.1f} CL {res['cl']:.1f} "
              f"mean {res['mean']:.2f}")

        if res["mean"] > best:
            best, best_epoch = res["mean"], epoch
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "num_classes": num_classes, "epoch": epoch,
                        "val_mean": best}, ckpt_path)
            print(f"  -> saved (best val mean Rank-1 {best:.2f})")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, tr_loss, tr_acc, res["nm"], res["bg"], res["cl"], res["mean"]])

    print(f"Xong. Best epoch {best_epoch}, val mean Rank-1 {best:.2f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
