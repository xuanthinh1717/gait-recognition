"""
train_v2.py

Train bang softmax tren cac nguoi train; CHON EPOCH bang Rank-1 open-set
(embedding + cosine) tren cac nguoi validation giu rieng. Khong dung nguoi test.

Vi du:
    python train_v2.py --model gei  --seed 1
    python train_v2.py --model lstm --seed 1
    python train_v2.py --model mean --seed 1

Dau ra (khong de len ban cu, ten co tag rieng):
    outputs_v2/casiab_<model>_seed<N>.pth
    logs_v2/casiab_<model>_seed<N>.csv
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
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
    args = ap.parse_args()

    if args.batch_size is None:
        args.batch_size = 128 if args.model == "gei" else 16

    tag = f"casiab_{args.model}_seed{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{tag}.pth")
    log_path = os.path.join(args.log_dir, f"{tag}.csv")
    if (os.path.exists(ckpt_path) or os.path.exists(log_path)) and not args.overwrite:
        sys.exit(f"{tag} da ton tai. Dung --overwrite neu muon ghi de.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| run:", tag)

    sp = args.splits
    num_classes = json.load(open(os.path.join(sp, "split_info.json")))["num_train_classes"]
    print("num_classes:", num_classes)

    train_ds = make_dataset(args.model, f"{sp}/train.csv", args.frames, args.T, train=True)
    val_g = make_dataset(args.model, f"{sp}/val_gallery.csv", args.frames, args.T, train=False)
    val_p = make_dataset(args.model, f"{sp}/val_probe.csv", args.frames, args.T, train=False)

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=gen,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model, num_classes, args.dropout).to(device)
    print("Tham so:", sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
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
