"""
evaluate_openset_v2.py

Danh gia open-set: trich embedding, so khop cosine gallery-probe, bo cung goc.

Vi du:
    python evaluate_openset_v2.py --ckpt outputs_v2/casiab_gei_seed1.pth
    python evaluate_openset_v2.py --ckpt outputs_v2/casiab_lstm_seed1.pth \
        outputs_v2/casiab_lstm_seed2.pth outputs_v2/casiab_lstm_seed3.pth

Nhieu checkpoint (cung model, khac seed) -> in them trung binh +- do lech chuan.
Chi dung --split test sau khi da chon xong cau hinh; val de debug.
"""
import argparse
import json
import os

import numpy as np
import torch

from src.evaluation.openset_utils_v2 import (CONDS, build_model, extract_embeddings,
                              make_dataset, rank1_openset)


def evaluate(ckpt_path, split, splits_dir, frames, device, num_workers):
    ck = torch.load(ckpt_path, map_location=device)
    a = ck["args"]
    model = build_model(a["model"], ck["num_classes"], a["dropout"],
                        pool=a.get("emb_pool", "mean")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    eval_sample = a.get("eval_sample", "center")
    g = make_dataset(a["model"], f"{splits_dir}/{split}_gallery.csv", frames, a["T"],
                     train=False, eval_sample=eval_sample)
    p = make_dataset(a["model"], f"{splits_dir}/{split}_probe.csv", frames, a["T"],
                     train=False, eval_sample=eval_sample)
    bs = 64 if a["model"] == "gei" else 16
    g_emb = extract_embeddings(model, g, device, bs, num_workers)
    p_emb = extract_embeddings(model, p, device, bs, num_workers)
    per_cond, per_angle = rank1_openset(g.df, g_emb, p.df, p_emb)
    return a, ck, per_cond, per_angle


def print_run(name, ck, per_cond, per_angle):
    print(f"\n=== {name} (epoch {ck['epoch']}, val mean {ck['val_mean']:.2f}) ===")
    print("Rank-1 (%):  " + "  ".join(f"{c.upper()} {per_cond[c]:.2f}"
                                       for c in CONDS if c in per_cond)
          + f"  | mean {per_cond['mean']:.2f}")
    angles = sorted({ang for r in per_angle.values() for ang in r})
    print("Goc      " + " ".join(f"{a:>6d}" for a in angles))
    for c in CONDS:
        if c in per_angle:
            print(f"{c.upper():8s} " + " ".join(
                f"{per_angle[c].get(a, float('nan')):6.1f}" for a in angles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--splits", default="data/splits_v2")
    ap.add_argument("--frames", default="data/npy/casiab_v2/frames.npy")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_dir", default="results_v2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    runs = []
    for path in args.ckpt:
        a, ck, per_cond, per_angle = evaluate(
            path, args.split, args.splits, args.frames, device, args.num_workers)
        name = os.path.splitext(os.path.basename(path))[0]
        print_run(name, ck, per_cond, per_angle)
        key = a["model"] + (f"_{a['tag']}" if a.get("tag") else "")
        runs.append((name, key, per_cond))
        with open(os.path.join(args.out_dir, f"{name}_{args.split}.json"), "w") as f:
            json.dump({"ckpt": path, "split": args.split, "per_cond": per_cond,
                       "per_angle": per_angle}, f, indent=2)

    if len(runs) > 1:
        models = {m for _, m, _ in runs}
        if len(models) > 1:
            print("\n(Cac checkpoint thuoc nhieu model/bien the khac nhau, bo qua mean+-std.)")
            return
        print(f"\n=== Tong hop {len(runs)} run, model {models.pop()}, split {args.split} ===")
        for k in list(CONDS) + ["mean"]:
            v = np.array([r[k] for _, _, r in runs if k in r])
            print(f"{k.upper():5s} {v.mean():.2f} +- {v.std(ddof=1):.2f}")


if __name__ == "__main__":
    main()
