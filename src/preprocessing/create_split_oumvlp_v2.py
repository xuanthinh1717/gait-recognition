"""
create_split_oumvlp_v2.py

Chia open-set THEO NGUOI cho OU-MVLP (chuan Takemura et al., 2018 / OpenGait):
    - Train pool : cac subject le (subject % 2 == 1, ~5154 nguoi)
    - Test       : cac subject chan (subject % 2 == 0, ~5153 nguoi, khong bao gio dung train/val)
    - Val        : n_val nguoi (mac dinh 500) lay ngau nhien co seed tu Train pool de chon epoch nhanh
    - Train      : phan con lai cua Train pool (~4654 nguoi), danh nhan 0..N-1

Gallery = seq 00 (seq == 0) ; Probe = seq 01 (seq == 1), cond = "nm" (14 goc nhin: 0..90, 180..270).
Mỗi dòng CSV có thêm cột `npy_idx` (0..267385) khớp 1-1 với `data/npy/oumvlp_images.npy`
để `GEIDataset` đọc trực tiếp từ file `.npy` (mmap) siêu tốc mà không cần giải nén hàng trăm nghìn file PNG.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


def collect_from_existing_splits(splits_dir: Path):
    dfs = [pd.read_csv(splits_dir / f"{s}.csv") for s in ("train", "val", "test")]
    df = pd.concat(dfs, ignore_index=True)
    df["gei_path"] = df["path"].str.replace("\\", "/", regex=False)
    df = df.sort_values("gei_path").reset_index(drop=True)
    df["npy_idx"] = np.arange(len(df), dtype=np.int64)
    parts = df["gei_path"].str.extract(r"/(\d+)/(\d+)-(\d+)\.png$")
    df["subject"] = parts[0].astype(int)
    df["subj_dir"] = parts[0]
    df["angle"] = parts[1].astype(int)
    df["angle_dir"] = parts[1]
    df["seq"] = parts[2].astype(int)
    df["cond"] = "nm"
    df["seq_name"] = df["seq"].map(lambda s: f"nm-{s:02d}")
    return df[["subject", "subj_dir", "cond", "seq", "seq_name",
               "angle", "angle_dir", "npy_idx", "gei_path"]].copy()


def collect_from_gei_root(gei_root: Path):
    rows = []
    npy_idx = 0
    for subj_dir in sorted(os.listdir(gei_root)):
        sp = gei_root / subj_dir
        if not sp.is_dir():
            continue
        subj = int(subj_dir)
        for fname in sorted(os.listdir(sp)):
            if not fname.endswith(".png"):
                continue
            stem = fname[:-4]
            ang_str, seq_str = stem.split("-")
            rows.append(dict(
                subject=subj, subj_dir=subj_dir,
                cond="nm", seq=int(seq_str), seq_name=f"nm-{int(seq_str):02d}",
                angle=int(ang_str), angle_dir=ang_str,
                npy_idx=npy_idx,
                gei_path=f"{gei_root.as_posix()}/{subj_dir}/{fname}",
            ))
            npy_idx += 1
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_splits", default="data/splits",
                    help="thu muc chua train/val/test.csv cu de tai su dung danh sach file cuc nhanh")
    ap.add_argument("--gei_root", default="data/processed/gei_oumvlp")
    ap.add_argument("--labels_npy", default="data/npy/oumvlp_labels.npy")
    ap.add_argument("--out", default="data/splits_oumvlp_v2")
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    old_sp = Path(args.old_splits)
    if (old_sp / "train.csv").exists():
        print(f"Doc danh sach anh tu {old_sp}/{{train,val,test}}.csv ...")
        df = collect_from_existing_splits(old_sp)
    else:
        print(f"Quet thu muc {args.gei_root} ...")
        df = collect_from_gei_root(Path(args.gei_root))

    if Path(args.labels_npy).exists():
        lbl = np.load(args.labels_npy)
        assert len(lbl) == len(df) and np.array_equal(df["subject"].to_numpy() - 1, lbl), \
            "Thu tu npy_idx khong khop voi oumvlp_labels.npy!"
        print(f"Da xac nhan npy_idx khop 100% voi {args.labels_npy} ({len(lbl)} anh).")

    subjects = sorted(int(s) for s in df["subject"].unique())
    train_pool = [s for s in subjects if s % 2 == 1]
    test_pool = [s for s in subjects if s % 2 == 0]

    # Dam bao moi subject trong val/test deu co ca gallery (seq 0) va probe (seq 1)
    has_g = set(df[df["seq"] == 0]["subject"].unique())
    has_p = set(df[df["seq"] == 1]["subject"].unique())
    valid_eval_subs = has_g & has_p

    test_subj = [s for s in test_pool if s in valid_eval_subs]
    rng = random.Random(args.seed)
    val_candidates = [s for s in train_pool if s in valid_eval_subs]
    val_subj = sorted(rng.sample(val_candidates, args.n_val))
    val_set = set(val_subj)
    train_subj = [s for s in train_pool if s not in val_set]

    label_map = {s: i for i, s in enumerate(train_subj)}

    train_df = df[df["subject"].isin(train_subj)].copy()
    train_df["label"] = train_df["subject"].map(label_map)

    val_df = df[df["subject"].isin(val_set)].copy()
    val_df["label"] = -1

    test_df = df[df["subject"].isin(set(test_subj))].copy()
    test_df["label"] = -1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "train.csv": train_df,
        "val_gallery.csv": val_df[val_df["seq"] == 0].copy(),
        "val_probe.csv": val_df[val_df["seq"] == 1].copy(),
        "test_gallery.csv": test_df[test_df["seq"] == 0].copy(),
        "test_probe.csv": test_df[test_df["seq"] == 1].copy(),
    }
    for name, d in files.items():
        d.sort_values(["subject", "seq", "angle"]).to_csv(out / name, index=False)

    info = dict(
        dataset="oumvlp",
        seed=args.seed,
        num_train_classes=len(train_subj),
        train_subjects=train_subj,
        val_subjects=val_subj,
        test_subjects=test_subj,
        gallery=[["nm", [0]]],
        probe=[["nm", [1]]],
    )
    (out / "split_info.json").write_text(json.dumps(info, indent=2))

    print(f"Subject: train={len(train_subj)}  val={len(val_subj)}  test={len(test_subj)}")
    print(f"num_classes cho classifier = {len(train_subj)}")
    for name, d in files.items():
        print(f"{name:18s} {len(d):7d} anh, {d['subject'].nunique():5d} subject")


if __name__ == "__main__":
    main()
