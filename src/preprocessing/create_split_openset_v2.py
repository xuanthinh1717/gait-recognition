"""
create_split_openset_v2.py

Chia open-set THEO NGUOI cho CASIA-B (doc index.csv tu build_npy_casiab_v2.py).

    - Test : subject 075-124 (50 nguoi, khong bao gio dung de train/chon epoch)
    - Train: subject 001-074, tru n_val nguoi giu rieng lam validation
    - Val  : n_val nguoi lay NGAU NHIEN CO SEED tu 74 nguoi train
    - Nhan train duoc danh lai lien mach 0..N-1 (N = 74 - n_val)

Gallery = nm-01..04 ; Probe = nm-05..06, bg-01..02, cl-01..02
(val va test dung chung quy tac gallery/probe)

Dau ra (data/splits_v2/):
    train.csv, val_gallery.csv, val_probe.csv, test_gallery.csv, test_probe.csv, split_info.json
Moi dong la mot video: cac cot cua index.csv + label + gei_path
(gei_path tro toi anh GEI da co san de chay cau hinh D).
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd

GALLERY = {"nm": {1, 2, 3, 4}}
PROBE = {"nm": {5, 6}, "bg": {1, 2}, "cl": {1, 2}}


def in_rule(row, rule):
    return row["seq"] in rule.get(row["cond"], set())


def subset(df, rule):
    mask = df.apply(lambda r: in_rule(r, rule), axis=1)
    return df[mask].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/npy/casiab_v2/index.csv")
    ap.add_argument("--out", default="data/splits_v2")
    ap.add_argument("--gei_root", default="data/processed/gei")
    ap.add_argument("--n_train_subjects", type=int, default=74)
    ap.add_argument("--n_val", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    df = pd.read_csv(args.index, dtype={"subj_dir": str, "angle_dir": str})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df["gei_path"] = (
        args.gei_root + "/" + df["subj_dir"] + "/" + df["seq_name"] + "_" + df["angle_dir"] + ".png"
    )

    subjects = sorted(int(s) for s in df["subject"].unique())
    train_pool = [s for s in subjects if s <= args.n_train_subjects]
    test_subj = [s for s in subjects if s > args.n_train_subjects]

    rng = random.Random(args.seed)
    val_subj = sorted(rng.sample(train_pool, args.n_val))
    train_subj = [s for s in train_pool if s not in val_subj]

    label_map = {s: i for i, s in enumerate(train_subj)}

    # ---------------- train: moi video cua nguoi train ----------------
    train = df[df["subject"].isin(train_subj)].copy()
    train["label"] = train["subject"].map(label_map)

    def with_label(d):
        d = d.copy()
        d["label"] = -1  # val/test: khong co trong classifier
        return d

    val_df = df[df["subject"].isin(val_subj)]
    test_df = df[df["subject"].isin(test_subj)]

    files = {
        "train.csv": train,
        "val_gallery.csv": with_label(subset(val_df, GALLERY)),
        "val_probe.csv": with_label(subset(val_df, PROBE)),
        "test_gallery.csv": with_label(subset(test_df, GALLERY)),
        "test_probe.csv": with_label(subset(test_df, PROBE)),
    }
    for name, d in files.items():
        d.sort_values(["subject", "cond", "seq", "angle"]).to_csv(out / name, index=False)

    info = dict(
        seed=args.seed,
        num_train_classes=len(train_subj),
        train_subjects=train_subj,
        val_subjects=val_subj,
        test_subjects=test_subj,
        label_map={str(k): v for k, v in label_map.items()},
        gallery=sorted((k, sorted(v)) for k, v in GALLERY.items()),
        probe=sorted((k, sorted(v)) for k, v in PROBE.items()),
    )
    (out / "split_info.json").write_text(json.dumps(info, indent=2))

    # ---------------- bao cao + kiem tra ----------------
    print(f"Subject: train={len(train_subj)}  val={len(val_subj)}  test={len(test_subj)}")
    print(f"num_classes cho classifier = {len(train_subj)}")
    print(f"Val subjects: {val_subj}")
    for name, d in files.items():
        print(f"{name:18s} {len(d):6d} video, {d['subject'].nunique():3d} subject")

    assert not set(train_subj) & set(val_subj)
    assert not set(train_pool) & set(test_subj)
    assert not set(val_subj) & set(test_subj)
    assert sorted(train["label"].unique()) == list(range(len(train_subj)))

    # moi nguoi val/test phai co gallery va probe
    for tag, subs in (("val", val_subj), ("test", test_subj)):
        g = files[f"{tag}_gallery.csv"].groupby("subject").size()
        p = files[f"{tag}_probe.csv"].groupby("subject").size()
        missing = [s for s in subs if s not in g.index or s not in p.index]
        if missing:
            print(f"CANH BAO: {tag} thieu gallery/probe o subject {missing}")

    # kiem tra GEI ton tai (cho cau hinh D)
    miss = sum(1 for d in files.values() for p in d["gei_path"] if not Path(p).exists())
    if miss:
        print(f"CANH BAO: {miss} gei_path khong ton tai (chay tu thu muc goc project? "
              f"hoac sua --gei_root). Chi anh huong cau hinh D.")
    else:
        print("Tat ca gei_path deu ton tai.")


if __name__ == "__main__":
    main()
