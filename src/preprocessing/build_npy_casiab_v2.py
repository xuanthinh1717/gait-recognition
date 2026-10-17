"""
build_npy_casiab_v2.py

Dong goi silhouette goc cua CASIA-B thanh:
    data/npy/casiab_v2/frames.npy   (N_frames, S, S) uint8, cac frame noi tiep nhau
    data/npy/casiab_v2/index.csv    mot dong = mot video (subject, dieu kien, goc)

Cau truc thu muc mong doi:
    CASIA-B/<subject>/<cond-NN>/<angle>/<subject>-<cond>-<NN>-<angle>-<frame>.png
    vi du: 001/bg-01/000/001-bg-01-000-001.png

Chay:
    python build_npy_casiab_v2.py --check     # chi quet + bao cao, khong ghi gi
    python build_npy_casiab_v2.py             # dong goi that
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

SEQ_RE = re.compile(r"^(nm|bg|cl)-(\d+)$", re.I)
ANG_RE = re.compile(r"^\d{1,3}$")
FILE_RE = re.compile(r"^(\d+)-(nm|bg|cl)-(\d+)-(\d+)-(\d+)\.png$", re.I)


def scan(root: Path):
    """Quet cay thu muc. Tra ve (videos, problems). Frame duoc sap theo SO frame."""
    videos, problems = [], []

    for subj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not subj_dir.name.isdigit():
            problems.append(f"thu muc subject la: {subj_dir}")
            continue
        subj = int(subj_dir.name)

        for seq_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir()):
            m = SEQ_RE.match(seq_dir.name)
            if not m:
                problems.append(f"thu muc sequence la: {seq_dir}")
                continue
            cond, num = m.group(1).lower(), int(m.group(2))

            for ang_dir in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
                if not ANG_RE.match(ang_dir.name):
                    problems.append(f"thu muc goc la: {ang_dir}")
                    continue
                angle = int(ang_dir.name)

                frames = []
                for f in ang_dir.iterdir():
                    if f.suffix.lower() != ".png":
                        continue
                    fm = FILE_RE.match(f.name)
                    if not fm:
                        problems.append(f"ten file la: {f}")
                        continue
                    s, c, n, a, fr = fm.groups()
                    if (int(s), c.lower(), int(n), int(a)) != (subj, cond, num, angle):
                        problems.append(f"ten file khong khop thu muc: {f}")
                        continue
                    frames.append((int(fr), f))

                if not frames:
                    problems.append(f"video rong: {ang_dir}")
                    continue

                frames.sort(key=lambda t: t[0])
                videos.append(dict(
                    subject=subj, subj_dir=subj_dir.name,
                    cond=cond, seq=num, seq_name=seq_dir.name,
                    angle=angle, angle_dir=ang_dir.name,
                    frames=[f for _, f in frames],
                    frame_nums=[n for n, _ in frames],
                ))
    return videos, problems


def report(videos, problems, root):
    print(f"Root: {root}")
    print(f"Video: {len(videos)}  | Subject: {len({v['subject'] for v in videos})}")
    print("Dieu kien :", dict(sorted(Counter((v['cond'], v['seq']) for v in videos).items())))
    print("Goc       :", sorted({v['angle'] for v in videos}))

    lens = np.array([len(v["frames"]) for v in videos])
    print(f"So frame/video: min={lens.min()} median={int(np.median(lens))} max={lens.max()} "
          f"| tong={lens.sum()}")

    gaps = sum(1 for v in videos
               if v["frame_nums"][-1] - v["frame_nums"][0] + 1 != len(v["frame_nums"]))
    print(f"Video co frame bi nhay so (thieu frame giua chung): {gaps}")

    shapes = Counter()
    for v in videos:
        img = cv2.imread(str(v["frames"][0]), cv2.IMREAD_GRAYSCALE)
        shapes[None if img is None else img.shape] += 1
    print("Kich thuoc frame dau moi video:", dict(shapes))

    per_subj = Counter(v["subject"] for v in videos)
    odd = {s: c for s, c in per_subj.items() if c != 110}
    print(f"Subject khong du 110 video (6 nm + 2 bg + 2 cl) x 11 goc: {len(odd)}")
    if odd:
        print("  ", dict(sorted(odd.items())[:15]), "..." if len(odd) > 15 else "")

    print(f"Van de ve ten/cau truc: {len(problems)}")
    for p in problems[:20]:
        print("  -", p)
    if len(problems) > 20:
        print(f"  ... con {len(problems) - 20} muc nua")


def build(videos, out_dir: Path, size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(len(v["frames"]) for v in videos)
    print(f"Ghi {total} frame ({total * size * size / 1e9:.2f} GB) vao {out_dir}")

    arr = np.lib.format.open_memmap(
        out_dir / "frames.npy", mode="w+", dtype=np.uint8, shape=(total, size, size)
    )

    resized = 0
    pos = 0
    rows = []
    for i, v in enumerate(videos):
        start = pos
        for f in v["frames"]:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Khong doc duoc anh: {f}")
            if img.shape != (size, size):
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                resized += 1
            arr[pos] = img
            pos += 1
        rows.append(dict(
            subject=v["subject"], subj_dir=v["subj_dir"],
            cond=v["cond"], seq=v["seq"], seq_name=v["seq_name"],
            angle=v["angle"], angle_dir=v["angle_dir"],
            start=start, length=pos - start,
        ))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(videos)} video")

    arr.flush()
    assert pos == total

    with open(out_dir / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Xong. frames.npy {arr.shape}, index.csv {len(rows)} dong")
    if resized:
        print(f"CANH BAO: {resized} frame phai resize ve {size}x{size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/CASIA-B")
    ap.add_argument("--out", default="data/npy/casiab_v2")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--check", action="store_true", help="chi quet va bao cao")
    ap.add_argument("--force", action="store_true", help="van build du co van de ve ten")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"Khong thay thu muc: {root}")

    videos, problems = scan(root)
    if not videos:
        sys.exit("Khong tim thay video nao.")

    report(videos, problems, root)

    if args.check:
        return
    if problems and not args.force:
        sys.exit("\nCo van de ve ten/cau truc o tren. Sua hoac chay lai voi --force.")

    build(videos, Path(args.out), args.size)


if __name__ == "__main__":
    main()
