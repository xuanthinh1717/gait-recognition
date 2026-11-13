import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import numpy as np
import pandas as pd

VARIANTS = {
    "samarc": {
        "optimizer": "sam",
        "rho": 0.05,
        "arcface": True,
        "switch_arcface_acc": 80.0,
        "label_smoothing": 0.1,
    },
    "asamarc": {
        "optimizer": "asam",
        "rho": 0.5,
        "arcface": True,
        "switch_arcface_acc": 80.0,
        "label_smoothing": 0.1,
    },
}

def update_summary():
    files = glob.glob("results_v2_gei/*_test.json")
    data = {}
    for f in files:
        m = re.match(r".*casiab_gei_(.+)_seed(\d+)_test\.json", os.path.basename(f))
        if not m:
            continue
        variant, seed = m.groups()
        with open(f, "r", encoding="utf-8") as fp:
            res = json.load(fp).get("per_cond", {})
        data.setdefault(variant, []).append(res)

    rows = []
    for var, runs in sorted(data.items()):
        n = len(runs)
        row = {"variant": f"gei_{var}", "n_seed": n}
        for k in ["nm", "bg", "cl", "mean"]:
            vals = [r[k] for r in runs if k in r]
            if len(vals) > 1:
                row[k.upper() if k != "mean" else "mean"] = f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}"
            elif len(vals) == 1:
                row[k.upper() if k != "mean" else "mean"] = f"{vals[0]:.2f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results_v2_gei/summary_test.csv", index=False)
    print("\n[SUMMARY] Updated results_v2_gei/summary_test.csv", flush=True)


def run_cmd(cmd_list):
    print(f"\n[RUN] {' '.join(cmd_list)}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd_list, check=True)
    dt = time.time() - t0
    print(f"[DONE] Elapsed: {dt:.1f}s", flush=True)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["samarc", "asamarc"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 1806])
    parser.add_argument("--epochs", type=int, default=45)
    args = parser.parse_args()

    python_exe = sys.executable

    print(f"=== Starting Experiments for {args.variants} on seeds {args.seeds} ===", flush=True)

    for var_name in args.variants:
        cfg = VARIANTS[var_name]
        print(f"\n=======================================================", flush=True)
        print(f" Variant: {var_name} | Config: {cfg}", flush=True)
        print(f"=======================================================", flush=True)

        ckpt_list = []
        for s in args.seeds:
            ckpt_path = f"outputs_v2_gei/casiab_gei_{var_name}_seed{s}.pth"
            ckpt_list.append(ckpt_path)

            train_cmd = [
                python_exe, "-m", "src.training.train_v2",
                "--model", "gei",
                "--seed", str(s),
                "--epochs", str(args.epochs),
                "--lr", "0.001",
                "--batch_size", "128",
                "--dropout", "0.1",
                "--num_workers", "0",
                "--out_dir", "outputs_v2_gei",
                "--log_dir", "logs_v2_gei",
                "--clip_grad", "1.0",
                "--weight_decay", "1e-4",
                "--warmup_epochs", "3.0",
                "--triplet_weight", "1.0",
                "--triplet_margin", "0.2",
                "--pk_k", "4",
                "--label_smoothing", str(cfg["label_smoothing"]),
                "--decay_acc", "95.0",
                "--min_lr", "1e-5",
                "--optimizer", cfg["optimizer"],
                "--rho", str(cfg["rho"]),
                "--arcface",
                "--switch_arcface_acc", str(cfg["switch_arcface_acc"]),
                "--tag", var_name,
                "--overwrite"
            ]

            run_cmd(train_cmd)

            eval_single_cmd = [
                python_exe, "-m", "src.evaluation.evaluate_openset_v2",
                "--ckpt", ckpt_path,
                "--split", "test",
                "--num_workers", "0",
                "--out_dir", "results_v2_gei"
            ]
            run_cmd(eval_single_cmd)
            update_summary()

        eval_all_cmd = [
            python_exe, "-m", "src.evaluation.evaluate_openset_v2",
            "--ckpt", *ckpt_list,
            "--split", "test",
            "--num_workers", "0",
            "--out_dir", "results_v2_gei"
        ]
        run_cmd(eval_all_cmd)
        update_summary()

    print("\n=======================================================", flush=True)
    print(" ALL EXPERIMENTS COMPLETE! ", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    main()
