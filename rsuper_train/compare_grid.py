#!/usr/bin/env python3
"""
compare_grid.py -- Compare les runs du grid d'hyperparametres.

Pour chaque run <name> :
  - lit le meilleur Val/Dice (events TensorBoard) -> selection / overfit-check ;
  - evalue son best-model (fold_0_best.pth) sur le TEST held-out (via eval_ich.py)
    -> Dice median/moyen, NSD median, taux de detection ;
Ecrit un CSV comparatif trie par Dice test median (la metrique qui tranche).

A lancer depuis rsuper_train/ (sur GPU, car l'eval fait de l'inference).
"""
import argparse
import csv
import glob
import os
import subprocess
import sys

import numpy as np


def best_val_dice(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    best = None
    for ef in sorted(glob.glob(os.path.join(logdir, "events.out.tfevents.*"))):
        ea = EventAccumulator(ef, size_guidance={"scalars": 0})
        ea.Reload()
        if "Val/Dice" in ea.Tags().get("scalars", []):
            for s in ea.Scalars("Val/Dice"):
                best = s.value if best is None else max(best, s.value)
    return best


def eval_on_test(ckpt, out_csv, gpu):
    subprocess.run([sys.executable, "eval_ich.py", "--load", ckpt,
                    "--save_csv", out_csv, "--gpu", str(gpu)],
                   check=True)
    rows = list(csv.DictReader(open(out_csv)))
    d = np.array([float(r["dice"]) for r in rows])
    nsd = np.array([float(r["nsd"]) for r in rows])
    det = np.array([int(r["detected"]) for r in rows])
    return dict(test_dice_median=float(np.median(d)), test_dice_mean=float(d.mean()),
                test_nsd_median=float(np.median(nsd)), detect_rate=float(det.mean()), n_test=len(d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+",
                   default=["R0_base", "R1_lrlo", "R2_lrhi", "R3_patch", "R4_aug"])
    p.add_argument("--exp_dir", default="/home/pauldcrm/links/scratch/R-SuperCerv/exp/ich")
    p.add_argument("--log_dir", default="/home/pauldcrm/links/scratch/R-SuperCerv/log/ich")
    p.add_argument("--eval_dir", default="/home/pauldcrm/links/scratch/R-SuperCerv/eval/grid")
    p.add_argument("--out", default="/home/pauldcrm/links/scratch/R-SuperCerv/eval/grid_comparison.csv")
    p.add_argument("--gpu", default="0")
    args = p.parse_args()
    os.makedirs(args.eval_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for name in args.names:
        run = f"ich_{name}"
        ckpt = os.path.join(args.exp_dir, run, "fold_0_best.pth")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(args.exp_dir, run, "fold_0_latest.pth")  # fallback
        if not os.path.exists(ckpt):
            print(f"[skip] {run} : aucun checkpoint")
            continue

        vb = best_val_dice(os.path.join(args.log_dir, run, "fold_0"))
        print(f"\n>>> {name} : Val best Dice = {vb if vb is None else round(vb,4)} | eval test ({os.path.basename(ckpt)})...")
        m = eval_on_test(ckpt, os.path.join(args.eval_dir, f"{name}.csv"), args.gpu)
        rows.append(dict(config=name, ckpt=os.path.basename(ckpt),
                         val_best_dice=None if vb is None else round(vb, 4),
                         **{k: round(v, 4) for k, v in m.items()}))

    rows.sort(key=lambda r: r["test_dice_median"], reverse=True)

    fields = ["config", "val_best_dice", "test_dice_median", "test_dice_mean",
              "test_nsd_median", "detect_rate", "n_test", "ckpt"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n================= COMPARATIF (trie par Dice test median) =================")
    print(f"{'config':10s} {'val_best':>9s} {'test_Dice_med':>13s} {'test_Dice_mean':>14s} "
          f"{'NSD_med':>8s} {'detect':>7s}")
    for r in rows:
        vb = "n/a" if r["val_best_dice"] is None else f"{r['val_best_dice']:.3f}"
        print(f"{r['config']:10s} {vb:>9s} {r['test_dice_median']:>13.3f} "
              f"{r['test_dice_mean']:>14.3f} {r['test_nsd_median']:>8.3f} {r['detect_rate']:>7.2f}")
    print(f"\nCSV -> {args.out}")


if __name__ == "__main__":
    main()
