#!/usr/bin/env python3
"""
plot_loss.py -- Trace l'evolution de la loss d'entrainement depuis les events TensorBoard.

Lit TOUS les fichiers d'events d'un dossier (gere les reprises : un step ecrit
plusieurs fois est ecrase par la valeur la plus recente), puis sauve un PNG.

  python plot_loss.py --logdir /home/.../log/ich/ich_stage1/fold_0 --out loss_curve.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(logdir, tag):
    pts = {}
    for ef in sorted(glob.glob(os.path.join(logdir, "events.out.tfevents.*"))):
        ea = EventAccumulator(ef, size_guidance={"scalars": 0})
        ea.Reload()
        if tag in ea.Tags().get("scalars", []):
            for s in ea.Scalars(tag):
                pts[s.step] = s.value  # reprise : le dernier fichier ecrase
    steps = sorted(pts)
    return steps, [pts[s] for s in steps]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="/home/pauldcrm/links/scratch/R-SuperCerv/log/ich/ich_stage1/fold_0")
    p.add_argument("--out", default="/home/pauldcrm/links/scratch/R-SuperCerv/loss_curve.png")
    p.add_argument("--tags", nargs="+", default=["Train/overall", "Train/segmentation"])
    args = p.parse_args()

    plt.figure(figsize=(9, 5))
    printed = False
    for tag in args.tags:
        steps, vals = load_scalar(args.logdir, tag)
        if not steps:
            continue
        plt.plot(steps, vals, marker="o", ms=2, label=tag)
        if not printed:
            print(f"{tag}: {len(steps)} epochs | premier={vals[0]:.3f} (ep {steps[0]}) "
                  f"-> dernier={vals[-1]:.3f} (ep {steps[-1]}) | min={min(vals):.3f}")
            printed = True

    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("ICH stage 1 - loss d'entrainement")
    plt.grid(True, alpha=0.3); plt.legend()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=130)
    print(f"PNG -> {args.out}")


if __name__ == "__main__":
    main()
