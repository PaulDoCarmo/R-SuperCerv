#!/usr/bin/env python3
"""Agrege les CSV d'eval de la sweep -> tableau comparatif + courbes (Dice & detection vs X,
stage1 vs stage2_base0 vs stage2_all). Tolerant aux CSV manquants (jobs echoues)."""
import argparse, glob, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RX = re.compile(r"X(\d+)_(stage1|stage2_base0|stage2_all)__(masks|rep0|repall)\.csv$")


def masks_summary(df):
    d = df["dice"].astype(float)
    return dict(dice_med=round(d.median(), 3), dice_mean=round(d.mean(), 3),
                nsd_med=round(df["nsd"].astype(float).median(), 3),
                hd95_med=round(pd.to_numeric(df["hd95_mm"], errors="coerce").median(), 1),
                det_pct=round(100*df["detected"].astype(float).mean(), 0), n=len(df))


def reports_summary(df):
    sens = 100*df["case_detected"].astype(float).mean()
    tp = pd.to_numeric(df["region_tp"], errors="coerce").sum()
    fp = pd.to_numeric(df["region_fp"], errors="coerce").sum()
    fn = pd.to_numeric(df["region_fn"], errors="coerce").sum()
    rec = tp/(tp+fn) if (tp+fn) else float("nan")
    prec = tp/(tp+fp) if (tp+fp) else float("nan")
    f1 = 2*prec*rec/(prec+rec) if (prec and rec and prec+rec) else float("nan")
    return dict(sens_pct=round(sens, 0), recall=round(rec, 3), precision=round(prec, 3),
                f1=round(f1, 3), n=int(df.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

    rows = []
    for f in sorted(glob.glob(f"{a.eval_dir}/*.csv")):
        m = RX.search(os.path.basename(f))
        if not m: continue
        X, model, mset = int(m.group(1)), m.group(2), m.group(3)
        try: df = pd.read_csv(f)
        except Exception: continue
        if df.empty: continue
        s = {"X": X, "model": model, "metric_set": mset}
        s.update(masks_summary(df) if mset == "masks" else reports_summary(df))
        rows.append(s)
    if not rows:
        print("aucun CSV d'eval trouve"); return
    tab = pd.DataFrame(rows).sort_values(["metric_set", "X", "model"])
    tab.to_csv(f"{a.out_dir}/summary_long.csv", index=False)

    # --- tableau masques (Dice vs X, par modele) ---
    mk = tab[tab.metric_set == "masks"].pivot_table(index="X", columns="model", values="dice_med")
    mk.to_csv(f"{a.out_dir}/masks_dice_median.csv")
    print("=== Dice median (masques test) vs X ===\n", mk.to_string())

    # --- courbe Dice masques ---
    if not mk.empty:
        plt.figure(figsize=(7, 5), dpi=120)
        for col in mk.columns:
            plt.plot(mk.index, mk[col], marker="o", label=col)
        plt.xlabel("X (nombre de masques d'entrainement)"); plt.ylabel("Dice median (test masques, 54)")
        plt.title("Dice vs nombre de masques — stage1 vs stage2"); plt.grid(alpha=.3); plt.legend()
        plt.tight_layout(); plt.savefig(f"{a.out_dir}/curve_masks_dice.png"); plt.close()

    # --- courbe detection (F1 + precision region) sur rep0 et repall ---
    for mset in ("rep0", "repall"):
        sub = tab[tab.metric_set == mset]
        if sub.empty: continue
        for metric in ("f1", "precision", "recall"):
            piv = sub.pivot_table(index="X", columns="model", values=metric)
            piv.to_csv(f"{a.out_dir}/det_{mset}_{metric}.csv")
        plt.figure(figsize=(7, 5), dpi=120)
        piv = sub.pivot_table(index="X", columns="model", values="f1")
        for col in piv.columns: plt.plot(piv.index, piv[col], marker="o", label=col)
        plt.xlabel("X (masques)"); plt.ylabel(f"F1 localisation ({mset})")
        plt.title(f"Detection F1 vs X — {mset}"); plt.grid(alpha=.3); plt.legend()
        plt.tight_layout(); plt.savefig(f"{a.out_dir}/curve_det_{mset}_f1.png"); plt.close()

    print(f"\n-> {a.out_dir}/summary_long.csv + masks_dice_median.csv + courbes PNG")


if __name__ == "__main__":
    main()
