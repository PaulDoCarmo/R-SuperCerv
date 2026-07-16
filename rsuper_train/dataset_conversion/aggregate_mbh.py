#!/usr/bin/env python3
"""Agrege les CSV d'eval MBH de tous les modeles du sweep -> tableau (sens/spec/sous-types/AUC)
+ courbes (sensibilite & specificite vs X, stage1 vs stage2) + LaTeX."""
import argparse, glob, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RX = re.compile(r"(stage1|stage2)_X(\d+)(?:_(base0|all))?_mbh\.csv$")
SUBS = ["intraparenchymal", "intraventricular", "subarachnoid", "subdural", "epidural"]


def summ(df):
    pos = df[df.positive == 1]; neg = df[df.healthy == 1]; iph = df[df.intraparenchymal == 1]
    extra = df[(df.positive == 1) & (df.intraparenchymal == 0)]
    d = dict(n_pos=len(pos), n_neg=len(neg),
             sens=round(100*pos.detected.mean(), 1), spec=round(100*(1-neg.detected.mean()), 1),
             fp=int(neg.detected.sum()),
             sens_iph=round(100*iph.detected.mean(), 1),
             sens_extra=round(100*extra.detected.mean(), 1) if len(extra) else np.nan)
    for s in SUBS:
        sub = df[df[s] == 1]; d[f"sens_{s}"] = round(100*sub.detected.mean(), 1) if len(sub) else np.nan
    try:
        from sklearn.metrics import roc_auc_score
        d["auc"] = round(roc_auc_score(df.positive, df.pred_ml), 3)
    except Exception:
        d["auc"] = np.nan
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbh_dir", required=True); ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(f"{a.mbh_dir}/*_mbh.csv")):
        m = RX.search(os.path.basename(f))
        if not m:
            continue
        stage, X, cfg = m.group(1), int(m.group(2)), m.group(3)
        model = "stage1" if stage == "stage1" else f"stage2_{cfg}"
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        s = {"X": X, "model": model}; s.update(summ(df)); rows.append(s)
    if not rows:
        print("aucun CSV MBH"); return
    tab = pd.DataFrame(rows).sort_values(["model", "X"])
    tab.to_csv(f"{a.out_dir}/mbh_summary.csv", index=False)
    print("=== MBH : sensibilite / specificite / sens_IPH / sens_extra / AUC ===")
    print(tab[["X", "model", "sens", "spec", "sens_iph", "sens_extra", "auc"]].to_string(index=False))

    # courbes sens & spec vs X
    for metric, ttl in [("sens", "Sensibilite (any ICH)"), ("spec", "Specificite (cas sains)"),
                        ("sens_iph", "Sensibilite IPH"), ("auc", "AUC")]:
        piv = tab.pivot_table(index="X", columns="model", values=metric)
        plt.figure(figsize=(7, 5), dpi=120)
        for col in piv.columns:
            plt.plot(piv.index, piv[col], "o-", label=col)
        plt.xlabel("X = nombre de masques"); plt.ylabel(metric)
        plt.title(f"MBH — {ttl} vs X"); plt.grid(alpha=.3); plt.legend()
        plt.xticks(sorted(tab.X.unique())); plt.tight_layout()
        plt.savefig(f"{a.out_dir}/mbh_{metric}_vs_X.png"); plt.close()

    # LaTeX
    lat = [r"\begin{table}[t]\centering",
           r"\caption{Detection ICH sur MBH (n=1939, hors fuite) : sensibilite (\%), specificite (\%), "
           r"sensibilite IPH et extra-axial pur, AUC. Stage~1 = masques ; Stage~2 = + rapports.}",
           r"\label{tab:mbh}", r"\begin{tabular}{ll|ccccc}", r"\toprule",
           r"$X$ & Mod\`ele & Sens. & Sp\'ec. & Sens.\,IPH & Sens.\,extra & AUC \\", r"\midrule"]
    for _, r in tab.iterrows():
        lat.append(f"{r.X} & {r.model.replace('_',' ')} & {r.sens} & {r.spec} & {r.sens_iph} & "
                   f"{r.sens_extra if pd.notna(r.sens_extra) else '--'} & {r.auc} \\\\")
    lat += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{a.out_dir}/table_mbh.tex", "w").write("\n".join(lat))
    print(f"\n-> {a.out_dir}/mbh_summary.csv + courbes + table_mbh.tex")


if __name__ == "__main__":
    main()
