#!/usr/bin/env python3
"""Figure de generalisation par classe (ICH/IVH/PHE) : 3 panneaux, barres groupees
RSNA (in-domaine) vs CHUM-rapports (hors-domaine), x = nb de masques stage-1.
Lit $D/eval/ich3_generalization.csv (produit par eval_ich3_gen.py) et ecrit le PNG.
Reutilisable : re-lancer apres avoir ajoute des modeles a la sweep (X5, X10, ...).
  python plot_ich3_generalization.py
"""
import os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.environ.get("RSUPER_DATA", "/mnt/Data/data_paul")
CSV = f"{D}/eval/ich3_generalization.csv"
OUT = f"{D}/previews/generalization_ich_ivh_phe.png"
CLS = ["ICH", "IVH", "PHE"]


def xnum(m):  # "X25" -> 25, pour trier
    g = re.search(r"\d+", m); return int(g.group()) if g else 1e9


def main():
    df = pd.read_csv(CSV)
    models = sorted(df["model"].unique(), key=xnum)
    med = df.pivot_table(index=["model", "cls"], columns="domain", values="dice", aggfunc="median")
    cnt = df.pivot_table(index=["model", "cls"], columns="domain", values="dice", aggfunc="count")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), sharey=True)
    fig.patch.set_facecolor("white")
    x = np.arange(len(models)); w = 0.38
    for ax, cl in zip(axes, CLS):
        rsna = [med.loc[(m, cl), "RSNA_test"] if (m, cl) in med.index else np.nan for m in models]
        chum = [med.loc[(m, cl), "CHUM_report"] if (m, cl) in med.index else np.nan for m in models]
        b1 = ax.bar(x - w/2, rsna, w, label="RSNA (in-domaine)", color="#1f77b4")
        b2 = ax.bar(x + w/2, chum, w, label="CHUM (hors-domaine)", color="#d62728")
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                if not np.isnan(h):
                    ax.text(b.get_x()+b.get_width()/2, h+0.012, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(cl, fontsize=15, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
        ax.set_xlabel("nb de masques", fontsize=11)
        ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Dice median", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=10, framealpha=0.95)
    # n cas (ICH RSNA/CHUM) pour la legende du titre
    m0 = models[0]
    fig.suptitle("Generalisation RSNA -> CHUM par classe et nb de masques stage-1  "
                 "(X5/X10 = latest.pth epoch 60 ; ICH ~ pas de gap ; IVH gap modere qui se reduit ; "
                 "PHE gros gap constant = artefact d'annotation RSNA oedeme 4x CHUM)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor="white", bbox_inches="tight")
    print("SAVED", OUT)
    print("\nmodeles:", models)
    print(med.round(3).to_string())
    print("\n(n cas)\n", cnt.to_string())


if __name__ == "__main__":
    main()
