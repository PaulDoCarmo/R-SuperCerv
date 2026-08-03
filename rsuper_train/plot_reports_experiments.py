#!/usr/bin/env python3
"""Figure des experiences supervision-rapports pour l'ICH SEULEMENT, 2 panneaux separes
RSNA (in-domaine masques) et CHUM (domaine des rapports). Lit eval/reports_exp_eval.csv.
Histoire : reports-only from-scratch marche pour l'ICH ; size-loss > ancre seule ; sur masques
les rapports degradent surtout RSNA (specialisation CHUM). python plot_reports_experiments.py
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

D = os.environ.get("RSUPER_DATA", "/mnt/Data/data_paul")
df = pd.read_csv(f"{D}/eval/reports_exp_eval.csv")
med = df.groupby(["model", "cls", "domain"]).dice.median().reset_index()


def g(model, dom):
    r = med[(med.model == model) & (med.cls == "ICH") & (med.domain == dom)].dice.values
    return r[0] if len(r) else np.nan


BARS = [
    ("ancre seule\n(scratch)", "rexp_scratch_anchor", "scratch"),
    ("ancre + size\n(scratch)", "rexp_scratch_sl05", "scratch"),
    ("X5", "base_X5", "base"), ("X5\n+rapports", "rexp_X5_sl05", "rep"),
    ("X10", "base_X10", "base"), ("X10\n+rapports", "rexp_X10_sl05", "rep"),
    ("X25", "base_X25", "base"), ("X25\n+rapports", "rexp_X25_sl05", "rep"),
]
COL = {"scratch": "#27ae60", "base": "#2980b9", "rep": "#e67e22"}
x = np.arange(len(BARS))
# paires (base, +rapports) pour dessiner le delta
PAIRS = [(2, 3), (4, 5), (6, 7)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), sharey=True); fig.patch.set_facecolor("white")
for ax, dom, dlabel in zip(axes, ["RSNA_test", "CHUM_report"], ["RSNA (in-domaine masques)", "CHUM (domaine des rapports)"]):
    vals = [g(m, dom) for _, m, _ in BARS]
    cols = [COL[k] for _, _, k in BARS]
    ax.bar(x, vals, 0.7, color=cols, edgecolor="black", lw=0.8, zorder=3)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    # delta base -> +rapports
    for bi, ri in PAIRS:
        d = vals[ri] - vals[bi]
        ax.annotate("", xy=(ri, vals[ri]), xytext=(bi, vals[bi]),
                    arrowprops=dict(arrowstyle="->", color="#c0392b" if d < -0.02 else "#7f8c8d", lw=1.6))
        ax.text((bi + ri) / 2, max(vals[bi], vals[ri]) + 0.05, f"{d:+.2f}",
                ha="center", fontsize=10, fontweight="bold", color="#c0392b" if d < -0.02 else "#555")
    ax.set_title(dlabel, fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in BARS], fontsize=9.5)
    ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3, zorder=0); ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(1.5, color="gray", lw=0.9, ls=":")
axes[0].set_ylabel("Dice médian ICH", fontsize=12)
leg = [Patch(fc=COL["scratch"], label="reports-only (from scratch, 0 masque)"),
       Patch(fc=COL["base"], label="baseline masques (stage-1)"),
       Patch(fc=COL["rep"], label="masques + rapports (fine-tune)")]
fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("ICH — supervision par RAPPORTS (85 cas ancrés) : RSNA vs CHUM\n"
             "reports SEULS (from scratch) ≈ niveau masques ; sur masques la dégradation frappe surtout RSNA "
             "(spécialisation domaine CHUM des rapports)",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout(rect=[0, 0.03, 1, 1])
out = f"{D}/previews/reports_experiments_ich.png"
fig.savefig(out, dpi=145, facecolor="white", bbox_inches="tight"); print("SAVED", out)
