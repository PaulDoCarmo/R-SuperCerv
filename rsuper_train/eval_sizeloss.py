#!/usr/bin/env python3
"""Evalue les stage-2 supervision-rapports vs baselines stage-1 : Dice ICH/IVH/PHE sur
test RSNA + GT-rapports CHUM (meme protocole que eval_ich3_gen.eval_model).
But : la supervision-rapports (ancre + size-loss, ICH seul) ameliore/degrade/laisse neutre
l'ICH ? (IVH/PHE = temoins non supervises.)

Usage : python eval_sizeloss.py [PATTERN] [OUT_CSV]
  PATTERN : glob des dossiers exp/ich_ufo a evaluer (defaut 'rexp_*').
  Inclut toujours les baselines stage-1 (X5/X10/X25) pour comparaison.
  Checkpoint : fold_0_latest.pth de preference (val OFF), sinon fold_0_best.pth."""
import os, glob, sys, pandas as pd
from eval_ich3_gen import eval_model, D


def ckpt(dirp):
    for name in ("fold_0_latest.pth", "fold_0_best.pth"):
        p = os.path.join(dirp, name)
        if os.path.exists(p):
            return p
    return None


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "rexp_*"
    out = sys.argv[2] if len(sys.argv) > 2 else f"{D}/eval/reports_exp_eval.csv"

    models = {}
    for b in ("X5", "X10", "X25"):                     # baselines stage-1 (avant rapports)
        p = ckpt(f"{D}/exp/ich/ich3_stage1_{b}")
        if p:
            models[f"base_{b}"] = p
    for d in sorted(glob.glob(f"{D}/exp/ich_ufo/{pattern}")):   # experiences rapports
        p = ckpt(d)
        if p:
            models[os.path.basename(d)] = p

    rows = []
    for name, ck in models.items():
        print(f"=== eval {name}  ({os.path.basename(ck)}) ===", flush=True)
        rows += eval_model(name, ck)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    piv = df.groupby(["model", "cls", "domain"]).dice.median().reset_index()
    print("\n===== DICE MEDIAN (ICH = cible ; IVH/PHE = temoins non supervises) =====")
    for cl in ["ICH", "IVH", "PHE"]:
        print(f"\n--- {cl} ---")
        for name in models:
            r = piv[(piv.model == name) & (piv.cls == cl)]
            rsna = r[r.domain == "RSNA_test"].dice.values
            chum = r[r.domain == "CHUM_report"].dice.values
            if len(rsna) and len(chum):
                print(f"  {name:26s} RSNA={rsna[0]:.3f}  CHUM={chum[0]:.3f}")
    print(f"\n-> {out}")
    print("Lecture : comparer chaque modele-rapports a sa baseline (base_X5/X10/X25). "
          "ICH en hausse = gain de la supervision-rapports ; ~egal = neutre ; en baisse = degrade.")


if __name__ == "__main__":
    os.makedirs(f"{D}/eval", exist_ok=True)
    main()
