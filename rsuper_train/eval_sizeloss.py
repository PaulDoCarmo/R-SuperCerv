#!/usr/bin/env python3
"""Evalue les stage-2 'size-loss' vs le stage-1 X25 (baseline) : Dice ICH/IVH/PHE sur
test RSNA + GT-rapports CHUM. Reutilise eval_model de eval_ich3_gen (meme protocole).
But : la size-loss (ICH seul) degrade-t-elle / ameliore-t-elle / laisse-t-elle neutre
la segmentation ICH ? (IVH/PHE servent de temoin : ils ne sont pas supervises.)
Lancer depuis rsuper_train/ apres run_sizeloss.sh."""
import os, glob, pandas as pd
from eval_ich3_gen import eval_model, D

BASE = f"{D}/exp/ich/ich3_stage1_X25/fold_0_best.pth"          # reference (avant size-loss)
UFO = f"{D}/exp/ich_ufo"


def main():
    models = {"stage1_X25 (baseline)": BASE}
    for ck in sorted(glob.glob(f"{UFO}/sizeloss_X25_*/fold_0_best.pth")):
        models[os.path.basename(os.path.dirname(ck))] = ck

    rows = []
    for name, ck in models.items():
        if not os.path.exists(ck):
            print(f"[skip] {name} (pas de ckpt)"); continue
        print(f"=== eval {name} ===", flush=True)
        rows += eval_model(name, ck)
    df = pd.DataFrame(rows); df.to_csv(f"{D}/eval/sizeloss_eval.csv", index=False)

    piv = df.groupby(["model", "cls", "domain"]).dice.median().reset_index()
    print("\n===== DICE MEDIAN (ICH d'abord ; IVH/PHE = temoins non supervises) =====")
    for cl in ["ICH", "IVH", "PHE"]:
        print(f"\n--- {cl} ---")
        for name in models:
            r = piv[(piv.model == name) & (piv.cls == cl)]
            rsna = r[r.domain == "RSNA_test"].dice.values
            chum = r[r.domain == "CHUM_report"].dice.values
            if len(rsna) and len(chum):
                print(f"  {name:28s} RSNA={rsna[0]:.3f}  CHUM={chum[0]:.3f}")
    print(f"\n-> {D}/eval/sizeloss_eval.csv")
    print("Lecture : comparer chaque sizeloss_* au baseline. ICH ~egal = NEUTRE (attendu, "
          "pas de marge) ; ICH en baisse = la size-loss degrade (comme la ball) ; en hausse = gain.")


if __name__ == "__main__":
    os.makedirs(f"{D}/eval", exist_ok=True); main()
