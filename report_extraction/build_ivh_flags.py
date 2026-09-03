#!/usr/bin/env python3
"""Construit ivh_flags.csv (BDMAP_ID, y_ivh) depuis un CSV formaté prompt6.

y_ivh = 1 si le scan (ID) a >=1 lésion dont le type contient 'IVH', sinon 0. C'est exactement
le flag rapport par-scan consommé par le training (--ivh_flags) et par l'éval détection (vérité).
Aucun script du repo ne l'écrivait ; validé identique à l'ivh_flags.csv existant (0 désaccord/183).
"""
import argparse, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV formaté prompt6 (colonnes ID, type, ...)")
    ap.add_argument("--output", required=True, help="Chemin de sortie ivh_flags.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.input)
    y = (d.groupby("ID")["type"]
           .apply(lambda s: int(s.astype(str).str.upper().str.contains("IVH").any()))
           .rename("y_ivh").reset_index().rename(columns={"ID": "BDMAP_ID"})
           .sort_values("BDMAP_ID"))
    y.to_csv(a.output, index=False)
    print(f"ivh_flags -> {a.output} : {len(y)} scans, dont IVH+ = {int(y.y_ivh.sum())}")


if __name__ == "__main__":
    main()
