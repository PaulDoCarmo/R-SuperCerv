#!/usr/bin/env python3
"""Build reports.csv (ID, Report) from a folder of .txt radiology reports.

Variante de build_reports_csv.py (qui ne lit que du JSON avec champ 'dictation') pour les
rapports fournis en .txt (1 fichier par cas, nom = <ID>.txt, ex. 013F49AD_0.txt).

Particularites du corpus local : encodage UTF-8 **avec BOM** + fins de ligne CRLF -> on lit en
'utf-8-sig' (retire le BOM) et on normalise les CRLF, sinon le BOM corrompt le 1er champ.

On garde TOUS les cas (baseline _0 ET suivi _1) : le filtrage _0 se fera A L'ENTRAINEMENT.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def derive_id(p: Path) -> str:
    name = p.name
    for ext in (".txt",):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description="Build reports.csv from .txt reports.")
    ap.add_argument("txt_folder", help="Dossier des rapports .txt (<ID>.txt)")
    ap.add_argument("output_folder", help="Dossier de sortie du CSV")
    ap.add_argument("--output-name", default="reports.csv")
    args = ap.parse_args()

    txt_folder = Path(args.txt_folder)
    out_folder = Path(args.output_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / args.output_name

    rows, skipped = [], 0
    for txt in sorted(txt_folder.glob("*.txt")):
        try:
            text = txt.read_text(encoding="utf-8-sig")  # retire le BOM
        except OSError:
            skipped += 1
            continue
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            skipped += 1
            continue
        rows.append({"ID": derive_id(txt), "Report": text})

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ID", "Report"])
        w.writeheader()
        w.writerows(rows)

    print(f"reports.csv ecrit : {out_path}")
    print(f"  cas ecrits : {len(rows)} | vides/illisibles sautes : {skipped}")
    n0 = sum(1 for r in rows if r["ID"].endswith("_0"))
    n1 = sum(1 for r in rows if r["ID"].endswith("_1"))
    print(f"  dont baseline _0 : {n0} | suivi _1 : {n1} (filtrage _0 = a l'entrainement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
