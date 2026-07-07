#!/usr/bin/env python3
"""
compare_methods.py -- Compare nnU-Net vs MedFormer sur la classe ICH (54 cas de test).

Comparaison EQUITABLE : les deux dans le MEME espace (grille 1 mm), meme GT, meme Dice.
- MedFormer : Dice deja calcule en 1 mm (eval/test_metrics.csv).
- nnU-Net   : predit en NATIF (multi-classe) -> on rebascule sa prediction ICH sur la
              grille 1 mm de chaque cas (NearestNeighbor) -> Dice vs le meme GT 1 mm.

Sortie : un CSV par cas + un resume (Dice median) pour :
  MedFormer | nnU-Net ensemble 5-fold | nnU-Net fold 0 (1-contre-1).
"""
import argparse
import csv
import os

import numpy as np
import SimpleITK as sitk


def resample_label_to_ref(lab, ref):
    r = sitk.Image(ref.GetSize(), lab.GetPixelIDValue())
    r.SetSpacing(ref.GetSpacing()); r.SetOrigin(ref.GetOrigin()); r.SetDirection(ref.GetDirection())
    return sitk.Resample(lab, r, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor)


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else 2.0 * int(np.logical_and(a, b).sum()) / s


def read_ids(path):
    with open(path, newline="") as f:
        return [r[[c for c in r if "BDMAP" in c][0]].strip() for r in csv.DictReader(f)]


def read_medformer(csv_path):
    d = {}
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path)):
            d[r["id"]] = float(r["dice"])
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default="/home/pauldcrm/links/scratch/R-SuperCerv/splits/test_ids.csv")
    p.add_argument("--dir_1mm", default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_1mm",
                   help="Grille 1 mm de reference (<ID>.nii.gz) + GT (<ID>/ich_lesion.nii.gz).")
    p.add_argument("--nnunet_pred", default="/home/pauldcrm/links/scratch/R-SuperCerv/nnUNet_predictions")
    p.add_argument("--medformer_csv", default="/home/pauldcrm/links/scratch/R-SuperCerv/eval/test_metrics.csv")
    p.add_argument("--out", default="/home/pauldcrm/links/scratch/R-SuperCerv/eval/methods_comparison.csv")
    p.add_argument("--ich_label", type=int, default=1)
    args = p.parse_args()

    ids = read_ids(args.ids)
    med = read_medformer(args.medformer_csv)
    rows = []
    for cid in ids:
        ref_p = os.path.join(args.dir_1mm, cid + ".nii.gz")
        gt_p = os.path.join(args.dir_1mm, cid, "ich_lesion.nii.gz")
        if not (os.path.exists(ref_p) and os.path.exists(gt_p)):
            continue
        ref = sitk.ReadImage(ref_p)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_p)) > 0

        row = {"id": cid, "medformer": med.get(cid, np.nan)}
        for tag, sub in (("nnunet_ens", "ensemble"), ("nnunet_fold0", "fold0")):
            pth = os.path.join(args.nnunet_pred, sub, cid + ".nii.gz")
            if not os.path.exists(pth):
                row[tag] = np.nan; continue
            native = sitk.ReadImage(pth)
            ich = sitk.BinaryThreshold(native, args.ich_label, args.ich_label, 1, 0)  # classe ICH
            ich_1mm = sitk.GetArrayFromImage(resample_label_to_ref(ich, ref)) > 0
            row[tag] = round(dice(ich_1mm, gt), 4)
        rows.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "medformer", "nnunet_ens", "nnunet_fold0"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n=== COMPARAISON ICH sur {len(rows)} cas de test (Dice, espace 1 mm) ===")
    for k, lab in (("medformer", "MedFormer (unique)"),
                   ("nnunet_fold0", "nnU-Net fold 0 (1-contre-1)"),
                   ("nnunet_ens", "nnU-Net ensemble 5-fold")):
        v = np.array([r[k] for r in rows if not np.isnan(r.get(k, np.nan))], float)
        if len(v):
            print(f"  {lab:30s} : Dice median={np.median(v):.3f}  moyen={v.mean():.3f}  (n={len(v)})")
    print(f"\nCSV -> {args.out}")


if __name__ == "__main__":
    main()
