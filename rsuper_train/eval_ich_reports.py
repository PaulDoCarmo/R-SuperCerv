#!/usr/bin/env python3
"""
eval_ich_reports.py -- Metriques de DETECTION / LOCALISATION sur le set de TEST RAPPORTS.

Les cas-rapports n'ont PAS de masque per-voxel -> pas de Dice. La verite vient des
METADONNEES du rapport (report_to_rsuper_metadata.py -> ich_per_tumor_metadata.csv) :
la/les REGION(S) ou le rapport situe l'ICH (colonne 'Standardized Location', separateur ' / ',
'UNMAPPED' exclu). Le masque de chaque region = le canal-structure correspondant du GT UFO
(les 12 structures TotalSegmentator du npz-rapports, label_names_ich_organs.yaml).

Pour chaque cas de --ids :
  - inference sliding-window -> proba -> seuil -> masque LESION predit (canal ich_lesion, 13cls) ;
  - detection CAS : volume lesion predit >= --detect_min_ml (sensibilite) ;
  - LOCALISATION : pour chaque structure S, la lesion predite deborde-t-elle >= --min_overlap_ml
    dans S -> "regions predites" ; comparaison aux regions GT (metadata) -> TP/FP/FN region.

Sorties : CSV par cas + resume (sensibilite cas ; recall/precision/F1 region ; volumes).
S'utilise pour comparer STAGE 1 vs STAGE 2 : c'est la ou la supervision-rapports doit aider.
Lancer depuis rsuper_train/.
"""
import argparse
import csv
import os
from types import SimpleNamespace

import numpy as np
import torch
import yaml
import pandas as pd

from eval_ich import build_args, load_model, read_ids   # reutilise la machinerie du stage 1


def parse_args():
    p = argparse.ArgumentParser(description="Detection/localisation ICH sur le test-rapports (via metadata).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--load", required=True, help="Checkpoint .pth (stage 1 ou stage 2).")
    p.add_argument("--ufo_npz_dir", required=True, help="dataset_ich_reports_npz (CT + 12 structures GT).")
    p.add_argument("--ids", required=True, help="CSV du split test rapports (col BDMAP_ID).")
    p.add_argument("--reports", required=True, help="ich_per_tumor_metadata.csv.")
    p.add_argument("--config", default="config/ich_ufo/medformer_3d.yaml")
    p.add_argument("--class_list", required=True, help="label_names du MODELE (13cls) -> canal lesion.")
    p.add_argument("--ufo_class_list", required=True, help="label_names_ich_organs.yaml (12) -> canaux structures GT UFO.")
    p.add_argument("--save_csv", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--threshold", type=float, default=0.5, help="Seuil sigmoid sur la proba lesion.")
    p.add_argument("--detect_min_ml", type=float, default=0.5, help="Volume lesion mini (mL) pour 'cas detecte'.")
    p.add_argument("--min_overlap_ml", type=float, default=0.1,
                   help="Chevauchement lesion∩structure mini (mL) pour compter la lesion 'dans' cette region.")
    p.add_argument("--no_ema", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from inference.inference3d import inference_sliding_window

    class_list = sorted(yaml.safe_load(open(args.class_list)))
    lesion_idx = [i for i, c in enumerate(class_list) if "lesion" in c.lower()]
    assert lesion_idx, f"pas de classe 'lesion' dans {class_list}"
    ich_idx = lesion_idx[0]
    ufo_classes = sorted(yaml.safe_load(open(args.ufo_class_list)))
    ch_of = {c: i for i, c in enumerate(ufo_classes)}   # nom structure -> canal du GT UFO
    print(f"Modele: canal lesion={ich_idx} ({class_list[ich_idx]}) | structures UFO={len(ufo_classes)}")

    margs = build_args(args.config, class_list)
    net = load_model(margs, class_list, args.load, use_ema=not args.no_ema)

    meta = pd.read_csv(args.reports)
    if "BDMAP ID" in meta.columns:
        meta = meta.rename(columns={"BDMAP ID": "BDMAP_ID"})

    def gt_regions(cid):
        sub = meta[meta["BDMAP_ID"] == cid]
        regs, unmapped = set(), 0
        for loc in sub["Standardized Location"].dropna():
            for r in str(loc).split(" / "):
                r = r.strip()
                if r in ch_of:
                    regs.add(r)
                elif r and r.lower() not in ("nan", "u"):
                    unmapped += 1
        return regs, unmapped

    def report_ml(cid):
        sub = meta[meta["BDMAP_ID"] == cid]
        v = pd.to_numeric(sub.get("volume_ml", pd.Series(dtype=float)), errors="coerce").sum()
        return float(v)

    ids = read_ids(args.ids)
    print(f"Cas test-rapports : {len(ids)}")

    rows, TP, FP, FN = [], 0, 0, 0
    n_case, n_det, n_with_regions, n_region_evaluable = 0, 0, 0, 0
    for k, cid in enumerate(ids, 1):
        img_p = os.path.join(args.ufo_npz_dir, cid + ".npz")
        gt_p = os.path.join(args.ufo_npz_dir, cid + "_gt.npz")
        if not (os.path.exists(img_p) and os.path.exists(gt_p)):
            print(f"  [skip] npz manquant {cid}"); continue
        if cid not in set(meta["BDMAP_ID"]):
            print(f"  [skip] pas de metadata (aucune lesion ICH extraite) {cid}"); continue

        img = np.load(img_p)["arr_0"].astype(np.float32)
        ufo_gt = np.load(gt_p)["arr_0"]
        if ufo_gt.shape[0] != len(ufo_classes):
            ufo_gt = np.unpackbits(ufo_gt, axis=0)[:len(ufo_classes)]

        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda().float()
        prob = inference_sliding_window(net, t, margs)
        pred = prob[0, ich_idx].numpy() > args.threshold           # (Z,Y,X) bool
        pred_ml = float(pred.sum()) / 1000.0
        case_detected = int(pred_ml >= args.detect_min_ml)

        regs, n_unmapped = gt_regions(cid)
        # regions predites : structures ou la lesion predite deborde >= min_overlap_ml
        pred_regs = set()
        for r, ch in ch_of.items():
            ov = float((pred & (ufo_gt[ch] > 0)).sum()) / 1000.0
            if ov >= args.min_overlap_ml:
                pred_regs.add(r)

        n_case += 1; n_det += case_detected
        tp = fp = fn = None
        if regs:  # localisation evaluable seulement si >=1 region GT mappee
            n_with_regions += 1
            tp = len(regs & pred_regs); fp = len(pred_regs - regs); fn = len(regs - pred_regs)
            TP += tp; FP += fp; FN += fn
            n_region_evaluable += len(regs)

        rows.append(dict(id=cid, case_detected=case_detected, pred_ml=round(pred_ml, 2),
                         report_ml=round(report_ml(cid), 2),
                         gt_regions="|".join(sorted(regs)) if regs else "",
                         n_gt_regions=len(regs), n_unmapped=n_unmapped,
                         pred_regions="|".join(sorted(pred_regs)) if pred_regs else "",
                         region_tp=tp, region_fp=fp, region_fn=fn))
        if k % 5 == 0 or k == len(ids):
            print(f"  ... {k}/{len(ids)}  {cid}: det={case_detected} pred={pred_ml:.1f}mL "
                  f"GT={sorted(regs)} pred_reg={sorted(pred_regs)}", flush=True)

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    fields = ["id", "case_detected", "pred_ml", "report_ml", "gt_regions", "n_gt_regions",
              "n_unmapped", "pred_regions", "region_tp", "region_fp", "region_fn"]
    with open(args.save_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow(r)

    sens = n_det / n_case if n_case else float("nan")
    rec = TP / (TP + FN) if (TP + FN) else float("nan")
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and prec + rec) else float("nan")
    # correlation volumes (pred vs report ABC/2)
    pm = np.array([r["pred_ml"] for r in rows], float)
    rm = np.array([r["report_ml"] for r in rows], float)
    mask = np.isfinite(pm) & np.isfinite(rm) & (rm > 0)
    corr = float(np.corrcoef(pm[mask], rm[mask])[0, 1]) if mask.sum() > 2 else float("nan")

    print("\n========== RESUME DETECTION TEST-RAPPORTS (n=%d) ==========" % n_case)
    print(f"Sensibilite CAS (lesion predite >= {args.detect_min_ml} mL) : {n_det}/{n_case} = {100*sens:.0f}%")
    print(f"LOCALISATION region (chevauchement >= {args.min_overlap_ml} mL), sur {n_with_regions} cas / {n_region_evaluable} regions GT :")
    print(f"   recall={rec:.3f}  precision={prec:.3f}  F1={f1:.3f}   (TP={TP} FP={FP} FN={FN})")
    print(f"Volume predit vs rapport (ABC/2) : correlation r={corr:.2f}  (n={int(mask.sum())})")
    print(f"\nCSV par cas -> {args.save_csv}")


if __name__ == "__main__":
    main()
