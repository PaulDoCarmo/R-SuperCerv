#!/usr/bin/env python3
"""
eval_ich.py -- Evaluation du stage 1 (segmentation ICH) sur le set de TEST held-out.

Travaille directement en espace npz 1 mm (prediction et GT sur la MEME grille
isotrope 1x1x1 -> Dice/NSD corrects, pas de reprojection necessaire).

Pour chaque cas de --ids :
  - inference sliding-window -> proba sigmoid -> seuil -> masque binaire ICH ;
  - compare au GT (canal ich_lesion) ;
  - metriques : Dice, NSD@tol (surface dice), HD95 (mm), volumes pred/GT (mL), detection.

Sorties : un CSV par cas + un resume (moyenne/mediane).
Les 54 cas du test n'ont JAMAIS ete vus a l'entrainement (voir splits/test_ids.csv).

Lancer depuis rsuper_train/ (imports relatifs model/ inference/ metric/).
"""
import argparse
import csv
import os
from types import SimpleNamespace

import numpy as np
import torch
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Eval ICH (Dice/NSD/HD95) sur le test held-out.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--load", required=True, help="Checkpoint .pth (ex: exp/ich/ich_stage1/fold_0_latest.pth)")
    p.add_argument("--npz_dir", default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_npz",
                   help="Dossier npz (contient img + _gt de tous les cas, dont le test).")
    p.add_argument("--ids", default="/home/pauldcrm/links/scratch/R-SuperCerv/splits/test_ids.csv",
                   help="CSV avec colonne 'BDMAP ID' (cas de test).")
    p.add_argument("--config", default="config/ich/medformer_3d.yaml")
    p.add_argument("--class_list", default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_npz/list/label_names.yaml")
    p.add_argument("--save_csv", default="/home/pauldcrm/links/scratch/R-SuperCerv/eval/test_metrics.csv")
    p.add_argument("--gpu", default="0")
    p.add_argument("--threshold", type=float, default=0.5, help="Seuil sur la proba sigmoid.")
    p.add_argument("--nsd_tol", type=float, default=1.0, help="Tolerance (mm) pour le NSD (surface dice).")
    p.add_argument("--detect_min_ml", type=float, default=0.5, help="Volume mini (mL) pour compter une detection.")
    p.add_argument("--no_ema", action="store_true", help="Utiliser le modele brut au lieu de l'EMA.")
    return p.parse_args()


def build_args(config_path, class_list):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    a = SimpleNamespace(**cfg)
    a.model = "medformer"
    a.dimension = "3d"
    a.classes = len(class_list)
    a.classification_branch = False
    a.clip_loss = False
    a.pretrain = False
    if not hasattr(a, "window_size") or a.window_size is None:
        a.window_size = a.training_size
    return a


def load_model(args, class_list, ckpt_path, use_ema=True):
    from model.utils import get_model
    net = get_model(args, classes=class_list)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = "ema_model_state_dict" if (use_ema and "ema_model_state_dict" in ckpt) else "model_state_dict"
    sd = ckpt[key]
    sd = sd.state_dict() if hasattr(sd, "state_dict") else sd
    missing = net.load_state_dict(sd, strict=False)
    net.cuda().eval()
    print(f"Checkpoint '{key}' charge depuis {ckpt_path} (missing/unexpected ignores si peu nombreux)")
    return net


def dice_score(pred, gt):
    p, g = int(pred.sum()), int(gt.sum())
    if p + g == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (p + g))


def read_ids(path):
    ids = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            col = "BDMAP ID" if "BDMAP ID" in row else ("BDMAP_ID" if "BDMAP_ID" in row else list(row)[0])
            ids.append(row[col].strip())
    return ids


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from inference.inference3d import inference_sliding_window
    from metric.metrics import (compute_surface_distances,
                                compute_surface_dice_at_tolerance,
                                compute_robust_hausdorff)

    with open(args.class_list) as f:
        class_list = sorted(yaml.safe_load(f))
    lesion_idx = [i for i, c in enumerate(class_list) if "lesion" in c.lower()]
    assert lesion_idx, f"Aucune classe 'lesion' dans {class_list}"
    ich_idx = lesion_idx[0]
    print(f"Classes: {class_list} | canal ICH = index {ich_idx}")

    margs = build_args(args.config, class_list)
    net = load_model(margs, class_list, args.load, use_ema=not args.no_ema)

    ids = read_ids(args.ids)
    print(f"Cas de test a evaluer : {len(ids)}")

    spacing = (1.0, 1.0, 1.0)  # npz isotrope 1 mm
    rows = []
    for n, cid in enumerate(ids, 1):
        img_p = os.path.join(args.npz_dir, cid + ".npz")
        gt_p = os.path.join(args.npz_dir, cid + "_gt.npz")
        if not (os.path.exists(img_p) and os.path.exists(gt_p)):
            print(f"  [skip] npz manquant pour {cid}")
            continue

        img = np.load(img_p)["arr_0"].astype(np.float32)          # (Z,Y,X)
        gt_all = np.load(gt_p)["arr_0"]                            # (C,Z,Y,X)
        if gt_all.shape[0] != len(class_list):                    # securite (labels bit-packes)
            gt_all = np.unpackbits(gt_all, axis=0)[:len(class_list)]
        gt = gt_all[ich_idx] > 0                                  # (Z,Y,X) bool

        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda().float()
        prob = inference_sliding_window(net, t, margs)            # (1,C,Z,Y,X) proba
        pred = prob[0, ich_idx].numpy() > args.threshold          # (Z,Y,X) bool

        dice = dice_score(pred, gt)
        if pred.sum() > 0 and gt.sum() > 0:
            sd = compute_surface_distances(gt, pred, spacing)
            nsd = float(compute_surface_dice_at_tolerance(sd, args.nsd_tol))
            hd95 = float(compute_robust_hausdorff(sd, 95))
        elif pred.sum() == 0 and gt.sum() == 0:
            nsd, hd95 = 1.0, 0.0
        else:
            nsd, hd95 = 0.0, float("nan")

        pred_ml = float(pred.sum()) / 1000.0
        gt_ml = float(gt.sum()) / 1000.0
        detected = int(pred_ml >= args.detect_min_ml)
        rows.append(dict(id=cid, dice=round(dice, 4), nsd=round(nsd, 4),
                         hd95_mm=round(hd95, 2), pred_ml=round(pred_ml, 2),
                         gt_ml=round(gt_ml, 2), detected=detected))
        if n % 10 == 0 or n == len(ids):
            print(f"  ... {n}/{len(ids)}  (dernier {cid}: Dice={dice:.3f} NSD={nsd:.3f})", flush=True)

    # --- sauvegarde + resume ---
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    with open(args.save_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "dice", "nsd", "hd95_mm", "pred_ml", "gt_ml", "detected"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    dices = np.array([r["dice"] for r in rows])
    nsds = np.array([r["nsd"] for r in rows])
    hd = np.array([r["hd95_mm"] for r in rows], dtype=float)
    det = np.array([r["detected"] for r in rows])
    print("\n========== RESUME TEST (n=%d) ==========" % len(rows))
    print(f"Dice : mean={dices.mean():.3f}  median={np.median(dices):.3f}  "
          f"min={dices.min():.3f}  max={dices.max():.3f}")
    print(f"NSD@{args.nsd_tol}mm : mean={nsds.mean():.3f}  median={np.median(nsds):.3f}")
    print(f"HD95 (mm) : median={np.nanmedian(hd):.1f}")
    print(f"Detection (>= {args.detect_min_ml} mL) : {det.sum()}/{len(det)} = {100*det.mean():.0f}%")
    print(f"\nCSV par cas -> {args.save_csv}")


if __name__ == "__main__":
    main()
