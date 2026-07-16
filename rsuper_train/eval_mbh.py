#!/usr/bin/env python3
"""Eval DETECTION sur MBH (classification ICH, PAS de masque voxel). Contrairement aux
rapports (ICH toujours present), MBH a des cas SAINS -> on mesure enfin la SPECIFICITE
(le modele hallucine-t-il des lesions sur cerveau sain ?), + la sensibilite par sous-type.
Notre modele cible l'ICH=intraparenchymateux -> il devrait rater l'extra-axial (SAH/SDH/EDH).

Pour chaque cas : inference -> volume lesion predit (mL) -> 'detecte' si >= --detect_min_ml.
Sorties : CSV par cas + resume (sensibilite/specificite/AUC). Lancer depuis rsuper_train/."""
import argparse, csv, os
import numpy as np, torch, yaml, pandas as pd
from eval_ich import build_args, load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--load", required=True)
    p.add_argument("--npz_dir", required=True, help="mbh/npz (image seule)")
    p.add_argument("--labels", required=True, help="mbh_labels.csv")
    p.add_argument("--config", default="config/ich/medformer_3d.yaml")
    p.add_argument("--class_list", required=True)
    p.add_argument("--save_csv", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--detect_min_ml", type=float, default=0.5)
    p.add_argument("--no_ema", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    from inference.inference3d import inference_sliding_window

    class_list = sorted(yaml.safe_load(open(a.class_list)))
    ich_idx = [i for i, c in enumerate(class_list) if "lesion" in c.lower()][0]
    margs = build_args(a.config, class_list)
    net = load_model(margs, class_list, a.load, use_ema=not a.no_ema)

    lab = pd.read_csv(a.labels)
    lab = lab[lab["leak"] == 0].reset_index(drop=True)   # hors fuite
    subs = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
    rows = []
    for i in range(len(lab)):
        r = lab.iloc[i]; cid = r["id"]; p = os.path.join(a.npz_dir, cid + ".npz")
        if not os.path.exists(p):
            continue
        img = np.load(p)["arr_0"].astype(np.float32)
        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda().float()
        prob = inference_sliding_window(net, t, margs)
        pred_ml = float((prob[0, ich_idx].numpy() > a.threshold).sum()) / 1000.0
        rows.append(dict(id=cid, healthy=int(r["healthy"]), positive=int(r["any"]),
                         **{s: int(r[s]) for s in subs},
                         pred_ml=round(pred_ml, 2), detected=int(pred_ml >= a.detect_min_ml)))
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(lab)}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.save_csv), exist_ok=True)
    df.to_csv(a.save_csv, index=False)

    pos = df[df.positive == 1]; neg = df[df.healthy == 1]
    iph = df[df.intraparenchymal == 1]
    extra = df[(df.positive == 1) & (df.intraparenchymal == 0)]  # extra-axial pur (pas d'IPH)
    print("\n========== RESUME MBH (n=%d, hors fuite) ==========" % len(df))
    print(f"Sensibilite globale (any ICH, n={len(pos)}) : {100*pos.detected.mean():.1f}%")
    print(f"SPECIFICITE (cas SAINS, n={len(neg)})        : {100*(1-neg.detected.mean()):.1f}%   "
          f"(faux positifs : {int(neg.detected.sum())}/{len(neg)})")
    print(f"Sensibilite IPH (intraparenchymateux, n={len(iph)}) : {100*iph.detected.mean():.1f}%")
    print(f"Sensibilite EXTRA-AXIAL pur (SAH/SDH/EDH/IVH sans IPH, n={len(extra)}) : {100*extra.detected.mean():.1f}%  "
          f"(attendu bas : modele IPH-specifique)")
    for s in subs:
        sub = df[df[s] == 1]
        if len(sub): print(f"    sens {s} (n={len(sub)}) : {100*sub.detected.mean():.1f}%")
    try:
        from sklearn.metrics import roc_auc_score
        print(f"AUC (volume predit vs any ICH) : {roc_auc_score(df.positive, df.pred_ml):.3f}")
    except Exception as e:
        print("AUC indispo:", e)
    print(f"\nCSV -> {a.save_csv}")


if __name__ == "__main__":
    main()
