#!/usr/bin/env python3
"""
make_split.py -- Cree un split train/val/test propre pour l'ICH (stage 1).

- TEST held-out : ~test_frac des cas, JAMAIS vus en entrainement/validation.
  Stratifie par volume d'ICH (terciles) pour couvrir petits/moyens/gros saignements.
- TRAINVAL : le reste. On en fait une "vue" par symlinks (dataset_ich_npz_trainval/)
  pour que le framework R-Super n'y voie QUE ces cas (il fera lui-meme son split
  train/val interne, ~10% en val).

Sorties :
  <npz>_trainval/                 -> symlinks des npz trainval + list/{dataset,label_names}.yaml
  splits/test_ids.csv             -> colonne 'BDMAP ID' (pour predict --ids)
  splits/trainval_ids.csv
  splits/split_summary.txt

Aucune copie lourde (symlinks). Aucun cas n'est duplique entre les splits.
"""
import argparse
import csv
import os
import random

import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Split train/val/test ICH (stratifie par volume).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--npz_dir", default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_npz")
    p.add_argument("--manifest", default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich/manifest_ich.csv",
                   help="manifest_ich.csv (pour les volumes -> stratification).")
    p.add_argument("--trainval_dir", default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_npz_trainval",
                   help="Vue symlink des cas trainval (= --data_root de l'entrainement).")
    p.add_argument("--splits_dir", default="/home/pauldcrm/links/scratch/R-SuperCerv/splits")
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # 1) IDs disponibles (ceux qui ont reellement un npz)
    with open(os.path.join(args.npz_dir, "list", "dataset.yaml")) as f:
        all_ids = yaml.safe_load(f)
    all_ids = [i for i in all_ids
               if os.path.exists(os.path.join(args.npz_dir, i + ".npz"))
               and os.path.exists(os.path.join(args.npz_dir, i + "_gt.npz"))]
    print(f"Cas avec npz : {len(all_ids)}")

    # 2) Volumes depuis le manifest (mm3) pour stratifier
    vol = {}
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            vol[r["id"]] = float(r["ich_volume_mm3"]) if r["ich_volume_mm3"] else 0.0
    vols = np.array([vol.get(i, 0.0) for i in all_ids])

    # 3) Terciles de volume -> 3 strates
    q1, q2 = np.quantile(vols, [1/3, 2/3])
    strata = {0: [], 1: [], 2: []}
    for i, v in zip(all_ids, vols):
        s = 0 if v <= q1 else (1 if v <= q2 else 2)
        strata[s].append(i)

    # 4) Tirage du test dans chaque strate (~test_frac)
    test_ids = []
    for s, ids in strata.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * args.test_frac))
        test_ids += ids[:n_test]
    test_ids = sorted(test_ids)
    trainval_ids = sorted(set(all_ids) - set(test_ids))
    assert not (set(test_ids) & set(trainval_ids))
    print(f"TEST held-out : {len(test_ids)} | TRAINVAL : {len(trainval_ids)} "
          f"(le framework prendra ~{max(1, len(trainval_ids)//10)} en val interne)")

    # 5) Vue symlink trainval
    os.makedirs(os.path.join(args.trainval_dir, "list"), exist_ok=True)
    n_link = 0
    for i in trainval_ids:
        for suff in (".npz", "_gt.npz"):
            src = os.path.realpath(os.path.join(args.npz_dir, i + suff))
            dst = os.path.join(args.trainval_dir, i + suff)
            if not os.path.exists(dst):
                os.symlink(src, dst)
                n_link += 1
    with open(os.path.join(args.trainval_dir, "list", "dataset.yaml"), "w") as f:
        yaml.dump(trainval_ids, f)
    # label_names.yaml : recopie
    import shutil
    shutil.copy(os.path.join(args.npz_dir, "list", "label_names.yaml"),
                os.path.join(args.trainval_dir, "list", "label_names.yaml"))
    print(f"Vue trainval : {args.trainval_dir} ({n_link} symlinks crees)")

    # 6) CSV des splits
    os.makedirs(args.splits_dir, exist_ok=True)
    for name, ids in (("test_ids.csv", test_ids), ("trainval_ids.csv", trainval_ids)):
        with open(os.path.join(args.splits_dir, name), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["BDMAP ID"])
            for i in ids:
                w.writerow([i])

    # 7) Resume volumes par split (mL)
    def stats(ids):
        a = np.array([vol.get(i, 0.0) for i in ids]) / 1000.0
        return f"n={len(a)} vol_mL: median={np.median(a):.1f} mean={a.mean():.1f} min={a.min():.1f} max={a.max():.1f}"
    summary = (f"seed={args.seed} test_frac={args.test_frac}\n"
               f"TEST     : {stats(test_ids)}\n"
               f"TRAINVAL : {stats(trainval_ids)}\n")
    with open(os.path.join(args.splits_dir, "split_summary.txt"), "w") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"CSV -> {args.splits_dir}/test_ids.csv , trainval_ids.csv")


if __name__ == "__main__":
    main()
