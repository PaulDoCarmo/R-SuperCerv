#!/usr/bin/env python3
"""Cree des sous-ensembles EMBOITES du pool trainval masques (S_25 ⊂ S_50 ⊂ S_100 ⊂ ...),
en vues symlinks (npz + _gt.npz + list/), pour la sweep 'nombre de masques'.

Emboites = memes cas au fur et a mesure qu'on augmente X -> comparaison controlee.
Melange stratifie par volume ICH (via manifest) pour que chaque S_X couvre la distribution.
"""
import argparse, os, glob, csv, random
import yaml
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trainval_dir", required=True, help="dataset_ich_full_npz_trainval (pool)")
    p.add_argument("--manifest", required=True, help="manifest_ich.csv (pour stratifier par volume)")
    p.add_argument("--out_root", required=True, help="racine des sous-ensembles ($D/subsets)")
    p.add_argument("--sizes", type=int, nargs="+", required=True, help="ex: 25 50 100")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    stems = sorted(os.path.basename(f)[:-len("_gt.npz")]
                   for f in glob.glob(f"{a.trainval_dir}/*_gt.npz"))
    # ordre stratifie par volume : trie par volume, melange par bins -> ordre emboite representatif
    man = pd.read_csv(a.manifest)
    vcol = "ich_volume_mm3" if "ich_volume_mm3" in man.columns else man.columns[-1]
    vol = {r["id"]: float(r[vcol]) for _, r in man.iterrows()} if "id" in man.columns else {}
    order = sorted(stems, key=lambda s: vol.get(s, 0.0))          # tri par volume croissant
    rng = random.Random(a.seed)
    # interleave par terciles pour un ordre "emboite" qui couvre petit/moyen/gros a chaque X
    n = len(order); thirds = [order[:n//3], order[n//3:2*n//3], order[2*n//3:]]
    for t in thirds: rng.shuffle(t)
    interleaved = [x for trip in zip(*[iter_pad(t, n) for t in thirds]) for x in trip if x]
    seen, nested = set(), []
    for x in interleaved:
        if x not in seen:
            seen.add(x); nested.append(x)
    for s in stems:                                              # securite : ajoute les oublies
        if s not in seen: nested.append(s); seen.add(s)

    for N in sorted(a.sizes):
        sub = nested[:N]
        d = f"{a.out_root}/S_{N}"; os.makedirs(d, exist_ok=True)
        # list/ SPECIFIQUE au sous-ensemble : dataset.yaml = SEULEMENT les X cas (sinon
        # dataset_ich.py lit les 305 cas du trainval et cherche des npz absents -> crash),
        # + label_names.yaml (identique, symlink).
        os.makedirs(f"{d}/list", exist_ok=True)
        with open(f"{d}/list/dataset.yaml", "w") as fh:
            yaml.safe_dump(sorted(sub), fh)
        lnsrc = os.path.join(a.trainval_dir, "list", "label_names.yaml")
        if os.path.exists(lnsrc) and not os.path.exists(f"{d}/list/label_names.yaml"):
            os.symlink(os.path.realpath(lnsrc), f"{d}/list/label_names.yaml")
        n_link = 0
        for cid in sub:
            for suf in (".npz", "_gt.npz"):
                src = os.path.join(a.trainval_dir, cid + suf)
                dst = os.path.join(d, cid + suf)
                if os.path.exists(src) and not os.path.exists(dst):
                    os.symlink(os.path.realpath(src), dst); n_link += 1
        with open(f"{d}/ids.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["BDMAP_ID"]); [w.writerow([c]) for c in sub]
        print(f"S_{N}: {len(sub)} cas ({n_link} symlinks) -> {d}")
    # verif emboitement
    sets = {N: set(nested[:N]) for N in sorted(a.sizes)}
    ok = all(sets[a] <= sets[b] for a, b in zip(sorted(a.sizes), sorted(a.sizes)[1:]))
    print(f"emboitement S_x ⊂ S_y : {'OK' if ok else 'CASSE'}")


def iter_pad(lst, n):
    return list(lst) + [None] * (n - len(lst))


if __name__ == "__main__":
    main()
