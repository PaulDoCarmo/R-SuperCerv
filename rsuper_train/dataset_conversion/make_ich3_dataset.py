#!/usr/bin/env python3
"""Etend un dataset npz existant (N classes) en ajoutant K canaux lesion, SANS re-tourner
tout le pipeline de resampling. Utilise pour passer du dataset ICH-only (13 classes :
12 structures + ich_lesion) au dataset ICH+IVH+PHE (15 classes) en reutilisant les _gt.npz
existants + en extrayant les labels supplementaires depuis les masques bruts multi-classe.

Principe (aligne EXACTEMENT sur la grille des _gt.npz existants) :
  1. On lit chaque <ID>_gt.npz existant  -> (N, Z, Y, X), canaux dans l'ordre trie de label_names.
  2. On resample le masque brut multi-classe sur la grille 1mm du cas (fichier de reference
     <masks_1mm>/<ID>/<ref_channel>.nii.gz, ex ich_lesion.nii.gz), plus proche voisin.
  3. On applique le MEME padding que nii_to_npz_ich.pad (chaque axe -> >=128) pour retomber
     sur la grille (post-pad) des _gt.npz.
  4. On reconstruit un _gt.npz (N+K, Z, Y, X) : dict nom->canal, empile dans l'ordre trie
     des nouveaux label_names (les indices sont deduits du tri, jamais codes en dur).
  5. On (re)cree les dossiers trainval + subsets par symlinks vers le nouveau dataset.

Reproductible sur d'autres donnees : passer les chemins et --add_labels.
Exemple (celui utilise pour ce projet) :
  python make_ich3_dataset.py \
    --src_npz  $D/dataset_ich_full_npz \
    --masks_1mm $D/dataset_ich_full_1mm \
    --raw_masks $D/data_laurent/NIFTI/masks \
    --out      $D/dataset_ich3_full_npz \
    --add_labels 2:ivh_lesion,3:phe_lesion \
    --trainval_ids $D/splits/trainval_ids.csv --trainval_out $D/dataset_ich3_full_npz_trainval \
    --subset_src $D/subsets --subset_out $D/subsets3 --subset_sizes 25,50,100
"""
import argparse, glob, math, os, shutil
import numpy as np, yaml, SimpleITK as sitk


def pad3d(a):
    """Meme padding que nii_to_npz_ich.pad : chaque axe spatial -> >=128, symetrique."""
    z, y, x = a.shape
    if z < 128:
        d = int(math.ceil((128. - z) / 2)); a = np.pad(a, ((d, d), (0, 0), (0, 0)))
    if y < 128:
        d = int(math.ceil((128. - y) / 2)); a = np.pad(a, ((0, 0), (d, d), (0, 0)))
    if x < 128:
        d = int(math.ceil((128. - x) / 2)); a = np.pad(a, ((0, 0), (0, 0), (d, d)))
    return a


def parse_add(spec):
    out = []
    for tok in spec.split(","):
        lab, name = tok.split(":"); out.append((int(lab.strip()), name.strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_npz", required=True, help="Dataset npz existant (contient <ID>.npz + <ID>_gt.npz + list/label_names.yaml).")
    ap.add_argument("--masks_1mm", required=True, help="Dossier <ID>/<channel>.nii.gz (grille 1mm de reference, pre-pad).")
    ap.add_argument("--raw_masks", required=True, help="Masques bruts multi-classe <ID>.nii.gz.")
    ap.add_argument("--out", required=True, help="Nouveau dataset npz (cree).")
    ap.add_argument("--add_labels", required=True, help="Labels a ajouter : 'label:nom,...' ex '2:ivh_lesion,3:phe_lesion'.")
    ap.add_argument("--ref_channel", default="ich_lesion", help="Canal nifti servant de reference geometrique dans --masks_1mm/<ID>/.")
    ap.add_argument("--trainval_ids", default=None); ap.add_argument("--trainval_out", default=None)
    ap.add_argument("--subset_src", default=None); ap.add_argument("--subset_out", default=None)
    ap.add_argument("--subset_sizes", default="25,50,100")
    a = ap.parse_args()

    old_names = yaml.safe_load(open(f"{a.src_npz}/list/label_names.yaml"))
    add = parse_add(a.add_labels)
    new_names = sorted(old_names + [nm for _, nm in add])
    os.makedirs(f"{a.out}/list", exist_ok=True)
    yaml.safe_dump(new_names, open(f"{a.out}/list/label_names.yaml", "w"))
    print(f"{len(old_names)} -> {len(new_names)} classes ; ajout {add}")

    ids = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{a.src_npz}/*_gt.npz"))
    ok, err = 0, []
    for i, cid in enumerate(ids, 1):
        try:
            old_gt = np.load(f"{a.src_npz}/{cid}_gt.npz")["arr_0"]           # (N,Z,Y,X) ordre old_names
            ref = sitk.ReadImage(f"{a.masks_1mm}/{cid}/{a.ref_channel}.nii.gz")
            src = sitk.ReadImage(f"{a.raw_masks}/{cid}.nii.gz")
            s1 = sitk.GetArrayFromImage(sitk.Resample(src, ref, sitk.Transform(),
                                                      sitk.sitkNearestNeighbor, 0, src.GetPixelID()))
            chan = {nm: old_gt[j] for j, nm in enumerate(old_names)}
            for label, name in add:
                chan[name] = pad3d((s1 == label).astype(old_gt.dtype))
            new_gt = np.stack([chan[nm] for nm in new_names], axis=0)         # ordre trie -> indices deduits
            assert new_gt.shape[1:] == old_gt.shape[1:], f"{cid} {new_gt.shape} vs {old_gt.shape}"
            np.savez_compressed(f"{a.out}/{cid}_gt.npz", new_gt)
            img = f"{a.out}/{cid}.npz"
            if not os.path.lexists(img):
                os.symlink(f"{a.src_npz}/{cid}.npz", img)
            ok += 1
        except Exception as e:
            err.append(cid); print("ERR", cid, repr(e)[:90], flush=True)
        if i % 80 == 0:
            print(f"  {i}/{len(ids)} (ok={ok})", flush=True)
    yaml.safe_dump(ids, open(f"{a.out}/list/dataset.yaml", "w"))
    print(f"GT: {ok}/{len(ids)} ok, {len(err)} erreurs")

    def mkds(dst, cids):
        os.makedirs(f"{dst}/list", exist_ok=True)
        cids = [c for c in cids if os.path.exists(f"{a.out}/{c}_gt.npz")]
        yaml.safe_dump(sorted(cids), open(f"{dst}/list/dataset.yaml", "w"))
        shutil.copy(f"{a.out}/list/label_names.yaml", f"{dst}/list/label_names.yaml")
        for cid in cids:
            for suf in (".npz", "_gt.npz"):
                d = f"{dst}/{cid}{suf}"
                if not os.path.lexists(d):
                    os.symlink(f"{a.out}/{cid}{suf}", d)
        return len(cids)

    if a.trainval_ids and a.trainval_out:
        tv = [l.strip() for l in open(a.trainval_ids) if l.strip().startswith("ID_")]
        print("trainval:", mkds(a.trainval_out, tv))
    if a.subset_src and a.subset_out:
        for X in a.subset_sizes.split(","):
            sub = yaml.safe_load(open(f"{a.subset_src}/S_{X.strip()}/list/dataset.yaml"))
            print(f"S_{X.strip()}:", mkds(f"{a.subset_out}/S_{X.strip()}", sub))
    print("DONE")


if __name__ == "__main__":
    main()
