#!/usr/bin/env python3
"""
nii_to_npz_ich.py -- Etape C-2 du pipeline R-SuperCerv.

Convertit le dataset reechantillonne 1 mm (sortie de resample_ich_3d.py, format
MedFormer-nii) vers le format .npz attendu par l'entrainement MedFormer/R-Super,
en appliquant la normalisation d'intensite adaptee au CT cranien :

    1) fenetre cerveau/sang : clip des HU a [hu_min, hu_max] (defaut [0, 100]) ;
    2) z-score : (x - mean) / std sur tout le volume ;
    3) padding a >= 128 voxels par axe (taille de patch d'entrainement) ;
    4) sauvegarde :
         <ID>.npz       -> image float32, shape (Z, Y, X)
         <ID>_gt.npz    -> labels  int8,   shape (C, Z, Y, X)  (C = nb de classes)

Remplace nii2npz.py de R-Super. Seule difference de fond : la fenetre HU,
qui passe de l'abdominal [-991, 500] au cranien [0, 100].

Entree :
    src/<ID>.nii.gz                 (CT 1 mm)
    src/<ID>/<label>.nii.gz         (labels 1 mm)
    src/list/label_names.yaml       (liste des classes)
Sortie :
    tgt/<ID>.npz, tgt/<ID>_gt.npz, tgt/list/{dataset,label_names}.yaml
"""

import argparse
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import yaml
import SimpleITK as sitk
from tqdm import tqdm


def pad(img, lab):
    """Pad symetriquement chaque axe spatial a au moins 128 voxels."""
    z, y, x = img.shape
    if z < 128:
        d = int(math.ceil((128. - z) / 2))
        img = np.pad(img, ((d, d), (0, 0), (0, 0)))
        lab = np.pad(lab, ((0, 0), (d, d), (0, 0), (0, 0)))
    if y < 128:
        d = int(math.ceil((128. - y) / 2))
        img = np.pad(img, ((0, 0), (d, d), (0, 0)))
        lab = np.pad(lab, ((0, 0), (0, 0), (d, d), (0, 0)))
    if x < 128:
        d = int(math.ceil((128. - x) / 2))
        img = np.pad(img, ((0, 0), (0, 0), (d, d)))
        lab = np.pad(lab, ((0, 0), (0, 0), (0, 0), (d, d)))
    return img, lab


def process_file(name, source_path, target_path, lab_name_list, hu_min, hu_max):
    """Traite un cas : CT -> fenetre+zscore, labels -> multi-canal, sauvegarde npz."""
    try:
        base = name.replace(".nii.gz", "")

        # --- CT ---
        img = sitk.ReadImage(os.path.join(source_path, name))
        img = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)

        # --- Labels (un canal par classe) ---
        lab = []
        create_bkg = False
        bkg_index = None
        for i, cls in enumerate(sorted(lab_name_list)):
            if cls == "background":
                p_bkg = os.path.join(source_path, base, "background.nii.gz")
                if not os.path.exists(p_bkg):
                    create_bkg = True
                    bkg_index = i
                    continue
            p = os.path.join(source_path, base, f"{cls}.nii.gz")
            item = sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.int8)
            lab.append(item)
        try:
            lab = np.stack(lab, axis=0)  # (C, Z, Y, X)
        except Exception:
            print(f"[error] stack labels {name}")
            return None
        if create_bkg:
            background = 1 - np.sum(lab, axis=0)
            lab = np.insert(lab, bkg_index, background, axis=0)

        # --- Fenetre HU (cerveau/sang) ---
        img = np.clip(img, hu_min, hu_max)

        # --- z-score ---
        mean = float(np.mean(img))
        std = float(np.std(img))
        std = std if std > 1e-8 else 1.0
        img = (img - mean) / std

        # --- padding + sauvegarde ---
        img, lab = pad(img, lab)
        img, lab = img.astype(np.float32), lab.astype(np.int8)
        np.savez_compressed(os.path.join(target_path, f"{base}.npz"), img)
        np.savez_compressed(os.path.join(target_path, f"{base}_gt.npz"), lab)
        return name
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {name}: {exc!r}")
        return None


def main():
    p = argparse.ArgumentParser(
        description="Convertit le dataset ICH 1mm (nii) en npz, avec fenetre cerveau + z-score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src_path", default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_1mm",
                   help="Dataset reechantillonne 1mm (sortie de resample_ich_3d.py).")
    p.add_argument("--tgt_path", default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_npz",
                   help="Destination des .npz.")
    p.add_argument("--hu_min", type=float, default=0.0, help="Borne basse de la fenetre HU.")
    p.add_argument("--hu_max", type=float, default=100.0, help="Borne haute de la fenetre HU.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--parts", type=int, default=1, help="Decoupage en parties (jobs paralleles).")
    p.add_argument("--current_part", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src, tgt = args.src_path, args.tgt_path
    os.makedirs(tgt, exist_ok=True)

    with open(os.path.join(src, "list", "label_names.yaml"), encoding="utf-8") as f:
        lab_name_list = yaml.safe_load(f)
    print(f"Classes ({len(lab_name_list)}) : {lab_name_list}")
    print(f"Fenetre HU : [{args.hu_min}, {args.hu_max}]  +  z-score")

    names = sorted(n for n in os.listdir(src) if n.endswith(".nii.gz"))
    if not args.overwrite:
        names = [n for n in names
                 if not (os.path.exists(os.path.join(tgt, n.replace('.nii.gz', '') + ".npz"))
                         and os.path.exists(os.path.join(tgt, n.replace('.nii.gz', '') + "_gt.npz")))]
    if args.parts > 1:
        names = np.array_split(names, args.parts)[args.current_part].tolist()
    print(f"Cas a convertir : {len(names)}")

    worker = partial(process_file, source_path=src, target_path=tgt,
                     lab_name_list=lab_name_list, hu_min=args.hu_min, hu_max=args.hu_max)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(worker, names), total=len(names), desc="nii->npz"))

    os.makedirs(os.path.join(tgt, "list"), exist_ok=True)
    for y in ("dataset.yaml", "label_names.yaml"):
        srcy = os.path.join(src, "list", y)
        if os.path.exists(srcy):
            shutil.copy(srcy, os.path.join(tgt, "list", y))
    print("Termine.")


if __name__ == "__main__":
    main()
