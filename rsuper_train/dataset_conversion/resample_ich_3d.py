#!/usr/bin/env python3
"""
resample_ich_3d.py -- Etape C-1 du pipeline R-SuperCerv.

Reechantillonne le dataset ICH (sortie de build_ich_dataset.py) vers une grille
ISOTROPE 1x1x1 mm, et reorganise dans le format intermediaire "MedFormer nii"
attendu par l'etape C-2 (nii_to_npz_ich.py).

Entree (format build_ich_dataset.py) :
    src/
      <ID>/
        ct.nii.gz
        segmentations/<label>.nii.gz      (ex: ich_lesion.nii.gz)

Sortie (format MedFormer nii) :
    tgt/
      <ID>.nii.gz                         (CT reechantillonne 1mm)
      <ID>/<label>.nii.gz                 (labels reechantillonnes 1mm)
      list/dataset.yaml                   (liste des IDs)
      list/label_names.yaml               (liste des classes)

Resampling (fidele a R-Super) :
  - reorientation canonique RAI ;
  - plan XY : interpolation BSpline (image), NearestNeighbor (labels) ;
  - axe Z (anisotrope, ~5 mm chez nous) : NearestNeighbor pour l'image aussi
    -> approche "separate-z" type nnU-Net : on ne LISSE pas le travers-plan,
    on replique les coupes. C'est volontaire (et un bon revelateur des limites
    sur coupes epaisses). Voir --z_bspline pour interpoler le Z en BSpline.

PAS de fenetrage HU ici : la normalisation d'intensite (fenetre cerveau [0,100]
+ z-score) se fait a l'etape C-2.
"""

import argparse
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import yaml
import nibabel as nib
import SimpleITK as sitk

from resample_utils import ResampleXYZAxis, ResampleLabelToRef, reorient_image

sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(8)


# --------------------------------------------------------------------------- #
# Lecture robuste (affines non-orthonormales frequentes en CT clinique)
# --------------------------------------------------------------------------- #
def _fix_nifti_affine(in_path, out_path):
    img = nib.load(in_path)
    data = img.get_fdata(dtype=np.float32)
    aff = img.affine
    U, _, Vt = np.linalg.svd(aff[:3, :3])
    aff = aff.copy()
    aff[:3, :3] = U @ Vt
    nib.Nifti1Image(data, aff, img.header).to_filename(out_path)


def read_sitk_with_nib_fallback(path, tmp_dir="tmp_resample"):
    try:
        return sitk.ReadImage(path)
    except RuntimeError as e:
        if "orthonormal" in str(e).lower():
            os.makedirs(tmp_dir, exist_ok=True)
            fixed = os.path.join(tmp_dir, f"fixed_{uuid.uuid4().hex}.nii.gz")
            _fix_nifti_affine(path, fixed)
            img = sitk.ReadImage(fixed)
            try:
                os.remove(fixed)
            except OSError:
                pass
            return img
        raise


# --------------------------------------------------------------------------- #
# Coeur du resampling
# --------------------------------------------------------------------------- #
def resample_image_and_labels(im_image, im_labels, save_path, name,
                              target_spacing=(1., 1., 1.), z_bspline=False):
    im_image = reorient_image(im_image, "RAI")
    for k in im_labels:
        im_labels[k] = reorient_image(im_labels[k], "RAI")
        assert im_labels[k].GetSize() == im_image.GetSize(), f"size mismatch for {k}"

    spacing = im_image.GetSpacing()

    # 1) plan XY : BSpline pour l'image, en gardant le Z d'origine
    re_img_xy = ResampleXYZAxis(im_image,
                                space=(target_spacing[0], target_spacing[1], spacing[2]),
                                interp=sitk.sitkBSpline)
    re_lab_xy = {k: ResampleLabelToRef(v, re_img_xy, interp=sitk.sitkNearestNeighbor)
                 for k, v in im_labels.items()}

    # 2) axe Z : NearestNeighbor (defaut, separate-z) ou BSpline (--z_bspline)
    z_interp = sitk.sitkBSpline if z_bspline else sitk.sitkNearestNeighbor
    re_img = ResampleXYZAxis(re_img_xy, space=target_spacing, interp=z_interp)
    re_lab = {k: ResampleLabelToRef(v, re_img, interp=sitk.sitkNearestNeighbor)
              for k, v in re_lab_xy.items()}

    os.makedirs(save_path, exist_ok=True)
    sitk.WriteImage(re_img, os.path.join(save_path, f"{name}.nii.gz"))
    os.makedirs(os.path.join(save_path, name), exist_ok=True)
    for k, v in re_lab.items():
        sitk.WriteImage(v, os.path.join(save_path, name, f"{k}.nii.gz"))


def process_case(name, src_path, label_path, tgt_path, lab_name_list,
                 target_spacing, z_bspline, overwrite):
    try:
        out_ct = os.path.join(tgt_path, f"{name}.nii.gz")
        out_dir = os.path.join(tgt_path, name)
        if (os.path.exists(out_ct)
                and all(os.path.exists(os.path.join(out_dir, f"{l}.nii.gz")) for l in lab_name_list)
                and not overwrite):
            return (name, "exists")

        ct = os.path.join(src_path, name, "ct.nii.gz")
        if not os.path.exists(ct):
            ct = os.path.join(src_path, f"{name}.nii.gz")
        itk_img = read_sitk_with_nib_fallback(ct)

        lab_dict = {}
        for lab in lab_name_list:
            p = os.path.join(label_path, name, "segmentations", f"{lab}.nii.gz")
            if not os.path.exists(p):
                # label manquant -> volume vide aligne sur le CT
                empty = sitk.Image(itk_img.GetSize(), sitk.sitkUInt8)
                empty.SetSpacing(itk_img.GetSpacing())
                empty.SetOrigin(itk_img.GetOrigin())
                empty.SetDirection(itk_img.GetDirection())
                lab_dict[lab] = empty
            else:
                lab_dict[lab] = read_sitk_with_nib_fallback(p)

        resample_image_and_labels(itk_img, lab_dict, tgt_path, name,
                                   target_spacing=target_spacing, z_bspline=z_bspline)
        return (name, "ok")
    except Exception as exc:  # noqa: BLE001
        return (name, f"error: {exc!r}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Reechantillonne le dataset ICH en 1x1x1 mm (format MedFormer nii).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src_path", default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich",
                   help="Dataset ICH (sortie de build_ich_dataset.py). CT = <ID>/ct.nii.gz.")
    p.add_argument("--label_path", default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich",
                   help="Meme dossier que src par defaut : labels = <ID>/segmentations/<label>.nii.gz.")
    p.add_argument("--tgt_path", default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_1mm",
                   help="Destination (format MedFormer nii, 1 mm).")
    p.add_argument("--label_yaml",
                   default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich/label_names_ich.yaml",
                   help="YAML listant les classes a reechantillonner. Si absent : auto-detection.")
    p.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   help="Espacement cible (mm).")
    p.add_argument("--z_bspline", action="store_true",
                   help="Interpoler l'axe Z en BSpline (defaut : NearestNeighbor, separate-z).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="Ne traiter que les N premiers cas (debug).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    # Liste des cas = sous-dossiers contenant un ct.nii.gz
    name_list = sorted(
        d for d in os.listdir(args.src_path)
        if os.path.isdir(os.path.join(args.src_path, d))
        and os.path.exists(os.path.join(args.src_path, d, "ct.nii.gz"))
    )
    if args.limit:
        name_list = name_list[: args.limit]
    print(f"Cas a reechantillonner : {len(name_list)}")
    print(f"Spacing cible : {tuple(args.spacing)} mm | axe Z : {'BSpline' if args.z_bspline else 'NearestNeighbor (separate-z)'}")

    # Classes
    if args.label_yaml and os.path.exists(args.label_yaml):
        with open(args.label_yaml) as f:
            lab_name_list = yaml.safe_load(f)
    else:
        seg_dir = os.path.join(args.label_path, name_list[0], "segmentations")
        lab_name_list = sorted(fn[:-7] for fn in os.listdir(seg_dir)
                               if fn.endswith(".nii.gz") and "background" not in fn)
    print(f"Classes ({len(lab_name_list)}) : {lab_name_list}")

    os.makedirs(os.path.join(args.tgt_path, "list"), exist_ok=True)
    with open(os.path.join(args.tgt_path, "list", "dataset.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(name_list, f)
    with open(os.path.join(args.tgt_path, "list", "label_names.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(lab_name_list, f)

    worker = partial(process_case, src_path=args.src_path, label_path=args.label_path,
                     tgt_path=args.tgt_path, lab_name_list=lab_name_list,
                     target_spacing=tuple(args.spacing), z_bspline=args.z_bspline,
                     overwrite=args.overwrite)

    from collections import Counter
    status = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (name, st) in enumerate(ex.map(worker, name_list), 1):
            status["ok" if st == "ok" else ("exists" if st == "exists" else "error")] += 1
            if st.startswith("error"):
                print(f"  [error] {name} : {st}")
            if i % 50 == 0:
                print(f"  ... {i}/{len(name_list)}", flush=True)

    print("\n=== RECAP ===")
    for k, v in sorted(status.items()):
        print(f"  {k}: {v}")

    # Sanity : geometrie d'un cas de sortie
    ex_ct = os.path.join(args.tgt_path, f"{name_list[0]}.nii.gz")
    if os.path.exists(ex_ct):
        im = sitk.ReadImage(ex_ct)
        print(f"\nExemple {name_list[0]} : size {im.GetSize()} | spacing "
              f"{tuple(round(s, 3) for s in im.GetSpacing())}")


if __name__ == "__main__":
    main()
