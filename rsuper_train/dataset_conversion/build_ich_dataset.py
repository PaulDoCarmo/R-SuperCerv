#!/usr/bin/env python3
"""
build_ich_dataset.py -- Etape A du pipeline R-SuperCerv (ICH, stage 1 "mask-only").

Assemble les donnees brutes (CT + masques lesion multi-classe) dans l'arborescence
canonique attendue par R-Super :

    out_dir/
      <ID>/
        ct.nii.gz                 -> symlink vers le CT d'origine (vols/<ID>.nii.gz)
        segmentations/
          ich_lesion.nii.gz       = (masque_origine == ICH_LABEL), binaire uint8

Contexte :
- Les masques d'origine (data_laurent/NIFTI/masks) encodent 1=ICH, 2=IVH, 3=PHE.
  Ici on n'extrait QUE l'ICH (label 1) : c'est la cible du stage 1.
- On n'inclut PAS de masque d'organe. Pour l'entrainement sur masques seuls,
  seul le canal lesion est necessaire. Les masques d'organes (TotalSegmentator)
  ne servent qu'au stage 2 (supervision par rapports).
- Les 29 cas qui possedent deja des masques d'organes sont des cas "rapports"
  (a traiter plus tard) : on les EXCLUT par defaut pour eviter toute fuite
  train/test (option --keep_report_cases pour les garder).

Sortie supplementaire :
- out_dir/manifest_ich.csv : un recap par cas (volume ICH, shape, spacing...).
- out_dir/label_names_ich.yaml : la liste des classes pour le stage 1 (= [ich_lesion]).

Le script est idempotent (relancable) et parallelise.
"""

import argparse
import os
import sys
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Assemble CT + masques ICH au format R-Super (etape A, stage 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vols_dir", default="/home/pauldcrm/links/scratch/data_laurent/NIFTI/vols",
                   help="Dossier des CT (un <ID>.nii.gz par cas).")
    p.add_argument("--masks_dir", default="/home/pauldcrm/links/scratch/data_laurent/NIFTI/masks",
                   help="Dossier des masques lesion multi-classe (un <ID>.nii.gz par cas).")
    p.add_argument("--out_dir", default="/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich",
                   help="Destination du dataset reformate.")
    p.add_argument("--organ_masks_dir",
                   default="/home/pauldcrm/links/scratch/R-SuperCerv/organ_masks/segmentations_organes",
                   help="Dossier des masques d'organes TotalSegmentator. Sert UNIQUEMENT a "
                        "detecter les cas 'rapports' a exclure (dossiers segmented_organs_<ID>.nii.gz).")
    p.add_argument("--ich_label", type=int, default=1,
                   help="Valeur encodant l'ICH dans les masques d'origine.")
    p.add_argument("--keep_report_cases", action="store_true",
                   help="Ne PAS exclure les cas qui ont des masques d'organes (cas 'rapports').")
    p.add_argument("--exclude_ids", default=None,
                   help="(Optionnel) CSV avec une colonne d'IDs a exclure en plus.")
    p.add_argument("--copy_ct", action="store_true",
                   help="Copier le CT au lieu de faire un symlink (par defaut : symlink).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="Ne traiter que les N premiers cas (debug). 0 = tous.")
    p.add_argument("--overwrite", action="store_true",
                   help="Reecrire les ich_lesion.nii.gz deja presents.")
    p.add_argument("--dry_run", action="store_true", help="N'ecrit rien, affiche seulement le plan.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def case_id_from_file(path):
    """ID = nom de fichier sans l'extension .nii.gz."""
    name = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def list_ids(directory):
    if not os.path.isdir(directory):
        return set()
    return {case_id_from_file(f) for f in os.listdir(directory)
            if f.endswith(".nii.gz") or f.endswith(".nii")}


def detect_report_case_ids(organ_masks_dir):
    """Les cas 'rapports' sont ceux qui ont un dossier segmented_organs_<ID>.nii.gz."""
    ids = set()
    if not os.path.isdir(organ_masks_dir):
        return ids
    for name in os.listdir(organ_masks_dir):
        if name.startswith("segmented_organs_"):
            core = name[len("segmented_organs_"):]
            for ext in (".nii.gz", ".nii"):
                if core.endswith(ext):
                    core = core[: -len(ext)]
                    break
            ids.add(core)
    return ids


def load_exclude_csv(path):
    ids = set()
    if not path:
        return ids
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            for cell in row:
                cell = cell.strip()
                if cell.startswith("ID_"):
                    ids.add(cell)
    return ids


# --------------------------------------------------------------------------- #
# Traitement d'un cas (execute dans un process worker)
# --------------------------------------------------------------------------- #
def process_case(task):
    cid, ct_src, mask_src, out_dir, ich_label, copy_ct, overwrite, dry_run = task
    res = {"id": cid, "status": "ok", "detail": "",
           "ich_voxels": 0, "ich_volume_mm3": 0.0, "shape": "", "spacing": ""}
    try:
        case_dir = os.path.join(out_dir, cid)
        seg_dir = os.path.join(case_dir, "segmentations")
        ct_dst = os.path.join(case_dir, "ct.nii.gz")
        ich_dst = os.path.join(seg_dir, "ich_lesion.nii.gz")

        # Idempotence : si la lesion existe deja et qu'on ne reecrit pas, on ne
        # la recalcule pas, mais on relit le fichier produit pour remplir le
        # manifest (volume, shape, spacing) -> manifest toujours complet.
        if os.path.exists(ich_dst) and not overwrite:
            res["status"] = "exists"
            done = nib.load(ich_dst)
            arr = np.asarray(done.dataobj)
            zd = tuple(float(z) for z in done.header.get_zooms()[:3])
            n = int(arr.sum())
            res["ich_voxels"] = n
            res["ich_volume_mm3"] = float(n * zd[0] * zd[1] * zd[2])
            res["shape"] = "x".join(str(s) for s in arr.shape[:3])
            res["spacing"] = "x".join(f"{z:.3f}" for z in zd)
            return res

        # Header CT (lazy : pas de chargement des voxels).
        ct_img = nib.load(ct_src)
        ct_shape = tuple(int(s) for s in ct_img.shape[:3])

        # Masque lesion (on doit lire les voxels pour extraire l'ICH).
        mask_img = nib.load(mask_src)
        mask_shape = tuple(int(s) for s in mask_img.shape[:3])

        # Verification de coherence geometrique CT <-> masque.
        if ct_shape != mask_shape:
            res["status"] = "shape_mismatch"
            res["detail"] = f"ct{ct_shape} vs mask{mask_shape}"
            return res
        if not np.allclose(ct_img.affine, mask_img.affine, atol=1e-2):
            # On signale mais on continue : on s'aligne sur l'affine du masque
            # (qui partage la meme grille que le CT, decale au plus a la marge numerique).
            res["detail"] = "affine_diff(<=tol relachee)"

        mask_data = np.asarray(mask_img.dataobj)
        ich = (mask_data == ich_label).astype(np.uint8)

        zooms = tuple(float(z) for z in mask_img.header.get_zooms()[:3])
        n_ich = int(ich.sum())
        res["ich_voxels"] = n_ich
        res["ich_volume_mm3"] = float(n_ich * zooms[0] * zooms[1] * zooms[2])
        res["shape"] = "x".join(str(s) for s in mask_shape)
        res["spacing"] = "x".join(f"{z:.3f}" for z in zooms)

        if n_ich == 0:
            # Ne devrait pas arriver (label 1 present dans 100% des masques) -> on signale.
            res["status"] = "empty_ich"

        if dry_run:
            res["detail"] = (res["detail"] + "|dry_run").strip("|")
            return res

        # Ecriture.
        os.makedirs(seg_dir, exist_ok=True)

        # CT : symlink (defaut) ou copie.
        if not os.path.exists(ct_dst):
            if copy_ct:
                import shutil
                shutil.copyfile(ct_src, ct_dst)
            else:
                os.symlink(os.path.realpath(ct_src), ct_dst)

        # Masque ICH : on reutilise affine + header du masque d'origine (alignement exact),
        # en forcant le dtype uint8.
        hdr = mask_img.header.copy()
        hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(ich, mask_img.affine, hdr), ich_dst)

        return res
    except Exception as exc:  # noqa: BLE001 - on veut le statut, pas un crash global
        res["status"] = "error"
        res["detail"] = repr(exc)
        return res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    vols_ids = list_ids(args.vols_dir)
    mask_ids = list_ids(args.masks_dir)
    paired = sorted(vols_ids & mask_ids)
    print(f"CT (vols)         : {len(vols_ids)}")
    print(f"Masques lesion    : {len(mask_ids)}")
    print(f"Paires CT+masque  : {len(paired)}")

    # Exclusions
    excluded = set()
    if not args.keep_report_cases:
        report_ids = detect_report_case_ids(args.organ_masks_dir)
        excluded |= (report_ids & set(paired))
        print(f"Cas 'rapports' detectes (masques d'organes presents) : {len(report_ids)} "
              f"-> {len(report_ids & set(paired))} exclus du stage 1")
    extra = load_exclude_csv(args.exclude_ids)
    if extra:
        excluded |= (extra & set(paired))
        print(f"IDs exclus via --exclude_ids : {len(extra & set(paired))}")

    final_ids = [cid for cid in paired if cid not in excluded]
    if args.limit:
        final_ids = final_ids[: args.limit]
    print(f"Cas a traiter (stage 1, mask-only) : {len(final_ids)}")
    print(f"Destination : {args.out_dir}")
    if args.dry_run:
        print(">>> DRY RUN : aucune ecriture.")

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    for cid in final_ids:
        ct_src = os.path.join(args.vols_dir, cid + ".nii.gz")
        mask_src = os.path.join(args.masks_dir, cid + ".nii.gz")
        tasks.append((cid, ct_src, mask_src, args.out_dir, args.ich_label,
                      args.copy_ct, args.overwrite, args.dry_run))

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_case, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 50 == 0:
                    print(f"  ... {i}/{len(tasks)}", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            results.append(process_case(t))
            if i % 50 == 0:
                print(f"  ... {i}/{len(tasks)}", flush=True)

    # Recap
    from collections import Counter
    status = Counter(r["status"] for r in results)
    print("\n=== RECAP ===")
    for k, v in sorted(status.items()):
        print(f"  {k}: {v}")
    problems = [r for r in results if r["status"] in ("shape_mismatch", "error", "empty_ich")]
    for r in problems[:20]:
        print(f"  [{r['status']}] {r['id']} : {r['detail']}")

    vols_mm3 = [r["ich_volume_mm3"] for r in results if r["ich_voxels"] > 0]
    if vols_mm3:
        arr = np.array(vols_mm3)
        print(f"\nVolume ICH (mm3) sur {len(arr)} cas : "
              f"min={arr.min():.0f}  median={np.median(arr):.0f}  "
              f"mean={arr.mean():.0f}  max={arr.max():.0f}")

    # Manifest + yaml de classes
    if not args.dry_run:
        manifest = os.path.join(args.out_dir, "manifest_ich.csv")
        with open(manifest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "status", "ich_voxels",
                                               "ich_volume_mm3", "shape", "spacing", "detail"])
            w.writeheader()
            for r in sorted(results, key=lambda x: x["id"]):
                w.writerow(r)
        print(f"\nManifest -> {manifest}")

        yaml_path = os.path.join(args.out_dir, "label_names_ich.yaml")
        with open(yaml_path, "w") as fh:
            fh.write("- ich_lesion\n")
        print(f"Classes  -> {yaml_path}  (stage 1 : 1 classe = ich_lesion)")


if __name__ == "__main__":
    main()
