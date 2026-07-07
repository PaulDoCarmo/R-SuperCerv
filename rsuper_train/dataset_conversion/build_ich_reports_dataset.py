#!/usr/bin/env python3
"""
build_ich_reports_dataset.py -- Assemble le DATASET-RAPPORTS (UFO) au format R-Super.

Cas-rapports = CT + masques de structures cerebrales (TotalSegmentator), SANS lesion
(la lesion est deduite du rapport via les losses Volume/Ball). Structure produite :

    out_dir/<ID>/
        ct.nii.gz                       -> symlink vers le CT du rapport
        segmentations/
            frontal_lobe.nii.gz  ...    -> les 12 structures utilisees pour chosen_segment_mask

Entree : la sortie TotalSegmentator est deja "1 dossier par cas, 1 nii par structure"
(segmented_organs_<ID>.nii.gz/<structure>.nii.gz) -> on ne fait que renommer/symlinker.

C'est l'analogue de build_ich_dataset.py, mais avec les STRUCTURES (pas la lesion).
"""
import argparse
import os

# Les 12 structures que report_to_rsuper_metadata.py peut cibler (= label_names_ich_organs.yaml)
STRUCTURES = ["brainstem", "caudate_nucleus", "cerebellum", "frontal_lobe",
              "insular_cortex", "internal_capsule", "lentiform_nucleus",
              "occipital_lobe", "parietal_lobe", "temporal_lobe", "thalamus", "ventricle"]


def parse_args():
    p = argparse.ArgumentParser(description="Assemble le dataset-rapports ICH (CT + structures).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # >>> CHEMINS A ADAPTER QUAND LES DONNEES ARRIVENT <<<
    p.add_argument("--vols_dir", required=True,
                   help="Dossier des CT des cas-rapports (<ID>.nii.gz).")
    p.add_argument("--totalseg_dir", required=True,
                   help="Sortie TotalSegmentator : dossiers 'segmented_organs_<ID>.nii.gz/'.")
    p.add_argument("--out_dir", required=True,
                   help="Destination du dataset-rapports (format R-Super).")
    p.add_argument("--totalseg_prefix", default="segmented_organs_",
                   help="Prefixe des dossiers TotalSegmentator.")
    p.add_argument("--copy_ct", action="store_true", help="Copier le CT au lieu de symlink.")
    return p.parse_args()


def case_id_from_ct(fname):
    for ext in (".nii.gz", ".nii"):
        if fname.endswith(ext):
            return fname[:-len(ext)]
    return fname


def main():
    a = parse_args()
    cts = sorted(f for f in os.listdir(a.vols_dir) if f.endswith((".nii.gz", ".nii")))
    print(f"CT trouves : {len(cts)}")

    ok, miss_ts, miss_struct = 0, 0, 0
    for f in cts:
        cid = case_id_from_ct(f)
        # dossier TotalSegmentator du cas (essaie avec et sans .nii.gz dans le nom)
        cand = [os.path.join(a.totalseg_dir, a.totalseg_prefix + cid + ".nii.gz"),
                os.path.join(a.totalseg_dir, a.totalseg_prefix + cid)]
        ts_dir = next((c for c in cand if os.path.isdir(c)), None)
        if ts_dir is None:
            miss_ts += 1
            continue

        case_dir = os.path.join(a.out_dir, cid)
        seg_dir = os.path.join(case_dir, "segmentations")
        os.makedirs(seg_dir, exist_ok=True)

        # CT
        ct_dst = os.path.join(case_dir, "ct.nii.gz")
        ct_src = os.path.realpath(os.path.join(a.vols_dir, f))
        if not os.path.exists(ct_dst):
            if a.copy_ct:
                import shutil; shutil.copyfile(ct_src, ct_dst)
            else:
                os.symlink(ct_src, ct_dst)

        # 12 structures (symlink)
        missing_here = 0
        for s in STRUCTURES:
            src = os.path.join(ts_dir, s + ".nii.gz")
            if not os.path.exists(src):
                missing_here += 1
                continue
            dst = os.path.join(seg_dir, s + ".nii.gz")
            if not os.path.exists(dst):
                os.symlink(os.path.realpath(src), dst)
        if missing_here:
            miss_struct += 1
        ok += 1

    print(f"Cas assembles : {ok} | sans dossier TotalSeg : {miss_ts} | "
          f"cas avec >=1 structure manquante : {miss_struct}")
    print(f"-> {a.out_dir}/<ID>/{{ct.nii.gz, segmentations/<structures>.nii.gz}}")
    print("Puis : resample_ich_3d.py (--label_yaml label_names_ich_organs.yaml) -> nii_to_npz_ich.py")


if __name__ == "__main__":
    main()
