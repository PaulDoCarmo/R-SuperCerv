# dataset_conversion — CT/masques bruts → `.npz` R-Super

Chaîne de préparation des données (identique pour le dataset **MASQUES** et le dataset
**RAPPORTS**, seul le YAML de classes change). Tout est **CPU** (SimpleITK), pas de GPU.

```
build_*.py  →  resample_ich_3d.py  →  nii_to_npz_ich.py
(assemble)     (1mm iso)              (fenêtre HU + z-score + npz)
```

> Tous les chemins ci-dessous sont des **placeholders** Alliance → remplace par tes chemins
> locaux. Chaque script prend ses chemins en argument (aucune édition de code requise).

---

## Scripts

| Script | Rôle | Entrée | Sortie |
|---|---|---|---|
| `build_ich_dataset.py` | Assemble CT + masque ICH (+ structures si `--with_structures`) | `vols/<ID>.nii.gz`, `masks/<ID>.nii.gz`, `organ_masks/segmented_organs_<ID>.nii.gz/` | `<ID>/{ct.nii.gz, segmentations/*.nii.gz}` + `label_names_ich*.yaml` |
| `build_ich_reports_dataset.py` | Assemble CT-rapports + 12 structures (**sans** lésion) | `vols/<ID>.nii.gz`, `organ_masks/...` | `<ID>/{ct.nii.gz, segmentations/<12 struct>.nii.gz}` |
| `resample_ich_3d.py` | Resample **1×1×1 mm** (XY BSpline image / NN labels ; Z NN « separate-z ») | dossier `<ID>/ct.nii.gz` + `segmentations/` | `<ID>.nii.gz` (CT 1mm) + `<ID>/<label>.nii.gz` |
| `nii_to_npz_ich.py` | Fenêtre HU cerveau **[0,100]** + z-score → `.npz` | sortie resample | `<ID>.npz` (image), `<ID>_gt.npz` (labels CxZxYxX) + `list/{dataset,label_names}.yaml` |
| `make_split.py` | Split stratifié train/val/test (test held-out) | dossier npz | `splits/{test_ids,trainval_ids}.csv` + vue symlink `*_trainval` |
| `resample_utils.py` | Helpers de resampling (importé) | — | — |

---

## Recette — dataset MASQUES

**Stage 1 (1 classe, actuel)** :
```bash
python build_ich_dataset.py --vols_dir $D/vols --masks_dir $D/masks \
    --out_dir $D/dataset_ich --keep_report_cases
python resample_ich_3d.py --src_path $D/dataset_ich --label_path $D/dataset_ich \
    --tgt_path $D/dataset_ich_1mm --label_yaml $D/dataset_ich/label_names_ich.yaml --workers 16
python nii_to_npz_ich.py --src_path $D/dataset_ich_1mm --tgt_path $D/dataset_ich_npz \
    --hu_min 0 --hu_max 100 --workers 16
python make_split.py ...   # produit la vue *_trainval (test exclu)
```

**Stage 2 (13 classes = 12 structures + ich_lesion)** : ajoute `--with_structures` +
`--organ_masks_dir`, et resample avec `--label_yaml .../label_names_ich_full.yaml` :
```bash
python build_ich_dataset.py --vols_dir $D/vols --masks_dir $D/masks \
    --organ_masks_dir $D/organ_masks/segmentations \
    --out_dir $D/dataset_ich_full --keep_report_cases --with_structures
python resample_ich_3d.py --src_path $D/dataset_ich_full --label_path $D/dataset_ich_full \
    --tgt_path $D/dataset_ich_full_1mm --label_yaml $D/dataset_ich_full/label_names_ich_full.yaml --workers 16
python nii_to_npz_ich.py --src_path $D/dataset_ich_full_1mm --tgt_path $D/dataset_ich_full_npz \
    --hu_min 0 --hu_max 100 --workers 16
```

## Recette — dataset RAPPORTS (UFO, 12 structures, SANS lésion)

```bash
python build_ich_reports_dataset.py --vols_dir $D/reports_ct \
    --totalseg_dir $D/organ_masks/segmentations --out_dir $D/dataset_ich_reports
python resample_ich_3d.py --src_path $D/dataset_ich_reports --label_path $D/dataset_ich_reports \
    --tgt_path $D/dataset_ich_reports_1mm --label_yaml label_names_ich_organs.yaml --workers 16
python nii_to_npz_ich.py --src_path $D/dataset_ich_reports_1mm --tgt_path $D/dataset_ich_reports_npz \
    --hu_min 0 --hu_max 100 --workers 16
```
Le stem `<ID>` du npz **doit** matcher `BDMAP_ID` des métadonnées (voir report_extraction).

---

## Détails importants

- **Fenêtre HU [0,100]** : fenêtre « cerveau/sang » (le sang aigu ≈ 50–70 HU). On clippe puis
  z-score. Élargir la fenêtre efface le contraste substance blanche/grise.
- **Resampling anisotrope → 1mm** : les CT natifs sont ~0.49×0.49×5 mm. On resample XY en
  BSpline (image), Z en NearestNeighbor (`separate-z`) pour ne pas inventer d'information
  travers-plan. Les labels toujours en NN. `--z_bspline` pour forcer BSpline en Z (déconseillé).
- **Naming npz** : `<ID>.npz` + `<ID>_gt.npz`. L'ID est **arbitraire** (le code de dataset a été
  rendu naming-agnostic, pas de préfixe `BDMAP`).
- **`list/label_names.yaml`** (écrit par `nii_to_npz_ich.py`) est **trié** et fait foi pour
  l'ordre des canaux → doit être cohérent avec `--classes` de l'entraînement.
