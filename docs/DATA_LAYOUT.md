# Layout des données locales (glossaire chemins) — machine GPU interne

Racine unique : **`RSUPER_DATA=/mnt/Data/data_paul`** (noté `$D`). Tout ce que produit le
pipeline s'écrit sous `$D`. Les scripts Python prennent leurs chemins en **argument** (rien à
éditer dans le code) ; seuls les `.sh` Alliance et 2 configs YAML ont des chemins en dur à porter.

## Arborescence cible

```
$D = /mnt/Data/data_paul
├── data_laurent/NIFTI/                 [FAIT] ENTREE branche MASQUES (stage 1 / Atlas)
│   ├── vols/   <ID>.nii.gz             359 CT natifs (symlinks)   ID = ID_<hash>_instUid_ID_<hash>
│   ├── masks/  <ID>.nii.gz             359 masques {1=ICH,2=IVH,3=PHE}
│   └── meta/   df_cases.csv, stage_2_*.csv
├── reports_ct/                         [A FAIRE] ENTREE branche RAPPORTS (stage 2 / UFO)
│   └── <ID>.nii.gz                     65 CT baseline (_0) (symlinks)  ID = <patientID>_0
├── organ_masks/
│   ├── weights/                        cache poids TotalSeg (TOTALSEG_WEIGHTS_DIR, 1x)
│   └── segmentations/                  SORTIE TotalSegmentator (les 2 branches, IDs disjoints)
│       └── segmented_organs_<ID>.nii.gz/{brainstem,caudate_nucleus,...,ventricle}.nii.gz
│                                        359 (masques) + 65 (rapports) = 424 dossiers
├── report_extraction/                  pipeline LLM (plus tard)
│   ├── reports.csv  raw/  post_processed/  format/  metadata/  HFModels/  HFCache/
├── dataset_ich_full/                   SORTIE build_ich_dataset.py --with_structures (Atlas 13cls)
│   └── <ID>/{ct.nii.gz, segmentations/{ich_lesion,<12 structures>}.nii.gz}
│       + label_names_ich_full.yaml     (13 classes)
├── dataset_ich_full_1mm/               SORTIE resample_ich_3d.py (1x1x1mm)
├── dataset_ich_full_npz/               SORTIE nii_to_npz_ich.py  -> data_root (stage 1 & 2)
├── dataset_ich_reports/                SORTIE build_ich_reports_dataset.py (UFO, 12 struct, PAS de lésion)
│   └── <ID>/{ct.nii.gz, segmentations/<12 structures>.nii.gz}
├── dataset_ich_reports_1mm/            SORTIE resample (label_names_ich_organs.yaml)
├── dataset_ich_reports_npz/            SORTIE npz  -> UFO_root (stage 2)
├── splits/                             SORTIE make_split.py  {test_ids,trainval_ids}.csv + vue *_trainval
├── exp/                                checkpoints entraînement (fold_0_best.pth, ...)
├── log/                                tensorboard
└── logs/                               stdout/stderr des jobs
```

## Contrat lecture → écriture par étape

| Étape | Lit | Écrit |
|---|---|---|
| **1. TotalSegmentator** (GPU) | `data_laurent/NIFTI/vols/<ID>.nii.gz` **et** `reports_ct/<ID>.nii.gz` | `organ_masks/segmentations/segmented_organs_<ID>.nii.gz/<struct>.nii.gz` |
| **2. LLM → métadonnées** (GPU) | rapports `.txt` (via adaptateur) | `report_extraction/metadata/ich_per_tumor_metadata.csv` |
| **3a. build Atlas** | `data_laurent/NIFTI/{vols,masks}` + `organ_masks/segmentations` | `dataset_ich_full/` + `label_names_ich_full.yaml` |
| **3b. resample Atlas** | `dataset_ich_full/` | `dataset_ich_full_1mm/` |
| **3c. npz Atlas** | `dataset_ich_full_1mm/` | `dataset_ich_full_npz/` (+ `list/label_names.yaml`) |
| **4a. build Rapports** | `reports_ct/` + `organ_masks/segmentations` | `dataset_ich_reports/` |
| **4b/c. resample+npz Rapports** | `dataset_ich_reports/` | `dataset_ich_reports_1mm/` → `dataset_ich_reports_npz/` |
| **5. re-train stage 1 (13cls)** | `dataset_ich_full_npz/` | `exp/ich/ich_stage1_13cls/fold_0_best.pth` |
| **6. fine-tune stage 2** | `dataset_ich_full_npz/` + `dataset_ich_reports_npz/` + `metadata` | `exp/.../ich_stage2/` |

## Chemins à adapter (Alliance → local)

- **Scripts Python** (`build_*`, `resample_ich_3d`, `nii_to_npz_ich`, `make_split`, `report_to_rsuper_metadata`) :
  chemins **en argument** → il suffit de passer les `--*_dir/--*_path` ci-dessus. Aucune édition.
- **`.sh` Alliance** (TotalSeg, train, eval) : retirer `#SBATCH` + `module load`, activer le venv
  `~/envs/*`, `CUDA_VISIBLE_DEVICES=0`. Versions portées : `*/create_*_env.local.sh` (envs faits).
- **2 configs YAML à éditer** (défauts, surchargées par CLI) :
  `rsuper_train/config/ich/medformer_3d.yaml` (`data_root`),
  `rsuper_train/config/ich_ufo/medformer_3d.yaml` (`data_root`, `UFO_root`, `reports`).
- **Défauts argparse** (optionnel, sinon passer en CLK) : `report_to_rsuper_metadata.py`,
  `build_ich_dataset.py`, `resample_ich_3d.py`, `nnunet/*`, `compare_methods.py`, `plot_loss.py`.

## Structure d'ENTREE attendue par les scripts (déjà satisfaite pour les masques)

- TotalSeg : un dossier plat de `<ID>.nii.gz` → `data_laurent/NIFTI/vols` ✓ et `reports_ct/` (à créer).
- build_ich_dataset : `vols/<ID>.nii.gz` + `masks/<ID>.nii.gz` appariés par **stem identique** ✓.
- build_ich_reports_dataset : `reports_ct/<ID>.nii.gz` + `segmented_organs_<ID>.nii.gz/`.
- Appariement partout = **stem de fichier** (`<ID>`), qui devient le `BDMAP_ID` des métadonnées.
