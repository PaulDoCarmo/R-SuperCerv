# STAGE 2 — Runbook (supervision par rapports)

Étapes ordonnées à exécuter sur la **machine à GPU interne** (accès aux rapports privés).
Chemins = **placeholders** : remplace par tes chemins locaux (cf.
[`docs/PORTAGE_SLURM_TO_LOCAL.md`](docs/PORTAGE_SLURM_TO_LOCAL.md)). On note `$D = $RSUPER_DATA`.

> Rappel design : le modèle stage 2 a **13 classes** = 12 structures cérébrales + `ich_lesion`
> (`rsuper_train/dataset_conversion/label_names_ich_full.yaml`). Le dataset-RAPPORTS a **12
> structures** SANS lésion (`label_names_ich_organs.yaml`).

---

## Vue d'ensemble des données à produire

| Dataset | Contenu | Classes | Sert à |
|---|---|---|---|
| **Atlas** (masques) `$D/dataset_ich_full_npz/` | CT + 12 structures + `ich_lesion` | 13 | stage 1 (13cls) + branche Atlas du stage 2 |
| **UFO** (rapports) `$D/dataset_ich_reports_npz/` | CT + 12 structures (pas de lésion) | 12 | branche rapports du stage 2 |
| **Métadonnées** `$D/report_extraction/metadata/ich_per_tumor_metadata.csv` | 1 ligne/lésion (région, volume, diamètres) | — | Volume/Ball loss |

---

## Étape 0 — Vérifier ce qui est présent sur la machine

Cette machine a **les rapports**. À confirmer :
- Les **CT liés aux rapports** (`.nii.gz`) — indispensables pour le dataset UFO.
- Les **359 cas à masques ICH** (`data_laurent/NIFTI/{vols,masks}`) — indispensables pour
  l'Atlas 13 classes + le re-train stage 1. **S'ils ne sont PAS là, il faut les transférer.**
- Le CSV LLM formaté `*_formated.csv` (sortie de `report_extraction`).

---

## Étape 1 — TotalSegmentator `brain_structures` (GPU)

Sur **tous** les CT (cas-masques ET cas-rapports). Voir
[`organ_masks/README_brain_structures.md`](organ_masks/README_brain_structures.md).
Sortie attendue : **un dossier par cas, un `.nii.gz` par structure** :
```
$D/organ_masks/segmentations/segmented_organs_<ID>.nii.gz/<structure>.nii.gz
```
(12 structures utiles : brainstem, caudate_nucleus, cerebellum, frontal_lobe, insular_cortex,
internal_capsule, lentiform_nucleus, occipital_lobe, parietal_lobe, temporal_lobe, thalamus,
ventricle).

---

## Étape 2 — Rapports → métadonnées (LLM local, GPU)

⚠️ Les rapports étant **ici** (privés), il faut **re-faire tourner toute la partie
`report_extraction`**, ce qui implique de **re-télécharger le LLM en local** (sur Alliance il
était pré-téléchargé + servi hors-ligne). Chaîne complète et setup du LLM :
[`report_extraction/README.md`](report_extraction/README.md).

Résumé :
```bash
# a) JSON rapports -> reports.csv
python report_extraction/csv_builders/build_reports_csv.py $D/reports_json $D/report_extraction
# b) télécharger le LLM (Qwen2.5-72B-Instruct-AWQ, ~40GB, ~48GB VRAM) puis vLLM serve + inférence
#    (adapter LaunchMonoGPU.sh en exécution directe ; prompt_id 5)
# c) postprocess + format -> results_<MODEL>_prompt5_formated.csv (2 commandes directes, sans GT)
python report_extraction/postprocess.py --input $D/report_extraction/raw/results_<MODEL>_prompt5.csv \
    --output_root $D/report_extraction/post_processed
python report_extraction/format_metrics.py --all \
    --input $D/report_extraction/post_processed/prompt5/results_<MODEL>_prompt5_postprocessed.csv \
    --output_root $D/report_extraction/format
# d) métadonnées R-Super :
python report_extraction/report_to_rsuper_metadata.py \
    --input  $D/report_extraction/format/prompt5/results_<MODEL>_prompt5_formated.csv \
    --out_dir $D/report_extraction/metadata --types ICH
```
Produit `ich_per_tumor_metadata.csv` (+ `ich_per_CT_metadata.csv`). Vérifie le résumé imprimé :
% de tailles connues (le reste sera jeté), et le nombre d'`UNMAPPED` (régions non mappées →
cas jetés). Détails métadonnées : [`report_extraction/README_rsuper_metadata.md`](report_extraction/README_rsuper_metadata.md).

> **Machine = RTX 6000 Ada 48 GB (1 GPU)** : Qwen-72B-AWQ tient avec `--max-model-len 8192`
> (le défaut 32768 fait OOM). Voir la config exacte dans `report_extraction/README.md`.

---

## Étape 3 — Dataset ATLAS 13 classes (masques + structures)

1. Assembler CT + masque ICH + 12 structures au format R-Super (mode **`--with_structures`**,
   qui symlink les structures depuis `--organ_masks_dir` et écrit `label_names_ich_full.yaml`) :
   ```bash
   python rsuper_train/dataset_conversion/build_ich_dataset.py \
       --vols_dir  $D/data_laurent/NIFTI/vols \
       --masks_dir $D/data_laurent/NIFTI/masks \
       --organ_masks_dir $D/organ_masks/segmentations \
       --out_dir   $D/dataset_ich_full \
       --keep_report_cases --with_structures
   ```
   → `$D/dataset_ich_full/<ID>/segmentations/{ich_lesion, <12 structures>}.nii.gz`
   + `label_names_ich_full.yaml` (13 classes). Les cas avec structures manquantes sont
   signalés (colonne `detail` du manifest).
2. Resample 1mm puis npz (voir [`dataset_conversion/README.md`](rsuper_train/dataset_conversion/README.md)),
   avec `--label_yaml $D/dataset_ich_full/label_names_ich_full.yaml` (13 classes).
   Sortie : `$D/dataset_ich_full_npz/{<ID>.npz,<ID>_gt.npz,list/label_names.yaml}`.

---

## Étape 4 — Dataset RAPPORTS (CT + structures, PAS de lésion)

```bash
python rsuper_train/dataset_conversion/build_ich_reports_dataset.py \
    --vols_dir      $D/reports_ct \
    --totalseg_dir  $D/organ_masks/segmentations \
    --out_dir       $D/dataset_ich_reports
```
Puis resample 1mm + npz avec `--label_yaml label_names_ich_organs.yaml` (12 structures).
Sortie : `$D/dataset_ich_reports_npz/{<ID>.npz,<ID>_gt.npz,list/label_names.yaml}`.
Le stem `<ID>` **doit** correspondre à la colonne `BDMAP_ID` des métadonnées (étape 2).

---

## Étape 5 — RE-TRAIN stage 1 avec 13 classes

Le checkpoint `--pretrained` du stage 2 doit avoir la **même tête (13 canaux)**.
```bash
python rsuper_train/train_ddp.py \
    --dataset ich --model medformer --dimension 3d \
    --data_root $D/dataset_ich_full_npz \
    --classes_number 13 \
    --crop_on_tumor --report_volume_loss_basic 0 \
    --gpu '0' --batch_size 2 --epochs 100 \
    --unique_name ich_stage1_13cls ...
```
(mets à jour `config/ich/medformer_3d.yaml:data_root`, ou passe `--data_root`.)

---

## Étape 6 — Fine-tune stage 2 (Volume + Ball loss)

```bash
python rsuper_train/train_ddp.py \
    --dataset ich_ufo --model medformer --dimension 3d \
    --pretrained $D/exp/ich/ich_stage1_13cls/fold_0_best.pth \
    --report_volume_loss_basic 0.1 --loss ball_dice_last \
    --gpu '0' --batch_size 2 --epochs 100 \
    --unique_name ich_stage2 ...
```
`config/ich_ufo/medformer_3d.yaml` fournit `data_root` (Atlas 13cls), `UFO_root` (rapports),
`reports` (CSV) — **édite ces 3 chemins**. `tumor_classes` non passé → défaut `['brain']`.

---

## Étape 7 — Évaluation & comparaison

- Éval segmentation sur le test held-out (`eval_ich.py`), stage 1 vs stage 2.
- Métriques *détection* à partir des rapports (`ich_per_CT_metadata.csv`) : la lésion est-elle
  détectée dans la bonne région ? (à adapter de `predict_abdomenatlas.py` — **TODO**).
- Comparer à la baseline nnU-Net (`nnunet/compare_methods.py`).

**Message scientifique visé** : quantifier le gain (ou son absence / ses limites) apporté par
la supervision-rapports du stage 2, par-dessus le stage 1 et la baseline nnU-Net.

---

## Pièges connus

- **Séparateur régions** : les métadonnées utilisent `' / '` (natif R-Super). Ne pas le casser.
- **`UNMAPPED` / taille inconnue** : ces lésions/cas sont **volontairement jetés** par
  `clean_ufo` (on ne peut pas superviser un volume/une région inconnus).
- **Cohérence des classes** : `classes=13` (config) ⇔ `label_names_ich_full.yaml` trié ⇔ tête
  du checkpoint `--pretrained`. Toute divergence casse le chargement.
- **`BDMAP_ID`** : nom de colonne interne hérité de R-Super ; c'est juste l'**ID de cas** (le
  stem du `.npz`). Aucune contrainte de format « BDMAP » (code rendu naming-agnostic).
