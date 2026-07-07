# R-SuperCerv — Segmentation d'hémorragie crânienne (ICH) supervisée par rapports

Transposition de **R-Super** (MICCAI 2025, apprentissage de la segmentation tumorale à
partir de rapports radiologiques via *Volume Loss* + *Ball Loss*) à la **segmentation
d'hémorragie intracrânienne** sur CT cérébral. Objectif scientifique : **montrer la valeur
ajoutée (et les limites) des rapports écrits** comme supervision, face à une baseline nnU-Net.

> **Contexte machine.** Le code a été développé sur **Compute Canada / Alliance** (SLURM,
> modules, wheelhouse `--no-index`). Il est en cours de portage sur une **machine à GPU
> interne** qui a accès aux **rapports radiologiques privés**. Voir
> [`docs/PORTAGE_SLURM_TO_LOCAL.md`](docs/PORTAGE_SLURM_TO_LOCAL.md) et
> [`STAGE2_RUNBOOK.md`](STAGE2_RUNBOOK.md).

---

## Vue d'ensemble : pipeline en 2 stages

```
                 CT (.nii.gz)                         Rapports radiologiques (privés)
                      │                                          │
      ┌───────────────┴───────────────┐            ┌─────────────┴─────────────┐
      │ masques ICH per-voxel (Laurent)│            │ LLM extraction (FAIT)     │
      │                                │            │  -> *_formated.csv        │
      │  TotalSegmentator              │            │  report_to_rsuper_metadata│
      │  (brain_structures)            │            │  -> ich_per_tumor_meta.csv│
      └───────────────┬───────────────┘            └─────────────┬─────────────┘
                      │                                          │
       dataset-MASQUES (Atlas, 13 classes)         dataset-RAPPORTS (UFO, 12 structures)
       CT + 12 structures + ich_lesion             CT + 12 structures (PAS de lésion)
                      │                                          │
                      └────────────────┬─────────────────────────┘
                                       ▼
                     STAGE 1 : segmentation (masques seuls)   ← baseline supervisée
                                       ▼
                     STAGE 2 : + Volume/Ball loss (rapports)  ← l'apport R-Super
```

- **Stage 1** — MedFormer entraîné sur les masques ICH seuls (`--report_volume_loss_basic 0`).
- **Stage 2** — on repart du checkpoint stage 1 (`--pretrained`) et on ajoute la supervision
  par rapports (`--report_volume_loss_basic 0.1 --loss ball_dice_last`), qui apprend la lésion
  là où on n'a **que** le rapport (localisation = structure, taille → volume via ABC/2).
- **Baseline** — nnU-Net v2 multi-classes (`nnunet/`), pour comparer.

**Décision de design clé (stage 2) : « structures = classes ».** Le modèle prédit
**13 canaux** = 12 structures cérébrales (TotalSegmentator `brain_structures`) + `ich_lesion`.
Raison : R-Super construit le `chosen_segment_mask` (région où le rapport situe la lésion)
à partir des canaux *structures du label du modèle* → en faisant des structures des classes,
l'adaptation du Dataset est minimale. **Conséquence** : le stage 1 doit être ré-entraîné avec
ces 13 classes pour que le checkpoint `--pretrained` corresponde à la tête du stage 2.

---

## Carte du dépôt

| Dossier | Rôle | README |
|---|---|---|
| `report_extraction/` | Extraction LLM des rapports (vLLM) → métadonnées R-Super | [`README.md`](report_extraction/README.md) |
| `organ_masks/` | TotalSegmentator `brain_structures` (12 masques/cas) | [`README_brain_structures.md`](organ_masks/README_brain_structures.md) |
| `rsuper_train/dataset_conversion/` | Assemblage + resample 1mm + npz (masques ET rapports) | [`README.md`](rsuper_train/dataset_conversion/README.md) |
| `rsuper_train/` | Entraînement MedFormer (stage 1 & 2), configs, éval | [`README.md`](rsuper_train/README.md) |
| `nnunet/` | Baseline nnU-Net v2 | [`README.md`](nnunet/README.md) |
| `docs/` | Portage SLURM→local, glossaire chemins | — |

Chaque environnement Python est un dossier `*_env/` **spécifique à Alliance** (à NE PAS
copier sur la nouvelle machine — recréer via `requirements-*.txt`, cf. portage).

---

## Où en est-on ? (2026-07-07)

- ✅ **Stage 1** validé sur Alliance : MedFormer, Dice test médian **0.906**.
- ✅ **Baseline nnU-Net** : Dice médian **0.924** (bat MedFormer ; confond prétraitement
  anisotrope natif vs 1mm iso — voir `nnunet/README.md`).
- ✅ **Extraction LLM des rapports** : pipeline au point (Qwen2.5-72B-AWQ + prompt5 via vLLM),
  déjà tourné sur Alliance — **à re-faire tourner en local** sur la machine interne (rapports
  privés → re-télécharger le LLM). Voir `report_extraction/README.md`.
- ✅ **Code stage 2 (supervision par rapports)** : **complet et validé** là où testable sans
  les données privées (`report_to_rsuper_metadata.py`, `ICHReportsDataset`, config `ich_ufo`).
- ⏳ **BLOQUÉ sur données privées** (→ nouvelle machine) : TotalSeg sur tous les cas,
  rebuild Atlas 13 classes, **re-train stage 1 (13 classes)**, build dataset-rapports,
  fine-tune stage 2. Runbook détaillé : [`STAGE2_RUNBOOK.md`](STAGE2_RUNBOOK.md).

---

## Démarrage rapide (nouvelle machine)

```bash
# 1) Environnement (Python 3.10 + internet, PAS de modules/wheelhouse)
python3.10 -m venv ~/envs/rsuper && source ~/envs/rsuper/bin/activate
pip install -r rsuper_train/requirements-rsuper.txt
#   torch : réinstalle la variante qui matche ton CUDA (cf. requirements-rsuper.txt)

# 2) Choisis un ROOT data local et exporte-le une fois pour toutes
export RSUPER_DATA=/chemin/local/vers/data   # remplace partout les /home/pauldcrm/links/scratch/...

# 3) Suis STAGE2_RUNBOOK.md étape par étape.
```
