#!/bin/bash
# preprocess.sh -- nnU-Net : fingerprint + auto-config (patch/stages/pooling anisotropes)
#                  + CTNormalization + preprocessing des 305 cas. Definit aussi les 5 folds.
# Job Slurm CPU (~1-2 h). A lancer AVANT les entrainements.
#
#   sbatch preprocess.sh
#
#SBATCH --job-name=nnunet_prep
#SBATCH --account=def-josedolz
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.out
#SBATCH --error=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.err

module purge
source /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/nnunet/nnunet_paths.sh

# Auto-configuration + preprocessing (3d_fullres). --verify verifie geometrie image/label.
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -c 3d_fullres --verify_dataset_integrity -np 12

echo "=== plans generes -> $nnUNet_preprocessed/Dataset$(printf %03d $DATASET_ID)_${DATASET_NAME}/ ==="
echo "Config auto-derivee :"
python - <<'PY'
import json, os, glob
pp=os.environ["nnUNet_preprocessed"]; d=glob.glob(pp+"/Dataset001_*")[0]
plans=json.load(open(os.path.join(d,"nnUNetPlans.json")))
cfg=plans["configurations"]["3d_fullres"]
print("  patch_size   :", cfg["patch_size"])
print("  spacing      :", cfg.get("spacing"))
print("  n_stages     :", len(cfg["architecture"]["arch_kwargs"]["strides"]))
print("  strides      :", cfg["architecture"]["arch_kwargs"]["strides"])
print("  batch_size   :", cfg["batch_size"])
print("  normalization:", cfg.get("normalization_schemes"))
PY
