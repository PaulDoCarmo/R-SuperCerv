#!/bin/bash
# predict.sh -- Predit les 54 cas de TEST avec nnU-Net.
#   - ensemble 5-fold  (fidele au papier)
#   - fold 0 seul      (comparaison 1-contre-1 equitable vs MedFormer unique)
# Job Slurm GPU (~15-30 min). A lancer QUAND les folds sont entraines.
#
#   sbatch predict.sh
#
#SBATCH --job-name=nnunet_predict
#SBATCH --account=rrg-josedolz
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.out
#SBATCH --error=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.err

module purge
source /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/nnunet/nnunet_paths.sh

DS=Dataset$(printf %03d $DATASET_ID)_${DATASET_NAME}
IN="$nnUNet_raw/$DS/imagesTs"
OUT=/home/pauldcrm/links/scratch/R-SuperCerv/nnUNet_predictions
TR="${TRAINER:-nnUNetTrainer}"

echo "=== Prediction ENSEMBLE 5-fold (papier) ==="
nnUNetv2_predict -i "$IN" -o "$OUT/ensemble" -d "$DATASET_ID" -c 3d_fullres -tr "$TR" -f 0 1 2 3 4

echo "=== Prediction FOLD 0 seul (1-contre-1) ==="
nnUNetv2_predict -i "$IN" -o "$OUT/fold0" -d "$DATASET_ID" -c 3d_fullres -tr "$TR" -f 0

echo "=== Predictions -> $OUT/{ensemble,fold0} ==="
