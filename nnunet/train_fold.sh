#!/bin/bash
# train_fold.sh -- Entraine UN fold nnU-Net (3d_fullres). Lance par launch_nnunet_folds.sh
#                  (qui passe FOLD par --export). --requeue + reprise auto (nnU-Net --c).
# Job Slurm GPU.
#
#SBATCH --account=rrg-josedolz
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.out
#SBATCH --error=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.err

module purge
source /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/nnunet/nnunet_paths.sh

: "${FOLD:?manque FOLD}"
TRAINER="${TRAINER:-nnUNetTrainer}"        # nnUNetTrainer = 1000 epochs (defaut, papier)
                                           # alternative rapide : nnUNetTrainer_250epochs
echo "=== nnU-Net train : dataset $DATASET_ID | 3d_fullres | fold $FOLD | trainer $TRAINER ==="

# Reprise auto : si un checkpoint existe deja pour ce fold -> --c (continue)
CKPT="$nnUNet_results/Dataset$(printf %03d $DATASET_ID)_${DATASET_NAME}/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}/checkpoint_latest.pth"
CONT=""; [ -f "$CKPT" ] && { CONT="--c"; echo "Reprise (--c) depuis $CKPT"; }

nnUNetv2_train "$DATASET_ID" 3d_fullres "$FOLD" -tr "$TRAINER" $CONT

echo "=== fold $FOLD termine ==="
