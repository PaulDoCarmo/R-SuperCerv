#!/bin/bash
# launch_nnunet_folds.sh -- Soumet les 5 folds nnU-Net en parallele (1 GPU chacun, independants).
# A lancer sur un noeud de LOGIN, APRES preprocess.sh.
#
#   bash launch_nnunet_folds.sh                 # 5 folds, 1000 epochs (defaut, papier)
#   TRAINER=nnUNetTrainer_250epochs bash launch_nnunet_folds.sh   # version rapide

TRAIN_FOLD=/home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/nnunet/train_fold.sh
TRAINER="${TRAINER:-nnUNetTrainer}"

for f in 0 1 2 3 4; do
  echo "sbatch nnunet_f${f} (trainer=$TRAINER)"
  sbatch --job-name="nnunet_f${f}" \
    --export=ALL,FOLD=${f},TRAINER=${TRAINER} \
    "$TRAIN_FOLD"
done
echo
echo "5 folds soumis. Suivi : squeue -u \$USER"
