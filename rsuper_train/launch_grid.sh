#!/bin/bash
# launch_grid.sh -- Lance les 5 runs du grid d'hyperparametres en parallele (1 GPU chacun).
# A executer sur un noeud de LOGIN (il fait juste des sbatch).
#
#   bash launch_grid.sh
#
# Grid (un facteur a la fois autour de la baseline R0) :
#   name      lr       patch (D H W)   rotation   port
#   R0_base   6e-4     128 128 128     30         baseline
#   R1_lrlo   3e-4     128 128 128     30         LR plus bas
#   R2_lrhi   1e-3     128 128 128     30         LR plus haut
#   R3_patch  6e-4     96  128 128     30         patch anisotropie-aware (moins de Z)
#   R4_aug    6e-4     128 128 128     12         augmentation moins agressive

TRAIN_ONE=/home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train/train_one.sh

# "name lr td th tw rot port"
configs=(
  "R0_base 0.0006 128 128 128 30 8801"
  "R1_lrlo 0.0003 128 128 128 30 8802"
  "R2_lrhi 0.001  128 128 128 30 8803"
  "R3_patch 0.0006 96 128 128 30 8804"
  "R4_aug  0.0006 128 128 128 12 8805"
)

for c in "${configs[@]}"; do
  read -r name lr td th tw rot port <<< "$c"
  echo "sbatch ich_${name} : lr=${lr} patch=[${td} ${th} ${tw}] rot=${rot} port=${port}"
  sbatch --job-name="ich_${name}" \
    --export=ALL,RUN_NAME=${name},RUN_LR=${lr},RUN_TD=${td},RUN_TH=${th},RUN_TW=${tw},RUN_ROT=${rot},RUN_PORT=${port} \
    "$TRAIN_ONE"
done

echo
echo "5 jobs soumis. Suivi : squeue -u \$USER"
echo "Best-models attendus dans : /home/pauldcrm/links/scratch/R-SuperCerv/exp/ich/ich_<name>/fold_0_best.pth"
