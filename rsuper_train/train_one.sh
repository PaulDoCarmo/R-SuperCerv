#!/bin/bash
# train_one.sh -- 1 run d'entrainement parametre (grid d'hyperparametres).
# Lance via launch_grid.sh (qui passe les variables RUN_* par --export).
# Job Slurm GPU (Trillium / Alliance).
#
#SBATCH --account=rrg-josedolz
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.out
#SBATCH --error=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.err

module purge
module load StdEnv/2023 python/3.10
source /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train/rsuper_env/bin/activate

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"; mkdir -p "$MPLCONFIGDIR"

RTRAIN=/home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train
cd "$RTRAIN"

DATA=/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_npz_trainval   # vue TRAINVAL (test held-out exclu)
EXP=/home/pauldcrm/links/scratch/R-SuperCerv/exp
LOG=/home/pauldcrm/links/scratch/R-SuperCerv/log
AUG=/home/pauldcrm/links/scratch/R-SuperCerv/ich_augmented
mkdir -p "$EXP" "$LOG" "$AUG"

# Variables passees par launch_grid.sh (--export)
: "${RUN_NAME:?manque RUN_NAME}"; : "${RUN_LR:?}"; : "${RUN_TD:?}"; : "${RUN_TH:?}"; : "${RUN_TW:?}"; : "${RUN_ROT:?}"; : "${RUN_PORT:?}"
echo "=== Run ich_${RUN_NAME} : lr=$RUN_LR  patch=[$RUN_TD $RUN_TH $RUN_TW]  rot=$RUN_ROT  port=$RUN_PORT ==="

# Reprise auto si checkpoint existe
CKPT="$EXP/ich/ich_${RUN_NAME}/fold_0_latest.pth"
RESUME=""; [ -f "$CKPT" ] && { RESUME="--resume --load $CKPT"; echo "Reprise depuis $CKPT"; }

python train_ddp.py \
    --dataset ich --model medformer --dimension 3d \
    --data_root "$DATA" --save_destination "$AUG" \
    --cp_path "$EXP/" --log_path "$LOG/" \
    --unique_name "ich_${RUN_NAME}" \
    --crop_on_tumor --report_volume_loss_basic 0 \
    --gpu '0' --workers 8 --batch_size 2 \
    --epochs 100 --iter_per_epoch_override 250 \
    --lr "$RUN_LR" \
    --training_size_override "$RUN_TD" "$RUN_TH" "$RUN_TW" \
    --rotate_override "$RUN_ROT" \
    --dist_url "tcp://127.0.0.1:${RUN_PORT}" \
    $RESUME

echo "=== ich_${RUN_NAME} termine. best=$EXP/ich/ich_${RUN_NAME}/fold_0_best.pth ==="
