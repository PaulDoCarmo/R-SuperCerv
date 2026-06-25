#!/bin/bash
# train_ich.sh -- Stage 1 R-SuperCerv : entrainement reel (segmentation ICH, masques seuls).
# Job Slurm GPU (Trillium / Alliance). NE PAS lancer sur le login.
#
#   sbatch train_ich.sh
#   # reprise apres coupure : ajouter au bas de la commande python :
#   #   --resume --load /home/pauldcrm/links/scratch/R-SuperCerv/exp/ich/ich_stage1/fold_0_latest.pth
#
#SBATCH --job-name=ich_stage1
#SBATCH --account=def-josedolz
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.out
#SBATCH --error=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.err

module purge
module load StdEnv/2023 python/3.10

source /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train/rsuper_env/bin/activate

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

RTRAIN=/home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train
cd "$RTRAIN"

# IMPORTANT : on entraine sur la VUE TRAINVAL (305 cas) -> le test (54) reste held-out.
DATA=/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_npz_trainval
EXP=/home/pauldcrm/links/scratch/R-SuperCerv/exp
LOG=/home/pauldcrm/links/scratch/R-SuperCerv/log
AUG=/home/pauldcrm/links/scratch/R-SuperCerv/ich_augmented   # requis par le dataset (non ecrit sans --save_augmented)
mkdir -p "$EXP" "$LOG" "$AUG"

# Reprise automatique si un checkpoint existe (sinon, run frais)
CKPT="$EXP/ich/ich_stage1/fold_0_latest.pth"
RESUME=""
if [ -f "$CKPT" ]; then RESUME="--resume --load $CKPT"; echo "Reprise depuis $CKPT"; fi

python train_ddp.py \
    --dataset ich --model medformer --dimension 3d \
    --data_root "$DATA" \
    --save_destination "$AUG" \
    --cp_path "$EXP/" --log_path "$LOG/" \
    --unique_name ich_stage1 \
    --crop_on_tumor \
    --report_volume_loss_basic 0 \
    --gpu '0' --workers 8 \
    --batch_size 2 --crop_size 128 \
    --epochs 100 --iter_per_epoch_override 250 \
    --dist_url tcp://127.0.0.1:8770 \
    $RESUME

echo "=== entrainement termine. Checkpoints : $EXP/ich/ich_stage1/ ==="
ls -lh "$EXP/ich/ich_stage1/" 2>/dev/null
