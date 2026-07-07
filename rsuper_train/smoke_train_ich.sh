#!/bin/bash
# smoke_train_ich.sh -- D-3 : test de bout en bout de l'entrainement (stage 1, masques seuls).
# But : valider data -> modele -> loss (BCE+Dice) -> backward -> checkpoint, en ~quelques minutes.
# Job Slurm GPU (Trillium / Alliance). NE PAS lancer sur le login.
#
#   sbatch smoke_train_ich.sh
#
#SBATCH --job-name=ich_smoke
#SBATCH --account=def-josedolz
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.out
#SBATCH --error=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.err

module purge
module load StdEnv/2023 python/3.10

source /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train/rsuper_env/bin/activate

# matplotlib (importe par train_ddp) : cache ecrivable
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

RTRAIN=/home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train
cd "$RTRAIN"

DATA=/home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_npz
EXP=/home/pauldcrm/links/scratch/R-SuperCerv/exp
LOG=/home/pauldcrm/links/scratch/R-SuperCerv/log
AUG=/home/pauldcrm/links/scratch/R-SuperCerv/ich_augmented   # requis par le dataset (non ecrit sans --save_augmented)
mkdir -p "$EXP" "$LOG" "$AUG"

# Smoke test : 1 epoch, 10 iterations, patch 96^3, batch 2, 1 GPU, SANS losses-rapport.
python train_ddp.py \
    --dataset ich --model medformer --dimension 3d \
    --data_root "$DATA" \
    --save_destination "$AUG" \
    --cp_path "$EXP/" --log_path "$LOG/" \
    --unique_name ich_smoke \
    --crop_on_tumor \
    --report_volume_loss_basic 0 \
    --gpu '0' --workers 4 \
    --batch_size 2 --crop_size 96 \
    --epochs 1 --iter_per_epoch_override 10 \
    --lr 0.0001 \
    --dist_url tcp://127.0.0.1:8765

echo "=== smoke test termine. Checkpoint attendu : $EXP/ich/ich_smoke/fold_0_latest.pth ==="
ls -lh "$EXP/ich/ich_smoke/" 2>/dev/null
