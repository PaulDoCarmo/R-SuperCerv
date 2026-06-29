#!/bin/bash
# eval_ich.sh -- Evaluation du stage 1 sur le set de TEST held-out (54 cas).
# Job Slurm GPU (Trillium / Alliance). A lancer quand l'entrainement a produit un checkpoint.
#
#   sbatch eval_ich.sh
#
#SBATCH --job-name=ich_eval
#SBATCH --account=def-josedolz
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.out
#SBATCH --error=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.err

module purge
module load StdEnv/2023 python/3.10
source /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train/rsuper_env/bin/activate

RTRAIN=/home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train
cd "$RTRAIN"

CKPT=/home/pauldcrm/links/scratch/R-SuperCerv/exp/ich/ich_stage1/fold_0_latest.pth

python eval_ich.py \
    --load "$CKPT" \
    --npz_dir /home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_npz \
    --ids     /home/pauldcrm/links/scratch/R-SuperCerv/splits/test_ids.csv \
    --config  config/ich/medformer_3d.yaml \
    --class_list /home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_npz/list/label_names.yaml \
    --save_csv /home/pauldcrm/links/scratch/R-SuperCerv/eval/test_metrics.csv \
    --gpu 0 --threshold 0.5 --nsd_tol 1.0
