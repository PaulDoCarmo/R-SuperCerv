#!/bin/bash
# npz_ich.sh -- Etape C-2 a l'echelle : nii (1mm) -> npz, fenetre cerveau [0,100] + z-score.
# Job Slurm (Compute Canada / Alliance) : CPU only. A lancer APRES resample_ich.sh.
#
#   sbatch npz_ich.sh
#
#SBATCH --job-name=npz_ich
#SBATCH --account=def-josedolz
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.out
#SBATCH --error=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.err

module purge
module load StdEnv/2023 python/3.10

source /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train/rsuper_env/bin/activate

CONV_DIR=/home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train/dataset_conversion
cd "$CONV_DIR"

python nii_to_npz_ich.py \
    --src_path /home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_1mm \
    --tgt_path /home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_npz \
    --hu_min 0 --hu_max 100 \
    --workers 16
