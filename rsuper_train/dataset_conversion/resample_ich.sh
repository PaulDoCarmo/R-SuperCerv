#!/bin/bash
# resample_ich.sh -- Etape C-1 a l'echelle : resampling 1x1x1 mm des 359 cas ICH.
# Job Slurm (Compute Canada / Alliance) : CPU only (SimpleITK), pas de GPU.
#
#   sbatch resample_ich.sh
#
#SBATCH --job-name=resample_ich
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

python resample_ich_3d.py \
    --src_path  /home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich \
    --label_path /home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich \
    --tgt_path  /home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich_1mm \
    --label_yaml /home/pauldcrm/links/scratch/R-SuperCerv/dataset_ich/label_names_ich.yaml \
    --spacing 1.0 1.0 1.0 \
    --workers 16
