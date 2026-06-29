#!/bin/bash
# eval_grid.sh -- Evalue les 5 best-models du grid sur le TEST + CSV comparatif.
# A lancer QUAND les 5 runs sont finis (best-models presents).
# Job Slurm GPU (~15 min : 5 x eval de 54 cas).
#
#   sbatch eval_grid.sh
#
#SBATCH --job-name=ich_grid_cmp
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
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"; mkdir -p "$MPLCONFIGDIR"

cd /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train
python compare_grid.py --gpu 0
