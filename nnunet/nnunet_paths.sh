# nnunet_paths.sh -- variables communes nnU-Net (a "source" dans chaque job).
module load StdEnv/2023 python/3.10
source /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/nnunet/nnunet_env/bin/activate

export nnUNet_raw=/home/pauldcrm/links/scratch/R-SuperCerv/nnUNet_raw
export nnUNet_preprocessed=/home/pauldcrm/links/scratch/R-SuperCerv/nnUNet_preprocessed
export nnUNet_results=/home/pauldcrm/links/scratch/R-SuperCerv/nnUNet_results
mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"

export DATASET_ID=1
export DATASET_NAME=ICH
