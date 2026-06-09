#!/bin/bash
#SBATCH --job-name=TotalSeg_R-SuperCerv       # Nom du job
#SBATCH --account=rrg-josedolz          # REMPLACEZ par votre groupe de recherche (ex: def-professeur)
#SBATCH --time=05:00:00                       # Temps alloué (1 heure est large pour un ou deux scans)
#SBATCH --nodes=1                             # Un seul nœud
#SBATCH --gpus-per-node=1                     # Demande 1 GPU (essentiel pour TotalSegmentator)
#SBATCH --cpus-per-task=8                     # 8 cœurs CPU pour accélérer le prétraitement et la lecture NIfTI
#SBATCH --output=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.out                    # Fichier de log standard (NomDuJob-ID.out)
#SBATCH --error=/home/pauldcrm/links/scratch/R-SuperCerv/logs/%x-%j.err                     # Fichier de log d'erreur (NomDuJob-ID.err)

module purge

module load StdEnv/2023 gcc/12.3 python/3.11 arrow/17.0.0 cuda/12.2 vtk

source /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/organ_masks/organ_masks_env/bin/activate

export HOME="/home/pauldcrm/links/scratch/R-SuperCerv/organ_masks"

# source /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/organ_masks/secrets.sh
# totalseg_set_license -l "$TOTALSEG_LICENSE"

WEIGHTS_DIR="/home/pauldcrm/links/scratch/R-SuperCerv/organ_masks"
mkdir -p "$WEIGHTS_DIR"
export TOTALSEG_WEIGHTS_DIR="$WEIGHTS_DIR"

# 5. Exécution de TotalSegmentator
# Variables pour simplifier la lecture (à adapter selon vos dossiers)
DOSSIER_ENTREE="/home/pauldcrm/links/scratch/data_laurent/test_cases 2026_04_22/"
DOSSIER_SORTIE="/home/pauldcrm/links/scratch/R-SuperCerv/organ_masks/segmentations"
FICHIER_SCAN="ID_ca7bbeab_instUid_ID_a3bc73b11.nii.gz"

echo "Début de la segmentation pour $FICHIER_SCAN à $(date)"

# Lancement de la tâche cerveau avec fusion des labels (--ml) et aperçu 3D (--preview)
TotalSegmentator -i "$DOSSIER_ENTREE/$FICHIER_SCAN" \
                 -o "$DOSSIER_SORTIE/cerveau_$FICHIER_SCAN" \
                 -ta brain_structures \
                 --preview

# Lancement de la tâche saignements
# TotalSegmentator -i "$DOSSIER_ENTREE/$FICHIER_SCAN" \
#                  -o "$DOSSIER_SORTIE/saignements_$FICHIER_SCAN" \
#                  -ta cerebral_bleed \
#                  --ml

echo "Fin de la segmentation à $(date)"