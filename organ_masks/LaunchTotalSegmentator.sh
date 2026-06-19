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
DOSSIER_ENTREE="/home/pauldcrm/links/scratch/data_laurent/test_cases 2026_04_22/"
DOSSIER_SORTIE="/home/pauldcrm/links/scratch/R-SuperCerv/organ_masks/segmentations_organes"
FICHIER_SCAN="ID_ca7bbeab_instUid_ID_a3bc73b11.nii.gz"
DOSSIER_SCAN="/home/pauldcrm/links/scratch/data_laurent/test_cases 2026_04_22/"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights-dir)   WEIGHTS_DIR="$2";    shift 2 ;;
        --entree)        DOSSIER_ENTREE="$2"; shift 2 ;;
        --sortie)        DOSSIER_SORTIE="$2"; shift 2 ;;
        --scan)          FICHIER_SCAN="$2";   shift 2 ;;
        --dossier-scan)  DOSSIER_SCAN="$2";   shift 2 ;;
        *) echo "Argument inconnu: $1"; exit 1 ;;
    esac
done

mkdir -p "$WEIGHTS_DIR" "$DOSSIER_SORTIE"
export TOTALSEG_WEIGHTS_DIR="$WEIGHTS_DIR"

run_totalsegmentator() {
    local fichier_entree="$1"
    local nom_scan
    nom_scan=$(basename "$fichier_entree")

    echo "Début de la segmentation pour $nom_scan à $(date)"

    # Lancement de la tâche cerveau avec fusion des labels (--ml) et aperçu 3D (--preview)
    TotalSegmentator -i "$fichier_entree" \
                     -o "$DOSSIER_SORTIE/segmented_organs_$nom_scan" \
                     -ta brain_structures \
                     --preview

    # Lancement de la tâche saignements
    # TotalSegmentator -i "$fichier_entree" \
    #                  -o "$DOSSIER_SORTIE/saignements_$nom_scan" \
    #                  -ta cerebral_bleed \
    #                  --ml

    echo "Fin de la segmentation pour $nom_scan à $(date)"
}

if [[ -n "$DOSSIER_SCAN" ]]; then
    # Mode dossier : traiter tous les .nii.gz dans DOSSIER_SCAN
    shopt -s nullglob
    scans=("$DOSSIER_SCAN"/*.nii.gz)
    if [[ ${#scans[@]} -eq 0 ]]; then
        echo "Aucun fichier .nii.gz trouvé dans $DOSSIER_SCAN"
        exit 1
    fi
    echo "${#scans[@]} fichier(s) trouvé(s) dans $DOSSIER_SCAN"
    for scan in "${scans[@]}"; do
        run_totalsegmentator "$scan"
    done
else
    # Mode fichier : traiter un seul scan
    run_totalsegmentator "$DOSSIER_ENTREE/$FICHIER_SCAN"
fi