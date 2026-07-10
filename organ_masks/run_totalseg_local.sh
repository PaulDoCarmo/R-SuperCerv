#!/bin/bash
# run_totalseg_local.sh -- Port MACHINE LOCALE de LaunchTotalSegmentator.sh (Alliance).
# Lance TotalSegmentator brain_structures sur TOUS les <ID>.nii.gz d'un dossier d'entree,
# vers $OUT/segmented_organs_<ID>.nii.gz/  (convention lue par build_ich*_dataset.py).
#
# Robustesse :
#   - PAS de `set -o pipefail` et sortie de chaque cas redirigee vers un FICHIER (pas un pipe)
#     -> evite le SIGPIPE (exit 141) des barres tqdm quand on detache le process.
#   - Idempotent : saute un cas si ses 12 structures utiles sont deja presentes.
#   - Foreground par cas (un seul GPU) ; relançable.
#
# Usage :
#   bash run_totalseg_local.sh <input_dir> [output_dir]
#   # ex. branche MASQUES :
#   bash run_totalseg_local.sh /mnt/Data/data_paul/data_laurent/NIFTI/vols
#   # ex. branche RAPPORTS :
#   bash run_totalseg_local.sh /mnt/Data/data_paul/reports_ct
#
# Pour un long batch sans bloquer le terminal, detacher :
#   nohup bash run_totalseg_local.sh <input_dir> < /dev/null > $D/logs/totalseg_batch.log 2>&1 &
set -eu

D="${RSUPER_DATA:-/mnt/Data/data_paul}"
IN="${1:?usage: run_totalseg_local.sh <input_dir> [output_dir]}"
OUT="${2:-$D/organ_masks/segmentations}"
LOGDIR="$D/logs/totalseg"; mkdir -p "$OUT" "$LOGDIR"

source "$HOME/envs/totalseg/bin/activate"
source "$(dirname "${BASH_SOURCE[0]}")/secrets.sh"   # TOTALSEG_LICENSE (gitignored)
export TOTALSEG_WEIGHTS_DIR="$D/organ_masks/weights"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

STRUCTS12="brainstem caudate_nucleus cerebellum frontal_lobe insular_cortex internal_capsule \
lentiform_nucleus occipital_lobe parietal_lobe temporal_lobe thalamus ventricle"

shopt -s nullglob
cts=("$IN"/*.nii.gz)
echo "[$(date)] $IN : ${#cts[@]} CT -> $OUT"
done=0; skipped=0; failed=0
for ct in "${cts[@]}"; do
    id=$(basename "$ct")                       # <ID>.nii.gz
    dst="$OUT/segmented_organs_$id"
    # idempotence : les 12 structures utiles deja la ?
    ok=1
    for s in $STRUCTS12; do [ -s "$dst/$s.nii.gz" ] || { ok=0; break; }; done
    if [ "$ok" = 1 ]; then skipped=$((skipped+1)); continue; fi
    if TotalSegmentator -i "$ct" -o "$dst" -ta brain_structures \
           < /dev/null > "$LOGDIR/$id.log" 2>&1; then
        done=$((done+1)); echo "  ok   $id"
    else
        failed=$((failed+1)); echo "  FAIL $id (voir $LOGDIR/$id.log)"
    fi
done
echo "[$(date)] termine : nouveaux=$done, sautes(deja faits)=$skipped, echecs=$failed"
