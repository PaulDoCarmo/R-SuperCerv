#!/bin/bash
# eval_ich.local.sh -- Port MACHINE LOCALE de eval_ich.sh.
# Evalue un checkpoint sur le set de TEST held-out (54 cas masques) : Dice / NSD@1mm / HD95 /
# volumes pred vs GT / detection. Metriques sur le canal 'ich_lesion' uniquement.
#
# IMPORTANT : --npz_dir = dataset_ich_full_npz (359, INCLUT les 54 cas test) ; le split
# test_ids.csv selectionne les 54. (Le dataset ...npz_trainval, lui, EXCLUT le test.)
#
#   bash eval_ich.local.sh [CKPT] [SAVE_CSV]
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT="${1:-$D/exp/ich/ich_stage1_13cls/fold_0_best.pth}"
SAVE_CSV="${2:-$D/eval/ich_stage1_13cls_test.csv}"

source "$HOME/envs/rsuper/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"; mkdir -p "$MPLCONFIGDIR"
cd "$REPO"

python eval_ich.py \
    --load "$CKPT" \
    --npz_dir "$D/dataset_ich_full_npz" \
    --ids     "$D/splits/test_ids.csv" \
    --config  config/ich/medformer_3d.yaml \
    --class_list "$D/dataset_ich_full_npz/list/label_names.yaml" \
    --save_csv "$SAVE_CSV" \
    --gpu 0 --threshold 0.5 --nsd_tol 1.0
echo "=== EVAL termine -> $SAVE_CSV ==="
