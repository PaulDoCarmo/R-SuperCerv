#!/bin/bash
# eval_ich_reports.local.sh -- Eval DETECTION/localisation sur le test-rapports (via metadata).
#   bash eval_ich_reports.local.sh [CKPT] [SPLIT_CSV] [SAVE_CSV]
# defaut : stage1 best-model, split _0 (reports_baseline0_test), sortie eval/.
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT="${1:-$D/exp/ich/ich_stage1_13cls/fold_0_best.pth}"
SPLIT="${2:-$D/splits/reports_baseline0_test_ids.csv}"
SAVE="${3:-$D/eval/ich_stage1_reports_baseline0_test.csv}"

source "$HOME/envs/rsuper/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"; mkdir -p "$MPLCONFIGDIR"
cd "$REPO"

python eval_ich_reports.py \
    --load "$CKPT" \
    --ufo_npz_dir "$D/dataset_ich_reports_npz" \
    --ids "$SPLIT" \
    --reports "$D/report_extraction/metadata/ich_per_tumor_metadata.csv" \
    --config config/ich_ufo/medformer_3d.yaml \
    --class_list "$D/dataset_ich_full_npz/list/label_names.yaml" \
    --ufo_class_list "$D/dataset_ich_reports_npz/list/label_names.yaml" \
    --save_csv "$SAVE" \
    --gpu 0 --threshold 0.5 --detect_min_ml 0.5 --min_overlap_ml 0.1
echo "=== EVAL RAPPORTS termine -> $SAVE ==="
