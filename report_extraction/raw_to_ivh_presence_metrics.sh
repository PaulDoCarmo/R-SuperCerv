#!/usr/bin/env bash
set -euo pipefail

RAW_CSV="${1:-}"
GROUND_TRUTH="${2:-/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/ground_truth_from_booleans.csv}"

if [[ -z "$RAW_CSV" ]]; then
  echo "Usage: $0 /path/to/raw/results_..._promptN.csv [ground_truth.csv]"
  exit 1
fi

if [[ ! -f "$RAW_CSV" ]]; then
  echo "Raw CSV not found: $RAW_CSV"
  exit 1
fi

if [[ ! -f "$GROUND_TRUTH" ]]; then
  echo "Ground truth not found: $GROUND_TRUTH"
  exit 1
fi

PROMPT_ID="U"
if [[ "$RAW_CSV" =~ /prompt([0-9]+)/ ]]; then
  PROMPT_ID="${BASH_REMATCH[1]}"
elif [[ "$RAW_CSV" =~ _prompt([0-9]+)\.csv$ ]]; then
  PROMPT_ID="${BASH_REMATCH[1]}"
fi

BASE_NAME="$(basename "$RAW_CSV")"
MODEL_NAME="U"
if [[ "$BASE_NAME" =~ ^results_(.+)_prompt[0-9]+\.csv$ ]]; then
  MODEL_NAME="${BASH_REMATCH[1]}"
elif [[ "$BASE_NAME" =~ ^results_(.+)\.csv$ ]]; then
  MODEL_NAME="${BASH_REMATCH[1]}"
fi

POSTPROCESSED_PATH="/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/post_processed/prompt${PROMPT_ID}/results_${MODEL_NAME}_prompt${PROMPT_ID}_postprocessed.csv"
FORMATTED_PATH="/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/format/prompt${PROMPT_ID}/results_${MODEL_NAME}_prompt${PROMPT_ID}_formated.csv"

python /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/report_extraction/postprocess.py \
  --input "$RAW_CSV"

python /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/report_extraction/format_metrics.py \
  --input "$POSTPROCESSED_PATH" \
  --all

python /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/report_extraction/compute_metrics/compute_ivh_presence_metrics.py \
  --ground_truth "$GROUND_TRUTH" \
  --formatted_results "$FORMATTED_PATH"

printf "Done.\n- Postprocessed: %s\n- Formatted: %s\n" "$POSTPROCESSED_PATH" "$FORMATTED_PATH"
