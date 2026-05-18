#!/usr/bin/env bash
set -euo pipefail

RAW_INPUT="${1:-}"
GROUND_TRUTH="${2:-/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/ground_truth_from_booleans.csv}"

if [[ -z "$RAW_INPUT" ]]; then
  echo "Usage: $0 /path/to/raw/results_..._promptN.csv|/path/to/raw_dir [ground_truth.csv]"
  exit 1
fi

if [[ ! -f "$RAW_INPUT" && ! -d "$RAW_INPUT" ]]; then
  echo "Raw input not found: $RAW_INPUT"
  exit 1
fi

if [[ ! -f "$GROUND_TRUTH" ]]; then
  echo "Ground truth not found: $GROUND_TRUTH"
  exit 1
fi

POSTPROCESSED_ROOT="/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/post_processed"
FORMATTED_ROOT="/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/format"

if [[ -f "$RAW_INPUT" ]]; then
  PROMPT_ID="U"
  if [[ "$RAW_INPUT" =~ /prompt([0-9]+)/ ]]; then
    PROMPT_ID="${BASH_REMATCH[1]}"
  elif [[ "$RAW_INPUT" =~ _prompt([0-9]+)\.csv$ ]]; then
    PROMPT_ID="${BASH_REMATCH[1]}"
  fi

  BASE_NAME="$(basename "$RAW_INPUT")"
  MODEL_NAME="U"
  if [[ "$BASE_NAME" =~ ^results_(.+)_prompt[0-9]+\.csv$ ]]; then
    MODEL_NAME="${BASH_REMATCH[1]}"
  elif [[ "$BASE_NAME" =~ ^results_(.+)\.csv$ ]]; then
    MODEL_NAME="${BASH_REMATCH[1]}"
  fi

  POSTPROCESSED_PATH="${POSTPROCESSED_ROOT}/prompt${PROMPT_ID}/results_${MODEL_NAME}_prompt${PROMPT_ID}_postprocessed.csv"
  FORMATTED_PATH="${FORMATTED_ROOT}/prompt${PROMPT_ID}/results_${MODEL_NAME}_prompt${PROMPT_ID}_formated.csv"

  python /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/report_extraction/postprocess.py \
    --input "$RAW_INPUT"

  python /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/report_extraction/format_metrics.py \
    --input "$POSTPROCESSED_PATH" \
    --all

  python /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/report_extraction/compute_metrics/compute_ivh_presence_metrics.py \
    --ground_truth "$GROUND_TRUTH" \
    --formatted_results "$FORMATTED_PATH"

  printf "Done.\n- Postprocessed: %s\n- Formatted: %s\n" "$POSTPROCESSED_PATH" "$FORMATTED_PATH"
  exit 0
fi

python /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/report_extraction/postprocess.py \
  --input "$RAW_INPUT"

for prompt_dir in "${POSTPROCESSED_ROOT}"/prompt*; do
  if [[ -d "$prompt_dir" ]]; then
    python /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/report_extraction/format_metrics.py \
      --input "$prompt_dir" \
      --all
  fi
done

for prompt_dir in "${FORMATTED_ROOT}"/prompt*; do
  if [[ -d "$prompt_dir" ]]; then
    python /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/report_extraction/compute_metrics/compute_ivh_presence_metrics.py \
      --ground_truth "$GROUND_TRUTH" \
      --formatted_results "$prompt_dir"
  fi
done

printf "Done.\n"
