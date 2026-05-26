#!/usr/bin/env bash
set -euo pipefail

RAW_INPUT="${1:-}"
GROUND_TRUTH_JSON="${2:-/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/ground_truth_from_json.csv}"

shift_count=0
if [[ -n "$RAW_INPUT" ]]; then shift_count=$((shift_count + 1)); fi
if [[ -n "$GROUND_TRUTH_JSON" ]]; then shift_count=$((shift_count + 1)); fi
shift "$shift_count" || true

IGNORE_ALL_FALSE=false
TOLERANCE_MM=5
ALL_PROMPTS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ignore_all_false)
      IGNORE_ALL_FALSE=true
      shift
      ;;
    --tolerance_mm)
      TOLERANCE_MM="${2:-}"
      shift 2
      ;;
    --all_prompts)
      ALL_PROMPTS=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$RAW_INPUT" ]]; then
  echo "Usage: $0 /path/to/raw/results_..._promptN.csv|/path/to/raw_dir [ground_truth_json.csv] [--ignore_all_false] [--tolerance_mm N] [--all_prompts]"
  exit 1
fi

if [[ ! -f "$RAW_INPUT" && ! -d "$RAW_INPUT" ]]; then
  echo "Raw input not found: $RAW_INPUT"
  exit 1
fi

if [[ ! -f "$GROUND_TRUTH_JSON" ]]; then
  echo "Ground truth json not found: $GROUND_TRUTH_JSON"
  exit 1
fi

POSTPROCESSED_ROOT="/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/post_processed"
FORMATTED_ROOT="/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/format"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ALL="$SCRIPT_DIR/run_all_metrics.sh"

COMMON_ARGS=()
if [[ "$IGNORE_ALL_FALSE" == "true" ]]; then
  COMMON_ARGS+=("--ignore_all_false")
fi
COMMON_ARGS+=("--tolerance_mm" "$TOLERANCE_MM")

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

  python "$SCRIPT_DIR/postprocess.py" \
    --input "$RAW_INPUT"

  python "$SCRIPT_DIR/format_metrics.py" \
    --input "$POSTPROCESSED_PATH" \
    --all

  "$RUN_ALL" "$FORMATTED_PATH" "$GROUND_TRUTH_JSON" "${COMMON_ARGS[@]}"

  printf "Done.\n- Postprocessed: %s\n- Formatted: %s\n" "$POSTPROCESSED_PATH" "$FORMATTED_PATH"
  exit 0
fi

python "$SCRIPT_DIR/postprocess.py" \
  --input "$RAW_INPUT"

if [[ "$ALL_PROMPTS" == "true" ]]; then
  for prompt_dir in "${POSTPROCESSED_ROOT}"/prompt*; do
    if [[ -d "$prompt_dir" ]]; then
      python "$SCRIPT_DIR/format_metrics.py" \
        --input "$prompt_dir" \
        --all
    fi
  done

  for prompt_dir in "${FORMATTED_ROOT}"/prompt*; do
    if [[ -d "$prompt_dir" ]]; then
      "$RUN_ALL" "$prompt_dir" "$GROUND_TRUTH_JSON" "${COMMON_ARGS[@]}"
    fi
  done
else
  PROMPT_DIR_NAME="$(basename "$RAW_INPUT")"
  POSTPROCESSED_PATH="$POSTPROCESSED_ROOT/$PROMPT_DIR_NAME"
  FORMATTED_PATH="$FORMATTED_ROOT/$PROMPT_DIR_NAME"

  python "$SCRIPT_DIR/format_metrics.py" \
    --input "$POSTPROCESSED_PATH" \
    --all

  "$RUN_ALL" "$FORMATTED_PATH" "$GROUND_TRUTH_JSON" "${COMMON_ARGS[@]}"
fi

printf "Done.\n"
