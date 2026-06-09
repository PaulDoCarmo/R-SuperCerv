#!/usr/bin/env bash
set -euo pipefail

FORMATTED_RESULTS="${1:-}"
GROUND_TRUTH_JSON="${2:-/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/ground_truth_from_json.csv}"

shift_count=0
if [[ -n "$FORMATTED_RESULTS" ]]; then shift_count=$((shift_count + 1)); fi
if [[ -n "$GROUND_TRUTH_JSON" ]]; then shift_count=$((shift_count + 1)); fi
shift "$shift_count" || true

IGNORE_ALL_FALSE=false
TOLERANCE_MM=5

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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$FORMATTED_RESULTS" ]]; then
  echo "Usage: $0 <formatted_results_path> [ground_truth_json.csv] [--ignore_all_false] [--tolerance_mm N]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRICS_DIR="$SCRIPT_DIR/compute_metrics"

COMMON_ARGS=("--formatted_results" "$FORMATTED_RESULTS")
if [[ "$IGNORE_ALL_FALSE" == "true" ]]; then
  COMMON_ARGS+=("--ignore_all_false")
fi

python "$METRICS_DIR/compute_ivh_presence_metrics.py" \
  --ground_truth "$GROUND_TRUTH_JSON" \
  "${COMMON_ARGS[@]}"

python "$METRICS_DIR/compute_phe_presence_metrics.py" \
  --ground_truth "$GROUND_TRUTH_JSON" \
  "${COMMON_ARGS[@]}"

python "$METRICS_DIR/compute_phe_severity_metrics.py" \
  --ground_truth "$GROUND_TRUTH_JSON" \
  "${COMMON_ARGS[@]}"

python "$METRICS_DIR/compute_ivh_location_metrics.py" \
  --ground_truth "$GROUND_TRUTH_JSON" \
  "${COMMON_ARGS[@]}" \
  --all_modes

python "$METRICS_DIR/compute_ich_location_metrics.py" \
  --ground_truth "$GROUND_TRUTH_JSON" \
  "${COMMON_ARGS[@]}" \
  --all_modes

python "$METRICS_DIR/compute_ich_size_metrics.py" \
  --ground_truth "$GROUND_TRUTH_JSON" \
  --formatted_results "$FORMATTED_RESULTS" \
  --all_modes \
  --tolerance_mm "$TOLERANCE_MM"

echo "All metrics computed."
