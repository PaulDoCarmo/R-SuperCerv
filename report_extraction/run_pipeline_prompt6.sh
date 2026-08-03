#!/bin/bash
# run_pipeline_prompt6.sh -- Full pipeline d'extraction rapports avec PROMPT 6
# (detection IVH indirecte). Chaine : serve vLLM + inference -> postprocess -> format -> metadata.
# Ecrit dans metadata_prompt6/ (ne TOUCHE PAS le pipeline prompt5 en place).
#   tmux new -d -s p6 "bash run_pipeline_prompt6.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
RE="$D/report_extraction"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID=6
LOG="$D/logs/pipeline_prompt6.log"; mkdir -p "$D/logs"
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

log "===== PIPELINE PROMPT6 ====="
log "1/4 serve vLLM + inference (prompt 6, ~40-55 min)"
if ! bash "$HERE/serve_llm_local.sh" "$RE/reports.csv" "$RE/results.csv" qwen2-5-72b-awq 0 0.95 6 >> "$LOG" 2>&1; then
  log "FAIL serve/inference (voir $LOG)"; exit 1
fi
RAW=$(ls -t "$RE/raw/prompt6/"*_prompt6.csv 2>/dev/null | head -1)
[ -f "$RAW" ] || { log "FAIL : raw prompt6 introuvable"; exit 1; }
log "  raw = $RAW"

source "$HOME/envs/report/bin/activate"
log "2/4 postprocess"
python "$HERE/postprocess.py" --input "$RAW" --output_root "$RE/post_processed" >> "$LOG" 2>&1 || { log "FAIL postprocess"; exit 1; }
POST=$(ls -t "$RE/post_processed/prompt6/"*prompt6*.csv 2>/dev/null | head -1)
[ -f "$POST" ] || { log "FAIL : post introuvable"; exit 1; }
log "  post = $POST"

log "3/4 format_metrics --all"
python "$HERE/format_metrics.py" --input "$POST" --output_root "$RE/format" --all >> "$LOG" 2>&1 || { log "FAIL format"; exit 1; }
FMT=$(ls -t "$RE/format/prompt6/"*_formated.csv 2>/dev/null | head -1)
[ -f "$FMT" ] || { log "FAIL : format introuvable"; exit 1; }
log "  format = $FMT"

log "4/4 metadata (equal_A) -> metadata_prompt6/"
python "$HERE/report_to_rsuper_metadata.py" --input "$FMT" --out_dir "$RE/metadata_prompt6" --third_axis equal_A >> "$LOG" 2>&1 || { log "FAIL metadata"; exit 1; }

# comparaison rapide prompt5 vs prompt6 : nb de lignes IVH detectees
log "--- comparaison IVH detectees (type=IVH) ---"
python - "$FMT" "$RE/format/prompt5/"*_formated.csv >> "$LOG" 2>&1 <<'PY' || true
import sys, pandas as pd
f6, f5 = sys.argv[1], sys.argv[2]
for tag, f in [("prompt5", f5), ("prompt6", f6)]:
    d = pd.read_csv(f); n = d[d['type'].astype(str).str.upper().str.contains('IVH', na=False)]['ID'].nunique()
    print(f"  {tag}: {n} cas avec >=1 IVH detectee")
PY
log "===== PIPELINE PROMPT6 FINI -> $RE/metadata_prompt6/ ====="
