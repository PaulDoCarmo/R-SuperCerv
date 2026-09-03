#!/bin/bash
# run_pipeline_prompt6_rsna.sh -- Pipeline prompt6 pour les rapports RSNA (JSON->dictation).
# Produit un metadata RSNA ÉQUIVALENT au CHUM, dans des dossiers SÉPARÉS (report_extraction/rsna/
# + metadata_prompt6_rsna/) pour ne pas mélanger avec le CHUM (on pourra utiliser l'un, l'autre, ou les deux).
# Entrée = reports_rsna.csv (déjà construit depuis les JSON via csv_builders/build_reports_csv.py).
# ⚠ Étape LLM = vLLM Qwen-72B (~44 GB) : vérifie que le GPU est libre.  tmux new -d -s p6rsna "bash run_pipeline_prompt6_rsna.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
RE="$D/report_extraction"; HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSNA="$RE/rsna"; META="$RE/metadata_prompt6_rsna"; mkdir -p "$RSNA" "$META"
LOG="$D/logs/pipeline_prompt6_rsna.log"; mkdir -p "$D/logs"
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
[ -f "$RE/reports_rsna.csv" ] || { log "FAIL : $RE/reports_rsna.csv absent"; exit 1; }

log "===== PIPELINE PROMPT6 RSNA ($(($(wc -l < "$RE/reports_rsna.csv")-1)) rapports) ====="
log "1/5 serve vLLM + inference (prompt 6)"
if ! bash "$HERE/serve_llm_local.sh" "$RE/reports_rsna.csv" "$RSNA/results.csv" qwen2-5-72b-awq 0 0.95 6 >> "$LOG" 2>&1; then
  log "FAIL serve/inference"; exit 1; fi
RAW=$(ls -t "$RSNA/raw/prompt6/"*_prompt6.csv 2>/dev/null | head -1)
[ -f "$RAW" ] || { log "FAIL : raw introuvable"; exit 1; }; log "  raw = $RAW"

source "$HOME/envs/report/bin/activate"
log "2/5 postprocess"
python "$HERE/postprocess.py" --input "$RAW" --output_root "$RSNA/post_processed" >> "$LOG" 2>&1 || { log "FAIL postprocess"; exit 1; }
POST=$(ls -t "$RSNA/post_processed/prompt6/"*prompt6*.csv 2>/dev/null | head -1)
log "3/5 format_metrics --all"
python "$HERE/format_metrics.py" --input "$POST" --output_root "$RSNA/format" --all >> "$LOG" 2>&1 || { log "FAIL format"; exit 1; }
FMT=$(ls -t "$RSNA/format/prompt6/"*_formated.csv 2>/dev/null | head -1)
[ -f "$FMT" ] || { log "FAIL : format introuvable"; exit 1; }; log "  format = $FMT"
log "4/5 metadata (equal_A) -> $META"
python "$HERE/report_to_rsuper_metadata.py" --input "$FMT" --out_dir "$META" --types ICH --third_axis equal_A >> "$LOG" 2>&1 || { log "FAIL metadata"; exit 1; }
log "5/5 ivh_flags"
python "$HERE/build_ivh_flags.py" --input "$FMT" --output "$META/ivh_flags.csv" >> "$LOG" 2>&1 || { log "FAIL ivh_flags"; exit 1; }
log "===== FINI -> $META/ {ich_per_tumor_metadata, ich_per_CT_metadata, ivh_flags}.csv ====="
python - "$FMT" >> "$LOG" 2>&1 <<'PY' || true
import sys,pandas as pd
d=pd.read_csv(sys.argv[1]); ivh=d[d['type'].astype(str).str.upper().str.contains('IVH',na=False)]['ID'].nunique()
print(f"  scans formatés={d['ID'].nunique()} | dont IVH+={ivh}")
PY
