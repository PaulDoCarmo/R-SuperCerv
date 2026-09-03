#!/bin/bash
# run_pipeline_prompt6_incremental.sh -- Ajoute un NOUVEAU lot de rapports au pipeline prompt6
# SANS re-inférer les rapports déjà traités (LLM = étape coûteuse). Détecte le net-nouveau par
# différence d'ID avec le raw existant, ne passe QUE ceux-là au LLM, puis fusionne et régénère
# raw/post/format/metadata (275) en écrasant les fichiers canoniques APRÈS backup horodaté.
#
# 3 phases :
#   A (sans GPU) : construit reports_new.csv = uniquement les ID absents du raw prompt6 existant.
#   B (GPU ~44GB) : vLLM Qwen-72B prompt6 sur reports_new.csv -> raw NEW séparé.
#   C (sans GPU) : concat raw(existant+NEW) -> postprocess -> format -> metadata + ivh_flags.
#
# Usage :
#   bash run_pipeline_prompt6_incremental.sh prepare      # phase A seule (teste le net-nouveau)
#   bash run_pipeline_prompt6_incremental.sh infer        # phase B (exige GPU libre ~44GB)
#   bash run_pipeline_prompt6_incremental.sh merge        # phase C (fusion + metadata)
#   bash run_pipeline_prompt6_incremental.sh all          # A puis B puis C
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
RE="$D/report_extraction"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEW_TXT="${NEW_TXT:-$D/CT-Reports/FINAL_275_2026_08_06/matched_reports}"   # dossier .txt du nouveau lot
PID=6
TAG="Qwen2.5-72B-Instruct-AWQ"
EXIST_RAW="$RE/raw/prompt6/results_${TAG}_prompt6.csv"          # raw canonique (à étendre)
NEW_RAW="$RE/raw/prompt6/results_NEWBATCH_${TAG}_prompt6.csv"   # raw du seul nouveau lot (pristine)
REPORTS_NEW="$RE/reports_newbatch.csv"                          # entrée LLM = net-nouveau seulement
STAMP="$(date +%Y%m%d_%H%M%S)"
BK="$RE/_backup_incremental_$STAMP"
LOG="$D/logs/pipeline_prompt6_incremental.log"; mkdir -p "$D/logs"
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
act(){ source "$HOME/envs/report/bin/activate"; }

phase_A(){
  log "===== PHASE A : détection du net-nouveau (sans GPU) ====="
  [ -d "$NEW_TXT" ] || { log "FAIL : dossier .txt introuvable : $NEW_TXT"; return 1; }
  [ -f "$EXIST_RAW" ] || { log "FAIL : raw prompt6 existant introuvable : $EXIST_RAW"; return 1; }
  act
  # 1) reports.csv du lot COMPLET (275)
  python "$HERE/csv_builders/build_reports_csv_from_txt.py" "$NEW_TXT" "$RE" --output-name reports_newbatch_all.csv >>"$LOG" 2>&1 \
    || { log "FAIL build_reports_csv"; return 1; }
  # 2) filtre -> uniquement les ID ABSENTS du raw existant (= net-nouveau, pas de redondance LLM)
  python - "$RE/reports_newbatch_all.csv" "$EXIST_RAW" "$REPORTS_NEW" <<'PY' >>"$LOG" 2>&1 || { log "FAIL filtre net-nouveau"; return 1; }
import sys, pandas as pd
allrep, exist_raw, out = sys.argv[1:4]
rep = pd.read_csv(allrep)
done = set(pd.read_csv(exist_raw)["ID"].astype(str))
new = rep[~rep["ID"].astype(str).isin(done)].copy()
new.to_csv(out, index=False)
print(f"lot complet={len(rep)} | déjà traités(raw)={len(done)} | NET-NOUVEAU={len(new)} -> {out}")
PY
  local n; n=$(python -c "import pandas as pd;print(len(pd.read_csv('$REPORTS_NEW')))" 2>/dev/null)
  log "  net-nouveau à inférer : ${n:-?} rapports -> $REPORTS_NEW"
  [ "${n:-0}" -gt 0 ] || log "  RIEN de nouveau (déjà tout traité) — phases B/C inutiles."
}

phase_B(){
  log "===== PHASE B : inférence LLM prompt6 sur le net-nouveau (GPU ~44GB) ====="
  [ -f "$REPORTS_NEW" ] || { log "FAIL : $REPORTS_NEW absent (lancer 'prepare' d'abord)"; return 1; }
  local n; n=$(python -c "import pandas as pd;print(len(pd.read_csv('$REPORTS_NEW')))" 2>/dev/null)
  [ "${n:-0}" -gt 0 ] || { log "  net-nouveau vide -> skip inférence."; return 0; }
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  log "  VRAM libre : ${free} MiB (besoin ~44000)"
  [ "${free:-0}" -ge 44000 ] || { log "FAIL : pas assez de VRAM libre (attendre que le GPU se libère)"; return 1; }
  rm -f "$NEW_RAW"                                    # idempotence : raw NEW propre
  # serve_llm_local.sh ajoute _<TAG> au base de SAVE_PATH -> raw/prompt6/results_NEWBATCH_<TAG>_prompt6.csv
  bash "$HERE/serve_llm_local.sh" "$REPORTS_NEW" "$RE/results_NEWBATCH.csv" qwen2-5-72b-awq 0 0.95 6 >>"$LOG" 2>&1 \
    || { log "FAIL serve/inference"; return 1; }
  [ -f "$NEW_RAW" ] || { log "FAIL : raw NEW introuvable ($NEW_RAW)"; return 1; }
  log "  raw NEW : $NEW_RAW ($(($(wc -l < "$NEW_RAW") - 1)) lignes-lésion)"
}

phase_C(){
  log "===== PHASE C : fusion + regen post/format/metadata (sans GPU) ====="
  [ -f "$NEW_RAW" ] || { log "FAIL : raw NEW absent (lancer 'infer' d'abord)"; return 1; }
  act
  mkdir -p "$BK"
  log "  backup -> $BK"
  cp -a "$EXIST_RAW" "$BK/" 2>/dev/null || true
  cp -a "$RE/post_processed/prompt6" "$BK/post_processed_prompt6" 2>/dev/null || true
  cp -a "$RE/format/prompt6" "$BK/format_prompt6" 2>/dev/null || true
  cp -a "$RE/metadata_prompt6" "$BK/metadata_prompt6" 2>/dev/null || true
  # 1) concat raw existant + NEW, dédup par (ID, Lesion Index) en gardant l'existant -> écrase le raw canonique
  python - "$EXIST_RAW" "$NEW_RAW" <<'PY' >>"$LOG" 2>&1 || { log "FAIL concat raw"; return 1; }
import sys, pandas as pd
exist, new = sys.argv[1], sys.argv[2]
a = pd.read_csv(exist); b = pd.read_csv(new)
before = a["ID"].nunique()
both = pd.concat([a, b], ignore_index=True).drop_duplicates(subset=["ID", "Lesion Index"], keep="first")
both.to_csv(exist, index=False)
print(f"raw fusionné : {before} -> {both['ID'].nunique()} scans ({len(both)} lignes-lésion) -> {exist}")
PY
  # 2) postprocess -> format --all -> metadata (ICH) -> ivh_flags  (tous sur le raw canonique fusionné)
  python "$HERE/postprocess.py" --input "$EXIST_RAW" --output_root "$RE/post_processed" >>"$LOG" 2>&1 || { log "FAIL postprocess"; return 1; }
  POST=$(ls -t "$RE/post_processed/prompt6/"*prompt6*.csv 2>/dev/null | head -1)
  python "$HERE/format_metrics.py" --input "$POST" --output_root "$RE/format" --all >>"$LOG" 2>&1 || { log "FAIL format"; return 1; }
  FMT=$(ls -t "$RE/format/prompt6/"*_formated.csv 2>/dev/null | head -1)
  python "$HERE/report_to_rsuper_metadata.py" --input "$FMT" --out_dir "$RE/metadata_prompt6" --types ICH --third_axis equal_A >>"$LOG" 2>&1 || { log "FAIL metadata"; return 1; }
  python "$HERE/build_ivh_flags.py" --input "$FMT" --output "$RE/metadata_prompt6/ivh_flags.csv" >>"$LOG" 2>&1 || { log "FAIL ivh_flags"; return 1; }
  log "  metadata_prompt6/ régénéré (per_tumor, per_CT, ivh_flags) sur $(python -c "import pandas as pd;print(pd.read_csv('$FMT')['ID'].nunique())") scans"
  log "===== FUSION FINIE. Backup : $BK ====="
}

case "${1:-all}" in
  prepare) phase_A ;;
  infer)   phase_B ;;
  merge)   phase_C ;;
  all)     phase_A && phase_B && phase_C ;;
  *) echo "usage: $0 {prepare|infer|merge|all}"; exit 1 ;;
esac
