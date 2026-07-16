#!/bin/bash
# run_mbh_eval.sh -- Eval MBH (detection) pour TOUS les modeles du sweep, sequentiel, 1 GPU.
# Idempotent (skip si CSV deja la), log/job, tolerant. stage1 -> best.pth ; stage2 -> latest.pth.
#   tmux new -d -s mbh "bash run_mbh_eval.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$D/logs/mbh"; mkdir -p "$LOGD" "$D/eval/mbh"
MASTER="$LOGD/mbh_master.log"
source "$HOME/envs/rsuper/bin/activate"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl
cd "$REPO"
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }

eval_one(){ # <label> <ckpt>
  local label="$1" ck="$2" out="$D/eval/mbh/${1}_mbh.csv"
  [ -f "$out" ] && { log "SKIP  $label (deja)"; return; }
  [ -f "$ck" ]  || { log "NO-CKPT $label ($ck)"; return; }
  log "START $label"
  if python eval_mbh.py --load "$ck" --npz_dir "$D/mbh/npz" --labels "$D/mbh/mbh_labels.csv" \
       --config config/ich/medformer_3d.yaml --class_list "$D/dataset_ich_full_npz/list/label_names.yaml" \
       --save_csv "$out" --gpu 0 --detect_min_ml 0.5 > "$LOGD/${label}.log" 2>&1; then
    log "OK    $label ($(grep -oE 'Sensibilite globale.*: [0-9.]+%' "$LOGD/${label}.log" | tail -1))"
  else
    log "FAIL  $label (voir $LOGD/${label}.log)"
  fi
}

log "===== MBH EVAL BATCH (12 modeles) ====="
while read label ck; do
  [ -z "$label" ] && continue
  eval_one "$label" "$ck"
done <<CASES
stage1_X25 $D/exp/ich/ich_stage1_X25/fold_0_best.pth
stage1_X50 $D/exp/ich/ich_stage1_X50/fold_0_best.pth
stage1_X100 $D/exp/ich/ich_stage1_X100/fold_0_best.pth
stage1_X305 $D/exp/ich/ich_stage1_13cls/fold_0_best.pth
stage2_X25_base0 $D/exp/ich_ufo/ich_stage2_X25_base0/fold_0_latest.pth
stage2_X25_all $D/exp/ich_ufo/ich_stage2_X25_all/fold_0_latest.pth
stage2_X50_base0 $D/exp/ich_ufo/ich_stage2_X50_base0/fold_0_latest.pth
stage2_X50_all $D/exp/ich_ufo/ich_stage2_X50_all/fold_0_latest.pth
stage2_X100_base0 $D/exp/ich_ufo/ich_stage2_X100_base0/fold_0_latest.pth
stage2_X100_all $D/exp/ich_ufo/ich_stage2_X100_all/fold_0_latest.pth
stage2_X305_base0 $D/exp/ich_ufo/ich_stage2_allmask_base0/fold_0_latest.pth
stage2_X305_all $D/exp/ich_ufo/ich_stage2_X305_all/fold_0_latest.pth
CASES
log "AGREGATION MBH"
python dataset_conversion/aggregate_mbh.py --mbh_dir "$D/eval/mbh" --out_dir "$D/eval/mbh_summary" >> "$MASTER" 2>&1 \
  && log "OK AGREGATION -> $D/eval/mbh_summary/" || log "FAIL AGREGATION"
log "===== MBH BATCH FINI ====="
