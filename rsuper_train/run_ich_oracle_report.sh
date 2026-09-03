#!/bin/bash
# run_ich_oracle_report.sh -- ICH size-loss : ORACLE (volume masque) vs REPORT (ABC/2), sur les 38 _0
# ancrés (mêmes cas). Config identique à run_reports_experiments (size_anchor_last, seg_loss 1,
# report_volume_loss_basic 0.5, crop 128, 60ep, pseudo-gated npz pour l'ancre). X10 seulement (pas X5/X25).
# ORDRE : attend fin de l'IVH -> PHASE1 tol0.4 (5 runs) + éval -> PHASE2 tol0.1 (4 runs) + éval.
#   tmux new -d -s ichor "bash run_ich_oracle_report.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$D/logs/reports_exp"; mkdir -p "$LOGD"; MASTER="$LOGD/oracle_report.log"
source "$HOME/envs/rsuper/bin/activate"; export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl
cd "$REPO"
UFO="$D/dataset_ich_reports_pseudo_gated_npz"                     # pseudo-masque = ancre
UCSF="$D/splits/reports_anchored38_0_ids.csv"                     # 38 _0 (oracle dispo)
META_REP="$D/report_extraction/metadata/ich_per_tumor_metadata.csv"          # volume ABC/2 (report)
META_ORAC="$D/report_extraction/metadata/ich_per_tumor_metadata_ORACLE38.csv" # volume masque (oracle)
X10="$D/exp/ich/ich3_stage1_X10/fold_0_latest.pth"
PORT=29900
for f in "$UCSF" "$META_REP" "$META_ORAC" "$X10"; do [ -f "$f" ] || { echo "manque $f"; exit 1; }; done
[ -d "$UFO" ] || { echo "manque $UFO"; exit 1; }
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }
COMMON="--dataset ich_ufo --model medformer --dimension 3d --UFO_root $UFO --ucsf_ids $UCSF \
  --classes_number 15 --save_destination $D/ich3_augmented_ufo --cp_path $D/exp/ --log_path $D/log/ \
  --crop_on_tumor --loss size_anchor_last --bf16 --gpu 0 --workers 4 --batch_size 2 --crop_size 128 \
  --epochs 60 --iter_per_epoch_override 250 --val_freq 999"

t_scratch(){ # <name> <meta> <lam> <tol>
  local name="$1"; [ -f "$D/exp/ich_ufo/$name/fold_0_latest.pth" ] && { log "SKIP $name"; return; }
  log "TRAIN $name (scratch, meta=$(basename "$2") lam=$3 tol=$4)"
  python train_ddp.py $COMMON --UFO_only --data_root "$D/subsets3/S_5" --reports "$2" \
    --volume_loss_tolerance "$4" --unique_name "$name" --report_volume_loss_basic "$3" \
    --dist_url "tcp://127.0.0.1:$((PORT++))" > "$LOGD/$name.log" 2>&1 && log "OK $name" || log "FAIL $name"
}
t_ft(){ # <name> <meta> <lam> <tol>
  local name="$1"; [ -f "$D/exp/ich_ufo/$name/fold_0_latest.pth" ] && { log "SKIP $name"; return; }
  log "TRAIN $name (FT X10, meta=$(basename "$2") lam=$3 tol=$4)"
  python train_ddp.py $COMMON --pretrained "$X10" --data_root "$D/subsets3/S_10" --reports "$2" \
    --volume_loss_tolerance "$4" --unique_name "$name" --report_volume_loss_basic "$3" --lr 0.0001 \
    --dist_url "tcp://127.0.0.1:$((PORT++))" > "$LOGD/$name.log" 2>&1 && log "OK $name" || log "FAIL $name"
}

log "===== ATTENTE fin IVH (chumL025) ====="
while tmux has-session -t chumL025 2>/dev/null; do sleep 120; done
log "  IVH fini, GPU libre -> démarrage ICH"

log "===== PHASE 1 (tol 0.4) : oracle vs report, ancre-seule + ancre+size + X10 ====="
t_scratch rexp38_anchor       "$META_REP"  0    0.4          # ancre seule (pas de volume)
t_scratch rexp38_scratch_orac_t40 "$META_ORAC" 0.5 0.4
t_scratch rexp38_scratch_rep_t40  "$META_REP"  0.5 0.4
t_ft      rexp38_X10_orac_t40     "$META_ORAC" 0.5 0.4
t_ft      rexp38_X10_rep_t40      "$META_REP"  0.5 0.4
log "EVAL phase1 (tol0.4)"
python eval_sizeloss.py "rexp38_*" "$D/eval/ich_oracle_report_t40.csv" > "$LOGD/eval_t40.log" 2>&1 && log "OK eval t40" || log "FAIL eval t40"

log "===== PHASE 2 (tol 0.1) : oracle vs report, ancre+size + X10 ====="
t_scratch rexp38_scratch_orac_t10 "$META_ORAC" 0.5 0.1
t_scratch rexp38_scratch_rep_t10  "$META_REP"  0.5 0.1
t_ft      rexp38_X10_orac_t10     "$META_ORAC" 0.5 0.1
t_ft      rexp38_X10_rep_t10      "$META_REP"  0.5 0.1
log "EVAL phase2 (tol0.1 + tout)"
python eval_sizeloss.py "rexp38_*" "$D/eval/ich_oracle_report_all.csv" > "$LOGD/eval_all.log" 2>&1 && log "OK eval all" || log "FAIL eval all"
log "===== FINI ====="
