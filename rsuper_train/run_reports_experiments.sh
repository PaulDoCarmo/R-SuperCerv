#!/bin/bash
# run_reports_experiments.sh -- 5 experiences de supervision-RAPPORTS (ICH), sur les 85 rapports
# ANCRES (pseudo-masque non-vide, gate ratio [0.15,1.3]). Ancre haute-precision (erosion 4mm) +
# crop PSEUDO-CENTRE (capture ~100% -> volume size-loss correct). val OFF -> fold_0_latest.pth.
#
#   1. rexp_scratch_sl05   : FROM SCRATCH, reports-only (UFO_only), ancre + size-loss lambda 0.5
#   2. rexp_scratch_anchor : FROM SCRATCH, reports-only, ANCRE SEULE (lambda 0, pas de size-loss)
#   3. rexp_X5_sl05        : pretrain stage-1 X5  + masques S_5  + rapports, ancre + size-loss 0.5
#   4. rexp_X10_sl05       : pretrain stage-1 X10 + masques S_10 + rapports, ancre + size-loss 0.5
#   5. rexp_X25_sl05       : pretrain stage-1 X25 + masques S_25 + rapports, ancre + size-loss 0.5
# Puis EVAL des 5 (+ baselines stage-1) : Dice ICH/IVH/PHE sur RSNA-test + GT-rapports CHUM.
#   tmux new -d -s rexp "bash run_reports_experiments.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$D/logs/reports_exp"; mkdir -p "$LOGD"; MASTER="$LOGD/master.log"
source "$HOME/envs/rsuper/bin/activate"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl
cd "$REPO"
PORT=29810
CROP="${CROP:-128}"                                              # patch (capture ~100%)
TOL="${TOL:-0.4}"; LAM="${LAM:-0.5}"
UFO="$D/dataset_ich_reports_pseudo_gated_npz"                    # pseudo GATE (ancre)
META="$D/report_extraction/metadata/ich_per_tumor_metadata.csv"
IDS="$D/splits/reports_anchored_train_ids.csv"                   # 85 rapports ancres
for f in "$META" "$IDS"; do [ -f "$f" ] || { echo "manque $f"; exit 1; }; done
[ -d "$UFO" ] || { echo "manque $UFO"; exit 1; }
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }

COMMON="--dataset ich_ufo --model medformer --dimension 3d --UFO_root $UFO --reports $META \
  --ucsf_ids $IDS --classes_number 15 --save_destination $D/ich3_augmented_ufo --cp_path $D/exp/ \
  --log_path $D/log/ --crop_on_tumor --loss size_anchor_last --volume_loss_tolerance $TOL --bf16 \
  --gpu 0 --workers 4 --batch_size 2 --crop_size $CROP --epochs 60 --iter_per_epoch_override 250 --val_freq 999"

done_ckpt(){ [ -f "$D/exp/ich_ufo/$1/fold_0_latest.pth" ]; }

train_scratch(){ # <name> <lambda>  -- from scratch, reports-only
  local name="$1" lam="$2" logf="$LOGD/$1.log"
  if done_ckpt "$name"; then log "SKIP $name"; return; fi
  log "START $name (SCRATCH reports-only UFO_only, lambda=$lam, crop=$CROP)"
  if python train_ddp.py $COMMON --UFO_only --data_root "$D/subsets3/S_5" \
      --unique_name "$name" --report_volume_loss_basic "$lam" \
      --dist_url "tcp://127.0.0.1:$((PORT++))" > "$logf" 2>&1; then log "OK    $name"; else log "FAIL  $name (voir $logf)"; fi
}
train_ft(){ # <name> <pretrain_ckpt> <masks_dir> <lambda>
  local name="$1" pre="$2" masks="$3" lam="$4" logf="$LOGD/$1.log"
  if done_ckpt "$name"; then log "SKIP $name"; return; fi
  [ -f "$pre" ] || { log "NO-PRETRAIN $pre"; return; }
  log "START $name (pretrain=$(basename "$(dirname "$pre")") masks=$(basename "$masks") lambda=$lam crop=$CROP)"
  if python train_ddp.py $COMMON --pretrained "$pre" --data_root "$masks" \
      --unique_name "$name" --report_volume_loss_basic "$lam" --lr 0.0001 \
      --dist_url "tcp://127.0.0.1:$((PORT++))" > "$logf" 2>&1; then log "OK    $name"; else log "FAIL  $name (voir $logf)"; fi
}

log "===== 5 EXPERIENCES SUPERVISION-RAPPORTS (85 ancres, crop $CROP) ====="
train_scratch rexp_scratch_sl05   "$LAM"
train_scratch rexp_scratch_anchor 0
train_ft rexp_X5_sl05  "$D/exp/ich/ich3_stage1_X5/fold_0_latest.pth"  "$D/subsets3/S_5"  "$LAM"
train_ft rexp_X10_sl05 "$D/exp/ich/ich3_stage1_X10/fold_0_latest.pth" "$D/subsets3/S_10" "$LAM"
train_ft rexp_X25_sl05 "$D/exp/ich/ich3_stage1_X25/fold_0_best.pth"   "$D/subsets3/S_25" "$LAM"

log "===== EVAL des 5 + baselines stage-1 ====="
python eval_sizeloss.py "rexp_*" "$D/eval/reports_exp_eval.csv" > "$LOGD/eval.log" 2>&1 && log "OK eval -> eval/reports_exp_eval.csv" || log "FAIL eval (voir $LOGD/eval.log)"
log "===== FINI ====="
