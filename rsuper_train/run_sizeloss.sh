#!/bin/bash
# run_sizeloss.sh -- Stage-2 "volume/size-loss ONLY" sur l'ICH, a partir du stage-1 15-cls X25.
# Loss = contrainte de taille Kervadec (sans H, PAS DE REGION, normalisee), tolerance 0.4.
# Matrice : 2 configs rapports {_0 seul (baseline0), _0+_1 (all)} x sweep lambda.
# Garde les masques (dataset ich_ufo = balance masques+rapports). Sequentiel, 1 GPU, tmux.
#   tmux new -d -s sizeloss "bash run_sizeloss.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$D/logs/sizeloss"; mkdir -p "$LOGD"
MASTER="$LOGD/master.log"
source "$HOME/envs/rsuper/bin/activate"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl
cd "$REPO"
PORT=29680
LAMBDAS="${LAMBDAS:-0.1 0.5 1.0}"        # sweep lambda (report_volume_loss_basic). Override: LAMBDAS="0.5" bash ...
TOL="${TOL:-0.4}"                        # tolerance data-driven
PRE="$D/exp/ich/ich3_stage1_X25/fold_0_best.pth"    # stage-1 15-cls, X25
MASKS="$D/subsets3/S_25"                 # 25 masques (branche pleinement supervisee)
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }

train(){ # <name> <ucsf_ids> <lambda>
  local name="$1" ids="$2" lam="$3"
  local logf="$LOGD/${name}.log"
  if [ -f "$D/exp/ich_ufo/${name}/fold_0_best.pth" ]; then log "SKIP $name (best.pth deja la)"; return; fi
  [ -f "$PRE" ] || { log "NO-PRETRAIN $PRE"; return; }
  [ -f "$ids" ] || { log "NO-IDS $ids"; return; }
  log "START $name (ids=$(basename "$ids") lambda=$lam tol=$TOL)"
  if python train_ddp.py --dataset ich_ufo --model medformer --dimension 3d \
      --data_root "$MASKS" --UFO_root "$D/dataset_ich_reports_npz" \
      --reports "$D/report_extraction/metadata/ich_per_tumor_metadata.csv" --ucsf_ids "$ids" \
      --classes_number 15 --pretrained "$PRE" --save_destination "$D/ich3_augmented_ufo" \
      --cp_path "$D/exp/" --log_path "$D/log/" --unique_name "$name" \
      --crop_on_tumor --loss size_last --report_volume_loss_basic "$lam" --volume_loss_tolerance "$TOL" \
      --bf16 --gpu '0' --workers 4 --batch_size 2 --crop_size 96 --epochs 60 --iter_per_epoch_override 250 \
      --lr 0.0001 --dist_url "tcp://127.0.0.1:$((PORT++))" > "$logf" 2>&1; then
    log "OK    $name"
  else
    log "FAIL  $name (voir $logf)"
  fi
}

log "===== SIZE-LOSS ONLY : X25 base, configs {_0, _0+_1} x lambda {$LAMBDAS} ====="
for lam in $LAMBDAS; do
  L=$(echo "$lam" | tr -d '.')
  train "sizeloss_X25_base0_L${L}" "$D/splits/reports_baseline0_train_ids.csv" "$lam"
  train "sizeloss_X25_all_L${L}"   "$D/splits/reports_all_train_ids.csv"       "$lam"
done
log "===== SIZE-LOSS SWEEP FINI ====="
