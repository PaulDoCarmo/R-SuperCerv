#!/bin/bash
# run_sweep15.sh -- Sweep stage-1 15 classes (12 structures + ICH+IVH+PHE), from scratch.
# N'ecrase PAS les modeles ICH-only (noms ich3_*, dossiers dedies). Sequentiel, 1 GPU, tmux.
#   tmux new -d -s sweep15 "bash run_sweep15.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$D/logs/sweep15"; mkdir -p "$LOGD"
MASTER="$LOGD/master.log"
source "$HOME/envs/rsuper/bin/activate"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl
cd "$REPO"
PORT=29610
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }

train_stage1(){ # <name> <data_root>
  local name="$1" dr="$2"
  local logf="$LOGD/${name}.log"
  if [ -f "$D/exp/ich/${name}/fold_0_best.pth" ]; then log "SKIP $name (best.pth deja la)"; return; fi
  log "START $name  (data=$dr) -> ${name}.log"
  if python train_ddp.py --dataset ich --model medformer --dimension 3d \
      --data_root "$dr" --classes_number 15 --save_destination "$D/ich3_augmented" \
      --cp_path "$D/exp/" --log_path "$D/log/" --unique_name "$name" \
      --crop_on_tumor --report_volume_loss_basic 0 --bf16 --gpu '0' --workers 8 --batch_size 3 \
      --epochs 60 --iter_per_epoch_override 250 --dist_url "tcp://127.0.0.1:$((PORT++))" \
      > "$logf" 2>&1; then
    log "OK    $name"
  else
    log "FAIL  $name (voir $logf)"
  fi
}

log "===== SWEEP15 stage-1 (from scratch, 15 classes) ====="
train_stage1 ich3_stage1_X25  "$D/subsets3/S_25"
train_stage1 ich3_stage1_X50  "$D/subsets3/S_50"
train_stage1 ich3_stage1_X100 "$D/subsets3/S_100"
train_stage1 ich3_stage1_X305 "$D/dataset_ich3_full_npz_trainval"
log "===== SWEEP15 FINI ====="
