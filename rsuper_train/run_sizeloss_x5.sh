#!/bin/bash
# run_sizeloss_x5.sh -- EXPERIENCE FINALE size-loss : base bas-regime X5 + FULL rapports.
# Objectif : a X=5 masques (la ou le stage-1 ICH est faible, CHUM Dice ~0.785), tester si la
# supervision faible par rapports (size-loss Kervadec + ANCRE pseudo-masque haute-precision)
# ameliore l'ICH sur les cas-rapports.
#
# Ameliorations vs run_sizeloss_anchor.sh (etudes data-driven 65 GT) :
#   - ANCRE : pseudo-masque GATE (ratio vol/Vr in [0.15,1.3]) + erosion 4mm cote loss
#     -> precision voxels-ancre ~1.0 (0.6% faux). cf make_ich_pseudomasks.py + losses_foundation.
#   - PATCH : crop 128 (au lieu de 96) -> capture ~100% des lesions (CHUM tiennent dans 128mm)
#     -> supprime la troncature du volume identifie sur le patch.
#   - Rapports : FULL (reports_all = _0+_1).
# Base = stage-1 X5 (fold_0_latest.pth, pas de best.pth bruite). Sequentiel, 1 GPU, tmux.
#   tmux new -d -s slx5 "bash run_sizeloss_x5.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$D/logs/sizeloss_x5"; mkdir -p "$LOGD"
MASTER="$LOGD/master.log"
source "$HOME/envs/rsuper/bin/activate"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl
cd "$REPO"
PORT=29760
LAMBDAS="${LAMBDAS:-0.1 0.5 1.0}"
TOL="${TOL:-0.4}"
CROP="${CROP:-128}"                                   # patch : capture ~100% des lesions
PRE="$D/exp/ich/ich3_stage1_X5/fold_0_latest.pth"     # base bas-regime, dernier checkpoint
MASKS="$D/subsets3/S_5"                                # 5 masques (branche supervisee)
UFO="$D/dataset_ich_reports_pseudo_gated_npz"          # pseudo-masque GATE (ancre haute-precision)
REPORTS_META="${REPORTS_META:-$D/report_extraction/metadata/ich_per_tumor_metadata.csv}"
IDS="${IDS:-$D/splits/reports_all_train_ids.csv}"      # FULL rapports
[ -f "$REPORTS_META" ] || { echo "REPORTS_META introuvable: $REPORTS_META"; exit 1; }
[ -f "$PRE" ] || { echo "PRETRAIN X5 introuvable: $PRE"; exit 1; }
[ -d "$UFO" ] || { echo "UFO gate introuvable: $UFO"; exit 1; }
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }

train(){ # <name> <lambda>
  local name="$1" lam="$2"
  local logf="$LOGD/${name}.log"
  if [ -f "$D/exp/ich_ufo/${name}/fold_0_best.pth" ] || [ -f "$D/exp/ich_ufo/${name}/fold_0_latest.pth" ]; then log "SKIP $name"; return; fi
  log "START $name (lambda=$lam tol=$TOL crop=$CROP anchor=pseudo-gate full-rapports)"
  if python train_ddp.py --dataset ich_ufo --model medformer --dimension 3d \
      --data_root "$MASKS" --UFO_root "$UFO" \
      --reports "$REPORTS_META" --ucsf_ids "$IDS" \
      --classes_number 15 --pretrained "$PRE" --save_destination "$D/ich3_augmented_ufo" \
      --cp_path "$D/exp/" --log_path "$D/log/" --unique_name "$name" \
      --crop_on_tumor --loss size_anchor_last --report_volume_loss_basic "$lam" --volume_loss_tolerance "$TOL" \
      --bf16 --gpu '0' --workers 4 --batch_size 2 --crop_size "$CROP" --epochs 60 --iter_per_epoch_override 250 \
      --lr 0.0001 --val_freq 999 --dist_url "tcp://127.0.0.1:$((PORT++))" > "$logf" 2>&1; then
    log "OK    $name"
  else
    log "FAIL  $name (voir $logf)"
  fi
}

log "===== SIZE-LOSS X5 + FULL rapports (ancre gate + crop $CROP) : lambda {$LAMBDAS} ====="
for lam in $LAMBDAS; do
  L=$(echo "$lam" | tr -d '.')
  train "slx5_all_L${L}" "$lam"
done
log "===== SIZE-LOSS X5 FINI ====="
