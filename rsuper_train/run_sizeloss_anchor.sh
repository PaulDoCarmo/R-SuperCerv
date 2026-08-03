#!/bin/bash
# run_sizeloss_anchor.sh -- Stage-2 size-loss AVEC ANCRE SPATIALE (pseudo-masque ICH par
# intensite, terme H de Kervadec). Le canal ICH des cas-rapports est supervise par le
# pseudo-masque (BCE+Dice) + la contrainte de taille C(V_S). Teste si l'ancre empeche le
# collapse observe sans ancre. Base = stage-1 15-cls X25. Sequentiel, 1 GPU, tmux.
#   tmux new -d -s slanchor "bash run_sizeloss_anchor.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$D/logs/sizeloss_anchor"; mkdir -p "$LOGD"
MASTER="$LOGD/master.log"
source "$HOME/envs/rsuper/bin/activate"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl
cd "$REPO"
PORT=29720
LAMBDAS="${LAMBDAS:-0.1 0.5 1.0}"
TOL="${TOL:-0.4}"
PRE="$D/exp/ich/ich3_stage1_X25/fold_0_best.pth"
MASKS="$D/subsets3/S_25"
UFO="$D/dataset_ich_reports_pseudo_npz"        # rapports AVEC pseudo-masque ICH (ancre)
# Metadonnees rapports : configurable. Defaut = prompt5 ; pour prompt6 (meilleure detection IVH) :
#   REPORTS_META="$D/report_extraction/metadata_prompt6/ich_per_tumor_metadata.csv" bash run_...
REPORTS_META="${REPORTS_META:-$D/report_extraction/metadata/ich_per_tumor_metadata.csv}"
[ -f "$REPORTS_META" ] || { echo "REPORTS_META introuvable: $REPORTS_META"; exit 1; }
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }

train(){ # <name> <ucsf_ids> <lambda>
  local name="$1" ids="$2" lam="$3"
  local logf="$LOGD/${name}.log"
  if [ -f "$D/exp/ich_ufo/${name}/fold_0_best.pth" ]; then log "SKIP $name"; return; fi
  [ -f "$PRE" ] || { log "NO-PRETRAIN $PRE"; return; }
  [ -d "$UFO" ] || { log "NO-UFO $UFO"; return; }
  log "START $name (ids=$(basename "$ids") lambda=$lam anchor=pseudo)"
  if python train_ddp.py --dataset ich_ufo --model medformer --dimension 3d \
      --data_root "$MASKS" --UFO_root "$UFO" \
      --reports "$REPORTS_META" --ucsf_ids "$ids" \
      --classes_number 15 --pretrained "$PRE" --save_destination "$D/ich3_augmented_ufo" \
      --cp_path "$D/exp/" --log_path "$D/log/" --unique_name "$name" \
      --crop_on_tumor --loss size_anchor_last --report_volume_loss_basic "$lam" --volume_loss_tolerance "$TOL" \
      --bf16 --gpu '0' --workers 4 --batch_size 2 --crop_size 96 --epochs 60 --iter_per_epoch_override 250 \
      --lr 0.0001 --dist_url "tcp://127.0.0.1:$((PORT++))" > "$logf" 2>&1; then
    log "OK    $name"
  else
    log "FAIL  $name (voir $logf)"
  fi
}

log "===== SIZE-LOSS + ANCRE pseudo-masque : X25 base, {_0, _0+_1} x lambda {$LAMBDAS} ====="
for lam in $LAMBDAS; do
  L=$(echo "$lam" | tr -d '.')
  train "slanchor_X25_base0_L${L}" "$D/splits/reports_baseline0_train_ids.csv" "$lam"
  train "slanchor_X25_all_L${L}"   "$D/splits/reports_all_train_ids.csv"       "$lam"
done
log "===== SIZE-LOSS ANCHOR SWEEP FINI ====="
