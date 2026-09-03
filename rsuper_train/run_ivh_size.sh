#!/bin/bash
# run_ivh_size.sh -- SIZE-LOSS IVH "ORACLE" (cheat) : teste si une supervision VOLUME-seul corrige
# la sous-segmentation IVH. Cible = volume du masque _0 (bande ±tol), ancre R_pool = ventricule+15mm,
# head FINAL seg[0]. Co-training X25 masques RSNA (seg-loss complète = le H(S) de Kervadec).
# Réutilise training/size_loss.volume_size_loss (region_mask + normalize). Mot-clé loss : "ivhvol"
# (SANS 'size' -> n'active PAS le chemin size-loss ICH). Cas-rapport = _0 seulement (masques valides).
#
# 2 setups = régime epoch/it/LR (teste la sur-dose du régime iter-fixe sur petit data) :
#   heavy : 60 ep, lr 1e-4 (régime habituel)   |   light : 25 ep, lr 5e-5 (fine-tune conservateur)
#   tmux new -d -s ivhsize "bash run_ivh_size.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$D/logs/ivh_size"; mkdir -p "$LOGD"; MASTER="$LOGD/master.log"
source "$HOME/envs/rsuper/bin/activate"; export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl
cd "$REPO"
PORT=29840
LAMBDA="${LAMBDA:-0.5}"          # <- fixé après calibration smoke
CROP="${CROP:-144}"              # 144 (÷16) ; passer 160 si le modèle exige ÷32
DIL="${DIL:-15}"; TOL="${TOL:-0.1}"
PRE="$D/exp/ich/ich3_stage1_X25/fold_0_best.pth"
MASKS="$D/subsets3/S_25"
UFO="$D/dataset_ich_reports_npz"
META="$D/report_extraction/metadata_prompt6/ich_per_tumor_metadata.csv"
ORACLE="$D/report_extraction/metadata_prompt6/ivh_oracle_volumes.csv"
IDS="$D/splits/reports_baseline0_train_ids.csv"     # _0 uniquement (masques IVH valides)
for f in "$PRE" "$META" "$ORACLE" "$IDS"; do [ -f "$f" ] || { echo "manque $f"; exit 1; }; done
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }

train(){ # <name> <epochs> <lr> <ramp>
  local name="$1" ep="$2" lr="$3" ramp="$4"
  if [ -f "$D/exp/ich_ufo/$name/fold_0_latest.pth" ]; then log "SKIP $name"; return; fi
  log "START $name (crop=$CROP lambda=$LAMBDA ep=$ep lr=$lr ramp=$ramp dil=$DIL tol=$TOL)"
  if python train_ddp.py --dataset ich_ufo --model medformer --dimension 3d \
      --data_root "$MASKS" --UFO_root "$UFO" --reports "$META" --ucsf_ids "$IDS" \
      --loss ivhvol --ivh_size_oracle "$ORACLE" --ivh_size_lambda "$LAMBDA" \
      --ivh_size_dil "$DIL" --ivh_size_tol "$TOL" --ivh_ramp_epochs "$ramp" \
      --classes_number 15 --pretrained "$PRE" --save_destination "$D/ich3_augmented_ufo" \
      --cp_path "$D/exp/" --log_path "$D/log/" --unique_name "$name" \
      --crop_on_tumor --bf16 --gpu '0' --workers 4 --batch_size 2 --crop_size "$CROP" \
      --epochs "$ep" --iter_per_epoch_override 250 --val_freq 999 --lr "$lr" \
      --dist_url "tcp://127.0.0.1:$((PORT++))" > "$LOGD/$name.log" 2>&1; then
    log "OK    $name"
  else
    log "FAIL  $name (voir $LOGD/$name.log)"
  fi
}

log "===== SIZE-LOSS ORACLE IVH : 2 setups régime (lambda=$LAMBDA, crop=$CROP) ====="
train "ivhsize_X25_heavy" 60 0.0001  10
train "ivhsize_X25_light" 25 0.00005 5
log "===== FINI — reste : éval Dice IVH + vol ratio held-out (CHUM _0 test + RSNA) ====="
