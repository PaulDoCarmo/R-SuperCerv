#!/bin/bash
# run_ivh_presence.sh -- SWEEP feasibility : loss de PRÉSENCE IVH (détection faible-supervisée
# par rapports). Fine-tune conjoint X25 : masques RSNA (seg-loss anti-dérive, aug ON) + rapports
# CHUM (loss de présence sur le canal IVH, aug d'intensité OFF pour cohérence avec la calibration).
# Sweep sur lambda (poids de la présence). Détail : training/ivh_presence_loss.py.
#
# Design figé (études 1-4) : R_pool=ventricule+10mm, R_supp=+15mm, γ=10, λ⁻=0.9, pas de p_ICH/seed.
# τ⁺/τ⁻ = calibrés sur le R_pool EXACT du training (dilate_volume, cube) -> voir TAU_POS/TAU_NEG.
# ⚠ FEASIBILITY : n petit (29 IVH+/36 IVH−). Preuve de mécanisme, pas résultat final.
#   tmux new -d -s ivhpres "bash run_ivh_presence.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$D/logs/ivh_presence"; mkdir -p "$LOGD"; MASTER="$LOGD/master.log"
source "$HOME/envs/rsuper/bin/activate"; export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl
cd "$REPO"
PORT=29815
CROP="${CROP:-128}"
LAMBDAS="${LAMBDAS:-0.05 0.1 0.3}"        # sweep du poids de la présence
RAMP="${RAMP:-10}"
TAU_POS="${TAU_POS:-0.42}"                 # <- injecter les τ recalibrés (cube) ici
TAU_NEG="${TAU_NEG:-0.30}"
LN="${LN:-0.9}"                            # lambda- (poids côté négatif). LN=0 => positive-only
BETA="${BETA:-1.0}"                        # poids suppression de fond. BETA=0 => pas de L_bkg
TAG="${TAG:-}"                             # suffixe de nom d'expérience (ex: _pos)
PRE="$D/exp/ich/ich3_stage1_X25/fold_0_best.pth"
MASKS="$D/subsets3/S_25"
UFO="$D/dataset_ich_reports_npz"
META="$D/report_extraction/metadata_prompt6/ich_per_tumor_metadata.csv"
FLAGS="$D/report_extraction/metadata_prompt6/ivh_flags.csv"
IDS="$D/splits/reports_all_train_ids.csv"
for f in "$PRE" "$META" "$FLAGS" "$IDS"; do [ -f "$f" ] || { echo "manque $f"; exit 1; }; done
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }

train(){ # <lambda>
  local lam="$1" name="ivhpres_X25${TAG}_L$(echo "$1"|tr -d '.')"
  if [ -f "$D/exp/ich_ufo/$name/fold_0_latest.pth" ]; then log "SKIP $name"; return; fi
  log "START $name (X25 crop=$CROP lambda=$lam ramp=$RAMP τ+=$TAU_POS τ-=$TAU_NEG λ-=$LN β=$BETA, aug-intensité OFF)"
  if python train_ddp.py --dataset ich_ufo --model medformer --dimension 3d \
      --data_root "$MASKS" --UFO_root "$UFO" --reports "$META" --ucsf_ids "$IDS" \
      --ivh_flags "$FLAGS" --loss ivh_presence \
      --ivh_lambda "$lam" --ivh_beta "$BETA" --ivh_gamma 10 --ivh_tau_pos "$TAU_POS" --ivh_tau_neg "$TAU_NEG" \
      --ivh_lambda_neg "$LN" --ivh_ramp_epochs "$RAMP" \
      --classes_number 15 --pretrained "$PRE" --save_destination "$D/ich3_augmented_ufo" \
      --cp_path "$D/exp/" --log_path "$D/log/" --unique_name "$name" \
      --crop_on_tumor --bf16 --gpu '0' --workers 4 --batch_size 2 --crop_size "$CROP" \
      --epochs 60 --iter_per_epoch_override 250 --val_freq 999 --lr 0.0001 \
      --dist_url "tcp://127.0.0.1:$((PORT++))" > "$LOGD/$name.log" 2>&1; then
    log "OK    $name"
  else
    log "FAIL  $name (voir $LOGD/$name.log)"
  fi
}

log "===== SWEEP PRÉSENCE IVH (X25) : lambda {$LAMBDAS}, τ+=$TAU_POS τ-=$TAU_NEG ====="
for lam in $LAMBDAS; do train "$lam"; done
log "===== SWEEP FINI — reste : éval détection (sens/FN vs baseline X25, PAS Dice) ====="
