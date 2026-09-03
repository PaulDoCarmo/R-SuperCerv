#!/bin/bash
# run_ivh_overnight.sh -- NUIT : (1) contrôle size-loss OFF (isole gain size-loss vs +60ep X25),
# (2) sweep lambda heavy {1.0, 2.0}. Chaque run suivi de son éval (Dice+ratio, cache probas).
# Heavy = 60ep / lr 1e-4 / crop 144 / _0-only (reports_baseline0_train). ~6.7h/run.
#   tmux new -d -s ivhnuit "bash run_ivh_overnight.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$D/logs/ivh_size"; mkdir -p "$LOGD"; MASTER="$LOGD/overnight.log"
source "$HOME/envs/rsuper/bin/activate"; export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl
cd "$REPO"
PRE="$D/exp/ich/ich3_stage1_X25/fold_0_best.pth"
MASKS="$D/subsets3/S_25"; UFO="$D/dataset_ich_reports_npz"
META="$D/report_extraction/metadata_prompt6/ich_per_tumor_metadata.csv"
ORACLE="$D/report_extraction/metadata_prompt6/ivh_oracle_volumes.csv"
IDS="$D/splits/reports_baseline0_train_ids.csv"
PORT=29860
for f in "$PRE" "$META" "$ORACLE" "$IDS"; do [ -f "$f" ] || { echo "manque $f"; exit 1; }; done
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }

run(){ # <tag> <lambda>
  local tag="$1" lam="$2" name="ivhsize_X25_$1"
  if [ -f "$D/exp/ich_ufo/$name/fold_0_latest.pth" ]; then
    log "SKIP train $name (déjà là)"
  else
    log "TRAIN $name (lambda=$lam, heavy 60ep/lr1e-4/crop144)"
    if python train_ddp.py --dataset ich_ufo --model medformer --dimension 3d \
        --data_root "$MASKS" --UFO_root "$UFO" --reports "$META" --ucsf_ids "$IDS" \
        --loss ivhvol --ivh_size_oracle "$ORACLE" --ivh_size_lambda "$lam" \
        --ivh_size_dil 15 --ivh_size_tol 0.1 --ivh_ramp_epochs 10 \
        --classes_number 15 --pretrained "$PRE" --save_destination "$D/ich3_augmented_ufo" \
        --cp_path "$D/exp/" --log_path "$D/log/" --unique_name "$name" \
        --crop_on_tumor --bf16 --gpu '0' --workers 4 --batch_size 2 --crop_size 144 \
        --epochs 60 --iter_per_epoch_override 250 --val_freq 999 --lr 0.0001 \
        --dist_url "tcp://127.0.0.1:$((PORT++))" > "$LOGD/$name.log" 2>&1; then
      log "OK train $name"
    else
      log "FAIL train $name (voir $LOGD/$name.log)"; return
    fi
  fi
  log "EVAL $name (Dice+ratio, cache probas)"
  python eval_ivh_size_cli.py "$tag=$D/exp/ich_ufo/$name/fold_0_latest.pth" > "$LOGD/eval_$tag.log" 2>&1 \
    && log "OK eval $name" || log "FAIL eval $name"
}

log "===== NUIT : contrôle + sweep lambda heavy ====="
run ctrl 0
run L10  1.0
run L20  2.0

log "===== EVAL FINALE COMBINÉE (baseline + tous les lambda) ====="
python eval_ivh_size_cli.py \
  heavy0.5="$D/exp/ich_ufo/ivhsize_X25_heavy/fold_0_latest.pth" \
  ctrl="$D/exp/ich_ufo/ivhsize_X25_ctrl/fold_0_latest.pth" \
  L10="$D/exp/ich_ufo/ivhsize_X25_L10/fold_0_latest.pth" \
  L20="$D/exp/ich_ufo/ivhsize_X25_L20/fold_0_latest.pth" \
  > "$LOGD/eval_final.log" 2>&1 || true
log "===== NUIT FINIE — résultats : $LOGD/eval_final.log ====="
