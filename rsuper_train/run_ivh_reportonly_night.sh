#!/bin/bash
# run_ivh_reportonly_night.sh -- 4 exp size-loss REPORT-ONLY : {floor(p10), tiered(p25-p75)} × {X25, X10}.
# Présence=flag rapport, bornes=ncomp->niveau->percentiles (calibrés masques _0). heavy 60ep/lr1e-4/crop144/λ=0.5.
# Chaque run suivi de son éval (Dice+ratio, cache). ~6.7h/run.  tmux new -d -s ivhrep "bash run_ivh_reportonly_night.sh"
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$D/logs/ivh_size"; mkdir -p "$LOGD"; MASTER="$LOGD/reportonly.log"
source "$HOME/envs/rsuper/bin/activate"; export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl
cd "$REPO"
META="$D/report_extraction/metadata_prompt6/ich_per_tumor_metadata.csv"
IDS="$D/splits/reports_baseline0_train_ids.csv"
UFO="$D/dataset_ich_reports_npz"
FLOOR="$D/report_extraction/metadata_prompt6/ivh_bounds_floor.csv"
TIER="$D/report_extraction/metadata_prompt6/ivh_bounds_tiered.csv"
PORT=29870
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }

run(){ # <tag> <bounds_csv> <pretrained_ckpt> <masks_dir>
  local tag="$1" bounds="$2" pre="$3" masks="$4" name="ivhbounds_$1"
  if [ -f "$D/exp/ich_ufo/$name/fold_0_latest.pth" ]; then log "SKIP train $name"; else
    log "TRAIN $name (bounds=$(basename "$bounds") pre=$(basename "$(dirname "$pre")") masks=$(basename "$masks"))"
    if python train_ddp.py --dataset ich_ufo --model medformer --dimension 3d \
        --data_root "$masks" --UFO_root "$UFO" --reports "$META" --ucsf_ids "$IDS" \
        --loss ivhbounds --ivh_bounds_table "$bounds" --ivh_size_lambda 0.5 \
        --ivh_size_dil 15 --ivh_ramp_epochs 10 \
        --classes_number 15 --pretrained "$pre" --save_destination "$D/ich3_augmented_ufo" \
        --cp_path "$D/exp/" --log_path "$D/log/" --unique_name "$name" \
        --crop_on_tumor --bf16 --gpu '0' --workers 4 --batch_size 2 --crop_size 144 \
        --epochs 60 --iter_per_epoch_override 250 --val_freq 999 --lr 0.0001 \
        --dist_url "tcp://127.0.0.1:$((PORT++))" > "$LOGD/$name.log" 2>&1; then
      log "OK train $name"
    else
      log "FAIL train $name (voir $LOGD/$name.log)"; return
    fi
  fi
  log "EVAL $name"
  python eval_ivh_size_cli.py "$tag=$D/exp/ich_ufo/$name/fold_0_latest.pth" > "$LOGD/eval_$tag.log" 2>&1 \
    && log "OK eval $name" || log "FAIL eval $name"
}

log "===== NUIT REPORT-ONLY : 4 exp {floor,tiered} × {X25,X10} ====="
run X25_floor  "$FLOOR" "$D/exp/ich/ich3_stage1_X25/fold_0_best.pth"    "$D/subsets3/S_25"
run X25_tiered "$TIER"  "$D/exp/ich/ich3_stage1_X25/fold_0_best.pth"    "$D/subsets3/S_25"
run X10_floor  "$FLOOR" "$D/exp/ich/ich3_stage1_X10/fold_0_latest.pth"  "$D/subsets3/S_10"
run X10_tiered "$TIER"  "$D/exp/ich/ich3_stage1_X10/fold_0_latest.pth"  "$D/subsets3/S_10"

log "===== EVAL FINALE COMBINÉE ====="
python eval_ivh_size_cli.py \
  baseX10="$D/exp/ich/ich3_stage1_X10/fold_0_latest.pth" \
  X25_floor="$D/exp/ich_ufo/ivhbounds_X25_floor/fold_0_latest.pth" \
  X25_tiered="$D/exp/ich_ufo/ivhbounds_X25_tiered/fold_0_latest.pth" \
  X10_floor="$D/exp/ich_ufo/ivhbounds_X10_floor/fold_0_latest.pth" \
  X10_tiered="$D/exp/ich_ufo/ivhbounds_X10_tiered/fold_0_latest.pth" \
  > "$LOGD/eval_reportonly_final.log" 2>&1 || true
log "===== NUIT REPORT-ONLY FINIE — $LOGD/eval_reportonly_final.log ====="
