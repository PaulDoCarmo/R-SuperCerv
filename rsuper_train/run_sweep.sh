#!/bin/bash
# run_sweep.sh -- Chef d'orchestre de la SWEEP (nombre de masques) + configs rapports.
# Enchaine SEQUENTIELLEMENT (1 GPU) : stage1(X) -> eval -> stage2(X,_0) -> eval -> stage2(X,all) -> eval,
# pour X dans SIZES, + X=305 (existant), puis agregation.
# Robuste : gate GPU au demarrage, log/job, IDEMPOTENT (skip si deja fait), tolerant aux erreurs.
# A lancer detache :  tmux new -d -s sweep "bash run_sweep.sh"
set -uo pipefail

D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP="/tmp/claude-1005/-home-pdocarmosilva/0c39b71d-6898-448c-bb18-f9db6ffc6fd0/scratchpad"
LOGD="$D/logs/sweep"; mkdir -p "$LOGD" "$D/subsets"
MASTER="$LOGD/sweep_master.log"
SIZES=(25 50 100)                       # X nouveaux (305 = existant, traite a part)

source "$HOME/envs/rsuper/bin/activate"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR="$SP/mpl" RSUPER_DEBUG_ROOT="$D/debug"
cd "$REPO"
PORT=9100

log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }

# ---- gate : attendre que le GPU soit libre (fin du stage2 courant) ----
gate_gpu(){
  log "GATE: attente GPU libre (>=40 Go) ..."
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ -n "$free" ] && [ "$free" -ge 40000 ] && { log "GATE: GPU libre ($free Mo)"; break; }
    sleep 60
  done
}

# ---- job generique : idempotent + log + tolerant ----
# job <label> <donefile> <logfile> -- <cmd...>
job(){
  local label="$1" done="$2" logf="$3"; shift 3; [ "$1" = "--" ] && shift
  if [ -e "$done" ]; then log "SKIP  $label (deja: $(basename "$done"))"; return 0; fi
  log "START $label  -> $(basename "$logf")"
  if "$@" > "$logf" 2>&1; then log "OK    $label"; else log "FAIL  $label (voir $logf)"; fi
}

# ---- entrainements ----
train_stage1(){ # <name> <data_root>
  local name="$1" dr="$2"
  python train_ddp.py --dataset ich --model medformer --dimension 3d \
    --data_root "$dr" --classes_number 13 --save_destination "$D/ich_augmented" \
    --cp_path "$D/exp/" --log_path "$D/log/" --unique_name "$name" \
    --crop_on_tumor --report_volume_loss_basic 0 --bf16 --gpu '0' --workers 8 --batch_size 3 \
    --epochs 60 --iter_per_epoch_override 250 --dist_url "tcp://127.0.0.1:$((PORT++))"
}
train_stage2(){ # <name> <data_root> <ucsf_ids> <pretrained>
  local name="$1" dr="$2" ids="$3" pre="$4"
  python train_ddp.py --dataset ich_ufo --model medformer --dimension 3d \
    --data_root "$dr" --UFO_root "$D/dataset_ich_reports_npz" \
    --reports "$D/report_extraction/metadata/ich_per_tumor_metadata.csv" --ucsf_ids "$ids" \
    --classes_number 13 --pretrained "$pre" --save_destination "$D/ich_augmented_ufo" \
    --cp_path "$D/exp/" --log_path "$D/log/" --unique_name "$name" \
    --crop_on_tumor --report_volume_loss_basic 0.1 --loss ball_dice_last \
    --gpu '0' --workers 8 --batch_size 2 --crop_size 96 --epochs 60 --iter_per_epoch_override 250 \
    --lr 0.0001 --dist_url "tcp://127.0.0.1:$((PORT++))"
}
# ---- evals ----
eval_masks(){ python eval_ich.py --load "$1" --npz_dir "$D/dataset_ich_full_npz" \
    --ids "$D/splits/test_ids.csv" --config config/ich/medformer_3d.yaml \
    --class_list "$D/dataset_ich_full_npz/list/label_names.yaml" --save_csv "$2" --gpu 0 --threshold 0.5 --nsd_tol 1.0; }
eval_reports(){ python eval_ich_reports.py --load "$1" --ufo_npz_dir "$D/dataset_ich_reports_npz" \
    --ids "$2" --reports "$D/report_extraction/metadata/ich_per_tumor_metadata.csv" \
    --config config/ich_ufo/medformer_3d.yaml --class_list "$D/dataset_ich_full_npz/list/label_names.yaml" \
    --ufo_class_list "$D/dataset_ich_reports_npz/list/label_names.yaml" --save_csv "$3" \
    --gpu 0 --threshold 0.5 --detect_min_ml 0.5 --min_overlap_ml 0.1; }

# ================= PIPELINE =================
log "===== SWEEP DEBUT (X = ${SIZES[*]} + 305) ====="
gate_gpu

# 1) sous-ensembles emboites
job "subsets S_${SIZES[*]}" "$D/subsets/S_${SIZES[-1]}/list" "$LOGD/subsets.log" -- \
  python dataset_conversion/make_mask_subsets.py --trainval_dir "$D/dataset_ich_full_npz_trainval" \
    --manifest "$D/dataset_ich_full/manifest_ich.csv" --out_root "$D/subsets" --sizes "${SIZES[@]}" --seed 0

# fonction : une experience complete pour une taille X (name_s1, data_root_s1 donnes)
experiment(){ # <X> <stage1_name> <stage1_dataroot>
  local X="$1" s1="$2" dr="$3" E="$D/eval/sweep"; mkdir -p "$E"
  local best1="$D/exp/ich/$s1/fold_0_best.pth"
  job "stage1 X$X"        "$best1" "$LOGD/${s1}.log"            -- train_stage1 "$s1" "$dr"
  job "eval s1 X$X masks" "$E/X${X}_stage1__masks.csv" "$LOGD/eval_${s1}_masks.log" -- eval_masks "$best1" "$E/X${X}_stage1__masks.csv"
  job "eval s1 X$X rep0"  "$E/X${X}_stage1__rep0.csv"  "$LOGD/eval_${s1}_rep0.log"  -- eval_reports "$best1" "$D/splits/reports_baseline0_test_ids.csv" "$E/X${X}_stage1__rep0.csv"
  job "eval s1 X$X repall" "$E/X${X}_stage1__repall.csv" "$LOGD/eval_${s1}_repall.log" -- eval_reports "$best1" "$D/splits/reports_all_test_ids.csv" "$E/X${X}_stage1__repall.csv"
  for cfg in base0 all; do
    local ids; [ "$cfg" = base0 ] && ids="$D/splits/reports_baseline0_train_ids.csv" || ids="$D/splits/reports_all_train_ids.csv"
    local s2="ich_stage2_X${X}_${cfg}"; local best2="$D/exp/ich_ufo/$s2/fold_0_best.pth"
    job "stage2 X$X $cfg"        "$best2" "$LOGD/${s2}.log" -- train_stage2 "$s2" "$dr" "$ids" "$best1"
    job "eval s2 X$X $cfg masks" "$E/X${X}_stage2_${cfg}__masks.csv" "$LOGD/eval_${s2}_masks.log" -- eval_masks "$best2" "$E/X${X}_stage2_${cfg}__masks.csv"
    job "eval s2 X$X $cfg rep0"  "$E/X${X}_stage2_${cfg}__rep0.csv"  "$LOGD/eval_${s2}_rep0.log"  -- eval_reports "$best2" "$D/splits/reports_baseline0_test_ids.csv" "$E/X${X}_stage2_${cfg}__rep0.csv"
    job "eval s2 X$X $cfg repall" "$E/X${X}_stage2_${cfg}__repall.csv" "$LOGD/eval_${s2}_repall.log" -- eval_reports "$best2" "$D/splits/reports_all_test_ids.csv" "$E/X${X}_stage2_${cfg}__repall.csv"
  done
}

# 2) X=305 (stage1 existant = ich_stage1_13cls ; stage2 _0 = ich_stage2_allmask_base0 deja lance)
#    on evalue l'existant + on ajoute la config 'all'
E="$D/eval/sweep"; mkdir -p "$E"
BEST1_305="$D/exp/ich/ich_stage1_13cls/fold_0_best.pth"
BEST2_305_0="$D/exp/ich_ufo/ich_stage2_allmask_base0/fold_0_best.pth"
job "eval s1 X305 masks" "$E/X305_stage1__masks.csv" "$LOGD/eval_s1_305_masks.log" -- eval_masks "$BEST1_305" "$E/X305_stage1__masks.csv"
job "eval s1 X305 rep0"  "$E/X305_stage1__rep0.csv"  "$LOGD/eval_s1_305_rep0.log"  -- eval_reports "$BEST1_305" "$D/splits/reports_baseline0_test_ids.csv" "$E/X305_stage1__rep0.csv"
job "eval s1 X305 repall" "$E/X305_stage1__repall.csv" "$LOGD/eval_s1_305_repall.log" -- eval_reports "$BEST1_305" "$D/splits/reports_all_test_ids.csv" "$E/X305_stage1__repall.csv"
job "eval s2 X305 base0 masks" "$E/X305_stage2_base0__masks.csv" "$LOGD/eval_s2_305_0_masks.log" -- eval_masks "$BEST2_305_0" "$E/X305_stage2_base0__masks.csv"
job "eval s2 X305 base0 rep0"  "$E/X305_stage2_base0__rep0.csv"  "$LOGD/eval_s2_305_0_rep0.log"  -- eval_reports "$BEST2_305_0" "$D/splits/reports_baseline0_test_ids.csv" "$E/X305_stage2_base0__rep0.csv"
job "eval s2 X305 base0 repall" "$E/X305_stage2_base0__repall.csv" "$LOGD/eval_s2_305_0_repall.log" -- eval_reports "$BEST2_305_0" "$D/splits/reports_all_test_ids.csv" "$E/X305_stage2_base0__repall.csv"
# stage2(305, all)
S2_305_ALL="ich_stage2_X305_all"; BEST2_305_ALL="$D/exp/ich_ufo/$S2_305_ALL/fold_0_best.pth"
job "stage2 X305 all" "$BEST2_305_ALL" "$LOGD/${S2_305_ALL}.log" -- train_stage2 "$S2_305_ALL" "$D/dataset_ich_full_npz_trainval" "$D/splits/reports_all_train_ids.csv" "$BEST1_305"
job "eval s2 X305 all masks" "$E/X305_stage2_all__masks.csv" "$LOGD/eval_s2_305_all_masks.log" -- eval_masks "$BEST2_305_ALL" "$E/X305_stage2_all__masks.csv"
job "eval s2 X305 all rep0"  "$E/X305_stage2_all__rep0.csv"  "$LOGD/eval_s2_305_all_rep0.log"  -- eval_reports "$BEST2_305_ALL" "$D/splits/reports_baseline0_test_ids.csv" "$E/X305_stage2_all__rep0.csv"
job "eval s2 X305 all repall" "$E/X305_stage2_all__repall.csv" "$LOGD/eval_s2_305_all_repall.log" -- eval_reports "$BEST2_305_ALL" "$D/splits/reports_all_test_ids.csv" "$E/X305_stage2_all__repall.csv"

# 3) sweep des nouvelles tailles
for X in "${SIZES[@]}"; do experiment "$X" "ich_stage1_X${X}" "$D/subsets/S_${X}"; done

# 4) agregation finale (toujours executee, pas d'idempotence -> reflete tous les evals dispo)
log "AGREGATION"
if python dataset_conversion/aggregate_sweep.py --eval_dir "$D/eval/sweep" --out_dir "$D/eval/sweep_summary" > "$LOGD/aggregate.log" 2>&1; then
  log "OK AGREGATION -> $D/eval/sweep_summary/"
else
  log "FAIL AGREGATION (voir $LOGD/aggregate.log)"
fi
log "===== SWEEP TERMINEE ====="
