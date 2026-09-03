#!/bin/bash
# run_ivh_oracle_rsna.sh -- Size-loss IVH ORACLE avec CHUM + RSNA reports (22 nouveaux, hors S_25).
# heavy 60ep/lr1e-4/crop144, fine-tune depuis stage1_X25, co-train S_25. λ ∈ {0.5, 0.25}.
# Dépend du metadata-texte RSNA (les cas UFO doivent être dans le CSV reports pour survivre à clean_ufo).
# Étapes : (prepare) construit reports combiné + ucsf combiné ; (train) 2 λ + éval.
#   bash run_ivh_oracle_rsna.sh prepare   # construit le combiné (exige metadata RSNA)
#   bash run_ivh_oracle_rsna.sh all        # prepare puis 2 runs
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGD="$D/logs/ivh_size"; mkdir -p "$LOGD"; MASTER="$LOGD/oracle_rsna.log"
source "$HOME/envs/rsuper/bin/activate"; export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/mpl
cd "$REPO"
MP="$D/report_extraction/metadata_prompt6"; MPR="$D/report_extraction/metadata_prompt6_rsna"
REPORTS_COMB="$MP/ich_per_tumor_combined.csv"
UCSF_COMB="$D/splits/reports_chum_rsna_train.csv"
ORACLE="$MP/ivh_oracle_volumes_combined.csv"
UFO="$D/dataset_ich_reports_combined_npz"
PRE="$D/exp/ich/ich3_stage1_X25/fold_0_best.pth"; MASKS="$D/subsets3/S_25"
PORT=29880
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$MASTER"; }

prepare(){
  log "===== PREPARE combiné CHUM+RSNA ====="
  [ -f "$MPR/ich_per_tumor_metadata.csv" ] || { log "FAIL : metadata RSNA absent ($MPR) — lancer le pipeline RSNA d'abord"; return 1; }
  python - <<PY
import pandas as pd, glob, os
D="$D"
c=pd.read_csv("$MP/ich_per_tumor_metadata.csv"); r=pd.read_csv("$MPR/ich_per_tumor_metadata.csv")
comb=pd.concat([c,r],ignore_index=True); comb.to_csv("$REPORTS_COMB",index=False)
# ucsf = baseline0_train (CHUM _0) + les 22 RSNA (clean_ufo filtrera les inexploitables)
b0=set(pd.read_csv("$D/splits/reports_baseline0_train_ids.csv").BDMAP_ID.astype(str))
rsna=set(pd.read_csv("$D/report_extraction/metadata_prompt6_rsna/ivh_oracle_volumes.csv").BDMAP_ID.astype(str))
ids=sorted(b0|rsna)
pd.DataFrame({"BDMAP_ID":ids}).to_csv("$UCSF_COMB",index=False)
print(f"reports combiné : {comb['BDMAP_ID'].nunique()} scans -> $REPORTS_COMB")
print(f"ucsf combiné : {len(ids)} (CHUM_0 {len(b0)} + RSNA {len(rsna)}) -> $UCSF_COMB")
PY
}

train(){ # <tag> <lambda>
  local name="ivhsize_X25rsna_$1" lam="$2"
  if [ -f "$D/exp/ich_ufo/$name/fold_0_latest.pth" ]; then log "SKIP $name"; else
    log "TRAIN $name (λ=$lam, CHUM+RSNA oracle, heavy)"
    if python train_ddp.py --dataset ich_ufo --model medformer --dimension 3d \
        --data_root "$MASKS" --UFO_root "$UFO" --reports "$REPORTS_COMB" --ucsf_ids "$UCSF_COMB" \
        --loss ivhvol --ivh_size_oracle "$ORACLE" --ivh_size_lambda "$lam" \
        --ivh_size_dil 15 --ivh_size_tol 0.1 --ivh_ramp_epochs 10 \
        --classes_number 15 --pretrained "$PRE" --save_destination "$D/ich3_augmented_ufo" \
        --cp_path "$D/exp/" --log_path "$D/log/" --unique_name "$name" \
        --crop_on_tumor --bf16 --gpu '0' --workers 4 --batch_size 2 --crop_size 144 \
        --epochs 60 --iter_per_epoch_override 250 --val_freq 999 --lr 0.0001 \
        --dist_url "tcp://127.0.0.1:$((PORT++))" > "$LOGD/$name.log" 2>&1; then log "OK train $name"; else log "FAIL train $name"; return; fi
  fi
  log "EVAL $name"; python eval_ivh_size_cli.py "$1=$D/exp/ich_ufo/$name/fold_0_latest.pth" > "$LOGD/eval_rsna_$1.log" 2>&1 && log "OK eval $name" || log "FAIL eval $name"
}

case "${1:-all}" in
  prepare) prepare ;;
  all) prepare && { train L05 0.5; train L025 0.25; } ;;
  *) echo "usage: $0 {prepare|all}"; exit 1 ;;
esac
log "===== FINI ====="
