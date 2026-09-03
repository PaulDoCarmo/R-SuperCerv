#!/bin/bash
# build_reports_npz_incremental.sh -- Construit les npz d'un NOUVEAU lot de cas-rapports et les
# INTÈGRE directement dans le dossier npz existant (dataset_ich_reports_npz), SANS reprocesser
# ni écraser les cas déjà présents. Chaîne : TotalSeg(GPU) -> build -> resample1mm -> npz(CPU).
#
# Isolation : build/resample/npz tournent dans un STAGING 74-only ; seuls les <ID>.npz/<ID>_gt.npz
# finaux sont COPIÉS dans dataset_ich_reports_npz (le manifeste list/ existant n'est PAS touché ;
# on vérifie d'abord que l'ordre des canaux (label_names) est identique).
#
# Usage :
#   bash build_reports_npz_incremental.sh prepare    # symlinks CT des nouveaux (sans GPU)
#   bash build_reports_npz_incremental.sh totalseg   # TotalSeg brain_structures (GPU)
#   bash build_reports_npz_incremental.sh npz        # build+resample+npz (CPU) + intégration
#   bash build_reports_npz_incremental.sh all
set -uo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # rsuper_train
DC="$REPO/dataset_conversion"
NEW_IDS="${NEW_IDS:-$D/report_extraction/reports_newbatch.csv}"  # colonne ID = stems <patient>_<0|1>
VOLS_SRC="${VOLS_SRC:-$D/CT-Reports/FINAL_275_2026_08_06/matched_volumes}"
REPORTS_CT="$D/reports_ct"                                        # set canonique de CT (symlinks)
STAGE="$D/_staging_reports_npz74"
NPZ_FINAL="$D/dataset_ich_reports_npz"                            # destination partagée (201 -> 275)
TS_OUT="$D/organ_masks/segmentations"
LOG="$D/logs/build_reports_npz.log"; mkdir -p "$D/logs"
log(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

ids(){ python3 -c "import pandas as pd;print('\n'.join(str(x) for x in pd.read_csv('$NEW_IDS')['ID']))"; }

phase_prepare(){
  log "===== PREPARE : symlinks CT des nouveaux (sans GPU) ====="
  source "$HOME/envs/rsuper/bin/activate"          # pandas pour lire NEW_IDS (Report multi-lignes)
  mkdir -p "$REPORTS_CT" "$STAGE/vols"
  local n=0 miss=0
  while read -r id; do
    [ -z "$id" ] && continue
    src="$VOLS_SRC/$id.nii.gz"
    if [ ! -f "$src" ]; then log "  MANQUE CT: $src"; miss=$((miss+1)); continue; fi
    ln -sf "$(realpath "$src")" "$REPORTS_CT/$id.nii.gz"
    ln -sf "$(realpath "$src")" "$STAGE/vols/$id.nii.gz"
    n=$((n+1))
  done < <(ids)
  log "  symlinks CT créés : $n (manquants: $miss) -> $STAGE/vols et $REPORTS_CT"
  log "  reports_ct total : $(ls "$REPORTS_CT"/*.nii.gz 2>/dev/null | wc -l)"
}

phase_totalseg(){
  log "===== TOTALSEG : brain_structures sur le staging (GPU) ====="
  [ -d "$STAGE/vols" ] || { log "FAIL: $STAGE/vols absent (prepare d'abord)"; return 1; }
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  log "  VRAM libre: ${free} MiB"
  bash "$REPO/../organ_masks/run_totalseg_local.sh" "$STAGE/vols" "$TS_OUT" >>"$LOG" 2>&1 \
    || { log "FAIL totalseg"; return 1; }
  log "  TotalSeg fini (sorties dans $TS_OUT/segmented_organs_<ID>.nii.gz/)"
}

phase_npz(){
  log "===== NPZ : build -> resample1mm -> npz -> intégration (CPU) ====="
  source "$HOME/envs/rsuper/bin/activate"
  # 1) assemble CT + 12 structures (staging 74-only)
  python "$DC/build_ich_reports_dataset.py" --vols_dir "$STAGE/vols" \
    --totalseg_dir "$TS_OUT" --out_dir "$STAGE/dataset" >>"$LOG" 2>&1 || { log "FAIL build"; return 1; }
  # 2) resample 1mm iso
  python "$DC/resample_ich_3d.py" --src_path "$STAGE/dataset" --label_path "$STAGE/dataset" \
    --tgt_path "$STAGE/dataset_1mm" --label_yaml "$DC/label_names_ich_organs.yaml" --workers 16 >>"$LOG" 2>&1 || { log "FAIL resample"; return 1; }
  # 3) HU[0,100]+zscore -> npz (staging)
  python "$DC/nii_to_npz_ich.py" --src_path "$STAGE/dataset_1mm" --tgt_path "$STAGE/npz" \
    --hu_min 0 --hu_max 100 --workers 16 >>"$LOG" 2>&1 || { log "FAIL npz"; return 1; }
  # 4) vérif ordre des canaux identique à l'existant AVANT intégration
  python - "$STAGE/npz" "$NPZ_FINAL" <<'PY' >>"$LOG" 2>&1 || { log "FAIL: label_names divergent -> PAS d'intégration"; return 1; }
import sys, yaml, os
new, old = sys.argv[1], sys.argv[2]
def lab(root):
    p=os.path.join(root,"list","label_names.yaml")
    return yaml.safe_load(open(p)) if os.path.exists(p) else None
ln_new, ln_old = lab(new), lab(old)
print("label_names NEW:", ln_new)
assert ln_old is None or ln_new==ln_old, f"ORDRE CANAUX DIVERGENT\nnew={ln_new}\nold={ln_old}"
print("ordre des canaux OK (identique à l'existant)")
PY
  # 5) intégration : copie <ID>.npz + <ID>_gt.npz dans le dossier partagé (n'écrase pas le manifeste list/)
  local n=0
  for f in "$STAGE/npz/"*.npz; do
    b=$(basename "$f"); [ "$b" = "list" ] && continue
    [ -e "$NPZ_FINAL/$b" ] && { log "  DÉJÀ présent, saute: $b"; continue; }
    cp -a "$f" "$NPZ_FINAL/$b"; n=$((n+1))
  done
  log "  intégrés dans $NPZ_FINAL : $n fichiers npz"
  log "  total cas npz (_gt) : $(ls "$NPZ_FINAL"/*_gt.npz 2>/dev/null | wc -l)"
  log "===== FINI ====="
}

case "${1:-all}" in
  prepare) phase_prepare ;;
  totalseg) phase_totalseg ;;
  npz) phase_npz ;;
  all) phase_prepare && phase_totalseg && phase_npz ;;
  *) echo "usage: $0 {prepare|totalseg|npz|all}"; exit 1 ;;
esac
