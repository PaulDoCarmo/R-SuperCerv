#!/bin/bash
D=/mnt/Data/data_paul
REPO=/home/pdocarmosilva/projects/R-SuperCerv
echo "=== BATCH TotalSeg debut $(date) ==="
bash "$REPO/organ_masks/run_totalseg_local.sh" "$D/data_laurent/NIFTI/vols"
bash "$REPO/organ_masks/run_totalseg_local.sh" "$D/reports_ct"
echo "=== BATCH TotalSeg FIN $(date) ==="
