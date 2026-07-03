#!/bin/bash
# create_nnunet_env.sh -- Construit l'env nnU-Net (Compute Canada / Alliance).
# nnunetv2 2.5.1 (le papier utilise v2.0 ; v2.x compatible).
set -euo pipefail
ENV="${1:-/home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/nnunet/nnunet_env}"
module purge
module load StdEnv/2023 python/3.10
[ -d "$ENV" ] || virtualenv --no-download "$ENV"
source "$ENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index nnunetv2
pip install "setuptools<81"   # pkg_resources (retire de setuptools 82 du wheelhouse)

# Fix nnU-Net 2.5.1 + torch 2.6 : torch.load defaut weights_only=True casse le
# chargement des checkpoints (scalaires numpy). Nos checkpoints sont surs -> False.
PRED="$ENV/lib/python3.10/site-packages/nnunetv2/inference/predict_from_raw_data.py"
sed -i "s|map_location=torch.device('cpu'))|map_location=torch.device('cpu'), weights_only=False)|" "$PRED"

python -c "import nnunetv2, torch; print('nnunetv2 OK, torch', torch.__version__)"
