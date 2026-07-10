#!/bin/bash
# create_nnunet_env.local.sh -- Env nnU-Net (baseline), MACHINE LOCALE.
# Port de create_nnunet_env.sh (Alliance) :
#   - PAS de module / PAS de --no-index.
#   - torch 2.6.0 + torchvision 0.21.0 cu118 D'ABORD (driver 535), sinon nnunetv2 tire cu124.
#   - nnunetv2 PIN 2.5.1 (version validee Alliance) pour que le patch weights_only matche.
#   - garde setuptools<81 + patch torch.load(weights_only=False) (torch 2.6).
#   - RAPPEL a l'usage : export nnUNet_compile=f  (torch.compile casse avec torch 2.6 / nnU-Net).
#   bash create_nnunet_env.local.sh [ /chemin/env ]
set -euo pipefail
ENV_DIR="${1:-$HOME/envs/nnunet}"
echo "==> [nnunet] venv @ $ENV_DIR"
python3.10 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
echo "==> [nnunet] torch 2.6.0 + torchvision 0.21.0 cu118"
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
echo "==> [nnunet] nnunetv2==2.5.1"
pip install nnunetv2==2.5.1
pip install "setuptools<81"

# Patch nnU-Net 2.5.1 + torch 2.6 : torch.load defaut weights_only=True casse les checkpoints.
PRED="$ENV_DIR/lib/python3.10/site-packages/nnunetv2/inference/predict_from_raw_data.py"
if grep -q "map_location=torch.device('cpu'))" "$PRED"; then
    sed -i "s|map_location=torch.device('cpu'))|map_location=torch.device('cpu'), weights_only=False)|" "$PRED"
    echo "==> [nnunet] patch weights_only=False APPLIQUE"
else
    echo "==> [nnunet] ATTENTION: motif du patch introuvable (a verifier a la main) : $PRED"
fi
grep -n "weights_only=False" "$PRED" | head -3 || true

python -c "import nnunetv2, torch; print('nnunetv2 OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "==> [nnunet] DONE ($ENV_DIR) -- RAPPEL: export nnUNet_compile=f a l'usage"
