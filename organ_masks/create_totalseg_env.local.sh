#!/bin/bash
# create_totalseg_env.local.sh -- Env TotalSegmentator (brain_structures), MACHINE LOCALE.
# torch 2.6.0 + torchvision 0.21.0 en cu118 D'ABORD (driver 535), sinon TotalSegmentator/timm
# tirent un torch/torchvision cu124 incompatible (CUDA major mismatch).
#   bash create_totalseg_env.local.sh [ /chemin/env ]
set -euo pipefail
ENV_DIR="${1:-$HOME/envs/totalseg}"
echo "==> [totalseg] venv @ $ENV_DIR"
python3.10 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
echo "==> [totalseg] torch 2.6.0 + torchvision 0.21.0 cu118"
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
echo "==> [totalseg] TotalSegmentator"
pip install TotalSegmentator
python -c "import torch, torchvision, timm, totalsegmentator; print('torch', torch.__version__, '| tv', torchvision.__version__, '| cuda', torch.cuda.is_available())"
echo "==> [totalseg] DONE ($ENV_DIR)"
