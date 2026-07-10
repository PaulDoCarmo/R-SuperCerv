#!/bin/bash
# create_env.local.sh -- Port MACHINE LOCALE (GPU interne) de create_env.sh (Alliance).
# Differences vs la version Alliance :
#   - PAS de `module load` / PAS de `virtualenv --no-download` / PAS de `--no-index` (internet dispo).
#   - torch 2.6.0 en cu118 : le driver local est 535.x (= CUDA 12.2). torch 2.6.0 n'existe PAS
#     en cu121 (l'index cu121 s'arrete a 2.5.1) et cu124 exigerait un driver >= 550. cu118 tourne
#     sur 535 et supporte l'Ada (sm_89) -> on garde la version torch validee (2.6.0) en cu118.
#   - torchvision DOIT venir du meme index cu118 (timm l'importe -> sinon CUDA major mismatch).
#   - on garde le fix setuptools<81 (pkg_resources / tensorboard) via requirements-rsuper.txt.
#
#   bash create_env.local.sh              # env par defaut ($HOME/envs/rsuper)
#   bash create_env.local.sh /chemin/env  # autre emplacement
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${1:-$HOME/envs/rsuper}"
REQ="$HERE/requirements-rsuper.txt"

echo "==> [rsuper] python3.10 venv @ $ENV_DIR"
python3.10 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

echo "==> [rsuper] torch 2.6.0 + torchvision 0.21.0 (cu118, apparies)"
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

echo "==> [rsuper] requirements-rsuper.txt (torch deja satisfait -> pas de reinstall)"
pip install -r "$REQ"

echo "==> [rsuper] verif GPU + deps"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
import SimpleITK, nibabel, yaml, skimage, monai, timm, einops, tensorboard
print("deps OK: SimpleITK", SimpleITK.Version.VersionString(), "| monai", monai.__version__)
PY
echo "==> [rsuper] DONE ($ENV_DIR)"
