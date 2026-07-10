#!/bin/bash
# create_report_env.local.sh -- Env report_extraction (vLLM + inference LLM), MACHINE LOCALE.
# Port de la partie env de LaunchMonoGPU.sh (Alliance : module + HF_HUB_OFFLINE=1).
# Contraintes materielles locales : 1x RTX 6000 Ada 48 GB, driver 535 (= CUDA 12.2).
#   - Il faut du cu121 : vLLM >= 0.7 passe en cu124 (exige driver >= 550). On PIN vllm==0.6.6,
#     derniere lignee buildee cu121, qui supporte Qwen2.5-72B-AWQ + fp8 KV cache sur Ada.
#   - torch 2.5.1+cu121 installe D'ABORD depuis l'index cu121, sinon vLLM tire un torch cu124.
#   - transformers borne a 4.47.1 : le pin de vLLM 0.6.6 n'a pas de borne haute -> pip prendrait
#     la 5.x (incompatible). torchvision 0.20.1 force en cu121 (defaut PyPI = cu124).
# NB : ne telecharge PAS le modele. Le LLM (Qwen2.5-72B-Instruct-AWQ ~40 GB) se telecharge a
#      l'etape LLM (huggingface-cli), cf. report_extraction/README.md.
#   bash create_report_env.local.sh [ /chemin/env ]
set -euo pipefail
ENV_DIR="${1:-$HOME/envs/report}"
echo "==> [report] venv @ $ENV_DIR"
python3.10 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
echo "==> [report] torch 2.5.1 cu121 (driver 535)"
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
echo "==> [report] vllm==0.6.6 + clients"
pip install vllm==0.6.6 pandas openai httpx
echo "==> [report] pin transformers 4.47.1 (compat vLLM 0.6.6)"
pip install "transformers==4.47.1"
echo "==> [report] torchvision 0.20.1 cu121 (apparie torch 2.5.1 ; ecrase le cu124 du defaut PyPI)"
pip install torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps
echo "==> [report] verif imports"
python - <<'PY'
import torch, torchvision, transformers, vllm
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("transformers", transformers.__version__, "| vllm", vllm.__version__)
print("cuda avail:", torch.cuda.is_available(), "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
echo "==> [report] DONE ($ENV_DIR)"
