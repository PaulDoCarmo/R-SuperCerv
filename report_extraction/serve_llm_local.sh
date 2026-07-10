#!/bin/bash
# serve_llm_local.sh -- Port MACHINE LOCALE de LaunchMonoGPU.sh (Alliance).
# Sert le LLM avec vLLM puis lance l'inference sur reports.csv. 1x RTX 6000 Ada 48 GB.
#
# Differences vs Alliance :
#   - PAS de #SBATCH / module load ; on active ~/envs/report.
#   - qwen2-5-72b-awq : --max-model-len 8192 (le defaut 32768 fait OOM sur 48 GB ; rapports courts).
#   - modele lu EN LOCAL (pre-telecharge dans HFModels/) -> HF_HUB_OFFLINE=1.
#   - GPU PARTAGE : verifier `nvidia-smi` avant (le 72B AWQ ~40 GB + KV ~3 GB exige ~44 GB libres ;
#     si atibi ou autre occupe le GPU -> attendre, sinon OOM au chargement).
#
# Usage :
#   bash serve_llm_local.sh [DATA_PATH] [SAVE_PATH] [LLM_NAME] [GPU] [GPU_UTIL] [PROMPT_ID]
#   defaut : reports.csv -> results.csv, qwen2-5-72b-awq, GPU 0, util 0.92, prompt 5
set -euo pipefail
D="${RSUPER_DATA:-/mnt/Data/data_paul}"
RE="$D/report_extraction"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_PATH="${1:-$RE/reports.csv}"
SAVE_PATH="${2:-$RE/results.csv}"
LLM_NAME="${3:-qwen2-5-72b-awq}"
GPU="${4:-0}"
GPU_UTIL="${5:-0.95}"   # 0.92 laissait le KV cache trop court pour max_model_len=8192 (tenait 7968 tok)
PROMPT_ID="${6:-5}"
HF_MODELS_DIR="$RE/HFModels"
HF_CACHE="$RE/HFCache"
LOG_DIR="$D/logs"; mkdir -p "$LOG_DIR"

source "$HOME/envs/report/bin/activate"

case "$LLM_NAME" in
  qwen2-5-72b-awq)
    MODEL="$HF_MODELS_DIR/Qwen2.5-72B-Instruct-AWQ"
    MODEL_OPTS="--dtype half --max-model-len 8192 --tensor-parallel-size 1"
    # Si OOM au chargement malgre 8192 : ajouter --kv-cache-dtype fp8 (supporte par l'Ada).
    ;;
  qwen2-5-32b-awq)
    MODEL="$HF_MODELS_DIR/Qwen2.5-32B-Instruct-AWQ"   # ~20 GB, confortable (extraction DIFFERENTE)
    MODEL_OPTS="--dtype half --max-model-len 16384 --tensor-parallel-size 1"
    ;;
  *) echo "LLM_NAME inconnu: $LLM_NAME"; exit 1 ;;
esac
[ -d "$MODEL" ] || { echo "Modele absent: $MODEL (lancer d'abord: hf download ...)"; exit 1; }

# Naming de sortie fidele a LaunchMonoGPU (-> results_<MODEL_TAG>.csv, puis Run_LLM_inference
# ajoute /raw/prompt<ID>/..._prompt<ID>.csv)
MODEL_TAG="$(basename "$MODEL")"; MODEL_TAG="${MODEL_TAG//[^A-Za-z0-9._-]/_}"
SAVE_PATH="${SAVE_PATH%.csv}_${MODEL_TAG}.csv"

# Port libre aleatoire
BASE_PORT=0
while :; do p=$((1024 + RANDOM % 8000)); if ! (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; then BASE_PORT=$p; break; fi; done
echo "vLLM port=$BASE_PORT  modele=$MODEL  opts=$MODEL_OPTS"

export VLLM_NO_USAGE_STATS=1 TORCH_COMPILE=0 VLLM_USE_TORCH_COMPILE=0
APILOG="$LOG_DIR/vllm_api.log"
# --disable-frontend-multiprocessing : evite le canal ZMQ frontend<->moteur (pyzmq 27 casse
# le multiprocessing de vLLM 0.6.6 -> zmq.error.ZMQError: Operation not supported). Moteur
# in-process : suffisant pour une inference batch mono-client.
HF_HOME="$HF_CACHE" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
  vllm serve "$MODEL" $MODEL_OPTS --port "$BASE_PORT" \
       --gpu_memory_utilization "$GPU_UTIL" --enforce-eager \
       --disable-frontend-multiprocessing \
       > "$APILOG" 2>&1 &
VLLM_PID=$!
trap 'echo "arret vLLM"; kill $VLLM_PID 2>/dev/null || true' EXIT

echo "attente API (log: $APILOG)..."
until curl -s "http://localhost:${BASE_PORT}/v1/models" >/dev/null 2>&1; do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM s'est arrete (voir $APILOG)"; tail -20 "$APILOG"; exit 1; }
  sleep 5
done
echo "API prete. Inference (prompt $PROMPT_ID)..."
python "$HERE/LLM_inference/Run_LLM_inference.py" \
    --port "$BASE_PORT" --data_path "$DATA_PATH" --save_path "$SAVE_PATH" --prompt_id "$PROMPT_ID"
echo "Inference terminee. Sortie sous $RE/raw/prompt${PROMPT_ID}/"
