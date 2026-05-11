#!/usr/bin/env bash
#SBATCH --account=rrg-josedolz
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --job-name=rsupercerv_llm
#SBATCH --output=/home/pauldcrm/links/scratch/R-SuperCerv/logs/llm_%j.out
#SBATCH --error=/home/pauldcrm/links/scratch/R-SuperCerv/logs/llm_%j.err

# Usage:
#   sbatch LaunchMonoGPU.sh [DATA_PATH] [SAVE_PATH] [LLM_SIZE] [BASE_GPU] [TOP_GPU_USAGE] [HF_CACHE] [HF_MODELS_DIR] [PROMPT_ID]
#
# Example:
#   sbatch LaunchMonoGPU.sh /path/to/reports.csv /path/to/output.csv large 0 0.9 /path/to/HFCache /path/to/HFModels 1

set -euxo pipefail

trap "echo 'Slurm requested stop. Shutting down background jobs...'; jobs -p | xargs -r kill; exit" SIGINT SIGTERM EXIT

LOG_DIR="/home/pauldcrm/links/scratch/R-SuperCerv/logs"
HF_CACHE_DEFAULT="/home/pauldcrm/links/scratch/R-SuperCerv/HFCache"
HF_MODELS_DIR_DEFAULT="/home/pauldcrm/links/scratch/R-SuperCerv/HFModels"
mkdir -p "$LOG_DIR"
mkdir -p "$HF_CACHE_DEFAULT"
mkdir -p "$HF_MODELS_DIR_DEFAULT"

module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.9 rust
source /home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/report_extraction/report_env/bin/activate

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"

DATA_PATH="${1:-/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/reports.csv}"
SAVE_PATH="${2:-/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/raw/results.csv}"
LLM_SIZE="${3:-large}"
BASE_GPU="${4:-0}"
TOP_GPU_USAGE="${5:-0.9}"
HF_CACHE="${6:-$HF_CACHE_DEFAULT}"
HF_MODELS_DIR="${7:-$HF_MODELS_DIR_DEFAULT}"
PROMPT_ID="${8:-1}"

fname="$(basename "$DATA_PATH")"
base="${fname%.*}"

randomize_base_port() {
  local start_range=1000
  local end_range=9999
  while true; do
    try_port=$((start_range + RANDOM % (end_range - start_range + 1)))
    if ! lsof -i :"$try_port" -sTCP:LISTEN > /dev/null 2>&1; then
      echo "$try_port"
      return 0
    fi
  done
}

BASE_PORT=$(randomize_base_port)
echo "Selected BASE_PORT=$BASE_PORT"

case "$LLM_SIZE" in
  small)
    MODEL_REPO="iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8"
    MODEL="$HF_MODELS_DIR/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8"
    MODEL_OPTS="--max-model-len 12000 --dtype float16"
    ;;
  large)
    MODEL_REPO="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    MODEL="$HF_MODELS_DIR/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    MODEL_OPTS="--dtype half --max-model-len 60000 --tensor-parallel-size 1"
    ;;
  deepseek)
    MODEL_REPO="Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"
    MODEL="$HF_MODELS_DIR/DeepSeek-R1-Distill-Llama-70B-AWQ"
    MODEL_OPTS="--dtype half --max-model-len 60000 --tensor-parallel-size 1"
    ;;
  *)
    echo "Unknown LLM_SIZE: '$LLM_SIZE'. Must be 'small', 'large', or 'deepseek'."
    exit 1
    ;;
esac

MODEL_TAG="$(basename "$MODEL")"
MODEL_TAG="${MODEL_TAG//[^A-Za-z0-9._-]/_}"
if [[ "$SAVE_PATH" == *.csv ]]; then
  SAVE_PATH="${SAVE_PATH%.csv}_${MODEL_TAG}.csv"
else
  SAVE_PATH="${SAVE_PATH}_${MODEL_TAG}"
fi

echo "Launching vLLM on GPU $BASE_GPU (port $BASE_PORT)"
echo "HF_CACHE: $HF_CACHE"
echo "HF_MODELS_DIR: $HF_MODELS_DIR"
export TRITON_PTXAS_PATH="$(which ptxas)"
export VLLM_USE_TORCH_COMPILE=0
export TORCH_COMPILE=0
export VLLM_NO_USAGE_STATS=1
export PROMETHEUS_MULTIPROC_DIR="$LOG_DIR/prometheus"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
rm -f "API_GPU${BASE_GPU}_${base}.log"
HF_HOME="$HF_CACHE" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$BASE_GPU" \
  vllm serve "$MODEL" \
             $MODEL_OPTS \
             --port "$BASE_PORT" \
             --gpu_memory_utilization "$TOP_GPU_USAGE" \
             --enforce-eager \
             > "$LOG_DIR/API_GPU${BASE_GPU}_${base}.log" 2>&1 &

while ! curl -s "http://localhost:${BASE_PORT}/v1/models" > /dev/null; do
  echo "Waiting for API on port $BASE_PORT..."
  sleep 5
done

echo "vLLM ready. Running inference..."
JOB_TAG="${SLURM_JOB_ID:-local}"
rm -f "$LOG_DIR/LLM_${base}_${JOB_TAG}.log"
python RunRadGPT.py \
  --port "$BASE_PORT" \
  --data_path "$DATA_PATH" \
  --save_path "$SAVE_PATH" \
  --prompt_id "$PROMPT_ID" \
  2>&1 | tee -a "$LOG_DIR/LLM_${base}_${JOB_TAG}.log"
