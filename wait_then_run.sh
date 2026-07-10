#!/bin/bash
# wait_then_run.sh -- Attend que des conditions soient reunies, puis lance une commande.
# Conditions supportees (toutes optionnelles, combinables ; ET logique) :
#   --gpu-free-mb N    attendre >= N Mo LIBRES sur le GPU (nvidia-smi)
#   --gpu-idle P       attendre utilisation GPU <= P %  (defaut si donne sans valeur: 10)
#   --after-tmux NAME  attendre que la session tmux NAME n'existe PLUS (job fini)
#   --after-pid PID    attendre que le process PID se termine
#   --need-file PATH   attendre que PATH existe (ex. modele telecharge)
#   --gpu-index I      index GPU a interroger (defaut 0)
#   --interval S       periode de sondage en secondes (defaut 30)
# Puis tout ce qui suit `--` est execute.
#
# Exemples :
#   # lancer l'inference LLM quand TotalSeg est fini ET >=44 Go GPU libres :
#   bash wait_then_run.sh --after-tmux totalseg --gpu-free-mb 44000 -- \
#        bash report_extraction/serve_llm_local.sh
#   # a mettre soi-meme dans tmux pour survivre a la deconnexion :
#   tmux new -d -s gate "bash wait_then_run.sh --gpu-free-mb 44000 -- bash mon_job.sh"
set -u
GPU_IDX=0; INTERVAL=30
GPU_FREE=""; GPU_IDLE=""; AFTER_TMUX=""; AFTER_PID=""; NEED_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-free-mb) GPU_FREE="$2"; shift 2;;
    --gpu-idle)    GPU_IDLE="${2:-10}"; shift 2;;
    --after-tmux)  AFTER_TMUX="$2"; shift 2;;
    --after-pid)   AFTER_PID="$2"; shift 2;;
    --need-file)   NEED_FILE="$2"; shift 2;;
    --gpu-index)   GPU_IDX="$2"; shift 2;;
    --interval)    INTERVAL="$2"; shift 2;;
    --) shift; break;;
    *) echo "arg inconnu: $1"; exit 2;;
  esac
done
[[ $# -gt 0 ]] || { echo "rien a executer apres --"; exit 2; }

q(){ nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits -i "$GPU_IDX" 2>/dev/null | head -1 | tr -d ' '; }

echo "[gate $(date)] attente des conditions..."
while :; do
  ok=1; why=""
  if [[ -n "$NEED_FILE" && ! -e "$NEED_FILE" ]]; then ok=0; why+=" need-file"; fi
  if [[ -n "$AFTER_TMUX" ]] && tmux has-session -t "$AFTER_TMUX" 2>/dev/null; then ok=0; why+=" tmux:$AFTER_TMUX"; fi
  if [[ -n "$AFTER_PID" ]] && kill -0 "$AFTER_PID" 2>/dev/null; then ok=0; why+=" pid:$AFTER_PID"; fi
  if [[ -n "$GPU_FREE" ]]; then f=$(q memory.free); [[ -n "$f" && "$f" -ge "$GPU_FREE" ]] || { ok=0; why+=" gpu_free=${f:-?}/<$GPU_FREE"; }; fi
  if [[ -n "$GPU_IDLE" ]]; then u=$(q utilization.gpu); [[ -n "$u" && "$u" -le "$GPU_IDLE" ]] || { ok=0; why+=" gpu_util=${u:-?}/>$GPU_IDLE"; }; fi
  if [[ "$ok" = 1 ]]; then echo "[gate $(date)] conditions OK -> lancement"; break; fi
  echo "[gate $(date)] pas encore :$why"; sleep "$INTERVAL"
done
exec "$@"
