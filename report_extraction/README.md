# report_extraction — Rapports bruts → métadonnées R-Super (pipeline LLM)

Extrait, à partir des **rapports radiologiques (texte libre, français)**, une structure
tabulaire (type de lésion, nombre, taille, structure, latéralisation) via un **LLM local
servi par vLLM**, puis la met au format R-Super pour la supervision stage 2.

> ⚠️ Sur la **nouvelle machine** il faut **re-faire tourner cette partie** (les rapports sont
> ici, privés) — donc **re-télécharger le LLM en local** (voir §2). Sur Alliance les poids
> étaient pré-téléchargés et servis hors-ligne (`HF_HUB_OFFLINE=1`, nœuds sans internet).

---

## Chaîne complète

```
 rapports (1 JSON/cas)
     │  csv_builders/build_reports_csv.py
     ▼
 reports.csv                (colonnes : ID, Report)
     │  LaunchMonoGPU.sh  =  vLLM serve <LLM-AWQ>  +  LLM_inference/Run_LLM_inference.py --prompt_id 5
     ▼
 raw/results_<MODEL>_prompt5.csv       (extraction brute : type,count,size,structure,lateralization)
     │  postprocess.py
     ▼
 post_processed/prompt5/..._postprocessed.csv
     │  format_metrics.py               (normalise l'anatomie : front→frontal, putamen→lentiform, …)
     ▼
 format/prompt5/results_<MODEL>_prompt5_formated.csv   ←── ENTRÉE de report_to_rsuper_metadata.py
     │  report_to_rsuper_metadata.py    (voir README_rsuper_metadata.md)
     ▼
 metadata/ich_per_tumor_metadata.csv   (consommé par le training stage 2)
```

`raw_to_all_metrics.sh` orchestre postprocess → format (→ métriques d'éval optionnelles vs
vérité-terrain, non requises pour le training). Le **modèle + prompt utilisés jusqu'ici** :
**`Qwen2.5-72B-Instruct-AWQ` + `prompt5`** (fichier `results_Qwen2.5-72B-Instruct-AWQ_prompt5_formated.csv`).
Reproduis ce couple pour rester cohérent.

---

## 1. Environnement (nouvelle machine)
```bash
python3.10 -m venv ~/envs/report && source ~/envs/report/bin/activate
pip install -r report_extraction/requirements.txt      # pandas openai httpx vllm
```
vLLM tire un torch compatible CUDA ; vérifie qu'il matche ton driver (`nvidia-smi`).
`LaunchMonoGPU.sh` est la version **Alliance** (SLURM + `module load cuda/rust` + `HF_HUB_OFFLINE=1`)
— à adapter (retirer `#SBATCH`/`module`, activer internet, cf. §2).

## 2. Télécharger le LLM en local (LE point à adapter)

Sur Alliance, les poids étaient dans `HFModels/<repo>` et vLLM les lisait hors-ligne
(`HF_HUB_OFFLINE=1`). Sur ta machine **avec internet**, deux options :

**Option a — pré-télécharger explicitement (recommandé)** :
```bash
pip install "huggingface_hub[cli]"
export HF_HOME=$D/report_extraction/HFCache            # cache HF
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ \
    --local-dir $D/report_extraction/HFModels/Qwen2.5-72B-Instruct-AWQ
# (certains modèles gated -> huggingface-cli login d'abord)
```
puis pointer vLLM sur ce dossier local (comme le fait `LaunchMonoGPU.sh` via `$MODEL`).

**Option b — laisser vLLM télécharger** : passer directement le **repo ID** à `vllm serve`
(`Qwen/Qwen2.5-72B-Instruct-AWQ`) et **NE PAS** exporter `HF_HUB_OFFLINE=1`.

### Taille / VRAM (crucial)
| Modèle (`LLM_NAME`) | Repo HF | ~Poids | VRAM min |
|---|---|---|---|
| `qwen2-5-72b-awq` (utilisé) | `Qwen/Qwen2.5-72B-Instruct-AWQ` | ~40 GB | **~48 GB** (A100 80G ok ; sinon `--tensor-parallel-size N` sur plusieurs GPU) |
| `qwen2-5-32b-awq` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | ~20 GB | ~24 GB |
| `llama3-3-70b-awq` | `casperhansen/Llama-3.3-70B-Instruct-AWQ` | ~40 GB | ~48 GB |
| `llama3-8b-gptq` | `iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8` | ~9 GB | ~12 GB |

Si ton GPU interne fait **< 48 GB**, soit tu utilises `--tensor-parallel-size` sur plusieurs
GPU, soit tu prends un modèle plus petit (**mais l'extraction changera** → refaire tourner et
re-vérifier la qualité avant le stage 2). `LaunchMonoGPU.sh` connaît déjà tous ces `LLM_NAME`.

### Cas de CETTE machine : RTX 6000 Ada, 48 GB (un seul GPU)
Qwen2.5-72B-AWQ **tient**, mais c'est serré : ~40 GB de poids, il reste ~6-8 GB pour le KV
cache. Le défaut `--max-model-len 32768` = ~10 GB/séquence → **OOM**. Comme les rapports sont
courts, **réduis le contexte**. Dans `LaunchMonoGPU.sh` (cas `qwen2-5-72b-awq`), remplace
`MODEL_OPTS` par :
```
MODEL_OPTS="--dtype half --max-model-len 8192 --tensor-parallel-size 1"
# + sur la ligne `vllm serve` : --gpu_memory_utilization 0.92 --enforce-eager (déjà présent)
```
KV cache à `--max-model-len 8192` ≈ ~2.6 GB/séquence → OK. Si OOM persiste au chargement :
ajoute `--kv-cache-dtype fp8` (supporté par l'Ada, divise le KV cache par 2), ou en dernier
recours bascule sur `qwen2-5-32b-awq` (~20 GB, confortable — **extraction différente**).
Un seul GPU → **pas** de tensor-parallel (`-size 1`).

## 3. Lancer le pipeline
```bash
# a) construire reports.csv depuis les JSON de rapports (défaut --output-name reports.csv)
python csv_builders/build_reports_csv.py $D/reports_json $D/report_extraction

# b) inférence LLM (adapter LaunchMonoGPU.sh en exécution directe) :
#    - lance `vllm serve <MODEL_LOCAL> --port P --gpu_memory_utilization 0.9 --enforce-eager ...`
#    - attend l'API, puis :
python LLM_inference/Run_LLM_inference.py --port P \
    --data_path $D/report_extraction/reports.csv \
    --save_path $D/report_extraction/raw/results.csv --prompt_id 5

# c) postprocess + format -> *_formated.csv  (chemin DIRECT, sans vérité-terrain)
python postprocess.py --input $D/report_extraction/raw/results_<MODEL>_prompt5.csv \
    --output_root $D/report_extraction/post_processed
python format_metrics.py --all \
    --input $D/report_extraction/post_processed/prompt5/results_<MODEL>_prompt5_postprocessed.csv \
    --output_root $D/report_extraction/format
#    -> format/prompt5/results_<MODEL>_prompt5_formated.csv
#    (raw_to_all_metrics.sh fait la même chose PUIS l'éval vs vérité-terrain, qui EXIGE un
#     ground_truth.csv — inutile pour le training, ne l'utilise que pour valider l'extraction)

# d) métadonnées R-Super
python report_to_rsuper_metadata.py \
    --input $D/report_extraction/format/prompt5/results_<MODEL>_prompt5_formated.csv \
    --out_dir $D/report_extraction/metadata --types ICH
```

## Fichiers
- `csv_builders/build_reports_csv.py` — JSON rapports → `reports.csv` (ID, Report).
- `LLM_inference/LLM_inference.py` — SYSTEM_PROMPT + `USER_PROMPT_1..N` (schéma d'extraction),
  appel API OpenAI-compatible (vLLM), parsing.
- `LLM_inference/Run_LLM_inference.py` — boucle sur les rapports (args `--port/--data_path/--save_path/--prompt_id`).
- `LaunchMonoGPU.sh` — sert le LLM (vLLM) + lance l'inférence (**version SLURM à porter**).
- `postprocess.py`, `format_metrics.py` — nettoyage + normalisation anatomique.
- `raw_to_all_metrics.sh`, `run_all_metrics.sh`, `compute_metrics/` — éval qualité extraction (optionnel).
- `report_to_rsuper_metadata.py` — étape finale → métadonnées (voir [`README_rsuper_metadata.md`](README_rsuper_metadata.md)).

## Sécurité
Rapports = **données privées**. Tout tourne **en local** (LLM inclus, aucune API externe).
Garder les CSV intermédiaires sur cette machine.
