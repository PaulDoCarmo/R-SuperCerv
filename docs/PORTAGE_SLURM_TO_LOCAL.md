# Portage Compute Canada / Alliance → machine à GPU interne

Le code a été écrit pour **SLURM + modules + wheelhouse** (Alliance). Rien dans les scripts
**Python** n'est spécifique à SLURM : tout passe par des arguments `--*_path` / `--*_dir`.
Ce qui doit être adapté est **autour** : environnement, chemins, soumission des jobs.

---

## 1. Environnement Python (le point le plus important)

| Alliance (ancien) | Nouvelle machine (à faire) |
|---|---|
| `module load StdEnv/2023 python/3.10` | utiliser le Python système / conda (viser **3.10**) |
| `virtualenv --no-download env` | `python3.10 -m venv env` (ou conda) |
| `pip install --no-index <pkg>` (wheelhouse local) | `pip install -r requirements-*.txt` (**internet**) |
| torch compilé CUDA 12.2 (wheelhouse) | `pip install torch==2.6.0 --index-url .../whl/cuXXX` selon `nvidia-smi` |

**Règles de portage des `create_env.sh` / `create_nnunet_env.sh` :**
1. Supprimer les lignes `module purge` / `module load ...`.
2. Remplacer `virtualenv --no-download` par `python3.10 -m venv`.
3. Retirer **tous** les `--no-index` des `pip install`.
4. Garder le fix `pip install "setuptools<81"` (pkg_resources / tensorboard).
5. nnU-Net : garder le patch `weights_only=False` (torch 2.6) et `export nnUNet_compile=f`.

Il y a **4 environnements** (garde-les séparés) :
- `rsuper_train/requirements-rsuper.txt` — resample + entraînement R-Super.
- `report_extraction/requirements.txt` — `pandas openai httpx vllm` (inférence LLM, voir ci-dessous).
- TotalSegmentator : `pip install TotalSegmentator` (voir `organ_masks/README_brain_structures.md`).
- nnU-Net : `pip install nnunetv2` (voir `nnunet/README.md`).

**Cas particulier — LLM (report_extraction)** : sur Alliance les poids étaient **pré-téléchargés**
dans `HFModels/` et servis **hors-ligne** (`HF_HUB_OFFLINE=1`, nœuds compute sans internet). Sur
cette machine avec internet, il faut **télécharger le modèle depuis HuggingFace** (`huggingface-cli
download Qwen/Qwen2.5-72B-Instruct-AWQ ...`) ou laisser vLLM le tirer (repo ID + PAS de
`HF_HUB_OFFLINE=1`). ⚠️ **VRAM** : Qwen-72B-AWQ ≈ ~40 GB → ~48 GB VRAM (sinon `--tensor-parallel-size`
multi-GPU ou modèle plus petit). Détails : `report_extraction/README.md`.

---

## 2. Chemins de données

Sur Alliance, les chemins passent par des **symlinks Alliance** qui n'existent pas ailleurs :
- `~/links/scratch/...`  → espace scratch (données).
- `~/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/...` → le **code** (ce dépôt).

**Stratégie recommandée :** définir 2 variables et les réutiliser partout.

```bash
export RSUPER_CODE=/chemin/local/vers/R-SuperCerv        # ce dépôt
export RSUPER_DATA=/chemin/local/vers/data               # racine des données locales
```

Tous les scripts Python acceptent les chemins en argument → **aucune édition de code** n'est
nécessaire, seulement passer les bons `--src_path`, `--tgt_path`, `--vols_dir`, etc.

**Seuls fichiers avec des chemins EN DUR à éditer** (défauts argparse + configs YAML) :
- `rsuper_train/config/ich/medformer_3d.yaml` (stage 1) : `data_root`.
- `rsuper_train/config/ich_ufo/medformer_3d.yaml` (stage 2) : `data_root`, `UFO_root`, `reports`.
- `report_extraction/report_to_rsuper_metadata.py` : défauts `--input` / `--out_dir`
  (ou passe-les en CLI, plus simple).

> Astuce : `grep -rn "/home/pauldcrm/links" --include=*.py --include=*.yaml --include=*.sh .`
> liste tous les chemins Alliance restants à remplacer.

---

## 3. Soumission des jobs (SLURM → exécution directe)

Les `.sh` contiennent des en-têtes `#SBATCH` et se lançaient avec `sbatch`. Sur une machine
sans SLURM, on exécute **directement**. Pour chaque script :

| Directive SLURM | Équivalent local |
|---|---|
| `#SBATCH --account=... --time=... --requeue` | (supprimer) |
| `#SBATCH --gpus-per-node=1` | `export CUDA_VISIBLE_DEVICES=0` |
| `#SBATCH --cpus-per-task=16` | `--workers 16` (déjà un arg des scripts) |
| `$SLURM_TMPDIR` | `/tmp` (ou un scratch local) |
| `sbatch resample_ich.sh` | `bash resample_ich.sh` (après avoir retiré les en-têtes + `module`) |
| `sbatch train_one.sh` | lancer `python train_ddp.py ...` directement |

Le training utilise DDP mono-GPU via `--dist_url tcp://127.0.0.1:PORT`. Sur 1 GPU local, ça
marche tel quel (`--gpu '0'`). Pour multi-GPU, ajuster `--world_size` / `--gpu '0,1'`.

**Reprise auto** : les scripts cherchent un `*_latest.pth` et ajoutent `--resume --load`.
Sur une machine stable (pas de préemption comme sur Alliance), c'est un simple confort.

---

## 4. Checklist de portage

- [ ] Créer l'env rsuper (`requirements-rsuper.txt`) + torch au bon CUDA.
- [ ] Créer l'env TotalSegmentator (GPU) et l'env nnU-Net si baseline voulue.
- [ ] Définir `RSUPER_CODE` / `RSUPER_DATA`, remplacer les chemins Alliance (grep ci-dessus).
- [ ] Éditer les 2 configs YAML + défauts de `report_to_rsuper_metadata.py`.
- [ ] Convertir chaque `.sh` utile (retirer `#SBATCH` + `module`, `bash` au lieu de `sbatch`).
- [ ] Vérifier un petit run (smoke) avant de lancer les jobs complets.
