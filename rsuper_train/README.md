# rsuper_train — Entraînement MedFormer (stage 1 & 2)

Cœur R-Super : MedFormer 3D + losses de segmentation, et au **stage 2** les losses
*Volume* + *Ball* qui apprennent la lésion à partir des rapports.

## Environnement
```bash
python3.10 -m venv ~/envs/rsuper && source ~/envs/rsuper/bin/activate
pip install -r requirements-rsuper.txt   # + torch au bon CUDA (cf. le fichier)
```
`create_env.sh` est la version **Alliance** (modules + `--no-index`) — à adapter (cf.
[`../docs/PORTAGE_SLURM_TO_LOCAL.md`](../docs/PORTAGE_SLURM_TO_LOCAL.md)).

## Configs (`config/<dataset>/<model>_<dimension>.yaml`)
Le chemin est dérivé de `--dataset` :
- `config/ich/medformer_3d.yaml` → **stage 1** (`--dataset ich`, `classes: 1`).
- `config/ich_ufo/medformer_3d.yaml` → **stage 2** (`--dataset ich_ufo`, `classes: 13`,
  + `data_root` Atlas, `UFO_root` rapports, `reports` CSV). **Édite ces chemins.**

Le YAML remplit les défauts ; les CLI `--data_root`, `--ufo_root`, `--reports`,
`--classes_number`, `--lr`, `--epochs` **surchargent** le YAML.

## Lancement

**Stage 1** (masques seuls) :
```bash
python train_ddp.py --dataset ich --model medformer --dimension 3d \
  --data_root $D/dataset_ich_full_npz --classes_number 13 \
  --crop_on_tumor --report_volume_loss_basic 0 \
  --gpu '0' --batch_size 2 --epochs 100 --unique_name ich_stage1_13cls
```

**Stage 2** (rapports) — repart du checkpoint stage 1 :
```bash
python train_ddp.py --dataset ich_ufo --model medformer --dimension 3d \
  --pretrained $D/exp/ich/ich_stage1_13cls/fold_0_best.pth \
  --report_volume_loss_basic 0.1 --loss ball_dice_last \
  --gpu '0' --batch_size 2 --epochs 100 --unique_name ich_stage2
```

## Méthodologie val/test
- **Validation** (EMA, `val_freq`) : sélectionne le meilleur epoch + hyperparamètres,
  sauvegarde `*_best.pth`, log `Val/Dice` (TensorBoard).
- **Test** (held-out, `make_split.py`) : touché **une seule fois** pour le rapport final,
  jamais pour la sélection. Éval via `eval_ich.py`.

## Fichiers clés
- `train_ddp.py` — boucle d'entraînement (DDP mono/multi-GPU).
- `training/losses_foundation.py` — `volume_loss_basic`, `ball_loss`, `calculate_loss`
  (génériques, 1×1×1mm iso + lésion ~sphérique). Debug gaté par `RSUPER_SANITY=1`.
- `training/dataset/dim3/dataset_ich.py` — dataset stage 1 (masques).
- `training/dataset/dim3/dataset_ich_reports.py` — dataset stage 2 (`ICHReportsDataset`,
  adapté ICH : structures=classes, `clean_ufo`, `estimate_tumor_volume`, `assign_labels`).
- `training/validation.py` — `validation_binary` (Dice binaire par canal lésion).
- `config/`, `dataset_conversion/` (voir son README), `eval_ich.py`, `plot_loss.py`.
- Grille d'hyperparamètres : `launch_grid.sh` / `train_one.sh` / `compare_grid.py`
  (Alliance/SLURM — convertir en exécution directe).

## Adaptation SLURM→local
Scripts `.sh` = jobs SLURM. Retire les `#SBATCH` + `module load`, lance en direct
(`bash ...` pour le CPU, `python train_ddp.py ...` pour le GPU). Détails dans le doc portage.
