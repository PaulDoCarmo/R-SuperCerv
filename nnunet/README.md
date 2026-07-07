# nnunet — Baseline nnU-Net v2 (multi-classes ICH/IVH/PHE)

Baseline de comparaison (segmentation supervisée classique, sans rapports). Multi-classes
softmax en **espace anisotrope natif** (auto-config nnU-Net), labels
`{background:0, ICH:1, IVH:2, PHE:3}` (Dataset001_ICH).

> Résultat Alliance : Dice médian **0.924** (ensemble) — bat MedFormer stage 1 (0.906), mais
> ce gain **confond architecture et prétraitement** (nnU-Net = anisotrope natif ; MedFormer =
> 1mm iso). À garder en tête pour l'interprétation.

## Flux
```bash
source nnunet_paths.sh              # définit nnUNet_raw / _preprocessed / _results (+ compile=f)
python make_nnunet_dataset.py \     # CT+masques → Dataset001_ICH (utilise le MÊME split held-out)
    --vols_dir $D/vols --masks_dir $D/masks \
    --trainval_ids $D/splits/trainval_ids.csv --test_ids $D/splits/test_ids.csv \
    --nnunet_raw $nnUNet_raw
bash preprocess.sh                  # nnUNetv2_plan_and_preprocess
bash train_fold.sh 0               # (launch_nnunet_folds.sh pour les 5 folds)
bash predict.sh                    # inférence sur le test held-out
python compare_methods.py          # nnU-Net vs MedFormer (Dice/NSD/HD95, Wilcoxon)
```

## Environnement (nouvelle machine)
```bash
python3.10 -m venv ~/envs/nnunet && source ~/envs/nnunet/bin/activate
pip install nnunetv2
pip install "setuptools<81"        # pkg_resources
```
Puis appliquer les 2 fixes (déjà dans `create_nnunet_env.sh`, version Alliance à adapter) :
- **torch 2.6** : patch `predict_from_raw_data.py` `weights_only=False` (chargement checkpoints).
- **triton/compile** : `export nnUNet_compile=f` (dans `nnunet_paths.sh`).

## Adaptation SLURM→local
`nnunet_paths.sh` charge des `module` Alliance → retirer, garder juste les `export nnUNet_*` et
`source` de ton env. Les `.sh` de train/predict sont des jobs SLURM → lancer en direct.
Pointer les chemins `nnUNet_raw/_preprocessed/_results` vers ton scratch local.
