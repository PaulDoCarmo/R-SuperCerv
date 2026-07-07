# organ_masks — TotalSegmentator `brain_structures`

Produit les **12 masques de structures cérébrales** par cas, utilisés :
- au **stage 2** pour construire le `chosen_segment_mask` (région où le rapport situe la lésion) ;
- comme **canaux du modèle** (design « structures = classes »).

## Commande
```bash
export TOTALSEG_WEIGHTS_DIR=$D/organ_masks/weights   # cache des poids (téléchargés 1×)
TotalSegmentator -i <CT>.nii.gz \
    -o $D/organ_masks/segmentations/segmented_organs_<ID>.nii.gz \
    -ta brain_structures
```
- **`-ta brain_structures`** : sort **un dossier par cas** contenant **un `.nii.gz` par
  structure** (PAS `--ml`, on veut des fichiers séparés que les scripts d'assemblage symlinkent).
- Sur les CT natifs directement (pas besoin de clip HU ni de réorientation préalable).
- **GPU recommandé** (sinon très lent).

## Sortie attendue (convention utilisée partout)
```
$D/organ_masks/segmentations/
    segmented_organs_<ID>.nii.gz/
        frontal_lobe.nii.gz   parietal_lobe.nii.gz   occipital_lobe.nii.gz
        temporal_lobe.nii.gz  insular_cortex.nii.gz  lentiform_nucleus.nii.gz
        caudate_nucleus.nii.gz thalamus.nii.gz       cerebellum.nii.gz
        brainstem.nii.gz      internal_capsule.nii.gz ventricle.nii.gz
        ...(TotalSeg produit ~16 structures ; on n'utilise que ces 12)
```
`build_ich_dataset.py --with_structures` et `build_ich_reports_dataset.py` lisent exactement
ce chemin (`--organ_masks_dir` / `--totalseg_dir`, préfixe `segmented_organs_`).

## À exécuter sur
- **Tous** les cas-rapports (dataset RAPPORTS).
- **Tous** les 359 cas-masques (dataset ATLAS 13 classes) → nécessaire pour re-train stage 1.

## Environnement (nouvelle machine)
```bash
python3.10 -m venv ~/envs/totalseg && source ~/envs/totalseg/bin/activate
pip install TotalSegmentator
```
`LaunchTotalSegmentator.sh` est la version **Alliance** (SLURM + modules `cuda/vtk` +
`--no-index`) — retirer `#SBATCH`/`module`, garder juste la boucle `run_totalsegmentator`.
Licence : `brain_structures` peut nécessiter une licence TotalSegmentator (`totalseg_set_license`).
