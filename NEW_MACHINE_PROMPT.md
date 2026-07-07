# Prompt de passation — nouvelle conversation Claude (machine GPU interne)

Copie-colle le bloc ci-dessous comme premier message de ta nouvelle conversation.

---

Je travaille sur **R-SuperCerv** : la transposition de **R-Super** (MICCAI 2025 — apprentissage
de la segmentation de tumeurs à partir de **rapports radiologiques** via *Volume Loss* + *Ball
Loss*) à la **segmentation d'hémorragie intracrânienne (ICH)** sur CT cérébral. Tu as accès à
cette codebase **et** au dépôt R-Super original.

**Objectif scientifique** : montrer la valeur ajoutée (et les limites) des rapports écrits comme
supervision faible, face à une baseline **nnU-Net**. Pipeline en 2 stages : stage 1 = MedFormer
sur masques seuls ; stage 2 = fine-tuning avec la supervision-rapports (Volume/Ball loss).

**Où j'en suis** (tout le détail est dans les READMEs du repo — LIS-LES d'ABORD) :
- ✅ Stage 1 (masques), baseline nnU-Net, et extraction LLM des rapports : FAITS.
- ✅ Code du **stage 2** (supervision par rapports) : **complet et validé** là où c'était
  testable sans les données privées. Fichiers clés : `report_extraction/report_to_rsuper_metadata.py`,
  `rsuper_train/training/dataset/dim3/dataset_ich_reports.py` (`ICHReportsDataset`),
  `rsuper_train/config/ich_ufo/medformer_3d.yaml`.
- ⏳ Il reste à **exécuter le pipeline stage 2 de bout en bout**, ce qui nécessite les données.

**Cette machine** : GPU **interne** (PAS Compute Canada/Alliance) qui a accès aux **rapports
radiologiques privés**. Le code a été écrit pour **SLURM + modules + wheelhouse `--no-index`**.

**Je referai tourner `report_extraction` sur cette machine** (les rapports sont ici, privés) →
il faut **re-télécharger le LLM en local** (sur Alliance il était pré-téléchargé + servi
hors-ligne). Modèle : `Qwen/Qwen2.5-72B-Instruct-AWQ` + prompt5, servi par **vLLM** (~40 GB).
**Matériel : 1× RTX 6000 Ada, 48 GB VRAM.** Le 72B tient à condition de réduire le contexte
(`--max-model-len 8192` au lieu de 32768, sinon OOM sur le KV cache) ; 1 seul GPU → pas de
tensor-parallel. Config exacte dans `report_extraction/README.md` (§ RTX 6000 Ada).

**Décision de design tranchée : on reste sur l'OPTION A** (fidèle à R-Super) — modèle **13
classes** (12 structures + `ich_lesion`), TotalSegmentator sur **tous** les CT (masques ET
rapports), re-train stage-1 en 13 classes. (On a vérifié dans le repo original que les organes
sont bien supervisés sur les deux datasets ; `balance_supervision=True` = ~50 % masques / 50 %
rapports par batch.)

**Ton défi principal = PORTAGE SLURM → cette machine**, sur 2 axes :
1. **Environnements / dépendances** : recréer les envs (Python 3.10) avec `pip` + internet à
   partir des `requirements-*.txt` fournis, en installant le **torch qui matche le CUDA local**
   (`nvidia-smi`). Retirer `module load` et `--no-index`. Garder le fix `setuptools<81`.
   Il y a **4 envs** : rsuper (training), report (vLLM/LLM), totalseg, nnunet.
2. **Chemins de données** : les scripts utilisent des chemins Alliance en dur
   (`/home/pauldcrm/links/scratch/...`, `links/projects/...`). Les scripts **Python** prennent
   tous leurs chemins en argument → il faut surtout : (a) inventorier ce qui est présent sur la
   machine, (b) définir `RSUPER_DATA`, (c) éditer 2 configs YAML + les défauts de
   `report_to_rsuper_metadata.py`, (d) convertir les `.sh` SLURM en exécution directe.
   `grep -rn "/home/pauldcrm/links" --include=*.py --include=*.yaml --include=*.sh .` liste tout.

**Documents à lire dans l'ordre** :
1. `README.md` (racine) — vue d'ensemble + carte du dépôt.
2. `docs/PORTAGE_SLURM_TO_LOCAL.md` — **ton guide principal** (envs, chemins, SLURM→direct).
3. `STAGE2_RUNBOOK.md` — les étapes ordonnées à exécuter (0 → 7).
4. Les READMEs par dossier : `rsuper_train/`, `rsuper_train/dataset_conversion/`,
   `report_extraction/README.md` (**pipeline LLM + téléchargement local du modèle**),
   `organ_masks/`, `nnunet/`.

**Détails importants à ne pas casser** :
- **Design « structures = classes »** : le modèle stage 2 a **13 classes** = 12 structures
  cérébrales (TotalSegmentator `brain_structures`) + `ich_lesion`. ⇒ il faut **ré-entraîner le
  stage 1 avec 13 classes** (mode `build_ich_dataset.py --with_structures`) pour que le
  checkpoint `--pretrained` corresponde à la tête du stage 2.
- Métadonnées rapports : séparateur des régions = **`' / '`** (natif R-Super) ; lésions à
  taille inconnue ou région **`UNMAPPED`** sont **volontairement jetées** par `clean_ufo`.
- `BDMAP_ID` = simple **ID de cas** (= stem du `.npz`), pas de format « BDMAP » imposé.
- Les **rapports sont privés** → ne rien exfiltrer, garder les sorties sur cette machine.

**Commence par** : (1) lire `README.md` + `docs/PORTAGE_SLURM_TO_LOCAL.md` ; (2) faire un
inventaire de ce qui est présent sur la machine : les JSON de rapports + les CT-rapports ? les
359 cas-masques `data_laurent` (vols+masks) ? le GPU visible (`nvidia-smi` → CUDA du driver) ? ;
(3) me proposer un plan de portage (4 envs + chemins + téléchargement LLM) avant de lancer quoi
que ce soit. Explique-moi ce que tu fais au fur et à mesure.

---
