#!/bin/bash
# create_env.sh -- Construit l'environnement Python "rsuper_train" sur Compute Canada / Alliance.
#
# Convention Alliance : modules systeme + virtualenv + wheels locaux (--no-index).
# PAS de conda. A lancer sur un noeud de LOGIN (acces au wheelhouse cvmfs).
#
#   bash create_env.sh                 # env par defaut (voir ENV_DIR)
#   bash create_env.sh /chemin/env     # env a un autre endroit
#
# Python 3.10 : toutes les dependances R-Super sont dispo en cp310 dans le
# wheelhouse Alliance (torch 2.6, mmcv 2.1, monai 1.5, SimpleITK 2.3, ...).

set -euo pipefail

ENV_DIR="${1:-/home/pauldcrm/links/projects/rrg-josedolz/pauldcrm/R-SuperCerv/rsuper_train/rsuper_env}"

module purge
module load StdEnv/2023 python/3.10

# 1) Creation du virtualenv (--no-download : on n'ira PAS sur PyPI, que le wheelhouse)
if [ ! -d "$ENV_DIR" ]; then
    virtualenv --no-download "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
pip install --no-index --upgrade pip

# 2) Coeur scientifique
pip install --no-index numpy scipy pandas matplotlib seaborn

# 3) Imagerie medicale + resampling (etape C : resample + nii->npz)
pip install --no-index SimpleITK nibabel PyYAML tqdm scikit-image scikit-learn

# 4) Entrainement R-Super (etape D et au-dela)
pip install --no-index torch einops timm monai batchgenerators tensorboard cvxpy mmcv

echo "==> Environnement pret : $ENV_DIR"
python - <<'PY'
import SimpleITK, nibabel, yaml, skimage, numpy
print("Deps etape C OK :", "SimpleITK", SimpleITK.Version.VersionString())
PY
