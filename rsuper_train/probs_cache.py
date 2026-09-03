#!/usr/bin/env python3
"""Cache des probas d'inférence (par modèle × cas) pour ne PLUS ré-inférer entre études.

L'inférence fenêtre-glissante est le coût dominant des évals. On persiste, après une seule
inférence, les canaux UTILES (lésions ICH/IVH/PHE + ventricule prédit) en float16 compressé.
Les cartes de proba étant quasi-nulles hors lésion, le npz compressé est minuscule (~0.1-0.5 MB/cas).

Toute étude (seuils volumiques, variantes de s_i, analyses par-région...) appelle get_probs()
et relit le cache si présent. fp16 : précision ~3 décimales, négligeable pour vol(>0.5) et s_i(LSE).

Cache : {D}/eval/probs/<model_name>/<cid>.npz  (clés: 'p' (K,D,H,W) fp16, 'chans' noms de canaux).
⚠ Clé = NOM du modèle : si un modèle est ré-entraîné sous le même nom, vider son dossier cache.
"""
import os, numpy as np, torch
from eval_ich3_gen import run, D

CACHE = f"{D}/eval/probs"
CHANS = ["ich_lesion", "ivh_lesion", "phe_lesion", "ventricle"]   # canaux mis en cache


def get_probs(model_name, net_fn, margs, cid, npz_path, classes):
    """Retourne (probs (K,D,H,W) float32 CPU, noms_canaux). Lit le cache sinon infère + sauve.

    net_fn : callable SANS argument renvoyant le modèle chargé — appelé UNIQUEMENT en cas de
    cache-miss (chargement paresseux -> une étude 100% cache-chaud ne touche pas le GPU).
    classes : liste des classes du modèle (pour mapper les indices des canaux)."""
    cdir = f"{CACHE}/{model_name}"
    cpath = f"{cdir}/{cid}.npz"
    if os.path.exists(cpath):
        z = np.load(cpath)
        return torch.from_numpy(z["p"].astype(np.float32)), list(z["chans"])
    idxs = [classes.index(c) for c in CHANS]
    prob = run(net_fn(), margs, np.load(npz_path)["arr_0"])       # infère seulement si miss
    p = prob[idxs].float().cpu().numpy().astype(np.float16)
    os.makedirs(cdir, exist_ok=True)
    np.savez_compressed(cpath, p=p, chans=np.array(CHANS))
    return torch.from_numpy(p.astype(np.float32)), CHANS


def channel(probs, chans, name):
    """Extrait un canal par nom depuis le tenseur (K,D,H,W)."""
    return probs[chans.index(name)]
