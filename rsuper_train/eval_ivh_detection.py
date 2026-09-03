#!/usr/bin/env python3
"""Éval DÉTECTION IVH — matrice de confusion RÉFÉRÉE AU RAPPORT (pas au masque).

Au test on n'a PAS les masques : la VÉRITÉ est donc le flag rapport IVH (prompt6, par scan,
matching exact du cid). La PRÉDICTION vient de l'output de SEGMENTATION du modèle : on binarise
p_IVH>0.5, on calcule le VOLUME IVH prédit en mL (en tenant compte du spacing voxel), et on
balaie un seuil volumique V : prédiction positive ssi vol_prédit > V.

Sortie, par modèle : un CSV, 1 ligne = 1 seuil V, colonnes [V_mL, TP, FP, TN, FN, Se, Sp, F1].
Plus, sur le score continu (vol prédit vs flag rapport) : AUC (ROC) et AUPRC.

Population = cas-rapport ayant un npz d'entrée ET un flag rapport.
Modèles : baseline stage-1 X25 + les stage-2 présence (defaut : X25 + L03, le seul complet).
⚠ Le fine-tuning a vu reports_all -> recouvrement train/éval (feasibility, à noter).
Usage : python eval_ivh_detection.py [MODELE1=ckpt MODELE2=ckpt ...]  (defaut X25 + L03)
"""
import sys, os, glob, numpy as np, pandas as pd, SimpleITK as sitk, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ich3_gen import build_args, load_model, run, CLASSES, LES, D, MM, RNPZ
from training.ivh_presence_loss import presence_score
from training.losses_foundation import dilate_volume
from probs_cache import get_probs, channel
from sklearn.metrics import roc_auc_score, average_precision_score

THRESHOLDS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]   # seuils volumiques (mL)
VENT_GT_IDX = 11    # ventricule dans le _gt npz (12 canaux) -> R_pool = dilaté 10mm (kernel 21)


def report_flags():
    ff = sorted(glob.glob(f"{D}/report_extraction/format/prompt6/*_formated.csv"))[-1]
    fmt = pd.read_csv(ff)
    ivh = set(str(i) for i, t in zip(fmt['ID'], fmt['type']) if "IVH" in str(t).upper())
    allids = set(str(i) for i in fmt['ID'])
    return allids, ivh


def predicted_volumes(name, ckpt, allids, ivh_cids):
    """Retourne un df [cid, y (flag rapport), vol_ml (volume IVH prédit), s_i (score présence)].

    UNE SEULE passe d'inférence produit à la fois vol_ml (détection) ET s_i (LSE sur le R_pool =
    ventricule TotalSeg du _gt dilaté 10mm) — évite la 2e passe séparée. s_i = NaN si _gt absent."""
    margs = build_args("config/ich/medformer_3d.yaml", CLASSES)
    _net = {}                                                       # chargement paresseux (GPU seulement si cache-miss)
    def net_fn():
        if "n" not in _net:
            _net["n"] = load_model(margs, CLASSES, ckpt, use_ema=True)
        return _net["n"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for f in sorted(glob.glob(f"{RNPZ}/*.npz")):
        cid = os.path.basename(f)[:-4]
        if cid.endswith("_gt"):
            continue
        if cid not in allids:                              # pas de rapport parsé -> hors population
            continue
        mmf = f"{MM}/{cid}.nii.gz"
        if not os.path.exists(mmf):
            continue
        vox_ml = float(np.prod(sitk.ReadImage(mmf).GetSpacing())) / 1000.0   # spacing -> mm3 -> mL
        probs, chans = get_probs(name, net_fn, margs, cid, f, CLASSES)       # cache: lit ou infère+sauve
        pivh = channel(probs, chans, "ivh_lesion")
        vol_ml = float((pivh > 0.5).sum().item()) * vox_ml
        s_i = np.nan
        gtf = f[:-4] + "_gt.npz"
        if os.path.exists(gtf):
            vent = torch.from_numpy(np.load(gtf)["arr_0"][VENT_GT_IDX]).float().to(dev)
            rpool = dilate_volume(vent[None, None], 21)[0, 0]
            s_i = round(float(presence_score(pivh.float().to(dev)[None], rpool[None], gamma=10.)[0]), 4)
        rows.append(dict(cid=cid, y=int(cid in ivh_cids), vol_ml=vol_ml, s_i=s_i))
        if int(os.environ.get("MAXN", 0)) and len(rows) >= int(os.environ["MAXN"]):
            break
    return pd.DataFrame(rows)


def sweep_csv(df, out):
    """Matrice de confusion par seuil volumique -> CSV."""
    y = df.y.values; v = df.vol_ml.values
    rows = []
    for V in THRESHOLDS:
        pred = (v > V).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        se = tp / (tp + fn) if (tp + fn) else np.nan
        sp = tn / (tn + fp) if (tn + fp) else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
        rows.append(dict(seuil_mL=V, TP=tp, FP=fp, TN=tn, FN=fn,
                         Se=round(se, 3), Sp=round(sp, 3), F1=round(f1, 3)))
    out_df = pd.DataFrame(rows); out_df.to_csv(out, index=False)
    auc = roc_auc_score(y, v) if len(set(y)) == 2 else np.nan
    auprc = average_precision_score(y, v) if len(set(y)) == 2 else np.nan
    return out_df, auc, auprc


def main():
    allids, ivh_cids = report_flags()
    models = {}
    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            k, v = a.split("="); models[k] = v
    else:                                                  # defaut : baseline + sweep positive-only HEAD CORRIGÉ
        models["stage1_X25"] = f"{D}/exp/ich/ich3_stage1_X25/fold_0_best.pth"
        for l in ("05", "20", "50"):
            models[f"posfix_L{l}"] = f"{D}/exp/ich_ufo/ivhpres_X25_posfix_L{l}/fold_0_latest.pth"

    summary, percase = [], {}
    for name, ck in models.items():
        if not os.path.exists(ck):
            print(f"[skip] {name} : {ck} absent"); continue
        print(f"=== {name} ===", flush=True)
        df = predicted_volumes(name, ck, allids, ivh_cids)
        df.to_csv(f"{D}/eval/ivh_percase_{name}.csv", index=False)     # per-cas (vol_ml + s_i)
        percase[name] = df.set_index("cid")
        out = f"{D}/eval/ivh_detection_{name}.csv"
        table, auc, auprc = sweep_csv(df, out)
        si_pos = df.loc[df.y == 1, "s_i"].median()
        print(f"  n={len(df)} (IVH+ rapport={df.y.sum()}) | AUC={auc:.3f} | AUPRC={auprc:.3f}"
              f" | s_i méd(IVH+)={si_pos:.3f} -> {out}")
        print(table.to_string(index=False))
        summary.append(dict(model=name, n=len(df), pos=int(df.y.sum()),
                            AUC=round(auc, 3), AUPRC=round(auprc, 3), si_pos_med=round(float(si_pos), 3)))

    # Δ s_i / volume vs baseline sur les cas IVH+ appariés (est-ce que la présence a POUSSÉ s_i ?)
    base = "stage1_X25"
    if base in percase:
        b = percase[base]
        print(f"\n===== Δ vs {base} sur cas report-IVH+ (s_i doit MONTER si la loss marche) =====")
        for name, d in percase.items():
            if name == base:
                continue
            common = [c for c in d.index if c in b.index and b.loc[c, "y"] == 1]
            ds = float((d.loc[common, "s_i"] - b.loc[common, "s_i"]).median())
            dv = float((d.loc[common, "vol_ml"] - b.loc[common, "vol_ml"]).median())
            up = int((d.loc[common, "s_i"] > b.loc[common, "s_i"]).sum())
            print(f"  {name:12s} Δs_i méd={ds:+.3f} | Δvol méd={dv:+.2f} mL | s_i monté sur {up}/{len(common)}")
    print("\n===== RÉSUMÉ =====")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    os.makedirs(f"{D}/eval", exist_ok=True); main()
