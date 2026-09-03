#!/usr/bin/env python3
"""Le positive-only a-t-il RÉELLEMENT poussé s_i ? — mesure par cas.

Question : positive-only optimise directement s_i (LSE pooling) vers le haut sur les
report-IVH+. On veut vérifier que s_i A MONTÉ sur ces cas (ce qui prouverait le mécanisme
"pic monté / volume rétréci"), OU au contraire qu'il n'a pas bougé (loss inopérante).

Pour chaque modèle et chaque cas report-IVH+ :
  - inférence -> p_IVH (canal ivh_lesion) au format du npz d'entrée,
  - R_pool = ventricule TotalSeg (canal 11 du _gt) dilaté 10mm CUBE (dilate_volume k=21) = R_pool EXACT du training,
  - s_i = presence_score(p_IVH, R_pool, γ=10)  [la vraie fonction du module de loss],
  - vol_ml = #(p_IVH>0.5) * volume-voxel.

Sortie : CSV par cas [cid, y, model, s_i, vol_ml] + résumé (Δs_i, Δvol vs baseline, appariés).
Usage : python eval_si_pushed.py [--all] MODELE=ckpt ...  (defaut : baseline X25 + 3 positive-only)
"""
import sys, os, glob, numpy as np, torch, pandas as pd, SimpleITK as sitk
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ich3_gen import build_args, load_model, run, CLASSES, LES, D, MM, RNPZ
from training.ivh_presence_loss import presence_score
from training.losses_foundation import dilate_volume
from eval_ivh_detection import report_flags

VENT_GT_IDX = 11      # ventricule dans le _gt npz (12 canaux)
GAMMA = 10.0
KERNEL = 21           # dilatation 10mm cube (identique au training)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def measure(name, ckpt, allids, ivh_cids, only_pos=True, sanity=False):
    margs = build_args("config/ich/medformer_3d.yaml", CLASSES)
    net = load_model(margs, CLASSES, ckpt, use_ema=True)
    ivh_i = LES["IVH"]
    rows = []
    for f in sorted(glob.glob(f"{RNPZ}/*.npz")):
        cid = os.path.basename(f)[:-4]
        if cid.endswith("_gt") or cid not in allids:
            continue
        y = int(cid in ivh_cids)
        if only_pos and y == 0:
            continue
        gtf, mmf = f[:-4] + "_gt.npz", f"{MM}/{cid}.nii.gz"
        if not (os.path.exists(gtf) and os.path.exists(mmf)):
            continue
        vox_ml = float(np.prod(sitk.ReadImage(mmf).GetSpacing())) / 1000.0
        prob = run(net, margs, np.load(f)["arr_0"])
        pivh = prob[ivh_i].float().to(DEV)                                   # (D,H,W)
        vent = torch.from_numpy(np.load(gtf)["arr_0"][VENT_GT_IDX]).float().to(DEV)
        if sanity and not rows:
            c = torch.nonzero(vent > 0.5).float()
            print(f"  [sanity] vent idx{VENT_GT_IDX}: vol={float((vent>0.5).sum())*vox_ml:.1f}mL "
                  f"centroid={c.mean(0).tolist() if len(c) else 'VIDE'} shape p_ivh={tuple(pivh.shape)} vent={tuple(vent.shape)}")
        rpool = dilate_volume(vent[None, None], KERNEL)[0, 0]
        s = float(presence_score(pivh[None], rpool[None], gamma=GAMMA)[0])
        vol = float((pivh > 0.5).sum().item()) * vox_ml
        rows.append(dict(cid=cid, y=y, model=name, s_i=round(s, 4), vol_ml=round(vol, 3)))
        if int(os.environ.get("MAXN", 0)) and len(rows) >= int(os.environ["MAXN"]):
            break
    return pd.DataFrame(rows)


def main():
    args = [a for a in sys.argv[1:] if a != "--all"]
    only_pos = "--all" not in sys.argv
    allids, ivh_cids = report_flags()
    if args:
        models = dict(a.split("=") for a in args)
    else:
        models = {
            "stage1_X25": f"{D}/exp/ich/ich3_stage1_X25/fold_0_best.pth",
            "pos_L05": f"{D}/exp/ich_ufo/ivhpres_X25_pos_L05/fold_0_latest.pth",
            "pos_L20": f"{D}/exp/ich_ufo/ivhpres_X25_pos_L20/fold_0_latest.pth",
            "pos_L50": f"{D}/exp/ich_ufo/ivhpres_X25_pos_L50/fold_0_latest.pth",
        }
    dfs = []
    for name, ck in models.items():
        if not os.path.exists(ck):
            print(f"[skip] {name}: {ck} absent"); continue
        print(f"=== {name} ===", flush=True)
        dfs.append(measure(name, ck, allids, ivh_cids, only_pos=only_pos, sanity=True))
    df = pd.concat(dfs, ignore_index=True)
    out = f"{D}/eval/ivh_si_percase.csv"; df.to_csv(out, index=False)
    print(f"\n-> {out}  ({len(df)} lignes)")

    base = "stage1_X25"
    piv_s = df.pivot_table(index="cid", columns="model", values="s_i")
    piv_v = df.pivot_table(index="cid", columns="model", values="vol_ml")
    print("\n===== s_i moyen / médian (cas report-IVH+) =====")
    for m in piv_s.columns:
        d = f"  Δ vs base med={float((piv_s[m]-piv_s[base]).median()):+.3f}" if m != base else ""
        print(f"  {m:12s} mean={piv_s[m].mean():.3f} med={piv_s[m].median():.3f}"
              f" | #s_i>0.409(τ+)={int((piv_s[m]>0.409).sum())}/{piv_s[m].notna().sum()}{d}")
    print("\n===== volume prédit mL (cas report-IVH+) =====")
    for m in piv_v.columns:
        d = f"  Δ vs base med={float((piv_v[m]-piv_v[base]).median()):+.3f}" if m != base else ""
        print(f"  {m:12s} mean={piv_v[m].mean():.2f} med={piv_v[m].median():.2f}{d}")
    if base in piv_s.columns:
        for m in [c for c in piv_s.columns if c != base]:
            up_s = int((piv_s[m] > piv_s[base]).sum()); up_v = int((piv_v[m] > piv_v[base]).sum())
            n = int(piv_s[m].notna().sum())
            print(f"\n{m}: s_i monté sur {up_s}/{n} cas | volume monté sur {up_v}/{n} cas (vs baseline)")


if __name__ == "__main__":
    os.makedirs(f"{D}/eval", exist_ok=True); main()
