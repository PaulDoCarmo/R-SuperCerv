#!/usr/bin/env python3
"""Eval generalisation par classe (ICH/IVH/PHE) des modeles stage-1 15-classes.
Compare Dice sur : (a) test set RSNA (in-domain, _gt.npz 15-ch) vs (b) GT-rapports CHUM
(hors-domaine, masque natif resample a la volee). Sortie: CSV + medianes par classe/domaine/modele.
Lancer depuis rsuper_train/ apres le sweep15."""
import sys, os, glob, yaml, numpy as np, pandas as pd, torch, SimpleITK as sitk
D = os.environ.get("RSUPER_DATA", "/mnt/Data/data_paul")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ich import build_args, load_model
from inference.inference3d import inference_sliding_window

NPZ3 = f"{D}/dataset_ich3_full_npz"; RNPZ = f"{D}/dataset_ich_reports_npz"; MM = f"{D}/dataset_ich_reports_1mm"
SEG = f"{D}/CT-Reports/FINAL - Batch 1/matched_segmentations"
CLASSES = yaml.safe_load(open(f"{NPZ3}/list/label_names.yaml"))          # 15, tries
LES = {"ICH": CLASSES.index("ich_lesion"), "IVH": CLASSES.index("ivh_lesion"), "PHE": CLASSES.index("phe_lesion")}
LAB = {"ICH": 1, "IVH": 2, "PHE": 3}

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan

def run(net, margs, img):
    t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda().float()
    with torch.no_grad():
        p = inference_sliding_window(net, t, margs)
    del t; torch.cuda.empty_cache()
    return p[0]  # (C,Z,Y,X)

def eval_model(name, ckpt):
    margs = build_args(f"{D}/config_placeholder", CLASSES) if False else build_args("config/ich/medformer_3d.yaml", CLASSES)
    net = load_model(margs, CLASSES, ckpt, use_ema=True)
    rows = []
    # (a) RSNA test
    test_ids = [l.strip() for l in open(f"{D}/splits/test_ids.csv") if l.strip().startswith("ID_")]
    for cid in test_ids:
        f = f"{NPZ3}/{cid}.npz"; g = f"{NPZ3}/{cid}_gt.npz"
        if not (os.path.exists(f) and os.path.exists(g)):
            continue
        prob = run(net, margs, np.load(f)["arr_0"]); gt = np.load(g)["arr_0"]
        for cl, ci in LES.items():
            if gt[ci].sum() > 50:
                rows.append(dict(model=name, domain="RSNA_test", cls=cl,
                                 dice=dice((prob[ci].numpy() > 0.5), gt[ci] > 0)))
    # (b) CHUM reports (_0 valides)
    valid = list(pd.read_csv(f"{D}/report_extraction/valid_mask_ids.csv")["BDMAP_ID"])
    for cid in valid:
        f = f"{RNPZ}/{cid}.npz"; mm = f"{MM}/{cid}.nii.gz"; sg = f"{SEG}/{cid}.nii.gz"
        if not all(os.path.exists(x) for x in [f, mm, sg]):
            continue
        prob = run(net, margs, np.load(f)["arr_0"])
        ref = sitk.ReadImage(mm); segn = sitk.ReadImage(sg)
        gt = sitk.GetArrayFromImage(sitk.Resample(segn, ref, sitk.Transform(),
                                                   sitk.sitkNearestNeighbor, 0, segn.GetPixelID()))
        for cl, ci in LES.items():
            gtc = (gt == LAB[cl])
            if gtc.sum() > 50:
                rows.append(dict(model=name, domain="CHUM_report", cls=cl,
                                 dice=dice((prob[ci].numpy() > 0.5), gtc)))
    return rows

def main():
    # (nom_dossier, checkpoint) : X5/X10 = petits pools sans best.pth -> latest.pth (epoch 60).
    models = {"X5": ("ich3_stage1_X5", "fold_0_latest.pth"),
              "X10": ("ich3_stage1_X10", "fold_0_latest.pth"),
              "X25": ("ich3_stage1_X25", "fold_0_best.pth"),
              "X50": ("ich3_stage1_X50", "fold_0_best.pth"),
              "X100": ("ich3_stage1_X100", "fold_0_best.pth"),
              "X305": ("ich3_stage1_X305", "fold_0_best.pth")}
    csv_path = f"{D}/eval/ich3_generalization.csv"
    existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    done = set(existing["model"].unique()) if len(existing) else set()
    allrows = []
    for tag, (nm, ckname) in models.items():
        if tag in done:
            print(f"[skip] {tag}: deja dans le CSV"); continue
        ck = f"{D}/exp/ich/{nm}/{ckname}"
        if not os.path.exists(ck):
            print(f"[skip] {tag}: pas de checkpoint ({ck})"); continue
        print(f"=== eval {tag} ({ckname}) ===", flush=True)
        allrows += eval_model(tag, ck)
    df = pd.concat([existing, pd.DataFrame(allrows)], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print("\n===== DICE MEDIAN par modele x domaine x classe =====")
    piv = df.pivot_table(index=["model", "cls"], columns="domain", values="dice", aggfunc="median")
    print(piv.round(3).to_string())
    print("\n(n cas par cellule)")
    print(df.pivot_table(index=["model", "cls"], columns="domain", values="dice", aggfunc="count").to_string())

if __name__ == "__main__":
    os.makedirs(f"{D}/eval", exist_ok=True); main()
