#!/usr/bin/env python3
"""Éval size-loss IVH (Dice + vol ratio) par modèle, avec CACHE des probas. Réutilisable.
Datasets : RSNA test (GT npz) + CHUM _0 train (in-sample) + CHUM _0 test (held-out, GT masque resamplé).
Usage : python eval_ivh_size_cli.py NAME1=ckpt1 NAME2=ckpt2 ...  (baseline auto-ajouté si absent)."""
import sys, os, glob, numpy as np, SimpleITK as sitk, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ich3_gen import build_args, load_model, CLASSES, D
from probs_cache import get_probs, channel
NPZ3=f"{D}/dataset_ich3_full_npz"; RNPZ=f"{D}/dataset_ich_reports_npz"; MM=f"{D}/dataset_ich_reports_1mm"
SEG=f"{D}/CT-Reports/FINAL_275_2026_08_06/matched_segmentations"; IVHgt=CLASSES.index("ivh_lesion")
margs=build_args("config/ich/medformer_3d.yaml", CLASSES)
def netfn(ck):
    box={}
    def f():
        if "n" not in box: box["n"]=load_model(margs,CLASSES,ck,use_ema=True)
        return box["n"]
    return f
def dice(a,b): a=a>0.5;b=b>0.5;s=a.sum()+b.sum();return 2*(a&b).sum()/s if s else np.nan
def resample_to(ref,m):
    r=sitk.ResampleImageFilter();r.SetReferenceImage(ref);r.SetInterpolator(sitk.sitkNearestNeighbor);r.SetDefaultPixelValue(0);return r.Execute(m)
def gt_rsna(cid):
    return np.load(f"{NPZ3}/{cid}_gt.npz")["arr_0"][IVHgt]>0
def gt_chum(cid):
    segf=f"{SEG}/{cid}.nii.gz"; mmf=f"{MM}/{cid}.nii.gz"
    if not (os.path.exists(segf) and os.path.exists(mmf)): return None
    gi=sitk.ReadImage(segf)
    if not set(np.unique(sitk.GetArrayFromImage(gi))).issubset({0,1,2,3}): return None
    return sitk.GetArrayFromImage(resample_to(sitk.ReadImage(mmf),gi))==2

tr0=set(pd.read_csv(f"{D}/splits/reports_baseline0_train_ids.csv").BDMAP_ID.astype(str))
te0=set(pd.read_csv(f"{D}/splits/reports_baseline0_test_ids.csv").BDMAP_ID.astype(str))
rsna=[str(x) for x in pd.read_csv(f"{D}/splits/test_ids.csv").iloc[:,0]]
SETS=[("RSNA test (held-out)", rsna, NPZ3, gt_rsna),
      ("CHUM _0 train (in-samp)", sorted(tr0), RNPZ, gt_chum),
      ("CHUM _0 test (held-out)", sorted(te0), RNPZ, gt_chum)]

def eval_set(model,ck,cids,npzdir,gtfn):
    fn=netfn(ck); rows=[]
    for cid in cids:
        f=f"{npzdir}/{cid}.npz"
        if not os.path.exists(f): continue
        try: gt=gtfn(cid)
        except Exception: gt=None
        if gt is None: continue
        probs,chans=get_probs(model,fn,margs,cid,f,CLASSES); p=channel(probs,chans,"ivh_lesion").numpy()
        if gt.shape!=p.shape: continue
        vgt=gt.sum()/1000.0
        if vgt<0.1: continue
        rows.append((dice(p>0.5,gt),dice(p>0.3,gt),(p>0.5).sum()/1000.0/vgt))
    a=np.array(rows); return len(a),(np.median(a,0) if len(a) else [np.nan]*3)

models=dict(a.split("=") for a in sys.argv[1:]) if len(sys.argv)>1 else {}
if "baseline" not in models: models={"baseline":f"{D}/exp/ich/ich3_stage1_X25/fold_0_best.pth", **models}
print(f"{'dataset':24s} {'model':16s} {'n':>3} {'Dice@0.5':>9} {'Dice@0.3':>9} {'volratio':>9}", flush=True)
out=[]
for ds,cids,npzdir,gtfn in SETS:
    for m,ck in models.items():
        if not os.path.exists(ck): print(f"{ds:24s} {m:16s}  [ckpt absent]"); continue
        n,(d5,d3,vr)=eval_set(m,ck,cids,npzdir,gtfn)
        print(f"{ds:24s} {m:16s} {n:3d} {d5:9.3f} {d3:9.3f} {vr:9.2f}", flush=True)
        out.append(dict(dataset=ds,model=m,n=n,dice05=round(float(d5),3),dice03=round(float(d3),3),volratio=round(float(vr),2)))
pd.DataFrame(out).to_csv(f"{D}/eval/ivh_size_summary.csv",index=False)
print(f"\n-> {D}/eval/ivh_size_summary.csv")
