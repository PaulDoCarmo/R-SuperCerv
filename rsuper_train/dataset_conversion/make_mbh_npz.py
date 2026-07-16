#!/usr/bin/env python3
"""Prétraite les CT MBH (image seule, pas de masque) EXACTEMENT comme notre pipeline
d'entrainement : reorient RAI -> resample 1mm (XY BSpline, Z NN separate-z) -> clip HU
[hu_min,hu_max] -> z-score -> npz. Exclut les patients en fuite (deja dans notre training).
Sortie : <out_dir>/<ID>.npz (image, cle 'arr_0'), lisible tel quel par l'inference."""
import os, glob, argparse
import numpy as np, SimpleITK as sitk, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resample_utils import ResampleXYZAxis, reorient_image


def preprocess(src, dst, hu_min, hu_max, spacing=(1., 1., 1.)):
    im = reorient_image(sitk.ReadImage(src), "RAI")
    sp = im.GetSpacing()
    im_xy = ResampleXYZAxis(im, space=(spacing[0], spacing[1], sp[2]), interp=sitk.sitkBSpline)
    im_1mm = ResampleXYZAxis(im_xy, space=spacing, interp=sitk.sitkNearestNeighbor)  # separate-z
    a = sitk.GetArrayFromImage(im_1mm).astype(np.float32)     # (Z,Y,X)
    a = np.clip(a, hu_min, hu_max)
    m, s = float(a.mean()), float(a.std()); s = s if s > 1e-8 else 1.0
    a = (a - m) / s
    np.savez_compressed(dst, a.astype(np.float32))


def work(t):
    src, dst, hu_min, hu_max = t
    if os.path.exists(dst):
        return ("skip", os.path.basename(dst))
    try:
        preprocess(src, dst, hu_min, hu_max); return ("ok", os.path.basename(dst))
    except Exception as e:
        return ("err", f"{os.path.basename(src)}: {repr(e)[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbh_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude_patients", help="CSV col 'patient' (1er ID) a exclure (fuite)")
    ap.add_argument("--hu_min", type=float, default=0.0)
    ap.add_argument("--hu_max", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    excl = set()
    if a.exclude_patients and os.path.exists(a.exclude_patients):
        excl = set(pd.read_csv(a.exclude_patients)["patient"])
    tasks = []
    for f in sorted(glob.glob(f"{a.mbh_dir}/*.nii.gz")):
        cid = os.path.basename(f)[:-7]
        if cid.split("_ID_")[0] in excl:
            continue
        tasks.append((f, f"{a.out_dir}/{cid}.npz", a.hu_min, a.hu_max))
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"À traiter (hors fuite) : {len(tasks)} | workers={a.workers}", flush=True)
    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            st, msg = fu.result()
            ok += st == "ok"; skip += st == "skip"; err += st == "err"
            if st == "err":
                print("  ERR", msg, flush=True)
            if i % 200 == 0:
                print(f"  ... {i}/{len(tasks)} (ok={ok} skip={skip} err={err})", flush=True)
    print(f"TERMINE ok={ok} skip={skip} err={err} -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
