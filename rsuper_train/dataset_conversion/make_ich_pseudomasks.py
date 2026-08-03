#!/usr/bin/env python3
"""Precalcule les PSEUDO-MASQUES ICH par intensite pour les cas-rapports, alignes sur la
grille des npz d'entrainement, et construit un dataset-rapports ou l'ICH est un canal
per-voxel (le pseudo-masque). Sert d'ANCRE SPATIALE (terme H de Kervadec) pour la size-loss.

Pseudo-masque (valide, Dice ~0.73 vs GT sur les cas _0) : plus grosse composante connexe
COMPACTE (rayon inscrit >=3mm) de sang [50,85] HU dans l'hemisphere rapporte (cavite
16-structures resamplee, erodee 3mm), matchee au volume rapporte V_r.

Sortie : <out>/<cid>.npz (symlink image) + <cid>_gt.npz (13 canaux = 12 structures + ich_lesion
= pseudo-masque, ordre trie) + list/label_names.yaml (13). Pointer --UFO_root dessus au stage-2.

Reproductible : chemins parametres.
"""
import argparse, glob, os, yaml
import numpy as np, SimpleITK as sitk, pandas as pd
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, label as cclabel, generate_binary_structure, distance_transform_edt

ALL16 = ["brainstem","caudate_nucleus","central_sulcus","cerebellum","frontal_lobe","insular_cortex",
         "internal_capsule","lentiform_nucleus","occipital_lobe","parietal_lobe","septum_pellucidum",
         "subarachnoid_space","temporal_lobe","thalamus","venous_sinuses","ventricle"]
S6 = generate_binary_structure(3, 1)


def norm_lat(s):
    s = str(s).strip().strip('.').lower()
    return 'Left' if s.startswith('left') else ('Right' if (s.startswith('right') or 'parasagittal right' in s) else 'Other')


def pseudo_mask(hu, cavity, side, Vr, minsz_vox):
    """hu,cavity : (Z,Y,X) 1mm. side in {Left,Right,Other}. Vr en voxels."""
    core = distance_transform_edt(cavity, sampling=(1, 1, 1)) > 3
    if side in ('Left', 'Right'):
        # RAI via sitk : axe 2 = R (croissant vers la droite patient)
        idx = np.where(cavity.any(axis=(0, 1)))[0]; mid = (idx.min() + idx.max()) / 2
        c2 = np.arange(cavity.shape[2]).reshape(1, 1, -1)
        right = c2 > mid
        core = core & (right if side == 'Right' else ~right)
    blood = core & (hu >= 50) & (hu <= 85)
    lab, n = cclabel(blood, S6)
    if n == 0:
        return np.zeros_like(cavity, np.uint8)
    size = np.bincount(lab.ravel(), minlength=n + 1); size[0] = 0
    edt = distance_transform_edt(blood, sampling=(1, 1, 1))
    inr = ndimage.maximum(edt, labels=lab, index=np.arange(1, n + 1)); inrf = np.zeros(n + 1); inrf[1:] = inr
    elig = np.where((size >= minsz_vox) & (inrf >= 3.0))[0]
    if len(elig) == 0:
        elig = np.where(size >= minsz_vox)[0]
    if len(elig) == 0:
        return np.zeros_like(cavity, np.uint8)
    k = elig[np.argmin(np.abs(size[elig] - Vr))]
    return binary_fill_holes(lab == k).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_npz", required=True, help="dataset_ich_reports_npz (image+_gt 12 struct)")
    ap.add_argument("--reports_1mm", required=True, help="dataset_ich_reports_1mm/<cid>.nii.gz (HU 1mm)")
    ap.add_argument("--organ_masks", required=True, help="organ_masks/segmentations (structures natives)")
    ap.add_argument("--reports_csv", required=True, help="ich_per_tumor_metadata.csv (volumes, lateralite)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio_gate", type=float, nargs=2, default=[0.15, 1.3], metavar=("LO", "HI"),
                    help="Gate qualite : garde le pseudo-masque SEULEMENT si vol(pseudo)/Vr in [LO,HI]. "
                         "Sinon -> pseudo-masque VIDE (mauvaise composante connexe : precision ancre ~0). "
                         "Etude 65 GT (erosion 4mm) : [0.15,1.3] = 46 cas gardes, 91% prec=1.0, 1 seule "
                         "catastrophe (7 vox). LO bas car le pseudo SOUS-capture (bons cas ratio 0.24-0.85). "
                         "[0,1e9] pour desactiver.")
    a = ap.parse_args()
    RLO, RHI = a.ratio_gate

    md = pd.read_csv(a.reports_csv)
    md["vmm3"] = pd.to_numeric(md["volume_mm3"], errors="coerce")
    md["lat"] = md["lateralization"].map(norm_lat)
    vr = md[md.vmm3.notna()].groupby("BDMAP_ID")["vmm3"].sum()
    lat = md.groupby("BDMAP_ID")["lat"].agg(lambda x: x.mode().iat[0] if len(x.mode()) else 'Other')
    old = yaml.safe_load(open(f"{a.reports_npz}/list/label_names.yaml"))   # 12 structures
    new = sorted(old + ["ich_lesion"]); ich_pos = new.index("ich_lesion")
    os.makedirs(f"{a.out}/list", exist_ok=True)
    yaml.safe_dump(new, open(f"{a.out}/list/label_names.yaml", "w"))

    ids = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{a.reports_npz}/*_gt.npz"))
    ok = miss = gated = 0; ncov = []
    for i, cid in enumerate(ids, 1):
        try:
            if cid not in vr.index:      # pas de volume -> pseudo-masque vide (ICH connu = rien)
                Vr = 0
            else:
                Vr = int(round(vr[cid]))
            mmf = f"{a.reports_1mm}/{cid}.nii.gz"; od = f"{a.organ_masks}/segmented_organs_{cid}.nii.gz"
            gt12 = np.load(f"{a.reports_npz}/{cid}_gt.npz")["arr_0"]           # (12,Z,Y,X)
            ref = sitk.ReadImage(mmf); hu = sitk.GetArrayFromImage(ref).astype(np.float32)
            # cavite 16-struct native -> resample 1mm
            cav = None
            for s in ALL16:
                f = f"{od}/{s}.nii.gz"
                if os.path.exists(f):
                    m = sitk.GetArrayFromImage(sitk.ReadImage(f)) > 0
                    cav = m if cav is None else (cav | m)
            cs = sitk.GetImageFromArray(binary_fill_holes(cav).astype(np.uint8)); cs.CopyInformation(sitk.ReadImage(f"{od}/{ALL16[0]}.nii.gz"))
            cav1 = sitk.GetArrayFromImage(sitk.Resample(cs, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0)) > 0
            if hu.shape != gt12.shape[1:]:
                raise ValueError(f"shape hu {hu.shape} vs gt {gt12.shape}")
            pm = pseudo_mask(hu, cav1, lat.get(cid, 'Other'), Vr, minsz_vox=1000) if Vr > 0 else np.zeros(gt12.shape[1:], np.uint8)
            # GATE qualite : jette le pseudo-masque si le volume ne matche pas Vr (mauvaise
            # composante -> ancre fausse). Cas gate = pseudo VIDE -> pas de BCE-ancre au training.
            if Vr > 0 and pm.sum() > 0:
                ratio = pm.sum() / Vr
                if not (RLO <= ratio <= RHI):
                    pm = np.zeros_like(pm); gated += 1
            # inserer a la position triee de ich_lesion
            newgt = np.concatenate([gt12[:ich_pos], pm[None].astype(gt12.dtype), gt12[ich_pos:]], axis=0)
            assert newgt.shape[0] == len(new)
            np.savez_compressed(f"{a.out}/{cid}_gt.npz", newgt)
            img = f"{a.out}/{cid}.npz"
            if not os.path.lexists(img):
                os.symlink(f"{a.reports_npz}/{cid}.npz", img)
            if Vr > 0 and pm.sum() > 0:
                ncov.append(pm.sum() / max(Vr, 1))
            ok += 1
        except Exception as e:
            miss += 1; print("ERR", cid, repr(e)[:90], flush=True)
        if i % 60 == 0:
            print(f"  {i}/{len(ids)} (ok={ok})", flush=True)
    yaml.safe_dump(ids, open(f"{a.out}/list/dataset.yaml", "w"))
    print(f"OK {ok}/{len(ids)} (err {miss}) | gate ratio [{RLO},{RHI}] -> {gated} pseudo-masques jetes (ancre fausse)")
    print(f"ratio médian vol(pseudo)/Vr (cas gardes) = {np.median(ncov):.2f}")
    print(f"-> {a.out}  (ich_lesion au canal {ich_pos})")


if __name__ == "__main__":
    main()
