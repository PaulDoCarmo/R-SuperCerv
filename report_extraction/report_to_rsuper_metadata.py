#!/usr/bin/env python3
"""
report_to_rsuper_metadata.py
Convertit le CSV LLM FORMATE (sortie de format_metrics.py) -> metadonnees R-Super
(per_tumor + per_CT) pour la supervision par rapports (stage 2).

Ne re-normalise PAS la structure (format_metrics.py l'a deja fait : frontal, parietal,
lentiform, thalamus, insular, ...). Ce script ajoute les 2 couches manquantes :
  (a) structure normalisee -> masque TotalSegmentator (chosen_segment_mask)
  (b) taille (1-2 diametres) -> volume (ABC/2, 3e axe estime) + diametres (mm)
  (c) eclatement des lesions multiples ("/") -> 1 ligne par lesion

Sorties (format proche du Merlin_per_tumor_metadata de R-Super) :
  <out>/ich_per_tumor_metadata.csv  (1 ligne par lesion)
  <out>/ich_per_CT_metadata.csv     (1 ligne par cas, pour l'eval detection)
"""
import argparse
import os
import re

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# (a) structure normalisee (sortie format_metrics) -> masque TotalSegmentator.
#     Match par SOUS-CHAINE -> robuste aux termes bruts non normalises
#     (ex. "Corona Radiata", "Posterior Limb of Internal Capsule").
#     Masques dispo (TotalSegmentator -ta brain_structures) :
#       frontal_lobe, parietal_lobe, occipital_lobe, temporal_lobe, insular_cortex,
#       lentiform_nucleus, caudate_nucleus, thalamus, cerebellum, brainstem,
#       internal_capsule, ventricle, ...
# ------------------------------------------------------------------ #
REGION_TO_TOTALSEG = {
    "frontal": "frontal_lobe",
    "parietal": "parietal_lobe",
    "occipital": "occipital_lobe",
    "temporal": "temporal_lobe",
    "insula": "insular_cortex",        # 'insular' contient 'insula'
    "lentif": "lentiform_nucleus",
    "lentic": "lentiform_nucleus",
    "putam": "lentiform_nucleus",             # putamen / putaminal
    "pallid": "lentiform_nucleus",            # globus pallidus
    "basal gangli": "lentiform_nucleus",      # ICH hypertensive typique (putamen) -> lentiforme
    "gangli": "lentiform_nucleus",            # 'ganglionic' / 'basal ganglionic'
    "caudate": "caudate_nucleus",
    "thalam": "thalamus",
    "cerebel": "cerebellum",
    "brainstem": "brainstem",
    "pons": "brainstem",
    "capsule": "internal_capsule",     # internal capsule / posterior|anterior limb
    "corona radiata": "internal_capsule",
    # IVH (ventricules) -> un seul masque 'ventricle' cote TotalSegmentator
    "ventricle": "ventricle",
    "lateral": "ventricle",
    "third": "ventricle",
    "fourth": "ventricle",
}

QUALITATIVE = {"u", "", "nan", "small", "tiny", "mild", "moderate", "severe",
               "large", "massive", "minime", "modere", "leger"}


def map_region(structure_part: str):
    """Une portion de structure (ex 'parietal-occipital') -> liste de masques TotalSeg."""
    if structure_part is None:
        return []
    text = str(structure_part).lower()
    masks = []
    for sub in re.split(r"[-]", text):          # 'parietal-occipital' -> ['parietal','occipital']
        sub = sub.strip()
        for key, mask in REGION_TO_TOTALSEG.items():
            if key in sub and mask not in masks:
                masks.append(mask)
    return masks


def parse_diameters_cm(size_part: str):
    """'8.6 x 4.9 cm' -> [8.6, 4.9] (cm) ; 'U'/'moderate' -> [] ; detecte mm."""
    if size_part is None:
        return []
    raw = str(size_part).strip().lower()
    if raw in QUALITATIVE:
        return []
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", raw)]
    if not nums:
        return []
    if "mm" in raw and "cm" not in raw:         # convertir mm -> cm
        nums = [n / 10.0 for n in nums]
    return sorted(nums, reverse=True)           # A >= B >= C


def estimate_volume_ml(diams_cm, third_axis="equal_B"):
    """
    ABC/2 (mL) a partir de 1-2-3 diametres (cm). Retourne (volume_mL, [A,B,C]_cm, known).
    - 3+ diam : A,B,C = les 3 plus grands.
    - 2 diam  : 3e axe estime (equal_B: C=B ; mean: C=(A+B)/2 ; min: C=B).
    - 1 diam  : sphere A=B=C=D.
    - 0 diam  : inconnu -> (None, [], False).
    """
    if not diams_cm:
        return None, [], False
    if len(diams_cm) >= 3:
        A, B, C = diams_cm[0], diams_cm[1], diams_cm[2]
    elif len(diams_cm) == 2:
        A, B = diams_cm[0], diams_cm[1]
        C = {"equal_B": B, "min": B, "mean": (A + B) / 2.0}.get(third_axis, B)
    else:  # 1 diametre
        A = B = C = diams_cm[0]
    vol_ml = A * B * C / 2.0                     # ABC/2 (cm^3 = mL)
    return vol_ml, [A, B, C], True


def explode(field, n):
    """Split un champ sur '/' en n parties (pad avec la derniere)."""
    parts = [p.strip() for p in str(field).split("/")] if pd.notna(field) else [""]
    if len(parts) < n:
        parts += [parts[-1]] * (n - len(parts))
    return parts[:n]


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", default="/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/format/prompt5/results_Qwen2.5-72B-Instruct-AWQ_prompt5_formated.csv")
    ap.add_argument("--out_dir", default="/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/metadata")
    ap.add_argument("--types", nargs="+", default=["ICH"], help="Types de lesion a exporter.")
    ap.add_argument("--third_axis", default="equal_B", choices=["equal_B", "mean", "min"],
                    help="Estimation du 3e diametre quand seuls 2 sont donnes.")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = pd.read_csv(a.input)
    df = df[df["type"].astype(str).str.upper().isin([t.upper() for t in a.types])].copy()

    tumor_rows = []
    for _, r in df.iterrows():
        cid = r["ID"]
        ttype = str(r["type"]).upper()
        try:
            n = int(r.get("count", 1))
        except Exception:
            n = 1
        n = max(n, 1)
        sizes = explode(r.get("size"), n)
        structs = explode(r.get("structure"), n)
        lats = explode(r.get("lateralization"), n)

        for i in range(n):
            diams = parse_diameters_cm(sizes[i])
            vol_ml, abc, known = estimate_volume_ml(diams, a.third_axis)
            masks = map_region(structs[i])
            tumor_rows.append({
                "BDMAP_ID": cid,
                "Tumor ID": f"tumor {i + 1}",
                "Tumor Type": ttype,
                "Standardized Organ": "brain",
                "structure_raw": structs[i],
                # separateur ' / ' = format natif R-Super (clean_subseg_list split sur ' / ').
                # Une lesion couvrant 2 regions -> "frontal_lobe / parietal_lobe".
                "Standardized Location": " / ".join(masks) if masks else "UNMAPPED",
                "lateralization": lats[i],
                "Tumor Size (cm)": sizes[i],
                "diameter_mm_1": round(abc[0] * 10, 1) if abc else np.nan,
                "diameter_mm_2": round(abc[1] * 10, 1) if abc else np.nan,
                "diameter_mm_3": round(abc[2] * 10, 1) if abc else np.nan,
                "volume_mm3": round(vol_ml * 1000, 1) if known else np.nan,   # 1mm iso: mm3 = voxels
                "volume_ml": round(vol_ml, 2) if known else np.nan,
                "Unknow Tumor Size": "no" if known else "yes",
                "no lesion": False,
            })

    per_tumor = pd.DataFrame(tumor_rows)
    per_tumor.to_csv(os.path.join(a.out_dir, "ich_per_tumor_metadata.csv"), index=False)

    # --- per_CT (agrege par cas, pour l'eval detection) ---
    ct_rows = []
    for cid, g in per_tumor.groupby("BDMAP_ID"):
        known = g[g["Unknow Tumor Size"] == "no"]
        ct_rows.append({
            "BDMAP_ID": cid,
            "number of ich lesion instances": len(g),
            "number with known size": len(known),
            "largest ich lesion diameter (cm)": round(g["diameter_mm_1"].max() / 10, 2) if known.shape[0] else np.nan,
            "total ich volume (ml)": round(known["volume_ml"].sum(), 2) if known.shape[0] else np.nan,
            "ich regions": ";".join(sorted(set(m for locs in g["Standardized Location"] for m in str(locs).split(" / ") if m not in ("UNMAPPED", "nan")))),
        })
    per_ct = pd.DataFrame(ct_rows)
    per_ct.to_csv(os.path.join(a.out_dir, "ich_per_CT_metadata.csv"), index=False)

    # --- resume / limites ---
    n_les = len(per_tumor)
    n_known = int((per_tumor["Unknow Tumor Size"] == "no").sum())
    n_unmapped = int((per_tumor["Standardized Location"] == "UNMAPPED").sum())
    print(f"Cas: {per_ct.shape[0]} | lesions {a.types}: {n_les}")
    print(f"  taille connue (volume calculable): {n_known}/{n_les} "
          f"({100*n_known//max(n_les,1)}%)  <- le reste sera JETE par le volume-loss (comme R-Super)")
    print(f"  structure -> masque OK: {n_les - n_unmapped}/{n_les} | UNMAPPED: {n_unmapped}")
    if n_known:
        v = per_tumor["volume_ml"].dropna().values
        print(f"  volume estime (mL): median={np.median(v):.1f}  mean={v.mean():.1f}  min={v.min():.1f}  max={v.max():.1f}")
    print(f"-> {a.out_dir}/ich_per_tumor_metadata.csv , ich_per_CT_metadata.csv")


if __name__ == "__main__":
    main()
