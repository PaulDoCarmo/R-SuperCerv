# report_extraction — Rapports → métadonnées R-Super

L'extraction LLM des rapports est **FAITE** (sortie : `*_formated.csv`, produit par
`format_metrics.py`, avec les colonnes `ID, type, count, size, structure, lateralization`).
Il reste **une** étape avant l'entraînement stage 2 : convertir ce CSV en métadonnées R-Super.

## `report_to_rsuper_metadata.py`

Convertit `*_formated.csv` → métadonnées per-lésion + per-CT consommées par la Volume/Ball loss.

```bash
python report_to_rsuper_metadata.py \
    --input  $D/report_extraction/.../results_..._formated.csv \
    --out_dir $D/report_extraction/metadata \
    --types  ICH \
    --third_axis equal_B      # estimation du 3e diamètre quand 2 donnés (equal_B|mean|min)
```

### Ce qu'il fait (2 couches ajoutées, la normalisation étant déjà faite par format_metrics.py)
1. **Structure → masque TotalSegmentator** (`REGION_TO_TOTALSEG`, match par sous-chaîne) :
   `frontal→frontal_lobe`, `putamen/pallidum/basal ganglia→lentiform_nucleus`,
   `capsule/corona radiata→internal_capsule`, `ventricle/lateral/third/fourth→ventricle`, …
   Une région non reconnue → **`UNMAPPED`** (la lésion sera jetée par `clean_ufo`).
2. **Taille (1–2 diamètres) → volume** via **ABC/2** (formule clinique ICH). 3e axe estimé si
   absent ; 1 seul diamètre → sphère. À 1mm iso, `volume_mm3` = nb de voxels (ce qu'attend la loss).
3. **Lésions multiples** (`/` dans les champs) → 1 ligne par lésion.

### Sorties (dans `--out_dir`)
- **`ich_per_tumor_metadata.csv`** (1 ligne/lésion) — colonnes clés consommées par
  `ICHReportsDataset` : `BDMAP_ID`, `Standardized Organ` (=`brain`),
  `Standardized Location` (masques joints par **`' / '`**), `Unknow Tumor Size` (`no`/`yes`),
  `volume_mm3`, `diameter_mm_{1,2,3}`, `no lesion`.
- **`ich_per_CT_metadata.csv`** (1 ligne/cas) — pour l'éval *détection*.

### Vérifs à faire sur le résumé imprimé
- **% tailles connues** : le reste (`Unknow Tumor Size = yes`) est **jeté** (volume non supervisable).
- **`UNMAPPED`** : si beaucoup, enrichir `REGION_TO_TOTALSEG` avec le vocabulaire réel des rapports
  (le script imprime le compte). C'est le principal levier de rétention de données.

## Points importants
- **`BDMAP_ID`** = l'`ID` du CSV = doit correspondre au **stem `<ID>` des npz** du dataset
  RAPPORTS (sinon `read_report` ne trouvera pas le cas). Vérifie la cohérence de nommage.
- **Séparateur `' / '`** entre régions : format **natif R-Super** attendu par le dataset. Ne pas
  le remplacer par `|`/`;`.
- Les rapports sont **privés** : garder les sorties sur la machine interne.
