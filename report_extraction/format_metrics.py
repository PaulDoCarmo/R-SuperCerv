import argparse
import os
import re
import unicodedata

import pandas as pd


MAPPING_ANATOMIQUE = {
    "front": "frontal",
    "pariet": "parietal",
    "occipit": "occipital",
    "tempor": "temporal",
    "insula": "insular",
    "lentic": "lentiform",
    "lentif": "lentiform",
    "putamen": "lentiform",
    "caudate": "caudate",
    "thalam": "thalamus",
    "cerebel": "cerebellum",
    "brainstem": "brainstem",
    "tronc": "brainstem",
    "pons": "brainstem",
    "mesencephalon": "brainstem",
}

IVH_STRUCTURES = {
    "third": "third",
    "troisieme": "third",
    "fourth": "fourth",
    "quatrieme": "fourth",
    "v3": "third",
    "v4": "fourth",
    "lateral": "lateral",
    "occipital": "lateral",
    "frontal": "lateral",
    "corne": "lateral",
}

PHE_SIZE_MAPPING = {
    "mild": "mild",
    "gret": "mild",
    "discret": "mild",
    "small": "mild",
    "tiny": "mild",
    "minime": "mild",
    "minimal": "mild",
    "leger": "mild",
    "peu": "mild",
    "moderate": "moderate",
    "modere": "moderate",
    "medium": "moderate",
    "moyen": "moderate",
    "severe": "severe",
    "large": "severe",
    "important": "severe",
    "massive": "severe",
    "massif": "severe",
    "extensive": "severe",
    "extensif": "severe",
    "volumineux": "severe",
    "grand": "severe",
}


def detect_prompt_id(path: str) -> str:
    match = re.search(r"/prompt(\d+)/", path)
    if match:
        return match.group(1)
    match = re.search(r"_prompt(\d+)", os.path.basename(path))
    if match:
        return match.group(1)
    return "U"


def build_output_path(input_path: str, output_root: str) -> str:
    prompt_id = detect_prompt_id(input_path)
    prompt_dir = os.path.join(output_root, f"prompt{prompt_id}")
    os.makedirs(prompt_dir, exist_ok=True)

    base = os.path.basename(input_path)
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".csv"

    if stem.endswith("_postprocessed"):
        stem = stem[: -len("_postprocessed")]

    out_name = f"{stem}_formated{ext}"
    return os.path.join(prompt_dir, out_name)


def list_csv_files(input_path: str) -> list[str]:
    if os.path.isdir(input_path):
        entries = [
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if name.lower().endswith(".csv")
        ]
        return sorted(entries)
    return [input_path]


def keep_only_ich_ivh_phe(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["type"].isin(["ICH", "IVH", "PHE"])].copy()


def group_by_lesion_type(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in ["Lesion Index", "certainty"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    group_cols = [c for c in ["ID", "type"] if c in df.columns]
    if len(group_cols) != 2:
        raise ValueError("Expected columns 'ID' and 'type' for grouping")

    def join_values(values: pd.Series) -> str:
        cleaned = [str(v) for v in values if pd.notna(v)]
        return "/".join(cleaned) if cleaned else "U"

    agg_map = {col: join_values for col in df.columns if col not in group_cols}
    grouped = df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    grouped["count"] = df.groupby(group_cols, dropna=False).size().values
    return grouped


def map_ich_location(df: pd.DataFrame) -> pd.DataFrame:
    if "type" not in df.columns or "structure" not in df.columns:
        return df

    def map_structure(value: object) -> object:
        if pd.isna(value):
            return value

        text = str(value)
        lowered = text.lower()
        mapped = []
        for key, target in MAPPING_ANATOMIQUE.items():
            if key in lowered and target not in mapped:
                mapped.append(target)

        return "-".join(mapped) if mapped else text

    df = df.copy()
    mask = df["type"].astype(str).str.upper() == "ICH"
    df.loc[mask, "structure"] = df.loc[mask, "structure"].apply(map_structure)
    return df


def map_ivh_location(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"type", "structure", "lateralization"}
    if not needed.issubset(df.columns):
        return df

    def map_structure(value: object, lateralization: object) -> object:
        if pd.isna(value):
            return value

        text = str(value)
        lowered = text.lower()
        mapped = []
        for key, target in IVH_STRUCTURES.items():
            if key in lowered and target not in mapped:
                mapped.append(target)

        has_ventricle = "ventricle" in lowered
        has_ventricles = "ventricles" in lowered
        has_lateral_term = any(
            term in lowered for term in ["lateral", "occipital", "frontal", "corne"]
        )
        if (has_ventricle or has_ventricles) and "lateral" not in mapped:
            if has_lateral_term or (not mapped and (has_ventricle or has_ventricles)):
                mapped.append("lateral")

        if not mapped:
            return text

        lat = str(lateralization).strip().lower()
        if "lateral" in mapped:
            if lat == "left":
                mapped = ["lateral_left" if m == "lateral" else m for m in mapped]
            elif lat == "right":
                mapped = ["lateral_right" if m == "lateral" else m for m in mapped]
            elif lat == "bilateral":
                mapped = [
                    "lateral_left-lateral_right" if m == "lateral" else m
                    for m in mapped
                ]

        return "-".join(mapped)

    def map_row(row: pd.Series) -> object:
        structure = row["structure"]
        lateralization = row["lateralization"]
        if pd.isna(structure):
            return structure

        structure_parts = [part.strip() for part in str(structure).split("/")]
        lat_parts = [part.strip() for part in str(lateralization).split("/")]
        if len(lat_parts) < len(structure_parts):
            lat_parts.extend([lat_parts[-1]] * (len(structure_parts) - len(lat_parts)))

        mapped_parts = [
            map_structure(struct_part, lat_parts[idx] if lat_parts else "")
            for idx, struct_part in enumerate(structure_parts)
        ]
        return "/".join(mapped_parts)

    df = df.copy()
    mask = df["type"].astype(str).str.upper() == "IVH"
    df.loc[mask, "structure"] = df.loc[mask].apply(map_row, axis=1)
    return df


def map_phe_severity(df: pd.DataFrame) -> pd.DataFrame:
    if "type" not in df.columns or "size" not in df.columns:
        return df

    def normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()

    def map_size(value: object) -> object:
        if pd.isna(value):
            return value

        raw = str(value)
        lowered = normalize_text(raw)
        mapped = []
        for key, target in PHE_SIZE_MAPPING.items():
            if key in lowered and target not in mapped:
                mapped.append(target)

        return "-".join(mapped) if mapped else raw

    df = df.copy()
    mask = df["type"].astype(str).str.upper() == "PHE"
    df.loc[mask, "size"] = df.loc[mask, "size"].apply(map_size)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Format metrics from post-processed CSVs.")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument(
        "--output_root",
        default="/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/format",
        help="Root output directory",
    )
    parser.add_argument(
        "--keep_only_ICH/IVH/PHE",
        dest="keep_only_ich_ivh_phe",
        action="store_true",
        help="Keep only rows where type is ICH, IVH, or PHE",
    )
    parser.add_argument(
        "--group_by_lesion_type",
        action="store_true",
        help="Group by (ID, type) and aggregate other columns with '/'",
    )
    parser.add_argument(
        "--map_ich_location",
        action="store_true",
        help="Map ICH structure values to anatomical locations",
    )
    parser.add_argument(
        "--map_ivh_location",
        action="store_true",
        help="Map IVH structure values to ventricular locations",
    )
    parser.add_argument(
        "--map_phe_severity",
        action="store_true",
        help="Map PHE size values to severity labels",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all operations",
    )
    args = parser.parse_args()

    if args.all:
        args.keep_only_ich_ivh_phe = True
        args.map_ich_location = True
        args.map_ivh_location = True
        args.map_phe_severity = True
        args.group_by_lesion_type = True

    input_files = list_csv_files(args.input)
    if not input_files:
        raise ValueError(f"No CSV files found in {args.input}")

    for input_file in input_files:
        df = pd.read_csv(input_file)

        if args.keep_only_ich_ivh_phe:
            df = keep_only_ich_ivh_phe(df)

        if args.map_ich_location:
            df = map_ich_location(df)

        if args.map_ivh_location:
            df = map_ivh_location(df)

        if args.map_phe_severity:
            df = map_phe_severity(df)

        if args.group_by_lesion_type:
            df = group_by_lesion_type(df)

        output_path = build_output_path(input_file, args.output_root)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
