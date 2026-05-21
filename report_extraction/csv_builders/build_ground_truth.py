#!/usr/bin/env python3
"""Build a CSV with ground-truth flags from booleans and measurements."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


PHE_INTENSITY_COLUMNS = ("phe_mild", "phe_moderate", "phe_severe")
IVH_PREFIX = "ivh_"
LOC_PREFIX = "loc_"
PHE_NONE_COLUMN = "phe_none"
IVH_NONE_COLUMN = "ivh_none"
TRUE_VALUES = {"true", "1", "yes", "y", "t"}
UNIT_TO_CM = {
    "mm": 0.1,
    "millimeter": 0.1,
    "millimetre": 0.1,
    "cm": 1.0,
    "centimeter": 1.0,
    "centimetre": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CSV with ground-truth flags from booleans and measurements."
    )
    parser.add_argument("output_folder", help="Folder to write the CSV")
    parser.add_argument(
        "--booleans",
        help="Path to checklist_booleans.csv",
    )
    parser.add_argument(
        "--measurements",
        help="Path to measurements.csv",
    )
    parser.add_argument(
        "--output-name",
        default="ground_truth_from_json.csv",
        help="CSV filename to create in output_folder (default: ground_truth_from_json.csv)",
    )
    parser.add_argument(
        "--all-ich-true",
        action="store_true",
        help="Force ich_presence to True for all rows",
    )
    return parser.parse_args()


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def parse_bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in TRUE_VALUES
    return False


def coerce_bool_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=df.index)
    output = {}
    for column in columns:
        if column in df.columns:
            output[column] = df[column].map(parse_bool_value)
    return pd.DataFrame(output, index=df.index)


def find_columns_by_prefix(
    columns: Iterable[str],
    prefix: str,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    exclude_set = set(exclude or [])
    return [name for name in columns if name.startswith(prefix) and name not in exclude_set]


def compute_any_true(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    bool_frame = coerce_bool_frame(df, columns)
    if bool_frame.empty:
        return pd.Series([False] * len(df), index=df.index)
    return bool_frame.any(axis=1)


def compute_all_false(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    bool_frame = coerce_bool_frame(df, columns)
    if bool_frame.empty:
        return pd.Series([True] * len(df), index=df.index)
    return ~bool_frame.any(axis=1)


def compute_presence_with_none(
    df: pd.DataFrame,
    columns: Iterable[str],
    none_column: str,
) -> pd.Series:
    none_series = (
        df[none_column].map(parse_bool_value)
        if none_column in df.columns
        else pd.Series([False] * len(df), index=df.index)
    )
    return (~none_series) & compute_any_true(df, columns)


def compute_multi_label(
    df: pd.DataFrame,
    columns: Iterable[str],
    label_prefix: str,
) -> pd.Series:
    bool_frame = coerce_bool_frame(df, columns)
    if bool_frame.empty:
        return pd.Series([None] * len(df), index=df.index)

    labels = [name.split(label_prefix, 1)[1] for name in columns]

    def join_labels(row: pd.Series) -> str | None:
        chosen = [label for label, is_true in zip(labels, row) if is_true]
        if not chosen:
            return None
        return "/".join(chosen)

    return bool_frame.apply(join_labels, axis=1)


def compute_phe_intensity(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    return compute_multi_label(df, columns, "phe_")


def build_booleans_output(
    df: pd.DataFrame,
    all_ich_true: bool,
) -> pd.DataFrame:
    boolean_columns = [name for name in df.columns if name != "ID"]
    loc_columns = find_columns_by_prefix(df.columns, LOC_PREFIX)
    ivh_columns = find_columns_by_prefix(
        df.columns,
        IVH_PREFIX,
        exclude=[IVH_NONE_COLUMN],
    )
    phe_columns = [name for name in PHE_INTENSITY_COLUMNS if name in df.columns]

    ich_presence = (
        pd.Series([True] * len(df), index=df.index)
        if all_ich_true
        else compute_any_true(df, loc_columns)
    )

    return pd.DataFrame(
        {
            "ID": df.get("ID", ""),
            "all_booleans_false": compute_all_false(df, boolean_columns),
            "ich_presence": ich_presence,
            "ivh_presence": compute_presence_with_none(df, ivh_columns, IVH_NONE_COLUMN),
            "phe_presence": compute_presence_with_none(df, phe_columns, PHE_NONE_COLUMN),
            "ich_localisation": compute_multi_label(df, loc_columns, LOC_PREFIX),
            "ivh_location": compute_multi_label(df, ivh_columns, IVH_PREFIX),
            "phe_intensity": compute_phe_intensity(df, phe_columns),
        }
    )


def normalize_unit(unit: object) -> str:
    if unit is None:
        return ""
    text = str(unit).strip().lower()
    if text in UNIT_TO_CM:
        return text
    if text == "" or text == "nan":
        return ""
    raise ValueError(f"Unsupported unit_measure '{unit}'.")


def convert_measurement_to_cm(value: object, unit: object) -> float:
    unit_key = normalize_unit(unit)
    if unit_key == "":
        raise ValueError("Missing unit_measure for a measurement value.")
    return float(value) * UNIT_TO_CM[unit_key]


def sort_measurements(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    group["_row_order"] = range(len(group))
    if "measurement_index" in group.columns:
        group["_measure_order"] = pd.to_numeric(group["measurement_index"], errors="coerce")
        if group["_measure_order"].notna().any():
            return group.sort_values(["_measure_order", "_row_order"])
    return group.sort_values("_row_order")


def format_size(values_cm: list[float]) -> str | None:
    if not values_cm:
        return None
    parts: list[str] = []
    idx = 0
    while idx < len(values_cm):
        if idx + 1 < len(values_cm):
            parts.append(f"{values_cm[idx]} x {values_cm[idx + 1]} cm")
            idx += 2
        else:
            parts.append(f"{values_cm[idx]} cm")
            idx += 1
    return "/".join(parts)


def build_size_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ID", "size"])

    def build_size(group: pd.DataFrame) -> str | None:
        ordered = sort_measurements(group)
        measurements = pd.to_numeric(ordered.get("measurement"), errors="coerce")
        units = ordered.get("unit_measure")

        values_cm: list[float] = []
        for value, unit in zip(measurements, units):
            if pd.isna(value):
                continue
            values_cm.append(convert_measurement_to_cm(value, unit))

        return format_size(values_cm)

    size_series = df.groupby("ID", sort=False).apply(build_size)
    return size_series.reset_index(name="size")


def write_dataframe(output_path: Path, df: pd.DataFrame) -> None:
    df.to_csv(output_path, index=False)


def main() -> int:
    args = parse_args()

    if not args.booleans and not args.measurements:
        raise SystemExit("Provide --booleans and/or --measurements.")

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_df: pd.DataFrame | None = None

    if args.booleans:
        booleans_df = load_dataframe(Path(args.booleans))
        output_df = build_booleans_output(booleans_df, args.all_ich_true)

    if args.measurements:
        measurements_df = load_dataframe(Path(args.measurements))
        size_df = build_size_dataframe(measurements_df)
        if output_df is None:
            output_df = size_df
        else:
            output_df = output_df.merge(size_df, on="ID", how="outer")

    output_path = output_folder / args.output_name
    write_dataframe(output_path, output_df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
