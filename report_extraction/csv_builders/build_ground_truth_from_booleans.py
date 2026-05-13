#!/usr/bin/env python3
"""Build a CSV with ground-truth flags from checklist booleans."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CSV with ground-truth flags from checklist booleans."
    )
    parser.add_argument("input_csv", help="Path to checklist_booleans.csv")
    parser.add_argument("output_folder", help="Folder to write the CSV")
    parser.add_argument(
        "--output-name",
        default="ground_truth_from_booleans.csv",
        help="CSV filename to create in output_folder (default: ground_truth_from_booleans.csv)",
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


def build_output_dataframe(
    df: pd.DataFrame,
    boolean_columns: Iterable[str],
    loc_columns: list[str],
    ivh_columns: list[str],
    phe_columns: Iterable[str],
    all_ich_true: bool,
) -> pd.DataFrame:
    ich_presence = (
        pd.Series([True] * len(df), index=df.index)
        if all_ich_true
        else compute_any_true(df, loc_columns)
    )
    output_df = pd.DataFrame(
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
    return output_df


def write_dataframe(output_path: Path, df: pd.DataFrame) -> None:
    df.to_csv(output_path, index=False)


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(input_csv)

    boolean_columns = [name for name in df.columns if name != "ID"]
    loc_columns = find_columns_by_prefix(df.columns, LOC_PREFIX)
    ivh_columns = find_columns_by_prefix(
        df.columns,
        IVH_PREFIX,
        exclude=[IVH_NONE_COLUMN],
    )

    phe_columns = [name for name in PHE_INTENSITY_COLUMNS if name in df.columns]

    output_df = build_output_dataframe(
        df,
        boolean_columns,
        loc_columns,
        ivh_columns,
        phe_columns,
        args.all_ich_true,
    )

    output_path = output_folder / args.output_name
    write_dataframe(output_path, output_df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
