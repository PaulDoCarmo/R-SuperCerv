import argparse
import glob
import os
from typing import Any

import pandas as pd

from metrics_utils import detect_prompt_id


def read_metric_value(path: str, column: str) -> str:
    if not os.path.isfile(path):
        return "N/A"
    df = pd.read_csv(path)
    if column not in df.columns or df.empty:
        return "N/A"
    value = df.iloc[0][column]
    if pd.isna(value):
        return "N/A"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize metrics into a single CSV.")
    parser.add_argument("--metrics_dir", required=True, help="Metrics folder (e.g. .../metrics/prompt3)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV name (written inside metrics_dir)",
    )
    args = parser.parse_args()

    metrics_dir = args.metrics_dir
    if not os.path.isdir(metrics_dir):
        raise ValueError(f"Metrics directory not found: {metrics_dir}")

    model_dirs = [
        name
        for name in os.listdir(metrics_dir)
        if os.path.isdir(os.path.join(metrics_dir, name))
    ]
    model_dirs = sorted(model_dirs)

    metrics_map: list[tuple[str, str, str]] = [
        (
            "phe_presence_sensitivity",
            os.path.join("phe_presence", "metrics_{model}_prompt*_phe_presence_all_false_ignored.csv"),
            "sensitivity",
        ),
        (
            "ivh_presence_sensitivity",
            os.path.join("ivh_presence", "metrics_{model}_prompt*_ivh_presence_all_false_ignored.csv"),
            "sensitivity",
        ),
        (
            "ivh_presence_specificity",
            os.path.join("ivh_presence", "metrics_{model}_prompt*_ivh_presence_all_false_ignored.csv"),
            "specificity",
        ),
        (
            "ich_location_accuracy",
            os.path.join(
                "ich_location",
                "strict",
                "metrics_{model}_prompt*_ich_location_strict_all_false_ignored.csv",
            ),
            "accuracy",
        ),
        (
            "ivh_location_accuracy",
            os.path.join(
                "ivh_location",
                "strict",
                "metrics_{model}_prompt*_ivh_location_strict_all_false_ignored.csv",
            ),
            "accuracy",
        ),
        (
            "phe_severity_accuracy",
            os.path.join("phe_severity", "metrics_{model}_prompt*_phe_severity_all_false_ignored.csv"),
            "accuracy",
        ),
        (
            "ich_size_accuracy",
            os.path.join(
                "ich_size",
                "strict",
                "metrics_{model}_prompt*_ich_size_strict.csv",
            ),
            "accuracy",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for metric_name, rel_pattern, column in metrics_map:
        row: dict[str, Any] = {"metric": metric_name}
        for model in model_dirs:
            model_dir = os.path.join(metrics_dir, model)
            pattern = rel_pattern.format(model=model)
            search_root = os.path.join(model_dir, os.path.dirname(pattern))
            glob_pattern = os.path.join(search_root, os.path.basename(pattern))
            paths = sorted(glob.glob(glob_pattern))
            row[model] = read_metric_value(paths[0], column) if paths else "N/A"
        rows.append(row)

    if args.output:
        output_name = args.output
    else:
        prompt_id = detect_prompt_id(metrics_dir)
        output_name = f"summary_metrics_prompt{prompt_id}.csv"

    output_path = os.path.join(metrics_dir, output_name)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
