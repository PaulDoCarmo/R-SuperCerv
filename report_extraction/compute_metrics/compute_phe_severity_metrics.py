import argparse
import os
import re

import pandas as pd


BOOL_TRUE = {"true", "1", "yes", "y", "t"}


def detect_prompt_id(path: str) -> str:
    match = re.search(r"/prompt(\d+)/", path)
    if match:
        return match.group(1)
    match = re.search(r"_prompt(\d+)", os.path.basename(path))
    if match:
        return match.group(1)
    return "U"


def detect_model_name(path: str) -> str:
    base = os.path.basename(path)
    match = re.search(r"results_(.+?)_prompt\d+", base)
    if match:
        return match.group(1)
    match = re.search(r"results_(.+)", base)
    if match:
        name = match.group(1)
        name = re.sub(r"_formated$", "", name)
        name = re.sub(r"_formatted$", "", name)
        return name
    return "U"


def list_csv_files(input_path: str) -> list[str]:
    if os.path.isdir(input_path):
        entries = [
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if name.lower().endswith(".csv")
        ]
        return sorted(entries)
    return [input_path]


def build_output_paths(
    formatted_results_path: str, output_root: str, suffix: str = ""
) -> tuple[str, str]:
    prompt_id = detect_prompt_id(formatted_results_path)
    model_name = detect_model_name(formatted_results_path)
    output_dir = os.path.join(output_root, f"prompt{prompt_id}", model_name, "phe_severity")
    os.makedirs(output_dir, exist_ok=True)

    csv_name = f"metrics_{model_name}_prompt{prompt_id}_phe_severity{suffix}.csv"
    txt_name = f"errors_{model_name}_prompt{prompt_id}_phe_severity{suffix}.txt"
    return os.path.join(output_dir, csv_name), os.path.join(output_dir, txt_name)


def parse_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in BOOL_TRUE


def normalize_intensity(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def extract_predicted_intensities(values: list[object]) -> tuple[list[str], bool]:
    intensities: list[str] = []
    has_slash = False
    for value in values:
        text = normalize_intensity(value)
        if not text:
            continue
        if "/" in text:
            has_slash = True
        parts = [part.strip() for part in text.split("/") if part.strip()]
        intensities.extend(parts)
    return intensities, has_slash


def compute_metrics_for_file(
    gt_df: pd.DataFrame,
    formatted_results_path: str,
    output_root: str,
    ignore_all_false: bool,
) -> None:
    res_df = pd.read_csv(formatted_results_path)

    required_gt = {"ID", "phe_presence", "phe_intensity"}
    if not required_gt.issubset(gt_df.columns):
        raise ValueError("Ground truth must contain columns 'ID', 'phe_presence', and 'phe_intensity'.")
    if ignore_all_false and "all_booleans_false" not in gt_df.columns:
        raise ValueError("Ground truth must contain column 'all_booleans_false'.")
    if "ID" not in res_df.columns or "type" not in res_df.columns or "size" not in res_df.columns:
        raise ValueError("Formatted results must contain columns 'ID', 'type', and 'size'.")

    ignored_ids = []
    eval_df = gt_df
    if ignore_all_false:
        ignore_mask = gt_df["all_booleans_false"].apply(parse_bool)
        ignored_ids = gt_df.loc[ignore_mask, "ID"].dropna().astype(str).tolist()
        eval_df = gt_df.loc[~ignore_mask].copy()

    eval_df = eval_df[eval_df["phe_intensity"].apply(normalize_intensity) != ""].copy()

    total = 0
    correct = 0
    errors_no_detection: list[tuple[str, str]] = []
    errors_wrong_value: list[tuple[str, str, str]] = []
    warnings_slash: list[tuple[str, str, str]] = []

    for _, row in eval_df.iterrows():
        sample_id = str(row["ID"])
        gt_intensity = normalize_intensity(row["phe_intensity"])
        total += 1

        phe_rows = res_df[
            (res_df["ID"].astype(str) == sample_id)
            & (res_df["type"].astype(str).str.strip().str.upper() == "PHE")
        ]

        if phe_rows.empty:
            errors_no_detection.append((sample_id, gt_intensity))
            continue

        predicted_values, has_slash = extract_predicted_intensities(phe_rows["size"].tolist())
        if has_slash:
            joined_size = "/".join([normalize_intensity(v) for v in phe_rows["size"].tolist()])
            warnings_slash.append((sample_id, gt_intensity, joined_size))

        if gt_intensity and gt_intensity in predicted_values:
            correct += 1
        else:
            joined_pred = "/".join(predicted_values) if predicted_values else ""
            errors_wrong_value.append((sample_id, gt_intensity, joined_pred))

    accuracy = correct / total if total else 0.0

    metrics_df = pd.DataFrame(
        [
            {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
            }
        ]
    )

    suffix = "_all_false_ignored" if ignore_all_false else ""
    output_csv, output_txt = build_output_paths(formatted_results_path, output_root, suffix)
    metrics_df.to_csv(output_csv, index=False)

    with open(output_txt, "w", encoding="utf-8") as handle:
        handle.write("PHE severity errors\n")
        handle.write(f"total={total} correct={correct} accuracy={accuracy}\n")
        handle.write("\n")
        handle.write("Errors - no PHE detected\n")
        handle.write("ID\tground_truth_phe_intensity\n")
        for sample_id, gt_intensity in errors_no_detection:
            handle.write(f"{sample_id}\t{gt_intensity}\n")
        handle.write("\n")
        handle.write("Errors - wrong PHE intensity\n")
        handle.write("ID\tground_truth_phe_intensity\tpredicted_intensity\n")
        for sample_id, gt_intensity, pred_intensity in errors_wrong_value:
            handle.write(f"{sample_id}\t{gt_intensity}\t{pred_intensity}\n")
        if warnings_slash:
            handle.write("\n")
            handle.write("Warnings - multiple predicted intensities ('/' in size)\n")
            handle.write("ID\tground_truth_phe_intensity\tpredicted_size\n")
            for sample_id, gt_intensity, pred_size in warnings_slash:
                handle.write(f"{sample_id}\t{gt_intensity}\t{pred_size}\n")
        if ignored_ids:
            handle.write("\n")
            handle.write("Ignored IDs (all_booleans_false == true)\n")
            for sample_id in ignored_ids:
                handle.write(f"{sample_id}\n")

    print(f"Saved metrics to {output_csv}")
    print(f"Saved errors to {output_txt}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute PHE severity metrics.")
    parser.add_argument("--ground_truth", required=True, help="Ground truth CSV file")
    parser.add_argument("--formatted_results", required=True, help="Formatted results CSV file")
    parser.add_argument(
        "--output_root",
        default="/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/metrics",
        help="Root output directory",
    )
    parser.add_argument(
        "--ignore_all_false",
        action="store_true",
        help="Ignore rows with all_booleans_false == true in ground truth",
    )
    args = parser.parse_args()

    gt_df = pd.read_csv(args.ground_truth)
    formatted_files = list_csv_files(args.formatted_results)
    if not formatted_files:
        raise ValueError(f"No CSV files found in {args.formatted_results}")

    for formatted_file in formatted_files:
        compute_metrics_for_file(
            gt_df=gt_df,
            formatted_results_path=formatted_file,
            output_root=args.output_root,
            ignore_all_false=args.ignore_all_false,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
