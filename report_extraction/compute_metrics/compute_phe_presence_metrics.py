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
    output_dir = os.path.join(output_root, f"prompt{prompt_id}", model_name, "phe_presence")
    os.makedirs(output_dir, exist_ok=True)

    csv_name = f"metrics_{model_name}_prompt{prompt_id}_phe_presence{suffix}.csv"
    txt_name = f"errors_{model_name}_prompt{prompt_id}_phe_presence{suffix}.txt"
    return os.path.join(output_dir, csv_name), os.path.join(output_dir, txt_name)


def parse_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in BOOL_TRUE


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics_for_file(
    gt_df: pd.DataFrame,
    formatted_results_path: str,
    output_root: str,
    ignore_all_false: bool,
) -> None:
    res_df = pd.read_csv(formatted_results_path)

    if "ID" not in gt_df.columns or "phe_presence" not in gt_df.columns:
        raise ValueError("Ground truth must contain columns 'ID' and 'phe_presence'.")
    if ignore_all_false and "all_booleans_false" not in gt_df.columns:
        raise ValueError("Ground truth must contain column 'all_booleans_false'.")
    if "ID" not in res_df.columns or "type" not in res_df.columns:
        raise ValueError("Formatted results must contain columns 'ID' and 'type'.")

    phe_mask = res_df["type"].astype(str).str.strip().str.upper() == "PHE"
    phe_ids = set(res_df.loc[phe_mask, "ID"].dropna().astype(str))

    ignored_ids = []
    eval_df = gt_df
    if ignore_all_false:
        ignore_mask = gt_df["all_booleans_false"].apply(parse_bool)
        ignored_ids = gt_df.loc[ignore_mask, "ID"].dropna().astype(str).tolist()
        eval_df = gt_df.loc[~ignore_mask].copy()

    tp = fp = fn = tn = 0
    errors = []

    for _, row in eval_df.iterrows():
        sample_id = str(row["ID"])
        gt_phe = parse_bool(row["phe_presence"])
        pred_phe = sample_id in phe_ids

        if gt_phe and pred_phe:
            tp += 1
        elif gt_phe and not pred_phe:
            fn += 1
            errors.append((sample_id, "true", "false"))
        elif not gt_phe and pred_phe:
            fp += 1
            errors.append((sample_id, "false", "true"))
        else:
            tn += 1

    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    f1_score = safe_div(2 * precision * sensitivity, precision + sensitivity)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    metrics_df = pd.DataFrame(
        [
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1_score": f1_score,
                "accuracy": accuracy,
            }
        ]
    )

    suffix = "_all_false_ignored" if ignore_all_false else ""
    output_csv, output_txt = build_output_paths(
        formatted_results_path, output_root, suffix
    )
    metrics_df.to_csv(output_csv, index=False)

    gt_ids = set(eval_df["ID"].dropna().astype(str))
    extra_pred_ids = sorted(phe_ids - gt_ids)

    with open(output_txt, "w", encoding="utf-8") as handle:
        handle.write("PHE presence errors\n")
        handle.write(f"tp={tp} fp={fp} fn={fn} tn={tn}\n")
        handle.write("\n")
        handle.write("ID\tground_truth\tprediction\n")
        for sample_id, gt_value, pred_value in errors:
            handle.write(f"{sample_id}\t{gt_value}\t{pred_value}\n")
        if extra_pred_ids:
            handle.write("\n")
            handle.write("Predicted PHE IDs missing from ground truth\n")
            for sample_id in extra_pred_ids:
                handle.write(f"{sample_id}\n")
        if ignored_ids:
            handle.write("\n")
            handle.write("Ignored IDs (all_booleans_false == true)\n")
            for sample_id in ignored_ids:
                handle.write(f"{sample_id}\n")

    print(f"Saved metrics to {output_csv}")
    print(f"Saved errors to {output_txt}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute PHE presence metrics.")
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
