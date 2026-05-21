import argparse

import pandas as pd

from metrics_utils import build_output_paths, list_csv_files, parse_bool


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics_for_file(
    gt_df: pd.DataFrame,
    formatted_results_path: str,
    output_root: str,
    ignore_all_false: bool,
) -> None:
    res_df = pd.read_csv(formatted_results_path)

    if "ID" not in gt_df.columns or "ivh_presence" not in gt_df.columns:
        raise ValueError("Ground truth must contain columns 'ID' and 'ivh_presence'.")
    if ignore_all_false and "all_booleans_false" not in gt_df.columns:
        raise ValueError("Ground truth must contain column 'all_booleans_false'.")
    if "ID" not in res_df.columns or "type" not in res_df.columns:
        raise ValueError("Formatted results must contain columns 'ID' and 'type'.")

    ivh_mask = res_df["type"].astype(str).str.upper() == "IVH"
    ivh_ids = set(res_df.loc[ivh_mask, "ID"].dropna().astype(str))

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
        gt_ivh = parse_bool(row["ivh_presence"])
        pred_ivh = sample_id in ivh_ids

        if gt_ivh and pred_ivh:
            tp += 1
        elif gt_ivh and not pred_ivh:
            fn += 1
            errors.append((sample_id, "true", "false"))
        elif not gt_ivh and pred_ivh:
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
        formatted_results_path, output_root, "ivh_presence", suffix=suffix
    )
    metrics_df.to_csv(output_csv, index=False)

    gt_ids = set(eval_df["ID"].dropna().astype(str))
    extra_pred_ids = sorted(ivh_ids - gt_ids)

    with open(output_txt, "w", encoding="utf-8") as handle:
        handle.write("IVH presence errors\n")
        handle.write(f"tp={tp} fp={fp} fn={fn} tn={tn}\n")
        handle.write("\n")
        handle.write("ID\tground_truth\tprediction\n")
        for sample_id, gt_value, pred_value in errors:
            handle.write(f"{sample_id}\t{gt_value}\t{pred_value}\n")
        if extra_pred_ids:
            handle.write("\n")
            handle.write("Predicted IVH IDs missing from ground truth\n")
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
    parser = argparse.ArgumentParser(description="Compute IVH presence metrics.")
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
