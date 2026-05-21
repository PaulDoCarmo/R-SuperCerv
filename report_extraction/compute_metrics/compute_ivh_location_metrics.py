import argparse

import pandas as pd

from metrics_utils import build_output_paths, list_csv_files, parse_bool


def normalize_location(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def extract_locations(values: list[object]) -> list[str]:
    found: list[str] = []
    for value in values:
        text = normalize_location(value)
        if not text:
            continue
        parts = []
        for slash_part in text.split("/"):
            parts.extend([p for p in slash_part.split("-") if p])
        for part in parts:
            cleaned = part.strip()
            if cleaned == "u":
                continue
            if cleaned and cleaned not in found:
                found.append(cleaned)
    return found


def compute_strict_metrics(
    eval_df: pd.DataFrame,
    res_df: pd.DataFrame,
) -> tuple[int, int, int, list[tuple[str, str]], list[tuple[str, str, str]]]:
    total = 0
    correct = 0
    errors_no_detection: list[tuple[str, str]] = []
    errors_mismatch: list[tuple[str, str, str]] = []

    for _, row in eval_df.iterrows():
        sample_id = str(row["ID"])
        gt_locations = extract_locations([row["ivh_location"]])
        total += 1

        ivh_rows = res_df[
            (res_df["ID"].astype(str) == sample_id)
            & (res_df["type"].astype(str).str.strip().str.upper() == "IVH")
        ]
        if ivh_rows.empty:
            errors_no_detection.append((sample_id, "/".join(gt_locations)))
            continue

        pred_locations = extract_locations(ivh_rows["structure"].tolist())
        if set(gt_locations) == set(pred_locations):
            correct += 1
        else:
            errors_mismatch.append(
                (sample_id, "/".join(gt_locations), "/".join(pred_locations))
            )

    return total, correct, len(errors_no_detection), errors_no_detection, errors_mismatch


def compute_flexible_metrics(
    eval_df: pd.DataFrame,
    res_df: pd.DataFrame,
) -> tuple[
    int,
    int,
    float,
    float,
    float,
    list[tuple[str, str]],
    list[tuple[str, str, str, str, str, str]],
]:
    total = 0
    no_detection = 0
    sum_gt_coverage = 0.0
    sum_pred_precision = 0.0
    sum_jaccard = 0.0
    errors_no_detection: list[tuple[str, str]] = []
    errors_mismatch: list[tuple[str, str, str, str, str, str]] = []

    for _, row in eval_df.iterrows():
        sample_id = str(row["ID"])
        gt_locations = extract_locations([row["ivh_location"]])
        total += 1

        ivh_rows = res_df[
            (res_df["ID"].astype(str) == sample_id)
            & (res_df["type"].astype(str).str.strip().str.upper() == "IVH")
        ]
        if ivh_rows.empty:
            no_detection += 1
            errors_no_detection.append((sample_id, "/".join(gt_locations)))
            continue

        pred_locations = extract_locations(ivh_rows["structure"].tolist())
        gt_set = set(gt_locations)
        pred_set = set(pred_locations)
        intersection = gt_set & pred_set
        union = gt_set | pred_set

        gt_coverage = len(intersection) / len(gt_set) if gt_set else 0.0
        pred_precision = len(intersection) / len(pred_set) if pred_set else 0.0
        jaccard = len(intersection) / len(union) if union else 0.0

        sum_gt_coverage += gt_coverage
        sum_pred_precision += pred_precision
        sum_jaccard += jaccard

        if intersection != gt_set or intersection != pred_set:
            errors_mismatch.append(
                (
                    sample_id,
                    "/".join(gt_locations),
                    "/".join(pred_locations),
                    "/".join(sorted(gt_set - pred_set)),
                    "/".join(sorted(pred_set - gt_set)),
                    f"gt_coverage={gt_coverage:.3f} pred_precision={pred_precision:.3f} jaccard={jaccard:.3f}",
                )
            )

    avg_gt_coverage = sum_gt_coverage / total if total else 0.0
    avg_pred_precision = sum_pred_precision / total if total else 0.0
    avg_jaccard = sum_jaccard / total if total else 0.0

    return (
        total,
        no_detection,
        avg_gt_coverage,
        avg_pred_precision,
        avg_jaccard,
        errors_no_detection,
        errors_mismatch,
    )


def compute_metrics_for_file(
    gt_df: pd.DataFrame,
    formatted_results_path: str,
    output_root: str,
    ignore_all_false: bool,
    run_strict: bool,
    run_flexible: bool,
) -> None:
    res_df = pd.read_csv(formatted_results_path)

    required_gt = {"ID", "ivh_presence", "ivh_location"}
    if not required_gt.issubset(gt_df.columns):
        raise ValueError("Ground truth must contain columns 'ID', 'ivh_presence', and 'ivh_location'.")
    if ignore_all_false and "all_booleans_false" not in gt_df.columns:
        raise ValueError("Ground truth must contain column 'all_booleans_false'.")
    if "ID" not in res_df.columns or "type" not in res_df.columns or "structure" not in res_df.columns:
        raise ValueError("Formatted results must contain columns 'ID', 'type', and 'structure'.")

    ignored_ids = []
    eval_df = gt_df
    if ignore_all_false:
        ignore_mask = gt_df["all_booleans_false"].apply(parse_bool)
        ignored_ids = gt_df.loc[ignore_mask, "ID"].dropna().astype(str).tolist()
        eval_df = gt_df.loc[~ignore_mask].copy()

    eval_df = eval_df[eval_df["ivh_location"].apply(normalize_location) != ""].copy()

    suffix = "_all_false_ignored" if ignore_all_false else ""

    if run_strict:
        total, correct, no_detection, errors_no_detection, errors_mismatch = compute_strict_metrics(
            eval_df=eval_df,
            res_df=res_df,
        )
        accuracy = correct / total if total else 0.0

        metrics_df = pd.DataFrame(
            [
                {
                    "total": total,
                    "correct": correct,
                    "no_detection": no_detection,
                    "accuracy": accuracy,
                }
            ]
        )
        output_csv, output_txt = build_output_paths(
            formatted_results_path,
            output_root,
            "ivh_location",
            mode="strict",
            suffix=suffix,
        )
        metrics_df.to_csv(output_csv, index=False)

        with open(output_txt, "w", encoding="utf-8") as handle:
            handle.write("IVH location errors (strict)\n")
            handle.write(
                f"total={total} correct={correct} no_detection={no_detection} accuracy={accuracy}\n"
            )
            handle.write("\n")
            handle.write("Errors - no IVH detected\n")
            handle.write("ID\tground_truth_locations\n")
            for sample_id, gt_locations in errors_no_detection:
                handle.write(f"{sample_id}\t{gt_locations}\n")
            handle.write("\n")
            handle.write("Errors - location mismatch\n")
            handle.write("ID\tground_truth_locations\tpredicted_locations\n")
            for sample_id, gt_locations, pred_locations in errors_mismatch:
                handle.write(f"{sample_id}\t{gt_locations}\t{pred_locations}\n")
            if ignored_ids:
                handle.write("\n")
                handle.write("Ignored IDs (all_booleans_false == true)\n")
                for sample_id in ignored_ids:
                    handle.write(f"{sample_id}\n")

        print(f"Saved metrics to {output_csv}")
        print(f"Saved errors to {output_txt}")

    if run_flexible:
        (
            total,
            no_detection,
            avg_gt_coverage,
            avg_pred_precision,
            avg_jaccard,
            errors_no_detection,
            errors_mismatch,
        ) = compute_flexible_metrics(
            eval_df=eval_df,
            res_df=res_df,
        )

        metrics_df = pd.DataFrame(
            [
                {
                    "total": total,
                    "no_detection": no_detection,
                    "avg_gt_coverage": avg_gt_coverage,
                    "avg_pred_precision": avg_pred_precision,
                    "avg_jaccard": avg_jaccard,
                }
            ]
        )
        output_csv, output_txt = build_output_paths(
            formatted_results_path,
            output_root,
            "ivh_location",
            mode="flexible",
            suffix=suffix,
        )
        metrics_df.to_csv(output_csv, index=False)

        with open(output_txt, "w", encoding="utf-8") as handle:
            handle.write("IVH location errors (flexible)\n")
            handle.write(
                "total={total} no_detection={no_detection} avg_gt_coverage={avg_gt_coverage} "
                "avg_pred_precision={avg_pred_precision} avg_jaccard={avg_jaccard}\n".format(
                    total=total,
                    no_detection=no_detection,
                    avg_gt_coverage=avg_gt_coverage,
                    avg_pred_precision=avg_pred_precision,
                    avg_jaccard=avg_jaccard,
                )
            )
            handle.write("\n")
            handle.write("Errors - no IVH detected\n")
            handle.write("ID\tground_truth_locations\n")
            for sample_id, gt_locations in errors_no_detection:
                handle.write(f"{sample_id}\t{gt_locations}\n")
            handle.write("\n")
            handle.write("Errors - location mismatch\n")
            handle.write(
                "ID\tground_truth_locations\tpredicted_locations\tmissing_gt\textra_pred\tper_id_scores\n"
            )
            for (
                sample_id,
                gt_locations,
                pred_locations,
                missing_gt,
                extra_pred,
                per_id_scores,
            ) in errors_mismatch:
                handle.write(
                    f"{sample_id}\t{gt_locations}\t{pred_locations}\t{missing_gt}\t{extra_pred}\t{per_id_scores}\n"
                )
            if ignored_ids:
                handle.write("\n")
                handle.write("Ignored IDs (all_booleans_false == true)\n")
                for sample_id in ignored_ids:
                    handle.write(f"{sample_id}\n")

        print(f"Saved metrics to {output_csv}")
        print(f"Saved errors to {output_txt}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute IVH location metrics.")
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Compute strict IVH location metrics",
    )
    parser.add_argument(
        "--flexible",
        action="store_true",
        help="Compute flexible IVH location metrics",
    )
    parser.add_argument(
        "--all_modes",
        action="store_true",
        help="Compute all modes (strict and flexible)",
    )
    args = parser.parse_args()

    run_strict = args.strict
    run_flexible = args.flexible
    if args.all_modes or (not args.strict and not args.flexible):
        run_strict = True
        run_flexible = True

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
            run_strict=run_strict,
            run_flexible=run_flexible,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
