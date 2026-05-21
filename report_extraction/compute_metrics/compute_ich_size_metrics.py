import argparse
import re

import pandas as pd

from metrics_utils import build_output_paths, list_csv_files, parse_bool

DEFAULT_TOLERANCE_MM = 5.0


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def format_dims(dims: tuple[float, ...]) -> str:
    if len(dims) == 1:
        return f"{dims[0]:.1f}mm"
    return f"{dims[0]:.1f}x{dims[1]:.1f}mm"


def parse_segment(segment: str) -> list[tuple[float, ...]]:
    text = segment.strip().lower()
    if not text:
        return []

    unit = "cm"
    if "mm" in text:
        unit = "mm"
    elif "cm" in text:
        unit = "cm"

    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return []

    factor = 10.0 if unit == "cm" else 1.0

    if "x" in text or "*" in text:
        if len(numbers) >= 2:
            a = numbers[0] * factor
            b = numbers[1] * factor
            dims = tuple(sorted((a, b)))
            return [dims]
        return []

    if len(numbers) == 1:
        return [(numbers[0] * factor,)]

    return []


def parse_size_list(value: object) -> list[tuple[float, ...]]:
    text = normalize_text(value)
    if not text:
        return []
    parts = [p for p in text.split("/") if p.strip()]
    sizes: list[tuple[float, ...]] = []
    for part in parts:
        sizes.extend(parse_segment(part))
    return sizes


def match_dims(gt: tuple[float, ...], pred: tuple[float, ...], tolerance_mm: float) -> bool:
    if len(gt) != len(pred):
        return False
    if len(gt) == 1:
        return abs(gt[0] - pred[0]) <= tolerance_mm
    return abs(gt[0] - pred[0]) <= tolerance_mm and abs(gt[1] - pred[1]) <= tolerance_mm


def greedy_match(
    gt_sizes: list[tuple[float, ...]],
    pred_sizes: list[tuple[float, ...]],
    tolerance_mm: float,
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
    remaining_pred = pred_sizes[:]
    unmatched_gt: list[tuple[float, ...]] = []

    for gt in gt_sizes:
        matched = False
        for idx, pred in enumerate(remaining_pred):
            if match_dims(gt, pred, tolerance_mm):
                matched = True
                remaining_pred.pop(idx)
                break
        if not matched:
            unmatched_gt.append(gt)

    return unmatched_gt, remaining_pred


def compute_strict_metrics(
    eval_df: pd.DataFrame,
    res_df: pd.DataFrame,
    tolerance_mm: float,
) -> tuple[int, int, int, list[tuple[str, str]], list[tuple[str, str, str]]]:
    total = 0
    correct = 0
    errors_no_detection: list[tuple[str, str]] = []
    errors_mismatch: list[tuple[str, str, str]] = []

    for _, row in eval_df.iterrows():
        sample_id = str(row["ID"])
        gt_sizes = parse_size_list(row["size"])
        total += 1

        ich_rows = res_df[
            (res_df["ID"].astype(str) == sample_id)
            & (res_df["type"].astype(str).str.strip().str.upper() == "ICH")
        ]
        if ich_rows.empty:
            errors_no_detection.append((sample_id, "/".join(format_dims(s) for s in gt_sizes)))
            continue

        pred_sizes: list[tuple[float, ...]] = []
        for value in ich_rows["size"].tolist():
            pred_sizes.extend(parse_size_list(value))

        unmatched_gt, unmatched_pred = greedy_match(gt_sizes, pred_sizes, tolerance_mm)
        if not unmatched_gt and not unmatched_pred:
            correct += 1
        else:
            errors_mismatch.append(
                (
                    sample_id,
                    "/".join(format_dims(s) for s in gt_sizes),
                    "/".join(format_dims(s) for s in pred_sizes),
                )
            )

    return total, correct, len(errors_no_detection), errors_no_detection, errors_mismatch


def compute_flexible_metrics(
    eval_df: pd.DataFrame,
    res_df: pd.DataFrame,
    tolerance_mm: float,
) -> tuple[
    int,
    int,
    float,
    float,
    list[tuple[str, str]],
    list[tuple[str, str, str, str, str, str]],
]:
    total = 0
    no_detection = 0
    sum_gt_coverage = 0.0
    sum_pred_precision = 0.0
    errors_no_detection: list[tuple[str, str]] = []
    errors_mismatch: list[tuple[str, str, str, str, str, str]] = []

    for _, row in eval_df.iterrows():
        sample_id = str(row["ID"])
        gt_sizes = parse_size_list(row["size"])
        total += 1

        ich_rows = res_df[
            (res_df["ID"].astype(str) == sample_id)
            & (res_df["type"].astype(str).str.strip().str.upper() == "ICH")
        ]
        if ich_rows.empty:
            no_detection += 1
            errors_no_detection.append((sample_id, "/".join(format_dims(s) for s in gt_sizes)))
            continue

        pred_sizes: list[tuple[float, ...]] = []
        for value in ich_rows["size"].tolist():
            pred_sizes.extend(parse_size_list(value))

        unmatched_gt, unmatched_pred = greedy_match(gt_sizes, pred_sizes, tolerance_mm)
        matched_gt_count = len(gt_sizes) - len(unmatched_gt)
        matched_pred_count = len(pred_sizes) - len(unmatched_pred)

        gt_coverage = matched_gt_count / len(gt_sizes) if gt_sizes else 0.0
        pred_precision = matched_pred_count / len(pred_sizes) if pred_sizes else 0.0

        sum_gt_coverage += gt_coverage
        sum_pred_precision += pred_precision

        if unmatched_gt or unmatched_pred:
            errors_mismatch.append(
                (
                    sample_id,
                    "/".join(format_dims(s) for s in gt_sizes),
                    "/".join(format_dims(s) for s in pred_sizes),
                    "/".join(format_dims(s) for s in unmatched_gt),
                    "/".join(format_dims(s) for s in unmatched_pred),
                    f"gt_coverage={gt_coverage:.3f} pred_precision={pred_precision:.3f}",
                )
            )

    avg_gt_coverage = sum_gt_coverage / total if total else 0.0
    avg_pred_precision = sum_pred_precision / total if total else 0.0

    return (
        total,
        no_detection,
        avg_gt_coverage,
        avg_pred_precision,
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
    tolerance_mm: float,
) -> None:
    res_df = pd.read_csv(formatted_results_path)

    required_gt = {"ID", "size"}
    if not required_gt.issubset(gt_df.columns):
        raise ValueError("Ground truth must contain columns 'ID' and 'size'.")
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

    eval_df = eval_df[eval_df["size"].apply(normalize_text) != ""].copy()

    suffix = "_all_false_ignored" if ignore_all_false else ""

    if run_strict:
        total, correct, no_detection, errors_no_detection, errors_mismatch = compute_strict_metrics(
            eval_df=eval_df,
            res_df=res_df,
            tolerance_mm=tolerance_mm,
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
            "ich_size",
            mode="strict",
            suffix=suffix,
        )
        metrics_df.to_csv(output_csv, index=False)

        with open(output_txt, "w", encoding="utf-8") as handle:
            handle.write("ICH size errors (strict)\n")
            handle.write(
                f"total={total} correct={correct} no_detection={no_detection} accuracy={accuracy}\n"
            )
            handle.write("\n")
            handle.write("Errors - no ICH detected\n")
            handle.write("ID\tground_truth_sizes\n")
            for sample_id, gt_sizes in errors_no_detection:
                handle.write(f"{sample_id}\t{gt_sizes}\n")
            handle.write("\n")
            handle.write("Errors - size mismatch\n")
            handle.write("ID\tground_truth_sizes\tpredicted_sizes\n")
            for sample_id, gt_sizes, pred_sizes in errors_mismatch:
                handle.write(f"{sample_id}\t{gt_sizes}\t{pred_sizes}\n")
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
            errors_no_detection,
            errors_mismatch,
        ) = compute_flexible_metrics(
            eval_df=eval_df,
            res_df=res_df,
            tolerance_mm=tolerance_mm,
        )

        metrics_df = pd.DataFrame(
            [
                {
                    "total": total,
                    "no_detection": no_detection,
                    "avg_gt_coverage": avg_gt_coverage,
                    "avg_pred_precision": avg_pred_precision,
                }
            ]
        )
        output_csv, output_txt = build_output_paths(
            formatted_results_path,
            output_root,
            "ich_size",
            mode="flexible",
            suffix=suffix,
        )
        metrics_df.to_csv(output_csv, index=False)

        with open(output_txt, "w", encoding="utf-8") as handle:
            handle.write("ICH size errors (flexible)\n")
            handle.write(
                "total={total} no_detection={no_detection} avg_gt_coverage={avg_gt_coverage} "
                "avg_pred_precision={avg_pred_precision}\n".format(
                    total=total,
                    no_detection=no_detection,
                    avg_gt_coverage=avg_gt_coverage,
                    avg_pred_precision=avg_pred_precision,
                )
            )
            handle.write("\n")
            handle.write("Errors - no ICH detected\n")
            handle.write("ID\tground_truth_sizes\n")
            for sample_id, gt_sizes in errors_no_detection:
                handle.write(f"{sample_id}\t{gt_sizes}\n")
            handle.write("\n")
            handle.write("Errors - size mismatch\n")
            handle.write(
                "ID\tground_truth_sizes\tpredicted_sizes\tmissing_gt\textra_pred\tper_id_scores\n"
            )
            for (
                sample_id,
                gt_sizes,
                pred_sizes,
                missing_gt,
                extra_pred,
                per_id_scores,
            ) in errors_mismatch:
                handle.write(
                    f"{sample_id}\t{gt_sizes}\t{pred_sizes}\t{missing_gt}\t{extra_pred}\t{per_id_scores}\n"
                )
            if ignored_ids:
                handle.write("\n")
                handle.write("Ignored IDs (all_booleans_false == true)\n")
                for sample_id in ignored_ids:
                    handle.write(f"{sample_id}\n")

        print(f"Saved metrics to {output_csv}")
        print(f"Saved errors to {output_txt}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute ICH size metrics.")
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
        "--tolerance_mm",
        type=float,
        default=DEFAULT_TOLERANCE_MM,
        help="Tolerance in mm (default: 5)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Compute strict ICH size metrics",
    )
    parser.add_argument(
        "--flexible",
        action="store_true",
        help="Compute flexible ICH size metrics",
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
            tolerance_mm=args.tolerance_mm,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
