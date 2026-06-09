import argparse
import glob
import os
from typing import Any

import pandas as pd


DEFAULT_METRICS_ROOT = "/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/metrics"


def load_prompt_summary(path: str, model: str) -> tuple[list[str], dict[str, str]]:
    df = pd.read_csv(path)
    if "metric" not in df.columns:
        return [], {}

    metrics = df["metric"].tolist()
    if model not in df.columns:
        return metrics, {m: "N/A" for m in metrics}

    values = df[model].tolist()
    mapping = {}
    for metric, value in zip(metrics, values):
        if pd.isna(value):
            mapping[metric] = "N/A"
        else:
            mapping[metric] = str(value)
    return metrics, mapping


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize metrics across prompts for a single model."
    )
    parser.add_argument("--model", help="Model name (folder name in metrics/prompt*)")
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Generate summaries for all models present in every prompt summary",
    )
    parser.add_argument(
        "--metrics_root",
        default=DEFAULT_METRICS_ROOT,
        help="Root metrics folder (default: /.../report_extraction/metrics)",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_METRICS_ROOT,
        help="Output folder (default: metrics root)",
    )
    args = parser.parse_args()

    metrics_root = args.metrics_root
    if not os.path.isdir(metrics_root):
        raise ValueError(f"Metrics root not found: {metrics_root}")

    summary_paths = sorted(
        glob.glob(os.path.join(metrics_root, "prompt*", "summary_metrics_prompt*.csv"))
    )
    if not summary_paths:
        raise ValueError(f"No summary_metrics_prompt*.csv found in {metrics_root}")

    if not args.all_models and not args.model:
        raise ValueError("Provide --model or use --all_models.")

    model_sets = []
    for summary_path in summary_paths:
        df = pd.read_csv(summary_path)
        columns = [c for c in df.columns if c != "metric"]
        model_sets.append(set(columns))
    common_models = sorted(set.intersection(*model_sets)) if model_sets else []

    models_to_process = [args.model] if not args.all_models else common_models
    if args.all_models and not common_models:
        raise ValueError("No common models across summary files.")

    prompt_columns: list[str] = []
    prompt_values: dict[str, dict[str, str]] = {}
    metric_order: list[str] = []
    metric_seen = set()

    os.makedirs(args.output_dir, exist_ok=True)

    for model in models_to_process:
        prompt_columns = []
        prompt_values = {}
        metric_order = []
        metric_seen = set()

        for summary_path in summary_paths:
            base = os.path.basename(summary_path)
            prompt_id = "U"
            match = None
            for token in base.split("_"):
                if token.startswith("prompt"):
                    match = token.replace("prompt", "")
                    break
            if match:
                prompt_id = match.replace(".csv", "")
            prompt_name = f"prompt{prompt_id}"
            prompt_columns.append(prompt_name)

            metrics, mapping = load_prompt_summary(summary_path, model)
            for metric in metrics:
                if metric not in metric_seen:
                    metric_seen.add(metric)
                    metric_order.append(metric)
            prompt_values[prompt_name] = mapping

        rows: list[dict[str, Any]] = []
        for metric in metric_order:
            row: dict[str, Any] = {"metric": metric}
            for prompt_name in prompt_columns:
                row[prompt_name] = prompt_values.get(prompt_name, {}).get(metric, "N/A")
            rows.append(row)

        output_path = os.path.join(args.output_dir, f"summary_prompts_{model}.csv")
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Saved summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
