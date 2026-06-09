"""Run inference on a CSV with ID and Report columns."""

from __future__ import annotations

import argparse
import os
import pandas as pd

import LLM_inference as rgpt


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    if path.endswith(".feather"):
        return pd.read_feather(path)
    raise ValueError("Data file must be .csv, .xlsx, or .feather")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference on radiology reports.")
    parser.add_argument("--port", required=True, help="vLLM port")
    parser.add_argument("--data_path", required=True, help="Path to the input CSV/XLSX/FEATHER")
    parser.add_argument("--save_path", required=True, help="Path to the output CSV file")
    parser.add_argument("--prompt_id", type=int, default=1, help="Prompt id (int)")
    parser.add_argument("--restart", action="store_true", help="Overwrite the output CSV")
    args = parser.parse_args()

    base_url = f"http://0.0.0.0:{args.port}/v1"
    data = load_data(args.data_path)
    data = data.dropna(subset=["ID", "Report"])

    save_root = os.path.dirname(args.save_path)
    save_dir = os.path.join(save_root, "raw")
    save_name = os.path.basename(args.save_path)
    prompt_dir = os.path.join(save_dir, f"prompt{args.prompt_id}")
    os.makedirs(prompt_dir, exist_ok=True)

    base, ext = os.path.splitext(save_name)
    output_path = os.path.join(
        prompt_dir,
        f"{base}_prompt{args.prompt_id}{ext or '.csv'}",
    )

    rgpt.run_inference(
        data,
        base_url=base_url,
        output_path=output_path,
        prompt_id=args.prompt_id,
        restart=args.restart,
    )
    print(f"Inference completed. Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
