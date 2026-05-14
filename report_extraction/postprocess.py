import argparse
import os
import re

import pandas as pd


def detect_prompt_id(path: str) -> str:
    match = re.search(r"/prompt(\d+)/", path)
    if match:
        return match.group(1)
    match = re.search(r"_prompt(\d+)", os.path.basename(path))
    if match:
        return match.group(1)
    return "U"


def build_output_path(input_path: str, output_root: str) -> str:
    prompt_id = detect_prompt_id(input_path)
    prompt_dir = os.path.join(output_root, f"prompt{prompt_id}")
    os.makedirs(prompt_dir, exist_ok=True)

    base = os.path.basename(input_path)
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".csv"
    out_name = f"{stem}_postprocessed{ext}"
    return os.path.join(prompt_dir, out_name)


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in ["DNN Answer", "Report"] if c in df.columns]
    return df.drop(columns=cols_to_drop)


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process RadGPT results.")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument(
        "--output_root",
        default="/home/pauldcrm/links/scratch/R-SuperCerv/report_extraction/post_processed",
        help="Root output directory",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = drop_useless_columns(df)

    output_path = build_output_path(args.input, args.output_root)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


