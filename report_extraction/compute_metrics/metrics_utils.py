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
    formatted_results_path: str,
    output_root: str,
    metric_name: str,
    mode: str | None = None,
    suffix: str = "",
) -> tuple[str, str]:
    prompt_id = detect_prompt_id(formatted_results_path)
    model_name = detect_model_name(formatted_results_path)
    parts = [output_root, f"prompt{prompt_id}", model_name, metric_name]
    if mode:
        parts.append(mode)
    output_dir = os.path.join(*parts)
    os.makedirs(output_dir, exist_ok=True)

    mode_part = f"_{mode}" if mode else ""
    csv_name = f"metrics_{model_name}_prompt{prompt_id}_{metric_name}{mode_part}{suffix}.csv"
    txt_name = f"errors_{model_name}_prompt{prompt_id}_{metric_name}{mode_part}{suffix}.txt"
    return os.path.join(output_dir, csv_name), os.path.join(output_dir, txt_name)


def parse_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in BOOL_TRUE
