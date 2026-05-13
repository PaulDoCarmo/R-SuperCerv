#!/usr/bin/env python3
"""Build a CSV with ID and checklist booleans from report JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def derive_id(file_path: Path) -> str:
    name = file_path.name
    if name.endswith(".json"):
        name = name[: -len(".json")]
    if name.endswith(".legacy"):
        name = name[: -len(".legacy")]
    if name.endswith("_report"):
        name = name[: -len("_report")]
    return name


def load_json_paths(json_folder: Path) -> Iterable[Path]:
    return sorted(json_folder.glob("*.json"))


def collect_common_checklist_keys(json_paths: Iterable[Path]) -> list[str]:
    common_keys: set[str] | None = None
    for json_path in json_paths:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        checklist = data.get("checklist")
        if not isinstance(checklist, dict):
            continue

        keys = set(checklist.keys())
        if common_keys is None:
            common_keys = keys
        else:
            common_keys &= keys

    if not common_keys:
        return []

    return sorted(common_keys)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a CSV with checklist booleans from report JSON files."
    )
    parser.add_argument("json_folder", help="Folder containing JSON report files")
    parser.add_argument("output_folder", help="Folder to write the CSV")
    parser.add_argument(
        "--output-name",
        default="checklist_booleans.csv",
        help="CSV filename to create in output_folder (default: checklist_booleans.csv)",
    )
    args = parser.parse_args()

    json_folder = Path(args.json_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    json_paths = list(load_json_paths(json_folder))
    checklist_keys = collect_common_checklist_keys(json_paths)

    output_path = output_folder / args.output_name

    fieldnames = ["ID", *checklist_keys]
    rows: list[dict[str, object]] = []

    for json_path in json_paths:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        dictation = data.get("dictation", "")
        if not isinstance(dictation, str) or dictation == "":
            continue

        checklist = data.get("checklist", {})
        if not isinstance(checklist, dict):
            checklist = {}

        row = {"ID": derive_id(json_path)}
        for key in checklist_keys:
            value = checklist.get(key)
            row[key] = value if isinstance(value, bool) else False
        rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
