#!/usr/bin/env python3
"""Build a CSV with ID and Report from a folder of report JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def derive_id(file_path: Path) -> str:
    name = file_path.name
    if name.endswith(".json"):
        name = name[: -len(".json")]
    if name.endswith(".legacy"):
        name = name[: -len(".legacy")]
    if name.endswith("_report"):
        name = name[: -len("_report")]
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a CSV from report JSON files.")
    parser.add_argument("json_folder", help="Folder containing JSON report files")
    parser.add_argument("output_folder", help="Folder to write the CSV")
    parser.add_argument(
        "--output-name",
        default="reports.csv",
        help="CSV filename to create in output_folder (default: reports.csv)",
    )
    args = parser.parse_args()

    json_folder = Path(args.json_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / args.output_name

    rows = []
    for json_path in sorted(json_folder.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        dictation = data.get("dictation", "")
        if not isinstance(dictation, str) or dictation == "":
            continue

        rows.append({"ID": derive_id(json_path), "Report": dictation})

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Report"])
        writer.writeheader()
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
